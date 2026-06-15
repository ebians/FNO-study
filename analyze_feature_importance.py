import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn as nn


class FeatureImportanceError(ValueError):
    """Raised when feature importance analysis fails."""


def choose_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


class SpectralConv2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, modes_h: int, modes_w: int):
        super().__init__()
        self.modes_h = modes_h
        self.modes_w = modes_w
        scale = 1.0 / math.sqrt(in_channels * out_channels)
        self.weight_pos = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, modes_h, modes_w, dtype=torch.cfloat)
        )
        self.weight_neg = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, modes_h, modes_w, dtype=torch.cfloat)
        )

    def compl_mul2d(self, x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        return torch.einsum("bihw,iohw->bohw", x, weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, _, height, width = x.shape
        x_ft = torch.fft.rfft2(x)
        out_ft = torch.zeros(
            batch_size,
            self.weight_pos.shape[1],
            height,
            width // 2 + 1,
            device=x.device,
            dtype=torch.cfloat,
        )

        mh = min(self.modes_h, height)
        mw = min(self.modes_w, width // 2 + 1)
        out_ft[:, :, :mh, :mw] = self.compl_mul2d(x_ft[:, :, :mh, :mw], self.weight_pos[:, :, :mh, :mw])
        out_ft[:, :, -mh:, :mw] = self.compl_mul2d(x_ft[:, :, -mh:, :mw], self.weight_neg[:, :, :mh, :mw])
        return torch.fft.irfft2(out_ft, s=(height, width))


class FNOBlock(nn.Module):
    def __init__(self, width: int, modes_h: int, modes_w: int):
        super().__init__()
        self.spectral = SpectralConv2d(width, width, modes_h, modes_w)
        self.pointwise = nn.Conv2d(width, width, kernel_size=1)
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(self.spectral(x) + self.pointwise(x))


class FNO2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, width: int, depth: int, modes_h: int, modes_w: int):
        super().__init__()
        self.input_proj = nn.Conv2d(in_channels, width, kernel_size=1)
        self.blocks = nn.ModuleList([FNOBlock(width, modes_h, modes_w) for _ in range(depth)])
        self.output_proj = nn.Sequential(
            nn.Conv2d(width, width, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(width, out_channels, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)
        for block in self.blocks:
            x = block(x)
        return self.output_proj(x)


def compute_baseline_loss(model: nn.Module, x_batch: torch.Tensor, y_batch: torch.Tensor, device: torch.device) -> float:
    """Compute baseline prediction loss (all inputs)."""
    x_batch = x_batch.to(device)
    y_batch = y_batch.to(device)
    with torch.no_grad():
        pred = model(x_batch)
    loss = float(torch.mean((pred - y_batch) ** 2).item())
    return loss


def compute_occlusion_importance(
    model: nn.Module,
    x_batch: torch.Tensor,
    y_batch: torch.Tensor,
    baseline_loss: float,
    num_channels: int,
    device: torch.device,
) -> Dict[int, float]:
    """Compute feature importance by occlusion (zeroing each channel)."""
    importance = {}

    for ch in range(num_channels):
        x_occluded = x_batch.clone()
        x_occluded[:, ch, :, :] = 0.0

        x_occluded = x_occluded.to(device)
        y_batch_dev = y_batch.to(device)

        with torch.no_grad():
            pred = model(x_occluded)
        loss_with_occlusion = float(torch.mean((pred - y_batch_dev) ** 2).item())

        importance[ch] = loss_with_occlusion - baseline_loss

    return importance


def compute_gradient_importance(
    model: nn.Module, x_batch: torch.Tensor, y_batch: torch.Tensor, device: torch.device
) -> Dict[int, float]:
    """Compute feature importance by gradient magnitude."""
    x_batch = x_batch.to(device).requires_grad_(True)
    y_batch = y_batch.to(device)

    pred = model(x_batch)
    loss = torch.mean((pred - y_batch) ** 2)
    loss.backward()

    gradients = x_batch.grad
    importance = {}

    for ch in range(x_batch.shape[1]):
        ch_grad = gradients[:, ch, :, :]
        importance[ch] = float(torch.mean(torch.abs(ch_grad)).item())

    return importance


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze feature importance for FNO model.")
    parser.add_argument("--checkpoint", type=Path, default=Path("artifacts/training/fno_best.pt"), help="Checkpoint path")
    parser.add_argument("--dataset", type=Path, default=Path("artifacts/fno_dataset.npz"), help="Dataset NPZ path")
    parser.add_argument("--method", choices=["occlusion", "gradient", "both"], default="both", help="Importance method")
    parser.add_argument("--num-samples", type=int, default=32, help="Number of samples for analysis")
    parser.add_argument("--output", type=Path, help="Save report to JSON")
    parser.add_argument("--device", type=str, default="auto", help="Device: auto, cpu, cuda")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    device = choose_device(args.device)

    checkpoint = torch.load(args.checkpoint, map_location=device)
    model_config = checkpoint.get("model_config")
    normalization = checkpoint.get("normalization")

    if not isinstance(model_config, dict) or not isinstance(normalization, dict):
        raise FeatureImportanceError("Checkpoint missing model_config or normalization")

    data = np.load(args.dataset)
    inputs = data["inputs"].astype(np.float32)
    targets = data["targets"].astype(np.float32) if "targets" in data else None

    if targets is None:
        raise FeatureImportanceError("Dataset must contain targets")

    num_samples = min(args.num_samples, inputs.shape[0])
    x_sample = torch.from_numpy(inputs[:num_samples])
    y_sample = torch.from_numpy(targets[:num_samples])

    x_mean = normalization["x_mean"]
    x_std = normalization["x_std"]
    y_mean = normalization["y_mean"]
    y_std = normalization["y_std"]

    x_norm = (x_sample - x_mean) / x_std
    y_norm = (y_sample - y_mean) / y_std

    model = FNO2d(
        in_channels=int(model_config["in_channels"]),
        out_channels=int(model_config["out_channels"]),
        width=int(model_config["width"]),
        depth=int(model_config["depth"]),
        modes_h=int(model_config["modes_h"]),
        modes_w=int(model_config["modes_w"]),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    num_channels = x_norm.shape[1]
    print(f"Analyzing {num_channels} channels with {num_samples} samples...")
    print("=" * 70)

    baseline_loss = compute_baseline_loss(model, x_norm, y_norm, device)
    print(f"Baseline Loss (all channels): {baseline_loss:.6e}")

    report = {"num_channels": num_channels, "num_samples": num_samples, "baseline_loss": baseline_loss}

    if args.method in ["occlusion", "both"]:
        print("\nComputing occlusion importance...")
        occlusion_imp = compute_occlusion_importance(model, x_norm, y_norm, baseline_loss, num_channels, device)
        report["occlusion_importance"] = {str(k): v for k, v in occlusion_imp.items()}

        print("Top 10 Important Channels (by occlusion):")
        sorted_occ = sorted(occlusion_imp.items(), key=lambda x: x[1], reverse=True)
        for ch, score in sorted_occ[:10]:
            print(f"  Channel {ch:2d}: {score:.6e}")

    if args.method in ["gradient", "both"]:
        print("\nComputing gradient importance...")
        gradient_imp = compute_gradient_importance(model, x_norm, y_norm, device)
        report["gradient_importance"] = {str(k): v for k, v in gradient_imp.items()}

        print("Top 10 Important Channels (by gradient):")
        sorted_grad = sorted(gradient_imp.items(), key=lambda x: x[1], reverse=True)
        for ch, score in sorted_grad[:10]:
            print(f"  Channel {ch:2d}: {score:.6e}")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"\nReport saved: {args.output}")

    print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
