import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


class ComparisonError(ValueError):
    """Raised when comparison inputs are invalid."""


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


def predict_batch(
    model: nn.Module,
    x_batch: torch.Tensor,
    x_mean: torch.Tensor,
    x_std: torch.Tensor,
    y_mean: torch.Tensor,
    y_std: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    x_norm = (x_batch - x_mean) / x_std
    x_norm = x_norm.to(device)
    with torch.no_grad():
        pred_norm = model(x_norm)
    pred = pred_norm * y_std + y_mean
    return pred.cpu()


def compute_metrics(pred: np.ndarray, target: np.ndarray) -> Dict[str, float]:
    mae = float(np.mean(np.abs(pred - target)))
    rmse = float(np.sqrt(np.mean((pred - target) ** 2)))
    max_err = float(np.max(np.abs(pred - target)))
    return {"mae": mae, "rmse": rmse, "max_error": max_err}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare predicted fields with target fields using trained model.")
    parser.add_argument("--checkpoint", type=Path, default=Path("artifacts/training/fno_best.pt"), help="Checkpoint path.")
    parser.add_argument("--dataset", type=Path, default=Path("artifacts/fno_dataset.npz"), help="Dataset NPZ path.")
    parser.add_argument("--num-cases", type=int, default=4, help="Number of cases to visualize.")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/comparison"), help="Output directory for plots.")
    parser.add_argument("--device", type=str, default="auto", help="Device: auto, cpu, cuda.")
    parser.add_argument("--no-plot", action="store_true", help="Disable matplotlib plots (just compute metrics).")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    device = choose_device(args.device)

    checkpoint = torch.load(args.checkpoint, map_location=device)
    model_config = checkpoint.get("model_config")
    normalization = checkpoint.get("normalization")
    if not isinstance(model_config, dict) or not isinstance(normalization, dict):
        raise ComparisonError("Checkpoint missing model_config or normalization.")

    data = np.load(args.dataset)
    inputs = data["inputs"].astype(np.float32)
    targets = data["targets"].astype(np.float32) if "targets" in data else None
    if targets is None:
        raise ComparisonError("Dataset must contain targets.")

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

    x_mean = normalization["x_mean"]
    x_std = normalization["x_std"]
    y_mean = normalization["y_mean"]
    y_std = normalization["y_std"]

    num_cases = min(args.num_cases, inputs.shape[0])
    all_metrics: List[Dict[str, Any]] = []

    x_test = torch.from_numpy(inputs[:num_cases])
    predictions = predict_batch(model, x_test, x_mean, x_std, y_mean, y_std, device).numpy()

    for idx in range(num_cases):
        pred = predictions[idx]
        target = targets[idx]
        metrics = compute_metrics(pred, target)
        all_metrics.append({"case_index": idx, **metrics})
        print(f"Case {idx}: MAE={metrics['mae']:.6e}, RMSE={metrics['rmse']:.6e}, MaxErr={metrics['max_error']:.6e}")

    if not args.no_plot and HAS_MATPLOTLIB:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        for idx in range(num_cases):
            pred = predictions[idx, 0]
            target = targets[idx, 0]
            error = pred - target

            fig, axes = plt.subplots(1, 3, figsize=(15, 4))
            vmin_max = max(np.abs(pred).max(), np.abs(target).max())
            vmin_err = np.abs(error).max()

            im0 = axes[0].imshow(target, cmap="viridis", vmin=-vmin_max, vmax=vmin_max)
            axes[0].set_title(f"Case {idx}: Target")
            plt.colorbar(im0, ax=axes[0])

            im1 = axes[1].imshow(pred, cmap="viridis", vmin=-vmin_max, vmax=vmin_max)
            axes[1].set_title(f"Case {idx}: Predicted")
            plt.colorbar(im1, ax=axes[1])

            im2 = axes[2].imshow(error, cmap="RdBu_r", vmin=-vmin_err, vmax=vmin_err)
            axes[2].set_title(f"Case {idx}: Error (MAE={all_metrics[idx]['mae']:.3e})")
            plt.colorbar(im2, ax=axes[2])

            fig.tight_layout()
            fig.savefig(args.output_dir / f"case_{idx:04d}_comparison.png", dpi=100, bbox_inches="tight")
            plt.close(fig)

        print(f"Plots saved to {args.output_dir}/")

    summary = {"num_cases": num_cases, "metrics_per_case": all_metrics}
    summary_file = args.output_dir / "comparison_summary.json"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with summary_file.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Summary: {summary_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
