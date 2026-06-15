import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn


class BatchInferError(ValueError):
    """Raised when batch inference fails."""


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
    r2 = 1.0 - np.sum((pred - target) ** 2) / np.sum((target - np.mean(target)) ** 2)
    return {"mae": mae, "rmse": rmse, "max_error": max_err, "r2": float(r2)}


def generate_markdown_report(summary: Dict[str, Any], output_path: Path) -> None:
    lines = [
        "# FNO Batch Inference Report",
        "",
        "## Summary",
        f"- **Total Cases**: {summary['num_cases']}",
        f"- **Mean MAE**: {summary.get('mean_mae', 'N/A'):.6e}",
        f"- **Mean RMSE**: {summary.get('mean_rmse', 'N/A'):.6e}",
        f"- **Mean Max Error**: {summary.get('mean_max_error', 'N/A'):.6e}",
        f"- **Mean R²**: {summary.get('mean_r2', 'N/A'):.4f}",
        "",
        "## Per-Case Results",
        "",
        "| Case | MAE | RMSE | Max Error | R² |",
        "|------|-----|------|-----------|-----|",
    ]

    for item in summary.get("per_case_metrics", []):
        mae = item.get("mae", 0.0)
        rmse = item.get("rmse", 0.0)
        max_err = item.get("max_error", 0.0)
        r2 = item.get("r2", 0.0)
        lines.append(f"| {item['case_index']} | {mae:.3e} | {rmse:.3e} | {max_err:.3e} | {r2:.4f} |")

    lines.append("")
    lines.append("---")
    lines.append(f"*Report generated at {summary.get('timestamp', 'unknown')}*")

    with output_path.open("w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch inference and evaluation report generation.")
    parser.add_argument("--checkpoint", type=Path, default=Path("artifacts/training/fno_best.pt"), help="Checkpoint path.")
    parser.add_argument("--dataset", type=Path, default=Path("artifacts/fno_dataset.npz"), help="Dataset NPZ path.")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/batch_inference"), help="Output directory.")
    parser.add_argument("--device", type=str, default="auto", help="Device: auto, cpu, cuda.")
    parser.add_argument("--split-type", type=str, default="all", help="Split to infer: all, train, val, test.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    device = choose_device(args.device)

    checkpoint = torch.load(args.checkpoint, map_location=device)
    model_config = checkpoint.get("model_config")
    normalization = checkpoint.get("normalization")
    split_info = checkpoint.get("split_info")

    if not isinstance(model_config, dict) or not isinstance(normalization, dict):
        raise BatchInferError("Checkpoint missing model_config or normalization.")

    data = np.load(args.dataset)
    inputs = data["inputs"].astype(np.float32)
    targets = data["targets"].astype(np.float32) if "targets" in data else None

    if targets is None:
        raise BatchInferError("Dataset must contain targets.")

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

    indices = list(range(inputs.shape[0]))
    if args.split_type != "all" and split_info:
        indices = split_info.get(args.split_type + "_indices", indices)

    x_test = torch.from_numpy(inputs[indices])
    predictions = predict_batch(model, x_test, x_mean, x_std, y_mean, y_std, device).numpy()

    per_case_metrics: List[Dict[str, Any]] = []
    for i, idx in enumerate(indices):
        pred = predictions[i]
        target = targets[idx]
        metrics = compute_metrics(pred, target)
        per_case_metrics.append({"case_index": idx, **metrics})

    summary = {
        "num_cases": len(indices),
        "split_type": args.split_type,
        "checkpoint_path": str(args.checkpoint),
        "mean_mae": float(np.mean([m["mae"] for m in per_case_metrics])),
        "mean_rmse": float(np.mean([m["rmse"] for m in per_case_metrics])),
        "mean_max_error": float(np.mean([m["max_error"] for m in per_case_metrics])),
        "mean_r2": float(np.mean([m["r2"] for m in per_case_metrics])),
        "per_case_metrics": per_case_metrics,
        "timestamp": str(Path.cwd()),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)

    summary_file = args.output_dir / "batch_inference_summary.json"
    with summary_file.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    report_file = args.output_dir / "report.md"
    generate_markdown_report(summary, report_file)

    print(f"Batch inference complete: {len(indices)} cases")
    print(f"  Mean MAE:   {summary['mean_mae']:.6e}")
    print(f"  Mean RMSE:  {summary['mean_rmse']:.6e}")
    print(f"  Mean R²:    {summary['mean_r2']:.4f}")
    print(f"Summary: {summary_file}")
    print(f"Report:  {report_file}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
