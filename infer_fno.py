import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn


class InferenceError(ValueError):
    """Raised when inference inputs are invalid."""


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run single-case field inference with a trained FNO checkpoint.")
    parser.add_argument("--checkpoint", type=Path, default=Path("artifacts/training/fno_best.pt"), help="Checkpoint path.")
    parser.add_argument("--dataset", type=Path, default=Path("artifacts/fno_dataset.npz"), help="Dataset NPZ path.")
    parser.add_argument("--case-index", type=int, default=None, help="Case index in dataset.")
    parser.add_argument("--case-id", type=str, default=None, help="Case ID in dataset.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/prediction_case.json"),
        help="Output prediction JSON path.",
    )
    parser.add_argument("--device", type=str, default="auto", help="Device: auto, cpu, cuda.")
    parser.add_argument("--quantity", type=str, default="top_surface_warpage", help="Output quantity name.")
    parser.add_argument("--unit", type=str, default="mm", help="Output unit.")
    return parser.parse_args()


def _select_case_index(case_ids: List[str], case_index: Optional[int], case_id: Optional[str]) -> int:
    if case_index is not None and case_id is not None:
        raise InferenceError("Specify either --case-index or --case-id, not both.")
    if case_index is None and case_id is None:
        return 0

    if case_index is not None:
        if case_index < 0 or case_index >= len(case_ids):
            raise InferenceError(f"case-index out of range: {case_index}")
        return case_index

    assert case_id is not None
    try:
        return case_ids.index(case_id)
    except ValueError as exc:
        raise InferenceError(f"case-id not found in dataset: {case_id}") from exc


def _to_hw_list(chw: np.ndarray) -> List[List[float]]:
    if chw.ndim != 3 or chw.shape[0] != 1:
        raise InferenceError(f"Expected shape [1,H,W], got {chw.shape}")
    return [[float(v) for v in row] for row in chw[0].tolist()]


def main() -> int:
    args = parse_args()
    device = choose_device(args.device)

    checkpoint = torch.load(args.checkpoint, map_location=device)
    model_config = checkpoint.get("model_config")
    normalization = checkpoint.get("normalization")
    if not isinstance(model_config, dict) or not isinstance(normalization, dict):
        raise InferenceError("Checkpoint missing model_config or normalization.")

    data = np.load(args.dataset)
    if "inputs" not in data:
        raise InferenceError("Dataset must contain inputs.")
    inputs = data["inputs"].astype(np.float32)
    targets = data["targets"].astype(np.float32) if "targets" in data else None
    case_ids = [str(v) for v in data["case_ids"].tolist()] if "case_ids" in data else [f"sample_{i:05d}" for i in range(inputs.shape[0])]

    idx = _select_case_index(case_ids, args.case_index, args.case_id)
    case_id = case_ids[idx]

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

    x_mean = normalization["x_mean"].to(device)
    x_std = normalization["x_std"].to(device)
    y_mean = normalization["y_mean"].to(device)
    y_std = normalization["y_std"].to(device)

    x_case = torch.from_numpy(inputs[idx : idx + 1]).to(device)
    x_case_norm = (x_case - x_mean) / x_std

    with torch.no_grad():
        pred_norm = model(x_case_norm)
        pred = pred_norm * y_std + y_mean

    pred_np = pred.cpu().numpy()[0]
    result: Dict[str, Any] = {
        "case_id": case_id,
        "quantity": args.quantity,
        "unit": args.unit,
        "grid_shape": {"height": int(pred_np.shape[1]), "width": int(pred_np.shape[2])},
        "predicted_field_hw": _to_hw_list(pred_np),
        "source": {
            "checkpoint": str(args.checkpoint),
            "dataset": str(args.dataset),
            "case_index": idx,
        },
    }

    if targets is not None:
        target_np = targets[idx]
        if target_np.shape == pred_np.shape:
            mae = float(np.mean(np.abs(pred_np - target_np)))
            rmse = float(np.sqrt(np.mean((pred_np - target_np) ** 2)))
            result["target_field_hw"] = _to_hw_list(target_np)
            result["metrics"] = {"mae": mae, "rmse": rmse}

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print(f"Predicted field for case_id={case_id}")
    print(f"Output: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())