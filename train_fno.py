import argparse
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


class TrainingDataError(ValueError):
    """Raised when the dataset does not satisfy required constraints."""


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def _validate_split_ratios(train_ratio: float, val_ratio: float, test_ratio: float) -> None:
    ratios = [train_ratio, val_ratio, test_ratio]
    if any(r < 0.0 for r in ratios):
        raise TrainingDataError("train/val/test ratios must be >= 0.")
    total = train_ratio + val_ratio + test_ratio
    if abs(total - 1.0) > 1e-6:
        raise TrainingDataError("train_ratio + val_ratio + test_ratio must equal 1.0.")
    if train_ratio <= 0.0:
        raise TrainingDataError("train_ratio must be > 0.")


def split_indices(
    num_samples: int,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if num_samples < 1:
        raise TrainingDataError("Dataset must contain at least one sample.")
    _validate_split_ratios(train_ratio, val_ratio, test_ratio)

    indices = np.arange(num_samples)
    rng = np.random.default_rng(seed)
    rng.shuffle(indices)

    val_count = int(round(num_samples * val_ratio))
    test_count = int(round(num_samples * test_ratio))

    if val_ratio > 0.0 and num_samples >= 2:
        val_count = max(1, val_count)
    if test_ratio > 0.0 and num_samples >= 3:
        test_count = max(1, test_count)

    if val_count + test_count >= num_samples:
        overflow = val_count + test_count - (num_samples - 1)
        reduce_test = min(overflow, test_count)
        test_count -= reduce_test
        overflow -= reduce_test
        if overflow > 0:
            val_count = max(0, val_count - overflow)

    train_count = num_samples - val_count - test_count
    if train_count <= 0:
        raise TrainingDataError("Split produced zero train samples. Adjust ratios.")

    val_start = train_count
    test_start = train_count + val_count

    train_indices = indices[:train_count]
    val_indices = indices[val_start:test_start]
    test_indices = indices[test_start:]
    return train_indices, val_indices, test_indices


def _to_int_list(values: np.ndarray) -> List[int]:
    return [int(v) for v in values.tolist()]


def _load_or_create_split_indices(
    split_path: Path,
    num_samples: int,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if split_path.exists():
        with split_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)

        try:
            train_indices = np.array(payload["train_indices"], dtype=np.int64)
            val_indices = np.array(payload["val_indices"], dtype=np.int64)
            test_indices = np.array(payload["test_indices"], dtype=np.int64)
            saved_num_samples = int(payload["num_samples"])
        except (KeyError, TypeError, ValueError) as exc:
            raise TrainingDataError(f"Invalid split file format: {split_path}") from exc

        if saved_num_samples != num_samples:
            raise TrainingDataError(
                f"Split file sample count ({saved_num_samples}) does not match dataset ({num_samples})."
            )

        all_indices = np.concatenate([train_indices, val_indices, test_indices])
        if len(all_indices) != len(np.unique(all_indices)):
            raise TrainingDataError("Split file contains duplicated indices across train/val/test.")
        if np.any(all_indices < 0) or np.any(all_indices >= num_samples):
            raise TrainingDataError("Split file contains out-of-range indices.")
        if len(train_indices) == 0:
            raise TrainingDataError("Split file must contain at least one train sample.")
        return train_indices, val_indices, test_indices

    train_indices, val_indices, test_indices = split_indices(
        num_samples=num_samples,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
    )

    split_path.parent.mkdir(parents=True, exist_ok=True)
    with split_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "num_samples": int(num_samples),
                "seed": int(seed),
                "train_ratio": train_ratio,
                "val_ratio": val_ratio,
                "test_ratio": test_ratio,
                "train_indices": _to_int_list(train_indices),
                "val_indices": _to_int_list(val_indices),
                "test_indices": _to_int_list(test_indices),
            },
            f,
            indent=2,
        )

    return train_indices, val_indices, test_indices


class TensorDataset(Dataset):
    def __init__(self, x: torch.Tensor, y: torch.Tensor):
        self.x = x
        self.y = y

    def __len__(self) -> int:
        return int(self.x.shape[0])

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.x[index], self.y[index]


class SpectralConv2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, modes_h: int, modes_w: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
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
        # x: (batch, in_channel, h, w_ft), weight: (in_channel, out_channel, h_modes, w_modes)
        return torch.einsum("bihw,iohw->bohw", x, weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, _, height, width = x.shape
        x_ft = torch.fft.rfft2(x)
        out_ft = torch.zeros(
            batch_size,
            self.out_channels,
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
        x = self.spectral(x) + self.pointwise(x)
        return self.activation(x)


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


@dataclass
class TrainingConfig:
    dataset_path: str
    output_dir: str
    split_file: str
    train_ratio: float
    val_ratio: float
    test_ratio: float
    batch_size: int
    epochs: int
    early_stopping_patience: int
    early_stopping_min_delta: float
    learning_rate: float
    scheduler_factor: float
    scheduler_patience: int
    min_learning_rate: float
    weight_decay: float
    width: int
    depth: int
    modes_h: int
    modes_w: int
    seed: int
    device: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a simple FNO model on fno_dataset.npz")
    parser.add_argument("--dataset", type=Path, default=Path("artifacts/fno_dataset.npz"), help="Input dataset NPZ path.")
    parser.add_argument(
        "--output-dir", type=Path, default=Path("artifacts/training"), help="Directory to save checkpoints and logs."
    )
    parser.add_argument(
        "--split-file",
        type=Path,
        default=Path("artifacts/splits/train_val_test_split.json"),
        help="Path to a fixed split JSON. If missing, it is created once and reused.",
    )
    parser.add_argument("--train-ratio", type=float, default=0.7, help="Train split ratio.")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="Validation split ratio.")
    parser.add_argument("--test-ratio", type=float, default=0.1, help="Test split ratio.")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size.")
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs.")
    parser.add_argument("--early-stopping-patience", type=int, default=20, help="Epochs to wait without val improvement.")
    parser.add_argument(
        "--early-stopping-min-delta",
        type=float,
        default=1e-6,
        help="Minimum validation loss improvement to reset early stopping counter.",
    )
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument(
        "--scheduler-factor",
        type=float,
        default=0.5,
        help="ReduceLROnPlateau factor applied when val loss stagnates.",
    )
    parser.add_argument(
        "--scheduler-patience",
        type=int,
        default=8,
        help="ReduceLROnPlateau patience in epochs.",
    )
    parser.add_argument("--min-learning-rate", type=float, default=1e-6, help="Lower bound for learning rate scheduler.")
    parser.add_argument("--weight-decay", type=float, default=1e-6, help="Weight decay.")
    parser.add_argument("--width", type=int, default=32, help="Hidden channel width.")
    parser.add_argument("--depth", type=int, default=4, help="Number of FNO blocks.")
    parser.add_argument("--modes-h", type=int, default=8, help="Fourier modes along height.")
    parser.add_argument("--modes-w", type=int, default=8, help="Fourier modes along width.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--device", type=str, default="auto", help="Device: auto, cpu, cuda.")
    return parser.parse_args()


def load_dataset(path: Path) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
    if not path.exists():
        raise TrainingDataError(f"Dataset not found: {path}")

    data = np.load(path)
    if "inputs" not in data or "targets" not in data:
        raise TrainingDataError("Dataset NPZ must contain inputs and targets arrays.")

    x = data["inputs"].astype(np.float32)
    y = data["targets"].astype(np.float32)
    if x.ndim != 4:
        raise TrainingDataError(f"inputs must be rank-4 [N,C,H,W], got shape={x.shape}")
    if y.ndim != 4:
        raise TrainingDataError(f"targets must be rank-4 [N,C,H,W], got shape={y.shape}")
    if x.shape[0] != y.shape[0]:
        raise TrainingDataError("inputs and targets must have same sample count.")

    if "case_ids" in data:
        case_ids = [str(v) for v in data["case_ids"].tolist()]
    else:
        case_ids = [f"sample_{i:05d}" for i in range(x.shape[0])]

    return torch.from_numpy(x), torch.from_numpy(y), case_ids


def compute_normalization(x_train: torch.Tensor, y_train: torch.Tensor) -> Dict[str, torch.Tensor]:
    x_mean = x_train.mean(dim=(0, 2, 3), keepdim=True)
    x_std = x_train.std(dim=(0, 2, 3), keepdim=True).clamp_min(1e-6)

    y_mean = y_train.mean(dim=(0, 2, 3), keepdim=True)
    y_std = y_train.std(dim=(0, 2, 3), keepdim=True).clamp_min(1e-6)

    return {
        "x_mean": x_mean,
        "x_std": x_std,
        "y_mean": y_mean,
        "y_std": y_std,
    }


def normalize(x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    return (x - mean) / std


def denormalize(x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    return x * std + mean


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    y_mean: torch.Tensor,
    y_std: torch.Tensor,
) -> Tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_mae = 0.0
    total_count = 0

    with torch.no_grad():
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            pred = model(x_batch)
            loss = criterion(pred, y_batch)

            pred_phys = denormalize(pred, y_mean, y_std)
            y_phys = denormalize(y_batch, y_mean, y_std)
            mae = torch.mean(torch.abs(pred_phys - y_phys))

            batch_size = x_batch.shape[0]
            total_loss += float(loss.item()) * batch_size
            total_mae += float(mae.item()) * batch_size
            total_count += int(batch_size)

    return total_loss / total_count, total_mae / total_count


def main() -> int:
    args = parse_args()
    set_seed(args.seed)
    device = choose_device(args.device)

    x_all, y_all, case_ids = load_dataset(args.dataset)
    train_idx, val_idx, test_idx = _load_or_create_split_indices(
        split_path=args.split_file,
        num_samples=x_all.shape[0],
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )

    x_train_raw, y_train_raw = x_all[train_idx], y_all[train_idx]
    x_val_raw, y_val_raw = x_all[val_idx], y_all[val_idx]
    x_test_raw, y_test_raw = x_all[test_idx], y_all[test_idx]

    norm = compute_normalization(x_train_raw, y_train_raw)
    x_train = normalize(x_train_raw, norm["x_mean"], norm["x_std"])
    y_train = normalize(y_train_raw, norm["y_mean"], norm["y_std"])
    x_val = normalize(x_val_raw, norm["x_mean"], norm["x_std"]) if len(val_idx) > 0 else x_val_raw
    y_val = normalize(y_val_raw, norm["y_mean"], norm["y_std"]) if len(val_idx) > 0 else y_val_raw
    x_test = normalize(x_test_raw, norm["x_mean"], norm["x_std"]) if len(test_idx) > 0 else x_test_raw
    y_test = normalize(y_test_raw, norm["y_mean"], norm["y_std"]) if len(test_idx) > 0 else y_test_raw

    train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(x_val, y_val), batch_size=args.batch_size, shuffle=False) if len(val_idx) > 0 else None
    test_loader = DataLoader(TensorDataset(x_test, y_test), batch_size=args.batch_size, shuffle=False) if len(test_idx) > 0 else None

    model = FNO2d(
        in_channels=int(x_all.shape[1]),
        out_channels=int(y_all.shape[1]),
        width=args.width,
        depth=args.depth,
        modes_h=args.modes_h,
        modes_w=args.modes_w,
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=args.scheduler_factor,
        patience=args.scheduler_patience,
        min_lr=args.min_learning_rate,
    )

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    best_val_loss = float("inf")
    best_epoch = 0
    no_improve_epochs = 0
    history: List[Dict[str, float]] = []

    y_mean_device = norm["y_mean"].to(device)
    y_std_device = norm["y_std"].to(device)

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        seen = 0

        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad(set_to_none=True)
            pred = model(x_batch)
            loss = criterion(pred, y_batch)
            loss.backward()
            optimizer.step()

            batch_size = x_batch.shape[0]
            running_loss += float(loss.item()) * batch_size
            seen += int(batch_size)

        train_loss = running_loss / max(1, seen)
        if val_loader is not None:
            val_loss, val_mae = evaluate(model, val_loader, criterion, device, y_mean_device, y_std_device)
        else:
            val_loss, val_mae = train_loss, float("nan")

        scheduler.step(val_loss)
        current_lr = float(optimizer.param_groups[0]["lr"])

        history.append(
            {
                "epoch": epoch,
                "train_mse": train_loss,
                "val_mse": val_loss,
                "val_mae": val_mae,
                "learning_rate": current_lr,
            }
        )
        print(
            f"Epoch {epoch:04d} | train_mse={train_loss:.6e} | val_mse={val_loss:.6e} "
            f"| val_mae={val_mae:.6e} | lr={current_lr:.3e}"
        )

        improvement = best_val_loss - val_loss
        if improvement > args.early_stopping_min_delta:
            best_val_loss = val_loss
            best_epoch = epoch
            no_improve_epochs = 0
            checkpoint = {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "normalization": {
                    "x_mean": norm["x_mean"],
                    "x_std": norm["x_std"],
                    "y_mean": norm["y_mean"],
                    "y_std": norm["y_std"],
                },
                "model_config": {
                    "in_channels": int(x_all.shape[1]),
                    "out_channels": int(y_all.shape[1]),
                    "width": args.width,
                    "depth": args.depth,
                    "modes_h": args.modes_h,
                    "modes_w": args.modes_w,
                },
                "split": {
                    "train_indices": _to_int_list(train_idx),
                    "val_indices": _to_int_list(val_idx),
                    "test_indices": _to_int_list(test_idx),
                    "split_file": str(args.split_file),
                },
                "case_ids": case_ids,
                "best_val_mse": best_val_loss,
                "best_epoch": best_epoch,
            }
            torch.save(checkpoint, output_dir / "fno_best.pt")
        else:
            no_improve_epochs += 1

        if args.early_stopping_patience > 0 and no_improve_epochs >= args.early_stopping_patience:
            print(f"Early stopping at epoch {epoch} (no improvement for {no_improve_epochs} epochs).")
            break

    checkpoint_path = output_dir / "fno_best.pt"
    if not checkpoint_path.exists():
        raise TrainingDataError("No checkpoint was saved during training.")

    best_checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(best_checkpoint["model_state_dict"])

    if test_loader is not None:
        test_loss, test_mae = evaluate(model, test_loader, criterion, device, y_mean_device, y_std_device)
    else:
        test_loss, test_mae = float("nan"), float("nan")

    config = TrainingConfig(
        dataset_path=str(args.dataset),
        output_dir=str(output_dir),
        split_file=str(args.split_file),
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        batch_size=args.batch_size,
        epochs=args.epochs,
        early_stopping_patience=args.early_stopping_patience,
        early_stopping_min_delta=args.early_stopping_min_delta,
        learning_rate=args.learning_rate,
        scheduler_factor=args.scheduler_factor,
        scheduler_patience=args.scheduler_patience,
        min_learning_rate=args.min_learning_rate,
        weight_decay=args.weight_decay,
        width=args.width,
        depth=args.depth,
        modes_h=args.modes_h,
        modes_w=args.modes_w,
        seed=args.seed,
        device=str(device),
    )

    with (output_dir / "training_history.json").open("w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    summary = {
        "config": asdict(config),
        "num_samples": int(x_all.shape[0]),
        "num_train_samples": int(len(train_idx)),
        "num_val_samples": int(len(val_idx)),
        "num_test_samples": int(len(test_idx)),
        "best_val_mse": best_val_loss,
        "best_epoch": int(best_epoch),
        "test_mse": test_loss,
        "test_mae": test_mae,
        "split": {
            "train_indices": _to_int_list(train_idx),
            "val_indices": _to_int_list(val_idx),
            "test_indices": _to_int_list(test_idx),
            "train_case_ids": [case_ids[i] for i in _to_int_list(train_idx)],
            "val_case_ids": [case_ids[i] for i in _to_int_list(val_idx)],
            "test_case_ids": [case_ids[i] for i in _to_int_list(test_idx)],
        },
        "checkpoint_path": str(checkpoint_path),
    }
    with (output_dir / "training_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("Training finished")
    print(f"Best checkpoint: {checkpoint_path}")
    print(f"History: {output_dir / 'training_history.json'}")
    if test_loader is not None:
        print(f"Test: mse={test_loss:.6e}, mae={test_mae:.6e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())