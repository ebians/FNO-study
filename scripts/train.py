"""Training script for Fourier Neural Operator.

Usage examples::

    # Train on the 1-D Burgers' equation (default)
    python scripts/train.py

    # Train on 2-D Navier-Stokes with a custom config
    python scripts/train.py --config configs/navier_stokes.yaml

    # Quick smoke-test with a tiny synthetic dataset
    python scripts/train.py --problem burgers --n_samples 64 --n_epochs 2

Run ``python scripts/train.py --help`` for the full list of options.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running the script from the repo root without installing the package
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml
import torch

from fno import FNO1d, FNO2d
from fno.data import BurgersDataset, NavierStokesDataset, make_dataloaders
from fno.train import Trainer


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a Fourier Neural Operator (FNO).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to a YAML config file. CLI flags override config values.",
    )
    parser.add_argument(
        "--problem",
        choices=["burgers", "navier_stokes"],
        default="burgers",
        help="PDE problem to train on.",
    )

    # ---- Data ----
    data = parser.add_argument_group("data")
    data.add_argument("--data_path", type=Path, default=None, help="Path to dataset (.npz).")
    data.add_argument("--n_samples", type=int, default=1000)
    data.add_argument("--n_x", type=int, default=128, help="1-D spatial resolution.")
    data.add_argument("--h", type=int, default=64, help="2-D height resolution.")
    data.add_argument("--w", type=int, default=64, help="2-D width resolution.")
    data.add_argument("--T_in", type=int, default=10, help="Input time steps (Navier-Stokes).")
    data.add_argument("--train_ratio", type=float, default=0.8)
    data.add_argument("--batch_size", type=int, default=32)
    data.add_argument("--num_workers", type=int, default=0)

    # ---- Model ----
    model = parser.add_argument_group("model")
    model.add_argument("--modes", type=int, default=16)
    model.add_argument("--modes2", type=int, default=None, help="Modes for axis 2 (2-D only, defaults to --modes).")
    model.add_argument("--width", type=int, default=64)
    model.add_argument("--n_layers", type=int, default=4)

    # ---- Training ----
    train = parser.add_argument_group("training")
    train.add_argument("--n_epochs", type=int, default=100)
    train.add_argument("--lr", type=float, default=1e-3)
    train.add_argument("--weight_decay", type=float, default=1e-4)
    train.add_argument("--seed", type=int, default=42)
    train.add_argument(
        "--checkpoint_dir",
        type=Path,
        default=Path("checkpoints"),
        help="Directory to save the best checkpoint.",
    )

    args = parser.parse_args(argv)

    # Merge YAML config (CLI takes precedence)
    if args.config is not None:
        with open(args.config) as f:
            cfg = yaml.safe_load(f)
        for key, value in cfg.items():
            if not hasattr(args, key):
                parser.error(f"Unknown config key: {key!r}")
            # Only set if not explicitly provided on CLI
            if getattr(args, key) == parser.get_default(key):
                setattr(args, key, value)

    if args.modes2 is None:
        args.modes2 = args.modes

    return args


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    torch.manual_seed(args.seed)

    # ---- Dataset ----
    if args.problem == "burgers":
        dataset = BurgersDataset(
            n_samples=args.n_samples,
            n_x=args.n_x,
            path=args.data_path,
        )
        model = FNO1d(
            modes=args.modes,
            width=args.width,
            in_channels=2,      # u0 + grid
            out_channels=1,
            n_layers=args.n_layers,
        )
    else:  # navier_stokes
        dataset = NavierStokesDataset(
            n_samples=args.n_samples,
            h=args.h,
            w=args.w,
            T_in=args.T_in,
            path=args.data_path,
        )
        model = FNO2d(
            modes1=args.modes,
            modes2=args.modes2,
            width=args.width,
            in_channels=args.T_in + 2,   # time steps + 2-D grid
            out_channels=1,
            n_layers=args.n_layers,
        )

    train_loader, val_loader = make_dataloaders(
        dataset,
        train_ratio=args.train_ratio,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
    )

    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Problem   : {args.problem}")
    print(f"Model     : {type(model).__name__}  |  params: {param_count:,}")

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        n_epochs=args.n_epochs,
        checkpoint_dir=args.checkpoint_dir,
    )
    trainer.train()


if __name__ == "__main__":
    main()
