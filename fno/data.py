"""Data utilities for FNO training.

Provides:
* Dataset classes for common benchmark PDEs.
* Helper functions to generate synthetic data for testing.
* ``normalize`` / ``UnitGaussianNormalizer`` for input / output scaling.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split


# ---------------------------------------------------------------------------
# Normalizers
# ---------------------------------------------------------------------------


class UnitGaussianNormalizer:
    """Normalize a tensor to zero mean and unit variance, channel-wise.

    Args:
        x: Reference tensor used to compute mean / std.
            Shape can be anything; statistics are computed over all dimensions
            except the last one (channel dimension).
        eps: Small constant for numerical stability.
    """

    def __init__(self, x: torch.Tensor, eps: float = 1e-5) -> None:
        self.mean = x.mean(dim=list(range(x.ndim - 1)))
        self.std = x.std(dim=list(range(x.ndim - 1)))
        self.eps = eps

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / (self.std + self.eps)

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        return x * (self.std + self.eps) + self.mean

    def to(self, device: torch.device | str) -> "UnitGaussianNormalizer":
        self.mean = self.mean.to(device)
        self.std = self.std.to(device)
        return self


# ---------------------------------------------------------------------------
# Synthetic dataset generators (for benchmarking / quick tests)
# ---------------------------------------------------------------------------


def _grid1d(n: int) -> torch.Tensor:
    """Return a 1-D uniform grid on [0, 1] of size ``n``."""
    return torch.linspace(0, 1, n).unsqueeze(-1)  # (n, 1)


def _grid2d(h: int, w: int) -> torch.Tensor:
    """Return a 2-D meshgrid on [0,1]^2 of shape ``(h, w, 2)``."""
    x = torch.linspace(0, 1, w)
    y = torch.linspace(0, 1, h)
    grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")
    return torch.stack([grid_x, grid_y], dim=-1)  # (h, w, 2)


class BurgersDataset(Dataset):
    """Synthetic dataset for the 1-D viscous Burgers' equation.

    Each sample is a pair ``(u0, uT)`` where ``u0`` is the initial condition
    and ``uT`` is the solution at time ``T``.  The dataset can optionally be
    loaded from a pre-generated ``.npz`` file; if the file does not exist a
    simple sinusoidal IC is used as a stand-in.

    Args:
        n_samples: Number of samples to generate / load.
        n_x: Spatial resolution.
        path: Optional path to a ``.npz`` file with keys ``"a"`` (input) and
            ``"u"`` (output), both of shape ``(N, n_x)``.
    """

    def __init__(
        self,
        n_samples: int = 1000,
        n_x: int = 128,
        path: Optional[Path] = None,
    ) -> None:
        super().__init__()
        if path is not None and Path(path).exists():
            data = np.load(path)
            a = torch.from_numpy(data["a"].astype(np.float32))
            u = torch.from_numpy(data["u"].astype(np.float32))
            self.a = a[:n_samples]
            self.u = u[:n_samples]
        else:
            # Synthetic sinusoidal initial conditions
            x = torch.linspace(0, 2 * math.pi, n_x)
            phases = torch.rand(n_samples) * 2 * math.pi
            amplitudes = 0.5 + torch.rand(n_samples) * 0.5
            self.a = amplitudes.unsqueeze(1) * torch.sin(
                x.unsqueeze(0) + phases.unsqueeze(1)
            )  # (n_samples, n_x)
            # "target" = damped version (stand-in for actual PDE solve)
            self.u = self.a * 0.5

        self.grid = _grid1d(self.a.shape[1])  # (n_x, 1)

    def __len__(self) -> int:
        return len(self.a)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        a = self.a[idx]  # (n_x,)
        u = self.u[idx]  # (n_x,)
        # concatenate initial condition and grid → (n_x, 2)
        x_in = torch.cat([a.unsqueeze(-1), self.grid], dim=-1)
        return x_in, u.unsqueeze(-1)  # inputs, targets


class NavierStokesDataset(Dataset):
    """Dataset for the 2-D Navier-Stokes equation (vorticity formulation).

    Each sample is a pair ``(w0, wT)`` where ``w0`` is the initial vorticity
    field (possibly stacked over multiple time steps) and ``wT`` is the
    target field at a future time.

    Args:
        n_samples: Number of samples.
        h: Spatial resolution (height).
        w: Spatial resolution (width).
        T_in: Number of input time steps.
        path: Optional path to a ``.npz`` file with keys ``"a"``
            (shape ``(N, h, w, T_in)``) and ``"u"`` (shape ``(N, h, w)``).
    """

    def __init__(
        self,
        n_samples: int = 200,
        h: int = 64,
        w: int = 64,
        T_in: int = 10,
        path: Optional[Path] = None,
    ) -> None:
        super().__init__()
        if path is not None and Path(path).exists():
            data = np.load(path)
            a = torch.from_numpy(data["a"].astype(np.float32))
            u = torch.from_numpy(data["u"].astype(np.float32))
            self.a = a[:n_samples]
            self.u = u[:n_samples]
        else:
            # Random Gaussian fields as stand-in
            self.a = torch.randn(n_samples, h, w, T_in)
            self.u = torch.randn(n_samples, h, w)

        self.grid = _grid2d(h, w)  # (h, w, 2)

    def __len__(self) -> int:
        return len(self.a)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        a = self.a[idx]   # (h, w, T_in)
        u = self.u[idx]   # (h, w)
        grid = self.grid  # (h, w, 2)
        # concatenate time-series input and grid → (h, w, T_in + 2)
        x_in = torch.cat([a, grid], dim=-1)
        return x_in, u.unsqueeze(-1)


# ---------------------------------------------------------------------------
# Data-loader factory
# ---------------------------------------------------------------------------


def make_dataloaders(
    dataset: Dataset,
    train_ratio: float = 0.8,
    batch_size: int = 32,
    num_workers: int = 0,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader]:
    """Split *dataset* into train/validation sets and return ``DataLoader`` s.

    Args:
        dataset: Source dataset.
        train_ratio: Fraction of samples used for training.
        batch_size: Mini-batch size.
        num_workers: Number of worker processes for data loading.
        seed: Random seed for reproducible splits.

    Returns:
        A tuple ``(train_loader, val_loader)``.
    """
    n_total = len(dataset)
    n_train = int(n_total * train_ratio)
    n_val = n_total - n_train
    generator = torch.Generator().manual_seed(seed)
    train_ds, val_ds = random_split(dataset, [n_train, n_val], generator=generator)
    shuffle_generator = torch.Generator().manual_seed(seed + 1)
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False,
        generator=shuffle_generator,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
    )
    return train_loader, val_loader
