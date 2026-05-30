"""Training utilities for FNO models.

Provides:
* ``relative_l2_loss`` – the standard relative L2 loss used in FNO papers.
* ``Trainer`` – a lightweight trainer that handles the training loop,
  validation, checkpointing, and basic logging.
"""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Callable, Dict, Optional

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------


def relative_l2_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Relative L2 (Frobenius) loss, averaged over the batch.

    .. math::

        \\mathcal{L} = \\frac{1}{B} \\sum_{b=1}^{B}
            \\frac{\\|\\hat{u}_b - u_b\\|_2}{\\|u_b\\|_2}

    Args:
        pred: Predicted tensor of shape ``(batch, ...)``.
        target: Ground-truth tensor of the same shape.

    Returns:
        Scalar loss tensor.
    """
    batch = pred.shape[0]
    diff = (pred - target).reshape(batch, -1).norm(dim=1)
    norm = target.reshape(batch, -1).norm(dim=1).clamp(min=1e-8)
    return (diff / norm).mean()


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------


class Trainer:
    """Lightweight FNO trainer.

    Args:
        model: The FNO model to train.
        train_loader: DataLoader for training data.
        val_loader: DataLoader for validation data.
        learning_rate: Initial learning rate (default ``1e-3``).
        weight_decay: AdamW weight decay (default ``1e-4``).
        n_epochs: Total number of training epochs (default 100).
        scheduler: Optional LR scheduler; if ``None`` a cosine annealing
            schedule is used.
        loss_fn: Loss function ``(pred, target) → scalar``; defaults to
            :func:`relative_l2_loss`.
        checkpoint_dir: Directory to save best-model checkpoints.
            Pass ``None`` to disable checkpointing.
        device: Target device.  Defaults to CUDA if available, else CPU.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        n_epochs: int = 100,
        scheduler: Optional[object] = None,
        loss_fn: Optional[Callable] = None,
        checkpoint_dir: Optional[Path] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.n_epochs = n_epochs
        self.loss_fn = loss_fn or relative_l2_loss
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None
        if self.checkpoint_dir is not None:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.optimizer = Adam(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
        self.scheduler = scheduler or CosineAnnealingLR(
            self.optimizer, T_max=n_epochs
        )

        self._best_val_loss = math.inf
        self.history: Dict[str, list] = {"train_loss": [], "val_loss": []}

    # ------------------------------------------------------------------
    # internal helpers
    # ------------------------------------------------------------------

    def _train_epoch(self) -> float:
        self.model.train()
        total_loss = 0.0
        for x, y in self.train_loader:
            x, y = x.to(self.device), y.to(self.device)
            self.optimizer.zero_grad()
            pred = self.model(x)
            loss = self.loss_fn(pred, y)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item() * x.shape[0]
        return total_loss / len(self.train_loader.dataset)

    @torch.no_grad()
    def _val_epoch(self) -> float:
        self.model.eval()
        total_loss = 0.0
        for x, y in self.val_loader:
            x, y = x.to(self.device), y.to(self.device)
            pred = self.model(x)
            loss = self.loss_fn(pred, y)
            total_loss += loss.item() * x.shape[0]
        return total_loss / len(self.val_loader.dataset)

    def _save_checkpoint(self, epoch: int, val_loss: float) -> None:
        if self.checkpoint_dir is None:
            return
        path = self.checkpoint_dir / "best_model.pt"
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "val_loss": val_loss,
            },
            path,
        )

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    def train(self) -> Dict[str, list]:
        """Run the full training loop.

        Returns:
            A dict with keys ``"train_loss"`` and ``"val_loss"`` containing
            per-epoch loss values.
        """
        print(
            f"Training on {self.device} for {self.n_epochs} epochs "
            f"| train batches: {len(self.train_loader)} "
            f"| val batches: {len(self.val_loader)}"
        )
        for epoch in range(1, self.n_epochs + 1):
            t0 = time.time()
            train_loss = self._train_epoch()
            val_loss = self._val_epoch()
            self.scheduler.step()

            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)

            if val_loss < self._best_val_loss:
                self._best_val_loss = val_loss
                self._save_checkpoint(epoch, val_loss)

            elapsed = time.time() - t0
            print(
                f"Epoch {epoch:4d}/{self.n_epochs} | "
                f"train {train_loss:.4e} | val {val_loss:.4e} | "
                f"{elapsed:.1f}s"
            )

        return self.history
