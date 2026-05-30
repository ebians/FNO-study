"""Tests for training utilities."""

import pytest
import torch
import torch.nn as nn

from fno import FNO1d
from fno.data import BurgersDataset, make_dataloaders
from fno.train import Trainer, relative_l2_loss


class TestRelativeL2Loss:
    def test_zero_loss_on_perfect_prediction(self):
        x = torch.randn(4, 32, 1)
        loss = relative_l2_loss(x, x)
        assert loss.item() == pytest.approx(0.0, abs=1e-6)

    def test_loss_is_positive(self):
        pred = torch.randn(4, 32, 1)
        target = torch.randn(4, 32, 1)
        loss = relative_l2_loss(pred, target)
        assert loss.item() >= 0.0

    def test_scaled_target_gives_higher_loss(self):
        # Relative L2: ||pred - target|| / ||target||
        # A prediction that is identically zero has loss == 1.0 (relative norm of target).
        # A prediction equal to target has loss == 0.0.
        # A prediction twice the target: ||target|| / ||target|| == 1.0.
        target = torch.ones(4, 10)
        pred_good = target + 0.01 * torch.ones_like(target)  # close to target
        pred_bad = torch.zeros_like(target)                   # zero prediction

        loss_good = relative_l2_loss(pred_good, target)
        loss_bad = relative_l2_loss(pred_bad, target)
        assert loss_good.item() < loss_bad.item()


class TestTrainer:
    def _make_trainer(self, n_epochs: int = 2) -> Trainer:
        torch.manual_seed(0)
        model = FNO1d(modes=4, width=8, in_channels=2, out_channels=1, n_layers=2)
        dataset = BurgersDataset(n_samples=20, n_x=16)
        train_loader, val_loader = make_dataloaders(
            dataset, train_ratio=0.8, batch_size=4
        )
        return Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            n_epochs=n_epochs,
            checkpoint_dir=None,
            device=torch.device("cpu"),
        )

    def test_history_length(self):
        trainer = self._make_trainer(n_epochs=3)
        history = trainer.train()
        assert len(history["train_loss"]) == 3
        assert len(history["val_loss"]) == 3

    def test_loss_values_are_finite(self):
        trainer = self._make_trainer(n_epochs=2)
        history = trainer.train()
        for loss in history["train_loss"] + history["val_loss"]:
            assert torch.isfinite(torch.tensor(loss))

    def test_checkpoint_saved(self, tmp_path):
        torch.manual_seed(0)
        model = FNO1d(modes=4, width=8, in_channels=2, out_channels=1, n_layers=1)
        dataset = BurgersDataset(n_samples=10, n_x=16)
        train_loader, val_loader = make_dataloaders(
            dataset, train_ratio=0.8, batch_size=4
        )
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            n_epochs=2,
            checkpoint_dir=tmp_path,
            device=torch.device("cpu"),
        )
        trainer.train()
        assert (tmp_path / "best_model.pt").exists()
