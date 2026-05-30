"""Tests for data utilities."""

import pytest
import torch

from fno.data import (
    BurgersDataset,
    NavierStokesDataset,
    UnitGaussianNormalizer,
    make_dataloaders,
)


class TestUnitGaussianNormalizer:
    def test_encode_decode_roundtrip(self):
        x = torch.randn(100, 10)
        norm = UnitGaussianNormalizer(x)
        x_enc = norm.encode(x)
        x_dec = norm.decode(x_enc)
        assert torch.allclose(x, x_dec, atol=1e-5)

    def test_encoded_statistics(self):
        torch.manual_seed(0)
        x = torch.randn(1000, 4) * 3 + 5
        norm = UnitGaussianNormalizer(x)
        x_enc = norm.encode(x)
        # mean ~ 0, std ~ 1 (within tolerance for finite samples)
        assert x_enc.mean(0).abs().max() < 0.1
        assert (x_enc.std(0) - 1).abs().max() < 0.1

    def test_to_device(self):
        x = torch.randn(50, 3)
        norm = UnitGaussianNormalizer(x)
        norm.to("cpu")  # should not raise


class TestBurgersDataset:
    def test_length(self):
        ds = BurgersDataset(n_samples=50, n_x=64)
        assert len(ds) == 50

    def test_item_shapes(self):
        ds = BurgersDataset(n_samples=10, n_x=64)
        x, y = ds[0]
        assert x.shape == (64, 2)
        assert y.shape == (64, 1)

    def test_grid_in_range(self):
        ds = BurgersDataset(n_samples=5, n_x=32)
        x, _ = ds[0]
        grid = x[:, 1]
        assert grid.min() >= 0.0
        assert grid.max() <= 1.0


class TestNavierStokesDataset:
    def test_length(self):
        ds = NavierStokesDataset(n_samples=20, h=16, w=16, T_in=5)
        assert len(ds) == 20

    def test_item_shapes(self):
        ds = NavierStokesDataset(n_samples=5, h=16, w=16, T_in=5)
        x, y = ds[0]
        assert x.shape == (16, 16, 7)   # T_in + 2 grid channels
        assert y.shape == (16, 16, 1)


class TestMakeDataloaders:
    def test_split_sizes(self):
        ds = BurgersDataset(n_samples=100, n_x=32)
        train_loader, val_loader = make_dataloaders(
            ds, train_ratio=0.8, batch_size=16
        )
        assert len(train_loader.dataset) == 80
        assert len(val_loader.dataset) == 20

    def test_batch_shapes(self):
        ds = BurgersDataset(n_samples=20, n_x=32)
        train_loader, _ = make_dataloaders(ds, train_ratio=0.8, batch_size=4)
        x, y = next(iter(train_loader))
        assert x.shape == (4, 32, 2)
        assert y.shape == (4, 32, 1)

    def test_reproducible_split(self):
        ds = BurgersDataset(n_samples=50, n_x=16)
        train1, _ = make_dataloaders(ds, seed=42, batch_size=10)
        train2, _ = make_dataloaders(ds, seed=42, batch_size=10)
        x1, _ = next(iter(train1))
        x2, _ = next(iter(train2))
        assert torch.allclose(x1, x2)
