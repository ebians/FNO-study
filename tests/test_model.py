"""Tests for FNO model forward passes."""

import pytest
import torch

from fno import FNO1d, FNO2d, FNO3d


class TestFNO1d:
    def test_output_shape(self):
        model = FNO1d(modes=16, width=32, in_channels=2, out_channels=1)
        x = torch.randn(4, 128, 2)
        out = model(x)
        assert out.shape == (4, 128, 1)

    def test_multi_output(self):
        model = FNO1d(modes=8, width=16, in_channels=3, out_channels=2, n_layers=2)
        x = torch.randn(2, 64, 3)
        out = model(x)
        assert out.shape == (2, 64, 2)

    def test_parameter_count_scales_with_width(self):
        m16 = FNO1d(modes=8, width=16)
        m32 = FNO1d(modes=8, width=32)
        p16 = sum(p.numel() for p in m16.parameters())
        p32 = sum(p.numel() for p in m32.parameters())
        assert p32 > p16

    def test_backward_runs(self):
        model = FNO1d(modes=8, width=16, in_channels=2, out_channels=1, n_layers=2)
        x = torch.randn(2, 32, 2)
        y = torch.randn(2, 32, 1)
        pred = model(x)
        loss = (pred - y).pow(2).mean()
        loss.backward()
        for param in model.parameters():
            assert param.grad is not None

    def test_different_layer_counts(self):
        for n in [1, 2, 4, 6]:
            model = FNO1d(modes=4, width=8, n_layers=n)
            x = torch.randn(1, 16, 2)
            out = model(x)
            assert out.shape == (1, 16, 1)


class TestFNO2d:
    def test_output_shape(self):
        model = FNO2d(modes1=12, modes2=12, width=20, in_channels=12, out_channels=1)
        x = torch.randn(2, 64, 64, 12)
        out = model(x)
        assert out.shape == (2, 64, 64, 1)

    def test_asymmetric_resolution(self):
        model = FNO2d(modes1=4, modes2=6, width=8, in_channels=3, out_channels=1)
        x = torch.randn(1, 16, 24, 3)
        out = model(x)
        assert out.shape == (1, 16, 24, 1)

    def test_backward_runs(self):
        model = FNO2d(modes1=4, modes2=4, width=8, in_channels=4, out_channels=1, n_layers=2)
        x = torch.randn(2, 16, 16, 4)
        y = torch.randn(2, 16, 16, 1)
        pred = model(x)
        loss = (pred - y).pow(2).mean()
        loss.backward()
        for param in model.parameters():
            assert param.grad is not None


class TestFNO3d:
    def test_output_shape(self):
        model = FNO3d(
            modes1=4, modes2=4, modes3=4,
            width=8, in_channels=4, out_channels=1,
            n_layers=2,
        )
        x = torch.randn(1, 8, 8, 8, 4)
        out = model(x)
        assert out.shape == (1, 8, 8, 8, 1)

    def test_backward_runs(self):
        model = FNO3d(
            modes1=2, modes2=2, modes3=2,
            width=4, in_channels=2, out_channels=1,
            n_layers=2,
        )
        x = torch.randn(1, 4, 4, 4, 2)
        y = torch.randn(1, 4, 4, 4, 1)
        pred = model(x)
        loss = (pred - y).pow(2).mean()
        loss.backward()
        for param in model.parameters():
            assert param.grad is not None
