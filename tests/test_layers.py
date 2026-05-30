"""Tests for spectral convolution layers."""

import pytest
import torch

from fno.layers import SpectralConv1d, SpectralConv2d, SpectralConv3d


class TestSpectralConv1d:
    def test_output_shape(self):
        layer = SpectralConv1d(in_channels=4, out_channels=8, modes=12)
        x = torch.randn(2, 4, 64)
        out = layer(x)
        assert out.shape == (2, 8, 64)

    def test_modes_smaller_than_n(self):
        layer = SpectralConv1d(in_channels=2, out_channels=2, modes=5)
        x = torch.randn(1, 2, 16)
        out = layer(x)
        assert out.shape == (1, 2, 16)

    def test_output_is_real(self):
        layer = SpectralConv1d(in_channels=2, out_channels=2, modes=8)
        x = torch.randn(3, 2, 32)
        out = layer(x)
        assert out.is_floating_point()
        assert not out.is_complex()

    def test_backward_runs(self):
        layer = SpectralConv1d(in_channels=3, out_channels=3, modes=8)
        x = torch.randn(2, 3, 32, requires_grad=True)
        loss = layer(x).sum()
        loss.backward()
        assert x.grad is not None

    def test_parameter_dtype(self):
        layer = SpectralConv1d(in_channels=2, out_channels=2, modes=4)
        assert layer.weights.dtype == torch.cfloat


class TestSpectralConv2d:
    def test_output_shape(self):
        layer = SpectralConv2d(in_channels=4, out_channels=8, modes1=6, modes2=6)
        x = torch.randn(2, 4, 32, 32)
        out = layer(x)
        assert out.shape == (2, 8, 32, 32)

    def test_asymmetric_modes(self):
        layer = SpectralConv2d(in_channels=2, out_channels=2, modes1=4, modes2=6)
        x = torch.randn(1, 2, 16, 24)
        out = layer(x)
        assert out.shape == (1, 2, 16, 24)

    def test_backward_runs(self):
        layer = SpectralConv2d(in_channels=2, out_channels=2, modes1=4, modes2=4)
        x = torch.randn(2, 2, 16, 16, requires_grad=True)
        loss = layer(x).sum()
        loss.backward()
        assert x.grad is not None


class TestSpectralConv3d:
    def test_output_shape(self):
        layer = SpectralConv3d(
            in_channels=2, out_channels=4, modes1=4, modes2=4, modes3=4
        )
        x = torch.randn(1, 2, 8, 8, 8)
        out = layer(x)
        assert out.shape == (1, 4, 8, 8, 8)

    def test_backward_runs(self):
        layer = SpectralConv3d(
            in_channels=2, out_channels=2, modes1=2, modes2=2, modes3=2
        )
        x = torch.randn(1, 2, 8, 8, 8, requires_grad=True)
        loss = layer(x).sum()
        loss.backward()
        assert x.grad is not None
