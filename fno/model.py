"""Fourier Neural Operator (FNO) model definitions.

Implements FNO1d, FNO2d, and FNO3d as described in:

    Li, Z., et al. (2021). Fourier Neural Operator for Parametric Partial
    Differential Equations. ICLR 2021. https://arxiv.org/abs/2010.08895
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import SpectralConv1d, SpectralConv2d, SpectralConv3d


class FNO1d(nn.Module):
    """Fourier Neural Operator for 1-D problems (e.g. Burgers' equation).

    Architecture::

        x  →  P  →  [FourierLayer] × n_layers  →  Q  →  y

    where each *FourierLayer* combines a spectral convolution with a local
    linear skip connection followed by an activation.

    Args:
        modes: Number of Fourier modes to retain per layer.
        width: Feature width (number of channels inside the FNO trunk).
        in_channels: Number of input channels (default 2 = solution + grid).
        out_channels: Number of output channels (default 1).
        n_layers: Number of Fourier layers (default 4).
        activation: Pointwise activation applied between layers (default GELU).
    """

    def __init__(
        self,
        modes: int,
        width: int,
        in_channels: int = 2,
        out_channels: int = 1,
        n_layers: int = 4,
        activation: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.modes = modes
        self.width = width
        self.n_layers = n_layers
        self.activation = activation or nn.GELU()

        # lifting layer: R^{in_channels} → R^{width}
        self.p = nn.Linear(in_channels, width)

        self.spectral_convs = nn.ModuleList(
            [SpectralConv1d(width, width, modes) for _ in range(n_layers)]
        )
        self.skip_convs = nn.ModuleList(
            [nn.Conv1d(width, width, 1) for _ in range(n_layers)]
        )

        # projection layers: R^{width} → R^{out_channels}
        self.q1 = nn.Linear(width, width * 4)
        self.q2 = nn.Linear(width * 4, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape ``(batch, n, in_channels)``.

        Returns:
            Output tensor of shape ``(batch, n, out_channels)``.
        """
        # lift
        x = self.p(x)           # (batch, n, width)
        x = x.permute(0, 2, 1)  # (batch, width, n)

        for spectral, skip in zip(self.spectral_convs, self.skip_convs):
            x1 = spectral(x)
            x2 = skip(x)
            x = self.activation(x1 + x2)

        x = x.permute(0, 2, 1)  # (batch, n, width)
        x = self.activation(self.q1(x))
        x = self.q2(x)
        return x


class FNO2d(nn.Module):
    """Fourier Neural Operator for 2-D problems (e.g. Navier-Stokes).

    Args:
        modes1: Fourier modes along the first spatial axis.
        modes2: Fourier modes along the second spatial axis.
        width: Feature width.
        in_channels: Number of input channels (default 12 = 10 time steps + 2 grid coords).
        out_channels: Number of output channels (default 1).
        n_layers: Number of Fourier layers (default 4).
        activation: Pointwise activation (default GELU).
    """

    def __init__(
        self,
        modes1: int,
        modes2: int,
        width: int,
        in_channels: int = 12,
        out_channels: int = 1,
        n_layers: int = 4,
        activation: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.n_layers = n_layers
        self.activation = activation or nn.GELU()

        self.p = nn.Linear(in_channels, width)

        self.spectral_convs = nn.ModuleList(
            [
                SpectralConv2d(width, width, modes1, modes2)
                for _ in range(n_layers)
            ]
        )
        self.skip_convs = nn.ModuleList(
            [nn.Conv2d(width, width, 1) for _ in range(n_layers)]
        )

        self.q1 = nn.Linear(width, width * 4)
        self.q2 = nn.Linear(width * 4, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape ``(batch, h, w, in_channels)``.

        Returns:
            Output tensor of shape ``(batch, h, w, out_channels)``.
        """
        # lift
        x = self.p(x)                   # (batch, h, w, width)
        x = x.permute(0, 3, 1, 2)       # (batch, width, h, w)

        for spectral, skip in zip(self.spectral_convs, self.skip_convs):
            x1 = spectral(x)
            x2 = skip(x)
            x = self.activation(x1 + x2)

        x = x.permute(0, 2, 3, 1)       # (batch, h, w, width)
        x = self.activation(self.q1(x))
        x = self.q2(x)
        return x


class FNO3d(nn.Module):
    """Fourier Neural Operator for 3-D problems (e.g. spatiotemporal PDEs).

    Args:
        modes1: Fourier modes along axis 1.
        modes2: Fourier modes along axis 2.
        modes3: Fourier modes along axis 3 (time or depth).
        width: Feature width.
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        n_layers: Number of Fourier layers (default 4).
        activation: Pointwise activation (default GELU).
    """

    def __init__(
        self,
        modes1: int,
        modes2: int,
        modes3: int,
        width: int,
        in_channels: int = 4,
        out_channels: int = 1,
        n_layers: int = 4,
        activation: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.modes1 = modes1
        self.modes2 = modes2
        self.modes3 = modes3
        self.width = width
        self.n_layers = n_layers
        self.activation = activation or nn.GELU()

        self.p = nn.Linear(in_channels, width)

        self.spectral_convs = nn.ModuleList(
            [
                SpectralConv3d(width, width, modes1, modes2, modes3)
                for _ in range(n_layers)
            ]
        )
        self.skip_convs = nn.ModuleList(
            [nn.Conv3d(width, width, 1) for _ in range(n_layers)]
        )

        self.q1 = nn.Linear(width, width * 4)
        self.q2 = nn.Linear(width * 4, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape ``(batch, d, h, w, in_channels)``.

        Returns:
            Output tensor of shape ``(batch, d, h, w, out_channels)``.
        """
        # lift
        x = self.p(x)                      # (batch, d, h, w, width)
        x = x.permute(0, 4, 1, 2, 3)       # (batch, width, d, h, w)

        for spectral, skip in zip(self.spectral_convs, self.skip_convs):
            x1 = spectral(x)
            x2 = skip(x)
            x = self.activation(x1 + x2)

        x = x.permute(0, 2, 3, 4, 1)       # (batch, d, h, w, width)
        x = self.activation(self.q1(x))
        x = self.q2(x)
        return x
