"""Spectral convolution layers used in the Fourier Neural Operator."""

import torch
import torch.nn as nn


class SpectralConv1d(nn.Module):
    """1-D spectral (Fourier) convolution layer.

    Applies a linear transform in the truncated Fourier domain, then maps back
    to the spatial domain via the inverse FFT.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        modes: Number of Fourier modes to retain (must be ≤ ``n // 2 + 1``
            where ``n`` is the spatial resolution).
    """

    def __init__(self, in_channels: int, out_channels: int, modes: int) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes

        scale = 1.0 / (in_channels * out_channels)
        self.weights = nn.Parameter(
            scale * torch.rand(in_channels, out_channels, modes, dtype=torch.cfloat)
        )

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _complex_mul1d(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        """Batched complex matrix-vector multiply along the channel axis.

        Args:
            x: Shape ``(batch, in_channels, modes)``, complex.
            w: Shape ``(in_channels, out_channels, modes)``, complex.

        Returns:
            Shape ``(batch, out_channels, modes)``, complex.
        """
        return torch.einsum("bim,iom->bom", x, w)

    # ------------------------------------------------------------------
    # forward
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply spectral convolution.

        Args:
            x: Real-valued input tensor of shape ``(batch, in_channels, n)``.

        Returns:
            Real-valued output tensor of shape ``(batch, out_channels, n)``.
        """
        batch, _, n = x.shape

        x_ft = torch.fft.rfft(x)
        out_ft = torch.zeros(
            batch, self.out_channels, n // 2 + 1, dtype=torch.cfloat, device=x.device
        )
        out_ft[:, :, : self.modes] = self._complex_mul1d(
            x_ft[:, :, : self.modes], self.weights
        )
        return torch.fft.irfft(out_ft, n=n)


class SpectralConv2d(nn.Module):
    """2-D spectral (Fourier) convolution layer.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        modes1: Number of Fourier modes along the first spatial axis.
        modes2: Number of Fourier modes along the second spatial axis.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        modes1: int,
        modes2: int,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2

        scale = 1.0 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(
            scale
            * torch.rand(
                in_channels, out_channels, modes1, modes2, dtype=torch.cfloat
            )
        )
        self.weights2 = nn.Parameter(
            scale
            * torch.rand(
                in_channels, out_channels, modes1, modes2, dtype=torch.cfloat
            )
        )

    @staticmethod
    def _complex_mul2d(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        return torch.einsum("bixy,ioxy->boxy", x, w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply 2-D spectral convolution.

        Args:
            x: Shape ``(batch, in_channels, h, w)``.

        Returns:
            Shape ``(batch, out_channels, h, w)``.
        """
        batch, _, h, w = x.shape

        x_ft = torch.fft.rfft2(x)
        out_ft = torch.zeros(
            batch,
            self.out_channels,
            h,
            w // 2 + 1,
            dtype=torch.cfloat,
            device=x.device,
        )

        # upper-left block
        out_ft[:, :, : self.modes1, : self.modes2] = self._complex_mul2d(
            x_ft[:, :, : self.modes1, : self.modes2], self.weights1
        )
        # lower-left block (negative frequencies along first axis)
        out_ft[:, :, -self.modes1 :, : self.modes2] = self._complex_mul2d(
            x_ft[:, :, -self.modes1 :, : self.modes2], self.weights2
        )

        return torch.fft.irfft2(out_ft, s=(h, w))


class SpectralConv3d(nn.Module):
    """3-D spectral (Fourier) convolution layer.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        modes1: Fourier modes along spatial axis 1.
        modes2: Fourier modes along spatial axis 2.
        modes3: Fourier modes along spatial axis 3.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        modes1: int,
        modes2: int,
        modes3: int,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2
        self.modes3 = modes3

        scale = 1.0 / (in_channels * out_channels)
        shape = (in_channels, out_channels, modes1, modes2, modes3)
        self.weights1 = nn.Parameter(scale * torch.rand(*shape, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(scale * torch.rand(*shape, dtype=torch.cfloat))
        self.weights3 = nn.Parameter(scale * torch.rand(*shape, dtype=torch.cfloat))
        self.weights4 = nn.Parameter(scale * torch.rand(*shape, dtype=torch.cfloat))

    @staticmethod
    def _complex_mul3d(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        return torch.einsum("bixyz,ioxyz->boxyz", x, w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply 3-D spectral convolution.

        Args:
            x: Shape ``(batch, in_channels, d, h, w)``.

        Returns:
            Shape ``(batch, out_channels, d, h, w)``.
        """
        batch, _, d, h, w = x.shape

        x_ft = torch.fft.rfftn(x, dim=[-3, -2, -1])
        out_ft = torch.zeros(
            batch,
            self.out_channels,
            d,
            h,
            w // 2 + 1,
            dtype=torch.cfloat,
            device=x.device,
        )

        m1, m2, m3 = self.modes1, self.modes2, self.modes3
        out_ft[:, :, :m1, :m2, :m3] = self._complex_mul3d(
            x_ft[:, :, :m1, :m2, :m3], self.weights1
        )
        out_ft[:, :, -m1:, :m2, :m3] = self._complex_mul3d(
            x_ft[:, :, -m1:, :m2, :m3], self.weights2
        )
        out_ft[:, :, :m1, -m2:, :m3] = self._complex_mul3d(
            x_ft[:, :, :m1, -m2:, :m3], self.weights3
        )
        out_ft[:, :, -m1:, -m2:, :m3] = self._complex_mul3d(
            x_ft[:, :, -m1:, -m2:, :m3], self.weights4
        )

        return torch.fft.irfftn(out_ft, s=(d, h, w))
