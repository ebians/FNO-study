"""
FNO (Fourier Neural Operator) package for learning solution operators of PDEs.

References:
    Li, Z., et al. (2021). Fourier Neural Operator for Parametric Partial
    Differential Equations. ICLR 2021. https://arxiv.org/abs/2010.08895
"""

from .model import FNO1d, FNO2d, FNO3d
from .layers import SpectralConv1d, SpectralConv2d, SpectralConv3d

__all__ = [
    "FNO1d",
    "FNO2d",
    "FNO3d",
    "SpectralConv1d",
    "SpectralConv2d",
    "SpectralConv3d",
]
