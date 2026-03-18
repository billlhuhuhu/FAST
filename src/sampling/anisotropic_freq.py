"""Frequency-library initialization for the FAST scaffold."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass
class FrequencyLibrary:
    """Container for candidate frequencies.

    Attributes:
        omega: Frequency tensor with shape ``[F, D]``.
        norms: Norm tensor with shape ``[F]``.
    """

    omega: Tensor
    norms: Tensor


def build_anisotropic_frequency_library(
    feature_dim: int,
    num_frequencies: int,
    max_norm: float,
    device: torch.device | None = None,
) -> FrequencyLibrary:
    """Create a simple placeholder frequency library.

    TODO:
        - Replace isotropic sampling with the paper's AFL procedure.
        - Add low/mid/high frequency band initialization.
    """

    omega = torch.randn(num_frequencies, feature_dim, device=device)
    omega = omega / torch.clamp(torch.linalg.norm(omega, dim=1, keepdim=True), min=1e-12)
    scales = torch.linspace(0.1, max_norm, num_frequencies, device=device).unsqueeze(1)
    omega = omega * scales
    norms = torch.linalg.norm(omega, dim=1)
    return FrequencyLibrary(omega=omega, norms=norms)
