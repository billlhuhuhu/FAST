"""Diversity-related placeholder losses for coreset optimization."""

from __future__ import annotations

import torch
from torch import Tensor


def rbf_kernel(features: Tensor, gamma: float = 1.0) -> Tensor:
    """Construct an RBF kernel matrix.

    Args:
        features: Tensor of shape ``[M, D]``.
        gamma: Kernel sharpness.

    Returns:
        Kernel matrix of shape ``[M, M]``.
    """

    distances = torch.cdist(features, features, p=2.0) ** 2
    return torch.exp(-gamma * distances)


def dpp_diversity_loss(features: Tensor, eps: float = 1e-6) -> Tensor:
    """Compute a simple DPP-style diversity loss.

    Args:
        features: Tensor of shape ``[M, D]``.
        eps: Diagonal jitter for numerical stability.

    Returns:
        Scalar tensor.

    TODO:
        - Verify the exact feature space used in FAST for DPP.
        - Support RFF-based kernels if that matches the paper more closely.
    """

    kernel = rbf_kernel(features)
    kernel = kernel + eps * torch.eye(kernel.shape[0], device=kernel.device, dtype=kernel.dtype)
    sign, logabsdet = torch.linalg.slogdet(kernel)
    if torch.any(sign <= 0):
        return torch.tensor(0.0, device=features.device, dtype=features.dtype)
    return -logabsdet
