"""Topology-aware placeholder losses used by FAST."""

from __future__ import annotations

import torch
from torch import Tensor


def alignment_loss(proxies: Tensor, matched_points: Tensor) -> Tensor:
    """Penalize distance between continuous proxies and matched real samples.

    Args:
        proxies: Tensor of shape ``[M, D]``.
        matched_points: Tensor of shape ``[M, D]``.

    Returns:
        Scalar tensor.
    """

    return torch.mean((proxies - matched_points) ** 2)


def laplacian_smoothness_loss(features: Tensor, laplacian: Tensor) -> Tensor:
    """Simple graph smoothness regularizer.

    Args:
        features: Tensor of shape ``[N, D]``.
        laplacian: Tensor of shape ``[N, N]``.

    Returns:
        Scalar tensor.

    TODO:
        - Revisit whether this should act on proxies, matched nodes, or both.
    """

    smooth = torch.trace(features.T @ laplacian @ features)
    return smooth / max(features.numel(), 1)


def topology_loss_bundle(
    proxies: Tensor,
    matched_points: Tensor,
    proxy_laplacian: Tensor | None = None,
) -> dict[str, Tensor]:
    """Return a small bundle of graph-related losses.

    This shape-stable helper makes it easy to log separate terms later.
    """

    losses = {"align": alignment_loss(proxies, matched_points)}
    if proxy_laplacian is not None:
        losses["graph"] = laplacian_smoothness_loss(proxies, proxy_laplacian)
    else:
        losses["graph"] = torch.zeros((), dtype=proxies.dtype, device=proxies.device)
    return losses
