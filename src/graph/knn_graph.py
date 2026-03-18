"""kNN graph construction utilities for the FAST scaffold."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass
class KnnGraph:
    """Lightweight kNN graph container.

    Attributes:
        indices: Neighbor indices with shape ``[N, k]``.
        distances: Pairwise distances with shape ``[N, k]``.
        weights: Edge weights with shape ``[N, k]``.
    """

    indices: Tensor
    distances: Tensor
    weights: Tensor


def pairwise_squared_distances(features: Tensor) -> Tensor:
    """Compute dense squared Euclidean distances.

    Args:
        features: Tensor of shape ``[N, D]``.

    Returns:
        Tensor of shape ``[N, N]``.

    TODO:
        - Replace with a memory-aware chunked version for larger datasets.
    """

    if features.ndim != 2:
        raise ValueError(f"Expected features with shape [N, D], got {tuple(features.shape)}")
    return torch.cdist(features, features, p=2.0) ** 2


def build_knn_graph(features: Tensor, k: int, sigma: float = 1.0) -> KnnGraph:
    """Build a simple weighted kNN graph.

    Args:
        features: Tensor of shape ``[N, D]``.
        k: Number of neighbors per node.
        sigma: RBF width for edge weighting.

    Returns:
        A :class:`KnnGraph` object with shape-safe tensors.

    TODO:
        - Add mutual kNN and multi-scale graph options.
        - Add fuzzy simplicial set construction closer to the paper.
    """

    if k <= 0:
        raise ValueError("k must be positive")

    distances = pairwise_squared_distances(features)
    distances.fill_diagonal_(float("inf"))
    knn_distances, knn_indices = torch.topk(distances, k=k, largest=False, dim=1)
    weights = torch.exp(-knn_distances / max(sigma, 1e-12))
    return KnnGraph(indices=knn_indices, distances=knn_distances, weights=weights)
