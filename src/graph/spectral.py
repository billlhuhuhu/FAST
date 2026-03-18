"""Spectral graph helpers for topology-aware losses."""

from __future__ import annotations

import torch
from torch import Tensor

from src.graph.knn_graph import KnnGraph


def knn_graph_to_adjacency(graph: KnnGraph, num_nodes: int) -> Tensor:
    """Convert a kNN graph into a dense adjacency matrix.

    Args:
        graph: Graph container with tensors of shape ``[N, k]``.
        num_nodes: Number of nodes ``N``.

    Returns:
        Dense adjacency matrix with shape ``[N, N]``.
    """

    adjacency = torch.zeros((num_nodes, num_nodes), dtype=graph.weights.dtype, device=graph.weights.device)
    row_ids = torch.arange(num_nodes, device=graph.indices.device).unsqueeze(1).expand_as(graph.indices)
    adjacency[row_ids, graph.indices] = graph.weights
    adjacency = torch.maximum(adjacency, adjacency.T)
    return adjacency


def compute_graph_laplacian(adjacency: Tensor, normalized: bool = True) -> Tensor:
    """Compute the combinatorial or normalized graph Laplacian.

    Args:
        adjacency: Tensor of shape ``[N, N]``.
        normalized: Whether to return the symmetric normalized Laplacian.

    Returns:
        Tensor of shape ``[N, N]``.

    TODO:
        - Add sparse support for large graphs.
    """

    degree = adjacency.sum(dim=1)
    if not normalized:
        return torch.diag(degree) - adjacency

    inv_sqrt_degree = torch.rsqrt(torch.clamp(degree, min=1e-12))
    d_mat = torch.diag(inv_sqrt_degree)
    identity = torch.eye(adjacency.shape[0], dtype=adjacency.dtype, device=adjacency.device)
    return identity - d_mat @ adjacency @ d_mat


def spectral_embedding(laplacian: Tensor, num_dims: int) -> Tensor:
    """Return a small spectral embedding placeholder.

    Args:
        laplacian: Tensor of shape ``[N, N]``.
        num_dims: Number of eigenvectors to keep.

    Returns:
        Tensor of shape ``[N, num_dims]``.

    TODO:
        - Validate the exact eigenvector selection strategy used by FAST.
    """

    eigenvalues, eigenvectors = torch.linalg.eigh(laplacian)
    _ = eigenvalues
    return eigenvectors[:, :num_dims]
