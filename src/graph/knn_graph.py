"""Multi-scale kNN graph construction utilities for the FAST scaffold.

This module implements a clear, correctness-first version of a multi-scale
fuzzy kNN graph:

1. build kNN neighborhoods for each ``k`` in ``k_list``,
2. estimate local smoothing parameters ``rho_i`` and ``sigma_i``,
3. construct directed fuzzy adjacency matrices,
4. apply fuzzy union symmetrization,
5. fuse multiple scales,
6. add minimum spanning tree (MST) edges for connectivity.

The implementation intentionally prioritizes readability over speed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Sequence, Union

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import minimum_spanning_tree
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import NearestNeighbors

try:
    import torch
    from torch import Tensor
except ImportError:  # pragma: no cover - torch is expected in the project env
    torch = None
    Tensor = object


ArrayLike2D = Union[np.ndarray, Tensor]


@dataclass
class KnnGraph:
    """Single-scale kNN graph container.

    Attributes:
        indices:
            Neighbor indices with shape ``[N, k]``.
        distances:
            Neighbor distances with shape ``[N, k]``.
        weights:
            Directed fuzzy membership weights with shape ``[N, k]``.
        adjacency:
            Symmetric sparse adjacency matrix in CSR format with shape ``[N, N]``.
        rho:
            Local connectivity offsets with shape ``[N]``.
        sigma:
            Local smoothing scales with shape ``[N]``.
    """

    indices: np.ndarray
    distances: np.ndarray
    weights: np.ndarray
    adjacency: sp.csr_matrix
    rho: np.ndarray
    sigma: np.ndarray


@dataclass
class MultiScaleKnnGraph:
    """Multi-scale fuzzy kNN graph container.

    Attributes:
        scale_graphs:
            Mapping ``k -> KnnGraph``.
        fused_graph:
            Fused multi-scale graph in CSR format with shape ``[N, N]``.
        mst_graph:
            Symmetric MST edge graph in CSR format with shape ``[N, N]``.
        combined_graph:
            Final graph after merging the fused graph with MST edges.
    """

    scale_graphs: Dict[int, KnnGraph]
    fused_graph: sp.csr_matrix
    mst_graph: sp.csr_matrix
    combined_graph: sp.csr_matrix


def to_numpy_2d(X: ArrayLike2D) -> np.ndarray:
    """Convert a 2D tensor/array to a NumPy array.

    Args:
        X:
            Feature matrix with shape ``[N, p]``.

    Returns:
        NumPy array with shape ``[N, p]`` and dtype ``float64``.
    """

    if torch is not None and isinstance(X, torch.Tensor):
        array = X.detach().cpu().numpy()
    else:
        array = np.asarray(X)

    if array.ndim != 2:
        raise ValueError(f"Expected a 2D feature matrix [N, p], got shape {array.shape}")
    return np.asarray(array, dtype=np.float64)


def pairwise_squared_distances(features: ArrayLike2D) -> np.ndarray:
    """Compute dense squared Euclidean distances.

    Args:
        features:
            Feature matrix with shape ``[N, p]``.

    Returns:
        Dense squared-distance matrix with shape ``[N, N]``.
    """

    X = to_numpy_2d(features)
    distances = pairwise_distances(X, metric="euclidean")
    return distances ** 2


def compute_knn_neighbors(X: ArrayLike2D, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Compute k-nearest neighbors excluding self-neighbors.

    Args:
        X:
            Feature matrix with shape ``[N, p]``.
        k:
            Number of neighbors per point.

    Returns:
        A tuple ``(indices, distances)`` where both arrays have shape ``[N, k]``.
    """

    if k <= 0:
        raise ValueError("k must be positive")

    X_np = to_numpy_2d(X)
    num_nodes = X_np.shape[0]
    effective_k = min(k + 1, num_nodes)
    nbrs = NearestNeighbors(n_neighbors=effective_k, metric="euclidean")
    nbrs.fit(X_np)
    distances, indices = nbrs.kneighbors(X_np, return_distance=True)

    if effective_k > 1:
        distances = distances[:, 1:]
        indices = indices[:, 1:]
    else:
        distances = np.zeros((num_nodes, 0), dtype=np.float64)
        indices = np.zeros((num_nodes, 0), dtype=np.int64)

    return indices, distances


def compute_rho(distances: np.ndarray) -> np.ndarray:
    """Compute local connectivity offsets ``rho_i``.

    Args:
        distances:
            Neighbor distances with shape ``[N, k]``.

    Returns:
        Array with shape ``[N]`` where each entry is the first non-zero neighbor
        distance. If all neighbor distances are zero, ``rho_i`` is set to ``0``.
    """

    rho = np.zeros(distances.shape[0], dtype=np.float64)
    for i in range(distances.shape[0]):
        positive = distances[i][distances[i] > 0.0]
        rho[i] = float(positive[0]) if positive.size > 0 else 0.0
    return rho


def solve_sigmas(
    distances: np.ndarray,
    rho: np.ndarray,
    target: float | None = None,
    n_iter: int = 64,
    tol: float = 1e-5,
) -> np.ndarray:
    """Estimate local smoothing scales ``sigma_i`` using binary search.

    The solver follows the UMAP-style smooth-kNN idea: for each point ``i``,
    find ``sigma_i`` such that the local membership strengths approximately sum
    to a target value.

    Args:
        distances:
            Neighbor distances with shape ``[N, k]``.
        rho:
            Local offsets with shape ``[N]``.
        target:
            Target sum of memberships. Defaults to ``log2(k + 1)``.
        n_iter:
            Number of binary-search iterations.
        tol:
            Tolerance for the target match.

    Returns:
        Array with shape ``[N]``.
    """

    num_nodes, num_neighbors = distances.shape
    if target is None:
        target = np.log2(max(num_neighbors, 1) + 1.0)

    sigmas = np.zeros(num_nodes, dtype=np.float64)
    for i in range(num_nodes):
        lo = 0.0
        hi = 1.0

        def membership_sum(sigma_value: float) -> float:
            total = 0.0
            for d in distances[i]:
                adjusted = d - rho[i]
                if adjusted <= 0.0:
                    total += 1.0
                elif sigma_value <= 1e-12:
                    total += 0.0
                else:
                    total += np.exp(-adjusted / sigma_value)
            return total

        while membership_sum(hi) < target and hi < 1e6:
            hi *= 2.0

        for _ in range(n_iter):
            mid = 0.5 * (lo + hi)
            value = membership_sum(mid)
            if abs(value - target) <= tol:
                lo = mid
                hi = mid
                break
            if value < target:
                lo = mid
            else:
                hi = mid

        sigma = 0.5 * (lo + hi)
        if not np.isfinite(sigma) or sigma <= 0.0:
            sigma = 1e-3
        sigmas[i] = sigma

    return sigmas


def compute_membership_weights(
    distances: np.ndarray,
    rho: np.ndarray,
    sigma: np.ndarray,
) -> np.ndarray:
    """Compute directed fuzzy membership weights.

    Args:
        distances:
            Neighbor distances with shape ``[N, k]``.
        rho:
            Local offsets with shape ``[N]``.
        sigma:
            Local smoothing scales with shape ``[N]``.

    Returns:
        Weight matrix with shape ``[N, k]`` and values in ``[0, 1]``.
    """

    num_nodes, num_neighbors = distances.shape
    weights = np.zeros((num_nodes, num_neighbors), dtype=np.float64)
    for i in range(num_nodes):
        for j in range(num_neighbors):
            adjusted = distances[i, j] - rho[i]
            if adjusted <= 0.0:
                weights[i, j] = 1.0
            else:
                weights[i, j] = np.exp(-adjusted / max(sigma[i], 1e-12))
    return weights


def build_directed_adjacency(
    indices: np.ndarray,
    weights: np.ndarray,
    num_nodes: int,
) -> sp.csr_matrix:
    """Build a directed sparse adjacency matrix.

    Args:
        indices:
            Neighbor indices with shape ``[N, k]``.
        weights:
            Directed edge weights with shape ``[N, k]``.
        num_nodes:
            Number of nodes ``N``.

    Returns:
        CSR sparse matrix with shape ``[N, N]``.
    """

    row_ids = np.repeat(np.arange(num_nodes, dtype=np.int64), indices.shape[1])
    col_ids = indices.reshape(-1)
    data = weights.reshape(-1)
    adjacency = sp.csr_matrix((data, (row_ids, col_ids)), shape=(num_nodes, num_nodes))
    adjacency.eliminate_zeros()
    return adjacency


def fuzzy_union_symmetric(adjacency: sp.spmatrix) -> sp.csr_matrix:
    """Apply fuzzy union symmetrization.

    The formula is:

    ``A_sym = A + A^T - A * A^T``

    where ``*`` denotes elementwise multiplication.

    Args:
        adjacency:
            Sparse adjacency matrix with shape ``[N, N]``.

    Returns:
        Symmetric CSR sparse matrix with shape ``[N, N]``.
    """

    adjacency = adjacency.tocsr()
    transpose = adjacency.T.tocsr()
    sym = adjacency + transpose - adjacency.multiply(transpose)
    sym = sym.tocsr()
    sym.data = np.clip(sym.data, 0.0, 1.0)
    sym.eliminate_zeros()
    return sym


def build_single_scale_graph(X: ArrayLike2D, k: int) -> KnnGraph:
    """Build a single-scale fuzzy kNN graph.

    Args:
        X:
            Feature matrix with shape ``[N, p]``.
        k:
            Neighborhood size.

    Returns:
        A :class:`KnnGraph` object for one scale.
    """

    X_np = to_numpy_2d(X)
    indices, distances = compute_knn_neighbors(X_np, k=k)
    rho = compute_rho(distances)
    sigma = solve_sigmas(distances, rho)
    weights = compute_membership_weights(distances, rho, sigma)
    directed = build_directed_adjacency(indices, weights, num_nodes=X_np.shape[0])
    adjacency = fuzzy_union_symmetric(directed)
    return KnnGraph(
        indices=indices,
        distances=distances,
        weights=weights,
        adjacency=adjacency,
        rho=rho,
        sigma=sigma,
    )


def fuse_multiscale_graphs(scale_graphs: Dict[int, KnnGraph]) -> sp.csr_matrix:
    """Fuse multiple symmetric fuzzy graphs into one graph.

    Args:
        scale_graphs:
            Mapping ``k -> KnnGraph``.

    Returns:
        Fused CSR sparse matrix with shape ``[N, N]``.
    """

    if not scale_graphs:
        raise ValueError("scale_graphs must not be empty")

    fused = None
    for graph in scale_graphs.values():
        fused = graph.adjacency.copy() if fused is None else fuzzy_union_symmetric(fused + graph.adjacency)
    assert fused is not None
    fused = fused.tocsr()
    fused.data = np.clip(fused.data, 0.0, 1.0)
    fused.eliminate_zeros()
    return fused


def build_mst_graph(X: ArrayLike2D) -> sp.csr_matrix:
    """Build a symmetric MST graph from pairwise Euclidean distances.

    Args:
        X:
            Feature matrix with shape ``[N, p]``.

    Returns:
        Symmetric CSR sparse matrix with shape ``[N, N]`` whose nonzero entries
        correspond to MST edges converted into similarity-like weights.
    """

    X_np = to_numpy_2d(X)
    distances = pairwise_distances(X_np, metric="euclidean")
    distance_graph = sp.csr_matrix(distances)
    mst = minimum_spanning_tree(distance_graph).tocsr()
    mst = mst + mst.T
    mst = mst.tocsr()

    if mst.nnz == 0:
        return mst

    positive = mst.data[mst.data > 0.0]
    scale = float(np.median(positive)) if positive.size > 0 else 1.0
    scale = max(scale, 1e-12)
    mst.data = np.exp(-mst.data / scale)
    mst.data = np.clip(mst.data, 0.0, 1.0)
    mst.eliminate_zeros()
    return mst


def add_mst_edges(graph: sp.spmatrix, mst_graph: sp.spmatrix) -> sp.csr_matrix:
    """Merge MST edges into an existing symmetric graph.

    Args:
        graph:
            Base graph with shape ``[N, N]``.
        mst_graph:
            Symmetric MST graph with shape ``[N, N]``.

    Returns:
        CSR sparse matrix with shape ``[N, N]``.
    """

    combined = graph.tocsr().maximum(mst_graph.tocsr())
    combined.eliminate_zeros()
    return combined


def build_multiscale_knn_graph(X: ArrayLike2D, k_list: Sequence[int]) -> MultiScaleKnnGraph:
    """Build the full multi-scale fuzzy graph with MST enhancement.

    Args:
        X:
            PCA features with shape ``[N, p]``.
        k_list:
            Sequence of neighborhood sizes, e.g. ``[5, 10, 20]``.

    Returns:
        A :class:`MultiScaleKnnGraph` object.
    """

    if len(k_list) == 0:
        raise ValueError("k_list must not be empty")

    scale_graphs = {int(k): build_single_scale_graph(X, int(k)) for k in k_list}
    fused_graph = fuse_multiscale_graphs(scale_graphs)
    mst_graph = build_mst_graph(X)
    combined_graph = add_mst_edges(fused_graph, mst_graph)
    return MultiScaleKnnGraph(
        scale_graphs=scale_graphs,
        fused_graph=fused_graph,
        mst_graph=mst_graph,
        combined_graph=combined_graph,
    )


def build_knn_graph(features: ArrayLike2D, k: int) -> KnnGraph:
    """Compatibility wrapper for the single-scale graph builder.

    Args:
        features:
            Feature matrix with shape ``[N, p]``.
        k:
            Neighborhood size.

    Returns:
        A :class:`KnnGraph` object.
    """

    return build_single_scale_graph(features, k=k)
