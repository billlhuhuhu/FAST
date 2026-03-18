"""Multi-scale kNN graph construction utilities for the FAST scaffold.

This module implements a correctness-first version of the graph construction
pipeline described around FAST Section 3.1:

1. build kNN neighborhoods for each ``k`` in ``k_list``;
2. estimate local connectivity offsets ``rho_i`` and smooth scales ``sigma_i``;
3. build a directed fuzzy graph per scale;
4. apply scale-wise fuzzy union ``A + A^T - A * A^T``;
5. fuse multiple scales with another fuzzy union pass;
6. add MST edges to improve connectivity;
7. expose graph statistics for debugging.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Sequence, Union

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components, minimum_spanning_tree
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
        stats:
            Graph statistics dictionary for this scale.
    """

    indices: np.ndarray
    distances: np.ndarray
    weights: np.ndarray
    adjacency: sp.csr_matrix
    rho: np.ndarray
    sigma: np.ndarray
    stats: Dict[str, float | int]


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
        stats:
            Statistics dictionary for fused/MST/final graph structure.
    """

    scale_graphs: Dict[int, KnnGraph]
    fused_graph: sp.csr_matrix
    mst_graph: sp.csr_matrix
    combined_graph: sp.csr_matrix
    stats: Dict[str, float | int]


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
        A tuple ``(indices, distances)`` where both arrays have shape ``[N, k_eff]``.
    """

    if k <= 0:
        raise ValueError("k must be positive")

    X_np = to_numpy_2d(X)
    num_nodes = X_np.shape[0]
    effective_k = min(k + 1, num_nodes)
    nbrs = NearestNeighbors(n_neighbors=effective_k, metric="euclidean", n_jobs=1)
    nbrs.fit(X_np)
    distances, indices = nbrs.kneighbors(X_np, return_distance=True)

    if effective_k > 1:
        distances = distances[:, 1:]
        indices = indices[:, 1:]
    else:
        distances = np.zeros((num_nodes, 0), dtype=np.float64)
        indices = np.zeros((num_nodes, 0), dtype=np.int64)

    return indices, distances


def compute_rho(distances: np.ndarray, zero_tol: float = 1e-12) -> np.ndarray:
    """Compute local connectivity offsets ``rho_i``.

    ``rho_i`` is the first non-zero neighbor distance for point ``i``. When all
    neighbor distances are effectively zero, the offset falls back to ``0``.

    Args:
        distances:
            Neighbor distances with shape ``[N, k]``.
        zero_tol:
            Threshold for treating a distance as zero.

    Returns:
        Array with shape ``[N]``.
    """

    if distances.ndim != 2:
        raise ValueError("distances must have shape [N, k]")

    rho = np.zeros(distances.shape[0], dtype=np.float64)
    for i in range(distances.shape[0]):
        row = np.asarray(distances[i], dtype=np.float64)
        positive = row[row > zero_tol]
        rho[i] = float(positive[0]) if positive.size > 0 else 0.0
    return rho


def _membership_sum_for_sigma(row_distances: np.ndarray, rho_i: float, sigma_value: float) -> float:
    """Compute the smooth-kNN membership sum for a single row and sigma."""

    adjusted = row_distances - rho_i
    if sigma_value <= 1e-12:
        return float(np.sum(adjusted <= 0.0))

    positive = np.maximum(adjusted, 0.0)
    memberships = np.exp(-positive / sigma_value)
    memberships[adjusted <= 0.0] = 1.0
    return float(np.sum(memberships))


def solve_sigmas(
    distances: np.ndarray,
    rho: np.ndarray,
    target: float | None = None,
    n_iter: int = 64,
    tol: float = 1e-5,
    min_sigma: float = 1e-3,
) -> np.ndarray:
    """Estimate local smoothing scales ``sigma_i`` using a stable binary search.

    The solver follows the smooth-kNN idea used in topology-aware fuzzy graphs:
    for each point ``i``, find ``sigma_i`` such that the local membership mass
    approximately matches ``target``.

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
        min_sigma:
            Lower bound used for numerical stability.

    Returns:
        Array with shape ``[N]``.
    """

    num_nodes, num_neighbors = distances.shape
    if rho.shape[0] != num_nodes:
        raise ValueError("rho must have shape [N]")
    if target is None:
        target = np.log2(max(num_neighbors, 1) + 1.0)

    positive_distances = distances[distances > 1e-12]
    global_scale = float(np.median(positive_distances)) if positive_distances.size > 0 else 1.0
    global_scale = max(global_scale, min_sigma)

    sigmas = np.zeros(num_nodes, dtype=np.float64)
    for i in range(num_nodes):
        row = np.asarray(distances[i], dtype=np.float64)
        if row.size == 0:
            sigmas[i] = global_scale
            continue

        lo = 0.0
        hi = max(global_scale, np.max(np.maximum(row - rho[i], 0.0)) + global_scale)
        hi = max(hi, min_sigma)

        current = _membership_sum_for_sigma(row, rho[i], hi)
        expand_steps = 0
        while current < target and expand_steps < 32:
            hi *= 2.0
            current = _membership_sum_for_sigma(row, rho[i], hi)
            expand_steps += 1

        for _ in range(n_iter):
            mid = 0.5 * (lo + hi)
            value = _membership_sum_for_sigma(row, rho[i], mid)
            if abs(value - target) <= tol:
                lo = mid
                hi = mid
                break
            if value < target:
                lo = mid
            else:
                hi = mid

        sigma_i = 0.5 * (lo + hi)
        if not np.isfinite(sigma_i) or sigma_i <= 0.0:
            sigma_i = global_scale
        sigmas[i] = max(float(sigma_i), min_sigma)

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
    if rho.shape[0] != num_nodes or sigma.shape[0] != num_nodes:
        raise ValueError("rho and sigma must both have shape [N]")

    weights = np.zeros((num_nodes, num_neighbors), dtype=np.float64)
    for i in range(num_nodes):
        adjusted = distances[i] - rho[i]
        positive = np.maximum(adjusted, 0.0)
        sigma_i = max(float(sigma[i]), 1e-12)
        row_weights = np.exp(-positive / sigma_i)
        row_weights[adjusted <= 0.0] = 1.0
        weights[i] = np.clip(row_weights, 0.0, 1.0)
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

    if indices.shape != weights.shape:
        raise ValueError("indices and weights must share shape [N, k]")

    row_ids = np.repeat(np.arange(num_nodes, dtype=np.int64), indices.shape[1])
    col_ids = indices.reshape(-1)
    data = weights.reshape(-1)
    adjacency = sp.csr_matrix((data, (row_ids, col_ids)), shape=(num_nodes, num_nodes))
    adjacency.data = np.clip(adjacency.data, 0.0, 1.0)
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

    adjacency = adjacency.tocsr().astype(np.float64)
    transpose = adjacency.T.tocsr()
    sym = adjacency + transpose - adjacency.multiply(transpose)
    sym = 0.5 * (sym + sym.T)
    sym = sym.tocsr()
    sym.data = np.clip(sym.data, 0.0, 1.0)
    sym.eliminate_zeros()
    return sym


def _count_undirected_edges(graph: sp.csr_matrix) -> int:
    """Count undirected edges of a symmetric sparse graph."""

    graph = graph.tocsr()
    off_diagonal = graph.tocoo()
    mask = off_diagonal.row < off_diagonal.col
    return int(np.sum(mask))


def compute_graph_statistics(graph: sp.spmatrix, mst_graph: sp.spmatrix | None = None) -> Dict[str, float | int]:
    """Compute lightweight graph statistics.

    Args:
        graph:
            Symmetric sparse graph with shape ``[N, N]``.
        mst_graph:
            Optional MST graph used to compute usage statistics.

    Returns:
        Dictionary with edge count, connected components and degree summaries.
    """

    csr = graph.tocsr().astype(np.float64)
    degree = np.asarray(csr.sum(axis=1)).reshape(-1)
    components, _ = connected_components(csr, directed=False)

    stats: Dict[str, float | int] = {
        'num_nodes': int(csr.shape[0]),
        'edge_count': _count_undirected_edges(csr),
        'connected_components': int(components),
        'degree_min': float(np.min(degree)) if degree.size > 0 else 0.0,
        'degree_max': float(np.max(degree)) if degree.size > 0 else 0.0,
        'degree_mean': float(np.mean(degree)) if degree.size > 0 else 0.0,
        'degree_std': float(np.std(degree)) if degree.size > 0 else 0.0,
    }

    if mst_graph is not None:
        mst_edges = max(_count_undirected_edges(mst_graph.tocsr()), 1)
        fused_without_mst = csr.minimum(mst_graph.tocsr())
        reused_edges = _count_undirected_edges(fused_without_mst)
        added_edges = mst_edges - reused_edges
        stats['mst_edge_count'] = int(mst_edges)
        stats['mst_reused_edge_count'] = int(reused_edges)
        stats['mst_added_edge_count'] = int(max(added_edges, 0))
        stats['mst_edge_usage_ratio'] = float(max(added_edges, 0) / mst_edges)

    return stats


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
    stats = compute_graph_statistics(adjacency)
    stats['k'] = int(k)
    return KnnGraph(
        indices=indices,
        distances=distances,
        weights=weights,
        adjacency=adjacency,
        rho=rho,
        sigma=sigma,
        stats=stats,
    )


def fuse_multiscale_graphs(scale_graphs: Dict[int, KnnGraph]) -> sp.csr_matrix:
    """Fuse multiple symmetric fuzzy graphs into one graph.

    The fusion is also done with a fuzzy union operator so that combining scales
    follows the same topology-preserving logic used inside each scale.
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



def _normalize_csr_for_csgraph(graph: sp.spmatrix) -> sp.csr_matrix:
    """Normalize CSR dtypes for SciPy csgraph routines.

    Some SciPy builds are strict about CSR index dtypes when calling graph
    algorithms such as ``minimum_spanning_tree``. This helper makes the matrix
    contiguous and casts indices/indptr to the integer dtype typically expected
    by ``scipy.sparse.csgraph`` on Linux wheels.

    Args:
        graph:
            Sparse matrix with shape ``[N, N]``.

    Returns:
        CSR matrix with ``float64`` data and normalized index arrays.
    """

    csr = graph.tocsr().astype(np.float64, copy=False)
    csr.sort_indices()
    csr.sum_duplicates()
    csr.data = np.ascontiguousarray(csr.data, dtype=np.float64)
    csr.indices = np.ascontiguousarray(csr.indices, dtype=np.int32)
    csr.indptr = np.ascontiguousarray(csr.indptr, dtype=np.int32)
    return csr


def build_sparse_distance_graph(scale_graphs: Dict[int, KnnGraph]) -> sp.csr_matrix:
    """Build a sparse symmetric distance graph from multi-scale kNN edges.

    Args:
        scale_graphs:
            Mapping ``k -> KnnGraph``. Each graph contributes neighbor indices and
            Euclidean distances with shape ``[N, k]``.

    Returns:
        Symmetric CSR distance graph with shape ``[N, N]``.
    """

    if not scale_graphs:
        raise ValueError("scale_graphs must not be empty")

    example_graph = next(iter(scale_graphs.values()))
    num_nodes = int(example_graph.indices.shape[0])
    row_parts: list[np.ndarray] = []
    col_parts: list[np.ndarray] = []
    data_parts: list[np.ndarray] = []

    for graph in scale_graphs.values():
        if graph.indices.shape != graph.distances.shape:
            raise ValueError("graph.indices and graph.distances must share shape [N, k]")
        row_ids = np.repeat(np.arange(num_nodes, dtype=np.int64), graph.indices.shape[1])
        row_parts.append(row_ids)
        col_parts.append(graph.indices.reshape(-1).astype(np.int64, copy=False))
        data_parts.append(graph.distances.reshape(-1).astype(np.float64, copy=False))

    rows = np.concatenate(row_parts, axis=0)
    cols = np.concatenate(col_parts, axis=0)
    data = np.concatenate(data_parts, axis=0)

    sym_rows = np.concatenate([rows, cols], axis=0)
    sym_cols = np.concatenate([cols, rows], axis=0)
    sym_data = np.concatenate([data, data], axis=0)

    sparse_graph = sp.coo_matrix((sym_data, (sym_rows, sym_cols)), shape=(num_nodes, num_nodes), dtype=np.float64)
    sparse_graph = sparse_graph.tocsr()
    sparse_graph.sum_duplicates()
    sparse_graph.eliminate_zeros()
    return sparse_graph


def build_mst_graph(
    X: ArrayLike2D,
    scale_graphs: Dict[int, KnnGraph] | None = None,
    dense_threshold: int = 4000,
) -> sp.csr_matrix:
    """Build a symmetric MST graph for connectivity enhancement.

    For small datasets, this uses the exact dense pairwise-distance graph.
    For larger datasets, it falls back to a sparse distance graph assembled from
    already-computed multi-scale kNN neighborhoods to avoid ``O(N^2)`` memory.

    Args:
        X:
            Feature matrix with shape ``[N, p]``.
        scale_graphs:
            Optional multi-scale kNN graphs used to assemble a sparse distance
            graph for large-scale runs.
        dense_threshold:
            Maximum node count for the exact dense MST path.

    Returns:
        Symmetric CSR sparse matrix with shape ``[N, N]`` whose nonzero entries
        correspond to MST edges converted into similarity-like weights.
    """

    X_np = to_numpy_2d(X)
    num_nodes = int(X_np.shape[0])

    if scale_graphs is not None and num_nodes > dense_threshold:
        distance_graph = build_sparse_distance_graph(scale_graphs)
        distance_graph = _normalize_csr_for_csgraph(distance_graph)
    else:
        distances = pairwise_distances(X_np, metric="euclidean", n_jobs=1)
        distance_graph = _normalize_csr_for_csgraph(sp.csr_matrix(distances, dtype=np.float64))

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
    combined = 0.5 * (combined + combined.T)
    combined = combined.tocsr()
    combined.data = np.clip(combined.data, 0.0, 1.0)
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
    mst_graph = build_mst_graph(X, scale_graphs=scale_graphs)
    combined_graph = add_mst_edges(fused_graph, mst_graph)

    stats = compute_graph_statistics(combined_graph, mst_graph=mst_graph)
    stats['fused_edge_count'] = _count_undirected_edges(fused_graph)
    stats['scale_count'] = int(len(scale_graphs))

    return MultiScaleKnnGraph(
        scale_graphs=scale_graphs,
        fused_graph=fused_graph,
        mst_graph=mst_graph,
        combined_graph=combined_graph,
        stats=stats,
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
