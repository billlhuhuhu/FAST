"""Sparse spectral graph helpers for the FAST scaffold.

This module provides a correctness-first implementation of normalized graph
Laplacian construction and robust spectral decomposition on sparse graphs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh


@dataclass
class SpectralDecompositionResult:
    """Outputs of the spectral decomposition step.

    Attributes:
        laplacian:
            Symmetric normalized graph Laplacian with shape ``[N, N]`` in CSR format.
        eigenvalues:
            Selected nonzero eigenvalues with shape ``[d_effective]``.
        eigenvectors:
            Selected eigenvectors with shape ``[N, d_effective]``.
    """

    laplacian: sp.csr_matrix
    eigenvalues: np.ndarray
    eigenvectors: np.ndarray


def to_csr_symmetric(matrix: sp.spmatrix) -> sp.csr_matrix:
    """Convert a sparse matrix to a numerically symmetric CSR matrix.

    Args:
        matrix:
            Sparse matrix with shape ``[N, N]``.

    Returns:
        Symmetric CSR matrix with shape ``[N, N]``.
    """

    csr = matrix.tocsr().astype(np.float64)
    symmetric = 0.5 * (csr + csr.T)
    symmetric = symmetric.tocsr()
    symmetric.eliminate_zeros()
    return symmetric


def compute_degree_vector(B: sp.spmatrix) -> np.ndarray:
    """Compute the graph degree vector.

    Args:
        B:
            Sparse adjacency matrix with shape ``[N, N]``.

    Returns:
        Degree vector with shape ``[N]``.
    """

    csr = B.tocsr()
    return np.asarray(csr.sum(axis=1)).reshape(-1).astype(np.float64)


def compute_degree_matrix(B: sp.spmatrix) -> sp.csr_matrix:
    """Construct the sparse degree matrix ``D``.

    Args:
        B:
            Sparse adjacency matrix with shape ``[N, N]``.

    Returns:
        Sparse diagonal degree matrix with shape ``[N, N]``.
    """

    degree = compute_degree_vector(B)
    return sp.diags(degree, offsets=0, format="csr")


def compute_symmetric_normalized_laplacian(B: sp.spmatrix) -> sp.csr_matrix:
    """Construct the symmetric normalized Laplacian.

    The formula is:

    ``L_sym = I - D^{-1/2} B D^{-1/2}``

    Args:
        B:
            Sparse adjacency matrix with shape ``[N, N]``.

    Returns:
        Symmetric normalized Laplacian in CSR format with shape ``[N, N]``.

    Notes:
        Isolated nodes are handled by setting the corresponding inverse square
        root degree to zero.
    """

    B_sym = to_csr_symmetric(B)
    degree = compute_degree_vector(B_sym)
    inv_sqrt = np.zeros_like(degree)
    positive = degree > 0.0
    inv_sqrt[positive] = 1.0 / np.sqrt(degree[positive])
    D_inv_sqrt = sp.diags(inv_sqrt, offsets=0, format="csr")
    identity = sp.identity(B_sym.shape[0], format="csr", dtype=np.float64)
    L_sym = identity - D_inv_sqrt @ B_sym @ D_inv_sqrt
    L_sym = to_csr_symmetric(L_sym)
    return L_sym


def _dense_fallback_eigh(L_sym: sp.csr_matrix) -> Tuple[np.ndarray, np.ndarray]:
    """Dense fallback eigendecomposition for very small graphs or solver failures."""

    dense = L_sym.toarray()
    eigenvalues, eigenvectors = np.linalg.eigh(dense)
    return eigenvalues.astype(np.float64), eigenvectors.astype(np.float64)


def compute_smallest_eigenpairs(
    L_sym: sp.csr_matrix,
    num_pairs: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the smallest-magnitude eigenpairs of ``L_sym``.

    Args:
        L_sym:
            Symmetric normalized Laplacian with shape ``[N, N]``.
        num_pairs:
            Number of eigenpairs to request.

    Returns:
        ``(eigenvalues, eigenvectors)`` where:
        - ``eigenvalues`` has shape ``[m]``
        - ``eigenvectors`` has shape ``[N, m]``
    """

    num_nodes = L_sym.shape[0]
    if num_nodes == 0:
        raise ValueError("L_sym must have at least one node")

    max_pairs = max(1, num_nodes - 1)
    k = min(max(num_pairs, 1), max_pairs)

    if num_nodes <= 6 or k >= num_nodes - 1:
        return _dense_fallback_eigh(L_sym)

    try:
        eigenvalues, eigenvectors = eigsh(L_sym, k=k, which="SM")
        order = np.argsort(eigenvalues)
        return eigenvalues[order], eigenvectors[:, order]
    except Exception:
        return _dense_fallback_eigh(L_sym)


def select_nonzero_spectral_components(
    eigenvalues: np.ndarray,
    eigenvectors: np.ndarray,
    d: int,
    zero_tol: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray]:
    """Skip near-zero eigenvalues and keep the first ``d`` nonzero components.

    Args:
        eigenvalues:
            Eigenvalues with shape ``[m]``.
        eigenvectors:
            Eigenvectors with shape ``[N, m]``.
        d:
            Number of nonzero components to keep.
        zero_tol:
            Threshold below which eigenvalues are treated as zero.

    Returns:
        ``(selected_eigenvalues, V_full)`` where:
        - ``selected_eigenvalues`` has shape ``[d_effective]``
        - ``V_full`` has shape ``[N, d_effective]``
    """

    if d <= 0:
        raise ValueError("d must be positive")

    cleaned = np.asarray(eigenvalues, dtype=np.float64).copy()
    cleaned[np.abs(cleaned) < zero_tol] = 0.0
    cleaned[(cleaned < 0.0) & (cleaned > -zero_tol)] = 0.0

    keep_mask = cleaned > zero_tol
    kept_values = cleaned[keep_mask]
    kept_vectors = np.asarray(eigenvectors[:, keep_mask], dtype=np.float64)

    if kept_values.size == 0:
        return np.zeros((0,), dtype=np.float64), np.zeros((eigenvectors.shape[0], 0), dtype=np.float64)

    d_effective = min(d, kept_values.shape[0])
    return kept_values[:d_effective], kept_vectors[:, :d_effective]


def spectral_decomposition(
    B: sp.spmatrix,
    d: int,
    zero_tol: float = 1e-8,
    extra_eigs: int = 8,
) -> SpectralDecompositionResult:
    """Compute the normalized Laplacian and a nonzero spectral embedding.

    Args:
        B:
            Sparse adjacency matrix with shape ``[N, N]``.
        d:
            Number of nonzero eigenvectors to keep.
        zero_tol:
            Threshold used to skip near-zero eigenvalues.
        extra_eigs:
            Extra eigenpairs requested to make zero-eigenvalue skipping more robust
            on disconnected graphs.

    Returns:
        A :class:`SpectralDecompositionResult` with:
        - ``laplacian``: shape ``[N, N]``
        - ``eigenvalues``: shape ``[d_effective]``
        - ``eigenvectors`` / ``V_full``: shape ``[N, d_effective]``
    """

    if d <= 0:
        raise ValueError("d must be positive")

    L_sym = compute_symmetric_normalized_laplacian(B)
    num_nodes = L_sym.shape[0]
    requested_pairs = min(max(d + extra_eigs, d + 1), max(1, num_nodes - 1))
    raw_eigenvalues, raw_eigenvectors = compute_smallest_eigenpairs(L_sym, num_pairs=requested_pairs)
    selected_eigenvalues, V_full = select_nonzero_spectral_components(
        eigenvalues=raw_eigenvalues,
        eigenvectors=raw_eigenvectors,
        d=d,
        zero_tol=zero_tol,
    )
    return SpectralDecompositionResult(
        laplacian=L_sym,
        eigenvalues=selected_eigenvalues,
        eigenvectors=V_full,
    )
