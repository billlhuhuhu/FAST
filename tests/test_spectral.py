"""Minimal tests for the spectral graph module."""

from __future__ import annotations

import unittest

import numpy as np
import scipy.sparse as sp

from src.graph.knn_graph import build_multiscale_knn_graph
from src.graph.spectral import compute_symmetric_normalized_laplacian, spectral_decomposition


class SpectralTestCase(unittest.TestCase):
    """Small tests for normalized Laplacian and spectral embedding."""

    def test_laplacian_shape_and_symmetry(self) -> None:
        rng = np.random.default_rng(0)
        X = rng.normal(size=(30, 6)).astype(np.float64)
        graph = build_multiscale_knn_graph(X, k_list=[5, 10])
        L_sym = compute_symmetric_normalized_laplacian(graph.combined_graph)
        self.assertEqual(L_sym.shape, (30, 30))
        diff = L_sym - L_sym.T
        max_abs = 0.0 if diff.nnz == 0 else float(np.max(np.abs(diff.data)))
        self.assertLessEqual(max_abs, 1e-8)

    def test_spectral_embedding_shape(self) -> None:
        rng = np.random.default_rng(1)
        X = rng.normal(size=(24, 5)).astype(np.float64)
        graph = build_multiscale_knn_graph(X, k_list=[5, 10])
        result = spectral_decomposition(graph.combined_graph, d=4)
        self.assertEqual(result.laplacian.shape, (24, 24))
        self.assertEqual(result.eigenvectors.shape[0], 24)
        self.assertEqual(result.eigenvectors.shape[1], 4)
        self.assertEqual(result.eigenvalues.shape[0], 4)

    def test_skip_zero_eigenvalues_logic(self) -> None:
        block1 = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float64)
        block2 = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float64)
        B = sp.block_diag((block1, block2), format='csr')
        result = spectral_decomposition(B, d=1, zero_tol=1e-8, extra_eigs=3)
        self.assertEqual(result.eigenvectors.shape, (4, 1))
        self.assertTrue(np.all(result.eigenvalues > 1e-8))


if __name__ == "__main__":
    unittest.main()
