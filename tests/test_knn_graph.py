"""Minimal tests for the kNN graph module."""

from __future__ import annotations

import unittest

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components

from src.graph.knn_graph import build_multiscale_knn_graph, build_single_scale_graph
from src.graph.spectral import spectral_decomposition


class KnnGraphTestCase(unittest.TestCase):
    """Small random-data tests for the graph builder."""

    def setUp(self) -> None:
        rng = np.random.default_rng(0)
        self.X = rng.normal(size=(24, 8)).astype(np.float64)

    def test_single_scale_graph_shape_nonnegative_and_finite(self) -> None:
        graph = build_single_scale_graph(self.X, k=5)
        self.assertEqual(graph.adjacency.shape, (24, 24))
        self.assertTrue(sp.isspmatrix_csr(graph.adjacency))
        self.assertTrue(np.all(graph.adjacency.data >= 0.0))
        self.assertFalse(np.isnan(graph.adjacency.data).any())
        self.assertFalse(np.isinf(graph.adjacency.data).any())
        self.assertFalse(np.isnan(graph.rho).any())
        self.assertFalse(np.isnan(graph.sigma).any())
        self.assertTrue(np.all(graph.sigma > 0.0))
        self.assertIn('edge_count', graph.stats)
        self.assertIn('connected_components', graph.stats)

    def test_fuzzy_union_output_is_symmetric(self) -> None:
        graph = build_single_scale_graph(self.X, k=10)
        diff = graph.adjacency - graph.adjacency.T
        max_abs = 0.0 if diff.nnz == 0 else float(np.max(np.abs(diff.data)))
        self.assertLessEqual(max_abs, 1e-8)

    def test_multiscale_graph_stats_and_connectivity_are_usable(self) -> None:
        graph = build_multiscale_knn_graph(self.X, k_list=[5, 10, 20])
        self.assertTrue(sp.isspmatrix_csr(graph.fused_graph))
        self.assertTrue(sp.isspmatrix_csr(graph.mst_graph))
        self.assertTrue(sp.isspmatrix_csr(graph.combined_graph))
        self.assertEqual(graph.combined_graph.shape, (24, 24))
        self.assertTrue(np.all(graph.combined_graph.data >= 0.0))
        self.assertFalse(np.isnan(graph.combined_graph.data).any())
        self.assertFalse(np.isinf(graph.combined_graph.data).any())
        diff = graph.combined_graph - graph.combined_graph.T
        max_abs = 0.0 if diff.nnz == 0 else float(np.max(np.abs(diff.data)))
        self.assertLessEqual(max_abs, 1e-8)
        n_components, _ = connected_components(graph.combined_graph, directed=False)
        self.assertEqual(n_components, 1)
        self.assertEqual(int(graph.stats['connected_components']), 1)
        self.assertGreaterEqual(int(graph.stats['edge_count']), 23)
        self.assertGreaterEqual(float(graph.stats['mst_edge_usage_ratio']), 0.0)
        self.assertLessEqual(float(graph.stats['mst_edge_usage_ratio']), 1.0)

    def test_smallest_nonzero_eigenvectors_run_on_constructed_graph(self) -> None:
        graph = build_multiscale_knn_graph(self.X, k_list=[5, 10, 20])
        result = spectral_decomposition(graph.combined_graph, d=4)
        self.assertEqual(result.laplacian.shape, (24, 24))
        self.assertEqual(result.eigenvectors.shape, (24, 4))
        self.assertEqual(result.eigenvalues.shape, (4,))
        self.assertTrue(np.all(np.isfinite(result.eigenvalues)))
        self.assertTrue(np.all(result.eigenvalues > 0.0))


if __name__ == "__main__":
    unittest.main()
