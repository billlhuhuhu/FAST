"""Minimal tests for the kNN graph module."""

from __future__ import annotations

import unittest

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components

from src.graph.knn_graph import build_multiscale_knn_graph, build_single_scale_graph


class KnnGraphTestCase(unittest.TestCase):
    """Small random-data tests for the graph builder."""

    def setUp(self) -> None:
        rng = np.random.default_rng(0)
        self.X = rng.normal(size=(24, 8)).astype(np.float64)

    def test_single_scale_graph_shape_and_nonnegative(self) -> None:
        graph = build_single_scale_graph(self.X, k=5)
        self.assertEqual(graph.adjacency.shape, (24, 24))
        self.assertTrue(sp.isspmatrix_csr(graph.adjacency))
        self.assertTrue(np.all(graph.adjacency.data >= 0.0))
        self.assertFalse(np.isnan(graph.adjacency.data).any())
        self.assertFalse(np.isinf(graph.adjacency.data).any())

    def test_fuzzy_union_output_is_symmetric(self) -> None:
        graph = build_single_scale_graph(self.X, k=10)
        diff = graph.adjacency - graph.adjacency.T
        self.assertEqual(diff.nnz, 0)

    def test_multiscale_graph_and_mst_are_usable(self) -> None:
        graph = build_multiscale_knn_graph(self.X, k_list=[5, 10, 20])
        self.assertTrue(sp.isspmatrix_csr(graph.fused_graph))
        self.assertTrue(sp.isspmatrix_csr(graph.mst_graph))
        self.assertTrue(sp.isspmatrix_csr(graph.combined_graph))
        self.assertEqual(graph.combined_graph.shape, (24, 24))
        self.assertTrue(np.all(graph.combined_graph.data >= 0.0))
        self.assertFalse(np.isnan(graph.combined_graph.data).any())
        self.assertFalse(np.isinf(graph.combined_graph.data).any())
        diff = graph.combined_graph - graph.combined_graph.T
        self.assertEqual(diff.nnz, 0)
        n_components, _ = connected_components(graph.combined_graph, directed=False)
        self.assertEqual(n_components, 1)


if __name__ == "__main__":
    unittest.main()
