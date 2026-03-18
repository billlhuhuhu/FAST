"""Smoke tests for the FAST scaffold."""

from __future__ import annotations

import unittest

import torch

from src.graph.knn_graph import build_knn_graph
from src.losses.pdcfd import pd_cfd_loss
from src.optimize.optimize_coreset import optimize_coreset
from src.sampling.anisotropic_freq import build_anisotropic_frequency_library


class SmokeTestCase(unittest.TestCase):
    """Import- and shape-level tests for the minimal scaffold."""

    def test_knn_graph_shapes(self) -> None:
        features = torch.randn(8, 4)
        graph = build_knn_graph(features, k=3)
        self.assertEqual(tuple(graph.indices.shape), (8, 3))
        self.assertEqual(tuple(graph.distances.shape), (8, 3))
        self.assertEqual(tuple(graph.weights.shape), (8, 3))

    def test_pdcfd_shapes(self) -> None:
        reference = torch.randn(10, 5)
        proxies = torch.randn(4, 5)
        frequencies = torch.randn(6, 5)
        outputs = pd_cfd_loss(reference=reference, proxies=proxies, frequencies=frequencies)
        self.assertEqual(tuple(outputs.ref_cf.shape), (6,))
        self.assertEqual(tuple(outputs.proxy_cf.shape), (6,))
        self.assertEqual(outputs.loss.ndim, 0)

    def test_optimize_coreset_smoke(self) -> None:
        reference = torch.randn(12, 6)
        library = build_anisotropic_frequency_library(feature_dim=6, num_frequencies=16, max_norm=4.0)
        result = optimize_coreset(reference=reference, frequency_library=library, keep_ratio=0.25, num_steps=2, lr=1e-3)
        self.assertEqual(result.proxies.ndim, 2)
        self.assertEqual(result.subset_indices.ndim, 1)
        self.assertEqual(len(result.history), 2)


if __name__ == "__main__":
    unittest.main()
