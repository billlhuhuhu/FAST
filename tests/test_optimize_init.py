"""Minimal tests for coreset-variable initialization."""

from __future__ import annotations

import unittest

import torch

from src.optimize.optimize_coreset import initialize_coreset_variable, resolve_num_coreset_points


class OptimizeInitTestCase(unittest.TestCase):
    """Lightweight tests for Y initialization."""

    def setUp(self) -> None:
        torch.manual_seed(0)
        self.V_full = torch.randn(50, 8)

    def test_shape_from_explicit_M(self) -> None:
        result = initialize_coreset_variable(self.V_full, M=7, init_mode="random_subset")
        self.assertEqual(tuple(result.Y.shape), (7, 8))
        self.assertEqual(tuple(result.init_indices.shape), (7,))
        self.assertEqual(result.M, 7)

    def test_keep_ratio_to_M_conversion(self) -> None:
        M = resolve_num_coreset_points(num_points=50, keep_ratio=0.1)
        self.assertEqual(M, 5)
        result = initialize_coreset_variable(self.V_full, keep_ratio=0.1, init_mode="random_subset")
        self.assertEqual(result.M, 5)
        self.assertEqual(tuple(result.Y.shape), (5, 8))

    def test_random_init_indices_unique_and_in_range(self) -> None:
        result = initialize_coreset_variable(self.V_full, M=10, init_mode="random_subset")
        indices = result.init_indices
        self.assertEqual(torch.unique(indices).numel(), 10)
        self.assertGreaterEqual(int(indices.min().item()), 0)
        self.assertLess(int(indices.max().item()), 50)

    def test_parameter_can_be_optimized(self) -> None:
        result = initialize_coreset_variable(self.V_full, M=6, init_mode="kmeans++")
        optimizer = torch.optim.Adam([result.Y], lr=1e-3)
        loss = (result.Y ** 2).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        self.assertIsNotNone(result.Y.grad)


if __name__ == "__main__":
    unittest.main()
