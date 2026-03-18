"""Minimal tests for assignment and graph losses."""

from __future__ import annotations

import unittest

import numpy as np
import scipy.sparse as sp
import torch

from src.graph.assign import hungarian_match
from src.losses.graph_losses import compute_graph_loss, compute_match_loss


class AssignmentAndGraphLossesTestCase(unittest.TestCase):
    """Small random-data tests for matching and graph losses."""

    def setUp(self) -> None:
        torch.manual_seed(0)
        self.Y = torch.randn(5, 4)
        self.V_full = torch.randn(12, 4)
        self.V_full[:5] = self.Y + 0.02 * torch.randn_like(self.Y)
        degree = torch.arange(1, 13, dtype=torch.float32)
        self.degree = degree
        diag = np.full(12, 2.0, dtype=np.float64)
        off = np.full(11, -1.0, dtype=np.float64)
        self.L_sym = sp.diags([off, diag, off], offsets=[-1, 0, 1], format='csr')

    def test_matched_indices_shape_and_bounds(self) -> None:
        result = hungarian_match(self.Y, self.V_full, degree=self.degree, mode='full')
        self.assertEqual(tuple(result.matched_indices.shape), (5,))
        self.assertGreaterEqual(int(result.matched_indices.min().item()), 0)
        self.assertLess(int(result.matched_indices.max().item()), 12)

    def test_match_loss_is_finite(self) -> None:
        result = hungarian_match(self.Y, self.V_full, degree=self.degree, mode='full')
        loss = compute_match_loss(self.Y, self.V_full, result.matched_indices)
        self.assertEqual(loss.ndim, 0)
        self.assertTrue(torch.isfinite(loss).item())

    def test_graph_loss_is_finite(self) -> None:
        result = hungarian_match(self.Y, self.V_full, degree=self.degree, mode='full')
        loss = compute_graph_loss(self.Y, self.L_sym, result.matched_indices)
        self.assertEqual(loss.ndim, 0)
        self.assertTrue(torch.isfinite(loss).item())

    def test_matching_cost_is_finite(self) -> None:
        result = hungarian_match(self.Y, self.V_full, degree=self.degree, mode='full')
        self.assertTrue(np.isfinite(result.matching_cost))

    def test_full_and_pruned_modes_are_consistent(self) -> None:
        full = hungarian_match(self.Y, self.V_full, degree=self.degree, mode='full')
        pruned = hungarian_match(self.Y, self.V_full, degree=self.degree, mode='pruned', prune_topk=4)

        print(
            '\nassignment_perf:',
            {
                'full_build_sec': round(full.candidate_stats['cost_build_time_sec'], 6),
                'full_match_sec': round(full.candidate_stats['matching_time_sec'], 6),
                'pruned_build_sec': round(pruned.candidate_stats['cost_build_time_sec'], 6),
                'pruned_match_sec': round(pruned.candidate_stats['matching_time_sec'], 6),
            },
        )

        self.assertEqual(tuple(full.matched_indices.shape), tuple(pruned.matched_indices.shape))
        self.assertTrue(((pruned.matched_indices >= 0) & (pruned.matched_indices < self.V_full.shape[0])).all().item())
        self.assertEqual(pruned.mode, 'pruned')
        self.assertLessEqual(abs(pruned.matching_cost - full.matching_cost), 1e-3)
        self.assertTrue(np.isfinite(pruned.candidate_stats['cost_build_time_sec']))
        self.assertTrue(np.isfinite(pruned.candidate_stats['matching_time_sec']))


if __name__ == "__main__":
    unittest.main()
