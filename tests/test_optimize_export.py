"""Minimal tests for final discrete subset export logic."""

from __future__ import annotations

import unittest

import torch

from src.optimize.optimize_coreset import export_selected_subset


class OptimizeExportTestCase(unittest.TestCase):
    """Small tests for final subset export and refill behavior."""

    def setUp(self) -> None:
        torch.manual_seed(0)
        self.V_full = torch.randn(10, 4)
        self.Y = self.V_full[:4].clone()

    def test_selected_indices_length_and_uniqueness(self) -> None:
        matched = torch.tensor([0, 1, 2, 3], dtype=torch.long)
        exported = export_selected_subset(matched_indices=matched, Y=self.Y, V_full=self.V_full, M=4)
        self.assertEqual(tuple(exported.selected_indices.shape), (4,))
        self.assertEqual(torch.unique(exported.selected_indices).numel(), 4)
        self.assertGreaterEqual(int(exported.selected_indices.min().item()), 0)
        self.assertLess(int(exported.selected_indices.max().item()), 10)

    def test_duplicate_matches_are_filled(self) -> None:
        matched = torch.tensor([0, 0, 1, 1], dtype=torch.long)
        exported = export_selected_subset(matched_indices=matched, Y=self.Y, V_full=self.V_full, M=4)
        self.assertEqual(tuple(exported.selected_indices.shape), (4,))
        self.assertEqual(torch.unique(exported.selected_indices).numel(), 4)
        self.assertEqual(exported.stats['unique_before_fill'], 2)
        self.assertEqual(exported.stats['filled_count'], 2)


if __name__ == '__main__':
    unittest.main()
