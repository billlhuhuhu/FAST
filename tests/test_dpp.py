"""Minimal tests for the DPP loss module."""

from __future__ import annotations

import unittest

import torch

from src.losses.dpp import compute_dpp_loss


class DppTestCase(unittest.TestCase):
    """Small tests for the RFF-based DPP loss."""

    def setUp(self) -> None:
        torch.manual_seed(0)

    def test_loss_computes_and_is_finite(self) -> None:
        Y = torch.randn(8, 4)
        result = compute_dpp_loss(Y, rff_dim=32, sigma=1.0, delta=1e-6)
        self.assertEqual(result.loss.ndim, 0)
        self.assertTrue(torch.isfinite(result.loss).item())
        self.assertTrue(torch.isfinite(result.logdet).item())
        self.assertTrue(torch.isfinite(result.sign).item())

    def test_repeated_points_do_not_crash(self) -> None:
        base = torch.randn(1, 4)
        Y = base.repeat(8, 1)
        result = compute_dpp_loss(Y, rff_dim=32, sigma=1.0, delta=1e-5)
        self.assertTrue(torch.isfinite(result.loss).item())
        self.assertTrue(torch.isfinite(result.logdet).item())

    def test_kernel_has_stable_diagonal(self) -> None:
        Y = torch.randn(6, 3)
        result = compute_dpp_loss(Y, rff_dim=16, sigma=0.8, delta=1e-6)
        diagonal = torch.diag(result.kernel)
        self.assertTrue(torch.all(diagonal > 0).item())
        self.assertFalse(torch.isnan(result.kernel).any().item())
        self.assertFalse(torch.isinf(result.kernel).any().item())


if __name__ == "__main__":
    unittest.main()
