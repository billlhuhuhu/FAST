"""Minimal tests for the PD-CFD loss module."""

from __future__ import annotations

import unittest

import torch

from src.losses.pdcfd import pd_cfd_loss, wrapped_phase_difference


class PdCfdTestCase(unittest.TestCase):
    """Small tests for PD-CFD correctness and stability."""

    def setUp(self) -> None:
        torch.manual_seed(0)
        self.Y_ref = torch.randn(12, 4, dtype=torch.float32)
        self.freqs = torch.randn(7, 4, dtype=torch.float32)

    def test_loss_is_near_zero_when_inputs_match(self) -> None:
        outputs = pd_cfd_loss(self.Y_ref, self.Y_ref.clone(), self.freqs, lambda_p=1.0, alpha=1.0)
        self.assertTrue(torch.isfinite(outputs.loss).item())
        self.assertLess(float(outputs.loss.item()), 1e-6)

    def test_loss_increases_under_perturbation(self) -> None:
        identical = pd_cfd_loss(self.Y_ref, self.Y_ref.clone(), self.freqs, lambda_p=1.0, alpha=1.0)
        perturbed_Y = self.Y_ref + 0.2 * torch.randn_like(self.Y_ref)
        perturbed = pd_cfd_loss(self.Y_ref, perturbed_Y, self.freqs, lambda_p=1.0, alpha=1.0)
        self.assertGreater(float(perturbed.loss.item()), float(identical.loss.item()))

    def test_phase_wrap_output_is_finite(self) -> None:
        a = torch.tensor([3.13, -3.13, 0.5], dtype=torch.float32)
        b = torch.tensor([-3.13, 3.13, -0.5], dtype=torch.float32)
        wrapped = wrapped_phase_difference(a, b)
        self.assertEqual(tuple(wrapped.shape), (3,))
        self.assertTrue(torch.isfinite(wrapped).all().item())

    def test_frequency_output_shapes(self) -> None:
        Y = self.Y_ref[:5]
        outputs = pd_cfd_loss(self.Y_ref, Y, self.freqs, lambda_p=0.5, alpha=0.3)
        self.assertEqual(tuple(outputs.per_frequency_loss.shape), (7,))
        self.assertEqual(tuple(outputs.amplitude_diff.shape), (7,))
        self.assertEqual(tuple(outputs.phase_diff.shape), (7,))
        self.assertEqual(tuple(outputs.attenuation.shape), (7,))
        self.assertTrue(torch.isfinite(outputs.per_frequency_loss).all().item())
        self.assertTrue(torch.isfinite(outputs.phase_diff).all().item())


if __name__ == "__main__":
    unittest.main()
