"""Regression tests for the upgraded PD-CFD loss module."""

from __future__ import annotations

import unittest

import torch

from src.losses.pdcfd import pd_cfd_loss, wrapped_phase_difference


class PdCfdRegressionTestCase(unittest.TestCase):
    """Stricter tests for PD-CFD correctness and stability."""

    def setUp(self) -> None:
        torch.manual_seed(0)
        self.Y_ref = torch.randn(24, 4, dtype=torch.float32)
        low = 0.25 * torch.randn(4, 4, dtype=torch.float32)
        mid = 1.00 * torch.randn(4, 4, dtype=torch.float32)
        high = 3.00 * torch.randn(4, 4, dtype=torch.float32)
        self.freqs = torch.cat([low, mid, high], dim=0)

    def test_loss_grows_with_distribution_shift(self) -> None:
        identical = pd_cfd_loss(self.Y_ref, self.Y_ref.clone(), self.freqs, lambda_p=0.8, alpha=0.5)
        small_shift = pd_cfd_loss(
            self.Y_ref,
            self.Y_ref + 0.05 * torch.randn_like(self.Y_ref),
            self.freqs,
            lambda_p=0.8,
            alpha=0.5,
        )
        large_shift = pd_cfd_loss(
            self.Y_ref,
            self.Y_ref + 0.60,
            self.freqs,
            lambda_p=0.8,
            alpha=0.5,
        )

        self.assertTrue(torch.isfinite(identical.loss).item())
        self.assertTrue(torch.isfinite(small_shift.loss).item())
        self.assertTrue(torch.isfinite(large_shift.loss).item())
        self.assertLess(float(identical.loss.item()), 1e-6)
        self.assertGreater(float(small_shift.loss.item()), float(identical.loss.item()))
        self.assertGreater(float(large_shift.loss.item()), float(small_shift.loss.item()))

    def test_amplitude_and_phase_terms_are_finite(self) -> None:
        perturbed = pd_cfd_loss(
            self.Y_ref,
            self.Y_ref + 0.15 * torch.randn_like(self.Y_ref),
            self.freqs,
            lambda_p=1.0,
            alpha=0.7,
        )
        self.assertTrue(torch.isfinite(perturbed.per_freq_amplitude_error).all().item())
        self.assertTrue(torch.isfinite(perturbed.per_freq_phase_error).all().item())
        self.assertTrue(torch.isfinite(perturbed.total_per_freq_loss).all().item())
        self.assertTrue(torch.isfinite(perturbed.ecf_ref.real).all().item())
        self.assertTrue(torch.isfinite(perturbed.ecf_y.imag).all().item())

    def test_high_frequency_lambda_is_smaller_than_low_frequency_lambda(self) -> None:
        outputs = pd_cfd_loss(self.Y_ref, self.Y_ref + 0.1, self.freqs, lambda_p=1.0, alpha=1.0)
        norms = outputs.frequency_norms
        low_idx = int(torch.argmin(norms).item())
        high_idx = int(torch.argmax(norms).item())
        self.assertLess(float(outputs.lambda_phi[high_idx].item()), float(outputs.lambda_phi[low_idx].item()))

    def test_phase_wrap_output_is_finite(self) -> None:
        a = torch.tensor([3.13, -3.13, 0.5], dtype=torch.float32)
        b = torch.tensor([-3.13, 3.13, -0.5], dtype=torch.float32)
        wrapped = wrapped_phase_difference(a, b)
        self.assertEqual(tuple(wrapped.shape), (3,))
        self.assertTrue(torch.isfinite(wrapped).all().item())

    def test_output_shapes_and_aliases_are_consistent(self) -> None:
        Y = self.Y_ref[:9]
        outputs = pd_cfd_loss(self.Y_ref, Y, self.freqs, lambda_p=0.5, alpha=0.3)
        self.assertEqual(tuple(outputs.total_per_freq_loss.shape), (12,))
        self.assertEqual(tuple(outputs.per_freq_amplitude_error.shape), (12,))
        self.assertEqual(tuple(outputs.per_freq_phase_error.shape), (12,))
        self.assertEqual(tuple(outputs.lambda_phi.shape), (12,))
        self.assertEqual(tuple(outputs.phase_confidence.shape), (12,))
        self.assertEqual(tuple(outputs.raw_phase_difference.shape), (12,))
        self.assertTrue(torch.allclose(outputs.per_frequency_loss, outputs.total_per_freq_loss))
        self.assertTrue(torch.allclose(outputs.ref_cf.real, outputs.ecf_ref.real))
        self.assertTrue(torch.allclose(outputs.attenuation, outputs.lambda_phi))


if __name__ == "__main__":
    unittest.main()
