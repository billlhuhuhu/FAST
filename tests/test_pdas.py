"""AFL-focused tests for sampling utilities."""

from __future__ import annotations

import unittest

import torch

from src.sampling.anisotropic_freq import build_anisotropic_frequency_library


class AflTestCase(unittest.TestCase):
    """Lightweight tests for the upgraded AFL builder."""

    def setUp(self) -> None:
        torch.manual_seed(0)
        self.feature_dim = 6
        self.num_frequencies = 12
        self.max_norm = 6.0
        self.band_ranges = {
            'low': [0.0, 2.0],
            'medium': [2.0, 4.0],
            'high': [4.0, 6.0],
        }
        self.band_counts = {'low': 4, 'medium': 4, 'high': 4}
        self.Y_ref = torch.randn(32, self.feature_dim)
        self.Y_current = self.Y_ref + 0.25 * torch.randn_like(self.Y_ref)

    def test_afl_output_shapes_and_finite_values(self) -> None:
        library = build_anisotropic_frequency_library(
            feature_dim=self.feature_dim,
            num_frequencies=self.num_frequencies,
            max_norm=self.max_norm,
            Y_ref=self.Y_ref,
            Y_current=self.Y_current,
            band_ranges=self.band_ranges,
            band_sample_counts=self.band_counts,
            candidate_scales=[0.5, 1.0, 1.5],
        )
        self.assertEqual(tuple(library.omega.shape), (12, 6))
        self.assertEqual(tuple(library.norms.shape), (12,))
        self.assertEqual(tuple(library.band_ids.shape), (12,))
        self.assertTrue(torch.isfinite(library.omega).all().item())
        self.assertTrue(torch.isfinite(library.norms).all().item())

    def test_band_norm_ranges_match_config(self) -> None:
        library = build_anisotropic_frequency_library(
            feature_dim=self.feature_dim,
            num_frequencies=self.num_frequencies,
            max_norm=self.max_norm,
            Y_ref=self.Y_ref,
            Y_current=self.Y_current,
            band_ranges=self.band_ranges,
            band_sample_counts=self.band_counts,
            candidate_scales=[0.5, 1.0, 1.5],
        )
        for band_name, band_id in [('low', 0), ('medium', 1), ('high', 2)]:
            mask = library.band_ids == band_id
            band_norms = library.norms[mask]
            low, high = self.band_ranges[band_name]
            self.assertTrue(torch.all(band_norms >= max(low, 1e-6) - 1e-4).item())
            self.assertTrue(torch.all(band_norms <= high + 1e-4).item())

    def test_per_band_scaling_is_used(self) -> None:
        library = build_anisotropic_frequency_library(
            feature_dim=self.feature_dim,
            num_frequencies=self.num_frequencies,
            max_norm=self.max_norm,
            Y_ref=self.Y_ref,
            Y_current=self.Y_current,
            band_ranges=self.band_ranges,
            band_sample_counts=self.band_counts,
            candidate_scales=[0.5, 1.0, 1.5, 2.0],
        )
        low_scale = library.per_band_scaling['low']
        medium_scale = library.per_band_scaling['medium']
        high_scale = library.per_band_scaling['high']
        self.assertFalse(torch.allclose(low_scale, torch.ones_like(low_scale)))
        self.assertFalse(torch.allclose(low_scale, medium_scale) and torch.allclose(medium_scale, high_scale))

    def test_afl_distribution_differs_from_simple_sampling(self) -> None:
        afl = build_anisotropic_frequency_library(
            feature_dim=self.feature_dim,
            num_frequencies=self.num_frequencies,
            max_norm=self.max_norm,
            Y_ref=self.Y_ref,
            Y_current=self.Y_current,
            band_ranges=self.band_ranges,
            band_sample_counts=self.band_counts,
            candidate_scales=[0.5, 1.0, 1.5],
        )
        baseline = build_anisotropic_frequency_library(
            feature_dim=self.feature_dim,
            num_frequencies=self.num_frequencies,
            max_norm=self.max_norm,
            Y_ref=None,
            Y_current=None,
            band_ranges=self.band_ranges,
            band_sample_counts=self.band_counts,
        )
        afl_std = afl.omega.std(dim=0, unbiased=False)
        baseline_std = baseline.omega.std(dim=0, unbiased=False)
        diff = torch.mean(torch.abs(afl_std - baseline_std))
        self.assertGreater(float(diff.item()), 1e-2)


if __name__ == '__main__':
    unittest.main()


from src.sampling.pdas import select_progressive_frequencies


class PdasSelectionTestCase(unittest.TestCase):
    """Lightweight tests for discrepancy-aware progressive frequency selection."""

    def setUp(self) -> None:
        torch.manual_seed(1)
        self.feature_dim = 5
        self.Y_ref = torch.randn(24, self.feature_dim)
        self.Y = self.Y_ref[:6] + 0.15 * torch.randn(6, self.feature_dim)
        self.library = build_anisotropic_frequency_library(
            feature_dim=self.feature_dim,
            num_frequencies=15,
            max_norm=6.0,
            Y_ref=self.Y_ref,
            Y_current=self.Y,
            band_ranges={'low': [0.0, 2.0], 'medium': [2.0, 4.0], 'high': [4.0, 6.0]},
            band_sample_counts={'low': 5, 'medium': 5, 'high': 5},
            candidate_scales=[0.5, 1.0, 1.5],
        )
        self.config = {
            'pdas_frequencies_per_iter': 4,
            'tau_start_ratio': 0.2,
            'tau_end_ratio': 1.0,
            'diversity_beta': 4.0,
            'lcf_lambda_p': 0.5,
            'lcf_alpha': 0.5,
        }

    def test_candidate_pool_and_outputs_are_consistent(self) -> None:
        state = select_progressive_frequencies(
            library=self.library,
            step=0,
            total_steps=5,
            Y_ref=self.Y_ref,
            Y=self.Y,
            config=self.config,
        )
        self.assertEqual(state.selected_frequencies.ndim, 2)
        self.assertEqual(state.selected_indices.ndim, 1)
        self.assertEqual(tuple(state.per_freq_lcf.shape), tuple(state.per_freq_diversity.shape))
        self.assertEqual(tuple(state.per_freq_lcf.shape), tuple(state.final_scores.shape))
        self.assertTrue(torch.isfinite(state.per_freq_lcf).all().item())
        self.assertTrue(torch.isfinite(state.per_freq_diversity).all().item())
        self.assertTrue(torch.isfinite(state.final_scores).all().item())

    def test_tau_progressively_expands_candidate_pool(self) -> None:
        early = select_progressive_frequencies(
            library=self.library,
            step=0,
            total_steps=5,
            Y_ref=self.Y_ref,
            Y=self.Y,
            config=self.config,
        )
        late = select_progressive_frequencies(
            library=self.library,
            step=4,
            total_steps=5,
            Y_ref=self.Y_ref,
            Y=self.Y,
            config=self.config,
        )
        self.assertLessEqual(early.candidate_pool_stats['candidate_count'], late.candidate_pool_stats['candidate_count'])
        self.assertLessEqual(early.candidate_pool_stats['tau_t'], late.candidate_pool_stats['tau_t'])

    def test_diversity_affects_scores(self) -> None:
        state = select_progressive_frequencies(
            library=self.library,
            step=2,
            total_steps=5,
            Y_ref=self.Y_ref,
            Y=self.Y,
            config=self.config,
        )
        self.assertTrue(torch.any(state.per_freq_diversity < 1.0).item() or state.selected_indices.numel() == 1)
