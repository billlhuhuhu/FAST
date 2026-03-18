"""Smoke test for the first-pass FAST joint optimization loop."""

from __future__ import annotations

import unittest

import numpy as np
import scipy.sparse as sp
import torch

from src.optimize.optimize_coreset import optimize_coreset


class OptimizeJointSmokeTestCase(unittest.TestCase):
    """Small random-data smoke tests for joint optimization."""

    def test_joint_optimization_runs(self) -> None:
        torch.manual_seed(0)
        V_full = torch.randn(20, 6)
        diag = np.full(20, 2.0, dtype=np.float64)
        off = np.full(19, -1.0, dtype=np.float64)
        L_sym = sp.diags([off, diag, off], offsets=[-1, 0, 1], format='csr')

        config = {
            'keep_ratio': 0.25,
            'init_mode': 'random_subset',
            'iterations': 4,
            'lr': 1e-2,
            'lambda_match': 1.0,
            'lambda_graph': 0.05,
            'lambda_div': 0.01,
            'lambda_pdcfd': 1.0,
            'num_frequencies': 16,
            'max_frequency_norm': 4.0,
            'lambda_p': 0.5,
            'alpha': 0.5,
            'rff_dim': 16,
            'dpp_sigma': 1.0,
            'dpp_delta': 1e-5,
        }

        result = optimize_coreset(V_full=V_full, L_sym=L_sym, config=config)
        self.assertEqual(tuple(result.Y.shape), (5, 6))
        self.assertEqual(tuple(result.matched_indices.shape), (5,))
        self.assertIn('loss_total', result.logs)
        self.assertIn('loss_match', result.logs)
        self.assertIn('loss_graph', result.logs)
        self.assertIn('loss_div', result.logs)
        self.assertIn('loss_pdcfd', result.logs)
        self.assertEqual(len(result.logs['loss_total']), 4)


if __name__ == '__main__':
    unittest.main()
