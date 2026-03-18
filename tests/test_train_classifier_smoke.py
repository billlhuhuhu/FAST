"""Lightweight smoke test for classifier training on a small CIFAR subset."""

from __future__ import annotations

import unittest

from src.eval.train_classifier import sample_random_subset, train_classifier_on_subset


class TrainClassifierSmokeTestCase(unittest.TestCase):
    """Small end-to-end smoke test for dataloader, forward, and backward."""

    def test_small_subset_training_smoke(self) -> None:
        selected_indices = sample_random_subset(train_size=50000, keep_ratio=64 / 50000.0, seed=0)
        result = train_classifier_on_subset(
            selected_indices=selected_indices,
            backbone='resnet18',
            root='./data',
            epochs=1,
            batch_size=16,
            download=True,
            train_max_batches=2,
            eval_max_batches=1,
            num_workers=0,
        )
        self.assertGreaterEqual(len(result.train_logs), 1)
        self.assertIn('train_loss', result.train_logs[0])
        self.assertIn('train_accuracy', result.train_logs[0])
        self.assertTrue(result.test_accuracy >= 0.0)
        self.assertTrue(result.test_accuracy <= 1.0)


if __name__ == '__main__':
    unittest.main()
