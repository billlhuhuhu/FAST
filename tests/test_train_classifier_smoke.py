"""Lightweight smoke test for classifier training on a small CIFAR subset."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.eval.train_classifier import build_backbone, sample_random_subset, train_classifier_on_subset


class TrainClassifierSmokeTestCase(unittest.TestCase):
    """Small end-to-end smoke test for dataloader, forward, backward, and save path."""

    def test_small_subset_training_smoke(self) -> None:
        selected_indices = sample_random_subset(train_size=50000, keep_ratio=64 / 50000.0, seed=0)
        with tempfile.TemporaryDirectory() as tmpdir:
            result = train_classifier_on_subset(
                selected_indices=selected_indices,
                backbone='resnet18',
                root='./data',
                epochs=1,
                batch_size=16,
                optimizer_name='sgd',
                scheduler_name='none',
                lr=0.01,
                momentum=0.9,
                weight_decay=5e-4,
                seed=7,
                download=True,
                train_max_batches=2,
                eval_max_batches=1,
                num_workers=0,
                output_dir=tmpdir,
                extra_config={'smoke_test': True},
            )
            result_path = Path(result.result_path)
            payload = json.loads(result_path.read_text(encoding='utf-8'))
        self.assertGreaterEqual(len(result.train_logs), 1)
        self.assertGreaterEqual(len(result.eval_logs), 1)
        self.assertIn('train_loss', result.train_logs[0])
        self.assertIn('train_accuracy', result.train_logs[0])
        self.assertIn('lr', result.train_logs[0])
        self.assertTrue(result.test_accuracy >= 0.0)
        self.assertTrue(result.test_accuracy <= 1.0)
        self.assertTrue(result.best_accuracy >= 0.0)
        self.assertIsNotNone(result.result_path)
        self.assertEqual(result.config_snapshot['seed'], 7)
        self.assertEqual(payload['config_snapshot']['optimizer'], 'sgd')
        self.assertEqual(payload['config_snapshot']['scheduler'], 'none')
        self.assertEqual(payload['config_snapshot']['epochs'], 1)
        self.assertEqual(payload['config_snapshot']['batch_size'], 16)

    def test_resnet50_build_smoke(self) -> None:
        model = build_backbone('resnet50', num_classes=10)
        self.assertEqual(model.fc.out_features, 10)


if __name__ == '__main__':
    unittest.main()
