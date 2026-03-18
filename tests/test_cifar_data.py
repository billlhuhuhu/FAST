"""Lightweight tests for the CIFAR-10 data pipeline."""

from __future__ import annotations

import unittest
from pathlib import Path

from src.data.cifar import flatten_images, prepare_cifar10_data


class CifarDataTestCase(unittest.TestCase):
    """Minimal tests for loading, flattening, and PCA preprocessing."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.root = Path("./data")
        cls.train_num_samples = 64
        cls.test_num_samples = 16
        cls.pca_dim = 16
        cls.prepared = prepare_cifar10_data(
            root=cls.root,
            flatten=True,
            standardize=True,
            pca_dim=cls.pca_dim,
            train_num_samples=cls.train_num_samples,
            test_num_samples=cls.test_num_samples,
            download=True,
        )

    def test_cifar10_loads(self) -> None:
        self.assertEqual(tuple(self.prepared.train.images.shape), (64, 3, 32, 32))
        self.assertEqual(tuple(self.prepared.test.images.shape), (16, 3, 32, 32))
        self.assertEqual(tuple(self.prepared.train.labels.shape), (64,))
        self.assertEqual(tuple(self.prepared.test.labels.shape), (16,))

    def test_flatten_dimension(self) -> None:
        flattened = flatten_images(self.prepared.train.images)
        self.assertEqual(tuple(flattened.shape), (64, 3 * 32 * 32))

    def test_pca_shape(self) -> None:
        self.assertEqual(tuple(self.prepared.train.X.shape), (64, 16))
        self.assertEqual(tuple(self.prepared.test.X.shape), (16, 16))

    def test_feature_label_count_match(self) -> None:
        self.assertEqual(self.prepared.train.X.shape[0], self.prepared.train.labels.shape[0])
        self.assertEqual(self.prepared.test.X.shape[0], self.prepared.test.labels.shape[0])


if __name__ == "__main__":
    unittest.main()
