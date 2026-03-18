"""CIFAR-10 data utilities for the FAST reproduction project.

This module provides a minimal but practical data pipeline for the first
reproduction stage:

1. load CIFAR-10 train/test splits,
2. optionally flatten images into vectors,
3. optionally standardize features using train-set statistics,
4. optionally apply PCA on train features and transform test features,
5. return original images, labels, and graph-building features ``X``.

The implementation is intentionally focused on clarity and shape safety.
It does not contain any training code.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from torch import Tensor
from torch.utils.data import Dataset
from torchvision.datasets import CIFAR10


@dataclass
class CifarSplit:
    """A single CIFAR split with original images, labels, and graph features.

    Attributes:
        images:
            Tensor with shape ``[N, C, H, W]`` and dtype ``float32`` in ``[0, 1]``.
        labels:
            Tensor with shape ``[N]`` and dtype ``long``.
        X:
            Tensor with shape ``[N, D]`` used for graph construction, PCA, and
            later coreset selection.
    """

    images: Tensor
    labels: Tensor
    X: Tensor


@dataclass
class CifarPreparedData:
    """Prepared CIFAR-10 train/test data.

    Attributes:
        train:
            Train split container.
        test:
            Test split container.
        scaler:
            Fitted ``StandardScaler`` used on the feature vectors, or ``None``.
        pca:
            Fitted ``PCA`` model used on the standardized features, or ``None``.
    """

    train: CifarSplit
    test: CifarSplit
    scaler: Optional[StandardScaler] = None
    pca: Optional[PCA] = None


def flatten_images(images: Tensor) -> Tensor:
    """Flatten images from ``[N, C, H, W]`` to ``[N, C * H * W]``.

    Args:
        images:
            Tensor with shape ``[N, C, H, W]``.

    Returns:
        Tensor with shape ``[N, D]`` where ``D = C * H * W``.
    """

    if images.ndim != 4:
        raise ValueError(f"Expected images with shape [N, C, H, W], got {tuple(images.shape)}")
    return images.reshape(images.shape[0], -1)


def _slice_num_samples(
    images: Tensor,
    labels: Tensor,
    num_samples: Optional[int],
) -> tuple[Tensor, Tensor]:
    """Optionally keep only the first ``num_samples`` examples.

    Args:
        images:
            Tensor with shape ``[N, C, H, W]``.
        labels:
            Tensor with shape ``[N]``.
        num_samples:
            Number of examples to keep, or ``None`` for all examples.

    Returns:
        A tuple ``(images, labels)`` with matching leading dimension.
    """

    if num_samples is None:
        return images, labels
    if num_samples <= 0:
        raise ValueError("num_samples must be positive when provided")
    return images[:num_samples], labels[:num_samples]


def load_cifar10_split(
    root: str | Path,
    train: bool,
    flatten: bool = True,
    num_samples: Optional[int] = None,
    download: bool = True,
) -> CifarSplit:
    """Load one CIFAR-10 split and return initial graph features.

    Args:
        root:
            Dataset root directory.
        train:
            ``True`` for the train split, ``False`` for the test split.
        flatten:
            Whether to flatten the original images into feature vectors.
            If ``False``, the returned feature tensor ``X`` still falls back to
            a flattened representation because later graph modules expect
            2D features.
        num_samples:
            Optional number of examples to keep for lightweight debugging/tests.
        download:
            Whether torchvision may download the dataset if not present.

    Returns:
        A :class:`CifarSplit` where:
        - ``images`` has shape ``[N, 3, 32, 32]``,
        - ``labels`` has shape ``[N]``,
        - ``X`` has shape ``[N, D]``.
    """

    dataset = CIFAR10(root=str(root), train=train, download=download)
    images = torch.tensor(dataset.data, dtype=torch.float32).permute(0, 3, 1, 2) / 255.0
    labels = torch.tensor(dataset.targets, dtype=torch.long)
    images, labels = _slice_num_samples(images, labels, num_samples=num_samples)

    if flatten:
        X = flatten_images(images)
    else:
        X = flatten_images(images)

    return CifarSplit(images=images, labels=labels, X=X)


def standardize_train_test_features(
    train_X: Tensor,
    test_X: Tensor,
) -> tuple[Tensor, Tensor, StandardScaler]:
    """Standardize train/test features using train-set statistics only.

    Args:
        train_X:
            Train feature tensor with shape ``[N_train, D]``.
        test_X:
            Test feature tensor with shape ``[N_test, D]``.

    Returns:
        ``(train_X_std, test_X_std, scaler)`` where the feature tensors keep the
        same shapes as the inputs.
    """

    scaler = StandardScaler()
    train_np = train_X.detach().cpu().numpy()
    test_np = test_X.detach().cpu().numpy()
    train_std = scaler.fit_transform(train_np)
    test_std = scaler.transform(test_np)
    train_tensor = torch.from_numpy(train_std.astype(np.float32))
    test_tensor = torch.from_numpy(test_std.astype(np.float32))
    return train_tensor, test_tensor, scaler


def apply_pca_train_test(
    train_X: Tensor,
    test_X: Tensor,
    pca_dim: int,
) -> tuple[Tensor, Tensor, PCA]:
    """Fit PCA on train features and transform both train and test features.

    Args:
        train_X:
            Train feature tensor with shape ``[N_train, D]``.
        test_X:
            Test feature tensor with shape ``[N_test, D]``.
        pca_dim:
            Target PCA dimension.

    Returns:
        ``(train_X_pca, test_X_pca, pca)`` where:
        - ``train_X_pca`` has shape ``[N_train, pca_dim_effective]``,
        - ``test_X_pca`` has shape ``[N_test, pca_dim_effective]``.

    Notes:
        ``pca_dim_effective`` is clipped to the valid PCA limit
        ``min(N_train, D)`` to keep the function robust on tiny test subsets.
    """

    if pca_dim <= 0:
        raise ValueError("pca_dim must be positive")

    train_np = train_X.detach().cpu().numpy()
    test_np = test_X.detach().cpu().numpy()
    max_components = min(train_np.shape[0], train_np.shape[1])
    effective_dim = min(pca_dim, max_components)
    pca = PCA(n_components=effective_dim, svd_solver="auto", random_state=0)
    train_pca = pca.fit_transform(train_np)
    test_pca = pca.transform(test_np)
    train_tensor = torch.from_numpy(train_pca.astype(np.float32))
    test_tensor = torch.from_numpy(test_pca.astype(np.float32))
    return train_tensor, test_tensor, pca


def prepare_cifar10_data(
    root: str | Path,
    flatten: bool = True,
    standardize: bool = True,
    pca_dim: Optional[int] = None,
    train_num_samples: Optional[int] = None,
    test_num_samples: Optional[int] = None,
    download: bool = True,
) -> CifarPreparedData:
    """Prepare CIFAR-10 train/test data for graph construction modules.

    Args:
        root:
            Dataset root directory.
        flatten:
            Whether to flatten images before feature processing.
        standardize:
            Whether to standardize train/test feature vectors using train-set
            statistics.
        pca_dim:
            Optional PCA target dimension. If ``None``, PCA is skipped.
        train_num_samples:
            Optional number of train examples to keep for debugging/tests.
        test_num_samples:
            Optional number of test examples to keep for debugging/tests.
        download:
            Whether torchvision may download the dataset if not present.

    Returns:
        A :class:`CifarPreparedData` object containing:
        - train/test original image tensors,
        - train/test labels,
        - train/test graph features ``X``.
    """

    train_split = load_cifar10_split(
        root=root,
        train=True,
        flatten=flatten,
        num_samples=train_num_samples,
        download=download,
    )
    test_split = load_cifar10_split(
        root=root,
        train=False,
        flatten=flatten,
        num_samples=test_num_samples,
        download=download,
    )

    train_X = train_split.X
    test_X = test_split.X
    scaler: Optional[StandardScaler] = None
    pca: Optional[PCA] = None

    if standardize:
        train_X, test_X, scaler = standardize_train_test_features(train_X, test_X)

    if pca_dim is not None:
        train_X, test_X, pca = apply_pca_train_test(train_X, test_X, pca_dim=pca_dim)

    train_prepared = CifarSplit(images=train_split.images, labels=train_split.labels, X=train_X)
    test_prepared = CifarSplit(images=test_split.images, labels=test_split.labels, X=test_X)
    return CifarPreparedData(train=train_prepared, test=test_prepared, scaler=scaler, pca=pca)


class IndexedTensorDataset(Dataset[tuple[Tensor, Tensor, Tensor]]):
    """Dataset wrapper that returns ``(image, label, index)``.

    Args:
        images:
            Tensor with shape ``[N, C, H, W]``.
        labels:
            Tensor with shape ``[N]``.
    """

    def __init__(self, images: Tensor, labels: Tensor) -> None:
        if images.shape[0] != labels.shape[0]:
            raise ValueError("images and labels must share the same first dimension")
        self.images = images
        self.labels = labels

    def __len__(self) -> int:
        return int(self.labels.shape[0])

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor, Tensor]:
        return self.images[index], self.labels[index], torch.tensor(index, dtype=torch.long)


def _self_check() -> None:
    """Run a tiny self-check for the data pipeline.

    This function is intentionally lightweight. It loads only a small subset and
    prints the dataset size and PCA output shape for quick inspection.
    """

    prepared = prepare_cifar10_data(
        root="./data",
        flatten=True,
        standardize=True,
        pca_dim=32,
        train_num_samples=128,
        test_num_samples=32,
        download=True,
    )
    print(f"train_size={prepared.train.labels.shape[0]}")
    print(f"train_pca_shape={tuple(prepared.train.X.shape)}")


if __name__ == "__main__":
    _self_check()
