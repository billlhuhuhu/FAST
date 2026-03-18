"""CIFAR dataset helpers for the FAST reproduction scaffold.

This module intentionally keeps the implementation simple for the first stage.
Later stages can add transforms, caching, and richer feature representations.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
from torch import Tensor
from torch.utils.data import Dataset


@dataclass
class CifarBatch:
    """Container for a materialized CIFAR split.

    Attributes:
        images: Tensor of shape ``[N, C, H, W]``.
        labels: Tensor of shape ``[N]``.
        flat_images: Optional tensor of shape ``[N, D]`` when flattening is used.
    """

    images: Tensor
    labels: Tensor
    flat_images: Optional[Tensor] = None


def flatten_images(images: Tensor) -> Tensor:
    """Flatten image tensors into vectors.

    Args:
        images: Tensor with shape ``[N, C, H, W]``.

    Returns:
        Tensor with shape ``[N, C * H * W]``.
    """

    if images.ndim != 4:
        raise ValueError(f"Expected images with 4 dims [N, C, H, W], got {tuple(images.shape)}")
    return images.reshape(images.shape[0], -1)


def load_cifar10_batch(
    root: str | Path,
    train: bool = True,
    flatten: bool = True,
) -> CifarBatch:
    """Load CIFAR-10 data into memory.

    TODO:
        - Add normalization and augmentation hooks.
        - Add support for CIFAR-100 and other image datasets.
        - Optionally return only a debug subset.
    """

    from torchvision.datasets import CIFAR10

    dataset = CIFAR10(root=str(root), train=train, download=True)
    images = torch.tensor(dataset.data, dtype=torch.float32).permute(0, 3, 1, 2) / 255.0
    labels = torch.tensor(dataset.targets, dtype=torch.long)
    flat_images = flatten_images(images) if flatten else None
    return CifarBatch(images=images, labels=labels, flat_images=flat_images)


class IndexedTensorDataset(Dataset[tuple[Tensor, Tensor, Tensor]]):
    """Simple dataset that also returns sample indices.

    This is useful later for mapping optimized proxies back to discrete samples.
    """

    def __init__(self, images: Tensor, labels: Tensor) -> None:
        """Initialize the dataset.

        Args:
            images: Tensor of shape ``[N, C, H, W]``.
            labels: Tensor of shape ``[N]``.
        """

        if images.shape[0] != labels.shape[0]:
            raise ValueError("images and labels must share the same first dimension")
        self.images = images
        self.labels = labels

    def __len__(self) -> int:
        return int(self.labels.shape[0])

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor, Tensor]:
        """Return ``(image, label, index_tensor)`` for a single sample."""

        return self.images[index], self.labels[index], torch.tensor(index, dtype=torch.long)
