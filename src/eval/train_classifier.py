"""Minimal downstream training and evaluation utilities for FAST."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import CIFAR10


CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


@dataclass
class EpochTrainLog:
    """Summary of one training epoch."""

    epoch: int
    train_loss: float
    train_accuracy: float
    num_samples: int


@dataclass
class EvaluationLog:
    """Summary of one evaluation pass."""

    loss: float
    accuracy: float
    num_samples: int


@dataclass
class ClassifierRunResult:
    """Outputs of training a classifier on a selected subset."""

    train_logs: List[Dict[str, Any]]
    test_accuracy: float
    test_loss: float
    selected_indices: Tensor


@dataclass
class StrategyComparisonResult:
    """Outputs of comparing subset strategies."""

    keep_ratio: float
    results: Dict[str, Dict[str, float]]


class IndexedDataset(torch.utils.data.Dataset):
    """Dataset wrapper that returns ``(image, label, original_index)``."""

    def __init__(self, base_dataset: torch.utils.data.Dataset) -> None:
        self.base_dataset = base_dataset

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor, Tensor]:
        image, label = self.base_dataset[index]
        return image, torch.tensor(label, dtype=torch.long), torch.tensor(index, dtype=torch.long)


def get_cifar10_transforms() -> tuple[transforms.Compose, transforms.Compose]:
    """Return standard CIFAR-10 train/test transforms."""

    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ]
    )
    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ]
    )
    return train_transform, test_transform


def load_cifar10_datasets(root: str | Path = './data', download: bool = True) -> tuple[IndexedDataset, IndexedDataset]:
    """Load CIFAR-10 train and test datasets with standard transforms.

    Returns:
        ``(train_dataset, test_dataset)`` where each wrapped dataset returns:
        - image tensor with shape ``[3, 32, 32]``
        - label tensor with shape ``[]``
        - original index tensor with shape ``[]``
    """

    train_transform, test_transform = get_cifar10_transforms()
    train_dataset = CIFAR10(root=str(root), train=True, transform=train_transform, download=download)
    test_dataset = CIFAR10(root=str(root), train=False, transform=test_transform, download=download)
    return IndexedDataset(train_dataset), IndexedDataset(test_dataset)


def build_resnet18(num_classes: int = 10) -> nn.Module:
    """Create a CIFAR-friendly ResNet-18 classifier.

    The torchvision ResNet-18 stem is slightly adapted for 32x32 CIFAR input.
    """

    from torchvision.models import resnet18

    model = resnet18(num_classes=num_classes)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    return model


def build_backbone(backbone: str, num_classes: int = 10) -> nn.Module:
    """Build the requested backbone. Currently only ``resnet18`` is supported."""

    if backbone.lower() != 'resnet18':
        raise ValueError("Only resnet18 is supported in the first version")
    return build_resnet18(num_classes=num_classes)


def _to_long_tensor(indices: Sequence[int] | np.ndarray | Tensor) -> Tensor:
    """Convert indices to a 1D long tensor."""

    if isinstance(indices, Tensor):
        tensor = indices.detach().cpu().to(dtype=torch.long)
    else:
        tensor = torch.as_tensor(indices, dtype=torch.long)
    if tensor.ndim != 1:
        raise ValueError("selected_indices must be a 1D sequence")
    return tensor


def build_subset_dataloader(
    dataset: torch.utils.data.Dataset,
    selected_indices: Sequence[int] | np.ndarray | Tensor,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 0,
) -> DataLoader:
    """Create a subset dataloader from discrete indices."""

    indices = _to_long_tensor(selected_indices)
    subset = Subset(dataset, indices.tolist())
    return DataLoader(subset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)


def _compute_accuracy(logits: Tensor, labels: Tensor) -> float:
    """Compute batch accuracy."""

    preds = logits.argmax(dim=1)
    return float((preds == labels).float().mean().item())


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int = 0,
    max_batches: Optional[int] = None,
) -> EpochTrainLog:
    """Run one training epoch.

    Args:
        model:
            Network to train.
        dataloader:
            Training dataloader yielding ``(image, label, index)``.
        optimizer:
            PyTorch optimizer.
        device:
            Single-device training target.
        epoch:
            Epoch number for logging.
        max_batches:
            Optional cap for lightweight smoke tests.
    """

    model.train()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for batch_idx, (images, labels, _indices) in enumerate(dataloader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        images = images.to(device)
        labels = labels.to(device)

        logits = model(images)
        loss = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_size = int(labels.shape[0])
        total_loss += float(loss.detach().cpu().item()) * batch_size
        total_correct += int((logits.argmax(dim=1) == labels).sum().detach().cpu().item())
        total_samples += batch_size

    avg_loss = total_loss / max(total_samples, 1)
    avg_acc = total_correct / max(total_samples, 1)
    return EpochTrainLog(epoch=epoch, train_loss=avg_loss, train_accuracy=avg_acc, num_samples=total_samples)


def evaluate_classifier(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    max_batches: Optional[int] = None,
) -> EvaluationLog:
    """Evaluate a classifier on a dataloader."""

    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for batch_idx, (images, labels, _indices) in enumerate(dataloader):
            if max_batches is not None and batch_idx >= max_batches:
                break
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            loss = criterion(logits, labels)

            batch_size = int(labels.shape[0])
            total_loss += float(loss.detach().cpu().item()) * batch_size
            total_correct += int((logits.argmax(dim=1) == labels).sum().detach().cpu().item())
            total_samples += batch_size

    avg_loss = total_loss / max(total_samples, 1)
    avg_acc = total_correct / max(total_samples, 1)
    return EvaluationLog(loss=avg_loss, accuracy=avg_acc, num_samples=total_samples)


def train_classifier_on_subset(
    selected_indices: Sequence[int] | np.ndarray | Tensor,
    backbone: str = 'resnet18',
    root: str | Path = './data',
    epochs: int = 1,
    batch_size: int = 64,
    lr: float = 0.01,
    weight_decay: float = 5e-4,
    momentum: float = 0.9,
    device: Optional[str | torch.device] = None,
    download: bool = True,
    train_max_batches: Optional[int] = None,
    eval_max_batches: Optional[int] = None,
    num_workers: int = 0,
) -> ClassifierRunResult:
    """Train a classifier on a selected CIFAR-10 subset and evaluate on test.

    Args:
        selected_indices:
            1D subset indices into the CIFAR-10 train set.
        backbone:
            Backbone name. First version only supports ``resnet18``.
        root:
            Dataset root.
        epochs:
            Number of training epochs.
        batch_size:
            Batch size for train and test.
        lr:
            SGD learning rate.
        weight_decay:
            Optimizer weight decay.
        momentum:
            SGD momentum.
        device:
            Training device. Defaults to CUDA if available, else CPU.
        download:
            Whether torchvision may download the dataset.
        train_max_batches:
            Optional lightweight cap for smoke tests.
        eval_max_batches:
            Optional lightweight cap for smoke tests.
        num_workers:
            Dataloader workers.
    """

    selected_indices_tensor = _to_long_tensor(selected_indices)
    train_dataset, test_dataset = load_cifar10_datasets(root=root, download=download)
    if int(selected_indices_tensor.min().item()) < 0 or int(selected_indices_tensor.max().item()) >= len(train_dataset):
        raise ValueError('selected_indices are out of bounds for CIFAR-10 train set')

    train_loader = build_subset_dataloader(
        dataset=train_dataset,
        selected_indices=selected_indices_tensor,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    resolved_device = torch.device(device) if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = build_backbone(backbone=backbone, num_classes=10).to(resolved_device)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)

    train_logs: List[Dict[str, Any]] = []
    for epoch in range(epochs):
        epoch_log = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            device=resolved_device,
            epoch=epoch,
            max_batches=train_max_batches,
        )
        train_logs.append(
            {
                'epoch': epoch_log.epoch,
                'train_loss': epoch_log.train_loss,
                'train_accuracy': epoch_log.train_accuracy,
                'num_samples': epoch_log.num_samples,
            }
        )

    eval_log = evaluate_classifier(
        model=model,
        dataloader=test_loader,
        device=resolved_device,
        max_batches=eval_max_batches,
    )
    return ClassifierRunResult(
        train_logs=train_logs,
        test_accuracy=eval_log.accuracy,
        test_loss=eval_log.loss,
        selected_indices=selected_indices_tensor,
    )


def sample_random_subset(
    train_size: int,
    keep_ratio: float,
    seed: int = 0,
) -> Tensor:
    """Sample a random subset of training indices."""

    if keep_ratio <= 0.0 or keep_ratio > 1.0:
        raise ValueError('keep_ratio must satisfy 0 < keep_ratio <= 1')
    subset_size = max(1, int(round(train_size * keep_ratio)))
    generator = torch.Generator().manual_seed(seed)
    return torch.randperm(train_size, generator=generator)[:subset_size].to(dtype=torch.long)


def compare_subset_strategies(
    keep_ratio: float,
    fast_selected_indices: Sequence[int] | np.ndarray | Tensor,
    root: str | Path = './data',
    epochs: int = 1,
    batch_size: int = 64,
    seed: int = 0,
    device: Optional[str | torch.device] = None,
    download: bool = True,
    train_max_batches: Optional[int] = None,
    eval_max_batches: Optional[int] = None,
    num_workers: int = 0,
) -> StrategyComparisonResult:
    """Compare `random` and `FAST` subset strategies on CIFAR-10.

    Args:
        keep_ratio:
            Fraction of the train set to keep.
        fast_selected_indices:
            1D FAST-selected indices.
        root:
            Dataset root.
        epochs:
            Number of training epochs.
        batch_size:
            Batch size.
        seed:
            Random seed for the random baseline.
    """

    train_dataset, _test_dataset = load_cifar10_datasets(root=root, download=download)
    random_indices = sample_random_subset(train_size=len(train_dataset), keep_ratio=keep_ratio, seed=seed)
    fast_indices = _to_long_tensor(fast_selected_indices)

    random_result = train_classifier_on_subset(
        selected_indices=random_indices,
        backbone='resnet18',
        root=root,
        epochs=epochs,
        batch_size=batch_size,
        device=device,
        download=download,
        train_max_batches=train_max_batches,
        eval_max_batches=eval_max_batches,
        num_workers=num_workers,
    )
    fast_result = train_classifier_on_subset(
        selected_indices=fast_indices,
        backbone='resnet18',
        root=root,
        epochs=epochs,
        batch_size=batch_size,
        device=device,
        download=download,
        train_max_batches=train_max_batches,
        eval_max_batches=eval_max_batches,
        num_workers=num_workers,
    )

    results = {
        'random': {
            'test_accuracy': random_result.test_accuracy,
            'test_loss': random_result.test_loss,
        },
        'FAST': {
            'test_accuracy': fast_result.test_accuracy,
            'test_loss': fast_result.test_loss,
        },
    }
    return StrategyComparisonResult(keep_ratio=keep_ratio, results=results)
