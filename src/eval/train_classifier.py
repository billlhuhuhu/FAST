"""Downstream training and evaluation utilities for FAST on CIFAR-10.

This module remains intentionally lightweight, but it now exposes the core
configuration surface needed to move beyond smoke tests toward the paper-style
Section 4.1 evaluation loop:

- configurable seed
- configurable optimizer hyperparameters
- configurable scheduler
- configurable epochs / batch size
- backbone selection (ResNet-18 / ResNet-50)
- repeat-friendly result saving
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import CIFAR10

from src.utils.io import ensure_dir
from src.utils.seed import set_seed


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
    eval_logs: List[Dict[str, Any]]
    test_accuracy: float
    test_loss: float
    best_accuracy: float
    best_epoch: int
    train_loss_summary: Dict[str, float]
    config_snapshot: Dict[str, Any]
    selected_indices: Tensor
    result_path: Optional[str]


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
    """Load CIFAR-10 train and test datasets with standard transforms."""

    train_transform, test_transform = get_cifar10_transforms()
    train_dataset = CIFAR10(root=str(root), train=True, transform=train_transform, download=download)
    test_dataset = CIFAR10(root=str(root), train=False, transform=test_transform, download=download)
    return IndexedDataset(train_dataset), IndexedDataset(test_dataset)


def _adapt_resnet_for_cifar(model: nn.Module) -> nn.Module:
    """Apply a CIFAR-friendly stem to a torchvision ResNet."""

    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    return model


def build_resnet18(num_classes: int = 10) -> nn.Module:
    """Create a CIFAR-friendly ResNet-18 classifier."""

    from torchvision.models import resnet18

    model = resnet18(num_classes=num_classes)
    return _adapt_resnet_for_cifar(model)


def build_resnet50(num_classes: int = 10) -> nn.Module:
    """Create a CIFAR-friendly ResNet-50 classifier."""

    from torchvision.models import resnet50

    model = resnet50(num_classes=num_classes)
    return _adapt_resnet_for_cifar(model)


def build_backbone(backbone: str, num_classes: int = 10) -> nn.Module:
    """Build the requested backbone."""

    backbone_name = backbone.lower()
    if backbone_name == 'resnet18':
        return build_resnet18(num_classes=num_classes)
    if backbone_name == 'resnet50':
        return build_resnet50(num_classes=num_classes)
    raise ValueError("Supported backbones are: resnet18, resnet50")


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


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int = 0,
    max_batches: Optional[int] = None,
) -> EpochTrainLog:
    """Run one training epoch."""

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


def build_optimizer(
    model: nn.Module,
    optimizer_name: str = 'sgd',
    lr: float = 0.1,
    momentum: float = 0.9,
    weight_decay: float = 5e-4,
) -> torch.optim.Optimizer:
    """Build the requested optimizer from a minimal config surface."""

    name = optimizer_name.lower()
    if name == 'sgd':
        return torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    if name == 'adam':
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    raise ValueError('Supported optimizers are: sgd, adam')


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    scheduler_name: str = 'none',
    epochs: int = 200,
    milestones: Optional[Sequence[int]] = None,
    gamma: float = 0.1,
) -> Optional[object]:
    """Build a lightweight learning-rate scheduler."""

    name = scheduler_name.lower()
    if name == 'none':
        return None
    if name == 'multistep':
        resolved_milestones = [100, 150] if milestones is None else [int(step) for step in milestones]
        return torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=resolved_milestones, gamma=float(gamma))
    if name == 'cosine':
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(int(epochs), 1))
    raise ValueError('Supported schedulers are: none, multistep, cosine')


def _build_result_snapshot(
    backbone: str,
    optimizer_name: str,
    scheduler_name: str,
    epochs: int,
    batch_size: int,
    lr: float,
    momentum: float,
    weight_decay: float,
    seed: int,
    device: str,
    selected_count: int,
    optimizer_config: Dict[str, Any],
    scheduler_config: Dict[str, Any],
    extra_config: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """Build a JSON-friendly training config snapshot."""

    snapshot = {
        'backbone': backbone,
        'optimizer': optimizer_name,
        'scheduler': scheduler_name,
        'epochs': int(epochs),
        'batch_size': int(batch_size),
        'lr': float(lr),
        'momentum': float(momentum),
        'weight_decay': float(weight_decay),
        'seed': int(seed),
        'device': device,
        'selected_count': int(selected_count),
        'optimizer_config': optimizer_config,
        'scheduler_config': scheduler_config,
    }
    if extra_config is not None:
        snapshot['extra_config'] = extra_config
    return snapshot


def save_classifier_run_result(result: ClassifierRunResult, output_dir: str | Path) -> Path:
    """Save classifier run artifacts to a directory."""

    output_dir = ensure_dir(output_dir)
    result_path = output_dir / 'classifier_result.json'
    payload = {
        'train_logs': result.train_logs,
        'eval_logs': result.eval_logs,
        'test_accuracy': result.test_accuracy,
        'test_loss': result.test_loss,
        'best_accuracy': result.best_accuracy,
        'best_epoch': result.best_epoch,
        'train_loss_summary': result.train_loss_summary,
        'config_snapshot': result.config_snapshot,
        'selected_count': int(result.selected_indices.shape[0]),
    }
    result_path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding='utf-8')
    np.save(output_dir / 'selected_indices.npy', result.selected_indices.detach().cpu().numpy().astype(np.int64))
    return result_path


def train_classifier_on_subset(
    selected_indices: Sequence[int] | np.ndarray | Tensor,
    backbone: str = 'resnet18',
    root: str | Path = './data',
    epochs: int = 1,
    batch_size: int = 64,
    optimizer_name: str = 'sgd',
    scheduler_name: str = 'none',
    lr: float = 0.01,
    weight_decay: float = 5e-4,
    momentum: float = 0.9,
    scheduler_milestones: Optional[Sequence[int]] = None,
    scheduler_gamma: float = 0.1,
    seed: int = 42,
    device: Optional[str | torch.device] = None,
    download: bool = True,
    train_max_batches: Optional[int] = None,
    eval_max_batches: Optional[int] = None,
    num_workers: int = 0,
    output_dir: Optional[str | Path] = None,
    extra_config: Optional[Dict[str, Any]] = None,
) -> ClassifierRunResult:
    """Train a classifier on a selected CIFAR-10 subset and evaluate on test."""

    set_seed(seed)
    selected_indices_tensor = _to_long_tensor(selected_indices)
    train_dataset, test_dataset = load_cifar10_datasets(root=root, download=download)
    if selected_indices_tensor.numel() == 0:
        raise ValueError('selected_indices must not be empty')
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
    optimizer = build_optimizer(
        model=model,
        optimizer_name=optimizer_name,
        lr=lr,
        momentum=momentum,
        weight_decay=weight_decay,
    )
    scheduler = build_scheduler(
        optimizer=optimizer,
        scheduler_name=scheduler_name,
        epochs=epochs,
        milestones=scheduler_milestones,
        gamma=scheduler_gamma,
    )

    optimizer_config = {
        'name': optimizer_name,
        'lr': float(lr),
        'momentum': float(momentum),
        'weight_decay': float(weight_decay),
    }
    scheduler_config = {
        'name': scheduler_name,
        'milestones': None if scheduler_milestones is None else [int(step) for step in scheduler_milestones],
        'gamma': float(scheduler_gamma),
    }
    config_snapshot = _build_result_snapshot(
        backbone=backbone,
        optimizer_name=optimizer_name,
        scheduler_name=scheduler_name,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        momentum=momentum,
        weight_decay=weight_decay,
        seed=seed,
        device=str(resolved_device),
        selected_count=int(selected_indices_tensor.shape[0]),
        optimizer_config=optimizer_config,
        scheduler_config=scheduler_config,
        extra_config=extra_config,
    )

    train_logs: List[Dict[str, Any]] = []
    eval_logs: List[Dict[str, Any]] = []
    best_accuracy = -1.0
    best_epoch = -1

    for epoch in range(epochs):
        epoch_log = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            device=resolved_device,
            epoch=epoch,
            max_batches=train_max_batches,
        )
        eval_log = evaluate_classifier(
            model=model,
            dataloader=test_loader,
            device=resolved_device,
            max_batches=eval_max_batches,
        )
        current_lr = float(optimizer.param_groups[0]['lr'])
        train_logs.append({**asdict(epoch_log), 'lr': current_lr})
        eval_logs.append({'epoch': epoch, **asdict(eval_log), 'lr': current_lr})

        if eval_log.accuracy > best_accuracy:
            best_accuracy = float(eval_log.accuracy)
            best_epoch = int(epoch)

        if scheduler is not None:
            scheduler.step()

    train_losses = [float(item['train_loss']) for item in train_logs]
    train_loss_summary = {
        'first': float(train_losses[0]) if train_losses else 0.0,
        'last': float(train_losses[-1]) if train_losses else 0.0,
        'min': float(min(train_losses)) if train_losses else 0.0,
        'max': float(max(train_losses)) if train_losses else 0.0,
    }

    result = ClassifierRunResult(
        train_logs=train_logs,
        eval_logs=eval_logs,
        test_accuracy=float(eval_logs[-1]['accuracy']) if eval_logs else 0.0,
        test_loss=float(eval_logs[-1]['loss']) if eval_logs else 0.0,
        best_accuracy=best_accuracy if best_accuracy >= 0.0 else 0.0,
        best_epoch=best_epoch,
        train_loss_summary=train_loss_summary,
        config_snapshot=config_snapshot,
        selected_indices=selected_indices_tensor,
        result_path=None,
    )

    if output_dir is not None:
        result_path = save_classifier_run_result(result, output_dir)
        result.result_path = str(result_path)

    return result


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
    backbone: str = 'resnet18',
    optimizer_name: str = 'sgd',
    scheduler_name: str = 'none',
    lr: float = 0.01,
    momentum: float = 0.9,
    weight_decay: float = 5e-4,
    scheduler_milestones: Optional[Sequence[int]] = None,
    scheduler_gamma: float = 0.1,
    device: Optional[str | torch.device] = None,
    download: bool = True,
    train_max_batches: Optional[int] = None,
    eval_max_batches: Optional[int] = None,
    num_workers: int = 0,
) -> StrategyComparisonResult:
    """Compare `random` and `FAST` subset strategies on CIFAR-10."""

    train_dataset, _test_dataset = load_cifar10_datasets(root=root, download=download)
    random_indices = sample_random_subset(train_size=len(train_dataset), keep_ratio=keep_ratio, seed=seed)
    fast_indices = _to_long_tensor(fast_selected_indices)

    random_result = train_classifier_on_subset(
        selected_indices=random_indices,
        backbone=backbone,
        root=root,
        epochs=epochs,
        batch_size=batch_size,
        optimizer_name=optimizer_name,
        scheduler_name=scheduler_name,
        lr=lr,
        momentum=momentum,
        weight_decay=weight_decay,
        scheduler_milestones=scheduler_milestones,
        scheduler_gamma=scheduler_gamma,
        seed=seed,
        device=device,
        download=download,
        train_max_batches=train_max_batches,
        eval_max_batches=eval_max_batches,
        num_workers=num_workers,
    )
    fast_result = train_classifier_on_subset(
        selected_indices=fast_indices,
        backbone=backbone,
        root=root,
        epochs=epochs,
        batch_size=batch_size,
        optimizer_name=optimizer_name,
        scheduler_name=scheduler_name,
        lr=lr,
        momentum=momentum,
        weight_decay=weight_decay,
        scheduler_milestones=scheduler_milestones,
        scheduler_gamma=scheduler_gamma,
        seed=seed,
        device=device,
        download=download,
        train_max_batches=train_max_batches,
        eval_max_batches=eval_max_batches,
        num_workers=num_workers,
    )

    results = {
        'random': {
            'test_accuracy': random_result.test_accuracy,
            'best_accuracy': random_result.best_accuracy,
            'test_loss': random_result.test_loss,
        },
        'FAST': {
            'test_accuracy': fast_result.test_accuracy,
            'best_accuracy': fast_result.best_accuracy,
            'test_loss': fast_result.test_loss,
        },
    }
    return StrategyComparisonResult(keep_ratio=keep_ratio, results=results)
