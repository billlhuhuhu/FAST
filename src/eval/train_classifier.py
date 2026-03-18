"""Minimal downstream training entry points for FAST evaluation."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, Subset


@dataclass
class TrainResult:
    """Small training result bundle."""

    train_loss: float
    num_samples: int


def build_resnet18(num_classes: int) -> nn.Module:
    """Create a torchvision ResNet-18 classifier.

    TODO:
        - Add support for more backbones used in the paper.
        - Make input resolution handling configurable.
    """

    from torchvision.models import resnet18

    model = resnet18(num_classes=num_classes)
    return model


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> TrainResult:
    """Run a single minimal training epoch.

    This is only a smoke-level implementation for now.
    """

    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    total_loss = 0.0
    total_samples = 0
    for images, labels, _indices in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        logits = model(images)
        loss = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_size = int(labels.shape[0])
        total_loss += float(loss.detach().cpu().item()) * batch_size
        total_samples += batch_size

    average_loss = total_loss / max(total_samples, 1)
    return TrainResult(train_loss=average_loss, num_samples=total_samples)


def build_subset_dataloader(
    dataset: torch.utils.data.Dataset,
    subset_indices: Tensor,
    batch_size: int,
) -> DataLoader:
    """Create a dataloader from selected discrete indices."""

    subset = Subset(dataset, subset_indices.detach().cpu().tolist())
    return DataLoader(subset, batch_size=batch_size, shuffle=True)
