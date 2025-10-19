"""Train ResNet20 on the CIFAR10-ST dataset.

The CIFAR10-ST subset should be generated via
`prepare_datasets.py cifar10-st --output-dir <path>` so that
`<path>/cifar10_st/train.npz` and `test.npz` exist. This script initialises a
ResNet20 model from scratch and optimises it with SGD + momentum as described in
the project specification.
"""
from __future__ import annotations

import argparse
import json
import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image


LOGGER = logging.getLogger("train_cifar10_st")
CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2470, 0.2435, 0.2616)


class CIFAR10NPZDataset(Dataset):
    """Simple dataset wrapper around the generated NPZ files."""

    def __init__(self, npz_path: Path, transform: transforms.Compose | None = None) -> None:
        if not npz_path.exists():
            raise FileNotFoundError(f"Missing dataset file: {npz_path}")
        data = np.load(npz_path)
        self.images = data["data"]
        self.targets = data["targets"]
        self.transform = transform

    def __len__(self) -> int:
        return int(self.targets.shape[0])

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        image = Image.fromarray(self.images[idx])
        if self.transform is not None:
            image = self.transform(image)
        target = int(self.targets[idx])
        return image, target


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes: int, planes: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNetCIFAR(nn.Module):
    """Minimal ResNet implementation for CIFAR inputs."""

    def __init__(self, block: type[BasicBlock], num_blocks: Tuple[int, int, int], num_classes: int = 10) -> None:
        super().__init__()
        self.in_planes = 16

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block: type[BasicBlock], planes: int, num_blocks: int, stride: int) -> nn.Sequential:
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out


@dataclass
class TrainConfig:
    data_root: Path
    output_dir: Path
    epochs: int = 90
    batch_size: int = 128
    learning_rate: float = 0.1
    momentum: float = 0.9
    weight_decay: float = 5e-4
    num_workers: int = 4
    seed: int = 42
    log_interval: int = 50


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def prepare_dataloaders(config: TrainConfig) -> Tuple[DataLoader, DataLoader]:
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
    eval_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])

    train_dataset = CIFAR10NPZDataset(config.data_root / "train.npz", transform=train_transform)
    test_dataset = CIFAR10NPZDataset(config.data_root / "test.npz", transform=eval_transform)

    pin_memory = torch.cuda.is_available()
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=pin_memory,
    )
    return train_loader, test_loader


def accuracy(outputs: torch.Tensor, targets: torch.Tensor) -> float:
    preds = outputs.argmax(dim=1)
    correct = (preds == targets).sum().item()
    return correct / targets.size(0)


def train_one_epoch(model: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer, device: torch.device, epoch: int, log_interval: int) -> Tuple[float, float]:
    model.train()
    running_loss = 0.0
    running_acc = 0.0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(loader, start=1):
        inputs = inputs.to(device, non_blocking=True)
        targets = torch.as_tensor(targets, device=device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = F.cross_entropy(outputs, targets)
        loss.backward()
        optimizer.step()

        batch_size = targets.size(0)
        running_loss += loss.item() * batch_size
        running_acc += accuracy(outputs, targets) * batch_size
        total += batch_size

        if batch_idx % log_interval == 0:
            LOGGER.info("Epoch %d | Step %d/%d | Loss %.4f", epoch, batch_idx, len(loader), loss.item())

    return running_loss / total, running_acc / total


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    total = 0

    for inputs, targets in loader:
        inputs = inputs.to(device, non_blocking=True)
        targets = torch.as_tensor(targets, device=device)
        outputs = model(inputs)
        loss = F.cross_entropy(outputs, targets)

        batch_size = targets.size(0)
        total_loss += loss.item() * batch_size
        total_acc += accuracy(outputs, targets) * batch_size
        total += batch_size

    return {
        "loss": total_loss / total,
        "accuracy": total_acc / total,
    }


def run_training(config: TrainConfig) -> None:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    (config.output_dir / "checkpoints").mkdir(exist_ok=True)

    set_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    LOGGER.info("Using device: %s", device)

    train_loader, test_loader = prepare_dataloaders(config)
    LOGGER.info("Loaded CIFAR10-ST | train=%d samples | test=%d samples", len(train_loader.dataset), len(test_loader.dataset))
    model = ResNetCIFAR(BasicBlock, (3, 3, 3), num_classes=10).to(device)

    optimizer = SGD(
        model.parameters(),
        lr=config.learning_rate,
        momentum=config.momentum,
        weight_decay=config.weight_decay,
        nesterov=False,
    )
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[60, 75], gamma=0.1)

    best_acc = 0.0
    history = []

    for epoch in range(1, config.epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, device, epoch, config.log_interval)
        metrics = evaluate(model, test_loader, device)
        scheduler.step()

        LOGGER.info(
            "Epoch %d complete | train_loss=%.4f train_acc=%.3f val_loss=%.4f val_acc=%.3f",
            epoch,
            train_loss,
            train_acc,
            metrics["loss"],
            metrics["accuracy"],
        )

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_acc,
            "val_loss": metrics["loss"],
            "val_accuracy": metrics["accuracy"],
            "lr": optimizer.param_groups[0]["lr"],
        })

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "history": history,
        }
        torch.save(checkpoint, config.output_dir / "checkpoints" / "last.pt")

        if metrics["accuracy"] > best_acc:
            best_acc = metrics["accuracy"]
            torch.save(checkpoint, config.output_dir / "checkpoints" / "best.pt")

    with (config.output_dir / "metrics.json").open("w", encoding="utf-8") as fh:
        json.dump({
            "best_val_accuracy": best_acc,
            "epochs": history,
        }, fh, indent=2)

    LOGGER.info("Training complete. Best validation accuracy: %.3f", best_acc)


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description="Train ResNet20 on CIFAR10-ST")
    parser.add_argument("--data-root", type=Path, required=True, help="Directory containing train.npz/test.npz")
    parser.add_argument("--output-dir", type=Path, required=True, help="Where to store checkpoints and metrics")
    parser.add_argument("--epochs", type=int, default=90, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=128, help="Mini-batch size")
    parser.add_argument("--lr", type=float, default=0.1, help="Initial learning rate")
    parser.add_argument("--momentum", type=float, default=0.9, help="Momentum for SGD")
    parser.add_argument("--weight-decay", type=float, default=5e-4, help="Weight decay")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader worker count")
    parser.add_argument("--log-interval", type=int, default=50, help="Steps between logging updates")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="[%(asctime)s] [%(levelname)s] %(message)s",
    )

    return TrainConfig(
        data_root=args.data_root,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        num_workers=args.num_workers,
        seed=args.seed,
        log_interval=args.log_interval,
    )


if __name__ == "__main__":
    cfg = parse_args()
    run_training(cfg)
