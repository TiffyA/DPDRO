"""Dataset helpers for CIFAR10-ST and ImageNet."""
from __future__ import annotations

import tarfile
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import datasets, transforms


CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2470, 0.2435, 0.2616)
IMAGENET_MEAN = (0.4850, 0.4560, 0.4060)
IMAGENET_STD = (0.2290, 0.2240, 0.2250)


class CIFAR10STNPZ(Dataset):
    """Wrapper around the saved CIFAR10-ST NPZ files produced by prepare_datasets."""

    def __init__(self, path: Path, transform: transforms.Compose | None = None) -> None:
        if not path.exists():
            raise FileNotFoundError(f"Missing dataset file: {path}")
        data = np.load(path)
        self.images = data["data"]
        self.targets = data["targets"]
        self.transform = transform

    def __len__(self) -> int:  # type: ignore[override]
        return int(self.targets.shape[0])

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:  # type: ignore[override]
        image = Image.fromarray(self.images[idx])
        if self.transform is not None:
            image = self.transform(image)
        target = int(self.targets[idx])
        return image, target


def build_transforms(train: bool = True) -> transforms.Compose:
    if train:
        return transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
        ])
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])


def _safe_extract_inner_tar(inner: tarfile.TarFile, destination: Path) -> None:
    for member in inner.getmembers():
        if not member.isfile():
            continue
        name = Path(member.name).name
        if not name:
            continue
        destination.mkdir(parents=True, exist_ok=True)
        target = destination / name
        with inner.extractfile(member) as src:
            if src is None:
                continue
            with target.open("wb") as dst:
                dst.write(src.read())


def _maybe_materialise_imagenet_split(root: Path, split: str) -> bool:
    split_dir = root / split
    patterns = [
        f"*{split}.tar",
        f"*{split}.tar.gz",
        f"*{split}.tgz",
    ]
    archive: Path | None = None
    for pattern in patterns:
        matches = sorted(root.glob(pattern))
        if matches:
            archive = matches[0]
            break
    if archive is None and split == "train":
        candidates = sorted(root.glob("ILSVRC2010*_train.tar"))
        if candidates:
            archive = candidates[0]

    if split_dir.exists() and archive is None:
        return any(child.is_file() for child in split_dir.glob("*/*"))
    if archive is None:
        return False

    split_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "r:*") as outer:
        for member in outer.getmembers():
            if not member.isfile():
                continue
            member_name = Path(member.name)
            if member_name.suffix.lower() != ".tar":
                class_dir = split_dir / member_name.parent.name
                class_dir.mkdir(parents=True, exist_ok=True)
                with outer.extractfile(member) as src:
                    if src is None:
                        continue
                    target = class_dir / member_name.name
                    with target.open("wb") as dst:
                        dst.write(src.read())
                continue

            class_name = member_name.stem
            class_dir = split_dir / class_name
            if class_dir.exists() and any(class_dir.iterdir()):
                continue
            class_dir.mkdir(parents=True, exist_ok=True)
            with outer.extractfile(member) as inner_file:
                if inner_file is None:
                    continue
                with tarfile.open(fileobj=inner_file, mode="r:*") as inner_tar:
                    _safe_extract_inner_tar(inner_tar, class_dir)
    return True


class ImageNetFolder(Dataset):
    """Simple ImageNet loader based on torchvision.datasets.ImageFolder."""

    def __init__(self, root: Path, split: str, transform: transforms.Compose | None = None) -> None:
        split_dir = root / split
        if not split_dir.exists():
            prepared = _maybe_materialise_imagenet_split(root, split)
            if not prepared or not split_dir.exists():
                raise FileNotFoundError(f"Expected ImageNet split directory at {split_dir}")
        self.split = split
        self.dataset = datasets.ImageFolder(split_dir.as_posix(), transform=transform)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:  # type: ignore[override]
        return self.dataset[idx]

    @property
    def classes(self) -> Tuple[str, ...]:
        return tuple(self.dataset.classes)

    @property
    def num_classes(self) -> int:
        return len(self.dataset.classes)


def build_imagenet_transforms(train: bool = True, image_size: int = 224) -> transforms.Compose:
    if train:
        return transforms.Compose([
            transforms.RandomResizedCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    resize_size = int(round(image_size * 256 / 224))
    return transforms.Compose([
        transforms.Resize(resize_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
