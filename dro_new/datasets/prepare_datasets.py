"""Prepare CIFAR10-ST, ImageNet-LT, and iNaturalist2018 datasets.

This script centralises dataset downloads and construction steps needed by the
DRO experiments. CIFAR10-ST is materialised locally by sub-sampling the CIFAR10
training set. ImageNet-LT and iNaturalist2018 are constructed by combining
public metadata with user-provided raw data directories; the script can either
emit manifests pointing to the raw files or materialise symlinks/hardlinks.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys
import tarfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Tuple
from urllib.error import URLError
from urllib.request import urlopen


LOGGER = logging.getLogger("dataset_preparer")
DEFAULT_CACHE = Path.home() / ".cache" / "dro_new" / "raw"

# Known metadata mirrors. These may change over time; override via CLI if they do.
IMAGENET_LT_METADATA_CANDIDATES = [
    "https://raw.githubusercontent.com/ssvision/long-tailed-recognition-pytorch/master/data/ImageNet_LT/{name}",
    "https://raw.githubusercontent.com/csjliang/long-tailed-recognition-master/master/data/ImageNet_LT/{name}",
]
INAT2018_METADATA_CANDIDATES = {
    "train": [
        "https://storage.googleapis.com/lila-datasets/inaturalist2018/train2018.json",
    ],
    "val": [
        "https://storage.googleapis.com/lila-datasets/inaturalist2018/val2018.json",
    ],
    "categories": [
        "https://storage.googleapis.com/lila-datasets/inaturalist2018/categories.json",
    ],
}

INAT2018_IMAGE_ARCHIVE_URL = "https://ml-inat-competition-datasets.s3.amazonaws.com/2018/train_val_images.tar.gz"


class DatasetPreparationError(RuntimeError):
    """Raised when dataset preparation fails."""


@dataclass
class LinkOptions:
    link_type: str = "symlink"
    manifest_only: bool = False

    def __post_init__(self) -> None:
        valid = {"symlink", "hardlink", "copy"}
        if self.link_type not in valid:
            raise ValueError(f"Unsupported link_type '{self.link_type}' (choose from {valid})")


def _mkdir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _download_stream(url: str, destination: Path, chunk_size: int = 1 << 20) -> None:
    LOGGER.info("Downloading %s -> %s", url, destination)
    _mkdir(destination.parent)
    try:
        with urlopen(url) as response, destination.open("wb") as fh:
            while True:
                chunk = response.read(chunk_size)
                if not chunk:
                    break
                fh.write(chunk)
    except URLError as exc:
        if destination.exists():
            destination.unlink(missing_ok=True)
        raise DatasetPreparationError(f"Failed to download {url}: {exc}") from exc


def _ensure_download(destination: Path, urls: Iterable[str]) -> Path:
    if destination.exists():
        return destination
    last_error: Optional[Exception] = None
    for url in urls:
        try:
            _download_stream(url, destination)
            return destination
        except DatasetPreparationError as exc:
            last_error = exc
            LOGGER.warning("Download failed from %s: %s", url, exc)
    raise DatasetPreparationError(
        f"Could not download {destination.name}; supply it manually. Last error: {last_error}"
    )


def _create_link(src: Path, dst: Path, link_type: str) -> None:
    if not src.exists():
        raise FileNotFoundError(src)
    if dst.exists():
        return
    _mkdir(dst.parent)
    if link_type == "copy":
        shutil.copy2(src, dst)
    elif link_type == "hardlink":
        os.link(src, dst)
    else:
        relative = os.path.relpath(src, dst.parent)
        os.symlink(relative, dst)


def _write_json(path: Path, payload: dict) -> None:
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)


def _safe_extract_all(tar: tarfile.TarFile, path: Path) -> None:
    path = Path(path).resolve()
    for member in tar.getmembers():
        member_path = path / member.name
        try:
            member_path.resolve().relative_to(path)
        except ValueError as exc:
            raise DatasetPreparationError(f"Unsafe path in archive: {member.name}") from exc
    tar.extractall(path)


def _find_inat_split_dirs(base: Path) -> Tuple[Optional[Path], Optional[Path]]:
    train_dir = None
    val_dir = None
    for root, dirs, _ in os.walk(base):
        root_path = Path(root)
        for d in dirs:
            if d == "train2018":
                candidate = root_path / d
                if candidate.is_dir():
                    train_dir = candidate
            elif d == "val2018":
                candidate = root_path / d
                if candidate.is_dir():
                    val_dir = candidate
        if train_dir and val_dir:
            break
    return train_dir, val_dir


def _ensure_inat_images(inat_root: Path, cache_dir: Path, auto_download: bool, archive_url: str) -> Path:
    if inat_root.exists() and any(inat_root.iterdir()):
        return inat_root
    if not auto_download:
        raise DatasetPreparationError(
            f"iNaturalist root {inat_root} does not exist. Pass --auto-download or supply the directory manually."
        )

    cache_root = _mkdir(cache_dir / "inat2018")
    archive_path = cache_root / Path(archive_url).name
    LOGGER.info("Downloading iNaturalist2018 archive to %s", archive_path)
    _ensure_download(archive_path, [archive_url])

    extract_root = _mkdir(cache_root / "extracted")
    marker = extract_root / "_extracted.marker"
    if not marker.exists():
        LOGGER.info("Extracting %s (this may take a while)", archive_path)
        with tarfile.open(archive_path, "r:gz") as tar:
            _safe_extract_all(tar, extract_root)
        marker.touch()

    train_dir, val_dir = _find_inat_split_dirs(extract_root)
    if not train_dir or not val_dir:
        raise DatasetPreparationError(
            f"Could not locate train2018/val2018 directories after extracting {archive_path}"
        )

    destination_root = _mkdir(inat_root)
    for src_dir in (train_dir, val_dir):
        target = destination_root / src_dir.name
        if target.exists():
            continue
        LOGGER.info("Moving %s -> %s", src_dir, target)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src_dir), str(target))

    return destination_root


def prepare_cifar10_st(cache_dir: Path, output_dir: Path) -> None:
    try:
        from torchvision import datasets
    except ImportError as exc:
        raise DatasetPreparationError("torchvision is required for CIFAR10-ST preparation") from exc

    import numpy as np

    cache_dir = _mkdir(cache_dir.expanduser())
    output_dir = _mkdir(output_dir.expanduser())
    LOGGER.info("Preparing CIFAR10-ST in %s", output_dir)
    train = datasets.CIFAR10(root=str(cache_dir), train=True, download=True)
    test = datasets.CIFAR10(root=str(cache_dir), train=False, download=True)

    train_targets = np.array(train.targets)
    selected_indices: List[int] = []
    for cls in range(10):
        cls_indices = np.flatnonzero(train_targets == cls)
        if cls < 5:
            cls_indices = cls_indices[-100:]
        selected_indices.extend(cls_indices.tolist())
    selected_indices = np.array(sorted(selected_indices))
    new_train_data = train.data[selected_indices]
    new_train_targets = train_targets[selected_indices]

    _mkdir(output_dir / "cifar10_st")
    train_path = output_dir / "cifar10_st" / "train.npz"
    test_path = output_dir / "cifar10_st" / "test.npz"
    np.savez_compressed(train_path, data=new_train_data, targets=new_train_targets)
    np.savez_compressed(test_path, data=test.data, targets=np.array(test.targets))

    class_counts = {str(cls): int((new_train_targets == cls).sum()) for cls in range(10)}
    metadata = {
        "train_samples": int(new_train_targets.size),
        "test_samples": int(len(test.targets)),
        "class_counts": class_counts,
        "description": "CIFAR10-ST keeps last 100 samples for classes 0-4 and all others.",
    }
    _write_json(output_dir / "cifar10_st" / "metadata.json", metadata)
    LOGGER.info("CIFAR10-ST ready: %s", metadata)


@dataclass
class ImageNetLTConfig:
    metadata_dir: Path
    imagenet_root: Path
    output_dir: Path
    link_options: LinkOptions
    metadata_overrides: Optional[Dict[str, str]] = None


def _load_imagenet_manifest(path: Path) -> Iterator[Tuple[str, int]]:
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            stripped = line.strip()
            if not stripped:
                continue
            rel_path, label = stripped.split()
            yield rel_path, int(label)


def prepare_imagenet_lt(config: ImageNetLTConfig) -> None:
    target_root = _mkdir(config.output_dir.expanduser() / "imagenet_lt")
    metadata_dir = _mkdir(config.metadata_dir.expanduser())
    imagenet_root = config.imagenet_root.expanduser()
    if not imagenet_root.exists():
        raise DatasetPreparationError(f"ImageNet root {imagenet_root} does not exist")

    file_map = {
        "train": "ImageNet_LT_train.txt",
        "val": "ImageNet_LT_val.txt",
        "test": "ImageNet_LT_test.txt",
        "classnames": "ImageNet_LT_classnames.txt",
        "class_counts": "ImageNet_LT_class_counts.txt",
    }
    overrides = config.metadata_overrides or {}
    for key, filename in file_map.items():
        dest = metadata_dir / filename
        if key in overrides:
            _ensure_download(dest, [overrides[key]])
        else:
            urls = [pattern.format(name=filename) for pattern in IMAGENET_LT_METADATA_CANDIDATES]
            _ensure_download(dest, urls)

    stats = {}
    manifest_dir = _mkdir(target_root / "manifests")
    missing: Dict[str, int] = {"train": 0, "val": 0, "test": 0}
    for split in ("train", "val", "test"):
        manifest_path = metadata_dir / file_map[split]
        rows: List[Dict[str, object]] = []
        counts: Dict[int, int] = {}
        for rel_path, label in _load_imagenet_manifest(manifest_path):
            src = imagenet_root / rel_path
            if not src.exists():
                missing[split] += 1
                continue
            counts[label] = counts.get(label, 0) + 1
            if not config.link_options.manifest_only:
                dst = target_root / split / rel_path
                try:
                    _create_link(src, dst, config.link_options.link_type)
                except FileNotFoundError:
                    missing[split] += 1
                    continue
            rows.append({
                "absolute_path": str(src.resolve()),
                "relative_path": rel_path,
                "label": label,
            })
        with (manifest_dir / f"{split}.jsonl").open("w", encoding="utf-8") as fh:
            for row in rows:
                json.dump(row, fh)
                fh.write("\n")
        stats[split] = {
            "samples": len(rows),
            "unique_labels": len(counts),
            "missing": missing[split],
        }
    with (target_root / "metadata.json").open("w", encoding="utf-8") as fh:
        json.dump({
            "manifests": {split: str((manifest_dir / f"{split}.jsonl").resolve()) for split in ("train", "val", "test")},
            "class_counts_file": str((metadata_dir / file_map["class_counts"]).resolve()),
            "classnames_file": str((metadata_dir / file_map["classnames"]).resolve()),
            "stats": stats,
            "link_type": config.link_options.link_type,
            "materialised": not config.link_options.manifest_only,
        }, fh, indent=2, sort_keys=True)
    LOGGER.info("ImageNet-LT manifests written to %s", manifest_dir)
    for split, info in stats.items():
        LOGGER.info("%s: %s", split, info)
    if any(missing.values()):
        LOGGER.warning("Missing files detected: %s", missing)


@dataclass
class INatConfig:
    metadata_dir: Path
    inat_root: Path
    output_dir: Path
    link_options: LinkOptions
    metadata_overrides: Optional[Dict[str, str]] = None
    cache_dir: Path = DEFAULT_CACHE
    auto_download: bool = True
    archive_url: str = INAT2018_IMAGE_ARCHIVE_URL


def _download_inat_metadata(metadata_dir: Path, overrides: Optional[Dict[str, str]]) -> Dict[str, Path]:
    paths = {}
    overrides = overrides or {}
    for key, candidates in INAT2018_METADATA_CANDIDATES.items():
        dest = metadata_dir / f"{key}2018.json" if key in ("train", "val") else metadata_dir / f"{key}.json"
        if key in overrides:
            _ensure_download(dest, [overrides[key]])
        else:
            _ensure_download(dest, candidates)
        paths[key] = dest
    return paths


def _load_inat_split(data: dict) -> Tuple[Dict[int, dict], Dict[int, int]]:
    images = {image["id"]: image for image in data["images"]}
    labels: Dict[int, int] = {}
    for ann in data["annotations"]:
        labels[ann["image_id"]] = ann["category_id"]
    return images, labels


def prepare_inaturalist2018(config: INatConfig) -> None:
    target_root = _mkdir(config.output_dir.expanduser() / "inat2018")
    metadata_dir = _mkdir(config.metadata_dir.expanduser())
    inat_root = _ensure_inat_images(
        config.inat_root.expanduser(),
        config.cache_dir.expanduser(),
        config.auto_download,
        config.archive_url,
    )

    metadata_paths = _download_inat_metadata(metadata_dir, config.metadata_overrides)
    with metadata_paths["categories"].open("r", encoding="utf-8") as fh:
        categories = json.load(fh)
    id_to_index = {cat["id"]: idx for idx, cat in enumerate(categories)}
    id_to_name = {cat["id"]: cat.get("name", cat.get("display_name", "")) for cat in categories}

    manifest_dir = _mkdir(target_root / "manifests")
    stats = {}

    for split in ("train", "val"):
        with metadata_paths[split].open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        images, labels = _load_inat_split(data)
        rows: List[Dict[str, object]] = []
        class_counts: Dict[int, int] = {}
        missing = 0
        for image_id, image_info in images.items():
            file_name = image_info["file_name"]
            category_id = labels.get(image_id)
            if category_id is None:
                continue
            label = id_to_index[category_id]
            src = inat_root / file_name
            if not src.exists():
                missing += 1
                continue
            class_counts[label] = class_counts.get(label, 0) + 1
            if not config.link_options.manifest_only:
                dst = target_root / split / file_name
                try:
                    _create_link(src, dst, config.link_options.link_type)
                except FileNotFoundError:
                    missing += 1
                    continue
            rows.append({
                "absolute_path": str(src.resolve()),
                "relative_path": file_name,
                "label": label,
                "original_category_id": category_id,
            })
        manifest_file = manifest_dir / f"{split}.jsonl"
        with manifest_file.open("w", encoding="utf-8") as fh:
            for row in rows:
                json.dump(row, fh)
                fh.write("\n")
        stats[split] = {
            "samples": len(rows),
            "missing": missing,
            "unique_labels": len(class_counts),
        }
        LOGGER.info("iNat2018 %s manifest: %s", split, manifest_file)

    metadata = {
        "categories": [
            {
                "index": id_to_index[cat["id"]],
                "id": cat["id"],
                "name": id_to_name[cat["id"]],
                "supercategory": cat.get("supercategory"),
            }
            for cat in categories
        ],
        "manifests": {split: str((manifest_dir / f"{split}.jsonl").resolve()) for split in ("train", "val")},
        "link_type": config.link_options.link_type,
        "materialised": not config.link_options.manifest_only,
        "stats": stats,
    }
    _write_json(target_root / "metadata.json", metadata)
    LOGGER.info("iNaturalist2018 manifests written to %s", manifest_dir)


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="[%(levelname)s] %(message)s")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Download and prepare DRO datasets")
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE, help="Shared cache for downloads")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    subparsers = parser.add_subparsers(dest="task", required=True)

    cifar_parser = subparsers.add_parser("cifar10-st", help="Prepare CIFAR10-ST subset")
    cifar_parser.add_argument("--output-dir", type=Path, required=True, help="Destination directory")

    imagenet_parser = subparsers.add_parser("imagenet-lt", help="Prepare ImageNet-LT manifests")
    imagenet_parser.add_argument("--imagenet-root", type=Path, required=True, help="Original ImageNet tree")
    imagenet_parser.add_argument("--output-dir", type=Path, required=True, help="Destination directory")
    imagenet_parser.add_argument(
        "--metadata-dir", type=Path, default=None, help="Where to cache metadata (defaults to cache-dir/imagenet_lt)"
    )
    imagenet_parser.add_argument("--link-type", choices=["symlink", "hardlink", "copy"], default="symlink")
    imagenet_parser.add_argument("--manifest-only", action="store_true", help="Do not materialise files, emit manifests")
    imagenet_parser.add_argument(
        "--metadata-url", action="append", default=None,
        help="Override metadata URLs as key=url (e.g., train=https://.../ImageNet_LT_train.txt)"
    )

    inat_parser = subparsers.add_parser("inat2018", help="Prepare iNaturalist2018 manifests")
    inat_parser.add_argument("--inat-root", type=Path, required=True, help="Root directory with extracted images")
    inat_parser.add_argument("--output-dir", type=Path, required=True, help="Destination directory")
    inat_parser.add_argument(
        "--metadata-dir", type=Path, default=None, help="Where to cache metadata (defaults to cache-dir/inat2018)"
    )
    inat_parser.add_argument("--link-type", choices=["symlink", "hardlink", "copy"], default="symlink")
    inat_parser.add_argument("--manifest-only", action="store_true", help="Only emit manifests")
    inat_parser.add_argument(
        "--no-auto-download",
        dest="auto_download",
        action="store_false",
        help="Skip automatic download of the official train/val archive",
    )
    inat_parser.set_defaults(auto_download=True)
    inat_parser.add_argument(
        "--archive-url",
        type=str,
        default=INAT2018_IMAGE_ARCHIVE_URL,
        help="Custom URL for the iNaturalist2018 train/val archive",
    )
    inat_parser.add_argument(
        "--metadata-url", action="append", default=None,
        help=(
            "Override metadata URLs as key=url (keys: train, val, categories). "
            "Example: train=https://.../train2018.json"
        ),
    )

    return parser


def _parse_overrides(entries: Optional[List[str]]) -> Dict[str, str]:
    overrides: Dict[str, str] = {}
    if not entries:
        return overrides
    for item in entries:
        if "=" not in item:
            raise argparse.ArgumentTypeError(f"Override '{item}' must be key=url")
        key, url = item.split("=", 1)
        overrides[key.strip()] = url.strip()
    return overrides


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    _setup_logging(args.verbose)

    if args.task == "cifar10-st":
        prepare_cifar10_st(args.cache_dir / "cifar10", args.output_dir)
        return 0

    if args.task == "imagenet-lt":
        metadata_dir = args.metadata_dir or args.cache_dir / "imagenet_lt"
        overrides = _parse_overrides(args.metadata_url)
        config = ImageNetLTConfig(
            metadata_dir=metadata_dir,
            imagenet_root=args.imagenet_root,
            output_dir=args.output_dir,
            link_options=LinkOptions(link_type=args.link_type, manifest_only=args.manifest_only),
            metadata_overrides=overrides,
        )
        prepare_imagenet_lt(config)
        return 0

    if args.task == "inat2018":
        metadata_dir = args.metadata_dir or args.cache_dir / "inat2018"
        overrides = _parse_overrides(args.metadata_url)
        config = INatConfig(
            metadata_dir=metadata_dir,
            inat_root=args.inat_root,
            output_dir=args.output_dir,
            link_options=LinkOptions(link_type=args.link_type, manifest_only=args.manifest_only),
            metadata_overrides=overrides,
            cache_dir=args.cache_dir,
            auto_download=args.auto_download,
            archive_url=args.archive_url,
        )
        prepare_inaturalist2018(config)
        return 0

    parser.error(f"Unsupported task {args.task}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
