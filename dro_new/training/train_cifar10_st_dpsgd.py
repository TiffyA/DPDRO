#!/usr/bin/env python3
"""Train ResNet on ImageNet with DP-SGD."""
from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional

# Ensure repository root is on sys.path when running as a script.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn.functional as F
try:
    from opacus import PrivacyEngine
    from opacus.validators import ModuleValidator
except ModuleNotFoundError as exc:  # pragma: no cover - import guard
    raise ModuleNotFoundError(
        "The 'opacus' package is required for DP-SGD training. "
        "Install it with 'pip install opacus' before running this script."
    ) from exc
from torch.utils.data import DataLoader, Dataset, Subset

from dro_new.ascdro.datasets import ImageNetFolder, build_imagenet_transforms
from dro_new.ascdro.models import build_resnet_imagenet


class DPSGDModel(torch.nn.Module):
    def __init__(self, backbone: torch.nn.Module, lambda0: float, lambda_init: float) -> None:
        super().__init__()
        self.backbone = backbone
        self.lambda0 = lambda0
        lambda_offset = max(0.0, lambda_init - lambda0)
        # initialise so that softplus(lambda_raw) \approx lambda_offset
        if lambda_offset > 0:
            init_raw = math.log(math.expm1(lambda_offset))
        else:
            init_raw = -5.0
        self.lambda_raw = torch.nn.Parameter(torch.tensor(init_raw, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def lambda_value(self) -> torch.Tensor:
        return self.lambda0 + torch.nn.functional.softplus(self.lambda_raw)


def get_lambda_value(module: torch.nn.Module) -> torch.Tensor:
    if hasattr(module, "lambda_value"):
        return module.lambda_value()
    inner = getattr(module, "_module", None)
    if inner is not None and hasattr(inner, "lambda_value"):
        return inner.lambda_value()
    raise AttributeError("DP-SGD model does not expose lambda_value()")


LOGGER = logging.getLogger("dpsgd")


def setup_logging(log_file: Optional[Path], verbose: bool) -> None:
    handlers = [logging.StreamHandler()]
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, mode="w"))
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers,
    )


def set_seed(seed: Optional[int]) -> None:
    if seed is None:
        return
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


@dataclass
class MetricsHistoryEntry:
    step: int
    train_objective: float
    train_acc: float
    val_loss: float
    val_acc: float
    lambda_value: float


def _apply_debug_subset(dataset: Dataset, limit: int | None) -> Dataset:
    if limit is None or limit <= 0:
        return dataset
    subset = Subset(dataset, range(min(limit, len(dataset))))
    if hasattr(dataset, "num_classes"):
        setattr(subset, "num_classes", getattr(dataset, "num_classes"))
    return subset


def build_dataloaders(
    data_root: Path,
    batch_size: int,
    num_workers: int,
    image_size: int,
    debug_samples: int | None,
) -> tuple[Dict[str, DataLoader], int]:
    data_root = data_root.expanduser()
    train_dataset = ImageNetFolder(data_root, "train", transform=build_imagenet_transforms(train=True, image_size=image_size))
    val_split = "val" if (data_root / "val").exists() else "train"
    val_dataset = ImageNetFolder(data_root, val_split, transform=build_imagenet_transforms(train=False, image_size=image_size))

    train_dataset = _apply_debug_subset(train_dataset, debug_samples)
    val_dataset = _apply_debug_subset(val_dataset, debug_samples)

    pin_memory = torch.cuda.is_available()
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
    eval_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    loaders = {"train": train_loader, "eval": eval_loader, "test": test_loader}
    return loaders, train_dataset.num_classes


def evaluate(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device)
            targets = targets.to(device)
            logits = model(images)
            loss = F.cross_entropy(logits, targets, reduction="sum")
            total_loss += loss.item()
            preds = logits.argmax(dim=1)
            total_correct += (preds == targets).sum().item()
            total_samples += targets.size(0)
    avg_loss = total_loss / max(1, total_samples)
    acc = 100.0 * total_correct / max(1, total_samples)
    return {"loss": avg_loss, "acc": acc}


def train(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loaders, num_classes = build_dataloaders(
        args.data_root,
        args.batch_size,
        args.num_workers,
        args.image_size,
        args.debug_samples,
    )
    train_loader = loaders["train"]
    eval_loader = loaders["eval"]

    LOGGER.info(
        "Loaded ImageNet dataset | train=%d val=%d classes=%d",
        len(train_loader.dataset),
        len(eval_loader.dataset),
        num_classes,
    )

    backbone = build_resnet_imagenet(depth=args.model_depth, num_classes=num_classes).to(device)

    if not ModuleValidator.is_valid(backbone):
        LOGGER.info("Converting BatchNorm layers to GroupNorm for DP compatibility")
        backbone = ModuleValidator.fix(backbone)
        backbone = backbone.to(device)

    ModuleValidator.validate(backbone, strict=True)

    model = DPSGDModel(backbone, args.lambda0, args.lambda_init).to(device)

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        nesterov=args.nesterov,
    )

    steps_per_epoch = math.ceil(len(train_loader.dataset) / args.batch_size)
    max_steps = args.max_steps
    epochs = max(1, math.ceil(max_steps / steps_per_epoch))

    LOGGER.info(
        "Preparing DP-SGD: epsilon=%.3f delta=%.1e | max_grad_norm=%.3f | epochs=%d | steps_per_epoch=%d",
        args.epsilon,
        args.delta,
        args.max_grad_norm,
        epochs,
        steps_per_epoch,
    )

    privacy_engine = PrivacyEngine()
    model, optimizer, private_loader = privacy_engine.make_private_with_epsilon(
        module=model,
        optimizer=optimizer,
        data_loader=train_loader,
        epochs=epochs,
        target_epsilon=args.epsilon,
        target_delta=args.delta,
        max_grad_norm=args.max_grad_norm,
    )

    train_loader = private_loader

    history: List[MetricsHistoryEntry] = []
    running_objective = 0.0
    running_ce = 0.0
    objective_batches = 0
    running_correct = 0
    running_samples = 0
    step = 0
    start_time = time.time()

    for epoch in range(epochs):
        model.train()
        for images, targets in train_loader:
            images = images.to(device)
            targets = targets.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            per_sample_loss = F.cross_entropy(logits, targets, reduction="none")
            lambda_value = get_lambda_value(model)
            scaled_losses = per_sample_loss / lambda_value
            log_mean = torch.logsumexp(scaled_losses, dim=0) - math.log(per_sample_loss.numel())
            objective = lambda_value * log_mean + (lambda_value - args.lambda0) * args.rho
            objective.backward()
            optimizer.step()

            running_objective += objective.item()
            running_ce += per_sample_loss.mean().item()
            objective_batches += 1
            running_correct += (logits.argmax(dim=1) == targets).sum().item()
            running_samples += targets.size(0)

            step += 1

            if step % args.log_interval == 0:
                avg_objective = running_objective / max(1, objective_batches)
                avg_acc = 100.0 * running_correct / max(1, running_samples)
                LOGGER.info(
                    "step=%d objective=%.4f ce=%.4f train_acc=%.2f lambda=%.4f",
                    step,
                    avg_objective,
                    running_ce / max(1, objective_batches),
                    avg_acc,
                    lambda_value.detach().item(),
                )

            if step % args.eval_interval == 0 or step >= max_steps:
                train_objective = running_objective / max(1, objective_batches)
                train_acc = 100.0 * running_correct / max(1, running_samples)
                running_objective = 0.0
                running_ce = 0.0
                objective_batches = 0
                running_correct = 0
                running_samples = 0

                val_metrics = evaluate(model, eval_loader, device)
                LOGGER.info(
                    "[EVAL] step=%d objective=%.4f train_acc=%.2f val_loss=%.4f val_acc=%.2f lambda=%.4f",
                    step,
                    train_objective,
                    train_acc,
                    val_metrics["loss"],
                    val_metrics["acc"],
                    get_lambda_value(model).detach().item(),
                )
                history.append(
                    MetricsHistoryEntry(
                        step=step,
                        train_objective=train_objective,
                        train_acc=train_acc,
                        val_loss=val_metrics["loss"],
                        val_acc=val_metrics["acc"],
                        lambda_value=get_lambda_value(model).detach().item(),
                    )
                )

            if step >= max_steps:
                break
        if step >= max_steps:
            break

    elapsed = time.time() - start_time
    privacy_spent = privacy_engine.get_epsilon(args.delta)
    LOGGER.info(
        "Training completed in %.2f seconds | steps=%d | epsilon_spent=%.3f",
        elapsed,
        step,
        privacy_spent,
    )

    if not history:
        final_eval = evaluate(model, eval_loader, device)
        history.append(
            MetricsHistoryEntry(
                step=step,
                train_objective=final_eval["loss"],
                train_acc=final_eval["acc"],
                val_loss=final_eval["loss"],
                val_acc=final_eval["acc"],
                lambda_value=get_lambda_value(model).detach().item(),
            )
        )

    metrics_path = args.output_dir / "metrics_dpsgd.json"
    payload = {
        "history": [asdict(entry) for entry in history],
        "privacy": {
            "target_epsilon": args.epsilon,
            "target_delta": args.delta,
            "spent_epsilon": privacy_spent,
            "max_grad_norm": args.max_grad_norm,
        },
        "training_seconds": elapsed,
        "total_steps": step,
        "lambda_final": get_lambda_value(model).detach().item(),
    }
    payload["dataset"] = {
        "train_size": len(train_loader.dataset),
        "val_size": len(eval_loader.dataset),
        "num_classes": num_classes,
        "image_size": args.image_size,
    }
    payload["args"] = {
        key: (str(value) if isinstance(value, Path) else value)
        for key, value in vars(args).items()
    }
    with metrics_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    LOGGER.info("Metrics written to %s", metrics_path)

    torch.save(model.state_dict(), args.output_dir / "checkpoints" / "final_model.pt")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train ResNet on ImageNet with DP-SGD")
    parser.add_argument("--data-root", type=Path, required=True, help="Directory containing ImageNet train/ and val/ folders")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory to write checkpoints and metrics")
    parser.add_argument("--epsilon", type=float, required=True, help="Target privacy budget ε")
    parser.add_argument("--delta", type=float, default=0.1, help="Privacy parameter δ")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--nesterov", action="store_true")
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--max-steps", type=int, default=2048)
    parser.add_argument("--eval-interval", type=int, default=128)
    parser.add_argument("--log-interval", type=int, default=64)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--model-depth", type=int, default=50, help="ResNet depth (choose from 18, 34, 50, 101, 152)")
    parser.add_argument("--image-size", type=int, default=224, help="Input image size for ImageNet crops")
    parser.add_argument("--debug-samples", type=int, default=None, help="Limit dataset to first N samples for debugging")
    parser.add_argument("--lambda0", type=float, default=0.01, help="Lower bound for lambda")
    parser.add_argument("--rho", type=float, default=0.5, help="Penalty coefficient rho")
    parser.add_argument("--lambda-init", type=float, default=0.1, help="Initial lambda value")
    parser.add_argument("--log-file", type=Path, default=None)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.data_root = args.data_root.expanduser()
    args.output_dir = args.output_dir.expanduser()
    if args.log_file is not None:
        args.log_file = args.log_file.expanduser()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "checkpoints").mkdir(exist_ok=True)
    setup_logging(args.log_file, args.verbose)
    set_seed(args.seed)
    try:
        train(args)
    except Exception:
        LOGGER.exception("Training failed")
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
