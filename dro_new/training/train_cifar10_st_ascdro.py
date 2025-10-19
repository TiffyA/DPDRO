"""Train ResNet on ImageNet using the ASCDRO algorithm with Private SpiderBoost."""
from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dro_new.ascdro.algorithms import ASCDROConfig, ASCDROTrainer, DPSpiderConfig
from dro_new.ascdro.datasets import ImageNetFolder, build_imagenet_transforms
from dro_new.ascdro.models import build_resnet_imagenet
from dro_new.ascdro.risk import RiskModel
from dro_new.ascdro.utils import AverageMeter, count_parameters, get_device, set_seed


LOGGER = logging.getLogger("train_imagenet_ascdro")


def setup_logging(log_path: Path, verbose: bool) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    handlers: List[logging.Handler] = [logging.StreamHandler(sys.stdout), logging.FileHandler(log_path, mode="w")]
    logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO, format="[%(asctime)s] [%(levelname)s] %(message)s", handlers=handlers)


def warmup_cross_entropy(
    model: torch.nn.Module,
    train_loader: DataLoader,
    device: torch.device,
    steps: int,
    lr: float,
    grad_clip: Optional[float],
) -> Dict[str, float]:
    if steps <= 0:
        return {"steps": 0, "avg_loss": 0.0, "avg_acc": 0.0}
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    criterion = torch.nn.CrossEntropyLoss()
    step = 0
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    data_iter = iter(train_loader)
    while step < steps:
        try:
            images, targets = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            images, targets = next(data_iter)
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, targets)
        loss.backward()
        if grad_clip is not None and grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        batch_size = targets.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (logits.argmax(dim=1) == targets).sum().item()
        total_samples += batch_size
        step += 1
    avg_loss = total_loss / max(total_samples, 1)
    avg_acc = 100.0 * total_correct / max(total_samples, 1)
    LOGGER.info("CE warmup completed | steps=%d avg_loss=%.4f avg_acc=%.2f", step, avg_loss, avg_acc)
    return {"steps": step, "avg_loss": avg_loss, "avg_acc": avg_acc}

def estimate_initial_loss(model: torch.nn.Module, loader: DataLoader, device: torch.device, max_batches: int = 100) -> float:
    model.eval()
    loss_meter = AverageMeter()
    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(loader, start=1):
            images = images.to(device)
            targets = targets.to(device)
            logits = model(images)
            loss = F.cross_entropy(logits, targets, reduction="mean")
            loss_meter.update(loss.item(), n=images.size(0))
            if batch_idx >= max_batches:
                break
    model.train()
    return max(loss_meter.avg, 1e-6)


def _gradient_norm(model: torch.nn.Module, images: torch.Tensor, targets: torch.Tensor, device: torch.device) -> float:
    images = images.to(device)
    targets = targets.to(device)
    loss = F.cross_entropy(model(images), targets, reduction="mean")
    grads = torch.autograd.grad(loss, list(model.parameters()), retain_graph=False, create_graph=False)
    total = 0.0
    for g in grads:
        total += g.detach().pow(2).sum().item()
    return math.sqrt(total)


def estimate_gradient_lipschitz(model: torch.nn.Module, loader: DataLoader, device: torch.device, max_batches: int = 20) -> float:
    model.eval()
    max_norm = 0.0
    for batch_idx, (images, targets) in enumerate(loader, start=1):
        norm = _gradient_norm(model, images, targets, device)
        max_norm = max(max_norm, norm)
        if batch_idx >= max_batches:
            break
    model.train()
    return max(max_norm, 1e-6)


def estimate_smoothness(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    model_builder: Callable[[], torch.nn.Module],
    sigma: float = 1e-3,
) -> float:
    base_model = model.module if isinstance(model, torch.nn.DataParallel) else model
    model.eval()
    grads_base = None
    count = 0
    for images, targets in loader:
        grads = torch.autograd.grad(
            F.cross_entropy(model(images.to(device)), targets.to(device), reduction="mean"),
            list(model.parameters()),
            retain_graph=False,
            create_graph=False,
        )
        grads = [g.detach() for g in grads]
        if grads_base is None:
            grads_base = grads
        else:
            grads_base = [gb + g for gb, g in zip(grads_base, grads)]
        count += 1
        if count >= 5:
            break
    if grads_base is None:
        raise RuntimeError("Unable to estimate smoothness: empty loader")
    grads_base = [g / count for g in grads_base]

    perturbed = model_builder().to(device)
    perturbed.load_state_dict(base_model.state_dict())
    perturbed.eval()
    perturbed.eval()
    with torch.no_grad():
        for p in perturbed.parameters():
            p.add_(torch.randn_like(p) * sigma)
    diff_norm = 0.0
    for p_orig, p_new in zip(base_model.parameters(), perturbed.parameters()):
        diff_norm += (p_orig.detach() - p_new.detach()).pow(2).sum().item()
    diff_norm = math.sqrt(max(diff_norm, 1e-12))

    grads_perturbed = None
    count = 0
    for images, targets in loader:
        grads = torch.autograd.grad(
            F.cross_entropy(perturbed(images.to(device)), targets.to(device), reduction="mean"),
            list(perturbed.parameters()),
            retain_graph=False,
            create_graph=False,
        )
        grads = [g.detach() for g in grads]
        if grads_perturbed is None:
            grads_perturbed = grads
        else:
            grads_perturbed = [gb + g for gb, g in zip(grads_perturbed, grads)]
        count += 1
        if count >= 5:
            break
    grads_perturbed = [g / count for g in grads_perturbed]

    grad_diff = 0.0
    for g0, g1 in zip(grads_base, grads_perturbed):
        grad_diff += (g0 - g1).pow(2).sum().item()
    grad_diff = math.sqrt(max(grad_diff, 1e-12))
    model.train()
    return max(grad_diff / diff_norm, 1e-6)


def compute_dp_parameters(
    *,
    n: int,
    d: int,
    constants: Dict[str, float],
    eps: float,
    delta: float,
) -> Dict[str, int | float]:
    F0 = max(constants["F0"], 1e-6)
    G = max(constants["G"], 1e-6)
    L = max(constants["L"], 1e-6)
    L0 = max(constants["L0"], 1e-6)
    L1 = max(constants["L1"], 1e-6)
    log_term = max(math.log(1.0 / delta + 1e-12), 1e-6)

    b1 = n
    term1 = (L0 * n * eps / math.sqrt(F0 * L1 * d * log_term)) ** (2.0 / 3.0)
    term2_num = (L0 * n * d * log_term) ** (1.0 / 3.0)
    term2_den = max(1e-12, (L1 * F0) ** (1.0 / 6.0) * (eps ** (2.0 / 3.0)))
    term2 = term2_num / term2_den
    b2 = int(max(1.0, min(n, math.floor(max(term1, term2)))))

    term_t1 = ((F0 * L) ** 0.25) * n * eps / math.sqrt(L0 * d * log_term)
    term_t_candidate = term_t1 ** (4.0 / 3.0)
    term_t_alt = n * eps / math.sqrt(d * log_term)
    term_t = int(max(1.0, math.floor(max(term_t_candidate, term_t_alt))))

    q_val = int(max(1.0, math.floor(n ** (2.0 / 3.0))))

    return {
        "b1": b1,
        "b2": b2,
        "T": term_t,
        "q": q_val,
        "F0": F0,
        "G": G,
        "L": L,
        "L0": L0,
        "L1": L1,
    }


def _apply_debug_subset(dataset: Dataset, limit: int | None) -> Dataset:
    if limit is None or limit <= 0:
        return dataset
    max_count = min(limit, len(dataset))
    subset = Subset(dataset, range(max_count))
    if hasattr(dataset, "num_classes"):
        setattr(subset, "num_classes", getattr(dataset, "num_classes"))
    return subset


def prepare_dataloaders(
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train deep ResNet on ImageNet using ASCDRO")
    parser.add_argument("--data-root", type=Path, required=True, help="Directory containing ImageNet train/ and val/ folders")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory to store checkpoints and metrics")
    parser.add_argument("--epsilon", type=float, required=True, help="Privacy budget ε (choose from 0.5,1,2,4,8)")
    parser.add_argument("--delta", type=float, default=0.1, help="Privacy parameter δ")
    parser.add_argument("--batch-size", type=int, default=192)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--rho", type=float, default=0.5)
    parser.add_argument("--lambda0", type=float, default=0.1)
    parser.add_argument("--beta", type=float, default=0.5)
    parser.add_argument("--eta", type=float, default=0.05)
    parser.add_argument(
        "--loss-scale",
        type=float,
        default=None,
        help="Factor to scale per-sample loss before the exponential (defaults to lambda0)",
    )
    parser.add_argument("--model-depth", type=int, default=50, help="ResNet depth (choose from 18, 34, 50, 101, 152)")
    parser.add_argument("--image-size", type=int, default=224, help="Input image size for ImageNet crops")
    parser.add_argument("--debug-samples", type=int, default=None, help="Limit training/validation datasets to the first N samples (for debugging)")
    parser.add_argument("--c", type=float, default=1.0, help="Noise calibration constant c")
    parser.add_argument("--max-steps", type=int, default=2048, help="Optional cap on total ASCDRO iterations")
    parser.add_argument("--min-steps", type=int, default=512, help="Ensure at least this many optimization steps")
    parser.add_argument("--grad-clip", type=float, default=1.0, help="Gradient clipping threshold (set <=0 to disable)")
    parser.add_argument("--exp-clip", type=float, default=10.0, help="Clamp exp(loss/lam) input to this maximum value")
    parser.add_argument("--ce-warmup-steps", type=int, default=1024, help="Number of CE warmup steps before ASCDRO")
    parser.add_argument("--ce-warmup-lr", type=float, default=None, help="Learning rate for CE warmup (defaults to eta)")
    parser.add_argument("--log-file", type=Path, default=None, help="Optional log file path")
    parser.add_argument("--override-constants", action="append", default=None, help="Override estimates as key=value for F0,G,L,L0,L1")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def load_constant_overrides(entries: Optional[List[str]]) -> Dict[str, float]:
    overrides: Dict[str, float] = {}
    if not entries:
        return overrides
    for item in entries:
        if "=" not in item:
            raise argparse.ArgumentTypeError(f"Constant override '{item}' must be key=value")
        key, value = item.split("=", 1)
        overrides[key.strip()] = float(value.strip())
    return overrides


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "checkpoints").mkdir(exist_ok=True)

    log_path = args.log_file if args.log_file is not None else output_dir / "training.log"
    setup_logging(log_path, args.verbose)
    LOGGER.info("Logging to %s", log_path)

    if args.seed is not None:
        set_seed(args.seed)

    device = get_device()
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
    data_root = args.data_root

    loaders, num_classes = prepare_dataloaders(
        data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        image_size=args.image_size,
        debug_samples=args.debug_samples,
    )
    train_loader = loaders["train"]
    eval_loader = loaders["eval"]
    test_loader = loaders["test"]

    LOGGER.info(
        "Loaded ImageNet dataset | train=%d val=%d classes=%d",
        len(train_loader.dataset),
        len(eval_loader.dataset),
        num_classes,
    )

    def model_builder() -> torch.nn.Module:
        return build_resnet_imagenet(depth=args.model_depth, num_classes=num_classes)

    model = model_builder().to(device)
    if device.type == "cuda" and torch.cuda.device_count() > 1:
        LOGGER.info("Using DataParallel across %d GPUs", torch.cuda.device_count())
        model = torch.nn.DataParallel(model)
    n = len(train_loader.dataset)
    d = count_parameters(model)
    LOGGER.info("Dataset size n=%d | parameter dimension d=%d", n, d)

    ce_warmup_lr = args.ce_warmup_lr if args.ce_warmup_lr is not None else args.eta
    warmup_stats = warmup_cross_entropy(
        model,
        train_loader,
        device,
        steps=args.ce_warmup_steps,
        lr=ce_warmup_lr,
        grad_clip=args.grad_clip,
    )

    constants: Dict[str, float] = {}
    overrides = load_constant_overrides(args.override_constants)

    # Estimate constants when not provided.
    with torch.enable_grad():
        constants["G"] = overrides.get("G", estimate_gradient_lipschitz(model, train_loader, device))
        constants["L"] = overrides.get("L", estimate_smoothness(model, train_loader, device, model_builder))
        constants["F0"] = overrides.get("F0", estimate_initial_loss(model, train_loader, device))
        constants["L0"] = overrides.get("L0", constants["L"])
        constants["L1"] = overrides.get("L1", constants["L"])

    LOGGER.info("Estimated constants: %s", {k: float(v) for k, v in constants.items()})

    dp_params = compute_dp_parameters(
        n=n,
        d=d,
        constants=constants,
        eps=args.epsilon,
        delta=args.delta,
    )

    steps_per_epoch = len(train_loader)
    min_steps = args.min_steps if args.min_steps is not None else steps_per_epoch
    total_steps = max(int(dp_params["T"]), int(min_steps))
    if args.max_steps is not None:
        total_steps = min(total_steps, int(args.max_steps))
    dp_params["T"] = int(total_steps)
    dp_params["steps_per_epoch"] = steps_per_epoch
    dp_params["q"] = int(max(1.0, math.floor(n ** (2.0 / 3.0))))
    LOGGER.info(
        "DP Spider parameters: %s | using total_steps=%d (steps/epoch=%d)",
        {k: v for k, v in dp_params.items() if k not in {"steps_per_epoch"}},
        total_steps,
        steps_per_epoch,
    )

    loss_scale = args.loss_scale if args.loss_scale is not None else args.lambda0

    spider_cfg = DPSpiderConfig(
        eta=args.eta,
        q=int(dp_params["q"]),
        b1=int(dp_params["b1"]),
        b2=int(dp_params["b2"]),
        c=args.c,
        L0=float(dp_params["L0"]),
        L1=float(dp_params["L1"]),
        eps=args.epsilon,
        delta=args.delta,
        T_total=int(dp_params["T"]),
        n=n,
        d=d,
        loss_scale=float(loss_scale),
        exp_clip=args.exp_clip,
    )

    ascdro_cfg = ASCDROConfig(
        eta=args.eta,
        beta=args.beta,
        rho=args.rho,
        lambda0=args.lambda0,
        spider=spider_cfg,
        grad_clip=args.grad_clip if args.grad_clip is not None and args.grad_clip > 0 else None,
    )

    risk = RiskModel(rho=args.rho)
    trainer = ASCDROTrainer(model, risk, ascdro_cfg, device, fullpass_loader=eval_loader)
    start_time = time.perf_counter()
    history = trainer.fit(train_loader, test_loader)
    training_seconds = time.perf_counter() - start_time
    LOGGER.info("Training completed in %.2f seconds over %d steps", training_seconds, trainer.global_step)

    metrics_path = output_dir / "metrics_ascdro.json"
    args_dict = vars(args).copy()
    args_dict["num_classes"] = num_classes
    args_dict["data_root"] = str(args.data_root)
    args_dict["output_dir"] = str(args.output_dir)
    if args.override_constants:
        args_dict["override_constants"] = list(args.override_constants)
    args_dict["log_file"] = str(log_path)
    with metrics_path.open("w", encoding="utf-8") as fh:
        json.dump({
            "history": history,
            "epsilon": args.epsilon,
            "delta": args.delta,
            "dp_params": dp_params,
            "constants": constants,
            "args": args_dict,
            "training_seconds": training_seconds,
            "total_steps": trainer.global_step,
            "warmup": warmup_stats,
        }, fh, indent=2)
    torch.save({
        "model_state_dict": model.state_dict(),
        "lambda": trainer.lam.detach().cpu(),
        "dp_params": dp_params,
        "history": history,
        "args": args_dict,
        "training_seconds": training_seconds,
        "total_steps": trainer.global_step,
        "warmup": warmup_stats,
    }, output_dir / "checkpoints" / "final_ascdro.pt")

    if history:
        final = history[-1]
        LOGGER.info(
            "Final metrics | step=%d train_loss=%.4f train_acc=%.2f val_loss=%.4f val_acc=%.2f",
            final.get("step", -1),
            final.get("train_loss", float("nan")),
            final.get("train_acc", float("nan")),
            final.get("val_loss", float("nan")),
            final.get("val_acc", float("nan")),
        )
    LOGGER.info("Training complete. Metrics saved to %s", metrics_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
