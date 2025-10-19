"""Utility helpers for ASCDRO training."""
from __future__ import annotations

import math
import os
import random
from typing import Iterable, List, Sequence

import torch
from torch import nn


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class AverageMeter:
    """Tracks running averages for scalars."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1) -> None:
        self.val = float(val)
        self.sum += float(val) * n
        self.count += n
        self.avg = self.sum / self.count if self.count else 0.0


def accuracy(output: torch.Tensor, target: torch.Tensor, topk: Sequence[int] = (1,)) -> List[torch.Tensor]:
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def clone_model(model: nn.Module) -> nn.Module:
    import copy
    clone = copy.deepcopy(model)
    for p in clone.parameters():
        p.requires_grad = True
    return clone


def model_parameter_vector(model: nn.Module) -> torch.Tensor:
    return torch.cat([p.data.reshape(-1) for p in model.parameters()])


def l2_distance_model(model_a: nn.Module, model_b: nn.Module) -> float:
    with torch.no_grad():
        va = model_parameter_vector(model_a)
        vb = model_parameter_vector(model_b)
        return torch.norm(va - vb, p=2).item()


def apply_update(model: nn.Module, grads: Iterable[torch.Tensor], lr: float) -> None:
    with torch.no_grad():
        for param, grad in zip(model.parameters(), grads):
            param.add_(-lr * grad)


def tensor_list_norm(grads: Iterable[torch.Tensor]) -> float:
    return math.sqrt(sum(g.detach().pow(2).sum().item() for g in grads))


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_device(prefer_cuda: bool = True) -> torch.device:
    if prefer_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
