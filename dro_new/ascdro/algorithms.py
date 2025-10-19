"""Implementation of ASCDRO with a private SpiderBoost estimator for CIFAR10-ST."""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .risk import RiskModel
from .utils import (
    AverageMeter,
    apply_update,
    clone_model,
    l2_distance_model,
)


LOGGER = logging.getLogger(__name__)


@dataclass
class DPSpiderConfig:
    eta: float
    q: int
    b1: int
    b2: int
    c: float
    L0: float
    L1: float
    eps: float
    delta: float
    T_total: int
    n: int
    d: int
    loss_scale: float
    exp_clip: float = 10.0


class DatasetPreparationError(RuntimeError):
    """Raised for configuration issues."""


class PrivateSpiderEstimator:
    """DP-SpiderBoost estimator for gradient tracking in ASCDRO."""

    def __init__(self, model: nn.Module, cfg: DPSpiderConfig, device: torch.device) -> None:
        self.model = model
        self.prev_model = clone_model(model).to(device)
        self.cfg = cfg
        self.device = device
        self.prev_lambda: Optional[torch.Tensor] = None
        self.loss_scale = cfg.loss_scale
        self.exp_clip = cfg.exp_clip
        self.logger = logging.getLogger(f"{__name__}.PrivateSpiderEstimator")

        self.v_est: Optional[List[torch.Tensor]] = None
        self.u_est: Optional[torch.Tensor] = None
        self.t = 0  # iteration counter

        self._calibrate_sigmas()

    def _calibrate_sigmas(self) -> None:
        c = self.cfg.c
        eps = self.cfg.eps
        delta = self.cfg.delta
        T = max(1, self.cfg.T_total)
        n = max(1, self.cfg.n)
        q = max(1, self.cfg.q)
        b1 = max(1, self.cfg.b1)
        b2 = max(1, self.cfg.b2)
        log_term = math.sqrt(math.log(1.0 / delta + 1e-12))
        self.sigma1 = (c * self.cfg.L0 * log_term / eps) * max(1.0 / b1, math.sqrt(T) / (q * n))
        self.sigma2 = (c * self.cfg.L1 * log_term / eps) * max(1.0 / b2, math.sqrt(T) / n)
        self.hat_sigma2 = (2 * c * self.cfg.L0 * log_term / eps) * max(1.0 / b2, math.sqrt(T) / n)
        self.logger.info(
            "sigma_calibrated: sigma1=%.6f variance=%.6f sigma2=%.6f hat_sigma2=%.6f",
            self.sigma1,
            self.sigma1 ** 2,
            self.sigma2,
            self.hat_sigma2,
        )

    def _compute_batch_stats(
        self,
        model: nn.Module,
        images: torch.Tensor,
        targets: torch.Tensor,
        lam_value: torch.Tensor,
    ) -> Tuple[torch.Tensor, List[torch.Tensor], torch.Tensor]:
        lam = lam_value.detach().clone().requires_grad_(True)
        logits = model(images)
        losses = F.cross_entropy(logits, targets, reduction="none") * self.loss_scale
        scaled = losses / (lam + 1e-12)
        if self.exp_clip is not None:
            clipped = torch.clamp(scaled, max=self.exp_clip)
            if torch.any(clipped.detach() != scaled.detach()):
                self.logger.debug("exp_input_clipped max=%.4f", self.exp_clip)
        else:
            clipped = scaled
        g_scalar = torch.exp(clipped).mean()

        grad_params = torch.autograd.grad(g_scalar, list(model.parameters()), retain_graph=True, create_graph=False)
        grad_lambda = torch.autograd.grad(g_scalar, lam, retain_graph=False, create_graph=False)[0]

        return g_scalar.detach(), [g.detach() for g in grad_params], grad_lambda.detach()

    def _add_noise_to_vec(self, vec: Iterable[torch.Tensor], std: float) -> List[torch.Tensor]:
        return [v + torch.randn_like(v) * std for v in vec]

    def _add_noise_to_scalar(self, value: torch.Tensor, std: float) -> torch.Tensor:
        return value + torch.randn_like(value) * std

    def anchor_due(self) -> bool:
        return self.t % max(1, self.cfg.q) == 0

    def refresh_anchor_fullpass(
        self,
        fullpass_loader: Iterable[Tuple[torch.Tensor, torch.Tensor]],
        lam: torch.Tensor,
    ) -> Tuple[torch.Tensor, List[torch.Tensor], torch.Tensor]:
        v_acc = [torch.zeros_like(p, device=self.device) for p in self.model.parameters()]
        u_acc = torch.zeros((), device=self.device)
        s_acc = torch.zeros((), device=self.device)
        n_total = 0
        for images, targets in fullpass_loader:
            images = images.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)
            batch_size = images.size(0)
            n_total += batch_size
            g_b, v_b, u_b = self._compute_batch_stats(self.model, images, targets, lam)
            weight = batch_size
            for idx, v in enumerate(v_b):
                v_acc[idx].add_(v * weight)
            u_acc += u_b * weight
            s_acc += g_b * weight
        if n_total == 0:
            raise DatasetPreparationError("Fullpass loader is empty; cannot refresh anchor.")
        for idx in range(len(v_acc)):
            v_acc[idx].div_(n_total)
        u_acc.div_(n_total)
        s_acc.div_(n_total)

        v_noisy = self._add_noise_to_vec(v_acc, self.sigma1)
        u_noisy = self._add_noise_to_scalar(u_acc, self.sigma1)
        self.logger.info(
            "anchor_step=%d variance=%.6f sigma=%.6f",
            self.t,
            self.sigma1 ** 2,
            self.sigma1,
        )
        self.v_est = v_noisy
        self.u_est = u_noisy
        self.t += 1

        with torch.no_grad():
            for p_prev, p in zip(self.prev_model.parameters(), self.model.parameters()):
                p_prev.data.copy_(p.data)
        self.prev_lambda = lam.detach().clone()

        return s_acc.detach(), self.v_est, self.u_est

    def step(
        self,
        images: torch.Tensor,
        targets: torch.Tensor,
        lam: torch.Tensor,
    ) -> Tuple[torch.Tensor, List[torch.Tensor], torch.Tensor]:
        images = images.to(self.device, non_blocking=True)
        targets = targets.to(self.device, non_blocking=True)
        g_curr, v_curr, u_curr = self._compute_batch_stats(self.model, images, targets, lam)
        lam_prev = self.prev_lambda if self.prev_lambda is not None else lam
        g_prev, v_prev, u_prev = self._compute_batch_stats(self.prev_model, images, targets, lam_prev)

        dv = [a - b for a, b in zip(v_curr, v_prev)]
        du = u_curr - u_prev
        dist = l2_distance_model(self.model, self.prev_model)
        std = min(self.sigma2 * dist, self.hat_sigma2)

        dv_noisy = self._add_noise_to_vec(dv, std)
        du_noisy = self._add_noise_to_scalar(du, std)
        self.logger.info(
            "step=%d sigma2=%.6f hat_sigma2=%.6f distance=%.6f std=%.6f variance=%.6f",
            self.t,
            self.sigma2,
            self.hat_sigma2,
            dist,
            std,
            std ** 2,
        )

        if self.v_est is None or self.u_est is None:
            self.v_est = dv_noisy
            self.u_est = du_noisy
        else:
            self.v_est = [vi + dvi for vi, dvi in zip(self.v_est, dv_noisy)]
            self.u_est = self.u_est + du_noisy

        self.t += 1
        return g_curr.detach(), self.v_est, self.u_est

    def commit(self, lam: torch.Tensor) -> None:
        with torch.no_grad():
            for p_prev, p in zip(self.prev_model.parameters(), self.model.parameters()):
                p_prev.data.copy_(p.data)
        self.prev_lambda = lam.detach().clone()


@dataclass
class ASCDROConfig:
    eta: float
    beta: float
    rho: float
    lambda0: float
    spider: DPSpiderConfig
    grad_clip: Optional[float] = None


class ASCDROTrainer:
    def __init__(
        self,
        model: nn.Module,
        risk: RiskModel,
        cfg: ASCDROConfig,
        device: torch.device,
        fullpass_loader: Iterable[Tuple[torch.Tensor, torch.Tensor]],
    ) -> None:
        self.model = model.to(device)
        self.risk = risk
        self.cfg = cfg
        self.device = device
        self.lam = torch.tensor(float(cfg.lambda0), device=device)
        self.spider = PrivateSpiderEstimator(self.model, cfg.spider, device)
        self.fullpass_loader = fullpass_loader
        self.global_step = 0
        self.T_target = cfg.spider.T_total
        self.s_t: Optional[torch.Tensor] = None
        self.prev_g: Optional[torch.Tensor] = None
        self.grad_clip = cfg.grad_clip

    def _project_lambda(self) -> None:
        with torch.no_grad():
            if self.lam.item() < self.cfg.lambda0:
                self.lam.fill_(self.cfg.lambda0)

    def fit(self, train_loader: torch.utils.data.DataLoader, eval_loader: torch.utils.data.DataLoader) -> List[Dict[str, float]]:
        history: List[Dict[str, float]] = []
        while self.global_step < self.T_target:
            loss_meter = AverageMeter()
            acc_meter = AverageMeter()

            for images, targets in train_loader:
                if self.global_step >= self.T_target:
                    break

                if self.spider.anchor_due():
                    g_value, v_est, u_est = self.spider.refresh_anchor_fullpass(self.fullpass_loader, self.lam)
                else:
                    g_value, v_est, u_est = self.spider.step(images, targets, self.lam)

                if self.s_t is None:
                    self.s_t = g_value.clone()
                    self.prev_g = g_value.clone()
                else:
                    beta = self.cfg.beta
                    self.s_t = g_value.clone() + (1.0 - beta) * (self.s_t - self.prev_g)
                    self.prev_g = g_value.clone()

                grad_f = self.risk.grad_f(self.s_t, self.lam)
                grad_w = [grad_f * g for g in v_est]
                z_lambda = grad_f * u_est + torch.log(self.s_t + 1e-12) + self.cfg.rho

                if not self._is_finite_list(grad_w) or not torch.isfinite(z_lambda):
                    LOGGER.warning("Non-finite gradient detected at step %d; skipping update", self.global_step)
                    continue

                if self.grad_clip is not None:
                    total_norm = torch.sqrt(torch.sum(torch.stack([g.pow(2).sum() for g in grad_w])))
                    if torch.isnan(total_norm) or torch.isinf(total_norm):
                        LOGGER.warning("Gradient norm non-finite at step %d; skipping update", self.global_step)
                        continue
                    if total_norm > self.grad_clip:
                        scale = self.grad_clip / (total_norm + 1e-12)
                        grad_w = [g * scale for g in grad_w]
                        LOGGER.debug("Gradient clipped at step %d with scale %.6f", self.global_step, float(scale))

                prev_params = [p.detach().clone() for p in self.model.parameters()]
                prev_lambda = self.lam.detach().clone()

                apply_update(self.model, grad_w, lr=self.cfg.eta)
                with torch.no_grad():
                    self.lam.add_(-self.cfg.eta * z_lambda.item())
                self._project_lambda()

                if not self._is_finite_model() or not torch.isfinite(self.lam):
                    LOGGER.warning("Non-finite parameters after update at step %d; reverting", self.global_step)
                    with torch.no_grad():
                        for p, backup in zip(self.model.parameters(), prev_params):
                            p.copy_(backup)
                        self.lam.copy_(prev_lambda)
                    continue

                self.spider.commit(self.lam)

                logits = self.model(images.to(self.device))
                loss = F.cross_entropy(logits, targets.to(self.device), reduction="mean")
                acc1 = (logits.argmax(dim=1) == targets.to(self.device)).float().mean().item() * 100.0
                loss_meter.update(loss.item(), n=targets.size(0))
                acc_meter.update(acc1, n=targets.size(0))

                self.global_step += 1

            val_stats = self.evaluate(eval_loader)
            history.append({
                "step": self.global_step,
                "train_loss": loss_meter.avg,
                "train_acc": acc_meter.avg,
                "lambda": float(self.lam.item()),
                "s_t": float(self.s_t.item()) if self.s_t is not None else 0.0,
                **val_stats,
            })

        return history

    def _is_finite_list(self, tensors: Iterable[torch.Tensor]) -> bool:
        return all(torch.isfinite(t).all() for t in tensors)

    def _is_finite_model(self) -> bool:
        return all(torch.isfinite(p).all() for p in self.model.parameters())

    @torch.no_grad()
    def evaluate(self, loader: torch.utils.data.DataLoader) -> Dict[str, float]:
        self.model.eval()
        loss_meter = AverageMeter()
        acc_meter = AverageMeter()
        for images, targets in loader:
            images = images.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)
            logits = self.model(images)
            loss = F.cross_entropy(logits, targets, reduction="mean")
            acc1 = (logits.argmax(dim=1) == targets).float().mean().item() * 100.0
            loss_meter.update(loss.item(), n=targets.size(0))
            acc_meter.update(acc1, n=targets.size(0))
        self.model.train()
        return {"val_loss": loss_meter.avg, "val_acc": acc_meter.avg}
