"""Risk model utilities."""
from __future__ import annotations

import torch


class RiskModel:
    """Implements the KL-based risk used in the ASCDRO specification."""

    def __init__(self, rho: float = 0.5) -> None:
        self.rho = rho

    def grad_f(self, s: torch.Tensor, lam: torch.Tensor) -> torch.Tensor:
        return lam / (s + 1e-12)

    def f(self, s: torch.Tensor, lam: torch.Tensor) -> torch.Tensor:
        return lam * torch.log(s + 1e-12) + lam * self.rho
