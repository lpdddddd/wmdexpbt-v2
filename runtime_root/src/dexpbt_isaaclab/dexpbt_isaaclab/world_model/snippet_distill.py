"""First-step-only snippet actor distillation (CSER-v0)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import torch
import torch.nn.functional as F

from .wm_activation_gate import WMActivationConfig, compute_wm_activation


@dataclass
class SnippetDistillConfig:
    enabled: bool = False
    weight: float = 0.001
    loss_cap: float = 0.05
    huber_delta: float = 0.1
    start_after_frames: int = 300_000_000
    topk: int = 8
    mu_head_only: bool = True
    gate_mode: str = "soft_tol"
    tol_start: float = 0.070
    tol_full: float = 0.045
    ramp_frames: int = 200_000_000
    nsucc_ref: float = 15.0

    def activation_config(self) -> WMActivationConfig:
        return WMActivationConfig(
            enable=True,
            gate_mode=str(self.gate_mode),
            tol_start=float(self.tol_start),
            tol_full=float(self.tol_full),
            start_after_frames=int(self.start_after_frames),
            ramp_frames=int(self.ramp_frames),
            nsucc_ref=float(self.nsucc_ref),
        )


def compute_snippet_distill_loss(
    *,
    mu_fn: Callable[[torch.Tensor], torch.Tensor],
    feat_start: torch.Tensor,
    teacher_first_action: torch.Tensor,
    mask: torch.Tensor,
    cfg: SnippetDistillConfig,
    row_weight: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    if not cfg.enabled or not bool(mask.any()):
        z = feat_start.new_zeros(())
        return z, {"snippet_distill/loss": 0.0, "snippet_distill/active_frac": 0.0}
    mu_live = mu_fn(feat_start.detach())
    teacher = teacher_first_action.detach()
    per_row = F.smooth_l1_loss(mu_live, teacher, beta=float(cfg.huber_delta), reduction="none").sum(dim=-1)
    w = mask.float()
    if row_weight is not None:
        w = w * row_weight.float().clamp(0.0, 1.0)
    loss = (per_row * w).sum() / w.sum().clamp_min(1.0)
    if cfg.loss_cap > 0:
        loss = loss.clamp(max=float(cfg.loss_cap))
    loss = loss * float(cfg.weight)
    return loss, {"snippet_distill/loss": float(loss.detach().item()), "snippet_distill/active_frac": float(mask.float().mean().item())}


def compute_snippet_activation_weight(*, cfg: SnippetDistillConfig, frames: int, tol_mean: float, nsucc_mean: float = -1.0) -> float:
    return float(compute_wm_activation(cfg.activation_config(), frames=int(frames), tol=float(tol_mean), nsucc_mean=float(nsucc_mean), eval_valid=True).weight)
