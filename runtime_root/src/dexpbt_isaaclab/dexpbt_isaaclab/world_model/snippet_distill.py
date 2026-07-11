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


def _weight_stats(w: torch.Tensor, *, min_gate: float) -> dict[str, float]:
    """Effective-weight diagnostics for a continuous row-weight vector."""
    n = int(w.numel())
    if n == 0:
        return {
            "snippet_distill/active_frac": 0.0,
            "snippet_distill/weight_pos_frac": 0.0,
            "snippet_distill/weight_ge_min_frac": 0.0,
            "snippet_distill/weight_mean": 0.0,
            "snippet_distill/weight_p50": 0.0,
            "snippet_distill/weight_p90": 0.0,
            "snippet_distill/effective_sample_size": 0.0,
        }
    wf = w.float()
    pos = wf > 0.0
    ge = wf >= float(min_gate)
    s1 = float(wf.sum().item())
    s2 = float((wf * wf).sum().item())
    ess = (s1 * s1) / s2 if s2 > 0.0 else 0.0
    try:
        p50 = float(torch.quantile(wf, 0.50).item())
        p90 = float(torch.quantile(wf, 0.90).item())
    except Exception:  # quantile can fail on tiny/edge tensors
        srt = torch.sort(wf).values
        p50 = float(srt[min(n - 1, n // 2)].item())
        p90 = float(srt[min(n - 1, int(0.9 * n))].item())
    return {
        "snippet_distill/active_frac": float(pos.float().mean().item()),
        "snippet_distill/weight_pos_frac": float(pos.float().mean().item()),
        "snippet_distill/weight_ge_min_frac": float(ge.float().mean().item()),
        "snippet_distill/weight_mean": float(wf.mean().item()),
        "snippet_distill/weight_p50": p50,
        "snippet_distill/weight_p90": p90,
        "snippet_distill/effective_sample_size": float(ess),
    }


def compute_snippet_distill_loss(
    *,
    mu_fn: Callable[[torch.Tensor], torch.Tensor],
    feat_start: torch.Tensor,
    teacher_first_action: torch.Tensor,
    cfg: SnippetDistillConfig,
    row_weight: Optional[torch.Tensor] = None,
    mask: Optional[torch.Tensor] = None,
    min_gate: float = 0.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    """First-step snippet distillation with continuous per-row weights.

    ``row_weight`` is the authoritative continuous weight ``w_t`` (typically the
    product of the progress gate, teacher confidence, and retrieval-valid
    indicator).  ``mask`` is kept for backward compatibility: when provided it is
    combined multiplicatively with ``row_weight``.  The loss is the
    weight-normalized Huber distance::

        L = weight * clamp(sum_i w_i Huber(mu(f_i), a*_i) / (sum_i w_i + eps), cap)

    Statistics reflect the *effective* weights (not a single global scalar):
    positive-weight fraction, ``>= min_gate`` fraction, mean, p50, p90, and Kish
    effective sample size.  All quantities are finite-guarded so a NaN/inf can
    never propagate into the PPO loss.
    """
    if row_weight is None and mask is None:
        raise ValueError("compute_snippet_distill_loss requires row_weight and/or mask")

    device = feat_start.device
    B = int(feat_start.shape[0])
    if row_weight is not None:
        w = row_weight.float().to(device)
        w = torch.where(torch.isfinite(w), w, torch.zeros_like(w)).clamp(0.0, 1.0)
    else:
        w = torch.ones(B, device=device)
    if mask is not None:
        m = mask.float().to(device)
        m = torch.where(torch.isfinite(m), m, torch.zeros_like(m))
        w = w * (m > 0.5).float() if m.dtype != torch.bool else w * m.float()

    stats = _weight_stats(w, min_gate=float(min_gate))
    stats["snippet_distill/loss"] = 0.0
    stats["snippet_distill/nonfinite_count"] = 0.0

    if not cfg.enabled or B == 0 or float(w.sum().item()) <= 0.0:
        return feat_start.new_zeros(()), stats

    mu_live = mu_fn(feat_start.detach())
    teacher = teacher_first_action.detach()
    per_row = F.smooth_l1_loss(
        mu_live, teacher, beta=float(cfg.huber_delta), reduction="none"
    ).sum(dim=-1)
    finite_rows = torch.isfinite(per_row)
    stats["snippet_distill/nonfinite_count"] = float(int((~finite_rows).sum().item()))
    per_row = torch.where(finite_rows, per_row, torch.zeros_like(per_row))
    w = w * finite_rows.float()

    denom = w.sum().clamp_min(1e-6)
    loss = (per_row * w).sum() / denom
    if cfg.loss_cap > 0:
        loss = loss.clamp(max=float(cfg.loss_cap))
    loss = loss * float(cfg.weight)
    if not torch.isfinite(loss):
        loss = feat_start.new_zeros(())
    stats["snippet_distill/loss"] = float(loss.detach().item())
    # Refresh weight stats after finite masking so they match the loss weights.
    stats.update(_weight_stats(w, min_gate=float(min_gate)))
    stats["snippet_distill/loss"] = float(loss.detach().item())
    return loss, stats


def compute_snippet_activation_weight(*, cfg: SnippetDistillConfig, frames: int, tol_mean: float, nsucc_mean: float = -1.0) -> float:
    return float(
        compute_wm_activation(
            cfg=cfg.activation_config(),
            frames=int(frames),
            tol=float(tol_mean),
            nsucc_mean=float(nsucc_mean),
            eval_valid=True,
        ).weight
    )
