"""WM-guided snippet reranking (CSER-v0)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional

import torch

from .rssm_mppi_planner import _extract_head_probs
from .wm_activation_gate import clamp01


@dataclass
class SnippetWMRerankConfig:
    enabled: bool = False
    w_rec: float = 0.3
    w_success: float = 0.2
    w_drop: float = 0.4
    w_unc: float = 0.1
    w_delta: float = 0.05
    score_cap: float = 0.25


def score_snippet_candidates(
    *,
    priority: torch.Tensor,
    p_recovery: torch.Tensor,
    p_success: torch.Tensor,
    p_drop: torch.Tensor,
    uncertainty: torch.Tensor,
    action_delta: torch.Tensor,
    alpha_wm: float,
    cfg: SnippetWMRerankConfig,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Return per-row best candidate index and diagnostics."""
    B, K = priority.shape
    pr_min = priority.min(dim=1, keepdim=True).values
    pr_max = priority.max(dim=1, keepdim=True).values
    pr_norm = (priority - pr_min) / (pr_max - pr_min + 1e-6)
    wm_term = (
        float(cfg.w_rec) * p_recovery
        + float(cfg.w_success) * p_success
        - float(cfg.w_drop) * p_drop
        - float(cfg.w_unc) * uncertainty
    )
    cap = float(cfg.score_cap)
    wm_term = wm_term.clamp(-cap, cap)
    alpha = clamp01(float(alpha_wm)) if cfg.enabled else 0.0
    total = pr_norm + alpha * wm_term - float(cfg.w_delta) * action_delta
    best_idx = total.argmax(dim=1)
    priority_best = priority.argmax(dim=1)
    changed = (best_idx != priority_best).float().mean().item()
    return best_idx, {
        "snippet_wm_rerank/changed_frac": float(changed),
        "snippet_wm_rerank/wm_term_mean": float(wm_term.mean().item()) if alpha > 0 else 0.0,
    }


def effective_alpha_wm(*, wm_calibration_enable: bool, alpha_wm: float) -> float:
    if not wm_calibration_enable:
        return 0.0
    return clamp01(float(alpha_wm))


def wm_head_probs_for_candidates(
    *,
    belief_flat: torch.Tensor,
    cand_first_actions: torch.Tensor,
    predict_fn: Optional[Callable[[torch.Tensor, torch.Tensor], dict[str, Any]]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return ``p_recovery, p_success, p_drop, uncertainty`` with shape ``[B, K]``."""
    B, K, _A = cand_first_actions.shape
    dev, dtype = cand_first_actions.device, cand_first_actions.dtype
    zeros = torch.zeros(B, K, device=dev, dtype=dtype)
    if predict_fn is None or belief_flat is None:
        return zeros, zeros, zeros, zeros
    belief_rep = belief_flat[:, None, :].expand(B, K, belief_flat.shape[-1]).reshape(B * K, -1)
    cand_flat = cand_first_actions.reshape(B * K, -1)
    try:
        preds = predict_fn(belief_rep, cand_flat)
    except Exception:
        return zeros, zeros, zeros, zeros
    probs = _extract_head_probs(preds, dev, dtype)
    p_rec = probs.get("recovery", torch.zeros(B * K, device=dev, dtype=dtype)).view(B, K)
    p_succ = probs.get("success", torch.zeros(B * K, device=dev, dtype=dtype)).view(B, K)
    p_drop = probs.get("drop", torch.zeros(B * K, device=dev, dtype=dtype)).view(B, K)
    unc = 1.0 - (p_rec - 0.5).abs() * 2.0
    return p_rec, p_succ, p_drop, unc.clamp(0.0, 1.0)
