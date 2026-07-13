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
    wm_valid_mask: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    """Return ``(best_idx, candidate_score, diagnostics)``.

    ``candidate_score`` is the per-candidate total used for ``argmax`` (and must
    also be used for teacher confidence). Rows with no valid WM candidates fall
    back to raw ``priority`` selection and scores so invalid WM never looks like
    ``uncertainty=0`` high-confidence predictions.
    """
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

    if wm_valid_mask is None:
        cand_valid = torch.ones(B, K, dtype=torch.bool, device=priority.device)
    else:
        cand_valid = wm_valid_mask.to(device=priority.device, dtype=torch.bool)
        if cand_valid.ndim == 1:
            cand_valid = cand_valid[:, None].expand(B, K)
    row_wm_ok = cand_valid.any(dim=1)
    valid_f = cand_valid.to(dtype=pr_norm.dtype)

    # Invalid candidates contribute no WM term (neutral), but are excluded from
    # WM-driven selection via -inf so they cannot win through fake zero unc.
    total = pr_norm + alpha * wm_term * valid_f - float(cfg.w_delta) * action_delta
    select_score = total.masked_fill(~cand_valid, -1e9)
    wm_best = select_score.argmax(dim=1)
    priority_best = priority.argmax(dim=1)
    best_idx = torch.where(row_wm_ok, wm_best, priority_best)

    # Confidence / telemetry must read the same score family as selection.
    candidate_score = torch.where(row_wm_ok[:, None], total, priority)

    changed = (best_idx != priority_best).float()
    # Only count changes on rows that actually used WM.
    changed_wm = torch.where(row_wm_ok, changed, torch.zeros_like(changed))
    valid_frac = float(row_wm_ok.float().mean().item()) if B else 0.0
    fallback_frac = float((~row_wm_ok).float().mean().item()) if B else 0.0
    wm_term_used = wm_term * valid_f
    return best_idx, candidate_score, {
        "snippet_wm_rerank/changed_frac": float(changed_wm.mean().item()) if B else 0.0,
        "snippet_wm_rerank/wm_term_mean": (
            float(wm_term_used.mean().item()) if alpha > 0 and bool(row_wm_ok.any()) else 0.0
        ),
        "snippet_wm_rerank/valid_frac": valid_frac,
        "snippet_wm_rerank/fallback_frac": fallback_frac,
    }


def effective_alpha_wm(*, wm_calibration_enable: bool, alpha_wm: float) -> float:
    if not wm_calibration_enable:
        return 0.0
    return clamp01(float(alpha_wm))


def wm_head_probs_for_candidates(
    *,
    belief_flat: Optional[torch.Tensor],
    cand_first_actions: torch.Tensor,
    predict_fn: Optional[Callable[[torch.Tensor, torch.Tensor], dict[str, Any]]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, float]]:
    """Return ``p_recovery, p_success, p_drop, uncertainty, wm_valid_mask, diag``.

    ``wm_valid_mask`` has shape ``[B, K]``. Invalid entries must not be treated as
    ``uncertainty=0`` reliable predictions: callers fall back to snippet priority.
    """
    B, K, _A = cand_first_actions.shape
    dev, dtype = cand_first_actions.device, cand_first_actions.dtype
    zeros = torch.zeros(B, K, device=dev, dtype=dtype)
    invalid = torch.zeros(B, K, dtype=torch.bool, device=dev)
    base_diag = {
        "snippet_wm_rerank/predict_error_count": 0.0,
        "snippet_wm_rerank/fallback_reason_id": 0.0,
        "snippet_wm_rerank/valid_frac": 0.0,
        "snippet_wm_rerank/fallback_frac": 1.0,
    }
    # reason_id: 0=ok, 1=predict_fn_none, 2=belief_missing, 3=belief_nonfinite,
    # 4=predict_error, 5=missing_heads, 6=nonfinite_preds
    if predict_fn is None:
        base_diag["snippet_wm_rerank/fallback_reason_id"] = 1.0
        return zeros, zeros, zeros, zeros, invalid, base_diag
    if belief_flat is None:
        base_diag["snippet_wm_rerank/fallback_reason_id"] = 2.0
        return zeros, zeros, zeros, zeros, invalid, base_diag
    if not torch.isfinite(belief_flat).all():
        base_diag["snippet_wm_rerank/fallback_reason_id"] = 3.0
        return zeros, zeros, zeros, zeros, invalid, base_diag

    belief_rep = belief_flat[:, None, :].expand(B, K, belief_flat.shape[-1]).reshape(B * K, -1)
    cand_flat = cand_first_actions.reshape(B * K, -1)
    try:
        preds = predict_fn(belief_rep, cand_flat)
    except Exception:
        base_diag["snippet_wm_rerank/predict_error_count"] = 1.0
        base_diag["snippet_wm_rerank/fallback_reason_id"] = 4.0
        return zeros, zeros, zeros, zeros, invalid, base_diag

    if not isinstance(preds, dict):
        base_diag["snippet_wm_rerank/fallback_reason_id"] = 5.0
        return zeros, zeros, zeros, zeros, invalid, base_diag

    probs = _extract_head_probs(preds, dev, dtype)
    # Require the heads used by the WM term; missing heads => invalid (not zero).
    if not all(k in probs for k in ("recovery", "success", "drop")):
        base_diag["snippet_wm_rerank/fallback_reason_id"] = 5.0
        return zeros, zeros, zeros, zeros, invalid, base_diag

    try:
        p_rec = probs["recovery"].view(B, K)
        p_succ = probs["success"].view(B, K)
        p_drop = probs["drop"].view(B, K)
    except Exception:
        base_diag["snippet_wm_rerank/fallback_reason_id"] = 5.0
        return zeros, zeros, zeros, zeros, invalid, base_diag

    unc = (1.0 - (p_rec - 0.5).abs() * 2.0).clamp(0.0, 1.0)
    finite = (
        torch.isfinite(p_rec)
        & torch.isfinite(p_succ)
        & torch.isfinite(p_drop)
        & torch.isfinite(unc)
    )
    # Neutralize invalid candidates so they cannot leak into scoring if a caller
    # forgets the mask; the mask remains the source of truth.
    p_rec = torch.where(finite, p_rec, zeros)
    p_succ = torch.where(finite, p_succ, zeros)
    p_drop = torch.where(finite, p_drop, zeros)
    unc = torch.where(finite, unc, zeros)

    valid_frac = float(finite.float().mean().item()) if finite.numel() else 0.0
    diag = {
        "snippet_wm_rerank/predict_error_count": 0.0,
        "snippet_wm_rerank/fallback_reason_id": 0.0 if bool(finite.any()) else 6.0,
        "snippet_wm_rerank/valid_frac": valid_frac,
        "snippet_wm_rerank/fallback_frac": float(1.0 - valid_frac),
    }
    return p_rec, p_succ, p_drop, unc, finite, diag
