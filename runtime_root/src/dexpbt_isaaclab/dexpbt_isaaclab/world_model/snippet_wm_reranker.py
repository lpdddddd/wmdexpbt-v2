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


# fallback_reason_id:
# 0=ok, 1=predict_fn_none, 2=belief_missing, 3=belief_nonfinite_rows,
# 4=predict_error, 5=missing_heads, 6=nonfinite_preds,
# 7=belief_shape, 8=cand_shape_or_nonfinite, 9=pred_batch_mismatch,
# 11=alpha_zero_or_disabled, 12=partial_candidate_invalid
_REASON_OK = 0.0
_REASON_PREDICT_NONE = 1.0
_REASON_BELIEF_MISSING = 2.0
_REASON_BELIEF_NONFINITE = 3.0
_REASON_PREDICT_ERROR = 4.0
_REASON_MISSING_HEADS = 5.0
_REASON_NONFINITE_PREDS = 6.0
_REASON_BELIEF_SHAPE = 7.0
_REASON_CAND_INVALID = 8.0
_REASON_PRED_BATCH = 9.0
_REASON_ALPHA_ZERO = 11.0
_REASON_PARTIAL_INVALID = 12.0


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

    Row-level fail-closed WM rerank:
    - ``row_prediction_valid = wm_valid_mask.all(dim=1)``
    - ``row_wm_active = row_prediction_valid & (cfg.enabled and alpha_wm > 0)``
    - Active rows use ``pr_norm + alpha * wm_term - w_delta * action_delta``
    - Fallback rows use ``pr_norm`` only (no WM term / action_delta). Ranking
      matches raw ``priority.argmax`` under non-degenerate inputs, while the
      score scale stays comparable to WM-active rows for teacher confidence.

    Invariant: ``best_idx == candidate_score.argmax(dim=1)``.
    """
    B, K = priority.shape
    device = priority.device
    dtype = priority.dtype

    wm_globally_active = bool(cfg.enabled) and float(alpha_wm) > 0.0
    alpha = clamp01(float(alpha_wm)) if wm_globally_active else 0.0

    if wm_valid_mask is None:
        candidate_wm_valid = torch.ones(B, K, dtype=torch.bool, device=device)
    else:
        candidate_wm_valid = wm_valid_mask.to(device=device, dtype=torch.bool)
        if candidate_wm_valid.ndim == 1:
            candidate_wm_valid = candidate_wm_valid[:, None].expand(B, K)
        elif tuple(candidate_wm_valid.shape) != (B, K):
            # Shape mismatch => treat entire batch as prediction-invalid.
            candidate_wm_valid = torch.zeros(B, K, dtype=torch.bool, device=device)

    row_prediction_valid = candidate_wm_valid.all(dim=1)
    row_wm_active = row_prediction_valid & wm_globally_active

    # Normalize in fp32 for stable margins under AMP (bf16 priority / scores).
    pri_f = priority.float()
    pr_min = pri_f.min(dim=1, keepdim=True).values
    pr_max = pri_f.max(dim=1, keepdim=True).values
    pr_norm = (pri_f - pr_min) / (pr_max - pr_min + 1e-6)
    wm_term = (
        float(cfg.w_rec) * p_recovery.float()
        + float(cfg.w_success) * p_success.float()
        - float(cfg.w_drop) * p_drop.float()
        - float(cfg.w_unc) * uncertainty.float()
    )
    wm_term = wm_term.clamp(-float(cfg.score_cap), float(cfg.score_cap))

    wm_total_score = pr_norm + alpha * wm_term - float(cfg.w_delta) * action_delta.float()
    # Single score tensor for both selection and confidence (comparable scale).
    candidate_score = torch.where(
        row_wm_active[:, None],
        wm_total_score.to(dtype=dtype),
        pr_norm.to(dtype=dtype),
    )
    best_idx = candidate_score.argmax(dim=1)
    priority_best = priority.argmax(dim=1)

    changed = (best_idx != priority_best).float()
    n_active = int(row_wm_active.sum().item())
    changed_given_active = (
        float(changed[row_wm_active].mean().item()) if n_active > 0 else 0.0
    )
    cand_valid_frac = (
        float(candidate_wm_valid.float().mean().item()) if candidate_wm_valid.numel() else 0.0
    )
    row_pred_frac = float(row_prediction_valid.float().mean().item()) if B else 0.0
    row_active_frac = float(row_wm_active.float().mean().item()) if B else 0.0
    row_fallback_frac = float((~row_wm_active).float().mean().item()) if B else 0.0

    reason = _REASON_OK
    if not wm_globally_active:
        reason = _REASON_ALPHA_ZERO
    elif B > 0 and bool((~candidate_wm_valid).any()) and bool(candidate_wm_valid.any()):
        # Any mix of valid/invalid candidates (incl. within-row) => partial.
        reason = _REASON_PARTIAL_INVALID
    elif B > 0 and not bool(row_prediction_valid.any()):
        reason = _REASON_NONFINITE_PREDS

    wm_term_mean = 0.0
    if n_active > 0 and alpha > 0.0:
        wm_term_mean = float(wm_term[row_wm_active].mean().item())

    return best_idx, candidate_score, {
        "snippet_wm_rerank/changed_frac": float(changed.mean().item()) if B else 0.0,
        "snippet_wm_rerank/changed_frac_overall": float(changed.mean().item()) if B else 0.0,
        "snippet_wm_rerank/changed_frac_given_active": changed_given_active,
        "snippet_wm_rerank/wm_term_mean": wm_term_mean,
        "snippet_wm_rerank/candidate_valid_frac": cand_valid_frac,
        "snippet_wm_rerank/row_prediction_valid_frac": row_pred_frac,
        "snippet_wm_rerank/row_active_frac": row_active_frac,
        "snippet_wm_rerank/row_fallback_frac": row_fallback_frac,
        # Back-compat aliases used by older telemetry consumers.
        "snippet_wm_rerank/valid_frac": row_active_frac,
        "snippet_wm_rerank/fallback_frac": row_fallback_frac,
        "snippet_wm_rerank/fallback_reason_id": reason,
    }


def effective_alpha_wm(*, wm_calibration_enable: bool, alpha_wm: float) -> float:
    if not wm_calibration_enable:
        return 0.0
    return clamp01(float(alpha_wm))


def _empty_outputs(
    B: int,
    K: int,
    *,
    device,
    dtype,
    reason_id: float,
    predict_error_count: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, float]]:
    zeros = torch.zeros(B, K, device=device, dtype=dtype)
    ones_unc = torch.ones(B, K, device=device, dtype=dtype)
    invalid = torch.zeros(B, K, dtype=torch.bool, device=device)
    return zeros, zeros, zeros, ones_unc, invalid, {
        "snippet_wm_rerank/predict_error_count": float(predict_error_count),
        "snippet_wm_rerank/fallback_reason_id": float(reason_id),
        "snippet_wm_rerank/candidate_valid_frac": 0.0,
        "snippet_wm_rerank/row_prediction_valid_frac": 0.0,
        "snippet_wm_rerank/valid_frac": 0.0,
        "snippet_wm_rerank/fallback_frac": 1.0,
    }


def wm_head_probs_for_candidates(
    *,
    belief_flat: Optional[torch.Tensor],
    cand_first_actions: torch.Tensor,
    predict_fn: Optional[Callable[[torch.Tensor, torch.Tensor], dict[str, Any]]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, float]]:
    """Return ``p_recovery, p_success, p_drop, uncertainty, wm_valid_mask, diag``.

    ``wm_valid_mask`` is ``[B, K]``. Invalid entries use ``uncertainty=1`` (not 0)
    so a missed mask cannot look like high-confidence WM. Callers apply row-level
    fail-closed via ``wm_valid_mask.all(dim=1)``.
    """
    if not isinstance(cand_first_actions, torch.Tensor) or cand_first_actions.ndim != 3:
        # Cannot even infer B,K safely.
        B = int(cand_first_actions.shape[0]) if getattr(cand_first_actions, "ndim", 0) >= 1 else 1
        K = int(cand_first_actions.shape[1]) if getattr(cand_first_actions, "ndim", 0) >= 2 else 1
        device = getattr(cand_first_actions, "device", torch.device("cpu"))
        dtype = getattr(cand_first_actions, "dtype", torch.float32)
        return _empty_outputs(B, K, device=device, dtype=dtype, reason_id=_REASON_CAND_INVALID)

    B, K, _A = cand_first_actions.shape
    dev, dtype = cand_first_actions.device, cand_first_actions.dtype

    if predict_fn is None:
        return _empty_outputs(B, K, device=dev, dtype=dtype, reason_id=_REASON_PREDICT_NONE)
    if belief_flat is None:
        return _empty_outputs(B, K, device=dev, dtype=dtype, reason_id=_REASON_BELIEF_MISSING)
    if not isinstance(belief_flat, torch.Tensor) or belief_flat.ndim != 2:
        return _empty_outputs(B, K, device=dev, dtype=dtype, reason_id=_REASON_BELIEF_SHAPE)
    if int(belief_flat.shape[0]) != int(B) or int(belief_flat.shape[1]) <= 0:
        return _empty_outputs(B, K, device=dev, dtype=dtype, reason_id=_REASON_BELIEF_SHAPE)
    if not torch.isfinite(cand_first_actions).all():
        return _empty_outputs(B, K, device=dev, dtype=dtype, reason_id=_REASON_CAND_INVALID)

    belief_row_valid = torch.isfinite(belief_flat).all(dim=1)  # [B]
    belief_safe = torch.where(
        belief_row_valid[:, None],
        belief_flat,
        torch.zeros_like(belief_flat),
    )

    belief_rep = belief_safe[:, None, :].expand(B, K, belief_safe.shape[-1]).reshape(B * K, -1)
    cand_flat = cand_first_actions.reshape(B * K, -1)
    try:
        preds = predict_fn(belief_rep, cand_flat)
    except Exception:
        return _empty_outputs(
            B, K, device=dev, dtype=dtype, reason_id=_REASON_PREDICT_ERROR, predict_error_count=1.0
        )

    if not isinstance(preds, dict):
        return _empty_outputs(B, K, device=dev, dtype=dtype, reason_id=_REASON_MISSING_HEADS)

    probs = _extract_head_probs(preds, dev, dtype)
    if not all(k in probs for k in ("recovery", "success", "drop")):
        return _empty_outputs(B, K, device=dev, dtype=dtype, reason_id=_REASON_MISSING_HEADS)

    try:
        p_rec_flat = probs["recovery"].reshape(-1)
        p_succ_flat = probs["success"].reshape(-1)
        p_drop_flat = probs["drop"].reshape(-1)
    except Exception:
        return _empty_outputs(B, K, device=dev, dtype=dtype, reason_id=_REASON_MISSING_HEADS)

    need = B * K
    if (
        int(p_rec_flat.numel()) != need
        or int(p_succ_flat.numel()) != need
        or int(p_drop_flat.numel()) != need
    ):
        return _empty_outputs(B, K, device=dev, dtype=dtype, reason_id=_REASON_PRED_BATCH)

    try:
        p_rec = p_rec_flat.view(B, K)
        p_succ = p_succ_flat.view(B, K)
        p_drop = p_drop_flat.view(B, K)
    except Exception:
        return _empty_outputs(B, K, device=dev, dtype=dtype, reason_id=_REASON_PRED_BATCH)

    unc = (1.0 - (p_rec - 0.5).abs() * 2.0).clamp(0.0, 1.0)
    finite = (
        torch.isfinite(p_rec)
        & torch.isfinite(p_succ)
        & torch.isfinite(p_drop)
        & torch.isfinite(unc)
    )
    # Belief-invalid rows => all candidates invalid for that row.
    finite = finite & belief_row_valid[:, None]

    zeros = torch.zeros(B, K, device=dev, dtype=dtype)
    ones = torch.ones(B, K, device=dev, dtype=dtype)
    # Conservative uncertainty on invalid entries (not zero).
    p_rec = torch.where(finite, p_rec, zeros)
    p_succ = torch.where(finite, p_succ, zeros)
    p_drop = torch.where(finite, p_drop, zeros)
    unc = torch.where(finite, unc, ones)

    cand_valid_frac = float(finite.float().mean().item()) if finite.numel() else 0.0
    row_pred_frac = float(finite.all(dim=1).float().mean().item()) if B else 0.0
    if bool(finite.all()):
        reason = _REASON_OK
    elif bool((~belief_row_valid).any()) and bool(belief_row_valid.any()):
        reason = _REASON_BELIEF_NONFINITE
    elif not bool(belief_row_valid.any()):
        reason = _REASON_BELIEF_NONFINITE
    elif bool(finite.any()) and not bool(finite.all(dim=1).all()):
        reason = _REASON_PARTIAL_INVALID
    else:
        reason = _REASON_NONFINITE_PREDS

    diag = {
        "snippet_wm_rerank/predict_error_count": 0.0,
        "snippet_wm_rerank/fallback_reason_id": reason,
        "snippet_wm_rerank/candidate_valid_frac": cand_valid_frac,
        "snippet_wm_rerank/row_prediction_valid_frac": row_pred_frac,
        "snippet_wm_rerank/valid_frac": row_pred_frac,
        "snippet_wm_rerank/fallback_frac": float(1.0 - row_pred_frac),
    }
    return p_rec, p_succ, p_drop, unc, finite, diag
