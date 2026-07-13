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


def normalize_priority_scores(priority: torch.Tensor) -> torch.Tensor:
    """Row-wise min-max normalize priority for selection + teacher confidence.

    Computed in fp32 for AMP stability:

    - Ordinary finite non-degenerate rows → strict ``[0, 1]`` min-max.
    - Finite constant rows → all finite entries ``1`` (outrank invalids).
    - Partial non-finite rows → non-finite ``0``; finite constants ``1``.
    - All-non-finite rows → all zeros (deterministic ``argmax`` → index 0).

    For ordinary finite non-degenerate inputs, ``argmax`` matches raw
    ``priority.argmax``. Affine transforms of finite priorities leave the
    normalized scores unchanged (exact span denominator, not ``span + eps``).
    """
    pri_f = priority.float()
    finite = torch.isfinite(pri_f)
    row_has_valid = finite.any(dim=1, keepdim=True)
    row_min = torch.where(
        finite, pri_f, torch.full_like(pri_f, float("inf"))
    ).min(dim=1, keepdim=True).values
    row_max = torch.where(
        finite, pri_f, torch.full_like(pri_f, float("-inf"))
    ).max(dim=1, keepdim=True).values
    row_min = torch.where(row_has_valid, row_min, torch.zeros_like(row_min))
    row_max = torch.where(row_has_valid, row_max, torch.zeros_like(row_max))
    span = row_max - row_min
    has_span = row_has_valid & (span > 0.0)
    safe_span = torch.where(has_span, span, torch.ones_like(span))
    normalized = (pri_f - row_min) / safe_span
    normalized = torch.where(finite & has_span, normalized, torch.zeros_like(normalized))
    # Degenerate but at least one finite candidate: finite outranks invalid.
    normalized = torch.where(
        finite & row_has_valid & (~has_span),
        torch.ones_like(normalized),
        normalized,
    )
    normalized = torch.where(finite, normalized, torch.zeros_like(normalized))
    normalized = torch.where(row_has_valid, normalized, torch.zeros_like(normalized))
    return normalized.to(dtype=priority.dtype)


def mean_squared_action_delta(
    cand_first: torch.Tensor,
    actor_action: torch.Tensor,
) -> torch.Tensor:
    """Per-candidate mean squared action delta ``[B, K]`` (action-dim invariant)."""
    return ((cand_first.float() - actor_action.float()[:, None, :]) ** 2).mean(dim=-1)


def binary_head_uncertainty(prob: torch.Tensor) -> torch.Tensor:
    """Uncertainty in ``[0, 1]``: high when probability is near 0.5."""
    return (1.0 - (prob.float() - 0.5).abs() * 2.0).clamp(0.0, 1.0)


def combined_head_uncertainty(
    p_recovery: torch.Tensor,
    p_success: torch.Tensor,
    p_drop: torch.Tensor,
) -> torch.Tensor:
    """Max uncertainty across recovery/success/drop (conservative aggregate)."""
    return torch.stack(
        [
            binary_head_uncertainty(p_recovery),
            binary_head_uncertainty(p_success),
            binary_head_uncertainty(p_drop),
        ],
        dim=0,
    ).amax(dim=0)


def compute_snippet_wm_term(
    *,
    p_recovery: torch.Tensor,
    p_success: torch.Tensor,
    p_drop: torch.Tensor,
    uncertainty: torch.Tensor,
    w_rec: float,
    w_success: float,
    w_drop: float,
    w_unc: float,
    score_cap: float,
) -> torch.Tensor:
    """Shared capped WM term used by snippet reranker and calibration."""
    term = (
        float(w_rec) * p_recovery.float()
        + float(w_success) * p_success.float()
        - float(w_drop) * p_drop.float()
        - float(w_unc) * uncertainty.float()
    )
    return term.clamp(-float(score_cap), float(score_cap))


def compute_realized_wm_utility(
    *,
    y_recovery: torch.Tensor,
    y_success: torch.Tensor,
    y_drop: torch.Tensor,
    w_rec: float,
    w_success: float,
    w_drop: float,
) -> torch.Tensor:
    """Realized utility aligned with WM term direction (no uncertainty term)."""
    return (
        float(w_rec) * y_recovery.float()
        + float(w_success) * y_success.float()
        - float(w_drop) * y_drop.float()
    )


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
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float], torch.Tensor]:
    """Return ``(best_idx, candidate_score, diagnostics, row_wm_active)``.

    Row-level fail-closed WM rerank:
    - ``row_prediction_valid = candidate_valid.all(dim=1)``
    - ``row_wm_active = row_prediction_valid & (cfg.enabled and alpha_wm > 0)``
    - Active rows use ``pr_norm + alpha * (wm_term - w_delta * action_delta)``
    - Fallback rows use ``pr_norm`` only (no WM term / action_delta). Ranking
      matches raw ``priority.argmax`` under ordinary finite non-degenerate
      inputs, while the score scale stays comparable for teacher confidence.

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
            candidate_wm_valid = torch.zeros(B, K, dtype=torch.bool, device=device)

    score_input_valid = (
        torch.isfinite(priority)
        & torch.isfinite(p_recovery)
        & torch.isfinite(p_success)
        & torch.isfinite(p_drop)
        & torch.isfinite(uncertainty)
        & torch.isfinite(action_delta)
    )
    candidate_valid = candidate_wm_valid & score_input_valid

    row_prediction_valid = candidate_valid.all(dim=1)
    row_wm_active = row_prediction_valid & wm_globally_active

    pr_norm = normalize_priority_scores(priority).float()
    wm_term = compute_snippet_wm_term(
        p_recovery=p_recovery,
        p_success=p_success,
        p_drop=p_drop,
        uncertainty=uncertainty,
        w_rec=float(cfg.w_rec),
        w_success=float(cfg.w_success),
        w_drop=float(cfg.w_drop),
        w_unc=float(cfg.w_unc),
        score_cap=float(cfg.score_cap),
    )
    action_delta_f = action_delta.float()
    action_penalty = float(cfg.w_delta) * action_delta_f

    # Alpha-continuous: alpha=0 ⇒ exact pr_norm; action penalty scales with alpha.
    wm_total_score = pr_norm + alpha * (wm_term - action_penalty)
    candidate_score = torch.where(
        row_wm_active[:, None],
        wm_total_score.to(dtype=dtype),
        pr_norm.to(dtype=dtype),
    )
    best_idx = candidate_score.argmax(dim=1)
    priority_best = pr_norm.argmax(dim=1)

    changed = (best_idx != priority_best).float()
    n_active = int(row_wm_active.sum().item())
    changed_given_active = (
        float(changed[row_wm_active].mean().item()) if n_active > 0 else 0.0
    )
    cand_valid_frac = (
        float(candidate_valid.float().mean().item()) if candidate_valid.numel() else 0.0
    )
    row_pred_frac = float(row_prediction_valid.float().mean().item()) if B else 0.0
    row_active_frac = float(row_wm_active.float().mean().item()) if B else 0.0
    row_fallback_frac = float((~row_wm_active).float().mean().item()) if B else 0.0

    row_any_valid = candidate_valid.any(dim=1)
    row_all_valid = candidate_valid.all(dim=1)
    row_partial = row_any_valid & (~row_all_valid)

    reason = _REASON_OK
    if not wm_globally_active:
        reason = _REASON_ALPHA_ZERO
    elif B > 0 and bool(row_partial.any()):
        reason = _REASON_PARTIAL_INVALID
    elif B > 0 and bool((~row_all_valid).any()):
        reason = _REASON_NONFINITE_PREDS

    wm_term_mean = 0.0
    action_delta_mean = 0.0
    action_penalty_mean = 0.0
    if n_active > 0 and alpha > 0.0:
        wm_term_mean = float(wm_term[row_wm_active].mean().item())
        action_delta_mean = float(action_delta_f[row_wm_active].mean().item())
        action_penalty_mean = float(action_penalty[row_wm_active].mean().item())

    return best_idx, candidate_score, {
        "snippet_wm_rerank/changed_frac": float(changed.mean().item()) if B else 0.0,
        "snippet_wm_rerank/changed_frac_overall": float(changed.mean().item()) if B else 0.0,
        "snippet_wm_rerank/changed_frac_given_active": changed_given_active,
        "snippet_wm_rerank/wm_term_mean": wm_term_mean,
        "snippet_wm_rerank/action_delta_mean": action_delta_mean,
        "snippet_wm_rerank/action_penalty_mean": action_penalty_mean,
        "snippet_wm_rerank/candidate_valid_frac": cand_valid_frac,
        "snippet_wm_rerank/row_prediction_valid_frac": row_pred_frac,
        "snippet_wm_rerank/row_active_frac": row_active_frac,
        "snippet_wm_rerank/row_fallback_frac": row_fallback_frac,
        "snippet_wm_rerank/valid_frac": row_active_frac,
        "snippet_wm_rerank/fallback_frac": row_fallback_frac,
        "snippet_wm_rerank/fallback_reason_id": reason,
    }, row_wm_active


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

    # Per-row action validity: one bad row must not invalidate the whole batch.
    cand_row_valid = torch.isfinite(cand_first_actions).all(dim=(1, 2))  # [B]
    cand_safe = torch.where(
        cand_row_valid[:, None, None],
        cand_first_actions,
        torch.zeros_like(cand_first_actions),
    )

    belief_row_valid = torch.isfinite(belief_flat).all(dim=1)  # [B]
    belief_safe = torch.where(
        belief_row_valid[:, None],
        belief_flat,
        torch.zeros_like(belief_flat),
    )

    belief_rep = belief_safe[:, None, :].expand(B, K, belief_safe.shape[-1]).reshape(B * K, -1)
    cand_flat = cand_safe.reshape(B * K, -1)
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

    unc = combined_head_uncertainty(p_rec, p_succ, p_drop)
    finite = (
        torch.isfinite(p_rec)
        & torch.isfinite(p_succ)
        & torch.isfinite(p_drop)
        & torch.isfinite(unc)
    )
    # Belief-invalid or action-invalid rows => all candidates invalid for that row.
    finite = finite & belief_row_valid[:, None] & cand_row_valid[:, None]

    zeros = torch.zeros(B, K, device=dev, dtype=dtype)
    ones = torch.ones(B, K, device=dev, dtype=dtype)
    p_rec = torch.where(finite, p_rec, zeros)
    p_succ = torch.where(finite, p_succ, zeros)
    p_drop = torch.where(finite, p_drop, zeros)
    unc = torch.where(finite, unc, ones)

    cand_valid_frac = float(finite.float().mean().item()) if finite.numel() else 0.0
    row_all_valid = finite.all(dim=1)
    row_any_valid = finite.any(dim=1)
    row_partial = row_any_valid & (~row_all_valid)
    row_pred_frac = float(row_all_valid.float().mean().item()) if B else 0.0

    if bool(finite.all()):
        reason = _REASON_OK
    elif bool((~belief_row_valid).any()) and not bool((~cand_row_valid).any()):
        # Prefer belief reason when belief rows are the only structural issue.
        reason = _REASON_BELIEF_NONFINITE
    elif bool((~cand_row_valid).any()) and bool(cand_row_valid.any()) and not bool(row_partial.any()):
        # Whole bad-action rows mixed with good rows (no within-row action partial).
        reason = _REASON_CAND_INVALID
    elif bool(row_partial.any()):
        reason = _REASON_PARTIAL_INVALID
    elif not bool(belief_row_valid.any()):
        reason = _REASON_BELIEF_NONFINITE
    elif not bool(cand_row_valid.any()):
        reason = _REASON_CAND_INVALID
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
