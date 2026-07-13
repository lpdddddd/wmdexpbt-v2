"""Authoritative WM head probability extraction (runtime + calibration).

Recovery uses the single recovery head. Success/drop are averaged over the
configured ``horizon_steps`` (keys sorted numerically). Any missing required
horizon head, shape mismatch, or nonfinite value fails closed for that sample
(or the whole batch for structural errors).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

import torch

# Shared failure reason ids (wm_cal/failure_reason_id + diagnostics).
FAILURE_OK = 0.0
FAILURE_MISSING_HEAD = 1.0
FAILURE_SHAPE_MISMATCH = 2.0
FAILURE_NONFINITE = 3.0
FAILURE_HORIZON_MISMATCH = 4.0
FAILURE_INSUFFICIENT_SAMPLES = 5.0
FAILURE_PREDICT_EXCEPTION = 6.0
FAILURE_MISSING_TARGET_SOURCE = 7.0
FAILURE_INVALID_PARTITION = 8.0
FAILURE_INSUFFICIENT_HELDOUT_SAMPLES = 9.0

# Authoritative graduation-mode classification for Profile 23 readiness.
STRUCTURAL_FAILURE_IDS: frozenset[float] = frozenset(
    {
        FAILURE_MISSING_HEAD,
        FAILURE_SHAPE_MISMATCH,
        FAILURE_NONFINITE,
        FAILURE_HORIZON_MISMATCH,
        FAILURE_PREDICT_EXCEPTION,
    }
)
INSUFFICIENT_FAILURE_IDS: frozenset[float] = frozenset(
    {
        FAILURE_INSUFFICIENT_SAMPLES,
        FAILURE_INSUFFICIENT_HELDOUT_SAMPLES,
        FAILURE_MISSING_TARGET_SOURCE,
        FAILURE_INVALID_PARTITION,
    }
)


def graduation_outcome_mode(
    *,
    eval_valid: bool,
    failure_reason_id: float,
    calibration_pass: bool,
) -> str:
    """Map calibration validity / failure id to a graduation outcome mode.

    Returns one of ``structural`` / ``insufficient`` / ``valid`` / ``metric_fail``.
    """
    if not bool(eval_valid):
        fid = float(failure_reason_id)
        if fid in STRUCTURAL_FAILURE_IDS:
            return "structural"
        if fid in INSUFFICIENT_FAILURE_IDS:
            return "insufficient"
        # Unknown non-OK failure: fail closed as structural.
        if fid >= 1.0:
            return "structural"
        return "metric_fail"
    if bool(calibration_pass):
        return "valid"
    return "metric_fail"


def apply_calibration_failure(
    state: Any,
    *,
    cfg: Any,
    frames: int,
    metrics: dict[str, Any],
    failure_reason_id: float,
) -> Any:
    """Unify calibration failure → graduation outcome mode + evaluator apply."""
    from .wm_calibration_evaluator import apply_calibration_outcome

    mode = graduation_outcome_mode(
        eval_valid=False,
        failure_reason_id=float(failure_reason_id),
        calibration_pass=False,
    )
    metrics = dict(metrics)
    metrics["wm_cal/failure_reason_id"] = float(failure_reason_id)
    metrics["wm_cal/outcome_mode"] = mode
    metrics["wm_cal/eval_valid"] = 0.0
    metrics.setdefault("wm_cal/valid_prediction_frac", 0.0)
    return apply_calibration_outcome(
        state,
        cfg=cfg,
        frames=frames,
        metrics=metrics,
        mode=mode,
        failure_reason_id=float(failure_reason_id),
    )


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


def normalize_horizon_steps(
    horizon_steps: Optional[Sequence[int]] = None,
    *,
    default: tuple[int, ...] = (4, 8),
) -> tuple[int, ...]:
    """Return positive horizon ints sorted ascending (stable, insertion-agnostic)."""
    if horizon_steps is None:
        steps = list(default)
    else:
        steps = [int(h) for h in horizon_steps if int(h) > 0]
    if not steps:
        steps = list(default)
    return tuple(sorted(set(steps)))


def infer_horizon_steps_from_preds(preds: Any) -> tuple[int, ...]:
    """Horizons that expose both success and drop heads, sorted numerically."""
    steps: list[int] = []
    if not isinstance(preds, dict):
        return tuple()
    h_logits = preds.get("horizon_logits")
    if not isinstance(h_logits, dict):
        return tuple()
    for k, v in h_logits.items():
        try:
            hk = int(k)
        except (TypeError, ValueError):
            continue
        if isinstance(v, dict) and "success" in v and "drop" in v:
            steps.append(hk)
    return tuple(sorted(set(steps)))


@dataclass
class WMHeadProbResult:
    """Shared head-probability bundle for reranker and calibration."""

    p_recovery: torch.Tensor
    p_success: torch.Tensor
    p_drop: torch.Tensor
    uncertainty: torch.Tensor
    valid_mask: torch.Tensor
    diagnostics: dict[str, float] = field(default_factory=dict)
    success_stack: Optional[torch.Tensor] = None  # [H, N]
    drop_stack: Optional[torch.Tensor] = None
    structural_ok: bool = False

    @property
    def ok(self) -> bool:
        return bool(self.structural_ok)


def _empty_result(
    n: int,
    *,
    device,
    dtype,
    reason: float,
    extra: Optional[dict[str, float]] = None,
) -> WMHeadProbResult:
    zeros = torch.zeros(n, device=device, dtype=dtype)
    ones = torch.ones(n, device=device, dtype=dtype)
    invalid = torch.zeros(n, device=device, dtype=torch.bool)
    diag = {
        "wm_cal/missing_head": 1.0 if reason == FAILURE_MISSING_HEAD else 0.0,
        "wm_cal/shape_mismatch": 1.0 if reason == FAILURE_SHAPE_MISMATCH else 0.0,
        "wm_cal/horizon_mismatch": 1.0 if reason == FAILURE_HORIZON_MISMATCH else 0.0,
        "wm_cal/nonfinite_prediction": 1.0 if reason == FAILURE_NONFINITE else 0.0,
        "wm_cal/failure_reason_id": float(reason),
        "wm_cal/eval_valid": 0.0,
    }
    if extra:
        diag.update(extra)
    return WMHeadProbResult(
        p_recovery=zeros,
        p_success=zeros.clone(),
        p_drop=zeros.clone(),
        uncertainty=ones,
        valid_mask=invalid,
        diagnostics=diag,
        structural_ok=False,
    )


def extract_required_wm_head_probs(
    preds: Any,
    *,
    horizon_steps: Sequence[int],
    expected_batch: Optional[int] = None,
    device=None,
    dtype=None,
) -> WMHeadProbResult:
    """Extract recovery + mean success/drop over configured horizons.

    Semantics
    ---------
    - recovery: single ``recovery_logits`` head (not horizon-averaged).
    - success/drop: for each ``h`` in sorted ``horizon_steps``, require both
      ``horizon_logits[h]['success']`` and ``['drop']``, then mean over H.
    - valid_mask: True only when all required tensors are finite and shapes match.
    - Nonfinite entries get uncertainty=1 and valid_mask=False (fail-closed).
    - Structural failure (missing head / horizon / shape) returns ``structural_ok=False``
      with all rows invalid.
    """
    steps = normalize_horizon_steps(horizon_steps)
    if not isinstance(preds, dict):
        n = int(expected_batch) if expected_batch is not None else 0
        dev = device if device is not None else torch.device("cpu")
        dt = dtype if dtype is not None else torch.float32
        return _empty_result(n, device=dev, dtype=dt, reason=FAILURE_MISSING_HEAD)

    rec = preds.get("recovery_logits")
    if rec is None:
        n = int(expected_batch) if expected_batch is not None else 0
        if hasattr(rec, "shape"):
            pass
        # Infer N from any horizon logit if possible.
        h_logits = preds.get("horizon_logits")
        if isinstance(h_logits, dict) and h_logits:
            sample = next(iter(h_logits.values()))
            if isinstance(sample, dict):
                for v in sample.values():
                    if isinstance(v, torch.Tensor):
                        n = int(v.reshape(-1).shape[0])
                        device = v.device
                        dtype = v.dtype
                        break
        dev = device if device is not None else torch.device("cpu")
        dt = dtype if dtype is not None else torch.float32
        return _empty_result(max(n, 0), device=dev, dtype=dt, reason=FAILURE_MISSING_HEAD)

    if not isinstance(rec, torch.Tensor):
        n = int(expected_batch) if expected_batch is not None else 0
        dev = device if device is not None else torch.device("cpu")
        dt = dtype if dtype is not None else torch.float32
        return _empty_result(n, device=dev, dtype=dt, reason=FAILURE_MISSING_HEAD)

    p_rec = torch.sigmoid(rec.float()).reshape(-1)
    n = int(p_rec.shape[0])
    dev = p_rec.device if device is None else device
    dt = p_rec.dtype if dtype is None else dtype
    if expected_batch is not None and int(expected_batch) != n:
        return _empty_result(
            int(expected_batch),
            device=dev,
            dtype=dt,
            reason=FAILURE_SHAPE_MISMATCH,
        )

    h_logits = preds.get("horizon_logits")
    if not isinstance(h_logits, dict) or not h_logits:
        return _empty_result(n, device=dev, dtype=dt, reason=FAILURE_MISSING_HEAD)

    # Sort horizon keys numerically; never rely on dict insertion order.
    available = {}
    for k, v in h_logits.items():
        try:
            available[int(k)] = v
        except (TypeError, ValueError):
            continue

    succ_list: list[torch.Tensor] = []
    drop_list: list[torch.Tensor] = []
    for h in steps:
        if h not in available:
            return _empty_result(
                n,
                device=dev,
                dtype=dt,
                reason=FAILURE_HORIZON_MISMATCH,
                extra={"wm_cal/missing_horizon": float(h)},
            )
        head_out = available[h]
        if not isinstance(head_out, dict):
            return _empty_result(n, device=dev, dtype=dt, reason=FAILURE_MISSING_HEAD)
        if "success" not in head_out or "drop" not in head_out:
            return _empty_result(n, device=dev, dtype=dt, reason=FAILURE_MISSING_HEAD)
        s = head_out["success"]
        d = head_out["drop"]
        if not isinstance(s, torch.Tensor) or not isinstance(d, torch.Tensor):
            return _empty_result(n, device=dev, dtype=dt, reason=FAILURE_MISSING_HEAD)
        s_flat = torch.sigmoid(s.float()).reshape(-1)
        d_flat = torch.sigmoid(d.float()).reshape(-1)
        if int(s_flat.shape[0]) != n or int(d_flat.shape[0]) != n:
            return _empty_result(n, device=dev, dtype=dt, reason=FAILURE_SHAPE_MISMATCH)
        succ_list.append(s_flat.to(device=dev, dtype=dt))
        drop_list.append(d_flat.to(device=dev, dtype=dt))

    success_stack = torch.stack(succ_list, dim=0)  # [H, N]
    drop_stack = torch.stack(drop_list, dim=0)
    p_succ = success_stack.mean(dim=0)
    p_drop = drop_stack.mean(dim=0)
    p_rec = p_rec.to(device=dev, dtype=dt)

    finite = (
        torch.isfinite(p_rec)
        & torch.isfinite(p_succ)
        & torch.isfinite(p_drop)
        & torch.isfinite(success_stack).all(dim=0)
        & torch.isfinite(drop_stack).all(dim=0)
    )
    zeros = torch.zeros(n, device=dev, dtype=dt)
    ones = torch.ones(n, device=dev, dtype=dt)
    p_rec_safe = torch.where(finite, p_rec, zeros)
    p_succ_safe = torch.where(finite, p_succ, zeros)
    p_drop_safe = torch.where(finite, p_drop, zeros)
    unc = combined_head_uncertainty(p_rec_safe, p_succ_safe, p_drop_safe)
    unc = torch.where(finite, unc, ones)

    reason = FAILURE_OK
    if not bool(finite.any()):
        reason = FAILURE_NONFINITE
    elif not bool(finite.all()):
        reason = FAILURE_NONFINITE

    diag = {
        "wm_cal/missing_head": 0.0,
        "wm_cal/shape_mismatch": 0.0,
        "wm_cal/horizon_mismatch": 0.0,
        "wm_cal/nonfinite_prediction": 0.0 if bool(finite.all()) else 1.0,
        "wm_cal/failure_reason_id": float(reason) if reason != FAILURE_OK else FAILURE_OK,
        "wm_cal/eval_valid": 1.0 if bool(finite.any()) else 0.0,
        "wm_cal/horizon_count": float(len(steps)),
        "wm_cal/valid_prediction_frac": float(finite.float().mean().item()) if n else 0.0,
    }
    return WMHeadProbResult(
        p_recovery=p_rec_safe,
        p_success=p_succ_safe,
        p_drop=p_drop_safe,
        uncertainty=unc,
        valid_mask=finite,
        diagnostics=diag,
        success_stack=success_stack,
        drop_stack=drop_stack,
        structural_ok=True,
    )


def extract_head_probs_compat(
    preds: dict[str, Any],
    device,
    dtype,
    *,
    horizon_steps: Optional[Sequence[int]] = None,
) -> dict[str, torch.Tensor]:
    """Backward-compatible dict view used by RSSM planner / legacy callers.

    When ``horizon_steps`` is None, averages over all numeric horizon keys that
    provide both success and drop (sorted). Structural failures return {}.
    """
    if horizon_steps is None:
        h_logits = preds.get("horizon_logits") if isinstance(preds, dict) else None
        steps: list[int] = []
        if isinstance(h_logits, dict):
            for k, v in h_logits.items():
                try:
                    hk = int(k)
                except (TypeError, ValueError):
                    continue
                if isinstance(v, dict) and "success" in v and "drop" in v:
                    steps.append(hk)
        horizon_steps = tuple(sorted(set(steps))) if steps else (4, 8)

    result = extract_required_wm_head_probs(
        preds, horizon_steps=horizon_steps, device=device, dtype=dtype
    )
    if not result.structural_ok:
        return {}
    out: dict[str, torch.Tensor] = {
        "recovery": result.p_recovery,
        "success": result.p_success,
        "drop": result.p_drop,
        "uncertainty": result.uncertainty,
        "valid_mask": result.valid_mask.float(),
    }
    if result.success_stack is not None:
        out["success_stack"] = result.success_stack
    if result.drop_stack is not None:
        out["drop_stack"] = result.drop_stack
    return out
