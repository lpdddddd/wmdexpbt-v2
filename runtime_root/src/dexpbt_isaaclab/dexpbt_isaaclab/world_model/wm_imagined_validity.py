"""Calibration-aware imagined-validity helpers (WM Head Rescue patch)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ImaginedValidityResult:
    imagined_valid: bool
    imagined_valid_strict: bool
    imagined_valid_soft: bool
    imagined_head_has_signal: bool
    imagined_valid_mode: str
    imagined_valid_reason: str
    wm_validity_weight: float


def compute_head_has_signal(
    *,
    logit_abs_mean: float,
    success_prob_std: float,
    drop_risk_std: float,
    recovery_prob_std: float,
    min_logit_abs_mean: float,
    min_head_signal_std: float,
) -> bool:
    return (
        float(logit_abs_mean) >= float(min_logit_abs_mean)
        or float(success_prob_std) >= float(min_head_signal_std)
        or float(drop_risk_std) >= float(min_head_signal_std)
        or float(recovery_prob_std) >= float(min_head_signal_std)
    )


def compute_imagined_validity(
    *,
    uncertainty: float,
    logit_abs_mean: float = 0.0,
    success_prob_std: float = 0.0,
    drop_risk_std: float = 0.0,
    recovery_prob_std: float = 0.0,
    valid_mode: str = "strict_uncertainty",
    strict_uncertainty_threshold: float = 0.5,
    soft_uncertainty_threshold: float = 0.95,
    min_logit_abs_mean: float = 0.05,
    min_head_signal_std: float = 0.02,
    soft_validity_weight: float = 0.25,
    stale: bool = False,
    stale_reason: str = "",
) -> ImaginedValidityResult:
    mode = str(valid_mode or "strict_uncertainty")
    if stale or stale_reason in ("stale_wm_imagined", "missing", "no_payload"):
        return ImaginedValidityResult(
            imagined_valid=False,
            imagined_valid_strict=False,
            imagined_valid_soft=False,
            imagined_head_has_signal=False,
            imagined_valid_mode=mode,
            imagined_valid_reason=stale_reason or "stale",
            wm_validity_weight=0.0,
        )
    head_has_signal = compute_head_has_signal(
        logit_abs_mean=logit_abs_mean,
        success_prob_std=success_prob_std,
        drop_risk_std=drop_risk_std,
        recovery_prob_std=recovery_prob_std,
        min_logit_abs_mean=min_logit_abs_mean,
        min_head_signal_std=min_head_signal_std,
    )
    strict_valid = float(uncertainty) <= float(strict_uncertainty_threshold)
    soft_valid = (
        float(uncertainty) <= float(soft_uncertainty_threshold) and head_has_signal
    )
    if mode == "shadow_only":
        imagined_valid = False
        reason = "shadow_only"
        weight = 0.0
    elif mode == "soft_calibrated":
        imagined_valid = bool(strict_valid or soft_valid)
        if strict_valid:
            reason = "strict_valid"
            weight = 1.0
        elif soft_valid:
            reason = "soft_valid"
            weight = float(soft_validity_weight)
        elif not head_has_signal:
            reason = "no_head_signal"
            weight = 0.0
        else:
            reason = "high_uncertainty"
            weight = 0.0
    else:
        imagined_valid = bool(strict_valid)
        reason = "strict_valid" if strict_valid else "strict_invalid"
        weight = 1.0 if strict_valid else 0.0
    return ImaginedValidityResult(
        imagined_valid=imagined_valid,
        imagined_valid_strict=bool(strict_valid),
        imagined_valid_soft=bool(soft_valid),
        imagined_head_has_signal=bool(head_has_signal),
        imagined_valid_mode=mode,
        imagined_valid_reason=reason,
        wm_validity_weight=float(weight),
    )


def compute_wm_pbt_validity_weight(
    validity: ImaginedValidityResult,
) -> tuple[float, bool, bool]:
    """Return (weight, used_strict, used_soft)."""
    if validity.wm_validity_weight <= 0.0:
        return 0.0, False, False
    if validity.imagined_valid_strict:
        return 1.0, True, False
    if validity.imagined_valid_soft:
        return float(validity.wm_validity_weight), False, True
    return 0.0, False, False


__all__ = [
    "ImaginedValidityResult",
    "compute_head_has_signal",
    "compute_imagined_validity",
    "compute_wm_pbt_validity_weight",
]
