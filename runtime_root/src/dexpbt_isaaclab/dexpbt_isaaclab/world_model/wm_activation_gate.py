"""PBT-WM Soft Activation Gate v1.

A single, reusable continuous activation weight that replaces the scattered
``tol <= threshold`` hard gates used by WM / EIR / actor-distill paths.

The unified weight is a product of independent [0, 1] components::

    w = w_time * w_tol * w_perf * w_conf * w_unc * w_source

Different WM modules share this form but bring their own ``WMActivationConfig``
parameters.  This module is pure (no torch dependency) so it can be unit-tested
in isolation and called per-row from tensor code by the caller.

NOTE: this soft gate intentionally does NOT relax the residual-rollout-apply
path.  ``eir_residual_apply_to_rollout`` keeps its own hard tol / confidence
gate; the soft activation only governs actor_distill / WM PBT source score /
curriculum diagnostics.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any


@dataclass
class WMActivationConfig:
    enable: bool = True

    # tolerance soft schedule
    gate_mode: str = "soft_tol"  # off | hard_tol | soft_tol | confidence_only | shrink_candidate
    tol_start: float = 0.065
    tol_full: float = 0.040
    hard_tol: float = 0.040
    min_weight: float = 0.0
    max_weight: float = 1.0

    # time schedule
    start_after_frames: int = 500_000_000
    ramp_frames: int = 300_000_000

    # performance gate
    nsucc_ref: float = 20.0
    min_nsucc_mean: float = 0.0

    # confidence / uncertainty
    min_margin: float = 0.0
    margin_norm: float = 0.05
    uncertainty_threshold: float = 2.0

    # source weights
    rssm_source_weight: float = 0.8
    elite_source_weight: float = 1.0
    archive_source_weight: float = 1.0
    gaussian_source_weight: float = 0.0
    actor_source_weight: float = 0.0


@dataclass
class WMActivationAudit:
    active: bool
    weight: float
    time_weight: float
    tol_weight: float
    perf_weight: float
    conf_weight: float
    unc_weight: float
    source_weight: float
    tol_bin: str
    reason: str


def clamp01(x: float) -> float:
    if not math.isfinite(float(x)):
        return 0.0
    return max(0.0, min(1.0, float(x)))


def soft_tol_weight(
    tol: float,
    *,
    tol_start: float,
    tol_full: float,
    min_w: float = 0.0,
    max_w: float = 1.0,
) -> float:
    """Continuous tolerance schedule.

    Larger tolerance = easier stage (low weight).  Smaller tolerance = harder
    stage (high weight).

    - ``tol >= tol_start``  -> ``min_w``
    - ``tol <= tol_full``   -> ``max_w``
    - in between            -> linear ramp.
    """
    if tol_start <= tol_full:
        # Degenerate schedule: fall back to hard-ish behaviour.
        return float(max_w if tol <= tol_full else min_w)
    if tol >= tol_start:
        return float(min_w)
    if tol <= tol_full:
        return float(max_w)
    x = (tol_start - tol) / (tol_start - tol_full)
    return float(min_w + clamp01(x) * (max_w - min_w))


def tol_bin(tol: float) -> str:
    if tol > 0.06:
        return "gt_006"
    if tol > 0.05:
        return "005_006"
    if tol > 0.04:
        return "004_005"
    return "le_004"


TOL_BINS: tuple[str, ...] = ("gt_006", "005_006", "004_005", "le_004")


def source_weight(source: str | int, cfg: WMActivationConfig) -> float:
    """Map a source string OR an existing EIR integer source id to its weight."""
    s = str(source).lower()
    if s in {"rssm", "src_rssm", "4"}:
        return cfg.rssm_source_weight
    if s in {"elite", "elite_local", "src_elite_local", "2"}:
        return cfg.elite_source_weight
    if s in {"archive", "elite_archive", "src_elite_archive", "3"}:
        return cfg.archive_source_weight
    if s in {"gaussian", "local", "src_gaussian", "1"}:
        return cfg.gaussian_source_weight
    if s in {"actor", "src_actor", "0"}:
        return cfg.actor_source_weight
    return 0.0


def compute_wm_activation(
    *,
    cfg: WMActivationConfig,
    frames: int,
    tol: float,
    nsucc_mean: float = 0.0,
    margin: float = 0.0,
    uncertainty: float | None = None,
    source: str | int = "rssm",
    eval_valid: bool = True,
    shrink_candidate: bool = False,
) -> WMActivationAudit:
    """Compute continuous WM activation weight.

    Rules:
    - ``cfg.enable=False`` or ``gate_mode=off`` -> weight 0
    - ``confidence_only``: no tol/perf gate (w_tol=1, w_perf=1); mainly
      eval_valid + time + uncertainty/source.
    - ``shrink_candidate``: only active when ``shrink_candidate=True``.
    - ``hard_tol``: legacy behaviour, ``tol <= hard_tol``.
    - ``soft_tol``: continuous schedule between ``tol_start`` and ``tol_full``.

    Note: ``min_weight`` only feeds ``soft_tol_weight``'s tolerance component;
    it is not a floor on the overall weight, which can still be driven to 0 by
    confidence / uncertainty / perf / source gates.
    """
    tb = tol_bin(tol)
    zero = WMActivationAudit(
        active=False,
        weight=0.0,
        time_weight=0.0,
        tol_weight=0.0,
        perf_weight=0.0,
        conf_weight=0.0,
        unc_weight=0.0,
        source_weight=0.0,
        tol_bin=tb,
        reason="disabled",
    )
    if not cfg.enable or str(cfg.gate_mode) == "off":
        zero.reason = "disabled" if not cfg.enable else "gate_off"
        return zero
    if not eval_valid:
        zero.reason = "invalid_eval"
        return zero

    mode = str(cfg.gate_mode)

    # --- time ramp ---
    if frames < cfg.start_after_frames:
        w_time = 0.0
    else:
        w_time = clamp01(
            (frames - cfg.start_after_frames) / max(1, int(cfg.ramp_frames))
        )

    # --- tolerance component ---
    if mode == "hard_tol":
        w_tol = 1.0 if tol <= cfg.hard_tol else 0.0
    elif mode == "soft_tol":
        w_tol = soft_tol_weight(
            tol,
            tol_start=cfg.tol_start,
            tol_full=cfg.tol_full,
            min_w=cfg.min_weight,
            max_w=cfg.max_weight,
        )
    elif mode == "confidence_only":
        w_tol = 1.0
    elif mode == "shrink_candidate":
        w_tol = 1.0 if shrink_candidate else 0.0
    else:
        zero.reason = f"unknown_gate_mode:{mode}"
        return zero

    # --- performance gate ---
    # confidence_only / shrink_candidate modes do not apply the perf gate here
    # (the WM source / curriculum callers handle nsucc separately).  A negative
    # nsucc_mean means "unknown" and keeps the perf gate neutral.
    if mode in ("confidence_only", "shrink_candidate") or nsucc_mean < 0.0:
        w_perf = 1.0
    else:
        w_perf = clamp01(nsucc_mean / max(1e-6, cfg.nsucc_ref))
        if nsucc_mean < cfg.min_nsucc_mean:
            w_perf = 0.0

    # --- confidence gate ---
    if margin < cfg.min_margin:
        w_conf = 0.0
    else:
        w_conf = clamp01(margin / max(1e-6, cfg.margin_norm))

    # --- uncertainty gate ---
    if uncertainty is None:
        w_unc = 1.0
    elif not math.isfinite(float(uncertainty)):
        w_unc = 0.0
    else:
        w_unc = clamp01(1.0 - float(uncertainty) / max(1e-6, cfg.uncertainty_threshold))

    # --- source gate ---
    w_source = source_weight(source, cfg)

    weight = (
        float(cfg.max_weight)
        * w_time
        * w_tol
        * w_perf
        * w_conf
        * w_unc
        * w_source
    )
    weight = clamp01(weight / max(1e-6, float(cfg.max_weight))) * float(cfg.max_weight)
    active = weight > 0.0

    reason = "active"
    if not active:
        if w_time <= 0.0:
            reason = "before_start" if frames < cfg.start_after_frames else "time_zero"
        elif w_tol <= 0.0:
            reason = "tol_zero"
        elif w_perf <= 0.0:
            reason = "perf_zero"
        elif w_conf <= 0.0:
            reason = "conf_zero"
        elif w_unc <= 0.0:
            reason = "unc_zero"
        elif w_source <= 0.0:
            reason = "source_zero"
        else:
            reason = "zero"

    return WMActivationAudit(
        active=active,
        weight=float(weight),
        time_weight=float(w_time),
        tol_weight=float(w_tol),
        perf_weight=float(w_perf),
        conf_weight=float(w_conf),
        unc_weight=float(w_unc),
        source_weight=float(w_source),
        tol_bin=tb,
        reason=reason,
    )


__all__ = [
    "WMActivationConfig",
    "WMActivationAudit",
    "clamp01",
    "soft_tol_weight",
    "tol_bin",
    "TOL_BINS",
    "source_weight",
    "compute_wm_activation",
]
