"""Readiness-Adaptive CSER-DexPBT (Profile 23) pure control logic.

This module is intentionally free of Isaac / simulator imports so every quantity
can be unit-tested on CPU.  It owns bank-readiness math, the three-phase state
machine, WM graduation / alpha, ESS-controlled distill scaling, and shadow
retrieval budgeting.  Wiring into train / EIR is done elsewhere.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, fields
from typing import Any, Mapping, Optional

import torch

PHASE_WARMUP = 0
PHASE_PRIORITY_BOOTSTRAP = 1
PHASE_CALIBRATED_REFINEMENT = 2


def _clip(x: float, lo: float, hi: float) -> float:
    return float(min(hi, max(lo, float(x))))


@dataclass
class ReadinessAdaptiveConfig:
    """All Profile-23 readiness / graduation / ESS knobs (defaults OFF / safe)."""

    readiness_adaptive_enable: bool = False
    readiness_min_frames: int = 100_000_000
    readiness_bank_size_low: int = 1000
    readiness_bank_size_full: int = 3000
    readiness_coverage_target: float = 0.20
    readiness_coverage_ema_beta: float = 0.9
    readiness_enter_threshold: float = 0.50
    readiness_exit_threshold: float = 0.20
    readiness_required_windows: int = 2
    readiness_ramp_frames: int = 150_000_000
    readiness_eval_interval_frames: int = 10_000_000
    readiness_shadow_retrieval_fraction: float = 0.05

    # WM graduation
    wm_graduation_enable: bool = False
    wm_graduation_required_passes: int = 2
    wm_graduation_early_until_frames: int = 500_000_000
    wm_graduation_early_eval_interval_frames: int = 25_000_000
    wm_graduation_late_eval_interval_frames: int = 50_000_000
    wm_graduation_early_beta_up: float = 0.7
    wm_graduation_late_beta_up: float = 0.9
    wm_graduation_beta_down: float = 0.5
    wm_graduation_early_alpha_cap: float = 0.5
    wm_graduation_late_alpha_cap: float = 1.0
    wm_graduation_min_valid_prediction_frac: float = 0.90

    # ESS distill
    readiness_bootstrap_distill_weight: float = 0.001
    readiness_calibrated_distill_weight: float = 0.0015
    readiness_ess_target: float = 128.0
    readiness_ess_min_for_update: float = 8.0

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        """Full range + finiteness validation. Raises ``ValueError`` on bad config."""

        def _finite(name: str, value: float) -> float:
            v = float(value)
            if v != v or v in (float("inf"), float("-inf")):
                raise ValueError(f"ReadinessAdaptiveConfig.{name} must be finite, got {value!r}")
            return v

        def _pos(name: str, value: float) -> float:
            v = _finite(name, value)
            if v <= 0.0:
                raise ValueError(f"ReadinessAdaptiveConfig.{name} must be > 0, got {value!r}")
            return v

        def _nonneg(name: str, value: float) -> float:
            v = _finite(name, value)
            if v < 0.0:
                raise ValueError(f"ReadinessAdaptiveConfig.{name} must be >= 0, got {value!r}")
            return v

        def _unit(name: str, value: float) -> float:
            v = _finite(name, value)
            if not (0.0 <= v <= 1.0):
                raise ValueError(
                    f"ReadinessAdaptiveConfig.{name} must be in [0, 1], got {value!r}"
                )
            return v

        self.readiness_adaptive_enable = bool(self.readiness_adaptive_enable)
        self.readiness_min_frames = int(self.readiness_min_frames)
        if self.readiness_min_frames < 0:
            raise ValueError(
                f"ReadinessAdaptiveConfig.readiness_min_frames must be >= 0, "
                f"got {self.readiness_min_frames!r}"
            )
        self.readiness_bank_size_low = int(self.readiness_bank_size_low)
        self.readiness_bank_size_full = int(self.readiness_bank_size_full)
        if self.readiness_bank_size_low < 0:
            raise ValueError(
                f"ReadinessAdaptiveConfig.readiness_bank_size_low must be >= 0, "
                f"got {self.readiness_bank_size_low!r}"
            )
        if self.readiness_bank_size_full <= self.readiness_bank_size_low:
            raise ValueError(
                "ReadinessAdaptiveConfig.readiness_bank_size_full must be > "
                f"readiness_bank_size_low, got full={self.readiness_bank_size_full!r} "
                f"low={self.readiness_bank_size_low!r}"
            )

        self.readiness_coverage_target = _pos(
            "readiness_coverage_target", self.readiness_coverage_target
        )
        self.readiness_coverage_ema_beta = _unit(
            "readiness_coverage_ema_beta", self.readiness_coverage_ema_beta
        )
        self.readiness_enter_threshold = _unit(
            "readiness_enter_threshold", self.readiness_enter_threshold
        )
        self.readiness_exit_threshold = _unit(
            "readiness_exit_threshold", self.readiness_exit_threshold
        )
        if self.readiness_enter_threshold <= self.readiness_exit_threshold:
            raise ValueError(
                "ReadinessAdaptiveConfig.readiness_enter_threshold must be > "
                f"readiness_exit_threshold, got enter={self.readiness_enter_threshold!r} "
                f"exit={self.readiness_exit_threshold!r}"
            )
        self.readiness_required_windows = int(self.readiness_required_windows)
        if self.readiness_required_windows < 1:
            raise ValueError(
                f"ReadinessAdaptiveConfig.readiness_required_windows must be >= 1, "
                f"got {self.readiness_required_windows!r}"
            )
        self.readiness_ramp_frames = int(self.readiness_ramp_frames)
        if self.readiness_ramp_frames < 0:
            raise ValueError(
                f"ReadinessAdaptiveConfig.readiness_ramp_frames must be >= 0, "
                f"got {self.readiness_ramp_frames!r}"
            )
        self.readiness_eval_interval_frames = int(self.readiness_eval_interval_frames)
        if self.readiness_eval_interval_frames < 1:
            raise ValueError(
                "ReadinessAdaptiveConfig.readiness_eval_interval_frames must be >= 1, "
                f"got {self.readiness_eval_interval_frames!r}"
            )
        self.readiness_shadow_retrieval_fraction = _unit(
            "readiness_shadow_retrieval_fraction",
            self.readiness_shadow_retrieval_fraction,
        )

        self.wm_graduation_enable = bool(self.wm_graduation_enable)
        self.wm_graduation_required_passes = int(self.wm_graduation_required_passes)
        if self.wm_graduation_required_passes < 1:
            raise ValueError(
                "ReadinessAdaptiveConfig.wm_graduation_required_passes must be >= 1, "
                f"got {self.wm_graduation_required_passes!r}"
            )
        self.wm_graduation_early_until_frames = int(self.wm_graduation_early_until_frames)
        if self.wm_graduation_early_until_frames < 0:
            raise ValueError(
                "ReadinessAdaptiveConfig.wm_graduation_early_until_frames must be >= 0, "
                f"got {self.wm_graduation_early_until_frames!r}"
            )
        self.wm_graduation_early_eval_interval_frames = int(
            self.wm_graduation_early_eval_interval_frames
        )
        self.wm_graduation_late_eval_interval_frames = int(
            self.wm_graduation_late_eval_interval_frames
        )
        if self.wm_graduation_early_eval_interval_frames < 1:
            raise ValueError(
                "ReadinessAdaptiveConfig.wm_graduation_early_eval_interval_frames "
                f"must be >= 1, got {self.wm_graduation_early_eval_interval_frames!r}"
            )
        if self.wm_graduation_late_eval_interval_frames < 1:
            raise ValueError(
                "ReadinessAdaptiveConfig.wm_graduation_late_eval_interval_frames "
                f"must be >= 1, got {self.wm_graduation_late_eval_interval_frames!r}"
            )
        self.wm_graduation_early_beta_up = _unit(
            "wm_graduation_early_beta_up", self.wm_graduation_early_beta_up
        )
        self.wm_graduation_late_beta_up = _unit(
            "wm_graduation_late_beta_up", self.wm_graduation_late_beta_up
        )
        self.wm_graduation_beta_down = _unit(
            "wm_graduation_beta_down", self.wm_graduation_beta_down
        )
        self.wm_graduation_early_alpha_cap = _unit(
            "wm_graduation_early_alpha_cap", self.wm_graduation_early_alpha_cap
        )
        self.wm_graduation_late_alpha_cap = _unit(
            "wm_graduation_late_alpha_cap", self.wm_graduation_late_alpha_cap
        )
        self.wm_graduation_min_valid_prediction_frac = _unit(
            "wm_graduation_min_valid_prediction_frac",
            self.wm_graduation_min_valid_prediction_frac,
        )

        self.readiness_bootstrap_distill_weight = _nonneg(
            "readiness_bootstrap_distill_weight", self.readiness_bootstrap_distill_weight
        )
        self.readiness_calibrated_distill_weight = _nonneg(
            "readiness_calibrated_distill_weight",
            self.readiness_calibrated_distill_weight,
        )
        self.readiness_ess_target = _pos("readiness_ess_target", self.readiness_ess_target)
        self.readiness_ess_min_for_update = _nonneg(
            "readiness_ess_min_for_update", self.readiness_ess_min_for_update
        )


@dataclass
class ReadinessAdaptiveState:
    """Persistent readiness / graduation controller state (checkpointable)."""

    phase: int = PHASE_WARMUP
    phase_transition_count: int = 0
    coverage_ema: float = 0.0
    bank_ready: bool = False
    enter_window_streak: int = 0
    exit_window_streak: int = 0
    ready_accumulated_frames: int = 0
    last_readiness_eval_frame: int = 0
    last_frames_seen: int = 0
    wm_pass_streak: int = 0
    wm_graduated: bool = False
    wm_ema_raw: float = 0.0
    last_bank_size_readiness: float = 0.0
    last_coverage_readiness: float = 0.0
    last_bank_readiness: float = 0.0
    last_refinement_ramp: float = 0.0
    last_alpha_cap: float = 0.0
    last_alpha_effective: float = 0.0
    last_lambda_effective: float = 0.0
    last_active_budget: float = 0.15
    last_max_log_gap: float = 0.30
    last_soft_scale: float = 0.15
    last_ess: float = 0.0
    last_ess_scale: float = 0.0
    last_ess_update_skipped: bool = False
    last_unscaled_distill_loss: float = 0.0
    last_scaled_distill_loss: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for f in fields(self):
            v = getattr(self, f.name)
            if isinstance(v, bool):
                out[f.name] = bool(v)
            elif isinstance(v, int) and not isinstance(v, bool):
                out[f.name] = int(v)
            else:
                out[f.name] = float(v)
        return out

    @classmethod
    def from_dict(cls, d: Mapping[str, Any] | None) -> "ReadinessAdaptiveState":
        st = cls()
        if not isinstance(d, Mapping):
            return st
        bool_fields = {
            "bank_ready",
            "wm_graduated",
            "last_ess_update_skipped",
        }
        int_fields = {
            "phase",
            "phase_transition_count",
            "enter_window_streak",
            "exit_window_streak",
            "ready_accumulated_frames",
            "last_readiness_eval_frame",
            "last_frames_seen",
            "wm_pass_streak",
        }
        for f in fields(cls):
            if f.name not in d:
                continue
            raw = d[f.name]
            if f.name in bool_fields:
                setattr(st, f.name, bool(raw))
            elif f.name in int_fields:
                setattr(st, f.name, int(raw))
            else:
                setattr(st, f.name, float(raw))
        return st


def bank_size_readiness(bank_size: float, low: float, full: float) -> float:
    """Log-space bank fill readiness in ``[0, 1]``.

    ``r_size = clip((log1p(n) - log1p(low)) / (log1p(full) - log1p(low)), 0, 1)``.
    """
    n = max(0.0, float(bank_size))
    lo = max(0.0, float(low))
    hi = max(0.0, float(full))
    denom = math.log1p(hi) - math.log1p(lo)
    if denom <= 0.0:
        return 1.0 if n >= hi else 0.0
    return _clip((math.log1p(n) - math.log1p(lo)) / denom, 0.0, 1.0)


def coverage_readiness(coverage_ema: float, target: float) -> float:
    """``r_coverage = clip(coverage_ema / target, 0, 1)``."""
    t = float(target)
    if t <= 0.0:
        return 1.0 if float(coverage_ema) > 0.0 else 0.0
    return _clip(float(coverage_ema) / t, 0.0, 1.0)


def bank_readiness(r_size: float, r_coverage: float) -> float:
    """``r_bank = r_size * r_coverage``."""
    return float(r_size) * float(r_coverage)


def update_coverage_ema(ema: float, observation: float, beta: float) -> float:
    """EMA update: ``beta * ema + (1 - beta) * observation``."""
    b = _clip(float(beta), 0.0, 1.0)
    return b * float(ema) + (1.0 - b) * float(observation)


def refinement_ramp(
    ready_accumulated_frames: int, ramp_frames: int, r_bank: float
) -> float:
    """``r_refine = clip(ready / ramp, 0, 1) * r_bank``."""
    ramp = int(ramp_frames)
    if ramp <= 0:
        frac = 1.0 if int(ready_accumulated_frames) > 0 else 0.0
    else:
        frac = _clip(float(ready_accumulated_frames) / float(ramp), 0.0, 1.0)
    return frac * float(r_bank)


def dynamic_active_budget(r_refine: float) -> float:
    """``clip(0.15 + 0.20 * r_refine, 0.15, 0.35)``."""
    return _clip(0.15 + 0.20 * float(r_refine), 0.15, 0.35)


def dynamic_max_log_gap(r_bank: float) -> float:
    """``clip(0.30 - 0.10 * r_bank, 0.20, 0.30)``."""
    return _clip(0.30 - 0.10 * float(r_bank), 0.20, 0.30)


def dynamic_soft_scale(r_bank: float) -> float:
    """``clip(0.15 - 0.05 * r_bank, 0.10, 0.15)``."""
    return _clip(0.15 - 0.05 * float(r_bank), 0.10, 0.15)


def phase_lambda(phase: int, cfg: ReadinessAdaptiveConfig) -> float:
    """Phase distill weight: warmup=0, bootstrap=0.001, calibrated=0.0015."""
    p = int(phase)
    if p == PHASE_PRIORITY_BOOTSTRAP:
        return float(cfg.readiness_bootstrap_distill_weight)
    if p == PHASE_CALIBRATED_REFINEMENT:
        return float(cfg.readiness_calibrated_distill_weight)
    return 0.0


def ess_and_scale(
    weights: torch.Tensor,
    ess_target: float,
    ess_min: float,
    eps: float = 1e-8,
) -> tuple[float, float, bool]:
    """Effective sample size, scale ``min(1, ess/target)``, and skip if ``ess < min``."""
    w = weights.detach().float().reshape(-1)
    if int(w.numel()) == 0:
        return 0.0, 0.0, True
    w = torch.where(torch.isfinite(w), w, torch.zeros_like(w)).clamp_min(0.0)
    sum_w = float(w.sum().item())
    sum_w2 = float((w * w).sum().item())
    ess = (sum_w * sum_w) / (sum_w2 + float(eps))
    target = max(float(ess_target), float(eps))
    scale = min(1.0, ess / target)
    skip = bool(ess < float(ess_min))
    if skip:
        scale = 0.0
    return float(ess), float(scale), skip


def shadow_row_budget(n_eligible: int, fraction: float = 0.05) -> int:
    """``min(n, max(0, floor(n * fraction)))`` — never exceed the fraction share."""
    n = int(n_eligible)
    if n <= 0:
        return 0
    frac = max(0.0, float(fraction))
    return min(n, max(0, int(math.floor(n * frac))))


def select_shadow_indices(
    eligible_mask: torch.Tensor,
    fraction: float,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Batched shadow-row index selection (no Python per-row retrieval loop).

    Takes the first ``floor(n * fraction)`` eligible indices after an optional
    ``randperm``.  Returns a 1-D ``long`` tensor of selected row indices.
    """
    mask = eligible_mask.reshape(-1).to(dtype=torch.bool)
    eligible = mask.nonzero(as_tuple=False).reshape(-1)
    n = int(eligible.numel())
    budget = shadow_row_budget(n, fraction)
    if budget <= 0:
        return eligible.new_empty((0,), dtype=torch.long)
    if generator is not None and n > 1:
        perm = torch.randperm(n, generator=generator, device=eligible.device)
        return eligible[perm[:budget]].to(dtype=torch.long)
    return eligible[:budget].to(dtype=torch.long)


def evaluate_bank_ready_windows(
    state: ReadinessAdaptiveState,
    *,
    frames: int,
    r_bank: float,
    cfg: ReadinessAdaptiveConfig,
) -> ReadinessAdaptiveState:
    """Update enter/exit streaks and bank_ready with hysteresis (caller owns interval).

    Enter when ``frames >= min_frames`` and ``r_bank >= enter_threshold`` for
    ``required_windows`` consecutive evals.  Exit when ``r_bank < exit_threshold``
    for ``required_windows`` consecutive evals while ready.  Single-window flips
    are rejected by the streak requirement.
    """
    frames_i = int(frames)
    r = float(r_bank)
    state.last_readiness_eval_frame = frames_i
    state.last_bank_readiness = r

    enter_ok = frames_i >= int(cfg.readiness_min_frames) and r >= float(
        cfg.readiness_enter_threshold
    )
    if enter_ok:
        state.enter_window_streak = int(state.enter_window_streak) + 1
    else:
        state.enter_window_streak = 0

    if r < float(cfg.readiness_exit_threshold):
        state.exit_window_streak = int(state.exit_window_streak) + 1
    else:
        state.exit_window_streak = 0

    need = int(cfg.readiness_required_windows)
    if (not state.bank_ready) and int(state.enter_window_streak) >= need:
        state.bank_ready = True
        state.enter_window_streak = 0
        state.exit_window_streak = 0
    elif state.bank_ready and int(state.exit_window_streak) >= need:
        state.bank_ready = False
        state.enter_window_streak = 0
        state.exit_window_streak = 0
    return state


def accumulate_ready_frames(
    state: ReadinessAdaptiveState,
    *,
    frames: int,
    cfg: ReadinessAdaptiveConfig,
) -> ReadinessAdaptiveState:
    """Accumulate ready frames only while ``bank_ready``; pause (never reset) otherwise."""
    del cfg  # reserved for future gating; signature kept for callers
    frames_i = int(frames)
    delta = max(0, frames_i - int(state.last_frames_seen))
    state.last_frames_seen = frames_i
    if state.bank_ready:
        state.ready_accumulated_frames = int(state.ready_accumulated_frames) + int(delta)
    return state


def sync_phase(
    state: ReadinessAdaptiveState,
    *,
    wm_structural_failure: bool = False,
) -> ReadinessAdaptiveState:
    """Derive phase from bank_ready / wm_graduated; count transitions.

    Structural WM failure clears graduation / streak / EMA and forces alpha
    fields to 0, then lands in PRIORITY_BOOTSTRAP if bank_ready else WARMUP.
    """
    prev = int(state.phase)

    if wm_structural_failure:
        state.wm_graduated = False
        state.wm_pass_streak = 0
        state.wm_ema_raw = 0.0
        state.last_alpha_cap = 0.0
        state.last_alpha_effective = 0.0
        if state.bank_ready:
            state.phase = PHASE_PRIORITY_BOOTSTRAP
        else:
            state.phase = PHASE_WARMUP
    elif not state.bank_ready:
        state.phase = PHASE_WARMUP
    elif not state.wm_graduated:
        state.phase = PHASE_PRIORITY_BOOTSTRAP
    else:
        state.phase = PHASE_CALIBRATED_REFINEMENT

    if int(state.phase) != prev:
        state.phase_transition_count = int(state.phase_transition_count) + 1
    return state


def _graduation_raw(metrics: Mapping[str, Any]) -> float:
    if "wm_cal/raw_gate" in metrics:
        return float(metrics["wm_cal/raw_gate"])
    gate_keys = (
        "wm_cal/gate_recovery_auc",
        "wm_cal/gate_recovery_ece",
        "wm_cal/gate_combined_rank",
    )
    if all(k in metrics for k in gate_keys):
        return (
            float(metrics["wm_cal/gate_recovery_auc"])
            * float(metrics["wm_cal/gate_recovery_ece"])
            * float(metrics["wm_cal/gate_combined_rank"])
        )
    if "wm_cal/calibration_pass" in metrics:
        return float(metrics["wm_cal/calibration_pass"])
    return 0.0


def _valid_prediction_frac(metrics: Mapping[str, Any]) -> float:
    if "wm_cal/valid_prediction_frac" in metrics:
        return float(metrics["wm_cal/valid_prediction_frac"])
    if "wm_cal/row_prediction_valid_frac" in metrics:
        return float(metrics["wm_cal/row_prediction_valid_frac"])
    return 1.0  # caller may have already enforced; default permissive


def apply_wm_graduation(
    state: ReadinessAdaptiveState,
    *,
    frames: int,
    metrics: dict,
    mode: str,
    cfg: ReadinessAdaptiveConfig,
    r_bank: float,
) -> tuple[ReadinessAdaptiveState, float]:
    """Update WM pass streak / EMA / graduation and return ``alpha_effective``.

    ``mode`` is one of ``{"valid", "insufficient", "structural", "metric_fail"}``.
    Alpha is zero until graduated (Phase-1 force-off) and is capped early/late.
    """
    mode_norm = str(mode or "").strip().lower()
    frames_i = int(frames)
    early = frames_i < int(cfg.wm_graduation_early_until_frames)
    beta_up = (
        float(cfg.wm_graduation_early_beta_up)
        if early
        else float(cfg.wm_graduation_late_beta_up)
    )
    beta_down = float(cfg.wm_graduation_beta_down)
    alpha_cap = (
        float(cfg.wm_graduation_early_alpha_cap)
        if early
        else float(cfg.wm_graduation_late_alpha_cap)
    )

    if mode_norm == "structural":
        state.wm_pass_streak = 0
        state.wm_ema_raw = 0.0
        state.wm_graduated = False
    elif mode_norm in ("insufficient", "metric_fail"):
        state.wm_pass_streak = 0
        state.wm_ema_raw = beta_down * float(state.wm_ema_raw)
        state.wm_graduated = False
    elif mode_norm == "valid":
        # Caller-prechecked pass; still respect min valid-prediction frac when present.
        vpf = _valid_prediction_frac(metrics)
        eval_ok = float(metrics.get("wm_cal/eval_valid", 1.0)) > 0.5
        if (
            eval_ok
            and vpf >= float(cfg.wm_graduation_min_valid_prediction_frac)
        ):
            raw = _graduation_raw(metrics)
            state.wm_pass_streak = int(state.wm_pass_streak) + 1
            state.wm_ema_raw = beta_up * float(state.wm_ema_raw) + (1.0 - beta_up) * float(
                raw
            )
        else:
            state.wm_pass_streak = 0
            state.wm_ema_raw = beta_down * float(state.wm_ema_raw)
        state.wm_graduated = int(state.wm_pass_streak) >= int(
            cfg.wm_graduation_required_passes
        )
    else:
        raise ValueError(
            f"apply_wm_graduation mode must be valid|insufficient|structural|metric_fail, "
            f"got {mode!r}"
        )

    # Re-evaluate graduated after streak mutations for non-valid modes.
    if mode_norm != "valid":
        state.wm_graduated = int(state.wm_pass_streak) >= int(
            cfg.wm_graduation_required_passes
        )

    ema = float(state.wm_ema_raw)
    if state.wm_graduated:
        alpha_effective = _clip(ema, 0.0, alpha_cap) * float(r_bank)
    else:
        alpha_effective = 0.0

    # Phase-1 force-off: not graduated ⇒ alpha stays 0 (already).
    state.last_alpha_cap = float(alpha_cap)
    state.last_alpha_effective = float(alpha_effective)
    return state, float(alpha_effective)


def telemetry(state: ReadinessAdaptiveState) -> dict[str, float]:
    """TensorBoard-ready ``readiness_adaptive/*`` scalars."""
    return {
        "readiness_adaptive/phase": float(state.phase),
        "readiness_adaptive/phase_transition_count": float(state.phase_transition_count),
        "readiness_adaptive/bank_readiness": float(state.last_bank_readiness),
        "readiness_adaptive/bank_size_readiness": float(state.last_bank_size_readiness),
        "readiness_adaptive/coverage_readiness": float(state.last_coverage_readiness),
        "readiness_adaptive/coverage_ema": float(state.coverage_ema),
        "readiness_adaptive/bank_ready": float(1.0 if state.bank_ready else 0.0),
        "readiness_adaptive/ready_accumulated_frames": float(
            state.ready_accumulated_frames
        ),
        "readiness_adaptive/refinement_ramp": float(state.last_refinement_ramp),
        "readiness_adaptive/wm_graduated": float(1.0 if state.wm_graduated else 0.0),
        "readiness_adaptive/wm_pass_streak": float(state.wm_pass_streak),
        "readiness_adaptive/wm_ema_raw": float(state.wm_ema_raw),
        "readiness_adaptive/alpha_cap": float(state.last_alpha_cap),
        "readiness_adaptive/alpha_effective": float(state.last_alpha_effective),
        "readiness_adaptive/lambda_effective": float(state.last_lambda_effective),
        "readiness_adaptive/active_budget": float(state.last_active_budget),
        "readiness_adaptive/retrieval_max_log_gap": float(state.last_max_log_gap),
        "readiness_adaptive/retrieval_soft_scale": float(state.last_soft_scale),
        "readiness_adaptive/ess": float(state.last_ess),
        "readiness_adaptive/ess_scale": float(state.last_ess_scale),
        "readiness_adaptive/ess_update_skipped": float(
            1.0 if state.last_ess_update_skipped else 0.0
        ),
        "readiness_adaptive/unscaled_distill_loss": float(
            state.last_unscaled_distill_loss
        ),
        "readiness_adaptive/scaled_distill_loss": float(state.last_scaled_distill_loss),
    }
