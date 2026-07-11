"""CSER-v0 staged PBT consolidation schedule.

Wraps the existing tick scheduler and bottom-restart gates with
frame-based stages (exploration / consolidation / selection) so late-run
PBT churn can be reduced without forking the core observer logic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from dexpbt_isaaclab.pbt.pbt_control_plane import evaluate_pbt_tick_schedule

STAGE_EXPLORATION = "exploration"
STAGE_CONSOLIDATION = "consolidation"
STAGE_SELECTION = "selection"

STAGE_IDS = {
    STAGE_EXPLORATION: 0,
    STAGE_CONSOLIDATION: 1,
    STAGE_SELECTION: 2,
}


@dataclass
class PbtConsolidationConfig:
    enabled: bool = False
    stage1_end_frames: int = 2_200_000_000
    stage2_end_frames: int = 3_500_000_000
    stage3_end_frames: int = 5_000_000_000
    stage2_interval_mult: float = 2.0
    stage3_interval_mult: float = 4.0
    stage2_mutation_rate_mult: float = 0.5
    stage3_bottom_only: bool = True
    stage3_hard_collapse_only: bool = True
    stage2_retention_threshold: float = 0.85
    collapse_nsucc_mean_threshold: float = 1.0
    # Schedule keying: "frames" (legacy, default for all old profiles) or
    # "curriculum_progress" (progress_refine).  In curriculum_progress mode the
    # stage boundaries below are dimensionless curriculum progress c in [0, 1].
    schedule_mode: str = "frames"
    stage_consolidation_progress: float = 0.50
    stage_selection_progress: float = 0.85


@dataclass
class BottomRestartContext:
    promising_protected: bool = False
    floor_recovery_protected: bool = False
    hard_collapse: bool = False
    collapse_override: bool = False
    nsucc_mean: float = 0.0
    recent_peak_nsucc_mean: float = 0.0
    tier: str = "bottom"
    late_degradation_rescue: bool = False


def resolve_pbt_stage(
    raw_frame: int,
    cfg: PbtConsolidationConfig,
    *,
    curriculum_progress: float | None = None,
) -> str:
    """Return consolidation stage name.

    In ``frames`` mode the stage is keyed on the population env frame (legacy).
    In ``curriculum_progress`` mode the stage is keyed on the dimensionless
    curriculum progress ``c`` (falls back to frames when ``c`` is unavailable).
    """
    if not cfg.enabled:
        return STAGE_EXPLORATION
    if str(cfg.schedule_mode) == "curriculum_progress" and curriculum_progress is not None:
        c = float(curriculum_progress)
        if c < float(cfg.stage_consolidation_progress):
            return STAGE_EXPLORATION
        if c < float(cfg.stage_selection_progress):
            return STAGE_CONSOLIDATION
        return STAGE_SELECTION
    f = int(raw_frame)
    if f < int(cfg.stage1_end_frames):
        return STAGE_EXPLORATION
    if f < int(cfg.stage2_end_frames):
        return STAGE_CONSOLIDATION
    if f < int(cfg.stage3_end_frames):
        return STAGE_SELECTION
    return STAGE_SELECTION


def stage_id(stage: str) -> int:
    return int(STAGE_IDS.get(str(stage), 0))


def effective_tick_interval(base_interval: int, stage: str, cfg: PbtConsolidationConfig) -> int:
    """Effective PBT tick interval after stage multiplier."""
    base = max(int(base_interval), 1)
    if not cfg.enabled:
        return base
    if stage == STAGE_CONSOLIDATION:
        return max(base, int(round(base * float(cfg.stage2_interval_mult))))
    if stage == STAGE_SELECTION:
        return max(base, int(round(base * float(cfg.stage3_interval_mult))))
    return base


def effective_mutation_rate(base_rate: float, stage: str, cfg: PbtConsolidationConfig) -> float:
    if not cfg.enabled:
        return float(base_rate)
    if stage == STAGE_CONSOLIDATION:
        return float(base_rate) * float(cfg.stage2_mutation_rate_mult)
    if stage == STAGE_SELECTION:
        return 0.0
    return float(base_rate)


def evaluate_consolidated_tick(
    *,
    raw_frame: int,
    self_pbt_iteration: int,
    last_mutation_frame: int,
    start_after: int,
    initial_delay: int,
    interval_steps: int,
    cfg: PbtConsolidationConfig,
    curriculum_progress: float | None = None,
) -> tuple[bool, int, str, str, int]:
    """Return (should_tick, derived_iteration, skip_reason, stage, effective_interval)."""
    stage = resolve_pbt_stage(raw_frame, cfg, curriculum_progress=curriculum_progress)
    eff_interval = effective_tick_interval(interval_steps, stage, cfg)
    should_tick, derived, skip_reason = evaluate_pbt_tick_schedule(
        raw_frame=int(raw_frame),
        self_pbt_iteration=int(self_pbt_iteration),
        last_mutation_frame=int(last_mutation_frame),
        start_after=int(start_after),
        initial_delay=int(initial_delay),
        interval_steps=int(eff_interval),
    )
    if cfg.enabled and should_tick:
        skip_reason = skip_reason or f"stage_{stage}"
    return should_tick, derived, skip_reason, stage, eff_interval


def should_allow_bottom_restart(
    *,
    stage: str,
    ctx: BottomRestartContext,
    cfg: PbtConsolidationConfig,
) -> tuple[bool, str]:
    """Return (allow_restart, block_reason). Empty block_reason when allowed."""
    if not cfg.enabled or stage == STAGE_EXPLORATION:
        return True, ""

    override = bool(ctx.collapse_override and ctx.hard_collapse)

    if stage == STAGE_CONSOLIDATION:
        if ctx.promising_protected and not override:
            return False, "consolidation_promising_protected"
        if ctx.floor_recovery_protected and not override:
            return False, "consolidation_floor_recovery_protected"
        peak = float(ctx.recent_peak_nsucc_mean)
        if peak > 1e-9:
            retention = float(ctx.nsucc_mean) / peak
            if retention >= float(cfg.stage2_retention_threshold) and not override:
                return False, "consolidation_retention_protected"
        return True, ""

    if stage == STAGE_SELECTION:
        # A separately-qualified rescue may bypass the normal late-stage
        # hard-collapse-only gate; selection/source/copy remain in the
        # observer so this never re-enables mutation or generic churn.
        if ctx.late_degradation_rescue:
            return True, "late_degradation_rescue"
        if cfg.stage3_hard_collapse_only:
            if ctx.hard_collapse or override:
                return True, ""
            if float(ctx.nsucc_mean) < float(cfg.collapse_nsucc_mean_threshold):
                return True, ""
            return False, "selection_hard_collapse_only"
        return True, ""

    return True, ""


def consolidation_meta(
    raw_frame: int,
    cfg: PbtConsolidationConfig,
    *,
    curriculum_progress: float | None = None,
) -> dict[str, Any]:
    stage = resolve_pbt_stage(raw_frame, cfg, curriculum_progress=curriculum_progress)
    return {
        "stage": stage,
        "stage_id": stage_id(stage),
        "raw_frame": int(raw_frame),
        "curriculum_progress": (
            float(curriculum_progress) if curriculum_progress is not None else -1.0
        ),
    }


__all__ = [
    "STAGE_EXPLORATION",
    "STAGE_CONSOLIDATION",
    "STAGE_SELECTION",
    "STAGE_IDS",
    "PbtConsolidationConfig",
    "BottomRestartContext",
    "resolve_pbt_stage",
    "stage_id",
    "effective_tick_interval",
    "effective_mutation_rate",
    "evaluate_consolidated_tick",
    "should_allow_bottom_restart",
    "consolidation_meta",
]
