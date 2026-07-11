"""P1 curriculum / stability helpers (mastery gate, collapse freeze, rolling windows)."""

from __future__ import annotations

import json
import math
import os
import time
import uuid as _uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


SKILL_DELTA_WINDOW = 6
SKILL_DELTA_REF = 5.0
SKILL_DELTA_IMPROVE_EPS = 0.05
SKILL_DELTA_STABLE_POS_EPS = 0.02


@dataclass
class WindowSample:
    env_frames: int
    nsucc_mean: float
    nsucc_max: float
    success_tolerance: float
    pbt_iteration: int = 0
    lineage_id: str | None = None


@dataclass
class RollingNsuccWindow:
    """Fixed-length deque of PBT-tick samples per policy."""

    max_len: int = 8
    samples: list[WindowSample] = field(default_factory=list)

    def append(self, sample: WindowSample) -> None:
        self.samples.append(sample)
        if len(self.samples) > self.max_len:
            self.samples = self.samples[-self.max_len :]

    def last_n(self, n: int) -> list[WindowSample]:
        if n <= 0:
            return []
        return self.samples[-n:]

    def rolling_mean(self, n: int) -> float:
        xs = self.last_n(n)
        if not xs:
            return float("nan")
        return float(sum(s.nsucc_mean for s in xs) / len(xs))

    def drop_ratio_vs_prior(self, n: int = 2) -> float:
        xs = self.last_n(n)
        if len(xs) < 2:
            return 0.0
        prev, curr = xs[-2], xs[-1]
        if prev.nsucc_mean <= 1e-9:
            return 0.0
        return max(0.0, 1.0 - curr.nsucc_mean / prev.nsucc_mean)

    def tolerance_stable_for_intervals(self, n: int = 1) -> bool:
        xs = self.last_n(n + 1)
        if len(xs) < n + 1:
            return False
        tail = xs[-(n + 1) :]
        ref = tail[0].success_tolerance
        return all(abs(s.success_tolerance - ref) < 1e-9 for s in tail)


def compute_skill_delta(
    window: RollingNsuccWindow | None,
    *,
    window_len: int = SKILL_DELTA_WINDOW,
    interval_steps: int,
    delta_ref: float = SKILL_DELTA_REF,
) -> float | None:
    """Return bounded skill_delta in [-1, 1], or None when causally invalid.

    INVALID (None) when the window spans a lineage change or curriculum shift.
    Computed only from PBT tick buffer samples, never from live rollout noise.
    """
    if window is None or not window.samples:
        return None
    samples = window.last_n(int(window_len))
    if len(samples) < int(window_len):
        return None
    lineage_ids = {s.lineage_id for s in samples}
    if None in lineage_ids or len(lineage_ids) != 1:
        return None
    tol_bins = {round(float(s.success_tolerance), 6) for s in samples}
    if len(tol_bins) != 1:
        return None
    tick = max(int(interval_steps), 1)
    ref = max(float(delta_ref), 1e-6)
    pf = np.array([float(s.env_frames) for s in samples], dtype=np.float64)
    y = np.array([float(s.nsucc_mean) for s in samples], dtype=np.float64)
    alpha = 2.0 / (len(samples) + 1.0)
    ema = np.empty_like(y)
    ema[0] = y[0]
    for i in range(1, len(y)):
        ema[i] = alpha * y[i] + (1.0 - alpha) * ema[i - 1]
    pf_var = float(np.var(pf))
    if pf_var <= 1e-12:
        return None
    slope = float(np.cov(pf, ema, bias=True)[0, 1] / pf_var)
    raw_delta = slope * float(tick)
    norm_delta = raw_delta / ref
    return float(max(-1.0, min(1.0, norm_delta)))


@dataclass
class CurriculumControlState:
    curriculum_mode: str = "legacy"
    allow_shrink: bool = False
    shrink_token: bool = False
    shrink_blocked_reason: str = ""
    mastery_gate_pass: bool = False
    mastery_consecutive_count: int = 0
    last_mastered_tolerance: float = 0.075
    collapse_freeze_active: bool = False
    collapse_freeze_until_env_frames: int = 0
    nsucc_mean_drop_ratio: float = 0.0
    recent_peak_nsucc_mean: float = 0.0
    post_shrink_freeze_env_frames: int = 0
    post_shrink_consolidation_until_restart_step: int = 0
    just_shrank: bool = False
    tolerance_stable_intervals: int = 0
    shrinks_in_window: int = 0
    forced_success_tolerance: float = 0.0
    collapse_rollback_active: bool = False
    early_collapse_triggered: bool = False
    hard_collapse_triggered: bool = False
    rollback_and_freeze_same_tick: bool = False
    rollback_cooldown_active: bool = False
    rollback_count: int = 0
    last_rollback_frame: int = -10**18
    rollback_reason: str = ""
    # Phase A4 P2 (2026-06-08): adaptive-fast curriculum -- optional fields
    # written by the observer when --curriculum_speed_mode=adaptive_fast.
    # Defaults are inert: zero values fall back to the legacy single global
    # shrink_factor / consolidation_env_frames so existing runs are
    # bit-for-bit unchanged.
    speed_mode: str = "normal"
    adaptive_shrink_factor: float = 0.0
    adaptive_consolidation_frames: int = 0
    shrink_reason: str = ""
    nsucc_mean_used_for_shrink: float = 0.0
    retention_used_for_shrink: float = 0.0
    blocked_by_recovery: bool = False
    blocked_by_collapse: bool = False
    blocked_by_consolidation: bool = False
    blocked_by_mastery_gate: bool = False
    blocked_by_adaptive_fast_low_nsucc: bool = False
    pre_shrink_tolerance: float = 0.0
    post_shrink_tolerance: float = 0.0
    # Phase A4 P3 (2026-06-09): source-safe curriculum inheritance audit.
    # Defaults are inert; populated by the observer at copy time when
    # --pbt_curriculum_inherit_mode=source_safe and consumed back via JSONL
    # for reporting. The runtime curriculum logic does not branch on these
    # fields; they only audit the most-recent copy event into this slot.
    inheritance_mode: str = "reset"
    inheritance_reason: str = ""
    inherited_tolerance: float = 0.0
    target_tolerance_before_copy: float = 0.0
    target_tolerance_after_copy: float = 0.0
    source_policy_idx_for_copy: int = -1
    source_checkpoint_type_for_copy: str = ""
    source_success_tolerance_for_copy: float = 0.0
    source_nsucc_mean_for_copy: float = 0.0
    source_nsucc_max_for_copy: float = 0.0
    source_retention_for_copy: float = 0.0
    source_safe_for_curriculum: bool = False
    strong_live_inherit_exact: bool = False
    recovery_consolidation_frames_selected: int = 0
    recovery_consolidation_source: str = ""
    # Phase A4 P4 (2026-06-09): floor-recovery-aware PBT audit fields.
    # Defaults inert; the runtime only consumes them when the matching
    # observer-side flag --pbt_floor_recovery_aware is true. They are
    # safe to write to the JSON on every tick because the reader-side
    # apply_env_tolerance_curriculum does not branch on them.
    floor_stage: str = "none"  # one of {"none","floor","hard_floor"}
    first_floor_frame: int = -1
    first_hard_floor_frame: int = -1
    frames_since_floor: int = 0
    floor_recovery_slope: float = 0.0
    floor_recovery_score: float = 0.0
    floor_recovery_score_real: float = 0.0
    floor_recovery_score_wm: float = 0.0
    floor_recovery_score_norm: float = 0.0
    floor_recovery_protect_until: int = 0
    best_floor_recovery_archive_used: bool = False
    # PBT-WM Strategy Zoo v1.0: WM curriculum controller audit fields.
    wm_curriculum_controller_active: bool = False
    wm_curriculum_eval_valid: bool = False
    wm_curriculum_drop_risk: float = 0.0
    wm_curriculum_success_prob: float = 0.0
    wm_curriculum_recovery_prob: float = 0.0
    wm_curriculum_uncertainty: float = 0.0
    wm_curriculum_blocked_shrink: bool = False
    wm_curriculum_freeze_until: int = 0
    wm_curriculum_fallback_reason_id: int = 0
    # PBT-WM Soft Activation Gate v1: tol-bin diagnostics (backward compatible).
    wm_curriculum_tol_bin: str = "le_004"
    wm_curriculum_blocked_shrink_tol_gt_006: float = 0.0
    wm_curriculum_blocked_shrink_tol_005_006: float = 0.0
    wm_curriculum_blocked_shrink_tol_004_005: float = 0.0
    wm_curriculum_blocked_shrink_tol_le_004: float = 0.0
    # Hybrid Curriculum Safety Patch (2026-06-15): audit fields. Defaults
    # inert; runtime gates only when matching observer flags are enabled.
    curriculum_safety_active: bool = False
    curriculum_safety_real_fallback_used: bool = False
    curriculum_safety_blocked_shrink: bool = False
    curriculum_safety_block_reason: str = ""
    curriculum_safety_global_retention: float = 0.0
    curriculum_safety_recent_drop_ratio: float = 0.0
    curriculum_safety_recent_slope: float = 0.0
    curriculum_safety_stable_ticks: int = 0
    curriculum_safety_repeated_collapse_count: int = 0
    curriculum_safety_danger_zone: bool = False
    curriculum_safety_imagined_valid: bool = False
    curriculum_safety_wm_risk: bool = False
    curriculum_safety_real_risk: bool = False
    curriculum_global_retention: float = 0.0
    curriculum_recent_peak_nsucc_mean_long: float = 0.0
    curriculum_strong_shrink_allowed: bool = True
    curriculum_strong_shrink_block_reason: str = ""
    curriculum_strong_shrink_global_gate_active: bool = False
    curriculum_danger_zone: bool = False
    curriculum_danger_tol: float = 0.03
    curriculum_danger_blocked_shrink: bool = False
    curriculum_danger_block_reason: str = ""
    curriculum_danger_stable_ticks: int = 0
    curriculum_danger_interval_mult: float = 1.0
    curriculum_last_mastered_tolerance: float = 0.0
    curriculum_last_mastered_frame: int = -1
    curriculum_last_mastered_nsucc_mean: float = 0.0
    curriculum_last_mastered_global_retention: float = 0.0
    curriculum_rollback_triggered: bool = False
    curriculum_rollback_from_tol: float = 0.0
    curriculum_rollback_to_tol: float = 0.0
    curriculum_rollback_freeze_until: int = 0
    curriculum_repeated_collapse_count: int = 0
    wm_imagined_valid: bool = False
    wm_imagined_uncertainty: float = 1.0
    wm_imagined_success_prob: float = 0.0
    wm_imagined_drop_risk: float = 0.0
    wm_imagined_recovery_prob: float = 0.0
    wm_imagined_invalid_reason: str = ""
    wm_imagined_shadow_block_decision: bool = False
    wm_imagined_shadow_block_reason: str = ""
    # Late-floor retention gate audit (CSER patch).
    late_floor_retention_active: bool = False
    late_floor_retention_retention: float = 0.0
    late_floor_retention_recent_peak: float = 0.0
    late_floor_retention_blocked_shrink: bool = False
    late_floor_retention_freeze_until: int = 0
    late_floor_retention_reason: str = ""
    # Pre-floor peak retention guard audit (CSER retention patch).
    pre_floor_peak_guard_active: bool = False
    pre_floor_peak_guard_blocked: bool = False
    pre_floor_peak_guard_retention: float = 0.0
    pre_floor_peak_guard_recent_peak: float = 0.0
    pre_floor_peak_guard_stable_ticks: int = 0
    pre_floor_peak_guard_reason: str = ""
    # CSER 2cm bridge audit/control.  Defaults keep existing profiles inert.
    same_tol_bin_id: int = -1
    same_tol_retention: float = 1.0
    same_tol_retention_valid: bool = False
    same_tol_peak_nsucc_mean: float = 0.0
    same_tol_sample_count: int = 0
    same_tol_stable_ticks: int = 0
    retention_reference_tolerance: float = 0.0
    cross_tol_peak_ignored: bool = False
    retention_tracker: dict[str, Any] = field(default_factory=dict)
    bridge_active: bool = False
    bridge_level: int = -1
    bridge_entry_frame: int = -1
    bridge_level_entry_frame: int = -1
    bridge_shrink_count: int = 0
    bridge_next_tolerance: float = 0.0
    bridge_shrink_blocked: bool = False
    bridge_block_reason: str = ""
    bridge_probation_active: bool = False
    bridge_probation_entry_frame: int = -1
    bridge_probation_rollback_blocked: bool = False
    bridge_probation_hard_collapse: bool = False
    bridge_probation_exit: bool = False
    # Bridge probation collapse audit (defaults inert for non-bridge profiles).
    bridge_probation_legacy_collapse_suppressed: bool = False
    bridge_probation_hard_collapse_triggered: bool = False
    bridge_probation_bad_streak: int = 0
    bridge_probation_rollback_reason: str = ""
    bridge_probation_suppressed_legacy_collapse_count: int = 0
    bridge_pending_next_tolerance: float = 0.0
    bridge_level_timer_pending: bool = False
    late_rescue_attempt_count: int = 0
    late_rescue_selected_count: int = 0
    late_rescue_executed_count: int = 0
    late_rescue_blocked_count: int = 0
    late_rescue_streak: int = 0
    late_rescue_last_frame: int = -1

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


class LateRescueControlMissingError(RuntimeError):
    """Target curriculum_control JSON is missing during a restart transaction."""


class LateRescueControlParseError(RuntimeError):
    """Target curriculum_control JSON cannot be parsed during a restart transaction."""


def _atomic_write_text(
    path: Path,
    payload: str,
    *,
    max_retries: int = 3,
    retry_sleep_s: float = 0.05,
) -> None:
    """Atomically write text via unique tmp + os.replace with retries."""
    path.parent.mkdir(parents=True, exist_ok=True)
    last_err: Exception | None = None
    for attempt in range(max(1, int(max_retries))):
        tmp = path.parent / f"{path.stem}.{os.getpid()}.{_uuid.uuid4().hex[:8]}.tmp"
        try:
            tmp.write_text(payload)
            os.replace(str(tmp), str(path))
            return
        except (FileNotFoundError, OSError) as err:
            last_err = err
            try:
                if tmp.exists():
                    tmp.unlink()
            except Exception:
                pass
            if attempt + 1 < int(max_retries):
                time.sleep(float(retry_sleep_s))
                continue
            raise
    if last_err is not None:
        raise last_err  # pragma: no cover (defensive)


def write_curriculum_control_dict(
    path: Path,
    data: dict[str, Any],
    *,
    max_retries: int = 3,
    retry_sleep_s: float = 0.05,
) -> None:
    """Atomically write a full control JSON dict (preserves unknown fields)."""
    _atomic_write_text(
        path,
        json.dumps(data, indent=2),
        max_retries=max_retries,
        retry_sleep_s=retry_sleep_s,
    )


def write_curriculum_control(
    path: Path,
    state: CurriculumControlState,
    *,
    max_retries: int = 3,
    retry_sleep_s: float = 0.05,
) -> None:
    """Phase A4 P4 (2026-06-09): per-writer unique tmp + os.replace + retry.

    Previously the writer used a shared ``path.with_suffix(".tmp")`` (e.g.
    ``000.tmp``) for every policy / supervisor restart. Multiple concurrent
    PBT writers that touched the same slot could observe a stale ``.tmp``
    being renamed away mid-flight, surfacing as
    ``FileNotFoundError: 000.tmp -> 000.json`` in the env-side reader.
    The race is purely an infra issue, not algorithm logic.

    Mitigation:
      * tmp filename = ``{stem}.{pid}.{uuid8}.tmp`` so concurrent writers
        never share a tmp path.
      * ``os.replace`` (atomic on POSIX) instead of ``Path.replace`` (same
        semantics; spelling the system call out for clarity).
      * Up to ``max_retries`` attempts on transient ``OSError`` /
        ``FileNotFoundError`` (e.g. someone else just renamed our tmp).
    """
    _atomic_write_text(
        path,
        json.dumps(state.to_json(), indent=2),
        max_retries=max_retries,
        retry_sleep_s=retry_sleep_s,
    )


def read_curriculum_control(path: Path) -> CurriculumControlState | None:
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text())
        known = {f.name for f in CurriculumControlState.__dataclass_fields__.values()}
        return CurriculumControlState(**{k: v for k, v in data.items() if k in known})
    except Exception:
        return None


def read_curriculum_control_required(path: Path) -> tuple[CurriculumControlState, dict[str, Any]]:
    """Fail-closed read for restart / late-rescue transactions.

    Returns ``(typed_state, raw_dict)`` so callers can update known fields while
    preserving unknown JSON keys on write-back.
    """
    if not path.is_file():
        raise LateRescueControlMissingError(
            f"curriculum_control missing: {path}"
        )
    try:
        raw = path.read_text()
        data = json.loads(raw)
        if not isinstance(data, dict):
            raise LateRescueControlParseError(
                f"curriculum_control is not a JSON object: {path}"
            )
    except LateRescueControlParseError:
        raise
    except Exception as exc:
        raise LateRescueControlParseError(
            f"curriculum_control parse failed: {path}: {exc}"
        ) from exc
    known = {f.name for f in CurriculumControlState.__dataclass_fields__.values()}
    control = CurriculumControlState(**{k: v for k, v in data.items() if k in known})
    return control, data


def post_shrink_consolidation_active(
    frames_since_restart: int,
    control: CurriculumControlState,
) -> bool:
    until = int(control.post_shrink_consolidation_until_restart_step)
    return until > 0 and int(frames_since_restart) < until


def apply_source_safe_consolidation_on_copy(
    path: Path,
    *,
    consolidation_env_frames: int,
    consolidation_source: str,
    num_envs: int,
    inherited_tolerance: float,
    inheritance_audit: dict[str, Any] | None = None,
    require_existing: bool = False,
) -> None:
    """Phase A4 P3 (2026-06-09): on a source_safe cross/recovery copy.

    1. Clear stale slot history that came from the previous owner of this
       policy slot:
         post_shrink_consolidation_until_restart_step = 0
         blocked_by_consolidation = False
    2. Re-arm a fresh source-aware recovery consolidation window starting
       from frame_since_restart=0 (the supervisor wipes that on relaunch):
         post_shrink_consolidation_until_restart_step = freeze_steps
       50M for stable source, 100M for unstable / missing metadata.
    3. Sync last_mastered_tolerance to the inherited tolerance so a later
       last_mastered cross-copy cascade stays consistent.
    4. Persist the inheritance audit fields for downstream JSONL reporting.

    The control JSON is the single source of truth for the env loop
    (`apply_env_tolerance_curriculum`); we do NOT add any --*_resume flag.

    When ``require_existing`` is True (restart / late-rescue transactions),
    missing or unparseable control fails closed instead of creating defaults.
    """
    from dexpbt_isaaclab.dexpbt_utils import env_frames_to_sim_steps

    if require_existing:
        control, raw = read_curriculum_control_required(path)
    else:
        raw = None
        control = read_curriculum_control(path) or CurriculumControlState()
    control.post_shrink_consolidation_until_restart_step = 0
    control.blocked_by_consolidation = False
    freeze_steps = env_frames_to_sim_steps(
        int(consolidation_env_frames), int(num_envs)
    )
    control.post_shrink_consolidation_until_restart_step = int(freeze_steps)
    control.post_shrink_freeze_env_frames = int(consolidation_env_frames)
    control.recovery_consolidation_frames_selected = int(consolidation_env_frames)
    control.recovery_consolidation_source = str(consolidation_source)
    if float(inherited_tolerance) > 0.0:
        control.last_mastered_tolerance = float(inherited_tolerance)
    if inheritance_audit:
        for k, v in inheritance_audit.items():
            if hasattr(control, k):
                setattr(control, k, v)
    if raw is not None:
        # Preserve unknown fields from the original JSON document.
        raw.update(control.to_json())
        write_curriculum_control_dict(path, raw)
    else:
        write_curriculum_control(path, control)


def mark_shrink_consumed(
    path: Path,
    *,
    frames_since_restart: int,
    consolidation_env_frames: int,
    num_envs: int,
    pre_shrink_tolerance: float = 0.0,
    post_shrink_tolerance: float = 0.0,
) -> None:
    """Env-side: consume shrink_token and start post-shrink consolidation."""
    from dexpbt_isaaclab.dexpbt_utils import env_frames_to_sim_steps

    control = read_curriculum_control(path) or CurriculumControlState()
    control.shrink_token = False
    control.allow_shrink = False
    control.just_shrank = True
    control.shrinks_in_window = max(1, int(control.shrinks_in_window))
    freeze_steps = env_frames_to_sim_steps(int(consolidation_env_frames), int(num_envs))
    control.post_shrink_consolidation_until_restart_step = int(frames_since_restart) + int(
        freeze_steps
    )
    if pre_shrink_tolerance > 0.0:
        control.pre_shrink_tolerance = float(pre_shrink_tolerance)
    if post_shrink_tolerance > 0.0:
        control.post_shrink_tolerance = float(post_shrink_tolerance)
    # Confirm bridge shrink consumption so the observer may advance the
    # per-level timer only after tolerance actually changed.
    if float(getattr(control, "bridge_next_tolerance", 0.0) or 0.0) > 1e-9:
        control.bridge_shrink_count = int(getattr(control, "bridge_shrink_count", 0) or 0) + 1
        control.bridge_next_tolerance = 0.0
        control.bridge_pending_next_tolerance = 0.0
        control.bridge_level_timer_pending = False
        control.just_shrank = True
    write_curriculum_control(path, control)


def evaluate_mastery_gate(
    window: RollingNsuccWindow,
    *,
    mastery_mean: float,
    mastery_max: float,
    consecutive_windows: int,
) -> tuple[bool, int, str]:
    """Return (gate_pass, consecutive_count, blocked_reason)."""
    need = max(1, int(consecutive_windows))
    recent = window.last_n(need)
    if len(recent) < need:
        return False, len(recent), "consecutive_insufficient"
    for i, s in enumerate(recent):
        if s.nsucc_mean < mastery_mean:
            return False, i, "mean_below_threshold"
        if s.nsucc_max < mastery_max:
            return False, i, "max_below_threshold"
    return True, need, "ok"


def evaluate_collapse_freeze(
    window: RollingNsuccWindow,
    *,
    drop_ratio_threshold: float,
    patience: int,
    recent_peak: float,
) -> tuple[bool, float, int]:
    """Return (should_freeze, drop_ratio, bad_streak).

    Used for both early (e.g. 30%% / 1 window) and hard (50%% / 2) collapse.
    """
    bad = 0
    last_drop = 0.0
    for s in window.last_n(max(patience, 1)):
        if recent_peak > 1e-9:
            drop = 1.0 - s.nsucc_mean / recent_peak
        else:
            drop = 0.0
        last_drop = drop
        if drop > drop_ratio_threshold:
            bad += 1
        else:
            bad = 0
    return bad >= max(1, patience), last_drop, bad


def apply_env_tolerance_curriculum(
    *,
    curriculum_mode: str,
    curriculum_control_path: Path | None,
    curriculum_shrink_factor: float,
    last_curriculum_update: int,
    frames_since_restart: int,
    curriculum_interval: int,
    prev_episode_successes,
    success_tolerance: float,
    initial_tolerance: float,
    target_tolerance: float,
    curriculum_freeze_frames_after_restart: int = 0,
    post_shrink_consolidation_env_frames: int = 100_000_000,
    num_envs: int = 1,
    pbt_single_shrink_per_window: bool = True,
) -> tuple[float, int]:
    """Env-side curriculum dispatch: legacy or mastery via control JSON."""
    from dexpbt_isaaclab.dexpbt_utils import dexpbt_tolerance_curriculum

    legacy_kwargs = dict(
        last_curriculum_update=last_curriculum_update,
        frames_since_restart=frames_since_restart,
        curriculum_interval=curriculum_interval,
        prev_episode_successes=prev_episode_successes,
        success_tolerance=success_tolerance,
        initial_tolerance=initial_tolerance,
        target_tolerance=target_tolerance,
        curriculum_freeze_frames_after_restart=curriculum_freeze_frames_after_restart,
    )
    if curriculum_mode != "mastery" or curriculum_control_path is None:
        return dexpbt_tolerance_curriculum(**legacy_kwargs)

    control = read_curriculum_control(curriculum_control_path)
    if control is None:
        return dexpbt_tolerance_curriculum(**legacy_kwargs)

    if control.forced_success_tolerance > 1e-9:
        forced = float(control.forced_success_tolerance)
        forced = min(forced, initial_tolerance)
        forced = max(forced, target_tolerance)
        if abs(forced - success_tolerance) > 1e-12:
            return forced, frames_since_restart

    use_token = bool(pbt_single_shrink_per_window)
    may_shrink = control.shrink_token if use_token else control.allow_shrink

    # The bridge value is observer-authoritative.  It is deliberately applied
    # before multiplicative adaptive-fast logic so a rung cannot be skipped by
    # a stale factor in a restarted worker.
    if may_shrink and control.bridge_next_tolerance > 1e-9:
        bridged = max(float(control.bridge_next_tolerance), target_tolerance)
        bridged = min(bridged, initial_tolerance)
        if bridged < float(success_tolerance) - 1e-12:
            mark_shrink_consumed(
                curriculum_control_path,
                frames_since_restart=frames_since_restart,
                consolidation_env_frames=max(
                    int(post_shrink_consolidation_env_frames),
                    int(control.adaptive_consolidation_frames or 0),
                ),
                num_envs=num_envs,
                pre_shrink_tolerance=float(success_tolerance),
                post_shrink_tolerance=bridged,
            )
            return bridged, frames_since_restart

    # Phase A4 P2 (2026-06-08): adaptive-fast may override the shrink factor
    # and consolidation length per-tick via the observer-written control
    # state. A zero/empty value preserves legacy behavior.
    effective_shrink_factor = float(curriculum_shrink_factor)
    if float(control.adaptive_shrink_factor or 0.0) > 0.0:
        effective_shrink_factor = float(control.adaptive_shrink_factor)
    effective_consolidation_frames = int(post_shrink_consolidation_env_frames)
    if int(control.adaptive_consolidation_frames or 0) > 0:
        effective_consolidation_frames = int(control.adaptive_consolidation_frames)

    new_tol, lcu, did_shrink, _ = dexpbt_tolerance_curriculum_mastery(
        last_curriculum_update,
        frames_since_restart,
        curriculum_interval,
        prev_episode_successes,
        success_tolerance,
        initial_tolerance,
        target_tolerance,
        curriculum_freeze_frames_after_restart=curriculum_freeze_frames_after_restart,
        allow_shrink=may_shrink,
        shrink_factor=effective_shrink_factor,
        collapse_freeze_active=control.collapse_freeze_active,
        post_shrink_consolidation_until_restart_step=int(
            control.post_shrink_consolidation_until_restart_step
        ),
    )
    if did_shrink and curriculum_control_path is not None:
        mark_shrink_consumed(
            curriculum_control_path,
            frames_since_restart=frames_since_restart,
            consolidation_env_frames=effective_consolidation_frames,
            num_envs=num_envs,
            pre_shrink_tolerance=float(success_tolerance),
            post_shrink_tolerance=float(new_tol),
        )
    return new_tol, lcu


def dexpbt_tolerance_curriculum_mastery(
    last_curriculum_update: int,
    frames_since_restart: int,
    curriculum_interval: int,
    prev_episode_successes,
    success_tolerance: float,
    initial_tolerance: float,
    target_tolerance: float,
    *,
    curriculum_freeze_frames_after_restart: int = 0,
    allow_shrink: bool = False,
    shrink_factor: float = 0.95,
    collapse_freeze_active: bool = False,
    post_shrink_consolidation_until_restart_step: int = 0,
) -> tuple[float, int, bool, str]:
    """Mastery-mode shrink. Returns (tol, lcu, did_shrink, reason)."""
    if (
        post_shrink_consolidation_until_restart_step > 0
        and frames_since_restart < post_shrink_consolidation_until_restart_step
    ):
        return success_tolerance, last_curriculum_update, False, "post_shrink_consolidation"
    if collapse_freeze_active:
        return success_tolerance, frames_since_restart, False, "collapse_freeze_active"
    if (
        curriculum_freeze_frames_after_restart > 0
        and frames_since_restart < curriculum_freeze_frames_after_restart
    ):
        return success_tolerance, frames_since_restart, False, "post_restart_freeze"

    if frames_since_restart - last_curriculum_update < curriculum_interval:
        return success_tolerance, last_curriculum_update, False, "interval"

    if not allow_shrink:
        return success_tolerance, last_curriculum_update, False, "mastery_gate"

    mean_nsucc = float(prev_episode_successes.mean().item())
    if mean_nsucc <= 0.0:
        return success_tolerance, last_curriculum_update, False, "batch_mean_zero"

    new_tol = max(float(shrink_factor) * success_tolerance, target_tolerance)
    new_tol = min(new_tol, initial_tolerance)
    if abs(new_tol - success_tolerance) < 1e-12:
        return success_tolerance, last_curriculum_update, False, "at_target"
    return new_tol, frames_since_restart, True, "shrank"


def is_stable_source_window(
    window: RollingNsuccWindow,
    *,
    min_mean: float,
    min_max: float,
    max_drop_ratio: float,
    drop_windows: int = 2,
) -> tuple[bool, str]:
    if not window.samples:
        return False, "no_window"
    last = window.samples[-1]
    if last.nsucc_mean < min_mean:
        return False, "stable_mean"
    if last.nsucc_max < min_max:
        return False, "stable_max"
    if window.drop_ratio_vs_prior(drop_windows) > max_drop_ratio:
        return False, "drop_ratio"
    return True, "ok"


def is_usable_live_window(
    window: RollingNsuccWindow,
    *,
    min_mean: float = 1.0,
    min_max: float = 10.0,
    max_drop_ratio: float = 1.0,
) -> tuple[bool, str]:
    """L2 usable_live: some signal, not necessarily stable."""
    if not window.samples:
        return False, "no_window"
    last = window.samples[-1]
    if last.nsucc_mean < float(min_mean):
        return False, "usable_mean"
    if last.nsucc_max < float(min_max):
        return False, "usable_max"
    drop_ratio = window.drop_ratio_vs_prior(2)
    if drop_ratio > float(max_drop_ratio):
        return False, "usable_drop_ratio"
    return True, "ok"


def qualifies_best_stable_save(
    window: RollingNsuccWindow,
    *,
    min_mean: float,
    min_max: float,
    max_drop_ratio: float,
    stable_tolerance_intervals: int = 1,
    required_stable_windows: int = 2,
    consolidation_active: bool = False,
) -> tuple[bool, str]:
    if not window.samples:
        return False, "no_window"
    if consolidation_active:
        return False, "consolidation_active"
    need = max(1, int(required_stable_windows))
    recent = window.last_n(need)
    if len(recent) < need:
        return False, "not_two_stable_windows"
    for sample in recent:
        if sample.nsucc_mean < min_mean:
            return False, "mean"
        if sample.nsucc_max < min_max:
            return False, "max"
    last = window.samples[-1]
    if window.drop_ratio_vs_prior(2) > max_drop_ratio:
        return False, "drop_rate"
    if not window.tolerance_stable_for_intervals(stable_tolerance_intervals):
        return False, "tolerance_not_stable"
    # reject just-shrink window: tolerance changed in most recent interval
    if len(window.samples) >= 2:
        if abs(window.samples[-1].success_tolerance - window.samples[-2].success_tolerance) > 1e-9:
            return False, "just_shrank"
    return True, "ok"


CURRICULUM_RESUME_SNAPSHOT_NAME = "curriculum_resume_snapshot.json"


def curriculum_resume_snapshot_path(policy_run_dir: Path) -> Path:
    return Path(policy_run_dir) / CURRICULUM_RESUME_SNAPSHOT_NAME


def write_curriculum_resume_snapshot(
    policy_run_dir: Path,
    *,
    curriculum: dict[str, Any],
    env_frames: int,
    policy_idx: int,
) -> None:
    """Persist env-side curriculum for supervisor crash-resume (not PBT exit 42)."""
    path = curriculum_resume_snapshot_path(policy_run_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "policy_idx": int(policy_idx),
        "env_frames": int(env_frames),
        "timestamp": time.time(),
        "curriculum": {
            key: curriculum[key]
            for key in (
                "success_tolerance",
                "initial_tolerance",
                "target_tolerance",
                "last_curriculum_update",
                "frame_since_restart",
                "curriculum_freeze_frames_after_restart",
            )
            if key in curriculum
        },
    }
    tmp = path.parent / f"{path.stem}.{os.getpid()}.{_uuid.uuid4().hex[:8]}.tmp"
    tmp.write_text(json.dumps(payload, indent=2))
    os.replace(str(tmp), str(path))


def load_curriculum_resume_snapshot(policy_run_dir: Path) -> dict[str, Any] | None:
    path = curriculum_resume_snapshot_path(policy_run_dir)
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text())
    except Exception:
        return None
    cur = data.get("curriculum")
    if not isinstance(cur, dict):
        return None
    return data


def should_apply_supervisor_curriculum_resume(
    *,
    has_checkpoint: bool,
    success_tolerance_resume: float | None,
    restart_source_policy: int | None,
    env_success_tolerance: float,
    env_initial_tolerance: float,
    snapshot: dict[str, Any] | None,
) -> bool:
    """Restore snapshot tolerance after supervisor resume, not PBT cross-copy."""
    if not has_checkpoint or success_tolerance_resume is not None:
        return False
    if restart_source_policy is not None:
        return False
    if snapshot is None:
        return False
    cur = snapshot.get("curriculum") or {}
    snap_tol = float(cur.get("success_tolerance", float("nan")))
    if not math.isfinite(snap_tol):
        return False
    initial = float(env_initial_tolerance)
    if snap_tol >= initial - 1e-6:
        return False
    if float(env_success_tolerance) < initial - 1e-6:
        return False
    return True


def apply_curriculum_resume_env_state(
    raw_env: Any,
    curriculum: dict[str, Any],
) -> dict[str, Any]:
    resume_log: dict[str, Any] = {}
    if "success_tolerance" in curriculum and hasattr(raw_env, "success_tolerance"):
        raw_env.success_tolerance = float(curriculum["success_tolerance"])
        resume_log["success_tolerance"] = raw_env.success_tolerance
    if "initial_tolerance" in curriculum and hasattr(raw_env, "initial_tolerance"):
        raw_env.initial_tolerance = float(curriculum["initial_tolerance"])
        resume_log["initial_tolerance"] = raw_env.initial_tolerance
    if "target_tolerance" in curriculum and hasattr(raw_env, "target_tolerance"):
        raw_env.target_tolerance = float(curriculum["target_tolerance"])
        resume_log["target_tolerance"] = raw_env.target_tolerance
    if "last_curriculum_update" in curriculum and hasattr(raw_env, "last_curriculum_update"):
        raw_env.last_curriculum_update = int(curriculum["last_curriculum_update"])
        resume_log["last_curriculum_update"] = raw_env.last_curriculum_update
    if "frame_since_restart" in curriculum and hasattr(raw_env, "frame_since_restart"):
        raw_env.frame_since_restart = int(curriculum["frame_since_restart"])
        resume_log["frame_since_restart"] = raw_env.frame_since_restart
    if (
        "curriculum_freeze_frames_after_restart" in curriculum
        and hasattr(raw_env, "curriculum_freeze_frames_after_restart")
    ):
        raw_env.curriculum_freeze_frames_after_restart = int(
            curriculum["curriculum_freeze_frames_after_restart"]
        )
        resume_log["curriculum_freeze_frames_after_restart"] = (
            raw_env.curriculum_freeze_frames_after_restart
        )
    return resume_log
