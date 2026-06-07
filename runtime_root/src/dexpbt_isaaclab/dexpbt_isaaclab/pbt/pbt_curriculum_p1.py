"""P1 curriculum / stability helpers (mastery gate, collapse freeze, rolling windows)."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class WindowSample:
    env_frames: int
    nsucc_mean: float
    nsucc_max: float
    success_tolerance: float
    pbt_iteration: int = 0


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

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


def write_curriculum_control(path: Path, state: CurriculumControlState) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(state.to_json(), indent=2))
    tmp.replace(path)


def read_curriculum_control(path: Path) -> CurriculumControlState | None:
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text())
        known = {f.name for f in CurriculumControlState.__dataclass_fields__.values()}
        return CurriculumControlState(**{k: v for k, v in data.items() if k in known})
    except Exception:
        return None


def post_shrink_consolidation_active(
    frames_since_restart: int,
    control: CurriculumControlState,
) -> bool:
    until = int(control.post_shrink_consolidation_until_restart_step)
    return until > 0 and int(frames_since_restart) < until


def mark_shrink_consumed(
    path: Path,
    *,
    frames_since_restart: int,
    consolidation_env_frames: int,
    num_envs: int,
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
        shrink_factor=curriculum_shrink_factor,
        collapse_freeze_active=control.collapse_freeze_active,
        post_shrink_consolidation_until_restart_step=int(
            control.post_shrink_consolidation_until_restart_step
        ),
    )
    if did_shrink and curriculum_control_path is not None:
        mark_shrink_consumed(
            curriculum_control_path,
            frames_since_restart=frames_since_restart,
            consolidation_env_frames=post_shrink_consolidation_env_frames,
            num_envs=num_envs,
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
