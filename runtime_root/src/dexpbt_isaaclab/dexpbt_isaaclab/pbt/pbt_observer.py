"""DexPBT runtime observer for rl_games with decentralized filesystem coordination."""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import shutil
import sys
import time
import uuid as _uuid_mod
from pathlib import Path
from typing import Any

import numpy as np
import torch
from rl_games.common.algo_observer import IsaacAlgoObserver

from dexpbt_isaaclab.dexpbt_utils import (
    PBT_RANKING_METRIC_SUCCESS_AWARE,
    PBT_RANKING_METRICS,
    compute_pbt_ranking_score,
    env_frames_to_sim_steps,
    split_r_meta_components,
)
from dexpbt_isaaclab.pbt.pbt_curriculum_p1 import (
    CurriculumControlState,
    LateRescueControlMissingError,
    LateRescueControlParseError,
    RollingNsuccWindow,
    WindowSample,
    SKILL_DELTA_IMPROVE_EPS,
    SKILL_DELTA_STABLE_POS_EPS,
    SKILL_DELTA_WINDOW,
    apply_source_safe_consolidation_on_copy,
    evaluate_collapse_freeze,
    evaluate_mastery_gate,
    is_stable_source_window,
    is_usable_live_window,
    post_shrink_consolidation_active,
    qualifies_best_stable_save,
    read_curriculum_control,
    read_curriculum_control_required,
    write_curriculum_control,
    write_curriculum_control_dict,
    write_curriculum_resume_snapshot,
)
from dexpbt_isaaclab.pbt.pbt_tolerance_bridge import (
    DEFAULT_BRIDGE_LADDER,
    DEFAULT_RETENTION_BINS,
    ToleranceBinRetentionTracker,
    bridge_level_id,
    next_bridge_tolerance,
    parse_float_tuple,
    previous_bridge_tolerance,
)
from dexpbt_isaaclab.pbt.tolerance_conditioned_archive import select_compatible_archive
from dexpbt_isaaclab.pbt.late_degradation_rescue import (
    control_has_collapse_signal,
    select_late_degradation_rescue_source,
    source_has_collapse_signal,
)

# WM imports (lazy — only needed when enable_wm=True)
_WM_COUNTERFACTUAL = None
_WM_POSTERIOR = None

from dexpbt_isaaclab.pbt.pbt_consolidation_schedule import (
    BottomRestartContext,
    PbtConsolidationConfig,
    evaluate_consolidated_tick,
    effective_mutation_rate,
    resolve_pbt_stage,
    should_allow_bottom_restart,
    stage_id,
)
from dexpbt_isaaclab.pbt.pbt_control_plane import (
    PBT_SNAPSHOT_SCHEMA_VERSION,
    PbtStateSnapshot,
    apply_pbt_state_snapshot,
    assert_pbt_clock_invariants,
    derive_lineage_frame,
    derive_pbt_iteration,
    evaluate_pbt_tick_schedule,
    pbt_copy_allowed_for_adaptive_rescue_target,
    pbt_copy_allowed_for_live_source,
    pbt_copy_allowed_for_target,
    pbt_population_frame,
    population_frame_target_reached,
    pbt_raw_training_frame,
    pbt_scheduler_sync_clock,
    read_latest_commit_info,
    restore_pbt_tick_state_from_workspace,
    snapshot_commit_barrier,
    validate_snapshot_commit,
)


def _sample_factor(change_min: float, change_max: float) -> float:
    factor = random.uniform(change_min, change_max)
    return 1.0 / factor if random.random() < 0.5 else factor


def _mutate_discount(value: float, change_min: float, change_max: float) -> float:
    return float(min(0.9999, max(0.9, value * _sample_factor(change_min, change_max))))


def _mutate_positive(value: float, change_min: float, change_max: float, minimum: float) -> float:
    return float(max(minimum, value * _sample_factor(change_min, change_max)))


def _mutate_eps_clip(value: float, change_min: float, change_max: float) -> float:
    return float(min(0.4, max(0.01, value * _sample_factor(change_min, change_max))))


def _mutate_mini_epochs(value: int) -> int:
    choices = [1, 2] if value <= 2 else [max(1, value - 1), value, value + 1]
    return int(random.choice(choices))


# Keys explicitly excluded from PBT mutation.
# These are method-configuration switches (WM/CF) that MUST stay identical across
# the population for a controlled "baseline vs new method" comparison.
#
# - enable_wm / enable_counterfactual: method-level switches (bool already skipped, listed for clarity)
# - wm_strategy: posterior selector handles strategy selection, not mutation
# - wm_ensemble_size: architectural; mutating would change WM capacity mid-training
# - cf_interval_gens: counterfactual cadence is a methodology constant, not a hyperparameter
_NON_MUTABLE_KEYS: set[str] = {
    "enable_wm",
    "enable_counterfactual",
    "wm_strategy",
    "wm_ensemble_size",
    "cf_interval_gens",
    "cf_steps",
    "cf_num_envs",
    "cf_max_parallel_children",
    "cf_cuda_visible_devices",
    "cf_horizon",
    "require_counterfactual_runner",
    # Keep these fixed to stay close to paper's mutable set.
    "bounds_loss_coef",
    "learning_rate",
    # Keep action-velocity penalty fixed; paper PBT tuning focuses on RL/reward stages.
    "kuka_actions_penalty_scale",
    "allegro_actions_penalty_scale",
}

_PARAM_BOUNDS: dict[str, tuple[float, float]] = {
    # PPO / optimizer
    "gamma": (0.9, 0.9999),
    "kl_threshold": (1.0e-3, 3.0e-2),
    "grad_norm": (0.25, 5.0),
    "critic_coef": (1.0, 8.0),
    "bounds_loss_coef": (1.0e-5, 1.0e-3),
    "e_clip": (0.01, 0.4),
    "learning_rate": (1.0e-5, 5.0e-4),
    # Reward shaping (paper Table II stays in moderate ranges)
    "alpha_reach": (10.0, 200.0),
    "alpha_pick": (5.0, 80.0),
    "r_picked": (100.0, 600.0),
    "alpha_targ": (50.0, 500.0),
    "r_success": (500.0, 3000.0),
}


def _clamp_param(key: str, value: float | int) -> float | int:
    bounds = _PARAM_BOUNDS.get(key)
    if bounds is None:
        return value
    lo, hi = bounds
    if isinstance(value, int):
        return int(max(round(lo), min(round(hi), value)))
    return float(max(lo, min(hi, float(value))))


# Explicit PBT tier splits: (top, mid, bottom) policy counts by rank.
# For n=6: rank 1 | ranks 2-4 | ranks 5-6.
_TIER_SPLITS: dict[int, tuple[int, int, int]] = {
    6: (1, 3, 2),
    8: (2, 3, 3),
}


def _compute_tier_counts(population_size: int) -> tuple[int, int, int]:
    """Compute top/mid/bottom counts for PBT ranking tiers."""
    n = max(1, int(population_size))
    if n in _TIER_SPLITS:
        return _TIER_SPLITS[n]
    n_top = int(round(n * 0.3))
    n_mid = int(round(n * 0.4))
    n_bot = n - n_top - n_mid

    # Keep all tiers non-empty when population allows it.
    if n >= 3:
        if n_top < 1:
            n_top = 1
        if n_mid < 1:
            n_mid = 1
        n_bot = max(1, n - n_top - n_mid)
        while n_top + n_mid + n_bot > n:
            if n_mid > 1:
                n_mid -= 1
            elif n_top > 1:
                n_top -= 1
            else:
                n_bot -= 1
        while n_top + n_mid + n_bot < n:
            n_bot += 1
    else:
        # Degenerate tiny populations: keep at least one top and one bottom when possible.
        n_top = 1
        n_bot = 1 if n > 1 else 0
        n_mid = max(0, n - n_top - n_bot)
    return n_top, n_mid, n_bot


def _mutate_params(params: dict[str, float | int], mutation_rate: float, change_min: float, change_max: float) -> dict[str, float | int]:
    """Mutate the DexPBT subset of PPO/reward hyperparameters."""
    result = dict(params)
    for key, value in result.items():
        if key in _NON_MUTABLE_KEYS:
            continue
        if random.random() > mutation_rate:
            continue
        # Skip non-numeric params (strings like wm_strategy, booleans like enable_wm)
        if isinstance(value, str) or isinstance(value, bool):
            continue
        if key == "gamma":
            result[key] = _mutate_discount(float(value), change_min, change_max)
        elif key == "e_clip":
            result[key] = _mutate_eps_clip(float(value), change_min, change_max)
        elif key == "mini_epochs":
            result[key] = _mutate_mini_epochs(int(value))
        elif key == "learning_rate":
            result[key] = _mutate_positive(float(value), change_min, change_max, minimum=1e-5)
        elif isinstance(value, int):
            result[key] = max(1, int(round(int(value) * _sample_factor(change_min, change_max))))
        else:
            result[key] = _mutate_positive(float(value), change_min, change_max, minimum=1.0e-6)
        result[key] = _clamp_param(key, result[key])
    return result


def _get_raw_env(algo):
    """Walk rl_games env wrappers down to the underlying DexPBT env.

    Chain: algo.vec_env (rl_games IVecEnv) -> .env (IsaacRlgWrapper or RlGamesVecEnvWrapper)
        -> .env (gym OrderEnforcing) -> .unwrapped (DexPBTKukaAllegroEnv subclass)
    Falls back to whatever has `.success_tolerance` along the chain.
    """
    env = getattr(algo, "vec_env", None)
    visited: list = []
    for _ in range(8):  # bounded depth, avoid cycles
        if env is None or env in visited:
            break
        visited.append(env)
        if hasattr(env, "success_tolerance"):
            return env
        nxt = getattr(env, "unwrapped", None)
        if nxt is not None and nxt is not env:
            env = nxt
            continue
        nxt = getattr(env, "env", None)
        if nxt is not None and nxt is not env:
            env = nxt
            continue
        break
    return env



def _decision_uses_live_source(decision: dict[str, Any]) -> bool:
    """True when a PBT copy decision uses a live peer checkpoint.

    Live sources include ``live`` and ``latest_healthy`` checkpoint types, plus
    live-tier labels from ordinary and adaptive cascades. Archive tiers must
    return False so they skip the live skill_delta gate.
    """
    if not isinstance(decision, dict):
        return False
    ckpt_type = str(decision.get("checkpoint_type", "") or "")
    tier = str(decision.get("tier", "") or "")
    live_types = {"live", "latest_healthy"}
    live_tiers = {
        "strict_live",
        "strict_stable_live",
        "usable_live",
        "relaxed_live",
        "late_rescue_live",
        "adaptive_rescue_live",
    }
    archive_types = {
        "archive",
        "best_stable_archive",
        "best_floor_recovery_archive",
        "late_rescue_archive",
    }
    archive_tiers = {
        "adaptive_rescue_archive",
        "adaptive_rescue_best_stable_archive",
        "adaptive_rescue_floor_recovery_archive",
        "adaptive_rescue_best_nsucc_archive",
        "late_rescue_archive",
        "best_stable_archive",
        "best_floor_recovery_archive",
        "best_nsucc_archive",
    }
    if ckpt_type in archive_types or tier in archive_tiers:
        return False
    if ckpt_type in live_types or tier in live_tiers:
        return True
    # Adaptive / ordinary source_kind markers.
    kind = str(decision.get("adaptive_rescue_source_kind", "") or "")
    if kind == "live":
        return True
    if kind == "archive":
        return False
    return False



class DexPBTPbtAlgoObserver(IsaacAlgoObserver):
    """rl_games observer implementing decentralized DexPBT with ternary split."""

    def __init__(
        self,
        policy_idx: int,
        num_policies: int,
        mutable_params: dict[str, float | int],
        workspace: str = "dexpbt_single_arm_ws",
        train_dir: str = ".",
        task_name: str = "DexPBT",
        interval_steps: int = 20_000_000,
        start_after: int = 200_000_000,
        initial_delay: int = 50_000_000,
        mutation_rate: float = 0.2,
        change_min: float = 1.1,
        change_max: float = 1.5,
        ternary_split: bool = True,
        r_meta_window: bool = True,
        num_envs: int = 8192,
        pbt_ranking_metric: str = PBT_RANKING_METRIC_SUCCESS_AWARE,
        pbt_nsucc_zero_cap: float = 0.05,
        # Patch B (rev 3, 2026-05-30): low-nsucc cap covering 0<nsucc<X.
        # Defaults match the Patch A source gate so a worker that fails
        # the source gate also has its ranking score capped (defence-in-
        # depth: even if a stale meta.json sneaks in via fallback, the
        # ranking still demotes the bad worker out of `top` tier).
        pbt_nsucc_low_threshold: float = 1.0,
        pbt_nsucc_low_cap: float = 0.05,
        # Patch 2 (rev 2, 2026-05-30): anti-churn / N_adapt knobs.
        # See docstring of `_compute_eligibility` for the two-gate design.
        pbt_min_frames_as_source_after_restart: int = 50_000_000,
        pbt_min_frames_between_restarts: int = 50_000_000,
        # Patch A (rev 3, 2026-05-30): nsucc-threshold source gate. A worker
        # whose recent success signal is below these thresholds must not be
        # copied as a top source even if its reporting `r_meta_raw` (heavily
        # weighted by tolerance progress) ranks it highest. Empirically the
        # "low nsucc + high r_meta" leader is the dominant collapse vector
        # (see regrasping_cbwm_20260530_160513 policy_1: nsucc=0.17,
        # r_raw=0.94 still selected as PBT #1).
        pbt_min_nsucc_mean_as_source: float = 1.0,
        pbt_min_nsucc_max_as_source: float = 5.0,
        # Patch A (rev 3): when ALL top candidates fail the nsucc gate, the
        # default behaviour is now to SKIP this PBT replacement tick rather
        # than fall back to a low-nsucc source. Set True to restore legacy
        # fallback (collapse-era behaviour) for rollback / reproduction.
        pbt_allow_low_nsucc_source_fallback: bool = False,
        # Patch F (2026-05-31): mid-tier self-restart governor. The historic
        # ternary mid branch always self-mutated + relaunched every PBT tick,
        # which produced 100% self-restart traffic and broke continuity of
        # PPO state / curriculum. The defaults below DISABLE mid self-restart
        # (mid tier just keeps training); legacy behaviour can be restored
        # for ablation via --pbt_allow_mid_self_mutation true.
        pbt_disable_self_restart: bool = True,
        pbt_allow_mid_self_mutation: bool = False,
        pbt_max_self_restarts_per_policy: int = 1,
        pbt_min_frames_between_self_restarts: int = 100_000_000,
        # Patch G (2026-05-31): tiered source selection. The legacy bottom
        # branch went STRICT-or-SKIP, which produced 100% skipped ticks once
        # all live workers degraded together. Patch G adds two more tiers:
        #   L2 relaxed -- pick a "best-of-bad" live source under loosened gates
        #   L3 recover -- restore from best_nsucc/ peak archive (offline)
        # See `_select_source()` for the precise cascade.
        pbt_source_selection_mode: str = "tiered",
        pbt_relaxed_nsucc_mean_as_source: float = 0.5,
        pbt_relaxed_nsucc_max_as_source: float = 2.0,
        pbt_enable_best_nsucc_recovery: bool = True,
        pbt_recovery_min_nsucc_mean: float = 2.0,
        pbt_recovery_min_nsucc_max: float = 10.0,
        pbt_recovery_cooldown_frames: int = 200_000_000,
        # Patch H (2026-05-31): post-restart curriculum freeze. Every PBT
        # cross-copy / archive recovery propagates this value into the
        # spawned worker's env via train.py argparse. The env reads it
        # inside `_extra_curriculum` and skips tightening while frozen.
        pbt_curriculum_freeze_frames: int = 100_000_000,
        pbt_curriculum_freeze_on_recovery_frames: int = 100_000_000,
        # Patch P1 (2026-06-01): mastery curriculum + stable-source cascade.
        curriculum_mode: str = "legacy",
        pbt_source_require_stable_mean: bool = False,
        mastery_nsucc_mean: float = 4.0,
        mastery_nsucc_max: float = 20.0,
        mastery_consecutive_windows: int = 2,
        curriculum_shrink_factor: float = 0.95,
        pbt_allow_best_nsucc_archive_recovery: bool = False,
        curriculum_collapse_detect: bool = True,
        collapse_drop_ratio: float = 0.5,
        collapse_patience: int = 2,
        collapse_freeze_env_frames: int = 200_000_000,
        stable_source_min_mean: float = 3.0,
        stable_source_min_max: float = 20.0,
        stable_source_max_drop_ratio: float = 0.3,
        stable_source_drop_windows: int = 2,
        # Patch P2-A (2026-06-01): single-shot shrink token per PBT window.
        pbt_single_shrink_per_window: bool = True,
        post_shrink_consolidation_env_frames: int = 100_000_000,
        # Phase A4 P2 (2026-06-08): adaptive-fast curriculum (RSSM coupling=
        # none fastcurr 800M). All defaults below are inert when
        # curriculum_speed_mode='normal' (the default), so existing runs
        # are bit-for-bit unchanged.
        curriculum_speed_mode: str = "normal",
        curriculum_fast_strong_mean: float = 30.0,
        curriculum_fast_mid_mean: float = 20.0,
        curriculum_fast_low_mean: float = 10.0,
        curriculum_fast_retention_threshold: float = 0.8,
        curriculum_fast_retention_window: int = 4,
        curriculum_fast_shrink_strong: float = 0.90,
        curriculum_fast_shrink_mid: float = 0.925,
        curriculum_fast_shrink_low: float = 0.95,
        curriculum_fast_consolidation_frames: int = 50_000_000,
        curriculum_fast_recovery_consolidation_frames: int = 100_000_000,
        curriculum_fast_target_tolerance_800m: float = 0.04,
        # Phase A4 P3 (2026-06-09): source-safe curriculum inheritance and
        # promising-worker anti-churn. Defaults are inert: 'reset' inherit
        # mode + protection off preserves legacy behavior bit-for-bit.
        pbt_curriculum_inherit_mode: str = "reset",
        pbt_source_safe_recovery_consolidation_frames: int = 50_000_000,
        pbt_source_unsafe_recovery_consolidation_frames: int = 100_000_000,
        pbt_source_safe_min_mean: float = 20.0,
        pbt_source_safe_min_retention: float = 0.7,
        pbt_protect_promising_workers: bool = False,
        pbt_promising_nsucc_mean: float = 20.0,
        pbt_promising_peak_mean: float = 25.0,
        pbt_promising_min_retention: float = 0.5,
        pbt_promising_protect_frames: int = 150_000_000,
        pbt_promising_allow_hard_collapse_override: bool = True,
        # CSER patch: strong live cross-copy exact tolerance inheritance.
        pbt_strong_live_inherit_exact_enable: bool = True,
        pbt_strong_live_inherit_max_tol: float = 0.035,
        pbt_strong_live_inherit_min_mean: float = 30.0,
        pbt_strong_live_inherit_min_max: float = 35.0,
        # CSER patch: late-floor retention gate.
        late_floor_retention_gate_enable: bool = True,
        late_floor_retention_tol_threshold: float = 0.030,
        late_floor_retention_peak_threshold: float = 35.0,
        late_floor_retention_min_retention: float = 0.88,
        late_floor_retention_hard_min_retention: float = 0.75,
        late_floor_retention_patience: int = 2,
        late_floor_retention_freeze_frames: int = 250_000_000,
        late_floor_retention_block_shrink: bool = True,
        late_floor_retention_freeze_mutation: bool = True,
        late_floor_retention_rollback_enable: bool = False,
        # CSER retention patch: pre-floor peak retention guard.
        pre_floor_peak_guard_enable: bool = True,
        pre_floor_peak_guard_tol_threshold: float = 0.030,
        pre_floor_peak_guard_target_tol: float = 0.025,
        pre_floor_peak_guard_peak_threshold: float = 35.0,
        pre_floor_peak_guard_min_retention: float = 0.92,
        pre_floor_peak_guard_min_stable_ticks: int = 3,
        pre_floor_peak_guard_freeze_frames: int = 200_000_000,
        # Patch P2-B (2026-06-01): early collapse freeze + tolerance rollback.
        pbt_early_collapse_detect: bool = True,
        early_collapse_drop_ratio: float = 0.3,
        early_collapse_patience: int = 1,
        curriculum_rollback_cooldown_frames: int = 200_000_000,
        pbt_usable_live_min_mean: float = 1.5,
        pbt_usable_live_min_max: float = 20.0,
        pbt_usable_live_max_drop_ratio: float = 0.5,
        # Patch P2-C (2026-06-01): recovery bypass elimination burn-in.
        pbt_recovery_bypass_elimination_burnin: bool = True,
        # Patch P2-D (2026-06-01): recovery tolerance mode.
        pbt_recovery_tolerance_mode: str = "last_mastered",
        # Patch P2.1 (2026-06-02): cross-copy tolerance mode for live restarts.
        pbt_crosscopy_tolerance_mode: str = "checkpoint",
        pbt_recovery_source_cooldown_frames: int = 50_000_000,
        # Patch P2-E (2026-06-01): freeze hyperparam mutation after stable.
        pbt_freeze_mutation_after_stable_frames: int = 100_000_000,
        # Patch J (2026-05-31): fixed WM-strategy ablation. When
        # `wm_strategy_mode='fixed'`, the posterior selector and CF manager
        # are disabled; the worker keeps using `fixed_wm_strategy` for the
        # entire run. Use for ablation experiments (DexPBT baseline,
        # per-strategy curves).
        wm_strategy_mode: str = "bayesian",
        fixed_wm_strategy: str = "value_imag_h3",
        # Track C (2026-06-06): explicit Bayesian strategy pool. None means
        # "use the canonical 5-strategy default" (legacy behaviour).
        wm_strategy_pool: str | list[str] | None = None,
        # ------------------------------------------------------------------
        # Phase A4 P4 (2026-06-09): floor-recovery-aware PBT (Part A).
        # Defaults are inert: pbt_floor_recovery_aware=False keeps the entire
        # legacy bottom-replacement / archive cascade unchanged bit-for-bit.
        # ------------------------------------------------------------------
        pbt_floor_recovery_aware: bool = False,
        pbt_floor_tol_threshold: float = 0.030,
        pbt_hard_floor_tol_threshold: float = 0.025,
        pbt_floor_recovery_window: int = 4,
        pbt_floor_recovery_slope_weight: float = 0.5,
        pbt_floor_recovery_retention_weight: float = 0.3,
        pbt_floor_recovery_nsucc_max_weight: float = 0.2,
        pbt_floor_recovery_drop_penalty: float = 0.5,
        pbt_floor_recovery_protect_frames: int = 200_000_000,
        pbt_floor_recovery_min_nsucc_protect: float = 15.0,
        pbt_floor_recovery_min_slope_protect: float = 0.05,
        pbt_floor_recovery_min_retention_for_protect: float = 0.3,
        pbt_enable_best_floor_recovery_archive: bool = False,
        # Phase A4 P4 (2026-06-09): WM-guided PBT ranking (Part E).
        # Tri-state: off | shadow | active. shadow records would-select but
        # never overrides real ranking. active mixes WM into the score
        # subject to all kill switches. Default 'off' keeps legacy behavior.
        pbt_wm_score_mode: str = "off",
        pbt_use_wm_recovery_score: bool = False,  # legacy alias kept for compat
        pbt_wm_recovery_weight: float = 0.3,
        pbt_wm_success_weight: float = 0.2,
        pbt_wm_drop_penalty: float = 0.3,
        pbt_wm_uncertainty_penalty: float = 0.2,
        pbt_wm_max_uncertainty: float = 0.5,
        # Phase A4 P4 (2026-06-09): experimental stack preset (Item 10).
        # Convenience flag that does NOT bypass any per-feature kill switch;
        # train.py expands it into individual flags BEFORE constructing the
        # observer and prints the expansion. The observer just records that
        # the preset was active for audit.
        experimental_rssm_floor_recovery_stack: bool = False,
        # PBT-WM Strategy Zoo v1.0
        wm_strategy_profile_name: str = "legacy",
        wm_strategy_profile_id: int = 0,
        wm_strategy_profile_generation: int | None = None,
        wm_strategy_profile_parent: int | None = None,
        wm_strategy_profile_copied_from_source: int | None = None,
        wm_curriculum_controller_enable: bool = False,
        wm_curriculum_controller_start_after_frames: int = 500_000_000,
        wm_curriculum_controller_min_nsucc_mean: float = 10.0,
        wm_curriculum_controller_drop_risk_threshold: float = 0.45,
        wm_curriculum_controller_uncertainty_threshold: float = 0.55,
        wm_curriculum_controller_min_success_prob: float = 0.20,
        wm_curriculum_controller_freeze_frames: int = 100_000_000,
        wm_curriculum_controller_hard_block_shrink: bool = True,
        wm_curriculum_controller_use_recovery_prob: bool = False,
        # Hybrid Curriculum Safety Patch (2026-06-15): real-signal fallback,
        # strong-shrink gate, danger zone, rollback-to-last-mastered-safe.
        curriculum_safety_enable: bool = False,
        curriculum_safety_start_after_frames: int = 500_000_000,
        curriculum_safety_min_nsucc_mean: float = 10.0,
        curriculum_safety_global_retention_threshold: float = 0.80,
        curriculum_safety_drop_ratio_threshold: float = 0.30,
        curriculum_safety_slope_threshold: float = -2.0,
        curriculum_safety_window: int = 6,
        curriculum_safety_block_frames: int = 100_000_000,
        curriculum_safety_use_real_fallback: bool = False,
        curriculum_strong_shrink_global_retention_gate: bool = False,
        curriculum_strong_shrink_min_global_retention: float = 0.85,
        curriculum_strong_shrink_min_stable_ticks: int = 3,
        curriculum_strong_shrink_long_window: int = 12,
        curriculum_danger_zone_enable: bool = False,
        curriculum_danger_tol: float = 0.030,
        curriculum_danger_disable_strong_shrink: bool = True,
        curriculum_danger_min_stable_ticks: int = 4,
        curriculum_danger_shrink_interval_mult: float = 2.0,
        curriculum_danger_global_retention_threshold: float = 0.85,
        curriculum_rollback_on_repeated_collapse: bool = False,
        curriculum_rollback_collapse_patience: int = 2,
        curriculum_rollback_to_last_mastered: bool = False,
        curriculum_rollback_freeze_frames: int = 200_000_000,
        curriculum_last_mastered_min_nsucc_mean: float = 30.0,
        curriculum_last_mastered_min_global_retention: float = 0.85,
        curriculum_last_mastered_stable_ticks: int = 3,
        wm_imagined_diagnostic_enable: bool = False,
        wm_imagined_shadow_block_enable: bool = False,
        wm_imagined_shadow_uncertainty_threshold: float = 0.95,
        wm_imagined_valid_mode: str = "strict_uncertainty",
        wm_imagined_valid_uncertainty_threshold: float = 0.5,
        wm_imagined_soft_uncertainty_threshold: float = 0.95,
        wm_imagined_min_logit_abs_mean: float = 0.05,
        wm_imagined_min_head_signal_std: float = 0.02,
        wm_imagined_soft_validity_weight: float = 0.25,
        # PBT-WM Soft Activation Gate v1: WM PBT source-score soft gate
        # (confidence_only: eval_valid + time ramp + uncertainty, no tol gate).
        wm_pbt_source_gate_mode: str = "confidence_only",
        wm_pbt_source_start_after_frames: int = 300_000_000,
        wm_pbt_source_ramp_frames: int = 300_000_000,
        wm_curriculum_gate_mode: str = "shrink_candidate",
        wm_activation_enable: bool = True,
        pbt_wm_profile_copy_enable: bool = False,
        pbt_wm_profile_copy_after_frames: int = 1_000_000_000,
        pbt_wm_profile_copy_prob: float = 0.3,
        pbt_wm_profile_keep_diversity_min_profiles: int = 3,
        pbt_wm_profile_copy_min_nsucc_gap: float = 0.0,
        pbt_wm_profile_copy_min_source_retention: float = 0.0,
        pbt_wm_profile_copy_cooldown_frames: int = 0,
        pbt_wm_profile_copy_bottom_k: int = 0,
        pbt_wm_profile_copy_top_k: int = 0,
        # Deep-diag fix (2026-06-20): bottom-slot health gates. Defaults are
        # 0/() so existing v17.x runs are unaffected; v17.4+ launchers set
        # real values to break the bottom-slot death spiral.
        pbt_wm_profile_copy_source_min_nsucc_mean: float = 0.0,
        pbt_wm_profile_copy_source_min_frames: int = 0,
        pbt_wm_profile_copy_target_min_nsucc_mean: float = 0.0,
        pbt_wm_profile_copy_target_min_nsucc_max: float = 0.0,
        pbt_wm_profile_copy_locked_target_slots: (
            tuple[int, ...] | list[int] | frozenset[int] | str
        ) = (),
        profile_actor_distill_enabled: bool = False,
        profile_residual_apply_enabled: bool = False,
        profile_rssm_mppi_enabled: bool = False,
        profile_wm_pbt_score_enabled: bool = False,
        profile_wm_curriculum_controller_enabled: bool = False,
        profile_elite_bank_enabled: bool = False,
        profile_conservative_planner_enabled: bool = False,
        # CSER-v0: staged PBT consolidation schedule (defaults inert).
        pbt_consolidation_enable: bool = False,
        pbt_consolidation_stage1_end_frames: int = 2_200_000_000,
        pbt_consolidation_stage2_end_frames: int = 3_500_000_000,
        pbt_consolidation_stage3_end_frames: int = 5_000_000_000,
        pbt_consolidation_stage2_interval_mult: float = 2.0,
        pbt_consolidation_stage3_interval_mult: float = 4.0,
        pbt_consolidation_stage2_mutation_rate_mult: float = 0.5,
        pbt_consolidation_stage3_bottom_only: bool = True,
        pbt_consolidation_stage3_hard_collapse_only: bool = True,
        pbt_consolidation_stage2_retention_threshold: float = 0.85,
        pbt_consolidation_collapse_nsucc_mean_threshold: float = 1.0,
        frame_stop_enable: bool = False,
        target_population_frames: int = 0,
        frame_stop_grace_iterations: int = 2,
        # CSER 2cm bridge.  All flags default off so legacy profiles preserve
        # their previous control-plane behaviour.
        same_tol_retention_enable: bool = False,
        same_tol_retention_bins: str = "0.030,0.025,0.022,0.020,0.018",
        same_tol_retention_min_samples: int = 3,
        same_tol_retention_warmup_ticks: int = 3,
        same_tol_retention_reset_on_lineage_change: bool = True,
        same_tol_retention_reset_on_cross_copy: bool = True,
        same_tol_retention_keep_on_rollback: bool = False,
        bridge_curriculum_enable: bool = False,
        bridge_curriculum_start_tol: float = 0.025,
        bridge_curriculum_probation_tol: float = 0.0205,
        bridge_curriculum_target_tol: float = 0.018,
        bridge_curriculum_mode: str = "ladder",
        bridge_tolerance_ladder: str = "0.025,0.0235,0.022,0.021,0.0205,0.020,0.0195,0.019,0.0185,0.018",
        bridge_min_same_tol_retention: float = 0.90,
        bridge_min_stable_ticks: int = 4,
        bridge_min_frames_per_level: int = 150_000_000,
        bridge_post_shrink_consolidation_frames: int = 200_000_000,
        bridge_probation_enable: bool = False,
        bridge_probation_entry_tol: float = 0.0205,
        bridge_probation_frames: int = 400_000_000,
        bridge_probation_min_ticks: int = 4,
        bridge_probation_block_further_shrink: bool = True,
        bridge_probation_block_ordinary_rollback: bool = True,
        bridge_probation_allow_hard_collapse_rollback: bool = True,
        bridge_probation_hard_collapse_mean: float = 3.0,
        bridge_probation_hard_collapse_patience: int = 4,
        late_degradation_rescue_enable: bool = False,
        late_degradation_rescue_tol_threshold: float = 0.022,
        late_degradation_rescue_peak_threshold: float = 30.0,
        late_degradation_rescue_current_mean_threshold: float = 22.0,
        late_degradation_rescue_retention_threshold: float = 0.65,
        late_degradation_rescue_patience: int = 3,
        late_degradation_rescue_max_tol_gap: float = 0.002,
        late_degradation_rescue_cooldown_frames: int = 300_000_000,
        late_degradation_rescue_max_per_policy: int = 1,
        late_degradation_rescue_source_min_mean: float = 28.0,
        late_degradation_rescue_min_source_advantage: float = 8.0,
        late_degradation_rescue_source_min_retention: float = 0.85,
        late_degradation_rescue_reject_collapse_source: bool = True,
        late_degradation_rescue_require_source_control: bool = True,
        late_degradation_rescue_bypass_elimination_burnin: bool = False,
        # progress_refine: curriculum-progress consolidation schedule + adaptive
        # precision probation/rescue.  Defaults keep the frames schedule so all
        # existing profiles are unaffected.
        pbt_consolidation_schedule_mode: str = "frames",
        adaptive_consolidation_stage_consolidation_progress: float = 0.50,
        adaptive_consolidation_stage_selection_progress: float = 0.85,
        adaptive_precision_consolidation_enable: bool = False,
        adaptive_consolidation_tolerance_log_quant: float = 0.05,
        adaptive_probation_min_frames: int = 200_000_000,
        adaptive_probation_min_stable_ticks: int = 4,
        adaptive_probation_min_same_level_retention: float = 0.90,
        adaptive_probation_block_ordinary_rollback: bool = True,
        adaptive_probation_allow_hard_collapse_rollback: bool = True,
        adaptive_probation_hard_collapse_mean: float = 3.0,
        adaptive_probation_hard_collapse_patience: int = 4,
        adaptive_rescue_enable: bool = False,
        adaptive_rescue_retention_threshold: float = 0.70,
        adaptive_rescue_degradation_patience: int = 3,
        adaptive_rescue_population_quantile: float = 0.60,
        adaptive_rescue_source_top_fraction: float | None = None,
        adaptive_rescue_source_min_retention: float = 0.85,
        adaptive_rescue_min_source_advantage_sigma: float = 1.0,
        adaptive_rescue_max_log_tol_gap: float = 0.20,
        adaptive_rescue_cooldown_frames: int = 300_000_000,
        adaptive_rescue_max_per_lineage: int = 1,
        adaptive_rescue_require_source_control: bool = True,
        adaptive_rescue_reject_collapse_source: bool = True,
        adaptive_probation_rollback_max_log_gap: float = 0.30,
        adaptive_probation_freeze_only_max_ticks: int = 3,
        adaptive_rescue_source_min_samples: int = 3,
        adaptive_rescue_source_min_stable_ticks: int = 2,
        adaptive_rescue_source_max_staleness_frames: int = 40_000_000,
        adaptive_rescue_require_lineage_match_with_checkpoint: bool = True,
        adaptive_rescue_archive_enable: bool = True,
        adaptive_rescue_archive_max_age_frames: int = 1_000_000_000,
        adaptive_rescue_archive_require_retention: bool = True,
        adaptive_rescue_archive_min_nsucc_mean: float = 0.0,
        adaptive_rescue_archive_min_nsucc_max: float = 0.0,
        adaptive_rescue_archive_require_age: bool = True,
        adaptive_rescue_archive_require_collapse_metadata: bool = True,
        adaptive_rescue_archive_require_quality_score: bool = True,
        adaptive_rescue_archive_min_population_samples: int = 2,
        adaptive_rescue_archive_advantage_sigma_floor: float = 1.0,
        adaptive_rescue_archive_min_absolute_advantage: float = 5.0,
        adaptive_rescue_bypass_elimination_burnin: bool = False,
        adaptive_rescue_bypass_on_hard_collapse: bool = True,
        adaptive_rescue_min_frames_before_emergency_bypass: int = 40_000_000,
        adaptive_rescue_emergency_require_archive_or_stable_source: bool = True,
        adaptive_lineage_diversity_enable: bool = False,
        adaptive_min_distinct_lineages_exploration: int = 3,
        adaptive_min_distinct_lineages_consolidation: int = 2,
        adaptive_min_distinct_lineages_selection: int = 1,
        adaptive_replacement_budget_exploration: int = 1,
        adaptive_replacement_budget_consolidation: int = 1,
        adaptive_replacement_budget_selection: int = 2,
        adaptive_probation_stage_aware_enable: bool = False,
        adaptive_probation_min_frames_exploration: int = 75_000_000,
        adaptive_probation_min_frames_other: int = 100_000_000,
        adaptive_probation_max_frames: int = 200_000_000,
        adaptive_probation_min_stable_ticks_exploration: int = 2,
        adaptive_probation_min_stable_ticks_other: int = 3,
        adaptive_probation_early_exit_retention: float = 0.97,
        adaptive_probation_normal_exit_retention: float = 0.92,
        adaptive_probation_early_exit_nsucc_drop_frac: float = 0.15,
        adaptive_rescue_source_freshness_mode: str = "hard_max",
        adaptive_rescue_source_freshness_tau_frames: int = 80_000_000,
        adaptive_rescue_source_hard_max_age_frames: int = 160_000_000,
        wm_calibration_enable: bool = False,
        # Resume hooks; populated by train.py when the supervisor relaunches
        # this worker. They are None on first launch and on cold start.
        restart_count_resume: int | None = None,
        frames_at_restart_resume: int | None = None,
        last_mutation_frame_resume: int | None = None,
        pbt_lineage_id_resume: str | None = None,
        pbt_lineage_start_pf_resume: int | None = None,
        pbt_weight_age_frame_resume: int | None = None,
        # Patch F (2026-05-31): self/cross-restart accounting carried across exec.
        self_restart_count_resume: int | None = None,
        last_self_restart_frame_resume: int | None = None,
        cross_restart_count_resume: int | None = None,
        # Patch G (2026-05-31): recovery-from-archive accounting.
        recovery_restart_count_resume: int | None = None,
        last_recovery_frame_resume: int | None = None,
        last_profile_copy_frame_resume: int | None = None,
    ):
        super().__init__()
        self.policy_idx = policy_idx
        self.num_policies = num_policies
        self.mutable_params = dict(mutable_params)
        self.train_dir = Path(train_dir).resolve()
        ws_path = Path(workspace)
        if ws_path.is_absolute():
            self.workspace = ws_path.resolve()
        else:
            self.workspace = (self.train_dir / workspace).resolve()
        self.task_name = task_name
        self.interval_steps = interval_steps
        self.start_after = start_after
        self.initial_delay = initial_delay
        self.mutation_rate = mutation_rate
        self.change_min = change_min
        self.change_max = change_max
        self.ternary_split = ternary_split
        self.r_meta_window_enabled = r_meta_window
        self.num_envs = num_envs

        # Seed diversity audit (B1/B2): set by train.py after env creation
        # via set_seeding_audit(); included in every training_curve point.
        self._effective_seed: int | None = None
        self._env_randomization_hash: str = ""

        # Patch J (2026-05-31): fixed-strategy ablation. When `fixed`, the
        # observer disables CF / posterior entirely and forces every
        # restart (and the initial state) to use `fixed_wm_strategy`.
        if wm_strategy_mode not in ("bayesian", "fixed"):
            raise ValueError(
                f"Unknown wm_strategy_mode={wm_strategy_mode!r} "
                f"(expected bayesian | fixed)"
            )
        self.wm_strategy_mode = str(wm_strategy_mode)
        self.fixed_wm_strategy = str(fixed_wm_strategy)
        # Track C (2026-06-06): parse + validate the explicit strategy pool.
        # Accept either a list or a comma-separated string. None / empty
        # falls back to the canonical 5-strategy default.
        # Phase A4 (2026-06-08): canonical pool widened to include the new
        # sequence WM family. Ordering kept: legacy single-step strategies
        # first, then RSSM, then Transformer.
        _canonical_pool = [
            "none", "aux_low", "aux_mid", "value_imag_h1", "value_imag_h3",
            "rssm_aux", "rssm_belief_value",
            "trans_aux", "trans_belief_value",
        ]
        if wm_strategy_pool is None or wm_strategy_pool == "":
            _pool = list(_canonical_pool)
        elif isinstance(wm_strategy_pool, (list, tuple)):
            _pool = [str(s).strip() for s in wm_strategy_pool if str(s).strip()]
        else:
            _pool = [s.strip() for s in str(wm_strategy_pool).split(",") if s.strip()]
        _bad = [s for s in _pool if s not in _canonical_pool]
        if _bad:
            raise ValueError(
                f"wm_strategy_pool contains unknown strategy/strategies "
                f"{_bad!r}; must be a subset of {_canonical_pool}"
            )
        if not _pool:
            raise ValueError("wm_strategy_pool resolved to empty list")
        self.wm_strategy_pool = _pool
        # WM / Counterfactual state
        if self.wm_strategy_mode == "fixed":
            self._wm_strategy = self.fixed_wm_strategy
            # Force-disable CF: posterior cannot learn anything when only
            # one strategy is ever active. Persist into both the local
            # `mutable_params` (used by the lines below) AND the already
            # materialised `self.mutable_params` (used by downstream
            # consumers such as _init_wm_components, params published
            # back to the supervisor, etc.).
            mutable_params = dict(mutable_params)
            mutable_params["wm_strategy"] = self.fixed_wm_strategy
            mutable_params["enable_counterfactual"] = False
            self.mutable_params["wm_strategy"] = self.fixed_wm_strategy
            self.mutable_params["enable_counterfactual"] = False
        else:
            self._wm_strategy = mutable_params.get("wm_strategy", "none")
        self._wm_ensemble_size = mutable_params.get("wm_ensemble_size", 3)
        self._enable_wm = mutable_params.get("enable_wm", False)
        self._enable_counterfactual = mutable_params.get("enable_counterfactual", False)
        self._cf_manager = None
        self._posterior_selector = None
        self._wm_checkpoint_manager = None
        self._last_cf_result_step = 0
        self._last_consumed_cf_block_id = -1
        self._is_posterior_writer = bool(self.policy_idx == 0)
        self._last_posterior_update_policy_idx = -1
        self._cf_request_count = 0
        self._cf_result_count = 0
        self._cf_posterior_update_count = 0
        self._cf_skipped_stale_meta_count = 0
        self._last_cf_result_timestamp = 0.0

        # Phase A4 P4 (2026-06-09): floor-recovery-aware PBT params.
        # Defaults inert: pbt_floor_recovery_aware=False short-circuits all
        # new code paths. Values are still stored so JSONL / TB can audit
        # the configuration even when the feature is off.
        self.pbt_floor_recovery_aware = bool(pbt_floor_recovery_aware)
        self.pbt_floor_tol_threshold = float(pbt_floor_tol_threshold)
        self.pbt_hard_floor_tol_threshold = float(pbt_hard_floor_tol_threshold)
        self.pbt_floor_recovery_window = max(2, int(pbt_floor_recovery_window))
        self.pbt_floor_recovery_slope_weight = float(pbt_floor_recovery_slope_weight)
        self.pbt_floor_recovery_retention_weight = float(
            pbt_floor_recovery_retention_weight
        )
        self.pbt_floor_recovery_nsucc_max_weight = float(
            pbt_floor_recovery_nsucc_max_weight
        )
        self.pbt_floor_recovery_drop_penalty = float(pbt_floor_recovery_drop_penalty)
        self.pbt_floor_recovery_protect_frames = int(pbt_floor_recovery_protect_frames)
        self.pbt_floor_recovery_min_nsucc_protect = float(
            pbt_floor_recovery_min_nsucc_protect
        )
        self.pbt_floor_recovery_min_slope_protect = float(
            pbt_floor_recovery_min_slope_protect
        )
        self.pbt_floor_recovery_min_retention_for_protect = float(
            pbt_floor_recovery_min_retention_for_protect
        )
        self.pbt_enable_best_floor_recovery_archive = bool(
            pbt_enable_best_floor_recovery_archive
        )
        if self.pbt_floor_recovery_aware:
            if not (
                0.0 < self.pbt_hard_floor_tol_threshold
                <= self.pbt_floor_tol_threshold
            ):
                raise RuntimeError(
                    "pbt_hard_floor_tol_threshold must satisfy "
                    "0 < hard <= floor; got "
                    f"hard={self.pbt_hard_floor_tol_threshold} "
                    f"floor={self.pbt_floor_tol_threshold}"
                )
        # Phase A4 P4 (2026-06-09): WM-guided PBT ranking three-mode (Item 3).
        if pbt_wm_score_mode not in ("off", "shadow", "active"):
            raise RuntimeError(
                f"Unknown pbt_wm_score_mode={pbt_wm_score_mode!r} "
                f"(expected 'off' | 'shadow' | 'active')"
            )
        self.pbt_wm_score_mode = str(pbt_wm_score_mode)
        # Legacy bool flag (pbt_use_wm_recovery_score) is mapped onto the
        # tri-state mode for back-compat: True -> 'active' iff mode=='off',
        # otherwise it is ignored. Tests must be able to set the bool alone.
        if bool(pbt_use_wm_recovery_score) and self.pbt_wm_score_mode == "off":
            self.pbt_wm_score_mode = "active"
        self.pbt_use_wm_recovery_score = bool(pbt_use_wm_recovery_score)
        self.pbt_wm_recovery_weight = float(pbt_wm_recovery_weight)
        self.pbt_wm_success_weight = float(pbt_wm_success_weight)
        self.pbt_wm_drop_penalty = float(pbt_wm_drop_penalty)
        self.pbt_wm_uncertainty_penalty = float(pbt_wm_uncertainty_penalty)
        self.pbt_wm_max_uncertainty = float(pbt_wm_max_uncertainty)
        self.experimental_rssm_floor_recovery_stack = bool(
            experimental_rssm_floor_recovery_stack
        )
        # PBT-WM Strategy Zoo v1.0
        self.wm_strategy_profile_name = str(wm_strategy_profile_name or "legacy")
        self.wm_strategy_profile_id = int(wm_strategy_profile_id)
        self._wm_strategy_profile_generation = int(
            wm_strategy_profile_generation
            if wm_strategy_profile_generation is not None
            else 0
        )
        self._wm_strategy_profile_parent = int(
            wm_strategy_profile_parent
            if wm_strategy_profile_parent is not None
            else policy_idx
        )
        self._wm_strategy_profile_copied_from_source = bool(
            int(wm_strategy_profile_copied_from_source or 0)
        )
        self.wm_curriculum_controller_enable = bool(wm_curriculum_controller_enable)
        self.wm_curriculum_controller_start_after_frames = int(
            wm_curriculum_controller_start_after_frames
        )
        self.wm_curriculum_controller_min_nsucc_mean = float(
            wm_curriculum_controller_min_nsucc_mean
        )
        self.wm_curriculum_controller_drop_risk_threshold = float(
            wm_curriculum_controller_drop_risk_threshold
        )
        self.wm_curriculum_controller_uncertainty_threshold = float(
            wm_curriculum_controller_uncertainty_threshold
        )
        self.wm_curriculum_controller_min_success_prob = float(
            wm_curriculum_controller_min_success_prob
        )
        self.wm_curriculum_controller_freeze_frames = int(
            wm_curriculum_controller_freeze_frames
        )
        self.wm_curriculum_controller_hard_block_shrink = bool(
            wm_curriculum_controller_hard_block_shrink
        )
        self.wm_curriculum_controller_use_recovery_prob = bool(
            wm_curriculum_controller_use_recovery_prob
        )
        # PBT-WM Soft Activation Gate v1.
        self.wm_activation_enable = bool(wm_activation_enable)
        self.wm_pbt_source_gate_mode = str(wm_pbt_source_gate_mode)
        self.wm_pbt_source_start_after_frames = int(wm_pbt_source_start_after_frames)
        self.wm_pbt_source_ramp_frames = int(wm_pbt_source_ramp_frames)
        self.wm_curriculum_gate_mode = str(wm_curriculum_gate_mode)
        self._last_wm_source_activation_weight = 0.0
        self._last_wm_source_activation_nonzero = 0.0
        self._last_wm_source_unc_weight = 0.0
        self._last_wm_source_time_weight = 0.0
        self._last_wm_validity_weight = 0.0
        self._last_wm_score_effective_weight = 0.0
        self._last_wm_score_used_strict = 0.0
        self._last_wm_score_used_soft = 0.0
        self._last_wm_ranking_uncertainty_threshold = 0.0
        self._last_wm_ranking_uncertainty_gate_mode = "none"
        self.pbt_wm_profile_copy_enable = bool(pbt_wm_profile_copy_enable)
        self.pbt_wm_profile_copy_after_frames = int(pbt_wm_profile_copy_after_frames)
        self.pbt_wm_profile_copy_prob = float(pbt_wm_profile_copy_prob)
        self.pbt_wm_profile_keep_diversity_min_profiles = int(
            pbt_wm_profile_keep_diversity_min_profiles
        )
        self.pbt_wm_profile_copy_min_nsucc_gap = float(
            pbt_wm_profile_copy_min_nsucc_gap
        )
        self.pbt_wm_profile_copy_min_source_retention = float(
            pbt_wm_profile_copy_min_source_retention
        )
        self.pbt_wm_profile_copy_cooldown_frames = int(
            pbt_wm_profile_copy_cooldown_frames
        )
        self.pbt_wm_profile_copy_bottom_k = int(pbt_wm_profile_copy_bottom_k)
        self.pbt_wm_profile_copy_top_k = int(pbt_wm_profile_copy_top_k)
        # Deep-diag fix (2026-06-20): wire bottom-slot health gates.
        self.pbt_wm_profile_copy_source_min_nsucc_mean = float(
            pbt_wm_profile_copy_source_min_nsucc_mean
        )
        self.pbt_wm_profile_copy_source_min_frames = int(
            pbt_wm_profile_copy_source_min_frames
        )
        self.pbt_wm_profile_copy_target_min_nsucc_mean = float(
            pbt_wm_profile_copy_target_min_nsucc_mean
        )
        self.pbt_wm_profile_copy_target_min_nsucc_max = float(
            pbt_wm_profile_copy_target_min_nsucc_max
        )
        _locked_raw = pbt_wm_profile_copy_locked_target_slots
        if isinstance(_locked_raw, str):
            _locked_iter = (
                int(x) for x in _locked_raw.split(",") if x.strip()
            )
        else:
            _locked_iter = (int(x) for x in (_locked_raw or ()))
        self.pbt_wm_profile_copy_locked_target_slots = tuple(
            sorted({int(x) for x in _locked_iter})
        )
        self.profile_actor_distill_enabled = bool(profile_actor_distill_enabled)
        self.profile_residual_apply_enabled = bool(profile_residual_apply_enabled)
        self.profile_rssm_mppi_enabled = bool(profile_rssm_mppi_enabled)
        self.profile_wm_pbt_score_enabled = bool(profile_wm_pbt_score_enabled)
        self.profile_wm_curriculum_controller_enabled = bool(
            profile_wm_curriculum_controller_enabled
        )
        self.profile_elite_bank_enabled = bool(profile_elite_bank_enabled)
        self.profile_conservative_planner_enabled = bool(
            profile_conservative_planner_enabled
        )
        self._wm_curriculum_freeze_until: dict[int, int] = {}
        self._last_wm_curriculum_audit: dict[str, Any] = {}
        self._last_pbt_wm_profile_copy_decision: str = "disabled"
        self._last_pbt_wm_profile_diversity_count: int = 0
        self._last_profile_copy_eligible: bool = False
        self._last_profile_copy_reject_reason: str = ""
        self._last_profile_copy_source_policy: int = -1
        self._last_profile_copy_source_profile: str = ""
        self._last_profile_copy_target_old_profile: str = ""
        self._last_profile_copy_target_new_profile: str = ""
        self._last_profile_copy_weighted_prob: float = 0.0
        self._profile_copy_attempt_count: int = 0
        self._profile_copy_accept_count: int = 0
        self._profile_copy_reject_count: int = 0
        self._profile_copy_reject_reason_counts: dict[str, int] = {}
        self._weight_copy_count: int = 0
        self._weight_copy_with_profile_copy_count: int = 0
        self._weight_copy_without_profile_copy_count: int = 0
        # Per-policy floor-recovery state (Part A).
        self._first_floor_frame: dict[int, int] = {}
        self._first_hard_floor_frame: dict[int, int] = {}
        self._floor_recovery_protect_until: dict[int, int] = {}
        self._floor_recovery_score_history: dict[int, list[float]] = {}
        self._best_floor_recovery_state: dict[str, Any] | None = None
        # Per-tick imagined-evaluator output (Part D); cached inside
        # _ternary_pbt and consumed by ranking + audit. None means evaluator
        # disabled or its output is not trusted (high uncertainty / teacher
        # not promoted) -> WM ranking falls back to real-only.
        self._wm_imagined_eval_per_policy: dict[int, dict[str, float]] = {}
        # Counters for shadow-mode reporting.
        self._wm_shadow_observations: int = 0
        self._wm_shadow_changed_selection: int = 0

        self.curr_policy_workspace = self.workspace / f"{self.policy_idx:03d}"
        self.best_policy_workspace = self.workspace / "best"
        # Patch 5 (rev 2, 2026-05-30): separate, peak-protected best archives.
        # `best/` is kept for legacy compatibility (rank-by-`objective`), but
        # `best_r_meta/` and `best_nsucc/` are append-only (never overwritten)
        # and indexed by reporting-only `r_meta_raw` / `nsucc_mean` so that
        # peak DexPBT checkpoints survive even when ranking score collapses.
        self.best_r_meta_workspace = self.workspace / "best_r_meta"
        self.best_nsucc_workspace = self.workspace / "best_nsucc"
        self.best_stable_workspace = self.workspace / "best_stable"
        # Phase A4 P4 (2026-06-09): floor-stage recovery archive (Part A).
        # Created lazily; if the feature is off the dir is never populated.
        self.best_floor_recovery_workspace = self.workspace / "best_floor_recovery"
        self._curriculum_control_dir = self.workspace / "curriculum_control"
        # Phase A4 P3 (2026-06-09): one JSONL per copy/recovery event.
        self._pbt_copy_events_path = self.workspace / "metrics" / "pbt_copy_events.jsonl"

        if self._enable_wm:
            self._init_wm_components()

        self.algo = None
        self.pbt_iteration = -1
        self.initial_env_frames = -1
        # Patch 2 (rev 2, 2026-05-30): persistence of `last_mutation_frame`,
        # `_restart_count`, `_frames_at_restart` across supervisor restarts.
        # `__init__` used to reset these to 0/0/0 every relaunch, which
        # silently disabled the 50M initial_delay between PBT ticks and let
        # workers re-tick almost immediately after a restart. The supervisor
        # now forwards them via the new `*_resume` CLI args.
        if last_mutation_frame_resume is not None:
            self.last_mutation_frame = int(last_mutation_frame_resume)
        else:
            self.last_mutation_frame = 0
        self.pbt_iteration, self.last_mutation_frame = restore_pbt_tick_state_from_workspace(
            policy_workspace=self.curr_policy_workspace,
            self_pbt_iteration=self.pbt_iteration,
            last_mutation_frame=self.last_mutation_frame,
            last_mutation_frame_resume=last_mutation_frame_resume,
        )
        if restart_count_resume is not None:
            self._restart_count = int(restart_count_resume)
        else:
            self._restart_count = 0
        if frames_at_restart_resume is not None:
            self._frames_at_restart = int(frames_at_restart_resume)
        else:
            self._frames_at_restart = 0
        if pbt_lineage_id_resume:
            self._lineage_id = str(pbt_lineage_id_resume)
        else:
            self._lineage_id = str(_uuid_mod.uuid4())
        if pbt_lineage_start_pf_resume is not None:
            self._lineage_start_pf = int(pbt_lineage_start_pf_resume)
        else:
            self._lineage_start_pf = 0
        if pbt_weight_age_frame_resume is not None:
            self._weight_age_frame = int(pbt_weight_age_frame_resume)
        else:
            self._weight_age_frame = 0
        if pbt_lineage_id_resume is None:
            latest = self._load_latest_policy_meta()
            snap = PbtStateSnapshot.from_meta(latest)
            if snap is not None:
                self._lineage_id = snap.clocks.lineage_id
                self._lineage_start_pf = snap.clocks.lineage_start_pf
                self._weight_age_frame = snap.clocks.weight_age_frame
        self._r_meta_window: list[float] = []
        self.curr_target_objective_value = float("-inf")
        # Patch 1 (rev 2, 2026-05-30): r_meta_raw / pbt_ranking_score split.
        #   curr_r_meta_raw  -> alias of curr_target_objective_value (reporting)
        #   curr_ranking_score -> success-aware score used for PBT selection
        # Both are stored on every meta.json and training_curve point.
        self.pbt_ranking_metric = self._validate_ranking_metric(pbt_ranking_metric)
        self.pbt_nsucc_zero_cap = float(pbt_nsucc_zero_cap)
        # Patch B (rev 3, 2026-05-30): low-nsucc ranking cap parameters.
        self.pbt_nsucc_low_threshold = float(pbt_nsucc_low_threshold)
        self.pbt_nsucc_low_cap = float(pbt_nsucc_low_cap)
        self.curr_r_meta_raw: float = float("-inf")
        self.curr_ranking_score: float = float("-inf")
        self.curr_nsucc_mean: float = 0.0
        self.curr_nsucc_max: float = 0.0
        self.target_objective_known = False
        # Patch 2 (rev 2, 2026-05-30): anti-churn / N_adapt configuration.
        # See `_compute_eligibility`. Two independent gates:
        #   - source-cooldown: prevents a freshly-restarted worker from being
        #     copied out again as a top source.
        #   - elimination-cooldown: prevents a worker from being killed twice
        #     in a row before training has produced new evidence.
        self.min_frames_as_source_after_restart = int(pbt_min_frames_as_source_after_restart)
        self.min_frames_between_restarts = int(pbt_min_frames_between_restarts)
        # Patch A (rev 3, 2026-05-30): nsucc-based source gate config.
        self.min_nsucc_mean_as_source = float(pbt_min_nsucc_mean_as_source)
        self.min_nsucc_max_as_source = float(pbt_min_nsucc_max_as_source)
        self.allow_low_nsucc_source_fallback = bool(pbt_allow_low_nsucc_source_fallback)
        # Patch F (2026-05-31): self-restart governor state.
        self.disable_self_restart = bool(pbt_disable_self_restart)
        self.allow_mid_self_mutation = bool(pbt_allow_mid_self_mutation)
        self.max_self_restarts_per_policy = int(pbt_max_self_restarts_per_policy)
        self.min_frames_between_self_restarts = int(pbt_min_frames_between_self_restarts)
        if self_restart_count_resume is not None:
            self._self_restart_count = int(self_restart_count_resume)
        else:
            self._self_restart_count = 0
        if last_self_restart_frame_resume is not None:
            self._last_self_restart_frame = int(last_self_restart_frame_resume)
        else:
            self._last_self_restart_frame = -10**18  # never restarted
        if cross_restart_count_resume is not None:
            self._cross_restart_count = int(cross_restart_count_resume)
        else:
            self._cross_restart_count = 0
        # Patch G (2026-05-31): tiered source selection state.
        self.source_selection_mode = str(pbt_source_selection_mode)
        if self.source_selection_mode not in ("tiered", "strict"):
            raise ValueError(
                f"Unknown pbt_source_selection_mode={self.source_selection_mode!r}"
                f" (expected one of: tiered | strict)"
            )
        self.relaxed_nsucc_mean_as_source = float(pbt_relaxed_nsucc_mean_as_source)
        self.relaxed_nsucc_max_as_source = float(pbt_relaxed_nsucc_max_as_source)
        self.enable_best_nsucc_recovery = bool(pbt_enable_best_nsucc_recovery)
        self.recovery_min_nsucc_mean = float(pbt_recovery_min_nsucc_mean)
        self.recovery_min_nsucc_max = float(pbt_recovery_min_nsucc_max)
        self.recovery_cooldown_frames = int(pbt_recovery_cooldown_frames)
        # Patch H (2026-05-31): post-restart curriculum freeze.
        self.curriculum_freeze_frames = int(pbt_curriculum_freeze_frames)
        self.curriculum_freeze_on_recovery_frames = int(
            pbt_curriculum_freeze_on_recovery_frames
        )
        if curriculum_mode not in ("legacy", "mastery"):
            raise ValueError(
                f"Unknown curriculum_mode={curriculum_mode!r} "
                f"(expected legacy | mastery)"
            )
        self.curriculum_mode = str(curriculum_mode)
        self.source_require_stable_mean = bool(pbt_source_require_stable_mean)
        self.mastery_nsucc_mean = float(mastery_nsucc_mean)
        self.mastery_nsucc_max = float(mastery_nsucc_max)
        self.mastery_consecutive_windows = int(mastery_consecutive_windows)
        self.curriculum_shrink_factor = float(curriculum_shrink_factor)
        self.allow_best_nsucc_archive_recovery = bool(
            pbt_allow_best_nsucc_archive_recovery
        )
        self.curriculum_collapse_detect = bool(curriculum_collapse_detect)
        self.collapse_drop_ratio = float(collapse_drop_ratio)
        self.collapse_patience = int(collapse_patience)
        self.collapse_freeze_env_frames = int(collapse_freeze_env_frames)
        self.stable_source_min_mean = float(stable_source_min_mean)
        self.stable_source_min_max = float(stable_source_min_max)
        self.stable_source_max_drop_ratio = float(stable_source_max_drop_ratio)
        self.stable_source_drop_windows = int(stable_source_drop_windows)
        self.pbt_single_shrink_per_window = bool(pbt_single_shrink_per_window)
        self.post_shrink_consolidation_env_frames = int(
            post_shrink_consolidation_env_frames
        )
        # Phase A4 P2 (2026-06-08): adaptive-fast curriculum.
        if curriculum_speed_mode not in ("normal", "adaptive_fast"):
            raise RuntimeError(
                f"Unknown curriculum_speed_mode={curriculum_speed_mode!r} "
                f"(expected 'normal' | 'adaptive_fast')"
            )
        self.curriculum_speed_mode = str(curriculum_speed_mode)
        self.curriculum_fast_strong_mean = float(curriculum_fast_strong_mean)
        self.curriculum_fast_mid_mean = float(curriculum_fast_mid_mean)
        self.curriculum_fast_low_mean = float(curriculum_fast_low_mean)
        self.curriculum_fast_retention_threshold = float(
            curriculum_fast_retention_threshold
        )
        self.curriculum_fast_retention_window = max(
            1, int(curriculum_fast_retention_window)
        )
        self.curriculum_fast_shrink_strong = float(curriculum_fast_shrink_strong)
        self.curriculum_fast_shrink_mid = float(curriculum_fast_shrink_mid)
        self.curriculum_fast_shrink_low = float(curriculum_fast_shrink_low)
        self.curriculum_fast_consolidation_frames = int(
            curriculum_fast_consolidation_frames
        )
        self.curriculum_fast_recovery_consolidation_frames = int(
            curriculum_fast_recovery_consolidation_frames
        )
        self.curriculum_fast_target_tolerance_800m = float(
            curriculum_fast_target_tolerance_800m
        )
        if self.curriculum_speed_mode == "adaptive_fast":
            # Fail-fast: tier thresholds must be strictly decreasing
            # (strong > mid > low) and shrink factors must be in (0, 1].
            if not (
                self.curriculum_fast_strong_mean
                > self.curriculum_fast_mid_mean
                > self.curriculum_fast_low_mean
                > 0.0
            ):
                raise RuntimeError(
                    "adaptive_fast tier thresholds must satisfy "
                    "strong_mean > mid_mean > low_mean > 0; got "
                    f"strong={self.curriculum_fast_strong_mean} "
                    f"mid={self.curriculum_fast_mid_mean} "
                    f"low={self.curriculum_fast_low_mean}"
                )
            for name, val in (
                ("shrink_strong", self.curriculum_fast_shrink_strong),
                ("shrink_mid", self.curriculum_fast_shrink_mid),
                ("shrink_low", self.curriculum_fast_shrink_low),
            ):
                if not (0.0 < float(val) <= 1.0):
                    raise RuntimeError(
                        f"adaptive_fast {name}={val} must be in (0, 1]"
                    )
            if self.curriculum_fast_consolidation_frames <= 0:
                raise RuntimeError(
                    "adaptive_fast consolidation_frames must be > 0"
                )
            if self.curriculum_fast_recovery_consolidation_frames <= 0:
                raise RuntimeError(
                    "adaptive_fast recovery_consolidation_frames must be > 0"
                )
            if self.curriculum_mode != "mastery":
                raise RuntimeError(
                    "adaptive_fast requires --curriculum_mode=mastery; "
                    f"got {self.curriculum_mode!r}"
                )
        # Phase A4 P3 (2026-06-09): source-safe inheritance + promising guard.
        if pbt_curriculum_inherit_mode not in ("reset", "source_safe"):
            raise RuntimeError(
                f"Unknown pbt_curriculum_inherit_mode="
                f"{pbt_curriculum_inherit_mode!r} (expected 'reset' | "
                f"'source_safe')"
            )
        self.pbt_curriculum_inherit_mode = str(pbt_curriculum_inherit_mode)
        self.pbt_source_safe_recovery_consolidation_frames = int(
            pbt_source_safe_recovery_consolidation_frames
        )
        self.pbt_source_unsafe_recovery_consolidation_frames = int(
            pbt_source_unsafe_recovery_consolidation_frames
        )
        self.pbt_source_safe_min_mean = float(pbt_source_safe_min_mean)
        self.pbt_source_safe_min_retention = float(pbt_source_safe_min_retention)
        self.pbt_protect_promising_workers = bool(
            pbt_protect_promising_workers
        )
        self.pbt_promising_nsucc_mean = float(pbt_promising_nsucc_mean)
        self.pbt_promising_peak_mean = float(pbt_promising_peak_mean)
        self.pbt_promising_min_retention = float(pbt_promising_min_retention)
        self.pbt_promising_protect_frames = int(pbt_promising_protect_frames)
        self.pbt_promising_allow_hard_collapse_override = bool(
            pbt_promising_allow_hard_collapse_override
        )
        self.pbt_strong_live_inherit_exact_enable = bool(
            pbt_strong_live_inherit_exact_enable
        )
        self.pbt_strong_live_inherit_max_tol = float(pbt_strong_live_inherit_max_tol)
        self.pbt_strong_live_inherit_min_mean = float(pbt_strong_live_inherit_min_mean)
        self.pbt_strong_live_inherit_min_max = float(pbt_strong_live_inherit_min_max)
        self.late_floor_retention_gate_enable = bool(late_floor_retention_gate_enable)
        self.late_floor_retention_tol_threshold = float(
            late_floor_retention_tol_threshold
        )
        self.late_floor_retention_peak_threshold = float(
            late_floor_retention_peak_threshold
        )
        self.late_floor_retention_min_retention = float(
            late_floor_retention_min_retention
        )
        self.late_floor_retention_hard_min_retention = float(
            late_floor_retention_hard_min_retention
        )
        self.late_floor_retention_patience = int(late_floor_retention_patience)
        self.late_floor_retention_freeze_frames = int(
            late_floor_retention_freeze_frames
        )
        self.late_floor_retention_block_shrink = bool(late_floor_retention_block_shrink)
        self.late_floor_retention_freeze_mutation = bool(
            late_floor_retention_freeze_mutation
        )
        self.late_floor_retention_rollback_enable = bool(
            late_floor_retention_rollback_enable
        )
        self.pre_floor_peak_guard_enable = bool(pre_floor_peak_guard_enable)
        self.pre_floor_peak_guard_tol_threshold = float(
            pre_floor_peak_guard_tol_threshold
        )
        self.pre_floor_peak_guard_target_tol = float(pre_floor_peak_guard_target_tol)
        self.pre_floor_peak_guard_peak_threshold = float(
            pre_floor_peak_guard_peak_threshold
        )
        self.pre_floor_peak_guard_min_retention = float(
            pre_floor_peak_guard_min_retention
        )
        self.pre_floor_peak_guard_min_stable_ticks = max(
            1, int(pre_floor_peak_guard_min_stable_ticks)
        )
        self.pre_floor_peak_guard_freeze_frames = int(
            pre_floor_peak_guard_freeze_frames
        )
        if self.pbt_curriculum_inherit_mode == "source_safe":
            if self.pbt_source_safe_recovery_consolidation_frames <= 0:
                raise RuntimeError(
                    "source_safe inherit mode requires "
                    "--pbt_source_safe_recovery_consolidation_frames > 0; "
                    f"got {self.pbt_source_safe_recovery_consolidation_frames}"
                )
            if self.pbt_source_unsafe_recovery_consolidation_frames <= 0:
                raise RuntimeError(
                    "source_safe inherit mode requires "
                    "--pbt_source_unsafe_recovery_consolidation_frames > 0; "
                    f"got {self.pbt_source_unsafe_recovery_consolidation_frames}"
                )
            if not (0.0 < self.pbt_source_safe_min_retention <= 1.0):
                raise RuntimeError(
                    "--pbt_source_safe_min_retention must be in (0, 1]; "
                    f"got {self.pbt_source_safe_min_retention}"
                )
        if self.pbt_protect_promising_workers:
            if self.pbt_promising_protect_frames <= 0:
                raise RuntimeError(
                    "--pbt_promising_protect_frames must be > 0 when "
                    "--pbt_protect_promising_workers is enabled; got "
                    f"{self.pbt_promising_protect_frames}"
                )
            if not (0.0 < self.pbt_promising_min_retention <= 1.0):
                raise RuntimeError(
                    "--pbt_promising_min_retention must be in (0, 1]; "
                    f"got {self.pbt_promising_min_retention}"
                )
        self._promising_protect_until: dict[int, int] = {}
        self._curriculum_inherited_count: int = 0
        self._curriculum_reset_count: int = 0
        self._curriculum_missing_metadata_count: int = 0
        self._policy_nsucc_windows: dict[int, RollingNsuccWindow] = {}
        self._recent_peak_nsucc_mean: dict[int, float] = {}
        self._collapse_freeze_until: dict[int, int] = {}
        self._last_mastered_tolerance: dict[int, float] = {}
        self._collapse_detected_this_tick = False
        self._shrink_issued_this_window: dict[int, bool] = {}
        self._last_curriculum_pbt_iteration: int = -1
        self.pbt_early_collapse_detect = bool(pbt_early_collapse_detect)
        self.early_collapse_drop_ratio = float(early_collapse_drop_ratio)
        self.early_collapse_patience = int(early_collapse_patience)
        self.curriculum_rollback_cooldown_frames = int(
            curriculum_rollback_cooldown_frames
        )
        self._last_rollback_frame: dict[int, int] = {}
        self._rollback_count: dict[int, int] = {}
        self._shrinks_since_rollback: dict[int, int] = {}
        self._rollback_reason: dict[int, str] = {}
        # Hybrid Curriculum Safety Patch (2026-06-15).
        self.curriculum_safety_enable = bool(curriculum_safety_enable)
        self.curriculum_safety_start_after_frames = int(
            curriculum_safety_start_after_frames
        )
        self.curriculum_safety_min_nsucc_mean = float(
            curriculum_safety_min_nsucc_mean
        )
        self.curriculum_safety_global_retention_threshold = float(
            curriculum_safety_global_retention_threshold
        )
        self.curriculum_safety_drop_ratio_threshold = float(
            curriculum_safety_drop_ratio_threshold
        )
        self.curriculum_safety_slope_threshold = float(
            curriculum_safety_slope_threshold
        )
        self.curriculum_safety_window = max(2, int(curriculum_safety_window))
        self.curriculum_safety_block_frames = int(curriculum_safety_block_frames)
        self.curriculum_safety_use_real_fallback = bool(
            curriculum_safety_use_real_fallback
        )
        self.curriculum_strong_shrink_global_retention_gate = bool(
            curriculum_strong_shrink_global_retention_gate
        )
        self.curriculum_strong_shrink_min_global_retention = float(
            curriculum_strong_shrink_min_global_retention
        )
        self.curriculum_strong_shrink_min_stable_ticks = max(
            1, int(curriculum_strong_shrink_min_stable_ticks)
        )
        self.curriculum_strong_shrink_long_window = max(
            2, int(curriculum_strong_shrink_long_window)
        )
        self.curriculum_danger_zone_enable = bool(curriculum_danger_zone_enable)
        self.curriculum_danger_tol = float(curriculum_danger_tol)
        self.curriculum_danger_disable_strong_shrink = bool(
            curriculum_danger_disable_strong_shrink
        )
        self.curriculum_danger_min_stable_ticks = max(
            1, int(curriculum_danger_min_stable_ticks)
        )
        self.curriculum_danger_shrink_interval_mult = float(
            curriculum_danger_shrink_interval_mult
        )
        self.curriculum_danger_global_retention_threshold = float(
            curriculum_danger_global_retention_threshold
        )
        self.curriculum_rollback_on_repeated_collapse = bool(
            curriculum_rollback_on_repeated_collapse
        )
        self.curriculum_rollback_collapse_patience = max(
            1, int(curriculum_rollback_collapse_patience)
        )
        self.curriculum_rollback_to_last_mastered = bool(
            curriculum_rollback_to_last_mastered
        )
        self.curriculum_rollback_freeze_frames = int(
            curriculum_rollback_freeze_frames
        )
        self.curriculum_last_mastered_min_nsucc_mean = float(
            curriculum_last_mastered_min_nsucc_mean
        )
        self.curriculum_last_mastered_min_global_retention = float(
            curriculum_last_mastered_min_global_retention
        )
        self.curriculum_last_mastered_stable_ticks = max(
            1, int(curriculum_last_mastered_stable_ticks)
        )
        self.wm_imagined_diagnostic_enable = bool(wm_imagined_diagnostic_enable)
        self.wm_imagined_shadow_block_enable = bool(wm_imagined_shadow_block_enable)
        self.wm_imagined_shadow_uncertainty_threshold = float(
            wm_imagined_shadow_uncertainty_threshold
        )
        self.wm_imagined_valid_mode = str(wm_imagined_valid_mode)
        self.wm_imagined_valid_uncertainty_threshold = float(
            wm_imagined_valid_uncertainty_threshold
        )
        self.wm_imagined_soft_uncertainty_threshold = float(
            wm_imagined_soft_uncertainty_threshold
        )
        self.wm_imagined_min_logit_abs_mean = float(wm_imagined_min_logit_abs_mean)
        self.wm_imagined_min_head_signal_std = float(wm_imagined_min_head_signal_std)
        self.wm_imagined_soft_validity_weight = float(wm_imagined_soft_validity_weight)
        self._last_mastered_safe_tolerance: dict[int, float] = {}
        self._last_mastered_safe_frame: dict[int, int] = {}
        self._last_mastered_safe_nsucc_mean: dict[int, float] = {}
        self._last_mastered_safe_global_retention: dict[int, float] = {}
        self._repeated_collapse_count: dict[int, int] = {}
        self._curriculum_safety_freeze_until: dict[int, int] = {}
        self._curriculum_safety_events_path = (
            self.workspace / "metrics" / "curriculum_safety_events.jsonl"
        )
        self._last_curriculum_safety_audit: dict[str, Any] = {}
        self.frame_stop_enable = bool(frame_stop_enable)
        self.target_population_frames = max(0, int(target_population_frames))
        self.frame_stop_grace_iterations = max(0, int(frame_stop_grace_iterations))
        self._frame_stop_triggered = False
        self.same_tol_retention_enable = bool(same_tol_retention_enable)
        self.same_tol_retention_bins = parse_float_tuple(
            same_tol_retention_bins, DEFAULT_RETENTION_BINS
        )
        self.same_tol_retention_min_samples = max(1, int(same_tol_retention_min_samples))
        self.same_tol_retention_warmup_ticks = max(1, int(same_tol_retention_warmup_ticks))
        self.same_tol_retention_reset_on_lineage_change = bool(
            same_tol_retention_reset_on_lineage_change
        )
        self.same_tol_retention_reset_on_cross_copy = bool(
            same_tol_retention_reset_on_cross_copy
        )
        self.same_tol_retention_keep_on_rollback = bool(same_tol_retention_keep_on_rollback)
        self._same_tol_trackers: dict[int, ToleranceBinRetentionTracker] = {}
        self._bridge_state_restored: set[int] = set()
        self._last_confirmed_tolerance: dict[int, float] = {}
        self._bridge_shrink_count: dict[int, int] = {}
        self._late_rescue_attempt_count: dict[int, int] = {}
        self._late_rescue_selected_count: dict[int, int] = {}
        self._late_rescue_executed_count: dict[int, int] = {}
        self._late_rescue_blocked_count: dict[int, int] = {}
        self._pending_late_rescue: bool = False
        self.bridge_curriculum_enable = bool(bridge_curriculum_enable)
        self.bridge_curriculum_start_tol = float(bridge_curriculum_start_tol)
        self.bridge_curriculum_probation_tol = float(bridge_curriculum_probation_tol)
        self.bridge_curriculum_target_tol = float(bridge_curriculum_target_tol)
        self.bridge_curriculum_mode = str(bridge_curriculum_mode)
        self.bridge_tolerance_ladder = parse_float_tuple(
            bridge_tolerance_ladder, DEFAULT_BRIDGE_LADDER
        )
        self.bridge_min_same_tol_retention = float(bridge_min_same_tol_retention)
        self.bridge_min_stable_ticks = max(1, int(bridge_min_stable_ticks))
        self.bridge_min_frames_per_level = max(0, int(bridge_min_frames_per_level))
        self.bridge_post_shrink_consolidation_frames = max(
            0, int(bridge_post_shrink_consolidation_frames)
        )
        self.bridge_probation_enable = bool(bridge_probation_enable)
        self.bridge_probation_entry_tol = float(bridge_probation_entry_tol)
        self.bridge_probation_frames = max(0, int(bridge_probation_frames))
        self.bridge_probation_min_ticks = max(1, int(bridge_probation_min_ticks))
        self.bridge_probation_block_further_shrink = bool(bridge_probation_block_further_shrink)
        self.bridge_probation_block_ordinary_rollback = bool(
            bridge_probation_block_ordinary_rollback
        )
        self.bridge_probation_allow_hard_collapse_rollback = bool(
            bridge_probation_allow_hard_collapse_rollback
        )
        self.bridge_probation_hard_collapse_mean = float(bridge_probation_hard_collapse_mean)
        self.bridge_probation_hard_collapse_patience = max(
            1, int(bridge_probation_hard_collapse_patience)
        )
        self._bridge_entry_frame: dict[int, int] = {}
        self._bridge_level_entry_frame: dict[int, int] = {}
        self._bridge_probation_entry_frame: dict[int, int] = {}
        self._bridge_probation_bad_streak: dict[int, int] = {}
        self._bridge_probation_suppressed_legacy_collapse_count: dict[int, int] = {}
        self._bridge_probation_exit_cleared: set[int] = set()
        self.late_degradation_rescue_enable = bool(late_degradation_rescue_enable)
        self.late_degradation_rescue_tol_threshold = float(late_degradation_rescue_tol_threshold)
        self.late_degradation_rescue_peak_threshold = float(late_degradation_rescue_peak_threshold)
        self.late_degradation_rescue_current_mean_threshold = float(late_degradation_rescue_current_mean_threshold)
        self.late_degradation_rescue_retention_threshold = float(late_degradation_rescue_retention_threshold)
        self.late_degradation_rescue_patience = max(1, int(late_degradation_rescue_patience))
        self.late_degradation_rescue_max_tol_gap = float(late_degradation_rescue_max_tol_gap)
        self.late_degradation_rescue_cooldown_frames = int(late_degradation_rescue_cooldown_frames)
        self.late_degradation_rescue_max_per_policy = max(1, int(late_degradation_rescue_max_per_policy))
        self.late_degradation_rescue_source_min_mean = float(
            late_degradation_rescue_source_min_mean
        )
        self.late_degradation_rescue_min_source_advantage = float(
            late_degradation_rescue_min_source_advantage
        )
        self.late_degradation_rescue_source_min_retention = float(
            late_degradation_rescue_source_min_retention
        )
        self.late_degradation_rescue_reject_collapse_source = bool(
            late_degradation_rescue_reject_collapse_source
        )
        self.late_degradation_rescue_require_source_control = bool(
            late_degradation_rescue_require_source_control
        )
        self.late_degradation_rescue_bypass_elimination_burnin = bool(
            late_degradation_rescue_bypass_elimination_burnin
        )
        self._late_rescue_streak: dict[int, int] = {}
        self._late_rescue_count: dict[int, int] = {}
        self._late_rescue_frame: dict[int, int] = {}
        self._late_rescue_last_source_control_valid = 0.0
        self._late_rescue_last_source_collapse_rejected = 0.0
        self._late_rescue_last_burnin_bypassed = 0.0
        self.pbt_usable_live_min_mean = float(pbt_usable_live_min_mean)
        self.pbt_usable_live_min_max = float(pbt_usable_live_min_max)
        self.pbt_usable_live_max_drop_ratio = float(
            pbt_usable_live_max_drop_ratio
        )
        self.pbt_recovery_bypass_elimination_burnin = bool(
            pbt_recovery_bypass_elimination_burnin
        )
        if pbt_recovery_tolerance_mode not in (
            "checkpoint",
            "last_mastered",
            "relax_one_level",
        ):
            raise ValueError(
                f"Unknown pbt_recovery_tolerance_mode="
                f"{pbt_recovery_tolerance_mode!r}"
            )
        self.pbt_recovery_tolerance_mode = str(pbt_recovery_tolerance_mode)
        if pbt_crosscopy_tolerance_mode not in (
            "checkpoint",
            "last_mastered",
            "relax_one_level",
        ):
            raise ValueError(
                f"Unknown pbt_crosscopy_tolerance_mode="
                f"{pbt_crosscopy_tolerance_mode!r}"
            )
        self.pbt_crosscopy_tolerance_mode = str(pbt_crosscopy_tolerance_mode)
        self.pbt_recovery_source_cooldown_frames = int(
            pbt_recovery_source_cooldown_frames
        )
        self.pbt_freeze_mutation_after_stable_frames = int(
            pbt_freeze_mutation_after_stable_frames
        )
        self._mutation_frozen_until: dict[int, int] = {}
        self._late_floor_retention_patience_count: dict[int, int] = {}
        self._best_stable_recovery_selected_count = 0
        self._best_stable_recovery_executed_count = 0
        self._best_stable_recovery_skipped_count = 0
        self._last_best_stable_recovery_outcome: str = "none"
        self._recovery_audit_path = self.workspace / "recovery_audit.jsonl"
        if recovery_restart_count_resume is not None:
            self._recovery_restart_count = int(recovery_restart_count_resume)
        else:
            self._recovery_restart_count = 0
        if last_recovery_frame_resume is not None:
            self._last_recovery_frame = int(last_recovery_frame_resume)
        else:
            self._last_recovery_frame = -10**18
        if last_profile_copy_frame_resume is not None:
            self._last_profile_copy_frame = int(last_profile_copy_frame_resume)
        else:
            self._last_profile_copy_frame = -10**18
        # Source selection tier counters; bumped per-tick whenever the
        # bottom branch terminates in each tier.
        self._source_tier_counts: dict[str, int] = {
            "strict_live": 0,
            "strict_stable_live": 0,
            "usable_live": 0,
            "relaxed_live": 0,
            "archive": 0,
            "best_stable_archive": 0,
            "best_floor_recovery_archive": 0,
            "late_rescue_live": 0,
            "late_rescue_archive": 0,
            "adaptive_rescue_live": 0,
            "adaptive_rescue_archive": 0,
            "skipped": 0,
        }
        # The "checkpoint provenance" for the current tick. live|archive.
        # Forwarded to manifest so post-hoc analysis can tell apart cross-
        # copies from peak recoveries.
        self._last_source_checkpoint_type: str = "live"
        # The source-selection tier label used at the current PBT tick.
        # One of: "strict_live", "relaxed_live", "archive", "skipped".
        self._last_source_tier_label: str = "none"
        # Per-tick blocked counters; bumped whenever a self-restart was
        # attempted but rejected by disable/cooldown/max-count gate.
        self._self_restart_blocked_count = 0
        # Mid-tier self-mutation attempts (whether or not they led to an
        # actual relaunch). For Patch F audit: in `allow_mid_self_mutation=
        # True` mode this is 1:1 with actual restarts; in disabled mode
        # this stays 0 because we early-return before _mutate_params.
        self._mid_self_mutation_count = 0
        self._consolidation_restart_blocked_count = 0
        self.pbt_consolidation_cfg = PbtConsolidationConfig(
            enabled=bool(pbt_consolidation_enable),
            stage1_end_frames=int(pbt_consolidation_stage1_end_frames),
            stage2_end_frames=int(pbt_consolidation_stage2_end_frames),
            stage3_end_frames=int(pbt_consolidation_stage3_end_frames),
            stage2_interval_mult=float(pbt_consolidation_stage2_interval_mult),
            stage3_interval_mult=float(pbt_consolidation_stage3_interval_mult),
            stage2_mutation_rate_mult=float(pbt_consolidation_stage2_mutation_rate_mult),
            stage3_bottom_only=bool(pbt_consolidation_stage3_bottom_only),
            stage3_hard_collapse_only=bool(pbt_consolidation_stage3_hard_collapse_only),
            stage2_retention_threshold=float(pbt_consolidation_stage2_retention_threshold),
            collapse_nsucc_mean_threshold=float(
                pbt_consolidation_collapse_nsucc_mean_threshold
            ),
            schedule_mode=str(pbt_consolidation_schedule_mode),
            stage_consolidation_progress=float(
                adaptive_consolidation_stage_consolidation_progress
            ),
            stage_selection_progress=float(
                adaptive_consolidation_stage_selection_progress
            ),
        )
        # progress_refine: adaptive precision consolidation controller (probation
        # + retention-conditioned rescue).  Built lazily; state persists via the
        # curriculum-control JSON.  Inert unless explicitly enabled.
        self.adaptive_precision_enable = bool(adaptive_precision_consolidation_enable)
        self._adaptive_curriculum_progress: float | None = None
        try:
            from dexpbt_isaaclab.pbt.adaptive_precision_consolidation import (
                AdaptivePrecisionConfig,
            )

            self._adaptive_precision_cfg = AdaptivePrecisionConfig(
                enabled=self.adaptive_precision_enable,
                tolerance_log_quant=float(adaptive_consolidation_tolerance_log_quant),
                stage_consolidation_start=float(
                    adaptive_consolidation_stage_consolidation_progress
                ),
                stage_selection_start=float(
                    adaptive_consolidation_stage_selection_progress
                ),
                probation_min_frames=int(adaptive_probation_min_frames),
                probation_min_stable_ticks=int(adaptive_probation_min_stable_ticks),
                probation_min_same_level_retention=float(
                    adaptive_probation_min_same_level_retention
                ),
                probation_block_ordinary_rollback=bool(
                    adaptive_probation_block_ordinary_rollback
                ),
                probation_allow_hard_collapse_rollback=bool(
                    adaptive_probation_allow_hard_collapse_rollback
                ),
                probation_hard_collapse_mean=float(adaptive_probation_hard_collapse_mean),
                probation_hard_collapse_patience=int(
                    adaptive_probation_hard_collapse_patience
                ),
                rescue_enable=bool(adaptive_rescue_enable),
                rescue_retention_threshold=float(adaptive_rescue_retention_threshold),
                rescue_degradation_patience=int(adaptive_rescue_degradation_patience),
                rescue_source_top_fraction=float(
                    adaptive_rescue_source_top_fraction
                    if adaptive_rescue_source_top_fraction is not None
                    else max(0.0, min(1.0, 1.0 - float(adaptive_rescue_population_quantile)))
                ),
                rescue_population_quantile=float(adaptive_rescue_population_quantile),
                rescue_source_min_retention=float(adaptive_rescue_source_min_retention),
                rescue_min_source_advantage_sigma=float(
                    adaptive_rescue_min_source_advantage_sigma
                ),
                rescue_max_log_tol_gap=float(adaptive_rescue_max_log_tol_gap),
                rescue_cooldown_frames=int(adaptive_rescue_cooldown_frames),
                rescue_max_per_lineage=int(adaptive_rescue_max_per_lineage),
                rescue_require_source_control=bool(
                    adaptive_rescue_require_source_control
                ),
                rescue_reject_collapse_source=bool(
                    adaptive_rescue_reject_collapse_source
                ),
                rescue_source_min_samples=int(adaptive_rescue_source_min_samples),
                rescue_source_min_stable_ticks=int(adaptive_rescue_source_min_stable_ticks),
                rescue_source_max_staleness_frames=int(
                    adaptive_rescue_source_max_staleness_frames
                ),
                rescue_require_lineage_match_with_checkpoint=bool(
                    adaptive_rescue_require_lineage_match_with_checkpoint
                ),
                rescue_archive_enable=bool(adaptive_rescue_archive_enable),
                rescue_archive_max_age_frames=int(adaptive_rescue_archive_max_age_frames),
                rescue_archive_require_retention=bool(
                    adaptive_rescue_archive_require_retention
                ),
                rescue_archive_min_nsucc_mean=float(
                    adaptive_rescue_archive_min_nsucc_mean
                ),
                rescue_archive_min_nsucc_max=float(
                    adaptive_rescue_archive_min_nsucc_max
                ),
                rescue_archive_require_age=bool(adaptive_rescue_archive_require_age),
                rescue_archive_require_collapse_metadata=bool(
                    adaptive_rescue_archive_require_collapse_metadata
                ),
                rescue_archive_require_quality_score=bool(
                    adaptive_rescue_archive_require_quality_score
                ),
                rescue_archive_min_population_samples=int(
                    adaptive_rescue_archive_min_population_samples
                ),
                rescue_archive_advantage_sigma_floor=float(
                    adaptive_rescue_archive_advantage_sigma_floor
                ),
                rescue_archive_min_absolute_advantage=float(
                    adaptive_rescue_archive_min_absolute_advantage
                ),
                rescue_bypass_elimination_burnin=bool(
                    adaptive_rescue_bypass_elimination_burnin
                ),
                rescue_bypass_on_hard_collapse=bool(
                    adaptive_rescue_bypass_on_hard_collapse
                ),
                rescue_min_frames_before_emergency_bypass=int(
                    adaptive_rescue_min_frames_before_emergency_bypass
                ),
                rescue_emergency_require_archive_or_stable_source=bool(
                    adaptive_rescue_emergency_require_archive_or_stable_source
                ),
                probation_rollback_max_log_gap=float(
                    adaptive_probation_rollback_max_log_gap
                ),
                probation_freeze_only_max_ticks=int(
                    adaptive_probation_freeze_only_max_ticks
                ),
                lineage_diversity_enable=bool(adaptive_lineage_diversity_enable),
                min_distinct_lineages_exploration=int(
                    adaptive_min_distinct_lineages_exploration
                ),
                min_distinct_lineages_consolidation=int(
                    adaptive_min_distinct_lineages_consolidation
                ),
                min_distinct_lineages_selection=int(
                    adaptive_min_distinct_lineages_selection
                ),
                replacement_budget_exploration=int(
                    adaptive_replacement_budget_exploration
                ),
                replacement_budget_consolidation=int(
                    adaptive_replacement_budget_consolidation
                ),
                replacement_budget_selection=int(adaptive_replacement_budget_selection),
                probation_stage_aware_enable=bool(
                    adaptive_probation_stage_aware_enable
                ),
                probation_min_frames_exploration=int(
                    adaptive_probation_min_frames_exploration
                ),
                probation_min_frames_other=int(adaptive_probation_min_frames_other),
                probation_max_frames=int(adaptive_probation_max_frames),
                probation_min_stable_ticks_exploration=int(
                    adaptive_probation_min_stable_ticks_exploration
                ),
                probation_min_stable_ticks_other=int(
                    adaptive_probation_min_stable_ticks_other
                ),
                probation_early_exit_retention=float(
                    adaptive_probation_early_exit_retention
                ),
                probation_normal_exit_retention=float(
                    adaptive_probation_normal_exit_retention
                ),
                probation_early_exit_nsucc_drop_frac=float(
                    adaptive_probation_early_exit_nsucc_drop_frac
                ),
                rescue_source_freshness_mode=str(adaptive_rescue_source_freshness_mode),
                rescue_source_freshness_tau_frames=int(
                    adaptive_rescue_source_freshness_tau_frames
                ),
                rescue_source_hard_max_age_frames=int(
                    adaptive_rescue_source_hard_max_age_frames
                ),
            )
        except Exception as _apc_exc:
            _prof = str(getattr(self, "wm_strategy_profile_name", "") or "")
            if _prof in (
                "cser_dexpbt_progress_refine",
                "cser_dexpbt_readiness_adaptive",
            ):
                raise RuntimeError(
                    "AdaptivePrecisionConfig construction failed for formal "
                    f"profile {_prof}: {_apc_exc}"
                ) from _apc_exc
            print(f"[adaptive-precision] config build failed (disabled): {_apc_exc}")
            self._adaptive_precision_cfg = None
            self.adaptive_precision_enable = False
        # Per-policy adaptive precision runtime state (serialized to sidecar JSON).
        self._adaptive_probation: dict[int, Any] = {}
        self._adaptive_retention: dict[int, Any] = {}
        self._adaptive_degradation_streak: dict[int, int] = {}
        self._adaptive_last_rescue_frame: dict[int, int] = {}
        self._adaptive_rescue_executed: dict[int, int] = {}
        self._adaptive_rescue_executed_for_lineage: dict[int, int] = {}
        self._adaptive_lineage_id: dict[int, str] = {}
        self._adaptive_rescue_attempt_count: dict[int, int] = {}
        self._adaptive_rescue_selected_count: dict[int, int] = {}
        self._adaptive_rescue_blocked_count: dict[int, int] = {}
        self._adaptive_rescue_last_reason: dict[int, str] = {}
        self._adaptive_rescue_last_source: dict[int, int] = {}
        self._adaptive_rescue_last_source_rank: dict[int, int] = {}
        self._adaptive_rescue_last_advantage_sigma: dict[int, float] = {}
        self._adaptive_last_audit: dict[int, dict[str, Any]] = {}
        self._adaptive_freeze_only_streak: dict[int, int] = {}
        self._adaptive_rollback_freeze_only_count: dict[int, int] = {}
        self._adaptive_probation_timeout: dict[int, bool] = {}
        self._adaptive_probation_timeout_reason: dict[int, str] = {}
        self._adaptive_ordinary_replacements_this_tick: int = 0
        self._adaptive_diversity_blocked_count: int = 0
        self._adaptive_replacement_budget_blocked_count: int = 0
        self._last_adaptive_diversity_blocked: bool = False
        self._last_adaptive_distinct_lineage_count: int = 0
        self._last_adaptive_min_distinct_lineages: int = 0
        self._last_adaptive_replacement_budget: int = 0
        self._last_adaptive_replacement_budget_used: int = 0
        self._adaptive_hard_collapse_trigger_source: dict[int, str] = {}
        # Per-tick hard-collapse latch: frame at which adaptive hard collapse was
        # confirmed in `_build_curriculum_control_state`. Survives streak reset so
        # `_ternary_pbt` emergency bypass still sees the event in the same tick.
        self._adaptive_hard_collapse_event_frame: dict[int, int] = {}
        self._adaptive_peer_reject_counts: dict[int, int] = {}
        self._adaptive_rescue_live_selected: dict[int, int] = {}
        self._adaptive_rescue_archive_selected: dict[int, int] = {}
        self._adaptive_rescue_ordinary_fallback: dict[int, int] = {}
        self._adaptive_rescue_last_source_kind: dict[int, str] = {}
        self._adaptive_rescue_burnin_blocked: dict[int, int] = {}
        self._adaptive_rescue_burnin_bypassed: dict[int, int] = {}
        self._adaptive_rescue_emergency_bypass: dict[int, int] = {}
        self._adaptive_last_checkpoint_frame: dict[int, int] = {}
        self._adaptive_last_checkpoint_path: dict[int, str] = {}
        self._adaptive_sidecar_meta: dict[int, dict[str, Any]] = {}
        self._adaptive_last_update_frame: dict[int, int] = {}
        self._adaptive_peer_unavailable: dict[int, str] = {}
        self.wm_calibration_enable = bool(wm_calibration_enable)
        # Last decision type the observer emitted at the current PBT tick.
        # 0=none, 1=self, 2=cross, 3=skipped. Used both for TB log and for
        # write_training_curve.
        self._last_restart_type: int = 0
        self.curr_eligible_as_source: bool = True
        self.curr_eligible_for_elimination: bool = True
        # Reason string carried into training_curve so it is auditable why a
        # given worker is/is not eligible at a given tick.
        self.curr_eligible_source_reason: str = "ok"
        # M5: derived state for training curve logging
        self._last_window_size: int = 0
        self._curriculum_update_count: int = 0
        self._metrics_dir = self.workspace / "metrics"
        # EIR-MPC Phase D: PBT elite broadcast counters.
        self._eir_broadcast_count: int = 0
        self._eir_bank_merge_count: int = 0
        self._eir_last_source_broadcast_frame: int = -1
        # Patch 5: in-memory trackers for peak archives. Loaded from disk
        # lazily in `after_init` so PBT workers that come up after a crash
        # respect existing peak files.
        self._best_r_meta_state: dict[str, Any] | None = None
        self._best_nsucc_state: dict[str, Any] | None = None
        self._best_stable_state: dict[str, Any] | None = None

    def _uses_legacy_source_selection(self) -> bool:
        """Legacy Patch G cascade when mastery/stable-source mode is off."""
        return (
            self.curriculum_mode == "legacy"
            and not self.source_require_stable_mean
        )

    def _best_nsucc_recovery_enabled(self) -> bool:
        """best_nsucc recovery only in legacy mode or when explicitly allowed."""
        if self._uses_legacy_source_selection():
            return self.enable_best_nsucc_recovery
        return self.allow_best_nsucc_archive_recovery

    def _policy_window(self, policy_idx: int) -> RollingNsuccWindow:
        if policy_idx not in self._policy_nsucc_windows:
            self._policy_nsucc_windows[policy_idx] = RollingNsuccWindow()
        return self._policy_nsucc_windows[policy_idx]

    def _load_latest_policy_meta(self) -> dict[str, Any] | None:
        if not self.curr_policy_workspace.exists():
            return None
        for candidate in sorted(self.curr_policy_workspace.glob("*.json"), reverse=True):
            if candidate.name.endswith(".tmp") or candidate.name == "LATEST.commit":
                continue
            try:
                data = json.loads(candidate.read_text())
            except Exception:
                continue
            if isinstance(data, dict) and data.get("lineage_id"):
                return data
        return None

    def _read_prev_commit_id(self) -> str | None:
        latest = self.curr_policy_workspace / "LATEST.commit"
        if not latest.is_file():
            return None
        try:
            data = json.loads(latest.read_text())
        except Exception:
            return None
        if isinstance(data, dict):
            cid = data.get("commit_id")
            return str(cid) if cid else None
        return None

    @staticmethod
    def _window_sample_from_state(
        state: dict[str, Any], *, env_frames: int, pbt_iteration: int
    ) -> WindowSample:
        cur = state.get("curriculum") or {}
        lineage_id = state.get("lineage_id")
        if lineage_id is not None:
            lineage_id = str(lineage_id)
        return WindowSample(
            env_frames=int(state.get("frame", env_frames)),
            nsucc_mean=float(
                state.get(
                    "nsucc_mean",
                    cur.get("mean_prev_episode_successes", 0.0),
                )
                or 0.0
            ),
            nsucc_max=float(
                state.get(
                    "nsucc_max",
                    cur.get("max_prev_episode_successes", 0.0),
                )
                or 0.0
            ),
            success_tolerance=float(cur.get("success_tolerance", 0.075) or 0.075),
            pbt_iteration=int(state.get("iteration", pbt_iteration)),
            lineage_id=lineage_id,
        )

    def _adaptive_precision_path(self, policy_idx: int) -> "Path":
        d = self.workspace / "adaptive_precision"
        d.mkdir(parents=True, exist_ok=True)
        return d / f"{int(policy_idx):03d}.json"

    def _adaptive_precision_required(self) -> bool:
        return str(self.wm_strategy_profile_name or "") in (
            "cser_dexpbt_progress_refine",
            "cser_dexpbt_readiness_adaptive",
        )

    def _adaptive_precision_state(
        self,
        policy_idx: int,
        *,
        strict_local: bool | None = None,
    ):
        """Return (cfg, probation, tracker) or None when adaptive is inert.

        Local formal-profile reads fail closed (``strict_local=True``). Peer
        reads used by rescue selection stay soft (``strict_local=False``) so a
        corrupt peer sidecar cannot abort the local worker.
        """
        if not self.adaptive_precision_enable:
            return None
        if strict_local is None:
            strict_local = (
                int(policy_idx) == int(self.policy_idx)
                and self._adaptive_precision_required()
            )
        try:
            self._adaptive_precision_restore(policy_idx, strict=bool(strict_local))
        except Exception as exc:
            if strict_local:
                raise RuntimeError(
                    f"adaptive precision restore failed for formal profile "
                    f"(policy={policy_idx}): {exc}"
                ) from exc
            return None
        cfg = self._adaptive_precision_cfg
        if cfg is None:
            if strict_local:
                raise RuntimeError(
                    "adaptive precision required but AdaptivePrecisionConfig is missing"
                )
            return None
        probation = self._adaptive_probation.get(policy_idx)
        tracker = self._adaptive_retention.get(policy_idx)
        if probation is None or tracker is None:
            if strict_local:
                raise RuntimeError(
                    f"adaptive precision state missing for policy {policy_idx}"
                )
            return None
        return cfg, probation, tracker

    def _adaptive_read_sidecar_raw(self, policy_idx: int) -> dict[str, Any] | None:
        """Best-effort raw sidecar JSON read (no cache mutation, no raise)."""
        path = self._adaptive_precision_path(policy_idx)
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text())
            return data if isinstance(data, dict) else None
        except Exception:
            return None

    def _adaptive_precision_restore(self, policy_idx: int, *, strict: bool = True) -> None:
        """Restore probation / retention / rescue state from the sidecar JSON."""
        if policy_idx in self._adaptive_probation:
            return
        from dexpbt_isaaclab.pbt.adaptive_precision_consolidation import (
            ADAPTIVE_SIDECAR_SCHEMA_VERSION,
            PrecisionRetentionTracker,
            ProbationState,
        )

        cfg = self._adaptive_precision_cfg
        path = self._adaptive_precision_path(policy_idx)
        data: dict[str, Any] = {}
        if not path.exists():
            if not strict:
                self._adaptive_peer_unavailable[policy_idx] = (
                    "source_adaptive_sidecar_missing"
                )
                raise RuntimeError(
                    f"adaptive sidecar missing for policy {policy_idx}"
                )
            # Local/strict cold start: materialise empty in-memory state.
        else:
            last_err: Exception | None = None
            for _attempt in range(3):
                try:
                    data = json.loads(path.read_text()) or {}
                    last_err = None
                    break
                except Exception as exc:
                    last_err = exc
                    data = {}
            if last_err is not None:
                if strict:
                    raise RuntimeError(
                        f"adaptive sidecar parse failed for policy {policy_idx}: {last_err}"
                    ) from last_err
                self._adaptive_peer_unavailable[policy_idx] = (
                    "source_adaptive_sidecar_parse_error"
                )
                raise RuntimeError(
                    f"adaptive sidecar parse failed for policy {policy_idx}: {last_err}"
                ) from last_err
        self._adaptive_peer_unavailable.pop(policy_idx, None)
        # Schema migrate: missing schema_version treated as v1 in memory.
        if "schema_version" not in data:
            data = dict(data)
            data["schema_version"] = 1
        if data.get("probation"):
            self._adaptive_probation[policy_idx] = ProbationState.from_json(data["probation"])
        else:
            self._adaptive_probation[policy_idx] = ProbationState()
        if data.get("retention"):
            self._adaptive_retention[policy_idx] = PrecisionRetentionTracker.from_json(
                data["retention"]
            )
        else:
            quant = float(cfg.tolerance_log_quant) if cfg is not None else 0.05
            self._adaptive_retention[policy_idx] = PrecisionRetentionTracker(quant=quant)
        self._adaptive_degradation_streak[policy_idx] = int(data.get("degradation_streak", 0))
        self._adaptive_last_rescue_frame[policy_idx] = int(
            data.get("last_rescue_frame", -(10**18))
        )
        self._adaptive_rescue_executed[policy_idx] = int(data.get("rescue_executed", 0))
        self._adaptive_rescue_executed_for_lineage[policy_idx] = int(
            data.get("rescue_executed_for_lineage", data.get("rescue_executed", 0))
        )
        self._adaptive_lineage_id[policy_idx] = str(data.get("adaptive_lineage_id", "") or "")
        self._adaptive_rescue_attempt_count[policy_idx] = int(
            data.get("rescue_attempt_count", 0)
        )
        self._adaptive_rescue_selected_count[policy_idx] = int(
            data.get("rescue_selected_count", 0)
        )
        self._adaptive_rescue_blocked_count[policy_idx] = int(
            data.get("rescue_blocked_count", 0)
        )
        self._adaptive_rescue_last_reason[policy_idx] = str(
            data.get("rescue_last_reason", "") or ""
        )
        self._adaptive_rescue_last_source[policy_idx] = int(
            data.get("rescue_last_source", -1)
        )
        self._adaptive_rescue_last_source_rank[policy_idx] = int(
            data.get("rescue_last_source_rank", -1)
        )
        self._adaptive_rescue_last_advantage_sigma[policy_idx] = float(
            data.get("rescue_last_advantage_sigma", 0.0) or 0.0
        )
        self._adaptive_prev_tol = getattr(self, "_adaptive_prev_tol", {})
        self._adaptive_prev_tol[policy_idx] = data.get("prev_tolerance")
        self._adaptive_eps_init = getattr(self, "_adaptive_eps_init", {})
        self._adaptive_eps_init[policy_idx] = float(data.get("eps_init", 0.0) or 0.0)
        self._adaptive_last_update_frame[policy_idx] = int(
            data.get("last_update_frame", -(10**18))
        )
        self._adaptive_last_checkpoint_frame[policy_idx] = int(
            data.get("last_checkpoint_frame", -1)
        )
        self._adaptive_last_checkpoint_path[policy_idx] = str(
            data.get("last_checkpoint_path", "") or ""
        )
        self._adaptive_freeze_only_streak[policy_idx] = int(
            data.get("freeze_only_streak", 0) or 0
        )
        self._adaptive_sidecar_meta[policy_idx] = {
            "schema_version": int(
                data.get("schema_version", ADAPTIVE_SIDECAR_SCHEMA_VERSION)
            ),
            "last_update_frame": int(data.get("last_update_frame", -(10**18))),
            "last_update_timestamp": float(data.get("last_update_timestamp", 0.0) or 0.0),
            "last_update_lineage_id": str(data.get("last_update_lineage_id", "") or ""),
            "last_checkpoint_frame": int(data.get("last_checkpoint_frame", -1)),
            "last_checkpoint_path": str(data.get("last_checkpoint_path", "") or ""),
            "freeze_only_streak": int(data.get("freeze_only_streak", 0) or 0),
        }

    def _adaptive_precision_persist(self, policy_idx: int) -> None:
        import time as _time

        from dexpbt_isaaclab.pbt.adaptive_precision_consolidation import (
            ADAPTIVE_SIDECAR_SCHEMA_VERSION,
        )

        path = self._adaptive_precision_path(policy_idx)
        lineage = str(self._adaptive_lineage_id.get(policy_idx, "") or "")
        payload = {
            "policy_idx": int(policy_idx),
            "schema_version": int(ADAPTIVE_SIDECAR_SCHEMA_VERSION),
            "probation": self._adaptive_probation[policy_idx].to_json(),
            "retention": self._adaptive_retention[policy_idx].to_json(),
            "degradation_streak": int(self._adaptive_degradation_streak.get(policy_idx, 0)),
            "last_rescue_frame": int(
                self._adaptive_last_rescue_frame.get(policy_idx, -(10**18))
            ),
            "rescue_executed": int(self._adaptive_rescue_executed.get(policy_idx, 0)),
            "rescue_executed_for_lineage": int(
                self._adaptive_rescue_executed_for_lineage.get(policy_idx, 0)
            ),
            "adaptive_lineage_id": lineage,
            "rescue_attempt_count": int(
                self._adaptive_rescue_attempt_count.get(policy_idx, 0)
            ),
            "rescue_selected_count": int(
                self._adaptive_rescue_selected_count.get(policy_idx, 0)
            ),
            "rescue_blocked_count": int(
                self._adaptive_rescue_blocked_count.get(policy_idx, 0)
            ),
            "rescue_last_reason": str(
                self._adaptive_rescue_last_reason.get(policy_idx, "")
            ),
            "rescue_last_source": int(
                self._adaptive_rescue_last_source.get(policy_idx, -1)
            ),
            "rescue_last_source_rank": int(
                self._adaptive_rescue_last_source_rank.get(policy_idx, -1)
            ),
            "rescue_last_advantage_sigma": float(
                self._adaptive_rescue_last_advantage_sigma.get(policy_idx, 0.0)
            ),
            "prev_tolerance": getattr(self, "_adaptive_prev_tol", {}).get(policy_idx),
            "eps_init": float(getattr(self, "_adaptive_eps_init", {}).get(policy_idx, 0.0)),
            "last_update_frame": int(
                self._adaptive_last_update_frame.get(policy_idx, -(10**18))
            ),
            "last_update_timestamp": float(_time.time()),
            "last_update_lineage_id": lineage,
            "last_checkpoint_frame": int(
                self._adaptive_last_checkpoint_frame.get(policy_idx, -1)
            ),
            "last_checkpoint_path": str(
                self._adaptive_last_checkpoint_path.get(policy_idx, "")
            ),
            "freeze_only_streak": int(
                self._adaptive_freeze_only_streak.get(policy_idx, 0)
            ),
        }
        self._adaptive_sidecar_meta[policy_idx] = {
            "schema_version": int(ADAPTIVE_SIDECAR_SCHEMA_VERSION),
            "last_update_frame": int(payload["last_update_frame"]),
            "last_update_timestamp": float(payload["last_update_timestamp"]),
            "last_update_lineage_id": lineage,
            "last_checkpoint_frame": int(payload["last_checkpoint_frame"]),
            "last_checkpoint_path": str(payload["last_checkpoint_path"]),
            "freeze_only_streak": int(payload["freeze_only_streak"]),
        }
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload))
        os.replace(str(tmp), str(path))

    def _adaptive_reset_lineage_quota(self, policy_idx: int, lineage_id: str) -> None:
        """Reset per-lineage adaptive rescue quota and degradation streak."""
        stored = str(self._adaptive_lineage_id.get(policy_idx, "") or "")
        current = str(lineage_id or "")
        if current == stored:
            return
        self._adaptive_lineage_id[policy_idx] = current
        self._adaptive_degradation_streak[policy_idx] = 0
        self._adaptive_rescue_executed_for_lineage[policy_idx] = 0
        self._adaptive_last_rescue_frame[policy_idx] = -(10**18)

    def _adaptive_precision_update(
        self, *, sample: Any, state: dict[str, Any] | None, env_frames: int
    ) -> None:
        """Advance the adaptive precision controller for the current policy.

        Maintains per-precision-band retention, generalized post-shrink
        probation, and the degradation streak used by rescue.  State survives
        restart via a sidecar JSON.  Emits ``adaptive_consolidation/*`` metrics.
        """
        from dexpbt_isaaclab.pbt.adaptive_precision_consolidation import (
            curriculum_progress as _curr_prog,
        )

        cfg = self._adaptive_precision_cfg
        if cfg is None:
            return
        pidx = int(self.policy_idx)
        self._adaptive_precision_restore(pidx, strict=True)

        cur_tol = float(getattr(sample, "success_tolerance", 0.075) or 0.075)
        nsucc_mean = float(getattr(sample, "nsucc_mean", 0.0) or 0.0)
        lineage_id = str(getattr(sample, "lineage_id", "") or "")
        self._adaptive_reset_lineage_quota(pidx, lineage_id)

        # Target tolerance from the state's curriculum block; fall back to the
        # configured bridge target (both default to the env target tolerance).
        target_tol = float(self.bridge_curriculum_target_tol)
        if isinstance(state, dict):
            curr = state.get("curriculum") or {}
            try:
                target_tol = float(
                    curr.get("target_tolerance", curr.get("target_success_tolerance", target_tol))
                )
            except Exception:
                pass
        # Initial (loosest) tolerance = monotonic max observed.
        eps_init_map = getattr(self, "_adaptive_eps_init", {})
        eps_init = max(float(eps_init_map.get(pidx, 0.0) or 0.0), cur_tol)
        eps_init_map[pidx] = eps_init
        self._adaptive_eps_init = eps_init_map

        c = _curr_prog(eps_init, target_tol, cur_tol, eps=float(cfg.eps))
        self._adaptive_curriculum_progress = float(c)

        tracker = self._adaptive_retention[pidx]
        tracker.update(
            cur_tol,
            nsucc_mean,
            lineage_id=lineage_id,
            reset_on_lineage_change=True,
            min_retention_for_stable=float(cfg.rescue_source_min_retention),
        )
        snap = tracker.snapshot(cur_tol, min_samples=1)
        retention = float(snap["retention"])

        # Detect a stricter shrink -> arm probation for the new precision band.
        prev_map = getattr(self, "_adaptive_prev_tol", {})
        prev_tol = prev_map.get(pidx)
        probation = self._adaptive_probation[pidx]
        if prev_tol is not None and cur_tol < float(prev_tol) - 1e-9:
            probation.arm(
                tolerance=cur_tol,
                frame=int(env_frames),
                nsucc_mean=nsucc_mean,
                previous_tolerance=float(prev_tol),
            )
        prev_map[pidx] = cur_tol
        self._adaptive_prev_tol = prev_map

        probation.update(
            nsucc_mean=nsucc_mean,
            same_level_retention=retention,
            frame=int(env_frames),
            cfg=cfg,
        )
        from dexpbt_isaaclab.pbt.adaptive_precision_consolidation import (
            resolve_stage as _resolve_stage,
        )

        stage_name = _resolve_stage(float(c), cfg)
        if bool(getattr(cfg, "probation_stage_aware_enable", False)):
            exit_kind, exit_reason = probation.evaluate_exit(
                frame=int(env_frames), stage=str(stage_name), cfg=cfg
            )
            if exit_kind in ("early_exit", "normal_exit"):
                probation.active = False
            elif exit_kind == "max_frames_timeout":
                # Arm the same rollback/freeze path as hard-collapse.
                self._adaptive_probation_timeout[pidx] = True
                self._adaptive_probation_timeout_reason[pidx] = str(exit_reason)
        else:
            probation.maybe_exit(frame=int(env_frames), cfg=cfg)

        # Degradation streak feeds retention-conditioned rescue.
        if retention < float(cfg.rescue_retention_threshold):
            self._adaptive_degradation_streak[pidx] = (
                int(self._adaptive_degradation_streak.get(pidx, 0)) + 1
            )
        else:
            self._adaptive_degradation_streak[pidx] = 0

        self._adaptive_last_update_frame[pidx] = int(env_frames)
        self._adaptive_precision_persist(pidx)

        if self.writer is not None:
            stage = resolve_pbt_stage(
                int(env_frames), self.pbt_consolidation_cfg, curriculum_progress=c
            )
            can_shrink, shrink_block = probation.can_shrink(frame=int(env_frames), cfg=cfg)
            for name, val in (
                ("adaptive_consolidation/curriculum_progress", float(c)),
                ("adaptive_consolidation/stage", float(stage_id(stage))),
                ("adaptive_consolidation/local_retention", float(retention)),
                ("adaptive_consolidation/probation_active", 1.0 if probation.active else 0.0),
                (
                    "adaptive_consolidation/probation_frames",
                    float(max(0, int(env_frames) - int(probation.entry_frame)))
                    if probation.active
                    else 0.0,
                ),
                ("adaptive_consolidation/shrink_blocked", 0.0 if can_shrink else 1.0),
                ("adaptive_consolidation/probation_can_shrink", 1.0 if can_shrink else 0.0),
                (
                    "adaptive_consolidation/hard_collapse_streak",
                    float(probation.hard_collapse_streak),
                ),
                (
                    "adaptive_consolidation/degradation_streak",
                    float(self._adaptive_degradation_streak.get(pidx, 0)),
                ),
                (
                    "adaptive_consolidation/rescue_quota_used",
                    float(self._adaptive_rescue_executed_for_lineage.get(pidx, 0)),
                ),
                (
                    "adaptive_consolidation/distinct_lineage_count",
                    float(self._last_adaptive_distinct_lineage_count),
                ),
                (
                    "adaptive_consolidation/min_distinct_lineages",
                    float(self._last_adaptive_min_distinct_lineages),
                ),
                (
                    "adaptive_consolidation/replacement_budget",
                    float(self._last_adaptive_replacement_budget),
                ),
                (
                    "adaptive_consolidation/replacement_budget_used",
                    float(self._last_adaptive_replacement_budget_used),
                ),
                (
                    "adaptive_consolidation/diversity_blocked",
                    1.0 if self._last_adaptive_diversity_blocked else 0.0,
                ),
            ):
                try:
                    self.writer.add_scalar(name, val, int(env_frames))
                except Exception:
                    pass

    def _adaptive_degradation_rescue_ready(
        self,
        *,
        policy_idx: int,
        tolerance: float,
        nsucc_mean: float,
        env_frames: int,
        lineage_id: str,
        hard_collapse_now: bool = False,
    ) -> tuple[bool, str]:
        """Retention-conditioned adaptive rescue trigger (no legacy late rescue).

        When ``hard_collapse_now`` is True, hard collapse independently arms
        adaptive rescue without retention warmup / degradation patience, while
        still respecting lineage quota and cooldown.
        """
        from dexpbt_isaaclab.pbt.adaptive_precision_consolidation import (
            evaluate_degradation_rescue,
        )

        state = self._adaptive_precision_state(policy_idx)
        if state is None:
            return False, "disabled"
        cfg, _probation, tracker = state
        if not bool(cfg.rescue_enable):
            return False, "disabled"
        self._adaptive_reset_lineage_quota(policy_idx, lineage_id)
        last_frame = int(self._adaptive_last_rescue_frame.get(policy_idx, -(10**18)))
        frames_since = int(env_frames) - last_frame
        executed = int(self._adaptive_rescue_executed_for_lineage.get(policy_idx, 0))
        if int(executed) >= int(cfg.rescue_max_per_lineage):
            reason = (
                "hard_collapse_quota_exhausted"
                if hard_collapse_now
                else "quota_exhausted"
            )
            self._adaptive_rescue_last_reason[policy_idx] = str(reason)
            return False, str(reason)
        if int(frames_since) < int(cfg.rescue_cooldown_frames):
            reason = (
                "hard_collapse_cooldown" if hard_collapse_now else "cooldown"
            )
            self._adaptive_rescue_last_reason[policy_idx] = str(reason)
            return False, str(reason)
        if hard_collapse_now:
            reason = "hard_collapse_ready"
            self._adaptive_rescue_last_reason[policy_idx] = str(reason)
            return True, str(reason)
        snap = tracker.snapshot(float(tolerance), min_samples=1)
        ready, reason = evaluate_degradation_rescue(
            tolerance=float(tolerance),
            nsucc_mean=float(nsucc_mean),
            same_level_retention=float(snap["retention"]),
            same_level_valid=bool(snap["valid"]),
            degradation_streak=int(self._adaptive_degradation_streak.get(policy_idx, 0)),
            executed_this_lineage=int(executed),
            frames_since_last_rescue=int(frames_since),
            cfg=cfg,
        )
        if ready:
            reason = "retention_degradation_ready"
        elif str(reason) in {
            "retention_invalid",
            "retention_nonfinite",
            "retention_ok",
            "streak_below_patience",
        }:
            # Keep detailed diagnostic reasons for retention path; also expose
            # a stable not-ready token for control-plane consumers.
            reason = str(reason)
        self._adaptive_rescue_last_reason[policy_idx] = str(reason)
        return bool(ready), str(reason)

    def _select_adaptive_degradation_rescue_source(
        self,
        *,
        sorted_policies: list[int],
        population: dict[int, dict[str, Any] | None],
        env_frames: int,
        target_tolerance: float,
        target_nsucc_mean: float,
    ) -> dict[str, Any]:
        """Live-only adaptive rescue source selection."""
        from dexpbt_isaaclab.pbt.adaptive_precision_consolidation import (
            evaluate_sidecar_freshness,
            log_tolerance_gap,
            rescue_source_top_k,
            select_rescue_source,
            validate_source_retention_snapshot,
        )

        empty = {
            "action": "skip",
            "tier": "skipped",
            "checkpoint_type": "none",
            "source_policy": -1,
            "source_state": None,
            "strict_candidates": [],
            "relaxed_candidates": [],
            "strict_filter_reasons": {},
            "relaxed_filter_reasons": {},
            "archive_reason": "no_compatible_source",
            "adaptive_rescue_reason": "no_compatible_source",
            "adaptive_source_rank": -1,
            "adaptive_source_advantage_sigma": 0.0,
        }
        state = self._adaptive_precision_state(self.policy_idx, strict_local=True)
        if state is None:
            empty["adaptive_rescue_reason"] = "disabled"
            empty["archive_reason"] = "disabled"
            return empty
        cfg, _probation, _tracker = state
        candidates: list[dict[str, Any]] = []
        pop_means: list[float] = []
        filter_reasons: dict[int, str] = {}
        pop_lineage = str(getattr(self, "_lineage_id", "") or "")
        for idx in sorted_policies:
            if int(idx) == int(self.policy_idx):
                continue
            st = population.get(idx)
            if not isinstance(st, dict):
                filter_reasons[int(idx)] = "missing_state"
                continue
            ckpt = st.get("checkpoint")
            if not ckpt or not Path(str(ckpt)).is_file():
                filter_reasons[int(idx)] = "missing_checkpoint"
                continue
            curr = st.get("curriculum") or {}
            try:
                s_mean = float(st.get("nsucc_mean", float("nan")))
                s_tol = float(curr.get("success_tolerance", float("nan")))
                s_rank = float(st.get("pbt_ranking_score", st.get("objective", float("nan"))))
            except Exception:
                filter_reasons[int(idx)] = "nonfinite"
                continue
            if not all(math.isfinite(x) for x in (s_mean, s_tol, s_rank)):
                filter_reasons[int(idx)] = "nonfinite"
                continue
            pop_means.append(s_mean)
            src_state = self._adaptive_precision_state(int(idx), strict_local=False)
            if src_state is None:
                reason = self._adaptive_peer_unavailable.get(
                    int(idx), "source_adaptive_sidecar_missing"
                )
                filter_reasons[int(idx)] = str(reason)
                self._adaptive_peer_reject_counts[int(idx)] = (
                    int(self._adaptive_peer_reject_counts.get(int(idx), 0)) + 1
                )
                continue
            _c, _p, src_tracker = src_state
            snap = src_tracker.snapshot(
                s_tol,
                min_samples=int(cfg.rescue_source_min_samples),
                min_stable_ticks=int(cfg.rescue_source_min_stable_ticks),
            )
            ret_ok, ret_reason = validate_source_retention_snapshot(snap, cfg=cfg)
            if not ret_ok:
                filter_reasons[int(idx)] = str(ret_reason)
                self._adaptive_peer_reject_counts[int(idx)] = (
                    int(self._adaptive_peer_reject_counts.get(int(idx), 0)) + 1
                )
                continue
            raw_side = self._adaptive_read_sidecar_raw(int(idx))
            if raw_side is None:
                filter_reasons[int(idx)] = "source_adaptive_sidecar_missing"
                self._adaptive_peer_reject_counts[int(idx)] = (
                    int(self._adaptive_peer_reject_counts.get(int(idx), 0)) + 1
                )
                continue
            ckpt_frame = None
            try:
                if "frame" in st:
                    ckpt_frame = int(st.get("frame"))
                elif "env_frames" in st:
                    ckpt_frame = int(st.get("env_frames"))
            except Exception:
                ckpt_frame = None
            fresh_ok, fresh_reason = evaluate_sidecar_freshness(
                sidecar=raw_side,
                population_frame=int(st.get("frame", env_frames) or env_frames),
                population_lineage_id=str(
                    st.get("lineage_id", pop_lineage) or pop_lineage
                ),
                checkpoint_frame=ckpt_frame,
                interval_steps=int(self.interval_steps),
                cfg=cfg,
                require_freshness_fields=True,
            )
            if not fresh_ok:
                filter_reasons[int(idx)] = str(fresh_reason)
                self._adaptive_peer_reject_counts[int(idx)] = (
                    int(self._adaptive_peer_reject_counts.get(int(idx), 0)) + 1
                )
                continue
            s_ret = float(snap["retention"])
            ctrl_path = self._curriculum_control_dir / f"{int(idx):03d}.json"
            has_control = ctrl_path.is_file()
            collapsed = False
            if has_control:
                try:
                    control = read_curriculum_control(ctrl_path)
                except Exception:
                    control = None
                if control is None:
                    if cfg.rescue_require_source_control:
                        filter_reasons[int(idx)] = "source_control_parse_error"
                        continue
                    has_control = False
                else:
                    collapsed, _reason = control_has_collapse_signal(control)
            elif cfg.rescue_require_source_control:
                filter_reasons[int(idx)] = "source_control_missing"
                continue
            candidates.append(
                {
                    "idx": int(idx),
                    "nsucc_mean": s_mean,
                    "retention": s_ret,
                    "retention_valid": True,
                    "tolerance": s_tol,
                    "ranking_score": s_rank,
                    "has_control": bool(has_control),
                    "collapsed": bool(collapsed),
                    "source_age_frames": max(
                        0,
                        int(env_frames)
                        - int(raw_side.get("last_update_frame", env_frames) or env_frames),
                    ),
                }
            )
        if math.isfinite(float(target_nsucc_mean)):
            pop_means.append(float(target_nsucc_mean))
        top_k = rescue_source_top_k(
            len(sorted_policies), float(cfg.rescue_source_top_fraction)
        )
        allowed = {int(x) for x in sorted_policies[:top_k]}
        decision = select_rescue_source(
            target_tolerance=float(target_tolerance),
            target_nsucc_mean=float(target_nsucc_mean),
            candidates=candidates,
            population_nsucc_means=pop_means,
            cfg=cfg,
            allowed_source_ids=allowed,
            sorted_population=list(sorted_policies),
        )
        if not decision.eligible:
            empty["adaptive_rescue_reason"] = str(decision.reason)
            empty["archive_reason"] = str(decision.reason)
            empty["relaxed_filter_reasons"] = {
                str(k): v for k, v in filter_reasons.items()
            }
            return empty
        src_idx = int(decision.source_idx)
        src_state = population.get(src_idx)
        assert src_state is not None
        src_tol = float(
            (src_state.get("curriculum") or {}).get("success_tolerance", float("nan"))
        )
        return {
            "action": "copy",
            "tier": "adaptive_rescue_live",
            "checkpoint_type": "live",
            "source_policy": src_idx,
            "source_state": src_state,
            "strict_candidates": [src_idx],
            "relaxed_candidates": [c["idx"] for c in candidates],
            "strict_filter_reasons": {},
            "relaxed_filter_reasons": {
                str(k): v for k, v in filter_reasons.items()
            },
            "archive_reason": str(decision.reason),
            "adaptive_rescue_reason": "selected",
            "adaptive_source_rank": int(decision.source_rank),
            "adaptive_source_advantage_sigma": float(decision.source_advantage_sigma),
            "adaptive_log_tolerance_gap": float(
                log_tolerance_gap(src_tol, target_tolerance, eps=cfg.eps)
                if math.isfinite(src_tol)
                else float("nan")
            ),
        }

    def _select_adaptive_archive_rescue_source(
        self,
        *,
        env_frames: int,
        target_tolerance: float,
        target_nsucc_mean: float,
        population_nsucc_means: list[float] | None = None,
    ) -> dict[str, Any]:
        """Try best_stable / floor_recovery / best_nsucc archives for adaptive rescue."""
        from dexpbt_isaaclab.pbt.adaptive_precision_consolidation import (
            log_tolerance_gap,
        )

        empty = {
            "action": "skip",
            "tier": "skipped",
            "checkpoint_type": "none",
            "source_policy": -1,
            "source_state": None,
            "strict_candidates": [],
            "relaxed_candidates": [],
            "strict_filter_reasons": {},
            "relaxed_filter_reasons": {},
            "archive_reason": "no_compatible_archive",
            "adaptive_rescue_reason": "no_compatible_archive",
            "adaptive_rescue_source_kind": "none",
            "adaptive_source_rank": -1,
            "adaptive_source_advantage_sigma": 0.0,
        }
        state = self._adaptive_precision_state(self.policy_idx, strict_local=True)
        if state is None:
            empty["adaptive_rescue_reason"] = "disabled"
            return empty
        cfg, _probation, _tracker = state
        if not bool(cfg.rescue_archive_enable):
            empty["adaptive_rescue_reason"] = "archive_disabled"
            return empty

        means = [
            float(x)
            for x in (population_nsucc_means or [])
            if isinstance(x, (int, float)) and math.isfinite(float(x))
        ]
        min_pop_samples = int(cfg.rescue_archive_min_population_samples)
        if len(means) < min_pop_samples:
            empty["archive_reason"] = "archive_population_stats_insufficient"
            empty["adaptive_rescue_reason"] = "archive_population_stats_insufficient"
            empty["adaptive_population_nsucc_sample_count"] = int(len(means))
            empty["adaptive_population_nsucc_sigma"] = 0.0
            empty["adaptive_population_nsucc_effective_sigma"] = 0.0
            empty["adaptive_source_absolute_advantage"] = 0.0
            empty["adaptive_source_advantage_sigma"] = 0.0
            return empty
        if len(means) >= 2:
            mu = sum(means) / float(len(means))
            var = sum((m - mu) ** 2 for m in means) / float(len(means) - 1)
            pop_sigma = math.sqrt(max(0.0, var))
        else:
            pop_sigma = 0.0
        sigma_floor = float(cfg.rescue_archive_advantage_sigma_floor)
        effective_sigma = max(float(pop_sigma), float(sigma_floor))

        tiers = (
            (
                "adaptive_rescue_best_stable_archive",
                "best_stable_archive",
                self._select_best_stable_recovery_source,
            ),
            (
                "adaptive_rescue_floor_recovery_archive",
                "best_floor_recovery_archive",
                self._select_best_floor_recovery_recovery_source,
            ),
            (
                "adaptive_rescue_best_nsucc_archive",
                "archive",
                self._select_best_nsucc_recovery_source,
            ),
        )
        reject_reasons: list[str] = []
        collapse_keys = (
            "collapse_rollback_active",
            "hard_collapse_triggered",
            "collapsed",
            "adaptive_probation_hard_collapse",
        )
        quality_keys = (
            "pbt_ranking_score",
            "ranking_score",
            "archive_quality_score",
            "r_meta_raw",
            "objective",
        )
        for tier_label, ckpt_type, selector in tiers:
            synth, reason = selector(int(env_frames))
            if synth is None:
                reject_reasons.append(f"{tier_label}:{reason}")
                continue
            ckpt = synth.get("checkpoint")
            if not ckpt or not Path(str(ckpt)).is_file():
                reject_reasons.append(f"{tier_label}:checkpoint_missing")
                continue
            try:
                s_tol = float(
                    (synth.get("curriculum") or {}).get(
                        "success_tolerance",
                        synth.get("success_tolerance", float("nan")),
                    )
                )
                s_mean = float(synth.get("nsucc_mean", float("nan")))
                s_max = float(
                    synth.get(
                        "nsucc_max",
                        (synth.get("curriculum") or {}).get(
                            "max_prev_episode_successes", float("nan")
                        ),
                    )
                )
            except Exception:
                reject_reasons.append(f"{tier_label}:nonfinite")
                continue
            if not all(math.isfinite(x) for x in (s_tol, s_mean, s_max)):
                reject_reasons.append(f"{tier_label}:nonfinite")
                continue
            if s_tol <= 0.0:
                reject_reasons.append(f"{tier_label}:tolerance_incompatible")
                continue
            if s_mean < float(cfg.rescue_archive_min_nsucc_mean):
                reject_reasons.append(f"{tier_label}:nsucc_below_min")
                continue
            if s_max < float(cfg.rescue_archive_min_nsucc_max):
                reject_reasons.append(f"{tier_label}:nsucc_max_below_min")
                continue
            quality = float("nan")
            for qk in quality_keys:
                if qk in synth:
                    try:
                        quality = float(synth.get(qk))
                    except Exception:
                        quality = float("nan")
                    break
                curr = synth.get("curriculum") or {}
                if qk in curr:
                    try:
                        quality = float(curr.get(qk))
                    except Exception:
                        quality = float("nan")
                    break
            if bool(cfg.rescue_archive_require_quality_score):
                if not math.isfinite(quality):
                    reject_reasons.append(f"{tier_label}:quality_score_missing")
                    continue
            gap = log_tolerance_gap(s_tol, target_tolerance, eps=cfg.eps)
            if gap > float(cfg.rescue_max_log_tol_gap) + 1e-12:
                reject_reasons.append(f"{tier_label}:log_gap")
                continue
            ret = float("nan")
            ret_valid = False
            for key in (
                "same_tol_retention_at_save",
                "source_retention_at_save",
                "same_tol_retention",
                "retention",
            ):
                if key in synth:
                    try:
                        ret = float(synth.get(key))
                    except Exception:
                        ret = float("nan")
                    break
                curr = synth.get("curriculum") or {}
                if key in curr:
                    try:
                        ret = float(curr.get(key))
                    except Exception:
                        ret = float("nan")
                    break
            if "same_tol_retention_valid_at_save" in synth:
                ret_valid = bool(synth.get("same_tol_retention_valid_at_save"))
            elif math.isfinite(ret):
                ret_valid = True
            if bool(cfg.rescue_archive_require_retention):
                if (not ret_valid) or (not math.isfinite(ret)):
                    reject_reasons.append(f"{tier_label}:retention_missing")
                    continue
                if ret < float(cfg.rescue_source_min_retention):
                    reject_reasons.append(f"{tier_label}:retention_below_min")
                    continue
            age_frame = None
            age_parse_failed = False
            for key in ("env_frames", "frame", "population_frame", "saved_frame"):
                if key in synth:
                    try:
                        age_frame = int(synth.get(key))
                    except Exception:
                        age_frame = None
                        age_parse_failed = True
                    break
            age_known = False
            age_frames = -1
            if age_parse_failed:
                reject_reasons.append(f"{tier_label}:archive_age_nonfinite")
                continue
            if age_frame is None:
                if bool(cfg.rescue_archive_require_age):
                    reject_reasons.append(f"{tier_label}:archive_age_missing")
                    continue
                # require_age=False: skip age gate; do not invent age=0.
                age_known = False
                age_frames = -1
            else:
                age = int(env_frames) - int(age_frame)
                if age < 0:
                    reject_reasons.append(f"{tier_label}:archive_age_from_future")
                    continue
                if age > int(cfg.rescue_archive_max_age_frames):
                    reject_reasons.append(f"{tier_label}:archive_too_old")
                    continue
                age_known = True
                age_frames = int(age)
            has_collapse_meta = any(k in synth for k in collapse_keys)
            if bool(cfg.rescue_archive_require_collapse_metadata) and not has_collapse_meta:
                reject_reasons.append(f"{tier_label}:collapse_metadata_missing")
                continue
            collapsed = bool(
                synth.get("collapse_rollback_active")
                or synth.get("hard_collapse_triggered")
                or synth.get("collapsed")
                or synth.get("adaptive_probation_hard_collapse")
            )
            if bool(cfg.rescue_reject_collapse_source) and collapsed:
                reject_reasons.append(f"{tier_label}:collapsed")
                continue
            absolute_advantage = float(s_mean) - float(target_nsucc_mean)
            if absolute_advantage < float(cfg.rescue_archive_min_absolute_advantage):
                reject_reasons.append(f"{tier_label}:low_absolute_advantage")
                continue
            adv_sigma = float(absolute_advantage) / float(effective_sigma)
            if (not math.isfinite(adv_sigma)) or adv_sigma < float(
                cfg.rescue_min_source_advantage_sigma
            ):
                reject_reasons.append(f"{tier_label}:low_advantage")
                continue
            return {
                "action": "copy",
                "tier": tier_label,
                "checkpoint_type": ckpt_type,
                "source_policy": int(synth.get("policy_idx", -1)),
                "source_state": synth,
                "strict_candidates": [],
                "relaxed_candidates": [],
                "strict_filter_reasons": {},
                "relaxed_filter_reasons": {},
                "archive_reason": "ok",
                "adaptive_rescue_reason": "selected_archive",
                "adaptive_rescue_source_kind": "archive",
                "adaptive_source_rank": -1,
                "adaptive_source_advantage_sigma": float(adv_sigma),
                "adaptive_source_absolute_advantage": float(absolute_advantage),
                "adaptive_population_nsucc_sample_count": int(len(means)),
                "adaptive_population_nsucc_sigma": float(pop_sigma),
                "adaptive_population_nsucc_effective_sigma": float(effective_sigma),
                "adaptive_archive_age_known": bool(age_known),
                "adaptive_archive_age_frames": int(age_frames),
                "adaptive_log_tolerance_gap": float(gap),
                "adaptive_rescue_archive_tier": tier_label,
            }
        empty["archive_reason"] = ";".join(reject_reasons) or "no_compatible_archive"
        empty["adaptive_rescue_reason"] = empty["archive_reason"]
        return empty

    def _select_adaptive_rescue_source_cascade(
        self,
        *,
        sorted_policies: list[int],
        population: dict[int, dict[str, Any] | None],
        env_frames: int,
        target_tolerance: float,
        target_nsucc_mean: float,
        n_top: int,
    ) -> dict[str, Any]:
        """Cascade: live adaptive -> archive adaptive -> ordinary fallback."""
        pidx = int(self.policy_idx)
        pop_means: list[float] = []
        for st in (population or {}).values():
            if not st:
                continue
            try:
                m = float(st.get("nsucc_mean", float("nan")))
            except Exception:
                continue
            if math.isfinite(m):
                pop_means.append(m)
        live = self._select_adaptive_degradation_rescue_source(
            sorted_policies=sorted_policies,
            population=population,
            env_frames=env_frames,
            target_tolerance=target_tolerance,
            target_nsucc_mean=target_nsucc_mean,
        )
        if live.get("action") == "copy":
            live = dict(live)
            live["adaptive_rescue_source_kind"] = "live"
            live.setdefault("adaptive_rescue_fallback_to_ordinary", False)
            return live
        state = self._adaptive_precision_state(pidx, strict_local=True)
        cfg = state[0] if state is not None else None
        if cfg is not None and bool(cfg.rescue_archive_enable):
            arch = self._select_adaptive_archive_rescue_source(
                env_frames=env_frames,
                target_tolerance=target_tolerance,
                target_nsucc_mean=target_nsucc_mean,
                population_nsucc_means=pop_means,
            )
            if arch.get("action") == "copy":
                arch = dict(arch)
                arch["adaptive_rescue_source_kind"] = "archive"
                arch.setdefault("adaptive_rescue_fallback_to_ordinary", False)
                return arch
        ordinary = self._select_source(
            sorted_policies=sorted_policies,
            population=population,
            env_frames=env_frames,
            n_top=n_top,
        )
        if ordinary.get("action") == "copy":
            ordinary = dict(ordinary)
            ordinary["adaptive_rescue_fallback_to_ordinary"] = True
            ordinary["adaptive_rescue_reason"] = "ordinary_fallback"
            ordinary["adaptive_rescue_source_kind"] = "ordinary_fallback"
            return ordinary
        skip = {
            "action": "skip",
            "tier": "skipped",
            "checkpoint_type": "none",
            "source_policy": -1,
            "source_state": None,
            "strict_candidates": [],
            "relaxed_candidates": [],
            "strict_filter_reasons": {},
            "relaxed_filter_reasons": {},
            "archive_reason": str(live.get("archive_reason", "no_compatible_source")),
            "adaptive_rescue_reason": str(
                live.get("adaptive_rescue_reason", "no_compatible_source")
            ),
            "adaptive_rescue_source_kind": "none",
            "adaptive_source_rank": -1,
            "adaptive_source_advantage_sigma": 0.0,
        }
        return skip

    def _update_curriculum_windows_and_control(
        self,
        population: dict[int, dict[str, Any] | None],
        env_frames: int,
    ) -> None:
        """Append rolling nsucc samples and write per-policy control JSON."""
        self._collapse_detected_this_tick = False
        if self.pbt_iteration != self._last_curriculum_pbt_iteration:
            self._last_curriculum_pbt_iteration = int(self.pbt_iteration)
            self._shrink_issued_this_window = {
                i: False for i in range(self.num_policies)
            }
            self._adaptive_ordinary_replacements_this_tick = 0
            self._last_adaptive_diversity_blocked = False
        self._curriculum_control_dir.mkdir(parents=True, exist_ok=True)
        for policy_idx in range(self.num_policies):
            state = population.get(policy_idx)
            window = self._policy_window(policy_idx)
            if state is not None:
                sample = self._window_sample_from_state(
                    state,
                    env_frames=env_frames,
                    pbt_iteration=self.pbt_iteration,
                )
                window.append(sample)
                if self.adaptive_precision_enable and policy_idx == self.policy_idx:
                    try:
                        self._adaptive_precision_update(
                            sample=sample, state=state, env_frames=int(env_frames)
                        )
                    except Exception as _apc_exc:
                        if self._adaptive_precision_required():
                            raise RuntimeError(
                                f"adaptive precision update failed for formal profile: {_apc_exc}"
                            ) from _apc_exc
                        print(f"[adaptive-precision] update failed (ignored): {_apc_exc}")
                if self.same_tol_retention_enable:
                    self._maybe_restore_bridge_state(policy_idx)
                    tracker = self._same_tol_trackers.setdefault(
                        policy_idx,
                        ToleranceBinRetentionTracker(
                            bins=self.same_tol_retention_bins,
                            ladder=(
                                self.bridge_tolerance_ladder
                                if self.bridge_curriculum_enable else ()
                            ),
                        ),
                    )
                    if self.bridge_curriculum_enable and not tracker.ladder:
                        tracker.ladder = self.bridge_tolerance_ladder
                    prev_tol = self._last_confirmed_tolerance.get(policy_idx)
                    cur_sample_tol = float(sample.success_tolerance)
                    if (
                        self.bridge_curriculum_enable
                        and prev_tol is not None
                        and cur_sample_tol < float(prev_tol) - 1e-9
                    ):
                        # Tolerance actually moved stricter: stamp level timer now.
                        self._bridge_level_entry_frame[policy_idx] = int(env_frames)
                        self._bridge_probation_bad_streak[policy_idx] = 0
                        self._repeated_collapse_count[policy_idx] = 0
                        self._bridge_shrink_count[policy_idx] = (
                            int(self._bridge_shrink_count.get(policy_idx, 0)) + 1
                        )
                    prev_lineage = tracker.lineage_id
                    tracker.update(
                        tolerance=sample.success_tolerance,
                        nsucc_mean=sample.nsucc_mean,
                        frame=env_frames,
                        lineage_id=sample.lineage_id,
                        reset_on_lineage_change=self.same_tol_retention_reset_on_lineage_change,
                        use_bridge_levels=bool(self.bridge_curriculum_enable),
                    )
                    if (
                        self.same_tol_retention_reset_on_lineage_change
                        and prev_lineage is not None
                        and sample.lineage_id is not None
                        and str(prev_lineage) != str(sample.lineage_id)
                    ):
                        # Cross-copy / lineage change: ordinary rollback streak
                        # must not carry into the new lineage.
                        self._repeated_collapse_count[policy_idx] = 0
                        self._bridge_probation_bad_streak[policy_idx] = 0
                    self._last_confirmed_tolerance[policy_idx] = cur_sample_tol
                    if (
                        self.bridge_curriculum_enable
                        and prev_tol is not None
                        and abs(cur_sample_tol - float(prev_tol)) > 1e-9
                    ):
                        # Level change (any direction) clears probation bad streak.
                        level_now = bridge_level_id(
                            cur_sample_tol, self.bridge_tolerance_ladder
                        )
                        level_prev = bridge_level_id(
                            float(prev_tol), self.bridge_tolerance_ladder
                        )
                        if level_now != level_prev:
                            self._bridge_probation_bad_streak[policy_idx] = 0
                            self._repeated_collapse_count[policy_idx] = 0
                            # Lineage-local ordinary rollback streak must not
                            # carry across ladder rungs after a real shrink.
                nm = window.samples[-1].nsucc_mean
                peak = self._recent_peak_nsucc_mean.get(policy_idx, 0.0)
                if nm > peak:
                    self._recent_peak_nsucc_mean[policy_idx] = nm

            control = self._build_curriculum_control_state(
                policy_idx,
                window,
                env_frames,
                population=population,
            )
            write_curriculum_control(
                self._curriculum_control_dir / f"{policy_idx:03d}.json",
                control,
            )

        if self.writer is not None and self.policy_idx in self._policy_nsucc_windows:
            own_window = self._policy_nsucc_windows[self.policy_idx]
            own_control = self._build_curriculum_control_state(
                self.policy_idx,
                own_window,
                env_frames,
                population=population,
            )
            self._log_curriculum_control_scalars(own_control, own_window, env_frames)

    @staticmethod
    def _policy_restart_sim_steps(
        policy_idx: int,
        population: dict[int, dict[str, Any] | None] | None,
    ) -> int:
        if not population:
            return 0
        state = population.get(policy_idx)
        if not state:
            return 0
        cur = state.get("curriculum") or {}
        fsr = cur.get("frame_since_restart")
        if isinstance(fsr, (int, float)) and not math.isnan(float(fsr)):
            return int(fsr)
        return 0

    def _load_policy_wm_imagined_eval(self, policy_idx: int) -> dict[str, Any] | None:
        """Load fresh imagined-eval JSON written by train.py for one policy."""
        path = self.workspace / "wm_imagined" / f"{int(policy_idx):03d}.json"
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text())
        except Exception:
            return None

    def _evaluate_wm_curriculum_controller(
        self,
        policy_idx: int,
        window: RollingNsuccWindow,
        env_frames: int,
    ) -> dict[str, Any]:
        """Return WM curriculum controller audit dict for this tick.

        Soft Activation Gate v1: this is a ``shrink_candidate`` gate.  There is
        NO ``tol <= 0.04`` hard gate; eval missing / stale / invalid fails OPEN
        (does not block).  ``tol_bin`` and per-bin blocked-shrink counters are
        recorded for diagnostics.
        """
        from dexpbt_isaaclab.world_model.wm_activation_gate import TOL_BINS, tol_bin

        cur_tol = float("nan")
        if window.samples:
            cur_tol = float(window.samples[-1].success_tolerance)
        tb = tol_bin(cur_tol) if cur_tol == cur_tol else "le_004"
        out: dict[str, Any] = {
            "active": False,
            "eval_valid": False,
            "drop_risk": 0.0,
            "success_prob": 0.0,
            "recovery_prob": 0.0,
            "uncertainty": 1.0,
            "blocked_shrink": False,
            "tol_bin": tb,
            "freeze_until": int(
                self._wm_curriculum_freeze_until.get(policy_idx, 0)
            ),
            "fallback_reason_id": 0,
        }
        for _b in TOL_BINS:
            out[f"blocked_shrink_tol_{_b}"] = 0.0

        gate_off = (
            not getattr(self, "wm_activation_enable", True)
            or str(getattr(self, "wm_curriculum_gate_mode", "shrink_candidate")) == "off"
        )

        def _mark_blocked() -> None:
            out["blocked_shrink"] = True
            out[f"blocked_shrink_tol_{tb}"] = 1.0
            counts = getattr(self, "_wm_curriculum_blocked_by_tol_bin", None)
            if counts is None:
                counts = {b: 0 for b in TOL_BINS}
                self._wm_curriculum_blocked_by_tol_bin = counts
            counts[tb] = int(counts.get(tb, 0)) + 1

        if gate_off or not self.wm_curriculum_controller_enable:
            out["fallback_reason_id"] = 1
            return out
        if int(env_frames) < int(self.wm_curriculum_controller_start_after_frames):
            out["fallback_reason_id"] = 2
            return out
        out["active"] = True
        wm_freeze_until = int(self._wm_curriculum_freeze_until.get(policy_idx, 0))
        if int(env_frames) < wm_freeze_until:
            _mark_blocked()
            out["freeze_until"] = wm_freeze_until
            out["fallback_reason_id"] = 6
            return out
        payload = self._load_policy_wm_imagined_eval(policy_idx)
        if not payload:
            out["fallback_reason_id"] = 3
            return out
        out["drop_risk"] = float(payload.get("wm_drop_risk", 0.0))
        out["success_prob"] = float(payload.get("wm_success_prob", 0.0))
        out["recovery_prob"] = float(payload.get("wm_recovery_prob", 0.0))
        out["uncertainty"] = float(payload.get("uncertainty", 1.0))
        out["imagined_valid_strict"] = bool(
            payload.get(
                "imagined_valid_strict",
                payload.get("imagined_valid", False),
            )
        )
        out["imagined_valid_soft"] = bool(payload.get("imagined_valid_soft", False))
        out["eval_valid"] = bool(out["imagined_valid_strict"])
        if not out["eval_valid"]:
            out["fallback_reason_id"] = 3
            return out
        nsucc_mean = 0.0
        if window.samples:
            nsucc_mean = float(window.samples[-1].nsucc_mean)
        risk_high = (
            out["drop_risk"] >= self.wm_curriculum_controller_drop_risk_threshold
            or out["uncertainty"]
            >= self.wm_curriculum_controller_uncertainty_threshold
            or out["success_prob"] < self.wm_curriculum_controller_min_success_prob
        )
        if self.wm_curriculum_controller_use_recovery_prob:
            risk_high = risk_high or (
                out["recovery_prob"]
                < self.wm_curriculum_controller_min_success_prob
            )
        if (
            risk_high
            and nsucc_mean >= self.wm_curriculum_controller_min_nsucc_mean
            and self.wm_curriculum_controller_hard_block_shrink
        ):
            _mark_blocked()
            out["freeze_until"] = int(env_frames) + int(
                self.wm_curriculum_controller_freeze_frames
            )
            self._wm_curriculum_freeze_until[policy_idx] = out["freeze_until"]
            out["fallback_reason_id"] = 5
        else:
            out["fallback_reason_id"] = 4
        return out

    @staticmethod
    def _rewrite_train_arg_profile(train_args: list[str], profile_name: str) -> list[str]:
        flag = "--wm_strategy_profile"
        out: list[str] = []
        i = 0
        while i < len(train_args):
            arg = train_args[i]
            if arg == flag:
                i += 2
                continue
            if arg.startswith(f"{flag}="):
                i += 1
                continue
            out.append(arg)
            i += 1
        out.append(f"{flag}={profile_name}")
        return out

    def _count_profile_diversity(
        self, population: dict[int, dict[str, Any] | None] | None
    ) -> tuple[int, set[str]]:
        profiles: set[str] = set()
        if population:
            for st in population.values():
                if not st:
                    continue
                name = str(st.get("wm_strategy_profile_name", "legacy") or "legacy")
                profiles.add(name)
        profiles.add(str(self.wm_strategy_profile_name or "legacy"))
        return len(profiles), profiles

    def _profile_from_population_state(
        self, population: dict[int, dict[str, Any] | None] | None, policy_idx: int
    ) -> tuple[str, int, int]:
        st = (population or {}).get(policy_idx) or {}
        name = str(st.get("wm_strategy_profile_name", "legacy") or "legacy")
        gen = int(st.get("wm_strategy_profile_generation", 0))
        return name, gen, int(st.get("wm_strategy_profile_id", 0))

    def _build_curriculum_control_state(
        self,
        policy_idx: int,
        window: RollingNsuccWindow,
        env_frames: int,
        population: dict[int, dict[str, Any] | None] | None = None,
    ) -> CurriculumControlState:
        # Per-tick transient: do not inherit a prior tick's hard-collapse latch.
        self._adaptive_hard_collapse_event_frame[int(policy_idx)] = -1
        gate_pass, consec_count, blocked_reason = evaluate_mastery_gate(
            window,
            mastery_mean=self.mastery_nsucc_mean,
            mastery_max=self.mastery_nsucc_max,
            consecutive_windows=self.mastery_consecutive_windows,
        )
        recent_peak = self._recent_peak_nsucc_mean.get(policy_idx, 0.0)
        collapse_detected = False
        early_collapse_detected = False
        hard_collapse_detected = False
        drop_ratio = window.drop_ratio_vs_prior(2)
        forced_tolerance = 0.0
        collapse_rollback_active = False
        adaptive_probation_active = False
        adaptive_probation_can_shrink = True
        adaptive_probation_block_reason = ""
        adaptive_probation_ordinary_rollback_blocked = False
        adaptive_probation_hard_collapse = False
        adaptive_probation_rollback_triggered = False
        adaptive_probation_rollback_from_tol = 0.0
        adaptive_probation_rollback_to_tol = 0.0
        adaptive_probation_hard_collapse_streak = 0
        adaptive_probation_entry_tolerance = 0.0
        adaptive_probation_entry_frame = -1
        adaptive_probation_hard_collapse_trigger_source = ""
        adaptive_probation_hard_collapse_independent = False
        adaptive_probation_legacy_collapse_also_active = False
        adaptive_probation_freeze_only_streak = 0
        adaptive_rollback_target_source = ""
        adaptive_rollback_log_gap = 0.0
        adaptive_precision_active = bool(self.adaptive_precision_enable)
        adaptive_curriculum_progress = float(
            getattr(self, "_adaptive_curriculum_progress", 0.0) or 0.0
        )
        rollback_cooldown_active = False
        rollback_reason = ""
        rollback_count = int(self._rollback_count.get(policy_idx, 0))
        last_rollback_frame = int(self._last_rollback_frame.get(policy_idx, -10**18))
        rollback_and_freeze_same_tick = False
        curriculum_rollback_triggered = False
        curriculum_rollback_from_tol = 0.0
        curriculum_rollback_to_tol = 0.0
        curriculum_rollback_freeze_until = int(
            self._curriculum_safety_freeze_until.get(policy_idx, 0)
        )
        global_retention, global_nsucc_mean, long_peak = self._compute_global_retention(
            window
        )
        stable_ticks = self._count_tolerance_stable_ticks(window)
        cur_tol = (
            float(window.samples[-1].success_tolerance) if window.samples else 0.075
        )
        same_tol = self._same_tol_retention_snapshot(policy_idx, cur_tol)
        # In bridge mode all ordinary curriculum decisions use only the
        # current ladder level (or coarse bin when bridge is off).  During
        # warmup retention is deliberately invalid, never a synthetic low.
        decision_retention = (
            float(same_tol["retention"]) if bool(same_tol["valid"]) else 1.0
        )
        if self.same_tol_retention_enable:
            global_retention = decision_retention
            global_nsucc_mean = (
                float(window.samples[-1].nsucc_mean) if window.samples else 0.0
            )
            long_peak = float(same_tol["peak"])
        if (
            self.bridge_curriculum_enable and self.bridge_probation_enable
            and cur_tol <= self.bridge_probation_entry_tol + 1e-9
            and policy_idx not in self._bridge_probation_entry_frame
        ):
            self._bridge_probation_entry_frame[policy_idx] = int(env_frames)
        probation_entry_early = int(self._bridge_probation_entry_frame.get(policy_idx, -1))
        probation_active_early = bool(
            self.bridge_probation_enable
            and probation_entry_early >= 0
            and int(env_frames) - probation_entry_early < self.bridge_probation_frames
        )
        probation_ordinary_block = bool(
            self.bridge_probation_block_ordinary_rollback
            and probation_active_early
        )
        adaptive_state = (
            self._adaptive_precision_state(policy_idx)
            if self.adaptive_precision_enable
            else None
        )
        adaptive_ordinary_block = False
        if adaptive_state is not None:
            _acfg, _aprob, _atrack = adaptive_state
            adaptive_probation_active = bool(_aprob.active)
            adaptive_probation_entry_tolerance = float(_aprob.entry_tolerance)
            adaptive_probation_entry_frame = int(_aprob.entry_frame)
            adaptive_probation_hard_collapse_streak = int(_aprob.hard_collapse_streak)
            adaptive_ordinary_block = bool(
                _aprob.active
                and _acfg.probation_block_ordinary_rollback
                and not _aprob.hard_collapsed(_acfg)
            )
            adaptive_probation_ordinary_rollback_blocked = bool(adaptive_ordinary_block)
        legacy_hard_collapse = False
        bridge_probation_hard_collapse = False
        bridge_probation_legacy_collapse_suppressed = False
        bridge_probation_rollback_reason = ""
        danger_zone = bool(
            self.curriculum_danger_zone_enable
            and cur_tol <= self.curriculum_danger_tol + 1e-9
        )
        repeated_collapse_count = int(
            self._repeated_collapse_count.get(policy_idx, 0)
        )
        if self.curriculum_collapse_detect:
            if self.pbt_early_collapse_detect:
                early_detected, early_drop, _ = evaluate_collapse_freeze(
                    window,
                    drop_ratio_threshold=self.early_collapse_drop_ratio,
                    patience=self.early_collapse_patience,
                    recent_peak=recent_peak,
                )
                if early_detected:
                    collapse_detected = True
                    early_collapse_detected = True
                    drop_ratio = early_drop
            hard_collapse_detected, hard_drop, _ = evaluate_collapse_freeze(
                window,
                drop_ratio_threshold=self.collapse_drop_ratio,
                patience=self.collapse_patience,
                recent_peak=recent_peak,
            )
            legacy_hard_collapse = bool(hard_collapse_detected)
            if hard_collapse_detected:
                collapse_detected = True
                drop_ratio = hard_drop

            # Bridge probation: evaluate same-level hard-collapse BEFORE
            # mutating ordinary repeated_collapse_count, so a suppressed
            # cross-tol peak cannot accumulate toward post-probation rollback.
            if probation_active_early:
                nsucc_now = (
                    float(window.samples[-1].nsucc_mean) if window.samples else 0.0
                )
                level_warm = bool(same_tol["valid"])
                if nsucc_now < self.bridge_probation_hard_collapse_mean and level_warm:
                    self._bridge_probation_bad_streak[policy_idx] = (
                        int(self._bridge_probation_bad_streak.get(policy_idx, 0)) + 1
                    )
                else:
                    if level_warm:
                        self._bridge_probation_bad_streak[policy_idx] = 0
                bad_streak = int(self._bridge_probation_bad_streak.get(policy_idx, 0))
                if (
                    self.bridge_probation_allow_hard_collapse_rollback
                    and level_warm
                    and bad_streak >= self.bridge_probation_hard_collapse_patience
                ):
                    bridge_probation_hard_collapse = True
                    hard_collapse_detected = True
                    bridge_probation_rollback_reason = "bridge_probation_hard_collapse"
                if legacy_hard_collapse and not bridge_probation_hard_collapse:
                    bridge_probation_legacy_collapse_suppressed = True
                    hard_collapse_detected = False
                    self._bridge_probation_suppressed_legacy_collapse_count[policy_idx] = (
                        int(
                            self._bridge_probation_suppressed_legacy_collapse_count.get(
                                policy_idx, 0
                            )
                        )
                        + 1
                    )
                    # Diagnostic freeze only; do not advance ordinary rollback streak.
                    self._collapse_detected_this_tick = True
                    self._collapse_freeze_until[policy_idx] = (
                        int(env_frames) + self.collapse_freeze_env_frames
                    )
                    collapse_detected = True
                    self._append_curriculum_safety_event({
                        "event_type": "bridge_probation_legacy_collapse_suppressed",
                        "bridge_probation/legacy_collapse_suppressed": True,
                        "bridge_probation/suppressed_legacy_collapse_count": int(
                            self._bridge_probation_suppressed_legacy_collapse_count[
                                policy_idx
                            ]
                        ),
                        "bridge_probation/ordinary_repeated_count": int(
                            self._repeated_collapse_count.get(policy_idx, 0)
                        ),
                        "bridge_probation/bad_streak": int(
                            self._bridge_probation_bad_streak.get(policy_idx, 0)
                        ),
                        "env_frames": int(env_frames),
                        "success_tolerance": float(cur_tol),
                    })

            if collapse_detected and not (
                probation_active_early
                and legacy_hard_collapse
                and not bridge_probation_hard_collapse
            ):
                self._collapse_detected_this_tick = True
                self._collapse_freeze_until[policy_idx] = (
                    int(env_frames) + self.collapse_freeze_env_frames
                )
                repeated_collapse_count += 1
                self._repeated_collapse_count[policy_idx] = repeated_collapse_count
            elif (
                probation_active_early
                and legacy_hard_collapse
                and not bridge_probation_hard_collapse
            ):
                # Keep local view of ordinary count unchanged (no += above).
                repeated_collapse_count = int(
                    self._repeated_collapse_count.get(policy_idx, 0)
                )

            # Adaptive hard-collapse is independent of the legacy/bridge signal.
            adaptive_hard_collapse_now = False
            _acfg = None
            _aprob = None
            if adaptive_state is not None:
                _acfg, _aprob, _atrack = adaptive_state
                timeout_armed = bool(
                    self._adaptive_probation_timeout.pop(policy_idx, False)
                )
                if (
                    _aprob.active
                    and _acfg.probation_allow_hard_collapse_rollback
                    and (
                        _aprob.hard_collapsed(_acfg)
                        or timeout_armed
                    )
                ):
                    adaptive_hard_collapse_now = True
                    adaptive_probation_hard_collapse = True
                    adaptive_ordinary_block = False
                    adaptive_probation_ordinary_rollback_blocked = False
                    # Latch before any streak reset so ternary emergency bypass
                    # still observes hard collapse on this tick.
                    self._adaptive_hard_collapse_event_frame[int(policy_idx)] = int(
                        env_frames
                    )
                    if timeout_armed:
                        bridge_probation_rollback_reason = str(
                            self._adaptive_probation_timeout_reason.pop(
                                policy_idx, "probation_max_frames"
                            )
                        )
            legacy_hard_collapse_now = bool(hard_collapse_detected)
            if bridge_probation_hard_collapse:
                adaptive_probation_hard_collapse_trigger_source = "bridge"
            elif adaptive_hard_collapse_now and legacy_hard_collapse:
                adaptive_probation_hard_collapse_trigger_source = "adaptive_and_legacy"
            elif adaptive_hard_collapse_now:
                adaptive_probation_hard_collapse_trigger_source = "adaptive_only"
            elif hard_collapse_detected:
                adaptive_probation_hard_collapse_trigger_source = "legacy_only"
            adaptive_probation_hard_collapse_independent = bool(
                adaptive_hard_collapse_now and not legacy_hard_collapse
            )
            adaptive_probation_legacy_collapse_also_active = bool(
                adaptive_hard_collapse_now and legacy_hard_collapse
            )
            self._adaptive_hard_collapse_trigger_source[policy_idx] = (
                adaptive_probation_hard_collapse_trigger_source
            )
            adaptive_probation_freeze_only_streak = int(
                self._adaptive_freeze_only_streak.get(policy_idx, 0)
            )

            need_rollback_eval = bool(hard_collapse_detected or adaptive_hard_collapse_now)
            if need_rollback_eval:
                rollback_cooldown_active = (
                    int(env_frames) - last_rollback_frame
                    < self.curriculum_rollback_cooldown_frames
                )
                allow_by_cooldown = not rollback_cooldown_active
                allow_by_shrink = (
                    int(self._shrinks_since_rollback.get(policy_idx, 0)) > 0
                )
                # During probation only bridge/adaptive hard-collapse may set forced tol.
                allow_forced = True
                if (
                    probation_active_early
                    and not bridge_probation_hard_collapse
                    and not adaptive_hard_collapse_now
                ):
                    allow_forced = False
                    bridge_probation_legacy_collapse_suppressed = True
                    bridge_probation_rollback_reason = (
                        bridge_probation_rollback_reason
                        or "legacy_hard_collapse_suppressed"
                    )
                if adaptive_ordinary_block and not adaptive_hard_collapse_now:
                    allow_forced = False
                    adaptive_probation_ordinary_rollback_blocked = True
                if allow_forced and (
                    allow_by_cooldown
                    or allow_by_shrink
                    or adaptive_hard_collapse_now
                ):
                    if bridge_probation_hard_collapse:
                        forced_tolerance = float(
                            previous_bridge_tolerance(
                                cur_tol, ladder=self.bridge_tolerance_ladder
                            )
                        )
                        rollback_reason = "bridge_probation_hard_collapse"
                        bridge_probation_rollback_reason = rollback_reason
                    elif adaptive_hard_collapse_now:
                        from dexpbt_isaaclab.pbt.adaptive_precision_consolidation import (
                            select_adaptive_rollback_target,
                        )

                        decision = select_adaptive_rollback_target(
                            current_tolerance=float(cur_tol),
                            previous_tolerance_before_entry=(
                                float(_aprob.previous_tolerance_before_entry)
                                if float(getattr(_aprob, "previous_tolerance_before_entry", 0.0) or 0.0) > 0
                                else None
                            ),
                            source_tolerance=(
                                float(_aprob.source_tolerance)
                                if float(getattr(_aprob, "source_tolerance", 0.0) or 0.0) > 0
                                else None
                            ),
                            last_mastered_safe_tolerance=(
                                float(self._last_mastered_safe_tolerance.get(policy_idx, 0.0) or 0.0)
                                or None
                            ),
                            initial_tolerance=(
                                float(getattr(self, "_adaptive_eps_init", {}).get(policy_idx, 0.0) or 0.0)
                                or None
                            ),
                            shrink_factor=float(self.curriculum_shrink_factor)
                            if getattr(self, "curriculum_shrink_factor", 0)
                            else 0.95,
                            max_log_gap=float(_acfg.probation_rollback_max_log_gap),
                            eps=float(_acfg.eps),
                        )
                        adaptive_rollback_target_source = str(decision.source)
                        adaptive_rollback_log_gap = float(decision.log_gap)
                        if decision.eligible:
                            forced_tolerance = float(decision.target_tolerance)
                            rollback_reason = (
                                f"adaptive_probation_hard_collapse:{decision.source}"
                            )
                            adaptive_probation_rollback_triggered = True
                            adaptive_probation_rollback_from_tol = float(cur_tol)
                            adaptive_probation_rollback_to_tol = float(forced_tolerance)
                            _aprob.hard_collapse_streak = 0
                            self._adaptive_freeze_only_streak[policy_idx] = 0
                            adaptive_probation_freeze_only_streak = 0
                        else:
                            allow_forced = False
                            forced_tolerance = 0.0
                            rollback_reason = (
                                f"adaptive_probation_hard_collapse_freeze_only:{decision.reason}"
                            )
                            adaptive_probation_hard_collapse = True
                            self._adaptive_freeze_only_streak[policy_idx] = (
                                int(self._adaptive_freeze_only_streak.get(policy_idx, 0)) + 1
                            )
                            adaptive_probation_freeze_only_streak = int(
                                self._adaptive_freeze_only_streak[policy_idx]
                            )
                            self._adaptive_rollback_freeze_only_count[policy_idx] = (
                                int(
                                    self._adaptive_rollback_freeze_only_count.get(
                                        policy_idx, 0
                                    )
                                )
                                + 1
                            )
                            self._collapse_freeze_until[policy_idx] = (
                                int(env_frames) + int(self.collapse_freeze_env_frames)
                            )
                            collapse_detected = True
                    else:
                        forced_tolerance = float(
                            self._last_mastered_tolerance.get(policy_idx, 0.075)
                        )
                        rollback_reason = (
                            "cooldown_elapsed"
                            if allow_by_cooldown
                            else "shrink_after_last_rollback"
                        )
                    if allow_forced and float(forced_tolerance) > 0:
                        collapse_rollback_active = True
                        self._last_rollback_frame[policy_idx] = int(env_frames)
                        last_rollback_frame = int(env_frames)
                        self._rollback_count[policy_idx] = rollback_count + 1
                        rollback_count += 1
                        self._shrinks_since_rollback[policy_idx] = 0
                        rollback_and_freeze_same_tick = collapse_detected
                        if bridge_probation_hard_collapse:
                            self._bridge_probation_bad_streak[policy_idx] = 0
                            self._repeated_collapse_count[policy_idx] = 0
                            repeated_collapse_count = 0
                        if adaptive_hard_collapse_now and adaptive_probation_rollback_triggered:
                            self._repeated_collapse_count[policy_idx] = 0
                            repeated_collapse_count = 0
                elif allow_forced:
                    rollback_reason = "cooldown_blocked"
            if (
                self.curriculum_rollback_on_repeated_collapse
                and self.curriculum_rollback_to_last_mastered
                and not probation_ordinary_block
                and not adaptive_ordinary_block
                and danger_zone
                and repeated_collapse_count
                >= self.curriculum_rollback_collapse_patience
            ):
                safe_tol = float(
                    self._last_mastered_safe_tolerance.get(policy_idx, 0.0)
                )
                if safe_tol > 1e-9 and cur_tol < safe_tol - 1e-9:
                    forced_tolerance = safe_tol
                    collapse_rollback_active = True
                    curriculum_rollback_triggered = True
                    curriculum_rollback_from_tol = float(cur_tol)
                    curriculum_rollback_to_tol = float(safe_tol)
                    rollback_reason = "repeated_collapse_to_last_mastered"
                    self._last_rollback_frame[policy_idx] = int(env_frames)
                    last_rollback_frame = int(env_frames)
                    self._rollback_count[policy_idx] = rollback_count + 1
                    rollback_count += 1
                    self._shrinks_since_rollback[policy_idx] = 0
                    self._repeated_collapse_count[policy_idx] = 0
                    repeated_collapse_count = 0
                    curriculum_rollback_freeze_until = int(env_frames) + int(
                        self.curriculum_rollback_freeze_frames
                    )
                    self._curriculum_safety_freeze_until[policy_idx] = (
                        curriculum_rollback_freeze_until
                    )
                    self._append_curriculum_safety_event(
                        {
                            "event": "repeated_collapse_rollback",
                            "policy_idx": int(policy_idx),
                            "env_frames": int(env_frames),
                            "from_tol": float(cur_tol),
                            "to_tol": float(safe_tol),
                            "repeated_collapse_count": int(
                                self.curriculum_rollback_collapse_patience
                            ),
                        }
                    )
                elif repeated_collapse_count >= self.curriculum_rollback_collapse_patience:
                    curriculum_rollback_freeze_until = int(env_frames) + int(
                        self.curriculum_rollback_freeze_frames
                    )
                    self._curriculum_safety_freeze_until[policy_idx] = (
                        curriculum_rollback_freeze_until
                    )
                    rollback_reason = "repeated_collapse_freeze_only"
                    self._append_curriculum_safety_event(
                        {
                            "event": "repeated_collapse_freeze_only",
                            "policy_idx": int(policy_idx),
                            "env_frames": int(env_frames),
                            "current_tol": float(cur_tol),
                        }
                    )

        freeze_until = self._collapse_freeze_until.get(policy_idx, 0)
        collapse_freeze_active = int(env_frames) < freeze_until

        control_path = self._curriculum_control_dir / f"{policy_idx:03d}.json"
        prev_control = read_curriculum_control(control_path)
        restart_sim_steps = self._policy_restart_sim_steps(policy_idx, population)
        in_consolidation = (
            prev_control is not None
            and post_shrink_consolidation_active(restart_sim_steps, prev_control)
        )
        consolidation_until = (
            int(prev_control.post_shrink_consolidation_until_restart_step)
            if prev_control
            else 0
        )
        shrink_issued = self._shrink_issued_this_window.get(policy_idx, False)
        shrinks_in_window = 1 if shrink_issued else 0

        if self.curriculum_mode == "mastery":
            if collapse_freeze_active:
                shrink_blocked_reason = "collapse_freeze_active"
                shrink_token = False
                allow_shrink = False
            elif in_consolidation:
                shrink_blocked_reason = "post_shrink_consolidation"
                shrink_token = False
                allow_shrink = False
            elif self.pbt_single_shrink_per_window:
                if gate_pass and not shrink_issued:
                    shrink_token = True
                    self._shrink_issued_this_window[policy_idx] = True
                    self._shrinks_since_rollback[policy_idx] = (
                        int(self._shrinks_since_rollback.get(policy_idx, 0)) + 1
                    )
                    shrink_blocked_reason = ""
                    allow_shrink = True
                else:
                    shrink_token = False
                    allow_shrink = False
                    if shrink_issued:
                        shrink_blocked_reason = "shrink_already_issued_this_window"
                    else:
                        shrink_blocked_reason = blocked_reason
            else:
                allow_shrink = gate_pass and not collapse_freeze_active
                shrink_token = allow_shrink
                shrink_blocked_reason = "" if allow_shrink else blocked_reason
        else:
            allow_shrink = True
            shrink_token = True
            shrink_blocked_reason = ""

        # Phase A4 P2 (2026-06-08): adaptive-fast curriculum tier evaluation.
        # Runs only AFTER the existing chain (mastery / collapse / consolidation /
        # post-restart-freeze / single-shot token) has decided shrink_token.
        # If shrink_token is already False, adaptive_fast does not flip it on;
        # it only:
        #   * picks a tier-specific shrink factor + consolidation length to
        #     write into the control JSON (consumed env-side), OR
        #   * blocks an otherwise-allowed shrink when nsucc_mean < low_mean.
        adaptive_shrink_factor: float = 0.0
        adaptive_consolidation_frames: int = 0
        shrink_reason: str = ""
        nsucc_mean_used = 0.0
        retention_used = 0.0
        blocked_by_recovery = False
        blocked_by_collapse = bool(collapse_freeze_active)
        blocked_by_consolidation = bool(in_consolidation)
        blocked_by_mastery_gate = bool(
            self.curriculum_mode == "mastery"
            and not gate_pass
            and not blocked_by_collapse
            and not blocked_by_consolidation
        )
        blocked_by_adaptive_fast_low_nsucc = False
        curriculum_strong_shrink_allowed = True
        curriculum_strong_shrink_block_reason = ""
        curriculum_strong_shrink_global_gate_active = bool(
            self.curriculum_strong_shrink_global_retention_gate
        )
        curriculum_danger_blocked_shrink = False
        curriculum_danger_block_reason = ""
        curriculum_danger_interval_mult = 1.0
        curriculum_last_mastered_tolerance = float(
            self._last_mastered_safe_tolerance.get(policy_idx, 0.0)
        )
        curriculum_last_mastered_frame = int(
            self._last_mastered_safe_frame.get(policy_idx, -1)
        )
        curriculum_last_mastered_nsucc_mean = float(
            self._last_mastered_safe_nsucc_mean.get(policy_idx, 0.0)
        )
        curriculum_last_mastered_global_retention = float(
            self._last_mastered_safe_global_retention.get(policy_idx, 0.0)
        )

        if self.curriculum_speed_mode == "adaptive_fast":
            # In recovery stage we still allow shrink but force the legacy
            # factor and use a longer consolidation window.
            in_recovery = (
                int(env_frames) - int(self._frames_at_restart)
                < int(self.curriculum_freeze_frames)
            ) or bool(rollback_cooldown_active)
            blocked_by_recovery = bool(in_recovery)

            # Retention proxy: nsucc_mean_now / max(nsucc_mean over last N
            # samples). Defaults to 1.0 when there is no signal yet so a
            # cold-start window is not penalised.
            ret_window = window.last_n(self.curriculum_fast_retention_window)
            if ret_window:
                nsucc_mean_used = float(ret_window[-1].nsucc_mean)
                peak_in_window = max(float(s.nsucc_mean) for s in ret_window)
                if peak_in_window > 1e-9:
                    retention_used = nsucc_mean_used / peak_in_window
                else:
                    retention_used = 1.0
            else:
                nsucc_mean_used = 0.0
                retention_used = 1.0

            if shrink_token:
                if in_recovery:
                    adaptive_shrink_factor = float(self.curriculum_shrink_factor)
                    adaptive_consolidation_frames = int(
                        self.curriculum_fast_recovery_consolidation_frames
                    )
                    shrink_reason = "recovery_legacy"
                else:
                    allow_strong = True
                    if self.curriculum_strong_shrink_global_retention_gate:
                        allow_strong = (
                            global_retention
                            >= self.curriculum_strong_shrink_min_global_retention
                            and stable_ticks
                            >= self.curriculum_strong_shrink_min_stable_ticks
                        )
                        if not allow_strong:
                            curriculum_strong_shrink_allowed = False
                            curriculum_strong_shrink_block_reason = (
                                "low_global_retention_or_unstable"
                            )
                    if (
                        danger_zone
                        and self.curriculum_danger_disable_strong_shrink
                    ):
                        allow_strong = False
                        curriculum_strong_shrink_allowed = False
                        if not curriculum_strong_shrink_block_reason:
                            curriculum_strong_shrink_block_reason = "danger_zone"
                    if (
                        nsucc_mean_used >= self.curriculum_fast_strong_mean
                        and retention_used >= self.curriculum_fast_retention_threshold
                        and allow_strong
                    ):
                        adaptive_shrink_factor = float(
                            self.curriculum_fast_shrink_strong
                        )
                        adaptive_consolidation_frames = int(
                            self.curriculum_fast_consolidation_frames
                        )
                        shrink_reason = "strong"
                    elif nsucc_mean_used >= self.curriculum_fast_mid_mean:
                        adaptive_shrink_factor = float(
                            self.curriculum_fast_shrink_mid
                        )
                        adaptive_consolidation_frames = int(
                            self.curriculum_fast_consolidation_frames
                        )
                        shrink_reason = "mid"
                    elif nsucc_mean_used >= self.curriculum_fast_low_mean:
                        adaptive_shrink_factor = float(
                            self.curriculum_fast_shrink_low
                        )
                        adaptive_consolidation_frames = int(
                            self.curriculum_fast_consolidation_frames
                        )
                        shrink_reason = "low"
                    else:
                        # nsucc < low_mean -> block this shrink. Roll back the
                        # single-shot bookkeeping so a future tick (with higher
                        # nsucc_mean) can still shrink in this PBT iteration.
                        shrink_token = False
                        allow_shrink = False
                        shrink_blocked_reason = "adaptive_fast_low_nsucc_mean"
                        blocked_by_adaptive_fast_low_nsucc = True
                        if (
                            self.pbt_single_shrink_per_window
                            and self._shrink_issued_this_window.get(policy_idx, False)
                        ):
                            self._shrink_issued_this_window[policy_idx] = False
                            cur = int(
                                self._shrinks_since_rollback.get(policy_idx, 0)
                            )
                            if cur > 0:
                                self._shrinks_since_rollback[policy_idx] = cur - 1
                        shrinks_in_window = 0
                        shrink_reason = "blocked_low_nsucc"
                if (
                    shrink_token
                    and danger_zone
                    and self.curriculum_danger_zone_enable
                    and adaptive_consolidation_frames > 0
                ):
                    curriculum_danger_interval_mult = float(
                        self.curriculum_danger_shrink_interval_mult
                    )
                    adaptive_consolidation_frames = int(
                        adaptive_consolidation_frames
                        * curriculum_danger_interval_mult
                    )
            else:
                # Token already cleared by an upstream gate; reflect why for
                # logging consumers.
                if blocked_by_recovery and not (
                    blocked_by_collapse
                    or blocked_by_consolidation
                    or blocked_by_mastery_gate
                ):
                    shrink_reason = "blocked_recovery"
                elif blocked_by_collapse:
                    shrink_reason = "blocked_collapse"
                elif blocked_by_consolidation:
                    shrink_reason = "blocked_consolidation"
                elif blocked_by_mastery_gate:
                    shrink_reason = "blocked_mastery_gate"
                else:
                    shrink_reason = "blocked_other"
        else:
            shrink_reason = "normal_mode"

        # 2cm bridge: below the bridge entry no multiplicative factor may
        # choose the next tolerance.  A level needs local evidence and time
        # before moving to exactly one stricter rung.
        bridge_active = bool(
            self.bridge_curriculum_enable
            and cur_tol <= self.bridge_curriculum_start_tol + 1e-9
            and cur_tol > self.bridge_curriculum_target_tol + 1e-9
        )
        bridge_entry = self._bridge_entry_frame.get(policy_idx, -1)
        level_entry = self._bridge_level_entry_frame.get(policy_idx, -1)
        if bridge_active and bridge_entry < 0:
            bridge_entry = level_entry = int(env_frames)
            self._bridge_entry_frame[policy_idx] = bridge_entry
            self._bridge_level_entry_frame[policy_idx] = level_entry
        probation_entry = self._bridge_probation_entry_frame.get(policy_idx, -1)
        if (
            bridge_active and self.bridge_probation_enable
            and cur_tol <= self.bridge_probation_entry_tol + 1e-9
            and probation_entry < 0
        ):
            probation_entry = int(env_frames)
            self._bridge_probation_entry_frame[policy_idx] = probation_entry
        probation_active = bool(
            probation_entry >= 0
            and int(env_frames) - probation_entry < self.bridge_probation_frames
        )
        if (
            probation_entry >= 0
            and not probation_active
            and policy_idx not in self._bridge_probation_exit_cleared
        ):
            # Probation ended cleanly: drop ordinary streak so post-probation
            # ticks cannot fire on suppressed legacy collapses.
            self._repeated_collapse_count[policy_idx] = 0
            repeated_collapse_count = 0
            self._bridge_probation_exit_cleared.add(policy_idx)
        elif probation_active and policy_idx in self._bridge_probation_exit_cleared:
            self._bridge_probation_exit_cleared.discard(policy_idx)
        bridge_next_tol = 0.0
        bridge_block_reason = ""
        if bridge_active and shrink_token:
            level_frames = max(0, int(env_frames) - max(level_entry, 0))
            if probation_active and self.bridge_probation_block_further_shrink:
                shrink_token = allow_shrink = False
                bridge_block_reason = "probation_active"
            elif not bool(same_tol["valid"]):
                shrink_token = allow_shrink = False
                bridge_block_reason = "same_tol_retention_warmup"
            elif float(same_tol["retention"]) < self.bridge_min_same_tol_retention:
                shrink_token = allow_shrink = False
                bridge_block_reason = "same_tol_retention"
            elif int(same_tol["stable_ticks"]) < self.bridge_min_stable_ticks:
                shrink_token = allow_shrink = False
                bridge_block_reason = "same_tol_stability"
            elif level_frames < self.bridge_min_frames_per_level:
                shrink_token = allow_shrink = False
                bridge_block_reason = "min_frames_per_level"
            else:
                bridge_next_tol = next_bridge_tolerance(
                    cur_tol, ladder=self.bridge_tolerance_ladder,
                    mode=self.bridge_curriculum_mode,
                )
                adaptive_consolidation_frames = max(
                    adaptive_consolidation_frames,
                    self.bridge_post_shrink_consolidation_frames,
                )
                # Do NOT stamp _bridge_level_entry_frame here.  Safety / WM /
                # pre-floor gates may still clear shrink_token later; the
                # level timer advances only after tolerance actually changes.
        tol_stable = window.tolerance_stable_for_intervals(1)
        last_tol = self._last_mastered_tolerance.get(policy_idx, 0.075)
        if gate_pass and window.samples:
            last_tol = window.samples[-1].success_tolerance
            self._last_mastered_tolerance[policy_idx] = last_tol
        if (
            global_nsucc_mean >= self.curriculum_last_mastered_min_nsucc_mean
            and global_retention >= self.curriculum_last_mastered_min_global_retention
            and stable_ticks >= self.curriculum_last_mastered_stable_ticks
            and not collapse_freeze_active
            and not hard_collapse_detected
            and window.samples
        ):
            safe_tol = float(window.samples[-1].success_tolerance)
            self._last_mastered_safe_tolerance[policy_idx] = safe_tol
            self._last_mastered_safe_frame[policy_idx] = int(env_frames)
            self._last_mastered_safe_nsucc_mean[policy_idx] = float(global_nsucc_mean)
            self._last_mastered_safe_global_retention[policy_idx] = float(
                global_retention
            )
            curriculum_last_mastered_tolerance = safe_tol
            curriculum_last_mastered_frame = int(env_frames)
            curriculum_last_mastered_nsucc_mean = float(global_nsucc_mean)
            curriculum_last_mastered_global_retention = float(global_retention)
            if not collapse_detected:
                self._repeated_collapse_count[policy_idx] = 0
                repeated_collapse_count = 0

        if self.pbt_freeze_mutation_after_stable_frames > 0:
            stable_ok, _ = is_stable_source_window(
                window,
                min_mean=self.stable_source_min_mean,
                min_max=self.stable_source_min_max,
                max_drop_ratio=self.stable_source_max_drop_ratio,
                drop_windows=self.stable_source_drop_windows,
            )
            if stable_ok:
                self._mutation_frozen_until[policy_idx] = int(env_frames) + int(
                    self.pbt_freeze_mutation_after_stable_frames
                )
        if rollback_reason:
            self._rollback_reason[policy_idx] = rollback_reason
        else:
            rollback_reason = str(self._rollback_reason.get(policy_idx, ""))
        if (
            not rollback_reason
            and int(self._restart_count) > 0
            and int(env_frames) - int(self._frames_at_restart) < int(self.interval_steps)
        ):
            rollback_reason = "restart_reset"

        # Carry pre/post-shrink tolerance forward from the previous control
        # tick so analyzer scripts can read the most recent shrink event
        # without scanning every JSONL row. The env writes these via
        # mark_shrink_consumed() when an actual shrink happens.
        prev_pre_shrink = float(prev_control.pre_shrink_tolerance) if prev_control else 0.0
        prev_post_shrink = float(prev_control.post_shrink_tolerance) if prev_control else 0.0

        # Phase A4 P4 (2026-06-09): floor-recovery audit fields.
        # Defaults are inert ("none" / 0 / 0.0); only populated when the
        # observer-side feature is enabled. The env-side curriculum logic
        # does not branch on these fields, so writing them is safe even
        # in default mode (left at zero values).
        floor_stage_str = "none"
        first_floor_frame_val = -1
        first_hard_floor_frame_val = -1
        frames_since_floor_val = 0
        floor_recovery_slope_val = 0.0
        floor_recovery_score_val = 0.0
        floor_recovery_protect_until_val = int(
            self._floor_recovery_protect_until.get(policy_idx, 0)
        )
        if self.pbt_floor_recovery_aware:
            cur = (population or {}).get(policy_idx) or {}
            curriculum_for_stage = cur.get("curriculum") or {}
            floor_stage_str = self._floor_stage_for(curriculum_for_stage)
            self._track_first_floor_frames(
                policy_idx, env_frames, floor_stage_str
            )
            first_floor_frame_val = int(
                self._first_floor_frame.get(policy_idx, -1)
            )
            first_hard_floor_frame_val = int(
                self._first_hard_floor_frame.get(policy_idx, -1)
            )
            if first_floor_frame_val >= 0:
                frames_since_floor_val = max(
                    0, int(env_frames) - first_floor_frame_val
                )
            score_50, components, slope_pos = self._floor_recovery_score_for(
                policy_idx,
                window,
                env_frames,
                hard_collapse=hard_collapse_detected,
            )
            floor_recovery_slope_val = float(components.get("slope_raw", 0.0))
            floor_recovery_score_val = float(score_50)

        wm_cc = self._evaluate_wm_curriculum_controller(
            policy_idx, window, env_frames
        )
        self._last_wm_curriculum_audit = dict(wm_cc)
        safety = self._evaluate_curriculum_safety(
            policy_idx,
            window,
            env_frames,
            cur_tol,
            shrink_token,
            wm_cc=wm_cc,
            global_retention=global_retention,
            stable_ticks=stable_ticks,
            danger_zone=danger_zone,
            repeated_collapse_count=repeated_collapse_count,
        )
        self._last_curriculum_safety_audit = dict(safety)
        if safety.get("danger_blocked"):
            curriculum_danger_blocked_shrink = True
            curriculum_danger_block_reason = str(
                safety.get("danger_block_reason", "")
            )
        blocked_by_safety = bool(safety.get("blocked"))
        blocked_by_wm = bool(wm_cc.get("blocked_shrink"))
        if (blocked_by_safety or blocked_by_wm) and shrink_token:
            shrink_token = False
            allow_shrink = False
            if blocked_by_safety:
                shrink_blocked_reason = str(
                    safety.get("block_reason", "curriculum_safety_blocked")
                )
                shrink_reason = "curriculum_safety_blocked"
                if shrink_blocked_reason == "real_signal_risk":
                    self._append_curriculum_safety_event(
                        {
                            "event": "real_signal_block",
                            "policy_idx": int(policy_idx),
                            "env_frames": int(env_frames),
                            "global_retention": float(global_retention),
                            "drop_ratio": float(safety.get("recent_drop_ratio", 0.0)),
                        }
                    )
                elif shrink_blocked_reason.startswith("danger_zone"):
                    self._append_curriculum_safety_event(
                        {
                            "event": "danger_zone_block",
                            "policy_idx": int(policy_idx),
                            "env_frames": int(env_frames),
                            "reason": shrink_blocked_reason,
                            "stable_ticks": int(stable_ticks),
                        }
                    )
            else:
                shrink_blocked_reason = "wm_curriculum_high_risk"
                shrink_reason = "wm_curriculum_blocked"
            if (
                self.pbt_single_shrink_per_window
                and self._shrink_issued_this_window.get(policy_idx, False)
            ):
                self._shrink_issued_this_window[policy_idx] = False
                cur = int(self._shrinks_since_rollback.get(policy_idx, 0))
                if cur > 0:
                    self._shrinks_since_rollback[policy_idx] = cur - 1
            shrinks_in_window = 0
        elif (
            not curriculum_strong_shrink_allowed
            and shrink_reason == "strong"
            and curriculum_strong_shrink_block_reason
        ):
            self._append_curriculum_safety_event(
                {
                    "event": "strong_shrink_blocked",
                    "policy_idx": int(policy_idx),
                    "env_frames": int(env_frames),
                    "reason": curriculum_strong_shrink_block_reason,
                    "global_retention": float(global_retention),
                }
            )

        pre_floor_peak = self._evaluate_pre_floor_peak_guard(
            policy_idx,
            window,
            env_frames,
            cur_tol=cur_tol,
            shrink_token=shrink_token,
            recent_peak=recent_peak,
            adaptive_shrink_factor=adaptive_shrink_factor,
            stable_ticks=stable_ticks,
        )
        if pre_floor_peak.get("blocked") and shrink_token:
            shrink_token = False
            allow_shrink = False
            shrink_blocked_reason = "pre_floor_peak_retention_guard"
            shrink_reason = "pre_floor_peak_retention_guard"
            if (
                self.pbt_single_shrink_per_window
                and self._shrink_issued_this_window.get(policy_idx, False)
            ):
                self._shrink_issued_this_window[policy_idx] = False
                cur = int(self._shrinks_since_rollback.get(policy_idx, 0))
                if cur > 0:
                    self._shrinks_since_rollback[policy_idx] = cur - 1
            shrinks_in_window = 0

        late_floor_retention, forced_tolerance = self._evaluate_late_floor_retention_gate(
            policy_idx,
            window,
            env_frames,
            cur_tol=cur_tol,
            shrink_token=shrink_token,
            recent_peak=recent_peak,
            forced_tolerance=forced_tolerance,
        )
        if late_floor_retention.get("rollback_applied") and adaptive_ordinary_block:
            # Adaptive probation owns ordinary rollback; undo late-floor force.
            late_floor_retention = dict(late_floor_retention)
            late_floor_retention["rollback_applied"] = False
            forced_tolerance = 0.0
            adaptive_probation_ordinary_rollback_blocked = True
        if late_floor_retention.get("rollback_applied"):
            collapse_rollback_active = True
            curriculum_rollback_triggered = True
            curriculum_rollback_from_tol = float(cur_tol)
            curriculum_rollback_to_tol = float(forced_tolerance)
            curriculum_rollback_freeze_until = int(
                late_floor_retention.get("freeze_until", 0)
            )
            rollback_reason = str(late_floor_retention.get("reason", ""))
        if (
            late_floor_retention.get("blocked")
            and shrink_token
            and self.late_floor_retention_block_shrink
        ):
            shrink_token = False
            allow_shrink = False
            shrink_blocked_reason = "late_floor_low_retention"
            shrink_reason = "late_floor_low_retention"
            if (
                self.pbt_single_shrink_per_window
                and self._shrink_issued_this_window.get(policy_idx, False)
            ):
                self._shrink_issued_this_window[policy_idx] = False
                cur = int(self._shrinks_since_rollback.get(policy_idx, 0))
                if cur > 0:
                    self._shrinks_since_rollback[policy_idx] = cur - 1
            shrinks_in_window = 0

        # Adaptive probation shrink gate: may only clear an already-issued token.
        if adaptive_state is not None and shrink_token:
            _acfg, _aprob, _atrack = adaptive_state
            can_shrink, block_reason = _aprob.can_shrink(
                frame=int(env_frames), cfg=_acfg
            )
            adaptive_probation_can_shrink = bool(can_shrink)
            adaptive_probation_block_reason = str(block_reason)
            adaptive_probation_active = bool(_aprob.active)
            if not can_shrink:
                shrink_token = False
                allow_shrink = False
                shrink_blocked_reason = f"adaptive_{block_reason}"
                shrink_reason = f"adaptive_{block_reason}"
                if (
                    self.pbt_single_shrink_per_window
                    and self._shrink_issued_this_window.get(policy_idx, False)
                ):
                    self._shrink_issued_this_window[policy_idx] = False
                    current = int(self._shrinks_since_rollback.get(policy_idx, 0))
                    if current > 0:
                        self._shrinks_since_rollback[policy_idx] = current - 1
                shrinks_in_window = 0
        elif adaptive_state is not None:
            _acfg, _aprob, _atrack = adaptive_state
            can_shrink, block_reason = _aprob.can_shrink(
                frame=int(env_frames), cfg=_acfg
            )
            adaptive_probation_can_shrink = bool(can_shrink)
            adaptive_probation_block_reason = str(block_reason)
            adaptive_probation_active = bool(_aprob.active)

        self._adaptive_last_audit[policy_idx] = {
            "adaptive_precision_active": bool(adaptive_precision_active),
            "adaptive_curriculum_progress": float(adaptive_curriculum_progress),
            "adaptive_probation_active": bool(adaptive_probation_active),
            "adaptive_probation_entry_tolerance": float(adaptive_probation_entry_tolerance),
            "adaptive_probation_entry_frame": int(adaptive_probation_entry_frame),
            "adaptive_probation_can_shrink": bool(adaptive_probation_can_shrink),
            "adaptive_probation_block_reason": str(adaptive_probation_block_reason),
            "adaptive_probation_ordinary_rollback_blocked": bool(
                adaptive_probation_ordinary_rollback_blocked
            ),
            "adaptive_probation_hard_collapse": bool(adaptive_probation_hard_collapse),
            "adaptive_probation_hard_collapse_streak": int(
                adaptive_probation_hard_collapse_streak
            ),
            "adaptive_probation_rollback_triggered": bool(
                adaptive_probation_rollback_triggered
            ),
            "adaptive_probation_rollback_from_tol": float(
                adaptive_probation_rollback_from_tol
            ),
            "adaptive_probation_rollback_to_tol": float(
                adaptive_probation_rollback_to_tol
            ),
            "adaptive_probation_hard_collapse_trigger_source": str(
                adaptive_probation_hard_collapse_trigger_source
            ),
            "adaptive_probation_hard_collapse_independent": bool(
                adaptive_probation_hard_collapse_independent
            ),
            "adaptive_probation_legacy_collapse_also_active": bool(
                adaptive_probation_legacy_collapse_also_active
            ),
            "adaptive_probation_freeze_only_streak": int(
                adaptive_probation_freeze_only_streak
            ),
            "adaptive_rollback_target_source": str(adaptive_rollback_target_source),
            "adaptive_rollback_log_gap": float(adaptive_rollback_log_gap),
        }

        wm_payload = self._load_policy_wm_imagined_eval(policy_idx) or {}
        wm_imagined_valid = bool(wm_payload.get("imagined_valid", False))
        wm_imagined_uncertainty = float(wm_payload.get("uncertainty", 1.0))
        wm_imagined_success_prob = float(wm_payload.get("wm_success_prob", 0.0))
        wm_imagined_drop_risk = float(wm_payload.get("wm_drop_risk", 0.0))
        wm_imagined_recovery_prob = float(wm_payload.get("wm_recovery_prob", 0.0))
        wm_imagined_invalid_reason = str(
            wm_payload.get("invalid_reason", "")
        )
        wm_imagined_shadow_block_decision = bool(
            wm_payload.get("shadow_block_decision", False)
        )
        wm_imagined_shadow_block_reason = str(
            wm_payload.get("shadow_block_reason", "")
        )
        if (
            wm_imagined_shadow_block_decision
            and not safety.get("real_risk")
        ):
            wm_imagined_shadow_block_reason = (
                wm_imagined_shadow_block_reason or "shadow_only"
            )

        return CurriculumControlState(
            curriculum_mode=self.curriculum_mode,
            allow_shrink=allow_shrink,
            shrink_token=shrink_token,
            shrink_blocked_reason=shrink_blocked_reason,
            mastery_gate_pass=gate_pass,
            mastery_consecutive_count=consec_count,
            last_mastered_tolerance=last_tol,
            collapse_freeze_active=collapse_freeze_active,
            collapse_freeze_until_env_frames=freeze_until,
            nsucc_mean_drop_ratio=drop_ratio,
            recent_peak_nsucc_mean=recent_peak,
            post_shrink_consolidation_until_restart_step=consolidation_until,
            post_shrink_freeze_env_frames=self.post_shrink_consolidation_env_frames,
            tolerance_stable_intervals=1 if tol_stable else 0,
            shrinks_in_window=shrinks_in_window,
            forced_success_tolerance=forced_tolerance,
            collapse_rollback_active=collapse_rollback_active,
            early_collapse_triggered=early_collapse_detected,
            hard_collapse_triggered=hard_collapse_detected,
            rollback_and_freeze_same_tick=rollback_and_freeze_same_tick,
            rollback_cooldown_active=rollback_cooldown_active,
            rollback_count=rollback_count,
            last_rollback_frame=last_rollback_frame,
            rollback_reason=rollback_reason,
            # Phase A4 P2 (2026-06-08): adaptive-fast bookkeeping.
            speed_mode=self.curriculum_speed_mode,
            adaptive_shrink_factor=adaptive_shrink_factor,
            adaptive_consolidation_frames=adaptive_consolidation_frames,
            shrink_reason=shrink_reason,
            nsucc_mean_used_for_shrink=nsucc_mean_used,
            retention_used_for_shrink=retention_used,
            blocked_by_recovery=blocked_by_recovery,
            blocked_by_collapse=blocked_by_collapse,
            blocked_by_consolidation=blocked_by_consolidation,
            blocked_by_mastery_gate=blocked_by_mastery_gate,
            blocked_by_adaptive_fast_low_nsucc=blocked_by_adaptive_fast_low_nsucc,
            pre_shrink_tolerance=prev_pre_shrink,
            post_shrink_tolerance=prev_post_shrink,
            # Phase A4 P4 (2026-06-09): floor-recovery audit fields.
            floor_stage=floor_stage_str,
            first_floor_frame=first_floor_frame_val,
            first_hard_floor_frame=first_hard_floor_frame_val,
            frames_since_floor=frames_since_floor_val,
            floor_recovery_slope=floor_recovery_slope_val,
            floor_recovery_score=floor_recovery_score_val,
            floor_recovery_score_real=floor_recovery_score_val,
            floor_recovery_score_wm=0.0,
            floor_recovery_score_norm=floor_recovery_score_val / 50.0
            if floor_recovery_score_val
            else 0.0,
            floor_recovery_protect_until=floor_recovery_protect_until_val,
            wm_curriculum_controller_active=bool(wm_cc.get("active", False)),
            wm_curriculum_eval_valid=bool(wm_cc.get("eval_valid", False)),
            wm_curriculum_drop_risk=float(wm_cc.get("drop_risk", 0.0)),
            wm_curriculum_success_prob=float(wm_cc.get("success_prob", 0.0)),
            wm_curriculum_recovery_prob=float(wm_cc.get("recovery_prob", 0.0)),
            wm_curriculum_uncertainty=float(wm_cc.get("uncertainty", 1.0)),
            wm_curriculum_blocked_shrink=bool(wm_cc.get("blocked_shrink", False)),
            wm_curriculum_freeze_until=int(wm_cc.get("freeze_until", 0)),
            wm_curriculum_fallback_reason_id=int(
                wm_cc.get("fallback_reason_id", 0)
            ),
            wm_curriculum_tol_bin=str(wm_cc.get("tol_bin", "le_004")),
            wm_curriculum_blocked_shrink_tol_gt_006=float(
                wm_cc.get("blocked_shrink_tol_gt_006", 0.0)
            ),
            wm_curriculum_blocked_shrink_tol_005_006=float(
                wm_cc.get("blocked_shrink_tol_005_006", 0.0)
            ),
            wm_curriculum_blocked_shrink_tol_004_005=float(
                wm_cc.get("blocked_shrink_tol_004_005", 0.0)
            ),
            wm_curriculum_blocked_shrink_tol_le_004=float(
                wm_cc.get("blocked_shrink_tol_le_004", 0.0)
            ),
            curriculum_safety_active=bool(safety.get("active", False)),
            curriculum_safety_real_fallback_used=bool(
                safety.get("real_fallback_used", False)
            ),
            curriculum_safety_blocked_shrink=bool(safety.get("blocked", False)),
            curriculum_safety_block_reason=str(safety.get("block_reason", "")),
            curriculum_safety_global_retention=float(
                safety.get("global_retention", global_retention)
            ),
            curriculum_safety_recent_drop_ratio=float(
                safety.get("recent_drop_ratio", 0.0)
            ),
            curriculum_safety_recent_slope=float(safety.get("recent_slope", 0.0)),
            curriculum_safety_stable_ticks=int(safety.get("stable_ticks", stable_ticks)),
            curriculum_safety_repeated_collapse_count=int(repeated_collapse_count),
            curriculum_safety_danger_zone=bool(danger_zone),
            curriculum_safety_imagined_valid=bool(safety.get("imagined_valid", False)),
            curriculum_safety_wm_risk=bool(safety.get("wm_risk", False)),
            curriculum_safety_real_risk=bool(safety.get("real_risk", False)),
            curriculum_global_retention=float(global_retention),
            curriculum_recent_peak_nsucc_mean_long=float(long_peak),
            curriculum_strong_shrink_allowed=bool(curriculum_strong_shrink_allowed),
            curriculum_strong_shrink_block_reason=str(
                curriculum_strong_shrink_block_reason
            ),
            curriculum_strong_shrink_global_gate_active=bool(
                curriculum_strong_shrink_global_gate_active
            ),
            curriculum_danger_zone=bool(danger_zone),
            curriculum_danger_tol=float(self.curriculum_danger_tol),
            curriculum_danger_blocked_shrink=bool(curriculum_danger_blocked_shrink),
            curriculum_danger_block_reason=str(curriculum_danger_block_reason),
            curriculum_danger_stable_ticks=int(stable_ticks),
            curriculum_danger_interval_mult=float(curriculum_danger_interval_mult),
            curriculum_last_mastered_tolerance=float(
                curriculum_last_mastered_tolerance
            ),
            curriculum_last_mastered_frame=int(curriculum_last_mastered_frame),
            curriculum_last_mastered_nsucc_mean=float(
                curriculum_last_mastered_nsucc_mean
            ),
            curriculum_last_mastered_global_retention=float(
                curriculum_last_mastered_global_retention
            ),
            curriculum_rollback_triggered=bool(curriculum_rollback_triggered),
            curriculum_rollback_from_tol=float(curriculum_rollback_from_tol),
            curriculum_rollback_to_tol=float(curriculum_rollback_to_tol),
            curriculum_rollback_freeze_until=int(curriculum_rollback_freeze_until),
            curriculum_repeated_collapse_count=int(repeated_collapse_count),
            wm_imagined_valid=wm_imagined_valid,
            wm_imagined_uncertainty=wm_imagined_uncertainty,
            wm_imagined_success_prob=wm_imagined_success_prob,
            wm_imagined_drop_risk=wm_imagined_drop_risk,
            wm_imagined_recovery_prob=wm_imagined_recovery_prob,
            wm_imagined_invalid_reason=wm_imagined_invalid_reason,
            wm_imagined_shadow_block_decision=wm_imagined_shadow_block_decision,
            wm_imagined_shadow_block_reason=wm_imagined_shadow_block_reason,
            late_floor_retention_active=bool(late_floor_retention.get("active", False)),
            late_floor_retention_retention=float(
                late_floor_retention.get("retention", 0.0)
            ),
            late_floor_retention_recent_peak=float(
                late_floor_retention.get("recent_peak_nsucc_mean", 0.0)
            ),
            late_floor_retention_blocked_shrink=bool(
                late_floor_retention.get("blocked", False)
            ),
            late_floor_retention_freeze_until=int(
                late_floor_retention.get("freeze_until", 0)
            ),
            late_floor_retention_reason=str(late_floor_retention.get("reason", "")),
            pre_floor_peak_guard_active=bool(pre_floor_peak.get("active", False)),
            pre_floor_peak_guard_blocked=bool(pre_floor_peak.get("blocked", False)),
            pre_floor_peak_guard_retention=float(pre_floor_peak.get("retention", 0.0)),
            pre_floor_peak_guard_recent_peak=float(
                pre_floor_peak.get("recent_peak_nsucc_mean", 0.0)
            ),
            pre_floor_peak_guard_stable_ticks=int(
                pre_floor_peak.get("stable_ticks", 0)
            ),
            pre_floor_peak_guard_reason=str(pre_floor_peak.get("reason", "")),
            same_tol_bin_id=int(same_tol["bin_id"]),
            same_tol_retention=float(same_tol["retention"]),
            same_tol_retention_valid=bool(same_tol["valid"]),
            same_tol_peak_nsucc_mean=float(same_tol["peak"]),
            same_tol_sample_count=int(same_tol["samples"]),
            same_tol_stable_ticks=int(same_tol["stable_ticks"]),
            retention_reference_tolerance=float(cur_tol),
            cross_tol_peak_ignored=bool(self.same_tol_retention_enable),
            retention_tracker=(
                same_tol["tracker"].to_json()
                if self.same_tol_retention_enable else {}
            ),
            bridge_active=bridge_active,
            bridge_level=(
                bridge_level_id(cur_tol, self.bridge_tolerance_ladder)
                if self.bridge_curriculum_enable and self.bridge_tolerance_ladder
                else -1
            ),
            bridge_entry_frame=bridge_entry,
            bridge_level_entry_frame=int(
                self._bridge_level_entry_frame.get(policy_idx, level_entry)
            ),
            bridge_shrink_count=int(self._bridge_shrink_count.get(policy_idx, 0)),
            bridge_next_tolerance=bridge_next_tol,
            bridge_shrink_blocked=bool(bridge_block_reason),
            bridge_block_reason=bridge_block_reason,
            bridge_probation_active=probation_active,
            bridge_probation_entry_frame=probation_entry,
            bridge_probation_rollback_blocked=probation_ordinary_block,
            bridge_probation_hard_collapse=bool(bridge_probation_hard_collapse),
            bridge_probation_exit=bool(
                probation_entry >= 0 and not probation_active
            ),
            bridge_probation_legacy_collapse_suppressed=bool(
                bridge_probation_legacy_collapse_suppressed
            ),
            bridge_probation_hard_collapse_triggered=bool(
                bridge_probation_hard_collapse
            ),
            bridge_probation_bad_streak=int(
                self._bridge_probation_bad_streak.get(policy_idx, 0)
            ),
            bridge_probation_rollback_reason=str(bridge_probation_rollback_reason),
            bridge_probation_suppressed_legacy_collapse_count=int(
                self._bridge_probation_suppressed_legacy_collapse_count.get(
                    policy_idx, 0
                )
            ),
            bridge_pending_next_tolerance=float(bridge_next_tol),
            bridge_level_timer_pending=bool(bridge_next_tol > 1e-9),
            late_rescue_attempt_count=int(
                self._late_rescue_attempt_count.get(policy_idx, 0)
            ),
            late_rescue_selected_count=int(
                self._late_rescue_selected_count.get(policy_idx, 0)
            ),
            late_rescue_executed_count=int(
                self._late_rescue_executed_count.get(policy_idx, 0)
            ),
            late_rescue_blocked_count=int(
                self._late_rescue_blocked_count.get(policy_idx, 0)
            ),
            late_rescue_streak=int(self._late_rescue_streak.get(policy_idx, 0)),
            late_rescue_last_frame=int(self._late_rescue_frame.get(policy_idx, -1)),
            adaptive_precision_active=bool(adaptive_precision_active),
            adaptive_curriculum_progress=float(adaptive_curriculum_progress),
            adaptive_probation_active=bool(adaptive_probation_active),
            adaptive_probation_entry_tolerance=float(adaptive_probation_entry_tolerance),
            adaptive_probation_entry_frame=int(adaptive_probation_entry_frame),
            adaptive_probation_can_shrink=bool(adaptive_probation_can_shrink),
            adaptive_probation_block_reason=str(adaptive_probation_block_reason),
            adaptive_probation_ordinary_rollback_blocked=bool(
                adaptive_probation_ordinary_rollback_blocked
            ),
            adaptive_probation_hard_collapse=bool(adaptive_probation_hard_collapse),
            adaptive_probation_hard_collapse_streak=int(
                adaptive_probation_hard_collapse_streak
            ),
            adaptive_probation_hard_collapse_event_frame=(
                int(env_frames) if adaptive_probation_hard_collapse else -1
            ),
            adaptive_probation_rollback_triggered=bool(
                adaptive_probation_rollback_triggered
            ),
            adaptive_probation_rollback_from_tol=float(
                adaptive_probation_rollback_from_tol
            ),
            adaptive_probation_rollback_to_tol=float(
                adaptive_probation_rollback_to_tol
            ),
            adaptive_probation_hard_collapse_trigger_source=str(
                adaptive_probation_hard_collapse_trigger_source
            ),
            adaptive_probation_hard_collapse_independent=bool(
                adaptive_probation_hard_collapse_independent
            ),
            adaptive_probation_legacy_collapse_also_active=bool(
                adaptive_probation_legacy_collapse_also_active
            ),
            adaptive_probation_freeze_only_streak=int(
                adaptive_probation_freeze_only_streak
            ),
            adaptive_rollback_target_source=str(adaptive_rollback_target_source),
            adaptive_rollback_log_gap=float(adaptive_rollback_log_gap),
            adaptive_rescue_ready=bool(
                self._adaptive_rescue_last_reason.get(policy_idx, "")
                in (
                    "ready",
                    "retention_degradation_ready",
                    "hard_collapse_ready",
                )
            ),
            adaptive_rescue_reason=str(
                self._adaptive_rescue_last_reason.get(policy_idx, "")
            ),
            adaptive_rescue_attempt_count=int(
                self._adaptive_rescue_attempt_count.get(policy_idx, 0)
            ),
            adaptive_rescue_selected_count=int(
                self._adaptive_rescue_selected_count.get(policy_idx, 0)
            ),
            adaptive_rescue_executed_count=int(
                self._adaptive_rescue_executed.get(policy_idx, 0)
            ),
            adaptive_rescue_blocked_count=int(
                self._adaptive_rescue_blocked_count.get(policy_idx, 0)
            ),
            adaptive_rescue_source_policy=int(
                self._adaptive_rescue_last_source.get(policy_idx, -1)
            ),
            adaptive_rescue_source_rank=int(
                self._adaptive_rescue_last_source_rank.get(policy_idx, -1)
            ),
            adaptive_rescue_source_advantage_sigma=float(
                self._adaptive_rescue_last_advantage_sigma.get(policy_idx, 0.0)
            ),
            adaptive_rescue_last_frame=int(
                self._adaptive_last_rescue_frame.get(policy_idx, -1)
            ),
            adaptive_rescue_lineage_id=str(
                self._adaptive_lineage_id.get(policy_idx, "")
            ),
            adaptive_rescue_executed_for_lineage=int(
                self._adaptive_rescue_executed_for_lineage.get(policy_idx, 0)
            ),
        )

    def _log_curriculum_control_scalars(
        self,
        control: CurriculumControlState,
        window: RollingNsuccWindow,
        env_frames: int,
    ) -> None:
        if self.writer is None:
            return
        self.writer.add_scalar(
            "dexpbt/mastery_gate_pass",
            1.0 if control.mastery_gate_pass else 0.0,
            env_frames,
        )
        self.writer.add_scalar(
            "dexpbt/mastery_consecutive_count",
            float(control.mastery_consecutive_count),
            env_frames,
        )
        self.writer.add_scalar(
            "dexpbt/curriculum_shrink_blocked_reason",
            1.0 if control.shrink_blocked_reason else 0.0,
            env_frames,
        )
        self.writer.add_scalar(
            "dexpbt/last_mastered_tolerance",
            float(control.last_mastered_tolerance),
            env_frames,
        )
        self.writer.add_scalar(
            "dexpbt/collapse_detected",
            1.0 if self._collapse_detected_this_tick else 0.0,
            env_frames,
        )
        self.writer.add_scalar(
            "dexpbt/collapse_freeze_active",
            1.0 if control.collapse_freeze_active else 0.0,
            env_frames,
        )
        self.writer.add_scalar(
            "dexpbt/nsucc_mean_drop_ratio",
            float(control.nsucc_mean_drop_ratio),
            env_frames,
        )
        stable_ok, _ = is_stable_source_window(
            window,
            min_mean=self.stable_source_min_mean,
            min_max=self.stable_source_min_max,
            max_drop_ratio=self.stable_source_max_drop_ratio,
            drop_windows=self.stable_source_drop_windows,
        )
        self.writer.add_scalar(
            "dexpbt/source_stable_mean",
            float(window.rolling_mean(1)) if window.samples else float("nan"),
            env_frames,
        )
        self.writer.add_scalar(
            "dexpbt/source_stable_ok",
            1.0 if stable_ok else 0.0,
            env_frames,
        )
        self.writer.add_scalar(
            "dexpbt/shrink_token",
            1.0 if control.shrink_token else 0.0,
            env_frames,
        )
        if self.adaptive_precision_enable:
            self.writer.add_scalar(
                "adaptive_consolidation/probation_can_shrink",
                1.0 if control.adaptive_probation_can_shrink else 0.0,
                env_frames,
            )
            self.writer.add_scalar(
                "adaptive_consolidation/ordinary_rollback_blocked",
                1.0 if control.adaptive_probation_ordinary_rollback_blocked else 0.0,
                env_frames,
            )
            self.writer.add_scalar(
                "adaptive_consolidation/hard_collapse_rollback",
                1.0 if control.adaptive_probation_rollback_triggered else 0.0,
                env_frames,
            )
            self.writer.add_scalar(
                "adaptive_consolidation/rescue_blocked_count",
                float(control.adaptive_rescue_blocked_count),
                env_frames,
            )
            trig = str(
                getattr(control, "adaptive_probation_hard_collapse_trigger_source", "")
                or ""
            )
            self.writer.add_scalar(
                "adaptive_consolidation/hard_collapse_adaptive_only",
                1.0 if trig == "adaptive_only" else 0.0,
                env_frames,
            )
            self.writer.add_scalar(
                "adaptive_consolidation/hard_collapse_legacy_only",
                1.0 if trig == "legacy_only" else 0.0,
                env_frames,
            )
            self.writer.add_scalar(
                "adaptive_consolidation/hard_collapse_both",
                1.0 if trig == "adaptive_and_legacy" else 0.0,
                env_frames,
            )
            src_id = {
                "": 0,
                "none": 0,
                "previous_tolerance": 1,
                "source_tolerance": 2,
                "last_mastered_safe": 3,
                "scale_fallback": 4,
            }.get(str(getattr(control, "adaptive_rollback_target_source", "") or ""), -1)
            self.writer.add_scalar(
                "adaptive_consolidation/rollback_target_source_id",
                float(src_id),
                env_frames,
            )
            self.writer.add_scalar(
                "adaptive_consolidation/freeze_only_streak",
                float(getattr(control, "adaptive_probation_freeze_only_streak", 0) or 0),
                env_frames,
            )
            pidx = int(self.policy_idx)
            self.writer.add_scalar(
                "adaptive_consolidation/rescue_live_selected",
                float(self._adaptive_rescue_live_selected.get(pidx, 0)),
                env_frames,
            )
            self.writer.add_scalar(
                "adaptive_consolidation/rescue_archive_selected",
                float(self._adaptive_rescue_archive_selected.get(pidx, 0)),
                env_frames,
            )
            self.writer.add_scalar(
                "adaptive_consolidation/rescue_ordinary_fallback",
                float(self._adaptive_rescue_ordinary_fallback.get(pidx, 0)),
                env_frames,
            )
            self.writer.add_scalar(
                "adaptive_consolidation/rescue_burnin_blocked",
                float(self._adaptive_rescue_burnin_blocked.get(pidx, 0)),
                env_frames,
            )
            self.writer.add_scalar(
                "adaptive_consolidation/rescue_burnin_bypassed",
                float(self._adaptive_rescue_burnin_bypassed.get(pidx, 0)),
                env_frames,
            )
            self.writer.add_scalar(
                "adaptive_consolidation/rescue_emergency_bypass",
                float(self._adaptive_rescue_emergency_bypass.get(pidx, 0)),
                env_frames,
            )
        self.writer.add_scalar(
            "dexpbt/post_shrink_consolidation_active",
            1.0
            if control.post_shrink_consolidation_until_restart_step > 0
            else 0.0,
            env_frames,
        )
        self.writer.add_scalar(
            "dexpbt/shrinks_in_window",
            float(control.shrinks_in_window),
            env_frames,
        )
        self.writer.add_scalar(
            "dexpbt/collapse_rollback_active",
            1.0 if control.collapse_rollback_active else 0.0,
            env_frames,
        )
        self.writer.add_scalar(
            "dexpbt/forced_success_tolerance",
            float(control.forced_success_tolerance),
            env_frames,
        )
        self.writer.add_scalar(
            "dexpbt/early_freeze_triggered",
            1.0 if control.early_collapse_triggered else 0.0,
            env_frames,
        )
        self.writer.add_scalar(
            "dexpbt/hard_collapse_triggered",
            1.0 if control.hard_collapse_triggered else 0.0,
            env_frames,
        )
        self.writer.add_scalar(
            "bridge_probation/legacy_collapse_suppressed",
            1.0 if control.bridge_probation_legacy_collapse_suppressed else 0.0,
            env_frames,
        )
        self.writer.add_scalar(
            "bridge_probation/suppressed_legacy_collapse_count",
            float(control.bridge_probation_suppressed_legacy_collapse_count),
            env_frames,
        )
        self.writer.add_scalar(
            "bridge_probation/ordinary_repeated_count",
            float(control.curriculum_repeated_collapse_count),
            env_frames,
        )
        self.writer.add_scalar(
            "bridge_probation/bad_streak",
            float(control.bridge_probation_bad_streak),
            env_frames,
        )
        self.writer.add_scalar(
            "pbt/late_rescue_source_min_mean",
            float(self.late_degradation_rescue_source_min_mean),
            env_frames,
        )
        self.writer.add_scalar(
            "pbt/late_rescue_min_source_advantage",
            float(self.late_degradation_rescue_min_source_advantage),
            env_frames,
        )
        self.writer.add_scalar(
            "pbt/late_rescue_source_min_retention",
            float(self.late_degradation_rescue_source_min_retention),
            env_frames,
        )
        self.writer.add_scalar(
            "pbt/late_rescue_reject_collapse_source",
            1.0 if self.late_degradation_rescue_reject_collapse_source else 0.0,
            env_frames,
        )
        self.writer.add_scalar(
            "pbt/late_rescue_attempt_count",
            float(control.late_rescue_attempt_count),
            env_frames,
        )
        self.writer.add_scalar(
            "pbt/late_rescue_selected_count",
            float(control.late_rescue_selected_count),
            env_frames,
        )
        self.writer.add_scalar(
            "pbt/late_rescue_executed_count",
            float(control.late_rescue_executed_count),
            env_frames,
        )
        self.writer.add_scalar(
            "pbt/late_rescue_blocked_count",
            float(control.late_rescue_blocked_count),
            env_frames,
        )
        self.writer.add_scalar(
            "pbt/late_rescue_last_frame",
            float(control.late_rescue_last_frame),
            env_frames,
        )
        self.writer.add_scalar(
            "pbt/late_rescue_source_control_valid",
            float(self._late_rescue_last_source_control_valid),
            env_frames,
        )
        self.writer.add_scalar(
            "pbt/late_rescue_source_collapse_rejected",
            float(self._late_rescue_last_source_collapse_rejected),
            env_frames,
        )
        self.writer.add_scalar(
            "pbt/late_rescue_burnin_bypassed",
            float(self._late_rescue_last_burnin_bypassed),
            env_frames,
        )
        self.writer.add_scalar(
            "dexpbt/collapse_freeze_until_env_frames",
            float(control.collapse_freeze_until_env_frames),
            env_frames,
        )
        self.writer.add_scalar(
            "dexpbt/rollback_and_freeze_same_tick",
            1.0 if control.rollback_and_freeze_same_tick else 0.0,
            env_frames,
        )
        self.writer.add_scalar(
            "dexpbt/rollback_cooldown_active",
            1.0 if control.rollback_cooldown_active else 0.0,
            env_frames,
        )
        self.writer.add_scalar(
            "dexpbt/last_rollback_frame",
            float(control.last_rollback_frame),
            env_frames,
        )
        self.writer.add_scalar(
            "dexpbt/rollback_count",
            float(control.rollback_count),
            env_frames,
        )
        # Phase A4 P2 (2026-06-08): adaptive-fast TB scalars.
        speed_mode_id = 1.0 if control.speed_mode == "adaptive_fast" else 0.0
        self.writer.add_scalar(
            "curriculum/speed_mode", speed_mode_id, env_frames
        )
        self.writer.add_scalar(
            "curriculum/adaptive_fast_enabled",
            1.0 if self.curriculum_speed_mode == "adaptive_fast" else 0.0,
            env_frames,
        )
        self.writer.add_scalar(
            "curriculum/adaptive_shrink_factor",
            float(control.adaptive_shrink_factor),
            env_frames,
        )
        self.writer.add_scalar(
            "curriculum/post_shrink_consolidation_frames",
            float(control.adaptive_consolidation_frames),
            env_frames,
        )
        self.writer.add_scalar(
            "curriculum/retention_used_for_shrink",
            float(control.retention_used_for_shrink),
            env_frames,
        )
        self.writer.add_scalar(
            "curriculum/nsucc_mean_used_for_shrink",
            float(control.nsucc_mean_used_for_shrink),
            env_frames,
        )
        self.writer.add_scalar(
            "curriculum/blocked_by_recovery",
            1.0 if control.blocked_by_recovery else 0.0,
            env_frames,
        )
        self.writer.add_scalar(
            "curriculum/blocked_by_collapse",
            1.0 if control.blocked_by_collapse else 0.0,
            env_frames,
        )
        self.writer.add_scalar(
            "curriculum/blocked_by_consolidation",
            1.0 if control.blocked_by_consolidation else 0.0,
            env_frames,
        )
        self.writer.add_scalar(
            "curriculum/blocked_by_mastery_gate",
            1.0 if control.blocked_by_mastery_gate else 0.0,
            env_frames,
        )
        self.writer.add_scalar(
            "curriculum/blocked_by_adaptive_fast_low_nsucc",
            1.0 if control.blocked_by_adaptive_fast_low_nsucc else 0.0,
            env_frames,
        )
        self.writer.add_scalar(
            "wm_curriculum/controller_active",
            1.0 if control.wm_curriculum_controller_active else 0.0,
            env_frames,
        )
        self.writer.add_scalar(
            "wm_curriculum/eval_valid",
            1.0 if control.wm_curriculum_eval_valid else 0.0,
            env_frames,
        )
        self.writer.add_scalar(
            "wm_curriculum/drop_risk",
            float(control.wm_curriculum_drop_risk),
            env_frames,
        )
        self.writer.add_scalar(
            "wm_curriculum/success_prob",
            float(control.wm_curriculum_success_prob),
            env_frames,
        )
        self.writer.add_scalar(
            "wm_curriculum/recovery_prob",
            float(control.wm_curriculum_recovery_prob),
            env_frames,
        )
        self.writer.add_scalar(
            "wm_curriculum/uncertainty",
            float(control.wm_curriculum_uncertainty),
            env_frames,
        )
        self.writer.add_scalar(
            "wm_curriculum/blocked_shrink",
            1.0 if control.wm_curriculum_blocked_shrink else 0.0,
            env_frames,
        )
        self.writer.add_scalar(
            "wm_curriculum/freeze_until",
            float(control.wm_curriculum_freeze_until),
            env_frames,
        )
        self.writer.add_scalar(
            "wm_curriculum/fallback_reason_id",
            float(control.wm_curriculum_fallback_reason_id),
            env_frames,
        )
        self.writer.add_scalar(
            "curriculum/pre_shrink_tolerance",
            float(control.pre_shrink_tolerance),
            env_frames,
        )
        self.writer.add_scalar(
            "curriculum/post_shrink_tolerance",
            float(control.post_shrink_tolerance),
            env_frames,
        )
        # Phase A4 P3 (2026-06-09): inheritance / promising-gate scalars.
        self.writer.add_scalar(
            "curriculum/inherited_tolerance_count",
            float(self._curriculum_inherited_count),
            env_frames,
        )
        self.writer.add_scalar(
            "curriculum/reset_to_initial_tolerance_count",
            float(self._curriculum_reset_count),
            env_frames,
        )
        self.writer.add_scalar(
            "curriculum/missing_source_curriculum_metadata_count",
            float(self._curriculum_missing_metadata_count),
            env_frames,
        )
        self.writer.add_scalar(
            "curriculum/recovery_consolidation_frames_selected",
            float(control.recovery_consolidation_frames_selected),
            env_frames,
        )
        self.writer.add_scalar(
            "curriculum/source_safe_for_curriculum",
            1.0 if control.source_safe_for_curriculum else 0.0,
            env_frames,
        )
        self.writer.add_scalar(
            "curriculum/inherited_tolerance",
            float(control.inherited_tolerance),
            env_frames,
        )
        self.writer.add_scalar(
            "curriculum/target_tolerance_after_copy",
            float(control.target_tolerance_after_copy),
            env_frames,
        )
        self.writer.add_scalar(
            "pbt/curriculum_inherit_mode",
            1.0 if self.pbt_curriculum_inherit_mode == "source_safe" else 0.0,
            env_frames,
        )
        self.writer.add_scalar(
            "pbt/protect_promising_workers_enabled",
            1.0 if self.pbt_protect_promising_workers else 0.0,
            env_frames,
        )
        self.writer.add_scalar(
            "curriculum/fast_target_tolerance_800m",
            float(self.curriculum_fast_target_tolerance_800m),
            env_frames,
        )
        # Phase A4 P4 (2026-06-09): floor-recovery scalars (Part A).
        self.writer.add_scalar(
            "curriculum/floor_stage_hash",
            float({"none": 0, "floor": 1, "hard_floor": 2}.get(
                str(control.floor_stage), 0
            )),
            env_frames,
        )
        self.writer.add_scalar(
            "curriculum/first_floor_frame",
            float(control.first_floor_frame),
            env_frames,
        )
        self.writer.add_scalar(
            "curriculum/first_hard_floor_frame",
            float(control.first_hard_floor_frame),
            env_frames,
        )
        self.writer.add_scalar(
            "curriculum/frames_since_floor",
            float(control.frames_since_floor),
            env_frames,
        )
        self.writer.add_scalar(
            "curriculum/floor_recovery_slope",
            float(control.floor_recovery_slope),
            env_frames,
        )
        self.writer.add_scalar(
            "curriculum/floor_recovery_score",
            float(control.floor_recovery_score),
            env_frames,
        )
        self.writer.add_scalar(
            "curriculum/floor_recovery_score_real",
            float(control.floor_recovery_score_real),
            env_frames,
        )
        self.writer.add_scalar(
            "curriculum/floor_recovery_score_wm",
            float(control.floor_recovery_score_wm),
            env_frames,
        )
        self.writer.add_scalar(
            "curriculum/floor_recovery_protect_until",
            float(control.floor_recovery_protect_until),
            env_frames,
        )
        self.writer.add_scalar(
            "pbt/floor_recovery_aware_enabled",
            1.0 if self.pbt_floor_recovery_aware else 0.0,
            env_frames,
        )
        self.writer.add_scalar(
            "curriculum/late_floor_retention_active",
            1.0 if control.late_floor_retention_active else 0.0,
            env_frames,
        )
        self.writer.add_scalar(
            "curriculum/late_floor_retention_ratio",
            float(control.late_floor_retention_retention),
            env_frames,
        )
        self.writer.add_scalar(
            "curriculum/late_floor_retention_peak",
            float(control.late_floor_retention_recent_peak),
            env_frames,
        )
        self.writer.add_scalar(
            "curriculum/late_floor_retention_blocked",
            1.0 if control.late_floor_retention_blocked_shrink else 0.0,
            env_frames,
        )
        self.writer.add_scalar(
            "curriculum/late_floor_retention_freeze_until",
            float(control.late_floor_retention_freeze_until),
            env_frames,
        )
        self.writer.add_scalar(
            "curriculum/pre_floor_peak_guard_active",
            1.0 if control.pre_floor_peak_guard_active else 0.0,
            env_frames,
        )
        self.writer.add_scalar(
            "curriculum/pre_floor_peak_guard_blocked",
            1.0 if control.pre_floor_peak_guard_blocked else 0.0,
            env_frames,
        )
        self.writer.add_scalar(
            "curriculum/pre_floor_peak_guard_retention",
            float(control.pre_floor_peak_guard_retention),
            env_frames,
        )
        self.writer.add_scalar(
            "curriculum/pre_floor_peak_guard_peak",
            float(control.pre_floor_peak_guard_recent_peak),
            env_frames,
        )
        self.writer.add_scalar(
            "curriculum/pre_floor_peak_guard_stable_ticks",
            float(control.pre_floor_peak_guard_stable_ticks),
            env_frames,
        )

    @staticmethod
    def _validate_ranking_metric(mode: str) -> str:
        """Validate PBT ranking metric; raise on unknown to fail fast."""
        if mode not in PBT_RANKING_METRICS:
            raise ValueError(
                f"Unknown pbt_ranking_metric={mode!r}; "
                f"expected one of {PBT_RANKING_METRICS}"
            )
        return str(mode)

    # ------------------------------------------------------------------ #
    # Patch 2 (rev 2, 2026-05-30): anti-churn / N_adapt helpers.
    #
    # Two independent gates are computed per worker:
    #
    #   eligible_as_source       = (fsr >= min_frames_as_source_after_restart)
    #                              AND (nsucc_mean > 0)
    #   eligible_for_elimination = (fsr >= min_frames_between_restarts)
    #
    # `fsr` (frames_since_restart) must be in *env-frame* units to match
    # PBT interval / anti-churn thresholds (e.g. 50M). We resolve through:
    #   1. frame - frames_at_restart           -- observer-tracked env frames
    #   2. curriculum["frame_since_restart"]   -- env step counter (legacy)
    #   3. 0                                   -- safest assumption
    #
    # NOTE: the env increments curriculum["frame_since_restart"] once per
    # simulation step, NOT once per env-frame (algo.frame ≈ step * num_envs).
    # Preferring (1) avoids a num_envs-scaled unit mismatch that otherwise
    # blocks every PBT restart forever when num_envs >> 1.
    # ------------------------------------------------------------------ #

    @staticmethod
    def _frames_since_restart(state: dict[str, Any] | None) -> int:
        if not state:
            return 0
        frame = state.get("frame")
        far = state.get("frames_at_restart")
        if isinstance(frame, (int, float)) and not math.isnan(float(frame)):
            if isinstance(far, (int, float)) and not math.isnan(float(far)):
                return max(0, int(frame) - int(far))
        cur = state.get("curriculum") or {}
        fsr = cur.get("frame_since_restart")
        if isinstance(fsr, (int, float)) and not math.isnan(float(fsr)):
            return int(fsr)
        return 0

    @staticmethod
    def _state_nsucc_mean(state: dict[str, Any] | None) -> float:
        if not state:
            return 0.0
        if "nsucc_mean" in state and state["nsucc_mean"] is not None:
            val = float(state["nsucc_mean"])
        else:
            cur = state.get("curriculum") or {}
            val = float(cur.get("mean_prev_episode_successes", 0.0) or 0.0)
        if math.isnan(val):
            return 0.0
        return val

    @staticmethod
    def _state_nsucc_max(state: dict[str, Any] | None) -> float:
        """Patch A (rev 3): mirror of `_state_nsucc_mean` for nsucc_max."""
        if not state:
            return 0.0
        if "nsucc_max" in state and state["nsucc_max"] is not None:
            val = float(state["nsucc_max"])
        else:
            cur = state.get("curriculum") or {}
            val = float(cur.get("max_prev_episode_successes", 0.0) or 0.0)
        if math.isnan(val):
            return 0.0
        return val

    def _eligible_as_source_with_reason(
        self, state: dict[str, Any] | None, env_frames: int
    ) -> tuple[bool, str]:
        """Source-cooldown + nsucc gate. Returns (eligible, reason).

        Patch A (rev 3, 2026-05-30): a worker is eligible as a PBT top
        source only if ALL of the following hold:

          1. `frames_since_restart >= min_frames_as_source_after_restart`
             (anti-churn: source must have stabilised since last restart)
          2. `nsucc_mean >= min_nsucc_mean_as_source` (default 1.0)
          3. `nsucc_max  >= min_nsucc_max_as_source`  (default 5)

        Clauses (2) and (3) replace the previous `nsucc_mean > 0` gate,
        which let "low-nsucc + high-r_meta" leaders (e.g. nsucc_mean=0.17
        with r_meta_raw=0.94) keep their top status -- the dominant
        collapse vector observed in regrasping_cbwm_20260530_160513.

        The `reason` string is exposed in the training_curve and PBT logs
        so we can audit *why* a candidate was filtered out.
        """
        if state is None:
            return False, "no_state"
        fsr = self._frames_since_restart(state)
        if fsr < self.min_frames_as_source_after_restart:
            return False, (
                f"frames_since_restart<{self.min_frames_as_source_after_restart}"
            )
        nsucc_mean = self._state_nsucc_mean(state)
        if nsucc_mean < self.min_nsucc_mean_as_source:
            return False, (
                f"nsucc_mean<{self.min_nsucc_mean_as_source}"
            )
        nsucc_max = self._state_nsucc_max(state)
        if nsucc_max < self.min_nsucc_max_as_source:
            return False, (
                f"nsucc_max<{self.min_nsucc_max_as_source}"
            )
        return True, "ok"

    def _is_eligible_as_source(
        self, state: dict[str, Any] | None, env_frames: int
    ) -> bool:
        """Thin bool-only wrapper around `_eligible_as_source_with_reason`.

        Kept for call-sites that do not need the reason string.
        """
        ok, _ = self._eligible_as_source_with_reason(state, env_frames)
        return ok

    # ------------------------------------------------------------------ #
    # Patch G (2026-05-31): relaxed live-source gate + archive recovery.
    # The cascade order is L1 (strict) -> L2 (relaxed) -> L3 (archive)
    # -> L4 (skip). Each tier carries its own (eligible, reason) decision
    # so all rejections are auditable in training_curve / TB.
    # ------------------------------------------------------------------ #
    def _eligible_as_source_relaxed_with_reason(
        self, state: dict[str, Any] | None, env_frames: int
    ) -> tuple[bool, str]:
        """L2 gate: a worker who fails the strict gate may still be picked
        as a source if it has *some* success signal under loosened
        thresholds. The frame-cooldown anti-churn gate from Patch 2 still
        applies (a freshly-restarted worker is never a good source).
        """
        if not state:
            return False, "no_state"
        fsr = self._frames_since_restart(state)
        if fsr < self.min_frames_as_source_after_restart:
            return False, f"frames_since_restart<{self.min_frames_as_source_after_restart}"
        nm = self._state_nsucc_mean(state)
        if nm < self.relaxed_nsucc_mean_as_source:
            return False, f"relaxed_nsucc_mean<{self.relaxed_nsucc_mean_as_source}"
        nx = self._state_nsucc_max(state)
        if nx < self.relaxed_nsucc_max_as_source:
            return False, f"relaxed_nsucc_max<{self.relaxed_nsucc_max_as_source}"
        return True, "ok"

    def _eligible_as_stable_source_with_reason(
        self,
        state: dict[str, Any] | None,
        env_frames: int,
        policy_idx: int,
    ) -> tuple[bool, str]:
        """P1 L1: stable live source (mean/max/drop over rolling window)."""
        if not state:
            return False, "no_state"
        fsr = self._frames_since_restart(state)
        if fsr < self.min_frames_as_source_after_restart:
            return False, (
                f"frames_since_restart<{self.min_frames_as_source_after_restart}"
            )
        window = self._policy_window(policy_idx)
        return is_stable_source_window(
            window,
            min_mean=self.stable_source_min_mean,
            min_max=self.stable_source_min_max,
            max_drop_ratio=self.stable_source_max_drop_ratio,
            drop_windows=self.stable_source_drop_windows,
        )

    def _eligible_as_usable_live_with_reason(
        self,
        state: dict[str, Any] | None,
        env_frames: int,
        policy_idx: int,
    ) -> tuple[bool, str]:
        """P1 L2: usable live source (some signal, not necessarily stable)."""
        if not state:
            return False, "no_state"
        fsr = self._frames_since_restart(state)
        if fsr < self.min_frames_as_source_after_restart:
            return False, (
                f"frames_since_restart<{self.min_frames_as_source_after_restart}"
            )
        window = self._policy_window(policy_idx)
        return is_usable_live_window(
            window,
            min_mean=self.pbt_usable_live_min_mean,
            min_max=self.pbt_usable_live_min_max,
            max_drop_ratio=self.pbt_usable_live_max_drop_ratio,
        )

    def _select_best_nsucc_recovery_source(
        self, env_frames: int
    ) -> tuple[dict[str, Any] | None, str]:
        """L3: try to recover from the workspace `best_nsucc/` archive.

        Returns `(state_dict, reason)`. `state_dict` is a synthetic state
        whose `checkpoint` field points to the archived `.pth` file (so
        the rest of `_restart_process_with_new_params` works as for live
        sources). `state_dict` is None when:
          * recovery is disabled,
          * no archive index exists,
          * archived peak doesn't meet `recovery_min_nsucc_*` health gates,
          * archived `.pth` is missing on disk,
          * we are still inside the recovery cooldown.

        The cooldown is anchored to `_last_recovery_frame` (per-worker) so
        a single peak is not redundantly replayed every PBT tick.
        """
        if not self._best_nsucc_recovery_enabled():
            return None, "recovery_disabled"
        if env_frames - self._last_recovery_frame < self.recovery_cooldown_frames:
            return None, "recovery_cooldown"
        try:
            best = self._load_best_index(self.best_nsucc_workspace)
        except Exception as exc:  # pragma: no cover - defensive
            return None, f"archive_read_error:{type(exc).__name__}"
        if not best:
            return None, "archive_empty"
        nm = float(best.get("nsucc_mean", float("-inf")))
        nx = float(best.get("nsucc_max", float("-inf")))
        if math.isnan(nm) or nm < self.recovery_min_nsucc_mean:
            return None, f"archive_nsucc_mean<{self.recovery_min_nsucc_mean}"
        if math.isnan(nx) or nx < self.recovery_min_nsucc_max:
            return None, f"archive_nsucc_max<{self.recovery_min_nsucc_max}"
        # Reconstruct the `.pth` location next to `_index.json`.
        prefix = self._peak_filename("nsucc", best)
        pth = self.best_nsucc_workspace / f"{prefix}.pth"
        if not pth.exists():
            return None, "archive_pth_missing"
        synth = dict(best)
        synth["checkpoint"] = str(pth)
        # Mark provenance so callers can tell apart live vs archive.
        synth["_source_checkpoint_type"] = "archive"
        return synth, "ok"

    def _append_recovery_audit(self, record: dict[str, Any]) -> None:
        self._recovery_audit_path.parent.mkdir(parents=True, exist_ok=True)
        record = dict(record)
        record.setdefault("policy_idx", int(self.policy_idx))
        record.setdefault("env_frames", int(record.get("env_frames", 0)))
        record.setdefault("timestamp", time.time())
        with open(self._recovery_audit_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + "\n")

    def _append_curriculum_safety_event(self, record: dict[str, Any]) -> None:
        self._curriculum_safety_events_path.parent.mkdir(parents=True, exist_ok=True)
        record = dict(record)
        record.setdefault("policy_idx", int(self.policy_idx))
        record.setdefault("env_frames", int(record.get("env_frames", 0)))
        record.setdefault("timestamp", time.time())
        with open(self._curriculum_safety_events_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + "\n")

    @staticmethod
    def _count_tolerance_stable_ticks(window: RollingNsuccWindow) -> int:
        if not window.samples:
            return 0
        ref = float(window.samples[-1].success_tolerance)
        count = 0
        for sample in reversed(window.samples):
            if abs(float(sample.success_tolerance) - ref) < 1e-9:
                count += 1
            else:
                break
        return count

    def _compute_global_retention(
        self, window: RollingNsuccWindow
    ) -> tuple[float, float, float]:
        """Return (global_retention, current_nsucc_mean, long_window_peak)."""
        long_n = max(2, int(self.curriculum_strong_shrink_long_window))
        xs = window.last_n(long_n)
        if not xs:
            return 1.0, 0.0, 0.0
        cur = float(xs[-1].nsucc_mean)
        peak = max(float(s.nsucc_mean) for s in xs)
        if peak <= 1e-9:
            return 1.0, cur, peak
        return cur / peak, cur, peak

    def _same_tol_retention_snapshot(
        self, policy_idx: int, tolerance: float
    ) -> dict[str, Any]:
        """Single source of truth for all 2cm retention decisions."""
        if not self.same_tol_retention_enable:
            return {
                "bin_id": -1, "retention": 1.0, "valid": True, "peak": 0.0,
                "samples": 0, "stable_ticks": 0, "first_frame": -1, "last_frame": -1,
                "level_id": -1,
            }
        self._maybe_restore_bridge_state(policy_idx)
        tracker = self._same_tol_trackers.setdefault(
            policy_idx,
            ToleranceBinRetentionTracker(
                bins=self.same_tol_retention_bins,
                ladder=(
                    self.bridge_tolerance_ladder
                    if self.bridge_curriculum_enable else ()
                ),
            ),
        )
        if self.bridge_curriculum_enable and not tracker.ladder:
            tracker.ladder = self.bridge_tolerance_ladder
        use_levels = bool(self.bridge_curriculum_enable and tracker.ladder)
        state = tracker.state(tolerance, use_bridge_levels=use_levels)
        return {
            "bin_id": tracker.current_bin(tolerance),
            "level_id": tracker.current_level(tolerance) if use_levels else -1,
            "retention": tracker.retention(tolerance, use_bridge_levels=use_levels),
            "valid": tracker.is_valid(
                tolerance,
                min_samples=self.same_tol_retention_min_samples,
                warmup_ticks=self.same_tol_retention_warmup_ticks,
                use_bridge_levels=use_levels,
            ),
            "peak": state.peak_nsucc_mean,
            "samples": state.sample_count,
            "stable_ticks": state.stable_ticks,
            "first_frame": state.first_frame,
            "last_frame": state.last_frame,
            "tracker": tracker,
        }

    def _maybe_restore_bridge_state(self, policy_idx: int) -> None:
        """Restore tracker / probation / late-rescue state from control JSON."""
        if policy_idx in self._bridge_state_restored:
            return
        self._bridge_state_restored.add(policy_idx)
        if not (self.same_tol_retention_enable or self.bridge_curriculum_enable
                or self.late_degradation_rescue_enable):
            return
        path = self._curriculum_control_dir / f"{policy_idx:03d}.json"
        control = read_curriculum_control(path)
        if control is None:
            return
        try:
            lineage = None
            window = self._policy_nsucc_windows.get(policy_idx)
            if window and window.samples:
                lineage = getattr(window.samples[-1], "lineage_id", None)
            tracker_data = getattr(control, "retention_tracker", None) or {}
            if isinstance(tracker_data, dict) and tracker_data:
                restored = ToleranceBinRetentionTracker.from_json(tracker_data)
                if self.bridge_curriculum_enable and not restored.ladder:
                    restored.ladder = self.bridge_tolerance_ladder
                saved_lineage = restored.lineage_id
                if (
                    self.same_tol_retention_reset_on_lineage_change
                    and lineage is not None
                    and saved_lineage is not None
                    and lineage != saved_lineage
                ):
                    restored.states.clear()
                    restored.lineage_id = lineage
                    restored.last_bridge_level_id = -1
                self._same_tol_trackers[policy_idx] = restored
            if int(getattr(control, "bridge_entry_frame", -1)) >= 0:
                self._bridge_entry_frame[policy_idx] = int(control.bridge_entry_frame)
            if int(getattr(control, "bridge_level_entry_frame", -1)) >= 0:
                self._bridge_level_entry_frame[policy_idx] = int(
                    control.bridge_level_entry_frame
                )
            if int(getattr(control, "bridge_probation_entry_frame", -1)) >= 0:
                self._bridge_probation_entry_frame[policy_idx] = int(
                    control.bridge_probation_entry_frame
                )
            self._bridge_probation_bad_streak[policy_idx] = int(
                getattr(control, "bridge_probation_bad_streak", 0) or 0
            )
            self._bridge_probation_suppressed_legacy_collapse_count[policy_idx] = int(
                getattr(
                    control, "bridge_probation_suppressed_legacy_collapse_count", 0
                )
                or 0
            )
            self._bridge_shrink_count[policy_idx] = int(
                getattr(control, "bridge_shrink_count", 0) or 0
            )
            self._late_rescue_streak[policy_idx] = int(
                getattr(control, "late_rescue_streak", 0) or 0
            )
            # Prefer executed count as the quota-consuming counter.
            executed = int(getattr(control, "late_rescue_executed_count", 0) or 0)
            legacy = int(self._late_rescue_count.get(policy_idx, 0))
            self._late_rescue_executed_count[policy_idx] = max(executed, legacy)
            self._late_rescue_count[policy_idx] = self._late_rescue_executed_count[
                policy_idx
            ]
            self._late_rescue_attempt_count[policy_idx] = int(
                getattr(control, "late_rescue_attempt_count", 0) or 0
            )
            self._late_rescue_selected_count[policy_idx] = int(
                getattr(control, "late_rescue_selected_count", 0) or 0
            )
            self._late_rescue_blocked_count[policy_idx] = int(
                getattr(control, "late_rescue_blocked_count", 0) or 0
            )
            last_frame = int(getattr(control, "late_rescue_last_frame", -1) or -1)
            if last_frame >= 0:
                self._late_rescue_frame[policy_idx] = last_frame
            ref_tol = float(getattr(control, "retention_reference_tolerance", 0.0) or 0.0)
            if ref_tol > 1e-9:
                self._last_confirmed_tolerance[policy_idx] = ref_tol
        except Exception as exc:
            print(
                f"[PBT] bridge state restore skipped for policy {policy_idx}: {exc}"
            )

    def _late_degradation_rescue_ready(
        self, policy_idx: int, *, tolerance: float, nsucc_mean: float, env_frames: int
    ) -> bool:
        if not self.late_degradation_rescue_enable:
            return False
        same = self._same_tol_retention_snapshot(policy_idx, tolerance)
        executed = int(
            self._late_rescue_executed_count.get(
                policy_idx, self._late_rescue_count.get(policy_idx, 0)
            )
        )
        eligible = (
            bool(same["valid"])
            and tolerance <= self.late_degradation_rescue_tol_threshold
            and float(same["peak"]) >= self.late_degradation_rescue_peak_threshold
            and nsucc_mean <= self.late_degradation_rescue_current_mean_threshold
            and float(same["retention"]) <= self.late_degradation_rescue_retention_threshold
            and executed < self.late_degradation_rescue_max_per_policy
            and env_frames - self._late_rescue_frame.get(policy_idx, -10**18)
            >= self.late_degradation_rescue_cooldown_frames
        )
        self._late_rescue_streak[policy_idx] = (
            self._late_rescue_streak.get(policy_idx, 0) + 1 if eligible else 0
        )
        return self._late_rescue_streak[policy_idx] >= self.late_degradation_rescue_patience

    def _late_rescue_source_has_collapse(
        self,
        source_policy_idx: int,
        source_state: dict[str, Any],
    ) -> tuple[bool, str]:
        """Authoritative collapse gate using curriculum_control JSON."""
        path = self._curriculum_control_dir / f"{int(source_policy_idx):03d}.json"
        if not path.is_file():
            self._late_rescue_last_source_control_valid = 0.0
            if self.late_degradation_rescue_require_source_control:
                self._late_rescue_last_source_collapse_rejected = 1.0
                self._write_pbt_copy_event({
                    "event_type": "late_degradation_rescue_source_control_missing",
                    "source_policy": int(source_policy_idx),
                    "target_policy": int(self.policy_idx),
                })
                return True, "source_control_missing"
            # Compatibility: fall back to population meta only.
            if source_has_collapse_signal(source_state):
                self._late_rescue_last_source_collapse_rejected = 1.0
                return True, "collapse_signal"
            return False, "ok_no_control"
        try:
            control = read_curriculum_control(path)
        except Exception:
            control = None
        if control is None:
            self._late_rescue_last_source_control_valid = 0.0
            if self.late_degradation_rescue_require_source_control:
                self._late_rescue_last_source_collapse_rejected = 1.0
                self._write_pbt_copy_event({
                    "event_type": "late_degradation_rescue_source_control_parse_error",
                    "source_policy": int(source_policy_idx),
                    "target_policy": int(self.policy_idx),
                })
                return True, "source_control_parse_error"
            if source_has_collapse_signal(source_state):
                self._late_rescue_last_source_collapse_rejected = 1.0
                return True, "collapse_signal"
            return False, "ok_parse_failed_compat"
        self._late_rescue_last_source_control_valid = 1.0
        collapsed, reason = control_has_collapse_signal(control)
        if collapsed:
            self._late_rescue_last_source_collapse_rejected = 1.0
            self._write_pbt_copy_event({
                "event_type": "late_degradation_rescue_source_collapse_rejected",
                "source_policy": int(source_policy_idx),
                "target_policy": int(self.policy_idx),
                "reason": reason,
            })
            return True, reason
        self._late_rescue_last_source_collapse_rejected = 0.0
        return False, "ok"

    def _persist_late_rescue_runtime_state(
        self,
        policy_idx: int,
        *,
        env_frames: int,
    ) -> None:
        """Atomically persist late-rescue quota/cooldown into curriculum_control.

        Fail-closed: missing or unparseable target control must not be replaced
        with an empty ``CurriculumControlState`` (would wipe retention / bridge /
        probation / consolidation history).
        """
        path = self._curriculum_control_dir / f"{int(policy_idx):03d}.json"
        try:
            control, raw = read_curriculum_control_required(path)
        except LateRescueControlMissingError:
            self._safe_write_pbt_copy_event({
                "event_type": "late_degradation_rescue_target_control_missing",
                "policy_idx": int(policy_idx),
                "env_frames": int(env_frames),
                "path": str(path),
            })
            raise
        except LateRescueControlParseError:
            self._safe_write_pbt_copy_event({
                "event_type": "late_degradation_rescue_target_control_parse_error",
                "policy_idx": int(policy_idx),
                "env_frames": int(env_frames),
                "path": str(path),
            })
            raise
        control.late_rescue_attempt_count = int(
            self._late_rescue_attempt_count.get(policy_idx, 0)
        )
        control.late_rescue_selected_count = int(
            self._late_rescue_selected_count.get(policy_idx, 0)
        )
        control.late_rescue_executed_count = int(
            self._late_rescue_executed_count.get(
                policy_idx, self._late_rescue_count.get(policy_idx, 0)
            )
        )
        control.late_rescue_blocked_count = int(
            self._late_rescue_blocked_count.get(policy_idx, 0)
        )
        control.late_rescue_streak = int(self._late_rescue_streak.get(policy_idx, 0))
        control.late_rescue_last_frame = int(
            self._late_rescue_frame.get(policy_idx, -1)
        )
        # Preserve unknown JSON fields; only overwrite late-rescue keys.
        raw.update(control.to_json())
        write_curriculum_control_dict(path, raw)
        self._safe_write_pbt_copy_event({
            "event_type": "late_degradation_rescue_state_persisted",
            "policy_idx": int(policy_idx),
            "env_frames": int(env_frames),
            "late_rescue_attempt_count": int(control.late_rescue_attempt_count),
            "late_rescue_selected_count": int(control.late_rescue_selected_count),
            "late_rescue_executed_count": int(control.late_rescue_executed_count),
            "late_rescue_blocked_count": int(control.late_rescue_blocked_count),
            "late_rescue_streak": int(control.late_rescue_streak),
            "late_rescue_last_frame": int(control.late_rescue_last_frame),
            "path": str(path),
        })

    def _snapshot_restart_runtime_state(self) -> dict[str, Any]:
        pidx = int(self.policy_idx)
        last_rollback = getattr(self, "_last_rollback_frame", {}) or {}
        shrinks = getattr(self, "_shrinks_since_rollback", {}) or {}
        rollback_reason = getattr(self, "_rollback_reason", {}) or {}
        return {
            "last_mutation_frame": int(getattr(self, "last_mutation_frame", -10**18)),
            "restart_count": int(getattr(self, "_restart_count", 0)),
            "frames_at_restart": int(getattr(self, "_frames_at_restart", -10**18)),
            "last_rollback_frame": int(last_rollback.get(pidx, -10**18)),
            "shrinks_since_rollback": int(shrinks.get(pidx, 0)),
            "rollback_reason": str(rollback_reason.get(pidx, "")),
            "lineage_id": str(getattr(self, "_lineage_id", "")),
            "lineage_start_pf": int(getattr(self, "_lineage_start_pf", 0)),
            "weight_age_frame": int(getattr(self, "_weight_age_frame", 0)),
            "recovery_restart_count": int(getattr(self, "_recovery_restart_count", 0)),
            "last_recovery_frame": int(getattr(self, "_last_recovery_frame", -10**18)),
            "last_restart_type": int(getattr(self, "_last_restart_type", 0)),
            "cross_restart_count": int(getattr(self, "_cross_restart_count", 0)),
            "self_restart_count": int(getattr(self, "_self_restart_count", 0)),
            "last_self_restart_frame": int(
                getattr(self, "_last_self_restart_frame", -10**18)
            ),
            "wm_strategy_profile_generation": int(
                getattr(self, "_wm_strategy_profile_generation", 0)
            ),
            "wm_strategy_profile_parent": int(
                getattr(self, "_wm_strategy_profile_parent", -1)
            ),
            "wm_strategy_profile_copied_from_source": bool(
                getattr(self, "_wm_strategy_profile_copied_from_source", False)
            ),
            "last_profile_copy_frame": int(
                getattr(self, "_last_profile_copy_frame", -10**18)
            ),
            "late_rescue_executed_count": int(
                getattr(self, "_late_rescue_executed_count", {}).get(pidx, 0)
            ),
            "late_rescue_frame": int(
                getattr(self, "_late_rescue_frame", {}).get(pidx, -10**18)
            ),
            "late_rescue_count": int(
                getattr(self, "_late_rescue_count", {}).get(pidx, 0)
            ),
            "late_rescue_selected_count": int(
                getattr(self, "_late_rescue_selected_count", {}).get(pidx, 0)
            ),
            "adaptive_degradation_streak": int(
                getattr(self, "_adaptive_degradation_streak", {}).get(pidx, 0)
            ),
            "adaptive_last_rescue_frame": int(
                getattr(self, "_adaptive_last_rescue_frame", {}).get(pidx, -(10**18))
            ),
            "adaptive_rescue_executed": int(
                getattr(self, "_adaptive_rescue_executed", {}).get(pidx, 0)
            ),
            "adaptive_rescue_executed_for_lineage": int(
                getattr(self, "_adaptive_rescue_executed_for_lineage", {}).get(pidx, 0)
            ),
            "adaptive_lineage_id": str(
                getattr(self, "_adaptive_lineage_id", {}).get(pidx, "")
            ),
            "adaptive_rescue_attempt_count": int(
                getattr(self, "_adaptive_rescue_attempt_count", {}).get(pidx, 0)
            ),
            "adaptive_rescue_selected_count": int(
                getattr(self, "_adaptive_rescue_selected_count", {}).get(pidx, 0)
            ),
            "adaptive_rescue_blocked_count": int(
                getattr(self, "_adaptive_rescue_blocked_count", {}).get(pidx, 0)
            ),
            "adaptive_rescue_last_reason": str(
                getattr(self, "_adaptive_rescue_last_reason", {}).get(pidx, "")
            ),
            "adaptive_rescue_last_source": int(
                getattr(self, "_adaptive_rescue_last_source", {}).get(pidx, -1)
            ),
            "adaptive_rescue_last_source_rank": int(
                getattr(self, "_adaptive_rescue_last_source_rank", {}).get(pidx, -1)
            ),
            "adaptive_rescue_last_advantage_sigma": float(
                getattr(self, "_adaptive_rescue_last_advantage_sigma", {}).get(pidx, 0.0)
            ),
            "adaptive_freeze_only_streak": int(
                getattr(self, "_adaptive_freeze_only_streak", {}).get(pidx, 0)
            ),
            "adaptive_rollback_freeze_only_count": int(
                getattr(self, "_adaptive_rollback_freeze_only_count", {}).get(pidx, 0)
            ),
            "adaptive_hard_collapse_trigger_source": str(
                getattr(self, "_adaptive_hard_collapse_trigger_source", {}).get(pidx, "")
            ),
            "adaptive_rescue_live_selected": int(
                getattr(self, "_adaptive_rescue_live_selected", {}).get(pidx, 0)
            ),
            "adaptive_rescue_archive_selected": int(
                getattr(self, "_adaptive_rescue_archive_selected", {}).get(pidx, 0)
            ),
            "adaptive_rescue_ordinary_fallback": int(
                getattr(self, "_adaptive_rescue_ordinary_fallback", {}).get(pidx, 0)
            ),
            "adaptive_rescue_last_source_kind": str(
                getattr(self, "_adaptive_rescue_last_source_kind", {}).get(pidx, "none")
            ),
            "adaptive_rescue_burnin_blocked": int(
                getattr(self, "_adaptive_rescue_burnin_blocked", {}).get(pidx, 0)
            ),
            "adaptive_rescue_burnin_bypassed": int(
                getattr(self, "_adaptive_rescue_burnin_bypassed", {}).get(pidx, 0)
            ),
            "adaptive_rescue_emergency_bypass": int(
                getattr(self, "_adaptive_rescue_emergency_bypass", {}).get(pidx, 0)
            ),
            "adaptive_last_checkpoint_frame": int(
                getattr(self, "_adaptive_last_checkpoint_frame", {}).get(pidx, -1)
            ),
            "adaptive_last_checkpoint_path": str(
                getattr(self, "_adaptive_last_checkpoint_path", {}).get(pidx, "")
            ),
            "adaptive_last_update_frame": int(
                getattr(self, "_adaptive_last_update_frame", {}).get(pidx, -(10**18))
            ),
        }

    def _restore_restart_runtime_state(self, snap: dict[str, Any]) -> None:
        pidx = int(self.policy_idx)
        self.last_mutation_frame = int(snap["last_mutation_frame"])
        self._restart_count = int(snap["restart_count"])
        self._frames_at_restart = int(snap["frames_at_restart"])
        if not hasattr(self, "_last_rollback_frame"):
            self._last_rollback_frame = {}
        if not hasattr(self, "_shrinks_since_rollback"):
            self._shrinks_since_rollback = {}
        if not hasattr(self, "_rollback_reason"):
            self._rollback_reason = {}
        self._last_rollback_frame[pidx] = int(snap["last_rollback_frame"])
        self._shrinks_since_rollback[pidx] = int(snap["shrinks_since_rollback"])
        self._rollback_reason[pidx] = str(snap["rollback_reason"])
        self._lineage_id = str(snap["lineage_id"])
        self._lineage_start_pf = int(snap["lineage_start_pf"])
        self._weight_age_frame = int(snap["weight_age_frame"])
        self._recovery_restart_count = int(snap["recovery_restart_count"])
        self._last_recovery_frame = int(snap["last_recovery_frame"])
        self._last_restart_type = int(snap["last_restart_type"])
        self._cross_restart_count = int(snap["cross_restart_count"])
        self._self_restart_count = int(snap["self_restart_count"])
        self._last_self_restart_frame = int(snap["last_self_restart_frame"])
        self._wm_strategy_profile_generation = int(
            snap["wm_strategy_profile_generation"]
        )
        self._wm_strategy_profile_parent = int(snap["wm_strategy_profile_parent"])
        self._wm_strategy_profile_copied_from_source = bool(
            snap["wm_strategy_profile_copied_from_source"]
        )
        self._last_profile_copy_frame = int(snap["last_profile_copy_frame"])
        if not hasattr(self, "_late_rescue_executed_count"):
            self._late_rescue_executed_count = {}
        if not hasattr(self, "_late_rescue_frame"):
            self._late_rescue_frame = {}
        if not hasattr(self, "_late_rescue_count"):
            self._late_rescue_count = {}
        if not hasattr(self, "_late_rescue_selected_count"):
            self._late_rescue_selected_count = {}
        self._late_rescue_executed_count[pidx] = int(
            snap["late_rescue_executed_count"]
        )
        self._late_rescue_frame[pidx] = int(snap["late_rescue_frame"])
        self._late_rescue_count[pidx] = int(snap["late_rescue_count"])
        self._late_rescue_selected_count[pidx] = int(
            snap["late_rescue_selected_count"]
        )
        for attr, default in (
            ("_adaptive_degradation_streak", {}),
            ("_adaptive_last_rescue_frame", {}),
            ("_adaptive_rescue_executed", {}),
            ("_adaptive_rescue_executed_for_lineage", {}),
            ("_adaptive_lineage_id", {}),
            ("_adaptive_rescue_attempt_count", {}),
            ("_adaptive_rescue_selected_count", {}),
            ("_adaptive_rescue_blocked_count", {}),
            ("_adaptive_rescue_last_reason", {}),
            ("_adaptive_rescue_last_source", {}),
            ("_adaptive_rescue_last_source_rank", {}),
            ("_adaptive_rescue_last_advantage_sigma", {}),
            ("_adaptive_freeze_only_streak", {}),
            ("_adaptive_rollback_freeze_only_count", {}),
            ("_adaptive_hard_collapse_trigger_source", {}),
            ("_adaptive_rescue_live_selected", {}),
            ("_adaptive_rescue_archive_selected", {}),
            ("_adaptive_rescue_ordinary_fallback", {}),
            ("_adaptive_rescue_last_source_kind", {}),
            ("_adaptive_rescue_burnin_blocked", {}),
            ("_adaptive_rescue_burnin_bypassed", {}),
            ("_adaptive_rescue_emergency_bypass", {}),
            ("_adaptive_last_checkpoint_frame", {}),
            ("_adaptive_last_checkpoint_path", {}),
            ("_adaptive_last_update_frame", {}),
            ("_adaptive_sidecar_meta", {}),
            ("_adaptive_peer_unavailable", {}),
            ("_adaptive_peer_reject_counts", {}),
        ):
            if not hasattr(self, attr):
                setattr(self, attr, default if isinstance(default, dict) else {})
        self._adaptive_degradation_streak[pidx] = int(
            snap.get("adaptive_degradation_streak", 0)
        )
        self._adaptive_last_rescue_frame[pidx] = int(
            snap.get("adaptive_last_rescue_frame", -(10**18))
        )
        self._adaptive_rescue_executed[pidx] = int(
            snap.get("adaptive_rescue_executed", 0)
        )
        self._adaptive_rescue_executed_for_lineage[pidx] = int(
            snap.get("adaptive_rescue_executed_for_lineage", 0)
        )
        self._adaptive_lineage_id[pidx] = str(snap.get("adaptive_lineage_id", ""))
        self._adaptive_rescue_attempt_count[pidx] = int(
            snap.get("adaptive_rescue_attempt_count", 0)
        )
        self._adaptive_rescue_selected_count[pidx] = int(
            snap.get("adaptive_rescue_selected_count", 0)
        )
        self._adaptive_rescue_blocked_count[pidx] = int(
            snap.get("adaptive_rescue_blocked_count", 0)
        )
        self._adaptive_rescue_last_reason[pidx] = str(
            snap.get("adaptive_rescue_last_reason", "")
        )
        self._adaptive_rescue_last_source[pidx] = int(
            snap.get("adaptive_rescue_last_source", -1)
        )
        self._adaptive_rescue_last_source_rank[pidx] = int(
            snap.get("adaptive_rescue_last_source_rank", -1)
        )
        self._adaptive_rescue_last_advantage_sigma[pidx] = float(
            snap.get("adaptive_rescue_last_advantage_sigma", 0.0)
        )
        self._adaptive_freeze_only_streak[pidx] = int(
            snap.get("adaptive_freeze_only_streak", 0)
        )
        self._adaptive_rollback_freeze_only_count[pidx] = int(
            snap.get("adaptive_rollback_freeze_only_count", 0)
        )
        self._adaptive_hard_collapse_trigger_source[pidx] = str(
            snap.get("adaptive_hard_collapse_trigger_source", "")
        )
        self._adaptive_rescue_live_selected[pidx] = int(
            snap.get("adaptive_rescue_live_selected", 0)
        )
        self._adaptive_rescue_archive_selected[pidx] = int(
            snap.get("adaptive_rescue_archive_selected", 0)
        )
        self._adaptive_rescue_ordinary_fallback[pidx] = int(
            snap.get("adaptive_rescue_ordinary_fallback", 0)
        )
        self._adaptive_rescue_last_source_kind[pidx] = str(
            snap.get("adaptive_rescue_last_source_kind", "none")
        )
        self._adaptive_rescue_burnin_blocked[pidx] = int(
            snap.get("adaptive_rescue_burnin_blocked", 0)
        )
        self._adaptive_rescue_burnin_bypassed[pidx] = int(
            snap.get("adaptive_rescue_burnin_bypassed", 0)
        )
        self._adaptive_rescue_emergency_bypass[pidx] = int(
            snap.get("adaptive_rescue_emergency_bypass", 0)
        )
        self._adaptive_last_checkpoint_frame[pidx] = int(
            snap.get("adaptive_last_checkpoint_frame", -1)
        )
        self._adaptive_last_checkpoint_path[pidx] = str(
            snap.get("adaptive_last_checkpoint_path", "")
        )
        self._adaptive_last_update_frame[pidx] = int(
            snap.get("adaptive_last_update_frame", -(10**18))
        )
        # Drop in-memory adaptive objects so restore reloads sidecar bytes.
        self._adaptive_probation.pop(pidx, None)
        self._adaptive_retention.pop(pidx, None)
        self._adaptive_sidecar_meta.pop(pidx, None)
        self._adaptive_peer_unavailable.pop(pidx, None)

    def _cleanup_failed_restart_artifacts(self) -> None:
        tmp_dir = self.workspace / f"restart_p{self.policy_idx}"
        if tmp_dir.exists():
            try:
                shutil.rmtree(tmp_dir)
            except Exception:
                pass
        try:
            supervisor_run_dir = self.train_dir.parent
            manifest_path = supervisor_run_dir / self.RESTART_MANIFEST
            if manifest_path.exists():
                manifest_path.unlink()
            for tmp in supervisor_run_dir.glob(f"{Path(self.RESTART_MANIFEST).stem}*.tmp"):
                try:
                    tmp.unlink()
                except Exception:
                    pass
        except Exception:
            pass

    def _snapshot_restart_persistent_state(self) -> dict[str, Any]:
        """Byte-level snapshot of control JSON + restart manifest.

        Restoring raw bytes preserves unknown JSON fields and avoids
        reconstructing ``CurriculumControlState`` (which drops extras).
        """
        control_path = (
            self._curriculum_control_dir / f"{int(self.policy_idx):03d}.json"
        )
        manifest_path = Path(self.train_dir).parent / self.RESTART_MANIFEST
        adaptive_path = self._adaptive_precision_path(int(self.policy_idx))
        control_existed = control_path.is_file()
        manifest_existed = manifest_path.is_file()
        adaptive_existed = adaptive_path.is_file()
        return {
            "control_path": control_path,
            "control_existed": bool(control_existed),
            "control_bytes": control_path.read_bytes() if control_existed else None,
            "manifest_path": manifest_path,
            "manifest_existed": bool(manifest_existed),
            "manifest_bytes": (
                manifest_path.read_bytes() if manifest_existed else None
            ),
            "adaptive_path": adaptive_path,
            "adaptive_existed": bool(adaptive_existed),
            "adaptive_bytes": (
                adaptive_path.read_bytes() if adaptive_existed else None
            ),
        }

    @staticmethod
    def _atomic_restore_file_bytes(
        path: Path,
        *,
        existed: bool,
        data: bytes | None,
    ) -> None:
        """Restore a file from a byte snapshot (or delete if it did not exist)."""
        path = Path(path)
        if existed:
            assert data is not None
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.parent / (
                f"{path.stem}.{os.getpid()}.{_uuid_mod.uuid4().hex[:8]}.restore.tmp"
            )
            tmp.write_bytes(data)
            os.replace(str(tmp), str(path))
            return
        if path.exists():
            try:
                path.unlink()
            except Exception:
                pass

    def _restore_restart_persistent_state(self, snapshot: dict[str, Any]) -> None:
        self._atomic_restore_file_bytes(
            Path(snapshot["control_path"]),
            existed=bool(snapshot["control_existed"]),
            data=snapshot.get("control_bytes"),
        )
        self._atomic_restore_file_bytes(
            Path(snapshot["manifest_path"]),
            existed=bool(snapshot["manifest_existed"]),
            data=snapshot.get("manifest_bytes"),
        )
        if "adaptive_path" in snapshot:
            self._atomic_restore_file_bytes(
                Path(snapshot["adaptive_path"]),
                existed=bool(snapshot.get("adaptive_existed")),
                data=snapshot.get("adaptive_bytes"),
            )
            # Force next adaptive access to re-read restored bytes.
            pidx = int(self.policy_idx)
            self._adaptive_probation.pop(pidx, None)
            self._adaptive_retention.pop(pidx, None)

    def _abort_restart_transaction(
        self,
        *,
        runtime_snap: dict[str, Any],
        persistent_snap: dict[str, Any],
        source_policy: int,
        env_frames: int,
        late_rescue: bool,
        reason: str,
        adaptive_rescue: bool = False,
    ) -> None:
        """Single rollback owner: restore memory + files, cleanup, one audit."""
        try:
            self._restore_restart_runtime_state(runtime_snap)
        except Exception as exc:
            print(f"[PBT] restart runtime restore failed: {exc}")
        try:
            self._restore_restart_persistent_state(persistent_snap)
        except Exception as exc:
            print(f"[PBT] restart persistent restore failed: {exc}")
        try:
            self._cleanup_failed_restart_artifacts()
        except Exception as exc:
            print(f"[PBT] restart artifact cleanup failed: {exc}")
        self._safe_write_pbt_copy_event({
            "event_type": "pbt_restart_transaction_rolled_back",
            "source_policy": int(source_policy),
            "target_policy": int(self.policy_idx),
            "late_rescue": bool(late_rescue),
            "adaptive_rescue": bool(adaptive_rescue),
            "env_frames": int(env_frames),
            "reason": str(reason),
        })
        if adaptive_rescue:
            self._safe_write_pbt_copy_event({
                "event_type": "adaptive_degradation_rescue_rolled_back",
                "rescue_mode": "adaptive",
                "source_policy": int(source_policy),
                "target_policy": int(self.policy_idx),
                "env_frames": int(env_frames),
                "reason": str(reason),
            })

    @staticmethod
    def _recent_slope(window: RollingNsuccWindow, n: int) -> float:
        xs = window.last_n(max(2, int(n)))
        if len(xs) < 2:
            return 0.0
        return float(xs[-1].nsucc_mean) - float(xs[0].nsucc_mean)

    def _evaluate_curriculum_safety(
        self,
        policy_idx: int,
        window: RollingNsuccWindow,
        env_frames: int,
        success_tolerance: float,
        shrink_token: bool,
        *,
        wm_cc: dict[str, Any],
        global_retention: float,
        stable_ticks: int,
        danger_zone: bool,
        repeated_collapse_count: int,
    ) -> dict[str, Any]:
        """Hybrid curriculum safety: real-signal fallback + WM + danger zone."""
        out: dict[str, Any] = {
            "active": False,
            "blocked": False,
            "block_reason": "",
            "real_fallback_used": False,
            "global_retention": float(global_retention),
            "recent_drop_ratio": 0.0,
            "recent_slope": 0.0,
            "stable_ticks": int(stable_ticks),
            "repeated_collapse_count": int(repeated_collapse_count),
            "danger_zone": bool(danger_zone),
            "imagined_valid": bool(wm_cc.get("imagined_valid_strict", False)),
            "imagined_valid_strict": bool(wm_cc.get("imagined_valid_strict", False)),
            "imagined_valid_soft": bool(wm_cc.get("imagined_valid_soft", False)),
            "wm_risk": False,
            "real_risk": False,
            "danger_blocked": False,
            "danger_block_reason": "",
            "danger_interval_mult": 1.0,
        }
        if not self.curriculum_safety_enable:
            return out
        if int(env_frames) < int(self.curriculum_safety_start_after_frames):
            return out
        if int(env_frames) < int(
            self._curriculum_safety_freeze_until.get(policy_idx, 0)
        ):
            out["active"] = True
            out["blocked"] = True
            out["block_reason"] = "curriculum_safety_freeze_active"
            return out
        out["active"] = True
        out["recent_drop_ratio"] = float(
            window.drop_ratio_vs_prior(max(2, self.curriculum_safety_window))
        )
        out["recent_slope"] = float(
            self._recent_slope(window, self.curriculum_safety_window)
        )
        nsucc_mean = 0.0
        if window.samples:
            nsucc_mean = float(window.samples[-1].nsucc_mean)
        risk_real = False
        if self.curriculum_safety_use_real_fallback:
            risk_real = (
                global_retention
                < self.curriculum_safety_global_retention_threshold
                or out["recent_drop_ratio"]
                > self.curriculum_safety_drop_ratio_threshold
                or out["recent_slope"] < self.curriculum_safety_slope_threshold
                or repeated_collapse_count
                >= self.curriculum_rollback_collapse_patience
            )
            out["real_fallback_used"] = True
        out["real_risk"] = bool(risk_real)
        risk_wm = False
        strict_valid = bool(
            wm_cc.get("imagined_valid_strict", wm_cc.get("eval_valid", False))
        )
        if strict_valid:
            risk_wm = (
                float(wm_cc.get("drop_risk", 0.0))
                >= self.wm_curriculum_controller_drop_risk_threshold
                or float(wm_cc.get("uncertainty", 1.0))
                >= self.wm_curriculum_controller_uncertainty_threshold
                or float(wm_cc.get("success_prob", 0.0))
                < self.wm_curriculum_controller_min_success_prob
            )
        out["wm_risk"] = bool(risk_wm)
        if danger_zone and self.curriculum_danger_zone_enable:
            out["danger_interval_mult"] = float(
                self.curriculum_danger_shrink_interval_mult
            )
        if not shrink_token:
            return out
        apply_block = nsucc_mean >= self.curriculum_safety_min_nsucc_mean
        if not apply_block:
            return out
        if risk_real:
            out["blocked"] = True
            out["block_reason"] = "real_signal_risk"
            self._curriculum_safety_freeze_until[policy_idx] = int(env_frames) + int(
                self.curriculum_safety_block_frames
            )
            return out
        if risk_wm:
            out["blocked"] = True
            out["block_reason"] = "wm_risk_valid"
            return out
        if (
            danger_zone
            and self.curriculum_danger_zone_enable
            and stable_ticks < self.curriculum_danger_min_stable_ticks
        ):
            out["blocked"] = True
            out["block_reason"] = "danger_zone_not_stable"
            out["danger_blocked"] = True
            out["danger_block_reason"] = "danger_zone_insufficient_stability"
            return out
        if (
            danger_zone
            and self.curriculum_danger_zone_enable
            and global_retention
            < self.curriculum_danger_global_retention_threshold
        ):
            out["blocked"] = True
            out["block_reason"] = "danger_zone_low_global_retention"
            out["danger_blocked"] = True
            out["danger_block_reason"] = "danger_zone_low_global_retention"
        return out

    def _evaluate_pre_floor_peak_guard(
        self,
        policy_idx: int,
        window: RollingNsuccWindow,
        env_frames: int,
        *,
        cur_tol: float,
        shrink_token: bool,
        recent_peak: float,
        adaptive_shrink_factor: float,
        stable_ticks: int,
    ) -> dict[str, Any]:
        """Block shrink before crossing target tol if peak retention is weak."""
        out: dict[str, Any] = {
            "active": False,
            "blocked": False,
            "reason": "",
            "retention": 1.0,
            "recent_peak_nsucc_mean": float(recent_peak),
            "current_nsucc_mean": 0.0,
            "stable_ticks": int(stable_ticks),
            "freeze_until": 0,
        }
        if not self.pre_floor_peak_guard_enable or not shrink_token:
            return out
        if not window.samples:
            return out

        eff_factor = (
            float(adaptive_shrink_factor)
            if float(adaptive_shrink_factor) > 0.0
            else float(self.curriculum_shrink_factor)
        )
        next_tol = max(
            eff_factor * float(cur_tol),
            float(self.pbt_hard_floor_tol_threshold),
        )
        current_nsucc = float(window.samples[-1].nsucc_mean)
        same_tol = self._same_tol_retention_snapshot(policy_idx, cur_tol)
        if self.same_tol_retention_enable and not bool(same_tol["valid"]):
            out["reason"] = "same_tol_retention_warmup"
            return out
        peak = float(same_tol["peak"]) if self.same_tol_retention_enable else float(recent_peak)
        if peak <= 1e-9:
            peak = max(float(s.nsucc_mean) for s in window.samples)
        retention = current_nsucc / peak if peak > 1e-9 else 1.0
        out["current_nsucc_mean"] = current_nsucc
        out["recent_peak_nsucc_mean"] = peak
        out["retention"] = float(retention)
        out["stable_ticks"] = int(stable_ticks)

        in_zone = (
            float(cur_tol) <= self.pre_floor_peak_guard_tol_threshold + 1e-9
            and float(next_tol) <= self.pre_floor_peak_guard_target_tol + 1e-9
            and peak >= self.pre_floor_peak_guard_peak_threshold
        )
        if not in_zone:
            return out

        out["active"] = True
        allow = (
            retention >= self.pre_floor_peak_guard_min_retention
            and int(stable_ticks) >= self.pre_floor_peak_guard_min_stable_ticks
        )
        if allow:
            out["reason"] = "pre_floor_peak_retention_ok"
            return out

        out["blocked"] = True
        out["reason"] = "pre_floor_peak_retention_guard"
        freeze_until = int(env_frames) + int(self.pre_floor_peak_guard_freeze_frames)
        out["freeze_until"] = freeze_until
        self._curriculum_safety_freeze_until[policy_idx] = freeze_until
        self._append_curriculum_safety_event(
            {
                "event": "pre_floor_peak_retention_guard",
                "policy_idx": int(policy_idx),
                "env_frames": int(env_frames),
                "success_tolerance": float(cur_tol),
                "next_tolerance": float(next_tol),
                "current_nsucc_mean": float(current_nsucc),
                "recent_peak_nsucc_mean": float(peak),
                "retention": float(retention),
                "stable_ticks": int(stable_ticks),
                "reason": "pre_floor_peak_retention_guard",
                "freeze_until": int(freeze_until),
                "shrink_blocked": True,
            }
        )
        return out

    def _evaluate_late_floor_retention_gate(
        self,
        policy_idx: int,
        window: RollingNsuccWindow,
        env_frames: int,
        *,
        cur_tol: float,
        shrink_token: bool,
        recent_peak: float,
        forced_tolerance: float,
    ) -> tuple[dict[str, Any], float]:
        """Late-floor retention gate: protect peak capability in strict tol band."""
        out: dict[str, Any] = {
            "active": False,
            "blocked": False,
            "reason": "",
            "retention": 1.0,
            "recent_peak_nsucc_mean": float(recent_peak),
            "current_nsucc_mean": 0.0,
            "freeze_until": 0,
            "rollback_applied": False,
        }
        new_forced = float(forced_tolerance)
        if not self.late_floor_retention_gate_enable:
            return out, new_forced
        if not window.samples:
            return out, new_forced

        current_nsucc = float(window.samples[-1].nsucc_mean)
        same_tol = self._same_tol_retention_snapshot(policy_idx, cur_tol)
        if self.same_tol_retention_enable and not bool(same_tol["valid"]):
            out["reason"] = "same_tol_retention_warmup"
            return out, new_forced
        peak = float(same_tol["peak"]) if self.same_tol_retention_enable else float(recent_peak)
        if peak <= 1e-9:
            peak = max(float(s.nsucc_mean) for s in window.samples)
        retention = current_nsucc / peak if peak > 1e-9 else 1.0
        out["current_nsucc_mean"] = current_nsucc
        out["recent_peak_nsucc_mean"] = peak
        out["retention"] = float(retention)

        late_floor = cur_tol <= self.late_floor_retention_tol_threshold + 1e-9
        strong_peak = peak >= self.late_floor_retention_peak_threshold
        low_retention = retention < self.late_floor_retention_min_retention
        hard_low_retention = (
            retention < self.late_floor_retention_hard_min_retention
        )

        if not (late_floor and strong_peak):
            self._late_floor_retention_patience_count[policy_idx] = 0
            return out, new_forced

        out["active"] = True
        freeze_until = int(env_frames) + int(self.late_floor_retention_freeze_frames)

        if hard_low_retention:
            self._late_floor_retention_patience_count[policy_idx] = int(
                self._late_floor_retention_patience_count.get(policy_idx, 0)
            ) + 1
        else:
            self._late_floor_retention_patience_count[policy_idx] = 0

        if (
            self.late_floor_retention_rollback_enable
            and hard_low_retention
            and self._late_floor_retention_patience_count.get(policy_idx, 0)
            >= int(self.late_floor_retention_patience)
        ):
            safe_tol = float(
                self._last_mastered_safe_tolerance.get(policy_idx, 0.0)
            )
            if safe_tol <= 0.0:
                safe_tol = float(
                    self._last_mastered_tolerance.get(policy_idx, 0.075)
                )
            if safe_tol > 1e-9 and cur_tol < safe_tol - 1e-9:
                new_forced = safe_tol
                out["rollback_applied"] = True
                out["reason"] = "late_floor_retention_rollback"
                self._late_floor_retention_patience_count[policy_idx] = 0
                self._curriculum_safety_freeze_until[policy_idx] = freeze_until
                self._append_curriculum_safety_event(
                    {
                        "event": "late_floor_retention_rollback",
                        "policy_idx": int(policy_idx),
                        "env_frames": int(env_frames),
                        "success_tolerance": float(cur_tol),
                        "current_nsucc_mean": float(current_nsucc),
                        "recent_peak_nsucc_mean": float(peak),
                        "retention": float(retention),
                        "reason": "late_floor_retention_rollback",
                        "freeze_until": int(freeze_until),
                        "shrink_blocked": True,
                        "rollback_to_tol": float(safe_tol),
                    }
                )
                return out, new_forced

        if low_retention and (
            self.late_floor_retention_block_shrink or shrink_token
        ):
            out["blocked"] = True
            out["reason"] = "late_floor_low_retention"
            out["freeze_until"] = freeze_until
            self._curriculum_safety_freeze_until[policy_idx] = freeze_until
            if self.late_floor_retention_freeze_mutation:
                self._mutation_frozen_until[policy_idx] = freeze_until
            self._append_curriculum_safety_event(
                {
                    "event": "late_floor_retention_block",
                    "policy_idx": int(policy_idx),
                    "env_frames": int(env_frames),
                    "success_tolerance": float(cur_tol),
                    "current_nsucc_mean": float(current_nsucc),
                    "recent_peak_nsucc_mean": float(peak),
                    "retention": float(retention),
                    "reason": "late_floor_low_retention",
                    "freeze_until": int(freeze_until),
                    "shrink_blocked": bool(shrink_token),
                }
            )
        return out, new_forced

    def _read_last_mastered_tolerance(self, policy_idx: int) -> float:
        path = self._curriculum_control_dir / f"{int(policy_idx):03d}.json"
        ctrl = read_curriculum_control(path)
        if ctrl is not None and ctrl.last_mastered_tolerance > 1e-9:
            return float(ctrl.last_mastered_tolerance)
        return float(self._last_mastered_tolerance.get(policy_idx, 0.075))

    def _resolve_restart_success_tolerance(
        self,
        *,
        source_policy: int,
        source_curriculum: dict[str, Any],
        checkpoint_type: str,
    ) -> tuple[float, str]:
        """Return (tolerance, mode_used) for restart curriculum resume."""
        ckpt_tol = float(
            source_curriculum.get("success_tolerance", 0.075) or 0.075
        )
        if checkpoint_type in (
            "archive",
            "best_stable_archive",
            "best_floor_recovery_archive",
        ):
            mode = self.pbt_recovery_tolerance_mode
        else:
            mode = self.pbt_crosscopy_tolerance_mode
        if mode == "checkpoint":
            return ckpt_tol, mode
        if mode == "last_mastered":
            return self._read_last_mastered_tolerance(source_policy), mode
        if mode == "relax_one_level":
            factor = float(self.curriculum_shrink_factor)
            relaxed = min(0.075, ckpt_tol / max(factor, 1e-6))
            return float(relaxed), mode
        return ckpt_tol, mode

    # ------------------------------------------------------------------ #
    # Phase A4 P3 (2026-06-09): source-safe curriculum inheritance.
    # ------------------------------------------------------------------ #
    def _extract_source_curriculum_metadata(
        self,
        *,
        source_state: dict[str, Any],
        source_policy_idx: int,
    ) -> dict[str, Any]:
        """Pull source success_tolerance / nsucc / retention for inheritance.

        Live cross-copy: source_state comes from the source's per-tick
            workspace meta (`_save_workspace_checkpoint`).
        Archive recovery: source_state is the archive _index.json dict.

        Returns a dict with floats and a ``has_*`` boolean flag set so the
        caller can decide stable vs unstable vs missing-metadata.
        """
        cur = dict(source_state.get("curriculum") or {})
        # Tolerance: prefer curriculum.success_tolerance; missing -> NaN.
        src_tol = cur.get("success_tolerance")
        if src_tol is None or (
            isinstance(src_tol, float) and math.isnan(src_tol)
        ):
            has_tol = False
            src_tol_f = float("nan")
        else:
            has_tol = True
            src_tol_f = float(src_tol)
        # nsucc_mean / nsucc_max: top-level first, fall back to curriculum.
        nm = source_state.get("nsucc_mean")
        if nm is None or (isinstance(nm, float) and math.isnan(nm)):
            nm = cur.get("mean_prev_episode_successes")
        nx = source_state.get("nsucc_max")
        if nx is None or (isinstance(nx, float) and math.isnan(nx)):
            nx = cur.get("max_prev_episode_successes")
        try:
            nm_f = float(nm) if nm is not None else float("nan")
        except Exception:
            nm_f = float("nan")
        try:
            nx_f = float(nx) if nx is not None else float("nan")
        except Exception:
            nx_f = float("nan")
        # Retention: prefer enriched archive field, then curriculum_control
        # of source policy idx, then live observer rolling window proxy.
        retention = source_state.get("source_retention_at_save")
        if retention is None:
            retention = cur.get("retention_used_for_shrink")
        if retention is None:
            ctrl_path = (
                self._curriculum_control_dir / f"{int(source_policy_idx):03d}.json"
            )
            ctrl = read_curriculum_control(ctrl_path)
            if ctrl is not None and ctrl.retention_used_for_shrink > 0.0:
                retention = float(ctrl.retention_used_for_shrink)
        if retention is None:
            window = self._policy_nsucc_windows.get(int(source_policy_idx))
            if window is not None and window.samples:
                end_mean = float(window.samples[-1].nsucc_mean)
                peak = max(float(s.nsucc_mean) for s in window.samples)
                retention = end_mean / peak if peak > 1e-9 else 1.0
        if retention is None or (
            isinstance(retention, float) and math.isnan(float(retention))
        ):
            retention_f = float("nan")
            has_retention = False
        else:
            retention_f = float(retention)
            has_retention = True
        return {
            "source_success_tolerance": src_tol_f,
            "has_source_success_tolerance": has_tol,
            "source_nsucc_mean": nm_f,
            "source_nsucc_max": nx_f,
            "source_retention": retention_f,
            "has_source_retention": has_retention,
        }

    @staticmethod
    def _is_source_safe_for_curriculum(
        meta: dict[str, Any],
        *,
        min_mean: float,
        min_retention: float,
    ) -> bool:
        nm = float(meta.get("source_nsucc_mean", float("nan")))
        if math.isnan(nm) or nm < float(min_mean):
            return False
        if not bool(meta.get("has_source_retention")):
            return False
        ret = float(meta.get("source_retention", float("nan")))
        if math.isnan(ret) or ret < float(min_retention):
            return False
        return True

    def _resolve_curriculum_inheritance(
        self,
        *,
        source_policy: int,
        source_state: dict[str, Any],
        source_curriculum: dict[str, Any],
        checkpoint_type: str,
        target_current_tolerance: float,
    ) -> dict[str, Any]:
        """Source-safe curriculum inheritance dispatcher.

        Returns a dict with the resolved post-copy tolerance plus a full
        audit payload for the copy-event logger and target control JSON.

        Branches:
          * inherit_mode == 'reset' -> delegate to legacy resolver, label
            disabled_reset_mode (legacy bit-for-bit when this flag is the
            default).
          * Source missing tolerance metadata -> legacy + missing_reason.
          * Source unstable (low nsucc / low retention) -> legacy +
            source_unstable_legacy.
          * Source stable -> inherited = max(src_tol/0.95, target_floor),
            target_after = min(current_target, inherited). Floor-clamp
            audited via floor_clamped reason.
        """
        meta = self._extract_source_curriculum_metadata(
            source_state=source_state, source_policy_idx=source_policy
        )
        ckpt_tol = float(
            source_curriculum.get("success_tolerance", 0.075) or 0.075
        )
        # Floor: never inherit below the env's target_success_tolerance.
        floor = float(
            source_curriculum.get("target_tolerance", 0.0) or 0.0
        )
        if floor <= 0.0:
            floor = 0.0  # legacy floor; env already clamps when applying
        legacy_tol, legacy_mode = self._resolve_restart_success_tolerance(
            source_policy=source_policy,
            source_curriculum=source_curriculum,
            checkpoint_type=checkpoint_type,
        )
        target_before = (
            float(target_current_tolerance)
            if target_current_tolerance is not None
            and not (
                isinstance(target_current_tolerance, float)
                and math.isnan(float(target_current_tolerance))
            )
            else 0.075
        )
        out: dict[str, Any] = {
            "tol_after": float(legacy_tol),
            "legacy_tol": float(legacy_tol),
            "legacy_mode": str(legacy_mode),
            "inheritance_mode": str(self.pbt_curriculum_inherit_mode),
            "inheritance_reason": "",
            "inherited_tolerance": 0.0,
            "target_tolerance_before_copy": target_before,
            "source_safe_for_curriculum": False,
            "source_meta": meta,
            "ckpt_tol": ckpt_tol,
        }

        if self.pbt_curriculum_inherit_mode != "source_safe":
            out["inheritance_reason"] = "disabled_reset_mode"
            return out

        if not bool(meta.get("has_source_success_tolerance")):
            out["inheritance_reason"] = "missing_source_tolerance"
            return out

        archive_like = checkpoint_type in (
            "archive",
            "best_stable_archive",
            "best_floor_recovery_archive",
            "late_rescue_archive",
        )
        is_live_cross_copy = not archive_like
        src_tol = float(meta.get("source_success_tolerance", float("nan")))
        src_mean = float(meta.get("source_nsucc_mean", float("nan")))
        src_max = float(meta.get("source_nsucc_max", float("nan")))
        strong_live = (
            self.pbt_strong_live_inherit_exact_enable
            and is_live_cross_copy
            and bool(meta.get("has_source_success_tolerance"))
            and math.isfinite(src_tol)
            and src_tol <= self.pbt_strong_live_inherit_max_tol
            and math.isfinite(src_mean)
            and src_mean >= self.pbt_strong_live_inherit_min_mean
            and math.isfinite(src_max)
            and src_max >= self.pbt_strong_live_inherit_min_max
        )
        if strong_live:
            tol_after = src_tol
            if floor > 0.0:
                tol_after = max(tol_after, floor)
            if target_before > 0.0 and math.isfinite(target_before):
                tol_after = min(tol_after, target_before)
            out["tol_after"] = float(tol_after)
            out["inherited_tolerance"] = float(tol_after)
            out["source_safe_for_curriculum"] = True
            out["inheritance_reason"] = "source_strong_live_inherit_exact"
            out["strong_live_inherit_exact"] = True
            return out

        is_safe = self._is_source_safe_for_curriculum(
            meta,
            min_mean=self.pbt_source_safe_min_mean,
            min_retention=self.pbt_source_safe_min_retention,
        )
        if not is_safe:
            out["inheritance_reason"] = "source_unstable_legacy"
            return out

        # Stable source -> one-level-looser inheritance.
        src_tol = float(meta["source_success_tolerance"])
        legacy_factor = float(self.curriculum_shrink_factor)
        if legacy_factor <= 0.0:
            legacy_factor = 0.95
        relaxed = src_tol / legacy_factor
        floor_applied = False
        if floor > 0.0 and relaxed < floor:
            relaxed = floor
            floor_applied = True
        # Never re-loosen tolerance: target keeps the tighter of (current
        # target, relaxed-from-source).
        if (
            target_before > 0.0
            and not math.isnan(target_before)
            and target_before < relaxed
        ):
            tol_after = target_before
            out["inheritance_reason"] = (
                "floor_clamped" if floor_applied else "target_already_tighter"
            )
        else:
            tol_after = relaxed
            out["inheritance_reason"] = (
                "floor_clamped" if floor_applied else "source_stable_relax_one_level"
            )
        out["tol_after"] = float(tol_after)
        out["inherited_tolerance"] = float(tol_after)
        out["source_safe_for_curriculum"] = True
        return out

    def _is_promising_protected(
        self,
        policy_idx: int,
        env_frames: int,
        own_state: dict[str, Any] | None = None,
    ) -> tuple[bool, str]:
        """Return (protected, reason). Override checks live at call site.

        Promotion gate: end_mean >= pbt_promising_nsucc_mean OR
        (recent_peak_nsucc_mean >= pbt_promising_peak_mean AND
         retention >= pbt_promising_min_retention).

        The retention guard prevents protecting a worker that once peaked
        high but has since collapsed (its peak still scores high but the
        retention proxy reveals the regression).
        """
        if not self.pbt_protect_promising_workers:
            return False, "feature_disabled"
        deadline = int(self._promising_protect_until.get(policy_idx, 0))
        if int(env_frames) >= deadline:
            return False, "no_active_window"
        return True, "active_window"

    def _maybe_enter_promising_protection(
        self,
        policy_idx: int,
        env_frames: int,
        window: RollingNsuccWindow,
    ) -> None:
        if not self.pbt_protect_promising_workers:
            return
        if not window.samples:
            return
        end_mean = float(window.samples[-1].nsucc_mean)
        peak = float(self._recent_peak_nsucc_mean.get(policy_idx, 0.0))
        retention = end_mean / peak if peak > 1e-9 else 0.0
        promising = end_mean >= float(self.pbt_promising_nsucc_mean) or (
            peak >= float(self.pbt_promising_peak_mean)
            and retention >= float(self.pbt_promising_min_retention)
        )
        if promising:
            self._promising_protect_until[policy_idx] = (
                int(env_frames) + int(self.pbt_promising_protect_frames)
            )

    def _write_pbt_copy_event(self, payload: dict[str, Any]) -> None:
        try:
            self._pbt_copy_events_path.parent.mkdir(
                parents=True, exist_ok=True
            )
            payload.setdefault("timestamp", time.time())
            payload.setdefault("policy_idx", int(self.policy_idx))
            payload.setdefault("pbt_iteration", int(self.pbt_iteration))
            with open(self._pbt_copy_events_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(payload) + "\n")
        except Exception as exc:  # pragma: no cover - best effort
            print(f"[pbt-copy-events] write failed: {exc}")

    def _safe_write_pbt_copy_event(self, payload: dict[str, Any]) -> None:
        """Best-effort audit write; never raises into restart rollback paths."""
        try:
            self._write_pbt_copy_event(payload)
        except Exception as exc:  # pragma: no cover - defensive
            print(f"[pbt-copy-events] safe write failed: {exc}")

    def _write_profile_copy_event(self, payload: dict[str, Any]) -> None:
        try:
            path = self.workspace / "metrics" / "profile_copy_events.jsonl"
            path.parent.mkdir(parents=True, exist_ok=True)
            payload.setdefault("timestamp", time.time())
            payload.setdefault("policy_idx", int(self.policy_idx))
            payload.setdefault("pbt_iteration", int(self.pbt_iteration))
            with open(path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(payload) + "\n")
        except Exception as exc:  # pragma: no cover - best effort
            print(f"[profile-copy-events] write failed: {exc}")

    def _update_policy_lineage(self, env_frames: int) -> None:
        try:
            path = self.workspace / "policy_lineage.json"
            lineage: dict[str, Any] = {}
            if path.is_file():
                lineage = json.loads(path.read_text())
            policies = lineage.setdefault("policies", {})
            policies[f"{int(self.policy_idx):03d}"] = {
                "policy_idx": int(self.policy_idx),
                "wm_strategy_profile_name": str(self.wm_strategy_profile_name),
                "wm_strategy_profile_id": int(self.wm_strategy_profile_id),
                "wm_strategy_profile_generation": int(
                    self._wm_strategy_profile_generation
                ),
                "wm_strategy_profile_parent": int(self._wm_strategy_profile_parent),
                "wm_strategy_profile_copied_from_source": bool(
                    self._wm_strategy_profile_copied_from_source
                ),
                "env_frames": int(env_frames),
                "pbt_iteration": int(self.pbt_iteration),
            }
            lineage["updated_at"] = time.time()
            path.write_text(json.dumps(lineage, indent=2))
        except Exception as exc:  # pragma: no cover - best effort
            print(f"[policy-lineage] write failed: {exc}")

    # ------------------------------------------------------------------
    # Phase A4 P4 (2026-06-09): floor-recovery-aware PBT helpers (Part A).
    # All methods are read-only / pure-derivation when called from a
    # disabled (default) configuration: callers must guard each call with
    # `self.pbt_floor_recovery_aware`.
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Phase A4 P4 wired (2026-06-10): cross-process WM imagined eval
    # mailbox. Each policy's ``train.py`` writes its own JSON entry into
    # ``<workspace>/wm_imagined/<pidx:03d>.json`` after each PPO epoch
    # (atomic temp+rename). Observers in *every* process refresh the
    # in-memory ``_wm_imagined_eval_per_policy`` from disk at PBT tick
    # time so the WM-guided ranking has a recent value for every peer
    # — not just the local policy.
    # ------------------------------------------------------------------

    def _refresh_wm_imagined_eval_from_disk(
        self, env_frames: int, *, max_age_frames: int = 200_000_000
    ) -> int:
        """Refresh the WM imagined-eval mapping from disk; returns count.

        Stale entries (writer ``frames`` older than ``max_age_frames``
        before ``env_frames``) keep the file payload but flip
        ``imagined_valid`` to False so the ranking falls back. The
        ``invalid_reason`` is set to ``"stale_wm_imagined"`` for audit.
        """
        wm_dir = self.workspace / "wm_imagined"
        if not wm_dir.exists():
            return 0
        n_loaded = 0
        for path in wm_dir.glob("*.json"):
            try:
                payload = json.loads(path.read_text())
            except Exception:
                continue
            try:
                pidx = int(
                    payload.get("policy_idx", int(path.stem.lstrip("0") or "0"))
                )
            except (TypeError, ValueError):
                continue
            try:
                payload_frames = int(payload.get("frames", 0))
            except (TypeError, ValueError):
                payload_frames = 0
            age = max(0, int(env_frames) - payload_frames)
            entry = dict(payload)
            if age > int(max_age_frames):
                entry["imagined_valid"] = False
                entry["invalid_reason"] = "stale_wm_imagined"
                entry["age_frames"] = int(age)
            else:
                entry["age_frames"] = int(age)
            self._wm_imagined_eval_per_policy[pidx] = entry
            n_loaded += 1
        return n_loaded

    def _floor_stage_for(
        self, curriculum: dict[str, Any] | None
    ) -> str:
        """Classify floor stage from a curriculum dict (env meta.json).

        Returns one of {"none","floor","hard_floor"}. Returns "none" on
        missing data so an empty population state never spuriously enters
        floor stage.
        """
        from dexpbt_isaaclab.dexpbt_utils import floor_stage_classify

        cur = curriculum or {}
        try:
            tol = float(cur.get("success_tolerance", float("nan")))
            tgt = float(cur.get("target_tolerance", 0.0))
        except (TypeError, ValueError):
            return "none"
        if math.isnan(tol):
            return "none"
        return floor_stage_classify(
            tol,
            tgt,
            floor_tol_threshold=self.pbt_floor_tol_threshold,
            hard_floor_tol_threshold=self.pbt_hard_floor_tol_threshold,
        )

    def _track_first_floor_frames(
        self, policy_idx: int, env_frames: int, stage: str
    ) -> None:
        """Record first time a policy hits floor / hard_floor (per-process)."""
        if stage in ("floor", "hard_floor") and policy_idx not in self._first_floor_frame:
            self._first_floor_frame[policy_idx] = int(env_frames)
        if stage == "hard_floor" and policy_idx not in self._first_hard_floor_frame:
            self._first_hard_floor_frame[policy_idx] = int(env_frames)

    def _floor_recovery_score_for(
        self,
        policy_idx: int,
        window: RollingNsuccWindow,
        env_frames: int,
        *,
        hard_collapse: bool = False,
    ) -> tuple[float, dict[str, float], float]:
        """Compute (score_50, components, slope_positive_norm) for one policy.

        Pure helper: caller is expected to have already classified the
        policy as being in (hard_)floor stage. ``hard_collapse`` should be
        True if the env-side curriculum control marks a hard collapse this
        tick; that flips the collapse-penalty term.
        """
        from dexpbt_isaaclab.dexpbt_utils import (
            compute_floor_recovery_score,
            compute_floor_recovery_slope,
        )

        # Recent nsucc_mean history for slope.
        history = [s.nsucc_mean for s in window.last_n(self.pbt_floor_recovery_window + 1)]
        slope_raw, slope_pos = compute_floor_recovery_slope(
            history, window=self.pbt_floor_recovery_window
        )

        if window.samples:
            last = window.samples[-1]
            nsucc_mean = float(last.nsucc_mean)
            nsucc_max = float(last.nsucc_max)
        else:
            nsucc_mean = 0.0
            nsucc_max = 0.0
        peak = float(self._recent_peak_nsucc_mean.get(policy_idx, 0.0))
        retention = nsucc_mean / peak if peak > 1e-9 else 1.0
        retention = float(max(0.0, min(1.0, retention)))
        drop_ratio = float(window.drop_ratio_vs_prior(2))

        score_50, components = compute_floor_recovery_score(
            nsucc_mean=nsucc_mean,
            nsucc_max=nsucc_max,
            retention=retention,
            slope_positive_norm=slope_pos,
            drop_ratio=drop_ratio,
            hard_collapse=hard_collapse,
            slope_weight=self.pbt_floor_recovery_slope_weight,
            retention_weight=self.pbt_floor_recovery_retention_weight,
            nsucc_max_weight=self.pbt_floor_recovery_nsucc_max_weight,
            drop_penalty=self.pbt_floor_recovery_drop_penalty,
        )
        components["slope_raw"] = float(slope_raw)
        # Track the score history for in-process reporting; bounded length.
        hist = self._floor_recovery_score_history.setdefault(policy_idx, [])
        hist.append(float(score_50))
        if len(hist) > 32:
            del hist[: len(hist) - 32]
        return float(score_50), components, float(slope_pos)

    def _is_floor_recovery_protected(
        self, policy_idx: int, env_frames: int
    ) -> tuple[bool, str]:
        """Return (protected, reason). Independent from promising gate."""
        if not self.pbt_floor_recovery_aware:
            return False, "feature_disabled"
        deadline = int(self._floor_recovery_protect_until.get(policy_idx, 0))
        if int(env_frames) >= deadline:
            return False, "no_active_window"
        return True, "active_window"

    def _maybe_enter_floor_recovery_protection(
        self,
        policy_idx: int,
        env_frames: int,
        window: RollingNsuccWindow,
        stage: str,
        *,
        hard_collapse: bool = False,
    ) -> None:
        """Promote a hard-floor worker into a recovery-protection window.

        Gate: policy must be at hard_floor and either
          (a) nsucc_mean >= pbt_floor_recovery_min_nsucc_protect, OR
          (b) slope_positive >= pbt_floor_recovery_min_slope_protect AND
              retention >= pbt_floor_recovery_min_retention_for_protect.
        Hard collapse cancels the gate (no protection extension).
        """
        if not self.pbt_floor_recovery_aware:
            return
        if stage != "hard_floor":
            return
        if hard_collapse:
            return
        if not window.samples:
            return
        score_50, components, slope_pos = self._floor_recovery_score_for(
            policy_idx, window, env_frames, hard_collapse=False
        )
        end_mean = float(window.samples[-1].nsucc_mean)
        retention = float(components.get("retention", 0.0))
        promising = end_mean >= float(self.pbt_floor_recovery_min_nsucc_protect) or (
            slope_pos >= float(self.pbt_floor_recovery_min_slope_protect)
            and retention >= float(self.pbt_floor_recovery_min_retention_for_protect)
        )
        if promising:
            self._floor_recovery_protect_until[policy_idx] = (
                int(env_frames) + int(self.pbt_floor_recovery_protect_frames)
            )

    def _select_best_floor_recovery_recovery_source(
        self, env_frames: int
    ) -> tuple[dict[str, Any] | None, str]:
        """Recover from workspace `best_floor_recovery/` archive.

        Mirrors `_select_best_stable_recovery_source` but uses the floor-
        recovery archive populated only when
        ``pbt_enable_best_floor_recovery_archive`` is True. Returns
        ``(synth, reason)`` with synth=None on any failure.
        """
        if not self.pbt_enable_best_floor_recovery_archive:
            return None, "feature_disabled"
        if env_frames - self._last_recovery_frame < self.recovery_cooldown_frames:
            return None, "recovery_cooldown"
        try:
            best = self._load_best_index(self.best_floor_recovery_workspace)
        except Exception as exc:  # pragma: no cover - defensive
            return None, f"archive_read_error:{type(exc).__name__}"
        if not best:
            return None, "archive_empty"
        # Floor recovery archive saves only at hard_floor; no nsucc gate
        # because nsucc at floor can be modest. Sanity-check the score.
        score = float(best.get("floor_recovery_score", float("-inf")))
        if math.isnan(score) or score <= 0.0:
            return None, "archive_score_nonpositive"
        prefix = self._peak_filename("floor_recovery", best)
        pth = self.best_floor_recovery_workspace / f"{prefix}.pth"
        if not pth.exists():
            return None, "archive_pth_missing"
        synth = dict(best)
        synth["checkpoint"] = str(pth)
        synth["_source_checkpoint_type"] = "best_floor_recovery_archive"
        return synth, "ok"

    def _select_tolerance_compatible_archive(
        self, *, target_tolerance: float, max_gap: float, min_mean: float,
        min_retention: float, require_retention_metadata: bool = False,
    ) -> tuple[dict[str, Any] | None, str]:
        """Reuse existing best_* checkpoint files, filtered by saved tolerance."""
        reject_log: list[dict[str, Any]] = []
        for directory, kind in (
            (self.best_floor_recovery_workspace, "best_floor_recovery_archive"),
            (self.best_stable_workspace, "best_stable_archive"),
            (self.best_nsucc_workspace, "archive"),
        ):
            state = select_compatible_archive(
                directory, target_tolerance=target_tolerance, max_gap=max_gap,
                min_mean=min_mean, min_retention=min_retention,
                require_retention_metadata=require_retention_metadata,
                reject_log=reject_log,
            )
            if state is None:
                continue
            checkpoint = Path(str(state.get("checkpoint", "")))
            # Archived metadata retains the original live path; derive the
            # colocated archive PTH from its JSON filename when needed.
            if not checkpoint.exists():
                meta = Path(str(state.get("_archive_metadata_path", "")))
                checkpoint = meta.with_suffix(".pth")
            if not checkpoint.exists():
                reject_log.append({
                    "path": str(state.get("_archive_metadata_path", "")),
                    "reason": "checkpoint_missing",
                })
                continue
            state["checkpoint"] = str(checkpoint)
            state["_source_checkpoint_type"] = kind
            state["_archive_reject_log"] = reject_log
            return state, "ok"
        return None, "no_compatible_archive"

    def _select_late_degradation_rescue_source(
        self,
        *,
        sorted_policies: list[int],
        population: dict[int, dict[str, Any] | None],
        env_frames: int,
        n_top: int,
        target_tolerance: float,
        target_nsucc_mean: float,
    ) -> dict[str, Any]:
        """Tolerance → quality → ranking. ``sorted_policies`` is ``list[int]``."""

        def _retention(src_idx: int, src_tol: float) -> float:
            snap = self._same_tol_retention_snapshot(int(src_idx), float(src_tol))
            if not bool(snap.get("valid", False)):
                return 0.0
            return float(snap.get("retention", 0.0))

        def _eligible(state: dict[str, Any], frames: int) -> tuple[bool, str]:
            return self._eligible_as_source_with_reason(state, frames)

        def _archive() -> tuple[dict[str, Any] | None, str]:
            return self._select_tolerance_compatible_archive(
                target_tolerance=target_tolerance,
                max_gap=float(self.late_degradation_rescue_max_tol_gap),
                min_mean=float(self.late_degradation_rescue_source_min_mean),
                min_retention=float(self.late_degradation_rescue_source_min_retention),
                require_retention_metadata=True,
            )

        def _collapse(src_idx: int, state: dict[str, Any]) -> tuple[bool, str]:
            return self._late_rescue_source_has_collapse(int(src_idx), state)

        return select_late_degradation_rescue_source(
            sorted_policies=list(sorted_policies),
            population=population,
            target_policy_idx=int(self.policy_idx),
            target_tolerance=float(target_tolerance),
            target_nsucc_mean=float(target_nsucc_mean),
            env_frames=int(env_frames),
            n_top=int(n_top),
            max_tol_gap=float(self.late_degradation_rescue_max_tol_gap),
            source_min_mean=float(self.late_degradation_rescue_source_min_mean),
            min_source_advantage=float(self.late_degradation_rescue_min_source_advantage),
            source_min_retention=float(self.late_degradation_rescue_source_min_retention),
            reject_collapse_source=bool(
                self.late_degradation_rescue_reject_collapse_source
            ),
            source_retention_fn=_retention,
            eligible_fn=_eligible,
            archive_fn=_archive,
            source_collapse_fn=_collapse,
        )

    def _log_best_stable_recovery_outcome(
        self,
        *,
        env_frames: int,
        outcome: str,
        skipped_reason: str | None = None,
        decision: dict[str, Any] | None = None,
    ) -> None:
        self._last_best_stable_recovery_outcome = str(outcome)
        record: dict[str, Any] = {
            "tier": "best_stable_archive",
            "outcome": outcome,
            "env_frames": int(env_frames),
            "pbt_iteration": int(self.pbt_iteration),
        }
        if skipped_reason:
            record["skipped_reason"] = skipped_reason
        if decision is not None:
            record["archive_reason"] = decision.get("archive_reason")
            record["source_policy"] = decision.get("source_policy")
        self._append_recovery_audit(record)
        if self.writer is None:
            return
        self.writer.add_scalar(
            "pbt/best_stable_recovery_selected",
            1.0 if outcome == "selected" else 0.0,
            env_frames,
        )
        self.writer.add_scalar(
            "pbt/best_stable_recovery_executed",
            1.0 if outcome == "executed" else 0.0,
            env_frames,
        )
        self.writer.add_scalar(
            "pbt/best_stable_recovery_skipped",
            1.0 if outcome == "skipped" else 0.0,
            env_frames,
        )

    def _select_best_stable_recovery_source(
        self, env_frames: int
    ) -> tuple[dict[str, Any] | None, str]:
        """P1 L4: recover from workspace `best_stable/` archive."""
        if env_frames - self._last_recovery_frame < self.recovery_cooldown_frames:
            return None, "recovery_cooldown"
        try:
            best = self._load_best_index(self.best_stable_workspace)
        except Exception as exc:  # pragma: no cover - defensive
            return None, f"archive_read_error:{type(exc).__name__}"
        if not best:
            return None, "archive_empty"
        nm = float(best.get("nsucc_mean", float("-inf")))
        nx = float(best.get("nsucc_max", float("-inf")))
        if math.isnan(nm) or nm < self.stable_source_min_mean:
            return None, f"archive_nsucc_mean<{self.stable_source_min_mean}"
        if math.isnan(nx) or nx < self.stable_source_min_max:
            return None, f"archive_nsucc_max<{self.stable_source_min_max}"
        prefix = self._peak_filename("stable", best)
        pth = self.best_stable_workspace / f"{prefix}.pth"
        if not pth.exists():
            return None, "archive_pth_missing"
        synth = dict(best)
        synth["checkpoint"] = str(pth)
        synth["_source_checkpoint_type"] = "best_stable_archive"
        return synth, "ok"

    def _select_source_stable_mean(
        self,
        sorted_policies: list[int],
        population: dict[int, dict[str, Any] | None],
        env_frames: int,
        n_top: int,
    ) -> dict[str, Any]:
        """P1 cascade: L1 stable -> L2 usable -> L3 relaxed -> L4 best_stable -> skip."""
        top_raw = sorted_policies[:n_top]

        stable_filter_reasons: dict[int, str] = {}
        stable_candidates: list[int] = []
        for p in top_raw:
            ok, reason = self._eligible_as_stable_source_with_reason(
                population.get(p), env_frames, p
            )
            if ok:
                stable_candidates.append(p)
            else:
                stable_filter_reasons[p] = reason

        if stable_candidates:
            picked = random.choice(stable_candidates)
            return {
                "action": "copy",
                "tier": "strict_stable_live",
                "source_policy": picked,
                "source_state": population[picked],
                "checkpoint_type": "latest_healthy",
                "strict_candidates": stable_candidates,
                "relaxed_candidates": [],
                "strict_filter_reasons": stable_filter_reasons,
                "relaxed_filter_reasons": {},
                "archive_reason": "not_attempted",
                "skip_reason": None,
            }

        usable_filter_reasons: dict[int, str] = {}
        usable_candidates: list[int] = []
        for p, state in population.items():
            if p == self.policy_idx:
                continue
            ok, reason = self._eligible_as_usable_live_with_reason(
                state, env_frames, p
            )
            if ok:
                usable_candidates.append(p)
            else:
                usable_filter_reasons[p] = reason

        if usable_candidates:
            def _score(idx: int) -> float:
                s = population.get(idx) or {}
                v = s.get("pbt_ranking_score", s.get("objective", float("-inf")))
                try:
                    return float(v)
                except Exception:
                    return float("-inf")

            picked = max(usable_candidates, key=_score)
            return {
                "action": "copy",
                "tier": "usable_live",
                "source_policy": picked,
                "source_state": population[picked],
                "checkpoint_type": "latest_healthy",
                "strict_candidates": [],
                "relaxed_candidates": usable_candidates,
                "strict_filter_reasons": stable_filter_reasons,
                "relaxed_filter_reasons": usable_filter_reasons,
                "archive_reason": "not_attempted",
                "skip_reason": None,
            }

        relaxed_filter_reasons: dict[int, str] = {}
        relaxed_candidates: list[int] = []
        for p, state in population.items():
            if p == self.policy_idx:
                continue
            ok, reason = self._eligible_as_source_relaxed_with_reason(
                state, env_frames
            )
            if ok:
                relaxed_candidates.append(p)
            else:
                relaxed_filter_reasons[p] = reason

        if relaxed_candidates:
            def _score(idx: int) -> float:
                s = population.get(idx) or {}
                v = s.get("pbt_ranking_score", s.get("objective", float("-inf")))
                try:
                    return float(v)
                except Exception:
                    return float("-inf")

            picked = max(relaxed_candidates, key=_score)
            return {
                "action": "copy",
                "tier": "relaxed_live",
                "source_policy": picked,
                "source_state": population[picked],
                "checkpoint_type": "latest_healthy",
                "strict_candidates": [],
                "relaxed_candidates": relaxed_candidates,
                "strict_filter_reasons": stable_filter_reasons,
                "relaxed_filter_reasons": relaxed_filter_reasons,
                "archive_reason": "not_attempted",
                "skip_reason": None,
            }

        # Phase A4 P4 (2026-06-09): floor-recovery archive comes first
        # because it is only populated at hard floor where existing
        # best_stable / best_nsucc archives may be far behind. If the
        # feature is off this returns immediately with synth=None.
        synth_fr, archive_reason_fr = self._select_best_floor_recovery_recovery_source(
            env_frames
        )
        if synth_fr is not None:
            return {
                "action": "copy",
                "tier": "best_floor_recovery_archive",
                "source_policy": int(synth_fr.get("policy_idx", -1)),
                "source_state": synth_fr,
                "checkpoint_type": "best_floor_recovery_archive",
                "strict_candidates": [],
                "relaxed_candidates": relaxed_candidates,
                "strict_filter_reasons": stable_filter_reasons,
                "relaxed_filter_reasons": relaxed_filter_reasons,
                "archive_reason": archive_reason_fr,
                "skip_reason": None,
            }

        synth, archive_reason = self._select_best_stable_recovery_source(env_frames)
        if synth is not None:
            return {
                "action": "copy",
                "tier": "best_stable_archive",
                "source_policy": int(synth.get("policy_idx", -1)),
                "source_state": synth,
                "checkpoint_type": "best_stable_archive",
                "strict_candidates": [],
                "relaxed_candidates": relaxed_candidates,
                "strict_filter_reasons": stable_filter_reasons,
                "relaxed_filter_reasons": relaxed_filter_reasons,
                "archive_reason": archive_reason,
                "skip_reason": None,
            }

        if self._best_nsucc_recovery_enabled():
            synth, archive_reason = self._select_best_nsucc_recovery_source(env_frames)
            if synth is not None:
                return {
                    "action": "copy",
                    "tier": "archive",
                    "source_policy": int(synth.get("policy_idx", -1)),
                    "source_state": synth,
                    "checkpoint_type": "archive",
                    "strict_candidates": [],
                    "relaxed_candidates": relaxed_candidates,
                    "strict_filter_reasons": stable_filter_reasons,
                    "relaxed_filter_reasons": relaxed_filter_reasons,
                    "archive_reason": archive_reason,
                    "skip_reason": None,
                }

        if self.allow_low_nsucc_source_fallback and top_raw:
            picked = random.choice(top_raw)
            return {
                "action": "copy",
                "tier": "strict_stable_live",
                "source_policy": picked,
                "source_state": population[picked],
                "checkpoint_type": "latest_healthy",
                "strict_candidates": [],
                "relaxed_candidates": relaxed_candidates,
                "strict_filter_reasons": stable_filter_reasons,
                "relaxed_filter_reasons": relaxed_filter_reasons,
                "archive_reason": archive_reason,
                "skip_reason": "legacy_low_nsucc_fallback",
            }

        return {
            "action": "skip",
            "tier": "skipped",
            "source_policy": -1,
            "source_state": None,
            "checkpoint_type": "none",
            "strict_candidates": [],
            "relaxed_candidates": relaxed_candidates,
            "strict_filter_reasons": stable_filter_reasons,
            "relaxed_filter_reasons": relaxed_filter_reasons,
            "archive_reason": archive_reason,
            "skip_reason": "no_eligible_source_any_tier",
        }

    def _select_source(
        self,
        sorted_policies: list[int],
        population: dict[int, dict[str, Any] | None],
        env_frames: int,
        n_top: int,
    ) -> dict[str, Any]:
        """Patch G (2026-05-31) cascade: L1 -> L2 -> L3 -> L4.

        Returns a dict describing the bottom-branch decision:
          {
            "action": "copy" | "skip",
            "tier": "strict_live" | "relaxed_live" | "archive" | "skipped",
            "source_policy": int (live source idx; -1 for archive/skip),
            "source_state": dict | None,
            "checkpoint_type": "live" | "archive" | "none",
            "strict_candidates": list[int],
            "relaxed_candidates": list[int],
            "strict_filter_reasons": dict[int, str],
            "relaxed_filter_reasons": dict[int, str],
            "archive_reason": str,
            "skip_reason": str | None,
          }

        Each tier is evaluated independently so the full audit trail is
        emitted regardless of which tier the cascade ultimately fires at.
        """
        if self.source_require_stable_mean:
            return self._select_source_stable_mean(
                sorted_policies, population, env_frames, n_top
            )

        top_raw = sorted_policies[:n_top]
        strict_filter_reasons: dict[int, str] = {}
        strict_candidates: list[int] = []
        for p in top_raw:
            ok, reason = self._eligible_as_source_with_reason(
                population.get(p), env_frames
            )
            if ok:
                strict_candidates.append(p)
            else:
                strict_filter_reasons[p] = reason

        # L1 strict ----------------------------------------------------
        if strict_candidates:
            picked = random.choice(strict_candidates)
            return {
                "action": "copy",
                "tier": "strict_live",
                "source_policy": picked,
                "source_state": population[picked],
                "checkpoint_type": "live",
                "strict_candidates": strict_candidates,
                "relaxed_candidates": [],
                "strict_filter_reasons": strict_filter_reasons,
                "relaxed_filter_reasons": {},
                "archive_reason": "not_attempted",
                "skip_reason": None,
            }

        # In strict-only mode we replicate the old Patch A behaviour and
        # either fall back (legacy) or skip — without consulting L2/L3.
        if self.source_selection_mode == "strict":
            if self.allow_low_nsucc_source_fallback:
                picked = random.choice(top_raw) if top_raw else self.policy_idx
                return {
                    "action": "copy",
                    "tier": "strict_live",  # legacy fallback path
                    "source_policy": picked,
                    "source_state": population[picked],
                    "checkpoint_type": "live",
                    "strict_candidates": strict_candidates,
                    "relaxed_candidates": [],
                    "strict_filter_reasons": strict_filter_reasons,
                    "relaxed_filter_reasons": {},
                    "archive_reason": "not_attempted",
                    "skip_reason": "strict_fallback",
                }
            return {
                "action": "skip",
                "tier": "skipped",
                "source_policy": -1,
                "source_state": None,
                "checkpoint_type": "none",
                "strict_candidates": [],
                "relaxed_candidates": [],
                "strict_filter_reasons": strict_filter_reasons,
                "relaxed_filter_reasons": {},
                "archive_reason": "not_attempted",
                "skip_reason": "strict_mode_no_eligible",
            }

        # L2 relaxed ---------------------------------------------------
        relaxed_filter_reasons: dict[int, str] = {}
        relaxed_candidates: list[int] = []
        # Consider all peers (top + mid + bottom) under the relaxed gate
        # so we can rescue PBT even if every "top" candidate is degraded.
        for p, state in population.items():
            if p == self.policy_idx:
                continue
            ok, reason = self._eligible_as_source_relaxed_with_reason(
                state, env_frames
            )
            if ok:
                relaxed_candidates.append(p)
            else:
                relaxed_filter_reasons[p] = reason
        if relaxed_candidates:
            # Prefer the relaxed candidate with the highest pbt_ranking_score
            # so we still pick "best of the bad" rather than uniform random.
            def _score(idx: int) -> float:
                s = population.get(idx) or {}
                v = s.get("pbt_ranking_score", s.get("objective", float("-inf")))
                try:
                    return float(v)
                except Exception:
                    return float("-inf")
            picked = max(relaxed_candidates, key=_score)
            return {
                "action": "copy",
                "tier": "relaxed_live",
                "source_policy": picked,
                "source_state": population[picked],
                "checkpoint_type": "live",
                "strict_candidates": strict_candidates,
                "relaxed_candidates": relaxed_candidates,
                "strict_filter_reasons": strict_filter_reasons,
                "relaxed_filter_reasons": relaxed_filter_reasons,
                "archive_reason": "not_attempted",
                "skip_reason": None,
            }

        # L3 archive ---------------------------------------------------
        synth, archive_reason = self._select_best_nsucc_recovery_source(env_frames)
        if synth is not None:
            return {
                "action": "copy",
                "tier": "archive",
                "source_policy": int(synth.get("policy_idx", -1)),
                "source_state": synth,
                "checkpoint_type": "archive",
                "strict_candidates": strict_candidates,
                "relaxed_candidates": relaxed_candidates,
                "strict_filter_reasons": strict_filter_reasons,
                "relaxed_filter_reasons": relaxed_filter_reasons,
                "archive_reason": archive_reason,
                "skip_reason": None,
            }

        # Legacy escape hatch ------------------------------------------
        # Before falling all the way to L4 skip, honour the Patch A
        # `pbt_allow_low_nsucc_source_fallback` flag for backward
        # compatibility / ablation. This re-admits the raw top-N (even
        # though they failed the strict gate) so PBT does not stall on a
        # legacy run; we tag the tier as `strict_live` with provenance
        # `live` since the source is still a peer worker.
        if self.allow_low_nsucc_source_fallback and top_raw:
            picked = random.choice(top_raw)
            return {
                "action": "copy",
                "tier": "strict_live",
                "source_policy": picked,
                "source_state": population[picked],
                "checkpoint_type": "live",
                "strict_candidates": strict_candidates,
                "relaxed_candidates": relaxed_candidates,
                "strict_filter_reasons": strict_filter_reasons,
                "relaxed_filter_reasons": relaxed_filter_reasons,
                "archive_reason": archive_reason,
                "skip_reason": "legacy_low_nsucc_fallback",
            }

        # L4 skip ------------------------------------------------------
        return {
            "action": "skip",
            "tier": "skipped",
            "source_policy": -1,
            "source_state": None,
            "checkpoint_type": "none",
            "strict_candidates": strict_candidates,
            "relaxed_candidates": relaxed_candidates,
            "strict_filter_reasons": strict_filter_reasons,
            "relaxed_filter_reasons": relaxed_filter_reasons,
            "archive_reason": archive_reason,
            "skip_reason": "no_eligible_source_any_tier",
        }

    def _eligible_for_self_restart(self, env_frames: int) -> tuple[bool, str]:
        """Patch F (2026-05-31): governor for mid-tier self-mutation.

        Returns (eligible, reason). When `eligible=False`, the caller must
        NOT relaunch the worker and instead let mid tier keep training.

        Order of checks:
          1. `disable_self_restart` master switch (default ON).
          2. `allow_mid_self_mutation` legacy escape hatch (default OFF).
          3. `max_self_restarts_per_policy` lifetime cap.
          4. `min_frames_between_self_restarts` cooldown since previous
             self-restart for this worker.
        """
        if self.disable_self_restart and not self.allow_mid_self_mutation:
            return False, "disabled_by_default"
        if self._self_restart_count >= self.max_self_restarts_per_policy:
            return False, (
                f"self_restart_count>={self.max_self_restarts_per_policy}"
            )
        # _last_self_restart_frame defaults to a very negative value, so the
        # first attempt always passes the cooldown gate.
        if (
            env_frames - self._last_self_restart_frame
            < self.min_frames_between_self_restarts
        ):
            return False, (
                f"frames_since_self_restart<"
                f"{self.min_frames_between_self_restarts}"
            )
        return True, "ok"

    def _is_eligible_for_elimination(
        self, state: dict[str, Any] | None, env_frames: int
    ) -> bool:
        """Elimination-cooldown gate. Independent from source-cooldown.

        Once a worker has trained at least `min_frames_between_restarts`
        frames it may be killed again. This is decoupled from
        `eligible_as_source` so a bad worker does not get to camp out
        indefinitely just because it recently restarted.
        """
        if state is None:
            return True  # no meta yet -> nothing to protect
        fsr = self._frames_since_restart(state)
        return fsr >= self.min_frames_between_restarts

    def set_seeding_audit(self, seed: int | None, env_randomization_hash: str) -> None:
        """Record per-policy seed + env randomization fingerprint.

        Called by train.py after the env is created so the diversity values
        appear in every subsequent training_curve point.
        """

        self._effective_seed = int(seed) if seed is not None else None
        self._env_randomization_hash = str(env_randomization_hash or "")

    def _append_pbt_tick_debug(
        self,
        *,
        env_frames: int,
        population_frame: int,
        derived_iteration: int,
        should_tick: bool,
        skip_reason: str,
        error_reason: str = "",
    ) -> None:
        """v17.2 debug hotfix: append-only audit for tick scheduler decisions."""
        try:
            self._metrics_dir.mkdir(parents=True, exist_ok=True)
            out_path = self._metrics_dir / f"pbt_tick_debug_policy_{self.policy_idx:02d}.jsonl"
            latest = read_latest_commit_info(self.curr_policy_workspace)
            row = {
                "env_frames": int(env_frames),
                "population_frame": int(population_frame),
                "derived_iteration": int(derived_iteration),
                "self_pbt_iteration": int(self.pbt_iteration),
                "last_mutation_frame": int(self.last_mutation_frame),
                "interval_steps": int(self.interval_steps),
                "initial_delay": int(self.initial_delay),
                "should_tick": bool(should_tick),
                "skip_reason": str(error_reason or skip_reason or ""),
                **latest,
            }
            with out_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row) + "\n")
        except Exception as exc:  # pragma: no cover - audit must not crash training
            print(f"[pbt-tick-debug] append failed: {exc}")

    def _write_training_curve_point(
        self,
        own_state: dict[str, Any],
        env_frames: int,
        curriculum: dict,
        tier: str = "unknown",
    ) -> None:
        """M5: append one JSON line per PBT tick to a per-policy curve file.

        Patch 1 (rev 2): record both `r_meta_raw` (legacy DexPBT formula)
        and `pbt_ranking_score` (used by selection). `r_meta` is retained
        as an alias of `r_meta_raw` for backward compatibility with
        existing analyzer scripts (analyze_training_curve.py etc).
        """
        try:
            self._metrics_dir.mkdir(parents=True, exist_ok=True)
            out_path = self._metrics_dir / f"training_curve_policy_{self.policy_idx:02d}.jsonl"
            r_meta_raw = float(
                own_state.get(
                    "r_meta_raw",
                    own_state.get("objective", float("nan")),
                )
            )
            ranking_score = float(
                own_state.get("pbt_ranking_score", own_state.get("objective", float("nan")))
            )
            own_ctrl = read_curriculum_control(
                self._curriculum_control_dir / f"{self.policy_idx:03d}.json"
            )
            from dexpbt_isaaclab.world_model.wm_activation_gate import tol_bin as _wm_tol_bin

            _succ_tol = float(curriculum.get("success_tolerance", float("nan")))
            _curriculum_tol_bin = (
                _wm_tol_bin(_succ_tol) if math.isfinite(_succ_tol) else "le_004"
            )
            _actor_distill_tol_bin = str(
                curriculum.get("actor_distill_tol_bin") or _curriculum_tol_bin
            )
            point = {
                "env_frames": int(env_frames),
                "policy_idx": int(self.policy_idx),
                "pbt_iteration": int(self.pbt_iteration),
                # Patch 1: keep legacy alias `r_meta` -> r_meta_raw so older
                # analyzers continue to parse without changes. New fields are
                # explicit.
                "r_meta": r_meta_raw,
                "r_meta_raw": r_meta_raw,
                "pbt_ranking_score": ranking_score,
                "ranking_metric_mode": str(
                    own_state.get("ranking_metric_mode", self.pbt_ranking_metric)
                ),
                "nsucc_zero_cap": float(
                    own_state.get("nsucc_zero_cap", self.pbt_nsucc_zero_cap)
                ),
                # Patch B (rev 3, 2026-05-30): expose low-nsucc cap params
                # in the training_curve so the analyzer/TB consumer can
                # tell whether a `pbt_ranking_score` < `r_meta_raw` is due
                # to the zero-cap, low-cap, or unmodified.
                "nsucc_low_threshold": float(
                    own_state.get("nsucc_low_threshold", self.pbt_nsucc_low_threshold)
                ),
                "nsucc_low_cap": float(
                    own_state.get("nsucc_low_cap", self.pbt_nsucc_low_cap)
                ),
                "r_meta_window_size": int(self._last_window_size),
                "nsucc_mean": float(curriculum.get("mean_prev_episode_successes", float("nan"))),
                "nsucc_max": float(curriculum.get("max_prev_episode_successes", float("nan"))),
                "success_tolerance": float(curriculum.get("success_tolerance", float("nan"))),
                "target_success_tolerance": float(curriculum.get("target_tolerance", float("nan"))),
                "initial_tolerance": float(curriculum.get("initial_tolerance", float("nan"))),
                "frame_since_restart": int(curriculum.get("frame_since_restart", 0)),
                "last_curriculum_update": int(curriculum.get("last_curriculum_update", 0)),
                "restart_count": int(self._restart_count),
                # Patch 2 (rev 2, 2026-05-30): anti-churn columns for the
                # analyzer and ad-hoc TB-less inspection.
                "frames_at_restart": int(self._frames_at_restart),
                "last_mutation_frame": int(self.last_mutation_frame),
                "eligible_as_source": bool(self.curr_eligible_as_source),
                "eligible_for_elimination": bool(self.curr_eligible_for_elimination),
                # Patch A (rev 3, 2026-05-30): why was/wasn't this worker
                # eligible at this tick. One of:
                # "ok" | "no_state" | "frames_since_restart<N" |
                # "nsucc_mean<X" | "nsucc_max<Y"
                "eligible_source_reason": str(self.curr_eligible_source_reason),
                "min_frames_as_source_after_restart": int(
                    self.min_frames_as_source_after_restart
                ),
                "min_frames_between_restarts": int(
                    self.min_frames_between_restarts
                ),
                "min_nsucc_mean_as_source": float(self.min_nsucc_mean_as_source),
                "min_nsucc_max_as_source": float(self.min_nsucc_max_as_source),
                "allow_low_nsucc_source_fallback": bool(
                    self.allow_low_nsucc_source_fallback
                ),
                # Patch F (2026-05-31): self/cross-restart governor state.
                "disable_self_restart": bool(self.disable_self_restart),
                "allow_mid_self_mutation": bool(self.allow_mid_self_mutation),
                "max_self_restarts_per_policy": int(
                    self.max_self_restarts_per_policy
                ),
                "min_frames_between_self_restarts": int(
                    self.min_frames_between_self_restarts
                ),
                "self_restart_count": int(self._self_restart_count),
                "cross_restart_count": int(self._cross_restart_count),
                "self_restart_blocked_count": int(self._self_restart_blocked_count),
                "mid_self_mutation_count": int(self._mid_self_mutation_count),
                "last_self_restart_frame": int(self._last_self_restart_frame),
                "last_restart_type": int(self._last_restart_type),
                # Patch G (2026-05-31): tiered source selection state.
                "source_selection_mode": str(self.source_selection_mode),
                "last_source_tier_label": str(self._last_source_tier_label),
                "last_source_checkpoint_type": str(
                    self._last_source_checkpoint_type
                ),
                "source_tier_strict_live_count": int(
                    self._source_tier_counts["strict_live"]
                ),
                "source_tier_relaxed_live_count": int(
                    self._source_tier_counts["relaxed_live"]
                ),
                "source_tier_archive_count": int(
                    self._source_tier_counts["archive"]
                ),
                "source_tier_skipped_count": int(
                    self._source_tier_counts["skipped"]
                ),
                "recovery_restart_count": int(self._recovery_restart_count),
                "last_recovery_frame": int(self._last_recovery_frame),
                "enable_best_nsucc_recovery": bool(self.enable_best_nsucc_recovery),
                "relaxed_nsucc_mean_as_source": float(
                    self.relaxed_nsucc_mean_as_source
                ),
                "relaxed_nsucc_max_as_source": float(
                    self.relaxed_nsucc_max_as_source
                ),
                "recovery_min_nsucc_mean": float(self.recovery_min_nsucc_mean),
                "recovery_min_nsucc_max": float(self.recovery_min_nsucc_max),
                "recovery_cooldown_frames": int(self.recovery_cooldown_frames),
                "best_stable_recovery_selected_count": int(
                    self._best_stable_recovery_selected_count
                ),
                "best_stable_recovery_executed_count": int(
                    self._best_stable_recovery_executed_count
                ),
                "best_stable_recovery_skipped_count": int(
                    self._best_stable_recovery_skipped_count
                ),
                "last_best_stable_recovery_outcome": str(
                    self._last_best_stable_recovery_outcome
                ),
                "pbt_recovery_tolerance_mode": str(
                    self.pbt_recovery_tolerance_mode
                ),
                "pbt_crosscopy_tolerance_mode": str(
                    self.pbt_crosscopy_tolerance_mode
                ),
                "mutation_frozen": bool(
                    int(env_frames)
                    < self._mutation_frozen_until.get(self.policy_idx, 0)
                ),
                "hyperparam_gamma": float(
                    (own_state.get("params") or {}).get("gamma", float("nan"))
                    if own_state
                    else float("nan")
                ),
                "hyperparam_learning_rate": float(
                    (own_state.get("params") or {}).get(
                        "learning_rate", float("nan")
                    )
                    if own_state
                    else float("nan")
                ),
                "hyperparam_wm_strategy": str(
                    (own_state.get("params") or {}).get("wm_strategy", "")
                    if own_state
                    else ""
                ),
                "shrink_token": bool(own_ctrl.shrink_token) if own_ctrl else False,
                "shrinks_in_window": int(own_ctrl.shrinks_in_window) if own_ctrl else 0,
                "forced_success_tolerance": float(
                    own_ctrl.forced_success_tolerance if own_ctrl else 0.0
                ),
                "collapse_freeze_active": bool(
                    own_ctrl.collapse_freeze_active if own_ctrl else False
                ),
                "collapse_freeze_until_env_frames": int(
                    own_ctrl.collapse_freeze_until_env_frames if own_ctrl else 0
                ),
                "collapse_rollback_active": bool(
                    own_ctrl.collapse_rollback_active if own_ctrl else False
                ),
                "early_collapse_triggered": bool(
                    own_ctrl.early_collapse_triggered if own_ctrl else False
                ),
                "hard_collapse_triggered": bool(
                    own_ctrl.hard_collapse_triggered if own_ctrl else False
                ),
                "rollback_and_freeze_same_tick": bool(
                    own_ctrl.rollback_and_freeze_same_tick if own_ctrl else False
                ),
                "rollback_cooldown_active": bool(
                    own_ctrl.rollback_cooldown_active if own_ctrl else False
                ),
                "rollback_count": int(own_ctrl.rollback_count) if own_ctrl else 0,
                "last_rollback_frame": int(own_ctrl.last_rollback_frame)
                if own_ctrl
                else -10**18,
                "rollback_reason": str(own_ctrl.rollback_reason) if own_ctrl else "",
                "source_checkpoint_is_archive": bool(
                    self._last_source_checkpoint_type
                    in ("archive", "best_stable_archive")
                ),
                "source_checkpoint_is_best_stable_archive": bool(
                    self._last_source_checkpoint_type == "best_stable_archive"
                ),
                "usable_live_min_mean": float(self.pbt_usable_live_min_mean),
                "usable_live_min_max": float(self.pbt_usable_live_min_max),
                "usable_live_max_drop_ratio": float(self.pbt_usable_live_max_drop_ratio),
                # Patch H (2026-05-31): post-restart curriculum freeze.
                "curriculum_freeze_frames": int(self.curriculum_freeze_frames),
                "curriculum_freeze_on_recovery_frames": int(
                    self.curriculum_freeze_on_recovery_frames
                ),
                "curriculum_freeze_frames_after_restart_env": int(
                    curriculum.get("curriculum_freeze_frames_after_restart", 0)
                ),
                # Phase A4 P2 (2026-06-08): adaptive-fast curriculum
                # diagnostics. All fields are present for both 'normal' and
                # 'adaptive_fast' modes; in normal mode adaptive_shrink_factor
                # is 0.0 and shrink_reason is "normal_mode".
                "curriculum_speed_mode": str(self.curriculum_speed_mode),
                "curriculum_adaptive_fast_enabled": bool(
                    self.curriculum_speed_mode == "adaptive_fast"
                ),
                "curriculum_adaptive_shrink_factor": float(
                    own_ctrl.adaptive_shrink_factor if own_ctrl else 0.0
                ),
                "curriculum_adaptive_consolidation_frames": int(
                    own_ctrl.adaptive_consolidation_frames if own_ctrl else 0
                ),
                "curriculum_shrink_reason": str(
                    own_ctrl.shrink_reason if own_ctrl else ""
                ),
                "curriculum_nsucc_mean_used_for_shrink": float(
                    own_ctrl.nsucc_mean_used_for_shrink if own_ctrl else 0.0
                ),
                "curriculum_retention_used_for_shrink": float(
                    own_ctrl.retention_used_for_shrink if own_ctrl else 0.0
                ),
                "curriculum_blocked_by_recovery": bool(
                    own_ctrl.blocked_by_recovery if own_ctrl else False
                ),
                "curriculum_blocked_by_collapse": bool(
                    own_ctrl.blocked_by_collapse if own_ctrl else False
                ),
                "curriculum_blocked_by_consolidation": bool(
                    own_ctrl.blocked_by_consolidation if own_ctrl else False
                ),
                "curriculum_blocked_by_mastery_gate": bool(
                    own_ctrl.blocked_by_mastery_gate if own_ctrl else False
                ),
                "curriculum_blocked_by_adaptive_fast_low_nsucc": bool(
                    own_ctrl.blocked_by_adaptive_fast_low_nsucc
                    if own_ctrl
                    else False
                ),
                "curriculum_pre_shrink_tolerance": float(
                    own_ctrl.pre_shrink_tolerance if own_ctrl else 0.0
                ),
                "curriculum_post_shrink_tolerance": float(
                    own_ctrl.post_shrink_tolerance if own_ctrl else 0.0
                ),
                "curriculum_fast_target_tolerance_800m": float(
                    self.curriculum_fast_target_tolerance_800m
                ),
                # Phase A4 P3 (2026-06-09): inheritance / promising audit.
                "pbt_curriculum_inherit_mode": str(
                    self.pbt_curriculum_inherit_mode
                ),
                "pbt_protect_promising_workers": bool(
                    self.pbt_protect_promising_workers
                ),
                "promising_protect_until": int(
                    self._promising_protect_until.get(self.policy_idx, 0)
                ),
                "protected_by_promising_gate_now": bool(
                    self._is_promising_protected(self.policy_idx, env_frames)[0]
                ),
                "curriculum_inheritance_mode_state": str(
                    own_ctrl.inheritance_mode if own_ctrl else "reset"
                ),
                "curriculum_inheritance_reason_state": str(
                    own_ctrl.inheritance_reason if own_ctrl else ""
                ),
                "curriculum_inherited_tolerance_state": float(
                    own_ctrl.inherited_tolerance if own_ctrl else 0.0
                ),
                "curriculum_target_tolerance_before_copy_state": float(
                    own_ctrl.target_tolerance_before_copy if own_ctrl else 0.0
                ),
                "curriculum_target_tolerance_after_copy_state": float(
                    own_ctrl.target_tolerance_after_copy if own_ctrl else 0.0
                ),
                "curriculum_source_policy_idx_for_copy_state": int(
                    own_ctrl.source_policy_idx_for_copy if own_ctrl else -1
                ),
                "curriculum_source_checkpoint_type_for_copy_state": str(
                    own_ctrl.source_checkpoint_type_for_copy if own_ctrl else ""
                ),
                "curriculum_source_safe_for_curriculum_state": bool(
                    own_ctrl.source_safe_for_curriculum if own_ctrl else False
                ),
                "curriculum_source_nsucc_mean_for_copy_state": float(
                    own_ctrl.source_nsucc_mean_for_copy if own_ctrl else 0.0
                ),
                "curriculum_source_retention_for_copy_state": float(
                    own_ctrl.source_retention_for_copy if own_ctrl else 0.0
                ),
                "curriculum_recovery_consolidation_frames_selected_state": int(
                    own_ctrl.recovery_consolidation_frames_selected
                    if own_ctrl
                    else 0
                ),
                "curriculum_recovery_consolidation_source_state": str(
                    own_ctrl.recovery_consolidation_source if own_ctrl else ""
                ),
                "curriculum_inherited_tolerance_count": int(
                    self._curriculum_inherited_count
                ),
                "curriculum_reset_to_initial_tolerance_count": int(
                    self._curriculum_reset_count
                ),
                "curriculum_missing_source_curriculum_metadata_count": int(
                    self._curriculum_missing_metadata_count
                ),
                # Phase A4 P4 (2026-06-09): floor-recovery audit (Part A+E).
                "pbt_floor_recovery_aware": bool(self.pbt_floor_recovery_aware),
                "pbt_enable_best_floor_recovery_archive": bool(
                    self.pbt_enable_best_floor_recovery_archive
                ),
                "pbt_floor_tol_threshold": float(self.pbt_floor_tol_threshold),
                "pbt_hard_floor_tol_threshold": float(
                    self.pbt_hard_floor_tol_threshold
                ),
                "floor_stage": str(own_ctrl.floor_stage if own_ctrl else "none"),
                "first_floor_frame": int(
                    own_ctrl.first_floor_frame if own_ctrl else -1
                ),
                "first_hard_floor_frame": int(
                    own_ctrl.first_hard_floor_frame if own_ctrl else -1
                ),
                "frames_since_floor": int(
                    own_ctrl.frames_since_floor if own_ctrl else 0
                ),
                "floor_recovery_slope": float(
                    own_ctrl.floor_recovery_slope if own_ctrl else 0.0
                ),
                "floor_recovery_score": float(
                    own_ctrl.floor_recovery_score if own_ctrl else 0.0
                ),
                "floor_recovery_score_real": float(
                    own_ctrl.floor_recovery_score_real if own_ctrl else 0.0
                ),
                "floor_recovery_score_wm": float(
                    own_ctrl.floor_recovery_score_wm if own_ctrl else 0.0
                ),
                "floor_recovery_protect_until": int(
                    self._floor_recovery_protect_until.get(self.policy_idx, 0)
                ),
                "protected_by_floor_recovery_gate_now": bool(
                    self._is_floor_recovery_protected(
                        self.policy_idx, env_frames
                    )[0]
                ),
                # Phase A4 P4: WM-guided ranking three-mode + shadow audit.
                "pbt_wm_score_mode": str(self.pbt_wm_score_mode),
                "pbt_use_wm_recovery_score": bool(self.pbt_use_wm_recovery_score),
                "pbt_wm_recovery_weight": float(self.pbt_wm_recovery_weight),
                "pbt_wm_success_weight": float(self.pbt_wm_success_weight),
                "pbt_wm_drop_penalty": float(self.pbt_wm_drop_penalty),
                "pbt_wm_uncertainty_penalty": float(
                    self.pbt_wm_uncertainty_penalty
                ),
                "pbt_wm_max_uncertainty": float(self.pbt_wm_max_uncertainty),
                "wm_shadow_observations": int(self._wm_shadow_observations),
                "wm_shadow_changed_selection": int(
                    self._wm_shadow_changed_selection
                ),
                # Phase A4 P4 wired (2026-06-10): per-tick WM wiring audit.
                "wm_score_used": bool(
                    getattr(self, "_last_wm_score_used", False)
                ),
                "wm_fallback_reason": str(
                    getattr(self, "_last_wm_fallback_reason", "feature_off")
                ),
                "experimental_rssm_floor_recovery_stack": bool(
                    self.experimental_rssm_floor_recovery_stack
                ),
                # PBT-WM Strategy Zoo v1.0
                "wm_strategy_profile_name": str(self.wm_strategy_profile_name),
                "wm_strategy_profile_id": int(self.wm_strategy_profile_id),
                "wm_strategy_profile_generation": int(
                    self._wm_strategy_profile_generation
                ),
                "wm_strategy_profile_parent": int(self._wm_strategy_profile_parent),
                "wm_strategy_profile_copied_from_source": bool(
                    self._wm_strategy_profile_copied_from_source
                ),
                "profile_actor_distill_enabled": bool(
                    self.profile_actor_distill_enabled
                ),
                "profile_residual_apply_enabled": bool(
                    self.profile_residual_apply_enabled
                ),
                "profile_rssm_mppi_enabled": bool(self.profile_rssm_mppi_enabled),
                "profile_wm_pbt_score_enabled": bool(
                    self.profile_wm_pbt_score_enabled
                ),
                "profile_wm_curriculum_controller_enabled": bool(
                    self.profile_wm_curriculum_controller_enabled
                ),
                "profile_elite_bank_enabled": bool(self.profile_elite_bank_enabled),
                "profile_conservative_planner_enabled": bool(
                    self.profile_conservative_planner_enabled
                ),
                "pbt_wm_profile_copy_enable": bool(self.pbt_wm_profile_copy_enable),
                "pbt_wm_profile_copy_decision": str(
                    getattr(self, "_last_pbt_wm_profile_copy_decision", "disabled")
                ),
                "pbt_wm_profile_copy_prob": float(self.pbt_wm_profile_copy_prob),
                "pbt_wm_profile_diversity_count": int(
                    getattr(self, "_last_pbt_wm_profile_diversity_count", 0)
                ),
                "profile_copy_enabled": bool(self.pbt_wm_profile_copy_enable),
                "profile_copy_eligible": bool(
                    getattr(self, "_last_profile_copy_eligible", False)
                ),
                "profile_copy_attempt_count": int(
                    getattr(self, "_profile_copy_attempt_count", 0)
                ),
                "profile_copy_accept_count": int(
                    getattr(self, "_profile_copy_accept_count", 0)
                ),
                "profile_copy_reject_count": int(
                    getattr(self, "_profile_copy_reject_count", 0)
                ),
                "profile_copy_reject_reason": str(
                    getattr(self, "_last_profile_copy_reject_reason", "")
                ),
                "profile_copy_reject_reason_counts": dict(
                    getattr(self, "_profile_copy_reject_reason_counts", {})
                ),
                "profile_copy_source_policy": int(
                    getattr(self, "_last_profile_copy_source_policy", -1)
                ),
                "profile_copy_target_policy": int(self.policy_idx),
                "profile_copy_source_profile": str(
                    getattr(self, "_last_profile_copy_source_profile", "")
                ),
                "profile_copy_target_old_profile": str(
                    getattr(self, "_last_profile_copy_target_old_profile", "")
                ),
                "profile_copy_target_new_profile": str(
                    getattr(self, "_last_profile_copy_target_new_profile", "")
                ),
                "profile_copy_generation": int(self._wm_strategy_profile_generation),
                "weight_generation": int(self._wm_strategy_profile_generation),
                "weight_copy_count": int(getattr(self, "_weight_copy_count", 0)),
                "weight_copy_with_profile_copy_count": int(
                    getattr(self, "_weight_copy_with_profile_copy_count", 0)
                ),
                "weight_copy_without_profile_copy_count": int(
                    getattr(self, "_weight_copy_without_profile_copy_count", 0)
                ),
                "profile_diversity_count": int(
                    getattr(self, "_last_pbt_wm_profile_diversity_count", 0)
                ),
                "wm_curriculum_controller_active": bool(
                    own_ctrl.wm_curriculum_controller_active if own_ctrl else False
                ),
                "wm_curriculum_eval_valid": bool(
                    own_ctrl.wm_curriculum_eval_valid if own_ctrl else False
                ),
                "wm_curriculum_drop_risk": float(
                    own_ctrl.wm_curriculum_drop_risk if own_ctrl else 0.0
                ),
                "wm_curriculum_success_prob": float(
                    own_ctrl.wm_curriculum_success_prob if own_ctrl else 0.0
                ),
                "wm_curriculum_recovery_prob": float(
                    own_ctrl.wm_curriculum_recovery_prob if own_ctrl else 0.0
                ),
                "wm_curriculum_uncertainty": float(
                    own_ctrl.wm_curriculum_uncertainty if own_ctrl else 0.0
                ),
                "wm_curriculum_blocked_shrink": bool(
                    own_ctrl.wm_curriculum_blocked_shrink if own_ctrl else False
                ),
                "wm_curriculum_freeze_until": int(
                    own_ctrl.wm_curriculum_freeze_until if own_ctrl else 0
                ),
                "wm_curriculum_fallback_reason_id": int(
                    own_ctrl.wm_curriculum_fallback_reason_id if own_ctrl else 0
                ),
                # Hybrid Curriculum Safety Patch (2026-06-15).
                "curriculum_safety_active": bool(
                    own_ctrl.curriculum_safety_active if own_ctrl else False
                ),
                "curriculum_safety_real_fallback_used": bool(
                    own_ctrl.curriculum_safety_real_fallback_used if own_ctrl else False
                ),
                "curriculum_safety_blocked_shrink": bool(
                    own_ctrl.curriculum_safety_blocked_shrink if own_ctrl else False
                ),
                "curriculum_safety_block_reason": str(
                    own_ctrl.curriculum_safety_block_reason if own_ctrl else ""
                ),
                "curriculum_safety_global_retention": float(
                    own_ctrl.curriculum_safety_global_retention if own_ctrl else 0.0
                ),
                "curriculum_safety_recent_drop_ratio": float(
                    own_ctrl.curriculum_safety_recent_drop_ratio if own_ctrl else 0.0
                ),
                "curriculum_safety_recent_slope": float(
                    own_ctrl.curriculum_safety_recent_slope if own_ctrl else 0.0
                ),
                "curriculum_safety_stable_ticks": int(
                    own_ctrl.curriculum_safety_stable_ticks if own_ctrl else 0
                ),
                "curriculum_safety_repeated_collapse_count": int(
                    own_ctrl.curriculum_safety_repeated_collapse_count
                    if own_ctrl
                    else 0
                ),
                "curriculum_safety_danger_zone": bool(
                    own_ctrl.curriculum_safety_danger_zone if own_ctrl else False
                ),
                "curriculum_safety_imagined_valid": bool(
                    own_ctrl.curriculum_safety_imagined_valid if own_ctrl else False
                ),
                "curriculum_safety_wm_risk": bool(
                    own_ctrl.curriculum_safety_wm_risk if own_ctrl else False
                ),
                "curriculum_safety_real_risk": bool(
                    own_ctrl.curriculum_safety_real_risk if own_ctrl else False
                ),
                "curriculum_global_retention": float(
                    own_ctrl.curriculum_global_retention if own_ctrl else 0.0
                ),
                "curriculum_recent_peak_nsucc_mean_long": float(
                    own_ctrl.curriculum_recent_peak_nsucc_mean_long
                    if own_ctrl
                    else 0.0
                ),
                "curriculum_strong_shrink_allowed": bool(
                    own_ctrl.curriculum_strong_shrink_allowed if own_ctrl else True
                ),
                "curriculum_strong_shrink_block_reason": str(
                    own_ctrl.curriculum_strong_shrink_block_reason if own_ctrl else ""
                ),
                "curriculum_strong_shrink_global_gate_active": bool(
                    own_ctrl.curriculum_strong_shrink_global_gate_active
                    if own_ctrl
                    else False
                ),
                "curriculum_danger_zone": bool(
                    own_ctrl.curriculum_danger_zone if own_ctrl else False
                ),
                "curriculum_danger_tol": float(
                    own_ctrl.curriculum_danger_tol if own_ctrl else 0.03
                ),
                "curriculum_danger_blocked_shrink": bool(
                    own_ctrl.curriculum_danger_blocked_shrink if own_ctrl else False
                ),
                "curriculum_danger_block_reason": str(
                    own_ctrl.curriculum_danger_block_reason if own_ctrl else ""
                ),
                "curriculum_danger_stable_ticks": int(
                    own_ctrl.curriculum_danger_stable_ticks if own_ctrl else 0
                ),
                "curriculum_danger_interval_mult": float(
                    own_ctrl.curriculum_danger_interval_mult if own_ctrl else 1.0
                ),
                "curriculum_last_mastered_tolerance": float(
                    own_ctrl.curriculum_last_mastered_tolerance if own_ctrl else 0.0
                ),
                "curriculum_last_mastered_frame": int(
                    own_ctrl.curriculum_last_mastered_frame if own_ctrl else -1
                ),
                "curriculum_last_mastered_nsucc_mean": float(
                    own_ctrl.curriculum_last_mastered_nsucc_mean if own_ctrl else 0.0
                ),
                "curriculum_last_mastered_global_retention": float(
                    own_ctrl.curriculum_last_mastered_global_retention
                    if own_ctrl
                    else 0.0
                ),
                "curriculum_rollback_triggered": bool(
                    own_ctrl.curriculum_rollback_triggered if own_ctrl else False
                ),
                "curriculum_rollback_reason": str(
                    own_ctrl.rollback_reason if own_ctrl else ""
                ),
                "curriculum_rollback_from_tol": float(
                    own_ctrl.curriculum_rollback_from_tol if own_ctrl else 0.0
                ),
                "curriculum_rollback_to_tol": float(
                    own_ctrl.curriculum_rollback_to_tol if own_ctrl else 0.0
                ),
                "curriculum_rollback_freeze_until": int(
                    own_ctrl.curriculum_rollback_freeze_until if own_ctrl else 0
                ),
                "curriculum_repeated_collapse_count": int(
                    own_ctrl.curriculum_repeated_collapse_count if own_ctrl else 0
                ),
                "wm_imagined_valid": bool(
                    own_ctrl.wm_imagined_valid if own_ctrl else False
                ),
                "wm_imagined_uncertainty": float(
                    own_ctrl.wm_imagined_uncertainty if own_ctrl else 1.0
                ),
                "wm_imagined_success_prob": float(
                    own_ctrl.wm_imagined_success_prob if own_ctrl else 0.0
                ),
                "wm_imagined_drop_risk": float(
                    own_ctrl.wm_imagined_drop_risk if own_ctrl else 0.0
                ),
                "wm_imagined_recovery_prob": float(
                    own_ctrl.wm_imagined_recovery_prob if own_ctrl else 0.0
                ),
                "wm_imagined_invalid_reason": str(
                    own_ctrl.wm_imagined_invalid_reason if own_ctrl else ""
                ),
                "wm_imagined_shadow_block_decision": bool(
                    own_ctrl.wm_imagined_shadow_block_decision if own_ctrl else False
                ),
                "wm_imagined_shadow_block_reason": str(
                    own_ctrl.wm_imagined_shadow_block_reason if own_ctrl else ""
                ),
                # PBT-WM Soft Activation Gate v1 diagnostics.
                "wm_curriculum_tol_bin": str(
                    own_ctrl.wm_curriculum_tol_bin if own_ctrl else "le_004"
                ),
                "wm_source_activation_weight": float(
                    getattr(self, "_last_wm_source_activation_weight", 0.0)
                ),
                "wm_source_activation_nonzero": float(
                    getattr(self, "_last_wm_source_activation_nonzero", 0.0)
                ),
                "wm_source_unc_weight": float(
                    getattr(self, "_last_wm_source_unc_weight", 0.0)
                ),
                "wm_source_time_weight": float(
                    getattr(self, "_last_wm_source_time_weight", 0.0)
                ),
                "wm_validity_weight": float(
                    getattr(self, "_last_wm_validity_weight", 0.0)
                ),
                "wm_score_effective_weight": float(
                    getattr(self, "_last_wm_score_effective_weight", 0.0)
                ),
                "wm_score_used_strict": float(
                    getattr(self, "_last_wm_score_used_strict", 0.0)
                ),
                "wm_score_used_soft": float(
                    getattr(self, "_last_wm_score_used_soft", 0.0)
                ),
                "wm_ranking_uncertainty_threshold": float(
                    getattr(self, "_last_wm_ranking_uncertainty_threshold", 0.0)
                ),
                "wm_ranking_uncertainty_gate_mode": str(
                    getattr(self, "_last_wm_ranking_uncertainty_gate_mode", "none")
                ),
                "curriculum_tol_bin": str(_curriculum_tol_bin),
                "actor_distill_tol_bin": _actor_distill_tol_bin,
                "tier": str(tier),
                "checkpoint": own_state.get("checkpoint", ""),
                "params": {k: own_state["params"][k] for k in own_state.get("params", {})},
                # Seed diversity audit (B1/B2): set by train.py after env creation.
                "seed": int(self._effective_seed) if self._effective_seed is not None else None,
                "env_randomization_hash": str(self._env_randomization_hash or ""),
                "timestamp": float(own_state.get("timestamp", time.time())),
            }
            awr_bridge = getattr(self.algo, "_awr_bridge", None)
            if awr_bridge is not None and getattr(awr_bridge, "buffer", None) is not None:
                for tag, val in dict(getattr(awr_bridge, "stats", {}) or {}).items():
                    if (
                        isinstance(tag, str)
                        and tag.startswith("awr/")
                        and isinstance(val, (int, float))
                        and math.isfinite(float(val))
                    ):
                        point[tag.replace("awr/", "awr_")] = float(val)
                point["awr_buffer_steps"] = float(awr_bridge.buffer.total_steps())
                point["awr_buffer_sequences"] = float(awr_bridge.buffer.num_sequences())
                point["awr_enabled"] = float(
                    0.0 if awr_bridge.disabled else awr_bridge.stats.get("awr/enabled", 1.0)
                )
                point.setdefault(
                    "awr_effective_lambda",
                    float(awr_bridge.stats.get("awr/effective_lambda", 0.0)),
                )
            if self.algo is not None:
                for key, attr in (
                    ("pbt_copy_reset_progress_due_to_frame_gap", "_pbt_copy_reset_progress_due_to_frame_gap"),
                    ("pbt_copy_source_frame", "_pbt_copy_source_frame"),
                    ("pbt_copy_target_frame_before_restore", "_pbt_copy_target_frame_before_restore"),
                    ("pbt_copy_frame_gap", "_pbt_copy_frame_gap"),
                    ("pbt_copy_frame_gap_threshold", "_pbt_copy_frame_gap_threshold"),
                ):
                    val = getattr(self.algo, attr, None)
                    if val is not None:
                        point[key] = float(val)
            with open(out_path, "a") as fh:
                fh.write(json.dumps(point) + "\n")
                fh.flush()
                try:
                    os.fsync(fh.fileno())
                except OSError:
                    pass
            policy_run_dir = self.workspace.parent / f"policy_{self.policy_idx:02d}"
            write_curriculum_resume_snapshot(
                policy_run_dir,
                curriculum=curriculum,
                env_frames=int(env_frames),
                policy_idx=int(self.policy_idx),
            )
        except Exception as e:
            print(f"[curve] Failed to write training curve for policy {self.policy_idx}: {e}")

    def _write_population_aggregate(
        self,
        population: dict[int, dict[str, Any] | None],
        env_frames: int,
    ) -> None:
        """M5: workspace-level aggregate (best/mean across policies) at this tick.

        We compute aggregate from whatever peer meta.json files are visible at
        the moment of THIS policy's tick. Each policy writes its own row; the
        analyzer can dedup or average across writers per env_frames bucket.
        """
        rows = [s for s in population.values() if s is not None]
        if not rows:
            return
        # Patch 1 (rev 2): with the new schema, `objective` carries the
        # PBT ranking score (used by sort/tier) while `r_meta_raw` keeps
        # the legacy DexPBT formula value. Aggregate both:
        #   * population/best_r_meta        -> raw (legacy alias)
        #   * population/best_ranking_score -> success-aware selector
        objs = [
            float(s.get("r_meta_raw", s.get("objective", float("nan"))))
            for s in rows
        ]
        objs_valid = [v for v in objs if not math.isnan(v) and not math.isinf(v)]
        ranking_scores = [
            float(s.get("pbt_ranking_score", s.get("objective", float("nan"))))
            for s in rows
        ]
        ranking_scores_valid = [
            v for v in ranking_scores if not math.isnan(v) and not math.isinf(v)
        ]
        tols = [
            float((s.get("curriculum") or {}).get("success_tolerance", float("nan")))
            for s in rows
        ]
        tols_valid = [v for v in tols if not math.isnan(v)]
        nsucc_mean = [
            float((s.get("curriculum") or {}).get("mean_prev_episode_successes", float("nan")))
            for s in rows
        ]
        nsucc_mean_valid = [v for v in nsucc_mean if not math.isnan(v)]
        nsucc_max = [
            float((s.get("curriculum") or {}).get("max_prev_episode_successes", float("nan")))
            for s in rows
        ]
        nsucc_max_valid = [v for v in nsucc_max if not math.isnan(v)]
        target_tols = [
            float((s.get("curriculum") or {}).get("target_tolerance", float("nan")))
            for s in rows
        ]
        if self.writer is not None:
            if objs_valid:
                self.writer.add_scalar("population/best_r_meta", max(objs_valid), env_frames)
                self.writer.add_scalar("population/mean_r_meta", sum(objs_valid) / len(objs_valid), env_frames)
                best_idx = int(rows[objs.index(max(objs_valid))].get("policy_idx", -1))
                self.writer.add_scalar("population/best_policy_idx", float(best_idx), env_frames)
            if ranking_scores_valid:
                self.writer.add_scalar(
                    "population/best_ranking_score",
                    max(ranking_scores_valid),
                    env_frames,
                )
                self.writer.add_scalar(
                    "population/mean_ranking_score",
                    sum(ranking_scores_valid) / len(ranking_scores_valid),
                    env_frames,
                )
            if nsucc_max_valid:
                self.writer.add_scalar("population/best_nsucc_max", max(nsucc_max_valid), env_frames)
                if nsucc_mean_valid:
                    self.writer.add_scalar("population/best_nsucc_mean", max(nsucc_mean_valid), env_frames)
                # Backward-compatible alias
                self.writer.add_scalar("population/best_nsucc", max(nsucc_max_valid), env_frames)
            if nsucc_mean_valid:
                self.writer.add_scalar(
                    "population/mean_nsucc_mean",
                    sum(nsucc_mean_valid) / len(nsucc_mean_valid),
                    env_frames,
                )
                # Backward-compatible alias
                self.writer.add_scalar(
                    "population/mean_nsucc",
                    sum(nsucc_mean_valid) / len(nsucc_mean_valid),
                    env_frames,
                )
            if tols_valid:
                self.writer.add_scalar("population/best_success_tolerance", min(tols_valid), env_frames)
                self.writer.add_scalar(
                    "population/mean_success_tolerance",
                    sum(tols_valid) / len(tols_valid),
                    env_frames,
                )
            if tols_valid and target_tols:
                target_tol = next((t for t in target_tols if not math.isnan(t)), None)
                if target_tol is not None:
                    at_target = sum(1 for t in tols_valid if t <= target_tol * 1.0001)
                    self.writer.add_scalar(
                        "population/num_policies_at_target_tolerance",
                        float(at_target),
                        env_frames,
                    )

    def after_init(self, algo):
        super().after_init(algo)
        self.algo = algo
        self.workspace.mkdir(parents=True, exist_ok=True)
        self.curr_policy_workspace.mkdir(parents=True, exist_ok=True)
        self.best_policy_workspace.mkdir(parents=True, exist_ok=True)
        # Patch 5 (rev 2, 2026-05-30): ensure peak archive dirs exist and
        # rebuild in-memory best trackers from disk so that a crash/restart
        # does not lose peak DexPBT checkpoints. Each directory has its own
        # `_index.json` summarizing the current best entry.
        self.best_r_meta_workspace.mkdir(parents=True, exist_ok=True)
        self.best_nsucc_workspace.mkdir(parents=True, exist_ok=True)
        self.best_stable_workspace.mkdir(parents=True, exist_ok=True)
        self._best_r_meta_state = self._load_best_index(self.best_r_meta_workspace)
        self._best_nsucc_state = self._load_best_index(self.best_nsucc_workspace)
        self._best_stable_state = self._load_best_index(self.best_stable_workspace)
        # Phase A4 P4 (2026-06-09): floor-recovery archive bootstrap.
        # Always rebuild the in-memory state from disk so a supervisor
        # restart picks up an existing archive even if the feature was
        # toggled mid-experiment. The directory is only created when the
        # feature is enabled to keep default runs noise-free.
        if self.pbt_enable_best_floor_recovery_archive:
            self.best_floor_recovery_workspace.mkdir(parents=True, exist_ok=True)
        if self.best_floor_recovery_workspace.exists():
            self._best_floor_recovery_state = self._load_best_index(
                self.best_floor_recovery_workspace
            )

    def process_infos(self, infos, done_indices):
        super().process_infos(infos, done_indices)
        if "true_objective" not in infos:
            return
        done_lst = done_indices.squeeze(-1).tolist()
        if self.r_meta_window_enabled:
            for done_idx in done_lst:
                self._r_meta_window.append(float(infos["true_objective"][done_idx].item()))
        else:
            latest_val = None
            for done_idx in done_lst:
                latest_val = float(infos["true_objective"][done_idx].item())
            if latest_val is not None:
                self.curr_target_objective_value = latest_val
                self.target_objective_known = True

    def after_steps(self):
        if self.algo is None:
            return
        raw_frames = pbt_raw_training_frame(self.algo)
        population_frame = pbt_population_frame(self.algo)

        # P0: policy0 consumes CF results every step (not only on PBT restart).
        if self._is_posterior_writer and self._enable_counterfactual:
            self._maybe_consume_counterfactual_result(raw_frames)

        if self.initial_env_frames == -1:
            self.initial_env_frames = raw_frames
        if self.pbt_iteration == -1:
            self.pbt_iteration, self.last_mutation_frame = restore_pbt_tick_state_from_workspace(
                policy_workspace=self.curr_policy_workspace,
                self_pbt_iteration=self.pbt_iteration,
                last_mutation_frame=self.last_mutation_frame,
            )
        if self.pbt_iteration == -1:
            self.last_mutation_frame = raw_frames
            self.pbt_iteration = derive_pbt_iteration(raw_frames, self.interval_steps)
        elif int(self.last_mutation_frame) <= 0:
            self.last_mutation_frame = raw_frames

        derived_iteration = derive_pbt_iteration(raw_frames, self.interval_steps)
        if self.pbt_consolidation_cfg.enabled:
            should_tick, derived_iteration, skip_reason, stage, eff_interval = (
                evaluate_consolidated_tick(
                    raw_frame=raw_frames,
                    self_pbt_iteration=self.pbt_iteration,
                    last_mutation_frame=self.last_mutation_frame,
                    start_after=self.start_after,
                    initial_delay=self.initial_delay,
                    interval_steps=self.interval_steps,
                    cfg=self.pbt_consolidation_cfg,
                    curriculum_progress=self._adaptive_curriculum_progress,
                )
            )
            if self.writer is not None:
                self.writer.add_scalar(
                    "pbt_consolidation/stage",
                    float(stage_id(stage)),
                    raw_frames,
                )
                self.writer.add_scalar(
                    "pbt_consolidation/restart_blocked_count",
                    float(self._consolidation_restart_blocked_count),
                    raw_frames,
                )
                self.writer.add_scalar(
                    "pbt/effective_tick_interval",
                    float(eff_interval),
                    raw_frames,
                )
        else:
            should_tick, derived_iteration, skip_reason = evaluate_pbt_tick_schedule(
                raw_frame=raw_frames,
                self_pbt_iteration=self.pbt_iteration,
                last_mutation_frame=self.last_mutation_frame,
                start_after=self.start_after,
                initial_delay=self.initial_delay,
                interval_steps=self.interval_steps,
            )
        self._append_pbt_tick_debug(
            env_frames=raw_frames,
            population_frame=population_frame,
            derived_iteration=derived_iteration,
            should_tick=should_tick,
            skip_reason=skip_reason,
        )
        if not should_tick:
            return

        skip_pbt_selection = False
        if self.r_meta_window_enabled:
            if not self._r_meta_window:
                print(
                    f"Policy {self.policy_idx}: r_meta window empty, "
                    "observation tick only (curve+snapshot)"
                )
                skip_pbt_selection = True
            else:
                r_meta_arr = np.array(self._r_meta_window, dtype=np.float64)
                self.curr_target_objective_value = float(r_meta_arr.mean())
                # M5: also accumulate Nsucc window statistics for the curve logger
                self._last_window_size = int(r_meta_arr.size)
                self.target_objective_known = True
                if self.writer is not None:
                    self.writer.add_scalar("pbt/r_meta_agg", self.curr_target_objective_value, raw_frames)
                    self.writer.add_scalar("pbt/r_meta_count", len(self._r_meta_window), raw_frames)
                self._r_meta_window.clear()
        elif not self.target_objective_known:
            print(
                f"Policy {self.policy_idx}: not enough data, "
                "observation tick only (curve+snapshot)"
            )
            skip_pbt_selection = True

        self.pbt_iteration = derived_iteration
        if not skip_pbt_selection:
            self.target_objective_known = False

        population_frame = pbt_scheduler_sync_clock(
            self.algo,
            tick=self.interval_steps,
            lineage_id=self._lineage_id,
            lineage_start_pf=self._lineage_start_pf,
            weight_age_frame=self._weight_age_frame,
        )
        env_frames = population_frame
        self.last_mutation_frame = population_frame
        self._lineage_start_pf = int(self._lineage_start_pf or population_frame)
        self.algo._pbt_derived_iteration = derive_pbt_iteration(
            population_frame, self.interval_steps
        )
        assert_pbt_clock_invariants(self.algo, tick=self.interval_steps)

        own_state = self._save_workspace_checkpoint()
        if (
            self.frame_stop_enable
            and population_frame_target_reached(
                population_frame, self.target_population_frames, self.interval_steps
            )
        ):
            tick = max(int(self.interval_steps), 1)
            target = int(self.target_population_frames)
            effective_target = ((target + tick - 1) // tick) * tick
            payload = {
                "policy_idx": int(self.policy_idx),
                "population_frame": int(population_frame),
                "target_population_frames": target,
                "effective_target_population_frames": int(effective_target),
                "pbt_iteration": int(self.pbt_iteration),
                "completed_at": time.time(),
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime()),
                "checkpoint": str(own_state.get("checkpoint", "")),
            }
            self._write_population_frame_complete_markers(payload)
            self._frame_stop_triggered = True
            # Normal runner termination on the next epoch boundary; do not
            # write pbt_restart.json or enter source selection.
            self.algo.max_epochs = int(getattr(self.algo, "epoch_num", 0)) + 1
            if self.writer is not None:
                self.writer.add_scalar("training/frame_stop_triggered", 1.0, population_frame)
            print(
                "[PBT] population_frame_target_reached "
                f"policy={self.policy_idx} frame={population_frame} "
                f"target={self.target_population_frames}"
            )
            return
        population = self._load_population_checkpoints()
        population[self.policy_idx] = own_state
        self._update_curriculum_windows_and_control(population, env_frames)
        own_curriculum = own_state.get("curriculum", {}) or {}
        if self._is_posterior_writer and self._enable_counterfactual and self._cf_manager is not None:
            self._check_cf_trigger(population, env_frames)
        elif self._enable_wm and self._posterior_selector is not None:
            self._posterior_selector.load_or_init(policy_idx=self.policy_idx)
        self._log_cf_runtime_status(env_frames)

        # Patch 1 (rev 2): emit explicit reporting/selection scalars and
        # gap-to-best so it's easy to visually verify on TB that ranking
        # score is the metric driving PBT decisions.
        own_fsr = self._frames_since_restart(own_state)
        own_eligible_source, own_eligible_reason = self._eligible_as_source_with_reason(
            own_state, env_frames
        )
        own_eligible_elim = self._is_eligible_for_elimination(own_state, env_frames)
        self.curr_eligible_as_source = bool(own_eligible_source)
        self.curr_eligible_for_elimination = bool(own_eligible_elim)
        self.curr_eligible_source_reason = str(own_eligible_reason)
        if self.writer is not None:
            self.writer.add_scalar("pbt/r_meta_raw", float(self.curr_r_meta_raw), env_frames)
            self.writer.add_scalar("pbt/ranking_score", float(self.curr_ranking_score), env_frames)
            self.writer.add_scalar(
                "pbt/ranking_metric_mode_hash",
                float(abs(hash(self.pbt_ranking_metric)) % 100),
                env_frames,
            )
            pop_scores = [
                float(s.get("pbt_ranking_score", s.get("objective", float("nan"))))
                for s in population.values()
                if s is not None
            ]
            pop_scores = [v for v in pop_scores if not math.isnan(v) and not math.isinf(v)]
            if pop_scores:
                gap = max(pop_scores) - float(self.curr_ranking_score)
                self.writer.add_scalar("pbt/ranking_score_gap", gap, env_frames)
            # Patch 2 (rev 2): anti-churn observability. Each scalar is
            # per-tick so PBT decisions are auditable on TB.
            self.writer.add_scalar("pbt/frames_since_restart", float(own_fsr), env_frames)
            self.writer.add_scalar(
                "pbt/eligible_as_source",
                1.0 if own_eligible_source else 0.0,
                env_frames,
            )
            self.writer.add_scalar(
                "pbt/eligible_for_elimination",
                1.0 if own_eligible_elim else 0.0,
                env_frames,
            )
            self.writer.add_scalar(
                "pbt/restart_count", float(self._restart_count), env_frames
            )
            self.writer.add_scalar(
                "pbt/min_frames_as_source_after_restart",
                float(self.min_frames_as_source_after_restart),
                env_frames,
            )
            self.writer.add_scalar(
                "pbt/min_frames_between_restarts",
                float(self.min_frames_between_restarts),
                env_frames,
            )
            # Patch F (2026-05-31): per-tick observability of self-restart
            # governor (independent of whether a restart actually happens
            # at this tick). These let the analyzer plot cross/self ratios
            # over the whole training curve.
            self.writer.add_scalar(
                "pbt/self_restart_count",
                float(self._self_restart_count),
                env_frames,
            )
            self.writer.add_scalar(
                "pbt/cross_restart_count",
                float(self._cross_restart_count),
                env_frames,
            )
            self.writer.add_scalar(
                "pbt/self_restart_blocked",
                float(self._self_restart_blocked_count),
                env_frames,
            )
            self.writer.add_scalar(
                "pbt/mid_self_mutation_count",
                float(self._mid_self_mutation_count),
                env_frames,
            )
            self.writer.add_scalar(
                "pbt/disable_self_restart_flag",
                1.0 if self.disable_self_restart else 0.0,
                env_frames,
            )
            self.writer.add_scalar(
                "pbt/allow_mid_self_mutation_flag",
                1.0 if self.allow_mid_self_mutation else 0.0,
                env_frames,
            )
            self.writer.add_scalar(
                "pbt/max_self_restarts_per_policy",
                float(self.max_self_restarts_per_policy),
                env_frames,
            )
            self.writer.add_scalar(
                "pbt/min_frames_between_self_restarts",
                float(self.min_frames_between_self_restarts),
                env_frames,
            )
            # Patch G (2026-05-31): tiered source selection observability.
            self.writer.add_scalar(
                "pbt/source_selection_mode_hash",
                float(abs(hash(self.source_selection_mode)) % 100),
                env_frames,
            )
            self.writer.add_scalar(
                "pbt/recovery_restart_count",
                float(self._recovery_restart_count),
                env_frames,
            )
            self.writer.add_scalar(
                "pbt/enable_best_nsucc_recovery_flag",
                1.0 if self.enable_best_nsucc_recovery else 0.0,
                env_frames,
            )
            for k, v in self._source_tier_counts.items():
                self.writer.add_scalar(
                    f"pbt/source_tier_{k}_count_total",
                    float(v),
                    env_frames,
                )
            # Patch H (2026-05-31): expose freeze configuration + the
            # env-observed freeze field so it's auditable how aggressive
            # curriculum-tightening is at any moment.
            self.writer.add_scalar(
                "pbt/curriculum_freeze_frames",
                float(self.curriculum_freeze_frames),
                env_frames,
            )
            self.writer.add_scalar(
                "pbt/curriculum_freeze_on_recovery_frames",
                float(self.curriculum_freeze_on_recovery_frames),
                env_frames,
            )
            env_freeze_steps = int(
                own_curriculum.get("curriculum_freeze_frames_after_restart", 0) or 0
            )
            env_fsr_steps = int(own_curriculum.get("frame_since_restart", 0) or 0)
            remaining_steps = max(0, env_freeze_steps - env_fsr_steps)
            remaining_env_frames = remaining_steps * max(1, int(self.num_envs))
            frozen = env_freeze_steps > 0 and remaining_steps > 0
            self.writer.add_scalar("dexpbt/curriculum_freeze_unit", 0.0, env_frames)
            self.writer.add_scalar(
                "dexpbt/curriculum_freeze_config_env_frames",
                float(self.curriculum_freeze_frames),
                env_frames,
            )
            self.writer.add_scalar(
                "dexpbt/curriculum_freeze_remaining_steps",
                float(remaining_steps),
                env_frames,
            )
            self.writer.add_scalar(
                "dexpbt/curriculum_freeze_remaining_env_frames",
                float(remaining_env_frames),
                env_frames,
            )
            self.writer.add_scalar(
                "dexpbt/curriculum_frozen",
                1.0 if frozen else 0.0,
                env_frames,
            )
            self.writer.add_scalar(
                "pbt/env_curriculum_freeze_frames_after_restart",
                float(env_freeze_steps),
                env_frames,
            )
            self.writer.add_scalar(
                "pbt/curriculum_currently_frozen",
                1.0 if frozen else 0.0,
                env_frames,
            )
            # Patch A (rev 3, 2026-05-30): per-policy nsucc + source filter
            # thresholds so TB shows why a worker is or is not eligible.
            self.writer.add_scalar(
                "pbt/nsucc_mean", float(self.curr_nsucc_mean), env_frames
            )
            self.writer.add_scalar(
                "pbt/nsucc_max", float(self.curr_nsucc_max), env_frames
            )
            self.writer.add_scalar(
                "pbt/source_filter_nsucc_mean",
                float(self.min_nsucc_mean_as_source),
                env_frames,
            )
            self.writer.add_scalar(
                "pbt/source_filter_nsucc_max",
                float(self.min_nsucc_max_as_source),
                env_frames,
            )
            self.writer.add_scalar(
                "pbt/eligible_source_reason_hash",
                float(abs(hash(self.curr_eligible_source_reason)) % 100),
                env_frames,
            )
            # Population-wide eligible source count (audit how many workers
            # could currently serve as a source). Only `policy_idx==0`
            # writes this to avoid duplicate scalar writes from all peers.
            if self.policy_idx == 0:
                eligible_count = sum(
                    1
                    for s in population.values()
                    if s is not None
                    and self._is_eligible_as_source(s, env_frames)
                )
                self.writer.add_scalar(
                    "pbt/eligible_source_count",
                    float(eligible_count),
                    env_frames,
                )
        # PBT ranking from now on uses `pbt_ranking_score`; meta.json's
        # `"objective"` field already carries that value, so legacy sort
        # logic below continues to work without behavioural change beyond
        # what the success-aware cap produces.
        objectives = {idx: ckpt["objective"] for idx, ckpt in population.items() if ckpt is not None}
        sorted_policies = sorted(objectives.keys(), key=lambda key: objectives[key], reverse=True) if objectives else []

        # Phase A4 P4 (2026-06-09 / wired 2026-06-10): floor-recovery-aware
        # ranking (Part A+E). The shadow path always runs when the feature
        # is on; the alt ordering only replaces the legacy ordering when:
        #   * pbt_wm_score_mode == "active"
        #   * the local policy is at ``hard_floor``
        #   * no policy in the population has hard_collapse_triggered
        #   * the WM imagined eval is valid AND uncertainty <= max_unc
        #   * the recovery_pos_rate window is non-degenerate
        # Otherwise we record a fallback reason and keep ``sorted_policies``
        # untouched. Real ``objective`` values are NEVER mutated.
        sorted_policies_real = list(sorted_policies)
        sorted_policies_alt = list(sorted_policies)
        wm_changed_selection = False
        wm_score_used_now = False
        wm_fallback_reason = "feature_off"
        local_wm_unc = 1.0
        local_wm_valid = False
        floor_recovery_score_real_local = 0.0
        floor_recovery_score_wm_local = 0.0
        # Phase A4 P4 wired (2026-06-10): pull the latest WMImaginedEvaluator
        # output for each policy. train.py writes
        # ``<workspace>/wm_imagined/<pidx:03d>.json`` atomically each PPO
        # epoch. Missing / stale files keep ``imagined_valid=False`` so
        # active ranking falls back to real-only.
        try:
            self._refresh_wm_imagined_eval_from_disk(env_frames)
        except Exception as _refresh_exc:  # pragma: no cover - never crash
            print(
                f"[floor-recovery-ranking] WM imagined eval disk refresh "
                f"failed: {_refresh_exc}"
            )
        try:
            if self.pbt_floor_recovery_aware and objectives:
                # Hard-collapse population gate: any policy with hard
                # collapse forces real-only ranking (Item: WM cannot
                # override hard collapse).
                any_hard_collapse = False
                for _pidx, _ckpt in population.items():
                    if _ckpt is None:
                        continue
                    cur = _ckpt.get("curriculum") or {}
                    if bool(cur.get("hard_collapse_triggered", False)):
                        any_hard_collapse = True
                        break

                alt_objectives: dict[int, float] = {}
                wm_mode = self.pbt_wm_score_mode
                # Shadow logging zone: include every policy whose
                # tolerance is at or below the shadow zone (default 2x
                # the strict floor threshold) so we can observe
                # would-change events before reaching hard_floor. Active
                # source replacement still gates on hard_floor.
                shadow_tol_zone = max(
                    float(self.pbt_floor_tol_threshold) * 2.0, 0.04
                )
                for pidx, ckpt in population.items():
                    if ckpt is None:
                        continue
                    cur = ckpt.get("curriculum") or {}
                    stage = self._floor_stage_for(cur)
                    cur_tol = float(cur.get("success_tolerance", 1.0))
                    base_obj = float(ckpt.get("objective", float("-inf")))
                    if stage != "hard_floor" and cur_tol > shadow_tol_zone:
                        alt_objectives[pidx] = base_obj
                        continue
                    win = self._policy_window(pidx)
                    score_50, components, slope_pos = self._floor_recovery_score_for(
                        pidx, win, env_frames, hard_collapse=False
                    )
                    score_real_for_pidx = float(score_50)
                    score_with_wm = score_real_for_pidx
                    used_wm_for_pidx = False
                    fallback_reason_pidx = "wm_off"
                    if wm_mode in ("shadow", "active"):
                        wm_eval = self._wm_imagined_eval_per_policy.get(pidx) or {}
                        wm_strict = bool(
                            wm_eval.get(
                                "imagined_valid_strict",
                                wm_eval.get("imagined_valid", False),
                            )
                        )
                        wm_soft = bool(wm_eval.get("imagined_valid_soft", False))
                        wm_has_signal = bool(
                            wm_eval.get(
                                "imagined_head_has_signal",
                                wm_strict or wm_soft,
                            )
                        )
                        wm_v_weight = float(wm_eval.get("wm_validity_weight", 0.0))
                        wm_any_valid = wm_strict or wm_soft or bool(
                            wm_eval.get("imagined_valid", False)
                        )
                        wm_unc = float(wm_eval.get("uncertainty", 1.0))
                        teacher_step = float(
                            wm_eval.get("teacher_step", -1.0)
                        )
                        h_rates = wm_eval.get("horizon_pos_rate", {}) or {}
                        target_degenerate = False
                        for _k, _v in h_rates.items():
                            try:
                                vv = float(_v)
                                if vv <= 0.0 or vv >= 1.0:
                                    target_degenerate = True
                                    break
                            except (TypeError, ValueError):
                                continue
                        if not wm_any_valid or not wm_has_signal or wm_v_weight <= 0.0:
                            fallback_reason_pidx = wm_eval.get(
                                "valid_reason",
                                wm_eval.get("invalid_reason", "no_imagined_valid"),
                            )
                        else:
                            from dexpbt_isaaclab.world_model.wm_activation_gate import (
                                resolve_wm_ranking_uncertainty_gate,
                            )
                            (
                                ranking_unc_threshold,
                                ranking_gate_mode,
                                unc_fallback,
                            ) = resolve_wm_ranking_uncertainty_gate(
                                wm_strict=wm_strict,
                                wm_soft=wm_soft,
                                wm_unc=wm_unc,
                                strict_threshold=float(self.pbt_wm_max_uncertainty),
                                soft_threshold=float(
                                    self.wm_imagined_soft_uncertainty_threshold
                                ),
                            )
                            if unc_fallback is not None:
                                fallback_reason_pidx = unc_fallback
                            elif target_degenerate:
                                fallback_reason_pidx = "target_degenerate"
                            elif teacher_step < 0:
                                fallback_reason_pidx = "no_promoted_teacher"
                            else:
                                from dexpbt_isaaclab.dexpbt_utils import (
                                    combine_wm_recovery_score,
                                )
                                from dexpbt_isaaclab.world_model.wm_activation_gate import (
                                    blend_wm_ranking_score,
                                    compute_wm_pbt_source_activation_weight,
                                    effective_wm_trust_weight,
                                )
                                score_with_wm, _ = combine_wm_recovery_score(
                                    score_real_for_pidx,
                                    wm_recovery_prob=float(
                                        wm_eval.get("wm_recovery_prob", 0.0)
                                    ),
                                    wm_success_prob=float(
                                        wm_eval.get("wm_success_prob", 0.0)
                                    ),
                                    wm_drop_risk=float(
                                        wm_eval.get("wm_drop_risk", 0.0)
                                    ),
                                    wm_uncertainty=wm_unc,
                                    recovery_weight=self.pbt_wm_recovery_weight,
                                    success_weight=self.pbt_wm_success_weight,
                                    drop_penalty=self.pbt_wm_drop_penalty,
                                    uncertainty_penalty=self.pbt_wm_uncertainty_penalty,
                                )
                                (
                                    _src_w,
                                    _w_time_p,
                                    _w_unc_p,
                                    _v_w_p,
                                    _used_strict_p,
                                    _used_soft_p,
                                ) = compute_wm_pbt_source_activation_weight(
                                    enable=bool(self.wm_activation_enable),
                                    gate_mode=str(self.wm_pbt_source_gate_mode),
                                    frames=int(env_frames),
                                    start_after_frames=int(
                                        self.wm_pbt_source_start_after_frames
                                    ),
                                    ramp_frames=int(self.wm_pbt_source_ramp_frames),
                                    wm_uncertainty=float(wm_unc),
                                    max_uncertainty=float(ranking_unc_threshold),
                                    imagined_valid_strict=wm_strict,
                                    imagined_valid_soft=wm_soft,
                                    imagined_head_has_signal=wm_has_signal,
                                    wm_validity_weight=wm_v_weight,
                                    soft_validity_weight=float(
                                        self.wm_imagined_soft_validity_weight
                                    ),
                                )
                                _alpha_wm = 0.0
                                if self.wm_calibration_enable and self.algo is not None:
                                    _eb = getattr(self.algo, "_eir_bridge", None)
                                    _wcs = getattr(_eb, "wm_calibration_state", None) if _eb else None
                                    if isinstance(_wcs, dict):
                                        _alpha_wm = float(_wcs.get("alpha_wm", 0.0))
                                _eff_w = effective_wm_trust_weight(
                                    _src_w,
                                    _alpha_wm,
                                    wm_calibration_enable=bool(self.wm_calibration_enable),
                                )
                                score_with_wm = blend_wm_ranking_score(
                                    score_real_for_pidx, score_with_wm, _eff_w
                                )
                                used_wm_for_pidx = _eff_w > 0.0
                                fallback_reason_pidx = (
                                    "ok" if used_wm_for_pidx else "wm_source_activation_zero"
                                )
                                if pidx == self.policy_idx:
                                    self._last_wm_source_activation_weight = float(_src_w)
                                    self._last_wm_source_activation_nonzero = (
                                        1.0 if _src_w > 0.0 else 0.0
                                    )
                                    self._last_wm_source_unc_weight = float(_w_unc_p)
                                    self._last_wm_source_time_weight = float(_w_time_p)
                                    self._last_wm_validity_weight = float(_v_w_p)
                                    self._last_wm_score_effective_weight = float(_eff_w)
                                    self._last_wm_score_used_strict = (
                                        1.0 if _used_strict_p else 0.0
                                    )
                                    self._last_wm_score_used_soft = (
                                        1.0 if _used_soft_p else 0.0
                                    )
                                    self._last_wm_ranking_uncertainty_threshold = float(
                                        ranking_unc_threshold
                                    )
                                    self._last_wm_ranking_uncertainty_gate_mode = str(
                                        ranking_gate_mode
                                    )
                    # Both paths place the alt score on the 0..50 scale so
                    # mixing with non-floor base_obj is meaningful.
                    alt_objectives[pidx] = float(score_with_wm)
                    if pidx == self.policy_idx:
                        floor_recovery_score_real_local = float(
                            score_real_for_pidx
                        )
                        floor_recovery_score_wm_local = float(score_with_wm)
                        if used_wm_for_pidx:
                            wm_score_used_now = True
                        wm_fallback_reason = fallback_reason_pidx
                        if wm_mode in ("shadow", "active"):
                            _le = self._wm_imagined_eval_per_policy.get(pidx) or {}
                            local_wm_valid = bool(
                                _le.get("imagined_valid_strict")
                                or _le.get("imagined_valid_soft")
                                or _le.get("imagined_valid", False)
                            )
                            local_wm_unc = float(_le.get("uncertainty", 1.0))
                if alt_objectives:
                    sorted_policies_alt = sorted(
                        alt_objectives.keys(),
                        key=lambda k: alt_objectives[k],
                        reverse=True,
                    )
                    if (
                        sorted_policies_alt
                        and sorted_policies_real
                        and sorted_policies_alt[0] != sorted_policies_real[0]
                    ):
                        wm_changed_selection = True
                self._wm_shadow_observations += 1
                if wm_changed_selection:
                    self._wm_shadow_changed_selection += 1
                # Active replacement is gated by:
                #   * mode == active
                #   * no hard collapse anywhere in the population
                #   * the local policy must be in hard_floor (the alt
                #     loop already only mixes WM for those entries; if
                #     the local one isn't in hard_floor we still flip
                #     the order which can still demote / promote others,
                #     so we additionally require ``wm_score_used_now``).
                source_selected_by_wm = False
                if (
                    self.pbt_wm_score_mode == "active"
                    and not any_hard_collapse
                    and wm_score_used_now
                ):
                    objectives = dict(alt_objectives)
                    sorted_policies = list(sorted_policies_alt)
                    source_selected_by_wm = wm_changed_selection
                elif (
                    self.pbt_wm_score_mode == "active"
                    and any_hard_collapse
                ):
                    wm_fallback_reason = "hard_collapse_population"
                if self.writer is not None:
                    self.writer.add_scalar(
                        "pbt/wm_score_mode_hash",
                        float(
                            {"off": 0, "shadow": 1, "active": 2}.get(
                                self.pbt_wm_score_mode, 0
                            )
                        ),
                        env_frames,
                    )
                    self.writer.add_scalar(
                        "pbt/wm_changed_source_selection",
                        1.0 if wm_changed_selection else 0.0,
                        env_frames,
                    )
                    self.writer.add_scalar(
                        "pbt/wm_shadow_observations",
                        float(self._wm_shadow_observations),
                        env_frames,
                    )
                    self.writer.add_scalar(
                        "pbt/wm_shadow_changed_selection",
                        float(self._wm_shadow_changed_selection),
                        env_frames,
                    )
                    # Phase A4 P4 wired (2026-06-10): expose the wiring
                    # status separately from the legacy shadow tag set.
                    self.writer.add_scalar(
                        "pbt/wm_score_used",
                        1.0 if wm_score_used_now else 0.0,
                        env_frames,
                    )
                    self.writer.add_scalar(
                        "pbt/source_selected_by_wm",
                        1.0 if source_selected_by_wm else 0.0,
                        env_frames,
                    )
                    self.writer.add_scalar(
                        "pbt/wm_fallback_reason_hash",
                        float(hash(wm_fallback_reason) % 1_000_003),
                        env_frames,
                    )
                    self.writer.add_scalar(
                        "pbt/floor_recovery_score_real",
                        float(floor_recovery_score_real_local),
                        env_frames,
                    )
                    self.writer.add_scalar(
                        "pbt/floor_recovery_score_wm",
                        float(floor_recovery_score_wm_local),
                        env_frames,
                    )
                    self.writer.add_scalar(
                        "pbt/hard_collapse_population",
                        1.0 if any_hard_collapse else 0.0,
                        env_frames,
                    )
                    # PBT-WM Soft Activation Gate v1 diagnostics (local policy).
                    from dexpbt_isaaclab.world_model.wm_activation_gate import (
                        compute_wm_pbt_source_activation_weight,
                        effective_wm_trust_weight,
                        resolve_wm_ranking_uncertainty_gate,
                    )
                    _le_diag = self._wm_imagined_eval_per_policy.get(self.policy_idx) or {}
                    _diag_strict = bool(
                        _le_diag.get(
                            "imagined_valid_strict",
                            _le_diag.get("imagined_valid", False),
                        )
                    )
                    _diag_soft = bool(_le_diag.get("imagined_valid_soft", False))
                    (
                        _rank_unc_threshold,
                        _rank_gate_mode,
                        _rank_unc_fallback,
                    ) = resolve_wm_ranking_uncertainty_gate(
                        wm_strict=_diag_strict,
                        wm_soft=_diag_soft,
                        wm_unc=float(local_wm_unc),
                        strict_threshold=float(self.pbt_wm_max_uncertainty),
                        soft_threshold=float(
                            self.wm_imagined_soft_uncertainty_threshold
                        ),
                    )
                    self._last_wm_ranking_uncertainty_threshold = float(
                        _rank_unc_threshold
                    )
                    self._last_wm_ranking_uncertainty_gate_mode = str(_rank_gate_mode)
                    (
                        _src_w,
                        _w_time,
                        _w_unc,
                        _v_w,
                        _used_strict,
                        _used_soft,
                    ) = compute_wm_pbt_source_activation_weight(
                        enable=bool(self.wm_activation_enable),
                        gate_mode=str(self.wm_pbt_source_gate_mode),
                        frames=int(env_frames),
                        start_after_frames=int(self.wm_pbt_source_start_after_frames),
                        ramp_frames=int(self.wm_pbt_source_ramp_frames),
                        wm_uncertainty=float(local_wm_unc),
                        max_uncertainty=float(_rank_unc_threshold),
                        imagined_valid_strict=bool(
                            _le_diag.get(
                                "imagined_valid_strict",
                                _le_diag.get("imagined_valid", False),
                            )
                        ),
                        imagined_valid_soft=bool(
                            _le_diag.get("imagined_valid_soft", False)
                        ),
                        imagined_head_has_signal=bool(
                            _le_diag.get(
                                "imagined_head_has_signal",
                                local_wm_valid,
                            )
                        ),
                        wm_validity_weight=float(
                            _le_diag.get("wm_validity_weight", 0.0)
                        ),
                    )
                    self._last_wm_source_activation_weight = float(_src_w)
                    self._last_wm_source_activation_nonzero = 1.0 if _src_w > 0.0 else 0.0
                    self._last_wm_source_unc_weight = float(_w_unc)
                    self._last_wm_source_time_weight = float(_w_time)
                    self._last_wm_validity_weight = float(_v_w)
                    _alpha_wm_diag = 0.0
                    if self.wm_calibration_enable and self.algo is not None:
                        _eb_diag = getattr(self.algo, "_eir_bridge", None)
                        _wcs_diag = getattr(_eb_diag, "wm_calibration_state", None) if _eb_diag else None
                        if isinstance(_wcs_diag, dict):
                            _alpha_wm_diag = float(_wcs_diag.get("alpha_wm", 0.0))
                    _eff_w_diag = effective_wm_trust_weight(
                        _src_w,
                        _alpha_wm_diag,
                        wm_calibration_enable=bool(self.wm_calibration_enable),
                    )
                    self._last_wm_score_effective_weight = float(_eff_w_diag)
                    self._last_wm_score_used_strict = 1.0 if _used_strict else 0.0
                    self._last_wm_score_used_soft = 1.0 if _used_soft else 0.0
                    self.writer.add_scalar(
                        "pbt/wm_source_activation_weight", float(_src_w), env_frames
                    )
                    self.writer.add_scalar(
                        "pbt/wm_source_activation_nonzero",
                        1.0 if _src_w > 0.0 else 0.0,
                        env_frames,
                    )
                    self.writer.add_scalar(
                        "pbt/wm_source_unc_weight", float(_w_unc), env_frames
                    )
                    self.writer.add_scalar(
                        "pbt/wm_source_time_weight", float(_w_time), env_frames
                    )
                    self.writer.add_scalar(
                        "pbt/wm_score_effective_weight",
                        float(_eff_w_diag),
                        env_frames,
                    )
        except Exception as exc:  # pragma: no cover - guard against mis-init
            print(f"[floor-recovery-ranking] guarded fallback: {exc}")
            sorted_policies = list(sorted_policies_real)
            wm_fallback_reason = f"observer_exception:{type(exc).__name__}"
            if self.writer is not None:
                try:
                    self.writer.add_scalar(
                        "pbt/wm_fallback_reason_hash",
                        float(hash(wm_fallback_reason) % 1_000_003),
                        env_frames,
                    )
                except Exception:
                    pass
        # Stash for jsonl audit / curriculum control.
        self._last_wm_score_used = bool(wm_score_used_now)
        self._last_wm_fallback_reason = str(wm_fallback_reason)
        self._last_floor_recovery_score_real = float(
            floor_recovery_score_real_local
        )
        self._last_floor_recovery_score_wm = float(
            floor_recovery_score_wm_local
        )

        point_tier = "unknown"
        if sorted_policies and self.policy_idx in sorted_policies:
            rank = sorted_policies.index(self.policy_idx)
            n_top, n_mid, _ = _compute_tier_counts(self.num_policies)
            if rank < n_top:
                point_tier = "top"
            elif rank < n_top + n_mid:
                point_tier = "mid"
            else:
                point_tier = "bottom"

        # M4: emit per-tick curriculum scalars regardless of whether enough peers
        # are available for tier classification. This keeps TB curves continuous.
        own_curriculum = own_state.get("curriculum", {}) or {}
        if self.writer is not None:
            if "success_tolerance" in own_curriculum:
                self.writer.add_scalar(
                    "dexpbt/success_tolerance",
                    float(own_curriculum["success_tolerance"]),
                    env_frames,
                )
                # alias following the legacy aggregate_single_arm_suite tag name
                self.writer.add_scalar(
                    "Episode/success_tolerance",
                    float(own_curriculum["success_tolerance"]),
                    env_frames,
                )
                self.writer.add_scalar(
                    "pbt/restart_current_success_tolerance",
                    float(own_curriculum["success_tolerance"]),
                    env_frames,
                )
            if "target_tolerance" in own_curriculum:
                self.writer.add_scalar(
                    "dexpbt/target_success_tolerance",
                    float(own_curriculum["target_tolerance"]),
                    env_frames,
                )
            if "last_curriculum_update" in own_curriculum:
                self._curriculum_update_count = max(
                    self._curriculum_update_count,
                    int(own_curriculum["last_curriculum_update"]),
                )
            self.writer.add_scalar(
                "dexpbt/curriculum_update_count",
                float(self._curriculum_update_count),
                env_frames,
            )
            if "mean_prev_episode_successes" in own_curriculum:
                self.writer.add_scalar(
                    "dexpbt/mean_prev_episode_successes",
                    float(own_curriculum["mean_prev_episode_successes"]),
                    env_frames,
                )

        # M5: write per-policy training curve point + workspace-level aggregate
        self._write_training_curve_point(own_state, env_frames, own_curriculum, tier=point_tier)
        self._write_population_aggregate(population, env_frames)

        # EIR-MPC Phase D: PBT elite broadcast. Both the consume (target side)
        # and the broadcast (source side) run on EVERY PBT tick, independent of
        # whether this worker is eliminated / restarted below (constraint 19.4).
        try:
            self._consume_eir_broadcast(env_frames)
        except Exception as _eir_c_exc:  # pragma: no cover - never crash PBT
            print(f"[EIR-broadcast] consume failed: {_eir_c_exc}")
        try:
            self._maybe_broadcast_elite(population, env_frames)
        except Exception as _eir_b_exc:  # pragma: no cover - never crash PBT
            print(f"[EIR-broadcast] broadcast failed: {_eir_b_exc}")

        if skip_pbt_selection:
            return

        if len(objectives) < max(2, self.num_policies // 2):
            print(f"Policy {self.policy_idx}: only {len(objectives)} objectives available, skip")
            return

        if self.ternary_split:
            self._ternary_pbt(sorted_policies, objectives, population, env_frames)
        else:
            print(f"Policy {self.policy_idx}: non-ternary PBT not implemented in this port")

    def _ternary_pbt(
        self,
        sorted_policies: list[int],
        objectives: dict[int, float],
        population: dict[int, dict[str, Any] | None],
        env_frames: int,
    ) -> None:
        n = self.num_policies
        n_top, n_mid, n_bot = _compute_tier_counts(n)

        rank = sorted_policies.index(self.policy_idx) if self.policy_idx in sorted_policies else n - 1
        if rank < n_top:
            tier = "top"
        elif rank < n_top + n_mid:
            tier = "mid"
        else:
            tier = "bottom"

        # Patch F (2026-05-31): start each PBT tick by resetting the
        # last_restart_type to "none". The branches below set it to
        # self/cross/skipped before returning.
        self._last_restart_type = 0

        print(
            f"Policy {self.policy_idx}: DexPBT tier={tier} rank={rank}/{n} "
            f"obj={objectives.get(self.policy_idx, float('-inf')):.4f} "
            f"(split={n_top}/{n_mid}/{n_bot})"
        )
        if self.writer is not None:
            self.writer.add_scalar("pbt/dexpbt_tier_rank", float(rank), env_frames)

        self._maybe_save_best_policy(population[sorted_policies[0]])

        # Phase A4 P4 (2026-06-09): cache hard-collapse signal for the
        # floor-recovery gate downstream (independent from promising
        # protection so we keep both kill switches working).
        own_ctrl_for_floor = read_curriculum_control(
            self._curriculum_control_dir / f"{self.policy_idx:03d}.json"
        )
        own_curriculum_for_floor = (
            (population.get(self.policy_idx) or {}).get("curriculum") or {}
        )
        own_stage = self._floor_stage_for(own_curriculum_for_floor)
        self._track_first_floor_frames(self.policy_idx, env_frames, own_stage)
        own_hard_collapse_signal = bool(
            own_ctrl_for_floor is not None
            and own_ctrl_for_floor.hard_collapse_triggered
        )

        if tier == "top":
            print(f"Policy {self.policy_idx}: top tier -- keep training")
            self._maybe_enter_promising_protection(
                self.policy_idx,
                env_frames,
                self._policy_window(self.policy_idx),
            )
            self._maybe_enter_floor_recovery_protection(
                self.policy_idx,
                env_frames,
                self._policy_window(self.policy_idx),
                own_stage,
                hard_collapse=own_hard_collapse_signal,
            )
            return

        # Phase A4 P3 (2026-06-09): mid-tier promising protection update so
        # workers earn protection before they get pushed into bottom.
        late_rescue = False
        adaptive_rescue = False
        rescue_mode = "none"
        rescue_active = False
        adaptive_rescue_reason = "disabled"
        if tier == "mid":
            self._maybe_enter_promising_protection(
                self.policy_idx,
                env_frames,
                self._policy_window(self.policy_idx),
            )
            self._maybe_enter_floor_recovery_protection(
                self.policy_idx,
                env_frames,
                self._policy_window(self.policy_idx),
                own_stage,
                hard_collapse=own_hard_collapse_signal,
            )

        if tier == "bottom":
            own_window_for_rescue = self._policy_window(self.policy_idx)
            own_rescue_nsucc = float(
                own_window_for_rescue.samples[-1].nsucc_mean
                if own_window_for_rescue.samples else 0.0
            )
            own_rescue_tol = float(
                (own_curriculum_for_floor or {}).get("success_tolerance", 0.075)
            )
            late_rescue = self._late_degradation_rescue_ready(
                self.policy_idx, tolerance=own_rescue_tol,
                nsucc_mean=own_rescue_nsucc, env_frames=env_frames,
            )
            own_lineage = str(
                ((population.get(self.policy_idx) or {}).get("lineage_id"))
                or getattr(self, "_lineage_id", "")
                or ""
            )
            if self.adaptive_precision_enable:
                latch_frame_for_ready = int(
                    self._adaptive_hard_collapse_event_frame.get(self.policy_idx, -1)
                )
                ctrl_event_for_ready = -1
                if own_ctrl_for_floor is not None:
                    ctrl_event_for_ready = int(
                        getattr(
                            own_ctrl_for_floor,
                            "adaptive_probation_hard_collapse_event_frame",
                            -1,
                        )
                        or -1
                    )
                adaptive_hard_collapse_this_tick = bool(
                    latch_frame_for_ready == int(env_frames)
                    or ctrl_event_for_ready == int(env_frames)
                )
                adaptive_rescue, adaptive_rescue_reason = (
                    self._adaptive_degradation_rescue_ready(
                        policy_idx=self.policy_idx,
                        tolerance=own_rescue_tol,
                        nsucc_mean=own_rescue_nsucc,
                        env_frames=env_frames,
                        lineage_id=own_lineage,
                        hard_collapse_now=adaptive_hard_collapse_this_tick,
                    )
                )
            else:
                adaptive_rescue, adaptive_rescue_reason = False, "disabled"
            if adaptive_rescue:
                rescue_mode = "adaptive"
            elif late_rescue:
                rescue_mode = "legacy"
            else:
                rescue_mode = "none"
            rescue_active = rescue_mode != "none"
            # Keep legacy bool for legacy-only accounting; shared gates use rescue_active.
            late_rescue = bool(rescue_mode == "legacy")
            adaptive_rescue = bool(rescue_mode == "adaptive")
            if self.writer is not None:
                self.writer.add_scalar(
                    "pbt/late_degradation_rescue_triggered",
                    1.0 if late_rescue else 0.0, env_frames,
                )
                self.writer.add_scalar(
                    "pbt/late_degradation_rescue_count",
                    float(self._late_rescue_count.get(self.policy_idx, 0)),
                    env_frames,
                )
                self.writer.add_scalar(
                    "adaptive_consolidation/rescue_ready",
                    1.0 if adaptive_rescue else 0.0,
                    env_frames,
                )
                reason_hash = float(abs(hash(adaptive_rescue_reason)) % 10_000)
                self.writer.add_scalar(
                    "adaptive_consolidation/rescue_reason_hash",
                    reason_hash,
                    env_frames,
                )
            # Phase A4 P3 (2026-06-09): promising-worker anti-churn gate.
            # If THIS bottom worker is currently inside a promising
            # protection window, skip replacement entirely. Hard collapse
            # / early collapse / NaN can override (when allowed).
            protected, _proto_reason = self._is_promising_protected(
                self.policy_idx, env_frames
            )
            override_allowed = self.pbt_promising_allow_hard_collapse_override
            collapse_override = bool(
                self._collapse_freeze_until.get(self.policy_idx, 0)
                > int(env_frames)
            )
            if protected and not rescue_active and not (override_allowed and collapse_override):
                print(
                    f"Policy {self.policy_idx}: skip restart (promising "
                    f"protection until env_frames="
                    f"{self._promising_protect_until.get(self.policy_idx)})"
                )
                if self.writer is not None:
                    self.writer.add_scalar(
                        "pbt/protected_by_promising_gate", 1.0, env_frames
                    )
                    self.writer.add_scalar(
                        "pbt/promising_protect_until",
                        float(
                            self._promising_protect_until.get(
                                self.policy_idx, 0
                            )
                        ),
                        env_frames,
                    )
                self._last_restart_type = 3  # skipped
                return
            if self.writer is not None:
                self.writer.add_scalar(
                    "pbt/protected_by_promising_gate", 0.0, env_frames
                )
                self.writer.add_scalar(
                    "pbt/promising_protect_until",
                    float(
                        self._promising_protect_until.get(self.policy_idx, 0)
                    ),
                    env_frames,
                )

            # Phase A4 P4 (2026-06-09): floor-recovery anti-churn gate.
            # Hard collapse / OOM / NaN can override (when allowed by the
            # promising override flag) so that a degraded floor-recovery
            # worker still gets pulled out of bottom by collapse logic.
            fr_protected, fr_reason = self._is_floor_recovery_protected(
                self.policy_idx, env_frames
            )
            if fr_protected and not rescue_active and not (override_allowed and collapse_override):
                print(
                    f"Policy {self.policy_idx}: skip restart (floor recovery "
                    f"protection until env_frames="
                    f"{self._floor_recovery_protect_until.get(self.policy_idx)})"
                )
                if self.writer is not None:
                    self.writer.add_scalar(
                        "pbt/protected_by_floor_recovery_gate",
                        1.0,
                        env_frames,
                    )
                    self.writer.add_scalar(
                        "pbt/floor_recovery_protect_until",
                        float(
                            self._floor_recovery_protect_until.get(
                                self.policy_idx, 0
                            )
                        ),
                        env_frames,
                    )
                self._last_restart_type = 3  # skipped
                return
            if self.writer is not None:
                self.writer.add_scalar(
                    "pbt/protected_by_floor_recovery_gate",
                    0.0,
                    env_frames,
                )
                self.writer.add_scalar(
                    "pbt/floor_recovery_protect_until",
                    float(
                        self._floor_recovery_protect_until.get(
                            self.policy_idx, 0
                        )
                    ),
                    env_frames,
                )
            if self.pbt_consolidation_cfg.enabled:
                own_window = self._policy_window(self.policy_idx)
                own_nsucc = float(
                    own_window.samples[-1].nsucc_mean
                    if own_window.samples
                    else 0.0
                )
                cons_stage = resolve_pbt_stage(
                    env_frames,
                    self.pbt_consolidation_cfg,
                    curriculum_progress=self._adaptive_curriculum_progress,
                )
                own_tol = float(
                    ((population.get(self.policy_idx) or {}).get("curriculum") or {})
                    .get("success_tolerance", 0.075)
                )
                cons_ctx = BottomRestartContext(
                    promising_protected=bool(protected),
                    floor_recovery_protected=bool(fr_protected),
                    hard_collapse=bool(own_hard_collapse_signal),
                    collapse_override=bool(collapse_override),
                    nsucc_mean=own_nsucc,
                    recent_peak_nsucc_mean=float(
                        self._recent_peak_nsucc_mean.get(self.policy_idx, 0.0)
                    ),
                    tier="bottom",
                    late_degradation_rescue=rescue_active,
                )
                cons_allow, cons_reason = should_allow_bottom_restart(
                    stage=cons_stage,
                    ctx=cons_ctx,
                    cfg=self.pbt_consolidation_cfg,
                )
                if not cons_allow:
                    self._consolidation_restart_blocked_count += 1
                    print(
                        f"Policy {self.policy_idx}: skip restart "
                        f"(consolidation gate: {cons_reason}, stage={cons_stage})"
                    )
                    if self.writer is not None:
                        self.writer.add_scalar(
                            "pbt_consolidation/restart_blocked_count",
                            float(self._consolidation_restart_blocked_count),
                            env_frames,
                        )
                        self.writer.add_scalar(
                            "pbt_consolidation/stage",
                            float(stage_id(cons_stage)),
                            env_frames,
                        )
                    self._last_restart_type = 3
                    return
            own_window = self._policy_window(self.policy_idx)
            adaptive_hard_collapse_for_gate = False
            if rescue_mode == "adaptive":
                latch_frame = int(
                    self._adaptive_hard_collapse_event_frame.get(self.policy_idx, -1)
                )
                ctrl_event_frame = -1
                if own_ctrl_for_floor is not None:
                    ctrl_event_frame = int(
                        getattr(
                            own_ctrl_for_floor,
                            "adaptive_probation_hard_collapse_event_frame",
                            -1,
                        )
                        or -1
                    )
                # Same-tick only: latch or control event_frame == env_frames.
                # Stale adaptive_probation_hard_collapse bool must not arm gates.
                adaptive_hard_collapse_for_gate = bool(
                    latch_frame == int(env_frames)
                    or ctrl_event_frame == int(env_frames)
                )
                allowed, reason, d_self = pbt_copy_allowed_for_adaptive_rescue_target(
                    own_window,
                    interval_steps=self.interval_steps,
                    hard_collapse=adaptive_hard_collapse_for_gate,
                )
            else:
                allowed, reason, d_self = pbt_copy_allowed_for_target(
                    own_window,
                    interval_steps=self.interval_steps,
                )
            if self.algo is not None:
                self.algo._pbt_skill_delta = (
                    float(d_self) if d_self is not None else 0.0
                )
            if self.writer is not None:
                self.writer.add_scalar(
                    "pbt/skill_delta",
                    float(d_self) if d_self is not None else -999.0,
                    env_frames,
                )
                self.writer.add_scalar(
                    "pbt/skill_delta_valid",
                    1.0 if d_self is not None else 0.0,
                    env_frames,
                )
                if rescue_mode == "adaptive":
                    self.writer.add_scalar(
                        "pbt/adaptive_target_gate_allowed",
                        1.0 if allowed else 0.0,
                        env_frames,
                    )
                    self.writer.add_scalar(
                        "pbt/adaptive_target_skill_delta",
                        float(d_self) if d_self is not None else -999.0,
                        env_frames,
                    )
                    self.writer.add_scalar(
                        "pbt/adaptive_target_skill_delta_valid",
                        1.0 if d_self is not None else 0.0,
                        env_frames,
                    )
                    self.writer.add_scalar(
                        "pbt/adaptive_target_gate_reason_hash",
                        float(abs(hash(str(reason))) % 10_000),
                        env_frames,
                    )
                else:
                    self.writer.add_scalar(
                        "pbt/skill_delta_copy_allowed",
                        1.0 if allowed else 0.0,
                        env_frames,
                    )
            if not allowed:
                print(
                    f"Policy {self.policy_idx}: skip restart "
                    f"({'adaptive_target' if rescue_mode == 'adaptive' else 'skill_delta'} "
                    f"gate: {reason})"
                )
                self._last_restart_type = 3
                return
            if self.writer is not None:
                self.writer.add_scalar(
                    "pbt/protected_by_skill_delta_gate", 0.0, env_frames
                )
            # Patch G (2026-05-31): tiered source selection cascade.
            # Late degradation rescue filters by tolerance compatibility
            # *before* quality ranking so a slightly worse but compatible
            # source beats an incompatible top-ranked source.
            if rescue_mode == "adaptive":
                self._adaptive_rescue_attempt_count[self.policy_idx] = (
                    int(self._adaptive_rescue_attempt_count.get(self.policy_idx, 0)) + 1
                )
                decision = self._select_adaptive_rescue_source_cascade(
                    sorted_policies=sorted_policies,
                    population=population,
                    env_frames=env_frames,
                    target_tolerance=own_rescue_tol,
                    target_nsucc_mean=own_rescue_nsucc,
                    n_top=n_top,
                )
            elif rescue_mode == "legacy":
                self._late_rescue_attempt_count[self.policy_idx] = (
                    int(self._late_rescue_attempt_count.get(self.policy_idx, 0)) + 1
                )
                decision = self._select_late_degradation_rescue_source(
                    sorted_policies=sorted_policies,
                    population=population,
                    env_frames=env_frames,
                    n_top=n_top,
                    target_tolerance=own_rescue_tol,
                    target_nsucc_mean=own_rescue_nsucc,
                )
            else:
                decision = self._select_source(
                    sorted_policies=sorted_policies,
                    population=population,
                    env_frames=env_frames,
                    n_top=n_top,
                )
            self._last_source_tier_label = str(decision["tier"])
            self._last_source_checkpoint_type = str(decision["checkpoint_type"])
            tier_label = self._last_source_tier_label
            if tier_label == "best_stable_archive":
                self._best_stable_recovery_selected_count += 1
                self._log_best_stable_recovery_outcome(
                    env_frames=env_frames,
                    outcome="selected",
                    decision=decision,
                )
            if self.writer is not None:
                # Per-tick view of which tier fired (one-hot bitmask).
                tier_one_hot = (
                    "strict_live",
                    "strict_stable_live",
                    "usable_live",
                    "relaxed_live",
                    "archive",
                    "best_stable_archive",
                    "best_floor_recovery_archive",
                    "late_rescue_live",
                    "late_rescue_archive",
                    "adaptive_rescue_live",
                    "adaptive_rescue_archive",
                    "adaptive_rescue_best_stable_archive",
                    "adaptive_rescue_floor_recovery_archive",
                    "adaptive_rescue_best_nsucc_archive",
                    "skipped",
                )
                for k in tier_one_hot:
                    self.writer.add_scalar(
                        f"pbt/source_tier_{k}",
                        1.0 if tier_label == k else 0.0,
                        env_frames,
                    )
                # Cumulative tier counters survive across PBT ticks.
                for k, v in self._source_tier_counts.items():
                    self.writer.add_scalar(
                        f"pbt/source_tier_{k}_count", float(v), env_frames
                    )
                self.writer.add_scalar(
                    "pbt/source_strict_candidates_count",
                    float(len(decision["strict_candidates"])),
                    env_frames,
                )
                self.writer.add_scalar(
                    "pbt/source_relaxed_candidates_count",
                    float(len(decision["relaxed_candidates"])),
                    env_frames,
                )
                self.writer.add_scalar(
                    "pbt/source_checkpoint_is_archive",
                    1.0
                    if self._last_source_checkpoint_type
                    in (
                        "archive",
                        "best_stable_archive",
                        "best_floor_recovery_archive",
                    )
                    else 0.0,
                    env_frames,
                )
                self.writer.add_scalar(
                    "pbt/source_checkpoint_is_best_stable_archive",
                    1.0
                    if self._last_source_checkpoint_type == "best_stable_archive"
                    else 0.0,
                    env_frames,
                )
                self.writer.add_scalar(
                    "pbt/source_checkpoint_is_best_floor_recovery_archive",
                    1.0
                    if self._last_source_checkpoint_type
                    == "best_floor_recovery_archive"
                    else 0.0,
                    env_frames,
                )
                self.writer.add_scalar(
                    "pbt/usable_live_min_mean",
                    float(self.pbt_usable_live_min_mean),
                    env_frames,
                )
                self.writer.add_scalar(
                    "pbt/usable_live_min_max",
                    float(self.pbt_usable_live_min_max),
                    env_frames,
                )
                self.writer.add_scalar(
                    "pbt/usable_live_drop_ratio",
                    float(self.pbt_usable_live_max_drop_ratio),
                    env_frames,
                )
                usable_reasons = [
                    str(v)
                    for v in decision["relaxed_filter_reasons"].values()
                    if str(v).startswith("usable_")
                ]
                self.writer.add_text(
                    "pbt/usable_live_reject_reason",
                    ",".join(sorted(set(usable_reasons))) if usable_reasons else "none",
                    env_frames,
                )
                self.writer.add_scalar(
                    "pbt/source_selection_skipped_no_eligible",
                    1.0 if decision["action"] == "skip" else 0.0,
                    env_frames,
                )

            print(
                f"Policy {self.policy_idx}: source_selection tier="
                f"{tier_label} action={decision['action']} "
                f"checkpoint_type={self._last_source_checkpoint_type} "
                f"strict={decision['strict_candidates']} "
                f"relaxed={decision['relaxed_candidates']} "
                f"strict_filter={decision['strict_filter_reasons']} "
                f"relaxed_filter={decision['relaxed_filter_reasons']} "
                f"archive_reason={decision['archive_reason']}"
            )

            if decision["action"] == "skip":
                # IMPORTANT: do not call _restart_process_with_new_params;
                # the bottom worker keeps training in place.
                if tier_label in self._source_tier_counts:
                    self._source_tier_counts[tier_label] += 1
                if late_rescue:
                    self._late_rescue_blocked_count[self.policy_idx] = (
                        int(self._late_rescue_blocked_count.get(self.policy_idx, 0)) + 1
                    )
                if adaptive_rescue:
                    self._adaptive_rescue_blocked_count[self.policy_idx] = (
                        int(self._adaptive_rescue_blocked_count.get(self.policy_idx, 0)) + 1
                    )
                    self._adaptive_rescue_last_reason[self.policy_idx] = str(
                        decision.get("adaptive_rescue_reason", "no_compatible_source")
                    )
                self._last_restart_type = 3  # skipped
                return

            if _decision_uses_live_source(decision):
                src_idx = int(decision["source_policy"])
                src_ok, src_reason = pbt_copy_allowed_for_live_source(
                    self._policy_nsucc_windows.get(src_idx),
                    interval_steps=self.interval_steps,
                )
                if not src_ok:
                    print(
                        f"Policy {self.policy_idx}: skip restart "
                        f"({src_reason})"
                    )
                    if late_rescue:
                        self._late_rescue_blocked_count[self.policy_idx] = (
                            int(self._late_rescue_blocked_count.get(self.policy_idx, 0))
                            + 1
                        )
                    if adaptive_rescue:
                        self._adaptive_rescue_blocked_count[self.policy_idx] = (
                            int(self._adaptive_rescue_blocked_count.get(self.policy_idx, 0))
                            + 1
                        )
                    self._last_restart_type = 3
                    return

            source_policy = int(decision["source_policy"])
            source_state = decision["source_state"]
            assert source_state is not None

            # Profile 23: stage-aware lineage diversity + ordinary replacement budget.
            # Hard-collapse / adaptive rescue may bypass the diversity guard.
            _ap_cfg = getattr(self, "_adaptive_precision_cfg", None)
            _is_ordinary_bottom = (
                str(tier) == "bottom"
                and not bool(adaptive_rescue)
                and not bool(late_rescue)
                and not str(tier_label).startswith("adaptive_rescue")
            )
            if (
                _ap_cfg is not None
                and bool(getattr(_ap_cfg, "enabled", False))
                and _is_ordinary_bottom
            ):
                from dexpbt_isaaclab.pbt.adaptive_precision_consolidation import (
                    copy_would_drop_below_min_lineages,
                    count_distinct_lineages,
                    resolve_stage as _rs,
                    stage_min_distinct_lineages,
                    stage_replacement_budget,
                )

                c_prog = float(self._adaptive_curriculum_progress or 0.0)
                stage_name = _rs(c_prog, _ap_cfg)
                budget = int(stage_replacement_budget(stage_name, _ap_cfg))
                used = int(self._adaptive_ordinary_replacements_this_tick)
                self._last_adaptive_replacement_budget = budget
                self._last_adaptive_replacement_budget_used = used
                lineages: dict[int, str] = {}
                for pi in range(int(self.num_policies)):
                    st = population.get(pi) or {}
                    lin = str(st.get("lineage_id") or "")
                    if not lin:
                        lin = str(self._adaptive_lineage_id.get(pi, "") or "")
                    lineages[int(pi)] = lin
                distinct = count_distinct_lineages(list(lineages.values()))
                min_distinct = int(stage_min_distinct_lineages(stage_name, _ap_cfg))
                self._last_adaptive_distinct_lineage_count = distinct
                self._last_adaptive_min_distinct_lineages = min_distinct
                if used >= budget:
                    self._adaptive_replacement_budget_blocked_count += 1
                    print(
                        f"Policy {self.policy_idx}: skip restart "
                        f"(replacement_budget stage={stage_name} used={used}>={budget})"
                    )
                    if self.writer is not None:
                        self.writer.add_scalar(
                            "adaptive_consolidation/replacement_budget",
                            float(budget),
                            env_frames,
                        )
                        self.writer.add_scalar(
                            "adaptive_consolidation/replacement_budget_used",
                            float(used),
                            env_frames,
                        )
                    self._last_restart_type = 3
                    return
                if bool(getattr(_ap_cfg, "lineage_diversity_enable", False)):
                    src_lin = str(
                        (source_state.get("lineage_id") if source_state else None)
                        or lineages.get(int(source_policy), "")
                        or ""
                    )
                    if copy_would_drop_below_min_lineages(
                        lineages,
                        int(self.policy_idx),
                        src_lin,
                        min_distinct,
                    ):
                        self._last_adaptive_diversity_blocked = True
                        self._adaptive_diversity_blocked_count += 1
                        print(
                            f"Policy {self.policy_idx}: skip restart "
                            f"(diversity_blocked min={min_distinct} distinct={distinct})"
                        )
                        if self.writer is not None:
                            self.writer.add_scalar(
                                "adaptive_consolidation/diversity_blocked",
                                1.0,
                                env_frames,
                            )
                            self.writer.add_scalar(
                                "adaptive_consolidation/distinct_lineage_count",
                                float(distinct),
                                env_frames,
                            )
                            self.writer.add_scalar(
                                "adaptive_consolidation/min_distinct_lineages",
                                float(min_distinct),
                                env_frames,
                            )
                        self._last_restart_type = 3
                        return
                # Reserve budget slot for this ordinary restart attempt.
                self._adaptive_ordinary_replacements_this_tick = used + 1
                self._last_adaptive_replacement_budget_used = used + 1

            if rescue_mode == "legacy":
                source_tol = float(
                    (source_state.get("curriculum") or {}).get(
                        "success_tolerance", float("nan")
                    )
                )
                if (
                    not math.isfinite(source_tol)
                    or abs(source_tol - own_rescue_tol)
                    > self.late_degradation_rescue_max_tol_gap
                ):
                    print(
                        "Policy %s: skip restart (late_degradation_no_compatible_source)"
                        % self.policy_idx
                    )
                    self._late_rescue_blocked_count[self.policy_idx] = (
                        int(self._late_rescue_blocked_count.get(self.policy_idx, 0)) + 1
                    )
                    self._last_restart_type = 3
                    return
            source_params = source_state.get("params") or {}
            checkpoint_to_load = source_state["checkpoint"]
            if not checkpoint_to_load or not Path(str(checkpoint_to_load)).is_file():
                print(
                    f"Policy {self.policy_idx}: skip restart "
                    f"(source checkpoint missing: {checkpoint_to_load})"
                )
                if late_rescue:
                    self._late_rescue_blocked_count[self.policy_idx] = (
                        int(self._late_rescue_blocked_count.get(self.policy_idx, 0)) + 1
                    )
                if adaptive_rescue:
                    self._adaptive_rescue_blocked_count[self.policy_idx] = (
                        int(self._adaptive_rescue_blocked_count.get(self.policy_idx, 0)) + 1
                    )
                self._last_restart_type = 3
                return

            # Elimination-cooldown: block restarting THIS worker too soon
            # after its last restart, regardless of source tier. This is
            # an anti-churn guard on the *bottom worker*, distinct from
            # the source-side eligibility above.
            own_state = population.get(self.policy_idx)
            historical_archive_types = (
                "archive",
                "best_stable_archive",
                "best_floor_recovery_archive",
                "late_rescue_archive",
            )
            archive_recovery_bypass = (
                str(decision.get("checkpoint_type", "")) in historical_archive_types
                or tier_label in historical_archive_types
            ) and self.pbt_recovery_bypass_elimination_burnin
            live_late_rescue_bypass = (
                tier_label == "late_rescue_live"
                and self.late_degradation_rescue_bypass_elimination_burnin
            )
            adaptive_stable_tiers = (
                "adaptive_rescue_live",
                "adaptive_rescue_archive",
                "adaptive_rescue_best_stable_archive",
                "adaptive_rescue_floor_recovery_archive",
                "adaptive_rescue_best_nsucc_archive",
            )
            adaptive_hard_collapse_active = False
            adapt_cfg = None
            st_adapt = self._adaptive_precision_state(
                self.policy_idx, strict_local=bool(self._adaptive_precision_required())
            )
            if st_adapt is not None:
                adapt_cfg, aprob, _ = st_adapt
                latch_frame = int(
                    self._adaptive_hard_collapse_event_frame.get(self.policy_idx, -1)
                )
                ctrl_event_frame = -1
                if own_ctrl_for_floor is not None:
                    ctrl_event_frame = int(
                        getattr(
                            own_ctrl_for_floor,
                            "adaptive_probation_hard_collapse_event_frame",
                            -1,
                        )
                        or -1
                    )
                # Same-tick event only. Stale bool / residual streak after
                # rollback must not trigger emergency burn-in bypass.
                adaptive_hard_collapse_active = bool(
                    latch_frame == int(env_frames)
                    or ctrl_event_frame == int(env_frames)
                )
            frames_since_restart = int(env_frames) - int(self._frames_at_restart)
            decision_source_is_stable = bool(
                tier_label in adaptive_stable_tiers
                or str(decision.get("checkpoint_type", ""))
                in (
                    "live",
                    "best_stable_archive",
                    "best_floor_recovery_archive",
                    "archive",
                )
            )
            if adapt_cfg is not None and bool(
                adapt_cfg.rescue_emergency_require_archive_or_stable_source
            ):
                decision_source_is_stable = bool(tier_label in adaptive_stable_tiers)
            adaptive_emergency_bypass = bool(
                adaptive_rescue
                and adapt_cfg is not None
                and bool(adapt_cfg.rescue_bypass_elimination_burnin)
                and bool(adapt_cfg.rescue_bypass_on_hard_collapse)
                and adaptive_hard_collapse_active
                and frames_since_restart
                >= int(adapt_cfg.rescue_min_frames_before_emergency_bypass)
                and decision_source_is_stable
            )
            if adaptive_emergency_bypass:
                self._adaptive_rescue_emergency_bypass[self.policy_idx] = (
                    int(self._adaptive_rescue_emergency_bypass.get(self.policy_idx, 0))
                    + 1
                )
                self._adaptive_rescue_burnin_bypassed[self.policy_idx] = (
                    int(self._adaptive_rescue_burnin_bypassed.get(self.policy_idx, 0))
                    + 1
                )
            bypass_burnin = bool(
                archive_recovery_bypass
                or live_late_rescue_bypass
                or adaptive_emergency_bypass
            )
            self._late_rescue_last_burnin_bypassed = (
                1.0 if (late_rescue and bypass_burnin) else 0.0
            )
            recovery_tier = tier_label in (
                "archive",
                "best_stable_archive",
                "best_floor_recovery_archive",
                "late_rescue_archive",
                "late_rescue_live",
                "adaptive_rescue_live",
                "adaptive_rescue_archive",
                "adaptive_rescue_best_stable_archive",
                "adaptive_rescue_floor_recovery_archive",
                "adaptive_rescue_best_nsucc_archive",
            )
            if not bypass_burnin and not self._is_eligible_for_elimination(
                own_state, env_frames
            ):
                print(
                    f"Policy {self.policy_idx}: skip restart (anti-churn) -- "
                    f"frames_since_restart < min_frames_between_restarts="
                    f"{self.min_frames_between_restarts}"
                )
                if late_rescue:
                    self._late_rescue_blocked_count[self.policy_idx] = (
                        int(self._late_rescue_blocked_count.get(self.policy_idx, 0)) + 1
                    )
                    self._write_pbt_copy_event({
                        "event_type": "late_degradation_rescue_target_burnin_blocked",
                        "source_policy": source_policy,
                        "target_policy": self.policy_idx,
                        "tier": tier_label,
                        "env_frames": int(env_frames),
                    })
                if adaptive_rescue:
                    self._adaptive_rescue_blocked_count[self.policy_idx] = (
                        int(self._adaptive_rescue_blocked_count.get(self.policy_idx, 0)) + 1
                    )
                    self._adaptive_rescue_burnin_blocked[self.policy_idx] = (
                        int(
                            self._adaptive_rescue_burnin_blocked.get(self.policy_idx, 0)
                        )
                        + 1
                    )
                    self._write_pbt_copy_event({
                        "event_type": "adaptive_degradation_rescue_target_burnin_blocked",
                        "rescue_mode": "adaptive",
                        "source_policy": source_policy,
                        "target_policy": self.policy_idx,
                        "tier": tier_label,
                        "env_frames": int(env_frames),
                    })
                if tier_label == "best_stable_archive":
                    self._best_stable_recovery_skipped_count += 1
                    self._log_best_stable_recovery_outcome(
                        env_frames=env_frames,
                        outcome="skipped",
                        skipped_reason="anti_churn_elimination",
                        decision=decision,
                    )
                if self.writer is not None:
                    self.writer.add_scalar(
                        "pbt/restart_skipped_burnin", 1.0, env_frames
                    )
                self._last_restart_type = 3  # skipped
                return

            # selected_count only after all gates that can still abort without
            # attempting a restart (tol/quality already done; burn-in passed).
            if late_rescue:
                self._late_rescue_selected_count[self.policy_idx] = (
                    int(self._late_rescue_selected_count.get(self.policy_idx, 0)) + 1
                )
                source_tol = float(
                    (source_state.get("curriculum") or {}).get(
                        "success_tolerance", float("nan")
                    )
                )
                self._write_pbt_copy_event({
                    "event_type": "late_degradation_rescue_selected",
                    "source_type": str(decision.get("checkpoint_type", "live")),
                    "source_policy": source_policy,
                    "target_policy": self.policy_idx,
                    "source_tol": source_tol,
                    "target_tol": own_rescue_tol,
                    "tol_gap": abs(source_tol - own_rescue_tol),
                    "source_nsucc_mean": float(
                        source_state.get("nsucc_mean", float("nan"))
                    ),
                    "target_nsucc_mean": float(own_rescue_nsucc),
                    "source_advantage": float(
                        float(source_state.get("nsucc_mean", float("nan")))
                        - float(own_rescue_nsucc)
                    ),
                    "source_same_tol_retention": self._same_tol_retention_snapshot(
                        source_policy, source_tol
                    )["retention"] if math.isfinite(source_tol) else float("nan"),
                    "target_same_tol_retention": self._same_tol_retention_snapshot(
                        self.policy_idx, own_rescue_tol
                    )["retention"],
                    "source_control_valid": float(
                        self._late_rescue_last_source_control_valid
                    ),
                    "source_collapse_rejected": float(
                        self._late_rescue_last_source_collapse_rejected
                    ),
                    "checkpoint_type": str(decision.get("checkpoint_type", "live")),
                    "rescue_selected_count": self._late_rescue_selected_count[
                        self.policy_idx
                    ],
                    "env_frames": int(env_frames),
                })

            if adaptive_rescue:
                source_tol = float(
                    (source_state.get("curriculum") or {}).get(
                        "success_tolerance", float("nan")
                    )
                )
                src_ret = float("nan")
                tgt_ret = float("nan")
                src_adapt = self._adaptive_precision_state(int(source_policy))
                if src_adapt is not None and math.isfinite(source_tol):
                    _c, _p, tr = src_adapt
                    src_ret = float(tr.snapshot(source_tol, min_samples=1)["retention"])
                tgt_adapt = self._adaptive_precision_state(int(self.policy_idx))
                if tgt_adapt is not None:
                    _c, _p, tr = tgt_adapt
                    tgt_ret = float(tr.snapshot(own_rescue_tol, min_samples=1)["retention"])
                self._adaptive_rescue_last_source[self.policy_idx] = int(source_policy)
                self._adaptive_rescue_last_source_rank[self.policy_idx] = int(
                    decision.get("adaptive_source_rank", -1)
                )
                self._adaptive_rescue_last_advantage_sigma[self.policy_idx] = float(
                    decision.get("adaptive_source_advantage_sigma", 0.0)
                )
                self._adaptive_rescue_last_reason[self.policy_idx] = "selected"
                source_kind = str(
                    decision.get("adaptive_rescue_source_kind", "none") or "none"
                )
                self._adaptive_rescue_last_source_kind[self.policy_idx] = source_kind
                self._write_pbt_copy_event({
                    "event_type": "adaptive_degradation_rescue_selected",
                    "rescue_mode": "adaptive",
                    "adaptive_rescue_source_kind": source_kind,
                    "adaptive_rescue_fallback_to_ordinary": bool(
                        decision.get("adaptive_rescue_fallback_to_ordinary", False)
                    ),
                    "source_policy": int(source_policy),
                    "target_policy": int(self.policy_idx),
                    "source_tolerance": source_tol,
                    "target_tolerance": float(own_rescue_tol),
                    "log_tolerance_gap": float(
                        decision.get("adaptive_log_tolerance_gap", float("nan"))
                    ),
                    "source_retention": src_ret,
                    "target_retention": tgt_ret,
                    "source_rank": int(decision.get("adaptive_source_rank", -1)),
                    "source_advantage_sigma": float(
                        decision.get("adaptive_source_advantage_sigma", 0.0)
                    ),
                    "env_frames": int(env_frames),
                })
                if self.writer is not None:
                    self.writer.add_scalar(
                        "adaptive_consolidation/rescue_source_policy",
                        float(source_policy),
                        env_frames,
                    )
                    self.writer.add_scalar(
                        "adaptive_consolidation/rescue_source_rank",
                        float(decision.get("adaptive_source_rank", -1)),
                        env_frames,
                    )
                    self.writer.add_scalar(
                        "adaptive_consolidation/rescue_source_advantage_sigma",
                        float(decision.get("adaptive_source_advantage_sigma", 0.0)),
                        env_frames,
                    )
                    self.writer.add_scalar(
                        "adaptive_consolidation/rescue_attempt_count",
                        float(self._adaptive_rescue_attempt_count.get(self.policy_idx, 0)),
                        env_frames,
                    )


            if tier_label in self._source_tier_counts:
                self._source_tier_counts[tier_label] += 1

            # Snapshot before recovery / adaptive kind accounting so a failed
            # restart rolls back quota, cooldown, and selected counters.
            restart_txn_snap = self._snapshot_restart_runtime_state()

            if adaptive_rescue:
                self._adaptive_rescue_selected_count[self.policy_idx] = (
                    int(self._adaptive_rescue_selected_count.get(self.policy_idx, 0)) + 1
                )
                source_kind = str(
                    decision.get("adaptive_rescue_source_kind", "none") or "none"
                )
                if source_kind == "live":
                    self._adaptive_rescue_live_selected[self.policy_idx] = (
                        int(self._adaptive_rescue_live_selected.get(self.policy_idx, 0))
                        + 1
                    )
                elif source_kind == "archive":
                    self._adaptive_rescue_archive_selected[self.policy_idx] = (
                        int(
                            self._adaptive_rescue_archive_selected.get(self.policy_idx, 0)
                        )
                        + 1
                    )
                elif source_kind == "ordinary_fallback":
                    # Conservative: ordinary fallback still consumes adaptive
                    # quota/cooldown on success; count separately from live/archive.
                    self._adaptive_rescue_ordinary_fallback[self.policy_idx] = (
                        int(
                            self._adaptive_rescue_ordinary_fallback.get(
                                self.policy_idx, 0
                            )
                        )
                        + 1
                    )
                if self.writer is not None:
                    self.writer.add_scalar(
                        "adaptive_consolidation/rescue_selected_count",
                        float(self._adaptive_rescue_selected_count[self.policy_idx]),
                        env_frames,
                    )
                    self.writer.add_scalar(
                        "adaptive_consolidation/rescue_live_selected",
                        float(
                            self._adaptive_rescue_live_selected.get(self.policy_idx, 0)
                        ),
                        env_frames,
                    )
                    self.writer.add_scalar(
                        "adaptive_consolidation/rescue_archive_selected",
                        float(
                            self._adaptive_rescue_archive_selected.get(self.policy_idx, 0)
                        ),
                        env_frames,
                    )
                    self.writer.add_scalar(
                        "adaptive_consolidation/rescue_ordinary_fallback_selected",
                        float(
                            self._adaptive_rescue_ordinary_fallback.get(
                                self.policy_idx, 0
                            )
                        ),
                        env_frames,
                    )

            # Account based on the tier of this replacement.
            if recovery_tier:
                self._recovery_restart_count += 1
                self._last_recovery_frame = int(env_frames)
                self._last_restart_type = 4  # recovery (Patch G)
                if tier_label == "best_stable_archive":
                    self._best_stable_recovery_executed_count += 1
                    self._log_best_stable_recovery_outcome(
                        env_frames=env_frames,
                        outcome="executed",
                        decision=decision,
                    )
            else:
                self._cross_restart_count += 1
                self._last_restart_type = 2  # cross
        else:
            restart_txn_snap = self._snapshot_restart_runtime_state()
            # Patch F (2026-05-31): mid-tier self-mutation governor. The
            # legacy path mutated own params + relaunched on every tick
            # which produced 100% self-restart churn. Default behaviour now
            # is to skip self-mutation entirely; legacy path is gated
            # behind --pbt_allow_mid_self_mutation.
            self._mid_self_mutation_count += 1
            ok_self, reason_self = self._eligible_for_self_restart(env_frames)
            if not ok_self:
                self._self_restart_blocked_count += 1
                self._last_restart_type = 3  # skipped
                print(
                    f"Policy {self.policy_idx}: skip self-restart (mid tier)"
                    f" -- {reason_self} (self_restart_count="
                    f"{self._self_restart_count}, last_self_restart_frame="
                    f"{self._last_self_restart_frame})"
                )
                if self.writer is not None:
                    self.writer.add_scalar(
                        "pbt/self_restart_blocked",
                        float(self._self_restart_blocked_count),
                        env_frames,
                    )
                    self.writer.add_scalar(
                        "pbt/mid_self_mutation_count",
                        float(self._mid_self_mutation_count),
                        env_frames,
                    )
                return
            source_policy = self.policy_idx
            source_state = population[source_policy]
            assert source_state is not None
            source_params = source_state["params"]
            checkpoint_to_load = source_state["checkpoint"]
            # Will be performed below; account it here so we don't have to
            # thread state into _restart_process_with_new_params.
            self._self_restart_count += 1
            self._last_self_restart_frame = int(env_frames)
            self._last_restart_type = 1  # self

        # M1 + M2: forward source worker's curriculum snapshot so the spawned
        # process can resume the tolerance schedule. For "mid" tier the source
        # is self, which already has the latest in-memory tolerance.
        source_curriculum = dict(source_state.get("curriculum") or {})
        ckpt_type = str(self._last_source_checkpoint_type)
        tol_before = float(
            source_curriculum.get("success_tolerance", float("nan")) or float("nan")
        )
        # Phase A4 P3 (2026-06-09): source-safe inheritance dispatcher.
        # In legacy reset mode this delegates to the legacy resolver
        # bit-for-bit and the source-safe consolidation re-arm is skipped
        # below. Mid-tier self-restart (source==self) bypasses the
        # dispatcher and keeps the legacy resolver to avoid degenerate
        # self->self inheritance accounting.
        own_state = population.get(self.policy_idx) or {}
        own_curriculum = dict(own_state.get("curriculum") or {})
        target_current_tol = float(
            own_curriculum.get("success_tolerance", float("nan")) or float("nan")
        )
        if tier == "bottom":
            inherit_audit = self._resolve_curriculum_inheritance(
                source_policy=source_policy,
                source_state=source_state,
                source_curriculum=source_curriculum,
                checkpoint_type=ckpt_type,
                target_current_tolerance=target_current_tol,
            )
            tol_after = float(inherit_audit["tol_after"])
            tol_mode = str(inherit_audit["legacy_mode"])
        else:
            legacy_tol_v, legacy_mode_v = self._resolve_restart_success_tolerance(
                source_policy=source_policy,
                source_curriculum=source_curriculum,
                checkpoint_type=ckpt_type,
            )
            tol_after = float(legacy_tol_v)
            tol_mode = str(legacy_mode_v)
            inherit_audit = {
                "tol_after": tol_after,
                "legacy_tol": tol_after,
                "legacy_mode": tol_mode,
                "inheritance_mode": str(self.pbt_curriculum_inherit_mode),
                "inheritance_reason": "mid_self_restart_legacy",
                "inherited_tolerance": 0.0,
                "target_tolerance_before_copy": target_current_tol
                if not (
                    isinstance(target_current_tol, float)
                    and math.isnan(target_current_tol)
                )
                else 0.0,
                "source_safe_for_curriculum": False,
                "source_meta": {
                    "source_success_tolerance": float("nan"),
                    "has_source_success_tolerance": False,
                    "source_nsucc_mean": float("nan"),
                    "source_nsucc_max": float("nan"),
                    "source_retention": float("nan"),
                    "has_source_retention": False,
                },
                "ckpt_tol": tol_after,
            }
        source_curriculum["success_tolerance"] = float(tol_after)
        # Update inheritance counters (post-hoc reporting only).
        if inherit_audit["inheritance_reason"] == "missing_source_tolerance":
            self._curriculum_missing_metadata_count += 1
        if inherit_audit["source_safe_for_curriculum"]:
            self._curriculum_inherited_count += 1
        else:
            if self.pbt_curriculum_inherit_mode == "reset":
                self._curriculum_reset_count += 1
        # Phase A4 P3: source-aware recovery consolidation re-arm. Only
        # active under source_safe inherit mode. We always clear the stale
        # target-slot consolidation; if the source is stable we use 50M,
        # otherwise 100M (configurable).
        recovery_consolidation_source = ""
        recovery_consolidation_frames = 0
        source_safe_apply: dict[str, Any] | None = None
        # Mid-tier self-restart copies from self -> self; treat it like a
        # legacy event for inheritance purposes. The bottom-tier branch is
        # the only one that crosses worker boundaries.
        is_cross_or_recovery = tier == "bottom"
        if (
            self.pbt_curriculum_inherit_mode == "source_safe"
            and is_cross_or_recovery
        ):
            if inherit_audit["source_safe_for_curriculum"]:
                recovery_consolidation_frames = int(
                    self.pbt_source_safe_recovery_consolidation_frames
                )
                recovery_consolidation_source = "recovery_consolidation_source_safe"
            elif (
                inherit_audit["inheritance_reason"]
                == "missing_source_tolerance"
            ):
                recovery_consolidation_frames = int(
                    self.pbt_source_unsafe_recovery_consolidation_frames
                )
                recovery_consolidation_source = (
                    "recovery_consolidation_missing_metadata"
                )
            else:
                recovery_consolidation_frames = int(
                    self.pbt_source_unsafe_recovery_consolidation_frames
                )
                recovery_consolidation_source = (
                    "recovery_consolidation_source_unsafe"
                )
            def _coerce_finite_audit(v: Any) -> float:
                try:
                    f = float(v)
                except Exception:
                    return 0.0
                if math.isnan(f) or math.isinf(f):
                    return 0.0
                return f

            ctrl_path = (
                self._curriculum_control_dir / f"{int(self.policy_idx):03d}.json"
            )
            audit_payload = {
                "inheritance_mode": str(inherit_audit["inheritance_mode"]),
                "inheritance_reason": str(inherit_audit["inheritance_reason"]),
                "inherited_tolerance": _coerce_finite_audit(
                    inherit_audit["inherited_tolerance"]
                ),
                "target_tolerance_before_copy": _coerce_finite_audit(
                    inherit_audit["target_tolerance_before_copy"]
                ),
                "target_tolerance_after_copy": _coerce_finite_audit(tol_after),
                "source_policy_idx_for_copy": int(source_policy),
                "source_checkpoint_type_for_copy": str(ckpt_type),
                "source_success_tolerance_for_copy": _coerce_finite_audit(
                    inherit_audit["source_meta"]["source_success_tolerance"]
                ),
                "source_nsucc_mean_for_copy": _coerce_finite_audit(
                    inherit_audit["source_meta"]["source_nsucc_mean"]
                ),
                "source_nsucc_max_for_copy": _coerce_finite_audit(
                    inherit_audit["source_meta"]["source_nsucc_max"]
                ),
                "source_retention_for_copy": _coerce_finite_audit(
                    inherit_audit["source_meta"]["source_retention"]
                ),
                "source_safe_for_curriculum": bool(
                    inherit_audit["source_safe_for_curriculum"]
                ),
                "strong_live_inherit_exact": bool(
                    inherit_audit.get("strong_live_inherit_exact", False)
                ),
            }
            # Defer control JSON mutation until inside the restart transaction
            # so a failed manifest/checkpoint path can restore raw bytes.
            source_safe_apply = {
                "path": ctrl_path,
                "consolidation_env_frames": recovery_consolidation_frames,
                "consolidation_source": recovery_consolidation_source,
                "num_envs": int(self.num_envs),
                "inherited_tolerance": float(tol_after),
                "inheritance_audit": audit_payload,
                "require_existing": bool(late_rescue),
            }
        # Persist a copy-event record for downstream analysis. Note that
        # ``tier_label`` is only defined when the bottom-tier source-selection
        # cascade ran; mid-tier self-restarts log a synthetic label.
        _event_tier_label = (
            str(tier_label) if tier == "bottom" else "mid_self_restart"  # type: ignore[name-defined]
        )

        def _coerce_finite(v: Any) -> float:
            try:
                f = float(v)
            except Exception:
                return 0.0
            if math.isnan(f) or math.isinf(f):
                return 0.0
            return f

        smeta = inherit_audit["source_meta"]
        self._write_pbt_copy_event(
            {
                "env_frames": int(env_frames),
                "tier": _event_tier_label,
                "tier_top_level": str(tier),
                "curriculum_inherit_mode": str(self.pbt_curriculum_inherit_mode),
                "source_policy_idx": int(source_policy),
                "source_checkpoint_type": str(ckpt_type),
                "source_success_tolerance": _coerce_finite(
                    smeta["source_success_tolerance"]
                ),
                "source_nsucc_mean": _coerce_finite(smeta["source_nsucc_mean"]),
                "source_nsucc_max": _coerce_finite(smeta["source_nsucc_max"]),
                "source_retention": _coerce_finite(smeta["source_retention"]),
                "source_safe_for_curriculum": bool(
                    inherit_audit["source_safe_for_curriculum"]
                ),
                "target_tolerance_before_copy": _coerce_finite(
                    inherit_audit["target_tolerance_before_copy"]
                ),
                "target_tolerance_after_copy": _coerce_finite(tol_after),
                "inherited_tolerance": _coerce_finite(
                    inherit_audit["inherited_tolerance"]
                ),
                "inheritance_reason": str(
                    inherit_audit["inheritance_reason"]
                ),
                "strong_live_inherit_exact": bool(
                    inherit_audit.get("strong_live_inherit_exact", False)
                ),
                "legacy_tolerance_mode": str(tol_mode),
                "legacy_tolerance_value": _coerce_finite(
                    inherit_audit["legacy_tol"]
                ),
                "recovery_consolidation_frames_selected": int(
                    recovery_consolidation_frames
                ),
                "recovery_consolidation_source": str(
                    recovery_consolidation_source
                ),
                "protected_by_promising_gate": bool(
                    self._is_promising_protected(self.policy_idx, env_frames)[0]
                ),
                "promising_protect_until": int(
                    self._promising_protect_until.get(self.policy_idx, 0)
                ),
            }
        )
        if self.writer is not None:
            tol_mode_idx = float(
                ["checkpoint", "last_mastered", "relax_one_level"].index(tol_mode)
            )
            if ckpt_type in ("archive", "best_stable_archive"):
                self.writer.add_scalar(
                    "pbt/recovery_tolerance_mode", tol_mode_idx, env_frames
                )
                self.writer.add_scalar(
                    "pbt/recovery_tolerance_before", tol_before, env_frames
                )
                self.writer.add_scalar(
                    "pbt/recovery_tolerance_after", float(tol_after), env_frames
                )
                self.writer.add_scalar(
                    "pbt/recovery_last_mastered_tolerance",
                    self._read_last_mastered_tolerance(source_policy),
                    env_frames,
                )
            else:
                self.writer.add_scalar(
                    "pbt/crosscopy_tolerance_mode", tol_mode_idx, env_frames
                )
                self.writer.add_scalar(
                    "pbt/crosscopy_tolerance_before", tol_before, env_frames
                )
                self.writer.add_scalar(
                    "pbt/crosscopy_tolerance_after", float(tol_after), env_frames
                )
                self.writer.add_scalar(
                    "pbt/crosscopy_last_mastered_tolerance",
                    self._read_last_mastered_tolerance(source_policy),
                    env_frames,
                )

        mutation_frozen = int(env_frames) < self._mutation_frozen_until.get(
            self.policy_idx, 0
        )
        if rescue_active or mutation_frozen:
            # Rescue restores a compatible policy; never mutate params.
            new_params = dict(source_params)
        else:
            cons_stage = resolve_pbt_stage(
                env_frames,
                self.pbt_consolidation_cfg,
                curriculum_progress=self._adaptive_curriculum_progress,
            )
            eff_mut_rate = effective_mutation_rate(
                self.mutation_rate, cons_stage, self.pbt_consolidation_cfg
            )
            new_params = _mutate_params(
                source_params, eff_mut_rate, self.change_min, self.change_max
            )
        # Keep all bounded parameters inside paper-aligned safe ranges even when
        # inherited from an already-diverged source policy.
        for k in list(new_params.keys()):
            new_params[k] = _clamp_param(k, new_params[k])
        src_state = population.get(source_policy) or {}
        source_weight_age = int(
            src_state.get(
                "weight_age_frame",
                src_state.get("frame", env_frames),
            )
            or 0
        )
        # Transaction owner is `_restart_process_with_new_params`. Caller only
        # reacts to the boolean; do NOT restore/cleanup/emit rollback here.
        restart_ok = self._restart_process_with_new_params(
            new_params,
            checkpoint_to_load,
            source_policy,
            env_frames,
            tier,
            source_curriculum=source_curriculum,
            source_weight_age_frame=source_weight_age,
            late_rescue=bool(late_rescue),
            late_rescue_decision=decision if late_rescue else None,
            adaptive_rescue=bool(adaptive_rescue),
            adaptive_rescue_decision=decision if adaptive_rescue else None,
            runtime_snap=restart_txn_snap,
            source_safe_apply=source_safe_apply,
        )
        if not restart_ok:
            if late_rescue:
                self._late_rescue_blocked_count[self.policy_idx] = (
                    int(self._late_rescue_blocked_count.get(self.policy_idx, 0)) + 1
                )
                self._safe_write_pbt_copy_event({
                    "event_type": "late_degradation_rescue_restart_failed",
                    "source_policy": source_policy,
                    "target_policy": self.policy_idx,
                    "env_frames": int(env_frames),
                })
            if adaptive_rescue:
                self._adaptive_rescue_blocked_count[self.policy_idx] = (
                    int(self._adaptive_rescue_blocked_count.get(self.policy_idx, 0)) + 1
                )
                self._safe_write_pbt_copy_event({
                    "event_type": "adaptive_degradation_rescue_restart_failed",
                    "rescue_mode": "adaptive",
                    "source_policy": source_policy,
                    "target_policy": self.policy_idx,
                    "env_frames": int(env_frames),
                })
            self._last_restart_type = 3
            return

    def _policy_run_dir(self) -> Path:
        """Supervisor RUN_DIR for this worker: ``.../policy_XX``."""
        return self.workspace.parent / f"policy_{self.policy_idx:02d}"

    def _population_frame_complete_marker_paths(self) -> tuple[Path, Path]:
        """Return (workspace policy marker, supervisor policy_run_dir marker)."""
        ws_marker = self.curr_policy_workspace / "population_frame_complete.json"
        run_marker = self._policy_run_dir() / "population_frame_complete.json"
        return ws_marker, run_marker

    @staticmethod
    def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
        tmp.write_text(json.dumps(payload, indent=2))
        tmp.replace(path)

    def _write_population_frame_complete_markers(self, payload: dict[str, Any]) -> None:
        ws_marker, run_marker = self._population_frame_complete_marker_paths()
        self._atomic_write_json(ws_marker, payload)
        self._atomic_write_json(run_marker, payload)

    def _policy_checkpoint_prefix(self, iteration: int) -> str:
        return f"{iteration:06d}"

    def _read_curriculum_state(self) -> dict[str, float | int]:
        """Read DexPBT curriculum / task state from the underlying env.

        These fields are NOT part of rl_games' algo checkpoint, but are required
        to correctly resume after a PBT restart (otherwise tolerance resets to
        the cfg default ~0.075 even for a worker copied from a near-target peer).
        Returns dict with NaN/None on failure (resume side tolerates missing keys).
        """
        env = _get_raw_env(self.algo)
        out: dict[str, float | int] = {}
        if env is None:
            return out
        for k in (
            "success_tolerance",
            "initial_tolerance",
            "target_tolerance",
            "last_curriculum_update",
            "frame_since_restart",
            # Patch H (2026-05-31): env-side curriculum freeze counter.
            "curriculum_freeze_frames_after_restart",
        ):
            try:
                v = getattr(env, k, None)
                if v is None:
                    continue
                # success_tolerance etc may be python float or tensor — coerce to scalar
                if hasattr(v, "item"):
                    v = v.item()
                out[k] = float(v) if not isinstance(v, int) else int(v)
            except Exception:
                continue
        # Also surface mean of prev_episode_successes for logging only
        try:
            pes = getattr(env, "prev_episode_successes", None)
            if pes is not None and hasattr(pes, "mean"):
                out["mean_prev_episode_successes"] = float(pes.mean().item())
                out["max_prev_episode_successes"] = float(pes.max().item())
        except Exception:
            pass
        return out

    def _eir_bridge_and_cfg(self):
        """Return ``(eir_bridge, broadcast_cfg)`` or ``(None, None)``."""
        bridge = getattr(self.algo, "_eir_bridge", None)
        if bridge is None:
            return None, None
        cfg = getattr(bridge, "broadcast_cfg", None)
        if cfg is None or not getattr(cfg, "enabled", False):
            return bridge, None
        return bridge, cfg

    def _pop_nsucc_means(
        self, population: dict[int, dict[str, Any] | None]
    ) -> dict[int, float]:
        out: dict[int, float] = {}
        for idx, st in population.items():
            if st is None:
                continue
            try:
                out[int(idx)] = float(st.get("nsucc_mean", 0.0) or 0.0)
            except (TypeError, ValueError):
                out[int(idx)] = 0.0
        return out

    def _maybe_broadcast_elite(
        self, population: dict[int, dict[str, Any] | None], env_frames: int
    ) -> None:
        """Source side: when this worker is a strong recovery performer, write a
        broadcast payload (top bank entries + residual head + scorer) into the
        bottom-k workers' mailboxes (constraint 19.4)."""
        from dexpbt_isaaclab.world_model import eir_broadcast as _bc

        bridge, cfg = self._eir_bridge_and_cfg()
        if bridge is None or cfg is None:
            return
        bank = getattr(bridge, "bank", None)
        bank_size = int(getattr(bank, "size", 0)) if bank is not None else 0
        pop_nsucc = self._pop_nsucc_means(population)
        pop_vals = list(pop_nsucc.values())
        pop_mean = float(sum(pop_vals) / len(pop_vals)) if pop_vals else 0.0
        source_nsucc = float(self.curr_nsucc_mean)

        do_bc = _bc.should_broadcast(
            source_nsucc, pop_mean, cfg=cfg, source_bank_size=bank_size
        )
        source_cooled = _bc.source_broadcast_allowed(
            self._eir_last_source_broadcast_frame, env_frames, cfg=cfg
        )
        if self.writer is not None:
            self.writer.add_scalar(
                "eir_pbt/broadcast_eligible", 1.0 if do_bc else 0.0, env_frames
            )
            self.writer.add_scalar(
                "eir_pbt/broadcast_source_cooled",
                1.0 if source_cooled else 0.0,
                env_frames,
            )
            self.writer.add_scalar("eir_pbt/source_bank_size", float(bank_size), env_frames)
        if not do_bc:
            return
        if not source_cooled:
            if self.writer is not None:
                self.writer.add_scalar("eir_pbt/broadcast_cooldown_blocked", 1.0, env_frames)
            return

        targets = _bc.select_broadcast_targets(
            self.policy_idx, pop_nsucc, cfg=cfg
        )
        targets = [
            t for t in targets
            if _bc.target_broadcast_allowed(self.workspace, t, env_frames, cfg=cfg)
        ]
        if not targets:
            if self.writer is not None:
                self.writer.add_scalar("eir_pbt/broadcast_cooldown_blocked", 1.0, env_frames)
            return
        if self.writer is not None:
            self.writer.add_scalar("eir_pbt/broadcast_cooldown_blocked", 0.0, env_frames)
        payload = _bc.build_broadcast_payload(
            bridge,
            cfg=cfg,
            source_idx=int(self.policy_idx),
            source_nsucc_mean=source_nsucc,
            source_nsucc_max=float(self.curr_nsucc_max),
            frame=int(env_frames),
        )
        n_written = 0
        for tgt in targets:
            try:
                _bc.write_broadcast(self.workspace, tgt, payload)
                n_written += 1
                _bc.append_broadcast_event(
                    self._metrics_dir,
                    {
                        "event": "broadcast",
                        "source": int(self.policy_idx),
                        "target": int(tgt),
                        "frame": int(env_frames),
                        "entries": int(payload.get("bank_entries", 0)),
                        "source_nsucc_mean": source_nsucc,
                        "pop_mean": pop_mean,
                    },
                )
            except Exception as exc:  # pragma: no cover
                print(f"[EIR-broadcast] write to policy {tgt} failed: {exc}")
        if n_written > 0:
            self._eir_last_source_broadcast_frame = int(env_frames)
        self._eir_broadcast_count += n_written
        if self.writer is not None:
            self.writer.add_scalar(
                "eir_pbt/broadcast_count", float(self._eir_broadcast_count), env_frames
            )
            self.writer.add_scalar(
                "eir_pbt/broadcast_source_policy", float(self.policy_idx), env_frames
            )
            self.writer.add_scalar(
                "eir_pbt/broadcast_bank_entries",
                float(payload.get("bank_entries", 0)),
                env_frames,
            )
        print(
            f"[EIR-broadcast] policy {self.policy_idx} (nsucc_mean={source_nsucc:.1f}) "
            f"broadcast {payload.get('bank_entries', 0)} bank entries to {targets}"
        )

    def _consume_eir_broadcast(self, env_frames: int) -> None:
        """Target side: live-merge any payload addressed to this policy into the
        running bridge each PBT tick (constraint 19.4)."""
        from dexpbt_isaaclab.world_model import eir_broadcast as _bc

        bridge, cfg = self._eir_bridge_and_cfg()
        if cfg is None:
            return
        meta = _bc.read_broadcast(self.workspace, self.policy_idx)
        consumed = 0.0
        if self.writer is not None:
            self.writer.add_scalar("eir_pbt/broadcast_pending", 1.0 if meta else 0.0, env_frames)
        if meta is None:
            return
        mode = _bc.decide_consume_mode(bridge is not None)
        if mode == _bc.CONSUME_LIVE:
            payload = _bc.load_payload(meta)
            summary = (
                _bc.merge_payload_into_bridge(bridge, payload, cfg=cfg)
                if payload is not None else {"merged_entries": 0}
            )
            _bc.mark_consumed(
                self.workspace, self.policy_idx,
                frame=int(meta.get("frame", 0)),
                source_policy=int(meta.get("source_policy", -1)),
                mode=mode,
                merged_entries=int(summary.get("merged_entries", 0)),
            )
            self._eir_bank_merge_count += int(summary.get("merged_entries", 0))
            consumed = 1.0
            _bc.append_broadcast_event(
                self._metrics_dir,
                {
                    "event": "consume",
                    "source": int(meta.get("source_policy", -1)),
                    "target": int(self.policy_idx),
                    "frame": int(env_frames),
                    "entries": int(summary.get("merged_entries", 0)),
                    "mode": mode,
                    "consumed": True,
                },
            )
            if self.writer is not None:
                self.writer.add_scalar(
                    "eir_pbt/broadcast_consumed_source_policy",
                    float(meta.get("source_policy", -1)), env_frames,
                )
                self.writer.add_scalar(
                    "eir_pbt/broadcast_consumed_bank_entries",
                    float(summary.get("merged_entries", 0)), env_frames,
                )
                self.writer.add_scalar(
                    "eir_pbt/bank_merge_count",
                    float(self._eir_bank_merge_count), env_frames,
                )
                self.writer.add_scalar(
                    "eir_pbt/residual_head_copied",
                    1.0 if summary.get("residual_copied") else 0.0, env_frames,
                )
            print(
                f"[EIR-broadcast] policy {self.policy_idx} consumed "
                f"{summary.get('merged_entries', 0)} entries from policy "
                f"{meta.get('source_policy')} (mode={mode})"
            )
        else:
            # forced-restart hand-off: leave the payload for the next process
            # incarnation to consume on startup; record the deferral.
            _bc.append_broadcast_event(
                self._metrics_dir,
                {
                    "event": "consume",
                    "source": int(meta.get("source_policy", -1)),
                    "target": int(self.policy_idx),
                    "frame": int(env_frames),
                    "entries": 0,
                    "mode": mode,
                    "consumed": False,
                },
            )
        if self.writer is not None:
            self.writer.add_scalar("eir_pbt/broadcast_consumed", consumed, env_frames)

    def _save_workspace_checkpoint(self) -> dict[str, Any]:
        prefix = self._policy_checkpoint_prefix(self.pbt_iteration)
        checkpoint_path = self.curr_policy_workspace / f"{prefix}.pth"
        ckpt_tmp = self.curr_policy_workspace / f"{prefix}.pth.tmp"
        meta_tmp = self.curr_policy_workspace / f"{prefix}.json.tmp"
        meta_path = self.curr_policy_workspace / f"{prefix}.json"
        algo_state = self.algo.get_full_state_weights()
        erd_bridge = getattr(self.algo, "_erd_bridge", None)
        erd_scorer = getattr(erd_bridge, "scorer", None) if erd_bridge is not None else None
        if erd_scorer is not None:
            algo_state["erd_scorer"] = erd_scorer.state_dict()
            algo_state["erd_scorer_warmup_iters_done"] = int(
                getattr(erd_bridge, "scorer_warmup_iters_done", 0)
            )
        # EIR-MPC (Phase D): persist scorer / residual head / scorer warmup and a
        # priority-capped slice of the elite bank INLINE in the .pth so a PBT
        # restart that copies the source .pth inherits the source's recovery
        # memory automatically (train.py `_wm_safe_restore` reloads these keys).
        eir_bridge = getattr(self.algo, "_eir_bridge", None)
        if eir_bridge is not None:
            eir_scorer = getattr(eir_bridge, "scorer", None)
            if eir_scorer is not None:
                algo_state["eir_scorer"] = eir_scorer.state_dict()
                algo_state["eir_scorer_warmup_iters_done"] = int(
                    getattr(eir_bridge, "scorer_warmup_iters_done", 0)
                )
            _eir_rb = getattr(eir_bridge, "residual_bridge", None)
            _eir_head = getattr(_eir_rb, "head", None) if _eir_rb is not None else None
            if _eir_head is not None:
                algo_state["eir_residual_head"] = _eir_head.state_dict()
            _eir_bank = getattr(eir_bridge, "bank", None)
            if _eir_bank is not None and getattr(_eir_bank, "size", 0) > 0:
                _bcfg = getattr(eir_bridge, "broadcast_cfg", None)
                _cap = int(getattr(_bcfg, "max_bank_merge_entries", 50000)) if _bcfg else 50000
                try:
                    algo_state["eir_elite_bank"] = _eir_bank.top_entries_state(_cap)
                except Exception as _eir_bank_exc:  # pragma: no cover
                    print(f"[EIR] checkpoint bank export failed: {_eir_bank_exc}")
            _eir_snippet_bank = getattr(eir_bridge, "snippet_bank", None)
            if _eir_snippet_bank is not None and getattr(_eir_snippet_bank, "size", 0) > 0:
                _bcfg = getattr(eir_bridge, "broadcast_cfg", None)
                _cap = int(getattr(_bcfg, "max_bank_merge_entries", 50000)) if _bcfg else 50000
                try:
                    algo_state["eir_elite_snippet_bank"] = _eir_snippet_bank.top_entries_state(
                        _cap
                    )
                except Exception as _eir_sb_exc:  # pragma: no cover
                    print(f"[EIR] checkpoint snippet bank export failed: {_eir_sb_exc}")
            _wm_cal = getattr(eir_bridge, "wm_calibration_state", None)
            if isinstance(_wm_cal, dict) and _wm_cal:
                algo_state["wm_calibration_state"] = dict(_wm_cal)
            _ra_st = getattr(eir_bridge, "readiness_adaptive_state", None)
            if _ra_st is not None:
                if isinstance(_ra_st, dict):
                    algo_state["readiness_adaptive_state"] = dict(_ra_st)
                elif hasattr(_ra_st, "to_dict"):
                    algo_state["readiness_adaptive_state"] = _ra_st.to_dict()
        awr_bridge = getattr(self.algo, "_awr_bridge", None)
        sidecar_saved_path: Path | None = None
        if awr_bridge is not None and getattr(awr_bridge, "buffer", None) is not None:
            from dexpbt_isaaclab.awr.elite_awr import save_awr_sidecar_topk

            sidecar_rel = f"awr_buffers/policy_{self.policy_idx:02d}_awr_topk.pt"
            sidecar_path = self.workspace / sidecar_rel
            ok, awr_meta = save_awr_sidecar_topk(
                awr_bridge.buffer,
                sidecar_path,
                step_budget=int(getattr(awr_bridge.cfg, "copy_top_k_steps", 50_000)),
                broadcast_step_budget=0,
            )
            if ok:
                algo_state["awr_buffer_ref"] = sidecar_rel
                sidecar_saved_path = sidecar_path
            else:
                algo_state["awr_buffer_ref"] = None
            algo_state["awr_buffer_ref_meta"] = awr_meta
            try:
                if self.writer is not None:
                    env_frames = int(getattr(self.algo, "frame", 0) or 0)
                    self.writer.add_scalar(
                        "awr/sidecar_num_sequences",
                        float(awr_meta.get("num_sequences", 0)),
                        env_frames,
                    )
                    self.writer.add_scalar(
                        "awr/sidecar_num_steps",
                        float(awr_meta.get("num_steps", 0)),
                        env_frames,
                    )
                    self.writer.add_scalar(
                        "awr/sidecar_size_mb",
                        float(awr_meta.get("size_mb", 0.0)),
                        env_frames,
                    )
                    self.writer.add_scalar(
                        "awr/sidecar_save_failed",
                        float(awr_meta.get("save_failed", 0.0)),
                        env_frames,
                    )
            except Exception:
                pass
        curriculum = self._read_curriculum_state()
        population_frame = pbt_population_frame(self.algo)
        lineage_start_pf = int(self._lineage_start_pf or population_frame)
        weight_age_frame = int(self._weight_age_frame) + derive_lineage_frame(
            population_frame, lineage_start_pf
        )
        self.algo._pbt_weight_age_frame = int(weight_age_frame)
        self.algo._pbt_lineage_start_pf = int(lineage_start_pf)

        # ------------------------------------------------------------------
        # Patch 1 (rev 2, 2026-05-30): compute success-aware PBT ranking
        # score and persist both `r_meta_raw` and `pbt_ranking_score`.
        #
        # `curr_target_objective_value` is set upstream by `process_infos`
        # or `after_steps` from the env's `true_objective` (the legacy
        # DexPBT formula). We treat it as `r_meta_raw` (reporting-only)
        # and compute a separate `pbt_ranking_score` that is robust to
        # tolerance-progress dominating r_meta when nsucc=0.
        #
        # For backward compatibility we also write `"objective"` = the
        # ranking score (sort/tier code reads this key). The new
        # `"r_meta_raw"` field carries the original value.
        # ------------------------------------------------------------------
        r_meta_raw = float(self.curr_target_objective_value)
        nsucc_mean = float(curriculum.get("mean_prev_episode_successes", 0.0) or 0.0)
        nsucc_max = float(curriculum.get("max_prev_episode_successes", 0.0) or 0.0)
        ranking_score = compute_pbt_ranking_score(
            r_meta_raw,
            nsucc_mean,
            mode=self.pbt_ranking_metric,
            nsucc_zero_cap=self.pbt_nsucc_zero_cap,
            nsucc_low_threshold=self.pbt_nsucc_low_threshold,
            nsucc_low_cap=self.pbt_nsucc_low_cap,
        )
        self.curr_r_meta_raw = r_meta_raw
        self.curr_ranking_score = ranking_score
        self.curr_nsucc_mean = nsucc_mean
        self.curr_nsucc_max = nsucc_max

        # Phase A4 P3 (2026-06-09): enriched archive metadata for source-safe
        # inheritance. Compute retention proxy and surface last_mastered_
        # tolerance into the checkpoint state so archive recovery can pull
        # them without scanning curriculum_control JSON.
        own_window = self._policy_nsucc_windows.get(self.policy_idx)
        retention_at_save = float("nan")
        if own_window is not None and own_window.samples:
            end_mean_w = float(own_window.samples[-1].nsucc_mean)
            peak_in_window = max(float(s.nsucc_mean) for s in own_window.samples)
            if peak_in_window > 1e-9:
                retention_at_save = end_mean_w / peak_in_window
            else:
                retention_at_save = 1.0
        tol_at_save = float(curriculum.get("success_tolerance", 0.075) or 0.075)
        same_tol_at_save = self._same_tol_retention_snapshot(self.policy_idx, tol_at_save)
        retention_valid_at_save = bool(same_tol_at_save.get("valid", False))
        if self.same_tol_retention_enable:
            if retention_valid_at_save:
                retention_at_save = float(same_tol_at_save["retention"])
            else:
                # Do not serialize NaN into archive JSON; strict late rescue
                # keys off same_tol_retention_valid_at_save instead.
                retention_at_save = None
        last_mastered_at_save = float(
            self._last_mastered_tolerance.get(self.policy_idx, 0.075)
        )
        # source_safe_for_curriculum flag is computed against fastcurr-style
        # defaults (mean 20, retention 0.7). Consumers can recompute their
        # own gate using the raw fields below; this is a convenience hint.
        try:
            source_safe_at_save = bool(
                retention_at_save is not None
                and math.isfinite(float(retention_at_save))
                and nsucc_mean >= 20.0
                and float(retention_at_save) >= 0.7
            )
        except Exception:
            source_safe_at_save = False
        state = {
            "policy_idx": self.policy_idx,
            # Patch 1: "objective" is now the PBT *ranking* score (used by
            # _ternary_pbt sort). The original DexPBT value lives under
            # "r_meta_raw". Legacy readers that index "objective" continue
            # to function -- only the semantics shift to ranking_score.
            "objective": ranking_score,
            "r_meta_raw": r_meta_raw,
            "pbt_ranking_score": ranking_score,
            "ranking_metric_mode": self.pbt_ranking_metric,
            "nsucc_zero_cap": self.pbt_nsucc_zero_cap,
            # Phase A4 P3 (2026-06-09): source-safe metadata.
            "source_retention_at_save": (
                float(retention_at_save) if retention_at_save is not None else None
            ),
            "tolerance_bin_at_save": int(same_tol_at_save["bin_id"]),
            "same_tol_retention_at_save": (
                float(same_tol_at_save["retention"])
                if retention_valid_at_save else None
            ),
            "same_tol_retention_valid_at_save": bool(retention_valid_at_save),
            "same_tol_peak_nsucc_mean_at_save": float(same_tol_at_save["peak"]),
            "last_mastered_tolerance_at_save": float(last_mastered_at_save),
            "source_safe_for_curriculum_at_save": bool(source_safe_at_save),
            # Patch B (rev 3, 2026-05-30): low-nsucc cap params persisted
            # on every checkpoint so post-hoc analysis can reconstruct
            # exactly how a ranking_score was derived from r_meta_raw.
            "nsucc_low_threshold": self.pbt_nsucc_low_threshold,
            "nsucc_low_cap": self.pbt_nsucc_low_cap,
            "nsucc_mean": nsucc_mean,
            "nsucc_max": nsucc_max,
            "frame": int(population_frame),
            "population_frame": int(population_frame),
            "lineage_id": str(self._lineage_id),
            "lineage_start_pf": int(lineage_start_pf),
            "weight_age_frame": int(weight_age_frame),
            "schema_version": int(PBT_SNAPSHOT_SCHEMA_VERSION),
            "iteration": self.pbt_iteration,
            "timestamp": time.time(),
            "params": self.mutable_params,
            "checkpoint": str(checkpoint_path.resolve()),
            "experiment_name": self.algo.experiment_name,
            # NEW (M1): per-policy curriculum + task state needed for PBT restart
            "curriculum": curriculum,
            # Patch 2 (rev 2, 2026-05-30): anti-churn snapshot. `curriculum`
            # also surfaces `frame_since_restart`, but peers may read the
            # state before the curriculum block is populated (e.g. on the
            # very first tick after launch), so we duplicate the data here.
            "restart_count": int(self._restart_count),
            "frames_at_restart": int(self._frames_at_restart),
            "last_mutation_frame": int(self.last_mutation_frame),
            # PBT-WM Strategy Zoo v1.0
            "wm_strategy_profile_name": str(self.wm_strategy_profile_name),
            "wm_strategy_profile_id": int(self.wm_strategy_profile_id),
            "wm_strategy_profile_family": str(
                getattr(self, "wm_strategy_profile_family", self.wm_strategy_profile_name)
            ),
            "wm_strategy_profile_generation": int(self._wm_strategy_profile_generation),
            "wm_strategy_profile_parent": int(self._wm_strategy_profile_parent),
            "wm_strategy_profile_copied_from_source": bool(
                self._wm_strategy_profile_copied_from_source
            ),
            "profile_actor_distill_enabled": bool(self.profile_actor_distill_enabled),
            "profile_residual_apply_enabled": bool(
                self.profile_residual_apply_enabled
            ),
            "profile_rssm_mppi_enabled": bool(self.profile_rssm_mppi_enabled),
            "profile_wm_pbt_score_enabled": bool(self.profile_wm_pbt_score_enabled),
            "profile_wm_curriculum_controller_enabled": bool(
                self.profile_wm_curriculum_controller_enabled
            ),
            "profile_elite_bank_enabled": bool(self.profile_elite_bank_enabled),
            "profile_conservative_planner_enabled": bool(
                self.profile_conservative_planner_enabled
            ),
        }
        torch.save(algo_state, ckpt_tmp)
        meta_tmp.write_text(json.dumps(state))
        prev_commit_id = self._read_prev_commit_id()
        commit_manifest = {
            "commit_id": f"{prefix}",
            "schema_version": int(PBT_SNAPSHOT_SCHEMA_VERSION),
            "population_frame": int(population_frame),
            "lineage_id": str(self._lineage_id),
            "lineage_start_pf": int(lineage_start_pf),
            "weight_age_frame": int(weight_age_frame),
            "policy_idx": int(self.policy_idx),
            "pbt_iteration": int(self.pbt_iteration),
            "optimizer_step": int(getattr(self.algo, "_pbt_optimizer_step", 0) or 0),
        }
        if prev_commit_id:
            commit_manifest["prev_commit_id"] = str(prev_commit_id)
        try:
            snapshot_commit_barrier(
                staging_dir=self.curr_policy_workspace / f"{prefix}.staging",
                model_src=ckpt_tmp,
                meta_src=meta_tmp,
                publish_model=checkpoint_path,
                publish_meta=meta_path,
                manifest=commit_manifest,
                sidecar_src=sidecar_saved_path,
                tick=int(self.interval_steps),
            )
        except RuntimeError as exc:
            self._append_pbt_tick_debug(
                env_frames=int(population_frame),
                population_frame=int(population_frame),
                derived_iteration=int(self.pbt_iteration),
                should_tick=False,
                skip_reason="",
                error_reason=f"snapshot_commit_failed:{exc}",
            )
            raise
        ckpt_tmp.unlink(missing_ok=True)
        meta_tmp.unlink(missing_ok=True)

        # Patch 5 (rev 2, 2026-05-30): append-only peak archives that are
        # decoupled from `_maybe_save_best_policy` (which is rank-based on
        # `objective`/ranking_score). These archives index by reporting
        # metrics so that peak DexPBT performance is recoverable even when
        # ranking score collapses.
        try:
            self._update_best_archives(state)
        except Exception as exc:  # pragma: no cover - archive is best-effort
            print(f"[best-archive] Failed to update peak archive: {exc}")

        return state

    def _load_population_checkpoints(self) -> dict[int, dict[str, Any] | None]:
        checkpoints: dict[int, dict[str, Any] | None] = {}
        for policy_idx in range(self.num_policies):
            policy_dir = self.workspace / f"{policy_idx:03d}"
            checkpoints[policy_idx] = None
            if not policy_dir.exists():
                continue
            candidates = sorted(policy_dir.glob("*.json"), reverse=True)
            for candidate in candidates:
                try:
                    data = json.loads(candidate.read_text())
                except Exception:
                    continue
                if int(data.get("iteration", -1)) <= self.pbt_iteration:
                    checkpoints[policy_idx] = data
                    break
        return checkpoints

    def _maybe_save_best_policy(self, best_state: dict[str, Any] | None) -> None:
        if best_state is None:
            return
        metric = float(best_state["objective"])
        checkpoint = Path(best_state["checkpoint"])
        dst_prefix = (
            f"{self.task_name}_best_obj_{metric:015.5f}_iter_{int(best_state['iteration']):04d}"
            f"_policy{int(best_state['policy_idx']):03d}"
        )
        dst_model = self.best_policy_workspace / f"{dst_prefix}.pth"
        dst_meta = self.best_policy_workspace / f"{dst_prefix}.json"
        if dst_model.exists():
            return
        shutil.copyfile(checkpoint, dst_model)
        dst_meta.write_text(json.dumps(best_state, indent=2))

    # ------------------------------------------------------------------ #
    # Patch 5 (rev 2, 2026-05-30): peak-protected, append-only archives.
    #
    # Two independent archives:
    #   best_r_meta/ -- indexed by reporting r_meta_raw (NOT ranking score)
    #   best_nsucc/  -- indexed lexicographically by (nsucc_mean, nsucc_max)
    #
    # Both are append-only: filenames embed the metric value plus iter and
    # policy index, so a newer winner does not overwrite an older one.
    # Each archive has a `_index.json` summarizing the current best entry
    # so post-hoc tooling does not have to enumerate filenames.
    #
    # These archives are decoupled from `_maybe_save_best_policy`, which is
    # rank-based on `objective` and therefore subject to the same
    # tolerance-progress inflation as the live PBT ranking.
    # ------------------------------------------------------------------ #

    _BEST_INDEX_FILE = "_index.json"

    def _load_best_index(self, archive_dir: Path) -> dict[str, Any] | None:
        """Read `_index.json` from an archive directory if present.

        Returns the persisted "best" state dict (the same shape as the
        per-tick `state` written by `_save_workspace_checkpoint`). Returns
        None on first launch (no index yet) or on parse failure.
        """
        idx = archive_dir / self._BEST_INDEX_FILE
        if not idx.exists():
            return None
        try:
            data = json.loads(idx.read_text())
            if isinstance(data, dict):
                return data
        except Exception as exc:  # pragma: no cover
            print(f"[best-archive] Failed to read {idx}: {exc}")
        return None

    @staticmethod
    def _is_strictly_better_r_meta(new: dict[str, Any], old: dict[str, Any] | None) -> bool:
        """Strict > on `r_meta_raw`; treat missing/NaN as -inf."""
        if old is None:
            return True
        new_v = float(new.get("r_meta_raw", float("-inf")))
        old_v = float(old.get("r_meta_raw", float("-inf")))
        if math.isnan(new_v):
            return False
        if math.isnan(old_v):
            return True
        return new_v > old_v

    @staticmethod
    def _is_strictly_better_nsucc(new: dict[str, Any], old: dict[str, Any] | None) -> bool:
        """Lex-strict on (nsucc_mean, nsucc_max); missing treated as -inf.

        We require strict improvement on `nsucc_mean` first; if tied, allow
        strict improvement on `nsucc_max` to qualify. This avoids ratcheting
        on noise but still archives within-run peak counts.
        """
        if old is None:
            return True
        nm_new = float(new.get("nsucc_mean", float("-inf")))
        nm_old = float(old.get("nsucc_mean", float("-inf")))
        nx_new = float(new.get("nsucc_max", float("-inf")))
        nx_old = float(old.get("nsucc_max", float("-inf")))
        if math.isnan(nm_new) or math.isnan(nx_new):
            return False
        if math.isnan(nm_old) or math.isnan(nx_old):
            return True
        if nm_new > nm_old:
            return True
        if nm_new == nm_old and nx_new > nx_old:
            return True
        return False

    def _peak_filename(self, kind: str, state: dict[str, Any]) -> str:
        """Format an append-only filename for the peak archive."""
        iteration = int(state.get("iteration", 0))
        policy_idx = int(state.get("policy_idx", -1))
        frame = int(state.get("frame", 0))
        # Floats are clamped to non-negative for filename sort order.
        r_meta_raw = max(0.0, float(state.get("r_meta_raw", 0.0) or 0.0))
        nsucc_mean = max(0.0, float(state.get("nsucc_mean", 0.0) or 0.0))
        nsucc_max = max(0.0, float(state.get("nsucc_max", 0.0) or 0.0))
        return (
            f"{self.task_name}_best_{kind}"
            f"_iter_{iteration:04d}_policy{policy_idx:03d}"
            f"_frame_{frame // 1_000_000:05d}M"
            f"_nsuccM_{nsucc_mean:07.3f}_nsuccX_{nsucc_max:06.1f}"
            f"_rmeta_{r_meta_raw:08.4f}"
        )

    def _save_peak_archive(self, kind: str, state: dict[str, Any]) -> None:
        """Copy `.pth` + write metadata `.json` into the peak archive.

        Append-only: never overwrites an existing entry, but always
        refreshes `_index.json` so the latest peak is easy to look up.
        """
        if kind == "r_meta":
            archive_dir = self.best_r_meta_workspace
        elif kind == "nsucc":
            archive_dir = self.best_nsucc_workspace
        elif kind == "stable":
            archive_dir = self.best_stable_workspace
        elif kind == "floor_recovery":
            archive_dir = self.best_floor_recovery_workspace
            archive_dir.mkdir(parents=True, exist_ok=True)
        else:
            raise ValueError(f"Unknown peak archive kind={kind!r}")

        src_ckpt = Path(state.get("checkpoint", ""))
        if not src_ckpt.exists():
            print(f"[best-archive/{kind}] source checkpoint missing: {src_ckpt}")
            return

        prefix = self._peak_filename(kind, state)
        dst_ckpt = archive_dir / f"{prefix}.pth"
        dst_meta = archive_dir / f"{prefix}.json"

        if not dst_ckpt.exists():
            # `shutil.copy2` preserves mtime so an analyzer can recover
            # the original tick timestamp from the archive.
            shutil.copy2(src_ckpt, dst_ckpt)
        # Metadata is rewritten every time (cheap; latest copy = current
        # peak description for this kind).
        try:
            dst_meta.write_text(json.dumps(state, indent=2))
        except Exception as exc:  # pragma: no cover
            print(f"[best-archive/{kind}] meta write failed: {exc}")

        # _index.json: atomically replace so concurrent readers see a
        # consistent view.
        index_path = archive_dir / self._BEST_INDEX_FILE
        index_tmp = archive_dir / f"{self._BEST_INDEX_FILE}.tmp"
        try:
            index_tmp.write_text(json.dumps(state, indent=2))
            index_tmp.replace(index_path)
        except Exception as exc:  # pragma: no cover
            print(f"[best-archive/{kind}] index update failed: {exc}")

        print(
            f"[best-archive/{kind}] new peak from policy "
            f"{state.get('policy_idx')} iter={state.get('iteration')} "
            f"r_meta_raw={state.get('r_meta_raw')} "
            f"nsucc_mean={state.get('nsucc_mean')} "
            f"nsucc_max={state.get('nsucc_max')}"
        )

    def _update_best_archives(self, state: dict[str, Any]) -> None:
        """Conditionally update best_r_meta and best_nsucc archives.

        Called at the end of `_save_workspace_checkpoint`. Both archives are
        independent so a tick can be a peak on one metric but not the other.
        """
        # `algo.frame` is the global env-frame counter, not the per-tick
        # iteration; we use the meta.json values as-is.
        if self._is_strictly_better_r_meta(state, self._best_r_meta_state):
            self._save_peak_archive("r_meta", state)
            self._best_r_meta_state = dict(state)
        if self._is_strictly_better_nsucc(state, self._best_nsucc_state):
            self._save_peak_archive("nsucc", state)
            self._best_nsucc_state = dict(state)
        window = self._policy_window(self.policy_idx)
        eval_window = RollingNsuccWindow(max_len=window.max_len)
        eval_window.samples = list(window.samples)
        eval_window.append(
            self._window_sample_from_state(
                state,
                env_frames=int(state.get("frame", 0)),
                pbt_iteration=int(state.get("iteration", self.pbt_iteration)),
            )
        )
        own_ctrl = read_curriculum_control(
            self._curriculum_control_dir / f"{self.policy_idx:03d}.json"
        )
        restart_steps = int((state.get("curriculum") or {}).get("frame_since_restart", 0))
        consolidation_active = bool(
            own_ctrl is not None
            and post_shrink_consolidation_active(restart_steps, own_ctrl)
        )
        ok_stable, _ = qualifies_best_stable_save(
            eval_window,
            min_mean=self.stable_source_min_mean,
            min_max=self.stable_source_min_max,
            max_drop_ratio=self.stable_source_max_drop_ratio,
            stable_tolerance_intervals=1,
            required_stable_windows=2,
            consolidation_active=consolidation_active,
        )
        if ok_stable and (
            self._best_stable_state is None
            or float(state.get("nsucc_mean", 0.0))
            >= float(self._best_stable_state.get("nsucc_mean", float("-inf")))
        ):
            stable_state = dict(state)
            stable_state["next_window_validated"] = True
            stable_state["consolidation_active"] = False
            self._save_peak_archive("stable", stable_state)
            self._best_stable_state = dict(stable_state)

        # Phase A4 P4 (2026-06-09): floor-recovery archive (Part A).
        # Save criteria:
        #   * feature explicitly enabled
        #   * current state is at hard_floor (success_tolerance ~ target)
        #   * recovery score strictly improves the archive AND is positive
        #   * not in collapse / not in rollback this tick
        #   * worker shows real signal: nsucc_mean >= protect threshold OR
        #     positive slope above slope-protect threshold
        if (
            self.pbt_floor_recovery_aware
            and self.pbt_enable_best_floor_recovery_archive
        ):
            cur_curriculum = state.get("curriculum") or {}
            stage = self._floor_stage_for(cur_curriculum)
            if stage == "hard_floor" and not consolidation_active:
                hard_collapse = bool(
                    own_ctrl is not None
                    and own_ctrl.hard_collapse_triggered
                )
                if not hard_collapse:
                    score_50, components, slope_pos = self._floor_recovery_score_for(
                        self.policy_idx,
                        eval_window,
                        int(state.get("frame", 0)),
                        hard_collapse=False,
                    )
                    end_mean = float(state.get("nsucc_mean", 0.0) or 0.0)
                    qualifies = end_mean >= float(
                        self.pbt_floor_recovery_min_nsucc_protect
                    ) or slope_pos >= float(
                        self.pbt_floor_recovery_min_slope_protect
                    )
                    prev_best = self._best_floor_recovery_state
                    prev_score = (
                        float(prev_best.get("floor_recovery_score", float("-inf")))
                        if prev_best is not None
                        else float("-inf")
                    )
                    if qualifies and score_50 > 0.0 and score_50 > prev_score:
                        floor_state = dict(state)
                        floor_state["floor_recovery_score"] = float(score_50)
                        floor_state["floor_recovery_components"] = dict(
                            components
                        )
                        floor_state["floor_recovery_slope_positive"] = float(
                            slope_pos
                        )
                        self._save_peak_archive("floor_recovery", floor_state)
                        self._best_floor_recovery_state = dict(floor_state)

    def _rewrite_checkpoint(
        self,
        checkpoint_path: Path,
        env_frames: int,
        wm_strategy: str | None = None,
    ) -> None:
        state = torch.load(checkpoint_path, weights_only=False)
        state["frame"] = env_frames
        pbt_history = state.get("pbt_history", [])
        pbt_history.append((self.policy_idx, env_frames, self.curr_target_objective_value))
        state["pbt_history"] = pbt_history
        # Drop aux_value_head weights when the restarted worker will not use
        # value_imag (e.g. posterior sampled aux_mid). train.py can also register
        # a load-only head, but stripping keeps checkpoints strategy-consistent.
        if wm_strategy is not None:
            from dexpbt_isaaclab.world_model.wm_trainer import get_strategy

            if not get_strategy(str(wm_strategy)).use_value_imag:
                model_sd = state.get("model")
                if isinstance(model_sd, dict):
                    for key in list(model_sd.keys()):
                        if str(key).startswith("aux_value_head"):
                            del model_sd[key]
                self._strip_aux_optimizer_state(state)
        torch.save(state, checkpoint_path)

    @staticmethod
    def _strip_aux_optimizer_state(state: dict) -> None:
        """Remove aux_value_head param group from rl_games optimizer state."""
        opt_sd = state.get("optimizer")
        if not isinstance(opt_sd, dict):
            return
        groups = opt_sd.get("param_groups")
        if not isinstance(groups, list):
            return
        keep_groups: list[dict] = []
        drop_param_ids: set[int] = set()
        for group in groups:
            if not isinstance(group, dict):
                continue
            if group.get("name") == "aux_value_head":
                drop_param_ids.update(int(p) for p in group.get("params", []))
                continue
            keep_groups.append(group)
        if len(keep_groups) == len(groups):
            return
        opt_sd["param_groups"] = keep_groups
        opt_state = opt_sd.get("state")
        if isinstance(opt_state, dict):
            for pid in drop_param_ids:
                opt_state.pop(pid, None)
                opt_state.pop(str(pid), None)

    # Exit code used to signal the supervisor that a PBT restart is requested.
    # The supervisor script (run_train_supervisor.sh) detects this exit code
    # and restarts the training with new parameters from a restart manifest file.
    RESTART_EXIT_CODE = 42
    RESTART_MANIFEST = "pbt_restart.json"

    def _restart_process_with_new_params(
        self,
        new_params: dict[str, float | int],
        checkpoint_path: str,
        source_policy: int,
        env_frames: int,
        tier: str,
        source_curriculum: dict[str, float | int] | None = None,
        source_weight_age_frame: int = 0,
        late_rescue: bool = False,
        late_rescue_decision: dict[str, Any] | None = None,
        adaptive_rescue: bool = False,
        adaptive_rescue_decision: dict[str, Any] | None = None,
        runtime_snap: dict[str, Any] | None = None,
        source_safe_apply: dict[str, Any] | None = None,
    ) -> bool:
        """Own the full restart transaction (memory + control + manifest).

        Commit order (documented intentionally):
          1. snapshot runtime + persistent (control/manifest bytes)
          2. apply source-safe consolidation to control (if requested)
          3. prepare checkpoint / train_args / memory mutations
          4. late-rescue: bump executed + persist quota/cooldown into control
          5. atomic commit restart manifest
          6. mark committed — post-commit audit/writer are best-effort only
          7. SIGKILL / exit

        Any failure before step 6 restores runtime + persistent bytes, cleans
        restart artifacts, and emits exactly one
        ``pbt_restart_transaction_rolled_back`` event. Callers must not also
        restore/cleanup/emit that event.
        """
        snap = (
            runtime_snap
            if runtime_snap is not None
            else self._snapshot_restart_runtime_state()
        )
        persistent_snap = self._snapshot_restart_persistent_state()
        # Mutable commit flag shared with impl so post-commit exceptions do
        # not enter the rollback branch.
        txn = {"committed": False}
        try:
            if source_safe_apply is not None:
                apply_kwargs = dict(source_safe_apply)
                apply_path = Path(apply_kwargs.pop("path"))
                apply_source_safe_consolidation_on_copy(apply_path, **apply_kwargs)
            ok = self._restart_process_with_new_params_impl(
                new_params,
                checkpoint_path,
                source_policy,
                env_frames,
                tier,
                source_curriculum=source_curriculum,
                source_weight_age_frame=source_weight_age_frame,
                late_rescue=late_rescue,
                late_rescue_decision=late_rescue_decision,
                adaptive_rescue=adaptive_rescue,
                adaptive_rescue_decision=adaptive_rescue_decision,
                txn=txn,
            )
            if not ok and not txn["committed"]:
                self._abort_restart_transaction(
                    runtime_snap=snap,
                    persistent_snap=persistent_snap,
                    source_policy=int(source_policy),
                    env_frames=int(env_frames),
                    late_rescue=bool(late_rescue),
                    adaptive_rescue=bool(adaptive_rescue),
                    reason="restart_returned_false",
                )
            return bool(ok) or bool(txn["committed"])
        except Exception as exc:
            print(
                f"[PBT] restart manifest failed for policy {self.policy_idx}: {exc}"
            )
            if txn["committed"]:
                # Commit point already passed; do not roll back durable state.
                self._safe_write_pbt_copy_event({
                    "event_type": "pbt_restart_post_commit_error",
                    "source_policy": int(source_policy),
                    "target_policy": int(self.policy_idx),
                    "late_rescue": bool(late_rescue),
                    "adaptive_rescue": bool(adaptive_rescue),
                    "env_frames": int(env_frames),
                    "reason": f"exception:{type(exc).__name__}",
                    "error": str(exc),
                })
                return True
            self._abort_restart_transaction(
                runtime_snap=snap,
                persistent_snap=persistent_snap,
                source_policy=int(source_policy),
                env_frames=int(env_frames),
                late_rescue=bool(late_rescue),
                adaptive_rescue=bool(adaptive_rescue),
                reason=f"exception:{type(exc).__name__}",
            )
            return False

    def _restart_process_with_new_params_impl(
        self,
        new_params: dict[str, float | int],
        checkpoint_path: str,
        source_policy: int,
        env_frames: int,
        tier: str,
        source_curriculum: dict[str, float | int] | None = None,
        source_weight_age_frame: int = 0,
        late_rescue: bool = False,
        late_rescue_decision: dict[str, Any] | None = None,
        adaptive_rescue: bool = False,
        adaptive_rescue_decision: dict[str, Any] | None = None,
        txn: dict[str, Any] | None = None,
    ) -> bool:
        src = Path(checkpoint_path).resolve()
        tmp_dir = self.workspace / f"restart_p{self.policy_idx}"
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)
        tmp_dir.mkdir(parents=True, exist_ok=True)
        restart_checkpoint = tmp_dir / src.name
        shutil.copyfile(src, restart_checkpoint)

        self.last_mutation_frame = env_frames
        # Patch 2 (rev 2, 2026-05-30): bump in-process restart_count and
        # remember the env-frame at which this restart fires. These get
        # forwarded to the spawned process via `--*_resume` args so that
        # the new __init__ rebuilds anti-churn state correctly (the same
        # values are also embedded in the restart manifest for diagnosis).
        self._restart_count += 1
        self._frames_at_restart = env_frames
        self._last_rollback_frame[self.policy_idx] = -10**18
        self._shrinks_since_rollback[self.policy_idx] = 0
        self._rollback_reason[self.policy_idx] = "restart_reset"

        # ------------------------------------------------------------------
        # M7: posterior sampling MUST happen BEFORE train_args construction
        # so the spawned process uses the freshly sampled wm_strategy.
        # (Previously sampling happened after train_args was already frozen.)
        # ------------------------------------------------------------------
        selected_strategy = new_params.get("wm_strategy", "none")
        wm_ckpt_path = None
        wm_ckpt_strategy = "none"
        wm_ckpt_compatible = False
        # Patch J (2026-05-31): in `fixed` ablation mode we bypass the
        # posterior entirely and force the worker to keep using
        # `fixed_wm_strategy` on every restart. CF stays off so the
        # spawned process never tries to launch a counterfactual block.
        if self.wm_strategy_mode == "fixed":
            selected_strategy = self.fixed_wm_strategy
            self._wm_strategy = self.fixed_wm_strategy
            new_params["wm_strategy"] = self.fixed_wm_strategy
            new_params["enable_counterfactual"] = False
            wm_ckpt_path = self._get_wm_checkpoint_path(selected_strategy, source_policy)
            if wm_ckpt_path:
                wm_ckpt_compatible = True
                marker = Path(wm_ckpt_path).parent / "strategy.txt"
                if marker.exists():
                    wm_ckpt_strategy = marker.read_text().strip() or "unknown"
            print(
                f"[WM] Fixed-strategy mode: keeping wm_strategy="
                f"{selected_strategy} (no posterior sample)"
            )
            if self.writer is not None:
                self.writer.add_scalar(
                    "wm_strategy/current", hash(selected_strategy) % 100, env_frames
                )
                self.writer.add_scalar("wm_strategy/fixed_mode", 1.0, env_frames)
        elif self._enable_wm and self._posterior_selector is not None:
            if not self._is_posterior_writer:
                self._posterior_selector.load_or_init(policy_idx=self.policy_idx)
            else:
                self._maybe_consume_counterfactual_result(env_frames)

            selected_strategy = self._posterior_selector.sample()
            wm_ckpt_path = self._get_wm_checkpoint_path(selected_strategy, source_policy)
            if wm_ckpt_path:
                wm_ckpt_compatible = True
                marker = Path(wm_ckpt_path).parent / "strategy.txt"
                if marker.exists():
                    wm_ckpt_strategy = marker.read_text().strip() or "unknown"
            print(f"[WM] Strategy selected: {selected_strategy}  (ckpt={wm_ckpt_path})")
            self._wm_strategy = selected_strategy

            if self.writer is not None:
                self.writer.add_scalar("wm_strategy/current", hash(selected_strategy) % 100, env_frames)
                self.writer.add_scalar(
                    "wm_posterior/is_writer",
                    float(1.0 if self._is_posterior_writer else 0.0),
                    env_frames,
                )
                self.writer.add_scalar(
                    "wm_posterior/last_update_policy_idx",
                    float(self._last_posterior_update_policy_idx),
                    env_frames,
                )
                self.writer.add_scalar(
                    "wm_posterior/last_consumed_block_id",
                    float(self._last_consumed_cf_block_id),
                    env_frames,
                )
                for s, p in self._posterior_selector.posterior.items():
                    self.writer.add_scalar(f"wm_posterior/mu_{s}", p.mean, env_frames)
                    self.writer.add_scalar(f"wm_posterior/sigma_{s}", p.std, env_frames)
                    self.writer.add_scalar(f"wm_posterior/n_{s}", float(p.n), env_frames)
                self.writer.add_scalar("wm_posterior/selected_strategy",
                                       hash(selected_strategy) % 100, env_frames)

        self._rewrite_checkpoint(restart_checkpoint, env_frames, wm_strategy=selected_strategy)

        # Strip flags the supervisor re-injects per attempt (--task, --train_root,
        # --experiment_name) so each restart's logs/checkpoints land under the new
        # attempt name rather than the original one. Mutated hyperparameters and
        # --checkpoint are stripped because we replace them below.
        skip_flags = {
            "--task",
            "--train_root",
            "--experiment_name",
            "--checkpoint",
            "--gamma",
            "--kl_threshold",
            "--grad_norm",
            "--critic_coef",
            "--bounds_loss_coef",
            "--e_clip",
            "--mini_epochs",
            "--alpha_reach",
            "--alpha_pick",
            "--r_picked",
            "--alpha_targ",
            "--r_success",
            "--kuka_actions_penalty_scale",
            "--allegro_actions_penalty_scale",
            "--learning_rate",
            "--enable_wm",
            "--enable_counterfactual",
            "--cf_interval_gens",
            "--wm_strategy",
            "--wm_ensemble_size",
            "--wm_lr",
            "--wm_update_every_n_epochs",
            "--wm_batch_size",
            "--wm_checkpoint_path",
            "--wm_checkpoint_strategy",
            "--wm_load_mode",
            "--wm_strategy_mode",
            "--fixed_wm_strategy",
            "--require_counterfactual_runner",
            # M2: also strip previous resume flags so a chain of restarts always
            # carries the freshest curriculum snapshot, not a stale one.
            "--success_tolerance_resume",
            "--initial_tolerance_resume",
            "--target_tolerance_resume",
            "--last_curriculum_update_resume",
            "--frame_since_restart_resume",
            "--restart_source_policy",
            # Patch 2 (rev 2, 2026-05-30): anti-churn resume hooks. Strip
            # stale values so each restart writes the freshest snapshot.
            "--restart_count_resume",
            "--frames_at_restart_resume",
            "--last_mutation_frame_resume",
            "--last_profile_copy_frame_resume",
            "--pbt_lineage_id_resume",
            "--pbt_lineage_start_pf_resume",
            "--pbt_weight_age_frame_resume",
            "--pbt_copy_target_frame_before_restore",
            # The supervisor re-injects DEXPBT_KIT_ARGS at attempt start; if we
            # keep --kit_args from _original_sys_argv it would be duplicated.
            "--kit_args",
        }
        from dexpbt_isaaclab.world_model.wm_strategy_profiles import (
            profile_controlled_restart_skip_flags,
        )

        skip_flags = set(skip_flags) | set(profile_controlled_restart_skip_flags())
        # Use original sys.argv (before train.py strips parsed args).
        # train.py is loaded as __main__ via `python -m`, so it lives in
        # sys.modules['__main__'] — importing the dotted name would re-execute
        # the module, triggering a second AppLauncher() in the same process
        # and deadlocking Isaac Sim.
        _train_mod = sys.modules.get("__main__")
        argv = getattr(_train_mod, "_original_sys_argv", sys.argv)
        # Filter out python -m args and module name that get mixed into sys.argv
        _skip_prefixes = {"-u", "-m", "dexpbt_isaaclab.scripts.train"}
        _orphan_strategy_tokens = {
            "none", "aux_low", "aux_mid", "value_imag_h1", "value_imag_h3",
            "rssm_aux", "rssm_belief_value", "trans_aux", "trans_belief_value",
        }
        base_args: list[str] = []
        idx = 1
        while idx < len(argv):
            arg = argv[idx]
            flag = arg.split("=")[0]
            if flag in skip_flags:
                if "=" not in arg and idx + 1 < len(argv):
                    idx += 2
                    continue
                idx += 1
                continue
            if arg in _skip_prefixes:
                idx += 1
                continue
            # Prior restarts may leave bare strategy names / numeric tails when
            # `--wm_strategy value_imag_h1` was stripped without its value.
            if arg in _orphan_strategy_tokens:
                idx += 1
                continue
            if arg.isdigit() and idx > 1:
                prev = argv[idx - 1].split("=")[0]
                if prev in ("--wm_ensemble_size", "--cf_interval_gens"):
                    idx += 1
                    continue
            base_args.append(arg)
            idx += 1

        # Build the full arg list for the new training run
        train_args = base_args + [
            f"--checkpoint={restart_checkpoint}",
            f"--gamma={new_params['gamma']}",
            f"--kl_threshold={new_params['kl_threshold']}",
            f"--grad_norm={new_params['grad_norm']}",
            f"--critic_coef={new_params['critic_coef']}",
            f"--bounds_loss_coef={new_params['bounds_loss_coef']}",
            f"--e_clip={new_params['e_clip']}",
            f"--mini_epochs={new_params['mini_epochs']}",
            f"--learning_rate={new_params.get('learning_rate', 3e-4)}",
            f"--alpha_reach={new_params['alpha_reach']}",
            f"--alpha_pick={new_params['alpha_pick']}",
            f"--r_picked={new_params['r_picked']}",
            f"--alpha_targ={new_params['alpha_targ']}",
            f"--r_success={new_params['r_success']}",
            f"--kuka_actions_penalty_scale={new_params['kuka_actions_penalty_scale']}",
            f"--allegro_actions_penalty_scale={new_params['allegro_actions_penalty_scale']}",
            f"--learning_rate={new_params.get('learning_rate', 3e-4)}",
        ]
        profile_is_authoritative = str(
            self.wm_strategy_profile_name or "legacy"
        ) != "legacy"
        if not profile_is_authoritative:
            # Legacy path: forward source WM flags unchanged.
            train_args.extend(
                [
                    f"--wm_strategy={selected_strategy}",
                    f"--wm_ensemble_size={int(new_params.get('wm_ensemble_size', 3))}",
                    "--wm_load_mode=auto",
                    f"--wm_checkpoint_strategy={wm_ckpt_strategy}",
                ]
            )
            if bool(new_params.get("enable_wm", False)):
                train_args.append("--enable_wm")
                if wm_ckpt_path:
                    train_args.append(f"--wm_checkpoint_path={wm_ckpt_path}")
            train_args.append(f"--wm_strategy_mode={self.wm_strategy_mode}")
            train_args.append(f"--fixed_wm_strategy={self.fixed_wm_strategy}")
        else:
            # Strategy-zoo path: profile expansion is authoritative; only
            # checkpoint warm-start hints may pass through.
            if wm_ckpt_path:
                train_args.append(f"--wm_checkpoint_path={wm_ckpt_path}")
            train_args.append(f"--wm_checkpoint_strategy={wm_ckpt_strategy}")
        if bool(new_params.get("enable_counterfactual", False)):
            train_args.append("--enable_counterfactual")
            train_args.append(f"--cf_interval_gens={int(new_params.get('cf_interval_gens', 5))}")
            train_args.append(f"--cf_steps={int(new_params.get('cf_steps', 5_000_000))}")
            train_args.append(f"--cf_num_envs={int(new_params.get('cf_num_envs', 4096))}")
            train_args.append(f"--cf_max_parallel_children={int(new_params.get('cf_max_parallel_children', 2))}")
            train_args.append(f"--cf_cuda_visible_devices={str(new_params.get('cf_cuda_visible_devices', '4,5'))}")
            train_args.append(f"--cf_horizon={int(new_params.get('cf_horizon', 16))}")
        if bool(new_params.get("require_counterfactual_runner", False)):
            train_args.append("--require_counterfactual_runner")

        # M2: pass source worker's curriculum state explicitly so the spawned
        # process can resume success_tolerance / last_curriculum_update instead
        # of falling back to the cfg default (0.075). Without this, every PBT
        # bottom replacement effectively rewinds the curriculum.
        sc = source_curriculum or {}
        if "success_tolerance" in sc:
            train_args.append(f"--success_tolerance_resume={float(sc['success_tolerance'])}")
        if "initial_tolerance" in sc:
            train_args.append(f"--initial_tolerance_resume={float(sc['initial_tolerance'])}")
        if "target_tolerance" in sc:
            train_args.append(f"--target_tolerance_resume={float(sc['target_tolerance'])}")
        if "last_curriculum_update" in sc:
            train_args.append(f"--last_curriculum_update_resume={int(sc['last_curriculum_update'])}")
        if "frame_since_restart" in sc:
            train_args.append(f"--frame_since_restart_resume={int(sc['frame_since_restart'])}")
        train_args.append(f"--restart_source_policy={int(source_policy)}")
        train_args.append(f"--pbt_copy_target_frame_before_restore={int(env_frames)}")
        self._lineage_id = str(_uuid_mod.uuid4())
        self._lineage_start_pf = int(env_frames)
        self._weight_age_frame = int(source_weight_age_frame)
        train_args.append(f"--pbt_lineage_id_resume={self._lineage_id}")
        train_args.append(
            f"--pbt_lineage_start_pf_resume={int(self._lineage_start_pf)}"
        )
        train_args.append(
            f"--pbt_weight_age_frame_resume={int(self._weight_age_frame)}"
        )
        # Patch H (2026-05-31): post-restart curriculum freeze. Observer
        # config is env frames; env stores simulation steps (Scheme A).
        freeze_env_frames = (
            self.curriculum_freeze_on_recovery_frames
            if (
                self._last_source_checkpoint_type
                in (
                    "archive",
                    "best_stable_archive",
                    "best_floor_recovery_archive",
                    "late_rescue_archive",
                )
                or self._last_source_tier_label
                in (
                    "late_rescue_live",
                    "late_rescue_archive",
                    "adaptive_rescue_live",
                    "adaptive_rescue_archive",
                    "adaptive_rescue_best_stable_archive",
                    "adaptive_rescue_floor_recovery_archive",
                    "adaptive_rescue_best_nsucc_archive",
                )
            )
            else self.curriculum_freeze_frames
        )
        freeze_steps = env_frames_to_sim_steps(freeze_env_frames, self.num_envs)
        train_args.append(
            f"--curriculum_freeze_steps_after_restart_resume={int(freeze_steps)}"
        )
        if self._last_source_checkpoint_type in (
            "archive",
            "best_stable_archive",
        ) and self.writer is not None:
            if self._last_source_checkpoint_type == "best_stable_archive":
                self.writer.add_scalar(
                    "pbt/best_stable_recovery_executed", 1.0, env_frames
                )
            sc_log = source_curriculum or {}
            self.writer.add_scalar("pbt/archive_recovery_executed", 1.0, env_frames)
            self.writer.add_scalar(
                "pbt/recovery_checkpoint_nsucc_mean",
                float(sc_log.get("mean_prev_episode_successes", 0.0) or 0.0),
                env_frames,
            )
            self.writer.add_scalar(
                "pbt/recovery_checkpoint_tolerance",
                float(sc_log.get("success_tolerance", float("nan"))),
                env_frames,
            )
            self.writer.add_scalar(
                "pbt/recovery_curriculum_freeze_env_frames",
                float(freeze_env_frames),
                env_frames,
            )
            self.writer.add_scalar(
                "pbt/recovery_curriculum_freeze_steps",
                float(freeze_steps),
                env_frames,
            )
            self.writer.add_text(
                "pbt/recovery_checkpoint_path",
                str(restart_checkpoint),
                env_frames,
            )

        # Patch J (2026-05-31): forward fixed-strategy ablation mode so
        # the spawned process stays on the same wm_strategy across
        # restarts. Strategy-zoo profiles derive these from the table.
        if not profile_is_authoritative:
            train_args.append(f"--wm_strategy_mode={self.wm_strategy_mode}")
            train_args.append(f"--fixed_wm_strategy={self.fixed_wm_strategy}")

        # Patch 2 (rev 2, 2026-05-30): forward in-process anti-churn state so
        # the spawned worker can rebuild it via train.py argparse without
        # silently resetting to 0 (the historic bug that disabled
        # initial_delay across restarts).
        train_args.append(f"--restart_count_resume={int(self._restart_count)}")
        train_args.append(f"--frames_at_restart_resume={int(self._frames_at_restart)}")
        train_args.append(f"--last_mutation_frame_resume={int(self.last_mutation_frame)}")
        # Patch F (2026-05-31): self/cross-restart accounting forwarded.
        train_args.append(
            f"--self_restart_count_resume={int(self._self_restart_count)}"
        )
        train_args.append(
            f"--last_self_restart_frame_resume={int(self._last_self_restart_frame)}"
        )
        train_args.append(
            f"--cross_restart_count_resume={int(self._cross_restart_count)}"
        )
        # Patch G (2026-05-31): archive-recovery accounting forwarded so the
        # spawned worker honours the recovery cooldown even after exec.
        train_args.append(
            f"--recovery_restart_count_resume={int(self._recovery_restart_count)}"
        )
        train_args.append(
            f"--last_recovery_frame_resume={int(self._last_recovery_frame)}"
        )
        train_args.append(
            f"--last_profile_copy_frame_resume={int(self._last_profile_copy_frame)}"
        )

        # PBT-WM Strategy Zoo v1.2: optional profile inheritance on cross-copy.
        from dexpbt_isaaclab.pbt.pbt_profile_copy import (
            ProfileCopyConfig,
            decide_profile_copy,
        )
        from dexpbt_isaaclab.world_model.wm_strategy_profiles import (
            PROFILE_IDS,
            profile_copy_restart_train_args,
            resolve_profile_copy_restart_config,
        )

        population_meta = self._load_population_checkpoints()
        diversity_count, profiles_set = self._count_profile_diversity(population_meta)
        self._last_pbt_wm_profile_diversity_count = int(diversity_count)
        (
            copy_enable,
            copy_after,
            copy_prob,
            copy_keep_min,
            copy_min_nsucc_gap,
            copy_min_source_retention,
            copy_cooldown_frames,
            copy_bottom_k,
            copy_top_k,
            copy_source_min_nsucc_mean,
            copy_source_min_frames,
            copy_target_min_nsucc_mean,
            copy_target_min_nsucc_max,
            copy_locked_target_slots,
        ) = resolve_profile_copy_restart_config(
            enable=self.pbt_wm_profile_copy_enable,
            after_frames=self.pbt_wm_profile_copy_after_frames,
            prob=self.pbt_wm_profile_copy_prob,
            keep_diversity_min_profiles=(
                self.pbt_wm_profile_keep_diversity_min_profiles
            ),
            min_nsucc_gap=self.pbt_wm_profile_copy_min_nsucc_gap,
            min_source_retention=self.pbt_wm_profile_copy_min_source_retention,
            cooldown_frames=self.pbt_wm_profile_copy_cooldown_frames,
            bottom_k=self.pbt_wm_profile_copy_bottom_k,
            top_k=self.pbt_wm_profile_copy_top_k,
            source_min_nsucc_mean=self.pbt_wm_profile_copy_source_min_nsucc_mean,
            source_min_frames=self.pbt_wm_profile_copy_source_min_frames,
            target_min_nsucc_mean=self.pbt_wm_profile_copy_target_min_nsucc_mean,
            target_min_nsucc_max=self.pbt_wm_profile_copy_target_min_nsucc_max,
            locked_target_slots=self.pbt_wm_profile_copy_locked_target_slots,
            metadata_path=self.train_dir.parent / "metadata.txt",
        )
        old_profile_name = str(self.wm_strategy_profile_name or "legacy")
        source_name, source_gen, _ = self._profile_from_population_state(
            population_meta, int(source_policy)
        )
        source_state = (population_meta or {}).get(int(source_policy)) or {}
        target_state = (population_meta or {}).get(int(self.policy_idx)) or {}
        source_meta = self._extract_source_curriculum_metadata(
            source_state=source_state,
            source_policy_idx=int(source_policy),
        )
        target_meta = self._extract_source_curriculum_metadata(
            source_state=target_state,
            source_policy_idx=int(self.policy_idx),
        )
        window_nsucc: dict[int, float] = {}
        for pid, win in self._policy_nsucc_windows.items():
            if win.samples:
                window_nsucc[int(pid)] = float(win.samples[-1].nsucc_mean)
        source_ctrl_path = (
            self._curriculum_control_dir / f"{int(source_policy):03d}.json"
        )
        source_ctrl = read_curriculum_control(source_ctrl_path)
        source_in_collapse = bool(
            source_ctrl is not None
            and bool(getattr(source_ctrl, "hard_collapse_triggered", False))
        )
        if not source_in_collapse and source_meta.get("has_source_retention"):
            ret = float(source_meta.get("source_retention", float("nan")))
            source_in_collapse = not math.isnan(ret) and ret < 0.50
        target_protected_until = max(
            int(self._promising_protect_until.get(self.policy_idx, 0)),
            int(self._floor_recovery_protect_until.get(self.policy_idx, 0)),
        )
        copy_cfg = ProfileCopyConfig(
            enable=bool(copy_enable),
            after_frames=int(copy_after),
            base_prob=float(copy_prob),
            keep_diversity_min_profiles=int(copy_keep_min),
            min_nsucc_gap=float(copy_min_nsucc_gap),
            min_source_retention=float(copy_min_source_retention),
            cooldown_frames=int(copy_cooldown_frames),
            bottom_k=int(copy_bottom_k),
            top_k=int(copy_top_k),
            source_min_nsucc_mean=float(copy_source_min_nsucc_mean),
            source_min_frames=int(copy_source_min_frames),
            target_min_nsucc_mean=float(copy_target_min_nsucc_mean),
            target_min_nsucc_max=float(copy_target_min_nsucc_max),
            locked_target_slots=frozenset(copy_locked_target_slots),
        )
        copy_result = decide_profile_copy(
            cfg=copy_cfg,
            env_frames=int(env_frames),
            target_policy_idx=int(self.policy_idx),
            source_policy_idx=int(source_policy),
            target_profile_name=old_profile_name,
            source_profile_name=source_name,
            source_generation=int(source_gen),
            diversity_count=int(diversity_count),
            profiles_set=set(profiles_set),
            population=population_meta,
            source_meta=source_meta,
            target_meta=target_meta,
            source_checkpoint_type=str(self._last_source_checkpoint_type),
            target_protected_until=int(target_protected_until),
            last_profile_copy_frame=int(self._last_profile_copy_frame),
            window_nsucc=window_nsucc,
            source_in_collapse=bool(source_in_collapse),
        )
        profile_copy_decision = str(copy_result.decision)
        new_profile_name = str(copy_result.new_profile_name or old_profile_name)
        self._last_profile_copy_eligible = bool(copy_result.eligible)
        self._last_profile_copy_reject_reason = str(copy_result.reject_reason or "")
        self._last_profile_copy_source_policy = int(source_policy)
        self._last_profile_copy_source_profile = str(source_name)
        self._last_profile_copy_target_old_profile = str(old_profile_name)
        self._last_profile_copy_target_new_profile = str(new_profile_name)
        self._last_profile_copy_weighted_prob = float(copy_result.weighted_prob)
        self._weight_copy_count += 1
        if copy_result.attempted:
            self._profile_copy_attempt_count += 1
        if copy_result.accepted:
            self._profile_copy_accept_count += 1
            self._wm_strategy_profile_generation = int(copy_result.source_generation) + 1
            self._wm_strategy_profile_parent = int(source_policy)
            self._wm_strategy_profile_copied_from_source = True
            self._last_profile_copy_frame = int(env_frames)
            self._weight_copy_with_profile_copy_count += 1
        else:
            if copy_result.reject_reason:
                reason = str(copy_result.reject_reason)
                self._profile_copy_reject_reason_counts[reason] = (
                    int(self._profile_copy_reject_reason_counts.get(reason, 0)) + 1
                )
            if copy_result.attempted:
                self._profile_copy_reject_count += 1
            self._weight_copy_without_profile_copy_count += 1
        self.wm_strategy_profile_name = new_profile_name
        self.wm_strategy_profile_id = int(PROFILE_IDS.get(new_profile_name, 0))
        train_args = self._rewrite_train_arg_profile(train_args, new_profile_name)
        train_args.append(
            f"--wm_strategy_profile_generation_resume="
            f"{int(self._wm_strategy_profile_generation)}"
        )
        train_args.append(
            f"--wm_strategy_profile_parent_resume="
            f"{int(self._wm_strategy_profile_parent)}"
        )
        train_args.append(
            f"--wm_strategy_profile_copied_from_source_resume="
            f"{1 if self._wm_strategy_profile_copied_from_source else 0}"
        )
        train_args.extend(
            profile_copy_restart_train_args(
                enable=copy_enable,
                after_frames=copy_after,
                prob=copy_prob,
                keep_diversity_min_profiles=copy_keep_min,
                min_nsucc_gap=copy_min_nsucc_gap,
                min_source_retention=copy_min_source_retention,
                cooldown_frames=copy_cooldown_frames,
                bottom_k=copy_bottom_k,
                top_k=copy_top_k,
                source_min_nsucc_mean=copy_source_min_nsucc_mean,
                source_min_frames=copy_source_min_frames,
                target_min_nsucc_mean=copy_target_min_nsucc_mean,
                target_min_nsucc_max=copy_target_min_nsucc_max,
                locked_target_slots=copy_locked_target_slots,
            )
        )
        self._last_pbt_wm_profile_copy_decision = str(profile_copy_decision)
        self._write_profile_copy_event(
            {
                "env_frames": int(env_frames),
                "source_policy": int(source_policy),
                "target_policy": int(self.policy_idx),
                "source_profile": str(source_name),
                "target_old_profile": str(old_profile_name),
                "target_new_profile": str(new_profile_name),
                "decision": str(profile_copy_decision),
                "reject_reason": str(copy_result.reject_reason or ""),
                "eligible": bool(copy_result.eligible),
                "attempted": bool(copy_result.attempted),
                "accepted": bool(copy_result.accepted),
                "weighted_prob": float(copy_result.weighted_prob),
                "source_bonus": float(copy_result.source_bonus),
                "target_need_factor": float(copy_result.target_need_factor),
                "diversity_count": int(diversity_count),
                "diversity_after_copy": int(copy_result.diversity_after_copy),
                "source_nsucc_mean": float(source_meta.get("source_nsucc_mean", float("nan"))),
                "target_nsucc_mean": float(target_meta.get("source_nsucc_mean", float("nan"))),
                "source_has_retention": bool(
                    source_meta.get("has_source_retention", False)
                ),
                "target_has_retention": bool(
                    target_meta.get("has_source_retention", False)
                ),
                "source_retention": (
                    float(source_meta["source_retention"])
                    if source_meta.get("has_source_retention")
                    else None
                ),
                "target_retention": (
                    float(target_meta["source_retention"])
                    if target_meta.get("has_source_retention")
                    else None
                ),
                "profile_generation": int(self._wm_strategy_profile_generation),
                "weight_generation": int(self._wm_strategy_profile_generation),
            }
        )
        self._update_policy_lineage(int(env_frames))

        # Write restart manifest so the supervisor can pick up new params.
        # The supervisor script ($RUN_DIR) is the parent of train_dir ($RUN_DIR/rl_games).
        supervisor_run_dir = self.train_dir.parent
        manifest_path = supervisor_run_dir / self.RESTART_MANIFEST
        manifest = {
            "source_policy": source_policy,
            "tier": tier,
            "env_frames": env_frames,
            "checkpoint": str(restart_checkpoint),
            "train_args": train_args,
            "wm_strategy": selected_strategy,
            "wm_ensemble_size": int(new_params.get("wm_ensemble_size", 3)),
            "wm_checkpoint_path": wm_ckpt_path,
            "wm_checkpoint_source_strategy": wm_ckpt_strategy,
            "wm_checkpoint_compatible": wm_ckpt_compatible,
            "source_curriculum": sc,
            "policy_idx": int(self.policy_idx),
            # Patch 2 (rev 2, 2026-05-30): anti-churn snapshot for diagnosis.
            "restart_count": int(self._restart_count),
            "frames_at_restart": int(self._frames_at_restart),
            "last_mutation_frame": int(self.last_mutation_frame),
            "min_frames_as_source_after_restart": int(
                self.min_frames_as_source_after_restart
            ),
            "min_frames_between_restarts": int(
                self.min_frames_between_restarts
            ),
            # Patch F (2026-05-31): self-restart governor state in manifest.
            "self_restart_count": int(self._self_restart_count),
            "cross_restart_count": int(self._cross_restart_count),
            "last_self_restart_frame": int(self._last_self_restart_frame),
            "restart_type": int(self._last_restart_type),
            "disable_self_restart": bool(self.disable_self_restart),
            "allow_mid_self_mutation": bool(self.allow_mid_self_mutation),
            "max_self_restarts_per_policy": int(self.max_self_restarts_per_policy),
            "min_frames_between_self_restarts": int(
                self.min_frames_between_self_restarts
            ),
            # Patch G (2026-05-31): tiered source selection metadata.
            "source_selection_mode": str(self.source_selection_mode),
            "source_checkpoint_type": str(self._last_source_checkpoint_type),
            "source_tier_label": str(self._last_source_tier_label),
            "recovery_restart_count": int(self._recovery_restart_count),
            "last_recovery_frame": int(self._last_recovery_frame),
            "enable_best_nsucc_recovery": bool(self.enable_best_nsucc_recovery),
            "pbt_wm_profile_copy_enable": bool(copy_enable),
            "pbt_wm_profile_copy_decision": str(profile_copy_decision),
            "pbt_wm_profile_copy_prob": float(copy_prob),
            "pbt_wm_profile_diversity_count": int(diversity_count),
            "wm_strategy_profile_name": str(new_profile_name),
            "wm_strategy_profile_id": int(PROFILE_IDS.get(new_profile_name, 0)),
            "wm_strategy_profile_generation": int(self._wm_strategy_profile_generation),
            "wm_strategy_profile_parent": int(self._wm_strategy_profile_parent),
            "wm_strategy_profile_copied_from_source": bool(
                self._wm_strategy_profile_copied_from_source
            ),
            "late_rescue": bool(late_rescue),
            "late_rescue_attempt_count": int(
                self._late_rescue_attempt_count.get(self.policy_idx, 0)
            ),
            "late_rescue_selected_count": int(
                self._late_rescue_selected_count.get(self.policy_idx, 0)
            ),
            "late_rescue_executed_count": int(
                self._late_rescue_executed_count.get(self.policy_idx, 0)
            ),
            "late_rescue_blocked_count": int(
                self._late_rescue_blocked_count.get(self.policy_idx, 0)
            ),
            "late_rescue_last_frame": int(
                self._late_rescue_frame.get(self.policy_idx, -1)
            ),
            "late_rescue_streak": int(
                self._late_rescue_streak.get(self.policy_idx, 0)
            ),
        }

        # ------------------------------------------------------------------
        # Commit point preparation.
        # Order: (optional source-safe already applied) → late-rescue control
        # persist → atomic manifest → mark committed → best-effort audit /
        # writer → SIGKILL. Any failure before committed triggers full
        # rollback of runtime + persistent byte snapshots.
        # ------------------------------------------------------------------
        if late_rescue:
            # Fail-closed if target control is missing/corrupt before we touch
            # executed_count or write a supervisor-consumable manifest.
            ctrl_path = (
                self._curriculum_control_dir / f"{int(self.policy_idx):03d}.json"
            )
            try:
                read_curriculum_control_required(ctrl_path)
            except LateRescueControlMissingError:
                self._safe_write_pbt_copy_event({
                    "event_type": "late_degradation_rescue_target_control_missing",
                    "source_policy": int(source_policy),
                    "target_policy": int(self.policy_idx),
                    "env_frames": int(env_frames),
                    "path": str(ctrl_path),
                })
                raise
            except LateRescueControlParseError:
                self._safe_write_pbt_copy_event({
                    "event_type": "late_degradation_rescue_target_control_parse_error",
                    "source_policy": int(source_policy),
                    "target_policy": int(self.policy_idx),
                    "env_frames": int(env_frames),
                    "path": str(ctrl_path),
                })
                raise
            prev_executed = int(
                self._late_rescue_executed_count.get(self.policy_idx, 0)
            )
            self._late_rescue_executed_count[self.policy_idx] = prev_executed + 1
            self._late_rescue_count[self.policy_idx] = self._late_rescue_executed_count[
                self.policy_idx
            ]
            self._late_rescue_frame[self.policy_idx] = int(env_frames)
            manifest["late_rescue_executed_count"] = int(
                self._late_rescue_executed_count[self.policy_idx]
            )
            manifest["late_rescue_last_frame"] = int(env_frames)
            # Persist quota/cooldown into control BEFORE manifest so a persist
            # failure never leaves a consumable restart without durable quota.
            self._persist_late_rescue_runtime_state(
                self.policy_idx, env_frames=int(env_frames)
            )

        if adaptive_rescue:
            prev_executed = int(
                self._adaptive_rescue_executed.get(self.policy_idx, 0)
            )
            self._adaptive_rescue_executed[self.policy_idx] = prev_executed + 1
            self._adaptive_rescue_executed_for_lineage[self.policy_idx] = int(
                self._adaptive_rescue_executed_for_lineage.get(self.policy_idx, 0)
            ) + 1
            self._adaptive_last_rescue_frame[self.policy_idx] = int(env_frames)
            self._adaptive_rescue_last_reason[self.policy_idx] = "executed"
            manifest["adaptive_rescue"] = True
            manifest["adaptive_rescue_executed_count"] = int(
                self._adaptive_rescue_executed[self.policy_idx]
            )
            manifest["adaptive_rescue_executed_for_lineage"] = int(
                self._adaptive_rescue_executed_for_lineage[self.policy_idx]
            )
            manifest["adaptive_rescue_last_frame"] = int(env_frames)
            # Persist adaptive sidecar BEFORE manifest so a persist failure
            # never leaves a consumable restart without durable quota.
            self._adaptive_precision_persist(self.policy_idx)

        # Atomic manifest write (commit step for supervisor consumption).
        manifest_tmp = manifest_path.with_suffix(
            f".{os.getpid()}.{_uuid_mod.uuid4().hex[:8]}.tmp"
        )
        manifest_tmp.write_text(json.dumps(manifest))
        os.replace(str(manifest_tmp), str(manifest_path))

        if txn is not None:
            txn["committed"] = True

        # ---- Post-commit: best-effort only; must not trigger rollback. ----
        if late_rescue:
            self._safe_write_pbt_copy_event({
                "event_type": "late_degradation_rescue_restart_manifest_committed",
                "source_type": str(
                    (late_rescue_decision or {}).get("checkpoint_type", "live")
                ),
                "source_policy": int(source_policy),
                "target_policy": int(self.policy_idx),
                "env_frames": int(env_frames),
                "manifest_path": str(manifest_path),
            })
            self._safe_write_pbt_copy_event({
                "event_type": "late_degradation_rescue_executed",
                "source_type": str(
                    (late_rescue_decision or {}).get("checkpoint_type", "live")
                ),
                "source_policy": int(source_policy),
                "target_policy": int(self.policy_idx),
                "mutation_disabled": True,
                "rescue_executed_count": self._late_rescue_executed_count[
                    self.policy_idx
                ],
                "late_rescue_last_frame": int(env_frames),
                "env_frames": int(env_frames),
            })

        if adaptive_rescue:
            self._safe_write_pbt_copy_event({
                "event_type": "adaptive_degradation_rescue_restart_manifest_committed",
                "rescue_mode": "adaptive",
                "source_type": str(
                    (adaptive_rescue_decision or {}).get("checkpoint_type", "live")
                ),
                "source_policy": int(source_policy),
                "target_policy": int(self.policy_idx),
                "env_frames": int(env_frames),
                "manifest_path": str(manifest_path),
            })
            self._safe_write_pbt_copy_event({
                "event_type": "adaptive_degradation_rescue_executed",
                "rescue_mode": "adaptive",
                "source_type": str(
                    (adaptive_rescue_decision or {}).get("checkpoint_type", "live")
                ),
                "source_policy": int(source_policy),
                "target_policy": int(self.policy_idx),
                "mutation_disabled": True,
                "rescue_executed_count": int(
                    self._adaptive_rescue_executed[self.policy_idx]
                ),
                "rescue_executed_for_lineage": int(
                    self._adaptive_rescue_executed_for_lineage[self.policy_idx]
                ),
                "adaptive_rescue_last_frame": int(env_frames),
                "env_frames": int(env_frames),
            })
            if self.writer is not None:
                try:
                    self.writer.add_scalar(
                        "adaptive_consolidation/rescue_executed_count",
                        float(self._adaptive_rescue_executed[self.policy_idx]),
                        env_frames,
                    )
                    self.writer.add_scalar(
                        "adaptive_consolidation/rescue_quota_used",
                        float(
                            self._adaptive_rescue_executed_for_lineage[self.policy_idx]
                        ),
                        env_frames,
                    )
                except Exception:
                    pass


        self.mutable_params = dict(new_params)
        self.mutable_params["wm_strategy"] = selected_strategy
        try:
            if self.writer is not None:
                # M4: restart-related scalars
                self.writer.add_scalar("pbt/source_policy", float(source_policy), env_frames)
                self.writer.add_scalar("pbt/tier_bottom", float(tier == "bottom"), env_frames)
                self.writer.add_scalar("pbt/restart_source_policy_idx", float(source_policy), env_frames)
                if "success_tolerance" in sc:
                    self.writer.add_scalar(
                        "pbt/restart_loaded_success_tolerance",
                        float(sc["success_tolerance"]),
                        env_frames,
                    )
                # Patch F (2026-05-31): emit restart_type + lifetime counters
                # for the soon-to-die process. The relaunched worker will pick
                # up the resume hooks and continue incrementing these.
                self.writer.add_scalar(
                    "pbt/restart_type", float(self._last_restart_type), env_frames
                )
                self.writer.add_scalar(
                    "pbt/self_restart_count",
                    float(self._self_restart_count),
                    env_frames,
                )
                self.writer.add_scalar(
                    "pbt/cross_restart_count",
                    float(self._cross_restart_count),
                    env_frames,
                )
                self.writer.flush()
                self.writer.close()
        except Exception as writer_exc:
            print(f"[PBT] post-commit writer failed (ignored): {writer_exc}")

        print(
            f"Policy {self.policy_idx}: DexPBT {tier} restart from policy {source_policy} "
            f"with checkpoint {restart_checkpoint} — "
            f"requesting supervisor restart (exit code {self.RESTART_EXIT_CODE})"
        )
        try:
            sys.stdout.flush()
            sys.stderr.flush()
        except Exception:
            pass

        # NOTE: We intentionally do NOT close env or SimulationApp here.
        # Isaac Sim's close() deadlocks when called from within an RL callback.
        # The supervisor restarts us as a fresh process, so GPU resources are
        # reclaimed by the OS when this process exits.
        #
        # Use SIGKILL to guarantee the process dies — os._exit() alone can be
        # blocked by native extensions (Omniverse/Carb) that install atexit
        # handlers or hold internal locks during shutdown.
        # Return True for unit tests that mock process exit; production never
        # observes the return because the process is killed immediately after.
        import signal as _signal
        try:
            sys.stdout.flush()
            sys.stderr.flush()
        except Exception:
            pass
        try:
            _signal.raise_signal(_signal.SIGKILL)
        except Exception:
            pass
        try:
            os._exit(self.RESTART_EXIT_CODE)
        except Exception:
            return True
        return True

    def after_print_stats(self, frame, epoch_num, total_time):
        super().after_print_stats(frame, epoch_num, total_time)
        if (
            self.frame_stop_enable
            and self.target_population_frames > 0
            and self.algo is not None
        ):
            population_frame = pbt_population_frame(self.algo)
            progress = min(1.0, population_frame / float(self.target_population_frames))
            if self.writer is not None:
                self.writer.add_scalar("training/population_frame", float(population_frame), population_frame)
                self.writer.add_scalar("training/frame_target", float(self.target_population_frames), population_frame)
                self.writer.add_scalar("training/frame_target_progress", progress, population_frame)
            if (
                not self._frame_stop_triggered
                and population_frame >= self.target_population_frames
            ):
                # rl_games checks max_epochs at its next epoch boundary.  Do
                # not call os._exit: that would bypass normal checkpoint and
                # observer teardown.
                self._frame_stop_triggered = True
                self.algo.max_epochs = int(epoch_num) + self.frame_stop_grace_iterations
                if self.writer is not None:
                    self.writer.add_scalar("training/frame_stop_triggered", 1.0, population_frame)

    # ------------------------------------------------------------------
    # WM / Counterfactual helper methods
    # ------------------------------------------------------------------

    def _init_wm_components(self) -> None:
        """Lazy-init WM components only when enable_wm=True."""
        global _WM_COUNTERFACTUAL, _WM_POSTERIOR
        if _WM_COUNTERFACTUAL is None:
            from dexpbt_isaaclab.world_model.wm_counterfactual import CounterfactualManager
            _WM_COUNTERFACTUAL = CounterfactualManager
        if _WM_POSTERIOR is None:
            from dexpbt_isaaclab.world_model.wm_posterior_selector import SharedPosteriorSelector
            _WM_POSTERIOR = SharedPosteriorSelector

        # Track C (2026-06-06): use the run-configured pool instead of a
        # hardcoded list. self.wm_strategy_pool is validated in __init__.
        strategies = list(self.wm_strategy_pool)

        # Posterior selector (workspace-level shared)
        shared_posterior_path = self.workspace / "wm_posterior_latest.json"
        self._posterior_selector = _WM_POSTERIOR(
            shared_path=shared_posterior_path,
            strategies=strategies,
            none_fixed_mean=True,
        )
        self._posterior_selector.load_or_init(policy_idx=self.policy_idx)

        # WM checkpoint manager
        policy_wm_dir = self.curr_policy_workspace
        from dexpbt_isaaclab.world_model.wm_checkpoint import WMCheckpointManager
        self._wm_checkpoint_manager = WMCheckpointManager(policy_wm_dir)

        # Counterfactual manager
        if self._enable_counterfactual:
            mp = self.mutable_params
            self._cf_manager = _WM_COUNTERFACTUAL(
                workspace=self.workspace,
                strategies=strategies,
                interval_gens=int(mp.get("cf_interval_gens", 5)),
                env_task=self.task_name,
                num_envs=int(mp.get("cf_num_envs", 4096)),
                source_num_envs=int(self.num_envs),
                max_parallel_children=int(mp.get("cf_max_parallel_children", 2)),
                cf_cuda_visible_devices=str(mp.get("cf_cuda_visible_devices", "4,5")),
                cf_horizon=int(mp.get("cf_horizon", 16)),
                cf_steps=int(mp.get("cf_steps", 5_000_000)),
                # Patch 3 + 4 (rev 2, 2026-05-30): per-block child / delta
                # configuration. Read from mutable_params so a launcher
                # (start_regrasping_cbwm.sh) or restart manifest can
                # override per-run without re-deploying the package.
                cf_train_frames=int(mp.get("cf_train_frames", 20_000_000)),
                cf_reset_epoch_counter=bool(mp.get("cf_reset_epoch_counter", True)),
                cf_min_eval_s=float(mp.get("cf_min_eval_s", 30.0)),
                cf_delta_mode=str(mp.get("cf_delta_mode", "learning_improvement")),
                cf_low_signal_action=str(mp.get("cf_low_signal_action", "skip")),
                cf_low_signal_weight=float(mp.get("cf_low_signal_weight", 0.25)),
                cf_require_nsucc_gt_zero=bool(
                    mp.get("cf_require_nsucc_gt_zero", True)
                ),
                # Patch I (2026-05-31): tighter source-health gates and
                # nsucc-based delta metric. Defaults preserve Patch 3
                # behaviour (no extra gating, r_meta-based delta).
                cf_source_min_nsucc_mean=float(
                    mp.get("cf_source_min_nsucc_mean", 0.0)
                ),
                cf_source_min_nsucc_max=float(
                    mp.get("cf_source_min_nsucc_max", 0.0)
                ),
                cf_source_min_ranking_score=float(
                    mp.get("cf_source_min_ranking_score", float("-inf"))
                ),
                cf_delta_metric=str(mp.get("cf_delta_metric", "r_meta")),
            )

        print(f"[WM] init: strategy={self._wm_strategy}, ensemble={self._wm_ensemble_size}, "
              f"cf={self._enable_counterfactual}")

    def _maybe_consume_counterfactual_result(self, env_frames: int) -> bool:
        """Policy0-only: consume ``cf_block_result.json`` without a PBT restart."""
        if not self._is_posterior_writer:
            return False
        if self._cf_manager is None or self._posterior_selector is None:
            return False

        result_payload = self._cf_manager.peek_unconsumed_result(
            self._last_consumed_cf_block_id
        )
        if self.writer is not None:
            self.writer.add_scalar(
                "wm_cf/result_seen",
                1.0 if result_payload else 0.0,
                env_frames,
            )
        if not result_payload:
            return False

        block_id = int(result_payload.get("block_id", -1))
        if block_id <= self._last_consumed_cf_block_id:
            print(
                f"[CF] skip posterior update: duplicate block_id={block_id} "
                f"(last_consumed={self._last_consumed_cf_block_id})"
            )
            if self.writer is not None:
                self.writer.add_scalar(
                    "wm_cf/result_skipped_duplicate", 1.0, env_frames
                )
            return False

        delta_w = result_payload.get("delta_w", {}) or {}
        posterior_weights = result_payload.get("posterior_weights", {}) or {}
        invalid_block = bool(result_payload.get("invalid_block", False))
        all_strategies_tied = bool(result_payload.get("all_strategies_tied", False))
        cf_delta_mode_log = str(result_payload.get("cf_delta_mode", ""))

        self._cf_result_count += 1
        self._last_cf_result_timestamp = float(
            result_payload.get("timestamp", time.time())
        )

        if self.writer is not None:
            self.writer.add_scalar(
                "wm_cf/result_count", float(self._cf_result_count), env_frames
            )
            self.writer.add_scalar(
                "wm_cf/invalid_block", 1.0 if invalid_block else 0.0, env_frames
            )
            self.writer.add_scalar(
                "wm_cf/all_strategies_tied",
                1.0 if all_strategies_tied else 0.0,
                env_frames,
            )
            self.writer.add_scalar(
                "wm_cf/delta_mode_hash",
                float(abs(hash(cf_delta_mode_log)) % 100),
                env_frames,
            )

        consumed = False
        if invalid_block:
            print(
                f"[CF] skip posterior update: invalid_block "
                f"(reason={result_payload.get('invalid_block_reason')})"
            )
        elif delta_w:
            self._posterior_selector.update_batch(
                delta_w,
                step=env_frames,
                weights=posterior_weights or None,
            )
            self._posterior_selector.save()
            self._last_posterior_update_policy_idx = int(self.policy_idx)
            self._cf_posterior_update_count += 1
            consumed = True
            print(
                f"[CF] Updated posterior with delta_w={delta_w}, "
                f"weights={posterior_weights}, mode={cf_delta_mode_log}, "
                f"block_id={block_id}"
            )
            if self.writer is not None:
                self.writer.add_scalar(
                    "wm_cf/result_consumed", 1.0, env_frames
                )
                self.writer.add_scalar(
                    "wm_cf/posterior_update_count",
                    float(self._cf_posterior_update_count),
                    env_frames,
                )
                self.writer.add_scalar(
                    "wm_posterior/last_consumed_block_id",
                    float(block_id),
                    env_frames,
                )
                for s, d in delta_w.items():
                    self.writer.add_scalar(f"wm_cf/delta_{s}", d, env_frames)
                for s, w in (posterior_weights or {}).items():
                    self.writer.add_scalar(
                        f"wm_cf/posterior_weight_{s}", float(w), env_frames
                    )
        else:
            print("[CF] skip posterior update: empty delta_w")

        self._cf_manager.consume_result_file()
        self._last_consumed_cf_block_id = block_id
        return consumed

    def _check_cf_trigger(
        self,
        population: dict[int, dict[str, Any] | None],
        env_frames: int,
    ) -> None:
        """Check if CF block should be triggered at current PBT iteration.

        Patch I (2026-05-31): the source pick is now health-aware. We
        iterate over the *entire* population (not just the rank-1 worker)
        and keep the highest `pbt_ranking_score` candidate that *also*
        passes the configured CF source-health gates. This avoids the
        smoke-run 20260530_200619 failure mode where the apparent top
        worker had a high `objective` driven entirely by tolerance
        progress (with `nsucc_mean=0`), wasting CF compute on a baseline
        whose successes were structurally zero.
        """
        if self._cf_manager is None:
            return

        # Health gate thresholds from the manager (set by Patch I CLI).
        gate_n_mean = float(self._cf_manager.cf_source_min_nsucc_mean)
        gate_n_max = float(self._cf_manager.cf_source_min_nsucc_max)
        gate_rs = float(self._cf_manager.cf_source_min_ranking_score)

        top_state = None
        top_r_meta = float("-inf")
        best_score = float("-inf")
        stale_meta_count = 0
        top_stable_mean = float("nan")
        for policy_idx, state in population.items():
            if state is None:
                continue
            n_mean = self._state_nsucc_mean(state)
            n_max = self._state_nsucc_max(state)
            rs = float(
                state.get(
                    "pbt_ranking_score",
                    state.get("objective", float("-inf")),
                )
                or float("-inf")
            )
            source_frame = int(state.get("frame", 0) or 0)
            if source_frame < env_frames - self.interval_steps:
                stale_meta_count += 1
                continue
            if n_mean < gate_n_mean or n_max < gate_n_max or rs < gate_rs:
                continue
            if self.source_require_stable_mean:
                window = self._policy_window(policy_idx)
                stable_ok, stable_reason = is_stable_source_window(
                    window,
                    min_mean=self.stable_source_min_mean,
                    min_max=self.stable_source_min_max,
                    max_drop_ratio=self.stable_source_max_drop_ratio,
                    drop_windows=self.stable_source_drop_windows,
                )
                if not stable_ok:
                    continue
                if (
                    n_mean < self.stable_source_min_mean
                    or n_max < self.stable_source_min_max
                ):
                    continue
            obj = float(state.get("objective", float("-inf")))
            if math.isnan(obj) or math.isinf(obj):
                continue
            if rs > best_score:
                best_score = rs
                top_r_meta = obj
                top_state = state
                if self.source_require_stable_mean:
                    top_stable_mean = float(
                        window.samples[-1].nsucc_mean if window.samples else n_mean
                    )

        if top_state is None:
            if stale_meta_count > 0:
                self._cf_skipped_stale_meta_count += 1
            skip_msg = (
                "[CF] skip trigger: no policy passed health gates "
                f"(min_nsucc_mean={gate_n_mean}, min_nsucc_max={gate_n_max}, "
                f"min_ranking_score={gate_rs}, stale_meta={stale_meta_count}"
            )
            if self.source_require_stable_mean:
                skip_msg += (
                    f", require_stable_mean=True "
                    f"(stable_min_mean={self.stable_source_min_mean}, "
                    f"stable_min_max={self.stable_source_min_max})"
                )
            skip_msg += ")"
            print(skip_msg)
            if self.writer is not None:
                self.writer.add_scalar(
                    "wm_cf/skipped_stale_source_meta",
                    float(self._cf_skipped_stale_meta_count),
                    env_frames,
                )
                self.writer.add_scalar(
                    "wm_cf/skipped_source_health_gate",
                    float(self._cf_manager.skipped_source_health_gate_count + 1),
                    env_frames,
                )
            # Bump the manager-side counter so the per-tick TB scalar tracks.
            self._cf_manager.skipped_source_health_gate_count += 1
            return

        top_checkpoint = str(top_state.get("checkpoint", ""))
        if not top_checkpoint or not Path(top_checkpoint).exists():
            if self.writer is not None:
                self.writer.add_scalar("wm_cf/source_checkpoint_exists", 0.0, env_frames)
            print("[CF] skip trigger: top worker checkpoint missing/unreadable")
            return

        top_curriculum = top_state.get("curriculum") or {}
        top_tolerance = top_curriculum.get("success_tolerance", None)
        if top_tolerance is None:
            print("[CF] skip trigger: top worker success_tolerance unavailable")
            return

        top_params = dict(top_state.get("params", {}))
        top_policy_idx = int(top_state.get("policy_idx", -1))
        # Patch 3 (rev 2, 2026-05-30): pass source nsucc + ranking_score
        # so the manager's nsucc-gate has real data (not just defaults).
        source_nsucc_mean = float(
            top_state.get(
                "nsucc_mean",
                (top_curriculum or {}).get("mean_prev_episode_successes", 0.0),
            ) or 0.0
        )
        source_nsucc_max = float(
            top_state.get(
                "nsucc_max",
                (top_curriculum or {}).get("max_prev_episode_successes", 0.0),
            ) or 0.0
        )
        source_ranking_score = float(
            top_state.get("pbt_ranking_score", top_state.get("objective", 0.0))
            or 0.0
        )
        source_meta_frame = int(top_state.get("frame", env_frames))
        gate_pass = (
            source_nsucc_mean >= gate_n_mean
            and source_nsucc_max >= gate_n_max
            and source_ranking_score >= gate_rs
        )
        if self.writer is not None:
            self.writer.add_scalar(
                "wm_cf/source_gate_pass", 1.0 if gate_pass else 0.0, env_frames
            )
            self.writer.add_scalar(
                "wm_cf/source_meta_frame", float(source_meta_frame), env_frames
            )
            self.writer.add_scalar(
                "wm_cf/source_checkpoint_type", 0.0, env_frames
            )
            if self.source_require_stable_mean:
                self.writer.add_scalar(
                    "dexpbt/source_stable_mean",
                    float(top_stable_mean),
                    env_frames,
                )
        triggered = self._cf_manager.check_trigger(
            pbt_generation=self.pbt_iteration,
            source_policy_idx=top_policy_idx,
            source_r_meta=float(top_r_meta),
            top_checkpoint=top_checkpoint,
            top_params=top_params,
            top_tolerance=float(top_tolerance),
            top_curriculum=top_curriculum,
            source_env_frames=int(top_state.get("frame", env_frames)),
            source_nsucc_mean=source_nsucc_mean,
            source_nsucc_max=source_nsucc_max,
            source_ranking_score=source_ranking_score,
            source_meta_frame=source_meta_frame,
            source_checkpoint_type="live",
        )
        if triggered:
            self._cf_request_count += 1
        if self.writer is not None:
            # Patch 3 (rev 2): nsucc-gate observability fires every tick so
            # we can see the skip-counter grow without waiting for a
            # successful trigger.
            self.writer.add_scalar(
                "wm_cf/skipped_nsucc_gate",
                float(self._cf_manager.skipped_nsucc_gate_count),
                env_frames,
            )
            # Patch I (2026-05-31): tier-specific skip counter.
            self.writer.add_scalar(
                "wm_cf/skipped_source_health_gate",
                float(self._cf_manager.skipped_source_health_gate_count),
                env_frames,
            )
            self.writer.add_scalar(
                "wm_cf/source_min_nsucc_mean",
                float(self._cf_manager.cf_source_min_nsucc_mean),
                env_frames,
            )
            self.writer.add_scalar(
                "wm_cf/source_min_nsucc_max",
                float(self._cf_manager.cf_source_min_nsucc_max),
                env_frames,
            )
            self.writer.add_scalar(
                "wm_cf/source_min_ranking_score",
                float(self._cf_manager.cf_source_min_ranking_score),
                env_frames,
            )
            self.writer.add_scalar(
                "wm_cf/delta_metric_hash",
                float(abs(hash(self._cf_manager.cf_delta_metric)) % 100),
                env_frames,
            )
            self.writer.add_scalar(
                "wm_cf/source_nsucc_mean",
                float(source_nsucc_mean),
                env_frames,
            )
            self.writer.add_scalar(
                "wm_cf/source_ranking_score",
                float(source_ranking_score),
                env_frames,
            )
        if triggered and self.writer is not None:
            self.writer.add_scalar("wm_cf/request_written", 1.0, env_frames)
            self.writer.add_scalar("wm_cf/request_count", float(self._cf_request_count), env_frames)
            self.writer.add_scalar("wm_cf/source_policy_idx", float(top_policy_idx), env_frames)
            self.writer.add_scalar("wm_cf/source_r_meta", float(top_r_meta), env_frames)
            self.writer.add_scalar("wm_cf/source_success_tolerance", float(top_tolerance), env_frames)
            self.writer.add_scalar("wm_cf/source_checkpoint_exists", 1.0, env_frames)
            self.writer.add_scalar("wm_cf/cf_num_envs", float(self._cf_manager.num_envs), env_frames)
            self.writer.add_scalar("wm_cf/source_num_envs", float(self.num_envs), env_frames)
            self.writer.add_text("wm_cf/source_checkpoint", top_checkpoint, env_frames)

    def _get_wm_checkpoint_path(self, strategy: str, source_policy: int) -> str | None:
        """
        WM checkpoint warm-start rules:
        aux → value_imag:  inherit WM weights
        value_imag_h1 → h3: inherit
        rssm_* / trans_* within-family: inherit wm_seq_latest.pt
        any → none:         reset
        """
        if strategy == "none":
            return None

        from dexpbt_isaaclab.world_model.wm_trainer import (
            RSSM_FAMILY,
            SEQUENCE_STRATEGIES,
            TRANS_FAMILY,
        )

        if strategy in SEQUENCE_STRATEGIES:
            src_dir = self.workspace / f"{source_policy:03d}" / "wm_seq_checkpoints"
            src_ckpt = src_dir / "wm_seq_latest.pt"
            if not src_ckpt.exists():
                return None
            src_strategy = None
            try:
                import torch

                payload = torch.load(
                    src_ckpt, map_location="cpu", weights_only=False
                )
                src_strategy = str(payload.get("strategy") or "")
            except Exception:
                src_strategy = None
            if strategy in RSSM_FAMILY and (src_strategy or "") in RSSM_FAMILY:
                return str(src_ckpt)
            if strategy in TRANS_FAMILY and (src_strategy or "") in TRANS_FAMILY:
                return str(src_ckpt)
            if src_strategy == strategy:
                return str(src_ckpt)
            return None

        src_dir = self.workspace / f"{source_policy:03d}" / "wm_checkpoints"
        src_ckpt = src_dir / "wm_latest.pt"
        if not src_ckpt.exists():
            return None

        src_strategy = None
        marker = src_dir / "strategy.txt"
        if marker.exists():
            src_strategy = marker.read_text().strip()

        warm_start_pairs = {
            ("aux_low", "value_imag_h1"): True,
            ("aux_mid", "value_imag_h1"): True,
            ("aux_low", "value_imag_h3"): True,
            ("aux_mid", "value_imag_h3"): True,
            ("value_imag_h1", "value_imag_h3"): True,
            ("aux_low", "aux_low"): True,
            ("aux_mid", "aux_mid"): True,
            ("value_imag_h1", "value_imag_h1"): True,
            ("value_imag_h3", "value_imag_h3"): True,
        }

        key = (src_strategy or "none", strategy)
        if warm_start_pairs.get(key, False):
            return str(src_ckpt)
        return None

    def log_wm_metrics(self, metrics: dict, step: int) -> None:
        """Log WM training metrics to tensorboard."""
        if self.writer is None or not metrics:
            return
        for key, val in metrics.items():
            self.writer.add_scalar(key, val, step)

    def _log_cf_runtime_status(self, env_frames: int) -> None:
        if self.writer is None or not self._enable_counterfactual:
            return
        heartbeat_age = self._read_runner_heartbeat_age()
        self.writer.add_scalar("wm_cf/request_count", float(self._cf_request_count), env_frames)
        self.writer.add_scalar("wm_cf/result_count", float(self._cf_result_count), env_frames)
        self.writer.add_scalar("wm_cf/posterior_update_count", float(self._cf_posterior_update_count), env_frames)
        self.writer.add_scalar(
            "wm_cf/runner_heartbeat_age_sec",
            float(heartbeat_age if math.isfinite(heartbeat_age) else -1.0),
            env_frames,
        )
        latest_result_age = float("inf")
        if self._last_cf_result_timestamp > 0.0:
            latest_result_age = max(0.0, time.time() - self._last_cf_result_timestamp)
        self.writer.add_scalar(
            "wm_cf/latest_result_age_sec",
            float(latest_result_age if math.isfinite(latest_result_age) else -1.0),
            env_frames,
        )
        if bool(self.mutable_params.get("require_counterfactual_runner", False)) and (
            not math.isfinite(heartbeat_age) or heartbeat_age > 300.0
        ):
            raise RuntimeError("require_counterfactual_runner=True but CF runner heartbeat is stale/missing")
        if heartbeat_age > 900.0:
            print(f"[CF] WARNING: runner heartbeat stale: age={heartbeat_age:.1f}s")
        if self._cf_request_count > 0 and self._cf_posterior_update_count == 0:
            print("[CF] WARNING: CF requests exist but posterior_update_count is still 0")

    def _read_runner_heartbeat_age(self) -> float:
        hb_path = self.workspace / "counterfactual" / "runner_heartbeat.json"
        if not hb_path.exists():
            return float("inf")
        try:
            data = json.loads(hb_path.read_text())
            ts = float(data.get("timestamp", 0.0))
            if ts <= 0.0:
                return float("inf")
            return max(0.0, time.time() - ts)
        except Exception:
            return float("inf")
