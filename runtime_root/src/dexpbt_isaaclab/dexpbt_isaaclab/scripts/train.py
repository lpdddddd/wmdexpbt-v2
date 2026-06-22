"""
Training script for DexPBT Kuka-Allegro tasks with RL-Games.

Usage:
    python -m dexpbt_isaaclab.scripts.train --task DexPBT-Reorientation-KukaAllegro-Direct-v0 --headless
    python -m dexpbt_isaaclab.scripts.train --task DexPBT-Regrasping-KukaAllegro-Direct-v0 --headless --num_envs 512
"""

import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Train DexPBT with RL-Games on Isaac Lab.")
parser.add_argument("--task", type=str, default="DexPBT-Reorientation-KukaAllegro-Direct-v0")
parser.add_argument("--num_envs", type=int, default=None)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--max_iterations", type=int, default=None)
parser.add_argument("--checkpoint", type=str, default=None)
parser.add_argument("--train_root", type=str, default=None)
parser.add_argument("--experiment_name", type=str, default=None)
parser.add_argument("--observer_mode", type=str, choices=["default", "pbt"], default="default")
parser.add_argument("--policy_idx", type=int, default=0)
parser.add_argument("--num_policies", type=int, default=6)
parser.add_argument("--workspace", type=str, default="dexpbt_single_arm_ws")
parser.add_argument("--pbt_interval_steps", type=int, default=20_000_000)
parser.add_argument("--pbt_start_after", type=int, default=200_000_000)
parser.add_argument("--pbt_initial_delay", type=int, default=50_000_000)
parser.add_argument("--pbt_mutation_rate", type=float, default=0.2)
parser.add_argument("--pbt_change_min", type=float, default=1.1)
parser.add_argument("--pbt_change_max", type=float, default=1.5)
# Patch 1 (rev 2, 2026-05-30): success-aware PBT ranking score (decoupled
# from the raw DexPBT tolerance_successes_objective formula). Default mode
# avoids ranking workers with nsucc=0 by their tolerance-progress alone.
# See dexpbt_utils.compute_pbt_ranking_score docstring.
parser.add_argument(
    "--pbt_ranking_metric",
    type=str,
    choices=["raw_r_meta", "success_aware_r_meta"],
    default="success_aware_r_meta",
    help="PBT selection score mode. raw_r_meta = legacy DexPBT formula; "
    "success_aware_r_meta caps score to --pbt_nsucc_zero_cap when nsucc=0.",
)
parser.add_argument(
    "--pbt_nsucc_zero_cap",
    type=float,
    default=0.05,
    help="Cap applied to pbt_ranking_score when mean_nsucc <= 0 "
    "(only when --pbt_ranking_metric=success_aware_r_meta).",
)
# Patch B (rev 3, 2026-05-30): low-nsucc cap for the 0<nsucc<threshold
# band. The default of 1.0 mirrors the Patch A source gate so the cap is
# triggered for the same workers the gate filters out. Set threshold to
# 0.0 to disable this Patch B layer (recovers Patch 1 behaviour).
parser.add_argument(
    "--pbt_nsucc_low_threshold",
    type=float,
    default=1.0,
    help="Patch B ranking-cap activation threshold: when 0<nsucc<X the "
    "ranking_score is capped at --pbt_nsucc_low_cap. Set to 0.0 to "
    "disable (recover Patch 1 behaviour).",
)
parser.add_argument(
    "--pbt_nsucc_low_cap",
    type=float,
    default=0.05,
    help="Cap value applied when 0<nsucc<--pbt_nsucc_low_threshold.",
)
# Patch 2 (rev 2, 2026-05-30): anti-churn / N_adapt. Two independent gates;
# see pbt_observer._compute_eligibility helpers.
parser.add_argument(
    "--pbt_min_frames_as_source_after_restart",
    type=int,
    default=50_000_000,
    help="Source-cooldown: a worker is not eligible as PBT top source until "
    "it has trained at least this many frames since its last restart.",
)
parser.add_argument(
    "--pbt_min_frames_between_restarts",
    type=int,
    default=50_000_000,
    help="Elimination-cooldown: a worker is not eligible for elimination "
    "until it has trained at least this many frames since its last restart.",
)
# Patch A (rev 3, 2026-05-30): nsucc-threshold source gate. A worker whose
# success signal is below these thresholds is not allowed to be copied as
# a PBT top source -- this prevents the "low nsucc + high r_meta_raw"
# collapse vector observed in regrasping_cbwm_20260530_160513.
parser.add_argument(
    "--pbt_min_nsucc_mean_as_source",
    type=float,
    default=1.0,
    help="Source-eligibility gate: nsucc_mean must be >= this to be a "
    "PBT source. Default 1.0 filters out 'leading workers' that have "
    "near-zero successes but high r_meta_raw from tolerance progress.",
)
parser.add_argument(
    "--pbt_min_nsucc_max_as_source",
    type=float,
    default=5.0,
    help="Source-eligibility gate: nsucc_max must be >= this to be a "
    "PBT source. Default 5 filters out workers with only one-off "
    "lucky successes.",
)
parser.add_argument(
    "--pbt_allow_low_nsucc_source_fallback",
    action=argparse.BooleanOptionalAction,
    default=False,
    help="When ALL top candidates fail the nsucc gate, default behaviour "
    "is to SKIP this PBT replacement tick. Set --pbt_allow_low_nsucc_"
    "source_fallback to restore the legacy fallback (re-admit ineligible "
    "top sources) for rollback / ablation.",
)
# Patch F (2026-05-31): mid-tier self-restart governor. The legacy DexPBT
# behaviour was to relaunch the mid worker on every tick with mutated
# params; smoke 20260530 showed 100% of restarts were self-restarts which
# broke curriculum continuity. Defaults below DISABLE mid self-restart.
parser.add_argument(
    "--pbt_disable_self_restart",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Master switch: when True, mid-tier workers never relaunch "
    "themselves (legacy DexPBT self-mutation off). Set False (or pair "
    "with --pbt_allow_mid_self_mutation) to recover legacy behaviour.",
)
parser.add_argument(
    "--pbt_allow_mid_self_mutation",
    action=argparse.BooleanOptionalAction,
    default=False,
    help="Legacy escape hatch: when True overrides --pbt_disable_self_restart "
    "and re-enables the mid-tier self-mutation + restart path subject to "
    "the count/cooldown gates below.",
)
parser.add_argument(
    "--pbt_max_self_restarts_per_policy",
    type=int,
    default=1,
    help="Lifetime cap on self-restarts per worker (only consulted when "
    "self-restart is enabled).",
)
parser.add_argument(
    "--pbt_min_frames_between_self_restarts",
    type=int,
    default=100_000_000,
    help="Cooldown (frames) between consecutive self-restarts of the same "
    "worker (only consulted when self-restart is enabled).",
)
# Patch G (2026-05-31): tiered source selection + best_nsucc recovery.
parser.add_argument(
    "--pbt_source_selection_mode",
    type=str,
    default="tiered",
    choices=["tiered", "strict"],
    help="Bottom-tier source selection mode. tiered (default) cascades "
    "L1 strict -> L2 relaxed -> L3 archive -> L4 skip. strict is the "
    "Patch A behaviour (strict-or-skip) and is provided for ablation.",
)
parser.add_argument(
    "--pbt_relaxed_nsucc_mean_as_source",
    type=float,
    default=0.5,
    help="L2 relaxed gate: a worker is a relaxed source when "
    "nsucc_mean >= this threshold (default 0.5).",
)
parser.add_argument(
    "--pbt_relaxed_nsucc_max_as_source",
    type=float,
    default=2.0,
    help="L2 relaxed gate: a worker is a relaxed source when "
    "nsucc_max >= this threshold (default 2.0).",
)
parser.add_argument(
    "--pbt_enable_best_nsucc_recovery",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Enable L3 recovery: when no live source passes either gate, "
    "restart the bottom worker from the best_nsucc/ peak archive.",
)
parser.add_argument(
    "--pbt_recovery_min_nsucc_mean",
    type=float,
    default=2.0,
    help="L3 recovery gate: the archived peak must have "
    "nsucc_mean >= this to be eligible for recovery (default 2.0).",
)
parser.add_argument(
    "--pbt_recovery_min_nsucc_max",
    type=float,
    default=10.0,
    help="L3 recovery gate: the archived peak must have "
    "nsucc_max >= this to be eligible for recovery (default 10.0).",
)
parser.add_argument(
    "--pbt_recovery_cooldown_frames",
    type=int,
    default=200_000_000,
    help="Cooldown (frames) between consecutive archive recoveries of "
    "the same worker (default 200M).",
)
# Patch H (2026-05-31): post-restart curriculum freeze knobs (set on PBT
# observer; the env-side gate is set by pbt_observer via the
# --curriculum_freeze_frames_after_restart_resume hook on each restart).
parser.add_argument(
    "--pbt_curriculum_freeze_frames",
    type=int,
    default=100_000_000,
    help="Frames (since restart) during which curriculum tightening is "
    "frozen after a cross-copy / self-restart (default 100M).",
)
parser.add_argument(
    "--pbt_curriculum_freeze_on_recovery_frames",
    type=int,
    default=100_000_000,
    help="Freeze applied after archive / best_stable recovery "
    "(default 100M env frames).",
)
# Patch P1 (2026-06-01): mastery curriculum + stable-source cascade.
parser.add_argument(
    "--curriculum_mode",
    type=str,
    default="legacy",
    choices=["legacy", "mastery"],
    help="Curriculum control mode written by the PBT observer. legacy "
    "preserves existing env-side tightening; mastery gates shrink via "
    "curriculum_control/*.json.",
)
parser.add_argument(
    "--pbt_source_require_stable_mean",
    action=argparse.BooleanOptionalAction,
    default=False,
    help="When True, bottom-tier source selection uses the P1 stable "
    "cascade (strict_stable -> usable -> relaxed -> best_stable -> skip).",
)
parser.add_argument(
    "--mastery_nsucc_mean",
    type=float,
    default=4.0,
    help="Mastery gate: consecutive windows must have nsucc_mean >= this "
    "(default 4.0 for smoke).",
)
parser.add_argument(
    "--mastery_nsucc_max",
    type=float,
    default=20.0,
    help="Mastery gate: consecutive windows must have nsucc_max >= this.",
)
parser.add_argument(
    "--mastery_consecutive_windows",
    type=int,
    default=2,
    help="Mastery gate: number of consecutive PBT windows required.",
)
parser.add_argument(
    "--curriculum_shrink_factor",
    type=float,
    default=0.95,
    help="Tolerance shrink factor when mastery gate passes (default 0.95).",
)
parser.add_argument(
    "--pbt_allow_best_nsucc_archive_recovery",
    action=argparse.BooleanOptionalAction,
    default=False,
    help="Allow best_nsucc/ archive recovery when stable-source mode is "
    "enabled. Legacy mode still uses --pbt_enable_best_nsucc_recovery.",
)
parser.add_argument(
    "--curriculum_collapse_detect",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Enable collapse detection and post-collapse curriculum freeze.",
)
parser.add_argument(
    "--collapse_drop_ratio",
    type=float,
    default=0.5,
    help="Collapse detection: nsucc_mean drop vs recent peak threshold.",
)
parser.add_argument(
    "--collapse_patience",
    type=int,
    default=2,
    help="Collapse detection: consecutive bad windows before freeze.",
)
parser.add_argument(
    "--collapse_freeze_env_frames",
    type=int,
    default=200_000_000,
    help="Env frames to freeze curriculum after collapse detection.",
)
parser.add_argument(
    "--stable_source_min_mean",
    type=float,
    default=3.0,
    help="Stable live-source gate: rolling nsucc_mean >= this.",
)
parser.add_argument(
    "--stable_source_min_max",
    type=float,
    default=20.0,
    help="Stable live-source gate: rolling nsucc_max >= this.",
)
parser.add_argument(
    "--stable_source_max_drop_ratio",
    type=float,
    default=0.3,
    help="Stable live-source gate: max drop ratio over drop_windows.",
)
parser.add_argument(
    "--stable_source_drop_windows",
    type=int,
    default=2,
    help="Stable live-source gate: window length for drop-ratio check.",
)
# Patch P2-A (2026-06-01): single-shot shrink token per PBT window.
parser.add_argument(
    "--pbt_single_shrink_per_window",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Issue at most one shrink_token per policy per PBT window; "
    "post-shrink consolidation blocks further shrink.",
)
parser.add_argument(
    "--post_shrink_consolidation_env_frames",
    type=int,
    default=100_000_000,
    help="Env frames to hold tolerance after a shrink (consolidation).",
)
# Phase A4 P2 (2026-06-08): adaptive-fast curriculum (RSSM coupling=none
# fastcurr 800M). Default `normal` preserves legacy single-factor shrink
# behaviour bit-for-bit; `adaptive_fast` picks shrink factor + consolidation
# length per-tick from a tier table once the existing mastery / collapse /
# consolidation / post-restart-freeze gates have already permitted shrink.
parser.add_argument(
    "--curriculum_speed_mode",
    type=str,
    default="normal",
    choices=["normal", "adaptive_fast"],
    help="Curriculum speed mode. 'normal' uses --curriculum_shrink_factor "
    "for every shrink (legacy). 'adaptive_fast' selects shrink_factor "
    "0.90 / 0.925 / 0.95 by tier of recent nsucc_mean and retention.",
)
parser.add_argument(
    "--curriculum_fast_strong_mean",
    type=float,
    default=30.0,
    help="adaptive_fast: nsucc_mean threshold for the 'strong' tier "
    "(shrink_factor=--curriculum_fast_shrink_strong) when retention "
    ">= --curriculum_fast_retention_threshold.",
)
parser.add_argument(
    "--curriculum_fast_mid_mean",
    type=float,
    default=20.0,
    help="adaptive_fast: nsucc_mean threshold for the 'mid' tier "
    "(shrink_factor=--curriculum_fast_shrink_mid).",
)
parser.add_argument(
    "--curriculum_fast_low_mean",
    type=float,
    default=10.0,
    help="adaptive_fast: nsucc_mean threshold for the 'low' tier "
    "(shrink_factor=--curriculum_fast_shrink_low). Below this the tick "
    "is blocked with reason adaptive_fast_low_nsucc_mean.",
)
parser.add_argument(
    "--curriculum_fast_retention_threshold",
    type=float,
    default=0.8,
    help="adaptive_fast: minimum retention_proxy (=last/peak over the "
    "last --curriculum_fast_retention_window samples) to qualify for "
    "the 'strong' tier.",
)
parser.add_argument(
    "--curriculum_fast_retention_window",
    type=int,
    default=4,
    help="adaptive_fast: number of trailing PBT samples used for the "
    "retention_proxy denominator (default 4 ~= 80M env frames).",
)
parser.add_argument(
    "--curriculum_fast_shrink_strong",
    type=float,
    default=0.90,
    help="adaptive_fast: shrink factor used when (nsucc_mean >= strong_mean "
    "AND retention >= retention_threshold).",
)
parser.add_argument(
    "--curriculum_fast_shrink_mid",
    type=float,
    default=0.925,
    help="adaptive_fast: shrink factor used when nsucc_mean >= mid_mean.",
)
parser.add_argument(
    "--curriculum_fast_shrink_low",
    type=float,
    default=0.95,
    help="adaptive_fast: shrink factor used when nsucc_mean >= low_mean. "
    "Same as legacy default.",
)
parser.add_argument(
    "--curriculum_fast_consolidation_frames",
    type=int,
    default=50_000_000,
    help="adaptive_fast: post-shrink consolidation length (env frames) "
    "applied for strong/mid/low shrinks. Defaults to 50M for the fast "
    "schedule.",
)
parser.add_argument(
    "--curriculum_fast_recovery_consolidation_frames",
    type=int,
    default=100_000_000,
    help="adaptive_fast: longer consolidation (env frames) used when the "
    "policy is in recovery stage (post-restart freeze window or rollback "
    "cooldown). Defaults to 100M to be conservative.",
)
parser.add_argument(
    "--curriculum_fast_target_tolerance_800m",
    type=float,
    default=0.04,
    help="adaptive_fast: report-only target tolerance at 800M env frames; "
    "logged but does NOT change the runtime floor (which stays at "
    "target_success_tolerance, typically 0.01).",
)
# Phase A4 P3 (2026-06-09): source-safe curriculum inheritance + promising
# anti-churn for fastcurr_inherit 800M. Defaults are inert ("reset" /
# protection off) so legacy / fastcurr runs are bit-for-bit unchanged.
parser.add_argument(
    "--pbt_curriculum_inherit_mode",
    type=str,
    default="reset",
    choices=["reset", "source_safe"],
    help="On a PBT cross/recovery copy, how to inherit the source's "
    "curriculum tolerance into the target policy. 'reset' delegates to "
    "--pbt_recovery_tolerance_mode / --pbt_crosscopy_tolerance_mode "
    "(legacy). 'source_safe' inherits one level looser than the source "
    "(source_tol/0.95) when source is stable (nsucc_mean >= "
    "--pbt_source_safe_min_mean AND retention >= "
    "--pbt_source_safe_min_retention), clamped to the runtime floor; "
    "unstable / missing metadata falls back to legacy.",
)
parser.add_argument(
    "--pbt_source_safe_recovery_consolidation_frames",
    type=int,
    default=50_000_000,
    help="source_safe: recovery consolidation length (env frames) re-armed "
    "on the target slot when the source is stable. Stale post_shrink "
    "consolidation from the previous slot owner is cleared first.",
)
parser.add_argument(
    "--pbt_source_unsafe_recovery_consolidation_frames",
    type=int,
    default=100_000_000,
    help="source_safe: recovery consolidation length (env frames) re-armed "
    "on the target slot when the source is unstable or its curriculum "
    "metadata is missing.",
)
parser.add_argument(
    "--pbt_source_safe_min_mean",
    type=float,
    default=20.0,
    help="source_safe: source.nsucc_mean threshold above which the source "
    "is considered stable for inheritance purposes.",
)
parser.add_argument(
    "--pbt_source_safe_min_retention",
    type=float,
    default=0.7,
    help="source_safe: source rolling end_mean / peak_mean threshold "
    "above which the source is considered stable for inheritance.",
)
parser.add_argument(
    "--pbt_protect_promising_workers",
    action=argparse.BooleanOptionalAction,
    default=False,
    help="When true, workers that recently reached high nsucc are "
    "shielded from bottom-tier replacement for "
    "--pbt_promising_protect_frames (anti-churn). Hard collapse / NaN "
    "/ OOM still override the protection.",
)
parser.add_argument(
    "--pbt_promising_nsucc_mean",
    type=float,
    default=20.0,
    help="Promising-worker gate: protect if rolling end_mean >= this.",
)
parser.add_argument(
    "--pbt_promising_peak_mean",
    type=float,
    default=25.0,
    help="Promising-worker gate (with retention): protect if "
    "recent_peak_nsucc_mean >= this AND current retention >= "
    "--pbt_promising_min_retention.",
)
parser.add_argument(
    "--pbt_promising_min_retention",
    type=float,
    default=0.5,
    help="Retention guard for the peak-based promising promotion. Avoids "
    "protecting a worker that once peaked high but has since degraded.",
)
parser.add_argument(
    "--pbt_promising_protect_frames",
    type=int,
    default=150_000_000,
    help="Promising-worker protection window length (env frames).",
)
parser.add_argument(
    "--pbt_promising_allow_hard_collapse_override",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="When true (default), hard collapse / early collapse on the "
    "protected worker bypasses the protection so the supervisor can "
    "still replace it.",
)
# CSER-v0: staged PBT consolidation schedule (defaults inert).
parser.add_argument(
    "--pbt_consolidation_enable",
    action=argparse.BooleanOptionalAction,
    default=False,
    help="Enable staged PBT consolidation (exploration / consolidation / selection).",
)
parser.add_argument(
    "--pbt_consolidation_stage1_end_frames",
    type=int,
    default=2_200_000_000,
    help="End of exploration stage (env frames). Default 2.2B.",
)
parser.add_argument(
    "--pbt_consolidation_stage2_end_frames",
    type=int,
    default=3_500_000_000,
    help="End of consolidation stage (env frames). Default 3.5B.",
)
parser.add_argument(
    "--pbt_consolidation_stage3_end_frames",
    type=int,
    default=5_000_000_000,
    help="End of selection stage (env frames). Default 5.0B.",
)
parser.add_argument(
    "--pbt_consolidation_stage2_interval_mult",
    type=float,
    default=2.0,
    help="PBT tick interval multiplier during consolidation stage.",
)
parser.add_argument(
    "--pbt_consolidation_stage3_interval_mult",
    type=float,
    default=4.0,
    help="PBT tick interval multiplier during selection stage.",
)
parser.add_argument(
    "--pbt_consolidation_stage2_mutation_rate_mult",
    type=float,
    default=0.5,
    help="Hyperparam mutation rate multiplier during consolidation stage.",
)
parser.add_argument(
    "--pbt_consolidation_stage3_bottom_only",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Selection stage: restrict restart policy to bottom tier rules.",
)
parser.add_argument(
    "--pbt_consolidation_stage3_hard_collapse_only",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Selection stage: only replace on hard collapse or low nsucc.",
)
parser.add_argument(
    "--pbt_consolidation_stage2_retention_threshold",
    type=float,
    default=0.85,
    help="Consolidation stage: block bottom restart when nsucc retention "
    "vs slot peak exceeds this threshold.",
)
parser.add_argument(
    "--pbt_consolidation_collapse_nsucc_mean_threshold",
    type=float,
    default=1.0,
    help="Selection stage: allow bottom restart when nsucc_mean below this.",
)
# ----------------------------------------------------------------------
# Phase A4 P4 (2026-06-09): floor-recovery-aware PBT (Part A) +
# task-aware RSSM (Part B) + iterative teacher (Part C) + imagined
# evaluator (Part D) + WM-guided ranking (Part E) + replay
# prioritization (Part F). Defaults are inert; legacy runs unchanged.
# ----------------------------------------------------------------------
# Part A: Floor-recovery PBT.
parser.add_argument(
    "--pbt_floor_recovery_aware",
    action=argparse.BooleanOptionalAction,
    default=False,
    help="Enable floor-stage recovery-aware PBT ranking + protection "
    "+ archive. Default false keeps legacy behavior bit-for-bit.",
)
parser.add_argument(
    "--pbt_floor_tol_threshold", type=float, default=0.02,
    help="success_tolerance below this counts as floor stage.",
)
parser.add_argument(
    "--pbt_hard_floor_tol_threshold", type=float, default=0.012,
    help="success_tolerance below this counts as hard floor.",
)
parser.add_argument(
    "--pbt_floor_recovery_window", type=int, default=4,
    help="Rolling window length for floor-recovery slope estimation.",
)
parser.add_argument(
    "--pbt_floor_recovery_slope_weight", type=float, default=0.5,
    help="Weight on positive recovery slope inside floor_recovery_score.",
)
parser.add_argument(
    "--pbt_floor_recovery_retention_weight", type=float, default=0.3,
    help="Weight on retention proxy inside floor_recovery_score.",
)
parser.add_argument(
    "--pbt_floor_recovery_nsucc_max_weight", type=float, default=0.2,
    help="Weight on normalised nsucc_max inside floor_recovery_score.",
)
parser.add_argument(
    "--pbt_floor_recovery_drop_penalty", type=float, default=0.5,
    help="Drop-ratio penalty inside floor_recovery_score.",
)
parser.add_argument(
    "--pbt_floor_recovery_protect_frames", type=int, default=200_000_000,
    help="Floor-recovery protection window length (env frames).",
)
parser.add_argument(
    "--pbt_floor_recovery_min_nsucc_protect", type=float, default=15.0,
    help="Min nsucc_mean (alone) to enter floor-recovery protection.",
)
parser.add_argument(
    "--pbt_floor_recovery_min_slope_protect", type=float, default=0.05,
    help="Min positive-slope to enter floor-recovery protection (with retention).",
)
parser.add_argument(
    "--pbt_floor_recovery_min_retention_for_protect", type=float, default=0.3,
    help="Min retention to enter floor-recovery protection via slope.",
)
parser.add_argument(
    "--pbt_enable_best_floor_recovery_archive",
    action=argparse.BooleanOptionalAction,
    default=False,
    help="Save / select best_floor_recovery checkpoints.",
)
# Part E: WM-guided PBT ranking three-mode.
parser.add_argument(
    "--pbt_wm_score_mode", type=str, default="off",
    choices=["off", "shadow", "active"],
    help="off: real-only; shadow: log would-change without replacing; "
    "active: combine real + WM in ranking. Default off.",
)
parser.add_argument(
    "--pbt_use_wm_recovery_score",
    action=argparse.BooleanOptionalAction,
    default=False,
    help="Legacy alias: True -> --pbt_wm_score_mode=active when mode=off.",
)
parser.add_argument(
    "--pbt_wm_recovery_weight", type=float, default=0.3,
    help="WM recovery probability weight in combined score.",
)
parser.add_argument(
    "--pbt_wm_success_weight", type=float, default=0.2,
    help="WM success probability weight in combined score.",
)
parser.add_argument(
    "--pbt_wm_drop_penalty", type=float, default=0.3,
    help="WM drop risk penalty weight in combined score.",
)
parser.add_argument(
    "--pbt_wm_uncertainty_penalty", type=float, default=0.2,
    help="WM uncertainty penalty weight in combined score.",
)
parser.add_argument(
    "--pbt_wm_max_uncertainty", type=float, default=0.5,
    help="WM eval ignored when uncertainty > this; falls back to real.",
)
# PBT-WM Strategy Zoo v1.0: optional profile copy on cross-copy restart.
parser.add_argument(
    "--pbt_wm_profile_copy_enable",
    action=argparse.BooleanOptionalAction,
    default=False,
    help="When enabled, target may inherit source wm_strategy_profile after "
    "copy_after_frames (subject to diversity guard + copy_prob).",
)
parser.add_argument(
    "--pbt_wm_profile_copy_after_frames", type=int, default=1_000_000_000,
    help="Earliest env frame at which profile copy may occur.",
)
parser.add_argument(
    "--pbt_wm_profile_copy_prob", type=float, default=0.3,
    help="Probability of inheriting source profile on eligible cross-copy.",
)
parser.add_argument(
    "--pbt_wm_profile_keep_diversity_min_profiles", type=int, default=3,
    help="Profile copy is blocked only when distinct profiles after copy "
    "would be below this count.",
)
parser.add_argument(
    "--pbt_wm_profile_copy_min_nsucc_gap", type=float, default=0.0,
    help="Minimum nsucc gap (source - target) required for profile copy.",
)
parser.add_argument(
    "--pbt_wm_profile_copy_min_source_retention", type=float, default=0.0,
    help="Minimum source retention required for profile copy.",
)
parser.add_argument(
    "--pbt_wm_profile_copy_cooldown_frames", type=int, default=0,
    help="Minimum frames between profile copies for a target policy.",
)
parser.add_argument(
    "--pbt_wm_profile_copy_bottom_k", type=int, default=0,
    help="Target must be in bottom-k by ranking (or retention collapse).",
)
parser.add_argument(
    "--pbt_wm_profile_copy_top_k", type=int, default=0,
    help="Source must be in top-k by ranking (unless archive source).",
)
# Deep-diag fix (2026-06-20): wire bottom-slot health gates that were
# previously dead code (ProfileCopyConfig fields with no CLI binding).
parser.add_argument(
    "--pbt_wm_profile_copy_source_min_nsucc_mean", type=float, default=0.0,
    help=(
        "Profile copy is rejected when source nsucc_mean < this floor. "
        "Set to e.g. 15.0 to block early-window copies from sources that "
        "haven't yet mastered the basic task (root cause of the v17.3 5B "
        "p01 churn at 400M)."
    ),
)
parser.add_argument(
    "--pbt_wm_profile_copy_source_min_frames", type=int, default=0,
    help=(
        "Profile copy is rejected when source has trained < this many "
        "env frames. Set to e.g. 200_000_000 so we never copy a profile "
        "from a worker that hasn't run long enough to be evaluable."
    ),
)
parser.add_argument(
    "--pbt_wm_profile_copy_target_min_nsucc_mean", type=float, default=0.0,
    help=(
        "Profile copy is rejected when the target slot's nsucc_mean is "
        "below this floor (i.e. the slot is broken). Weight copy still "
        "proceeds; only the profile / zoo role is preserved. Set to e.g. "
        "5.0 to break the bottom-slot death spiral."
    ),
)
parser.add_argument(
    "--pbt_wm_profile_copy_target_min_nsucc_max", type=float, default=0.0,
    help=(
        "Companion to --pbt_wm_profile_copy_target_min_nsucc_mean: when "
        "the target's nsucc_mean is too low but its nsucc_max has crossed "
        "this floor (i.e. it has shown bursts of success), allow profile "
        "swap. Set to 0.0 to disable this escape hatch."
    ),
)
parser.add_argument(
    "--pbt_wm_profile_copy_locked_target_slots", type=str, default="",
    help=(
        "Comma-separated list of policy_idx values that must keep their "
        "launcher-assigned profile across PBT cross-restarts (e.g. "
        "'0,1' to lock p00=ppo_only anchor and p01=AWR rescue). Weight "
        "copy still proceeds for these slots."
    ),
)
# WM curriculum controller (Strategy Zoo profile: wm_curriculum_controller).
parser.add_argument(
    "--wm_curriculum_controller_enable",
    action=argparse.BooleanOptionalAction,
    default=False,
    help="Use RSSM imagined risk to block/delay curriculum shrink.",
)
parser.add_argument(
    "--wm_curriculum_controller_start_after_frames", type=int, default=500_000_000,
)
parser.add_argument(
    "--wm_curriculum_controller_min_nsucc_mean", type=float, default=10.0,
)
parser.add_argument(
    "--wm_curriculum_controller_drop_risk_threshold", type=float, default=0.45,
)
parser.add_argument(
    "--wm_curriculum_controller_uncertainty_threshold", type=float, default=0.55,
)
parser.add_argument(
    "--wm_curriculum_controller_min_success_prob", type=float, default=0.20,
)
parser.add_argument(
    "--wm_curriculum_controller_freeze_frames", type=int, default=100_000_000,
)
parser.add_argument(
    "--wm_curriculum_controller_hard_block_shrink",
    action=argparse.BooleanOptionalAction,
    default=True,
)
parser.add_argument(
    "--wm_curriculum_controller_use_recovery_prob",
    action=argparse.BooleanOptionalAction,
    default=False,
)
# Part B: task-aware RSSM extensions.
parser.add_argument(
    "--seq_wm_use_task_aux_inputs",
    action=argparse.BooleanOptionalAction,
    default=False,
    help="Add aux scalar encoder to RSSM posterior input.",
)
parser.add_argument(
    "--seq_wm_aux_embed_dim", type=int, default=64,
    help="Aux scalar encoder output dim.",
)
parser.add_argument(
    "--seq_wm_loss_preset", type=str, default="default",
    choices=["default", "task_focused"],
    help="default keeps legacy lambdas; task_focused biases task losses.",
)
parser.add_argument(
    "--seq_wm_floor_aware_weighting",
    action=argparse.BooleanOptionalAction,
    default=False,
    help="Multiply task-side losses by floor / hard-floor multipliers.",
)
parser.add_argument(
    "--seq_wm_floor_tol_threshold", type=float, default=0.02,
    help="seq_wm-side floor stage threshold (mirrors PBT one).",
)
parser.add_argument(
    "--seq_wm_floor_tol_floor", type=float, default=0.012,
    help="seq_wm-side hard-floor threshold (mirrors PBT one).",
)
parser.add_argument(
    "--seq_wm_floor_weight_mid", type=float, default=2.0,
    help="task-loss multiplier in floor stage.",
)
parser.add_argument(
    "--seq_wm_floor_weight_floor", type=float, default=3.0,
    help="task-loss multiplier in hard-floor stage.",
)
parser.add_argument(
    "--seq_wm_use_horizon_heads",
    action=argparse.BooleanOptionalAction,
    default=False,
    help="Add horizon (success/drop) heads to RSSM PredictionHeads.",
)
parser.add_argument(
    "--seq_wm_horizon_steps", type=str, default="4,8",
    help="Comma-separated horizon lengths for the horizon heads. "
    "Default 4,8 because seq_len=16 makes h>=15 degenerate "
    "(target undefined inside the window). Set to 8,16 only if "
    "seq_wm_seq_len is also raised.",
)
parser.add_argument(
    "--seq_wm_horizon_skip_too_long",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Skip any horizon h > seq_len-2 (target/mask degenerate).",
)
parser.add_argument(
    "--seq_wm_recovery_shadow_tol_threshold", type=float, default=0.04,
    help="Widen recovery target/mask up to this tolerance so that "
    "recovery_pos_rate is non-zero before the curriculum reaches "
    "floor (≤floor_tol_threshold). The loss is still emphasised "
    "on floor via floor-aware weighting; this only widens which "
    "steps are eligible for target/mask. Default 0.04.",
)
parser.add_argument(
    "--seq_wm_teacher_eval_buffer_capacity", type=int, default=8,
    help="FloorRecoveryEvalBuffer capacity (held-out batches).",
)
parser.add_argument(
    "--seq_wm_teacher_record_below_tol", type=float, default=0.04,
    help="Record into the eval buffer when ANY step's "
    "success_tolerance is at or below this value. Default 0.04 "
    "(matches the shadow target threshold).",
)
parser.add_argument(
    "--seq_wm_teacher_require_recovery_pos_low", type=float, default=0.05,
    help="Lower bound on recovery_pos_rate for promotion (gate).",
)
parser.add_argument(
    "--seq_wm_teacher_require_recovery_pos_high", type=float, default=0.95,
    help="Upper bound on recovery_pos_rate for promotion (gate).",
)
parser.add_argument(
    "--seq_wm_use_recovery_head",
    action=argparse.BooleanOptionalAction,
    default=False,
    help="Add soft-OR recovery head to RSSM PredictionHeads.",
)
parser.add_argument(
    "--seq_wm_recovery_horizon", type=int, default=16,
    help="Recovery head horizon length.",
)
parser.add_argument(
    "--seq_wm_lambda_horizon_success", type=float, default=0.0,
    help="Loss weight on horizon-success heads (default 0 -> disabled).",
)
parser.add_argument(
    "--seq_wm_lambda_drop_risk", type=float, default=0.0,
    help="Loss weight on drop-risk heads (default 0 -> disabled).",
)
parser.add_argument(
    "--seq_wm_lambda_recovery", type=float, default=0.0,
    help="Loss weight on recovery head (default 0 -> disabled).",
)
parser.add_argument(
    "--seq_wm_head_balance_enable",
    action=argparse.BooleanOptionalAction,
    default=False,
    help="Apply pos_weight balancing on horizon/recovery head BCE losses.",
)
parser.add_argument(
    "--seq_wm_success_pos_weight", type=float, default=1.0,
    help="Manual pos_weight for horizon-success head when auto is off.",
)
parser.add_argument(
    "--seq_wm_drop_pos_weight", type=float, default=1.0,
    help="Manual pos_weight for drop-risk head when auto is off.",
)
parser.add_argument(
    "--seq_wm_recovery_pos_weight", type=float, default=1.0,
    help="Manual pos_weight for recovery head when auto is off.",
)
parser.add_argument(
    "--seq_wm_head_pos_weight_auto",
    action=argparse.BooleanOptionalAction,
    default=False,
    help="Auto-compute pos_weight from batch label pos_frac (clamped).",
)
parser.add_argument(
    "--seq_wm_head_pos_weight_max", type=float, default=10.0,
    help="Upper clamp for auto pos_weight.",
)
parser.add_argument(
    "--seq_wm_recovery_d_hat_margin", type=float, default=0.005,
    help="d_hat improvement margin used in soft-OR recovery target.",
)
parser.add_argument(
    "--seq_wm_recovery_d_hat_tol_min_steps", type=int, default=0,
    help="Min steps with d_hat<=tolerance to mark soft-OR recovery.",
)
# Part C: iterative teacher.
parser.add_argument(
    "--seq_wm_enable_iterative_teacher",
    action=argparse.BooleanOptionalAction,
    default=False,
    help="Snapshot RSSM and run held-out floor evaluation with promotion gate.",
)
parser.add_argument(
    "--seq_wm_teacher_eval_interval_frames", type=int, default=50_000_000,
    help="env-frame interval between teacher snapshot evaluations.",
)
parser.add_argument(
    "--seq_wm_teacher_promote_min_improvement", type=float, default=0.02,
    help="min eval-score improvement required to promote a candidate.",
)
parser.add_argument(
    "--seq_wm_teacher_keep_last_k", type=int, default=6,
    help="number of teacher snapshots to keep on disk. Phase A4 P5 "
    "(2026-06-10): default raised 3->6 so rejected candidates are "
    "retained for offline forensics.",
)
parser.add_argument(
    "--seq_wm_teacher_metric", type=str, default="floor_recovery_eval_score",
    help="Metric driving teacher promotion. Currently only floor_recovery_eval_score.",
)
# Phase A4 P5 (2026-06-10): teacher-gate fix knobs.
parser.add_argument(
    "--seq_wm_teacher_allow_first_promotion",
    action=argparse.BooleanOptionalAction, default=True,
    help="When True, the first valid candidate is promoted even if no "
    "previous teacher exists, provided gate (n_samples, uncertainty, "
    "non-degenerate horizon-success target) is satisfied. Default True.",
)
parser.add_argument(
    "--seq_wm_teacher_min_eval_size", type=int, default=512,
    help="Minimum eval buffer sample count (sum of batch starts) required "
    "for a promotion attempt. Below this the attempt is rejected with "
    "reason=no_eval_data and persisted to teacher_eval_events.jsonl.",
)
parser.add_argument(
    "--seq_wm_teacher_max_uncertainty", type=float, default=0.5,
    help="Reject candidates whose imagined-eval uncertainty exceeds this. "
    "Phase A4 P5: dedicated knob; the legacy `--pbt_wm_max_uncertainty` "
    "is reused only as PBT-side gate.",
)
parser.add_argument(
    "--seq_wm_teacher_pos_rate_low", type=float, default=0.001,
    help="Lower edge of the non-degenerate pos-rate range. Pos rates "
    "<= low or >= high are saturated. Default 0.001 ensures only truly "
    "degenerate targets are rejected.",
)
parser.add_argument(
    "--seq_wm_teacher_pos_rate_high", type=float, default=0.999,
    help="Upper edge of the non-degenerate pos-rate range. See "
    "`--seq_wm_teacher_pos_rate_low`.",
)
# Part D: imagined evaluator.
parser.add_argument(
    "--seq_wm_enable_imagined_evaluator",
    action=argparse.BooleanOptionalAction,
    default=False,
    help="Run short-horizon imagined evaluation each PBT tick.",
)
parser.add_argument(
    "--seq_wm_imagine_horizons", type=str, default="8,16",
    help="Comma-separated horizon lengths for imagined evaluation.",
)
parser.add_argument(
    "--seq_wm_imagine_action_mode", type=str, default="recorded",
    choices=["recorded", "policy"],
    help="recorded uses stored actions; policy raises NotImplementedError.",
)
parser.add_argument(
    "--seq_wm_imagine_num_starts", type=int, default=256,
    help="Number of start states sampled for imagined evaluation.",
)
parser.add_argument(
    "--seq_wm_imagine_use_teacher",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Use the promoted teacher for imagined evaluation when available.",
)
# Part F: replay prioritization.
parser.add_argument(
    "--seq_wm_replay_prioritize_floor_recovery",
    action=argparse.BooleanOptionalAction,
    default=False,
    help="Bias the SeqWM sampler toward floor / recovery / drop-near-success windows.",
)
parser.add_argument(
    "--seq_wm_replay_floor_weight", type=float, default=2.0,
    help="Floor-stage replay weight.",
)
parser.add_argument(
    "--seq_wm_replay_hard_floor_weight", type=float, default=3.0,
    help="Hard-floor stage replay weight.",
)
parser.add_argument(
    "--seq_wm_replay_positive_slope_weight", type=float, default=2.0,
    help="Positive-slope replay weight.",
)
parser.add_argument(
    "--seq_wm_replay_drop_near_success_weight", type=float, default=1.5,
    help="Drop-near-success replay weight.",
)
parser.add_argument(
    "--seq_wm_replay_uniform_floor", type=float, default=0.3,
    help="Min fraction of batch reserved for uniform sampling.",
)
# Convenience preset (Item 10). Does NOT bypass kill switches; train.py
# expands it into individual flags BEFORE constructing the observer and
# logs the expanded flag set.
parser.add_argument(
    "--experimental_rssm_floor_recovery_stack",
    action=argparse.BooleanOptionalAction,
    default=False,
    help="Convenience preset that turns on Part A+B+C+D+E+F. Each "
    "feature has its own --no-* kill switch which always wins.",
)
# Patch P2-B (2026-06-01): early collapse freeze + tolerance rollback.
parser.add_argument(
    "--pbt_early_collapse_detect",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Enable early collapse freeze (default 30%% drop, 1 window).",
)
parser.add_argument(
    "--early_collapse_drop_ratio",
    type=float,
    default=0.3,
    help="Early collapse: nsucc_mean drop vs recent peak threshold.",
)
parser.add_argument(
    "--early_collapse_patience",
    type=int,
    default=1,
    help="Early collapse: consecutive bad windows before freeze.",
)
parser.add_argument(
    "--curriculum_rollback_cooldown_frames",
    type=int,
    default=200_000_000,
    help="Cooldown between hard-collapse tolerance rollbacks unless a new shrink happened.",
)
# Hybrid Curriculum Safety Patch (2026-06-15): real-signal fallback + gates.
parser.add_argument(
    "--curriculum_safety_enable",
    action=argparse.BooleanOptionalAction,
    default=False,
    help="Enable hybrid curriculum safety controller (real-signal fallback).",
)
parser.add_argument(
    "--curriculum_safety_start_after_frames",
    type=int,
    default=500_000_000,
    help="Env frames before curriculum safety controller activates.",
)
parser.add_argument(
    "--curriculum_safety_min_nsucc_mean",
    type=float,
    default=10.0,
    help="Min nsucc_mean before real-signal safety may block shrink.",
)
parser.add_argument(
    "--curriculum_safety_global_retention_threshold",
    type=float,
    default=0.80,
    help="Real-signal risk when global_retention drops below this.",
)
parser.add_argument(
    "--curriculum_safety_drop_ratio_threshold",
    type=float,
    default=0.30,
    help="Real-signal risk when recent drop_ratio exceeds this.",
)
parser.add_argument(
    "--curriculum_safety_slope_threshold",
    type=float,
    default=-2.0,
    help="Real-signal risk when recent nsucc slope falls below this.",
)
parser.add_argument(
    "--curriculum_safety_window",
    type=int,
    default=6,
    help="Window length for slope / drop-ratio safety signals.",
)
parser.add_argument(
    "--curriculum_safety_block_frames",
    type=int,
    default=100_000_000,
    help="Post-block shrink freeze (env frames) after real-signal safety block.",
)
parser.add_argument(
    "--curriculum_safety_use_real_fallback",
    action=argparse.BooleanOptionalAction,
    default=False,
    help="Use real PBT signals when WM imagined eval is invalid.",
)
parser.add_argument(
    "--curriculum_strong_shrink_global_retention_gate",
    action=argparse.BooleanOptionalAction,
    default=False,
    help="Gate adaptive_fast strong shrink on global retention + stability.",
)
parser.add_argument(
    "--curriculum_strong_shrink_min_global_retention",
    type=float,
    default=0.85,
    help="Min global retention to allow adaptive_fast strong shrink.",
)
parser.add_argument(
    "--curriculum_strong_shrink_min_stable_ticks",
    type=int,
    default=3,
    help="Min stable tolerance ticks to allow adaptive_fast strong shrink.",
)
parser.add_argument(
    "--curriculum_strong_shrink_long_window",
    type=int,
    default=12,
    help="Long rolling window for global retention peak denominator.",
)
parser.add_argument(
    "--curriculum_danger_zone_enable",
    action=argparse.BooleanOptionalAction,
    default=False,
    help="Enable tol<=danger_tol conservative shrink rules.",
)
parser.add_argument(
    "--curriculum_danger_tol",
    type=float,
    default=0.030,
    help="Success tolerance threshold defining the danger zone.",
)
parser.add_argument(
    "--curriculum_danger_disable_strong_shrink",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Disable adaptive_fast strong shrink inside danger zone.",
)
parser.add_argument(
    "--curriculum_danger_min_stable_ticks",
    type=int,
    default=4,
    help="Min stable ticks required before shrink in danger zone.",
)
parser.add_argument(
    "--curriculum_danger_shrink_interval_mult",
    type=float,
    default=2.0,
    help="Multiply consolidation frames in danger zone.",
)
parser.add_argument(
    "--curriculum_danger_global_retention_threshold",
    type=float,
    default=0.85,
    help="Block shrink in danger zone when global retention below this.",
)
parser.add_argument(
    "--curriculum_rollback_on_repeated_collapse",
    action=argparse.BooleanOptionalAction,
    default=False,
    help="Rollback to last-mastered-safe tolerance on repeated collapse.",
)
parser.add_argument(
    "--curriculum_rollback_collapse_patience",
    type=int,
    default=2,
    help="Repeated collapse count before rollback-to-last-mastered.",
)
parser.add_argument(
    "--curriculum_rollback_to_last_mastered",
    action=argparse.BooleanOptionalAction,
    default=False,
    help="Apply forced_success_tolerance rollback to last mastered safe tol.",
)
parser.add_argument(
    "--curriculum_rollback_freeze_frames",
    type=int,
    default=200_000_000,
    help="Shrink freeze after repeated-collapse rollback.",
)
parser.add_argument(
    "--curriculum_last_mastered_min_nsucc_mean",
    type=float,
    default=30.0,
    help="Min nsucc_mean to update last-mastered-safe tolerance.",
)
parser.add_argument(
    "--curriculum_last_mastered_min_global_retention",
    type=float,
    default=0.85,
    help="Min global retention to update last-mastered-safe tolerance.",
)
parser.add_argument(
    "--curriculum_last_mastered_stable_ticks",
    type=int,
    default=3,
    help="Min stable ticks to update last-mastered-safe tolerance.",
)
parser.add_argument(
    "--wm_imagined_diagnostic_enable",
    action=argparse.BooleanOptionalAction,
    default=False,
    help="Emit WM imagined head diagnostic fields in training_curve.",
)
parser.add_argument(
    "--wm_imagined_shadow_block_enable",
    action=argparse.BooleanOptionalAction,
    default=False,
    help="Record shadow WM block decisions (record-only unless real risk).",
)
parser.add_argument(
    "--wm_imagined_shadow_uncertainty_threshold",
    type=float,
    default=0.95,
    help="Uncertainty threshold for shadow block decision recording.",
)
parser.add_argument(
    "--wm_imagined_valid_mode",
    type=str,
    default="strict_uncertainty",
    choices=["strict_uncertainty", "soft_calibrated", "shadow_only"],
    help="Imagined-valid gating mode for WM PBT ranking / diagnostics.",
)
parser.add_argument(
    "--wm_imagined_valid_uncertainty_threshold",
    type=float,
    default=0.5,
    help="Strict imagined-valid uncertainty threshold.",
)
parser.add_argument(
    "--wm_imagined_soft_uncertainty_threshold",
    type=float,
    default=0.95,
    help="Soft imagined-valid uncertainty threshold (requires head signal).",
)
parser.add_argument(
    "--wm_imagined_min_logit_abs_mean",
    type=float,
    default=0.05,
    help="Min mean |logit| to count as head signal for soft validity.",
)
parser.add_argument(
    "--wm_imagined_min_head_signal_std",
    type=float,
    default=0.02,
    help="Min head prob std to count as head signal for soft validity.",
)
parser.add_argument(
    "--wm_imagined_soft_validity_weight",
    type=float,
    default=0.25,
    help="WM ranking weight multiplier when soft-valid (not strict-valid).",
)
parser.add_argument(
    "--pbt_usable_live_min_mean",
    type=float,
    default=1.5,
    help="usable_live gate: latest nsucc_mean must be >= this.",
)
parser.add_argument(
    "--pbt_usable_live_min_max",
    type=float,
    default=20.0,
    help="usable_live gate: latest nsucc_max must be >= this.",
)
parser.add_argument(
    "--pbt_usable_live_max_drop_ratio",
    type=float,
    default=0.5,
    help="usable_live gate: drop ratio vs prior window must be <= this.",
)
# Patch P2-C (2026-06-01): best_stable recovery execution audit.
parser.add_argument(
    "--pbt_recovery_bypass_elimination_burnin",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Allow archive / best_stable recovery to bypass elimination "
    "burn-in anti-churn.",
)
# Patch P2-D (2026-06-01): recovery tolerance mode.
parser.add_argument(
    "--pbt_recovery_tolerance_mode",
    type=str,
    default="last_mastered",
    choices=["checkpoint", "last_mastered", "relax_one_level"],
    help="Tolerance applied on archive / best_stable recovery restart.",
)
parser.add_argument(
    "--pbt_crosscopy_tolerance_mode",
    type=str,
    default="checkpoint",
    choices=["checkpoint", "last_mastered", "relax_one_level"],
    help="Tolerance mode for live cross-copy restarts (strict_stable/usable/relaxed). "
    "Default keeps source's checkpoint tolerance; last_mastered avoids "
    "pushing recovered policy back to 0.075 after OOM-driven cross-copy.",
)
parser.add_argument(
    "--pbt_recovery_source_cooldown_frames",
    type=int,
    default=50_000_000,
    help="Post-recovery env frames before a recovered worker is eligible "
    "as a PBT source.",
)
# Patch P2-E (2026-06-01): freeze hyperparam mutation after stable window.
parser.add_argument(
    "--pbt_freeze_mutation_after_stable_frames",
    type=int,
    default=100_000_000,
    help="Env frames to skip PPO hyperparam mutation after stable source "
    "window (PPO training continues).",
)
# These are set ONLY by pbt_observer when relaunching a worker after a
# PBT replacement. They must not be in the user-facing launch command;
# their job is to forward in-process anti-churn state across exec().
parser.add_argument("--restart_count_resume", type=int, default=None)
parser.add_argument("--frames_at_restart_resume", type=int, default=None)
parser.add_argument("--last_mutation_frame_resume", type=int, default=None)
# Patch F (2026-05-31): self/cross-restart accounting resume hooks.
parser.add_argument("--self_restart_count_resume", type=int, default=None)
parser.add_argument("--last_self_restart_frame_resume", type=int, default=None)
parser.add_argument("--cross_restart_count_resume", type=int, default=None)
# Patch G (2026-05-31): archive-recovery accounting resume hooks.
parser.add_argument("--recovery_restart_count_resume", type=int, default=None)
parser.add_argument("--last_recovery_frame_resume", type=int, default=None)
parser.add_argument("--last_profile_copy_frame_resume", type=int, default=None)
# PBT-WM Strategy Zoo v1.0: profile lineage resume hooks (forwarded by observer).
parser.add_argument("--wm_strategy_profile_generation_resume", type=int, default=None)
parser.add_argument("--wm_strategy_profile_parent_resume", type=int, default=None)
parser.add_argument("--wm_strategy_profile_copied_from_source_resume", type=int, default=None)
parser.add_argument("--gamma", type=float, default=None)
parser.add_argument("--kl_threshold", type=float, default=None)
parser.add_argument("--grad_norm", type=float, default=None)
parser.add_argument("--critic_coef", type=float, default=None)
parser.add_argument("--bounds_loss_coef", type=float, default=None)
parser.add_argument("--e_clip", type=float, default=None)
parser.add_argument("--mini_epochs", type=int, default=None)
# M8: learning_rate is part of mutable_params, but was previously dropped because
# the parser did not accept it. Restoring it here keeps PBT learning-rate
# mutations effective end-to-end (manifest -> argparse -> agent_cfg).
parser.add_argument("--learning_rate", type=float, default=None)
# M3: PBT restart resume hooks for env-side curriculum state. These are NOT in
# argparse default flow; they are only emitted by pbt_observer when a worker is
# replaced. Train.py overwrites env attributes AFTER gym.make() and BEFORE
# runner.run() so the new policy starts at the source worker's tolerance.
parser.add_argument("--success_tolerance_resume", type=float, default=None)
parser.add_argument("--initial_tolerance_resume", type=float, default=None)
parser.add_argument("--target_tolerance_resume", type=float, default=None)
parser.add_argument("--last_curriculum_update_resume", type=int, default=None)
parser.add_argument("--frame_since_restart_resume", type=int, default=None)
# Patch H (2026-05-31): post-restart curriculum freeze resume hook.
parser.add_argument(
    "--curriculum_freeze_steps_after_restart_resume",
    type=int,
    default=None,
    help="Simulation-step cutoff under which the curriculum is held at its "
    "inherited tolerance. Preferred over the env-frame alias below.",
)
parser.add_argument(
    "--curriculum_freeze_frames_after_restart_resume",
    type=int,
    default=None,
    help="Legacy env-frame alias; converted to simulation steps via num_envs.",
)
parser.add_argument("--restart_source_policy", type=int, default=None)
parser.add_argument("--alpha_reach", type=float, default=None)
parser.add_argument("--alpha_pick", type=float, default=None)
parser.add_argument("--r_picked", type=float, default=None)
parser.add_argument("--alpha_targ", type=float, default=None)
parser.add_argument("--r_success", type=float, default=None)
parser.add_argument("--kuka_actions_penalty_scale", type=float, default=None)
parser.add_argument("--allegro_actions_penalty_scale", type=float, default=None)
parser.add_argument("--sigma", type=str, default=None)
parser.add_argument("--test", action="store_true", default=False, help="Run in play / evaluation mode.")
# ---- World Model (WM) args ----
parser.add_argument(
    "--wm_strategy_profile",
    type=str,
    default="legacy",
    choices=[
        "legacy",
        "ppo_only",
        "rssm_mppi_actor_distill",
        "wm_curriculum_controller",
        "wm_pbt_source_selector",
        "elite_bank_actor_distill",
        "conservative_rssm_planner_distill",
        "light_rssm_mppi_actor_distill",
        "wm_pbt_source_selector_v2",
        "light_conservative_rssm_planner_distill",
        "elite_bank_actor_distill_fast",
        "elite_bank_actor_distill_strong",
        "elite_bank_actor_distill_fast_awr",
        "wm_pbt_source_selector_awr",
        "elite_bank_actor_distill_awr",
        "elite_bank_actor_distill_strong_awr",
        "unified_elite_wm_pbt",
    ],
    help="PBT-WM Strategy Zoo profile (expands to a full WM usage config). "
    "legacy = no expansion (existing launcher behaviour).",
)
parser.add_argument(
    "--wm_strategy_profile_allow_cli_override",
    action="store_true",
    default=False,
    help="Debug only: allow explicit CLI flags to override non-legacy profile "
    "table values (default: profile is authoritative).",
)
parser.add_argument("--enable_wm", action="store_true", default=False,
                    help="Enable world model training.")
parser.add_argument("--wm_strategy", type=str, default="none",
                    choices=[
                        "none", "aux_low", "aux_mid",
                        "value_imag_h1", "value_imag_h3",
                        # Phase A4: sequence WM family.
                        "rssm_aux", "rssm_belief_value",
                        "trans_aux", "trans_belief_value",
                    ],
                    help="WM strategy for this policy.")
parser.add_argument("--wm_checkpoint_path", type=str, default=None,
                    help="Optional source WM checkpoint path for warm start.")
parser.add_argument("--wm_checkpoint_strategy", type=str, default="unknown",
                    help="Optional source strategy hint for WM checkpoint compatibility checks.")
parser.add_argument("--wm_load_mode", type=str, default="auto", choices=["auto", "force", "reset"],
                    help="WM load mode: auto (compat check), force (load if possible), reset (never load).")
parser.add_argument("--wm_ensemble_size", type=int, default=3,
                    help="Number of ensemble members (default 3).")
parser.add_argument("--wm_lr", type=float, default=1e-3,
                    help="WM learning rate (default 1e-3).")
parser.add_argument("--wm_update_every_n_epochs", type=int, default=2,
                    help="WM update frequency: every N PPO epochs (default 2).")
parser.add_argument("--wm_batch_size", type=int, default=512,
                    help="WM batch size (default 512).")
parser.add_argument("--enable_counterfactual", action="store_true", default=False,
                    help="Enable counterfactual block triggers.")
parser.add_argument("--cf_interval_gens", type=int, default=5,
                    help="CF trigger every N PBT generations (default 5).")
parser.add_argument("--cf_steps", type=int, default=5_000_000,
                    help="Steps per CF child (default 5M).")
parser.add_argument("--cf_num_envs", type=int, default=4096,
                    help="Number of envs for CF children (default 4096).")
parser.add_argument("--cf_max_parallel_children", type=int, default=2,
                    help="Max parallel CF children (default 2).")
parser.add_argument("--cf_cuda_visible_devices", type=str, default="4,5",
                    help="CUDA_VISIBLE_DEVICES for CF children, comma separated.")
parser.add_argument("--cf_horizon", type=int, default=16,
                    help="Horizon used to map cf_steps to CF child iterations.")
parser.add_argument("--require_counterfactual_runner", action="store_true", default=False,
                    help="Fail fast when enable_counterfactual=True but runner heartbeat is missing/stale.")
parser.add_argument("--status_output_path", type=str, default=None,
                    help="Optional status output file (used by CF runner).")
# Patch 3 (rev 2, 2026-05-30): CF child training / eval configuration.
# These flags are only meaningful when this train.py invocation is a CF
# child spawned by run_counterfactual_block.py (i.e. status_output_path set).
# In a normal PBT worker they have no effect.
parser.add_argument(
    "--cf_train_frames",
    type=int,
    default=20_000_000,
    help="Approximate env-frames a CF child should train. Used by "
    "run_counterfactual_block to compute max_iterations.",
)
parser.add_argument(
    "--cf_reset_epoch_counter",
    action="store_true",
    default=False,
    help="If set, reset agent.frame/epoch_num to 0 after loading the "
    "source checkpoint so the child does not exit with MAX EPOCHS NUM.",
)
parser.add_argument(
    "--cf_min_eval_s",
    type=float,
    default=30.0,
    help="Minimum wall-time (seconds) a CF child must train; if elapsed_s "
    "is below this, the CF block runner marks this strategy invalid.",
)
parser.add_argument(
    "--cf_delta_mode",
    type=str,
    choices=["learning_improvement", "raw_after"],
    default="learning_improvement",
    help="How CF block runner computes delta_w for posterior updates.",
)
parser.add_argument(
    "--cf_low_signal_action",
    type=str,
    choices=["skip", "downweight"],
    default="skip",
    help="Posterior update action when a CF child result is low_signal.",
)
parser.add_argument(
    "--cf_low_signal_weight",
    type=float,
    default=0.25,
    help="Posterior weight when cf_low_signal_action=downweight.",
)
parser.add_argument(
    "--cf_require_nsucc_gt_zero",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Skip CF trigger when source worker nsucc_mean <= 0.",
)
# Patch I (2026-05-31): tighter CF source health gates. Defaults match
# Patch 3 (only require nsucc>0); set to numeric thresholds to demand
# a meaningfully healthy source before firing CF compute.
parser.add_argument(
    "--cf_source_min_nsucc_mean",
    type=float,
    default=0.0,
    help="Health gate: CF source must have nsucc_mean >= this (default 0.0; "
    "set to e.g. 1.5 to require a clearly-non-zero baseline).",
)
parser.add_argument(
    "--cf_source_min_nsucc_max",
    type=float,
    default=0.0,
    help="Health gate: CF source must have nsucc_max >= this (default 0.0).",
)
parser.add_argument(
    "--cf_source_min_ranking_score",
    type=float,
    default=float("-inf"),
    help="Health gate: CF source must have pbt_ranking_score >= this "
    "(default -inf = disabled).",
)
parser.add_argument(
    "--cf_delta_metric",
    type=str,
    choices=["r_meta", "nsucc"],
    default="r_meta",
    help="Posterior delta metric: r_meta (legacy) or nsucc (Patch I). "
    "nsucc is more robust under curriculum tightening because it does "
    "not include the tolerance progress term.",
)
# Patch J (2026-05-31): fixed-strategy ablation mode. When `fixed`, the
# observer skips the posterior selector and CF manager; the worker keeps
# `--fixed_wm_strategy` for the whole run. Use this for ablation
# experiments such as the DexPBT baseline (`fixed_wm_strategy=none`) or
# per-strategy learning curves.
parser.add_argument(
    "--wm_strategy_mode",
    type=str,
    choices=["bayesian", "fixed"],
    default="bayesian",
    help="WM strategy selection mode. `bayesian` (default) uses the "
    "posterior selector + CF. `fixed` forces --fixed_wm_strategy on every "
    "restart and disables CF (ablation use).",
)
parser.add_argument(
    "--fixed_wm_strategy",
    type=str,
    default="value_imag_h3",
    help="WM strategy to keep when --wm_strategy_mode=fixed. Use "
    "`none` for a pure DexPBT baseline.",
)
# Track C (2026-06-06): explicit Bayesian strategy pool. Defaults to the
# canonical 5 strategies so existing launchers (fixed_h1, cbwm smoke)
# behave exactly as before. Track C `bayesian-no-h3` passes a narrowed
# pool such as `none,aux_mid,value_imag_h1` to permanently exclude
# `value_imag_h3` (and optionally `aux_low`) from the posterior selector
# and the CF runner.
parser.add_argument(
    "--wm_strategy_pool",
    type=str,
    default="none,aux_low,aux_mid,value_imag_h1,value_imag_h3",
    help="Comma-separated list of WM strategies the bayesian selector / "
    "CF runner are allowed to pick from. Must be a non-empty subset of "
    "{none,aux_low,aux_mid,value_imag_h1,value_imag_h3,"
    "rssm_aux,rssm_belief_value,trans_aux,trans_belief_value}. Default keeps "
    "the legacy 5-strategy set; Track C used `none,aux_mid,value_imag_h1`; "
    "Phase A4 sequence runs use e.g. "
    "`none,aux_mid,rssm_belief_value,trans_belief_value`.",
)
# --- Phase A4: stage-aware Bayesian selector + sequence WM family ---
parser.add_argument(
    "--wm_selector_context",
    type=str,
    choices=["global", "stage"],
    default="global",
    help="Posterior context for Thompson sampling. `global` (default; "
    "matches existing runs) uses one posterior over the entire training "
    "history. `stage` partitions the posterior by stage "
    "(recovery/early/acquisition/mastery) and falls back to global on a "
    "per-strategy basis when the stage entry has fewer than min_stage_n "
    "samples.",
)
parser.add_argument(
    "--wm_selector_min_stage_n",
    type=int, default=3,
    help="Per-strategy threshold below which sample_stage falls back to "
    "the global posterior for that strategy.",
)
parser.add_argument("--seq_wm_seq_len", type=int, default=16,
                    help="Sequence length T for the SeqWM sampler. Must be <= "
                    "PPO horizon_length (default 16). Drop to 8 to halve cost.")
parser.add_argument("--seq_wm_seq_batch_size", type=int, default=1024,
                    help="Per-update sequence batch size B.")
parser.add_argument("--seq_wm_updates_per_ppo", type=int, default=1,
                    help="Number of SeqWM gradient steps per PPO epoch.")
parser.add_argument("--seq_wm_lr", type=float, default=1e-3,
                    help="AdamW learning rate for SeqWM.")
parser.add_argument("--seq_wm_grad_clip", type=float, default=100.0,
                    help="Grad-norm clip for SeqWM optimizer step.")
parser.add_argument("--seq_wm_lambda_total", type=float, default=0.05,
                    help="Outer scale on the SeqWM total loss before backward.")
parser.add_argument("--seq_wm_lambda_state", type=float, default=1.0,
                    help="Per-term weight: state_delta MSE.")
parser.add_argument("--seq_wm_lambda_reward", type=float, default=1.0,
                    help="Per-term weight: reward MSE.")
parser.add_argument("--seq_wm_lambda_done", type=float, default=0.2,
                    help="Per-term weight: done BCE.")
parser.add_argument("--seq_wm_lambda_success", type=float, default=1.0,
                    help="Per-term weight: success BCE.")
parser.add_argument("--seq_wm_lambda_task_error", type=float, default=1.0,
                    help="Per-term weight: task_error_delta MSE.")
parser.add_argument("--seq_wm_lambda_value", type=float, default=0.1,
                    help="Per-term weight: V_seq(b_t) MSE for *_belief_value.")
parser.add_argument(
    "--seq_wm_value_detach_belief",
    action=argparse.BooleanOptionalAction,
    default=False,
    help="If true, the SeqWM value head is trained on b_t.detach(), "
    "decoupling its gradient from the SeqWM backbone. Default false.",
)
parser.add_argument("--seq_wm_beta_kl_start", type=float, default=0.05,
                    help="RSSM KL coefficient at frame 0 (warmup start).")
parser.add_argument("--seq_wm_beta_kl_end", type=float, default=0.20,
                    help="RSSM KL coefficient after warmup.")
parser.add_argument("--seq_wm_beta_kl_warmup_frames", type=int,
                    default=200_000_000,
                    help="Linear warmup duration for beta_kl.")
parser.add_argument("--seq_wm_free_nats", type=float, default=1.0,
                    help="RSSM free-nats floor on per-step KL.")
parser.add_argument("--seq_wm_max_seq_len", type=int, default=32,
                    help="Transformer-lite positional embedding capacity.")
# --- Phase A4 P0: PPO-facing distillation ---
parser.add_argument(
    "--seq_wm_policy_coupling",
    type=str,
    choices=[
        "none",
        "distill_value",
        "distill_value_success",
        "critic_distill_value_norm",
        "shared_aux_critic_distill_value_norm",
    ],
    default="none",
    help="How the sequence WM influences the PPO actor-critic. "
    "`none` (default; backwards-compatible -- reproduces pre-patch "
    "behaviour where the seq WM has zero gradient effect on PPO). "
    "Legacy negative ablations: `distill_value` and "
    "`distill_value_success` use the actor-direct path and require "
    "feature_source=actor (P0; the 200M sanity AB showed this hurts "
    "policy learning). v2: `critic_distill_value_norm` (alias "
    "`shared_aux_critic_distill_value_norm`) routes the distill loss "
    "through a private seq_aux_critic_mlp sub-tower; requires "
    "feature_source=shared_to_aux_critic and "
    "teacher_value_norm=rollout_zscore.",
)
parser.add_argument(
    "--seq_wm_policy_coupling_weight",
    type=float, default=0.05,
    help="Outer scalar on the seq distill aux loss (separate from "
    "--seq_wm_lambda_total which scales the seq WM's own training).",
)
parser.add_argument(
    "--seq_wm_policy_coupling_value_weight",
    type=float, default=1.0,
    help="Per-term weight: distill_value MSE.",
)
parser.add_argument(
    "--seq_wm_policy_coupling_success_weight",
    type=float, default=0.5,
    help="Per-term weight: distill_success BCE-with-logits.",
)
parser.add_argument(
    "--seq_wm_policy_coupling_warmup_frames",
    type=int, default=100_000_000,
    help="Frames of pure seq-WM-only training before the distill loss "
    "is enabled. During warmup the seq WM still trains on its own "
    "optimizer; only the PPO distill head and gradient flow are "
    "suppressed (get_aux_loss returns {}). Default 100M.",
)
parser.add_argument(
    "--seq_wm_distill_feature_source",
    type=str, choices=["actor", "shared_to_aux_critic"],
    default="actor",
    help="Feature path for the distill loss. `actor` is the P0 "
    "actor-direct path used by legacy `distill_value`. "
    "`shared_to_aux_critic` is the v2 path: shared actor_mlp_out "
    "is consumed exclusively by a private seq_aux_critic_mlp "
    "sub-tower (NOT the rl_games critic_mlp; that module is empty "
    "under network.separate=false). Required for "
    "mode=critic_distill_value_norm.",
)
parser.add_argument(
    "--seq_wm_teacher_value_norm",
    type=str, choices=["none", "rollout_zscore"], default="none",
    help="Per-rollout teacher value normalization. `none` keeps raw "
    "teacher targets. `rollout_zscore` standardizes the (T, N) "
    "teacher tensor to mean 0 std 1 over the entire rollout before "
    "flattening. Required for mode=critic_distill_value_norm.",
)
parser.add_argument(
    "--seq_wm_teacher_value_norm_min_std",
    type=float, default=1e-3,
    help="Below this rollout-level std, the gate blocks distill rather "
    "than producing exploding zscore targets.",
)
parser.add_argument(
    "--seq_wm_policy_coupling_gate",
    type=str, choices=["none", "teacher_corr"], default="none",
    help="Quality gate evaluated once per rollout. `none` always "
    "lets distill through. `teacher_corr` blocks the rollout when "
    "Pearson(teacher_value, returns) is below "
    "--seq_wm_teacher_corr_threshold or undefined.",
)
parser.add_argument(
    "--seq_wm_teacher_corr_threshold",
    type=float, default=0.2,
    help="Threshold for the teacher_corr gate. Default 0.2.",
)
parser.add_argument(
    "--seq_wm_policy_coupling_ramp_frames",
    type=int, default=0,
    help="After warmup, linearly ramp the effective weight from 0 to "
    "--seq_wm_policy_coupling_weight over this many frames. Default 0 "
    "(no ramp -- weight jumps to its target value immediately).",
)
parser.add_argument(
    "--seq_wm_aux_critic_subtower_hidden",
    type=int, default=256,
    help="Hidden width for the v2 seq_aux_critic_mlp sub-tower. "
    "Default 256 (matches actor_mlp_out width; sub-tower output "
    "stays at 256 so seq_aux_value_head requires no change).",
)

# ----------------------------------------------------------------------
# FloorCritic (Real-target) — task-progress auxiliary critic learning.
# Default-off: with --no-wm_floorcritic_enable (the default) NOTHING is
# constructed, hooked, or patched, so behaviour is bit-for-bit unchanged.
# See results/FLOORCRITIC_IMPLEMENTATION_PLAN.md.
# ----------------------------------------------------------------------
parser.add_argument(
    "--wm_floorcritic_enable",
    action=argparse.BooleanOptionalAction, default=False,
    help="Enable Real-target FloorCritic auxiliary critic heads (default off).",
)
parser.add_argument(
    "--wm_floorcritic_feature_source",
    type=str, default="actor_mlp_out_detached",
    choices=["actor_mlp_out_detached", "actor_mlp_out_shared", "critic_feature"],
    help="Aux tower input. detached=Phase0 (actor_grad_blocked); shared=Phase1 "
    "(regularises shared backbone, never connects mu head); critic_feature "
    "degrades to shared under rl_games separate=False.",
)
parser.add_argument(
    "--wm_floorcritic_max_grad_ratio", type=float, default=0.1,
    help="Upper bound on ||grad_FC|| / ||grad_PPO|| on the shared feature "
    "(shared mode only). Above this the FC loss is auto-scaled down.",
)
parser.add_argument(
    "--wm_floorcritic_auto_scale_by_grad_ratio",
    action=argparse.BooleanOptionalAction, default=True,
    help="Auto-scale FC loss so its backbone grad stays within max_grad_ratio.",
)
parser.add_argument("--wm_floorcritic_start_tol", type=float, default=0.04)
parser.add_argument("--wm_floorcritic_floor_tol", type=float, default=0.02)
parser.add_argument("--wm_floorcritic_hard_floor_tol", type=float, default=0.012)
parser.add_argument(
    "--wm_floorcritic_horizons", type=str, default="4,8",
    help="Comma-separated horizons for FC targets (e.g. 4,8).",
)
parser.add_argument("--wm_floorcritic_weight", type=float, default=0.005)
parser.add_argument("--wm_floorcritic_floor_weight", type=float, default=0.01)
parser.add_argument("--wm_floorcritic_hard_floor_weight", type=float, default=0.02)
parser.add_argument("--wm_floorcritic_near_success_mult", type=float, default=1.5)
parser.add_argument("--wm_floorcritic_dhat_improve_margin", type=float, default=0.005)
parser.add_argument("--wm_floorcritic_drop_dhat_margin", type=float, default=0.01)
parser.add_argument(
    "--wm_floorcritic_actor_detach",
    action=argparse.BooleanOptionalAction, default=True,
    help="Compat/redundant detach switch; feature_source is authoritative.",
)
parser.add_argument(
    "--wm_floorcritic_update_actor",
    action=argparse.BooleanOptionalAction, default=False,
    help="Experimental: allow non-detached feature (still never connects mu).",
)
parser.add_argument(
    "--wm_floorcritic_use_success",
    action=argparse.BooleanOptionalAction, default=True,
)
parser.add_argument(
    "--wm_floorcritic_use_near_success",
    action=argparse.BooleanOptionalAction, default=True,
)
parser.add_argument(
    "--wm_floorcritic_use_dhat_improve",
    action=argparse.BooleanOptionalAction, default=True,
)
parser.add_argument(
    "--wm_floorcritic_use_drop_risk",
    action=argparse.BooleanOptionalAction, default=True,
)

# ----------------------------------------------------------------------
# ERD-MPC (Elite-Recovery Distillation): action-conditioned recovery scorer
# + candidate-action search + actor-mu distillation. Default-off.
# ----------------------------------------------------------------------
parser.add_argument("--erd_enable", action=argparse.BooleanOptionalAction, default=False)
parser.add_argument("--erd_horizons", type=str, default="4,8")
parser.add_argument("--erd_candidate_k", type=int, default=32)
parser.add_argument("--erd_action_noise_std", type=float, default=0.15)
parser.add_argument("--erd_max_action_delta", type=float, default=0.5)
parser.add_argument("--erd_noise_cap_by_tol", action=argparse.BooleanOptionalAction, default=True)
parser.add_argument("--erd_score_success_weight", type=float, default=1.0)
parser.add_argument("--erd_score_improve_weight", type=float, default=0.7)
parser.add_argument("--erd_score_drop_weight", type=float, default=1.5)
parser.add_argument("--erd_scorer_loss_weight", type=float, default=0.01)
parser.add_argument("--erd_distill_loss_weight", type=float, default=0.03)
parser.add_argument("--erd_floor_distill_loss_weight", type=float, default=0.05)
parser.add_argument("--erd_hard_floor_distill_loss_weight", type=float, default=0.08)
parser.add_argument("--erd_start_tol", type=float, default=0.04)
parser.add_argument("--erd_floor_tol", type=float, default=0.02)
parser.add_argument("--erd_hard_floor_tol", type=float, default=0.012)
parser.add_argument("--erd_dhat_improve_margin", type=float, default=0.005)
parser.add_argument("--erd_drop_dhat_margin", type=float, default=0.01)
parser.add_argument("--erd_near_success_mult", type=float, default=2.0)
parser.add_argument("--erd_max_grad_ratio", type=float, default=0.2)
parser.add_argument("--erd_auto_scale_by_grad_ratio", action=argparse.BooleanOptionalAction, default=True)
parser.add_argument("--erd_only_lifted", action=argparse.BooleanOptionalAction, default=True)
parser.add_argument("--erd_min_risky_fraction", type=float, default=0.02)
parser.add_argument("--erd_start_after_frames", type=int, default=200000000)
parser.add_argument("--erd_scorer_warmup_iters", type=int, default=2000)
parser.add_argument("--erd_min_score_margin", type=float, default=0.05)
parser.add_argument("--erd_min_target_valid_fraction", type=float, default=0.2)
parser.add_argument("--erd_max_drop_prob_for_teacher", type=float, default=0.5)

# ----------------------------------------------------------------------
# EIR-MPC (Elite-Imitation Recovery MPC). Full aggressive method: elite
# recovery bank + elite candidate retrieval + gated recovery residual head
# (with gated rollout apply) + light RSSM reranking. PBT elite broadcast is
# wired separately in pbt_observer (Phase D). Default-off. When --eir_enable is
# set, the legacy ERD raw-mu distill path is NOT attached (constraint 19.1):
# EIR trains the residual head, never pulls main actor mu.
# ----------------------------------------------------------------------
parser.add_argument("--eir_enable", action=argparse.BooleanOptionalAction, default=False)
parser.add_argument("--eir_use_elite_bank", action=argparse.BooleanOptionalAction, default=True)
parser.add_argument("--eir_use_recovery_residual", action=argparse.BooleanOptionalAction, default=True)
parser.add_argument("--eir_use_rssm_mppi", action=argparse.BooleanOptionalAction, default=False)
parser.add_argument("--eir_horizons", type=str, default="4,8")
# elite bank
parser.add_argument("--eir_elite_bank_size", type=int, default=500000)
parser.add_argument("--eir_elite_sample_pool", type=int, default=4096)
parser.add_argument("--eir_elite_knn_topk", type=int, default=32)
parser.add_argument("--eir_elite_tol_threshold", type=float, default=0.02)
parser.add_argument("--eir_elite_hard_tol_threshold", type=float, default=0.012)
parser.add_argument("--eir_elite_near_success_mult", type=float, default=3.0)
parser.add_argument("--eir_elite_min_priority", type=float, default=1.0)
# CSER-v0 elite snippet bank + first-step distillation
parser.add_argument("--elite_snippet_enable", action=argparse.BooleanOptionalAction, default=False)
parser.add_argument("--elite_snippet_horizon", type=int, default=4, choices=[4, 8])
parser.add_argument("--elite_snippet_bank_size", type=int, default=50000)
parser.add_argument("--elite_snippet_sample_pool", type=int, default=4096)
parser.add_argument("--elite_snippet_knn_topk", type=int, default=32)
parser.add_argument("--elite_snippet_min_priority", type=float, default=1.0)
parser.add_argument("--snippet_distill_enable", action=argparse.BooleanOptionalAction, default=False)
parser.add_argument("--snippet_distill_weight", type=float, default=0.001)
parser.add_argument("--snippet_distill_loss_cap", type=float, default=0.05)
parser.add_argument("--snippet_distill_start_after_frames", type=int, default=300000000)
parser.add_argument("--snippet_distill_topk", type=int, default=8)
parser.add_argument("--snippet_distill_gate_mode", type=str, default="soft_tol",
                    choices=["off", "hard_tol", "soft_tol", "confidence_only"])
parser.add_argument("--snippet_distill_tol_start", type=float, default=0.070)
parser.add_argument("--snippet_distill_tol_full", type=float, default=0.045)
parser.add_argument("--snippet_distill_ramp_frames", type=int, default=200000000)
parser.add_argument("--snippet_wm_rerank_enable", action=argparse.BooleanOptionalAction, default=False)
parser.add_argument("--snippet_wm_rerank_w_rec", type=float, default=0.3)
parser.add_argument("--snippet_wm_rerank_w_success", type=float, default=0.2)
parser.add_argument("--snippet_wm_rerank_w_drop", type=float, default=0.4)
parser.add_argument("--snippet_wm_rerank_w_unc", type=float, default=0.1)
parser.add_argument("--snippet_wm_rerank_w_delta", type=float, default=0.05)
parser.add_argument("--snippet_wm_rerank_score_cap", type=float, default=0.25)
parser.add_argument("--wm_calibration_enable", action=argparse.BooleanOptionalAction, default=False)
parser.add_argument("--wm_calibration_eval_interval_frames", type=int, default=50000000)
parser.add_argument("--wm_calibration_min_samples", type=int, default=512)
parser.add_argument("--wm_calibration_heldout_fraction", type=float, default=0.2)
parser.add_argument("--wm_calibration_alpha_min", type=float, default=0.0)
parser.add_argument("--wm_calibration_alpha_max", type=float, default=1.0)
parser.add_argument("--wm_calibration_gate_auc_recovery", type=float, default=0.55)
parser.add_argument("--wm_calibration_gate_ece_max", type=float, default=0.15)
parser.add_argument("--wm_calibration_gate_rank_corr_min", type=float, default=0.1)
# candidates
parser.add_argument("--eir_local_candidate_k", type=int, default=16)
parser.add_argument("--eir_elite_candidate_k", type=int, default=32)
parser.add_argument("--eir_archive_candidate_k", type=int, default=16)
parser.add_argument("--eir_action_noise_std", type=float, default=0.15)
parser.add_argument("--eir_max_action_delta", type=float, default=0.5)
parser.add_argument("--eir_noise_cap_by_tol", action=argparse.BooleanOptionalAction, default=True)
# reranking score weights / gates
parser.add_argument("--eir_score_success_weight", type=float, default=1.0)
parser.add_argument("--eir_score_improve_weight", type=float, default=0.7)
parser.add_argument("--eir_score_recovery_weight", type=float, default=0.5)
parser.add_argument("--eir_score_drop_weight", type=float, default=1.5)
parser.add_argument("--eir_elite_priority_weight", type=float, default=0.2)
parser.add_argument("--eir_elite_similarity_weight", type=float, default=0.1)
parser.add_argument("--eir_action_delta_weight", type=float, default=0.05)
parser.add_argument("--eir_clip_penalty_weight", type=float, default=0.1)
parser.add_argument("--eir_min_score_margin", type=float, default=0.02)
parser.add_argument("--eir_min_elite_margin", type=float, default=0.01)
parser.add_argument("--eir_max_drop_prob_for_teacher", type=float, default=0.35)
# staging
parser.add_argument("--eir_start_tol", type=float, default=0.04)
parser.add_argument("--eir_floor_tol", type=float, default=0.02)
parser.add_argument("--eir_hard_floor_tol", type=float, default=0.012)
parser.add_argument("--eir_near_success_mult", type=float, default=2.0)
parser.add_argument("--eir_only_lifted", action=argparse.BooleanOptionalAction, default=True)
parser.add_argument("--eir_scorer_loss_weight", type=float, default=0.01)
parser.add_argument("--eir_start_after_frames", type=int, default=200000000)
parser.add_argument("--eir_scorer_warmup_iters", type=int, default=2000)
parser.add_argument("--eir_min_target_valid_fraction", type=float, default=0.2)
parser.add_argument("--eir_min_risky_fraction", type=float, default=0.02)
# stage-aware guidance (opt-in; default OFF keeps legacy floor-only behaviour).
# Adds a weak "mid" band above floor for elite bank collection + residual apply,
# plus a small source-gated RSSM rerank bonus. Floor/hard semantics unchanged.
parser.add_argument("--eir_guidance_enable", action=argparse.BooleanOptionalAction, default=False)
parser.add_argument("--eir_guidance_start_tol", type=float, default=0.04)
parser.add_argument("--eir_guidance_bank_tol", type=float, default=0.04)
parser.add_argument("--eir_guidance_apply_tol", type=float, default=0.04)
parser.add_argument("--eir_guidance_mid_near_success_mult", type=float, default=1.5)
parser.add_argument("--eir_guidance_mid_priority_scale", type=float, default=0.5)
parser.add_argument("--eir_guidance_mid_residual_alpha", type=float, default=0.15)
parser.add_argument("--eir_guidance_mid_gate_confidence_threshold", type=float, default=0.6)
parser.add_argument("--eir_guidance_rssm_source_bonus", type=float, default=0.03)
parser.add_argument("--eir_guidance_rssm_bonus_margin_scale", type=float, default=1.0)
parser.add_argument("--eir_guidance_rssm_bonus_uncertainty_penalty", type=float, default=0.01)
# recovery residual head + rollout apply
parser.add_argument("--eir_residual_scale", type=float, default=0.5)
parser.add_argument("--eir_residual_distill_weight", type=float, default=0.05)
parser.add_argument("--eir_residual_floor_distill_weight", type=float, default=0.08)
parser.add_argument("--eir_residual_hard_floor_distill_weight", type=float, default=0.12)
parser.add_argument("--eir_gate_sparsity_weight", type=float, default=0.001)
parser.add_argument("--eir_delta_l2_weight", type=float, default=0.001)
parser.add_argument("--eir_mu_base_detach", action=argparse.BooleanOptionalAction, default=True)
parser.add_argument("--eir_residual_apply_to_rollout", action=argparse.BooleanOptionalAction, default=False)
parser.add_argument("--eir_residual_apply_after_frames", type=int, default=300000000)
parser.add_argument("--eir_residual_apply_alpha_mid", type=float, default=0.25)
parser.add_argument("--eir_residual_apply_alpha_floor", type=float, default=0.5)
parser.add_argument("--eir_residual_apply_alpha_hard_floor", type=float, default=0.75)
parser.add_argument("--eir_residual_apply_tol_threshold", type=float, default=0.02)
parser.add_argument("--eir_residual_near_success_mult", type=float, default=2.0)
parser.add_argument("--eir_residual_gate_confidence_threshold", type=float, default=0.5)
parser.add_argument("--eir_residual_apply_max_aux_age_frames", type=int, default=100000000)
# WM-MPC actor distillation (v1): auxiliary Huber loss on actor mu head.
parser.add_argument("--eir_actor_distill_enable", action=argparse.BooleanOptionalAction, default=False)
parser.add_argument("--eir_actor_distill_weight", type=float, default=0.001)
parser.add_argument("--eir_actor_distill_start_after_frames", type=int, default=500000000)
parser.add_argument("--eir_actor_distill_tol", type=float, default=0.04)
parser.add_argument("--eir_actor_distill_min_margin", type=float, default=0.02)
parser.add_argument("--eir_actor_distill_margin_norm", type=float, default=0.05)
parser.add_argument("--eir_actor_distill_max_delta", type=float, default=0.35)
parser.add_argument("--eir_actor_distill_allowed_sources", type=str, default="rssm,elite,archive")
parser.add_argument("--eir_actor_distill_uncertainty_threshold", type=float, default=2.0)
parser.add_argument("--eir_actor_distill_loss_cap", type=float, default=0.05)
parser.add_argument("--eir_actor_distill_feature_detach", action=argparse.BooleanOptionalAction, default=True)
parser.add_argument("--eir_actor_distill_mu_head_only", action=argparse.BooleanOptionalAction, default=True)
# PBT-WM Soft Activation Gate v1 (shared toggles + per-module soft tol schedule).
parser.add_argument("--wm_activation_enable", action=argparse.BooleanOptionalAction, default=True,
                    help="Master switch for the soft activation gate (per-module).")
parser.add_argument("--wm_activation_log_by_tol_bin", action=argparse.BooleanOptionalAction, default=True,
                    help="Emit per-tol-bin activation telemetry.")
# actor-distill soft gate (default hard_tol preserves legacy behaviour).
parser.add_argument("--eir_actor_distill_gate_mode", type=str, default="hard_tol",
                    choices=["off", "hard_tol", "soft_tol", "confidence_only"])
parser.add_argument("--eir_actor_distill_tol_start", type=float, default=0.060)
parser.add_argument("--eir_actor_distill_tol_full", type=float, default=0.040)
parser.add_argument("--eir_actor_distill_ramp_frames", type=int, default=300000000)
parser.add_argument("--eir_actor_distill_nsucc_ref", type=float, default=20.0)
# elite-distill soft gate (profile elite_bank_actor_distill diagnostics).
parser.add_argument("--elite_distill_gate_mode", type=str, default="hard_tol",
                    choices=["off", "hard_tol", "soft_tol", "confidence_only"])
parser.add_argument("--elite_distill_tol_start", type=float, default=0.070)
parser.add_argument("--elite_distill_tol_full", type=float, default=0.045)
parser.add_argument("--elite_distill_ramp_frames", type=int, default=200000000)
parser.add_argument("--elite_distill_nsucc_ref", type=float, default=15.0)
# conservative planner soft gate (profile conservative_rssm_planner_distill).
parser.add_argument("--conservative_planner_gate_mode", type=str, default="hard_tol",
                    choices=["off", "hard_tol", "soft_tol", "confidence_only"])
parser.add_argument("--conservative_planner_tol_start", type=float, default=0.055)
parser.add_argument("--conservative_planner_tol_full", type=float, default=0.035)
parser.add_argument("--conservative_planner_ramp_frames", type=int, default=300000000)
parser.add_argument("--conservative_planner_nsucc_ref", type=float, default=25.0)
# Soft candidate-retrieval gate (default hard_tol = legacy behaviour).
parser.add_argument(
    "--eir_candidate_gate_mode",
    type=str,
    default="hard_tol",
    choices=["hard_tol", "soft_tol", "profile_soft"],
    help="Soft tolerance gate for elite/archive/RSSM candidate retrieval.",
)
parser.add_argument("--eir_elite_candidate_tol_start", type=float, default=0.070)
parser.add_argument("--eir_elite_candidate_tol_full", type=float, default=0.045)
parser.add_argument("--eir_rssm_candidate_tol_start", type=float, default=0.060)
parser.add_argument("--eir_rssm_candidate_tol_full", type=float, default=0.040)
parser.add_argument(
    "--eir_conservative_rssm_candidate_tol_start", type=float, default=0.055
)
parser.add_argument(
    "--eir_conservative_rssm_candidate_tol_full", type=float, default=0.035
)
parser.add_argument(
    "--eir_candidate_min_weight_for_retrieval", type=float, default=0.05
)
# WM PBT source selector soft gate (no tol hard gate).
parser.add_argument("--wm_pbt_source_gate_mode", type=str, default="confidence_only",
                    choices=["off", "confidence_only"])
parser.add_argument("--wm_pbt_source_start_after_frames", type=int, default=300000000)
parser.add_argument("--wm_pbt_source_ramp_frames", type=int, default=300000000)
# WM curriculum controller gate mode (shrink-candidate only; fail-open).
parser.add_argument("--wm_curriculum_gate_mode", type=str, default="shrink_candidate",
                    choices=["off", "shrink_candidate"])
# RSSM reranking / full H-step MPPI
parser.add_argument("--eir_mppi_mode", type=str, default="full", choices=["off", "light", "full"])
parser.add_argument("--eir_mppi_start_after_frames", type=int, default=800000000)
parser.add_argument("--eir_mppi_uncertainty_threshold", type=float, default=0.4)
parser.add_argument("--eir_mppi_horizon", type=int, default=4)
parser.add_argument("--eir_mppi_num_sequences", type=int, default=64)
parser.add_argument("--eir_mppi_num_elite", type=int, default=8)
parser.add_argument("--eir_mppi_cem_iters", type=int, default=1)
parser.add_argument("--eir_mppi_action_noise_std", type=float, default=0.10)
parser.add_argument("--eir_mppi_action_noise_decay", type=float, default=0.85)
parser.add_argument("--eir_mppi_max_action_delta", type=float, default=0.5)
parser.add_argument("--eir_mppi_smooth_noise", action=argparse.BooleanOptionalAction, default=True)
parser.add_argument("--eir_mppi_min_sequence_margin", type=float, default=0.02)
parser.add_argument("--eir_mppi_w_success", type=float, default=1.0)
parser.add_argument("--eir_mppi_w_recovery", type=float, default=0.8)
parser.add_argument("--eir_mppi_w_drop", type=float, default=2.0)
parser.add_argument("--eir_mppi_w_value", type=float, default=0.2)
parser.add_argument("--eir_mppi_w_task_improve", type=float, default=0.5)
parser.add_argument("--eir_mppi_w_uncertainty", type=float, default=0.2)
parser.add_argument("--eir_mppi_w_action_smoothness", type=float, default=0.05)
parser.add_argument("--eir_mppi_w_action_delta", type=float, default=0.02)
parser.add_argument("--eir_mppi_fail_fast", action=argparse.BooleanOptionalAction, default=False)
parser.add_argument("--eir_mppi_stochastic_rollouts", action=argparse.BooleanOptionalAction, default=False)
parser.add_argument("--eir_mppi_num_latent_samples", type=int, default=1)
parser.add_argument("--eir_mppi_sample_temperature", type=float, default=1.0)
parser.add_argument("--eir_mppi_risk_beta", type=float, default=0.0)
parser.add_argument("--eir_mppi_drop_cvar_alpha", type=float, default=0.25)
parser.add_argument("--eir_mppi_drop_cvar_weight", type=float, default=0.0)
parser.add_argument("--eir_mppi_uncertainty_cvar_weight", type=float, default=0.0)
parser.add_argument(
    "--eir_mppi_score_aggregation",
    type=str,
    default="mean",
    choices=["mean", "mean_minus_std", "cvar"],
)
parser.add_argument("--eir_mppi_max_rollout_batch", type=int, default=65536)
# Full RSSM ensemble (MPPI scoring only). Default off -> single-model behaviour.
parser.add_argument("--seq_wm_ensemble_enable", action=argparse.BooleanOptionalAction, default=False)
parser.add_argument("--seq_wm_ensemble_size", type=int, default=1)
parser.add_argument("--seq_wm_ensemble_bootstrap", action=argparse.BooleanOptionalAction, default=True)
parser.add_argument("--seq_wm_ensemble_bootstrap_keep_prob", type=float, default=0.8)
parser.add_argument("--seq_wm_ensemble_save", action=argparse.BooleanOptionalAction, default=True)
parser.add_argument("--seq_wm_ensemble_load_mode", type=str, default="reset", choices=["reset", "resume"])
parser.add_argument("--eir_mppi_use_model_ensemble", action=argparse.BooleanOptionalAction, default=False)
parser.add_argument("--eir_mppi_ensemble_risk_beta", type=float, default=0.0)
parser.add_argument("--eir_mppi_ensemble_drop_weight", type=float, default=0.0)
parser.add_argument("--eir_mppi_ensemble_uncertainty_weight", type=float, default=0.0)
parser.add_argument("--eir_mppi_ensemble_min_valid_models", type=int, default=2)
parser.add_argument("--eir_mppi_ensemble_max_models_per_plan", type=int, default=3)
# Ensemble worst-case / CVaR uncertainty is on a different scale than the
# single-model prior_std, so the ensemble path uses its own absolute gate.
parser.add_argument("--eir_mppi_ensemble_uncertainty_threshold", type=float, default=2.0)
parser.add_argument("--eir_mppi_ensemble_uncertainty_allow_margin", type=float, default=0.10)
# PBT elite broadcast (consumed in pbt_observer Phase D; flags defined here so
# launchers can set them ahead of Phase D wiring).
parser.add_argument("--eir_pbt_broadcast_elite_bank", action=argparse.BooleanOptionalAction, default=True)
parser.add_argument("--eir_pbt_broadcast_threshold_nsucc_mean", type=float, default=38.0)
parser.add_argument("--eir_pbt_broadcast_gap_over_pop_mean", type=float, default=6.0)
parser.add_argument("--eir_pbt_broadcast_to_bottom_k", type=int, default=3)
parser.add_argument("--eir_pbt_freeze_elite_curriculum_frames", type=int, default=100000000)
parser.add_argument("--eir_pbt_max_bank_merge_entries", type=int, default=50000)
parser.add_argument("--eir_pbt_min_frames_between_broadcasts_per_source", type=int, default=100000000)
parser.add_argument("--eir_pbt_min_frames_between_broadcasts_per_target", type=int, default=100000000)
# Strategy Zoo v1.7 — Online Elite-AWR (default off; profiles enable per-worker).
parser.add_argument("--awr_enable", action=argparse.BooleanOptionalAction, default=False)
parser.add_argument("--awr_loss_weight", type=float, default=0.0)
parser.add_argument("--awr_start_after_frames", type=int, default=800_000_000)
parser.add_argument("--awr_ramp_frames", type=int, default=300_000_000)
parser.add_argument("--awr_min_buffer_size", type=int, default=50_000)
parser.add_argument("--awr_buffer_capacity", type=int, default=400_000)
parser.add_argument(
    "--awr_batch_steps",
    type=int,
    default=4096,
    help="Number of transition steps per AWR update, not number of episode entries.",
)
parser.add_argument(
    "--awr_batch_size",
    type=int,
    default=None,
    help="Deprecated alias for --awr_batch_steps.",
)
parser.add_argument("--awr_minibatch_max_sequences", type=int, default=128)
parser.add_argument("--awr_update_every", type=int, default=4)
parser.add_argument("--awr_max_episode_len", type=int, default=200)
parser.add_argument("--awr_min_burnin_steps", type=int, default=32)
parser.add_argument(
    "--awr_min_loss_steps_per_entry",
    type=int,
    default=None,
    help="Min loss-masked steps per AWR entry; default 4 (rollout_buffer) or 16 (interceptor).",
)
parser.add_argument("--awr_weight_mode", type=str, default="exp_quality",
                    choices=["quality", "exp_quality", "advantage"])
parser.add_argument("--awr_beta", type=float, default=1.0)
parser.add_argument("--awr_weight_max", type=float, default=5.0)
parser.add_argument("--awr_quality_min", type=float, default=1.5)
parser.add_argument(
    "--awr_delta_metric",
    type=str,
    default="mean",
    choices=("linf", "mean", "rms", "p90"),
)
parser.add_argument("--awr_delta_soft", type=float, default=0.20)
parser.add_argument("--awr_delta_mid", type=float, default=0.45)
parser.add_argument("--awr_delta_hard", type=float, default=0.80)
parser.add_argument("--awr_delta_mid_weight", type=float, default=0.5)
parser.add_argument("--awr_delta_far_weight", type=float, default=0.2)
parser.add_argument("--awr_loss_cap", type=float, default=0.03)
parser.add_argument(
    "--awr_kl_gate_enable",
    action=argparse.BooleanOptionalAction,
    default=False,
)
parser.add_argument("--awr_kl_soft", type=float, default=0.05)
parser.add_argument("--awr_kl_hard", type=float, default=0.10)
parser.add_argument("--awr_ref_update_interval_frames", type=int, default=200_000_000)
parser.add_argument("--awr_local_sample_frac", type=float, default=0.7)
parser.add_argument("--awr_inherited_sample_frac", type=float, default=0.2)
parser.add_argument("--awr_broadcast_sample_frac", type=float, default=0.1)
parser.add_argument("--awr_copy_top_k_steps", type=int, default=50_000)
parser.add_argument("--awr_copy_top_k", type=int, default=None,
                    help="Deprecated alias for --awr_copy_top_k_steps.")
parser.add_argument("--awr_broadcast_top_k_steps", type=int, default=10_000)
parser.add_argument("--awr_broadcast_top_k", type=int, default=None,
                    help="Deprecated alias for --awr_broadcast_top_k_steps.")
parser.add_argument("--awr_broadcast_min_source_nsucc", type=float, default=35.0)
parser.add_argument("--awr_pause_on_regression", action=argparse.BooleanOptionalAction, default=True)
parser.add_argument("--awr_regression_drop_threshold", type=float, default=4.0)
parser.add_argument("--awr_regression_pause_frames", type=int, default=100_000_000)
parser.add_argument(
    "--awr_collection_mode",
    type=str,
    default="rollout_buffer",
    choices=("rollout_buffer", "interceptor"),
    help="AWR data collection path: rollout_buffer (default) or legacy env interceptor.",
)
parser.add_argument(
    "--awr_collect_after_frames",
    type=int,
    default=600_000_000,
    help="Skip AWR rollout collection until this env-frame count.",
)
parser.add_argument(
    "--awr_collect_every",
    type=int,
    default=4,
    help="After collect_after_frames, run builder every N PPO rollouts (1=every rollout).",
)
parser.add_argument(
    "--pbt_copy_reset_progress_on_large_frame_gap",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="On PBT copy restore, do not inherit donor epoch/frame when gap exceeds threshold.",
)
parser.add_argument(
    "--pbt_copy_frame_gap_reset_threshold",
    type=int,
    default=500_000_000,
    help="Frame gap above which PBT restore keeps local progress counters.",
)
parser.add_argument(
    "--pbt_copy_target_frame_before_restore",
    type=int,
    default=None,
    help="Target slot population_frame at PBT cross-copy restore (snapshot-driven).",
)
parser.add_argument(
    "--pbt_lineage_id_resume",
    type=str,
    default=None,
    help="Lineage id carried across supervisor restarts for causal skill_delta windows.",
)
parser.add_argument(
    "--pbt_lineage_start_pf_resume",
    type=int,
    default=None,
    help="Population frame at which the current lineage began (snapshot-driven).",
)
parser.add_argument(
    "--pbt_weight_age_frame_resume",
    type=int,
    default=None,
    help="Weight age at lineage start, carried across supervisor restarts.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# ----------------------------------------------------------------------
# Phase A4 P4 (2026-06-09): expand --experimental_rssm_floor_recovery_stack
# (Item 10). The preset toggles ON every per-feature flag that has
# default-False. Each kill switch keeps overriding via argparse normal
# semantics: any explicit --no-* on the command line is parsed BEFORE we
# get here, so users can still disable individual layers. We log the
# expanded flag set so the run audit captures every switch flipped.
# ----------------------------------------------------------------------
if bool(getattr(args_cli, "experimental_rssm_floor_recovery_stack", False)):
    _stack_flags = [
        ("pbt_floor_recovery_aware", True),
        ("pbt_enable_best_floor_recovery_archive", True),
        ("seq_wm_use_task_aux_inputs", True),
        ("seq_wm_floor_aware_weighting", True),
        ("seq_wm_use_horizon_heads", True),
        ("seq_wm_use_recovery_head", True),
        ("seq_wm_enable_iterative_teacher", True),
        ("seq_wm_enable_imagined_evaluator", True),
    ]
    _audit_lines: list[str] = []
    for flag_name, target in _stack_flags:
        cur_val = getattr(args_cli, flag_name, None)
        if cur_val is False:  # only flip implicit defaults
            setattr(args_cli, flag_name, target)
            _audit_lines.append(f"{flag_name}: False -> True (preset)")
        else:
            _audit_lines.append(
                f"{flag_name}: kept {cur_val!r} (kill switch wins)"
            )
    # WM-guided ranking: preset bumps to 'shadow' (NEVER 'active' by
    # default — Item 3 requires explicit opt-in for active mode).
    if str(getattr(args_cli, "pbt_wm_score_mode", "off")) == "off":
        args_cli.pbt_wm_score_mode = "shadow"
        _audit_lines.append("pbt_wm_score_mode: off -> shadow (preset)")
    else:
        _audit_lines.append(
            f"pbt_wm_score_mode: kept {args_cli.pbt_wm_score_mode!r} "
            "(kill switch wins)"
        )
    # Lambdas for the new heads default to 0; the preset turns them on
    # so the heads actually contribute to the loss when use_*_head flags
    # are true. Users can still override via --seq_wm_lambda_*.
    if float(getattr(args_cli, "seq_wm_lambda_horizon_success", 0.0)) <= 0.0:
        args_cli.seq_wm_lambda_horizon_success = 0.5
    if float(getattr(args_cli, "seq_wm_lambda_drop_risk", 0.0)) <= 0.0:
        args_cli.seq_wm_lambda_drop_risk = 0.5
    if float(getattr(args_cli, "seq_wm_lambda_recovery", 0.0)) <= 0.0:
        args_cli.seq_wm_lambda_recovery = 1.0
    print(
        "[experimental_rssm_floor_recovery_stack] expansion:\n  "
        + "\n  ".join(_audit_lines)
    )

# PBT-WM Strategy Zoo v1.0: expand --wm_strategy_profile into concrete flags.
from dexpbt_isaaclab.world_model.wm_strategy_profiles import (  # noqa: E402
    PROFILE_IDS,
    apply_wm_strategy_profile,
    profile_capability_flags,
)

_wm_profile_audit = apply_wm_strategy_profile(
    args_cli,
    argv=sys.argv,
    authoritative=(
        str(args_cli.wm_strategy_profile) != "legacy"
        and not bool(args_cli.wm_strategy_profile_allow_cli_override)
    ),
)
_wm_profile_caps = profile_capability_flags(args_cli)

# Save original sys.argv before modification so PBT observer can reconstruct
# the restart command with all parsed arguments (e.g. --task, --observer_mode).
_original_sys_argv = list(sys.argv)

sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# --------------------------------------------------------------------------
# Imports that require Omniverse runtime
# --------------------------------------------------------------------------

import math
import json
import os
import time
from datetime import datetime
from importlib import resources
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
import yaml
from rl_games.common import env_configurations, vecenv
from rl_games.common.algo_observer import IsaacAlgoObserver
from rl_games.torch_runner import Runner

from isaaclab.utils.io import dump_yaml

from isaaclab_rl.rl_games import RlGamesGpuEnv, RlGamesVecEnvWrapper

# register our envs so gym.make can find them
import dexpbt_isaaclab  # noqa: F401
from dexpbt_isaaclab.pbt.pbt_observer import DexPBTPbtAlgoObserver

# Phase A4: sequence WM (RSSM-lite, Transformer-lite) — kept import-light
# so legacy strategies do not pay the cost.
from dexpbt_isaaclab.world_model.wm_trainer import (
    SEQUENCE_STRATEGIES, RSSM_FAMILY, TRANS_FAMILY,
)
from dexpbt_isaaclab.world_model.seq_wm_trainer import SeqWMConfig, SeqWMTrainer
from dexpbt_isaaclab.world_model.wm_calibration_evaluator import (
    WMCalibrationConfig,
    WMCalibrationEvaluator,
    WMCalibrationState,
)
from dexpbt_isaaclab.world_model.seq_aux_collector import (
    SeqAuxCollector, SeqAuxInterceptorEnv,
)
from dexpbt_isaaclab.world_model.seq_policy_aux import (
    COUPLING_MODE_ID,
    FEATURE_SOURCE_ID,
    CapturedFeature,
    SeqDistillBridge,
    SeqPolicyCouplingConfig,
    attach_seq_aux_heads,
    compute_seq_distill_aux_loss,
    register_actor_mlp_capture,
    register_shared_to_aux_critic_capture,
)
from dexpbt_isaaclab.world_model.wm_aux_buffer import (
    DEFAULT_SEQ_AUX_FIELDS,
    EXTENDED_SEQ_AUX_FIELDS,
    register_seq_aux_fields,
)
from dexpbt_isaaclab.world_model.rssm_lite import RSSMLiteWorldModel
from dexpbt_isaaclab.world_model.transformer_lite import TransformerLiteWorldModel
# Phase A4 P4 wired (2026-06-10): runtime wiring for the iterative
# teacher manager (Part C) and the short-horizon imagined evaluator
# (Part D). Both are gated by their CLI flags; default-off path is
# bit-for-bit unchanged.
from dexpbt_isaaclab.world_model.wm_teacher_manager import (
    FloorRecoveryEvalBuffer,
    TeacherEvalMetrics,
    TeacherManager,
)
from dexpbt_isaaclab.world_model.wm_imagined_evaluator import WMImaginedEvaluator
# FloorCritic (Real-target) — default-off auxiliary critic heads. Import is
# cheap (no torch model construction at import time).
from dexpbt_isaaclab.world_model.floorcritic import (
    FloorCriticBridge,
    FloorCriticConfig,
    FloorCriticTargetBuilder,
    attach_floorcritic_heads,
    compute_floorcritic_loss,
    register_actor_mlp_grad_probe,
    register_floorcritic_capture,
)
from dexpbt_isaaclab.world_model.elite_recovery_distill import (
    ERDBridge,
    ERDConfig,
    ERDTargetBuilder,
    _flatten_tn as _erd_flatten_tn,
    attach_erd_modules,
    compute_erd_loss,
    register_erd_actor_mlp_grad_probe,
    register_erd_capture,
)
from dexpbt_isaaclab.world_model.eir_mpc import (
    EIRBridge,
    EIRConfig,
    compute_eir_loss,
    parse_actor_distill_allowed_sources,
    validate_actor_distill_v1_flags,
)
from dexpbt_isaaclab.world_model.eir_broadcast import (
    EliteBroadcastConfig,
    decide_consume_mode,
    load_payload,
    merge_payload_into_bridge,
    mark_consumed,
    read_broadcast,
)
from dexpbt_isaaclab.world_model.elite_recovery_bank import (
    EliteRecoveryBank,
    EliteRecoveryBankConfig,
)
from dexpbt_isaaclab.world_model.elite_snippet_bank import (
    EliteSnippetBank,
    EliteSnippetBankConfig,
)
from dexpbt_isaaclab.world_model.elite_snippet_builder import SnippetTargetBuilder
from dexpbt_isaaclab.world_model.snippet_distill import SnippetDistillConfig
from dexpbt_isaaclab.world_model.snippet_wm_reranker import SnippetWMRerankConfig
from dexpbt_isaaclab.world_model.recovery_residual import (
    RecoveryResidualBridge,
    RecoveryResidualConfig,
    attach_recovery_residual,
    compute_rollout_residual_action,
)
from dexpbt_isaaclab.world_model.rssm_mppi_planner import (
    RSSMReranker,
    RSSMRerankConfig,
    RSSMMPPIConfig,
    RSSMMPPIPlanner,
)
from dexpbt_isaaclab.world_model.rssm_belief_provider import build_rssm_belief_cache
from dexpbt_isaaclab.world_model.rssm_ensemble import (
    RSSMEnsembleConfig,
    RSSMEnsembleManager,
    restore_rssm_ensemble_checkpoint,
)


# ----------------------------------------------------------------------
# CF meta tracker (counterfactual block r_meta aggregation)
# ----------------------------------------------------------------------


from dexpbt_isaaclab.cf_utils import compute_child_train_frames as _compute_child_train_frames  # noqa: E402, F401


class _CfMetaTracker:
    """Track r_meta + nsucc snapshots during a CF child run.

    Patch 3 (rev 2, 2026-05-30): now records a parallel `nsucc` channel
    so that the CF block runner can detect low_signal blocks where
    r_meta moves but nsucc stays at zero (i.e. the change is entirely
    explained by tolerance progress, not by real learning).
    """

    def __init__(self) -> None:
        self.samples: list[float] = []
        # Parallel list with identical length: nsucc_mean at each tick.
        self.nsucc_samples: list[float] = []
        # success_tolerance at each tick; useful when reconstructing the
        # r_meta tolerance-component change.
        self.tolerance_samples: list[float] = []

    def update(self, r_meta: float, nsucc_mean: float | None = None,
               success_tolerance: float | None = None) -> None:
        if r_meta is None:
            return
        val = float(r_meta)
        if math.isnan(val) or math.isinf(val):
            return
        self.samples.append(val)
        if nsucc_mean is None or math.isnan(float(nsucc_mean)):
            self.nsucc_samples.append(float("nan"))
        else:
            self.nsucc_samples.append(float(nsucc_mean))
        if success_tolerance is None or math.isnan(float(success_tolerance)):
            self.tolerance_samples.append(float("nan"))
        else:
            self.tolerance_samples.append(float(success_tolerance))

    def summary(self) -> dict[str, float]:
        if not self.samples:
            return {"r_meta": float("nan"), "r_meta_mean": float("nan"), "r_meta_max": float("nan")}
        arr = np.array(self.samples, dtype=np.float64)
        out = {
            "r_meta": float(arr[-1]),
            "r_meta_mean": float(arr.mean()),
            "r_meta_max": float(arr.max()),
        }
        # Patch 3 (rev 2): expose first/last snapshots so the CF block
        # runner can compute learning_improvement = (after - before).
        out["r_meta_before"] = float(arr[0])
        out["r_meta_after"] = float(arr[-1])
        if self.nsucc_samples:
            n_arr = np.array(self.nsucc_samples, dtype=np.float64)
            valid = n_arr[~np.isnan(n_arr)]
            out["nsucc_before"] = (
                float(n_arr[0]) if not math.isnan(float(n_arr[0])) else float("nan")
            )
            out["nsucc_after"] = (
                float(n_arr[-1]) if not math.isnan(float(n_arr[-1])) else float("nan")
            )
            out["nsucc_max_over_training"] = (
                float(valid.max()) if valid.size else float("nan")
            )
        if self.tolerance_samples:
            t_arr = np.array(self.tolerance_samples, dtype=np.float64)
            out["success_tolerance_before"] = (
                float(t_arr[0]) if not math.isnan(float(t_arr[0])) else float("nan")
            )
            out["success_tolerance_after"] = (
                float(t_arr[-1]) if not math.isnan(float(t_arr[-1])) else float("nan")
            )
        return out


def _compute_env_r_meta(raw_env) -> tuple[float, float]:
    """Return (r_meta, mean_nsucc) from current env state."""
    from dexpbt_isaaclab.dexpbt_utils import tolerance_successes_objective

    successes_tensor = getattr(raw_env, "prev_episode_successes", None)
    if successes_tensor is not None and hasattr(successes_tensor, "mean"):
        mean_nsucc = float(successes_tensor.mean().item())
    else:
        mean_nsucc = float("nan")

    success_tol = float(getattr(raw_env, "success_tolerance", float("nan")))
    init_tol = float(getattr(raw_env, "initial_tolerance", success_tol))
    target_tol = float(getattr(raw_env, "target_tolerance", success_tol))

    if math.isnan(mean_nsucc) or math.isnan(success_tol):
        return float("nan"), mean_nsucc

    r_meta_tensor = tolerance_successes_objective(
        success_tol,
        init_tol,
        target_tol,
        torch.tensor([mean_nsucc], dtype=torch.float32),
    )
    return float(r_meta_tensor[0].item()), mean_nsucc


def _patch_actor_train_for_cf_meta(actor, raw_env, cf_tracker: _CfMetaTracker | None) -> None:
    """Record r_meta + nsucc + tolerance each PPO epoch when running a CF child.

    Patch 3 (rev 2, 2026-05-30): also push nsucc_mean and success_tolerance
    onto the tracker so the CF block runner can spot low_signal blocks
    where r_meta moves only because tolerance shrank but nsucc stayed 0.
    """
    if cf_tracker is None or raw_env is None:
        return
    import functools

    original_train_epoch = actor.train_epoch

    @functools.wraps(original_train_epoch)
    def patched_train_epoch(*args, **kwargs):
        result = original_train_epoch(*args, **kwargs)
        r_meta, mean_nsucc = _compute_env_r_meta(raw_env)
        success_tol = float(getattr(raw_env, "success_tolerance", float("nan")))
        cf_tracker.update(r_meta, nsucc_mean=mean_nsucc, success_tolerance=success_tol)
        return result

    actor.train_epoch = patched_train_epoch


# ----------------------------------------------------------------------
# WM Actor Training Patch
# ----------------------------------------------------------------------

def _patch_actor_train_for_wm(actor, wm_trainer, wm_ckpt_manager, wm_strategy):
    """
    Monkey-patch rl_games A2CAgent.train_epoch() to call WM trainer each epoch.

    train() is a while-True loop that calls train_epoch() once per epoch.
    Patching train_epoch() ensures wm_trainer.step() runs after EVERY epoch,
    not just once at the end of the entire training run.
    """
    import functools

    original_train_epoch = actor.train_epoch

    @functools.wraps(original_train_epoch)
    def patched_train_epoch(*args, **kwargs):
        result = original_train_epoch(*args, **kwargs)
        # After each PPO epoch, run WM step
        if wm_trainer is not None:
            frames = getattr(actor, "frame", 0)
            # Lazily connect tensorboard writer (available after A2CCommon.__init__)
            if wm_trainer.tensorboard is None:
                tb = getattr(actor, "writer", None)
                if tb is not None:
                    wm_trainer.tensorboard = tb
            if (
                wm_trainer.tensorboard is not None
                and hasattr(wm_trainer, "_warm_start_log")
                and not getattr(wm_trainer, "_warm_start_log_emitted", False)
            ):
                wm_log = wm_trainer._warm_start_log
                wm_trainer.tensorboard.add_scalar(
                    "wm/checkpoint_loaded",
                    float(bool(wm_log.get("loaded", False))),
                    frames,
                )
                wm_trainer.tensorboard.add_scalar(
                    "wm/checkpoint_compatible",
                    float(bool(wm_log.get("compatible", False))),
                    frames,
                )
                wm_trainer.tensorboard.add_text(
                    "wm/checkpoint_source_strategy",
                    str(wm_log.get("checkpoint_strategy", "unknown")),
                    frames,
                )
                wm_trainer.tensorboard.add_text(
                    "wm/checkpoint_target_strategy",
                    str(wm_log.get("target_strategy", "unknown")),
                    frames,
                )
                wm_trainer.tensorboard.add_text(
                    "wm/checkpoint_reset_reason",
                    str(wm_log.get("reset_reason", "")),
                    frames,
                )
                wm_trainer._warm_start_log_emitted = True
            metrics = wm_trainer.step(frames=frames)
            if metrics and wm_trainer.strategy.trains_wm:
                if wm_trainer.tensorboard is not None:
                    for key, val in metrics.items():
                        wm_trainer.tensorboard.add_scalar(key, val, frames)
                # Save WM checkpoint periodically
                if wm_trainer._epoch_count % 10 == 0 and wm_ckpt_manager is not None:
                    aux_state = wm_trainer.get_aux_value_head_state()
                    wm_ckpt_manager.save(
                        wm_module=wm_trainer.module,
                        wm_optimizer=wm_trainer.optimizer,
                        strategy=wm_strategy,
                        iteration=wm_trainer._epoch_count,
                        frames=frames,
                        aux_value_head_state=aux_state,
                    )
                    print(f"[WM] checkpoint saved at epoch {wm_trainer._epoch_count}")
        return result

    actor.train_epoch = patched_train_epoch


# ----------------------------------------------------------------------
# Sequence WM Actor Training Patch (Phase A4)
# ----------------------------------------------------------------------

def _atomic_write_json(path: "Path", payload: dict) -> None:
    """Write ``payload`` as JSON to ``path`` using a unique tmp + rename.

    Used by the WM runtime wiring to publish per-policy ``wm_imagined``
    snapshots that observers in other processes consume at PBT ticks.
    """
    import os as _os
    import uuid as _uuid

    tmp = path.parent / f"{path.stem}.{_os.getpid()}.{_uuid.uuid4().hex[:8]}.tmp"
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        tmp.write_text(json.dumps(payload, indent=2))
        _os.replace(str(tmp), str(path))
    finally:
        try:
            if tmp.exists():
                tmp.unlink()
        except OSError:
            pass


def _run_wm_runtime_wiring(
    *,
    actor,
    seq_wm_trainer: SeqWMTrainer,
    frames: int,
    last_obs,
    returns_full,
    last_d_hat,
    last_success_count,
) -> dict:
    """Run the iterative-teacher and imagined-evaluator side-channels.

    All work is gated by per-module ``enabled`` flags; if both are off
    this function only returns three counters (``teacher_disabled``,
    ``imagine_disabled``, ``invalid_reason_hash``). Failures are logged
    but never raise — the WM runtime wiring is a best-effort overlay
    and must never destabilise PPO/PBT.
    """
    out: dict[str, float] = {}
    teacher = getattr(seq_wm_trainer, "_teacher_manager", None)
    imag = getattr(seq_wm_trainer, "_imagined_evaluator", None)
    eval_buf = getattr(seq_wm_trainer, "_floor_eval_buffer", None)
    record_below_tol = float(
        getattr(seq_wm_trainer, "_teacher_record_below_tol", 0.04)
    )

    teacher_enabled = bool(teacher is not None and teacher.enabled)
    imagine_enabled = bool(imag is not None and imag.enabled)
    out["teacher_disabled"] = float(0.0 if teacher_enabled else 1.0)
    out["imagine_disabled"] = float(0.0 if imagine_enabled else 1.0)

    if not (teacher_enabled or imagine_enabled):
        return out

    # Re-sample one batch through the trainer's sampler so eval +
    # imagined-eval consume the same canonical pipeline used for
    # training. This costs roughly one extra sampler.sample() per PPO
    # epoch — negligible vs the actual model update.
    import torch as _t
    from dexpbt_isaaclab.world_model.wm_aux_buffer import read_aux_full

    try:
        td = actor.experience_buffer.tensor_dict
        with _t.no_grad():
            obs_full = td["obses"]
            action_full = td["actions"]
            reward_full = td["rewards"]
            done_full = td["dones"].to(obs_full.dtype)
            d_hat_full = read_aux_full(actor.experience_buffer, "aux_d_hat")
            failure_full = read_aux_full(actor.experience_buffer, "aux_failure")
            lifted_full = read_aux_full(actor.experience_buffer, "aux_lifted")
            success_count_full = read_aux_full(
                actor.experience_buffer, "aux_success_cnt"
            )
            picked_full = read_aux_full(actor.experience_buffer, "aux_picked")
            dropped_full = read_aux_full(actor.experience_buffer, "aux_dropped")
            closest_full = read_aux_full(
                actor.experience_buffer, "aux_closest_d_hat"
            )
            success_tol_full = read_aux_full(
                actor.experience_buffer, "aux_success_tolerance"
            )
            try:
                returns_for_sample = returns_full
            except NameError:
                returns_for_sample = None
            sample_batch = seq_wm_trainer.sampler.sample(
                obs_full=obs_full,
                action_full=action_full,
                reward_full=reward_full,
                done_full=done_full,
                returns_full=returns_for_sample,
                last_obs=last_obs.to(obs_full.dtype),
                d_hat_full=d_hat_full,
                last_d_hat=last_d_hat,
                failure_full=failure_full,
                success_count_full=success_count_full,
                last_success_count=last_success_count,
                lifted_full=lifted_full,
                picked_full=picked_full,
                dropped_full=dropped_full,
                closest_d_hat_full=closest_full,
                success_tolerance_full=success_tol_full,
            )
            sample_batch = sample_batch.to(seq_wm_trainer.device)
    except Exception as exc:
        out["sample_failed"] = 1.0
        out["invalid_reason_hash"] = float(hash("sample_failed") % 1_000_003)
        out["imagine_invalid_reason"] = 0.0
        seq_wm_trainer._last_imagine_invalid_reason = "sample_failed"
        return out

    # ----- Eval-buffer recording (record any tight-tol batch, gated) -----
    recorded = False
    if teacher_enabled and eval_buf is not None:
        try:
            recorded = eval_buf.record_candidate(
                sample_batch,
                floor_tol_threshold=record_below_tol,
                hard_floor_tol_threshold=float(
                    getattr(seq_wm_trainer.cfg, "floor_tol_floor", 0.012)
                ),
                require_floor_signal=True,
            )
        except Exception:
            recorded = False
    out["teacher_eval_buffer_size"] = float(len(eval_buf) if eval_buf else 0)
    out["teacher_recorded_now"] = float(1.0 if recorded else 0.0)

    # ----- Teacher snapshot + eval + maybe_promote -----
    if teacher_enabled and teacher.should_eval(frames):
        out["teacher_eval_attempted"] = 1.0
        seq_wm_trainer._teacher_eval_attempted_total += 1
        try:
            snap_path = teacher.maybe_save_candidate_snapshot(
                env_frames=frames,
                trainer_state_dict=seq_wm_trainer.state_dict(),
            )
            if eval_buf is not None and len(eval_buf) > 0:
                metrics_obj = teacher.evaluate(
                    seq_wm_trainer.model,
                    eval_buf,
                    device=seq_wm_trainer.device,
                    floor_tol_threshold=float(
                        getattr(seq_wm_trainer.cfg, "floor_tol_threshold", 0.02)
                    ),
                    floor_tol_floor=float(
                        getattr(seq_wm_trainer.cfg, "floor_tol_floor", 0.012)
                    ),
                )
                promoted, reason = teacher.maybe_promote(
                    candidate_path=snap_path
                    if snap_path is not None
                    else (teacher.teacher_dir / "wm_teacher_inline.pt"),
                    candidate_metrics=metrics_obj,
                    env_frames=frames,
                    global_step=int(seq_wm_trainer.global_step),
                )
                out["teacher_promoted"] = 1.0 if promoted else 0.0
                out["teacher_promotion_count"] = float(teacher.promotions)
                out["teacher_rejection_count"] = float(teacher.rejections)
                out["teacher_eval_attempt_count"] = float(teacher.eval_attempts)
                out["teacher_eval_score"] = float(metrics_obj.score)
                out["teacher_eval_recovery_pos"] = float(
                    metrics_obj.recovery_pos_rate
                )
                # Phase A4 P5 (2026-06-10): per-head pos rates and the
                # gate-decision improvement so TB shows the live state.
                out["teacher_horizon_pos_rate_h4"] = float(
                    metrics_obj.horizon_success_pos_rate_h4
                )
                out["teacher_horizon_pos_rate_h8"] = float(
                    metrics_obj.horizon_success_pos_rate_h8
                )
                out["teacher_drop_pos_rate_h4"] = float(
                    metrics_obj.drop_risk_pos_rate_h4
                )
                out["teacher_drop_pos_rate_h8"] = float(
                    metrics_obj.drop_risk_pos_rate_h8
                )
                out["teacher_eval_buffer_size"] = float(len(eval_buf))
                out["teacher_n_samples"] = float(metrics_obj.n_samples)
                out["teacher_uncertainty"] = float(metrics_obj.uncertainty)
                out["teacher_improvement"] = float(teacher.last_improvement)
                out["teacher_first_promotion"] = float(
                    1.0 if (promoted and reason == "first_teacher") else 0.0
                )
                out["teacher_target_degenerate"] = float(
                    1.0 if teacher.last_target_degenerate else 0.0
                )
                out["teacher_rejected_reason_hash"] = float(
                    hash(reason) % 1_000_003
                )
                out["teacher_no_eval_data"] = 0.0
                if metrics_obj.has_nan:
                    seq_wm_trainer._teacher_target_degenerate_total += 1
                if promoted:
                    seq_wm_trainer._teacher_promoted_total += 1
                if teacher.last_target_degenerate:
                    seq_wm_trainer._teacher_target_degenerate_total += 1
            else:
                # Phase A4 P5 (2026-06-10): record_no_eval_data() persists
                # the attempt to index.json + events.jsonl so the analyzer
                # can show every eval attempt — even those that were skipped
                # because the buffer was empty.
                buf_size = int(len(eval_buf) if eval_buf is not None else 0)
                teacher.record_no_eval_data(
                    env_frames=frames, eval_buffer_size=buf_size
                )
                out["teacher_no_eval_data"] = 1.0
                out["teacher_eval_attempt_count"] = float(teacher.eval_attempts)
                out["teacher_promotion_count"] = float(teacher.promotions)
                out["teacher_rejection_count"] = float(teacher.rejections)
                out["teacher_eval_buffer_size"] = float(buf_size)
                out["teacher_rejected_reason_hash"] = float(
                    hash("no_eval_data") % 1_000_003
                )
                seq_wm_trainer._teacher_no_eval_data_total += 1
        except Exception as exc:
            out["teacher_error"] = 1.0
            out["teacher_rejected_reason_hash"] = float(
                hash(f"err:{type(exc).__name__}") % 1_000_003
            )
        out["teacher_promoted_total"] = float(
            seq_wm_trainer._teacher_promoted_total
        )
        out["teacher_eval_attempted_total"] = float(
            seq_wm_trainer._teacher_eval_attempted_total
        )
        out["teacher_no_eval_data_total"] = float(
            seq_wm_trainer._teacher_no_eval_data_total
        )

    # Always emit current teacher step (-1 if no teacher yet). Phase A4
    # P5 also exposes ``teacher_current_teacher_step`` as canonical TB key.
    if teacher_enabled and teacher.current_teacher() is not None:
        cur_step = float(teacher.current_teacher().frame)
    else:
        cur_step = -1.0
    out["teacher_current_step"] = cur_step
    out["teacher_current_teacher_step"] = cur_step

    # ----- Imagined evaluator -----
    if imagine_enabled:
        try:
            imag_out = imag.evaluate_batch(
                seq_wm_trainer.model,
                sample_batch,
                device=seq_wm_trainer.device,
            )
            valid = bool(imag_out.get("imagined_valid", False))
            if valid:
                seq_wm_trainer._imagine_valid_total += 1
            else:
                seq_wm_trainer._imagine_invalid_total += 1
                seq_wm_trainer._last_imagine_invalid_reason = str(
                    imag_out.get("invalid_reason", "high_uncertainty")
                )
            out["imagine_valid"] = 1.0 if valid else 0.0
            out["imagine_valid_total"] = float(
                seq_wm_trainer._imagine_valid_total
            )
            out["imagine_invalid_total"] = float(
                seq_wm_trainer._imagine_invalid_total
            )
            for k in (
                "wm_recovery_prob",
                "wm_success_prob",
                "wm_drop_risk",
                "uncertainty",
                "recovery_h16",
                "value_h16",
                "logit_abs_mean",
                "head_signal_std",
                "imagined_valid_strict",
                "imagined_valid_soft",
                "imagined_head_has_signal",
                "wm_validity_weight",
                "imagined_valid_mode",
                "imagined_valid_reason",
                "checkpoint_age_frames",
            ):
                if k in imag_out and isinstance(imag_out[k], (int, float, bool)):
                    out[f"imagine_{k}"] = float(imag_out[k])
            for k, v in imag_out.items():
                if k.startswith("success_h") or k.startswith("drop_h") or k.startswith("recovery_h"):
                    if isinstance(v, (int, float)):
                        out[f"imagine_{k}"] = float(v)
            invalid_reason = "valid" if valid else (
                imag_out.get("imagined_valid_reason")
                or seq_wm_trainer._last_imagine_invalid_reason
                or "high_uncertainty"
            )
            out["imagine_invalid_reason_hash"] = float(
                hash(invalid_reason) % 1_000_003
            )
            # ----- Publish to disk for cross-process observers -----
            wm_imagine_dir = getattr(seq_wm_trainer, "_wm_imagine_dir", None)
            policy_idx_for_file = (
                int(teacher.policy_idx)
                if (teacher is not None and teacher.policy_idx is not None)
                else 0
            )
            if wm_imagine_dir is not None:
                shadow_block_decision = False
                shadow_block_reason = ""
                unc_val = float(imag_out.get("uncertainty", 1.0))
                if getattr(args_cli, "wm_imagined_shadow_block_enable", False):
                    thr = float(
                        getattr(
                            args_cli,
                            "wm_imagined_shadow_uncertainty_threshold",
                            0.95,
                        )
                    )
                    if unc_val >= thr:
                        shadow_block_decision = True
                        shadow_block_reason = "shadow_high_uncertainty"
                    elif not valid:
                        shadow_block_decision = True
                        shadow_block_reason = "shadow_imagined_invalid"
                payload = {
                    "policy_idx": int(policy_idx_for_file),
                    "frames": int(frames),
                    "global_step": int(seq_wm_trainer.global_step),
                    "imagined_valid": bool(valid),
                    "imagined_valid_strict": bool(
                        imag_out.get("imagined_valid_strict", valid)
                    ),
                    "imagined_valid_soft": bool(
                        imag_out.get("imagined_valid_soft", False)
                    ),
                    "imagined_head_has_signal": bool(
                        imag_out.get("imagined_head_has_signal", False)
                    ),
                    "imagined_valid_mode": str(
                        imag_out.get(
                            "imagined_valid_mode",
                            getattr(imag, "valid_mode", "strict_uncertainty"),
                        )
                    ),
                    "valid_reason": str(
                        imag_out.get("imagined_valid_reason", invalid_reason)
                    ),
                    "invalid_reason": invalid_reason,
                    "wm_validity_weight": float(
                        imag_out.get("wm_validity_weight", 0.0)
                    ),
                    "uncertainty": unc_val,
                    "wm_recovery_prob": float(
                        imag_out.get("wm_recovery_prob", 0.0)
                    ),
                    "wm_success_prob": float(
                        imag_out.get("wm_success_prob", 0.0)
                    ),
                    "wm_drop_risk": float(imag_out.get("wm_drop_risk", 0.0)),
                    "logit_abs_mean": float(
                        imag_out.get("logit_abs_mean", 0.0)
                    ),
                    "head_signal_std": float(
                        imag_out.get("head_signal_std", 0.0)
                    ),
                    "checkpoint_age_frames": float(
                        imag_out.get("checkpoint_age_frames", 0.0)
                    ),
                    "stale_reason": str(
                        imag_out.get("stale_reason", "live_model")
                    ),
                    "shadow_block_decision": bool(shadow_block_decision),
                    "shadow_block_reason": str(shadow_block_reason),
                    "teacher_step": float(
                        teacher.current_teacher().frame
                        if (teacher is not None and teacher.current_teacher() is not None)
                        else -1.0
                    ),
                    "horizon_pos_rate": {
                        k.replace("imagine_", ""): float(v)
                        for k, v in out.items()
                        if k.startswith("imagine_success_h")
                        or k.startswith("imagine_drop_h")
                        or k.startswith("imagine_recovery_h")
                    },
                    "eval_num_starts": int(
                        getattr(imag, "num_starts", 0)
                    ),
                    "writer_pid": int(os.getpid()),
                    "writer_ts": float(time.time()),
                }
                _atomic_write_json(
                    wm_imagine_dir / f"{int(policy_idx_for_file):03d}.json",
                    payload,
                )
                out["imagine_published"] = 1.0
        except NotImplementedError:
            # Per user constraint #4 the policy mode must fail fast: re-raise.
            raise
        except Exception as exc:
            seq_wm_trainer._imagine_invalid_total += 1
            seq_wm_trainer._last_imagine_invalid_reason = (
                f"err:{type(exc).__name__}"
            )
            out["imagine_valid"] = 0.0
            out["imagine_invalid_reason_hash"] = float(
                hash(seq_wm_trainer._last_imagine_invalid_reason) % 1_000_003
            )
    return out


def _patch_actor_train_for_seq_wm(
    actor,
    seq_wm_trainer: SeqWMTrainer,
    seq_aux_collector: SeqAuxCollector,
    seq_wm_save_dir,
    save_every_epochs: int = 50,
    seq_aux_env_wrap: "SeqAuxInterceptorEnv | None" = None,
    seq_distill_bridge: "SeqDistillBridge | None" = None,
    awr_bridge=None,
    awr_env_wrap=None,
):
    """Wire SeqWMTrainer into the rl_games agent.

    Patches (the experience buffer doesn't yet exist when this is called —
    rl_games allocates it inside ``init_tensors()`` which runs at the top
    of ``agent.train()``. We therefore wrap ``init_tensors`` to register
    the aux fields right after the buffer is built):

      1. agent.init_tensors → wraps to register aux_d_hat / aux_failure /
         aux_lifted / aux_success_cnt fields on the freshly-created
         experience_buffer.
      2. agent.play_steps_rnn → wraps to (a) reset_window at rollout start,
         (b) capture batch_dict for GAE returns, (c) call
         ``record_terminal()`` after the rollout returns so ``last_d_hat``
         / ``last_success_count`` are step-H snapshots (Phase A4 P0
         Task C — terminal aux alignment fix).
      3. agent.train_epoch → wraps to (a) copy collector window into the
         buffer's aux fields, (b) build the seq distill teacher cache
         on the FULL (T, N) rollout (when coupling != none), (c) call
         seq_wm_trainer.step() for the seq WM's own training, and
         (d) periodically save a SeqWM checkpoint.

    The legacy WMTrainer paths (model.get_aux_loss patch, _patch_actor_train_for_wm)
    are NOT applied for sequence strategies — see _WmEnabledRunner.run_train.
    """
    # 1) Defer aux-field registration to right after init_tensors().
    original_init_tensors = actor.init_tensors

    def patched_init_tensors(*args, **kwargs):
        original_init_tensors(*args, **kwargs)
        register_seq_aux_fields(
            actor.experience_buffer, fields=seq_aux_collector.fields
        )
        # Re-bind play_steps_rnn here — init_tensors does not rebind it,
        # but defensive: ensure our patched version stays installed even
        # after any rl_games hooks reset it.
        actor.play_steps_rnn = patched_play_steps_rnn

    # 2) Capture latest batch_dict from play_steps_rnn so train_epoch tail
    #    can read GAE returns.
    capture: dict = {"batch_dict": None}
    # NOTE: we must read the live play_steps_rnn at first invocation because
    # ContinuousA2CBase.play_steps_rnn is a method on the class; binding
    # before init_tensors is fine.
    original_play_steps_rnn = actor.play_steps_rnn

    # Phase A4 P0 metrics: count rollouts where record_terminal() actually
    # fired and the magnitude of the last_d_hat / last_success_count
    # corrections. These confirm the off-by-one fix is live.
    terminal_metrics: dict[str, float] = {
        "terminal_recorded": 0.0,
        "last_d_hat_delta_mean": 0.0,
        "last_success_delta_mean": 0.0,
    }
    # Bridge bookkeeping: rollout generation increments on every
    # play_steps_rnn return so the teacher cache can be invalidated cleanly.
    rollout_state = {"generation": 0}

    def _record_terminal_and_copy_aux() -> tuple[
        "torch.Tensor | None", "torch.Tensor | None"
    ]:
        """Run after rollout, before PPO consumes the data.

        1. ``record_terminal`` so ``last_d_hat`` / ``last_success_count``
           reflect step H rather than step H-1 (Phase A4 P0 Task C).
        2. Copy collector.full_window() into experience_buffer aux fields
           so the seq WM trainer (and any future consumer) can read them.

        Returns ``(last_d_hat, last_success_count)`` as 1-D tensors or
        None if the field was not produced by the env.
        """
        if seq_aux_env_wrap is not None and seq_aux_env_wrap.last_infos is not None:
            pre = seq_aux_collector.last_step()
            d_hat_pre = pre.get("aux_d_hat")
            succ_pre = pre.get("aux_success_cnt")
            d_hat_pre = d_hat_pre.clone() if d_hat_pre is not None else None
            succ_pre = succ_pre.clone() if succ_pre is not None else None
            seq_aux_collector.record_terminal(
                infos=seq_aux_env_wrap.last_infos,
                env_attrs=seq_aux_env_wrap.last_env_attrs,
            )
            post = seq_aux_collector.last_step()
            d_hat_post = post.get("aux_d_hat")
            succ_post = post.get("aux_success_cnt")
            if d_hat_pre is not None and d_hat_post is not None:
                terminal_metrics["last_d_hat_delta_mean"] = float(
                    (d_hat_post - d_hat_pre).abs().mean().item()
                )
            if succ_pre is not None and succ_post is not None:
                terminal_metrics["last_success_delta_mean"] = float(
                    (succ_post - succ_pre).abs().mean().item()
                )
            terminal_metrics["terminal_recorded"] = 1.0
        else:
            terminal_metrics["terminal_recorded"] = 0.0

        # Copy collector full window into the experience buffer aux fields
        # so seq WM trainer + teacher cache see the same numbers.
        full_aux = seq_aux_collector.full_window()
        for name, tensor in full_aux.items():
            store = actor.experience_buffer.tensor_dict.get(name)
            if store is None:
                continue
            store.copy_(tensor)

        last_aux = seq_aux_collector.last_step()
        last_d_hat = last_aux.get("aux_d_hat")
        last_success_count = last_aux.get("aux_success_cnt")
        if last_d_hat is not None:
            last_d_hat = last_d_hat.squeeze(-1)
        if last_success_count is not None:
            last_success_count = last_success_count.squeeze(-1)
        return last_d_hat, last_success_count

    def patched_play_steps_rnn(*args, **kwargs):
        seq_aux_collector.reset_window()
        if awr_env_wrap is not None:
            awr_env_wrap.reset_rollout_window()
        bd = original_play_steps_rnn(*args, **kwargs)
        # Phase A4 P0 Task C — terminal aux alignment + aux copy.
        # Without this, collector._last == storage[H-1] (off-by-one) and
        # next_d_hat[T-1] is biased by one step. We also copy full_aux
        # into experience_buffer here (instead of at the end of train_epoch)
        # so the upcoming PPO minibatch loop sees up-to-date aux fields
        # in case any consumer of get_aux_loss reads them.
        last_d_hat_t, last_success_t = _record_terminal_and_copy_aux()
        capture["batch_dict"] = bd
        capture["last_d_hat"] = last_d_hat_t
        capture["last_success_count"] = last_success_t
        rollout_state["generation"] += 1

        if awr_bridge is not None:
            awr_cfg = awr_bridge.cfg
            frames = int(getattr(actor, "frame", 0) or 0)
            awr_bridge.stats["awr/collect_after_frames"] = float(
                getattr(awr_cfg, "collect_after_frames", 0)
            )
            awr_bridge.stats["awr/start_after_frames"] = float(awr_cfg.start_after_frames)

            if awr_bridge.disabled:
                awr_bridge.stats["awr/skipped_disabled"] = 1.0
            elif frames < int(getattr(awr_cfg, "collect_after_frames", 0)):
                awr_bridge.stats["awr/skipped_collect_not_started"] = 1.0
            elif str(getattr(awr_cfg, "collection_mode", "rollout_buffer")) == "rollout_buffer":
                from dexpbt_isaaclab.awr.elite_awr_rollout_builder import (
                    build_awr_entries_from_experience_buffer,
                    check_rollout_buffer_action_sanity,
                    should_collect_awr,
                )

                if should_collect_awr(frames, awr_cfg, awr_bridge):
                    from dexpbt_isaaclab.awr.elite_awr import (
                        awr_apply_builder_run,
                        awr_builder_collection_paused,
                        awr_record_builder_fuse_skip,
                        awr_should_run_builder_this_rollout,
                    )

                    if not awr_should_run_builder_this_rollout(awr_bridge, awr_cfg):
                        pass
                    elif awr_builder_collection_paused(awr_bridge, frames):
                        awr_record_builder_fuse_skip(awr_bridge, frames)
                    else:
                        td = actor.experience_buffer.tensor_dict
                        if not getattr(awr_bridge, "_rollout_action_sanity_checked", False):
                            check_rollout_buffer_action_sanity(awr_bridge, td)
                            awr_bridge._rollout_action_sanity_checked = True
                        meta_provider = getattr(awr_bridge, "source_meta_provider", None)
                        meta = dict(meta_provider() or {}) if callable(meta_provider) else {}
                        source_nsucc = meta.get("source_nsucc")
                        if isinstance(source_nsucc, torch.Tensor):
                            source_nsucc = float(source_nsucc.mean().item())
                        builder_t0 = time.perf_counter()
                        entries, builder_stats = build_awr_entries_from_experience_buffer(
                            experience_buffer=actor.experience_buffer,
                            cfg=awr_cfg,
                            source_policy=int(
                                meta.get("source_policy", getattr(awr_bridge, "policy_idx", 0))
                            ),
                            source_profile=str(
                                meta.get("source_profile")
                                or getattr(awr_bridge, "profile_name", "")
                            ),
                            source_nsucc=source_nsucc,
                            source_retention=meta.get("source_retention"),
                            env_frames=frames,
                            source_type=str(meta.get("source_type") or "local"),
                        )
                        builder_time_ms = (time.perf_counter() - builder_t0) * 1000.0
                        for entry in entries:
                            awr_bridge.buffer.insert(entry)
                        awr_apply_builder_run(
                            awr_bridge, frames, builder_time_ms, builder_stats
                        )
            elif awr_env_wrap is not None:
                if not getattr(awr_bridge, "_action_checked", False):
                    try:
                        captured = awr_env_wrap.consume_rollout_actions()
                        ppo_actions = actor.experience_buffer.tensor_dict.get("actions")
                        if captured is None or ppo_actions is None:
                            awr_bridge.record_action_convention_unavailable(
                                "missing_capture_or_ppo_actions"
                            )
                        else:
                            awr_bridge.maybe_check_action_convention(
                                captured, ppo_actions
                            )
                    except Exception as _awr_act_exc:
                        print(f"[AWR] action convention check failed: {_awr_act_exc}")
                        awr_bridge.record_action_convention_unavailable(
                            f"exception:{type(_awr_act_exc).__name__}"
                        )

        # FloorCritic: build the per-rollout (T,N) target cache NOW (after the
        # aux copy, before PPO's minibatch loop / get_aux_loss). Reset the
        # feature capture so a stale tensor cannot be reused. No-op when the
        # bridge / target builder is absent (FloorCritic disabled).
        fc_bridge = getattr(seq_wm_trainer, "_floorcritic_bridge", None)
        fc_builder = getattr(seq_wm_trainer, "_floorcritic_target_builder", None)
        if fc_bridge is not None and fc_builder is not None:
            fc_bridge.capture.reset()
            try:
                td = actor.experience_buffer.tensor_dict
                fc_cache = fc_builder.build(
                    d_hat=td["aux_d_hat"],
                    success_cnt=td["aux_success_cnt"],
                    success_tol=td["aux_success_tolerance"],
                    failure=td.get("aux_failure"),
                    lifted=td.get("aux_lifted"),
                    dones=td["dones"],
                    generation=rollout_state["generation"],
                )
                fc_bridge.cache = fc_cache
            except Exception as _fc_exc:
                print(f"[FloorCritic] target cache build failed: {_fc_exc}")
                fc_bridge.cache = None

        erd_bridge = getattr(seq_wm_trainer, "_erd_bridge", None)
        erd_builder = getattr(seq_wm_trainer, "_erd_target_builder", None)
        if erd_bridge is not None and erd_builder is not None:
            erd_bridge.capture.reset()
            try:
                td = actor.experience_buffer.tensor_dict
                erd_cache = erd_builder.build(
                    d_hat=td["aux_d_hat"],
                    success_cnt=td["aux_success_cnt"],
                    success_tol=td["aux_success_tolerance"],
                    failure=td.get("aux_failure"),
                    lifted=td.get("aux_lifted"),
                    dones=td["dones"],
                    actions=td["actions"],
                    generation=rollout_state["generation"],
                )
                erd_bridge.cache = erd_cache
            except Exception as _erd_exc:
                print(f"[ERD] target cache build failed: {_erd_exc}")
                erd_bridge.cache = None

        eir_bridge = getattr(seq_wm_trainer, "_eir_bridge", None)
        eir_builder = getattr(seq_wm_trainer, "_eir_target_builder", None)
        if eir_bridge is not None and eir_builder is not None:
            if eir_bridge.capture is not None:
                eir_bridge.capture.reset()
            try:
                td = actor.experience_buffer.tensor_dict
                eir_cache = eir_builder.build(
                    d_hat=td["aux_d_hat"],
                    success_cnt=td["aux_success_cnt"],
                    success_tol=td["aux_success_tolerance"],
                    failure=td.get("aux_failure"),
                    lifted=td.get("aux_lifted"),
                    dones=td["dones"],
                    actions=td["actions"],
                    generation=rollout_state["generation"],
                )
                eir_bridge.cache = eir_cache
                feat_flat, fsrc = None, 0
                if eir_bridge.bank is not None or eir_bridge.snippet_bank is not None:
                    feat_flat, fsrc = _eir_extract_rollout_features(
                        actor.model, td, eir_bridge.eir_cfg.feature_dim, actor=actor
                    )
                src_stats = {
                    "source_policy_idx": float(getattr(args_cli, "policy_idx", -1)),
                    "source_nsucc_mean": float(
                        getattr(seq_wm_trainer, "_eir_source_nsucc_mean", 0.0)
                    ),
                    "source_nsucc_max": float(
                        getattr(seq_wm_trainer, "_eir_source_nsucc_max", 0.0)
                    ),
                }
                if eir_bridge.bank is not None and feat_flat is not None:
                    eir_bridge.bank.update_from_cache(
                        eir_cache, feat_flat, source_stats=src_stats,
                        frame=int(getattr(actor, "frame", 0) or 0),
                    )
                    eir_bridge.record_bank_feature_source(int(fsrc))
                    eir_bridge.last_bank_metrics = eir_bridge.bank.metrics()
                if eir_bridge.snippet_bank is not None and feat_flat is not None:
                    snippet_builder = getattr(seq_wm_trainer, "_snippet_target_builder", None)
                    if snippet_builder is not None:
                        snippet_batch = snippet_builder.build(eir_cache, feat_flat)
                        eir_bridge.snippet_bank.update_from_builder_batch(
                            snippet_batch,
                            source_stats=src_stats,
                            frame=int(getattr(actor, "frame", 0) or 0),
                        )
                # (19.3) refresh per-step aux snapshot for rollout apply: use the
                # latest window step per env (approx current state at next play).
                res_bridge = eir_bridge.residual_bridge
                if res_bridge is not None and res_bridge.cfg.enabled:
                    try:
                        dh = td["aux_d_hat"].float()
                        tl = td["aux_success_tolerance"].float()
                        lf = td.get("aux_lifted")
                        # shape [T, N, (1)] -> last step [N]
                        last_dh = dh[-1].reshape(-1)
                        last_tl = tl[-1].reshape(-1)
                        last_lf = lf[-1].reshape(-1).float() if lf is not None else None
                        res_bridge.set_aux_snapshot(
                            d_hat=last_dh, tol=last_tl, lifted=last_lf,
                            frame=int(getattr(actor, "frame", 0) or 0),
                            generation=int(rollout_state.get("generation", -1)),
                        )
                    except Exception:
                        res_bridge.set_aux_snapshot(d_hat=None, tol=None, lifted=None, frame=0)
            except Exception as _eir_exc:
                print(f"[EIR] target cache / bank update failed: {_eir_exc}")
                eir_bridge.cache = None

            # Full H-step MPPI: build the per-rollout RSSM belief cache ONCE
            # (deterministic full-rollout encode). compute_eir_loss slices it
            # per minibatch; we never re-encode the RSSM per minibatch. Cleanly
            # invalidated to a fallback cache on any failure / non-RSSM backbone.
            _need_belief = (
                getattr(eir_bridge, "rssm_planner", None) is not None
                or getattr(eir_bridge, "rssm_reranker", None) is not None
            )
            if _need_belief:
                try:
                    T_b = int(actor.horizon_length)
                    N_b = int(actor.num_actors * actor.num_agents)
                    _returns_flat = bd.get("returns")
                    _returns_full = None
                    if _returns_flat is not None:
                        _vs = _returns_flat.shape[-1] if _returns_flat.ndim == 2 else 1
                        try:
                            _returns_full = (
                                _returns_flat.view(N_b, T_b, _vs).transpose(0, 1).contiguous()
                            )
                        except Exception:
                            _returns_full = _returns_flat.view(T_b, N_b, _vs)
                        if _returns_full.shape[-1] == 1:
                            _returns_full = _returns_full.squeeze(-1)
                    _last_obs = actor.obs["obs"] if isinstance(actor.obs, dict) else actor.obs
                    # ensemble member beliefs only when the bridge actually uses
                    # the ensemble MPPI path (rssm_models set during wiring).
                    _ens_models_for_cache = None
                    _ens_mgr = getattr(seq_wm_trainer, "_rssm_ensemble_manager", None)
                    if (
                        getattr(eir_bridge, "rssm_models", None) is not None
                        and _ens_mgr is not None
                        and _ens_mgr.size > 1
                    ):
                        _ens_models_for_cache = _ens_mgr.models
                    eir_bridge.rssm_belief_cache = build_rssm_belief_cache(
                        seq_wm_trainer=seq_wm_trainer,
                        experience_buffer=actor.experience_buffer,
                        returns_full=_returns_full,
                        last_obs=_last_obs,
                        last_d_hat=last_d_hat_t,
                        last_success_count=last_success_t,
                        frames=int(getattr(actor, "frame", 0) or 0),
                        generation=int(rollout_state.get("generation", -1)),
                        device=seq_wm_trainer.device,
                        enabled=True,
                        ensemble_models=_ens_models_for_cache,
                    )
                except Exception as _bc_exc:
                    print(f"[EIR-belief] cache build failed: {_bc_exc}")
                    eir_bridge.rssm_belief_cache = None

        if seq_distill_bridge is not None:
            seq_distill_bridge.rollout_generation = rollout_state["generation"]
            seq_distill_bridge.feature_capture.reset()
            # Build the teacher cache NOW from the freshly-collected rollout
            # so PPO's minibatch loop (which consumes get_aux_loss) sees a
            # cache whose generation matches the live rollout. If coupling
            # is disabled we skip the work entirely.
            if seq_distill_bridge.cfg.enabled:
                try:
                    T = int(actor.horizon_length)
                    N = int(actor.num_actors * actor.num_agents)
                    returns_flat = bd.get("returns")
                    if returns_flat is not None:
                        value_size = (
                            returns_flat.shape[-1] if returns_flat.ndim == 2 else 1
                        )
                        try:
                            returns_full = (
                                returns_flat
                                .view(N, T, value_size)
                                .transpose(0, 1)
                                .contiguous()
                            )
                        except Exception:
                            returns_full = returns_flat.view(T, N, value_size)
                        last_obs = (
                            actor.obs["obs"] if isinstance(actor.obs, dict)
                            else actor.obs
                        )
                        bridge_cfg = seq_distill_bridge.cfg
                        compute_corr = bridge_cfg.gate != "none"
                        cache = seq_wm_trainer.compute_teacher_cache(
                            experience_buffer=actor.experience_buffer,
                            last_obs=last_obs,
                            returns_full=(
                                returns_full.squeeze(-1)
                                if returns_full.shape[-1] == 1 else returns_full
                            ),
                            last_d_hat=last_d_hat_t,
                            last_success_count=last_success_t,
                            generation=rollout_state["generation"],
                            teacher_value_norm=bridge_cfg.teacher_value_norm,
                            teacher_value_norm_min_std=(
                                bridge_cfg.teacher_value_norm_min_std
                            ),
                            compute_teacher_return_corr=compute_corr,
                        )
                        seq_distill_bridge.cache = cache
                        seq_distill_bridge.last_frames = int(
                            getattr(actor, "frame", 0) or 0
                        )
                except Exception as exc:
                    print(f"[SeqDistill] teacher cache build failed: {exc}")
        return bd

    actor.init_tensors = patched_init_tensors
    actor.play_steps_rnn = patched_play_steps_rnn

    # 3) Patch train_epoch to (a) copy collector window into the buffer's
    #    aux fields, (b) build the seq distill teacher cache on the full
    #    (T, N) rollout (only when coupling enabled), (c) call
    #    SeqWMTrainer.step() with returns from capture["batch_dict"],
    #    (d) periodically save a SeqWM checkpoint.
    original_train_epoch = actor.train_epoch
    seq_wm_trainer._epochs_since_save = 0
    seq_wm_trainer._first_distill_assert_done = False

    def patched_train_epoch(*args, **kwargs):
        # rl_games train_epoch flow:
        #   play_steps_rnn() -> prepare_dataset() -> minibatch loop ->
        #   (get_aux_loss is invoked here per minibatch)
        # We installed terminal-aux + teacher-cache work inside
        # patched_play_steps_rnn so the cache is ready BEFORE the
        # minibatch loop. This tail just trains the seq WM on the
        # collected rollout and emits TB metrics.
        result = original_train_epoch(*args, **kwargs)
        try:
            bd = capture["batch_dict"]
            if bd is None:
                return result
            T = int(actor.horizon_length)
            N = int(actor.num_actors * actor.num_agents)
            returns_flat = bd.get("returns")
            if returns_flat is None:
                return result
            value_size = (
                returns_flat.shape[-1] if returns_flat.ndim == 2 else 1
            )
            try:
                returns_full = returns_flat.view(N, T, value_size).transpose(0, 1).contiguous()
            except Exception:
                returns_full = returns_flat.view(T, N, value_size)

            obs_full = actor.experience_buffer.tensor_dict["obses"]
            last_obs = actor.obs["obs"] if isinstance(actor.obs, dict) else actor.obs
            last_obs_t = (
                last_obs if hasattr(last_obs, "shape")
                else torch.zeros(N, obs_full.shape[-1], device=obs_full.device)
            )

            last_d_hat = capture.get("last_d_hat")
            last_success_count = capture.get("last_success_count")

            frames = int(getattr(actor, "frame", 0) or 0)
            _returns_for_step = (
                returns_full.squeeze(-1) if returns_full.shape[-1] == 1 else returns_full
            )
            metrics = seq_wm_trainer.step(
                experience_buffer=actor.experience_buffer,
                frames=frames,
                last_obs=last_obs_t,
                returns_full=_returns_for_step,
                last_d_hat=last_d_hat,
                last_success_count=last_success_count,
            )
            _eir_bridge_cal = getattr(seq_wm_trainer, "_eir_bridge", None)
            _wm_cal_ev = getattr(seq_wm_trainer, "_wm_calibration_evaluator", None)
            if _eir_bridge_cal is not None and _wm_cal_ev is not None:
                _cal_state = getattr(_eir_bridge_cal, "wm_calibration_state", None)
                if not isinstance(_cal_state, WMCalibrationState):
                    _cal_state = WMCalibrationState.from_dict(
                        _cal_state if isinstance(_cal_state, dict) else {}
                    )
                _cal_state = seq_wm_trainer.run_wm_calibration_eval(
                    experience_buffer=actor.experience_buffer,
                    frames=frames,
                    last_obs=last_obs_t,
                    returns_full=_returns_for_step,
                    evaluator=_wm_cal_ev,
                    state=_cal_state,
                    last_d_hat=last_d_hat,
                    last_success_count=last_success_count,
                )
                _eir_bridge_cal.wm_calibration_state = _cal_state.to_dict()

            # ---- ensemble member training (MPPI scoring ensemble only) ----
            # Members 1..E-1 train on the SAME rollout but independently sampled
            # sequence minibatches. Member 0 is the primary trainer stepped
            # above. No-op when the ensemble is disabled.
            _ens_trainers = getattr(seq_wm_trainer, "_rssm_ensemble_trainers", [seq_wm_trainer])
            _ens_member_metrics: list[dict] = []
            if len(_ens_trainers) > 1:
                for _em_i, _em_trainer in enumerate(_ens_trainers):
                    if _em_i == 0:
                        _ens_member_metrics.append(metrics)
                        continue
                    try:
                        _m = _em_trainer.step(
                            experience_buffer=actor.experience_buffer,
                            frames=frames,
                            last_obs=last_obs_t,
                            returns_full=_returns_for_step,
                            last_d_hat=last_d_hat,
                            last_success_count=last_success_count,
                        )
                        _ens_member_metrics.append(_m)
                    except Exception as _em_exc:
                        print(f"[RSSM-ensemble] member {_em_i} step failed: {_em_exc}")

            # ----------------------------------------------------------
            # Phase A4 P4 wired (2026-06-10): teacher eval + imagined eval.
            # No-op when both modules are disabled or wiring failed.
            # All work happens under torch.no_grad() and writes per-policy
            # JSON into <workspace>/wm_imagined/<policy>.json (atomic) for
            # cross-process consumption by PBT observers. Failures are
            # logged but never crash the training loop.
            # ----------------------------------------------------------
            wiring_metrics: dict[str, float] = _run_wm_runtime_wiring(
                actor=actor,
                seq_wm_trainer=seq_wm_trainer,
                frames=frames,
                last_obs=last_obs_t,
                returns_full=(
                    returns_full.squeeze(-1)
                    if returns_full.shape[-1] == 1 else returns_full
                ),
                last_d_hat=last_d_hat,
                last_success_count=last_success_count,
            )

            # TB logging.
            tb_writer = getattr(actor, "writer", None) or getattr(actor, "tensorboard_writer", None)
            if tb_writer is not None and metrics:
                for k, v in metrics.items():
                    if isinstance(v, (int, float)):
                        prefix = (
                            "wm_rssm/" if isinstance(seq_wm_trainer.model, RSSMLiteWorldModel)
                            else ("wm_trans/" if isinstance(seq_wm_trainer.model, TransformerLiteWorldModel)
                                  else "wm_seq/")
                        )
                        tag = (
                            f"wm_seq/{k}" if k.startswith("loss_")
                            else f"{prefix}{k}"
                        )
                        try:
                            tb_writer.add_scalar(tag, float(v), frames)
                        except Exception:
                            pass
            if tb_writer is not None:
                _restore_metrics = getattr(
                    seq_wm_trainer, "_rssm_ensemble_restore_metrics", None
                )
                if isinstance(_restore_metrics, dict):
                    for _rk, _rv in _restore_metrics.items():
                        if isinstance(_rv, (int, float)):
                            try:
                                tb_writer.add_scalar(
                                    f"seq_wm_ensemble/{_rk}", float(_rv), frames
                                )
                            except Exception:
                                pass
                    seq_wm_trainer._rssm_ensemble_restore_metrics = None
            if tb_writer is not None and len(_ens_trainers) > 1:
                # per-member + aggregate ensemble seq-WM training metrics.
                _loss_totals: list[float] = []
                for _em_i, _m in enumerate(_ens_member_metrics):
                    if not isinstance(_m, dict):
                        continue
                    for _kk in ("loss_total", "loss_kl", "loss_success", "loss_recovery"):
                        _vv = _m.get(_kk)
                        if isinstance(_vv, (int, float)):
                            try:
                                tb_writer.add_scalar(
                                    f"seq_wm_ensemble/member_{_em_i}/{_kk}",
                                    float(_vv), frames,
                                )
                            except Exception:
                                pass
                    if isinstance(_m.get("loss_total"), (int, float)):
                        _loss_totals.append(float(_m["loss_total"]))
                try:
                    import statistics as _stats
                    tb_writer.add_scalar("seq_wm_ensemble/size", float(len(_ens_trainers)), frames)
                    tb_writer.add_scalar("seq_wm_ensemble/train_active", 1.0, frames)
                    tb_writer.add_scalar("seq_wm_ensemble/bootstrap_mask_active", 0.0, frames)
                    if _loss_totals:
                        tb_writer.add_scalar(
                            "seq_wm_ensemble/loss_total_mean",
                            float(sum(_loss_totals) / len(_loss_totals)), frames,
                        )
                        tb_writer.add_scalar(
                            "seq_wm_ensemble/loss_total_std",
                            float(_stats.pstdev(_loss_totals)) if len(_loss_totals) > 1 else 0.0,
                            frames,
                        )
                except Exception:
                    pass
            if tb_writer is not None:
                for k, v in terminal_metrics.items():
                    try:
                        tb_writer.add_scalar(f"seq_aux/{k}", float(v), frames)
                    except Exception:
                        pass
                # Phase A4 P4 wired (2026-06-10): teacher + imagined eval
                # audit fields. Tags grouped under wm_teacher/* and
                # wm_imagine/*. All scalars are forward-safe to read from
                # downstream analyzers (analyze_wm_runtime_wiring.py).
                for k, v in wiring_metrics.items():
                    if not isinstance(v, (int, float)):
                        continue
                    if k.startswith("teacher_"):
                        tag = f"wm_teacher/{k[len('teacher_'):]}"
                    elif k.startswith("imagine_"):
                        tag = f"wm_imagine/{k[len('imagine_'):]}"
                    else:
                        tag = f"wm_runtime_wiring/{k}"
                    try:
                        tb_writer.add_scalar(tag, float(v), frames)
                    except Exception:
                        pass

                # FloorCritic per-rollout logging snapshot.
                fc_bridge_log = getattr(
                    seq_wm_trainer, "_floorcritic_bridge", None
                )
                if fc_bridge_log is not None:
                    fcb = fc_bridge_log
                    fc_scalars = {
                        "active": fcb.last_active,
                        "lambda": fcb.last_lambda,
                        "loss_total": fcb.last_loss_total,
                        "feature_source_id": float(fcb.cfg.feature_source_id),
                        "target_valid_fraction": fcb.last_valid_fraction,
                        "fc_grad_ratio_to_ppo_grad": fcb.last_fc_grad_ratio,
                        "auto_scale_factor": fcb.last_auto_scale_factor,
                        "shared_backbone_grad_from_fc": fcb.last_fc_feat_grad_norm,
                        "actor_head_grad_from_fc": 0.0,
                        "value_head_grad_from_fc": 0.0,
                        "actor_grad_blocked": 1.0 if fcb.last_actor_grad_blocked else 0.0,
                        "aux_head_grad_norm": fcb.last_aux_head_grad_norm,
                        "critic_grad_norm": fcb.last_fc_feat_grad_norm,
                    }
                    for k, v in fc_scalars.items():
                        try:
                            tb_writer.add_scalar(
                                f"wm_floorcritic/{k}", float(v), frames
                            )
                        except Exception:
                            pass
                    for key, lv in fcb.last_losses.items():
                        try:
                            tb_writer.add_scalar(
                                f"wm_floorcritic/loss_{key}", float(lv), frames
                            )
                        except Exception:
                            pass
                    for key, pr in fcb.last_pos_rate.items():
                        try:
                            tb_writer.add_scalar(
                                f"wm_floorcritic/pos_rate_{key}", float(pr), frames
                            )
                        except Exception:
                            pass

                erd_bridge_log = getattr(seq_wm_trainer, "_erd_bridge", None)
                if erd_bridge_log is not None:
                    eb = erd_bridge_log
                    erd_scalars = {
                        "active": eb.last_active,
                        "scorer_active": eb.last_scorer_active,
                        "distill_active": eb.last_distill_active,
                        "distill_block_reason": float(eb.last_distill_block_reason),
                        "risky_fraction": eb.last_risky_fraction,
                        "scorer_loss": eb.last_scorer_loss,
                        "distill_loss": eb.last_distill_loss,
                        "total_loss": eb.last_total_loss,
                        "best_score": eb.last_best_score,
                        "current_actor_score": eb.last_actor_score,
                        "score_margin": eb.last_score_margin,
                        "best_success_prob": eb.last_best_success_prob,
                        "best_improve_prob": eb.last_best_improve_prob,
                        "best_drop_prob": eb.last_best_drop_prob,
                        "actor_success_prob": eb.last_actor_success_prob,
                        "actor_drop_prob": eb.last_actor_drop_prob,
                        "action_delta_norm": eb.last_action_delta_norm,
                        "raw_mu_abs_mean": eb.last_raw_mu_abs_mean,
                        "raw_mu_clip_fraction": eb.last_raw_mu_clip_fraction,
                        "candidate_clip_fraction": eb.last_candidate_clip_fraction,
                        "distill_grad_ratio": eb.last_distill_grad_ratio,
                        "auto_scale_factor": eb.last_auto_scale_factor,
                        "target_valid_fraction": eb.last_target_valid_fraction,
                        "scorer_warmup_iters_done": float(eb.scorer_warmup_iters_done),
                        "candidate_k": float(eb.cfg.candidate_k),
                    }
                    for k, v in erd_scalars.items():
                        try:
                            tb_writer.add_scalar(f"erd/{k}", float(v), frames)
                        except Exception:
                            pass
                    for key, pr in eb.last_pos_rate.items():
                        try:
                            tb_writer.add_scalar(
                                f"erd/target_pos_rate_{key}", float(pr), frames
                            )
                        except Exception:
                            pass

                eir_bridge_log = getattr(seq_wm_trainer, "_eir_bridge", None)
                if eir_bridge_log is not None:
                    xb = eir_bridge_log
                    rb = xb.residual_bridge
                    rr = xb.rssm_reranker
                    eir_scalars = {
                        "active": xb.last_active,
                        "scorer_active": xb.last_scorer_active,
                        "residual_active": xb.last_residual_active,
                        "block_reason": float(xb.last_block_reason),
                        "scorer_loss": xb.last_scorer_loss,
                        "total_loss": xb.last_total_loss,
                        "risky_fraction": xb.last_risky_fraction,
                        "target_valid_fraction": xb.last_target_valid_fraction,
                        "score_margin": xb.last_score_margin,
                        "best_score": xb.last_best_score,
                        "current_actor_score": xb.last_actor_score,
                        "elite_similarity_mean": xb.last_elite_similarity_mean,
                        "elite_action_delta_norm": xb.last_elite_action_delta_norm,
                        "elite_bank_size": xb.last_elite_bank_size,
                        "bank_feature_source": float(xb.last_bank_feature_source),
                        "bank_feature_source_actor_fraction": xb.bank_feature_source_actor_fraction,
                        "bank_feature_source_fallback_fraction": xb.bank_feature_source_fallback_fraction,
                        "bank_feature_update_count": float(xb.bank_feature_update_count),
                        "scorer_warmup_iters_done": float(xb.scorer_warmup_iters_done),
                    }
                    for k, v in (xb.last_candidate_source_frac or {}).items():
                        eir_scalars[f"candidate_source_frac_{k}"] = float(v)
                    for k, v in (xb.last_best_source_frac or {}).items():
                        eir_scalars[f"best_source_frac_{k}"] = float(v)
                    for k, v in (xb.last_bank_metrics or {}).items():
                        eir_scalars[f"bank_{k}"] = float(v)
                    # elite bank insertion diagnostics (observation only): why is
                    # the bank empty? logged under both eir_bank/* and eir/bank_*.
                    _bank_diag = getattr(xb, "last_bank_update_diagnostics", None) or {}
                    for k, v in _bank_diag.items():
                        eir_scalars[f"bank_{k}"] = float(v)
                        try:
                            tb_writer.add_scalar(f"eir_bank/{k}", float(v), frames)
                        except Exception:
                            pass
                    if rb is not None:
                        eir_scalars.update({
                            "residual_loss": rb.last_residual_loss,
                            "residual_gate_mean": rb.last_gate_mean,
                            "residual_delta_norm": rb.last_delta_norm,
                            "residual_action_delta_norm": rb.last_residual_action_delta_norm,
                            "residual_score_margin": rb.last_residual_score_margin,
                            "residual_best_source_frac_elite": rb.last_residual_best_source_frac_elite,
                            "residual_apply_active": rb.last_apply_active,
                            "residual_apply_fraction": rb.last_apply_fraction,
                            "residual_apply_alpha": rb.last_apply_alpha,
                            "residual_gate_mean_on_rollout": rb.last_gate_mean_on_rollout,
                            "residual_apply_fallback_reason": float(rb.last_apply_fallback_reason),
                        })
                        # residual-apply funnel diagnostics (observation only):
                        # which gate drops the batch before apply.
                        _res_diag = getattr(rb, "last_apply_diagnostics", None) or {}
                        for k, v in _res_diag.items():
                            eir_scalars[k] = float(v)
                    eir_scalars.update({
                        "actor_distill_active": xb.last_actor_distill_active,
                        "actor_distill_active_fraction": xb.last_actor_distill_active_fraction,
                        "actor_distill_loss": xb.last_actor_distill_loss,
                        "actor_distill_weighted_loss": xb.last_actor_distill_weighted_loss,
                        "actor_distill_total": xb.last_actor_distill_total,
                        "actor_distill_loss_cap": xb.last_actor_distill_loss_cap,
                        "actor_distill_loss_cap_hit": xb.last_actor_distill_loss_cap_hit,
                        "actor_distill_confidence_mean": xb.last_actor_distill_confidence_mean,
                        "actor_distill_margin_mean": xb.last_actor_distill_margin_mean,
                        "actor_distill_delta_linf_mean": xb.last_actor_distill_delta_linf_mean,
                        "actor_distill_delta_l2_mean": xb.last_actor_distill_delta_l2_mean,
                        "actor_distill_source_frac_rssm": xb.last_actor_distill_source_frac_rssm,
                        "actor_distill_source_frac_elite": xb.last_actor_distill_source_frac_elite,
                        "actor_distill_source_frac_archive": xb.last_actor_distill_source_frac_archive,
                        "actor_distill_reject_low_margin": xb.last_actor_distill_reject_low_margin,
                        "actor_distill_reject_bad_source": xb.last_actor_distill_reject_bad_source,
                        "actor_distill_reject_large_delta": xb.last_actor_distill_reject_large_delta,
                        "actor_distill_reject_high_uncertainty": xb.last_actor_distill_reject_high_uncertainty,
                        "actor_distill_reject_tol": xb.last_actor_distill_reject_tol,
                        "actor_distill_reject_uncertainty_missing": xb.last_actor_distill_reject_uncertainty_missing,
                        # Soft Activation Gate v1 telemetry.
                        "actor_distill_activation_weight_mean": xb.last_actor_distill_activation_weight_mean,
                        "actor_distill_tol_bin": xb.last_actor_distill_tol_bin,
                        "actor_distill_activation_nonzero_frac": xb.last_actor_distill_activation_nonzero_frac,
                        "actor_distill_tol_weight_mean": xb.last_actor_distill_tol_weight_mean,
                        "actor_distill_time_weight_mean": xb.last_actor_distill_time_weight_mean,
                        "actor_distill_perf_weight_mean": xb.last_actor_distill_perf_weight_mean,
                        "actor_distill_conf_weight_mean": xb.last_actor_distill_conf_weight_mean,
                        "actor_distill_unc_weight_mean": xb.last_actor_distill_unc_weight_mean,
                        "actor_distill_source_weight_mean": xb.last_actor_distill_source_weight_mean,
                        "actor_distill_active_frac_tol_gt_006": xb.last_actor_distill_active_frac_tol_gt_006,
                        "actor_distill_active_frac_tol_005_006": xb.last_actor_distill_active_frac_tol_005_006,
                        "actor_distill_active_frac_tol_004_005": xb.last_actor_distill_active_frac_tol_004_005,
                        "actor_distill_active_frac_tol_le_004": xb.last_actor_distill_active_frac_tol_le_004,
                    })
                    for k, v in eir_scalars.items():
                        try:
                            tb_writer.add_scalar(f"eir/{k}", float(v), frames)
                        except Exception:
                            pass
                    if getattr(xb, "snippet_bank", None) is not None:
                        for k, v in (getattr(xb.snippet_bank, "metrics", lambda: {})() or {}).items():
                            try:
                                tb_writer.add_scalar(str(k), float(v), frames)
                            except Exception:
                                pass
                    _snippet_scalars = {
                        "snippet_distill/loss": float(xb.last_snippet_distill_loss),
                        "snippet_distill/active_frac": float(xb.last_snippet_distill_active_frac),
                        "snippet_wm_rerank/changed_frac": float(
                            xb.last_snippet_wm_rerank_changed_frac
                        ),
                        "snippet_wm_rerank/wm_term_mean": float(
                            xb.last_snippet_wm_rerank_wm_term_mean
                        ),
                    }
                    for k, v in _snippet_scalars.items():
                        try:
                            tb_writer.add_scalar(k, float(v), frames)
                        except Exception:
                            pass
                    _wcs = getattr(xb, "wm_calibration_state", None)
                    if isinstance(_wcs, dict):
                        for k, v in (_wcs.get("last_metrics") or {}).items():
                            try:
                                tb_writer.add_scalar(str(k), float(v), frames)
                            except Exception:
                                pass
                        try:
                            tb_writer.add_scalar(
                                "wm_cal/alpha_wm", float(_wcs.get("alpha_wm", 0.0)), frames
                            )
                        except Exception:
                            pass
                    # extra EIR scalars: rssm candidate injection + belief cache
                    _imagine_code = {
                        "prior_rollout": 0.0,
                        "stochastic_prior_rollout": 1.0,
                        "fixed_belief_fallback": 2.0,
                        "ensemble_prior_rollout": 4.0,
                        "ensemble_stochastic_prior_rollout": 5.0,
                    }
                    _bc = getattr(xb, "rssm_belief_cache", None)
                    extra_eir = {
                        "rssm_candidate_added": float(xb.last_rssm_candidate_added),
                        "mppi_active": float(xb.last_mppi_active),
                        "mppi_belief_valid": float(xb.last_mppi_belief_valid),
                        "bank_feature_source_actor_fraction": xb.bank_feature_source_actor_fraction,
                        "bank_feature_source_fallback_fraction": xb.bank_feature_source_fallback_fraction,
                        "rssm_source_bonus_mean": float(getattr(xb, "last_rssm_source_bonus_mean", 0.0)),
                        "rssm_source_bonus_applied_fraction": float(
                            getattr(xb, "last_rssm_source_bonus_applied_fraction", 0.0)
                        ),
                        "rssm_candidate_margin_mean": float(
                            getattr(xb, "last_rssm_candidate_margin_mean", 0.0)
                        ),
                        "rssm_candidate_uncertainty_mean": float(
                            getattr(xb, "last_rssm_candidate_uncertainty_mean", 0.0)
                        ),
                        "elite_candidate_gate_weight": float(
                            xb.last_elite_candidate_gate_weight
                        ),
                        "rssm_candidate_gate_weight": float(
                            xb.last_rssm_candidate_gate_weight
                        ),
                        "elite_candidate_retrieval_allowed": float(
                            xb.last_elite_candidate_retrieval_allowed
                        ),
                        "rssm_candidate_retrieval_allowed": float(
                            xb.last_rssm_candidate_retrieval_allowed
                        ),
                        "elite_teacher_candidate_available_above_004": float(
                            xb.elite_teacher_candidate_available_above_004
                        ),
                        "rssm_teacher_candidate_available_above_004": float(
                            xb.rssm_teacher_candidate_available_above_004
                        ),
                        "any_teacher_candidate_available_above_004": float(
                            xb.any_teacher_candidate_available_above_004
                        ),
                        "teacher_candidate_available_above_004": float(
                            xb.teacher_candidate_available_above_004
                        ),
                        "teacher_selected_above_004": float(
                            xb.teacher_selected_above_004
                        ),
                        "teacher_selected_below_004": float(
                            getattr(xb, "teacher_selected_below_004", 0.0)
                        ),
                        "best_source_frac_elite_below_004": float(
                            getattr(xb, "best_source_frac_elite_below_004", 0.0)
                        ),
                        "actor_distill_nonzero_below_004": float(
                            getattr(xb, "actor_distill_nonzero_below_004", 0.0)
                        ),
                    }
                    for k, v in extra_eir.items():
                        try:
                            tb_writer.add_scalar(f"eir/{k}", float(v), frames)
                        except Exception:
                            pass
                    # full H-step MPPI planner scalars (preferred); else light rr.
                    planner = getattr(xb, "rssm_planner", None)
                    if planner is not None:
                        mppi_scalars = {
                            "active": planner.last_active,
                            "mode": {"off": 0.0, "light": 1.0, "full": 2.0}.get(planner.last_mode, -1.0),
                            "imagine_mode": _imagine_code.get(planner.last_imagine_mode, 3.0),
                            "best_score": planner.last_best_score,
                            "actor_score": planner.last_actor_score,
                            "score_margin": planner.last_score_margin,
                            "uncertainty": planner.last_uncertainty,
                            "best_first_action_delta": planner.last_best_first_action_delta,
                            "fallback_reason": float(planner.last_fallback_reason),
                            "low_margin_fraction": planner.last_low_margin_fraction,
                            "high_uncertainty_fraction": planner.last_high_uncertainty_fraction,
                            "sequence_horizon": float(planner.last_horizon),
                            "num_sequences": float(planner.last_num_sequences),
                            "stochastic_rollouts": float(planner.last_stochastic_rollouts),
                            "num_latent_samples": float(planner.last_num_latent_samples),
                            "sample_temperature": float(planner.last_sample_temperature),
                            "score_std": float(planner.last_score_std),
                            "score_std_mean": float(planner.last_score_std),
                            "drop_cvar": float(planner.last_drop_cvar),
                            "drop_cvar_mean": float(planner.last_drop_cvar),
                            "uncertainty_cvar": float(planner.last_uncertainty_cvar),
                            "risk_adjusted_score_margin": float(planner.last_risk_adjusted_margin),
                            "risk_beta": float(planner.last_risk_beta),
                            "drop_cvar_weight": float(planner.last_drop_cvar_weight),
                            "belief_cache_valid": float(bool(_bc is not None and getattr(_bc, "valid", False))),
                            "belief_cache_rows": float(getattr(_bc, "rows", 0) if _bc is not None else 0),
                            "belief_cache_dim": float(getattr(_bc, "belief_dim", 0) if _bc is not None else 0),
                            "belief_cache_fallback_reason": float(
                                getattr(_bc, "fallback_reason", -1) if _bc is not None else -1
                            ),
                            "belief_cache_aux_zero_filled": float(
                                bool(getattr(_bc, "aux_zero_filled", False)) if _bc is not None else 0.0
                            ),
                            "belief_cache_ensemble_size": float(
                                getattr(_bc, "ensemble_size", 1) if _bc is not None else 1
                            ),
                            "belief_cache_ensemble_valid_size": float(
                                getattr(_bc, "ensemble_valid_size", 1) if _bc is not None else 1
                            ),
                            "belief_cache_ensemble_partial_failure_count": float(
                                getattr(_bc, "ensemble_partial_failure_count", 0) if _bc is not None else 0
                            ),
                            "use_model_ensemble": float(getattr(planner, "last_use_model_ensemble", 0.0)),
                            "ensemble_size": float(getattr(planner, "last_ensemble_size", 0.0)),
                            "ensemble_valid_size": float(getattr(planner, "last_ensemble_valid_size", 0.0)),
                            "ensemble_score_std": float(getattr(planner, "last_ensemble_score_std", 0.0)),
                            "ensemble_score_std_mean": float(getattr(planner, "last_ensemble_score_std", 0.0)),
                            "ensemble_drop_worst": float(getattr(planner, "last_ensemble_drop_worst", 0.0)),
                            "ensemble_uncertainty_worst": float(getattr(planner, "last_ensemble_uncertainty_worst", 0.0)),
                            "ensemble_risk_beta": float(getattr(planner, "last_ensemble_risk_beta", 0.0)),
                            "ensemble_drop_weight": float(getattr(planner, "last_ensemble_drop_weight", 0.0)),
                            "ensemble_uncertainty_weight": float(getattr(planner, "last_ensemble_uncertainty_weight", 0.0)),
                            "ensemble_configured": float(getattr(planner, "last_ensemble_configured", 0.0)),
                            "ensemble_available_size": float(getattr(planner, "last_ensemble_available_size", 0.0)),
                            "ensemble_used": float(getattr(planner, "last_ensemble_used", 0.0)),
                            "ensemble_fallback_reason": float(getattr(planner, "last_ensemble_fallback_reason", 0.0)),
                            "effective_uncertainty_threshold": float(getattr(planner, "last_effective_uncertainty_threshold", 0.0)),
                            "ensemble_uncertainty_threshold": float(getattr(planner, "last_ensemble_uncertainty_threshold", 0.0)),
                            "ensemble_uncertainty_allow_margin": float(getattr(planner, "last_ensemble_uncertainty_allow_margin", 0.0)),
                            "uncertainty_gate_allow_by_margin": float(getattr(planner, "last_uncertainty_gate_allow_by_margin", 0.0)),
                            "uncertainty_gate_blocked": float(getattr(planner, "last_uncertainty_gate_blocked", 0.0)),
                        }
                        for k, v in mppi_scalars.items():
                            try:
                                tb_writer.add_scalar(f"eir_mppi/{k}", float(v), frames)
                            except Exception:
                                pass
                    elif rr is not None:
                        rssm_scalars = {
                            "active": rr.last_active,
                            "mode": 1.0,  # light
                            "best_score": rr.last_best_score,
                            "actor_score": rr.last_actor_score,
                            "score_margin": rr.last_score_margin,
                            "uncertainty": rr.last_uncertainty,
                            "best_first_action_delta": rr.last_best_first_action_delta,
                            "fallback_reason": float(rr.last_fallback_reason),
                            "belief_cache_valid": float(bool(_bc is not None and getattr(_bc, "valid", False))),
                        }
                        for k, v in rssm_scalars.items():
                            try:
                                tb_writer.add_scalar(f"eir_mppi/{k}", float(v), frames)
                            except Exception:
                                pass

                if awr_bridge is not None:
                    awr_stats = dict(getattr(awr_bridge, "stats", {}) or {})
                    if awr_env_wrap is not None:
                        awr_stats.update(getattr(awr_env_wrap, "telemetry", {}) or {})
                    awr_stats.setdefault(
                        "awr/buffer_steps", float(awr_bridge.buffer.total_steps())
                    )
                    awr_stats.setdefault(
                        "awr/buffer_sequences", float(awr_bridge.buffer.num_sequences())
                    )
                    for k, v in awr_stats.items():
                        if not isinstance(v, (int, float)):
                            continue
                        tag = k if k.startswith("awr/") else f"awr/{k}"
                        try:
                            tb_writer.add_scalar(tag, float(v), frames)
                        except Exception:
                            pass

                if seq_distill_bridge is not None:
                    cfg = seq_distill_bridge.cfg
                    cache_meta = (
                        seq_distill_bridge.cache.meta
                        if seq_distill_bridge.cache is not None else {}
                    )
                    inner_model = (
                        actor.model.model
                        if hasattr(actor.model, "model") else actor.model
                    )
                    aux_critic = getattr(inner_model, "seq_aux_critic_mlp", None)
                    aux_critic_grad_norm = 0.0
                    if aux_critic is not None:
                        with torch.no_grad():
                            grad_sq = 0.0
                            for p in aux_critic.parameters():
                                if p.grad is not None:
                                    grad_sq += float(
                                        p.grad.detach().norm().item() ** 2
                                    )
                            aux_critic_grad_norm = float(grad_sq) ** 0.5
                    seq_distill_bridge.last_aux_critic_subtower_grad_norm = (
                        aux_critic_grad_norm
                    )
                    try:
                        tb_writer.add_scalar(
                            "seq_policy_aux/active",
                            float(seq_distill_bridge.last_active),
                            frames,
                        )
                        tb_writer.add_scalar(
                            "seq_policy_aux/warmup_active",
                            float(seq_distill_bridge.last_warmup_active),
                            frames,
                        )
                        tb_writer.add_scalar(
                            "seq_policy_aux/loss_value_distill",
                            float(seq_distill_bridge.last_loss_value),
                            frames,
                        )
                        tb_writer.add_scalar(
                            "seq_policy_aux/loss_success_distill",
                            float(seq_distill_bridge.last_loss_success),
                            frames,
                        )
                        tb_writer.add_scalar(
                            "seq_policy_aux/base_weight",
                            float(seq_distill_bridge.last_base_weight),
                            frames,
                        )
                        tb_writer.add_scalar(
                            "seq_policy_aux/effective_weight",
                            float(seq_distill_bridge.last_effective_weight),
                            frames,
                        )
                        tb_writer.add_scalar(
                            "seq_policy_aux/ramp_factor",
                            float(seq_distill_bridge.last_ramp_factor),
                            frames,
                        )
                        tb_writer.add_scalar(
                            "seq_policy_aux/teacher_value_mean",
                            float(cache_meta.get("teacher_value_mean", 0.0)),
                            frames,
                        )
                        tb_writer.add_scalar(
                            "seq_policy_aux/teacher_value_std",
                            float(cache_meta.get("teacher_value_std", 0.0)),
                            frames,
                        )
                        tb_writer.add_scalar(
                            "seq_policy_aux/teacher_value_norm_enabled",
                            float(bool(
                                cache_meta.get("teacher_value_norm_enabled", False)
                            )),
                            frames,
                        )
                        tb_writer.add_scalar(
                            "seq_policy_aux/teacher_value_norm_std_too_small",
                            float(bool(
                                cache_meta.get(
                                    "teacher_value_norm_std_too_small", False
                                )
                            )),
                            frames,
                        )
                        tb_writer.add_scalar(
                            "seq_policy_aux/teacher_return_corr",
                            float(cache_meta.get("teacher_return_corr", 0.0)),
                            frames,
                        )
                        tb_writer.add_scalar(
                            "seq_policy_aux/teacher_return_corr_valid",
                            float(bool(
                                cache_meta.get("teacher_return_corr_valid", False)
                            )),
                            frames,
                        )
                        tb_writer.add_scalar(
                            "seq_policy_aux/gate_pass",
                            float(seq_distill_bridge.last_gate_pass),
                            frames,
                        )
                        tb_writer.add_scalar(
                            "seq_policy_aux/gate_blocked",
                            float(seq_distill_bridge.last_gate_blocked),
                            frames,
                        )
                        tb_writer.add_scalar(
                            "seq_policy_aux/gate_blocked_count",
                            float(seq_distill_bridge.gate_blocked_count),
                            frames,
                        )
                        tb_writer.add_scalar(
                            "seq_policy_aux/gate_threshold",
                            float(cfg.teacher_corr_threshold),
                            frames,
                        )
                        tb_writer.add_scalar(
                            "seq_policy_aux/gate_user_overridden",
                            float(bool(cfg.gate_user_overridden)),
                            frames,
                        )
                        tb_writer.add_scalar(
                            "seq_policy_aux/feature_source",
                            float(FEATURE_SOURCE_ID.get(cfg.feature_source, -1)),
                            frames,
                        )
                        tb_writer.add_scalar(
                            "seq_policy_aux/feature_dim",
                            256.0,
                            frames,
                        )
                        tb_writer.add_scalar(
                            "seq_policy_aux/actor_feature_captured",
                            float(seq_distill_bridge.last_actor_feature_captured),
                            frames,
                        )
                        tb_writer.add_scalar(
                            "seq_policy_aux/shared_to_aux_critic_feature_captured",
                            float(
                                seq_distill_bridge.last_shared_to_aux_critic_feature_captured
                            ),
                            frames,
                        )
                        tb_writer.add_scalar(
                            "seq_policy_aux/aux_critic_subtower_grad_norm",
                            float(aux_critic_grad_norm),
                            frames,
                        )
                        tb_writer.add_scalar(
                            "seq_policy_aux/coupling_mode_id",
                            float(COUPLING_MODE_ID.get(cfg.mode, -1)),
                            frames,
                        )
                    except Exception:
                        pass

            # One-shot grad-flow log: when coupling != none, past warmup,
            # and the bridge marks active=True, confirm the loss did flow.
            if (
                seq_distill_bridge is not None
                and not seq_wm_trainer._first_distill_assert_done
                and seq_distill_bridge.last_active
                and seq_distill_bridge.last_frames >= int(
                    seq_distill_bridge.cfg.warmup_frames
                )
            ):
                # Sanity check: in v2 the sub-tower must have observed
                # gradient at least once before we declare grad-flow OK.
                if seq_distill_bridge.cfg.is_v2:
                    inner_model = (
                        actor.model.model
                        if hasattr(actor.model, "model") else actor.model
                    )
                    sub = getattr(inner_model, "seq_aux_critic_mlp", None)
                    has_sub_grad = sub is not None and any(
                        (p.grad is not None and p.grad.abs().sum().item() > 0.0)
                        for p in sub.parameters()
                    )
                    if not has_sub_grad:
                        # Defer the assertion to the next epoch instead of
                        # raising; the very first active epoch may run
                        # before the optimizer has seen the new param group.
                        pass
                    else:
                        print(
                            f"[SeqWM] policy coupling={seq_distill_bridge.cfg.mode} "
                            f"feature_source={seq_distill_bridge.cfg.feature_source}: "
                            "PPO model params updated by seq_aux_critic_mlp + "
                            "seq_aux_value_head: yes"
                        )
                        seq_wm_trainer._first_distill_assert_done = True
                else:
                    print(
                        f"[SeqWM] policy coupling={seq_distill_bridge.cfg.mode}: "
                        "PPO model params updated by seq aux head: yes"
                    )
                    seq_wm_trainer._first_distill_assert_done = True

            seq_wm_trainer._epochs_since_save += 1
            if (seq_wm_save_dir is not None
                    and seq_wm_trainer._epochs_since_save >= save_every_epochs):
                seq_wm_trainer._epochs_since_save = 0
                try:
                    save_dir = Path(seq_wm_save_dir)
                    save_dir.mkdir(parents=True, exist_ok=True)
                    payload = seq_wm_trainer.state_dict()
                    payload["frames"] = frames
                    # Minimal ensemble checkpoint: member model state_dicts only
                    # (optimizers reset on resume). Member 0 == primary model,
                    # already covered by payload["model_state_dict"].
                    _ens_mgr = getattr(seq_wm_trainer, "_rssm_ensemble_manager", None)
                    if (
                        _ens_mgr is not None
                        and _ens_mgr.cfg.enabled
                        and _ens_mgr.size > 1
                    ):
                        try:
                            payload["rssm_ensemble"] = _ens_mgr.state_dict()
                        except Exception as _ens_exc:
                            print(f"[RSSM-ensemble] checkpoint export failed: {_ens_exc}")
                    tmp = save_dir / "wm_seq_latest.pt.tmp"
                    final = save_dir / "wm_seq_latest.pt"
                    torch.save(payload, tmp)
                    tmp.replace(final)
                except Exception as exc:
                    print(f"[SeqWM] checkpoint save failed: {exc}")
        except Exception as exc:
            print(f"[SeqWM] train_epoch tail failed: {exc}")
        return result

    actor.train_epoch = patched_train_epoch


def _load_rl_checkpoint(checkpoint_path: str) -> dict | None:
    import torch

    try:
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except Exception:
        return None
    return state if isinstance(state, dict) else None


def _checkpoint_has_aux_value_head(checkpoint_path: str) -> bool:
    """True if rl_games checkpoint model weights include aux_value_head."""
    state = _load_rl_checkpoint(checkpoint_path)
    if state is None:
        return False
    model_sd = state.get("model", {})
    if not isinstance(model_sd, dict):
        return False
    return any(str(k).startswith("aux_value_head") for k in model_sd.keys())


def _checkpoint_optimizer_has_aux_group(checkpoint_path: str) -> bool:
    """True if checkpoint optimizer has more param groups than base PPO (aux head)."""
    state = _load_rl_checkpoint(checkpoint_path)
    if state is None:
        return False
    opt_sd = state.get("optimizer")
    if not isinstance(opt_sd, dict):
        return False
    groups = opt_sd.get("param_groups")
    if not isinstance(groups, list):
        return False
    if len(groups) > 2:
        return True
    return any(isinstance(g, dict) and g.get("name") == "aux_value_head" for g in groups)


def _align_optimizer_state_to_agent(
    agent,
    opt_sd: dict,
) -> dict:
    """Drop checkpoint optimizer groups absent on the live agent.

    PBT cross-copy can load a donor checkpoint (EIR/elite/floorcritic param
    groups) into a target worker with fewer optimizer groups (seq_wm-only or
    ppo_only).  Aligning by group ``name`` lets us load the base PPO moments
    when counts match; otherwise the caller falls back to fresh optimizer state.
    """
    import copy

    agent_groups = list(getattr(agent.optimizer, "param_groups", []) or [])
    ckpt_groups = opt_sd.get("param_groups")
    if not isinstance(ckpt_groups, list) or not agent_groups:
        return opt_sd

    agent_names = [g.get("name") for g in agent_groups if isinstance(g, dict)]
    agent_named = {n for n in agent_names if n}
    agent_unnamed = sum(1 for n in agent_names if not n)

    keep: list[dict] = []
    drop_param_ids: set[int] = set()
    unnamed_kept = 0
    for group in ckpt_groups:
        if not isinstance(group, dict):
            continue
        name = group.get("name")
        if name is None:
            if unnamed_kept < agent_unnamed:
                keep.append(group)
                unnamed_kept += 1
            else:
                drop_param_ids.update(int(p) for p in group.get("params", []))
        elif name in agent_named:
            keep.append(group)
        else:
            drop_param_ids.update(int(p) for p in group.get("params", []))

    if len(keep) == len(ckpt_groups):
        return opt_sd

    aligned = copy.deepcopy(opt_sd)
    aligned["param_groups"] = keep
    opt_state = aligned.get("state")
    if isinstance(opt_state, dict):
        for pid in drop_param_ids:
            opt_state.pop(pid, None)
            opt_state.pop(str(pid), None)
    return aligned


def _pbt_restore_runner_args(
    args_cli,
    *,
    train: bool = True,
    checkpoint: str | None = None,
    sigma=None,
) -> dict:
    """Build rl_games runner args including PBT restore control-plane fields."""
    out: dict = {"train": train, "play": False, "sigma": sigma}
    if checkpoint is not None:
        out["checkpoint"] = checkpoint
    if getattr(args_cli, "pbt_copy_target_frame_before_restore", None) is not None:
        out["pbt_copy_target_frame_before_restore"] = int(
            args_cli.pbt_copy_target_frame_before_restore
        )
    out["pbt_copy_reset_progress_on_large_frame_gap"] = bool(
        getattr(args_cli, "pbt_copy_reset_progress_on_large_frame_gap", True)
    )
    out["pbt_copy_frame_gap_reset_threshold"] = int(
        getattr(args_cli, "pbt_copy_frame_gap_reset_threshold", 500_000_000)
    )
    if getattr(args_cli, "restart_source_policy", None) is not None:
        out["restart_source_policy"] = int(args_cli.restart_source_policy)
    if getattr(args_cli, "pbt_lineage_id_resume", None) is not None:
        out["pbt_lineage_id_resume"] = str(args_cli.pbt_lineage_id_resume)
    if getattr(args_cli, "pbt_lineage_start_pf_resume", None) is not None:
        out["pbt_lineage_start_pf_resume"] = int(args_cli.pbt_lineage_start_pf_resume)
    if getattr(args_cli, "pbt_weight_age_frame_resume", None) is not None:
        out["pbt_weight_age_frame_resume"] = int(args_cli.pbt_weight_age_frame_resume)
    return out


def _wm_safe_restore(
    agent,
    checkpoint_path: str,
    *,
    workspace: str | None = None,
    target_frame_before_restore: int | None = None,
    restart_source_policy: int | None = None,
    pbt_lineage_id_resume: str | None = None,
) -> None:
    """Restore rl_games agent tolerating aux_value_head / optimizer group mismatches."""
    import copy
    import torch
    from dexpbt_isaaclab.pbt.pbt_control_plane import (
        load_meta_for_checkpoint,
        pbt_pure_restore_clocks,
        validate_published_meta,
    )

    from pathlib import Path as _Path

    if restart_source_policy is not None and target_frame_before_restore is None:
        raise RuntimeError("PBT_FRAME_MUST_BE_SNAPSHOT_DRIVEN")
    if pbt_lineage_id_resume:
        agent._pbt_lineage_id_resume = str(pbt_lineage_id_resume)

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(ckpt, dict):
        raise RuntimeError(f"Invalid checkpoint format: {checkpoint_path}")

    model_sd = ckpt.get("model", {})
    if isinstance(model_sd, dict):
        result = agent.model.load_state_dict(model_sd, strict=False)
        aux_missing = [k for k in result.missing_keys if k.startswith("aux_value_head")]
        aux_unexpected = [k for k in result.unexpected_keys if k.startswith("aux_value_head")]
        if aux_missing or aux_unexpected:
            print(
                "[WM] restore: aux_value_head mismatch "
                f"(missing={len(aux_missing)}, unexpected={len(aux_unexpected)}) — strict=False"
            )

    if hasattr(agent, "set_stats_weights"):
        agent.set_stats_weights(ckpt)

    erd_bridge = getattr(agent, "_erd_bridge", None)
    if erd_bridge is not None and getattr(erd_bridge, "scorer", None) is not None:
        erd_sd = ckpt.get("erd_scorer")
        if isinstance(erd_sd, dict):
            result = erd_bridge.scorer.load_state_dict(erd_sd, strict=False)
            print(
                "[ERD] restore: loaded scorer "
                f"(missing={len(result.missing_keys)}, "
                f"unexpected={len(result.unexpected_keys)})"
            )
        else:
            print("[ERD] restore: no erd_scorer in checkpoint; using fresh scorer")
        warmup_done = ckpt.get("erd_scorer_warmup_iters_done")
        if warmup_done is not None:
            erd_bridge.scorer_warmup_iters_done = int(warmup_done)
            print(
                "[ERD] restore: scorer warmup iters done="
                f"{erd_bridge.scorer_warmup_iters_done}"
            )
        else:
            print("[ERD] restore: no scorer warmup counter in checkpoint; using 0")

    # EIR-MPC restore (train-side). The observer-side save of these keys is wired
    # in Phase D; restore is tolerant of their absence.
    eir_bridge = getattr(agent, "_eir_bridge", None)
    if eir_bridge is not None:
        if getattr(eir_bridge, "scorer", None) is not None:
            sd = ckpt.get("eir_scorer")
            if isinstance(sd, dict):
                eir_bridge.scorer.load_state_dict(sd, strict=False)
                print("[EIR] restore: loaded scorer")
            warm = ckpt.get("eir_scorer_warmup_iters_done")
            if warm is not None:
                eir_bridge.scorer_warmup_iters_done = int(warm)
        rb = getattr(eir_bridge, "residual_bridge", None)
        if rb is not None and getattr(rb, "head", None) is not None:
            rsd = ckpt.get("eir_residual_head")
            if isinstance(rsd, dict):
                rb.head.load_state_dict(rsd, strict=False)
                print("[EIR] restore: loaded recovery residual head")
        if getattr(eir_bridge, "bank", None) is not None:
            bsd = ckpt.get("eir_elite_bank")
            if isinstance(bsd, dict):
                try:
                    eir_bridge.bank.load_state_dict(bsd)
                    print(f"[EIR] restore: loaded elite bank size={eir_bridge.bank.size}")
                except Exception as _eir_b_exc:
                    print(f"[EIR] restore: elite bank load failed: {_eir_b_exc}")
        if getattr(eir_bridge, "snippet_bank", None) is not None:
            ssd = ckpt.get("eir_elite_snippet_bank")
            if isinstance(ssd, dict):
                try:
                    eir_bridge.snippet_bank.load_state_dict(ssd)
                    print(
                        f"[EIR] restore: loaded snippet bank size="
                        f"{eir_bridge.snippet_bank.size}"
                    )
                except Exception as _eir_sb_exc:
                    print(f"[EIR] restore: snippet bank load failed: {_eir_sb_exc}")
        wcs = ckpt.get("wm_calibration_state")
        if isinstance(wcs, dict):
            eir_bridge.wm_calibration_state = wcs
            print(f"[WM-cal] restore: alpha_wm={wcs.get('alpha_wm', 0.0)}")

    awr_bridge = getattr(agent, "_awr_bridge", None)
    if awr_bridge is not None and getattr(awr_bridge, "buffer", None) is not None:
        from dexpbt_isaaclab.awr.elite_awr import consume_awr_sidecar_from_checkpoint

        ws = workspace or getattr(agent, "_awr_workspace", None)
        loaded = consume_awr_sidecar_from_checkpoint(
            awr_bridge, checkpoint_path, workspace=ws
        )
        if loaded > 0:
            print(
                f"[AWR] restore: inherited sidecar from checkpoint "
                f"({loaded} steps, source_policy="
                f"{awr_bridge.stats.get('awr/sidecar_source_policy', -1)})"
            )
        else:
            print("[AWR] restore: checkpoint sidecar missing or empty")

    meta_path = _Path(checkpoint_path).with_suffix(".json")
    if meta_path.is_file():
        tick = int(getattr(agent, "_pbt_interval_steps", 20_000_000) or 20_000_000)
        try:
            validate_published_meta(meta_path, tick=tick)
        except RuntimeError as exc:
            print(f"[PBT] restore: published meta validation warning: {exc}")

    meta = load_meta_for_checkpoint(checkpoint_path)
    pbt_pure_restore_clocks(
        agent,
        ckpt,
        meta=meta,
        target_frame_before_restore=target_frame_before_restore,
        restart_source_policy=restart_source_policy,
    )
    agent.last_mean_rewards = ckpt.get(
        "last_mean_rewards", getattr(agent, "last_mean_rewards", -1e9)
    )

    opt_sd = ckpt.get("optimizer")
    if not isinstance(opt_sd, dict):
        return

    agent_groups = len(getattr(agent.optimizer, "param_groups", []))
    ckpt_groups = len(opt_sd.get("param_groups", []))

    def _try_load_opt(state: dict) -> bool:
        try:
            agent.optimizer.load_state_dict(state)
            return True
        except ValueError:
            return False

    if agent_groups == ckpt_groups:
        if _try_load_opt(opt_sd):
            return
    # Strip aux group from checkpoint when target model has fewer groups.
    from dexpbt_isaaclab.pbt.pbt_observer import DexPBTPbtAlgoObserver

    patched = {"optimizer": copy.deepcopy(opt_sd)}
    DexPBTPbtAlgoObserver._strip_aux_optimizer_state(patched)
    if _try_load_opt(patched["optimizer"]):
        print("[WM] restore: loaded optimizer after stripping aux_value_head group from checkpoint")
        return
    aligned = _align_optimizer_state_to_agent(agent, patched["optimizer"])
    if len(aligned.get("param_groups", [])) == agent_groups:
        if _try_load_opt(aligned):
            print(
                "[WM] restore: loaded optimizer after aligning param groups "
                f"(agent_groups={agent_groups}, ckpt_groups={ckpt_groups})"
            )
            return
    print(
        f"[WM] restore: skipped optimizer state "
        f"(agent_groups={agent_groups}, ckpt_groups={ckpt_groups}) — using fresh moments"
    )


def _patch_model_get_aux_loss(model, wm_trainer):
    """
    Monkey-patch rl_games Network.get_aux_loss() to inject value_imag loss.

    This is the cleanest integration path: rl_games A2CAgent.calc_gradients()
    already calls model.get_aux_loss() and adds each returned loss to the PPO total loss.
    The gradient path becomes:
      value_imag_loss.backward() → auxiliary imagined value head
      → shared encoder representation → PPO model params
    """
    original_get_aux_loss = model.get_aux_loss

    def patched_get_aux_loss():
        # Get value_imag loss from WM trainer (returns dict or None)
        wm_loss_dict = wm_trainer.compute_value_imag_loss_dict()
        if wm_loss_dict is None:
            wm_loss_dict = {}

        # Also include any losses from the original method
        base = original_get_aux_loss()
        if base is not None:
            wm_loss_dict.update(base)

        return wm_loss_dict if wm_loss_dict else None

    model.get_aux_loss = patched_get_aux_loss


def _patch_model_get_aux_loss_for_seq(model, seq_distill_bridge: "SeqDistillBridge"):
    """Phase A4 P0 — install seq distill loss into rl_games' get_aux_loss hook.

    rl_games' ``A2CAgent.calc_gradients`` calls ``model.get_aux_loss()`` AFTER
    ``self.model(batch_dict)`` and adds each returned tensor to the PPO loss
    before ``backward()``. By returning a tensor with grad on
    ``seq_aux_value_head`` (and the captured ``actor_mlp_out``) we close the
    gradient loop:

        seq distill loss
            -> seq_aux_value_head params
            -> actor_mlp_out (captured by forward hook)
            -> shared LSTM/MLP encoder
            -> PPO model params

    The teacher targets are read from the pre-built ``SeqDistillBridge.cache``
    (no per-minibatch fwd through the seq WM). Fail-fast on missing
    minibatch indices, stale cache generation, or shape mismatches.
    """
    original_get_aux_loss = model.get_aux_loss

    def patched_get_aux_loss():
        seq_loss_dict = compute_seq_distill_aux_loss(seq_distill_bridge)
        # Also include any losses from the original method (almost always
        # None — rl_games' default Network.get_aux_loss returns None).
        base = original_get_aux_loss()
        if base is not None:
            seq_loss_dict = dict(seq_loss_dict)
            seq_loss_dict.update(base)
        return seq_loss_dict if seq_loss_dict else None

    model.get_aux_loss = patched_get_aux_loss


def _patch_model_get_aux_loss_for_floorcritic(model, floorcritic_bridge):
    """Install the FloorCritic loss into rl_games' get_aux_loss hook.

    Chained: wraps whatever ``model.get_aux_loss`` is currently set to (the
    base method or the seq-distill patch), so FloorCritic and seq distill can
    coexist. The FloorCritic loss is added under key ``floorcritic_total`` and
    rl_games' ``calc_gradients`` adds it to the PPO loss before ``backward()``.
    """
    original_get_aux_loss = model.get_aux_loss

    def patched_get_aux_loss():
        fc_loss_dict = compute_floorcritic_loss(floorcritic_bridge)
        base = original_get_aux_loss()
        if base is not None:
            merged = dict(fc_loss_dict)
            merged.update(base)
            return merged if merged else None
        return fc_loss_dict if fc_loss_dict else None

    model.get_aux_loss = patched_get_aux_loss


def _eir_extract_rollout_features(model, td, feature_dim, actor=None):
    """Full-rollout actor feature extraction for the elite bank (constraint 19.2).

    The ERD/EIR forward-capture hook only sees PPO *minibatch* features, so the
    bank cannot reuse it. Here we replay the rollout through ``a2c_network`` with
    the stored ``mb_rnn_states`` captured during ``play_steps_rnn`` so features
    match the shared encoder path (RNN -> actor_mlp), using the same flatten
    convention as the target cache (``flat[n*T+t]``).

    Returns ``(feature_flat[T*N, feature_dim], source_flag)`` where source_flag is
    0 for actor_mlp features and 1 for the fallback retrieval key.
    """
    import torch as _torch

    def _resolve_a2c(m):
        inner = m.model if hasattr(m, "model") else m
        return getattr(inner, "a2c_network", None)

    try:
        obs = td.get("obses") if hasattr(td, "get") else td["obses"]
        actions = td.get("actions") if hasattr(td, "get") else td["actions"]
        if obs is None or actions is None:
            raise KeyError("obses/actions")
        T, N = int(obs.shape[0]), int(obs.shape[1])
        a2c = _resolve_a2c(model)
        if a2c is None or not hasattr(a2c, "actor_mlp"):
            raise RuntimeError("no a2c_network.actor_mlp")
        device = next(a2c.parameters()).device
        act_dim = int(actions.shape[-1])
        norm_fn = getattr(model, "norm_obs", None)
        mb_rnn = getattr(actor, "mb_rnn_states", None) if actor is not None else None
        seq_length = max(1, int(getattr(actor, "seq_length", 1)) if actor is not None else 1)

        captured: list[_torch.Tensor] = []
        feats: list[_torch.Tensor] = []

        def hook_fn(_module, _inp, out):
            captured.append(out.to(_torch.float32))

        handle = a2c.actor_mlp.register_forward_hook(hook_fn)
        a2c_was = a2c.training
        a2c.eval()
        try:
            with _torch.no_grad():
                for t in range(T):
                    obs_t = obs[t].float().to(device)
                    if callable(norm_fn):
                        obs_t = norm_fn(obs_t)
                    if t == 0:
                        prev_a = _torch.zeros(N, act_dim, device=device, dtype=obs_t.dtype)
                    else:
                        prev_a = actions[t - 1].to(device).float()
                        if prev_a.ndim == 3 and prev_a.shape[-1] == 1:
                            prev_a = prev_a.squeeze(-1)
                    rnn_states = None
                    if mb_rnn is not None and bool(getattr(a2c, "has_rnn", False)):
                        seq_idx = t // seq_length
                        rnn_states = tuple(
                            mb_s[seq_idx].contiguous().to(device) for mb_s in mb_rnn
                        )
                    batch_dict = {
                        "is_train": False,
                        "prev_actions": prev_a,
                        "obs": obs_t,
                        "seq_length": 1,
                    }
                    if rnn_states is not None:
                        batch_dict["rnn_states"] = rnn_states
                    a2c(batch_dict)
                    if not captured:
                        raise RuntimeError("actor_mlp hook did not fire")
                    feats.append(captured.pop())
        finally:
            handle.remove()
            if a2c_was:
                a2c.train()

        feat_tn = _torch.stack(feats, dim=0)
        feat_flat = _erd_flatten_tn(feat_tn)
        if feat_flat.dim() == 2 and int(feat_flat.shape[-1]) == int(feature_dim):
            return feat_flat.detach().cpu(), 0
        raise RuntimeError(
            f"feature width {tuple(feat_flat.shape)} != feature_dim {feature_dim}"
        )
    except Exception as exc:
        # Fallback retrieval key: pack normalized task context into a width-F key.
        try:
            d_hat = td["aux_d_hat"]
            tol = td["aux_success_tolerance"]
            lifted = td.get("aux_lifted") if hasattr(td, "get") else None
            dev = d_hat.device
            dh = _erd_flatten_tn(d_hat.float()).reshape(-1, 1).to(dev)
            tl = _erd_flatten_tn(tol.float()).reshape(-1, 1).to(dev)
            if lifted is not None:
                lf = _erd_flatten_tn(lifted.float()).reshape(-1, 1).to(dev)
            else:
                lf = _torch.ones_like(dh)
            key = _torch.cat([dh, tl, lf, dh / (tl + 1e-6)], dim=-1)
            n = key.shape[0]
            fd = int(feature_dim)
            if key.shape[-1] < fd:
                pad = _torch.zeros(n, fd - key.shape[-1], device=dev, dtype=key.dtype)
                key = _torch.cat([key, pad], dim=-1)
            else:
                key = key[:, :fd]
            print(f"[EIR] feature extraction fell back to task-context key: {exc}")
            return key.detach().cpu(), 1
        except Exception as exc2:
            print(f"[EIR] feature extraction + fallback failed: {exc2}")
            return None, 1


def _install_eir_rollout_apply(agent, eir_bridge):
    """Install the gated recovery-residual rollout apply (constraint 19.3).

    Wraps the model forward so that ONLY at play time (``is_train`` False) and
    ONLY when the residual apply gate fires (floor / hard-floor near-success +
    warmup + aux available) the action mean is shifted by the residual. The shift
    is applied identically to ``mus`` and ``actions`` so the sampled exploration
    noise is preserved. Default-off and exception-safe: any failure falls back to
    the unmodified policy output, so baseline behavior is never broken."""
    import torch as _torch

    model = agent.model
    res_bridge = getattr(eir_bridge, "residual_bridge", None)
    if res_bridge is None or res_bridge.head is None:
        return
    if getattr(model, "_eir_rollout_apply_installed", False):
        return
    orig_forward = model.forward

    def wrapped_forward(input_dict, *a, **kw):
        out = orig_forward(input_dict, *a, **kw)
        try:
            is_train = bool(input_dict.get("is_train", False)) if isinstance(input_dict, dict) else False
            if (not is_train) and isinstance(out, dict) and "mus" in out and "actions" in out:
                feat = getattr(eir_bridge.capture, "tensor", None)
                mus = out["mus"]
                if feat is not None and int(feat.shape[0]) == int(mus.shape[0]):
                    frames = int(getattr(agent, "frame", 0) or 0)
                    res = compute_rollout_residual_action(
                        res_bridge, feat=feat.detach(), mu=mus.detach(), frames=frames
                    )
                    if bool(res["apply_mask"].any()):
                        delta = (res["action"] - mus.detach())
                        out["mus"] = (mus + delta).clamp(-1.0, 1.0)
                        out["actions"] = (out["actions"] + delta).clamp(-1.0, 1.0)
        except Exception:
            pass
        return out

    model.forward = wrapped_forward
    model._eir_rollout_apply_installed = True


def _consume_eir_broadcast_on_startup(agent) -> None:
    """Merge a pending PBT elite-broadcast payload addressed to this policy.

    Runs once at process startup (constraint 19.4 forced-restart hand-off):
    when the observer could not live-merge a payload into a running bridge it
    leaves it in ``<workspace>/eir_broadcast/<policy_idx>.json`` for the next
    process incarnation to pick up here.
    """
    eir_bridge = getattr(agent, "_eir_bridge", None)
    if eir_bridge is None:
        return
    bcfg = getattr(eir_bridge, "broadcast_cfg", None)
    if bcfg is None or not getattr(bcfg, "enabled", False):
        return
    workspace = Path(args_cli.workspace)
    pidx = int(args_cli.policy_idx)
    meta = read_broadcast(workspace, pidx)
    if meta is None:
        return
    payload = load_payload(meta)
    if payload is None:
        return
    summary = merge_payload_into_bridge(eir_bridge, payload, cfg=bcfg)
    mark_consumed(
        workspace,
        pidx,
        frame=int(meta.get("frame", 0)),
        source_policy=int(meta.get("source_policy", -1)),
        mode="restart",
        merged_entries=int(summary.get("merged_entries", 0)),
    )
    print(
        f"[EIR] startup consumed broadcast from policy "
        f"{meta.get('source_policy')}: merged {summary.get('merged_entries')} "
        f"bank entries, residual={summary.get('residual_copied')}, "
        f"scorer={summary.get('scorer_copied')}"
    )


def _patch_model_get_aux_loss_for_eir(model, eir_bridge):
    """Install EIR-MPC losses into rl_games' get_aux_loss hook.

    Returns ``{"eir_total": ...}`` = scorer BCE + gated residual distill. The
    main actor mu receives no direct gradient (constraint 19.1)."""
    original_get_aux_loss = model.get_aux_loss

    def patched_get_aux_loss():
        agent = getattr(eir_bridge, "agent_ref", None)
        frames = int(getattr(agent, "frame", 0) or 0) if agent is not None else 0
        eir_loss_dict = compute_eir_loss(eir_bridge, frames=frames)
        base = original_get_aux_loss()
        if base is not None:
            merged = dict(eir_loss_dict)
            merged.update(base)
            return merged if merged else None
        return eir_loss_dict if eir_loss_dict else None

    model.get_aux_loss = patched_get_aux_loss


def _patch_model_get_aux_loss_for_erd(model, erd_bridge):
    """Install ERD-MPC losses into rl_games' get_aux_loss hook."""
    original_get_aux_loss = model.get_aux_loss

    def patched_get_aux_loss():
        agent = getattr(erd_bridge, "agent_ref", None)
        frames = int(getattr(agent, "frame", 0) or 0) if agent is not None else 0
        erd_loss_dict = compute_erd_loss(erd_bridge, frames=frames)
        base = original_get_aux_loss()
        if base is not None:
            merged = dict(erd_loss_dict)
            merged.update(base)
            return merged if merged else None
        return erd_loss_dict if erd_loss_dict else None

    model.get_aux_loss = patched_get_aux_loss


def _patch_model_get_aux_loss_for_awr(model, awr_bridge):
    from dexpbt_isaaclab.awr.elite_awr import patch_model_get_aux_loss_for_awr

    patch_model_get_aux_loss_for_awr(model, awr_bridge)


def _is_wm_strategy_compatible(src_strategy: str | None, dst_strategy: str) -> tuple[bool, str]:
    src = str(src_strategy or "unknown")
    dst = str(dst_strategy)

    if dst == "none":
        return False, "target_strategy_none_skip_load"
    if src == "none":
        return False, "source_strategy_none_reset"
    if src == "unknown":
        return False, "source_strategy_unknown_reset"
    if src == dst:
        return True, "same_strategy"

    compatible_pairs = {
        # Legacy single-step family.
        ("aux_low", "value_imag_h1"),
        ("aux_mid", "value_imag_h1"),
        ("aux_low", "value_imag_h3"),
        ("aux_mid", "value_imag_h3"),
        ("value_imag_h1", "value_imag_h3"),
        # Phase A4 sequence WM: within-family swaps OK; everything else
        # forces a reset (the SeqWMTrainer's checkpoint schema differs
        # from the legacy WMTrainer's single-step ensemble).
        ("rssm_aux", "rssm_belief_value"),
        ("rssm_belief_value", "rssm_aux"),
        ("trans_aux", "trans_belief_value"),
        ("trans_belief_value", "trans_aux"),
    }
    if (src, dst) in compatible_pairs:
        return True, "compatible_upgrade_path"
    return False, f"incompatible_{src}_to_{dst}"


def _load_wm_checkpoint_with_strategy(
    wm_checkpoint_path: str,
    wm_module,
    wm_optimizer,
    target_strategy: str,
    source_strategy_hint: str = "unknown",
    wm_load_mode: str = "auto",
) -> dict[str, str | bool]:
    from dexpbt_isaaclab.world_model import WMCheckpointManager

    result: dict[str, str | bool] = {
        "loaded": False,
        "checkpoint_strategy": "unknown",
        "target_strategy": str(target_strategy),
        "compatible": False,
        "reset_reason": "",
        "aux_value_head_state_available": False,
    }
    if wm_load_mode == "reset":
        wm_module.reset_parameters()
        if wm_optimizer is not None:
            wm_optimizer.state = {}
        result["checkpoint_strategy"] = str(source_strategy_hint or "unknown")
        result["reset_reason"] = "wm_load_mode_reset"
        return result

    ckpt_path = Path(wm_checkpoint_path).resolve()
    if not ckpt_path.exists():
        wm_module.reset_parameters()
        if wm_optimizer is not None:
            wm_optimizer.state = {}
        result["reset_reason"] = "checkpoint_not_found"
        return result

    try:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    except Exception as exc:
        wm_module.reset_parameters()
        if wm_optimizer is not None:
            wm_optimizer.state = {}
        result["reset_reason"] = f"checkpoint_read_error:{exc}"
        return result
    if not isinstance(ckpt, dict):
        wm_module.reset_parameters()
        if wm_optimizer is not None:
            wm_optimizer.state = {}
        result["reset_reason"] = "invalid_checkpoint_format"
        return result

    ckpt_strategy = str(ckpt.get("strategy", source_strategy_hint or "unknown"))
    result["checkpoint_strategy"] = ckpt_strategy
    compatible, reason = _is_wm_strategy_compatible(ckpt_strategy, target_strategy)
    if wm_load_mode == "force":
        compatible = True
        reason = "wm_load_mode_force"
    result["compatible"] = bool(compatible)
    if not compatible:
        wm_module.reset_parameters()
        if wm_optimizer is not None:
            wm_optimizer.state = {}
        result["reset_reason"] = reason
        return result

    loaded = False
    try:
        # Prefer manager load_* path when checkpoint layout matches wm_checkpoints/.
        # policy_workspace is parent of wm_checkpoints.
        wm_dir = ckpt_path.parent
        if wm_dir.name == "wm_checkpoints":
            manager = WMCheckpointManager(wm_dir.parent)
            if ckpt_path.name == "wm_latest.pt":
                loaded = bool(manager.load_latest(wm_module, wm_optimizer))
            elif ckpt_path.stem.startswith("wm_iter"):
                iter_str = ckpt_path.stem.removeprefix("wm_iter")
                if iter_str.isdigit():
                    loaded = bool(manager.load_iteration(wm_module, wm_optimizer, int(iter_str)))
        if not loaded:
            wm_module.load_state_dict(ckpt.get("wm_module_state_dict", {}), strict=False)
            opt_sd = ckpt.get("wm_optimizer_state_dict")
            if wm_optimizer is not None and isinstance(opt_sd, dict):
                wm_optimizer.load_state_dict(opt_sd)
            norm_sd = ckpt.get("wm_normalizer_state_dict")
            if isinstance(norm_sd, dict):
                if hasattr(wm_module, "obs_normalizer") and getattr(wm_module, "obs_normalizer") is not None:
                    try:
                        wm_module.obs_normalizer.load_state_dict(norm_sd, strict=False)
                    except Exception:
                        pass
            loaded = True
        aux_state = ckpt.get("aux_value_head_state_dict")
        if isinstance(aux_state, dict):
            result["aux_value_head_state_available"] = True
            # Stash for delayed restore after aux head is registered on PPO model.
            result["_aux_value_head_state"] = aux_state
    except Exception as exc:
        wm_module.reset_parameters()
        if wm_optimizer is not None:
            wm_optimizer.state = {}
        result["reset_reason"] = f"checkpoint_load_error:{exc}"
        return result

    result["loaded"] = bool(loaded)
    if not loaded:
        wm_module.reset_parameters()
        if wm_optimizer is not None:
            wm_optimizer.state = {}
        result["reset_reason"] = "checkpoint_load_failed"
    return result


class _CheckpointSafeRunner(Runner):
    """Runner that tolerates cross-profile PBT checkpoint restore.

    ppo_only and other non-WM profiles use the stock ``Runner``, which calls
    rl_games ``_restore`` and crashes when a donor checkpoint carries extra
    optimizer param groups (EIR/elite/floorcritic).  This override mirrors the
    restore path already used by ``_WmEnabledRunner``.
    """

    def run_train(self, args):
        import torch
        from rl_games import torch_runner

        print("Started to train")

        agent = self.algo_factory.create(self.algo_name, base_name="run", params=self.params)

        ckpt_path = args.get("checkpoint")
        if ckpt_path:
            if args.get("restart_source_policy") is not None and args.get(
                "pbt_copy_target_frame_before_restore"
            ) is None:
                raise RuntimeError("PBT_FRAME_MUST_BE_SNAPSHOT_DRIVEN")
            _wm_safe_restore(
                agent,
                ckpt_path,
                workspace=getattr(self, "workspace", None),
                target_frame_before_restore=(
                    int(args["pbt_copy_target_frame_before_restore"])
                    if args.get("pbt_copy_target_frame_before_restore") is not None
                    else None
                ),
                restart_source_policy=(
                    int(args["restart_source_policy"])
                    if args.get("restart_source_policy") is not None
                    else None
                ),
                pbt_lineage_id_resume=args.get("pbt_lineage_id_resume"),
            )
        else:
            torch_runner._restore(agent, args)
        torch_runner._override_sigma(agent, args)

        compile_config = self.params.get("config", {}).get("torch_compile", True)
        if compile_config is False:
            print("torch.compile: Disabled")
        else:
            if isinstance(compile_config, bool):
                actor_mode = "default"
                critic_mode = "default"
            elif isinstance(compile_config, str):
                actor_mode = compile_config
                critic_mode = compile_config
            elif isinstance(compile_config, dict):
                actor_mode = compile_config.get("mode", "default")
                critic_mode = compile_config.get("critic_mode", actor_mode)
            else:
                print(f"Warning: Invalid torch_compile config {compile_config}, using default")
                actor_mode = "default"
                critic_mode = "default"

            print(f"torch.compile: Enabled for actor with mode='{actor_mode}'")
            agent.model = torch.compile(agent.model, mode=actor_mode)

            if hasattr(agent, "central_value_net") and agent.central_value_net is not None:
                print(f"torch.compile: Enabled for central value critic with mode='{critic_mode}'")
                agent.central_value_net.model = torch.compile(
                    agent.central_value_net.model,
                    mode=critic_mode,
                )

        self.algo = agent
        agent.train()


class _WmEnabledRunner(Runner):
    """
    Custom Runner that injects WM trainer into A2CAgent after creation.

    Overrides run_train() to:
    1. Create A2CAgent as normal
    2. Connect WM trainer to agent.model + agent.optimizer
    3. Monkey-patch model.get_aux_loss() for value_imag
    4. Monkey-patch agent.train() for WM dynamics training
    5. Run the normal training loop
    """

    def __init__(self, observer, wm_trainer, wm_ckpt_manager, wm_strategy,
                 tensorboard_writer=None, cf_tracker=None, raw_env=None,
                 cf_reset_epoch_counter: bool = False,
                 seq_wm_trainer=None, seq_aux_collector=None,
                 seq_wm_save_dir=None, seq_aux_env_wrap=None,
                 seq_policy_coupling_cfg: "SeqPolicyCouplingConfig | None" = None,
                 awr_bridge=None, awr_env_wrap=None):
        super().__init__(observer)
        self.wm_trainer = wm_trainer
        self.wm_ckpt_manager = wm_ckpt_manager
        self.wm_strategy = wm_strategy
        self.tensorboard_writer = tensorboard_writer
        self.cf_tracker = cf_tracker
        self.raw_env = raw_env
        # Phase A4: sequence WM (RSSM/Transformer) is wired in mutually
        # exclusive to wm_trainer (legacy single-step). When seq_wm_trainer
        # is provided, run_train uses _patch_actor_train_for_seq_wm and
        # SKIPS _patch_model_get_aux_loss / _patch_actor_train_for_wm.
        self.seq_wm_trainer = seq_wm_trainer
        self.seq_aux_collector = seq_aux_collector
        self.seq_wm_save_dir = seq_wm_save_dir
        self.seq_aux_env_wrap = seq_aux_env_wrap
        self.awr_bridge = awr_bridge
        self.awr_env_wrap = awr_env_wrap
        self.seq_policy_coupling_cfg = (
            seq_policy_coupling_cfg or SeqPolicyCouplingConfig()
        )
        # Patch 3 (rev 2, 2026-05-30): CF child needs to reset rl_games
        # `agent.frame` / `agent.epoch_num` / `last_mean_rewards` after
        # restoring the source checkpoint, otherwise the agent's recorded
        # epoch (e.g. 12671) exceeds CF's local `max_epochs` (e.g. 77) and
        # the training loop exits immediately with "MAX EPOCHS NUM!"
        # (observed in regrasping_cbwm_20260530_090008 cf_block_85 logs:
        # all children completed in ~3s without any training).
        self.cf_reset_epoch_counter = bool(cf_reset_epoch_counter)
        # Patch C (rev 3, 2026-05-30): expose the created A2CAgent and
        # remember the env-frame boundaries of a single `agent.train()`
        # call. The rl_games base `Runner.run_train` does NOT assign
        # `self.algo = agent`, so any caller that does
        #     getattr(runner, "algo", None)
        # after run_train ends would see None and child_train_frames
        # would be reported as 0 -- the bug behind every CF block being
        # marked invalid in regrasping_cbwm_20260530_160513.
        self.algo = None
        self._start_frame: int = 0
        self._end_frame: int = 0

    def run_train(self, args):
        """Override to inject WM trainer after A2CAgent creation."""
        import torch
        from rl_games.common import a2c_common
        from rl_games.algos_torch import a2c_continuous
        from rl_games import torch_runner

        # Replicate runner.run_train but with WM injection
        agent = self.algo_factory.create(self.algo_name, base_name='run', params=self.params)

        # =========================================================================
        # CRITICAL FIX: All WM injection MUST happen before _restore().
        #
        # On restart (PBT replacement), the checkpoint contains aux_value_head keys
        # (saved by WMCheckpointManager from a previous value_imag_h1 run). If we
        # call _restore() before registering aux_value_head, PyTorch will reject the
        # unexpected keys and crash with:
        #   RuntimeError: Unexpected key(s) in state_dict: "aux_value_head.0.weight" ...
        #
        # The fix: register all model extensions (aux_value_head, patches) BEFORE
        # checkpoint restore, so load_state_dict() accepts the extra keys.
        # =========================================================================

        wm_trainer = self.wm_trainer
        if wm_trainer is not None:
            # Connect WM trainer to A2CAgent first (before any checkpoint I/O)
            wm_trainer.ppo_model = agent.model       # rl_games Network
            wm_trainer.ppo_optimizer = agent.optimizer  # Adam optimizer
            # NOTE: tensorboard writer is lazily connected in patched_train_epoch()
            # (actor.writer is only available after A2CCommon.__init__ runs)

            # Register aux_value_head BEFORE _restore for any WM checkpoint load.
            # PBT donors often used value_imag_h1 (model + optimizer param groups).
            # Posterior may resample aux_mid: model keys may be stripped in
            # _rewrite_checkpoint, but optimizer state still has the extra group.
            ckpt_path = args.get("checkpoint")
            needs_aux_head = wm_trainer.strategy.use_value_imag
            if not needs_aux_head and ckpt_path:
                needs_aux_head = (
                    _checkpoint_has_aux_value_head(ckpt_path)
                    or _checkpoint_optimizer_has_aux_group(ckpt_path)
                )
            if needs_aux_head:
                wm_trainer._add_aux_value_head()
                if not wm_trainer.strategy.use_value_imag:
                    print(
                        "[WM] Registered aux_value_head for checkpoint load only "
                        f"(strategy={self.wm_strategy} has no value_imag loss)"
                    )
                pending_aux_sd = getattr(wm_trainer, "_pending_aux_value_head_state", None)
                if isinstance(pending_aux_sd, dict):
                    try:
                        wm_trainer.aux_value_head.load_state_dict(pending_aux_sd, strict=False)
                        print("[WM] Warm-start restored aux_value_head state from WM checkpoint")
                    except Exception as exc:
                        print(f"[WM] Failed to restore aux_value_head from WM checkpoint: {exc}")

            # Monkey-patch model.get_aux_loss() for value_imag
            _patch_model_get_aux_loss(agent.model, wm_trainer)

            # Monkey-patch agent.train() for WM dynamics training
            _patch_actor_train_for_wm(agent, wm_trainer, self.wm_ckpt_manager, self.wm_strategy)

        # Phase A4: sequence WM path (mutually exclusive with legacy WMTrainer).
        if self.seq_wm_trainer is not None:
            seq_distill_bridge = None
            seq_policy_cfg = self.seq_policy_coupling_cfg
            if seq_policy_cfg is not None and seq_policy_cfg.enabled:
                seq_distill_bridge = SeqDistillBridge(cfg=seq_policy_cfg)
                seq_distill_bridge.agent_ref = agent
                # Register student heads on the rl_games model + add them
                # to the PPO optimizer's param groups.
                heads = attach_seq_aux_heads(
                    rl_games_model=agent.model,
                    optimizer=agent.optimizer,
                    cfg=seq_policy_cfg,
                    feature_dim=256,
                    device=next(agent.model.parameters()).device,
                )
                seq_distill_bridge.seq_aux_value_head = heads.get(
                    "seq_aux_value_head"
                )
                seq_distill_bridge.seq_aux_success_head = heads.get(
                    "seq_aux_success_head"
                )
                seq_distill_bridge.seq_aux_critic_mlp = heads.get(
                    "seq_aux_critic_mlp"
                )
                if seq_policy_cfg.is_v2 and seq_distill_bridge.seq_aux_critic_mlp is None:
                    raise RuntimeError(
                        "[SeqDistill] mode=critic_distill_value_norm requires "
                        "seq_aux_critic_mlp to be attached, but attach_seq_aux_heads "
                        "did not return one (refusing silent fallback to "
                        "actor-direct path)."
                    )
                # Install the shared-encoder forward hook. Both legacy and
                # v2 paths target a2c_network.actor_mlp (the only populated
                # feature module under network.separate=false), but the
                # captured tensor is consumed differently downstream:
                # legacy feeds seq_aux_value_head directly, v2 feeds
                # seq_aux_critic_mlp -> seq_aux_value_head. The bridge
                # resets the capture at rollout start so a stale tensor
                # cannot be silently reused.
                if seq_policy_cfg.uses_shared_to_aux_critic:
                    self._seq_distill_hook_handle = register_shared_to_aux_critic_capture(
                        agent.model,
                        seq_distill_bridge.feature_capture,
                    )
                else:
                    self._seq_distill_hook_handle = register_actor_mlp_capture(
                        agent.model,
                        seq_distill_bridge.feature_capture,
                    )
                # Install the get_aux_loss patch BEFORE _restore so any
                # checkpoint restore that happens later sees the head
                # weights / param groups already wired up.
                _patch_model_get_aux_loss_for_seq(agent.model, seq_distill_bridge)
                print(
                    f"[SeqDistill] enabled: mode={seq_policy_cfg.mode} "
                    f"weight={seq_policy_cfg.weight} "
                    f"value_w={seq_policy_cfg.value_weight} "
                    f"success_w={seq_policy_cfg.success_weight} "
                    f"warmup_frames={seq_policy_cfg.warmup_frames} "
                    f"feature_source={seq_policy_cfg.feature_source} "
                    f"value_norm={seq_policy_cfg.teacher_value_norm} "
                    f"gate={seq_policy_cfg.gate} "
                    f"corr_threshold={seq_policy_cfg.teacher_corr_threshold} "
                    f"ramp_frames={seq_policy_cfg.ramp_frames}"
                )

            # FloorCritic (Real-target) — independent hook + get_aux_loss
            # patch, chained AFTER the seq-distill patch so both can coexist.
            fc_cfg = getattr(self.seq_wm_trainer, "_floorcritic_cfg", None)
            if fc_cfg is not None and fc_cfg.enabled:
                fc_bridge = FloorCriticBridge(cfg=fc_cfg)
                fc_bridge.agent_ref = agent
                fc_bridge.heads = attach_floorcritic_heads(
                    optimizer=agent.optimizer,
                    cfg=fc_cfg,
                    device=next(agent.model.parameters()).device,
                )
                # Independent feature capture (separate container from the
                # seq-distill capture; both target a2c_network.actor_mlp).
                self._floorcritic_hook_handle = register_floorcritic_capture(
                    agent.model, fc_bridge.capture
                )
                # Backward probe for grad-ratio (shared mode auto-scale).
                if fc_cfg.effective_feature_source == "actor_mlp_out_shared":
                    self._floorcritic_grad_probe_handle = (
                        register_actor_mlp_grad_probe(agent.model, fc_bridge)
                    )
                # Chain get_aux_loss: wrap whatever is currently installed
                # (base model method or the seq-distill patch).
                _patch_model_get_aux_loss_for_floorcritic(agent.model, fc_bridge)
                self.seq_wm_trainer._floorcritic_bridge = fc_bridge
                print(
                    f"[FloorCritic] enabled: feature_source={fc_cfg.feature_source} "
                    f"(eff={fc_cfg.effective_feature_source}) "
                    f"weight={fc_cfg.weight}/{fc_cfg.floor_weight}/{fc_cfg.hard_floor_weight} "
                    f"horizons={fc_cfg.horizons} "
                    f"max_grad_ratio={fc_cfg.max_grad_ratio} "
                    f"auto_scale={fc_cfg.auto_scale_by_grad_ratio} "
                    f"detach={fc_cfg.should_detach} heads={fc_cfg.head_keys()}"
                )

            erd_cfg = getattr(self.seq_wm_trainer, "_erd_cfg", None)
            _eir_cfg_chk = getattr(self.seq_wm_trainer, "_eir_cfg", None)
            _eir_on = bool(_eir_cfg_chk is not None and _eir_cfg_chk.enabled)
            if erd_cfg is not None and erd_cfg.enabled and not _eir_on:
                erd_bridge = ERDBridge(cfg=erd_cfg)
                erd_bridge.agent_ref = agent
                erd_bridge.scorer = attach_erd_modules(
                    optimizer=agent.optimizer,
                    cfg=erd_cfg,
                    device=next(agent.model.parameters()).device,
                )
                self._erd_hook_handle = register_erd_capture(
                    agent.model, erd_bridge.capture
                )
                self._erd_grad_probe_handle = register_erd_actor_mlp_grad_probe(
                    agent.model, erd_bridge
                )
                _patch_model_get_aux_loss_for_erd(agent.model, erd_bridge)
                self.seq_wm_trainer._erd_bridge = erd_bridge
                agent._erd_bridge = erd_bridge
                print(
                    f"[ERD] enabled: horizons={erd_cfg.horizons} "
                    f"candidate_k={erd_cfg.candidate_k} "
                    f"noise={erd_cfg.action_noise_std} "
                    f"delta_cap={erd_cfg.max_action_delta} "
                    f"scorer_w={erd_cfg.scorer_loss_weight} "
                    f"distill_w={erd_cfg.distill_loss_weight}/"
                    f"{erd_cfg.floor_distill_loss_weight}/"
                    f"{erd_cfg.hard_floor_distill_loss_weight} "
                    f"warmup_frames={erd_cfg.start_after_frames} "
                    f"warmup_iters={erd_cfg.scorer_warmup_iters} "
                    f"max_grad_ratio={erd_cfg.max_grad_ratio}"
                )

            # ---------------- EIR-MPC attach (supersedes ERD) ----------------
            eir_cfg = getattr(self.seq_wm_trainer, "_eir_cfg", None)
            if eir_cfg is not None and eir_cfg.enabled:
                dev = next(agent.model.parameters()).device
                eir_scorer_cfg = self.seq_wm_trainer._eir_scorer_cfg
                eir_bridge = EIRBridge(eir_cfg=eir_cfg, erd_cfg=eir_scorer_cfg)
                eir_bridge.agent_ref = agent
                eir_bridge.scorer = attach_erd_modules(
                    optimizer=agent.optimizer, cfg=eir_scorer_cfg, device=dev
                )
                # gated recovery residual head (separate optimizer group)
                res_cfg = self.seq_wm_trainer._eir_residual_cfg
                res_bridge = RecoveryResidualBridge(cfg=res_cfg)
                if res_cfg.enabled:
                    res_bridge.head = attach_recovery_residual(
                        optimizer=agent.optimizer, cfg=res_cfg, device=dev
                    )
                eir_bridge.residual_bridge = res_bridge
                # elite recovery bank (CPU-resident) + archive placeholder
                bank_cfg = self.seq_wm_trainer._eir_bank_cfg
                eir_bridge.bank = EliteRecoveryBank(bank_cfg) if bank_cfg.enabled else None
                eir_bridge.archive_bank = None  # populated by PBT broadcast (Phase D)
                snippet_bank_cfg = getattr(self.seq_wm_trainer, "_eir_snippet_bank_cfg", None)
                eir_bridge.snippet_bank = (
                    EliteSnippetBank(snippet_bank_cfg)
                    if snippet_bank_cfg is not None and snippet_bank_cfg.enabled
                    else None
                )
                eir_bridge.snippet_distill_cfg = getattr(
                    self.seq_wm_trainer, "_snippet_distill_cfg", None
                )
                eir_bridge.snippet_wm_rerank_cfg = getattr(
                    self.seq_wm_trainer, "_snippet_wm_rerank_cfg", None
                )
                eir_bridge.wm_calibration_enable = bool(
                    getattr(self.args_cli, "wm_calibration_enable", False)
                )
                # RSSM reranking / full H-step MPPI.
                rssm_cfg = self.seq_wm_trainer._eir_rssm_cfg
                mppi_cfg = getattr(self.seq_wm_trainer, "_eir_mppi_cfg", None)
                _wm_model = getattr(self.seq_wm_trainer, "model", None)
                _is_rssm = _wm_model is not None and hasattr(_wm_model, "get_state_from_belief")
                eir_bridge.rssm_reranker = RSSMReranker(rssm_cfg) if rssm_cfg.enabled else None
                # full planner only when the backbone supports prior rollout
                # (RSSM family). For non-RSSM backbones the planner stays off
                # and the light reranker / elite fallback handles everything.
                if mppi_cfg is not None and mppi_cfg.enabled and _is_rssm:
                    eir_bridge.rssm_planner = RSSMMPPIPlanner(mppi_cfg)
                    eir_bridge.rssm_model = _wm_model
                    # ensemble models for MPPI scoring (only when both the
                    # ensemble manager exists AND --eir_mppi_use_model_ensemble
                    # is set). Otherwise stays None -> single-model MPPI.
                    _ens_mgr = getattr(self.seq_wm_trainer, "_rssm_ensemble_manager", None)
                    if (
                        getattr(mppi_cfg, "use_model_ensemble", False)
                        and _ens_mgr is not None
                        and _ens_mgr.size > 1
                    ):
                        eir_bridge.rssm_models = _ens_mgr.models
                        print(
                            f"[EIR-MPPI] model ensemble attached: size={_ens_mgr.size} "
                            f"min_valid={mppi_cfg.ensemble_min_valid_models} "
                            f"ens_risk_beta={mppi_cfg.ensemble_risk_beta} "
                            f"ens_drop_w={mppi_cfg.ensemble_drop_weight}"
                        )
                    else:
                        eir_bridge.rssm_models = None
                    print(
                        f"[EIR-MPPI] full H-step planner attached: "
                        f"H={mppi_cfg.horizon} K={mppi_cfg.num_sequences} "
                        f"cem_iters={mppi_cfg.cem_iters} start={mppi_cfg.start_after_frames} "
                        f"prior_rollout={getattr(_wm_model, 'supports_prior_rollout', False)}"
                    )
                elif mppi_cfg is not None and mppi_cfg.enabled and not _is_rssm:
                    print(
                        "[EIR-MPPI] full mode requested but backbone is not RSSM "
                        "(no get_state_from_belief); MPPI disabled, using light/elite fallback."
                    )
                # heads for RSSM reranking / snippet WM rerank (fixed-belief rescoring)
                _heads = getattr(_wm_model, "heads", None)
                if _heads is not None and hasattr(_heads, "predict_transition"):
                    eir_bridge.rssm_predict_fn = _heads.predict_transition
                    if getattr(_heads, "has_value_head", False):
                        eir_bridge.rssm_value_fn = _heads.predict_value
                # per-rollout belief cache is populated in patched_play_steps_rnn
                eir_bridge.rssm_belief_provider = None
                eir_bridge.rssm_belief_cache = None
                # PBT elite broadcast config (consumed by the observer / startup)
                eir_bridge.broadcast_cfg = getattr(
                    self.seq_wm_trainer, "_eir_broadcast_cfg", None
                )
                # reuse the actor_mlp forward capture for scorer + residual
                eir_bridge.capture = ERDBridge(cfg=eir_scorer_cfg).capture
                self._eir_hook_handle = register_erd_capture(
                    agent.model, eir_bridge.capture
                )
                _patch_model_get_aux_loss_for_eir(agent.model, eir_bridge)
                if res_cfg.enabled and res_cfg.apply_to_rollout:
                    _install_eir_rollout_apply(agent, eir_bridge)
                self.seq_wm_trainer._eir_bridge = eir_bridge
                agent._eir_bridge = eir_bridge
                print(
                    f"[EIR] enabled: horizons={eir_cfg.horizons} "
                    f"cand(local/elite/archive)={eir_cfg.local_candidate_k}/"
                    f"{eir_cfg.elite_candidate_k}/{eir_cfg.archive_candidate_k} "
                    f"bank={'on' if eir_bridge.bank is not None else 'off'} "
                    f"residual={'on' if res_cfg.enabled else 'off'} "
                    f"residual_apply={res_cfg.apply_to_rollout}@{res_cfg.apply_after_frames} "
                    f"rssm_rerank={'on' if eir_bridge.rssm_reranker is not None else 'off'} "
                    f"min_margin={eir_cfg.min_score_margin} "
                    f"mu_base_detach={res_cfg.mu_base_detach} "
                    f"(ERD raw-mu distill DISABLED)"
                )

            if self.awr_bridge is not None:
                self.awr_bridge.agent_ref = agent
                _patch_model_get_aux_loss_for_awr(agent.model, self.awr_bridge)
                agent._awr_bridge = self.awr_bridge
                agent._awr_workspace = getattr(self, "workspace", None)
                print(
                    f"[AWR] aux loss patch installed (loss_weight={self.awr_bridge.cfg.loss_weight}, "
                    f"kl_gate={'on' if self.awr_bridge.cfg.kl_gate_enable else 'off'})"
                )

            _patch_actor_train_for_seq_wm(
                agent,
                self.seq_wm_trainer,
                self.seq_aux_collector,
                self.seq_wm_save_dir,
                seq_aux_env_wrap=self.seq_aux_env_wrap,
                seq_distill_bridge=seq_distill_bridge,
                awr_bridge=self.awr_bridge,
                awr_env_wrap=self.awr_env_wrap,
            )
            print(
                f"[SeqWM] Connected to A2CAgent: strategy={self.wm_strategy}, "
                f"backbone={type(self.seq_wm_trainer.model).__name__}, "
                f"seq_len={self.seq_wm_trainer.cfg.seq_len}, "
                f"seq_batch_size={self.seq_wm_trainer.cfg.seq_batch_size}"
            )

        _patch_actor_train_for_cf_meta(agent, self.raw_env, self.cf_tracker)

        if wm_trainer is not None:
            print(f"[WM] Connected to A2CAgent: strategy={self.wm_strategy}, "
                  f"model={type(agent.model).__name__}, "
                  f"optimizer={type(agent.optimizer).__name__}")

        # Restore weights if checkpoint specified — AFTER aux_value_head is registered
        ckpt_path = args.get("checkpoint")
        if ckpt_path:
            if args.get("restart_source_policy") is not None and args.get(
                "pbt_copy_target_frame_before_restore"
            ) is None:
                raise RuntimeError("PBT_FRAME_MUST_BE_SNAPSHOT_DRIVEN")
            # Always use the tolerant restore path on PBT restart / warm-start.
            # seq_wm-only profiles (wm_curriculum_controller, wm_pbt_source_selector)
            # previously fell through to rl_games _restore and crashed when a donor
            # checkpoint carried EIR/elite optimizer param groups.
            _wm_safe_restore(
                agent,
                ckpt_path,
                workspace=getattr(self, "workspace", None),
                target_frame_before_restore=(
                    int(args["pbt_copy_target_frame_before_restore"])
                    if args.get("pbt_copy_target_frame_before_restore") is not None
                    else None
                ),
                restart_source_policy=(
                    int(args["restart_source_policy"])
                    if args.get("restart_source_policy") is not None
                    else None
                ),
                pbt_lineage_id_resume=args.get("pbt_lineage_id_resume"),
            )
        else:
            torch_runner._restore(agent, args)
        torch_runner._override_sigma(agent, args)

        # EIR-MPC (Phase D, constraint 19.4): a forced-restart broadcast target
        # consumes its mailbox on startup so a payload that could not be
        # live-merged by the observer is still folded into the fresh process.
        try:
            _consume_eir_broadcast_on_startup(agent)
        except Exception as _eir_bc_exc:  # pragma: no cover - never crash startup
            print(f"[EIR] startup broadcast consume skipped: {_eir_bc_exc}")

        # Patch 3 (rev 2, 2026-05-30): reset training-loop counters BEFORE
        # `agent.train()` so a CF child that loaded a source checkpoint with
        # `frame=1.7B, epoch=12671` does not instantly hit max_epochs and
        # exit with "MAX EPOCHS NUM!" (collapse-era CF children completed
        # in ~3 seconds with samples=1; see cf_block_85/*/train.log).
        if self.cf_reset_epoch_counter:
            before_frame = int(getattr(agent, "frame", 0) or 0)
            before_epoch = int(getattr(agent, "epoch_num", 0) or 0)
            agent.frame = 0
            agent.epoch_num = 0
            # `last_mean_rewards` is used by rl_games to track best-so-far;
            # resetting it lets the child's own training results be recorded
            # without being shadowed by source's historical bests.
            if hasattr(agent, "last_mean_rewards"):
                agent.last_mean_rewards = -1e6
            print(
                "[CF] reset agent counters before train: "
                f"frame {before_frame} -> 0, "
                f"epoch_num {before_epoch} -> 0"
            )

        # Patch C (rev 3, 2026-05-30): record the env-frame boundary
        # AFTER any reset, so `_end_frame - _start_frame` reliably equals
        # the frames consumed by this `agent.train()` call. Stash the
        # live `agent` on `self.algo` BEFORE training so a future crash
        # path still leaves the counters accessible to post-mortem code.
        self.algo = agent
        self._start_frame = int(getattr(agent, "frame", 0) or 0)

        # Disable torch.compile (Blackwell/rl_games/Triton stability)
        compile_config = self.params.get('config', {}).get('torch_compile', True)
        if compile_config is False:
            print("[WM] torch.compile disabled")
        else:
            agent.model = torch.compile(agent.model, mode="default")

        agent.train()

        # Patch C (rev 3, 2026-05-30): snapshot the post-train env-frame
        # so the CF status writer can compute `child_train_frames` even
        # when `agent.frame` itself was clobbered by a downstream restore.
        self._end_frame = int(getattr(agent, "frame", 0) or 0)
        self.algo = agent


def _load_agent_cfg() -> dict:
    """Load the YAML agent config shipped with this package."""
    cfg_file = resources.files("dexpbt_isaaclab.agents").joinpath("rl_games_ppo_cfg.yaml")
    with open(str(cfg_file)) as f:
        return yaml.safe_load(f)


def _load_env_cfg(task_name: str):
    """Instantiate the env cfg class registered for this task."""
    spec = gym.spec(task_name)
    entry = spec.kwargs["env_cfg_entry_point"]
    module_path, cls_name = entry.rsplit(":", 1)
    import importlib
    mod = importlib.import_module(module_path)
    cfg_cls = getattr(mod, cls_name)
    return cfg_cls()


def _write_status_file(path: str, payload: dict[str, float | int | str]) -> None:
    out = Path(path).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".tmp")
    # Key=value format is easier for shell parsing; keep JSON-compatible values.
    lines = [f"{k}={v}" for k, v in payload.items()]
    tmp.write_text("\n".join(lines) + "\n")
    tmp.replace(out)


def main():
    env_cfg = _load_env_cfg(args_cli.task)
    agent_cfg = _load_agent_cfg()

    # CLI overrides
    if args_cli.num_envs is not None:
        env_cfg.scene.num_envs = args_cli.num_envs
        # minibatch_size must divide batch_size (num_envs * horizon_length = 512 * 16 = 8192).
        # Set minibatch_size = min(num_envs, 32768) clamped to power of 2, max = num_envs.
        num_envs_val = args_cli.num_envs
        agent_cfg["params"]["config"]["minibatch_size"] = num_envs_val
    # Seed diversity audit (B1): derive a per-policy seed so the population
    # explores genuinely independent rollouts. The +1000 offset spreads the
    # downstream torch / numpy / random RNGs far apart between workers.
    effective_seed = int(args_cli.seed) + 1000 * int(args_cli.policy_idx)
    print(
        f"[train] policy_idx={args_cli.policy_idx} base_seed={args_cli.seed} "
        f"effective_seed={effective_seed}",
        flush=True,
    )
    agent_cfg["params"]["seed"] = effective_seed
    if args_cli.max_iterations is not None:
        agent_cfg["params"]["config"]["max_epochs"] = args_cli.max_iterations
    if args_cli.checkpoint is not None:
        agent_cfg["params"]["load_checkpoint"] = True
        agent_cfg["params"]["load_path"] = args_cli.checkpoint
    train_sigma = float(args_cli.sigma) if args_cli.sigma is not None else None

    env_cfg.seed = effective_seed

    mutable_env_overrides = {
        "alpha_reach": args_cli.alpha_reach,
        "alpha_pick": args_cli.alpha_pick,
        "r_picked": args_cli.r_picked,
        "alpha_targ": args_cli.alpha_targ,
        "r_success": args_cli.r_success,
        "kuka_actions_penalty_scale": args_cli.kuka_actions_penalty_scale,
        "allegro_actions_penalty_scale": args_cli.allegro_actions_penalty_scale,
    }
    for key, value in mutable_env_overrides.items():
        if value is not None:
            setattr(env_cfg, key, value)

    # Blackwell + current rl_games/Triton stack is more stable without torch.compile.
    agent_cfg["params"]["config"]["torch_compile"] = False

    mutable_agent_overrides = {
        "gamma": args_cli.gamma,
        "kl_threshold": args_cli.kl_threshold,
        "grad_norm": args_cli.grad_norm,
        "critic_coef": args_cli.critic_coef,
        "bounds_loss_coef": args_cli.bounds_loss_coef,
        "e_clip": args_cli.e_clip,
        "mini_epochs": args_cli.mini_epochs,
        # M8: previously missing — PBT mutated learning_rate but the value never
        # reached agent_cfg, so all policies trained at the YAML default.
        "learning_rate": args_cli.learning_rate,
    }
    for key, value in mutable_agent_overrides.items():
        if value is not None:
            agent_cfg["params"]["config"][key] = value

    # logging
    config_name = agent_cfg["params"]["config"]["name"]
    log_root = os.path.abspath(args_cli.train_root) if args_cli.train_root else os.path.abspath(os.path.join("logs", "rl_games", config_name))
    log_dir = args_cli.experiment_name if args_cli.experiment_name else datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    agent_cfg["params"]["config"]["train_dir"] = log_root
    agent_cfg["params"]["config"]["full_experiment_name"] = log_dir
    print(f"[INFO] Logging to: {os.path.join(log_root, log_dir)}")
    dump_yaml(os.path.join(log_root, log_dir, "params", "agent.yaml"), agent_cfg)

    # read wrapper params
    rl_device = agent_cfg["params"]["config"]["device"]
    clip_obs = agent_cfg["params"]["env"].get("clip_observations", math.inf)
    clip_actions = agent_cfg["params"]["env"].get("clip_actions", math.inf)

    # create env
    env = gym.make(args_cli.task, cfg=env_cfg)

    # ----------------------------------------------------------------------
    # M3: PBT restart curriculum resume.
    #
    # After gym.make() the env's success_tolerance / initial_tolerance /
    # target_tolerance / last_curriculum_update / frame_since_restart are all
    # initialized from cfg defaults (e.g. 0.075 for regrasping/reorientation).
    # If this process was spawned by a PBT supervisor restart, pbt_observer
    # has serialized the source worker's curriculum state into CLI flags.
    # We override the env BEFORE training starts so the new policy continues
    # from the source's tolerance schedule instead of rewinding to cfg default.
    #
    # See `_save_workspace_checkpoint` + `_restart_process_with_new_params`
    # in dexpbt_isaaclab/pbt/pbt_observer.py.
    # ----------------------------------------------------------------------
    raw_env = env.unwrapped
    resume_log = {}
    if args_cli.success_tolerance_resume is not None and hasattr(raw_env, "success_tolerance"):
        raw_env.success_tolerance = float(args_cli.success_tolerance_resume)
        resume_log["success_tolerance"] = raw_env.success_tolerance
    if args_cli.initial_tolerance_resume is not None and hasattr(raw_env, "initial_tolerance"):
        raw_env.initial_tolerance = float(args_cli.initial_tolerance_resume)
        resume_log["initial_tolerance"] = raw_env.initial_tolerance
    if args_cli.target_tolerance_resume is not None and hasattr(raw_env, "target_tolerance"):
        raw_env.target_tolerance = float(args_cli.target_tolerance_resume)
        resume_log["target_tolerance"] = raw_env.target_tolerance
    if args_cli.last_curriculum_update_resume is not None and hasattr(raw_env, "last_curriculum_update"):
        raw_env.last_curriculum_update = int(args_cli.last_curriculum_update_resume)
        resume_log["last_curriculum_update"] = raw_env.last_curriculum_update
    if args_cli.frame_since_restart_resume is not None and hasattr(raw_env, "frame_since_restart"):
        raw_env.frame_since_restart = int(args_cli.frame_since_restart_resume)
        resume_log["frame_since_restart"] = raw_env.frame_since_restart
    # Patch H (2026-05-31): post-restart curriculum freeze (env stores steps).
    if (
        args_cli.curriculum_freeze_steps_after_restart_resume is not None
        and hasattr(raw_env, "curriculum_freeze_frames_after_restart")
    ):
        raw_env.curriculum_freeze_frames_after_restart = int(
            args_cli.curriculum_freeze_steps_after_restart_resume
        )
        resume_log["curriculum_freeze_steps_after_restart"] = (
            raw_env.curriculum_freeze_frames_after_restart
        )
    elif (
        args_cli.curriculum_freeze_frames_after_restart_resume is not None
        and hasattr(raw_env, "curriculum_freeze_frames_after_restart")
    ):
        from dexpbt_isaaclab.dexpbt_utils import env_frames_to_sim_steps

        raw_env.curriculum_freeze_frames_after_restart = env_frames_to_sim_steps(
            int(args_cli.curriculum_freeze_frames_after_restart_resume),
            int(args_cli.num_envs),
        )
        resume_log["curriculum_freeze_steps_after_restart"] = (
            raw_env.curriculum_freeze_frames_after_restart
        )
    if resume_log:
        src = args_cli.restart_source_policy
        print(
            f"[RESUME] PBT restart from source_policy={src}: "
            + ", ".join(f"{k}={v}" for k, v in resume_log.items())
        )

    if (
        not resume_log
        and args_cli.checkpoint
        and str(getattr(args_cli, "observer_mode", "") or "") == "pbt"
        and args_cli.restart_source_policy is None
        and args_cli.train_root
    ):
        from dexpbt_isaaclab.pbt.pbt_curriculum_p1 import (
            apply_curriculum_resume_env_state,
            load_curriculum_resume_snapshot,
            should_apply_supervisor_curriculum_resume,
        )

        policy_dir = Path(args_cli.train_root).resolve().parent
        snap = load_curriculum_resume_snapshot(policy_dir)
        init_tol = float(
            getattr(
                raw_env,
                "initial_tolerance",
                getattr(raw_env, "success_tolerance", 0.075),
            )
        )
        if should_apply_supervisor_curriculum_resume(
            has_checkpoint=True,
            success_tolerance_resume=args_cli.success_tolerance_resume,
            restart_source_policy=args_cli.restart_source_policy,
            env_success_tolerance=float(getattr(raw_env, "success_tolerance", init_tol)),
            env_initial_tolerance=init_tol,
            snapshot=snap,
        ):
            resume_log = apply_curriculum_resume_env_state(raw_env, snap["curriculum"])
            resume_log["resume_source"] = "supervisor_snapshot"
            resume_log["snapshot_env_frames"] = int(snap.get("env_frames", 0))
            print(
                "[RESUME] Supervisor checkpoint resume from curriculum snapshot "
                f"(env_frames={resume_log['snapshot_env_frames']}): "
                + ", ".join(
                    f"{k}={v}"
                    for k, v in resume_log.items()
                    if k not in ("resume_source", "snapshot_env_frames")
                )
            )

    # Patch P1 (2026-06-01): env reads mastery control JSON from workspace.
    if hasattr(raw_env, "curriculum_mode"):
        raw_env.curriculum_mode = args_cli.curriculum_mode
    if hasattr(raw_env, "curriculum_shrink_factor"):
        raw_env.curriculum_shrink_factor = float(args_cli.curriculum_shrink_factor)
    if hasattr(raw_env, "curriculum_control_path"):
        raw_env.curriculum_control_path = (
            Path(args_cli.workspace).resolve()
            / "curriculum_control"
            / f"{args_cli.policy_idx:03d}.json"
        )
    if hasattr(raw_env, "post_shrink_consolidation_env_frames"):
        raw_env.post_shrink_consolidation_env_frames = int(
            args_cli.post_shrink_consolidation_env_frames
        )
    if hasattr(raw_env, "pbt_single_shrink_per_window"):
        raw_env.pbt_single_shrink_per_window = bool(
            args_cli.pbt_single_shrink_per_window
        )

    start_time = time.time()

    # ------------------------------------------------------------------
    # World Model (WM) integration
    # ------------------------------------------------------------------
    wm_trainer = None
    wm_ckpt_manager = None
    seq_wm_trainer = None
    seq_aux_collector = None
    seq_wm_save_dir = None
    seq_aux_env_wrap = None
    awr_bridge = None
    awr_env_wrap = None
    seq_policy_coupling_cfg = SeqPolicyCouplingConfig(
        mode=args_cli.seq_wm_policy_coupling,
        weight=float(args_cli.seq_wm_policy_coupling_weight),
        value_weight=float(args_cli.seq_wm_policy_coupling_value_weight),
        success_weight=float(args_cli.seq_wm_policy_coupling_success_weight),
        warmup_frames=int(args_cli.seq_wm_policy_coupling_warmup_frames),
        feature_source=str(args_cli.seq_wm_distill_feature_source),
        teacher_value_norm=str(args_cli.seq_wm_teacher_value_norm),
        teacher_value_norm_min_std=float(
            args_cli.seq_wm_teacher_value_norm_min_std
        ),
        gate=str(args_cli.seq_wm_policy_coupling_gate),
        teacher_corr_threshold=float(args_cli.seq_wm_teacher_corr_threshold),
        ramp_frames=int(args_cli.seq_wm_policy_coupling_ramp_frames),
        aux_critic_subtower_hidden=int(
            args_cli.seq_wm_aux_critic_subtower_hidden
        ),
    )
    # FloorCritic config (Real-target). Default-off; enabled only by the
    # explicit --wm_floorcritic_enable flag.
    _fc_horizons = tuple(
        int(s.strip())
        for s in str(args_cli.wm_floorcritic_horizons or "").split(",")
        if s.strip()
    ) or (4, 8)
    floorcritic_cfg = FloorCriticConfig(
        enabled=bool(args_cli.wm_floorcritic_enable),
        feature_source=str(args_cli.wm_floorcritic_feature_source),
        max_grad_ratio=float(args_cli.wm_floorcritic_max_grad_ratio),
        auto_scale_by_grad_ratio=bool(
            args_cli.wm_floorcritic_auto_scale_by_grad_ratio
        ),
        start_tol=float(args_cli.wm_floorcritic_start_tol),
        floor_tol=float(args_cli.wm_floorcritic_floor_tol),
        hard_floor_tol=float(args_cli.wm_floorcritic_hard_floor_tol),
        horizons=_fc_horizons,
        weight=float(args_cli.wm_floorcritic_weight),
        floor_weight=float(args_cli.wm_floorcritic_floor_weight),
        hard_floor_weight=float(args_cli.wm_floorcritic_hard_floor_weight),
        near_success_mult=float(args_cli.wm_floorcritic_near_success_mult),
        dhat_improve_margin=float(args_cli.wm_floorcritic_dhat_improve_margin),
        drop_dhat_margin=float(args_cli.wm_floorcritic_drop_dhat_margin),
        actor_detach=bool(args_cli.wm_floorcritic_actor_detach),
        update_actor=bool(args_cli.wm_floorcritic_update_actor),
        use_success=bool(args_cli.wm_floorcritic_use_success),
        use_near_success=bool(args_cli.wm_floorcritic_use_near_success),
        use_dhat_improve=bool(args_cli.wm_floorcritic_use_dhat_improve),
        use_drop_risk=bool(args_cli.wm_floorcritic_use_drop_risk),
    )
    if floorcritic_cfg.enabled:
        # feature_source is authoritative; warn on conflicting compat flags.
        if (
            floorcritic_cfg.feature_source == "actor_mlp_out_shared"
            and bool(args_cli.wm_floorcritic_actor_detach)
        ):
            print(
                "[FloorCritic] WARNING: feature_source=actor_mlp_out_shared but "
                "actor_detach=true; feature_source wins -> NOT detaching "
                "(shared backbone will receive FC gradient)."
            )
        if floorcritic_cfg.feature_source == "critic_feature":
            print(
                "[FloorCritic] WARNING: feature_source=critic_feature has no "
                "populated critic feature under rl_games separate=False; "
                "degrading to actor_mlp_out_shared."
            )
    _erd_horizons = tuple(
        int(s.strip())
        for s in str(args_cli.erd_horizons or "").split(",")
        if s.strip()
    ) or (4, 8)
    erd_cfg = ERDConfig(
        enabled=bool(args_cli.erd_enable),
        horizons=_erd_horizons,
        candidate_k=int(args_cli.erd_candidate_k),
        action_noise_std=float(args_cli.erd_action_noise_std),
        max_action_delta=float(args_cli.erd_max_action_delta),
        noise_cap_by_tol=bool(args_cli.erd_noise_cap_by_tol),
        score_success_weight=float(args_cli.erd_score_success_weight),
        score_improve_weight=float(args_cli.erd_score_improve_weight),
        score_drop_weight=float(args_cli.erd_score_drop_weight),
        scorer_loss_weight=float(args_cli.erd_scorer_loss_weight),
        distill_loss_weight=float(args_cli.erd_distill_loss_weight),
        floor_distill_loss_weight=float(args_cli.erd_floor_distill_loss_weight),
        hard_floor_distill_loss_weight=float(args_cli.erd_hard_floor_distill_loss_weight),
        start_tol=float(args_cli.erd_start_tol),
        floor_tol=float(args_cli.erd_floor_tol),
        hard_floor_tol=float(args_cli.erd_hard_floor_tol),
        dhat_improve_margin=float(args_cli.erd_dhat_improve_margin),
        drop_dhat_margin=float(args_cli.erd_drop_dhat_margin),
        near_success_mult=float(args_cli.erd_near_success_mult),
        max_grad_ratio=float(args_cli.erd_max_grad_ratio),
        auto_scale_by_grad_ratio=bool(args_cli.erd_auto_scale_by_grad_ratio),
        only_lifted=bool(args_cli.erd_only_lifted),
        min_risky_fraction=float(args_cli.erd_min_risky_fraction),
        start_after_frames=int(args_cli.erd_start_after_frames),
        scorer_warmup_iters=int(args_cli.erd_scorer_warmup_iters),
        min_score_margin=float(args_cli.erd_min_score_margin),
        min_target_valid_fraction=float(args_cli.erd_min_target_valid_fraction),
        max_drop_prob_for_teacher=float(args_cli.erd_max_drop_prob_for_teacher),
    )
    is_sequence_strategy = (
        args_cli.enable_wm and args_cli.wm_strategy in SEQUENCE_STRATEGIES
    )

    if is_sequence_strategy:
        # ---------- Phase A4 sequence WM path ----------
        # Resolve obs_dim / action_dim from the live env (paper-mode obs is 92
        # for regrasping, 110 for reorientation; action_dim is 23 for KukaAllegro).
        try:
            obs_space = env.unwrapped.single_observation_space["policy"]
            obs_dim = int(obs_space.shape[0])
        except Exception:
            obs_dim = 92
        try:
            action_space = env.unwrapped.single_action_space
            action_dim = int(action_space.shape[0])
        except Exception:
            action_dim = 23
        erd_cfg.action_dim = int(action_dim)

        # Phase A4 P4 (2026-06-09): horizon list parser (Part B / D).
        def _parse_int_list(spec: str) -> tuple[int, ...]:
            return tuple(
                int(s.strip()) for s in str(spec or "").split(",") if s.strip()
            )

        horizon_steps_arg = _parse_int_list(args_cli.seq_wm_horizon_steps) or (8, 16)
        imagine_horizons_arg = _parse_int_list(args_cli.seq_wm_imagine_horizons) or (8, 16)

        seq_cfg = SeqWMConfig(
            seq_len=int(args_cli.seq_wm_seq_len),
            seq_batch_size=int(args_cli.seq_wm_seq_batch_size),
            updates_per_ppo_epoch=int(args_cli.seq_wm_updates_per_ppo),
            lr=float(args_cli.seq_wm_lr),
            grad_clip=float(args_cli.seq_wm_grad_clip),
            lambda_total=float(args_cli.seq_wm_lambda_total),
            lambda_state=float(args_cli.seq_wm_lambda_state),
            lambda_reward=float(args_cli.seq_wm_lambda_reward),
            lambda_done=float(args_cli.seq_wm_lambda_done),
            lambda_success=float(args_cli.seq_wm_lambda_success),
            lambda_task_error=float(args_cli.seq_wm_lambda_task_error),
            lambda_value=float(args_cli.seq_wm_lambda_value),
            value_detach_belief=bool(args_cli.seq_wm_value_detach_belief),
            beta_kl_start=float(args_cli.seq_wm_beta_kl_start),
            beta_kl_end=float(args_cli.seq_wm_beta_kl_end),
            beta_kl_warmup_frames=int(args_cli.seq_wm_beta_kl_warmup_frames),
            free_nats=float(args_cli.seq_wm_free_nats),
            # Phase A4 P4 (2026-06-09): task-aware extras + replay + teacher.
            use_task_aux_inputs=bool(args_cli.seq_wm_use_task_aux_inputs),
            aux_embed_dim=int(args_cli.seq_wm_aux_embed_dim),
            loss_preset=str(args_cli.seq_wm_loss_preset),
            floor_aware_weighting=bool(args_cli.seq_wm_floor_aware_weighting),
            floor_tol_threshold=float(args_cli.seq_wm_floor_tol_threshold),
            floor_tol_floor=float(args_cli.seq_wm_floor_tol_floor),
            floor_weight_mid=float(args_cli.seq_wm_floor_weight_mid),
            floor_weight_floor=float(args_cli.seq_wm_floor_weight_floor),
            use_horizon_heads=bool(args_cli.seq_wm_use_horizon_heads),
            horizon_steps=horizon_steps_arg,
            use_recovery_head=bool(args_cli.seq_wm_use_recovery_head),
            recovery_horizon=int(args_cli.seq_wm_recovery_horizon),
            lambda_horizon_success=float(args_cli.seq_wm_lambda_horizon_success),
            lambda_drop_risk=float(args_cli.seq_wm_lambda_drop_risk),
            lambda_recovery=float(args_cli.seq_wm_lambda_recovery),
            head_balance_enable=bool(args_cli.seq_wm_head_balance_enable),
            success_pos_weight=float(args_cli.seq_wm_success_pos_weight),
            drop_pos_weight=float(args_cli.seq_wm_drop_pos_weight),
            recovery_pos_weight=float(args_cli.seq_wm_recovery_pos_weight),
            head_pos_weight_auto=bool(args_cli.seq_wm_head_pos_weight_auto),
            head_pos_weight_max=float(args_cli.seq_wm_head_pos_weight_max),
            recovery_d_hat_margin=float(args_cli.seq_wm_recovery_d_hat_margin),
            recovery_d_hat_tol_min_steps=int(
                args_cli.seq_wm_recovery_d_hat_tol_min_steps
            ),
            recovery_shadow_tol_threshold=float(
                args_cli.seq_wm_recovery_shadow_tol_threshold
            ),
            horizon_skip_too_long=bool(args_cli.seq_wm_horizon_skip_too_long),
            replay_prioritize_floor_recovery=bool(
                args_cli.seq_wm_replay_prioritize_floor_recovery
            ),
            replay_floor_weight=float(args_cli.seq_wm_replay_floor_weight),
            replay_hard_floor_weight=float(
                args_cli.seq_wm_replay_hard_floor_weight
            ),
            replay_positive_slope_weight=float(
                args_cli.seq_wm_replay_positive_slope_weight
            ),
            replay_drop_near_success_weight=float(
                args_cli.seq_wm_replay_drop_near_success_weight
            ),
            replay_uniform_floor=float(args_cli.seq_wm_replay_uniform_floor),
            enable_iterative_teacher=bool(
                args_cli.seq_wm_enable_iterative_teacher
            ),
            teacher_eval_interval_frames=int(
                args_cli.seq_wm_teacher_eval_interval_frames
            ),
            teacher_promote_min_improvement=float(
                args_cli.seq_wm_teacher_promote_min_improvement
            ),
            teacher_keep_last_k=int(args_cli.seq_wm_teacher_keep_last_k),
            teacher_metric=str(args_cli.seq_wm_teacher_metric),
            enable_imagined_evaluator=bool(
                args_cli.seq_wm_enable_imagined_evaluator
            ),
            imagine_horizons=imagine_horizons_arg,
            imagine_action_mode=str(args_cli.seq_wm_imagine_action_mode),
            imagine_num_starts=int(args_cli.seq_wm_imagine_num_starts),
            imagine_use_teacher=bool(args_cli.seq_wm_imagine_use_teacher),
        )
        with_value = args_cli.wm_strategy in ("rssm_belief_value", "trans_belief_value")
        # Aux dim is determined by the sampler at runtime (12 columns); the
        # RSSM uses 12 only when use_task_aux_inputs is True. The default
        # zero falls back to the legacy posterior input shape.
        rssm_aux_dim = 12 if seq_cfg.use_task_aux_inputs else 0
        _rssm_model_kwargs = dict(
            obs_dim=obs_dim,
            action_dim=action_dim,
            free_nats=seq_cfg.free_nats,
            with_value_head=with_value,
            aux_dim=rssm_aux_dim,
            aux_embed_dim=seq_cfg.aux_embed_dim,
            with_horizon_heads=seq_cfg.use_horizon_heads,
            horizon_steps=seq_cfg.horizon_steps,
            with_recovery_head=seq_cfg.use_recovery_head,
            recovery_horizon=seq_cfg.recovery_horizon,
        )
        if args_cli.wm_strategy in RSSM_FAMILY:
            seq_model = RSSMLiteWorldModel(**_rssm_model_kwargs)
        elif args_cli.wm_strategy in TRANS_FAMILY:
            seq_model = TransformerLiteWorldModel(
                obs_dim=obs_dim,
                action_dim=action_dim,
                max_seq_len=int(args_cli.seq_wm_max_seq_len),
                with_value_head=with_value,
            )
        else:
            raise ValueError(
                f"unreachable: sequence strategy {args_cli.wm_strategy} not "
                "in RSSM_FAMILY ∪ TRANS_FAMILY"
            )

        seq_wm_trainer = SeqWMTrainer(
            model=seq_model,
            cfg=seq_cfg,
            device=rl_device,
            strategy=args_cli.wm_strategy,
        )
        _wm_cal_cfg = WMCalibrationConfig(
            enabled=bool(args_cli.wm_calibration_enable),
            eval_interval_frames=int(args_cli.wm_calibration_eval_interval_frames),
            min_samples=int(args_cli.wm_calibration_min_samples),
            heldout_fraction=float(args_cli.wm_calibration_heldout_fraction),
            alpha_min=float(args_cli.wm_calibration_alpha_min),
            alpha_max=float(args_cli.wm_calibration_alpha_max),
            gate_auc_recovery=float(args_cli.wm_calibration_gate_auc_recovery),
            gate_ece_max=float(args_cli.wm_calibration_gate_ece_max),
            gate_rank_corr_min=float(args_cli.wm_calibration_gate_rank_corr_min),
        )
        seq_wm_trainer._wm_calibration_evaluator = (
            WMCalibrationEvaluator(_wm_cal_cfg) if _wm_cal_cfg.enabled else None
        )

        # ------------------------------------------------------------------
        # Full RSSM ensemble (MPPI scoring only). Member 0 is the primary
        # trainer above; members 1..E-1 are independent RSSM models with their
        # own trainers/optimizers. Diversity comes from independent init and
        # independently sampled sequence minibatches (per-sample bootstrap masks
        # are NOT active; see seq_wm_ensemble/bootstrap_mask_active).
        # ------------------------------------------------------------------
        _ens_enable = bool(args_cli.seq_wm_ensemble_enable)
        _ens_size = max(1, int(args_cli.seq_wm_ensemble_size))
        seq_wm_trainer._rssm_ensemble_manager = None
        seq_wm_trainer._rssm_ensemble_trainers = [seq_wm_trainer]
        if _ens_enable and _ens_size > 1 and args_cli.wm_strategy in RSSM_FAMILY:
            ensemble_cfg = RSSMEnsembleConfig(
                enabled=True,
                size=_ens_size,
                use_bootstrap_masks=bool(args_cli.seq_wm_ensemble_bootstrap),
                bootstrap_keep_prob=float(args_cli.seq_wm_ensemble_bootstrap_keep_prob),
                device=str(rl_device),
                save_ensemble=bool(args_cli.seq_wm_ensemble_save),
                load_mode=str(args_cli.seq_wm_ensemble_load_mode),
            )
            ensemble_trainers = [seq_wm_trainer]
            for _e in range(1, _ens_size):
                # Each member is an independently-initialised RSSM with its own
                # trainer/optimizer. Members are stepped sequentially so each
                # .sample() draws an independent sequence minibatch.
                _member_model = RSSMLiteWorldModel(**_rssm_model_kwargs)
                _member_trainer = SeqWMTrainer(
                    model=_member_model,
                    cfg=seq_cfg,
                    device=rl_device,
                    strategy=args_cli.wm_strategy,
                )
                ensemble_trainers.append(_member_trainer)
            seq_wm_trainer._rssm_ensemble_manager = RSSMEnsembleManager(
                [t.model for t in ensemble_trainers], ensemble_cfg
            )
            seq_wm_trainer._rssm_ensemble_trainers = ensemble_trainers
            print(
                f"[RSSM-ensemble] enabled: size={_ens_size} "
                f"independent_minibatch_ensemble=True "
                f"bootstrap_mask_active=False"
            )
        elif _ens_enable and _ens_size > 1 and args_cli.wm_strategy not in RSSM_FAMILY:
            print(
                "[RSSM-ensemble] requested but backbone is not RSSM family; "
                "ensemble disabled (single model)."
            )

        # FloorCritic: attach cfg + (lazy) bridge/target-builder on the trainer
        # so both the runner wiring (head/hook/patch) and the patched
        # play_steps_rnn (per-rollout target cache) can reach them. Bridge is
        # created in _WmEnabledRunner.run_train once the agent exists.
        seq_wm_trainer._floorcritic_cfg = floorcritic_cfg
        seq_wm_trainer._floorcritic_bridge = None
        seq_wm_trainer._floorcritic_target_builder = (
            FloorCriticTargetBuilder(floorcritic_cfg)
            if floorcritic_cfg.enabled
            else None
        )
        seq_wm_trainer._floorcritic_generation = 0

        seq_wm_trainer._erd_cfg = erd_cfg
        seq_wm_trainer._erd_bridge = None
        seq_wm_trainer._erd_target_builder = (
            ERDTargetBuilder(erd_cfg) if erd_cfg.enabled else None
        )
        seq_wm_trainer._erd_generation = 0

        # ------------------------------------------------------------------
        # EIR-MPC config assembly. EIR reuses the ERD action scorer + target
        # builder (same flatten convention) but routes the actor-improving
        # signal through the gated residual head and harvests an elite bank.
        # When EIR is enabled it SUPERSEDES the ERD raw-mu distill path.
        # ------------------------------------------------------------------
        _eir_horizons = tuple(
            int(s.strip()) for s in str(args_cli.eir_horizons or "").split(",") if s.strip()
        ) or (4, 8)
        _eir_feature_dim = int(getattr(erd_cfg, "feature_dim", 256))
        eir_enabled = bool(args_cli.eir_enable)
        validate_actor_distill_v1_flags(
            feature_detach=bool(args_cli.eir_actor_distill_feature_detach),
            mu_head_only=bool(args_cli.eir_actor_distill_mu_head_only),
        )
        _actor_distill_allowed = parse_actor_distill_allowed_sources(
            str(args_cli.eir_actor_distill_allowed_sources)
        )
        # EIR borrows an ERD scorer config (same scorer architecture / targets).
        eir_scorer_cfg = ERDConfig(
            enabled=eir_enabled,
            horizons=_eir_horizons,
            score_success_weight=float(args_cli.eir_score_success_weight),
            score_improve_weight=float(args_cli.eir_score_improve_weight),
            score_drop_weight=float(args_cli.eir_score_drop_weight),
            start_tol=float(args_cli.eir_start_tol),
            floor_tol=float(args_cli.eir_floor_tol),
            hard_floor_tol=float(args_cli.eir_hard_floor_tol),
            near_success_mult=float(args_cli.eir_near_success_mult),
            only_lifted=bool(args_cli.eir_only_lifted),
            min_risky_fraction=float(args_cli.eir_min_risky_fraction),
            feature_dim=_eir_feature_dim,
            action_dim=int(action_dim),
        )
        eir_cfg = EIRConfig(
            enabled=eir_enabled,
            horizons=_eir_horizons,
            action_dim=int(action_dim),
            feature_dim=_eir_feature_dim,
            local_candidate_k=int(args_cli.eir_local_candidate_k),
            elite_candidate_k=int(args_cli.eir_elite_candidate_k) if bool(args_cli.eir_use_elite_bank) else 0,
            archive_candidate_k=int(args_cli.eir_archive_candidate_k) if bool(args_cli.eir_use_elite_bank) else 0,
            action_noise_std=float(args_cli.eir_action_noise_std),
            max_action_delta=float(args_cli.eir_max_action_delta),
            noise_cap_by_tol=bool(args_cli.eir_noise_cap_by_tol),
            score_success_weight=float(args_cli.eir_score_success_weight),
            score_improve_weight=float(args_cli.eir_score_improve_weight),
            score_recovery_weight=float(args_cli.eir_score_recovery_weight),
            score_drop_weight=float(args_cli.eir_score_drop_weight),
            elite_priority_weight=float(args_cli.eir_elite_priority_weight),
            elite_similarity_weight=float(args_cli.eir_elite_similarity_weight),
            action_delta_weight=float(args_cli.eir_action_delta_weight),
            clip_penalty_weight=float(args_cli.eir_clip_penalty_weight),
            min_score_margin=float(args_cli.eir_min_score_margin),
            min_elite_margin=float(args_cli.eir_min_elite_margin),
            max_drop_prob_for_teacher=float(args_cli.eir_max_drop_prob_for_teacher),
            start_tol=float(args_cli.eir_start_tol),
            floor_tol=float(args_cli.eir_floor_tol),
            hard_floor_tol=float(args_cli.eir_hard_floor_tol),
            near_success_mult=float(args_cli.eir_near_success_mult),
            only_lifted=bool(args_cli.eir_only_lifted),
            scorer_loss_weight=float(args_cli.eir_scorer_loss_weight),
            start_after_frames=int(args_cli.eir_start_after_frames),
            scorer_warmup_iters=int(args_cli.eir_scorer_warmup_iters),
            min_target_valid_fraction=float(args_cli.eir_min_target_valid_fraction),
            min_risky_fraction=float(args_cli.eir_min_risky_fraction),
            guidance_enable=bool(args_cli.eir_guidance_enable),
            guidance_start_tol=float(args_cli.eir_guidance_start_tol),
            guidance_rssm_source_bonus=float(args_cli.eir_guidance_rssm_source_bonus),
            guidance_rssm_bonus_margin_scale=float(args_cli.eir_guidance_rssm_bonus_margin_scale),
            guidance_rssm_bonus_uncertainty_penalty=float(
                args_cli.eir_guidance_rssm_bonus_uncertainty_penalty
            ),
            actor_distill_enable=bool(args_cli.eir_actor_distill_enable),
            actor_distill_weight=float(args_cli.eir_actor_distill_weight),
            actor_distill_start_after_frames=int(args_cli.eir_actor_distill_start_after_frames),
            actor_distill_tol=float(args_cli.eir_actor_distill_tol),
            actor_distill_min_margin=float(args_cli.eir_actor_distill_min_margin),
            actor_distill_margin_norm=float(args_cli.eir_actor_distill_margin_norm),
            actor_distill_max_delta=float(args_cli.eir_actor_distill_max_delta),
            actor_distill_uncertainty_threshold=float(
                args_cli.eir_actor_distill_uncertainty_threshold
            ),
            actor_distill_loss_cap=float(args_cli.eir_actor_distill_loss_cap),
            actor_distill_feature_detach=bool(args_cli.eir_actor_distill_feature_detach),
            actor_distill_mu_head_only=bool(args_cli.eir_actor_distill_mu_head_only),
            allowed_source_ids=_actor_distill_allowed,
            actor_distill_activation_enable=bool(args_cli.wm_activation_enable),
            actor_distill_gate_mode=str(args_cli.eir_actor_distill_gate_mode),
            actor_distill_tol_start=float(args_cli.eir_actor_distill_tol_start),
            actor_distill_tol_full=float(args_cli.eir_actor_distill_tol_full),
            actor_distill_ramp_frames=int(args_cli.eir_actor_distill_ramp_frames),
            actor_distill_nsucc_ref=float(args_cli.eir_actor_distill_nsucc_ref),
            candidate_gate_mode=str(args_cli.eir_candidate_gate_mode),
            elite_candidate_tol_start=float(args_cli.eir_elite_candidate_tol_start),
            elite_candidate_tol_full=float(args_cli.eir_elite_candidate_tol_full),
            rssm_candidate_tol_start=float(args_cli.eir_rssm_candidate_tol_start),
            rssm_candidate_tol_full=float(args_cli.eir_rssm_candidate_tol_full),
            conservative_rssm_candidate_tol_start=float(
                args_cli.eir_conservative_rssm_candidate_tol_start
            ),
            conservative_rssm_candidate_tol_full=float(
                args_cli.eir_conservative_rssm_candidate_tol_full
            ),
            candidate_min_weight_for_retrieval=float(
                args_cli.eir_candidate_min_weight_for_retrieval
            ),
            wm_strategy_profile_name=str(
                getattr(args_cli, "wm_strategy_profile", "") or ""
            ),
        )
        eir_bank_cfg = EliteRecoveryBankConfig(
            enabled=eir_enabled and bool(args_cli.eir_use_elite_bank),
            bank_size=int(args_cli.eir_elite_bank_size),
            feature_dim=_eir_feature_dim,
            action_dim=int(action_dim),
            horizons=_eir_horizons,
            sample_pool=int(args_cli.eir_elite_sample_pool),
            knn_topk=int(args_cli.eir_elite_knn_topk),
            tol_threshold=float(args_cli.eir_elite_tol_threshold),
            hard_tol_threshold=float(args_cli.eir_elite_hard_tol_threshold),
            near_success_mult=float(args_cli.eir_elite_near_success_mult),
            min_priority=float(args_cli.eir_elite_min_priority),
            guidance_enable=bool(args_cli.eir_guidance_enable),
            guidance_bank_tol=float(args_cli.eir_guidance_bank_tol),
            guidance_mid_near_success_mult=float(args_cli.eir_guidance_mid_near_success_mult),
            guidance_mid_priority_scale=float(args_cli.eir_guidance_mid_priority_scale),
        )
        eir_snippet_bank_cfg = EliteSnippetBankConfig(
            enabled=eir_enabled and bool(args_cli.elite_snippet_enable),
            bank_size=int(args_cli.elite_snippet_bank_size),
            feature_dim=_eir_feature_dim,
            action_dim=int(action_dim),
            horizon=int(args_cli.elite_snippet_horizon),
            horizons=_eir_horizons,
            sample_pool=int(args_cli.elite_snippet_sample_pool),
            knn_topk=int(args_cli.elite_snippet_knn_topk),
            tol_threshold=float(args_cli.eir_elite_tol_threshold),
            hard_tol_threshold=float(args_cli.eir_elite_hard_tol_threshold),
            near_success_mult=float(args_cli.eir_elite_near_success_mult),
            min_priority=float(args_cli.elite_snippet_min_priority),
            guidance_enable=bool(args_cli.eir_guidance_enable),
            guidance_bank_tol=float(args_cli.eir_guidance_bank_tol),
            guidance_mid_near_success_mult=float(args_cli.eir_guidance_mid_near_success_mult),
            guidance_mid_priority_scale=float(args_cli.eir_guidance_mid_priority_scale),
        )
        snippet_distill_cfg = SnippetDistillConfig(
            enabled=eir_enabled and bool(args_cli.snippet_distill_enable),
            weight=float(args_cli.snippet_distill_weight),
            loss_cap=float(args_cli.snippet_distill_loss_cap),
            start_after_frames=int(args_cli.snippet_distill_start_after_frames),
            topk=int(args_cli.snippet_distill_topk),
            gate_mode=str(args_cli.snippet_distill_gate_mode),
            tol_start=float(args_cli.snippet_distill_tol_start),
            tol_full=float(args_cli.snippet_distill_tol_full),
            ramp_frames=int(args_cli.snippet_distill_ramp_frames),
        )
        snippet_wm_rerank_cfg = SnippetWMRerankConfig(
            enabled=eir_enabled and bool(args_cli.snippet_wm_rerank_enable),
            w_rec=float(args_cli.snippet_wm_rerank_w_rec),
            w_success=float(args_cli.snippet_wm_rerank_w_success),
            w_drop=float(args_cli.snippet_wm_rerank_w_drop),
            w_unc=float(args_cli.snippet_wm_rerank_w_unc),
            w_delta=float(args_cli.snippet_wm_rerank_w_delta),
            score_cap=float(args_cli.snippet_wm_rerank_score_cap),
        )
        eir_residual_cfg = RecoveryResidualConfig(
            enabled=eir_enabled and bool(args_cli.eir_use_recovery_residual),
            feature_dim=_eir_feature_dim,
            action_dim=int(action_dim),
            residual_scale=float(args_cli.eir_residual_scale),
            mu_base_detach=bool(args_cli.eir_mu_base_detach),
            distill_weight=float(args_cli.eir_residual_distill_weight),
            floor_distill_weight=float(args_cli.eir_residual_floor_distill_weight),
            hard_floor_distill_weight=float(args_cli.eir_residual_hard_floor_distill_weight),
            gate_sparsity_weight=float(args_cli.eir_gate_sparsity_weight),
            delta_l2_weight=float(args_cli.eir_delta_l2_weight),
            apply_to_rollout=bool(args_cli.eir_residual_apply_to_rollout),
            apply_after_frames=int(args_cli.eir_residual_apply_after_frames),
            apply_alpha_mid=float(args_cli.eir_residual_apply_alpha_mid),
            apply_alpha_floor=float(args_cli.eir_residual_apply_alpha_floor),
            apply_alpha_hard_floor=float(args_cli.eir_residual_apply_alpha_hard_floor),
            only_lifted=bool(args_cli.eir_only_lifted),
            apply_tol_threshold=float(args_cli.eir_residual_apply_tol_threshold),
            floor_tol=float(args_cli.eir_floor_tol),
            hard_floor_tol=float(args_cli.eir_hard_floor_tol),
            near_success_mult=float(args_cli.eir_residual_near_success_mult),
            gate_confidence_threshold=float(args_cli.eir_residual_gate_confidence_threshold),
            apply_max_aux_age_frames=int(args_cli.eir_residual_apply_max_aux_age_frames),
            guidance_enable=bool(args_cli.eir_guidance_enable),
            guidance_apply_tol=float(args_cli.eir_guidance_apply_tol),
            guidance_mid_near_success_mult=float(args_cli.eir_guidance_mid_near_success_mult),
            guidance_mid_alpha=float(args_cli.eir_guidance_mid_residual_alpha),
            guidance_mid_gate_confidence_threshold=float(
                args_cli.eir_guidance_mid_gate_confidence_threshold
            ),
        )
        _mppi_mode = str(args_cli.eir_mppi_mode)
        _mppi_on = eir_enabled and bool(args_cli.eir_use_rssm_mppi) and _mppi_mode != "off"
        # light reranker stays available as the fallback path (used when the
        # full planner is disabled or cannot produce a belief).
        eir_rssm_cfg = RSSMRerankConfig(
            enabled=_mppi_on,
            start_after_frames=int(args_cli.eir_mppi_start_after_frames),
            horizon=int(args_cli.eir_mppi_horizon),
            num_sequences=int(args_cli.eir_mppi_num_sequences),
            uncertainty_threshold=float(args_cli.eir_mppi_uncertainty_threshold),
        )
        eir_mppi_cfg = RSSMMPPIConfig(
            enabled=_mppi_on and _mppi_mode == "full",
            mode=_mppi_mode,
            start_after_frames=int(args_cli.eir_mppi_start_after_frames),
            horizon=int(args_cli.eir_mppi_horizon),
            num_sequences=int(args_cli.eir_mppi_num_sequences),
            num_elite=int(args_cli.eir_mppi_num_elite),
            cem_iters=int(args_cli.eir_mppi_cem_iters),
            action_noise_std=float(args_cli.eir_mppi_action_noise_std),
            action_noise_decay=float(args_cli.eir_mppi_action_noise_decay),
            max_action_delta=float(args_cli.eir_mppi_max_action_delta),
            smooth_noise=bool(args_cli.eir_mppi_smooth_noise),
            uncertainty_threshold=float(args_cli.eir_mppi_uncertainty_threshold),
            min_sequence_margin=float(args_cli.eir_mppi_min_sequence_margin),
            w_success=float(args_cli.eir_mppi_w_success),
            w_recovery=float(args_cli.eir_mppi_w_recovery),
            w_drop=float(args_cli.eir_mppi_w_drop),
            w_value=float(args_cli.eir_mppi_w_value),
            w_task_improve=float(args_cli.eir_mppi_w_task_improve),
            w_uncertainty=float(args_cli.eir_mppi_w_uncertainty),
            w_action_smoothness=float(args_cli.eir_mppi_w_action_smoothness),
            w_action_delta=float(args_cli.eir_mppi_w_action_delta),
            fail_fast=bool(args_cli.eir_mppi_fail_fast),
            stochastic_rollouts=bool(args_cli.eir_mppi_stochastic_rollouts),
            num_latent_samples=int(args_cli.eir_mppi_num_latent_samples),
            sample_temperature=float(args_cli.eir_mppi_sample_temperature),
            risk_beta=float(args_cli.eir_mppi_risk_beta),
            drop_cvar_alpha=float(args_cli.eir_mppi_drop_cvar_alpha),
            drop_cvar_weight=float(args_cli.eir_mppi_drop_cvar_weight),
            uncertainty_cvar_weight=float(args_cli.eir_mppi_uncertainty_cvar_weight),
            score_aggregation=str(args_cli.eir_mppi_score_aggregation),
            max_rollout_batch=int(args_cli.eir_mppi_max_rollout_batch),
            use_model_ensemble=bool(args_cli.eir_mppi_use_model_ensemble),
            ensemble_size=int(args_cli.seq_wm_ensemble_size),
            ensemble_risk_beta=float(args_cli.eir_mppi_ensemble_risk_beta),
            ensemble_drop_weight=float(args_cli.eir_mppi_ensemble_drop_weight),
            ensemble_uncertainty_weight=float(args_cli.eir_mppi_ensemble_uncertainty_weight),
            ensemble_min_valid_models=int(args_cli.eir_mppi_ensemble_min_valid_models),
            ensemble_max_models_per_plan=int(args_cli.eir_mppi_ensemble_max_models_per_plan),
            ensemble_uncertainty_threshold=float(args_cli.eir_mppi_ensemble_uncertainty_threshold),
            ensemble_uncertainty_allow_margin=float(args_cli.eir_mppi_ensemble_uncertainty_allow_margin),
        )
        eir_broadcast_cfg = EliteBroadcastConfig(
            enabled=eir_enabled
            and bool(args_cli.eir_use_elite_bank)
            and bool(args_cli.eir_pbt_broadcast_elite_bank),
            threshold_nsucc_mean=float(args_cli.eir_pbt_broadcast_threshold_nsucc_mean),
            gap_over_pop_mean=float(args_cli.eir_pbt_broadcast_gap_over_pop_mean),
            bottom_k=int(args_cli.eir_pbt_broadcast_to_bottom_k),
            max_bank_merge_entries=int(args_cli.eir_pbt_max_bank_merge_entries),
            freeze_elite_curriculum_frames=int(
                args_cli.eir_pbt_freeze_elite_curriculum_frames
            ),
            min_frames_between_broadcasts_per_source=int(
                args_cli.eir_pbt_min_frames_between_broadcasts_per_source
            ),
            min_frames_between_broadcasts_per_target=int(
                args_cli.eir_pbt_min_frames_between_broadcasts_per_target
            ),
        )
        seq_wm_trainer._eir_cfg = eir_cfg
        seq_wm_trainer._eir_scorer_cfg = eir_scorer_cfg
        seq_wm_trainer._eir_bank_cfg = eir_bank_cfg
        seq_wm_trainer._eir_snippet_bank_cfg = eir_snippet_bank_cfg
        seq_wm_trainer._snippet_distill_cfg = snippet_distill_cfg
        seq_wm_trainer._snippet_wm_rerank_cfg = snippet_wm_rerank_cfg
        seq_wm_trainer._eir_residual_cfg = eir_residual_cfg
        seq_wm_trainer._eir_rssm_cfg = eir_rssm_cfg
        seq_wm_trainer._eir_mppi_cfg = eir_mppi_cfg
        seq_wm_trainer._eir_broadcast_cfg = eir_broadcast_cfg
        seq_wm_trainer._eir_bridge = None
        seq_wm_trainer._eir_target_builder = (
            ERDTargetBuilder(eir_scorer_cfg) if eir_enabled else None
        )
        seq_wm_trainer._snippet_target_builder = (
            SnippetTargetBuilder(
                horizon=int(args_cli.elite_snippet_horizon),
                horizons=_eir_horizons,
            )
            if eir_enabled and bool(args_cli.elite_snippet_enable)
            else None
        )

        # ------------------------------------------------------------------
        # Phase A4 P4 wired (2026-06-10): runtime wiring for the iterative
        # teacher manager (Part C) + short-horizon imagined evaluator
        # (Part D). Each is independently gated by its CLI flag and is a
        # no-op when disabled. Both write per-policy JSON under
        # ``<workspace>/wm_imagined/`` and ``<workspace>/world_model_teachers/``;
        # PBT observers in other processes read those files at ranking time.
        # ------------------------------------------------------------------
        try:
            workspace_root = Path(args_cli.workspace).resolve()
            seq_wm_trainer._teacher_manager = TeacherManager(
                workspace_root=workspace_root,
                policy_idx=int(args_cli.policy_idx),
                enabled=bool(args_cli.seq_wm_enable_iterative_teacher),
                eval_interval_frames=int(
                    args_cli.seq_wm_teacher_eval_interval_frames
                ),
                promote_min_improvement=float(
                    args_cli.seq_wm_teacher_promote_min_improvement
                ),
                keep_last_k=int(args_cli.seq_wm_teacher_keep_last_k),
                metric=str(args_cli.seq_wm_teacher_metric),
                max_uncertainty=float(args_cli.seq_wm_teacher_max_uncertainty),
                require_recovery_pos_in=(
                    float(args_cli.seq_wm_teacher_require_recovery_pos_low),
                    float(args_cli.seq_wm_teacher_require_recovery_pos_high),
                ),
                # Phase A4 P5 (2026-06-10): gate-fix knobs.
                allow_first_promotion=bool(
                    args_cli.seq_wm_teacher_allow_first_promotion
                ),
                min_eval_size=int(args_cli.seq_wm_teacher_min_eval_size),
                pos_rate_low=float(args_cli.seq_wm_teacher_pos_rate_low),
                pos_rate_high=float(args_cli.seq_wm_teacher_pos_rate_high),
            )
            seq_wm_trainer._floor_eval_buffer = FloorRecoveryEvalBuffer(
                capacity=int(args_cli.seq_wm_teacher_eval_buffer_capacity)
            )
            seq_wm_trainer._teacher_record_below_tol = float(
                args_cli.seq_wm_teacher_record_below_tol
            )
            seq_wm_trainer._imagined_evaluator = WMImaginedEvaluator(
                enabled=bool(args_cli.seq_wm_enable_imagined_evaluator),
                horizons=imagine_horizons_arg,
                action_mode=str(args_cli.seq_wm_imagine_action_mode),
                num_starts=int(args_cli.seq_wm_imagine_num_starts),
                use_teacher=bool(args_cli.seq_wm_imagine_use_teacher),
                max_uncertainty=float(args_cli.pbt_wm_max_uncertainty),
                valid_mode=str(args_cli.wm_imagined_valid_mode),
                strict_uncertainty_threshold=float(
                    args_cli.wm_imagined_valid_uncertainty_threshold
                ),
                soft_uncertainty_threshold=float(
                    args_cli.wm_imagined_soft_uncertainty_threshold
                ),
                min_logit_abs_mean=float(args_cli.wm_imagined_min_logit_abs_mean),
                min_head_signal_std=float(args_cli.wm_imagined_min_head_signal_std),
                soft_validity_weight=float(args_cli.wm_imagined_soft_validity_weight),
            )
            # Per-process audit counters surfaced into TB.
            seq_wm_trainer._teacher_eval_attempted_total = 0
            seq_wm_trainer._teacher_promoted_total = 0
            seq_wm_trainer._teacher_no_eval_data_total = 0
            seq_wm_trainer._teacher_target_degenerate_total = 0
            seq_wm_trainer._imagine_valid_total = 0
            seq_wm_trainer._imagine_invalid_total = 0
            seq_wm_trainer._last_imagine_invalid_reason = ""
            # Disk-backed cross-process imagined eval mailbox.
            seq_wm_trainer._wm_imagine_dir = workspace_root / "wm_imagined"
            if seq_wm_trainer._imagined_evaluator.enabled:
                seq_wm_trainer._wm_imagine_dir.mkdir(parents=True, exist_ok=True)
        except Exception as _wire_exc:
            print(
                f"[wm-runtime-wiring] WARNING: teacher/imagined wiring "
                f"failed to construct: {_wire_exc!r}; downstream PBT will "
                f"fall back to real-only ranking."
            )
            seq_wm_trainer._teacher_manager = None
            seq_wm_trainer._floor_eval_buffer = None
            seq_wm_trainer._imagined_evaluator = None
            seq_wm_trainer._teacher_record_below_tol = 0.04
            seq_wm_trainer._teacher_eval_attempted_total = 0
            seq_wm_trainer._teacher_promoted_total = 0
            seq_wm_trainer._teacher_no_eval_data_total = 0
            seq_wm_trainer._teacher_target_degenerate_total = 0
            seq_wm_trainer._imagine_valid_total = 0
            seq_wm_trainer._imagine_invalid_total = 0
            seq_wm_trainer._last_imagine_invalid_reason = "wiring_error"
            seq_wm_trainer._wm_imagine_dir = None

        # Aux collector + env wrap (must happen BEFORE RlGamesVecEnvWrapper).
        # Phase A4 P5b (2026-06-10): register/collect task-aware aux fields
        # (incl. aux_success_tolerance) whenever task aux OR the iterative
        # teacher stack is enabled. Without success_tolerance in the PPO
        # buffer the FloorRecoveryEvalBuffer never fills and teacher eval
        # always rejects with no_eval_data.
        seq_aux_fields = (
            EXTENDED_SEQ_AUX_FIELDS
            if (
                bool(args_cli.seq_wm_use_task_aux_inputs)
                or bool(args_cli.seq_wm_enable_iterative_teacher)
                or bool(args_cli.wm_floorcritic_enable)
                or bool(args_cli.erd_enable)
                or bool(args_cli.eir_enable)
            )
            else DEFAULT_SEQ_AUX_FIELDS
        )
        seq_aux_collector = SeqAuxCollector(
            num_envs=env.unwrapped.num_envs,
            horizon_length=int(agent_cfg["params"]["config"].get("horizon_length", 16)),
            device=rl_device,
            fields=seq_aux_fields,
        )

        def _seq_env_attrs(e):
            # Used by SeqAuxInterceptorEnv to read per-env env attributes
            # (lifted_object, successes, success_tolerance, ...) that are
            # not exposed via infos["wm"].
            inner = getattr(e, "unwrapped", e)
            d: dict = {}
            for name in (
                "lifted_object",
                "successes",
                "success_tolerance",
                "closest_d_hat",
            ):
                v = getattr(inner, name, None)
                if v is not None:
                    d[name] = v
            if "lifted_object" in d:
                d["picked"] = d["lifted_object"]
            return d

        env = SeqAuxInterceptorEnv(env, seq_aux_collector,
                                    env_attrs_provider=_seq_env_attrs)
        # Phase A4 P0: keep a direct reference to the wrap so the runner
        # can read its last_infos / last_env_attrs after play_steps_rnn.
        seq_aux_env_wrap = env

        from dexpbt_isaaclab.awr.elite_awr import (
            EliteAWRBridge,
            EliteAWRBuffer,
            awr_active_from_args,
            build_elite_awr_config_from_args,
            load_awr_sidecar,
        )
        from dexpbt_isaaclab.awr.elite_awr_interceptor import EliteAWRInterceptorEnv

        if awr_active_from_args(args_cli):
            awr_cfg = build_elite_awr_config_from_args(args_cli)
            awr_buffer = EliteAWRBuffer(
                awr_cfg, policy_idx=int(args_cli.policy_idx)
            )
            sidecar_path = (
                Path(args_cli.workspace)
                / "awr_buffers"
                / f"policy_{int(args_cli.policy_idx):02d}_awr_topk.pt"
            )
            loaded_steps = load_awr_sidecar(awr_buffer, sidecar_path, source_type="local")
            if loaded_steps > 0:
                print(
                    f"[AWR] Loaded sidecar {sidecar_path} "
                    f"({loaded_steps} steps, {awr_buffer.num_sequences()} seqs)"
                )

            def _awr_source_meta():
                return {
                    "source_nsucc": getattr(
                        observer, "curr_nsucc_mean", None
                    ),
                    "source_retention": getattr(
                        observer, "curr_source_retention", None
                    ),
                    "source_type": "local",
                    "source_policy": int(args_cli.policy_idx),
                    "source_profile": str(args_cli.wm_strategy_profile),
                    "env_frames": int(
                        getattr(observer, "algo", None)
                        and getattr(getattr(observer, "algo", None), "frame", 0)
                        or 0
                    ),
                }

            collection_mode = str(
                getattr(awr_cfg, "collection_mode", "rollout_buffer") or "rollout_buffer"
            )
            if collection_mode == "interceptor":
                env = EliteAWRInterceptorEnv(
                    env,
                    awr_buffer,
                    awr_cfg,
                    num_envs=env.unwrapped.num_envs,
                    policy_idx=int(args_cli.policy_idx),
                    profile_name=str(args_cli.wm_strategy_profile),
                    env_attrs_provider=_seq_env_attrs,
                    source_meta_provider=_awr_source_meta,
                    device=rl_device,
                )
                awr_env_wrap = env
            awr_bridge = EliteAWRBridge(buffer=awr_buffer, cfg=awr_cfg)
            awr_bridge.source_meta_provider = _awr_source_meta
            awr_bridge.policy_idx = int(args_cli.policy_idx)
            awr_bridge.profile_name = str(args_cli.wm_strategy_profile)
            awr_bridge.stats["awr/collection_mode_code"] = float(
                hash(str(awr_cfg.collection_mode)) % 10_000
            )
            print(
                f"[AWR] enabled: loss_weight={awr_cfg.loss_weight} "
                f"batch_steps={awr_cfg.batch_steps} update_every={awr_cfg.update_every} "
                f"collection_mode={collection_mode} "
                f"collect_after_frames={awr_cfg.collect_after_frames}"
            )

        # Path used by patched_train_epoch to dump wm_seq_latest.pt.
        seq_wm_save_dir = Path(args_cli.workspace) / f"{args_cli.policy_idx:03d}" / "wm_seq_checkpoints"

        # Optional warm-start from prior wm_seq_latest.pt (within-family only;
        # cross-family load is a reset by --wm_load_mode contract).
        wm_seq_ckpt_path = args_cli.wm_checkpoint_path or ""
        if wm_seq_ckpt_path and Path(wm_seq_ckpt_path).is_file():
            try:
                payload = torch.load(wm_seq_ckpt_path, map_location=rl_device,
                                      weights_only=False)
                src_strategy = str(payload.get("strategy") or "unknown")
                compatible, compat_reason = _is_wm_strategy_compatible(
                    src_strategy, args_cli.wm_strategy
                )
                primary_loaded = False
                if args_cli.wm_load_mode != "reset":
                    if "model_state_dict" in payload and compatible:
                        seq_wm_trainer.load_state_dict(payload, strict=False)
                        primary_loaded = True
                        print(
                            f"[SeqWM] Loaded checkpoint from {wm_seq_ckpt_path} "
                            f"(src={src_strategy}, reason={compat_reason})"
                        )
                    elif "model_state_dict" in payload:
                        print(
                            f"[SeqWM] skip warm-start from {wm_seq_ckpt_path}: "
                            f"src={src_strategy} incompatible with "
                            f"{args_cli.wm_strategy} ({compat_reason})"
                        )
                _ens_mgr = getattr(seq_wm_trainer, "_rssm_ensemble_manager", None)
                if _ens_mgr is not None and _ens_mgr.size > 1:
                    _ens_metrics = restore_rssm_ensemble_checkpoint(
                        _ens_mgr,
                        payload,
                        wm_load_mode=str(args_cli.wm_load_mode),
                        ensemble_load_mode=str(args_cli.seq_wm_ensemble_load_mode),
                        primary_loaded=primary_loaded,
                    )
                    seq_wm_trainer._rssm_ensemble_restore_metrics = _ens_metrics
            except Exception as exc:
                print(f"[SeqWM] failed to warm-start from {wm_seq_ckpt_path}: {exc}")

        print(
            f"[SeqWM] enabled: strategy={args_cli.wm_strategy}, obs_dim={obs_dim}, "
            f"action_dim={action_dim}, seq_len={seq_cfg.seq_len}, "
            f"seq_batch_size={seq_cfg.seq_batch_size}, "
            f"lambda_total={seq_cfg.lambda_total}, "
            f"value_detach={seq_cfg.value_detach_belief}"
        )

    if args_cli.enable_wm and not is_sequence_strategy:
        from dexpbt_isaaclab.world_model import (
            WMConfig, WorldModel, WMBuffer, WMTrainer,
            get_strategy, WMDataCollector, WMInterceptorEnv,
            WMCheckpointManager,
        )

        # Determine obs_dim from env (policy obs dim)
        env_spec = gym.spec(args_cli.task)
        entry = env_spec.kwargs["env_cfg_entry_point"]
        mod_path, cls_name = entry.rsplit(":", 1)
        import importlib
        mod = importlib.import_module(mod_path)
        env_cfg_cls = getattr(mod, cls_name)
        env_cfg_obj = env_cfg_cls()
        # Get policy obs dim and action dim from env (not extending policy obs)
        try:
            obs_space = env.unwrapped.single_observation_space["policy"]
            obs_dim = int(obs_space.shape[0])
        except Exception:
            obs_dim = int(agent_cfg["params"]["env"].get("clip_observations", 92))
        try:
            action_space = env.unwrapped.single_action_space
            action_dim = int(action_space.shape[0])
        except Exception:
            action_dim = int(agent_cfg["params"]["env"].get("clip_actions", 23))

        wm_cfg = WMConfig(
            obs_dim=obs_dim,
            action_dim=action_dim,
            ensemble_size=args_cli.wm_ensemble_size,
            lr=args_cli.wm_lr,
            batch_size=args_cli.wm_batch_size,
            update_every_n_epochs=args_cli.wm_update_every_n_epochs,
        )

        wm_module = WorldModel(
            obs_dim=wm_cfg.obs_dim,
            action_dim=wm_cfg.action_dim,
            ensemble_size=wm_cfg.ensemble_size,
            hidden_sizes=wm_cfg.hidden_sizes,
        ).to(rl_device)

        wm_optimizer = torch.optim.Adam(
            wm_module.parameters(), lr=wm_cfg.lr, eps=1e-8,
            weight_decay=wm_cfg.weight_decay,
        )

        wm_buffer = WMBuffer(
            capacity=100_000,
            obs_dim=wm_cfg.obs_dim,
            action_dim=wm_cfg.action_dim,
            device=rl_device,
        )

        strategy_obj = get_strategy(args_cli.wm_strategy)

        # ppo_model and ppo_optimizer will be set after runner.load()
        wm_trainer = WMTrainer(
            wm_module=wm_module,
            wm_optimizer=wm_optimizer,
            wm_buffer=wm_buffer,
            config=wm_cfg,
            strategy=strategy_obj,
            ppo_model=None,   # set after runner.load()
            ppo_optimizer=None,
            tensorboard_writer=None,
            device=rl_device,
        )

        # Wrap env with WM interceptor
        wm_collector = WMDataCollector(
            num_envs=env.unwrapped.num_envs,
            wm_buffer=wm_buffer,
            device=rl_device,
        )
        env = WMInterceptorEnv(env, wm_collector)

        # WM checkpoint manager
        policy_ws = Path(args_cli.workspace) / f"{args_cli.policy_idx:03d}"
        wm_ckpt_manager = WMCheckpointManager(policy_ws)

        wm_warm_start_log = {
            "loaded": False,
            "checkpoint_strategy": str(args_cli.wm_checkpoint_strategy or "none"),
            "target_strategy": str(args_cli.wm_strategy),
            "compatible": False,
            "reset_reason": "not_requested",
        }
        if args_cli.wm_checkpoint_path:
            wm_warm_start_log = _load_wm_checkpoint_with_strategy(
                wm_checkpoint_path=args_cli.wm_checkpoint_path,
                wm_module=wm_module,
                wm_optimizer=wm_optimizer,
                target_strategy=args_cli.wm_strategy,
                source_strategy_hint=args_cli.wm_checkpoint_strategy,
                wm_load_mode=args_cli.wm_load_mode,
            )
        elif args_cli.wm_strategy != "none":
            wm_warm_start_log = {
                "loaded": False,
                "checkpoint_strategy": "none",
                "target_strategy": str(args_cli.wm_strategy),
                "compatible": False,
                "reset_reason": "no_checkpoint_path_provided",
            }
        wm_trainer._warm_start_log = wm_warm_start_log
        wm_trainer._warm_start_log_emitted = False
        wm_trainer._pending_aux_value_head_state = wm_warm_start_log.get("_aux_value_head_state")

        print(f"[WM] wm/checkpoint_loaded={int(bool(wm_warm_start_log['loaded']))}")
        print(f"[WM] wm/checkpoint_source_strategy={wm_warm_start_log['checkpoint_strategy']}")
        print(f"[WM] wm/checkpoint_target_strategy={wm_warm_start_log.get('target_strategy', args_cli.wm_strategy)}")
        print(f"[WM] wm/checkpoint_compatible={int(bool(wm_warm_start_log['compatible']))}")
        print(f"[WM] wm/checkpoint_reset_reason={wm_warm_start_log['reset_reason']}")

        print(f"[WM] enabled: strategy={args_cli.wm_strategy}, "
              f"ensemble={wm_cfg.ensemble_size}, obs_dim={wm_cfg.obs_dim}")

    # wrap for rl-games
    env = RlGamesVecEnvWrapper(env, rl_device, clip_obs, clip_actions)

    vecenv.register(
        "IsaacRlgWrapper",
        lambda config_name, num_actors, **kwargs: RlGamesGpuEnv(config_name, num_actors, **kwargs),
    )
    env_configurations.register("rlgpu", {"vecenv_type": "IsaacRlgWrapper", "env_creator": lambda **kwargs: env})

    agent_cfg["params"]["config"]["num_actors"] = env.unwrapped.num_envs

    mutable_params = {
        "gamma": float(agent_cfg["params"]["config"]["gamma"]),
        "kl_threshold": float(agent_cfg["params"]["config"]["kl_threshold"]),
        "grad_norm": float(agent_cfg["params"]["config"]["grad_norm"]),
        "critic_coef": float(agent_cfg["params"]["config"]["critic_coef"]),
        "bounds_loss_coef": float(agent_cfg["params"]["config"]["bounds_loss_coef"]),
        "e_clip": float(agent_cfg["params"]["config"]["e_clip"]),
        "mini_epochs": int(agent_cfg["params"]["config"]["mini_epochs"]),
        "learning_rate": float(agent_cfg["params"]["config"]["learning_rate"]),
        "alpha_reach": float(env_cfg.alpha_reach),
        "alpha_pick": float(env_cfg.alpha_pick),
        "r_picked": float(env_cfg.r_picked),
        "alpha_targ": float(env_cfg.alpha_targ),
        "r_success": float(env_cfg.r_success),
        "kuka_actions_penalty_scale": float(env_cfg.kuka_actions_penalty_scale),
        "allegro_actions_penalty_scale": float(env_cfg.allegro_actions_penalty_scale),
        "enable_wm": args_cli.enable_wm,
        "wm_strategy": args_cli.wm_strategy,
        "wm_ensemble_size": args_cli.wm_ensemble_size,
        "enable_counterfactual": args_cli.enable_counterfactual,
        "cf_interval_gens": args_cli.cf_interval_gens,
        "cf_steps": args_cli.cf_steps,
        "cf_num_envs": args_cli.cf_num_envs,
        "cf_max_parallel_children": args_cli.cf_max_parallel_children,
        "cf_cuda_visible_devices": args_cli.cf_cuda_visible_devices,
        "cf_horizon": args_cli.cf_horizon,
        "require_counterfactual_runner": args_cli.require_counterfactual_runner,
        # Patch 3 + 4 (rev 2, 2026-05-30): CF child / posterior config for
        # CounterfactualManager (read from mutable_params at init).
        "cf_train_frames": int(args_cli.cf_train_frames),
        "cf_reset_epoch_counter": bool(
            args_cli.cf_reset_epoch_counter or not args_cli.status_output_path
        ),
        "cf_min_eval_s": float(args_cli.cf_min_eval_s),
        "cf_delta_mode": str(args_cli.cf_delta_mode),
        "cf_low_signal_action": str(args_cli.cf_low_signal_action),
        "cf_low_signal_weight": float(args_cli.cf_low_signal_weight),
        "cf_require_nsucc_gt_zero": bool(args_cli.cf_require_nsucc_gt_zero),
        # Patch I (2026-05-31): tighter health gates + delta-metric switch.
        "cf_source_min_nsucc_mean": float(args_cli.cf_source_min_nsucc_mean),
        "cf_source_min_nsucc_max": float(args_cli.cf_source_min_nsucc_max),
        "cf_source_min_ranking_score": float(
            args_cli.cf_source_min_ranking_score
        ),
        "cf_delta_metric": str(args_cli.cf_delta_metric),
    }

    if args_cli.observer_mode == "pbt":
        observer = DexPBTPbtAlgoObserver(
            policy_idx=args_cli.policy_idx,
            num_policies=args_cli.num_policies,
            mutable_params=mutable_params,
            workspace=args_cli.workspace,
            train_dir=log_root,
            task_name=args_cli.task,
            interval_steps=args_cli.pbt_interval_steps,
            start_after=args_cli.pbt_start_after,
            initial_delay=args_cli.pbt_initial_delay,
            mutation_rate=args_cli.pbt_mutation_rate,
            change_min=args_cli.pbt_change_min,
            change_max=args_cli.pbt_change_max,
            ternary_split=True,
            r_meta_window=False,
            num_envs=env.unwrapped.num_envs,
            # Patch 1 (rev 2, 2026-05-30): explicit success-aware ranking
            # score config. Legacy behavior is preserved by passing
            # --pbt_ranking_metric raw_r_meta on the CLI.
            pbt_ranking_metric=args_cli.pbt_ranking_metric,
            pbt_nsucc_zero_cap=args_cli.pbt_nsucc_zero_cap,
            # Patch B (rev 3, 2026-05-30): low-nsucc ranking cap.
            pbt_nsucc_low_threshold=args_cli.pbt_nsucc_low_threshold,
            pbt_nsucc_low_cap=args_cli.pbt_nsucc_low_cap,
            # Patch 2 (rev 2, 2026-05-30): anti-churn knobs + resume hooks
            # (only set when pbt_observer relaunches this worker).
            pbt_min_frames_as_source_after_restart=(
                args_cli.pbt_min_frames_as_source_after_restart
            ),
            pbt_min_frames_between_restarts=args_cli.pbt_min_frames_between_restarts,
            # Patch A (rev 3, 2026-05-30): nsucc-threshold source gate.
            pbt_min_nsucc_mean_as_source=args_cli.pbt_min_nsucc_mean_as_source,
            pbt_min_nsucc_max_as_source=args_cli.pbt_min_nsucc_max_as_source,
            pbt_allow_low_nsucc_source_fallback=(
                args_cli.pbt_allow_low_nsucc_source_fallback
            ),
            # Patch F (2026-05-31): mid-tier self-restart governor.
            pbt_disable_self_restart=args_cli.pbt_disable_self_restart,
            pbt_allow_mid_self_mutation=args_cli.pbt_allow_mid_self_mutation,
            pbt_max_self_restarts_per_policy=(
                args_cli.pbt_max_self_restarts_per_policy
            ),
            pbt_min_frames_between_self_restarts=(
                args_cli.pbt_min_frames_between_self_restarts
            ),
            restart_count_resume=args_cli.restart_count_resume,
            frames_at_restart_resume=args_cli.frames_at_restart_resume,
            last_mutation_frame_resume=args_cli.last_mutation_frame_resume,
            pbt_lineage_id_resume=args_cli.pbt_lineage_id_resume,
            pbt_lineage_start_pf_resume=args_cli.pbt_lineage_start_pf_resume,
            pbt_weight_age_frame_resume=args_cli.pbt_weight_age_frame_resume,
            # Patch G (2026-05-31): tiered source selection + recovery.
            pbt_source_selection_mode=args_cli.pbt_source_selection_mode,
            pbt_relaxed_nsucc_mean_as_source=(
                args_cli.pbt_relaxed_nsucc_mean_as_source
            ),
            pbt_relaxed_nsucc_max_as_source=(
                args_cli.pbt_relaxed_nsucc_max_as_source
            ),
            pbt_enable_best_nsucc_recovery=(
                args_cli.pbt_enable_best_nsucc_recovery
            ),
            pbt_recovery_min_nsucc_mean=args_cli.pbt_recovery_min_nsucc_mean,
            pbt_recovery_min_nsucc_max=args_cli.pbt_recovery_min_nsucc_max,
            pbt_recovery_cooldown_frames=(
                args_cli.pbt_recovery_cooldown_frames
            ),
            # Patch H (2026-05-31): post-restart curriculum freeze.
            pbt_curriculum_freeze_frames=args_cli.pbt_curriculum_freeze_frames,
            pbt_curriculum_freeze_on_recovery_frames=(
                args_cli.pbt_curriculum_freeze_on_recovery_frames
            ),
            # Patch P1 (2026-06-01): mastery curriculum + stable-source cascade.
            curriculum_mode=args_cli.curriculum_mode,
            pbt_source_require_stable_mean=(
                args_cli.pbt_source_require_stable_mean
            ),
            mastery_nsucc_mean=args_cli.mastery_nsucc_mean,
            mastery_nsucc_max=args_cli.mastery_nsucc_max,
            mastery_consecutive_windows=args_cli.mastery_consecutive_windows,
            curriculum_shrink_factor=args_cli.curriculum_shrink_factor,
            pbt_allow_best_nsucc_archive_recovery=(
                args_cli.pbt_allow_best_nsucc_archive_recovery
            ),
            curriculum_collapse_detect=args_cli.curriculum_collapse_detect,
            collapse_drop_ratio=args_cli.collapse_drop_ratio,
            collapse_patience=args_cli.collapse_patience,
            collapse_freeze_env_frames=args_cli.collapse_freeze_env_frames,
            stable_source_min_mean=args_cli.stable_source_min_mean,
            stable_source_min_max=args_cli.stable_source_min_max,
            stable_source_max_drop_ratio=args_cli.stable_source_max_drop_ratio,
            stable_source_drop_windows=args_cli.stable_source_drop_windows,
            pbt_single_shrink_per_window=args_cli.pbt_single_shrink_per_window,
            post_shrink_consolidation_env_frames=(
                args_cli.post_shrink_consolidation_env_frames
            ),
            # Phase A4 P2 (2026-06-08): adaptive-fast curriculum kwargs.
            curriculum_speed_mode=args_cli.curriculum_speed_mode,
            curriculum_fast_strong_mean=args_cli.curriculum_fast_strong_mean,
            curriculum_fast_mid_mean=args_cli.curriculum_fast_mid_mean,
            curriculum_fast_low_mean=args_cli.curriculum_fast_low_mean,
            curriculum_fast_retention_threshold=(
                args_cli.curriculum_fast_retention_threshold
            ),
            curriculum_fast_retention_window=(
                args_cli.curriculum_fast_retention_window
            ),
            curriculum_fast_shrink_strong=args_cli.curriculum_fast_shrink_strong,
            curriculum_fast_shrink_mid=args_cli.curriculum_fast_shrink_mid,
            curriculum_fast_shrink_low=args_cli.curriculum_fast_shrink_low,
            curriculum_fast_consolidation_frames=(
                args_cli.curriculum_fast_consolidation_frames
            ),
            curriculum_fast_recovery_consolidation_frames=(
                args_cli.curriculum_fast_recovery_consolidation_frames
            ),
            curriculum_fast_target_tolerance_800m=(
                args_cli.curriculum_fast_target_tolerance_800m
            ),
            # Phase A4 P3 (2026-06-09): source-safe curriculum inheritance.
            pbt_curriculum_inherit_mode=args_cli.pbt_curriculum_inherit_mode,
            pbt_source_safe_recovery_consolidation_frames=(
                args_cli.pbt_source_safe_recovery_consolidation_frames
            ),
            pbt_source_unsafe_recovery_consolidation_frames=(
                args_cli.pbt_source_unsafe_recovery_consolidation_frames
            ),
            pbt_source_safe_min_mean=args_cli.pbt_source_safe_min_mean,
            pbt_source_safe_min_retention=(
                args_cli.pbt_source_safe_min_retention
            ),
            pbt_protect_promising_workers=(
                args_cli.pbt_protect_promising_workers
            ),
            pbt_promising_nsucc_mean=args_cli.pbt_promising_nsucc_mean,
            pbt_promising_peak_mean=args_cli.pbt_promising_peak_mean,
            pbt_promising_min_retention=(
                args_cli.pbt_promising_min_retention
            ),
            pbt_promising_protect_frames=(
                args_cli.pbt_promising_protect_frames
            ),
            pbt_promising_allow_hard_collapse_override=(
                args_cli.pbt_promising_allow_hard_collapse_override
            ),
            # Phase A4 P4 (2026-06-09): floor-recovery + WM-guided ranking.
            pbt_floor_recovery_aware=args_cli.pbt_floor_recovery_aware,
            pbt_floor_tol_threshold=args_cli.pbt_floor_tol_threshold,
            pbt_hard_floor_tol_threshold=args_cli.pbt_hard_floor_tol_threshold,
            pbt_floor_recovery_window=args_cli.pbt_floor_recovery_window,
            pbt_floor_recovery_slope_weight=(
                args_cli.pbt_floor_recovery_slope_weight
            ),
            pbt_floor_recovery_retention_weight=(
                args_cli.pbt_floor_recovery_retention_weight
            ),
            pbt_floor_recovery_nsucc_max_weight=(
                args_cli.pbt_floor_recovery_nsucc_max_weight
            ),
            pbt_floor_recovery_drop_penalty=(
                args_cli.pbt_floor_recovery_drop_penalty
            ),
            pbt_floor_recovery_protect_frames=(
                args_cli.pbt_floor_recovery_protect_frames
            ),
            pbt_floor_recovery_min_nsucc_protect=(
                args_cli.pbt_floor_recovery_min_nsucc_protect
            ),
            pbt_floor_recovery_min_slope_protect=(
                args_cli.pbt_floor_recovery_min_slope_protect
            ),
            pbt_floor_recovery_min_retention_for_protect=(
                args_cli.pbt_floor_recovery_min_retention_for_protect
            ),
            pbt_enable_best_floor_recovery_archive=(
                args_cli.pbt_enable_best_floor_recovery_archive
            ),
            pbt_wm_score_mode=args_cli.pbt_wm_score_mode,
            pbt_use_wm_recovery_score=args_cli.pbt_use_wm_recovery_score,
            pbt_wm_recovery_weight=args_cli.pbt_wm_recovery_weight,
            pbt_wm_success_weight=args_cli.pbt_wm_success_weight,
            pbt_wm_drop_penalty=args_cli.pbt_wm_drop_penalty,
            pbt_wm_uncertainty_penalty=args_cli.pbt_wm_uncertainty_penalty,
            pbt_wm_max_uncertainty=args_cli.pbt_wm_max_uncertainty,
            experimental_rssm_floor_recovery_stack=(
                args_cli.experimental_rssm_floor_recovery_stack
            ),
            pbt_early_collapse_detect=args_cli.pbt_early_collapse_detect,
            early_collapse_drop_ratio=args_cli.early_collapse_drop_ratio,
            early_collapse_patience=args_cli.early_collapse_patience,
            curriculum_rollback_cooldown_frames=(
                args_cli.curriculum_rollback_cooldown_frames
            ),
            curriculum_safety_enable=args_cli.curriculum_safety_enable,
            curriculum_safety_start_after_frames=(
                args_cli.curriculum_safety_start_after_frames
            ),
            curriculum_safety_min_nsucc_mean=args_cli.curriculum_safety_min_nsucc_mean,
            curriculum_safety_global_retention_threshold=(
                args_cli.curriculum_safety_global_retention_threshold
            ),
            curriculum_safety_drop_ratio_threshold=(
                args_cli.curriculum_safety_drop_ratio_threshold
            ),
            curriculum_safety_slope_threshold=(
                args_cli.curriculum_safety_slope_threshold
            ),
            curriculum_safety_window=args_cli.curriculum_safety_window,
            curriculum_safety_block_frames=args_cli.curriculum_safety_block_frames,
            curriculum_safety_use_real_fallback=(
                args_cli.curriculum_safety_use_real_fallback
            ),
            curriculum_strong_shrink_global_retention_gate=(
                args_cli.curriculum_strong_shrink_global_retention_gate
            ),
            curriculum_strong_shrink_min_global_retention=(
                args_cli.curriculum_strong_shrink_min_global_retention
            ),
            curriculum_strong_shrink_min_stable_ticks=(
                args_cli.curriculum_strong_shrink_min_stable_ticks
            ),
            curriculum_strong_shrink_long_window=(
                args_cli.curriculum_strong_shrink_long_window
            ),
            curriculum_danger_zone_enable=args_cli.curriculum_danger_zone_enable,
            curriculum_danger_tol=args_cli.curriculum_danger_tol,
            curriculum_danger_disable_strong_shrink=(
                args_cli.curriculum_danger_disable_strong_shrink
            ),
            curriculum_danger_min_stable_ticks=(
                args_cli.curriculum_danger_min_stable_ticks
            ),
            curriculum_danger_shrink_interval_mult=(
                args_cli.curriculum_danger_shrink_interval_mult
            ),
            curriculum_danger_global_retention_threshold=(
                args_cli.curriculum_danger_global_retention_threshold
            ),
            curriculum_rollback_on_repeated_collapse=(
                args_cli.curriculum_rollback_on_repeated_collapse
            ),
            curriculum_rollback_collapse_patience=(
                args_cli.curriculum_rollback_collapse_patience
            ),
            curriculum_rollback_to_last_mastered=(
                args_cli.curriculum_rollback_to_last_mastered
            ),
            curriculum_rollback_freeze_frames=(
                args_cli.curriculum_rollback_freeze_frames
            ),
            curriculum_last_mastered_min_nsucc_mean=(
                args_cli.curriculum_last_mastered_min_nsucc_mean
            ),
            curriculum_last_mastered_min_global_retention=(
                args_cli.curriculum_last_mastered_min_global_retention
            ),
            curriculum_last_mastered_stable_ticks=(
                args_cli.curriculum_last_mastered_stable_ticks
            ),
            wm_imagined_diagnostic_enable=args_cli.wm_imagined_diagnostic_enable,
            wm_imagined_shadow_block_enable=(
                args_cli.wm_imagined_shadow_block_enable
            ),
            wm_imagined_shadow_uncertainty_threshold=(
                args_cli.wm_imagined_shadow_uncertainty_threshold
            ),
            wm_imagined_valid_mode=str(args_cli.wm_imagined_valid_mode),
            wm_imagined_valid_uncertainty_threshold=(
                args_cli.wm_imagined_valid_uncertainty_threshold
            ),
            wm_imagined_soft_uncertainty_threshold=(
                args_cli.wm_imagined_soft_uncertainty_threshold
            ),
            wm_imagined_min_logit_abs_mean=args_cli.wm_imagined_min_logit_abs_mean,
            wm_imagined_min_head_signal_std=args_cli.wm_imagined_min_head_signal_std,
            wm_imagined_soft_validity_weight=args_cli.wm_imagined_soft_validity_weight,
            pbt_usable_live_min_mean=args_cli.pbt_usable_live_min_mean,
            pbt_usable_live_min_max=args_cli.pbt_usable_live_min_max,
            pbt_usable_live_max_drop_ratio=args_cli.pbt_usable_live_max_drop_ratio,
            pbt_recovery_bypass_elimination_burnin=(
                args_cli.pbt_recovery_bypass_elimination_burnin
            ),
            pbt_recovery_tolerance_mode=args_cli.pbt_recovery_tolerance_mode,
            pbt_crosscopy_tolerance_mode=args_cli.pbt_crosscopy_tolerance_mode,
            pbt_recovery_source_cooldown_frames=(
                args_cli.pbt_recovery_source_cooldown_frames
            ),
            pbt_freeze_mutation_after_stable_frames=(
                args_cli.pbt_freeze_mutation_after_stable_frames
            ),
            # Patch J (2026-05-31): fixed-strategy ablation mode.
            wm_strategy_mode=args_cli.wm_strategy_mode,
            fixed_wm_strategy=args_cli.fixed_wm_strategy,
            # Track C (2026-06-06): explicit Bayesian strategy pool.
            # Trim trailing commas / whitespace; PBTObserver re-validates.
            wm_strategy_pool=str(args_cli.wm_strategy_pool or "").strip(),
            # Patch F (2026-05-31): self/cross-restart accounting resume hooks.
            self_restart_count_resume=args_cli.self_restart_count_resume,
            last_self_restart_frame_resume=(
                args_cli.last_self_restart_frame_resume
            ),
            cross_restart_count_resume=args_cli.cross_restart_count_resume,
            # Patch G (2026-05-31): archive-recovery accounting resume hooks.
            recovery_restart_count_resume=(
                args_cli.recovery_restart_count_resume
            ),
            last_recovery_frame_resume=args_cli.last_recovery_frame_resume,
            last_profile_copy_frame_resume=(
                args_cli.last_profile_copy_frame_resume
            ),
            # PBT-WM Strategy Zoo v1.0
            wm_strategy_profile_name=str(args_cli.wm_strategy_profile),
            wm_strategy_profile_id=int(
                PROFILE_IDS.get(str(args_cli.wm_strategy_profile), 0)
            ),
            wm_strategy_profile_generation=(
                args_cli.wm_strategy_profile_generation_resume
            ),
            wm_strategy_profile_parent=args_cli.wm_strategy_profile_parent_resume,
            wm_strategy_profile_copied_from_source=(
                args_cli.wm_strategy_profile_copied_from_source_resume
            ),
            wm_curriculum_controller_enable=(
                args_cli.wm_curriculum_controller_enable
            ),
            wm_curriculum_controller_start_after_frames=(
                args_cli.wm_curriculum_controller_start_after_frames
            ),
            wm_curriculum_controller_min_nsucc_mean=(
                args_cli.wm_curriculum_controller_min_nsucc_mean
            ),
            wm_curriculum_controller_drop_risk_threshold=(
                args_cli.wm_curriculum_controller_drop_risk_threshold
            ),
            wm_curriculum_controller_uncertainty_threshold=(
                args_cli.wm_curriculum_controller_uncertainty_threshold
            ),
            wm_curriculum_controller_min_success_prob=(
                args_cli.wm_curriculum_controller_min_success_prob
            ),
            wm_curriculum_controller_freeze_frames=(
                args_cli.wm_curriculum_controller_freeze_frames
            ),
            wm_curriculum_controller_hard_block_shrink=(
                args_cli.wm_curriculum_controller_hard_block_shrink
            ),
            wm_curriculum_controller_use_recovery_prob=(
                args_cli.wm_curriculum_controller_use_recovery_prob
            ),
            wm_activation_enable=bool(args_cli.wm_activation_enable),
            wm_pbt_source_gate_mode=str(args_cli.wm_pbt_source_gate_mode),
            wm_pbt_source_start_after_frames=int(
                args_cli.wm_pbt_source_start_after_frames
            ),
            wm_pbt_source_ramp_frames=int(args_cli.wm_pbt_source_ramp_frames),
            wm_curriculum_gate_mode=str(args_cli.wm_curriculum_gate_mode),
            pbt_wm_profile_copy_enable=args_cli.pbt_wm_profile_copy_enable,
            pbt_wm_profile_copy_after_frames=(
                args_cli.pbt_wm_profile_copy_after_frames
            ),
            pbt_wm_profile_copy_prob=args_cli.pbt_wm_profile_copy_prob,
            pbt_wm_profile_keep_diversity_min_profiles=(
                args_cli.pbt_wm_profile_keep_diversity_min_profiles
            ),
            pbt_wm_profile_copy_min_nsucc_gap=(
                args_cli.pbt_wm_profile_copy_min_nsucc_gap
            ),
            pbt_wm_profile_copy_min_source_retention=(
                args_cli.pbt_wm_profile_copy_min_source_retention
            ),
            pbt_wm_profile_copy_cooldown_frames=(
                args_cli.pbt_wm_profile_copy_cooldown_frames
            ),
            pbt_wm_profile_copy_bottom_k=args_cli.pbt_wm_profile_copy_bottom_k,
            pbt_wm_profile_copy_top_k=args_cli.pbt_wm_profile_copy_top_k,
            pbt_wm_profile_copy_source_min_nsucc_mean=(
                args_cli.pbt_wm_profile_copy_source_min_nsucc_mean
            ),
            pbt_wm_profile_copy_source_min_frames=(
                args_cli.pbt_wm_profile_copy_source_min_frames
            ),
            pbt_wm_profile_copy_target_min_nsucc_mean=(
                args_cli.pbt_wm_profile_copy_target_min_nsucc_mean
            ),
            pbt_wm_profile_copy_target_min_nsucc_max=(
                args_cli.pbt_wm_profile_copy_target_min_nsucc_max
            ),
            pbt_wm_profile_copy_locked_target_slots=(
                args_cli.pbt_wm_profile_copy_locked_target_slots
            ),
            profile_actor_distill_enabled=_wm_profile_caps.get(
                "profile_actor_distill_enabled", False
            ),
            profile_residual_apply_enabled=_wm_profile_caps.get(
                "profile_residual_apply_enabled", False
            ),
            profile_rssm_mppi_enabled=_wm_profile_caps.get(
                "profile_rssm_mppi_enabled", False
            ),
            profile_wm_pbt_score_enabled=_wm_profile_caps.get(
                "profile_wm_pbt_score_enabled", False
            ),
            profile_wm_curriculum_controller_enabled=_wm_profile_caps.get(
                "profile_wm_curriculum_controller_enabled", False
            ),
            profile_elite_bank_enabled=_wm_profile_caps.get(
                "profile_elite_bank_enabled", False
            ),
            profile_conservative_planner_enabled=_wm_profile_caps.get(
                "profile_conservative_planner_enabled", False
            ),
            pbt_consolidation_enable=args_cli.pbt_consolidation_enable,
            pbt_consolidation_stage1_end_frames=(
                args_cli.pbt_consolidation_stage1_end_frames
            ),
            pbt_consolidation_stage2_end_frames=(
                args_cli.pbt_consolidation_stage2_end_frames
            ),
            pbt_consolidation_stage3_end_frames=(
                args_cli.pbt_consolidation_stage3_end_frames
            ),
            pbt_consolidation_stage2_interval_mult=(
                args_cli.pbt_consolidation_stage2_interval_mult
            ),
            pbt_consolidation_stage3_interval_mult=(
                args_cli.pbt_consolidation_stage3_interval_mult
            ),
            pbt_consolidation_stage2_mutation_rate_mult=(
                args_cli.pbt_consolidation_stage2_mutation_rate_mult
            ),
            pbt_consolidation_stage3_bottom_only=(
                args_cli.pbt_consolidation_stage3_bottom_only
            ),
            pbt_consolidation_stage3_hard_collapse_only=(
                args_cli.pbt_consolidation_stage3_hard_collapse_only
            ),
            pbt_consolidation_stage2_retention_threshold=(
                args_cli.pbt_consolidation_stage2_retention_threshold
            ),
            pbt_consolidation_collapse_nsucc_mean_threshold=(
                args_cli.pbt_consolidation_collapse_nsucc_mean_threshold
            ),
            wm_calibration_enable=bool(args_cli.wm_calibration_enable),
        )
    else:
        observer = IsaacAlgoObserver()

    if args_cli.enable_counterfactual and args_cli.observer_mode == "pbt":
        hb_path = Path(args_cli.workspace).resolve() / "counterfactual" / "runner_heartbeat.json"
        heartbeat_age = float("inf")
        if hb_path.exists():
            try:
                data = json.loads(hb_path.read_text())
                heartbeat_age = max(0.0, time.time() - float(data.get("timestamp", 0.0)))
            except Exception:
                heartbeat_age = float("inf")
        if args_cli.require_counterfactual_runner and (not math.isfinite(heartbeat_age) or heartbeat_age > 300.0):
            raise RuntimeError("Counterfactual runner heartbeat is missing/stale while require_counterfactual_runner=True")
        if not math.isfinite(heartbeat_age) or heartbeat_age > 900.0:
            print("[CF] WARNING: enable_counterfactual=True but runner heartbeat missing/stale.")

    cf_tracker = _CfMetaTracker() if args_cli.status_output_path else None

    # Use _WmEnabledRunner when EITHER legacy WM trainer OR sequence WM
    # trainer is wired (Phase A4: sequence path uses _WmEnabledRunner with
    # seq_wm_trainer so train_epoch / play_steps_rnn patches still apply).
    if wm_trainer is not None or seq_wm_trainer is not None:
        tb_writer = getattr(observer, "writer", None)
        runner = _WmEnabledRunner(
            observer=observer,
            wm_trainer=wm_trainer,
            wm_ckpt_manager=wm_ckpt_manager,
            wm_strategy=args_cli.wm_strategy,
            tensorboard_writer=tb_writer,
            cf_tracker=cf_tracker,
            raw_env=raw_env,
            # Patch 3 (rev 2, 2026-05-30): only flips on for CF children.
            cf_reset_epoch_counter=args_cli.cf_reset_epoch_counter,
            seq_wm_trainer=seq_wm_trainer,
            seq_aux_collector=seq_aux_collector,
            seq_wm_save_dir=seq_wm_save_dir,
            seq_aux_env_wrap=seq_aux_env_wrap,
            seq_policy_coupling_cfg=seq_policy_coupling_cfg,
            awr_bridge=awr_bridge,
            awr_env_wrap=awr_env_wrap,
        )
    elif args_cli.checkpoint is not None:
        # ppo_only and other non-WM profiles still receive cross-copy checkpoints
        # from EIR/elite donors; stock Runner::_restore crashes on optimizer
        # param-group mismatches.
        runner = _CheckpointSafeRunner(observer)
    else:
        runner = Runner(observer)

    runner.load(agent_cfg)
    runner.reset()

    # Seed diversity audit (B3): emit per-policy seed + env-randomization
    # fingerprint into every training_curve point. We hash a small set of
    # env attributes that affect rollout sampling but stay constant for the
    # process lifetime.
    if hasattr(observer, "set_seeding_audit"):
        try:
            import hashlib

            raw_env_for_hash = getattr(env, "unwrapped", env)
            object_scales = getattr(raw_env_for_hash, "object_scales", None)
            scales_repr = (
                object_scales.detach().cpu().numpy().tobytes()
                if hasattr(object_scales, "detach")
                else repr(object_scales)
            )
            noise_repr = repr(
                {
                    "reset_position_noise_x": getattr(env_cfg, "reset_position_noise_x", None),
                    "reset_position_noise_y": getattr(env_cfg, "reset_position_noise_y", None),
                    "reset_position_noise_z": getattr(env_cfg, "reset_position_noise_z", None),
                    "reset_rotation_noise": getattr(env_cfg, "reset_rotation_noise", None),
                    "reset_dof_pos_random_interval_lower": getattr(
                        env_cfg, "reset_dof_pos_random_interval_lower", None
                    ),
                    "reset_dof_pos_random_interval_upper": getattr(
                        env_cfg, "reset_dof_pos_random_interval_upper", None
                    ),
                    "randomize_object_dimensions": getattr(
                        env_cfg, "randomize_object_dimensions", None
                    ),
                }
            )
            digest_payload = (
                f"seed={effective_seed}|num_envs={env.unwrapped.num_envs}|"
                f"noise={noise_repr}".encode("utf-8")
                + b"|scales="
                + (scales_repr if isinstance(scales_repr, bytes) else scales_repr.encode("utf-8"))
            )
            env_randomization_hash = hashlib.sha1(digest_payload).hexdigest()[:16]
            observer.set_seeding_audit(effective_seed, env_randomization_hash)
            print(
                f"[train] policy_idx={args_cli.policy_idx} "
                f"env_randomization_hash={env_randomization_hash}",
                flush=True,
            )
        except Exception as e:  # noqa: BLE001
            print(
                f"[train] WARN: failed to compute env_randomization_hash: {e}",
                flush=True,
            )

    if args_cli.test:
        runner.run({"train": False, "play": True, "sigma": train_sigma})
    elif args_cli.checkpoint is not None:
        runner.run(
            _pbt_restore_runner_args(
                args_cli,
                train=True,
                checkpoint=args_cli.checkpoint,
                sigma=train_sigma,
            )
        )
    else:
        runner.run({"train": True, "play": False, "sigma": train_sigma})

    elapsed = round(time.time() - start_time, 2)
    print(f"Training time: {elapsed}s")

    # Optional CF status output for run_counterfactual_block.py.
    # Patch 3 (rev 2, 2026-05-30): include r_meta_before/after,
    # nsucc_before/after, success_tolerance_before/after, child_train_frames
    # and r_meta_samples so the runner can detect invalid (too-short)
    # and low_signal (tolerance-only) blocks.
    if args_cli.status_output_path:
        try:
            if cf_tracker is not None and cf_tracker.samples:
                meta = cf_tracker.summary()
                r_meta = float(meta["r_meta_max"])
            else:
                r_meta, _ = _compute_env_r_meta(raw_env)

            _, mean_nsucc = _compute_env_r_meta(raw_env)
            success_tol = float(getattr(raw_env, "success_tolerance", float("nan")))
            target_tol = float(getattr(raw_env, "target_success_tolerance", success_tol))
            # Patch C (rev 3, 2026-05-30): compute the real CF child
            # frame count via three-tier fallback. The previous code
            # only read `algo.frame` from a NEVER-ASSIGNED `runner.algo`
            # attribute and returned 0 for every CF block, marking the
            # entire posterior path invalid in
            # regrasping_cbwm_20260530_160513.
            try:
                horizon_length = int(
                    agent_cfg["params"]["config"].get("horizon_length", 16)
                )
            except Exception:
                horizon_length = int(args_cli.cf_horizon)
            child_train_frames, frame_method = _compute_child_train_frames(
                runner,
                cf_tracker,
                num_envs=int(env.unwrapped.num_envs),
                horizon_length=horizon_length,
            )

            status_payload = {
                "r_meta": r_meta,
                "mean_prev_episode_successes": mean_nsucc,
                "success_tolerance": success_tol,
                "target_success_tolerance": target_tol,
                "elapsed_s": elapsed,
                # Patch 3 (rev 2, 2026-05-30): explicit before/after for
                # learning_improvement delta_w. Falls back to current
                # env values when cf_tracker has fewer than 2 samples.
                "r_meta_before": float("nan"),
                "r_meta_after": r_meta,
                "nsucc_before": float("nan"),
                "nsucc_after": mean_nsucc,
                "success_tolerance_before": float("nan"),
                "success_tolerance_after": success_tol,
                "r_meta_samples": 0,
                "child_train_frames": int(child_train_frames),
                # Patch C: which of the three signals produced the count.
                # Audited downstream by run_counterfactual_block.py.
                "frame_estimation_method": str(frame_method),
                "cf_train_frames_target": int(args_cli.cf_train_frames),
                "cf_reset_epoch_counter": bool(args_cli.cf_reset_epoch_counter),
            }
            if cf_tracker is not None and cf_tracker.samples:
                meta = cf_tracker.summary()
                status_payload["r_meta_mean"] = float(meta["r_meta_mean"])
                status_payload["r_meta_max"] = float(meta["r_meta_max"])
                status_payload["r_meta_samples"] = len(cf_tracker.samples)
                status_payload["r_meta_before"] = float(meta.get("r_meta_before", float("nan")))
                status_payload["r_meta_after"] = float(meta.get("r_meta_after", r_meta))
                status_payload["nsucc_before"] = float(meta.get("nsucc_before", float("nan")))
                status_payload["nsucc_after"] = float(meta.get("nsucc_after", mean_nsucc))
                status_payload["success_tolerance_before"] = float(
                    meta.get("success_tolerance_before", float("nan"))
                )
                status_payload["success_tolerance_after"] = float(
                    meta.get("success_tolerance_after", success_tol)
                )
            _write_status_file(args_cli.status_output_path, status_payload)
            print(f"[CF] Wrote status file: {args_cli.status_output_path}")
        except Exception as exc:
            print(f"[CF] Failed to write status file: {exc}")

    # Save WM checkpoint on exit
    if wm_trainer is not None and wm_ckpt_manager is not None:
        aux_state = wm_trainer.get_aux_value_head_state()
        wm_ckpt_manager.save(
            wm_module=wm_trainer.module,
            wm_optimizer=wm_trainer.optimizer,
            strategy=args_cli.wm_strategy,
            iteration=int(agent_cfg["params"]["config"].get("max_epochs", 0)),
            frames=0,
            aux_value_head_state=aux_state,
        )
        print(f"[WM] Checkpoint saved: {wm_ckpt_manager.latest_path}")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
