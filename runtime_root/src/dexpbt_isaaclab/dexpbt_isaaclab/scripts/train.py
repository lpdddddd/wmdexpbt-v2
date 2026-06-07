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
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

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
from dexpbt_isaaclab.world_model.seq_aux_collector import (
    SeqAuxCollector, SeqAuxInterceptorEnv,
)
from dexpbt_isaaclab.world_model.wm_aux_buffer import register_seq_aux_fields
from dexpbt_isaaclab.world_model.rssm_lite import RSSMLiteWorldModel
from dexpbt_isaaclab.world_model.transformer_lite import TransformerLiteWorldModel


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

def _patch_actor_train_for_seq_wm(
    actor,
    seq_wm_trainer: SeqWMTrainer,
    seq_aux_collector: SeqAuxCollector,
    seq_wm_save_dir,
    save_every_epochs: int = 50,
):
    """Wire SeqWMTrainer into the rl_games agent.

    Patches:
      1. agent.experience_buffer aux_tensor_dict (registers
         aux_d_hat / aux_failure / aux_lifted / aux_success_cnt fields).
      2. agent.play_steps_rnn → wraps to capture batch_dict (used to read
         GAE returns + ensure aux fields are populated from the collector).
      3. agent.train_epoch → wraps to call seq_wm_trainer.step() each PPO
         epoch using the just-collected (T, N) tensors and the captured
         batch_dict["returns"].

    The legacy WMTrainer paths (model.get_aux_loss patch, _patch_actor_train_for_wm)
    are NOT applied for sequence strategies — see _WmEnabledRunner.run_train.
    """
    # 1) Register aux fields on the buffer (idempotent; safe to call once).
    register_seq_aux_fields(actor.experience_buffer)

    # 2) Capture latest batch_dict from play_steps_rnn so train_epoch tail
    #    can read GAE returns. Use a list-based one-element store to avoid
    #    rebinding closure.
    capture: dict = {"batch_dict": None}
    original_play_steps_rnn = actor.play_steps_rnn

    def patched_play_steps_rnn(*args, **kwargs):
        # Reset the aux-window at the START of each PPO epoch's rollout.
        seq_aux_collector.reset_window()
        bd = original_play_steps_rnn(*args, **kwargs)
        capture["batch_dict"] = bd
        return bd

    actor.play_steps_rnn = patched_play_steps_rnn

    # 3) Patch train_epoch to (a) copy collector window into the buffer's
    #    aux fields, (b) call SeqWMTrainer.step() with returns from
    #    capture["batch_dict"], (c) periodically save a SeqWM checkpoint.
    original_train_epoch = actor.train_epoch
    seq_wm_trainer._epochs_since_save = 0

    def patched_train_epoch(*args, **kwargs):
        result = original_train_epoch(*args, **kwargs)
        try:
            full_aux = seq_aux_collector.full_window()
            for name, tensor in full_aux.items():
                store = actor.experience_buffer.tensor_dict.get(name)
                if store is None:
                    continue
                store.copy_(tensor)

            bd = capture["batch_dict"]
            if bd is None:
                return result
            T = int(actor.horizon_length)
            N = int(actor.num_actors * actor.num_agents)
            returns_flat = bd.get("returns")
            if returns_flat is None:
                return result
            # rl_games swap_and_flatten01: shape (T*N, value_size). Reshape
            # back to (T, N) — value_size is 1 for our tasks.
            value_size = (
                returns_flat.shape[-1] if returns_flat.ndim == 2 else 1
            )
            try:
                returns_full = returns_flat.view(N, T, value_size).transpose(0, 1).contiguous()
            except Exception:
                # Fall back: assume already (T, N, *) layout.
                returns_full = returns_flat.view(T, N, value_size)

            obs_full = actor.experience_buffer.tensor_dict["obses"]
            last_obs = actor.obs["obs"] if isinstance(actor.obs, dict) else actor.obs
            last_obs_t = (
                last_obs if hasattr(last_obs, "shape")
                else torch.zeros(N, obs_full.shape[-1], device=obs_full.device)
            )

            last_aux = seq_aux_collector.last_step()
            last_d_hat = last_aux.get("aux_d_hat")
            last_success_count = last_aux.get("aux_success_cnt")
            if last_d_hat is not None:
                last_d_hat = last_d_hat.squeeze(-1)
            if last_success_count is not None:
                last_success_count = last_success_count.squeeze(-1)

            frames = int(getattr(actor, "frame", 0) or 0)
            metrics = seq_wm_trainer.step(
                experience_buffer=actor.experience_buffer,
                frames=frames,
                last_obs=last_obs_t,
                returns_full=returns_full.squeeze(-1) if returns_full.shape[-1] == 1 else returns_full,
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

            seq_wm_trainer._epochs_since_save += 1
            if (seq_wm_save_dir is not None
                    and seq_wm_trainer._epochs_since_save >= save_every_epochs):
                seq_wm_trainer._epochs_since_save = 0
                try:
                    save_dir = Path(seq_wm_save_dir)
                    save_dir.mkdir(parents=True, exist_ok=True)
                    payload = seq_wm_trainer.state_dict()
                    payload["frames"] = frames
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


def _wm_safe_restore(agent, checkpoint_path: str) -> None:
    """Restore rl_games agent tolerating aux_value_head / optimizer group mismatches."""
    import copy
    import torch
    from dexpbt_isaaclab.pbt.pbt_observer import DexPBTPbtAlgoObserver

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

    if "epoch" in ckpt:
        agent.epoch_num = ckpt["epoch"]
    if "frame" in ckpt:
        agent.frame = ckpt["frame"]
    agent.last_mean_rewards = ckpt.get("last_mean_rewards", getattr(agent, "last_mean_rewards", -1e9))

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
    patched = {"optimizer": copy.deepcopy(opt_sd)}
    DexPBTPbtAlgoObserver._strip_aux_optimizer_state(patched)
    if _try_load_opt(patched["optimizer"]):
        print("[WM] restore: loaded optimizer after stripping aux_value_head group from checkpoint")
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
                 seq_wm_save_dir=None):
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
            _patch_actor_train_for_seq_wm(
                agent,
                self.seq_wm_trainer,
                self.seq_aux_collector,
                self.seq_wm_save_dir,
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
        if wm_trainer is not None and ckpt_path:
            _wm_safe_restore(agent, ckpt_path)
        else:
            torch_runner._restore(agent, args)
        torch_runner._override_sigma(agent, args)

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
            f"[RESUME] PBT restart from source_policy={src}: " +
            ", ".join(f"{k}={v}" for k, v in resume_log.items())
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
        )
        with_value = args_cli.wm_strategy in ("rssm_belief_value", "trans_belief_value")
        if args_cli.wm_strategy in RSSM_FAMILY:
            seq_model = RSSMLiteWorldModel(
                obs_dim=obs_dim,
                action_dim=action_dim,
                free_nats=seq_cfg.free_nats,
                with_value_head=with_value,
            )
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

        # Aux collector + env wrap (must happen BEFORE RlGamesVecEnvWrapper).
        seq_aux_collector = SeqAuxCollector(
            num_envs=env.unwrapped.num_envs,
            horizon_length=int(agent_cfg["params"]["config"].get("horizon_length", 16)),
            device=rl_device,
        )

        def _seq_env_attrs(e):
            # Used by SeqAuxInterceptorEnv to read per-env env attributes
            # (lifted_object, successes) that are not exposed via infos.
            inner = getattr(e, "unwrapped", e)
            d: dict = {}
            for name in ("lifted_object", "successes"):
                v = getattr(inner, name, None)
                if v is not None:
                    d[name] = v
            return d

        env = SeqAuxInterceptorEnv(env, seq_aux_collector,
                                    env_attrs_provider=_seq_env_attrs)

        # Path used by patched_train_epoch to dump wm_seq_latest.pt.
        seq_wm_save_dir = Path(args_cli.workspace) / f"{args_cli.policy_idx:03d}" / "wm_seq_checkpoints"

        # Optional warm-start from prior wm_seq_latest.pt (within-family only;
        # cross-family load is a reset by --wm_load_mode contract).
        wm_seq_ckpt_path = args_cli.wm_checkpoint_path or ""
        if (wm_seq_ckpt_path
                and args_cli.wm_load_mode != "reset"
                and Path(wm_seq_ckpt_path).is_file()):
            try:
                payload = torch.load(wm_seq_ckpt_path, map_location=rl_device,
                                      weights_only=False)
                if "model_state_dict" in payload and payload.get("strategy") in (
                        args_cli.wm_strategy,
                ):
                    seq_wm_trainer.load_state_dict(payload, strict=False)
                    print(f"[SeqWM] Loaded checkpoint from {wm_seq_ckpt_path}")
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
            pbt_early_collapse_detect=args_cli.pbt_early_collapse_detect,
            early_collapse_drop_ratio=args_cli.early_collapse_drop_ratio,
            early_collapse_patience=args_cli.early_collapse_patience,
            curriculum_rollback_cooldown_frames=(
                args_cli.curriculum_rollback_cooldown_frames
            ),
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
        )
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
        runner.run({"train": True, "play": False, "sigma": train_sigma, "checkpoint": args_cli.checkpoint})
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
