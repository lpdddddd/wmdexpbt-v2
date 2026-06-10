#!/usr/bin/env bash
# Phase A4 P4 wired (2026-06-10): same RSSM iterative full-active-replay
# 3B configuration as start_regrasping_fixed_rssm_iterative_full_active_replay_3b.sh,
# but explicitly running the *wired* runtime: TeacherManager and
# WMImaginedEvaluator are now instantiated inside train.py and write
# per-policy mailboxes consumed by the PBT observer. Default horizon
# steps changed to 4,8 because seq_len=16 makes h16 degenerate.
#
# Diff vs the unwired predecessor:
#   1. run name suffix: ..._3b -> ..._3b_wired
#   2. --seq_wm_horizon_steps 8,16  ->  4,8
#   3. all wiring flags asserted true: --seq_wm_enable_iterative_teacher,
#      --seq_wm_enable_imagined_evaluator, --pbt_wm_score_mode active,
#      --seq_wm_replay_prioritize_floor_recovery
#
# Hard constraints preserved:
#   - reward / env / PPO actor-critic structure UNCHANGED.
#   - --seq_wm_policy_coupling=none (no PPO distillation).
#   - --wm_strategy=rssm_belief_value (no value_imag_h3).
#   - no counterfactual reward.
#   - --seq_wm_imagine_action_mode=recorded (policy mode raises).
#   - replay prioritization affects RSSM sampler ONLY; PPO replay
#     untouched.
#   - active WM ranking is gated: falls back to real-only score when
#     teacher is invalid, target degenerate, uncertainty high, or any
#     policy is in hard collapse / NaN.
#   - max_consecutive_successes=50 (env default; set inside the env).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNTIME_ROOT="/root/autodl-tmp/wmdexpbt/runtime_root"
RESULTS_ROOT="/root/autodl-tmp/wmdexpbt/results"
RUNS_ROOT="${RESULTS_ROOT}/runs"
DATE=$(date +%Y%m%d_%H%M%S)
RUN_DIR="${RUNS_ROOT}/regrasping_fixed_rssm_iterative_full_active_replay_3b_wired_${DATE}"
LOG="${RUN_DIR}/launch_master.log"
PIDFILE="${RUN_DIR}/nohup.pid"

WM_STRATEGY="rssm_belief_value"
BASE_SEED="${BASE_SEED:-42}"
NUM_POLICIES="${NUM_POLICIES:-6}"
DEXPBT_GPUS="${DEXPBT_GPUS:-0,1,2,3,4,5}"
SMOKE_MAX_ITER="${SMOKE_MAX_ITER:-22889}"
SMOKE_PBT_START_AFTER="${SMOKE_PBT_START_AFTER:-200000000}"
SMOKE_PBT_INITIAL_DELAY="${SMOKE_PBT_INITIAL_DELAY:-50000000}"
ROLLBACK_COOLDOWN_FRAMES="${ROLLBACK_COOLDOWN_FRAMES:-50000000}"
CROSSCOPY_TOLERANCE_MODE="${CROSSCOPY_TOLERANCE_MODE:-last_mastered}"
SEQ_WM_SEQ_LEN="${SEQ_WM_SEQ_LEN:-16}"
SEQ_WM_BATCH="${SEQ_WM_BATCH:-1024}"
SEQ_WM_LAMBDA_TOTAL="${SEQ_WM_LAMBDA_TOTAL:-0.05}"
SEQ_WM_LAMBDA_VALUE="${SEQ_WM_LAMBDA_VALUE:-0.1}"

FAST_STRONG_MEAN="${FAST_STRONG_MEAN:-30.0}"
FAST_MID_MEAN="${FAST_MID_MEAN:-20.0}"
FAST_LOW_MEAN="${FAST_LOW_MEAN:-10.0}"
FAST_RETENTION_THRESHOLD="${FAST_RETENTION_THRESHOLD:-0.8}"
FAST_RETENTION_WINDOW="${FAST_RETENTION_WINDOW:-4}"
FAST_SHRINK_STRONG="${FAST_SHRINK_STRONG:-0.90}"
FAST_SHRINK_MID="${FAST_SHRINK_MID:-0.925}"
FAST_SHRINK_LOW="${FAST_SHRINK_LOW:-0.95}"
FAST_CONSOLIDATION_FRAMES="${FAST_CONSOLIDATION_FRAMES:-50000000}"
FAST_RECOVERY_CONSOLIDATION_FRAMES="${FAST_RECOVERY_CONSOLIDATION_FRAMES:-100000000}"
FAST_TARGET_TOLERANCE_800M="${FAST_TARGET_TOLERANCE_800M:-0.04}"

INHERIT_MODE="${INHERIT_MODE:-source_safe}"
SOURCE_SAFE_RECOVERY_FRAMES="${SOURCE_SAFE_RECOVERY_FRAMES:-50000000}"
SOURCE_UNSAFE_RECOVERY_FRAMES="${SOURCE_UNSAFE_RECOVERY_FRAMES:-100000000}"
SOURCE_SAFE_MIN_MEAN="${SOURCE_SAFE_MIN_MEAN:-20.0}"
SOURCE_SAFE_MIN_RETENTION="${SOURCE_SAFE_MIN_RETENTION:-0.7}"
PROTECT_PROMISING="${PROTECT_PROMISING:-true}"
case "${PROTECT_PROMISING}" in
  1|true|TRUE|True) PROTECT_PROMISING_FLAG="--pbt_protect_promising_workers" ;;
  *)               PROTECT_PROMISING_FLAG="--no-pbt_protect_promising_workers" ;;
esac
PROMISING_NSUCC_MEAN="${PROMISING_NSUCC_MEAN:-20.0}"
PROMISING_PEAK_MEAN="${PROMISING_PEAK_MEAN:-25.0}"
PROMISING_MIN_RETENTION="${PROMISING_MIN_RETENTION:-0.5}"
PROMISING_PROTECT_FRAMES="${PROMISING_PROTECT_FRAMES:-150000000}"

# Floor-recovery PBT knobs (Part A).
FLOOR_TOL_THRESHOLD="${FLOOR_TOL_THRESHOLD:-0.02}"
HARD_FLOOR_TOL_THRESHOLD="${HARD_FLOOR_TOL_THRESHOLD:-0.012}"
FLOOR_RECOVERY_WINDOW="${FLOOR_RECOVERY_WINDOW:-4}"
FLOOR_RECOVERY_PROTECT_FRAMES="${FLOOR_RECOVERY_PROTECT_FRAMES:-200000000}"
FLOOR_RECOVERY_MIN_NSUCC_PROTECT="${FLOOR_RECOVERY_MIN_NSUCC_PROTECT:-15.0}"

# Phase A4 P4 wired (2026-06-10): horizon steps default 4,8 because
# seq_len=16 makes h16 degenerate (target undefined for the last 16
# steps of every window). The wired RSSM compute_aux_loss skips
# horizons exceeding seq_len-2 anyway, but we set the launcher
# explicitly so the diagnostic surface is clean.
SEQ_WM_HORIZON_STEPS="${SEQ_WM_HORIZON_STEPS:-4,8}"
SEQ_WM_RECOVERY_HORIZON="${SEQ_WM_RECOVERY_HORIZON:-8}"
SEQ_WM_RECOVERY_SHADOW_TOL_THRESHOLD="${SEQ_WM_RECOVERY_SHADOW_TOL_THRESHOLD:-0.04}"

export DEXPBT_MAX_ACTIVE="${DEXPBT_MAX_ACTIVE:-2}"
export DEXPBT_INTER_LAUNCH_DELAY_S="${DEXPBT_INTER_LAUNCH_DELAY_S:-90}"
export DEXPBT_STARTUP_TIMEOUT_S="${DEXPBT_STARTUP_TIMEOUT_S:-900}"
export DEXPBT_STARTUP_MAX_RESTARTS="${DEXPBT_STARTUP_MAX_RESTARTS:-1}"
export NUM_POLICIES
export DEXPBT_GPUS

mkdir -p "$RUN_DIR"
echo "$RUN_DIR" > "${RESULTS_ROOT}/pointers/latest_rssm_iterative_full_active_replay_3b_wired_run.txt"
echo "$RUN_DIR" > "${RUN_DIR}/RUN_DIR.txt"

cat > "${RUN_DIR}/RUN_CONFIG.txt" <<EOF
launcher=start_regrasping_fixed_rssm_iterative_full_active_replay_3b_wired.sh
profile=phase_a4_p4_wired_rssm_iterative_full_active_replay_3b
target_frames=3000000000
max_iterations=${SMOKE_MAX_ITER}
num_policies=${NUM_POLICIES}
gpus=${DEXPBT_GPUS}
base_seed=${BASE_SEED}
wm_strategy_mode=fixed
fixed_wm_strategy=${WM_STRATEGY}
seq_wm_policy_coupling=none
curriculum_speed_mode=adaptive_fast
pbt_curriculum_inherit_mode=${INHERIT_MODE}
pbt_protect_promising_workers=${PROTECT_PROMISING}
pbt_floor_recovery_aware=true
pbt_enable_best_floor_recovery_archive=true
pbt_wm_score_mode=active
pbt_use_wm_recovery_score=true
seq_wm_use_task_aux_inputs=true
seq_wm_loss_preset=task_focused
seq_wm_floor_aware_weighting=true
seq_wm_use_horizon_heads=true
seq_wm_horizon_steps=${SEQ_WM_HORIZON_STEPS}
seq_wm_use_recovery_head=true
seq_wm_recovery_horizon=${SEQ_WM_RECOVERY_HORIZON}
seq_wm_recovery_shadow_tol_threshold=${SEQ_WM_RECOVERY_SHADOW_TOL_THRESHOLD}
seq_wm_enable_iterative_teacher=true
seq_wm_enable_imagined_evaluator=true
seq_wm_imagine_action_mode=recorded
seq_wm_replay_prioritize_floor_recovery=true
runtime_wiring=teacher_manager+imagined_evaluator+observer_disk_refresh
EOF

source "${RUNTIME_ROOT}/scripts/env_dexpbt.sh"
export TORCHDYNAMO_DISABLE=1
export PYTHONUNBUFFERED=1
export OMNI_KIT_TELEMETRY=0
export CARB_CRASH_REPORTER=0

cd "${RUNTIME_ROOT}/src/dexpbt_isaaclab"

nohup bash "${RUNTIME_ROOT}/scripts/launch_8gpu_dexpbt.sh" \
  "$RUN_DIR" \
  --task DexPBT-Regrasping-KukaAllegro-Direct-v0 \
  --num-envs 8192 \
  --max-iterations "$SMOKE_MAX_ITER" \
  --pbt-interval-steps 20000000 \
  --pbt-start-after "$SMOKE_PBT_START_AFTER" \
  --pbt-initial-delay "$SMOKE_PBT_INITIAL_DELAY" \
  --extra \
  --seed "$BASE_SEED" \
  --enable_wm \
  --wm_strategy "$WM_STRATEGY" \
  --wm_strategy_mode fixed \
  --fixed_wm_strategy "$WM_STRATEGY" \
  --wm_load_mode reset \
  --seq_wm_seq_len "$SEQ_WM_SEQ_LEN" \
  --seq_wm_seq_batch_size "$SEQ_WM_BATCH" \
  --seq_wm_lambda_total "$SEQ_WM_LAMBDA_TOTAL" \
  --seq_wm_lambda_value "$SEQ_WM_LAMBDA_VALUE" \
  --no-seq_wm_value_detach_belief \
  --seq_wm_policy_coupling none \
  --pbt_ranking_metric success_aware_r_meta \
  --pbt_min_frames_as_source_after_restart 50000000 \
  --pbt_min_frames_between_restarts 50000000 \
  --pbt_disable_self_restart \
  --pbt_source_selection_mode tiered \
  --no-pbt_enable_best_nsucc_recovery \
  --pbt_allow_best_nsucc_archive_recovery \
  --pbt_recovery_cooldown_frames 400000000 \
  --pbt_curriculum_freeze_frames 100000000 \
  --pbt_curriculum_freeze_on_recovery_frames 100000000 \
  --curriculum_mode mastery \
  --pbt_source_require_stable_mean \
  --mastery_nsucc_mean 4.0 \
  --mastery_nsucc_max 20.0 \
  --mastery_consecutive_windows 2 \
  --curriculum_shrink_factor 0.95 \
  --pbt_single_shrink_per_window \
  --post_shrink_consolidation_env_frames 100000000 \
  --curriculum_speed_mode adaptive_fast \
  --curriculum_fast_strong_mean "$FAST_STRONG_MEAN" \
  --curriculum_fast_mid_mean "$FAST_MID_MEAN" \
  --curriculum_fast_low_mean "$FAST_LOW_MEAN" \
  --curriculum_fast_retention_threshold "$FAST_RETENTION_THRESHOLD" \
  --curriculum_fast_retention_window "$FAST_RETENTION_WINDOW" \
  --curriculum_fast_shrink_strong "$FAST_SHRINK_STRONG" \
  --curriculum_fast_shrink_mid "$FAST_SHRINK_MID" \
  --curriculum_fast_shrink_low "$FAST_SHRINK_LOW" \
  --curriculum_fast_consolidation_frames "$FAST_CONSOLIDATION_FRAMES" \
  --curriculum_fast_recovery_consolidation_frames "$FAST_RECOVERY_CONSOLIDATION_FRAMES" \
  --curriculum_fast_target_tolerance_800m "$FAST_TARGET_TOLERANCE_800M" \
  --pbt_curriculum_inherit_mode "$INHERIT_MODE" \
  --pbt_source_safe_recovery_consolidation_frames "$SOURCE_SAFE_RECOVERY_FRAMES" \
  --pbt_source_unsafe_recovery_consolidation_frames "$SOURCE_UNSAFE_RECOVERY_FRAMES" \
  --pbt_source_safe_min_mean "$SOURCE_SAFE_MIN_MEAN" \
  --pbt_source_safe_min_retention "$SOURCE_SAFE_MIN_RETENTION" \
  ${PROTECT_PROMISING_FLAG} \
  --pbt_promising_nsucc_mean "$PROMISING_NSUCC_MEAN" \
  --pbt_promising_peak_mean "$PROMISING_PEAK_MEAN" \
  --pbt_promising_min_retention "$PROMISING_MIN_RETENTION" \
  --pbt_promising_protect_frames "$PROMISING_PROTECT_FRAMES" \
  --pbt_promising_allow_hard_collapse_override \
  --pbt_floor_recovery_aware \
  --pbt_floor_tol_threshold "$FLOOR_TOL_THRESHOLD" \
  --pbt_hard_floor_tol_threshold "$HARD_FLOOR_TOL_THRESHOLD" \
  --pbt_floor_recovery_window "$FLOOR_RECOVERY_WINDOW" \
  --pbt_floor_recovery_protect_frames "$FLOOR_RECOVERY_PROTECT_FRAMES" \
  --pbt_floor_recovery_min_nsucc_protect "$FLOOR_RECOVERY_MIN_NSUCC_PROTECT" \
  --pbt_enable_best_floor_recovery_archive \
  --pbt_wm_score_mode active \
  --pbt_use_wm_recovery_score \
  --pbt_wm_recovery_weight 0.3 \
  --pbt_wm_success_weight 0.2 \
  --pbt_wm_drop_penalty 0.3 \
  --pbt_wm_uncertainty_penalty 0.2 \
  --pbt_wm_max_uncertainty 0.5 \
  --seq_wm_use_task_aux_inputs \
  --seq_wm_loss_preset task_focused \
  --seq_wm_floor_aware_weighting \
  --seq_wm_use_horizon_heads \
  --seq_wm_horizon_steps "$SEQ_WM_HORIZON_STEPS" \
  --seq_wm_horizon_skip_too_long \
  --seq_wm_use_recovery_head \
  --seq_wm_recovery_horizon "$SEQ_WM_RECOVERY_HORIZON" \
  --seq_wm_recovery_shadow_tol_threshold "$SEQ_WM_RECOVERY_SHADOW_TOL_THRESHOLD" \
  --seq_wm_lambda_horizon_success 0.5 \
  --seq_wm_lambda_drop_risk 0.5 \
  --seq_wm_lambda_recovery 1.0 \
  --seq_wm_enable_iterative_teacher \
  --seq_wm_teacher_eval_interval_frames 50000000 \
  --seq_wm_teacher_promote_min_improvement 0.02 \
  --seq_wm_teacher_keep_last_k 3 \
  --seq_wm_teacher_eval_buffer_capacity 8 \
  --seq_wm_teacher_record_below_tol 0.04 \
  --seq_wm_teacher_require_recovery_pos_low 0.05 \
  --seq_wm_teacher_require_recovery_pos_high 0.95 \
  --seq_wm_enable_imagined_evaluator \
  --seq_wm_imagine_horizons "$SEQ_WM_HORIZON_STEPS" \
  --seq_wm_imagine_action_mode recorded \
  --seq_wm_imagine_num_starts 256 \
  --seq_wm_imagine_use_teacher \
  --seq_wm_replay_prioritize_floor_recovery \
  --no-experimental_rssm_floor_recovery_stack \
  --pbt_early_collapse_detect \
  --early_collapse_drop_ratio 0.3 \
  --early_collapse_patience 1 \
  --curriculum_collapse_detect \
  --collapse_drop_ratio 0.5 \
  --collapse_patience 2 \
  --collapse_freeze_env_frames 200000000 \
  --stable_source_min_mean 3.0 \
  --stable_source_min_max 20.0 \
  --pbt_recovery_bypass_elimination_burnin \
  --pbt_recovery_tolerance_mode last_mastered \
  --pbt_crosscopy_tolerance_mode "$CROSSCOPY_TOLERANCE_MODE" \
  --curriculum_rollback_cooldown_frames "$ROLLBACK_COOLDOWN_FRAMES" \
  >"$LOG" 2>&1 &

echo $! > "$PIDFILE"
echo "launch_master pid=$(cat "$PIDFILE") log=$LOG"
