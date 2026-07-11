#!/usr/bin/env bash
# ERD-MPC clean isolation 800M / policy.
#
# Tests action-conditioned recovery scorer + candidate-action distillation
# without WM active ranking / teacher / imagined evaluator / CF / h3 /
# RSSM->PPO distillation.

set -euo pipefail

RUNTIME_ROOT="/root/autodl-tmp/wmdexpbt/runtime_root"
RESULTS_ROOT="/root/autodl-tmp/wmdexpbt/results"
RUNS_ROOT="${RESULTS_ROOT}/runs"
DATE=$(date +%Y%m%d_%H%M%S)
RUN_DIR="${RUNS_ROOT}/regrasping_erd_mpc_800m_${DATE}"
LOG="${RUN_DIR}/launch_master.log"
PIDFILE="${RUN_DIR}/nohup.pid"
COMMIT_SHA="${COMMIT_SHA:-unknown}"

WM_STRATEGY="rssm_belief_value"
BASE_SEED="${BASE_SEED:-42}"
NUM_POLICIES="${NUM_POLICIES:-6}"
DEXPBT_GPUS="${DEXPBT_GPUS:-0,1,2,3,4,5}"
MAX_ITER="${MAX_ITER:-6104}"
PBT_START_AFTER="${PBT_START_AFTER:-200000000}"
PBT_INITIAL_DELAY="${PBT_INITIAL_DELAY:-50000000}"

export DEXPBT_MAX_ACTIVE="${DEXPBT_MAX_ACTIVE:-2}"
export DEXPBT_INTER_LAUNCH_DELAY_S="${DEXPBT_INTER_LAUNCH_DELAY_S:-90}"
export DEXPBT_STARTUP_TIMEOUT_S="${DEXPBT_STARTUP_TIMEOUT_S:-900}"
export DEXPBT_STARTUP_MAX_RESTARTS="${DEXPBT_STARTUP_MAX_RESTARTS:-1}"
export NUM_POLICIES
export DEXPBT_GPUS

mkdir -p "$RUN_DIR"
echo "$RUN_DIR" > "${RESULTS_ROOT}/pointers/latest_erd_mpc_800m_run.txt"
echo "$RUN_DIR" > "${RUN_DIR}/RUN_DIR.txt"

cat > "${RUN_DIR}/RUN_CONFIG.txt" <<EOF
launcher=start_regrasping_erd_mpc_800m.sh
profile=erd_mpc_clean_800m
commit=${COMMIT_SHA}
target_frames=800000000
max_iterations=${MAX_ITER}
num_policies=${NUM_POLICIES}
gpus=${DEXPBT_GPUS}
base_seed=${BASE_SEED}
wm_strategy_mode=fixed
fixed_wm_strategy=${WM_STRATEGY}
seq_wm_policy_coupling=none
pbt_wm_score_mode=off
seq_wm_enable_iterative_teacher=false
seq_wm_enable_imagined_evaluator=false
erd_enable=true
erd_candidate_k=32
erd_action_noise_std=0.15
erd_max_action_delta=0.5
erd_scorer_loss_weight=0.01
erd_distill_loss_weight=0.03
erd_floor_distill_loss_weight=0.05
erd_hard_floor_distill_loss_weight=0.08
erd_max_grad_ratio=0.2
erd_start_after_frames=200000000
erd_scorer_warmup_iters=2000
note=Clean ERD-MPC isolation smoke; action-level recovery distillation without active WM ranking.
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
  --max-iterations "$MAX_ITER" \
  --pbt-interval-steps 20000000 \
  --pbt-start-after "$PBT_START_AFTER" \
  --pbt-initial-delay "$PBT_INITIAL_DELAY" \
  --extra \
  --seed "$BASE_SEED" \
  --enable_wm \
  --wm_strategy "$WM_STRATEGY" \
  --wm_strategy_mode fixed \
  --fixed_wm_strategy "$WM_STRATEGY" \
  --wm_load_mode reset \
  --seq_wm_seq_len 16 \
  --seq_wm_seq_batch_size 1024 \
  --seq_wm_lambda_total 0.05 \
  --seq_wm_lambda_value 0.1 \
  --no-seq_wm_value_detach_belief \
  --seq_wm_policy_coupling none \
  --no-seq_wm_enable_iterative_teacher \
  --no-seq_wm_enable_imagined_evaluator \
  --pbt_wm_score_mode off \
  --pbt_ranking_metric success_aware_r_meta \
  --pbt_min_frames_as_source_after_restart 50000000 \
  --pbt_min_frames_between_restarts 50000000 \
  --pbt_disable_self_restart \
  --pbt_source_selection_mode tiered \
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
  --curriculum_fast_strong_mean 30.0 \
  --curriculum_fast_mid_mean 20.0 \
  --curriculum_fast_low_mean 10.0 \
  --curriculum_fast_target_tolerance_800m 0.04 \
  --pbt_curriculum_inherit_mode source_safe \
  --pbt_source_safe_recovery_consolidation_frames 50000000 \
  --pbt_source_unsafe_recovery_consolidation_frames 100000000 \
  --pbt_source_safe_min_mean 20.0 \
  --pbt_source_safe_min_retention 0.7 \
  --pbt_protect_promising_workers \
  --pbt_promising_nsucc_mean 20.0 \
  --pbt_promising_peak_mean 25.0 \
  --pbt_promising_min_retention 0.5 \
  --pbt_promising_protect_frames 150000000 \
  --erd_enable \
  --erd_horizons 4,8 \
  --erd_candidate_k 32 \
  --erd_action_noise_std 0.15 \
  --erd_max_action_delta 0.5 \
  --erd_noise_cap_by_tol \
  --erd_score_success_weight 1.0 \
  --erd_score_improve_weight 0.7 \
  --erd_score_drop_weight 1.5 \
  --erd_scorer_loss_weight 0.01 \
  --erd_distill_loss_weight 0.03 \
  --erd_floor_distill_loss_weight 0.05 \
  --erd_hard_floor_distill_loss_weight 0.08 \
  --erd_start_tol 0.04 \
  --erd_floor_tol 0.02 \
  --erd_hard_floor_tol 0.012 \
  --erd_dhat_improve_margin 0.005 \
  --erd_drop_dhat_margin 0.01 \
  --erd_near_success_mult 2.0 \
  --erd_max_grad_ratio 0.2 \
  --erd_auto_scale_by_grad_ratio \
  --erd_only_lifted \
  --erd_min_risky_fraction 0.02 \
  --erd_start_after_frames 200000000 \
  --erd_scorer_warmup_iters 2000 \
  --erd_min_score_margin 0.05 \
  --erd_min_target_valid_fraction 0.2 \
  --erd_max_drop_prob_for_teacher 0.5 \
  --no-experimental_rssm_floor_recovery_stack \
  --pbt_early_collapse_detect \
  --early_collapse_drop_ratio 0.3 \
  --early_collapse_patience 1 \
  --curriculum_collapse_detect \
  --collapse_drop_ratio 0.5 \
  --collapse_patience 2 \
  --collapse_freeze_env_frames 200000000 \
  --pbt_recovery_tolerance_mode last_mastered \
  --pbt_crosscopy_tolerance_mode last_mastered \
  >"$LOG" 2>&1 &

echo $! > "$PIDFILE"
echo "launch_master pid=$(cat "$PIDFILE") log=$LOG"

