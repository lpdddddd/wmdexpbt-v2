#!/usr/bin/env bash
# FloorCritic Phase 1 — effective (shared) 800M / policy.
#
# Purpose: compare against the 800M DexPBT baseline to confirm FloorCritic
# does NOT hurt nsucc, while regularising the shared backbone in the floor
# stage. Watch wm_floorcritic/fc_grad_ratio_to_ppo_grad + auto_scale_factor
# + early nsucc; if nsucc drops early -> lower weight or fall back to
# actor_mlp_out_detached.
#
# Config (per FLOORCRITIC_IMPLEMENTATION_PLAN.md, section H):
#   - feature_source = actor_mlp_out_shared (Phase 1 main experiment)
#   - weight 0.001 (NOT 0.005), floor 0.002, hard_floor 0.004
#   - production tol thresholds (start 0.04 / floor 0.02 / hard 0.012)
#   - max_grad_ratio 0.1 + auto_scale on
#   - keep DexPBT adaptive_fast + source_safe
#   - NO active WM ranking, NO teacher/imagined, NO distillation, NO h3, NO CF
#
# Hard constraints preserved: reward / env / PPO actor-critic UNCHANGED;
# FloorCritic never connects the mu (actor logits) head.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNTIME_ROOT="/root/autodl-tmp/wmdexpbt/runtime_root"
RESULTS_ROOT="/root/autodl-tmp/wmdexpbt/results"
RUNS_ROOT="${RESULTS_ROOT}/runs"
DATE=$(date +%Y%m%d_%H%M%S)
RUN_NAME_PREFIX="${RUN_NAME_PREFIX:-regrasping_floorcritic_800m}"
RUN_DIR="${RUNS_ROOT}/${RUN_NAME_PREFIX}_${DATE}"
LOG="${RUN_DIR}/launch_master.log"
PIDFILE="${RUN_DIR}/nohup.pid"

WM_STRATEGY="rssm_belief_value"
BASE_SEED="${BASE_SEED:-42}"
NUM_POLICIES="${NUM_POLICIES:-6}"
DEXPBT_GPUS="${DEXPBT_GPUS:-0,1,2,3,4,5}"
# 800M / policy: 131072 frames/iter -> ~6104 iters.
MAX_ITER="${MAX_ITER:-6104}"
PBT_START_AFTER="${PBT_START_AFTER:-200000000}"
PBT_INITIAL_DELAY="${PBT_INITIAL_DELAY:-50000000}"

# FloorCritic (Phase 1 shared, small weight).
FC_FEATURE_SOURCE="${FC_FEATURE_SOURCE:-actor_mlp_out_shared}"
FC_WEIGHT="${FC_WEIGHT:-0.001}"
FC_FLOOR_WEIGHT="${FC_FLOOR_WEIGHT:-0.002}"
FC_HARD_FLOOR_WEIGHT="${FC_HARD_FLOOR_WEIGHT:-0.004}"
FC_START_TOL="${FC_START_TOL:-0.04}"
FC_FLOOR_TOL="${FC_FLOOR_TOL:-0.02}"
FC_HARD_FLOOR_TOL="${FC_HARD_FLOOR_TOL:-0.012}"
FC_HORIZONS="${FC_HORIZONS:-4,8}"
FC_MAX_GRAD_RATIO="${FC_MAX_GRAD_RATIO:-0.1}"

FAST_STRONG_MEAN="${FAST_STRONG_MEAN:-30.0}"
FAST_MID_MEAN="${FAST_MID_MEAN:-20.0}"
FAST_LOW_MEAN="${FAST_LOW_MEAN:-10.0}"
FAST_TARGET_TOLERANCE_800M="${FAST_TARGET_TOLERANCE_800M:-0.04}"

export DEXPBT_MAX_ACTIVE="${DEXPBT_MAX_ACTIVE:-2}"
export DEXPBT_INTER_LAUNCH_DELAY_S="${DEXPBT_INTER_LAUNCH_DELAY_S:-90}"
export DEXPBT_STARTUP_TIMEOUT_S="${DEXPBT_STARTUP_TIMEOUT_S:-900}"
export DEXPBT_STARTUP_MAX_RESTARTS="${DEXPBT_STARTUP_MAX_RESTARTS:-1}"
export NUM_POLICIES
export DEXPBT_GPUS

mkdir -p "$RUN_DIR"
echo "$RUN_DIR" > "${RESULTS_ROOT}/pointers/latest_${RUN_NAME_PREFIX}_run.txt"
echo "$RUN_DIR" > "${RUN_DIR}/RUN_DIR.txt"

cat > "${RUN_DIR}/RUN_CONFIG.txt" <<EOF
launcher=start_regrasping_floorcritic_800m.sh
profile=floorcritic_phase1_shared_800m
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
wm_floorcritic_enable=true
wm_floorcritic_feature_source=${FC_FEATURE_SOURCE}
wm_floorcritic_weight=${FC_WEIGHT}
wm_floorcritic_floor_weight=${FC_FLOOR_WEIGHT}
wm_floorcritic_hard_floor_weight=${FC_HARD_FLOOR_WEIGHT}
wm_floorcritic_max_grad_ratio=${FC_MAX_GRAD_RATIO}
wm_floorcritic_auto_scale_by_grad_ratio=true
curriculum_speed_mode=adaptive_fast
pbt_curriculum_inherit_mode=source_safe
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
  --curriculum_fast_strong_mean "$FAST_STRONG_MEAN" \
  --curriculum_fast_mid_mean "$FAST_MID_MEAN" \
  --curriculum_fast_low_mean "$FAST_LOW_MEAN" \
  --curriculum_fast_target_tolerance_800m "$FAST_TARGET_TOLERANCE_800M" \
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
  --wm_floorcritic_enable \
  --wm_floorcritic_feature_source "$FC_FEATURE_SOURCE" \
  --wm_floorcritic_weight "$FC_WEIGHT" \
  --wm_floorcritic_floor_weight "$FC_FLOOR_WEIGHT" \
  --wm_floorcritic_hard_floor_weight "$FC_HARD_FLOOR_WEIGHT" \
  --wm_floorcritic_start_tol "$FC_START_TOL" \
  --wm_floorcritic_floor_tol "$FC_FLOOR_TOL" \
  --wm_floorcritic_hard_floor_tol "$FC_HARD_FLOOR_TOL" \
  --wm_floorcritic_horizons "$FC_HORIZONS" \
  --wm_floorcritic_max_grad_ratio "$FC_MAX_GRAD_RATIO" \
  --wm_floorcritic_auto_scale_by_grad_ratio \
  --no-wm_floorcritic_actor_detach \
  --no-wm_floorcritic_update_actor \
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
