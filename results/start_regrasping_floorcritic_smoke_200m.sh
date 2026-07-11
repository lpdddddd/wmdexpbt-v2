#!/usr/bin/env bash
# FloorCritic Phase 0 — detached smoke (200M / policy).
#
# Purpose: ENGINEERING validation only. Verify:
#   * FloorCritic targets build + are non-degenerate,
#   * the FloorCritic loss is active (wm_floorcritic/active==1),
#   * actor_grad_blocked stays true (detached feature_source),
#   * TB keys exist, no crash.
# NOT a performance experiment.
#
# Config (per FLOORCRITIC_IMPLEMENTATION_PLAN.md, section H):
#   - feature_source = actor_mlp_out_detached (Phase 0)
#   - weight 0.005
#   - NO active WM ranking (pbt_wm_score_mode off)
#   - NO iterative teacher / imagined evaluator
#   - NO seq distillation (seq_wm_policy_coupling none)
#   - NO h3, NO counterfactual
#   - start_tol RAISED to 0.1 so FloorCritic activates across the whole
#     smoke (initial curriculum tol ~0.075 would otherwise keep lambda=0
#     for the entire 200M; this is a smoke, we WANT the loss exercised).
#
# Hard constraints preserved: reward / env / PPO actor-critic UNCHANGED.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNTIME_ROOT="/root/autodl-tmp/wmdexpbt/runtime_root"
RESULTS_ROOT="/root/autodl-tmp/wmdexpbt/results"
RUNS_ROOT="${RESULTS_ROOT}/runs"
DATE=$(date +%Y%m%d_%H%M%S)
RUN_DIR="${RUNS_ROOT}/regrasping_floorcritic_smoke_200m_${DATE}"
LOG="${RUN_DIR}/launch_master.log"
PIDFILE="${RUN_DIR}/nohup.pid"

WM_STRATEGY="rssm_belief_value"
BASE_SEED="${BASE_SEED:-42}"
NUM_POLICIES="${NUM_POLICIES:-6}"
DEXPBT_GPUS="${DEXPBT_GPUS:-0,1,2,3,4,5}"
# 200M / policy: 8192 envs * 16 horizon = 131072 frames/iter -> ~1526 iters.
MAX_ITER="${MAX_ITER:-1526}"
PBT_START_AFTER="${PBT_START_AFTER:-100000000}"
PBT_INITIAL_DELAY="${PBT_INITIAL_DELAY:-30000000}"

# FloorCritic (Phase 0 detached smoke).
FC_FEATURE_SOURCE="${FC_FEATURE_SOURCE:-actor_mlp_out_detached}"
FC_WEIGHT="${FC_WEIGHT:-0.005}"
FC_FLOOR_WEIGHT="${FC_FLOOR_WEIGHT:-0.01}"
FC_HARD_FLOOR_WEIGHT="${FC_HARD_FLOOR_WEIGHT:-0.02}"
FC_START_TOL="${FC_START_TOL:-0.1}"
FC_FLOOR_TOL="${FC_FLOOR_TOL:-0.05}"
FC_HARD_FLOOR_TOL="${FC_HARD_FLOOR_TOL:-0.03}"
FC_HORIZONS="${FC_HORIZONS:-4,8}"

export DEXPBT_MAX_ACTIVE="${DEXPBT_MAX_ACTIVE:-2}"
export DEXPBT_INTER_LAUNCH_DELAY_S="${DEXPBT_INTER_LAUNCH_DELAY_S:-90}"
export DEXPBT_STARTUP_TIMEOUT_S="${DEXPBT_STARTUP_TIMEOUT_S:-900}"
export DEXPBT_STARTUP_MAX_RESTARTS="${DEXPBT_STARTUP_MAX_RESTARTS:-1}"
export NUM_POLICIES
export DEXPBT_GPUS

mkdir -p "$RUN_DIR"
echo "$RUN_DIR" > "${RESULTS_ROOT}/pointers/latest_floorcritic_smoke_200m_run.txt"
echo "$RUN_DIR" > "${RUN_DIR}/RUN_DIR.txt"

cat > "${RUN_DIR}/RUN_CONFIG.txt" <<EOF
launcher=start_regrasping_floorcritic_smoke_200m.sh
profile=floorcritic_phase0_detached_smoke_200m
target_frames=200000000
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
wm_floorcritic_start_tol=${FC_START_TOL}
wm_floorcritic_floor_tol=${FC_FLOOR_TOL}
wm_floorcritic_hard_floor_tol=${FC_HARD_FLOOR_TOL}
note=Phase0 engineering smoke; start_tol raised so FC is active across 200M.
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
  --curriculum_mode mastery \
  --pbt_source_require_stable_mean \
  --mastery_nsucc_mean 4.0 \
  --mastery_nsucc_max 20.0 \
  --mastery_consecutive_windows 2 \
  --curriculum_shrink_factor 0.95 \
  --pbt_single_shrink_per_window \
  --post_shrink_consolidation_env_frames 100000000 \
  --curriculum_speed_mode adaptive_fast \
  --pbt_curriculum_inherit_mode source_safe \
  --pbt_source_safe_min_mean 20.0 \
  --pbt_source_safe_min_retention 0.7 \
  --wm_floorcritic_enable \
  --wm_floorcritic_feature_source "$FC_FEATURE_SOURCE" \
  --wm_floorcritic_weight "$FC_WEIGHT" \
  --wm_floorcritic_floor_weight "$FC_FLOOR_WEIGHT" \
  --wm_floorcritic_hard_floor_weight "$FC_HARD_FLOOR_WEIGHT" \
  --wm_floorcritic_start_tol "$FC_START_TOL" \
  --wm_floorcritic_floor_tol "$FC_FLOOR_TOL" \
  --wm_floorcritic_hard_floor_tol "$FC_HARD_FLOOR_TOL" \
  --wm_floorcritic_horizons "$FC_HORIZONS" \
  --wm_floorcritic_actor_detach \
  --no-wm_floorcritic_update_actor \
  --no-experimental_rssm_floor_recovery_stack \
  >"$LOG" 2>&1 &

echo $! > "$PIDFILE"
echo "launch_master pid=$(cat "$PIDFILE") log=$LOG"
