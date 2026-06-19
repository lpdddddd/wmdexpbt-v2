#!/usr/bin/env bash
# Smoke launcher for Strategy Zoo v1.7.1 rollout-buffer Online Elite-AWR.
# Validates collection path + action convention without a full 2B run.

set -euo pipefail

RUNTIME_ROOT="/root/autodl-tmp/wmdexpbt/runtime_root"
RESULTS_ROOT="/root/autodl-tmp/wmdexpbt/results"
RUNS_ROOT="${RESULTS_ROOT}/runs"
DATE=$(date +%Y%m%d_%H%M%S)
RUN_DIR="${RUNS_ROOT}/regrasping_pbt_wm_strategy_zoo_v17_1_online_elite_awr_smoke_${DATE}"
LOG="${RUN_DIR}/launch_master.log"
PIDFILE="${RUN_DIR}/nohup.pid"
COMMIT_SHA="${COMMIT_SHA:-unknown}"

BASE_SEED="${BASE_SEED:-42}"
NUM_POLICIES="${NUM_POLICIES:-6}"
DEXPBT_GPUS="${DEXPBT_GPUS:-0,1,2,3,4,5}"
MAX_ITER="${MAX_ITER:-400}"
PBT_START_AFTER="${PBT_START_AFTER:-50000000}"
PBT_INITIAL_DELAY="${PBT_INITIAL_DELAY:-10000000}"

PROFILE_LIST="ppo_only,elite_bank_actor_distill_fast_awr,wm_pbt_source_selector_v2,wm_pbt_source_selector_awr,elite_bank_actor_distill_awr,elite_bank_actor_distill_strong_awr"

export DEXPBT_MAX_ACTIVE="${DEXPBT_MAX_ACTIVE:-6}"
export DEXPBT_INTER_LAUNCH_DELAY_S="${DEXPBT_INTER_LAUNCH_DELAY_S:-60}"
export DEXPBT_STARTUP_TIMEOUT_S="${DEXPBT_STARTUP_TIMEOUT_S:-1200}"
export DEXPBT_STARTUP_MAX_RESTARTS="${DEXPBT_STARTUP_MAX_RESTARTS:-1}"
export NUM_POLICIES
export DEXPBT_GPUS

mkdir -p "$RUN_DIR"
echo "$RUN_DIR" > "${RESULTS_ROOT}/pointers/latest_pbt_wm_strategy_zoo_v17_1_online_elite_awr_smoke_run.txt"
echo "$RUN_DIR" > "${RUN_DIR}/RUN_DIR.txt"

cat > "${RUN_DIR}/RUN_CONFIG.txt" <<EOF
launcher=start_regrasping_pbt_wm_strategy_zoo_v17_1_online_elite_awr_smoke.sh
profile=strategy_zoo_v1.7.1_online_elite_awr_rollout_buffer
config_version=strategy_zoo_v1.7.1_online_elite_awr_rollout_buffer
commit=${COMMIT_SHA}
smoke_mode=true
not_for_performance_comparison=true
target_frames=50000000
max_iterations=${MAX_ITER}
awr_collection_mode=rollout_buffer
awr_collect_after_frames=0
awr_start_after_frames=0
awr_min_buffer_size=1000
awr_kl_gate_enable=false
awr_kl_telemetry_only=true
awr_broadcast_enabled=false
awr_inherit_enabled=true
pbt_copy_reset_progress_on_large_frame_gap=true
pbt_copy_frame_gap_reset_threshold=500000000
note=smoke only — not for FPS comparison vs v13.
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
  --profile-per-policy "$PROFILE_LIST" \
  --extra \
  --seed "$BASE_SEED" \
  --awr_collection_mode rollout_buffer \
  --awr_collect_after_frames 0 \
  --awr_start_after_frames 0 \
  --awr_min_buffer_size 1000 \
  --awr_batch_steps 4096 \
  --awr_max_episode_len 200 \
  --awr_min_burnin_steps 32 \
  --awr_min_loss_steps_per_entry 16 \
  --awr_copy_top_k_steps 50000 \
  --awr_broadcast_top_k_steps 10000 \
  --awr_ramp_frames 300000000 \
  --awr_update_every 4 \
  --awr_buffer_capacity 400000 \
  --awr_loss_cap 0.03 \
  --pbt_copy_reset_progress_on_large_frame_gap \
  --pbt_copy_frame_gap_reset_threshold 500000000 \
  --pbt_ranking_metric success_aware_r_meta \
  --pbt_disable_self_restart \
  --pbt_source_selection_mode tiered \
  --curriculum_mode mastery \
  --no-eir_residual_apply_to_rollout \
  --pbt_wm_profile_copy_enable \
  >"$LOG" 2>&1 &

echo $! > "$PIDFILE"
echo "launch_master pid=$(cat "$PIDFILE") log=$LOG"
