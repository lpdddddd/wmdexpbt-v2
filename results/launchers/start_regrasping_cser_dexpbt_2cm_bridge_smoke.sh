#!/usr/bin/env bash
# 2-policy frame-stop smoke: tiny target so one worker completes first while
# the other keeps training; supervisor must NOT restart the finished worker
# and must exit only after all workers finish.
set -euo pipefail

RUNTIME_ROOT="/root/autodl-tmp/wmdexpbt/runtime_root"
RESULTS_ROOT="/root/autodl-tmp/wmdexpbt/results"
DATE=$(date +%Y%m%d_%H%M%S)
RUN_DIR="${RESULTS_ROOT}/runs/smoke_cser_2cm_bridge_${DATE}"
MAX_ITER="${MAX_ITER:-4000}"           # safety cap only
TARGET_FRAMES="${TARGET_FRAMES:-400000000}"
NUM_POLICIES="${NUM_POLICIES:-2}"
DEXPBT_GPUS="${DEXPBT_GPUS:-0,1}"
PROFILE_LIST="cser_dexpbt_2cm_bridge,cser_dexpbt_2cm_bridge"

export NUM_POLICIES DEXPBT_GPUS
export DEXPBT_MAX_ACTIVE="${DEXPBT_MAX_ACTIVE:-2}"
mkdir -p "$RUN_DIR"
cat > "$RUN_DIR/RUN_CONFIG.txt" <<EOF
launcher=$(basename "$0")
method=cser_dexpbt_2cm_bridge
termination_mode=per_worker_population_frames
target_population_frames=${TARGET_FRAMES}
max_iterations=${MAX_ITER}
max_iterations_role=safety_cap_only
num_policies=${NUM_POLICIES}
num_envs=8192
smoke=true
EOF

source "${RUNTIME_ROOT}/scripts/env_dexpbt.sh"
cd "${RUNTIME_ROOT}/src/dexpbt_isaaclab"
nohup bash "${RUNTIME_ROOT}/scripts/launch_8gpu_dexpbt.sh" "$RUN_DIR" \
  --task DexPBT-Regrasping-KukaAllegro-Direct-v0 \
  --num-envs 8192 --max-iterations "$MAX_ITER" \
  --pbt-interval-steps 20000000 --pbt-start-after 40000000 --pbt-initial-delay 20000000 \
  --profile-per-policy "$PROFILE_LIST" --extra \
  --seed 42 --pbt_ranking_metric success_aware_r_meta \
  --pbt_disable_self_restart --no-pbt_allow_mid_self_mutation \
  --pbt_source_selection_mode tiered --pbt_source_require_stable_mean \
  --curriculum_mode mastery \
  --frame_stop_enable --target_population_frames "$TARGET_FRAMES" --frame_stop_grace_iterations 2 \
  --same_tol_retention_enable --bridge_curriculum_enable --bridge_probation_enable \
  --snippet_adjacent_stage_mixing_enable --snippet_tolerance_compatibility_enable \
  --late_degradation_rescue_enable \
  >"$RUN_DIR/launch_master.log" 2>&1 &
echo $! > "$RUN_DIR/nohup.pid"
echo "$RUN_DIR"
