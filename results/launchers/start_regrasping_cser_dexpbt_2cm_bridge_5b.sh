#!/usr/bin/env bash
# CSER-DexPBT 2cm Bridge formal launcher.  max iterations is safety-only:
# workers terminate individually after their authoritative population frame
# reaches the target.
set -euo pipefail

RUNTIME_ROOT="/root/autodl-tmp/wmdexpbt/runtime_root"
RESULTS_ROOT="/root/autodl-tmp/wmdexpbt/results"
DATE=$(date +%Y%m%d_%H%M%S)
RUN_DIR="${RESULTS_ROOT}/runs/regrasping_cser_dexpbt_2cm_bridge_5b_${DATE}"
MAX_ITER="${MAX_ITER:-70000}"
TARGET_FRAMES="${TARGET_FRAMES:-5000000000}"
PROFILE_LIST="cser_dexpbt_2cm_bridge,cser_dexpbt_2cm_bridge,cser_dexpbt_2cm_bridge,cser_dexpbt_2cm_bridge,cser_dexpbt_2cm_bridge,cser_dexpbt_2cm_bridge"

mkdir -p "$RUN_DIR"
cat > "$RUN_DIR/RUN_CONFIG.txt" <<EOF
launcher=$(basename "$0")
method=cser_dexpbt_2cm_bridge
termination_mode=per_worker_population_frames
target_population_frames=${TARGET_FRAMES}
max_iterations=${MAX_ITER}
max_iterations_role=safety_cap_only
num_policies=6
num_envs=8192
profile_all_workers=cser_dexpbt_2cm_bridge
EOF

source "${RUNTIME_ROOT}/scripts/env_dexpbt.sh"
cd "${RUNTIME_ROOT}/src/dexpbt_isaaclab"
nohup bash "${RUNTIME_ROOT}/scripts/launch_8gpu_dexpbt.sh" "$RUN_DIR" \
  --task DexPBT-Regrasping-KukaAllegro-Direct-v0 \
  --num-envs 8192 --max-iterations "$MAX_ITER" \
  --pbt-interval-steps 20000000 --pbt-start-after 200000000 --pbt-initial-delay 50000000 \
  --profile-per-policy "$PROFILE_LIST" --extra \
  --seed 42 --pbt_ranking_metric success_aware_r_meta \
  --pbt_min_frames_as_source_after_restart 50000000 \
  --pbt_min_frames_between_restarts 100000000 \
  --pbt_disable_self_restart --no-pbt_allow_mid_self_mutation \
  --pbt_source_selection_mode tiered --pbt_source_require_stable_mean \
  --curriculum_mode mastery --mastery_nsucc_mean 4 --mastery_nsucc_max 20 \
  --mastery_consecutive_windows 2 --curriculum_speed_mode adaptive_fast \
  --pbt_consolidation_enable --pbt_consolidation_stage1_end_frames 2200000000 \
  --pbt_consolidation_stage2_end_frames 3500000000 --pbt_consolidation_stage3_end_frames 5000000000 \
  --pbt_consolidation_stage3_hard_collapse_only \
  --frame_stop_enable --target_population_frames "$TARGET_FRAMES" --frame_stop_grace_iterations 2 \
  --same_tol_retention_enable --bridge_curriculum_enable --bridge_probation_enable \
  --snippet_adjacent_stage_mixing_enable --snippet_tolerance_compatibility_enable \
  --pbt_floor_recovery_aware --no-pbt_allow_mid_self_mutation \
  >"$RUN_DIR/launch_master.log" 2>&1 &
echo $! > "$RUN_DIR/nohup.pid"
echo "$RUN_DIR"
