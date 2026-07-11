#!/usr/bin/env bash
# Fork policy 05 from an existing strict-5B run at a chosen checkpoint iter and
# continue for a short budget to validate crossing 2cm under the bridge stack.
# Preserves weights + optimizer, restores the authoritative population frame,
# and clears stale cross-tolerance retention state.
set -euo pipefail

RUNTIME_ROOT="/root/autodl-tmp/wmdexpbt/runtime_root"
RESULTS_ROOT="/root/autodl-tmp/wmdexpbt/results"
SRC_RUN="${SRC_RUN:-${RESULTS_ROOT}/runs/regrasping_cser_dexpbt_conservative_retention_strict_5b_20260709_193412}"
SRC_WS="${SRC_RUN}/dexpbt_8gpu_ws"
SRC_ITER="${SRC_ITER:-131}"
SRC_CKPT="${SRC_CKPT:-${SRC_WS}/005/$(printf '%06d' "${SRC_ITER}").pth}"
START_TOL="${START_TOL:-0.021}"
CONTINUE_FRAMES="${CONTINUE_FRAMES:-800000000}"
MAX_ITER="${MAX_ITER:-9000}"
DATE=$(date +%Y%m%d_%H%M%S)
RUN_DIR="${RESULTS_ROOT}/runs/cont_cser_2cm_bridge_p05_${DATE}"

if [[ ! -f "$SRC_CKPT" ]]; then
  echo "[resume] source checkpoint not found: $SRC_CKPT" >&2
  echo "[resume] provide SRC_RUN / SRC_ITER / SRC_CKPT env vars to point at a real checkpoint." >&2
  exit 2
fi

mkdir -p "$RUN_DIR"
cat > "$RUN_DIR/RUN_CONFIG.txt" <<EOF
launcher=$(basename "$0")
method=cser_dexpbt_2cm_bridge
mode=continuation_policy05
source_checkpoint=${SRC_CKPT}
start_tolerance=${START_TOL}
termination_mode=per_worker_population_frames
target_population_frames=$(python3 -c "print(${CONTINUE_FRAMES})")
max_iterations=${MAX_ITER}
max_iterations_role=safety_cap_only
EOF

source "${RUNTIME_ROOT}/scripts/env_dexpbt.sh"
cd "${RUNTIME_ROOT}/src/dexpbt_isaaclab"
nohup python -u -m dexpbt_isaaclab.scripts.train \
  --task DexPBT-Regrasping-KukaAllegro-Direct-v0 --headless \
  --num_envs 8192 --max_iterations "$MAX_ITER" \
  --observer_mode pbt --policy_idx 0 --num_policies 1 \
  --workspace "$RUN_DIR/dexpbt_8gpu_ws" \
  --wm_strategy_profile cser_dexpbt_2cm_bridge \
  --checkpoint "$SRC_CKPT" \
  --target_tolerance_resume "$START_TOL" \
  --frame_stop_enable --target_population_frames "$CONTINUE_FRAMES" \
  --same_tol_retention_enable --same_tol_retention_reset_on_cross_copy \
  --bridge_curriculum_enable --bridge_probation_enable \
  --snippet_adjacent_stage_mixing_enable --snippet_tolerance_compatibility_enable \
  --late_degradation_rescue_enable \
  >"$RUN_DIR/launch_master.log" 2>&1 &
echo $! > "$RUN_DIR/nohup.pid"
echo "$RUN_DIR"
