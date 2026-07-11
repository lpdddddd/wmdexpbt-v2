#!/usr/bin/env bash
# Progress-Conditioned CSER-DexPBT smoke launcher.
#
# Short, low-frame run to validate the full progress_refine path end to end:
#   - authoritative profile expansion (id 22)
#   - elite snippet bank update (schema v2 with progress context)
#   - progress-context construction in the rollout cache
#   - row-wise progress-conditioned gate
#   - continuous log-relative tolerance-conditioned retrieval
#   - WM-unavailable fallback to priority ranking
#   - snippet distill loss backward
#   - checkpoint + PBT restart (profile re-expanded)
#   - progress_refine/* and adaptive_consolidation/* TensorBoard metrics
#
# Defaults are intentionally tiny; override via env vars.
set -euo pipefail

RUNTIME_ROOT="/root/autodl-tmp/wmdexpbt/runtime_root"
RESULTS_ROOT="/root/autodl-tmp/wmdexpbt/results"
RUNS_ROOT="${RESULTS_ROOT}/runs"
DATE=$(date +%Y%m%d_%H%M%S)
RUN_DIR="${RUNS_ROOT}/regrasping_cser_dexpbt_progress_refine_smoke_${DATE}"
LOG="${RUN_DIR}/launch_master.log"
PIDFILE="${RUN_DIR}/nohup.pid"
COMMIT_SHA="${COMMIT_SHA:-unknown}"

BASE_SEED="${BASE_SEED:-42}"
NUM_POLICIES="${NUM_POLICIES:-2}"
DEXPBT_GPUS="${DEXPBT_GPUS:-0,1}"
NUM_ENVS="${NUM_ENVS:-2048}"
MAX_ITER="${MAX_ITER:-4000}"
TARGET_FRAMES="${TARGET_FRAMES:-400000000}"
PBT_START_AFTER="${PBT_START_AFTER:-40000000}"
PBT_INITIAL_DELAY="${PBT_INITIAL_DELAY:-10000000}"
PBT_INTERVAL="${PBT_INTERVAL:-20000000}"

PROFILE="cser_dexpbt_progress_refine"
PROFILE_LIST="$(printf '%s,' $(seq 1 "$NUM_POLICIES" | sed "s/.*/${PROFILE}/") | sed 's/,$//')"

export DEXPBT_MAX_ACTIVE="${DEXPBT_MAX_ACTIVE:-${NUM_POLICIES}}"
export DEXPBT_INTER_LAUNCH_DELAY_S="${DEXPBT_INTER_LAUNCH_DELAY_S:-30}"
export DEXPBT_STARTUP_TIMEOUT_S="${DEXPBT_STARTUP_TIMEOUT_S:-1200}"
export NUM_POLICIES
export DEXPBT_GPUS

mkdir -p "$RUN_DIR" "${RESULTS_ROOT}/pointers"
echo "$RUN_DIR" > "${RESULTS_ROOT}/pointers/latest_cser_dexpbt_progress_refine_smoke_run.txt"

cat > "${RUN_DIR}/RUN_CONFIG.txt" <<EOF
launcher=start_regrasping_cser_dexpbt_progress_refine_smoke.sh
method=cser_dexpbt_progress_refine
profile_id=22
mode=smoke
commit=${COMMIT_SHA}
num_policies=${NUM_POLICIES}
num_envs=${NUM_ENVS}
gpus=${DEXPBT_GPUS}
target_population_frames=${TARGET_FRAMES}
max_iterations=${MAX_ITER}
progress_refinement=on
consolidation_schedule_mode=curriculum_progress
note=Smoke validation of the full progress_refine path; not a formal experiment.
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
  --num-envs "$NUM_ENVS" \
  --max-iterations "$MAX_ITER" \
  --pbt-interval-steps "$PBT_INTERVAL" \
  --pbt-start-after "$PBT_START_AFTER" \
  --pbt-initial-delay "$PBT_INITIAL_DELAY" \
  --profile-per-policy "$PROFILE_LIST" \
  --extra \
  --seed "$BASE_SEED" \
  --pbt_ranking_metric success_aware_r_meta \
  --pbt_disable_self_restart \
  --no-pbt_allow_mid_self_mutation \
  --pbt_source_selection_mode tiered \
  --curriculum_mode mastery \
  --mastery_nsucc_mean 4.0 \
  --mastery_nsucc_max 20.0 \
  --pbt_consolidation_enable \
  --pbt_consolidation_schedule_mode curriculum_progress \
  --adaptive_precision_consolidation_enable \
  --adaptive_rescue_enable \
  --progress_refinement_start_after_frames 40000000 \
  --progress_refinement_ramp_frames 40000000 \
  --no-pbt_wm_profile_copy_enable \
  --no-awr_enable \
  --no-eir_residual_apply_to_rollout \
  --no-eir_use_rssm_mppi \
  --eir_mppi_mode off \
  --frame_stop_enable \
  --target_population_frames "$TARGET_FRAMES" \
  --frame_stop_grace_iterations 2 \
  >>"$LOG" 2>&1 &

echo $! > "$PIDFILE"
echo "launch_master pid=$(cat "$PIDFILE") log=$LOG run_dir=$RUN_DIR"
