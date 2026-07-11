#!/usr/bin/env bash
# Progress-Conditioned CSER-DexPBT ablation launcher.
#
# Selects an ablation variant via the CSER_ABLATION environment variable and
# writes the resolved component matrix to RUN_CONFIG.txt.  Ablations toggle the
# progress_refine components on top of the authoritative profile using
# --wm_strategy_profile_allow_cli_override so the specific enable flags win.
#
# Variants (CSER_ABLATION):
#   basev2                    : cser_dexpbt_conservative base (no progress refine)
#   snippet_only              : snippet distill, no progress gate, no WM rerank
#   snippet_wm                : snippet distill + WM rerank, no progress gate
#   progress_gate_only        : progress gate, no WM rerank
#   progress_gate_wm          : progress gate + WM rerank (no adaptive consolidation)
#   adaptive_consolidation_only: adaptive consolidation only (no gate, no WM rerank)
#   full                      : all components on (default)
set -euo pipefail

CSER_ABLATION="${CSER_ABLATION:-full}"

RUNTIME_ROOT="/root/autodl-tmp/wmdexpbt/runtime_root"
RESULTS_ROOT="/root/autodl-tmp/wmdexpbt/results"
RUNS_ROOT="${RESULTS_ROOT}/runs"
DATE=$(date +%Y%m%d_%H%M%S)
RUN_DIR="${RUNS_ROOT}/regrasping_cser_dexpbt_progress_refine_ablation_${CSER_ABLATION}_${DATE}"
LOG="${RUN_DIR}/launch_master.log"
PIDFILE="${RUN_DIR}/nohup.pid"
COMMIT_SHA="${COMMIT_SHA:-unknown}"

BASE_SEED="${BASE_SEED:-42}"
NUM_POLICIES="${NUM_POLICIES:-6}"
DEXPBT_GPUS="${DEXPBT_GPUS:-0,1,2,3,4,5}"
NUM_ENVS="${NUM_ENVS:-8192}"
MAX_ITER="${MAX_ITER:-50000}"
TARGET_FRAMES="${TARGET_FRAMES:-3000000000}"
PBT_START_AFTER="${PBT_START_AFTER:-200000000}"
PBT_INITIAL_DELAY="${PBT_INITIAL_DELAY:-50000000}"

# Base profile + component toggles per variant.
PROFILE="cser_dexpbt_progress_refine"
ABLATION_FLAGS=(--wm_strategy_profile_allow_cli_override)
case "$CSER_ABLATION" in
  basev2)
    PROFILE="cser_dexpbt_conservative"
    ABLATION_FLAGS=()
    ;;
  snippet_only)
    ABLATION_FLAGS+=(--no-progress_refinement_enable --no-snippet_wm_rerank_enable
                     --no-adaptive_precision_consolidation_enable --no-adaptive_rescue_enable)
    ;;
  snippet_wm)
    ABLATION_FLAGS+=(--no-progress_refinement_enable
                     --no-adaptive_precision_consolidation_enable --no-adaptive_rescue_enable)
    ;;
  progress_gate_only)
    ABLATION_FLAGS+=(--no-snippet_wm_rerank_enable
                     --no-adaptive_precision_consolidation_enable --no-adaptive_rescue_enable)
    ;;
  progress_gate_wm)
    ABLATION_FLAGS+=(--no-adaptive_precision_consolidation_enable --no-adaptive_rescue_enable)
    ;;
  adaptive_consolidation_only)
    ABLATION_FLAGS+=(--no-progress_refinement_enable --no-snippet_wm_rerank_enable)
    ;;
  full)
    ABLATION_FLAGS=()  # profile default is fully on
    ;;
  *)
    echo "Unknown CSER_ABLATION=$CSER_ABLATION" >&2
    exit 2
    ;;
esac

PROFILE_LIST="$(printf '%s,' $(seq 1 "$NUM_POLICIES" | sed "s/.*/${PROFILE}/") | sed 's/,$//')"

export DEXPBT_MAX_ACTIVE="${DEXPBT_MAX_ACTIVE:-${NUM_POLICIES}}"
export DEXPBT_INTER_LAUNCH_DELAY_S="${DEXPBT_INTER_LAUNCH_DELAY_S:-180}"
export NUM_POLICIES
export DEXPBT_GPUS

mkdir -p "$RUN_DIR" "${RESULTS_ROOT}/pointers"
echo "$RUN_DIR" > "${RESULTS_ROOT}/pointers/latest_cser_dexpbt_progress_refine_ablation_run.txt"

cat > "${RUN_DIR}/RUN_CONFIG.txt" <<EOF
launcher=start_regrasping_cser_dexpbt_progress_refine_ablation.sh
method=cser_dexpbt_progress_refine
ablation=${CSER_ABLATION}
profile=${PROFILE}
ablation_flags=${ABLATION_FLAGS[*]:-none}
commit=${COMMIT_SHA}
num_policies=${NUM_POLICIES}
num_envs=${NUM_ENVS}
gpus=${DEXPBT_GPUS}
target_population_frames=${TARGET_FRAMES}
max_iterations=${MAX_ITER}
base_seed=${BASE_SEED}
note=Ablation matrix over progress gate / WM rerank / adaptive consolidation.
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
  --pbt-interval-steps 20000000 \
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
  --no-pbt_wm_profile_copy_enable \
  --no-awr_enable \
  --no-eir_residual_apply_to_rollout \
  --no-eir_use_rssm_mppi \
  --eir_mppi_mode off \
  --frame_stop_enable \
  --target_population_frames "$TARGET_FRAMES" \
  --frame_stop_grace_iterations 2 \
  "${ABLATION_FLAGS[@]}" \
  >>"$LOG" 2>&1 &

echo $! > "$PIDFILE"
echo "launch_master pid=$(cat "$PIDFILE") ablation=$CSER_ABLATION log=$LOG run_dir=$RUN_DIR"
