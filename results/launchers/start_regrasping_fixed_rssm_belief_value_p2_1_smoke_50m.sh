#!/usr/bin/env bash
# Phase B smoke: single-policy fixed rssm_belief_value run, 50 M frames.
#
# Goal: cheap end-to-end sanity check that
#   - no OOM with default seq_wm hyperparameters,
#   - no NaN/Inf in seq_wm losses (loss_total monotonically smooth, fps
#     drop relative to fixed_h1 baseline < 10%),
#   - SeqWMTrainer + SeqAuxCollector wiring is correct under the live
#     PPO ExperienceBuffer / play_steps_rnn / train_epoch loop.
#
# This script bypasses launch_pbt entirely: PBT plumbing is unnecessary
# for a single-policy smoke and would mask backbone failures behind the
# usual restart machinery.
#
# Resource: GPU 0, num_envs=8192, ~10 minutes wall on RTX 4090.
#
# Override env vars (all optional):
#   GPU                   default 0
#   NUM_ENVS              default 8192
#   MAX_ITER              default 380   (~50 M frames at 8192 envs * 16)
#   SEQ_WM_SEQ_LEN        default 16
#   SEQ_WM_BATCH          default 1024
#   SEQ_WM_LAMBDA_TOTAL   default 0.05
#   SEQ_WM_LAMBDA_VALUE   default 0.1
#   SEQ_WM_VALUE_DETACH   default false  (per plan revision)
#   SEED                  default 42

set -euo pipefail

WMDEXPBT_ROOT="/root/autodl-tmp/wmdexpbt"
RUNTIME_ROOT="${WMDEXPBT_ROOT}/runtime_root"
RESULTS_ROOT="${WMDEXPBT_ROOT}/results"
DATE=$(date +%Y%m%d_%H%M%S)
RUN_DIR="${RUNS_ROOT}/regrasping_fixed_rssm_belief_value_smoke_50m_${DATE}"
LOG="${RUN_DIR}/train.log"
WORKSPACE="${RUN_DIR}/dexpbt_ws"

GPU="${GPU:-0}"
NUM_ENVS="${NUM_ENVS:-8192}"
MAX_ITER="${MAX_ITER:-380}"
SEED="${SEED:-42}"
SEQ_WM_SEQ_LEN="${SEQ_WM_SEQ_LEN:-16}"
SEQ_WM_BATCH="${SEQ_WM_BATCH:-1024}"
SEQ_WM_LAMBDA_TOTAL="${SEQ_WM_LAMBDA_TOTAL:-0.05}"
SEQ_WM_LAMBDA_VALUE="${SEQ_WM_LAMBDA_VALUE:-0.1}"
SEQ_WM_VALUE_DETACH_FLAG=""
case "${SEQ_WM_VALUE_DETACH:-false}" in
  1|true|TRUE|True) SEQ_WM_VALUE_DETACH_FLAG="--seq_wm_value_detach_belief" ;;
  *)               SEQ_WM_VALUE_DETACH_FLAG="--no-seq_wm_value_detach_belief" ;;
esac

mkdir -p "${RUN_DIR}"
echo "${RUN_DIR}" > "${RESULTS_ROOT}/pointers/latest_seq_wm_smoke_run.txt"

cat > "${RUN_DIR}/RUN_CONFIG.txt" <<EOF
launcher=start_regrasping_fixed_rssm_belief_value_p2_1_smoke_50m.sh
profile=phase_b_fixed_rssm_belief_value_smoke_50m
target_frames=$(( NUM_ENVS * 16 * MAX_ITER ))
max_iterations=${MAX_ITER}
num_policies=1
gpus=${GPU}
base_seed=${SEED}
wm_strategy=rssm_belief_value
wm_strategy_mode=fixed
seq_wm_seq_len=${SEQ_WM_SEQ_LEN}
seq_wm_seq_batch_size=${SEQ_WM_BATCH}
seq_wm_lambda_total=${SEQ_WM_LAMBDA_TOTAL}
seq_wm_lambda_value=${SEQ_WM_LAMBDA_VALUE}
seq_wm_value_detach_belief=${SEQ_WM_VALUE_DETACH:-false}
cf_runner_enabled=false
max_consecutive_successes=50
EOF

# Hard-check (constraint 1): the strategy MUST be one of the new sequence
# strategies; abort if invoked with the wrong --wm_strategy override.
WM_STRATEGY="rssm_belief_value"
case "${WM_STRATEGY}" in
  rssm_aux|rssm_belief_value|trans_aux|trans_belief_value) ;;
  *) echo "[FATAL] not a sequence strategy: ${WM_STRATEGY}" >&2; exit 1 ;;
esac

# shellcheck source=/dev/null
source "${RUNTIME_ROOT}/scripts/env_dexpbt.sh"

export CUDA_VISIBLE_DEVICES="${GPU}"
export TORCHDYNAMO_DISABLE=1
export PYTHONUNBUFFERED=1
export OMNI_KIT_TELEMETRY=0
export CARB_CRASH_REPORTER=0

cd "${RUNTIME_ROOT}/src/dexpbt_isaaclab"

echo "[seq_wm_smoke] RUN_DIR=${RUN_DIR}"
echo "[seq_wm_smoke] strategy=${WM_STRATEGY} seq_len=${SEQ_WM_SEQ_LEN} batch=${SEQ_WM_BATCH}"
echo "[seq_wm_smoke] num_envs=${NUM_ENVS} max_iterations=${MAX_ITER} gpu=${GPU} seed=${SEED}"

nohup python -u -m dexpbt_isaaclab.scripts.train \
  --task DexPBT-Regrasping-KukaAllegro-Direct-v0 \
  --train_root "${RUN_DIR}/rl_games" \
  --experiment_name "smoke_50m_${DATE}" \
  --device=cuda:0 \
  --headless \
  --observer_mode=default \
  --workspace="${WORKSPACE}" \
  --num_envs="${NUM_ENVS}" \
  --max_iterations="${MAX_ITER}" \
  --seed="${SEED}" \
  --enable_wm \
  --wm_strategy="${WM_STRATEGY}" \
  --wm_strategy_mode=fixed \
  --fixed_wm_strategy="${WM_STRATEGY}" \
  --wm_strategy_pool="${WM_STRATEGY}" \
  --wm_load_mode=reset \
  --seq_wm_seq_len="${SEQ_WM_SEQ_LEN}" \
  --seq_wm_seq_batch_size="${SEQ_WM_BATCH}" \
  --seq_wm_lambda_total="${SEQ_WM_LAMBDA_TOTAL}" \
  --seq_wm_lambda_value="${SEQ_WM_LAMBDA_VALUE}" \
  ${SEQ_WM_VALUE_DETACH_FLAG} \
  --kit_args "${DEXPBT_KIT_ARGS}" \
  >"${LOG}" 2>&1 &

PID=$!
echo "${PID}" > "${RUN_DIR}/train.pid"
echo "[seq_wm_smoke] pid=${PID} log=${LOG}"
echo "[seq_wm_smoke] Tail with: tail -f ${LOG}"
echo "[seq_wm_smoke] Watch wm_seq metrics: grep -E '\\[SeqWM\\]|loss_total|loss_kl|grad_norm' ${LOG}"
