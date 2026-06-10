#!/usr/bin/env bash
# Phase A4 P0 sanity AB — control arm.
#
# Single-policy, no PBT, fixed `rssm_belief_value`, --seq_wm_policy_coupling=none.
# This run mirrors the legacy (pre-P0) wiring: the seq WM trains its own
# optimizer, but PPO gradients are NEVER influenced by it. Used as the
# baseline for the AB comparison against the distill_value arm.
#
# Constraints (per user):
#   * single policy (num_policies=1) so PBT replacement noise cannot mask
#     a real effect-path effect,
#   * identical seed, num_envs, max_iterations, curriculum and seq_wm
#     hyperparameters as the distill arm — the ONLY difference is
#     `--seq_wm_policy_coupling`,
#   * 200 M frames target (~1528 epochs at 8192 envs * 16 horizon).
#
# Override env vars (must match the distill arm if you change them):
#   GPU                       default 0
#   NUM_ENVS                  default 8192
#   MAX_ITER                  default 1528  (~200 M frames)
#   SEED                      default 42
#   SEQ_WM_SEQ_LEN            default 16
#   SEQ_WM_BATCH              default 1024
#   SEQ_WM_LAMBDA_TOTAL       default 0.05
#   SEQ_WM_LAMBDA_VALUE       default 0.1
#   SEQ_WM_VALUE_DETACH       default false

set -euo pipefail

WMDEXPBT_ROOT="/root/autodl-tmp/wmdexpbt"
RUNTIME_ROOT="${WMDEXPBT_ROOT}/runtime_root"
RESULTS_ROOT="${WMDEXPBT_ROOT}/results"
DATE=$(date +%Y%m%d_%H%M%S)
RUN_DIR="${RUNS_ROOT}/regrasping_fixed_rssm_belief_value_no_policy_coupling_200m_${DATE}"
LOG="${RUN_DIR}/train.log"
WORKSPACE="${RUN_DIR}/dexpbt_ws"

GPU="${GPU:-0}"
NUM_ENVS="${NUM_ENVS:-8192}"
MAX_ITER="${MAX_ITER:-1528}"
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
echo "${RUN_DIR}" > "${RESULTS_ROOT}/pointers/latest_seq_wm_ab_run.txt"

cat > "${RUN_DIR}/RUN_CONFIG.txt" <<EOF
launcher=start_regrasping_fixed_rssm_belief_value_no_policy_coupling_200m.sh
profile=phase_a4_p0_ab_control_no_policy_coupling
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
seq_wm_policy_coupling=none
seq_wm_policy_coupling_warmup_frames=0
cf_runner_enabled=false
max_consecutive_successes=50
EOF

# shellcheck source=/dev/null
source "${RUNTIME_ROOT}/scripts/env_dexpbt.sh"

export CUDA_VISIBLE_DEVICES="${GPU}"
export TORCHDYNAMO_DISABLE=1
export PYTHONUNBUFFERED=1
export OMNI_KIT_TELEMETRY=0
export CARB_CRASH_REPORTER=0

cd "${RUNTIME_ROOT}/src/dexpbt_isaaclab"

echo "[seq_wm_ab] CONTROL run dir=${RUN_DIR}"
echo "[seq_wm_ab] strategy=rssm_belief_value coupling=none seed=${SEED} max_iter=${MAX_ITER}"

nohup python -u -m dexpbt_isaaclab.scripts.train \
  --task DexPBT-Regrasping-KukaAllegro-Direct-v0 \
  --train_root "${RUN_DIR}/rl_games" \
  --experiment_name "ab_no_coupling_200m_${DATE}" \
  --device=cuda:0 \
  --headless \
  --observer_mode=default \
  --workspace="${WORKSPACE}" \
  --num_envs="${NUM_ENVS}" \
  --max_iterations="${MAX_ITER}" \
  --seed="${SEED}" \
  --enable_wm \
  --wm_strategy=rssm_belief_value \
  --wm_strategy_mode=fixed \
  --fixed_wm_strategy=rssm_belief_value \
  --wm_strategy_pool=rssm_belief_value \
  --wm_load_mode=reset \
  --seq_wm_seq_len="${SEQ_WM_SEQ_LEN}" \
  --seq_wm_seq_batch_size="${SEQ_WM_BATCH}" \
  --seq_wm_lambda_total="${SEQ_WM_LAMBDA_TOTAL}" \
  --seq_wm_lambda_value="${SEQ_WM_LAMBDA_VALUE}" \
  ${SEQ_WM_VALUE_DETACH_FLAG} \
  --seq_wm_policy_coupling=none \
  --kit_args "${DEXPBT_KIT_ARGS}" \
  >"${LOG}" 2>&1 &

PID=$!
echo "${PID}" > "${RUN_DIR}/train.pid"
echo "[seq_wm_ab] pid=${PID} log=${LOG}"
echo "[seq_wm_ab] Tail with: tail -f ${LOG}"
