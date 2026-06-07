#!/usr/bin/env bash
# 6 卡 DexPBT 启动（每卡 1 个 policy，共 6 个并行 Isaac 进程）
#
# 用法示例（冒烟，少 env、少 iteration）:
#   bash /root/autodl-tmp/wmdexpbt/runtime_root/scripts/launch_8gpu_dexpbt.sh \
#     /root/autodl-tmp/wmdexpbt/results/smoke_6gpu_$(date +%Y%m%d_%H%M) \
#     --num-envs 256 --max-iterations 10
#
# 正式训练:
#   bash .../launch_8gpu_dexpbt.sh /path/to/run --num-envs 8192 --max-iterations 38147
#
# Staggered launch knobs (env vars, optional):
#   DEXPBT_MAX_ACTIVE                limit on warming policies (default: per-GPU count)
#   DEXPBT_INTER_LAUNCH_DELAY_S      min seconds between successive launches
#   DEXPBT_STARTUP_TIMEOUT_S         per-attempt first-epoch timeout
#   DEXPBT_STARTUP_MAX_RESTARTS      retries on startup hang (default 1)
#   DEXPBT_NUM_POLICIES              policy count (default 6)
#   DEXPBT_GPUS                      comma-separated GPU ids (default 0,1,2,3,4,5)
set -euo pipefail

RUNTIME_ROOT="${RUNTIME_ROOT:-/root/autodl-tmp/wmdexpbt/runtime_root}"
NUM_POLICIES="${DEXPBT_NUM_POLICIES:-6}"
DEXPBT_GPUS="${DEXPBT_GPUS:-0,1,2,3,4,5}"
TRAIN_ROOT="${1:?usage: $0 TRAIN_ROOT [--num-envs N] [--max-iterations N] [--task TASK] ...}"
shift

NUM_ENVS=8192
MAX_ITER=38147
TASK="DexPBT-Reorientation-KukaAllegro-Direct-v0"
RECOVER_CHECKPOINT=""
RECOVER_POLICY_IDX=3
EXTRA=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --num-envs) NUM_ENVS="$2"; shift 2 ;;
    --max-iterations) MAX_ITER="$2"; shift 2 ;;
    --task) TASK="$2"; shift 2 ;;
    --recover-checkpoint) RECOVER_CHECKPOINT="$2"; shift 2 ;;
    --recover-policy-idx) RECOVER_POLICY_IDX="$2"; shift 2 ;;
    *) EXTRA+=("$1"); shift ;;
  esac
done

# shellcheck source=/dev/null
source "$RUNTIME_ROOT/scripts/env_dexpbt.sh"

cd "$DEXPBT_ROOT"
mkdir -p "$TRAIN_ROOT"

# Optional staggered-launch flags. We only emit the launch_pbt args when
# the env var is set so that older callers still get the historical default.
STAGGER_ARGS=()
if [[ -n "${DEXPBT_MAX_ACTIVE:-}" ]]; then
  STAGGER_ARGS+=(--max-active "$DEXPBT_MAX_ACTIVE")
fi
if [[ -n "${DEXPBT_INTER_LAUNCH_DELAY_S:-}" ]]; then
  STAGGER_ARGS+=(--inter-launch-delay-s "$DEXPBT_INTER_LAUNCH_DELAY_S")
fi
if [[ -n "${DEXPBT_STARTUP_TIMEOUT_S:-}" ]]; then
  STAGGER_ARGS+=(--startup-timeout-s "$DEXPBT_STARTUP_TIMEOUT_S")
fi
if [[ -n "${DEXPBT_STARTUP_MAX_RESTARTS:-}" ]]; then
  STAGGER_ARGS+=(--startup-max-restarts "$DEXPBT_STARTUP_MAX_RESTARTS")
fi

echo "[launch] train_root=$TRAIN_ROOT task=$TASK num_envs=$NUM_ENVS max_iterations=$MAX_ITER num_policies=$NUM_POLICIES gpus=$DEXPBT_GPUS"
if [[ "${#STAGGER_ARGS[@]}" -gt 0 ]]; then
  echo "[launch] stagger=${STAGGER_ARGS[*]}"
fi

exec python -u -m dexpbt_isaaclab.scripts.launch_pbt \
  --task="$TASK" \
  --num-policies="$NUM_POLICIES" \
  --num-envs="$NUM_ENVS" \
  --max-iterations="$MAX_ITER" \
  --train-root="$TRAIN_ROOT" \
  --workspace="dexpbt_8gpu_ws" \
  --gpus="$DEXPBT_GPUS" \
  "${STAGGER_ARGS[@]}" \
  ${RECOVER_CHECKPOINT:+--recover-checkpoint="$RECOVER_CHECKPOINT"} \
  ${RECOVER_CHECKPOINT:+--recover-policy-idx="$RECOVER_POLICY_IDX"} \
  "${EXTRA[@]}"
