#!/usr/bin/env bash
set -uo pipefail

RUN_DIR="${1:?usage: $0 RUN_DIR [TASK] [SEED_CHECKPOINT]}"
TASK="${2:-DexPBT-Reorientation-KukaAllegro-Direct-v0}"
SEED_CHECKPOINT="${3:-}"

NUM_ENVS="${NUM_ENVS:-8192}"
MAX_RESTARTS="${MAX_RESTARTS:-20}"
RESTART_SLEEP_SECONDS="${RESTART_SLEEP_SECONDS:-15}"
MAX_ITERATIONS="${MAX_ITERATIONS:-38147}"

export VULKAN_SDK=/root/autodl-tmp/vulkan_sdk_1.4.304.0/x86_64
export PATH="$VULKAN_SDK/bin:$PATH"
export LD_LIBRARY_PATH="$VULKAN_SDK/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export VK_LAYER_PATH="$VULKAN_SDK/share/vulkan/explicit_layer.d"
export VK_ICD_FILENAMES=/root/autodl-tmp/nvidia_egl_icd.json
export XDG_RUNTIME_DIR=/tmp/runtime-root
mkdir -p "$XDG_RUNTIME_DIR"
chmod 700 "$XDG_RUNTIME_DIR"

# Keep Python output streaming and explicitly disable the unstable compile path.
export TORCHDYNAMO_DISABLE=1
export PYTHONUNBUFFERED=1

source /root/miniconda3/bin/activate isaaclab311
cd /root/autodl-tmp/dexpbt_isaaclab

mkdir -p "$RUN_DIR"

LATEST_CKPT_FILE="$RUN_DIR/latest_checkpoint.txt"
STATUS_FILE="$RUN_DIR/status.txt"
SUPERVISOR_PID_FILE="$RUN_DIR/pid.txt"

pick_checkpoint() {
  if [[ -f "$LATEST_CKPT_FILE" ]]; then
    local latest
    latest="$(tr -d '\n' < "$LATEST_CKPT_FILE")"
    if [[ -n "$latest" && -f "$latest" ]]; then
      echo "$latest"
      return 0
    fi
  fi

  if [[ -n "$SEED_CHECKPOINT" && -f "$SEED_CHECKPOINT" ]]; then
    echo "$SEED_CHECKPOINT"
    return 0
  fi

  echo ""
}

record_latest_checkpoint() {
  local artifacts_dir="$1"
  local ckpt=""
  if [[ -d "$artifacts_dir/nn" ]]; then
    ckpt="$(python - "$artifacts_dir/nn" <<'PY'
from pathlib import Path
import sys

nn_dir = Path(sys.argv[1])
candidates = sorted(nn_dir.glob("last_*.pth"), key=lambda p: p.stat().st_mtime, reverse=True)
if not candidates:
    candidates = sorted(nn_dir.glob("*.pth"), key=lambda p: p.stat().st_mtime, reverse=True)
print(candidates[0] if candidates else "")
PY
)"
  fi

  if [[ -n "$ckpt" && -f "$ckpt" ]]; then
    echo "$ckpt" > "$LATEST_CKPT_FILE"
  fi
}

echo "$$" > "$SUPERVISOR_PID_FILE"
{
  echo "start_time=$(date --iso-8601=seconds)"
  echo "task=$TASK"
  echo "cwd=$(pwd)"
  echo "python=$(which python)"
  echo "python_version=$(python --version 2>&1)"
  echo "torchdynamo_disable=${TORCHDYNAMO_DISABLE}"
  echo "vulkan_sdk=${VULKAN_SDK}"
  echo "num_envs=${NUM_ENVS}"
  echo "max_iterations=${MAX_ITERATIONS}"
  echo "max_restarts=${MAX_RESTARTS}"
  echo "seed_checkpoint=${SEED_CHECKPOINT}"
} > "$RUN_DIR/metadata.txt"

attempt=0
while (( attempt < MAX_RESTARTS )); do
  attempt=$((attempt + 1))
  checkpoint="$(pick_checkpoint)"
  attempt_name="attempt_${attempt}_$(date +%Y-%m-%d_%H-%M-%S)"
  artifacts_dir="$RUN_DIR/rl_games/${attempt_name}"
  attempt_log="$RUN_DIR/${attempt_name}.log"

  {
    echo "status=starting"
    echo "attempt=${attempt}"
    echo "attempt_name=${attempt_name}"
    echo "start_time=$(date --iso-8601=seconds)"
    echo "checkpoint=${checkpoint}"
  } > "$STATUS_FILE"

  cmd=(
    python -u -m dexpbt_isaaclab.scripts.train
    --task "$TASK"
    --headless
    --num_envs "$NUM_ENVS"
    --max_iterations "$MAX_ITERATIONS"
    --train_root "$RUN_DIR/rl_games"
    --experiment_name "$attempt_name"
  )
  if [[ -n "$checkpoint" ]]; then
    cmd+=(--checkpoint "$checkpoint")
  fi

  echo "[SUPERVISOR] starting attempt ${attempt}/${MAX_RESTARTS} at $(date --iso-8601=seconds)"
  echo "[SUPERVISOR] command: ${cmd[*]}"
  if [[ -n "$checkpoint" ]]; then
    echo "[SUPERVISOR] resuming from checkpoint: $checkpoint"
  else
    echo "[SUPERVISOR] no checkpoint found, starting from scratch"
  fi

  set +e
  "${cmd[@]}" 2>&1 | tee "$attempt_log"
  exit_code=${PIPESTATUS[0]}
  set -e

  if [[ -d "$artifacts_dir" ]]; then
    ln -sfn "$artifacts_dir" "$RUN_DIR/latest_artifacts"
    record_latest_checkpoint "$artifacts_dir"
  fi

  {
    echo "status=stopped"
    echo "attempt=${attempt}"
    echo "attempt_name=${attempt_name}"
    echo "end_time=$(date --iso-8601=seconds)"
    echo "exit_code=${exit_code}"
    echo "checkpoint=$(pick_checkpoint)"
  } > "$STATUS_FILE"

  echo "[SUPERVISOR] attempt ${attempt} exited with code ${exit_code}"
  if [[ "$exit_code" -eq 0 ]]; then
    echo "[SUPERVISOR] training finished cleanly"
    exit 0
  fi

  echo "[SUPERVISOR] sleeping ${RESTART_SLEEP_SECONDS}s before restart"
  sleep "$RESTART_SLEEP_SECONDS"
done

echo "[SUPERVISOR] reached max restarts (${MAX_RESTARTS}), giving up"
exit 1
