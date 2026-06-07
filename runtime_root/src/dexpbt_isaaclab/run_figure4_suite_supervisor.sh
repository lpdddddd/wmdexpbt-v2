#!/usr/bin/env bash
set -euo pipefail

trap '' HUP

RUN_DIR="${1:?usage: $0 RUN_DIR [SUITE_ARGS...]}"
shift
SUITE_ARGS=("$@")

MAX_RESTARTS="${MAX_RESTARTS:-1000}"
RESTART_SLEEP_SECONDS="${RESTART_SLEEP_SECONDS:-30}"

export VULKAN_SDK=/root/autodl-tmp/vulkan_sdk_1.4.304.0/x86_64
export PATH="$VULKAN_SDK/bin:$PATH"
export LD_LIBRARY_PATH="$VULKAN_SDK/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export VK_LAYER_PATH="$VULKAN_SDK/share/vulkan/explicit_layer.d"
export VK_ICD_FILENAMES=/root/autodl-tmp/nvidia_egl_icd.json
export XDG_RUNTIME_DIR=/tmp/runtime-root
mkdir -p "$XDG_RUNTIME_DIR"
chmod 700 "$XDG_RUNTIME_DIR"

export TORCHDYNAMO_DISABLE=1
export PYTHONUNBUFFERED=1

source /root/miniconda3/bin/activate isaaclab311
cd /root/autodl-tmp/dexpbt_isaaclab

mkdir -p "$RUN_DIR"

STATUS_FILE="$RUN_DIR/suite_status.txt"
PID_FILE="$RUN_DIR/suite_pid.txt"
LOG_FILE="$RUN_DIR/suite_launcher.log"

echo "$$" > "$PID_FILE"
{
  echo "start_time=$(date --iso-8601=seconds)"
  echo "cwd=$(pwd)"
  echo "python=$(which python)"
  echo "python_version=$(python --version 2>&1)"
  echo "torchdynamo_disable=${TORCHDYNAMO_DISABLE}"
  echo "max_restarts=${MAX_RESTARTS}"
  echo "restart_sleep_seconds=${RESTART_SLEEP_SECONDS}"
  printf 'suite_args=%q ' "${SUITE_ARGS[@]}"
  printf '\n'
} > "$RUN_DIR/suite_metadata.txt"

attempt=0
while (( attempt < MAX_RESTARTS )); do
  attempt=$((attempt + 1))
  {
    echo "status=starting"
    echo "attempt=${attempt}"
    echo "start_time=$(date --iso-8601=seconds)"
  } > "$STATUS_FILE"

  cmd=(
    python -u -m dexpbt_isaaclab.scripts.launch_figure4_single_arm_suite
    --results-root "$RUN_DIR"
    "${SUITE_ARGS[@]}"
  )

  echo "[SUITE-SUPERVISOR] starting attempt ${attempt}/${MAX_RESTARTS} at $(date --iso-8601=seconds)" | tee -a "$LOG_FILE"
  echo "[SUITE-SUPERVISOR] command: ${cmd[*]}" | tee -a "$LOG_FILE"

  set +e
  "${cmd[@]}" 2>&1 | tee -a "$LOG_FILE"
  exit_code=${PIPESTATUS[0]}
  set -e

  {
    echo "status=stopped"
    echo "attempt=${attempt}"
    echo "end_time=$(date --iso-8601=seconds)"
    echo "exit_code=${exit_code}"
  } > "$STATUS_FILE"

  echo "[SUITE-SUPERVISOR] attempt ${attempt} exited with code ${exit_code}" | tee -a "$LOG_FILE"
  if [[ "$exit_code" -eq 0 ]]; then
    echo "[SUITE-SUPERVISOR] suite finished cleanly" | tee -a "$LOG_FILE"
    exit 0
  fi

  echo "[SUITE-SUPERVISOR] sleeping ${RESTART_SLEEP_SECONDS}s before restart" | tee -a "$LOG_FILE"
  sleep "$RESTART_SLEEP_SECONDS"
done

echo "[SUITE-SUPERVISOR] reached max restarts (${MAX_RESTARTS}), giving up" | tee -a "$LOG_FILE"
exit 1
