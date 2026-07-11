#!/usr/bin/env bash
set -euo pipefail

trap '' HUP

RUN_DIR="${1:?usage: $0 RUN_DIR TASK [TRAIN_ARGS...]}"
TASK="${2:?usage: $0 RUN_DIR TASK [TRAIN_ARGS...]}"
shift 2
TRAIN_ARGS=("$@")

MAX_RESTARTS="${MAX_RESTARTS:-1000}"
RESTART_SLEEP_SECONDS="${RESTART_SLEEP_SECONDS:-30}"
# Cap on startup-hang attempts (set by launch_pbt.py via env). Default falls
# back to the generic MAX_RESTARTS so direct invocations of the supervisor
# preserve historical behaviour.
STARTUP_MAX_RESTARTS="${STARTUP_MAX_RESTARTS:-${MAX_RESTARTS}}"
STARTUP_HANG_FLAG="${DEXPBT_STARTUP_HANG_FLAG:-$RUN_DIR/startup_hang.flag}"
TRAIN_PID_FILE="${DEXPBT_TRAIN_PID_FILE:-$RUN_DIR/train_pid.txt}"

# Runtime env
RUNTIME_ROOT="${RUNTIME_ROOT:-/root/autodl-tmp/wmdexpbt/runtime_root}"
# shellcheck source=/dev/null
source "$RUNTIME_ROOT/scripts/env_dexpbt.sh"

cd "$DEXPBT_ROOT"

mkdir -p "$RUN_DIR"

LATEST_CKPT_FILE="$RUN_DIR/latest_checkpoint.txt"
STATUS_FILE="$RUN_DIR/status.txt"
SUPERVISOR_PID_FILE="$RUN_DIR/pid.txt"

clean_per_policy_locks() {
  # Best-effort wipe of stale Omniverse / kvdb lock files inside the
  # per-policy XDG dirs. Safe no-op when the env vars are unset.
  local data_dir="${XDG_DATA_HOME:-}"
  local cache_dir="${XDG_CACHE_HOME:-}"
  for d in "$data_dir" "$cache_dir"; do
    [[ -n "$d" && -d "$d" ]] || continue
    find "$d" \
      \( -name '*.lock' -o -name 'registry.lock' -o -name '_lock' -o -name 'kvdb*.lock' \) \
      -type f -delete 2>/dev/null || true
  done
  rm -f "$STARTUP_HANG_FLAG" "$TRAIN_PID_FILE" 2>/dev/null || true
}

pick_checkpoint() {
  local latest=""
  if [[ -f "$LATEST_CKPT_FILE" ]]; then
    latest="$(tr -d '\n' < "$LATEST_CKPT_FILE")"
    if [[ -n "$latest" && ! -f "$latest" ]]; then
      latest=""
    fi
  fi

  local discovered=""
  discovered="$(python - "$RUN_DIR" <<'PY'
from pathlib import Path
import sys

run_dir = Path(sys.argv[1])
candidates = sorted(run_dir.rglob("last_*.pth"), key=lambda p: p.stat().st_mtime, reverse=True)
if not candidates:
    candidates = sorted(run_dir.rglob("*.pth"), key=lambda p: p.stat().st_mtime, reverse=True)
print(candidates[0] if candidates else "")
PY
)"
  if [[ -n "$discovered" && ! -f "$discovered" ]]; then
    discovered=""
  fi

  if [[ -n "$latest" && -n "$discovered" ]]; then
    local chosen
    chosen="$(python - "$latest" "$discovered" <<'PY'
from pathlib import Path
import re
import sys

import torch


def checkpoint_frame(path: Path) -> int:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if isinstance(payload, dict):
            frame = int(payload.get("frame", 0) or 0)
            if frame > 0:
                return frame
    except Exception:
        pass
    match = re.search(r"_ep_(\d+)_", path.name)
    return int(match.group(1)) if match else 0


latest = Path(sys.argv[1])
discovered = Path(sys.argv[2])
candidates = [latest, discovered]
best = max(candidates, key=checkpoint_frame)
print(best)
PY
)"
    echo "$chosen" > "$LATEST_CKPT_FILE"
    echo "$chosen"
    return 0
  fi

  if [[ -n "$latest" ]]; then
    echo "$latest" > "$LATEST_CKPT_FILE"
    echo "$latest"
    return 0
  fi

  if [[ -n "$discovered" ]]; then
    echo "$discovered" > "$LATEST_CKPT_FILE"
    echo "$discovered"
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
  echo "runtime_root=$RUNTIME_ROOT"
  echo "torchdynamo_disable=${TORCHDYNAMO_DISABLE}"
  echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-}"
  echo "xdg_data_home=${XDG_DATA_HOME:-}"
  echo "xdg_cache_home=${XDG_CACHE_HOME:-}"
  echo "max_restarts=${MAX_RESTARTS}"
  echo "startup_max_restarts=${STARTUP_MAX_RESTARTS}"
  echo "restart_sleep_seconds=${RESTART_SLEEP_SECONDS}"
  printf 'train_args=%q ' "${TRAIN_ARGS[@]}"
  printf '\n'
} > "$RUN_DIR/metadata.txt"

attempt=0
startup_hang_count=0
pbt_override_checkpoint=""
while (( attempt < MAX_RESTARTS )); do
  attempt=$((attempt + 1))
  if [[ -n "$pbt_override_checkpoint" ]]; then
    checkpoint="$pbt_override_checkpoint"
    pbt_override_checkpoint=""
  else
    checkpoint="$(pick_checkpoint)"
  fi
  attempt_name="attempt_${attempt}_$(date +%Y-%m-%d_%H-%M-%S)"
  artifacts_dir="$RUN_DIR/rl_games/${attempt_name}"
  attempt_log="$RUN_DIR/${attempt_name}.log"

  rm -f "$RUN_DIR/pbt_restart.json"
  clean_per_policy_locks

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
    --train_root "$RUN_DIR/rl_games"
    --experiment_name "$attempt_name"
    "${TRAIN_ARGS[@]}"
  )
  if [[ -n "${DEXPBT_KIT_ARGS:-}" ]]; then
    cmd+=(--kit_args "$DEXPBT_KIT_ARGS")
  fi
  if [[ -n "$checkpoint" ]] && [[ ! " ${TRAIN_ARGS[*]} " =~ " --checkpoint" ]]; then
    cmd+=(--checkpoint "$checkpoint")
  fi

  echo "[SUPERVISOR] starting attempt ${attempt}/${MAX_RESTARTS} at $(date --iso-8601=seconds)"
  echo "[SUPERVISOR] command: ${cmd[*]}"
  if [[ -n "$checkpoint" ]]; then
    echo "[SUPERVISOR] resuming from checkpoint: $checkpoint"
  else
    echo "[SUPERVISOR] no checkpoint found, starting from scratch"
  fi

  # Per-attempt log + append to train.log (launch_pbt initially tees train.log;
  # without this, tail -f policy_XX/train.log looks frozen after restarts).
  {
    echo ""
    echo "========== ${attempt_name} started $(date --iso-8601=seconds) =========="
  } >> "$RUN_DIR/train.log"

  # Spawn train.py as a child so we can publish train_pid.txt for the
  # external watchdog. Using process substitution for tee preserves
  # the behaviour of also recording per-attempt logs.
  set +e
  "${cmd[@]}" > >(tee -a "$RUN_DIR/train.log" "$attempt_log") 2>&1 &
  train_pid=$!
  echo "$train_pid" > "$TRAIN_PID_FILE"
  wait "$train_pid"
  exit_code=$?
  set -e
  rm -f "$TRAIN_PID_FILE" 2>/dev/null || true

  if [[ -d "$artifacts_dir" ]]; then
    ln -sfn "$artifacts_dir" "$RUN_DIR/latest_artifacts"
    record_latest_checkpoint "$artifacts_dir"
  fi

  # Detect external startup-hang kill: launch_pbt.py drops a flag file
  # before SIGTERM-ing train.py.
  if [[ -f "$STARTUP_HANG_FLAG" ]]; then
    startup_hang_count=$((startup_hang_count + 1))
    echo "[SUPERVISOR] attempt ${attempt} killed by startup watchdog (count=${startup_hang_count}/${STARTUP_MAX_RESTARTS})"
    rm -f "$STARTUP_HANG_FLAG" 2>/dev/null || true
    if (( startup_hang_count >= STARTUP_MAX_RESTARTS )); then
      {
        echo "status=stopped"
        echo "attempt=${attempt}"
        echo "attempt_name=${attempt_name}"
        echo "end_time=$(date --iso-8601=seconds)"
        echo "exit_code=0"
        echo "reason=startup_hang"
        echo "startup_hang_count=${startup_hang_count}"
      } > "$STATUS_FILE"
      echo "[SUPERVISOR] giving up after ${startup_hang_count} startup hangs; marking stopped (reason=startup_hang)"
      exit 0
    fi
    {
      echo "status=restarting_startup_hang"
      echo "attempt=${attempt}"
      echo "startup_hang_count=${startup_hang_count}"
      echo "start_time=$(date --iso-8601=seconds)"
    } > "$STATUS_FILE"
    echo "[SUPERVISOR] sleeping ${RESTART_SLEEP_SECONDS}s before startup-hang retry"
    sleep "$RESTART_SLEEP_SECONDS"
    continue
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

  if [[ "$exit_code" -eq 42 ]] || [[ -f "$RUN_DIR/pbt_restart.json" ]]; then
    manifest="$RUN_DIR/pbt_restart.json"
    if [[ -f "$manifest" ]]; then
      echo "[SUPERVISOR] PBT restart requested, reading manifest: $manifest"
      pbt_override_checkpoint=$(python3 -c "
import json
m = json.load(open('$manifest'))
print(m.get('checkpoint', ''))
")
      new_args=$(python3 -c "
import json
m = json.load(open('$manifest'))
args = m['train_args']
print(' '.join(args))
")
      echo "[SUPERVISOR] PBT restart checkpoint: $pbt_override_checkpoint"
      echo "[SUPERVISOR] PBT restart args: $new_args"
      rm -f "$manifest"
      TRAIN_ARGS=($new_args)
      {
        echo "status=restarting_pbt"
        echo "attempt=${attempt}"
        echo "pbt_manifest=${manifest}"
        echo "start_time=$(date --iso-8601=seconds)"
      } > "$STATUS_FILE"
      echo "[SUPERVISOR] sleeping ${RESTART_SLEEP_SECONDS}s before PBT restart"
      sleep "$RESTART_SLEEP_SECONDS"
      continue
    else
      echo "[SUPERVISOR] PBT restart code 42 but no manifest found, exiting"
      exit 0
    fi
  fi

  if [[ "$exit_code" -eq 0 ]]; then
    if grep -qE 'Traceback \(most recent call last\)|RuntimeError:' "$attempt_log" 2>/dev/null; then
      echo "[SUPERVISOR] exit code 0 but log contains Python error — treating as crash"
      exit_code=1
    elif grep -qF '[PBT] population_frame_target_reached' "$attempt_log" 2>/dev/null; then
      {
        echo "status=completed_population_frame"
        echo "attempt=${attempt}"
        echo "end_time=$(date --iso-8601=seconds)"
        echo "exit_code=0"
      } > "$STATUS_FILE"
      echo "[SUPERVISOR] worker completed population-frame target"
    elif grep -qE 'epoch: [0-9]+/[0-9]+' "$attempt_log" 2>/dev/null; then
      epoch_line="$(grep -oE 'epoch: [0-9]+/[0-9]+' "$attempt_log" | tail -1)"
      last_epoch="$(echo "$epoch_line" | grep -oE '[0-9]+' | head -1)"
      max_epoch="$(echo "$epoch_line" | grep -oE '[0-9]+' | tail -1)"
      if [[ -n "${last_epoch}" && -n "${max_epoch}" ]] && [[ "${last_epoch}" -lt "${max_epoch}" ]]; then
        echo "[SUPERVISOR] exit code 0 but stopped at epoch ${last_epoch}/${max_epoch} — treating as crash"
        exit_code=1
      fi
    fi
  fi

  if [[ "$exit_code" -eq 0 ]]; then
    echo "[SUPERVISOR] training finished cleanly"
    exit 0
  fi

  echo "[SUPERVISOR] sleeping ${RESTART_SLEEP_SECONDS}s before restart"
  sleep "$RESTART_SLEEP_SECONDS"
done

echo "[SUPERVISOR] reached max restarts (${MAX_RESTARTS}), giving up"
exit 1
