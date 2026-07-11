#!/usr/bin/env bash
# Reload DexPBT workers so they pick up runtime code changes without losing progress.
#
# Kills each policy supervisor + train child, then relaunches run_train_supervisor.sh
# with launch args from train.log (no stale PBT-restart checkpoint, copy-restore
# frame clamps, or *_resume flags). RL resume uses latest_checkpoint.txt; PBT tick
# state is restored from LATEST.commit inside pbt_observer.
#
# Usage:
#   bash reload_dexpbt_train_workers.sh /path/to/run_dir
#   bash reload_dexpbt_train_workers.sh /path/to/run_dir --policy 0,1,2
#   bash reload_dexpbt_train_workers.sh /path/to/run_dir --dry-run
set -euo pipefail

RUN_DIR="${1:?usage: $0 RUN_DIR [--policy 0,1,...] [--dry-run]}"
shift

POLICIES=""
DRY_RUN=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --policy) POLICIES="$2"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

if [[ -z "$POLICIES" ]]; then
  POLICIES="0,1,2,3,4,5"
fi

RUNTIME_ROOT="${RUNTIME_ROOT:-/root/autodl-tmp/wmdexpbt/runtime_root}"
SUPERVISOR="$RUNTIME_ROOT/src/dexpbt_isaaclab/run_train_supervisor.sh"
# shellcheck source=/dev/null
source "$RUNTIME_ROOT/scripts/env_dexpbt.sh"

IFS=',' read -r -a POLICY_LIST <<< "$POLICIES"

echo "[reload] run_dir=$RUN_DIR policies=${POLICIES} dry_run=$DRY_RUN"

kill_tree() {
  local pid="$1"
  [[ -n "$pid" ]] || return 0
  kill -KILL "$pid" 2>/dev/null || true
  pkill -KILL -P "$pid" 2>/dev/null || true
}

read_metadata_train_args() {
  local policy_dir="$1"
  python3 - "$policy_dir" <<'PY'
import re
import shlex
import sys
from pathlib import Path

policy_dir = Path(sys.argv[1])
meta = policy_dir / "metadata.txt"
log = policy_dir / "train.log"
task = "DexPBT-Regrasping-KukaAllegro-Direct-v0"
if meta.is_file():
    for line in meta.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith("task="):
            task = line.split("=", 1)[1]
            break

_PBT_COPY_RESTART_KEYS = {
    "--checkpoint",
    "--kit_args",
    "--restart_source_policy",
    "--pbt_copy_target_frame_before_restore",
}


def filter_train_args(parts: list[str]) -> list[str]:
    """Drop one-shot PBT restart / copy-restore flags from relaunch args."""
    out: list[str] = []
    skip_next = False
    for arg in parts:
        if skip_next:
            skip_next = False
            continue
        key = arg.split("=", 1)[0]
        if key in _PBT_COPY_RESTART_KEYS:
            if "=" not in arg:
                skip_next = True
            continue
        if key.endswith("_resume"):
            if "=" not in arg:
                skip_next = True
            continue
        out.append(arg)
    return out

args: list[str] = []
if log.is_file():
    text = log.read_text(encoding="utf-8", errors="replace")
    cut = text.find("========== code_reload")
    if cut >= 0:
        text = text[:cut]
    cmds = re.findall(r"\[SUPERVISOR\] command: (.+)", text)
    if cmds:
        parts = shlex.split(cmds[-1])
        # python -u -m dexpbt_isaaclab.scripts.train --task ... --train_root ... --experiment_name NAME ...
        try:
            train_idx = parts.index("-m") + 2
            while train_idx < len(parts) and parts[train_idx].startswith("-"):
                key = parts[train_idx].split("=", 1)[0]
                if key in {"--task", "--train_root", "--experiment_name"}:
                    train_idx += 1
                    if "=" not in parts[train_idx - 1]:
                        train_idx += 1
                    continue
                break
            args = filter_train_args(parts[train_idx:])
        except ValueError:
            args = []

if not args and meta.is_file():
    train_line = ""
    for line in meta.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith("train_args="):
            train_line = line
            break
    if train_line:
        for chunk in train_line.split(" train_args="):
            chunk = chunk.removeprefix("train_args=")
            chunk = chunk.replace("\\,", ",").rstrip(" \\").strip()
            if chunk.startswith("--"):
                args.append(chunk)

print(task)
for arg in args:
    print(arg)
PY
}

pin_best_checkpoint() {
  local policy_dir="$1"
  python3 - "$policy_dir" <<'PY'
import re
import sys
from pathlib import Path

import torch

policy_dir = Path(sys.argv[1])
rl_games = policy_dir / "rl_games"
if not rl_games.is_dir():
    sys.exit(0)

best_path: Path | None = None
best_frame = -1
for ckpt in rl_games.rglob("last_*.pth"):
    try:
        payload = torch.load(ckpt, map_location="cpu", weights_only=False)
    except Exception:
        continue
    if not isinstance(payload, dict):
        continue
    frame = int(payload.get("frame", 0) or 0)
    if frame <= best_frame:
        continue
    best_frame = frame
    best_path = ckpt

if best_path is None:
    sys.exit(0)

latest = policy_dir / "latest_checkpoint.txt"
latest.write_text(str(best_path) + "\n", encoding="utf-8")
print(f"[reload] pinned latest_checkpoint frame={best_frame} path={best_path}")
PY
}

for idx in "${POLICY_LIST[@]}"; do
  policy_dir="$RUN_DIR/policy_$(printf '%02d' "$idx")"
  meta="$policy_dir/metadata.txt"
  if [[ ! -f "$meta" ]]; then
    echo "[reload] policy_${idx}: missing metadata.txt (skip)"
    continue
  fi

  mapfile -t meta_lines < <(read_metadata_train_args "$policy_dir")
  task="${meta_lines[0]}"
  train_args=("${meta_lines[@]:1}")

  ckpt_file="$policy_dir/latest_checkpoint.txt"

  sup_pid=""
  [[ -f "$policy_dir/pid.txt" ]] && sup_pid="$(tr -d '\n' < "$policy_dir/pid.txt")"
  train_pid=""
  [[ -f "$policy_dir/train_pid.txt" ]] && train_pid="$(tr -d '\n' < "$policy_dir/train_pid.txt")"

  cuda_dev="$(grep '^cuda_visible_devices=' "$meta" | head -1 | cut -d= -f2-)"
  xdg_data="$(grep '^xdg_data_home=' "$meta" | head -1 | cut -d= -f2-)"
  xdg_cache="$(grep '^xdg_cache_home=' "$meta" | head -1 | cut -d= -f2-)"
  startup_max="$(grep '^startup_max_restarts=' "$meta" | head -1 | cut -d= -f2-)"
  startup_max="${startup_max:-2}"

  echo "[reload] policy_${idx}: kill supervisor=$sup_pid train=$train_pid"
  if [[ "$DRY_RUN" -eq 1 ]]; then
    pin_best_checkpoint "$policy_dir" || true
    ckpt_file="$policy_dir/latest_checkpoint.txt"
    if [[ -f "$ckpt_file" ]]; then
      echo "[reload] policy_${idx}: would resume_ckpt=$(tr -d '\n' < "$ckpt_file")"
    fi
    continue
  fi

  pin_best_checkpoint "$policy_dir" || true
  ckpt=""
  if [[ -f "$ckpt_file" ]]; then
    ckpt="$(tr -d '\n' < "$ckpt_file")"
    [[ -f "$ckpt" ]] || ckpt=""
  fi
  echo "[reload] policy_${idx}: resume_ckpt=${ckpt:-none}"

  kill_tree "$train_pid"
  kill_tree "$sup_pid"
  rm -f "$policy_dir/train_pid.txt" "$policy_dir/pbt_restart.json" 2>/dev/null || true
  sleep 1

  relaunch_args=("${train_args[@]}")
  if [[ -n "$ckpt" ]]; then
    relaunch_args+=("--checkpoint=$ckpt")
  fi

  {
    echo ""
    echo "========== code_reload $(date --iso-8601=seconds) policy_${idx} =========="
  } >> "$policy_dir/train.log"

  export CUDA_VISIBLE_DEVICES="${cuda_dev:-$idx}"
  export XDG_DATA_HOME="${xdg_data:-$policy_dir/.xdg_data}"
  export XDG_CACHE_HOME="${xdg_cache:-$policy_dir/.xdg_cache}"
  export STARTUP_MAX_RESTARTS="$startup_max"
  export DEXPBT_TRAIN_PID_FILE="$policy_dir/train_pid.txt"
  export DEXPBT_STARTUP_HANG_FLAG="$policy_dir/startup_hang.flag"
  export TORCHDYNAMO_DISABLE=1
  export PYTHONUNBUFFERED=1

  nohup bash "$SUPERVISOR" "$policy_dir" "$task" "${relaunch_args[@]}" \
    >> "$policy_dir/train.log" 2>&1 &
  echo "[reload] policy_${idx}: supervisor pid=$!"
done

if [[ "$DRY_RUN" -eq 0 ]]; then
  echo "[reload] done. Isaac startup ~2-5 min; then watch:"
  echo "  tail -f $RUN_DIR/dexpbt_8gpu_ws/metrics/pbt_tick_debug_policy_01.jsonl"
  echo "  tail -f $RUN_DIR/dexpbt_8gpu_ws/metrics/training_curve_policy_01.jsonl"
fi
