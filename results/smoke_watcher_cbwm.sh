#!/usr/bin/env bash
# Poll smoke run until ~100M frames, then write SMOKE_RUN_DATA_REPORT.md.
set -euo pipefail

RUN_DIR="${1:?usage: $0 RUN_DIR}"
TARGET_EPOCH="${2:-764}"
FRAMES_PER_EPOCH=$((8192 * 16))
TARGET_FRAMES=$((TARGET_EPOCH * FRAMES_PER_EPOCH))
POLL_S="${POLL_S:-300}"
REPORT_PY="/root/autodl-tmp/wmdexpbt/results/generate_smoke_report_cbwm.py"
LOG="${RUN_DIR}/smoke_watcher.log"

exec >>"$LOG" 2>&1
echo "[watcher] started $(date -Is) run_dir=$RUN_DIR target_epoch=$TARGET_EPOCH (~${TARGET_FRAMES} frames)"

while true; do
  max_epoch=0
  done_count=0
  for i in $(seq 0 7); do
    d="${RUN_DIR}/policy_$(printf '%02d' $i)"
    st=""
    [[ -f "$d/status.txt" ]] && st=$(grep -oP '^status=\K.*' "$d/status.txt" 2>/dev/null || true)
    [[ "$st" == "finished" || "$st" == "exited" || "$st" == "done" ]] && done_count=$((done_count + 1))
    live="${d}/train.log"
    an=$(grep -oP '^attempt_name=\K.*' "$d/status.txt" 2>/dev/null || true)
    [[ -n "$an" && -f "${d}/${an}.log" ]] && live="${d}/${an}.log"
    ep=$(grep -oE 'epoch: [0-9]+' "$live" 2>/dev/null | tail -1 | grep -oE '[0-9]+' || echo 0)
    [[ "$ep" =~ ^[0-9]+$ ]] || ep=0
    max_epoch=$(( ep > max_epoch ? ep : max_epoch ))
  done

  echo "[watcher] $(date -Is) max_epoch=$max_epoch done_policies=$done_count/8"

  if (( max_epoch >= TARGET_EPOCH )) || (( done_count >= 8 )); then
    echo "[watcher] target reached — generating report"
    python "$REPORT_PY" "$RUN_DIR"
    echo "[watcher] done $(date -Is)"
    exit 0
  fi

  # Bail if master and all supervisors gone for >10 min (crash)
  if ! pgrep -f "$RUN_DIR" >/dev/null 2>&1; then
    echo "[watcher] no processes for run_dir — generating partial report and exiting"
    python "$REPORT_PY" "$RUN_DIR" || true
    exit 1
  fi

  sleep "$POLL_S"
done
