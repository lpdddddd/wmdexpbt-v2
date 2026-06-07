#!/usr/bin/env bash
# 快速查看 6 卡训练进度（在任意目录执行，或传入 RUN_DIR）
# 用法: bash monitor_training.sh /root/autodl-tmp/wmdexpbt/results/regrasping_baseline_YYYYMMDD_HHMMSS

RUN_DIR="${1:-}"
if [[ -z "$RUN_DIR" ]]; then
  RUN_DIR=$(cat "${RESULTS_ROOT:-/root/autodl-tmp/wmdexpbt/results}/latest_run.txt" 2>/dev/null || true)
fi
if [[ ! -d "$RUN_DIR" ]]; then
  echo "Usage: $0 <RUN_DIR>"
  echo "Or: echo /path/to/run > $RESULTS_ROOT/latest_run.txt"
  ls -d /root/autodl-tmp/wmdexpbt/results/regrasping_* 2>/dev/null | tail -5
  exit 1
fi

echo "=== RUN_DIR: $RUN_DIR ==="
echo "=== Master log (last 15 lines) ==="
tail -15 "$RUN_DIR/launch_master.log" 2>/dev/null || true
echo ""
echo "=== Policy status ==="
printf "%-10s %-8s %-10s %-28s %s\n" "policy" "attempt" "status" "live_log" "last_epoch"
for i in $(seq 0 7); do
  d="$RUN_DIR/policy_$(printf '%02d' $i)"
  if [[ -f "$d/status.txt" ]]; then
    att=$(grep -oP '^attempt=\K.*' "$d/status.txt" 2>/dev/null || echo "?")
    st=$(grep -oP '^status=\K.*' "$d/status.txt" 2>/dev/null || echo "?")
    an=$(grep -oP '^attempt_name=\K.*' "$d/status.txt" 2>/dev/null || true)
    live="${d}/${an}.log"
    [[ -f "$live" ]] || live="${d}/train.log"
    ep=$(grep -oE 'epoch: [0-9]+' "$live" 2>/dev/null | tail -1 | grep -oE '[0-9]+' || echo "?")
    printf "policy_%02d  %-8s %-10s %-28s %s\n" "$i" "$att" "$st" "$(basename "$live")" "$ep"
  else
    printf "policy_%02d  %-8s %-10s %-28s %s\n" "$i" "?" "?" "?" "?"
  fi
done
echo ""
echo "=== Training curve lines (jsonl) ==="
wc -l "$RUN_DIR"/dexpbt_8gpu_ws/metrics/training_curve_policy_*.jsonl 2>/dev/null || echo "(no metrics yet)"
echo ""
echo "=== GPU (nvidia-smi) ==="
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv 2>/dev/null | head -10 || echo "nvidia-smi unavailable"
