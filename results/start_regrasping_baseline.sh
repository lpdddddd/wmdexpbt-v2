#!/usr/bin/env bash
# 6 卡 DexPBT Regrasping baseline 训练（nohup）
# 结果目录: /root/autodl-tmp/wmdexpbt/results/regrasping_baseline_<timestamp>/
#
# 监控:
#   tail -f $RUN_DIR/launch_master.log
#   tail -f $RUN_DIR/policy_00/train.log
#   watch -n 1 'cat $RUN_DIR/policy_*/status.txt 2>/dev/null | sort'
#   bash $RUN_DIR/monitor.sh   # 若已生成
#
set -euo pipefail

RUNTIME_ROOT="/root/autodl-tmp/wmdexpbt/runtime_root"
RESULTS_ROOT="/root/autodl-tmp/wmdexpbt/results"
DATE=$(date +%Y%m%d_%H%M%S)
RUN_DIR="${RESULTS_ROOT}/regrasping_baseline_${DATE}"
LOG="${RUN_DIR}/launch_master.log"
PIDFILE="${RUN_DIR}/nohup.pid"

mkdir -p "$RUN_DIR"
echo "$RUN_DIR" > "${RUN_DIR}/RUN_DIR.txt"

# shellcheck source=/dev/null
source "${RUNTIME_ROOT}/scripts/env_dexpbt.sh"

# Omniverse 扩展缓存（离线 6 卡必需）
mkdir -p "$(dirname /root/.local/share/ov/data)" /root/.cache/ov 2>/dev/null || true
ln -sfn "${RUNTIME_ROOT}/cache/ov/data" /root/.local/share/ov/data 2>/dev/null || true
ln -sfn "${RUNTIME_ROOT}/cache/ov_client" /root/.cache/ov 2>/dev/null || true

export TORCHDYNAMO_DISABLE=1
export PYTHONUNBUFFERED=1
export OMNI_KIT_TELEMETRY=0
export CARB_CRASH_REPORTER=0

cd "${RUNTIME_ROOT}/src/dexpbt_isaaclab"

echo "Starting 6-GPU DexPBT Regrasping baseline"
echo "  RUN_DIR=$RUN_DIR"
echo "  LOG=$LOG"
echo "  Task=DexPBT-Regrasping-KukaAllegro-Direct-v0"
echo "  num_envs=8192 max_iterations=38147"
echo "  pbt_interval_steps=20000000 start_after=200000000 initial_delay=50000000"

nohup bash "${RUNTIME_ROOT}/scripts/launch_8gpu_dexpbt.sh" \
  "$RUN_DIR" \
  --task DexPBT-Regrasping-KukaAllegro-Direct-v0 \
  --num-envs 8192 \
  --max-iterations 38147 \
  --pbt-interval-steps 20000000 \
  --pbt-start-after 200000000 \
  --pbt-initial-delay 50000000 \
  >"$LOG" 2>&1 &

echo $! > "$PIDFILE"
echo "nohup pid=$! (also saved to $PIDFILE)"
echo "Monitor: tail -f $LOG"
echo "Per-policy logs: $RUN_DIR/policy_XX/train.log"
