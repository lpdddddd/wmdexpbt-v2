#!/usr/bin/env bash
# Fixed WM strategy + P1 curriculum/stability (no Bayesian posterior).
# Usage: bash start_regrasping_fixed_wm_p1.sh

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/_p1_extra_flags.sh"

RUNTIME_ROOT="/root/autodl-tmp/wmdexpbt/runtime_root"
RESULTS_ROOT="/root/autodl-tmp/wmdexpbt/results"
RUNS_ROOT="${RESULTS_ROOT}/runs"
DATE=$(date +%Y%m%d_%H%M%S)
RUN_DIR="${RUNS_ROOT}/regrasping_fixed_wm_p1_${DATE}"
LOG="${RUN_DIR}/launch_master.log"
PIDFILE="${RUN_DIR}/nohup.pid"

mkdir -p "$RUN_DIR"
echo "$RUN_DIR" > "${RESULTS_ROOT}/pointers/latest_run.txt"

# shellcheck source=/dev/null
source "${RUNTIME_ROOT}/scripts/env_dexpbt.sh"
export TORCHDYNAMO_DISABLE=1 PYTHONUNBUFFERED=1
cd "${RUNTIME_ROOT}/src/dexpbt_isaaclab"

nohup bash "${RUNTIME_ROOT}/scripts/launch_8gpu_dexpbt.sh" "$RUN_DIR" \
  --task DexPBT-Regrasping-KukaAllegro-Direct-v0 --num-envs 8192 \
  --max-iterations 38147 --pbt-interval-steps 20000000 \
  --pbt-start-after 200000000 --pbt-initial-delay 50000000 --extra \
  --enable_wm --wm_strategy value_imag_h3 --wm_ensemble_size 3 --wm_lr 1e-3 \
  --wm_update_every_n_epochs 2 --wm_batch_size 512 \
  --pbt_ranking_metric success_aware_r_meta --pbt_nsucc_zero_cap 0.05 \
  --pbt_min_frames_as_source_after_restart 50000000 --pbt_min_frames_between_restarts 50000000 \
  --pbt_disable_self_restart --pbt_source_selection_mode tiered \
  --pbt_curriculum_freeze_frames 100000000 \
  --wm_strategy_mode fixed --fixed_wm_strategy value_imag_h3 \
  "${P1_EXTRA_FLAGS[@]}" >"$LOG" 2>&1 &

echo $! > "$PIDFILE"
echo "Started fixed-WM P1: $RUN_DIR"
