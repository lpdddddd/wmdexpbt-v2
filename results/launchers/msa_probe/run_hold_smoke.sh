#!/usr/bin/env bash
# Single-worker HOLD SMOKE for a running MSA arm. Prepared but NOT executed.
# Validates SIGSTOP/SIGCONT against a live Isaac trainer BEFORE any probe
# cycle: waits for the guard's readiness marker, holds ONE worker, verifies
# its committed frame is frozen for the hold window, releases, verifies the
# next committed tick appears. No tolerance writes, no MSA cycle, no eval.
# Result: <run_dir>/HOLD_SMOKE_RESULT.json (release guaranteed either way).
set -euo pipefail

RUN_DIR="${1:?usage: run_hold_smoke.sh <run_dir> [worker] [seconds]}"
WORKER="${2:-0}"
HOLD_SECONDS="${3:-60}"
RUNTIME_ROOT="/root/autodl-tmp/wmdexpbt/runtime_root"

source "${RUNTIME_ROOT}/scripts/env_dexpbt.sh"
cd "${RUNTIME_ROOT}/src/dexpbt_isaaclab"

python -u -m dexpbt_isaaclab.msa.sidecar \
  --workspace "$RUN_DIR/dexpbt_8gpu_ws" \
  --arm MSA \
  --readiness-marker "$RUN_DIR/FIRST_TICK_VALIDATED.json" \
  --hold-smoke-only \
  --hold-smoke-worker "$WORKER" \
  --hold-smoke-seconds "$HOLD_SECONDS"
echo "hold smoke result: $RUN_DIR/HOLD_SMOKE_RESULT.json"
