#!/usr/bin/env bash
# Single-worker HOLD SMOKE for a running arm — EXECUTION SMOKE ONLY.
#
# FORBIDDEN in the formal four-arm comparison: this smoke RELEASES the
# target worker and waits for one extra committed tick, which advances the
# worker past its matched intervention boundary. The formal MSA arms use
# the DISPOSABLE start_stateful_hold_smoke.sh certificate instead
# (HOLD_SMOKE_CERTIFICATE.json), validated by the sidecar at startup.
#
# Default worker is 4: it trains on a SHARED eval GPU (cuda:4), so the
# execution smoke exercises the shared-GPU hold path.
#
# Validates SIGSTOP/SIGCONT against a live Isaac trainer: waits for the
# guard's readiness marker, holds ONE worker, verifies its committed frame
# is frozen for the hold window, releases, verifies the next committed
# tick appears. No tolerance writes, no MSA cycle, no eval.
# Result: <run_dir>/HOLD_SMOKE_RESULT.json (release guaranteed either way).
# HOLD_SMOKE_PASSED.json is written atomically ONLY when the frame stayed
# frozen, the next committed tick appeared after release AND no residual
# holds remain. Any failure writes HOLD_SMOKE_FAILED.json and fail-closed
# stops the whole arm.
set -euo pipefail

RUN_DIR="${1:?usage: run_hold_smoke.sh <run_dir> [worker] [seconds]}"
WORKER="${2:-4}"
HOLD_SECONDS="${3:-60}"
RUNTIME_ROOT="/root/autodl-tmp/wmdexpbt/runtime_root"

echo "WARNING: execution smoke only — the released worker will advance one"
echo "extra committed tick; NEVER use inside the formal four-arm comparison."

source "${RUNTIME_ROOT}/scripts/env_dexpbt.sh"
cd "${RUNTIME_ROOT}/src/dexpbt_isaaclab"

python -u -m dexpbt_isaaclab.msa.sidecar \
  --workspace "$RUN_DIR/dexpbt_8gpu_ws" \
  --arm MSA_STATEFUL \
  --readiness-marker "$RUN_DIR/FIRST_TICK_VALIDATED.json" \
  --hold-smoke-only \
  --hold-smoke-worker "$WORKER" \
  --hold-smoke-seconds "$HOLD_SECONDS"
echo "hold smoke result: $RUN_DIR/HOLD_SMOKE_RESULT.json"
