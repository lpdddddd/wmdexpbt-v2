#!/bin/bash
# p1 fixed-precision extension launcher.
#
# 16 units = 4 new tolerances (0.016/0.014/0.012/0.010 m) x 4 seeds
# (10001..10004), p1 tip checkpoint only. The existing 0.018 results in
# ccrs_base_retention are reused read-only and never re-run; this launcher
# writes ONLY into results/analysis/ccrs_p1_precision_extension/.
#
# SAFETY MODEL (fail-closed):
# - `bash run_extension.sh` and `bash run_extension.sh plan` are dry runs:
#   they can NEVER start a GPU evaluation.
# - only `bash run_extension.sh execute` may launch, and only after the
#   preflight subprocess exits 0 AND its saved output contains no FAIL AND
#   confirms ALL CHECKS PASSED. Any import error, checkpoint-resolution
#   error or abnormal exit aborts before any GPU work.
# - shards only run units and write raw JSON (no aggregation inside
#   shards); this main process aggregates exactly once, and only after
#   BOTH shards exited 0. On any shard failure raw results are kept for
#   resume and no final aggregation/analysis is produced.
# - resumable: completed units are skipped via the strict is_done check;
#   re-running `execute` after an interruption is safe.
#
# Plan: 2 GPUs x 8 units, ~150 s/unit -> ~20 min wall time.
set -euo pipefail

# PRECISION_EXT_* env overrides exist ONLY so the CPU test suite can point
# the launcher at temp dirs / broken manifests and verify fail-closed
# behaviour; production use relies on the defaults below.
REPO=${PRECISION_EXT_REPO:-/root/autodl-tmp/wmdexpbt}
SRC=$REPO/runtime_root/src/dexpbt_isaaclab
PY=${PRECISION_EXT_PY:-$REPO/runtime_root/envs/isaaclab311/bin/python}
OUT=${PRECISION_EXT_OUT:-$REPO/results/analysis/ccrs_p1_precision_extension}
MANIFEST=${PRECISION_EXT_MANIFEST:-$REPO/results/analysis/ccrs_base_retention/checkpoint_manifest.csv}
TASK=DexPBT-Regrasping-KukaAllegro-Direct-v0

COMMON=(--manifest "$MANIFEST" --out-dir "$OUT"
        --base-ccrs-dir "$REPO/results/analysis/ccrs_base_retention"
        --task "$TASK" --num-gpus 2)

MODE="${1:-plan}"
case "$MODE" in
  plan|execute) ;;
  *)
    echo "usage: bash run_extension.sh [plan|execute]" >&2
    echo "  plan     dry run only (default): manifest + preflight, no GPU" >&2
    echo "  execute  run the 16 evaluation units on GPUs 0 and 1" >&2
    exit 64
    ;;
esac

cd "$SRC"

# ---- preflight (fail-closed) ----------------------------------------------
# Run the plan, capture BOTH the return code and the full output. Abort on:
# non-zero exit (import error, checkpoint resolution error, crash), any
# FAIL line, or a missing all-checks-pass confirmation.
PLAN_LOG=$OUT/preflight_plan.log
mkdir -p "$OUT"
set +e
"$PY" -m dexpbt_isaaclab.ccrs.precision_extension "${COMMON[@]}" --plan \
  > "$PLAN_LOG" 2>&1
PLAN_RC=$?
set -e
cat "$PLAN_LOG"
if [[ $PLAN_RC -ne 0 ]]; then
  echo "preflight ABORT: plan subprocess exited rc=$PLAN_RC (see $PLAN_LOG)" >&2
  exit 1
fi
if grep -q "FAIL" "$PLAN_LOG"; then
  echo "preflight ABORT: FAIL found in plan output (see $PLAN_LOG)" >&2
  exit 1
fi
if ! grep -q "\[ok\] unit_count_is_16" "$PLAN_LOG"; then
  echo "preflight ABORT: expected check confirmations missing (see $PLAN_LOG)" >&2
  exit 1
fi
echo "preflight: ALL CHECKS PASSED"

if [[ "$MODE" != "execute" ]]; then
  echo "plan only — no GPU was touched (use 'execute' to launch)"
  exit 0
fi

# ---- execute: shards run units only (no aggregation inside shards) --------
PIDS=()
for SHARD in 0 1; do
  CUDA_VISIBLE_DEVICES=$SHARD nohup "$PY" -m \
    dexpbt_isaaclab.ccrs.precision_extension "${COMMON[@]}" \
    --execute --shard-index "$SHARD" --num-shards 2 \
    > "$OUT/extension_shard$SHARD.log" 2>&1 &
  PIDS+=($!)
  echo "shard $SHARD -> GPU $SHARD (pid ${PIDS[$SHARD]}), log: extension_shard$SHARD.log"
done

FAILED=0
for SHARD in 0 1; do
  if ! wait "${PIDS[$SHARD]}"; then
    echo "shard $SHARD FAILED (see extension_shard$SHARD.log)" >&2
    FAILED=1
  fi
done
if [[ $FAILED -ne 0 ]]; then
  echo "one or more shards failed — raw results kept for resume;" >&2
  echo "NOT aggregating and NOT analysing. Re-run 'execute' to resume." >&2
  exit 1
fi

# ---- single aggregation by the main process only ---------------------------
echo "all shards finished; aggregating once"
"$PY" -m dexpbt_isaaclab.ccrs.precision_extension "${COMMON[@]}" --aggregate-only
echo "next (offline): $PY -m dexpbt_isaaclab.ccrs.precision_analysis \
  --extension-results $OUT/extension_results.jsonl \
  --base-results $REPO/results/analysis/ccrs_base_retention/ccrs_results.jsonl \
  --out-dir $OUT"
