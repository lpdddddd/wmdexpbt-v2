#!/bin/bash
# p1 fixed-precision extension launcher (RESUMABLE, DO NOT run without
# explicit approval — this starts GPU evaluation).
#
# 16 units = 4 new tolerances (0.016/0.014/0.012/0.010 m) x 4 seeds
# (10001..10004), p1 tip checkpoint only. The existing 0.018 results in
# ccrs_base_retention are reused read-only and never re-run; this launcher
# writes ONLY into results/analysis/ccrs_p1_precision_extension/.
#
# Resumable: completed units are skipped via the strict is_done raw-JSON
# check, so re-running this script after an interruption is safe.
#
# Plan: 2 GPUs x 8 units, ~150 s/unit -> ~20 min wall time.
#
# Usage:
#   bash run_extension.sh            # both shards (GPU 0 and 1), nohup
#   bash run_extension.sh plan       # dry run only (no GPU)
set -u

REPO=/root/autodl-tmp/wmdexpbt
SRC=$REPO/runtime_root/src/dexpbt_isaaclab
PY=$REPO/runtime_root/envs/isaaclab311/bin/python
OUT=$REPO/results/analysis/ccrs_p1_precision_extension
MANIFEST=$REPO/results/analysis/ccrs_base_retention/checkpoint_manifest.csv
TASK=DexPBT-Regrasping-KukaAllegro-Direct-v0

COMMON=(--manifest "$MANIFEST" --out-dir "$OUT"
        --base-ccrs-dir "$REPO/results/analysis/ccrs_base_retention"
        --task "$TASK" --num-gpus 2)

cd "$SRC" || exit 1

if [[ "${1:-}" == "plan" ]]; then
  exec "$PY" -m dexpbt_isaaclab.ccrs.precision_extension "${COMMON[@]}" --plan
fi

# Preflight must pass before anything launches (also enforced in-process).
"$PY" -m dexpbt_isaaclab.ccrs.precision_extension "${COMMON[@]}" --plan \
  | grep -q FAIL && { echo "preflight FAILED — not launching"; exit 2; }

for SHARD in 0 1; do
  CUDA_VISIBLE_DEVICES=$SHARD nohup "$PY" -m \
    dexpbt_isaaclab.ccrs.precision_extension "${COMMON[@]}" \
    --execute --shard-index "$SHARD" --num-shards 2 \
    > "$OUT/extension_shard$SHARD.log" 2>&1 &
  echo "shard $SHARD -> GPU $SHARD (pid $!), log: extension_shard$SHARD.log"
done
wait
echo "all shards finished; aggregating"
"$PY" -m dexpbt_isaaclab.ccrs.precision_extension "${COMMON[@]}" --aggregate-only
echo "next (offline): $PY -m dexpbt_isaaclab.ccrs.precision_analysis \
  --extension-results $OUT/extension_results.jsonl \
  --base-results $REPO/results/analysis/ccrs_base_retention/ccrs_results.jsonl \
  --out-dir $OUT"
