#!/bin/bash
# CCRS stage-1 pre-check: one worker's 6 units (3 tolerances x 2 seeds).
# Usage: run_stage1.sh <gpu> <worker_id> <iteration> <frame> <checkpoint_path>
set -u
GPU=$1; W=$2; IT=$3; FR=$4; CKPT=$5
cd /root/autodl-tmp/wmdexpbt/runtime_root/src/dexpbt_isaaclab
OUT=/root/autodl-tmp/wmdexpbt/results/analysis/ccrs_base_retention
PY=/root/autodl-tmp/wmdexpbt/runtime_root/envs/isaaclab311/bin/python
for TOL in 0.022 0.020 0.019; do
  for SEED in 10001 10002; do
    echo "=== stage1 w$W tol=$TOL seed=$SEED ($(date +%H:%M:%S)) ==="
    CUDA_VISIBLE_DEVICES=$GPU $PY -m dexpbt_isaaclab.scripts.ccrs_eval \
      --task DexPBT-Regrasping-KukaAllegro-Direct-v0 \
      --run-id regrasping_cser_dexpbt_conservative_retention_strict_5b_20260716_130631 \
      --worker-id $W --checkpoint-iteration $IT --checkpoint-frame $FR \
      --checkpoint-path "$CKPT" \
      --fixed-tolerance $TOL --reset-seed $SEED \
      --num-envs 512 --num-episodes 512 \
      --raw-dir $OUT/raw --headless 2>&1 | grep -E 'ccrs_eval\]|Traceback' | head -5
  done
done
echo "=== stage1 worker $W DONE ($(date +%H:%M:%S)) ==="
