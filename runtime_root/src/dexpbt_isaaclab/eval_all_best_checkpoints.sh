#!/bin/bash
# Evaluate all 8 best checkpoints

TASK="DexPBT-Regrasping-KukaAllegro-Direct-v0"
NUM_ENVS=512
NUM_EPISODES=512
BEST_DIR="/root/autodl-tmp/dexpbt_isaaclab/results/best_checkpoints"
EVAL_DIR="/root/autodl-tmp/dexpbt_isaaclab/results/best_checkpoints_eval"

mkdir -p $EVAL_DIR

for i in 0 1 2 3 4 5 6 7; do
    CKPT="$BEST_DIR/policy_0$i"_"best.pth"
    OUTPUT_JSON="$EVAL_DIR/policy_0$i"_"eval_results.json"

    echo "=========================================="
    echo "Evaluating policy_$i ..."
    echo "Checkpoint: $CKPT"
    echo "=========================================="

    cd /root/autodl-tmp/dexpbt_isaaclab

    python -m dexpbt_isaaclab.scripts.evaluate_checkpoint \
        --task $TASK \
        --checkpoint $CKPT \
        --num_envs $NUM_ENVS \
        --num_episodes $NUM_EPISODES \
        --headless \
        --output_json $OUTPUT_JSON \
        2>&1 | tee "$EVAL_DIR/policy_0$i"_"eval.log"

    echo ""
done

echo ""
echo "=========================================="
echo "All evaluations complete!"
echo "Results saved in: $EVAL_DIR"
echo "=========================================="