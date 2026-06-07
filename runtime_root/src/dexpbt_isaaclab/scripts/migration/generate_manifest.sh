#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${1:-/root/autodl-tmp/dexpbt_isaaclab}"
OUT_DIR="${2:-$PROJECT_ROOT/offline_bundle}"
MANIFEST_FILE="$OUT_DIR/MIGRATION_MANIFEST.md"

mkdir -p "$OUT_DIR"

cd "$PROJECT_ROOT"

CURRENT_COMMIT="$(git rev-parse HEAD 2>/dev/null || echo "N/A")"
CURRENT_BRANCH="$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo "N/A")"
DIRTY_STATUS="$(git status --porcelain 2>/dev/null || true)"
if [[ -n "$DIRTY_STATUS" ]]; then
  TREE_STATE="dirty"
else
  TREE_STATE="clean"
fi

PYTHON_BIN="$(command -v python || true)"
PYTHON_VER="$($PYTHON_BIN --version 2>/dev/null || echo "N/A")"
CONDA_ENV="${CONDA_DEFAULT_ENV:-N/A}"
TORCH_INFO="$($PYTHON_BIN - <<'PY'
try:
    import torch
    print(f"torch={torch.__version__},cuda={torch.version.cuda},cuda_available={torch.cuda.is_available()}")
except Exception as e:
    print(f"torch_info_error={e}")
PY
)"

cat > "$MANIFEST_FILE" <<EOF
# Migration Manifest

- Generated at: $(date --iso-8601=seconds)
- Project root: $PROJECT_ROOT
- Git branch: $CURRENT_BRANCH
- Git commit: $CURRENT_COMMIT
- Working tree: $TREE_STATE
- Conda env: $CONDA_ENV
- Python: $PYTHON_VER
- Python path: $PYTHON_BIN
- Torch info: $TORCH_INFO

## Training Entry Chain

- Supervisor: \`run_train_supervisor.sh\`
- Trainer: \`dexpbt_isaaclab/scripts/train.py\`
- Env registration: \`dexpbt_isaaclab/__init__.py\`
- Env implementation: \`dexpbt_isaaclab/dexpbt_kuka_allegro_env.py\`
- RL config: \`dexpbt_isaaclab/agents/rl_games_ppo_cfg.yaml\`
- PBT observer: \`dexpbt_isaaclab/pbt/pbt_observer.py\`
- World model: \`dexpbt_isaaclab/world_model/\`

## Suggested Runtime Command (template)

\`\`\`bash
CUDA_VISIBLE_DEVICES="0" bash run_train_supervisor.sh "<RUN_DIR>" \
  "DexPBT-Regrasping-KukaAllegro-Direct-v0" \
  --device=cuda:0 --headless --num_envs=2048 --max_iterations=300 \
  --observer_mode=pbt --policy_idx=0 --num_policies=2 \
  --workspace="<RUN_DIR>/pbt_ws" --pbt_interval_steps=1000000 \
  --pbt_start_after=500000 --pbt_initial_delay=600000 \
  --enable_wm --wm_strategy=value_imag_h1 --wm_ensemble_size=3
\`\`\`
EOF

echo "[OK] wrote manifest: $MANIFEST_FILE"
