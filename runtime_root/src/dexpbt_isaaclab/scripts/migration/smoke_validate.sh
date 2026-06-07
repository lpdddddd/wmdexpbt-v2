#!/usr/bin/env bash
set -euo pipefail

TARGET_DIR="${1:-/shared/dexpbt_runtime}"
RUN_DIR="${2:-$TARGET_DIR/smoke_run}"

source "$TARGET_DIR/env/venv/bin/activate"
cd "$TARGET_DIR/src"

echo "[1/4] Python/Torch check"
python - <<'PY'
import torch
print("torch", torch.__version__)
print("cuda_available", torch.cuda.is_available())
print("device_count", torch.cuda.device_count())
PY

echo "[2/4] Import chain check"
python - <<'PY'
import dexpbt_isaaclab
from dexpbt_isaaclab.scripts import train
print("import_ok")
PY

echo "[3/4] Prepare smoke run dir"
mkdir -p "$RUN_DIR"

echo "[4/4] Dryrun command (execute manually after env assets ready)"
cat <<EOF
CUDA_VISIBLE_DEVICES=0 bash run_train_supervisor.sh "$RUN_DIR" \
  "DexPBT-Regrasping-KukaAllegro-Direct-v0" \
  --device=cuda:0 --headless --num_envs=256 --max_iterations=20 \
  --observer_mode=pbt --policy_idx=0 --num_policies=2 \
  --workspace="$RUN_DIR/pbt_ws" --pbt_interval_steps=1000000 \
  --pbt_start_after=500000 --pbt_initial_delay=600000 \
  --enable_wm --wm_strategy=value_imag_h1 --wm_ensemble_size=3
EOF
