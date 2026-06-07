#!/usr/bin/env bash
set -euo pipefail

BUNDLE_DIR="${1:?usage: $0 <BUNDLE_DIR> [TARGET_DIR]}"
TARGET_DIR="${2:-/shared/dexpbt_runtime}"
ENV_DIR="$TARGET_DIR/env"
CODE_DIR="$TARGET_DIR/src"
WHEEL_DIR="$BUNDLE_DIR/wheels"

mkdir -p "$TARGET_DIR" "$ENV_DIR" "$CODE_DIR" "$TARGET_DIR/logs"

echo "[1/6] Verify bundle checksums"
(cd "$BUNDLE_DIR" && sha256sum -c SHA256SUMS) | tee "$TARGET_DIR/logs/checksum_verify.log"

echo "[2/6] Restore source code"
tar -xzf "$BUNDLE_DIR/code/dexpbt_isaaclab_code.tar.gz" -C "$CODE_DIR"

echo "[3/6] Prepare python environment (venv)"
python3 -m venv "$ENV_DIR/venv"
source "$ENV_DIR/venv/bin/activate"
python -m pip install --upgrade pip

echo "[4/6] Install offline wheels"
if [[ -d "$WHEEL_DIR" ]]; then
  python -m pip install --no-index --find-links "$WHEEL_DIR" -r "$BUNDLE_DIR/env/pip_freeze.txt" || {
    echo "[WARN] strict offline install failed; inspect wheel completeness."
  }
else
  echo "[WARN] wheel dir missing: $WHEEL_DIR"
fi

echo "[5/6] Link runtime artifacts"
mkdir -p "$TARGET_DIR/runtime"
if [[ -f "$BUNDLE_DIR/artifacts/vulkan_sdk_1.4.304.0.tar.xz" ]]; then
  tar -xf "$BUNDLE_DIR/artifacts/vulkan_sdk_1.4.304.0.tar.xz" -C "$TARGET_DIR/runtime"
fi
if [[ -f "$BUNDLE_DIR/artifacts/nvidia_egl_icd.json" ]]; then
  cp "$BUNDLE_DIR/artifacts/nvidia_egl_icd.json" "$TARGET_DIR/runtime/"
fi

echo "[6/6] Basic runtime checks"
python - <<'PY'
import sys
print("python", sys.version)
try:
    import torch
    print("torch", torch.__version__, "cuda", torch.version.cuda, "available", torch.cuda.is_available())
except Exception as e:
    print("torch import failed:", e)
PY

echo "[DONE] Installed to: $TARGET_DIR"
echo "Activate with: source $ENV_DIR/venv/bin/activate"
