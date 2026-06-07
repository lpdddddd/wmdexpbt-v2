#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${1:-/root/autodl-tmp/dexpbt_isaaclab}"
BUNDLE_DIR="${2:-$PROJECT_ROOT/offline_bundle}"
CONDA_ENV_NAME="${3:-isaaclab311}"

mkdir -p "$BUNDLE_DIR"/{code,env,wheels,artifacts,bin,logs}
cd "$PROJECT_ROOT"

echo "[1/8] Generate migration manifest"
bash "$PROJECT_ROOT/scripts/migration/generate_manifest.sh" "$PROJECT_ROOT" "$BUNDLE_DIR"

echo "[2/8] Snapshot git metadata"
git rev-parse HEAD > "$BUNDLE_DIR/code/GIT_COMMIT.txt" 2>/dev/null || true
git rev-parse --abbrev-ref HEAD > "$BUNDLE_DIR/code/GIT_BRANCH.txt" 2>/dev/null || true

echo "[3/8] Archive project code"
tar --exclude='.git' --exclude='results' --exclude='offline_bundle' \
  -czf "$BUNDLE_DIR/code/dexpbt_isaaclab_code.tar.gz" .

echo "[4/8] Export conda and pip package list"
if command -v conda >/dev/null 2>&1; then
  conda env export -n "$CONDA_ENV_NAME" > "$BUNDLE_DIR/env/conda_env_export.yml"
  conda list -n "$CONDA_ENV_NAME" --explicit > "$BUNDLE_DIR/env/conda_explicit.txt"
else
  echo "[WARN] conda not found, skip conda export"
fi

python -m pip freeze > "$BUNDLE_DIR/env/pip_freeze.txt"
python - <<PY
from pathlib import Path
src = Path("$BUNDLE_DIR/env/pip_freeze.txt")
dst = Path("$BUNDLE_DIR/env/pip_freeze_downloadable.txt")
lines = src.read_text(encoding="utf-8").splitlines()
filtered = []
for ln in lines:
    s = ln.strip()
    if not s or s.startswith("#"):
        continue
    # Skip local build-path requirements from conda/pip metadata snapshots.
    if " @ file://" in s or s.startswith("-e "):
        continue
    filtered.append(s)
dst.write_text("\\n".join(filtered) + ("\\n" if filtered else ""), encoding="utf-8")
print(f"[INFO] pip downloadable requirements: {len(filtered)}")
PY

echo "[5/8] Build wheelhouse from freeze list"
python -m pip download -r "$BUNDLE_DIR/env/pip_freeze_downloadable.txt" -d "$BUNDLE_DIR/wheels" || {
  echo "[WARN] Some pip packages failed to download; check network/package availability."
}

echo "[6/8] Capture runtime binaries"
if [[ -x "$PROJECT_ROOT/../cc-connect/cc-connect" ]]; then
  cp "$PROJECT_ROOT/../cc-connect/cc-connect" "$BUNDLE_DIR/bin/"
fi

if [[ -f "/root/autodl-tmp/vulkan_sdk_1.4.304.0.tar.xz" ]]; then
  cp "/root/autodl-tmp/vulkan_sdk_1.4.304.0.tar.xz" "$BUNDLE_DIR/artifacts/"
fi
if [[ -f "/root/autodl-tmp/nvidia_egl_icd.json" ]]; then
  cp "/root/autodl-tmp/nvidia_egl_icd.json" "$BUNDLE_DIR/artifacts/"
fi

echo "[7/8] Optional artifacts placeholder"
cat > "$BUNDLE_DIR/artifacts/README.txt" <<'EOF'
Put large runtime artifacts here before transfer:
- Training checkpoints (*.pth)
- World model checkpoints
- Required datasets / preprocessed assets
EOF

echo "[8/8] Checksums"
(cd "$BUNDLE_DIR" && find . -type f ! -name "SHA256SUMS" -print0 | xargs -0 sha256sum > SHA256SUMS)

echo "[DONE] Offline bundle prepared at: $BUNDLE_DIR"
