#!/usr/bin/env bash
# Pack code + isaaclab311 conda env + IsaacLab + Vulkan for offline transfer to CPU/shared disk.
#
# Usage (on SOURCE machine, with internet/disk space):
#   bash scripts/migration/bundle_full_to_shared.sh /root/autodl-tmp/dexpbt_full_migration
#
# Then copy the whole output dir (or split parts) to CPU shared disk, e.g.:
#   rsync -avP /root/autodl-tmp/dexpbt_full_migration/ user@cpu:/volume/chchen/Pd/wmdexpbt/full_migration/
set -euo pipefail

OUT_ROOT="${1:?usage: $0 <OUTPUT_DIR>}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-isaaclab311}"
DEXPBT_ROOT="${DEXPBT_ROOT:-/root/autodl-tmp/dexpbt_isaaclab}"
ISAACLAB_ROOT="${ISAACLAB_ROOT:-/root/autodl-tmp/IsaacLab}"
VULKAN_ROOT="${VULKAN_ROOT:-/root/autodl-tmp/vulkan_sdk_1.4.304.0}"
VK_ICD="${VK_ICD:-/root/autodl-tmp/nvidia_egl_icd.json}"
SPLIT_GB="${SPLIT_GB:-0}"  # e.g. 4 => split tar into 4G chunks for flaky transfers

mkdir -p "$OUT_ROOT"/{conda,code,runtime,scripts,logs}

log() { echo "[$(date +%H:%M:%S)] $*"; }

# conda-pack uses $TMP and may fill small overlay /tmp — force large disk
export TMPDIR="${TMPDIR:-/root/autodl-tmp/.pack_tmp}"
export TEMP="$TMPDIR"
export TMP="$TMPDIR"
mkdir -p "$TMPDIR" "$OUT_ROOT"

log "=== [1/6] tar conda env: $CONDA_ENV_NAME (TMPDIR=$TMPDIR) ==="
CONDA_BASE="$(conda info --base)"
ENV_SRC="$CONDA_BASE/envs/$CONDA_ENV_NAME"
[[ -d "$ENV_SRC" ]] || { echo "missing env: $ENV_SRC" >&2; exit 1; }
# conda-pack loads files into RAM and often OOMs under small cgroup limits (~2G on AutoDL).
# Stream tar on the large disk instead; install script relocates prefix or runs conda-unpack if present.
echo "$ENV_SRC" > "$OUT_ROOT/conda/.source_prefix"
tar -czf "$OUT_ROOT/conda/${CONDA_ENV_NAME}.tar.gz" -C "$CONDA_BASE/envs" "$CONDA_ENV_NAME"

log "=== [2/6] dexpbt_isaaclab code (exclude results/offline_bundle) ==="
tar -czf "$OUT_ROOT/code/dexpbt_isaaclab.tar.gz" \
  --exclude='results' \
  --exclude='offline_bundle' \
  --exclude='offline_bundle.tar.gz' \
  --exclude='dexpbt_full_migration' \
  --exclude='.git' \
  -C "$(dirname "$DEXPBT_ROOT")" "$(basename "$DEXPBT_ROOT")"

log "=== [3/6] IsaacLab ==="
tar -czf "$OUT_ROOT/code/IsaacLab.tar.gz" \
  --exclude='.git' \
  --exclude='__pycache__' \
  --exclude='*.pyc' \
  -C "$(dirname "$ISAACLAB_ROOT")" "$(basename "$ISAACLAB_ROOT")"

log "=== [4/6] Vulkan + EGL ICD ==="
if [[ -d "$VULKAN_ROOT" ]]; then
  tar -czf "$OUT_ROOT/runtime/vulkan_sdk.tar.gz" -C "$(dirname "$VULKAN_ROOT")" "$(basename "$VULKAN_ROOT")"
fi
if [[ -f "$VK_ICD" ]]; then
  cp "$VK_ICD" "$OUT_ROOT/runtime/nvidia_egl_icd.json"
fi

log "=== [5/6] install scripts + env template ==="
cp "$(dirname "$0")/install_full_from_shared.sh" \
   "$(dirname "$0")/relocate_conda_prefix.sh" \
   "$(dirname "$0")/env_dexpbt.sh.example" "$OUT_ROOT/scripts/"
chmod +x "$OUT_ROOT/scripts/install_full_from_shared.sh" "$OUT_ROOT/scripts/relocate_conda_prefix.sh"

cat > "$OUT_ROOT/00_README.txt" <<EOF
DexPBT full migration bundle (code + conda env + IsaacLab + Vulkan)

Target layout after install (example):
  /volume/chchen/Pd/wmdexpbt/runtime_root/
    envs/isaaclab311/          <- unpacked conda-pack
    src/dexpbt_isaaclab/
    src/IsaacLab/
    runtime/vulkan_sdk_1.4.304.0/
    runtime/nvidia_egl_icd.json

On GPU node:
  source /volume/.../runtime_root/scripts/env_dexpbt.sh
  cd \$DEXPBT_ROOT
  bash run_train_supervisor.sh ...

Verify after transfer:
  gzip -t conda/${CONDA_ENV_NAME}.tar.gz
  sha256sum -c SHA256SUMS
EOF

log "=== [6/6] checksums (+ optional split) ==="
( cd "$OUT_ROOT" && find . -type f ! -name 'SHA256SUMS' ! -name '*.part_*' -print0 | xargs -0 sha256sum > SHA256SUMS )

if [[ "$SPLIT_GB" != "0" ]]; then
  log "Splitting large archives into ${SPLIT_GB}G parts under $OUT_ROOT/split/"
  mkdir -p "$OUT_ROOT/split"
  for f in "$OUT_ROOT/conda"/*.tar.gz "$OUT_ROOT/code"/*.tar.gz "$OUT_ROOT/runtime"/*.tar.gz; do
    [[ -f "$f" ]] || continue
    base="$(basename "$f")"
    split -b "${SPLIT_GB}G" -d -a 3 "$f" "$OUT_ROOT/split/${base}.part_"
    rm -f "$f"
    log "split $base -> $OUT_ROOT/split/${base}.part_*"
  done
  ( cd "$OUT_ROOT" && find . -type f ! -name 'SHA256SUMS' -print0 | xargs -0 sha256sum > SHA256SUMS )
fi

du -sh "$OUT_ROOT"/* | tee "$OUT_ROOT/logs/sizes.txt"
log "DONE: $OUT_ROOT"
log "Copy this entire directory to CPU shared disk, then run scripts/install_full_from_shared.sh on GPU node."
