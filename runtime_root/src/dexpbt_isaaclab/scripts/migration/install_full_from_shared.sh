#!/usr/bin/env bash
# Install full migration bundle on target machine (shared disk, GPU node recommended for conda-unpack test).
#
# Usage:
#   bash install_full_from_shared.sh /volume/chchen/Pd/wmdexpbt/full_migration \
#                                    /volume/chchen/Pd/wmdexpbt/runtime_root
set -euo pipefail

BUNDLE_DIR="${1:?usage: $0 <BUNDLE_DIR> <RUNTIME_ROOT>}"
RUNTIME_ROOT="${2:?usage: $0 <BUNDLE_DIR> <RUNTIME_ROOT>}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-isaaclab311}"

log() { echo "[$(date +%H:%M:%S)] $*"; }

# Reassemble split parts if present
if [[ -d "$BUNDLE_DIR/split" ]]; then
  log "Reassembling split archives..."
  mkdir -p "$BUNDLE_DIR/conda" "$BUNDLE_DIR/code" "$BUNDLE_DIR/runtime"
  for prefix in "$BUNDLE_DIR/split"/*.part_000; do
    [[ -f "$prefix" ]] || continue
    base="$(basename "$prefix" .part_000)"
    out=""
    if [[ "$base" == *.tar.gz ]]; then
      if [[ "$base" == "${CONDA_ENV_NAME}.tar.gz" ]]; then
        out="$BUNDLE_DIR/conda/$base"
      elif [[ "$base" == "dexpbt_isaaclab.tar.gz" || "$base" == "IsaacLab.tar.gz" ]]; then
        out="$BUNDLE_DIR/code/$base"
      else
        out="$BUNDLE_DIR/runtime/$base"
      fi
      cat "$BUNDLE_DIR/split/${base}.part_"* > "$out"
      log "assembled $out"
    fi
  done
fi

log "[1/5] Verify SHA256SUMS"
( cd "$BUNDLE_DIR" && sha256sum -c SHA256SUMS )

log "[2/5] Unpack conda env"
mkdir -p "$RUNTIME_ROOT/envs"
tar -xzf "$BUNDLE_DIR/conda/${CONDA_ENV_NAME}.tar.gz" -C "$RUNTIME_ROOT/envs"
ENV_DIR="$RUNTIME_ROOT/envs/${CONDA_ENV_NAME}"
# Fix absolute paths baked into env
if [[ -x "$ENV_DIR/bin/conda-unpack" ]]; then
  "$ENV_DIR/bin/conda-unpack"
elif [[ -f "$BUNDLE_DIR/conda/.source_prefix" ]]; then
  OLD_PREFIX="$(tr -d '\n' < "$BUNDLE_DIR/conda/.source_prefix")"
  if [[ "$OLD_PREFIX" != "$ENV_DIR" ]]; then
    log "Relocating prefix: $OLD_PREFIX -> $ENV_DIR"
    bash "$(dirname "$0")/relocate_conda_prefix.sh" "$OLD_PREFIX" "$ENV_DIR"
  fi
else
  log "WARN: no conda-unpack and no .source_prefix; env may have broken paths"
fi

log "[3/5] Unpack code"
mkdir -p "$RUNTIME_ROOT/src"
tar -xzf "$BUNDLE_DIR/code/dexpbt_isaaclab.tar.gz" -C "$RUNTIME_ROOT/src"
tar -xzf "$BUNDLE_DIR/code/IsaacLab.tar.gz" -C "$RUNTIME_ROOT/src"

log "[4/5] Unpack runtime (Vulkan)"
mkdir -p "$RUNTIME_ROOT/runtime"
if [[ -f "$BUNDLE_DIR/runtime/vulkan_sdk.tar.gz" ]]; then
  tar -xzf "$BUNDLE_DIR/runtime/vulkan_sdk.tar.gz" -C "$RUNTIME_ROOT/runtime"
fi
if [[ -f "$BUNDLE_DIR/runtime/nvidia_egl_icd.json" ]]; then
  cp "$BUNDLE_DIR/runtime/nvidia_egl_icd.json" "$RUNTIME_ROOT/runtime/"
fi

log "[5/6] pip install -e (editable pkgs not in conda-pack)"
source "$RUNTIME_ROOT/envs/${CONDA_ENV_NAME}/bin/activate" 2>/dev/null || true
export PATH="$RUNTIME_ROOT/envs/${CONDA_ENV_NAME}/bin:$PATH"
pip install --no-index --find-links "$BUNDLE_DIR/wheels" 2>/dev/null || true
pip install -e "$RUNTIME_ROOT/src/IsaacLab/source/isaaclab" \
  -e "$RUNTIME_ROOT/src/IsaacLab/source/isaaclab_assets" \
  -e "$RUNTIME_ROOT/src/IsaacLab/source/isaaclab_rl" \
  -e "$RUNTIME_ROOT/src/IsaacLab/source/isaaclab_tasks" \
  -e "$RUNTIME_ROOT/src/dexpbt_isaaclab" 2>/dev/null || {
  log "WARN: pip install -e failed offline; ensure wheels exist or install on networked machine once"
}

log "[6/6] Write env_dexpbt.sh"
ENV_SH="$RUNTIME_ROOT/scripts/env_dexpbt.sh"
mkdir -p "$RUNTIME_ROOT/scripts"
cat > "$ENV_SH" <<EOF
# Source this before training: source $ENV_SH
export RUNTIME_ROOT="$RUNTIME_ROOT"
export DEXPBT_ROOT="\$RUNTIME_ROOT/src/dexpbt_isaaclab"
export ISAACLAB_ROOT="\$RUNTIME_ROOT/src/IsaacLab"
export CONDA_PREFIX="\$RUNTIME_ROOT/envs/${CONDA_ENV_NAME}"
export PATH="\$CONDA_PREFIX/bin:\$PATH"
export VULKAN_SDK="\$RUNTIME_ROOT/runtime/vulkan_sdk_1.4.304.0/x86_64"
export PATH="\$VULKAN_SDK/bin:\$PATH"
export LD_LIBRARY_PATH="\$VULKAN_SDK/lib\${LD_LIBRARY_PATH:+:\$LD_LIBRARY_PATH}"
export VK_LAYER_PATH="\$VULKAN_SDK/share/vulkan/explicit_layer.d"
export VK_ICD_FILENAMES="\$RUNTIME_ROOT/runtime/nvidia_egl_icd.json"
export XDG_RUNTIME_DIR="\${XDG_RUNTIME_DIR:-/tmp/runtime-\$USER}"
mkdir -p "\$XDG_RUNTIME_DIR" && chmod 700 "\$XDG_RUNTIME_DIR" 2>/dev/null || true
export TORCHDYNAMO_DISABLE=1
export PYTHONUNBUFFERED=1
# Isaac Lab typically expects this on PYTHONPATH
export PYTHONPATH="\$ISAACLAB_ROOT/source:\$ISAACLAB_ROOT:\${PYTHONPATH:-}"
EOF
chmod +x "$ENV_SH"
if [[ -f "$BUNDLE_DIR/scripts/install_full_from_shared.sh" ]]; then
  cp "$BUNDLE_DIR/scripts/install_full_from_shared.sh" "$RUNTIME_ROOT/scripts/"
fi

log "DONE. Next on GPU node:"
log "  source $ENV_SH"
log "  python -c \"import torch; print(torch.__version__, torch.cuda.is_available())\""
log "  cd \$DEXPBT_ROOT && bash run_train_supervisor.sh <RUN_DIR> DexPBT-Regrasping-KukaAllegro-Direct-v0 ..."
