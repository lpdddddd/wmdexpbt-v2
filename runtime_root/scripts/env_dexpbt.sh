# Source this before training: source /root/autodl-tmp/wmdexpbt/runtime_root/scripts/env_dexpbt.sh
export RUNTIME_ROOT="/root/autodl-tmp/wmdexpbt/runtime_root"
export DEXPBT_ROOT="$RUNTIME_ROOT/src/dexpbt_isaaclab"
export ISAACLAB_ROOT="$RUNTIME_ROOT/src/IsaacLab"
export CONDA_PREFIX="$RUNTIME_ROOT/envs/isaaclab311"
export PATH="$CONDA_PREFIX/bin:$PATH"
export VULKAN_SDK="$RUNTIME_ROOT/runtime/vulkan_sdk_1.4.304.0/x86_64"
export PATH="$VULKAN_SDK/bin:$PATH"
export LD_LIBRARY_PATH="$VULKAN_SDK/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export VK_LAYER_PATH="$VULKAN_SDK/share/vulkan/explicit_layer.d"
export VK_ICD_FILENAMES="$RUNTIME_ROOT/runtime/nvidia_egl_icd.json"
export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/tmp/runtime-$USER}"
mkdir -p "$XDG_RUNTIME_DIR" && chmod 700 "$XDG_RUNTIME_DIR" 2>/dev/null || true
export TORCHDYNAMO_DISABLE=1
export PYTHONUNBUFFERED=1
# Isaac Lab typically expects this on PYTHONPATH
export PYTHONPATH="$ISAACLAB_ROOT/source:$ISAACLAB_ROOT:${PYTHONPATH:-}"

# Offline USD assets (wget to $RUNTIME_ROOT/cache/isaac_assets/5.1/ on CPU node)
export ISAAC_ASSET_ROOT="${ISAAC_ASSET_ROOT:-$RUNTIME_ROOT/cache/isaac_assets/5.1}"
export DEXPBT_KIT_ARGS="--/persistent/isaac/asset_root/default=file://${ISAAC_ASSET_ROOT} --/persistent/isaac/asset_root/cloud=file://${ISAAC_ASSET_ROOT} --/persistent/isaac/asset_root/nvidia=file://${ISAAC_ASSET_ROOT} --/persistent/isaac/asset_root/timeout=5"
