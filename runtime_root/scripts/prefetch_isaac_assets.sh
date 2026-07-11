#!/usr/bin/env bash
# 在 CPU 节点（有网）运行。把 Isaac 5.1 资产拉到共享盘 cache/isaac_assets/5.1/
#
# 说明：S3 不支持「列目录」，wget -r 目录 URL 会 404，不代表文件不存在。
# 推荐优先用 aws s3 sync 拉整棵目录；没有 aws 时再逐个 wget。
set -euo pipefail

RUNTIME_ROOT="${RUNTIME_ROOT:-/root/autodl-tmp/wmdexpbt/runtime_root}"
DEST="$RUNTIME_ROOT/cache/isaac_assets/5.1"
BASE="https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1"
S3="s3://omniverse-content-production/Assets/Isaac/5.1"

log() { echo "[$(date +%H:%M:%S)] $*"; }

download_one() {
  local rel="$1"
  local out="$DEST/$rel"
  mkdir -p "$(dirname "$out")"
  log "wget $rel"
  if wget -q -c "$BASE/$rel" -O "$out"; then
    local sz
    sz=$(stat -c%s "$out" 2>/dev/null || echo 0)
    if [[ "$sz" -lt 200 ]]; then
      log "  WARN: only ${sz} bytes (可能 404 或错误页)"
      return 1
    fi
    log "  OK (${sz} bytes)"
  else
    log "  FAIL"
    return 1
  fi
}

mkdir -p "$DEST"

# --- 方式 A：aws s3 sync（推荐，一次拉全目录）---
if command -v aws >/dev/null 2>&1; then
  log "aws s3 sync KukaAllegro (整目录)..."
  aws s3 sync --no-sign-request \
    "${S3}/Isaac/IsaacLab/Robots/KukaAllegro/" \
    "${DEST}/Isaac/IsaacLab/Robots/KukaAllegro/" || log "WARN: KukaAllegro sync partial"

  log "aws s3 sync DexCube..."
  aws s3 sync --no-sign-request \
    "${S3}/Isaac/Props/Blocks/DexCube/" \
    "${DEST}/Isaac/Props/Blocks/DexCube/" || true

  log "aws s3 sync Grid environment..."
  aws s3 sync --no-sign-request \
    "${S3}/Isaac/Environments/Grid/" \
    "${DEST}/Isaac/Environments/Grid/" || true
else
  log "aws CLI 不可用，改用 wget 逐个文件"
fi

# --- 方式 B：wget 兜底（smoke test 已知依赖）---
WGET_FILES=(
  # KukaAllegro
  "Isaac/IsaacLab/Robots/KukaAllegro/kuka.usd"
  "Isaac/IsaacLab/Robots/KukaAllegro/configuration/kuka_primitive_collider.usd"
  "Isaac/IsaacLab/Robots/KukaAllegro/configuration/allegro_primitive_collider.usd"
  "Isaac/IsaacLab/Robots/KukaAllegro/Props/collisions.usd"
  "Isaac/IsaacLab/Robots/KukaAllegro/Props/visuals.usd"
  # DexCube
  "Isaac/Props/Blocks/DexCube/dex_cube_instanceable.usd"
  "Isaac/Props/Blocks/DexCube/Props/instanceable_meshes.usd"
  # 地面
  "Isaac/Environments/Grid/default_environment.usd"
)

fail=0
for f in "${WGET_FILES[@]}"; do
  [[ -f "$DEST/$f" ]] && continue
  download_one "$f" || fail=$((fail + 1))
done

log "--- 校验 ---"
for need in \
  "$DEST/Isaac/IsaacLab/Robots/KukaAllegro/Props/collisions.usd" \
  "$DEST/Isaac/IsaacLab/Robots/KukaAllegro/Props/visuals.usd" \
  "$DEST/Isaac/Props/Blocks/DexCube/Props/instanceable_meshes.usd" \
  "$DEST/Isaac/Environments/Grid/default_environment.usd"; do
  if [[ -f "$need" ]]; then
    log "OK $(basename "$need") ($(stat -c%s "$need") bytes)"
  else
    log "MISSING $need"
    fail=$((fail + 1))
  fi
done
du -sh "$DEST"
[[ "$fail" -eq 0 ]] || { log "仍有 $fail 个关键文件缺失"; exit 1; }
log "资产预取完成。GPU 节点: source env_dexpbt.sh && python smoke_test.py"
