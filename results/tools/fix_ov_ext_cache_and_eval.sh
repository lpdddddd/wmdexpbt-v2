#!/usr/bin/env bash
# One-shot: repair Isaac Kit extension symlinks, then launch pinned cap sweep.
#
# Uses runtime_root/scripts/restore_ov_ext_cache.py (fresh download + zip test).
# Slow CloudFront downloads (~88MB each for pip.compute / replicator.core) may
# take 10-20 min each on AutoDL; run inside screen/tmux.
#
# Usage:
#   nohup bash results/fix_ov_ext_cache_and_eval.sh > /tmp/fix_ov_ext.log 2>&1 &
#   tail -f /tmp/fix_ov_ext.log

set -euo pipefail

WMDEXPBT_ROOT="/root/autodl-tmp/wmdexpbt"
RUNTIME_ROOT="${WMDEXPBT_ROOT}/runtime_root"
export RUNTIME_ROOT
export OV_EXT_WORKERS="${OV_EXT_WORKERS:-1}"
export OV_EXT_MAX_RETRIES="${OV_EXT_MAX_RETRIES:-8}"

LOCAL_OV="/root/.local/share/ov/data/exts/v2"
CACHE_OV="${RUNTIME_ROOT}/cache/ov/data/exts/v2"

echo "[fix_ov] RUNTIME_ROOT=${RUNTIME_ROOT}"
echo "[fix_ov] OV_EXT_WORKERS=${OV_EXT_WORKERS} OV_EXT_MAX_RETRIES=${OV_EXT_MAX_RETRIES}"

mkdir -p "$(dirname "${LOCAL_OV}")"
if [[ -L "${LOCAL_OV}" ]] || [[ ! -d "${LOCAL_OV}" ]]; then
  rm -rf "${LOCAL_OV}"
  ln -sfn "${CACHE_OV}" "${LOCAL_OV}"
  echo "[fix_ov] linked ${LOCAL_OV} -> ${CACHE_OV}"
fi

# Kill stale eval before restore (avoid kit pulling extensions concurrently).
pkill -f eval_cap_sweep.py 2>/dev/null || true
pkill -f eval_fair 2>/dev/null || true
sleep 2

# Remove known-bad partial zips so restore re-downloads cleanly.
rm -f /tmp/ov_ext_zips/omni.pip.compute-1.6.3+107.3.3.lx64.cp311.zip
rm -f /tmp/ov_ext_zips/omni.replicator.core-1.12.27+107.3.3.lx64.r.cp311.zip
rm -f /tmp/ov_ext_zips/omni.hydra.rtx-1.0.0+69cbf6ad.lx64.r.zip
rm -f /tmp/ov_ext_zips/omni.hydra.rtx.shadercache.vulkan-1.0.0+69cbf6ad.lx64.r.zip
rm -f /tmp/ov_ext_zips/omni.sensors.nv.common-3.0.0+69cbf6ad.lx64.r.cp311.zip

attempt=0
while true; do
  attempt=$((attempt + 1))
  echo "[fix_ov] restore attempt ${attempt} ..."
  if python3 "${RUNTIME_ROOT}/scripts/restore_ov_ext_cache.py" 2>&1 | tee "/tmp/restore_ov_ext_attempt_${attempt}.log"; then
    echo "[fix_ov] restore OK on attempt ${attempt}"
    break
  fi
  if [[ "${attempt}" -ge 5 ]]; then
    echo "[fix_ov] FATAL: still broken after ${attempt} attempts" >&2
    exit 1
  fi
  echo "[fix_ov] retrying in 30s ..."
  sleep 30
done

# Kill stale eval if any.
pkill -f eval_cap_sweep.py 2>/dev/null || true
pkill -f eval_fair 2>/dev/null || true
sleep 2

echo "[fix_ov] launching pinned cap sweep ..."
bash "${WMDEXPBT_ROOT}/results/eval_pinned_cap_sweep.sh"
echo "[fix_ov] eval launched; tail eval_pinned.log under the formal run dir"
