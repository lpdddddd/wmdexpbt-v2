#!/usr/bin/env bash
# Build full_migration (if needed) and rsync directly to CPU shared disk.
#
# Usage:
#   export CPU_TARGET="root@chchen-gpu-4-0:/volume/chchen/Pd/wmdexpbt/full_migration"
#   bash scripts/migration/push_full_to_cpu.sh
#
# Or one line:
#   bash scripts/migration/push_full_to_cpu.sh root@chchen-gpu-4-0:/volume/chchen/Pd/wmdexpbt/full_migration
#
# Env:
#   LOCAL_ROOT   default /root/autodl-tmp/dexpbt_full_migration
#   SKIP_BUILD=1 skip conda-pack, only rsync existing LOCAL_ROOT
#   SPLIT_GB=4   split large tars before transfer (reliable on bad networks)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEXPBT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
CPU_TARGET="${1:-${CPU_TARGET:-}}"
LOCAL_ROOT="${LOCAL_ROOT:-/root/autodl-tmp/dexpbt_full_migration}"

if [[ -z "$CPU_TARGET" ]]; then
  echo "ERROR: set CPU target, e.g."
  echo "  bash $0 root@chchen-gpu-4-0:/volume/chchen/Pd/wmdexpbt/full_migration"
  exit 1
fi

# Parse user@host:path
if [[ "$CPU_TARGET" =~ ^([^@]+@[^:]+):(.+)$ ]]; then
  SSH_HOST="${BASH_REMATCH[1]}"
  REMOTE_DIR="${BASH_REMATCH[2]}"
else
  echo "ERROR: CPU_TARGET must look like user@host:/absolute/path"
  exit 1
fi

log() { echo "[$(date +%H:%M:%S)] $*"; }

log "Target: $SSH_HOST:$REMOTE_DIR"
ssh -o BatchMode=yes -o ConnectTimeout=10 "$SSH_HOST" "mkdir -p '$REMOTE_DIR'"

if [[ "${SKIP_BUILD:-0}" != "1" ]]; then
  log "Building full migration at $LOCAL_ROOT (may take 30-60+ min)..."
  SPLIT_GB="${SPLIT_GB:-0}" bash "$SCRIPT_DIR/bundle_full_to_shared.sh" "$LOCAL_ROOT"
else
  log "SKIP_BUILD=1, using existing $LOCAL_ROOT"
fi

if [[ ! -f "$LOCAL_ROOT/SHA256SUMS" ]]; then
  echo "ERROR: $LOCAL_ROOT/SHA256SUMS missing — run bundle first"
  exit 1
fi

log "Rsync to CPU (resumable, with checksum)..."
# Trailing slash on LOCAL_ROOT/ => copy contents into REMOTE_DIR
rsync -avh --progress --partial --inplace \
  -e "ssh -o ServerAliveInterval=30" \
  "$LOCAL_ROOT/" \
  "${SSH_HOST}:${REMOTE_DIR}/"

log "Remote verify SHA256 (sample)..."
ssh "$SSH_HOST" "cd '$REMOTE_DIR' && sha256sum -c SHA256SUMS 2>&1 | tail -5"

log "DONE. On GPU node install with:"
echo "  ssh $SSH_HOST"
echo "  bash $REMOTE_DIR/scripts/install_full_from_shared.sh $REMOTE_DIR /volume/chchen/Pd/wmdexpbt/runtime_root"
