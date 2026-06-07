#!/usr/bin/env bash
# Run this ON chchen-cpu-2-0 (CPU dev node), NOT on AutoDL.
# Pull full_migration from AutoDL via public SSH (from AutoDL web panel).
#
# Before run, set these from AutoDL console -> SSH connection:
#   AUTODL_SSH_HOST=connect.xxx.seetacloud.com
#   AUTODL_SSH_PORT=12345
#   AUTODL_SSH_USER=root
#   AUTODL_REMOTE_DIR=/root/autodl-tmp/dexpbt_full_migration
#
# Usage on CPU:
#   export AUTODL_SSH_HOST=... AUTODL_SSH_PORT=... 
#   bash pull_on_cpu_from_autodl.sh
set -euo pipefail

AUTODL_SSH_USER="${AUTODL_SSH_USER:-root}"
AUTODL_SSH_HOST="${AUTODL_SSH_HOST:?set AUTODL_SSH_HOST from AutoDL panel}"
AUTODL_SSH_PORT="${AUTODL_SSH_PORT:?set AUTODL_SSH_PORT from AutoDL panel}"
AUTODL_REMOTE_DIR="${AUTODL_REMOTE_DIR:-/root/autodl-tmp/dexpbt_full_migration}"
LOCAL_DIR="${LOCAL_DIR:-/volume/chchen/Pd/wmdexpbt/full_migration}"

mkdir -p "$LOCAL_DIR"
echo "Pulling ${AUTODL_SSH_USER}@${AUTODL_SSH_HOST}:${AUTODL_REMOTE_DIR}/ -> ${LOCAL_DIR}/"

rsync -avh --progress --partial --inplace \
  -e "ssh -p ${AUTODL_SSH_PORT} -o ServerAliveInterval=30" \
  "${AUTODL_SSH_USER}@${AUTODL_SSH_HOST}:${AUTODL_REMOTE_DIR}/" \
  "${LOCAL_DIR}/"

cd "$LOCAL_DIR" && sha256sum -c SHA256SUMS
echo "DONE. Install on GPU: bash scripts/install_full_from_shared.sh $LOCAL_DIR /volume/chchen/Pd/wmdexpbt/runtime_root"
