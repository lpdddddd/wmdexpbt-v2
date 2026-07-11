#!/usr/bin/env bash
# Sync wmdexpbt codebase to a remote machine via rsync over SSH.
#
# Usage:
#   ./scripts/rsync_to_remote.sh code
#   ./scripts/rsync_to_remote.sh code --dest /root/autodl-tmp/wmdexpbt
#   ./scripts/rsync_to_remote.sh full-env   # includes conda env + cache (~18G extra)
#   ./scripts/rsync_to_remote.sh run <RUN_DIR_NAME>  # one experiment run only
#
# Default remote (override with env vars):
#   RSYNC_SSH_PORT=22529
#   RSYNC_SSH_HOST=connect.westb.seetacloud.com
#   RSYNC_SSH_USER=root
#   RSYNC_DEST=/root/autodl-tmp/wmdexpbt

set -euo pipefail

MODE="${1:-code}"
shift || true

RSYNC_SSH_PORT="${RSYNC_SSH_PORT:-22529}"
RSYNC_SSH_HOST="${RSYNC_SSH_HOST:-connect.westb.seetacloud.com}"
RSYNC_SSH_USER="${RSYNC_SSH_USER:-root}"
RSYNC_DEST="${RSYNC_DEST:-/root/autodl-tmp/wmdexpbt}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC="$(cd "${SCRIPT_DIR}/.." && pwd)"

SSH_TARGET="${RSYNC_SSH_USER}@${RSYNC_SSH_HOST}"
SSH_CMD="ssh -p ${RSYNC_SSH_PORT}"
RSYNC_SSH="${SSH_CMD} -o ServerAliveInterval=60 -o ServerAliveCountMax=3"

COMMON_EXCLUDES=(
  --exclude '.git/'
  --exclude '__pycache__/'
  --exclude '*.pyc'
  --exclude '.pytest_cache/'
  --exclude '.mypy_cache/'
  --exclude '.cursor/'
  --exclude 'nohup_*.log'
)

CODE_EXCLUDES=(
  "${COMMON_EXCLUDES[@]}"
  --exclude 'results/runs/'
  --exclude 'runtime_root/envs/'
  --exclude 'runtime_root/cache/'
  --exclude 'runtime_root/runtime/'
  --exclude 'runtime_root/src/dexpbt_isaaclab/logs/'
  --exclude 'runtime_root/src/dexpbt_isaaclab/audit/'
  --exclude 'runtime_root/src/dexpbt_isaaclab/results/'
  --exclude '**/*.tfevents*'
  --exclude '**/*.pth'
  --exclude '**/*.pt'
)

FULL_ENV_EXCLUDES=(
  "${COMMON_EXCLUDES[@]}"
  --exclude 'results/runs/'
)

parse_dest() {
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --dest) RSYNC_DEST="$2"; shift 2 ;;
      *) echo "Unknown arg: $1" >&2; exit 1 ;;
    esac
  done
}

run_rsync() {
  local -a excludes=("$@")
  echo "==> rsync ${SRC}/ -> ${SSH_TARGET}:${RSYNC_DEST}/"
  echo "    mode=${MODE}  port=${RSYNC_SSH_PORT}"
  "${SSH_CMD}" "${SSH_TARGET}" "mkdir -p '${RSYNC_DEST}'"
  rsync -avh --progress --info=stats2 \
    -e "${RSYNC_SSH}" \
    "${excludes[@]}" \
    "${SRC}/" "${SSH_TARGET}:${RSYNC_DEST}/"
}

case "${MODE}" in
  code)
    parse_dest "$@"
    run_rsync "${CODE_EXCLUDES[@]}"
    ;;
  full-env)
    parse_dest "$@"
    run_rsync "${FULL_ENV_EXCLUDES[@]}"
    ;;
  run)
    RUN_NAME="${1:?usage: $0 run <run_dir_name> [--dest PATH]}"
    shift || true
    parse_dest "$@"
    RUN_SRC="${SRC}/results/runs/${RUN_NAME}"
    if [[ ! -d "${RUN_SRC}" ]]; then
      echo "Run directory not found: ${RUN_SRC}" >&2
      exit 1
    fi
    echo "==> rsync run ${RUN_NAME} -> ${SSH_TARGET}:${RSYNC_DEST}/results/runs/"
    "${SSH_CMD}" "${SSH_TARGET}" "mkdir -p '${RSYNC_DEST}/results/runs'"
    rsync -avh --progress --info=stats2 \
      -e "${RSYNC_SSH}" \
      "${RUN_SRC}/" "${SSH_TARGET}:${RSYNC_DEST}/results/runs/${RUN_NAME}/"
    ;;
  *)
    echo "Unknown mode: ${MODE}" >&2
    echo "Modes: code | full-env | run <name>" >&2
    exit 1
    ;;
esac

echo "Done."
