#!/usr/bin/env bash
# Disposable MSA-aware fusion smoke (control or 6-GPU).
# Synthetic decisions only — not algorithm results.
set -euo pipefail

MODE="${1:-control}"
EXTRA=()
if [[ "$MODE" == "gpu" ]]; then
  if [[ "${2:-}" != "--i-know-what-i-am-doing" ]]; then
    echo "REFUSING GPU smoke: pass --i-know-what-i-am-doing"
    echo "usage: $0 gpu --i-know-what-i-am-doing"
    exit 2
  fi
elif [[ "$MODE" != "control" ]]; then
  echo "usage: $0 [control|gpu --i-know-what-i-am-doing]"
  exit 2
fi

RUNTIME_ROOT="/root/autodl-tmp/wmdexpbt/runtime_root"
# shellcheck disable=SC1091
source "${RUNTIME_ROOT}/scripts/env_dexpbt.sh"
export MSA_FUSION_ALLOW_FAKE_DECISION=1
export PYTHONUNBUFFERED=1
cd "${RUNTIME_ROOT}/src/dexpbt_isaaclab"

python -u -m dexpbt_isaaclab.msa.fusion_smoke --mode "$MODE"
