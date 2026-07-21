#!/usr/bin/env bash
# Disposable integrated MSA-aware fusion smoke (6 GPU).
# Uses production launcher + sidecar adapter path. Optional eval fixture
# (MSA_FUSION_ALLOW_EVAL_FIXTURE=1) feeds episode-level evidence into the
# REAL decide_selective_promotion + fusion command/ack protocol.
# Formal profiles cannot enable the fixture.
set -euo pipefail

if [[ "${1:-}" != "--i-know-what-i-am-doing" ]]; then
  echo "REFUSING: pass --i-know-what-i-am-doing"
  exit 2
fi

RUNTIME_ROOT="/root/autodl-tmp/wmdexpbt/runtime_root"
# shellcheck disable=SC1091
source "${RUNTIME_ROOT}/scripts/env_dexpbt.sh"
export PYTHONUNBUFFERED=1
# Fixture allowed ONLY for this disposable smoke (not formal profiles).
export MSA_FUSION_ALLOW_EVAL_FIXTURE=1
export FUSION_SMOKE_NUM_ENVS="${FUSION_SMOKE_NUM_ENVS:-512}"
export FUSION_SMOKE_WAIT_S="${FUSION_SMOKE_WAIT_S:-1800}"
export DEXPBT_INTER_LAUNCH_DELAY_S="${DEXPBT_INTER_LAUNCH_DELAY_S:-15}"

cd "${RUNTIME_ROOT}/src/dexpbt_isaaclab"
python -u -m dexpbt_isaaclab.msa.fusion_smoke --mode integrated
