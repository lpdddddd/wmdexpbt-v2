#!/usr/bin/env bash
# P2.1 no-h3 ablation smoke: fixed value_imag_h1 + 50M rollback cooldown
# + cross-copy last_mastered tolerance + staggered isolated launch + CF off.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WM_DISABLE_VALUE_IMAG_H3=1 \
ROLLBACK_COOLDOWN_FRAMES=50000000 \
CROSSCOPY_TOLERANCE_MODE=last_mastered \
RUN_TAG=_1_no_h3 \
ENABLE_CF_RUNNER="${ENABLE_CF_RUNNER:-0}" \
DEXPBT_MAX_ACTIVE="${DEXPBT_MAX_ACTIVE:-2}" \
DEXPBT_INTER_LAUNCH_DELAY_S="${DEXPBT_INTER_LAUNCH_DELAY_S:-90}" \
DEXPBT_STARTUP_TIMEOUT_S="${DEXPBT_STARTUP_TIMEOUT_S:-900}" \
DEXPBT_STARTUP_MAX_RESTARTS="${DEXPBT_STARTUP_MAX_RESTARTS:-1}" \
bash "${SCRIPT_DIR}/start_regrasping_cbwm_smoke_800m_p2.sh"
