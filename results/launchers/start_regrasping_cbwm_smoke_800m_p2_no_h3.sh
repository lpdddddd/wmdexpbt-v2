#!/usr/bin/env bash
# Optional no_h3 smoke launcher (P2.1-F recommendation).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WM_DISABLE_VALUE_IMAG_H3=1 bash "${SCRIPT_DIR}/start_regrasping_cbwm_smoke_800m_p2.sh"
