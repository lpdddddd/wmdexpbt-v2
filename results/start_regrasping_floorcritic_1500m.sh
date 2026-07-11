#!/usr/bin/env bash
# FloorCritic Phase 1 — effective (shared) 1.5B / policy.
#
# Same shared config as start_regrasping_floorcritic_800m.sh, extended to
# 1.5B / policy to observe whether the floor stage (tol <= 0.02) is more
# stable with FloorCritic. ONLY run after the 800M comparison looks healthy
# (no early nsucc regression, grad ratio within bound).
#
# Implemented as a thin wrapper over the 800M launcher with MAX_ITER + run
# name overrides so the two stay in lock-step on every other flag.
#   1.5B / policy: 131072 frames/iter -> ~11444 iters.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export RUN_NAME_PREFIX="${RUN_NAME_PREFIX:-regrasping_floorcritic_1500m}"
export MAX_ITER="${MAX_ITER:-11444}"

exec bash "${SCRIPT_DIR}/start_regrasping_floorcritic_800m.sh" "$@"
