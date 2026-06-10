#!/usr/bin/env bash
# Fixed-h1 P2.1 800M repeat smoke launcher.
#
# Standalone: does NOT chain through start_regrasping_cbwm_smoke_800m_p2.sh,
# which sets value_imag_h3 and then expects an override. We set value_imag_h1
# directly here.
#
# Use cases:
#   - Reproducibility check on a different seed family.
#   - Bring-up sanity for the next PBT/observer change.
#
# Constraints:
#   - No PPO / reward / obs / WM-architecture / CF-runner change.
#   - No value_imag_h3 in the strategy pool.
#   - max_consecutive_successes stays at the env default 50.
#
# Common overrides (set in env before calling):
#   BASE_SEED                     - Per-policy seed = BASE + 1000*policy_idx (default 42).
#   DEXPBT_MAX_ACTIVE             - Concurrent warm-ups (default 2).
#   DEXPBT_INTER_LAUNCH_DELAY_S   - Inter-spawn delay seconds (default 90).
#   DEXPBT_STARTUP_TIMEOUT_S      - Watchdog startup timeout (default 900).
#   DEXPBT_STARTUP_MAX_RESTARTS   - Watchdog max restarts (default 1).
#   NUM_POLICIES / DEXPBT_GPUS    - Population shape (default 6 / 0,1,2,3,4,5).
#
# Usage:
#   BASE_SEED=43 nohup bash start_regrasping_fixed_h1_p2_1_smoke_800m.sh \
#     > nohup_repeat_smoke.log 2>&1 &
#
# Reporting:
#   python results/generate_smoke_report_p2.py "$RUN_DIR"

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNTIME_ROOT="/root/autodl-tmp/wmdexpbt/runtime_root"
RESULTS_ROOT="/root/autodl-tmp/wmdexpbt/results"
RUNS_ROOT="${RESULTS_ROOT}/runs"
DATE=$(date +%Y%m%d_%H%M%S)
RUN_DIR="${RUNS_ROOT}/regrasping_fixed_h1_p2_1_smoke_800m_${DATE}"
LOG="${RUN_DIR}/launch_master.log"
PIDFILE="${RUN_DIR}/nohup.pid"
WS="${RUN_DIR}/dexpbt_8gpu_ws"

BASE_SEED="${BASE_SEED:-42}"
NUM_POLICIES="${NUM_POLICIES:-6}"
DEXPBT_GPUS="${DEXPBT_GPUS:-0,1,2,3,4,5}"
SMOKE_MAX_ITER="${SMOKE_MAX_ITER:-6112}"
SMOKE_PBT_START_AFTER="${SMOKE_PBT_START_AFTER:-200000000}"
SMOKE_PBT_INITIAL_DELAY="${SMOKE_PBT_INITIAL_DELAY:-50000000}"
ROLLBACK_COOLDOWN_FRAMES="${ROLLBACK_COOLDOWN_FRAMES:-50000000}"
CROSSCOPY_TOLERANCE_MODE="${CROSSCOPY_TOLERANCE_MODE:-last_mastered}"

export DEXPBT_MAX_ACTIVE="${DEXPBT_MAX_ACTIVE:-2}"
export DEXPBT_INTER_LAUNCH_DELAY_S="${DEXPBT_INTER_LAUNCH_DELAY_S:-90}"
export DEXPBT_STARTUP_TIMEOUT_S="${DEXPBT_STARTUP_TIMEOUT_S:-900}"
export DEXPBT_STARTUP_MAX_RESTARTS="${DEXPBT_STARTUP_MAX_RESTARTS:-1}"
export NUM_POLICIES
export DEXPBT_GPUS

mkdir -p "$RUN_DIR"
echo "$RUN_DIR" > "${RESULTS_ROOT}/pointers/latest_run.txt"
echo "$RUN_DIR" > "${RESULTS_ROOT}/pointers/latest_smoke_run.txt"
echo "$RUN_DIR" > "${RUN_DIR}/RUN_DIR.txt"

cat > "${RUN_DIR}/RUN_CONFIG.txt" <<EOF
launcher=start_regrasping_fixed_h1_p2_1_smoke_800m.sh
profile=smoke_repeat
target_frames=800000000
max_iterations=${SMOKE_MAX_ITER}
num_policies=${NUM_POLICIES}
gpus=${DEXPBT_GPUS}
base_seed=${BASE_SEED}
wm_strategy_mode=fixed
fixed_wm_strategy=value_imag_h1
value_imag_h3_disabled=true
cf_runner_enabled=false
crosscopy_tolerance_mode=${CROSSCOPY_TOLERANCE_MODE}
recovery_tolerance_mode=last_mastered
rollback_cooldown_frames=${ROLLBACK_COOLDOWN_FRAMES}
max_active=${DEXPBT_MAX_ACTIVE}
inter_launch_delay_s=${DEXPBT_INTER_LAUNCH_DELAY_S}
startup_timeout_s=${DEXPBT_STARTUP_TIMEOUT_S}
startup_max_restarts=${DEXPBT_STARTUP_MAX_RESTARTS}
max_consecutive_successes=50
EOF

cat > "${RUN_DIR}/SMOKE_CONFIG.txt" <<EOF
smoke_run=true
patch=P2_stability
target_frames=800000000
max_iterations=${SMOKE_MAX_ITER}
curriculum_mode=mastery
pbt_single_shrink_per_window=true
pbt_early_collapse_detect=true
pbt_recovery_tolerance_mode=last_mastered
pbt_recovery_bypass_elimination_burnin=true
note=fixed-h1 repeat smoke; no h3; no CF; staggered launch
EOF

# shellcheck source=/dev/null
source "${RUNTIME_ROOT}/scripts/env_dexpbt.sh"
export TORCHDYNAMO_DISABLE=1
export PYTHONUNBUFFERED=1
export OMNI_KIT_TELEMETRY=0
export CARB_CRASH_REPORTER=0

cd "${RUNTIME_ROOT}/src/dexpbt_isaaclab"

echo "Starting fixed-h1 P2.1 800M smoke (no h3, no CF):"
echo "  RUN_DIR=$RUN_DIR"
echo "  num_policies=$NUM_POLICIES gpus=$DEXPBT_GPUS base_seed=$BASE_SEED"
echo "  staggered: max_active=$DEXPBT_MAX_ACTIVE delay=${DEXPBT_INTER_LAUNCH_DELAY_S}s"
echo "  rollback_cooldown=${ROLLBACK_COOLDOWN_FRAMES} crosscopy_tol=${CROSSCOPY_TOLERANCE_MODE}"

nohup bash "${RUNTIME_ROOT}/scripts/launch_8gpu_dexpbt.sh" \
  "$RUN_DIR" \
  --task DexPBT-Regrasping-KukaAllegro-Direct-v0 \
  --num-envs 8192 \
  --max-iterations "$SMOKE_MAX_ITER" \
  --pbt-interval-steps 20000000 \
  --pbt-start-after "$SMOKE_PBT_START_AFTER" \
  --pbt-initial-delay "$SMOKE_PBT_INITIAL_DELAY" \
  --extra \
  --seed "$BASE_SEED" \
  --enable_wm \
  --wm_strategy value_imag_h1 \
  --wm_ensemble_size 3 \
  --wm_lr 1e-3 \
  --wm_update_every_n_epochs 2 \
  --wm_batch_size 512 \
  --wm_strategy_mode fixed \
  --fixed_wm_strategy value_imag_h1 \
  --pbt_ranking_metric success_aware_r_meta \
  --pbt_nsucc_zero_cap 0.05 \
  --pbt_nsucc_low_threshold 1.0 \
  --pbt_nsucc_low_cap 0.05 \
  --pbt_min_frames_as_source_after_restart 50000000 \
  --pbt_min_frames_between_restarts 50000000 \
  --pbt_min_nsucc_mean_as_source 1.0 \
  --pbt_min_nsucc_max_as_source 5 \
  --pbt_allow_low_nsucc_source_fallback \
  --pbt_disable_self_restart \
  --pbt_max_self_restarts_per_policy 3 \
  --pbt_min_frames_between_self_restarts 200000000 \
  --pbt_source_selection_mode tiered \
  --pbt_relaxed_nsucc_mean_as_source 0.5 \
  --pbt_relaxed_nsucc_max_as_source 2 \
  --no-pbt_enable_best_nsucc_recovery \
  --pbt_allow_best_nsucc_archive_recovery \
  --pbt_recovery_min_nsucc_mean 2.0 \
  --pbt_recovery_min_nsucc_max 8 \
  --pbt_recovery_cooldown_frames 400000000 \
  --pbt_curriculum_freeze_frames 100000000 \
  --pbt_curriculum_freeze_on_recovery_frames 100000000 \
  --curriculum_mode mastery \
  --pbt_source_require_stable_mean \
  --mastery_nsucc_mean 4.0 \
  --mastery_nsucc_max 20.0 \
  --mastery_consecutive_windows 2 \
  --curriculum_shrink_factor 0.95 \
  --pbt_single_shrink_per_window \
  --post_shrink_consolidation_env_frames 100000000 \
  --pbt_early_collapse_detect \
  --early_collapse_drop_ratio 0.3 \
  --early_collapse_patience 1 \
  --curriculum_collapse_detect \
  --collapse_drop_ratio 0.5 \
  --collapse_patience 2 \
  --collapse_freeze_env_frames 200000000 \
  --stable_source_min_mean 3.0 \
  --stable_source_min_max 20.0 \
  --stable_source_max_drop_ratio 0.3 \
  --stable_source_drop_windows 2 \
  --pbt_recovery_bypass_elimination_burnin \
  --pbt_recovery_tolerance_mode last_mastered \
  --pbt_crosscopy_tolerance_mode "$CROSSCOPY_TOLERANCE_MODE" \
  --pbt_recovery_source_cooldown_frames 50000000 \
  --pbt_freeze_mutation_after_stable_frames 100000000 \
  --curriculum_rollback_cooldown_frames "$ROLLBACK_COOLDOWN_FRAMES" \
  --pbt_usable_live_min_mean 1.5 \
  --pbt_usable_live_min_max 20 \
  --pbt_usable_live_max_drop_ratio 0.5 \
  >"$LOG" 2>&1 &

echo $! > "$PIDFILE"
echo "launch_master pid=$(cat "$PIDFILE") log=$LOG"
echo "Report: python ${RESULTS_ROOT}/generate_smoke_report_p2.py ${RUN_DIR}"
