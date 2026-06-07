#!/usr/bin/env bash
# Track C: bayesian-no-h3 P2.1 3B formal launcher.
#
# Hybrid of fixed_h1 P2.1 5B (P2.1 stability + 6 policies + horizon
# scaffolding) and CB-WM smoke launcher (CF runner + Bayesian selector).
# Strategy pool is restricted to {none, aux_mid, value_imag_h1} via the
# new --wm_strategy_pool CLI from train.py — h3 and aux_low are excluded
# permanently. CF runner is ON.
#
# Constraints (all enforced as hard pre-launch asserts below):
#   - max_consecutive_successes stays at the env default 50 (training cap).
#   - wm_strategy_pool MUST NOT contain `value_imag_h3` or `aux_low`.
#   - PPO / reward / obs / WM-architecture untouched.
#   - All P2.1 stability patches active (mastery, single_shrink, freeze).
#
# Population: 6 policies on GPUs 0..5 (training); CF runner on GPUs 4,5.
#
# Common overrides:
#   BASE_SEED                     - default 42.
#   DEXPBT_MAX_ACTIVE             - default 2.
#   DEXPBT_INTER_LAUNCH_DELAY_S   - default 90.
#   DEXPBT_STARTUP_TIMEOUT_S      - default 900.
#   DEXPBT_STARTUP_MAX_RESTARTS   - default 1.
#   FORMAL_MAX_ITER               - default 22889 (~3B at 8192 envs *16 horizon).
#   FORMAL_TARGET_FRAMES          - documentation only; default 3000000000.
#   ROLLBACK_COOLDOWN_FRAMES      - default 200000000.
#   WM_STRATEGY_POOL              - default `none,aux_mid,value_imag_h1`.
#                                   MUST NOT include value_imag_h3 / aux_low.
#   CF_NUM_ENVS                   - default 4096; lower to 2048 on OOM.
#   CF_MAX_PARALLEL_CHILDREN      - default 2; lower to 1 on OOM.
#   CF_CUDA_VISIBLE_DEVICES       - default 4,5 (shared with policies 4,5).
#   ENABLE_CF_RUNNER              - default 1 (set 0 to disable CF).
#
# Usage:
#   nohup bash start_regrasping_bayesian_no_h3_p2_1_3b.sh \
#     > nohup_track_c_3b.log 2>&1 &
#
# Reporting (after run completes):
#   python results/generate_smoke_report_p2.py "$RUN_DIR" \
#     --compare-to <fixed_h1_5B_run_dir> --compare-frame-range 0:3000000000

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNTIME_ROOT="/root/autodl-tmp/wmdexpbt/runtime_root"
RESULTS_ROOT="/root/autodl-tmp/wmdexpbt/results"
DATE=$(date +%Y%m%d_%H%M%S)
RUN_DIR="${RESULTS_ROOT}/regrasping_bayesian_no_h3_p2_1_3b_${DATE}"
LOG="${RUN_DIR}/launch_master.log"
CF_LOG="${RUN_DIR}/cf_runner.log"
PIDFILE="${RUN_DIR}/nohup.pid"
CF_PIDFILE="${RUN_DIR}/cf_runner.pid"
WS="${RUN_DIR}/dexpbt_8gpu_ws"

BASE_SEED="${BASE_SEED:-42}"
NUM_POLICIES="${NUM_POLICIES:-6}"
DEXPBT_GPUS="${DEXPBT_GPUS:-0,1,2,3,4,5}"
FORMAL_MAX_ITER="${FORMAL_MAX_ITER:-22889}"
FORMAL_PBT_START_AFTER="${FORMAL_PBT_START_AFTER:-200000000}"
FORMAL_PBT_INITIAL_DELAY="${FORMAL_PBT_INITIAL_DELAY:-50000000}"
FORMAL_TARGET_FRAMES="${FORMAL_TARGET_FRAMES:-3000000000}"
ROLLBACK_COOLDOWN_FRAMES="${ROLLBACK_COOLDOWN_FRAMES:-200000000}"
CROSSCOPY_TOLERANCE_MODE="${CROSSCOPY_TOLERANCE_MODE:-last_mastered}"
PBT_MIN_FRAMES_AS_SOURCE_AFTER_RESTART="${PBT_MIN_FRAMES_AS_SOURCE_AFTER_RESTART:-100000000}"
PBT_MIN_FRAMES_BETWEEN_RESTARTS="${PBT_MIN_FRAMES_BETWEEN_RESTARTS:-100000000}"

# Track C: explicit Bayesian strategy pool. Constraint 1 hard-checks
# below ensure value_imag_h3 and aux_low are not present.
WM_STRATEGY_POOL="${WM_STRATEGY_POOL:-none,aux_mid,value_imag_h1}"

# Constraint 3: CF runner overridable knobs. Defaults match the proven
# CB-WM smoke (4096 envs / 2 children); on OOM relaunch with
#   CF_NUM_ENVS=2048 CF_MAX_PARALLEL_CHILDREN=1 ./<this script>
CF_NUM_ENVS="${CF_NUM_ENVS:-4096}"
CF_MAX_PARALLEL_CHILDREN="${CF_MAX_PARALLEL_CHILDREN:-2}"
CF_CUDA_VISIBLE_DEVICES="${CF_CUDA_VISIBLE_DEVICES:-4,5}"
ENABLE_CF_RUNNER="${ENABLE_CF_RUNNER:-1}"

export DEXPBT_MAX_ACTIVE="${DEXPBT_MAX_ACTIVE:-2}"
export DEXPBT_INTER_LAUNCH_DELAY_S="${DEXPBT_INTER_LAUNCH_DELAY_S:-90}"
export DEXPBT_STARTUP_TIMEOUT_S="${DEXPBT_STARTUP_TIMEOUT_S:-900}"
export DEXPBT_STARTUP_MAX_RESTARTS="${DEXPBT_STARTUP_MAX_RESTARTS:-1}"
export NUM_POLICIES
export DEXPBT_GPUS

# -------- Constraint 1: pool hard-check (FATAL exit) --------
if [[ "${WM_STRATEGY_POOL}" == *value_imag_h3* ]]; then
  echo "[FATAL] WM_STRATEGY_POOL=${WM_STRATEGY_POOL} contains value_imag_h3" >&2
  echo "        Track C must permanently exclude value_imag_h3." >&2
  exit 1
fi
if [[ "${WM_STRATEGY_POOL}" == *aux_low* ]]; then
  echo "[FATAL] WM_STRATEGY_POOL=${WM_STRATEGY_POOL} contains aux_low" >&2
  echo "        Track C strategy pool must be {none, aux_mid, value_imag_h1}." >&2
  exit 1
fi
if [[ -z "${WM_STRATEGY_POOL}" ]]; then
  echo "[FATAL] WM_STRATEGY_POOL is empty" >&2
  exit 1
fi

mkdir -p "$RUN_DIR"
echo "$RUN_DIR" > "${RESULTS_ROOT}/latest_run.txt"
echo "$RUN_DIR" > "${RESULTS_ROOT}/latest_track_c_run.txt"
echo "$RUN_DIR" > "${RUN_DIR}/RUN_DIR.txt"

# Emit human-readable RUN_CONFIG.txt for log archaeology (kept consistent
# with fixed_h1 5B), and a structured RUN_CONFIG.json that the report
# generator can ingest for the Track C pool hard-check (constraint 1).
cat > "${RUN_DIR}/RUN_CONFIG.txt" <<EOF
launcher=start_regrasping_bayesian_no_h3_p2_1_3b.sh
profile=track_c_bayesian_no_h3_3b
target_frames=${FORMAL_TARGET_FRAMES}
max_iterations=${FORMAL_MAX_ITER}
num_policies=${NUM_POLICIES}
gpus=${DEXPBT_GPUS}
base_seed=${BASE_SEED}
wm_strategy_mode=bayesian
wm_strategy_pool=${WM_STRATEGY_POOL}
cf_runner_enabled=$([[ "${ENABLE_CF_RUNNER}" == "1" ]] && echo true || echo false)
cf_num_envs=${CF_NUM_ENVS}
cf_max_parallel_children=${CF_MAX_PARALLEL_CHILDREN}
cf_cuda_visible_devices=${CF_CUDA_VISIBLE_DEVICES}
cf_delta_metric=nsucc
crosscopy_tolerance_mode=${CROSSCOPY_TOLERANCE_MODE}
recovery_tolerance_mode=last_mastered
rollback_cooldown_frames=${ROLLBACK_COOLDOWN_FRAMES}
pbt_min_frames_as_source_after_restart=${PBT_MIN_FRAMES_AS_SOURCE_AFTER_RESTART}
pbt_min_frames_between_restarts=${PBT_MIN_FRAMES_BETWEEN_RESTARTS}
max_active=${DEXPBT_MAX_ACTIVE}
inter_launch_delay_s=${DEXPBT_INTER_LAUNCH_DELAY_S}
startup_timeout_s=${DEXPBT_STARTUP_TIMEOUT_S}
startup_max_restarts=${DEXPBT_STARTUP_MAX_RESTARTS}
max_consecutive_successes=50
EOF

# JSON form — wm_strategy_pool emitted as a real JSON list so the report
# generator's `_pool_hard_check` can iterate it.
python3 - "$RUN_DIR" "$WM_STRATEGY_POOL" "$ENABLE_CF_RUNNER" "$CF_NUM_ENVS" \
  "$CF_MAX_PARALLEL_CHILDREN" "$CF_CUDA_VISIBLE_DEVICES" \
  "$FORMAL_TARGET_FRAMES" "$FORMAL_MAX_ITER" "$NUM_POLICIES" \
  "$BASE_SEED" "$ROLLBACK_COOLDOWN_FRAMES" <<'PY'
import json, sys
(run_dir, pool, en_cf, cf_envs, cf_par, cf_gpus, target, max_iter,
 num_pol, seed, rollback) = sys.argv[1:12]
cfg = {
    "launcher": "start_regrasping_bayesian_no_h3_p2_1_3b.sh",
    "profile": "track_c_bayesian_no_h3_3b",
    "target_frames": int(target),
    "max_iterations": int(max_iter),
    "num_policies": int(num_pol),
    "base_seed": int(seed),
    "wm_strategy_mode": "bayesian",
    "wm_strategy_pool": [s.strip() for s in pool.split(",") if s.strip()],
    "cf_runner_enabled": (en_cf == "1"),
    "cf_num_envs": int(cf_envs),
    "cf_max_parallel_children": int(cf_par),
    "cf_cuda_visible_devices": cf_gpus,
    "cf_delta_metric": "nsucc",
    "rollback_cooldown_frames": int(rollback),
    "max_consecutive_successes": 50,
}
with open(f"{run_dir}/RUN_CONFIG.json", "w") as f:
    json.dump(cfg, f, indent=2)
PY

cat > "${RUN_DIR}/SMOKE_CONFIG.txt" <<EOF
smoke_run=false
patch=track_c_bayesian_no_h3
target_frames=${FORMAL_TARGET_FRAMES}
max_iterations=${FORMAL_MAX_ITER}
curriculum_mode=mastery
pbt_single_shrink_per_window=true
pbt_early_collapse_detect=true
pbt_recovery_tolerance_mode=last_mastered
pbt_recovery_bypass_elimination_burnin=true
note=Track C bayesian-no-h3 3B; pool=${WM_STRATEGY_POOL}; CF on
EOF

# shellcheck source=/dev/null
source "${RUNTIME_ROOT}/scripts/env_dexpbt.sh"
export TORCHDYNAMO_DISABLE=1
export PYTHONUNBUFFERED=1
export OMNI_KIT_TELEMETRY=0
export CARB_CRASH_REPORTER=0

cd "${RUNTIME_ROOT}/src/dexpbt_isaaclab"

echo "Starting Track C bayesian-no-h3 P2.1 3B:"
echo "  RUN_DIR=$RUN_DIR"
echo "  pool=${WM_STRATEGY_POOL} (h3 EXCLUDED, aux_low EXCLUDED)"
echo "  CF: enabled=${ENABLE_CF_RUNNER} num_envs=${CF_NUM_ENVS} "
echo "      max_parallel_children=${CF_MAX_PARALLEL_CHILDREN} "
echo "      cf_gpus=${CF_CUDA_VISIBLE_DEVICES} cf_delta_metric=nsucc"
echo "  num_policies=$NUM_POLICIES gpus=$DEXPBT_GPUS base_seed=$BASE_SEED"
echo "  staggered: max_active=$DEXPBT_MAX_ACTIVE delay=${DEXPBT_INTER_LAUNCH_DELAY_S}s"
echo "  rollback_cooldown=${ROLLBACK_COOLDOWN_FRAMES} crosscopy_tol=${CROSSCOPY_TOLERANCE_MODE}"
echo "  PBT min frames between restarts: ${PBT_MIN_FRAMES_BETWEEN_RESTARTS}"

if [[ "${ENABLE_CF_RUNNER}" == "1" ]]; then
  nohup python -u -m dexpbt_isaaclab.scripts.run_counterfactual_block \
    --workspace "$WS" \
    --poll_interval 30 \
    --heartbeat_interval 30 \
    --cf_num_envs "${CF_NUM_ENVS}" \
    --cf_max_parallel_children "${CF_MAX_PARALLEL_CHILDREN}" \
    --cf_cuda_visible_devices "${CF_CUDA_VISIBLE_DEVICES}" \
    --cf_horizon 16 \
    >"$CF_LOG" 2>&1 &
  echo $! > "$CF_PIDFILE"
  echo "  cf_runner pid=$(cat "$CF_PIDFILE") log=$CF_LOG"
else
  echo "  cf_runner: SKIPPED (ENABLE_CF_RUNNER=0)"
fi

nohup bash "${RUNTIME_ROOT}/scripts/launch_8gpu_dexpbt.sh" \
  "$RUN_DIR" \
  --task DexPBT-Regrasping-KukaAllegro-Direct-v0 \
  --num-envs 8192 \
  --max-iterations "$FORMAL_MAX_ITER" \
  --pbt-interval-steps 20000000 \
  --pbt-start-after "$FORMAL_PBT_START_AFTER" \
  --pbt-initial-delay "$FORMAL_PBT_INITIAL_DELAY" \
  --extra \
  --seed "$BASE_SEED" \
  --enable_wm \
  --wm_strategy value_imag_h1 \
  --wm_ensemble_size 3 \
  --wm_lr 1e-3 \
  --wm_update_every_n_epochs 2 \
  --wm_batch_size 512 \
  --wm_strategy_mode bayesian \
  --wm_strategy_pool "$WM_STRATEGY_POOL" \
  --enable_counterfactual \
  --cf_interval_gens 5 \
  --cf_steps 5000000 \
  --cf_num_envs "$CF_NUM_ENVS" \
  --cf_max_parallel_children "$CF_MAX_PARALLEL_CHILDREN" \
  --cf_cuda_visible_devices "$CF_CUDA_VISIBLE_DEVICES" \
  --cf_horizon 16 \
  --cf_train_frames 20000000 \
  --cf_reset_epoch_counter \
  --cf_min_eval_s 30 \
  --cf_delta_mode learning_improvement \
  --cf_low_signal_action skip \
  --cf_low_signal_weight 0.25 \
  --cf_require_nsucc_gt_zero \
  --cf_source_min_nsucc_mean 3.0 \
  --cf_source_min_nsucc_max 20 \
  --cf_source_min_ranking_score 0.0 \
  --cf_delta_metric nsucc \
  --pbt_ranking_metric success_aware_r_meta \
  --pbt_nsucc_zero_cap 0.05 \
  --pbt_nsucc_low_threshold 1.0 \
  --pbt_nsucc_low_cap 0.05 \
  --pbt_min_frames_as_source_after_restart "$PBT_MIN_FRAMES_AS_SOURCE_AFTER_RESTART" \
  --pbt_min_frames_between_restarts "$PBT_MIN_FRAMES_BETWEEN_RESTARTS" \
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
  --pbt_curriculum_freeze_on_recovery_frames 200000000 \
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
  --pbt_usable_live_min_mean 2.0 \
  --pbt_usable_live_min_max 20 \
  --pbt_usable_live_max_drop_ratio 0.5 \
  >"$LOG" 2>&1 &

echo $! > "$PIDFILE"
echo "launch_master pid=$(cat "$PIDFILE") log=$LOG"
echo "Report (after run):"
echo "  python ${RESULTS_ROOT}/generate_smoke_report_p2.py ${RUN_DIR} \\"
echo "    --compare-to <fixed_h1_5B_run_dir> \\"
echo "    --compare-frame-range 0:3000000000"
