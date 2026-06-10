#!/usr/bin/env bash
# 100M-frame smoke run for Patches A-J (NOT the full 5B run).
#
# Same --extra patch flags as start_regrasping_cbwm.sh. Only structural
# overrides for smoke viability within 100M frames:
#   max-iterations=764          (~100M frames @ 8192 envs * horizon 16)
#   pbt-start-after=40000000    (default 200M would never fire PBT in smoke)
#   pbt-initial-delay=20000000  (default 50M; lowered so PBT ticks fit in 100M)
#
# Usage:
#   bash start_regrasping_cbwm_smoke_100m.sh
# After completion (~100M frames):
#   python generate_smoke_report_cbwm.py "$RUN_DIR"

set -euo pipefail
RUNTIME_ROOT="/root/autodl-tmp/wmdexpbt/runtime_root"
RESULTS_ROOT="/root/autodl-tmp/wmdexpbt/results"
RUNS_ROOT="${RESULTS_ROOT}/runs"
DATE=$(date +%Y%m%d_%H%M%S)
RUN_DIR="${RUNS_ROOT}/regrasping_cbwm_smoke_${DATE}"
LOG="${RUN_DIR}/launch_master.log"
CF_LOG="${RUN_DIR}/cf_runner.log"
PIDFILE="${RUN_DIR}/nohup.pid"
CF_PIDFILE="${RUN_DIR}/cf_runner.pid"
WS="${RUN_DIR}/dexpbt_8gpu_ws"

# 8192 envs * horizon 16 = 131072 frames/epoch; 764 epochs ≈ 100.1M frames
SMOKE_MAX_ITER=764
SMOKE_PBT_START_AFTER=40000000
SMOKE_PBT_INITIAL_DELAY=20000000
SMOKE_TARGET_FRAMES=100000000

mkdir -p "$RUN_DIR"
echo "$RUN_DIR" > "${RESULTS_ROOT}/pointers/latest_smoke_run.txt"
echo "$RUN_DIR" > "${RESULTS_ROOT}/pointers/latest_run.txt"
echo "$RUN_DIR" > "${RUN_DIR}/RUN_DIR.txt"

cat > "${RUN_DIR}/SMOKE_CONFIG.txt" <<EOF
smoke_run=true
target_frames=${SMOKE_TARGET_FRAMES}
max_iterations=${SMOKE_MAX_ITER}
pbt_start_after=${SMOKE_PBT_START_AFTER}
pbt_initial_delay=${SMOKE_PBT_INITIAL_DELAY}
patch_flags=identical_to_start_regrasping_cbwm.sh
note=Structural smoke overrides only; all A-J patch defaults preserved in --extra block.
EOF

# shellcheck source=/dev/null
source "${RUNTIME_ROOT}/scripts/env_dexpbt.sh"
mkdir -p "$(dirname /root/.local/share/ov/data)" /root/.cache/ov 2>/dev/null || true
ln -sfn "${RUNTIME_ROOT}/cache/ov/data" /root/.local/share/ov/data 2>/dev/null || true
ln -sfn "${RUNTIME_ROOT}/cache/ov_client" /root/.cache/ov 2>/dev/null || true

export TORCHDYNAMO_DISABLE=1
export PYTHONUNBUFFERED=1
export OMNI_KIT_TELEMETRY=0
export CARB_CRASH_REPORTER=0

cd "${RUNTIME_ROOT}/src/dexpbt_isaaclab"

echo "Starting 100M-frame smoke: DexPBT Regrasping + CB-WM (Patches A-J)"
echo "  RUN_DIR=$RUN_DIR"
echo "  workspace=$WS"
echo "  max_iterations=$SMOKE_MAX_ITER (~${SMOKE_TARGET_FRAMES} frames)"
echo "  pbt_start_after=$SMOKE_PBT_START_AFTER pbt_initial_delay=$SMOKE_PBT_INITIAL_DELAY"

# 1) CF runner
nohup python -u -m dexpbt_isaaclab.scripts.run_counterfactual_block \
  --workspace "$WS" \
  --poll_interval 30 \
  --heartbeat_interval 30 \
  --cf_num_envs 4096 \
  --cf_max_parallel_children 2 \
  --cf_cuda_visible_devices 4,5 \
  --cf_horizon 16 \
  >"$CF_LOG" 2>&1 &
echo $! > "$CF_PIDFILE"
echo "CF runner pid=$(cat "$CF_PIDFILE") log=$CF_LOG"

# 2) Main 8-policy PBT (patches A-J defaults, smoke frame cap)
nohup bash "${RUNTIME_ROOT}/scripts/launch_8gpu_dexpbt.sh" \
  "$RUN_DIR" \
  --task DexPBT-Regrasping-KukaAllegro-Direct-v0 \
  --num-envs 8192 \
  --max-iterations "$SMOKE_MAX_ITER" \
  --pbt-interval-steps 20000000 \
  --pbt-start-after "$SMOKE_PBT_START_AFTER" \
  --pbt-initial-delay "$SMOKE_PBT_INITIAL_DELAY" \
  --extra \
  --enable_wm \
  --wm_strategy value_imag_h1 \
  --wm_ensemble_size 3 \
  --wm_lr 1e-3 \
  --wm_update_every_n_epochs 2 \
  --wm_batch_size 512 \
  --enable_counterfactual \
  --cf_interval_gens 5 \
  --cf_steps 5000000 \
  --cf_num_envs 4096 \
  --cf_max_parallel_children 2 \
  --cf_cuda_visible_devices 4,5 \
  --cf_horizon 16 \
  --pbt_ranking_metric success_aware_r_meta \
  --pbt_nsucc_zero_cap 0.05 \
  --pbt_nsucc_low_threshold 1.0 \
  --pbt_nsucc_low_cap 0.05 \
  --pbt_min_frames_as_source_after_restart 50000000 \
  --pbt_min_frames_between_restarts 50000000 \
  --pbt_min_nsucc_mean_as_source 1.0 \
  --pbt_min_nsucc_max_as_source 5 \
  --cf_train_frames 20000000 \
  --cf_reset_epoch_counter \
  --cf_min_eval_s 30 \
  --cf_delta_mode learning_improvement \
  --cf_low_signal_action skip \
  --cf_low_signal_weight 0.25 \
  --cf_require_nsucc_gt_zero \
  --pbt_allow_low_nsucc_source_fallback \
  --pbt_disable_self_restart \
  --pbt_max_self_restarts_per_policy 3 \
  --pbt_min_frames_between_self_restarts 200000000 \
  --pbt_source_selection_mode tiered \
  --pbt_relaxed_nsucc_mean_as_source 0.5 \
  --pbt_relaxed_nsucc_max_as_source 2 \
  --pbt_enable_best_nsucc_recovery \
  --pbt_recovery_min_nsucc_mean 2.0 \
  --pbt_recovery_min_nsucc_max 8 \
  --pbt_recovery_cooldown_frames 400000000 \
  --pbt_curriculum_freeze_frames 100000000 \
  --pbt_curriculum_freeze_on_recovery_frames 200000000 \
  --cf_source_min_nsucc_mean 1.5 \
  --cf_source_min_nsucc_max 5 \
  --cf_source_min_ranking_score 0.0 \
  --cf_delta_metric nsucc \
  --wm_strategy_mode bayesian \
  --fixed_wm_strategy value_imag_h3 \
  >"$LOG" 2>&1 &

echo $! > "$PIDFILE"
echo "launch_master pid=$(cat "$PIDFILE") log=$LOG"
echo ""
echo "Monitor: bash ${RESULTS_ROOT}/monitor_training.sh ${RUN_DIR}"
echo "Report (after ~100M): python ${RESULTS_ROOT}/generate_smoke_report_cbwm.py ${RUN_DIR}"
