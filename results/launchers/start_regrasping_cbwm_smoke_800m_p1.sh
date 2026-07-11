#!/usr/bin/env bash
# 800M-frame smoke run for P1 curriculum/stability patch (CB-WM + mastery).
#
# Usage:
#   bash start_regrasping_cbwm_smoke_800m_p1.sh
# After completion (~800M frames):
#   python generate_smoke_report_p1.py "$RUN_DIR"
#
# Structural overrides vs formal run:
#   max-iterations=6112          (~800M frames @ 8192 envs * horizon 16)
#   pbt-start-after=200000000    (match formal run)
#   pbt-initial-delay=50000000   (match formal run)

set -euo pipefail
RUNTIME_ROOT="/root/autodl-tmp/wmdexpbt/runtime_root"
RESULTS_ROOT="/root/autodl-tmp/wmdexpbt/results"
RUNS_ROOT="${RESULTS_ROOT}/runs"
DATE=$(date +%Y%m%d_%H%M%S)
RUN_DIR="${RUNS_ROOT}/regrasping_cbwm_smoke_800m_p1_${DATE}"
LOG="${RUN_DIR}/launch_master.log"
CF_LOG="${RUN_DIR}/cf_runner.log"
PIDFILE="${RUN_DIR}/nohup.pid"
CF_PIDFILE="${RUN_DIR}/cf_runner.pid"
WS="${RUN_DIR}/dexpbt_8gpu_ws"

# 8192 envs * horizon 16 = 131072 frames/epoch; 6112 epochs ≈ 800.7M frames
SMOKE_MAX_ITER=6112
SMOKE_PBT_START_AFTER=200000000
SMOKE_PBT_INITIAL_DELAY=50000000
SMOKE_TARGET_FRAMES=800000000

mkdir -p "$RUN_DIR"
echo "$RUN_DIR" > "${RESULTS_ROOT}/pointers/latest_smoke_run.txt"
echo "$RUN_DIR" > "${RESULTS_ROOT}/pointers/latest_run.txt"
echo "$RUN_DIR" > "${RUN_DIR}/RUN_DIR.txt"

cat > "${RUN_DIR}/SMOKE_CONFIG.txt" <<EOF
smoke_run=true
patch=P1_curriculum_stability
target_frames=${SMOKE_TARGET_FRAMES}
max_iterations=${SMOKE_MAX_ITER}
pbt_start_after=${SMOKE_PBT_START_AFTER}
pbt_initial_delay=${SMOKE_PBT_INITIAL_DELAY}
curriculum_mode=mastery
mastery_nsucc_mean=4.0
pbt_source_require_stable_mean=true
pbt_allow_best_nsucc_archive_recovery=false
note=P1 smoke: mastery curriculum + stable source + collapse freeze
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

echo "Starting 800M-frame P1 smoke: DexPBT Regrasping + CB-WM"
echo "  RUN_DIR=$RUN_DIR"
echo "  workspace=$WS"
echo "  max_iterations=$SMOKE_MAX_ITER (~${SMOKE_TARGET_FRAMES} frames)"

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

# 2) Main 8-policy PBT (patches A-J + P1)
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
  --no-pbt_enable_best_nsucc_recovery \
  --no-pbt_allow_best_nsucc_archive_recovery \
  --pbt_recovery_min_nsucc_mean 2.0 \
  --pbt_recovery_min_nsucc_max 8 \
  --pbt_recovery_cooldown_frames 400000000 \
  --pbt_curriculum_freeze_frames 100000000 \
  --pbt_curriculum_freeze_on_recovery_frames 200000000 \
  --cf_source_min_nsucc_mean 3.0 \
  --cf_source_min_nsucc_max 20 \
  --cf_source_min_ranking_score 0.0 \
  --cf_delta_metric nsucc \
  --wm_strategy_mode bayesian \
  --fixed_wm_strategy value_imag_h3 \
  --curriculum_mode mastery \
  --pbt_source_require_stable_mean \
  --mastery_nsucc_mean 4.0 \
  --mastery_nsucc_max 20.0 \
  --mastery_consecutive_windows 2 \
  --curriculum_shrink_factor 0.95 \
  --curriculum_collapse_detect \
  --collapse_drop_ratio 0.5 \
  --collapse_patience 2 \
  --collapse_freeze_env_frames 200000000 \
  --stable_source_min_mean 3.0 \
  --stable_source_min_max 20.0 \
  --stable_source_max_drop_ratio 0.3 \
  --stable_source_drop_windows 2 \
  >"$LOG" 2>&1 &

echo $! > "$PIDFILE"
echo "launch_master pid=$(cat "$PIDFILE") log=$LOG"
echo ""
echo "Monitor: bash ${RESULTS_ROOT}/monitor_training.sh ${RUN_DIR}"
echo "Report (after ~800M): python ${RESULTS_ROOT}/generate_smoke_report_p1.py ${RUN_DIR}"
