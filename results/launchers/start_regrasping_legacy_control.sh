#!/usr/bin/env bash
# Legacy curriculum ablation control (CB-WM, no P1 mastery/stable-source).
# Usage: bash start_regrasping_legacy_control.sh

set -euo pipefail
RUNTIME_ROOT="/root/autodl-tmp/wmdexpbt/runtime_root"
RESULTS_ROOT="/root/autodl-tmp/wmdexpbt/results"
RUNS_ROOT="${RESULTS_ROOT}/runs"
DATE=$(date +%Y%m%d_%H%M%S)
RUN_DIR="${RUNS_ROOT}/regrasping_legacy_control_${DATE}"
LOG="${RUN_DIR}/launch_master.log"
CF_LOG="${RUN_DIR}/cf_runner.log"
PIDFILE="${RUN_DIR}/nohup.pid"
CF_PIDFILE="${RUN_DIR}/cf_runner.pid"
WS="${RUN_DIR}/dexpbt_8gpu_ws"

mkdir -p "$RUN_DIR"
echo "$RUN_DIR" > "${RESULTS_ROOT}/pointers/latest_run.txt"

# shellcheck source=/dev/null
source "${RUNTIME_ROOT}/scripts/env_dexpbt.sh"
export TORCHDYNAMO_DISABLE=1 PYTHONUNBUFFERED=1
cd "${RUNTIME_ROOT}/src/dexpbt_isaaclab"

nohup python -u -m dexpbt_isaaclab.scripts.run_counterfactual_block \
  --workspace "$WS" --poll_interval 30 --heartbeat_interval 30 \
  --cf_num_envs 4096 --cf_max_parallel_children 2 \
  --cf_cuda_visible_devices 4,5 --cf_horizon 16 >"$CF_LOG" 2>&1 &
echo $! > "$CF_PIDFILE"

nohup bash "${RUNTIME_ROOT}/scripts/launch_8gpu_dexpbt.sh" "$RUN_DIR" \
  --task DexPBT-Regrasping-KukaAllegro-Direct-v0 --num-envs 8192 \
  --max-iterations 38147 --pbt-interval-steps 20000000 \
  --pbt-start-after 200000000 --pbt-initial-delay 50000000 --extra \
  --enable_wm --wm_strategy value_imag_h1 --wm_ensemble_size 3 --wm_lr 1e-3 \
  --wm_update_every_n_epochs 2 --wm_batch_size 512 \
  --enable_counterfactual --cf_interval_gens 5 --cf_steps 5000000 \
  --cf_num_envs 4096 --cf_max_parallel_children 2 --cf_cuda_visible_devices 4,5 --cf_horizon 16 \
  --pbt_ranking_metric success_aware_r_meta --pbt_nsucc_zero_cap 0.05 \
  --pbt_min_frames_as_source_after_restart 50000000 --pbt_min_frames_between_restarts 50000000 \
  --pbt_min_nsucc_mean_as_source 1.0 --pbt_min_nsucc_max_as_source 5 \
  --cf_train_frames 20000000 --cf_reset_epoch_counter --cf_min_eval_s 30 \
  --cf_delta_mode learning_improvement --cf_low_signal_action skip \
  --pbt_disable_self_restart --pbt_source_selection_mode tiered \
  --pbt_relaxed_nsucc_mean_as_source 0.5 --pbt_relaxed_nsucc_max_as_source 2 \
  --pbt_enable_best_nsucc_recovery \
  --pbt_curriculum_freeze_frames 100000000 --pbt_curriculum_freeze_on_recovery_frames 200000000 \
  --cf_source_min_nsucc_mean 1.5 --cf_source_min_nsucc_max 5 \
  --wm_strategy_mode bayesian --fixed_wm_strategy value_imag_h3 \
  --curriculum_mode legacy --no-pbt_source_require_stable_mean \
  >"$LOG" 2>&1 &

echo $! > "$PIDFILE"
echo "Started legacy control (no P1): $RUN_DIR"
