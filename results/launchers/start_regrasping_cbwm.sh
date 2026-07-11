#!/usr/bin/env bash
# 6 卡 DexPBT Regrasping + Counterfactual Bayesian WM-DexPBT（nohup）
# 用法:
#   bash start_regrasping_cbwm.sh
#   RECOVER_CHECKPOINT=/path/to/000028.pth RECOVER_POLICY_IDX=3 bash start_regrasping_cbwm.sh
# 结果: /root/autodl-tmp/wmdexpbt/results/regrasping_cbwm_<timestamp>/
#
# 监控:
#   bash /root/autodl-tmp/wmdexpbt/results/monitor_training.sh "$RUN_DIR"
#   tail -f "$RUN_DIR/launch_master.log"
#   tail -f "$RUN_DIR/cf_runner.log"
#   cat "$RUN_DIR/dexpbt_8gpu_ws/counterfactual/runner_heartbeat.json"

set -euo pipefail
RUNTIME_ROOT="/root/autodl-tmp/wmdexpbt/runtime_root"
RESULTS_ROOT="/root/autodl-tmp/wmdexpbt/results"
RUNS_ROOT="${RESULTS_ROOT}/runs"
DATE=$(date +%Y%m%d_%H%M%S)
RUN_DIR="${RUNS_ROOT}/regrasping_cbwm_${DATE}"
LOG="${RUN_DIR}/launch_master.log"
CF_LOG="${RUN_DIR}/cf_runner.log"
PIDFILE="${RUN_DIR}/nohup.pid"
CF_PIDFILE="${RUN_DIR}/cf_runner.pid"
WS="${RUN_DIR}/dexpbt_8gpu_ws"

# Optional warm-start one policy from a prior peak checkpoint.
RECOVER_CHECKPOINT="${RECOVER_CHECKPOINT:-}"
RECOVER_POLICY_IDX="${RECOVER_POLICY_IDX:-3}"

mkdir -p "$RUN_DIR"
echo "$RUN_DIR" > "${RESULTS_ROOT}/pointers/latest_run.txt"
echo "$RUN_DIR" > "${RUN_DIR}/RUN_DIR.txt"

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

echo "Starting 6-GPU DexPBT Regrasping + CB-WM (Ours, patches A-J)"
echo "  RUN_DIR=$RUN_DIR"
echo "  workspace=$WS"
if [[ -n "$RECOVER_CHECKPOINT" ]]; then
  echo "  recover_checkpoint=$RECOVER_CHECKPOINT policy_idx=$RECOVER_POLICY_IDX"
fi

# 1) Counterfactual block runner (policy0 writes requests; runner executes children)
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

RECOVER_ARGS=()
if [[ -n "$RECOVER_CHECKPOINT" ]]; then
  RECOVER_ARGS+=(--recover-checkpoint "$RECOVER_CHECKPOINT" --recover-policy-idx "$RECOVER_POLICY_IDX")
fi

# 2) Main 6-policy PBT training (GPUs 0-5)
nohup bash "${RUNTIME_ROOT}/scripts/launch_8gpu_dexpbt.sh" \
  "$RUN_DIR" \
  --task DexPBT-Regrasping-KukaAllegro-Direct-v0 \
  --num-envs 8192 \
  --max-iterations 38147 \
  --pbt-interval-steps 20000000 \
  --pbt-start-after 200000000 \
  --pbt-initial-delay 50000000 \
  "${RECOVER_ARGS[@]}" \
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
