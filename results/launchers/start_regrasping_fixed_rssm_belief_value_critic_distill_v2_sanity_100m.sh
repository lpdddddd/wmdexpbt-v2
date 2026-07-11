#!/usr/bin/env bash
# Phase A4 P1 (v2 coupling) -- 100M single-policy ENGINEERING SANITY.
#
# Purpose: prove the v2 wiring (shared-to-aux-critic feature path,
# rollout zscore, teacher_corr gate, ramp) runs cleanly end-to-end.
# This launcher does NOT judge nsucc effect. Effect judgment lives at
# the 800M 6-policy smoke (start_regrasping_fixed_rssm_belief_value_
# critic_distill_v2_smoke_800m.sh).
#
# Engineering acceptance (recorded in TB):
#   * no crash / NaN / OOM / Traceback / fail-fast RuntimeError,
#   * seq_aux_critic_mlp grad norm > 0 at least once after warmup,
#   * seq_policy_aux/shared_to_aux_critic_feature_captured == 1
#     every minibatch after warmup,
#   * teacher cache built every epoch, zero generation mismatch,
#   * seq_policy_aux/teacher_value_norm_enabled == 1 throughout,
#   * post-norm targets running mean ~ 0, std ~ 1,
#   * seq_policy_aux/teacher_return_corr finite throughout,
#   * gate observed both blocking (early) and passing (later) at least
#     once each,
#   * seq_policy_aux/effective_weight tracks warmup -> ramp schedule,
#   * FPS regression vs the 200M no-coupling control < 10%.
#
# nsucc / r_meta / success_tolerance are explicitly NOT acceptance
# criteria at this stage.

set -euo pipefail

WMDEXPBT_ROOT="/root/autodl-tmp/wmdexpbt"
RUNTIME_ROOT="${WMDEXPBT_ROOT}/runtime_root"
RESULTS_ROOT="${WMDEXPBT_ROOT}/results"
DATE=$(date +%Y%m%d_%H%M%S)
RUN_DIR="${RUNS_ROOT}/regrasping_fixed_rssm_belief_value_critic_distill_v2_sanity_100m_${DATE}"
LOG="${RUN_DIR}/train.log"
WORKSPACE="${RUN_DIR}/dexpbt_ws"

GPU="${GPU:-0}"
NUM_ENVS="${NUM_ENVS:-8192}"
# 763 iters * 8192 envs * 16 horizon ~= 100M frames
MAX_ITER="${MAX_ITER:-763}"
SEED="${SEED:-42}"
SEQ_WM_SEQ_LEN="${SEQ_WM_SEQ_LEN:-16}"
SEQ_WM_BATCH="${SEQ_WM_BATCH:-1024}"
SEQ_WM_LAMBDA_TOTAL="${SEQ_WM_LAMBDA_TOTAL:-0.05}"
SEQ_WM_LAMBDA_VALUE="${SEQ_WM_LAMBDA_VALUE:-0.1}"
SEQ_WM_VALUE_DETACH_FLAG=""
case "${SEQ_WM_VALUE_DETACH:-false}" in
  1|true|TRUE|True) SEQ_WM_VALUE_DETACH_FLAG="--seq_wm_value_detach_belief" ;;
  *)               SEQ_WM_VALUE_DETACH_FLAG="--no-seq_wm_value_detach_belief" ;;
esac

# v2 hyperparameters (per plan section 10.1).
SEQ_WM_POLICY_COUPLING_WEIGHT="${SEQ_WM_POLICY_COUPLING_WEIGHT:-0.005}"
SEQ_WM_POLICY_COUPLING_VALUE_WEIGHT="${SEQ_WM_POLICY_COUPLING_VALUE_WEIGHT:-1.0}"
SEQ_WM_POLICY_COUPLING_WARMUP_FRAMES="${SEQ_WM_POLICY_COUPLING_WARMUP_FRAMES:-50000000}"
SEQ_WM_POLICY_COUPLING_RAMP_FRAMES="${SEQ_WM_POLICY_COUPLING_RAMP_FRAMES:-50000000}"
SEQ_WM_TEACHER_CORR_THRESHOLD="${SEQ_WM_TEACHER_CORR_THRESHOLD:-0.2}"
SEQ_WM_AUX_CRITIC_SUBTOWER_HIDDEN="${SEQ_WM_AUX_CRITIC_SUBTOWER_HIDDEN:-256}"

mkdir -p "${RUN_DIR}"
echo "${RUN_DIR}" > "${RESULTS_ROOT}/pointers/latest_seq_wm_v2_sanity_run.txt"

cat > "${RUN_DIR}/RUN_CONFIG.txt" <<EOF
launcher=start_regrasping_fixed_rssm_belief_value_critic_distill_v2_sanity_100m.sh
profile=phase_a4_p1_v2_sanity_100m
target_frames=$(( NUM_ENVS * 16 * MAX_ITER ))
max_iterations=${MAX_ITER}
num_policies=1
gpus=${GPU}
base_seed=${SEED}
wm_strategy=rssm_belief_value
wm_strategy_mode=fixed
seq_wm_seq_len=${SEQ_WM_SEQ_LEN}
seq_wm_seq_batch_size=${SEQ_WM_BATCH}
seq_wm_lambda_total=${SEQ_WM_LAMBDA_TOTAL}
seq_wm_lambda_value=${SEQ_WM_LAMBDA_VALUE}
seq_wm_value_detach_belief=${SEQ_WM_VALUE_DETACH:-false}
seq_wm_policy_coupling=critic_distill_value_norm
seq_wm_distill_feature_source=shared_to_aux_critic
seq_wm_teacher_value_norm=rollout_zscore
seq_wm_policy_coupling_gate=teacher_corr
seq_wm_teacher_corr_threshold=${SEQ_WM_TEACHER_CORR_THRESHOLD}
seq_wm_policy_coupling_weight=${SEQ_WM_POLICY_COUPLING_WEIGHT}
seq_wm_policy_coupling_value_weight=${SEQ_WM_POLICY_COUPLING_VALUE_WEIGHT}
seq_wm_policy_coupling_warmup_frames=${SEQ_WM_POLICY_COUPLING_WARMUP_FRAMES}
seq_wm_policy_coupling_ramp_frames=${SEQ_WM_POLICY_COUPLING_RAMP_FRAMES}
seq_wm_aux_critic_subtower_hidden=${SEQ_WM_AUX_CRITIC_SUBTOWER_HIDDEN}
cf_runner_enabled=false
max_consecutive_successes=50
EOF

# shellcheck source=/dev/null
source "${RUNTIME_ROOT}/scripts/env_dexpbt.sh"

export CUDA_VISIBLE_DEVICES="${GPU}"
export TORCHDYNAMO_DISABLE=1
export PYTHONUNBUFFERED=1
export OMNI_KIT_TELEMETRY=0
export CARB_CRASH_REPORTER=0

cd "${RUNTIME_ROOT}/src/dexpbt_isaaclab"

echo "[seq_wm_v2_sanity] run dir=${RUN_DIR}"
echo "[seq_wm_v2_sanity] strategy=rssm_belief_value coupling=critic_distill_value_norm seed=${SEED} max_iter=${MAX_ITER}"
echo "[seq_wm_v2_sanity] weight=${SEQ_WM_POLICY_COUPLING_WEIGHT} warmup=${SEQ_WM_POLICY_COUPLING_WARMUP_FRAMES} ramp=${SEQ_WM_POLICY_COUPLING_RAMP_FRAMES} corr_threshold=${SEQ_WM_TEACHER_CORR_THRESHOLD}"

nohup python -u -m dexpbt_isaaclab.scripts.train \
  --task DexPBT-Regrasping-KukaAllegro-Direct-v0 \
  --train_root "${RUN_DIR}/rl_games" \
  --experiment_name "v2_sanity_100m_${DATE}" \
  --device=cuda:0 \
  --headless \
  --observer_mode=default \
  --workspace="${WORKSPACE}" \
  --num_envs="${NUM_ENVS}" \
  --max_iterations="${MAX_ITER}" \
  --seed="${SEED}" \
  --enable_wm \
  --wm_strategy=rssm_belief_value \
  --wm_strategy_mode=fixed \
  --fixed_wm_strategy=rssm_belief_value \
  --wm_strategy_pool=rssm_belief_value \
  --wm_load_mode=reset \
  --seq_wm_seq_len="${SEQ_WM_SEQ_LEN}" \
  --seq_wm_seq_batch_size="${SEQ_WM_BATCH}" \
  --seq_wm_lambda_total="${SEQ_WM_LAMBDA_TOTAL}" \
  --seq_wm_lambda_value="${SEQ_WM_LAMBDA_VALUE}" \
  ${SEQ_WM_VALUE_DETACH_FLAG} \
  --seq_wm_policy_coupling=critic_distill_value_norm \
  --seq_wm_distill_feature_source=shared_to_aux_critic \
  --seq_wm_teacher_value_norm=rollout_zscore \
  --seq_wm_policy_coupling_gate=teacher_corr \
  --seq_wm_teacher_corr_threshold="${SEQ_WM_TEACHER_CORR_THRESHOLD}" \
  --seq_wm_policy_coupling_weight="${SEQ_WM_POLICY_COUPLING_WEIGHT}" \
  --seq_wm_policy_coupling_value_weight="${SEQ_WM_POLICY_COUPLING_VALUE_WEIGHT}" \
  --seq_wm_policy_coupling_warmup_frames="${SEQ_WM_POLICY_COUPLING_WARMUP_FRAMES}" \
  --seq_wm_policy_coupling_ramp_frames="${SEQ_WM_POLICY_COUPLING_RAMP_FRAMES}" \
  --seq_wm_aux_critic_subtower_hidden="${SEQ_WM_AUX_CRITIC_SUBTOWER_HIDDEN}" \
  --kit_args "${DEXPBT_KIT_ARGS}" \
  >"${LOG}" 2>&1 &

PID=$!
echo "${PID}" > "${RUN_DIR}/train.pid"
echo "[seq_wm_v2_sanity] pid=${PID} log=${LOG}"
echo "[seq_wm_v2_sanity] Tail with: tail -f ${LOG}"
echo "[seq_wm_v2_sanity] Distill activity: grep -E 'seq_policy_aux|SeqDistill' ${LOG}"
