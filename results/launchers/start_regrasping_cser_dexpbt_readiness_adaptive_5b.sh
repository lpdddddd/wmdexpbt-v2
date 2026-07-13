#!/usr/bin/env bash
# Readiness-Adaptive CSER-DexPBT formal 5B launcher.
#
# Method  = cser_dexpbt_readiness_adaptive (profile id 23, authoritative expansion).
# Control = readiness-driven fast start, WM graduation, ESS distill, and
#           accelerated adaptive PBT on top of the Profile-22 progress gate.
#
# The profile owns ALL method-specific flags and is restart-safe (re-expanded
# authoritatively after every PBT restart).  This launcher therefore contains
# NO fixed tolerance ladder and NO method distance constant: it only sets
# generic curriculum / PBT / safety / frame-stop flags shared with Profile 22.
set -euo pipefail

RUNTIME_ROOT="/root/autodl-tmp/wmdexpbt/runtime_root"
RESULTS_ROOT="/root/autodl-tmp/wmdexpbt/results"
RUNS_ROOT="${RESULTS_ROOT}/runs"
DATE=$(date +%Y%m%d_%H%M%S)
RUN_DIR="${RUNS_ROOT}/regrasping_cser_dexpbt_readiness_adaptive_5b_${DATE}"
LOG="${RUN_DIR}/launch_master.log"
PIDFILE="${RUN_DIR}/nohup.pid"
COMMIT_SHA="${COMMIT_SHA:-unknown}"

BASE_SEED="${BASE_SEED:-42}"
NUM_POLICIES="${NUM_POLICIES:-6}"
DEXPBT_GPUS="${DEXPBT_GPUS:-0,1,2,3,4,5}"
MAX_ITER="${MAX_ITER:-70000}"
TARGET_FRAMES="${TARGET_FRAMES:-5000000000}"
PBT_START_AFTER="${PBT_START_AFTER:-200000000}"
PBT_INITIAL_DELAY="${PBT_INITIAL_DELAY:-50000000}"

SHARED_WM_HEAD_FLAGS=(
  --seq_wm_head_balance_enable
  --seq_wm_head_pos_weight_auto
  --seq_wm_head_pos_weight_max 10.0
  --seq_wm_lambda_horizon_success 0.5
  --seq_wm_lambda_drop_risk 0.5
  --seq_wm_lambda_recovery 0.5
  --seq_wm_replay_prioritize_floor_recovery
  --seq_wm_floor_aware_weighting
  --eir_candidate_gate_mode profile_soft
  --eir_candidate_min_weight_for_retrieval 0.05
)

PROFILE="cser_dexpbt_readiness_adaptive"
PROFILE_LIST="${PROFILE},${PROFILE},${PROFILE},${PROFILE},${PROFILE},${PROFILE}"

export DEXPBT_MAX_ACTIVE="${DEXPBT_MAX_ACTIVE:-6}"
export DEXPBT_INTER_LAUNCH_DELAY_S="${DEXPBT_INTER_LAUNCH_DELAY_S:-180}"
export DEXPBT_STARTUP_TIMEOUT_S="${DEXPBT_STARTUP_TIMEOUT_S:-1200}"
export DEXPBT_STARTUP_MAX_RESTARTS="${DEXPBT_STARTUP_MAX_RESTARTS:-3}"
export NUM_POLICIES
export DEXPBT_GPUS

mkdir -p "$RUN_DIR" "${RESULTS_ROOT}/pointers"
echo "$RUN_DIR" > "${RESULTS_ROOT}/pointers/latest_cser_dexpbt_readiness_adaptive_5b_run.txt"
echo "$RUN_DIR" > "${RUN_DIR}/RUN_DIR.txt"

cat > "${RUN_DIR}/RUN_CONFIG.txt" <<EOF
launcher=start_regrasping_cser_dexpbt_readiness_adaptive_5b.sh
method=cser_dexpbt_readiness_adaptive
profile_id=23
config_version=cser_dexpbt_readiness_adaptive_5b
commit=${COMMIT_SHA}
termination_mode=per_worker_population_frames
target_population_frames=${TARGET_FRAMES}
max_iterations=${MAX_ITER}
max_iterations_role=safety_cap_only
max_active=${DEXPBT_MAX_ACTIVE}
num_policies=${NUM_POLICIES}
gpus=${DEXPBT_GPUS}
base_seed=${BASE_SEED}
profile_all_workers=${PROFILE}
profile_authoritative=true
profile_copy=false
awr=false
eir_residual_rollout_apply=false
eir_use_rssm_mppi=false
readiness_adaptive=on
snippet_retrieval_mode=log_relative
recovery_label_margin_mode=relative_tol
consolidation_schedule_mode=curriculum_progress
adaptive_probation=on
retention_conditioned_rescue=on
fixed_tolerance_ladder=none
note=Generalized progress-conditioned method; all method flags owned by the authoritative profile; no 2cm/0.0205/0.022 constants in this launcher.
EOF

source "${RUNTIME_ROOT}/scripts/env_dexpbt.sh"
export TORCHDYNAMO_DISABLE=1
export PYTHONUNBUFFERED=1
export OMNI_KIT_TELEMETRY=0
export CARB_CRASH_REPORTER=0

cd "${RUNTIME_ROOT}/src/dexpbt_isaaclab"

nohup bash "${RUNTIME_ROOT}/scripts/launch_8gpu_dexpbt.sh" \
  "$RUN_DIR" \
  --task DexPBT-Regrasping-KukaAllegro-Direct-v0 \
  --num-envs 8192 \
  --max-iterations "$MAX_ITER" \
  --pbt-interval-steps 20000000 \
  --pbt-start-after "$PBT_START_AFTER" \
  --pbt-initial-delay "$PBT_INITIAL_DELAY" \
  --profile-per-policy "$PROFILE_LIST" \
  --extra \
  --seed "$BASE_SEED" \
  --pbt_ranking_metric success_aware_r_meta \
  --pbt_min_frames_as_source_after_restart 50000000 \
  --pbt_min_frames_between_restarts 100000000 \
  --pbt_disable_self_restart \
  --no-pbt_allow_mid_self_mutation \
  --pbt_source_selection_mode tiered \
  --pbt_source_require_stable_mean \
  --curriculum_mode mastery \
  --mastery_nsucc_mean 4.0 \
  --mastery_nsucc_max 20.0 \
  --mastery_consecutive_windows 2 \
  --curriculum_shrink_factor 0.95 \
  --pbt_single_shrink_per_window \
  --post_shrink_consolidation_env_frames 150000000 \
  --curriculum_speed_mode adaptive_fast \
  --curriculum_fast_strong_mean 30.0 \
  --curriculum_fast_mid_mean 20.0 \
  --curriculum_fast_low_mean 10.0 \
  --curriculum_fast_retention_threshold 0.8 \
  --curriculum_fast_retention_window 4 \
  --pbt_curriculum_inherit_mode source_safe \
  --pbt_source_safe_min_mean 20.0 \
  --pbt_source_safe_min_retention 0.7 \
  --pbt_protect_promising_workers \
  --pbt_promising_nsucc_mean 20.0 \
  --pbt_promising_peak_mean 25.0 \
  --pbt_promising_min_retention 0.5 \
  --pbt_promising_protect_frames 250000000 \
  --pbt_promising_allow_hard_collapse_override \
  --no-pbt_wm_profile_copy_enable \
  --no-awr_enable \
  --no-eir_residual_apply_to_rollout \
  --no-eir_use_rssm_mppi \
  --eir_mppi_mode off \
  --pbt_consolidation_enable \
  --pbt_consolidation_schedule_mode curriculum_progress \
  --adaptive_precision_consolidation_enable \
  --adaptive_rescue_enable \
  --pbt_early_collapse_detect \
  --early_collapse_drop_ratio 0.3 \
  --early_collapse_patience 1 \
  --curriculum_collapse_detect \
  --collapse_drop_ratio 0.5 \
  --collapse_patience 2 \
  --collapse_freeze_env_frames 200000000 \
  --stable_source_min_mean 3.0 \
  --stable_source_min_max 20.0 \
  --pbt_recovery_bypass_elimination_burnin \
  --pbt_recovery_tolerance_mode last_mastered \
  --pbt_crosscopy_tolerance_mode last_mastered \
  --curriculum_safety_enable \
  --curriculum_safety_start_after_frames 500000000 \
  --curriculum_safety_min_nsucc_mean 10.0 \
  --curriculum_safety_global_retention_threshold 0.80 \
  --curriculum_safety_window 6 \
  --curriculum_safety_use_real_fallback \
  --wm_imagined_diagnostic_enable \
  --frame_stop_enable \
  --target_population_frames "$TARGET_FRAMES" \
  --frame_stop_grace_iterations 2 \
  "${SHARED_WM_HEAD_FLAGS[@]}" \
  >>"$LOG" 2>&1 &

echo $! > "$PIDFILE"
echo "launch_master pid=$(cat "$PIDFILE") log=$LOG run_dir=$RUN_DIR"
