#!/usr/bin/env bash
# Unified Elite-WM DexPBT — 1.2B smoke run (max_active=6).
#
# All six workers share the same unified substrate profile (EliteBank + WM
# diagnostic/shadow + PBT/curriculum). PBT copies weights and curriculum state,
# not discrete profile identities. Profile copy, AWR, and residual rollout
# apply are explicitly disabled.

set -euo pipefail

RUNTIME_ROOT="/root/autodl-tmp/wmdexpbt/runtime_root"
RESULTS_ROOT="/root/autodl-tmp/wmdexpbt/results"
RUNS_ROOT="${RESULTS_ROOT}/runs"
DATE=$(date +%Y%m%d_%H%M%S)
RUN_DIR="${RUNS_ROOT}/regrasping_unified_elite_wm_pbt_1p2b_smoke_${DATE}"
LOG="${RUN_DIR}/launch_master.log"
PIDFILE="${RUN_DIR}/nohup.pid"
COMMIT_SHA="${COMMIT_SHA:-unknown}"

BASE_SEED="${BASE_SEED:-42}"
NUM_POLICIES="${NUM_POLICIES:-6}"
DEXPBT_GPUS="${DEXPBT_GPUS:-0,1,2,3,4,5}"
MAX_ITER="${MAX_ITER:-9156}"
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
  --wm_imagined_valid_mode soft_calibrated
  --wm_imagined_valid_uncertainty_threshold 0.5
  --wm_imagined_soft_uncertainty_threshold 0.95
  --wm_imagined_min_logit_abs_mean 0.05
  --wm_imagined_min_head_signal_std 0.02
  --wm_imagined_soft_validity_weight 0.25
  --eir_candidate_gate_mode profile_soft
  --eir_candidate_min_weight_for_retrieval 0.05
)

PROFILE_LIST="unified_elite_wm_pbt,unified_elite_wm_pbt,unified_elite_wm_pbt,unified_elite_wm_pbt,unified_elite_wm_pbt,unified_elite_wm_pbt"

export DEXPBT_MAX_ACTIVE="${DEXPBT_MAX_ACTIVE:-6}"
export DEXPBT_INTER_LAUNCH_DELAY_S="${DEXPBT_INTER_LAUNCH_DELAY_S:-180}"
export DEXPBT_STARTUP_TIMEOUT_S="${DEXPBT_STARTUP_TIMEOUT_S:-1200}"
export DEXPBT_STARTUP_MAX_RESTARTS="${DEXPBT_STARTUP_MAX_RESTARTS:-1}"
export NUM_POLICIES
export DEXPBT_GPUS

mkdir -p "$RUN_DIR"
echo "$RUN_DIR" > "${RESULTS_ROOT}/pointers/latest_unified_elite_wm_pbt_1p2b_smoke_run.txt"
echo "$RUN_DIR" > "${RUN_DIR}/RUN_DIR.txt"

cat > "${RUN_DIR}/RUN_CONFIG.txt" <<EOF
launcher=start_regrasping_unified_elite_wm_pbt_1p2b_smoke.sh
method=unified_elite_wm_dexpbt
config_version=unified_elite_wm_pbt_1p2b_smoke
commit=${COMMIT_SHA}
target_frames=1200000000
max_iterations=${MAX_ITER}
max_active=6
num_policies=${NUM_POLICIES}
gpus=${DEXPBT_GPUS}
base_seed=${BASE_SEED}
profile_all_workers=unified_elite_wm_pbt
profile_copy=false
awr=false
eir_residual_rollout_apply=false
wm_pbt_score_mode=shadow
wm_pbt_score_mode_note=shadow_tiebreak not implemented in current scoring code; "shadow" publishes WM telemetry without driving PBT ranking.
method_note=All workers share EliteBank + WM selector/diagnostic + PBT/curriculum mechanisms. PBT propagates weights/curriculum state, not discrete profile identities.
pbt_min_frames_between_restarts=100000000
pbt_promising_protect_frames=250000000
pbt_floor_recovery_protect_frames=300000000
curriculum_fast_shrink_strong=0.925
curriculum_fast_shrink_mid=0.95
curriculum_fast_shrink_low=0.975
curriculum_fast_consolidation_frames=100000000
post_shrink_consolidation_env_frames=150000000
curriculum_fast_target_tolerance_800m=0.045
hybrid_curriculum_safety=true
real_signal_fallback=true
danger_tol=0.030
strong_shrink_global_retention_gate=true
rollback_to_last_mastered=true
wm_imagined_shadow_only=true
wm_imagined_head_rescue=true
wm_imagined_valid_mode=soft_calibrated
wm_head_balance=true
soft_candidate_retrieval=true
note=1.2B smoke: unified substrate for all workers; profile copy / AWR / residual rollout apply disabled.
soft_activation_gate=PBT-WM Soft Activation Gate v1: actor_distill / WM source score / WM curriculum diagnostics use a continuous activation weight.
residual_apply_note=eir_residual_apply_to_rollout stays OFF and HARD-gated. The soft activation gate does NOT relax residual rollout apply.
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
  --pbt_source_selection_mode tiered \
  --no-pbt_enable_best_nsucc_recovery \
  --pbt_allow_best_nsucc_archive_recovery \
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
  --post_shrink_consolidation_env_frames 150000000 \
  --curriculum_speed_mode adaptive_fast \
  --curriculum_fast_strong_mean 30.0 \
  --curriculum_fast_mid_mean 20.0 \
  --curriculum_fast_low_mean 10.0 \
  --curriculum_fast_retention_threshold 0.8 \
  --curriculum_fast_retention_window 4 \
  --curriculum_fast_shrink_strong 0.925 \
  --curriculum_fast_shrink_mid 0.95 \
  --curriculum_fast_shrink_low 0.975 \
  --curriculum_fast_consolidation_frames 100000000 \
  --curriculum_fast_recovery_consolidation_frames 100000000 \
  --curriculum_fast_target_tolerance_800m 0.045 \
  --pbt_curriculum_inherit_mode source_safe \
  --pbt_source_safe_recovery_consolidation_frames 100000000 \
  --pbt_source_safe_min_mean 20.0 \
  --pbt_source_safe_min_retention 0.7 \
  --pbt_protect_promising_workers \
  --pbt_promising_nsucc_mean 20.0 \
  --pbt_promising_peak_mean 25.0 \
  --pbt_promising_min_retention 0.5 \
  --pbt_promising_protect_frames 250000000 \
  --pbt_promising_allow_hard_collapse_override \
  --pbt_floor_recovery_aware \
  --pbt_floor_tol_threshold 0.02 \
  --pbt_hard_floor_tol_threshold 0.012 \
  --pbt_floor_recovery_window 4 \
  --pbt_floor_recovery_protect_frames 300000000 \
  --pbt_floor_recovery_min_nsucc_protect 15.0 \
  --pbt_enable_best_floor_recovery_archive \
  --no-pbt_wm_profile_copy_enable \
  --no-awr_enable \
  --no-eir_residual_apply_to_rollout \
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
  --curriculum_rollback_cooldown_frames 50000000 \
  --curriculum_safety_enable \
  --curriculum_safety_start_after_frames 500000000 \
  --curriculum_safety_min_nsucc_mean 10.0 \
  --curriculum_safety_global_retention_threshold 0.80 \
  --curriculum_safety_drop_ratio_threshold 0.30 \
  --curriculum_safety_slope_threshold -2.0 \
  --curriculum_safety_window 6 \
  --curriculum_safety_block_frames 100000000 \
  --curriculum_safety_use_real_fallback \
  --curriculum_strong_shrink_global_retention_gate \
  --curriculum_strong_shrink_min_global_retention 0.85 \
  --curriculum_strong_shrink_min_stable_ticks 3 \
  --curriculum_strong_shrink_long_window 12 \
  --curriculum_danger_zone_enable \
  --curriculum_danger_tol 0.030 \
  --curriculum_danger_disable_strong_shrink \
  --curriculum_danger_min_stable_ticks 4 \
  --curriculum_danger_shrink_interval_mult 2.0 \
  --curriculum_danger_global_retention_threshold 0.85 \
  --curriculum_rollback_on_repeated_collapse \
  --curriculum_rollback_collapse_patience 2 \
  --curriculum_rollback_to_last_mastered \
  --curriculum_rollback_freeze_frames 200000000 \
  --curriculum_last_mastered_min_nsucc_mean 30.0 \
  --curriculum_last_mastered_min_global_retention 0.85 \
  --curriculum_last_mastered_stable_ticks 3 \
  --wm_imagined_diagnostic_enable \
  --wm_imagined_shadow_block_enable \
  --wm_imagined_shadow_uncertainty_threshold 0.95 \
  "${SHARED_WM_HEAD_FLAGS[@]}" \
  >"$LOG" 2>&1 &

echo $! > "$PIDFILE"
echo "launch_master pid=$(cat "$PIDFILE") log=$LOG"
