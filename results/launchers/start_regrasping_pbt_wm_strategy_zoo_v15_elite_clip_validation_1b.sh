#!/usr/bin/env bash
# PBT-WM Strategy Zoo v1.5 — elite-heavy distill-clip validation (1B validation).
#
# Goal: restore the v1.3 elite-heavy precondition (elite bank fills, teachers are
# selected, soft activation gate opens) and validate ONLY the distill clip fix.
# p1/p5 are the clip-fix variants (fast_v2/strong_v2) with a diagnostic guidance
# band widened to 0.06 so they can populate the bank before tol reaches 0.04.
# p0/p2/p4 are the original elite_bank_actor_distill backbone (guidance 0.04).
# p3 is the single selector. No ppo_only, no selector_v2, no RSSM/MPPI profiles.
# Residual rollout apply stays OFF. This is a 1B validation, not a perf run.

set -euo pipefail

RUNTIME_ROOT="/root/autodl-tmp/wmdexpbt/runtime_root"
RESULTS_ROOT="/root/autodl-tmp/wmdexpbt/results"
RUNS_ROOT="${RESULTS_ROOT}/runs"
DATE=$(date +%Y%m%d_%H%M%S)
RUN_DIR="${RUNS_ROOT}/regrasping_pbt_wm_strategy_zoo_v15_elite_clip_validation_1b_${DATE}"
LOG="${RUN_DIR}/launch_master.log"
PIDFILE="${RUN_DIR}/nohup.pid"
COMMIT_SHA="${COMMIT_SHA:-unknown}"

BASE_SEED="${BASE_SEED:-42}"
NUM_POLICIES="${NUM_POLICIES:-6}"
DEXPBT_GPUS="${DEXPBT_GPUS:-0,1,2,3,4,5}"
# 1B target. 2B used MAX_ITER=15259, so 1B ~= 7630.
MAX_ITER="${MAX_ITER:-7630}"
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

PROFILE_LIST="elite_bank_actor_distill,elite_bank_actor_distill_fast_v2,elite_bank_actor_distill,wm_pbt_source_selector,elite_bank_actor_distill,elite_bank_actor_distill_strong_v2"

export DEXPBT_MAX_ACTIVE="${DEXPBT_MAX_ACTIVE:-6}"
export DEXPBT_INTER_LAUNCH_DELAY_S="${DEXPBT_INTER_LAUNCH_DELAY_S:-180}"
export DEXPBT_STARTUP_TIMEOUT_S="${DEXPBT_STARTUP_TIMEOUT_S:-1200}"
export DEXPBT_STARTUP_MAX_RESTARTS="${DEXPBT_STARTUP_MAX_RESTARTS:-1}"
export NUM_POLICIES
export DEXPBT_GPUS

mkdir -p "$RUN_DIR"
echo "$RUN_DIR" > "${RESULTS_ROOT}/pointers/latest_pbt_wm_strategy_zoo_v15_elite_clip_validation_1b_run.txt"
echo "$RUN_DIR" > "${RUN_DIR}/RUN_DIR.txt"

cat > "${RUN_DIR}/RUN_CONFIG.txt" <<EOF
launcher=start_regrasping_pbt_wm_strategy_zoo_v15_elite_clip_validation_1b.sh
profile=strategy_zoo_v1.5_elite_heavy_clip_validation
config_version=strategy_zoo_v1.5_elite_heavy_clip_validation
commit=${COMMIT_SHA}
target_frames=1000000000
max_iterations=${MAX_ITER}
max_active=6
num_policies=${NUM_POLICIES}
gpus=${DEXPBT_GPUS}
base_seed=${BASE_SEED}
policy_00_profile=elite_bank_actor_distill
policy_01_profile=elite_bank_actor_distill_fast_v2
policy_02_profile=elite_bank_actor_distill
policy_03_profile=wm_pbt_source_selector
policy_04_profile=elite_bank_actor_distill
policy_05_profile=elite_bank_actor_distill_strong_v2
rssm_mppi_in_performance_run=false
ppo_only_in_population=false
selector_v2_in_population=false
elite_heavy=true
clip_validation=true
clip_validation_bank_tol_p1_p5=0.06
elite_distill_large_delta_mode=clip
eir_residual_apply_to_rollout=false
residual_rollout_apply=false
pbt_wm_profile_copy_enable=true
pbt_wm_profile_copy_after_frames=300000000
pbt_wm_profile_copy_prob=0.7
pbt_wm_profile_keep_diversity_min_profiles=3
pbt_wm_profile_copy_min_nsucc_gap=3.0
pbt_wm_profile_copy_min_source_retention=0.50
pbt_wm_profile_copy_cooldown_frames=300000000
pbt_wm_profile_copy_bottom_k=2
pbt_wm_profile_copy_top_k=2
pbt_wm_profile_copy_source_min_nsucc_mean=0.0
pbt_wm_profile_copy_source_min_frames=0
pbt_wm_profile_copy_top1_override_enable=false
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
note=1B validation run: v1.5 elite-heavy distill-clip validation. Only clip fix carried over from v1.4; selector/profile copy and population ecology NOT changed for validation.
clip_validation_note=p1/p5 use clip profiles with guidance band widened to 0.06 (diagnostic). p0/p2/p4 keep original elite guidance 0.04.
soft_activation_gate=PBT-WM Soft Activation Gate v1.
residual_apply_note=eir_residual_apply_to_rollout stays OFF.
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
  --pbt_wm_profile_copy_enable \
  --pbt_wm_profile_copy_after_frames 300000000 \
  --pbt_wm_profile_copy_prob 0.7 \
  --pbt_wm_profile_keep_diversity_min_profiles 3 \
  --pbt_wm_profile_copy_min_nsucc_gap 3.0 \
  --pbt_wm_profile_copy_min_source_retention 0.50 \
  --pbt_wm_profile_copy_cooldown_frames 300000000 \
  --pbt_wm_profile_copy_bottom_k 2 \
  --pbt_wm_profile_copy_top_k 2 \
  --pbt_wm_profile_copy_source_min_nsucc_mean 0.0 \
  --pbt_wm_profile_copy_source_min_frames 0 \
  --no-pbt_wm_profile_copy_top1_override_enable \
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
