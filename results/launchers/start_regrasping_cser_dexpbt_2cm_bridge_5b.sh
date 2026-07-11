#!/usr/bin/env bash
# CSER-DexPBT 2cm Bridge formal 5B launcher.
#
# Base = conservative retention strict 5B formal config
# Plus = same-tol retention, bridge ladder/probation, snippet adjacent+compat,
#        late degradation rescue, per-worker population-frame stop.
#
# Rollback priority under bridge probation:
#   1) bridge hard-collapse (same-level mean streak) may roll back
#   2) legacy cross-tol peak hard-collapse is suppressed (logged only)
#   3) repeated-collapse ordinary rollback remains blocked while probation active
set -euo pipefail

RUNTIME_ROOT="/root/autodl-tmp/wmdexpbt/runtime_root"
RESULTS_ROOT="/root/autodl-tmp/wmdexpbt/results"
RUNS_ROOT="${RESULTS_ROOT}/runs"
DATE=$(date +%Y%m%d_%H%M%S)
RUN_DIR="${RUNS_ROOT}/regrasping_cser_dexpbt_2cm_bridge_5b_${DATE}"
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

PROFILE_LIST="cser_dexpbt_2cm_bridge,cser_dexpbt_2cm_bridge,cser_dexpbt_2cm_bridge,cser_dexpbt_2cm_bridge,cser_dexpbt_2cm_bridge,cser_dexpbt_2cm_bridge"

export DEXPBT_MAX_ACTIVE="${DEXPBT_MAX_ACTIVE:-6}"
export DEXPBT_INTER_LAUNCH_DELAY_S="${DEXPBT_INTER_LAUNCH_DELAY_S:-180}"
export DEXPBT_STARTUP_TIMEOUT_S="${DEXPBT_STARTUP_TIMEOUT_S:-1200}"
export DEXPBT_STARTUP_MAX_RESTARTS="${DEXPBT_STARTUP_MAX_RESTARTS:-3}"
export NUM_POLICIES
export DEXPBT_GPUS

mkdir -p "$RUN_DIR"
echo "$RUN_DIR" > "${RESULTS_ROOT}/pointers/latest_cser_dexpbt_2cm_bridge_5b_run.txt"
echo "$RUN_DIR" > "${RUN_DIR}/RUN_DIR.txt"

cat > "${RUN_DIR}/RUN_CONFIG.txt" <<EOF
launcher=start_regrasping_cser_dexpbt_2cm_bridge_5b.sh
method=cser_dexpbt_2cm_bridge
config_version=cser_dexpbt_2cm_bridge_5b
commit=${COMMIT_SHA}
base_config=cser_dexpbt_conservative_retention_strict_5b
termination_mode=per_worker_population_frames
target_population_frames=${TARGET_FRAMES}
max_iterations=${MAX_ITER}
max_iterations_role=safety_cap_only
max_active=6
num_policies=${NUM_POLICIES}
gpus=${DEXPBT_GPUS}
base_seed=${BASE_SEED}
profile_all_workers=cser_dexpbt_2cm_bridge
profile_copy=false
awr=false
eir_residual_rollout_apply=false
eir_actor_distill_fallback=true
wm_pbt_score_mode=active
pbt_allow_mid_self_mutation=false
mid_self_mutation_disabled=true
bridge_additions=same_tol_retention,bridge_ladder,bridge_probation,snippet_adjacent_compat,late_degradation_rescue,frame_stop
rollback_priority=bridge_hard_collapse_over_legacy_during_probation
note=Old conservative retention formal flags + 2cm bridge additions; hard-offs enforced.
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
  --pbt_strong_live_inherit_exact_enable \
  --pbt_strong_live_inherit_max_tol 0.035 \
  --pbt_strong_live_inherit_min_mean 30.0 \
  --pbt_strong_live_inherit_min_max 35.0 \
  --pre_floor_peak_guard_enable \
  --pre_floor_peak_guard_tol_threshold 0.030 \
  --pre_floor_peak_guard_target_tol 0.025 \
  --pre_floor_peak_guard_peak_threshold 35.0 \
  --pre_floor_peak_guard_min_retention 0.92 \
  --pre_floor_peak_guard_min_stable_ticks 3 \
  --pre_floor_peak_guard_freeze_frames 200000000 \
  --late_floor_retention_gate_enable \
  --late_floor_retention_tol_threshold 0.030 \
  --late_floor_retention_peak_threshold 35.0 \
  --late_floor_retention_min_retention 0.88 \
  --late_floor_retention_hard_min_retention 0.75 \
  --late_floor_retention_patience 2 \
  --late_floor_retention_freeze_frames 250000000 \
  --late_floor_retention_block_shrink \
  --late_floor_retention_freeze_mutation \
  --no-late_floor_retention_rollback_enable \
  --pbt_floor_recovery_aware \
  --pbt_floor_tol_threshold 0.030 \
  --pbt_hard_floor_tol_threshold 0.025 \
  --pbt_floor_recovery_window 4 \
  --pbt_floor_recovery_protect_frames 300000000 \
  --pbt_floor_recovery_min_nsucc_protect 15.0 \
  --pbt_enable_best_floor_recovery_archive \
  --no-pbt_wm_profile_copy_enable \
  --no-awr_enable \
  --no-eir_residual_apply_to_rollout \
  --no-eir_use_rssm_mppi \
  --eir_mppi_mode off \
  --elite_snippet_enable \
  --snippet_distill_enable \
  --snippet_wm_rerank_enable \
  --wm_calibration_enable \
  --wm_calibration_eval_tol_mode hybrid \
  --wm_calibration_eval_tol_threshold 0.05 \
  --wm_calibration_eval_tol_cap 0.06 \
  --wm_calibration_min_samples 128 \
  --eir_actor_distill_enable \
  --eir_actor_distill_allowed_sources elite,archive \
  --eir_actor_distill_weight 0.0005 \
  --eir_actor_distill_loss_cap 0.01 \
  --eir_actor_distill_max_delta 0.25 \
  --actor_distill_fallback_with_snippet_enable \
  --actor_distill_fallback_min_snippet_bank_size 5000 \
  --actor_distill_fallback_min_rerank_changed_frac 0.0001 \
  --pbt_consolidation_enable \
  --pbt_consolidation_stage1_end_frames 2200000000 \
  --pbt_consolidation_stage2_end_frames 3500000000 \
  --pbt_consolidation_stage3_end_frames 5000000000 \
  --pbt_consolidation_stage2_interval_mult 2.0 \
  --pbt_consolidation_stage3_interval_mult 4.0 \
  --pbt_consolidation_stage2_mutation_rate_mult 0.5 \
  --pbt_consolidation_stage3_bottom_only \
  --pbt_consolidation_stage3_hard_collapse_only \
  --pbt_consolidation_stage2_retention_threshold 0.85 \
  --pbt_consolidation_collapse_nsucc_mean_threshold 1.0 \
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
  --frame_stop_enable \
  --target_population_frames "$TARGET_FRAMES" \
  --frame_stop_grace_iterations 2 \
  --same_tol_retention_enable \
  --bridge_curriculum_enable \
  --bridge_probation_enable \
  --snippet_adjacent_stage_mixing_enable \
  --snippet_tolerance_compatibility_enable \
  --late_degradation_rescue_enable \
  "${SHARED_WM_HEAD_FLAGS[@]}" \
  >>"$LOG" 2>&1 &

echo $! > "$PIDFILE"
echo "launch_master pid=$(cat "$PIDFILE") log=$LOG run_dir=$RUN_DIR"
