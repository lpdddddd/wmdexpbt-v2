#!/usr/bin/env bash
# PBT-WM Strategy Zoo v17.3 — AWR rescue 5B formal run.
# Same population/scaffold as v17.3 900M smoke; target 5B env frames per worker.

set -euo pipefail

RUNTIME_ROOT="/root/autodl-tmp/wmdexpbt/runtime_root"
RESULTS_ROOT="/root/autodl-tmp/wmdexpbt/results"
RUNS_ROOT="${RESULTS_ROOT}/runs"
DATE=$(date +%Y%m%d_%H%M%S)
COMMIT_SHORT="$(git -C /root/autodl-tmp/wmdexpbt rev-parse --short HEAD 2>/dev/null || echo unknown)"
RUN_DIR="${RUNS_ROOT}/regrasping_pbt_wm_strategy_zoo_v17_3_awr_rescue_5b_${COMMIT_SHORT}_${DATE}"
LOG="${RUN_DIR}/launch_master.log"
PIDFILE="${RUN_DIR}/nohup.pid"

cd "${RUNTIME_ROOT}" || exit 1
COMMIT_SHA="$(git -C /root/autodl-tmp/wmdexpbt rev-parse HEAD 2>/dev/null || echo unknown)"
export COMMIT_SHA

BASE_SEED="${BASE_SEED:-42}"
NUM_POLICIES="${NUM_POLICIES:-6}"
DEXPBT_GPUS="${DEXPBT_GPUS:-0,1,2,3,4,5}"
# 5B env frames @ 8192 envs * horizon 16
MAX_ITER="${MAX_ITER:-38147}"
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

PROFILE_LIST="ppo_only,elite_bank_actor_distill_fast_awr,wm_pbt_source_selector_v2,wm_pbt_source_selector_awr,elite_bank_actor_distill_awr,elite_bank_actor_distill_strong_awr"

export DEXPBT_MAX_ACTIVE="${DEXPBT_MAX_ACTIVE:-6}"
export DEXPBT_INTER_LAUNCH_DELAY_S="${DEXPBT_INTER_LAUNCH_DELAY_S:-180}"
export DEXPBT_STARTUP_TIMEOUT_S="${DEXPBT_STARTUP_TIMEOUT_S:-1200}"
export DEXPBT_STARTUP_MAX_RESTARTS="${DEXPBT_STARTUP_MAX_RESTARTS:-1}"
export DEXPBT_AWR_BUILDER_MODE=vectorized
export DEXPBT_AWR_VECTORIZE_ALL_FRAGMENTS=1
export NUM_POLICIES
export DEXPBT_GPUS

mkdir -p "$RUN_DIR"
echo "$RUN_DIR" > "${RESULTS_ROOT}/pointers/latest_pbt_wm_strategy_zoo_v17_3_awr_rescue_5b_run.txt"
echo "$RUN_DIR" > "${RESULTS_ROOT}/pointers/latest_run.txt"
echo "$RUN_DIR" > "${RUN_DIR}/RUN_DIR.txt"

cat > "${RUN_DIR}/RUN_CONFIG.txt" <<EOF
launcher=start_regrasping_pbt_wm_strategy_zoo_v17_3_awr_rescue_5b.sh
run_name=v17_3_awr_rescue_5b_${COMMIT_SHORT}
run_type=5b_formal_awr_rescue_v17_3
profile=strategy_zoo_v1.7_online_elite_awr
config_version=strategy_zoo_v1.7.3_awr_rescue
commit=${COMMIT_SHA}
smoke_mode=false
target_frames=5000000000
max_iterations=${MAX_ITER}
max_active=6
num_policies=${NUM_POLICIES}
gpus=${DEXPBT_GPUS}
base_seed=${BASE_SEED}
pbt_interval_steps=20000000
pbt_start_after=${PBT_START_AFTER}
pbt_initial_delay=${PBT_INITIAL_DELAY}
policy_00_profile=ppo_only
policy_01_profile=elite_bank_actor_distill_fast_awr
policy_02_profile=wm_pbt_source_selector_v2
policy_03_profile=wm_pbt_source_selector_awr
policy_04_profile=elite_bank_actor_distill_awr
policy_05_profile=elite_bank_actor_distill_strong_awr
policy_01_awr_loss_weight=0.0005
policy_03_awr_loss_weight=0.0003
policy_04_awr_loss_weight=0.0007
policy_05_awr_loss_weight=0.0010
awr_builder_mode=vectorized
awr_vectorize_all_fragments=1
awr_collection_mode=rollout_buffer
awr_delta_metric=mean
awr_delta_soft=0.20
awr_delta_mid=0.45
awr_delta_hard=0.80
awr_delta_mid_weight=0.5
awr_delta_far_weight=0.2
awr_collect_after_frames=400000000
awr_collect_every=4
awr_start_after_frames=500000000
awr_ramp_frames=150000000
awr_max_episode_len=320
awr_batch_steps=4096
awr_update_every=4
awr_min_loss_steps_per_entry=16
awr_kl_gate_enable=false
awr_broadcast_enabled=false
awr_inherit_enabled=true
note=v17.3 AWR rescue 5B: same config as passed 900M smoke; vectorized builder + mean delta gate.
monitor=none
EOF

cat > "${RUN_DIR}/LOGGING_HOOKS.txt" <<'EOF'
Formal run hooks:
- training_curve_policy_XX.jsonl: env_frames, pbt_iteration, awr_*, rollout_builder_*
- pbt_tick_debug_policy_XX.jsonl: PBT tick debug growth
- dexpbt_8gpu_ws/metrics/pbt_copy_events.jsonl
- train.log: fps total, frames
EOF

source "${RUNTIME_ROOT}/scripts/env_dexpbt.sh"
export TORCHDYNAMO_DISABLE=1
export PYTHONUNBUFFERED=1
export OMNI_KIT_TELEMETRY=0
export CARB_CRASH_REPORTER=0

cd "${RUNTIME_ROOT}/src/dexpbt_isaaclab"

LAUNCH_CMD=(bash "${RUNTIME_ROOT}/scripts/launch_8gpu_dexpbt.sh"
  "$RUN_DIR"
  --task DexPBT-Regrasping-KukaAllegro-Direct-v0
  --num-envs 8192
  --max-iterations "$MAX_ITER"
  --pbt-interval-steps 20000000
  --pbt-start-after "$PBT_START_AFTER"
  --pbt-initial-delay "$PBT_INITIAL_DELAY"
  --profile-per-policy "$PROFILE_LIST"
  --extra
  --seed "$BASE_SEED"
  --awr_enable
  --awr_collection_mode rollout_buffer
  --awr_delta_metric mean
  --awr_delta_soft 0.20
  --awr_delta_mid 0.45
  --awr_delta_hard 0.80
  --awr_delta_mid_weight 0.5
  --awr_delta_far_weight 0.2
  --awr_collect_after_frames 400000000
  --awr_collect_every 4
  --awr_start_after_frames 500000000
  --awr_ramp_frames 150000000
  --awr_batch_steps 4096
  --awr_max_episode_len 320
  --awr_min_burnin_steps 32
  --awr_min_loss_steps_per_entry 16
  --awr_copy_top_k_steps 50000
  --awr_broadcast_top_k_steps 10000
  --awr_update_every 4
  --awr_min_buffer_size 50000
  --awr_buffer_capacity 400000
  --awr_loss_cap 0.03
  --pbt_ranking_metric success_aware_r_meta
  --pbt_min_frames_as_source_after_restart 50000000
  --pbt_min_frames_between_restarts 100000000
  --pbt_disable_self_restart
  --pbt_source_selection_mode tiered
  --no-pbt_enable_best_nsucc_recovery
  --pbt_allow_best_nsucc_archive_recovery
  --pbt_recovery_cooldown_frames 400000000
  --pbt_curriculum_freeze_frames 100000000
  --pbt_curriculum_freeze_on_recovery_frames 100000000
  --curriculum_mode mastery
  --pbt_source_require_stable_mean
  --mastery_nsucc_mean 4.0
  --mastery_nsucc_max 20.0
  --mastery_consecutive_windows 2
  --curriculum_shrink_factor 0.95
  --pbt_single_shrink_per_window
  --post_shrink_consolidation_env_frames 150000000
  --curriculum_speed_mode adaptive_fast
  --curriculum_fast_strong_mean 30.0
  --curriculum_fast_mid_mean 20.0
  --curriculum_fast_low_mean 10.0
  --curriculum_fast_retention_threshold 0.8
  --curriculum_fast_retention_window 4
  --curriculum_fast_shrink_strong 0.925
  --curriculum_fast_shrink_mid 0.95
  --curriculum_fast_shrink_low 0.975
  --curriculum_fast_consolidation_frames 100000000
  --curriculum_fast_recovery_consolidation_frames 100000000
  --curriculum_fast_target_tolerance_800m 0.045
  --pbt_curriculum_inherit_mode source_safe
  --pbt_source_safe_recovery_consolidation_frames 100000000
  --pbt_source_safe_min_mean 20.0
  --pbt_source_safe_min_retention 0.7
  --pbt_protect_promising_workers
  --pbt_promising_nsucc_mean 20.0
  --pbt_promising_peak_mean 25.0
  --pbt_promising_min_retention 0.5
  --pbt_promising_protect_frames 250000000
  --pbt_promising_allow_hard_collapse_override
  --pbt_floor_recovery_aware
  --pbt_floor_tol_threshold 0.02
  --pbt_hard_floor_tol_threshold 0.012
  --pbt_floor_recovery_window 4
  --pbt_floor_recovery_protect_frames 300000000
  --pbt_floor_recovery_min_nsucc_protect 15.0
  --pbt_enable_best_floor_recovery_archive
  --pbt_wm_profile_copy_enable
  --pbt_wm_profile_copy_after_frames 300000000
  --pbt_wm_profile_copy_prob 0.7
  --pbt_wm_profile_keep_diversity_min_profiles 3
  --pbt_wm_profile_copy_min_nsucc_gap 3.0
  --pbt_wm_profile_copy_min_source_retention 0.50
  --pbt_wm_profile_copy_cooldown_frames 300000000
  --pbt_wm_profile_copy_bottom_k 2
  --pbt_wm_profile_copy_top_k 2
  --no-eir_residual_apply_to_rollout
  --pbt_early_collapse_detect
  --early_collapse_drop_ratio 0.3
  --early_collapse_patience 1
  --curriculum_collapse_detect
  --collapse_drop_ratio 0.5
  --collapse_patience 2
  --collapse_freeze_env_frames 200000000
  --stable_source_min_mean 3.0
  --stable_source_min_max 20.0
  --pbt_recovery_bypass_elimination_burnin
  --pbt_recovery_tolerance_mode last_mastered
  --pbt_crosscopy_tolerance_mode last_mastered
  --curriculum_rollback_cooldown_frames 50000000
  --curriculum_safety_enable
  --curriculum_safety_start_after_frames 500000000
  --curriculum_safety_min_nsucc_mean 10.0
  --curriculum_safety_global_retention_threshold 0.80
  --curriculum_safety_drop_ratio_threshold 0.30
  --curriculum_safety_slope_threshold -2.0
  --curriculum_safety_window 6
  --curriculum_safety_block_frames 100000000
  --curriculum_safety_use_real_fallback
  --curriculum_strong_shrink_global_retention_gate
  --curriculum_strong_shrink_min_global_retention 0.85
  --curriculum_strong_shrink_min_stable_ticks 3
  --curriculum_strong_shrink_long_window 12
  --curriculum_danger_zone_enable
  --curriculum_danger_tol 0.030
  --curriculum_danger_disable_strong_shrink
  --curriculum_danger_min_stable_ticks 4
  --curriculum_danger_shrink_interval_mult 2.0
  --curriculum_danger_global_retention_threshold 0.85
  --curriculum_rollback_on_repeated_collapse
  --curriculum_rollback_collapse_patience 2
  --curriculum_rollback_to_last_mastered
  --curriculum_rollback_freeze_frames 200000000
  --curriculum_last_mastered_min_nsucc_mean 30.0
  --curriculum_last_mastered_min_global_retention 0.85
  --curriculum_last_mastered_stable_ticks 3
  --wm_imagined_diagnostic_enable
  --wm_imagined_shadow_block_enable
  --wm_imagined_shadow_uncertainty_threshold 0.95
  "${SHARED_WM_HEAD_FLAGS[@]}"
)

printf '%q ' "${LAUNCH_CMD[@]}" > "${RUN_DIR}/LAUNCH_CMD.sh"
echo >> "${RUN_DIR}/LAUNCH_CMD.sh"

nohup "${LAUNCH_CMD[@]}" >"$LOG" 2>&1 &
echo $! > "$PIDFILE"

echo "launch_master pid=$(cat "$PIDFILE") log=$LOG"
echo "run_dir=$RUN_DIR"
echo "commit=${COMMIT_SHA}"
echo "run_name=v17_3_awr_rescue_5b_${COMMIT_SHORT}"
