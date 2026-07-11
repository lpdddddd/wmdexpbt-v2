#!/usr/bin/env bash
# EIR-MPC FULL H-step MPPI MAIN run (~3B / policy).
#
# Wired active RSSM floor-recovery baseline + FULL EIR-MPC:
#   - elite recovery bank + elite candidate retrieval
#   - gated recovery residual head WITH gated rollout apply
#   - light RSSM reranking (kill-switch; production warmup 800M)
#   - PBT elite broadcast (observer-side consume + per-tick live merge)
#
# Production warmups (constraint 19.5): scorer 200M, residual apply 300M,
# RSSM rerank 800M. These are real-performance runs, not wiring smokes.

set -euo pipefail

RUNTIME_ROOT="/root/autodl-tmp/wmdexpbt/runtime_root"
RESULTS_ROOT="/root/autodl-tmp/wmdexpbt/results"
RUNS_ROOT="${RESULTS_ROOT}/runs"
DATE=$(date +%Y%m%d_%H%M%S)
RUN_DIR="${RUNS_ROOT}/regrasping_eir_mpc_full_mppi_3b_${DATE}"
LOG="${RUN_DIR}/launch_master.log"
PIDFILE="${RUN_DIR}/nohup.pid"
COMMIT_SHA="${COMMIT_SHA:-unknown}"

WM_STRATEGY="rssm_belief_value"
BASE_SEED="${BASE_SEED:-42}"
NUM_POLICIES="${NUM_POLICIES:-6}"
DEXPBT_GPUS="${DEXPBT_GPUS:-0,1,2,3,4,5}"
MAX_ITER="${MAX_ITER:-22889}"
PBT_START_AFTER="${PBT_START_AFTER:-200000000}"
PBT_INITIAL_DELAY="${PBT_INITIAL_DELAY:-50000000}"

# EIR production warmups (constraint 19.5: main runs keep these).
EIR_START_AFTER_FRAMES="${EIR_START_AFTER_FRAMES:-200000000}"
EIR_SCORER_WARMUP_ITERS="${EIR_SCORER_WARMUP_ITERS:-2000}"
EIR_RESIDUAL_APPLY_AFTER="${EIR_RESIDUAL_APPLY_AFTER:-300000000}"
EIR_MPPI_START_AFTER="${EIR_MPPI_START_AFTER:-500000000}"

export DEXPBT_MAX_ACTIVE="${DEXPBT_MAX_ACTIVE:-2}"
export DEXPBT_INTER_LAUNCH_DELAY_S="${DEXPBT_INTER_LAUNCH_DELAY_S:-90}"
export DEXPBT_STARTUP_TIMEOUT_S="${DEXPBT_STARTUP_TIMEOUT_S:-900}"
export DEXPBT_STARTUP_MAX_RESTARTS="${DEXPBT_STARTUP_MAX_RESTARTS:-1}"
export NUM_POLICIES
export DEXPBT_GPUS

mkdir -p "$RUN_DIR"
echo "$RUN_DIR" > "${RESULTS_ROOT}/pointers/latest_eir_mpc_full_mppi_3b_run.txt"
echo "$RUN_DIR" > "${RUN_DIR}/RUN_DIR.txt"

cat > "${RUN_DIR}/RUN_CONFIG.txt" <<EOF
launcher=start_regrasping_eir_mpc_full_mppi_3b.sh
profile=eir_mpc_full_mppi_3b
commit=${COMMIT_SHA}
target_frames=3000000000
max_iterations=${MAX_ITER}
num_policies=${NUM_POLICIES}
gpus=${DEXPBT_GPUS}
base_seed=${BASE_SEED}
eir_enable=true
eir_use_elite_bank=true
eir_use_recovery_residual=true
eir_residual_apply_to_rollout=true
eir_use_rssm_mppi=true
eir_pbt_broadcast_elite_bank=true
eir_start_after_frames=${EIR_START_AFTER_FRAMES}
eir_residual_apply_after_frames=${EIR_RESIDUAL_APPLY_AFTER}
eir_mppi_start_after_frames=${EIR_MPPI_START_AFTER}
note=FULL EIR-MPC FULL H-step MPPI main run; production warmups (constraint 19.5).
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
  --extra \
  --seed "$BASE_SEED" \
  --enable_wm \
  --wm_strategy "$WM_STRATEGY" \
  --wm_strategy_mode fixed \
  --fixed_wm_strategy "$WM_STRATEGY" \
  --wm_load_mode reset \
  --seq_wm_seq_len 16 \
  --seq_wm_seq_batch_size 1024 \
  --seq_wm_lambda_total 0.05 \
  --seq_wm_lambda_value 0.1 \
  --no-seq_wm_value_detach_belief \
  --seq_wm_policy_coupling none \
  --pbt_ranking_metric success_aware_r_meta \
  --pbt_min_frames_as_source_after_restart 50000000 \
  --pbt_min_frames_between_restarts 50000000 \
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
  --post_shrink_consolidation_env_frames 100000000 \
  --curriculum_speed_mode adaptive_fast \
  --curriculum_fast_strong_mean 30.0 \
  --curriculum_fast_mid_mean 20.0 \
  --curriculum_fast_low_mean 10.0 \
  --curriculum_fast_retention_threshold 0.8 \
  --curriculum_fast_retention_window 4 \
  --curriculum_fast_shrink_strong 0.90 \
  --curriculum_fast_shrink_mid 0.925 \
  --curriculum_fast_shrink_low 0.95 \
  --curriculum_fast_consolidation_frames 50000000 \
  --curriculum_fast_recovery_consolidation_frames 100000000 \
  --curriculum_fast_target_tolerance_800m 0.04 \
  --pbt_curriculum_inherit_mode source_safe \
  --pbt_source_safe_recovery_consolidation_frames 50000000 \
  --pbt_source_unsafe_recovery_consolidation_frames 100000000 \
  --pbt_source_safe_min_mean 20.0 \
  --pbt_source_safe_min_retention 0.7 \
  --pbt_protect_promising_workers \
  --pbt_promising_nsucc_mean 20.0 \
  --pbt_promising_peak_mean 25.0 \
  --pbt_promising_min_retention 0.5 \
  --pbt_promising_protect_frames 150000000 \
  --pbt_promising_allow_hard_collapse_override \
  --pbt_floor_recovery_aware \
  --pbt_floor_tol_threshold 0.02 \
  --pbt_hard_floor_tol_threshold 0.012 \
  --pbt_floor_recovery_window 4 \
  --pbt_floor_recovery_protect_frames 200000000 \
  --pbt_floor_recovery_min_nsucc_protect 15.0 \
  --pbt_enable_best_floor_recovery_archive \
  --pbt_wm_score_mode active \
  --pbt_use_wm_recovery_score \
  --pbt_wm_recovery_weight 0.3 \
  --pbt_wm_success_weight 0.2 \
  --pbt_wm_drop_penalty 0.3 \
  --pbt_wm_uncertainty_penalty 0.2 \
  --pbt_wm_max_uncertainty 0.5 \
  --seq_wm_use_task_aux_inputs \
  --seq_wm_loss_preset task_focused \
  --seq_wm_floor_aware_weighting \
  --seq_wm_use_horizon_heads \
  --seq_wm_horizon_steps 4,8 \
  --seq_wm_horizon_skip_too_long \
  --seq_wm_use_recovery_head \
  --seq_wm_recovery_horizon 8 \
  --seq_wm_recovery_shadow_tol_threshold 0.04 \
  --seq_wm_lambda_horizon_success 0.5 \
  --seq_wm_lambda_drop_risk 0.5 \
  --seq_wm_lambda_recovery 1.0 \
  --seq_wm_enable_iterative_teacher \
  --seq_wm_teacher_eval_interval_frames 50000000 \
  --seq_wm_teacher_promote_min_improvement 0.02 \
  --seq_wm_teacher_keep_last_k 6 \
  --seq_wm_teacher_eval_buffer_capacity 8 \
  --seq_wm_teacher_record_below_tol 0.04 \
  --seq_wm_teacher_require_recovery_pos_low 0.001 \
  --seq_wm_teacher_require_recovery_pos_high 0.999 \
  --seq_wm_teacher_allow_first_promotion \
  --seq_wm_teacher_min_eval_size 512 \
  --seq_wm_teacher_max_uncertainty 0.5 \
  --seq_wm_teacher_pos_rate_low 0.001 \
  --seq_wm_teacher_pos_rate_high 0.999 \
  --seq_wm_enable_imagined_evaluator \
  --seq_wm_imagine_horizons 4,8 \
  --seq_wm_imagine_action_mode recorded \
  --seq_wm_imagine_num_starts 256 \
  --seq_wm_imagine_use_teacher \
  --seq_wm_replay_prioritize_floor_recovery \
  --eir_enable \
  --eir_use_elite_bank \
  --eir_use_recovery_residual \
  --eir_use_rssm_mppi \
  --eir_horizons 4,8 \
  --eir_local_candidate_k 16 \
  --eir_elite_candidate_k 32 \
  --eir_archive_candidate_k 16 \
  --eir_action_noise_std 0.15 \
  --eir_max_action_delta 0.5 \
  --eir_min_score_margin 0.02 \
  --eir_min_elite_margin 0.01 \
  --eir_max_drop_prob_for_teacher 0.35 \
  --eir_start_tol 0.04 \
  --eir_floor_tol 0.02 \
  --eir_hard_floor_tol 0.012 \
  --eir_scorer_loss_weight 0.01 \
  --eir_start_after_frames "$EIR_START_AFTER_FRAMES" \
  --eir_scorer_warmup_iters "$EIR_SCORER_WARMUP_ITERS" \
  --eir_residual_scale 0.5 \
  --eir_residual_distill_weight 0.05 \
  --eir_residual_floor_distill_weight 0.08 \
  --eir_residual_hard_floor_distill_weight 0.12 \
  --eir_mu_base_detach \
  --eir_residual_apply_to_rollout \
  --eir_residual_apply_after_frames "$EIR_RESIDUAL_APPLY_AFTER" \
  --eir_residual_apply_alpha_mid 0.25 \
  --eir_residual_apply_alpha_floor 0.5 \
  --eir_residual_apply_alpha_hard_floor 0.75 \
  --eir_mppi_mode full \
  --eir_mppi_horizon 4 \
  --eir_mppi_num_sequences 64 \
  --eir_mppi_num_elite 8 \
  --eir_mppi_cem_iters 1 \
  --eir_mppi_action_noise_std 0.10 \
  --eir_mppi_max_action_delta 0.5 \
  --eir_mppi_uncertainty_threshold 0.4 \
  --eir_mppi_min_sequence_margin 0.02 \
  --eir_mppi_start_after_frames "$EIR_MPPI_START_AFTER" \
  --eir_pbt_broadcast_elite_bank \
  --eir_pbt_broadcast_threshold_nsucc_mean 38 \
  --eir_pbt_broadcast_to_bottom_k 3 \
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
  >"$LOG" 2>&1 &

echo $! > "$PIDFILE"
echo "launch_master pid=$(cat "$PIDFILE") log=$LOG"
