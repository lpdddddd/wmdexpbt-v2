"""PBT-WM Strategy Zoo v1.0/v1.2 — per-worker WM usage profiles.

Each profile expands to a full set of train.py CLI flag values.  ``legacy``
is a no-op so existing launchers remain unchanged.

``ppo_only`` is the PPO/PBT anchor under the shared strategy-zoo
PBT/curriculum scaffold.  It disables WM/EIR/action-level intervention, but
it is **not** an exact original DexPBT reproduction.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

PROFILE_IDS: dict[str, int] = {
    "legacy": 0,
    "ppo_only": 1,
    "rssm_mppi_actor_distill": 2,
    "wm_curriculum_controller": 3,
    "wm_pbt_source_selector": 4,
    "elite_bank_actor_distill": 5,
    "conservative_rssm_planner_distill": 6,
    # Strategy Zoo v1.2 performance-debug profiles.
    "light_rssm_mppi_actor_distill": 7,
    "wm_pbt_source_selector_v2": 8,
    "light_conservative_rssm_planner_distill": 9,
    # Strategy Zoo v1.3 elite-centered performance profiles.
    "elite_bank_actor_distill_fast": 10,
    "elite_bank_actor_distill_strong": 11,
    # Strategy Zoo v1.4 elite distill rescue profiles.
    "elite_bank_actor_distill_fast_v2": 12,
    "elite_bank_actor_distill_strong_v2": 13,
}

PROFILE_DESCRIPTIONS: dict[str, str] = {
    "legacy": "No profile expansion; legacy launcher behaviour.",
    "ppo_only": (
        "PPO/PBT anchor under the shared strategy-zoo PBT/curriculum scaffold. "
        "Disables WM/EIR/action-level intervention; not an exact original DexPBT "
        "reproduction."
    ),
    "rssm_mppi_actor_distill": "RSSM ensemble + MPPI candidates + actor distillation.",
    "wm_curriculum_controller": "Imagined-risk WM curriculum controller only.",
    "wm_pbt_source_selector": "Active WM score in PBT floor-recovery ranking.",
    "elite_bank_actor_distill": "Elite/archive guidance + actor distillation (no MPPI).",
    "conservative_rssm_planner_distill": "Conservative RSSM-MPPI + RSSM-only actor distill.",
    "light_rssm_mppi_actor_distill": (
        "Light RSSM-MPPI teacher-candidate provider + actor distill (v1.2)."
    ),
    "wm_pbt_source_selector_v2": (
        "Conservative WM PBT source selector (stricter soft gate, v1.2)."
    ),
    "light_conservative_rssm_planner_distill": (
        "Light conservative RSSM-MPPI planner + RSSM-only actor distill (v1.2)."
    ),
    "elite_bank_actor_distill_fast": (
        "Early elite/archive actor distill for v1.3 performance runs (no MPPI)."
    ),
    "elite_bank_actor_distill_strong": (
        "Strong elite/archive actor distill for v1.3 performance runs (no MPPI)."
    ),
    "elite_bank_actor_distill_fast_v2": (
        "v1.4 elite distill rescue: clip large delta, permissive margin (no MPPI)."
    ),
    "elite_bank_actor_distill_strong_v2": (
        "v1.4 strong elite distill rescue: clip large delta, higher weight (no MPPI)."
    ),
}

PROFILE_FAMILY: dict[str, str] = {
    "legacy": "legacy",
    "ppo_only": "anchor",
    "rssm_mppi_actor_distill": "actor_distill",
    "wm_curriculum_controller": "curriculum",
    "wm_pbt_source_selector": "pbt_selector",
    "elite_bank_actor_distill": "elite_imitation",
    "conservative_rssm_planner_distill": "planner",
    "light_rssm_mppi_actor_distill": "actor_distill",
    "wm_pbt_source_selector_v2": "pbt_selector",
    "light_conservative_rssm_planner_distill": "planner",
    "elite_bank_actor_distill_fast": "elite_imitation",
    "elite_bank_actor_distill_strong": "elite_imitation",
    "elite_bank_actor_distill_fast_v2": "elite_imitation",
    "elite_bank_actor_distill_strong_v2": "elite_imitation",
}

# Profile-copy source bonus weights (v1.4 selector-preserving).
PROFILE_COPY_SOURCE_BONUS: dict[str, float] = {
    "elite_bank_actor_distill_fast_v2": 1.0,
    "elite_bank_actor_distill": 1.1,
    "elite_bank_actor_distill_strong_v2": 1.1,
    "elite_bank_actor_distill_fast": 1.0,
    "elite_bank_actor_distill_strong": 1.0,
    "wm_pbt_source_selector": 1.3,
    "wm_pbt_source_selector_v2": 1.1,
    "ppo_only": 0.6,
    "light_rssm_mppi_actor_distill": 0.5,
    "light_conservative_rssm_planner_distill": 0.5,
    "rssm_mppi_actor_distill": 0.5,
    "conservative_rssm_planner_distill": 0.5,
    "wm_curriculum_controller": 0.8,
}

# Shared guard: no profile may turn on residual rollout apply in v1 zoo.
_NO_RESIDUAL_APPLY: dict[str, Any] = {
    "eir_residual_apply_to_rollout": False,
}

# ---------------------------------------------------------------------------
# Profile expansion tables (dest -> value)
# ---------------------------------------------------------------------------

_PPO_ONLY: dict[str, Any] = {
    **_NO_RESIDUAL_APPLY,
    "enable_wm": False,
    "wm_strategy_mode": "fixed",
    "fixed_wm_strategy": "none",
    "wm_strategy": "none",
    "eir_enable": False,
    "erd_enable": False,
    "wm_floorcritic_enable": False,
    "pbt_wm_score_mode": "off",
    "pbt_use_wm_recovery_score": False,
    "eir_actor_distill_enable": False,
    "wm_curriculum_controller_enable": False,
    # Soft Activation Gate v1: anchor never activates any WM soft gate.
    "wm_activation_enable": False,
}

_RSSM_MPPI_ACTOR_DISTILL: dict[str, Any] = {
    **_NO_RESIDUAL_APPLY,
    "enable_wm": True,
    "wm_strategy": "rssm_belief_value",
    "wm_strategy_mode": "fixed",
    "fixed_wm_strategy": "rssm_belief_value",
    "wm_load_mode": "auto",
    "seq_wm_seq_len": 16,
    "seq_wm_seq_batch_size": 1024,
    "seq_wm_lambda_total": 0.05,
    "seq_wm_lambda_value": 0.1,
    "seq_wm_value_detach_belief": False,
    "seq_wm_policy_coupling": "none",
    "seq_wm_use_task_aux_inputs": True,
    "seq_wm_loss_preset": "task_focused",
    "seq_wm_floor_aware_weighting": True,
    "seq_wm_use_horizon_heads": True,
    "seq_wm_horizon_steps": "4,8",
    "seq_wm_use_recovery_head": True,
    "seq_wm_recovery_horizon": 8,
    "seq_wm_ensemble_enable": True,
    "seq_wm_ensemble_size": 3,
    "seq_wm_ensemble_bootstrap": True,
    "seq_wm_ensemble_bootstrap_keep_prob": 0.8,
    "seq_wm_ensemble_load_mode": "resume",
    "seq_wm_ensemble_save": True,
    "eir_enable": True,
    "eir_use_elite_bank": True,
    "eir_use_recovery_residual": True,
    "eir_use_rssm_mppi": True,
    "eir_mppi_mode": "full",
    "eir_mppi_horizon": 4,
    "eir_mppi_num_sequences": 32,
    "eir_mppi_num_elite": 4,
    "eir_mppi_cem_iters": 1,
    "eir_mppi_stochastic_rollouts": True,
    "eir_mppi_num_latent_samples": 2,
    "eir_mppi_score_aggregation": "mean_minus_std",
    "eir_mppi_risk_beta": 0.5,
    "eir_mppi_drop_cvar_weight": 1.0,
    "eir_mppi_uncertainty_cvar_weight": 0.2,
    "eir_mppi_use_model_ensemble": True,
    "eir_mppi_ensemble_risk_beta": 0.5,
    "eir_mppi_ensemble_drop_weight": 0.5,
    "eir_mppi_ensemble_uncertainty_weight": 0.2,
    "eir_mppi_ensemble_min_valid_models": 2,
    "eir_mppi_ensemble_max_models_per_plan": 2,
    "eir_mppi_ensemble_uncertainty_threshold": 2.0,
    "eir_mppi_ensemble_uncertainty_allow_margin": 0.10,
    "eir_actor_distill_enable": True,
    "eir_actor_distill_weight": 0.001,
    "eir_actor_distill_loss_cap": 0.05,
    "eir_actor_distill_start_after_frames": 500_000_000,
    "eir_actor_distill_tol": 0.04,
    "eir_actor_distill_min_margin": 0.02,
    "eir_actor_distill_margin_norm": 0.05,
    "eir_actor_distill_max_delta": 0.35,
    "eir_actor_distill_allowed_sources": "rssm,elite,archive",
    "eir_actor_distill_uncertainty_threshold": 2.0,
    "eir_actor_distill_feature_detach": True,
    "eir_actor_distill_mu_head_only": True,
    # Soft Activation Gate v1: ramp actor-distill in from tol 0.060 -> 0.040.
    "wm_activation_enable": True,
    "eir_actor_distill_gate_mode": "soft_tol",
    "eir_actor_distill_tol_start": 0.060,
    "eir_actor_distill_tol_full": 0.040,
    "eir_actor_distill_ramp_frames": 300_000_000,
    "eir_actor_distill_nsucc_ref": 20.0,
    "eir_candidate_gate_mode": "profile_soft",
    "eir_rssm_candidate_tol_start": 0.060,
    "eir_rssm_candidate_tol_full": 0.040,
    "eir_candidate_min_weight_for_retrieval": 0.05,
    "seq_wm_head_balance_enable": True,
    "seq_wm_head_pos_weight_auto": True,
    "seq_wm_lambda_horizon_success": 0.5,
    "seq_wm_lambda_drop_risk": 0.5,
    "seq_wm_lambda_recovery": 0.5,
    "seq_wm_replay_prioritize_floor_recovery": True,
    "pbt_wm_score_mode": "off",
    "pbt_use_wm_recovery_score": False,
    "wm_curriculum_controller_enable": False,
}

_WM_CURRICULUM_CONTROLLER: dict[str, Any] = {
    **_NO_RESIDUAL_APPLY,
    "enable_wm": True,
    "wm_strategy": "rssm_belief_value",
    "wm_strategy_mode": "fixed",
    "fixed_wm_strategy": "rssm_belief_value",
    "seq_wm_use_task_aux_inputs": True,
    "seq_wm_loss_preset": "task_focused",
    "seq_wm_use_horizon_heads": True,
    "seq_wm_horizon_steps": "4,8",
    "seq_wm_use_recovery_head": True,
    "seq_wm_enable_imagined_evaluator": True,
    "seq_wm_imagine_horizons": "4,8",
    "seq_wm_imagine_action_mode": "recorded",
    "seq_wm_imagine_num_starts": 256,
    "seq_wm_imagine_use_teacher": True,
    "seq_wm_enable_iterative_teacher": True,
    "wm_curriculum_controller_enable": True,
    "wm_curriculum_controller_start_after_frames": 500_000_000,
    "wm_curriculum_controller_drop_risk_threshold": 0.45,
    "wm_curriculum_controller_uncertainty_threshold": 0.55,
    "wm_curriculum_controller_min_success_prob": 0.20,
    "wm_curriculum_controller_freeze_frames": 100_000_000,
    "wm_curriculum_controller_min_nsucc_mean": 10.0,
    "wm_curriculum_controller_hard_block_shrink": True,
    "wm_curriculum_controller_use_recovery_prob": False,
    # Soft Activation Gate v1: controller is a shrink-candidate gate (fail-open).
    "wm_activation_enable": True,
    "wm_curriculum_gate_mode": "shrink_candidate",
    "eir_enable": False,
    "eir_actor_distill_enable": False,
    "pbt_wm_score_mode": "off",
    "pbt_use_wm_recovery_score": False,
}

_WM_PBT_SOURCE_SELECTOR: dict[str, Any] = {
    **_NO_RESIDUAL_APPLY,
    "enable_wm": True,
    "wm_strategy": "rssm_belief_value",
    "wm_strategy_mode": "fixed",
    "fixed_wm_strategy": "rssm_belief_value",
    "seq_wm_use_task_aux_inputs": True,
    "seq_wm_loss_preset": "task_focused",
    "seq_wm_floor_aware_weighting": True,
    "seq_wm_use_horizon_heads": True,
    "seq_wm_horizon_steps": "4,8",
    "seq_wm_use_recovery_head": True,
    "seq_wm_recovery_horizon": 8,
    "seq_wm_enable_iterative_teacher": True,
    "seq_wm_teacher_eval_interval_frames": 50_000_000,
    "seq_wm_teacher_promote_min_improvement": 0.02,
    "seq_wm_teacher_keep_last_k": 6,
    "seq_wm_teacher_eval_buffer_capacity": 8,
    "seq_wm_teacher_record_below_tol": 0.04,
    "seq_wm_teacher_max_uncertainty": 0.5,
    "seq_wm_enable_imagined_evaluator": True,
    "seq_wm_imagine_horizons": "4,8",
    "seq_wm_imagine_action_mode": "recorded",
    "seq_wm_imagine_num_starts": 256,
    "seq_wm_imagine_use_teacher": True,
    "pbt_wm_score_mode": "active",
    "pbt_use_wm_recovery_score": True,
    "pbt_wm_recovery_weight": 0.3,
    "pbt_wm_success_weight": 0.2,
    "pbt_wm_drop_penalty": 0.3,
    "pbt_wm_uncertainty_penalty": 0.2,
    "pbt_wm_max_uncertainty": 0.5,
    "pbt_floor_recovery_aware": True,
    "pbt_enable_best_floor_recovery_archive": True,
    "wm_imagined_valid_mode": "soft_calibrated",
    "wm_imagined_valid_uncertainty_threshold": 0.5,
    "wm_imagined_soft_uncertainty_threshold": 0.95,
    "wm_imagined_min_logit_abs_mean": 0.05,
    "wm_imagined_min_head_signal_std": 0.02,
    "wm_imagined_soft_validity_weight": 0.25,
    # Soft Activation Gate v1: WM PBT source score uses a confidence-only gate
    # (eval_valid + time ramp + uncertainty), NOT a tol hard gate.
    "wm_activation_enable": True,
    "wm_pbt_source_gate_mode": "confidence_only",
    "wm_pbt_source_start_after_frames": 300_000_000,
    "wm_pbt_source_ramp_frames": 300_000_000,
    "eir_enable": False,
    "eir_actor_distill_enable": False,
    "wm_curriculum_controller_enable": False,
}

_ELITE_BANK_ACTOR_DISTILL: dict[str, Any] = {
    **_NO_RESIDUAL_APPLY,
    "enable_wm": True,
    "wm_strategy": "rssm_belief_value",
    "wm_strategy_mode": "fixed",
    "fixed_wm_strategy": "rssm_belief_value",
    "eir_enable": True,
    "eir_use_elite_bank": True,
    "eir_use_recovery_residual": True,
    "eir_use_rssm_mppi": False,
    "eir_mppi_mode": "off",
    "eir_guidance_enable": True,
    "eir_guidance_start_tol": 0.04,
    "eir_guidance_bank_tol": 0.04,
    "eir_guidance_apply_tol": 0.04,
    "eir_guidance_mid_near_success_mult": 1.5,
    "eir_guidance_mid_priority_scale": 0.5,
    "eir_local_candidate_k": 0,
    "eir_elite_candidate_k": 64,
    "eir_archive_candidate_k": 32,
    "eir_action_noise_std": 0.0,
    "eir_min_score_margin": 0.02,
    "eir_min_elite_margin": 0.01,
    "eir_max_drop_prob_for_teacher": 0.35,
    "eir_actor_distill_enable": True,
    "eir_actor_distill_weight": 0.001,
    "eir_actor_distill_loss_cap": 0.05,
    "eir_actor_distill_start_after_frames": 300_000_000,
    "eir_actor_distill_tol": 0.04,
    "eir_actor_distill_min_margin": 0.01,
    "eir_actor_distill_margin_norm": 0.05,
    "eir_actor_distill_max_delta": 0.35,
    "eir_actor_distill_allowed_sources": "elite,archive",
    "eir_actor_distill_feature_detach": True,
    "eir_actor_distill_mu_head_only": True,
    # Soft Activation Gate v1: elite distill softens EARLIER than RSSM-MPPI
    # (tol 0.070 -> 0.045, shorter 200M ramp, lower nsucc_ref).
    "wm_activation_enable": True,
    "elite_distill_gate_mode": "soft_tol",
    "elite_distill_tol_start": 0.070,
    "elite_distill_tol_full": 0.045,
    "elite_distill_ramp_frames": 200_000_000,
    "elite_distill_nsucc_ref": 15.0,
    "eir_actor_distill_gate_mode": "soft_tol",
    "eir_actor_distill_tol_start": 0.070,
    "eir_actor_distill_tol_full": 0.045,
    "eir_actor_distill_ramp_frames": 200_000_000,
    "eir_actor_distill_nsucc_ref": 15.0,
    "eir_candidate_gate_mode": "profile_soft",
    "eir_elite_candidate_tol_start": 0.070,
    "eir_elite_candidate_tol_full": 0.045,
    "eir_candidate_min_weight_for_retrieval": 0.05,
    "seq_wm_head_balance_enable": True,
    "seq_wm_head_pos_weight_auto": True,
    "seq_wm_lambda_horizon_success": 0.5,
    "seq_wm_lambda_drop_risk": 0.5,
    "seq_wm_lambda_recovery": 0.5,
    "seq_wm_replay_prioritize_floor_recovery": True,
    "eir_pbt_broadcast_elite_bank": True,
    "eir_pbt_broadcast_threshold_nsucc_mean": 25,
    "eir_pbt_broadcast_to_bottom_k": 3,
    "pbt_wm_score_mode": "off",
    "pbt_use_wm_recovery_score": False,
    "wm_curriculum_controller_enable": False,
}

_CONSERVATIVE_RSSM_PLANNER: dict[str, Any] = {
    **_NO_RESIDUAL_APPLY,
    "enable_wm": True,
    "wm_strategy": "rssm_belief_value",
    "wm_strategy_mode": "fixed",
    "fixed_wm_strategy": "rssm_belief_value",
    "seq_wm_use_task_aux_inputs": True,
    "seq_wm_loss_preset": "task_focused",
    "seq_wm_use_horizon_heads": True,
    "seq_wm_horizon_steps": "4,8",
    "seq_wm_use_recovery_head": True,
    "seq_wm_ensemble_enable": True,
    "seq_wm_ensemble_size": 3,
    "seq_wm_ensemble_bootstrap": True,
    "seq_wm_ensemble_bootstrap_keep_prob": 0.8,
    "seq_wm_ensemble_load_mode": "resume",
    "eir_enable": True,
    "eir_use_elite_bank": True,
    "eir_use_recovery_residual": True,
    "eir_use_rssm_mppi": True,
    "eir_mppi_mode": "full",
    "eir_mppi_horizon": 4,
    "eir_mppi_num_sequences": 32,
    "eir_mppi_num_elite": 4,
    "eir_mppi_cem_iters": 1,
    "eir_mppi_action_noise_std": 0.10,
    "eir_mppi_max_action_delta": 0.35,
    "eir_mppi_min_sequence_margin": 0.02,
    "eir_mppi_stochastic_rollouts": True,
    "eir_mppi_num_latent_samples": 2,
    "eir_mppi_score_aggregation": "mean_minus_std",
    "eir_mppi_risk_beta": 0.7,
    "eir_mppi_drop_cvar_alpha": 0.25,
    "eir_mppi_drop_cvar_weight": 1.2,
    "eir_mppi_uncertainty_cvar_weight": 0.4,
    "eir_mppi_use_model_ensemble": True,
    "eir_mppi_ensemble_risk_beta": 0.7,
    "eir_mppi_ensemble_drop_weight": 0.8,
    "eir_mppi_ensemble_uncertainty_weight": 0.4,
    "eir_mppi_ensemble_min_valid_models": 2,
    "eir_mppi_ensemble_max_models_per_plan": 2,
    "eir_mppi_ensemble_uncertainty_threshold": 1.5,
    "eir_mppi_ensemble_uncertainty_allow_margin": 0.12,
    "eir_guidance_enable": True,
    "eir_guidance_rssm_source_bonus": 0.05,
    "eir_guidance_rssm_bonus_margin_scale": 1.0,
    "eir_guidance_rssm_bonus_uncertainty_penalty": 0.02,
    "eir_actor_distill_enable": True,
    "eir_actor_distill_weight": 0.001,
    "eir_actor_distill_loss_cap": 0.05,
    "eir_actor_distill_start_after_frames": 500_000_000,
    "eir_actor_distill_tol": 0.04,
    "eir_actor_distill_min_margin": 0.015,
    "eir_actor_distill_margin_norm": 0.05,
    "eir_actor_distill_max_delta": 0.30,
    "eir_actor_distill_allowed_sources": "rssm",
    "eir_actor_distill_uncertainty_threshold": 1.5,
    "eir_actor_distill_feature_detach": True,
    "eir_actor_distill_mu_head_only": True,
    # Soft Activation Gate v1: conservative planner softens LATER and stricter
    # (tol 0.055 -> 0.035, 300M ramp, higher nsucc_ref, stricter uncertainty).
    "wm_activation_enable": True,
    "conservative_planner_gate_mode": "soft_tol",
    "conservative_planner_tol_start": 0.055,
    "conservative_planner_tol_full": 0.035,
    "conservative_planner_ramp_frames": 300_000_000,
    "conservative_planner_nsucc_ref": 25.0,
    "eir_actor_distill_gate_mode": "soft_tol",
    "eir_actor_distill_tol_start": 0.055,
    "eir_actor_distill_tol_full": 0.035,
    "eir_actor_distill_ramp_frames": 300_000_000,
    "eir_actor_distill_nsucc_ref": 25.0,
    "eir_candidate_gate_mode": "profile_soft",
    "eir_conservative_rssm_candidate_tol_start": 0.055,
    "eir_conservative_rssm_candidate_tol_full": 0.035,
    "eir_candidate_min_weight_for_retrieval": 0.05,
    "seq_wm_head_balance_enable": True,
    "seq_wm_head_pos_weight_auto": True,
    "seq_wm_lambda_horizon_success": 0.5,
    "seq_wm_lambda_drop_risk": 0.5,
    "seq_wm_lambda_recovery": 0.5,
    "seq_wm_replay_prioritize_floor_recovery": True,
    "pbt_wm_score_mode": "off",
    "pbt_use_wm_recovery_score": False,
    "wm_curriculum_controller_enable": False,
}

_LIGHT_RSSM_MPPI_ACTOR_DISTILL: dict[str, Any] = {
    **_RSSM_MPPI_ACTOR_DISTILL,
    "eir_mppi_num_sequences": 16,
    "eir_mppi_num_elite": 2,
    "eir_mppi_num_latent_samples": 1,
    "eir_mppi_use_model_ensemble": False,
    "eir_mppi_ensemble_max_models_per_plan": 1,
    "eir_mppi_ensemble_min_valid_models": 1,
    "eir_mppi_drop_cvar_weight": 0.5,
    "eir_mppi_uncertainty_cvar_weight": 0.1,
    "eir_mppi_min_sequence_margin": 0.02,
    "eir_actor_distill_start_after_frames": 300_000_000,
    "eir_actor_distill_tol_start": 0.065,
    "eir_actor_distill_tol_full": 0.040,
    "eir_actor_distill_min_margin": 0.01,
    "eir_rssm_candidate_tol_start": 0.065,
    "eir_elite_candidate_tol_start": 0.070,
    "eir_elite_candidate_tol_full": 0.045,
}

_WM_PBT_SOURCE_SELECTOR_V2: dict[str, Any] = {
    **_WM_PBT_SOURCE_SELECTOR,
    "pbt_wm_recovery_weight": 0.20,
    "pbt_wm_success_weight": 0.15,
    "pbt_wm_drop_penalty": 0.25,
    "pbt_wm_uncertainty_penalty": 0.20,
    "wm_imagined_soft_uncertainty_threshold": 0.90,
    "wm_imagined_min_logit_abs_mean": 0.10,
    "wm_imagined_min_head_signal_std": 0.05,
    "wm_imagined_soft_validity_weight": 0.10,
}

_LIGHT_CONSERVATIVE_RSSM_PLANNER: dict[str, Any] = {
    **_CONSERVATIVE_RSSM_PLANNER,
    "eir_mppi_num_sequences": 24,
    "eir_mppi_num_elite": 2,
    "eir_mppi_num_latent_samples": 1,
    "eir_mppi_use_model_ensemble": False,
    "eir_mppi_ensemble_max_models_per_plan": 1,
    "eir_mppi_ensemble_min_valid_models": 1,
    "eir_mppi_risk_beta": 0.7,
    "eir_mppi_drop_cvar_weight": 0.8,
    "eir_mppi_uncertainty_cvar_weight": 0.2,
    "eir_mppi_min_sequence_margin": 0.015,
    "eir_actor_distill_uncertainty_threshold": 1.5,
    "eir_actor_distill_min_margin": 0.01,
    "eir_actor_distill_max_delta": 0.30,
    "eir_actor_distill_allowed_sources": "rssm",
    "eir_actor_distill_start_after_frames": 400_000_000,
    "eir_actor_distill_tol_start": 0.060,
    "eir_actor_distill_tol_full": 0.035,
    "eir_actor_distill_nsucc_ref": 20.0,
    "conservative_planner_tol_start": 0.060,
    "conservative_planner_tol_full": 0.035,
    "conservative_planner_nsucc_ref": 20.0,
    "eir_conservative_rssm_candidate_tol_start": 0.060,
    "eir_conservative_rssm_candidate_tol_full": 0.035,
}

_ELITE_BANK_ACTOR_DISTILL_FAST: dict[str, Any] = {
    **_ELITE_BANK_ACTOR_DISTILL,
    "enable_wm": True,
    "eir_enable": True,
    "eir_use_elite_bank": True,
    "eir_use_recovery_residual": True,
    "eir_use_rssm_mppi": False,
    "eir_mppi_mode": "off",
    "eir_elite_candidate_k": 64,
    "eir_archive_candidate_k": 32,
    "eir_actor_distill_enable": True,
    "eir_actor_distill_start_after_frames": 200_000_000,
    "eir_actor_distill_gate_mode": "soft_tol",
    "eir_actor_distill_tol_start": 0.075,
    "eir_actor_distill_tol_full": 0.050,
    "eir_actor_distill_min_margin": 0.005,
    "eir_actor_distill_allowed_sources": "elite,archive",
    "eir_actor_distill_weight": 0.001,
    "eir_actor_distill_loss_cap": 0.05,
    "eir_candidate_gate_mode": "profile_soft",
    "eir_elite_candidate_tol_start": 0.075,
    "eir_elite_candidate_tol_full": 0.050,
    "eir_residual_apply_to_rollout": False,
}

_ELITE_BANK_ACTOR_DISTILL_STRONG: dict[str, Any] = {
    **_ELITE_BANK_ACTOR_DISTILL,
    "enable_wm": True,
    "eir_enable": True,
    "eir_use_elite_bank": True,
    "eir_use_recovery_residual": True,
    "eir_use_rssm_mppi": False,
    "eir_mppi_mode": "off",
    "eir_elite_candidate_k": 96,
    "eir_archive_candidate_k": 64,
    "eir_actor_distill_enable": True,
    "eir_actor_distill_start_after_frames": 300_000_000,
    "eir_actor_distill_gate_mode": "soft_tol",
    "eir_actor_distill_tol_start": 0.065,
    "eir_actor_distill_tol_full": 0.040,
    "eir_actor_distill_min_margin": 0.005,
    "eir_actor_distill_allowed_sources": "elite,archive",
    "eir_actor_distill_weight": 0.0015,
    "eir_actor_distill_loss_cap": 0.05,
    "eir_candidate_gate_mode": "profile_soft",
    "eir_elite_candidate_tol_start": 0.065,
    "eir_elite_candidate_tol_full": 0.040,
    "eir_residual_apply_to_rollout": False,
}

_ELITE_BANK_ACTOR_DISTILL_FAST_V2: dict[str, Any] = {
    **_ELITE_BANK_ACTOR_DISTILL_FAST,
    "eir_actor_distill_large_delta_mode": "clip",
    "eir_actor_distill_max_delta": 1.0,
    "eir_actor_distill_target_delta_clip": 0.35,
    "eir_actor_distill_min_margin": 0.0,
    "eir_actor_distill_weight": 0.002,
    "eir_actor_distill_loss_cap": 0.03,
    "eir_residual_apply_to_rollout": False,
    # Strategy Zoo v1.5 clip-validation: temporarily widen the guidance/bank band
    # to 0.06 so the elite bank can fill (and teachers/soft-gate engage) before
    # tol reaches the 0.04 floor. Diagnostic only; residual rollout stays OFF.
    "eir_guidance_start_tol": 0.06,
    "eir_guidance_bank_tol": 0.06,
    "eir_guidance_apply_tol": 0.06,
}

_ELITE_BANK_ACTOR_DISTILL_STRONG_V2: dict[str, Any] = {
    **_ELITE_BANK_ACTOR_DISTILL_STRONG,
    "eir_actor_distill_large_delta_mode": "clip",
    "eir_actor_distill_max_delta": 1.0,
    "eir_actor_distill_target_delta_clip": 0.50,
    "eir_actor_distill_min_margin": 0.0,
    "eir_actor_distill_weight": 0.003,
    "eir_actor_distill_loss_cap": 0.03,
    "eir_residual_apply_to_rollout": False,
    # Strategy Zoo v1.5 clip-validation: same diagnostic widening as fast_v2.
    "eir_guidance_start_tol": 0.06,
    "eir_guidance_bank_tol": 0.06,
    "eir_guidance_apply_tol": 0.06,
}

PROFILE_TABLES: dict[str, dict[str, Any]] = {
    "legacy": {},
    "ppo_only": _PPO_ONLY,
    "rssm_mppi_actor_distill": _RSSM_MPPI_ACTOR_DISTILL,
    "wm_curriculum_controller": _WM_CURRICULUM_CONTROLLER,
    "wm_pbt_source_selector": _WM_PBT_SOURCE_SELECTOR,
    "elite_bank_actor_distill": _ELITE_BANK_ACTOR_DISTILL,
    "conservative_rssm_planner_distill": _CONSERVATIVE_RSSM_PLANNER,
    "light_rssm_mppi_actor_distill": _LIGHT_RSSM_MPPI_ACTOR_DISTILL,
    "wm_pbt_source_selector_v2": _WM_PBT_SOURCE_SELECTOR_V2,
    "light_conservative_rssm_planner_distill": _LIGHT_CONSERVATIVE_RSSM_PLANNER,
    "elite_bank_actor_distill_fast": _ELITE_BANK_ACTOR_DISTILL_FAST,
    "elite_bank_actor_distill_strong": _ELITE_BANK_ACTOR_DISTILL_STRONG,
    "elite_bank_actor_distill_fast_v2": _ELITE_BANK_ACTOR_DISTILL_FAST_V2,
    "elite_bank_actor_distill_strong_v2": _ELITE_BANK_ACTOR_DISTILL_STRONG_V2,
}

# Map CLI flag strings (without --) to argparse dest names for explicit detection.
_FLAG_TO_DEST: dict[str, str] = {}


def _register_flag(flag: str, dest: str | None = None) -> None:
    d = dest or flag.lstrip("-").replace("-", "_")
    _FLAG_TO_DEST[flag.lstrip("-")] = d
    if flag.startswith("--no-"):
        base = flag[5:]
        _FLAG_TO_DEST[base.lstrip("-")] = d


def _build_flag_registry() -> None:
    if _FLAG_TO_DEST:
        return
    all_dests: set[str] = set()
    for table in PROFILE_TABLES.values():
        all_dests.update(table.keys())
    all_dests.update(
        {
            "wm_strategy_profile",
            "enable_wm",
            "eir_enable",
            "erd_enable",
            "wm_floorcritic_enable",
            "pbt_wm_score_mode",
            "pbt_use_wm_recovery_score",
            "eir_residual_apply_to_rollout",
            "eir_actor_distill_enable",
            "wm_curriculum_controller_enable",
            "pbt_wm_profile_copy_enable",
            "curriculum_safety_enable",
            "curriculum_safety_use_real_fallback",
            "curriculum_strong_shrink_global_retention_gate",
            "curriculum_danger_zone_enable",
            "curriculum_danger_disable_strong_shrink",
            "curriculum_rollback_on_repeated_collapse",
            "curriculum_rollback_to_last_mastered",
            "wm_imagined_diagnostic_enable",
            "wm_imagined_shadow_block_enable",
            "seq_wm_head_balance_enable",
            "seq_wm_head_pos_weight_auto",
            "seq_wm_head_pos_weight_max",
            "seq_wm_success_pos_weight",
            "seq_wm_drop_pos_weight",
            "seq_wm_recovery_pos_weight",
            "wm_imagined_valid_mode",
            "wm_imagined_valid_uncertainty_threshold",
            "wm_imagined_soft_uncertainty_threshold",
            "wm_imagined_min_logit_abs_mean",
            "wm_imagined_min_head_signal_std",
            "wm_imagined_soft_validity_weight",
            "eir_candidate_gate_mode",
            "eir_elite_candidate_tol_start",
            "eir_elite_candidate_tol_full",
            "eir_rssm_candidate_tol_start",
            "eir_rssm_candidate_tol_full",
            "eir_conservative_rssm_candidate_tol_start",
            "eir_conservative_rssm_candidate_tol_full",
            "eir_candidate_min_weight_for_retrieval",
        }
    )
    for dest in all_dests:
        _register_flag(f"--{dest}", dest)
        if "_" in dest:
            _register_flag(f"--{dest.replace('_', '-')}", dest)
        _register_flag(f"--no-{dest}", dest)
        if "_" in dest:
            _register_flag(f"--no-{dest.replace('_', '-')}", dest)


def collect_explicit_dests(argv: list[str] | None = None) -> set[str]:
    """Return argparse dest names explicitly present on the command line."""
    _build_flag_registry()
    argv = argv if argv is not None else sys.argv
    explicit: set[str] = set()
    i = 0
    while i < len(argv):
        tok = argv[i]
        if not tok.startswith("--"):
            i += 1
            continue
        body = tok[2:]
        if body.startswith("no-"):
            base = body[3:]
            dest = _FLAG_TO_DEST.get(base)
            if dest:
                explicit.add(dest)
            i += 1
            continue
        if "=" in body:
            key = body.split("=", 1)[0]
            dest = _FLAG_TO_DEST.get(key)
            if dest:
                explicit.add(dest)
            i += 1
            continue
        dest = _FLAG_TO_DEST.get(body)
        if dest:
            explicit.add(dest)
            if dest not in (
                "enable_wm",
                "wm_strategy_profile",
            ) and i + 1 < len(argv) and not argv[i + 1].startswith("--"):
                i += 2
                continue
        i += 1
    return explicit


def profile_controlled_restart_skip_flags() -> frozenset[str]:
    """CLI flags stripped from PBT restart argv before profile re-expansion."""
    skip: set[str] = {
        "--wm_strategy_profile",
        "--wm_strategy_profile_generation_resume",
        "--wm_strategy_profile_parent_resume",
        "--wm_strategy_profile_copied_from_source_resume",
        "--wm_curriculum_controller_enable",
        "--no-wm_curriculum_controller_enable",
        "--wm_curriculum_controller_start_after_frames",
        "--wm_curriculum_controller_min_nsucc_mean",
        "--wm_curriculum_controller_drop_risk_threshold",
        "--wm_curriculum_controller_uncertainty_threshold",
        "--wm_curriculum_controller_min_success_prob",
        "--wm_curriculum_controller_freeze_frames",
        "--wm_curriculum_controller_hard_block_shrink",
        "--no-wm_curriculum_controller_hard_block_shrink",
        "--wm_curriculum_controller_use_recovery_prob",
        "--no-wm_curriculum_controller_use_recovery_prob",
        "--pbt_wm_profile_copy_enable",
        "--no-pbt_wm_profile_copy_enable",
        "--pbt_wm_profile_copy_after_frames",
        "--pbt_wm_profile_copy_prob",
        "--pbt_wm_profile_keep_diversity_min_profiles",
        "--pbt_wm_profile_copy_min_nsucc_gap",
        "--pbt_wm_profile_copy_min_source_retention",
        "--pbt_wm_profile_copy_cooldown_frames",
        "--pbt_wm_profile_copy_bottom_k",
        "--pbt_wm_profile_copy_top_k",
        "--pbt_wm_profile_copy_source_min_nsucc_mean",
        "--pbt_wm_profile_copy_source_min_frames",
        "--pbt_wm_profile_copy_top1_override_enable",
        "--no-pbt_wm_profile_copy_top1_override_enable",
        "--pbt_wm_profile_copy_top1_override_after_frames",
        "--pbt_wm_profile_copy_top1_override_min_nsucc_gap",
        "--pbt_wm_profile_copy_top1_override_min_diversity",
        "--pbt_wm_profile_copy_top1_override_allow_target_protected",
        "--no-pbt_wm_profile_copy_top1_override_allow_target_protected",
    }
    for dest in _all_profile_controlled_dests():
        skip.add(f"--{dest}")
        skip.add(f"--{dest.replace('_', '-')}")
        skip.add(f"--no-{dest}")
        skip.add(f"--no-{dest.replace('_', '-')}")
    return frozenset(skip)


def profile_copy_restart_train_args(
    *,
    enable: bool,
    after_frames: int,
    prob: float,
    keep_diversity_min_profiles: int,
    min_nsucc_gap: float = 0.0,
    min_source_retention: float = 0.0,
    cooldown_frames: int = 0,
    bottom_k: int = 0,
    top_k: int = 0,
    source_min_nsucc_mean: float = 0.0,
    source_min_frames: int = 0,
    top1_override_enable: bool = False,
    top1_override_after_frames: int = 0,
    top1_override_min_nsucc_gap: float = 5.0,
    top1_override_min_diversity: int = 2,
    top1_override_allow_target_protected: bool = False,
) -> list[str]:
    """Re-inject profile-copy CLI after restart argv stripping."""
    return [
        (
            "--pbt_wm_profile_copy_enable"
            if enable
            else "--no-pbt_wm_profile_copy_enable"
        ),
        f"--pbt_wm_profile_copy_after_frames={int(after_frames)}",
        f"--pbt_wm_profile_copy_prob={float(prob)}",
        (
            "--pbt_wm_profile_keep_diversity_min_profiles="
            f"{int(keep_diversity_min_profiles)}"
        ),
        f"--pbt_wm_profile_copy_min_nsucc_gap={float(min_nsucc_gap)}",
        f"--pbt_wm_profile_copy_min_source_retention={float(min_source_retention)}",
        f"--pbt_wm_profile_copy_cooldown_frames={int(cooldown_frames)}",
        f"--pbt_wm_profile_copy_bottom_k={int(bottom_k)}",
        f"--pbt_wm_profile_copy_top_k={int(top_k)}",
        f"--pbt_wm_profile_copy_source_min_nsucc_mean={float(source_min_nsucc_mean)}",
        f"--pbt_wm_profile_copy_source_min_frames={int(source_min_frames)}",
        (
            "--pbt_wm_profile_copy_top1_override_enable"
            if top1_override_enable
            else "--no-pbt_wm_profile_copy_top1_override_enable"
        ),
        (
            "--pbt_wm_profile_copy_top1_override_after_frames="
            f"{int(top1_override_after_frames)}"
        ),
        (
            "--pbt_wm_profile_copy_top1_override_min_nsucc_gap="
            f"{float(top1_override_min_nsucc_gap)}"
        ),
        (
            "--pbt_wm_profile_copy_top1_override_min_diversity="
            f"{int(top1_override_min_diversity)}"
        ),
        (
            "--pbt_wm_profile_copy_top1_override_allow_target_protected"
            if top1_override_allow_target_protected
            else "--no-pbt_wm_profile_copy_top1_override_allow_target_protected"
        ),
    ]


def _parse_profile_copy_config_from_metadata(
    metadata_path: Path,
) -> dict[str, Any] | None:
    """Read profile-copy flags from supervisor ``metadata.txt`` train_args."""
    if not metadata_path.is_file():
        return None
    tokens: list[str] = []
    for line in metadata_path.read_text().splitlines():
        if not line.startswith("train_args="):
            continue
        for part in line.split("train_args=")[1:]:
            tok = part.strip()
            if tok:
                tokens.append(tok)
    if not tokens:
        return None

    enable: bool | None = None
    after_frames: int | None = None
    prob: float | None = None
    keep_diversity_min_profiles: int | None = None
    min_nsucc_gap: float | None = None
    min_source_retention: float | None = None
    cooldown_frames: int | None = None
    bottom_k: int | None = None
    top_k: int | None = None
    for tok in tokens:
        if tok == "--pbt_wm_profile_copy_enable":
            enable = True
        elif tok == "--no-pbt_wm_profile_copy_enable":
            enable = False
        elif tok.startswith("--pbt_wm_profile_copy_after_frames="):
            after_frames = int(tok.split("=", 1)[1])
        elif tok.startswith("--pbt_wm_profile_copy_prob="):
            prob = float(tok.split("=", 1)[1])
        elif tok.startswith("--pbt_wm_profile_keep_diversity_min_profiles="):
            keep_diversity_min_profiles = int(tok.split("=", 1)[1])
        elif tok.startswith("--pbt_wm_profile_copy_min_nsucc_gap="):
            min_nsucc_gap = float(tok.split("=", 1)[1])
        elif tok.startswith("--pbt_wm_profile_copy_min_source_retention="):
            min_source_retention = float(tok.split("=", 1)[1])
        elif tok.startswith("--pbt_wm_profile_copy_cooldown_frames="):
            cooldown_frames = int(tok.split("=", 1)[1])
        elif tok.startswith("--pbt_wm_profile_copy_bottom_k="):
            bottom_k = int(tok.split("=", 1)[1])
        elif tok.startswith("--pbt_wm_profile_copy_top_k="):
            top_k = int(tok.split("=", 1)[1])
    if enable is None and after_frames is None and prob is None:
        return None
    return {
        "enable": bool(enable) if enable is not None else False,
        "after_frames": int(after_frames if after_frames is not None else 1_000_000_000),
        "prob": float(prob if prob is not None else 0.3),
        "keep_diversity_min_profiles": int(
            keep_diversity_min_profiles
            if keep_diversity_min_profiles is not None
            else 3
        ),
        "min_nsucc_gap": float(min_nsucc_gap if min_nsucc_gap is not None else 0.0),
        "min_source_retention": float(
            min_source_retention if min_source_retention is not None else 0.0
        ),
        "cooldown_frames": int(cooldown_frames if cooldown_frames is not None else 0),
        "bottom_k": int(bottom_k if bottom_k is not None else 0),
        "top_k": int(top_k if top_k is not None else 0),
    }


def resolve_profile_copy_restart_config(
    *,
    enable: bool,
    after_frames: int,
    prob: float,
    keep_diversity_min_profiles: int,
    min_nsucc_gap: float = 0.0,
    min_source_retention: float = 0.0,
    cooldown_frames: int = 0,
    bottom_k: int = 0,
    top_k: int = 0,
    metadata_path: Path | None,
) -> tuple[bool, int, float, int, float, float, int, int, int]:
    """Resolve profile-copy settings for PBT restart argv re-injection."""
    defaults = (False, 1_000_000_000, 0.3, 3, 0.0, 0.0, 0, 0, 0)
    current = (
        bool(enable),
        int(after_frames),
        float(prob),
        int(keep_diversity_min_profiles),
        float(min_nsucc_gap),
        float(min_source_retention),
        int(cooldown_frames),
        int(bottom_k),
        int(top_k),
    )
    if current != defaults or metadata_path is None:
        return current
    parsed = _parse_profile_copy_config_from_metadata(metadata_path)
    if not parsed:
        return current
    return (
        bool(parsed["enable"]),
        int(parsed["after_frames"]),
        float(parsed["prob"]),
        int(parsed["keep_diversity_min_profiles"]),
        float(parsed["min_nsucc_gap"]),
        float(parsed["min_source_retention"]),
        int(parsed["cooldown_frames"]),
        int(parsed["bottom_k"]),
        int(parsed["top_k"]),
    )


def profile_copy_source_bonus(profile_name: str) -> float:
    """Return source-profile copy bonus weight."""
    return float(PROFILE_COPY_SOURCE_BONUS.get(str(profile_name or "legacy"), 1.0))


def _all_profile_controlled_dests() -> set[str]:
    dests: set[str] = set()
    for table in PROFILE_TABLES.values():
        dests.update(table.keys())
    return dests


def apply_wm_strategy_profile(
    args_cli,
    *,
    argv: list[str] | None = None,
    authoritative: bool | None = None,
) -> list[dict[str, Any]]:
    """Expand ``args_cli.wm_strategy_profile`` into concrete flag values.

    Mutates ``args_cli`` in-place.  Returns an audit list of changes.

    When ``authoritative`` is True (default for non-``legacy`` profiles),
    profile-table values override explicit CLI / PBT-restart args so restart
    machinery cannot pollute the selected profile.
    """
    profile = str(getattr(args_cli, "wm_strategy_profile", "legacy") or "legacy")
    if profile not in PROFILE_IDS:
        raise ValueError(f"unknown wm_strategy_profile {profile!r}")
    table = PROFILE_TABLES.get(profile, {})
    if not table:
        return []

    if authoritative is None:
        authoritative = profile != "legacy"

    explicit = collect_explicit_dests(argv)
    # Profile selector itself is never overridden by the table.
    explicit.discard("wm_strategy_profile")

    audit: list[dict[str, Any]] = []
    for dest, target in table.items():
        cur = getattr(args_cli, dest, None)
        if (not authoritative) and dest in explicit:
            audit.append(
                {
                    "flag": dest,
                    "profile_default": target,
                    "kept_user_value": cur,
                    "overridden": False,
                    "profile_authoritative": False,
                }
            )
            continue
        if cur != target:
            setattr(args_cli, dest, target)
            audit.append(
                {
                    "flag": dest,
                    "old": cur,
                    "new": target,
                    "source": "profile_authoritative" if authoritative else "profile",
                    "overridden": True,
                    "profile_authoritative": bool(authoritative),
                    "was_explicit": dest in explicit,
                }
            )
    lines = [
        f"{row['flag']}: "
        + (
            f"kept {row['kept_user_value']!r} (user override)"
            if not row.get("overridden", True)
            else (
                f"{row.get('old')!r} -> {row['new']!r} "
                f"({row.get('source', 'profile')}"
                + (
                    ", was explicit"
                    if row.get("was_explicit")
                    else ""
                )
                + ")"
            )
        )
        for row in audit
        if row.get("overridden", True) or "kept_user_value" in row
    ]
    header = f"[wm_strategy_profile={profile}] expansion"
    if authoritative:
        header += " (profile_authoritative=true)"
    if lines:
        print(
            f"{header}:\n  "
            + "\n  ".join(lines[:80])
            + (f"\n  ... ({len(lines) - 80} more)" if len(lines) > 80 else "")
        )
    elif authoritative:
        print(f"{header}: no changes (already matches profile table)")
    return audit


def profile_capability_flags(args_cli) -> dict[str, bool]:
    """Derive mode-aware capability booleans from post-expansion args."""
    profile = str(getattr(args_cli, "wm_strategy_profile", "legacy") or "legacy")
    return {
        "profile_actor_distill_enabled": bool(
            getattr(args_cli, "eir_actor_distill_enable", False)
        ),
        "profile_residual_apply_enabled": bool(
            getattr(args_cli, "eir_residual_apply_to_rollout", False)
        ),
        "profile_rssm_mppi_enabled": bool(
            getattr(args_cli, "eir_enable", False)
            and getattr(args_cli, "eir_use_rssm_mppi", False)
            and str(getattr(args_cli, "eir_mppi_mode", "off")) != "off"
        ),
        "profile_wm_pbt_score_enabled": str(
            getattr(args_cli, "pbt_wm_score_mode", "off")
        )
        == "active",
        "profile_wm_curriculum_controller_enabled": bool(
            getattr(args_cli, "wm_curriculum_controller_enable", False)
        ),
        "profile_elite_bank_enabled": bool(
            getattr(args_cli, "eir_enable", False)
            and getattr(args_cli, "eir_use_elite_bank", False)
        ),
        "profile_conservative_planner_enabled": PROFILE_FAMILY.get(profile, profile)
        in ("planner",),
        "wm_strategy_profile_name": profile,
        "wm_strategy_profile_id": int(PROFILE_IDS.get(profile, 0)),
        "wm_strategy_profile_family": PROFILE_FAMILY.get(profile, profile),
    }
