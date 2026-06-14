"""PBT-WM Strategy Zoo v1.0 — per-worker WM usage profiles.

Each profile expands to a full set of train.py CLI flag values.  ``legacy``
is a no-op so existing launchers remain unchanged.
"""
from __future__ import annotations

import sys
from typing import Any

PROFILE_IDS: dict[str, int] = {
    "legacy": 0,
    "ppo_only": 1,
    "rssm_mppi_actor_distill": 2,
    "wm_curriculum_controller": 3,
    "wm_pbt_source_selector": 4,
    "elite_bank_actor_distill": 5,
    "conservative_rssm_planner_distill": 6,
}

PROFILE_FAMILY: dict[str, str] = {
    "legacy": "legacy",
    "ppo_only": "anchor",
    "rssm_mppi_actor_distill": "actor_distill",
    "wm_curriculum_controller": "curriculum",
    "wm_pbt_source_selector": "pbt_selector",
    "elite_bank_actor_distill": "elite_imitation",
    "conservative_rssm_planner_distill": "planner",
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
    "eir_mppi_ensemble_max_models_per_plan": 3,
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
    "eir_mppi_num_sequences": 64,
    "eir_mppi_num_elite": 8,
    "eir_mppi_cem_iters": 1,
    "eir_mppi_action_noise_std": 0.10,
    "eir_mppi_max_action_delta": 0.35,
    "eir_mppi_min_sequence_margin": 0.03,
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
    "eir_mppi_ensemble_max_models_per_plan": 3,
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
    "eir_actor_distill_min_margin": 0.03,
    "eir_actor_distill_margin_norm": 0.05,
    "eir_actor_distill_max_delta": 0.30,
    "eir_actor_distill_allowed_sources": "rssm",
    "eir_actor_distill_uncertainty_threshold": 1.5,
    "eir_actor_distill_feature_detach": True,
    "eir_actor_distill_mu_head_only": True,
    "pbt_wm_score_mode": "off",
    "pbt_use_wm_recovery_score": False,
    "wm_curriculum_controller_enable": False,
}

PROFILE_TABLES: dict[str, dict[str, Any]] = {
    "legacy": {},
    "ppo_only": _PPO_ONLY,
    "rssm_mppi_actor_distill": _RSSM_MPPI_ACTOR_DISTILL,
    "wm_curriculum_controller": _WM_CURRICULUM_CONTROLLER,
    "wm_pbt_source_selector": _WM_PBT_SOURCE_SELECTOR,
    "elite_bank_actor_distill": _ELITE_BANK_ACTOR_DISTILL,
    "conservative_rssm_planner_distill": _CONSERVATIVE_RSSM_PLANNER,
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


def apply_wm_strategy_profile(args_cli, *, argv: list[str] | None = None) -> list[dict[str, Any]]:
    """Expand ``args_cli.wm_strategy_profile`` into concrete flag values.

    Mutates ``args_cli`` in-place.  Returns an audit list of changes.
    """
    profile = str(getattr(args_cli, "wm_strategy_profile", "legacy") or "legacy")
    if profile not in PROFILE_IDS:
        raise ValueError(f"unknown wm_strategy_profile {profile!r}")
    table = PROFILE_TABLES.get(profile, {})
    if not table:
        return []

    explicit = collect_explicit_dests(argv)
    # Profile selector itself is never overridden by the table.
    explicit.discard("wm_strategy_profile")

    audit: list[dict[str, Any]] = []
    for dest, target in table.items():
        cur = getattr(args_cli, dest, None)
        if dest in explicit:
            audit.append(
                {
                    "flag": dest,
                    "profile_default": target,
                    "kept_user_value": cur,
                    "overridden": False,
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
                    "source": "profile",
                    "overridden": True,
                }
            )
    lines = [
        f"{row['flag']}: "
        + (
            f"kept {row['kept_user_value']!r} (user override)"
            if not row.get("overridden", True)
            else f"{row.get('old')!r} -> {row['new']!r} (profile)"
        )
        for row in audit
        if row.get("overridden", True) or "kept_user_value" in row
    ]
    if lines:
        print(
            f"[wm_strategy_profile={profile}] expansion:\n  "
            + "\n  ".join(lines[:80])
            + (f"\n  ... ({len(lines) - 80} more)" if len(lines) > 80 else "")
        )
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
        "profile_conservative_planner_enabled": profile
        == "conservative_rssm_planner_distill",
        "wm_strategy_profile_name": profile,
        "wm_strategy_profile_id": int(PROFILE_IDS.get(profile, 0)),
        "wm_strategy_profile_family": PROFILE_FAMILY.get(profile, profile),
    }
