"""Soft tolerance candidate-retrieval gate helpers (EIR patch)."""

from __future__ import annotations

from dataclasses import dataclass

from .wm_activation_gate import soft_tol_weight


@dataclass(frozen=True)
class CandidateGateConfig:
    mode: str = "hard_tol"
    elite_tol_start: float = 0.070
    elite_tol_full: float = 0.045
    rssm_tol_start: float = 0.060
    rssm_tol_full: float = 0.040
    conservative_rssm_tol_start: float = 0.055
    conservative_rssm_tol_full: float = 0.035
    min_weight_for_retrieval: float = 0.05
    profile_name: str = ""


def candidate_gate_mode_id(mode: str) -> int:
    return {"hard_tol": 0, "soft_tol": 1, "profile_soft": 2}.get(str(mode), 0)


def elite_candidate_weight(cfg: CandidateGateConfig, tol_mean: float) -> float:
    if str(cfg.mode) in ("hard_tol", "off", ""):
        return 1.0
    return soft_tol_weight(
        float(tol_mean),
        tol_start=float(cfg.elite_tol_start),
        tol_full=float(cfg.elite_tol_full),
    )


def rssm_candidate_weight(cfg: CandidateGateConfig, tol_mean: float) -> float:
    if str(cfg.mode) in ("hard_tol", "off", ""):
        return 1.0
    prof = str(cfg.profile_name or "")
    if prof == "conservative_rssm_planner_distill":
        start, full = cfg.conservative_rssm_tol_start, cfg.conservative_rssm_tol_full
    else:
        start, full = cfg.rssm_tol_start, cfg.rssm_tol_full
    return soft_tol_weight(float(tol_mean), tol_start=float(start), tol_full=float(full))


def allow_retrieval(weight: float, min_weight: float) -> bool:
    return float(weight) >= float(min_weight)


__all__ = [
    "CandidateGateConfig",
    "candidate_gate_mode_id",
    "elite_candidate_weight",
    "rssm_candidate_weight",
    "allow_retrieval",
]
