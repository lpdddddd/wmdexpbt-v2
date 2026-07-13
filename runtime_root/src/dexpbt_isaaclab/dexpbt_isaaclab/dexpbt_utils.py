"""DexPBT utility functions: tolerance curriculum, objective, math helpers."""

from __future__ import annotations

import math
from typing import Tuple

import numpy as np
import torch
from torch import Tensor


def env_frames_to_sim_steps(env_frames: int, num_envs: int) -> int:
    """Convert an env-frame budget to simulation steps for env-side counters.

    Env ``frame_since_restart`` increments once per physics step; PBT/CLI
    thresholds are expressed in env frames (= steps * num_envs).
    """
    ne = max(1, int(num_envs))
    return max(1, math.ceil(int(env_frames) / ne))


@torch.jit.script
def scale(x: Tensor, lower: Tensor, upper: Tensor) -> Tensor:
    """Map [-1, 1] to [lower, upper]."""
    return 0.5 * (x + 1.0) * (upper - lower) + lower


@torch.jit.script
def unscale(x: Tensor, lower: Tensor, upper: Tensor) -> Tensor:
    """Map [lower, upper] to [-1, 1]."""
    return (2.0 * x - upper - lower) / (upper - lower)


@torch.jit.script
def quat_rotate(q: Tensor, v: Tensor) -> Tensor:
    """Rotate vector v by quaternion q (wxyz convention)."""
    q_w = q[:, 0:1]
    q_vec = q[:, 1:4]
    a = v * (2.0 * q_w * q_w - 1.0)
    b = torch.cross(q_vec, v, dim=-1) * q_w * 2.0
    c = q_vec * torch.sum(q_vec * v, dim=-1, keepdim=True) * 2.0
    return a + b + c


@torch.jit.script
def quat_conjugate(q: Tensor) -> Tensor:
    """Conjugate of quaternion (wxyz convention)."""
    return torch.cat([q[:, 0:1], -q[:, 1:4]], dim=-1)


@torch.jit.script
def quat_mul(a: Tensor, b: Tensor) -> Tensor:
    """Multiply two quaternions (wxyz convention)."""
    w1, x1, y1, z1 = a[:, 0:1], a[:, 1:2], a[:, 2:3], a[:, 3:4]
    w2, x2, y2, z2 = b[:, 0:1], b[:, 1:2], b[:, 2:3], b[:, 3:4]
    return torch.cat(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dim=-1,
    )


def random_quaternion(n: int, device: torch.device) -> Tensor:
    """Sample uniform random quaternions (wxyz convention)."""
    q = torch.randn(n, 4, device=device)
    q = q / torch.norm(q, p=2, dim=-1, keepdim=True)
    q[:, 0] = q[:, 0].abs()
    return q


def interp_0_1(x_curr: float, x_initial: float, x_target: float) -> float:
    """0 when x_curr == x_initial, 1 when x_curr == x_target."""
    span = x_initial - x_target
    if abs(span) < 1e-9:
        return 1.0
    return (x_initial - x_curr) / span


def dexpbt_tolerance_curriculum(
    last_curriculum_update: int,
    frames_since_restart: int,
    curriculum_interval: int,
    prev_episode_successes: Tensor,
    success_tolerance: float,
    initial_tolerance: float,
    target_tolerance: float,
    curriculum_freeze_frames_after_restart: int = 0,
) -> Tuple[float, int]:
    """
    DexPBT paper curriculum: tighten epsilon when mean Nsucc > 3.
    epsilon <- max(0.9 * epsilon, epsilon*).

    Patch H (2026-05-31): post-restart curriculum freeze. When
    ``frames_since_restart < curriculum_freeze_frames_after_restart`` we
    leave epsilon alone. **Note:** despite the parameter name, the env
    stores this threshold in *simulation steps*; the observer converts
    env-frame CLI budgets before resume.
    """
    if (
        curriculum_freeze_frames_after_restart > 0
        and frames_since_restart < curriculum_freeze_frames_after_restart
    ):
        # While frozen, do NOT tighten and keep the last_curriculum_update
        # advancing in lockstep with frames_since_restart so the immediately
        # post-freeze first tick is still gated by `curriculum_interval`.
        return success_tolerance, frames_since_restart
    if frames_since_restart - last_curriculum_update < curriculum_interval:
        return success_tolerance, last_curriculum_update

    mean_nsucc = float(prev_episode_successes.mean().item())
    if mean_nsucc <= 3.0:
        return success_tolerance, last_curriculum_update

    new_tol = max(0.9 * success_tolerance, target_tolerance)
    new_tol = min(new_tol, initial_tolerance)
    print(f"DexPBT curriculum: mean Nsucc={mean_nsucc:.3f}, success_tolerance -> {new_tol}")
    return new_tol, frames_since_restart


def tolerance_successes_objective(
    success_tolerance: float,
    initial_tolerance: float,
    target_tolerance: float,
    successes: Tensor,
) -> Tensor:
    """PBT meta-objective: prioritizes tolerance over successes during curriculum."""
    if initial_tolerance > target_tolerance:
        tolerance_objective = interp_0_1(success_tolerance, initial_tolerance, target_tolerance)
    else:
        tolerance_objective = 1.0

    if success_tolerance > target_tolerance:
        true_objective = (successes * 0.01) + tolerance_objective
    else:
        true_objective = successes + tolerance_objective

    return true_objective


# ---------------------------------------------------------------------------
# Patch 1 (rev 2, 2026-05-30): r_meta component split + PBT ranking score.
#
# `tolerance_successes_objective` (the original DexPBT formula) couples a
# *tolerance progress* term and a *successes* term. During curriculum
# tightening this becomes misleading for PBT selection: as epsilon shrinks
# the tolerance term alone climbs toward 1.0 even when nsucc collapses to
# zero (observed at 1.9B frames in regrasping_cbwm_20260530_090008 where all
# 8 policies reported r_meta=0.541 with nsucc=0).
#
# These helpers add a *reporting-only* decomposition (`split_r_meta_components`)
# and an explicit *PBT-selection score* (`compute_pbt_ranking_score`).
# Neither modifies the legacy `tolerance_successes_objective` math; the
# original `r_meta_raw` is still reported end-to-end.
# ---------------------------------------------------------------------------


def split_r_meta_components(
    success_tolerance: float,
    initial_tolerance: float,
    target_tolerance: float,
    mean_successes: float,
) -> Tuple[float, float, float]:
    """Return (r_meta_raw, r_meta_tolerance, r_meta_success).

    Mirrors `tolerance_successes_objective` but works on a scalar
    `mean_successes` and exposes the two additive components so that
    downstream code (CF low_signal detection, training_curve logging) can
    reason about which component changed.
    """
    if initial_tolerance > target_tolerance:
        tolerance_component = interp_0_1(
            success_tolerance, initial_tolerance, target_tolerance
        )
    else:
        tolerance_component = 1.0

    if success_tolerance > target_tolerance:
        success_component = float(mean_successes) * 0.01
    else:
        success_component = float(mean_successes)

    return (
        float(tolerance_component + success_component),
        float(tolerance_component),
        float(success_component),
    )


# Supported ranking modes for `compute_pbt_ranking_score`.
PBT_RANKING_METRIC_RAW = "raw_r_meta"
PBT_RANKING_METRIC_SUCCESS_AWARE = "success_aware_r_meta"
PBT_RANKING_METRICS = (
    PBT_RANKING_METRIC_RAW,
    PBT_RANKING_METRIC_SUCCESS_AWARE,
)


def compute_pbt_ranking_score(
    r_meta_raw: float,
    mean_nsucc: float,
    mode: str = PBT_RANKING_METRIC_SUCCESS_AWARE,
    nsucc_zero_cap: float = 0.05,
    nsucc_low_threshold: float = 1.0,
    nsucc_low_cap: float = 0.05,
) -> float:
    """Convert reporting `r_meta_raw` into a score for PBT tier/source selection.

    Modes:
      * raw_r_meta: legacy behavior; returns `r_meta_raw` unchanged. Kept as
        rollback path -- do NOT remove.
      * success_aware_r_meta (default): cap the score whenever the worker's
        reported `mean_nsucc` is too low to trust the original DexPBT
        formula. Three cases:

          - `mean_nsucc <= 0` (true collapse): cap at `nsucc_zero_cap`.
          - `0 < mean_nsucc < nsucc_low_threshold` (Patch B, rev 3):
            "low nsucc + high r_meta" -- the dominant collapse vector
            observed in regrasping_cbwm_20260530_160513 where
            `policy_1` ran at nsucc=0.17 / r_meta_raw=0.94 because
            tolerance progress had already maxed out the tolerance
            component. Cap at `nsucc_low_cap`.
          - `mean_nsucc >= nsucc_low_threshold`: return `r_meta_raw`
            verbatim, so workers with real success signal continue to
            be ranked by the original DexPBT formula.

    Note: setting `nsucc_low_threshold=0.0` disables the Patch B low-band
    cap and recovers the Patch 1 behaviour (only `<=0` is capped).
    Setting `mode="raw_r_meta"` disables both caps entirely.

    This function is intentionally pure and side-effect-free so it can be
    unit-tested without the rl_games / Isaac stack.
    """
    score = float(r_meta_raw)
    if mode == PBT_RANKING_METRIC_RAW:
        return score
    if mode != PBT_RANKING_METRIC_SUCCESS_AWARE:
        raise ValueError(
            f"Unknown pbt_ranking_metric={mode!r}; "
            f"expected one of {PBT_RANKING_METRICS}"
        )
    n = float(mean_nsucc)
    if n <= 0.0:
        return min(score, float(nsucc_zero_cap))
    if n < float(nsucc_low_threshold):
        return min(score, float(nsucc_low_cap))
    return score


# ---------------------------------------------------------------------------
# Phase A4 P4 (2026-06-09): floor-stage recovery scoring (Part A).
#
# Goal:
#   At hard floor (success_tolerance ~ target_tolerance ~ 0.01) the
#   "absolute nsucc_mean" signal alone misleads PBT: a worker that just
#   re-learned at floor with nsucc=8 trending up should not be replaced
#   by one stuck at nsucc=12 plateau with zero slope.
#
# The score below is a SECONDARY ranking signal, gated by
# --pbt_floor_recovery_aware. Default code paths must not call it.
#
# Convention (per user constraint #7): all sub-terms are normalized to
# 0..1 first, then the final score is multiplied by 50 to live on the
# same nsucc scale as the legacy ranking. This avoids mixing 0..1 and
# 0..50 metrics in WM-guided ranking later.
# ---------------------------------------------------------------------------

FLOOR_STAGE_NONE = "none"
FLOOR_STAGE_FLOOR = "floor"
FLOOR_STAGE_HARD_FLOOR = "hard_floor"


def floor_stage_classify(
    success_tolerance: float,
    target_tolerance: float,
    *,
    floor_tol_threshold: float = 0.02,
    hard_floor_tol_threshold: float = 0.012,
    floor_tol_eq_target_eps: float = 1e-9,
) -> str:
    """Classify the per-policy floor stage from current curriculum state.

    Returns one of {"none","floor","hard_floor"}. Pure function; safe to
    call from tests and reports without observer state.
    """
    tol = float(success_tolerance)
    tgt = float(target_tolerance)
    if tol <= float(hard_floor_tol_threshold) or abs(tol - tgt) <= floor_tol_eq_target_eps:
        return FLOOR_STAGE_HARD_FLOOR
    if tol <= float(floor_tol_threshold):
        return FLOOR_STAGE_FLOOR
    return FLOOR_STAGE_NONE


def compute_floor_recovery_slope(
    nsucc_mean_history: list[float],
    *,
    window: int = 4,
    nsucc_norm: float = 50.0,
) -> tuple[float, float]:
    """Return (slope_raw, slope_positive_norm) over the last ``window`` ticks.

    slope_raw  = (nsucc_now - nsucc_window_ago) / max(1, window-1) / nsucc_norm
    slope_positive_norm = clip(slope_raw, 0, 1)  # only positive recovery counts

    A fully-saturated recovery from 0 -> 50 over `window` ticks gives
    slope_positive_norm ~= 1. Negative slopes return (slope_raw, 0).
    Empty/short histories return (0, 0).
    """
    xs = [float(x) for x in nsucc_mean_history if x is not None]
    if len(xs) < 2:
        return 0.0, 0.0
    w = max(2, min(int(window), len(xs)))
    tail = xs[-w:]
    span = float(tail[-1]) - float(tail[0])
    denom = max(1, w - 1) * max(1e-6, float(nsucc_norm))
    slope = span / denom
    return float(slope), float(max(0.0, min(1.0, slope)))


def compute_floor_recovery_score(
    *,
    nsucc_mean: float,
    nsucc_max: float,
    retention: float,
    slope_positive_norm: float,
    drop_ratio: float,
    hard_collapse: bool = False,
    nsucc_norm: float = 50.0,
    slope_weight: float = 0.5,
    retention_weight: float = 0.3,
    nsucc_max_weight: float = 0.2,
    drop_penalty: float = 0.5,
    hard_collapse_penalty: float = 1.0,
) -> tuple[float, dict[str, float]]:
    """Return (score_50, components).

    All sub-terms are 0..1; ``score_50 = 50 * weighted_sum``. The legacy
    PBT ranking is on a ~0..50 nsucc scale, so this stays comparable.

    Components are returned for JSONL/TB audit logging.
    """
    nm = float(max(0.0, min(1.0, float(nsucc_mean) / max(1e-6, float(nsucc_norm)))))
    nx = float(max(0.0, min(1.0, float(nsucc_max) / max(1e-6, float(nsucc_norm)))))
    rt = float(max(0.0, min(1.0, float(retention))))
    sl = float(max(0.0, min(1.0, float(slope_positive_norm))))
    dp = float(max(0.0, min(1.0, float(drop_ratio))))
    coll = 1.0 if bool(hard_collapse) else 0.0

    weighted = (
        1.0 * nm
        + float(slope_weight) * sl
        + float(retention_weight) * rt
        + float(nsucc_max_weight) * nx
        - float(drop_penalty) * dp
        - float(hard_collapse_penalty) * coll
    )
    score_50 = 50.0 * float(weighted)
    components = {
        "nsucc_mean_norm": nm,
        "nsucc_max_norm": nx,
        "retention": rt,
        "slope_positive_norm": sl,
        "drop_ratio": dp,
        "hard_collapse": coll,
        "weighted_norm": float(weighted),
        "score_50": float(score_50),
    }
    return float(score_50), components


def combine_wm_recovery_score(
    real_score_50: float,
    *,
    wm_recovery_prob: float,
    wm_success_prob: float,
    wm_drop_risk: float,
    wm_uncertainty: float,
    recovery_weight: float = 0.3,
    success_weight: float = 0.2,
    drop_penalty: float = 0.3,
    uncertainty_penalty: float = 0.2,
    nsucc_norm: float = 50.0,
) -> tuple[float, dict[str, float]]:
    """Combine real floor-recovery score with imagined WM evaluation.

    Sub-terms are 0..1 probabilities; converted to the same 0..50 scale so
    PBT can sort with one number. The real score is preserved as the
    dominant additive term; WM contributes a bounded delta.
    """
    rp = float(max(0.0, min(1.0, float(wm_recovery_prob))))
    sp = float(max(0.0, min(1.0, float(wm_success_prob))))
    dr = float(max(0.0, min(1.0, float(wm_drop_risk))))
    un = float(max(0.0, min(1.0, float(wm_uncertainty))))
    delta_norm = (
        float(recovery_weight) * rp
        + float(success_weight) * sp
        - float(drop_penalty) * dr
        - float(uncertainty_penalty) * un
    )
    delta_50 = 50.0 * float(delta_norm)
    score = float(real_score_50) + float(delta_50)
    components = {
        "wm_recovery_prob": rp,
        "wm_success_prob": sp,
        "wm_drop_risk": dr,
        "wm_uncertainty": un,
        "delta_norm": float(delta_norm),
        "delta_50": float(delta_50),
        "real_score_50": float(real_score_50),
        "wm_score_50": float(score),
    }
    return float(score), components


# CSER-v0 TensorBoard scalar names (audit / test registry).
CSER_TB_SCALAR_NAMES: tuple[str, ...] = (
    "elite_snippet/bank_size",
    "elite_snippet/inserted_rows",
    "elite_snippet/retrieval_hit_rate",
    "snippet_distill/loss",
    "snippet_distill/active_frac",
    "snippet_wm_rerank/changed_frac",
    "snippet_wm_rerank/wm_term_mean",
    "wm_cal/alpha_wm",
    "wm_cal/AUC_recovery",
    "wm_cal/ECE",
    "wm_cal/ece_success",
    "wm_cal/ece_drop",
    "wm_cal/rank_corr",
    "wm_cal/rank_corr_combined",
    "wm_cal/rank_corr_recovery",
    "wm_cal/combined_score_mean",
    "wm_cal/realized_utility_mean",
    "wm_cal/gate_combined_rank",
    "pbt_consolidation/stage",
    "pbt_consolidation/restart_blocked_count",
)

# Progress-Conditioned CSER-DexPBT (cser_dexpbt_progress_refine) scalar names.
PROGRESS_REFINE_TB_SCALAR_NAMES: tuple[str, ...] = (
    "progress_refine/gate_mean",
    "progress_refine/gate_p50",
    "progress_refine/gate_p90",
    "progress_refine/active_frac",
    "progress_refine/history_valid_frac",
    "progress_refine/normalized_error_mean",
    "progress_refine/normalized_error_active_mean",
    "progress_refine/recent_progress_mean",
    "progress_refine/recent_progress_active_mean",
    "progress_refine/oscillation_mean",
    "progress_refine/teacher_confidence_mean",
    "progress_refine/teacher_margin_mean",
    "progress_refine/teacher_score_margin_mean",
    "progress_refine/retrieval_valid_frac",
    "progress_refine/retrieval_budget_frac",
    "progress_refine/effective_sample_size",
    "progress_refine/loss",
    "progress_refine/fallback_frac",
    "progress_refine/nonfinite_count",
    "snippet_retrieval/tolerance_weight_mean",
    "snippet_retrieval/log_tolerance_gap_mean",
    "snippet_retrieval/no_compatible_frac",
    "snippet_retrieval/nonfinite_candidate_row_frac",
    "snippet_wm_rerank/valid_frac",
    "snippet_wm_rerank/fallback_frac",
    "snippet_wm_rerank/candidate_valid_frac",
    "snippet_wm_rerank/row_prediction_valid_frac",
    "snippet_wm_rerank/row_active_frac",
    "snippet_wm_rerank/row_fallback_frac",
    "snippet_wm_rerank/changed_frac_overall",
    "snippet_wm_rerank/changed_frac_given_active",
    "snippet_wm_rerank/predict_error_count",
    "snippet_wm_rerank/selected_score_mean",
    "snippet_wm_rerank/actor_proxy_score_mean",
    "snippet_wm_rerank/changed_frac",
    "snippet_wm_rerank/wm_term_mean",
    "snippet_wm_rerank/action_delta_mean",
    "snippet_wm_rerank/action_penalty_mean",
    "snippet_wm_rerank/fallback_reason_id",
    "adaptive_consolidation/curriculum_progress",
    "adaptive_consolidation/stage",
    "adaptive_consolidation/local_retention",
    "adaptive_consolidation/probation_active",
    "adaptive_consolidation/probation_frames",
    "adaptive_consolidation/shrink_blocked",
    "adaptive_consolidation/hard_collapse_streak",
    "adaptive_consolidation/rescue_triggered",
    "adaptive_consolidation/rescue_source_rank",
)
