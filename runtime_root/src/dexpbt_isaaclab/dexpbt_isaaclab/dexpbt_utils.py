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
