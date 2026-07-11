"""Late-degradation rescue source selection (tolerance + quality fail-closed).

Kept free of rl_games / Isaac so unit tests can exercise the real selector
path without importing the full observer.
"""
from __future__ import annotations

import math
from typing import Any, Callable


def population_nsucc_mean(state: dict[str, Any] | None) -> float:
    """Read nsucc from the real checkpoint meta layout (top-level preferred)."""
    if not state:
        return float("nan")
    for key in ("nsucc_mean", "mean_prev_episode_successes"):
        if key in state and state.get(key) is not None:
            try:
                return float(state[key])
            except (TypeError, ValueError):
                pass
    cur = state.get("curriculum") or {}
    for key in ("mean_prev_episode_successes", "nsucc_mean"):
        if key in cur and cur.get(key) is not None:
            try:
                return float(cur[key])
            except (TypeError, ValueError):
                continue
    return float("nan")


def population_ranking_score(state: dict[str, Any] | None) -> float:
    if not state:
        return float("-inf")
    for key in ("pbt_ranking_score", "objective"):
        if key in state and state.get(key) is not None:
            try:
                val = float(state[key])
                if math.isfinite(val):
                    return val
            except (TypeError, ValueError):
                continue
    return float("-inf")


def population_success_tolerance(state: dict[str, Any] | None) -> float:
    if not state:
        return float("nan")
    cur = state.get("curriculum") or {}
    try:
        return float(cur.get("success_tolerance", float("nan")))
    except (TypeError, ValueError):
        return float("nan")


def source_has_collapse_signal(state: dict[str, Any] | None) -> bool:
    if not state:
        return True
    if bool(state.get("hard_collapse_triggered", False)):
        return True
    cur = state.get("curriculum") or {}
    if bool(cur.get("hard_collapse_triggered", False)):
        return True
    if bool(cur.get("collapse_freeze_active", False)):
        return True
    return False


def select_late_degradation_rescue_source(
    *,
    sorted_policies: list[int],
    population: dict[int, dict[str, Any] | None],
    target_policy_idx: int,
    target_tolerance: float,
    target_nsucc_mean: float,
    env_frames: int,
    n_top: int,
    max_tol_gap: float,
    source_min_mean: float,
    min_source_advantage: float,
    source_min_retention: float,
    reject_collapse_source: bool,
    source_retention_fn: Callable[[int, float], float],
    eligible_fn: Callable[[dict[str, Any], int], tuple[bool, str]],
    archive_fn: Callable[..., tuple[dict[str, Any] | None, str]] | None = None,
) -> dict[str, Any]:
    """Filter-first late rescue: tol → quality → ranking score.

    ``sorted_policies`` MUST be ``list[int]`` (population rank order), matching
    the rest of the PBT observer. Never unpack as ``(idx, score)``.
    """
    reject_reasons: dict[int, str] = {}
    live_candidates: list[tuple[float, int, dict[str, Any]]] = []

    for src_idx in sorted_policies:
        idx = int(src_idx)
        if idx == int(target_policy_idx):
            continue
        state = population.get(idx)
        if not state:
            reject_reasons[idx] = "missing_state"
            continue
        src_tol = population_success_tolerance(state)
        if not math.isfinite(src_tol) or abs(src_tol - float(target_tolerance)) > float(max_tol_gap):
            reject_reasons[idx] = "tolerance_gap"
            continue
        ok, reason = eligible_fn(state, int(env_frames))
        if not ok:
            reject_reasons[idx] = f"ineligible:{reason}"
            continue
        src_mean = population_nsucc_mean(state)
        if not math.isfinite(src_mean) or src_mean < float(source_min_mean):
            reject_reasons[idx] = "mean_below_threshold"
            continue
        if src_mean < float(target_nsucc_mean) + float(min_source_advantage):
            reject_reasons[idx] = "advantage_below_threshold"
            continue
        src_ret = float(source_retention_fn(idx, src_tol))
        if not math.isfinite(src_ret) or src_ret < float(source_min_retention):
            reject_reasons[idx] = "retention_below_threshold"
            continue
        if reject_collapse_source and source_has_collapse_signal(state):
            reject_reasons[idx] = "collapse_signal"
            continue
        score = population_ranking_score(state)
        live_candidates.append((score, idx, state))

    live_candidates.sort(key=lambda x: (x[0], x[1]), reverse=True)
    if live_candidates:
        _score, src_idx, state = live_candidates[0]
        return {
            "action": "copy",
            "tier": "late_rescue_live",
            "checkpoint_type": "live",
            "source_policy": int(src_idx),
            "source_state": state,
            "strict_candidates": [c[1] for c in live_candidates[: max(1, int(n_top))]],
            "relaxed_candidates": [],
            "strict_filter_reasons": reject_reasons,
            "relaxed_filter_reasons": {},
            "archive_reason": "late_rescue_quality_filtered",
        }

    if archive_fn is not None:
        archive, reason = archive_fn()
        if archive is not None:
            return {
                "action": "copy",
                "tier": "late_rescue_archive",
                "checkpoint_type": str(
                    archive.get("_source_checkpoint_type", "archive")
                ),
                "source_policy": int(archive.get("policy_idx", -1)),
                "source_state": archive,
                "strict_candidates": [],
                "relaxed_candidates": [],
                "strict_filter_reasons": reject_reasons,
                "relaxed_filter_reasons": {},
                "archive_reason": reason,
            }

    return {
        "action": "skip",
        "tier": "skipped",
        "checkpoint_type": "none",
        "source_policy": -1,
        "source_state": None,
        "strict_candidates": [],
        "relaxed_candidates": [],
        "strict_filter_reasons": reject_reasons,
        "relaxed_filter_reasons": {},
        "archive_reason": "late_degradation_no_compatible_qualified_source",
    }
