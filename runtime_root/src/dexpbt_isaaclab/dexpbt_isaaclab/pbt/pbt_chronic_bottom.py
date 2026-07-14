"""Chronic-bottom PBT exploit helpers (Experiment A).

Pure decision helpers kept free of the heavy DexPBT observer so they can be
unit-tested without training.  Wired into ``DexPBTPbtAlgoObserver`` when
``pbt_enable_chronic_bottom_exploit`` is True.

Chronic path may only replace the live-source *skill_delta* criterion. All
source health / eligibility checks go through
``pbt_live_source_health.evaluate_common_live_source_eligibility``.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any

from dexpbt_isaaclab.pbt.pbt_live_source_health import CommonSourceEligibility

CHRONIC_DEFAULT_MIN_BOTTOM_FRAMES = 200_000_000
CHRONIC_DEFAULT_MIN_BOTTOM_TICKS = 5
CHRONIC_DEFAULT_MIN_SCORE_GAP = 0.15
CHRONIC_DEFAULT_COOLDOWN_FRAMES = 400_000_000
CHRONIC_DEFAULT_MAX_COPIES_PER_LINEAGE = 1
CHRONIC_SCHEMA_VERSION = 1


@dataclass
class ChronicBottomState:
    """Per-policy persistent state for chronic-bottom exploit."""

    schema_version: int = CHRONIC_SCHEMA_VERSION
    continuous_bottom_frames: int = 0
    continuous_bottom_ticks: int = 0
    bottom_enter_frame: int = -1
    last_tier: str = ""
    chronic_copy_count: int = 0
    chronic_copy_cooldown_until: int = 0
    last_chronic_copy_frame: int = -1
    last_chronic_source: int = -1
    last_reason: str = ""

    def to_json(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_json(cls, data: dict[str, Any] | None) -> "ChronicBottomState":
        if not isinstance(data, dict):
            return cls()
        try:
            ver = int(data.get("schema_version", CHRONIC_SCHEMA_VERSION))
        except Exception:
            return cls(last_reason="schema_unreadable")
        if ver > CHRONIC_SCHEMA_VERSION:
            # Fail-closed safe init for unknown newer schemas.
            return cls(last_reason="schema_too_new")
        kwargs: dict[str, Any] = {}
        for k in cls.__dataclass_fields__:
            if k in data:
                kwargs[k] = data[k]
        try:
            return cls(**kwargs)
        except Exception:
            return cls(last_reason="schema_corrupt")


def reset_on_lineage_change(state: ChronicBottomState) -> ChronicBottomState:
    """Clear stale bottom span / quota on lineage change; keep cooldown."""
    state.chronic_copy_count = 0
    state.continuous_bottom_frames = 0
    state.continuous_bottom_ticks = 0
    state.bottom_enter_frame = -1
    state.last_tier = ""
    state.last_reason = "lineage_reset"
    return state


@dataclass
class ChronicEligibility:
    eligible: bool
    reason: str
    continuous_bottom_frames: int = 0
    continuous_bottom_ticks: int = 0
    score_gap: float = 0.0
    best_source: int = -1


@dataclass
class ChronicSourceDecision:
    action: str  # copy | skip
    source_policy: int = -1
    source_state: dict[str, Any] | None = None
    tier: str = "skipped"
    checkpoint_type: str = "none"
    skip_reason: str | None = None
    source_fallback_used: bool = False
    mixed_tol_fallback_used: bool = False
    source_skill_delta: float | None = None
    source_skill_delta_valid: bool = False
    source_skill_delta_invalid_reason: str = ""
    source_filter_reason: str = ""
    final_source_filter_reason: str = ""
    common_health_allowed: bool = False
    common_health_reason: str = ""
    score_gap: float = 0.0
    source_score: float = float("-inf")
    target_score: float = float("-inf")
    candidates_considered: list[dict[str, Any]] = field(default_factory=list)


def update_continuous_bottom(
    state: ChronicBottomState,
    *,
    tier: str,
    env_frames: int,
    interval_steps: int,
) -> ChronicBottomState:
    """Update continuous bottom span counters for one PBT tick.

    Leaving bottom clears duration/ticks (but not chronic_copy_count /
    cooldown, which persist across re-entry until reset elsewhere).
    """
    tier_s = str(tier or "")
    if tier_s != "bottom":
        state.continuous_bottom_frames = 0
        state.continuous_bottom_ticks = 0
        state.bottom_enter_frame = -1
        state.last_tier = tier_s
        return state
    if state.last_tier != "bottom" or state.bottom_enter_frame < 0:
        state.bottom_enter_frame = int(env_frames)
        state.continuous_bottom_ticks = 1
        state.continuous_bottom_frames = 0
    else:
        state.continuous_bottom_ticks = int(state.continuous_bottom_ticks) + 1
        state.continuous_bottom_frames = max(
            0, int(env_frames) - int(state.bottom_enter_frame)
        )
        approx = max(
            0, (int(state.continuous_bottom_ticks) - 1) * max(int(interval_steps), 1)
        )
        state.continuous_bottom_frames = max(int(state.continuous_bottom_frames), approx)
    state.last_tier = "bottom"
    return state


def reset_after_chronic_copy(
    state: ChronicBottomState, *, env_frames: int, cooldown: int
) -> ChronicBottomState:
    state.chronic_copy_count = int(state.chronic_copy_count) + 1
    state.last_chronic_copy_frame = int(env_frames)
    state.chronic_copy_cooldown_until = int(env_frames) + int(cooldown)
    state.continuous_bottom_frames = 0
    state.continuous_bottom_ticks = 0
    state.bottom_enter_frame = -1
    return state


def evaluate_chronic_eligibility(
    *,
    tier: str,
    state: ChronicBottomState,
    env_frames: int,
    promising_protected: bool,
    in_elimination_burnin: bool,
    in_probation: bool,
    target_score: float,
    ranked_scores: dict[int, float],
    target_idx: int,
    min_bottom_frames: int = CHRONIC_DEFAULT_MIN_BOTTOM_FRAMES,
    min_bottom_ticks: int = CHRONIC_DEFAULT_MIN_BOTTOM_TICKS,
    min_score_gap: float = CHRONIC_DEFAULT_MIN_SCORE_GAP,
    max_copies_per_lineage: int = CHRONIC_DEFAULT_MAX_COPIES_PER_LINEAGE,
) -> ChronicEligibility:
    if str(tier) != "bottom":
        return ChronicEligibility(False, "not_bottom")
    if promising_protected:
        return ChronicEligibility(False, "promising_protected")
    if in_elimination_burnin:
        return ChronicEligibility(False, "elimination_burnin")
    if in_probation:
        return ChronicEligibility(False, "probation_active")
    if int(env_frames) < int(state.chronic_copy_cooldown_until):
        return ChronicEligibility(
            False,
            "chronic_cooldown",
            continuous_bottom_frames=int(state.continuous_bottom_frames),
            continuous_bottom_ticks=int(state.continuous_bottom_ticks),
        )
    if int(state.chronic_copy_count) >= int(max_copies_per_lineage):
        return ChronicEligibility(
            False,
            "chronic_copy_quota",
            continuous_bottom_frames=int(state.continuous_bottom_frames),
            continuous_bottom_ticks=int(state.continuous_bottom_ticks),
        )
    if int(state.continuous_bottom_frames) < int(min_bottom_frames):
        return ChronicEligibility(
            False,
            "bottom_frames_below_threshold",
            continuous_bottom_frames=int(state.continuous_bottom_frames),
            continuous_bottom_ticks=int(state.continuous_bottom_ticks),
        )
    if int(state.continuous_bottom_ticks) < int(min_bottom_ticks):
        return ChronicEligibility(
            False,
            "bottom_ticks_below_threshold",
            continuous_bottom_frames=int(state.continuous_bottom_frames),
            continuous_bottom_ticks=int(state.continuous_bottom_ticks),
        )
    if not math.isfinite(float(target_score)):
        return ChronicEligibility(
            False,
            "target_score_nonfinite",
            continuous_bottom_frames=int(state.continuous_bottom_frames),
            continuous_bottom_ticks=int(state.continuous_bottom_ticks),
        )

    best_src = -1
    best_gap = float("-inf")
    t_score = float(target_score)
    for idx, sc in ranked_scores.items():
        if int(idx) == int(target_idx):
            continue
        try:
            s = float(sc)
        except Exception:
            continue
        if not math.isfinite(s):
            continue
        gap = s - t_score
        if gap > best_gap:
            best_gap = gap
            best_src = int(idx)
    if best_src < 0 or not math.isfinite(best_gap) or best_gap < float(min_score_gap):
        return ChronicEligibility(
            False,
            "score_gap_below_threshold",
            continuous_bottom_frames=int(state.continuous_bottom_frames),
            continuous_bottom_ticks=int(state.continuous_bottom_ticks),
            score_gap=float(best_gap if math.isfinite(best_gap) else 0.0),
            best_source=best_src,
        )
    return ChronicEligibility(
        True,
        "ok",
        continuous_bottom_frames=int(state.continuous_bottom_frames),
        continuous_bottom_ticks=int(state.continuous_bottom_ticks),
        score_gap=float(best_gap),
        best_source=best_src,
    )


def skill_delta_invalid_reason_is_mixed_tol(reason: str) -> bool:
    r = str(reason or "")
    return (
        r.startswith("tol_bins")
        or "tol_bins" in r
        or r == "mixed_tol_bins"
    )


def skill_delta_with_reason(
    window: Any,
    *,
    interval_steps: int,
    window_len: int = 6,
) -> tuple[float | None, str]:
    """Return (skill_delta, reason) for chronic live-source gating."""
    if window is None or not getattr(window, "samples", None):
        return None, "empty_window"
    samples = (
        window.last_n(int(window_len))
        if hasattr(window, "last_n")
        else list(window.samples)[-int(window_len) :]
    )
    if len(samples) < int(window_len):
        return None, "insufficient_samples"
    lineage_ids = {getattr(s, "lineage_id", None) for s in samples}
    if None in lineage_ids or len(lineage_ids) != 1:
        return None, "lineage_change"
    tol_bins = {
        round(float(getattr(s, "success_tolerance", float("nan"))), 6) for s in samples
    }
    if len(tol_bins) != 1:
        return None, "mixed_tol_bins"
    try:
        from dexpbt_isaaclab.pbt.pbt_curriculum_p1 import compute_skill_delta

        delta = compute_skill_delta(
            window, window_len=window_len, interval_steps=interval_steps
        )
    except Exception:
        return None, "compute_failed"
    if delta is None:
        return None, "invalid_other"
    return float(delta), "ok"


def live_source_ok_for_chronic(
    *,
    skill_delta: float | None,
    skill_delta_why: str,
    source_nsucc_mean: float,
    target_nsucc_mean: float,
    source_recent_drop_ratio: float = 0.0,
    min_stable_pos_eps: float = 0.02,
    max_drop_ratio: float = 0.30,
    min_nsucc_advantage: float = 3.0,
) -> tuple[bool, str, bool]:
    """Return (ok, reason, fallback_used).

    This replaces *only* the ordinary skill_delta live gate. Common health must
    already have passed.
    """
    if skill_delta is not None:
        if float(skill_delta) >= float(min_stable_pos_eps):
            return True, "ok_skill_delta", False
        return False, "source_skill_delta_unstable", False
    if skill_delta_invalid_reason_is_mixed_tol(skill_delta_why):
        if not math.isfinite(float(source_nsucc_mean)) or not math.isfinite(
            float(target_nsucc_mean)
        ):
            return False, "fallback_nonfinite_nsucc", False
        if float(source_recent_drop_ratio) > float(max_drop_ratio):
            return False, "fallback_source_dropping", False
        if float(source_nsucc_mean) < float(target_nsucc_mean) + float(
            min_nsucc_advantage
        ):
            return False, "fallback_advantage_too_small", False
        return True, "ok_mixed_tol_fallback", True
    return False, f"source_skill_delta_invalid:{skill_delta_why}", False


def pick_chronic_source(
    *,
    sorted_policies: list[int],
    population: dict[int, dict[str, Any] | None],
    target_idx: int,
    target_score: float,
    target_nsucc_mean: float,
    n_top: int,
    n_mid: int,
    min_score_gap: float,
    skill_delta_by_policy: dict[int, tuple[float | None, str]],
    common_health_by_policy: dict[int, CommonSourceEligibility],
    drop_ratio_by_policy: dict[int, float] | None = None,
) -> ChronicSourceDecision:
    """Prefer top then mid sources after common health + chronic skill gate."""
    drop_ratio_by_policy = drop_ratio_by_policy or {}
    if not math.isfinite(float(target_score)):
        return ChronicSourceDecision(
            action="skip",
            skip_reason="target_score_nonfinite",
            final_source_filter_reason="target_score_nonfinite",
            target_score=float(target_score),
        )
    t_score = float(target_score)
    considered: list[dict[str, Any]] = []

    def _score(idx: int) -> float:
        st = population.get(idx) or {}
        v = st.get("pbt_ranking_score", st.get("objective", float("-inf")))
        try:
            return float(v)
        except Exception:
            return float("-inf")

    def _nsucc(idx: int) -> float:
        st = population.get(idx) or {}
        try:
            return float(st.get("nsucc_mean") or 0.0)
        except Exception:
            return 0.0

    ordered: list[tuple[str, int]] = []
    for rank, idx in enumerate(sorted_policies):
        if int(idx) == int(target_idx):
            continue
        if rank < int(n_top):
            ordered.append(("top", int(idx)))
        elif rank < int(n_top) + int(n_mid):
            ordered.append(("mid", int(idx)))

    for tier_label, idx in ordered:
        st = population.get(idx)
        s_score = _score(idx)
        gap = s_score - t_score
        health = common_health_by_policy.get(idx)
        row: dict[str, Any] = {
            "source": idx,
            "source_tier": tier_label,
            "source_score": s_score,
            "score_gap": gap,
        }
        if st is None:
            row["filter"] = "no_state"
            considered.append(row)
            continue
        if not math.isfinite(s_score):
            row["filter"] = "source_score_nonfinite"
            considered.append(row)
            continue
        if health is None:
            row["filter"] = "common_health_missing"
            considered.append(row)
            continue
        row["common_health_reason"] = health.reason
        row["common_health_allowed"] = bool(health.eligible)
        if not health.eligible:
            row["filter"] = f"common_health:{health.reason}"
            considered.append(row)
            continue
        if gap < float(min_score_gap):
            row["filter"] = "score_gap_below_threshold"
            considered.append(row)
            continue
        delta, why = skill_delta_by_policy.get(idx, (None, "missing"))
        ok, reason, fallback = live_source_ok_for_chronic(
            skill_delta=delta,
            skill_delta_why=why,
            source_nsucc_mean=_nsucc(idx),
            target_nsucc_mean=float(target_nsucc_mean),
            source_recent_drop_ratio=float(drop_ratio_by_policy.get(idx, 0.0)),
        )
        row["filter"] = reason
        row["fallback"] = fallback
        row["skill_delta"] = delta
        row["skill_delta_why"] = why
        considered.append(row)
        if not ok:
            continue
        return ChronicSourceDecision(
            action="copy",
            source_policy=int(idx),
            source_state=st,
            tier=f"chronic_{tier_label}_live",
            checkpoint_type="latest_healthy",
            skip_reason=None,
            source_fallback_used=bool(fallback),
            mixed_tol_fallback_used=bool(fallback),
            source_skill_delta=delta,
            source_skill_delta_valid=delta is not None,
            source_skill_delta_invalid_reason="" if delta is not None else str(why),
            source_filter_reason=reason,
            final_source_filter_reason=reason,
            common_health_allowed=True,
            common_health_reason=str(health.reason),
            score_gap=float(gap),
            source_score=float(s_score),
            target_score=float(t_score),
            candidates_considered=considered,
        )

    return ChronicSourceDecision(
        action="skip",
        skip_reason="no_chronic_compatible_source",
        final_source_filter_reason="no_chronic_compatible_source",
        target_score=float(t_score),
        candidates_considered=considered,
    )
