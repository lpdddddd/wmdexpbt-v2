"""PBT-WM Strategy Zoo v1.2/v1.4 profile-copy eligibility and decision logic."""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any

from dexpbt_isaaclab.world_model.wm_strategy_profiles import profile_copy_source_bonus


@dataclass
class ProfileCopyConfig:
    enable: bool = False
    after_frames: int = 1_000_000_000
    base_prob: float = 0.3
    keep_diversity_min_profiles: int = 3
    min_nsucc_gap: float = 0.0
    min_source_retention: float = 0.0
    cooldown_frames: int = 0
    bottom_k: int = 0
    top_k: int = 0
    prob_clip_max: float = 0.9
    source_min_nsucc_mean: float = 0.0
    source_min_frames: int = 0
    # Deep-diag fix (2026-06-20): block profile swap when the target slot has
    # not yet shown any meaningful learning signal. Weight-only copy still
    # proceeds; only the profile (zoo role) is preserved. This stops the
    # death spiral where bottom slots churn through profiles every restart.
    target_min_nsucc_mean: float = 0.0
    # When the target's last successful nsucc_max is also < this, fall through
    # the same gate. Set to 0.0 to disable the secondary check.
    target_min_nsucc_max: float = 0.0
    # Per-slot lock: if the target's policy_idx appears in this set, profile
    # swap is forbidden regardless of other gates (weight copy still ok).
    locked_target_slots: frozenset[int] = field(default_factory=frozenset)
    top1_override_enable: bool = False
    top1_override_after_frames: int = 0
    top1_override_min_nsucc_gap: float = 5.0
    top1_override_min_diversity: int = 2
    top1_override_allow_target_protected: bool = False


@dataclass
class ProfileCopyDecision:
    decision: str = "disabled"
    eligible: bool = False
    attempted: bool = False
    accepted: bool = False
    reject_reason: str = ""
    new_profile_name: str = ""
    source_profile_name: str = ""
    source_generation: int = 0
    weighted_prob: float = 0.0
    diversity_count: int = 0
    diversity_after_copy: int = 0
    target_need_factor: float = 1.0
    source_bonus: float = 1.0
    top1_override_used: bool = False
    source_frames: int = 0
    source_min_frames: int = 0
    source_nsucc_mean: float = float("nan")
    source_min_nsucc_mean: float = 0.0
    target_nsucc_mean: float = float("nan")
    target_min_nsucc_mean: float = 0.0
    target_locked: bool = False


def _policy_score(state: dict[str, Any] | None) -> float:
    if not state:
        return float("-inf")
    v = state.get("pbt_ranking_score", state.get("objective", float("-inf")))
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("-inf")


def _policy_nsucc_mean(state: dict[str, Any] | None, window_mean: float | None) -> float:
    if window_mean is not None and not math.isnan(window_mean):
        return float(window_mean)
    if not state:
        return float("nan")
    for key in ("nsucc_mean", "mean_prev_episode_successes"):
        if key in state:
            try:
                return float(state[key])
            except (TypeError, ValueError):
                pass
    return float("nan")


def _policy_nsucc_max(state: dict[str, Any] | None) -> float:
    if not state:
        return float("nan")
    for key in ("nsucc_max", "max_prev_episode_successes"):
        if key in state:
            try:
                return float(state[key])
            except (TypeError, ValueError):
                pass
    return float("nan")


def _policy_frames(state: dict[str, Any] | None) -> int:
    if not state:
        return 0
    for key in ("frame", "env_frames"):
        if key in state:
            try:
                return int(state[key])
            except (TypeError, ValueError):
                pass
    return 0


def _ranked_policies(
    population: dict[int, dict[str, Any] | None],
    *,
    window_nsucc: dict[int, float] | None = None,
) -> list[tuple[int, float]]:
    rows: list[tuple[int, float]] = []
    for pid, st in population.items():
        if st is None:
            continue
        score = _policy_score(st)
        if math.isnan(score) or math.isinf(score):
            wm = (window_nsucc or {}).get(int(pid))
            score = float(wm) if wm is not None else float("-inf")
        rows.append((int(pid), float(score)))
    rows.sort(key=lambda x: x[1], reverse=True)
    return rows


def _top1_override_ok(
    *,
    cfg: ProfileCopyConfig,
    env_frames: int,
    source_rank: int,
    source_nsucc: float,
    target_nsucc: float,
) -> bool:
    if not cfg.top1_override_enable:
        return False
    if int(env_frames) < int(cfg.top1_override_after_frames):
        return False
    if int(source_rank) != 0:
        return False
    gap = float(cfg.top1_override_min_nsucc_gap)
    if math.isnan(source_nsucc) or math.isnan(target_nsucc):
        return False
    if source_nsucc < target_nsucc + gap:
        return False
    return True


def compute_target_need_factor(
    *,
    target_rank: int,
    num_policies: int,
    bottom_k: int,
    target_retention: float | None,
    retention_collapse_threshold: float = 0.70,
) -> float:
    need = 1.0
    if bottom_k > 0 and target_rank >= max(0, num_policies - bottom_k):
        need = max(need, 1.25)
    if (
        target_retention is not None
        and not math.isnan(target_retention)
        and target_retention < retention_collapse_threshold
    ):
        need = max(need, 1.15)
    return float(need)


def compute_weighted_copy_prob(
    *,
    base_prob: float,
    source_profile: str,
    target_need_factor: float,
    prob_clip_max: float = 0.9,
) -> float:
    bonus = profile_copy_source_bonus(source_profile)
    p = float(base_prob) * float(bonus) * float(target_need_factor)
    return float(min(max(p, 0.0), prob_clip_max))


def decide_profile_copy(
    *,
    cfg: ProfileCopyConfig,
    env_frames: int,
    target_policy_idx: int,
    source_policy_idx: int,
    target_profile_name: str,
    source_profile_name: str,
    source_generation: int,
    diversity_count: int,
    profiles_set: set[str],
    population: dict[int, dict[str, Any] | None],
    source_meta: dict[str, Any] | None,
    target_meta: dict[str, Any] | None,
    source_checkpoint_type: str,
    target_protected_until: int = 0,
    last_profile_copy_frame: int = -10**18,
    window_nsucc: dict[int, float] | None = None,
    source_in_collapse: bool = False,
    source_frames: int | None = None,
    rng: random.Random | None = None,
) -> ProfileCopyDecision:
    """Evaluate profile-copy eligibility and draw copy decision."""
    out = ProfileCopyDecision(
        new_profile_name=str(target_profile_name or "legacy"),
        source_profile_name=str(source_profile_name or "legacy"),
        source_generation=int(source_generation),
        diversity_count=int(diversity_count),
        source_min_frames=int(cfg.source_min_frames),
        source_min_nsucc_mean=float(cfg.source_min_nsucc_mean),
        target_min_nsucc_mean=float(cfg.target_min_nsucc_mean),
        target_locked=bool(int(target_policy_idx) in cfg.locked_target_slots),
    )
    if not cfg.enable:
        out.decision = "disabled"
        out.reject_reason = "disabled"
        return out

    if int(env_frames) < int(cfg.after_frames):
        out.decision = "before_after_frames"
        out.reject_reason = "before_after_frames"
        return out

    if (
        int(cfg.cooldown_frames) > 0
        and int(last_profile_copy_frame) >= 0
        and int(env_frames) - int(last_profile_copy_frame) < int(cfg.cooldown_frames)
    ):
        out.decision = "cooldown"
        out.reject_reason = "cooldown"
        return out

    ranked = _ranked_policies(population, window_nsucc=window_nsucc)
    rank_by_pid = {pid: i for i, (pid, _) in enumerate(ranked)}
    num_policies = max(len(ranked), 1)
    target_rank = rank_by_pid.get(int(target_policy_idx), num_policies - 1)
    source_rank = rank_by_pid.get(int(source_policy_idx), num_policies)

    source_state = (population or {}).get(int(source_policy_idx))
    target_state = (population or {}).get(int(target_policy_idx))
    target_nsucc = _policy_nsucc_mean(
        target_state,
        (window_nsucc or {}).get(int(target_policy_idx)),
    )
    source_nsucc = _policy_nsucc_mean(
        source_state,
        (window_nsucc or {}).get(int(source_policy_idx)),
    )
    out.source_nsucc_mean = float(source_nsucc)
    out.target_nsucc_mean = float(target_nsucc)
    if source_frames is None:
        source_frames = _policy_frames(source_state)
    out.source_frames = int(source_frames)

    top1_override = _top1_override_ok(
        cfg=cfg,
        env_frames=int(env_frames),
        source_rank=int(source_rank),
        source_nsucc=float(source_nsucc),
        target_nsucc=float(target_nsucc),
    )
    out.top1_override_used = bool(top1_override)

    target_protected = int(env_frames) < int(target_protected_until)
    if target_protected and not (
        top1_override and cfg.top1_override_allow_target_protected
    ):
        out.decision = "target_protected"
        out.reject_reason = "target_protected"
        return out

    target_retention = None
    if target_meta:
        if target_meta.get("has_source_retention"):
            target_retention = float(target_meta.get("source_retention", float("nan")))
        elif "retention_used_for_shrink" in target_meta:
            target_retention = float(target_meta.get("retention_used_for_shrink", float("nan")))

    bottom_ok = (
        int(cfg.bottom_k) <= 0
        or target_rank >= max(0, num_policies - int(cfg.bottom_k))
        or (
            target_retention is not None
            and not math.isnan(target_retention)
            and target_retention < 0.70
        )
    )
    if not bottom_ok:
        out.decision = "target_not_bottom"
        out.reject_reason = "target_not_bottom"
        return out

    source_is_archive = str(source_checkpoint_type) in (
        "archive",
        "best_stable_archive",
    )
    top_ok = (
        source_is_archive
        or int(cfg.top_k) <= 0
        or source_rank < int(cfg.top_k)
    )
    if not top_ok:
        out.decision = "source_not_top"
        out.reject_reason = "source_not_top"
        return out

    if float(cfg.source_min_nsucc_mean) > 0.0:
        if math.isnan(source_nsucc) or source_nsucc < float(cfg.source_min_nsucc_mean):
            out.decision = "source_nsucc_too_low"
            out.reject_reason = "source_nsucc_too_low"
            return out

    if int(cfg.source_min_frames) > 0:
        if int(source_frames) < int(cfg.source_min_frames):
            out.decision = "source_frames_too_low"
            out.reject_reason = "source_frames_too_low"
            return out

    # Deep-diag fix (2026-06-20): block profile swap on broken targets.
    # When the target's nsucc has not crossed `target_min_nsucc_mean`, do
    # not swap the profile. This prevents the bottom-slot death spiral
    # observed in v17.3 5B (p00/p01 churned through 4-5 profiles each
    # while staying at nsucc<1). The bottom worker still inherits the
    # source's weights via the surrounding restart logic.
    if not top1_override and float(cfg.target_min_nsucc_mean) > 0.0:
        target_max = _policy_nsucc_max(target_state)
        target_max_floor = float(cfg.target_min_nsucc_max)
        # If the target has at least proven bursts of success
        # (nsucc_max above floor), allow profile swap; otherwise block.
        if (
            math.isnan(target_nsucc)
            or target_nsucc < float(cfg.target_min_nsucc_mean)
        ) and (
            target_max_floor <= 0.0
            or math.isnan(target_max)
            or target_max < target_max_floor
        ):
            out.decision = "target_nsucc_too_low"
            out.reject_reason = "target_nsucc_too_low"
            return out

    # Deep-diag fix (2026-06-20): per-slot lock. Slots with explicit zoo
    # roles (e.g. p00=ppo_only anchor, p01=AWR rescue) opt out of profile
    # swap entirely. Weight copy still proceeds.
    if not top1_override and int(target_policy_idx) in cfg.locked_target_slots:
        out.decision = "target_slot_locked"
        out.reject_reason = "target_slot_locked"
        return out

    if source_in_collapse:
        out.decision = "source_unhealthy"
        out.reject_reason = "source_unhealthy"
        return out

    if (
        float(cfg.min_nsucc_gap) > 0.0
        and not math.isnan(source_nsucc)
        and not math.isnan(target_nsucc)
        and source_nsucc < target_nsucc + float(cfg.min_nsucc_gap)
    ):
        out.decision = "nsucc_gap"
        out.reject_reason = "nsucc_gap"
        return out

    source_retention = float("nan")
    has_retention = False
    if source_meta:
        has_retention = bool(source_meta.get("has_source_retention", False))
        if has_retention:
            source_retention = float(source_meta.get("source_retention", float("nan")))
    if (
        float(cfg.min_source_retention) > 0.0
        and (
            not has_retention
            or math.isnan(source_retention)
            or source_retention < float(cfg.min_source_retention)
        )
    ):
        out.decision = "retention_low"
        out.reject_reason = "retention_low"
        return out

    src_profile = str(source_profile_name or "legacy")
    tgt_profile = str(target_profile_name or "legacy")
    if src_profile == tgt_profile:
        out.decision = "same_profile"
        out.reject_reason = "same_profile"
        out.new_profile_name = tgt_profile
        out.diversity_after_copy = int(diversity_count)
        return out

    profiles_after = set(profiles_set)
    profiles_after.discard(str(target_profile_name or "legacy"))
    profiles_after.add(str(source_profile_name or "legacy"))
    diversity_after = len(profiles_after)
    out.diversity_after_copy = int(diversity_after)
    effective_min_diversity = (
        int(cfg.top1_override_min_diversity)
        if top1_override
        else int(cfg.keep_diversity_min_profiles)
    )
    if diversity_after < effective_min_diversity:
        out.decision = "diversity_min"
        out.reject_reason = "diversity_min"
        return out

    out.eligible = True
    out.attempted = True
    out.target_need_factor = compute_target_need_factor(
        target_rank=target_rank,
        num_policies=num_policies,
        bottom_k=int(cfg.bottom_k),
        target_retention=target_retention,
    )
    out.source_bonus = profile_copy_source_bonus(source_profile_name)
    out.weighted_prob = compute_weighted_copy_prob(
        base_prob=float(cfg.base_prob),
        source_profile=source_profile_name,
        target_need_factor=out.target_need_factor,
        prob_clip_max=float(cfg.prob_clip_max),
    )

    rng = rng or random.Random(
        int(env_frames) + int(target_policy_idx) + int(source_policy_idx)
    )
    if rng.random() < out.weighted_prob:
        out.decision = "copied"
        out.accepted = True
        out.new_profile_name = str(source_profile_name or "legacy")
        out.reject_reason = ""
    else:
        out.decision = "rng_skip"
        out.reject_reason = "rng_skip"
    return out
