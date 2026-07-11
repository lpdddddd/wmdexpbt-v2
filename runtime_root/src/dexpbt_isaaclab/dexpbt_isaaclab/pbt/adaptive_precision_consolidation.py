"""Curriculum-aware adaptive precision consolidation (progress_refine).

This is the generalized, ladder-free control plane for the
``cser_dexpbt_progress_refine`` profile.  It provides three pure, fully
JSON-serializable pieces that the PBT observer composes:

1. **Dimensionless curriculum progress** ``c_t`` and a stage schedule keyed on
   ``c_t`` instead of raw frames.
2. **General post-shrink probation** — every curriculum shrink automatically
   arms a probation window (no ``2 cm`` / fixed-ladder assumption); shrink is
   blocked until minimum frames and stable ticks are satisfied, ordinary
   rollback can be blocked, hard-collapse rollback stays available, and all
   state survives a restart.
3. **Retention-conditioned degradation rescue** — eligibility and source
   selection based on same-level retention, degradation streak, population rank,
   relative tolerance compatibility, source advantage normalized by population
   spread, collapse state and control-plane validity, with cooldown and
   per-lineage quota.  Fail-closed on any missing / non-finite / collapsed /
   incompatible source.

No physical distance constant or artificial tolerance ladder appears here;
tolerance keys use log quantization so retention state is isolated per precision
band automatically.  The module has no observer / env imports.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Optional

STAGE_EXPLORATION = "exploration"
STAGE_CONSOLIDATION = "consolidation"
STAGE_SELECTION = "selection"


def _finite(x: float, default: float = 0.0) -> float:
    v = float(x)
    if math.isnan(v) or math.isinf(v):
        return float(default)
    return v


@dataclass
class AdaptivePrecisionConfig:
    """All thresholds are configuration, never hard-coded tolerance constants."""

    enabled: bool = False
    eps: float = 1e-6

    # Stage boundaries on dimensionless curriculum progress c in [0, 1].
    stage_consolidation_start: float = 0.50
    stage_selection_start: float = 0.85

    # Log-quantization scale for per-precision-band retention keys.
    tolerance_log_quant: float = 0.05

    # Post-shrink probation.
    probation_min_frames: int = 200_000_000
    probation_min_stable_ticks: int = 4
    probation_min_same_level_retention: float = 0.90
    probation_block_ordinary_rollback: bool = True
    probation_allow_hard_collapse_rollback: bool = True
    probation_hard_collapse_mean: float = 3.0
    probation_hard_collapse_patience: int = 4

    # Retention-conditioned degradation rescue.
    rescue_enable: bool = False
    rescue_retention_threshold: float = 0.70
    rescue_degradation_patience: int = 3
    rescue_population_quantile: float = 0.60  # source must be in top (1-q) of pop
    rescue_source_min_retention: float = 0.85
    rescue_min_source_advantage_sigma: float = 1.0  # advantage in pop-std units
    rescue_max_log_tol_gap: float = 0.20
    rescue_cooldown_frames: int = 300_000_000
    rescue_max_per_lineage: int = 1
    rescue_require_source_control: bool = True
    rescue_reject_collapse_source: bool = True

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        self.enabled = bool(self.enabled)
        if not (self.eps > 0):
            raise ValueError("AdaptivePrecisionConfig.eps must be > 0")
        if not (0.0 <= self.stage_consolidation_start <= self.stage_selection_start <= 1.0):
            raise ValueError(
                "require 0 <= stage_consolidation_start <= stage_selection_start <= 1"
            )
        if not (self.tolerance_log_quant > 0):
            raise ValueError("tolerance_log_quant must be > 0")
        if int(self.probation_min_frames) < 0:
            raise ValueError("probation_min_frames must be >= 0")
        if int(self.probation_min_stable_ticks) < 0:
            raise ValueError("probation_min_stable_ticks must be >= 0")
        for name in (
            "probation_min_same_level_retention",
            "rescue_retention_threshold",
            "rescue_population_quantile",
            "rescue_source_min_retention",
        ):
            v = float(getattr(self, name))
            if not (0.0 <= v <= 1.0):
                raise ValueError(f"{name} must be in [0, 1]")
        if not (self.rescue_max_log_tol_gap >= 0):
            raise ValueError("rescue_max_log_tol_gap must be >= 0")


def curriculum_progress(
    eps_init: float, eps_target: float, eps_cur: float, *, eps: float = 1e-6
) -> float:
    """c = clip(log(eps_init / eps_cur) / log(eps_init / eps_target), 0, 1).

    Dimensionless: measures how far the curriculum has tightened from the initial
    tolerance toward the target, on a log scale.  Returns 0 when the denominator
    is degenerate (init == target) or inputs are non-finite.
    """
    ei = _finite(eps_init)
    et = _finite(eps_target)
    ec = _finite(eps_cur)
    if ei <= 0 or et <= 0 or ec <= 0:
        return 0.0
    denom = math.log((ei + eps) / (et + eps))
    if abs(denom) < 1e-9:
        return 0.0
    num = math.log((ei + eps) / (ec + eps))
    c = num / denom
    if math.isnan(c) or math.isinf(c):
        return 0.0
    return float(min(1.0, max(0.0, c)))


def resolve_stage(c: float, cfg: AdaptivePrecisionConfig) -> str:
    """Map curriculum progress to a consolidation stage."""
    cc = _finite(c)
    if cc < float(cfg.stage_consolidation_start):
        return STAGE_EXPLORATION
    if cc < float(cfg.stage_selection_start):
        return STAGE_CONSOLIDATION
    return STAGE_SELECTION


def log_tolerance_key(tolerance: float, quant: float) -> int:
    """Quantize a tolerance to a dimensionless log band index (ladder-free)."""
    t = _finite(tolerance)
    if t <= 0:
        return 0
    q = max(float(quant), 1e-6)
    return int(round(math.log(t) / q))


def log_tolerance_gap(tol_a: float, tol_b: float, *, eps: float = 1e-6) -> float:
    """|log((tol_a + eps) / (tol_b + eps))| — dimensionless tolerance distance."""
    a = _finite(tol_a)
    b = _finite(tol_b)
    return abs(math.log((a + eps) / (b + eps)))


@dataclass
class PrecisionRetentionState:
    sample_count: int = 0
    peak_nsucc_mean: float = 0.0
    last_nsucc_mean: float = 0.0

    def update(self, nsucc_mean: float) -> None:
        v = _finite(nsucc_mean)
        self.sample_count += 1
        self.last_nsucc_mean = v
        if v > self.peak_nsucc_mean:
            self.peak_nsucc_mean = v

    @property
    def retention(self) -> float:
        if self.peak_nsucc_mean <= 0:
            return 1.0
        return float(self.last_nsucc_mean / self.peak_nsucc_mean)


class PrecisionRetentionTracker:
    """Per-precision-band (log-quantized) retention, isolated across bands.

    A looser band's peak can never contaminate a stricter band's retention.
    Fully serializable for restart persistence.
    """

    def __init__(self, *, quant: float, lineage_id: str = "") -> None:
        self.quant = float(quant)
        self.lineage_id = str(lineage_id)
        self.states: dict[str, PrecisionRetentionState] = {}

    def _key(self, tolerance: float) -> str:
        return f"T{log_tolerance_key(tolerance, self.quant)}"

    def update(
        self,
        tolerance: float,
        nsucc_mean: float,
        *,
        lineage_id: Optional[str] = None,
        reset_on_lineage_change: bool = True,
    ) -> PrecisionRetentionState:
        if lineage_id is not None and reset_on_lineage_change and str(lineage_id) != self.lineage_id:
            self.states.clear()
            self.lineage_id = str(lineage_id)
        key = self._key(tolerance)
        st = self.states.setdefault(key, PrecisionRetentionState())
        st.update(nsucc_mean)
        return st

    def snapshot(self, tolerance: float, *, min_samples: int = 1) -> dict[str, Any]:
        st = self.states.get(self._key(tolerance))
        if st is None:
            return {"valid": False, "retention": 1.0, "peak": 0.0, "last": 0.0, "samples": 0}
        return {
            "valid": bool(st.sample_count >= int(min_samples)),
            "retention": st.retention,
            "peak": st.peak_nsucc_mean,
            "last": st.last_nsucc_mean,
            "samples": st.sample_count,
        }

    def reset(self) -> None:
        self.states.clear()

    def to_json(self) -> dict[str, Any]:
        return {
            "quant": self.quant,
            "lineage_id": self.lineage_id,
            "states": {k: asdict(v) for k, v in self.states.items()},
        }

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> "PrecisionRetentionTracker":
        obj = cls(quant=float(data.get("quant", 0.05)), lineage_id=str(data.get("lineage_id", "")))
        for k, v in (data.get("states") or {}).items():
            obj.states[str(k)] = PrecisionRetentionState(**v)
        return obj


@dataclass
class ProbationState:
    """Generalized post-shrink probation; armed after every shrink."""

    active: bool = False
    entry_tolerance: float = 0.0
    entry_frame: int = -1
    peak_nsucc_mean: float = 0.0
    current_nsucc_mean: float = 0.0
    same_level_retention: float = 1.0
    stable_ticks: int = 0
    hard_collapse_streak: int = 0

    def arm(self, *, tolerance: float, frame: int, nsucc_mean: float) -> None:
        self.active = True
        self.entry_tolerance = _finite(tolerance)
        self.entry_frame = int(frame)
        self.peak_nsucc_mean = _finite(nsucc_mean)
        self.current_nsucc_mean = _finite(nsucc_mean)
        self.same_level_retention = 1.0
        self.stable_ticks = 0
        self.hard_collapse_streak = 0

    def update(
        self,
        *,
        nsucc_mean: float,
        same_level_retention: float,
        frame: int,
        cfg: AdaptivePrecisionConfig,
    ) -> None:
        if not self.active:
            return
        v = _finite(nsucc_mean)
        self.current_nsucc_mean = v
        if v > self.peak_nsucc_mean:
            self.peak_nsucc_mean = v
        self.same_level_retention = _finite(same_level_retention, 1.0)
        if self.same_level_retention >= float(cfg.probation_min_same_level_retention):
            self.stable_ticks += 1
        else:
            self.stable_ticks = 0
        if v <= float(cfg.probation_hard_collapse_mean):
            self.hard_collapse_streak += 1
        else:
            self.hard_collapse_streak = 0

    def can_shrink(self, *, frame: int, cfg: AdaptivePrecisionConfig) -> tuple[bool, str]:
        if not self.active:
            return True, ""
        if int(frame) - int(self.entry_frame) < int(cfg.probation_min_frames):
            return False, "probation_min_frames"
        if self.stable_ticks < int(cfg.probation_min_stable_ticks):
            return False, "probation_min_stable_ticks"
        if self.same_level_retention < float(cfg.probation_min_same_level_retention):
            return False, "probation_low_retention"
        return True, ""

    def hard_collapsed(self, cfg: AdaptivePrecisionConfig) -> bool:
        return self.hard_collapse_streak >= int(cfg.probation_hard_collapse_patience)

    def maybe_exit(self, *, frame: int, cfg: AdaptivePrecisionConfig) -> None:
        ok, _ = self.can_shrink(frame=frame, cfg=cfg)
        if ok:
            self.active = False

    def to_json(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> "ProbationState":
        return cls(**{k: data[k] for k in data if k in cls.__dataclass_fields__})


@dataclass
class RescueDecision:
    eligible: bool
    reason: str
    source_idx: int = -1
    source_rank: int = -1
    source_advantage_sigma: float = 0.0


def evaluate_degradation_rescue(
    *,
    tolerance: float,
    nsucc_mean: float,
    same_level_retention: float,
    same_level_valid: bool,
    degradation_streak: int,
    executed_this_lineage: int,
    frames_since_last_rescue: int,
    cfg: AdaptivePrecisionConfig,
) -> tuple[bool, str]:
    """Retention-conditioned rescue *trigger* (no fixed tolerance thresholds)."""
    if not cfg.rescue_enable:
        return False, "disabled"
    if not bool(same_level_valid):
        return False, "retention_invalid"
    if not math.isfinite(_finite(same_level_retention, float("nan"))):
        return False, "retention_nonfinite"
    if float(same_level_retention) > float(cfg.rescue_retention_threshold):
        return False, "retention_ok"
    if int(executed_this_lineage) >= int(cfg.rescue_max_per_lineage):
        return False, "quota_exhausted"
    if int(frames_since_last_rescue) < int(cfg.rescue_cooldown_frames):
        return False, "cooldown"
    if int(degradation_streak) < int(cfg.rescue_degradation_patience):
        return False, "streak_below_patience"
    return True, "ready"


def select_rescue_source(
    *,
    target_tolerance: float,
    target_nsucc_mean: float,
    candidates: list[dict[str, Any]],
    population_nsucc_means: list[float],
    cfg: AdaptivePrecisionConfig,
) -> RescueDecision:
    """Pick a rescue source by population rank + retention + tolerance compat.

    Each candidate dict must provide: ``idx``, ``nsucc_mean``, ``retention``,
    ``tolerance``, ``ranking_score``, ``has_control`` (bool), ``collapsed``
    (bool).  Fail-closed: any missing control (strict), non-finite metric,
    collapsed source, or incompatible tolerance is rejected.
    """
    means = [_finite(m, float("-inf")) for m in population_nsucc_means if math.isfinite(_finite(m, float("nan")))]
    if len(means) >= 2:
        mu = sum(means) / len(means)
        var = sum((m - mu) ** 2 for m in means) / max(len(means) - 1, 1)
        sigma = math.sqrt(max(var, 0.0))
    else:
        mu = means[0] if means else 0.0
        sigma = 0.0
    sorted_means = sorted(means, reverse=True)
    n = len(sorted_means)
    # Rank threshold: source must be within the top (1 - quantile) of population.
    rank_cut_value = None
    if n >= 1:
        cut_index = max(0, min(n - 1, int(math.floor(float(cfg.rescue_population_quantile) * n))))
        rank_cut_value = sorted_means[cut_index]

    best: Optional[RescueDecision] = None
    best_score = float("-inf")
    for cand in candidates:
        idx = int(cand.get("idx", -1))
        s_mean = _finite(cand.get("nsucc_mean", float("nan")), float("nan"))
        s_ret = _finite(cand.get("retention", float("nan")), float("nan"))
        s_tol = _finite(cand.get("tolerance", float("nan")), float("nan"))
        s_rank_score = _finite(cand.get("ranking_score", float("nan")), float("nan"))
        has_control = bool(cand.get("has_control", False))
        collapsed = bool(cand.get("collapsed", False))
        if any(math.isnan(x) for x in (s_mean, s_ret, s_tol, s_rank_score)):
            continue  # non-finite -> reject
        if cfg.rescue_require_source_control and not has_control:
            continue
        if cfg.rescue_reject_collapse_source and collapsed:
            continue
        if s_ret < float(cfg.rescue_source_min_retention):
            continue
        if log_tolerance_gap(s_tol, target_tolerance, eps=cfg.eps) > float(cfg.rescue_max_log_tol_gap):
            continue
        if rank_cut_value is not None and s_mean < rank_cut_value:
            continue
        # Source advantage normalized by population spread.
        if sigma > 1e-9:
            adv_sigma = (s_mean - _finite(target_nsucc_mean)) / sigma
        else:
            adv_sigma = float("inf") if s_mean > _finite(target_nsucc_mean) else 0.0
        if adv_sigma < float(cfg.rescue_min_source_advantage_sigma):
            continue
        rank = sum(1 for m in sorted_means if m > s_mean)
        # Prefer the highest ranking-score compatible source.
        if s_rank_score > best_score:
            best_score = s_rank_score
            best = RescueDecision(
                eligible=True,
                reason="selected",
                source_idx=idx,
                source_rank=rank,
                source_advantage_sigma=float(min(adv_sigma, 1e6)),
            )
    if best is None:
        return RescueDecision(eligible=False, reason="no_compatible_source")
    return best
