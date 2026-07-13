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
    # Primary: only the top fraction of the ranked population may be a source.
    # Example: 0.40 with 6 policies → ceil(0.40*6)=3 allowed sources.
    rescue_source_top_fraction: float = 0.40
    # Legacy CLI/profile alias. When set and top_fraction is left at the
    # sentinel below, top_fraction becomes ``1 - quantile``. Prefer setting
    # ``rescue_source_top_fraction`` explicitly in new profiles.
    rescue_population_quantile: float = 0.60
    rescue_source_min_retention: float = 0.85
    rescue_min_source_advantage_sigma: float = 1.0  # advantage in pop-std units
    rescue_max_log_tol_gap: float = 0.20
    rescue_cooldown_frames: int = 300_000_000
    rescue_max_per_lineage: int = 1
    rescue_require_source_control: bool = True
    rescue_reject_collapse_source: bool = True
    # Source retention validity (single-sample retention==1 is not enough).
    rescue_source_min_samples: int = 3
    rescue_source_min_stable_ticks: int = 2
    # Peer sidecar freshness.
    rescue_source_max_staleness_frames: int = 40_000_000
    rescue_require_lineage_match_with_checkpoint: bool = True
    # Archive rescue cascade.
    rescue_archive_enable: bool = True
    rescue_archive_max_age_frames: int = 1_000_000_000
    rescue_archive_require_retention: bool = True
    rescue_archive_min_nsucc_mean: float = 0.0
    # Fail-closed archive health gates (old archives missing fields are rejected).
    rescue_archive_min_nsucc_max: float = 0.0
    rescue_archive_require_age: bool = True
    rescue_archive_require_collapse_metadata: bool = True
    rescue_archive_require_quality_score: bool = True
    # Archive advantage: never use inf when population variance is degenerate.
    rescue_archive_min_population_samples: int = 2
    rescue_archive_advantage_sigma_floor: float = 1.0
    rescue_archive_min_absolute_advantage: float = 5.0
    # Emergency burn-in bypass (off by default for ordinary degradation).
    rescue_bypass_elimination_burnin: bool = False
    rescue_bypass_on_hard_collapse: bool = True
    rescue_min_frames_before_emergency_bypass: int = 40_000_000
    rescue_emergency_require_archive_or_stable_source: bool = True
    # Probation rollback target selection.
    probation_rollback_max_log_gap: float = 0.30
    probation_freeze_only_max_ticks: int = 3

    # Stage-aware lineage diversity + replacement budget (Profile 23).
    lineage_diversity_enable: bool = False
    min_distinct_lineages_exploration: int = 3
    min_distinct_lineages_consolidation: int = 2
    min_distinct_lineages_selection: int = 1
    replacement_budget_exploration: int = 1
    replacement_budget_consolidation: int = 1
    replacement_budget_selection: int = 2

    # Stage-aware probation (Profile 23); when probation_stage_aware_enable=False
    # use legacy probation_min_frames / probation_min_stable_ticks fields.
    probation_stage_aware_enable: bool = False
    probation_min_frames_exploration: int = 75_000_000
    probation_min_frames_other: int = 100_000_000
    probation_max_frames: int = 200_000_000
    probation_min_stable_ticks_exploration: int = 2
    probation_min_stable_ticks_other: int = 3
    probation_early_exit_retention: float = 0.97
    probation_normal_exit_retention: float = 0.92
    # Track nsucc drop: significant drop blocks early exit.
    probation_early_exit_nsucc_drop_frac: float = 0.15

    # Rescue freshness soft decay (Profile 23).
    rescue_source_freshness_mode: str = "hard_max"  # "hard_max" | "exp_decay"
    rescue_source_freshness_tau_frames: int = 80_000_000
    rescue_source_hard_max_age_frames: int = 160_000_000

    # Internal: when True, derive top_fraction from rescue_population_quantile.
    _derive_top_fraction_from_quantile: bool = False

    def __post_init__(self) -> None:
        if self._derive_top_fraction_from_quantile:
            self.rescue_source_top_fraction = max(
                0.0, min(1.0, 1.0 - float(self.rescue_population_quantile))
            )
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
            "rescue_source_top_fraction",
            "rescue_source_min_retention",
        ):
            v = float(getattr(self, name))
            if not (0.0 <= v <= 1.0):
                raise ValueError(f"{name} must be in [0, 1]")
        if not (self.rescue_max_log_tol_gap >= 0):
            raise ValueError("rescue_max_log_tol_gap must be >= 0")
        if not (self.probation_rollback_max_log_gap >= 0):
            raise ValueError("probation_rollback_max_log_gap must be >= 0")
        if int(self.rescue_source_min_samples) < 1:
            raise ValueError("rescue_source_min_samples must be >= 1")
        if int(self.rescue_source_min_stable_ticks) < 0:
            raise ValueError("rescue_source_min_stable_ticks must be >= 0")
        if int(self.probation_freeze_only_max_ticks) < 0:
            raise ValueError("probation_freeze_only_max_ticks must be >= 0")
        if int(self.rescue_archive_min_population_samples) < 1:
            raise ValueError("rescue_archive_min_population_samples must be >= 1")
        if not (float(self.rescue_archive_advantage_sigma_floor) > 0.0):
            raise ValueError("rescue_archive_advantage_sigma_floor must be > 0")
        if not (float(self.rescue_archive_min_absolute_advantage) >= 0.0):
            raise ValueError("rescue_archive_min_absolute_advantage must be >= 0")
        mode = str(self.rescue_source_freshness_mode or "hard_max").strip().lower()
        if mode not in ("hard_max", "exp_decay"):
            raise ValueError(
                "rescue_source_freshness_mode must be 'hard_max' or 'exp_decay', "
                f"got {self.rescue_source_freshness_mode!r}"
            )
        self.rescue_source_freshness_mode = mode
        if int(self.rescue_source_freshness_tau_frames) <= 0:
            raise ValueError("rescue_source_freshness_tau_frames must be > 0")
        if int(self.rescue_source_hard_max_age_frames) < 0:
            raise ValueError("rescue_source_hard_max_age_frames must be >= 0")
        for name in (
            "probation_early_exit_retention",
            "probation_normal_exit_retention",
            "probation_early_exit_nsucc_drop_frac",
        ):
            v = float(getattr(self, name))
            if not (0.0 <= v <= 1.0):
                raise ValueError(f"{name} must be in [0, 1]")
        for name in (
            "min_distinct_lineages_exploration",
            "min_distinct_lineages_consolidation",
            "min_distinct_lineages_selection",
            "replacement_budget_exploration",
            "replacement_budget_consolidation",
            "replacement_budget_selection",
            "probation_min_frames_exploration",
            "probation_min_frames_other",
            "probation_max_frames",
            "probation_min_stable_ticks_exploration",
            "probation_min_stable_ticks_other",
        ):
            if int(getattr(self, name)) < 0:
                raise ValueError(f"{name} must be >= 0")


def stage_min_distinct_lineages(stage: str, cfg: AdaptivePrecisionConfig) -> int:
    """Minimum distinct lineages required for ordinary copy at ``stage``."""
    s = str(stage)
    if s == STAGE_EXPLORATION:
        return int(cfg.min_distinct_lineages_exploration)
    if s == STAGE_CONSOLIDATION:
        return int(cfg.min_distinct_lineages_consolidation)
    return int(cfg.min_distinct_lineages_selection)


def stage_replacement_budget(stage: str, cfg: AdaptivePrecisionConfig) -> int:
    """Ordinary bottom-restart replacement budget for ``stage`` (per tick)."""
    s = str(stage)
    if s == STAGE_EXPLORATION:
        return int(cfg.replacement_budget_exploration)
    if s == STAGE_CONSOLIDATION:
        return int(cfg.replacement_budget_consolidation)
    return int(cfg.replacement_budget_selection)


def count_distinct_lineages(lineage_ids: list[str]) -> int:
    """Count non-empty distinct lineage ids."""
    return len({str(x) for x in lineage_ids if str(x)})


def copy_would_drop_below_min_lineages(
    lineages_by_policy: dict[int, str],
    target_idx: int,
    source_lineage: str,
    min_distinct: int,
) -> bool:
    """Return True if copying ``source_lineage`` onto ``target_idx`` drops diversity.

    After the copy, the target adopts ``source_lineage``.  Empty lineage strings
    are ignored in the distinct count.
    """
    projected = {int(k): str(v) for k, v in lineages_by_policy.items()}
    projected[int(target_idx)] = str(source_lineage)
    return count_distinct_lineages(list(projected.values())) < int(min_distinct)


def source_freshness_weight(
    age_frames: int, cfg: AdaptivePrecisionConfig
) -> tuple[bool, float, str]:
    """Evaluate rescue-source age under hard_max or exp_decay freshness mode.

    Returns ``(ok, weight, reason)``.  Rejected sources have ``ok=False`` and
    ``weight=0``.  Accepted hard_max sources keep weight 1.0; exp_decay sources
    get ``exp(-age / tau)`` (diagnostic only — ranking uses
    :func:`freshness_adjusted_ranking_score`).
    """
    age = max(0, int(age_frames))
    mode = str(cfg.rescue_source_freshness_mode or "hard_max")
    if mode == "exp_decay":
        hard_max = int(cfg.rescue_source_hard_max_age_frames)
        if age > hard_max:
            return False, 0.0, "source_adaptive_state_stale"
        tau = max(1, int(cfg.rescue_source_freshness_tau_frames))
        w = math.exp(-float(age) / float(tau))
        if not math.isfinite(w):
            return False, 0.0, "source_adaptive_state_nonfinite"
        return True, float(w), "ok_exp_decay"
    # hard_max (legacy): reject past max_staleness; weight stays 1 when ok.
    if age > int(cfg.rescue_source_max_staleness_frames):
        return False, 0.0, "source_adaptive_state_stale"
    return True, 1.0, "ok"


def freshness_adjusted_ranking_score(
    ranking_score: float,
    age_frames: int,
    cfg: AdaptivePrecisionConfig,
) -> tuple[bool, float, str]:
    """Age-adjusted ranking that never prefers older sources for any score sign.

    Under ``hard_max``, the raw ranking score is unchanged (when age is accepted).
    Under ``exp_decay``, an additive penalty ``age / tau`` is subtracted so that
    equal scores always prefer the fresher source even when ``ranking_score < 0``
    (multiplicative ``score * exp(-age/tau)`` would reverse the order for
    negatives).
    """
    score = _finite(ranking_score, float("nan"))
    if not math.isfinite(score):
        return False, float("-inf"), "ranking_score_nonfinite"
    ok, _weight, reason = source_freshness_weight(age_frames, cfg)
    if not ok:
        return False, float("-inf"), reason
    mode = str(cfg.rescue_source_freshness_mode or "hard_max")
    if mode != "exp_decay":
        return True, float(score), reason
    age = max(0, int(age_frames))
    tau = max(1, int(cfg.rescue_source_freshness_tau_frames))
    adjusted = float(score) - (float(age) / float(tau))
    if not math.isfinite(adjusted):
        return False, float("-inf"), "source_adaptive_state_nonfinite"
    return True, float(adjusted), reason


def probation_requirements_for_stage(
    stage: str, cfg: AdaptivePrecisionConfig
) -> tuple[int, int, float, float]:
    """Return ``(min_frames, min_stable_ticks, early_ret, normal_ret)`` for stage."""
    if not bool(cfg.probation_stage_aware_enable):
        return (
            int(cfg.probation_min_frames),
            int(cfg.probation_min_stable_ticks),
            float(cfg.probation_min_same_level_retention),
            float(cfg.probation_min_same_level_retention),
        )
    s = str(stage)
    if s == STAGE_EXPLORATION:
        return (
            int(cfg.probation_min_frames_exploration),
            int(cfg.probation_min_stable_ticks_exploration),
            float(cfg.probation_early_exit_retention),
            float(cfg.probation_normal_exit_retention),
        )
    return (
        int(cfg.probation_min_frames_other),
        int(cfg.probation_min_stable_ticks_other),
        float(cfg.probation_early_exit_retention),
        float(cfg.probation_normal_exit_retention),
    )


def rescue_source_top_k(population_size: int, top_fraction: float) -> int:
    """How many top-ranked policies may serve as adaptive rescue sources."""
    n = max(0, int(population_size))
    if n <= 0:
        return 0
    frac = max(0.0, min(1.0, float(top_fraction)))
    return max(1, int(math.ceil(frac * n)))


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
    stable_ticks: int = 0

    def update(
        self,
        nsucc_mean: float,
        *,
        min_retention_for_stable: float = 0.85,
    ) -> None:
        v = _finite(nsucc_mean)
        self.sample_count += 1
        self.last_nsucc_mean = v
        if v > self.peak_nsucc_mean:
            self.peak_nsucc_mean = v
        if self.retention >= float(min_retention_for_stable) and self.peak_nsucc_mean > 0:
            self.stable_ticks += 1
        else:
            self.stable_ticks = 0

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
        min_retention_for_stable: float = 0.85,
    ) -> PrecisionRetentionState:
        if lineage_id is not None and reset_on_lineage_change and str(lineage_id) != self.lineage_id:
            self.states.clear()
            self.lineage_id = str(lineage_id)
        key = self._key(tolerance)
        st = self.states.setdefault(key, PrecisionRetentionState())
        st.update(nsucc_mean, min_retention_for_stable=float(min_retention_for_stable))
        return st

    def snapshot(
        self,
        tolerance: float,
        *,
        min_samples: int = 1,
        min_stable_ticks: int = 0,
    ) -> dict[str, Any]:
        st = self.states.get(self._key(tolerance))
        if st is None:
            return {
                "valid": False,
                "retention": 1.0,
                "peak": 0.0,
                "last": 0.0,
                "samples": 0,
                "stable_ticks": 0,
            }
        return {
            "valid": bool(
                st.sample_count >= int(min_samples)
                and int(st.stable_ticks) >= int(min_stable_ticks)
            ),
            "retention": st.retention,
            "peak": st.peak_nsucc_mean,
            "last": st.last_nsucc_mean,
            "samples": st.sample_count,
            "stable_ticks": int(st.stable_ticks),
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
    # Wider tolerance observed immediately before the shrink that armed
    # probation. Used as a ladder-free hard-collapse rollback target.
    previous_tolerance_before_entry: float = 0.0
    source_tolerance: float = 0.0
    peak_nsucc_mean: float = 0.0
    current_nsucc_mean: float = 0.0
    same_level_retention: float = 1.0
    stable_ticks: int = 0
    hard_collapse_streak: int = 0

    def arm(
        self,
        *,
        tolerance: float,
        frame: int,
        nsucc_mean: float,
        previous_tolerance: float | None = None,
    ) -> None:
        self.active = True
        self.entry_tolerance = _finite(tolerance)
        self.entry_frame = int(frame)
        prev = (
            _finite(previous_tolerance)
            if previous_tolerance is not None
            else 0.0
        )
        self.previous_tolerance_before_entry = prev
        self.source_tolerance = prev
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

    def evaluate_exit(
        self,
        *,
        frame: int,
        stage: str,
        cfg: AdaptivePrecisionConfig,
    ) -> tuple[str, str]:
        """Stage-aware probation exit decision for Profile 23.

        Returns ``(kind, reason)`` where kind is one of
        ``early_exit`` / ``normal_exit`` / ``max_frames_timeout`` / ``continue``.
        Does not mutate ``active``; the caller clears or rolls back.
        """
        if not self.active:
            return "continue", "inactive"
        frames_in = int(frame) - int(self.entry_frame)
        if frames_in < 0:
            frames_in = 0
        min_frames, min_stable, early_ret, normal_ret = probation_requirements_for_stage(
            stage, cfg
        )
        # Timeout takes priority when still active past max frames.
        if frames_in >= int(cfg.probation_max_frames):
            return "max_frames_timeout", "probation_max_frames"

        hard = self.hard_collapsed(cfg)
        peak = float(self.peak_nsucc_mean)
        cur = float(self.current_nsucc_mean)
        nsucc_drop_block = False
        if peak > 0.0:
            drop_floor = peak * (1.0 - float(cfg.probation_early_exit_nsucc_drop_frac))
            if cur < drop_floor:
                nsucc_drop_block = True

        # Early exit: exploration min frames (or stage min when not exploration),
        # high retention, stable ticks, no hard collapse, no significant nsucc drop.
        early_min = int(cfg.probation_min_frames_exploration)
        if frames_in >= early_min and not hard and not nsucc_drop_block:
            if (
                float(self.same_level_retention) >= float(early_ret)
                and int(self.stable_ticks) >= int(min_stable)
            ):
                return "early_exit", "probation_early_exit"

        # Normal exit: stage min frames + normal retention + stable ticks.
        if frames_in >= int(min_frames):
            if int(self.stable_ticks) < int(min_stable):
                return "continue", "probation_min_stable_ticks"
            if float(self.same_level_retention) < float(normal_ret):
                return "continue", "probation_low_retention"
            return "normal_exit", "probation_normal_exit"
        return "continue", "probation_min_frames"

    def maybe_exit(self, *, frame: int, cfg: AdaptivePrecisionConfig) -> None:
        """Legacy Profile-22 exit: clear active when ``can_shrink`` is satisfied."""
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
    allowed_source_ids: Optional[set[int]] = None,
    sorted_population: Optional[list[int]] = None,
) -> RescueDecision:
    """Pick a rescue source by population rank + retention + tolerance compat.

    Each candidate dict must provide: ``idx``, ``nsucc_mean``, ``retention``,
    ``tolerance``, ``ranking_score``, ``has_control`` (bool), ``collapsed``
    (bool).  Fail-closed: any missing control (strict), non-finite metric,
    collapsed source, or incompatible tolerance is rejected.

    Population top-fraction filter (explicit semantics):
      top_k = ceil(rescue_source_top_fraction * valid_population_size)
      only policies in ``sorted_population[:top_k]`` (or ``allowed_source_ids``)
      may be selected.  When neither is provided, fall back to an nsucc-mean
      rank cut equivalent to the top-fraction of the finite mean list.
    """
    means = [
        _finite(m, float("-inf"))
        for m in population_nsucc_means
        if math.isfinite(_finite(m, float("nan")))
    ]
    if len(means) >= 2:
        mu = sum(means) / len(means)
        var = sum((m - mu) ** 2 for m in means) / max(len(means) - 1, 1)
        sigma = math.sqrt(max(var, 0.0))
    else:
        mu = means[0] if means else 0.0
        sigma = 0.0
    sorted_means = sorted(means, reverse=True)
    n = len(sorted_means)

    allowed: Optional[set[int]] = None
    if allowed_source_ids is not None:
        allowed = {int(x) for x in allowed_source_ids}
    elif sorted_population is not None and len(sorted_population) > 0:
        top_k = rescue_source_top_k(
            len(sorted_population), float(cfg.rescue_source_top_fraction)
        )
        allowed = {int(x) for x in list(sorted_population)[:top_k]}
    else:
        allowed = None

    rank_cut_value = None
    if allowed is None and n >= 1:
        top_k = rescue_source_top_k(n, float(cfg.rescue_source_top_fraction))
        rank_cut_value = sorted_means[min(n - 1, top_k - 1)]

    best: Optional[RescueDecision] = None
    best_score = float("-inf")
    for cand in candidates:
        idx = int(cand.get("idx", -1))
        if allowed is not None and idx not in allowed:
            continue
        s_mean = _finite(cand.get("nsucc_mean", float("nan")), float("nan"))
        s_ret = _finite(cand.get("retention", float("nan")), float("nan"))
        s_tol = _finite(cand.get("tolerance", float("nan")), float("nan"))
        s_rank_score = _finite(cand.get("ranking_score", float("nan")), float("nan"))
        has_control = bool(cand.get("has_control", False))
        collapsed = bool(cand.get("collapsed", False))
        retention_valid = bool(cand.get("retention_valid", False))
        if any(math.isnan(x) for x in (s_mean, s_ret, s_tol, s_rank_score)):
            continue  # non-finite -> reject
        if not retention_valid:
            continue
        if cfg.rescue_require_source_control and not has_control:
            continue
        if cfg.rescue_reject_collapse_source and collapsed:
            continue
        if s_ret < float(cfg.rescue_source_min_retention):
            continue
        if log_tolerance_gap(s_tol, target_tolerance, eps=cfg.eps) > float(
            cfg.rescue_max_log_tol_gap
        ):
            continue
        age = int(cand.get("source_age_frames", 0) or 0)
        if "source_age_frames" in cand or str(cfg.rescue_source_freshness_mode) == "exp_decay":
            ok_f, adjusted_score, _fr = freshness_adjusted_ranking_score(
                s_rank_score, age, cfg
            )
            if not ok_f:
                continue
        else:
            adjusted_score = float(s_rank_score)
        if rank_cut_value is not None and s_mean < rank_cut_value - 1e-12:
            continue
        # Source advantage normalized by population spread.
        if sigma > 1e-9:
            adv_sigma = (s_mean - _finite(target_nsucc_mean)) / sigma
        else:
            adv_sigma = float("inf") if s_mean > _finite(target_nsucc_mean) else 0.0
        if adv_sigma < float(cfg.rescue_min_source_advantage_sigma):
            continue
        if sorted_population is not None and idx in sorted_population:
            rank = int(list(sorted_population).index(idx))
        else:
            rank = sum(1 for m in sorted_means if m > s_mean)
        # Prefer the highest age-adjusted ranking score. Additive freshness
        # penalty keeps "fresher is better" for both positive and negative scores.
        if adjusted_score > best_score:
            best_score = adjusted_score
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



# ---------------------------------------------------------------------------
# Adaptive hard-collapse rollback target selection (ladder-free, log-gap capped)
# ---------------------------------------------------------------------------

ROLLBACK_SOURCE_NONE = 0
ROLLBACK_SOURCE_PREVIOUS = 1
ROLLBACK_SOURCE_SOURCE_TOL = 2
ROLLBACK_SOURCE_LAST_MASTERED_SAFE = 3
ROLLBACK_SOURCE_SCALE_FALLBACK = 4

ROLLBACK_SOURCE_NAME = {
    ROLLBACK_SOURCE_NONE: "none",
    ROLLBACK_SOURCE_PREVIOUS: "previous_tolerance",
    ROLLBACK_SOURCE_SOURCE_TOL: "source_tolerance",
    ROLLBACK_SOURCE_LAST_MASTERED_SAFE: "last_mastered_safe",
    ROLLBACK_SOURCE_SCALE_FALLBACK: "scale_fallback",
}


@dataclass
class AdaptiveRollbackDecision:
    eligible: bool
    target_tolerance: float
    source: str
    source_id: int
    log_gap: float
    reason: str


def select_adaptive_rollback_target(
    *,
    current_tolerance: float,
    previous_tolerance_before_entry: float | None,
    source_tolerance: float | None,
    last_mastered_safe_tolerance: float | None,
    initial_tolerance: float | None,
    shrink_factor: float,
    max_log_gap: float,
    eps: float = 1e-6,
) -> AdaptiveRollbackDecision:
    """Pick a locally wider rollback target with fixed priority + log-gap cap.

    Priority (first legal wins — never ``max(wider)``):
      1. previous_tolerance_before_entry
      2. source_tolerance
      3. last_mastered_safe_tolerance
      4. scale-derived fallback ``current / shrink_factor`` (capped by initial)
    """
    cur = _finite(current_tolerance, float("nan"))
    if not math.isfinite(cur) or cur <= 0:
        return AdaptiveRollbackDecision(
            eligible=False,
            target_tolerance=0.0,
            source="none",
            source_id=ROLLBACK_SOURCE_NONE,
            log_gap=0.0,
            reason="current_tolerance_invalid",
        )
    sf = float(shrink_factor)
    if not (0.0 < sf < 1.0):
        return AdaptiveRollbackDecision(
            eligible=False,
            target_tolerance=0.0,
            source="none",
            source_id=ROLLBACK_SOURCE_NONE,
            log_gap=0.0,
            reason="shrink_factor_invalid",
        )
    init = _finite(initial_tolerance, float("nan")) if initial_tolerance is not None else float("nan")
    max_gap = float(max_log_gap)

    def _ok(cand: float | None) -> tuple[bool, float, float]:
        if cand is None:
            return False, 0.0, 0.0
        t = _finite(cand, float("nan"))
        if not math.isfinite(t) or t <= cur + eps:
            return False, 0.0, 0.0
        gap = log_tolerance_gap(t, cur, eps=eps)
        if gap > max_gap + 1e-12:
            return False, t, gap
        return True, t, gap

    ordered = [
        (ROLLBACK_SOURCE_PREVIOUS, previous_tolerance_before_entry),
        (ROLLBACK_SOURCE_SOURCE_TOL, source_tolerance),
        (ROLLBACK_SOURCE_LAST_MASTERED_SAFE, last_mastered_safe_tolerance),
    ]
    for src_id, cand in ordered:
        ok, t, gap = _ok(cand)
        if ok:
            return AdaptiveRollbackDecision(
                eligible=True,
                target_tolerance=float(t),
                source=ROLLBACK_SOURCE_NAME[src_id],
                source_id=int(src_id),
                log_gap=float(gap),
                reason="selected",
            )

    # Scale-derived fallback: widen by 1/shrink_factor, never above initial.
    fallback = cur / sf
    if math.isfinite(init) and init > 0:
        fallback = min(fallback, init)
    ok, t, gap = _ok(fallback)
    if ok:
        return AdaptiveRollbackDecision(
            eligible=True,
            target_tolerance=float(t),
            source=ROLLBACK_SOURCE_NAME[ROLLBACK_SOURCE_SCALE_FALLBACK],
            source_id=ROLLBACK_SOURCE_SCALE_FALLBACK,
            log_gap=float(gap),
            reason="scale_fallback",
        )
    return AdaptiveRollbackDecision(
        eligible=False,
        target_tolerance=0.0,
        source="none",
        source_id=ROLLBACK_SOURCE_NONE,
        log_gap=0.0,
        reason="no_legal_wider_tolerance",
    )


def validate_source_retention_snapshot(
    snap: dict[str, Any],
    *,
    cfg: AdaptivePrecisionConfig,
) -> tuple[bool, str]:
    """Fail-closed retention validity for adaptive rescue sources."""
    if not isinstance(snap, dict):
        return False, "source_retention_warmup"
    samples = int(snap.get("samples", 0) or 0)
    stable = int(snap.get("stable_ticks", 0) or 0)
    ret = _finite(snap.get("retention", float("nan")), float("nan"))
    peak = _finite(snap.get("peak", float("nan")), float("nan"))
    last = _finite(snap.get("last", float("nan")), float("nan"))
    if samples < int(cfg.rescue_source_min_samples):
        return False, "source_retention_insufficient_samples"
    if stable < int(cfg.rescue_source_min_stable_ticks):
        return False, "source_retention_insufficient_stability"
    if any(not math.isfinite(x) for x in (ret, peak, last)):
        return False, "source_retention_nonfinite"
    if not bool(snap.get("valid", False)):
        return False, "source_retention_warmup"
    if ret < float(cfg.rescue_source_min_retention):
        return False, "source_retention_below_min"
    return True, "ok"


def evaluate_sidecar_freshness(
    *,
    sidecar: dict[str, Any],
    population_frame: int,
    population_lineage_id: str = "",
    checkpoint_frame: int | None = None,
    interval_steps: int = 10_000_000,
    cfg: AdaptivePrecisionConfig,
    require_freshness_fields: bool = True,
) -> tuple[bool, str]:
    """Peer sidecar freshness / lineage gate. Fail-closed when strict."""
    if not isinstance(sidecar, dict):
        return False, "source_adaptive_sidecar_schema_error"
    has_frame = "last_update_frame" in sidecar
    if not has_frame:
        if require_freshness_fields:
            return False, "source_sidecar_freshness_unknown"
        return True, "freshness_unknown_compat"
    try:
        last_update = int(sidecar.get("last_update_frame", -(10**18)))
    except Exception:
        return False, "source_adaptive_state_nonfinite"
    pop_frame = int(population_frame)
    if pop_frame < last_update:
        return False, "source_adaptive_state_stale"
    age = pop_frame - last_update
    mode = str(cfg.rescue_source_freshness_mode or "hard_max")
    if mode == "exp_decay":
        ok_f, _w, reason_f = source_freshness_weight(age, cfg)
        if not ok_f:
            return False, reason_f
    elif age > int(cfg.rescue_source_max_staleness_frames):
        return False, "source_adaptive_state_stale"
    if cfg.rescue_require_lineage_match_with_checkpoint:
        side_lin = str(sidecar.get("last_update_lineage_id", "") or "")
        pop_lin = str(population_lineage_id or "")
        if side_lin and pop_lin and side_lin != pop_lin:
            return False, "source_adaptive_lineage_mismatch"
    if checkpoint_frame is not None and "last_checkpoint_frame" in sidecar:
        try:
            side_ckpt = int(sidecar.get("last_checkpoint_frame", -1))
        except Exception:
            return False, "source_adaptive_state_nonfinite"
        if side_ckpt >= 0 and abs(int(checkpoint_frame) - side_ckpt) > max(
            1, int(interval_steps)
        ):
            return False, "source_adaptive_state_stale"
    if mode == "exp_decay":
        return True, "ok_exp_decay"
    return True, "ok"


ADAPTIVE_SIDECAR_SCHEMA_VERSION = 2
