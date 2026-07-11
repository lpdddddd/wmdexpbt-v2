"""Tolerance-local retention and conservative curriculum bridge helpers.

This module deliberately has no observer or environment dependency.  Keeping
the state serialisable makes the control-plane decision reproducible across a
worker restart and prevents a loose-tolerance peak from being used as evidence
of failure at a stricter tolerance.

Coarse bins remain available for general statistics.  Bridge shrink / probation
/ rollback decisions must key retention by concrete ladder level so that e.g.
0.022 peak never contaminates 0.021 or 0.0205.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Iterable


DEFAULT_RETENTION_BINS = (0.030, 0.025, 0.022, 0.020, 0.018)
DEFAULT_BRIDGE_LADDER = (
    0.0250, 0.0235, 0.0220, 0.0210, 0.0205,
    0.0200, 0.0195, 0.0190, 0.0185, 0.0180,
)


def parse_float_tuple(value: str | Iterable[float], default: tuple[float, ...]) -> tuple[float, ...]:
    """Parse a descending threshold list, tolerating old/missing config."""
    try:
        values = (
            [float(v.strip()) for v in value.split(",") if v.strip()]
            if isinstance(value, str)
            else [float(v) for v in value]
        )
    except (TypeError, ValueError):
        values = []
    return tuple(sorted(set(values), reverse=True)) or default


def tolerance_bin_id(tolerance: float, bins: tuple[float, ...] = DEFAULT_RETENTION_BINS) -> int:
    """Return bin 0 above the first edge and the final bin below all edges."""
    tol = float(tolerance)
    for idx, edge in enumerate(bins):
        if tol > float(edge):
            return idx
    return len(bins)


def bridge_level_id(
    tolerance: float, ladder: tuple[float, ...] = DEFAULT_BRIDGE_LADDER
) -> int:
    """Return the ladder index for ``tolerance`` (exact match preferred).

    Off-ladder values map to the nearest rung so retention stays level-local
    rather than falling into a coarse bin shared by multiple rungs.
    """
    levels = tuple(sorted(set(float(v) for v in ladder), reverse=True))
    if not levels:
        return 0
    tol = float(tolerance)
    best_idx = 0
    best_dist = float("inf")
    for idx, level in enumerate(levels):
        dist = abs(float(level) - tol)
        if dist <= 1e-9:
            return idx
        if dist < best_dist - 1e-15:
            best_idx = idx
            best_dist = dist
    return best_idx


def bridge_level_key(level_id: int) -> str:
    return f"L{int(level_id)}"


@dataclass
class ToleranceBinRetentionState:
    sample_count: int = 0
    peak_nsucc_mean: float = 0.0
    recent_mean: float = 0.0
    last_nsucc_mean: float = 0.0
    stable_ticks: int = 0
    first_frame: int = -1
    last_frame: int = -1
    low_retention_streak: int = 0

    def update(self, nsucc_mean: float, frame: int) -> None:
        value = float(nsucc_mean)
        self.sample_count += 1
        self.peak_nsucc_mean = max(float(self.peak_nsucc_mean), value)
        alpha = 2.0 / (min(self.sample_count, 4) + 1.0)
        self.recent_mean = value if self.sample_count == 1 else (
            alpha * value + (1.0 - alpha) * float(self.recent_mean)
        )
        self.last_nsucc_mean = value
        self.stable_ticks += 1
        self.first_frame = int(frame) if self.first_frame < 0 else self.first_frame
        self.last_frame = int(frame)

    def retention(self) -> float:
        return (
            float(self.last_nsucc_mean) / float(self.peak_nsucc_mean)
            if self.peak_nsucc_mean > 1e-9 else 1.0
        )


@dataclass
class ToleranceBinRetentionTracker:
    """Per-lineage retention state persisted through control JSON.

    Coarse bin keys (``\"0\"``, ``\"1\"``, ...) remain for general stats.
    When ``ladder`` is set, bridge decisions also maintain isolated
    ``L{level_id}`` keys so adjacent ladder rungs never share a peak.
    """

    bins: tuple[float, ...] = DEFAULT_RETENTION_BINS
    ladder: tuple[float, ...] = ()
    lineage_id: str | None = None
    states: dict[str, ToleranceBinRetentionState] = field(default_factory=dict)
    last_bridge_level_id: int = -1

    def current_bin(self, tolerance: float) -> int:
        return tolerance_bin_id(tolerance, self.bins)

    def current_level(self, tolerance: float) -> int:
        if not self.ladder:
            return -1
        return bridge_level_id(tolerance, self.ladder)

    def _decision_key(self, tolerance: float, *, use_bridge_levels: bool) -> str:
        if use_bridge_levels and self.ladder:
            return bridge_level_key(self.current_level(tolerance))
        return str(self.current_bin(tolerance))

    def update(
        self,
        *,
        tolerance: float,
        nsucc_mean: float,
        frame: int,
        lineage_id: str | None,
        reset_on_lineage_change: bool = True,
        use_bridge_levels: bool = False,
    ) -> int:
        if reset_on_lineage_change and self.lineage_id is not None and lineage_id != self.lineage_id:
            self.states.clear()
            self.last_bridge_level_id = -1
        self.lineage_id = lineage_id
        bin_id = self.current_bin(tolerance)
        self.states.setdefault(str(bin_id), ToleranceBinRetentionState()).update(
            nsucc_mean, frame
        )
        if use_bridge_levels and self.ladder:
            level_id = self.current_level(tolerance)
            if (
                self.last_bridge_level_id >= 0
                and level_id != self.last_bridge_level_id
            ):
                # New ladder rung: ensure a fresh state (setdefault creates empty).
                self.states.setdefault(
                    bridge_level_key(level_id), ToleranceBinRetentionState()
                )
            self.states.setdefault(
                bridge_level_key(level_id), ToleranceBinRetentionState()
            ).update(nsucc_mean, frame)
            self.last_bridge_level_id = int(level_id)
            return level_id
        return bin_id

    def state(
        self, tolerance: float, *, use_bridge_levels: bool = False
    ) -> ToleranceBinRetentionState:
        key = self._decision_key(tolerance, use_bridge_levels=use_bridge_levels)
        return self.states.get(key, ToleranceBinRetentionState())

    def is_valid(
        self,
        tolerance: float,
        *,
        min_samples: int,
        warmup_ticks: int,
        use_bridge_levels: bool = False,
    ) -> bool:
        state = self.state(tolerance, use_bridge_levels=use_bridge_levels)
        return (
            state.sample_count >= max(1, int(min_samples))
            and state.stable_ticks >= max(1, int(warmup_ticks))
        )

    def retention(self, tolerance: float, *, use_bridge_levels: bool = False) -> float:
        return self.state(tolerance, use_bridge_levels=use_bridge_levels).retention()

    def reset_bin_streak(self, tolerance: float, *, use_bridge_levels: bool = False) -> None:
        self.state(tolerance, use_bridge_levels=use_bridge_levels).low_retention_streak = 0

    def to_json(self) -> dict:
        return {
            "bins": list(self.bins),
            "ladder": list(self.ladder),
            "lineage_id": self.lineage_id,
            "last_bridge_level_id": int(self.last_bridge_level_id),
            "states": {key: asdict(value) for key, value in self.states.items()},
        }

    @classmethod
    def from_json(cls, data: dict | None) -> "ToleranceBinRetentionTracker":
        data = data or {}
        tracker = cls(
            bins=parse_float_tuple(data.get("bins", DEFAULT_RETENTION_BINS), DEFAULT_RETENTION_BINS),
            ladder=parse_float_tuple(data.get("ladder", ()), ()) if data.get("ladder") else (),
            lineage_id=data.get("lineage_id"),
            last_bridge_level_id=int(data.get("last_bridge_level_id", -1) or -1),
        )
        for key, value in (data.get("states") or {}).items():
            if isinstance(value, dict):
                known = {
                    name: value.get(name)
                    for name in ToleranceBinRetentionState.__dataclass_fields__
                }
                tracker.states[str(key)] = ToleranceBinRetentionState(**{
                    name: (default if default is not None else 0)
                    for name, default in known.items()
                })
        return tracker


def next_bridge_tolerance(
    current_tolerance: float,
    *,
    ladder: tuple[float, ...] = DEFAULT_BRIDGE_LADDER,
    mode: str = "ladder",
) -> float:
    """Return exactly one stricter bridge level; never skip a ladder rung."""
    current = float(current_tolerance)
    levels = tuple(sorted(set(float(v) for v in ladder), reverse=True))
    if mode == "micro_factor":
        factor = 0.98 if current > 0.022 else 0.99
        return current * factor
    for level in levels:
        if level < current - 1e-9:
            return level
    return current


def previous_bridge_tolerance(
    current_tolerance: float, *, ladder: tuple[float, ...] = DEFAULT_BRIDGE_LADDER
) -> float:
    """Return one easier level for bounded bridge rollback."""
    current = float(current_tolerance)
    levels = tuple(sorted(set(float(v) for v in ladder), reverse=True))
    for level in reversed(levels):
        if level > current + 1e-9:
            return level
    return current
