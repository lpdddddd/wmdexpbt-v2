"""Tolerance-local retention and conservative curriculum bridge helpers.

This module deliberately has no observer or environment dependency.  Keeping
the state serialisable makes the control-plane decision reproducible across a
worker restart and prevents a loose-tolerance peak from being used as evidence
of failure at a stricter tolerance.
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
    """Per-lineage, per-bin retention state persisted through control JSON."""

    bins: tuple[float, ...] = DEFAULT_RETENTION_BINS
    lineage_id: str | None = None
    states: dict[str, ToleranceBinRetentionState] = field(default_factory=dict)

    def current_bin(self, tolerance: float) -> int:
        return tolerance_bin_id(tolerance, self.bins)

    def update(self, *, tolerance: float, nsucc_mean: float, frame: int, lineage_id: str | None,
               reset_on_lineage_change: bool = True) -> int:
        if reset_on_lineage_change and self.lineage_id is not None and lineage_id != self.lineage_id:
            self.states.clear()
        self.lineage_id = lineage_id
        bin_id = self.current_bin(tolerance)
        state = self.states.setdefault(str(bin_id), ToleranceBinRetentionState())
        state.update(nsucc_mean, frame)
        return bin_id

    def state(self, tolerance: float) -> ToleranceBinRetentionState:
        return self.states.get(str(self.current_bin(tolerance)), ToleranceBinRetentionState())

    def is_valid(self, tolerance: float, *, min_samples: int, warmup_ticks: int) -> bool:
        state = self.state(tolerance)
        return state.sample_count >= max(1, int(min_samples)) and state.stable_ticks >= max(1, int(warmup_ticks))

    def retention(self, tolerance: float) -> float:
        return self.state(tolerance).retention()

    def reset_bin_streak(self, tolerance: float) -> None:
        self.state(tolerance).low_retention_streak = 0

    def to_json(self) -> dict:
        return {
            "bins": list(self.bins),
            "lineage_id": self.lineage_id,
            "states": {key: asdict(value) for key, value in self.states.items()},
        }

    @classmethod
    def from_json(cls, data: dict | None) -> "ToleranceBinRetentionTracker":
        data = data or {}
        tracker = cls(
            bins=parse_float_tuple(data.get("bins", DEFAULT_RETENTION_BINS), DEFAULT_RETENTION_BINS),
            lineage_id=data.get("lineage_id"),
        )
        for key, value in (data.get("states") or {}).items():
            if isinstance(value, dict):
                known = {name: value.get(name) for name in ToleranceBinRetentionState.__dataclass_fields__}
                tracker.states[str(key)] = ToleranceBinRetentionState(**{
                    name: default if default is not None else 0
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
