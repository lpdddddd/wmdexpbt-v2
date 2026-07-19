"""Tolerance-tagged exposure provenance for MSA probes.

The provenance rides INSIDE the existing curriculum control state dict
(see integration notes in arms.py / IMPLEMENTATION_REPORT.md): it is
serialized with `to_dict()` into the curriculum resume snapshot chain and
restored with `from_dict()`. There is deliberately NO parallel checkpoint
system — a checkpoint that predates MSA simply has no provenance key, which
`probe_due` treats as immediately due (per spec: resume from an old
checkpoint without provenance is due at once, not after a fresh 100M).

All frame quantities are WORKER-LOCAL frames on the worker's monotonic
frame ledger. Probe frames are charged to the same ledger (probes are not
free extra training).
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Any, Optional

# tau_plus proposal rule (meters).
TAU_FLOOR = 0.010

# A worker is due for a probe when it has NO tau_plus exposure within the
# most recent window of worker-local frames.
EXPOSURE_WINDOW_FRAMES = 100_000_000

# At/below the floor there is no tighter sentinel to probe.
SEGMENT_TYPES = ("common_probe", "second_probe", "normal_training")


def tau_plus_of(tau: float) -> float:
    """tau_plus = max(0.010, 0.95 * tau). Meters in, meters out."""
    if not (0.001 <= tau <= 0.2):
        raise ValueError(f"tau {tau} outside sane meter range [0.001, 0.2]")
    return max(TAU_FLOOR, 0.95 * tau)


@dataclass
class ExposureSegment:
    """One contiguous training segment tagged with the tolerance it ran at."""
    segment_type: str                  # one of SEGMENT_TYPES
    tau: float                         # tolerance the env actually used
    segment_start_frame: int           # worker-local ledger frames
    segment_end_frame: int
    intervention_id: Optional[str] = None

    def __post_init__(self) -> None:
        if self.segment_type not in SEGMENT_TYPES:
            raise ValueError(f"unknown segment_type {self.segment_type}")
        if self.segment_end_frame < self.segment_start_frame:
            raise ValueError("segment_end_frame < segment_start_frame")

    @property
    def frames(self) -> int:
        return self.segment_end_frame - self.segment_start_frame


@dataclass
class InterventionRecord:
    """Full provenance of one MSA intervention (probe cycle) for one worker."""
    intervention_id: str
    worker_id: int
    lineage_id: Optional[str]
    source_checkpoint_id: str          # checkpoint the run resumed from
    active_tau: float
    proposed_tau: float                # tau_plus
    due_frame: int                     # worker-local frame when due fired
    evaluator_manifest_id: str = ""    # ordered seed manifest identity
    # segments actually trained during this intervention
    segments: list[dict] = field(default_factory=list)
    # snapshot IDs (immutable, read-only eval snapshots)
    snapshot_pre_id: str = ""
    snapshot_mid_id: str = ""
    snapshot_post_id: str = ""
    # routing outcome
    recipient_status: str = ""         # "" | "common_only" | "top3_recipient"
                                       # | "permuted_recipient"
    # decision outcome
    first_blocking_gate: str = ""      # "" until DECIDE; "none" if promoted
    final_decision: str = ""           # "" | "promote" | "stay" | "abort"
    created_unix: float = field(default_factory=time.time)

    def add_segment(self, seg: ExposureSegment) -> None:
        self.segments.append(asdict(seg))


@dataclass
class MSAProvenance:
    """Per-worker exposure provenance embedded in the curriculum state."""
    worker_id: int
    exposures: list[dict] = field(default_factory=list)   # ExposureSegment dicts
    interventions: list[dict] = field(default_factory=list)
    last_probe_end_frame: Optional[int] = None

    # -- serialization into the curriculum snapshot chain -------------------
    def to_dict(self) -> dict[str, Any]:
        return {
            "worker_id": self.worker_id,
            "exposures": list(self.exposures),
            "interventions": list(self.interventions),
            "last_probe_end_frame": self.last_probe_end_frame,
        }

    @staticmethod
    def from_dict(d: Optional[dict]) -> Optional["MSAProvenance"]:
        """None (missing key in an old snapshot) stays None: immediately due."""
        if d is None:
            return None
        return MSAProvenance(
            worker_id=int(d["worker_id"]),
            exposures=list(d.get("exposures", [])),
            interventions=list(d.get("interventions", [])),
            last_probe_end_frame=d.get("last_probe_end_frame"),
        )

    # -- recording -----------------------------------------------------------
    def record_exposure(self, seg: ExposureSegment) -> None:
        self.exposures.append(asdict(seg))

    def record_intervention(self, rec: InterventionRecord) -> None:
        # Idempotent by intervention_id: a resumed / retried cycle REPLACES
        # its earlier (possibly aborted) record instead of duplicating it.
        self.interventions = [
            r for r in self.interventions
            if r.get("intervention_id") != rec.intervention_id
        ]
        self.interventions.append(asdict(rec))
        self.last_probe_end_frame = max(
            (s["segment_end_frame"] for s in rec.segments),
            default=self.last_probe_end_frame,
        )

    # -- queries --------------------------------------------------------------
    def tau_plus_exposure_frames_since(self, tau_plus: float,
                                       window_start_frame: int) -> int:
        """Worker-local frames trained at tolerance <= tau_plus (tighter or
        equal counts as tau_plus exposure) after window_start_frame."""
        total = 0
        for e in self.exposures:
            if float(e["tau"]) <= tau_plus + 1e-12:
                lo = max(int(e["segment_start_frame"]), window_start_frame)
                hi = int(e["segment_end_frame"])
                if hi > lo:
                    total += hi - lo
        return total


def probe_due(
    provenance: Optional[MSAProvenance],
    current_frame: int,
    active_tau: float,
    is_floor_worker: bool = False,
    window: int = EXPOSURE_WINDOW_FRAMES,
) -> bool:
    """Whether this worker is due for an MSA probe.

    - floor workers are never probed;
    - at/below TAU_FLOOR there is no tighter sentinel -> never due;
    - missing provenance (resume from a pre-MSA checkpoint) -> immediately
      due, without waiting for a fresh window;
    - otherwise due iff there is NO tau_plus exposure within the most
      recent `window` worker-local frames.
    """
    if is_floor_worker:
        return False
    if active_tau <= TAU_FLOOR + 1e-12:
        return False
    if provenance is None:
        return True
    tau_plus = tau_plus_of(active_tau)
    window_start = max(0, current_frame - window)
    return provenance.tau_plus_exposure_frames_since(tau_plus, window_start) == 0
