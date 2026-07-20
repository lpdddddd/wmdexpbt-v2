"""MSA cycle budget admission: worst-case dose reservation before PRE_EVAL.

A probe cycle must not open unless every admitted due worker has enough
remaining continuation frames to finish the *worst-case* training dose
implied by the controller's phase transitions. Expanding the +480M cap or
blindly parking mid-cycle is forbidden; instead we refuse admission.

Dose derivation (see ``worst_case_remaining_cycle_dose``):
  COMMON_PROBE  — ``train_until(start + packet_frames)``; observed frame may
                  land up to one committed tick past the barrier before hold
                  (tick quantization documented in controller.py).
  SECOND_PROBE  — every due worker is released; the barrier waits on
                  recipients for ``mid + packet_frames``. Non-recipients
                  train for the same wall-clock at their own tau, so the
                  worst-case dose is a full second packet for *every* due
                  worker (any of them may be a top-k recipient).
  COMMITTING    — workers are released and ``commit_tolerance`` waits for
                  one committed tick that reports ``final_tol``.

Reservation is persisted (``CYCLE_BUDGET_RESERVATION.json`` + busy marker
fields) so a sidecar restart never re-admits, double-reserves, or changes
eligibility for an in-flight intervention.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# Must match controller.PROBE_PACKET_FRAMES / continuation_guard committed
# cadence (stage-3 SELECTION: 80M). Duplicated here to avoid an import cycle
# (controller imports budget for admission).
PROBE_PACKET_FRAMES = 80_000_000
COMMITTED_TICK_FRAMES = PROBE_PACKET_FRAMES

RESERVATION_MARKER = "CYCLE_BUDGET_RESERVATION.json"
NO_NEW_CYCLE_MARKER = "MSA_NO_NEW_CYCLE.json"


def worst_case_remaining_cycle_dose(
    packet_frames: int = PROBE_PACKET_FRAMES,
    committed_tick_frames: int = COMMITTED_TICK_FRAMES,
) -> int:
    """Maximum committed training frames one due worker may advance during
    a newly admitted cycle (derived from controller phase transitions).

    required_reserve =
        (packet_frames + committed_tick_frames)   # COMMON_PROBE + quant
      + (packet_frames + committed_tick_frames)   # SECOND_PROBE + quant
      + committed_tick_frames                     # COMMITTING confirm tick

    With the current profile (packet == tick == 80M) this is 400_000_000.
    """
    pf = int(packet_frames)
    tick = int(committed_tick_frames)
    if pf <= 0 or tick <= 0:
        raise ValueError("packet/tick frames must be positive")
    common = pf + tick
    second = pf + tick
    commit = tick
    return common + second + commit


@dataclass
class WorkerBudgetView:
    worker_id: int
    latest_committed_frame: int
    cap_target: int                     # baseline_tip_frame + CONTINUATION_CAP
    remaining_frames: int = 0
    required_reserve: int = 0
    eligible: bool = False
    reserved_ceiling: int = 0           # latest + required_reserve


@dataclass
class CycleBudgetAdmission:
    intervention_id: str
    packet_frames: int
    committed_tick_frames: int
    required_reserve: int
    cap_target_by_worker: dict[int, int]
    remaining_at_admission: dict[int, int]
    reserved_frames_by_worker: dict[int, int]
    reserved_ceiling_by_worker: dict[int, int]
    budget_eligible_workers: list[int]
    budget_ineligible_workers: list[int]
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "intervention_id": self.intervention_id,
            "packet_frames": self.packet_frames,
            "committed_tick_frames": self.committed_tick_frames,
            "required_reserve": self.required_reserve,
            "cap_target_by_worker": {
                str(k): int(v) for k, v in self.cap_target_by_worker.items()},
            "remaining_at_admission": {
                str(k): int(v) for k, v in self.remaining_at_admission.items()},
            "reserved_frames_by_worker": {
                str(k): int(v) for k, v in self.reserved_frames_by_worker.items()},
            "reserved_ceiling_by_worker": {
                str(k): int(v) for k, v in self.reserved_ceiling_by_worker.items()},
            "budget_eligible_workers": list(self.budget_eligible_workers),
            "budget_ineligible_workers": list(self.budget_ineligible_workers),
            "timestamp": float(self.timestamp),
            "dose_formula": (
                "common=(packet+tick) + second=(packet+tick) + commit=tick; "
                "tick quantization reserved as one committed tick per "
                "train_until segment"
            ),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "CycleBudgetAdmission":
        def _imap(key: str) -> dict[int, int]:
            return {int(k): int(v) for k, v in (d.get(key) or {}).items()}
        return cls(
            intervention_id=str(d["intervention_id"]),
            packet_frames=int(d["packet_frames"]),
            committed_tick_frames=int(d["committed_tick_frames"]),
            required_reserve=int(d["required_reserve"]),
            cap_target_by_worker=_imap("cap_target_by_worker"),
            remaining_at_admission=_imap("remaining_at_admission"),
            reserved_frames_by_worker=_imap("reserved_frames_by_worker"),
            reserved_ceiling_by_worker=_imap("reserved_ceiling_by_worker"),
            budget_eligible_workers=[int(w) for w in
                                     d.get("budget_eligible_workers", [])],
            budget_ineligible_workers=[int(w) for w in
                                       d.get("budget_ineligible_workers", [])],
            timestamp=float(d.get("timestamp", 0.0)),
        )


def admit_cycle_budget(
    *,
    intervention_id: str,
    due_workers: list[int],
    latest_frame_by_worker: dict[int, int],
    cap_target_by_worker: dict[int, int],
    packet_frames: int = PROBE_PACKET_FRAMES,
    committed_tick_frames: int = COMMITTED_TICK_FRAMES,
) -> CycleBudgetAdmission:
    """Classify due workers as budget-eligible vs ineligible.

    A worker is eligible iff
        remaining = cap_target - latest_committed_frame
        remaining >= worst_case_remaining_cycle_dose(...)
    Ineligible workers must NOT enter PRE_EVAL / MSA_BUSY for this cycle.
    """
    reserve = worst_case_remaining_cycle_dose(
        packet_frames=packet_frames,
        committed_tick_frames=committed_tick_frames)
    eligible: list[int] = []
    ineligible: list[int] = []
    remaining: dict[int, int] = {}
    reserved: dict[int, int] = {}
    ceilings: dict[int, int] = {}
    caps: dict[int, int] = {}
    for w in sorted(due_workers):
        latest = int(latest_frame_by_worker[w])
        cap = int(cap_target_by_worker[w])
        rem = cap - latest
        remaining[w] = rem
        caps[w] = cap
        reserved[w] = reserve
        ceilings[w] = latest + reserve
        if rem >= reserve:
            eligible.append(w)
        else:
            ineligible.append(w)
    return CycleBudgetAdmission(
        intervention_id=intervention_id,
        packet_frames=int(packet_frames),
        committed_tick_frames=int(committed_tick_frames),
        required_reserve=reserve,
        cap_target_by_worker=caps,
        remaining_at_admission=remaining,
        reserved_frames_by_worker=reserved,
        reserved_ceiling_by_worker=ceilings,
        budget_eligible_workers=eligible,
        budget_ineligible_workers=ineligible,
    )


def write_reservation(run_dir: Path, admission: CycleBudgetAdmission) -> Path:
    path = Path(run_dir) / RESERVATION_MARKER
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(admission.to_dict(), indent=2))
    tmp.replace(path)
    return path


def load_reservation(run_dir: Path) -> Optional[CycleBudgetAdmission]:
    path = Path(run_dir) / RESERVATION_MARKER
    if not path.is_file():
        return None
    return CycleBudgetAdmission.from_dict(json.loads(path.read_text()))


def clear_reservation(run_dir: Path) -> None:
    path = Path(run_dir) / RESERVATION_MARKER
    if path.is_file():
        path.unlink()


def write_no_new_cycle(run_dir: Path, payload: dict) -> Path:
    path = Path(run_dir) / NO_NEW_CYCLE_MARKER
    body = dict(payload)
    body.setdefault("timestamp", time.time())
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(body, indent=2))
    tmp.replace(path)
    return path
