"""Explicit MSA probe state machine.

One enum-valued phase per population probe cycle:

    NORMAL -> PRE_EVAL -> COMMON_PROBE -> MID_EVAL -> ROUTE_SECOND_HALF
           -> SECOND_PROBE -> POST_EVAL -> DECIDE -> NORMAL | PROMOTE

The machine is the ONLY authority on what the trainer may do next; the
individual freeze/gate booleans of the base curriculum are never used to
express probe phases implicitly. Transitions are validated: an illegal jump
raises, so a bug cannot silently skip an evaluation or a decision.

The machine is serializable (to_dict/from_dict) and rides in the curriculum
resume snapshot chain next to MSAProvenance.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


class MSAPhase(str, Enum):
    NORMAL = "NORMAL"
    PRE_EVAL = "PRE_EVAL"
    COMMON_PROBE = "COMMON_PROBE"
    MID_EVAL = "MID_EVAL"
    ROUTE_SECOND_HALF = "ROUTE_SECOND_HALF"
    SECOND_PROBE = "SECOND_PROBE"
    POST_EVAL = "POST_EVAL"
    DECIDE = "DECIDE"
    PROMOTE = "PROMOTE"


# Legal transitions. DECIDE fans out to NORMAL (stay) or PROMOTE; PROMOTE
# returns to NORMAL once the new tau is applied.
_TRANSITIONS: dict[MSAPhase, tuple[MSAPhase, ...]] = {
    MSAPhase.NORMAL: (MSAPhase.PRE_EVAL,),
    MSAPhase.PRE_EVAL: (MSAPhase.COMMON_PROBE,),
    MSAPhase.COMMON_PROBE: (MSAPhase.MID_EVAL,),
    MSAPhase.MID_EVAL: (MSAPhase.ROUTE_SECOND_HALF,),
    MSAPhase.ROUTE_SECOND_HALF: (MSAPhase.SECOND_PROBE,),
    MSAPhase.SECOND_PROBE: (MSAPhase.POST_EVAL,),
    MSAPhase.POST_EVAL: (MSAPhase.DECIDE,),
    MSAPhase.DECIDE: (MSAPhase.NORMAL, MSAPhase.PROMOTE),
    MSAPhase.PROMOTE: (MSAPhase.NORMAL,),
}

# What the trainer is allowed to do in each phase.
_TRAINING_ALLOWED = {
    MSAPhase.NORMAL: True,
    MSAPhase.PRE_EVAL: False,          # snapshot taken at step boundary
    MSAPhase.COMMON_PROBE: True,       # training at tau_plus
    MSAPhase.MID_EVAL: False,
    MSAPhase.ROUTE_SECOND_HALF: False,
    MSAPhase.SECOND_PROBE: True,       # top-k (or permuted) at tau_plus
    MSAPhase.POST_EVAL: False,
    MSAPhase.DECIDE: False,
    MSAPhase.PROMOTE: True,
}

_TRAINING_TAU = {
    # which tolerance the env must be set to while training is allowed
    MSAPhase.NORMAL: "tau",
    MSAPhase.COMMON_PROBE: "tau_plus",
    MSAPhase.SECOND_PROBE: "tau_plus",
    MSAPhase.PROMOTE: "tau_plus",      # promotion makes tau_plus permanent
}


class IllegalTransition(RuntimeError):
    pass


@dataclass
class MSAProbeStateMachine:
    """Population-level probe phase + per-cycle bookkeeping."""
    phase: MSAPhase = MSAPhase.NORMAL
    intervention_id: str = ""
    cycle_index: int = 0
    history: list[str] = field(default_factory=list)

    def can_transition(self, nxt: MSAPhase) -> bool:
        return nxt in _TRANSITIONS[self.phase]

    def transition(self, nxt: MSAPhase, reason: str = "") -> None:
        if not self.can_transition(nxt):
            raise IllegalTransition(
                f"{self.phase.value} -> {nxt.value} is not a legal MSA "
                f"transition (legal: "
                f"{[t.value for t in _TRANSITIONS[self.phase]]})"
            )
        self.history.append(f"{self.phase.value}->{nxt.value}"
                            + (f" [{reason}]" if reason else ""))
        if nxt is MSAPhase.PRE_EVAL:
            self.cycle_index += 1
        self.phase = nxt

    def abort(self, reason: str = "") -> None:
        """Crash-cleanup escape hatch: from ANY phase back to NORMAL.

        Not a normal transition (recorded as ABORT in the history); used
        only by the controller's exception path after it restored every
        worker's own tau and cleared holds/pins.
        """
        self.history.append(f"{self.phase.value}->NORMAL [ABORT"
                            + (f": {reason}]" if reason else "]"))
        self.phase = MSAPhase.NORMAL

    # -- what the trainer consults every step --------------------------------
    def training_allowed(self) -> bool:
        return _TRAINING_ALLOWED[self.phase]

    def required_training_tau(self) -> Optional[str]:
        """'tau' | 'tau_plus' | None (no training in this phase)."""
        return _TRAINING_TAU.get(self.phase)

    # -- snapshot chain -------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return {
            "phase": self.phase.value,
            "intervention_id": self.intervention_id,
            "cycle_index": self.cycle_index,
            "history": list(self.history),
        }

    @staticmethod
    def from_dict(d: Optional[dict]) -> "MSAProbeStateMachine":
        if d is None:
            return MSAProbeStateMachine()
        return MSAProbeStateMachine(
            phase=MSAPhase(d.get("phase", "NORMAL")),
            intervention_id=d.get("intervention_id", ""),
            cycle_index=int(d.get("cycle_index", 0)),
            history=list(d.get("history", [])),
        )
