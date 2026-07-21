"""Per-worker MSA control state for the fused DexPBT training path.

Modes: BUILD | PROBE | HOLD. ADVANCE / ROLLBACK are atomic events, not
long-lived modes. Schema is versioned via evidence_version /
msa_checkpoint_version for old-checkpoint migration.
"""

from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Optional

from dexpbt_isaaclab.msa.fusion_config import (
    EVIDENCE_VERSION,
    MSA_CHECKPOINT_VERSION,
    MSAFusionConfig,
)

MODE_BUILD = "BUILD"
MODE_PROBE = "PROBE"
MODE_HOLD = "HOLD"
VALID_MODES = frozenset({MODE_BUILD, MODE_PROBE, MODE_HOLD})


@dataclass
class WorkerMSAState:
    worker_id: int = 0
    mode: str = MODE_BUILD
    current_tolerance: float = 0.0
    anchor_tolerance: float = 0.0
    anchor_checkpoint: str = ""
    committed_stage: str = "BUILD"  # BUILD | MID | STRICT (tolerance band)
    pending_target_tolerance: float = 0.0
    cycle_id: str = ""
    probe_start_frame: int = -1
    probe_budget_frames: int = 0
    last_decision_frame: int = -1
    consecutive_passes: int = 0
    consecutive_failures: int = 0
    hold_until_frame: int = 0
    pre_probe_metrics: dict[str, Any] = field(default_factory=dict)
    post_probe_metrics: dict[str, Any] = field(default_factory=dict)
    decision_metrics: dict[str, Any] = field(default_factory=dict)
    decision_reason: str = ""
    evidence_version: str = EVIDENCE_VERSION
    msa_checkpoint_version: str = MSA_CHECKPOINT_VERSION
    config_hash: str = ""
    advance_count: int = 0
    rollback_count: int = 0
    hold_count: int = 0
    # Cleared on PBT copy — never inherit donor evidence.
    pass_evidence: dict[str, Any] = field(default_factory=dict)
    transaction_open: bool = False
    anchor_fingerprint: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: dict[str, Any] | None) -> "WorkerMSAState":
        if not raw:
            return cls()
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in raw.items() if k in known})

    def validate(self) -> None:
        if self.mode not in VALID_MODES:
            raise ValueError(f"invalid MSA mode: {self.mode}")
        if self.worker_id < 0:
            raise ValueError("worker_id must be >= 0")


def migrate_worker_state(
    raw: dict[str, Any] | None,
    *,
    worker_id: int,
    current_tolerance: float,
    anchor_checkpoint: str = "",
    cfg: Optional[MSAFusionConfig] = None,
) -> tuple[WorkerMSAState, list[str]]:
    """Migrate old / missing schema -> WorkerMSAState.

    Old checkpoints lack MSA fusion keys: start in BUILD, register the
    current reliable checkpoint as the initial anchor, and do NOT invent
    historical pass evidence.
    """
    logs: list[str] = []
    if not raw:
        logs.append(
            f"worker {worker_id}: no MSA state -> migrate to BUILD with "
            f"anchor_tol={current_tolerance}")
        st = WorkerMSAState(
            worker_id=worker_id,
            mode=MODE_BUILD,
            current_tolerance=float(current_tolerance),
            anchor_tolerance=float(current_tolerance),
            anchor_checkpoint=str(anchor_checkpoint or ""),
            evidence_version=EVIDENCE_VERSION,
            msa_checkpoint_version=MSA_CHECKPOINT_VERSION,
        )
        if cfg is not None:
            st.committed_stage = cfg.stage_for_tolerance(current_tolerance)
            st.config_hash = cfg.config_hash()
        return st, logs

    st = WorkerMSAState.from_dict(raw)
    st.worker_id = int(worker_id)
    if st.evidence_version != EVIDENCE_VERSION:
        logs.append(
            f"worker {worker_id}: evidence_version "
            f"{st.evidence_version!r} -> {EVIDENCE_VERSION!r}")
        st.evidence_version = EVIDENCE_VERSION
    if st.msa_checkpoint_version != MSA_CHECKPOINT_VERSION:
        logs.append(
            f"worker {worker_id}: checkpoint_version "
            f"{st.msa_checkpoint_version!r} -> {MSA_CHECKPOINT_VERSION!r}")
        st.msa_checkpoint_version = MSA_CHECKPOINT_VERSION
    if st.mode not in VALID_MODES:
        logs.append(f"worker {worker_id}: invalid mode {st.mode!r} -> BUILD")
        st.mode = MODE_BUILD
    if st.current_tolerance <= 0 and current_tolerance > 0:
        st.current_tolerance = float(current_tolerance)
    if st.anchor_tolerance <= 0 and st.current_tolerance > 0:
        st.anchor_tolerance = float(st.current_tolerance)
        logs.append(
            f"worker {worker_id}: seeded anchor_tolerance from current")
    if not st.anchor_checkpoint and anchor_checkpoint:
        st.anchor_checkpoint = str(anchor_checkpoint)
    # Never invent pass evidence on migrate.
    if raw.get("pass_evidence") is None and not st.pass_evidence:
        st.pass_evidence = {}
    if cfg is not None and not st.committed_stage:
        st.committed_stage = cfg.stage_for_tolerance(st.current_tolerance)
    st.validate()
    return st, logs


def clear_evidence_on_pbt_copy(state: WorkerMSAState) -> WorkerMSAState:
    """Recipient must not inherit donor MSA pass evidence / cycle / CIs."""
    state.cycle_id = ""
    state.pass_evidence = {}
    state.pre_probe_metrics = {}
    state.post_probe_metrics = {}
    state.decision_metrics = {}
    state.decision_reason = "pbt_copy_evidence_reset"
    state.consecutive_passes = 0
    state.pending_target_tolerance = 0.0
    state.transaction_open = False
    state.probe_start_frame = -1
    # Keep own identity; drop committed stricter-stage claim if needed —
    # caller re-establishes BUILD/HOLD after copy.
    return state


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / f".{path.name}.{os.getpid()}.{uuid.uuid4().hex[:8]}.tmp"
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True))
    os.replace(str(tmp), str(path))


def save_worker_state(path: Path, state: WorkerMSAState) -> None:
    state.validate()
    atomic_write_json(path, state.to_dict())


def load_worker_state(path: Path) -> Optional[WorkerMSAState]:
    path = Path(path)
    if not path.is_file():
        return None
    try:
        return WorkerMSAState.from_dict(json.loads(path.read_text()))
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return None


def enter_probe(
    state: WorkerMSAState,
    *,
    cycle_id: str,
    target_tol: float,
    frame: int,
    budget_frames: int,
) -> WorkerMSAState:
    state.mode = MODE_PROBE
    state.cycle_id = str(cycle_id)
    state.pending_target_tolerance = float(target_tol)
    state.probe_start_frame = int(frame)
    state.probe_budget_frames = int(budget_frames)
    state.transaction_open = True
    state.decision_reason = ""
    return state


def apply_advance(
    state: WorkerMSAState,
    *,
    new_tol: float,
    frame: int,
    anchor_checkpoint: str,
    metrics: dict[str, Any] | None = None,
    reason: str = "advance",
    cfg: Optional[MSAFusionConfig] = None,
) -> WorkerMSAState:
    state.mode = MODE_BUILD
    state.current_tolerance = float(new_tol)
    state.anchor_tolerance = float(new_tol)
    state.anchor_checkpoint = str(anchor_checkpoint)
    state.pending_target_tolerance = 0.0
    state.transaction_open = False
    state.consecutive_passes += 1
    state.consecutive_failures = 0
    state.last_decision_frame = int(frame)
    state.advance_count += 1
    state.decision_metrics = dict(metrics or {})
    state.decision_reason = reason
    state.pass_evidence = {}  # clear old evidence after commit
    state.pre_probe_metrics = {}
    state.post_probe_metrics = {}
    if cfg is not None:
        state.committed_stage = cfg.stage_for_tolerance(new_tol)
    return state


def apply_rollback_hold(
    state: WorkerMSAState,
    *,
    frame: int,
    cooldown_frames: int,
    metrics: dict[str, Any] | None = None,
    reason: str = "rollback",
) -> WorkerMSAState:
    """Restore to anchor tolerance conceptually; enter HOLD with cooldown."""
    state.mode = MODE_HOLD
    state.current_tolerance = float(state.anchor_tolerance)
    state.pending_target_tolerance = 0.0
    state.transaction_open = False
    state.consecutive_failures += 1
    state.consecutive_passes = 0
    state.last_decision_frame = int(frame)
    state.hold_until_frame = int(frame) + max(0, int(cooldown_frames))
    state.rollback_count += 1
    state.hold_count += 1
    state.decision_metrics = dict(metrics or {})
    state.decision_reason = reason
    # Keep failure evidence for diagnostics (pre/post metrics retained).
    return state


def maybe_exit_hold(state: WorkerMSAState, frame: int) -> WorkerMSAState:
    if state.mode == MODE_HOLD and int(frame) >= int(state.hold_until_frame):
        state.mode = MODE_BUILD
        state.decision_reason = "hold_cooldown_elapsed"
    return state
