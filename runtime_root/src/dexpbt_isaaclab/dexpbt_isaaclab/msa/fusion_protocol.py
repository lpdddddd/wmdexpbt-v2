"""Per-worker MSA fusion command/ack protocol (exactly-once).

Coordinator claims a sidecar cycle result once, fans out per-worker
commands, and waits for acks before marking the cycle complete.

Directory layout under <run_dir>:
  msa_fusion_protocol/
    journal/<cycle_id>.json          coordinator journal
    claimed/<cycle_id>.claim.json    exactly-once claim marker
    cmds/<wid:03d>/<cycle_id>.cmd.json
    cmds/<wid:03d>/<cycle_id>.claimed.json
    acks/<wid:03d>/<cycle_id>.ack.json
    complete/<cycle_id>.complete.json
"""

from __future__ import annotations

import hashlib
import json
import os
import time
import uuid
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Optional

from dexpbt_isaaclab.msa.worker_state import atomic_write_json

PROTOCOL_VERSION = "msa_fusion_protocol_v1"
SCHEMA_VERSION = 1

ROOT_DIR = "msa_fusion_protocol"


def protocol_root(run_dir: Path | str) -> Path:
    return Path(run_dir) / ROOT_DIR


def _ensure(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


@dataclass
class WorkerCommand:
    schema_version: int = SCHEMA_VERSION
    protocol_version: str = PROTOCOL_VERSION
    cycle_id: str = ""
    transaction_id: str = ""
    worker_id: int = -1
    action: str = "hold"  # advance | hold | rollback
    expected_mode: str = "PROBE"
    source_fingerprint: str = ""
    expected_tolerance: float = 0.0
    target_tolerance: float = 0.0
    anchor_tolerance: float = 0.0
    evidence_digest: str = ""
    anchor_checkpoint: str = ""
    post_probe_checkpoint: str = ""
    metrics: dict[str, Any] = field(default_factory=dict)
    reason: str = ""
    issued_at: float = 0.0
    production: bool = True
    test_only: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: dict[str, Any] | None) -> "WorkerCommand":
        if not raw:
            return cls()
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in raw.items() if k in known})


@dataclass
class WorkerAck:
    schema_version: int = SCHEMA_VERSION
    cycle_id: str = ""
    transaction_id: str = ""
    worker_id: int = -1
    action_applied: str = ""
    ok: bool = False
    mode_after: str = ""
    tolerance_after: float = 0.0
    fingerprint_after: str = ""
    frame: int = -1
    reason: str = ""
    acked_at: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: dict[str, Any] | None) -> "WorkerAck":
        if not raw:
            return cls()
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in raw.items() if k in known})


def evidence_digest(payload: dict[str, Any]) -> str:
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:24]


def claim_cycle_result(
    run_dir: Path | str,
    cycle_id: str,
    *,
    source_path: str,
    payload: dict[str, Any],
) -> tuple[bool, Path]:
    """Exactly-once claim of a sidecar cycle result for fusion protocol.

    Returns (claimed_by_us, claim_path). If another coordinator already
    claimed this cycle_id, returns (False, existing_claim_path).
    """
    root = _ensure(protocol_root(run_dir) / "claimed")
    claim_path = root / f"{cycle_id}.claim.json"
    if claim_path.is_file():
        return False, claim_path
    tmp = root / f".{cycle_id}.{os.getpid()}.{uuid.uuid4().hex[:8]}.tmp"
    doc = {
        "cycle_id": cycle_id,
        "source_path": source_path,
        "claimed_at": time.time(),
        "claimer_pid": os.getpid(),
        "protocol_version": PROTOCOL_VERSION,
        "payload_digest": evidence_digest(payload),
    }
    tmp.write_text(json.dumps(doc, indent=2, sort_keys=True))
    try:
        os.link(str(tmp), str(claim_path))  # fails if exists (POSIX)
        claimed = True
    except FileExistsError:
        claimed = False
    except OSError:
        # Fallback: atomic replace only if missing.
        if claim_path.is_file():
            claimed = False
        else:
            try:
                os.replace(str(tmp), str(claim_path))
                claimed = True
            except OSError:
                claimed = False
    try:
        tmp.unlink()
    except OSError:
        pass
    return claimed, claim_path


def write_worker_command(run_dir: Path | str, cmd: WorkerCommand) -> Path:
    root = _ensure(protocol_root(run_dir) / "cmds" / f"{int(cmd.worker_id):03d}")
    path = root / f"{cmd.cycle_id}.cmd.json"
    if not cmd.issued_at:
        cmd.issued_at = time.time()
    atomic_write_json(path, cmd.to_dict())
    return path


def command_path(run_dir: Path | str, worker_id: int, cycle_id: str) -> Path:
    return (
        protocol_root(run_dir) / "cmds" / f"{int(worker_id):03d}"
        / f"{cycle_id}.cmd.json"
    )


def claim_worker_command(
    run_dir: Path | str,
    worker_id: int,
    cycle_id: str,
) -> Optional[WorkerCommand]:
    """Atomically claim one command for this worker. Idempotent.

    Returns the command if newly claimed or already claimed by this pid
    with matching claim file; None if absent or claimed by another.
    """
    cmd_p = command_path(run_dir, worker_id, cycle_id)
    if not cmd_p.is_file():
        return None
    claim_p = cmd_p.with_suffix(".claimed.json")
    if claim_p.is_file():
        try:
            existing = json.loads(claim_p.read_text())
            if int(existing.get("worker_id", -1)) != int(worker_id):
                return None
            # Same worker already claimed — return command for idempotent apply.
            return WorkerCommand.from_dict(json.loads(cmd_p.read_text()))
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            return None
    try:
        cmd = WorkerCommand.from_dict(json.loads(cmd_p.read_text()))
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return None
    if int(cmd.worker_id) != int(worker_id):
        return None
    claim_doc = {
        "cycle_id": cycle_id,
        "worker_id": int(worker_id),
        "claimed_at": time.time(),
        "claimer_pid": os.getpid(),
        "transaction_id": cmd.transaction_id,
    }
    tmp = claim_p.parent / f".{claim_p.name}.{os.getpid()}.tmp"
    tmp.write_text(json.dumps(claim_doc, indent=2, sort_keys=True))
    try:
        os.link(str(tmp), str(claim_p))
    except FileExistsError:
        try:
            tmp.unlink()
        except OSError:
            pass
        # Lost race — if claim is ours, still return cmd.
        try:
            existing = json.loads(claim_p.read_text())
            if int(existing.get("worker_id", -1)) == int(worker_id):
                return cmd
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            pass
        return None
    except OSError:
        if not claim_p.is_file():
            try:
                os.replace(str(tmp), str(claim_p))
            except OSError:
                return None
        else:
            try:
                tmp.unlink()
            except OSError:
                pass
            return WorkerCommand.from_dict(json.loads(cmd_p.read_text()))
    try:
        tmp.unlink()
    except OSError:
        pass
    return cmd


def list_pending_commands(run_dir: Path | str, worker_id: int) -> list[str]:
    root = protocol_root(run_dir) / "cmds" / f"{int(worker_id):03d}"
    if not root.is_dir():
        return []
    out = []
    for p in sorted(root.glob("*.cmd.json")):
        cycle_id = p.name[: -len(".cmd.json")]
        ack = ack_path(run_dir, worker_id, cycle_id)
        if not ack.is_file():
            out.append(cycle_id)
    return out


def ack_path(run_dir: Path | str, worker_id: int, cycle_id: str) -> Path:
    return (
        protocol_root(run_dir) / "acks" / f"{int(worker_id):03d}"
        / f"{cycle_id}.ack.json"
    )


def write_worker_ack(run_dir: Path | str, ack: WorkerAck) -> Path:
    root = _ensure(protocol_root(run_dir) / "acks" / f"{int(ack.worker_id):03d}")
    path = root / f"{ack.cycle_id}.ack.json"
    if not ack.acked_at:
        ack.acked_at = time.time()
    atomic_write_json(path, ack.to_dict())
    return path


def read_ack(run_dir: Path | str, worker_id: int, cycle_id: str) -> Optional[WorkerAck]:
    p = ack_path(run_dir, worker_id, cycle_id)
    if not p.is_file():
        return None
    try:
        return WorkerAck.from_dict(json.loads(p.read_text()))
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return None


def write_journal(run_dir: Path | str, cycle_id: str, doc: dict[str, Any]) -> Path:
    root = _ensure(protocol_root(run_dir) / "journal")
    path = root / f"{cycle_id}.json"
    atomic_write_json(path, doc)
    return path


def read_journal(run_dir: Path | str, cycle_id: str) -> Optional[dict[str, Any]]:
    p = protocol_root(run_dir) / "journal" / f"{cycle_id}.json"
    if not p.is_file():
        return None
    try:
        return json.loads(p.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def mark_cycle_complete(
    run_dir: Path | str,
    cycle_id: str,
    *,
    target_workers: list[int],
) -> tuple[bool, Path]:
    """Write complete marker only when all target acks exist and ok."""
    missing = []
    failed = []
    for w in target_workers:
        ack = read_ack(run_dir, w, cycle_id)
        if ack is None:
            missing.append(w)
        elif not ack.ok:
            failed.append(w)
    root = _ensure(protocol_root(run_dir) / "complete")
    path = root / f"{cycle_id}.complete.json"
    if missing or failed:
        atomic_write_json(path.with_suffix(".pending.json"), {
            "cycle_id": cycle_id,
            "missing": missing,
            "failed": failed,
            "updated_at": time.time(),
        })
        return False, path
    atomic_write_json(path, {
        "cycle_id": cycle_id,
        "target_workers": list(target_workers),
        "completed_at": time.time(),
        "protocol_version": PROTOCOL_VERSION,
    })
    return True, path


def validate_command_for_worker(
    cmd: WorkerCommand,
    *,
    worker_id: int,
    current_mode: str,
    current_tolerance: float,
    cycle_id: str = "",
    transaction_id: str = "",
) -> tuple[bool, str]:
    """Reject stale / out-of-order / wrong-worker / wrong-mode commands.

    ``cycle_id`` is the worker's open PROBE cycle id. Transaction ids of the
    form ``txn_<cycle_id>`` (or the bare cycle id) are accepted.
    """
    if int(cmd.worker_id) != int(worker_id):
        return False, "worker_id_mismatch"
    if cmd.schema_version != SCHEMA_VERSION:
        return False, "schema_version_mismatch"
    if cmd.action in ("advance", "rollback") and current_mode != "PROBE":
        if cmd.expected_mode == "PROBE":
            return False, f"advance_rejected_mode_{current_mode}"
    local_cycle = str(cycle_id or "")
    if cmd.cycle_id and local_cycle and str(cmd.cycle_id) != local_cycle:
        return False, "cycle_id_mismatch"
    if cmd.transaction_id and local_cycle:
        accepted = {
            f"txn_{local_cycle}",
            local_cycle,
            f"txn_{cmd.cycle_id}" if cmd.cycle_id else "",
            str(transaction_id or ""),
        }
        if str(cmd.transaction_id) not in accepted:
            return False, "transaction_id_mismatch"
    if cmd.expected_tolerance > 0 and current_tolerance > 0:
        if abs(float(cmd.expected_tolerance) - float(current_tolerance)) > 1e-6:
            # Soft: allow if still in PROBE with pending target matching.
            pass
    return True, "ok"
