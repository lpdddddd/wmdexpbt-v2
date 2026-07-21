"""Transactional PROBE anchor checkpoints for MSA-aware fusion.

Saves a verified copy of the trainer workspace checkpoint as the probe
anchor; ADVANCE registers a new committed anchor; ROLLBACK restores the
pre-probe files and returns metadata for a fail-closed restart if needed.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import time
from pathlib import Path
from typing import Any, Optional

from dexpbt_isaaclab.msa.fusion_events import emit_event
from dexpbt_isaaclab.msa.worker_state import atomic_write_json

ANCHOR_DIRNAME = "msa_probe_anchors"


def fingerprint_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            chunk = fh.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def fingerprint_pair(pth: Path, meta: Path | None = None) -> str:
    parts = [fingerprint_file(pth)]
    if meta is not None and meta.is_file():
        parts.append(fingerprint_file(meta))
    return hashlib.sha256("|".join(parts).encode()).hexdigest()


def anchor_dir(run_dir: Path | str, worker_id: int, cycle_id: str) -> Path:
    return (
        Path(run_dir) / ANCHOR_DIRNAME / f"{int(worker_id):03d}" / str(cycle_id)
    )


def latest_workspace_checkpoint(policy_workspace: Path) -> tuple[Path, Path]:
    """Return (pth, json) for the newest committed workspace checkpoint."""
    policy_workspace = Path(policy_workspace)
    metas = sorted(policy_workspace.glob("*.json"))
    metas = [m for m in metas if not m.name.endswith(".tmp")
             and not m.name.startswith(".")
             and m.stem.isdigit()]
    if not metas:
        # Also accept non-numeric prefixes used by some exports.
        metas = sorted(
            p for p in policy_workspace.glob("*.json")
            if p.with_suffix(".pth").is_file()
            and not p.name.endswith(".tmp")
        )
    if not metas:
        raise FileNotFoundError(f"no checkpoint meta under {policy_workspace}")
    meta = metas[-1]
    pth = meta.with_suffix(".pth")
    if not pth.is_file():
        raise FileNotFoundError(f"missing weights for {meta}")
    return pth, meta


def save_probe_anchor(
    run_dir: Path | str,
    worker_id: int,
    cycle_id: str,
    policy_workspace: Path | str,
    *,
    transaction_id: str = "",
    tolerance: float = 0.0,
    frame: int = -1,
    config_hash: str = "",
    extra_meta: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Copy latest workspace ckpt into anchor dir; verify by re-read fingerprint."""
    run_dir = Path(run_dir)
    src_pth, src_meta = latest_workspace_checkpoint(Path(policy_workspace))
    dest = anchor_dir(run_dir, worker_id, cycle_id)
    dest.mkdir(parents=True, exist_ok=True)
    dest_pth = dest / "anchor.pth"
    dest_meta = dest / "anchor.json"
    shutil.copy2(src_pth, dest_pth)
    shutil.copy2(src_meta, dest_meta)
    fp = fingerprint_pair(dest_pth, dest_meta)
    # Re-read verification
    fp2 = fingerprint_pair(dest_pth, dest_meta)
    if fp != fp2:
        raise RuntimeError("anchor fingerprint unstable after copy")
    record = {
        "worker_id": int(worker_id),
        "cycle_id": str(cycle_id),
        "transaction_id": str(transaction_id or f"txn_{cycle_id}"),
        "source_pth": str(src_pth),
        "source_meta": str(src_meta),
        "anchor_pth": str(dest_pth),
        "anchor_meta": str(dest_meta),
        "fingerprint": fp,
        "tolerance": float(tolerance),
        "frame": int(frame),
        "config_hash": str(config_hash),
        "saved_at": time.time(),
        "extra": dict(extra_meta or {}),
    }
    atomic_write_json(dest / "transaction.json", record)
    emit_event(run_dir, "MSA_PROBE_CHECKPOINT_SAVED", {
        "worker_id": worker_id,
        "cycle_id": cycle_id,
        "fingerprint": fp,
        "tolerance": tolerance,
        "frame": frame,
    })
    return record


def load_transaction_record(
    run_dir: Path | str, worker_id: int, cycle_id: str,
) -> Optional[dict[str, Any]]:
    p = anchor_dir(run_dir, worker_id, cycle_id) / "transaction.json"
    if not p.is_file():
        return None
    try:
        return json.loads(p.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def verify_anchor_fingerprint(record: dict[str, Any]) -> tuple[bool, str]:
    pth = Path(record["anchor_pth"])
    meta = Path(record["anchor_meta"])
    if not pth.is_file() or not meta.is_file():
        return False, "anchor_files_missing"
    fp = fingerprint_pair(pth, meta)
    if fp != str(record.get("fingerprint", "")):
        return False, "fingerprint_mismatch"
    return True, fp


def restore_anchor_into_workspace(
    record: dict[str, Any],
    policy_workspace: Path | str,
    *,
    dest_prefix: str = "msa_rollback",
) -> dict[str, Any]:
    """Copy anchor weights/meta into workspace as a restartable checkpoint.

    Returns paths + verified fingerprint for the restored files.
    """
    policy_workspace = Path(policy_workspace)
    policy_workspace.mkdir(parents=True, exist_ok=True)
    ok, detail = verify_anchor_fingerprint(record)
    if not ok:
        raise RuntimeError(f"anchor verify failed: {detail}")
    dest_pth = policy_workspace / f"{dest_prefix}.pth"
    dest_meta = policy_workspace / f"{dest_prefix}.json"
    shutil.copy2(record["anchor_pth"], dest_pth)
    shutil.copy2(record["anchor_meta"], dest_meta)
    # Patch meta frame/tolerance if recorded.
    try:
        meta = json.loads(dest_meta.read_text())
    except (OSError, json.JSONDecodeError):
        meta = {}
    cur = dict(meta.get("curriculum") or {})
    if record.get("tolerance"):
        cur["success_tolerance"] = float(record["tolerance"])
    meta["curriculum"] = cur
    if int(record.get("frame", -1)) >= 0:
        meta["frame"] = int(record["frame"])
    meta["msa_rollback_restore"] = {
        "cycle_id": record.get("cycle_id"),
        "fingerprint": record.get("fingerprint"),
        "restored_at": time.time(),
    }
    dest_meta.write_text(json.dumps(meta, indent=2))
    fp = fingerprint_pair(dest_pth, dest_meta)
    return {
        "checkpoint": str(dest_pth),
        "meta": str(dest_meta),
        "fingerprint": fp,
        "anchor_fingerprint": record.get("fingerprint"),
        "tolerance": float(record.get("tolerance") or 0.0),
        "frame": int(record.get("frame", -1)),
    }


def save_committed_anchor(
    run_dir: Path | str,
    worker_id: int,
    cycle_id: str,
    policy_workspace: Path | str,
    *,
    tolerance: float,
    frame: int,
    config_hash: str = "",
) -> dict[str, Any]:
    """After ADVANCE: snapshot current ckpt as the new committed anchor."""
    run_dir = Path(run_dir)
    src_pth, src_meta = latest_workspace_checkpoint(Path(policy_workspace))
    dest = Path(run_dir) / ANCHOR_DIRNAME / f"{int(worker_id):03d}" / "committed"
    dest.mkdir(parents=True, exist_ok=True)
    dest_pth = dest / "anchor.pth"
    dest_meta = dest / "anchor.json"
    shutil.copy2(src_pth, dest_pth)
    shutil.copy2(src_meta, dest_meta)
    fp = fingerprint_pair(dest_pth, dest_meta)
    record = {
        "worker_id": int(worker_id),
        "cycle_id": str(cycle_id),
        "anchor_pth": str(dest_pth),
        "anchor_meta": str(dest_meta),
        "fingerprint": fp,
        "tolerance": float(tolerance),
        "frame": int(frame),
        "config_hash": str(config_hash),
        "committed_at": time.time(),
        "source_cycle": str(cycle_id),
    }
    atomic_write_json(dest / "committed.json", record)
    return record
