"""Structured MSA fusion event log helpers."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Optional

from dexpbt_isaaclab.msa.worker_state import atomic_write_json

EVENT_NAMES = (
    "MSA_TRIGGER",
    "MSA_PREFILTER_PASS",
    "MSA_PREFILTER_REJECT",
    "MSA_PROBE_START",
    "MSA_EVAL_START",
    "MSA_EVAL_COMPLETE",
    "MSA_ADVANCE_COMMIT",
    "MSA_HOLD",
    "MSA_ROLLBACK",
    "MSA_STATE_RESTORED",
    "MSA_FAIL_CLOSED",
)


def emit_event(
    run_dir: Path | str | None,
    event: str,
    payload: dict[str, Any] | None = None,
    *,
    also_print: bool = True,
) -> dict[str, Any]:
    if event not in EVENT_NAMES:
        raise ValueError(f"unknown MSA event: {event}")
    rec = {
        "event": event,
        "timestamp": time.time(),
        **(payload or {}),
    }
    line = json.dumps(rec, sort_keys=True)
    if also_print:
        print(f"[msa-fusion] {event} {line}", flush=True)
    if run_dir is not None:
        path = Path(run_dir) / "msa_fusion_events.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")
    return rec


def write_cycle_archive(run_dir: Path | str, name: str, doc: dict[str, Any]) -> Path:
    out = Path(run_dir) / "msa_fusion_archive" / f"{name}.json"
    atomic_write_json(out, doc)
    return out
