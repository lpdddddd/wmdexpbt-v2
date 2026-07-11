"""Compatibility filter for existing append-only PBT archive metadata.

This intentionally stores no checkpoint itself: callers keep using the normal
``best_*`` JSON + PTH format and select compatible historical entries by the
tolerance recorded in that metadata.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def archive_tolerance(state: dict[str, Any]) -> float | None:
    value = (state.get("curriculum") or {}).get("success_tolerance")
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def select_compatible_archive(
    archive_dir: Path, *, target_tolerance: float, max_gap: float,
    min_mean: float = 0.0, min_retention: float = 0.0,
) -> dict[str, Any] | None:
    """Select highest-nsucc compatible old/new archive metadata."""
    candidates: list[dict[str, Any]] = []
    for path in archive_dir.glob("*.json"):
        if path.name == "_index.json":
            continue
        try:
            state = json.loads(path.read_text())
        except Exception:
            continue
        tol = archive_tolerance(state)
        if tol is None or abs(tol - float(target_tolerance)) > float(max_gap):
            continue
        if float(state.get("nsucc_mean", 0.0) or 0.0) < float(min_mean):
            continue
        retention = state.get("same_tol_retention_at_save", state.get("source_retention_at_save", 1.0))
        if retention is not None and float(retention) < float(min_retention):
            continue
        state["_archive_metadata_path"] = str(path)
        candidates.append(state)
    if not candidates:
        return None
    return max(candidates, key=lambda s: (
        float(s.get("nsucc_mean", 0.0) or 0.0),
        float(s.get("nsucc_max", 0.0) or 0.0),
        int(s.get("frame", 0) or 0),
    ))
