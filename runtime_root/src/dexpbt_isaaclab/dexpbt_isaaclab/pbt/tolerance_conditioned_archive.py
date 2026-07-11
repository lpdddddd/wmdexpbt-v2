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


def archive_retention(state: dict[str, Any]) -> float | None:
    """Return saved retention, or None when metadata is missing."""
    for key in ("same_tol_retention_at_save", "source_retention_at_save"):
        if key in state and state.get(key) is not None:
            try:
                return float(state[key])
            except (TypeError, ValueError):
                return None
    return None


def select_compatible_archive(
    archive_dir: Path,
    *,
    target_tolerance: float,
    max_gap: float,
    min_mean: float = 0.0,
    min_retention: float = 0.0,
    require_retention_metadata: bool = False,
    reject_log: list[dict[str, Any]] | None = None,
) -> dict[str, Any] | None:
    """Select highest-nsucc compatible old/new archive metadata.

    When ``require_retention_metadata`` is True (strict late-degradation
    rescue), archives missing retention fields are rejected fail-closed.
    Ordinary emergency recovery keeps the legacy default of treating missing
    retention as 1.0 when ``require_retention_metadata`` is False.
    """
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
            if reject_log is not None:
                reject_log.append({
                    "path": str(path),
                    "reason": "tolerance_gap" if tol is not None else "tolerance_missing",
                })
            continue
        if float(state.get("nsucc_mean", 0.0) or 0.0) < float(min_mean):
            if reject_log is not None:
                reject_log.append({"path": str(path), "reason": "mean_below_threshold"})
            continue
        retention = archive_retention(state)
        if retention is None:
            if require_retention_metadata and float(min_retention) > 0.0:
                if reject_log is not None:
                    reject_log.append({
                        "path": str(path),
                        "reason": "missing_retention_metadata",
                    })
                continue
            retention = 1.0
        if float(retention) < float(min_retention):
            if reject_log is not None:
                reject_log.append({"path": str(path), "reason": "retention_below_threshold"})
            continue
        state["_archive_metadata_path"] = str(path)
        state["_archive_retention_used"] = float(retention)
        candidates.append(state)
    if not candidates:
        return None
    return max(candidates, key=lambda s: (
        float(s.get("nsucc_mean", 0.0) or 0.0),
        float(s.get("nsucc_max", 0.0) or 0.0),
        int(s.get("frame", 0) or 0),
    ))
