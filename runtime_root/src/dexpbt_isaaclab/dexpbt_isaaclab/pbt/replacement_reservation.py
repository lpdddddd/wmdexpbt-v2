"""Population-level ordinary replacement reservation (Profile 23).

Scope is ``(workspace, pbt_iteration/tick_id, stage)``, not a per-observer
counter.  Workers coordinate via a flock-protected JSON ledger under
``{workspace}/adaptive_replacement/tick_{tick_id}.json``.

Lifecycle::

    reserved -> committed
             -> released / expired

Hard-collapse / adaptive-rescue bypasses are recorded separately and never
consume the ordinary budget.
"""

from __future__ import annotations

import fcntl
import json
import math
import os
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional

from dexpbt_isaaclab.pbt.adaptive_precision_consolidation import (
    count_distinct_lineages,
)

SCHEMA_VERSION = 1
DEFAULT_RESERVATION_TIMEOUT_S = 300.0


def _finite(x: float, default: float = 0.0) -> float:
    v = float(x)
    if math.isnan(v) or math.isinf(v):
        return float(default)
    return v


def _now_mono() -> float:
    return float(time.monotonic())


def _now_wall() -> float:
    return float(time.time())


@dataclass
class ReservationResult:
    ok: bool
    reservation_id: str = ""
    reason: str = ""
    telemetry: dict[str, float] = field(default_factory=dict)


def projected_lineages_after_copies(
    base_lineages: Mapping[int, str],
    *,
    committed: list[dict[str, Any]],
    reserved: list[dict[str, Any]],
    proposed_target: int | None = None,
    proposed_source_lineage: str | None = None,
) -> dict[int, str]:
    """Apply committed + reserved + optional proposed copy onto base lineages."""
    projected = {int(k): str(v) for k, v in base_lineages.items()}
    for item in list(committed) + list(reserved):
        try:
            t = int(item.get("target", -1))
        except Exception:
            continue
        if t < 0:
            continue
        projected[t] = str(item.get("source_lineage", "") or "")
    if proposed_target is not None and proposed_source_lineage is not None:
        projected[int(proposed_target)] = str(proposed_source_lineage)
    return projected


def lineages_incomplete(lineages: Mapping[int, str], num_policies: int) -> bool:
    """Fail-closed when any policy in ``[0, num_policies)`` lacks a lineage id."""
    for i in range(int(num_policies)):
        if not str(lineages.get(i, "") or "").strip():
            return True
    return False


class PopulationReplacementLedger:
    """Flock-serialized per-tick ordinary replacement ledger."""

    def __init__(
        self,
        workspace: Path | str,
        *,
        reservation_timeout_s: float = DEFAULT_RESERVATION_TIMEOUT_S,
    ) -> None:
        self.workspace = Path(workspace)
        self.dir = self.workspace / "adaptive_replacement"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.reservation_timeout_s = float(reservation_timeout_s)

    def _paths(self, tick_id: int) -> tuple[Path, Path]:
        tick = int(tick_id)
        return (
            self.dir / f"tick_{tick}.json",
            self.dir / f"tick_{tick}.lock",
        )

    def _empty_state(self, tick_id: int, stage: str, budget: int) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "tick_id": int(tick_id),
            "stage": str(stage),
            "budget": int(budget),
            "reservations": {},
            "committed": [],
            "bypassed": [],
            "blocked_budget": 0,
            "blocked_diversity": 0,
            "blocked_lineage_missing": 0,
        }

    def _load_unlocked(self, path: Path, tick_id: int, stage: str, budget: int) -> dict[str, Any]:
        if not path.exists():
            return self._empty_state(tick_id, stage, budget)
        try:
            data = json.loads(path.read_text())
        except Exception:
            return self._empty_state(tick_id, stage, budget)
        if not isinstance(data, dict):
            return self._empty_state(tick_id, stage, budget)
        if int(data.get("tick_id", -1)) != int(tick_id):
            return self._empty_state(tick_id, stage, budget)
        data.setdefault("stage", str(stage))
        data["budget"] = int(budget)
        data.setdefault("reservations", {})
        data.setdefault("committed", [])
        data.setdefault("bypassed", [])
        data.setdefault("blocked_budget", 0)
        data.setdefault("blocked_diversity", 0)
        data.setdefault("blocked_lineage_missing", 0)
        return data

    def _save_unlocked(self, path: Path, state: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.parent / f"{path.stem}.{os.getpid()}.{uuid.uuid4().hex[:8]}.tmp"
        tmp.write_text(json.dumps(state, indent=2, sort_keys=True))
        os.replace(str(tmp), str(path))

    def _with_lock(self, tick_id: int, fn):
        json_path, lock_path = self._paths(tick_id)
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        with open(lock_path, "a+", encoding="utf-8") as lock_f:
            fcntl.flock(lock_f.fileno(), fcntl.LOCK_EX)
            try:
                return fn(json_path)
            finally:
                fcntl.flock(lock_f.fileno(), fcntl.LOCK_UN)

    def _expire_stale(self, state: dict[str, Any]) -> None:
        now = _now_mono()
        timeout = float(self.reservation_timeout_s)
        reservations = state.get("reservations") or {}
        if not isinstance(reservations, dict):
            state["reservations"] = {}
            return
        for rid, rec in list(reservations.items()):
            if not isinstance(rec, dict):
                reservations.pop(rid, None)
                continue
            status = str(rec.get("status", ""))
            if status != "reserved":
                continue
            created = float(rec.get("created_mono", 0.0) or 0.0)
            if created > 0.0 and (now - created) > timeout:
                rec["status"] = "expired"
                rec["expired_mono"] = now

    def _active_reserved(self, state: dict[str, Any]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for rec in (state.get("reservations") or {}).values():
            if isinstance(rec, dict) and str(rec.get("status", "")) == "reserved":
                out.append(rec)
        return out

    def _counts(self, state: dict[str, Any]) -> tuple[int, int]:
        reserved_n = len(self._active_reserved(state))
        committed_n = len(list(state.get("committed") or []))
        return reserved_n, committed_n

    def telemetry_from_state(self, state: dict[str, Any]) -> dict[str, float]:
        reserved_n, committed_n = self._counts(state)
        return {
            "adaptive_consolidation/replacement_budget": float(state.get("budget", 0)),
            "adaptive_consolidation/replacement_budget_reserved": float(reserved_n),
            "adaptive_consolidation/replacement_budget_used": float(committed_n),
            "adaptive_consolidation/replacement_budget_blocked": float(
                int(state.get("blocked_budget", 0) or 0)
                + int(state.get("blocked_diversity", 0) or 0)
                + int(state.get("blocked_lineage_missing", 0) or 0)
            ),
            "adaptive_consolidation/replacement_budget_bypassed": float(
                len(list(state.get("bypassed") or []))
            ),
            "adaptive_consolidation/replacement_blocked_budget": float(
                state.get("blocked_budget", 0) or 0
            ),
            "adaptive_consolidation/replacement_blocked_diversity": float(
                state.get("blocked_diversity", 0) or 0
            ),
            "adaptive_consolidation/replacement_blocked_lineage_missing": float(
                state.get("blocked_lineage_missing", 0) or 0
            ),
        }

    def try_reserve_ordinary(
        self,
        *,
        tick_id: int,
        stage: str,
        budget: int,
        min_distinct_lineages: int,
        population_lineages: Mapping[int, str],
        num_policies: int,
        target_idx: int,
        source_idx: int,
        source_lineage: str,
        owner_policy: int,
        owner_pid: int | None = None,
    ) -> ReservationResult:
        """Atomically reserve one ordinary replacement slot for this tick."""

        def _body(json_path: Path) -> ReservationResult:
            state = self._load_unlocked(json_path, tick_id, stage, budget)
            # Tick change already creates a fresh file via tick_id in path; still
            # expire stale reservations within the same tick (crashed owners).
            self._expire_stale(state)
            tele = self.telemetry_from_state(state)

            if lineages_incomplete(population_lineages, num_policies):
                state["blocked_lineage_missing"] = int(
                    state.get("blocked_lineage_missing", 0) or 0
                ) + 1
                self._save_unlocked(json_path, state)
                tele = self.telemetry_from_state(state)
                return ReservationResult(
                    ok=False,
                    reason="lineage_missing",
                    telemetry=tele,
                )

            reserved_n, committed_n = self._counts(state)
            if reserved_n + committed_n >= int(budget):
                state["blocked_budget"] = int(state.get("blocked_budget", 0) or 0) + 1
                self._save_unlocked(json_path, state)
                tele = self.telemetry_from_state(state)
                return ReservationResult(
                    ok=False,
                    reason="replacement_budget_exhausted",
                    telemetry=tele,
                )

            projected = projected_lineages_after_copies(
                population_lineages,
                committed=list(state.get("committed") or []),
                reserved=self._active_reserved(state),
                proposed_target=int(target_idx),
                proposed_source_lineage=str(source_lineage),
            )
            if lineages_incomplete(projected, num_policies):
                state["blocked_lineage_missing"] = int(
                    state.get("blocked_lineage_missing", 0) or 0
                ) + 1
                self._save_unlocked(json_path, state)
                tele = self.telemetry_from_state(state)
                return ReservationResult(
                    ok=False,
                    reason="projected_lineage_missing",
                    telemetry=tele,
                )
            distinct = count_distinct_lineages(list(projected.values()))
            if distinct < int(min_distinct_lineages):
                state["blocked_diversity"] = int(state.get("blocked_diversity", 0) or 0) + 1
                self._save_unlocked(json_path, state)
                tele = self.telemetry_from_state(state)
                tele["adaptive_consolidation/distinct_lineage_count"] = float(distinct)
                tele["adaptive_consolidation/min_distinct_lineages"] = float(
                    min_distinct_lineages
                )
                return ReservationResult(
                    ok=False,
                    reason="diversity_blocked",
                    telemetry=tele,
                )

            rid = uuid.uuid4().hex
            rec = {
                "reservation_id": rid,
                "owner_policy": int(owner_policy),
                "owner_pid": int(owner_pid if owner_pid is not None else os.getpid()),
                "target": int(target_idx),
                "source": int(source_idx),
                "source_lineage": str(source_lineage),
                "status": "reserved",
                "created_mono": _now_mono(),
                "created_wall": _now_wall(),
            }
            state.setdefault("reservations", {})[rid] = rec
            state["stage"] = str(stage)
            self._save_unlocked(json_path, state)
            tele = self.telemetry_from_state(state)
            tele["adaptive_consolidation/distinct_lineage_count"] = float(distinct)
            tele["adaptive_consolidation/min_distinct_lineages"] = float(
                min_distinct_lineages
            )
            return ReservationResult(
                ok=True,
                reservation_id=rid,
                reason="reserved",
                telemetry=tele,
            )

        return self._with_lock(int(tick_id), _body)

    def _read_existing(self, json_path: Path, tick_id: int) -> dict[str, Any] | None:
        if not json_path.exists():
            return None
        try:
            data = json.loads(json_path.read_text())
        except Exception:
            return None
        if not isinstance(data, dict):
            return None
        if int(data.get("tick_id", -1)) != int(tick_id):
            return None
        data.setdefault("reservations", {})
        data.setdefault("committed", [])
        data.setdefault("bypassed", [])
        data.setdefault("blocked_budget", 0)
        data.setdefault("blocked_diversity", 0)
        data.setdefault("blocked_lineage_missing", 0)
        data.setdefault("budget", 0)
        data.setdefault("stage", "")
        return data

    def commit(self, tick_id: int, reservation_id: str) -> bool:
        rid = str(reservation_id or "")
        if not rid:
            return False

        def _body(json_path: Path) -> bool:
            state = self._read_existing(json_path, tick_id)
            if state is None:
                return False
            self._expire_stale(state)
            rec = (state.get("reservations") or {}).get(rid)
            if not isinstance(rec, dict):
                return False
            if str(rec.get("status", "")) != "reserved":
                return False
            rec["status"] = "committed"
            rec["committed_mono"] = _now_mono()
            state.setdefault("committed", []).append(
                {
                    "reservation_id": rid,
                    "target": int(rec.get("target", -1)),
                    "source": int(rec.get("source", -1)),
                    "source_lineage": str(rec.get("source_lineage", "")),
                }
            )
            self._save_unlocked(json_path, state)
            return True

        return bool(self._with_lock(int(tick_id), _body))

    def release(self, tick_id: int, reservation_id: str) -> bool:
        rid = str(reservation_id or "")
        if not rid:
            return False

        def _body(json_path: Path) -> bool:
            state = self._read_existing(json_path, tick_id)
            if state is None:
                return False
            self._expire_stale(state)
            rec = (state.get("reservations") or {}).get(rid)
            if not isinstance(rec, dict):
                return False
            if str(rec.get("status", "")) != "reserved":
                return False
            rec["status"] = "released"
            rec["released_mono"] = _now_mono()
            self._save_unlocked(json_path, state)
            return True

        return bool(self._with_lock(int(tick_id), _body))

    def record_bypass(
        self,
        *,
        tick_id: int,
        stage: str,
        budget: int,
        target_idx: int,
        reason: str,
    ) -> dict[str, float]:
        def _body(json_path: Path) -> dict[str, float]:
            state = self._load_unlocked(json_path, tick_id, stage, budget)
            self._expire_stale(state)
            state.setdefault("bypassed", []).append(
                {
                    "target": int(target_idx),
                    "reason": str(reason),
                    "wall": _now_wall(),
                }
            )
            self._save_unlocked(json_path, state)
            return self.telemetry_from_state(state)

        return self._with_lock(int(tick_id), _body)

    def snapshot_telemetry(
        self, tick_id: int, *, stage: str = "", budget: int = 0
    ) -> dict[str, float]:
        def _body(json_path: Path) -> dict[str, float]:
            state = self._load_unlocked(json_path, tick_id, stage, budget)
            self._expire_stale(state)
            return self.telemetry_from_state(state)

        return self._with_lock(int(tick_id), _body)
