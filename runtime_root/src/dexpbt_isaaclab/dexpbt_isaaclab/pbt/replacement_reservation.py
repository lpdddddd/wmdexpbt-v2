"""Population-level ordinary replacement reservation (Profile 23).

Ledger identity is ``(workspace, tick_id, stage)`` with stage-scoped paths::

    adaptive_replacement/tick_{tick}_{stage}.json
    adaptive_replacement/tick_{tick}_{stage}.lock

Lifecycle::

    reserved -> committed | released | expired
    expired + actual restart success -> reconciled committed

Hard-collapse / adaptive-rescue / late-rescue bypasses are recorded separately
and never consume the ordinary budget.
"""

from __future__ import annotations

import fcntl
import json
import math
import os
import threading
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Mapping, Optional

from dexpbt_isaaclab.pbt.adaptive_precision_consolidation import (
    STAGE_CONSOLIDATION,
    STAGE_EXPLORATION,
    STAGE_SELECTION,
    count_distinct_lineages,
)

SCHEMA_VERSION = 2
DEFAULT_RESERVATION_TIMEOUT_S = 300.0
DEFAULT_HEARTBEAT_INTERVAL_S = 30.0

ALLOWED_STAGES: frozenset[str] = frozenset(
    {STAGE_EXPLORATION, STAGE_CONSOLIDATION, STAGE_SELECTION}
)


def normalize_stage(stage: str) -> str:
    """Whitelist stage names; raise on anything else."""
    s = str(stage or "").strip().lower()
    if s not in ALLOWED_STAGES:
        raise ValueError(
            f"invalid replacement ledger stage {stage!r}; "
            f"allowed={sorted(ALLOWED_STAGES)}"
        )
    return s


def _now_wall() -> float:
    return float(time.time())


@dataclass
class ReservationHandle:
    tick_id: int
    stage: str
    reservation_id: str
    owner_policy: int
    owner_pid: int
    owner_token: str
    target: int = -1
    source: int = -1
    source_lineage: str = ""
    lease_timeout_s: float = DEFAULT_RESERVATION_TIMEOUT_S


@dataclass
class ReservationResult:
    ok: bool
    reason: str = ""
    handle: Optional[ReservationHandle] = None
    telemetry: dict[str, float] = field(default_factory=dict)


@dataclass
class FinalizeResult:
    ok: bool
    reason: str = ""
    status: str = ""  # committed | released | reconciled | failed | already_*
    telemetry: dict[str, float] = field(default_factory=dict)


def projected_lineages_after_copies(
    base_lineages: Mapping[int, str],
    *,
    committed: list[dict[str, Any]],
    reserved: list[dict[str, Any]],
    proposed_target: int | None = None,
    proposed_source_lineage: str | None = None,
) -> dict[int, str]:
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
    for i in range(int(num_policies)):
        if not str(lineages.get(i, "") or "").strip():
            return True
    return False


class PopulationReplacementLedger:
    """Flock-serialized per-(tick, stage) ordinary replacement ledger."""

    def __init__(
        self,
        workspace: Path | str,
        *,
        reservation_timeout_s: float = DEFAULT_RESERVATION_TIMEOUT_S,
        heartbeat_interval_s: float = DEFAULT_HEARTBEAT_INTERVAL_S,
    ) -> None:
        self.workspace = Path(workspace)
        self.dir = self.workspace / "adaptive_replacement"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.reservation_timeout_s = float(reservation_timeout_s)
        self.heartbeat_interval_s = float(heartbeat_interval_s)
        self._last_wall_seen: dict[tuple[int, str], float] = {}

    def _paths(self, tick_id: int, stage: str) -> tuple[Path, Path]:
        st = normalize_stage(stage)
        tick = int(tick_id)
        return (
            self.dir / f"tick_{tick}_{st}.json",
            self.dir / f"tick_{tick}_{st}.lock",
        )

    def _empty_state(self, tick_id: int, stage: str, budget: int) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "tick_id": int(tick_id),
            "stage": normalize_stage(stage),
            "budget": int(budget),
            "reservations": {},
            "committed": [],
            "bypassed": [],
            "blocked_budget": 0,
            "blocked_diversity": 0,
            "blocked_lineage_missing": 0,
            "commit_failed": 0,
            "reconciled": 0,
        }

    def _validate_immutable(
        self, data: dict[str, Any], tick_id: int, stage: str, budget: int
    ) -> Optional[str]:
        try:
            sv = int(data.get("schema_version", -1))
        except Exception:
            return "ledger_schema_mismatch"
        if sv != SCHEMA_VERSION:
            return "ledger_schema_mismatch"
        try:
            tid = int(data.get("tick_id", -1))
        except Exception:
            return "ledger_schema_mismatch"
        if tid != int(tick_id):
            return "ledger_schema_mismatch"
        try:
            st = normalize_stage(str(data.get("stage", "")))
        except ValueError:
            return "ledger_stage_mismatch"
        if st != normalize_stage(stage):
            return "ledger_stage_mismatch"
        try:
            bud = int(data.get("budget", -1))
        except Exception:
            return "ledger_budget_mismatch"
        if bud != int(budget):
            return "ledger_budget_mismatch"
        return None

    def _load_or_create(
        self, path: Path, tick_id: int, stage: str, budget: int
    ) -> tuple[Optional[dict[str, Any]], str]:
        """Return ``(state, reason)``. reason empty on success."""
        st = normalize_stage(stage)
        if not path.exists():
            return self._empty_state(tick_id, st, budget), ""
        try:
            raw = path.read_text()
            data = json.loads(raw)
        except Exception:
            return None, "ledger_corrupt"
        if not isinstance(data, dict):
            return None, "ledger_corrupt"
        mismatch = self._validate_immutable(data, tick_id, st, budget)
        if mismatch:
            return None, mismatch
        data.setdefault("reservations", {})
        data.setdefault("committed", [])
        data.setdefault("bypassed", [])
        for k in (
            "blocked_budget",
            "blocked_diversity",
            "blocked_lineage_missing",
            "commit_failed",
            "reconciled",
        ):
            data.setdefault(k, 0)
        if not isinstance(data["reservations"], dict):
            return None, "ledger_corrupt"
        if not isinstance(data["committed"], list) or not isinstance(
            data["bypassed"], list
        ):
            return None, "ledger_corrupt"
        return data, ""

    def _save_unlocked(self, path: Path, state: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.parent / f"{path.stem}.{os.getpid()}.{uuid.uuid4().hex[:8]}.tmp"
        tmp.write_text(json.dumps(state, indent=2, sort_keys=True))
        os.replace(str(tmp), str(path))

    def _with_lock(self, tick_id: int, stage: str, fn):
        json_path, lock_path = self._paths(tick_id, stage)
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        with open(lock_path, "a+", encoding="utf-8") as lock_f:
            fcntl.flock(lock_f.fileno(), fcntl.LOCK_EX)
            try:
                return fn(json_path)
            finally:
                fcntl.flock(lock_f.fileno(), fcntl.LOCK_UN)

    def _check_wall_clock(self, tick_id: int, stage: str, now: float) -> Optional[str]:
        key = (int(tick_id), normalize_stage(stage))
        prev = self._last_wall_seen.get(key)
        if prev is not None and now + 1e-6 < prev:
            return "wall_clock_went_backwards"
        self._last_wall_seen[key] = float(now)
        return None

    def _expire_stale(self, state: dict[str, Any], *, now_wall: float) -> Optional[str]:
        key_err = self._check_wall_clock(
            int(state["tick_id"]), str(state["stage"]), now_wall
        )
        if key_err:
            return key_err
        reservations = state.get("reservations") or {}
        for rid, rec in list(reservations.items()):
            if not isinstance(rec, dict):
                reservations.pop(rid, None)
                continue
            if str(rec.get("status", "")) != "reserved":
                continue
            last_hb = float(
                rec.get("last_heartbeat_wall", rec.get("created_wall", 0.0)) or 0.0
            )
            lease = float(
                rec.get("lease_timeout_s", self.reservation_timeout_s) or self.reservation_timeout_s
            )
            if last_hb <= 0.0:
                rec["status"] = "expired"
                rec["expired_wall"] = float(now_wall)
                continue
            if now_wall + 1e-9 < last_hb:
                return "wall_clock_went_backwards"
            if (now_wall - last_hb) > lease:
                rec["status"] = "expired"
                rec["expired_wall"] = float(now_wall)
        return None

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
            "adaptive_consolidation/replacement_commit_failed": float(
                state.get("commit_failed", 0) or 0
            ),
            "adaptive_consolidation/replacement_reconciled": float(
                state.get("reconciled", 0) or 0
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
        st = normalize_stage(stage)

        def _body(json_path: Path) -> ReservationResult:
            state, err = self._load_or_create(json_path, tick_id, st, budget)
            if state is None:
                return ReservationResult(ok=False, reason=err or "ledger_corrupt")
            now = _now_wall()
            exp_err = self._expire_stale(state, now_wall=now)
            if exp_err:
                return ReservationResult(ok=False, reason=exp_err)

            if lineages_incomplete(population_lineages, num_policies):
                state["blocked_lineage_missing"] = int(
                    state.get("blocked_lineage_missing", 0) or 0
                ) + 1
                self._save_unlocked(json_path, state)
                return ReservationResult(
                    ok=False,
                    reason="lineage_missing",
                    telemetry=self.telemetry_from_state(state),
                )

            reserved_n, committed_n = self._counts(state)
            if reserved_n + committed_n >= int(state["budget"]):
                state["blocked_budget"] = int(state.get("blocked_budget", 0) or 0) + 1
                self._save_unlocked(json_path, state)
                return ReservationResult(
                    ok=False,
                    reason="replacement_budget_exhausted",
                    telemetry=self.telemetry_from_state(state),
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
                return ReservationResult(
                    ok=False,
                    reason="projected_lineage_missing",
                    telemetry=self.telemetry_from_state(state),
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
                    ok=False, reason="diversity_blocked", telemetry=tele
                )

            rid = uuid.uuid4().hex
            token = uuid.uuid4().hex
            pid = int(owner_pid if owner_pid is not None else os.getpid())
            lease = float(self.reservation_timeout_s)
            rec = {
                "reservation_id": rid,
                "owner_policy": int(owner_policy),
                "owner_pid": pid,
                "owner_token": token,
                "target": int(target_idx),
                "source": int(source_idx),
                "source_lineage": str(source_lineage),
                "status": "reserved",
                "created_wall": float(now),
                "last_heartbeat_wall": float(now),
                "lease_timeout_s": lease,
            }
            state.setdefault("reservations", {})[rid] = rec
            self._save_unlocked(json_path, state)
            tele = self.telemetry_from_state(state)
            tele["adaptive_consolidation/distinct_lineage_count"] = float(distinct)
            tele["adaptive_consolidation/min_distinct_lineages"] = float(
                min_distinct_lineages
            )
            handle = ReservationHandle(
                tick_id=int(tick_id),
                stage=st,
                reservation_id=rid,
                owner_policy=int(owner_policy),
                owner_pid=pid,
                owner_token=token,
                target=int(target_idx),
                source=int(source_idx),
                source_lineage=str(source_lineage),
                lease_timeout_s=lease,
            )
            return ReservationResult(
                ok=True, reason="reserved", handle=handle, telemetry=tele
            )

        return self._with_lock(int(tick_id), st, _body)

    def _owner_matches(self, rec: dict[str, Any], handle: ReservationHandle) -> bool:
        return (
            str(rec.get("owner_token", "")) == str(handle.owner_token)
            and int(rec.get("owner_policy", -1)) == int(handle.owner_policy)
            and int(rec.get("owner_pid", -1)) == int(handle.owner_pid)
        )

    def renew(self, handle: ReservationHandle) -> FinalizeResult:
        st = normalize_stage(handle.stage)

        def _body(json_path: Path) -> FinalizeResult:
            if not json_path.exists():
                return FinalizeResult(ok=False, reason="ledger_missing", status="failed")
            try:
                state = json.loads(json_path.read_text())
            except Exception:
                return FinalizeResult(ok=False, reason="ledger_corrupt", status="failed")
            if not isinstance(state, dict):
                return FinalizeResult(ok=False, reason="ledger_corrupt", status="failed")
            try:
                file_budget = int(state.get("budget", -1))
            except Exception:
                return FinalizeResult(ok=False, reason="ledger_corrupt", status="failed")
            mismatch = self._validate_immutable(state, handle.tick_id, st, file_budget)
            if mismatch:
                return FinalizeResult(ok=False, reason=mismatch, status="failed")
            state.setdefault("reservations", {})
            now = _now_wall()
            exp_err = self._expire_stale(state, now_wall=now)
            if exp_err:
                return FinalizeResult(ok=False, reason=exp_err, status="failed")
            rec = (state.get("reservations") or {}).get(handle.reservation_id)
            if not isinstance(rec, dict):
                return FinalizeResult(ok=False, reason="reservation_missing", status="failed")
            if not self._owner_matches(rec, handle):
                return FinalizeResult(ok=False, reason="owner_mismatch", status="failed")
            if str(rec.get("status", "")) != "reserved":
                return FinalizeResult(
                    ok=False,
                    reason=f"not_reserved:{rec.get('status')}",
                    status=str(rec.get("status", "")),
                )
            rec["last_heartbeat_wall"] = float(now)
            self._save_unlocked(json_path, state)
            return FinalizeResult(
                ok=True,
                reason="renewed",
                status="reserved",
                telemetry=self.telemetry_from_state(state),
            )

        return self._with_lock(int(handle.tick_id), st, _body)

    def _append_committed_idempotent(
        self, state: dict[str, Any], rec: dict[str, Any], rid: str
    ) -> bool:
        """Append committed entry if not already present. Returns True if newly added."""
        committed = state.setdefault("committed", [])
        for item in committed:
            if not isinstance(item, dict):
                continue
            if str(item.get("reservation_id", "")) == rid:
                return False
            if (
                int(item.get("target", -2)) == int(rec.get("target", -1))
                and int(item.get("source", -2)) == int(rec.get("source", -1))
                and str(item.get("source_lineage", "")) == str(rec.get("source_lineage", ""))
            ):
                return False
        committed.append(
            {
                "reservation_id": rid,
                "target": int(rec.get("target", -1)),
                "source": int(rec.get("source", -1)),
                "source_lineage": str(rec.get("source_lineage", "")),
            }
        )
        return True

    def commit(self, handle: ReservationHandle) -> FinalizeResult:
        st = normalize_stage(handle.stage)

        def _body(json_path: Path) -> FinalizeResult:
            if not json_path.exists():
                return FinalizeResult(ok=False, reason="ledger_missing", status="failed")
            try:
                state = json.loads(json_path.read_text())
            except Exception:
                return FinalizeResult(ok=False, reason="ledger_corrupt", status="failed")
            if not isinstance(state, dict):
                return FinalizeResult(ok=False, reason="ledger_corrupt", status="failed")
            mismatch = self._validate_immutable(
                state, handle.tick_id, st, int(state.get("budget", -999999))
            )
            if mismatch:
                return FinalizeResult(ok=False, reason=mismatch, status="failed")
            now = _now_wall()
            exp_err = self._expire_stale(state, now_wall=now)
            if exp_err:
                return FinalizeResult(ok=False, reason=exp_err, status="failed")
            rid = str(handle.reservation_id)
            rec = (state.get("reservations") or {}).get(rid)
            if not isinstance(rec, dict):
                return FinalizeResult(ok=False, reason="reservation_missing", status="failed")
            if not self._owner_matches(rec, handle):
                return FinalizeResult(ok=False, reason="owner_mismatch", status="failed")
            status = str(rec.get("status", ""))
            if status == "committed":
                return FinalizeResult(
                    ok=True,
                    reason="already_committed",
                    status="committed",
                    telemetry=self.telemetry_from_state(state),
                )
            if status == "expired":
                # Caller may reconcile separately.
                return FinalizeResult(
                    ok=False, reason="expired", status="expired",
                    telemetry=self.telemetry_from_state(state),
                )
            if status != "reserved":
                return FinalizeResult(
                    ok=False, reason=f"not_reserved:{status}", status=status
                )
            rec["status"] = "committed"
            rec["committed_wall"] = float(now)
            self._append_committed_idempotent(state, rec, rid)
            self._save_unlocked(json_path, state)
            return FinalizeResult(
                ok=True,
                reason="committed",
                status="committed",
                telemetry=self.telemetry_from_state(state),
            )

        return self._with_lock(int(handle.tick_id), st, _body)

    def reconcile_commit(self, handle: ReservationHandle) -> FinalizeResult:
        """Idempotently record a successful restart when normal commit failed."""
        st = normalize_stage(handle.stage)

        def _body(json_path: Path) -> FinalizeResult:
            if not json_path.exists():
                return FinalizeResult(ok=False, reason="ledger_missing", status="failed")
            try:
                state = json.loads(json_path.read_text())
            except Exception:
                return FinalizeResult(ok=False, reason="ledger_corrupt", status="failed")
            if not isinstance(state, dict):
                return FinalizeResult(ok=False, reason="ledger_corrupt", status="failed")
            mismatch = self._validate_immutable(
                state, handle.tick_id, st, int(state.get("budget", -999999))
            )
            if mismatch:
                return FinalizeResult(ok=False, reason=mismatch, status="failed")
            now = _now_wall()
            self._expire_stale(state, now_wall=now)
            rid = str(handle.reservation_id)
            rec = (state.get("reservations") or {}).get(rid)
            if not isinstance(rec, dict):
                # Reconstruct from handle for reconciliation.
                rec = {
                    "reservation_id": rid,
                    "owner_policy": int(handle.owner_policy),
                    "owner_pid": int(handle.owner_pid),
                    "owner_token": str(handle.owner_token),
                    "target": int(handle.target),
                    "source": int(handle.source),
                    "source_lineage": str(handle.source_lineage),
                    "status": "expired",
                }
                state.setdefault("reservations", {})[rid] = rec
            if not self._owner_matches(rec, handle):
                return FinalizeResult(ok=False, reason="owner_mismatch", status="failed")
            status = str(rec.get("status", ""))
            if status == "committed":
                return FinalizeResult(
                    ok=True,
                    reason="already_committed",
                    status="committed",
                    telemetry=self.telemetry_from_state(state),
                )
            added = self._append_committed_idempotent(state, rec, rid)
            rec["status"] = "committed"
            rec["reconciled_wall"] = float(now)
            if added:
                state["reconciled"] = int(state.get("reconciled", 0) or 0) + 1
                state["commit_failed"] = int(state.get("commit_failed", 0) or 0) + 1
            self._save_unlocked(json_path, state)
            return FinalizeResult(
                ok=True,
                reason="reconciled" if added else "already_committed",
                status="committed",
                telemetry=self.telemetry_from_state(state),
            )

        return self._with_lock(int(handle.tick_id), st, _body)

    def release(self, handle: ReservationHandle) -> FinalizeResult:
        st = normalize_stage(handle.stage)

        def _body(json_path: Path) -> FinalizeResult:
            if not json_path.exists():
                return FinalizeResult(ok=False, reason="ledger_missing", status="failed")
            try:
                state = json.loads(json_path.read_text())
            except Exception:
                return FinalizeResult(ok=False, reason="ledger_corrupt", status="failed")
            if not isinstance(state, dict):
                return FinalizeResult(ok=False, reason="ledger_corrupt", status="failed")
            mismatch = self._validate_immutable(
                state, handle.tick_id, st, int(state.get("budget", -999999))
            )
            if mismatch:
                return FinalizeResult(ok=False, reason=mismatch, status="failed")
            now = _now_wall()
            self._expire_stale(state, now_wall=now)
            rid = str(handle.reservation_id)
            rec = (state.get("reservations") or {}).get(rid)
            if not isinstance(rec, dict):
                return FinalizeResult(ok=False, reason="reservation_missing", status="failed")
            if not self._owner_matches(rec, handle):
                return FinalizeResult(ok=False, reason="owner_mismatch", status="failed")
            status = str(rec.get("status", ""))
            if status == "released":
                return FinalizeResult(
                    ok=True,
                    reason="already_released",
                    status="released",
                    telemetry=self.telemetry_from_state(state),
                )
            if status == "committed":
                return FinalizeResult(
                    ok=False, reason="already_committed", status="committed"
                )
            if status not in ("reserved", "expired"):
                return FinalizeResult(
                    ok=False, reason=f"not_releaseable:{status}", status=status
                )
            rec["status"] = "released"
            rec["released_wall"] = float(now)
            self._save_unlocked(json_path, state)
            return FinalizeResult(
                ok=True,
                reason="released",
                status="released",
                telemetry=self.telemetry_from_state(state),
            )

        return self._with_lock(int(handle.tick_id), st, _body)

    def record_bypass(
        self,
        *,
        tick_id: int,
        stage: str,
        budget: int,
        target_idx: int,
        reason: str,
    ) -> dict[str, float]:
        st = normalize_stage(stage)
        reason_n = str(reason or "unknown")

        def _body(json_path: Path) -> dict[str, float]:
            state, err = self._load_or_create(json_path, tick_id, st, budget)
            if state is None:
                return {
                    "adaptive_consolidation/replacement_budget_bypassed": 0.0,
                    "adaptive_consolidation/ledger_error": 1.0,
                }
            now = _now_wall()
            self._expire_stale(state, now_wall=now)
            bypassed = state.setdefault("bypassed", [])
            for item in bypassed:
                if (
                    isinstance(item, dict)
                    and int(item.get("target", -1)) == int(target_idx)
                    and str(item.get("reason", "")) == reason_n
                ):
                    return self.telemetry_from_state(state)
            bypassed.append(
                {
                    "target": int(target_idx),
                    "reason": reason_n,
                    "wall": float(now),
                }
            )
            self._save_unlocked(json_path, state)
            return self.telemetry_from_state(state)

        return self._with_lock(int(tick_id), st, _body)

    def snapshot_telemetry(
        self, tick_id: int, *, stage: str, budget: int
    ) -> dict[str, float]:
        st = normalize_stage(stage)

        def _body(json_path: Path) -> dict[str, float]:
            state, err = self._load_or_create(json_path, tick_id, st, budget)
            if state is None:
                return {"adaptive_consolidation/ledger_error": 1.0}
            self._expire_stale(state, now_wall=_now_wall())
            return self.telemetry_from_state(state)

        return self._with_lock(int(tick_id), st, _body)

    @contextmanager
    def reservation_lease(
        self,
        handle: ReservationHandle,
        *,
        heartbeat_interval_s: float | None = None,
    ) -> Iterator[None]:
        """Periodic renew while restart runs; always stops the heartbeat thread."""
        interval = float(
            heartbeat_interval_s
            if heartbeat_interval_s is not None
            else min(self.heartbeat_interval_s, max(0.05, handle.lease_timeout_s / 4.0))
        )
        stop = threading.Event()
        errors: list[str] = []

        def _loop() -> None:
            while not stop.wait(interval):
                try:
                    fr = self.renew(handle)
                    if not fr.ok:
                        errors.append(str(fr.reason))
                        break
                except Exception as exc:  # pragma: no cover
                    errors.append(str(exc))
                    break

        th = threading.Thread(
            target=_loop, name=f"repl-lease-{handle.reservation_id[:8]}", daemon=True
        )
        th.start()
        try:
            yield
        finally:
            stop.set()
            th.join(timeout=max(1.0, interval * 2.0))
            if errors:
                # Surface heartbeat failure without silencing the original exception.
                raise RuntimeError(
                    f"reservation heartbeat failed: {errors[-1]}"
                )
