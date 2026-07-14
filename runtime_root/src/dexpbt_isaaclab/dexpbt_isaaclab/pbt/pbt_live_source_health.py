"""Shared live-source health gate for ordinary and chronic PBT paths.

Chronic path may only replace the *skill_delta* criterion; every other source
health / eligibility check goes through ``evaluate_common_live_source_eligibility``.

Snapshot freshness uses the production commit barrier:

- ``LATEST.commit`` → commit directory
- ``validate_snapshot_commit(commit_dir, tick=...)``
- MANIFEST ↔ population-state lineage_id and population_frame (exact match)
- published checkpoint bytes match MANIFEST ``model.pth`` hash

``population_frame_complete.json`` is a *final* frame-stop marker and must
**not** gate mid-training PBT snapshot freshness.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from dexpbt_isaaclab.pbt.pbt_control_plane import validate_snapshot_commit


@dataclass
class CommonSourceEligibility:
    eligible: bool
    reason: str
    checkpoint_valid: bool = False
    lineage_valid: bool = False
    freshness_valid: bool = False
    restart_burnin_valid: bool = False
    probation_valid: bool = False
    collapse_valid: bool = False
    finite_metrics_valid: bool = False
    source_gate_valid: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PeerCheckpointHealth:
    """Structured peer snapshot health for chronic source selection."""

    checkpoint_exists: bool = False
    checkpoint_path: str | None = None
    latest_commit_exists: bool = False
    latest_commit_path: str | None = None
    latest_commit_parse_valid: bool = False
    commit_dir: str | None = None
    commit_dir_exists: bool = False
    snapshot_commit_valid: bool = False
    snapshot_commit_reason: str = "unchecked"
    manifest_valid: bool = False
    manifest_lineage_id: str | None = None
    state_lineage_id: str | None = None
    lineage_valid: bool = False
    manifest_population_frame: int | None = None
    state_population_frame: int | None = None
    checkpoint_identity_valid: bool = False
    freshness_valid: bool = False
    freshness_reason: str = "unchecked"
    final_health_allowed: bool = False
    final_health_reason: str = "unchecked"
    extras: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d.pop("extras", None)
        d.update(self.extras)
        return d

    @property
    def checkpoint_commit_ok(self) -> bool | None:
        """Map to common-gate commit field (fail-closed on unknown)."""
        if not self.latest_commit_exists:
            return False
        if not self.latest_commit_parse_valid:
            return False
        if not self.commit_dir_exists:
            return False
        return bool(self.snapshot_commit_valid)

    @property
    def lineage_ok(self) -> bool | None:
        if self.state_lineage_id is None or str(self.state_lineage_id).strip() == "":
            return False
        if self.manifest_lineage_id is None:
            return False
        return bool(self.lineage_valid)

    @property
    def freshness_ok(self) -> bool | None:
        if not self.snapshot_commit_valid:
            return False
        return bool(self.freshness_valid)


def _finite(*vals: float) -> bool:
    for v in vals:
        try:
            if not math.isfinite(float(v)):
                return False
        except Exception:
            return False
    return True


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _parse_population_frame(state: dict[str, Any] | None) -> int | None:
    if not state:
        return None
    for key in ("population_frame", "frame"):
        raw = state.get(key)
        if raw is None:
            continue
        try:
            val = int(raw)
        except Exception:
            return None
        if not math.isfinite(float(val)):
            return None
        return val
    return None


def inspect_peer_snapshot_health(
    *,
    policy_workspace: Path,
    state: dict[str, Any] | None,
    tick: int,
) -> PeerCheckpointHealth:
    """Inspect peer published snapshot against production commit barrier.

    Freshness semantics (exact, fail-closed):

    1. Population state has a checkpoint path that is an existing file.
    2. ``LATEST.commit`` exists and parses to a JSON dict with non-empty
       ``commit_dir``.
    3. That commit directory exists under ``policy_workspace``.
    4. ``validate_snapshot_commit(commit_dir, tick=tick)`` returns
       ``(True, "ok")`` (MANIFEST, member hashes, lineage_id, quantized frame).
    5. MANIFEST ``lineage_id`` and state ``lineage_id`` both present and equal.
    6. MANIFEST ``population_frame`` equals state ``population_frame``/``frame``
       (exact integer match; no completion-marker substitute).
    7. SHA-256 of the published checkpoint equals MANIFEST
       ``member_shas["model.pth"]`` (proves identity with LATEST commit).

    Does **not** consult ``population_frame_complete.json``.
    """
    out = PeerCheckpointHealth()
    st = state or {}
    checkpoint = st.get("checkpoint")
    out.checkpoint_path = str(checkpoint) if checkpoint else None
    ckpt_path = Path(str(checkpoint)) if checkpoint else None
    out.checkpoint_exists = bool(ckpt_path is not None and ckpt_path.is_file())
    out.state_lineage_id = (
        str(st["lineage_id"]) if st.get("lineage_id") is not None else None
    )
    out.state_population_frame = _parse_population_frame(st)

    latest_path = Path(policy_workspace) / "LATEST.commit"
    out.latest_commit_path = str(latest_path)
    out.latest_commit_exists = latest_path.is_file()
    if not out.latest_commit_exists:
        out.final_health_reason = "latest_commit_missing"
        out.snapshot_commit_reason = "latest_commit_missing"
        out.freshness_reason = "latest_commit_missing"
        return out

    try:
        latest_payload = json.loads(latest_path.read_text())
    except Exception:
        out.final_health_reason = "latest_commit_corrupt"
        out.snapshot_commit_reason = "latest_commit_corrupt"
        out.freshness_reason = "latest_commit_corrupt"
        return out
    if not isinstance(latest_payload, dict):
        out.final_health_reason = "latest_commit_not_dict"
        out.snapshot_commit_reason = "latest_commit_not_dict"
        out.freshness_reason = "latest_commit_not_dict"
        return out
    out.latest_commit_parse_valid = True
    commit_dir_name = str(latest_payload.get("commit_dir") or "").strip()
    if not commit_dir_name:
        out.final_health_reason = "commit_dir_empty"
        out.snapshot_commit_reason = "commit_dir_empty"
        out.freshness_reason = "commit_dir_empty"
        return out
    out.commit_dir = commit_dir_name
    commit_dir = Path(policy_workspace) / commit_dir_name
    out.commit_dir_exists = commit_dir.is_dir()
    if not out.commit_dir_exists:
        out.final_health_reason = "commit_dir_missing"
        out.snapshot_commit_reason = "commit_dir_missing"
        out.freshness_reason = "commit_dir_missing"
        return out

    ok, reason = validate_snapshot_commit(commit_dir, tick=int(tick))
    out.snapshot_commit_valid = bool(ok)
    out.snapshot_commit_reason = str(reason)
    if not ok:
        out.final_health_reason = f"snapshot_commit_invalid:{reason}"
        out.freshness_reason = f"snapshot_commit_invalid:{reason}"
        return out

    manifest_path = commit_dir / "MANIFEST"
    try:
        manifest = json.loads(manifest_path.read_text())
    except Exception:
        out.final_health_reason = "manifest_unreadable_after_validate"
        out.freshness_reason = "manifest_unreadable_after_validate"
        return out
    if not isinstance(manifest, dict):
        out.final_health_reason = "manifest_not_dict"
        out.freshness_reason = "manifest_not_dict"
        return out
    out.manifest_valid = True
    mid = manifest.get("lineage_id")
    out.manifest_lineage_id = str(mid) if mid is not None else None
    try:
        mpf = manifest.get("population_frame")
        out.manifest_population_frame = int(mpf) if mpf is not None else None
    except Exception:
        out.manifest_population_frame = None

    if not out.checkpoint_exists:
        out.final_health_reason = "checkpoint_missing"
        out.freshness_reason = "checkpoint_missing"
        return out

    if out.state_lineage_id is None or str(out.state_lineage_id).strip() == "":
        out.final_health_reason = "state_lineage_missing"
        out.freshness_reason = "state_lineage_missing"
        return out
    if out.manifest_lineage_id is None or str(out.manifest_lineage_id).strip() == "":
        out.final_health_reason = "manifest_lineage_missing"
        out.freshness_reason = "manifest_lineage_missing"
        return out
    out.lineage_valid = str(out.state_lineage_id) == str(out.manifest_lineage_id)
    if not out.lineage_valid:
        out.final_health_reason = "lineage_mismatch"
        out.freshness_reason = "lineage_mismatch"
        return out

    if out.manifest_population_frame is None:
        out.final_health_reason = "manifest_population_frame_missing"
        out.freshness_reason = "manifest_population_frame_missing"
        return out
    if out.state_population_frame is None:
        out.final_health_reason = "state_population_frame_missing"
        out.freshness_reason = "state_population_frame_missing"
        return out
    if int(out.manifest_population_frame) != int(out.state_population_frame):
        out.freshness_reason = "population_frame_mismatch"
        out.final_health_reason = "population_frame_mismatch"
        return out

    member_shas = manifest.get("member_shas")
    if not isinstance(member_shas, dict) or "model.pth" not in member_shas:
        out.freshness_reason = "manifest_model_hash_missing"
        out.final_health_reason = "manifest_model_hash_missing"
        return out
    try:
        published_sha = _sha256_file(ckpt_path)  # type: ignore[arg-type]
    except Exception:
        out.freshness_reason = "checkpoint_hash_unreadable"
        out.final_health_reason = "checkpoint_hash_unreadable"
        return out
    expected = str(member_shas["model.pth"])
    out.checkpoint_identity_valid = published_sha == expected
    if not out.checkpoint_identity_valid:
        out.freshness_reason = "checkpoint_identity_mismatch"
        out.final_health_reason = "checkpoint_identity_mismatch"
        return out

    out.freshness_valid = True
    out.freshness_reason = "ok"
    out.final_health_allowed = True
    out.final_health_reason = "ok"
    return out


def evaluate_common_live_source_eligibility(
    *,
    has_state: bool,
    ranking_score: float,
    nsucc_mean: float,
    source_gate_ok: bool,
    source_gate_reason: str,
    restart_burnin_ok: bool,
    restart_burnin_reason: str,
    checkpoint_exists: bool,
    checkpoint_commit_ok: bool | None,
    lineage_ok: bool | None,
    freshness_ok: bool | None,
    probation_ok: bool | None,
    collapse_ok: bool | None,
) -> CommonSourceEligibility:
    """Fail-closed common health gate used by ordinary and chronic live sources.

    Any missing / unknown health field (``None``) fails closed.
    """
    out = CommonSourceEligibility(eligible=False, reason="init")
    if not has_state:
        out.reason = "no_state"
        return out
    out.finite_metrics_valid = _finite(ranking_score, nsucc_mean)
    if not out.finite_metrics_valid:
        out.reason = "nonfinite_metrics"
        return out

    out.restart_burnin_valid = bool(restart_burnin_ok)
    if not out.restart_burnin_valid:
        out.reason = str(restart_burnin_reason or "restart_burnin")
        return out

    out.source_gate_valid = bool(source_gate_ok)
    if not out.source_gate_valid:
        out.reason = str(source_gate_reason or "source_gate_rejected")
        return out

    out.checkpoint_valid = bool(checkpoint_exists) and (
        checkpoint_commit_ok is True
    )
    if not bool(checkpoint_exists):
        out.reason = "checkpoint_missing"
        return out
    if checkpoint_commit_ok is None:
        out.reason = "checkpoint_commit_unknown"
        return out
    if checkpoint_commit_ok is False:
        out.reason = "checkpoint_commit_invalid"
        return out

    if lineage_ok is None:
        out.reason = "lineage_unknown"
        return out
    out.lineage_valid = bool(lineage_ok)
    if not out.lineage_valid:
        out.reason = "lineage_mismatch"
        return out

    if freshness_ok is None:
        out.reason = "freshness_unknown"
        return out
    out.freshness_valid = bool(freshness_ok)
    if not out.freshness_valid:
        out.reason = "snapshot_stale_or_incomplete"
        return out

    if probation_ok is None:
        out.reason = "probation_unknown"
        return out
    out.probation_valid = bool(probation_ok)
    if not out.probation_valid:
        out.reason = "source_probation"
        return out

    if collapse_ok is None:
        out.reason = "collapse_unknown"
        return out
    out.collapse_valid = bool(collapse_ok)
    if not out.collapse_valid:
        out.reason = "source_collapse_or_unhealthy"
        return out

    out.eligible = True
    out.reason = "ok"
    return out


def evaluate_peer_checkpoint_health(
    *,
    checkpoint: str | None,
    state_lineage_id: str | None,
    latest_commit_exists: bool | None,
    commit_lineage_id: str | None,
    population_frame_complete: bool | None = None,
    **_ignored: Any,
) -> tuple[bool | None, bool | None, bool | None, str]:
    """Deprecated shim — use ``inspect_peer_snapshot_health``.

    ``population_frame_complete`` is **ignored** (frame-stop marker must not
    gate snapshot freshness). Prefer the structured inspector for production.
    """
    del population_frame_complete  # intentionally unused
    if not checkpoint:
        return False, False, False, "checkpoint_missing"
    if latest_commit_exists is None:
        return None, None, None, "latest_commit_unknown"
    if not latest_commit_exists:
        return False, False, False, "latest_commit_missing"
    if state_lineage_id is None or str(state_lineage_id).strip() == "":
        return True, False, None, "state_lineage_missing"
    if commit_lineage_id is None:
        return True, None, None, "commit_lineage_unknown"
    lineage_ok = str(state_lineage_id) == str(commit_lineage_id)
    if not lineage_ok:
        return True, False, False, "lineage_mismatch"
    # Without a full workspace inspect we cannot prove freshness; fail-closed.
    return True, True, None, "freshness_requires_inspect_peer_snapshot_health"


@dataclass(frozen=True)
class SnapshotValidationFingerprint:
    """Lightweight identity for immutable snapshot commit validation cache."""

    policy_idx: int
    commit_id: str
    commit_dir: str
    latest_size: int
    latest_mtime_ns: int
    manifest_size: int
    manifest_mtime_ns: int
    checkpoint_path: str
    checkpoint_size: int
    checkpoint_mtime_ns: int
    state_lineage_id: str
    state_population_frame: int
    tick: int
    commit_model_size: int = 0
    commit_model_mtime_ns: int = 0
    commit_meta_size: int = 0
    commit_meta_mtime_ns: int = 0


@dataclass
class CachedSnapshotHealth:
    fingerprint: SnapshotValidationFingerprint
    health: PeerCheckpointHealth
    validated_at_monotonic: float


@dataclass
class SnapshotInspectTelemetry:
    health: PeerCheckpointHealth
    cache_hit: bool
    invalidated_reason: str
    validation_latency_s: float
    full_hash_executed: bool
    fingerprint: SnapshotValidationFingerprint | None = None


def _file_stat_size_mtime(path: Path) -> tuple[int, int] | None:
    try:
        st = path.stat()
        return int(st.st_size), int(getattr(st, "st_mtime_ns", int(st.st_mtime * 1e9)))
    except Exception:
        return None


def build_snapshot_validation_fingerprint(
    *,
    policy_idx: int,
    policy_workspace: Path,
    state: dict[str, Any] | None,
    tick: int,
) -> tuple[SnapshotValidationFingerprint | None, str]:
    """Build a lightweight fingerprint or explain why it cannot be built.

    Failures here force a full (uncached) inspect and never hit a prior entry.
    """
    st = state or {}
    lineage = st.get("lineage_id")
    if lineage is None or str(lineage).strip() == "":
        return None, "lineage_missing_for_fingerprint"
    frame = _parse_population_frame(st)
    if frame is None:
        return None, "frame_missing_for_fingerprint"
    checkpoint = st.get("checkpoint")
    if not checkpoint:
        return None, "checkpoint_missing_for_fingerprint"
    ckpt_path = Path(str(checkpoint))
    ckpt_stat = _file_stat_size_mtime(ckpt_path)
    if ckpt_stat is None:
        return None, "checkpoint_stat_unreadable"

    latest_path = Path(policy_workspace) / "LATEST.commit"
    latest_stat = _file_stat_size_mtime(latest_path)
    if latest_stat is None:
        return None, "latest_missing_or_unreadable"
    try:
        latest_payload = json.loads(latest_path.read_text())
    except Exception:
        return None, "latest_commit_corrupt"
    if not isinstance(latest_payload, dict):
        return None, "latest_commit_not_dict"
    commit_id = str(latest_payload.get("commit_id") or "").strip()
    commit_dir_name = str(latest_payload.get("commit_dir") or "").strip()
    if not commit_id or not commit_dir_name:
        return None, "latest_commit_incomplete"
    commit_dir = Path(policy_workspace) / commit_dir_name
    if not commit_dir.is_dir():
        return None, "commit_dir_missing"
    manifest_path = commit_dir / "MANIFEST"
    manifest_stat = _file_stat_size_mtime(manifest_path)
    if manifest_stat is None:
        return None, "manifest_missing_or_unreadable"
    model_stat = _file_stat_size_mtime(commit_dir / "model.pth") or (0, 0)
    meta_stat = _file_stat_size_mtime(commit_dir / "meta.json") or (0, 0)
    return (
        SnapshotValidationFingerprint(
            policy_idx=int(policy_idx),
            commit_id=commit_id,
            commit_dir=commit_dir_name,
            latest_size=int(latest_stat[0]),
            latest_mtime_ns=int(latest_stat[1]),
            manifest_size=int(manifest_stat[0]),
            manifest_mtime_ns=int(manifest_stat[1]),
            checkpoint_path=str(ckpt_path.resolve())
            if ckpt_path.exists()
            else str(ckpt_path),
            checkpoint_size=int(ckpt_stat[0]),
            checkpoint_mtime_ns=int(ckpt_stat[1]),
            state_lineage_id=str(lineage),
            state_population_frame=int(frame),
            tick=int(tick),
            commit_model_size=int(model_stat[0]),
            commit_model_mtime_ns=int(model_stat[1]),
            commit_meta_size=int(meta_stat[0]),
            commit_meta_mtime_ns=int(meta_stat[1]),
        ),
        "ok",
    )


def fingerprint_invalidation_reason(
    previous: SnapshotValidationFingerprint | None,
    current: SnapshotValidationFingerprint,
) -> str:
    if previous is None:
        return "first_validation"
    if (
        previous.commit_id != current.commit_id
        or previous.commit_dir != current.commit_dir
        or previous.latest_size != current.latest_size
        or previous.latest_mtime_ns != current.latest_mtime_ns
    ):
        return "latest_changed"
    if (
        previous.manifest_size != current.manifest_size
        or previous.manifest_mtime_ns != current.manifest_mtime_ns
        or previous.commit_model_size != current.commit_model_size
        or previous.commit_model_mtime_ns != current.commit_model_mtime_ns
        or previous.commit_meta_size != current.commit_meta_size
        or previous.commit_meta_mtime_ns != current.commit_meta_mtime_ns
    ):
        return "manifest_changed"
    if (
        previous.checkpoint_path != current.checkpoint_path
        or previous.checkpoint_size != current.checkpoint_size
        or previous.checkpoint_mtime_ns != current.checkpoint_mtime_ns
    ):
        return "checkpoint_stat_changed"
    if previous.state_lineage_id != current.state_lineage_id:
        return "lineage_changed"
    if previous.state_population_frame != current.state_population_frame:
        return "frame_changed"
    if previous.tick != current.tick:
        return "tick_changed"
    if previous != current:
        return "fingerprint_changed"
    return "cache_hit"


class PeerSnapshotHealthCache:
    """Process-local LRU of peer snapshot health keyed by fingerprint.

    Only immutable commit/identity results are cached. Probation, skill_delta,
    collapse, and ranking scores are never stored here.
    """

    def __init__(self, *, max_per_policy: int = 3) -> None:
        self.max_per_policy = max(1, int(max_per_policy))
        self._by_policy: dict[int, list[CachedSnapshotHealth]] = {}
        self.hit_count = 0
        self.miss_count = 0
        self.last_telemetry: SnapshotInspectTelemetry | None = None

    def entry_count(self) -> int:
        return sum(len(v) for v in self._by_policy.values())

    def clear(self) -> None:
        self._by_policy.clear()

    def _lookup(
        self, fp: SnapshotValidationFingerprint
    ) -> CachedSnapshotHealth | None:
        entries = self._by_policy.get(int(fp.policy_idx), [])
        for item in entries:
            if item.fingerprint == fp:
                return item
        return None

    def _store(self, item: CachedSnapshotHealth) -> None:
        pidx = int(item.fingerprint.policy_idx)
        entries = [
            e
            for e in self._by_policy.get(pidx, [])
            if e.fingerprint != item.fingerprint
        ]
        entries.insert(0, item)
        self._by_policy[pidx] = entries[: self.max_per_policy]

    def inspect(
        self,
        *,
        policy_idx: int,
        policy_workspace: Path,
        state: dict[str, Any] | None,
        tick: int,
        validate_fn: Any | None = None,
        sha256_fn: Any | None = None,
    ) -> SnapshotInspectTelemetry:
        """Inspect with fingerprint-gated cache. Fail-close on corruption."""
        import copy
        import time

        t0 = time.monotonic()
        fp, fp_reason = build_snapshot_validation_fingerprint(
            policy_idx=int(policy_idx),
            policy_workspace=Path(policy_workspace),
            state=state,
            tick=int(tick),
        )
        if fp is None:
            self.miss_count += 1
            health = self._full_inspect(
                policy_workspace=Path(policy_workspace),
                state=state,
                tick=int(tick),
                validate_fn=validate_fn,
                sha256_fn=sha256_fn,
            )
            tele = SnapshotInspectTelemetry(
                health=health,
                cache_hit=False,
                invalidated_reason=str(fp_reason),
                validation_latency_s=float(time.monotonic() - t0),
                full_hash_executed=True,
                fingerprint=None,
            )
            self.last_telemetry = tele
            return tele

        cached = self._lookup(fp)
        if cached is not None:
            self.hit_count += 1
            # Move to front (LRU).
            self._store(cached)
            health = copy.deepcopy(cached.health)
            tele = SnapshotInspectTelemetry(
                health=health,
                cache_hit=True,
                invalidated_reason="cache_hit",
                validation_latency_s=float(time.monotonic() - t0),
                full_hash_executed=False,
                fingerprint=fp,
            )
            self.last_telemetry = tele
            return tele

        # Find nearest prior entry for this policy to classify miss reason.
        prior = None
        prior_entries = self._by_policy.get(int(policy_idx), [])
        if prior_entries:
            prior = prior_entries[0].fingerprint
        reason = fingerprint_invalidation_reason(prior, fp)

        self.miss_count += 1
        health = self._full_inspect(
            policy_workspace=Path(policy_workspace),
            state=state,
            tick=int(tick),
            validate_fn=validate_fn,
            sha256_fn=sha256_fn,
        )
        # Cache successful validations always; also cache negative results that
        # completed under a full fingerprint (mtime/size still gates reuse).
        # Do not cache when fingerprint build failed (handled above).
        self._store(
            CachedSnapshotHealth(
                fingerprint=fp,
                health=copy.deepcopy(health),
                validated_at_monotonic=float(time.monotonic()),
            )
        )
        tele = SnapshotInspectTelemetry(
            health=health,
            cache_hit=False,
            invalidated_reason=reason if reason != "cache_hit" else "first_validation",
            validation_latency_s=float(time.monotonic() - t0),
            full_hash_executed=True,
            fingerprint=fp,
        )
        self.last_telemetry = tele
        return tele

    @staticmethod
    def _full_inspect(
        *,
        policy_workspace: Path,
        state: dict[str, Any] | None,
        tick: int,
        validate_fn: Any | None,
        sha256_fn: Any | None,
    ) -> PeerCheckpointHealth:
        """Run full validation, optionally injecting spies for tests."""
        if validate_fn is None and sha256_fn is None:
            return inspect_peer_snapshot_health(
                policy_workspace=policy_workspace,
                state=state,
                tick=tick,
            )
        # Spy path: temporarily patch module symbols used by inspect.
        import dexpbt_isaaclab.pbt.pbt_live_source_health as mod

        old_validate = mod.validate_snapshot_commit
        old_sha = mod._sha256_file
        try:
            if validate_fn is not None:
                mod.validate_snapshot_commit = validate_fn  # type: ignore[assignment]
            if sha256_fn is not None:
                mod._sha256_file = sha256_fn  # type: ignore[assignment]
            return inspect_peer_snapshot_health(
                policy_workspace=policy_workspace,
                state=state,
                tick=tick,
            )
        finally:
            mod.validate_snapshot_commit = old_validate
            mod._sha256_file = old_sha
