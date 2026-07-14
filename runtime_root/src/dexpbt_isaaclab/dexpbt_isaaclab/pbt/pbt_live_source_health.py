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
