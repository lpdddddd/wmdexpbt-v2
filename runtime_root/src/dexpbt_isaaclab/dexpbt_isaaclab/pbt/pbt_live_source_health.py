"""Shared live-source health gate for ordinary and chronic PBT paths.

Chronic path may only replace the *skill_delta* criterion; every other source
health / eligibility check goes through ``evaluate_common_live_source_eligibility``.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any


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


def _finite(*vals: float) -> bool:
    for v in vals:
        try:
            if not math.isfinite(float(v)):
                return False
        except Exception:
            return False
    return True


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
        out.reason = "checkpoint_stale_or_incomplete"
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
    population_frame_complete: bool | None,
) -> tuple[bool | None, bool | None, bool | None, str]:
    """Map peer meta into (commit_ok, lineage_ok, freshness_ok, reason).

    Missing inputs → corresponding ``None`` so the common gate fail-closes.
    """
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
    if population_frame_complete is None:
        return True, True, None, "frame_complete_unknown"
    if not population_frame_complete:
        return True, True, False, "population_frame_incomplete"
    return True, True, True, "ok"
