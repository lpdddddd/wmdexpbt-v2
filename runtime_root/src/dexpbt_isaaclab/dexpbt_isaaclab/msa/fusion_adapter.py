"""Adapter: MSA sidecar cycle summary → fusion per-worker commands.

Preserves pilot statistics (L_mid, retention, selective promotion lists).
Does not recompute bootstrap / LCB; only maps sidecar fields into the
fusion command schema for trainer-side ADVANCE/ROLLBACK + ack.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Optional

from dexpbt_isaaclab.msa.fusion_events import emit_event
from dexpbt_isaaclab.msa.fusion_protocol import (
    WorkerCommand,
    claim_cycle_result,
    evidence_digest,
    mark_cycle_complete,
    write_journal,
    write_worker_command,
)


def _as_float_map(raw: Any) -> dict[int, float]:
    if not isinstance(raw, dict):
        return {}
    out: dict[int, float] = {}
    for k, v in raw.items():
        try:
            out[int(k)] = float(v)
        except (TypeError, ValueError):
            continue
    return out


def _as_int_list(raw: Any) -> list[int]:
    if not raw:
        return []
    return [int(x) for x in raw]


def sidecar_summary_to_commands(
    summary: dict[str, Any],
    *,
    run_dir: Path | str,
    source_path: str = "",
    anchor_checkpoint_by_worker: Optional[dict[int, str]] = None,
    post_checkpoint_by_worker: Optional[dict[int, str]] = None,
    fingerprint_by_worker: Optional[dict[int, str]] = None,
) -> tuple[bool, list[WorkerCommand]]:
    """Claim cycle once and write per-worker commands.

    Returns (claimed_by_us, commands_written).
    """
    run_dir = Path(run_dir)
    cycle_id = str(summary.get("intervention_id") or "")
    if not cycle_id:
        raise ValueError("sidecar summary missing intervention_id")

    claimed, _ = claim_cycle_result(
        run_dir, cycle_id, source_path=source_path or "", payload=summary,
    )
    if not claimed:
        emit_event(run_dir, "MSA_DECISION_CLAIMED", {
            "cycle_id": cycle_id, "claimed": False, "reason": "already_claimed",
        })
        return False, []

    emit_event(run_dir, "MSA_DECISION_CLAIMED", {
        "cycle_id": cycle_id, "claimed": True, "source_path": source_path,
    })

    due = _as_int_list(summary.get("due_workers"))
    selected = set(_as_int_list(summary.get("selected_for_promotion")))
    # Fallback: if selective lists absent, treat promoted flag as all-or-none.
    if not selected and summary.get("promoted"):
        selected = set(due)
    tau = _as_float_map(summary.get("tau_by_worker"))
    tau_plus = _as_float_map(summary.get("tau_plus_by_worker"))
    final_tol = _as_float_map(summary.get("final_tol_by_worker"))
    l_mid = summary.get("l_mid_by_worker") or {}
    retention = summary.get("retention_by_worker") or {}
    reasons = summary.get("rejection_reason_by_worker") or {}
    eligible = summary.get("eligible_by_worker") or {}
    anchors = _as_int_list(summary.get("anchor_workers"))
    anchor_ck = dict(anchor_checkpoint_by_worker or {})
    post_ck = dict(post_checkpoint_by_worker or {})
    fps = dict(fingerprint_by_worker or {})

    digest = evidence_digest({
        "cycle_id": cycle_id,
        "selected": sorted(selected),
        "l_mid": l_mid,
        "retention": retention,
        "final_tol": final_tol,
    })
    txn = f"txn_{cycle_id}"
    commands: list[WorkerCommand] = []
    for w in due:
        advance = w in selected
        action = "advance" if advance else "rollback"
        metrics = {
            "l_mid": l_mid.get(str(w), l_mid.get(w)),
            "retention": retention.get(str(w), retention.get(w)),
            "eligible": eligible.get(str(w), eligible.get(w)),
            "rejection_reason": reasons.get(str(w), reasons.get(w, "")),
        }
        # Flatten retention object if present.
        ret = metrics.get("retention")
        if isinstance(ret, dict):
            metrics["rho"] = ret.get("rho")
            metrics["rho_available"] = not bool(ret.get("retention_unavailable"))
            metrics["l_lcb"] = None
        # Prefer L_mid CI lower if embedded in panel-style dicts.
        lm = metrics.get("l_mid")
        if isinstance(lm, dict):
            metrics["l_lcb"] = lm.get("lower_quantile", lm.get("l_lcb"))
            metrics["l_mid"] = lm.get("point", lm.get("l_mid", lm))

        tgt = float(final_tol.get(w, tau_plus.get(w, tau.get(w, 0.0))))
        if not advance:
            tgt = float(tau.get(w, 0.0))
        cmd = WorkerCommand(
            cycle_id=cycle_id,
            transaction_id=txn,
            worker_id=int(w),
            action=action,
            expected_mode="PROBE",
            source_fingerprint=str(fps.get(w, "")),
            # During PROBE, current/committed tolerance is still tau.
            expected_tolerance=float(tau.get(w, 0.0)),
            target_tolerance=tgt,
            anchor_tolerance=float(tau.get(w, 0.0)),
            evidence_digest=digest,
            anchor_checkpoint=str(anchor_ck.get(w, "")),
            post_probe_checkpoint=str(post_ck.get(w, "")),
            metrics=metrics,
            reason=str(
                reasons.get(str(w), reasons.get(w, ""))
                or ("selective_promote" if advance else "selective_hold")
            ),
            production=True,
            test_only=False,
            issued_at=time.time(),
        )
        write_worker_command(run_dir, cmd)
        emit_event(run_dir, "MSA_COMMAND_WRITTEN", {
            "cycle_id": cycle_id, "worker_id": w, "action": action,
        })
        commands.append(cmd)

    write_journal(run_dir, cycle_id, {
        "cycle_id": cycle_id,
        "transaction_id": txn,
        "due_workers": due,
        "selected_for_promotion": sorted(selected),
        "anchor_workers": anchors,
        "evidence_digest": digest,
        "source_path": source_path,
        "commands": [c.to_dict() for c in commands],
        "created_at": time.time(),
    })
    return True, commands


def try_complete_cycle(run_dir: Path | str, cycle_id: str,
                       target_workers: list[int]) -> bool:
    ok, _ = mark_cycle_complete(run_dir, cycle_id, target_workers=target_workers)
    if ok:
        emit_event(run_dir, "MSA_CYCLE_COMPLETE", {
            "cycle_id": cycle_id, "workers": list(target_workers),
        })
    return ok


def load_sidecar_cycle_summary(path: Path | str) -> dict[str, Any]:
    import json
    return json.loads(Path(path).read_text())
