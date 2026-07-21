"""Test-only episode-level eval fixture → real selective decision → fusion cmds.

Formal profiles cannot enable this. Requires:
  MSA_FUSION_ALLOW_EVAL_FIXTURE=1
and msa_inprocess_hook_enable on a test config.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from dexpbt_isaaclab.msa.decision import (
    RetentionResult,
    decide_selective_promotion,
)
from dexpbt_isaaclab.msa.fusion_adapter import sidecar_summary_to_commands
from dexpbt_isaaclab.msa.fusion_events import emit_event

ENV_FIXTURE = "MSA_FUSION_ALLOW_EVAL_FIXTURE"


def fixture_allowed(cfg_flag: bool) -> bool:
    return bool(cfg_flag) and os.environ.get(ENV_FIXTURE) == "1"


def build_fixture_summary(
    *,
    cycle_id: str,
    advance_workers: list[int],
    hold_workers: list[int],
    tau: float = 0.02,
    tau_plus: float = 0.019,
    rho_threshold: float = 0.88,
) -> dict[str, Any]:
    """Build a sidecar-shaped summary using real decide_selective_promotion."""
    due = sorted(set(advance_workers) | set(hold_workers))
    paired: dict[int, list[float]] = {}
    retention: dict[int, RetentionResult] = {}
    for w in due:
        if w in advance_workers:
            # Positive paired diffs → L_mid LCB > 0
            paired[w] = [0.05] * 16
            retention[w] = RetentionResult(
                tau=float(tau), rho=0.95, p_pre=0.5, p_post=0.48,
                retention_unavailable=False,
            )
        else:
            paired[w] = [-0.05] * 16
            retention[w] = RetentionResult(
                tau=float(tau), rho=0.5, p_pre=0.5, p_post=0.25,
                retention_unavailable=False,
            )
    tau_by = {w: float(tau) for w in due}
    tau_plus_by = {w: float(tau_plus) for w in due}
    sel = decide_selective_promotion(
        paired,
        retention,
        tau_by,
        tau_plus_by,
        rho_threshold,
        population_worker_ids=list(range(6)),
    )
    return {
        "intervention_id": cycle_id,
        "due_workers": due,
        "selected_for_promotion": list(sel.selected_for_promotion),
        "anchor_workers": list(sel.anchor_workers),
        "eligible_by_worker": {
            str(k): bool(v) for k, v in sel.eligible_by_worker.items()
        },
        "rejection_reason_by_worker": {
            str(k): str(v) for k, v in sel.rejection_reason_by_worker.items()
        },
        "tau_by_worker": {str(k): v for k, v in tau_by.items()},
        "tau_plus_by_worker": {str(k): v for k, v in tau_plus_by.items()},
        "final_tol_by_worker": {
            str(k): v for k, v in sel.final_tol_by_worker.items()
        },
        "promoted": bool(sel.promote_any),
        "l_mid_by_worker": {
            str(k): v for k, v in sel.l_mid_ci_by_worker.items()
        },
        "retention_by_worker": {
            str(w): {
                "rho": retention[w].rho,
                "p_pre": retention[w].p_pre,
                "p_post": retention[w].p_post,
                "retention_unavailable": retention[w].retention_unavailable,
            }
            for w in due
        },
        "fixture": True,
        "test_only": True,
    }


def publish_fixture_cycle(
    run_dir: Path | str,
    *,
    cycle_id: str,
    advance_workers: list[int],
    hold_workers: list[int],
    cfg_flag: bool,
) -> dict[str, Any]:
    if not fixture_allowed(cfg_flag):
        raise RuntimeError(
            f"eval fixture blocked: set {ENV_FIXTURE}=1 and test cfg flag"
        )
    summary = build_fixture_summary(
        cycle_id=cycle_id,
        advance_workers=advance_workers,
        hold_workers=hold_workers,
    )
    out = Path(run_dir) / "dexpbt_8gpu_ws" / "msa_cycles"
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{cycle_id}.json"
    import json
    from dexpbt_isaaclab.msa.worker_state import atomic_write_json
    atomic_write_json(path, summary)
    emit_event(run_dir, "MSA_EVAL_COMPLETE", {
        "cycle_id": cycle_id, "source": "eval_fixture", "test_only": True,
    })
    claimed, cmds = sidecar_summary_to_commands(
        summary, run_dir=run_dir, source_path=str(path),
    )
    return {
        "summary": summary,
        "claimed": claimed,
        "commands": [c.to_dict() for c in cmds],
        "path": str(path),
    }
