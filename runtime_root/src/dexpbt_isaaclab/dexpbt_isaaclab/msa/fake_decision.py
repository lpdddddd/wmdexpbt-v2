"""Test-only fake decision injector for disposable GPU smoke.

Formal profiles MUST keep:
  msa_fake_decision_injector_enable = False
  msa_inprocess_hook_enable = False

Enable only under an explicit test config / smoke launcher env guard.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from dexpbt_isaaclab.msa.fusion_hook import decision_path
from dexpbt_isaaclab.msa.worker_state import atomic_write_json

# Hard env gate — even if a misconfigured profile sets the flag, injector
# refuses to run unless this is exactly "1".
ENV_GUARD = "MSA_FUSION_ALLOW_FAKE_DECISION"


def injector_allowed(cfg_flag: bool) -> bool:
    return bool(cfg_flag) and os.environ.get(ENV_GUARD) == "1"


def write_fake_decisions(
    run_dir: Path | str,
    *,
    advance_workers: list[int],
    hold_workers: list[int],
    intervention_id: str = "fake_smoke_decision",
) -> Path:
    if os.environ.get(ENV_GUARD) != "1":
        raise RuntimeError(
            f"fake decision injector blocked: set {ENV_GUARD}=1 "
            "(test-only)")
    decisions: dict[str, Any] = {}
    for w in advance_workers:
        decisions[str(int(w))] = {
            "action": "advance",
            "l_lcb": 0.1,
            "rho": 1.0,
            "rho_available": True,
            "reason": "fake_injector_advance",
            "metrics": {"l_mid": 0.1, "l_lcb": 0.1, "rho": 1.0},
            "test_only": True,
        }
    for w in hold_workers:
        decisions[str(int(w))] = {
            "action": "hold",
            "l_lcb": -0.1,
            "rho": 0.5,
            "rho_available": True,
            "reason": "fake_injector_hold",
            "metrics": {"l_mid": -0.1, "l_lcb": -0.1, "rho": 0.5},
            "test_only": True,
        }
    doc = {
        "intervention_id": intervention_id,
        "source": "fake_decision_injector",
        "test_only": True,
        "decisions": decisions,
    }
    out = decision_path(Path(run_dir))
    atomic_write_json(out, doc)
    return out
