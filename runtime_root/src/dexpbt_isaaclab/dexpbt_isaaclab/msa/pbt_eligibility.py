"""Stage-aware PBT eligibility for MSA-aware DexPBT.

All donor / recipient inheritance rules live HERE so they are not
scattered across observer call sites. See module docstring comments on
each public function.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

from dexpbt_isaaclab.msa.fusion_config import MSAFusionConfig
from dexpbt_isaaclab.msa.worker_state import (
    MODE_BUILD,
    MODE_HOLD,
    MODE_PROBE,
    WorkerMSAState,
    clear_evidence_on_pbt_copy,
)

# Stage adjacency: BUILD <-> MID <-> STRICT
_STAGE_ORDER = ("BUILD", "MID", "STRICT")


def _stage_index(stage: str) -> int:
    try:
        return _STAGE_ORDER.index(stage)
    except ValueError:
        return 0


def stages_adjacent_or_same(a: str, b: str) -> bool:
    return abs(_stage_index(a) - _stage_index(b)) <= 1


@dataclass
class DonorDecision:
    eligible: bool
    reason: str


def donor_eligible(
    donor: WorkerMSAState,
    *,
    recipient_stage: str,
    population: Sequence[WorkerMSAState],
    cfg: MSAFusionConfig,
    allow_adjacent: bool = True,
) -> DonorDecision:
    """Donor eligibility (fusion profile).

    Rules:
    - donor must be committed (not mid-PROBE transaction);
    - PROBE temporary policies cannot be donors;
    - prefer same stage; default allow same or adjacent stage only;
    - population must retain >= msa_min_anchor_workers reliable anchors
      after considering that an anchor donor is still an anchor.
    """
    if donor.mode == MODE_PROBE or donor.transaction_open:
        return DonorDecision(False, "donor_in_probe")
    if not donor.anchor_checkpoint and donor.anchor_tolerance <= 0:
        return DonorDecision(False, "donor_missing_anchor")
    donor_stage = donor.committed_stage or cfg.stage_for_tolerance(
        donor.current_tolerance)
    if donor_stage != recipient_stage:
        if not allow_adjacent or not stages_adjacent_or_same(
                donor_stage, recipient_stage):
            return DonorDecision(
                False,
                f"stage_mismatch donor={donor_stage} "
                f"recipient={recipient_stage}")
    # Count reliable anchors (BUILD/HOLD with anchor, not open probe).
    anchors = [
        w for w in population
        if w.mode != MODE_PROBE
        and not w.transaction_open
        and (w.anchor_tolerance > 0 or bool(w.anchor_checkpoint))
    ]
    if len(anchors) < int(cfg.msa_min_anchor_workers):
        return DonorDecision(False, "insufficient_population_anchors")
    return DonorDecision(True, "ok")


def recipient_after_pbt_copy(
    recipient: WorkerMSAState,
    donor: WorkerMSAState,
    *,
    cfg: MSAFusionConfig,
) -> WorkerMSAState:
    """Apply PBT copy inheritance rules to recipient MSA state.

    - Does NOT inherit donor pass evidence / cycle id / CIs / consecutive_passes.
    - Clears recipient old evidence.
    - Keeps recipient worker_id.
    - If donor/recipient tolerances differ, recipient cannot keep a stricter
      committed stage: drop to BUILD needing re-validation (or HOLD).
    - Re-establish a consistent local anchor at the post-copy tolerance.
    """
    rid = int(recipient.worker_id)
    clear_evidence_on_pbt_copy(recipient)
    recipient.worker_id = rid
    # Post-copy live tolerance follows training resume semantics (caller
    # sets current_tolerance); here we only fix MSA bookkeeping.
    if abs(float(donor.current_tolerance) - float(recipient.current_tolerance)) > 1e-12:
        recipient.committed_stage = cfg.stage_for_tolerance(
            recipient.current_tolerance)
        recipient.mode = MODE_BUILD
        recipient.decision_reason = "pbt_copy_tol_mismatch_revalidate"
    else:
        recipient.committed_stage = cfg.stage_for_tolerance(
            recipient.current_tolerance)
        recipient.mode = MODE_BUILD
        recipient.decision_reason = "pbt_copy_same_tol_rebuild_anchor"
    # Anchor = post-copy local state (not donor's MSA evidence).
    recipient.anchor_tolerance = float(recipient.current_tolerance)
    recipient.consecutive_passes = 0
    recipient.pass_evidence = {}
    return recipient


def can_be_exploit_target(state: WorkerMSAState) -> bool:
    """PROBE workers must not be overwritten by PBT exploit."""
    return state.mode != MODE_PROBE and not state.transaction_open


def can_be_exploit_source(state: WorkerMSAState) -> bool:
    return state.mode != MODE_PROBE and not state.transaction_open


def count_reliable_anchors(population: Sequence[WorkerMSAState]) -> int:
    return sum(
        1 for w in population
        if w.mode != MODE_PROBE
        and not w.transaction_open
        and (w.anchor_tolerance > 0 or bool(w.anchor_checkpoint))
    )


def population_has_min_anchors(
    population: Sequence[WorkerMSAState],
    cfg: MSAFusionConfig,
) -> bool:
    return count_reliable_anchors(population) >= int(cfg.msa_min_anchor_workers)
