"""Matched Sentinel Advancement (MSA) probe toolkit.

Diagnoses and (conditionally) unlocks the tolerance progression deadlock
observed at tau ~ 0.022-0.028 in the base-retention population, WITHOUT
relaxing all gates, disabling safety mechanisms, or comparing retention
across different tolerances.

Modules (pure python; no Isaac import anywhere in this package):
- provenance:  tolerance-tagged exposure provenance records + due logic
               (tau_plus = max(0.010, 0.95 * tau); probe_due if no tau_plus
               exposure in the last 100M worker-local frames; resume without
               provenance -> immediately due).
- state_machine: the explicit probe state machine
               NORMAL -> PRE_EVAL -> COMMON_PROBE -> MID_EVAL ->
               ROUTE_SECOND_HALF -> SECOND_PROBE -> POST_EVAL -> DECIDE ->
               NORMAL | PROMOTE (single enum; never a pile of freeze flags).
- eval_bridge: fixed-tolerance evaluation requests against the existing
               CCRS policy-only evaluator (immutable snapshot, subprocess,
               inference-only, ordered seed manifest, per-seed results).
- decision:    routing (top-k by L_mid, stable tie-break; permuted-route
               control arm), same-tau retention rho, paired bootstrap panel
               statistic, promotion gates + first_blocking_gate.
- audit:       read-only 3B-5B first-blocker log audit.
- arms:        continuation arm configs (BASE_CONTINUATION / GATE_RELAX /
               MSA / MSA_PERMUTED_ROUTE) and dry-run launcher plan.

Design constraints honored throughout:
- extends the existing curriculum resume snapshot chain (provenance rides
  inside the curriculum state dict) — no parallel checkpoint system;
- probe frames are charged to the worker's monotonic frame ledger (never
  free extra training);
- retention rho compares p_post/p_pre AT THE SAME tau; cross-tolerance
  historical-peak retention reads are structurally impossible here;
- q_post is an unequal-dose follow-up only, never part of the common-dose
  L_mid statistic.
"""

from dexpbt_isaaclab.msa.provenance import (  # noqa: F401
    EXPOSURE_WINDOW_FRAMES,
    TAU_FLOOR,
    ExposureSegment,
    InterventionRecord,
    MSAProvenance,
    probe_due,
    tau_plus_of,
)
from dexpbt_isaaclab.msa.state_machine import (  # noqa: F401
    MSAPhase,
    MSAProbeStateMachine,
)
