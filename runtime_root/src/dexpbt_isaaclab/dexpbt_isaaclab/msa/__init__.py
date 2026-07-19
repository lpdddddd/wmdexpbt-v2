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
- decision:    routing (top-k by criterion-local L_mid, stable tie-break;
               permuted-route control arm), same-tau retention rho, paired
               bootstrap panel statistic, promotion gates +
               first_blocking_gate.
- controller:  drives one probe cycle (heterogeneous per-worker tau /
               tau_plus, PARALLEL 20M-committed-tick exposure packets,
               phase-barrier holds around every eval, persisted +
               idempotently resumable cycles, crash cleanup).
- sidecar:     production wiring (committed-tick worker views, SIGSTOP/
               SIGCONT holds, locked msa_* control updates, durable
               tolerance commits synced into the resume snapshot,
               explicit eval GPU).
- continuation_guard: fail-closed bootstrap verification + first-tick
               tolerance/lineage validation + worker-local +100M hard
               frame cap for ALL four continuation arms.
- audit:       read-only 3B-5B first-blocker log audit.
- arms:        continuation arm configs (BASE_CONTINUATION / GATE_RELAX /
               MSA / MSA_PERMUTED_ROUTE) and dry-run launcher plan.

Design constraints honored throughout:
- tolerances are HETEROGENEOUS: every worker keeps its own tau[w] and
  tau_plus[w] = max(0.010, 0.95 * tau[w]); all evals / pins / rho are
  worker-local, and L_mid is a criterion-local improvement (the routing
  panel conditions on this, documented at every consumer);
- msa_* state lives inside CurriculumControlState (locked read-modify-write
  via update_curriculum_control) — single source of truth on the existing
  control / resume chain, no parallel checkpoint system;
- probe frames are charged to the worker's monotonic frame ledger as
  OBSERVED committed-tick frames (never a fabricated start + n);
- retention rho compares p_post/p_pre AT THE SAME worker-local tau;
  cross-tolerance historical-peak retention reads are structurally
  impossible here;
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
