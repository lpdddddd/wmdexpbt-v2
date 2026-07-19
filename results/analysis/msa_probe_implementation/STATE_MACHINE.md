# MSA Probe State Machine

Implemented in `dexpbt_isaaclab/msa/state_machine.py` as a single
enum-valued phase (`MSAPhase`). Probe phases are **never** expressed through
combinations of independent freeze flags; the machine is the only authority
on whether training is allowed and at which tolerance.

Tolerances are **heterogeneous**: every worker `w` keeps its own
`tau[w] = active_tau` and `tau_plus[w] = max(0.010, 0.95 * tau[w])`. All
evaluations, pins, provenance records and rho are worker-local; `L_mid[w]`
is a CRITERION-LOCAL improvement (measured at the worker's own tau_plus)
and the routing/panel statistics condition on this panel of
criterion-local improvements.

```
NORMAL
  └─> PRE_EVAL            [BARRIER] all due workers HELD at a committed
                          checkpoint; snapshot_pre (immutable, committed);
                          p_pre = success(snapshot_pre, tau[w])
                          q_pre = success(snapshot_pre, tau_plus[w])
                          then pin every worker to its OWN tau_plus[w]
  └─> COMMON_PROBE        PARALLEL: release all, one train_until barrier
                          waits for every worker's committed ledger to pass
                          start[w] + 20M (one committed PBT tick).
                          Frames recorded = OBSERVED committed frames.
                          (no checkpoint reload; no policy/optimizer/replay/
                           PBT-lineage/curriculum reset)
  └─> MID_EVAL            [BARRIER] held; snapshot_mid;
                          q_mid = success(snapshot_mid, tau_plus[w])
                          L_mid[w] = q_mid - q_pre  (same tau_plus[w], same
                          ordered seed manifest — anything else fails closed)
  └─> ROUTE_SECOND_HALF   sort L_mid desc, tie = ascending worker_id;
                          top-3 get the second packet.
                          non-recipients re-pinned to their OWN tau[w]
                          IMMEDIATELY (durably committed after release).
                          MSA_PERMUTED_ROUTE: shuffle L_mid→worker mapping
                          with pre-registered seed 777001 BEFORE sorting
                          (same total dose / recipients / eval count)
  └─> SECOND_PROBE        PARALLEL: everyone released (non-recipients train
                          at their tau); barrier waits on the recipients'
                          second 20M committed tick at tau_plus[w]
  └─> POST_EVAL           [BARRIER] held; every pin reverted to tau[w]
                          (no checkpoint / optimizer restore); snapshot_post;
                          p_post = success(snapshot_post, tau[w])
                          rho[w] = p_post / p_pre  (same tau[w] ONLY;
                          requires p_pre >= 0.01 else retention_unavailable)
                          q_post = success(snapshot_post, tau_plus[w]) for
                          recipients only — unequal-dose follow-up, never a
                          common learning statistic
  └─> DECIDE              paired bootstrap per worker over the N pre/mid
                          tau_plus[w] seed diffs → six-worker L_mid median →
                          5th percentile (PRNG seed 20260719, 10000 repeats).
                          CRITERION-LOCAL conditional panel statistic for
                          THIS 6-checkpoint panel, not a population CI.
        ├─> NORMAL        any gate failed; first_blocking_gate recorded;
                          every worker durably committed back to tau[w]
        └─> PROMOTE       ALL of:
                            1. panel L_mid 5th-pct > 0
                            2. every worker rho >= base-retention threshold
                               (0.88), same-tau only, never unavailable
                            3. remaining safety gates pass (gates that read
                               cross-tolerance historical retention are
                               rejected at API level)
              └─> NORMAL  worker-local tau_plus[w] durably committed:
                          success_tolerance = tau_plus[w], resume snapshot
                          synced, pin cleared (survives restart)
```

## Phase permissions

| phase | training allowed | env tolerance |
|---|---|---|
| NORMAL | yes | tau[w] |
| PRE_EVAL / MID_EVAL / POST_EVAL / ROUTE_SECOND_HALF / DECIDE | no (workers HELD) | (pinned, frozen) |
| COMMON_PROBE | yes (all due, parallel) | tau_plus[w] |
| SECOND_PROBE | yes (all; recipients at tau_plus[w], others at tau[w]) | mixed, worker-local |
| PROMOTE | yes | tau_plus[w] (durably committed) |

Illegal transitions raise `IllegalTransition`; a bug cannot silently skip
an evaluation or a decision. `abort(reason)` is the crash-cleanup escape
hatch (any phase → NORMAL, recorded as ABORT in the history) and is used
ONLY after the controller restored every worker's own tau and cleared all
holds/pins.

## Persistence, resume, crash

- The machine, the provenance and the in-flight cycle context serialize
  into the existing curriculum control JSON (`msa_state_machine`,
  `msa_provenance`, `msa_cycle`) at EVERY phase transition, through the
  locked `update_curriculum_control` API — single source of truth on the
  same resume-snapshot chain as the rest of the curriculum state.
- Sidecar restart: a persisted non-terminal cycle is resumed idempotently
  (same intervention_id; completed exposure segments are skipped via
  provenance, snapshots and ccrs_eval raw results are reused; barriers on
  already-reached frames return immediately).
- Any exception: non-promoted workers are pinned back to tau[w]
  synchronously, holds released, tolerance durably committed, cycle marked
  FAILED with the failing phase, machine ABORTed to NORMAL. A FAILED cycle
  is never auto-resumed.

## tau_plus and due rule

- `tau_plus[w] = max(0.010, 0.95 * tau[w])` (meters, worker-local).
- A non-floor worker is `probe_due` when it has **no** exposure at
  tolerance <= tau_plus[w] within its most recent 100M worker-local frames.
- Resume from a checkpoint without provenance (any pre-MSA checkpoint):
  immediately due — no fresh 100M wait.
- At tau <= 0.010 there is no tighter sentinel: never due.
