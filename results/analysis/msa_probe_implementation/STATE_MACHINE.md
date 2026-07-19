# MSA Probe State Machine

Implemented in `dexpbt_isaaclab/msa/state_machine.py` as a single
enum-valued phase (`MSAPhase`). Probe phases are **never** expressed through
combinations of independent freeze flags; the machine is the only authority
on whether training is allowed and at which tolerance.

```
NORMAL
  └─> PRE_EVAL            snapshot_pre (immutable, step boundary)
                          p_pre = success(snapshot_pre, tau)
                          q_pre = success(snapshot_pre, tau_plus)
  └─> COMMON_PROBE        ALL due workers: 8M worker-local frames at tau_plus
                          (no checkpoint reload; no policy/optimizer/replay/
                           PBT-lineage/curriculum reset)
  └─> MID_EVAL            snapshot_mid; q_mid = success(snapshot_mid, tau_plus)
                          L_mid = q_mid - q_pre   (same tau_plus, same ordered
                          seed manifest — anything else fails closed)
  └─> ROUTE_SECOND_HALF   sort L_mid desc, tie = ascending worker_id;
                          top-3 get the second packet.
                          MSA_PERMUTED_ROUTE: shuffle L_mid→worker mapping
                          with pre-registered seed 777001 BEFORE sorting
                          (same total dose / recipients / eval count)
  └─> SECOND_PROBE        recipients only: second 8M at tau_plus
  └─> POST_EVAL           env tolerance reverts to tau (no checkpoint /
                          optimizer restore); snapshot_post;
                          p_post = success(snapshot_post, tau)
                          rho = p_post / p_pre  (same tau ONLY; requires
                          p_pre >= 0.01 else retention_unavailable)
                          q_post = success(snapshot_post, tau_plus) for
                          recipients only — unequal-dose follow-up, never a
                          common learning statistic
  └─> DECIDE              paired bootstrap per worker over the N pre/mid
                          tau_plus seed diffs → six-worker L_mid median →
                          5th percentile (PRNG seed 20260719, 10000 repeats).
                          Conditional panel statistic for THIS 6-checkpoint
                          panel, not a population CI.
        ├─> NORMAL        any gate failed; first_blocking_gate recorded
        └─> PROMOTE       ALL of:
                            1. panel L_mid 5th-pct > 0
                            2. every worker rho >= base-retention threshold
                               (0.88), same-tau only, never unavailable
                            3. remaining safety gates pass (gates that read
                               cross-tolerance historical retention are
                               rejected at API level)
              └─> NORMAL  tau_plus becomes the permanent tolerance
```

## Phase permissions

| phase | training allowed | env tolerance |
|---|---|---|
| NORMAL | yes | tau |
| PRE_EVAL / MID_EVAL / POST_EVAL / ROUTE_SECOND_HALF / DECIDE | no | (frozen) |
| COMMON_PROBE / SECOND_PROBE | yes | tau_plus |
| PROMOTE | yes | tau_plus (permanent) |

Illegal transitions raise `IllegalTransition`; a bug cannot silently skip
an evaluation or a decision. The machine serializes into the existing
curriculum control JSON (`msa_state_machine` field) and therefore rides the
same resume snapshot chain as the rest of the curriculum state.

## tau_plus and due rule

- `tau_plus = max(0.010, 0.95 * tau)` (meters).
- A non-floor worker is `probe_due` when it has **no** exposure at
  tolerance <= tau_plus within its most recent 100M worker-local frames.
- Resume from a checkpoint without provenance (any pre-MSA checkpoint):
  immediately due — no fresh 100M wait.
- At tau <= 0.010 there is no tighter sentinel: never due.
