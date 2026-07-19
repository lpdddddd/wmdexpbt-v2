# MSA Probe Dry-Run Trace (production-style)

- scenario: 6 workers with HETEROGENEOUS live tolerances (real 5B-tip values), tick-quantized parallel trainer (80,000,000 frames per committed tick), phase-barrier holds, N=16 pilot seeds (formal N=256 explicit only)
- intervention_id: `msa_1784465377_032f8891`
- arm: MSA

## Worker-local tolerances

| worker | tau (m) | tau_plus (m) |
|---|---|---|
| p0 | 0.0231124875097 | 0.0219568631342 |
| p1 | 0.0224754709524 | 0.0213516974048 |
| p2 | 0.0268315658866 | 0.0254899875923 |
| p3 | 0.0249035689223 | 0.0236583904762 |
| p4 | 0.0262142830761 | 0.0249035689223 |
| p5 | 0.0275939821854 | 0.0262142830761 |

## State machine transitions

1. `NORMAL->PRE_EVAL [msa_1784465377_032f8891]`
1. `PRE_EVAL->COMMON_PROBE`
1. `COMMON_PROBE->MID_EVAL`
1. `MID_EVAL->ROUTE_SECOND_HALF`
1. `ROUTE_SECOND_HALF->SECOND_PROBE`
1. `SECOND_PROBE->POST_EVAL`
1. `POST_EVAL->DECIDE`
1. `DECIDE->PROMOTE [all gates passed]`
1. `PROMOTE->NORMAL [promotion applied]`

## Common probe (parallel, one 20M committed tick each)

| worker | L_mid (criterion-local) | tau_plus frames |
|---|---|---|
| p0 | +0.0600 | 160,000,000 |
| p1 | +0.1000 | 160,000,000 |
| p2 | +0.0200 | 80,000,000 |
| p3 | +0.0800 | 160,000,000 |
| p4 | +0.0400 | 80,000,000 |
| p5 | +0.0600 | 80,000,000 |

## Second-half routing

- order (L_mid desc, tie=worker_id asc): [1, 3, 0, 5, 4, 2]
- recipients (top-3): [1, 3, 0]
- permutation_seed: None

## Retention (worker-local tau) and decision

- panel L_mid 5th-pct (criterion-local conditional panel): +0.0600
- promoted: True
- first_blocking_gate: none

| worker | tau | p_pre | p_post | rho | unavailable |
|---|---|---|---|---|---|
| p0 | 0.0231124875097 | 0.550 | 0.550 | 1.000 | False |
| p1 | 0.0224754709524 | 0.550 | 0.550 | 1.000 | False |
| p2 | 0.0268315658866 | 0.550 | 0.550 | 1.000 | False |
| p3 | 0.0249035689223 | 0.550 | 0.550 | 1.000 | False |
| p4 | 0.0262142830761 | 0.550 | 0.550 | 1.000 | False |
| p5 | 0.0275939821854 | 0.550 | 0.550 | 1.000 | False |

## Invariants observed

- reset calls (policy/optimizer/replay/lineage/curriculum): 0 (must be 0)
- durable tolerance commits (pin cleared): [(0, 0.0219568631342), (1, 0.0213516974048), (2, 0.0254899875923), (3, 0.0236583904762), (4, 0.0249035689223), (5, 0.0262142830761)]
- residual pins after cycle: {} (must be empty)
- total tau_plus dose: 720,000,000 frames (= 6 x 80,000,000 common + 3 x 80,000,000 second half)
- frames never advanced while a worker was held (asserted in FakeWorld.snapshot/evaluate/train_until).
- q_post evaluated for recipients only; never entered L_mid.

## Readiness hold / handoff dry run (REAL processes, 2026-07-19)

Six live worker processes, the guard's TRUE readiness barrier
(per-worker SIGSTOP freeze at the validated committed boundary) and the
sidecar's hold adoption were exercised end to end (`guard_loop` with
`arm=MSA_STATEFUL`, then `adopt_readiness_holds`, then the
controller-style release):

```
=== Stage A: guard readiness barrier (arm=MSA_STATEFUL) ===
[guard] FIRST_TICK_VALIDATED.json written (6/6 workers validated)
[guard] MSA arm: readiness holds retained for sidecar adoption (READINESS_HOLDS.json -> MSA_HOLDS.json)
guard_loop rc=0
  worker 0: pids=[634930] state=['T'] validated_frame=5080000000 tol=0.023112 lineage=lin0
  worker 1: pids=[634931] state=['T'] validated_frame=5080000000 tol=0.023112 lineage=lin1
  worker 2: pids=[634932] state=['T'] validated_frame=5080000000 tol=0.023112 lineage=lin2
  worker 3: pids=[634933] state=['T'] validated_frame=5080000000 tol=0.023112 lineage=lin3
  worker 4: pids=[634934] state=['T'] validated_frame=5080000000 tol=0.023112 lineage=lin4
  worker 5: pids=[634935] state=['T'] validated_frame=5080000000 tol=0.023112 lineage=lin5
  FIRST_TICK_VALIDATED.json: [0, 1, 2, 3, 4, 5] workers, frames=[5080000000, 5080000000, 5080000000, 5080000000, 5080000000, 5080000000]
  READINESS_HOLDS.json retained: True
=== Stage B: sidecar adoption (never releases in between) ===
  adopted workers: [0, 1, 2, 3, 4, 5]
  READINESS_HOLDS.json gone: True
  MSA_HOLDS worker 0: pids=[634930] state=['T']
  MSA_HOLDS worker 1: pids=[634931] state=['T']
  MSA_HOLDS worker 2: pids=[634932] state=['T']
  MSA_HOLDS worker 3: pids=[634933] state=['T']
  MSA_HOLDS worker 4: pids=[634934] state=['T']
  MSA_HOLDS worker 5: pids=[634935] state=['T']
  PRE snapshot frame == validated frame for all 6: VERIFIED
=== Stage C: controller release at COMMON_PROBE ===
  post-release states: {0: 'S', 1: 'S', 2: 'S', 3: 'S', 4: 'S', 5: 'S'}
dry run complete
```

Observed invariants:

- every validated worker is in `T` (stopped) state the moment its record
  lands in `READINESS_HOLDS.json`; nobody validated is left running;
- `FIRST_TICK_VALIDATED.json` appears only at 6/6 and the holds are
  RETAINED for the MSA arm (BASE/GATE arms release all six
  simultaneously instead);
- adoption converts `READINESS_HOLDS.json` -> `MSA_HOLDS.json` with the
  SAME pids, all still in `T` — the workers are never released between
  validation and the first PRE snapshot;
- the newest committed frame of every worker equals the validated frame
  recorded in the marker (`PRE snapshot frame == validated frame`);
- the controller-style release returns all six to running (`S`) state.
