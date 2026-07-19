# MSA Probe Dry-Run Trace (production-style)

- scenario: 6 workers with HETEROGENEOUS live tolerances (real 5B-tip values), tick-quantized parallel trainer (20,000,000 frames per committed tick), phase-barrier holds, N=16 pilot seeds (formal N=256 explicit only)
- intervention_id: `msa_1784447278_6058fb05`
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

1. `NORMAL->PRE_EVAL [msa_1784447278_6058fb05]`
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
| p0 | +0.0600 | 40,000,000 |
| p1 | +0.1000 | 40,000,000 |
| p2 | +0.0200 | 20,000,000 |
| p3 | +0.0800 | 40,000,000 |
| p4 | +0.0400 | 20,000,000 |
| p5 | +0.0600 | 20,000,000 |

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
- total tau_plus dose: 180,000,000 frames (= 6 x 20,000,000 common + 3 x 20,000,000 second half)
- frames never advanced while a worker was held (asserted in FakeWorld.snapshot/evaluate/train_until).
- q_post evaluated for recipients only; never entered L_mid.
