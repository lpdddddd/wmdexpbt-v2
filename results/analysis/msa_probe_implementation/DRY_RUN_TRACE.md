# MSA Probe Dry-Run Trace

- scenario: 6 workers deadlocked at tau=0.0235 m (tau_plus=0.022325 m), fake deterministic evaluator/trainer, N=16 seeds (formal N=256)
- intervention_id: `msa_1784433072_8cb7208b`
- arm: MSA

## State machine transitions

1. `NORMAL->PRE_EVAL [msa_1784433072_8cb7208b]`
1. `PRE_EVAL->COMMON_PROBE`
1. `COMMON_PROBE->MID_EVAL`
1. `MID_EVAL->ROUTE_SECOND_HALF`
1. `ROUTE_SECOND_HALF->SECOND_PROBE`
1. `SECOND_PROBE->POST_EVAL`
1. `POST_EVAL->DECIDE`
1. `DECIDE->PROMOTE [all gates passed]`
1. `PROMOTE->NORMAL [promotion applied]`

## Common probe (first half, all due workers, 8M frames each)

| worker | L_mid | frames trained |
|---|---|---|
| p0 | +0.0600 | 16,000,000 |
| p1 | +0.1000 | 16,000,000 |
| p2 | +0.0200 | 8,000,000 |
| p3 | +0.0800 | 16,000,000 |
| p4 | +0.0400 | 8,000,000 |
| p5 | +0.0600 | 8,000,000 |

## Second-half routing

- order (L_mid desc, tie=worker_id asc): [1, 3, 0, 5, 4, 2]
- recipients (top-3): [1, 3, 0]
- permutation_seed: None

## Retention (same tau) and decision

- panel L_mid 5th-pct (bootstrap, fixed seed): +0.0600
- promoted: True
- first_blocking_gate: none

| worker | p_pre | p_post | rho | unavailable |
|---|---|---|---|---|
| p0 | 0.550 | 0.550 | 1.000 | False |
| p1 | 0.550 | 0.550 | 1.000 | False |
| p2 | 0.550 | 0.550 | 1.000 | False |
| p3 | 0.550 | 0.550 | 1.000 | False |
| p4 | 0.550 | 0.550 | 1.000 | False |
| p5 | 0.550 | 0.550 | 1.000 | False |

## Invariants observed

- reset calls (policy/optimizer/replay/lineage/curriculum): 0 (must be 0)
- forced tolerance writes: [(0, 0.022324999999999998), (1, 0.022324999999999998), (2, 0.022324999999999998), (3, 0.022324999999999998), (4, 0.022324999999999998), (5, 0.022324999999999998), (0, 0.0235), (1, 0.0235), (2, 0.0235), (3, 0.0235), (4, 0.0235), (5, 0.0235), (0, 0.022324999999999998), (1, 0.022324999999999998), (2, 0.022324999999999998), (3, 0.022324999999999998), (4, 0.022324999999999998), (5, 0.022324999999999998)]
- total probe frames charged to ledgers: 72,000,000 (= 6x8M common + 3x8M second half)
- q_post evaluated for recipients only; never entered L_mid.
