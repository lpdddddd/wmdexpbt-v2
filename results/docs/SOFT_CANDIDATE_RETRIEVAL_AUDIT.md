# Soft Candidate Retrieval Audit

## Summary

Additive soft tolerance gating for elite/archive/RSSM candidate retrieval in EIR `generate_eir_candidates`. Does not relax MPPI uncertainty gates or residual rollout apply.

## Gate modes

- `hard_tol`: legacy (weight=1.0 always)
- `soft_tol` / `profile_soft`: `soft_tol_weight(tol, tol_start, tol_full)` scales priority/similarity; skip retrieval when weight `< eir_candidate_min_weight_for_retrieval`

## Per-profile behaviour

| Profile | Elite tol | RSSM tol |
|---------|-----------|----------|
| rssm_mppi_actor_distill | — | 0.060 → 0.040 |
| elite_bank_actor_distill | 0.070 → 0.045 | — |
| conservative_rssm_planner_distill | — | 0.055 → 0.035 |

## Residual safety

`eir_residual_apply_to_rollout` remains OFF for all zoo profiles (`--no-eir_residual_apply_to_rollout` in launcher).

## Tests

- `tests/test_eir_soft_candidate_gate.py`
