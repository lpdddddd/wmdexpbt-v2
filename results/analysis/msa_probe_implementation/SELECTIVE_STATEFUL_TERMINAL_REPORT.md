# MSA_SELECTIVE_STATEFUL terminal report (N=4 smoke + N=16 pilot)

Date: 2026-07-21 (UTC+8)
Branch: `base-retention`
Code under test: `8863b64901faa2673a36eea1252cd9ba3f4f41d1`
("Close MSA completion and failure supervision races")

No training algorithm thresholds were changed for these runs
(ρ=0.88, τ⁺=0.95τ, top-3 selective, 480M cap, frozen-frame budget).

## Verdict

**PASSED end-to-end** on GPUs 0–5:

1. Disposable hold/coexistence smoke → certificate matched `8863b64`
2. `MSA_SELECTIVE_STATEFUL` N=4 execution smoke → `ARM_COMPLETE`
3. `MSA_SELECTIVE_STATEFUL` N=16 one-step selective pilot → `ARM_COMPLETE`

No `GUARD_FAILED`, `MSA_CYCLE_FAILED.json`, `COMMIT_TIMEOUT.json`,
`MSA_SIDECAR_DIED.json`, or `STATE_FIDELITY_FAILED.json` on either arm.

## Hold smoke

- Run: `results/runs/msa_hold_smoke_disposable_20260720_230351`
- Certificate: `results/launchers/msa_probe/HOLD_SMOKE_CERTIFICATE.json`
- `ok=true`, `eval_coexistence_ok=true`
- `git_commit=8863b64901faa2673a36eea1252cd9ba3f4f41d1`
- task / profile / GPU (RTX 5090 index 4, driver 595.71.05) matched
- frames: hold at 5.08B → resume next tick 5.16B; eval ~144.6 s, exit 0
- Processes and GPU released cleanly after smoke

NOTE: the certificate binds to code commit `8863b64`. Archiving it in
git does not re-validate future HEADs; a new disposable smoke is
required before launching arms from any later commit.

## N=4 execution smoke

- Run: `results/runs/base_retention_msa_probe_msa_selective_stateful_20260721_002716`
- `MSA_NUM_SEEDS=4`, `DEXPBT_GPUS=0,1,2,3,4,5`, eval `cuda:4,cuda:5`
- 6/6 readiness + stall reproduction; sidecar adopted readiness holds
- One admitted cycle: `msa_1784568017_a355115e`
  - `promoted=true`, `selected_for_promotion=[1, 3]`, anchors `[0, 2, 4, 5]`
  - observed dose 240M all workers; final frames ≤ reserved ceilings
  - reject: w0/w4/w5 `L_mid_lower_quantile<=0`, w2 `same_tau_retention`
- Then `all_budget_ineligible` until park
- `ARM_COMPLETE` (~06:48): all six at worker-local +480M targets
- Full process teardown; GPUs freed before N=16

## N=16 one-step selective pilot

- Run: `results/runs/base_retention_msa_probe_msa_selective_stateful_20260721_064933`
- `MSA_NUM_SEEDS=16`, same GPU / eval layout
- 6/6 readiness + adopt; one admitted cycle: `msa_1784591014_9ddc9843`
  - `promoted=true`, `selected_for_promotion=[3, 1, 2]`, anchors `[0, 4, 5]`
  - observed dose 160M all workers; final frames ≤ reserved ceilings
  - reject: w0/w4/w5 `L_mid_lower_quantile<=0`
- Then budget-ineligible idle → park
- `ARM_COMPLETE` (~18:33): all six at +480M targets; clean exit

This pilot admits at most one new MSA cycle under the 480M continuation
cap after ~80M readiness. It does **not** claim iterative completion of
0.02→0.01.

## Cycle JSON paths

- N=4: `.../20260721_002716/dexpbt_8gpu_ws/msa_cycles/msa_1784568017_a355115e.json`
- N=16: `.../20260721_064933/dexpbt_8gpu_ws/msa_cycles/msa_1784591014_9ddc9843.json`

Run directories themselves are not committed (checkpoints / logs).
