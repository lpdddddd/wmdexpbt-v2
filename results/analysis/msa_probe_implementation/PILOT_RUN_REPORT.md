# MSA_STATEFUL N=16 formal pilot — run report

Date: 2026-07-20
Branch: `base-retention`
Code commit: `bf4d8b3` ("Propagate source lineage through stateful bootstrap")
Run dir: `results/runs/base_retention_msa_probe_msa_stateful_20260720_033431`

## Preconditions (all passed)

- Disposable GPU certificate smoke re-run after `bf4d8b3` (lineage
  resume invalidates any prior certificate): `ok=true`,
  `eval_coexistence_ok=true`, `eval_timed_out=false`,
  `eval_survivor_pids=[]`, `git_commit` matched HEAD.
- Launcher: `MSA_NUM_SEEDS=16`, eval devices `cuda:4,cuda:5`.
- Bootstrap: `STATE_FIDELITY.json` (`state_faithful=True`), fingerprint
  OK, `continuation_guard --verify-only` 6/6, sidecar validated the
  certificate, `--lineage-resume-per-policy` present on every worker.

## Milestone: first-tick gate

`FIRST_TICK_VALIDATED.json` written with 6/6 workers:
- each `validated_tolerance == source_tolerance` (exact),
- each `lineage_id` matched the manifest (closes the prior
  `GUARD_FAILED` on fresh-uuid lineage),
- `stall_reproduced=true` for all six.
- Sidecar adopted readiness holds → `MSA_HOLDS.json`; first PRE_EVAL
  started from the frozen validated boundary.

## MSA cycles completed

### Cycle 1 — `msa_1784492852_5fa1236a` (due = all 6)

- `promoted=false`
- `first_blocking_gate=panel_lower_quantile`
- panel median \(L_{mid}\) 5% lower quantile ≈ −0.071
- same-τ retention ρ (selected): w0≈0.68, w1≈1.01, w2≈0.92, w3≈1.54,
  w4≈1.03, **w5≈0.41**

### Cycle 2 — `msa_1784527942_af1468d9` (due = [0, 1, 5])

- `promoted=false`
- panel gate passed (lower quantile ≈ +0.160)
- `first_blocking_gate=same_tau_retention_w5` (w5 ρ≈0.75;
  w0≈1.26, w1≈1.36)

Cycle summaries live under
`dexpbt_8gpu_ws/msa_cycles/*.json`.

## Terminal stop (orchestration, not algorithm)

A third cycle (`msa_1784551259_a03ee1cf`, due=[2,3,4], phase PRE_EVAL)
was opening when the continuation guard fail-closed:

```
GUARD_FAILED: worker 2: observed frame 6120000000 exceeds
target + one tick of slack
(baseline 5080000000 + 480000000 + 80000000)
```

Guard logs show multiple workers hit `cap_pending` *during* an
in-flight MSA cycle (busy marker held the park). After the busy marker
cleared, observed frames had already overrun `target + 80M slack`.
Watchdog then observed the sidecar gone with active holds/busy and
wrote `MSA_SIDECAR_DIED.json` (released holds, stopped the arm). All
six trainers and control processes exited; GPUs freed.

## What this run verified on real GPUs

- End-to-end state-faithful bootstrap (tolerance + env clock + lineage
  + observer seed + control restore).
- First-tick stall reproduction + readiness hold barrier + sidecar
  adoption.
- Two full N=16 MSA probe cycles with durable cycle JSON outcomes
  (neither promoted).
- Fail-closed stop on cap overshoot and on sidecar death mid-hold.

## Open follow-up (orchestration only)

Cap enforcement vs long MSA cycles: workers released for probe packets
can advance past `baseline + 480M` while the guard is waiting on
`MSA_BUSY`; the post-busy check then fails with >1 tick of slack. Fix
should park/stop training at the cap even under busy, or widen the
busy-aware cap policy — **no algorithm / promotion-gate change**.
