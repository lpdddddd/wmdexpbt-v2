# base_retention_msa_probe — Launch Checklist (NOT YET LAUNCHED)

Nothing has been started. Every generated launcher refuses to run unless
invoked with `--i-know-what-i-am-doing` after this checklist is signed off,
and then STILL aborts unless the fail-closed bootstrap verification
(`continuation_guard --verify-only`) passes.

## Launchers (generated, syntax-checked, never executed)

| arm | script |
|---|---|
| BASE_CONTINUATION | `results/launchers/msa_probe/start_base_continuation.sh` |
| GATE_RELAX | `results/launchers/msa_probe/start_gate_relax.sh` |
| MSA | `results/launchers/msa_probe/start_msa.sh` |
| MSA_PERMUTED_ROUTE | `results/launchers/msa_probe/start_msa_permuted_route.sh` |
| hold smoke (single worker) | `results/launchers/msa_probe/run_hold_smoke.sh` |

Each arm has a machine-readable `<arm>.config.json` next to it, embedding
the resume checkpoints WITH bootstrap head hashes (`sha256_head`), live
tolerances, lineage ids, seed manifest identity, bootstrap parameters, the
27-cell eval cost table and the frame-ledger contract. Every arm launcher:

1. runs `continuation_guard --verify-only` and aborts on any failure
   (checkpoint missing/tampered, mapping error, empty manifest field);
2. passes `--recover-checkpoint-per-policy` AND
   `--tolerance-resume-per-policy` (explicit per-worker tolerance restore);
3. starts the `continuation_guard` loop:
   - **6/6 readiness barrier** — `FIRST_TICK_VALIDATED.json` is written
     atomically only after ALL SIX workers pass first-committed-tick
     validation (tolerance ≤1e-6 / lineage / ≥5B frame); any failure stops
     the whole arm (`GUARD_FAILED`) with no readiness marker;
   - **matched +100M cap** — each worker is parked at its FIRST committed
     frame ≥ baseline+100M; overshoot beyond one 20M tick of slack fails
     the arm; 6/6 parked → `ARM_COMPLETE.json` (normal completion) and a
     full stop of guard, sidecar, watchdog and every worker tree;
4. (MSA arms) starts the sidecar — which WAITS for the readiness marker —
   and the external `hold_watchdog` (releases SIGSTOPped workers and stops
   the arm if the sidecar dies mid-hold).

## Pre-launch checks

1. [ ] All six 5B tip checkpoints still exist and match their bootstrap
       hashes: `python -m dexpbt_isaaclab.msa.continuation_guard
       --config results/launchers/msa_probe/<arm>.config.json
       --verify-only` (already verified once at generation time:
       BOOTSTRAP OK 6/6).
2. [ ] GPUs 0–5 idle for training; GPUs 6 and 7 reserved for MSA eval
       (`MSA_EVAL_DEVICES`, default `cuda:6,cuda:7`; at most one Isaac
       eval process per device); no other jobs (`nvidia-smi`).
3. [ ] Disk headroom: each arm writes a new run dir (checkpoints ~180MB x
       6 workers x ~10 ticks for 100M frames ≈ 11GB per arm; MSA arms add
       read-only snapshots + eval raw ≈ +3GB).
4. [ ] `python -m pytest tests/test_msa_probe.py -q` green (95 tests).
5. [ ] Dry-run regenerates cleanly (heterogeneous taus, 20M packets,
       post-hold packet starts):
       `python -m dexpbt_isaaclab.msa.dry_run --out /tmp/trace.md`.
6. [ ] Decide arm ordering. Recommended: run arms SEQUENTIALLY (each needs
       all 6 training GPUs). Suggested order: BASE_CONTINUATION →
       GATE_RELAX → MSA → MSA_PERMUTED_ROUTE.
7. [ ] MSA arms — eval budget is `MSA_NUM_SEEDS=4` by default: an
       EXECUTION SMOKE that validates the machinery only, never a method
       conclusion (27 eval cells ≈ 5.4 h on 2 GPUs). Pilot N=16
       (≈ 21.6 h) must be set explicitly; N=256 is formal-only and NOT
       affordable on 2 GPUs (≈ 345.6 h) — see `EVAL_COST_ESTIMATE.json`.
8. [ ] **MANDATORY GPU gate — hold smoke BEFORE any probe cycle.**
       SIGSTOP/SIGCONT against a live Isaac trainer is validated in
       simulation and against real (non-Isaac) processes only. After the
       arm launches and `FIRST_TICK_VALIDATED.json` appears, run
       `bash results/launchers/msa_probe/run_hold_smoke.sh <run_dir> 0 60`
       and check `HOLD_SMOKE_RESULT.json` (`ok: true`, frame frozen during
       the hold, next committed tick after release). Abort the arm if the
       trainer misbehaves on resume. Only then allow the sidecar's first
       cycle (watch `msa_sidecar.log` + `nvidia-smi` through PRE_EVAL).
9. [ ] Confirm `curriculum_control/` of the new run dir is empty at start
       (fresh arm; provenance intentionally absent → probe immediately
       due, per spec — but gated behind the 6/6 readiness marker).

## Launch commands (DO NOT RUN until all boxes are checked)

```bash
bash results/launchers/msa_probe/start_base_continuation.sh --i-know-what-i-am-doing
bash results/launchers/msa_probe/start_gate_relax.sh        --i-know-what-i-am-doing
bash results/launchers/msa_probe/start_msa.sh               --i-know-what-i-am-doing
bash results/launchers/msa_probe/start_msa_permuted_route.sh --i-know-what-i-am-doing
# then, for MSA arms, after FIRST_TICK_VALIDATED.json appears:
bash results/launchers/msa_probe/run_hold_smoke.sh <run_dir> 0 60
```

## Monitoring once launched

- supervisor log: `<run_dir>/launch_master.log`
- continuation guard: `<run_dir>/continuation_guard.log`; markers:
  - `FIRST_TICK_VALIDATED.json` — 6/6 readiness (sidecar may start);
  - `GUARD_FAILED` — fail-closed stop with the reason;
  - `ARM_COMPLETE.json` — NORMAL completion (all six parked at +100M);
- MSA sidecar log: `<run_dir>/msa_sidecar.log`; hold state:
  `<run_dir>/MSA_HOLDS.json` (present only while workers are held);
- hold watchdog: `<run_dir>/hold_watchdog.log`; `MSA_SIDECAR_DIED.json`
  means the sidecar died mid-hold — workers were SIGCONTed and the arm
  was stopped;
- `COMMIT_TIMEOUT.json` — a durable tolerance commit did not confirm
  within `--commit-max-wait-s`; the arm was stopped, promotion NOT
  claimed;
- probe cycle outcomes: `<run_dir>/dexpbt_8gpu_ws/msa_cycles/*.json`
  (intervention id, per-worker tau/tau_plus, L_mid table, routing, panel
  statistic, rho table, promoted / first_blocking_gate, resumed flag)
- per-worker MSA state: `msa_provenance` / `msa_state_machine` /
  `msa_cycle` fields inside
  `<run_dir>/dexpbt_8gpu_ws/curriculum_control/00X.json`

## Abort criteria

- `GUARD_FAILED` marker (first-tick tolerance/lineage/frame mismatch, or
  cap overshoot) → the guard already stopped the arm; investigate before
  any relaunch;
- `ARM_COMPLETE.json` → normal end of the continuation arm (not an
  abort);
- `MSA_SIDECAR_DIED.json` → watchdog already released workers and stopped
  the arm; inspect `msa_sidecar.log` for the crash;
- `COMMIT_TIMEOUT.json` → arm stopped; check whether the worker's trainer
  was alive and committing ticks;
- any worker's ledger frame regresses (must be monotonic) → kill arm,
  investigate before resuming;
- hold smoke `ok: false` → do NOT run any probe cycle; the SIGSTOP
  approach needs rework (supervisor-level checkpoint-and-park fallback);
- nsucc collapse below the original run's collapse thresholds → the
  original gates (unchanged in every arm) handle it; do not intervene
  manually.
