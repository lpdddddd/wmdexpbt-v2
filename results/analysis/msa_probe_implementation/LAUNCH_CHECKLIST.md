# base_retention_msa_probe — Launch Checklist (STATEFUL arms, NOT YET LAUNCHED)

Nothing has been started. Every generated launcher refuses to run unless
invoked with `--i-know-what-i-am-doing` after this checklist is signed off,
and then STILL aborts unless BOTH the state-faithful bootstrap
(`state_bootstrap` + fingerprint verification) AND the fail-closed manifest
verification (`continuation_guard --verify-only`) pass.

Since 2026-07-19 the four arms are STATE-FAITHFUL: they restore the source
run's full stall state (curriculum_control gate state, rebuilt observer
memory, env clock), not just `.pth` + tolerance. The previous execution smoke
proved the bare warm start does NOT reproduce the stall (empty
`curriculum_control` → ungated legacy curriculum → 0.0231 → 0.0168 in one
80M interval).

## Launchers (generated, syntax-checked, never executed)

| arm | script |
|---|---|
| BASE_STATEFUL | `results/launchers/msa_probe/start_base_stateful.sh` |
| GATE_RELAX_STATEFUL | `results/launchers/msa_probe/start_gate_relax_stateful.sh` |
| MSA_STATEFUL | `results/launchers/msa_probe/start_msa_stateful.sh` |
| MSA_PERMUTED_STATEFUL | `results/launchers/msa_probe/start_msa_permuted_stateful.sh` |
| hold smoke (single worker) | `results/launchers/msa_probe/run_hold_smoke.sh` |

Each arm has a machine-readable `<arm>.config.json` next to it, embedding
the resume checkpoints WITH bootstrap head hashes (`sha256_head`), live
tolerances, lineage ids, the source run, the real committed cadence (80M),
the 6-tick budget breakdown, seed manifest identity, bootstrap parameters,
the 27-cell eval cost table and the frame-ledger contract. Every arm
launcher, in order:

1. runs the STATE-FAITHFUL BOOTSTRAP (`msa.state_bootstrap`): copies each
   worker's source control JSON (gate state intact, `msa_*` cleared),
   rebuilds the observer's in-memory state from the source control + last
   12 committed metas into `observer_state_seed/`, copies the resume
   snapshots, writes `STATE_FIDELITY.json` + `STATE_FINGERPRINT.json`;
   ANY missing critical state fails closed and aborts the launch;
2. re-validates the fingerprint (`state_bootstrap --verify`): restored
   state unchanged on disk, no consumed source file modified;
3. runs `continuation_guard --verify-only` and aborts on any failure
   (checkpoint missing/tampered, mapping error, empty manifest field);
4. passes `--recover-checkpoint-per-policy`,
   `--tolerance-resume-per-policy` AND
   `--curriculum-clock-resume-per-policy` (tolerance + restart-step clock,
   so `post_shrink_consolidation_until_restart_step` keeps its source
   meaning); train.py writes a structured
   `policy_XX/RESUME_APPLIED.json` audit of what was applied;
5. starts the `continuation_guard` loop:
   - **6/6 readiness barrier + STALL REPRODUCTION GATE** — each worker's
     first committed tick (one full 80M interval on the restored state)
     must: confirm the RESUME_APPLIED tolerance structurally, stay within
     ONE mastery shrink of its source tolerance (two shrinks in one
     interval = legacy-fallback cascade → `GUARD_FAILED`), keep lineage,
     and keep the restored control state alive (no default clobber of
     last-mastered/peak/freeze fields → `STATE_FIDELITY_FAILED.json`).
     Arm-level: ≥ 4/6 workers must hold EXACTLY their source tolerance,
     else `STATE_FIDELITY_FAILED.json` and a full stop — the stall was
     NOT reproduced and MSA must not start. Only then
     `FIRST_TICK_VALIDATED.json` (with the per-worker reproduction
     record) is written;
   - **matched worker-local cap** — target = baseline + 480M (6 committed
     ticks at the REAL 80M cadence); each worker parked at its FIRST
     committed frame ≥ target; overshoot beyond one tick (80M) of slack
     fails the arm; 6/6 parked → `ARM_COMPLETE.json` (normal completion)
     and a full stop of guard, sidecar, watchdog and every worker tree —
     every stop now escalates SIGCONT → SIGTERM → 20 s grace →
     re-enumerate → SIGKILL survivors → verify exit (Isaac trainers
     survived bare SIGTERM in the smoke);
6. (MSA arms) starts the sidecar — which WAITS for BOTH the readiness
   marker AND `HOLD_SMOKE_PASSED.json` — and the external `hold_watchdog`
   (releases SIGSTOPped workers and stops the arm if the sidecar dies
   mid-hold; also stops the arm if the sidecar dies while `MSA_BUSY.json`
   exists, even with no holds recorded).

## Mandatory execution order (MSA arms)

```
launch arm → state bootstrap + fingerprint (in-launcher, fail closed)
           → 6/6 readiness + stall reproduction (FIRST_TICK_VALIDATED.json)
           → run_hold_smoke.sh → HOLD_SMOKE_PASSED.json
           → first N=4 execution-smoke cycle
```

The formal sidecar enforces this order itself: it starts no cycle until
both markers exist. A failed smoke writes `HOLD_SMOKE_FAILED.json` (a
permanent stop marker), SIGCONTs everything and fail-closed stops the
whole arm; the PASSED marker is never written on any failure path.

## Pre-launch checks

1. [ ] All six 5B tip checkpoints still exist and match their bootstrap
       hashes: `python -m dexpbt_isaaclab.msa.continuation_guard
       --config results/launchers/msa_probe/<arm>.config.json
       --verify-only` (already verified once at generation time:
       BOOTSTRAP OK 6/6).
2. [ ] Source run untouched:
       `python -m dexpbt_isaaclab.msa.state_bootstrap --audit-cadence
       --source-run <source_run> --run-dir /tmp/x` reports
       `committed_tick_frames: 80000000, matches_module_constant: true`.
3. [ ] GPUs 0–5 idle for training; two eval GPUs reserved for MSA eval
       (`MSA_EVAL_DEVICES`; on the current 6-GPU machine use
       `cuda:4,cuda:5` — trainers then share 0–3 — or run eval-heavy
       phases sequentially); at most one Isaac eval process per device;
       no other jobs (`nvidia-smi`).
4. [ ] Disk headroom: each arm writes a new run dir (checkpoints ~180MB x
       6 workers x ~6 ticks for 480M frames ≈ 6.5GB per arm; MSA arms add
       read-only snapshots + eval raw ≈ +3GB).
5. [ ] `python -m pytest tests/test_msa_probe.py tests/test_state_bootstrap.py -q`
       green (114 + 32 tests).
6. [ ] Dry-run regenerates cleanly (heterogeneous taus, 80M packets,
       post-hold packet starts):
       `python -m dexpbt_isaaclab.msa.dry_run --out /tmp/trace.md`.
7. [ ] Decide arm ordering. Recommended: run arms SEQUENTIALLY (each needs
       all 6 training GPUs). Suggested order: BASE_STATEFUL →
       GATE_RELAX_STATEFUL → MSA_STATEFUL → MSA_PERMUTED_STATEFUL.
       BASE_STATEFUL first also doubles as the cheapest stall-reproduction
       validation of the bootstrap itself.
8. [ ] MSA arms — eval budget is `MSA_NUM_SEEDS=4` by default: an
       EXECUTION SMOKE that validates the machinery only, never a method
       conclusion (27 eval cells ≈ 5.4 h on 2 GPUs). Pilot N=16
       (≈ 21.6 h) must be set explicitly; N=256 is formal-only and NOT
       affordable on 2 GPUs (≈ 345.6 h) — see `EVAL_COST_ESTIMATE.json`.
9. [ ] **MANDATORY GPU gate — hold smoke BEFORE any probe cycle (ENFORCED
       by the sidecar).** After the arm launches and
       `FIRST_TICK_VALIDATED.json` appears, run
       `bash results/launchers/msa_probe/run_hold_smoke.sh <run_dir> 0 60`.
       Success (frame frozen during the hold + next committed tick after
       release + no residual holds) atomically writes
       `HOLD_SMOKE_PASSED.json`, which is the ONLY thing that opens the
       sidecar's cycle gate. Failure writes `HOLD_SMOKE_FAILED.json` and
       stops the arm. Watch the first cycle via `msa_sidecar.log` +
       `nvidia-smi` through PRE_EVAL.
10. [ ] Confirm the bootstrap PRE-SEEDED `curriculum_control/` and
        `observer_state_seed/` in the new run dir (a fresh/empty control
        dir is now a launch BLOCKER, not the expected state — readiness no
        longer assumes "fresh control"), and that `STATE_FIDELITY.json`
        says `state_faithful: true`.

## Launch commands (DO NOT RUN until all boxes are checked)

```bash
bash results/launchers/msa_probe/start_base_stateful.sh          --i-know-what-i-am-doing
bash results/launchers/msa_probe/start_gate_relax_stateful.sh    --i-know-what-i-am-doing
bash results/launchers/msa_probe/start_msa_stateful.sh           --i-know-what-i-am-doing
bash results/launchers/msa_probe/start_msa_permuted_stateful.sh  --i-know-what-i-am-doing
# then, for MSA arms, after FIRST_TICK_VALIDATED.json appears:
bash results/launchers/msa_probe/run_hold_smoke.sh <run_dir> 0 60
# the sidecar starts its first cycle only after HOLD_SMOKE_PASSED.json
```

Recommended next GPU smoke (single arm, machinery validation):

```bash
MSA_NUM_SEEDS=4 \
MSA_EVAL_DEVICES=cuda:4,cuda:5 \
bash results/launchers/msa_probe/start_msa_stateful.sh --i-know-what-i-am-doing
```

## Monitoring once launched

- supervisor log: `<run_dir>/launch_master.log`
- bootstrap artifacts: `<run_dir>/STATE_FIDELITY.json`,
  `<run_dir>/STATE_FINGERPRINT.json`, per-policy
  `policy_XX/RESUME_APPLIED.json` (written by train.py at env creation);
- continuation guard: `<run_dir>/continuation_guard.log`; markers:
  - `FIRST_TICK_VALIDATED.json` — 6/6 readiness + stall reproduction
    passed (sidecar may start);
  - `STATE_FIDELITY_FAILED.json` — the restored state did NOT reproduce
    the stall (majority breakout, or control state default-clobbered);
    the arm was stopped and MSA never started;
  - `GUARD_FAILED` — fail-closed stop with the reason;
  - `ARM_COMPLETE.json` — NORMAL completion (all six parked at +480M);
- hold smoke: `HOLD_SMOKE_RESULT.json` (detail),
  `HOLD_SMOKE_PASSED.json` (gate open) / `HOLD_SMOKE_FAILED.json`
  (permanent stop marker — arm was stopped);
- MSA sidecar log: `<run_dir>/msa_sidecar.log`; hold state:
  `<run_dir>/MSA_HOLDS.json` (present only while workers are held);
  cycle state: `<run_dir>/MSA_BUSY.json` (present from before the first
  hold of a cycle until DONE / failure cleanup — while it exists the
  guard defers park/cap actions as `cap_pending`);
- hold watchdog: `<run_dir>/hold_watchdog.log`; `MSA_SIDECAR_DIED.json`
  means the sidecar died mid-hold OR mid-cycle (busy marker present,
  even with no holds) — workers were SIGCONTed and the arm was stopped;
- `COMMIT_TIMEOUT.json` — a durable tolerance commit did not confirm
  within `--commit-max-wait-s`; includes the per-worker
  `commit_acks_by_worker` table; the arm was stopped, promotion NOT
  claimed;
- probe cycle outcomes: `<run_dir>/dexpbt_8gpu_ws/msa_cycles/*.json`
  (intervention id, per-worker tau/tau_plus, L_mid table, routing, panel
  statistic, rho table, promoted / first_blocking_gate, resumed flag)
- per-worker MSA state: `msa_provenance` / `msa_state_machine` /
  `msa_cycle` fields inside
  `<run_dir>/dexpbt_8gpu_ws/curriculum_control/00X.json`

## Abort criteria

- `STATE_FIDELITY_FAILED.json` → the stall was not reproduced on the
  restored state; the causal experiment premise is void — investigate the
  fidelity table before any relaunch; do NOT start MSA on such a run;
- `GUARD_FAILED` marker (missing RESUME_APPLIED, legacy shrink cascade,
  lineage/frame mismatch, or cap overshoot) → the guard already stopped
  the arm; investigate before any relaunch;
- `ARM_COMPLETE.json` → normal end of the continuation arm (not an
  abort);
- `MSA_SIDECAR_DIED.json` → watchdog already released workers and stopped
  the arm; inspect `msa_sidecar.log` for the crash;
- `COMMIT_TIMEOUT.json` → arm stopped; check whether the worker's trainer
  was alive and committing ticks;
- any worker's ledger frame regresses (must be monotonic) → kill arm,
  investigate before resuming;
- `HOLD_SMOKE_FAILED.json` → the arm was already fail-closed stopped and
  the sidecar gate can never open; the SIGSTOP approach needs rework
  (supervisor-level checkpoint-and-park fallback) before any relaunch;
- nsucc collapse below the original run's collapse thresholds → the
  original gates (unchanged in every arm) handle it; do not intervene
  manually.
