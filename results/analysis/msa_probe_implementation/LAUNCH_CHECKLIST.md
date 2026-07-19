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
| **disposable GPU hold smoke** (produces the certificate) | `results/launchers/msa_probe/start_stateful_hold_smoke.sh` |
| in-arm hold smoke — **EXECUTION SMOKE ONLY, forbidden in the formal four-arm comparison** | `results/launchers/msa_probe/run_hold_smoke.sh` |

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
   - **6/6 TRUE READINESS BARRIER + STALL REPRODUCTION GATE** — each
     worker's first committed tick (one full 80M interval on the restored
     state) must: confirm the RESUME_APPLIED tolerance structurally, stay
     within ONE mastery shrink of its source tolerance (two shrinks in
     one interval = legacy-fallback cascade → `GUARD_FAILED`), keep
     lineage, and keep the restored control state alive (no default
     clobber of last-mastered/peak/freeze fields →
     `STATE_FIDELITY_FAILED.json`). The moment a worker passes, its FULL
     process tree is SIGSTOP-frozen at that committed boundary (verified
     `T` state; persisted to `READINESS_HOLDS.json` with worker id,
     root/children PIDs, validated frame/tolerance/lineage, hold
     timestamp) — an early worker can NEVER train past its matched
     intervention boundary while the others catch up. If any worker
     fails, every readiness hold is SIGCONTed and the arm stops
     fail-closed. Arm-level: with 6/6 frozen, ≥ 4/6 workers must hold
     EXACTLY their source tolerance, else `STATE_FIDELITY_FAILED.json`
     and a full stop. Only then `FIRST_TICK_VALIDATED.json` (with the
     per-worker reproduction record) is written;
   - **handoff** — BASE_STATEFUL / GATE_RELAX_STATEFUL: all six workers
     are SIGCONTed SIMULTANEOUSLY and `READINESS_HOLDS.json` is deleted
     (matched continuation from each worker's validated tick).
     MSA_STATEFUL / MSA_PERMUTED_STATEFUL: workers are NOT released; the
     sidecar ADOPTS `READINESS_HOLDS.json` (verifies PIDs still `T`,
     committed frame == validated frame, imports the WorkerHolds state,
     atomically converts to `MSA_HOLDS.json`) and the first `PRE_EVAL`
     snapshot comes from EXACTLY the validated frame — any mismatch fails
     closed and stops the arm;
   - **matched worker-local cap** — target = baseline + 480M (6 committed
     ticks at the REAL 80M cadence); each worker parked at its FIRST
     committed frame ≥ target; overshoot beyond one tick (80M) of slack
     fails the arm; 6/6 parked → `ARM_COMPLETE.json` (normal completion)
     and a full stop of guard, sidecar, watchdog and every worker tree —
     every stop escalates SIGCONT → SIGTERM (descendants children-first,
     supervisor/root LAST) → 20 s grace → re-enumerate → SIGKILL
     children-first/root-last → verify no survivor (robust against a
     supervisor respawning children mid-escalation);
6. (MSA arms) starts the sidecar — which REQUIRES the readiness marker
   AND a validated `HOLD_SMOKE_CERTIFICATE.json` (git commit, task, wm
   profile, GPU model, driver must match the current environment; checked
   fail-closed at sidecar startup) — and the external `hold_watchdog`
   (releases SIGSTOPped workers and stops the arm if the sidecar dies
   mid-hold; also stops the arm if the sidecar dies while `MSA_BUSY.json`
   exists, even with no holds recorded).

## Mandatory execution order (MSA arms)

```
start_stateful_hold_smoke.sh  (DISPOSABLE, one throwaway worker on GPU 4)
           → HOLD_SMOKE_CERTIFICATE.json (ok=true)
launch arm → state bootstrap + fingerprint (in-launcher, fail closed)
           → sidecar validates the certificate at startup (fail closed)
           → 6/6 readiness: each validated worker frozen (READINESS_HOLDS.json)
           → stall reproduction gate → FIRST_TICK_VALIDATED.json
           → sidecar ADOPTS the frozen workers (→ MSA_HOLDS.json)
           → first N=4 execution-smoke cycle starts PRE_EVAL from the
             validated boundary (PRE snapshot frame == validated frame)
```

The DISPOSABLE hold smoke runs BEFORE the arm and OUTSIDE it: the old
in-arm `run_hold_smoke.sh` releases the target worker and waits for one
extra committed tick, which advances it past the matched intervention
boundary — it is therefore EXECUTION SMOKE ONLY and forbidden in the
formal four-arm comparison. A failed disposable smoke leaves
`ok=false` in the certificate and the formal sidecar refuses to start.

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
3. [ ] SIX-GPU configuration understood and idle (`nvidia-smi`): the six
       trainers occupy `cuda:0`–`cuda:5`; MSA eval runs on
       `MSA_EVAL_DEVICES` (default `cuda:4,cuda:5`) — the eval devices
       are SHARED with trainers 4/5. That is safe because during every
       eval barrier the GPU4/5 trainers are SIGSTOPped, but they KEEP
       their VRAM allocated, so both GPUs need headroom for one Isaac
       eval process on top of a resident (frozen) trainer. At most one
       Isaac eval process per device. The launcher records the actual
       devices in `<run_dir>/EVAL_DEVICES.json`.
4. [ ] Disk headroom: each arm writes a new run dir (checkpoints ~180MB x
       6 workers x ~6 ticks for 480M frames ≈ 6.5GB per arm; MSA arms add
       read-only snapshots + eval raw ≈ +3GB).
5. [ ] `python -m pytest tests/test_msa_probe.py tests/test_state_bootstrap.py -q`
       green (125 + 32 tests).
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
9. [ ] **MANDATORY GPU gate — DISPOSABLE hold smoke BEFORE the arm
       (ENFORCED by the sidecar).** Run
       `bash results/launchers/msa_probe/start_stateful_hold_smoke.sh
       --i-know-what-i-am-doing` (one throwaway worker, same code / task /
       profile / 8192 envs, GPU 4). It verifies the frame is frozen under
       SIGSTOP and the next committed tick appears after SIGCONT, writes
       `HOLD_SMOKE_CERTIFICATE.json` (git commit, task, profile, GPU
       UUID/model, CUDA/driver, process-tree shape, hold/resume frames,
       ok=true) and tears the throwaway run down completely. The formal
       MSA launcher refuses to start without the certificate file and the
       sidecar re-validates its contents against the live environment at
       startup. The certificate is INVALIDATED by any new commit / GPU /
       driver change — re-run the disposable smoke then. Do NOT run
       `run_hold_smoke.sh` inside a formal arm (it advances the released
       worker one extra committed tick past the matched boundary).
10. [ ] Confirm the bootstrap PRE-SEEDED `curriculum_control/` and
        `observer_state_seed/` in the new run dir (a fresh/empty control
        dir is now a launch BLOCKER, not the expected state — readiness no
        longer assumes "fresh control"), and that `STATE_FIDELITY.json`
        says `state_faithful: true`.

## Launch commands (DO NOT RUN until all boxes are checked)

### BASE_STATEFUL / GATE_RELAX_STATEFUL final start order

```bash
# 1. launch (bootstrap + fingerprint + manifest verify are in-launcher)
bash results/launchers/msa_probe/start_base_stateful.sh          --i-know-what-i-am-doing
# 2. the guard freezes each validated worker; on 6/6 + stall reproduction
#    it writes FIRST_TICK_VALIDATED.json and SIGCONTs all six
#    SIMULTANEOUSLY (READINESS_HOLDS.json removed) — no manual step.
# (same for start_gate_relax_stateful.sh)
```

### MSA_STATEFUL / MSA_PERMUTED_STATEFUL final start order

```bash
# 1. DISPOSABLE hold smoke (one throwaway worker on GPU 4) -> certificate
bash results/launchers/msa_probe/start_stateful_hold_smoke.sh --i-know-what-i-am-doing
# 2. verify: HOLD_SMOKE_CERTIFICATE.json has ok=true
# 3. launch the arm (refuses without the certificate; the sidecar
#    validates commit/task/profile/GPU env at startup)
MSA_NUM_SEEDS=4 \
MSA_EVAL_DEVICES=cuda:4,cuda:5 \
bash results/launchers/msa_probe/start_msa_stateful.sh --i-know-what-i-am-doing
# 4. the guard freezes each validated worker; on 6/6 it writes
#    FIRST_TICK_VALIDATED.json and KEEPS the workers frozen; the sidecar
#    adopts READINESS_HOLDS.json -> MSA_HOLDS.json and starts PRE_EVAL
#    from the frozen validated boundary — no manual step, no in-arm smoke.
```

## Monitoring once launched

- supervisor log: `<run_dir>/launch_master.log`
- bootstrap artifacts: `<run_dir>/STATE_FIDELITY.json`,
  `<run_dir>/STATE_FINGERPRINT.json`, per-policy
  `policy_XX/RESUME_APPLIED.json` (written by train.py at env creation);
- continuation guard: `<run_dir>/continuation_guard.log`; markers:
  - `READINESS_HOLDS.json` — workers frozen at their validated boundaries
    (present during the barrier; retained for MSA arms until the sidecar
    adopts it, deleted on the simultaneous BASE/GATE release);
  - `FIRST_TICK_VALIDATED.json` — 6/6 readiness + stall reproduction
    passed (BASE/GATE workers released simultaneously; MSA sidecar adopts
    the holds);
  - `STATE_FIDELITY_FAILED.json` — the restored state did NOT reproduce
    the stall (majority breakout, or control state default-clobbered);
    the arm was stopped and MSA never started;
  - `GUARD_FAILED` — fail-closed stop with the reason;
  - `ARM_COMPLETE.json` — NORMAL completion (all six parked at +480M);
- hold smoke: `HOLD_SMOKE_CERTIFICATE.json` (disposable-smoke gate for
  formal arms; ok must be true); for execution smokes only:
  `HOLD_SMOKE_RESULT.json` (detail), `HOLD_SMOKE_PASSED.json` /
  `HOLD_SMOKE_FAILED.json` (permanent stop marker — arm was stopped);
- eval devices actually used: `<run_dir>/EVAL_DEVICES.json` (written by
  the MSA launcher; default `cuda:4,cuda:5` shared with trainers 4/5);
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
