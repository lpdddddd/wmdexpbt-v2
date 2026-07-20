# base_retention_msa_probe — Implementation Report

Date: 2026-07-19 (initial), updated 2026-07-19 with the pilot-integration
fix pass ("Fix MSA pilot integration and phase safety"). Branch:
`base-retention`. No GPU training or evaluation was started; everything
below is code, tests, launcher generation, a read-only log audit, and CPU
dry-runs.

## 0. Pilot integration fix pass (this commit)

The MSA algorithm (state machine, criterion-local tau_plus, common
exposure, L_mid, top-3 routing, MSA_PERMUTED_ROUTE, same-tau rho, panel
bootstrap, four arms) is UNCHANGED. Ten production-integration bugs that
blocked a live pilot were fixed:

1. **Heterogeneous tolerances** — the "all due workers share one tau"
   restriction is removed. Each worker uses `tau[w] = active_tau` and
   `tau_plus[w] = max(0.010, 0.95 * tau[w])`; all pre/mid/post evals, pins,
   provenance and rho are worker-local. L_mid is documented as a
   CRITERION-LOCAL improvement everywhere it is consumed (routing/panel).
2. **Parallel exposure** — the controller pins every due worker to its own
   tau_plus first, then waits on a single `train_until` barrier over all
   committed-tick ledgers (never a serial per-worker loop). After MID_EVAL
   non-recipients are re-pinned to their own tau immediately; only the
   top-3 recipients run the second packet, in parallel; before POST_EVAL
   everyone is back at tau. All recorded frames are OBSERVED committed
   frames (a fabricated `start + n` is structurally impossible; the fake
   world and unit tests enforce this). Packet size is now **20M frames**
   (= one reliable committed PBT tick), updated in configs/dry-run/tests.
3. **Eval phase barrier** — PRE/MID/POST snapshots and evaluations happen
   with the participating workers HELD at a committed checkpoint
   (production: SIGSTOP/SIGCONT of the training process tree via
   `policy_NN/pid.txt`; released before any durable tolerance commit).
   Eval subprocesses require an explicit `--eval-device` (fail-closed;
   launcher pins `cuda:${MSA_EVAL_GPU:-6}`, outside `DEXPBT_GPUS=0-5`).
   Pilot eval size is N=16 fixed seeds; N=256 formal must be requested
   explicitly and is never a default.
4. **Unified state persistence** — `msa_provenance`, `msa_state_machine`
   and the in-flight `msa_cycle` context live INSIDE
   `CurriculumControlState` (single source of truth on the existing
   control / resume chain). Every phase transition persists; a restarted
   sidecar resumes the pending cycle idempotently (completed segments are
   skipped via provenance, snapshots and ccrs_eval results are reused).
   The old `msa_provenance_*.json` side files are gone.
5. **Crash cleanup** — any exception in any phase restores every
   non-promoted worker to its own tau (synchronous pin first, durable
   commit after holds are released), clears holds and pins, records
   FAILED + failing phase + ABORT history, and re-raises `CycleFailure`.
   A FAILED cycle is not auto-resumed; a hard kill (no cleanup) is
   resumed idempotently on restart.
6. **Control-JSON concurrency** — new locked
   `update_curriculum_control(path, mutator)` API (advisory flock on
   `<path>.lock`); ALL MSA writes go through it. The observer's per-tick
   full-state write goes through `write_curriculum_control_preserving_msa`
   which re-reads the on-disk msa_* fields under the same lock, so neither
   writer can clobber the other (`mark_shrink_consumed` /
   `apply_source_safe_consolidation_on_copy` also use the locked path).
   The sidecar additionally owns a DEDICATED pin field
   `msa_forced_success_tolerance` (priority over, and never touching, the
   observer-owned `forced_success_tolerance`).
7. **Canonical tolerance** — `canonical_tau(x) = format(x, ".12g")` is
   used end-to-end (eval request CLI, raw-result matching,
   `SeedEvalResult.tau`); the old `.6f`-write / `1e-9`-read mismatch is
   regression-tested with full-precision taus. The CCRS `EvalUnit.unit_key`
   also uses the canonical form so two close taus can no longer collide
   into one result file.
8. **Continuation resume validation** — all four arms: the launcher runs
   `continuation_guard --verify-only` (checkpoint existence + size/head
   sha256 + worker mapping + manifest fields) BEFORE anything launches,
   passes `--tolerance-resume-per-policy` (a bare `--checkpoint` does NOT
   restore the curriculum tolerance on first launch), and starts a guard
   loop that fail-closes the whole arm if the FIRST committed tick's
   tolerance/lineage/frame does not match the manifest, and hard-stops any
   worker exceeding baseline **+100M frames** (+1 tick slack) — no silent
   extra ~5B after an epoch reset.
9. **Durable promotion** — PROMOTE commits `success_tolerance =
   tau_plus[w]` per worker: pin → wait for a live committed tick at the
   new value → sync `curriculum_resume_snapshot.json` (tolerance +
   last_curriculum_update) → clear the pin. Stay/abort paths commit the
   original `tau[w]` the same way. Restart preserves the committed value;
   no worker is ever left on a forever-pin.
10. **Tests** — 72 MSA tests (30 new), full per-file CPU suite, and a
    production-style heterogeneous-tau dry run with real tick
    quantization (see `TEST_REPORT.md`, `DRY_RUN_TRACE.md`).

## Goal

Diagnose and (conditionally) unlock the tolerance progression deadlock at
tau ≈ 0.022–0.028 in the completed strict-5B base-retention population,
via Matched Sentinel Advancement (MSA) — without relaxing all gates,
disabling safety mechanisms, stacking freezes, or reading retention across
different tolerances.

## 1. Read-only blocker audit (3B–5B)

`python -m dexpbt_isaaclab.msa.audit` over
`regrasping_cser_dexpbt_conservative_retention_strict_5b_20260716_130631`
(sources: per-tick `curriculum_shrink_reason` in
`metrics/training_curve_policy_XX.jsonl`, sub-attributed with
`curriculum_safety_block_reason` and the frame-tagged
`curriculum_safety_events.jsonl`). Result
(`blocker_audit_3b_5b.json`): of 168 no-shrink snapshot intervals across
the 6 workers in 3B–5B:

| first blocker | count | share |
|---|---|---|
| rollback / collapse freeze | 91 | 54% |
| post_shrink_consolidation | 28 | 17% |
| late_floor_retention_gate | 26 | 15% |
| other (incl. low-nsucc) | 9 | 5% |
| danger_zone | 6 | 4% |
| pre_floor_peak_guard | 5 | 3% |
| global_retention (real-signal) | 3 | 2% |

Raw gate-block event counts tell the same story at higher granularity:
late_floor_retention_block fired 470 times and collapse/rollback freezes
335 times in the window. Interpretation: the deadlock is dominated by the
**collapse→freeze→re-collapse loop** (a worker attempts the shrink band,
drops, freezes 200–250M frames, recovers, repeats), with the late-floor
retention gate as the second structural blocker on p2/p3/p5. This is
exactly the failure mode MSA targets: give matched, budgeted tau_plus
exposure and promote only on demonstrated same-tau retention, instead of
letting workers oscillate into freezes.

## 2. What was implemented

New package `dexpbt_isaaclab/msa/` (pure python, no Isaac imports):

| module | contents |
|---|---|
| `provenance.py` | tolerance-tagged exposure provenance (`ExposureSegment`, `InterventionRecord` with intervention/worker/lineage/source-checkpoint/active-tau/proposed-tau/due-frame/segment/evaluator-manifest/snapshot-IDs/recipient-status/first-blocking-gate/final-decision), `tau_plus = max(0.010, 0.95*tau)`, `probe_due` (100M window; missing provenance ⇒ immediately due; floor workers and tau ≤ 0.010 never due) |
| `state_machine.py` | explicit `MSAPhase` enum machine NORMAL→PRE_EVAL→COMMON_PROBE→MID_EVAL→ROUTE_SECOND_HALF→SECOND_PROBE→POST_EVAL→DECIDE→NORMAL/PROMOTE; illegal transitions raise; serializes into the control-state chain; per-phase training permission + required tolerance |
| `decision.py` | fail-closed paired-eval contracts (same tau, same ordered seed manifest), `L_mid`, top-k routing with stable tie-break + pre-registered permuted route (seed 777001), same-tau `rho` (p_pre ≥ 0.01 else `retention_unavailable`), per-worker paired bootstrap → six-worker median → 5th percentile (seed 20260719, 10000 repeats; explicitly labeled a conditional panel statistic), promotion gates with `first_blocking_gate` and API-level rejection of cross-tolerance/historical gate names |
| `eval_bridge.py` | immutable read-only snapshots (`chmod a-w` + `assert_immutable`), ordered seed manifest (identity-hashed; N=16 pilot default, N=256 formal explicit-only), canonical `.12g` tolerances end-to-end, REQUIRED `eval_device` (maps to `CUDA_VISIBLE_DEVICES` for the child), ccrs_eval subprocess command construction (policy-only, deterministic, per-seed results), fail-closed per-seed result loading |
| `controller.py` | `MSAController.run_cycle`: one full probe cycle for the due panel with heterogeneous worker-local tau/tau_plus, PARALLEL 20M committed-tick packets (observed frames only), phase-barrier holds around every eval, per-transition persistence + idempotent resume, crash cleanup (revert-to-tau, clear pins/holds, FAILED record) |
| `sidecar.py` | production wiring: committed-tick worker views (`LATEST.commit` marker, never a bare newest-file glob); SIGSTOP/SIGCONT phase holds; locked msa_* control updates; `msa_forced_success_tolerance` pins; durable tolerance commits synced into `curriculum_resume_snapshot.json`; parallel `train_until` barrier over committed ledgers; explicit `--eval-device`; pending-cycle resume on startup |
| `continuation_guard.py` | fail-closed bootstrap verification (`--verify-only`: existence + head-sha256 + worker mapping), first-committed-tick tolerance/lineage/frame validation, worker-local +100M hard frame cap; stops the whole arm on any violation |
| `audit.py` | the read-only 3B–5B first-blocker audit above |
| `arms.py` | resolves the six 5B tip checkpoints from the CCRS manifest (fail-closed, with bootstrap head hashes), builds the four arm configs + launchers (guard preflight + tolerance resume + guard loop wired in) |
| `dry_run.py` | deterministic in-memory production-style simulation: heterogeneous real 5B-tip tolerances, tick-quantized parallel trainer, held-worker assertions, persistent state store, per-phase crash injection (soft + hard-kill) |

Changes to existing files (all inert with `msa_enabled=False`, which is
the default):

- `pbt/pbt_curriculum_p1.py`: `CurriculumControlState` gains
  `msa_enabled=False`, `msa_forced_success_tolerance=0.0`,
  `msa_provenance=None`, `msa_state_machine=None`, `msa_cycle=None` —
  serialized in the SAME control JSON / resume snapshot chain. The env
  honors the pin ONLY when `msa_enabled` is true (unit-tested inert
  otherwise). New locked `update_curriculum_control` /
  `write_curriculum_control_preserving_msa` APIs;
  `mark_shrink_consumed` and `apply_source_safe_consolidation_on_copy`
  now go through the locked read-modify-write.
- `pbt/pbt_observer.py`: the per-tick control write uses
  `write_curriculum_control_preserving_msa` (identical to the old write
  when no sidecar exists).
- `scripts/launch_pbt.py`: `--recover-checkpoint-per-policy` (unchanged)
  plus new `--tolerance-resume-per-policy` (per-policy
  `--success_tolerance_resume` injection; requires the per-policy
  checkpoints; validated length/range) — a bare `--checkpoint` does NOT
  restore the curriculum tolerance on a fresh run dir.
- `ccrs/eval_protocol.py`: `EvalUnit.unit_key` uses the canonical `.12g`
  tolerance (two taus identical at 6 decimals but different beyond can no
  longer collide into one result file). Raw-row content is unchanged;
  existing raw archives remain valid for analysis.

## 3. Key design decisions

- **No parallel checkpoint system**: provenance + state machine + cycle
  context ride inside the existing curriculum control JSON chain. A
  pre-MSA checkpoint simply lacks the keys → `probe_due` fires immediately
  on resume, per spec.
- **Sidecar, not trainer surgery**: the probe pins tau_plus/tau through
  the sidecar-owned `msa_forced_success_tolerance` (honored by the env
  only when `msa_enabled`), and holds workers with SIGSTOP/SIGCONT during
  eval phases. The trainer keeps running normally through probe segments,
  so policy, PPO optimizer, replay/snippet state, PBT lineage and
  curriculum state are physically untouched (nothing reloads or resets
  them), and probe frames land on the worker's monotonic ledger and
  consume the 100M budget.
- **Retention is same-tau by construction**: `same_tau_retention` takes two
  `SeedEvalResult`s and fails closed on any tau/manifest mismatch; there is
  no code path through which a historical peak at another tolerance could
  enter the promotion decision (gate names suggesting it are rejected).
- **q_post is quarantined**: evaluated only for second-packet recipients,
  stored in the intervention record as unequal-dose follow-up, never in the
  common-dose L_mid statistic (unit-tested).
- **Statistics pre-registered**: bootstrap seed 20260719 / 10000 repeats /
  5th percentile of the panel median; permuted-route seed 777001; pilot
  eval N=16 ordered seeds (formal N=256 explicit-only). The panel
  statistic is explicitly labeled as a CRITERION-LOCAL conditional panel
  statistic (each worker's L_mid at its own tau_plus), not a population CI.

## 4. Continuation arms (prepared, NOT launched)

See `CONFIG_DIFF.md` for the full diff table and `LAUNCH_CHECKLIST.md` for
launch preconditions. All four arms resume from the same six 5B tips with
matched seed 42, explicit per-policy tolerance resume, fail-closed
bootstrap verification and a +100M worker-local hard frame cap:

1. `BASE_CONTINUATION` — original flags, unchanged.
2. `GATE_RELAX` — single change: `late_floor_retention_min_retention`
   0.88 → 0.75 (the pre-existing hard minimum). Delivered via a dedicated
   `cser_dexpbt_conservative_gate_relax` profile (single-key table diff),
   NOT via the CLI flag alone: profile expansion is authoritative and would
   silently revert a CLI-only override back to 0.88. See `CONFIG_DIFF.md`.
3. `MSA` — probe sidecar; original numeric gates; cross-tolerance retention
   read replaced by same-tau rho (threshold 0.88); pilot N=16 evals on a
   dedicated eval GPU.
4. `MSA_PERMUTED_ROUTE` — routing negative control (same total dose).

## 5. Pre-pilot safety patch ("Harden MSA readiness holds and pilot budget")

A final patch pass before any pilot execution, fixing run-control gaps
that could only bite in production:

1. **Six-worker readiness barrier.** `continuation_guard` now writes
   `FIRST_TICK_VALIDATED.json` ATOMICALLY only after ALL SIX workers
   produced a validated first continuation tick (frame/tolerance/lineage/
   checkpoint identity + timestamp recorded per worker). Any failure →
   `GUARD_FAILED`, whole arm stopped, marker never written. The sidecar
   has a REQUIRED `--readiness-marker` and refuses to start OR resume any
   cycle until the marker exists and committed worker views are exactly
   {0..5} — a launcher that brought up a subset can never trigger a
   partial-panel cycle.
2. **Post-hold packet starts.** PRE_EVAL now holds FIRST, then re-reads
   every worker's newest committed meta (`refresh_view`); the packet
   start frames and the pre snapshots come from that post-hold read, and
   worker/tau/lineage are re-validated against the cycle. A tick that
   committed between due detection and the hold ran at tau and is
   EXCLUDED from tau_plus exposure (unit-tested). Snapshots now return
   their recorded frozen frame (`<snap>.meta.json`) and the controller
   enforces snapshot frame == phase boundary frame for pre/mid/post.
   Resume keeps persisted start_frames and validates current >= persisted.
3. **Hold safety + external watchdog.** `WorkerHolds` verifies every PID
   reached the stopped state (undo + fail closed otherwise), releases
   children BEFORE the supervisor/root, and persists `MSA_HOLDS.json`.
   A separate `hold_watchdog` process (independent of the sidecar's
   `finally`) SIGCONTs all recorded PIDs if the sidecar dies with active
   holds, writes `MSA_SIDECAR_DIED.json` and stops the whole arm. Every
   guard stop path SIGCONTs held PIDs first and terminates FULL process
   trees, never just the supervisor.
4. **Hold smoke mode.** `--hold-smoke-only --hold-smoke-worker N
   --hold-smoke-seconds S`: readiness-gated single-worker hold check
   (frame frozen during hold, next committed tick after release), no
   tolerance writes / cycle / eval, `HOLD_SMOKE_RESULT.json`, release
   guaranteed. Prepared launcher: `run_hold_smoke.sh` (not executed).
5. **Matched +100M cap.** `target[w] = baseline[w] + 100M`; each worker
   is individually parked at its FIRST committed frame >= target;
   overshoot beyond one tick of slack fails the arm (a worker can never
   silently run to ~140M). All six parked → `ARM_COMPLETE.json` (NORMAL
   completion — the old failure-semantics CAP_REACHED marker is gone)
   and guard/sidecar/watchdog/worker trees are fully stopped. Whatever
   the resume runner's `max_iterations` semantics (rl_games `max_epochs`
   on a warm start), this worker-local cap is the final authority.
6. **Bounded durable commits.** `--commit-max-wait-s` (default 3600); on
   timeout `CommitTimeout` carries worker/target-tau/last-frame/last-tau,
   the sidecar writes `COMMIT_TIMEOUT.json`, releases all holds and stops
   the arm. Crash cleanup restores the original-tau PIN synchronously and
   NEVER starts a second full bounded wait; promotion is never claimed.
7. **True eval cost + smoke budget.** One cycle = **27 eval cells**
   (PRE 12, MID 6, POST 9 incl. 3 q_post), not 5. Launcher default is now
   `MSA_NUM_SEEDS=4` — an EXECUTION SMOKE for machinery validation only,
   never a method conclusion; N=16 pilot is explicit opt-in; N=256 stays
   formal-only. Evals run over `--eval-devices` (default since §5d:
   `cuda:4,cuda:5`, shared with the SIGSTOPped GPU4/5 trainers) with a
   bounded scheduler (at most ONE Isaac process per device, deterministic
   round-robin, fail-closed cancellation). `EVAL_COST_ESTIMATE.json`:
   N=4/2 GPUs ≈ 5.4 h; N=16/2 GPUs ≈ 21.6 h; N=256/2 GPUs ≈ 345.6 h
   (unacceptable without further optimization).

## 5b. Orchestration blocker patch ("Close MSA smoke, commit, and cap races")

Three last pre-pilot orchestration blockers, no algorithm changes:

1. **Enforced hold smoke gate.** The formal sidecar's cycle gate
   (`cycle_gate_ok`) now requires BOTH `FIRST_TICK_VALIDATED.json` and
   `HOLD_SMOKE_PASSED.json` (new required `--hold-smoke-marker`); it can
   be launched early but idles without touching `MSA_HOLDS.json`.
   `hold_smoke` writes the PASSED marker atomically ONLY when the frame
   stayed frozen, the next committed tick appeared after release and
   residual holds are empty; any failure writes `HOLD_SMOKE_FAILED.json`
   (also a permanent sidecar stop marker), SIGCONTs everything and
   fail-closed stops the whole arm.
2. **Durable COMMITTING phase (DONE-before-commit fixed).** After DECIDE
   the controller persists `status=COMMITTING` + `decision` +
   `final_tol_by_worker` + all-false `commit_acks_by_worker` BEFORE any
   durable commit; workers are then released and committed IN PARALLEL
   (one bounded wait per worker — serial waits would let later workers
   train extra ticks), each ack persisted immediately. Only with 6/6 acks
   does the cycle become DONE and the machine return to NORMAL. A sidecar
   restart into COMMITTING runs `_resume_committing`: no re-eval, no
   re-training, idempotent commits for un-acked workers only. A commit
   timeout attaches the ack table to `CommitTimeout`; the sidecar writes
   it into `COMMIT_TIMEOUT.json`, never claims promotion, and stops the
   arm without a second full wait.
3. **Guard/watchdog coordination via `MSA_BUSY.json`.** The controller
   maintains the atomic busy marker (intervention_id, phase, status,
   due_workers, timestamp) from BEFORE the first PRE_EVAL hold until DONE
   or failure-cleanup completion (a hard kill leaves it in place). While
   it exists the guard defers +100M park/cap actions as `cap_pending`
   (no stop_worker, no SIGCONT of held workers) and re-evaluates on fresh
   committed metas after it clears. `stop_worker` now SIGCONTs ONLY the
   target worker's held PIDs; `stop_arm` remains the only all-holds
   release. The watchdog stops the arm on a dead sidecar whenever holds
   OR the busy marker exist (busy-with-empty-holds = death mid-COMMITTING).

## 5c. State-faithful stall bootstrap ("Add state-faithful stall bootstrap for MSA arms", 2026-07-19)

The GPU execution smoke invalidated the bare warm-start bootstrap: with an
empty `curriculum_control/` at worker start the env fell back to the ungated
LEGACY curriculum and worker 0 shrank 0.0231 → 0.0168 within the first
committed interval — the stall was NOT reproduced, so no arm built on that
bootstrap can support the causal MSA question. This pass replaces it:

1. **`msa/state_bootstrap.py` (new)** — before any worker starts it (a)
   copies each worker's source control JSON into the new workspace with
   every base-retention gate field intact and ONLY `msa_*` cleared; (b)
   rebuilds the observer's unpersisted in-memory state (rolling window from
   the last 12 committed metas, recent peak, collapse/safety/WM freeze
   horizons, last-mastered records, rollback counters,
   `shrinks_since_rollback` from meta history) into
   `observer_state_seed/<w>.json`; (c) copies the resume snapshots into the
   new policy dirs; (d) writes `STATE_FIDELITY.json` (per worker, per field:
   source vs restored vs category) and `STATE_FINGERPRINT.json` (sha256 of
   the restored state + every consumed source file); (e) FAILS CLOSED on any
   missing critical state. Full source-of-truth analysis:
   `STATE_SOURCE_AUDIT.md`.
2. **Observer seed loading** — `pbt_observer._maybe_apply_observer_state_seed`
   restores the seeded state once per process at the first curriculum tick,
   only inside the validity window (tip + 2 ticks) and never regressing live
   state; ordinary runs without the seed dir are bit-for-bit unchanged.
3. **Structured resume audit** — train.py writes
   `policy_XX/RESUME_APPLIED.json` (what tolerance/clock resume was actually
   applied to the env); the guard validates THIS instead of requiring the
   live first-tick tolerance to still equal the manifest tip.
4. **Stall reproduction gate** (in `continuation_guard`) — the first
   committed interval doubles as the reproduction interval: per worker at
   most ONE legal mastery shrink (two shrinks in one interval is the
   legacy-cascade signature → `GUARD_FAILED`), restored control state must
   not be default-clobbered (`verify_control_fidelity`: last-mastered ≠
   0.075-clobber, peak preserved, freeze horizons kept), and arm-level
   ≥ 4/6 workers must hold EXACTLY their source tolerance —
   otherwise `STATE_FIDELITY_FAILED.json` + full arm stop, MSA never starts.
5. **Env clock resume** — `launch_pbt --curriculum-clock-resume-per-policy`
   ("lcu:fsr" per policy, from the source resume snapshots) →
   `--last_curriculum_update_resume` / `--frame_since_restart_resume`, so
   `post_shrink_consolidation_until_restart_step` keeps its source meaning.
6. **Real committed cadence** — measured 80M frames/tick (stage-3 SELECTION:
   20M base × 4.0 mult; 40M in stage 2), so: `PROBE_PACKET_FRAMES` = 80M,
   cap slack = 80M, continuation budget = 6 ticks = 480M/worker
   (1 reproduction + 2 probe cycle + 2 observation + 1 slack),
   `max_iterations` = 44,500 + 4,280. Never assume the 20M pbt_interval is
   the committed quantum.
7. **Escalating stops** — every guard stop path now runs
   SIGCONT → SIGTERM → 20 s grace → re-enumerate the tree → SIGKILL
   survivors → verify exit (zombies excluded); the smoke had shown Isaac
   trainers surviving bare SIGTERM.
8. **Four stateful arms** — `BASE_STATEFUL` (matched "still stuck" control),
   `GATE_RELAX_STATEFUL`, `MSA_STATEFUL`, `MSA_PERMUTED_STATEFUL`; same six
   tip checkpoints, same full control state, same seed; launchers run the
   bootstrap + fingerprint verification fail-closed before anything starts.

## 5d. Experiment synchronization ("Make readiness a hold barrier with sidecar adoption", 2026-07-19)

The state-faithful pass still let a validated worker keep training while
the others caught up, so the six workers were NOT at matched intervention
boundaries when `FIRST_TICK_VALIDATED.json` appeared. This pass fixes the
last synchronization problems without touching the MSA algorithm,
tau_plus, decision gates or the state reconstruction:

1. **Readiness is a TRUE BARRIER** — the moment a worker passes
   RESUME_APPLIED validation + control fidelity + its first-80M stall
   reproduction checks, the guard SIGSTOPs its FULL process tree
   (children first, verified `T` state) and persists
   `READINESS_HOLDS.json` (worker id, root/children PIDs, validated
   frame/tolerance/lineage, hold timestamp). The freeze happens BEFORE
   the validation read is finalized (freeze → re-read committed meta →
   validate), so no tick can slip between validation and the hold.
   Validated workers stay frozen until all six are ready; any failure
   SIGCONTs every readiness hold (all stop paths cover both
   `MSA_HOLDS.json` and `READINESS_HOLDS.json`) and stops the arm. The
   population stall-reproduction gate and `FIRST_TICK_VALIDATED.json`
   only run/appear at 6/6.
2. **Distinct handoffs** — `BASE_STATEFUL`/`GATE_RELAX_STATEFUL`: all six
   SIGCONTed SIMULTANEOUSLY, `READINESS_HOLDS.json` deleted (synchronized
   continuation from each validated tick). `MSA_STATEFUL`/
   `MSA_PERMUTED_STATEFUL`: the guard retains the holds; the sidecar
   ADOPTS them (`adopt_readiness_holds`: PIDs still `T`, committed frame
   == the validated frame in the marker, imports `WorkerHolds` state,
   atomically converts to `MSA_HOLDS.json`) and starts `PRE_EVAL` from
   exactly that frozen boundary — the controller's `_snapshot_at` then
   enforces PRE snapshot frame == that frame. Any adoption mismatch
   releases everything and stops the arm (`GUARD_FAILED`).
3. **Disposable GPU hold smoke** — the manual in-arm hold smoke releases
   the target worker and waits for one extra committed tick, which breaks
   the matched intervention boundary. It is now EXECUTION SMOKE ONLY
   (default worker 4, exercising the shared-GPU path). The formal gate is
   `start_stateful_hold_smoke.sh`: a one-shot throwaway worker (same
   code, Isaac task, wm profile, 8192 envs, GPU 4) verified under
   SIGSTOP/SIGCONT, producing `HOLD_SMOKE_CERTIFICATE.json` (git commit,
   task/profile, GPU UUID/model, CUDA/driver, process-tree shape,
   hold/resume frames, ok=true) and then torn down completely. The formal
   MSA launchers refuse to start without the certificate file and the
   sidecar re-validates its contents (commit/task/profile/GPU
   model/driver) against the live environment at startup, fail closed.
4. **Stop order** — `_terminate_tree` signals descendants children-first
   and the supervisor/root LAST for SIGTERM and SIGKILL alike, with grace
   → re-enumeration → SIGKILL → repeated re-enumeration inside the verify
   window (accumulating every PID ever seen), so a supervisor respawning
   children mid-escalation cannot leak a survivor (regression-tested with
   a `trap '' TERM` respawn loop).
5. **Six-GPU configuration** — trainers on `cuda:0`–`cuda:5`; eval on
   `MSA_EVAL_DEVICES` (default `cuda:4,cuda:5`), SHARED with trainers 4/5
   which are SIGSTOPped during eval barriers but keep their VRAM
   allocated. The previous "trainers then share 0–3" wording was wrong
   and is removed everywhere. The launcher records the actual devices in
   `<run_dir>/EVAL_DEVICES.json`.

## 5e. Watchdog + shared-GPU preflight ("Close readiness watchdog and shared-GPU preflight gaps", 2026-07-19)

The last three formal-experiment blockers, orchestration-only (no MSA
algorithm, no training parameters, no bootstrap/controller/evaluator
refactor):

1. **Caller-safe `stop_arm` (P0-1).** The old order terminated
   `msa_sidecar.pid` / `hold_watchdog.pid` BEFORE the workers — but
   stop_arm is called BY the sidecar (certificate/adoption/commit-timeout
   fail paths) and BY the watchdog, so a self-referential pid file killed
   the caller mid-cleanup and worker termination could never run. New
   order: SIGCONT all MSA + readiness holds → terminate all six worker
   trees (full children-first escalation) → terminate the OTHER control
   processes, skipping any pid equal to `os.getpid()` (the caller exits
   naturally after stop_arm returns) → delete both holds files (no stale
   ownership). `_terminate_tree` gained `skip_pids` so the caller is
   protected even if it appears inside an enumerated tree. Verified with
   REAL subprocess callers registered as sidecar/watchdog pids: workers
   die, the sibling control process dies, the caller reaches its
   post-stop_arm completion marker.
2. **Readiness-phase watchdog (P0-2).** `hold_watchdog` now reads
   `READINESS_HOLDS.json`, `MSA_HOLDS.json`, `FIRST_TICK_VALIDATED.json`,
   `continuation_guard.pid` AND `msa_sidecar.pid`, and is launched for
   EVERY stateful arm (BASE/GATE included) by the main launcher body.
   While readiness holds exist it never idle-exits. Before the marker the
   GUARD owns the freeze: if it dies, the watchdog SIGCONTs every
   readiness-held pid, writes `GUARD_DIED_DURING_READINESS.json` and
   stops the arm (caller-safe). After the marker but before adoption
   (MSA pre-adoption window) the SIDECAR is the owner: its death equally
   releases the readiness holds and stops the arm. After adoption the
   existing MSA_HOLDS/MSA_BUSY logic applies unchanged. A holds file that
   exists but cannot be parsed is NEVER treated as empty (fail closed:
   dead owner → stop the arm); an owner pid file that has not appeared
   yet is tolerated for a bounded startup grace (`--startup-grace-s`,
   default 120 s) before it counts as dead.
3. **Shared-GPU eval coexistence smoke (P1).** The disposable certificate
   smoke previously proved only SIGSTOP/SIGCONT; it now ALSO proves that
   an Isaac eval can run on the SAME GPU while the frozen trainer holds
   its VRAM: after the frozen-frame check and BEFORE release, it launches
   one real policy-only eval cell through the production `eval_bridge`
   (same task, the trainer's own committed checkpoint frozen read-only,
   N=1 seed, `eval_num_envs=512`, `CUDA_VISIBLE_DEVICES` pinned to the
   smoke GPU) and requires a clean exit — any CUDA OOM or Isaac startup
   failure fails the certificate. New certificate fields:
   `eval_coexistence_ok`, `eval_device`, `eval_num_envs`,
   `eval_exit_code`, `eval_error`, `gpu_memory_before_hold`,
   `gpu_memory_while_trainer_frozen`, `gpu_memory_during_eval` (sampled
   peak). `certificate_ok` (the formal sidecar's startup gate) now
   REJECTS any certificate without `eval_coexistence_ok=true`, so old
   certificates are invalid and the disposable smoke must be re-run. The
   eval output is an execution certificate only — never a method result.

## 5f. Bounded coexistence eval + guard lifetime supervision ("Bound coexistence eval and supervise guard lifetime", 2026-07-19)

Final orchestration patch before the disposable GPU certificate smoke
(no MSA algorithm, tau_plus, rho threshold, routing, promotion gate,
training-parameter or bootstrap-semantics change):

1. **Coexistence eval: no PIPE deadlock, no infinite wait (A).** The old
   `_run_coexistence_eval` used `stdout=PIPE, stderr=PIPE` with NOBODY
   consuming them — an Isaac eval writing more than the ~64KB pipe buffer
   would block forever, and there was no wall timeout at all. New
   `_run_bounded_logged` core: stdout+stderr append to
   `<disposable run_dir>/cert_eval.log` (a file, nothing to consume);
   the evaluator starts with `start_new_session=True` (own process
   group); a hard `--cert-eval-timeout-s` (default 1800 s) SIGTERMs the
   WHOLE process group on expiry, waits a bounded grace, SIGKILLs the
   group, and verifies no survivor among every pid ever enumerated under
   the child. VRAM peak sampling is retained. New certificate fields:
   `eval_duration_s`, `eval_timed_out`, `eval_log_path`,
   `eval_error_tail` (log tail on failure), `eval_survivor_pids`,
   `eval_timeout_s`. Timeout, non-zero exit or a residual evaluator all
   yield `ok=false` + `eval_coexistence_ok=false`; `certificate_smoke`'s
   `finally` releases the trainer hold on success, failure, exception
   AND timeout (CPU-tested with a real SIGSTOPped trainer).
2. **Watchdog supervises the guard's WHOLE lifetime (B).** Previously
   guard death was only detected while readiness holds existed; a guard
   dying after the barrier left the arm without cap/gate enforcement and
   nobody noticed. Now, after the startup grace, `continuation_guard.pid`
   is monitored continuously for ALL FOUR stateful arms: as long as
   `ARM_COMPLETE.json` has not appeared, a missing or dead guard pid is
   fatal — the watchdog SIGCONTs the UNION of readiness and MSA holds,
   writes `CONTINUATION_GUARD_DIED.json` (guard pid, readiness-marker
   presence, READINESS_HOLDS / MSA_HOLDS / MSA_BUSY contents, the six
   workers' latest committed frames, timestamp) and caller-safely stops
   the workers and the sidecar. Guard death mid-barrier additionally
   keeps the readiness-specific `GUARD_DIED_DURING_READINESS.json`
   record. `ARM_COMPLETE.json` marks normal completion: after it the
   guard's exit is expected and never reported (quiet idle_exit). A dead
   sidecar with nothing frozen no longer idle-exits the watchdog — the
   guard is still supervised until ARM_COMPLETE.
3. **`stop_arm` cleans the guard sibling (C).** The control-pid cleanup
   set is now `continuation_guard.pid` + `msa_sidecar.pid` +
   `hold_watchdog.pid`, with the unchanged caller-safe order: SIGCONT all
   holds → write the failure marker → stop all six worker trees → stop
   the OTHER control processes (skip `os.getpid()`; the caller returns
   naturally) → clear stale readiness/MSA holds files. Verified with real
   subprocess callers registered as sidecar, watchdog AND guard pids.

## 5g. Source lineage propagation ("Propagate source lineage through stateful bootstrap", 2026-07-20)

Root cause found by the first N=16 pilot: the guard's first-tick
validation requires the committed meta's `lineage_id` to equal the
manifest lineage, but a per-policy warm start never propagated it — a
fresh workspace has no prior meta for the observer's fallback, so the
observer minted a FRESH uuid and the guard (correctly, per spec:
"lineage unchanged") stopped the arm with `GUARD_FAILED`. The same pilot
proved the tolerance side of the stall reproduction works: worker 0's
first committed tick held EXACTLY the source tolerance
(0.023112487509668322).

Fix (pure bootstrap plumbing, no algorithm change):

1. **`launch_pbt.py`**: new `--lineage-resume-per-policy`
   (`lineage_id:lineage_start_pf:weight_age_frame` triples, fail-closed
   parser, requires `--tolerance-resume-per-policy`). Warm-started
   policies now receive the pre-existing train.py flags
   `--pbt_lineage_id_resume` / `--pbt_lineage_start_pf_resume` /
   `--pbt_weight_age_frame_resume`, which the observer already consumed
   on supervisor restarts.
2. **`arms.py`**: `_lineage_list_from_tips` builds the list from each
   tip checkpoint's sibling meta JSON, failing closed on a missing meta,
   a missing field, or a meta/manifest lineage mismatch. Wired into all
   four stateful arm launchers (`LINEAGE_LIST`) and the disposable
   certificate smoke launcher.

## 5h. Budgeted worker-local MSA promotion ("Add budgeted worker-local MSA promotion", 2026-07-20)

Pilot evidence archived under `results/analysis/msa_stateful_n16_20260720/`
(POSTMORTEM + compact tables; raw eval sha256-only).

**Budget admission** (`msa/budget.py`): hold raw-due workers first,
refresh committed metas, then `admit_cycle_budget` requires
`remaining = cap_target - frozen_committed_frame >= worst_case_remaining_cycle_dose()`
where
`reserve = 2*(packet+tick) + tick` (= 400M when packet=tick=80M:
COMMON_PROBE + quant, SECOND_PROBE + quant, COMMITTING confirm tick).
Ineligible workers are released immediately and never enter PRE_EVAL /
MSA_BUSY / snapshot / pin; all-ineligible → release all holds, clear
`CYCLE_BUDGET_RESERVATION.json`, write `MSA_NO_NEW_CYCLE.json` (no
GUARD_FAILED). Eligible workers stay frozen; frozen frames are
`start_frames` and reservation ceilings. Observed cycle dose is
`final_committed_frame - frozen_start` after commit acks (includes the
confirmation tick) and fails closed vs reserved ceiling / dose / cap.

Cap arithmetic: readiness consumes 80M of the 480M continuation cap, so
worst-case 400M reserve means the current `MSA_SELECTIVE_STATEFUL` arm is
a **one-step** selective causal pilot (at most one new cycle). It must
not be described as already iterating from tol≈0.02 to 0.01.

**Selective promotion** (`promotion_mode=worker_local_selective`):
new arm `MSA_SELECTIVE_STATEFUL`. Eligible workers need p_pre≥0.01,
worker L_mid lower quantile >0, ρ≥0.88; select top-3 by
(-L_mid_lo, -ρ, worker_id). `anchor_workers` are population members
**not selected this cycle** (not a permanent cross-cycle cohort). Panel
statistic is diagnostic only. `MSA_STATEFUL` keeps the **global_panel
decision rule** unchanged; if that arm shares budget orchestration, do
not claim the entire run is bit-for-bit identical.

## 6. Known limitations / open points before launch

- **SIGSTOP holds are not GPU-verified.** Pausing a live Isaac trainer
  with SIGSTOP is expected to be safe for single-GPU rl_games workers (no
  NCCL peers). Run the DISPOSABLE `start_stateful_hold_smoke.sh` FIRST —
  this is now ENFORCED: the formal MSA launcher refuses to start without
  `HOLD_SMOKE_CERTIFICATE.json` and the sidecar validates it (commit,
  task, profile, GPU model, driver) fail-closed at startup.
- ~~If the guard process dies while workers are readiness-frozen, the
  frozen workers stay stopped~~ — CLOSED in §5e: the hold watchdog now
  covers the readiness barrier and the MSA pre-adoption window for every
  stateful arm (SIGCONT + `GUARD_DIED_DURING_READINESS.json` + caller-safe
  arm stop). §5f extends this to the guard's WHOLE lifetime: any guard
  death before `ARM_COMPLETE.json` writes `CONTINUATION_GUARD_DIED.json`
  and stops the arm; guard exit after ARM_COMPLETE is never a false
  positive.
- The disposable certificate binds to (git commit, task, profile, GPU
  model, driver) — any code change or driver update invalidates it and
  the disposable smoke must be re-run (~one committed interval of GPU
  time on GPU 4). Certificates produced BEFORE the §5e coexistence
  extension lack `eval_coexistence_ok` and are rejected by design.
- The shared-GPU eval coexistence proof itself is GPU work: it has NOT
  been executed in this pass (only the launcher and the fail-closed gate
  are CPU-tested). Re-running `start_stateful_hold_smoke.sh` is the
  single next allowed GPU command.
- The durable tolerance commit waits for a live committed tick (~20M
  frames) per transition; a full cycle therefore spans several ticks of
  wall time. The bounded wait (`--commit-max-wait-s`, default 1 h) stops
  the arm instead of hanging on a dead trainer.
- Training GPUs idle during eval barriers: with N=4 that is ~5.4 h of
  paused training per cycle across the eval waves; acceptable for the
  execution smoke, a real cost consideration for N=16.
- `snapshot_pre/mid/post` freeze the newest COMMITTED checkpoint
  (LATEST.commit); PBT save cadence defines the snapshot lag (≤ one tick,
  now provably aligned with the packet boundary via the post-hold read).
- The known pre-existing environment segfault when several heavy profile
  test suites are collected in one pytest process (and the pre-existing
  `test_l2_usable_when_stable_fails` failure) reproduce identically at the
  parent commit; they are unrelated to this change (see `TEST_REPORT.md`).

## 7. Artifacts

- code: `dexpbt_isaaclab/msa/` (10 modules incl. `hold_watchdog.py` and
  `state_bootstrap.py`), `tests/test_msa_probe.py` (143 tests),
  `tests/test_state_bootstrap.py` (32 tests)
- launchers: `results/launchers/msa_probe/` (4 STATEFUL arm scripts +
  4 configs + `start_stateful_hold_smoke.sh` (disposable certificate
  smoke) + `run_hold_smoke.sh` (execution smoke only) +
  `EVAL_COST_ESTIMATE.json`)
- audit: `blocker_audit_3b_5b.json`, `STATE_SOURCE_AUDIT.md`
  (this directory)
- docs: `STATE_MACHINE.md`, `CONFIG_DIFF.md`, `DRY_RUN_TRACE.md`,
  `TEST_REPORT.md`, `LAUNCH_CHECKLIST.md`, `EVAL_COST_ESTIMATE.json`
  (this directory)
