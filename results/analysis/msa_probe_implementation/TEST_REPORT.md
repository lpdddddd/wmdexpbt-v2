# base_retention_msa_probe — Test Report

Date: 2026-07-19, updated the same date for the pilot-integration fix pass
("Fix MSA pilot integration and phase safety"), the pre-pilot safety patch
("Harden MSA readiness holds and pilot budget"), the final orchestration
blocker patch ("Close MSA smoke, commit, and cap races"), the
state-faithful bootstrap pass ("Add state-faithful stall bootstrap for MSA
arms"), the synchronization pass ("Make readiness a hold barrier with
sidecar adoption"), the watchdog/preflight pass ("Close readiness
watchdog and shared-GPU preflight gaps"), the final orchestration
patch ("Bound coexistence eval and supervise guard lifetime") and the
lineage propagation fix ("Propagate source lineage through stateful
bootstrap", 2026-07-20).
Environment: isaaclab311 python
(CPU only; no Isaac import anywhere in the MSA package or its tests; no
GPU used).

Three evidence tiers, kept separate on purpose:

1. **Unit tests** — pure-python contracts (this file, sections below).
2. **Production-style simulation** — heterogeneous-tau full-cycle dry run
   against a tick-quantized parallel fake trainer with held-worker
   assertions and crash injection (`DRY_RUN_TRACE.md`).
3. **GPU validation** — NOT performed (no GPU tasks were run). SIGSTOP
   holds and live commit-wait behavior remain GPU-unverified; see
   `LAUNCH_CHECKLIST.md`.

## MSA suite: `tests/test_msa_probe.py` — 143/143 passed
## Bootstrap suite: `tests/test_state_bootstrap.py` — 37/37 passed

Lineage propagation fix (2026-07-20): 5 new tests in
`TestLineageResume` — `parse_lineage_resume_per_policy` valid/reject
shapes, `_lineage_list_from_tips` happy path + fail-closed (meta missing,
field missing, meta/manifest lineage mismatch), and a source-level check
that `launch_pbt` injects `--pbt_lineage_id_resume` /
`--pbt_lineage_start_pf_resume` / `--pbt_weight_age_frame_resume` on
per-policy warm starts. Root cause covered: a bare per-policy
`--checkpoint` mints a FRESH lineage uuid at the first committed tick
(fresh workspace = no prior meta for the observer fallback), which the
continuation guard rejects against the manifest lineage — the 2026-07-20
pilot `GUARD_FAILED`. Launcher-content assertions extended to require
`--lineage-resume-per-policy` in all stateful arm launchers and the
disposable certificate smoke launcher.

(72 retained from the pilot-integration pass + 23 safety-patch tests +
15 orchestration-blocker tests, updated in place for the real 80M cadence,
the 480M cap and the stateful arm names + 4 first-tick semantics tests +
11 synchronization tests — readiness hold barrier, sidecar adoption,
certificate gate, children-before-root termination, six-GPU metadata +
10 watchdog/preflight tests — caller-safe stop_arm (real subprocess
callers), readiness-phase watchdog coverage, fail-closed corrupt holds,
eval-coexistence certificate gate + 8 final-patch tests — bounded/logged
coexistence eval (no PIPE deadlock, process-group timeout kill, hold
release on timeout), guard-lifetime watchdog for all four stateful arms,
guard-as-caller stop_arm; plus the 32-test state-bootstrap suite, whose
guard-loop tests run against REAL per-worker processes frozen by the
barrier.)

Original 15 required behaviors (spec section J) — retained:

| # | required behavior | test(s) |
|---|---|---|
| 1 | tau=0.010 never probed | `TestDueLogic.test_tau_at_floor_never_due`, `test_tau_plus_floor_clamp` |
| 2 | missing-provenance resume immediately due | `TestDueLogic.test_missing_provenance_immediately_due`, `test_from_dict_none_stays_none` |
| 3 | probe at most once per 100M worker-local frames | `TestDueLogic.test_probe_at_most_once_per_window` |
| 4 | evaluator never modifies live state | `TestEvaluatorIsolation.*` (read-only snapshot enforcement + AST scan; control writes limited to the five `msa_*` fields — the observer-owned `forced_success_tolerance` is never written) |
| 5 | pre/mid tolerance or seed-manifest mismatch fails closed | `TestPairedEvalContracts.test_manifest_mismatch_fails_closed`, `test_tolerance_mismatch_fails_closed`, `test_seed_order_mismatch_fails_closed` |
| 6 | matched common dose for all workers (now one 20M committed tick) | `TestHeterogeneousDryRunCycle.test_common_exposure_parallel_exactly_one_packet` |
| 7 | second-half top-3 deterministic tie-break | `TestRouting.test_top3_stable_tie_break` |
| 8 | permuted route keeps total dose identical | `TestRouting.test_permuted_route_same_dose`, `TestHeterogeneousDryRunCycle.test_permuted_arm_same_dose_different_mapping` |
| 9 | q_post never enters common-dose L_mid | `TestHeterogeneousDryRunCycle.test_q_post_not_in_common_dose` |
| 10 | p_pre < 0.01 → retention unavailable, promotion blocked | `TestPairedEvalContracts.test_low_p_pre_retention_unavailable` |
| 11 | rho compares same tau only | `TestPairedEvalContracts.test_rho_same_tau_only`, `TestBootstrap.test_cross_tolerance_gate_names_rejected` |
| 12 | bootstrap deterministic (fixed PRNG) | `TestBootstrap.test_bootstrap_deterministic`, `test_bootstrap_is_labeled_criterion_local_conditional`, `test_negative_panel_blocks_promotion` |
| 13 | no optimizer/PBT-lineage reset through probe | `TestHeterogeneousDryRunCycle.test_no_resets` + isolation AST scan |
| 14 | frame ledger includes all probe frames (observed) | `TestHeterogeneousDryRunCycle.test_observed_frames_and_ledger` |
| 15 | original base-retention behavior unchanged | `TestOriginalProfileUnchanged.*` (inert defaults incl. `msa_forced_success_tolerance`; legacy control JSON round-trip; msa_enabled=False curriculum shrink bit-identical; GATE_RELAX vs BASE launcher diff exactly one line) |

New pilot-integration coverage (this fix pass):

| requirement | test(s) |
|---|---|
| six DIFFERENT taus, full cycle, worker-local tau_plus | `TestHeterogeneousDryRunCycle.test_worker_local_tolerances` (FakeWorld raises on any foreign-tolerance eval) |
| common exposure parallel (a serial loop would 6x the dose) | `test_common_exposure_parallel_exactly_one_packet` |
| top-3 second segment parallel; non-recipients train at their own tau | `test_second_half_parallel_and_non_recipients_reverted` |
| non-recipients re-pinned to own tau immediately after MID | `test_non_recipient_repinned_to_own_tau_after_mid` |
| real 20M committed-tick packet accounting, observed frames | `test_common_exposure_parallel_exactly_one_packet`, `test_observed_frames_and_ledger` |
| frames never advance during eval (phase barrier) | held-assertions in `FakeWorld.snapshot/evaluate/train_until` + `test_frames_do_not_advance_during_eval` |
| crash recovery in every phase (pins cleared, own tau restored, FAILED + ABORT recorded) | `TestCrashRecovery.test_crash_in_pre_eval` / `_common_probe` / `_mid_eval` / `_post_eval`, `test_failed_cycle_not_auto_resumed` |
| state machine + provenance real round-trip through the persisted store | `test_state_persisted_roundtrip`, `TestStateMachine.test_abort_from_any_phase`, `TestDueLogic.test_intervention_record_replaced_not_duplicated` |
| restart resume: hard kill → resume → completed segments NOT re-trained, snapshots reused | `TestRestartResume.test_resume_after_hard_kill_in_mid_eval` |
| concurrent control updates: no lost update between sidecar and observer | `TestConcurrentControlUpdates.test_no_lost_updates` (threaded flock stress), `test_mark_shrink_consumed_preserves_msa_fields` |
| non-6-decimal tau evaluation match (canonical .12g) | `TestCanonicalTolerance.test_canonical_roundtrip`, `test_non_six_decimal_tau_matches_own_request`, `test_six_decimal_rounding_no_longer_matches_silently`, `test_ccrs_unit_key_distinguishes_close_taus` |
| continuation first-tick validation (tolerance / frame / lineage fail-closed) | `TestContinuationGuard.test_first_tick_*` |
| +100M hard frame cap | `TestContinuationGuard.test_hard_cap_100m` |
| bootstrap manifest hash + worker-mapping verification | `TestContinuationGuard.test_bootstrap_verify_hash_and_mapping` |
| promotion: worker-local tau_plus durably committed, resume snapshot synced, pin (forced override) == 0 | `TestHeterogeneousDryRunCycle.test_promotion_commits_worker_local_tau_plus_and_clears_pins` |
| stay decision restores worker-local tau durably | `test_stay_decision_restores_worker_local_tau` |
| base-retention msa_enabled=False behavior unchanged | `TestEnvPinBehavior.test_msa_disabled_pin_is_inert`, `TestOriginalProfileUnchanged.test_msa_disabled_curriculum_behavior_unchanged` |
| MSA pin applies only when enabled and outranks observer forced | `TestEnvPinBehavior.test_msa_enabled_pin_applies_and_blocks_shrink`, `test_msa_pin_has_priority_over_observer_forced` |
| GATE_RELAX still exactly one 0.88→0.75 profile difference | `TestOriginalProfileUnchanged.test_gate_relax_goes_through_dedicated_profile`, `test_base_continuation_launcher_keeps_original_gates` |
| launchers: guard preflight, tolerance resume, eval devices, smoke N | `test_launchers_have_guard_tolerance_resume_and_eval_device`, `TestEvaluatorIsolation.test_eval_device_required`, `TestArmConfigs.test_arm_configs_matched_and_charged` |

Pre-pilot safety-patch coverage ("Harden MSA readiness holds and pilot
budget"; 23 new tests):

| requirement | test(s) |
|---|---|
| partial worker availability never starts a cycle | `TestReadinessBarrier.test_guard_loop_partial_workers_no_marker`, `test_sidecar_readiness_gate` |
| readiness marker requires 6/6 validated workers, atomic, full contents | `test_readiness_marker_requires_6_of_6`, `test_marker_contents_and_atomic_write`, `test_guard_loop_writes_marker_only_after_all_six` |
| any first-tick failure → GUARD_FAILED, no readiness marker | `test_guard_loop_any_failure_stops_arm_without_marker` |
| tick between due detection and hold NOT counted as tau_plus exposure | `TestHoldRaceStartFrames.test_stale_tick_not_counted_as_tau_plus` |
| snapshot frame == packet start / mid boundary frame | `test_snapshot_frame_equals_packet_start_frame` (+ controller `_snapshot_at` enforcement in every cycle test) |
| hold verifies stopped state; MSA_HOLDS.json persisted/removed | `TestWorkerHoldsSafety.test_hold_verifies_stopped_and_persists` (real SIGSTOPped process), `test_hold_verification_failure_fails_closed` |
| release order children before supervisor/root | `test_release_order_children_before_root` |
| sidecar death watchdog releases held workers + stops arm | `TestHoldWatchdog.test_dead_sidecar_with_holds_releases_and_stops` (real stopped process resumed), `test_sidecar_alive_is_ok`, `test_dead_sidecar_no_holds_keeps_watching_until_arm_complete` |
| normal +100M cap → ARM_COMPLETE.json (6/6 parked, no failure marker) | `TestMatchedCap.test_all_six_parked_writes_arm_complete` |
| cap never lets a worker run to ~140M | `test_overshoot_to_140m_fails_the_arm`, `TestContinuationGuard.test_hard_cap_100m` (run/park/fail) |
| commit timeout: ONE bounded wait per worker (parallel), no re-wait in cleanup, tau pin restored, ack table attached | `TestCommitTimeoutBounded.test_commit_timeout_single_bounded_wait` |
| 2-device eval scheduler: max concurrency 1 per device, round-robin | `TestEvalScheduler.test_round_robin_and_max_one_per_device` |
| eval job failure cancels remaining jobs (fail closed) | `test_failure_cancels_remaining_and_raises`, `test_device_validation` |
| 27 eval cells cost math (5.4 h / 21.6 h / 345.6 h on 2 GPUs) | `TestEvalCostEstimate.test_27_cells_and_wall_times`, `test_cost_estimate_file_generated` |
| N=4 launcher default (execution smoke), readiness/commit flags, watchdog | `test_launchers_have_guard_tolerance_resume_and_eval_device` |
| hold smoke end-to-end (frozen frame, release, next tick, result JSON) | `TestHoldSmoke.test_hold_smoke_end_to_end` (real SIGSTOP on a live process) |
| base-retention msa_enabled=False behavior unchanged | retained `TestEnvPinBehavior.*`, `TestOriginalProfileUnchanged.*` |

Orchestration-blocker coverage ("Close MSA smoke, commit, and cap races";
15 new tests):

| requirement | test(s) |
|---|---|
| smoke success alone writes HOLD_SMOKE_PASSED.json (atomic; also asserts no FAILED marker, no tmp files) | `TestHoldSmoke.test_hold_smoke_end_to_end` (extended) |
| smoke failure: FAILED marker, PIDs SIGCONTed, whole arm fail-closed stopped (waiting sidecar SIGTERMed), PASSED never written | `TestHoldSmoke.test_hold_smoke_failure_stops_arm_never_writes_passed` |
| no HOLD_SMOKE_PASSED → formal sidecar gate stays closed even with 6/6 readiness | `TestHoldSmokeGate.test_no_smoke_marker_gate_closed` |
| smoke marker alone (no readiness) still not sufficient | `test_smoke_marker_alone_not_sufficient` |
| HOLD_SMOKE_FAILED.json is a permanent sidecar stop marker | `test_smoke_failed_is_a_sidecar_stop_marker` |
| crash after DECIDE, before FIRST commit: persisted COMMITTING + all-false acks; restart commits ONLY (no re-eval, no re-train), then DONE | `TestCommittingPhase.test_committing_persisted_before_any_commit` |
| crash after PARTIAL acks: restart commits only un-acked workers (exactly 2 new attempts) | `test_partial_acks_resume_commits_only_unacked` |
| every persisted DONE carries a full-true ack table (no DONE-before-durable-commit) | `test_done_never_written_before_all_acks` |
| MSA_BUSY set before the FIRST hold, phase/status carried (incl. COMMITTING), cleared only at DONE | `TestBusyMarkerLifecycle.test_busy_set_before_first_hold_and_cleared_at_done` |
| soft failure cleanup clears busy; hard kill leaves it for the watchdog | `test_soft_failure_cleanup_clears_busy`, `test_hard_kill_leaves_busy_for_watchdog` |
| cap during PRE/MID/POST holds (busy present): no park, no fail, held worker untouched (real SIGSTOPped process stays 'T') | `TestGuardMSACoordination.test_cap_during_hold_defers_and_touches_nobody` |
| cap during COMMITTING: waits for busy clear, then re-reads metas and parks (ARM_COMPLETE) | `test_cap_waits_for_busy_clear_then_parks` |
| stop_worker SIGCONTs ONLY the target worker's held PIDs | `test_stop_worker_sigconts_only_target` (worker 1's process stays stopped) |
| sidecar dead + busy=true + holds EMPTY → watchdog writes MSA_SIDECAR_DIED.json and stops arm | `TestWatchdogBusyNoHolds.test_dead_sidecar_busy_no_holds_stops_arm` |
| sidecar dead + no busy + no holds → still a quiet idle_exit | `test_dead_sidecar_no_busy_no_holds_still_idle` |

State-faithful bootstrap coverage ("Add state-faithful stall bootstrap for
MSA arms"; 32 new tests in `tests/test_state_bootstrap.py` + 4 updated
first-tick tests in the MSA suite):

| requirement | test(s) |
|---|---|
| control JSON copied with every gate field intact, ONLY msa_* cleared | `TestStateBootstrap.test_control_copied_gate_state_intact_msa_cleared` |
| observer seed rebuilt from control + last-12 metas (window, peak, freezes, last-mastered, rollback state, safety=max(rollback, late-floor)) | `test_observer_seed_rebuilt_from_metas_and_control` |
| shrinks_since_rollback rebuilt counting only post-rollback shrinks | `test_shrinks_since_rollback_counts_only_post_rollback` |
| STATE_FIDELITY.json: all critical fields match; losses explicit (default_lost) | `test_fidelity_report_faithful_with_explicit_losses` |
| resume snapshots copied into new policy dirs; env clock exported | `test_resume_snapshot_copied_into_new_policy_dirs` |
| fail-closed on missing control / short meta history (nothing claims faithful) | `test_fail_closed_on_missing_control`, `test_fail_closed_on_short_meta_history` |
| fingerprint verify passes clean; detects restored-state tamper AND source change | `test_fingerprint_verify_and_tamper_detection`, `test_fingerprint_detects_source_change` |
| cadence audit measures the REAL 80M committed quantum | `test_cadence_audit_measures_real_80m` |
| observer restores windows + gate memory from the seed | `TestObserverSeedLoading.test_seed_restores_windows_and_gate_memory` |
| stale seed (past validity window) skipped; live state never regressed; one-shot; no-op without seed dir | `test_stale_seed_skipped_after_validity_window`, `test_live_state_never_regressed`, `test_applied_once_per_process`, `test_noop_without_seed_dir` |
| control fidelity check catches the 0.075 clobber / peak reset / freeze-horizon loss | `TestGuardControlFidelity.*` |
| stall reproduction gate: ≥4/6 exact-hold passes, 3/6 fails | `TestStallReproductionGate.*` |
| guard loop end-to-end: pass + marker with reproduction record; one legal shrink ok; majority breakout → STATE_FIDELITY_FAILED; missing RESUME_APPLIED → GUARD_FAILED; legacy cascade (0.016849) → GUARD_FAILED; clobbered control → STATE_FIDELITY_FAILED | `TestGuardLoopStateFaithful.*` (6 tests) |
| escalating stop: SIGTERM-ignoring process SIGKILLed and verified dead; cooperative process exits on SIGTERM; SIGSTOPped process SIGCONTed first | `TestEscalatingStop.*` (real processes, incl. `trap '' TERM`) |
| env-clock resume parsing (lcu:fsr) validated; arms clock list from source snapshots | `TestClockResume.*` |
| first-tick semantics: RESUME_APPLIED structural check, one-shrink band, legacy-cascade rejection, reproduced flag | `TestContinuationGuard.test_first_tick_*` (updated) |
| worker-local cap at real cadence: 480M target, 80M slack | `TestContinuationGuard.test_worker_local_cap_at_real_cadence`, `TestMatchedCap.*` (updated) |

Synchronization coverage ("Make readiness a hold barrier with sidecar
adoption"; 11 new tests + reworked barrier tests, all against REAL
processes):

| # | requirement | test(s) |
|---|---|---|
| 1 | an early validated worker is FROZEN — its frame/progress stops | `TestReadinessBarrier.test_early_frozen_worker_makes_no_progress` (real counter process), `test_guard_loop_partial_workers_no_marker` |
| 2 | before 6/6 there is NO validated-but-unfrozen worker | `test_guard_loop_partial_workers_no_marker` (validated 0-2 in 'T', unvalidated 3-5 running; READINESS_HOLDS.json contents asserted) |
| 3 | BASE/GATE: all six released SIMULTANEOUSLY, record removed | `test_guard_loop_writes_marker_only_after_all_six` |
| 4 | MSA sidecar adopts the readiness holds seamlessly (never releases in between) | `TestReadinessBarrier.test_guard_loop_msa_handoff_keeps_workers_frozen`, `TestSidecarHoldAdoption.test_adopt_seamless` |
| 5 | PRE snapshot frame EXACTLY equals the validated frame (mismatch fails closed) | `TestSidecarHoldAdoption.test_pre_snapshot_frame_equals_validated_frame` (+ controller `_snapshot_at` enforcement) |
| 6 | sidecar adoption failure releases ALL workers and stops the arm | `test_adopt_failure_releases_all_and_stops_arm` |
| 7 | children-before-root termination, robust to a supervisor respawning children | `TestTerminateTreeOrder.test_kill_order_children_before_root`, `test_supervisor_respawning_child_is_fully_stopped` (root with `trap '' TERM` + respawn loop) |
| 8 | six-GPU metadata: launchers record MSA_EVAL_DEVICES (default cuda:4,cuda:5, shared with trainers 4/5) into EVAL_DEVICES.json | `TestOriginalProfileUnchanged.test_six_gpu_metadata_recorded` |
| — | certificate gate: matching cert passes; ok=false / commit / task / profile / GPU model / driver mismatch fails closed; per-eval-device model check | `TestHoldSmokeCertificate.*` (3 tests) |
| — | any readiness failure SIGCONTs the already-frozen workers before the arm stops | `test_guard_loop_any_failure_stops_arm_without_marker` (extended: no pid left in 'T') |

Watchdog/preflight coverage ("Close readiness watchdog and shared-GPU
preflight gaps"; 10 new tests, real processes):

| # | requirement | test(s) |
|---|---|---|
| P0-1 A | stop_arm called FROM the process registered as msa_sidecar.pid: all six workers exit AND the caller reaches its post-stop_arm completion marker (never killed mid-cleanup) | `TestStopArmCallerSafe.test_caller_is_sidecar_kills_workers_and_all_siblings` (real python subprocess caller) |
| P0-1 B | same from a process registered as hold_watchdog.pid | `test_caller_is_watchdog_kills_workers_and_all_siblings` |
| P0-1 C | ALL sibling control processes (continuation_guard.pid + msa_sidecar.pid + hold_watchdog.pid minus the caller) ARE terminated; holds files cleared (no stale ownership) | asserted inside tests A/B + `test_caller_is_guard_kills_workers_and_all_siblings` |
| P0-1 | the real watchdog path (check_once → default stop_arm) is caller-safe end to end | `test_watchdog_check_once_stop_is_caller_safe` |
| P0-2 1 | guard dies mid-barrier with PARTIAL readiness holds → all frozen pids SIGCONTed, GUARD_DIED_DURING_READINESS.json, arm stopped | `TestReadinessWatchdog.test_guard_death_partial_readiness_releases_and_stops` |
| P0-2 2 | 6/6 marker written, sidecar dies BEFORE adoption (readiness holds still present) → release all + stop | `test_sidecar_death_before_adoption_releases_and_stops` |
| P0-2 3 | READINESS_HOLDS non-empty → never idle_exit | `test_readiness_present_never_idle` |
| P0-2 4 | corrupt/partial readiness file → fail closed (never the empty set); startup grace tolerated while pid files are absent | `test_corrupt_readiness_fails_closed` |
| P0-2 5 | after normal adoption (readiness gone, MSA_HOLDS + live sidecar) no double release / no false kill | `test_normal_adoption_no_false_action`, `test_guard_alive_during_barrier_is_ok` |
| P0-2 6 | watchdog calling stop_arm never kills itself first | `TestStopArmCallerSafe.test_watchdog_check_once_stop_is_caller_safe` |
| P1 | certificate WITHOUT eval_coexistence_ok (old cert, or coexistence failed) rejected by the formal sidecar gate | `TestHoldSmokeCertificate.test_certificate_without_coexistence_proof_rejected`, `test_mismatches_fail_closed` (eval_coexistence_ok=false case) |

Final orchestration patch ("Bound coexistence eval and supervise guard
lifetime"; 8 new tests, real processes):

| # | requirement | test(s) |
|---|---|---|
| A | evaluator writing MBs of stdout/stderr never deadlocks (log file, no unconsumed PIPE) | `TestCoexistenceEvalBounded.test_large_output_does_not_deadlock` (4MB output, exit 0, full log on disk) |
| A | eval timeout → SIGTERM whole process group → bounded grace → SIGKILL → verified NO survivor | `test_timeout_terminates_whole_process_group` (hung bash + spawned child, both dead, bounded wall time) |
| A | timeout fails the certificate (ok=false, eval_coexistence_ok=false, eval_timed_out=true, new fields recorded) AND the trainer hold is still released in the finally | `test_timeout_releases_trainer_hold_and_fails_certificate` (real SIGSTOPped trainer resumed after fake timeout) |
| B | BASE-shape arm (readiness done, no sidecar): guard death → CONTINUATION_GUARD_DIED.json with guard pid, marker presence, holds/busy, six worker frames, timestamp; arm stopped | `TestGuardLifetimeWatchdog.test_base_arm_guard_death_after_readiness_stops_arm` |
| B | MSA mid-cycle (sidecar ALIVE, MSA_HOLDS non-empty): guard death still fatal; union of holds SIGCONTed | `test_msa_adoption_guard_death_releases_and_stops` |
| B | guard death with MSA_BUSY.json present (holds empty, COMMITTING window) fails closed | `test_msa_busy_guard_death_fails_closed` |
| B | after ARM_COMPLETE.json the guard's exit is NEVER reported (dead or missing pid → quiet idle_exit, no marker, no stop) | `test_arm_complete_guard_exit_is_not_reported`, `TestHoldWatchdog.test_dead_sidecar_no_holds_keeps_watching_until_arm_complete` |
| C | caller registered as continuation_guard.pid survives its own stop_arm; sidecar + watchdog siblings all exit | `TestStopArmCallerSafe.test_caller_is_guard_kills_workers_and_all_siblings` |

CPU tests vs real GPU validation for P1: the eval-coexistence smoke itself
(Isaac eval cell on GPU4 beside a SIGSTOPped VRAM-holding trainer) is GPU
work and was NOT executed in this pass — only the launcher/certificate
plumbing and the fail-closed gate are CPU-tested. The certificate can ONLY
be produced by actually running `start_stateful_hold_smoke.sh` on the six-
GPU machine; no GPU smoke result is claimed here.

## Real execution order (enforced, not just procedural)

```
start_stateful_hold_smoke.sh (DISPOSABLE) → HOLD_SMOKE_CERTIFICATE.json
  (hold + frozen-frame proof + SHARED-GPU EVAL COEXISTENCE: one real
   policy-only eval cell, N=1, 512 envs, on the SAME GPU while the frozen
   trainer holds its VRAM → eval_coexistence_ok=true required)
launch arm → state bootstrap + fingerprint (in-launcher, fail closed)
           → hold watchdog started for EVERY stateful arm (guard +
             sidecar + readiness + MSA holds coverage)
           → sidecar validates the certificate incl. eval_coexistence_ok
             (fail closed at startup)
           → per-worker validation → immediate SIGSTOP freeze
             (READINESS_HOLDS.json) → 6/6 → stall reproduction
             → FIRST_TICK_VALIDATED.json
           → BASE/GATE: simultaneous SIGCONT of all six
           → MSA: sidecar adopts holds (→ MSA_HOLDS.json), first PRE_EVAL
             from the frozen validated boundary
failure paths (all caller-safe, workers first, caller last; stop_arm also
terminates the continuation_guard/msa_sidecar/hold_watchdog siblings):
  guard dies ANYWHERE before ARM_COMPLETE.json
                             → watchdog SIGCONTs the union of readiness/MSA
                               holds, writes CONTINUATION_GUARD_DIED.json
                               (plus GUARD_DIED_DURING_READINESS.json when
                               mid-barrier), stops workers + sidecar
  sidecar dies pre-adoption  → watchdog releases readiness holds, stops arm
  sidecar dies mid-cycle     → watchdog releases MSA holds, writes
                               MSA_SIDECAR_DIED.json, stops arm
  coexistence eval hangs     → hard --cert-eval-timeout-s (1800s default):
                               SIGTERM process group → grace → SIGKILL →
                               survivor check; certificate fails closed and
                               the frozen trainer is still released
ARM_COMPLETE.json = normal completion: guard exit after it is never
reported (no false positive).
```

The in-arm `run_hold_smoke.sh` is EXECUTION SMOKE ONLY (it advances the
released worker one extra committed tick) and is forbidden in the formal
four-arm comparison.

## Regression suites (per-file, one process per test module)

| suite | result |
|---|---|
| `tests/test_msa_probe.py` | 143 passed |
| `tests/test_state_bootstrap.py` | 32 passed |
| `tests/test_pbt_curriculum_p1.py` + `test_curriculum_freeze.py` + `test_env_mastery_curriculum.py` | 22 passed, 1 pre-existing failure (below) |
| `tests/test_ccrs_toolkit.py` + `tests/test_ccrs_precision_extension.py` | 116 passed |
| `tests/test_wm_strategy_profiles.py` | 31 passed |
| `tests/test_late_floor_retention_gate.py`, `test_env_mastery_curriculum.py`, `test_curriculum_adaptive_fast.py`, `test_hybrid_curriculum_safety.py`, `test_pbt_curriculum_inherit.py`, `test_curriculum_freeze_unit_conversion.py` | 76 passed |
| `tests/test_profile_copy_health_gates.py`, `test_v13_elite_profiles.py`, `test_profile_copy_v12.py` | 37 passed |
| `tests/test_pbt_curriculum_p1.py` | 11 passed, 1 pre-existing failure (`StableSourceTierTest::test_l2_usable_when_stable_fails`) — baseline: fails identically at the parent commit; unrelated |

Known pre-existing environment issue (unchanged, NOT fixed on purpose):
collecting many heavy profile suites in ONE pytest process segfaults
during import/teardown (reproduced identically at the parent commit).
Per-file runs are green.

## Production-style dry run (simulation tier)

`python -m dexpbt_isaaclab.msa.dry_run` — 6 workers with the REAL
heterogeneous 5B-tip tolerances (0.02247…0.02759, full precision, not
6-decimal-friendly), 80M tick-quantized parallel trainer (the REAL stage-3
committed cadence), phase-barrier holds with post-hold packet-start reads:
promoted=True, recipients=[1, 3, 0] (tie 0 vs 5 broken by worker id), total
tau_plus dose = 6×80M + 3×80M, zero residual pins/holds, all invariants
held. Permuted arm: same total dose. Full trace: `DRY_RUN_TRACE.md`.

State bootstrap against the REAL source run (read-only, no GPU):
`state_bootstrap --source-run <strict_5b_run> --run-dir /tmp/sb_dryrun` →
`STATE BOOTSTRAP OK: state_faithful=True (6 workers)`, fingerprint verify
clean, cadence audit `committed_tick_frames=80000000` for all six workers.

Plan-only continuation-guard verification against the real manifest:
`continuation_guard --verify-only` on `msa.config.json` →
`BOOTSTRAP OK: 6/6 checkpoints verified (hash, mapping, fields)`.

Readiness hold / handoff dry run with REAL processes (six live workers
frozen one by one by the guard, marker written with holds retained,
sidecar adoption converting READINESS_HOLDS.json → MSA_HOLDS.json with
frame verification, controller-style release): full trace appended to
`DRY_RUN_TRACE.md` (PIDs verified in 'T' through the whole handoff,
`PRE snapshot frame == validated frame` for all six).

Hold/watchdog/hold-smoke/barrier tests use REAL processes (`sleep` and
bash counter children, actual SIGSTOP/SIGCONT and /proc state checks) —
one tier above pure mocks, still no GPU. SIGSTOP against a live Isaac
trainer remains GPU-unverified; the prepared DISPOSABLE
`start_stateful_hold_smoke.sh` (certificate producer) is the required
first GPU validation step (see `LAUNCH_CHECKLIST.md`).

No GPU was used at any point; no training was started.
