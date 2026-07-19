# base_retention_msa_probe — Test Report

Date: 2026-07-19, updated the same date for the pilot-integration fix pass
("Fix MSA pilot integration and phase safety"), the pre-pilot safety patch
("Harden MSA readiness holds and pilot budget") and the final orchestration
blocker patch ("Close MSA smoke, commit, and cap races"). Environment:
isaaclab311 python (CPU only; no Isaac import anywhere in the MSA package
or its tests; no GPU used).

Three evidence tiers, kept separate on purpose:

1. **Unit tests** — pure-python contracts (this file, sections below).
2. **Production-style simulation** — heterogeneous-tau full-cycle dry run
   against a tick-quantized parallel fake trainer with held-worker
   assertions and crash injection (`DRY_RUN_TRACE.md`).
3. **GPU validation** — NOT performed (no GPU tasks were run). SIGSTOP
   holds and live commit-wait behavior remain GPU-unverified; see
   `LAUNCH_CHECKLIST.md`.

## MSA suite: `tests/test_msa_probe.py` — 110/110 passed

(72 retained from the pilot-integration pass + 23 safety-patch tests +
15 new orchestration-blocker tests.)

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
| sidecar death watchdog releases held workers + stops arm | `TestHoldWatchdog.test_dead_sidecar_with_holds_releases_and_stops` (real stopped process resumed), `test_sidecar_alive_is_ok`, `test_dead_sidecar_no_holds_idle_exit` |
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

## Real execution order (enforced, not just procedural)

```
launch arm → 6/6 readiness (FIRST_TICK_VALIDATED.json)
           → hold smoke (run_hold_smoke.sh)
           → HOLD_SMOKE_PASSED.json
           → first N=4 execution-smoke cycle
```

## Regression suites (per-file, one process per test module)

| suite | result |
|---|---|
| `tests/test_msa_probe.py` | 110 passed |
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
6-decimal-friendly), 20M tick-quantized parallel trainer, phase-barrier
holds with post-hold packet-start reads: promoted=True,
recipients=[1, 3, 0] (tie 0 vs 5 broken by worker id), total tau_plus
dose = 6×20M + 3×20M, zero residual pins/holds, all invariants held.
Permuted arm: same total dose. Full trace: `DRY_RUN_TRACE.md`.

Plan-only continuation-guard verification against the real manifest:
`continuation_guard --verify-only` on `msa.config.json` →
`BOOTSTRAP OK: 6/6 checkpoints verified (hash, mapping, fields)`.

Hold/watchdog/hold-smoke tests use REAL processes (`sleep` children,
actual SIGSTOP/SIGCONT and /proc state checks) — one tier above pure
mocks, still no GPU. SIGSTOP against a live Isaac trainer remains
GPU-unverified; the prepared `run_hold_smoke.sh` is the required first
GPU validation step (see `LAUNCH_CHECKLIST.md`).

No GPU was used at any point; no training was started.
