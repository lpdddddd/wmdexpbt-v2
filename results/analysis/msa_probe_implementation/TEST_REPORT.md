# base_retention_msa_probe — Test Report

Date: 2026-07-19. Environment: isaaclab311 conda python (CPU only; no Isaac
import anywhere in the MSA package or its tests; no GPU used).

## New suite: `tests/test_msa_probe.py` — 41/41 passed

(41st test added in the GATE_RELAX profile-delivery fix, see addendum below.)

Coverage of the 15 required behaviors (spec section J):

| # | required behavior | test(s) |
|---|---|---|
| 1 | tau=0.010 never probed | `TestDueLogic.test_tau_at_floor_never_due`, `test_tau_plus_floor_clamp` |
| 2 | missing-provenance resume immediately due | `TestDueLogic.test_missing_provenance_immediately_due`, `test_from_dict_none_stays_none` |
| 3 | probe at most once per 100M worker-local frames | `TestDueLogic.test_probe_at_most_once_per_window` |
| 4 | evaluator never modifies live state | `TestEvaluatorIsolation.*` (read-only snapshot enforcement + AST scan: no `load_state_dict`/`optimizer`/`replay_buffer`/`mutate`/`restart_policy` identifiers in sidecar/controller/eval_bridge; only `forced_success_tolerance` + `msa_enabled` written to control JSON) |
| 5 | pre/mid tolerance or seed-manifest mismatch fails closed | `TestPairedEvalContracts.test_manifest_mismatch_fails_closed`, `test_tolerance_mismatch_fails_closed`, `test_seed_order_mismatch_fails_closed` |
| 6 | all workers' first half exactly 8M | `TestDryRunCycle.test_common_probe_is_8m_for_all` |
| 7 | second-half top-3 deterministic tie-break | `TestRouting.test_top3_stable_tie_break`, `TestDryRunCycle.test_second_probe_deterministic_recipients` |
| 8 | permuted route keeps total dose identical | `TestRouting.test_permuted_route_same_dose`, `TestDryRunCycle.test_permuted_arm_same_dose_different_mapping` |
| 9 | q_post never enters common-dose L_mid | `TestDryRunCycle.test_q_post_not_in_common_dose` |
| 10 | p_pre < 0.01 → retention unavailable, promotion blocked | `TestPairedEvalContracts.test_low_p_pre_retention_unavailable` |
| 11 | rho compares same tau only | `TestPairedEvalContracts.test_rho_same_tau_only`, `TestBootstrap.test_cross_tolerance_gate_names_rejected` |
| 12 | bootstrap deterministic (fixed PRNG) | `TestBootstrap.test_bootstrap_deterministic`, `test_bootstrap_is_labeled_conditional`, `test_negative_panel_blocks_promotion` |
| 13 | no optimizer/PBT-lineage reset through probe | `TestDryRunCycle.test_no_resets` + isolation AST scan |
| 14 | frame ledger includes all probe frames | `TestDryRunCycle.test_frame_ledger_includes_probe_frames`, `test_forced_tolerance_sequence` |
| 15 | original base-retention behavior unchanged | `TestOriginalProfileUnchanged.*` (inert dataclass defaults; legacy control JSON round-trip; `apply_env_tolerance_curriculum` / `dexpbt_tolerance_curriculum_mastery` / `mark_shrink_consumed` contain no `msa` branch; GATE_RELAX vs BASE_CONTINUATION launcher diff is exactly one line) |

Plus: explicit state machine tests (full legal cycle, illegal jumps raise,
serialization round-trip, no implicit freeze flags), the read-only blocker
audit on a synthetic run, and arm-config invariants (matched seeds, same six
resume checkpoints, probe dose parity between MSA and MSA_PERMUTED_ROUTE).

## Regression suites (unchanged code paths) — all green

| suite | result |
|---|---|
| `tests/test_msa_probe.py` | 41 passed |
| `tests/test_ccrs_toolkit.py` + `tests/test_ccrs_precision_extension.py` | 116 passed |
| `tests/test_late_floor_retention_gate.py`, `test_env_mastery_curriculum.py`, `test_curriculum_adaptive_fast.py`, `test_hybrid_curriculum_safety.py`, `test_pbt_curriculum_inherit.py`, `test_curriculum_freeze_unit_conversion.py` | 76 passed |
| `tests/test_pbt_curriculum_p1.py` | 11 passed, 1 pre-existing failure (`StableSourceTierTest::test_l2_usable_when_stable_fails`) — verified identical at the pre-change commit via `git stash`; unrelated to this task |

## Dry-run

`python -m dexpbt_isaaclab.msa.dry_run` executes a full probe cycle against
a deterministic fake world (6 workers at tau=0.0235): promoted=True,
recipients=[1, 3, 0] (tie 0 vs 5 broken by worker id), all invariants held.
Permuted arm: recipients=[3, 2, 1] under seed 777001 with identical total
dose. Full trace: `DRY_RUN_TRACE.md`.

No GPU was used at any point; no training was started.

## Addendum: GATE_RELAX profile-delivery fix (same date)

Launcher/log exploration confirmed that `apply_wm_strategy_profile` is
authoritative for non-legacy profiles: profile-table values override
explicit CLI args, and `late_floor_retention_min_retention` (0.88) is in
the `cser_dexpbt_conservative` table. The originally generated GATE_RELAX
launcher relied on a CLI flag alone, which would have been silently
reverted at startup. Fix: registered a dedicated
`cser_dexpbt_conservative_gate_relax` profile (id 21; single-key table
diff to 0.75), pointed the GATE_RELAX launcher's `--profile-per-policy`
at it, and added fail-closed validation in `msa.arms` plus
`TestArmConfigs.test_gate_relax_goes_through_dedicated_profile`.
Re-verified: `tests/test_msa_probe.py` 41 passed;
`tests/test_wm_strategy_profiles.py` 31 passed;
`tests/test_profile_copy_health_gates.py` + `tests/test_v13_elite_profiles.py`
19 passed; `tests/test_profile_copy_v12.py` 18 passed. (Running the four
profile suites in one pytest process segfaults at teardown — reproduced
identically at the pre-change commit via `git stash`; pre-existing,
unrelated.)
