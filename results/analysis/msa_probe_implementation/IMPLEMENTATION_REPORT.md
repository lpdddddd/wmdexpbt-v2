# base_retention_msa_probe — Implementation Report

Date: 2026-07-19. Branch: `base-retention`. No GPU training or evaluation
was started; everything below is code, tests, launcher generation, a
read-only log audit, and a CPU dry-run.

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
| `eval_bridge.py` | immutable read-only snapshots (`chmod a-w` + `assert_immutable`), ordered seed manifest (identity-hashed, N=256 formal), ccrs_eval subprocess command construction (policy-only, deterministic, per-seed results), fail-closed per-seed result loading |
| `controller.py` | `MSAController.run_cycle`: drives one full probe cycle for the due panel with dependency-injected evaluate/snapshot/forced-tolerance/train callbacks; enforces exactly-8M packets; records provenance + q_post (recipients only, follow-up only) |
| `sidecar.py` | production wiring: workspace tick metas → worker views; snapshot freezing; ccrs_eval subprocess evals; `forced_success_tolerance` writes through the existing atomic control writer (the only live-state mutation channel, already honored by the env); packet completion by polling the worker's monotonic ledger |
| `audit.py` | the read-only 3B–5B first-blocker audit above |
| `arms.py` | resolves the six 5B tip checkpoints from the CCRS manifest (fail-closed), builds the four arm configs + launchers |
| `dry_run.py` | deterministic in-memory full-cycle simulation + `DRY_RUN_TRACE.md` generation |

Changes to existing files (both inert by default):

- `pbt/pbt_curriculum_p1.py`: `CurriculumControlState` gains
  `msa_enabled=False`, `msa_provenance=None`, `msa_state_machine=None` —
  serialized in the SAME control JSON / resume snapshot chain (no parallel
  checkpoint system); no runtime function branches on them (unit-tested).
- `scripts/launch_pbt.py`: new `--recover-checkpoint-per-policy` option
  (default empty = legacy behavior; validated length/paths; mutually
  exclusive with `--recover-checkpoint`) so continuation arms can warm-start
  all six workers from their own 5B checkpoints.

## 3. Key design decisions

- **No parallel checkpoint system**: provenance + state machine ride inside
  the existing curriculum control JSON chain. A pre-MSA checkpoint simply
  lacks the keys → `probe_due` fires immediately on resume, per spec.
- **Sidecar, not trainer surgery**: the probe forces tau_plus/tau through
  `forced_success_tolerance`, a field the env-side curriculum already obeys.
  The trainer keeps running normally through probe segments, so policy,
  PPO optimizer, replay/snippet state, PBT lineage and curriculum state are
  physically untouched (nothing reloads or resets them), and probe frames
  land on the worker's monotonic ledger and consume the 100M budget.
- **Retention is same-tau by construction**: `same_tau_retention` takes two
  `SeedEvalResult`s and fails closed on any tau/manifest mismatch; there is
  no code path through which a historical peak at another tolerance could
  enter the promotion decision (gate names suggesting it are rejected).
- **q_post is quarantined**: evaluated only for second-packet recipients,
  stored in the intervention record as unequal-dose follow-up, never in the
  common-dose L_mid statistic (unit-tested).
- **Statistics pre-registered**: bootstrap seed 20260719 / 10000 repeats /
  5th percentile of the panel median; permuted-route seed 777001; formal
  eval N=256 ordered seeds. The panel statistic is explicitly labeled as
  conditional on the current six-checkpoint panel, not a population CI.

## 4. Continuation arms (prepared, NOT launched)

See `CONFIG_DIFF.md` for the full diff table and `LAUNCH_CHECKLIST.md` for
launch preconditions. All four arms resume from the same six 5B tips with
matched seed 42 and +100M worker-local frames:

1. `BASE_CONTINUATION` — original flags, unchanged.
2. `GATE_RELAX` — single change: `late_floor_retention_min_retention`
   0.88 → 0.75 (the pre-existing hard minimum).
3. `MSA` — probe sidecar; original numeric gates; cross-tolerance retention
   read replaced by same-tau rho (threshold 0.88).
4. `MSA_PERMUTED_ROUTE` — routing negative control (same total dose).

## 5. Known limitations / open points before launch

- The formal N=256-seed probe evaluation reuses the CCRS
  one-Isaac-process-per-unit protocol; at ~6 min/unit this is ~26 GPU-hours
  per eval point, which is the dominant cost of an MSA cycle (6 pre + 6 mid
  + 6 post + tau_plus/tau pairs). The launch checklist requires an explicit
  decision on eval budget (smaller N, or a dedicated eval GPU) before start.
- The sidecar polls tick metas (~20M-frame quantization); packet boundaries
  are therefore tick-quantized in production (exact-8M enforcement is at
  the ledger-accounting level, as in the dry run).
- `snapshot_pre/mid/post` freeze the newest step-boundary checkpoint; PBT
  save cadence defines the snapshot lag (≤ one tick).

## 6. Artifacts

- code: `dexpbt_isaaclab/msa/` (8 modules), `tests/test_msa_probe.py`
- launchers: `results/launchers/msa_probe/` (4 scripts + 4 configs)
- audit: `blocker_audit_3b_5b.json` (this directory)
- docs: `STATE_MACHINE.md`, `CONFIG_DIFF.md`, `DRY_RUN_TRACE.md`,
  `TEST_REPORT.md`, `LAUNCH_CHECKLIST.md` (this directory)
