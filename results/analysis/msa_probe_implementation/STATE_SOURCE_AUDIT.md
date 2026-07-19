# STATE_SOURCE_AUDIT — where the 0.022–0.028 stall state actually lives

Date: 2026-07-19
Source run: `results/runs/regrasping_cser_dexpbt_conservative_retention_strict_5b_20260716_130631`
Scope: the six 5B tip workers; every state source that decides `allow_shrink` /
`shrink_token` or otherwise keeps tolerance pinned in the stall band.

Motivating failure (2026-07-19 execution smoke): the previous continuation
restored ONLY `.pth` + `--success_tolerance_resume`. `curriculum_control/` was
empty at worker start, the env fell back to the ungated LEGACY curriculum
(`dexpbt_utils.dexpbt_tolerance_curriculum`, 0.9× per window at mean nsucc > 3)
and worker 0 shrank 0.0231 → 0.0168 inside the first 80M-frame committed
interval. The stall was NOT reproduced; the guard failed the arm.

## 0. The six tips (source of truth for the stall)

| w | tip frame | live tol (m) | nsucc | allow_shrink | first blocker | active freezes beyond tip |
|---|-----------|--------------|-------|--------------|--------------------------|---------------------------|
| 0 | 5,000M | 0.023112 | 33.2 | False | post_shrink_consolidation (until step 716,224 vs clock 711,360) | none (collapse/rollback freezes expired) |
| 1 | 5,000M | 0.020284 | 40.0 | False | collapse_freeze_active | collapse 5,120M; safety 5,170M |
| 2 | 5,080M | 0.025490 | 30.6 | False | collapse_freeze_active | collapse 5,120M; safety 5,200M; late-floor 5,250M |
| 3 | 5,000M | 0.023658 | 39.7 | False | danger_zone_not_stable | rollback-freeze horizon 4,690M (expired) |
| 4 | 5,000M | 0.019270 | 31.8 | False | curriculum_safety_freeze_active | safety 5,090M; late-floor 5,250M |
| 5 | 5,000M | 0.021308 | 35.7 | False | collapse_freeze_active | collapse 5,040M; safety 5,200M; late-floor 5,250M |

All six were blocked at the tip; four of six had freeze horizons that extend
INTO the continuation window (up to +250M). A restore that drops these freezes
lets those workers shrink one tick after launch — exactly the non-reproduction
signature the new reproduction gate detects.

## 1. State already inside the checkpoint (`<prefix>.pth` / committed meta JSON)

- Policy/optimizer weights, RL epoch (`.pth`).
- Committed meta (`<prefix>.json`): `frame`, `iteration`, `lineage_id`,
  `nsucc_mean`, `nsucc_max`, `curriculum.success_tolerance`,
  `restart_count`, `frames_at_restart`, PBT clocks (via `PbtStateSnapshot`).
- `LATEST.commit`: the committed tip prefix.

Restored by: per-policy `--checkpoint` + the meta stays in the source
workspace (read-only). The observer reconstructs `pbt_iteration` /
`last_mutation_frame` from workspace metas (`restore_pbt_tick_state_from_workspace`).

## 2. State in `curriculum_control/<w>.json` (env-side gating + observer telemetry)

Fields the ENV actually branches on (`apply_env_tolerance_curriculum`):
`msa_enabled` / `msa_forced_success_tolerance`, `forced_success_tolerance`,
`shrink_token` / `allow_shrink`, `collapse_freeze_active`,
`post_shrink_consolidation_until_restart_step`, `adaptive_shrink_factor`,
`adaptive_consolidation_frames`.

Fields the OBSERVER re-reads at startup (the only carry-overs):
`post_shrink_consolidation_until_restart_step`, `pre/post_shrink_tolerance`,
`msa_*`. **Everything else in the JSON is telemetry that the observer's next
tick overwrites from its in-memory dicts.**

Restored by: `state_bootstrap` copies the full source control JSON per worker
into the new workspace with ONLY `msa_*` cleared. This alone fixes the legacy
fallback (the file is present and `allow_shrink=False` from tick zero) and
keeps the consolidation step gate (env-side) alive.

## 3. State in `curriculum_resume_snapshot.json` (env clock)

`success_tolerance`, `initial_tolerance`, `target_tolerance`,
`last_curriculum_update`, `frame_since_restart`,
`curriculum_freeze_frames_after_restart` — env-side tolerance/clock only,
no gate state. Note `post_shrink_consolidation_until_restart_step` is measured
in sim steps SINCE PROCESS RESTART: without restoring
`frame_since_restart` the consolidation gate loses its source meaning
(worker 0: blocked until step 716,224 vs restored clock 711,360 → ~40M frames
of remaining consolidation).

Restored by: snapshot copied into the new `policy_XX/` dirs AND the clock is
passed explicitly via `--curriculum-clock-resume-per-policy` →
`--last_curriculum_update_resume` / `--frame_since_restart_resume`
(together with `--success_tolerance_resume`; success is verified structurally
through the new `RESUME_APPLIED.json`, not by first-tick equality).

## 4. State only rebuildable from historical committed metas

- **Rolling nsucc window** (`_policy_nsucc_windows`, max_len 8): rebuilt from
  the last 12 committed metas with `_window_sample_from_state` semantics →
  `observer_state_seed/<w>.json:window_samples`. Substrate for the mastery
  gate, collapse detection, global retention, stable ticks, slopes.
- **`shrinks_since_rollback`**: not persisted anywhere; rebuilt by counting
  tolerance-decrease transitions after `last_rollback_frame` in the meta tail
  (exact when the rebuild horizon covers the rollback, else a conservative
  lower bound — recorded as such in STATE_FIDELITY.json).

## 5. State that was NOT persisted anywhere (observer in-memory only)

These dicts silently reset on every observer relaunch. They ARE written to
the control JSON as telemetry, so the bootstrap recovers their last values
from there and the new observer seed loader
(`pbt_observer._maybe_apply_observer_state_seed`) restores them in-process:

| observer dict | recovered from control field | impact if lost |
|---|---|---|
| `_recent_peak_nsucc_mean` | `recent_peak_nsucc_mean` | collapse detection blind until a new peak forms |
| `_collapse_freeze_until` | `collapse_freeze_until_env_frames` | active collapse freeze (w1/w2/w5 extend past 5B) silently lifted |
| `_curriculum_safety_freeze_until` | max(`curriculum_rollback_freeze_until`, `late_floor_retention_freeze_until`) | safety/late-floor freezes lifted |
| `_wm_curriculum_freeze_until` | `wm_curriculum_freeze_until` | WM risk freeze lifted |
| `_last_mastered_tolerance` | `last_mastered_tolerance` | **clobbered to 0.075 on the first post-restart write** → a rollback would jump tolerance to fully-relaxed 0.075 (biggest hazard) |
| `_last_mastered_safe_*` | `curriculum_last_mastered_*` | repeated-collapse / late-floor rollback target lost |
| `_repeated_collapse_count` | `curriculum_repeated_collapse_count` | rollback patience restarts |
| `_rollback_count` / `_last_rollback_frame` / `_rollback_reason` | `rollback_count` / `last_rollback_frame` / `rollback_reason` | rollback cooldown resets to "elapsed" |
| `_floor_recovery_protect_until` | `floor_recovery_protect_until` | floor-recovery protection lost |
| `_first_floor_frame` / `_first_hard_floor_frame` | same-name fields | frames_since_floor restarts |
| `_shrink_issued_this_window` | `shrinks_in_window` (per-window; N/A across a relaunch — reset each PBT tick anyway) | masked ~2 ticks by the empty mastery window |

**Truly unrecoverable** (no persisted trace; restored as defaults and declared
in `STATE_FIDELITY.json:known_losses`):

- `_promising_protect_until` (copy-event JSONL only) — elimination protection;
  no effect on the shrink gate.
- `_mutation_frozen_until` — stable workers may be mutated on the next copy;
  no effect on the shrink gate.
- `_late_floor_retention_patience_count` — late-floor rollback patience
  restarts from 0 (conservative: DELAYS rollbacks, never causes a shrink).
- `_floor_recovery_score_history` — reporting only.

None of the four decides `allow_shrink`/`shrink_token`, so their loss cannot
break out of the stall band; fidelity is sufficient for the causal contrast
(all four arms share exactly the same losses).

## 6. Committed cadence audit (never assume 20M)

Measured from source metas (`state_bootstrap --audit-cadence`):
- ~200M → ~3.48B frames: 40M deltas (stage-2 interval mult 2.0);
- ~3.48B → 5B: **80M deltas** (stage-3 SELECTION mult 4.0), all six workers,
  last 10+ deltas exactly 80,000,000; confirmed by the smoke run (first
  commit at baseline+80M).

Consequences (all applied in code):
- `PROBE_PACKET_FRAMES` = 80M (one REAL committed tick);
- reproduction interval = 1 tick = 80M;
- cap slack = 1 tick = 80M;
- continuation budget = 6 ticks = 480M/worker
  (1 reproduction + 2 probe cycle + 2 post-cycle observation + 1 slack);
- `max_iterations` = 44,500 + 4,280.

## 7. Fidelity verdict

- **Fully restorable (exact copy)**: all env-side gate state, all control-JSON
  gate/telemetry fields, env tolerance/clock, checkpoint identity, lineage.
- **Rebuildable (declared method)**: rolling windows (last 8 samples from last
  12 metas), `shrinks_since_rollback` (lower bound when horizon short).
- **Lost (defaults, no shrink-gate impact)**: promising/mutation-freeze/
  late-floor-patience/score-history.

This is sufficient for the causal experiment because (a) every state source
that gates shrinking is restored exactly, (b) the losses are identical across
all four arms, and (c) the stall reproduction gate (first committed interval,
≥4/6 workers must hold their source tolerance exactly, control state must not
be default-clobbered) verifies EMPIRICALLY that the stalled regime was
reproduced before any MSA intervention may start
(`STATE_FIDELITY_FAILED.json` + full arm stop otherwise).

Machine-readable artifacts written by the bootstrap into each new run dir:
`STATE_FIDELITY.json` (per worker, per field: source vs restored vs category),
`STATE_FINGERPRINT.json` (sha256 of the restored state + of every consumed
source file; validated again by the launcher before anything starts).
