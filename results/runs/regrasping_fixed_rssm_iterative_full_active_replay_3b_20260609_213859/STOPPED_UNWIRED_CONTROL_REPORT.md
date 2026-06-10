# Stop report — `iterative_full_active_replay_3b_20260609_213859`

**Status:** ⛔ Stopped by user (not a crash)
**Reason:** Full RSSM iterative-stack flags enabled, but the **active WM path is not actually wired** at runtime. This run can only serve as a "full-flags-but-unwired" control; it is not valid main-method evidence.
**Stop time:** 2026-06-10 ~01:36 (UTC+8) (see `RUN_STOPPED_BY_USER.json`)
**Stop method:** SIGTERM → 90 s wait → SIGKILL on launch_pbt + supervisor pgids → `pkill -KILL -f <run-tag>`. Other runs and GPU state confirmed clean afterwards. Run dir and checkpoints preserved.

---

## 1. Latest frames / tolerance / nsucc (per policy at stop)

| Policy | Frames | Tolerance | nsucc_mean | nsucc_max | floor_stage | restarts | pbt_iter |
|--------|-------:|----------:|-----------:|----------:|:-----------:|---------:|---------:|
| p0 | 1,240 M | 0.0259 | 22.3 | 50 | none | 0 | 62 |
| p1 | 1,140 M | 0.0362 | 30.6 | 50 | none | 12 | 57 |
| p2 | 1,260 M | 0.0264 | 28.4 | 50 | none | 3 | 63 |
| p3 | 1,180 M | 0.0245 | 28.2 | 50 | none | 1 | 59 |
| p4 | 1,280 M | 0.0233 | 23.8 | 50 | none | 2 | 64 |
| p5 | 1,220 M | 0.0310 | 31.4 | 50 | none | 2 | 61 |

- **tol_median ≈ 0.026** at stop, range 0.023 – 0.036.
- **No policy ever reached `floor` (≤0.02)** or `hard_floor` (≤0.012); `floor_stage` is `none` everywhere across the entire run.

## 2. WM-guided ranking activity

| Metric | Value |
|--------|------:|
| `pbt/wm_changed_source_selection` (active mode actual changes) | **0** across all ticks |
| `pbt/wm_shadow_changed_selection` (would-change in shadow path) | **0** |
| `wm_shadow_observations` (p0 example) | ~48 |
| `floor_recovery_score_wm` | **0.0** in every jsonl row (all 6 policies) |
| `pbt_wm_score_mode` (config) | `active` |
| `pbt_use_wm_recovery_score` (config) | `true` |

→ **active WM ranking never replaced source selection.** The `hard_floor` gate inside the alt-objective loop never fired (no policy reached hard_floor), and even if it had, `_wm_imagined_eval_per_policy` is empty so the fallback to real-only would still trigger.

## 3. Teacher promotion

| Metric | Value |
|--------|------:|
| Teacher promotion count | **0** |
| `[wm-teacher]` log lines in train.log | **0** |
| `wm_teacher/*` TB tags | **none emitted** |

→ **`TeacherManager` was never instantiated.** Class exists in `wm_teacher_manager.py`, only referenced from tests.

## 4. Imagined evaluator activity

| Metric | Value |
|--------|------:|
| `imagined_valid` count in `_wm_imagined_eval_per_policy` | **0** |
| `wm_imagine/*` TB tags | **none emitted** |

→ **`WMImaginedEvaluator` was never instantiated.** Class exists in `wm_imagined_evaluator.py`, only referenced from tests.

## 5. Replay prioritization metrics availability

| Metric | Status |
|--------|:------:|
| `seq_wm_replay_prioritize_floor_recovery` flag | ✅ true (sampler created with prio enabled) |
| `wm_replay/floor_fraction` | ❌ not logged |
| `wm_replay/hard_floor_fraction` | ❌ not logged |
| `wm_replay/recovery_fraction` | ❌ not logged |
| `wm_rssm/floor_loss_mult` | ✅ logged but **all 1.0** (no policy in floor → multiplier never elevated) |

→ Sampler can read `last_replay_metrics`, but those are never propagated to TB/jsonl. With tol > 0.02 throughout, prioritization is essentially uniform.

## 6. Horizon / recovery target health (degenerate)

| TB tag (p0, ~8771 steps) | mean | last | judgement |
|--------------------------|-----:|-----:|-----------|
| `wm_rssm/recovery_pos_rate` | **0.0** | 0.0 | ❌ degenerate |
| `wm_seq/loss_recovery` | **0.0** | 0.0 | ❌ no signal |
| `wm_seq/loss_horizon_success_h8` | 0.056 | 0.056 | ✅ has signal |
| `wm_seq/loss_horizon_success_h16` | **0.0** | 0.0 | ❌ degenerate (seq_len=16) |
| `wm_seq/loss_drop_risk_h8` | 0.0018 | ~0 | ⚠️ very weak |
| `wm_seq/loss_drop_risk_h16` | **0.0** | 0.0 | ❌ degenerate |

- `recovery_pos_rate=0` confirmed across all 6 policies.
- `h16` is mechanically too long for `seq_len=16` (only 1 valid time index).
- Recovery target is currently only built when (some derivative of) hard_floor data exists — under tol > 0.02, target mask is empty.

## 7. Conclusion

**This run is NOT valid main-method evidence.** It is a control showing that "full flags enabled" is not the same as "method actually engaged."

Specifically:

1. `TeacherManager` is implemented but **not connected** to `train.py` runtime; promotion count = 0.
2. `WMImaginedEvaluator` is implemented but **not connected** to `train.py` / `pbt_observer.py`; valid count = 0.
3. `_wm_imagined_eval_per_policy` is read but never written → active ranking always falls back to real-only.
4. `pbt/wm_changed_source_selection` = 0 throughout (active mode behaved identically to shadow / off).
5. Recovery head and `h16` horizon head are degenerate — `recovery_pos_rate = 0`, `loss_h16 = 0` — under current `seq_len=16`, `horizon_steps=8,16` and floor-only target mask.
6. Replay prioritization runs but with no floor signal it is effectively uniform; per-stage fractions are not logged.

The **adaptive_fast curriculum + Part-A floor-recovery-aware PBT infrastructure + RSSM self-training** are running correctly; the **iterative core (teacher → imagined eval → active WM ranking → floor-prioritized replay)** is not.

## 8. Preservation

- ✅ Run dir preserved at full path.
- ✅ Checkpoints not deleted (under `dexpbt_8gpu_ws/00*/wm_seq_checkpoints/` and `restart_p*/`).
- ✅ `RUN_STOPPED_BY_USER.json` written alongside this report.

## 9. Next step (separate task)

Implement the runtime wiring patch (B-block): instantiate `TeacherManager` and `WMImaginedEvaluator` in `train.py`, write per-policy imagined eval into the observer, expose the full set of WM/replay/horizon TB+jsonl audit fields, default `horizon_steps=4,8`, add `wired` launcher. Do **not** start the wired 3B run automatically — wait for explicit confirmation.
