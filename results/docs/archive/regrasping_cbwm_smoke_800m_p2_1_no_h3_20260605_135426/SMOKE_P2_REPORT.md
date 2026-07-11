# P2 Smoke Acceptance Report (800M)

**Run ID:** `regrasping_cbwm_smoke_800m_p2_1_no_h3_20260605_135426`
**Generated:** 2026-06-05 22:39 CST
**Run status:** STOPPED_BY_USER (5/6 policies completed, p1 partial)
**Stop time:** 2026-06-05T22:01+0800
**Reason:** smoke conclusion reached; p1 outlier not worth waiting
**P2 criteria:** 7/7 passed (criteria with N/A excluded)
**D criteria (PBT verification):** 4/4 passed

| ID | Criterion | Pass | Detail |
|----|-----------|------|--------|
| S1 | Single-shot shrink (no multi-shrink within window) | **PASS** | `max_shrinks_in_window=1 shrink_token_seen=False consolidation_active=True shrink_blocked={` |
| S2 | No unmitigated collapse (end ≥ 0.5×peak) | **PASS** | `peak=16.44 end=22.08` |
| S3 | Early freeze triggered (30% drop) | **PASS** | `collapse_freeze_seen=True` |
| S4 | Hard collapse tolerance rollback | **N/A** | `hard_collapse=False forced_tol=False (N/A: no hard collapse)` |
| S5 | best_stable: every selected resolved (executed+skipped) | **PASS** | `selected=3 executed=3 skipped=0 selected_for_best_stable=3` |
| S6 | Recovery uses last_mastered tolerance | **PASS** | `tol_logs=38` |
| S7 | Hyperparam / mutation freeze audit | **PASS** | `mutation_frozen=True` |
| S8 | Fair eval protocol documented | **PASS** | `eval_protocol.md` |

## Population aggregates

### A. Completed-only aggregate

- policies: [0, 2, 3, 4, 5]
- mean end_mean: **12.698**
- median end_mean: **9.104**
- max end_mean: **22.08**
- min end_mean: **7.258**
- median retention pct: **94.35**
- min retention pct: **76.52**
- note: —

### B. All-observed aggregate (p1 partial)

- policies: [0, 1, 2, 3, 4, 5]
- mean end_mean: **12.622**
- median end_mean: **10.673**
- max end_mean: **22.08**
- min end_mean: **7.258**
- median retention pct: **—**
- min retention pct: **—**
- note: p1 end_mean snapshot is at 440M, not 800M; counts as observed not completed

## Stopped-smoke conclusions

- no-h3 fixed `value_imag_h1` smoke confirms stable RL/PBT learning in completed workers.
- PBT replacement works: cross=8, recovery=4 (lifetime max).
- p5 recovery burst is the strongest signal.
- p1 is an outlier requiring separate bad-seed / restart-integrity audit; see `P1_OUTLIER_AUDIT.md`.

## Recovery audit (`recovery_audit.jsonl`)

Records: **6**

- outcome=selected skipped=-
- outcome=executed skipped=-
- outcome=selected skipped=-
- outcome=executed skipped=-
- outcome=selected skipped=-
- outcome=executed skipped=-

## OOM Diagnosis (P2.1-F)

- OOM-like log lines: **13**
- `value_imag_h3` mentions: **38**
- Approximate overlap hint: **13**
- Recommendation: keep the main WM pool unchanged, and run an optional no_h3 smoke ablation.
- Optional launch knobs for next smoke:
  - `--wm_disable_value_imag_h3` (if implemented in launch wrapper), or
  - a dedicated no_h3 smoke script using fixed `value_imag_h1` / no-h3 strategy set.

## PBT population verification (no-h3 full smoke)

- num_policies_expected: **6**
- all_policies_started: **6** -> [0, 1, 2, 3, 4, 5]
- startup_hang_count: **0** (failed_after_retry: **0**)
- population_diversity_failed: **False** (policy curves have different lengths; not strictly identical)
- population end_mean: mean=**10.582** median=**9.104** n=6

### Acceptance

| ID | Criterion | Pass | Detail |
|----|-----------|------|--------|
| D1 | all_policies_started == num_policies | **PASS** | `seen=[0, 1, 2, 3, 4, 5] expected=6` |
| D2 | startup_hang_count == 0 (warn=1, fail>=2) | **PASS** | `hangs=0 failed=0` |
| D3 | population_diversity_failed == False | **PASS** | `policy curves have different lengths; not strictly identical` |
| D4 | >=1 PBT replacement OR all policies meet source thresholds | **PASS** | `cross=8 recovery=4 thresholds_met=False` |

### Restart counts by reason

- `skipped`: **154**

- self_restart total: **0**
- cross_restart total: **8**
- recovery_restart total: **4**

### Per-policy table

| pid | seed | env_hash | peak_mean | end_mean | retention | startup_hangs | cross | recovery |
|-----|------|----------|-----------|----------|-----------|----------------|-------|----------|
| p0 | 42 | a25f656dee9e8f85 | 16.463 | 16.463 | 100.00% | 0 | 0 | 0 |
| p1 | 1042 | d5235f3b8e631ebc | 19.217 | 0.000 | 0.00% | 0 | 3 | 1 |
| p2 | 2042 | 69c001a21fe67f84 | 11.221 | 8.587 | 76.52% | 0 | 1 | 1 |
| p3 | 3042 | 18966fa7cab1733b | 9.648 | 9.104 | 94.35% | 0 | 1 | 0 |
| p4 | 4042 | d50c78aa793f81af | 8.972 | 7.258 | 80.90% | 0 | 1 | 1 |
| p5 | 5042 | 005baa14de809636 | 22.283 | 22.080 | 99.09% | 0 | 2 | 1 |

### PBT actual cross-copy restart (+50M frames)

| pid | restart_type | frame@restart (M) | mean@restart | mean@+50M | delta |
|-----|--------------|-------------------|--------------|-----------|-------|
| p1 | cross | 200 | 0.387 | 0.000 | -0.387 |
| p1 | cross | 200 | 8.195 | 0.000 | -8.195 |
| p1 | cross | 200 | 8.569 | 0.000 | -8.569 |
| p2 | cross | 200 | 8.205 | 0.000 | -8.205 |
| p3 | cross | 200 | 3.531 | 3.705 | 0.174 |
| p4 | cross | 200 | 2.110 | 2.457 | 0.347 |
| p5 | cross | 200 | 0.000 | 0.000 | 0.000 |
| p5 | cross | 200 | 7.479 | 0.000 | -7.479 |

### PBT actual recovery restart (+50M frames)

| pid | restart_type | frame@restart (M) | mean@restart | mean@+50M | delta |
|-----|--------------|-------------------|--------------|-----------|-------|
| p1 | recovery | 200 | 10.475 | 0.000 | -10.475 |
| p2 | recovery | 200 | 6.502 | 0.000 | -6.502 |
| p4 | recovery | 200 | 5.890 | 2.457 | -3.433 |
| p5 | recovery | 200 | 5.073 | 0.000 | -5.073 |

### PBT skipped ticks (last_restart_type=3, NOT a restart)

- count: **154**
- reasons (top 5):
  - `(blank)`: **78**
  - `restart_reset`: **42**
  - `cooldown_blocked`: **25**
  - `cooldown_elapsed`: **9**

### Self-improvement delta (last_restart_type=0, no restart)

- n=126 mean_delta=1.438 median_delta=0.374 max_delta=17.946 min_delta=-8.664
## Diagnostic Indicators (P2.1 no-h3 prep)

### nsucc_max saturation time (first env_frames with nsucc_max ≥ 50)

| Policy | First saturation (M frames) |
|--------|----------------------------|
| p0 | 640.0 |
| p1 | 200.0 |
| p2 | 200.0 |
| p3 | 320.1 |
| p4 | 200.0 |
| p5 | 200.0 |

### nsucc_mean retention (end_mean / peak_mean)

| Policy | peak_mean | end_mean | retention |
|--------|-----------|----------|-----------|
| p0 | 16.46 | 16.46 | 100.00% |
| p1 | 19.22 | nan | — |
| p2 | 11.22 | 8.59 | 76.52% |
| p3 | 9.65 | 9.10 | 94.35% |
| p4 | 8.97 | 7.26 | 80.90% |
| p5 | 22.28 | 22.08 | 99.09% |

- Median retention: **94.35%**
- Min retention: **76.52%**

### killed / restart count

| Policy | Killed (train.log) | restart_count (curve tail) |
|--------|--------------------|----------------------------|
| p0 | 0 | 0 |
| p1 | 5 | 0 |
| p2 | 2 | 2 |
| p3 | 1 | 1 |
| p4 | 2 | 2 |
| p5 | 3 | 3 |

- Total killed: **13**
- Max restart_count across policies: **3**

### OOM by wm_strategy

- `value_imag_h1`: **13**

### source tier distribution (`last_source_tier_label`)

- `none`: **222**
- `strict_stable_live`: **55**
- `usable_live`: **9**

### tolerance mode distribution

**Recovery (`pbt_recovery_tolerance_mode`):**

- `last_mastered`: **286**

**Cross-copy (`pbt_crosscopy_tolerance_mode`):**

- `last_mastered`: **286**

### cooldown_blocked count (from training curve `rollback_reason`)

- Total: **25**
- By policy:
  - p1: **17**
  - p2: **8**

---

# P1 Smoke Acceptance Report (800M)

**Run ID:** `regrasping_cbwm_smoke_800m_p2_1_no_h3_20260605_135426`
**Generated:** 2026-06-05 22:39 CST
**Overall:** 7/7 criteria passed

## P1 acceptance criteria

| ID | Criterion | Pass | Detail |
|----|-----------|------|--------|
| A1 | Peak nsucc_mean @480M±40M ≥ 3.5 | **PASS** | `peak=16.443 policy=5` |
| A2 | Best policy nsucc_mean @end ≥ 2.0 | **PASS** | `end=22.080 policy=5` |
| A3 | No monotonic collapse (end ≥ 0.5×peak) | **PASS** | `degradation=False` |
| B1 | Mastery shrink / gate activity | **PASS** | `{'mastery_shrink_log_lines': 38, 'collapse_log_lines': 0, 'control_allow_shrink_` |
| B2 | Collapse freeze detected (if collapse) | **PASS** | `collapse_logs=0` |
| C1 | Stable-source tier used | **PASS** | `{'strict_stable_live': 55, 'usable_live': 9, 'relaxed_live': 0, 'best_stable_arc` |
| C2 | CF gated on stable source or skipped | **PASS** | `cf_blocks=0 cf_skip_logs=0` |

## P1 tier breakdown

- `archive`: 0
- `best_stable_archive`: 0
- `relaxed_live`: 0
- `skipped`: 0
- `strict_stable_live`: 55
- `usable_live`: 9

## Curriculum control snapshots

Control files found: **6**

- `000.json`: allow_shrink=False mastery_consec=0 collapse_freeze=True
- `001.json`: allow_shrink=False mastery_consec=2 collapse_freeze=False
- `002.json`: allow_shrink=False mastery_consec=2 collapse_freeze=False
- `003.json`: allow_shrink=False mastery_consec=2 collapse_freeze=False
- `004.json`: allow_shrink=False mastery_consec=2 collapse_freeze=False
- `005.json`: allow_shrink=False mastery_consec=2 collapse_freeze=False

---

## Base CB-WM smoke report (Patches A-J)

# CB-WM DexPBT Smoke Report (Patches A-J)

**Run ID:** `regrasping_cbwm_smoke_800m_p2_1_no_h3_20260605_135426`
**Generated:** 2026-06-05 22:39 CST
**Run dir:** `/root/autodl-tmp/wmdexpbt/results/regrasping_cbwm_smoke_800m_p2_1_no_h3_20260605_135426`
**Max live frames (log est.):** 801.11M
**Training curve points:** 286
**PBT restart manifests:** 0

## Smoke schedule overrides

```
smoke_run=true
patch=P2_stability
target_frames=800000000
max_iterations=6112
curriculum_mode=mastery
pbt_single_shrink_per_window=true
pbt_early_collapse_detect=true
pbt_recovery_tolerance_mode=last_mastered
pbt_recovery_bypass_elimination_burnin=true
note=P2 smoke: single-shot shrink, early freeze, best_stable recovery audit
```

## 1. source_fallback count

- Manifest fallback restarts: **0**
- Log lines matching fallback: **38**

## 2. self_restart / cross_restart count

- self_restart (max across policies): **0**
- cross_restart (max across policies): **3**
- recovery_restart (archive, Patch G): **1**

## 3. skip replacement count

- skip events (tier_skipped counter or log grep): **469**

## 4. eligible source count

- eligible PBT ticks (sum): **106**
- ineligible PBT ticks (sum): **180**
- live eligible policies at end: **5**

Ineligible reasons:

- `frames_since_restart<50000000`: 114
- `nsucc_mean<1.0`: 66

## 5. best_nsucc / mean_nsucc

- peak nsucc_max (all ticks): **50.000**
- live best nsucc_max: **50.000**
- live mean nsucc_mean: **10.582**
- live mean nsucc_max: **41.667**

### Per-policy live frames (epoch est.)

- policy_00: 801.11M frames
- policy_01: 720.11M frames
- policy_02: 801.11M frames
- policy_03: 801.11M frames
- policy_04: 801.11M frames
- policy_05: 801.11M frames
- policy_06: 0.00M frames
- policy_07: 0.00M frames

## 6. CF child_train_frames

- CF blocks completed: **0**
- child_train_frames (min/mean/max): **0 / 0 / 0**


## 7. posterior_update_count

- **0**

## 8. wm_posterior mean / n

Source: `/root/autodl-tmp/wmdexpbt/results/regrasping_cbwm_smoke_800m_p2_1_no_h3_20260605_135426/dexpbt_8gpu_ws/wm_posterior_latest.json`

| strategy | mean | n |
|----------|------|---|
| aux_low | 0.0000 | 1 |
| aux_mid | 0.0000 | 1 |
| none | 0.0000 | 1 |
| value_imag_h1 | 0.0000 | 1 |
| value_imag_h3 | 0.0000 | 1 |

## 9. nsucc 先升后降 (pre-post degradation)

**Detected:** NO

No clear pre-post degradation pattern in training_curve (or insufficient PBT ticks).

## Tier breakdown (Patch G, last tick per policy)

- `archive`: 0
- `relaxed_live`: 0
- `skipped`: 0
- `strict_live`: 0
