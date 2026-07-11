# P2 Smoke Acceptance Report (800M)

**Run ID:** `regrasping_cbwm_smoke_800m_p2_1_no_h3_20260605_040330`
**Generated:** 2026-06-05 13:39 CST
**P2 criteria:** 5/8 passed
**D criteria (PBT verification):** 2/4 passed

| ID | Criterion | Pass | Detail |
|----|-----------|------|--------|
| S1 | Single-shot shrink token + consolidation | **FAIL** | `shrink_token=False consolidation=True` |
| S2 | No unmitigated collapse (end ≥ 0.5×peak) | **PASS** | `peak=0.02 end=17.34` |
| S3 | Early freeze triggered (30% drop) | **FAIL** | `collapse_freeze_seen=False` |
| S4 | Hard collapse tolerance rollback | **FAIL** | `forced_tol=False` |
| S5 | best_stable: executed or skipped_reason | **PASS** | `selected=0 executed=0 skipped=0` |
| S6 | Recovery uses last_mastered tolerance | **PASS** | `tol_logs=16` |
| S7 | Hyperparam / mutation freeze audit | **PASS** | `mutation_frozen=True` |
| S8 | Fair eval protocol documented | **PASS** | `eval_protocol.md` |

## Recovery audit (`recovery_audit.jsonl`)

Records: **0**


## OOM Diagnosis (P2.1-F)

- OOM-like log lines: **0**
- `value_imag_h3` mentions: **16**
- Approximate overlap hint: **0**
- Recommendation: keep the main WM pool unchanged, and run an optional no_h3 smoke ablation.
- Optional launch knobs for next smoke:
  - `--wm_disable_value_imag_h3` (if implemented in launch wrapper), or
  - a dedicated no_h3 smoke script using fixed `value_imag_h1` / no-h3 strategy set.

## PBT population verification (no-h3 full smoke)

- num_policies_expected: **5**
- all_policies_started: **4** -> [0, 2, 3, 4]
- startup_hang_count: **0** (failed_after_retry: **0**)
- population_diversity_failed: **True** (policies 0 and 2 produced bit-identical trajectories)
- population end_mean: mean=**17.337** median=**17.337** n=4

### Acceptance

| ID | Criterion | Pass | Detail |
|----|-----------|------|--------|
| D1 | all_policies_started == num_policies | **FAIL** | `seen=[0, 2, 3, 4] expected=5` |
| D2 | startup_hang_count == 0 (warn=1, fail>=2) | **PASS** | `hangs=0 failed=0` |
| D3 | population_diversity_failed == False | **FAIL** | `policies 0 and 2 produced bit-identical trajectories` |
| D4 | >=1 PBT replacement OR all policies meet source thresholds | **PASS** | `cross=0 recovery=0 thresholds_met=True` |

### Restart counts by reason

- `3`: **85**

- self_restart total: **0**
- cross_restart total: **0**
- recovery_restart total: **0**

### Per-policy table

| pid | seed | env_hash | peak_mean | end_mean | retention | startup_hangs | cross | recovery |
|-----|------|----------|-----------|----------|-----------|----------------|-------|----------|
| p0 | — | — | 17.337 | 17.337 | 100.00% | 0 | 0 | 0 |
| p2 | — | — | 17.337 | 17.337 | 100.00% | 0 | 0 | 0 |
| p3 | — | — | 17.337 | 17.337 | 100.00% | 0 | 0 | 0 |
| p4 | — | — | 17.337 | 17.337 | 100.00% | 0 | 0 | 0 |

### PBT replacement post-recovery (+50M frames)

| pid | restart_type | frame@restart (M) | mean@restart | mean@+50M | delta |
|-----|--------------|-------------------|--------------|-----------|-------|
| p0 | 3 | 600 | 5.018 | 9.812 | 4.794 |
| p0 | 3 | 640 | 8.602 | 13.004 | 4.401 |
| p0 | 3 | 740 | 15.285 | 17.337 | 2.051 |
| p2 | 3 | 220 | 0.000 | 0.000 | 0.000 |
| p3 | 3 | 240 | 0.000 | 0.000 | 0.000 |
| p3 | 3 | 640 | 8.602 | 13.004 | 4.401 |
| p3 | 3 | 740 | 15.285 | 17.337 | 2.051 |
| p4 | 3 | 220 | 0.000 | 0.000 | 0.000 |
## Diagnostic Indicators (P2.1 no-h3 prep)

### nsucc_max saturation time (first env_frames with nsucc_max ≥ 50)

| Policy | First saturation (M frames) |
|--------|----------------------------|
| p0 | 640.0 |
| p2 | 640.0 |
| p3 | 640.0 |
| p4 | 640.0 |

### nsucc_mean retention (end_mean / peak_mean)

| Policy | peak_mean | end_mean | retention |
|--------|-----------|----------|-----------|
| p0 | 17.34 | 17.34 | 100.00% |
| p2 | 17.34 | 17.34 | 100.00% |
| p3 | 17.34 | 17.34 | 100.00% |
| p4 | 17.34 | 17.34 | 100.00% |

- Median retention: **100.00%**
- Min retention: **100.00%**

### killed / restart count

| Policy | Killed (train.log) | restart_count (curve tail) |
|--------|--------------------|----------------------------|
| p0 | 0 | 0 |
| p1 | 0 | 0 |
| p2 | 0 | 0 |
| p3 | 0 | 0 |
| p4 | 0 | 0 |
| p5 | 0 | 0 |

- Total killed: **0**
- Max restart_count across policies: **0**

### OOM by wm_strategy

- No Killed lines with wm_strategy attribution found.

### source tier distribution (`last_source_tier_label`)

- `none`: **124**

### tolerance mode distribution

**Recovery (`pbt_recovery_tolerance_mode`):**

- `last_mastered`: **124**

**Cross-copy (`pbt_crosscopy_tolerance_mode`):**

- `last_mastered`: **124**

### cooldown_blocked count (from training curve `rollback_reason`)

- Total: **0**

---

# P1 Smoke Acceptance Report (800M)

**Run ID:** `regrasping_cbwm_smoke_800m_p2_1_no_h3_20260605_040330`
**Generated:** 2026-06-05 13:38 CST
**Overall:** 4/7 criteria passed

## P1 acceptance criteria

| ID | Criterion | Pass | Detail |
|----|-----------|------|--------|
| A1 | Peak nsucc_mean @480M±40M ≥ 3.5 | **FAIL** | `peak=0.022 policy=0` |
| A2 | Best policy nsucc_mean @end ≥ 2.0 | **PASS** | `end=17.337 policy=0` |
| A3 | No monotonic collapse (end ≥ 0.5×peak) | **PASS** | `degradation=False` |
| B1 | Mastery shrink / gate activity | **PASS** | `{'mastery_shrink_log_lines': 16, 'collapse_log_lines': 0, 'control_allow_shrink_` |
| B2 | Collapse freeze detected (if collapse) | **FAIL** | `collapse_logs=0` |
| C1 | Stable-source tier used | **FAIL** | `{'strict_stable_live': 0, 'usable_live': 0, 'relaxed_live': 0, 'best_stable_arch` |
| C2 | CF gated on stable source or skipped | **PASS** | `cf_blocks=0 cf_skip_logs=0` |

## P1 tier breakdown

- `archive`: 0
- `best_stable_archive`: 0
- `relaxed_live`: 0
- `skipped`: 0
- `strict_stable_live`: 0
- `usable_live`: 0

## Curriculum control snapshots

Control files found: **6**

- `000.json`: allow_shrink=False mastery_consec=2 collapse_freeze=False
- `001.json`: allow_shrink=False mastery_consec=0 collapse_freeze=False
- `002.json`: allow_shrink=False mastery_consec=2 collapse_freeze=False
- `003.json`: allow_shrink=False mastery_consec=2 collapse_freeze=False
- `004.json`: allow_shrink=False mastery_consec=2 collapse_freeze=False
- `005.json`: allow_shrink=False mastery_consec=0 collapse_freeze=False

---

## Base CB-WM smoke report (Patches A-J)

# CB-WM DexPBT Smoke Report (Patches A-J)

**Run ID:** `regrasping_cbwm_smoke_800m_p2_1_no_h3_20260605_040330`
**Generated:** 2026-06-05 13:38 CST
**Run dir:** `/root/autodl-tmp/wmdexpbt/results/regrasping_cbwm_smoke_800m_p2_1_no_h3_20260605_040330`
**Max live frames (log est.):** 801.11M
**Training curve points:** 124
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
- Log lines matching fallback: **16**

## 2. self_restart / cross_restart count

- self_restart (max across policies): **0**
- cross_restart (max across policies): **0**
- recovery_restart (archive, Patch G): **0**

## 3. skip replacement count

- skip events (tier_skipped counter or log grep): **246**

## 4. eligible source count

- eligible PBT ticks (sum): **48**
- ineligible PBT ticks (sum): **76**
- live eligible policies at end: **4**

Ineligible reasons:

- `nsucc_mean<1.0`: 76

## 5. best_nsucc / mean_nsucc

- peak nsucc_max (all ticks): **50.000**
- live best nsucc_max: **50.000**
- live mean nsucc_mean: **17.337**
- live mean nsucc_max: **50.000**

### Per-policy live frames (epoch est.)

- policy_00: 801.11M frames
- policy_01: 0.00M frames
- policy_02: 801.11M frames
- policy_03: 801.11M frames
- policy_04: 801.11M frames
- policy_05: 0.00M frames
- policy_06: 0.00M frames
- policy_07: 0.00M frames

## 6. CF child_train_frames

- CF blocks completed: **0**
- child_train_frames (min/mean/max): **0 / 0 / 0**


## 7. posterior_update_count

- **0**

## 8. wm_posterior mean / n

Source: `/root/autodl-tmp/wmdexpbt/results/regrasping_cbwm_smoke_800m_p2_1_no_h3_20260605_040330/dexpbt_8gpu_ws/wm_posterior_latest.json`

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
