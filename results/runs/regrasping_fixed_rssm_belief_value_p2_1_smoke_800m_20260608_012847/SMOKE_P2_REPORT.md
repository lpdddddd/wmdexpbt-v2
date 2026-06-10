# P2 Smoke Acceptance Report (800M)

**Run ID:** `regrasping_fixed_rssm_belief_value_p2_1_smoke_800m_20260608_012847`
**Generated:** 2026-06-08 09:59 CST
**Frame range:** all
**[FATAL] pool contains forbidden strategy — see CB-WM section**
**P2 criteria:** 6/7 passed (criteria with N/A excluded)
**D criteria (PBT verification):** 4/4 passed

| ID | Criterion | Pass | Detail |
|----|-----------|------|--------|
| S1 | Single-shot shrink (no multi-shrink within window) | **PASS** | `max_shrinks_in_window=0 shrink_token_seen=False consolidation_active=True shrink_blocked={` |
| S2 | No unmitigated collapse (end ≥ 0.5×peak) | **PASS** | `peak=31.52 end=31.02` |
| S3 | Early freeze triggered (30% drop) | **FAIL** | `collapse_freeze_seen=False` |
| S4 | Hard collapse tolerance rollback | **N/A** | `hard_collapse=False (N/A: no first-trigger seen in curves)` |
| S5 | best_stable: every selected resolved (executed+skipped) | **PASS** | `selected=0 executed=0 skipped=0 selected_for_best_stable=0` |
| S6 | Recovery uses last_mastered tolerance | **PASS** | `tol_logs=74` |
| S7 | Hyperparam / mutation freeze audit | **PASS** | `mutation_frozen=True` |
| S8 | Fair eval protocol documented | **PASS** | `eval_protocol.md` |

## Recovery audit (`recovery_audit.jsonl`)

Records: **0**


## OOM Diagnosis (P2.1-F)

- OOM-like log lines: **31**
- `value_imag_h3` mentions: **0**
- Approximate overlap hint: **0**
- Recommendation: keep the main WM pool unchanged, and run an optional no_h3 smoke ablation.
- Optional launch knobs for next smoke:
  - `--wm_disable_value_imag_h3` (if implemented in launch wrapper), or
  - a dedicated no_h3 smoke script using fixed `value_imag_h1` / no-h3 strategy set.

## PBT population verification (no-h3 full smoke)

- num_policies_expected: **6**
- all_policies_started: **6** -> [0, 1, 2, 3, 4, 5]
- startup_hang_count: **0** (failed_after_retry: **0**)
- population_diversity_failed: **False** (policy curves have different lengths; not strictly identical)
- population end_mean: mean=**29.942** median=**30.322** n=6

### Acceptance

| ID | Criterion | Pass | Detail |
|----|-----------|------|--------|
| D1 | all_policies_started == num_policies | **PASS** | `seen=[0, 1, 2, 3, 4, 5] expected=6` |
| D2 | startup_hang_count == 0 (warn=1, fail>=2) | **PASS** | `hangs=0 failed=0` |
| D3 | population_diversity_failed == False | **PASS** | `policy curves have different lengths; not strictly identical` |
| D4 | >=1 PBT replacement OR all policies meet source thresholds | **PASS** | `cross=28 recovery=0 thresholds_met=True` |

### Restart counts by reason

- `skipped`: **81**

- self_restart total: **0**
- cross_restart total: **28**
- recovery_restart total: **0**

### Per-policy table

| pid | seed | env_hash | peak_mean | end_mean | retention | startup_hangs | cross | recovery |
|-----|------|----------|-----------|----------|-----------|----------------|-------|----------|
| p0 | 42 | a25f656dee9e8f85 | 31.774 | 30.949 | 97.40% | 0 | 2 | 0 |
| p1 | 1042 | d5235f3b8e631ebc | 30.482 | 28.172 | 92.42% | 0 | 8 | 0 |
| p2 | 2042 | 69c001a21fe67f84 | 31.583 | 31.017 | 98.21% | 0 | 4 | 0 |
| p3 | 3042 | 18966fa7cab1733b | 34.567 | 30.322 | 87.72% | 0 | 9 | 0 |
| p4 | 4042 | d50c78aa793f81af | 32.190 | 30.080 | 93.44% | 0 | 4 | 0 |
| p5 | 5042 | 005baa14de809636 | 33.022 | 29.115 | 88.17% | 0 | 1 | 0 |

### PBT actual cross-copy restart (+50M frames)

| pid | restart_type | frame@restart (M) | mean@restart | mean@+50M | delta |
|-----|--------------|-------------------|--------------|-----------|-------|
| p0 | cross | 750 | 29.388 | 31.516 | 2.128 |
| p0 | cross | 850 | 31.516 | — | — |
| p1 | cross | 330 | 0.520 | 0.742 | 0.222 |
| p1 | cross | 380 | 0.742 | 2.015 | 1.273 |
| p1 | cross | 430 | 2.015 | 0.627 | -1.388 |
| p1 | cross | 480 | 0.627 | 1.305 | 0.678 |
| p1 | cross | 530 | 1.305 | 3.456 | 2.151 |
| p1 | cross | 581 | 3.456 | 29.950 | 26.495 |
| p1 | cross | 631 | 29.950 | 29.053 | -0.898 |
| p1 | cross | 790 | 26.993 | — | — |
| p2 | cross | 410 | 25.201 | 29.856 | 4.655 |
| p2 | cross | 490 | 29.856 | 29.037 | -0.818 |
| p2 | cross | 550 | 29.037 | 27.029 | -2.008 |
| p2 | cross | 710 | 30.689 | 31.309 | 0.620 |
| p3 | cross | 370 | 17.481 | 28.127 | 10.646 |
| p3 | cross | 430 | 28.127 | 28.134 | 0.007 |
| p3 | cross | 510 | 28.134 | 29.755 | 1.620 |
| p3 | cross | 560 | 29.755 | 28.019 | -1.736 |
| p3 | cross | 610 | 28.019 | 30.668 | 2.650 |
| p3 | cross | 660 | 30.668 | 28.007 | -2.662 |
| p3 | cross | 710 | 28.007 | 33.044 | 5.037 |
| p3 | cross | 760 | 33.044 | 30.322 | -2.722 |
| p3 | cross | 830 | 30.322 | — | — |
| p4 | cross | 290 | 13.034 | 17.188 | 4.155 |
| p4 | cross | 370 | 17.188 | 27.043 | 9.854 |
| p4 | cross | 510 | 27.806 | 29.569 | 1.763 |
| p4 | cross | 750 | 32.190 | 31.810 | -0.380 |
| p5 | cross | 290 | 6.368 | 18.989 | 12.621 |

### PBT actual recovery restart (+50M frames)

- (none)

### PBT skipped ticks (last_restart_type=3, NOT a restart)

- count: **81**
- reasons (top 5):
  - `(blank)`: **81**

### Self-improvement delta (last_restart_type=0, no restart)

- n=76 mean_delta=1.655 median_delta=0.803 max_delta=26.495 min_delta=-4.245
## Diagnostic Indicators (P2.1 no-h3 prep)

### nsucc_max saturation time (first env_frames with nsucc_max ≥ 50)

| Policy | First saturation (M frames) |
|--------|----------------------------|
| p0 | 280.1 |
| p1 | 630.6 |
| p2 | 220.1 |
| p3 | 280.1 |
| p4 | 290.2 |
| p5 | 320.1 |

### nsucc_mean retention (end_mean / peak_mean)

| Policy | peak_mean | end_mean | retention |
|--------|-----------|----------|-----------|
| p0 | 31.77 | 30.95 | 97.40% |
| p1 | 30.48 | 28.17 | 92.42% |
| p2 | 31.58 | 31.02 | 98.21% |
| p3 | 34.57 | 30.32 | 87.72% |
| p4 | 32.19 | 30.08 | 93.44% |
| p5 | 33.02 | 29.11 | 88.17% |

- Median retention: **93.44%**
- Min retention: **87.72%**

### killed / restart count

| Policy | Killed (train.log) | restart_count (curve tail) |
|--------|--------------------|----------------------------|
| p0 | 3 | 2 |
| p1 | 9 | 8 |
| p2 | 4 | 4 |
| p3 | 10 | 9 |
| p4 | 4 | 4 |
| p5 | 1 | 1 |

- Total killed: **31**
- Max restart_count across policies: **9**

### OOM by wm_strategy

- `rssm_belief_value`: **31**

### source tier distribution (`last_source_tier_label`)

- `none`: **163**

### tolerance mode distribution

**Recovery (`pbt_recovery_tolerance_mode`):**

- `last_mastered`: **163**

**Cross-copy (`pbt_crosscopy_tolerance_mode`):**

- `last_mastered`: **163**

### cooldown_blocked count (from training curve `rollback_reason`)

- Total: **0**

## CB-WM pool hard-check

- `RUN_CONFIG.json.wm_strategy_pool` = (not set)
- `wm_posterior_latest.json.__pool__` = `['none', 'aux_low', 'aux_mid', 'value_imag_h1', 'value_imag_h3']`
- **[FATAL] pool contains forbidden strategy: ['aux_low', 'value_imag_h3']**

## CB-WM run summary (Track C)

- CF blocks completed: **0** (acceptance >=3 → **FAIL**)
- posterior_update_count (sum of n-1 across strategies): **0** (acceptance >=3 → **FAIL**)
- strategies with posterior n>1: **0** (acceptance >=2 → **FAIL**)

### Posterior n / mean per strategy

| strategy | n | mean |
|----------|---|------|
| aux_low | 1 | 0.0000 |
| aux_mid | 1 | 0.0000 |
| none | 1 | 0.0000 |
| value_imag_h1 | 1 | 0.0000 |
| value_imag_h3 | 1 | 0.0000 |

### CF block stats

- blocks_completed: —
- recent_block_strategies: —
- skipped_reasons: —

### Selected-strategy distribution (training curve ticks)

- `rssm_belief_value`: 163

### Active strategy timeline per policy

- p0: 200M:rssm_belief_value
- p1: 200M:rssm_belief_value
- p2: 200M:rssm_belief_value
- p3: 200M:rssm_belief_value
- p4: 200M:rssm_belief_value
- p5: 200M:rssm_belief_value

### OOM / killed by WM strategy (from train.log)

- `rssm_belief_value`: 31

---

# P1 Smoke Acceptance Report (800M)

**Run ID:** `regrasping_fixed_rssm_belief_value_p2_1_smoke_800m_20260608_012847`
**Generated:** 2026-06-08 09:59 CST
**Overall:** 5/7 criteria passed

## P1 acceptance criteria

| ID | Criterion | Pass | Detail |
|----|-----------|------|--------|
| A1 | Peak nsucc_mean @480M±40M ≥ 3.5 | **PASS** | `peak=31.522 policy=0` |
| A2 | Best policy nsucc_mean @end ≥ 2.0 | **PASS** | `end=31.017 policy=2` |
| A3 | No monotonic collapse (end ≥ 0.5×peak) | **PASS** | `degradation=False` |
| B1 | Mastery shrink / gate activity | **PASS** | `{'mastery_shrink_log_lines': 74, 'collapse_log_lines': 0, 'control_allow_shrink_` |
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
- `001.json`: allow_shrink=False mastery_consec=2 collapse_freeze=False
- `002.json`: allow_shrink=False mastery_consec=2 collapse_freeze=False
- `003.json`: allow_shrink=False mastery_consec=2 collapse_freeze=False
- `004.json`: allow_shrink=False mastery_consec=2 collapse_freeze=False
- `005.json`: allow_shrink=False mastery_consec=2 collapse_freeze=False

---

## Base CB-WM smoke report (Patches A-J)

# CB-WM DexPBT Smoke Report (Patches A-J)

**Run ID:** `regrasping_fixed_rssm_belief_value_p2_1_smoke_800m_20260608_012847`
**Generated:** 2026-06-08 09:59 CST
**Run dir:** `/root/autodl-tmp/wmdexpbt/results/regrasping_fixed_rssm_belief_value_p2_1_smoke_800m_20260608_012847`
**Max live frames (log est.):** 801.11M
**Training curve points:** 163
**PBT restart manifests:** 0

## Smoke schedule overrides

(no SMOKE_CONFIG.txt — may be a full run dir)

## 1. source_fallback count

- Manifest fallback restarts: **0**
- Log lines matching fallback: **74**

## 2. self_restart / cross_restart count

- self_restart (max across policies): **0**
- cross_restart (max across policies): **9**
- recovery_restart (archive, Patch G): **0**

## 3. skip replacement count

- skip events (tier_skipped counter or log grep): **223**

## 4. eligible source count

- eligible PBT ticks (sum): **148**
- ineligible PBT ticks (sum): **15**
- live eligible policies at end: **6**

Ineligible reasons:

- `nsucc_mean<1.0`: 15

## 5. best_nsucc / mean_nsucc

- peak nsucc_max (all ticks): **50.000**
- live best nsucc_max: **50.000**
- live mean nsucc_mean: **29.942**
- live mean nsucc_max: **50.000**

### Per-policy live frames (epoch est.)

- policy_00: 801.11M frames
- policy_01: 801.11M frames
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

Source: `/root/autodl-tmp/wmdexpbt/results/regrasping_fixed_rssm_belief_value_p2_1_smoke_800m_20260608_012847/dexpbt_8gpu_ws/wm_posterior_latest.json`

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
