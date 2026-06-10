# P2 Smoke Acceptance Report (800M)

**Run ID:** `regrasping_fixed_trans_belief_value_p2_1_smoke_800m_20260608_100048`
**Generated:** 2026-06-08 12:37 CST
**Frame range:** all
**[FATAL] pool contains forbidden strategy — see CB-WM section**
**P2 criteria:** 8/8 passed (criteria with N/A excluded)
**D criteria (PBT verification):** 4/4 passed

| ID | Criterion | Pass | Detail |
|----|-----------|------|--------|
| S1 | Single-shot shrink (no multi-shrink within window) | **PASS** | `max_shrinks_in_window=1 shrink_token_seen=False consolidation_active=True shrink_blocked={` |
| S2 | No unmitigated collapse (end ≥ 0.5×peak) | **PASS** | `peak=20.76 end=22.87` |
| S3 | Early freeze triggered (30% drop) | **PASS** | `collapse_freeze_seen=True` |
| S4 | Hard collapse tolerance rollback | **PASS** | `first_triggers=1 rolled_back=1 unrolled=0 unrolled_examples=[]` |
| S5 | best_stable: every selected resolved (executed+skipped) | **PASS** | `selected=0 executed=0 skipped=0 selected_for_best_stable=0` |
| S6 | Recovery uses last_mastered tolerance | **PASS** | `tol_logs=50` |
| S7 | Hyperparam / mutation freeze audit | **PASS** | `mutation_frozen=True` |
| S8 | Fair eval protocol documented | **PASS** | `eval_protocol.md` |

## Recovery audit (`recovery_audit.jsonl`)

Records: **0**


## OOM Diagnosis (P2.1-F)

- OOM-like log lines: **19**
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
- population end_mean: mean=**16.260** median=**20.403** n=6

### Acceptance

| ID | Criterion | Pass | Detail |
|----|-----------|------|--------|
| D1 | all_policies_started == num_policies | **PASS** | `seen=[0, 1, 2, 3, 4, 5] expected=6` |
| D2 | startup_hang_count == 0 (warn=1, fail>=2) | **PASS** | `hangs=0 failed=0` |
| D3 | population_diversity_failed == False | **PASS** | `policy curves have different lengths; not strictly identical` |
| D4 | >=1 PBT replacement OR all policies meet source thresholds | **PASS** | `cross=17 recovery=0 thresholds_met=True` |

### Restart counts by reason

- `skipped`: **90**

- self_restart total: **0**
- cross_restart total: **17**
- recovery_restart total: **0**

### Per-policy table

| pid | seed | env_hash | peak_mean | end_mean | retention | startup_hangs | cross | recovery |
|-----|------|----------|-----------|----------|-----------|----------------|-------|----------|
| p0 | 42 | a25f656dee9e8f85 | 24.240 | 22.870 | 94.35% | 0 | 0 | 0 |
| p1 | 1042 | d5235f3b8e631ebc | 2.094 | 2.094 | 100.00% | 0 | 7 | 0 |
| p2 | 2042 | 69c001a21fe67f84 | 17.870 | 14.405 | 80.61% | 0 | 3 | 0 |
| p3 | 3042 | 18966fa7cab1733b | 21.619 | 20.403 | 94.38% | 0 | 2 | 0 |
| p4 | 4042 | d50c78aa793f81af | 20.780 | 16.244 | 78.17% | 0 | 1 | 0 |
| p5 | 5042 | 005baa14de809636 | 21.546 | 21.546 | 100.00% | 0 | 4 | 0 |

### PBT actual cross-copy restart (+50M frames)

| pid | restart_type | frame@restart (M) | mean@restart | mean@+50M | delta |
|-----|--------------|-------------------|--------------|-----------|-------|
| p1 | cross | 330 | 0.071 | 1.131 | 1.059 |
| p1 | cross | 410 | 1.131 | 0.726 | -0.405 |
| p1 | cross | 460 | 0.726 | 1.077 | 0.351 |
| p1 | cross | 510 | 1.077 | 1.385 | 0.308 |
| p1 | cross | 560 | 1.385 | 1.110 | -0.275 |
| p1 | cross | 610 | 1.110 | 2.094 | 0.984 |
| p1 | cross | 660 | 2.094 | — | — |
| p2 | cross | 450 | 14.453 | 17.052 | 2.599 |
| p2 | cross | 500 | 17.052 | 17.672 | 0.619 |
| p2 | cross | 570 | 17.672 | 16.433 | -1.239 |
| p3 | cross | 290 | 15.108 | 16.715 | 1.608 |
| p3 | cross | 350 | 16.715 | 19.907 | 3.192 |
| p4 | cross | 290 | 14.182 | 17.258 | 3.076 |
| p5 | cross | 250 | 8.145 | 17.365 | 9.220 |
| p5 | cross | 690 | 17.436 | 21.193 | 3.757 |
| p5 | cross | 740 | 21.193 | 21.546 | 0.354 |
| p5 | cross | 790 | 21.546 | — | — |

### PBT actual recovery restart (+50M frames)

- (none)

### PBT skipped ticks (last_restart_type=3, NOT a restart)

- count: **90**
- reasons (top 5):
  - `(blank)`: **89**
  - `cooldown_elapsed`: **1**

### Self-improvement delta (last_restart_type=0, no restart)

- n=53 mean_delta=1.331 median_delta=0.354 max_delta=15.108 min_delta=-2.741
## Diagnostic Indicators (P2.1 no-h3 prep)

### nsucc_max saturation time (first env_frames with nsucc_max ≥ 50)

| Policy | First saturation (M frames) |
|--------|----------------------------|
| p0 | 220.1 |
| p1 | — |
| p2 | 340.0 |
| p3 | 290.2 |
| p4 | 290.2 |
| p5 | 260.0 |

### nsucc_mean retention (end_mean / peak_mean)

| Policy | peak_mean | end_mean | retention |
|--------|-----------|----------|-----------|
| p0 | 24.24 | 22.87 | 94.35% |
| p1 | 2.09 | 2.09 | 100.00% |
| p2 | 17.87 | 14.41 | 80.61% |
| p3 | 21.62 | 20.40 | 94.38% |
| p4 | 20.78 | 16.24 | 78.17% |
| p5 | 21.55 | 21.55 | 100.00% |

- Median retention: **94.38%**
- Min retention: **78.17%**

### killed / restart count

| Policy | Killed (train.log) | restart_count (curve tail) |
|--------|--------------------|----------------------------|
| p0 | 0 | 0 |
| p1 | 8 | 7 |
| p2 | 3 | 3 |
| p3 | 2 | 2 |
| p4 | 1 | 1 |
| p5 | 5 | 4 |

- Total killed: **19**
- Max restart_count across policies: **7**

### OOM by wm_strategy

- `trans_belief_value`: **19**

### source tier distribution (`last_source_tier_label`)

- `none`: **149**

### tolerance mode distribution

**Recovery (`pbt_recovery_tolerance_mode`):**

- `last_mastered`: **149**

**Cross-copy (`pbt_crosscopy_tolerance_mode`):**

- `last_mastered`: **149**

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

- `trans_belief_value`: 149

### Active strategy timeline per policy

- p0: 200M:trans_belief_value
- p1: 200M:trans_belief_value
- p2: 200M:trans_belief_value
- p3: 200M:trans_belief_value
- p4: 200M:trans_belief_value
- p5: 200M:trans_belief_value

### OOM / killed by WM strategy (from train.log)

- `trans_belief_value`: 19

## Compare to reference run

- this run: `regrasping_fixed_trans_belief_value_p2_1_smoke_800m_20260608_100048`
- reference: `regrasping_fixed_rssm_belief_value_p2_1_smoke_800m_20260608_012847`  (frame_range=all)

| metric | this | reference |
|--------|------|-----------|
| peak_mean median | 21.16 | 31.98 |
| peak_mean max | 24.24 | 34.57 |
| best 3-tick rolling median | 20.13 | 31.58 |
| time-to-peak (90% of own peak) | min=380M / median=660M / max=740M | min=480M / median=600M / max=760M |

---

# P1 Smoke Acceptance Report (800M)

**Run ID:** `regrasping_fixed_trans_belief_value_p2_1_smoke_800m_20260608_100048`
**Generated:** 2026-06-08 12:37 CST
**Overall:** 6/7 criteria passed

## P1 acceptance criteria

| ID | Criterion | Pass | Detail |
|----|-----------|------|--------|
| A1 | Peak nsucc_mean @480M±40M ≥ 3.5 | **PASS** | `peak=20.758 policy=3` |
| A2 | Best policy nsucc_mean @end ≥ 2.0 | **PASS** | `end=22.870 policy=0` |
| A3 | No monotonic collapse (end ≥ 0.5×peak) | **PASS** | `degradation=False` |
| B1 | Mastery shrink / gate activity | **PASS** | `{'mastery_shrink_log_lines': 50, 'collapse_log_lines': 0, 'control_allow_shrink_` |
| B2 | Collapse freeze detected (if collapse) | **PASS** | `collapse_logs=0` |
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
- `001.json`: allow_shrink=False mastery_consec=0 collapse_freeze=True
- `002.json`: allow_shrink=False mastery_consec=2 collapse_freeze=False
- `003.json`: allow_shrink=False mastery_consec=2 collapse_freeze=False
- `004.json`: allow_shrink=False mastery_consec=2 collapse_freeze=False
- `005.json`: allow_shrink=False mastery_consec=2 collapse_freeze=False

---

## Base CB-WM smoke report (Patches A-J)

# CB-WM DexPBT Smoke Report (Patches A-J)

**Run ID:** `regrasping_fixed_trans_belief_value_p2_1_smoke_800m_20260608_100048`
**Generated:** 2026-06-08 12:37 CST
**Run dir:** `/root/autodl-tmp/wmdexpbt/results/regrasping_fixed_trans_belief_value_p2_1_smoke_800m_20260608_100048`
**Max live frames (log est.):** 801.11M
**Training curve points:** 149
**PBT restart manifests:** 0

## Smoke schedule overrides

(no SMOKE_CONFIG.txt — may be a full run dir)

## 1. source_fallback count

- Manifest fallback restarts: **0**
- Log lines matching fallback: **50**

## 2. self_restart / cross_restart count

- self_restart (max across policies): **0**
- cross_restart (max across policies): **7**
- recovery_restart (archive, Patch G): **0**

## 3. skip replacement count

- skip events (tier_skipped counter or log grep): **259**

## 4. eligible source count

- eligible PBT ticks (sum): **131**
- ineligible PBT ticks (sum): **18**
- live eligible policies at end: **6**

Ineligible reasons:

- `nsucc_mean<1.0`: 18

## 5. best_nsucc / mean_nsucc

- peak nsucc_max (all ticks): **50.000**
- live best nsucc_max: **50.000**
- live mean nsucc_mean: **16.260**
- live mean nsucc_max: **47.000**

### Per-policy live frames (epoch est.)

- policy_00: 801.11M frames
- policy_01: 670.96M frames
- policy_02: 758.51M frames
- policy_03: 753.27M frames
- policy_04: 786.56M frames
- policy_05: 796.52M frames
- policy_06: 0.00M frames
- policy_07: 0.00M frames

## 6. CF child_train_frames

- CF blocks completed: **0**
- child_train_frames (min/mean/max): **0 / 0 / 0**


## 7. posterior_update_count

- **0**

## 8. wm_posterior mean / n

Source: `/root/autodl-tmp/wmdexpbt/results/regrasping_fixed_trans_belief_value_p2_1_smoke_800m_20260608_100048/dexpbt_8gpu_ws/wm_posterior_latest.json`

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
