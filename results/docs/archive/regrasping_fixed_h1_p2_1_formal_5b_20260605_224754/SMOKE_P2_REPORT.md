# P2 Smoke Acceptance Report (800M)

**Run ID:** `regrasping_fixed_h1_p2_1_formal_5b_20260605_224754`
**Generated:** 2026-06-06 20:28 CST
**P2 criteria:** 7/8 passed (criteria with N/A excluded)
**D criteria (PBT verification):** 4/4 passed

| ID | Criterion | Pass | Detail |
|----|-----------|------|--------|
| S1 | Single-shot shrink (no multi-shrink within window) | **PASS** | `max_shrinks_in_window=0 shrink_token_seen=True consolidation_active=True shrink_blocked={'` |
| S2 | No unmitigated collapse (end ≥ 0.5×peak) | **PASS** | `peak=3.69 end=23.65` |
| S3 | Early freeze triggered (30% drop) | **PASS** | `collapse_freeze_seen=True` |
| S4 | Hard collapse tolerance rollback | **FAIL** | `hard_collapse=True forced_tol=False` |
| S5 | best_stable: every selected resolved (executed+skipped) | **PASS** | `selected=2 executed=2 skipped=0 selected_for_best_stable=2` |
| S6 | Recovery uses last_mastered tolerance | **PASS** | `tol_logs=387` |
| S7 | Hyperparam / mutation freeze audit | **PASS** | `mutation_frozen=True` |
| S8 | Fair eval protocol documented | **PASS** | `eval_protocol.md` |

## Recovery audit (`recovery_audit.jsonl`)

Records: **4**

- outcome=selected skipped=-
- outcome=executed skipped=-
- outcome=selected skipped=-
- outcome=executed skipped=-

## OOM Diagnosis (P2.1-F)

- OOM-like log lines: **187**
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
- population end_mean: mean=**12.003** median=**13.493** n=6

### Acceptance

| ID | Criterion | Pass | Detail |
|----|-----------|------|--------|
| D1 | all_policies_started == num_policies | **PASS** | `seen=[0, 1, 2, 3, 4, 5] expected=6` |
| D2 | startup_hang_count == 0 (warn=1, fail>=2) | **PASS** | `hangs=0 failed=0` |
| D3 | population_diversity_failed == False | **PASS** | `policy curves have different lengths; not strictly identical` |
| D4 | >=1 PBT replacement OR all policies meet source thresholds | **PASS** | `cross=182 recovery=2 thresholds_met=False` |

### Restart counts by reason

- `skipped`: **963**

- self_restart total: **0**
- cross_restart total: **182**
- recovery_restart total: **2**

### Per-policy table

| pid | seed | env_hash | peak_mean | end_mean | retention | startup_hangs | cross | recovery |
|-----|------|----------|-----------|----------|-----------|----------------|-------|----------|
| p0 | 42 | a25f656dee9e8f85 | 30.287 | 2.954 | 9.75% | 0 | 32 | 1 |
| p1 | 1042 | d5235f3b8e631ebc | 32.946 | 23.649 | 71.78% | 0 | 28 | 0 |
| p2 | 2042 | 69c001a21fe67f84 | 31.869 | 12.270 | 38.50% | 0 | 25 | 1 |
| p3 | 3042 | 18966fa7cab1733b | 30.512 | 18.936 | 62.06% | 0 | 29 | 0 |
| p4 | 4042 | d50c78aa793f81af | 32.312 | 13.493 | 41.76% | 0 | 32 | 0 |
| p5 | 5042 | 005baa14de809636 | 27.751 | 0.714 | 2.57% | 0 | 36 | 0 |

### PBT actual cross-copy restart (+50M frames)

| pid | restart_type | frame@restart (M) | mean@restart | mean@+50M | delta |
|-----|--------------|-------------------|--------------|-----------|-------|
| p0 | cross | 950 | 17.834 | 18.323 | 0.489 |
| p0 | cross | 1070 | 18.069 | 24.151 | 6.082 |
| p0 | cross | 1310 | 21.526 | 19.530 | -1.996 |
| p0 | cross | 1430 | 16.639 | 17.873 | 1.234 |
| p0 | cross | 1590 | 21.009 | 23.982 | 2.973 |
| p0 | cross | 1810 | 22.539 | 17.695 | -4.844 |
| p0 | cross | 2030 | 27.211 | 29.802 | 2.591 |
| p0 | cross | 2130 | 29.802 | 26.181 | -3.621 |
| p0 | cross | 2330 | 22.760 | 24.240 | 1.481 |
| p0 | cross | 2450 | 20.007 | 22.979 | 2.972 |
| p0 | cross | 2650 | 14.726 | 22.876 | 8.150 |
| p0 | cross | 2810 | 12.196 | 26.714 | 14.519 |
| p0 | cross | 2930 | 13.014 | 7.824 | -5.190 |
| p0 | cross | 3030 | 7.824 | 14.880 | 7.056 |
| p0 | cross | 3150 | 16.520 | 14.244 | -2.276 |
| p0 | cross | 3250 | 14.244 | 19.691 | 5.447 |
| p0 | cross | 3530 | 10.059 | 12.306 | 2.247 |
| p0 | cross | 3630 | 12.306 | 0.000 | -12.306 |
| p0 | cross | 3730 | 0.000 | 13.705 | 13.705 |
| p0 | cross | 3830 | 13.705 | 8.370 | -5.335 |
| p0 | cross | 3930 | 8.370 | 22.176 | 13.806 |
| p0 | cross | 4050 | 8.359 | 21.053 | 12.694 |
| p0 | cross | 4270 | 4.798 | 24.804 | 20.006 |
| p0 | cross | 4470 | 14.985 | 13.590 | -1.395 |
| p0 | cross | 4610 | 9.434 | 17.169 | 7.735 |
| p0 | cross | 4830 | 0.034 | 0.000 | -0.034 |
| p0 | cross | 4930 | 0.000 | 7.204 | 7.204 |
| p0 | cross | 5030 | 7.204 | 12.177 | 4.973 |
| p0 | cross | 5150 | 6.973 | 13.830 | 6.858 |
| p0 | cross | 5270 | 0.092 | 10.588 | 10.495 |
| p0 | cross | 5370 | 10.588 | 18.310 | 7.723 |
| p0 | cross | 5530 | 4.352 | 3.778 | -0.573 |
| p1 | cross | 530 | 3.247 | 7.187 | 3.939 |
| p1 | cross | 650 | 8.259 | 11.847 | 3.588 |
| p1 | cross | 770 | 5.721 | 14.847 | 9.126 |
| p1 | cross | 870 | 14.847 | 16.821 | 1.974 |
| p1 | cross | 1050 | 20.077 | 19.150 | -0.927 |
| p1 | cross | 1210 | 6.155 | 18.597 | 12.442 |
| p1 | cross | 1370 | 17.963 | 21.026 | 3.063 |
| p1 | cross | 1650 | 22.820 | 26.144 | 3.324 |
| p1 | cross | 1790 | 21.791 | 29.405 | 7.615 |
| p1 | cross | 1910 | 18.889 | 27.940 | 9.050 |
| p1 | cross | 2330 | 18.223 | 28.521 | 10.298 |
| p1 | cross | 2450 | 20.995 | 30.637 | 9.642 |
| p1 | cross | 2670 | 29.166 | 32.217 | 3.051 |
| p1 | cross | 2830 | 10.657 | 13.617 | 2.960 |
| p1 | cross | 2930 | 13.617 | 24.450 | 10.834 |
| p1 | cross | 3250 | 14.207 | 26.239 | 12.032 |
| p1 | cross | 3370 | 12.183 | 11.278 | -0.905 |
| p1 | cross | 3470 | 11.278 | 21.809 | 10.531 |
| p1 | cross | 3610 | 11.669 | 22.414 | 10.745 |
| p1 | cross | 3770 | 8.809 | 20.535 | 11.726 |
| p1 | cross | 3930 | 13.360 | 22.779 | 9.419 |
| p1 | cross | 4130 | 11.791 | 25.029 | 13.238 |
| p1 | cross | 4270 | 10.355 | 18.114 | 7.758 |
| p1 | cross | 4590 | 4.474 | 7.672 | 3.198 |
| p1 | cross | 4710 | 9.105 | 12.962 | 3.856 |
| p1 | cross | 5110 | 5.894 | 12.481 | 6.587 |
| p1 | cross | 5410 | 10.148 | 8.089 | -2.059 |
| p1 | cross | 5570 | 1.612 | 13.834 | 12.222 |
| p2 | cross | 510 | 3.447 | 5.061 | 1.614 |
| p2 | cross | 690 | 9.495 | 15.114 | 5.618 |
| p2 | cross | 950 | 20.967 | 25.018 | 4.050 |
| p2 | cross | 1110 | 22.358 | 19.606 | -2.752 |
| p2 | cross | 1510 | 23.065 | 27.363 | 4.299 |
| p2 | cross | 1890 | 26.647 | 24.491 | -2.156 |
| p2 | cross | 2110 | 23.349 | 11.506 | -11.843 |
| p2 | cross | 2210 | 11.506 | 24.459 | 12.953 |
| p2 | cross | 2330 | 17.512 | 11.254 | -6.258 |
| p2 | cross | 2710 | 22.264 | 25.477 | 3.213 |
| p2 | cross | 2830 | 5.673 | 12.737 | 7.064 |
| p2 | cross | 2930 | 12.737 | 10.443 | -2.294 |
| p2 | cross | 3030 | 10.443 | 14.026 | 3.583 |
| p2 | cross | 3170 | 13.857 | 21.196 | 7.339 |
| p2 | cross | 3410 | 10.516 | 23.326 | 12.810 |
| p2 | cross | 3790 | 11.237 | 20.851 | 9.613 |
| p2 | cross | 3930 | 11.690 | 20.029 | 8.339 |
| p2 | cross | 4070 | 11.906 | 18.777 | 6.871 |
| p2 | cross | 4210 | 10.073 | 22.182 | 12.109 |
| p2 | cross | 4350 | 11.918 | 4.830 | -7.088 |
| p2 | cross | 4450 | 4.830 | 8.186 | 3.356 |
| p2 | cross | 5150 | 7.673 | 13.114 | 5.441 |
| p2 | cross | 5270 | 9.698 | 23.144 | 13.446 |
| p2 | cross | 5410 | 0.174 | 2.060 | 1.886 |
| p2 | cross | 5510 | 2.060 | 10.582 | 8.522 |
| p3 | cross | 270 | 0.378 | 1.106 | 0.728 |
| p3 | cross | 530 | 4.122 | 4.287 | 0.165 |
| p3 | cross | 650 | 9.485 | 12.628 | 3.143 |
| p3 | cross | 750 | 12.628 | 18.424 | 5.796 |
| p3 | cross | 850 | 18.424 | 21.115 | 2.691 |
| p3 | cross | 1130 | 18.661 | 19.291 | 0.629 |
| p3 | cross | 1430 | 20.830 | 26.102 | 5.272 |
| p3 | cross | 1570 | 24.639 | 25.062 | 0.423 |
| p3 | cross | 1790 | 17.779 | 22.259 | 4.480 |
| p3 | cross | 1910 | 27.018 | 22.178 | -4.840 |
| p3 | cross | 2030 | 9.208 | 24.193 | 14.986 |
| p3 | cross | 2230 | 22.977 | 18.610 | -4.367 |
| p3 | cross | 2330 | 18.610 | 26.196 | 7.586 |
| p3 | cross | 2470 | 15.007 | 23.602 | 8.595 |
| p3 | cross | 2590 | 25.280 | 28.299 | 3.019 |
| p3 | cross | 2750 | 16.622 | 20.942 | 4.320 |
| p3 | cross | 3010 | 10.489 | 14.127 | 3.639 |
| p3 | cross | 3210 | 6.842 | 10.690 | 3.849 |
| p3 | cross | 3310 | 10.690 | 22.074 | 11.384 |
| p3 | cross | 3690 | 10.158 | 24.943 | 14.785 |
| p3 | cross | 4030 | 9.007 | 10.531 | 1.524 |
| p3 | cross | 4130 | 10.531 | 10.984 | 0.453 |
| p3 | cross | 4230 | 10.984 | 11.964 | 0.980 |
| p3 | cross | 4370 | 11.786 | 27.129 | 15.343 |
| p3 | cross | 4490 | 7.815 | 17.229 | 9.414 |
| p3 | cross | 4870 | 10.649 | 22.454 | 11.805 |
| p3 | cross | 5210 | 9.345 | 6.676 | -2.670 |
| p3 | cross | 5310 | 6.676 | 6.605 | -0.071 |
| p3 | cross | 5450 | 5.467 | 22.850 | 17.383 |
| p4 | cross | 250 | 0.000 | 0.000 | 0.000 |
| p4 | cross | 350 | 0.000 | 2.075 | 2.075 |
| p4 | cross | 450 | 2.075 | 3.488 | 1.413 |
| p4 | cross | 650 | 8.863 | 8.236 | -0.627 |
| p4 | cross | 770 | 10.900 | 20.015 | 9.115 |
| p4 | cross | 870 | 20.015 | 17.964 | -2.051 |
| p4 | cross | 1050 | 19.006 | 19.890 | 0.884 |
| p4 | cross | 1150 | 19.890 | 14.682 | -5.208 |
| p4 | cross | 1250 | 14.682 | 19.115 | 4.433 |
| p4 | cross | 1350 | 19.115 | 19.808 | 0.693 |
| p4 | cross | 1510 | 24.484 | 27.457 | 2.973 |
| p4 | cross | 1710 | 23.916 | 21.546 | -2.371 |
| p4 | cross | 2090 | 21.579 | 22.923 | 1.344 |
| p4 | cross | 2210 | 18.458 | 23.029 | 4.571 |
| p4 | cross | 2350 | 18.054 | 23.605 | 5.551 |
| p4 | cross | 2770 | 19.593 | 19.065 | -0.528 |
| p4 | cross | 2870 | 19.065 | 9.047 | -10.018 |
| p4 | cross | 3150 | 13.111 | 20.449 | 7.338 |
| p4 | cross | 3370 | 11.038 | 7.066 | -3.973 |
| p4 | cross | 3470 | 7.066 | 11.314 | 4.248 |
| p4 | cross | 3570 | 11.314 | 19.953 | 8.639 |
| p4 | cross | 3770 | 11.891 | 18.656 | 6.765 |
| p4 | cross | 3990 | 11.465 | 23.091 | 11.626 |
| p4 | cross | 4110 | 12.640 | 20.859 | 8.219 |
| p4 | cross | 4370 | 3.139 | 16.379 | 13.240 |
| p4 | cross | 4470 | 16.379 | 1.666 | -14.713 |
| p4 | cross | 4570 | 1.666 | 7.969 | 6.303 |
| p4 | cross | 4670 | 7.969 | 10.858 | 2.889 |
| p4 | cross | 4770 | 10.858 | 10.125 | -0.733 |
| p4 | cross | 4930 | 0.586 | 4.327 | 3.740 |
| p4 | cross | 5030 | 4.327 | 13.905 | 9.578 |
| p4 | cross | 5210 | 7.151 | 15.079 | 7.928 |
| p5 | cross | 270 | 0.000 | 0.000 | 0.000 |
| p5 | cross | 370 | 0.000 | 2.887 | 2.887 |
| p5 | cross | 470 | 2.887 | 4.253 | 1.366 |
| p5 | cross | 590 | 4.939 | 5.317 | 0.378 |
| p5 | cross | 910 | 17.257 | 21.967 | 4.710 |
| p5 | cross | 1010 | 21.967 | 20.337 | -1.630 |
| p5 | cross | 1110 | 20.337 | 19.363 | -0.973 |
| p5 | cross | 1210 | 19.363 | 22.805 | 3.442 |
| p5 | cross | 1450 | 21.202 | 22.638 | 1.435 |
| p5 | cross | 1590 | 26.646 | 17.131 | -9.514 |
| p5 | cross | 1710 | 23.706 | 21.302 | -2.404 |
| p5 | cross | 1810 | 21.302 | 24.579 | 3.277 |
| p5 | cross | 1950 | 18.102 | 23.713 | 5.610 |
| p5 | cross | 2050 | 23.713 | 17.851 | -5.862 |
| p5 | cross | 2150 | 17.851 | 23.296 | 5.445 |
| p5 | cross | 2370 | 16.051 | 17.457 | 1.406 |
| p5 | cross | 2470 | 17.457 | 19.646 | 2.189 |
| p5 | cross | 2590 | 9.786 | 12.837 | 3.051 |
| p5 | cross | 2690 | 12.837 | 23.523 | 10.686 |
| p5 | cross | 2830 | 20.508 | 16.017 | -4.491 |
| p5 | cross | 2950 | 11.161 | 11.663 | 0.503 |
| p5 | cross | 3050 | 11.663 | 13.141 | 1.478 |
| p5 | cross | 3170 | 12.888 | 22.613 | 9.725 |
| p5 | cross | 3430 | 12.621 | 15.602 | 2.982 |
| p5 | cross | 3590 | 11.179 | 16.885 | 5.706 |
| p5 | cross | 3710 | 12.021 | 24.209 | 12.188 |
| p5 | cross | 3950 | 11.319 | 19.372 | 8.052 |
| p5 | cross | 4090 | 6.501 | 23.861 | 17.361 |
| p5 | cross | 4270 | 8.165 | 19.562 | 11.397 |
| p5 | cross | 4430 | 11.037 | 23.206 | 12.169 |
| p5 | cross | 4750 | 0.000 | 5.867 | 5.867 |
| p5 | cross | 4850 | 5.867 | 15.430 | 9.563 |
| p5 | cross | 5170 | 5.943 | 12.236 | 6.293 |
| p5 | cross | 5310 | 9.296 | 13.826 | 4.530 |
| p5 | cross | 5490 | 5.710 | 8.653 | 2.943 |
| p5 | cross | 5610 | 0.234 | — | — |

### PBT actual recovery restart (+50M frames)

| pid | restart_type | frame@restart (M) | mean@restart | mean@+50M | delta |
|-----|--------------|-------------------|--------------|-----------|-------|
| p0 | recovery | 4170 | 9.473 | 4.798 | -4.674 |
| p2 | recovery | 2430 | 11.254 | 28.974 | 17.720 |

### PBT skipped ticks (last_restart_type=3, NOT a restart)

- count: **963**
- reasons (top 5):
  - `(blank)`: **896**
  - `cooldown_blocked`: **60**
  - `cooldown_elapsed`: **7**

### Self-improvement delta (last_restart_type=0, no restart)

- n=489 mean_delta=-1.168 median_delta=-0.096 max_delta=20.829 min_delta=-26.043
## Diagnostic Indicators (P2.1 no-h3 prep)

### nsucc_max saturation time (first env_frames with nsucc_max ≥ 50)

| Policy | First saturation (M frames) |
|--------|----------------------------|
| p0 | 640.0 |
| p1 | 580.1 |
| p2 | 580.1 |
| p3 | 660.1 |
| p4 | 580.1 |
| p5 | 540.0 |

### nsucc_mean retention (end_mean / peak_mean)

| Policy | peak_mean | end_mean | retention |
|--------|-----------|----------|-----------|
| p0 | 30.29 | 2.95 | 9.75% |
| p1 | 32.95 | 23.65 | 71.78% |
| p2 | 31.87 | 12.27 | 38.50% |
| p3 | 30.51 | 18.94 | 62.06% |
| p4 | 32.31 | 13.49 | 41.76% |
| p5 | 27.75 | 0.71 | 2.57% |

- Median retention: **41.76%**
- Min retention: **2.57%**

### killed / restart count

| Policy | Killed (train.log) | restart_count (curve tail) |
|--------|--------------------|----------------------------|
| p0 | 34 | 33 |
| p1 | 28 | 28 |
| p2 | 26 | 26 |
| p3 | 30 | 29 |
| p4 | 32 | 32 |
| p5 | 37 | 36 |

- Total killed: **187**
- Max restart_count across policies: **36**

### OOM by wm_strategy

- `value_imag_h1`: **187**

### source tier distribution (`last_source_tier_label`)

- `none`: **929**
- `strict_stable_live`: **333**
- `usable_live`: **189**
- `relaxed_live`: **7**

### tolerance mode distribution

**Recovery (`pbt_recovery_tolerance_mode`):**

- `last_mastered`: **1458**

**Cross-copy (`pbt_crosscopy_tolerance_mode`):**

- `last_mastered`: **1458**

### cooldown_blocked count (from training curve `rollback_reason`)

- Total: **66**
- By policy:
  - p1: **17**
  - p2: **30**
  - p3: **1**
  - p4: **15**
  - p5: **3**

---

# P1 Smoke Acceptance Report (800M)

**Run ID:** `regrasping_fixed_h1_p2_1_formal_5b_20260605_224754`
**Generated:** 2026-06-06 20:28 CST
**Overall:** 7/7 criteria passed

## P1 acceptance criteria

| ID | Criterion | Pass | Detail |
|----|-----------|------|--------|
| A1 | Peak nsucc_mean @480M±40M ≥ 3.5 | **PASS** | `peak=3.688 policy=5` |
| A2 | Best policy nsucc_mean @end ≥ 2.0 | **PASS** | `end=23.649 policy=1` |
| A3 | No monotonic collapse (end ≥ 0.5×peak) | **PASS** | `degradation=True` |
| B1 | Mastery shrink / gate activity | **PASS** | `{'mastery_shrink_log_lines': 387, 'collapse_log_lines': 0, 'control_allow_shrink` |
| B2 | Collapse freeze detected (if collapse) | **PASS** | `collapse_logs=0` |
| C1 | Stable-source tier used | **PASS** | `{'strict_stable_live': 333, 'usable_live': 189, 'relaxed_live': 7, 'best_stable_` |
| C2 | CF gated on stable source or skipped | **PASS** | `cf_blocks=0 cf_skip_logs=0` |

## P1 tier breakdown

- `archive`: 0
- `best_stable_archive`: 0
- `relaxed_live`: 7
- `skipped`: 0
- `strict_stable_live`: 333
- `usable_live`: 189

## Curriculum control snapshots

Control files found: **6**

- `000.json`: allow_shrink=False mastery_consec=0 collapse_freeze=True
- `001.json`: allow_shrink=False mastery_consec=2 collapse_freeze=True
- `002.json`: allow_shrink=False mastery_consec=2 collapse_freeze=False
- `003.json`: allow_shrink=True mastery_consec=2 collapse_freeze=False
- `004.json`: allow_shrink=True mastery_consec=2 collapse_freeze=False
- `005.json`: allow_shrink=False mastery_consec=0 collapse_freeze=True

---

## Base CB-WM smoke report (Patches A-J)

# CB-WM DexPBT Smoke Report (Patches A-J)

**Run ID:** `regrasping_fixed_h1_p2_1_formal_5b_20260605_224754`
**Generated:** 2026-06-06 20:28 CST
**Run dir:** `/root/autodl-tmp/wmdexpbt/results/regrasping_fixed_h1_p2_1_formal_5b_20260605_224754`
**Max live frames (log est.):** 5000.00M
**Training curve points:** 1458
**PBT restart manifests:** 0

## Smoke schedule overrides

```
smoke_run=false
patch=P2_stability
target_frames=5000000000
max_iterations=38147
curriculum_mode=mastery
pbt_single_shrink_per_window=true
pbt_early_collapse_detect=true
pbt_recovery_tolerance_mode=last_mastered
pbt_recovery_bypass_elimination_burnin=true
note=fixed-h1 5B formal; no h3; no CF; staggered launch
```

## 1. source_fallback count

- Manifest fallback restarts: **0**
- Log lines matching fallback: **387**

## 2. self_restart / cross_restart count

- self_restart (max across policies): **0**
- cross_restart (max across policies): **36**
- recovery_restart (archive, Patch G): **1**

## 3. skip replacement count

- skip events (tier_skipped counter or log grep): **2679**

## 4. eligible source count

- eligible PBT ticks (sum): **824**
- ineligible PBT ticks (sum): **634**
- live eligible policies at end: **5**

Ineligible reasons:

- `frames_since_restart<100000000`: 568
- `nsucc_mean<1.0`: 66

## 5. best_nsucc / mean_nsucc

- peak nsucc_max (all ticks): **50.000**
- live best nsucc_max: **50.000**
- live mean nsucc_mean: **12.003**
- live mean nsucc_max: **43.500**

### Per-policy live frames (epoch est.)

- policy_00: 5000.00M frames
- policy_01: 5000.00M frames
- policy_02: 5000.00M frames
- policy_03: 5000.00M frames
- policy_04: 5000.00M frames
- policy_05: 5000.00M frames
- policy_06: 0.00M frames
- policy_07: 0.00M frames

## 6. CF child_train_frames

- CF blocks completed: **0**
- child_train_frames (min/mean/max): **0 / 0 / 0**


## 7. posterior_update_count

- **0**

## 8. wm_posterior mean / n

Source: `/root/autodl-tmp/wmdexpbt/results/regrasping_fixed_h1_p2_1_formal_5b_20260605_224754/dexpbt_8gpu_ws/wm_posterior_latest.json`

| strategy | mean | n |
|----------|------|---|
| aux_low | 0.0000 | 1 |
| aux_mid | 0.0000 | 1 |
| none | 0.0000 | 1 |
| value_imag_h1 | 0.0000 | 1 |
| value_imag_h3 | 0.0000 | 1 |

## 9. nsucc 先升后降 (pre-post degradation)

**Detected:** YES

| policy | peak nsucc_mean | final nsucc_mean | peak frame | final frame |
|--------|-----------------|------------------|------------|-------------|
| p00 | 30.287 | 2.954 | 2160.1M | 5620.1M |
| p02 | 31.869 | 12.270 | 2600.1M | 5640.0M |
| p03 | 30.512 | 18.936 | 2700.1M | 5660.1M |
| p04 | 32.312 | 13.493 | 1660.0M | 5640.0M |
| p05 | 27.751 | 0.714 | 2260.1M | 5660.1M |

## Tier breakdown (Patch G, last tick per policy)

- `archive`: 0
- `relaxed_live`: 0
- `skipped`: 0
- `strict_live`: 0
