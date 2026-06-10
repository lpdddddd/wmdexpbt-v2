# RSSM adaptive_fast curriculum vs reference -- 800M comparison

target_frames=800000000
target_tolerance=0.04
reference_is_strict=false

> **Caveat**: reference is the legacy fixed RSSM 800M run and may differ from the fastcurr commit/config (e.g. control-state schema, P2.1 stability flags). For a strict commit/config-equivalent comparison run start_regrasping_fixed_rssm_belief_value_normalcurr_800m.sh.

## Verdict
- overall: **MARGINAL**
- primary curriculum (final tolerance <= target): False
- primary safety (end_mean >= 15 AND no collapse): True
- stronger success bonus: False
  - primary curriculum: final tolerance median 0.0496 > target 0.0400 -- FAIL
  - primary safety: end_mean median 29.51 >= 15 AND no collapse -- PASS
  - stronger success: peak_mean=33.21148681640625 retention=0.9351632376022598 -- NOT MET

## Tolerance trajectory (fastcurr)
| frame budget | median | min | max | n |
|---|---|---|---|---|
| tolerance_at_200M | n/a | n/a | n/a | 0 |
| tolerance_at_400M | 0.0712 | 0.0677 | 0.0750 | 6 |
| tolerance_at_600M | 0.0603 | 0.0549 | 0.0750 | 6 |
| tolerance_at_800M | 0.0496 | 0.0411 | 0.0675 | 6 |
| final_tolerance | 0.0496 | 0.0411 | 0.0675 | 6 |
| min_tolerance | 0.0496 | 0.0411 | 0.0611 | 6 |

## Tolerance trajectory (reference)
| frame budget | median | min | max | n |
|---|---|---|---|---|
| tolerance_at_200M | n/a | n/a | n/a | 0 |
| tolerance_at_400M | 0.0712 | 0.0677 | 0.0750 | 6 |
| tolerance_at_600M | 0.0643 | 0.0643 | 0.0750 | 6 |
| tolerance_at_800M | 0.0580 | 0.0580 | 0.0712 | 6 |
| final_tolerance | 0.0580 | 0.0580 | 0.0712 | 6 |
| min_tolerance | 0.0580 | 0.0580 | 0.0677 | 6 |

## Population stats (median / min / max / n_policies)

### fastcurr
| metric | median | min | max | n |
|---|---|---|---|---|
| peak_mean | 33.2115 | 28.8357 | 35.0651 | 6 |
| end_mean | 29.5113 | 26.0819 | 35.0651 | 6 |
| retention | 0.9352 | 0.7776 | 1.0000 | 6 |

### reference
| metric | median | min | max | n |
|---|---|---|---|---|
| peak_mean | 31.9821 | 30.4818 | 34.5670 | 6 |
| end_mean | 31.3156 | 26.9928 | 34.5670 | 6 |
| retention | 0.9810 | 0.8855 | 1.0000 | 6 |

## Time-to-threshold (median across policies that reached it)
| threshold | fastcurr (env_frames) | reference (env_frames) |
|---|---|---|
| nsucc>=10 | 295108608 | 295108608 |
| nsucc>=20 | 385089536 | 395116544 |
| nsucc>=30 | 580255744 | 620036096 |

## Population reach
- count peak_mean >= 30: fastcurr=5 ref=6
- count peak_mean >= 35: fastcurr=1 ref=0
- count end_mean >= 15: fastcurr=6 ref=6

## Shrink event statistics (fastcurr)
- median shrinks per policy: 6.0
- shrink-factor histogram: {'low_~0.95': 15, 'mid_~0.925': 11, 'other_~0.732': 1, 'strong_~0.90': 5, 'other_~0.811': 1, 'other_~0.855': 1}
- median rollback_after_shrink per policy: 0.0

## Shrink event statistics (reference)
- median shrinks per policy: 4.0
- shrink-factor histogram: {'low_~0.95': 22, 'other_~0.774': 1, 'strong_~0.90': 1}
- median rollback_after_shrink per policy: 0.0

## PBT event totals (last row per policy)
- fastcurr: cross={'median': 4.0, 'min': 0.0, 'max': 7.0, 'n': 6}, recovery={'median': 0.0, 'min': 0.0, 'max': 0.0, 'n': 6}, rollback={'median': 0.0, 'min': 0.0, 'max': 0.0, 'n': 6}
- reference: cross={'median': 4.0, 'min': 1.0, 'max': 8.0, 'n': 6}, recovery={'median': 0.0, 'min': 0.0, 'max': 0.0, 'n': 6}, rollback={'median': 0.0, 'min': 0.0, 'max': 0.0, 'n': 6}

## Engineering / safety
- fastcurr: engineering={'sigkill_137_total': 25, 'runtime_error_total': 0, 'fail_fast_total': 0}, fps={'fps_step_mean': 191720.64666666664, 'fps_total_mean': 111528.09333333334}
- reference: engineering={'sigkill_137_total': 31, 'runtime_error_total': 0, 'fail_fast_total': 0}, fps={'fps_step_mean': 169478.5, 'fps_total_mean': 101904.84666666666}

## Notes for adaptive_fast acceptance reading
- Inspect TB tags `curriculum/adaptive_shrink_factor`, `curriculum/retention_used_for_shrink`, `curriculum/blocked_by_*`, `curriculum/pre_shrink_tolerance`, `curriculum/post_shrink_tolerance` for the fastcurr run directly (no aggregation here).
- Fail conditions (plan section 8): primary curriculum AND primary safety both miss -> archive adaptive_fast as a curriculum negative ablation; keep normal mode as the main 3B candidate.
