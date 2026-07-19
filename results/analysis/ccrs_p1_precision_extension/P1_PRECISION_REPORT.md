# p1 tip fixed-precision extension — final analysis

Checkpoint: p1 it160 (5B frames). 1.8 cm cells reused read-only from ccrs_base_retention; 1.6–1.0 cm cells from the extension sweep. Same protocol (policy_only, deterministic, 512 envs, 512 first episodes, paired resets, M=50).

| tol (cm) | nsucc_mean [95% CI] | normalized_nsucc [95% CI] | terminal err q50 | dropped rate | n | source |
|---|---|---|---|---|---|---|
| 1.8 | 40.8 [40.0, 41.6] | 0.815 [0.799, 0.831] | 0.011 | 0.177 | 2048 | ccrs_base_retention (reused, not re-run) |
| 1.6 | 37.0 [35.8, 38.1] | 0.739 [0.717, 0.761] | 0.013 | 0.162 | 2048 | extension |
| 1.4 | 28.8 [27.8, 29.8] | 0.577 [0.556, 0.597] | 0.017 | 0.124 | 2048 | extension |
| 1.2 | 14.8 [14.0, 15.6] | 0.296 [0.280, 0.313] | 0.017 | 0.075 | 2048 | extension |
| 1.0 | 4.7 [4.3, 5.0] | 0.093 [0.087, 0.100] | 0.016 | 0.033 | 2048 | extension |

## Decision (pre-registered)

- nsucc_mean at fixed 1.0 cm: **4.7**
- label: **PRECISION_LIMIT**
  (TARGET_REACHED >= 40; NEAR_TARGET >= 30; PRECISION_LIMIT below)

## Recommendation (single)

- action: **SENTINEL_GRID_RETENTION**
- rationale: 1 cm capability is far from target (nsucc < 30); capability drops clearly from 1.4 cm on.

Figure: `figures/p1_precision_capability_curve.png`.

Single checkpoint, single population run; the 4 evaluation seeds quantify evaluation uncertainty only. This report does not modify the main CCRS conclusions.
