# Fair Eval vs Training Metric Protocol (P2-F)

This document defines how to compare **training-window** metrics (PBT observer / `training_curve_policy_*.jsonl`) with **fair eval** (`eval_fair.py`) without mixing them in mastery gates or collapse detection.

## Training metric (`nsucc_mean` at PBT tick)

| Property | Definition |
|----------|------------|
| Pool | Mean over all parallel envs (`prev_episode_successes`, typically 8192) |
| Tolerance | Live `success_tolerance` from mastery curriculum |
| Policy | On-policy stochastic rollouts during training |
| Checkpoint | Current live weights (not a frozen `.pth` eval) |
| Use | Mastery gate, collapse freeze, stable-source, PBT ranking |

## Fair eval (`eval_fair.py`)

| Property | Definition |
|----------|------------|
| Pool | Mean over completed episodes (`--num_episodes`, default 512) |
| Tolerance | `--eval_tolerance_mode fixed` with `--fixed_success_tolerance`, or `checkpoint` from meta |
| Policy | Deterministic (`--deterministic true`) |
| Checkpoint | Explicit `.pth` + optional `strict=False` for WM aux heads |
| Use | Generalization reports, smoke metric-gap checks — **not** mastery gate |

## Logged fields (P2-F)

`eval_fair.py` CSV / JSON rows include:

- `policy_idx`, `checkpoint_iteration`, `env_frames`
- `eval_tolerance_mode`, `tolerance_applied`
- `running_norm_loaded` (whether checkpoint contains running-norm tensors)
- Standard: `nsucc_mean`, `nsucc_max`, `success_rate`, `r_meta_eval`

## Smoke acceptance (metric gap)

After an 800M P2 smoke run, fair eval at the training peak tick with **training tolerance** should be within **50%** of the training-window `nsucc_mean`. If not, flag `metric_gap` in the smoke report.

## Do not

- Use fair eval `nsucc_mean` for `mastery_nsucc_mean` or collapse thresholds.
- Compare fair eval @ fixed tol=0.01 directly to training mean @ live tol ~0.04 without noting tolerance mismatch.
