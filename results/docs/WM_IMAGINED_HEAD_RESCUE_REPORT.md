# WM Imagined Head Rescue Report

## Summary

Patch adds calibration-aware imagined-validity modes, balanced head BCE (`pos_weight`), and telemetry for WM imagined heads used in PBT floor-recovery ranking.

## What was broken

- Sparse head labels kept logits near zero → `imagined_valid` always false under strict uncertainty gate.
- PBT WM source selector hard-failed when `imagined_valid=False`.

## Fixes (default OFF; enabled in strategy-zoo launcher)

- `seq_wm_head_balance_enable` + auto/manual `pos_weight` on horizon/recovery heads.
- `wm_imagined_valid_mode=soft_calibrated` with strict/soft validity and `wm_validity_weight`.
- Head lambdas (`seq_wm_lambda_horizon_success/drop_risk/recovery`) set to 0.5 in actor-distill profiles.

## Diagnostics

- Training: `seq_wm_head/*` via `_run_wm_runtime_wiring`.
- Eval: `wm_imagined/<pidx>.json` includes `imagined_valid_strict/soft`, `head_has_signal`, `valid_reason`, `checkpoint_age_frames`.

## Tests

- `tests/test_wm_imagined_head_diagnostics.py`

## Remaining limits

- Curriculum shrink still requires strict `imagined_valid_strict` (real-signal safety unchanged).
- Live-model eval only (`stale_reason=live_model`).
