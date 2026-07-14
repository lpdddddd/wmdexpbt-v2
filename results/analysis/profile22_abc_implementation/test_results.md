# Profile22 ABC — Test Results (cache + Profile24 CLI wiring)

Date: 2026-07-14

## Commits

- `5779d0f` Optimize ABC snapshot validation caching — **61 tests OK** (cache spy: 2nd inspect does not re-call validate_snapshot_commit)
- `c890ccd` Wire Profile24 into train.py wm_strategy_profile choices

## Commands

```bash
bash -n results/launchers/start_regrasping_cser_dexpbt_progress_refine_abc_5b.sh
cd runtime_root/src/dexpbt_isaaclab
PYTHONPATH=. python -m py_compile dexpbt_isaaclab/pbt/pbt_live_source_health.py dexpbt_isaaclab/pbt/pbt_observer.py dexpbt_isaaclab/scripts/train.py tests/test_profile22_abc.py
PYTHONPATH=. python -m unittest tests.test_profile22_abc tests.test_progress_refine.TestProfile tests.test_readiness_adaptive.TestProfile23Wiring
```

## Launch

See `ABC_5B_LAUNCH.md`. Active run uses `c890ccd`.
