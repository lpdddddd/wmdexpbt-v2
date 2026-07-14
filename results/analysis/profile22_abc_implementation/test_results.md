# Profile22 ABC — Test Results (snapshot validation cache)

Date: 2026-07-14  
Parent tip before this fix: `358133f`

## Commands

```bash
bash -n results/launchers/start_regrasping_cser_dexpbt_progress_refine_abc_5b.sh
cd runtime_root/src/dexpbt_isaaclab
PYTHONPATH=. python -m py_compile dexpbt_isaaclab/pbt/pbt_live_source_health.py dexpbt_isaaclab/pbt/pbt_observer.py tests/test_profile22_abc.py
PYTHONPATH=. python -m unittest tests.test_profile22_abc tests.test_progress_refine.TestProfile tests.test_readiness_adaptive.TestProfile23Wiring
```

## Results

| Check | Result |
|---|---|
| bash -n ABC launcher | PASS |
| py_compile | PASS |
| unit + Profile22/23 wiring | **Ran 61 tests … OK** |
| Training during tests | **none** |

## Cache call-count evidence

- First legal inspect: `validate_snapshot_commit` = 1, full hash executed
- Identical second / post-selection: cache hit, validate count stays 1, `full_hash_executed=False`
- Profile22 ABC flags still off; Profile24 still authoritative all-on
