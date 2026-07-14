# Profile22 ABC Implementation — Test Results (post review fixes)

Date: 2026-07-14  
Parent tip before fix commit: `b529bb6`

## Commands

```bash
bash -n results/launchers/start_regrasping_cser_dexpbt_progress_refine_abc_5b.sh

cd runtime_root/src/dexpbt_isaaclab
PYTHONPATH=. python -m py_compile \
  dexpbt_isaaclab/pbt/pbt_chronic_bottom.py \
  dexpbt_isaaclab/pbt/pbt_live_source_health.py \
  dexpbt_isaaclab/pbt/pbt_observer.py \
  dexpbt_isaaclab/world_model/eir_belief.py \
  dexpbt_isaaclab/world_model/eir_broadcast.py \
  dexpbt_isaaclab/world_model/elite_snippet_bank.py \
  dexpbt_isaaclab/scripts/train.py \
  tests/test_profile22_abc.py

PYTHONPATH=. python -m unittest \
  tests.test_profile22_abc \
  tests.test_progress_refine.TestProfile \
  tests.test_readiness_adaptive.TestProfile23Wiring
```

## Results

| Check | Result |
|---|---|
| `bash -n` ABC launcher | PASS |
| `py_compile` | PASS |
| `tests.test_profile22_abc` + Profile22/23 wiring | **Ran 35 tests … OK** |
| Training processes after tests | **none** |

## Coverage highlights

- Common health rejects: ineligible, burn-in, lineage, commit, probation, collapse, NaN, missing fields  
- Chronic valid Δ + mixed-tol fallback + non-mixed invalid rejects  
- Lineage reset keeps cooldown / clears span+quota  
- Full-bank merge `incoming_kept_count > 0`  
- Production `need_eir_belief_cache` matrix + Profile22/24 flags  
- Profile24 authoritative overrides `--no-*` ABC flags  
