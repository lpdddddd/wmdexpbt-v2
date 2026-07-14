# Profile22 ABC Implementation — Test Results (snapshot freshness fix)

Date: 2026-07-14  
Parent tip before this fix: `cc1b3a1`

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
| `tests.test_profile22_abc` + Profile22/23 wiring | **Ran 50 tests … OK** |
| Training processes after tests | **none started** |
| GPU / runtime smoke | **not executed** |

## Coverage highlights

- Real tempfile workspace via production `snapshot_commit_barrier`
- Legal mid-train snapshot **without** `population_frame_complete.json` → health PASS
- Completion marker neither required nor a substitute for valid commit
- Corrupt LATEST / missing commit_dir / missing MANIFEST → reject
- model.pth / meta.json hash mismatch → reject
- lineage / population_frame / checkpoint-missing / stale-vs-LATEST → reject
- Invalid snapshot not chronicaly selectable despite high score
- Legal snapshot enables skill_delta + mixed-tol fallback paths
- Belief `int_or_missing(0) == 0`
- Incoming local reflux eviction counting
- Profile22 ABC flags still off; Profile24 still authoritative all-on
