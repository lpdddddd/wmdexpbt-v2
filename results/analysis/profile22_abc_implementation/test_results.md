# Profile22 ABC Implementation — Test Results

Date: 2026-07-14  
Repo commit (parent tip, uncommitted ABC worktree): `b3cf704917318973f34b45916462c800a450f47b`  
Constraint: **no training processes** were started.

## Commands run

```bash
bash -n results/launchers/start_regrasping_cser_dexpbt_progress_refine_abc_5b.sh

cd runtime_root/src/dexpbt_isaaclab
PYTHONPATH=. python -m py_compile \
  dexpbt_isaaclab/pbt/pbt_chronic_bottom.py \
  dexpbt_isaaclab/pbt/pbt_observer.py \
  dexpbt_isaaclab/world_model/snippet_bank_tolerance.py \
  dexpbt_isaaclab/world_model/elite_snippet_bank.py \
  dexpbt_isaaclab/world_model/eir_broadcast.py \
  dexpbt_isaaclab/world_model/wm_strategy_profiles.py \
  dexpbt_isaaclab/scripts/train.py \
  tests/test_profile22_abc.py

PYTHONPATH=. python -m unittest \
  tests.test_profile22_abc \
  tests.test_progress_refine.TestProfile \
  tests.test_readiness_adaptive.TestProfile23Wiring
```

## Results

| Suite | Result |
|---|---|
| `bash -n` ABC launcher | PASS |
| `py_compile` touched modules | PASS |
| `tests.test_profile22_abc` (21) | PASS |
| `tests.test_progress_refine.TestProfile` | PASS |
| `tests.test_readiness_adaptive.TestProfile23Wiring` | PASS |
| Combined (`Ran 30 tests`) | **OK** |

## Coverage vs requested matrix

### Config
- Profile22 pins A/B/C enables to False; ABC turns them True — covered.
- CLI BooleanOptionalAction flags present — covered via profile expansion + launcher contents.
- Mid mutation / MPPI remain off — covered.

### PBT chronic (A)
- below 200M / promising / cooldown / probation / gap — covered (pure helpers).
- valid skill_delta + mixed-tol fallback + other invalid fail-close — covered.
- source prefers top; copy state machine / resume JSON — covered.
- Observer wiring exercised via compile + review (full observer integration not process-launched).

### Bank (B)
- 0% compatible reject overwrite — covered.
- compatible merge + capacity cap — covered.
- local reserve — covered.
- illegal tol fail-close — covered.
- save/load + old schema — covered.
- `merge_payload_into_bridge` rejection path — covered.

### WM belief (C)
- `_need_belief` predicate when MPPI off + snippet WM flag — covered (mirrored unit).
- Full GPU belief encode / reranker reason-id path — **not** run (would require training/GPU smoke); deferred to review/launch.

## Training process check

```
ps -ef | grep -E 'launch_8gpu|train.py'  → none
nvidia-smi compute apps                   → empty
```

**Conclusion:** no training / smoke / launcher execution from this task.
