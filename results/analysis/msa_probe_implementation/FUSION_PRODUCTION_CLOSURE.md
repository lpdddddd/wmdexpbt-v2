# MSA Fusion Production Closure

## Scope

Blocking patch on `base-retention` atop `f1ca5b4` to close the production loop:

```text
proposal → PROBE → real training → paired eval (or fixture evidence)
  → ADVANCE / ROLLBACK → per-worker ack → cycle complete
```

Formal long experiments are **not** started by this patch.

## Implementation map

| Process | Owns | Communicates via |
|---------|------|------------------|
| Trainer + `DexPBTPbtAlgoObserver` | Live weights, optimizer, env, `msa_worker_state/<wid>.json`, probe anchors under `msa_probe_anchors/` | Reads `MSA_BUSY.json`; claims `msa_fusion_protocol/cmds/<wid>/`; writes acks |
| MSA sidecar | Cycle orchestration, paired eval (pilot), durable tol pin/commit, fusion adapter fan-out | Writes `msa_cycles/*.json`, claims once → cmds; waits acks; `--fusion-fixture-once` is test-only |
| Continuation guard | Readiness / cap / park | `FIRST_TICK_VALIDATED.json`, holds |
| Hold watchdog | Fail-closed if sidecar/guard die under busy/holds | `msa_sidecar.pid`, `MSA_BUSY.json` |
| Coordinator (sidecar adapter) | Exactly-once claim of cycle result | `msa_fusion_protocol/claimed/`, `journal/`, `complete/` |

## What landed

1. **Production adapter** (`fusion_adapter.py`): sidecar cycle summary → per-worker commands; no fake injector.
2. **Command/ack protocol** (`fusion_protocol.py`): POSIX link claim, per-worker cmds, acks, complete marker.
3. **Transactional PROBE anchors** (`fusion_transaction.py`): verified fingerprint copy; ADVANCE committed snapshot; ROLLBACK restore + exit-42 restart.
4. **Observer wiring**: busy→PROBE+anchor; claim/apply cmds; PBT donor/recipient gates on real source/target paths.
5. **Eval fixture** (`eval_fixture.py` + sidecar `--fusion-fixture-once`): test-only; still uses `decide_selective_promotion` + adapter; formal profile cannot enable (`MSA_FUSION_ALLOW_EVAL_FIXTURE`).
6. **Integrated smoke** (`fusion_smoke --mode integrated`).

## Tolerance write points (fusion profile)

| Writer | Fusion profile permission |
|--------|---------------------------|
| `apply_env_tolerance_curriculum` mastery/adaptive/late-floor | **Proposal only** (`msa_tolerance_proposal`); does not change `success_tolerance` |
| MSA sidecar `pin_tolerance` / `commit_tolerance` | **Allowed** (production durable commit) |
| Fusion ADVANCE (`apply_advance` + committed anchor) | **Allowed** (only after verified production command) |
| Fusion ROLLBACK (restore anchor) | **Allowed** (verified fingerprint only) |
| PBT exploit copy | **Must not** inherit donor committed stage / pass evidence; gates reject PROBE; no direct stricter tol commit |
| Checkpoint resume | Restores **registered** curriculum/tolerance from checkpoint meta only |
| Fake decision injector | **Forbidden** unless test env `MSA_FUSION_ALLOW_FAKE_DECISION=1` |

Old profiles (`msa_integrated_enabled=False`) keep prior shrink / PBT behavior.

## Tests

```bash
PY=/root/autodl-tmp/wmdexpbt/runtime_root/envs/isaaclab311/bin/python
cd /root/autodl-tmp/wmdexpbt/runtime_root/src/dexpbt_isaaclab
unset MSA_FUSION_ALLOW_FAKE_DECISION MSA_FUSION_ALLOW_EVAL_FIXTURE
$PY -m pytest tests/test_msa_fusion_production.py tests/test_msa_fusion.py -q
```

## Integrated smoke

```bash
bash /root/autodl-tmp/wmdexpbt/results/launchers/msa_probe/run_msa_fusion_integrated_smoke.sh --i-know-what-i-am-doing
```

Requires 6 free GPUs. Uses tip recover + 512 envs. Fixture evidence only for the decision payload; trainers still apply ADVANCE/ROLLBACK.
