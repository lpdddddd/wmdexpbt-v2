# MSA-aware Fusion — Test Report

Date: 2026-07-21 / 2026-07-22

## Unit + integration

```bash
PY=/root/autodl-tmp/wmdexpbt/runtime_root/envs/isaaclab311/bin/python
cd /root/autodl-tmp/wmdexpbt/runtime_root/src/dexpbt_isaaclab
$PY -m pytest tests/test_msa_probe.py tests/test_msa_selective_budget.py \
  tests/test_state_bootstrap.py tests/test_msa_fusion.py -q
```

**Result: 237 passed**

## Control smoke

```bash
MSA_FUSION_ALLOW_FAKE_DECISION=1 $PY -m dexpbt_isaaclab.msa.fusion_smoke --mode control
```

Worker 0 ADVANCE, worker 1 HOLD, workers 2–5 safe-stop HOLD.

## Disposable 6-GPU smoke

```bash
bash results/launchers/msa_probe/run_msa_fusion_smoke.sh gpu --i-know-what-i-am-doing
```

Run: `results/runs/msa_fusion_smoke_gpu_20260722_000145`

- `ok=true`, `FUSION_SMOKE_PASSED.json` written
- `profile_expanded=true` (fusion knobs applied via profile authoritative expansion)
- 6 trainers launched on CUDA_VISIBLE_DEVICES 0–5
- Transitions: w0 ADVANCE→BUILD; w1 HOLD; w2–5 safe-stop HOLD
- `orphan_pids_after_cleanup=[]`
- Synthetic decisions only (`test_only=true`)

Note: Omniverse XDG cache corruption can delay Isaac `after_init`; smoke proceeds
once fusion profile expansion is observed and all 6 policy dirs exist, then tears
down trainers. Full sim step is not required for control-flow certification.
