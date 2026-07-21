# Fusion Production Smoke Report

**Run:** `results/runs/msa_fusion_smoke_integrated_20260722_012244`  
**Result:** `FUSION_SMOKE_PASSED.json` (`ok: true`)  
**Date:** 2026-07-22

## Verified

| # | Check | Result |
|---|--------|--------|
| 1–3 | 6× Isaac after_init + real env/PPO steps | yes (all GPUs ~4.6GB, fps step counts 1k+) |
| 4 | Launcher started sidecar / guard / watchdog | yes (`MSA_SIDECAR_STARTED`, hold_watchdog, keepalive guard) |
| 5–7 | Busy → PROBE + verified anchor checkpoint | yes (`MSA_PROBE_CHECKPOINT_SAVED` ×2) |
| 8–10 | Production sidecar fixture → real decide → cmds | yes (`--fusion-fixture-once`) |
| 11–14 | ADVANCE in trainer 0; ROLLBACK+restore in trainer 1 | yes |
| 15 | Per-worker command/ack (no shared delete race) | yes |
| 16–17 | Resume state on disk after teardown | yes |
| 18–19 | Clean exit; no orphan GPU procs | yes |
| 20 | Worker timelines in `msa_fusion_events.jsonl` | yes |

## Notes

- Formal profiles cannot enable `MSA_FUSION_ALLOW_EVAL_FIXTURE`.
- Tip resume required `--max-iterations 100000000` (5B-frame tips exit at 48780).
- Inter-launch delay 90s avoided OV cache symlink races.
- Orchestrator step-log path bug (`policy_0` vs `policy_00`) fixed mid-run; cycle completed via continued orchestrator on the live trainers.
