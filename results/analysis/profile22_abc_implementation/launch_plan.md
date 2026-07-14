# Profile22 ABC — Launch Plan (DO NOT EXECUTE YET)

## Status

Launcher created only. **Not executed.** Await human review of
`results/analysis/profile22_abc_implementation/REPORT.md`.

## Launcher path

`results/launchers/start_regrasping_cser_dexpbt_progress_refine_abc_5b.sh`

## Intended identity

| Field | Value |
|---|---|
| Profile | `cser_dexpbt_progress_refine_abc` (id **24**) |
| Baseline inherit | Profile 22 `cser_dexpbt_progress_refine` |
| Workers | 6 |
| Envs/worker | 8192 |
| Seed | 42 |
| Budget | 5e9 population frames |
| GPUs | 0–5 |
| Mid mutation | OFF |
| MPPI / RSSM-MPPI | OFF |
| AWR / residual apply | OFF |
| ABC flags | all ON (profile + explicit CLI) |

## Explicit ABC flags in launcher argv

```
--pbt_enable_chronic_bottom_exploit
--eir_tolerance_aware_bank_broadcast
--eir_preserve_local_bank_fraction 0.20
--eir_enable_belief_for_snippet_wm_rerank
```

## Suggested start command (for humans after review)

```bash
cd /root/autodl-tmp/wmdexpbt
COMMIT_SHA=$(git rev-parse HEAD)
export COMMIT_SHA
# AFTER review only:
# bash results/launchers/start_regrasping_cser_dexpbt_progress_refine_abc_5b.sh
```

Run directory pattern:

`results/runs/regrasping_cser_dexpbt_progress_refine_abc_5b_<timestamp>/`

will contain `RUN_CONFIG.txt`, `LAUNCH_IDENTITY.txt`, `launch_master.log`.

## Do not

- Start while baseline `092846` or any other run shares GPUs unless intentional.
- Hot-patch a live Profile22 process.
- Reuse the Profile22 run directory.
- Commit/push/PR from this round unless asked.
