# Profile24 ABC 5B Launch Report

Date: 2026-07-14  
Active run commit: `c890ccda63193c2b49da4549dbe451b55b39dbb2`  
Cache commit (parent): `5779d0f5e9965676d5ce4a53a7022fa22c40d53a`

## Commits in this session

| SHA | Message |
|---|---|
| `5779d0f` | Optimize ABC snapshot validation caching |
| `c890ccd` | Wire Profile24 into train.py wm_strategy_profile choices |

## First launch attempt (failed — kept for logs)

- Run: `regrasping_cser_dexpbt_progress_refine_abc_5b_20260714_221352`
- Commit recorded: `5779d0f…`
- Failure: `train.py: error: argument --wm_strategy_profile: invalid choice: 'cser_dexpbt_progress_refine_abc'`
- Cause: hardcoded argparse choices omitted Profile24 even though `PROFILE_IDS` registered it
- Action: **stopped only this ABC run**; fixed CLI choices to `sorted(PROFILE_IDS.keys())`; did **not** change algo thresholds

## Successful relaunch

- Launch time: 2026-07-14 22:23:00 (+08)
- Run name: `regrasping_cser_dexpbt_progress_refine_abc_5b_20260714_222300`
- Run dir: `/root/autodl-tmp/wmdexpbt/results/runs/regrasping_cser_dexpbt_progress_refine_abc_5b_20260714_222300`
- Commit: `c890ccda63193c2b49da4549dbe451b55b39dbb2`
- Profile: `cser_dexpbt_progress_refine_abc` (id 24)
- Master PID: `859798` (`launch_pbt`)

### Worker PIDs / GPU map

| Policy | Train PID | CUDA_VISIBLE_DEVICES | Physical GPU |
|---|---|---|---|
| 0 | 859911 | 0 | 0 |
| 1 | 860928 | 1 | 1 |
| 2 | 862237 | 2 | 2 |
| 3 | 863342 | 3 | 3 |
| 4 | 864297 | 4 | 4 |
| 5 | 865599 | 5 | 5 |

(`--device=cuda:0` is intentional under per-worker `CUDA_VISIBLE_DEVICES`.)

### Resolved ABC / safety flags

From launcher argv + `RUN_CONFIG.txt` + live belief telemetry:

| Flag | Value |
|---|---|
| `pbt_enable_chronic_bottom_exploit` | True (CLI) |
| `eir_tolerance_aware_bank_broadcast` | True (CLI) |
| `eir_enable_belief_for_snippet_wm_rerank` | True (CLI; belief events firing) |
| `pbt_allow_mid_self_mutation` | False (`--no-…`) |
| `eir_mppi_mode` | off |
| `eir_use_rssm_mppi` | False |
| `awr_enable` | False |
| `eir_residual_apply_to_rollout` | False |

Profile expansion logs `[wm_strategy_profile=cser_dexpbt_progress_refine_abc] expansion (profile_authoritative=true)`.

### Early process status (~18 min after launch)

All 6 workers alive; logs growing; no OOM / NaN / argparse invalid-choice / supervisor restart storm.

Sample throughput (wall fps total / epoch / frames):

| Policy | ~fps total | ~epoch | ~frames |
|---|---|---|---|
| 0 | ~45k | 400 | 52M |
| 1 | ~43k | 333 | 43M |
| 2 | ~71k | 317 | 41M |
| 3 | ~56k | 217 | 28M |
| 4 | ~70k | 149 | 19M |
| 5 | ~65k | 62 | 8M |

GPU memory ~9.3–9.5 GiB / card; util ~7–21%.

### Belief / snapshot cache initial state

- `dexpbt_8gpu_ws/metrics/eir_belief_events.jsonl` present
- Sample: `need_belief=true`, `belief_build_success=true`, `belief_cache_valid=true`, latency ≈ 6 ms
- Snapshot-cache TB/JSONL appears only on chronic selection ticks (after PBT start ≥200M); too early for chronic ticks now

### Known risks

1. First launch failed on missing CLI choices — fixed in `c890ccd`.
2. Isaac logs noisy USD material / CUDA_VISIBLE_DEVICES warnings (same as Profile22 baseline).
3. Snapshot cache telemetry not observable until chronic PBT source selection begins (~≥200M frames + bottom eligibility).
4. Failed run dir `…_221352` retained for forensics; do not resume it.

## Follow-up check plan (do not wait in this session)

### 100M

- All 6 workers healthy; fps/mem stable
- Belief build still succeeding; no exception storm
- Disk/I/O for workspace checkpoints

### 300M

- Progress refinement active
- Bank local/foreign/compatibility diagnostics
- WM rerank reason / changed_frac

### 600M

- Chronic bottom accumulation
- Source health + **snapshot cache hit/miss**
- PBT source selection counts
- Bank ESS / active frac

### 1B

- best / median / mean vs Profile22 same-frame
- Weakest worker + spread
- PBT copy counts (chronic vs ordinary)

**Next check gate: ~100M frames** (leader currently ~50M).
