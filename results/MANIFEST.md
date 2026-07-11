# Results Directory Manifest

Reorganized 2026-06-10 (Plan B cleanup).

## Layout

```
results/
├── MANIFEST.md              # this file
├── regrasping_*_wired_*/    # ACTIVE run only (stays at root while training)
├── runs/                    # completed / reference experiment directories
├── launchers/               # start_*.sh, restart_*.sh
├── tools/                   # analyze_*, compare_*, cleanup_run.sh, monitor_training.sh
├── docs/                    # roadmaps, protocol, diffs, AB reports
│   └── archive/             # extracted reports from deleted legacy runs
└── pointers/                # latest_*.txt → absolute run paths
```

## Active Run

| Run | Path | Notes |
|-----|------|-------|
| wired 3B iterative | `regrasping_fixed_rssm_iterative_full_active_replay_3b_wired_20260610_020914/` | TeacherManager + WMImaginedEvaluator wired; do not move while training |

Pointer: `pointers/latest_rssm_iterative_full_active_replay_3b_wired_run.txt`

## Reference Runs (in `runs/`)

| Run | Pointer | Purpose |
|-----|---------|---------|
| `regrasping_fixed_rssm_iterative_full_active_replay_3b_20260609_213859` | `latest_rssm_iterative_full_active_replay_3b_run.txt` | Unwired control; STOP report inside |
| `regrasping_fixed_rssm_iterative_shadow_800m_20260609_165636` | `latest_rssm_iterative_shadow_800m_run.txt` | 800M shadow diagnostic |
| `regrasping_fixed_rssm_belief_value_fastcurr_800m_20260609_003432` | `latest_rssm_fastcurr_run.txt` | adaptive_fast baseline |
| `regrasping_fixed_rssm_belief_value_fastcurr_inherit_3b_20260609_045840` | `latest_rssm_fastcurr_inherit_3b_run.txt` | source_safe inherit 3B (stopped) |
| `regrasping_fixed_rssm_belief_value_critic_distill_v2_smoke_800m_20260608_190922` | `latest_seq_wm_v2_smoke_run.txt` | seq-WM coupling v2 smoke |
| `regrasping_fixed_rssm_belief_value_distill_value_200m_20260608_165347` | `latest_seq_wm_ab_run.txt` | 200M AB distill_value |
| `regrasping_fixed_trans_belief_value_p2_1_smoke_800m_20260608_100048` | `latest_run.txt`, `latest_smoke_run.txt` | trans-belief smoke |
| Other 50m/200m smokes | — | early RSSM smoke tests |

## Deleted Legacy Runs (reports archived)

Full run directories removed; `.md` / config / stop markers kept under `docs/archive/`:

| Archived | Size freed | Archive path |
|----------|-----------|--------------|
| `regrasping_bayesian_no_h3_p2_1_3b_20260607_024931` | 1.2G | `docs/archive/regrasping_bayesian_no_h3_p2_1_3b_20260607_024931/` |
| `regrasping_fixed_h1_p2_1_formal_5b_20260605_224754` | 1.3G | `docs/archive/regrasping_fixed_h1_p2_1_formal_5b_20260605_224754/` |
| `regrasping_cbwm_smoke_800m_p2_1_no_h3_20260605_040330` | 171M | `docs/archive/...040330/` |
| `regrasping_cbwm_smoke_800m_p2_1_no_h3_20260605_134136` | 33M | `docs/archive/...134136/` |
| `regrasping_cbwm_smoke_800m_p2_1_no_h3_20260605_135426` | 400M | `docs/archive/...135426/` |

Pointers `latest_formal_run.txt` and `latest_track_c_run.txt` now point to archive dirs.

## Common Commands

```bash
# Read latest wired run path
cat results/pointers/latest_rssm_iterative_full_active_replay_3b_wired_run.txt

# Launch new experiment (outputs go to results/runs/)
bash results/launchers/start_regrasping_fixed_rssm_iterative_full_active_replay_3b_wired.sh

# Wiring diagnostic
python results/tools/analyze_wm_runtime_wiring.py --run-dir "$(cat results/pointers/latest_rssm_iterative_full_active_replay_3b_wired_run.txt)"

# Compact a finished run (keep top archives + final ckpt)
bash results/tools/cleanup_run.sh results/runs/<RUN_DIR> 3
```

## Disk

After Plan B: ~3G freed from legacy deletion; total `results/` ~2.6G (mostly reference runs + active).
