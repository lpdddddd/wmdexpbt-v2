# CSER-DexPBT (CSER-v0)

CSER-v0 upgrades the **Unified Elite-WM DexPBT** (basev2) substrate with three
additive modules and a staged low-churn PBT schedule. It does not replace
existing profiles or launchers.

## Method definition

```
CSER-DexPBT
  = Unified Elite-WM DexPBT (basev2)
  + EliteSnippetBank + first-step-only snippet distillation
  + WM calibration evaluator + WM-guided snippet rerank
  + staged PBT consolidation (exploration / consolidation / selection)
```

Profile name: `cser_dexpbt` (id 19).

Base substrate: `unified_elite_wm_pbt` — EliteBank actor distillation + active
WM PBT source selector (v13 stack) on every worker.

## CSER-v0 constraints

| Constraint | Enforcement |
|------------|-------------|
| Snippet distillation is **first-step only** | `snippet_distill` compares `mu(feat_start)` vs `teacher[:,0,:]`; H-step labels drive priority/WM rerank only |
| WM calibration **multiplies** activation weight | `effective_wm_weight = activation_weight * alpha_wm`; when calibration off, behavior is basev2-identical |
| Real nsucc remains PBT anchor | `blend_wm_ranking_score` keeps real score primary; WM term is bounded |
| No rollout action intervention | `eir_residual_apply_to_rollout=False`, `eir_use_rssm_mppi=False`, `eir_mppi_mode=off` |
| No profile copy | `pbt_wm_profile_copy_enable=False` + launcher `--no-pbt_wm_profile_copy_enable` |
| No AWR | `awr_enable=False` + launcher `--no-awr_enable` |
| No MPPI rollout intervention | profile hard-off |

All launchers keep these three flags explicit:

```bash
--no-pbt_wm_profile_copy_enable
--no-awr_enable
--no-eir_residual_apply_to_rollout
```

## PBT consolidation stages (5B formal)

| Stage | Frame range | Behavior |
|-------|-------------|----------|
| Exploration | 0 – 2.2B | Default PBT churn |
| Consolidation | 2.2B – 3.5B | Wider tick interval, lower mutation rate, retention-gated bottom restart |
| Selection | 3.5B – 5.0B | Widest interval, bottom-only + hard-collapse-only replacement |

Kill switch: `--no-pbt_consolidation_enable` restores basev2 PBT behavior.

## Launchers

| Launcher | Target | Purpose |
|----------|--------|---------|
| `start_regrasping_cser_dexpbt_1p2b_smoke.sh` | 1.2B (`MAX_ITER=9156`) | Wiring smoke: distill loss, TB scalars, no crash |
| `start_regrasping_cser_dexpbt_3b_ablation.sh` | 3B (`MAX_ITER=22888`) | Ablation matrix via `CSER_ABLATION` env var |
| `start_regrasping_cser_dexpbt_5b.sh` | 5B (`MAX_ITER=38147`) | Formal run with explicit consolidation flags |

Pointers:

- `results/pointers/latest_cser_dexpbt_1p2b_smoke_run.txt`
- `results/pointers/latest_cser_dexpbt_3b_ablation_<variant>_run.txt`
- `results/pointers/latest_cser_dexpbt_5b_run.txt`

## 3B ablation matrix

Set `CSER_ABLATION` before launch:

```bash
CSER_ABLATION=full bash results/launchers/start_regrasping_cser_dexpbt_3b_ablation.sh
```

| `CSER_ABLATION` | Profile | Module under test |
|-----------------|---------|-------------------|
| `basev2` | `unified_elite_wm_pbt` | Control (no CSER additions) |
| `low_churn` | `unified_elite_wm_pbt` | Staged PBT consolidation only |
| `snippet_only` | `unified_elite_wm_pbt` | Snippet bank + distill (no WM rerank/calibration) |
| `snippet_wm` | `unified_elite_wm_pbt` | Snippet + WM calibration rerank |
| `full` | `cser_dexpbt` | All CSER-v0 modules |

Use 3B ablation (not 1.2B smoke) to evaluate consolidation stage-2 behavior
and `pbt_consolidation/restart_blocked_count` trends.

## Key metrics

| Metric | Why |
|--------|-----|
| `snippet_distill/loss` | Snippet distillation active |
| `snippet_wm_rerank/changed_frac` | WM rerank impact on teacher selection |
| `wm_calibration/alpha_wm` | Calibration trust weight |
| `pbt_consolidation/stage` | Consolidation stage transitions |
| `pbt_consolidation/restart_blocked_count` | Low-churn gate activity |
| `nsucc_mean`, bottom-slot recovery | Population health vs basev2 |

## Related files

- Profile: `runtime_root/src/dexpbt_isaaclab/dexpbt_isaaclab/world_model/wm_strategy_profiles.py`
- Consolidation: `runtime_root/src/dexpbt_isaaclab/dexpbt_isaaclab/pbt/pbt_consolidation_schedule.py`
- Snippet bank: `runtime_root/src/dexpbt_isaaclab/dexpbt_isaaclab/world_model/elite_snippet_bank.py`
- WM calibration: `runtime_root/src/dexpbt_isaaclab/dexpbt_isaaclab/world_model/wm_calibration_evaluator.py`
- Base method doc: `results/docs/UNIFIED_ELITE_WM_DEXPBT.md`
