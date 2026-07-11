# Unified Elite-WM DexPBT

A clean experimental method variant on the current DexPBT codebase. This branch
does **not** replace or roll back v13/v17/v17.4 strategy-zoo runs; it adds a
new profile, launcher, and documentation while preserving all existing profiles
and launchers for reproducibility.

## Motivation

Prior strategy-zoo experiments centered worker diversity on **discrete profile
identities** and **profile copy** (swapping a worker's entire WM/EIR mechanism
stack). That story made bottom-slot recovery fragile: health gates could block
the only effective recovery path, while ungated copy caused profile churn and
death spirals.

**Unified Elite-WM DexPBT** reframes the experiment:

- Treat **EliteBank behavior reuse** and **WM PBT source selector** as a
  **shared training substrate** for every worker.
- Let PBT propagate **weights**, **checkpoints**, and **curriculum state** —
  not discrete profile identities.
- Worker diversity comes from seeds, PBT history, curriculum state, and
  stochastic optimization rather than mutually exclusive mechanism stacks.

## Method definition

```
Unified Elite-WM DexPBT
  = PPO/PBT
  + EliteBank behavior reuse (actor distillation from elite/archive)
  + WM PBT source selector (v13 wm_pbt_source_selector stack; active floor-recovery ranking)
  + curriculum safety (mastery, collapse detection, rollback-to-last-mastered)
```

Profile name: `unified_elite_wm_pbt` (id 18).

Elite base: conservative `elite_bank_actor_distill` settings (distill @300M,
sources `elite,archive`, soft tolerance gate, no MPPI).

WM mode: `pbt_wm_score_mode=active` with `pbt_use_wm_recovery_score=True` —
the v13 WM selector stack is **live** on every worker. When a policy enters
`hard_floor`, imagined recovery / drop / success signals can replace the legacy
ordering for source selection. Outside hard floor, PBT ranking stays on
`success_aware_r_meta`.

WM activation gate: `confidence_only` with 300M start + 300M ramp
(`wm_pbt_source_start_after_frames`, `wm_pbt_source_ramp_frames`).

## Hard exclusions

| Mechanism | Status |
|-----------|--------|
| Profile copy (`pbt_wm_profile_copy_enable`) | **Off** (launcher `--no-pbt_wm_profile_copy_enable`) |
| AWR (`awr_enable`) | **Off** (profile + launcher `--no-awr_enable`) |
| EIR residual rollout apply | **Off** (profile + launcher `--no-eir_residual_apply_to_rollout`) |
| RSSM-MPPI action intervention | **Off** (`eir_use_rssm_mppi=False`, `eir_mppi_mode=off`) |

All AWR, profile-copy, and strategy-zoo code paths remain in the codebase for
old launchers.

## Diversity sources

With every worker on the same profile:

- Random seeds and initialization
- Independent PBT restart / cross-copy history
- Per-worker curriculum tolerance and mastery state
- Stochastic rollouts and optimization noise

There are no mutually exclusive worker roles (no dedicated `ppo_only` anchor vs
elite-bank vs WM-selector slots).

## WM selector behavior (active, not shadow)

Unlike the initial unified smoke draft, the profile now merges the **full v13
`wm_pbt_source_selector` PBT stack** on top of EliteBank:

- Task-focused WM training (horizon heads, recovery head, iterative teacher)
- Imagined evaluator with soft-calibrated validity gates
- Active floor-recovery-aware PBT ranking when `hard_floor` + validity pass
- Confidence-only source activation ramp after 300M frames

Supported `pbt_wm_score_mode` values remain `off | shadow | active`. Unified
uses **`active`**.

## 1.2B smoke launcher

Launcher:

```bash
bash results/launchers/start_regrasping_unified_elite_wm_pbt_1p2b_smoke.sh
```

Configuration:

- 6 policies, all `unified_elite_wm_pbt`
- Target: **1.2B frames** (`MAX_ITER=9156`)
- Same task and core PPO/PBT/curriculum scaffold as stable v13/v17 launchers
- Pointer: `results/pointers/latest_unified_elite_wm_pbt_1p2b_smoke_run.txt`

## 2B debug launcher

```bash
bash results/launchers/start_regrasping_unified_elite_wm_pbt_2b_debug.sh
```

Configuration:

- 6 policies, all `unified_elite_wm_pbt`
- Target: **2B frames** (`MAX_ITER=15259`)
- PBT base metric: `success_aware_r_meta`; WM selector active in floor-recovery zone
- Profile copy, AWR, residual rollout apply, and RSSM-MPPI action intervention
  are all off
- Pointer: `results/pointers/latest_unified_elite_wm_pbt_2b_debug_run.txt`

The 1.2B smoke launcher remains available for quick validation; the primary
unified experiment path is the 2B debug run once tests pass.

**Note:** Runs started before the active-WM profile update were shadow-only.
Restart from a fresh launcher invocation to pick up `pbt_wm_score_mode=active`.

## Metrics to monitor

| Metric | Why |
|--------|-----|
| `nsucc_mean` | Primary per-policy success signal |
| `nsucc_max` | Peak episode success within window |
| Top-3 mean | Population health vs v13 strong-4 |
| Bottom-slot recovery | Whether weak slots recover without profile swap |
| Curriculum shrink / rollback events | Mastery progression and safety-net activity |
| `wm_source_activation_weight` / `source_selected_by_wm` | WM selector live participation |
| `wm_imagined_valid` / uncertainty | Imagined eval quality |
| FPS | Throughput with unified WM+EliteBank stack on all workers |
| PBT restart / cross-copy events | Weight propagation without profile churn |

## Related files

- Profile: `runtime_root/src/dexpbt_isaaclab/dexpbt_isaaclab/world_model/wm_strategy_profiles.py`
- Launchers:
  - `results/launchers/start_regrasping_unified_elite_wm_pbt_1p2b_smoke.sh`
  - `results/launchers/start_regrasping_unified_elite_wm_pbt_2b_debug.sh`
- Tests: `runtime_root/src/dexpbt_isaaclab/tests/test_unified_elite_wm_pbt.py`
