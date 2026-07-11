# TB EIR/WM Signal Diagnosis

## 0. Run metadata

- **Run path:** `/root/autodl-tmp/wmdexpbt/results/runs/regrasping_pbt_wm_strategy_zoo_full_3b_20260615_180739`
- **Commit SHA:** `780092276dee`
- **TensorBoard available:** True
- **Event files found:** 17

### Event files read (by policy)

- **p0** (3 files):
  - `/root/autodl-tmp/wmdexpbt/results/runs/regrasping_pbt_wm_strategy_zoo_full_3b_20260615_180739/policy_00/rl_games/attempt_1_2026-06-15_18-07-39/summaries/events.out.tfevents.1781518080.autodl-container-l15rjnak8t-75a31c36`
  - `/root/autodl-tmp/wmdexpbt/results/runs/regrasping_pbt_wm_strategy_zoo_full_3b_20260615_180739/policy_00/rl_games/attempt_2_2026-06-15_19-23-51/summaries/events.out.tfevents.1781522654.autodl-container-l15rjnak8t-75a31c36`
  - `/root/autodl-tmp/wmdexpbt/results/runs/regrasping_pbt_wm_strategy_zoo_full_3b_20260615_180739/policy_00/rl_games/attempt_3_2026-06-15_19-44-24/summaries/events.out.tfevents.1781523888.autodl-container-l15rjnak8t-75a31c36`
- **p1** (2 files):
  - `/root/autodl-tmp/wmdexpbt/results/runs/regrasping_pbt_wm_strategy_zoo_full_3b_20260615_180739/policy_01/rl_games/attempt_1_2026-06-15_18-09-09/summaries/events.out.tfevents.1781518172.autodl-container-l15rjnak8t-75a31c36`
  - `/root/autodl-tmp/wmdexpbt/results/runs/regrasping_pbt_wm_strategy_zoo_full_3b_20260615_180739/policy_01/rl_games/attempt_2_2026-06-15_19-58-21/summaries/events.out.tfevents.1781524726.autodl-container-l15rjnak8t-75a31c36`
- **p2** (5 files):
  - `/root/autodl-tmp/wmdexpbt/results/runs/regrasping_pbt_wm_strategy_zoo_full_3b_20260615_180739/policy_02/rl_games/attempt_1_2026-06-15_18-10-39/summaries/events.out.tfevents.1781518263.autodl-container-l15rjnak8t-75a31c36`
  - `/root/autodl-tmp/wmdexpbt/results/runs/regrasping_pbt_wm_strategy_zoo_full_3b_20260615_180739/policy_02/rl_games/attempt_2_2026-06-15_19-13-14/summaries/events.out.tfevents.1781522018.autodl-container-l15rjnak8t-75a31c36`
  - `/root/autodl-tmp/wmdexpbt/results/runs/regrasping_pbt_wm_strategy_zoo_full_3b_20260615_180739/policy_02/rl_games/attempt_3_2026-06-15_19-36-42/summaries/events.out.tfevents.1781523425.autodl-container-l15rjnak8t-75a31c36`
  - `/root/autodl-tmp/wmdexpbt/results/runs/regrasping_pbt_wm_strategy_zoo_full_3b_20260615_180739/policy_02/rl_games/attempt_4_2026-06-16_00-38-45/summaries/events.out.tfevents.1781541548.autodl-container-l15rjnak8t-75a31c36`
  - `/root/autodl-tmp/wmdexpbt/results/runs/regrasping_pbt_wm_strategy_zoo_full_3b_20260615_180739/policy_02/rl_games/attempt_5_2026-06-16_00-58-54/summaries/events.out.tfevents.1781542756.autodl-container-l15rjnak8t-75a31c36`
- **p3** (3 files):
  - `/root/autodl-tmp/wmdexpbt/results/runs/regrasping_pbt_wm_strategy_zoo_full_3b_20260615_180739/policy_03/rl_games/attempt_1_2026-06-15_18-12-09/summaries/events.out.tfevents.1781518353.autodl-container-l15rjnak8t-75a31c36`
  - `/root/autodl-tmp/wmdexpbt/results/runs/regrasping_pbt_wm_strategy_zoo_full_3b_20260615_180739/policy_03/rl_games/attempt_2_2026-06-15_19-09-30/summaries/events.out.tfevents.1781521792.autodl-container-l15rjnak8t-75a31c36`
  - `/root/autodl-tmp/wmdexpbt/results/runs/regrasping_pbt_wm_strategy_zoo_full_3b_20260615_180739/policy_03/rl_games/attempt_3_2026-06-15_19-25-15/summaries/events.out.tfevents.1781522738.autodl-container-l15rjnak8t-75a31c36`
- **p4** (2 files):
  - `/root/autodl-tmp/wmdexpbt/results/runs/regrasping_pbt_wm_strategy_zoo_full_3b_20260615_180739/policy_04/rl_games/attempt_1_2026-06-15_18-13-40/summaries/events.out.tfevents.1781518444.autodl-container-l15rjnak8t-75a31c36`
  - `/root/autodl-tmp/wmdexpbt/results/runs/regrasping_pbt_wm_strategy_zoo_full_3b_20260615_180739/policy_04/rl_games/attempt_2_2026-06-15_18-58-12/summaries/events.out.tfevents.1781521115.autodl-container-l15rjnak8t-75a31c36`
- **p5** (2 files):
  - `/root/autodl-tmp/wmdexpbt/results/runs/regrasping_pbt_wm_strategy_zoo_full_3b_20260615_180739/policy_05/rl_games/attempt_1_2026-06-15_18-15-10/summaries/events.out.tfevents.1781518533.autodl-container-l15rjnak8t-75a31c36`
  - `/root/autodl-tmp/wmdexpbt/results/runs/regrasping_pbt_wm_strategy_zoo_full_3b_20260615_180739/policy_05/rl_games/attempt_2_2026-06-15_18-58-39/summaries/events.out.tfevents.1781521142.autodl-container-l15rjnak8t-75a31c36`

### Policy / profile mapping

| Policy | Profile | Frames |
|--------|---------|--------|
| p0 | ppo_only | 2.98B |
| p1 | rssm_mppi_actor_distill | 1.04B |
| p2 | wm_curriculum_controller | 3.04B |
| p3 | wm_pbt_source_selector | 3.04B |
| p4 | elite_bank_actor_distill | 2.86B |
| p5 | conservative_rssm_planner_distill | 1.22B |

## 1. Executive summary

- WM head loss scale: **mixed** (TB `head_loss_weighted_fraction`).
- p1 EIR chain: **case4_active**.
- p4 EIR chain: **case1_pool_no_selection**.
- p5 EIR chain: **case2_selected_no_distill**.
- p3 WM source selector: **sustained_active** (64 ranking ticks, eff_w mean=0.0738).
- p2 curriculum driver: **mixed_real_and_wm_high_restarts**.
- p3 legacy `high_uncertainty` fallback count in JSONL: **0** (post-hotfix should be 0).
- Safety patch evidence: p2 real_signal_block events = 11.
- TensorBoard event files scanned: **17** across run_dir (+ outputs fallback).

## 2. WM head loss scale

| Policy | Profile | Frames | WHLF mean | last | max | healthy_frac | too_low_frac | too_high_frac | Verdict |
|--------|---------|--------|-----------|------|-----|--------------|--------------|---------------|---------|
| p0 | ppo_only | 2.98B | — | — | — | — | — | — | no_data |
| p1 | rssm_mppi_actor_distill | 1.04B | 0.1774 | 0.2537 | 0.4325 | 0.9248 | 0.0405 | 0.0003 | healthy |
| p2 | wm_curriculum_controller | 3.04B | 0.0079 | 0.0000 | 0.3568 | 0.0636 | 0.9207 | 0.0000 | too_weak |
| p3 | wm_pbt_source_selector | 3.04B | 0.0058 | 0.0000 | 0.3497 | 0.0469 | 0.9315 | 0.0000 | too_weak |
| p4 | elite_bank_actor_distill | 2.86B | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.0000 | too_weak |
| p5 | conservative_rssm_planner_distill | 1.22B | 0.1665 | 0.2191 | 0.7034 | 0.8903 | 0.0549 | 0.0001 | healthy |

## 3. WM imagined / source selector signal

| Policy | strict_valid | soft_valid | wm_score_used | strict_used | soft_used | eff_w mean | eff_w max | fallback top | Verdict |
|--------|--------------|------------|---------------|-------------|-----------|------------|-----------|--------------|---------|
| p0 | 0 | 0 | 0 | 0 | 0 | 0.0000 | 0.0000 | feature_off:138 | inactive |
| p1 | 0 | 0 | 0 | 0 | 0 | 0.0000 | 0.0000 | feature_off:42 | inactive |
| p2 | 139 | 0 | 0 | 0 | 0 | 0.0000 | 0.0000 | feature_off:139 | inactive |
| p3 | 141 | 4 | 64 | 64 | 0 | 0.0738 | 0.2130 | feature_off:76, ok:64, no_promoted_teacher:1 | sustained_active |
| p4 | 0 | 0 | 0 | 0 | 0 | 0.0000 | 0.0000 | feature_off:133 | inactive |
| p5 | 0 | 0 | 0 | 0 | 0 | 0.0000 | 0.0000 | feature_off:51 | inactive |

**p3 detail:** WM ranking uses strict path when `wm_validity_weight=1.0`; `high_uncertainty` fallback should be absent post-hotfix. Check `gate_mode_counts` and `fallback_reason_counts` in JSON.


## 4. EIR teacher supply chain

| Policy | Profile | cand_avail_frac | cand_count_mean | teacher_sel_frac | distill_nonzero mean | distill_w mean | Verdict |
|--------|---------|-----------------|-----------------|------------------|----------------------|----------------|---------|
| p0 | ppo_only | — | — | — | — | — | case3_no_candidates |
| p1 | rssm_mppi_actor_distill | 0.5518 | — | 0.1689 | 0.1601 | 0.0035 | case4_active |
| p2 | wm_curriculum_controller | — | — | — | — | — | case3_no_candidates |
| p3 | wm_pbt_source_selector | — | — | — | — | — | case3_no_candidates |
| p4 | elite_bank_actor_distill | 0.4135 | — | 0.0000 | 0.3084 | 0.3994 | case1_pool_no_selection |
| p5 | conservative_rssm_planner_distill | 0.5952 | — | 0.0213 | 0.0039 | 0.0012 | case2_selected_no_distill |

## 5. p2 curriculum controller diagnosis

- **Interpretation:** mixed_real_and_wm_high_restarts
- **cross_restart_count:** 4
- **block_reason_counts:** `{'wm_risk_valid': 5, 'real_signal_risk': 4, 'curriculum_safety_freeze_active': 16}`
- **real_risk ticks:** 25
- **wm_risk ticks:** 22
- **danger_zone ticks:** 29
- **wm_curriculum_blocked_shrink ticks:** 126
- **rollback_count:** 0
- **global_retention mean/min:** 0.9358 / 0.3304
- **recent_slope mean:** 0.0633
- **curriculum_safety_events (p2):** `{'total_events': 11, 'event_counts': {'real_signal_block': 11}}`

## 6. Actionable conclusions

- Head loss scale is policy-dependent; review per-policy TB before changing global lambdas.
- p1 rssm_mppi: EIR teacher supply + actor_distill path active (TB confirmed).
- p4 elite_bank: candidates enter pool but reranker rarely selects — review teacher margin / reranker.
- p5 conservative: teachers selected but actor_distill signal near zero — review distill mask/loss gate.
- p3 WM source selector is sustained-active; no ranking-weight hotfix needed.
- p2 curriculum blocks are mixed real+WM; tune controller only after separating block_reason_counts.

## 7. Missing data / caveats

**Missing TB tags / series:**
- `p0:weighted_head_loss_fraction`

*Note: EIR scalars log only on actor-distill profiles (p1/p4/p5). p0/p2/p3 may show `case3` or empty EIR TB by design.*
