# Sequence WM Policy Coupling — 200M Sanity A/B Report

**Date:** 2026-06-08  
**Arms:** `no_policy_coupling` (control) vs `distill_value` (experimental)  
**Target:** 200M env frames, single-policy, seed=42, fixed `rssm_belief_value`

| | Control | Distill |
|---|---|---|
| Run dir | `regrasping_fixed_rssm_belief_value_no_policy_coupling_200m_20260608_161206` | `regrasping_fixed_rssm_belief_value_distill_value_200m_20260608_165347` |
| Coupling | `none` | `distill_value`, weight=0.05, warmup=20M |
| Training time | 2323.7 s (~39 min) | 2466.2 s (~41 min) |
| Final checkpoint rew | 1250.66 | 510.43 |

---

## Acceptance Verdict: **FAIL**

| Criterion | Required | Result |
|---|---|---|
| Peak nsucc | distill ≥ control × 1.10 | 0.00 vs 0.74 → **FAIL** (ratio 0.00) |
| Time to nsucc_mean=10 | distill ≤ control × 0.80 | Neither arm reached 10 → **N/A / FAIL** |

**Do not proceed to 800M / 3B** until effect-path is re-audited and coupling hyperparameters or teacher signal quality are addressed.

---

## 1. nsucc_mean / nsucc_max

Metric source: TensorBoard `Episode/successes` (= `prev_episode_successes.mean()`, i.e. **nsucc_mean**).  
`nsucc_max` is **not logged** in single-policy runs (no PBT training_curve); only mean is available.

| Metric | Control | Distill | Δ |
|---|---|---|---|
| Peak nsucc_mean | **0.738** | **0.000** | −0.738 |
| End nsucc_mean | 0.611 | 0.000 | −0.611 |
| First frame nsucc_mean > 0 | **97.9M** | never | — |
| First frame nsucc_mean ≥ 5 | never | never | — |
| First frame nsucc_mean ≥ 10 | never | never | — |

Control shows a late breakthrough (~1050–1450 epochs ≈ 138–190M frames). Distill never achieves any env-level consecutive success.

---

## 2. Reward / r_meta

| Metric | Control | Distill |
|---|---|---|
| Peak r_meta (`Episode/true_objective_mean`) | 0.00772 | 0.000 |
| End r_meta | 0.00638 | 0.000 |
| Peak shaped reward (`Episode/dexpbt_reward_total`) | 6.13 | 4.99 |
| End shaped reward | 3.51 | 4.99 |
| Peak PPO reward (`rewards/step`) | 1766 | 527 |
| End PPO reward (`rewards/step`) | 1518 | 498 |

Distill maintains modest reach/pick reward (~5 shaped) but **never develops success-chain behavior**. Control's PPO reward peaks 3.3× higher.

---

## 3. success_tolerance Trajectory

Both arms: flat **0.075** throughout (no curriculum shrink in single-policy mode without PBT observer).

---

## 4. SeqWM Losses (end / peak / last-100 mean)

| Loss | Control end | Distill end | Control last-100 | Distill last-100 |
|---|---|---|---|---|
| loss_total | 1.014 | 0.922 | 1.010 | 0.951 |
| loss_state | 0.591 | 0.848 | 0.627 | 0.874 |
| loss_value | 1.829 | **0.140** | 1.443 | 0.152 |
| loss_kl | 0.084 | 0.061 | 0.082 | 0.067 |

RSSM trains in both arms. Distill arm's WM value head converges to much lower loss (teacher values ~1.78±1.6) while PPO policy never succeeds — suggesting **teacher value signal is not aligned with task success** at this training stage.

---

## 5. seq_policy_aux / Distillation Activity (distill arm only)

| Metric | Value |
|---|---|
| Warmup epochs (coupling inactive) | 153 (~20.1M frames) |
| Active epochs (`seq_policy_aux/active=1`) | **1375 / 1528** |
| Mean distill loss when active | 0.0345 |
| End distill loss | 0.0299 |
| PPO `losses/seq_distill_value` mean | 0.00173 |
| Teacher value mean (last 100 epochs) | 1.78 |
| Teacher value std (last 100 epochs) | 1.62 |
| Grad-flow confirmation | `[SeqWM] policy coupling=distill_value: PPO model params updated by seq aux head: yes` (epoch ~180) |

**Effect path is live** (gradients flow, cache works, no fail-fast), but coupling **actively harms** policy learning.

---

## 6. Terminal Aux / Teacher Cache

| Metric | Control | Distill |
|---|---|---|
| `seq_aux/terminal_recorded` count | 1528 / 1528 | 1528 / 1528 |
| `last_d_hat_delta_mean` | 0.0 (all epochs) | 0.0 |
| `last_success_delta_mean` | 0.0 | 0.0 |
| Teacher cache generation mismatch | **0** | **0** |

Terminal alignment path fires every epoch. Zero delta metrics likely because env tolerance did not shrink (no curriculum progression).

---

## 7. FPS & Stability

| Metric | Control | Distill |
|---|---|---|
| Mean FPS (step+infer+update) | 93,411 | 87,131 |
| P10 / P90 FPS | 79,602 / 112,232 | 72,702 / 107,884 |
| RuntimeError | 0 | 0 |
| OOM | 0 | 0 |
| Traceback | 0 | 0 |
| Fail-fast / generation mismatch | 0 | 0 |

Distill adds ~6% FPS overhead, no crashes.

---

## 8. Debug Env State (end / peak)

| Debug metric | Control end | Distill end |
|---|---|---|
| near_goal_rate | 0.065 / 0.082 | 0.015 / 0.019 |
| is_success_rate | 0.0015 / 0.0038 | **0.0 / 0.0** |
| lifted_object_rate | 0.667 / 0.687 | 0.555 / 0.671 |
| d_hat_mean | 0.498 / 0.849 | 0.741 / 0.874 |

Distill lifts objects similarly but **rarely approaches goal** and never triggers success chain.

---

## Analysis

1. **Gradient coupling works mechanically** — distill loss logged, active after 20M warmup, PPO params updated, zero fail-fast errors.
2. **Coupling hurts task performance** — distill arm ends with nsucc=0 and PPO reward ~3× lower than control despite identical seed/config except coupling.
3. **Likely root cause:** RSSM value teacher (mean ~1.78 on returns-to-go) provides a dense auxiliary target that **competes with sparse success learning**. With weight=0.05 and no task-success gating, distillation may pull shared encoder toward WM belief features that do not correlate with near-goal / success-chain behavior at 200M.
4. **WM value head itself trains** (loss_value→0.14) but this does not translate to PPO success — consistent with pre-P0 audit that independent WM training ≠ policy benefit.
5. **Neither arm reached nsucc_mean≥10** in 200M single-policy mode; acceptance time criterion is inapplicable. Even relative to control's modest peak (0.74), distill is a clear regression.

---

## Recommended Next Steps (before 800M/3B)

1. **Do not scale** — AB fails acceptance.
2. Re-run ablations at 200M:
   - Lower `--seq_wm_policy_coupling_weight` (0.005, 0.01)
   - Longer warmup (50M–100M) so WM stabilizes before coupling
   - Gate coupling on nsucc_mean > 0 or teacher value variance threshold
3. Inspect teacher value vs actual Monte-Carlo returns on successful vs failed rollouts.
4. Consider P1 aux scalars as WM inputs only after coupling no longer regresses baseline.

Raw JSON metrics: `results/seq_wm_ab_200m_report.json`
