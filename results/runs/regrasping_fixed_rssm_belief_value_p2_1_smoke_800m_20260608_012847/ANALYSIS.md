# Phase C Analysis — fixed `rssm_belief_value` 800M smoke

**Run:** `regrasping_fixed_rssm_belief_value_p2_1_smoke_800m_20260608_012847`
**Generated:** 2026-06-08 09:59 CST
**Strategy:** fixed `rssm_belief_value` (Phase A4 sequence WM, RSSM-lite + belief value head)
**Framework:** Stability-aware DexPBT P2.1 + PPO + 6-policy population, no CF, no h3

---

## 1. Headline numbers

- 6/6 policies **completed** 800M frames target (max 860M, 6112/6112 epoch).
- **Population peak `nsucc_mean`**: max **34.57**, median **32.19**, min **30.48**.
- **Population end `nsucc_mean`**: median **30.32**, mean **29.94**.
- **Retention** (end / peak): median **93.4 %**, min 87.7 % → no monotonic collapse.
- `nsucc_max = 50` saturated for all policies (env cap reached). p2 saturates first @ 220 M frames.

> Phase C gate from the plan was **best policy peak ≥ 20**. Margin here is comfortable: best policy is **34.57**, median is **32.19**.

## 2. Phase C learning curve (population aggregate)

| ≤ frames (M) | mean nsucc | median | min | max |
|---:|---:|---:|---:|---:|
| 400 | 16.62 | 20.26 | 0.74 | 26.30 |
| 600 | 24.30 | 28.23 | 3.46 | 29.75 |
| 800 | 30.97 | 31.32 | 26.99 | 34.57 |
| 850 | 30.01 | 30.32 | 28.17 | 31.34 |

- Strong monotonic improvement up to 800M.
- Mild dip 800 → 850M (≈1 unit, mostly from p1) — within 95 % retention band.

## 3. Cross-run comparison (800M smokes)

| Run | n | peak max | peak med | peak min | end med | min tol | frames |
|-----|---:|---:|---:|---:|---:|---:|---:|
| **fixed rssm_belief_value (this run)** | 6 | **34.57** | **32.19** | 30.48 | 30.32 | 0.0580 | 860M |
| cbwm no_h3 (`…135426`) | 6 | 22.28 | 16.46 | 8.97 | 12.24 | 0.0473 | 800M |
| cbwm no_h3 (`…040330`) | 4 | 17.34 | 17.34 | 17.34 | 17.34 | 0.0677 | 800M |

**Take-away:** under the same P2.1 stability stack, fixed `rssm_belief_value` reaches **median peak ~32** vs cbwm-no-h3 baselines at **median peak ~16–17**, a ~2× lift on this metric for the 800M budget.

> Caveats:
> 1. `min_tolerance` differs across runs. RSSM stayed at 0.058; cbwm `…135426` reached 0.047 (tighter curriculum). Lower tol means harder env, so head-to-head comparison is **not strictly apples-to-apples**.
> 2. fixed_h1 800M smoke directories were cleaned earlier in this session; only summary numbers from prior reports survive (peak ≈ 22.3 / median ≈ 16.5). RSSM is well above that.
> 3. This is a **smoke-budget run, not a formal 3B/5B**. Curriculum is still loose (tol ≈ 0.058 vs target 0.01). Expect numbers to compress when tolerance is forced down at longer horizons.

## 4. Stability / engineering audit (from `SMOKE_P2_REPORT.md`)

- All 6 policies started, **0 startup hangs**, **0 recovery restarts**.
- 28 PBT cross-copy restarts across the population (no PBT churn explosion).
- 31 OOM-style `Killed` events recovered by supervisor; max 9 `Killed` on p3 — slightly higher than fixed_h1 baseline (which usually had ≤ 3 per policy on this hardware).
- No collapse-freeze trigger fired (S3 marked FAIL, but S2 retention is healthy → likely no collapse to freeze in the first place; consistent with monotonic learning).

**Verdict:** functionally stable but a touch flakier than fixed_h1 on memory. RSSM adds GRU + Gaussian latent + 6 prediction heads on top of PPO; the OOMs are recoverable by supervisor.

## 5. Self-improvement signal (proxy for SeqWM auxiliary value)

- 76 in-policy ticks (no restart): mean Δ nsucc_mean per 20M tick = **+1.66**, median **+0.80**, max **+26.5** (post-cross-copy spike).
- This is the right sign: between PBT restarts, RSSM-augmented policies keep gaining.

## 6. Known noise in the report

- `CB-WM pool hard-check` flags `aux_low / value_imag_h3` in `wm_posterior_latest.json` as forbidden, and complains 0 CF blocks. Both are **expected for a fixed-strategy run** — the posterior is initialized with the legacy 5-arm pool but never sampled (mode = fixed); CF is disabled by design (`cf_runner_enabled=false`). The report was written for bayesian sweeps and over-flags here; not a real failure.
- `B2 collapse freeze detected` and `C1 stable-source tier used` failures are also vacuous: there was no collapse and no PBT replacement actually went through the “stable archive” tier (P2.1 only writes that tier on hard collapse).

## 7. What this tells us about RSSM-lite

- **Engineering**: SeqWM trainer + sampler + `rl_games` patches survive a 6-policy 800M run. Aux fields (`d_hat`, `failure`, `lifted`, `success_count`) and `wm_seq_checkpoints` are emitted; PBT cross-copy correctly inherits within-family (we added the `wm_seq_checkpoints` warm-start path earlier this session).
- **Optimization**: Loss scale stayed bounded; no NaN. SeqWM grad path does not destabilise PPO at `lambda_total=0.05`, `lambda_value=0.1`, `value_detach_belief=False`.
- **Performance signal**: median peak roughly **2×** the historical cbwm-no-h3 800M smoke median; comparable or above the fixed_h1 800M repeat smoke. **Necessary, not sufficient** for claiming RSSM beats fixed_h1 — the strong claim needs a longer horizon (3B) at tighter tolerance.

## 8. Next gates

- **Phase C, Trans variant**: launch `start_regrasping_fixed_trans_belief_value_p2_1_smoke_800m.sh` for the family-internal comparison. Same gate as RSSM (peak ≥ 20).
- **Phase D**: pick the stronger of {RSSM, Trans} (or both) for a 3B fixed run, where curriculum tolerance can actually approach 0.01 and we re-rank against fixed_h1 3B / formal 5B.
- **Phase E**: 3B bayesian with stage-aware selector (`none, aux_mid, rssm_belief_value, trans_belief_value`). The stage-selector library is in place; observer/CF wiring still needs the stage-context follow-up before this is a proper experiment.
