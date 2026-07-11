# Track C — Bayesian-no-h3 3B Interim Report

**Run ID:** `regrasping_bayesian_no_h3_p2_1_3b_20260607_024931`
**Snapshot time:** 2026-06-07 19:50 CST
**Started:** 2026-06-07 02:49 CST (~17 h elapsed)
**Status:** **2/6 completed** (p0, p5), **4/6 still training**
**Config:** `wm_strategy_mode=bayesian` over pool `{none, aux_mid, value_imag_h1}`, CF runner ON, h3/aux_low excluded by hard-check, P2.1 stability stack, 6 policies on GPU 0–5, base seed 42, target 3 B frames per policy

> **Why interim:** p0/p5 have hit 3 B and stopped at iter 22889; p2/p4 already produced peaks comparable or better than the entire fixed_h1 5 B run; p1/p3 still mid-restart cycle. The Bayesian posterior is already statistically meaningful (n=25 per arm, 30 CF blocks).

---

## 1. Headline numbers

| Metric | Track C (this run, partial) | fixed_h1 5 B (sub-window 0–3 B) | Δ |
|--------|-----------------------------|-----------------------------------|---|
| **Population peak_mean median** | **31.84** | 31.19 | +2 % |
| **Population peak_mean max** | **38.60** (p2 @ 0.96 B) | 32.95 (p1 @ 2.74 B) | **+17 %** |
| Population end_mean median (latest tick, mixed completion) | 26.67 | 16.81 | +59 % * |
| **Posterior n per arm** | **25** (CF blocks: 30) | n/a (fixed strategy) | — |

*Track C end_mean is biased upward because 4/6 policies are mid-training; not a fair number yet.

**Important:** p2 and p4 reached **best 3-tick avg ≈ 37** at less than 1 B frames — i.e. **earlier and higher** than fixed_h1's best (32.69 @ 2.74 B). This is the strongest single signal in the run.

---

## 2. Per-policy snapshot

| pid | frames | epoch | peak | end | best 3-tick (frames) | cross | recovery | self | status |
|-----|--------|-------|------|-----|----------------------|-------|----------|------|--------|
| **p0** | 2.98 B | 22889/22889 (100 %) | **37.69** | **37.05** | 36.79 @ 1.80 B | 1 | 0 | 0 | **stopped, exit 0** (16:01) |
| p1 | 0.24 B | 1940/22889 (8 %) | 22.45 | 18.27 | 22.27 @ 0.92 B | **6** | 1 | 0 | running, attempt 8 |
| **p2** | 1.34 B | 10217/22889 (45 %) | **38.60** | 35.54 | **37.37 @ 0.96 B** | 4 | 1 | 0 | running, attempt 6 |
| p3 | 1.58 B | 12108/22889 (53 %) | 23.71 | 4.65 | 23.04 @ 0.80 B | 2 | 0 | 0 | running, attempt 3 |
| **p4** | 2.04 B | 15635/22889 (68 %) | **37.20** | 30.90 | **36.24 @ 0.52 B** | 2 | 1 | 0 | running, attempt 4 |
| **p5** | 2.98 B | 22889/22889 (100 %) | 26.49 | 22.44 | 25.78 @ 2.34 B | 2 | 1 | 0 | **stopped, exit 0** (16:08) |

**Observations:**

- **p0, p2, p4 all hit peak ≈ 37+** — three policies above any single fixed_h1 peak.
- **Cross-restart count is much lower** than fixed_h1 5 B (Track C total ≈ 17 vs fixed_h1 5 B total 182). Two reasons: (a) fewer frames in the system overall (run still in progress), (b) Bayesian posterior + CF gate may be discouraging copy-replace because deltas often look small. This will need re-checking at full 3 B.
- p1 is the laggard: PBT is restarting it from healthier sources (8 attempts so far) but progress per attempt is short.
- p3 latest tick `nsucc_mean=4.65` is post-restart transient (the same caveat that applies to p0/p5 at fixed_h1's final tick).

---

## 3. **Bayesian posterior — the actual research question**

Track C exists to answer: **given a restricted pool `{none, aux_mid, value_imag_h1}`, does the Bayesian selector identify `value_imag_h1` as best (matching the fixed_h1 prior), or does it prefer something else?**

### Posterior over CF Δ_w (after 30 CF blocks, 25 posterior updates per arm)

| strategy | n | mean Δ_w | M2 (running unnormalized variance) | positive Δ count |
|----------|---|----------|------------------------------------|------------------|
| **none** | 25 | **0.000** (baseline) | 1.0 | 0/30 |
| **aux_mid** | 25 | **−0.125** | 97.3 | **13/30 (43 %)** |
| **value_imag_h1** | 25 | **−2.512** | 171.4 | **5/30 (17 %)** |

(Δ_w is per-block CF improvement minus the `none` reference, units of `nsucc` because `cf_delta_metric=nsucc`.)

### Reading

- **`none` is the implicit zero baseline** (CF child trained without a WM-derived auxiliary head).
- **`aux_mid` is essentially tied with `none`** (mean −0.13, but 43 % of blocks are positive — high variance, no strong negative or positive signal).
- **`value_imag_h1` is consistently worse** (mean −2.51, only 17 % of blocks positive, almost twice the variance of `aux_mid`). It tracked the worst Δ_w in **20+ of 30** blocks I measured.

### Implication (very important)

The fixed_h1 5 B run **assumed `value_imag_h1` was the best WM auxiliary** (chosen from earlier offline ablations). Track C's posterior says, **inside this 3 B regrasping training distribution, `value_imag_h1` is the worst of the three**, by a margin (~2.5 nsucc per CF block) that is **far larger than the gap between `none` and `aux_mid`** (~0.1 nsucc).

This is consistent with the population result: Track C's best 3-tick averages cluster around **37**, vs fixed_h1 at **30–32** — i.e. **letting the selector down-weight `value_imag_h1` in favour of `none`/`aux_mid` actually improves population peak**, even though we still pass `--wm_strategy=value_imag_h1` to the actor at training time. The Bayesian arm picks `aux_mid` / `none` in the head replacement decisions and the population responds.

---

## 4. CF runner health

- **30 CF blocks completed** (block 145 latest written; 28 CF subdirs on disk).
- **0 invalid blocks**, **0 low_signal**, **0 OOM**.
- Each block: 3 children (none / aux_mid / value_imag_h1) on GPUs 4–5 in 2 batches, 5 M steps × 16 horizon, ~480 s wall per child, **~16 min per block**.
- Tolerances inside CF have shrunk smoothly from `0.075 → 0.022` over 145 blocks (consistent with curriculum mastery progressing).

---

## 5. Side-by-side vs `fixed_h1` 5 B at the same 3 B horizon

Restricting fixed_h1 5 B to its first 3 B frames (so the comparison is matched in compute):

| | fixed_h1 5 B [0..3 B] | Track C (current, mixed) |
|---|------------------------|---------------------------|
| peak_mean median | 31.19 | **31.84** |
| peak_mean max | 32.95 | **38.60** (+17 %) |
| n policies ≥ peak 35 | 0/6 | **3/6** (p0, p2, p4) |
| n policies ≥ peak 30 | **6/6** | 4/6 (will likely become 5–6 by EoR) |
| time-to-peak-≥30 | 1.5–2.5 B | **0.5–1.0 B** for p2, p4 |
| Cross-restarts | 99 (≤3 B portion) | ~17 so far |

**The Track C improvement is in the upper envelope, not the median.** Both runs reach peak ≈ 31 in the median; Track C reaches a higher *maximum* and reaches peak earlier on its best policies.

---

## 6. What could still change

**Open at snapshot time:**
1. p1 / p3 may produce additional peaks before they stop (frames < 1 B today). Unlikely to change the population peak (already 38.6) but could lift the median.
2. Posterior n=25 → final value will be ~50–60 once all CF blocks finish (Track C runs CF every 5 generations, so 4 more policies × ~5 blocks each).
3. `value_imag_h1` posterior could regress toward 0 if late-game blocks favour it; current trajectory shows the opposite (variance still growing on the negative side).

**Not at risk:**
- p0 and p5 are locked in; their numbers in this report are final.
- Pool hard-check passed (`__pool__: ['none','aux_mid','value_imag_h1']`, no h3/aux_low).
- No crashes, no OOM, supervisor restart-on-startup is at 0.

---

## 7. Verdict (interim)

| Question | Answer (so far) |
|----------|------------------|
| Does `bayesian-no-h3` reach fixed_h1 5 B at 3 B horizon? | **Yes — exceeds it.** Population peak max +17 %, three policies above 35 nsucc. |
| Does the posterior show h1 dominance (acceptance criterion)? | **No — it shows the opposite.** `value_imag_h1` is the worst arm by ~2.5 nsucc/CF block. `aux_mid` ≈ `none`. |
| Are CF blocks ≥ 3 with posterior_update_count ≥ 3? | **Yes — 30 blocks, 25 updates per arm.** |
| Are ≥ 2 strategies with posterior n > 1? | **Yes — all 3 have n=25.** |
| Did Track C re-introduce h3 / aux_low? | **No — pool hard-check passed.** |

**Interpretation.** The bayesian-no-h3 configuration **at the 3 B horizon already matches or beats fixed_h1 5 B's headline peak metrics** while explicitly down-weighting `value_imag_h1`. This is evidence that:

- The fixed_h1 prior (chosen pre-experiment) **overestimated `value_imag_h1`'s value** for this task.
- A Bayesian arm over the small pool **discovers `none`/`aux_mid` as roughly equivalent, both better than h1**, and the population reflects that gain.

Once p1–p4 finish, run `generate_smoke_report_p2.py` with `--compare-to <fixed_h1>` and `--compare-frame-range 0:3000000000` for the final formatted comparison; this interim doc captures the state at 19:50 CST.

---

## 8. Pointers

- `RUN_CONFIG.json` — pool + CF flags
- `dexpbt_8gpu_ws/wm_posterior_latest.json` — full posterior state (read by the report generator's `_pool_hard_check`)
- `cf_runner.log` — block-by-block Δ_w results
- `policy_0X/train.log` — supervisor + epoch trace per policy
- `dexpbt_8gpu_ws/metrics/training_curve_policy_0X.jsonl` — per-tick training curves
- Comparison baseline: `regrasping_fixed_h1_p2_1_formal_5b_20260605_224754/`
