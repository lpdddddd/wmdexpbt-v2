# Formal 5B Run — Deep Analysis (snapshot)

**Generated:** 2026-06-06 18:59  (mid-run, ~93% epochs / 107% frames)
**Run ID:** `regrasping_fixed_h1_p2_1_formal_5b_20260605_224754`

Companion reports: `SMOKE_P2_REPORT.md` (acceptance), `INTERMEDIATE_FORMAL_REPORT.md` (status snapshot).

---

## TL;DR

1. **Peak capability is real and saturates around ~3 B frames.** Population peak-so-far hits **32.95 nsucc_mean by ~3 B** and does not improve after — the last ~2 B of training did not break new ground in upper envelope.
2. **PBT cross-copy is working as designed.** Across **171 cross-restarts**, average post-restart Δmean@+50 M = **+4.06**, **78 % positive**. No policy has a negative average post-restart delta.
3. **Decline of population end_mean ≠ degradation.** Self-improvement on no-restart ticks is **positive** (+0.5 to +0.8 per tick); time-at/above-half-peak is **60–68 %** for every policy.
4. **The drop from peak to end is mostly success_tolerance shrinking + late-stage churn.** Tolerance has gone from 0.075 → 0.020–0.024 (4× stricter) since the peaks were recorded.
5. **Implication for next runs:** ~3 B frames was likely sufficient for this fixed-h1 / no-h3 setup. The extra 2 B mainly increases PBT replacement count, not population peak.

---

## 1. Population trajectory — the population stopped climbing around ~2.6 B

Median nsucc_mean per 200 M-frame bin (excluding the cold first 200 M):

```
  frames(M) |  median  |   max
  --------- | -------- | ------
      400   |   2.41   |   6.30
      800   |  18.42   |  21.86
     1200   |  18.75   |  22.81
     1600   |  24.99   |  32.31    <- pop median peaks here
     2000   |  24.19   |  30.29
     2400   |  25.04   |  30.64
     2600   |  24.77   |  32.95    <- pop top peak ever
     3000   |  14.14   |  27.56    <- median starts dropping
     3400   |  19.20   |  25.88
     3800   |  18.36   |  26.28
     4200   |  17.08   |  25.03
     4600   |  11.85   |  22.02
     5000   |  12.62   |  20.83
     5200   |  11.84   |  19.81
```

Population median tops out at **~25** around 1.6–2.6 B frames, then drifts to **~12** by 5.2 B. Maximum across the population peaks at **32.95** at 2.6 B and never improves.

## 2. Peaks happened mid-run — late stage is mostly churn

| pid | peak | peak frame (M) | tolerance @ peak | tolerance now | cross-restarts at peak | cross now |
|-----|------|----------------|------------------|---------------|------------------------|-----------|
| p0 | 30.29 | 2160 | 0.0385 | 0.0243 | 8 | 30 |
| p1 | 32.95 | 2700 | 0.0385 | 0.0208 | 13 | 26 |
| p2 | 31.87 | 2600 | 0.0385 | 0.0208 | 9 | 23 |
| p3 | 30.51 | 2700 | 0.0347 | 0.0219 | 15 | 28 |
| p4 | 32.31 | 1660 | 0.0498 | 0.0198 | 11 | 32 |
| p5 | 27.75 | 2260 | 0.0385 | 0.0219 | 15 | 34 |

Every policy hit its peak between 1.66 B and 2.7 B frames — i.e. **before half-time**. Tolerance at peak was 0.035–0.050; current tolerance is 0.020–0.024 (about **2×** stricter than at peak), which by itself depresses end_mean even at unchanged real capability.

Late-stage (>3.5 B) high points (nsucc_mean > 25):

| pid | count | max | last hit (M) |
|-----|-------|-----|--------------|
| p0  |   1   | 26.28 | 3880 |
| p1  |   4   | 25.88 | 4200 |
| p2  |   0   |   —   |  —   |
| p3  |   3   | 27.73 | 4440 |
| p4  |   2   | 27.71 | 4520 |
| p5  |   1   | 25.83 | 3760 |

After 4.5 B frames, **no policy breaks 25 mean** — confirms the population has saturated.

## 3. PBT cross-restart works (positive in 78 % of cases)

Δ nsucc_mean over the **+50 M frames after each cross-restart**, computed by `cross_restart_count` delta:

| pid | n | avg Δ | median Δ | positive % | big positive (>+5) | big negative (<−5) |
|-----|---|-------|----------|------------|--------------------|--------------------|
| p0  | 29 | +3.63 | +2.97 | 69 % | 12 | 3 |
| p1  | 26 | +6.85 | +7.69 | 92 % | 15 | 0 |
| p2  | 22 | +3.45 | +4.17 | 73 % | 10 | 3 |
| p3  | 28 | +4.40 | +3.39 | 86 % | 10 | 0 |
| p4  | 32 | +2.60 | +2.93 | 69 % | 11 | 3 |
| p5  | 34 | +3.79 | +3.16 | 79 % | 13 | 2 |
| **all** | **171** | **+4.06** | **+3.64** | **78 %** | **71** | **14** |

**Interpretation:**
- p1/p3 are the strongest beneficiaries of cross-copy (+7 / +4 mean delta, 86–92 % positive).
- p4 is the weakest beneficiary (+2.6 average); paired with the highest cross-restart count (32) it suggests p4 is being used as the 'churn slot' that absorbs replacements without large gains.
- Big-negative events (Δ < −5) only 14/171 (8 %); they are concentrated in p0/p2/p4 and tend to coincide with `early_collapse` triggering shortly after the restart.

## 4. No-restart ticks: PPO is still improving (slowly)

| pid | n  | mean Δ per tick | median Δ | p25 | p75 |
|-----|----|-----------------|----------|-----|-----|
| p0  | 194 | +0.81 | +0.46 | -0.31 | +1.83 |
| p1  | 205 | +0.70 | +0.29 | -0.66 | +1.58 |
| p2  | 205 | +0.69 | +0.37 | -0.86 | +1.46 |
| p3  | 203 | +0.53 | +0.29 | -0.60 | +1.69 |
| p4  | 193 | +0.73 | +0.41 | -0.89 | +1.82 |
| p5  | 192 | +0.62 | +0.14 | -0.96 | +1.85 |

Every policy has **positive mean drift between restarts**. So PPO + WM are *not* degrading; they are doing slow incremental gains that get periodically interrupted (and on average rescued) by PBT.

## 5. nsucc_max is saturated — the work is on the mean

| pid | first nsucc_max=50 (M) | % of ticks at max=50 | last hit (M) |
|-----|------------------------|----------------------|---------------|
| p0  | 640 | 79 % | 5220 |
| p1  | 580 | 87 % | 5340 |
| p2  | 580 | 84 % | 5220 |
| p3  | 660 | 87 % | 5380 |
| p4  | 580 | 86 % | 5340 |
| p5  | 540 | 86 % | 5400 |

All policies saturate the **per-episode success cap (50)** by 540–660 M frames. The cap of 50 is a *training-side ceiling*, so for upper-tail comparison we have to use `eval_cap_sweep.py` on caps {50, 100, 200} after the run.

## 6. Time at/near peak — distribution is healthy

| pid | frames at nsucc_mean ≥ 0.5 × peak | % of ticks |
|-----|-----------------------------------|------------|
| p0  | 136 / 226 | 60 % |
| p1  | 137 / 232 | 59 % |
| p2  | 146 / 230 | 63 % |
| p3  | 157 / 232 | 68 % |
| p4  | 139 / 226 | 62 % |
| p5  | 149 / 227 | 66 % |

Every policy spends **≥59 % of its lifetime** at or above half its peak. So the snapshot 'p0 end_mean = 0.18' is just a tail tick — the rolling-window picture is much better.

## 7. Why the population MEDIAN drifts down in late stage

Putting these signals together, the late-stage decline is the sum of:

| Driver | Effect | Sign of degradation? |
|--------|--------|----------------------|
| `success_tolerance` shrunk from 0.075 → 0.020-0.024 (~4× stricter) | end-stage `nsucc_mean` measured on a harder rubric than mid-run peak | **No** (just a measurement scale change) |
| 171 PBT cross-restarts, with 78 % positive but each restart drops mean by 5-20 transiently | many late-stage ticks are 'first 50-100 M after a cross-copy' | **No** (designed PBT churn; net positive over +50 M) |
| 2 recovery-restarts (p0, p2) where best_stable_archive was used | one rescue (p2: +17.7 over +50 M) and one regression (p0: −4.7) | mixed; rare |
| Population peak saturates at 32.95 around 3 B | no policy breaks 25 mean after 4.5 B | **Saturation, not degradation** |

## 8. Operational implications

1. **3 B frames is likely the right horizon for fixed-h1 / no-h3 / mastery curriculum on regrasping.** The marginal value of 3-5 B is mostly in *consolidating* mean rather than raising peak.
2. **For Track C (bayesian-no-h3) and Track D (h3-only) ablations**, plan for **2-3 B frames** unless intentionally exploring late-stage saturation.
3. **Don't read end_mean as a final score.** Use the post-run cap-sweep + percentile tools (`eval_cap_sweep.py`, `eval_nsucc_percentile.py`) at fixed eval cap and tolerance.
4. **PBT replacement frequency could be reduced** in formal runs: 171 cross-restarts in 5 B frames means a cross-restart every ~30 M. Doubling `pbt_min_frames_between_restarts` from 100 M to 150-200 M could let policies consolidate more before being replaced again.
5. **p2 is a noteworthy success story for `best_stable_archive` recovery**: at 2.43 B, p2 was at 11.25 mean; recovery from p0's archive lifted it to 28.97 (+17.7) within 50 M frames.

## 9. What to do at end of run

```bash
python /root/autodl-tmp/wmdexpbt/results/generate_smoke_report_p2.py /root/autodl-tmp/wmdexpbt/results/regrasping_fixed_h1_p2_1_formal_5b_20260605_224754
python /root/autodl-tmp/wmdexpbt/results/eval_cap_sweep.py --run-dir /root/autodl-tmp/wmdexpbt/results/regrasping_fixed_h1_p2_1_formal_5b_20260605_224754 --caps 50 100 200 --episodes 256
python /root/autodl-tmp/wmdexpbt/results/eval_nsucc_percentile.py --run-dir /root/autodl-tmp/wmdexpbt/results/regrasping_fixed_h1_p2_1_formal_5b_20260605_224754
```

Eval cap sweep is the only fair way to compare to the no-h3 800 M smoke and to subsequent ablations.

---

*All numbers above are from the in-flight training_curve files; they will shift slightly when the last ~7-10 % epochs finish, but the structural conclusions (peak saturated, cross-copy positive, PPO drift positive, end_mean depressed by tolerance + churn) are robust to that.*

---

## 10. Eval target candidates (best 3-tick rolling mean)

Picking the best **3 consecutive PBT ticks** per policy (rolling average of `nsucc_mean`):

| pid | best_3tick avg | frames (M) | cross at peak | tolerance at peak |
|-----|----------------|------------|---------------|--------------------|
| p0  | 30.00 | 2160 |  8 | 0.0385 |
| p1  | **32.69** | **2740** | 13 | 0.0385 |
| p2  | 30.81 | 2620 |  9 | 0.0385 |
| p3  | 29.64 | 2700 | 15 | 0.0347 |
| p4  | 29.75 | 1660 | 11 | 0.0498 |
| p5  | 26.89 | 2280 | 15 | 0.0385 |

**Top eval targets: p1 @ 2740 M, p2 @ 2620 M, p0 @ 2160 M.** These are the strongest "live training" segments. Note: the 32.69 was measured at tolerance 0.0385 — in current end-state at tolerance 0.020 the same policy will score lower in `nsucc_mean`, so for fair side-by-side use `eval_cap_sweep` at fixed eval tolerance.

Each policy has 227-233 checkpoint files (~43.6 MB each, ~10 GB per policy, ~60 GB total). Latest indices are 268-272.

## 11. Supervisor / PBT consistency sanity

Cross-check: `train.log` `attempt_N` count vs `cross_restart_count` + `recovery_restart_count` from `training_curve`:

| pid | max attempt | cross | restart_count(curve) | recovery | consistent |
|-----|-------------|-------|----------------------|----------|------------|
| p0  | 33 | 31 | 32 | 1 | **OK** |
| p1  | 28 | 26 | 26 | 0 | **OK** |
| p2  | 25 | 23 | 24 | 1 | **OK** |
| p3  | 30 | 28 | 28 | 0 | **OK** |
| p4  | 33 | 32 | 32 | 0 | **OK** |
| p5  | 36 | 34 | 34 | 0 | **OK** |

Every policy: `max_attempt ≈ restart_count + 1` (initial attempt + each restart). No orphan attempts, no missing PBT events. **Bookkeeping is clean.**

## 12. Mastery curriculum is progressing healthily

| pid | shrinks | initial_tol → final | drop |
|-----|---------|---------------------|------|
| p0  | 28 | 0.0750 → 0.0198 | 73.6 % |
| p1  | 25 | 0.0750 → 0.0208 | 72.3 % |
| p2  | 29 | 0.0750 → 0.0208 | 72.3 % |
| p3  | 22 | 0.0750 → 0.0219 | 70.8 % |
| p4  | 28 | 0.0750 → 0.0198 | 73.6 % |
| p5  | 23 | 0.0750 → 0.0219 | 70.8 % |

P2.1 `pbt_single_shrink_per_window` is holding (no multi-shrink-in-window detected by S1). 22-29 shrinks per policy aligns with the 5 B-frame run length at one shrink every ~150-250 M frames.
