# Phase C Analysis — fixed `trans_belief_value` 800M smoke (INTERIM)

**Run:** `regrasping_fixed_trans_belief_value_p2_1_smoke_800m_20260608_100048`
**Generated:** 2026-06-08 12:37 CST
**Status:** **INTERIM — 1/6 policies completed (p0); p1–p5 still training (~660–800M frames)**
**Strategy:** fixed `trans_belief_value` (Phase A4 sequence WM, Transformer-lite + belief value head)
**Framework:** Stability-aware DexPBT P2.1 + PPO + 6-policy population, no CF, no h3
**Formal report:** `SMOKE_P2_REPORT.md` (auto-generated from current training curves)

---

## 0. How to read this document

Numbers below are **as-of snapshot time**, not final. Training curves update every ~20M PBT tick; p1–p5 will move before the run finishes. Treat population aggregates as **lower bounds on variance** (p1 is a known outlier) and **upper bounds on completion** (p5 is already at ~801M).

When all six policies hit 6112/6112 epoch, re-run:

```bash
python3 results/generate_smoke_report_p2.py \
  results/regrasping_fixed_trans_belief_value_p2_1_smoke_800m_20260608_100048 \
  --compare-to results/regrasping_fixed_rssm_belief_value_p2_1_smoke_800m_20260608_012847
```

---

## 1. Headline numbers (all observed policies)

| Metric | Trans (this run, interim) | RSSM (completed) | Δ (Trans − RSSM) |
|--------|---------------------------|------------------|------------------|
| Policies completed | **1 / 6** | 6 / 6 | — |
| Peak `nsucc_mean` max | **24.24** (p0) | **34.57** (p3) | **−10.3** |
| Peak `nsucc_mean` median | **21.55** | **32.19** | **−10.6** |
| Peak `nsucc_mean` min | **2.09** (p1) | **30.48** (p1) | **−28.4** |
| End `nsucc_mean` median | **20.40** | **30.32** | **−9.9** |
| Retention (end/peak) median | **94.4%** | **93.4%** | +1.0 pp |
| Retention min | **78.2%** (p4) | **87.7%** (p3) | −9.5 pp |
| OOM `Killed` events | **19** | **31** | −12 |
| PBT cross-copy restarts | **17** | **28** | −11 |
| Min success tolerance reached | **0.0522** | **0.0580** | slightly tighter |

**Phase C gate (plan: best policy peak ≥ 20):** **PASS on max** (24.24 ≥ 20). Median peak also clears the bar (21.55). The run is **not a population failure**, but it is **materially weaker than RSSM** on every aggregate except OOM count.

**Excluding p1 outlier** (see §5): peak median ≈ **21.2**, peak min ≈ **17.9** — still well below RSSM’s population floor (~30.5).

---

## 2. Per-policy snapshot

| pid | Status | frames (M) | peak | end | retention | cross | Killed | tol (tail) |
|-----|--------|----------:|-----:|----:|----------:|------:|-------:|-----------:|
| p0 | **DONE** | 800 | 24.24 | 22.87 | 94.4% | 0 | 0 | 0.0522 |
| p1 | running | 661 | 2.09 | 2.09 | 100%* | 7 | 8 | 0.0750 |
| p2 | running | 759 | 17.87 | 14.41 | 80.6% | 3 | 3 | 0.0522 |
| p3 | running | 753 | 21.62 | 20.40 | 94.4% | 2 | 2 | 0.0522 |
| p4 | running | 787 | 20.78 | 16.24 | 78.2% | 1 | 1 | 0.0522 |
| p5 | running | 797 | 21.55 | 21.55 | 100% | 4 | 5 | 0.0750 |

\*p1 retention is meaningless at peak = end = 2.09.

**Completed-only (p0):** end 22.87, peak 24.24 — healthy single-policy finish, comparable to a strong cbwm-no-h3 worker but far from RSSM p0 (peak 31.77).

---

## 3. Learning curve (population aggregate)

Snapshots at common frame budgets (nearest PBT tick):

| ≤ frames (M) | Trans mean | Trans med | Trans min–max | RSSM mean | RSSM med | RSSM min–max |
|---:|---:|---:|---:|---:|---:|---:|
| 400 | 15.23 | 15.77 | 7.05–19.52 | 24.26 | 23.76 | 20.99–28.03 |
| 600 | 18.50 | 18.22 | 16.02–21.49 | 28.34 | 28.18 | 26.79–30.22 |
| 800 | 22.87† | 22.87† | 22.87† | 30.22 | 30.75 | 27.12–31.58 |

†Only p0 has reached 800M; the 800M row is **not** a population aggregate yet.

**Take-away:** from 400M onward Trans trails RSSM by roughly **8–10 `nsucc_mean` units** at the population median. The gap is stable, not a late-run artifact — Transformer-lite is learning, but slower and with a lower ceiling on this budget.

---

## 4. RSSM vs Trans (same stack, same launch profile)

Reference comparison is appended in `SMOKE_P2_REPORT.md` § Compare to reference run.

| metric | Trans (interim) | RSSM (final) |
|--------|----------------:|-------------:|
| peak_mean median | 21.16 | 31.98 |
| peak_mean max | 24.24 | 34.57 |
| best 3-tick rolling median | 20.13 | 31.58 |
| time-to-90%-peak | 380M / **660M** / 740M | 480M / **600M** / 760M |

- **Sample efficiency:** Trans reaches 90% of its (lower) peak slightly earlier on the best policy (p0 @ 380M) but population median time-to-peak is **660M vs 600M** — not faster overall.
- **Ceiling:** RSSM population median peak is **~50% higher** than Trans (32.2 vs 21.5).
- **Stability:** Both runs pass P2 retention (S2). Trans triggered collapse-freeze once (S3 PASS) and one hard-collapse rollback (S4 PASS); RSSM had neither — consistent with noisier mid-run dips on p2/p4.
- **Engineering:** SeqWM connected cleanly (`TransformerLiteWorldModel`, seq_len=16, λ_total=0.05). Fewer OOM kills than RSSM (19 vs 31) despite attention block — likely run-length / restart luck, not a structural win.

**Phase D recommendation (interim):** advance **RSSM** to 3B fixed. Keep Trans as a **secondary** 3B candidate only if final p1–p5 numbers close >30% of the RSSM gap (unlikely from current trajectories).

---

## 5. p1 outlier audit

p1 (seed 1042, same as RSSM p1 which peaked at 30.48) is the dominant failure mode:

- **7 PBT cross-copy restarts**, **8 OOM kills**, **18 supervisor attempts** under `policy_01/`.
- Peak stuck at **2.09** after 661M frames; still on **tol = 0.075** (has not tightened like peers at 0.0522).
- Restart chain: bottom-ranked → cross from p0, p2, p5, … each followed by `Killed` during or right after resume — classic **restart + OOM + partial curriculum reset** loop.
- Cross-copy +50M deltas on p1 are tiny (+0.3 to +1.0) until the last event at 660M — no RSSM-style late jump (+26.5 @ 581M on RSSM p1).

This is **not** evidence that Transformer-lite is seed-sensitive in general (p0/p3/p5 are fine). It mirrors historical “bad p1” episodes in fixed_h1 smokes — **PBT + memory pressure on policy slot 1**, amplified here by longer wall time per attempt.

---

## 6. Stability / engineering (from `SMOKE_P2_REPORT.md`)

| Check | Result |
|-------|--------|
| All 6 started | PASS |
| Startup hangs | 0 |
| P2 S1–S8 (interim curves) | **8/8 PASS** (S4: 1 hard-collapse, rolled back) |
| PBT D1–D4 | **4/4 PASS** |
| CF / posterior sections | vacuous (fixed mode) — ignore FATAL pool flag |
| SeqWM checkpoints | emitted on completed p0; warm-start path exercised on cross-copy |

**Verdict:** engineering is **sound** (same as RSSM). Performance is the differentiator, not crashes or NaNs.

---

## 7. Cross-run context (800M smokes)

| Run | n | peak max | peak med | end med | notes |
|-----|---:|---:|---:|---:|---|
| **fixed rssm_belief_value** | 6 | 34.57 | 32.19 | 30.32 | Phase C reference |
| **fixed trans_belief_value (interim)** | 6‡ | 24.24 | 21.55 | 20.40 | this run |
| cbwm no_h3 (`…135426`) | 6 | 22.28 | 16.46 | 12.24 | bayesian pool |
| cbwm no_h3 (`…040330`) | 4 | 17.34 | 17.34 | 17.34 | partial pop |

‡1 policy at full 800M; others in flight.

Trans **beats cbwm-no-h3 on peak median** (21.5 vs 16.5) but **does not beat RSSM**. It lands near the **fixed_h1 800M repeat** band (~peak med 16–22 from prior smokes) — helpful but not the family winner.

---

## 8. What Transformer-lite tells us (interim)

**Working:**

- Full 800M budget is viable on p0; peak 24.2 clears smoke gate.
- Belief-value auxiliary + PPO co-training stable at configured λ.
- PBT cross-copy inherits `wm_seq_checkpoints` within Trans family (same patch as RSSM).

**Weak vs RSSM-lite:**

- Lower sample efficiency after 400M (~10-point median gap at 600M).
- Lower peak ceiling (population med ~21 vs ~32).
- p1-style restart churn wastes ~140M frames without recovery.

**Not yet known (needs final 6/6):**

- Whether p2/p4 retention dips (78–81%) recover or worsen at 800M.
- Final population end median vs RSSM end median.
- Whether p5 (already at 801M, peak 21.55) gains another +2–3 units in remaining epochs.

---

## 9. Next steps

1. **Let p1–p5 finish** — expect ~1–3 h for p2–p5, **several hours** for p1 if it keeps OOM-looping.
2. **Regenerate** `SMOKE_P2_REPORT.md` + replace this file with a FINAL version.
3. **Phase D:** launch **3B fixed `rssm_belief_value`**; defer Trans 3B unless final Trans peak med ≥ 28.
4. **Optional follow-up:** p1-only ablation (same seed, single-policy no-PBT 800M) to separate seed bad luck from Transformer-specific restart fragility.

---

## Appendix — file paths

| Artifact | Path |
|----------|------|
| Auto report | `…/100048/SMOKE_P2_REPORT.md` |
| RSSM completed run | `…/012847/` (+ `ANALYSIS.md`) |
| Launch script | `results/start_regrasping_fixed_trans_belief_value_p2_1_smoke_800m.sh` |
| Training curves | `…/100048/dexpbt_8gpu_ws/metrics/training_curve_policy_*.jsonl` |
