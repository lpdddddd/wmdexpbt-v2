# Formal 5B Run — Intermediate Snapshot

**Run ID:** `regrasping_fixed_h1_p2_1_formal_5b_20260605_224754`
**Snapshot at:** 2026-06-06 18:50  (run still in progress, NOT a final report)
**Elapsed:** 20:02:15.757734 (started 2026-06-05 22:47 CST)
**Config:** fixed `value_imag_h1`, no h3, no CF, P2.1 stability, 6 policies on GPU 0–5, seed_base=42

> Companion file: `SMOKE_P2_REPORT.md` (full acceptance report + raw tables).  
> This snapshot is for reading the run mid-flight; final numbers may shift in the last ~10% epochs.

## Status at snapshot

| Item | Value |
|------|-------|
| Frames (max policy) | **5380M** / 5000M (108%) |
| Population end_mean median | **13.70** |
| Population end_mean mean   | **10.76** |
| Population peak_mean median | **31.87** |
| Population peak_mean max    | **32.95** |
| Cross-restart total | **172** |
| Recovery-restart total | **2** |
| Startup hang count | **0** |

## Per-policy snapshot

| pid | frames(M) | end_mean | peak_mean | peak@(M) | post-restart peak | end_tol | cross/rec | tier |
|-----|-----------|----------|-----------|----------|-------------------|---------|-----------|------|
| p0 | 5280 | 0.18 | 30.29 | 2160 | 13.83 | 0.0208 | 30/1 | bottom |
| p1 | 5300 | 13.70 | 32.95 | 2700 | 14.53 | 0.0219 | 26/0 | mid |
| p2 | 5220 | 13.11 | 31.87 | 2600 | 19.43 | 0.0243 | 22/1 | bottom |
| p3 | 5340 | 6.59 | 30.51 | 2700 | 14.54 | 0.0219 | 28/0 | mid |
| p4 | 5300 | 17.14 | 32.31 | 1660 | 17.14 | 0.0198 | 32/0 | top |
| p5 | 5380 | 13.83 | 27.75 | 2260 | 16.19 | 0.0219 | 34/0 | mid |

## Key results from `SMOKE_P2_REPORT.md`

- **P2 acceptance:** 7/7 PASS (S4 N/A as no hard collapse).
- **PBT D-criteria:** 4/4 PASS (D1/D2/D3/D4) — all 6 policies started, 0 hangs, diverse curves, 174 PBT replacements.
- **P1 acceptance:** 7/7 PASS (peak ≥3.5, best end ≥2, no monotonic collapse, mastery shrink/freeze active, stable-source tier used).
- **Recovery audit:** 4 entries (2 selected + 2 executed); both recoveries used `best_stable_archive`.
- **Source tier mix:** strict_stable_live=312, usable_live=168, relaxed_live=7 → stable-source path dominates.

## Comparison vs no-h3 800M smoke (sealed prior run)

| Metric | no-h3 smoke (completed-only, p1 partial dropped) | Formal 5B (in-flight) |
|--------|-------------------------------------------------|------------------------|
| Population peak_mean median | ~22.5 | **~31.9** |
| Population peak_mean max    | ~25.0 | **~33.0** |
| Population end_mean median  | ~9.1  | **~13.8** (still in late-stage churn) |
| Cross restart total         | ~30   | **172** |
| Recovery restart total      | 3 selected / 3 executed | 2 selected / 2 executed |

Peak capability has clearly improved (≈ +9 nsucc on median peak, +8 on max peak); end_mean is depressed by the formal-run-specific late PBT churn (see caveats below).

## Caveats — read these before acting on the numbers

1. **Run is NOT finished.** Frames > 5B but max_iterations ≈ 38147 epochs is the actual stop condition; latest policy is at ~34.4–35.3k epochs. End-mean numbers will shift in the last ~10–15% epochs.
2. **end_mean ≪ peak_mean is dominated by PBT cross-copy churn**, not policy degradation. Of all tick-to-tick drops ≥1.0, ~70% (count-weighted, ~85% magnitude-weighted) are attributable to PBT_RESTART. Pure tolerance-shrink ticks contribute only ~−2 mean each.
3. **success_tolerance has shrunk from 0.075 → ~0.024**, so end-stage `nsucc_mean` is on a stricter scale than mid-run peak. Cross-run end-mean comparisons should be done at fixed eval cap via `eval_cap_sweep.py`, not on training-time `nsucc_mean`.
4. **p0 end_mean = 0.18 is a snapshot artifact** (latest tick is right after a cross-copy). Post-restart peak for p0 was ~13.8 at frame ~5150M; the next ~50–100M will tell whether p0 recovers or whether a recovery-restart is triggered.
5. **success_aware_r_meta ranking + tiered source selection** keeps replacing live policies whenever a strict_stable_live source is available, so late-stage churn is by design — it is not the population collapsing.

## Watchlist before final report

- **p0** (end_mean 0.18): expect either (a) self-recovery in next few ticks, (b) one more cross-copy.
- **p3** (peak 30.5, end ~7): chronically below post-restart peak; check whether retention recovers in last 10% epochs.
- **p5** (highest end_mean among 6, 16.2 / peak 27.8): currently the best 'live' policy in the population.
- Watch `recovery_audit.jsonl` for new entries (currently 2 executed, 0 skipped).

## Suggested actions after run finishes

1. Re-run `python results/generate_smoke_report_p2.py <RUN>` for the final acceptance numbers.
2. `python results/eval_cap_sweep.py --run-dir <RUN> --caps 50 100 200` to decouple end-mean from training tolerance.
3. `python results/eval_nsucc_percentile.py --run-dir <RUN>` to log per-policy p50/p75/p90/p99.
4. Decide on Track C (bayesian-no-h3 ablation) using formal peak/end metrics as gating criterion.

---

*This snapshot was generated mid-run. Treat numbers as provisional; the authoritative version will overwrite `SMOKE_P2_REPORT.md` after 6/6 policies hit max_iterations.*
