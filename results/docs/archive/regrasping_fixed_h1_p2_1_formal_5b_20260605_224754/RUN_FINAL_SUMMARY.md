# Formal 5B Run — Final Summary

**Run ID:** `regrasping_fixed_h1_p2_1_formal_5b_20260605_224754`
**Status:** **COMPLETED** (6/6 policies hit max_iterations 38147)
**Wall time:** ~21.7 hours (2026-06-05 22:47 → 2026-06-06 20:27 CST)
**Config:** fixed `value_imag_h1`, no h3, no CF, P2.1 stability, 6 policies on GPU 0–5, base seed 42

## Top-line numbers (final tick)

| Metric | Value |
|--------|-------|
| Population peak_mean median | **31.19** |
| Population peak_mean max    | **32.95** (p1 @ 2700M) |
| Population end_mean median  | 12.88 |
| Population end_mean max     | 23.65 (p1) |
| Cross-restart total | 182 |
| Recovery-restart total | 2 (both via best_stable_archive) |
| Startup hangs | 0 |

## Per-policy final

| pid | end_mean | peak_mean | retention | cross | recovery | tier (last) |
|-----|----------|-----------|-----------|-------|----------|-------------|
| p0 | 2.95   | 30.29 |  10% | 32 | 1 | bottom |
| p1 | **23.65** | **32.95** | 72% | 28 | 0 | bottom |
| p2 | 12.27  | 31.87 | 39% | 25 | 1 | bottom |
| p3 | 18.94  | 30.51 | 62% | 29 | 0 | bottom |
| p4 | 13.49  | 32.31 | 42% | 32 | 0 | bottom |
| p5 | 0.71   | 27.75 |   3% | 36 | 0 | bottom |

> **Caveat on end_mean:** p0 (2.95) and p5 (0.71) hit their final tick right after a PBT cross-copy and are in the post-restart transient. Live performance for these policies in the previous 50–100 M frames was 13–18. Use `eval_cap_sweep.py` for a fair fixed-tolerance comparison rather than reading the final tick.

## Acceptance verdict

| Group | Result | Note |
|-------|--------|------|
| P2 (S1–S8) | 7/8 PASS, 1 false-positive FAIL | See S4 note below |
| PBT D-criteria (D1–D4) | **4/4 PASS** | All 6 started, 0 hangs, diverse curves, 184 replacements |
| P1 (A1–C2) | **7/7 PASS** | Peak ≥3.5 met by all, end ≥2 met by p1/p2/p3/p4, no monotonic collapse |

### S4 FAIL is a parser false positive

Report flags S4 (`Hard collapse tolerance rollback`) as FAIL with `hard_collapse=True forced_tol=False`.

Raw curve inspection shows **every hard_collapse FIRST trigger has forced_tol active and rollback_active=True**:

| Policy | First hard_collapse @ frames(M) | nsucc_mean | forced_tol | rollback |
|--------|------------------------------|-----------|------------|----------|
| p1 | 260  | 0.00  | **0.0750** | active |
| p2 | 280  | 0.00  | **0.0750** | active |
| p2 | 4660 | 3.51  | **0.0219** | active |
| p3 | 3960 | 11.22 | **0.0255** | active |
| p4 | 2940 | 9.05  | **0.0298** | active |
| p5 | 3340 | 9.61  | **0.0283** | active |
| p5 | 4680 | 12.13 | **0.0243** | active |

The subsequent ticks marked `hard_collapse=True forced_tol=0` are **cooldown-blocked** (`rollback_reason=cooldown_blocked`), which is correct P2.1 behavior — we don't want forced_tol firing every tick during cooldown.

The S4 judge in `generate_smoke_report_p2.py` currently checks `any forced_tol` against `any hard_collapse`, treating cooldown-block ticks as failures. This is a **parser logic bug**; the actual stability mechanism worked.

**Real S4 status:** PASS for all 6 policies on first-trigger response.

## Comparison vs prior runs

| Metric | no-h3 800 M smoke (5/6 completed) | Formal 5 B (this run) |
|--------|-----------------------------------|------------------------|
| peak_mean median | ~22.5 | **31.19** (+39 %) |
| peak_mean max    | ~25.0 | **32.95** (+32 %) |
| end_mean median  | ~9.1  | 12.88 |
| cross_restart    | ~30   | 182 |

Peak capability improved substantially. End_mean appears similar despite higher peak because:
1. tolerance shrunk 4× harder than smoke (0.075 → 0.020 vs 0.075 → 0.040).
2. Late-stage PBT replacement count (182 vs 30) means many policies are mid-recovery at the final tick.

## Saturation finding

Population peak-so-far (best `nsucc_mean` ever seen by any live policy):

```
frames(M) | peak_so_far
   1000   |   21.86
   1500   |   28.03
   2000   |   32.31    <- 95 % of final peak hit by 2 B
   2500   |   32.31
   3000   |   32.95    <- final peak
   4000   |   32.95
   5000   |   32.95
```

**Conclusion: ~3 B frames was sufficient for this configuration.** The last 2 B mostly cycled the population (cross-copy churn) without raising the upper envelope.

## Recovery audit

2 archive recoveries executed:

| Target → Source | Frame (M) | Outcome |
|------------------|-----------|---------|
| p2 ← p0 best_stable_archive | 2380 | **+17.7 nsucc_mean over +50 M** (recovered from 11.25 to 28.97) |
| p0 ← p3 best_stable_archive | 4120 | −4.7 over +50 M (regression — p0 in chronic difficulty) |

## Cross-restart effectiveness

171 cross-restarts where +50 M trace was available, computed via `cross_restart_count` delta:

- **Average Δ nsucc_mean = +4.06**
- **Median Δ = +3.64**
- **78 % positive**, 14/171 (8 %) below −5

Per-policy:

| pid | n cross | avg Δ | positive % |
|-----|---------|-------|------------|
| p0 | 29 | +3.63 | 69 % |
| p1 | 26 | +6.85 | 92 % |
| p2 | 22 | +3.45 | 73 % |
| p3 | 28 | +4.40 | 86 % |
| p4 | 32 | +2.60 | 69 % |
| p5 | 34 | +3.79 | 79 % |

PBT replacement is doing what it should — moving means upward on average.

## Eval target picks (for `eval_cap_sweep.py`)

Best 3-tick rolling window of `nsucc_mean` per policy:

| pid | best 3-tick avg | frames (M) | tolerance at peak |
|-----|-----------------|------------|--------------------|
| **p1** | **32.69** | **2740** | 0.0385 |
| p2 | 30.81 | 2620 | 0.0385 |
| p0 | 30.00 | 2160 | 0.0385 |
| p4 | 29.75 | 1660 | 0.0498 |
| p3 | 29.64 | 2700 | 0.0347 |
| p5 | 26.89 | 2280 | 0.0385 |

p1 @ 2740M is the strongest training-time peak; will compare against final ckpt under fixed eval tolerance.

## What this validates / unblocks

- ✅ no-h3 fixed-h1 + P2.1 stability is **stable for full 5 B** without hangs / OOM.
- ✅ PBT cross-copy is empirically positive (78 %, +4.06 avg).
- ✅ Recovery-from-archive works (p2 case: +17.7).
- ✅ Curriculum mastery shrink + freeze + early collapse all triggering correctly.
- ❌ Going from 3 B → 5 B did **not** lift peak; 3 B is the natural horizon for this config.

## Next steps

1. **Eval cap sweep** (running now): caps {50, 100, 200} × 256 episodes per policy.
2. **Eval percentile**: cross-policy p50/p75/p90/p99 nsucc.
3. **Track C (bayesian-no-h3)**: replicate at 3 B horizon, compare posterior updates on `{none, aux_mid, value_imag_h1}` vs the fixed-h1 baseline here.
4. **Optional parser fix**: tighten S4 judge to only count first-trigger ticks (drop cooldown-blocked aftermath).

## Pointers

- `SMOKE_P2_REPORT.md` — full P2/P1 acceptance + raw tables (final, regenerated 20:28).
- `INTERMEDIATE_FORMAL_REPORT.md` — mid-run snapshot (kept for history).
- `FORMAL_DEEP_ANALYSIS.md` — trajectory / cross-restart / curriculum deep dive.
- `dexpbt_8gpu_ws/recovery_audit.jsonl` — 4 entries (2 selected + 2 executed).
- `dexpbt_8gpu_ws/metrics/training_curve_policy_*.jsonl` — full training curves.
- `eval_cap_sweep/cap_*.csv` — to be filled by post-run eval.
