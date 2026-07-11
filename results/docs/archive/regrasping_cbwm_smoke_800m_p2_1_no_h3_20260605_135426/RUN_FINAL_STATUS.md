# Run Final Status — no-h3 Full PBT Smoke (STOPPED_BY_USER)

**Run ID:** `regrasping_cbwm_smoke_800m_p2_1_no_h3_20260605_135426`
**Status:** STOPPED_BY_USER (5/6 policies completed, p1 partial)
**Stop time:** 2026-06-05 22:01 CST

## Summary

This smoke validates the no-h3 + fixed `value_imag_h1` configuration with
P2.1 stability patches, staggered launch, per-policy XDG isolation, and the
startup watchdog. Run was stopped before p1 attempt 6 finished because the
smoke conclusion was already reached and p1 was a clear outlier. All other
policies reached 6112/6112 epochs.

## Pointers

- `SMOKE_P2_REPORT.md` — full acceptance report (re-generated with parser fixes).
- `P1_OUTLIER_AUDIT.md` — diagnostic timeline of why p1 is an outlier.
- `RUN_STOPPED_BY_USER.json` — machine-readable stop record + dual aggregates.
- `dexpbt_8gpu_ws/recovery_audit.jsonl` — 6 recovery records (3 selected, 3 executed).
- `launch_audit.jsonl` — staggered launch + first-epoch timestamps.

## Population (completed-only: p0, p2, p3, p4, p5)

| metric        | value |
|---------------|-------|
| mean end_mean | 12.70 |
| median end_mean | 9.10 |
| max end_mean  | 22.08 (p5) |
| min retention | 76.52% (p2) |
| median retention | 94.35% |

## Population (all observed, p1 partial @ 440M)

| metric        | value |
|---------------|-------|
| mean end_mean | 12.62 |
| median end_mean | 10.67 |

## PBT replacement evidence

- cross_restart total (lifetime max): **8**
- recovery_restart total (lifetime max): **4**
- cross_restart latest-tick aggregate: 5
- recovery_restart latest-tick aggregate: 3
- self_restart total: 0
- startup_hang total: 0

> Lifetime max counts every unique cross/recovery event observed in any
> training-curve tick. Latest-tick aggregate sums each policy's literal
> final cumulative counter, which can be lower for partial p1 due to
> interleaved curve rows from multiple writers.

## Headline conclusions

- no-h3 fixed `value_imag_h1` smoke confirms stable RL/PBT learning in completed workers.
- PBT replacement works: cross=5, recovery=3.
- p5 recovery burst (7.5 → 22.0 over ~140M frames after 620M restart) is the strongest signal.
- p1 is an outlier requiring a separate bad-seed / restart-integrity audit; see `P1_OUTLIER_AUDIT.md`.

## Next phase

- 800M repeat smoke: `results/start_regrasping_fixed_h1_p2_1_smoke_800m.sh`
- 5B formal: `results/start_regrasping_fixed_h1_p2_1_formal_5b.sh`
- Roadmap: `results/ABLATION_ROADMAP.md`
