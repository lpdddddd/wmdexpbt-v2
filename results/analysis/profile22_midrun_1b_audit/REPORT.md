# Profile 22 Mid-Run Audit (~1B): `20260714_092846` vs Jul12

**NOW:** `regrasping_cser_dexpbt_progress_refine_5b_20260714_092846`  
**JUL12:** `regrasping_cser_dexpbt_progress_refine_5b_20260712_114254`  
**Scope:** readonly (JSONL / sidecar / TB). Training processes left running (`train_procs=6`, master alive).  
**Artifacts:** `audit_midrun_1b.py`, `audit_midrun_raw.json`

Tip at audit time (~0.96–1.08B): best/med/mean nsucc ≈ **37.3 / 29.2 / 28.1**, p4 tip, p2 still weak (~17.5 @ tol 0.061).

---

## 1. Same-frame reproduction (exact curve ticks only)

| Frame | Run | n | best | med | mean | tol min/med |
|---|---|---|---|---|---|---|
| 200M | NOW | 6 | 8.23 | 0.02 | 1.44 | 0.075 / 0.075 |
| 200M | JUL12 | 6 | 7.03 | 0.01 | 2.01 | 0.075 / 0.075 |
| 400M | NOW | 6 | 24.48 | **16.47** | **15.02** | 0.063 / 0.071 |
| 400M | JUL12 | 6 | 25.28 | 6.60 | 10.14 | 0.061 / 0.073 |
| 600M | NOW | 6 | 28.41 | 24.59 | 20.97 | 0.054 / 0.060 |
| 600M | JUL12 | 6 | **34.92** | **27.32** | 23.15 | 0.051 / 0.064 |
| 800M | NOW | 6 | 29.03 | 26.02 | 24.28 | 0.042 / 0.050 |
| 800M | JUL12 | 6 | **39.47** | **31.72** | 26.48 | 0.041 / 0.051 |
| 1000M | NOW | **5*** | 37.90 | 23.97 | 26.44 | 0.036 / 0.041 |
| 1000M | JUL12 | 6 | 39.50 | **32.64** | 26.56 | 0.041 / 0.042 |

\*NOW p1 had no exact `env_frames==1000M` curve point yet (latest ≤1B curve row at 960M, nsucc≈31.6).

### Per-policy nsucc @ exact frames

**@800M**
| | p0 | p1 | p2 | p3 | p4 | p5 |
|---|---|---|---|---|---|---|
| NOW | 29.0 | 29.0 | **12.9** | 23.2 | 28.9 | 22.7 |
| JUL12 | 18.8 | **39.5** | **0.0** | 32.6 | 30.8 | 37.2 |

**@1000M**
| | p0 | p1 | p2 | p3 | p4 | p5 |
|---|---|---|---|---|---|---|
| NOW | 34.0 | *(960M: 31.6)* | 14.3 | 22.0 | **37.9** | 24.0 |
| JUL12 | 17.8 | **39.5** | **0.0** | 31.9 | 33.3 | 36.8 |

### Curriculum / shrink ≤1B

| | NOW shrinks (Σ) | JUL12 shrinks (Σ) | factors |
|---|---|---|---|
| population | **48** | **36** | both use 0.90 / 0.925 / 0.95 |
| tol @~1B | NOW min **0.034–0.036** (tip) | JUL12 min **0.041** | NOW slightly **faster** on leaders |

`r_meta` med: 200M≈0; 400M≈0.07–0.08; 600M≈0.41; 800M NOW 0.71 vs JUL 0.65; 1000M NOW 0.75 vs JUL **0.97** (JUL tip cleaner).

TB `adaptive_consolidation/curriculum_progress` late ≈ **0.31–0.39** on strong workers (still **exploration** stage &lt;0.50).

### Answers (reproduction)

1. **Fully reproduce Jul12?** **No.** Early (≤400M) NOW is **as good or better on median**; from **600M onward tip/best lag Jul12 by ~6–10 nsucc** at same frames.  
2. Leaders: NOW **p4/p0/p1**; laggard **p2** (recovering, not zero). JUL12 tip **p1/p5**, dead **p2=0**.  
3. Tolerance: **similar to slightly faster** on NOW leaders (min tol tighter by ~0.005–0.007 @1B).  
4. Gap source: mainly **capability / tip formation rate**, not PBT restarts (both **0 copies ≤1B**) and not a wildly different curriculum skeleton. Random seed/path + one chronic weak worker.  
5. **Healthy enough to continue to 5B?** **Yes** — 6/6 alive, no OOM/traceback in this audit, tip climbing, p2 not fully dead, mechanism pathology matches known P22 (not a new crash mode).

---

## 2. PBT utilization ≤ ~1B (NOW)

### Config (curve fields @≤1B)

```text
allow_mid_self_mutation = False   ✓
disable_self_restart     = True   ✓
profile_id = 22, name = cser_dexpbt_progress_refine
```

### Tick / tier

| Metric | NOW ≤1.05B |
|---|---|
| tick debug lines | 56,615 |
| `should_tick` | **125** (~21 / policy) |
| Stage labels on ticks | all `stage_exploration` (125) |
| Real skips | `initial_delay` 47334, `before_start_after` 9156 |
| Tier visits (curve ≤1B) | top 23 / mid 66 / bottom 36 |
| Longest bottom span | **p2: 640M**; others ≤80M |

### Replacement outcomes

| Event | Count ≤1.05B |
|---|---|
| Ordinary bottom cross-copy executed | **0** (`pbt_copy_events.jsonl` empty) |
| Adaptive rescue attempt / selected / executed / blocked | **0 / 0 / 0 / 0** (sidecars) |
| `weight_copy_count` / `restart_count` / `cross_restart` / `recovery_restart` | **0** all policies |
| Mid self-mutation path entered | **62** (`mid_self_mutation_count` sum) |
| Mid self-restart blocked | **62** (`self_restart_blocked_count` sum) |
| Self-restart executed | **0** |
| Source tier strict/relaxed/archive hits | **0 / 0 / 0** (counters) |
| Ordinary mutation executed | **0** |
| Restart transaction success/fail | **0 attempts** → N/A |

**Copy execution rate ≤1B:**  
\(N_{\mathrm{executed}}=0\) / bottom opportunities (proxy bottom curve pts **36**) = **0%**.

EIR bank broadcast ≤1B: **7 broadcast / 5 consume** (weak population coupling without weight copy).

### Verdict

**PBT weight propagation remains essentially inactive through 1B**, reproducing the Jul12 early-phase pattern (Jul12’s sole rescue was @1000M on that run; NOW has not fired one by tip ~1.08B either). Mid path is exercised then **hard-blocked** as configured.

---

## 3. WM snippet rerank

From **NOW TensorBoard** (p0/p1/p2; late window near tip):

| Tag | p0 | p1 | p2 |
|---|---|---|---|
| `snippet_wm_rerank/changed_frac` | **0** | **0** | **0** |
| `row_active_frac` | **0** | **0** | **0** |
| `row_fallback_frac` | **1** | **1** | **1** |
| `fallback_reason_id` | **2** | **2** | **11** |
| `fallback_frac` (series) | 1 | 1 | 1 |

Reason IDs (`snippet_wm_reranker.py`): **2 = belief_missing**, **11 = alpha_zero_or_disabled**.

### Verdict

**Yes — WM snippet rerank is hard-broken on the live path:** teachers are not reordered (`changed_frac=0`). Strong workers fail with **belief_missing**; weak p2 additionally shows **alpha_zero** and `no_compatible_frac=1`.

---

## 4. Snippet distillation: frequent but weak vs PPO

| Tag (TB late) | p0 | p1 | p2 |
|---|---|---|---|
| `elite_snippet/bank_size` last | 50k | 50k | 50k |
| `bank_size_mid` | 50k | 50k | 50k |
| `bank_size_hard/floor` | 0 | 0 | 0 |
| `progress_refine/active_frac` | **0.29** | **0.28** | **0** |
| `progress_refine/effective_sample_size` | **~1216** | **~1189** | 0 |
| `progress_refine/retrieval_valid_frac` | ~0.33 | ~0.32 | 0 |
| `snippet_distill/loss` | ~2.9e-5 | ~4.4e-5 | 0 |
| `losses/a_loss` | ~5.5e-3 | ~4.1e-3 | ~5.4e-3 |
| `eir/actor_distill_loss` | 0 | 0 | 0 |
| `snippet_retrieval/no_compatible_frac` | 0 | 0 | **1** |

Loss ratio (strong workers): snippet distill ≈ **100–200× smaller** than PPO `a_loss` (and `progress_refine/loss` saturates at **5e-5** cap-like level). Fallback actor distill inactive when snippet path is on.

### Verdict

**High-frequency gate activity on healthy workers (active_frac≈0.28–0.29, ESS≈1200, near 0.35 budget) but actor impact remains weak by loss scale.** p2: bank full but refine/distill **off** (compat/progress fail-closed).

---

## Bottom-line answers

| Question | Answer |
|---|---|
| Reproducing Jul12 @ same frames? | **Partial:** early OK/better med; **600–800M tip behind**; @1B best nearly catches, median still softer |
| PBT still near-dead? | **Yes — 0 weight copies / 0 rescue ≤1B** |
| WM rerank belief_missing? | **Yes (reason_id=2)**; changed_frac=0 |
| Distill frequent but weak? | **Yes** on p0/p1; loss ≪ PPO |
| Continue to 5B? | **Yes** — healthy run; expect Jul12-like path unless tip accelerates |

## Missing / caveats

- Exact per-tick bottom **target-gate reject taxonomy** still not in tick JSONL (`not observable`).  
- @1000M NOW curve incomplete (5/6). Prefer tip table for live status.  
- TB “late” window approximates ~1B for NOW tip; not a hard frame join to 1000M scalars.
