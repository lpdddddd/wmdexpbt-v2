# Profile24 ABC 5B — ~2B Joint Mid-Run Audit

**Audit window (local):** 2026-07-15 ~22:16–22:25 Asia/Shanghai  
**Mode:** read-only (no code/config/run mutation; no full SHA; no GPU checkpoint load; ≤8 min light sampling)  
**Prior baseline:** [`../profile24_abc_1p5b_audit/PROFILE24_ABC_1P5B_AUDIT.md`](../profile24_abc_1p5b_audit/PROFILE24_ABC_1P5B_AUDIT.md) (2026-07-15 ~11:22)  
**Rule:** claims without logs/JSONL/TB/manifest/code citations are marked **无法确认**. Design intent is never substituted for runtime evidence.

---

## 1. Executive Summary

Profile24 ABC (`…_222300`) is **still healthy after ~24h wall**, tip frames ≈ **1.72–2.08B**. Config/A/B/C remain effective (p2 via `profile_authoritative`). **No new PBT copy since 600M**; the sole parameter-level interaction remains **p2←p0 chronic**.

**Since 1.5B audit (material deltas):**

| Item | 1.5B audit | Now (~2B) |
|---|---|---|
| Spread (tip pop) | ~17.5 → later spiked ~22 | **~13.7** (min/med/max ≈ 24.5 / 28.9 / 38.3) |
| p3 | chronic-blocked, nsucc≈21 | Crashed to **~8.6** @1.68B under freeze, then **self-recovered to ~27.4 mid** **without** copy |
| New chronic copies | — | **0** |
| Ordinary committed copies | 0 | **0** (one log `action=copy` without commit) |
| B | working; p3 had foreign | p3 **late consumes all reject**; recovery **without new accepted foreign** |
| C | calib mostly fail overall | **After 1.5B calib_pass often 80–100%**; still `rerank_reason_id=-1`; rebuild-every-rollout |
| E2E FPS | ~20–24k/worker lifetime | **Recent 2h ~14–17k**; early intervals ~36k → late ~14–16k (**slowdown confirmed**) |
| Profile22 | mid comparison possible | **Stalled ~1.12–1.48B, tips aged ~25h, nearly dead** → **exact 1.8B/2.0B P22 points N/A** |

**Recommendation: `CONTINUE_WITH_PERFORMANCE_CAUTION`** toward **3B event gate**, not blind 5B.  
**P0: 0 · P1: 0 · P2: several · P3: telemetry/perf.**

---

## 2. Run Identity and Config Verification

| Field | Value | Evidence |
|---|---|---|
| Run dir | `/root/autodl-tmp/wmdexpbt/results/runs/regrasping_cser_dexpbt_progress_refine_abc_5b_20260714_222300` | `LAUNCH_IDENTITY.txt` |
| Launch commit | `c890ccda63193c2b49da4549dbe451b55b39dbb2` | identity + `RUN_CONFIG.txt` |
| Repo HEAD | `62d4dd09fb4db9d1e3347589bcef7c27495ee29b` | `git rev-parse` (docs after launch) |
| Launcher | `results/launchers/start_regrasping_cser_dexpbt_progress_refine_abc_5b.sh` | `RUN_CONFIG.txt` |
| Master | PID `859798`, etime ~24h | `ps` / `nohup.pid` |
| Profile22 | `…/regrasping_cser_dexpbt_progress_refine_5b_20260714_092846` | stalled; see §11 |

**ABC / disabled features (RUN_CONFIG + live argv):** A/B/C on; mid-mutation / AWR / residual / RSSM-MPPI off; `target_population_frames=5000000000`.

**p2 restart:** attempt_2 argv still omits ABC CLI flags; startup log still shows `profile_authoritative` forcing A/B/C True (`policy_02/attempt_2_*.log`). **No worker with ABC off under profile authority.** No competing DexPBT train for other runs observed on GPUs 0–5 (`other_trainish` empty in audit dump).

### Worker identity table

| worker | frame (tip) | PID | GPU | attempt | lineage | profile | A | B | C | disabled OK | alive |
|---|---:|---:|---|---|---|---|---|---|---|---|---|
| 0 | 2.08B | 859911 | 0 | attempt_1 | `39283193-…` | abc/24 | CLI✓ | ✓ | ✓ | ✓ | ✓ |
| 1 | 1.72B | 860928 | 1 | attempt_1 | `8c286e41-…` | abc/24 | ✓ | ✓ | ✓ | ✓ | ✓ |
| 2 | 2.00B | 903198 | 2 | **attempt_2** | `ffdde91b-…` | abc/24 | auth✓ | auth✓ | auth✓ | ✓ (via profile) | ✓ |
| 3 | 1.84B | 863342 | 3 | attempt_1 | `38da4a27-…` | abc/24 | ✓ | ✓ | ✓ | ✓ | ✓ |
| 4 | 1.88B | 864297 | 4 | attempt_1 | `73c8a860-…` | abc/24 | ✓ | ✓ | ✓ | ✓ | ✓ |
| 5 | 1.84B | 865599 | 5 | attempt_1 | `b8047196-…` | abc/24 | ✓ | ✓ | ✓ | ✓ | ✓ |

Lineage/profile from tip `LATEST.commit` → `MANIFEST` / `meta.json` (size-matched `model.pth` vs sibling `.pth`; **full SHA not recomputed**).

---

## 3. Worker Health

Recent ~2MB log scan per attempt: **no OOM / CUDA / NaN / Traceback**. Log ages seconds. Disk `/root/autodl-tmp` ~35% used (339G free).

| worker | frame | last_update | restart | OOM | NaN | ckpt err | artifact | verdict |
|---|---:|---|---|---|---|---|---|---|
| 0 | 2.08B | ~seconds | 0 | N | N | none seen | ok | healthy |
| 1 | 1.72B | ~seconds | 0 | N | N | none | ok | healthy |
| 2 | 2.00B | ~seconds | **1 intentional chronic** | N | N | none | ok | healthy |
| 3 | 1.84B | ~seconds | 0 | N | N | none | ok | healthy (post-collapse freeze still true on tip) |
| 4 | 1.88B | ~seconds | 0 | N | N | none | ok | healthy |
| 5 | 1.84B | ~seconds | 0 | N | N | none | ok | healthy |

Crash/supervisor loops: **not observed**. Only intentional chronic relaunch (p2).

---

## 4. Changes Since the 1.5B Audit

1. **Frames advanced** ~0.4–0.6B/worker; wall +~12h.  
2. **No second copy**; `pbt_copy_events.jsonl` still 1 row; `pbt_chronic_source_health.jsonl` still only under `policy_02/` (2 lines).  
3. **p3:** long bottom → sharp collapse (nsucc 26→8.6) after `promising_protect` expiry → freeze → climb to 27.4 mid.  
4. **Spread** worsened mid-evening (~22) then **narrowed to ~13.7**.  
5. **B event count** ~26/25 → **42/42** broadcast/consume; p3’s post-1.28B consumes **all rejected**.  
6. **C events** ~74k → **~107k**; post-1.5B calibration much better than lifelong average.  
7. **Throughput regression** early~36k → late~14–16k interval FPS.  
8. **Profile22** no longer a live same-frame peer at 2B (stalled ~25h).

---

## 5. A Chronic-PBT Audit

### 5.1 Commit-level counts (hard evidence)

| Class | Count | Evidence |
|---|---:|---|
| Ordinary **committed** copies | **0** | `metrics/pbt_copy_events.jsonl` (only chronic row) |
| Ordinary log `action=copy` | **1** | `policy_03/...log`: `tier=strict_stable_live action=copy` — **no** matching restart / copy JSONL → **selection near-miss / aborted commit** |
| Chronic **successful** copies | **1** | same JSONL; p2←p0 @600M |
| New chronic since 1.5B | **0** | JSONL mtime/content unchanged in substance |

### 5.2 Funnel (population; counts)

Sources: published curves + `chronic_bottom/*.json` + TB tags (1.5B audit + presence of tags) + health JSONL. Fine-grained per-tick eligibility reasons after 1.5B: **largely 无法确认** without fresh full TB decode (slow; not completed for all workers this round).

| Stage | Count | Notes |
|---|---:|---|
| bottom observation (curve points) | many (ongoing) | `training_curve_*.jsonl` `tier=bottom` |
| duration≥200M **and** ticks≥5 (state or TB) | ≥1 historical (p2@600M); **p3 reached 9 ticks / 320M** then left bottom | `chronic_bottom/003.json` history @20:27 audit; now cleared |
| score-gap≥0.15 while bottom | frequent on tip reconstruction for p3/p5 | tip `pbt_ranking_score` gaps |
| chronic candidate armed (`candidate=1`) | **1** known (p2@600M); **0 new** known | TB 1.5B; health file still only p2 |
| source candidates / health / snapshot | **1 tick**, 2 validations (miss→hit) | `policy_02/pbt_chronic_source_health.jsonl` |
| copy committed | **1** | copy JSONL |
| target loaded / improved | **1 / 1** | attempt_2 + nsucc 2.64→23.7→38 |

### 5.3 Answers to A checklist (1–20)

1. Ordinary **attempts** (true reject journal): **无法确认**; log shows **1** `action=copy` without commit.  
2. Ordinary copies still 0: no committed event; stuck/promising/adaptive gates likely — full attempt telemetry **缺失**.  
3–6. Chronic eligibility/candidate: **1** completed path @600M; **still only that copy**.  
7. Near-miss: **p3** (duration+gap OK while protected / later freeze); **p5** briefly met 6 ticks/200M earlier then span reset.  
8. p1 now: enter 1.64B, **3 ticks / 80M** — need **+2 ticks (~40M)** and **+120M** duration → earliest ~**1.84B+** if uninterrupted.  
9. p5: enter 1.80B, **2 ticks / 40M** — earliest ~**1.88B+** duration/ticks if holds.  
10. p3 exited bottom @1.84B after nsucc climb — **no copy**.  
11–12. See §6 (attribution); gate prevented rescue during protect; whether “correct” vs “too strict” is **P2 design**, not confirmed bug.  
13–16. Simultaneous ticks: historical p3 showed stretches with duration OK + gap OK but protect True; later gap/duration OK with freeze — **candidate never observed new**. Exact same-tick conjunction without TB: **部分无法确认**.  
17. `max_copies_per_lineage=1` + 400M cooldown **strong but unused** post-p2 (p3 other lineage).  
18. Snapshot false rejects beyond p2 path: **no evidence** (selection never entered).  
19. p2 post-copy: supervisor reload ckpt + WM path + tolerance resume in argv; optimizer from copied ckpt — **as launched**; normalizer details **未单独审计**.  
20. p2 recovery: tip shows jump **exactly after 600M publish** with restart flags — **strong temporal association**.

### 5.4 Eligibility vs bottom objective (design tension — confirmed by events)

Code: `pbt_chronic_bottom.evaluate_chronic_eligibility` uses **`tier=="bottom"`** plus **`ranked_scores` gap** (`pbt_ranking_score` / success-aware), not raw nsucc (`pbt_chronic_bottom.py`).

**Examples:**

- **p2@600M:** bottom + huge rank gap (~0.69) → candidate → copy.  
- **p3@1.24B (1.5B audit):** bottom, nsucc weak, **rank gap &lt;0.15** → no candidate.  
- **p3@1.68–1.76B:** nsucc 8–15, gap often OK, but **prior promising_protect** then **no new health file** → no copy.  
- **p5 tip:** gap can OK while duration/ticks incomplete after span reset.

This **can** leave low-nsucc workers unrescued when ranking stays competitive — **confirmed pattern**, not proof of hash bug.

---

## 6. p3 Self-Recovery Analysis

### Timeline (curve `training_curve_policy_03.jsonl`)

| frame | nsucc | tier | protect | freeze | rank gap (recon) |
|---:|---:|---|---|---|---:|
| 1.56B | 25.8 | bottom | True | F | ~0.41 |
| 1.60B | 26.1 | bottom | True | F | ~0.08 |
| 1.64B | 23.3 | bottom | True | F | ~0.51 |
| 1.68B | **8.6** | bottom | False | **True** | ~0.54 |
| 1.72B | 13.5 | bottom | F | T | ~0.61 |
| 1.76B | 14.9 | bottom | F | T | ~0.57 |
| 1.80B | 20.4 | bottom | F | T | ~0.01 |
| 1.84B | **27.4** | **mid** | F | T | ~0.38 |

### Hypotheses

| Hypothesis | Support | Against |
|---|---|---|
| 1. PPO natural recovery | Climb 8.6→27 with **no** copy / restart | — strong default |
| 2. B foreign snippets | Earlier accepts @≤1.16B | **All consumes @1.28–1.76B rejected** (`eir_broadcast_events.jsonl`); foreign_after only decaying |
| 3. C belief rerank | Builds succeed; recovery-window `calib_pass` 100%, `alpha_wm`~0.45 | `rerank_reason_id` always **-1**; no accept/use counters |
| 4. Curriculum/tol | tol **unchanged** 0.0227 across crash | no shrink in window |
| 5. collapse_freeze | Freeze on during climb (blocks shrink) | Does not inject parameters; may stabilize |
| 6. Other | — | — |
| 7. Cannot attribute | **Selected** | Causal ID of B/C **not supported** |

**Verdict:** recovery is **most consistent with PPO/self-dynamics under freeze**, **not** B (new accepted foreign absent) and **not verified** for C.

---

## 7. B Tolerance-Aware Bank Audit

`eir_broadcast_events.jsonl`: broadcast **42**, consume **42** (was 26/25 @1.5B).

### Per-worker (lifetime consume summary)

| worker | consumes | accepted | rejected | incoming* | compatible* | ratio* | last local/foreign/bank | notes |
|---|---:|---:|---:|---:|---:|---:|---|---|
| 0 | 7 | 7 | 0 | — | — | ~0.62 | 0 / 50k / 50k | accepts |
| 1 | 8 | 3 | 5 | — | — | **0.00** on diag rejects | starving late | tight tol |
| 2 | 5 | 2 | 3 | | | ~0.20 | | |
| 3 | 10 | 5 | 5 | | | ~0.30 | foreign decaying | **no accept after 1.16B** |
| 4 | 2 | 2 | 0 | | | 0.5 | | few events; tight ~~0.014 |
| 5 | 10 | 5 | 5 | | | ~0.3 | | |

\*ratios from diagnostics-bearing events; early rows lack bank_diagnostics.

### Funnel

foreign broadcast → received → compatible → merged → retained → **sampled/scored/refined/used**  
Counts exist through merge; **sampled→used: 无法确认** (no use counter).

**Judgments:** B **pipeline works**; for p3 recovery window it **did not deliver new compatible foreign**. Preserve-local 0.20: fields show local/foreign splits but **exact fraction enforcement 未逐笔验证**. I/O: payload files exist; audit-window proc write ~0.1 MB/s — **not** current sampler bottleneck.

---

## 8. C Belief-Cache Audit

`eir_belief_events.jsonl` ≈ **107k** rows (+~33k since 1.5B).

Code (`train.py` ~3923–3994): whenever `need_belief`, it **always** `build_rssm_belief_cache(...)` and sets `belief_build_attempted=True`; `belief_cache_valid` := built object valid. **There is no hit/miss lookup counter in telemetry.** Lifetime “100% build_attempted” is **consistent with rebuild-every-need**, not with LRU hits.

| Metric | Finding |
|---|---|
| hit/miss | **无法确认** (instrumentation = build outcome) |
| After 1.5B calib_pass | p0 0.92, p1 0.80, p2–p4 **1.0**, p5 0.86 (improved vs early era) |
| alpha≈0 rate after 1.5B | often **~0** (unlike early p0-heavy zero era) |
| latency | still ~6–7 ms p50 class |
| shape/device in JSONL | still empty often |
| rerank_changed | **0 evidence** (`rerank_reason_id=-1`) |
| refinement accept | **无法确认** |

### Funnel counts

| Stage | Status |
|---|---|
| need belief | ~107k True |
| cache lookup hit/miss | **无法确认** |
| build / valid | ~100% of need |
| calibration pass | improved late; lifelong mix |
| nonzero alpha | improved late |
| rerank changed / refine / used | **无法确认** |

**C = constructs beliefs + logs; decision impact unproven.**

---

## 9. Snapshot Validation Cache

Still **only** the p2@600M path: 1 miss (full hash ~92 ms) + 1 hit (~1.4 ms).  
`pbt_chronic_source_health.jsonl` unchanged (2 lines).  
No new chronic selection ⇒ **no new cache telemetry** (not instrumentation death).

**Cannot explain multi-hour slowdown** (≪0.1% wall).

---

## 10. Overall PBT Utilization

| Question | Answer | Tier |
|---|---|---|
| Parameter-level cross-worker copies to ~2B | **1** | confirmed |
| PBT param contribution | sparse safety-net | confirmed |
| Ordinary PBT | effectively idle commits | confirmed |
| Primary cross-worker channel | **bank broadcast** (dozens of merges) | confirmed |
| ≈ 6× independent PPO + rare copy + bank? | **largely yes** | strong suspicion |
| Cost vs benefit of 6 GPUs | diversity used weakly for params | strong suspicion |
| Bottleneck class | eligibility / protect / ranking-objective mismatch + scarce ordinary stuck triggers | mix confirmed/suspect |

---

## 11. Profile22 vs Profile24 Same-Frame Comparison

**Critical confound:** Profile22 tips frozen ≈ **1.12–1.48B**, timestamps aged **~25h**, `alive` train under that root ≈1. **Exact 1.8B / 2.0B P22 rows = N/A** (Δ≫40M).

| Target | ABC (≤40M) | P22 (≤40M) |
|---|---|---|
| 500M | available both (historical) | available |
| 1.0B | available both | available |
| 1.5B | ABC n=6 med≈30.7 worst≈26.1 | P22 **only p2** within 40M |
| 1.8B | ABC n=5 worst≈20.4 (p3 dip) | **N/A** |
| 2.0B | ABC only p0/p2 exact | **N/A** |

**Fair AUC 0.5B→1.12B (all 6 within 20M grid):**

| | med | worst | best | spread |
|---|---:|---:|---:|---:|
| ABC | 27.47 | **17.27** | 31.68 | 14.41 |
| P22 | 26.34 | **9.05** | 31.26 | 22.21 |

ABC **better worst-AUC / lower spread-AUC to 1.12B**, largely from **avoiding P22’s deep early p2 hole** (helped by the 600M chronic copy).  
**Cannot claim ABC superior at 2.0B vs live P22** — baseline not there.

After 1.5B: ABC worst improved from crisis (~8.6) to ~24.5; spread narrowed vs evening peak — **intra-run improvement**, not vs P22@2B.

---

## 12. End-to-End Throughput Analysis

From curve `timestamp` deltas (`training_curve_*.jsonl`):

| worker | tip | total FPS | 6h | 2h | 0.5h | int dt p50/p95 (s) | ETA 3B (h) | ETA 5B (h) |
|---|---:|---:|---:|---:|---:|---|---:|---:|
| 0 | 2.08B | 23747 | 14032 | 14213 | 14369 | 1103 / 2867 | ~18 | ~57 |
| 1 | 1.72B | 18893 | 15018 | 14971 | 14921 | 2643 / 3000 | ~24 | ~61 |
| 2 | 2.00B | 22334 | 14558 | 14533 | 14596 | 1190 / 2871 | ~19 | ~57 |
| 3 | 1.84B | 20687 | 15163 | 15044 | 15081 | 2565 / 2704 | ~21 | ~58 |
| 4 | 1.88B | 21186 | 16131 | 16754 | 16643 | 2353 / 2703 | ~19 | ~52 |
| 5 | 1.84B | 20456 | 14493 | 14476 | 14378 | 1622 / 2836 | ~22 | ~61 |

Logs still show `fps step and policy inference` **~59–72k** — that is **subphase**, not E2E. Lifetime interval FPS early **~36k** vs late **~14–16k**.

**Skew:** launch stagger 180s×workers + p2 chronic downtime + persistent rate gaps (p0 fastest).

**Where time goes (evidence tiers):**

| Component | Role in slowdown | Tier |
|---|---|---|
| Snapshot cache / chronic | negligible | ruled out as main |
| Belief ~6 ms × ~100k | ~10 min total | ruled out as **sole** cause |
| GPU util mean ~8–11% in sample | under-fed GPU / wait | **likely bottleneck** |
| CPU ~240–260% | not single-core stuck | confirmed sampling |
| Disk write ~0.1 MB/s in sample | not saturating now | sampling window |
| Late-phase WM/refine/curriculum | coincides with FPS drop | **likely** |
| Checkpoint every ~40M (~181 MB×6) | adds pauses | **likely contributor** |
| Bank merge | rare vs steps | minor |
| Sync / Isaac overhead | **无法确认** fraction | unknown |

---

## 13. Resource Sampling

~**5.4 min**, ~3 s cadence, `/tmp/abc_2b_resource_sample.jsonl` (n≈109).

| GPU | util mean / p95 | low&lt;5% frac | power mean (W) |
|---|---|---|---|
| 0–5 | **~7.6–10.7 / ~17–22** | **0.10–0.33** | ~83–100 |

Train RSS ~10–13 GB class; CPU% ~243–259; load from `/proc/loadavg` recorded in samples. Swap not stressed. **No evidence of foreign GPU competitors.**

**Suspected bottleneck:** end-to-end pipeline leaving GPUs **idle often**, not raw FLOPs saturation.

---

## 14. ABC Overhead Breakdown

| Piece | Estimate | % wall (~24h) | Status |
|---|---|---|---|
| A eligibility/copy | one restart; hash 0.09 s | ≪0.1% | confirmed tiny |
| B broadcast I/O | tens of merges; payloads up to ~130 MB class | small–moderate spikes | likely minor average |
| C belief builds | ~107k × ~6.5 ms ≈ **12 min** | ~0.8% | confirmed modest |
| C JSONL/TB line writes | every belief event | unknown extra fsync cost | suspected minor–moderate |
| Checkpoint I/O | ~40–48 publishes/worker × 181 MB | minutes–tens of minutes | likely |
| PPO+env | dominant | — | confirmed |

**Main slowdown is not “ABC micro-ops alone”** but **phase-wise training/env efficiency drop**.

---

## 15. Method-Level Limitations

1. **A works but is extremely sparse**; ranking-gap vs nsucc-bottom mismatch.  
2. **Ordinary PBT param share ≈0**.  
3. **B merges, use unproven**; can starve tight-tol workers.  
4. **C rebuilds beliefs; rerank effect unproven**.  
5. Population gains vs P22 (fair early AUC) largely from **one rescue + avoiding early hole**, not dense ABC coupling.  
6. Engineering complexity ≫ measured param-copy benefit so far.

---

## 16. Confirmed / Suspected / Unknown

**Confirmed:** healthy run; 1 chronic copy only; p3 self-recover without new B accepts; FPS early→late drop; GPU under-utilized in sample; P22 stalled; belief rebuild semantics; snapshot cache only on p2 path.

**Suspected:** ranking/protect gates block needed chronic rescues; late-phase WM/refine drives FPS drop; 6-GPU PBT underused for parameters.

**Unknown:** ordinary attempt reject taxonomy; belief hit cache; snippet use after merge; precise % time in Isaac vs PPO vs WM; whether C changed any action.

---

## 17. P0 / P1 / P2 / P3 Issues

### P0
None.

### P1
None confirmed.

### P2
1. Chronic eligibility (rank gap + promising protect) missed p3’s severe dip.  
2. Ordinary PBT commit rate 0 through ~2B.  
3. B accepted-foreign absent in p3 recovery window.  
4. C decision impact unproven; always-rebuild cost.  
5. E2E FPS regression ~36k→15k.

### P3
1. Missing chronic eligibility JSONL per tick.  
2. Belief “cache” telemetry ≠ hit/miss.  
3. Restart argv drops ABC flags (mitigated).  
4. P22 baseline not maintained to 2B for comparison.

**None require stop.**

---

## 18. Continue / Stop Decision

### `CONTINUE_WITH_PERFORMANCE_CAUTION`

- Continue at least to **~3B** under event gates.  
- **Do not** assume 5B is automatically worth the wall-clock (~50–60h ETA remaining per worker at current 2h FPS) without a 3B joint check.  
- Prefer **CONTINUE_TO_3B** as the operational horizon; “full 5B” only if 3B shows (a) another chronic rescue or clear worst-AUC win, and (b) FPS not collapsing further.

### Next check triggers (event-first)

1. p1 or p5: continuous bottom **≥200M & ≥5 ticks**.  
2. Any new `chronic_candidate` / `pbt_chronic_source_health` / copy JSONL row.  
3. Any worker nsucc **&lt;20** for ≥3 published bottom tips.  
4. Tip set all ≥**3.0B**.  
5. OOM / NaN / restart loop.  
6. Median E2E FPS (2h) **&lt;12k**.

---

## 19. Ranked Next-Experiment Recommendations

*(Do not implement in this run.)*

| Rank | Change | Benefit | Risk | Evidence now |
|---:|---|---|---|---|
| 1 | Add JSONL: chronic eligibility reason every bottom tick + ordinary attempt rejects | Interpretability | low | funnel gaps |
| 2 | Chronic gate: `rank_gap OR nsucc_gap` (or sliding max gap over bottom span) | Rescue p3-like | medium | p3 events |
| 3 | Profile restart argv to always emit ABC CLI (not only profile_auth) | Robustness | low | p2 argv |
| 4 | Belief true hit/miss + rerank-delta counters; avoid rebuild if key unchanged | Perf + causal C | medium | train.py build path |
| 5 | Lower rank gap 0.15→~0.10 **or** shorten duration | More A | medium | near-misses |
| 6 | Soften B tol compatibility for bottom targets only | Feed weak workers | medium | p1/p3 rejects |
| 7 | Cut/calibrate C if rerank-delta≈0 after telemetry | Save ~1%+ unknown | medium | rerank_id=-1 |
| 8 | Keep config; observe to 3B | Zero risk | — | decision |
| 9 | Cooldown / max_copies changes | Low priority now | — | unused post-p2 |
| 10 | Kill ABC for speed only | Loses known A win | high | premature |

---

## Appendix — Key Paths

- Run / metrics / chronic_bottom / eir_broadcast / belief JSONL / copy JSONL as above  
- Prior audit: `results/analysis/profile24_abc_1p5b_audit/PROFILE24_ABC_1P5B_AUDIT.md`  
- Code: `pbt_chronic_bottom.py`, `pbt_observer.py`, `train.py` belief block, `eir_belief.py`  
- Sampling artifact: `/tmp/abc_2b_resource_sample.jsonl` (host temp; not run dir)  
- Metric dumps: `/tmp/abc_2b_metrics.json`, `/tmp/abc_2b_part1.json`
