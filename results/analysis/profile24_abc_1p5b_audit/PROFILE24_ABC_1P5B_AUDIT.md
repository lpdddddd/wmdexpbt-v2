# Profile24 ABC 5B — 1.5B Mid-Run Audit

**Audit time (UTC):** 2026-07-15T03:22–03:25Z (local ~11:22–11:25 Asia/Shanghai)  
**Mode:** read-only (no code/config/process/file mutation; run left alive)  
**Auditor note:** All counts below come from live processes, JSONL, TensorBoard event tags, checkpoint metadata, logs, or source code paths. Where evidence is missing, the report says **无法确认**.

---

## 1. Executive Summary

Profile24 ABC 5B (`…_222300`) is **healthy and still running** on GPUs 0–5. Six workers have tip population frames ≈ **1.12–1.56B** (live `pbt_tick_debug` slightly ahead of published curves). Config identity and Profile24/ABC flags verify **yes** (including after p2’s chronic restart via `profile_authoritative`).

**ABC status at ~1.5B:**

| Mechanism | Runtime evidence | Effect at ~1.5B |
|---|---|---|
| **A chronic-bottom PBT** | **1 successful chronic copy** (p2←p0 @ 600M) | Strong rescue: p2 nsucc **2.64→23.7→37.1** |
| **B tol-aware bank merge** | 26 broadcast / 25 consume; multiple accepted merges; some full rejects | Works; **p1 currently fails late consumes** (0 compatible); weak p3 **does** retain foreign snippets |
| **C belief cache** | ~73.9k `eir_belief_events`; build/success/cache_valid = 100% | Active; **calibration_pass mostly False**; `alpha_wm` often ~0; refinement funnel largely **无法确认** |
| Snapshot validation cache | Only exercised on the p2 chronic attempt: miss then hit; latency 92ms→1.4ms | Works when chronic selection runs; LRU entries=5, not unbounded in that tick |

Versus Profile22 at nearest same-frame points: **median nsucc similar**; **ABC worst worker is worse** (p3≈21 vs P22 worst≈29); ABC spread is larger. Ordinary PBT copies = **0** in both runs’ `pbt_copy_events` / selection logs through this progress.

**Recommendation: CONTINUE_WITH_CAUTION** — no P0 stop condition; A proven once; main open risk is **p3 remaining chronic-blocked by ranking score_gap** while nsucc stays weak, plus incomplete C/refinement telemetry.

**P0 count: 0 · P1 count: 0 · P2/P3: see §13**

---

## 2. Run Identity and Config Verification

### 2.1 Paths and commits

| Field | Value | Evidence |
|---|---|---|
| Run directory | `/root/autodl-tmp/wmdexpbt/results/runs/regrasping_cser_dexpbt_progress_refine_abc_5b_20260714_222300` | `LAUNCH_IDENTITY.txt`, live argv |
| Profile22 baseline | `/root/autodl-tmp/wmdexpbt/results/runs/regrasping_cser_dexpbt_progress_refine_5b_20260714_092846` | exists; curves present |
| Launch commit (train argv / identity) | `c890ccda63193c2b49da4549dbe451b55b39dbb2` | `LAUNCH_IDENTITY.txt`, `RUN_CONFIG.txt` |
| Repo `HEAD` at audit | `62d4dd09fb4db9d1e3347589bcef7c27495ee29b` (docs after launch) | `git rev-parse HEAD` |
| Launcher | `results/launchers/start_regrasping_cser_dexpbt_progress_refine_abc_5b.sh` | `RUN_CONFIG.txt` |
| Master PID | `859798` (`launch_pbt`) | `ps` |

### 2.2 Profile authority / ABC flags

From `RUN_CONFIG.txt` and live argv (p0/1/3/4/5):

- `wm_strategy_profile=cser_dexpbt_progress_refine_abc` (Profile ID **24**)
- `pbt_enable_chronic_bottom_exploit=true`
- `eir_tolerance_aware_bank_broadcast=true`
- `eir_preserve_local_bank_fraction=0.20`
- `eir_enable_belief_for_snippet_wm_rerank=true`
- `target_population_frames=5000000000`
- Mid mutation / AWR / residual apply / RSSM-MPPI **off** (`--no-pbt_allow_mid_self_mutation`, `--no-awr_enable`, `--no-eir_residual_apply_to_rollout`, `--no-eir_use_rssm_mppi`, `eir_mppi_mode=off`)

Profile table authority (`wm_strategy_profiles.py` `_CSER_DEXPBT_PROGRESS_REFINE_ABC`):

- `pbt_chronic_min_bottom_frames=200_000_000`
- `pbt_chronic_min_bottom_ticks=5`
- `pbt_chronic_min_score_gap=0.15`
- `pbt_chronic_cooldown_frames=400_000_000`
- `pbt_chronic_max_copies_per_lineage=1`

**p2 post-restart caveat:** attempt_2 train argv **omits** explicit `--pbt_enable_chronic_bottom_exploit` / bank / belief CLI flags, but startup log shows profile expansion:

```
pbt_enable_chronic_bottom_exploit: False -> True (profile_authoritative)
eir_tolerance_aware_bank_broadcast: False -> True (profile_authoritative)
eir_enable_belief_for_snippet_wm_rerank: False -> True (profile_authoritative)
```

Evidence: `policy_02/attempt_2_2026-07-15_02-39-44.log` lines ~72–74. Supervisor process for p2 still carries full ABC CLI flags. **Effective ABC on p2: yes via profile authority**, not via restarted train argv.

### 2.3 Checkpoint / lineage consistency (tip publishes)

`LATEST.commit` → `{commit_dir}` contains `MANIFEST` + `meta.json` + `model.pth` (not `MANIFEST.json` / `state.json` filenames).

| Worker | tip PF (meta/MANIFEST) | lineage_id | profile name/id | model.pth size == `{iter}.pth` |
|---|---|---|---|---|
| 0 | 1.52B | `39283193-…` | abc / 24 | yes (181446561) |
| 1 | 1.12B | `8c286e41-…` | abc / 24 | yes |
| 2 | 1.44B | `ffdde91b-…` | abc / 24 | yes |
| 3 | 1.24B | `38da4a27-…` | abc / 24 | yes |
| 4 | 1.28B | `73c8a860-…` | abc / 24 | yes |
| 5 | 1.24B | `b8047196-…` | abc / 24 | yes |

SHA-256 re-hash of 181MB checkpoints **not executed** in this audit (too heavy / unnecessary for liveness). MANIFEST lists `member_shas.model.pth`; byte-size equality between commit `model.pth` and published sibling `.pth` holds. **Byte-hash equality now: 无法确认** (would require full hash).

`policy_lineage.json` records p2 profile lineage update at env_frames=600M / pbt_iteration=30 (chronic copy moment).

No evidence of publish/validation fail storms in recent logs (`Traceback`/`OOM`/`CUDA`/`NaN` scans of last ~3MB of each attempt log: clean).

---

## 3. Worker Health Table

Live frames from `metrics/pbt_tick_debug_policy_XX.jsonl` last row; tip metrics from `training_curve_policy_XX.jsonl` last published row.

| worker | tip_frame | live_frame | progress | PID | GPU | alive | last_log_age_s | restart/OOM/NaN | config_valid |
|---|---:|---:|---:|---:|---:|---|---:|---|---|
| 0 | 1.52B | ~1.57B | 30–31% | 859911 | 0 | yes | ~3–7 | restart0 / clean | yes (Profile24+ABC CLI) |
| 1 | 1.12B | ~1.15B | 22–23% | 860928 | 1 | yes | ~2–8 | restart0 / clean | yes |
| 2 | 1.44B | ~1.46B | 29% | 903198 | 2 | yes | ~3–7 | **cross_restart=1 @600M** (chronic) / clean | yes (profile_authoritative ABC) |
| 3 | 1.24B | ~1.28B | 25% | 863342 | 3 | yes | ~1–3 | restart0 / clean | yes |
| 4 | 1.28B | ~1.29B | 26% | 864297 | 4 | yes | ~0–5 | restart0 / clean | yes |
| 5 | 1.24B | ~1.28B | 25% | 865599 | 5 | yes | ~1–5 | restart0 / clean | yes |

Additional health notes:

- Wall time: master ~12h50m+; p2 attempt_2 ~8h40m since chronic restart (`ps etime`).
- GPU mem ≈ 10–13 GiB / 32 GiB; util snapshots fluctuated **0–23%** during audit (not sustained 0 across long log silence — logs refresh every few seconds). **Long-term idle: not supported by log mtimes.**
- Instant util alone **cannot** prove regression; compare FPS from logs: recent `fps step and policy inference` ≈ **59k–72k** / worker.
- `mid_self_mutation_count` tip fields are 0 / `allow_mid_self_mutation` false on curves where present.
- Target 5B confirmed in argv + `RUN_CONFIG.txt`.

---

## 4. A Chronic-PBT Event Funnel

### 4.1 Ordinary vs chronic commit counts

| Class | Count | Evidence |
|---|---:|---|
| Ordinary exploit **committed** copies | **0** | `pbt_copy_events.jsonl` (only chronic row); log `source_selection … action=copy` only once and tier=`chronic_top_live` |
| Ordinary exploit **attempts** (rejected) | **无法确认** (no JSONL attempt journal); selection logs show no non-chronic `action=copy` | |
| Chronic **successful copies** | **1** | `metrics/pbt_copy_events.jsonl`; log Policy2 `tier=chronic_top_live action=copy` |
| Profile22 copies ≤~1.5B | **0** | baseline `pbt_copy_events.jsonl` empty |

### 4.2 The one chronic success (p2 ← p0 @ 600M)

From `pbt_copy_events.jsonl`:

- `chronic_path=true`, `tier=chronic_top_live`, `policy_idx=2`, `source_policy_idx=0`
- `env_frames=600000000`, `pbt_iteration=30`
- `chronic_eligibility_reason=ok`, `chronic_score_gap≈0.689`
- `chronic_source_filter_reason=ok_mixed_tol_fallback`
- `source_nsucc_mean≈28.58` vs target pre-copy nsucc≈2.64 (source clearly better)
- `mutation_applied=false`
- Target tol before/after: 0.075 → 0.0626 (`source_safe` inherit)
- Post-copy lineage for p2: `ffdde91b-…` (`policy_lineage.json`, restart argv `--pbt_lineage_id_resume=…`)
- Recovery: tip nsucc **37.13** mid; bottom span cleared after copy (TB `chronic_copy_count` on attempt_1 ended at 0 before process exit; post-restart chronic_bottom shows `last_reason=lineage_reset`, counters 0 — expected after lineage change helper `reset_on_lineage_change`)

Health / cache on that tick (`policy_02/pbt_chronic_source_health.jsonl` + TB):

1. selection validation: `cache_hit=false`, `full_hash_executed=true`, `validation_latency_s≈0.092`, all validity flags ok  
2. post-selection reuse: `cache_hit=true`, `full_hash_executed=false`, latency≈0.0014s  

### 4.3 Funnel counts (evidence-backed)

Population reconstruction uses TensorBoard scalars on p2 attempt_1 and p3 (streamed event loader), plus chronic_bottom JSON + copy JSONL.

| Stage | Count | Notes / evidence |
|---|---:|---|
| bottom observation (published curve points with `tier=bottom`) | 50 across workers (0:8,1:1,2:11,3:10,4:8,5:12) | `training_curve_policy_*.jsonl` |
| continuous bottom spans reaching **ticks≥5 & frames≥200M** | **≥2 workers historically/current**: p2 @600M (ticks=6, frames=200M); p3 current (ticks=7, frames=240M) | TB `pbt/chronic_continuous_bottom_*`; `chronic_bottom/003.json` |
| chronic condition satisfied (`chronic_candidate=1`) | **1** (p2@600M only) | TB p2; p3 candidate series n=28 **all 0** |
| source candidates found / snapshot validation passed | **1 tick**, **2 validations** (miss+hit) | TB snapshot_* + health JSONL |
| source selected | **1** (source=0) | copy event + health JSONL |
| exploit committed | **1** | `pbt_copy_events.jsonl` |
| target recovered/improved | **1/1** | p2 nsucc 2.64→23.7 within 40M frames; later 37+ |

### 4.4 p3: chronic thresholds met but candidate never selected

`chronic_bottom/003.json` now:

- `bottom_enter_frame=1000000000`
- `continuous_bottom_frames=240000000`, `continuous_bottom_ticks=7`
- `chronic_copy_count=0`

TB p3: `chronic_candidate` never 1; `chronic_score_gap` last=0.0 at 1.24B.

Reconstructed ranking gaps at p3 bottom publish points (`pbt_ranking_score`, peers within ≤40M frames):

- Several earlier bottoms had gap≥0.15, but continuous span was not yet ≥200M/5 ticks.
- After span matured (≥1.20B): tip gap at 1.24B ≈ **0.112 < 0.15** (`p4` rank 1.249 vs `p3` 1.136) despite nsucc 21 vs 39.

**Interpretation (evidence + code):** `evaluate_chronic_eligibility` (`pbt_chronic_bottom.py`) gates on **ranking score gap**, not raw nsucc. p3 is “bottom” by tier/nsucc but competitive on `success_aware_r_meta`, so chronic path does not arm. This is **design behavior**, not a failed snapshot-hash gate (p3 has **zero** snapshot_cache TB tags).

### 4.5 Checklist of failure modes (A)

| Concern | Status |
|---|---|
| Long bottom with chronic attempt=0 | **p3 yes** (blocked pre-candidate by score_gap / eligibility), **p2 no** |
| Candidates all rejected by one gate | Not observed on successful path; p3 never entered candidate |
| LATEST/lineage/hash false rejects | Not on p2 path (all ok). Ongoing mis-rejects: **no evidence** |
| Post-selection validation fails after selection | **Not observed** (post = cache hit, health allowed) |
| Copy without load/lineage change | **Contradicted** (attempt_2, lineage file, nsucc jump) |
| Rapid repeat copies same target | **No** (`max_copies_per_lineage=1`; only one copy) |
| Source not better than target | **No** for p2 (28.6 vs 2.6 nsucc) |
| Chronic bypasses common health | **No** — health JSONL `common_health_reason=ok` / `final_health_reason=ok` |

---

## 5. B Bank-Merge Audit

Source: `metrics/eir_broadcast_events.jsonl` (26 `broadcast`, 25 `consume`).

### 5.1 Per-worker consume summary

Fields from `bank_diagnostics` when present. “Compatible foreign ratio” for accepted batches = `incoming_compatible_count / incoming_count`. Rejected batches often have compatible=0.

| worker | consumes | reject_batches | accepted incoming | accepted compatible | kept foreign rows (sum kept) | last local | last foreign | last bank | compat ratio (accepted batches) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 4 | 0 | 81716 | 81716 | 81716 | 0 | 50000 | 50000 | **1.00** |
| 1 | 5 | 2 (late) | 0* | 0* | 0* | n/a | n/a | n/a | **0.00 on rejected**; early 3 lacks diagnostics (**无法确认**) |
| 2 | 3 | 1 | 50000 | 50000 | 26612 | 32055 | 17945 | 50000 | **1.00** |
| 3 | 5 | 0 | 150000 | 150000 | 68835 | 25346 | 24654 | 50000 | **1.00** |
| 4 | 2 | 0 | 50000 | 50000 | 17904 | 32096 | 17904 | 50000 | **1.00** |
| 5 | 6 | 1 | 150000 | 150000 | 50443 | 24695 | 25305 | 50000 | **1.00** |

\*p1 late rejects: source0→target1 at 1.04B/1.12B with `broadcast_rejected_incompatible=True`, `incoming_compatible_count=0`, `delta_eps_med≈0.21–0.32` (source tol≈0.038 vs target≈0.028–0.031).

Bank size after merges stays **50000** (capacity respected; no evidence of unbounded growth). Merge latency field: **absent** in JSONL → **无法确认**.

### 5.2 B judgment @ 1.5B

- **Already effectively working** for workers that accept merges (notably **weak p3** holds ~24.6k foreign).
- **Not inert.**
- Caveat: **tol-aware reject** currently starves **p1** (stronger/tighter worker) of foreign snippets from looser sources — correct for B’s gate, but means B does **not** automatically fix every lagging worker; it helps when tolerances align.
- Monopoly: sources in accepted consumes are mixed (0/1/2/5); no single-source lock on all banks (**cannot claim monopoly**).
- Near-duplicate occupancy: **无法确认** (no snippet-id telemetry in broadcast JSONL).
- Telemetry vs artifacts: payload present for some peers under `eir_broadcast/` (e.g. `003.payload.pt`); detailed content audit vs counters: only size/side files spot-checked → **partial consistency only**.

**B overall: already有效工作，有局限（兼容窗口）。**

---

## 6. C Belief-Cache Audit

Source: `metrics/eir_belief_events.jsonl` (≈73.9k rows at audit; growing).

| Metric | Value |
|---|---|
| Events | `belief_cache` only |
| `need_belief=True` | 100% |
| `belief_build_attempted=True` | 100% |
| `belief_build_success=True` | 100% |
| `belief_cache_valid=True` | 100% |
| `belief_failure_reason` empty | 100% |
| `calibration_pass` | True ≈10.1k / False ≈63.4k |
| `rerank_reason_id` | always `-1` |
| Build latency | p50≈6.4ms, p95≈7.2ms, p99≈9.3ms, max≈20ms |
| shape/dtype/device | **empty / 0 numel/ndim** in JSONL → **无法确认** |

Per-worker: all show build_ok=n and cache_valid=n; mean `alpha_wm` small (p0≈0.0002, others ≈0.02–0.05), median often 0.

### 6.1 Funnel (requested)

| Stage | Count | Evidence |
|---|---:|---|
| snippet received | **无法确认** as distinct counter (no matching JSONL) | |
| valid belief available | ≈73.9k (`belief_cache_valid`) | belief JSONL |
| candidate scored | **无法确认** (`rerank_reason_id=-1` always; no scored-count field) | |
| refinement accepted/rejected | **无法确认** (no accept/reject JSONL; logs lack reason2) | |
| snippet used in training/planning | **无法确认** | |

Invalidation counts / key coarseness wrong-reuse: **无法确认** (no invalidate events; only “cache_valid” after build). Hit vs miss: telemetry does **not** expose a hit/miss flag analogous to snapshot cache — every event also sets `belief_build_attempted=True`, so **cache hit rate cannot be computed as asked**.

Belief memory footprint on GPU/CPU: **无法确认**.

**C judgment:** mechanism is **on and producing events**; correctness of reuse and downstream refinement lift **cannot be confirmed** from available telemetry. Cost looks small (~6ms build). No evidence it is the FPS bottleneck.

---

## 7. Snapshot-Validation Cache Audit

Evidence exists **only** on p2’s single chronic selection (TB + `pbt_chronic_source_health.jsonl`).

| Quantity | Observed |
|---|---|
| Cache miss then hit same commit | miss (full hash) → hit |
| Full hash executions | 1 true + 1 false |
| Latency | 0.092s → 0.0014s |
| Cache entries (TB) | 5 |
| Fingerprint / invalidate reasons | `first_validation` then `cache_hit`; commit_id `ad4326176dedd8a3`, dir `000030.staging.commit` |

Code path: `pbt_live_source_health.py` `PeerSnapshotHealthCache` + `inspect_peer_snapshot_health`; wired in `pbt_observer.py` `_log_chronic_source_checkpoint_health`.

Unbounded growth: not evidenced (entries=5 that tick; no longer time series). Stale hit while snapshot changed: **no evidence**. Repeated full SHA in one tick after cache: **no** (second call skipped hash).

When telemetry is zero elsewhere: chronic path simply **did not enter source validation** for that worker — not instrumentation absence on code path (TB tags appear when chronic source health runs).

PBT tick pause impact: one ≈92ms hash on a rare chronic path is negligible vs multi-minute intervals between 20M-frame ticks. No training-curve iteration-time field; log gaps during audit were seconds, not minutes.

---

## 8. Profile22 vs Profile24 Same-Frame Comparison

**Controls / caveats**

- Same task, 6×8192 envs, seed 42 family, terminate @5B.
- Commits differ (ABC stack vs Profile22).
- Frame definition: `env_frames` / `population_frame` published on ≈40M curve grid.
- **Workers are not phase-locked** — nearest-point deltas reported.
- Profile22 tip at audit still ~1.12–1.48B (not finished 5B) — valid mid comparison only.
- At **240M**, ABC and Profile22 published curves are **identical** → shared early dynamics / seed; later divergence is meaningful.

### 8.1 Population nsucc summary (nearest points)

| Target | ABC best/med/worst/mean/spread | P22 best/med/worst/mean/spread | Notes |
|---|---|---|---|
| 250M | 15.66 / 0.86 / 0.00 / 3.07 / 15.66 | **identical** | Δframe≈10M |
| 500M | 29.28 / 20.66 / 0.87 / 18.02 / 28.42 | 24.18 / 20.31 / 0.87 / 17.79 / 23.32 | ABC best higher; same worst (p2) |
| 1.0B | 31.85 / 30.86 / 21.69 / 28.12 / 10.16 | 37.90 / 29.95 / 14.33 / 27.02 / 23.57 | ABC worse-best but **tighter spread**; P22 p2 still weak 14.3 |
| ~1.5B | 38.64 / 31.53 / **21.12** / 30.70 / **17.53** | 37.81 / 30.84 / **29.45** / 32.02 / **8.37** | ABC median ≈; **ABC worst far worse**; p1 ABC point is only 1.12B (Δ380M!) |

At ~1.5B, **median delta (ABC−P22) ≈ +0.68 nsucc**; **worst delta ≈ −8.3 nsucc**. Best ≈ +0.8.

### 8.2 Other requested metrics

| Metric | Result |
|---|---|
| coverage / progress refine scalars | tip curves lack clean `coverage` field → **无法确认** |
| episode return | not in tip keys used → **无法确认** |
| train/env FPS | ABC logs ~60–72k policy+infer; Profile22 same style **not fully re-binned** this audit → rough parity **无法严格确认** |
| iteration time / GPU mem | mem ABC 10–13GiB; no paired P22 live nvidia → **无法确认 regression** |
| PBT copies | ABC chronic 1 + ordinary 0; P22 0 |
| Bottom rescue rate | ABC: **1/1** chronic committed targets improved; additional chronic bottoms unresolved (p3) |
| Refinement accept / reason2 | **无法确认** (no reason2 in logs/JSONL) |
| Foreign snippet utilization | See §5; p3 yes; p1 late no |

Trend takeaway: ABC **rescued the historical chronic p2 hole** that still scars Profile22’s mid run (P22 p2≈30.7 bottom @1.48B with loose tol 0.055). ABC’s new hole is **p3**. Worker consistency **not improved** vs P22 at 1.5B.

---

## 9. Performance and Stability Regression Check

| Signal | Observation |
|---|---|
| Crashes / worker restart storms | Only intentional p2 supervisor restart after chronic copy |
| OOM / CUDA / NaN / Assert (recent 3MB logs) | None found |
| Checkpoint publish failures | No evidence |
| Log stalls | ages &lt;10s during audit |
| Snapshot hash cost | ≤92ms once / chronic tick |
| Belief build | ~6ms/event |
| Throughput regression vs P22 | **无法确认** without matched FPS time-series extraction |

GPU util momentarily low ≠ idle: FPS lines still emit.

---

## 10. Confirmed Findings

1. Run is Profile24 ABC 5B, commit `c890ccd`, 6 alive workers, target 5B, mid-mutation/MPPI/AWR/residual off.  
2. **A produced exactly one chronic copy** (p2←p0@600M) with health+cache behaving as designed; target recovered strongly.  
3. Ordinary PBT copies so far: **0** (same as Profile22 mid-run copy JSONL).  
4. **B** performs tolerance-aware accepts and rejects; bank remains capped at 50k; weak p3 holds substantial foreign mass.  
5. **C** continuously builds beliefs successfully at ~6ms; events exist for all workers.  
6. Snapshot validation cache **hit after miss** on the only chronic validation path; second call skipped full hash.  
7. p3 meets bottom-duration thresholds but **never** sets `chronic_candidate=1` in TB through 1.24B.  
8. At ~1.5B, ABC median ≈ Profile22; ABC **worst lag** is larger.

---

## 11. Suspected Findings

1. p3 chronic block is primarily **`score_gap_below_threshold` on ranking score**, not snapshot validation (TB candidate never fires; tip gap≈0.11&lt;0.15).  
2. Restart argv omitting ABC CLI flags is fragile; today saved by `profile_authoritative` — future restart paths that skip profile expansion would silently disable ABC.  
3. High `calibration_pass=False` + low `alpha_wm` may mean C rarely influences policy updates despite “successful” belief builds.  
4. Early identical 250M curves suggest limited early ABC effect (expected: chronic starts after long bottoms; bank/belief ramp later).

---

## 12. Missing Telemetry / Cannot Determine

- Ordinary PBT attempt/reject funnel counts  
- Per-tick chronic eligibility reason JSONL (only TB scalars + final copy event)  
- Belief true hit/miss, shape/device, RAM/VRAM, wrong-reuse detection  
- Snippet→refinement accept/reject and reason2 rates  
- Full SHA equality of tip checkpoints vs MANIFEST  
- Matched iteration-time / FPS regression vs live Profile22  
- Near-duplicate snippet occupancy inside banks  
- Whether p3 had any eligibility near-miss ticks between 1.16–1.20B beyond TB candidate=0 series  

---

## 13. P0 / P1 / P2 / P3 Issue List

### P0 (invalid result / must stop)
- **None.**

### P1 (correctness bug; low interpretability if continued)
- **None confirmed.** (p3 non-firing is consistent with coded score-gap gate.)

### P2 (works but weak / tune next round)
1. Chronic eligibility tied to ranking score leaves **low-nsucc / high-rank bottoms** unsaved (p3).  
2. B leaves **tight-tol workers** (p1) with 0 compatible foreign from looser elites.  
3. C’s downstream influence opaque; calibration mostly failing.  
4. Population spread vs Profile22 **worsened** at ~1.5B despite one strong rescue.

### P3 (telemetry / ops / docs)
1. No JSONL attempt journal for chronic eligibility rejects.  
2. Belief events omit hit/miss & tensor meta.  
3. Restart train argv drops ABC flags (mitigated by profile authority).  
4. MANIFEST filename is `MANIFEST` not `MANIFEST.json` (audit hazard only).

---

## 14. Continue / Stop Recommendation

### CONTINUE_WITH_CAUTION

Why continue:

- Healthy, correct Profile24 identity, ABC actually executing.  
- A demonstrated a real rescue with safe health/cache path.  
- No P0/P1 correctness break.  
- Mid-run only (~25–30% of 5B); more chronic opportunities remain if ranking gaps reopen.

Why caution:

- Current weakest worker not being chronically helped.  
- Spread vs Profile22 is worse at this slice.  
- C/refinement benefit still unproven.

**Do not stop** purely for mid-run lag; stop only if future chronic candidates show systematic health/hash false rejects or worker deaths/OOM.

---

## 15. 下一步最小实验建议

1. **Keep this run to ≥3B** and re-audit when p3 either (a) gets `chronic_candidate=1`, or (b) exits bottom with nsucc→median.  
2. **Telemetry-only patch (next code drop, not this run):** JSONL for `chronic_eligibility.reason` every bottom tick + belief hit/miss + refinement accept counters.  
3. **Ablation later (new run):** chronic score-gap on **nsucc_mean** (or dual gate) vs ranking-only — single-factor change.  
4. **Do not** alter Profile22 baseline; do not kill current ABC unless P0 appears.

---

## Appendix — Key Evidence Paths

- Run: `…/regrasping_cser_dexpbt_progress_refine_abc_5b_20260714_222300/`
- Identity: `LAUNCH_IDENTITY.txt`, `RUN_CONFIG.txt`
- Curves: `dexpbt_8gpu_ws/metrics/training_curve_policy_0{0-5}.jsonl`
- Copies: `dexpbt_8gpu_ws/metrics/pbt_copy_events.jsonl`
- Belief: `dexpbt_8gpu_ws/metrics/eir_belief_events.jsonl`
- Bank: `dexpbt_8gpu_ws/metrics/eir_broadcast_events.jsonl`
- Chronic state: `dexpbt_8gpu_ws/chronic_bottom/00{0-5}.json`
- Chronic health: `policy_02/pbt_chronic_source_health.jsonl`
- TB tags: `policy_0*/rl_games/attempt_*/summaries/events.out.tfevents*`
- Code: `pbt/pbt_chronic_bottom.py`, `pbt/pbt_observer.py`, `pbt/pbt_live_source_health.py`, `world_model/wm_strategy_profiles.py`
- Baseline: `…/regrasping_cser_dexpbt_progress_refine_5b_20260714_092846/`
