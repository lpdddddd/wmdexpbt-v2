# Profile24 ABC 5B — Pre-Termination Root-Cause Audit

**Audit window (local):** 2026-07-16 ~12:12–12:35 Asia/Shanghai  
**Mode:** read-only (no code/config/run mutation; training left running; no full SHA; no GPU ckpt load)  
**Prior reports:** [`../profile24_abc_1p5b_audit/PROFILE24_ABC_1P5B_AUDIT.md`](../profile24_abc_1p5b_audit/PROFILE24_ABC_1P5B_AUDIT.md), [`../profile24_abc_2b_joint_audit/PROFILE24_ABC_2B_JOINT_AUDIT.md`](../profile24_abc_2b_joint_audit/PROFILE24_ABC_2B_JOINT_AUDIT.md)  
**Focus:** degradation **after ~2B**, especially p1/p3/p4 collapse and the **p3→p4 ordinary copy @2.48B**.  
**Rule:** missing evidence → **无法确认**. Correlation ≠ causation.

---

## 1. Executive Summary

Profile24 ABC (`…_222300`) is **still alive** at tip frames **2.56–2.84B / 5B**. Tip metrics are **authentic** (curve `nsucc_mean` == workspace meta to numerical identity). Since the ~2B audit, the population has **materially degraded**:

| Metric | ~2B audit | Now |
|---|---|---|
| tip min / med / max nsucc | ~24.5 / 28.9 / 38.3 | **5.5 / 8.0 / 28.9** |
| workers nsucc&lt;20 | 0 | **3** (p1,p3,p4) |
| workers nsucc&lt;10 | 0 | **3** (p1,p3,p4) |
| spread | ~13.7 | **~23.4** |
| committed copies | 1 chronic | **1 chronic + 1 ordinary** |

**p3→p4 @2.48B** is a confirmed **ordinary** (`chronic_path=false`) `strict_stable_live` copy that **did load** (`attempt_2`, new lineage `004ed42a-…`, ckpt `restart_p4/000086.pth`). Post-copy trajectory: **4.2 → 13.2 → 19.4 → 24.9 → 15.9 → 7.2**. Classification: **temporary rescue then collapse** — not immediate harm, but **not a durable rescue**. Source metrics in the event (`nsucc=34.57 @ tol=0.0227`) match p3 meta **`000086` @2.04B**, while live p3 at copy time was already **~23 @0.0184** and then fell to 8 — **stale “latest_healthy” snapshot relative to live tip**.

**p5** at **tol≈0.0119, nsucc≈26.7, freeze_until=0** is the critical internal control: **tight tolerance alone does not force collapse**. Collapsed workers share **post-shrink freeze-heavy histories** and (for p1/p4) **hard-floor ~0.0126** with falling nsucc under freeze.

**A:** chronic still rarely commits; p1 is **one tick short** of chronic (ticks=4&lt;5, frames=240M≥200M).  
**B:** after 2B mostly `rejected_incompatible`; `snippet_bank_merged_entries` often 0.  
**C:** belief builds + high calib_pass in recent window, but **`rerank_reason_id=-1`** in sampled events → **decision impact still unproven**.

**Final recommendation: `CONTINUE_UNTIL_ALL_REACH_3B`** (event-gated), **not** blind 5B. Do **not** auto-stop; await user confirm. Preserve listed checkpoints/events.

**P0: 0 · P1: 2 · P2: several · P3: telemetry/perf**

---

## 2. Current Run Identity and Health

| Field | Value | Evidence |
|---|---|---|
| Run dir | `/root/autodl-tmp/wmdexpbt/results/runs/regrasping_cser_dexpbt_progress_refine_abc_5b_20260714_222300` | `LAUNCH_IDENTITY.txt` |
| Launch commit | `c890ccda63193c2b49da4549dbe451b55b39dbb2` | identity / `RUN_CONFIG.txt` |
| Repo HEAD | `62d4dd09fb4db9d1e3347589bcef7c27495ee29b` | `git rev-parse` (docs after launch) |
| Master | PID `859798`, etime ~1d14h | `nohup.pid` / `ps` `launch_pbt` |
| Profile | `cser_dexpbt_progress_refine_abc` (24) | argv / RUN_CONFIG |
| Target | 5B frames/worker | `--target_population_frames=5000000000` |

### Worker table (audit tip)

| worker | frame | nsucc | tol | tier | bottom ticks | bottom frames | protect (promising_until in copy evt) | freeze_until | hard_floor thr | last copy | alive | PID / attempt |
|---|---:|---:|---:|---|---:|---:|---|---:|---:|---|---|---|
| 0 | 2.80B | 28.87 | 0.0316 | top | 0 | 0 | n/a tip | 2.76B (inactive) | 0.025 | — | ✓ | 859911 / attempt_1 |
| 1 | 2.60B | **5.50** | 0.0126 | bottom | **4** | **240M** | — | **2.80B active** | 0.025 | — | ✓ | 860928 / attempt_1 |
| 2 | 2.72B | 20.34 | 0.0148 | mid | 0 | 0 | — | 2.92B active | 0.025 | received @0.60B | ✓ | 903198 / attempt_2 |
| 3 | 2.60B | **8.00** | 0.0184 | bottom | **1** | 0* | — | **2.80B active** | 0.025 | source @2.48B | ✓ | 863342 / attempt_1 |
| 4 | 2.84B | **7.16** | 0.0126 | bottom | **2** | 80M | promising_until=2.13B (expired before copy) | **3.04B active** | 0.025 | **received ordinary @2.48B** | ✓ | 397873 / **attempt_2** |
| 5 | 2.56B | **26.71** | **0.0119** | top | 0 | 0 | — | **0** | 0.025 | — | ✓ | 865599 / attempt_1 |

\* `chronic_bottom/003.json`: `bottom_enter_frame=2600000000`, `continuous_bottom_frames=0` with `ticks=1` (just entered).

**Metric authenticity:** for all workers, tip `training_curve_*.jsonl` `nsucc_mean` == latest `{iter}.json` `nsucc_mean` (exact). Not a parse/window artifact.

**OOM/NaN/CUDA/Traceback:** last ~3MB per active attempt — **no OOM / NaN / Traceback** observed (p4 attempt_1 ends with intentional exit 42). Log ages seconds.

**Lineages:** p0 `39283193-…`; p1 `8c286e41-…`; p2 `ffdde91b-…` (post-chronic); p3 `38da4a27-…`; p4 **`004ed42a-…` (new at copy)**; p5 `b8047196-…`.

---

## 3. Population Collapse Verification

**Verdict: yes — multi-worker hard-floor collapse after 2B, not a single-worker blip.**

Evidence:
- 3/6 workers tip nsucc&lt;10; median tip **8.0** (was ~29 at 2B).
- Collapse onsets after shrinks into **≤0.018** (esp. **~0.0126** for p1/p4).
- Not explained by tip staleness (meta match) or dead processes (all alive).
- Not purely “PBT负迁移 only”: p1 and p3 collapsed **without** being copy targets this window; p4 collapsed **before** copy, briefly recovered after copy, then collapsed again.

**Best-fit class:** **multi-worker synchronized pressure at tight tolerance + freeze-without-restore**, with **one ordinary copy that temporarily helped then failed**. Not “指标窗口变化”.

---

## 4. Trajectory Since 2B

Published points from `metrics/training_curve_policy_XX.jsonl` with `env_frames≥2e9` (see `traj_since_2b.json`).

### 4.1 Per-worker highlights

| w | 2B+ peak nsucc | tip | drop | shrinks after 2B | first &lt;20 | first &lt;10 | dur≈&lt;20 | freeze pts |
|---|---:|---:|---:|---|---|---|---:|---|
| 0 | 31.9 @2.52B /0.032 | 28.9 | 3.0 | 0.035→0.032 @2.44B | none | none | 0 | rare |
| 1 | 29.7 @2.00B /0.014 | **5.5** | **24.2** | 0.014→0.013 @2.04B | 2.12B | **2.60B tip** | ~0.32B | **7/9** |
| 2 | 38.3 @2.00B /0.020 | 20.3 | 17.9 | three shrinks to 0.015 | 2.48B | 2.52B (brief) | ~0.20B | many |
| 3 | 34.6 @2.12B /0.023 | **8.0** | **26.6** | to 0.018 @2.24B | 2.52B | tip | ~0.08B | **8/16** |
| 4 | 27.3 @2.16B /0.013 | **7.2** | **20.2** | none after 2B (already 0.013) | **2.00B** | **2.00B** | ~0.52B | **9/12** |
| 5 | 26.9 @2.12B /0.018 | **26.7** | 0.2 | **six** shrinks to **0.012** | none | none | 0 | **0/13** |

### 4.2 AUC since 2B (nsucc·B, trapezoid)

| w | AUC |
|---|---:|
| 0 | 23.15 |
| 1 | **11.15 (worst)** |
| 2 | 18.21 |
| 3 | 15.83 |
| 4 | 13.84 |
| 5 | 14.25 |

Median AUC ≈ **14.25**; worst = p1.

### 4.3 Population stats (tip)

min/med/max/mean/spread = **5.5 / 8.0 / 28.9 / 16.1 / 23.4**  
n&lt;20 = **3**; n&lt;10 = **3**.

---

## 5. p3→p4 Copy Event Study

### 5.1 Facts (hard evidence)

| Item | Value | Evidence |
|---|---|---|
| Time / frame | wall ts `1784153082.46`; **env_frames=2.48B** | `metrics/pbt_copy_events.jsonl` full row |
| Source / target | **p3 → p4** | same |
| Path | **`chronic_path=false`**, tier=`strict_stable_live` | same |
| Source ckpt type | `latest_healthy` | event + log |
| Source metrics in event | nsucc=**34.572**, tol=**0.022724**, retention=1.0, safe=true | event |
| Matching meta | **`003/000086.json` @ frame 2.04B** | workspace meta |
| Ckpt file | `dexpbt_8gpu_ws/restart_p4/000086.pth` | p4 attempt_1 log L23133 |
| Target tol before/after | **0.012596 → 0.012596** (unchanged) | event `inheritance_reason=source_strong_live_inherit_exact` |
| Mutation | `mutation_applied=true` | event |
| Load / lineage | attempt_2 start; lineage **`004ed42a-…`**; `frames_at_restart=2480000000`; restore log `pure copy snapshot target_pf=2480000000` | argv + attempt_2 log HEAD |
| Promising | `protected_by_promising_gate=false`; `promising_protect_until=2130000000` (expired) | event |

**Ordinary committed copies:** now **1** (this event).  
**Chronic committed copies:** still **1** (p0→p2 @0.60B).  
(`pbt_copy_events.jsonl` also has 3 incomplete companion rows without `source_policy_idx` at same timestamp — telemetry duplication, not extra copies.)

### 5.2 Pre/post trajectories (±200M)

**Source p3**

| rel to copy | frame | nsucc | tol |
|---|---:|---:|---:|
| −200M | 2.28B | 25.5 | 0.0184 |
| −40M | 2.44B | 26.4 | 0.0184 |
| **0** | **2.48B** | **23.1** | **0.0184** |
| +40M | 2.52B | 10.6 | 0.0184 |
| +120M | 2.60B | **8.0** | 0.0184 |

**Target p4**

| rel | frame | nsucc | tol | lineage |
|---|---:|---:|---:|---|
| −160M | 2.32B | 14.7 | 0.0126 | old `73c8a860` |
| −80M | 2.40B | 6.6 | 0.0126 | old |
| **0** | **2.48B** | **4.2** | **0.0126** | old (pre-restart publish) |
| +40M | 2.52B | **13.2** | 0.0126 | **new `004ed42a`** |
| +120M | 2.60B | 19.4 | 0.0126 | new |
| +200M | 2.68B | **24.9** | 0.0126 | new |
| tip | 2.84B | **7.2** | 0.0126 | new |

### 5.3 Answers to key questions

1–3. Time/frame/src/tgt/path: as table. **Ordinary, not chronic.**  
4–5. At decision, live p3≈23 @0.018; event advertised **34.6 @0.023 from 2.04B meta**. Live p4≈4.2 bottom under freeze.  
6–8. Source chosen as sole `strict=[3]` stable candidate (`policy_04/...attempt_1.log` L23131). **Source metric frame (2.04B) ≠ copy decision frame (2.48B)** — **stale relative to live tip**. Snapshot validation detail for this ordinary path: **无法确认** (no new `pbt_chronic_source_health.jsonl` under p4; that file exists only under p2 from the old chronic).  
9–11. Health gate: copy committed + supervisor restart succeeded → **passed enough to execute**. p3 was **already past local peak** and mid-decline.  
12. Source after copy: **yes, fell to 8** within ~120M.  
13–16. Parameters loaded: yes (attempt_2 + new lineage). Optimizer/WM: WM path `003/wm_seq_checkpoints/wm_seq_latest.pt` in argv; full optimizer identity **无法确认** beyond standard restart argv. **Tolerance did not jump to source 0.0227** — stayed **0.0126**.  
17–18. Short rise then fall → **not immediate harmful**; **not durable**. Counterfactual “without copy”: p4 was already 4–7 under freeze for hundreds of M — **likely still poor**; copy **did** produce a ~200M recovery window.  
19–20. Why not p0/p5: log shows `strict=[3]` only and **empty** `strict_filter` → `top_raw` effectively yielded only p3 as stable candidate at that tick (`_select_source_stable_mean` in `pbt_observer.py` ~7900–7935). Exact exclusion reason for p5/p0 this tick: **无法确认** beyond “not in strict_candidates”.

### 5.4 Classification

**`temporary rescue then collapse`**

(Not “immediate harmful transfer”; not “successful rescue”.)

---

## 6. p1/p3 Rescue Failure Analysis

Chronic thresholds (profile): `min_bottom_frames=200M`, `min_bottom_ticks=5`, `min_score_gap=0.15`, `max_copies_per_lineage=1`, `cooldown=400M`.

### 6.1 Funnel (post-2B, evidence-limited)

| Stage | p1 | p3 | p4 (new lineage) |
|---|---|---|---|
| bottom | yes | yes (newly) | yes |
| persistent frames≥200M | **yes (240M)** | no (0) | no (80M) |
| ticks≥5 | **no (4)** | no (1) | no (2) |
| candidate armed | **0 observed** | 0 | 0 |
| copy committed (chronic) | 0 | 0 | 0 (got **ordinary** instead) |

`pbt_tick_debug_*.jsonl` after 2B contains scheduling fields only — **no chronic eligibility reason stream** → fine-grained reject journal **无法确认**.

### 6.2 Direct answers

1. p1 bottom×4: `chronic_bottom/001.json` ticks=4, enter=2.36B; prior bottoms can reset on tier exits (`last_reason=lineage_reset` historically). **Current span has not reached 5 ticks.**  
2. Why no chronic at nsucc=5.5: **ticks gate (4&lt;5)** despite frames OK.  
3. p3 second crash: only **1** bottom tick at tip — chronic not armed.  
4. `max_copies_per_lineage`: p3 lineage never received a chronic copy (`chronic_copy_count=0`) — **not blocked by max_copies**.  
5. Prior self-recovery without copy may have **reset bottom counters**, delaying chronic — consistent with state machine, not separately proven as “protect forever”.  
6. Promising: for p4 copy event, promising **already expired** (until 2.13B). p1 promising at tip: **无法确认** from curve fields alone.  
7. Collapse freeze: tips show freeze_until ahead of frame for p1/p3/p4 — **stops further shrink**, does **not** roll back weights (no pre-collapse restore observed).  
8. Ranking: p0 remains top with nsucc~29 but **tol=0.032**; r_meta/ranking can look strong while strict precision is weak — **difficulty-normalized gap**.  
9. Duration vs ticks desync: **confirmed on p1** (frames pass, ticks fail).  
10. Healthy sources now: **p5** (26.7@0.012) and **p0** (28.9@0.032, loose).  
11. p0 as source for tight targets: risky / historically used with relax inheritance; **not selected** for p4’s ordinary copy.  
12. p5 not source that tick: **not in `strict=[3]`** — exact filter **无法确认**.

**Ordinary pre-copy blockers on p4 (log):** repeated `consolidation gate: selection_hard_collapse_only` then eventual copy when gate cleared (`attempt_1` ~L18823–23133).

---

## 7. p5 Internal Control Analysis

| Dimension | p5 | p1 / p4 (collapse) | p3 |
|---|---|---|---|
| tip tol | **0.0119** | 0.0126 | 0.0184 |
| tip nsucc | **26.7** | 5.5 / 7.2 | 8.0 |
| freeze since 2B | **0/13 pts** | 7/9 ; 9/12 | 8/16 |
| shrinks after 2B | many, **gradual**, nsucc stays ~23–27 | p1 one shrink then long freeze-decline; p4 already at floor | shrinks then late cliff |
| lineage | original | original / **post-copy** | original |
| belief (last 20k sample) | calib_pass **100%**, mean α≈0.59 | pass high but fails present; α≈0.50–0.55 | pass high |
| bank after 2B | some accepts from p5→p0 with merge 5000 | mostly reject incompatible as target | mixed |

**Answers:**
1. p5 survives 0.012 because it **kept nsucc≥~23 while shrinking**, and **never entered collapse-freeze** in the 2B+ window.  
2. p1/p4 fail at ~0.013 with **freeze-active decline** after earlier peaks.  
3. hard_floor **threshold field=0.025** is present on all tips; it did **not** stop p5 from shrinking below 0.025. Whether it “protected” p5: **cannot claim causal protection** — p5’s distinction is **no freeze + stable nsucc under shrink**.  
4–7. Curriculum history differs (p5 more shrinks but stable); elite-bank causality **无法确认**; best observed discriminator: **`collapse_freeze_until_env_frames` active + falling nsucc** vs **freeze_until=0 + flat-high nsucc**.

---

## 8. Curriculum and Collapse-Protection Audit

Configured (argv): shrink factor 0.95; post_shrink consolidation 150M; collapse_freeze 200M; promising protect 250M; hard_floor thr 0.025; adaptive_fast curriculum.

**Findings:**
1. Collapse pattern matches **tighten-then-fail**, not random weight corruption.  
2. p1 shrink to 0.0126 @2.04B with nsucc still ~29, then decay under freeze. p4 shrink to 0.0126 @1.92B (pre-2B) with immediate weakness into 2B.  
3. Recovery windows exist (p3 once self-healed earlier; p4 post-copy ~200M) but **not durable** at tip.  
4. Freeze = **stop further shrink**, not restore — confirmed by continued low nsucc while freeze_until advances.  
5. Pre-collapse ckpts exist on disk (inventory §13) but **are not auto-rolled back**.  
6. hard_floor thr identical across workers; behavior still diverges → thr alone ≠ outcome.  
7–9. p0 @0.032 top-ranked illustrates **missing difficulty normalization** in population “health” narrative.

---

## 9. A/B/C Causal Contribution

### A chronic PBT
- Executed historically: **1 chronic success** (p2←p0 @0.6B).  
- Post-2B chronic commits: **0**.  
- Ordinary commit: **1** (p3→p4) — A’s chronic path did **not** perform this rescue.  
- Rescue quality: chronic p2 = lasting; ordinary p4 = temporary.  
- Source gate gap: **stale latest_healthy metrics** on ordinary copy.

### B tol-aware bank
- After 2B: 44 broadcast events; many `broadcast_rejected_incompatible=true`; `snippet_bank_merged_entries` often **0**.  
- Some consumed recovery entries still merge (e.g. p5→p0 merge 5000).  
- **Downstream policy improvement from B after 2B: 无法确认.**  
- Harm from B: **无法确认** (rejects dominate on weak targets).

### C belief
- File `eir_belief_events.jsonl` ~150k lines; recent window: build/success/cache_valid true; calib_pass high (p5 100%).  
- Sampled events: **`rerank_reason_id=-1`**, `rerank_changed` not evidenced.  
- **Classification: running, calibration often passes, decision contribution still unproven.**  
- Harm amplification during collapse: **无法确认**.

---

## 10. Ordinary Copy No-Commit Root Cause

Prior near-misses (`action=copy` without JSONL) still stand for earlier p3 lines. **This audit adds the first successful ordinary commit.**

Code path: `_select_source_stable_mean` returns `action=copy` **as a selection decision**; commit requires bottom-restart / supervisor exit 42 (`pbt_observer.py`). Earlier misses likely failed later gates (`selection_hard_collapse_only`, promising, skill_delta) — p4 log shows those skips **immediately before** the successful copy.

**Not proven as a latent correctness bug that invalidates the run**, but telemetry for failed commits remains weak (P3).

---

## 11. Profile22 and Historical Comparison

### Profile22 (`…_092846`)
Tips aged **~40h**, frames **1.12–1.48B** only — **still stalled**.  
**No fair 2.6–2.8B comparison possible; no extrapolation.**

### Historical “~4B, tol≈0.22, nsucc≈42”
Automated scan of other run dirs for `tol≈0.22 ±0.03`, `nsucc≥35`, `frames≥3B` on policy_00 curves: **0 hits**.  
User-mentioned regime **not located in this pass** — list for manual confirm if they meant **tol≈0.022** (2.2cm) instead. **不得擅自认领历史 run。**

Fair@0.022 eval (prior, last ckpts ~1.8–2.1B): mean nsucc **12.7**, best p2 **29.5** — already showed **strict-tol stratification** before this collapse wave.

---

## 12. Marginal Value of Continued Training

| Option | Extra GPU-h (≈6×wall) | Expected info | P(change method conclusion) | Note |
|---|---:|---|---:|---|
| STOP_NOW | 0 | preserve evidence; lose p1 chronic tick opportunity | low | conclusions already strong |
| ≤200M more | ~6×(~2–3h) ≈ 12–18 | maybe p1 reaches ticks=5; p4 freeze expiry | low–mod | cheap option |
| **ALL→3B** | ~6×(~7h straggler) ≈ **40** | full 3B gate; possible chronic rescue | mod | **recommended** |
| →5B | ~6×(~40h) ≈ **240** | little new on B/C; more floor churn | **low** | poor ROI |

ETA (from tip FPS ~17–24k): straggler **~7h to 3B**, **~39h to 5B**.

Method-level conclusions already supported:
- A can work (chronic once) but is rare / mistimed;
- ordinary PBT can commit but source freshness + durability weak;
- B/C still lack proven causal gains;
- curriculum/freeze insufficient for hard-floor recovery.

Even if one worker recovers later, **unlikely to overturn** those method conclusions.

---

## 13. Checkpoints and Evidence to Preserve

**Do not delete/move.** Paths already on disk:

### Per-worker (primary)

| w | reason | frame | nsucc | tol | path |
|---|---|---:|---:|---:|---|
| 0 | highest raw / tip | 1.92B / 2.80B | 34.7 / 28.9 | 0.035 / 0.032 | `…/000/000096.pth`, `…/000/000140.pth` |
| 1 | pre-collapse / last healthy / tip | 2.12 / 2.36 / 2.60B | 17.7 / 20.6 / 5.5 | 0.0126 | `001/000081.pth`, `000084.pth`, `000087.pth` |
| 2 | peak / tip | 2.00 / 2.72B | 38.3 / 20.3 | 0.020 / 0.015 | `002/000089.pth`, `000107.pth` |
| 3 | peak / last healthy / tip / **copy source meta** | 2.12 / 2.48 / 2.60 / **2.04B** | 34.6 / 23.1 / 8.0 / **34.6** | 0.023 / 0.018 / 0.018 / **0.023** | `003/000088.pth`, `000097.pth`, `000100.pth`, **`000086.pth`** |
| 4 | pre-copy / post-copy / peak post-copy / tip / **restart blob** | 2.48 / 2.52 / 2.68 / 2.84 / 2.04B | 4.2 / 13.2 / 24.9 / 7.2 / 34.6 | 0.0126 | `004/000086–000091.pth`, **`restart_p4/000086.pth`** |
| 5 | tightest strong / tip | 2.56B | 26.7 | **0.0119** | `005/000098.pth` |

Full machine inventory: `checkpoint_inventory.json`.

### Must-keep artifacts
- `metrics/pbt_copy_events.jsonl`
- `metrics/eir_broadcast_events.jsonl`, `eir_belief_events.jsonl`
- `metrics/training_curve_policy_0{0-5}.jsonl`
- `chronic_bottom/00{0-5}.json`
- `policy_0{2,4}/attempt_2_*.log`, `policy_04/attempt_1_*.log` (copy lines)
- `policy_02/pbt_chronic_source_health.jsonl`
- TensorBoard under each `policy_XX/rl_games/...`
- `RUN_CONFIG.txt`, `LAUNCH_IDENTITY.txt`, launcher script, commit `c890ccd…`

---

## 14. P0/P1/P2/P3 Findings

### P0
*(none)* — identity valid; metrics consistent; no evidence overwrite risk requiring immediate kill.

### P1
1. **Stale ordinary source snapshot:** event metrics/ckpt from p3 **2.04B** used at target **2.48B** while live p3 already weaker. Impact: miscalibrated rescue. Confidence: **high**. Affects interpretability of “healthy source”. Supports tighter stop gates / source freshness fixes next run.  
2. **Ordinary copy temporary-only + population hard-floor collapse:** 3 workers &lt;10 nsucc. Impact: run no longer informative as a healthy 5B showcase. Confidence: **high** on collapse; **medium** on copy causality for final tip (post-copy recovery existed).

### P2
- Chronic ticks vs frames desync (p1 4/5).  
- Freeze stops shrink but does not restore.  
- Ranking favors loose-tol p0.  
- B mostly incompatible rejects after 2B.  
- C no proven rerank effect.  
- Ordinary PBT long silent then one fragile commit.

### P3
- Incomplete companion rows in copy JSONL.  
- tick_debug lacks eligibility reasons.  
- E2E FPS ~17–21k (was higher early).  
- Profile22 dead as comparator.

---

## 15. Final Stop/Continue Decision

### `CONTINUE_UNTIL_ALL_REACH_3B`

**Rationale (not sunk-cost):**
1. Systemic multi-worker collapse: **yes**.  
2. p3→p4 copy: **committed ordinary**; outcome **temporary rescue then collapse**.  
3. p1/p3/p4 recovery signals: **weak**; p1 is **1 chronic tick away** — waiting to 3B can observe whether A fires with p5 as potential source.  
4. p5 shows tight tol ≠ inevitable death.  
5. Second chronic rescue: **possible but not guaranteed** (p1 ticks; p4 new lineage counts reset).  
6. Source pool: **p5 healthy**, p0 strong-but-loose.  
7. B/C explanatory gains by 5B: **unlikely**.  
8. Method conclusions: **mostly already fixed**.  
9. Info value to 5B: **mostly exhausted**; to 3B: **small but real** (chronic opportunity + freeze expiry).  
10. Preserve evidence now; **user confirms any stop**.

**Not `CONTINUE_TO_5B`.**  
**`STOP_NOW` acceptable** if user prioritizes GPU reclaim over the last chronic observation.  
**`CONTINUE_FOR_AT_MOST_200M`** is a valid cheaper alternative focused on p1’s 5th tick.

---

## 16. Next-Run Recommendations

1. Enforce **source freshness**: `|source_meta_frame - decision_frame|` cap; log both.  
2. Chronic: align ticks/frames or emit eligibility JSON every bottom tick.  
3. Add **difficulty-normalized** ranking for source/target.  
4. On hard collapse, optional **restore pre-collapse ckpt** (freeze alone insufficient).  
5. Short ablations for B/C with forced downstream counters (rerank_changed, snippet_used).  
6. Keep mid-mutation off; do not treat this run as a clean 5B success curve.

---

*Training was not stopped by this audit.*
