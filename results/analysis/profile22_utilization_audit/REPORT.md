# Profile 22 Mechanism Utilization Audit

**Primary run:** `regrasping_cser_dexpbt_progress_refine_5b_20260712_114254`  
**Profile:** `cser_dexpbt_progress_refine` (id 22)  
**Launcher:** `results/launchers/start_regrasping_cser_dexpbt_progress_refine_5b.sh`  
**Commit at launch:** `24c6c737630db043774e0c28ebd12ecb6adc605b`  
**Analysis scripts:** `results/analysis/profile22_utilization_audit/audit_profile22.py`  
**Raw dump:** `results/analysis/profile22_utilization_audit/audit_raw.json`

**Evidence scope notes**
- Jul12 **TensorBoard events were deleted** during ckpt cleanup (`events*` count = 0).  
- Snippet / distill / WM **loss magnitudes** below use the **same Profile 22** live rerun `progress_refine_5b_20260714_092846` as a TB proxy, and are explicitly labeled `proxy`.  
- Jul12 PBT / curriculum / bank broadcast conclusions use Jul12 JSONL + sidecars only.

---

## 1. Executive Summary

Profile 22 is **configured richly** (PBT + adaptive consolidation + EliteSnippetBank + progress-conditioned distill + WM rerank), but the Jul12 5B run’s **actual executed shape** is:

```text
6 nearly independent PPO workers
+ high-frequency mastery / adaptive-fast curriculum (A)
+ EliteSnippetBank fill + cross-worker bank broadcast (B)
+ progress-conditioned snippet distill with nontrivial ESS but tiny loss vs PPO (B/C)
+ WM imagined diagnostics on; WM snippet rerank almost always fallen back (C/D)
+ PBT weight copy almost never executes (1 adaptive rescue in ~2.5B frames) (C)
```

**Headline rates (Jul12)**
| Mechanism | Configured | Runtime eligible | Actually executed |
|---|---|---|---|
| PPO | on | always | continuous |
| Mastery + adaptive-fast curriculum | on | frequent | **84 shrinks** across workers; factors 0.90/0.925/0.95 |
| EliteSnippetBank | on | after insert/broadcast | **bank_size→50k**; 56 broadcasts / 53 consumes |
| Progress-conditioned snippet distill | on | after 300M | **active_frac≈0.28–0.31 (proxy)**; loss ≈ `5e-5` |
| Fallback actor distill | on | flagged with snippet | **loss≈0 (proxy)**; not a second gradient path |
| WM snippet rerank | on | attempted | **changed_frac=0**; fallback_reason_id=**2 belief_missing** (proxy) |
| Ordinary bottom cross-copy | on | ticks exist | **0** ordinary copies in `pbt_copy_events.jsonl` |
| Adaptive rescue | on | p2: 9 attempts | **1 executed** (p1→p2 @ 1000M, mutation disabled) |
| Mid self-mutation | **off** | mid path entered 98× | **0 self-restarts**; all blocked |
| AWR / RSSM-MPPI / residual / WM profile copy / readiness / graduation | off | n/a | n/a (Profile 23 only for readiness/graduation) |

---

## 2. Resolved Profile 22 Configuration

### Precedence (authoritative)

1. Launcher CLI (`start_regrasping_cser_dexpbt_progress_refine_5b.sh`) sets **generic** curriculum / PBT / safety / frame-stop.  
2. `--wm_strategy_profile=cser_dexpbt_progress_refine` with **profile_authoritative=true** expands `_CSER_DEXPBT_PROGRESS_REFINE` from `wm_strategy_profiles.py` and **overrides** defaults (logged in each `policy_*/train.log`).  
3. Profile 23 (`cser_dexpbt_readiness_adaptive`) is a **separate** dict on top of P22; Jul12 run does **not** enable readiness / WM graduation.

Evidence: `RUN_CONFIG.txt` (`profile_id=22`, `readiness` absent, `awr=false`, residual/rssm false) + train.log expansion lines (`elite_snippet_enable False→True`, `snippet_distill_enable False→True`, `progress_refinement_enable False→True`, `pbt_allow_mid_self_mutation` CLI `--no-...`).

### Key resolved values

| Key | Effective value | Source |
|---|---|---|
| `pbt_allow_mid_self_mutation` | **False** | launcher `--no-pbt_allow_mid_self_mutation` + profile hard-off |
| `pbt_disable_self_restart` | **True** | launcher |
| Mid worker self-mutation / self-restart | **never executes** when mid path taken | `pbt_observer._eligible_for_self_restart`: `disable_self_restart and not allow_mid_self_mutation` → block |
| Top / mid / bottom | ternary ranking; mid attempts self-path then returns; bottom may copy | `pbt_observer` mid branch increments `mid_self_mutation_count` then blocks |
| Ordinary bottom mutation | only when a bottom copy executes **and** mutation not frozen / not rescue | rescue event records `mutation_disabled: true` |
| Exploration / consol / selection mutation | `base 0.2`; consol ×0.5; selection **0.0** | `pbt_consolidation_schedule.effective_mutation_rate` + stage via `curriculum_progress` @ 0.50 / 0.85 |
| Rescue / freeze | precise copy, mutation disabled on adaptive rescue path | `pbt_copy_events.jsonl` executed event |
| AWR / RSSM-MPPI / residual rollout / WM profile copy | **False / off** | launcher + profile |
| Fixed 2cm ladder / bridge probation / late fixed-threshold rescue | **False** | profile turns bridge/same_tol/late_rescue off |
| Readiness-adaptive / WM graduation | **Profile 23 only** | `_CSER_DEXPBT_READINESS_ADAPTIVE`; not in Jul12 expansion |
| `snippet_distill_weight` | **0.001** | `_CSER_DEXPBT_CONSERVATIVE` |
| `eir_actor_distill_weight` (fallback) | **0.0005** | profile expansion in train.log |
| Progress ramp | start **300M**, ramp **300M** | `progress_refinement_start_after_frames` / `ramp_frames` |
| Bank size | **50000** | `elite_snippet_bank_size` |
| `tolerance_max_log_gap` | **0.20** | profile |
| `progress_refinement_max_active_fraction` | **0.35** | profile |
| Adaptive consol stages | consolid progress **0.50**, selection **0.85** | profile |
| Probation | min frames **200M**, stable ticks **4**, retention **0.90** | profile |
| Rescue cooldown | **300M**; freshness hard max **40M** | profile |

Curve fields confirm runtime: `allow_mid_self_mutation=False`, `disable_self_restart=True`, `wm_strategy_profile_id=22`, `profile_elite_bank_enabled=True`, `profile_actor_distill_enabled=True`, `pbt_wm_profile_copy_enable=False`.

---

## 3. End-to-End Runtime Path

```text
Rollout → PPO update
       ↘ EliteSnippetBank insert (elite/mid buckets)
       ↘ progress_refine gate (≥300M) → retrieval (log-relative tol) → teacher
            ↘ snippet_wm_rerank (usually fallback: belief_missing)
            ↘ snippet_distill loss (weight 0.001) → actor
       ↘ fallback actor_distill (weight 0.0005) — inactive when snippet path present

PBT interval (~20M, stage-scaled):
  tick schedule → ternary tier
    top: no restart
    mid: attempt self-mutation path → blocked (disable_self_restart)
    bottom: target gates + consolidation stage gates → source cascade
      → rarely: adaptive rescue / ordinary copy → restart transaction
Curriculum mastery/adaptive-fast shrink on local windows (independent of PBT)
Bank merge via eir_broadcast (population sharing without weight copy)
```

Code anchors: `pbt_observer.py` (ternary / mid block / restart), `pbt_consolidation_schedule.py` (stages), `snippet_distill` / `progress_refine` / `snippet_wm_reranker.py`, elite bank + `eir_broadcast_events.jsonl`.

---

## 4. PBT Utilization

### Tick schedule (all policies’ `pbt_tick_debug_*.jsonl`)

| Metric | Value |
|---|---|
| Debug lines | 150,734 |
| `should_tick=true` | **348** (≈58 ticks/policy over ~2.3–2.5B) |
| Real skips | `initial_delay` 140085, `before_start_after` 9156, `interval_not_elapsed` 1145 |
| Stage **labels** on ticks (not skips) | exploration 261, consolidation 82, selection 5 |

Note: when `should_tick` is true, `skip_reason` is rewritten to `stage_*` (`pbt_consolidation_schedule.evaluate_consolidated_tick`). Do **not** treat those as blocked ticks.

### Tier occupancy (training_curve samples, 40M cadence)

| Tier | Curve points | Share |
|---|---|---|
| top | 56 | 16.1% |
| mid | 171 | 49.1% |
| bottom | 121 | 34.8% |

### Replacement / copy reality

| Event | Count |
|---|---|
| Adaptive rescue **selected** | 1 (p1→p2 @ 1000M, ordinary_fallback) |
| Adaptive rescue **executed** | **1** (`mutation_disabled=true`) |
| Ordinary bottom cross-copy events | **0** |
| Manifest committed | 1 |
| `weight_copy_count` on curve endings | **0 for all policies** (counter reset on process restart and/or rescue accounting path) |
| Self-restarts executed | **0** |
| Self-restart **blocked** | **98** (= mid_self_mutation attempts) |
| Best-stable recovery executed | 0 |
| Adaptive sidecar rescue attempts | p2: **9** attempts, **8** blocked, **1** executed; others 0 |

**Copy execution rate (proxy)**

\[
\frac{N_{\mathrm{executed\ bottom\ copy}}}{N_{\mathrm{estimated\ bottom\ opportunities}}}
\approx \frac{1}{121} \approx 0.83\%
\]

where opportunities ≈ bottom share × should_ticks (coarse). Finer per-tick target-gate pass/fail reasons are **not observable** from current tick debug (no reject taxonomy logged there).

### Source tier counters on curves

Final `source_tier_strict/relaxed/archive/skipped` are **all 0** on every policy → either never incremented on successful paths or wiped on restart. Only durable evidence of a live source hit is the copy event `tier=strict_stable_live` once.

### Skip / block evidence that *is* observable

| Class | Evidence | Count / note |
|---|---|---|
| Mid mutation hard-off | `self_restart_blocked_count` / `mid_self_mutation_count` | 98 / 98 |
| Adaptive rescue blocked | p2 sidecar `rescue_blocked_count` | 8 |
| Rescue last reason often | `retention_ok` | means non-rescue ticks, not a skip taxonomy |
| Consolidation / selection bottom gates | code exists (`should_allow_bottom_restart`) | **hit counts not observable** in Jul12 JSONL |
| Promising / cooldown / tol incompatible / stale source | would need observer audit logs | **not observable from current logs** |

### Answers (PBT)

1. **Is Profile 22 degenerate to 6 independent PPOs?**  
   **Mostly yes for weight-level search/propagation.** Curriculum + bank broadcast couple workers weakly; **weight copy almost never happens**.

2. **PBT’s real role?**  
   **Occasional collapse recovery** (one rescue), **not** continuous population search or elite propagation. Stage schedule also throttles late replacements (`stage3_hard_collapse_only`).

3. **Bottleneck for rare copies?**  
   Mix of **(a)** mid hard-off removing diversity churn, **(b)** consolidation/selection gates (code-level, counts not logged), **(c)** adaptive rescue patience/cooldown (8/9 blocked on the weak worker), **(d)** few workers bad enough / long enough to arm rescue while good sources exist. Target-gate skip table is incomplete → missing observability.

4. **Where do new hyperparameter branches come from with mid mutation off?**  
   Essentially **initial seed/env diversity only**. Ordinary bottom mutations did not execute; exploration-stage mutation never fired because no ordinary bottom copy succeeded.

5. **Ordinary bottom mutations executed?**  
   **0** in `pbt_copy_events.jsonl` (the only executed replacement disabled mutation).

---

## 5. Curriculum Dynamics

### Shrink totals (tol decreases on curves)

| Policy | Shrinks | Factors seen | Mastery-like (`mean≥4` & `max≥20` at pre-shrink) | Mean nsucc retention @100M after shrink |
|---|---|---|---|---|
| p0 | 19 | 0.90 / 0.925 / 0.95 | 19 | **1.03** |
| p1 | 15 | 0.90 / 0.925 / 0.95 | 15 | **1.03** |
| p2 | 4 | 0.548 (copy inherit) + 0.90 | 3 | ~1.00 (post-rescue) |
| p3 | 10 | 0.90 / 0.925 / 0.95 | 10 | **1.16** |
| p4 | 19 | 0.90 / 0.925 / 0.95 | 19 | **1.06** |
| p5 | 17 | 0.90 / 0.925 / 0.95 | 17 | **1.04** |

Adaptive-fast stronger shrinks (0.90 / 0.925) **did fire**; 0.95 is baseline.  
`shrink_then_drop_cycles` (ret&lt;0.5 within 200M) ≈ **0–1**, i.e. **Jul12 curriculum did not produce cascading shrink→collapse→rollback loops**.

Final tols reached ~0.012–0.015 on strong workers with nsucc still ~25–40 → curriculum progressed to high precision.

### Probation / rollback / hard-collapse

- End-state adaptive sidecars: probation_active true on most workers; **rollback_count=0**, hard_collapse curve flags ≈0.  
- Collapse→recover-to-peak timing: **not separable** without event timestamps beyond curve flags.  
- Mastery gate dependency on short peaks: shrinks occur with `nsucc_max=50` often while mean ≪ max early on, but **post-shrink mean retention stayed high** on Jul12 — so over-aggressiveness is **not** supported by Jul12 retention stats (may still hurt tip rate vs “wait longer”).

### Curriculum judgements

| Question | Verdict |
|---|---|
| Shrink faster than capability formation? | **Not on Jul12** (post-shrink retention ≈1) |
| `nsucc_max` with low mean? | Early yes; later both high |
| Adaptive-fast 0.90/0.925 harmful? | Factors used, but retention held |
| Probation mainly stalls? | Active late; **cannot quantify duration distribution** from retained sidecars alone |
| PBT helped after shrink collapse? | Only **one** rescue; curriculum mostly self-healed |

---

## 6. Snippet and Distillation Utilization

### Jul12 durable evidence

- `profile_elite_bank_enabled=True`, distill enabled in expansion.  
- `eir_broadcast_events.jsonl`: **56 broadcast / 53 consume**, `entries=50000` — banks fill and merge across workers **without** weight copy.  
- Curve JSONL **lacks** progress_gate / ESS / retrieval fractions → those must come from TB.

### Proxy TB (Jul14 same profile) — late-window

| Signal | p0 late | p1 late | Interpretation |
|---|---|---|---|
| `elite_snippet/bank_size` | →50k | →50k | bank saturates |
| `elite_snippet/bank_size_mid` | 50k | 50k | **mid bucket monopoly** |
| `bank_size_hard` / `floor` | 0 | 0 | little hard/floor diversity |
| `progress_refine/active_frac` | ~0.30 | ~0.28 | near `max_active_fraction=0.35` |
| `progress_refine/retrieval_valid_frac` | ~0.34 | ~0.33 | ≈ budget; valid≈budget |
| `progress_refine/history_valid_frac` | ~0.50 | ~0.50 | history gate removes ~½ |
| `progress_refine/gate_p50` | ~0 | ~0 | half rows gated off hard |
| `progress_refine/ess` (effective_sample_size) | ~1250 | ~1150 | non-trivial ESS when active |
| `snippet_distill/loss` | ~1.8e-5 | ~2.9e-5 | **tiny** |
| `losses/a_loss` (PPO) | ~5.2e-3 | ~4.6e-3 | **~150–300× larger** than snippet loss |
| `snippet_retrieval/no_compatible_frac` | **0** | **0** | log-gap gate not the killer |
| `eir/actor_distill_*` active/loss | **0** | **0** | fallback distill inactive |
| `actor_distill_fallback_with_snippet_used` | 1 | 1 | diagnostic: snippet path owns distill |

**Effective refinement rate (proxy):** `snippet_distill/active_frac` late ≈ **0.11–0.16** averaged over all steps (including pre-300M zeros); when refine logging runs (~every N steps), `progress_refine/active_frac≈0.30`.

**Which gate removes most mass (proxy):**  
1. progress / history / oscillation structure (`gate_p50≈0`, history_valid≈0.5)  
2. `max_active_fraction=0.35` caps retrieval (`retrieval_*≈0.33`)  
3. **not** `no_compatible` (≈0)

**Gradient fiction warning:** no per-layer grad norms logged. Conclusion is **loss-scale**: snippet weighted loss ≪ PPO actor loss → updates are likely **PPO-dominated** even when ESS&gt;1000.

---

## 7. World Model Utilization

### Floor-recovery / ranking score (Jul12 curves)

- `wm_score_used` true on ~42–57% of curve points; `wm_score_effective_weight` mean **0.001–0.006**, **zero-weight ~45–48%** of points.  
- `wm_shadow_changed_selection / observations` ≈ **28–36 / 31–40** (~85–95% shadow “would change”) — but shadow ≠ applied decision unless score used with nonzero weight.  
- `pbt_use_wm_recovery_score=True` configured; whether it **changed a chosen source** except via unused path: **not observable** (no counter for “WM flipped source”).

### Snippet WM rerank (proxy TB)

| Metric | Value |
|---|---|
| `snippet_wm_rerank/changed_frac` | **0** always |
| `row_active_frac` | **0** |
| `fallback_frac` / `row_fallback_frac` | high / **1.0** |
| `fallback_reason_id` | **2 = belief_missing** (`snippet_wm_reranker.py`) |

So WM **does not** reorder teachers in practice on the proxy run: belief missing → priority-only fallback. Calibration AUC/ECE time series: **not found as first-class TB tags in the prefer set / not summarized here** → treat detailed calib pass rates as **not fully observable** without a dedicated calib tag sweep.

### Answers (WM)

1. Most of training: **not meaningfully reranking snippets**.  
2. Effectively `alpha` path inactive (`row_active_frac=0`) due to **belief_missing**, not only ECE gates.  
3. Teacher order unchanged → **no distill impact from WM rerank**.  
4. Yes — closer to **diagnostic + optional floor-recovery soft score**.  
5. Floor-recovery source flips: **not observable**.  
6. Candidate set = actor + snippet first-actions (code/tests) → even if belief were present, quality ceiling is structurally limited; currently blocked earlier by belief.

---

## 8. Elite Bank Coverage

| Finding | Evidence |
|---|---|
| Bank fills | Jul12 broadcasts + proxy `bank_size→50000` |
| Mid-bucket monopoly | `bank_size_mid=50000`, hard/floor **0** |
| Cross-worker persistence/merge | Jul12 `eir_broadcast` 56/53 with 50k entries |
| Propagation with weight copy | **Almost never** (1 rescue); bank merge is the real coupling |
| Tol-compat fail-closed floods | **no** (`no_compatible_frac=0` proxy) |
| Monopoly by few snippets | priority histograms / reuse rates | **not observable from current logs** |
| Early-tol experience dominating late | mid-only fill suggests **stage coverage hole**, not proven early-tol bias without per-snippet tol dumps |
| Length-4 first-step distill → multi-step recovery | action_delta WM=0; no multi-step eval metric | **not observable**; architecture implies **local first-step bias** only |

---

## 9. Bottleneck Ranking & Module Grades

### Grades

| Module | Grade | Note |
|---|---|---|
| PPO | **A** | Dominates loss |
| Mastery curriculum | **A** | Frequent shrinks, good retention on Jul12 |
| Adaptive-fast curriculum | **A/B** | 0.90/0.925 used; not destabilizing on Jul12 |
| EliteSnippetBank fill/broadcast | **B** | Clearly executes; coverage skewed mid |
| Progress-conditioned retrieval + distill | **B/C** | Runs with ESS~1k & active~30%, but loss ≪ PPO |
| Consolidation schedule | **B** | Stages labeled; late hard-collapse-only likely suppresses copies |
| Probation | **B** | Sidecars show active; duration stats weak |
| Adaptive rescue | **C** | 1/9 on weak worker |
| PBT source selection / ordinary bottom copy | **C/D** | Effectively unused for weights |
| Hyperparam mutation | **D** | Mid off; no ordinary copies to mutate |
| Fallback actor distill | **D** | loss≈0 when snippet on |
| WM snippet rerank | **D** | belief_missing fallback |
| WM floor-recovery score | **C** | soft presence; flip not logged |
| Rollback | **D** | count 0 |
| Readiness adaptation | **E** | Profile 23 |
| AWR / RSSM-MPPI / residual | **E** | hard off |

### Top 3 real bottlenecks limiting P22

1. **PBT weight-propagation near-zero** → population cannot copy mature elites onto chronic weak workers (except 1 rescue). Supports the thesis that bottom copy budget/gates are too conservative.  
2. **Snippet distill loss scale vs PPO** (`~1e-5` vs `~5e-3`) → even with healthy gates/ESS, refinement is a **light regularizer**, not a primary skill driver.  
3. **WM rerank dead path (`belief_missing`)** + **bank mid-only coverage** → method narrative (WM-guided teachers / stage-diverse bank) ≠ runtime.

Honorable mention: mid mutation off + no bottom copies → **no ongoing hyperparam exploration**.

---

## 10. Missing Observability

Must add on next Profile 22 formal run (persist besides TB):

1. Per PBT tick: `tier`, `bottom_opportunity`, `target_gate_pass/fail + reason`, `source_attempts`, `source_tier_hit`, `mutation_applied`, `transaction_result`.  
2. Cumulative durable counters that **survive process restart** (copy/rescue/source hits).  
3. Snippet: `N(w>0)/N_rows`, gate kill histogram (lifted/history/proximity/stall/osc/tol/confidence), ESS, loss, weighted loss — **into training_curve JSONL**, not only TB.  
4. WM: calib pass/fail, AUC/ECE, `alpha_wm`, belief_missing rate, `changed_frac`, floor-recovery “score flipped argmax”.  
5. Bank: per-tol coverage, reuse entropy, hard/floor insert rates, post-restart bank hash.  
6. Curriculum: explicit shrink event log (reason, factor, peak, probation enter/exit frames).  
7. Grad norms optional: `||g_snippet||/||g_ppo||` once per N updates.

Jul12 TB deletion also shows **analysis fragility** — archival policy should keep `events` or JSONL mirrors.

---

## 11. Recommended Next Experiment

### Direct answers

1. **Top bottlenecks:** (1) near-zero PBT copies, (2) distill ≪ PPO, (3) WM rerank / bank stage coverage dead.  
2. **“PBT useless because bottom copies rare”?** **Supported** — 1 executed replacement in a full multi-billion-frame run; copy rate proxy &lt;1%.  
3. **Actual algorithm ≈**  
   `6 independent PPO + mastery curriculum + sparse snippet distill + occasional PBT recovery` — **yes**.  
4. **Configured-on but low utilization:** ordinary PBT copy, hyperparam mutation, WM snippet rerank, fallback actor distill, late selection replacements.  
5. **Over-conservative gates:** mid self-mutation hard-off + consolidation/selection bottom gates + adaptive rescue patience/cooldown (exact share of each **under-logged**). Tol log-gap **not** over-firing in proxy.  
6. **Best single interpretable next change:** **Guarded elite propagation** (Profile 24-style): keep P22 ramp/distill schedule, add **strict, budgeted mature→weak weight copies** after ~600M — directly attacks bottleneck #1 without re-enabling wild mid mutation. Alternative single knob: raise `snippet_distill_weight` 3–10× **only after** logging ESS/active_frac (interpretable but may fight PPO).  
7. **Must-log next rerun:** section 10 list; especially PBT reject reasons + snippet gate kill chart + WM belief/alpha.

### Concrete next A/B

| Arm | Change | Freeze |
|---|---|---|
| A | Profile 22 baseline | — |
| B | P22 + guarded elite propagation (≤2 copies/(tick,stage), source nsucc≥30, protect top-2, lineage floor≥2) | distill weights, curriculum mastery thresholds, env count |
| Optional C | B + snippet_distill_weight 0.003 | only if B’s tip still lags Jul12 peak density |

Do **not** re-enable readiness early refine (P23 lesson) as the first patch.

---

## Appendix A — Artifact index

| Path | Role |
|---|---|
| `audit_profile22.py` | Read-only auditor |
| `audit_raw.json` | Full JSON dump |
| `audit_stdout.json` | Script stdout excerpt |
| Jul12 `dexpbt_8gpu_ws/metrics/*.jsonl` | Primary PBT/curriculum/bank evidence |
| Jul12 `adaptive_precision/*.json` | Rescue/probation sidecars |
| Jul14 `**/summaries/events*` | Proxy snippet/WM TB |

## Appendix B — Status legend used

- **configured on** — launcher/profile/defaults  
- **runtime eligible** — conditions met at least sometimes  
- **actually executed** — durable event / nonzero training effect
