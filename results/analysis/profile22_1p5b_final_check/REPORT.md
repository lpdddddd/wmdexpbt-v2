# Profile 22 Final Check @ ~1.5B

**Run:** `regrasping_cser_dexpbt_progress_refine_5b_20260714_092846`  
**Commit:** `b3cf704917318973f34b45916462c800a450f47b` — *Fail-close when restart succeeds but ledger finalize cannot record.*  
**Mode:** readonly (no code/config/ckpt/sidecar changes; training left running).  
**Artifacts:** `latest_worker_snapshot.csv`, `p2_recent_pbt_ticks.csv`, `source_feasibility.csv`, `root_cause_status.json`, `audit_1p5b_final.py`

Evidence tags: **observed** · **reconstructed** · **not observable**

---

## 1. Run identity & integrity

| Item | Value | Evidence |
|---|---|---|
| Git HEAD | `b3cf704…` | **observed** |
| Profile | id **22**, `cser_dexpbt_progress_refine` | **observed** curve + launcher |
| `pbt_allow_mid_self_mutation` | **False** | **observed** curve + `--no-pbt_allow_mid_self_mutation` |
| `pbt_disable_self_restart` | **True** | **observed** |
| MPPI / RSSM reranker | off (`eir_mppi_mode=off`, `no-eir_use_rssm_mppi`) | **observed** launcher / process argv |
| Workers alive | master + 6 train (match `092846`) | **observed** |
| Curve continuity | no regression / duplicate frames / large gaps | **observed** `integrity.ok=true` |
| Worker PBT restarts | `restart_count=0`, `weight_copy_count=0` all policies | **observed** |
| Supervisor restarts | training_pid continuous ~11h; metadata `max_restarts=1000` but no evidence of mid-run train restart in curves | **observed** / no curve discontinuity |

Latest checkpoints (workspace `00i/*.pth`, mtime local):

| p | ckpt | mtime (approx) |
|---|---|---|
| 0 | 000062.pth | 20:17 |
| 1 | 000056.pth | 20:32 |
| 2 | 000074.pth | 20:40 |
| 3 | 000064.pth | 20:35 |
| 4 | 000062.pth | 20:23 |
| 5 | 000058.pth | 20:03 |

Frame skew across workers is expected (p2 tip ~1.48B, p1 ~1.12B). Same-frame 1.50B does **not** exist for all six; population cross-section below uses **nearest ≤~1.5B per worker** and is labeled as such — not a pure equal-frame slice.

**Identity anomalies:** none that invalidate trend conclusions.

---

## 2. 1.4B → ~1.5B incremental status

### Tip / near-1.5B population (per-worker nearest; **not** equal-frame)

| | best | median | mean | spread (max−min) |
|---|---|---|---|---|
| Now (tips ~1.12–1.48B) | **37.81** (p4) | **30.59** | **31.43** | **10.55** |
| ~100M earlier (reconstructed) | — | — | — | **15.18** |

Spread **narrowed** (~15.2 → 10.5) mainly because **p2 climbed**.

| p | frame | nsucc | Δnsucc≈100M | slope /100M | tier | tol | promising |
|---|---|---|---|---|---|---|---|
| 0 | 1240M | 30.44 | +0.43 | ~0.4 | mid | 0.0294 | True |
| 1 | 1120M | 29.45 | −0.50 | ~−0.5 | mid | 0.0294 | True |
| **2** | **1480M** | **30.74** | **+6.74** | **~+5.6** | **bottom** | **0.0551** | False |
| 3 | 1280M | 32.86 | +2.67 | ~+2.7 | mid | 0.0319 | True |
| 4 | 1240M | 37.81 | −0.64 | ~−0.6 | top | 0.0263 | True |
| 5 | 1160M | 27.26 | +3.99 | ~+4.0 | bottom | 0.0353 | True |

### p2 trajectory answers

1. Near-1.5B best/med/mean: **37.81 / 30.59 / 31.43** (skewed frame set).  
2. Recent 100M: p2 **+6.74 nsucc** — **clear improvement**, not stall / not collapse.  
3. Still **bottom** (**observed** tier). Contiguous bottom since **360M → 1480M = 1.12B frames** (**reconstructed** from curve tier run).  
4. Population spread **shrinking**.  
5. Ranking still weak: tip ranking score ≈0.81 vs p4 ≈1.25 (**observed** / feasibility table) despite nsucc≈median — curriculum/success-aware score keeps p2 bottom.

Fields mostly **not observable** on curve: `curriculum_progress`, per-tick `skill_delta` (TB only when bottom), probation details beyond APC tip. TB window 1.4–1.5 for p2 is available (below).

---

## 3. PBT final confirmation

### Counts

| Metric | Full run tip | 1.4–1.5B window |
|---|---|---|
| p2 `should_tick` | 33 **observed** | **3** **observed** |
| p2 bottom opportunities | 31 **observed**+recon tier | **3** |
| Target gate allow | **0** (**observed** TB `copy_allowed=0`; **reconstructed** reasons) | **0** |
| Ordinary `_select_source` entered | **0** | **0** |
| weight / cross / restart | **0 / 0 / 0** | **0** |
| Adaptive rescue attempt/selected/executed | **0/0/0**, reason `retention_ok` | still 0 |
| Mid self-mutation entered/blocked (p2) | 3 / 3 | no new mid entries in window (**observed** counters flat for p2) |
| Mid mut other workers | entered=blocked (12–21) | configs still block self-restart |

### 1.4B–1.5B p2 ticks (**observed** TB + **reconstructed** reason)

| step | skill_delta | valid | copy_allowed | reconstructed reason |
|---|---|---|---|---|
| ~1.41B | −999 | 0 | 0 | `skill_delta_invalid:tol_bins:[0.055132, 0.058034]` |
| ~1.45B | −999 | 0 | 0 | same |
| ~1.49B | −999 | 0 | 0 | same |

All three still return inside **`pbt_copy_allowed_for_target`** (`pbt_observer.py:10055–10108`) **before** `_select_source`.

**Never entered `_select_source`.** No new PBT behavior overturns `3.md` / root-cause review.

Adaptive rescue: APC tip `degradation_streak=0`, `rescue_last_reason=retention_ok` — **confirmed** still not armed despite chronic bottom (no retention collapse).

---

## 4. Source feasibility (if target gate opened)

Latest p2 bottom tick ≈ **1490M** (`should_tick`). Candidates:

| src | tier | nsucc | score_gap vs p2 | curve eligible | recon skill_Δ | post-select live gate |
|---|---|---|---|---|---|---|
| 0 | mid | 30.4 | +0.40 | ok | **invalid tol_bins** | **likely reject** |
| 1 | mid | 29.5 | +0.40 | ok | invalid tol_bins | likely reject |
| 3 | bottom* | 32.9 | −0.04 | ok | invalid | likely reject |
| 4 | **top** | 37.8 | +0.44 | ok | invalid | likely reject |
| 5 | mid | 27.3 | +0.09 | ok | invalid | likely reject |

\*rank tier from score sort at tick — **reconstructed**.

**Core answer:** Opening only the chronic-bottom **target** gate does **not** guarantee a copy. After selection, live sources still hit `pbt_copy_allowed_for_live_source` (`skill_delta ≥ 0.02`, `pbt_control_plane.py:506–514`). At this tick **all five peers have skill_delta invalidated by mixed tol bins** (**reconstructed**, matches the same causality rule as target). Outcome: **high probability that existing source cascade still ends in skip**, unless an archive tier fires (archive bypasses live skill_delta — **code**; archive hit rate **not observable** without that path executing).

Telemetry required next experiment (must-have):

1. `target_gate_reason` string every bottom tick  
2. `source_cascade_tier` / `action` / `skip_reason`  
3. per-candidate `source_filter_reason` + `live_source_skill_delta` + validity  
4. `continuous_bottom_frames` (new state)  
5. `copy_executed` / checkpoint_type / mutation_applied bool  

Full L1 drop-ratio internals for `is_stable_source_window`: **not observable** today.

---

## 5. p2 bank final confirmation

Latest `002/000074.pth` `eir_elite_snippet_bank` (**observed**):

| Field | Value |
|---|---|
| size / schema | 50000 / v2 |
| sources | p0: 37770 · p1: 12230 · **local p2: 0** |
| success_tol unique | {0.0294, 0.0326, 0.0363, 0.0382} — all ≪ query |
| tip query tol | **0.05513** @ 1480M |
| compat@0.20 | **0 / 50000 (0%)** |
| δ_ε med/min/max | 0.42 / 0.37 / 0.63 |
| TB `no_compatible_frac` 1.4–1.5 | **1.0** all samples |
| `active_frac` / ESS | **0 / 0** |
| Whole-bank consume | **yes** — `source=0 → target=2`, entries=50000 @ 1440M |

**Effective capacity ≈ 0** despite 50k rows. Still fully consistent with root-cause report.

---

## 6. WM final confirmation

| Worker | `mppi_belief_valid` | `alpha_wm` | `calibration_pass` | reason_id | `changed_frac` |
|---|---|---|---|---|---|
| p0 (strong) | 0 | (TB tag sparse on p0 for alpha; reason 2) | — | **2** belief_missing | 0 |
| p2 | **0** (full window) | **0** (1942 pts in 1.4–1.5) | **0** | **11** | 0 |
| p4 | 0 | **~0.24** | 1 | **2** | 0 |

Answers:

1. Strong workers still reason **2**.  
2. p2 still reason **11**.  
3. p2 `alpha_wm` **never** >0 in 1.4–1.5 (**observed**).  
4. No new belief-path activity (`mppi_belief_valid≡0`).  
5. No evidence overturning “Profile22 WM rerank wiring inactive.”

---

## 7. Go / No-Go

### **B. GO WITH TELEMETRY**

Minimum conditions vs data:

| Condition | Met? |
|---|---|
| p2 still long bottom | **Yes** (1.12B contiguous) |
| p2 weight_copy ≈ 0 | **Yes** (0) |
| p2 snippet refine useless | **Yes** (compat 0, ESS 0) |
| p2 WM rerank useless | **Yes** (α=0, reason 11) |
| Run continuous / identity OK | **Yes** |
| No new confounders of identity | **Yes** |
| Uncertainties | Source live-Δ gate may still kill copies; full ckpt copy also pulls bank/WM/optimizer |

Do **not** also patch WM or bank in experiment A.

---

## 8. Experiment A single-variable boundary (analysis only)

### Suggested trigger
```text
tier == bottom
AND continuous_bottom_frames >= 200M
AND ≥ N bottom PBT ticks in that span
AND not in copy cooldown / elimination burn-in
→ allow entering existing _select_source (bypass ordinary stuck gate only)
```

| Question | Finding |
|---|---|
| 1. `continuous_bottom_frames` exist? | **No** dedicated state today (**code search empty**). Only reconstructible from curve `tier` history. **Must add** observer field + persist in workspace JSON / APC / control for resume. |
| 2. Restart/resume correctness? | Today unused. If added, reset on successful cross-copy restart; restore from JSON on resume; do **not** reset on mid-tick observer restart-of-process without tier change. |
| 3. Insertion layer | After promising/floor/consolidation gates; **replace or OR with** `pbt_copy_allowed_for_target` in `_ternary_pbt` bottom branch (`pbt_observer.py ~10054–10108`). Do **not** bypass promising / elimination cooldown / live-source skill_delta / source eligibility cascade. |
| 4. Bypass risk | Chronic gate must **not** set `rescue_mode` (would change adaptive path + freeze mutation). Keep `rescue_mode=none`. |
| 5. Exact copy scope | Current ordinary path still calls `_mutate_params(..., mutation_rate=0.2)` unless rescue/frozen (`~11175–11192`). For exact copy must force `new_params=dict(source_params)` on this chronic path. Checkpoint `shutil.copyfile` loads **full** `.pth` including **model + optimizer + `eir_elite_snippet_bank` + `wm_calibration_state`** (`train.py` restore ~5158+). |
| 6. Accidental copies | **Yes risk:** bank + WM cal + optimizer come along with weight copy — secondary channels change even if eligibility is the only *algorithmic policy* change. Curriculum via `source_safe` inherit remains separate. |
| 7. Single variable discipline | Change **only** target eligibility predicate + mutation_rate=0 for that path + telemetry. Leave mid-mut / adaptive rescue / WM / bank / curriculum / PPO hparams untouched. Accept bank overwrite as inherited property of existing restart transaction (document as known secondary effect). |

---

## 9. Summary table

| Issue | ~1.44B conclusion | ~1.5B status | Evidence | Changed? | Implication |
|---|---|---|---|---|---|
| Ordinary PBT copy dead | target gate blocks before `_select_source` | **Still blocks**; 1.4–1.5 all `tol_bins` invalid | TB copy_allowed=0; replay CSV | No | Exp A still addresses primary fail-close |
| Adaptive rescue | retention_ok / 0 attempts | **Same** | APC sidecar | No | Don’t touch for A |
| p2 bank incompat | compat 0%, foreign low-tol | **Same** (+ more p0 share); consume 50k @1440M | ckpt bank + events | No (worse fill) | Leave bank alone for A |
| WM rerank | belief missing / α=0 | p0/p4 reason **2**; p2 reason **11**, α=0 | TB | No | Leave WM alone for A |
| p2 performance | weak climber | **Faster climb** (+6.7/100M), nsucc≈median but still bottom | curve | Yes (skill↑) | Still chronic-bottom by rank |
| Source cascade if gate open | unknown | **Likely all live sources fail skill_Δ** | reconstructed feasibility | New detail | Telemetry + interpret null copies carefully |

---

## Final decision (≤5 sentences)

Baseline is **healthy and continuous** (Profile 22 identity intact, no frame regressions, all 6 workers training). The **three inactivation chains still hold** at ~1.5B: PBT target-gate block, bank compat=0, WM rerank inactive. **Start single-variable experiment A (chronic-bottom → existing source selection, exact copy)** as **GO WITH TELEMETRY**. Before launch, add the minimal target/source reject + continuous-bottom + copy-executed logs listed in §4 (otherwise null results cannot distinguish “gate never opened” vs “source skill_delta rejected everyone”). **No need to hold baseline to 2B** for this decision — 1.5B evidence is sufficient; optionally keep baseline running as a parallel control if compute allows.
