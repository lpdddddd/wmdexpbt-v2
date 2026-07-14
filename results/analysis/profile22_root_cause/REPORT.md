# Profile 22 Root-Cause Review (readonly)

**Run:** `regrasping_cser_dexpbt_progress_refine_5b_20260714_092846`  
**Scope:** p2 bottom PBT deadness · WM `belief_missing` / inactive · bank `no_compatible_frac=1`  
**Constraint:** training left running; no code/config/ckpt/sidecar mutations.  
**Artifacts:** `audit_root_cause.py`, `p2_pbt_tick_replay.csv`, `p2_bank_*.csv`, `root_cause_raw.json`

Evidence tags used below: **observed** · **reconstructed** · **not observable**

---

## 1. Executive Summary

Three independent-but-compounding fail-closes:

1. **Ordinary PBT never copies p2** because every bottom tick returns before source selection on `pbt_copy_allowed_for_target` — stuck requires `nsucc_mean < 1` **and** `skill_delta ≤ 0`. p2 either had a slightly positive skill_delta while nsucc&lt;1, became “improving,” or invalidated skill_delta via curriculum tol changes. **Observed** TB: `pbt/skill_delta_copy_allowed=0` on all 28 bottom ticks; `source_tier_*_count=0`.
2. **WM snippet rerank is structurally belief-less** on Profile 22: MPPI/RSSM reranker off ⇒ `_need_belief=False` ⇒ belief cache never built; `eir/mppi_belief_valid=0`. Strong workers hit reason **2** after `alpha_wm>0`; p2 stays reason **11** because `wm_cal/alpha_wm=0`.
3. **p2’s 50k bank is 100% foreign low-tol snippets** (p0≈65%, p1≈35%, local=0); vs tip query tol≈0.055, `compat@0.20=0%`, so progress refine ESS/active stay 0.

p2 is in a confirmed recovery loop of **local PPO only**, with population coupling limited to EIR broadcasts that *hurt* retrieval compatibility.

---

## 2. P2 Timeline (observed curve)

| Frame | tier | nsucc | tol | global_ret | peak | notes |
|---|---|---|---|---|---|---|
| 200–320M | top→mid | ~0 | 0.075 | 1.0 | ~0 | mid self-mut blocked ×3 |
| **360M** | **bottom** | 0 | 0.075 | 1.0 | ~0 | bottom span starts |
| 400–480M | bottom | 0.004→0.87 | 0.075 | 1.0 | rising | nsucc&lt;1 but Δ&gt;0 |
| 520M | bottom | 1.19 | 0.075 | 1.0 | 1.19 | exits stuck band |
| 560M | bottom | **0.34** | 0.075 | **0.29** | 1.19 | only soft retention dip |
| 600–800M | bottom | 2.6→12.9 | 0.075 | ~1 | rising | climbing, never peak-collapse |
| 840M+ | bottom | 14→25 | shrinking | ~0.85–1 | rising | tol changes every ~40–80M |
| tip ~1.44B | bottom | ~25.2 | ~0.055 | 1.0 | 25.2 | still bottom; still no copy |

`promising_protect_until=0` entire span (**observed**). `weight_copy_count=0`, `restart_count=0`.

---

## 3. PBT Decision Replay

CSV: `results/analysis/profile22_root_cause/p2_pbt_tick_replay.csv`

| Metric | Value | Evidence |
|---|---|---|
| `should_tick` total (p2) | 32 | **observed** tick JSONL |
| Real bottom opportunities (360M→tip) | **28** | **observed** tier=bottom ∩ should_tick |
| Ordinary source cascade entered | **0** | **reconstructed** + **observed** counters |
| `pbt_copy_events.jsonl` | missing / empty | **observed** |
| Adaptive rescue attempt/selected/executed | 0/0/0 | **observed** APC sidecar |
| Mid self-mutation entered / blocked | 3 / 3 | **observed** curve |

### Target-gate reason histogram (reconstructed; TB-validated where overlapping)

| Reason | Count / 28 | Mechanism |
|---|---|---|
| `not_stuck` | **9** | Δ&gt;0 while crawling up, or nsucc≥1 |
| `skill_delta_improving` (Δ≥0.05) | **2** | 760–800M climb |
| `skill_delta_invalid:short_window` | **1** | first bottom tick |
| `skill_delta_invalid:tol_bins…` | **16** | shrinks keep 6-tick window mixed |

**Observed TB match** (`policy_02`):

| step | skill_delta | valid | copy_allowed |
|---|---|---|---|
| 360M | -999 | 0 | 0 |
| 400–720M | +1.7e-5 → +0.047 | 1 | **0** |
| 760–800M | +0.087 → +0.094 | 1 | **0** |
| 840–1440M | -999 | 0 | **0** |

Reconstructed Δ equals TB to ~1e-6 — offline replay is trustworthy for these ticks.

### Call-chain position of the block

```text
_ternary_pbt  (pbt_observer.py:9717)
  → tier=bottom (9806)
  → promising? NO (observed until=0)          # skipped
  → floor-recovery? no evidence of arming     # not observable / no return signal
  → should_allow_bottom_restart               # exploration → True (code)
  → pbt_copy_allowed_for_target (**BLOCK**)   # 10027–10108
  → ★ NEVER reaches _select_source            # 10142+
```

Config gates confirmed live: `allow_mid_self_mutation=False`, `disable_self_restart=True`.

---

## 4. Exact PBT Blocking Condition

### Must-answer sheet

1. **Bottom opportunities:** 28 real bottom PBT ticks in the ~360M–1440M span (~1080M duration; earlier “640M” was a partial lower bound).
2. **Where each returns:** all 28 at **target skill-delta gate** (`pbt_copy_allowed_for_target`); never at promising / never at source skip.
3. **Exploration allows bottom restart?** **Yes** — `should_allow_bottom_restart` returns `(True,"")` when `stage==exploration` (`pbt_consolidation_schedule.py:150–151`). Stage remains exploration while curriculum_progress≪0.5 (**likely** / TB curriculum_progress on strong workers ~0.3 earlier).
4. **Promising / retention / cooldown / probation protect p2?** Promising **No** (observed). Elimination cooldown uses `frames_at_restart`; no restart ⇒ eligible anyway. Probation active late for *curriculum shrink*, not copy-block (**observed** APC). Retention protection for adaptive rescue never armed (`rescue_last_reason=retention_ok`).
5. **Qualified live sources exist?** By stable-mean proxy (nsucc≥3) **yes — typically 5/5 other workers** (**reconstructed**). Live skill_delta≥0.02 source gate: **not observable** per tick (no per-source windows logged), but moot because cascade never ran.
6. **Source failure fields:** N/A — selection not reached.
7. **Ordinary cascade called?** **No** (**reconstructed** + `source_tier_*_count==0` **observed**).
8. **Code path returns before source selection?** **Yes** — `pbt_control_plane.py:463–475` inside observer 10101–10108.
9. **No retention collapse ⇒ no adaptive rescue?** **Yes, primarily.** Adaptive arms on retention≤0.70 × patience or same-tick hard collapse (`adaptive_precision_consolidation.py:675–701`). p2’s only soft dip @560M (`gret≈0.29`) either lacked streak/validity or recovered before arming; tip APC: `degradation_streak=0`, attempts=0. Chronic low rank without peak drop is **not** a rescue trigger.
10. **PBT only handles peak→degradation, not chronic lag?** **Yes for this profile's wired gates:** ordinary path needs *stuck* (`nsucc&lt;1`∧`Δ≤0`); adaptive needs *retention collapse*. Slow catch-up with Δ&gt;0 is intentionally ignored.

---

## 5. WM Belief Dataflow

```text
RSSM encode_sequence / encode_belief     (rssm_lite.py)          [producer exists]
        │
        ▼
build_rssm_belief_cache(...)             (rssm_belief_provider.py)
        │   gated by train.py:3860–3864
        │   _need_belief = (rssm_planner is not None)
        │                 or (rssm_reranker is not None)
        │   Profile 22: both None (eir_mppi_mode=off, eir_use_rssm_mppi=False)
        ▼
eir_bridge.rssm_belief_cache = None      (also forced None @ train.py:6162–6163)
eir_bridge.rssm_belief_provider = None
        │
        ▼
_belief_for_risky_rows(...)              (eir_mpc.py:1210–1231) → None
        │
        ▼
_compute_snippet_distill_loss_progress   (eir_mpc.py ~1556–1649)
  effective_alpha_wm(...)
     ├─ alpha==0 → force reason_id=11, skip belief lookup     [p2 observed]
     └─ alpha>0  → wm_head_probs_for_candidates(belief=None)
                      → reason_id=2 belief_missing             [p0 observed]
        │
        ▼
score_snippet_candidates / fallback      changed_frac=0
```

Reranker reason IDs: `snippet_wm_reranker.py:36–52, 339–340, 239`.

---

## 6. Exact `belief_missing` Root Cause

| Class | Role |
|---|---|
| **Primary: A — belief producer path not enabled for EIR** | `_need_belief` false because Profile 22 disables MPPI/RSSM reranker; cache builder never runs. |
| **Secondary: C — caller passes `None` belief** | After alpha&gt;0, `_belief_for_risky_rows` returns None → reason 2. |
| **Secondary (p2 only): G — alpha stays zero** | `wm_cal/alpha_wm=0`, `calibration_pass=0` → reason **11**, belief check never reached. |

Not B (no belief produced for EIR cache), not D/E/F.

**Blast radius**

| Question | Answer |
|---|---|
| All Profile 22 workers? | **Yes** for belief cache (`mppi_belief_valid=0` on p0 and p2). Symptoms differ by alpha. |
| Jul12 same bug? | **Likely yes** (same profile flags / same `_need_belief` gate; Jul12 TB deleted but prior audit saw reason 2). |
| Profile 23 same path? | **Yes** for cache gating if MPPI still off; P23 may additionally force `alpha_wm=0` in bootstrap (**code**; not re-audited here). |
| Only snippet rerank? | **Snippet WM rerank / any consumer of `rssm_belief_cache`**. Seq-WM training, imagined diagnostics, floor-recovery scores still run (launch flags + separate TB families). |
| Minimal fix locus | `train.py` `_need_belief` — include `snippet_wm_rerank_enable` (or always build cache when progress-refine distill needs WM heads). Optional: keep provider fallback. **Do not hot-patch this live run.** |

---

## 7. P2 Bank Tolerance Distribution

From latest `dexpbt_8gpu_ws/002/*.pth` key `eir_elite_snippet_bank` (**observed**, readonly load):

| | |
|---|---|
| size / schema | 50000 / v2 |
| success_tol | **only {0.03625, 0.03816}** |
| sources | p0: 32631 · p1: 17369 · **p2 local: 0** |
| tip query tol | ≈0.05513 |
| compat `δ≤0.20` | **0 / 50000 (0%)** |
| δ_ε median | ≈0.37–0.42 |

Files: `p2_bank_tolerance_hist.csv`, `p2_bank_source_hist.csv`.

EIR mailbox (**observed** `eir_broadcast_events.jsonl`): 16 broadcast / 12 consume; target=2 appears 7×; example `consume source=0 target=2 entries=50000` at 1440M — whole-bank overwrite by stronger workers.

Retrieval order (`elite_snippet_bank.py:524–544`): **compat over full bank first**, then sample-pool truncate — so eviction/`sample_pool` did not hide compatible rows; **there were none**.

---

## 8. Exact `no_compatible_frac=1` Root Cause

**Classification:** bank contains no compatible snippet (hard).

Chain:

```text
confirmed: leaders shrink tol faster (p0/p1 tip tol ~0.03)
confirmed: broadcast/consume installs their snippets into p2 bank (50k, sources 0/1)
confirmed: p2 query tol still ~0.055–0.061 for long stretches
confirmed: log-rel gap > 0.20 for every row
confirmed: TB no_compatible_frac=1, active_frac=0, ESS=0 from first logged refine window
rejected: query tol read as 0 / schema zero (tol fields finite & diverse across sources)
rejected: sample-pool removed compat before check (code order)
likely: priority eviction favors dense foreign mid/hard-progress snippets over any rare local mid-tol rows
not observable: exact insert/evict timeline of transient local-compatible rows before overwrite
```

Teacher confidence / progress gate never activate because retrieval_valid_frac=0 (`no_compatible_or_low_gate`).

---

## 9. Combined Failure Loop

| Step | Status |
|---|---|
| Long-term weak / bottom | **confirmed** |
| Ordinary PBT does not copy | **confirmed** (target gate) |
| Adaptive rescue does not trigger | **confirmed** (APC 0 attempts, `retention_ok`) |
| Broadcast bank incompatible with p2 tol | **confirmed** |
| Snippet refinement no effective samples | **confirmed** |
| WM rerank inactive on p2 | **confirmed** (α=0 → reason 11); population-wide belief cache dead (**confirmed**) |
| p2 relies on local PPO | **confirmed** (slow climb 0→~25 nsucc without weight copy) |

---

## 10. Minimal Fix Options (proposal only — **not applied**)

1. **PBT chronic-lag path:** extend ordinary target gate beyond `nsucc&lt;1∧Δ≤0` (e.g. bottom ∧ rank gap ∧ frames_in_bottom ≥ X), or arm adaptive rescue on “never-peaked bottom” / relative rank lag. Code: `pbt_copy_allowed_for_target` / `evaluate_degradation_rescue`.
2. **Belief for snippet WM:** `_need_belief |= snippet_wm_rerank_enable` in `train.py:3860`. Optionally wire light `rssm_belief_provider`.
3. **Bank compatibility:** reserve local capacity; block whole-bank overwrite when `|log ε_s/ε_t|` incompatible; or advertise source snippets only within log-gap of *target* current tol.
4. **Telemetry:** log `target_gate_reason` and source-reject reasons into tick JSONL (today only scheduler fields).

---

## 11. Risks and Required A/B Experiments

| Experiment | What it isolates | Risk |
|---|---|---|
| A: chronic-bottom copy enable (no other changes) | PBT utilization vs stability | may overwrite recovering climbers |
| B: belief-cache with snippet rerank only | reason 2→0? changed_frac | GPU cost / train slowdown |
| C: tol-aware broadcast reserve | no_compatible_frac on lagging workers | slows elite sharing |
| D: A+C together | whether lag closes faster than Jul12 tip | interaction effects |

Keep this run untouched as negative control for “P22 status quo.”

---

## Three-sentence answers

1. **p2 ordinary PBT did not copy because** every bottom tick failed `pbt_copy_allowed_for_target` (`not_stuck` / `skill_delta_improving` / `skill_delta_invalid` from tol-bin resets) and returned before `_select_source` — evidenced by TB `skill_delta_copy_allowed≡0`, replay CSV, and `source_tier_*_count≡0`.
2. **WM snippet rerank remained inactive because** Profile 22 never builds `rssm_belief_cache` (`_need_belief` false with MPPI/reranker off; `mppi_belief_valid≡0`), so strong workers hang on reason **2**, while p2 never raises `wm_cal/alpha_wm` and stays on reason **11**.
3. **p2 snippet refinement remained inactive because** its EliteSnippetBank is filled exclusively by p0/p1 snippets at tol≈0.036–0.038 (0 local; broadcast consume of 50k), yielding `δ_ε>0.20` vs p2’s query tol≈0.055 and `no_compatible_frac=1` / ESS=0.
