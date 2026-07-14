# Profile22 ABC Review Fixes

Date: 2026-07-14  
Parent: `b529bb6`  
Status: review fixes only — **no training / launcher execution**

---

## 1. Fix status per review item

| # | Issue | Status |
|---|---|---|
| 1 | Chronic source safety incomplete | **Fixed** — shared `evaluate_common_live_source_eligibility` |
| 2 | Lineage change keeps stale bottom span | **Fixed** — `reset_on_lineage_change` |
| 3 | Committed event trigger telemetry = 0 | **Fixed** — snapshot before reset |
| 4 | Full-bank merge kept count = 0 | **Fixed** — incoming index-range counting |
| 5 | B telemetry not logged | **Fixed** — TB + `eir_broadcast_events.jsonl` |
| 6 | C belief telemetry memory-only | **Fixed** — TB `eir_wm/*` + `eir_belief_events.jsonl` |
| 7 | Belief tests duplicated predicate | **Fixed** — production `need_eir_belief_cache` |
| 8 | Ablation wording wrong for Profile24 | **Fixed** — report + authoritative override test |

---

## 2. Modified files / functions

### Added
- `pbt/pbt_live_source_health.py` — `CommonSourceEligibility`, `evaluate_common_live_source_eligibility`, `evaluate_peer_checkpoint_health`
- `world_model/eir_belief.py` — `need_eir_belief_cache`, `need_eir_belief_false_reason`

### Updated
- `pbt/pbt_chronic_bottom.py` — lineage reset; `pick_chronic_source` requires common health; extended decision fields
- `pbt/pbt_observer.py` — peer health inspect; chronic selection/post-gate; trigger event snapshot; consume B logging
- `world_model/elite_snippet_bank.py` — accurate merge/eviction counters
- `world_model/eir_broadcast.py` — richer merge summary / diagnostics
- `scripts/train.py` — production belief helper + TB/JSONL belief events
- `tests/test_profile22_abc.py` — expanded matrix
- analysis reports under `results/analysis/profile22_abc_implementation/`

---

## 3. Common source health gate (full conditions)

Fail-closed if any field missing/`None`:

1. state present  
2. finite ranking score + nsucc_mean  
3. restart burn-in (`fsr >= min_frames_as_source_after_restart`)  
4. source gate: top → `_eligible_as_source_with_reason`; mid → `_eligible_as_source_relaxed_with_reason`  
5. checkpoint file exists  
6. peer `LATEST.commit` present and commit validation OK  
7. state lineage matches commit MANIFEST lineage  
8. `population_frame_complete` marker true  
9. not in adaptive probation (sidecar missing → fail-close when adaptive enabled)  
10. not collapse/unhealthy via curriculum_control probe  

Ordinary PBT **still** uses its historical `_select_source` + `pbt_copy_allowed_for_live_source` path unchanged (Profile22 behavior preserved). Chronic **calls** the shared health helper for selection and post-selection defense-in-depth.

---

## 4. Mixed-tol fallback replaces **only** skill_delta

After common health passes:

- valid Δ ≥ 0.02 → accept (ordinary live skill rule)  
- invalid **only** because `mixed_tol_bins` → stable nsucc/drop fallback may accept  
- other invalid reasons (`lineage_change`, short window, compute failure, …) → reject  

Replacement ledger / restart transaction untouched.

---

## 5. Lineage reset semantics

On lineage ID change:

- `chronic_copy_count = 0`
- `continuous_bottom_frames/ticks = 0`
- `bottom_enter_frame = -1`
- `last_tier = ""`
- `last_reason = lineage_reset`
- **retain** `chronic_copy_cooldown_until`, `last_chronic_copy_frame`, `last_chronic_source`
- atomic persist immediately

Same-lineage resume keeps bottom span.

---

## 6. Bank merge count definitions

After capacity selection over `old || new`:

- `incoming_kept_count` = kept indices in new range  
- `incoming_rejected_count` = incoming − kept  
- `existing_evicted_count` = old − kept-old  
- `local_*/foreign_*` before/after from `source_policy_idx`  
- `eviction_local_count` / `eviction_foreign_count` = **evicted** counts (not kept)  
- API still returns `int` = `incoming_kept_count`

---

## 7. B/C telemetry log paths

**B (consume):**
- TensorBoard: `eir_pbt/recovery_bank_merged_entries`, `eir_pbt/snippet_bank_merged_entries`, `eir_pbt/broadcast_rejected_incompatible`, plus diagnostic scalars  
- JSONL: `workspace/metrics/eir_broadcast_events.jsonl` consume events with `bank_diagnostics`  
- Console distinguishes recovery vs snippet / reject

**C (belief):**
- TensorBoard: `eir_wm/need_belief`, `belief_build_*`, `belief_cache_valid`, `belief_build_latency`, `belief_numel`, `belief_ndim`, `belief_failure_reason_hash`  
- JSONL: `workspace/metrics/eir_belief_events.jsonl` (one event per train frame, guarded)  
- Exact failure reason string in JSONL

---

## 8. Tests / results

```
PYTHONPATH=. python -m unittest \
  tests.test_profile22_abc \
  tests.test_progress_refine.TestProfile \
  tests.test_readiness_adaptive.TestProfile23Wiring
→ Ran 35 tests … OK
```

Plus `py_compile` of touched modules and `bash -n` ABC launcher — OK. No train/master/worker processes.

---

## 9. Profile22 regression

- Enable flags still False (table + authoritative expand)  
- Mid mutation / MPPI still off  
- Ordinary skill_delta post-gate path unchanged  

---

## 10. Not executed this round

- GPU belief encode cost measurement  
- Live chronic tick against real population  
- ABC training launch  

## 11. Training declaration

**No training, smoke, master, worker, supervisor, or launcher was started.**
