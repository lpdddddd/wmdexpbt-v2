# Profile22 ABC Review Fixes

Date: 2026-07-14 (updated same day — snapshot freshness fix)  
Parent: `cc1b3a1` (prior review-fix tip)  
Status: snapshot freshness / commit validation fix — **no training / launcher execution**

---

## Snapshot freshness bug (cc1b3a1) → fixed in this commit

### Why `population_frame_complete` must not gate snapshot freshness

`population_frame_complete.json` is written only when a worker reaches the
**final** `target_population_frames` (frame-stop). Mid-train PBT snapshots at
200M / 400M / 1B / 1.5B never create that marker.

cc1b3a1 mapped missing marker → `frame_complete=False` → `freshness_ok=False`
→ every chronic source failed common health → **zero chronic copies**.

### New freshness definition (exact, fail-closed)

Production path: `inspect_peer_snapshot_health()` in `pbt_live_source_health.py`,
called from `DexPBTPbtAlgoObserver._inspect_peer_checkpoint_health()`.

1. Population state has `checkpoint` path that is an existing file.
2. `LATEST.commit` exists and parses to a JSON dict with non-empty `commit_dir`.
3. That commit directory exists under the peer workspace.
4. **`validate_snapshot_commit(commit_dir, tick=interval_steps)`** returns
   `(True, "ok")` (MANIFEST, member hashes, lineage_id, quantized population_frame).
5. MANIFEST `lineage_id` and state `lineage_id` both present and equal.
6. MANIFEST `population_frame` **equals** state `population_frame`/`frame`
   (exact int match; no tick-slack).
7. SHA-256 of the published checkpoint equals MANIFEST `member_shas["model.pth"]`
   (proves identity with the LATEST commit, not an arbitrary older artifact).

`population_frame_complete.json` is **not** consulted for chronic source health.
Frame-stop marker write logic is unchanged.

### Mid-train snapshot without completion marker

A legal publish via `snapshot_commit_barrier` + matching state passes health
even when the completion marker is absent. Presence of the marker also does
**not** substitute for a valid commit (destroyed `commit_dir` still fails).

### Fail-close cases

| Failure | Result |
|---|---|
| Corrupt / missing `LATEST.commit` | reject |
| Missing `commit_dir` | reject |
| Missing / corrupt MANIFEST or member hash mismatch | `validate_snapshot_commit` fail |
| Tampered `model.pth` / `meta.json` in commit | `member_corrupt:*` |
| State vs MANIFEST lineage mismatch | reject |
| State vs MANIFEST population_frame mismatch | reject |
| Missing published checkpoint / hash ≠ MANIFEST | reject |
| Stale state while LATEST points to newer commit | reject |

### Telemetry (chronic selection ticks only)

TB: `pbt/chronic_source_checkpoint_valid`, `…_lineage_valid`,
`…_freshness_valid`, `…_health_allowed`, `…_health_reason_hash`

JSONL: `pbt_chronic_source_health.jsonl` under the policy run dir.

### Other non-blocking telemetry fixes in this commit

- Belief JSONL: `int_or_missing` preserves legitimate `generation=0` /
  `reason_id=0` (only `None` → `-1`).
- Bank merge: eviction / kept counts classified by old/new index range **and**
  per-row `source_policy_idx` (incoming local reflux ≠ foreign).

---

## 1. Fix status per review item (prior cc1b3a1)

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
| — | Snapshot freshness used final completion marker | **Fixed this commit** |

---

## 2. Modified files / functions (this freshness commit)

- `pbt/pbt_live_source_health.py` — `PeerCheckpointHealth`, `inspect_peer_snapshot_health`; remove completion-marker freshness
- `pbt/pbt_observer.py` — inspect + chronic TB/JSONL health logging
- `world_model/elite_snippet_bank.py` — precise local/foreign eviction telemetry
- `world_model/eir_belief.py` — `int_or_missing`
- `scripts/train.py` — belief event 0-value fix
- `tests/test_profile22_abc.py` — real workspace integration via `snapshot_commit_barrier`
- analysis reports under `results/analysis/profile22_abc_implementation/`

---

## 3. Common source health gate (full conditions)

Fail-closed if any field missing/`None`:

1. state present  
2. finite ranking score + nsucc_mean  
3. restart burn-in (`fsr >= min_frames_as_source_after_restart`)  
4. source gate: top → `_eligible_as_source_with_reason`; mid → `_eligible_as_source_relaxed_with_reason`  
5. checkpoint file exists  
6. LATEST commit parseable + `validate_snapshot_commit` OK  
7. state lineage matches commit MANIFEST lineage  
8. freshness: exact frame match + checkpoint identity vs MANIFEST model hash  
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
- `incoming_local_kept` / `incoming_foreign_kept` by `source_policy_idx`  
- `existing_local_kept` / `existing_foreign_kept` for old range survivors  
- `local_evicted_count` / `foreign_evicted_count` = existing before − existing kept  
- API still returns `int` = `incoming_kept_count`  
- selection / reserve behavior unchanged

---

## 7. B/C telemetry log paths

**B (consume):**
- TensorBoard: `eir_pbt/recovery_bank_merged_entries`, `eir_pbt/snippet_bank_merged_entries`, `eir_pbt/broadcast_rejected_incompatible`, plus diagnostic scalars  
- JSONL: `workspace/metrics/eir_broadcast_events.jsonl` consume events with `bank_diagnostics`  
- Console distinguishes recovery vs snippet / reject

**C (belief):**
- TensorBoard: `eir_wm/need_belief`, `belief_build_*`, `belief_cache_valid`, `belief_build_latency`, `belief_numel`, `belief_ndim`, `belief_failure_reason_hash`  
- JSONL: `workspace/metrics/eir_belief_events.jsonl` (one event per train frame, guarded)  
- Exact failure reason string in JSONL; `generation` / `rerank_reason_id` use `int_or_missing`

---

## 8. Integration test workspace structure

Temp peer dir `001/` after `snapshot_commit_barrier`:

```
001/
  LATEST.commit                 # {"commit_id", "commit_dir"}
  <prefix>.pth                  # published model
  <prefix>.json                 # published population state
  <prefix>.staging.commit/      # final commit dir
    model.pth
    meta.json
    MANIFEST                    # member_shas, lineage_id, population_frame, …
```

No `population_frame_complete.json` in the legal mid-train fixture.

---

## 9. Tests / results

```
PYTHONPATH=. python -m unittest \
  tests.test_profile22_abc \
  tests.test_progress_refine.TestProfile \
  tests.test_readiness_adaptive.TestProfile23Wiring
→ Ran 50 tests … OK
```

Plus `py_compile` of touched modules and `bash -n` ABC launcher — OK.
**No GPU/runtime smoke. No train/master/worker/supervisor/launcher.**

---

## 10. Profile22 regression

- Enable flags still False (table + authoritative expand)  
- Profile24 ABC flags still all on under authoritative expand  
- Mid mutation / MPPI still off  
- Ordinary skill_delta post-gate path unchanged  

---

## 11. Not executed this round

- GPU belief encode cost measurement  
- Live chronic tick against real population  
- ABC training launch  

## 12. Training declaration

**No training, smoke, master, worker, supervisor, or launcher was started.**
