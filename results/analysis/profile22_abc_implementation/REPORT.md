# Profile22 ABC Implementation Review Pack

Date: 2026-07-14  
Workspace tip before review-fix commit: `b529bb6`  
See also: [`REVIEW_FIXES.md`](REVIEW_FIXES.md) for post-review corrections.

Baseline (unchanged): `regrasping_cser_dexpbt_progress_refine_5b_20260714_092846`  
**STOP for human review — do not start ABC run.**

---

## Ablation clarification (authoritative profiles)

- A/B/C **code switches are independent**.
- Profile **24** is a fixed **authoritative all-on** ABC profile; `--no-*` is **not** a reliable ablation under restart-safe expansion.
- Future ablations should use new derived profiles (not allocated this round).
- Profile **22** pins A/B/C enables to **False**.

---

## Quick answers

| Question | Answer |
|---|---|
| Profile22 training behavior changed? | **No** (enables still False; ordinary live skill_delta path unchanged) |
| Chronic uses common source health gate? | **Yes** (selection + post-select). Ordinary keeps prior gates + skill_delta |
| Lineage change clears stale bottom span? | **Yes** (keeps cooldown) |
| Full-bank merge records `incoming_kept_count`? | **Yes** |
| B/C telemetry observable? | **Yes** — TB + JSONL |
| Training started? | **No** |

Full original implementation notes remain below; prefer REVIEW_FIXES for the latest semantics.

---

## 1. Modified / added files

### Added
- `runtime_root/.../pbt/pbt_chronic_bottom.py` — A helpers + state
- `runtime_root/.../world_model/snippet_bank_tolerance.py` — B filter helpers
- `runtime_root/.../tests/test_profile22_abc.py`
- `results/launchers/start_regrasping_cser_dexpbt_progress_refine_abc_5b.sh`
- `results/analysis/profile22_abc_implementation/*` (this pack)

### Modified
- `world_model/wm_strategy_profiles.py` — profile **24** + Profile22 ABC hard-off pins
- `scripts/train.py` — CLI flags, belief OR-gate, broadcast cfg wiring, observer kwargs
- `pbt/pbt_observer.py` — chronic path, sidecar, consume query-tol, live-tier labels
- `world_model/eir_broadcast.py` — tolerance-aware merge path
- `world_model/elite_snippet_bank.py` — `merge_tolerance_aware` / local reserve eviction

---

## 2. Code locations (functions)

| Area | Location |
|---|---|
| A eligibility / source / state | `pbt_chronic_bottom.py`: `evaluate_chronic_eligibility`, `pick_chronic_source`, `live_source_ok_for_chronic`, `update_continuous_bottom`, `reset_after_chronic_copy`, `skill_delta_with_reason` |
| A observer wire | `pbt_observer.py`: `_chronic_bottom_*`, `_select_chronic_bottom_source`; bottom branch ~10230–10405; exact mutation ~11474; post-commit ~11575 |
| B filter | `snippet_bank_tolerance.filter_state_by_tolerance` |
| B merge | `EliteSnippetBank.merge_tolerance_aware`, `_insert_with_local_reserve`; `eir_broadcast.merge_payload_into_bridge` |
| C belief | `train.py` `_need_belief` OR with `eir_enable_belief_for_snippet_wm_rerank` ∧ snippet WM enable; telemetry on bridge |
| Profile | `wm_strategy_profiles.py` id 24 `cser_dexpbt_progress_refine_abc` |

---

## 3. Flag defaults vs ABC profile

| Flag | CLI default | Profile22 | ABC (24) |
|---|---|---|---|
| `pbt_enable_chronic_bottom_exploit` | False | **False** (pinned) | **True** |
| `pbt_chronic_min_bottom_frames` | 200M | (default) | 200M |
| `pbt_chronic_min_bottom_ticks` | 5 | (default) | 5 |
| `pbt_chronic_min_score_gap` | 0.15 | (default) | 0.15 |
| `pbt_chronic_cooldown_frames` | 400M | (default) | 400M |
| `pbt_chronic_max_copies_per_lineage` | 1 | (default) | 1 |
| `eir_tolerance_aware_bank_broadcast` | False | **False** (pinned) | **True** |
| `eir_preserve_local_bank_fraction` | 0.20 | 0.20 default (unused if B off) | **0.20** |
| `eir_enable_belief_for_snippet_wm_rerank` | False | **False** (pinned) | **True** |

Independent ablation: any flag can be flipped via CLI; profile authoritative still applies table values for profiles that define them.

---

## 4. Profile22 vs ABC resolved config

See `config_diff.md` (9 differing keys only). PPO / curriculum / 6×8192 defaults remain launcher-owned as before; method substrate otherwise identical to Profile22.

**Profile22 behavioral intent preserved:** A/B/C enable flags resolve False. Table now pins them False for restart purity (same resolved value as prior CLI defaults).

---

## 5. Chronic eligibility + source fallback

**Eligibility (all required):**
- `tier == bottom`
- `continuous_bottom_frames >= 200M`
- `continuous_bottom_ticks >= 5`
- not promising-protected
- not elimination burn-in (`not eligible_for_elimination`)
- not adaptive probation
- past `chronic_copy_cooldown_until`
- `chronic_copy_count < max_copies_per_lineage` (default 1)
- best ranking score gap ≥ 0.15

When ordinary `pbt_copy_allowed_for_target` rejects **and** eligibility holds → `chronic_path=True` (does **not** set adaptive rescue mode).

**Source selection:** top then mid; live + checkpoint file present; gap ≥ 0.15; then:
- valid skill_delta ≥ 0.02 → accept
- invalid reason `mixed_tol_bins` → stable performance fallback (nsucc advantage ≥ 3, drop_ratio ≤ 0.30)
- other invalid reasons → fail-close

**Copy:** exact params (`mutation_rate` effectively 0), full `.pth` transaction unchanged, 400M cooldown + copy count after successful restart.

State persisted at `workspace/chronic_bottom/{pidx:03d}.json`.

---

## 6. Checkpoint transaction copy scope (unchanged; documented)

PBT restart copies the source `.pth` via existing `_restart_process_with_new_params` / `shutil.copyfile`. Checkpoint export (`get_full_state_weights` + EIR extras) includes:

- model / policy weights
- optimizer (+ rl_games scheduler state inside full weights)
- curriculum metadata (JSON + inheritance path)
- `eir_elite_bank` / `eir_elite_snippet_bank` top-k export
- `eir_scorer` / `eir_residual_head`
- `wm_calibration_state`
- optional readiness / AWR sidecar refs
- PBT lineage / restart args via supervisor manifest

ABC does **not** shrink this scope.

---

## 7. Bank merge / eviction / local reserve

1. Filter incoming by `δ_ε = |log((tol_s+ε)/(tol_t+ε))| ≤ 0.20` **before** truncate/merge.
2. If compatible count = 0 → reject overwrite; keep target bank; set `broadcast_rejected_incompatible`.
3. Else `merge_tolerance_aware`: insert filtered rows; on pressure, reserve `floor(capacity * 0.20)` local (`source_policy_idx == target`), then fill remainder by priority (local excess + foreign).
4. Capacity remains 50000 (or cfg bank_size). Distill weights / top-k / H / retrieval thresholds untouched.

---

## 8. Belief data flow (C)

```
snippet_wm_rerank_enable ∧ eir_enable_belief_for_snippet_wm_rerank
        │
        ▼
train rollout post-step: _need_belief = True
        │
        ▼
build_rssm_belief_cache(...)  →  eir_bridge.rssm_belief_cache
        │
        ▼
compute_eir_loss / snippet WM reranker reads cache
```

- Does **not** enable MPPI planner or RSSM-MPPI.
- Does **not** force `alpha_wm > 0`; calibration unchanged.
- Invalid/missing cache → prior safe fallbacks; telemetry: `last_need_belief`, `last_belief_build_*`, shape/latency/failure reason.

**Estimated cost (review):** one full-rollout RSSM encode per PPO epoch per worker when C on — same order as legacy MPPI belief path; VRAM = one belief tensor `[T,N,…]` on worker GPU (exact size depends on RSSM latent dims / envs). Expect non-trivial but smaller than enabling MPPI candidate rollouts.

---

## 9. New telemetry (summary)

**A:** `chronic_candidate`, continuous frames/ticks, copy count, score gap, target gate reason, source filter/fallback, mutation_applied=false, copy events `chronic_bottom_copy_committed`.

**B:** incoming/compatible counts & fracs, tol/delta hist stats, `broadcast_rejected_incompatible`, merge/overwrite counts, eviction local/foreign, existing retrieval diagnostics retained.

**C:** need_belief / build attempt/success / cache valid / shape / latency / failure reason; existing alpha_wm / calibration / rerank changed/fallback fracs remain.

---

## 10. Tests

See `test_results.md`. **30** unit/regress tests OK; launcher `bash -n` OK; no training PIDs.

---

## 11. Unresolved risks

1. Chronic path still requires healthy live sources; if all mid/tops fail checkpoint checks, skips remain.
2. Mixed-tol fallback thresholds (advantage 3 / drop 0.30) are heuristic — validate on first ABC ticks.
3. Belief encode cost under 8192 envs not measured in-process this round.
4. B local reserve uses soft priority fill — extreme priority imbalance could still shrink local below reserve if reserve request > available local (correct); empty local stays empty until local collection resumes.
5. Profile22 table gained ABC=False pins — resolved enables stay False, but table hash/diff vs pre-ABC tip is non-empty for those three keys.
6. Full end-to-end observer chronic tick not executed (no training).

---

## 12. Human review checklist

- [ ] Confirm Profile22 hard-off pins are acceptable.
- [ ] Read chronic override vs adaptive rescue interaction in `pbt_observer.py` bottom branch.
- [ ] Confirm exact-copy + full `.pth` inheritance is desired for chronic (bank/WM cal come along).
- [ ] Approve mixed-tol fallback thresholds.
- [ ] Review tolerance filter λ=0.20 alignment with retrieval.
- [ ] Review belief OR-gate for accidental MPPI enable (should be none).
- [ ] Sanity-check ABC launcher argv vs Profile22 launcher (only ABC deltas).
- [ ] Decide GPU schedule vs any existing runs before starting.

---

## 13. Subsequent start command (do not run now)

```bash
COMMIT_SHA=$(git -C /root/autodl-tmp/wmdexpbt rev-parse HEAD) \
  bash /root/autodl-tmp/wmdexpbt/results/launchers/start_regrasping_cser_dexpbt_progress_refine_abc_5b.sh
```

---

## Final answers (required)

| Question | Answer |
|---|---|
| Modified Profile22 **behavior**? | **No** for enables (still False). Table now explicitly pins A/B/C False; other Profile22 keys unchanged. |
| A/B/C independently switchable? | **Yes** (separate CLI/profile flags). |
| ABC profile all on? | **Yes** (24 turns all three enables True). |
| Launcher only created, not executed? | **Yes**. |
| Any training started by this task? | **No**. |
| Human focus areas? | Chronic override vs rescue; mixed-tol fallback; full `.pth` secondary effects; belief cost; Profile22 pin keys. |

**Stopped. Waiting for your review. Do not start the ABC run until you approve.**
