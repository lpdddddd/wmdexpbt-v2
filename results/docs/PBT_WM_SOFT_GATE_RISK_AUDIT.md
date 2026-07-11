# PBT-WM Soft Activation Gate Risk Audit

**Baseline commit:** `86bd7eb307b3dcdf3321abb721d160e7f6130971`  
**Audit date:** 2026-06-15  
**Scope:** Strategy Zoo soft activation gate v1 (actor-distill, WM PBT source selector, curriculum shrink gate, profile authoritative restart)

---

## Summary

| Risk | Verdict | Blocks smoke? | Action |
|------|---------|---------------|--------|
| A hard_tol consistency | limitation | no | doc + test only |
| B candidate pool / outer EIR tol gate | **bug** (at baseline) | **yes** | **fixed in audit** |
| C WM PBT source activation in ranking | **bug** (at baseline) | **yes** for `wm_pbt_source_selector` | **fixed in audit** |
| D profile authoritative | ok | no | existing tests pass |
| E perf gate inert | limitation | no | future patch point documented |
| F by-tol-bin fidelity | limitation | no | partial logging/analyzer fix |

At baseline `86bd7eb`, soft actor-distill could not activate above `tol=0.04` because the outer EIR `distill_lambda_for_tol` gate blocked the entire candidate + distill path. WM PBT source activation weight was logged but did not scale ranking scores. **Minimal fixes were applied during this audit** (see each section). All 61 targeted unit tests pass.

**Recommendation:** safe to proceed to a **short smoke** after applying audit patches; watch telemetry listed at the end.

---

## Risk A: hard_tol consistency

**Verdict:** limitation (not a zoo blocker)

### Evidence

1. **`compute_wm_activation` (`hard_tol`)** — tolerance pass/fail AND time ramp both apply (`wm_activation_gate.py` L186–195, L194–195).

2. **`compute_actor_distill_activation` (`hard_tol`)** — step-on at `start_after_frames`, no time ramp (`eir_mpc.py` L351–360).

3. **Zoo profiles** — all non-legacy profiles use `soft_tol`, `confidence_only`, or `shrink_candidate`; none use `hard_tol`:

| Profile | gate_mode |
|---------|-----------|
| `rssm_mppi_actor_distill` | `eir_actor_distill_gate_mode=soft_tol` |
| `elite_bank_actor_distill` | `eir_actor_distill_gate_mode=soft_tol` |
| `conservative_rssm_planner_distill` | `eir_actor_distill_gate_mode=soft_tol` |
| `wm_pbt_source_selector` | `wm_pbt_source_gate_mode=confidence_only` |
| `wm_curriculum_controller` | `wm_curriculum_gate_mode=shrink_candidate` |

4. **`train.py` default** — `eir_actor_distill_gate_mode=hard_tol` is legacy-only when no profile is set.

### Fix

None required for zoo. Added docstring note in `wm_activation_gate.py` and unit test `TestHardTolConsistency` in `tests/test_wm_soft_gate_risk_audit.py`.

---

## Risk B: candidate pool tol gate

**Verdict:** bug at baseline → **fixed in audit**

### Evidence

#### B1 — Outer EIR gate (primary bug)

At baseline, `compute_eir_loss` blocked the entire path when `tol_mean > start_tol (0.04)` via `distill_lambda_for_tol` → `EIR_TOL_ABOVE_START`. Soft actor-distill inside `if block == EIR_OK` never ran for tol ∈ (0.04, 0.07] despite profile `tol_start=0.060/0.070`.

**Audit fix:** when `actor_distill_gate_mode ∈ {soft_tol, confidence_only}`, bypass outer `EIR_TOL_ABOVE_START` while keeping `lam` scaling on residual apply only.

#### B2 — `_stage_for_tol` / elite retrieval (not the primary blocker)

For `tol > guidance_start_tol (0.04)`, `_stage_for_tol` returns `None`. `EliteRecoveryBank.retrieve_candidates(stage=None)` uses the **full bank**. Elite/archive candidates **are available** once the outer gate allows entry.

| tol | `_stage_for_tol` | elite/archive in pool? |
|-----|------------------|------------------------|
| 0.065 | None | yes (full bank) |
| 0.055 | None | yes |
| 0.045 | 3 (mid) | yes |
| 0.035 | 3 | yes |

`eir_elite_candidate_k=64`, `eir_archive_candidate_k=32` in elite profile — sufficient once `generate_eir_candidates` runs.

#### B3 — RSSM / conservative profiles

RSSM MPPI candidate injection shares the same outer gate; blocked by B1 above tol 0.04 at baseline, not by stage gate.

### Mock test result

`tests/test_wm_soft_gate_risk_audit.py::TestCandidatePoolTolGate`:

- `test_stage_and_elite_retrieval_above_start_tol` — mock banks called; `SRC_ELITE_LOCAL` and `SRC_ELITE_ARCHIVE` present for tol 0.065/0.055/0.045/0.035.
- `test_soft_actor_distill_not_blocked_by_outer_start_tol` — at `cache.tol_mean=0.055`, `last_block_reason != EIR_TOL_ABOVE_START` after fix.

### Fix recommendation (applied)

Minimal: bypass outer `distill_lambda_for_tol` block for soft-gated actor-distill. Rejected: raising `eir_guidance_start_tol` (unnecessary) or new `eir_candidate_gate_mode` (too large).

---

## Risk C: WM PBT source selector gate

**Verdict:** bug at baseline → **fixed in audit**

### Evidence

At baseline, ranking used raw `combine_wm_recovery_score` into `alt_objectives`. `_last_wm_source_activation_weight` was computed only in TB logging (time ramp × uncertainty) with **no effect** on ranking.

Only `wm_pbt_source_selector` sets `pbt_wm_score_mode=active` with `wm_pbt_source_start_after_frames=300M`.

### Whether activation affects ranking

**Before audit:** no — telemetry only.  
**After audit:** yes — `score_effective = real + w * (wm - real)`; `used_wm_for_pidx=False` when `w=0`.

Fail-open preserved: invalid eval / high uncertainty / before start frames → real score.

### Fix recommendation (applied)

`compute_wm_pbt_source_activation_weight` + `blend_wm_ranking_score` in `wm_activation_gate.py`; wired in `pbt_observer.py` ranking loop. Tests in `TestWMPBTSourceActivation`.

---

## Risk D: profile authoritative / restart pollution

**Verdict:** ok

### Evidence

- `apply_wm_strategy_profile(..., authoritative=True)` for non-legacy profiles.
- Restart skip flags + `normalize_extra_args()` boolean fix.

### Tests

| Case | Result |
|------|--------|
| `ppo_only` + conflicting WM/EIR flags | overridden to WM off |
| `rssm_mppi_actor_distill` + `--no-eir_enable` | `eir_enable=True` restored |

Existing + audit tests pass.

---

## Risk E: perf gate inert

**Verdict:** limitation (acceptable for v1)

### Evidence

- `nsucc_mean < 0` → `w_perf = 1.0` (does not kill actor_distill).
- `bridge.last_nsucc_mean` defaults `-1.0`; profile `nsucc_ref` inactive.

### Future patch point

Set `eir_bridge.last_nsucc_mean` from observer/training_curve `nsucc_mean` in train.py before `compute_eir_loss` (~L3091 hook). One-line minimal impl.

---

## Risk F: by-tol-bin fidelity

**Verdict:** limitation (partial fix in audit)

### Evidence

- Actor-distill bin from scalar `cache.tol_mean` per minibatch.
- Analyzer keyed on PBT-tick `success_tolerance` — can diverge within an interval.

### Fix (partial)

- `last_actor_distill_tol_bin` + TB logging
- training_curve: `curriculum_tol_bin`, `actor_distill_tol_bin`
- analyzer prefers `actor_distill_tol_bin`

---

## Tests run

```bash
py_compile (6 modules) — OK
python -m unittest tests.test_wm_activation_gate \
  tests.test_wm_strategy_profiles \
  tests.test_pbt_wm_strategy_zoo_full_preflight \
  tests.test_wm_soft_gate_risk_audit
# Ran 61 tests — OK
bash -n results/launchers/start_regrasping_pbt_wm_strategy_zoo_full_3b.sh — OK
```

---

## Remaining risks before smoke

1. Perf gate (`nsucc_ref`) inert until nsucc threaded.
2. `actor_distill_tol_bin` in training_curve uses curriculum fallback until checkpoint carries distill bin.
3. `hard_tol` inconsistency remains for legacy default only.
4. Residual apply intentionally hard-gated at `start_tol=0.04`.

---

## Smoke recommendation

**Proceed to smoke** after audit patches are applied.

| Profile | Must-see telemetry |
|---------|-------------------|
| `rssm_mppi_actor_distill` | `actor_distill_activation_weight_mean > 0` when tol ∈ (0.04, 0.06] and frames ≥ 500M; block ≠ `EIR_TOL_ABOVE_START` |
| `elite_bank_actor_distill` | `actor_distill_source_frac_elite/archive > 0`; soft ramp 0.07→0.045 |
| `wm_pbt_source_selector` | before 300M: `wm_source_activation_weight=0`; after ramp: `wm_score_used` may become true |
| all | `actor_distill_tol_bin` / `curriculum_tol_bin` in training_curve |
| restart | profile not polluted by legacy WM flags |

Watch: WM must not change source selection before 300M frames.

---

## Post-audit code changes

| File | Change |
|------|--------|
| `eir_mpc.py` | Soft outer gate bypass; `last_actor_distill_tol_bin` |
| `wm_activation_gate.py` | PBT activation helpers + Risk A doc |
| `pbt_observer.py` | Ranking blend; training_curve tol_bin |
| `train.py` | Log `actor_distill_tol_bin` |
| `analyze_pbt_wm_strategy_zoo_full_run.py` | Prefer `actor_distill_tol_bin` |
| `tests/test_wm_soft_gate_risk_audit.py` | New tests |
