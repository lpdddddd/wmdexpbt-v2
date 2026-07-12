# Progress-Conditioned CSER-DexPBT (`cser_dexpbt_progress_refine`)

Profile id **22**. The generalized, curriculum-agnostic formal method that
replaces the hand-tuned `cser_dexpbt_2cm_bridge` fixed-tolerance ladder with
scale-invariant, progress-conditioned control. The 2cm bridge profile is
**retained for reproducibility**; this profile is the method going forward.

## Motivation

`cser_dexpbt_2cm_bridge` works but is specialized: a fixed tolerance ladder
(`0.025 … 0.018`), a fixed `0.0205` probation entry, and fixed-threshold late
rescue (`tol ≤ 0.022`, `mean ≥ 28`, `gap ≤ 0.002`). These constants encode the
2 cm regrasping task and do not transfer. Progress-refine keeps every generic
CSER mechanism (elite snippet bank, first-step distillation, WM calibration
rerank, actor-distill fallback, PBT source safety, conservative consolidation,
hard safety flags) but drives refinement and consolidation from **dimensionless,
scale-invariant** signals so the same configuration transfers across tolerance
scales and tasks.

## Method overview

```
population rollout
  -> extract successful / recovery snippets (relative-tolerance labels)
  -> store in a shared tolerance-conditioned snippet bank (schema v2)
  -> detect rows near the success boundary AND locally stagnant / oscillatory
  -> retrieve compatible snippets (continuous log-relative tolerance)
  -> rerank with calibrated world-model predictions (when available)
  -> distill ONLY the first action with a continuous row-wise weight
  -> PPO update
  -> propagate stable policies via adaptive curriculum-progress consolidation
```

## Mathematical definition

Per PPO rollout row `t` on env `n` (tolerance `ε_t`, distance `d_t`):

- normalized error `ρ_t = d_t / (ε_t + eps)`
- normalized recent progress `v_t = (d_{t-K} − d_t) / (K (ε_t + eps))`
- normalized history `ρ_{t-i} = d_{t-i} / (ε_{t-i} + eps)`
- oscillation `o_t = Std(ρ_{t-K:t})`

Gates:

- proximity `q_prox = exp(−ρ_t / T_p)`
- stall `q_stall = σ((m − v_t) / T_s)`
- oscillation `q_osc = σ((o_t − o_0) / T_o)`
- local difficulty `q_local = 1 − (1 − q_stall)(1 − q_osc)`
- time ramp `r_t = clip((frames − start_after) / ramp, 0, 1)`
- refinement gate `g_t = r_t · q_prox · q_local · 1[history_valid] · 1[lifted]`

Teacher confidence and row weight:

- `c_margin = σ((S_best − S_actor) / T_m)`
- `c_wm = 1 − α_wm · clip(u_t, 0, 1)` (`= 1` when WM unavailable / uncalibrated)
- `c_teacher = c_margin · c_wm`
- `w_t = g_t · c_teacher · 1[retrieval_valid]`
- `L_refine = λ · Σ_i w_i · Huber(μ_θ(f_i), a*_i) / (Σ_i w_i + eps)`

Retrieval compatibility (dimensionless):

- gap `δ_ε = |log((ε_snip + eps)/(ε_query + eps))|`, soft weight `w_ε = exp(−δ_ε/τ_ε)`
- hard fail-closed when `δ_ε > tolerance_max_log_gap`
- score `S = λ_sim·cos + λ_pri·log(1+Q) + λ_tol·log(w_ε) − λ_ρ·|ρ_q − ρ_s|`

Curriculum progress and consolidation:

- `c = clip(log(ε_init/ε_t) / log(ε_init/ε_target), 0, 1)`
- stages: exploration `c < 0.50`, consolidation `0.50 ≤ c < 0.85`, selection `c ≥ 0.85`

## Data flow → code

| Stage | Code |
|-------|------|
| gate + context math | `world_model/progress_conditioned_refinement.py` |
| relative-tolerance recovery labels | `ERDConfig.dhat_margin_mode` in `world_model/elite_recovery_distill.py` |
| progress context in cache | `ERDTargetBuilder.build` → `ERDTargetCache.*_flat` |
| snippet fields | `SnippetBatch` in `world_model/elite_snippet_builder.py` |
| bank schema v2 + batched retrieval | `world_model/elite_snippet_bank.py` (`retrieve_candidates_batched`, `RetrievalResult`) |
| continuous-weight distillation | `world_model/snippet_distill.py::compute_snippet_distill_loss` |
| runtime gate + teacher confidence + retrieval | `world_model/eir_mpc.py::_compute_snippet_distill_loss_progress` |
| curriculum-progress consolidation | `pbt/pbt_consolidation_schedule.py` (`schedule_mode="curriculum_progress"`) |
| probation + retention-conditioned rescue | `pbt/adaptive_precision_consolidation.py` |
| adaptive probation shrink/rollback gate | `pbt_observer._build_curriculum_control_state` via `_adaptive_precision_state` |
| adaptive rescue trigger + source select | `_adaptive_degradation_rescue_ready` → `_select_adaptive_rescue_source_cascade` (live → archive → ordinary) → `_ternary_pbt` |
| adaptive sidecar byte-level txn | `_snapshot_restart_persistent_state` / `_restore_restart_persistent_state` |

## Key formula → code map

| Formula | Location |
|---------|----------|
| `ρ_t`, `v_t`, `o_t`, history validity | `build_progress_context_tn` |
| `q_prox/q_stall/q_osc/q_local`, `g_t` | `compute_refinement_gate` |
| budget selection | `select_retrieval_rows` |
| `c_teacher` | `teacher_confidence` |
| `δ_ε`, `w_ε`, retrieval `S` | `EliteSnippetBank.retrieve_candidates_batched` |
| `L_refine`, effective sample size | `compute_snippet_distill_loss` |
| `c` (curriculum progress) | `adaptive_precision_consolidation.curriculum_progress` |
| stage boundaries | `resolve_stage` / `resolve_pbt_stage(schedule_mode=curriculum_progress)` |
| probation / rescue | `ProbationState`, `evaluate_degradation_rescue`, `select_rescue_source` |
| rescue top-fraction | `rescue_source_top_k` / `rescue_source_top_fraction` (6 policies → top 3) |
| rollback target | `select_adaptive_rollback_target` (priority + log-gap, never `max(wider)`) |

## Adaptive PBT control chain (wired)

Adaptive probation and rescue are **execution-path gates**, not metrics-only:

1. **Shrink** — after mastery / adaptive-fast / safety / late-floor decide
   `shrink_token`, active probation may clear it via `ProbationState.can_shrink`
   and roll back single-shot bookkeeping (`adaptive_<block_reason>`).
2. **Ordinary rollback** — while probation is active and not hard-collapsed,
   ordinary forced-tolerance rollback (hard-collapse / repeated-collapse /
   late-floor) is blocked. Adaptive hard collapse lifts this suppression.
3. **Hard-collapse rollback** — adaptive hard collapse is an **independent**
   trigger (does not require legacy `evaluate_collapse_freeze`). Rollback target
   priority is previous → source → last mastered safe → scale fallback
   `current/shrink_factor` (capped by initial), each capped by
   `adaptive_probation_rollback_max_log_gap`. Illegal → freeze-only (no forced
   tolerance). Trigger source audit: `adaptive_only` / `legacy_only` /
   `adaptive_and_legacy` / `bridge`.
4. **Rescue cascade** — with `late_degradation_rescue_enable=False` and
   `adaptive_rescue_enable=True`, `_ternary_pbt` uses `rescue_mode=adaptive` and
   `_select_adaptive_rescue_source_cascade`: compatible live → validated archive
   (stable / floor / best-nsucc) → ordinary safe source. Peer sidecar parse /
   freshness / retention failures only reject that candidate; local formal
   adaptive state still fail-fasts. Legacy absolute tol-gap recheck applies
   **only** to `rescue_mode=legacy`.
5. **Source retention / freshness** — rescue sources need
   `min_samples` + `min_stable_ticks` (single-sample retention=1 is rejected)
   and a fresh sidecar (`last_update_frame` within
   `adaptive_rescue_source_max_staleness_frames`).
6. **Burn-in** — ordinary adaptive degradation does **not** bypass elimination
   burn-in (`adaptive_rescue_bypass_elimination_burnin=False` by default).
   Optional emergency bypass requires both bypass flags + adaptive hard collapse
   + age + stable/archive source.
7. **Quota** — `rescue_executed_for_lineage` increments only after a successful
   restart commit; lineage id change resets streak + lineage quota.
8. **Transaction** — adaptive sidecar bytes join curriculum control + manifest
   in the durable restart snapshot/rollback. Adaptive rescue tiers use recovery
   curriculum freeze frames.

Formal profile `cser_dexpbt_progress_refine` **fail-fasts** only on **local**
AdaptivePrecisionConfig / sidecar / update failures. Peer candidate failures are
soft rejects.

## Profile settings

The profile (`_CSER_DEXPBT_PROGRESS_REFINE` in `world_model/wm_strategy_profiles.py`)
inherits the generic `cser_dexpbt_conservative` substrate and:

- enables `progress_refinement_enable`, `adaptive_precision_consolidation_enable`,
  `adaptive_rescue_enable`, `adaptive_rescue_archive_enable`
- sets `pbt_consolidation_schedule_mode=curriculum_progress`,
  `eir_dhat_margin_mode=relative_tol`,
  `snippet_tolerance_compatibility_mode=log_relative`,
  `adaptive_probation_rollback_max_log_gap=0.30`,
  `adaptive_rescue_source_min_samples=3`,
  `adaptive_rescue_source_min_stable_ticks=2`,
  `adaptive_rescue_source_max_staleness_frames=40_000_000`,
  `adaptive_rescue_bypass_elimination_burnin=False` (emergency bypass opt-in),
  `adaptive_rescue_bypass_on_hard_collapse=True`,
  `adaptive_rescue_archive_min_nsucc_max=5.0`,
  `adaptive_rescue_archive_require_age=True`,
  `adaptive_rescue_archive_require_collapse_metadata=True`,
  `adaptive_rescue_archive_require_quality_score=True`
  (archive rescue fail-closed on incomplete metadata; ordinary fallback still
  consumes adaptive quota/cooldown but is counted as `ordinary_fallback`)
- adaptive bottom replacement uses `pbt_copy_allowed_for_adaptive_rescue_target`
  (not the generic stuck gate); hard-collapse emergency bypass uses a per-tick
  event-frame latch that survives streak reset
- disables `bridge_curriculum_enable`, `bridge_probation_enable`,
  `same_tol_retention_enable`, `late_degradation_rescue_enable`,
  `late_floor_retention_gate_enable`, `pre_floor_peak_guard_enable`,
  `snippet_adjacent_stage_mixing_enable`
- hard-off: `awr_enable=False`, `eir_use_rssm_mppi=False`, `eir_mppi_mode=off`,
  `eir_residual_apply_to_rollout=False`, `pbt_wm_profile_copy_enable=False`

Profile expansion is authoritative and re-applied after every PBT restart, so
these settings are restart-safe.

## Fallback logic

- WM unavailable / calibration not active / non-finite uncertainty → `c_wm = 1`;
  the snippet path stays open on real rollout labels + priority ranking.
- No tolerance-compatible snippet for a row → fail-closed (`retrieval_valid=0`,
  `w_t=0`), logged via `snippet_retrieval/no_compatible_frac` and
  `progress_refine/fallback_frac`.
- Any non-finite gate / weight / loss is replaced by a safe zero and counted in
  `progress_refine/nonfinite_count`; never propagated into PPO.

## Checkpoint compatibility

- Snippet bank schema is bumped to **v2**. v1 checkpoints migrate forward:
  `normalized_improvement` is reconstructed from stored distances; the remaining
  progress columns get neutral zeros; the migration is logged. New checkpoints
  retain all v2 fields.
- Old profiles, old checkpoints, and old launchers continue to work unchanged.
- Probation / retention / rescue state persists across restart in
  `{workspace}/adaptive_precision/{policy:03d}.json`.

## Metrics

`progress_refine/{gate_mean,gate_p50,gate_p90,active_frac,history_valid_frac,
normalized_error_mean,normalized_error_active_mean,recent_progress_mean,
recent_progress_active_mean,oscillation_mean,teacher_confidence_mean,
teacher_margin_mean,retrieval_valid_frac,retrieval_budget_frac,
effective_sample_size,loss,fallback_frac,nonfinite_count}`,
`snippet_retrieval/{tolerance_weight_mean,log_tolerance_gap_mean,no_compatible_frac}`,
`adaptive_consolidation/{curriculum_progress,stage,local_retention,
probation_active,probation_frames,shrink_blocked,probation_can_shrink,
hard_collapse_streak,degradation_streak,ordinary_rollback_blocked,
hard_collapse_rollback,rescue_ready,rescue_attempt_count,rescue_selected_count,
rescue_executed_count,rescue_blocked_count,rescue_source_policy,
rescue_source_rank,rescue_source_advantage_sigma,rescue_reason_hash,
rescue_quota_used}`. Registered in
`dexpbt_utils.PROGRESS_REFINE_TB_SCALAR_NAMES` (progress_refine scalars) with
additional adaptive consolidation scalars emitted by the observer.

## Launch commands

```bash
# Smoke (validates the full path end to end)
bash results/launchers/start_regrasping_cser_dexpbt_progress_refine_smoke.sh

# Formal 5B
bash results/launchers/start_regrasping_cser_dexpbt_progress_refine_5b.sh

# Ablation (CSER_ABLATION selects the variant)
CSER_ABLATION=full bash results/launchers/start_regrasping_cser_dexpbt_progress_refine_ablation.sh
```

## Ablation matrix

| `CSER_ABLATION` | progress gate | WM rerank | adaptive consolidation |
|-----------------|:-------------:|:---------:|:----------------------:|
| `basev2` | – (conservative base) | – | – |
| `snippet_only` | off | off | off |
| `snippet_wm` | off | on | off |
| `progress_gate_only` | on | off | off |
| `progress_gate_wm` | on | on | off |
| `adaptive_consolidation_only` | off | off | on |
| `full` | on | on | on |

## Analysis

```bash
python results/analysis/analyze_progress_refine.py --run-dir <RUN> \
  --compare-conservative <CONS_RUN> --compare-bridge <BRIDGE_RUN>
```

Checks: gate concentrates near the boundary in stagnant states; gate is
scale-invariant; WM actually changes snippet selection; loss is not permanently
zero; adaptive consolidation reduces post-shrink forgetting.

## Expected failure modes

- **Gate always ~0**: `start_after_frames`/`ramp_frames` not reached, or
  `only_lifted` filtering everything early. Check `progress_refine/history_valid_frac`.
- **`no_compatible_frac` high**: `tolerance_max_log_gap` too tight for a sparse
  bank; loosen it or warm the bank longer.
- **Loss permanently zero**: bank too small or teacher confidence ~0 (WM
  over-penalizing). Inspect `teacher_confidence_mean` and `effective_sample_size`.
- **Consolidation never reaches selection**: curriculum never approaches
  `ε_target`; check `adaptive_consolidation/curriculum_progress`.

## Rollback instructions

The method is additive and opt-in. To disable at any layer:

- Use a different profile (`cser_dexpbt_2cm_bridge`, `cser_dexpbt_conservative`,
  `cser_dexpbt`) — the new code paths are inert.
- Or pass `--no-progress_refinement_enable`,
  `--no-adaptive_precision_consolidation_enable`,
  `--pbt_consolidation_schedule_mode frames`,
  `--eir_dhat_margin_mode absolute`,
  `--snippet_tolerance_compatibility_mode absolute` to fall back to legacy
  behaviour while keeping the profile.
- No checkpoint migration is required to roll back; v2 banks are a superset of v1.
