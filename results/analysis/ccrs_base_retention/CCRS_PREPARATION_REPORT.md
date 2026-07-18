# CCRS Preparation Report — base-retention 5B (v2 protocol)

Date: 2026-07-18 (fix round after v1 review)
Repo: `lpdddddd/wmdexpbt-v2`, branch `base-retention`
HEAD before this fix round: `80446995018f46443d420dab422e69402aaa16e5`
Training commit for all checkpoints: `5cb2c7cdafb12e50b9b7db88fdc47e7ef1056f86`
(the CCRS toolkit commits are evaluation-only and postdate training)

Run under audit: `regrasping_cser_dexpbt_conservative_retention_strict_5b_20260716_130631`
Training status: **complete** — all six workers reached a 5.00B tick
(p2: 5.08B); no training processes remain; all GPUs idle.
No GPU task was launched during this preparation round.

---

## 1. What changed in this fix round (v1 -> v2 protocol)

### 1.1 Primary metric: normalized_nsucc (fixes saturation)

v1 used `P(nsucc > 0)` as the main metric. base-retention nsucc is typically
30–45, so that metric saturates at ~1.0 and cannot separate p1/p4/p5.

v2 primary capability surface:

    Q(k, τ) = E[min(nsucc, M) | checkpoint k, fixed tolerance τ] / M
    M = max_consecutive_successes = 50

- `normalized_nsucc` — PRIMARY CCRS capability metric (all curves, heatmap,
  crossing test, p5 peak-tip, frontier, default D/F/C).
- `nsucc_mean`, `nsucc_std` — raw capability metrics, kept for continuity.
- `success_any_rate = P(nsucc > 0)` — AUXILIARY only, never used for primary
  verdicts, and no longer described as `P(error <= τ)`.
- Pre-registered practical effect: `PREREG_MIN_EFFECT = 0.05` in
  normalized_nsucc (≈ 2.5 raw nsucc at M=50).

### 1.2 True paired resets (fingerprints, not seed integers)

- Protocol requires `num_episodes <= num_envs` (default 512/512, enforced in
  `EvalUnit.__post_init__`, `ccrs_eval` and `ccrs_sweep`).
- Only the FIRST completed episode of each env_id < num_episodes is recorded
  (`FirstEpisodeCollector` with `seen_env_ids`); completion order cannot
  change the sample set.
- Every sample is stored with `sample_id = seed_<seed>_env_<id>`, reset_seed,
  env_id, nsucc, normalized_nsucc, terminal_error(+valid), dropped.
- After the initial `env.reset()` and BEFORE the first action, a per-env
  **reset fingerprint** is computed over `object_pos`, `object_rot`,
  `goal_pos`, `goal_rot`, `dof_pos` (robot joints), with fixed-precision
  quantization (1e-5) before hashing. Raw results store both the per-env map
  and an aggregate fingerprint.
- Paired analysis (p1/p4, pre/post) first verifies fingerprint equality for
  every shared (seed, env_id). On any mismatch: paired bootstrap is refused,
  the report marks `reset_pairing_invalid`, and no significance conclusion is
  produced.
- Bootstrap is a **hierarchical paired bootstrap**: outer resample over
  seeds, inner resample over env_ids within each drawn seed, with the same
  resampled sample multiset applied to every cell. Statistical unit:
  seed × env_id paired samples.

### 1.3 Terminal error (fixes post-reset read)

Verified in `dexpbt_kuka_allegro_env.py`: `_reset_idx()` ends with
`_compute_intermediate_values()`, so after `env.step()` returns, `object_pos`
/ `goal_pos` for done envs already hold POST-reset state. The v1 read was
therefore invalid. Gymnasium `final_observation` is not provided by this
Direct-workflow env.

v2: an **eval-only `TerminalSnapshotHook`** wraps `env._reset_idx` inside the
evaluator process only; at call time the buffers still hold the pre-reset
terminal state, so it snapshots `||object_pos - goal_pos||` and
`dropped = object_pos.z < 0.1` for exactly the env_ids being reset.
- Training code is untouched (test asserts the env source has no CCRS
  references); hook default off, must be explicitly installed, uninstall
  restores the original method.
- If the hook cannot attach: `terminal_error = NaN`,
  `terminal_error_valid = false`, no terminal-based conclusions.

### 1.4 Incomplete & resume validation

- `status = ok` ONLY when `actual_num_episodes == requested_num_episodes`;
  otherwise `status = incomplete` with `requested/actual_num_episodes`,
  `max_steps_reached`, `missing_env_ids`, `stop_reason`.
- `is_done()` validates: status, `schema_version`, `config_hash` vs the
  current EvalUnit, actual==requested episodes, evaluation_mode,
  fixed_tolerance, checkpoint identity (path + frame + iteration), and reset
  protocol version. Incomplete results are never skipped.
- `EvalUnit.unit_key()/config_hash()` now include checkpoint_frame,
  checkpoint_path, max_steps, max_consecutive_successes,
  reset_protocol_version, metric_protocol_version.
- `SCHEMA_VERSION` bumped 1 -> 2; v1 dry/smoke results can never be mistaken
  for v2 results (excluded from resume, aggregation, and analysis tables).
- Raw JSON writes remain atomic (temp file + rename).

### 1.5 Exact vs grid-projected D/F/C, lineage

- **capability grid mode** (curves / crossing / heatmap / frontier): fixed
  grid [0.028, 0.025, 0.023, 0.022, 0.021, 0.020, 0.019, 0.018] m.
- **exact transition mode** (D/F/C): uses the REAL `pre_tolerance = a`,
  `post_tolerance = b` from `tolerance_transitions.csv`; needs the four cells
  Q(p,a), Q(p,b), Q(q,a), Q(q,b) evaluated at exactly a and b.
  `ccrs_sweep --transitions` schedules those cells automatically. Missing
  exact cells → the transition is reported under "missing exact transition
  cells" and NO D/F/C is generated for it.
- Nearest-grid analysis still exists but only as `grid_projected_dfc`,
  explicitly labelled, opt-in (`report --grid-projected`), never presented as
  exact and never used for primary conclusions.
- `TransitionPair` / `tolerance_transitions.csv` now carry `pre_lineage_id`,
  `post_lineage_id`, `lineage_changed`. Transitions with lineage change are
  labelled `lineage_replacement_transition` and reported separately: their F
  includes PBT copy/replacement effects and is excluded from within-lineage
  forgetting analysis. In the final manifest all 37 in-band transitions are
  within-lineage (lineage_changed=False).

D/F/C on Q (identity D = F + C preserved, tested):

    D = Q(p,a) − Q(q,b)
    F = ½[(Q(p,a)−Q(q,a)) + (Q(p,b)−Q(q,b))]
    C = ½[(Q(p,a)−Q(p,b)) + (Q(q,a)−Q(q,b))]

---

## 2. Final manifest (regenerated after training completion)

`checkpoint_manifest.csv`: 84 rows; 0 missing checkpoints, 0 zip-invalid;
12 key checkpoints torch_load-verified on CPU (all ok).
`tolerance_transitions.csv`: 37 in-band (0.018–0.0285 m) shrink transitions,
all within-lineage.

Key checkpoints (roles tip/peak/target — the 12-checkpoint main sweep set):

| worker | iter | frame | live tol (m) | live nsucc | roles |
|---|---|---|---|---|---|
| p0 | 160 | 5.00B | 0.0231 | 33.2 | tip;peak |
| p1 | 149 | 4.12B | 0.0237 | 43.4 | peak;target;transition_pre |
| p1 | 160 | 5.00B | 0.0203 | 40.0 | tip |
| p2 |  96 | 1.92B | 0.0365 | 44.8 | peak;target |
| p2 | 109 | 2.20B | 0.0297 | 44.7 | target |
| p2 | 161 | 5.08B | 0.0255 | 30.6 | tip |
| p3 | 159 | 4.92B | 0.0237 | 41.2 | peak;target |
| p3 | 160 | 5.00B | 0.0237 | 39.7 | tip |
| p4 | 141 | 3.48B | 0.0214 | 40.2 | peak;target |
| p4 | 160 | 5.00B | 0.0193 | 31.8 | tip |
| p5 | 117 | 2.52B | 0.0236 | 45.0 | peak;target;transition_post |
| p5 | 160 | 5.00B | 0.0213 | 35.7 | tip |

p1 now has an exact 5.00B tip (iteration 160); the earlier 4.84B provisional
tip is superseded. Nothing was fabricated: frames are read directly from the
workspace metadata.

## 3. Files changed in this round

- `dexpbt_isaaclab/ccrs/eval_protocol.py` — rewritten (schema v2, metrics,
  samples, fingerprints, strict is_done, incomplete status).
- `dexpbt_isaaclab/ccrs/rollout_collect.py` — NEW (FirstEpisodeCollector,
  extract_reset_fingerprints, TerminalSnapshotHook).
- `dexpbt_isaaclab/ccrs/analysis.py` — rewritten (Q tables, pairing checks,
  hierarchical bootstrap, exact vs grid_projected D/F/C, verdicts).
- `dexpbt_isaaclab/ccrs/manifest.py` — lineage fields on transitions,
  training_commit vs eval code commit separation.
- `dexpbt_isaaclab/ccrs/report.py` — rewritten (normalized_nsucc everywhere,
  exact D/F/C with missing-cell reporting, lineage sections, pairing flags).
- `dexpbt_isaaclab/scripts/ccrs_eval.py` — rewritten (paired resets,
  fingerprints, terminal hook, incomplete exit code, strict pre-boot checks).
- `dexpbt_isaaclab/scripts/ccrs_sweep.py` — exact-transition unit scheduling
  (`--transitions/--transition-workers`), num_episodes<=num_envs guard.
- `tests/test_ccrs_toolkit.py` — rewritten: 67 tests including all 18
  required behavioral tests (see §4).
- Regenerated artifacts: `checkpoint_manifest.csv`,
  `tolerance_transitions.csv`, `CHECKPOINT_AUDIT.md`, dry `CCRS_REPORT.md`,
  `ccrs_summary.json`, this report.

## 4. Test results

`python -m pytest tests/test_ccrs_toolkit.py -q` → **67 passed** (isaaclab311
env, no GPU, ~3 s). Also ran alongside existing suites
(`test_recover_best_nsucc_picker.py`, `test_curriculum_freeze.py`,
`test_curriculum_adaptive_fast.py`): **102 passed** total — eval_fair and
checkpoint-loading behavior untouched (`eval_fair.py` not modified;
`ccrs_eval` only imports its loaders).

All 18 required behavioral tests present and passing: normalized_nsucc
formula; success_any auxiliary-only; curve/DFC default metric; fingerprint
pairing accept + reject (incl. same-seed-different-fingerprint rejection);
first-episode-per-env; completion-order invariance of sample_ids;
incomplete status; incomplete-not-resumed; config-hash change invalidation;
checkpoint-identity change invalidation; terminal_error invalid without
reliable source (plus pre-reset hook behavior on a fake env reproducing the
v1 bug); exact transition uses real a/b; missing exact cells refused;
lineage change detection/exclusion; old-schema exclusion from aggregation,
resume and tables; empty-data report without fabrication.

Source-string checks (no optimizer/PBT/WM/snippet/MPPI in the evaluator) are
retained as secondary evidence only.

## 5. Sweep sizing (plan-only verified, nothing executed)

- Main grid: 12 checkpoints × 8 tolerances × 4 seeds = **384 units**.
- Exact transition cells (p1/p4/p5, 22 transitions): +44 checkpoint entries
  with per-checkpoint exact tolerance pairs → **736 units total** with grid.
- Defaults per unit: num_envs=512, num_episodes=512, deterministic, one
  Isaac process per unit, strict resume/skip.

## 6. Outstanding GPU validation (NOT done in this round)

- No GPU smoke test has been run under the v2 protocol; determinism of
  fingerprints across processes (same seed → identical per-env fingerprints
  for different checkpoints) is verified by construction and unit tests but
  not yet on the real env.
- Terminal snapshot hook has been validated against a fake env replicating
  the real reset flow, not yet against Isaac.
- Per-unit wall-time under 512 envs is unmeasured (v1 estimate: ~3–6 min).

## 7. Recommended two-stage commands (DO NOT run yet)

Stage 1 — 12-unit pre-check (p1 peak it149 + p4 peak it141, τ ∈ {0.022,
0.020, 0.019} m, seeds {10001, 10002}), single GPU:

    cd /root/autodl-tmp/wmdexpbt/runtime_root/src/dexpbt_isaaclab
    OUT=/root/autodl-tmp/wmdexpbt/results/analysis/ccrs_base_retention
    for it in "1 149 4120000000" "4 141 3480000000"; do
      set -- $it
      W=$1; IT=$2; FR=$3
      CKPT=$(python3 -c "import csv;print(next(r['checkpoint_path'] for r in csv.DictReader(open('$OUT/checkpoint_manifest.csv')) if r['worker_id']=='$W' and r['iteration']=='$IT'))")
      for TOL in 0.022 0.020 0.019; do
        for SEED in 10001 10002; do
          CUDA_VISIBLE_DEVICES=0 python3 -m dexpbt_isaaclab.scripts.ccrs_eval \
            --task DexPBT-Regrasping-KukaAllegro-Direct-v0 \
            --run-id regrasping_cser_dexpbt_conservative_retention_strict_5b_20260716_130631 \
            --worker-id $W --checkpoint-iteration $IT --checkpoint-frame $FR \
            --checkpoint-path "$CKPT" --fixed-tolerance $TOL --reset-seed $SEED \
            --num-envs 512 --num-episodes 512 --raw-dir $OUT/raw --headless
        done
      done
    done
    # then verify: distinguishable normalized_nsucc, identical fingerprints
    # across the two checkpoints per seed, paired analysis runs, terminal
    # metrics valid:
    python3 -m dexpbt_isaaclab.scripts.ccrs_sweep --aggregate-only \
      --manifest $OUT/checkpoint_manifest.csv --out-dir $OUT \
      --task DexPBT-Regrasping-KukaAllegro-Direct-v0
    python3 -m dexpbt_isaaclab.ccrs.report --results $OUT/ccrs_results.jsonl \
      --manifest $OUT/checkpoint_manifest.csv \
      --transitions $OUT/tolerance_transitions.csv --out-dir $OUT

Stage 2 — full sweep (ONLY after stage 1 passes), 6 GPU shards:

    for G in 0 1 2 3 4 5; do
      CUDA_VISIBLE_DEVICES=$G nohup python3 -m dexpbt_isaaclab.scripts.ccrs_sweep \
        --manifest $OUT/checkpoint_manifest.csv --roles tip,peak,target \
        --transitions $OUT/tolerance_transitions.csv --transition-workers 1,4,5 \
        --out-dir $OUT --task DexPBT-Regrasping-KukaAllegro-Direct-v0 \
        --num-envs 512 --num-episodes 512 \
        --execute --shard-index $G --num-shards 6 \
        > $OUT/sweep_shard$G.log 2>&1 &
    done

This preparation round ends here: code fixed, tested, manifest regenerated,
nothing executed on GPU.
