# CCRS Final Interpretation — base-retention 5B

Date: 2026-07-19. Offline analysis of the completed 736-unit sweep; no new
GPU evaluation and no training were run for this document.

Run: `regrasping_cser_dexpbt_conservative_retention_strict_5b_20260716_130631`
Primary metric: `normalized_nsucc` Q(k, τ) = E[min(nsucc, 50)]/50.
Practical threshold (pre-registered): |ΔQ| ≥ 0.05 (≈ 2.5 raw nsucc).
All comparisons: strict reset-fingerprint pairing, 2048 paired samples
(4 seeds × 512 envs) per cell, hierarchical paired bootstrap
(seeds outer × env_ids inner, 2000 replicates). Reset pairing was valid for
every comparison below; all paired-sample sets were complete (2048/2048).

---

## 0. Code provenance

| field | value |
|---|---|
| training_commit (all checkpoints) | `5cb2c7cdafb12e50b9b7db88fdc47e7ef1056f86` |
| recorded_runtime_HEAD (git_sha in ccrs_results.csv) | `bc749d2f792fe330ca29755754bf19293229a5e3` |
| committed_runtime_source_snapshot | `2147387528e77071eba080d15b19978f2c09045f` |

The sweep executed while a small fix to `ccrs_eval.py` was still uncommitted
in the worktree, so the rows record HEAD `bc749d2…`; the exact source that
ran was committed immediately afterwards as `2147387…`. The fix only inlined
three loader helpers (`_load_agent_cfg`, `_load_env_cfg`,
`_filtered_restore`) that previously came from `eval_fair.py`, whose
module-level argparse/AppLauncher aborts on cross-script import. It did not
change policy execution, environment behavior, tolerance freezing, or metric
semantics. The recorded `git_sha` values in existing CSV/JSON are left
untouched.

To prevent this ambiguity in the future, `ccrs_eval` now additionally
records per row: `git_dirty` (worktree state), `evaluator_source_sha256`
(hash of the evaluator + protocol sources actually executing), and
`analysis_protocol_version` (currently `v2.1_frontier`). Covered by unit
tests.

---

## 1. Best-checkpoint capability envelope (12 key checkpoints)

Numerically best / second checkpoint per fixed tolerance, with the paired CI
of the lead. "Numerically best" and "practically significantly best" are
separate statements: a numerical lead below ΔQ = 0.05 is not practically
meaningful even when statistically significant.

| tol (cm) | best | Q / nsucc | 2nd | Q(2nd) | lead ΔQ [95% CI] | sig | practical |
|---|---|---|---|---|---|---|---|
| 2.8 | p5 peak | 0.908 / 45.4 | p2 target | 0.886 | +0.022 [+0.007, +0.036] | yes | no |
| 2.5 | p5 peak | 0.901 / 45.1 | p1 tip | 0.879 | +0.023 [+0.003, +0.044] | yes | no |
| 2.3 | p5 peak | 0.898 / 44.9 | p1 tip | 0.875 | +0.023 [+0.005, +0.041] | yes | no |
| 2.2 | p5 peak | 0.892 / 44.6 | p1 tip | 0.866 | +0.026 [+0.010, +0.043] | yes | no |
| 2.1 | p5 peak | 0.882 / 44.1 | p1 tip | 0.865 | +0.017 [+0.000, +0.034] | yes | no |
| 2.0 | p1 tip | 0.860 / 43.0 | p5 peak | 0.853 | +0.008 [−0.013, +0.028] | no | no |
| 1.9 | p1 tip | 0.838 / 41.9 | p5 peak | 0.813 | +0.025 [−0.004, +0.054] | no | no |
| 1.8 | p1 tip | 0.816 / 40.8 | p5 peak | 0.760 | +0.056 [+0.036, +0.076] | yes | **yes** |

Verification of the three focal claims:

- **p5 peak is numerically best at 2.8–2.1 cm** — confirmed at all five
  tolerances; the lead over the runner-up is statistically significant at
  each, but never reaches the 0.05 practical threshold.
- **p1 tip is numerically best at 2.0–1.8 cm** — confirmed at all three
  tolerances; the lead is statistically significant AND practical only at
  1.8 cm (+0.056 over p5 peak).
- **p1 tip vs p5 peak crossing**: **numerical crossing confirmed** (p5 peak
  ahead at 2.8–2.1 cm with significant diffs of 0.017–0.026; p1 tip ahead at
  2.0–1.8 cm). **Practical crossing NOT confirmed**: p5 peak's loose-region
  lead never reaches 0.05, and p1 tip's tight-region lead is practical only
  at 1.8 cm. Accurate statement: the two checkpoints' capability curves
  cross numerically around 2.0–2.1 cm, with a practically meaningful gap
  emerging only at the 1.8 cm end.

Figure: `figures/best_checkpoint_envelope.png`.

## 2. Peak → tip temporal comparisons (paired, per worker)

ΔQ = Q(tip) − Q(peak); * = significant AND ≥ 0.05 practical threshold.
Full per-tolerance tables with Q, raw nsucc, CIs in
`frontier_analysis.json`.

| tol (cm) | p1 tip−peak | p3 tip−peak | p4 tip−peak | p5 tip−peak |
|---|---|---|---|---|
| 2.8 | +0.009 | −0.076* | −0.042 | −0.118* |
| 2.5 | +0.013 | −0.046 | −0.038 | −0.122* |
| 2.3 | +0.031 | +0.026 | −0.046 | −0.140* |
| 2.2 | +0.034 | +0.050* | −0.017 | −0.147* |
| 2.1 | +0.037 | +0.081* | +0.002 | −0.145* |
| 2.0 | +0.068* | +0.153* | +0.023 | −0.135* |
| 1.9 | +0.076* | +0.215* | +0.059* | −0.101* |
| 1.8 | +0.106* | +0.304* | +0.125* | −0.083* |

- **p1 improved from peak to tip in the high-precision region** (practical
  at 2.0/1.9/1.8 cm; up to +0.106 at 1.8 cm) with no significant loss at any
  looser tolerance. p1's late training was genuine precision acquisition.
- **p5 suffered real forgetting at ALL eight tolerances** (−0.083 to −0.147,
  all significant and practical) — not criterion reclassification.
- p3 traded loose-region capability (−0.076* at 2.8 cm) for large
  high-precision gains (+0.304* at 1.8 cm).
- p4 shows a weaker version of the same trade (practical gains only at
  1.9/1.8 cm), and remains globally below p1/p5/p3 levels.

Figure: `figures/peak_tip_temporal_differences.png`.

## 3. Final scientific conclusions (supersede earlier drafts)

1. p1 tip reaches ≈43.0 raw nsucc at fixed 2.0 cm and ≈40.8 at fixed 1.8 cm
   (Q = 0.860 / 0.816).
2. This base-retention run has therefore broken the 2 cm fixed-precision
   barrier: sub-2 cm tolerances still yield ≥ 0.81 normalized capability.
3. p4's smaller live tolerance (0.0193 m at tip) does NOT correspond to
   stronger fixed-precision capability: p1 peak dominates p4 peak at all
   eight tolerances (ΔQ −0.066…−0.127, all significant, 6/8 practical).
4. The original p1/p4 Anchor-Scout hypothesis is refuted by CCRS: there is
   no p4-held high-precision capability that p1 lacks, hence no
   stable-vs-precise division of labor between them.
5. p5 peak → tip is real policy forgetting across the entire tolerance
   grid (all eight diffs significant and practical), not criterion
   reclassification.
6. p1 peak → tip continued to improve in the high-precision region
   (practical gains at 2.0–1.8 cm), showing that late-stage precision
   acquisition without forgetting is possible in this population.
7. Of the 22 evaluated exact transitions (p1/p4/p5), 21 satisfy |F| > |C|;
   the single exception (p5 it112→113) has both effects near zero.
8. The policy-update effect (F) is the dominant source of capability change
   in this experiment.
9. Criterion drift (C) is a minor factor in this run. "Criterion
   reclassification as the main driver of apparent degradation" must NOT be
   presented as a confirmed core finding of this work.
10. The core problem is hereby re-framed as **worker-dependent long-horizon
    capability retention**: with identical hyperparameter profiles, p1
    retained and refined while p5 (and to a lesser degree p3's loose region)
    forgot.
11. The 4 evaluation seeds quantify evaluation uncertainty of FIXED
    checkpoints only; they are not independent training seeds and say
    nothing about run-to-run training variance.
12. All conclusions derive from ONE population run. No claim about average
    method benefit or training stability across runs can be made from this
    data.

## 4. Scope of the D/F/C evidence

- Exact D/F/C was computed for the 22 in-band transitions of p1/p4/p5
  (real pre/post tolerances, no grid projection, all reset-pairing valid).
- The 15 in-band transitions of p0/p2/p3 have NOT been evaluated (their
  exact cells were deliberately excluded from this sweep).
- The "F-dominated" conclusion is strictly limited to the 22 evaluated
  transitions. The missing 15 are left un-evaluated for now (no new GPU
  work in this task).

## 5. Next-step method direction (candidate only — NOT validated)

**Capability-Frontier Retention** (design sketch, no training code
implemented, no experimental validation yet, no novelty claim):

- Maintain a multi-tolerance capability envelope over training history:
  each archived checkpoint carries a fixed-tolerance capability vector
  (its CCRS row) rather than a single live nsucc.
- The observed envelope motivates this: p5 peak owns the loose region
  (2.8–2.1 cm) while p1 tip owns the high-precision region (2.0–1.8 cm) —
  a single live scalar cannot represent both.
- PBT promotion/elimination would consult the capability vector instead of
  only live nsucc at live tolerance: a candidate replaces an archive member
  only if it significantly dominates it across the vector; members that are
  dominated get pruned; complementary (non-dominated) checkpoints are kept
  as a multi-point archive.
- Research focus: make p1-style late high-precision improvement
  reproducible while preventing p5-style long-horizon forgetting.
- Status: next-step candidate direction only. It has not been implemented
  or validated, and nothing here establishes novelty over existing
  quality-diversity / archive-based PBT literature.

## 6. Artifacts and tests

- `frontier_analysis.json` — full machine-readable comparisons/envelope/
  crossing (analysis_protocol_version `v2.1_frontier`).
- `figures/best_checkpoint_envelope.png`,
  `figures/peak_tip_temporal_differences.png` — new.
- `ccrs_summary.json` — `frontier` and `provenance` sections added.
- Analysis code: `dexpbt_isaaclab/ccrs/frontier.py` (offline, no Isaac).
- Tests: `tests/test_ccrs_toolkit.py` — 75 passed, including new tests for
  provenance fields (git_dirty, evaluator_source_sha256,
  analysis_protocol_version), paired comparison fields and completeness
  flags, numerical-vs-practical envelope separation, and crossing
  assessment.
- No GPU task was launched for this analysis.
