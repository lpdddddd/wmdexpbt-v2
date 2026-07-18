# CCRS Final Interpretation — base-retention 5B

Date: 2026-07-19 (revised same day: statistical labels and retention
direction clarified; see CHANGELOG at the end). Offline analysis of the
completed 736-unit sweep; no new GPU evaluation and no training were run for
this document.

Run: `regrasping_cser_dexpbt_conservative_retention_strict_5b_20260716_130631`
Primary metric: `normalized_nsucc` Q(k, τ) = E[min(nsucc, 50)]/50.
Practical threshold (pre-registered): |ΔQ| ≥ 0.05 (≈ 2.5 raw nsucc).
All comparisons: strict reset-fingerprint pairing, 2048 paired samples
(4 seeds × 512 envs) per cell, hierarchical paired bootstrap
(seeds outer × env_ids inner, 2000 replicates). Reset pairing was valid for
every comparison below; all paired-sample sets were complete (2048/2048).

**Statistical labels used throughout (three separate judgements, never
conflated):**

- **statistically significant** — the 95% paired CI excludes zero.
- **practical (point estimate)** — |ΔQ point estimate| ≥ 0.05.
- **practical (CI-confirmed)** — the WHOLE 95% CI lies beyond the 0.05
  threshold on the side of the point estimate (ci_lo ≥ +0.05 for positive
  diffs, ci_hi ≤ −0.05 for negative diffs).

An effect can be statistically significant and practical at point-estimate
scale while NOT CI-confirmed (its CI straddles the threshold). The former
`practical` flag (significant AND |point| ≥ 0.05) is deprecated and no
longer used in this document.

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
`analysis_protocol_version`; these fields are part of `CSV_COLUMNS`, and
aggregation now emits `aggregation_provenance.json` (unique git_sha /
evaluator hash / protocol version, any-dirty flag) and loudly flags
`mixed_provenance=true` if rows from different evaluator code versions are
ever aggregated together. The historical 736 rows report their evaluator
hash as `unrecorded` — they are never rewritten. Covered by unit tests.

---

## 1. Best-checkpoint capability envelope (12 key checkpoints)

> **EXPLORATORY / POST-SELECTION.** The per-tolerance winner is selected
> from the same data on which the pointwise CI is computed, and the CIs
> carry NO family-wise multiple-comparison correction. Nothing in this
> section is a confirmatory "significantly best" claim; confirmatory
> statements are restricted to the pre-named comparisons in Sections 2–3.

Numerically best / second checkpoint per fixed tolerance, with the paired CI
of the lead over the runner-up.

| tol (cm) | best | Q / nsucc | 2nd | Q(2nd) | lead ΔQ [95% CI] | stat. sig | practical (point) | practical (CI-conf.) |
|---|---|---|---|---|---|---|---|---|
| 2.8 | p5 peak | 0.908 / 45.4 | p2 target | 0.886 | +0.022 [+0.007, +0.036] | yes | no | no |
| 2.5 | p5 peak | 0.901 / 45.1 | p1 tip | 0.879 | +0.023 [+0.003, +0.044] | yes | no | no |
| 2.3 | p5 peak | 0.898 / 44.9 | p1 tip | 0.875 | +0.023 [+0.005, +0.041] | yes | no | no |
| 2.2 | p5 peak | 0.892 / 44.6 | p1 tip | 0.866 | +0.026 [+0.010, +0.043] | yes | no | no |
| 2.1 | p5 peak | 0.882 / 44.1 | p1 tip | 0.865 | +0.017 [+0.000, +0.034] | yes | no | no |
| 2.0 | p1 tip | 0.860 / 43.0 | p5 peak | 0.853 | +0.008 [−0.013, +0.028] | no | no | no |
| 1.9 | p1 tip | 0.838 / 41.9 | p5 peak | 0.813 | +0.025 [−0.004, +0.054] | no | no | no |
| 1.8 | p1 tip | 0.816 / 40.8 | p5 peak | 0.760 | +0.056 [+0.036, +0.076] | yes | yes | **no** |

Figure: `figures/best_checkpoint_envelope.png` (labelled exploratory).

## 2. p1 tip vs p5 peak: crossing and epsilon dominance (named comparison)

Full per-tolerance table (ΔQ = Q(p1 tip) − Q(p5 peak)):

| tol (cm) | ΔQ [95% CI] | stat. sig | practical (point) | practical (CI-conf.) |
|---|---|---|---|---|
| 2.8 | −0.025 [−0.046, −0.002] | yes | no | no |
| 2.5 | −0.023 [−0.044, −0.003] | yes | no | no |
| 2.3 | −0.023 [−0.041, −0.005] | yes | no | no |
| 2.2 | −0.026 [−0.043, −0.010] | yes | no | no |
| 2.1 | −0.017 [−0.034, −0.000] | yes | no | no |
| 2.0 | +0.008 [−0.013, +0.028] | no | no | no |
| 1.9 | +0.025 [−0.004, +0.054] | no | no | no |
| 1.8 | +0.056 [+0.036, +0.076] | **yes** | **yes** | **no** |

The focal 1.8 cm point reads: statistically significant — yes; practical at
point-estimate scale — yes (+0.056 ≥ 0.05); practical effect CI-confirmed —
**no** (ci_lo = +0.036 < +0.05).

Crossing assessment (A = p1 tip, B = p5 peak):

- **Numerical crossing: YES.** p5 peak is numerically ahead at all five
  loose tolerances (2.8–2.1 cm, significant diffs of 0.017–0.026); p1 tip is
  numerically ahead at all three tight tolerances (2.0–1.8 cm). The point
  estimates cross around 2.0–2.1 cm.
- **Practical crossing (point estimate): NO.** p5 peak's loose-region lead
  never reaches 0.05 at any tolerance.
- **Practical crossing (CI-confirmed): NO.** No point on either side has its
  whole CI beyond the threshold.
- A numerical crossing on its own is a weak observation and is NOT taken as
  evidence for any architectural decision (e.g. a multi-checkpoint archive).

Epsilon practical dominance (ε = 0.05, over all eight tolerances,
A ε-dominates B iff A is never worse than B by more than ε and is better
than B by more than ε somewhere):

| A vs B | point-estimate ε-dominance | CI-confirmed ε-dominance |
|---|---|---|
| p1 tip vs p5 peak | **YES** (worst ΔQ −0.026 ≥ −0.05; +0.056 at 1.8 cm) | **NO** (no ci_lo ≥ +0.05; worst ci_lo −0.046 ≥ −0.05 holds) |
| p1 tip vs p1 peak | YES | YES (ci_lo +0.085 ≥ +0.05 at 1.8 cm; worst ci_lo −0.012) |
| p1 tip vs p3 tip | YES (all 8 diffs ≥ +0.05) | YES |
| p1 tip vs p4 tip | YES (all 8 diffs ≥ +0.05) | YES |
| p1 tip vs p5 tip | YES (all 8 diffs ≥ +0.05) | YES |

Summary: **p1 tip ε-dominates p5 peak at point-estimate scale** (it is never
worse by more than 0.05 anywhere and better by more than 0.05 at 1.8 cm),
but **this dominance is NOT CI-confirmed**. Against every other final
checkpoint (p1 peak, p3/p4/p5 tips) p1 tip's ε-dominance IS CI-confirmed.

## 3. Peak → tip temporal comparisons (paired, per worker, named)

ΔQ = Q(tip) − Q(peak). Symbols: `*` = practical CI-confirmed;
`+` = statistically significant with practical point estimate (CI straddles
threshold); unmarked = neither. Full tables with Q, raw nsucc, CIs in
`frontier_analysis.json`.

| tol (cm) | p1 tip−peak | p3 tip−peak | p4 tip−peak | p5 tip−peak |
|---|---|---|---|---|
| 2.8 | +0.009 | −0.076* | −0.042 | −0.118* |
| 2.5 | +0.013 | −0.046 | −0.038 | −0.122* |
| 2.3 | +0.031 | +0.026 | −0.046 | −0.140* |
| 2.2 | +0.034 | +0.050+ | −0.017 | −0.147* |
| 2.1 | +0.037 | +0.081* | +0.002 | −0.145* |
| 2.0 | +0.068+ | +0.153* | +0.023 | −0.135* |
| 1.9 | +0.076+ | +0.215* | +0.059+ | −0.101* |
| 1.8 | +0.106* | +0.304* | +0.125* | −0.083* |

- **p1 improved from peak to tip in the high-precision region**: practical
  point estimates at 2.0/1.9/1.8 cm, CI-confirmed at 1.8 cm (+0.106,
  CI [+0.085, +0.126]); no significant loss at any looser tolerance. p1's
  late training was genuine precision acquisition.
- **p5 suffered real forgetting at ALL eight tolerances** (−0.083 to −0.147,
  every point CI-confirmed practical) — not criterion reclassification.
  This is the strongest single result in the dataset.
- p3 traded loose-region capability (−0.076, CI-confirmed at 2.8 cm) for
  large high-precision gains (up to +0.304, CI-confirmed, at 1.8 cm).
- p4 shows a weaker version of the same trade (CI-confirmed gain only at
  1.8 cm), and remains globally below p1/p5/p3 levels.

Figure: `figures/peak_tip_temporal_differences.png`.

## 4. Final scientific conclusions (supersede earlier drafts)

1. **In this single population run, the p1 tip checkpoint achieved
   approximately 43.0 nsucc at fixed 2.0 cm and 40.8 at fixed 1.8 cm**
   (Q = 0.860 / 0.816). This is a statement about one checkpoint of one
   run under this evaluation protocol — not a claim that the method
   "breaks the 2 cm barrier" in general.
2. p5 peak leads numerically at 2.8–2.1 cm, but every lead is below the
   0.05 practical threshold. p1 tip's lead at 1.8 cm exceeds 0.05 at
   point-estimate scale (+0.056) but its CI does not fully clear the
   threshold (ci_lo +0.036). **This run does not confirm a practical
   capability crossing between p1 tip and p5 peak.**
3. p1 tip ε-dominates (ε = 0.05) p5 peak at point-estimate scale, not
   CI-confirmed; it CI-confirmed ε-dominates p1 peak and the p3/p4/p5 tips.
   On current evidence, choosing the single checkpoint p1 tip is not
   practically worse than keeping p5 peak anywhere on the grid.
4. **A multi-checkpoint frontier archive remains a candidate hypothesis
   only.** The current data are insufficient to show it outperforms simply
   selecting a single strong checkpoint such as p1 tip. p5 peak's main
   value is providing an unambiguous counterexample of long-horizon
   forgetting (see 6), not serving as a necessary archive member.
5. p4's smaller live tolerance (0.0193 m at tip) does NOT correspond to
   stronger fixed-precision capability: p1 peak leads p4 peak at all eight
   tolerances (ΔQ 0.066…0.127, all statistically significant, 6/8 above the
   practical threshold at point-estimate scale). The original p1/p4
   Anchor-Scout hypothesis is refuted by CCRS.
6. p5 peak → tip is real policy forgetting across the entire tolerance grid
   (all eight diffs CI-confirmed practical), not criterion reclassification.
7. p1 peak → tip continued to improve in the high-precision region,
   showing that late-stage precision acquisition without forgetting is
   possible in this population.
8. Of the 22 evaluated exact transitions (p1/p4/p5), 21 satisfy |F| > |C|;
   the single exception (p5 it112→113) has both effects near zero. Within
   those 22 transitions the policy-update effect (F) is the dominant source
   of capability change, and criterion drift (C) is a minor factor. This
   conclusion is strictly limited to the 22 evaluated transitions: the 15
   exact transitions of p0/p2/p3 were not evaluated.
9. The core problem is re-framed as **worker-dependent long-horizon
   capability retention**: with identical hyperparameter profiles, p1
   retained and refined while p5 (and to a lesser degree p3's loose region)
   forgot.
10. Scope: all conclusions derive from ONE population training run. The 4
    evaluation seeds quantify evaluation uncertainty of FIXED checkpoints
    only — they are not training seeds and say nothing about run-to-run
    variance. No claim about the average benefit of base-retention over a
    baseline, or about cross-training-seed stability, can be made from
    these data.

## 5. Scope of the D/F/C evidence

- Exact D/F/C was computed for the 22 in-band transitions of p1/p4/p5
  (real pre/post tolerances, no grid projection, all reset-pairing valid).
- The 15 in-band transitions of p0/p2/p3 have NOT been evaluated (their
  exact cells were deliberately excluded from this sweep).
- The "F-dominated" conclusion is strictly limited to the 22 evaluated
  transitions. The missing 15 are left un-evaluated for now (no new GPU
  work in this task).

## 6. Next-step method direction (candidate only — NOT validated)

**Fixed-Tolerance Capability Retention / Sentinel-Grid Retention** (design
sketch; not implemented, not validated, no novelty claim):

- Maintain a small set of fixed tolerance **sentinels**, e.g. 0.025, 0.022,
  0.020, 0.018 m, evaluated under the frozen CCRS protocol (fixed
  criterion, paired resets).
- Periodically compare the candidate policy against a retained anchor on a
  fixed reset probe at the sentinel tolerances, producing a small
  capability vector per candidate.
- When the capability vector regresses beyond a confidence threshold at any
  sentinel, trigger one of three INDEPENDENT candidate mechanisms (to be
  studied separately, not implemented together):
  a. update rejection;
  b. rollback;
  c. anchor distillation.
- A multi-checkpoint archive is deliberately NOT part of this sketch. It
  would only be justified as an upgrade if a true practical
  (CI-confirmed) curve crossing is observed across MULTIPLE independent
  training runs — this run observed only a numerical crossing.
- Motivation from this dataset: p5-style long-horizon forgetting is real
  and large (CI-confirmed at every tolerance), while p1 shows retention
  plus late precision acquisition is achievable; the mechanism above
  targets making the p1 outcome reproducible.
- Status: unimplemented, unvalidated, and nothing here establishes novelty
  over existing regression-testing / distillation / archive-based PBT
  literature.

## 7. Artifacts and tests

- `frontier_analysis.json` — full machine-readable comparisons (8 named),
  exploratory envelope, three-level crossing assessment, ε-dominance
  results.
- `aggregation_provenance.json` — provenance summary of the 736 aggregated
  rows (single git_sha `bc749d2…`, evaluator hash unrecorded for
  historical rows, mixed_provenance=false).
- `figures/best_checkpoint_envelope.png` (labelled exploratory),
  `figures/peak_tip_temporal_differences.png` (CI-confirmed vs
  point-estimate markers).
- `ccrs_summary.json` — `frontier` (three-level labels, ε-dominance) and
  `provenance` sections.
- Analysis code: `dexpbt_isaaclab/ccrs/frontier.py` (offline, no Isaac).
- No GPU task was launched for this analysis.

## CHANGELOG

- **2026-07-19 (rev. 2 — this document)**: interpretation and statistical
  labels only; NO new evaluation data and NO change to any recorded result
  rows. Replaced the mixed `practical` flag with three explicit judgements
  (statistically significant / practical point estimate / practical
  CI-confirmed); added ε-dominance analysis (ε = 0.05) for five named
  pairs; labelled the best-checkpoint envelope exploratory/post-selection
  without multiplicity correction; removed the unsupported
  "complementary checkpoints ⇒ multi-point archive" claim; re-scoped the
  "2 cm barrier" statement to the single run/checkpoint; replaced the
  Capability-Frontier Retention sketch with Fixed-Tolerance / Sentinel-Grid
  Retention; added end-to-end provenance fields (git_dirty,
  evaluator_source_sha256, analysis_protocol_version) to CSV_COLUMNS and an
  aggregation provenance summary with mixed-provenance flagging.
- **2026-07-19 (rev. 1)**: initial final interpretation after the 736-unit
  sweep.
