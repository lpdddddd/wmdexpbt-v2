# CCRS Report — base-retention 5B

## Pre-registered decision criteria (frozen before sweep results)

PRIMARY metric: normalized_nsucc Q(k, τ) = E[min(nsucc, M)]/M, M=50.
Effect threshold: |Δ normalized_nsucc| >= 0.05 (≈ 2.5 raw nsucc at M=50)
with 95% hierarchical paired-bootstrap CI excluding 0, computed ONLY when
reset fingerprints of the compared cells match exactly. success_any_rate
(P(nsucc>0)) is auxiliary and never used for these verdicts.
Tolerances in meters (cm shown for readability).

- **A (crossing)**: p4(peak) significantly better than p1(peak) at
  τ ∈ {0.020, 0.019, 0.018} AND p1 significantly better at
  τ ∈ {0.023, 0.022, 0.021} → capability curve crossing; population has a
  stable-capability vs high-precision division of labor → proceed to
  Anchor-Scout CD-ReST validation.
- **B (p4 dominant)**: p4 better on ≥70% of grid tolerances and all
  significant cells favor p4 → p4 is globally stronger; continue fine
  curriculum from p4 rather than adding cross-worker transfer mechanisms.
- **C (p1 still better at 0.020 m)**: p4's low live tolerance does not prove
  high-precision capability; audit dynamic curriculum, sampling variance and
  evaluation criteria.
- **D (p5 peak vs tip)**: significant fixed-tolerance Q drop (peak - tip) →
  real policy forgetting; otherwise (live nsucc drop but flat fixed-tol Q)
  → criterion reclassification dominant.
- If reset pairing is invalid for a comparison, its verdict degrades to
  reset_pairing_invalid and NO significance conclusion is produced.


**No evaluation data found.** Run the sweep first; this report was generated in dry mode and contains no results.
