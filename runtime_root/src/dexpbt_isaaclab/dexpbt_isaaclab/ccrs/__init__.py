"""CCRS (Criterion-Counterfactual Retention Surface) diagnostic toolkit.

Modules (pure python unless noted):
- eval_protocol:  fixed-tolerance evaluation protocol (schema v2): eval
                  units, config hashing, strict resume validation, per-env
                  samples, reset fingerprints, metric aggregation.
- rollout_collect: paired-reset rollout helpers: first-episode-per-env
                  collector, reset fingerprint extraction, eval-only
                  pre-reset terminal snapshot hook.
- manifest:       checkpoint manifest builder from PBT workspace metas
                  (incl. lineage-aware tolerance transitions).
- analysis:       Q(k, tau) tables (normalized_nsucc), exact vs
                  grid-projected D/F/C decomposition, reset-fingerprint
                  pairing validation, hierarchical paired bootstrap,
                  pre-registered decision criteria.
- report:         figures + CCRS_REPORT.md + ccrs_summary.json generation.

GPU entry points live in dexpbt_isaaclab.scripts.ccrs_eval (single unit,
Isaac required) and dexpbt_isaaclab.scripts.ccrs_sweep (subprocess scheduler,
no Isaac import).

PRIMARY metric: normalized_nsucc
  Q(k, tau) = E[min(nsucc, M) | checkpoint k, fixed tolerance tau] / M,
  M = max_consecutive_successes (default 50).
Auxiliary: nsucc_mean (raw), success_any_rate = P(nsucc > 0) (saturates,
never used for primary verdicts).

All tolerances in this package are expressed in meters (m). Reports may
additionally display centimeters, always derived as cm = m * 100.
"""
