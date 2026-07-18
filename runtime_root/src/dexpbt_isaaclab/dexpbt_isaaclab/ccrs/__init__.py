"""CCRS (Criterion-Counterfactual Retention Surface) diagnostic toolkit.

Modules (pure python unless noted):
- eval_protocol: fixed-tolerance evaluation protocol helpers (no Isaac import).
- manifest:      checkpoint manifest builder from PBT workspace metas.
- analysis:      R(k, tau) tables, D/F/C decomposition, paired bootstrap,
                 pre-registered decision criteria.
- report:        figures + CCRS_REPORT.md + ccrs_summary.json generation.

GPU entry points live in dexpbt_isaaclab.scripts.ccrs_eval (single unit,
Isaac required) and dexpbt_isaaclab.scripts.ccrs_sweep (subprocess scheduler,
no Isaac import).

All tolerances in this package are expressed in meters (m). Reports may
additionally display centimeters, always derived as cm = m * 100.
"""
