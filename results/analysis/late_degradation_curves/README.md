# Late-degradation training curves

Generated with `results/analysis/plot_dexpbt_style_regrasping.py`.

| # | Run | Why picked |
|---|-----|------------|
| 01 | `regrasping_cser_dexpbt_progress_refine_abc_5b_20260714_222300` | Hard collapse p1/p3/p4 after ~2B |
| 02 | `regrasping_fixed_rssm_iterative_full_active_replay_3b_wired_20260610_020914` | Hard collapse p2/p3/p4 near ~3B / tight tol |
| 03 | `cont6_cser_2cm_bridge_1p5b_20260711_145415` | Clearest remaining late tip drops (p0/p4/p5) among longer runs |

Each subfolder has PNG + CSV from the plotter.
