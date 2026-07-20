# Real-data Cycle 1/2 selective replay

Source: `results/analysis/msa_stateful_n16_20260720/SELECTIVE_REPLAY.json`

Metric: `normalized_nsucc`; bootstrap seed 20260719, 10k reps, 5th pct.
ρ gate: existing point estimate ≥ 0.88 (no invented ρ CI).

## `msa_1784492852_5fa1236a`

- **global_panel**: promoted=False, gate=`panel_lower_quantile`, panel_lq=-0.07104492187500003
- **selective**: eligible=[2, 3], selected=[3, 2], anchors=[0, 1, 4, 5]
- **rejection**: `{'0': ['L_mid_lower_quantile<=0', 'same_tau_retention'], '1': ['L_mid_lower_quantile<=0'], '2': [], '3': [], '4': ['L_mid_lower_quantile<=0'], '5': ['L_mid_lower_quantile<=0', 'same_tau_retention']}`

## `msa_1784527942_af1468d9`

- **global_panel**: promoted=False, gate=`same_tau_retention_w5`, panel_lq=0.15985839843750005
- **selective**: eligible=[0, 1], selected=[0, 1], anchors=[2, 3, 4, 5]
- **rejection**: `{'0': [], '1': [], '5': ['same_tau_retention']}`
