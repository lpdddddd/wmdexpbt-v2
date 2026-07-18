# Strategy Zoo v17 Online Elite-AWR Report

Run: `/root/autodl-tmp/wmdexpbt/results/runs/regrasping_pbt_wm_strategy_zoo_v17_online_elite_awr_2b_debug_20260619_040703`

## A. Same-frame nsucc
### 0.5B
- p0: 0.000
- p1: 0.000
- p2: 14.784
- p3: 0.000
- p4: 0.000
- p5: 2.871
- top1 mean: 14.784423828125
- top3 mean: 5.885172526041667

### 1.0B
- p0: 0.000
- p1: 0.000
- p2: 23.644
- p3: 0.000
- p4: 0.000
- p5: 2.871
- top1 mean: 23.6435546875
- top3 mean: 8.838216145833334

### 1.5B
- p0: 0.000
- p1: 0.000
- p2: 28.874
- p3: 0.000
- p4: 0.000
- p5: 2.871
- top1 mean: 28.8743896484375
- top3 mean: 10.581827799479166

### 2.0B
- p0: 0.000
- p1: 0.000
- p2: 35.174
- p3: 0.000
- p4: 0.000
- p5: 2.871
- top1 mean: 35.174072265625
- top3 mean: 12.681722005208334

## B. AWR active table
### p0
- profile: ppo_only
- awr_active: False
- awr_loss_weight: nan
- buffer_size: nan
- buffer_steps: nan
- sidecar_steps: nan
- sidecar_save_failed: nan
- action_convention_checked: 0.0
- action_match_error_mean: nan
- action_match_error_max: nan
- action_match_order_code: 0.0
- quality_mean: {'last': nan, 'mean': nan, 'late_mean': nan}
- delta_gate_health: {'full_frac': {'last': nan, 'mean': nan, 'late_mean': nan}, 'mid_frac': {'last': nan, 'mean': nan, 'late_mean': nan}, 'far_frac': {'last': nan, 'mean': nan, 'late_mean': nan}, 'drop_frac': {'last': nan, 'mean': nan, 'late_mean': nan}}
- local_inherited_status: {'buffer_steps': nan, 'inherit_loaded_steps': 0.0, 'sidecar_steps': nan}
- action_convention_result: {'checked': 0.0, 'mismatch': 0.0, 'order_code': 0.0}
- awr_weighted_loss: {'last': nan, 'mean': nan, 'late_mean': nan}
- awr_active_frac: {'last': 0.0, 'mean': 0.0, 'late_mean': 0.0}
- delta_gate_drop_frac: {'last': nan, 'mean': nan, 'late_mean': nan}
- kl_gate_enabled: 0.0
- kl_telemetry_only: 1.0
- kl_skip_count: 0.0
- loss_cap_hit_frac: {'last': nan, 'mean': nan, 'late_mean': nan}
- buffer_inherit_loaded_steps: 0.0
- buffer_broadcast_loaded_steps: 0.0
- broadcast_enabled: 0.0
- collection_mode: unknown
- rollout_builder_entries: 0.0
- rollout_builder_selected_steps: 0.0
- rollout_builder_time_ms: 0.0
- skipped_collect_not_started: 0.0
- action_source_code: 0.0
- action_out_of_bounds_frac_1: 0.0
- ppo_buffer_action_min: nan
- ppo_buffer_action_max: nan
- fps_last: nan

### p1
- profile: elite_bank_actor_distill_fast_awr
- awr_active: False
- awr_loss_weight: nan
- buffer_size: 2000.0
- buffer_steps: 400000.0
- sidecar_steps: nan
- sidecar_save_failed: nan
- action_convention_checked: 1.0
- action_match_error_mean: 0.16858915984630585
- action_match_error_max: 4.018723487854004
- action_match_order_code: 9.0
- quality_mean: {'last': nan, 'mean': nan, 'late_mean': nan}
- delta_gate_health: {'full_frac': {'last': nan, 'mean': nan, 'late_mean': nan}, 'mid_frac': {'last': nan, 'mean': nan, 'late_mean': nan}, 'far_frac': {'last': nan, 'mean': nan, 'late_mean': nan}, 'drop_frac': {'last': nan, 'mean': nan, 'late_mean': nan}}
- local_inherited_status: {'buffer_steps': 400000.0, 'inherit_loaded_steps': 0.0, 'sidecar_steps': nan}
- action_convention_result: {'checked': 1.0, 'mismatch': 1.0, 'order_code': 9.0}
- awr_weighted_loss: {'last': nan, 'mean': nan, 'late_mean': nan}
- awr_active_frac: {'last': 0.0, 'mean': 0.0, 'late_mean': 0.0}
- delta_gate_drop_frac: {'last': nan, 'mean': nan, 'late_mean': nan}
- kl_gate_enabled: 0.0
- kl_telemetry_only: 1.0
- kl_skip_count: 0.0
- loss_cap_hit_frac: {'last': nan, 'mean': nan, 'late_mean': nan}
- buffer_inherit_loaded_steps: 0.0
- buffer_broadcast_loaded_steps: 0.0
- broadcast_enabled: 0.0
- collection_mode: unknown
- rollout_builder_entries: 0.0
- rollout_builder_selected_steps: 0.0
- rollout_builder_time_ms: 0.0
- skipped_collect_not_started: 0.0
- action_source_code: 0.0
- action_out_of_bounds_frac_1: 0.0
- ppo_buffer_action_min: -4.864041805267334
- ppo_buffer_action_max: 5.018723487854004
- fps_last: nan

### p2
- profile: wm_pbt_source_selector_v2
- awr_active: False
- awr_loss_weight: nan
- buffer_size: nan
- buffer_steps: nan
- sidecar_steps: nan
- sidecar_save_failed: nan
- action_convention_checked: 0.0
- action_match_error_mean: nan
- action_match_error_max: nan
- action_match_order_code: 0.0
- quality_mean: {'last': nan, 'mean': nan, 'late_mean': nan}
- delta_gate_health: {'full_frac': {'last': nan, 'mean': nan, 'late_mean': nan}, 'mid_frac': {'last': nan, 'mean': nan, 'late_mean': nan}, 'far_frac': {'last': nan, 'mean': nan, 'late_mean': nan}, 'drop_frac': {'last': nan, 'mean': nan, 'late_mean': nan}}
- local_inherited_status: {'buffer_steps': nan, 'inherit_loaded_steps': 0.0, 'sidecar_steps': nan}
- action_convention_result: {'checked': 0.0, 'mismatch': 0.0, 'order_code': 0.0}
- awr_weighted_loss: {'last': nan, 'mean': nan, 'late_mean': nan}
- awr_active_frac: {'last': 0.0, 'mean': 0.0, 'late_mean': 0.0}
- delta_gate_drop_frac: {'last': nan, 'mean': nan, 'late_mean': nan}
- kl_gate_enabled: 0.0
- kl_telemetry_only: 1.0
- kl_skip_count: 0.0
- loss_cap_hit_frac: {'last': nan, 'mean': nan, 'late_mean': nan}
- buffer_inherit_loaded_steps: 0.0
- buffer_broadcast_loaded_steps: 0.0
- broadcast_enabled: 0.0
- collection_mode: unknown
- rollout_builder_entries: 0.0
- rollout_builder_selected_steps: 0.0
- rollout_builder_time_ms: 0.0
- skipped_collect_not_started: 0.0
- action_source_code: 0.0
- action_out_of_bounds_frac_1: 0.0
- ppo_buffer_action_min: nan
- ppo_buffer_action_max: nan
- fps_last: nan

### p3
- profile: wm_pbt_source_selector_awr
- awr_active: False
- awr_loss_weight: nan
- buffer_size: 2000.0
- buffer_steps: 400000.0
- sidecar_steps: nan
- sidecar_save_failed: nan
- action_convention_checked: 1.0
- action_match_error_mean: 0.16802595555782318
- action_match_error_max: 4.013198375701904
- action_match_order_code: 9.0
- quality_mean: {'last': nan, 'mean': nan, 'late_mean': nan}
- delta_gate_health: {'full_frac': {'last': nan, 'mean': nan, 'late_mean': nan}, 'mid_frac': {'last': nan, 'mean': nan, 'late_mean': nan}, 'far_frac': {'last': nan, 'mean': nan, 'late_mean': nan}, 'drop_frac': {'last': nan, 'mean': nan, 'late_mean': nan}}
- local_inherited_status: {'buffer_steps': 400000.0, 'inherit_loaded_steps': 0.0, 'sidecar_steps': nan}
- action_convention_result: {'checked': 1.0, 'mismatch': 1.0, 'order_code': 9.0}
- awr_weighted_loss: {'last': nan, 'mean': nan, 'late_mean': nan}
- awr_active_frac: {'last': 0.0, 'mean': 0.0, 'late_mean': 0.0}
- delta_gate_drop_frac: {'last': nan, 'mean': nan, 'late_mean': nan}
- kl_gate_enabled: 0.0
- kl_telemetry_only: 1.0
- kl_skip_count: 0.0
- loss_cap_hit_frac: {'last': nan, 'mean': nan, 'late_mean': nan}
- buffer_inherit_loaded_steps: 0.0
- buffer_broadcast_loaded_steps: 0.0
- broadcast_enabled: 0.0
- collection_mode: unknown
- rollout_builder_entries: 0.0
- rollout_builder_selected_steps: 0.0
- rollout_builder_time_ms: 0.0
- skipped_collect_not_started: 0.0
- action_source_code: 0.0
- action_out_of_bounds_frac_1: 0.0
- ppo_buffer_action_min: -5.013198375701904
- ppo_buffer_action_max: 5.006488800048828
- fps_last: nan

### p4
- profile: elite_bank_actor_distill_awr
- awr_active: False
- awr_loss_weight: nan
- buffer_size: 2000.0
- buffer_steps: 400000.0
- sidecar_steps: nan
- sidecar_save_failed: nan
- action_convention_checked: 1.0
- action_match_error_mean: 0.16829709708690643
- action_match_error_max: 3.9264445304870605
- action_match_order_code: 9.0
- quality_mean: {'last': nan, 'mean': nan, 'late_mean': nan}
- delta_gate_health: {'full_frac': {'last': nan, 'mean': nan, 'late_mean': nan}, 'mid_frac': {'last': nan, 'mean': nan, 'late_mean': nan}, 'far_frac': {'last': nan, 'mean': nan, 'late_mean': nan}, 'drop_frac': {'last': nan, 'mean': nan, 'late_mean': nan}}
- local_inherited_status: {'buffer_steps': 400000.0, 'inherit_loaded_steps': 0.0, 'sidecar_steps': nan}
- action_convention_result: {'checked': 1.0, 'mismatch': 1.0, 'order_code': 9.0}
- awr_weighted_loss: {'last': nan, 'mean': nan, 'late_mean': nan}
- awr_active_frac: {'last': 0.0, 'mean': 0.0, 'late_mean': 0.0}
- delta_gate_drop_frac: {'last': nan, 'mean': nan, 'late_mean': nan}
- kl_gate_enabled: 0.0
- kl_telemetry_only: 1.0
- kl_skip_count: 0.0
- loss_cap_hit_frac: {'last': nan, 'mean': nan, 'late_mean': nan}
- buffer_inherit_loaded_steps: 0.0
- buffer_broadcast_loaded_steps: 0.0
- broadcast_enabled: 0.0
- collection_mode: unknown
- rollout_builder_entries: 0.0
- rollout_builder_selected_steps: 0.0
- rollout_builder_time_ms: 0.0
- skipped_collect_not_started: 0.0
- action_source_code: 0.0
- action_out_of_bounds_frac_1: 0.0
- ppo_buffer_action_min: -4.824590682983398
- ppo_buffer_action_max: 4.9264445304870605
- fps_last: nan

### p5
- profile: elite_bank_actor_distill_strong_awr
- awr_active: False
- awr_loss_weight: nan
- buffer_size: 2000.0
- buffer_steps: 400000.0
- sidecar_steps: nan
- sidecar_save_failed: nan
- action_convention_checked: 1.0
- action_match_error_mean: 0.16834045946598053
- action_match_error_max: 4.372122764587402
- action_match_order_code: 9.0
- quality_mean: {'last': nan, 'mean': nan, 'late_mean': nan}
- delta_gate_health: {'full_frac': {'last': nan, 'mean': nan, 'late_mean': nan}, 'mid_frac': {'last': nan, 'mean': nan, 'late_mean': nan}, 'far_frac': {'last': nan, 'mean': nan, 'late_mean': nan}, 'drop_frac': {'last': nan, 'mean': nan, 'late_mean': nan}}
- local_inherited_status: {'buffer_steps': 400000.0, 'inherit_loaded_steps': 0.0, 'sidecar_steps': nan}
- action_convention_result: {'checked': 1.0, 'mismatch': 1.0, 'order_code': 9.0}
- awr_weighted_loss: {'last': nan, 'mean': nan, 'late_mean': nan}
- awr_active_frac: {'last': 0.0, 'mean': 0.0, 'late_mean': 0.0}
- delta_gate_drop_frac: {'last': nan, 'mean': nan, 'late_mean': nan}
- kl_gate_enabled: 0.0
- kl_telemetry_only: 1.0
- kl_skip_count: 0.0
- loss_cap_hit_frac: {'last': nan, 'mean': nan, 'late_mean': nan}
- buffer_inherit_loaded_steps: 0.0
- buffer_broadcast_loaded_steps: 0.0
- broadcast_enabled: 0.0
- collection_mode: unknown
- rollout_builder_entries: 0.0
- rollout_builder_selected_steps: 0.0
- rollout_builder_time_ms: 0.0
- skipped_collect_not_started: 0.0
- action_source_code: 0.0
- action_out_of_bounds_frac_1: 0.0
- ppo_buffer_action_min: -5.372122764587402
- ppo_buffer_action_max: 5.222724437713623
- fps_last: nan

## B2. AWR health summary
### Local/inherited status
{'p0': {'buffer_steps': nan, 'inherit_loaded_steps': 0.0, 'sidecar_steps': nan}, 'p1': {'buffer_steps': 400000.0, 'inherit_loaded_steps': 0.0, 'sidecar_steps': nan}, 'p2': {'buffer_steps': nan, 'inherit_loaded_steps': 0.0, 'sidecar_steps': nan}, 'p3': {'buffer_steps': 400000.0, 'inherit_loaded_steps': 0.0, 'sidecar_steps': nan}, 'p4': {'buffer_steps': 400000.0, 'inherit_loaded_steps': 0.0, 'sidecar_steps': nan}, 'p5': {'buffer_steps': 400000.0, 'inherit_loaded_steps': 0.0, 'sidecar_steps': nan}}
### Action convention check result
{'p0': {'checked': 0.0, 'mismatch': 0.0, 'order_code': 0.0}, 'p1': {'checked': 1.0, 'mismatch': 1.0, 'order_code': 9.0}, 'p2': {'checked': 0.0, 'mismatch': 0.0, 'order_code': 0.0}, 'p3': {'checked': 1.0, 'mismatch': 1.0, 'order_code': 9.0}, 'p4': {'checked': 1.0, 'mismatch': 1.0, 'order_code': 9.0}, 'p5': {'checked': 1.0, 'mismatch': 1.0, 'order_code': 9.0}}
### Delta gate health
{'p0': {'full_frac': {'last': nan, 'mean': nan, 'late_mean': nan}, 'mid_frac': {'last': nan, 'mean': nan, 'late_mean': nan}, 'far_frac': {'last': nan, 'mean': nan, 'late_mean': nan}, 'drop_frac': {'last': nan, 'mean': nan, 'late_mean': nan}}, 'p1': {'full_frac': {'last': nan, 'mean': nan, 'late_mean': nan}, 'mid_frac': {'last': nan, 'mean': nan, 'late_mean': nan}, 'far_frac': {'last': nan, 'mean': nan, 'late_mean': nan}, 'drop_frac': {'last': nan, 'mean': nan, 'late_mean': nan}}, 'p2': {'full_frac': {'last': nan, 'mean': nan, 'late_mean': nan}, 'mid_frac': {'last': nan, 'mean': nan, 'late_mean': nan}, 'far_frac': {'last': nan, 'mean': nan, 'late_mean': nan}, 'drop_frac': {'last': nan, 'mean': nan, 'late_mean': nan}}, 'p3': {'full_frac': {'last': nan, 'mean': nan, 'late_mean': nan}, 'mid_frac': {'last': nan, 'mean': nan, 'late_mean': nan}, 'far_frac': {'last': nan, 'mean': nan, 'late_mean': nan}, 'drop_frac': {'last': nan, 'mean': nan, 'late_mean': nan}}, 'p4': {'full_frac': {'last': nan, 'mean': nan, 'late_mean': nan}, 'mid_frac': {'last': nan, 'mean': nan, 'late_mean': nan}, 'far_frac': {'last': nan, 'mean': nan, 'late_mean': nan}, 'drop_frac': {'last': nan, 'mean': nan, 'late_mean': nan}}, 'p5': {'full_frac': {'last': nan, 'mean': nan, 'late_mean': nan}, 'mid_frac': {'last': nan, 'mean': nan, 'late_mean': nan}, 'far_frac': {'last': nan, 'mean': nan, 'late_mean': nan}, 'drop_frac': {'last': nan, 'mean': nan, 'late_mean': nan}}}
### Weighted loss health
{'p0': {'last': nan, 'mean': nan, 'late_mean': nan}, 'p1': {'last': nan, 'mean': nan, 'late_mean': nan}, 'p2': {'last': nan, 'mean': nan, 'late_mean': nan}, 'p3': {'last': nan, 'mean': nan, 'late_mean': nan}, 'p4': {'last': nan, 'mean': nan, 'late_mean': nan}, 'p5': {'last': nan, 'mean': nan, 'late_mean': nan}}

## B3. v17.1 rollout-buffer health
- checks: {'collection_performance_failed': False, 'action_convention_failed': True, 'action_source_failed': False}
### p0
- awr_collection_mode: unknown
- rollout_builder_entries: 0.0
- rollout_builder_selected_steps: 0.0
- rollout_builder_time_ms: 0.0
- skipped_collect_not_started: 0.0
- action_source_code: 0.0
- action_out_of_bounds_frac_1: 0.0
- ppo_buffer_action_min: nan
- ppo_buffer_action_max: nan
- fps_by_policy: nan
- v13_reference_fps: 95000.0
- collection_performance_ok: True
- action_convention_ok: True
- action_source_ok: True

### p1
- awr_collection_mode: unknown
- rollout_builder_entries: 0.0
- rollout_builder_selected_steps: 0.0
- rollout_builder_time_ms: 0.0
- skipped_collect_not_started: 0.0
- action_source_code: 0.0
- action_out_of_bounds_frac_1: 0.0
- ppo_buffer_action_min: -4.864041805267334
- ppo_buffer_action_max: 5.018723487854004
- fps_by_policy: nan
- v13_reference_fps: 90000.0
- collection_performance_ok: True
- action_convention_ok: False
- action_source_ok: True

### p2
- awr_collection_mode: unknown
- rollout_builder_entries: 0.0
- rollout_builder_selected_steps: 0.0
- rollout_builder_time_ms: 0.0
- skipped_collect_not_started: 0.0
- action_source_code: 0.0
- action_out_of_bounds_frac_1: 0.0
- ppo_buffer_action_min: nan
- ppo_buffer_action_max: nan
- fps_by_policy: nan
- v13_reference_fps: 95000.0
- collection_performance_ok: True
- action_convention_ok: True
- action_source_ok: True

### p3
- awr_collection_mode: unknown
- rollout_builder_entries: 0.0
- rollout_builder_selected_steps: 0.0
- rollout_builder_time_ms: 0.0
- skipped_collect_not_started: 0.0
- action_source_code: 0.0
- action_out_of_bounds_frac_1: 0.0
- ppo_buffer_action_min: -5.013198375701904
- ppo_buffer_action_max: 5.006488800048828
- fps_by_policy: nan
- v13_reference_fps: 90000.0
- collection_performance_ok: True
- action_convention_ok: False
- action_source_ok: True

### p4
- awr_collection_mode: unknown
- rollout_builder_entries: 0.0
- rollout_builder_selected_steps: 0.0
- rollout_builder_time_ms: 0.0
- skipped_collect_not_started: 0.0
- action_source_code: 0.0
- action_out_of_bounds_frac_1: 0.0
- ppo_buffer_action_min: -4.824590682983398
- ppo_buffer_action_max: 4.9264445304870605
- fps_by_policy: nan
- v13_reference_fps: 90000.0
- collection_performance_ok: True
- action_convention_ok: False
- action_source_ok: True

### p5
- awr_collection_mode: unknown
- rollout_builder_entries: 0.0
- rollout_builder_selected_steps: 0.0
- rollout_builder_time_ms: 0.0
- skipped_collect_not_started: 0.0
- action_source_code: 0.0
- action_out_of_bounds_frac_1: 0.0
- ppo_buffer_action_min: -5.372122764587402
- ppo_buffer_action_max: 5.222724437713623
- fps_by_policy: nan
- v13_reference_fps: 90000.0
- collection_performance_ok: True
- action_convention_ok: False
- action_source_ok: True


## C. v13 mechanism health
- profile_copy_events: 2
- lineage_events: 0

## D. Separation block (EIR actor_distill vs AWR)
### p0
- EIR actor_distill: {'actor_distill_active_fraction': {'last': nan, 'mean': nan, 'late_mean': nan}, 'actor_distill_weighted_loss': {'last': nan, 'mean': nan, 'late_mean': nan}, 'actor_distill_reject_large_delta': {'last': nan, 'mean': nan, 'late_mean': nan}}
- AWR: {'weighted_loss': {'last': nan, 'mean': nan, 'late_mean': nan}, 'active_frac': {'last': 0.0, 'mean': 0.0, 'late_mean': 0.0}, 'delta_gate_drop_frac': {'last': nan, 'mean': nan, 'late_mean': nan}, 'quality_mean': {'last': nan, 'mean': nan, 'late_mean': nan}}

### p1
- EIR actor_distill: {'actor_distill_active_fraction': {'last': nan, 'mean': nan, 'late_mean': nan}, 'actor_distill_weighted_loss': {'last': nan, 'mean': nan, 'late_mean': nan}, 'actor_distill_reject_large_delta': {'last': nan, 'mean': nan, 'late_mean': nan}}
- AWR: {'weighted_loss': {'last': nan, 'mean': nan, 'late_mean': nan}, 'active_frac': {'last': 0.0, 'mean': 0.0, 'late_mean': 0.0}, 'delta_gate_drop_frac': {'last': nan, 'mean': nan, 'late_mean': nan}, 'quality_mean': {'last': nan, 'mean': nan, 'late_mean': nan}}

### p2
- EIR actor_distill: {'actor_distill_active_fraction': {'last': nan, 'mean': nan, 'late_mean': nan}, 'actor_distill_weighted_loss': {'last': nan, 'mean': nan, 'late_mean': nan}, 'actor_distill_reject_large_delta': {'last': nan, 'mean': nan, 'late_mean': nan}}
- AWR: {'weighted_loss': {'last': nan, 'mean': nan, 'late_mean': nan}, 'active_frac': {'last': 0.0, 'mean': 0.0, 'late_mean': 0.0}, 'delta_gate_drop_frac': {'last': nan, 'mean': nan, 'late_mean': nan}, 'quality_mean': {'last': nan, 'mean': nan, 'late_mean': nan}}

### p3
- EIR actor_distill: {'actor_distill_active_fraction': {'last': nan, 'mean': nan, 'late_mean': nan}, 'actor_distill_weighted_loss': {'last': nan, 'mean': nan, 'late_mean': nan}, 'actor_distill_reject_large_delta': {'last': nan, 'mean': nan, 'late_mean': nan}}
- AWR: {'weighted_loss': {'last': nan, 'mean': nan, 'late_mean': nan}, 'active_frac': {'last': 0.0, 'mean': 0.0, 'late_mean': 0.0}, 'delta_gate_drop_frac': {'last': nan, 'mean': nan, 'late_mean': nan}, 'quality_mean': {'last': nan, 'mean': nan, 'late_mean': nan}}

### p4
- EIR actor_distill: {'actor_distill_active_fraction': {'last': nan, 'mean': nan, 'late_mean': nan}, 'actor_distill_weighted_loss': {'last': nan, 'mean': nan, 'late_mean': nan}, 'actor_distill_reject_large_delta': {'last': nan, 'mean': nan, 'late_mean': nan}}
- AWR: {'weighted_loss': {'last': nan, 'mean': nan, 'late_mean': nan}, 'active_frac': {'last': 0.0, 'mean': 0.0, 'late_mean': 0.0}, 'delta_gate_drop_frac': {'last': nan, 'mean': nan, 'late_mean': nan}, 'quality_mean': {'last': nan, 'mean': nan, 'late_mean': nan}}

### p5
- EIR actor_distill: {'actor_distill_active_fraction': {'last': nan, 'mean': nan, 'late_mean': nan}, 'actor_distill_weighted_loss': {'last': nan, 'mean': nan, 'late_mean': nan}, 'actor_distill_reject_large_delta': {'last': nan, 'mean': nan, 'late_mean': nan}}
- AWR: {'weighted_loss': {'last': nan, 'mean': nan, 'late_mean': nan}, 'active_frac': {'last': 0.0, 'mean': 0.0, 'late_mean': 0.0}, 'delta_gate_drop_frac': {'last': nan, 'mean': nan, 'late_mean': nan}, 'quality_mean': {'last': nan, 'mean': nan, 'late_mean': nan}}

## Run config notes
- awr_kl_gate_enable: False
- awr_kl_telemetry_only: True
- awr_broadcast_enabled: False
- awr_inherit_enabled: True