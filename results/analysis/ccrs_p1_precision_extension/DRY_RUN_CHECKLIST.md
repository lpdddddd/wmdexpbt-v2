# p1 fixed-precision extension — dry-run checklist

- checkpoint: `/root/autodl-tmp/wmdexpbt/results/runs/regrasping_cser_dexpbt_conservative_retention_strict_5b_20260716_130631/dexpbt_8gpu_ws/001/000160.pth`
  (p1 it160 frame 5000000000, roles: tip)
- units: 16 = 4 tolerances x 4 seeds
- new tolerances (m): [0.016, 0.014, 0.012, 0.01] (0.018 reused read-only from ccrs_base_retention, NOT re-run)
- seeds: [10001, 10002, 10003, 10004]
- protocol: policy_only, deterministic, 512 envs, 512 first episodes, frozen curriculum, paired reset fingerprints, max_consecutive_successes=50, terminal snapshot hook, v2_first_episode_per_env / v2_normalized_nsucc
- results dir: `/root/autodl-tmp/wmdexpbt/results/analysis/ccrs_p1_precision_extension` (original ccrs_base_retention outputs untouched)
- planned GPUs: 2; estimated wall time ≈ 8 units/GPU x 150s ≈ 20 min

## Preflight checks

| check | status | detail |
|---|---|---|
| checkpoint_exists | PASS | /root/autodl-tmp/wmdexpbt/results/runs/regrasping_cser_dexpbt_conservative_retention_strict_5b_20260716_130631/dexpbt_8gpu_ws/001/000160.pth |
| checkpoint_size_matches_manifest | PASS | 180443740 vs manifest 180443740 |
| unit_count_is_16 | PASS | 16 units |
| tolerances_are_new_grid | PASS | [0.016, 0.014, 0.012, 0.01] |
| no_overlap_with_existing_grid_incl_0018 | PASS | overlap=[] |
| tolerances_in_meters | PASS | meter-range sanity (0.001..0.2) |
| seeds_match_original_ccrs | PASS | [10001, 10002, 10003, 10004] |
| all_policy_only | PASS |  |
| all_deterministic | PASS |  |
| envs_512_episodes_512 | PASS |  |
| max_consecutive_successes_50 | PASS |  |
| reset_protocol_matches_original | PASS | v2_first_episode_per_env / v2_normalized_nsucc |
| single_checkpoint_p1_tip | PASS |  |
| unit_keys_unique | PASS |  |
| output_dir_separate_from_base_ccrs | PASS | /root/autodl-tmp/wmdexpbt/results/analysis/ccrs_p1_precision_extension vs /root/autodl-tmp/wmdexpbt/results/analysis/ccrs_base_retention |
| eval_script_exists | PASS | /root/autodl-tmp/wmdexpbt/runtime_root/src/dexpbt_isaaclab/dexpbt_isaaclab/scripts/ccrs_eval.py |
| eval_src_has[last_curriculum_update = 10 ** 9] | PASS |  |
| eval_src_has[assert not hasattr(agent, "optimizer")] | PASS |  |
| eval_src_has[FirstEpisodeCollector] | PASS |  |
| eval_src_has[extract_reset_fingerprints] | PASS |  |
| eval_src_has[TerminalSnapshotHook] | PASS |  |
| eval_src_lacks[PbtObserver] | PASS |  |
| eval_src_lacks[optimizer.step] | PASS |  |
| eval_src_lacks[loss.backward] | PASS |  |

**Overall: ALL CHECKS PASSED** (GPU launch requires --execute and all checks passing).
