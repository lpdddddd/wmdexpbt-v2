# Profile22 vs ABC resolved config diff
- Profile22 id: 22
- ABC id: 24

## Differing keys only

| key | Profile22 | ABC |
|---|---|---|
| `eir_enable_belief_for_snippet_wm_rerank` | `False` | `True` |
| `eir_preserve_local_bank_fraction` | `None` | `0.2` |
| `eir_tolerance_aware_bank_broadcast` | `False` | `True` |
| `pbt_chronic_cooldown_frames` | `None` | `400000000` |
| `pbt_chronic_max_copies_per_lineage` | `None` | `1` |
| `pbt_chronic_min_bottom_frames` | `None` | `200000000` |
| `pbt_chronic_min_bottom_ticks` | `None` | `5` |
| `pbt_chronic_min_score_gap` | `None` | `0.15` |
| `pbt_enable_chronic_bottom_exploit` | `False` | `True` |

## Notes

- All other Profile22 keys are identical (PPO/curriculum/WM/snippet/adaptive rescue/hard-offs).
- Profile22 now **pins** A/B/C enable flags to `False` for restart purity; runtime behavior with default CLI was already False.
- Mid mutation / MPPI / AWR / residual rollout remain off on both.
