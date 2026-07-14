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

## Ablation note (authoritative profiles)

- Code paths for A/B/C remain independently flaggable.
- **Profile24 (`cser_dexpbt_progress_refine_abc`) is authoritative all-on**; `--no-*` on ABC flags is **not** a reliable ablation under restart-safe profile expansion.
- Future A/B/C ablations should add separate derived profiles (not allocated this round).
- Profile22 pins A/B/C enables to False.
