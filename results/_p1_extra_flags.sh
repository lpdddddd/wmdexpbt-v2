# Shared P1 --extra flags (source from launch scripts).
# shellcheck shell=bash
P1_EXTRA_FLAGS=(
  --curriculum_mode mastery
  --pbt_source_require_stable_mean
  --mastery_nsucc_mean 4.0
  --mastery_nsucc_max 20.0
  --mastery_consecutive_windows 2
  --curriculum_shrink_factor 0.95
  --curriculum_collapse_detect
  --collapse_drop_ratio 0.5
  --collapse_patience 2
  --collapse_freeze_env_frames 200000000
  --stable_source_min_mean 3.0
  --stable_source_min_max 20.0
  --stable_source_max_drop_ratio 0.3
  --stable_source_drop_windows 2
  --no-pbt_enable_best_nsucc_recovery
  --no-pbt_allow_best_nsucc_archive_recovery
  --cf_source_min_nsucc_mean 3.0
  --cf_source_min_nsucc_max 20
)
