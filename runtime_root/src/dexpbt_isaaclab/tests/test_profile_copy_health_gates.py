"""Unit tests for the deep-diag profile-copy health gates (2026-06-20).

Covers:
  * `target_min_nsucc_mean` blocks profile swap when the target is broken,
    with `target_min_nsucc_max` as escape hatch for slots that have shown
    bursts of success.
  * `locked_target_slots` per-slot lock prevents profile swap regardless of
    other gates.
  * The `source_min_nsucc_mean` / `source_min_frames` gates (already in
    `decide_profile_copy`) are now correctly carried through PBT restart
    via `profile_copy_restart_train_args` / `resolve_profile_copy_restart_config`.
  * `_parse_profile_copy_config_from_metadata` + round-trip preserves the
    new fields.

These regression tests fail on the pre-fix codebase because:
  * `ProfileCopyConfig` had no `target_min_nsucc_mean` / `locked_target_slots`
  * `profile_copy_restart_train_args` did not emit `source_min_nsucc_mean`
    or `source_min_frames`, so launcher values were silently dropped on
    PBT restart, which is what allowed v17.3 5B to copy a profile at 400M
    from a source with nsucc_mean = 5.2.
"""

from __future__ import annotations

import unittest
from pathlib import Path
import tempfile

from dexpbt_isaaclab.pbt.pbt_profile_copy import (
    ProfileCopyConfig,
    decide_profile_copy,
)
from dexpbt_isaaclab.world_model.wm_strategy_profiles import (
    _parse_profile_copy_config_from_metadata,
    profile_controlled_restart_skip_flags,
    profile_copy_restart_train_args,
    resolve_profile_copy_restart_config,
)


def _population(
    scores: dict[int, float],
    *,
    nsucc_mean: dict[int, float] | None = None,
    nsucc_max: dict[int, float] | None = None,
    frames: dict[int, int] | None = None,
) -> dict[int, dict]:
    pop: dict[int, dict] = {}
    for pid, score in scores.items():
        st: dict = {"pbt_ranking_score": float(score), "objective": float(score)}
        if nsucc_mean and pid in nsucc_mean:
            st["nsucc_mean"] = float(nsucc_mean[pid])
        if nsucc_max and pid in nsucc_max:
            st["nsucc_max"] = float(nsucc_max[pid])
        if frames and pid in frames:
            st["frame"] = int(frames[pid])
        pop[pid] = st
    return pop


class TestTargetHealthGate(unittest.TestCase):
    def _base_cfg(self, **overrides):
        defaults = dict(
            enable=True,
            after_frames=0,
            base_prob=1.0,
            keep_diversity_min_profiles=2,
            min_nsucc_gap=0.0,
            cooldown_frames=0,
            bottom_k=2,
            top_k=2,
        )
        defaults.update(overrides)
        return ProfileCopyConfig(**defaults)

    def _decide(
        self,
        cfg,
        *,
        target_nsucc_mean=0.0,
        target_nsucc_max=0.0,
        target_idx=1,
        source_idx=0,
        source_nsucc_mean=25.0,
        source_nsucc_max=50.0,
    ):
        pop = _population(
            {0: 30.0, 1: 0.0},
            nsucc_mean={0: source_nsucc_mean, 1: target_nsucc_mean},
            nsucc_max={0: source_nsucc_max, 1: target_nsucc_max},
            frames={0: 1_000_000_000, 1: 1_000_000_000},
        )
        return decide_profile_copy(
            cfg=cfg,
            env_frames=1_000_000_000,
            target_policy_idx=target_idx,
            source_policy_idx=source_idx,
            target_profile_name="ppo_only",
            source_profile_name="wm_pbt_source_selector_v2",
            source_generation=0,
            diversity_count=4,
            profiles_set={"ppo_only", "wm_pbt_source_selector_v2", "elite"},
            population=pop,
            source_meta={},
            target_meta={},
            source_checkpoint_type="live",
        )

    def test_target_below_floor_blocks_profile_swap(self):
        """Repro of v17.3 5B p01: target nsucc=0 → no profile swap."""
        cfg = self._base_cfg(target_min_nsucc_mean=5.0)
        result = self._decide(cfg, target_nsucc_mean=0.0, target_nsucc_max=0.0)
        self.assertEqual(result.reject_reason, "target_nsucc_too_low")
        self.assertFalse(result.accepted)
        self.assertFalse(result.eligible)
        self.assertEqual(result.new_profile_name, "ppo_only")  # no swap

    def test_target_below_floor_but_burst_success_allowed(self):
        """nsucc_max escape hatch: slot has shown 30+ succ at peak → ok."""
        cfg = self._base_cfg(
            target_min_nsucc_mean=5.0, target_min_nsucc_max=20.0
        )
        result = self._decide(
            cfg, target_nsucc_mean=2.0, target_nsucc_max=35.0
        )
        # gate passes; final outcome depends on rng; we only need to verify
        # the gate did not reject.
        self.assertNotEqual(result.reject_reason, "target_nsucc_too_low")

    def test_target_above_floor_allows_profile_swap(self):
        cfg = self._base_cfg(target_min_nsucc_mean=5.0)
        result = self._decide(cfg, target_nsucc_mean=12.0, target_nsucc_max=30.0)
        self.assertNotEqual(result.reject_reason, "target_nsucc_too_low")

    def test_target_floor_zero_does_not_gate(self):
        cfg = self._base_cfg(target_min_nsucc_mean=0.0)
        result = self._decide(cfg, target_nsucc_mean=0.0, target_nsucc_max=0.0)
        self.assertNotEqual(result.reject_reason, "target_nsucc_too_low")


class TestLockedTargetSlots(unittest.TestCase):
    def test_locked_slot_blocks_profile_swap(self):
        cfg = ProfileCopyConfig(
            enable=True,
            after_frames=0,
            base_prob=1.0,
            keep_diversity_min_profiles=2,
            bottom_k=2,
            top_k=2,
            locked_target_slots=frozenset({0, 1}),
        )
        pop = _population(
            {0: 30.0, 1: 0.0},
            nsucc_mean={0: 25.0, 1: 20.0},
            nsucc_max={0: 50.0, 1: 50.0},
            frames={0: 1_000_000_000, 1: 1_000_000_000},
        )
        result = decide_profile_copy(
            cfg=cfg,
            env_frames=1_000_000_000,
            target_policy_idx=1,
            source_policy_idx=0,
            target_profile_name="elite_bank_actor_distill_fast_awr",
            source_profile_name="wm_pbt_source_selector_v2",
            source_generation=0,
            diversity_count=4,
            profiles_set={
                "elite_bank_actor_distill_fast_awr",
                "wm_pbt_source_selector_v2",
                "ppo_only",
            },
            population=pop,
            source_meta={},
            target_meta={},
            source_checkpoint_type="live",
        )
        self.assertEqual(result.reject_reason, "target_slot_locked")
        self.assertTrue(result.target_locked)
        # Profile preserved.
        self.assertEqual(
            result.new_profile_name, "elite_bank_actor_distill_fast_awr"
        )

    def test_unlocked_slot_unaffected(self):
        cfg = ProfileCopyConfig(
            enable=True,
            after_frames=0,
            base_prob=1.0,
            keep_diversity_min_profiles=2,
            bottom_k=4,
            top_k=4,
            locked_target_slots=frozenset({0}),
        )
        pop = _population(
            {0: 30.0, 1: 28.0, 2: 5.0},
            nsucc_mean={0: 25.0, 1: 22.0, 2: 8.0},
            nsucc_max={0: 50.0, 1: 50.0, 2: 30.0},
            frames={0: 1_000_000_000, 1: 1_000_000_000, 2: 1_000_000_000},
        )
        result = decide_profile_copy(
            cfg=cfg,
            env_frames=1_000_000_000,
            target_policy_idx=2,
            source_policy_idx=0,
            target_profile_name="ppo_only",
            source_profile_name="wm_pbt_source_selector_v2",
            source_generation=0,
            diversity_count=4,
            profiles_set={"ppo_only", "wm_pbt_source_selector_v2", "elite"},
            population=pop,
            source_meta={},
            target_meta={},
            source_checkpoint_type="live",
        )
        self.assertNotEqual(result.reject_reason, "target_slot_locked")


class TestRestartArgsRoundTrip(unittest.TestCase):
    """The pre-fix bug: source_min_nsucc_mean / source_min_frames silently
    dropped from PBT restart argv, so v17.3 5B launcher had no health gate
    even though the code in `decide_profile_copy` already supported it."""

    def test_restart_args_emit_new_health_flags(self):
        out = profile_copy_restart_train_args(
            enable=True,
            after_frames=300_000_000,
            prob=0.7,
            keep_diversity_min_profiles=3,
            min_nsucc_gap=8.0,
            min_source_retention=0.5,
            cooldown_frames=300_000_000,
            bottom_k=2,
            top_k=2,
            source_min_nsucc_mean=15.0,
            source_min_frames=200_000_000,
            target_min_nsucc_mean=5.0,
            target_min_nsucc_max=20.0,
            locked_target_slots=(0, 1),
        )
        self.assertIn(
            "--pbt_wm_profile_copy_source_min_nsucc_mean=15.0", out
        )
        self.assertIn(
            "--pbt_wm_profile_copy_source_min_frames=200000000", out
        )
        self.assertIn(
            "--pbt_wm_profile_copy_target_min_nsucc_mean=5.0", out
        )
        self.assertIn(
            "--pbt_wm_profile_copy_target_min_nsucc_max=20.0", out
        )
        self.assertIn(
            "--pbt_wm_profile_copy_locked_target_slots=0,1", out
        )

    def test_metadata_round_trip_preserves_new_flags(self):
        with tempfile.TemporaryDirectory() as td:
            metadata = Path(td) / "metadata.txt"
            args = profile_copy_restart_train_args(
                enable=True,
                after_frames=300_000_000,
                prob=0.7,
                keep_diversity_min_profiles=3,
                min_nsucc_gap=8.0,
                min_source_retention=0.5,
                cooldown_frames=300_000_000,
                bottom_k=2,
                top_k=2,
                source_min_nsucc_mean=15.0,
                source_min_frames=200_000_000,
                target_min_nsucc_mean=5.0,
                target_min_nsucc_max=20.0,
                locked_target_slots=(0, 1),
            )
            metadata.write_text(
                "\n".join(f"train_args={a}" for a in args)
            )
            parsed = _parse_profile_copy_config_from_metadata(metadata)
            assert parsed is not None
            self.assertEqual(parsed["source_min_nsucc_mean"], 15.0)
            self.assertEqual(parsed["source_min_frames"], 200_000_000)
            self.assertEqual(parsed["target_min_nsucc_mean"], 5.0)
            self.assertEqual(parsed["target_min_nsucc_max"], 20.0)
            self.assertEqual(parsed["locked_target_slots"], (0, 1))

    def test_resolve_falls_back_to_metadata_when_defaults(self):
        with tempfile.TemporaryDirectory() as td:
            metadata = Path(td) / "metadata.txt"
            args = profile_copy_restart_train_args(
                enable=True,
                after_frames=300_000_000,
                prob=0.7,
                keep_diversity_min_profiles=3,
                min_nsucc_gap=8.0,
                min_source_retention=0.5,
                cooldown_frames=300_000_000,
                bottom_k=2,
                top_k=2,
                source_min_nsucc_mean=15.0,
                source_min_frames=200_000_000,
                target_min_nsucc_mean=5.0,
                target_min_nsucc_max=20.0,
                locked_target_slots=(0, 1),
            )
            metadata.write_text(
                "\n".join(f"train_args={a}" for a in args)
            )
            # call resolve with all defaults so it should fall back to file
            resolved = resolve_profile_copy_restart_config(
                enable=False,
                after_frames=1_000_000_000,
                prob=0.3,
                keep_diversity_min_profiles=3,
                metadata_path=metadata,
            )
        # 14-tuple shape
        self.assertEqual(len(resolved), 14)
        (
            enable, after, prob, kdiv, mngap, msret, ccd, bk, tk,
            smin_n, smin_f, tmin_n, tmin_x, locks,
        ) = resolved
        self.assertTrue(enable)
        self.assertEqual(smin_n, 15.0)
        self.assertEqual(smin_f, 200_000_000)
        self.assertEqual(tmin_n, 5.0)
        self.assertEqual(tmin_x, 20.0)
        self.assertEqual(locks, (0, 1))

    def test_skip_flags_cover_new_health_gates(self):
        skip = profile_controlled_restart_skip_flags()
        for flag in (
            "--pbt_wm_profile_copy_source_min_nsucc_mean",
            "--pbt_wm_profile_copy_source_min_frames",
            "--pbt_wm_profile_copy_target_min_nsucc_mean",
            "--pbt_wm_profile_copy_target_min_nsucc_max",
            "--pbt_wm_profile_copy_locked_target_slots",
        ):
            self.assertIn(flag, skip)


class TestSourceHealthGateEnforced(unittest.TestCase):
    """Sanity: the existing source_min_nsucc_mean gate still rejects."""

    def test_source_below_floor_rejected(self):
        cfg = ProfileCopyConfig(
            enable=True,
            after_frames=0,
            base_prob=1.0,
            keep_diversity_min_profiles=2,
            bottom_k=2,
            top_k=2,
            source_min_nsucc_mean=15.0,
        )
        pop = _population(
            {0: 5.0, 1: 0.0},
            nsucc_mean={0: 5.2, 1: 0.0},
            nsucc_max={0: 45.0, 1: 0.0},
            frames={0: 400_000_000, 1: 400_000_000},
        )
        result = decide_profile_copy(
            cfg=cfg,
            env_frames=400_000_000,
            target_policy_idx=1,
            source_policy_idx=0,
            target_profile_name="elite_bank_actor_distill_fast_awr",
            source_profile_name="wm_pbt_source_selector_v2",
            source_generation=0,
            diversity_count=4,
            profiles_set={"elite_bank_actor_distill_fast_awr",
                          "wm_pbt_source_selector_v2"},
            population=pop,
            source_meta={},
            target_meta={},
            source_checkpoint_type="live",
            source_frames=400_000_000,
        )
        self.assertEqual(result.reject_reason, "source_nsucc_too_low")


if __name__ == "__main__":
    unittest.main()
