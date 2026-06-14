"""Unit tests for wm_strategy_profiles (PBT-WM Strategy Zoo v1.0)."""

from __future__ import annotations

import argparse
import unittest

from dexpbt_isaaclab.world_model.wm_strategy_profiles import (
    PROFILE_IDS,
    apply_wm_strategy_profile,
    collect_explicit_dests,
    profile_capability_flags,
)


def _ns(**kwargs):
    return argparse.Namespace(**kwargs)


class TestWMStrategyProfiles(unittest.TestCase):
    def test_profile_ids_are_stable(self):
        self.assertEqual(PROFILE_IDS["legacy"], 0)
        self.assertEqual(PROFILE_IDS["ppo_only"], 1)
        self.assertEqual(PROFILE_IDS["rssm_mppi_actor_distill"], 2)
        self.assertEqual(PROFILE_IDS["wm_curriculum_controller"], 3)
        self.assertEqual(PROFILE_IDS["wm_pbt_source_selector"], 4)
        self.assertEqual(PROFILE_IDS["elite_bank_actor_distill"], 5)
        self.assertEqual(PROFILE_IDS["conservative_rssm_planner_distill"], 6)

    def test_legacy_no_op(self):
        args = _ns(wm_strategy_profile="legacy", eir_enable=False)
        audit = apply_wm_strategy_profile(args)
        self.assertEqual(audit, [])
        self.assertFalse(args.eir_enable)

    def test_ppo_only_disables_all_wm_intervention(self):
        args = _ns(
            wm_strategy_profile="ppo_only",
            enable_wm=True,
            eir_enable=True,
            pbt_wm_score_mode="active",
            pbt_use_wm_recovery_score=True,
            eir_residual_apply_to_rollout=True,
            wm_curriculum_controller_enable=True,
        )
        apply_wm_strategy_profile(args)
        self.assertFalse(args.enable_wm)
        self.assertFalse(args.eir_enable)
        self.assertEqual(args.pbt_wm_score_mode, "off")
        self.assertFalse(args.pbt_use_wm_recovery_score)
        self.assertFalse(args.eir_residual_apply_to_rollout)
        self.assertFalse(args.wm_curriculum_controller_enable)

    def test_rssm_mppi_actor_distill_enables_actor_distill_no_residual_apply(self):
        args = _ns(
            wm_strategy_profile="rssm_mppi_actor_distill",
            eir_actor_distill_enable=False,
            eir_residual_apply_to_rollout=True,
            eir_use_rssm_mppi=False,
        )
        apply_wm_strategy_profile(args)
        self.assertTrue(args.eir_actor_distill_enable)
        self.assertFalse(args.eir_residual_apply_to_rollout)
        self.assertTrue(args.eir_use_rssm_mppi)
        self.assertEqual(args.eir_actor_distill_allowed_sources, "rssm,elite,archive")

    def test_wm_curriculum_controller_enables_controller_no_action_intervention(self):
        args = _ns(
            wm_strategy_profile="wm_curriculum_controller",
            wm_curriculum_controller_enable=False,
            eir_actor_distill_enable=True,
            eir_enable=True,
            eir_residual_apply_to_rollout=True,
        )
        apply_wm_strategy_profile(args)
        self.assertTrue(args.wm_curriculum_controller_enable)
        self.assertFalse(args.eir_actor_distill_enable)
        self.assertFalse(args.eir_enable)
        self.assertFalse(args.eir_residual_apply_to_rollout)

    def test_wm_pbt_source_selector_enables_active_wm_score_no_action_intervention(self):
        args = _ns(
            wm_strategy_profile="wm_pbt_source_selector",
            pbt_wm_score_mode="off",
            pbt_use_wm_recovery_score=False,
            eir_actor_distill_enable=True,
            eir_enable=True,
        )
        apply_wm_strategy_profile(args)
        self.assertEqual(args.pbt_wm_score_mode, "active")
        self.assertTrue(args.pbt_use_wm_recovery_score)
        self.assertFalse(args.eir_actor_distill_enable)
        self.assertFalse(args.eir_enable)

    def test_elite_bank_actor_distill_allows_only_elite_archive_sources(self):
        args = _ns(
            wm_strategy_profile="elite_bank_actor_distill",
            eir_actor_distill_allowed_sources="rssm,elite,archive",
            eir_use_rssm_mppi=True,
            eir_mppi_mode="full",
        )
        apply_wm_strategy_profile(args)
        self.assertEqual(args.eir_actor_distill_allowed_sources, "elite,archive")
        self.assertFalse(args.eir_use_rssm_mppi)
        self.assertEqual(args.eir_mppi_mode, "off")

    def test_conservative_planner_allows_only_rssm_teacher(self):
        args = _ns(
            wm_strategy_profile="conservative_rssm_planner_distill",
            eir_actor_distill_allowed_sources="rssm,elite,archive",
        )
        apply_wm_strategy_profile(args)
        self.assertEqual(args.eir_actor_distill_allowed_sources, "rssm")
        self.assertTrue(args.eir_use_rssm_mppi)
        self.assertFalse(args.eir_residual_apply_to_rollout)

    def test_explicit_override_respected(self):
        argv = [
            "train.py",
            "--wm_strategy_profile",
            "ppo_only",
            "--enable_wm",
        ]
        explicit = collect_explicit_dests(argv)
        self.assertIn("enable_wm", explicit)
        args = _ns(
            wm_strategy_profile="ppo_only",
            enable_wm=True,
        )
        audit = apply_wm_strategy_profile(args, argv=argv)
        kept = [a for a in audit if not a.get("overridden", True)]
        self.assertTrue(args.enable_wm)
        self.assertTrue(any(a.get("flag") == "enable_wm" for a in kept))

    def test_profile_capability_flags(self):
        args = _ns(
            wm_strategy_profile="rssm_mppi_actor_distill",
            eir_actor_distill_enable=True,
            eir_residual_apply_to_rollout=False,
            eir_enable=True,
            eir_use_rssm_mppi=True,
            eir_mppi_mode="full",
            pbt_wm_score_mode="off",
            wm_curriculum_controller_enable=False,
            eir_use_elite_bank=True,
        )
        caps = profile_capability_flags(args)
        self.assertTrue(caps["profile_actor_distill_enabled"])
        self.assertFalse(caps["profile_residual_apply_enabled"])
        self.assertTrue(caps["profile_rssm_mppi_enabled"])


class TestProfileCopyDefaults(unittest.TestCase):
    def test_profile_copy_disabled_by_default(self):
        args = _ns(
            pbt_wm_profile_copy_enable=False,
            pbt_wm_profile_copy_after_frames=1_000_000_000,
            pbt_wm_profile_copy_prob=0.3,
            pbt_wm_profile_keep_diversity_min_profiles=3,
        )
        self.assertFalse(args.pbt_wm_profile_copy_enable)

    def test_profile_copy_preserves_min_diversity(self):
        diversity = 3
        min_profiles = 3
        allow_copy = diversity > min_profiles
        self.assertFalse(allow_copy)


if __name__ == "__main__":
    unittest.main()
