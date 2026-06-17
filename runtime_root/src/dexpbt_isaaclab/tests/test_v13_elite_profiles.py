"""Unit tests for Strategy Zoo v1.3 elite-centered profiles and launcher."""

from __future__ import annotations

import argparse
import unittest
from pathlib import Path

from dexpbt_isaaclab.world_model.wm_strategy_profiles import (
    PROFILE_COPY_SOURCE_BONUS,
    PROFILE_IDS,
    PROFILE_TABLES,
    apply_wm_strategy_profile,
    profile_capability_flags,
    profile_copy_source_bonus,
)

REPO_ROOT = Path(__file__).resolve().parents[4]
V13_LAUNCHER = (
    REPO_ROOT
    / "results/launchers/start_regrasping_pbt_wm_strategy_zoo_v13_elite_2b_debug.sh"
)

ZOO_PROFILES_V13 = [
    "ppo_only",
    "elite_bank_actor_distill_fast",
    "wm_pbt_source_selector_v2",
    "wm_pbt_source_selector",
    "elite_bank_actor_distill",
    "elite_bank_actor_distill_strong",
]

V13_ELITE_PROFILES = (
    "elite_bank_actor_distill_fast",
    "elite_bank_actor_distill",
    "elite_bank_actor_distill_strong",
)


def _expand(profile: str):
    args = argparse.Namespace(wm_strategy_profile=profile)
    for dest in {k for t in PROFILE_TABLES.values() for k in t}:
        if not hasattr(args, dest):
            setattr(args, dest, None)
    apply_wm_strategy_profile(
        args,
        argv=["train.py", f"--wm_strategy_profile={profile}"],
        authoritative=(profile != "legacy"),
    )
    return args, profile_capability_flags(args)


class TestV13EliteProfiles(unittest.TestCase):
    def test_elite_fast_registered_and_residual_off(self):
        self.assertIn("elite_bank_actor_distill_fast", PROFILE_TABLES)
        self.assertEqual(PROFILE_IDS["elite_bank_actor_distill_fast"], 10)
        args, caps = _expand("elite_bank_actor_distill_fast")
        self.assertFalse(args.eir_residual_apply_to_rollout)
        self.assertFalse(args.eir_use_rssm_mppi)
        self.assertEqual(args.eir_mppi_mode, "off")
        self.assertTrue(caps["profile_actor_distill_enabled"])
        self.assertTrue(caps["profile_elite_bank_enabled"])
        self.assertFalse(caps["profile_rssm_mppi_enabled"])
        self.assertFalse(caps["profile_residual_apply_enabled"])

    def test_elite_strong_registered_and_residual_off(self):
        self.assertIn("elite_bank_actor_distill_strong", PROFILE_TABLES)
        self.assertEqual(PROFILE_IDS["elite_bank_actor_distill_strong"], 11)
        args, caps = _expand("elite_bank_actor_distill_strong")
        self.assertFalse(args.eir_residual_apply_to_rollout)
        self.assertFalse(args.eir_use_rssm_mppi)
        self.assertEqual(args.eir_mppi_mode, "off")
        self.assertTrue(caps["profile_actor_distill_enabled"])
        self.assertTrue(caps["profile_elite_bank_enabled"])
        self.assertFalse(caps["profile_rssm_mppi_enabled"])
        self.assertFalse(caps["profile_residual_apply_enabled"])

    def test_v13_launcher_profile_list_exact(self):
        self.assertTrue(V13_LAUNCHER.is_file(), f"missing {V13_LAUNCHER}")
        text = V13_LAUNCHER.read_text()
        line = next(l for l in text.splitlines() if l.startswith("PROFILE_LIST="))
        profiles = line.split("=", 1)[1].strip().strip('"').split(",")
        self.assertEqual(profiles, ZOO_PROFILES_V13)
        self.assertEqual(len(set(profiles)), 6)

    def test_v13_launcher_excludes_light_rssm_profiles(self):
        text = V13_LAUNCHER.read_text()
        self.assertNotIn("light_rssm_mppi_actor_distill", text)
        self.assertNotIn("light_conservative_rssm_planner_distill", text)

    def test_v13_launcher_uses_retention_050(self):
        text = V13_LAUNCHER.read_text()
        self.assertIn("pbt_wm_profile_copy_min_source_retention=0.50", text)
        self.assertIn("--pbt_wm_profile_copy_min_source_retention 0.50", text)

    def test_profile_copy_source_bonus_favors_elite_over_rssm(self):
        elite = profile_copy_source_bonus("elite_bank_actor_distill")
        elite_fast = profile_copy_source_bonus("elite_bank_actor_distill_fast")
        elite_strong = profile_copy_source_bonus("elite_bank_actor_distill_strong")
        rssm = profile_copy_source_bonus("light_rssm_mppi_actor_distill")
        planner = profile_copy_source_bonus("light_conservative_rssm_planner_distill")
        self.assertGreater(elite, rssm)
        self.assertGreater(elite_fast, rssm)
        self.assertGreater(elite_strong, rssm)
        self.assertGreater(elite, planner)
        self.assertEqual(PROFILE_COPY_SOURCE_BONUS["light_rssm_mppi_actor_distill"], 0.5)

    def test_residual_rollout_apply_false_for_all_v13_profiles(self):
        for profile in ZOO_PROFILES_V13:
            args, caps = _expand(profile)
            self.assertFalse(
                args.eir_residual_apply_to_rollout,
                msg=f"{profile} must keep residual rollout apply OFF",
            )
            self.assertFalse(
                caps["profile_residual_apply_enabled"],
                msg=f"{profile} capability flag must keep residual apply OFF",
            )


if __name__ == "__main__":
    unittest.main()
