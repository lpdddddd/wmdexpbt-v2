"""Unit tests for Strategy Zoo v1.5 elite-heavy distill-clip validation."""

from __future__ import annotations

import argparse
import unittest
from pathlib import Path

from dexpbt_isaaclab.world_model.wm_strategy_profiles import (
    PROFILE_TABLES,
    apply_wm_strategy_profile,
    profile_capability_flags,
)

REPO_ROOT = Path(__file__).resolve().parents[4]
V15_LAUNCHER = (
    REPO_ROOT
    / "results/launchers/start_regrasping_pbt_wm_strategy_zoo_v15_elite_clip_validation_1b.sh"
)
V15_ANALYZER = REPO_ROOT / "results/analyze_strategy_zoo_v15_clip_validation.py"

ZOO_PROFILES_V15 = [
    "elite_bank_actor_distill",
    "elite_bank_actor_distill_fast_v2",
    "elite_bank_actor_distill",
    "wm_pbt_source_selector",
    "elite_bank_actor_distill",
    "elite_bank_actor_distill_strong_v2",
]

FORBIDDEN_PROFILES = (
    "ppo_only",
    "wm_pbt_source_selector_v2",
    "light_rssm_mppi_actor_distill",
    "light_conservative_rssm_planner_distill",
    "rssm_mppi_actor_distill",
    "conservative_rssm_planner_distill",
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


class TestV15ClipProfiles(unittest.TestCase):
    def test_clip_profiles_widen_guidance_to_006(self):
        for prof in ("elite_bank_actor_distill_fast_v2", "elite_bank_actor_distill_strong_v2"):
            args, caps = _expand(prof)
            self.assertEqual(args.eir_actor_distill_large_delta_mode, "clip")
            self.assertFalse(args.eir_residual_apply_to_rollout)
            self.assertFalse(args.eir_use_rssm_mppi)
            self.assertFalse(caps["profile_rssm_mppi_enabled"])
            self.assertAlmostEqual(float(args.eir_guidance_bank_tol), 0.06)
            self.assertAlmostEqual(float(args.eir_guidance_start_tol), 0.06)
            self.assertAlmostEqual(float(args.eir_guidance_apply_tol), 0.06)
            # guidance band must be enabled for the mid-stage bank collection.
            self.assertTrue(bool(args.eir_guidance_enable))

    def test_original_elite_keeps_guidance_004(self):
        args, _ = _expand("elite_bank_actor_distill")
        self.assertAlmostEqual(float(args.eir_guidance_bank_tol), 0.04)
        self.assertAlmostEqual(float(args.eir_guidance_start_tol), 0.04)
        self.assertAlmostEqual(float(args.eir_guidance_apply_tol), 0.04)
        self.assertFalse(args.eir_residual_apply_to_rollout)
        self.assertFalse(args.eir_use_rssm_mppi)


class TestV15Launcher(unittest.TestCase):
    def setUp(self):
        self.assertTrue(V15_LAUNCHER.is_file())
        self.text = V15_LAUNCHER.read_text()

    def test_profile_list_exact(self):
        line = next(l for l in self.text.splitlines() if l.startswith("PROFILE_LIST="))
        profiles = line.split("=", 1)[1].strip().strip('"').split(",")
        self.assertEqual(profiles, ZOO_PROFILES_V15)

    def test_no_forbidden_profiles_in_profile_list(self):
        line = next(l for l in self.text.splitlines() if l.startswith("PROFILE_LIST="))
        profiles = set(line.split("=", 1)[1].strip().strip('"').split(","))
        for prof in FORBIDDEN_PROFILES:
            self.assertNotIn(prof, profiles)

    def test_copy_settings_v15(self):
        self.assertIn("--pbt_wm_profile_copy_after_frames 300000000", self.text)
        self.assertIn("--pbt_wm_profile_copy_prob 0.7", self.text)
        self.assertIn("--pbt_wm_profile_copy_source_min_nsucc_mean 0.0", self.text)
        self.assertIn("--pbt_wm_profile_copy_source_min_frames 0", self.text)
        self.assertIn("--no-pbt_wm_profile_copy_top1_override_enable", self.text)
        # The v1.4 conservative source gates must be gone.
        self.assertNotIn("--pbt_wm_profile_copy_source_min_frames 600000000", self.text)
        self.assertNotIn("--pbt_wm_profile_copy_source_min_nsucc_mean 20.0", self.text)

    def test_residual_rollout_apply_off(self):
        self.assertIn("--no-eir_residual_apply_to_rollout", self.text)
        self.assertNotIn("--eir_residual_apply_to_rollout ", self.text)

    def test_target_is_1b(self):
        self.assertIn("target_frames=1000000000", self.text)
        self.assertIn('MAX_ITER="${MAX_ITER:-7630}"', self.text)

    def test_run_config_records_clip_validation(self):
        self.assertIn("clip_validation=true", self.text)
        self.assertIn("clip_validation_bank_tol_p1_p5=0.06", self.text)
        self.assertIn("elite_heavy=true", self.text)
        self.assertIn("ppo_only_in_population=false", self.text)
        self.assertIn("selector_v2_in_population=false", self.text)
        self.assertIn("config_version=strategy_zoo_v1.5_elite_heavy_clip_validation", self.text)


class TestV15Analyzer(unittest.TestCase):
    def test_analyzer_exists_and_outputs_to_docs(self):
        self.assertTrue(V15_ANALYZER.is_file())
        text = V15_ANALYZER.read_text()
        self.assertIn("STRATEGY_ZOO_V15_CLIP_VALIDATION_REPORT", text)
        self.assertIn("RESULTS_DOCS", text)
        self.assertNotIn("wmdexpbt/docs/", text)
        # must focus on pipeline activation, not only nsucc
        self.assertIn("bank_elite_bank_size", text)
        self.assertIn("actor_distill_clip_large_delta", text)
        self.assertIn("hard_success", text)


if __name__ == "__main__":
    unittest.main()
