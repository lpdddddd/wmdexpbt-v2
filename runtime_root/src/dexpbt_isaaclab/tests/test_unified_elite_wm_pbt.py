"""Tests for Unified Elite-WM DexPBT profile and launcher."""

from __future__ import annotations

import argparse
import re
import unittest
from pathlib import Path

from dexpbt_isaaclab.world_model.wm_strategy_profiles import (
    PROFILE_COPY_SOURCE_BONUS,
    PROFILE_DESCRIPTIONS,
    PROFILE_FAMILY,
    PROFILE_IDS,
    PROFILE_TABLES,
    apply_wm_strategy_profile,
    profile_capability_flags,
)


REPO_ROOT = Path(__file__).resolve().parents[4]
UNIFIED_LAUNCHER_1P2B = (
    REPO_ROOT
    / "results/launchers/start_regrasping_unified_elite_wm_pbt_1p2b_smoke.sh"
)
UNIFIED_LAUNCHER_2B = (
    REPO_ROOT
    / "results/launchers/start_regrasping_unified_elite_wm_pbt_2b_debug.sh"
)

# Profile-authoritative WM imagined thresholds (must not be duplicated in launchers).
PROFILE_WM_IMAGINED_THRESHOLD_FLAGS = (
    "--wm_imagined_valid_mode soft_calibrated",
    "--wm_imagined_valid_uncertainty_threshold 0.5",
    "--wm_imagined_soft_uncertainty_threshold 0.95",
    "--wm_imagined_min_logit_abs_mean 0.05",
    "--wm_imagined_min_head_signal_std 0.02",
    "--wm_imagined_soft_validity_weight 0.25",
)

_POSITIVE_AWR = re.compile(r"(?<!\-\-no\-)\-\-awr_enable\b")
_POSITIVE_PROFILE_COPY = re.compile(
    r"(?<!\-\-no\-)\-\-pbt_wm_profile_copy_enable\b"
)


def _ns(**kwargs):
    return argparse.Namespace(**kwargs)


def _all_profile_dests() -> set[str]:
    dests: set[str] = set()
    for table in PROFILE_TABLES.values():
        dests.update(table.keys())
    return dests


def _expand_profile(profile: str, *, argv: list[str] | None = None):
    args = _ns(wm_strategy_profile=profile)
    for dest in _all_profile_dests():
        if not hasattr(args, dest):
            setattr(args, dest, None)
    apply_wm_strategy_profile(
        args,
        argv=argv or ["train.py", f"--wm_strategy_profile={profile}"],
        authoritative=True,
    )
    return args


def _profile_list(text: str) -> list[str]:
    match = re.search(r'PROFILE_LIST="([^"]+)"', text)
    assert match is not None
    return match.group(1).split(",")


class TestUnifiedProfileRegistration(unittest.TestCase):
    def test_unified_profile_registered(self):
        self.assertEqual(PROFILE_IDS["unified_elite_wm_pbt"], 18)
        self.assertIn("unified_elite_wm_pbt", PROFILE_DESCRIPTIONS)
        self.assertEqual(PROFILE_FAMILY["unified_elite_wm_pbt"], "unified")
        self.assertIn("unified_elite_wm_pbt", PROFILE_TABLES)
        self.assertEqual(
            PROFILE_COPY_SOURCE_BONUS["unified_elite_wm_pbt"],
            1.0,
        )

    def test_unified_profile_expansion_core_mechanisms(self):
        args = _expand_profile("unified_elite_wm_pbt")
        self.assertTrue(args.enable_wm)
        self.assertTrue(args.eir_enable)
        self.assertTrue(args.eir_use_elite_bank)
        self.assertTrue(args.eir_actor_distill_enable)
        self.assertFalse(args.awr_enable)
        self.assertEqual(args.awr_loss_weight, 0.0)
        self.assertFalse(args.eir_residual_apply_to_rollout)
        self.assertFalse(args.eir_use_rssm_mppi)
        self.assertEqual(args.eir_mppi_mode, "off")
        self.assertEqual(args.pbt_wm_score_mode, "active")
        self.assertTrue(args.pbt_use_wm_recovery_score)
        self.assertTrue(args.pbt_floor_recovery_aware)
        self.assertEqual(args.wm_pbt_source_start_after_frames, 300_000_000)
        self.assertEqual(args.wm_pbt_source_ramp_frames, 300_000_000)
        self.assertFalse(args.wm_curriculum_controller_enable)
        self.assertTrue(args.seq_wm_use_task_aux_inputs)
        self.assertEqual(args.seq_wm_loss_preset, "task_focused")
        self.assertTrue(args.seq_wm_floor_aware_weighting)
        self.assertEqual(args.seq_wm_recovery_horizon, 8)
        self.assertEqual(args.seq_wm_teacher_eval_interval_frames, 50_000_000)
        self.assertEqual(args.seq_wm_teacher_promote_min_improvement, 0.02)
        self.assertEqual(args.seq_wm_teacher_keep_last_k, 6)
        self.assertEqual(args.seq_wm_teacher_eval_buffer_capacity, 8)
        self.assertEqual(args.seq_wm_teacher_record_below_tol, 0.04)
        self.assertEqual(args.seq_wm_teacher_max_uncertainty, 0.5)
        self.assertEqual(args.seq_wm_imagine_num_starts, 256)
        self.assertEqual(args.wm_imagined_soft_uncertainty_threshold, 0.95)
        self.assertEqual(args.wm_imagined_min_logit_abs_mean, 0.05)
        self.assertEqual(args.wm_imagined_min_head_signal_std, 0.02)
        self.assertEqual(args.wm_imagined_soft_validity_weight, 0.25)

    def test_unified_profile_capability_flags(self):
        args = _expand_profile("unified_elite_wm_pbt")
        caps = profile_capability_flags(args)
        self.assertTrue(caps["profile_elite_bank_enabled"])
        self.assertTrue(caps["profile_actor_distill_enabled"])
        self.assertFalse(caps["profile_residual_apply_enabled"])
        self.assertFalse(caps["profile_rssm_mppi_enabled"])
        self.assertFalse(caps["profile_awr_enabled"])
        self.assertTrue(caps["profile_wm_pbt_score_enabled"])

    def test_train_py_accepts_unified_profile(self):
        train_path = (
            Path(__file__).resolve().parents[1]
            / "dexpbt_isaaclab/scripts/train.py"
        )
        text = train_path.read_text()
        self.assertIn('"unified_elite_wm_pbt"', text)

    def test_unified_profile_blocks_restart_pollution(self):
        argv = [
            "train.py",
            "--wm_strategy_profile=unified_elite_wm_pbt",
            "--awr_enable",
            "--eir_residual_apply_to_rollout",
            "--enable_wm",
            "--eir_use_rssm_mppi",
        ]
        args = _ns(
            wm_strategy_profile="unified_elite_wm_pbt",
            awr_enable=True,
            awr_loss_weight=0.001,
            eir_residual_apply_to_rollout=True,
            enable_wm=True,
            eir_use_rssm_mppi=True,
            eir_mppi_mode="full",
        )
        for dest in _all_profile_dests():
            if not hasattr(args, dest):
                setattr(args, dest, None)
        apply_wm_strategy_profile(args, argv=argv, authoritative=True)
        self.assertFalse(args.awr_enable)
        self.assertEqual(args.awr_loss_weight, 0.0)
        self.assertFalse(args.eir_residual_apply_to_rollout)
        self.assertFalse(args.eir_use_rssm_mppi)
        self.assertEqual(args.eir_mppi_mode, "off")


class TestUnifiedLauncherSanity(unittest.TestCase):
    def setUp(self):
        self.assertTrue(UNIFIED_LAUNCHER_1P2B.is_file(), msg=str(UNIFIED_LAUNCHER_1P2B))
        self.text = UNIFIED_LAUNCHER_1P2B.read_text()

    def test_all_workers_use_unified_profile(self):
        profiles = _profile_list(self.text)
        self.assertEqual(len(profiles), 6)
        self.assertTrue(all(p == "unified_elite_wm_pbt" for p in profiles))

    def test_explicit_disables_present(self):
        self.assertIn("--no-pbt_wm_profile_copy_enable", self.text)
        self.assertIn("--no-awr_enable", self.text)
        self.assertIn("--no-eir_residual_apply_to_rollout", self.text)

    def test_no_conflicting_enable_flags(self):
        self.assertIsNone(_POSITIVE_AWR.search(self.text))
        self.assertIsNone(_POSITIVE_PROFILE_COPY.search(self.text))

    def test_run_config_records_method_and_disables(self):
        self.assertIn("target_frames=1200000000", self.text)
        self.assertIn("profile_copy=false", self.text)
        self.assertIn("awr=false", self.text)
        self.assertIn("eir_residual_rollout_apply=false", self.text)
        self.assertIn("method=unified_elite_wm_dexpbt", self.text)
        self.assertIn("profile_all_workers=unified_elite_wm_pbt", self.text)

    def test_max_iter_default_is_9156(self):
        self.assertIn('MAX_ITER="${MAX_ITER:-9156}"', self.text)

    def test_no_conflicting_wm_imagined_threshold_flags(self):
        for flag in PROFILE_WM_IMAGINED_THRESHOLD_FLAGS:
            self.assertNotIn(flag, self.text, msg=f"found duplicated {flag}")


class TestUnified2BLauncherSanity(unittest.TestCase):
    def setUp(self):
        self.assertTrue(UNIFIED_LAUNCHER_2B.is_file(), msg=str(UNIFIED_LAUNCHER_2B))
        self.text = UNIFIED_LAUNCHER_2B.read_text()

    def test_file_exists(self):
        self.assertTrue(UNIFIED_LAUNCHER_2B.is_file())

    def test_all_workers_use_unified_profile(self):
        profiles = _profile_list(self.text)
        self.assertEqual(len(profiles), 6)
        self.assertTrue(all(p == "unified_elite_wm_pbt" for p in profiles))

    def test_explicit_disables_present(self):
        self.assertIn("--no-pbt_wm_profile_copy_enable", self.text)
        self.assertIn("--no-awr_enable", self.text)
        self.assertIn("--no-eir_residual_apply_to_rollout", self.text)

    def test_no_conflicting_enable_flags(self):
        self.assertIsNone(_POSITIVE_AWR.search(self.text))
        self.assertIsNone(_POSITIVE_PROFILE_COPY.search(self.text))

    def test_max_iter_default_is_15259(self):
        self.assertIn('MAX_ITER="${MAX_ITER:-15259}"', self.text)

    def test_run_config_records_2b_debug(self):
        self.assertIn("target_frames=2000000000", self.text)
        self.assertIn("method=unified_elite_wm_dexpbt", self.text)
        self.assertIn("config_version=unified_elite_wm_pbt_2b_debug", self.text)
        self.assertIn("profile_copy=false", self.text)
        self.assertIn("awr=false", self.text)
        self.assertIn("eir_residual_rollout_apply=false", self.text)
        self.assertIn("wm_pbt_score_mode=active", self.text)
        self.assertIn("profile_all_workers=unified_elite_wm_pbt", self.text)

    def test_no_conflicting_wm_imagined_threshold_flags(self):
        for flag in PROFILE_WM_IMAGINED_THRESHOLD_FLAGS:
            self.assertNotIn(flag, self.text, msg=f"found duplicated {flag}")


class TestBackwardCompatibility(unittest.TestCase):
    def test_existing_profile_ids_unchanged(self):
        self.assertEqual(PROFILE_IDS["ppo_only"], 1)
        self.assertEqual(PROFILE_IDS["elite_bank_actor_distill_fast"], 10)
        self.assertEqual(PROFILE_IDS["wm_pbt_source_selector_v2"], 8)
        self.assertEqual(PROFILE_IDS["wm_pbt_source_selector"], 4)
        self.assertEqual(PROFILE_IDS["elite_bank_actor_distill"], 5)
        self.assertEqual(PROFILE_IDS["elite_bank_actor_distill_strong"], 11)
        self.assertEqual(PROFILE_IDS["elite_bank_actor_distill_fast_awr"], 14)
        self.assertEqual(PROFILE_IDS["wm_pbt_source_selector_awr"], 15)
        self.assertEqual(PROFILE_IDS["elite_bank_actor_distill_awr"], 16)
        self.assertEqual(PROFILE_IDS["elite_bank_actor_distill_strong_awr"], 17)

    def test_legacy_profiles_still_expand(self):
        for profile in (
            "ppo_only",
            "elite_bank_actor_distill_fast",
            "wm_pbt_source_selector_v2",
            "wm_pbt_source_selector",
            "elite_bank_actor_distill",
            "elite_bank_actor_distill_strong",
            "elite_bank_actor_distill_fast_awr",
            "wm_pbt_source_selector_awr",
            "elite_bank_actor_distill_awr",
            "elite_bank_actor_distill_strong_awr",
        ):
            with self.subTest(profile=profile):
                args = _expand_profile(profile)
                self.assertEqual(args.wm_strategy_profile, profile)


if __name__ == "__main__":
    unittest.main()
