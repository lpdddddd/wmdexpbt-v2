"""Tests for CSER-DexPBT profile and launchers."""

from __future__ import annotations

import argparse
import re
import unittest
from pathlib import Path

from dexpbt_isaaclab.dexpbt_utils import CSER_TB_SCALAR_NAMES
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
CSER_LAUNCHER_1P2B = (
    REPO_ROOT / "results/launchers/start_regrasping_cser_dexpbt_1p2b_smoke.sh"
)
CSER_LAUNCHER_5B = REPO_ROOT / "results/launchers/start_regrasping_cser_dexpbt_5b.sh"
CSER_LAUNCHER_CONSERVATIVE_5B = (
    REPO_ROOT / "results/launchers/start_regrasping_cser_dexpbt_conservative_5b.sh"
)
CSER_DOC = REPO_ROOT / "results/docs/CSER_DEXPBT.md"

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


class TestCserProfileRegistration(unittest.TestCase):
    def test_cser_profile_registered(self):
        self.assertEqual(PROFILE_IDS["cser_dexpbt"], 19)
        self.assertIn("cser_dexpbt", PROFILE_DESCRIPTIONS)
        self.assertEqual(PROFILE_FAMILY["cser_dexpbt"], "unified")
        self.assertIn("cser_dexpbt", PROFILE_TABLES)
        self.assertEqual(PROFILE_COPY_SOURCE_BONUS["cser_dexpbt"], 1.0)

    def test_cser_profile_expansion_core_mechanisms(self):
        args = _expand_profile("cser_dexpbt")
        self.assertTrue(args.elite_snippet_enable)
        self.assertTrue(args.snippet_distill_enable)
        self.assertFalse(args.eir_actor_distill_enable)
        self.assertTrue(args.snippet_wm_rerank_enable)
        self.assertTrue(args.wm_calibration_enable)
        self.assertTrue(args.pbt_consolidation_enable)
        self.assertFalse(args.awr_enable)
        self.assertFalse(args.eir_residual_apply_to_rollout)
        self.assertFalse(args.eir_use_rssm_mppi)
        self.assertEqual(args.eir_mppi_mode, "off")
        self.assertFalse(args.pbt_wm_profile_copy_enable)

    def test_cser_tb_scalar_registry(self):
        self.assertIn("elite_snippet/bank_size", CSER_TB_SCALAR_NAMES)
        self.assertIn("pbt_consolidation/stage", CSER_TB_SCALAR_NAMES)
        self.assertEqual(len(CSER_TB_SCALAR_NAMES), 12)

    def test_cser_profile_blocks_restart_pollution(self):
        argv = [
            "train.py",
            "--wm_strategy_profile=cser_dexpbt",
            "--awr_enable",
            "--eir_residual_apply_to_rollout",
            "--eir_use_rssm_mppi",
            "--pbt_wm_profile_copy_enable",
        ]
        args = _ns(
            wm_strategy_profile="cser_dexpbt",
            awr_enable=True,
            eir_residual_apply_to_rollout=True,
            eir_use_rssm_mppi=True,
            eir_mppi_mode="full",
            pbt_wm_profile_copy_enable=True,
        )
        for dest in _all_profile_dests():
            if not hasattr(args, dest):
                setattr(args, dest, None)
        apply_wm_strategy_profile(args, argv=argv, authoritative=True)
        self.assertFalse(args.awr_enable)
        self.assertFalse(args.eir_residual_apply_to_rollout)
        self.assertFalse(args.eir_use_rssm_mppi)
        self.assertFalse(args.pbt_wm_profile_copy_enable)


class TestCserConservativeProfile(unittest.TestCase):
    def test_conservative_profile_registered(self):
        self.assertEqual(PROFILE_IDS["cser_dexpbt_conservative"], 20)

    def test_pure_cser_keeps_actor_distill_off(self):
        args = _expand_profile("cser_dexpbt")
        self.assertTrue(args.snippet_distill_enable)
        self.assertFalse(args.eir_actor_distill_enable)

    def test_conservative_enables_actor_distill_fallback(self):
        args = _expand_profile("cser_dexpbt_conservative")
        self.assertTrue(args.snippet_distill_enable)
        self.assertTrue(args.eir_actor_distill_enable)
        self.assertEqual(args.eir_actor_distill_allowed_sources, "elite,archive")
        self.assertAlmostEqual(float(args.eir_actor_distill_weight), 0.0005)
        self.assertEqual(args.wm_calibration_eval_tol_mode, "hybrid")
        self.assertAlmostEqual(float(args.wm_calibration_eval_tol_threshold), 0.05)
        self.assertEqual(int(args.wm_calibration_min_samples), 128)

    def test_conservative_hard_safety(self):
        args = _expand_profile("cser_dexpbt_conservative")
        self.assertFalse(args.awr_enable)
        self.assertFalse(args.eir_residual_apply_to_rollout)
        self.assertFalse(args.eir_use_rssm_mppi)
        self.assertEqual(args.eir_mppi_mode, "off")
        self.assertFalse(args.pbt_wm_profile_copy_enable)


class TestCserLauncherSanity(unittest.TestCase):
    def setUp(self):
        self.assertTrue(CSER_LAUNCHER_1P2B.is_file(), msg=str(CSER_LAUNCHER_1P2B))
        self.text = CSER_LAUNCHER_1P2B.read_text()

    def test_all_workers_use_cser_profile(self):
        profiles = _profile_list(self.text)
        self.assertEqual(len(profiles), 6)
        self.assertTrue(all(p == "cser_dexpbt" for p in profiles))

    def test_explicit_disables_present(self):
        self.assertIn("--no-pbt_wm_profile_copy_enable", self.text)
        self.assertIn("--no-awr_enable", self.text)
        self.assertIn("--no-eir_residual_apply_to_rollout", self.text)

    def test_no_conflicting_enable_flags(self):
        self.assertIsNone(_POSITIVE_AWR.search(self.text))
        self.assertIsNone(_POSITIVE_PROFILE_COPY.search(self.text))

    def test_run_config_records_method(self):
        self.assertIn("method=cser_dexpbt", self.text)
        self.assertIn("profile_all_workers=cser_dexpbt", self.text)
        self.assertIn("latest_cser_dexpbt_1p2b_smoke_run.txt", self.text)

    def test_max_iter_default_is_9156(self):
        self.assertIn('MAX_ITER="${MAX_ITER:-9156}"', self.text)

    def test_doc_and_5b_launcher_exist(self):
        self.assertTrue(CSER_DOC.is_file(), msg=str(CSER_DOC))
        self.assertTrue(CSER_LAUNCHER_5B.is_file(), msg=str(CSER_LAUNCHER_5B))
        self.assertIn("pbt_consolidation_enable", CSER_LAUNCHER_5B.read_text())

    def test_conservative_5b_launcher(self):
        self.assertTrue(
            CSER_LAUNCHER_CONSERVATIVE_5B.is_file(),
            msg=str(CSER_LAUNCHER_CONSERVATIVE_5B),
        )
        text = CSER_LAUNCHER_CONSERVATIVE_5B.read_text()
        self.assertIn("cser_dexpbt_conservative", text)
        self.assertIn("--eir_actor_distill_enable", text)
        self.assertIn("--no-eir_residual_apply_to_rollout", text)
        self.assertIn("--no-awr_enable", text)
        self.assertIn("--no-pbt_wm_profile_copy_enable", text)
        self.assertIn("pbt_consolidation_stage1_end_frames=2200000000", text)
