"""Preflight tests for PBT-WM Strategy Zoo v1.0 launcher + WM curriculum controller."""

from __future__ import annotations

import argparse
import json
import tempfile
import unittest
from pathlib import Path

from dexpbt_isaaclab.pbt.pbt_curriculum_p1 import RollingNsuccWindow, WindowSample
from dexpbt_isaaclab.pbt.pbt_observer import DexPBTPbtAlgoObserver
from dexpbt_isaaclab.scripts.launch_pbt import (
    LAUNCH_PBT_BOOLEAN_FLAGS,
    normalize_extra_args,
)
from dexpbt_isaaclab.world_model.wm_strategy_profiles import (
    PROFILE_TABLES,
    apply_wm_strategy_profile,
    profile_capability_flags,
)


REPO_ROOT = Path(__file__).resolve().parents[4]
LAUNCHER = (
    REPO_ROOT
    / "results/launchers/start_regrasping_pbt_wm_strategy_zoo_full_3b.sh"
)

ZOO_PROFILES = [
    "ppo_only",
    "rssm_mppi_actor_distill",
    "wm_curriculum_controller",
    "wm_pbt_source_selector",
    "elite_bank_actor_distill",
    "conservative_rssm_planner_distill",
]


def _ns(**kwargs):
    return argparse.Namespace(**kwargs)


def _simulate_profile_expansion(profile: str, *, restart_pollution: list[str] | None = None):
    """Dry-run: parse-like defaults + optional restart pollution, then expand."""
    args = _ns(wm_strategy_profile=profile)
    for dest in {k for t in PROFILE_TABLES.values() for k in t}:
        if not hasattr(args, dest):
            setattr(args, dest, None)
    argv = ["train.py", f"--wm_strategy_profile={profile}"]
    if restart_pollution:
        argv.extend(restart_pollution)
        for tok in restart_pollution:
            if tok == "--enable_wm":
                args.enable_wm = True
            elif tok.startswith("--wm_strategy="):
                args.wm_strategy = tok.split("=", 1)[1]
            elif tok == "--eir_enable":
                args.eir_enable = True
            elif tok == "--no-eir_enable":
                args.eir_enable = False
    apply_wm_strategy_profile(
        args,
        argv=argv,
        authoritative=(profile != "legacy"),
    )
    return args, profile_capability_flags(args)


class TestStrategyZooLauncher(unittest.TestCase):
    def test_strategy_zoo_launcher_has_six_distinct_profiles(self):
        self.assertTrue(LAUNCHER.is_file(), f"missing launcher {LAUNCHER}")
        text = LAUNCHER.read_text()
        self.assertIn("--profile-per-policy", text)
        for prof in ZOO_PROFILES:
            self.assertIn(prof, text)
        line = next(l for l in text.splitlines() if "PROFILE_LIST=" in l)
        profiles = line.split("=", 1)[1].strip().strip('"').split(",")
        self.assertEqual(len(profiles), 6)
        self.assertEqual(len(set(profiles)), 6)

    def test_strategy_zoo_launcher_does_not_assign_aggressive_residual(self):
        text = LAUNCHER.read_text()
        self.assertIn("--no-eir_residual_apply_to_rollout", text)
        self.assertNotIn(
            "--eir_residual_apply_to_rollout",
            text.replace("--no-eir_residual_apply_to_rollout", ""),
        )

    def test_launcher_documents_ppo_only_anchor_semantics(self):
        text = LAUNCHER.read_text()
        self.assertIn("ppo_only_note=", text)
        self.assertIn("not an exact original DexPBT reproduction", text)


class TestLaunchPbtNormalize(unittest.TestCase):
    def test_bool_flag_not_merged_with_next_flag(self):
        extra = [
            "--pbt_wm_profile_copy_enable",
            "--pbt_wm_profile_copy_after_frames",
            "1000000000",
        ]
        out = normalize_extra_args(extra)
        self.assertEqual(
            out,
            [
                "--pbt_wm_profile_copy_enable",
                "--pbt_wm_profile_copy_after_frames=1000000000",
            ],
        )
        self.assertNotIn(
            "--pbt_wm_profile_copy_enable=--pbt_wm_profile_copy_after_frames", out
        )

    def test_value_flag_still_normalized(self):
        extra = ["--pbt_wm_profile_copy_prob", "0.3"]
        out = normalize_extra_args(extra)
        self.assertEqual(out, ["--pbt_wm_profile_copy_prob=0.3"])

    def test_strategy_zoo_bool_flags_listed(self):
        required = {
            "--pbt_wm_profile_copy_enable",
            "--no-pbt_wm_profile_copy_enable",
            "--wm_curriculum_controller_enable",
            "--no-wm_curriculum_controller_enable",
            "--wm_curriculum_controller_hard_block_shrink",
            "--no-wm_curriculum_controller_hard_block_shrink",
            "--wm_curriculum_controller_use_recovery_prob",
            "--no-wm_curriculum_controller_use_recovery_prob",
        }
        self.assertTrue(required.issubset(LAUNCH_PBT_BOOLEAN_FLAGS))


class TestProfileContaminationPreflight(unittest.TestCase):
    def test_ppo_only_not_polluted_by_restart_wm_args(self):
        args, caps = _simulate_profile_expansion(
            "ppo_only",
            restart_pollution=[
                "--enable_wm",
                "--wm_strategy=rssm_belief_value",
                "--eir_enable",
            ],
        )
        self.assertFalse(args.enable_wm)
        self.assertFalse(args.eir_enable)
        self.assertEqual(args.pbt_wm_score_mode, "off")
        self.assertFalse(caps["profile_residual_apply_enabled"])

    def test_all_profiles_disable_residual_apply(self):
        for profile in ZOO_PROFILES:
            args, caps = _simulate_profile_expansion(profile)
            self.assertFalse(
                args.eir_residual_apply_to_rollout,
                msg=f"{profile} must keep residual rollout apply off",
            )
            self.assertFalse(caps["profile_residual_apply_enabled"])

    def test_wm_pbt_source_selector_active_score(self):
        args, caps = _simulate_profile_expansion("wm_pbt_source_selector")
        self.assertEqual(args.pbt_wm_score_mode, "active")
        self.assertTrue(caps["profile_wm_pbt_score_enabled"])

    def test_ppo_only_anchor_flags(self):
        args, caps = _simulate_profile_expansion("ppo_only")
        self.assertFalse(args.enable_wm)
        self.assertEqual(args.pbt_wm_score_mode, "off")
        self.assertFalse(caps["profile_wm_pbt_score_enabled"])


def _make_observer(tmp: Path) -> DexPBTPbtAlgoObserver:
    obs = object.__new__(DexPBTPbtAlgoObserver)
    obs.wm_curriculum_controller_enable = True
    obs.wm_curriculum_controller_start_after_frames = 0
    obs.wm_curriculum_controller_min_nsucc_mean = 10.0
    obs.wm_curriculum_controller_drop_risk_threshold = 0.45
    obs.wm_curriculum_controller_uncertainty_threshold = 0.55
    obs.wm_curriculum_controller_min_success_prob = 0.20
    obs.wm_curriculum_controller_freeze_frames = 100_000_000
    obs.wm_curriculum_controller_hard_block_shrink = True
    obs.wm_curriculum_controller_use_recovery_prob = False
    obs.workspace = tmp
    obs.policy_idx = 0
    obs._wm_curriculum_freeze_until = {}
    return obs


class TestWMCurriculumController(unittest.TestCase):
    def _window(self, nsucc_mean: float = 15.0) -> RollingNsuccWindow:
        w = RollingNsuccWindow(max_len=8)
        w.append(
            WindowSample(
                env_frames=600_000_000,
                nsucc_mean=nsucc_mean,
                nsucc_max=nsucc_mean + 5,
                success_tolerance=0.05,
            )
        )
        return w

    def test_wm_curriculum_controller_blocks_shrink_on_high_drop_risk(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            (tmp / "wm_imagined").mkdir()
            payload = {
                "imagined_valid": True,
                "wm_drop_risk": 0.9,
                "wm_success_prob": 0.1,
                "wm_recovery_prob": 0.1,
                "uncertainty": 0.2,
            }
            (tmp / "wm_imagined" / "000.json").write_text(json.dumps(payload))
            obs = _make_observer(tmp)
            out = obs._evaluate_wm_curriculum_controller(0, self._window(), 600_000_000)
            self.assertTrue(out["blocked_shrink"])
            self.assertEqual(out["fallback_reason_id"], 5)

    def test_wm_curriculum_controller_does_not_block_on_invalid_eval(self):
        with tempfile.TemporaryDirectory() as td:
            obs = _make_observer(Path(td))
            out = obs._evaluate_wm_curriculum_controller(0, self._window(), 600_000_000)
            self.assertFalse(out["blocked_shrink"])
            self.assertEqual(out["fallback_reason_id"], 3)

    def test_wm_curriculum_controller_logs_fallback(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            (tmp / "wm_imagined").mkdir()
            (tmp / "wm_imagined" / "000.json").write_text("{bad json")
            obs = _make_observer(tmp)
            out = obs._evaluate_wm_curriculum_controller(0, self._window(), 600_000_000)
            self.assertFalse(out["eval_valid"])
            self.assertEqual(out["fallback_reason_id"], 3)


if __name__ == "__main__":
    unittest.main()
