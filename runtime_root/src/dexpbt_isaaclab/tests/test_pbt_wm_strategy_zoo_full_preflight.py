"""Preflight tests for PBT-WM Strategy Zoo v1.0 launcher + WM curriculum controller."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from dexpbt_isaaclab.pbt.pbt_curriculum_p1 import RollingNsuccWindow, WindowSample
from dexpbt_isaaclab.pbt.pbt_observer import DexPBTPbtAlgoObserver


REPO_ROOT = Path(__file__).resolve().parents[4]
LAUNCHER = (
    REPO_ROOT
    / "results/launchers/start_regrasping_pbt_wm_strategy_zoo_full_3b.sh"
)


class TestStrategyZooLauncher(unittest.TestCase):
    def test_strategy_zoo_launcher_has_six_distinct_profiles(self):
        self.assertTrue(LAUNCHER.is_file(), f"missing launcher {LAUNCHER}")
        text = LAUNCHER.read_text()
        self.assertIn("--profile-per-policy", text)
        expected = [
            "ppo_only",
            "rssm_mppi_actor_distill",
            "wm_curriculum_controller",
            "wm_pbt_source_selector",
            "elite_bank_actor_distill",
            "conservative_rssm_planner_distill",
        ]
        for prof in expected:
            self.assertIn(prof, text)
        line = next(l for l in text.splitlines() if "PROFILE_LIST=" in l)
        profiles = line.split("=", 1)[1].strip().strip('"').split(",")
        self.assertEqual(len(profiles), 6)
        self.assertEqual(len(set(profiles)), 6)

    def test_strategy_zoo_launcher_does_not_assign_aggressive_residual(self):
        text = LAUNCHER.read_text()
        self.assertIn("--no-eir_residual_apply_to_rollout", text)
        self.assertNotIn("--eir_residual_apply_to_rollout", text.replace("--no-eir_residual_apply_to_rollout", ""))


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
