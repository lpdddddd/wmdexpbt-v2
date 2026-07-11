"""Patch P1 (2026-06-01): env-side mastery curriculum dispatch."""

import tempfile
import unittest
from pathlib import Path

import torch

from dexpbt_isaaclab.dexpbt_utils import dexpbt_tolerance_curriculum
from dexpbt_isaaclab.pbt.pbt_curriculum_p1 import (
    CurriculumControlState,
    apply_env_tolerance_curriculum,
    write_curriculum_control,
)


def _ts(mean_nsucc: float) -> torch.Tensor:
    return torch.full((4,), float(mean_nsucc))


class ApplyEnvToleranceCurriculumTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.control_path = Path(self._tmp.name) / "000.json"

    def _base_kwargs(self, **overrides):
        kwargs = dict(
            curriculum_mode="legacy",
            curriculum_control_path=None,
            curriculum_shrink_factor=0.95,
            last_curriculum_update=0,
            frames_since_restart=100_000_000,
            curriculum_interval=10_000_000,
            prev_episode_successes=_ts(5.0),
            success_tolerance=0.075,
            initial_tolerance=0.075,
            target_tolerance=0.01,
            curriculum_freeze_frames_after_restart=0,
        )
        kwargs.update(overrides)
        return kwargs

    def test_legacy_mode_matches_direct_helper(self):
        kwargs = self._base_kwargs(curriculum_mode="legacy")
        tol, lcu = apply_env_tolerance_curriculum(**kwargs)
        exp_tol, exp_lcu = dexpbt_tolerance_curriculum(
            kwargs["last_curriculum_update"],
            kwargs["frames_since_restart"],
            kwargs["curriculum_interval"],
            kwargs["prev_episode_successes"],
            kwargs["success_tolerance"],
            kwargs["initial_tolerance"],
            kwargs["target_tolerance"],
            curriculum_freeze_frames_after_restart=0,
        )
        self.assertAlmostEqual(tol, exp_tol, places=6)
        self.assertEqual(lcu, exp_lcu)

    def test_mastery_shrink_when_gate_passes(self):
        write_curriculum_control(
            self.control_path,
            CurriculumControlState(
                curriculum_mode="mastery",
                allow_shrink=True,
                shrink_token=True,
                collapse_freeze_active=False,
            ),
        )
        tol, lcu = apply_env_tolerance_curriculum(
            **self._base_kwargs(
                curriculum_mode="mastery",
                curriculum_control_path=self.control_path,
            )
        )
        self.assertAlmostEqual(tol, 0.075 * 0.95, places=6)
        self.assertEqual(lcu, 100_000_000)

    def test_mastery_blocked_when_gate_fails(self):
        write_curriculum_control(
            self.control_path,
            CurriculumControlState(
                curriculum_mode="mastery",
                allow_shrink=False,
                shrink_blocked_reason="mean_below_threshold",
            ),
        )
        tol, lcu = apply_env_tolerance_curriculum(
            **self._base_kwargs(
                curriculum_mode="mastery",
                curriculum_control_path=self.control_path,
            )
        )
        self.assertEqual(tol, 0.075)
        self.assertEqual(lcu, 0)

    def test_mastery_collapse_freeze_holds_tolerance(self):
        write_curriculum_control(
            self.control_path,
            CurriculumControlState(
                curriculum_mode="mastery",
                allow_shrink=True,
                collapse_freeze_active=True,
            ),
        )
        tol, lcu = apply_env_tolerance_curriculum(
            **self._base_kwargs(
                curriculum_mode="mastery",
                curriculum_control_path=self.control_path,
            )
        )
        self.assertEqual(tol, 0.075)
        self.assertEqual(lcu, 100_000_000)

    def test_mastery_missing_control_falls_back_to_legacy(self):
        missing = Path(self._tmp.name) / "missing.json"
        tol, lcu = apply_env_tolerance_curriculum(
            **self._base_kwargs(
                curriculum_mode="mastery",
                curriculum_control_path=missing,
            )
        )
        self.assertAlmostEqual(tol, 0.0675, places=6)
        self.assertEqual(lcu, 100_000_000)


if __name__ == "__main__":
    unittest.main()
