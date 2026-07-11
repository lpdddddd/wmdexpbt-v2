"""Patch P2-A (2026-06-01): single-shot shrink token + post-shrink consolidation."""

import tempfile
import unittest
from pathlib import Path

import torch

from dexpbt_isaaclab.dexpbt_utils import env_frames_to_sim_steps
from dexpbt_isaaclab.pbt.pbt_curriculum_p1 import (
    CurriculumControlState,
    apply_env_tolerance_curriculum,
    dexpbt_tolerance_curriculum_mastery,
    mark_shrink_consumed,
    post_shrink_consolidation_active,
    read_curriculum_control,
    write_curriculum_control,
)


def _ts(mean_nsucc: float) -> torch.Tensor:
    return torch.full((4,), float(mean_nsucc))


class P2SingleShrinkTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.control_path = Path(self._tmp.name) / "000.json"

    def test_consolidation_blocks_second_shrink(self):
        until = 50_000
        blocked, _, _ = dexpbt_tolerance_curriculum_mastery(
            0,
            10_000,
            1,
            _ts(5.0),
            0.075,
            0.075,
            0.01,
            allow_shrink=True,
            post_shrink_consolidation_until_restart_step=until,
        )
        self.assertFalse(blocked[2])
        self.assertEqual(blocked[3], "post_shrink_consolidation")

    def test_mark_shrink_consumed_sets_consolidation(self):
        write_curriculum_control(
            self.control_path,
            CurriculumControlState(shrink_token=True, allow_shrink=True),
        )
        mark_shrink_consumed(
            self.control_path,
            frames_since_restart=1000,
            consolidation_env_frames=100_000_000,
            num_envs=8192,
        )
        ctrl = read_curriculum_control(self.control_path)
        assert ctrl is not None
        self.assertFalse(ctrl.shrink_token)
        self.assertTrue(ctrl.just_shrank)
        expected_until = 1000 + env_frames_to_sim_steps(100_000_000, 8192)
        self.assertEqual(ctrl.post_shrink_consolidation_until_restart_step, expected_until)
        self.assertTrue(post_shrink_consolidation_active(1000, ctrl))
        self.assertFalse(post_shrink_consolidation_active(expected_until, ctrl))

    def test_env_single_shrink_consumes_token_once(self):
        write_curriculum_control(
            self.control_path,
            CurriculumControlState(
                curriculum_mode="mastery",
                shrink_token=True,
                allow_shrink=True,
            ),
        )
        kwargs = dict(
            curriculum_mode="mastery",
            curriculum_control_path=self.control_path,
            curriculum_shrink_factor=0.95,
            last_curriculum_update=0,
            frames_since_restart=100_000_000,
            curriculum_interval=10_000_000,
            prev_episode_successes=_ts(5.0),
            success_tolerance=0.075,
            initial_tolerance=0.075,
            target_tolerance=0.01,
            post_shrink_consolidation_env_frames=100_000_000,
            num_envs=8192,
            pbt_single_shrink_per_window=True,
        )
        tol1, _ = apply_env_tolerance_curriculum(**kwargs)
        self.assertAlmostEqual(tol1, 0.075 * 0.95, places=6)
        ctrl = read_curriculum_control(self.control_path)
        assert ctrl is not None
        self.assertFalse(ctrl.shrink_token)

        tol2, _ = apply_env_tolerance_curriculum(**kwargs)
        self.assertAlmostEqual(tol2, tol1, places=6)


if __name__ == "__main__":
    unittest.main()
