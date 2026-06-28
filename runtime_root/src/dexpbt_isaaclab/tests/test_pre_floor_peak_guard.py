"""Pre-floor peak retention guard tests (CSER retention patch)."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from dexpbt_isaaclab.pbt.pbt_curriculum_p1 import WindowSample
from dexpbt_isaaclab.pbt.pbt_observer import DexPBTPbtAlgoObserver


def _sample(*, mean: float, max_: float, tol: float = 0.026) -> WindowSample:
    return WindowSample(
        env_frames=0,
        nsucc_mean=mean,
        nsucc_max=max_,
        success_tolerance=tol,
        pbt_iteration=0,
    )


def _make_observer(workspace: Path) -> DexPBTPbtAlgoObserver:
    obs = DexPBTPbtAlgoObserver(
        policy_idx=1,
        num_policies=2,
        mutable_params={"enable_wm": False, "wm_strategy": "none"},
        workspace=str(workspace),
        train_dir=str(workspace),
        task_name="test-task",
        interval_steps=10_000_000,
        start_after=0,
        initial_delay=0,
        num_envs=1,
        pbt_min_frames_as_source_after_restart=0,
        pbt_min_frames_between_restarts=0,
        curriculum_mode="mastery",
        curriculum_shrink_factor=0.95,
        pbt_hard_floor_tol_threshold=0.025,
        pre_floor_peak_guard_enable=True,
        pre_floor_peak_guard_tol_threshold=0.030,
        pre_floor_peak_guard_target_tol=0.025,
        pre_floor_peak_guard_peak_threshold=35.0,
        pre_floor_peak_guard_min_retention=0.92,
        pre_floor_peak_guard_min_stable_ticks=3,
        pre_floor_peak_guard_freeze_frames=200_000_000,
    )
    obs.writer = None
    return obs


class PreFloorPeakGuardTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.ws = Path(self._tmp.name)

    def test_blocks_low_retention_before_target_tol(self):
        obs = _make_observer(self.ws)
        window = obs._policy_window(1)
        window.samples = [_sample(mean=40.0, max_=50.0), _sample(mean=34.0, max_=50.0)]
        out = obs._evaluate_pre_floor_peak_guard(
            1,
            window,
            env_frames=1_000_000_000,
            cur_tol=0.026,
            shrink_token=True,
            recent_peak=40.0,
            adaptive_shrink_factor=0.95,
            stable_ticks=2,
        )
        self.assertTrue(out["active"])
        self.assertTrue(out["blocked"])
        self.assertEqual(out["reason"], "pre_floor_peak_retention_guard")

    def test_allows_when_retention_and_stable_ticks_ok(self):
        obs = _make_observer(self.ws)
        window = obs._policy_window(1)
        window.samples = [_sample(mean=40.0, max_=50.0), _sample(mean=37.0, max_=50.0)]
        out = obs._evaluate_pre_floor_peak_guard(
            1,
            window,
            env_frames=1_000_000_000,
            cur_tol=0.026,
            shrink_token=True,
            recent_peak=40.0,
            adaptive_shrink_factor=0.95,
            stable_ticks=3,
        )
        self.assertTrue(out["active"])
        self.assertFalse(out["blocked"])
        self.assertEqual(out["reason"], "pre_floor_peak_retention_ok")

    def test_no_trigger_when_peak_too_low(self):
        obs = _make_observer(self.ws)
        window = obs._policy_window(1)
        window.samples = [_sample(mean=30.0, max_=50.0), _sample(mean=26.0, max_=50.0)]
        out = obs._evaluate_pre_floor_peak_guard(
            1,
            window,
            env_frames=1_000_000_000,
            cur_tol=0.026,
            shrink_token=True,
            recent_peak=30.0,
            adaptive_shrink_factor=0.95,
            stable_ticks=3,
        )
        self.assertFalse(out["active"])

    def test_no_trigger_when_next_tol_stays_above_target(self):
        obs = _make_observer(self.ws)
        window = obs._policy_window(1)
        window.samples = [_sample(mean=40.0, max_=50.0), _sample(mean=34.0, max_=50.0)]
        out = obs._evaluate_pre_floor_peak_guard(
            1,
            window,
            env_frames=1_000_000_000,
            cur_tol=0.026,
            shrink_token=True,
            recent_peak=40.0,
            adaptive_shrink_factor=0.99,
            stable_ticks=1,
        )
        self.assertFalse(out["active"])

    def test_writes_safety_event_on_block(self):
        obs = _make_observer(self.ws)
        window = obs._policy_window(1)
        window.samples = [_sample(mean=40.0, max_=50.0), _sample(mean=34.0, max_=50.0)]
        obs._evaluate_pre_floor_peak_guard(
            1,
            window,
            env_frames=1_000_000_000,
            cur_tol=0.026,
            shrink_token=True,
            recent_peak=40.0,
            adaptive_shrink_factor=0.95,
            stable_ticks=1,
        )
        events_path = Path(obs.workspace) / "metrics" / "curriculum_safety_events.jsonl"
        self.assertTrue(events_path.is_file())
        text = events_path.read_text()
        self.assertIn("pre_floor_peak_retention_guard", text)


if __name__ == "__main__":
    unittest.main()
