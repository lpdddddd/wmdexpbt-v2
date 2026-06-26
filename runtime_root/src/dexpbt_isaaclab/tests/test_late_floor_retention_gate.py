"""Late-floor retention gate tests (CSER patch)."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from dexpbt_isaaclab.pbt.pbt_curriculum_p1 import WindowSample
from dexpbt_isaaclab.pbt.pbt_observer import DexPBTPbtAlgoObserver


def _sample(*, mean: float, max_: float, tol: float = 0.023) -> WindowSample:
    return WindowSample(
        env_frames=0,
        nsucc_mean=mean,
        nsucc_max=max_,
        success_tolerance=tol,
        pbt_iteration=0,
    )


def _make_observer(
    workspace: Path,
    *,
    gate_enable: bool = True,
    rollback_enable: bool = False,
) -> DexPBTPbtAlgoObserver:
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
        late_floor_retention_gate_enable=gate_enable,
        late_floor_retention_tol_threshold=0.025,
        late_floor_retention_peak_threshold=35.0,
        late_floor_retention_min_retention=0.85,
        late_floor_retention_hard_min_retention=0.75,
        late_floor_retention_patience=2,
        late_floor_retention_freeze_frames=200_000_000,
        late_floor_retention_block_shrink=True,
        late_floor_retention_freeze_mutation=True,
        late_floor_retention_rollback_enable=rollback_enable,
    )
    obs.writer = None
    return obs


class _TmpWs(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.ws = Path(self._tmp.name)


class LateFloorRetentionGateTest(_TmpWs):
    def test_blocks_shrink_on_late_floor_low_retention(self):
        obs = _make_observer(self.ws)
        window = obs._policy_window(1)
        window.samples = [_sample(mean=40.0, max_=50.0), _sample(mean=32.0, max_=50.0)]
        obs._recent_peak_nsucc_mean[1] = 40.0
        out, forced = obs._evaluate_late_floor_retention_gate(
            1,
            window,
            env_frames=1_000_000_000,
            cur_tol=0.023,
            shrink_token=True,
            recent_peak=40.0,
            forced_tolerance=0.0,
        )
        self.assertTrue(out["active"])
        self.assertTrue(out["blocked"])
        self.assertEqual(out["reason"], "late_floor_low_retention")
        self.assertAlmostEqual(out["retention"], 0.8, places=6)
        self.assertEqual(forced, 0.0)

    def test_no_trigger_when_not_late_floor(self):
        obs = _make_observer(self.ws)
        window = obs._policy_window(1)
        window.samples = [_sample(mean=32.0, max_=50.0, tol=0.030)]
        out, _ = obs._evaluate_late_floor_retention_gate(
            1,
            window,
            env_frames=1_000_000_000,
            cur_tol=0.030,
            shrink_token=True,
            recent_peak=40.0,
            forced_tolerance=0.0,
        )
        self.assertFalse(out["active"])
        self.assertFalse(out["blocked"])

    def test_no_trigger_when_peak_not_strong(self):
        obs = _make_observer(self.ws)
        window = obs._policy_window(1)
        window.samples = [_sample(mean=30.0, max_=50.0), _sample(mean=24.0, max_=50.0)]
        out, _ = obs._evaluate_late_floor_retention_gate(
            1,
            window,
            env_frames=1_000_000_000,
            cur_tol=0.023,
            shrink_token=True,
            recent_peak=30.0,
            forced_tolerance=0.0,
        )
        self.assertFalse(out["active"])

    def test_no_trigger_when_retention_ok(self):
        obs = _make_observer(self.ws)
        window = obs._policy_window(1)
        window.samples = [_sample(mean=40.0, max_=50.0), _sample(mean=36.0, max_=50.0)]
        out, _ = obs._evaluate_late_floor_retention_gate(
            1,
            window,
            env_frames=1_000_000_000,
            cur_tol=0.023,
            shrink_token=True,
            recent_peak=40.0,
            forced_tolerance=0.0,
        )
        self.assertTrue(out["active"])
        self.assertFalse(out["blocked"])

    def test_mutation_freeze_is_set(self):
        obs = _make_observer(self.ws)
        window = obs._policy_window(1)
        window.samples = [_sample(mean=40.0, max_=50.0), _sample(mean=32.0, max_=50.0)]
        obs._evaluate_late_floor_retention_gate(
            1,
            window,
            env_frames=1_000_000_000,
            cur_tol=0.023,
            shrink_token=True,
            recent_peak=40.0,
            forced_tolerance=0.0,
        )
        self.assertGreater(obs._mutation_frozen_until.get(1, 0), 1_000_000_000)

    def test_rollback_disabled_by_default(self):
        obs = _make_observer(self.ws, rollback_enable=False)
        window = obs._policy_window(1)
        window.samples = [_sample(mean=40.0, max_=50.0), _sample(mean=28.0, max_=50.0)]
        obs._last_mastered_safe_tolerance[1] = 0.030
        for _ in range(3):
            out, forced = obs._evaluate_late_floor_retention_gate(
                1,
                window,
                env_frames=1_000_000_000,
                cur_tol=0.023,
                shrink_token=True,
                recent_peak=40.0,
                forced_tolerance=0.0,
            )
        self.assertFalse(out["rollback_applied"])
        self.assertEqual(forced, 0.0)


if __name__ == "__main__":
    unittest.main()
