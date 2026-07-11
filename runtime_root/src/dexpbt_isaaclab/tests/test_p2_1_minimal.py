"""Patch P2.1 minimal stability tests."""

import tempfile
import unittest
from pathlib import Path

from dexpbt_isaaclab.pbt.pbt_curriculum_p1 import (
    CurriculumControlState,
    RollingNsuccWindow,
    WindowSample,
    is_usable_live_window,
    qualifies_best_stable_save,
    read_curriculum_control,
    write_curriculum_control,
)
from dexpbt_isaaclab.pbt.pbt_observer import DexPBTPbtAlgoObserver


def _make_observer_for_tolerance(
    workspace: Path,
    *,
    crosscopy_mode: str = "last_mastered",
    recovery_mode: str = "checkpoint",
) -> DexPBTPbtAlgoObserver:
    obs = DexPBTPbtAlgoObserver(
        policy_idx=1,
        num_policies=3,
        mutable_params={
            "enable_wm": False,
            "enable_counterfactual": False,
            "wm_strategy": "none",
        },
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
        pbt_recovery_tolerance_mode=recovery_mode,
        pbt_crosscopy_tolerance_mode=crosscopy_mode,
    )
    obs.writer = None
    return obs


def _sample(mean: float, max_: float, tol: float = 0.05) -> WindowSample:
    return WindowSample(
        env_frames=100,
        nsucc_mean=float(mean),
        nsucc_max=float(max_),
        success_tolerance=float(tol),
        pbt_iteration=0,
    )


class P21MinimalTest(unittest.TestCase):
    def test_usable_live_thresholds_default_p21(self):
        w = RollingNsuccWindow()
        w.append(_sample(1.6, 22.0))
        ok, reason = is_usable_live_window(
            w,
            min_mean=1.5,
            min_max=20.0,
            max_drop_ratio=0.5,
        )
        self.assertTrue(ok, reason)

    def test_usable_live_rejects_high_drop(self):
        w = RollingNsuccWindow()
        w.append(_sample(4.0, 30.0))
        w.append(_sample(1.8, 22.0))
        ok, reason = is_usable_live_window(
            w,
            min_mean=1.5,
            min_max=20.0,
            max_drop_ratio=0.5,
        )
        self.assertFalse(ok)
        self.assertEqual(reason, "usable_drop_ratio")

    def test_best_stable_requires_two_windows(self):
        w = RollingNsuccWindow()
        w.append(_sample(3.2, 22.0, tol=0.05))
        ok, reason = qualifies_best_stable_save(
            w,
            min_mean=3.0,
            min_max=20.0,
            max_drop_ratio=0.3,
            stable_tolerance_intervals=1,
            required_stable_windows=2,
            consolidation_active=False,
        )
        self.assertFalse(ok)
        self.assertEqual(reason, "not_two_stable_windows")

    def test_best_stable_rejects_consolidation(self):
        w = RollingNsuccWindow()
        w.append(_sample(3.2, 22.0, tol=0.05))
        w.append(_sample(3.1, 21.0, tol=0.05))
        ok, reason = qualifies_best_stable_save(
            w,
            min_mean=3.0,
            min_max=20.0,
            max_drop_ratio=0.3,
            stable_tolerance_intervals=1,
            required_stable_windows=2,
            consolidation_active=True,
        )
        self.assertFalse(ok)
        self.assertEqual(reason, "consolidation_active")

    def test_best_stable_two_windows_pass(self):
        w = RollingNsuccWindow()
        w.append(_sample(3.3, 24.0, tol=0.05))
        w.append(_sample(3.2, 23.0, tol=0.05))
        ok, reason = qualifies_best_stable_save(
            w,
            min_mean=3.0,
            min_max=20.0,
            max_drop_ratio=0.3,
            stable_tolerance_intervals=1,
            required_stable_windows=2,
            consolidation_active=False,
        )
        self.assertTrue(ok, reason)

    def test_curriculum_control_roundtrip_with_rollback_reason(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "000.json"
            state = CurriculumControlState(
                rollback_reason="cooldown_elapsed",
                rollback_count=2,
                last_rollback_frame=12345,
                rollback_cooldown_active=True,
            )
            write_curriculum_control(path, state)
            loaded = read_curriculum_control(path)
            self.assertIsNotNone(loaded)
            assert loaded is not None
            self.assertEqual(loaded.rollback_reason, "cooldown_elapsed")
            self.assertEqual(loaded.rollback_count, 2)
            self.assertEqual(loaded.last_rollback_frame, 12345)
            self.assertTrue(loaded.rollback_cooldown_active)

    def test_crosscopy_mode_last_mastered_for_live_cross_copy(self):
        with tempfile.TemporaryDirectory() as tmp:
            obs = _make_observer_for_tolerance(
                Path(tmp),
                crosscopy_mode="last_mastered",
                recovery_mode="checkpoint",
            )
            source_policy = 2
            obs._last_mastered_tolerance[source_policy] = 0.05
            source_curriculum = {"success_tolerance": 0.075}
            tol, mode = obs._resolve_restart_success_tolerance(
                source_policy=source_policy,
                source_curriculum=source_curriculum,
                checkpoint_type="latest_healthy",
            )
            self.assertEqual(mode, "last_mastered")
            self.assertAlmostEqual(tol, 0.05)

    def test_archive_recovery_unchanged_by_crosscopy_mode(self):
        with tempfile.TemporaryDirectory() as tmp:
            obs = _make_observer_for_tolerance(
                Path(tmp),
                crosscopy_mode="last_mastered",
                recovery_mode="checkpoint",
            )
            source_policy = 2
            obs._last_mastered_tolerance[source_policy] = 0.05
            source_curriculum = {"success_tolerance": 0.075}
            tol, mode = obs._resolve_restart_success_tolerance(
                source_policy=source_policy,
                source_curriculum=source_curriculum,
                checkpoint_type="best_stable_archive",
            )
            self.assertEqual(mode, "checkpoint")
            self.assertAlmostEqual(tol, 0.075)


if __name__ == "__main__":
    unittest.main()
