"""Hybrid Curriculum Safety Patch tests (2026-06-15)."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

from dexpbt_isaaclab.pbt.pbt_curriculum_p1 import (
    WindowSample,
    apply_env_tolerance_curriculum,
    read_curriculum_control,
    write_curriculum_control,
    CurriculumControlState,
)
from dexpbt_isaaclab.pbt.pbt_observer import DexPBTPbtAlgoObserver

# Analyzer lives under results/; add repo root for import in tests.
_REPO_ROOT = Path(__file__).resolve().parents[4]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
from results.analyze_pbt_wm_strategy_zoo_full_run import (  # noqa: E402
    _shrink_safety_stats,
    _tol_exposure_stats,
)


def _sample(
    *,
    mean: float,
    max_: float,
    tol: float = 0.075,
    iteration: int = 0,
    env_frames: int = 0,
) -> WindowSample:
    return WindowSample(
        env_frames=env_frames,
        nsucc_mean=mean,
        nsucc_max=max_,
        success_tolerance=tol,
        pbt_iteration=iteration,
    )


def _make_observer(
    workspace: Path,
    *,
    curriculum_safety_enable: bool = True,
    curriculum_safety_use_real_fallback: bool = True,
    curriculum_strong_shrink_global_retention_gate: bool = True,
    curriculum_danger_zone_enable: bool = True,
    curriculum_rollback_on_repeated_collapse: bool = True,
    curriculum_rollback_to_last_mastered: bool = True,
    speed_mode: str = "adaptive_fast",
    **extra,
) -> DexPBTPbtAlgoObserver:
    defaults = dict(
        policy_idx=0,
        num_policies=1,
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
        pbt_curriculum_freeze_frames=100_000_000,
        curriculum_mode="mastery",
        mastery_nsucc_mean=4.0,
        mastery_nsucc_max=20.0,
        mastery_consecutive_windows=2,
        curriculum_shrink_factor=0.95,
        pbt_single_shrink_per_window=True,
        post_shrink_consolidation_env_frames=100_000_000,
        curriculum_collapse_detect=True,
        pbt_early_collapse_detect=True,
        curriculum_speed_mode=speed_mode,
        curriculum_fast_strong_mean=30.0,
        curriculum_fast_mid_mean=20.0,
        curriculum_fast_low_mean=10.0,
        curriculum_fast_retention_threshold=0.8,
        curriculum_fast_retention_window=4,
        curriculum_fast_shrink_strong=0.925,
        curriculum_fast_shrink_mid=0.95,
        curriculum_fast_shrink_low=0.975,
        curriculum_fast_consolidation_frames=50_000_000,
        curriculum_fast_recovery_consolidation_frames=100_000_000,
        curriculum_safety_enable=curriculum_safety_enable,
        curriculum_safety_start_after_frames=0,
        curriculum_safety_use_real_fallback=curriculum_safety_use_real_fallback,
        curriculum_safety_global_retention_threshold=0.80,
        curriculum_safety_drop_ratio_threshold=0.30,
        curriculum_safety_slope_threshold=-2.0,
        curriculum_safety_window=6,
        curriculum_safety_block_frames=10_000_000,
        curriculum_strong_shrink_global_retention_gate=(
            curriculum_strong_shrink_global_retention_gate
        ),
        curriculum_strong_shrink_min_global_retention=0.85,
        curriculum_strong_shrink_min_stable_ticks=3,
        curriculum_strong_shrink_long_window=12,
        curriculum_danger_zone_enable=curriculum_danger_zone_enable,
        curriculum_danger_tol=0.030,
        curriculum_danger_disable_strong_shrink=True,
        curriculum_danger_min_stable_ticks=4,
        curriculum_danger_global_retention_threshold=0.85,
        curriculum_rollback_on_repeated_collapse=(
            curriculum_rollback_on_repeated_collapse
        ),
        curriculum_rollback_collapse_patience=2,
        curriculum_rollback_to_last_mastered=curriculum_rollback_to_last_mastered,
        curriculum_rollback_freeze_frames=50_000_000,
        curriculum_last_mastered_min_nsucc_mean=30.0,
        curriculum_last_mastered_min_global_retention=0.85,
        curriculum_last_mastered_stable_ticks=3,
        wm_curriculum_controller_enable=True,
        wm_curriculum_controller_start_after_frames=0,
    )
    defaults.update(extra)
    obs = DexPBTPbtAlgoObserver(**defaults)
    obs.writer = None
    return obs


class CurriculumSafetyFallbackTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.ws = Path(self._tmp.name)
        (self.ws / "curriculum_control").mkdir(parents=True, exist_ok=True)

    def test_real_fallback_blocks_shrink_when_imagined_invalid_and_retention_low(self):
        obs = _make_observer(self.ws)
        window = obs._policy_window(0)
        window.samples = [
            _sample(mean=35.0, max_=50.0, tol=0.04, env_frames=600_000_000 + i)
            for i in range(6)
        ] + [
            _sample(mean=20.0, max_=50.0, tol=0.04, env_frames=600_000_010),
        ]
        obs._recent_peak_nsucc_mean[0] = 50.0
        safety = obs._evaluate_curriculum_safety(
            0,
            window,
            600_000_010,
            0.04,
            True,
            wm_cc={"eval_valid": False},
            global_retention=0.4,
            stable_ticks=6,
            danger_zone=False,
            repeated_collapse_count=0,
        )
        self.assertTrue(safety["blocked"])
        self.assertEqual(safety["block_reason"], "real_signal_risk")
        self.assertTrue(safety["real_fallback_used"])

    def test_imagined_invalid_does_not_disable_real_safety(self):
        obs = _make_observer(self.ws)
        window = obs._policy_window(0)
        window.samples = [_sample(mean=15.0, max_=40.0, tol=0.05)]
        safety = obs._evaluate_curriculum_safety(
            0,
            window,
            600_000_000,
            0.05,
            True,
            wm_cc={"eval_valid": False, "drop_risk": 0.0, "uncertainty": 1.0},
            global_retention=0.5,
            stable_ticks=1,
            danger_zone=False,
            repeated_collapse_count=0,
        )
        self.assertTrue(safety["real_risk"])
        self.assertFalse(safety["wm_risk"])

    def test_valid_wm_risk_blocks_shrink(self):
        obs = _make_observer(self.ws)
        window = obs._policy_window(0)
        window.samples = [_sample(mean=35.0, max_=40.0, tol=0.05)]
        safety = obs._evaluate_curriculum_safety(
            0,
            window,
            600_000_000,
            0.05,
            True,
            wm_cc={
                "eval_valid": True,
                "drop_risk": 0.9,
                "uncertainty": 0.2,
                "success_prob": 0.5,
            },
            global_retention=0.95,
            stable_ticks=5,
            danger_zone=False,
            repeated_collapse_count=0,
        )
        self.assertTrue(safety["blocked"])
        self.assertEqual(safety["block_reason"], "wm_risk_valid")

    def test_no_risk_allows_shrink(self):
        obs = _make_observer(self.ws)
        window = obs._policy_window(0)
        window.samples = [_sample(mean=35.0, max_=40.0, tol=0.05)]
        safety = obs._evaluate_curriculum_safety(
            0,
            window,
            600_000_000,
            0.05,
            True,
            wm_cc={"eval_valid": False},
            global_retention=0.95,
            stable_ticks=5,
            danger_zone=False,
            repeated_collapse_count=0,
        )
        self.assertFalse(safety["blocked"])


class StrongShrinkGateTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.ws = Path(self._tmp.name)
        (self.ws / "curriculum_control").mkdir(parents=True, exist_ok=True)

    def _ctrl(self, obs, samples, env_frames=800_000_000):
        window = obs._policy_window(0)
        window.samples = samples
        obs._recent_peak_nsucc_mean[0] = max(s.nsucc_mean for s in samples)
        return obs._build_curriculum_control_state(
            policy_idx=0,
            window=window,
            env_frames=env_frames,
            population={
                0: {
                    "frame": env_frames,
                    "frames_at_restart": 0,
                    "curriculum": {"frame_since_restart": env_frames},
                }
            },
        )

    def test_strong_shrink_blocked_by_low_global_retention(self):
        obs = _make_observer(self.ws)
        samples = [
            _sample(mean=50.0, max_=55.0, tol=0.05, iteration=i, env_frames=i)
            for i in range(8)
        ] + [
            _sample(mean=40.0, max_=55.0, tol=0.05, iteration=8, env_frames=8)
            for _ in range(4)
        ]
        ctrl = self._ctrl(obs, samples)
        self.assertNotEqual(ctrl.shrink_reason, "strong")
        self.assertFalse(ctrl.curriculum_strong_shrink_allowed)

    def test_strong_shrink_allowed_when_global_retention_stable(self):
        obs = _make_observer(self.ws)
        samples = [
            _sample(mean=35.0, max_=40.0, tol=0.05, iteration=i, env_frames=i)
            for i in range(8)
        ]
        ctrl = self._ctrl(obs, samples)
        self.assertEqual(ctrl.shrink_reason, "strong")
        self.assertTrue(ctrl.curriculum_strong_shrink_allowed)


class DangerZoneTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.ws = Path(self._tmp.name)

    def test_danger_zone_disables_strong_shrink(self):
        obs = _make_observer(self.ws)
        window = obs._policy_window(0)
        window.samples = [
            _sample(mean=35.0, max_=40.0, tol=0.025, iteration=i, env_frames=i)
            for i in range(8)
        ]
        obs._recent_peak_nsucc_mean[0] = 40.0
        ctrl = obs._build_curriculum_control_state(
            0, window, 800_000_000, population={0: {"frame": 800_000_000}}
        )
        self.assertTrue(ctrl.curriculum_danger_zone)
        self.assertNotEqual(ctrl.shrink_reason, "strong")

    def test_danger_zone_requires_stable_ticks(self):
        obs = _make_observer(self.ws)
        window = obs._policy_window(0)
        window.samples = [
            _sample(mean=35.0, max_=40.0, tol=0.025, iteration=0, env_frames=0),
            _sample(mean=34.0, max_=40.0, tol=0.020, iteration=1, env_frames=1),
        ]
        safety = obs._evaluate_curriculum_safety(
            0,
            window,
            600_000_000,
            0.025,
            True,
            wm_cc={"eval_valid": False},
            global_retention=0.95,
            stable_ticks=1,
            danger_zone=True,
            repeated_collapse_count=0,
        )
        self.assertTrue(safety["blocked"])
        self.assertEqual(safety["block_reason"], "danger_zone_not_stable")


class RollbackTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.ws = Path(self._tmp.name)
        (self.ws / "curriculum_control").mkdir(parents=True, exist_ok=True)

    def test_repeated_collapse_rolls_back_to_last_mastered(self):
        obs = _make_observer(self.ws)
        obs._last_mastered_safe_tolerance[0] = 0.04
        obs._repeated_collapse_count[0] = 2
        window = obs._policy_window(0)
        window.samples = [
            _sample(mean=10.0, max_=40.0, tol=0.025, iteration=0, env_frames=0),
            _sample(mean=5.0, max_=40.0, tol=0.025, iteration=1, env_frames=1),
        ]
        obs._recent_peak_nsucc_mean[0] = 40.0
        ctrl = obs._build_curriculum_control_state(
            0, window, 800_000_000, population={0: {"frame": 800_000_000}}
        )
        self.assertTrue(ctrl.curriculum_rollback_triggered)
        self.assertAlmostEqual(ctrl.forced_success_tolerance, 0.04)
        self.assertEqual(ctrl.rollback_reason, "repeated_collapse_to_last_mastered")

    def test_no_last_mastered_freezes_without_rollback(self):
        obs = _make_observer(self.ws)
        obs._repeated_collapse_count[0] = 2
        window = obs._policy_window(0)
        window.samples = [
            _sample(mean=10.0, max_=40.0, tol=0.025, iteration=0, env_frames=0),
            _sample(mean=5.0, max_=40.0, tol=0.025, iteration=1, env_frames=1),
        ]
        obs._recent_peak_nsucc_mean[0] = 40.0
        ctrl = obs._build_curriculum_control_state(
            0, window, 800_000_000, population={0: {"frame": 800_000_000}}
        )
        self.assertFalse(ctrl.curriculum_rollback_triggered)
        self.assertEqual(ctrl.rollback_reason, "repeated_collapse_freeze_only")
        self.assertGreater(ctrl.curriculum_rollback_freeze_until, 800_000_000)

    def test_rollback_clamps_within_target_initial(self):
        path = self.ws / "curriculum_control" / "000.json"
        write_curriculum_control(
            path,
            CurriculumControlState(
                forced_success_tolerance=0.04,
                shrink_token=False,
            ),
        )
        tol, _ = apply_env_tolerance_curriculum(
            curriculum_mode="mastery",
            curriculum_control_path=path,
            curriculum_shrink_factor=0.95,
            last_curriculum_update=0,
            frames_since_restart=1000,
            curriculum_interval=100,
            prev_episode_successes=[],
            success_tolerance=0.025,
            initial_tolerance=0.075,
            target_tolerance=0.02,
        )
        self.assertGreaterEqual(tol, 0.02 - 1e-9)
        self.assertLessEqual(tol, 0.075 + 1e-9)
        self.assertAlmostEqual(tol, 0.04)


class AnalyzerSafetyTest(unittest.TestCase):
    def test_analyzer_reports_tol_exposure_and_survivorship_bias(self):
        p0_rows = [{"success_tolerance": 0.019, "nsucc_mean": 38.0}] * 2
        p3_rows = [{"success_tolerance": 0.019, "nsucc_mean": 25.0}] * 20
        p0_exp = _tol_exposure_stats(p0_rows)
        p3_exp = _tol_exposure_stats(p3_rows)
        self.assertEqual(p0_exp["le_002"]["ticks_by_tol_bin"], 2)
        self.assertEqual(p3_exp["le_002"]["ticks_by_tol_bin"], 20)

    def test_analyzer_reports_blocked_shrink_reason_counts(self):
        rows = [
            {
                "curriculum_safety_blocked_shrink": True,
                "curriculum_safety_block_reason": "real_signal_risk",
            },
            {
                "curriculum_safety_blocked_shrink": True,
                "curriculum_safety_block_reason": "danger_zone_not_stable",
            },
        ]
        stats = _shrink_safety_stats(rows)
        self.assertEqual(stats["blocked_shrink_count"], 2)
        self.assertEqual(stats["blocked_shrink_reason_counts"]["real_signal_risk"], 1)


if __name__ == "__main__":
    unittest.main()
