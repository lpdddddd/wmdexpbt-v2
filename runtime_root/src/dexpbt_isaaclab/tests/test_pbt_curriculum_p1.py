"""Patch P1 (2026-06-01): curriculum / stability helpers + observer tier cascade."""

import json
import tempfile
import unittest
from pathlib import Path

from dexpbt_isaaclab.pbt.pbt_curriculum_p1 import (
    RollingNsuccWindow,
    WindowSample,
    evaluate_collapse_freeze,
    evaluate_mastery_gate,
    is_stable_source_window,
    is_usable_live_window,
    qualifies_best_stable_save,
    read_curriculum_control,
    write_curriculum_control,
    CurriculumControlState,
)
from dexpbt_isaaclab.pbt.pbt_observer import DexPBTPbtAlgoObserver


def _sample(
    *,
    mean: float,
    max_: float,
    tol: float = 0.05,
    iteration: int = 0,
) -> WindowSample:
    return WindowSample(
        env_frames=100,
        nsucc_mean=mean,
        nsucc_max=max_,
        success_tolerance=tol,
        pbt_iteration=iteration,
    )


def _make_observer(
    workspace: Path,
    *,
    policy_idx: int = 2,
    source_require_stable_mean: bool = True,
    allow_best_nsucc_archive_recovery: bool = False,
) -> DexPBTPbtAlgoObserver:
    obs = DexPBTPbtAlgoObserver(
        policy_idx=policy_idx,
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
        pbt_source_require_stable_mean=source_require_stable_mean,
        pbt_allow_best_nsucc_archive_recovery=allow_best_nsucc_archive_recovery,
        pbt_enable_best_nsucc_recovery=False,
        curriculum_mode="mastery",
    )
    obs.writer = None
    return obs


def _state(
    *,
    policy_idx: int = 0,
    nsucc_mean: float = 5.0,
    nsucc_max: float = 25.0,
    frame_since_restart: int = 200_000_000,
    success_tolerance: float = 0.05,
    pbt_ranking_score: float = 100.0,
):
    return {
        "policy_idx": policy_idx,
        "frame": 300_000_000,
        "frames_at_restart": 0,
        "iteration": 10,
        "objective": pbt_ranking_score,
        "pbt_ranking_score": pbt_ranking_score,
        "nsucc_mean": nsucc_mean,
        "nsucc_max": nsucc_max,
        "checkpoint": "/tmp/dummy.pth",
        "curriculum": {
            "frame_since_restart": frame_since_restart,
            "mean_prev_episode_successes": nsucc_mean,
            "max_prev_episode_successes": nsucc_max,
            "success_tolerance": success_tolerance,
        },
        "params": {},
    }


class MasteryGateHelperTest(unittest.TestCase):
    def test_passes_after_consecutive_windows(self):
        w = RollingNsuccWindow()
        w.append(_sample(mean=4.5, max_=22.0))
        w.append(_sample(mean=4.2, max_=21.0))
        ok, count, reason = evaluate_mastery_gate(
            w, mastery_mean=4.0, mastery_max=20.0, consecutive_windows=2
        )
        self.assertTrue(ok)
        self.assertEqual(count, 2)
        self.assertEqual(reason, "ok")

    def test_blocks_on_low_mean(self):
        w = RollingNsuccWindow()
        w.append(_sample(mean=3.0, max_=25.0))
        w.append(_sample(mean=4.5, max_=25.0))
        ok, _, reason = evaluate_mastery_gate(
            w, mastery_mean=4.0, mastery_max=20.0, consecutive_windows=2
        )
        self.assertFalse(ok)
        self.assertEqual(reason, "mean_below_threshold")


class StableWindowHelperTest(unittest.TestCase):
    def test_stable_source_passes(self):
        w = RollingNsuccWindow()
        w.append(_sample(mean=3.5, max_=22.0))
        w.append(_sample(mean=3.2, max_=21.0))
        ok, reason = is_stable_source_window(
            w, min_mean=3.0, min_max=20.0, max_drop_ratio=0.3
        )
        self.assertTrue(ok, reason)

    def test_usable_live_requires_min_signal(self):
        w = RollingNsuccWindow()
        w.append(_sample(mean=1.5, max_=12.0))
        ok, reason = is_usable_live_window(w)
        self.assertTrue(ok, reason)

    def test_qualifies_best_stable_rejects_tolerance_change(self):
        w = RollingNsuccWindow()
        w.append(_sample(mean=3.5, max_=22.0, tol=0.05))
        w.append(_sample(mean=3.4, max_=21.0, tol=0.0475))
        ok, reason = qualifies_best_stable_save(
            w, min_mean=3.0, min_max=20.0, max_drop_ratio=0.3
        )
        self.assertFalse(ok)
        self.assertIn(reason, ("just_shrank", "tolerance_not_stable"))


class CollapseFreezeHelperTest(unittest.TestCase):
    def test_detects_sustained_drop(self):
        w = RollingNsuccWindow()
        w.append(_sample(mean=4.0, max_=30.0))
        w.append(_sample(mean=1.5, max_=10.0))
        w.append(_sample(mean=1.0, max_=8.0))
        freeze, drop, streak = evaluate_collapse_freeze(
            w, drop_ratio_threshold=0.5, patience=2, recent_peak=4.0
        )
        self.assertTrue(freeze)
        self.assertGreater(drop, 0.5)
        self.assertGreaterEqual(streak, 2)


class CurriculumControlIOTest(unittest.TestCase):
    def test_round_trip(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "000.json"
            state = CurriculumControlState(
                curriculum_mode="mastery",
                allow_shrink=True,
                mastery_gate_pass=True,
            )
            write_curriculum_control(path, state)
            loaded = read_curriculum_control(path)
            self.assertIsNotNone(loaded)
            assert loaded is not None
            self.assertTrue(loaded.allow_shrink)
            self.assertTrue(loaded.mastery_gate_pass)


class StableSourceTierTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.workspace = Path(self._tmp.name)

    def _seed_window(self, obs, policy_idx, samples):
        w = obs._policy_window(policy_idx)
        for s in samples:
            w.append(s)

    def test_l1_strict_stable_live(self):
        obs = _make_observer(self.workspace)
        self._seed_window(
            obs,
            0,
            [
                _sample(mean=3.5, max_=22.0),
                _sample(mean=3.2, max_=21.0),
            ],
        )
        population = {
            0: _state(policy_idx=0, nsucc_mean=3.2, nsucc_max=21.0),
            1: _state(policy_idx=1, nsucc_mean=0.0, nsucc_max=0.0),
            2: _state(policy_idx=2, nsucc_mean=0.0, nsucc_max=0.0),
        }
        d = obs._select_source(
            sorted_policies=[0, 1, 2],
            population=population,
            env_frames=300_000_000,
            n_top=1,
        )
        self.assertEqual(d["tier"], "strict_stable_live")
        self.assertEqual(d["checkpoint_type"], "latest_healthy")
        self.assertEqual(d["source_policy"], 0)

    def test_l2_usable_when_stable_fails(self):
        obs = _make_observer(self.workspace)
        self._seed_window(
            obs,
            1,
            [_sample(mean=1.5, max_=12.0)],
        )
        population = {
            0: _state(policy_idx=0, nsucc_mean=1.0, nsucc_max=5.0),
            1: _state(policy_idx=1, nsucc_mean=1.5, nsucc_max=12.0),
            2: _state(policy_idx=2, nsucc_mean=0.0, nsucc_max=0.0),
        }
        d = obs._select_source(
            sorted_policies=[0, 1, 2],
            population=population,
            env_frames=300_000_000,
            n_top=1,
        )
        self.assertEqual(d["tier"], "usable_live")
        self.assertEqual(d["source_policy"], 1)

    def test_legacy_cascade_when_stable_mode_off(self):
        obs = _make_observer(
            self.workspace, source_require_stable_mean=False
        )
        population = {
            0: _state(policy_idx=0, nsucc_mean=5.0, nsucc_max=25.0),
            1: _state(policy_idx=1, nsucc_mean=0.0, nsucc_max=0.0),
            2: _state(policy_idx=2, nsucc_mean=0.0, nsucc_max=0.0),
        }
        d = obs._select_source(
            sorted_policies=[0, 1, 2],
            population=population,
            env_frames=300_000_000,
            n_top=1,
        )
        self.assertEqual(d["tier"], "strict_live")
        self.assertEqual(d["checkpoint_type"], "live")


if __name__ == "__main__":
    unittest.main()
