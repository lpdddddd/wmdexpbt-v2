"""Patch C (rev 3, 2026-05-30): real CF child_train_frames via runner.algo.

Pins down the bug observed in regrasping_cbwm_20260530_160513 where every
CF block was marked invalid because the CF status writer in train.py
read `getattr(runner, "algo", None)` from a never-assigned attribute on
the custom `_WmEnabledRunner`, then reported `child_train_frames=0`
despite the child training for 600s with 306 samples.

Patch C fixes this with a three-tier fallback (`_compute_child_train_
frames`) and exposes the source of the count via `frame_estimation_
method`. These tests exercise the pure helper without bringing up
Isaac Sim.
"""

import unittest
from types import SimpleNamespace

# Patch C (rev 3, 2026-05-30): the helper now lives in a lightweight
# `dexpbt_isaaclab.cf_utils` module with no Isaac Sim / gymnasium
# imports, so the unit test does not need the Omniverse stack to run.
from dexpbt_isaaclab.cf_utils import compute_child_train_frames as _compute_child_train_frames


def _mock_runner(*, algo_frame=None, start_frame=0, end_frame=0):
    """Build a minimal duck-typed runner matching the helper's contract."""
    algo = None
    if algo_frame is not None:
        algo = SimpleNamespace(frame=int(algo_frame))
    return SimpleNamespace(
        algo=algo,
        _start_frame=int(start_frame),
        _end_frame=int(end_frame),
    )


def _mock_tracker(n_samples):
    return SimpleNamespace(samples=[0.0] * int(n_samples))


class ComputeChildTrainFramesTest(unittest.TestCase):
    """Three-tier fallback for the CF child frame count."""

    def test_algo_frame_path(self):
        """When `runner.algo.frame > 0`, that value wins."""
        runner = _mock_runner(algo_frame=10_000_000, start_frame=0, end_frame=10_000_000)
        tracker = _mock_tracker(300)
        frames, method = _compute_child_train_frames(
            runner, tracker, num_envs=4096, horizon_length=16
        )
        self.assertEqual(frames, 10_000_000)
        self.assertEqual(method, "algo_frame")

    def test_end_minus_start_when_algo_frame_zero(self):
        """If algo.frame is 0 but boundary is positive, use the boundary."""
        runner = _mock_runner(algo_frame=0, start_frame=2_000_000, end_frame=10_000_000)
        tracker = _mock_tracker(300)
        frames, method = _compute_child_train_frames(
            runner, tracker, num_envs=4096, horizon_length=16
        )
        self.assertEqual(frames, 8_000_000)
        self.assertEqual(method, "end_minus_start")

    def test_samples_estimate_when_no_frame_signal(self):
        """If both algo.frame and boundary are unusable, estimate from samples."""
        runner = _mock_runner(algo_frame=None, start_frame=0, end_frame=0)
        tracker = _mock_tracker(200)
        frames, method = _compute_child_train_frames(
            runner, tracker, num_envs=4096, horizon_length=16
        )
        # 200 samples * 4096 envs * 16 horizon = 13.1M
        self.assertEqual(frames, 200 * 4096 * 16)
        self.assertEqual(method, "samples_estimate")

    def test_zero_when_no_signal_at_all(self):
        """All three signals absent -> 0, method='zero'."""
        runner = _mock_runner(algo_frame=None, start_frame=0, end_frame=0)
        tracker = _mock_tracker(0)
        frames, method = _compute_child_train_frames(
            runner, tracker, num_envs=4096, horizon_length=16
        )
        self.assertEqual(frames, 0)
        self.assertEqual(method, "zero")

    def test_runner_without_algo_attribute(self):
        """Tolerate `runner.algo` missing entirely."""
        runner = SimpleNamespace(_start_frame=0, _end_frame=5_000_000)
        tracker = _mock_tracker(0)
        frames, method = _compute_child_train_frames(
            runner, tracker, num_envs=4096, horizon_length=16
        )
        self.assertEqual(frames, 5_000_000)
        self.assertEqual(method, "end_minus_start")

    def test_no_cf_tracker_falls_through_to_zero(self):
        """`cf_tracker=None` must not crash; just skip the samples branch."""
        runner = _mock_runner(algo_frame=None, start_frame=0, end_frame=0)
        frames, method = _compute_child_train_frames(
            runner, None, num_envs=4096, horizon_length=16
        )
        self.assertEqual(frames, 0)
        self.assertEqual(method, "zero")

    def test_realistic_600s_run_with_zero_algo_frame(self):
        """Reproduces the regrasping_cbwm_20260530_160513 collapse-era bug:

        - elapsed_s ~ 600s
        - algo.frame = 0 (because `runner.algo` was never assigned)
        - cf_tracker.samples = 306 (real PPO epochs observed)
        - num_envs = 4096, horizon = 16

        With Patch C the helper now returns a meaningful estimate via
        the samples path, instead of 0.
        """
        runner = SimpleNamespace(algo=None, _start_frame=0, _end_frame=0)
        tracker = _mock_tracker(306)
        frames, method = _compute_child_train_frames(
            runner, tracker, num_envs=4096, horizon_length=16
        )
        self.assertGreater(frames, 0)
        # samples_estimate must be positive enough to clear the new
        # `cf_invalid_frames_ratio=0.25` threshold against the 20M target.
        self.assertGreater(frames, 5_000_000)
        self.assertEqual(method, "samples_estimate")


# Even when train.py is not importable, we can still test the classify
# path (which lives in run_counterfactual_block.py, no Isaac deps).
from dexpbt_isaaclab.scripts.run_counterfactual_block import (
    _classify_child_result,
    DEFAULT_CF_MIN_EVAL_S,
    DEFAULT_CF_TRAIN_FRAMES,
)


def _status(**overrides) -> dict:
    base = {
        "r_meta": 0.5,
        "r_meta_before": 0.45,
        "r_meta_after": 0.5,
        "nsucc_before": 1.5,
        "nsucc_after": 2.0,
        "success_tolerance_before": 0.05,
        "success_tolerance_after": 0.05,
        "elapsed_s": 600.0,
        "child_train_frames": 15_000_000,
        "r_meta_samples": 300,
        "frame_estimation_method": "algo_frame",
    }
    base.update(overrides)
    return base


def _req(**overrides) -> dict:
    base = {
        "cf_min_eval_s": DEFAULT_CF_MIN_EVAL_S,
        "cf_train_frames": DEFAULT_CF_TRAIN_FRAMES,
        "source_curriculum": {
            "initial_tolerance": 0.075,
            "target_tolerance": 0.01,
        },
    }
    base.update(overrides)
    return base


class ClassifyInvalidReasonEnumTest(unittest.TestCase):
    """Patch C: `invalid_reason` is now a stable enum string."""

    def test_valid_when_elapsed_and_frames_ok(self):
        c = _classify_child_result(_status(), _req())
        self.assertFalse(c["invalid"])

    def test_invalid_only_when_elapsed_short(self):
        c = _classify_child_result(_status(elapsed_s=10.0), _req())
        self.assertTrue(c["invalid"])
        self.assertEqual(c["invalid_reason"], "elapsed_too_short")

    def test_no_samples_invalid(self):
        c = _classify_child_result(_status(r_meta_samples=0), _req())
        self.assertTrue(c["invalid"])
        self.assertEqual(c["invalid_reason"], "no_samples")

    def test_frame_count_missing_when_zero_with_samples(self):
        """elapsed_s ok, samples ok, but frame counter is 0 -> explicit
        `frame_count_missing` reason (distinct from `insufficient_frames`).
        """
        c = _classify_child_result(
            _status(child_train_frames=0, r_meta_samples=300),
            _req(),
        )
        self.assertTrue(c["invalid"])
        self.assertEqual(c["invalid_reason"], "frame_count_missing")

    def test_insufficient_frames_below_ratio(self):
        """child_train_frames > 0 but below ratio threshold."""
        # ratio default 0.25; target 20M -> threshold 5M. Use 2M.
        c = _classify_child_result(
            _status(child_train_frames=2_000_000),
            _req(),
        )
        self.assertTrue(c["invalid"])
        self.assertEqual(c["invalid_reason"], "insufficient_frames")

    def test_valid_when_frame_estimated_via_samples(self):
        """Reproduces the post-Patch-C target: 306 samples * 4096 * 16
        ~= 20M frames, which clears the 5M threshold.
        """
        c = _classify_child_result(
            _status(
                child_train_frames=306 * 4096 * 16,
                frame_estimation_method="samples_estimate",
            ),
            _req(),
        )
        self.assertFalse(c["invalid"])

    def test_ratio_is_configurable(self):
        """`cf_invalid_frames_ratio` can be tightened back to 0.5."""
        c = _classify_child_result(
            _status(child_train_frames=8_000_000),  # > 5M but < 10M
            _req(cf_invalid_frames_ratio=0.5),
        )
        self.assertTrue(c["invalid"])
        self.assertEqual(c["invalid_reason"], "insufficient_frames")

    def test_missing_r_meta_returns_missing_enum(self):
        s = _status()
        s.pop("r_meta_after")
        s.pop("r_meta")
        c = _classify_child_result(s, _req())
        self.assertTrue(c["invalid"])
        self.assertEqual(c["invalid_reason"], "missing_r_meta")

    def test_invalid_reason_is_enum_value(self):
        """All invalid_reason values must be from the documented enum."""
        enum = {
            "missing_r_meta",
            "elapsed_too_short",
            "no_samples",
            "frame_count_missing",
            "insufficient_frames",
        }
        scenarios = [
            (_status(elapsed_s=10.0), "elapsed_too_short"),
            (_status(r_meta_samples=0), "no_samples"),
            (_status(child_train_frames=0, r_meta_samples=300),
             "frame_count_missing"),
            (_status(child_train_frames=1_000_000),
             "insufficient_frames"),
        ]
        for status, expected in scenarios:
            c = _classify_child_result(status, _req())
            self.assertIn(c["invalid_reason"], enum)
            self.assertEqual(c["invalid_reason"], expected)


if __name__ == "__main__":
    unittest.main()
