"""Patch 3 (rev 2, 2026-05-30): CF child invalid/low_signal classification.

Pins down the failure modes observed in regrasping_cbwm_20260530_090008
at CF block 85 -- five children all reported r_meta=0.541 after running
for ~3 seconds, because the source checkpoint's `frame=1.7B` exceeded the
local `max_iterations=77` and the child exited immediately with
`MAX EPOCHS NUM!`.

These tests verify the classification function on synthetic status dicts,
without launching subprocesses.
"""

import unittest

from dexpbt_isaaclab.scripts.run_counterfactual_block import (
    _classify_child_result,
    DEFAULT_CF_MIN_EVAL_S,
    DEFAULT_CF_TRAIN_FRAMES,
)


def _request(**overrides) -> dict:
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


def _good_status(**overrides) -> dict:
    s = {
        "r_meta": 0.7,
        "r_meta_before": 0.5,
        "r_meta_after": 0.7,
        "nsucc_before": 1.5,
        "nsucc_after": 2.5,
        "success_tolerance_before": 0.05,
        "success_tolerance_after": 0.045,
        "elapsed_s": 120.0,
        "child_train_frames": DEFAULT_CF_TRAIN_FRAMES,
        "r_meta_samples": 8,
    }
    s.update(overrides)
    return s


class InvalidClassificationTest(unittest.TestCase):
    def test_good_status_is_neither_invalid_nor_low_signal(self):
        c = _classify_child_result(_good_status(), _request())
        self.assertFalse(c["invalid"])
        self.assertFalse(c["low_signal"])

    def test_short_elapsed_marks_invalid(self):
        # Reproduces CF block 85: elapsed_s ~= 3 with samples=1.
        # Patch C (rev 3): invalid_reason is now the enum value
        # "elapsed_too_short".
        c = _classify_child_result(
            _good_status(elapsed_s=3.2, r_meta_samples=1, child_train_frames=0),
            _request(),
        )
        self.assertTrue(c["invalid"])
        self.assertEqual(c["invalid_reason"], "elapsed_too_short")

    def test_low_train_frames_marks_invalid(self):
        # Patch C (rev 3): threshold relaxed from 0.5 to 0.25; pass a
        # frame count below the new threshold (0.25 * 20M = 5M; use 1M).
        c = _classify_child_result(
            _good_status(child_train_frames=1_000_000),
            _request(),
        )
        self.assertTrue(c["invalid"])
        self.assertEqual(c["invalid_reason"], "insufficient_frames")

    def test_few_samples_marks_invalid(self):
        # Patch C (rev 3): invalid_reason enum is "no_samples".
        c = _classify_child_result(
            _good_status(r_meta_samples=1),
            _request(),
        )
        self.assertTrue(c["invalid"])
        self.assertEqual(c["invalid_reason"], "no_samples")

    def test_missing_r_meta_marks_invalid(self):
        s = _good_status()
        s.pop("r_meta_after")
        s.pop("r_meta")
        c = _classify_child_result(s, _request())
        self.assertTrue(c["invalid"])
        self.assertEqual(c["invalid_reason"], "missing_r_meta")

    def test_cf_min_eval_s_is_configurable(self):
        # Lower the threshold; what used to be invalid is now valid.
        c = _classify_child_result(
            _good_status(elapsed_s=5.0),
            _request(cf_min_eval_s=2.0),
        )
        self.assertFalse(c["invalid"])


class LowSignalClassificationTest(unittest.TestCase):
    def test_zero_nsucc_with_only_tolerance_change_is_low_signal(self):
        # Source eps=0.05 -> child eps=0.04. nsucc stays 0 both before
        # and after. r_meta moves but only because tolerance shrunk.
        status = _good_status(
            r_meta_before=0.30,
            r_meta_after=0.55,
            nsucc_before=0.0,
            nsucc_after=0.0,
            success_tolerance_before=0.05,
            success_tolerance_after=0.04,
        )
        c = _classify_child_result(status, _request())
        self.assertFalse(c["invalid"])
        self.assertTrue(c["low_signal"])
        self.assertIn("success_component_unchanged", str(c["low_signal_reason"]))

    def test_zero_nsucc_with_real_success_change_not_low_signal(self):
        # If success component moves (e.g. nsucc was 0 then briefly hit
        # something non-zero), do NOT mark low_signal.
        status = _good_status(
            nsucc_before=0.0,
            nsucc_after=0.001,
            r_meta_before=0.30,
            r_meta_after=0.31,
            success_tolerance_before=0.05,
            success_tolerance_after=0.05,
        )
        c = _classify_child_result(status, _request())
        self.assertFalse(c["low_signal"])

    def test_positive_nsucc_bypasses_low_signal_check(self):
        c = _classify_child_result(_good_status(), _request())
        self.assertFalse(c["low_signal"])

    def test_invalid_child_is_not_also_flagged_low_signal(self):
        c = _classify_child_result(
            _good_status(elapsed_s=2.0, r_meta_samples=1),
            _request(),
        )
        self.assertTrue(c["invalid"])
        # low_signal short-circuited once invalid is set.
        self.assertFalse(c["low_signal"])


class RunCfBlockDispatchTest(unittest.TestCase):
    """Verify that _classify_child_result correctly drives the result dict."""

    def test_classification_is_attached_to_status_dict(self):
        status = _good_status()
        c = _classify_child_result(status, _request())
        # Original keys preserved.
        for key in ("r_meta", "r_meta_before", "r_meta_after", "elapsed_s"):
            self.assertIn(key, c)
        # New keys added.
        self.assertIn("invalid", c)
        self.assertIn("invalid_reason", c)
        self.assertIn("low_signal", c)
        self.assertIn("low_signal_reason", c)


if __name__ == "__main__":
    unittest.main()
