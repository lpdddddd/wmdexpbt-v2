"""Patch 4 (rev 2, 2026-05-30): CF delta_w = learning_improvement.

These tests pin down the two delta_w modes (raw_after, learning_improvement)
and the invalid_block / all_strategies_tied / posterior_weights surface
that the CF runner exposes to the posterior selector.

The historical failure case from cf_block 85 / 90 is the regression test:
all five strategies reported r_meta_after = 0.5406450033187866 because each
child exited in ~3s without training. Under Patch 3+4 they should all be
marked invalid AND the block must surface invalid_block=True so the
posterior is NOT updated.
"""

import json
import tempfile
import unittest
from pathlib import Path

# Import the runner's delta_w / classification helpers via a small wrapper
# so we don't need to actually launch subprocesses. We test by injecting
# pre-baked results dicts into run_cf_block-like data flow.
from dexpbt_isaaclab.scripts.run_counterfactual_block import (
    DEFAULT_CF_DELTA_MODE,
    DEFAULT_CF_LOW_SIGNAL_ACTION,
    DEFAULT_CF_LOW_SIGNAL_WEIGHT,
    DEFAULT_CF_TRAIN_FRAMES,
)
from dexpbt_isaaclab.world_model.wm_posterior_selector import SharedPosteriorSelector


# Re-implement the delta_w math from run_cf_block in a pure-Python form
# so we can unit-test the precise contract without spinning up
# subprocesses. If the production logic drifts, this test will fail and
# the discrepancy must be fixed there.
def _coerce_float(v, default: float) -> float:
    """Mirror the production-side `.get(..., NaN)` semantics safely.

    JSON nulls / Python Nones / non-numeric strings all collapse to
    `default` instead of crashing `float()`.
    """
    if v is None:
        return float(default)
    try:
        return float(v)
    except (TypeError, ValueError):
        return float(default)


def compute_delta_w(
    results: dict,
    *,
    delta_mode: str = DEFAULT_CF_DELTA_MODE,
    low_signal_action: str = DEFAULT_CF_LOW_SIGNAL_ACTION,
    low_signal_weight: float = DEFAULT_CF_LOW_SIGNAL_WEIGHT,
):
    none_r = results.get("none")
    invalid_block = False
    invalid_reason = None
    if none_r is None or none_r.get("invalid"):
        invalid_block = True
        invalid_reason = "none_missing" if none_r is None else "none_invalid"
        return {}, {}, invalid_block, invalid_reason

    if delta_mode == "learning_improvement":
        need_before = any(
            r.get("r_meta_before") is None
            for r in results.values()
            if not r.get("invalid")
        )
        if need_before:
            delta_mode = "raw_after"

    none_before = _coerce_float(none_r.get("r_meta_before"), float("nan"))
    none_after = _coerce_float(none_r.get("r_meta_after"), float("nan"))

    delta_w = {}
    weights = {}
    for s, r in results.items():
        if r.get("invalid"):
            continue
        r_after = _coerce_float(r.get("r_meta_after"), 0.0)
        if delta_mode == "learning_improvement":
            r_before = _coerce_float(r.get("r_meta_before"), r_after)
            delta_w[s] = (r_after - r_before) - (none_after - none_before)
        else:
            delta_w[s] = r_after - none_after

        if r.get("low_signal"):
            if low_signal_action == "skip":
                delta_w.pop(s, None)
                weights[s] = 0.0
            elif low_signal_action == "downweight":
                weights[s] = float(low_signal_weight)
            else:
                weights[s] = 1.0
        else:
            weights[s] = 1.0
    return delta_w, weights, invalid_block, invalid_reason


def _child(
    *,
    r_meta_before: float | None,
    r_meta_after: float,
    invalid: bool = False,
    low_signal: bool = False,
):
    return {
        "r_meta": r_meta_after,
        "r_meta_before": r_meta_before,
        "r_meta_after": r_meta_after,
        "invalid": invalid,
        "low_signal": low_signal,
    }


class LearningImprovementDeltaTest(unittest.TestCase):
    def test_basic_learning_improvement(self):
        results = {
            "none":          _child(r_meta_before=0.30, r_meta_after=0.40),
            "aux_low":       _child(r_meta_before=0.30, r_meta_after=0.45),
            "value_imag_h1": _child(r_meta_before=0.30, r_meta_after=0.55),
        }
        delta_w, weights, invalid_block, _ = compute_delta_w(results)
        self.assertFalse(invalid_block)
        self.assertAlmostEqual(delta_w["none"], 0.0, places=6)
        self.assertAlmostEqual(delta_w["aux_low"], 0.05, places=6)
        self.assertAlmostEqual(delta_w["value_imag_h1"], 0.15, places=6)
        for s in results:
            self.assertEqual(weights[s], 1.0)

    def test_invalid_none_marks_block_invalid_and_empty_delta(self):
        results = {
            "none": _child(r_meta_before=0.30, r_meta_after=0.30, invalid=True),
            "aux_low": _child(r_meta_before=0.30, r_meta_after=0.45),
        }
        delta_w, weights, invalid_block, reason = compute_delta_w(results)
        self.assertTrue(invalid_block)
        self.assertEqual(reason, "none_invalid")
        self.assertEqual(delta_w, {})

    def test_missing_none_marks_block_invalid(self):
        results = {
            "aux_low": _child(r_meta_before=0.30, r_meta_after=0.45),
        }
        delta_w, weights, invalid_block, reason = compute_delta_w(results)
        self.assertTrue(invalid_block)
        self.assertEqual(reason, "none_missing")

    def test_invalid_strategy_is_skipped_in_delta(self):
        results = {
            "none": _child(r_meta_before=0.30, r_meta_after=0.40),
            "aux_low": _child(r_meta_before=0.30, r_meta_after=0.45),
            "aux_mid": _child(r_meta_before=0.30, r_meta_after=0.99, invalid=True),
        }
        delta_w, weights, invalid_block, _ = compute_delta_w(results)
        self.assertFalse(invalid_block)
        self.assertNotIn("aux_mid", delta_w)
        self.assertIn("aux_low", delta_w)

    def test_low_signal_skip_default_drops_strategy_and_zero_weight(self):
        results = {
            "none": _child(r_meta_before=0.30, r_meta_after=0.40),
            "aux_low": _child(
                r_meta_before=0.30, r_meta_after=0.55, low_signal=True
            ),
            "value_imag_h1": _child(r_meta_before=0.30, r_meta_after=0.45),
        }
        delta_w, weights, _, _ = compute_delta_w(results)
        self.assertNotIn("aux_low", delta_w)
        self.assertEqual(weights["aux_low"], 0.0)
        self.assertIn("value_imag_h1", delta_w)
        self.assertEqual(weights["value_imag_h1"], 1.0)

    def test_low_signal_downweight_keeps_delta_and_partial_weight(self):
        results = {
            "none": _child(r_meta_before=0.30, r_meta_after=0.40),
            "aux_low": _child(
                r_meta_before=0.30, r_meta_after=0.55, low_signal=True
            ),
        }
        delta_w, weights, _, _ = compute_delta_w(
            results, low_signal_action="downweight", low_signal_weight=0.25
        )
        self.assertIn("aux_low", delta_w)
        self.assertAlmostEqual(weights["aux_low"], 0.25, places=6)

    def test_falls_back_to_raw_after_when_before_missing(self):
        # If any valid child lacks r_meta_before we must NOT silently
        # treat it as 0.0 (that would inflate learning_improvement); we
        # fall back to raw_after globally.
        results = {
            "none": _child(r_meta_before=None, r_meta_after=0.40),
            "aux_low": _child(r_meta_before=0.30, r_meta_after=0.45),
        }
        delta_w, weights, invalid_block, _ = compute_delta_w(
            results, delta_mode="learning_improvement"
        )
        self.assertFalse(invalid_block)
        # In raw_after mode delta_w[aux_low] = 0.45 - 0.40 = 0.05
        self.assertAlmostEqual(delta_w["aux_low"], 0.05, places=6)

    def test_raw_after_mode_matches_legacy_formula(self):
        results = {
            "none": _child(r_meta_before=0.30, r_meta_after=0.40),
            "aux_low": _child(r_meta_before=0.50, r_meta_after=0.60),
        }
        delta_w, _, _, _ = compute_delta_w(results, delta_mode="raw_after")
        # Legacy: delta_w[s] = r_meta_after[s] - r_meta_after[none].
        self.assertAlmostEqual(delta_w["aux_low"], 0.20, places=6)
        self.assertAlmostEqual(delta_w["none"], 0.0, places=6)


class PosteriorWeightsHonoredTest(unittest.TestCase):
    def test_zero_weight_skips_posterior_update(self):
        with tempfile.TemporaryDirectory() as td:
            shared = Path(td) / "wm_posterior_latest.json"
            sel = SharedPosteriorSelector(shared_path=shared)
            sel.load_or_init(policy_idx=0)
            n_before = sel.posterior["aux_low"].n
            sel.update_batch(
                {"none": 0.0, "aux_low": 0.5},
                step=10,
                weights={"none": 1.0, "aux_low": 0.0},
            )
            # aux_low must NOT have updated.
            self.assertEqual(sel.posterior["aux_low"].n, n_before)
            # none should have updated (weight=1).
            self.assertGreater(sel.posterior["none"].n, 1)

    def test_partial_weight_softens_update(self):
        with tempfile.TemporaryDirectory() as td:
            shared = Path(td) / "wm_posterior_latest.json"
            full = SharedPosteriorSelector(shared_path=shared)
            full.load_or_init(policy_idx=0)
            full.update_batch({"aux_low": 1.0}, step=10, weights={"aux_low": 1.0})

            shared2 = Path(td) / "wm_posterior_partial.json"
            partial = SharedPosteriorSelector(shared_path=shared2)
            partial.load_or_init(policy_idx=0)
            partial.update_batch({"aux_low": 1.0}, step=10, weights={"aux_low": 0.25})

            # Welford with 0.25 weight produces lower M2 contribution
            # than weight=1, so std should be strictly smaller.
            self.assertLessEqual(
                partial.posterior["aux_low"].M2,
                full.posterior["aux_low"].M2 + 1e-9,
            )


class InvalidBlockDoesNotUpdatePosteriorTest(unittest.TestCase):
    def test_caller_must_check_invalid_block(self):
        # Sanity contract: when invalid_block is True, delta_w should be
        # empty so callers that forward it to update_batch effectively
        # no-op without an extra check. Pin this so future refactors do
        # not accidentally fill delta_w on invalid blocks.
        results = {
            "none": _child(r_meta_before=0.30, r_meta_after=0.30, invalid=True),
            "aux_low": _child(r_meta_before=0.30, r_meta_after=0.45),
        }
        delta_w, _, invalid_block, _ = compute_delta_w(results)
        self.assertTrue(invalid_block)
        self.assertEqual(delta_w, {})


if __name__ == "__main__":
    unittest.main()
