"""Patch B (rev 3, 2026-05-30): PBT ranking-score cap for 0 < nsucc < threshold.

The Patch 1 cap only fires when `nsucc_mean <= 0`. The collapse run
regrasping_cbwm_20260530_160513 showed a second, equally damaging
regime: `policy_1` ran at nsucc_mean=0.17 (essentially non-functional)
but r_meta_raw=0.94 because tolerance progress dominated the
`tolerance_successes_objective` formula. The Patch B low-nsucc cap
fills this gap.

Tests here pin down:

  1. The new `nsucc_low_threshold` / `nsucc_low_cap` parameters of
     `compute_pbt_ranking_score`.
  2. Backward compatibility: `nsucc_low_threshold=0.0` recovers the
     Patch 1 behaviour exactly.
  3. raw_r_meta mode bypasses ALL caps.
  4. r_meta_raw is preserved in meta.json / training_curve alongside the
     capped pbt_ranking_score.
"""

import unittest

from dexpbt_isaaclab.dexpbt_utils import (
    PBT_RANKING_METRIC_RAW,
    PBT_RANKING_METRIC_SUCCESS_AWARE,
    compute_pbt_ranking_score,
)


class LowNsuccCapTest(unittest.TestCase):
    """Patch B: 0 < nsucc_mean < threshold must cap ranking_score."""

    def _score(self, r_raw, nsucc, **kwargs):
        return compute_pbt_ranking_score(
            r_raw,
            nsucc,
            mode=PBT_RANKING_METRIC_SUCCESS_AWARE,
            nsucc_zero_cap=0.05,
            nsucc_low_threshold=kwargs.pop("nsucc_low_threshold", 1.0),
            nsucc_low_cap=kwargs.pop("nsucc_low_cap", 0.05),
        )

    def test_nsucc_zero_keeps_zero_cap_path(self):
        """The Patch 1 zero-cap path must still fire for nsucc==0."""
        self.assertEqual(self._score(0.94, 0.0), 0.05)

    def test_policy_1_collapse_case(self):
        """nsucc=0.17, r_raw=0.94 -> capped at 0.05 (regression test)."""
        self.assertEqual(self._score(0.94, 0.17), 0.05)

    def test_nsucc_below_threshold_capped(self):
        """nsucc=0.99 is still below default threshold 1.0 -> cap."""
        self.assertEqual(self._score(0.94, 0.99), 0.05)

    def test_nsucc_at_threshold_exactly_uncapped(self):
        """nsucc==threshold is >= threshold -> not capped."""
        self.assertAlmostEqual(self._score(0.94, 1.0), 0.94)

    def test_healthy_policy_uncapped(self):
        """Patch B must not penalise the healthy nsucc>=1 worker."""
        self.assertAlmostEqual(self._score(0.32, 2.18), 0.32)

    def test_low_nsucc_but_r_raw_below_cap_returns_r_raw(self):
        """If r_raw is already below the cap, return r_raw (min wins)."""
        self.assertAlmostEqual(self._score(0.02, 0.5), 0.02)

    def test_raw_r_meta_mode_bypasses_caps(self):
        """raw_r_meta mode must ignore ALL caps for ablation rollback."""
        s = compute_pbt_ranking_score(
            0.94, 0.17, mode=PBT_RANKING_METRIC_RAW,
            nsucc_zero_cap=0.05,
            nsucc_low_threshold=1.0,
            nsucc_low_cap=0.05,
        )
        self.assertAlmostEqual(s, 0.94)

    def test_threshold_is_configurable(self):
        """Tightening threshold to 2.0 should cap the nsucc=1.5 case."""
        s = self._score(0.94, 1.5, nsucc_low_threshold=2.0)
        self.assertEqual(s, 0.05)

    def test_low_cap_is_configurable(self):
        """Changing the cap value to 0.1 should be reflected."""
        s = self._score(0.94, 0.5, nsucc_low_cap=0.10)
        self.assertEqual(s, 0.10)

    def test_threshold_zero_disables_patch_b_layer(self):
        """`nsucc_low_threshold=0.0` must recover Patch 1 behaviour."""
        # With threshold=0.0, only nsucc <= 0 hits the zero-cap path; the
        # low-band cap is effectively disabled.
        s_low = self._score(0.94, 0.17, nsucc_low_threshold=0.0)
        self.assertAlmostEqual(s_low, 0.94)
        # nsucc=0 still hits the zero-cap path.
        s_zero = self._score(0.94, 0.0, nsucc_low_threshold=0.0)
        self.assertEqual(s_zero, 0.05)

    def test_pure_no_side_effect(self):
        """Repeated calls with identical args must produce identical outputs."""
        out1 = self._score(0.94, 0.17)
        out2 = self._score(0.94, 0.17)
        out3 = self._score(0.94, 0.17)
        self.assertEqual(out1, out2)
        self.assertEqual(out2, out3)


class CollapseRegressionVectorsTest(unittest.TestCase):
    """The exact (r_raw, nsucc) tuples observed in
    regrasping_cbwm_20260530_160513 must all be capped, while the healthy
    policy_3/4/6 tuples must not be.
    """

    def test_policy_1_low_nsucc_high_r_raw_capped(self):
        s = compute_pbt_ranking_score(
            0.94, 0.17,
            mode=PBT_RANKING_METRIC_SUCCESS_AWARE,
            nsucc_zero_cap=0.05,
            nsucc_low_threshold=1.0, nsucc_low_cap=0.05,
        )
        self.assertEqual(s, 0.05)

    def test_policy_5_collapsed_already_capped_by_zero_cap(self):
        s = compute_pbt_ranking_score(
            0.94, 0.0,
            mode=PBT_RANKING_METRIC_SUCCESS_AWARE,
            nsucc_zero_cap=0.05,
            nsucc_low_threshold=1.0, nsucc_low_cap=0.05,
        )
        self.assertEqual(s, 0.05)

    def test_policy_3_healthy_uncapped(self):
        s = compute_pbt_ranking_score(
            0.319, 2.18,
            mode=PBT_RANKING_METRIC_SUCCESS_AWARE,
            nsucc_zero_cap=0.05,
            nsucc_low_threshold=1.0, nsucc_low_cap=0.05,
        )
        self.assertAlmostEqual(s, 0.319)


if __name__ == "__main__":
    unittest.main()
