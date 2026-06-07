"""Patch 1 (rev 2, 2026-05-30): PBT success-aware ranking score.

These tests pin down the semantics introduced by
`dexpbt_utils.compute_pbt_ranking_score` and the matching reporting split
(`r_meta_raw` vs `pbt_ranking_score`). They guard against the failure mode
observed in regrasping_cbwm_20260530_090008 at 1.9B frames: when curriculum
tightens success_tolerance, the original DexPBT meta-objective gives
r_meta=0.541 even with nsucc=0, which previously misled PBT tier
selection into preserving collapsed workers.
"""

import math
import unittest

from dexpbt_isaaclab.dexpbt_utils import (
    PBT_RANKING_METRIC_RAW,
    PBT_RANKING_METRIC_SUCCESS_AWARE,
    compute_pbt_ranking_score,
    split_r_meta_components,
    tolerance_successes_objective,
)
import torch


class SplitRMetaComponentsTest(unittest.TestCase):
    """split_r_meta_components must mirror tolerance_successes_objective."""

    def _legacy(self, eps, init, target, mean_nsucc):
        out = tolerance_successes_objective(
            eps, init, target, torch.tensor([float(mean_nsucc)], dtype=torch.float32)
        )
        return float(out[0].item())

    def test_split_matches_legacy_during_curriculum(self):
        for eps in (0.075, 0.060, 0.040, 0.020):
            for nsucc in (0.0, 1.5, 3.0):
                raw, tol_c, suc_c = split_r_meta_components(
                    eps, 0.075, 0.01, nsucc
                )
                self.assertAlmostEqual(raw, tol_c + suc_c, places=6)
                self.assertAlmostEqual(
                    raw, self._legacy(eps, 0.075, 0.01, nsucc), places=6
                )

    def test_split_matches_legacy_after_curriculum(self):
        # At/below target tolerance the success term is no longer scaled by 0.01.
        raw, tol_c, suc_c = split_r_meta_components(0.01, 0.075, 0.01, 4.0)
        self.assertAlmostEqual(raw, tol_c + suc_c, places=6)
        self.assertAlmostEqual(
            raw, self._legacy(0.01, 0.075, 0.01, 4.0), places=6
        )
        # Tolerance progress should be saturated at 1.0.
        self.assertAlmostEqual(tol_c, 1.0, places=6)

    def test_split_handles_degenerate_initial_eq_target(self):
        raw, tol_c, suc_c = split_r_meta_components(0.05, 0.05, 0.05, 2.0)
        # interp_0_1 collapses to 1.0 when initial==target.
        self.assertAlmostEqual(tol_c, 1.0, places=6)
        # eps==target so success term is not scaled.
        self.assertAlmostEqual(suc_c, 2.0, places=6)
        self.assertAlmostEqual(raw, 3.0, places=6)


class ComputePbtRankingScoreTest(unittest.TestCase):
    """Core ranking-score behaviour expected after Patch 1 (rev 2)."""

    def test_raw_mode_returns_input_unchanged(self):
        for raw in (-1.0, 0.0, 0.05, 0.541, 1.0, 2.5):
            for nsucc in (0.0, 0.5, 3.0):
                self.assertAlmostEqual(
                    compute_pbt_ranking_score(
                        raw, nsucc, mode=PBT_RANKING_METRIC_RAW
                    ),
                    raw,
                    places=12,
                )

    def test_success_aware_caps_score_when_nsucc_zero(self):
        # Reproduces the 1.9B-frame collapse: r_meta_raw=0.541, nsucc=0
        # should rank below a fresh worker with low r_meta but real successes.
        score = compute_pbt_ranking_score(
            0.5406450033187866, 0.0,
            mode=PBT_RANKING_METRIC_SUCCESS_AWARE,
            nsucc_zero_cap=0.05,
        )
        self.assertAlmostEqual(score, 0.05, places=12)

    def test_success_aware_does_not_inflate_when_raw_below_cap(self):
        # A worker that is already worse than the cap should not be lifted.
        score = compute_pbt_ranking_score(
            0.01, 0.0,
            mode=PBT_RANKING_METRIC_SUCCESS_AWARE,
            nsucc_zero_cap=0.05,
        )
        self.assertAlmostEqual(score, 0.01, places=12)

    def test_success_aware_keeps_raw_when_nsucc_positive(self):
        score = compute_pbt_ranking_score(
            0.2, 1.5,
            mode=PBT_RANKING_METRIC_SUCCESS_AWARE,
            nsucc_zero_cap=0.05,
        )
        self.assertAlmostEqual(score, 0.2, places=12)

    def test_success_aware_nsucc_zero_cap_is_configurable(self):
        score = compute_pbt_ranking_score(
            0.541, 0.0,
            mode=PBT_RANKING_METRIC_SUCCESS_AWARE,
            nsucc_zero_cap=0.20,
        )
        self.assertAlmostEqual(score, 0.20, places=12)

    def test_unknown_mode_raises(self):
        with self.assertRaises(ValueError):
            compute_pbt_ranking_score(0.5, 1.0, mode="not_a_mode")

    def test_strict_negative_nsucc_treated_as_zero_signal(self):
        # nsucc shouldn't be negative, but if it ever is we still cap.
        score = compute_pbt_ranking_score(
            0.6, -0.1,
            mode=PBT_RANKING_METRIC_SUCCESS_AWARE,
            nsucc_zero_cap=0.05,
        )
        self.assertAlmostEqual(score, 0.05, places=12)


class RankingPreservesOrderingTest(unittest.TestCase):
    """PBT sort behaviour: success-aware mode demotes nsucc=0 workers."""

    def _rank(self, rows, mode, cap=0.05):
        scored = [
            (idx, compute_pbt_ranking_score(raw, nsucc, mode=mode, nsucc_zero_cap=cap))
            for idx, raw, nsucc in rows
        ]
        scored.sort(key=lambda kv: kv[1], reverse=True)
        return [idx for idx, _ in scored]

    def test_raw_mode_lets_tolerance_dominated_worker_top(self):
        # Three workers: two collapsed (nsucc=0 but r_meta=0.541), one healthy
        # (nsucc=2, r_meta=0.30). Under raw_r_meta, collapsed workers win.
        rows = [
            (0, 0.541, 0.0),
            (1, 0.541, 0.0),
            (2, 0.30, 2.0),
        ]
        order = self._rank(rows, mode=PBT_RANKING_METRIC_RAW)
        self.assertEqual(order[0], 0)
        self.assertEqual(order[-1], 2)

    def test_success_aware_mode_promotes_healthy_worker(self):
        rows = [
            (0, 0.541, 0.0),
            (1, 0.541, 0.0),
            (2, 0.30, 2.0),
        ]
        order = self._rank(rows, mode=PBT_RANKING_METRIC_SUCCESS_AWARE)
        # Healthy worker must outrank both collapsed peers regardless of
        # their tolerance progress.
        self.assertEqual(order[0], 2)
        # Tie among collapsed workers is acceptable but they must come last.
        self.assertIn(order[-1], (0, 1))


if __name__ == "__main__":
    unittest.main()
