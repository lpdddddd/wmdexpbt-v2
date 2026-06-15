"""Tests for WM imagined-head validity helpers and evaluator modes."""

from __future__ import annotations

import unittest

from dexpbt_isaaclab.world_model.wm_activation_gate import (
    compute_wm_pbt_source_activation_weight,
    resolve_wm_ranking_uncertainty_gate,
)
from dexpbt_isaaclab.world_model.wm_imagined_evaluator import WMImaginedEvaluator
from dexpbt_isaaclab.world_model.wm_imagined_validity import (
    compute_head_has_signal,
    compute_imagined_validity,
    compute_wm_pbt_validity_weight,
)


class TestImaginedValidityHelpers(unittest.TestCase):
    def test_soft_valid_requires_head_signal(self):
        v = compute_imagined_validity(
            uncertainty=0.9,
            logit_abs_mean=0.01,
            success_prob_std=0.005,
            valid_mode="soft_calibrated",
            soft_uncertainty_threshold=0.95,
            min_logit_abs_mean=0.05,
            min_head_signal_std=0.02,
        )
        self.assertFalse(v.imagined_valid_soft)
        self.assertFalse(v.imagined_head_has_signal)
        self.assertEqual(v.wm_validity_weight, 0.0)

    def test_weak_but_nonzero_signal_soft_valid(self):
        v = compute_imagined_validity(
            uncertainty=0.9,
            logit_abs_mean=0.06,
            success_prob_std=0.01,
            valid_mode="soft_calibrated",
            soft_uncertainty_threshold=0.95,
            min_logit_abs_mean=0.05,
            soft_validity_weight=0.25,
        )
        self.assertTrue(v.imagined_valid_soft)
        self.assertTrue(v.imagined_head_has_signal)
        self.assertTrue(v.imagined_valid)
        self.assertAlmostEqual(v.wm_validity_weight, 0.25)

    def test_validity_weight_strict_soft_none(self):
        strict = compute_imagined_validity(
            uncertainty=0.3,
            logit_abs_mean=0.1,
            success_prob_std=0.05,
            valid_mode="soft_calibrated",
        )
        w, used_strict, used_soft = compute_wm_pbt_validity_weight(strict)
        self.assertEqual(w, 1.0)
        self.assertTrue(used_strict)
        self.assertFalse(used_soft)

        soft = compute_imagined_validity(
            uncertainty=0.9,
            logit_abs_mean=0.08,
            success_prob_std=0.03,
            valid_mode="soft_calibrated",
            soft_validity_weight=0.25,
        )
        w2, us2, uf2 = compute_wm_pbt_validity_weight(soft)
        self.assertAlmostEqual(w2, 0.25)
        self.assertFalse(us2)
        self.assertTrue(uf2)

        none = compute_imagined_validity(
            uncertainty=0.99,
            logit_abs_mean=0.0,
            valid_mode="soft_calibrated",
        )
        w3, us3, uf3 = compute_wm_pbt_validity_weight(none)
        self.assertEqual(w3, 0.0)
        self.assertFalse(us3)
        self.assertFalse(uf3)

    def test_shadow_only_never_controls_ranking(self):
        v = compute_imagined_validity(
            uncertainty=0.1,
            logit_abs_mean=0.2,
            success_prob_std=0.1,
            valid_mode="shadow_only",
        )
        self.assertFalse(v.imagined_valid)
        self.assertEqual(v.wm_validity_weight, 0.0)
        w, *_ = compute_wm_pbt_source_activation_weight(
            enable=True,
            gate_mode="confidence_only",
            frames=500_000_000,
            start_after_frames=300_000_000,
            ramp_frames=300_000_000,
            wm_uncertainty=0.1,
            max_uncertainty=0.5,
            imagined_valid_strict=False,
            imagined_valid_soft=False,
            imagined_head_has_signal=True,
            wm_validity_weight=0.0,
        )
        self.assertEqual(w, 0.0)

    def test_head_has_signal_or_logic(self):
        self.assertTrue(
            compute_head_has_signal(
                logit_abs_mean=0.06,
                success_prob_std=0.0,
                drop_risk_std=0.0,
                recovery_prob_std=0.0,
                min_logit_abs_mean=0.05,
                min_head_signal_std=0.02,
            )
        )
        self.assertTrue(
            compute_head_has_signal(
                logit_abs_mean=0.01,
                success_prob_std=0.03,
                drop_risk_std=0.0,
                recovery_prob_std=0.0,
                min_logit_abs_mean=0.05,
                min_head_signal_std=0.02,
            )
        )


class TestWMImaginedEvaluatorModes(unittest.TestCase):
    def test_disabled_returns_invalid(self):
        ev = WMImaginedEvaluator(enabled=False)
        self.assertEqual(ev.evaluate_batch.__name__, "evaluate_batch")

    def test_strict_mode_defaults(self):
        ev = WMImaginedEvaluator(
            enabled=True,
            valid_mode="strict_uncertainty",
            strict_uncertainty_threshold=0.5,
        )
        self.assertEqual(ev.valid_mode, "strict_uncertainty")
        self.assertAlmostEqual(ev.soft_validity_weight, 0.25)


class TestSoftValidWMRankingUncertaintyGate(unittest.TestCase):
    def test_soft_valid_uses_soft_uncertainty_threshold_for_wm_ranking(self):
        threshold, mode, fallback = resolve_wm_ranking_uncertainty_gate(
            wm_strict=False,
            wm_soft=True,
            wm_unc=0.80,
            strict_threshold=0.5,
            soft_threshold=0.95,
        )
        self.assertIsNone(fallback)
        self.assertAlmostEqual(threshold, 0.95)
        self.assertEqual(mode, "soft")
        w, _t, _u, _v, _us, used_soft = compute_wm_pbt_source_activation_weight(
            enable=True,
            gate_mode="confidence_only",
            frames=500_000_000,
            start_after_frames=300_000_000,
            ramp_frames=300_000_000,
            wm_uncertainty=0.80,
            max_uncertainty=threshold,
            imagined_valid_strict=False,
            imagined_valid_soft=True,
            imagined_head_has_signal=True,
            wm_validity_weight=0.25,
            soft_validity_weight=0.25,
        )
        self.assertGreater(w, 0.0)
        self.assertTrue(used_soft)

    def test_soft_valid_still_falls_back_when_above_soft_uncertainty_threshold(self):
        threshold, mode, fallback = resolve_wm_ranking_uncertainty_gate(
            wm_strict=False,
            wm_soft=True,
            wm_unc=0.98,
            strict_threshold=0.5,
            soft_threshold=0.95,
        )
        self.assertEqual(fallback, "high_uncertainty_soft")
        self.assertEqual(mode, "soft")
        w, *_ = compute_wm_pbt_source_activation_weight(
            enable=True,
            gate_mode="confidence_only",
            frames=500_000_000,
            start_after_frames=300_000_000,
            ramp_frames=300_000_000,
            wm_uncertainty=0.98,
            max_uncertainty=threshold,
            imagined_valid_strict=False,
            imagined_valid_soft=True,
            imagined_head_has_signal=True,
            wm_validity_weight=0.25,
            soft_validity_weight=0.25,
        )
        self.assertEqual(w, 0.0)


if __name__ == "__main__":
    unittest.main()
