"""Tests for WM calibration."""

from __future__ import annotations

import unittest

from dexpbt_isaaclab.world_model.wm_activation_gate import effective_wm_trust_weight
from dexpbt_isaaclab.world_model.wm_calibration_evaluator import (
    WMCalibrationConfig,
    WMCalibrationEvaluator,
    WMCalibrationState,
)
from dexpbt_isaaclab.world_model.wm_calibration_metrics import auc_binary, clamp01, ece


class TestWMCalibration(unittest.TestCase):
    def test_metrics_bounds(self):
        probs = [0.1, 0.2, 0.8, 0.9]
        labels = [0.0, 0.0, 1.0, 1.0]
        self.assertGreater(auc_binary(probs, labels), 0.5)
        self.assertLessEqual(ece(probs, labels), 1.0)

    def test_alpha_when_cal_off(self):
        w = effective_wm_trust_weight(0.8, 0.5, wm_calibration_enable=False)
        self.assertAlmostEqual(w, 0.8)

    def test_alpha_when_cal_on(self):
        w = effective_wm_trust_weight(0.8, 0.5, wm_calibration_enable=True)
        self.assertAlmostEqual(w, 0.4)

    def test_state_roundtrip(self):
        st = WMCalibrationState(alpha_wm=0.3, ema_raw=0.3, calibration_pass=True)
        st2 = WMCalibrationState.from_dict(st.to_dict())
        self.assertAlmostEqual(st2.alpha_wm, 0.3)

    def test_compute_alpha_zero_when_bad(self):
        ev = WMCalibrationEvaluator(WMCalibrationConfig(enabled=True))
        st = WMCalibrationState()
        metrics = {"wm_cal/AUC_recovery": 0.1, "wm_cal/ECE": 0.9, "wm_cal/rank_corr": -0.5}
        metrics = ev.evaluate_batch(
            {
                "p_recovery": [0.1, 0.2],
                "y_recovery": [0, 1],
                "p_drop": [0.1, 0.2],
                "y_drop": [0, 0],
                "p_success": [0.1, 0.2],
                "y_success": [0, 1],
                "uncertainty": [0.1, 0.2],
                "error": [0.5, 0.4],
                "wm_recovery_score": [0.1, 0.2],
                "y_improve": [0, 1],
            }
        )
        st = ev.compute_alpha(metrics, st)
        self.assertLessEqual(st.alpha_wm, 0.5)
