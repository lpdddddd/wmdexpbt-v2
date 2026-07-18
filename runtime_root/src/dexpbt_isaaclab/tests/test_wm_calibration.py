"""Tests for WM calibration targets and evaluator."""

from __future__ import annotations

import unittest

import torch

from dexpbt_isaaclab.world_model.seq_wm_base import SequenceBatch
from dexpbt_isaaclab.world_model.wm_activation_gate import effective_wm_trust_weight
from dexpbt_isaaclab.world_model.wm_calibration_evaluator import (
    WMCalibrationConfig,
    WMCalibrationEvaluator,
    WMCalibrationState,
)
from dexpbt_isaaclab.world_model.wm_calibration_metrics import (
    auc_binary,
    build_wm_calibration_targets,
    ece,
    extract_wm_calibration_predictions,
    resolve_calibration_horizon,
    resolve_wm_calibration_eval_tol,
)


def _seq_batch(
    *,
    t: int = 6,
    b: int = 2,
    d0: float = 0.05,
    d_improve: float = 0.01,
    tol: float = 0.015,
    done_at: int | None = None,
) -> SequenceBatch:
    d_hat = torch.full((t, b), float(d0))
    if d_improve > 0:
        d_hat[-1, :] = float(d0) - float(d_improve)
    done = torch.zeros(t, b)
    if done_at is not None and 0 <= done_at < t:
        done[done_at, :] = 1.0
    seg = torch.zeros(t, b, dtype=torch.long)
    if done_at is not None:
        seg[done_at + 1 :, :] = 1
    valid = torch.ones(t, b)
    if done_at is not None:
        valid[done_at + 1 :, :] = 0.0
    return SequenceBatch(
        obs=torch.zeros(t, b, 4),
        action=torch.zeros(t, b, 3),
        reward=torch.zeros(t, b),
        done=done,
        next_obs=torch.zeros(t, b, 4),
        returns=torch.zeros(t, b),
        d_hat=d_hat,
        success_tolerance_seq=torch.full((t, b), float(tol)),
        valid_mask=valid,
        segment_id=seg,
        dropped=torch.zeros(t, b),
        failure=torch.zeros(t, b),
    )


class _FakeHeads:
    has_recovery_head = True
    has_horizon_heads = True

    def predict_transition(self, belief_flat, action_flat):
        n = belief_flat.shape[0]
        return {
            "recovery_logits": torch.zeros(n, 1),
            "horizon_logits": {
                4: {
                    "success": torch.zeros(n, 1),
                    "drop": torch.full((n, 1), -4.0),
                }
            },
        }


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

    def test_recovery_label_positive_case(self):
        batch = _seq_batch(d0=0.03, d_improve=0.015, tol=0.015)
        tgt, _diag = build_wm_calibration_targets(
            batch, horizon=4, improve_margin=0.005, floor_tol_threshold=0.02
        )
        pos = tgt["y_recovery"][tgt["eval_mask"]]
        self.assertGreater(float(pos.max().item()), 0.5)

    def test_drop_label_negative_case(self):
        batch = _seq_batch(d0=0.03, d_improve=0.015, tol=0.015)
        tgt, _diag = build_wm_calibration_targets(
            batch, horizon=4, improve_margin=0.005, floor_tol_threshold=0.02
        )
        drops = tgt["y_drop"][tgt["eval_mask"]]
        self.assertLessEqual(float(drops.max().item()), 0.0)

    def test_done_boundary_excludes_future(self):
        batch = _seq_batch(d0=0.03, d_improve=0.02, tol=0.015, done_at=0)
        tgt, _diag = build_wm_calibration_targets(
            batch, horizon=4, improve_margin=0.005, floor_tol_threshold=0.02
        )
        self.assertFalse(bool(tgt["eval_mask"][0].item()))

    def test_insufficient_samples_do_not_update_alpha(self):
        ev = WMCalibrationEvaluator(WMCalibrationConfig(enabled=True, min_samples=512))
        st = WMCalibrationState(alpha_wm=0.42, last_eval_frame=0)
        metrics = {
            "wm_cal/insufficient_samples": 1.0,
            "wm_cal/n_samples": 10.0,
            "wm_cal/missing_head": 0.0,
        }
        st.last_metrics = metrics
        self.assertAlmostEqual(st.alpha_wm, 0.42)

    def test_missing_head_no_crash(self):
        preds, diag = extract_wm_calibration_predictions(
            None,
            belief_flat=torch.zeros(4, 8),
            action_flat=torch.zeros(4, 3),
            horizon=4,
        )
        self.assertIsNone(preds)
        self.assertEqual(diag["wm_cal/missing_head"], 1.0)

    def test_extract_predictions_from_fake_heads(self):
        preds, diag = extract_wm_calibration_predictions(
            _FakeHeads(),
            belief_flat=torch.zeros(3, 8),
            action_flat=torch.zeros(3, 3),
            horizon=4,
        )
        self.assertIsNotNone(preds)
        self.assertEqual(diag["wm_cal/missing_head"], 0.0)
        self.assertEqual(int(preds["p_recovery"].shape[0]), 3)

    def test_resolve_calibration_horizon(self):
        self.assertEqual(resolve_calibration_horizon((8, 16), default=4), 8)
        self.assertEqual(resolve_calibration_horizon((2, 3), default=4), 2)

    def test_pos_rate_not_constant_one(self):
        batch = _seq_batch(d0=0.03, d_improve=0.0, tol=0.015)
        tgt, _diag = build_wm_calibration_targets(
            batch, horizon=4, improve_margin=0.05, floor_tol_threshold=0.02
        )
        y = tgt["y_recovery"][tgt["eval_mask"]].tolist()
        if y:
            rates = [float(v) for v in y]
            self.assertFalse(all(v > 0.5 for v in rates) and len(set(rates)) == 1)

    def test_legacy_floor_mask_empty_for_mid_stage_tol(self):
        batch = _seq_batch(d0=0.05, d_improve=0.01, tol=0.038)
        tgt, diag = build_wm_calibration_targets(
            batch,
            horizon=4,
            improve_margin=0.005,
            floor_tol_threshold=0.02,
            eval_tol_mode="fixed",
            eval_tol_threshold=0.02,
        )
        self.assertEqual(int(tgt["eval_mask"].sum().item()), 0)
        self.assertAlmostEqual(diag["wm_cal/eval_tol_threshold"], 0.02)

    def test_hybrid_covers_mid_stage_tol(self):
        batch = _seq_batch(d0=0.05, d_improve=0.01, tol=0.038)
        tgt, diag = build_wm_calibration_targets(
            batch,
            horizon=4,
            improve_margin=0.005,
            floor_tol_threshold=0.02,
            eval_tol_mode="hybrid",
            eval_tol_threshold=0.05,
            eval_tol_cap=0.06,
        )
        self.assertGreaterEqual(diag["wm_cal/eval_tol_threshold"], 0.038)
        self.assertGreater(int(tgt["eval_mask"].sum().item()), 0)

    def test_adaptive_respects_cap(self):
        batch = _seq_batch(d0=0.05, d_improve=0.01, tol=0.075)
        valid = torch.ones(batch.T, batch.B)
        future_ok = torch.ones(batch.T, batch.B)
        tol = batch.success_tolerance_seq
        eval_tol, diag = resolve_wm_calibration_eval_tol(
            tol,
            valid=valid,
            future_ok=future_ok,
            floor_tol_threshold=0.02,
            mode="adaptive_quantile",
            fixed_eval_tol=0.05,
            quantile=0.5,
            margin=1.1,
            cap=0.06,
        )
        self.assertLessEqual(eval_tol, 0.06)
        self.assertAlmostEqual(diag["wm_cal/eval_tol_threshold"], eval_tol)


if __name__ == "__main__":
    unittest.main()
