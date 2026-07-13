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

    def test_combined_uncertainty_max_over_heads(self):
        from dexpbt_isaaclab.world_model.snippet_wm_reranker import (
            binary_head_uncertainty,
            combined_head_uncertainty,
        )

        p_rec = torch.tensor([0.95, 0.9])
        p_succ = torch.tensor([0.5, 0.9])
        p_drop = torch.tensor([0.1, 0.1])
        u = combined_head_uncertainty(p_rec, p_succ, p_drop)
        self.assertGreater(float(u[0].item()), 0.9)
        self.assertLess(float(u[1].item()), 0.3)
        self.assertGreaterEqual(
            float(u[1].item()), float(binary_head_uncertainty(p_rec[1]).item()) - 1e-5
        )

    def test_nan_head_forces_uncertainty_one(self):
        class _NanHeads(_FakeHeads):
            def predict_transition(self, belief_flat, action_flat):
                n = belief_flat.shape[0]
                return {
                    "recovery_logits": torch.full((n, 1), 2.0),
                    "horizon_logits": {
                        4: {
                            "success": torch.full((n, 1), float("nan")),
                            "drop": torch.full((n, 1), -4.0),
                        }
                    },
                }

        preds, _diag = extract_wm_calibration_predictions(
            _NanHeads(),
            belief_flat=torch.zeros(2, 8),
            action_flat=torch.zeros(2, 3),
            horizon=4,
        )
        self.assertIsNotNone(preds)
        self.assertTrue(torch.allclose(preds["uncertainty"], torch.ones(2)))
        self.assertFalse(bool(preds["valid_mask"].all().item()))

    def test_combined_score_matches_runtime_helper(self):
        from dexpbt_isaaclab.world_model.snippet_wm_reranker import (
            SnippetWMRerankConfig,
            compute_snippet_wm_term,
            combined_head_uncertainty,
        )

        cfg = SnippetWMRerankConfig(
            w_rec=0.55, w_success=0.15, w_drop=0.25, w_unc=0.05, score_cap=0.4
        )
        cal = WMCalibrationEvaluator(WMCalibrationConfig(enabled=True, min_samples=4))
        cal.apply_snippet_rerank_weights(cfg)
        p_rec = torch.tensor([0.8, 0.2, 0.7, 0.3, 0.9])
        p_succ = torch.tensor([0.7, 0.1, 0.6, 0.4, 0.8])
        p_drop = torch.tensor([0.2, 0.8, 0.3, 0.5, 0.1])
        unc = combined_head_uncertainty(p_rec, p_succ, p_drop)
        expected = compute_snippet_wm_term(
            p_recovery=p_rec,
            p_success=p_succ,
            p_drop=p_drop,
            uncertainty=unc,
            w_rec=cfg.w_rec,
            w_success=cfg.w_success,
            w_drop=cfg.w_drop,
            w_unc=cfg.w_unc,
            score_cap=cfg.score_cap,
        )
        metrics = cal.evaluate_batch(
            {
                "y_recovery": [1.0, 0.0, 1.0, 0.0, 1.0],
                "y_success": [1.0, 0.0, 1.0, 0.0, 1.0],
                "y_drop": [0.0, 1.0, 0.0, 1.0, 0.0],
                "y_improve": [1.0, 0.0, 1.0, 0.0, 1.0],
                "p_recovery": p_rec.tolist(),
                "p_success": p_succ.tolist(),
                "p_drop": p_drop.tolist(),
                "uncertainty": unc.tolist(),
            }
        )
        self.assertAlmostEqual(
            metrics["wm_cal/combined_score_mean"],
            float(expected.mean().item()),
            places=5,
        )

    def test_recovery_good_combined_bad_keeps_alpha_low(self):
        """Recovery ranks well but success/drop invert combined ranking → low alpha."""
        cal = WMCalibrationEvaluator(
            WMCalibrationConfig(
                enabled=True,
                min_samples=8,
                gate_auc_recovery=0.55,
                gate_ece_max=0.35,
                gate_rank_corr_min=0.05,
                alpha_max=0.5,
                ema_beta=0.0,
            )
        )
        cal.apply_snippet_rerank_weights(
            type(
                "C",
                (),
                {
                    "w_rec": 0.1,
                    "w_success": 0.4,
                    "w_drop": 0.5,
                    "w_unc": 0.0,
                    "score_cap": 10.0,
                },
            )()
        )
        n = 12
        p_rec = torch.linspace(0.1, 0.9, n)
        y_rec = (p_rec > 0.5).float()
        p_succ = 1.0 - p_rec
        p_drop = p_rec.clone()
        y_succ = y_rec.clone()
        y_drop = 1.0 - y_rec
        unc = torch.zeros(n)
        metrics = cal.evaluate_batch(
            {
                "y_recovery": y_rec.tolist(),
                "y_success": y_succ.tolist(),
                "y_drop": y_drop.tolist(),
                "y_improve": y_rec.tolist(),
                "p_recovery": p_rec.tolist(),
                "p_success": p_succ.tolist(),
                "p_drop": p_drop.tolist(),
                "uncertainty": unc.tolist(),
            }
        )
        self.assertGreater(metrics["wm_cal/rank_corr_recovery"], 0.5)
        self.assertLess(metrics["wm_cal/rank_corr_combined"], 0.05)
        st = WMCalibrationState(alpha_wm=0.0)
        st = cal.compute_alpha(metrics, st)
        self.assertLessEqual(st.alpha_wm, 1e-6)

    def test_combined_good_raises_alpha(self):
        cal = WMCalibrationEvaluator(
            WMCalibrationConfig(
                enabled=True,
                min_samples=8,
                gate_auc_recovery=0.55,
                gate_ece_max=0.35,
                gate_rank_corr_min=0.05,
                alpha_max=0.5,
                ema_beta=0.0,
            )
        )
        cal.apply_snippet_rerank_weights(
            type(
                "C",
                (),
                {
                    "w_rec": 0.4,
                    "w_success": 0.3,
                    "w_drop": 0.3,
                    "w_unc": 0.0,
                    "score_cap": 10.0,
                },
            )()
        )
        n = 16
        p_rec = torch.linspace(0.05, 0.95, n)
        p_succ = p_rec.clone()
        p_drop = 1.0 - p_rec
        y_rec = (p_rec > 0.5).float()
        y_succ = y_rec.clone()
        y_drop = 1.0 - y_rec
        unc = torch.zeros(n)
        metrics = cal.evaluate_batch(
            {
                "y_recovery": y_rec.tolist(),
                "y_success": y_succ.tolist(),
                "y_drop": y_drop.tolist(),
                "y_improve": y_rec.tolist(),
                "p_recovery": p_rec.tolist(),
                "p_success": p_succ.tolist(),
                "p_drop": p_drop.tolist(),
                "uncertainty": unc.tolist(),
            }
        )
        self.assertGreater(metrics["wm_cal/rank_corr_combined"], 0.2)
        self.assertEqual(metrics["wm_cal/rank_corr"], metrics["wm_cal/rank_corr_combined"])
        st = WMCalibrationState(alpha_wm=0.0)
        st = cal.compute_alpha(metrics, st)
        self.assertGreater(st.alpha_wm, 0.05)

    def test_single_class_success_drop_finite_no_min_gate_lock(self):
        import math

        cal = WMCalibrationEvaluator(
            WMCalibrationConfig(
                enabled=True,
                min_samples=8,
                gate_auc_recovery=0.55,
                gate_ece_max=0.35,
                gate_rank_corr_min=0.0,
                alpha_max=0.5,
                ema_beta=0.0,
            )
        )
        n = 12
        p_rec = torch.linspace(0.1, 0.9, n)
        y_rec = (p_rec > 0.5).float()
        metrics = cal.evaluate_batch(
            {
                "y_recovery": y_rec.tolist(),
                "y_success": [1.0] * n,
                "y_drop": [0.0] * n,
                "y_improve": y_rec.tolist(),
                "p_recovery": p_rec.tolist(),
                "p_success": [0.8] * n,
                "p_drop": [0.1] * n,
                "uncertainty": [0.0] * n,
            }
        )
        for k, v in metrics.items():
            self.assertTrue(math.isfinite(float(v)), msg=k)
        st = WMCalibrationState(alpha_wm=0.0)
        st = cal.compute_alpha(metrics, st)
        self.assertGreaterEqual(st.alpha_wm, 0.0)
        self.assertTrue(math.isfinite(st.alpha_wm))

    def test_rerank_weight_change_syncs_calibration_score(self):
        from dexpbt_isaaclab.world_model.snippet_wm_reranker import (
            SnippetWMRerankConfig,
            compute_snippet_wm_term,
            combined_head_uncertainty,
            score_snippet_candidates,
        )

        cfg_a = SnippetWMRerankConfig(
            enabled=True,
            w_rec=0.3,
            w_success=0.2,
            w_drop=0.4,
            w_unc=0.1,
            score_cap=0.25,
        )
        cfg_b = SnippetWMRerankConfig(
            enabled=True,
            w_rec=0.7,
            w_success=0.05,
            w_drop=0.05,
            w_unc=0.2,
            score_cap=0.5,
        )
        p_rec = torch.tensor([[0.9, 0.2]])
        p_succ = torch.tensor([[0.8, 0.1]])
        p_drop = torch.tensor([[0.1, 0.7]])
        unc = combined_head_uncertainty(
            p_rec.reshape(-1), p_succ.reshape(-1), p_drop.reshape(-1)
        ).view(1, 2)
        term_a = compute_snippet_wm_term(
            p_recovery=p_rec,
            p_success=p_succ,
            p_drop=p_drop,
            uncertainty=unc,
            w_rec=cfg_a.w_rec,
            w_success=cfg_a.w_success,
            w_drop=cfg_a.w_drop,
            w_unc=cfg_a.w_unc,
            score_cap=cfg_a.score_cap,
        )
        term_b = compute_snippet_wm_term(
            p_recovery=p_rec,
            p_success=p_succ,
            p_drop=p_drop,
            uncertainty=unc,
            w_rec=cfg_b.w_rec,
            w_success=cfg_b.w_success,
            w_drop=cfg_b.w_drop,
            w_unc=cfg_b.w_unc,
            score_cap=cfg_b.score_cap,
        )
        self.assertFalse(torch.allclose(term_a, term_b))
        cal = WMCalibrationEvaluator(WMCalibrationConfig(enabled=True, min_samples=2))
        cal.apply_snippet_rerank_weights(cfg_a)
        self.assertAlmostEqual(cal.cfg.w_rec, cfg_a.w_rec)
        cal.apply_snippet_rerank_weights(cfg_b)
        self.assertAlmostEqual(cal.cfg.w_rec, cfg_b.w_rec)
        self.assertAlmostEqual(cal.cfg.score_cap, cfg_b.score_cap)

        pri = torch.ones(1, 2)
        action_delta = torch.zeros(1, 2)
        _, _, diag_a, _ = score_snippet_candidates(
            priority=pri,
            p_recovery=p_rec,
            p_success=p_succ,
            p_drop=p_drop,
            uncertainty=unc,
            action_delta=action_delta,
            alpha_wm=1.0,
            cfg=cfg_a,
        )
        _, _, diag_b, _ = score_snippet_candidates(
            priority=pri,
            p_recovery=p_rec,
            p_success=p_succ,
            p_drop=p_drop,
            uncertainty=unc,
            action_delta=action_delta,
            alpha_wm=1.0,
            cfg=cfg_b,
        )
        self.assertNotAlmostEqual(
            diag_a["snippet_wm_rerank/wm_term_mean"],
            diag_b["snippet_wm_rerank/wm_term_mean"],
            places=4,
        )
        # Calibration combined score must track the same helper under cfg_b weights.
        metrics = cal.evaluate_batch(
            {
                "p_recovery": p_rec.reshape(-1).tolist(),
                "p_success": p_succ.reshape(-1).tolist(),
                "p_drop": p_drop.reshape(-1).tolist(),
                "uncertainty": unc.reshape(-1).tolist(),
                "y_recovery": [1.0, 0.0],
                "y_success": [1.0, 0.0],
                "y_drop": [0.0, 1.0],
                "y_improve": [1.0, 0.0],
            }
        )
        self.assertAlmostEqual(
            metrics["wm_cal/combined_score_mean"],
            float(term_b.mean().item()),
            places=5,
        )

    def test_state_backward_compat_missing_metrics(self):
        st = WMCalibrationState.from_dict({"alpha_wm": 0.12, "last_eval_frame": 3})
        self.assertAlmostEqual(st.alpha_wm, 0.12)
        self.assertEqual(st.last_metrics, {})
        cal = WMCalibrationEvaluator(WMCalibrationConfig(enabled=True, min_samples=4))
        metrics = cal.evaluate_batch(
            {
                "y_recovery": [1.0, 0.0, 1.0, 0.0],
                "y_success": [1.0, 0.0, 1.0, 0.0],
                "y_drop": [0.0, 1.0, 0.0, 1.0],
                "y_improve": [1.0, 0.0, 1.0, 0.0],
                "p_recovery": [0.9, 0.1, 0.8, 0.2],
                "p_success": [0.8, 0.2, 0.7, 0.3],
                "p_drop": [0.1, 0.9, 0.2, 0.8],
                "uncertainty": [0.0, 0.0, 0.0, 0.0],
            }
        )
        self.assertIn("wm_cal/rank_corr", metrics)
        self.assertIn("wm_cal/rank_corr_combined", metrics)
        self.assertEqual(metrics["wm_cal/rank_corr"], metrics["wm_cal/rank_corr_combined"])


if __name__ == "__main__":
    unittest.main()
