"""Unit tests for tie-aware calibration metrics and shared head extraction."""

from __future__ import annotations

import math
import random
import unittest

import torch

from dexpbt_isaaclab.world_model.snippet_wm_reranker import (
    SnippetWMRerankConfig,
    compute_snippet_wm_term,
    wm_head_probs_for_candidates,
)
from dexpbt_isaaclab.world_model.wm_calibration_evaluator import (
    WMCalibrationConfig,
    WMCalibrationEvaluator,
    WMCalibrationState,
    apply_calibration_outcome,
)
from dexpbt_isaaclab.world_model.wm_calibration_metrics import (
    auc_binary,
    spearman_rank,
)
from dexpbt_isaaclab.world_model.wm_head_probs import (
    combined_head_uncertainty,
    extract_required_wm_head_probs,
)


class TestTieAwareMetrics(unittest.TestCase):
    def test_spearman_average_ranks_for_ties(self):
        # x has ties at 1.0; average ranks should be used.
        x = [1.0, 1.0, 2.0, 3.0]
        y = [0.0, 1.0, 2.0, 3.0]
        rho = spearman_rank(x, y)
        # Hand: ranks_x = [1.5, 1.5, 3, 4], ranks_y = [1,2,3,4]
        rx = [1.5, 1.5, 3.0, 4.0]
        ry = [1.0, 2.0, 3.0, 4.0]
        mx, my = sum(rx) / 4, sum(ry) / 4
        num = sum((rx[i] - mx) * (ry[i] - my) for i in range(4))
        den = math.sqrt(sum((rx[i] - mx) ** 2 for i in range(4))) * math.sqrt(
            sum((ry[i] - my) ** 2 for i in range(4))
        )
        self.assertAlmostEqual(rho, num / den, places=6)

    def test_spearman_permutation_invariant(self):
        x = [0.1, 0.2, 0.2, 0.9, 0.5]
        y = [0.0, 1.0, 0.0, 1.0, 1.0]
        base = spearman_rank(x, y)
        idx = list(range(len(x)))
        random.seed(0)
        for _ in range(20):
            random.shuffle(idx)
            xp = [x[i] for i in idx]
            yp = [y[i] for i in idx]
            self.assertAlmostEqual(spearman_rank(xp, yp), base, places=6)

    def test_constant_spearman_zero(self):
        self.assertEqual(spearman_rank([1.0, 1.0, 1.0], [0.0, 1.0, 0.5]), 0.0)
        self.assertEqual(spearman_rank([0.0, 1.0, 0.5], [2.0, 2.0, 2.0]), 0.0)

    def test_auc_tied_predictions(self):
        # All predictions tied with both classes -> 0.5
        self.assertAlmostEqual(auc_binary([0.4, 0.4, 0.4, 0.4], [0, 1, 0, 1]), 0.5, places=6)
        # Classic separable
        self.assertGreater(auc_binary([0.1, 0.2, 0.8, 0.9], [0, 0, 1, 1]), 0.9)

    def test_auc_permutation_invariant(self):
        p = [0.1, 0.5, 0.5, 0.9, 0.2]
        y = [0.0, 1.0, 0.0, 1.0, 0.0]
        base = auc_binary(p, y)
        idx = list(range(len(p)))
        random.seed(1)
        for _ in range(20):
            random.shuffle(idx)
            self.assertAlmostEqual(
                auc_binary([p[i] for i in idx], [y[i] for i in idx]), base, places=6
            )

    def test_clipped_wm_term_rank_permutation_invariant(self):
        cfg = SnippetWMRerankConfig(score_cap=0.05, w_rec=0.3, w_success=0.2, w_drop=0.4, w_unc=0.1)
        p_rec = torch.tensor([0.9, 0.8, 0.1, 0.2, 0.95, 0.05])
        p_succ = torch.tensor([0.9, 0.7, 0.2, 0.1, 0.8, 0.15])
        p_drop = torch.tensor([0.1, 0.2, 0.8, 0.7, 0.05, 0.9])
        unc = combined_head_uncertainty(p_rec, p_succ, p_drop)
        pred = compute_snippet_wm_term(
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
        util = cfg.w_rec * torch.tensor([1.0, 1.0, 0.0, 0.0, 1.0, 0.0]) + cfg.w_success * torch.tensor(
            [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
        ) - cfg.w_drop * torch.tensor([0.0, 0.0, 1.0, 1.0, 0.0, 1.0])
        base = spearman_rank(pred.tolist(), util.tolist())
        idx = list(range(pred.numel()))
        random.seed(2)
        for _ in range(15):
            random.shuffle(idx)
            self.assertAlmostEqual(
                spearman_rank([pred[i].item() for i in idx], [util[i].item() for i in idx]),
                base,
                places=6,
            )


class TestHorizonAlignment(unittest.TestCase):
    def test_runtime_calibration_horizon_mean_identity(self):
        n = 4
        # Distinct logits so H4 vs H8 means differ from either alone.
        preds = {
            "recovery_logits": torch.linspace(-2, 2, n),
            "horizon_logits": {
                8: {
                    "success": torch.linspace(-1, 3, n),
                    "drop": torch.linspace(2, -2, n),
                },
                4: {
                    "success": torch.linspace(3, -1, n),
                    "drop": torch.linspace(-2, 2, n),
                },
            },
        }
        steps = (4, 8)
        cal = extract_required_wm_head_probs(preds, horizon_steps=steps, expected_batch=n)

        def _predict(belief_rep, cand_flat):
            m = belief_rep.shape[0]
            # Expand single-row style: reuse first n then tile — here B*K = n with B=n,K=1
            return {
                "recovery_logits": preds["recovery_logits"],
                "horizon_logits": {
                    4: {
                        "success": preds["horizon_logits"][4]["success"],
                        "drop": preds["horizon_logits"][4]["drop"],
                    },
                    8: {
                        "success": preds["horizon_logits"][8]["success"],
                        "drop": preds["horizon_logits"][8]["drop"],
                    },
                },
            }

        belief = torch.zeros(n, 3)
        acts = torch.zeros(n, 1, 2)
        p_rec, p_succ, p_drop, unc, valid, _ = wm_head_probs_for_candidates(
            belief_flat=belief,
            cand_first_actions=acts,
            predict_fn=_predict,
            horizon_steps=steps,
        )
        self.assertTrue(bool(valid.all()))
        self.assertTrue(torch.allclose(p_rec.reshape(-1), cal.p_recovery, atol=1e-5))
        self.assertTrue(torch.allclose(p_succ.reshape(-1), cal.p_success, atol=1e-5))
        self.assertTrue(torch.allclose(p_drop.reshape(-1), cal.p_drop, atol=1e-5))
        self.assertTrue(torch.allclose(unc.reshape(-1), cal.uncertainty, atol=1e-5))
        # Mean of 4 and 8 differs from horizon-4-only.
        only4 = extract_required_wm_head_probs(preds, horizon_steps=(4,), expected_batch=n)
        self.assertFalse(torch.allclose(cal.p_success, only4.p_success))
        cfg = SnippetWMRerankConfig()
        term_rt = compute_snippet_wm_term(
            p_recovery=p_rec.reshape(-1),
            p_success=p_succ.reshape(-1),
            p_drop=p_drop.reshape(-1),
            uncertainty=unc.reshape(-1),
            w_rec=cfg.w_rec,
            w_success=cfg.w_success,
            w_drop=cfg.w_drop,
            w_unc=cfg.w_unc,
            score_cap=cfg.score_cap,
        )
        term_cal = compute_snippet_wm_term(
            p_recovery=cal.p_recovery,
            p_success=cal.p_success,
            p_drop=cal.p_drop,
            uncertainty=cal.uncertainty,
            w_rec=cfg.w_rec,
            w_success=cfg.w_success,
            w_drop=cfg.w_drop,
            w_unc=cfg.w_unc,
            score_cap=cfg.score_cap,
        )
        self.assertTrue(torch.allclose(term_rt, term_cal, atol=1e-5))


class TestFailureAlphaTransitions(unittest.TestCase):
    def test_structural_clears_stale_alpha(self):
        cfg = WMCalibrationConfig(enabled=True, ema_beta=0.9, alpha_max=1.0)
        st = WMCalibrationState(alpha_wm=0.8, ema_raw=0.8, calibration_pass=True)
        st = apply_calibration_outcome(
            st,
            cfg=cfg,
            frames=100,
            metrics={"wm_cal/missing_head": 1.0},
            mode="structural",
        )
        self.assertEqual(st.alpha_wm, 0.0)
        self.assertEqual(st.ema_raw, 0.0)
        self.assertFalse(st.calibration_pass)
        self.assertEqual(st.last_eval_frame, 100)

    def test_insufficient_decays_ema(self):
        cfg = WMCalibrationConfig(enabled=True, ema_beta=0.9, alpha_max=1.0)
        st = WMCalibrationState(alpha_wm=0.8, ema_raw=0.8)
        st = apply_calibration_outcome(
            st,
            cfg=cfg,
            frames=200,
            metrics={},
            mode="insufficient",
        )
        self.assertAlmostEqual(st.ema_raw, 0.9 * 0.8, places=6)
        self.assertAlmostEqual(st.alpha_wm, 0.9 * 0.8, places=6)
        self.assertFalse(st.calibration_pass)

    def test_length_mismatch_no_silent_zero(self):
        ev = WMCalibrationEvaluator(WMCalibrationConfig(enabled=True))
        metrics = ev.evaluate_batch(
            {
                "p_recovery": [0.1, 0.9],
                "y_recovery": [0.0, 1.0],
                "p_success": [0.1],  # mismatch
                "y_success": [0.0, 1.0],
                "p_drop": [0.1, 0.2],
                "y_drop": [0.0, 1.0],
                "uncertainty": [0.1, 0.2],
            }
        )
        self.assertEqual(metrics["wm_cal/batch_invalid"], 1.0)
        self.assertEqual(metrics["wm_cal/rank_corr_combined"], 0.0)

    def test_tensor_input_no_ambiguous_truth(self):
        ev = WMCalibrationEvaluator(WMCalibrationConfig(enabled=True))
        metrics = ev.evaluate_batch(
            {
                "p_recovery": torch.tensor([0.1, 0.9, 0.2, 0.8]),
                "y_recovery": torch.tensor([0.0, 1.0, 0.0, 1.0]),
                "p_success": torch.tensor([0.2, 0.8, 0.3, 0.7]),
                "y_success": torch.tensor([0.0, 1.0, 0.0, 1.0]),
                "p_drop": torch.tensor([0.8, 0.1, 0.7, 0.2]),
                "y_drop": torch.tensor([1.0, 0.0, 1.0, 0.0]),
                "uncertainty": torch.tensor([0.1, 0.1, 0.1, 0.1]),
            }
        )
        self.assertEqual(metrics["wm_cal/batch_invalid"], 0.0)
        self.assertEqual(metrics["wm_cal/eval_valid"], 1.0)


if __name__ == "__main__":
    unittest.main()
