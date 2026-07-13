"""Tests for snippet WM reranker (row-level fail-closed selection)."""

from __future__ import annotations

import unittest

import torch

from dexpbt_isaaclab.world_model.snippet_wm_reranker import (
    SnippetWMRerankConfig,
    effective_alpha_wm,
    score_snippet_candidates,
    wm_head_probs_for_candidates,
)


def _score(**kw):
    return score_snippet_candidates(**kw)


class TestSnippetWMReranker(unittest.TestCase):
    def test_all_valid_best_idx_equals_candidate_score_argmax(self):
        cfg = SnippetWMRerankConfig(enabled=True, w_rec=1.0, score_cap=1.0, w_delta=0.0)
        pri = torch.tensor([[1.0, 5.0, 2.0]])
        p_rec = torch.tensor([[0.9, 0.1, 0.2]])
        best_idx, cand_score, _ = _score(
            priority=pri,
            p_recovery=p_rec,
            p_success=torch.zeros(1, 3),
            p_drop=torch.zeros(1, 3),
            uncertainty=torch.zeros(1, 3),
            action_delta=torch.zeros(1, 3),
            alpha_wm=1.0,
            cfg=cfg,
            wm_valid_mask=torch.ones(1, 3, dtype=torch.bool),
        )
        self.assertTrue(torch.equal(best_idx, cand_score.argmax(dim=1)))

    def test_partial_valid_row_falls_back_to_priority(self):
        """Invalid candidate has highest priority; row must still use raw priority."""
        cfg = SnippetWMRerankConfig(enabled=True, w_rec=2.0, score_cap=2.0, w_delta=0.0)
        # Invalid idx0 has highest priority; WM would prefer valid idx1.
        pri = torch.tensor([[10.0, 1.0, 2.0]])
        p_rec = torch.tensor([[0.0, 1.0, 0.0]])
        valid = torch.tensor([[False, True, True]])
        best_idx, cand_score, diag = _score(
            priority=pri,
            p_recovery=p_rec,
            p_success=torch.zeros(1, 3),
            p_drop=torch.zeros(1, 3),
            uncertainty=torch.zeros(1, 3),
            action_delta=torch.zeros(1, 3),
            alpha_wm=1.0,
            cfg=cfg,
            wm_valid_mask=valid,
        )
        self.assertEqual(int(best_idx.item()), 0)  # raw priority
        self.assertTrue(torch.equal(cand_score, pri))
        self.assertEqual(diag["snippet_wm_rerank/row_active_frac"], 0.0)
        self.assertEqual(diag["snippet_wm_rerank/changed_frac"], 0.0)
        self.assertEqual(diag["snippet_wm_rerank/fallback_reason_id"], 12.0)

    def test_partial_valid_invalid_actor_proxy_no_score_mismatch(self):
        """Invalid cand closest to actor must not participate via WM scores."""
        cfg = SnippetWMRerankConfig(enabled=True, w_rec=2.0, score_cap=2.0, w_delta=0.05)
        pri = torch.tensor([[1.0, 5.0, 2.0]])
        p_rec = torch.tensor([[0.9, 0.1, 0.1]])
        # Invalid idx0 is closest to actor (delta=0); valid rows would prefer WM on 0.
        action_delta = torch.tensor([[0.0, 4.0, 3.0]])
        valid = torch.tensor([[False, True, True]])
        best_idx, cand_score, _ = _score(
            priority=pri,
            p_recovery=p_rec,
            p_success=torch.zeros(1, 3),
            p_drop=torch.zeros(1, 3),
            uncertainty=torch.zeros(1, 3),
            action_delta=action_delta,
            alpha_wm=1.0,
            cfg=cfg,
            wm_valid_mask=valid,
        )
        # Fail-closed => raw priority selection (idx1), not WM/action_delta.
        self.assertEqual(int(best_idx.item()), 1)
        self.assertTrue(torch.equal(best_idx, cand_score.argmax(dim=1)))
        ar = torch.arange(1)
        score_selected = cand_score[ar, best_idx]
        actor_like_idx = action_delta.argmin(dim=1)
        score_actor_proxy = cand_score[ar, actor_like_idx]
        # Both from the same candidate_score family (raw priority).
        self.assertAlmostEqual(float(score_selected.item()), float(pri[0, 1].item()), places=5)
        self.assertAlmostEqual(float(score_actor_proxy.item()), float(pri[0, 0].item()), places=5)

    def test_alpha_zero_ignores_action_delta(self):
        cfg = SnippetWMRerankConfig(enabled=True, w_rec=1.0, w_delta=10.0, score_cap=1.0)
        pri = torch.tensor([[1.0, 5.0, 2.0]])
        # Huge action_delta on the priority winner would flip selection if WM path ran.
        action_delta = torch.tensor([[0.0, 100.0, 0.0]])
        best_idx, cand_score, diag = _score(
            priority=pri,
            p_recovery=torch.ones(1, 3),
            p_success=torch.ones(1, 3),
            p_drop=torch.zeros(1, 3),
            uncertainty=torch.zeros(1, 3),
            action_delta=action_delta,
            alpha_wm=0.0,
            cfg=cfg,
            wm_valid_mask=torch.ones(1, 3, dtype=torch.bool),
        )
        self.assertEqual(int(best_idx.item()), 1)
        self.assertTrue(torch.equal(cand_score, pri))
        self.assertEqual(diag["snippet_wm_rerank/changed_frac"], 0.0)
        self.assertEqual(diag["snippet_wm_rerank/wm_term_mean"], 0.0)
        self.assertEqual(diag["snippet_wm_rerank/fallback_reason_id"], 11.0)

    def test_calibration_disabled_effective_alpha_zero(self):
        self.assertEqual(effective_alpha_wm(wm_calibration_enable=False, alpha_wm=0.9), 0.0)
        cfg = SnippetWMRerankConfig(enabled=True, w_rec=2.0, score_cap=2.0, w_delta=0.0)
        pri = torch.tensor([[10.0, 1.0]])
        p_rec = torch.tensor([[0.0, 1.0]])
        alpha = effective_alpha_wm(wm_calibration_enable=False, alpha_wm=1.0)
        best_idx, cand_score, diag = _score(
            priority=pri,
            p_recovery=p_rec,
            p_success=torch.zeros(1, 2),
            p_drop=torch.zeros(1, 2),
            uncertainty=torch.zeros(1, 2),
            action_delta=torch.zeros(1, 2),
            alpha_wm=alpha,
            cfg=cfg,
            wm_valid_mask=torch.ones(1, 2, dtype=torch.bool),
        )
        self.assertEqual(int(best_idx.item()), 0)
        self.assertTrue(torch.equal(cand_score, pri))
        self.assertEqual(diag["snippet_wm_rerank/row_active_frac"], 0.0)

    def test_alpha_positive_can_change(self):
        cfg = SnippetWMRerankConfig(enabled=True, w_rec=1.0, score_cap=1.0)
        pri = torch.tensor([[1.0, 1.0, 1.0]])
        p_rec = torch.tensor([[0.1, 0.9, 0.2]])
        idx, score, diag = _score(
            priority=pri,
            p_recovery=p_rec,
            p_success=torch.zeros(1, 3),
            p_drop=torch.zeros(1, 3),
            uncertainty=torch.zeros(1, 3),
            action_delta=torch.zeros(1, 3),
            alpha_wm=1.0,
            cfg=cfg,
            wm_valid_mask=torch.ones(1, 3, dtype=torch.bool),
        )
        self.assertEqual(int(idx.item()), 1)
        self.assertGreaterEqual(diag["snippet_wm_rerank/wm_term_mean"], 0.0)
        self.assertTrue(torch.equal(idx, score.argmax(dim=1)))
        self.assertGreater(diag["snippet_wm_rerank/row_active_frac"], 0.0)

    def test_mixed_rows_only_invalid_row_falls_back(self):
        cfg = SnippetWMRerankConfig(enabled=True, w_rec=2.0, score_cap=2.0, w_delta=0.0)
        pri = torch.tensor([[1.0, 1.0, 1.0], [9.0, 1.0, 2.0]])
        p_rec = torch.tensor([[0.9, 0.1, 0.1], [0.1, 0.9, 0.1]])
        valid = torch.tensor([[True, True, True], [False, True, True]])
        best_idx, cand_score, diag = _score(
            priority=pri,
            p_recovery=p_rec,
            p_success=torch.zeros(2, 3),
            p_drop=torch.zeros(2, 3),
            uncertainty=torch.zeros(2, 3),
            action_delta=torch.zeros(2, 3),
            alpha_wm=1.0,
            cfg=cfg,
            wm_valid_mask=valid,
        )
        self.assertEqual(int(best_idx[0].item()), 0)  # WM active
        self.assertEqual(int(best_idx[1].item()), 0)  # priority fallback
        self.assertTrue(torch.equal(cand_score[1], pri[1]))
        self.assertAlmostEqual(diag["snippet_wm_rerank/row_active_frac"], 0.5, places=5)
        self.assertAlmostEqual(diag["snippet_wm_rerank/row_fallback_frac"], 0.5, places=5)


class TestWMHeadProbsValidity(unittest.TestCase):
    def test_belief_batch_mismatch_safe_fallback(self):
        acts = torch.randn(2, 3, 4)
        belief = torch.randn(3, 5)  # B mismatch
        _p_rec, _p_succ, _p_drop, unc, valid, diag = wm_head_probs_for_candidates(
            belief_flat=belief, cand_first_actions=acts, predict_fn=lambda b, a: {}
        )
        self.assertFalse(bool(valid.any()))
        self.assertTrue(torch.equal(unc, torch.ones_like(unc)))
        self.assertEqual(diag["snippet_wm_rerank/fallback_reason_id"], 7.0)

    def test_belief_row_nan_only_that_row_invalid(self):
        acts = torch.randn(2, 2, 3)

        def _ok(belief_rep, cand_flat):
            n = belief_rep.shape[0]
            return {
                "recovery_logits": torch.zeros(n),
                "horizon_logits": {1: {"success": torch.zeros(n), "drop": torch.zeros(n)}},
            }

        belief = torch.randn(2, 4)
        belief[0, 0] = float("nan")
        _p_rec, _p_succ, _p_drop, unc, valid, diag = wm_head_probs_for_candidates(
            belief_flat=belief, cand_first_actions=acts, predict_fn=_ok
        )
        self.assertFalse(bool(valid[0].any()))
        self.assertTrue(bool(valid[1].all()))
        self.assertTrue(torch.equal(unc[0], torch.ones(2)))
        self.assertEqual(diag["snippet_wm_rerank/fallback_reason_id"], 3.0)

    def test_pred_batch_size_mismatch_safe_fallback(self):
        acts = torch.randn(1, 3, 2)
        belief = torch.randn(1, 4)

        def _wrong_batch(_b, _a):
            return {
                "recovery_logits": torch.zeros(2),  # need 3
                "horizon_logits": {1: {"success": torch.zeros(2), "drop": torch.zeros(2)}},
            }

        _p_rec, _p_succ, _p_drop, unc, valid, diag = wm_head_probs_for_candidates(
            belief_flat=belief, cand_first_actions=acts, predict_fn=_wrong_batch
        )
        self.assertFalse(bool(valid.any()))
        self.assertTrue(torch.equal(unc, torch.ones_like(unc)))
        self.assertEqual(diag["snippet_wm_rerank/fallback_reason_id"], 9.0)

    def test_partial_nan_prediction_row_fail_closed(self):
        acts = torch.randn(1, 3, 2)
        belief = torch.randn(1, 3)

        def _bad(_b, _a):
            return {
                "recovery_logits": torch.tensor([10.0, float("nan"), 0.0]),
                "horizon_logits": {
                    1: {
                        "success": torch.tensor([0.0, 0.0, 0.0]),
                        "drop": torch.tensor([0.0, 0.0, 0.0]),
                    }
                },
            }

        p_rec, p_succ, p_drop, unc, valid, diag = wm_head_probs_for_candidates(
            belief_flat=belief, cand_first_actions=acts, predict_fn=_bad
        )
        self.assertFalse(bool(valid[0, 1]))
        self.assertTrue(bool(valid[0, 0]))
        self.assertAlmostEqual(float(unc[0, 1].item()), 1.0, places=5)
        # Scoring fail-closes the whole row.
        best_idx, cand_score, score_diag = score_snippet_candidates(
            priority=torch.tensor([[1.0, 9.0, 2.0]]),
            p_recovery=p_rec,
            p_success=p_succ,
            p_drop=p_drop,
            uncertainty=unc,
            action_delta=torch.zeros(1, 3),
            alpha_wm=1.0,
            cfg=SnippetWMRerankConfig(enabled=True, w_rec=2.0, score_cap=2.0),
            wm_valid_mask=valid,
        )
        self.assertEqual(int(best_idx.item()), 1)
        self.assertTrue(torch.equal(cand_score, torch.tensor([[1.0, 9.0, 2.0]])))
        self.assertEqual(score_diag["snippet_wm_rerank/row_active_frac"], 0.0)
        self.assertGreater(diag["snippet_wm_rerank/fallback_frac"], 0.0)

    def test_invalid_uncertainty_is_one(self):
        acts = torch.randn(2, 3, 4)
        belief = torch.randn(2, 5)
        _p_rec, _p_succ, _p_drop, unc, valid, _diag = wm_head_probs_for_candidates(
            belief_flat=belief, cand_first_actions=acts, predict_fn=None
        )
        self.assertFalse(bool(valid.any()))
        self.assertTrue(torch.equal(unc, torch.ones_like(unc)))


if __name__ == "__main__":
    unittest.main()
