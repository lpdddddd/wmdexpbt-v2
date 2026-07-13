"""Tests for snippet WM reranker (row-level fail-closed selection)."""

from __future__ import annotations

import unittest

import torch

from dexpbt_isaaclab.world_model.progress_conditioned_refinement import (
    ProgressRefinementConfig,
    teacher_confidence,
)
from dexpbt_isaaclab.world_model.snippet_wm_reranker import (
    SnippetWMRerankConfig,
    effective_alpha_wm,
    mean_squared_action_delta,
    normalize_priority_scores,
    score_snippet_candidates,
    wm_head_probs_for_candidates,
)


def _score(**kw):
    return score_snippet_candidates(**kw)


def _assert_fallback_norm(testcase, best_idx, cand_score, pri):
    """Fallback uses pr_norm: same ranking as raw priority, finite normalized scores."""
    testcase.assertTrue(torch.equal(best_idx, pri.argmax(dim=1)))
    testcase.assertTrue(torch.equal(best_idx, cand_score.argmax(dim=1)))
    testcase.assertTrue(torch.isfinite(cand_score).all())
    testcase.assertTrue(bool((cand_score >= -1e-5).all().item()))
    testcase.assertTrue(bool((cand_score <= 1.0 + 1e-5).all().item()))


class TestNormalizePriorityScores(unittest.TestCase):
    def test_finite_argmax_matches_raw(self):
        pri = torch.tensor([[1.0, 5.0, 2.0], [3.0, 1.0, 4.0]])
        norm = normalize_priority_scores(pri)
        self.assertTrue(torch.isfinite(norm).all())
        self.assertTrue(torch.equal(norm.argmax(dim=1), pri.argmax(dim=1)))
        self.assertTrue(torch.allclose(norm[0], torch.tensor([0.0, 1.0, 0.25]), atol=1e-5))

    def test_constant_finite_row_ones(self):
        pri = torch.tensor([[3.0, 3.0, 3.0]])
        norm = normalize_priority_scores(pri)
        self.assertTrue(torch.isfinite(norm).all())
        self.assertTrue(torch.allclose(norm, torch.ones(1, 3), atol=1e-5))
        self.assertEqual(int(norm.argmax(dim=1).item()), 0)

    def test_nan_plus_finite_selects_finite(self):
        pri = torch.tensor([[float("nan"), 3.0]])
        norm = normalize_priority_scores(pri)
        self.assertTrue(torch.isfinite(norm).all())
        self.assertEqual(int(norm.argmax(dim=1).item()), 1)
        self.assertAlmostEqual(float(norm[0, 0].item()), 0.0, places=5)
        self.assertAlmostEqual(float(norm[0, 1].item()), 1.0, places=5)

    def test_nan_plus_constant_finites_select_among_finite(self):
        pri = torch.tensor([[float("nan"), 3.0, 3.0]])
        norm = normalize_priority_scores(pri)
        self.assertTrue(torch.isfinite(norm).all())
        self.assertAlmostEqual(float(norm[0, 0].item()), 0.0, places=5)
        self.assertTrue(torch.allclose(norm[0, 1:], torch.ones(2), atol=1e-5))
        self.assertIn(int(norm.argmax(dim=1).item()), (1, 2))

    def test_single_nan_does_not_poison_row(self):
        pri = torch.tensor([[1.0, float("nan"), 3.0]])
        norm = normalize_priority_scores(pri)
        self.assertTrue(torch.isfinite(norm).all())
        self.assertAlmostEqual(float(norm[0, 1].item()), 0.0, places=5)
        self.assertEqual(int(norm.argmax(dim=1).item()), 2)
        self.assertAlmostEqual(float(norm[0, 2].item()), 1.0, places=5)

    def test_all_nonfinite_row_zeros_index0(self):
        pri = torch.tensor([[float("nan"), float("inf"), float("-inf")]])
        norm = normalize_priority_scores(pri)
        self.assertTrue(torch.isfinite(norm).all())
        self.assertTrue(torch.allclose(norm, torch.zeros(1, 3), atol=1e-5))
        self.assertEqual(int(norm.argmax(dim=1).item()), 0)

    def test_bf16_input_finite(self):
        pri = torch.tensor([[1.0, 5.0, 2.0]], dtype=torch.bfloat16)
        norm = normalize_priority_scores(pri)
        self.assertEqual(norm.dtype, torch.bfloat16)
        self.assertTrue(torch.isfinite(norm.float()).all())
        self.assertEqual(int(norm.argmax(dim=1).item()), 1)

    def test_affine_invariance_exact_span(self):
        pri = torch.tensor([[1.0, 5.0, 2.0]])
        self.assertTrue(
            torch.allclose(
                normalize_priority_scores(pri).float(),
                normalize_priority_scores(7.0 * pri + 3.0).float(),
                atol=1e-6,
            )
        )

    def test_disabled_reranker_path_affine_and_action_delta_ignored(self):
        """Simulates progress default branch: normalize only (no WM / action_delta)."""
        pri = torch.tensor([[1.0, 5.0, 2.0]])
        pri_aff = 4.0 * pri + 9.0
        action_delta = torch.tensor([[0.0, 1000.0, 0.0]])  # would flip WM path
        for p in (pri, pri_aff):
            score = normalize_priority_scores(p)
            best = score.argmax(dim=1)
            self.assertEqual(int(best.item()), 1)
            self.assertTrue(torch.equal(best, score.argmax(dim=1)))
        conf_cfg = ProgressRefinementConfig(
            enabled=True, min_teacher_confidence=0.0, teacher_margin_temperature=0.05
        )
        s0 = normalize_priority_scores(pri)
        s1 = normalize_priority_scores(pri_aff)
        self.assertTrue(torch.allclose(s0.float(), s1.float(), atol=1e-5))
        ar = torch.arange(1)
        actor_idx = torch.tensor([0])
        c0, _ = teacher_confidence(
            score_best=s0[ar, s0.argmax(dim=1)],
            score_actor=s0[ar, actor_idx],
            cfg=conf_cfg,
        )
        c1, _ = teacher_confidence(
            score_best=s1[ar, s1.argmax(dim=1)],
            score_actor=s1[ar, actor_idx],
            cfg=conf_cfg,
        )
        self.assertAlmostEqual(float(c0.item()), float(c1.item()), places=5)
        _ = action_delta


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
        """Invalid candidate has highest priority; row must still use priority ranking."""
        cfg = SnippetWMRerankConfig(enabled=True, w_rec=2.0, score_cap=2.0, w_delta=0.0)
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
        _assert_fallback_norm(self, best_idx, cand_score, pri)
        self.assertEqual(int(best_idx.item()), 0)
        self.assertEqual(diag["snippet_wm_rerank/row_active_frac"], 0.0)
        self.assertEqual(diag["snippet_wm_rerank/changed_frac"], 0.0)
        self.assertEqual(diag["snippet_wm_rerank/fallback_reason_id"], 12.0)

    def test_partial_valid_invalid_actor_proxy_no_score_mismatch(self):
        """Invalid cand closest to actor must not participate via WM scores."""
        cfg = SnippetWMRerankConfig(enabled=True, w_rec=2.0, score_cap=2.0, w_delta=0.05)
        pri = torch.tensor([[1.0, 5.0, 2.0]])
        p_rec = torch.tensor([[0.9, 0.1, 0.1]])
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
        _assert_fallback_norm(self, best_idx, cand_score, pri)
        self.assertEqual(int(best_idx.item()), 1)
        ar = torch.arange(1)
        score_selected = cand_score[ar, best_idx]
        actor_like_idx = action_delta.argmin(dim=1)
        score_actor_proxy = cand_score[ar, actor_like_idx]
        # Same candidate_score family (normalized priority).
        self.assertAlmostEqual(float(score_selected.item()), 1.0, places=5)
        self.assertAlmostEqual(float(score_actor_proxy.item()), 0.0, places=5)

    def test_alpha_zero_ignores_action_delta(self):
        cfg = SnippetWMRerankConfig(enabled=True, w_rec=1.0, w_delta=10.0, score_cap=1.0)
        pri = torch.tensor([[1.0, 5.0, 2.0]])
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
        _assert_fallback_norm(self, best_idx, cand_score, pri)
        self.assertEqual(int(best_idx.item()), 1)
        self.assertEqual(diag["snippet_wm_rerank/changed_frac"], 0.0)
        self.assertEqual(diag["snippet_wm_rerank/wm_term_mean"], 0.0)
        self.assertEqual(diag["snippet_wm_rerank/fallback_reason_id"], 11.0)

    def test_fallback_affine_priority_confidence_invariant(self):
        """Positive affine transform of priority leaves fallback confidence unchanged."""
        cfg = SnippetWMRerankConfig(enabled=True)
        pri = torch.tensor([[1.0, 5.0, 2.0]])
        pri_aff = 3.0 * pri + 7.0
        action_delta = torch.tensor([[0.5, 0.0, 0.2]])  # actor-like = idx1 (also best)
        zeros = torch.zeros(1, 3)
        kwargs = dict(
            p_recovery=zeros,
            p_success=zeros,
            p_drop=zeros,
            uncertainty=torch.ones(1, 3),
            action_delta=action_delta,
            alpha_wm=0.0,
            cfg=cfg,
            wm_valid_mask=torch.ones(1, 3, dtype=torch.bool),
        )
        best_a, score_a, _ = _score(priority=pri, **kwargs)
        best_b, score_b, _ = _score(priority=pri_aff, **kwargs)
        self.assertTrue(torch.equal(best_a, best_b))
        self.assertTrue(torch.allclose(score_a, score_b, atol=1e-5))
        ar = torch.arange(1)
        conf_cfg = ProgressRefinementConfig(
            enabled=True, min_teacher_confidence=0.0, teacher_margin_temperature=0.05
        )
        # Force a non-zero margin: actor proxy = lowest priority cand.
        actor_idx = torch.tensor([0])
        c_a, _ = teacher_confidence(
            score_best=score_a[ar, best_a],
            score_actor=score_a[ar, actor_idx],
            cfg=conf_cfg,
        )
        c_b, _ = teacher_confidence(
            score_best=score_b[ar, best_b],
            score_actor=score_b[ar, actor_idx],
            cfg=conf_cfg,
        )
        self.assertAlmostEqual(float(c_a.item()), float(c_b.item()), places=5)

    def test_constant_priority_finite(self):
        cfg = SnippetWMRerankConfig(enabled=True)
        pri = torch.tensor([[3.0, 3.0, 3.0]])
        best_idx, cand_score, _ = _score(
            priority=pri,
            p_recovery=torch.zeros(1, 3),
            p_success=torch.zeros(1, 3),
            p_drop=torch.zeros(1, 3),
            uncertainty=torch.ones(1, 3),
            action_delta=torch.zeros(1, 3),
            alpha_wm=0.0,
            cfg=cfg,
        )
        self.assertTrue(torch.isfinite(cand_score).all())
        self.assertTrue(torch.equal(best_idx, cand_score.argmax(dim=1)))
        self.assertTrue(torch.allclose(cand_score, torch.ones(1, 3), atol=1e-5))

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
        _assert_fallback_norm(self, best_idx, cand_score, pri)
        self.assertEqual(int(best_idx.item()), 0)
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
        self.assertTrue(torch.equal(best_idx[1:2], pri[1:2].argmax(dim=1)))
        self.assertTrue(torch.equal(best_idx, cand_score.argmax(dim=1)))
        self.assertTrue(torch.isfinite(cand_score[1]).all())
        self.assertAlmostEqual(diag["snippet_wm_rerank/row_active_frac"], 0.5, places=5)
        self.assertAlmostEqual(diag["snippet_wm_rerank/row_fallback_frac"], 0.5, places=5)
        self.assertEqual(diag["snippet_wm_rerank/fallback_reason_id"], 12.0)


class TestWMScoreContinuityAndReasons(unittest.TestCase):
    def test_mean_action_delta_dim_invariant(self):
        cand = torch.randn(2, 3, 4)
        mu = torch.randn(2, 4)
        d4 = mean_squared_action_delta(cand, mu)
        cand_dup = torch.cat([cand, cand], dim=-1)
        mu_dup = torch.cat([mu, mu], dim=-1)
        d8 = mean_squared_action_delta(cand_dup, mu_dup)
        self.assertTrue(torch.allclose(d4, d8, atol=1e-5))

    def test_alpha_zero_ignores_action_delta_exactly(self):
        cfg = SnippetWMRerankConfig(enabled=True, w_rec=1.0, w_delta=10.0, score_cap=1.0)
        pri = torch.tensor([[1.0, 5.0, 2.0]])
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
        pr_norm = normalize_priority_scores(pri)
        self.assertTrue(torch.allclose(cand_score, pr_norm, atol=1e-6))
        self.assertEqual(int(best_idx.item()), 1)
        self.assertEqual(diag["snippet_wm_rerank/fallback_reason_id"], 11.0)

    def test_alpha_continuity_toward_pr_norm(self):
        cfg = SnippetWMRerankConfig(enabled=True, w_rec=0.5, w_delta=0.2, score_cap=1.0)
        pri = torch.tensor([[1.0, 2.0, 3.0]])
        p_rec = torch.tensor([[0.2, 0.5, 0.8]])
        action_delta = torch.tensor([[0.5, 0.1, 0.0]])
        base = dict(
            priority=pri,
            p_recovery=p_rec,
            p_success=torch.zeros(1, 3),
            p_drop=torch.zeros(1, 3),
            uncertainty=torch.zeros(1, 3),
            action_delta=action_delta,
            cfg=cfg,
            wm_valid_mask=torch.ones(1, 3, dtype=torch.bool),
        )
        _, s0, _ = _score(alpha_wm=0.0, **base)
        _, s_tiny, _ = _score(alpha_wm=1e-8, **base)
        _, s_small, _ = _score(alpha_wm=0.1, **base)
        _, s1, _ = _score(alpha_wm=1.0, **base)
        self.assertTrue(torch.allclose(s0, normalize_priority_scores(pri), atol=1e-6))
        self.assertTrue(torch.allclose(s_tiny, s0, atol=1e-5))
        # Monotone drift away from pr_norm as alpha grows (L2 distance).
        d_tiny = float((s_tiny - s0).pow(2).sum().item())
        d_small = float((s_small - s0).pow(2).sum().item())
        d1 = float((s1 - s0).pow(2).sum().item())
        self.assertLessEqual(d_tiny, d_small + 1e-9)
        self.assertLessEqual(d_small, d1 + 1e-9)

    def test_whole_valid_plus_whole_invalid_reason_6(self):
        cfg = SnippetWMRerankConfig(enabled=True, w_rec=1.0, score_cap=1.0, w_delta=0.0)
        pri = torch.tensor([[1.0, 1.0, 1.0], [9.0, 1.0, 2.0]])
        valid = torch.tensor([[True, True, True], [False, False, False]])
        _, _, diag = _score(
            priority=pri,
            p_recovery=torch.zeros(2, 3),
            p_success=torch.zeros(2, 3),
            p_drop=torch.zeros(2, 3),
            uncertainty=torch.zeros(2, 3),
            action_delta=torch.zeros(2, 3),
            alpha_wm=1.0,
            cfg=cfg,
            wm_valid_mask=valid,
        )
        self.assertEqual(diag["snippet_wm_rerank/fallback_reason_id"], 6.0)

    def test_within_row_partial_reason_12(self):
        cfg = SnippetWMRerankConfig(enabled=True)
        _, _, diag = _score(
            priority=torch.tensor([[1.0, 2.0, 3.0]]),
            p_recovery=torch.zeros(1, 3),
            p_success=torch.zeros(1, 3),
            p_drop=torch.zeros(1, 3),
            uncertainty=torch.zeros(1, 3),
            action_delta=torch.zeros(1, 3),
            alpha_wm=1.0,
            cfg=cfg,
            wm_valid_mask=torch.tensor([[True, False, True]]),
        )
        self.assertEqual(diag["snippet_wm_rerank/fallback_reason_id"], 12.0)

    def test_bad_action_row_does_not_kill_batch(self):
        acts = torch.randn(2, 2, 3)
        acts[0, 0, 0] = float("nan")
        belief = torch.randn(2, 4)

        def _ok(belief_rep, cand_flat):
            n = belief_rep.shape[0]
            return {
                "recovery_logits": torch.zeros(n),
                "horizon_logits": {1: {"success": torch.zeros(n), "drop": torch.zeros(n)}},
            }

        _p_rec, _p_succ, _p_drop, unc, valid, diag = wm_head_probs_for_candidates(
            belief_flat=belief, cand_first_actions=acts, predict_fn=_ok
        )
        self.assertFalse(bool(valid[0].any()))
        self.assertTrue(bool(valid[1].all()))
        self.assertTrue(torch.equal(unc[0], torch.ones(2)))
        # Row1 remains usable for scoring.
        best_idx, _, score_diag = score_snippet_candidates(
            priority=torch.tensor([[1.0, 2.0], [3.0, 1.0]]),
            p_recovery=_p_rec,
            p_success=_p_succ,
            p_drop=_p_drop,
            uncertainty=unc,
            action_delta=torch.zeros(2, 2),
            alpha_wm=1.0,
            cfg=SnippetWMRerankConfig(enabled=True, w_rec=1.0, score_cap=1.0),
            wm_valid_mask=valid,
        )
        self.assertAlmostEqual(score_diag["snippet_wm_rerank/row_active_frac"], 0.5, places=5)
        self.assertEqual(int(best_idx[1].item()), 0)
        self.assertEqual(diag["snippet_wm_rerank/fallback_reason_id"], 8.0)


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
        pri = torch.tensor([[1.0, 9.0, 2.0]])
        best_idx, cand_score, score_diag = score_snippet_candidates(
            priority=pri,
            p_recovery=p_rec,
            p_success=p_succ,
            p_drop=p_drop,
            uncertainty=unc,
            action_delta=torch.zeros(1, 3),
            alpha_wm=1.0,
            cfg=SnippetWMRerankConfig(enabled=True, w_rec=2.0, score_cap=2.0),
            wm_valid_mask=valid,
        )
        _assert_fallback_norm(self, best_idx, cand_score, pri)
        self.assertEqual(int(best_idx.item()), 1)
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
