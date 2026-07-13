"""Tests for WM validity + teacher-confidence/score alignment."""

from __future__ import annotations

import unittest

import torch
import torch.nn as nn

from dexpbt_isaaclab.world_model.elite_snippet_bank import (
    EliteSnippetBank,
    EliteSnippetBankConfig,
)
from dexpbt_isaaclab.world_model.elite_snippet_builder import SnippetBatch
from dexpbt_isaaclab.world_model.progress_conditioned_refinement import (
    ProgressRefinementConfig,
    teacher_confidence,
)
from dexpbt_isaaclab.world_model.snippet_distill import SnippetDistillConfig
from dexpbt_isaaclab.world_model.snippet_wm_reranker import (
    SnippetWMRerankConfig,
    normalize_priority_scores,
    score_snippet_candidates,
    wm_head_probs_for_candidates,
)


def _snippet_batch(tols, *, feat_dim=8, action_dim=3, horizon=4, norm_err=None):
    n = len(tols)
    if norm_err is None:
        norm_err = [0.5] * n
    return SnippetBatch(
        feat_start=torch.randn(n, feat_dim),
        actions=torch.randn(n, horizon, action_dim).clamp(-1, 1),
        d_hat=torch.full((n,), 0.02),
        success_tol=torch.tensor([float(t) for t in tols]),
        lifted=torch.ones(n),
        picked=torch.zeros(n),
        dropped=torch.zeros(n),
        closest_d_hat=torch.full((n,), 0.01),
        success_h=torch.ones(n),
        improve_h=torch.ones(n),
        drop_h=torch.zeros(n),
        recovery_label=torch.ones(n),
        s0=torch.ones(n),
        s1=torch.ones(n),
        i0=torch.ones(n),
        d0=torch.zeros(n),
        d1=torch.zeros(n),
        normalized_error=torch.tensor([float(x) for x in norm_err]),
        recent_progress=torch.zeros(n),
        oscillation=torch.full((n,), 0.1),
        progress_history_valid=torch.ones(n),
        normalized_improvement=torch.full((n,), 0.5),
    )


def _bank(**over) -> EliteSnippetBank:
    cfg = EliteSnippetBankConfig(
        enabled=True,
        feature_dim=8,
        action_dim=3,
        horizon=4,
        horizons=(4, 8),
        min_priority=0.0,
        **over,
    )
    return EliteSnippetBank(cfg)


class _A2C(nn.Module):
    def __init__(self, f=8, a=3):
        super().__init__()
        self.mu = nn.Linear(f, a)


class _Inner:
    def __init__(self, a2c):
        self.a2c_network = a2c


class _Model:
    def __init__(self, a2c):
        self.model = _Inner(a2c)


class _Agent:
    def __init__(self, a2c):
        self.model = _Model(a2c)


class _FakeBridge:
    def __init__(self, bank, sd_cfg, pr_cfg, a2c, *, rerank_cfg=None, predict_fn=None, alpha_wm=1.0):
        self.snippet_bank = bank
        self.snippet_distill_cfg = sd_cfg
        self.progress_refine_cfg = pr_cfg
        self.snippet_wm_rerank_cfg = rerank_cfg
        self.wm_calibration_enable = bool(rerank_cfg is not None) and float(alpha_wm) > 0.0
        self.wm_calibration_state = (
            {"alpha_wm": float(alpha_wm)} if rerank_cfg is not None else None
        )
        self.agent_ref = _Agent(a2c)
        self.rssm_belief_cache = None
        self.rssm_belief_provider = None
        self.rssm_predict_fn = predict_fn
        self.capture = None
        self.last_progress_refine = {}
        self.last_snippet_distill_loss = 0.0
        self.last_snippet_distill_active_frac = 0.0
        self.last_snippet_wm_rerank_changed_frac = 0.0
        self.last_snippet_wm_rerank_wm_term_mean = 0.0
        self.last_snippet_wm_rerank_valid_frac = 0.0
        self.last_snippet_wm_rerank_fallback_frac = 0.0
        self.last_snippet_wm_rerank_predict_error_count = 0.0
        self.last_snippet_retrieval_valid = 0.0
        self.last_snippet_no_tolerance_compatible = 0.0
        self.last_actor_distill_fallback_reason = ""


class _FakeBeliefCache:
    def __init__(self, belief: torch.Tensor):
        self.valid = True
        self._belief = belief

    def slice_rows(self, start: int, end: int):
        return self._belief[start:end]


class _FakeCache:
    pass


class TestScoreAlignment(unittest.TestCase):
    def test_best_idx_confidence_share_candidate_score(self):
        cfg = SnippetWMRerankConfig(enabled=True, w_rec=1.0, score_cap=1.0, w_delta=0.0)
        pri = torch.tensor([[1.0, 5.0, 2.0]])
        p_rec = torch.tensor([[0.9, 0.1, 0.2]])
        best_idx, cand_score, _, _row = score_snippet_candidates(
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
        ar = torch.arange(1)
        score_selected = cand_score[ar, best_idx]
        self.assertAlmostEqual(
            float(score_selected.item()), float(cand_score.max(dim=1).values.item()), places=5
        )
        self.assertTrue(torch.equal(best_idx, cand_score.argmax(dim=1)))

    def test_wm_rerank_low_base_priority_uses_selected_score(self):
        """WM can pick a low-priority candidate; confidence must use that score."""
        cfg = SnippetWMRerankConfig(enabled=True, w_rec=2.0, score_cap=2.0, w_delta=0.0)
        pri = torch.tensor([[10.0, 1.0, 2.0]])  # priority prefers idx 0
        p_rec = torch.tensor([[0.0, 1.0, 0.0]])  # WM prefers idx 1
        best_idx, cand_score, diag, _row = score_snippet_candidates(
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
        self.assertEqual(int(best_idx.item()), 1)
        self.assertGreater(diag["snippet_wm_rerank/changed_frac"], 0.0)
        score_selected = cand_score[0, best_idx]
        conf, _ = teacher_confidence(
            score_best=score_selected,
            score_actor=cand_score[0, torch.tensor([0])],
            cfg=ProgressRefinementConfig(teacher_margin_temperature=0.05),
        )
        self.assertTrue(torch.isfinite(conf).all())
        self.assertNotAlmostEqual(float(score_selected.item()), float(pri.max().item()), places=5)


class TestWMValidity(unittest.TestCase):
    def test_predict_fn_none_invalid_mask(self):
        acts = torch.randn(2, 3, 4)
        belief = torch.randn(2, 5)
        p_rec, p_succ, p_drop, unc, valid, diag = wm_head_probs_for_candidates(
            belief_flat=belief, cand_first_actions=acts, predict_fn=None
        )
        self.assertFalse(bool(valid.any()))
        self.assertEqual(diag["snippet_wm_rerank/fallback_reason_id"], 1.0)
        self.assertTrue(torch.equal(unc, torch.ones_like(unc)))

        best_idx, cand_score, score_diag, _row = score_snippet_candidates(
            priority=torch.tensor([[1.0, 5.0, 2.0], [3.0, 1.0, 4.0]]),
            p_recovery=p_rec,
            p_success=p_succ,
            p_drop=p_drop,
            uncertainty=unc,
            action_delta=torch.zeros(2, 3),
            alpha_wm=1.0,
            cfg=SnippetWMRerankConfig(enabled=True),
            wm_valid_mask=valid,
        )
        pri = torch.tensor([[1.0, 5.0, 2.0], [3.0, 1.0, 4.0]])
        self.assertTrue(torch.equal(best_idx, pri.argmax(dim=1)))
        self.assertTrue(torch.equal(best_idx, cand_score.argmax(dim=1)))
        self.assertEqual(score_diag["snippet_wm_rerank/fallback_frac"], 1.0)
        self.assertTrue(torch.isfinite(cand_score).all())
        self.assertTrue(bool((cand_score >= -1e-5).all().item()))
        self.assertTrue(bool((cand_score <= 1.0 + 1e-5).all().item()))

    def test_predict_exception_no_crash_priority_fallback(self):
        acts = torch.randn(1, 2, 3)
        belief = torch.randn(1, 4)

        def _boom(_b, _a):
            raise RuntimeError("wm down")

        p_rec, p_succ, p_drop, unc, valid, diag = wm_head_probs_for_candidates(
            belief_flat=belief, cand_first_actions=acts, predict_fn=_boom
        )
        self.assertEqual(diag["snippet_wm_rerank/predict_error_count"], 1.0)
        self.assertFalse(bool(valid.any()))
        self.assertTrue(torch.equal(unc, torch.ones_like(unc)))
        best_idx, _, score_diag, _row = score_snippet_candidates(
            priority=torch.tensor([[2.0, 9.0]]),
            p_recovery=p_rec,
            p_success=p_succ,
            p_drop=p_drop,
            uncertainty=unc,
            action_delta=torch.zeros(1, 2),
            alpha_wm=1.0,
            cfg=SnippetWMRerankConfig(enabled=True, w_unc=1.0),
            wm_valid_mask=valid,
        )
        self.assertEqual(int(best_idx.item()), 1)
        self.assertEqual(score_diag["snippet_wm_rerank/changed_frac"], 0.0)

    def test_nan_inf_candidates_row_fail_closed(self):
        acts = torch.randn(1, 3, 2)
        belief = torch.randn(1, 3)

        def _bad(_b, _a):
            return {
                "recovery_logits": torch.tensor([10.0, float("nan"), 0.0]),
                "horizon_logits": {
                    1: {
                        "success": torch.tensor([0.0, 0.0, 0.0]),
                        "drop": torch.tensor([0.0, 0.0, float("nan")]),
                    }
                },
            }

        p_rec, p_succ, p_drop, unc, valid, diag = wm_head_probs_for_candidates(
            belief_flat=belief, cand_first_actions=acts, predict_fn=_bad
        )
        self.assertFalse(bool(valid[0, 1]))
        self.assertFalse(bool(valid[0, 2]))
        self.assertTrue(bool(valid[0, 0]))
        self.assertAlmostEqual(float(unc[0, 1].item()), 1.0, places=5)
        pri = torch.tensor([[1.0, 8.0, 2.0]])
        best_idx, cand_score, score_diag, _row = score_snippet_candidates(
            priority=pri,
            p_recovery=p_rec,
            p_success=p_succ,
            p_drop=p_drop,
            uncertainty=unc,
            action_delta=torch.tensor([[0.0, 1.0, 2.0]]),
            alpha_wm=1.0,
            cfg=SnippetWMRerankConfig(enabled=True, w_rec=2.0, score_cap=2.0, w_delta=0.1),
            wm_valid_mask=valid,
        )
        self.assertEqual(int(best_idx.item()), 1)
        self.assertTrue(torch.equal(best_idx, pri.argmax(dim=1)))
        self.assertTrue(torch.equal(best_idx, cand_score.argmax(dim=1)))
        self.assertTrue(torch.isfinite(cand_score).all())
        self.assertEqual(score_diag["snippet_wm_rerank/row_active_frac"], 0.0)
        self.assertGreater(diag["snippet_wm_rerank/fallback_frac"], 0.0)

    def test_partial_row_valid_mixed_batch(self):
        cfg = SnippetWMRerankConfig(enabled=True, w_rec=2.0, score_cap=2.0, w_delta=0.0)
        pri = torch.tensor([[1.0, 1.0, 1.0], [9.0, 1.0, 2.0]])
        p_rec = torch.tensor([[0.9, 0.1, 0.1], [0.1, 0.9, 0.1]])
        valid = torch.tensor([[True, True, True], [False, False, False]])
        best_idx, cand_score, diag, _row = score_snippet_candidates(
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
        self.assertEqual(int(best_idx[0].item()), 0)
        self.assertEqual(int(best_idx[1].item()), 0)
        self.assertAlmostEqual(diag["snippet_wm_rerank/row_active_frac"], 0.5, places=5)
        self.assertTrue(torch.equal(best_idx[1:2], pri[1:2].argmax(dim=1)))
        self.assertTrue(torch.equal(best_idx, cand_score.argmax(dim=1)))
        self.assertTrue(torch.isfinite(cand_score[1]).all())
        self.assertTrue(bool((cand_score[1] >= -1e-5).all().item()))
        self.assertTrue(bool((cand_score[1] <= 1.0 + 1e-5).all().item()))
        self.assertEqual(diag["snippet_wm_rerank/fallback_reason_id"], 6.0)


class TestProgressRuntimeWM(unittest.TestCase):
    def _run(
        self,
        *,
        predict_fn,
        belief=None,
        a2c=None,
        alpha_wm=1.0,
        calibration=True,
        rerank_cfg="default",
        patch_retrieve=None,
        feat=None,
        max_active_fraction=0.35,
    ):
        from dexpbt_isaaclab.world_model.eir_mpc import _compute_snippet_distill_loss_progress

        a2c = a2c or _A2C()
        bank = _bank(
            tolerance_compatibility_mode="log_relative",
            tolerance_max_log_gap=0.3,
            progress_priority_weight=0.5,
        )
        bank.update_from_builder_batch(_snippet_batch([0.02, 0.021, 0.019]), frame=1)
        sd = SnippetDistillConfig(enabled=True, weight=1.0, topk=4, loss_cap=0.0)
        pr = ProgressRefinementConfig(
            enabled=True,
            history_steps=1,
            start_after_frames=0,
            ramp_frames=0,
            only_lifted=False,
            min_gate_for_retrieval=0.0,
            min_teacher_confidence=0.0,
            max_active_fraction=float(max_active_fraction),
        )
        if rerank_cfg == "default":
            rerank = SnippetWMRerankConfig(enabled=True, w_rec=1.0, score_cap=1.0)
        else:
            rerank = rerank_cfg
        bridge = _FakeBridge(
            bank, sd, pr, a2c, rerank_cfg=rerank, predict_fn=predict_fn, alpha_wm=alpha_wm
        )
        if not calibration:
            bridge.wm_calibration_enable = False
        if patch_retrieve is not None:
            bank.retrieve_candidates_batched = patch_retrieve  # type: ignore[method-assign]
        R = 4
        if belief is not None:
            bridge.rssm_belief_cache = _FakeBeliefCache(belief)
        cache = _FakeCache()
        cache.success_tol_flat = torch.full((R,), 0.02)
        cache.lifted_flat = torch.ones(R)
        cache.normalized_error_flat = torch.tensor([0.2, 0.5, 0.3, 0.25])
        cache.recent_progress_flat = torch.zeros(R)
        cache.oscillation_flat = torch.full((R,), 0.2)
        cache.progress_history_valid_flat = torch.ones(R)
        feat = torch.zeros(R, 8) if feat is None else feat
        risky = torch.ones(R, dtype=torch.bool)
        loss = _compute_snippet_distill_loss_progress(
            bridge,
            eir_cfg=None,
            cache=cache,
            feat=feat,
            risky=risky,
            frames=10**9,
            start=0,
            end=R,
        )
        return loss, bridge

    def _make_retrieve(self, priority: torch.Tensor):
        """Controlled retrieve returning fixed priority (and matching action shapes)."""
        from types import SimpleNamespace

        n, k = priority.shape
        a_dim = 3
        actions = torch.zeros(n, k, 4, a_dim)
        # Distinct first actions so action_delta is non-uniform if ever used.
        for j in range(k):
            actions[:, j, 0, :] = float(j)
        valid = torch.ones(n)

        def _retrieve(feat, **kwargs):
            nn = int(feat.shape[0])
            return SimpleNamespace(
                actions=actions[:nn].clone(),
                priority=priority[:nn].clone(),
                valid_mask=valid[:nn].clone(),
                diagnostics={},
            )

        return _retrieve

    def test_predict_none_runtime_fallback_finite(self):
        loss, bridge = self._run(predict_fn=None, belief=torch.randn(4, 5))
        self.assertTrue(torch.isfinite(loss.float()))
        self.assertEqual(bridge.last_progress_refine["snippet_wm_rerank/valid_frac"], 0.0)
        self.assertEqual(bridge.last_progress_refine["snippet_wm_rerank/fallback_frac"], 1.0)
        self.assertEqual(bridge.last_progress_refine["snippet_wm_rerank/fallback_reason_id"], 1.0)

    def test_predict_exception_runtime_finite(self):
        def _boom(_b, _a):
            raise RuntimeError("boom")

        loss, bridge = self._run(predict_fn=_boom, belief=torch.randn(4, 5))
        self.assertTrue(torch.isfinite(loss.float()))
        self.assertEqual(bridge.last_progress_refine["snippet_wm_rerank/predict_error_count"], 1.0)
        self.assertEqual(bridge.last_progress_refine["snippet_wm_rerank/fallback_frac"], 1.0)
        self.assertEqual(bridge.last_progress_refine["snippet_wm_rerank/fallback_reason_id"], 4.0)

    def test_belief_missing_no_feat_substitute(self):
        def _ok(belief_rep, cand_flat):
            n = belief_rep.shape[0]
            return {
                "recovery_logits": torch.zeros(n),
                "horizon_logits": {
                    1: {"success": torch.zeros(n), "drop": torch.zeros(n)},
                },
            }

        loss, bridge = self._run(predict_fn=_ok, belief=None)
        self.assertTrue(torch.isfinite(loss.float()))
        self.assertEqual(bridge.last_progress_refine["snippet_wm_rerank/valid_frac"], 0.0)
        self.assertEqual(bridge.last_progress_refine["snippet_wm_rerank/fallback_reason_id"], 2.0)

    def test_alpha_zero_runtime_skips_wm(self):
        def _should_not_run(_b, _a):
            raise AssertionError("WM predict must not run when alpha_wm=0")

        loss, bridge = self._run(
            predict_fn=_should_not_run, belief=torch.randn(4, 5), alpha_wm=0.0
        )
        self.assertTrue(torch.isfinite(loss.float()))
        self.assertEqual(bridge.last_progress_refine["snippet_wm_rerank/fallback_reason_id"], 11.0)
        self.assertEqual(bridge.last_progress_refine["snippet_wm_rerank/row_active_frac"], 0.0)
        self.assertEqual(bridge.last_progress_refine["snippet_wm_rerank/wm_term_mean"], 0.0)

    def test_uncalibrated_runtime_skips_wm(self):
        def _should_not_run(_b, _a):
            raise AssertionError("WM predict must not run when uncalibrated")

        loss, bridge = self._run(
            predict_fn=_should_not_run,
            belief=torch.randn(4, 5),
            alpha_wm=1.0,
            calibration=False,
        )
        self.assertTrue(torch.isfinite(loss.float()))
        self.assertEqual(bridge.last_progress_refine["snippet_wm_rerank/fallback_reason_id"], 11.0)

    def test_rerank_none_skips_wm_uses_normalized_priority(self):
        def _should_not_run(_b, _a):
            raise AssertionError("WM predict must not run when rerank_cfg is None")

        a2c = _A2C()
        feat = torch.zeros(4, 8)
        belief = torch.zeros(4, 5)
        pri = torch.tensor([[1.0, 5.0, 2.0, 0.5]]).expand(4, 4).contiguous()
        loss, bridge = self._run(
            predict_fn=_should_not_run,
            belief=belief,
            a2c=a2c,
            feat=feat,
            rerank_cfg=None,
            patch_retrieve=self._make_retrieve(pri),
        )
        self.assertTrue(torch.isfinite(loss.float()))
        self.assertEqual(bridge.last_progress_refine["snippet_wm_rerank/row_active_frac"], 0.0)
        self.assertEqual(bridge.last_progress_refine["snippet_wm_rerank/wm_term_mean"], 0.0)
        self.assertAlmostEqual(
            bridge.last_progress_refine["snippet_wm_rerank/selected_score_mean"], 1.0, places=5
        )
        self.assertEqual(
            bridge.last_progress_refine["snippet_wm_rerank/fallback_reason_id"], 11.0
        )
        loss2, bridge2 = self._run(
            predict_fn=_should_not_run,
            belief=belief,
            a2c=a2c,
            feat=feat,
            rerank_cfg=None,
            patch_retrieve=self._make_retrieve(3.0 * pri + 11.0),
        )
        self.assertTrue(torch.isfinite(loss2.float()))
        self.assertAlmostEqual(
            bridge.last_progress_refine["snippet_wm_rerank/selected_score_mean"],
            bridge2.last_progress_refine["snippet_wm_rerank/selected_score_mean"],
            places=5,
        )
        self.assertAlmostEqual(
            bridge.last_progress_refine["snippet_wm_rerank/actor_proxy_score_mean"],
            bridge2.last_progress_refine["snippet_wm_rerank/actor_proxy_score_mean"],
            places=5,
        )
        self.assertAlmostEqual(
            bridge.last_progress_refine["progress_refine/teacher_confidence_mean"],
            bridge2.last_progress_refine["progress_refine/teacher_confidence_mean"],
            places=4,
        )

    def test_rerank_disabled_skips_wm_affine_and_ignores_action_delta(self):
        def _should_not_run(_b, _a):
            raise AssertionError("WM predict must not run when reranker disabled")

        a2c = _A2C()
        feat = torch.zeros(4, 8)
        belief = torch.zeros(4, 5)
        # Highest priority at idx1; action_delta would prefer idx0 if applied.
        pri = torch.tensor([[1.0, 9.0, 2.0, 0.5]]).expand(4, 4).contiguous()
        disabled = SnippetWMRerankConfig(enabled=False, w_delta=100.0)
        loss, bridge = self._run(
            predict_fn=_should_not_run,
            belief=belief,
            a2c=a2c,
            feat=feat,
            rerank_cfg=disabled,
            patch_retrieve=self._make_retrieve(pri),
        )
        self.assertTrue(torch.isfinite(loss.float()))
        self.assertEqual(bridge.last_progress_refine["snippet_wm_rerank/row_active_frac"], 0.0)
        self.assertAlmostEqual(
            bridge.last_progress_refine["snippet_wm_rerank/selected_score_mean"], 1.0, places=5
        )
        self.assertEqual(
            bridge.last_progress_refine["snippet_wm_rerank/fallback_reason_id"], 11.0
        )
        loss2, bridge2 = self._run(
            predict_fn=_should_not_run,
            belief=belief,
            a2c=a2c,
            feat=feat,
            rerank_cfg=disabled,
            patch_retrieve=self._make_retrieve(2.5 * pri + 4.0),
        )
        self.assertAlmostEqual(
            bridge.last_progress_refine["snippet_wm_rerank/selected_score_mean"],
            bridge2.last_progress_refine["snippet_wm_rerank/selected_score_mean"],
            places=5,
        )
        self.assertAlmostEqual(
            bridge.last_progress_refine["progress_refine/teacher_confidence_mean"],
            bridge2.last_progress_refine["progress_refine/teacher_confidence_mean"],
            places=4,
        )
        score = normalize_priority_scores(pri[:1])
        self.assertEqual(int(score.argmax(dim=1).item()), int(pri[:1].argmax(dim=1).item()))

    def test_nonfinite_retrieval_row_zero_weight_teacher_actor(self):
        from dexpbt_isaaclab.world_model import snippet_distill as sd_mod

        a2c = _A2C()
        with torch.no_grad():
            a2c.mu.weight.zero_()
            a2c.mu.bias.zero_()
        feat = torch.zeros(4, 8)
        belief = torch.zeros(4, 5)
        pri = torch.tensor([[1.0, 5.0, 2.0, 0.5]]).expand(4, 4).contiguous().clone()
        pri[0, 0] = float("nan")  # row0 retrieval data invalid
        captured = {}
        orig = sd_mod.compute_snippet_distill_loss

        def _spy(*, mu_fn, feat_start, teacher_first_action, cfg, row_weight=None, **kwargs):
            captured["teacher"] = teacher_first_action.detach().clone()
            captured["weight"] = None if row_weight is None else row_weight.detach().clone()
            captured["mu"] = mu_fn(feat_start).detach().clone()
            return orig(
                mu_fn=mu_fn,
                feat_start=feat_start,
                teacher_first_action=teacher_first_action,
                cfg=cfg,
                row_weight=row_weight,
                **kwargs,
            )

        sd_mod.compute_snippet_distill_loss = _spy  # type: ignore[assignment]
        try:
            loss, bridge = self._run(
                predict_fn=None,
                belief=belief,
                a2c=a2c,
                feat=feat,
                rerank_cfg=None,
                patch_retrieve=self._make_retrieve(pri),
                max_active_fraction=1.0,
            )
        finally:
            sd_mod.compute_snippet_distill_loss = orig  # type: ignore[assignment]

        self.assertTrue(torch.isfinite(loss.float()))
        self.assertAlmostEqual(
            bridge.last_progress_refine["snippet_retrieval/nonfinite_candidate_row_frac"],
            0.25,
            places=5,
        )
        self.assertAlmostEqual(
            bridge.last_progress_refine["progress_refine/fallback_frac"], 0.25, places=5
        )
        self.assertAlmostEqual(
            bridge.last_progress_refine["progress_refine/retrieval_valid_frac"], 0.75, places=5
        )
        self.assertLessEqual(
            bridge.last_progress_refine["progress_refine/active_frac"], 0.75 + 1e-6
        )
        self.assertGreaterEqual(
            bridge.last_progress_refine["progress_refine/nonfinite_count"], 1.0
        )
        self.assertIn("teacher", captured)
        self.assertIn("weight", captured)
        teacher = captured["teacher"]
        weight = captured["weight"]
        mu = captured["mu"]
        self.assertTrue(torch.allclose(teacher[0].float(), mu[0].float(), atol=1e-5))
        self.assertAlmostEqual(float(weight[0].item()), 0.0, places=5)
        self.assertGreater(float((weight[1:] > 0).float().sum().item()), 0.0)

    def test_bf16_actor_fp32_snippet_with_wm_fallback(self):
        class _Bf16A2C(nn.Module):
            def __init__(self):
                super().__init__()
                self._lin = nn.Linear(8, 3)

            def mu(self, f):
                return self._lin(f.float()).to(dtype=torch.bfloat16)

        loss, bridge = self._run(
            predict_fn=None, belief=torch.randn(4, 5), a2c=_Bf16A2C()
        )
        self.assertTrue(torch.isfinite(loss.float()))
        self.assertIn("progress_refine/teacher_margin_mean", bridge.last_progress_refine)
        self.assertIn("snippet_wm_rerank/selected_score_mean", bridge.last_progress_refine)
        self.assertIn("snippet_wm_rerank/row_active_frac", bridge.last_progress_refine)


if __name__ == "__main__":
    unittest.main()
