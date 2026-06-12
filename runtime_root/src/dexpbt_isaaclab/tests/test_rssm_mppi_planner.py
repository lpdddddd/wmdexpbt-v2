"""Unit tests for the EIR-MPC light RSSM reranker (Phase E)."""

from __future__ import annotations

import unittest

import torch

from dexpbt_isaaclab.world_model.rssm_mppi_planner import (
    RSSMReranker,
    RSSMRerankConfig,
    RERANK_OK,
    RERANK_DISABLED,
    RERANK_BEFORE_START,
    RERANK_MODEL_UNAVAILABLE,
    RERANK_HIGH_UNCERTAINTY,
)


def _predict_fn_near(target, horizons=(4, 8), noise=0.0):
    """Heads that predict high success when action near ``target``; low drop."""

    def fn(belief_flat, action_flat):
        d = (action_flat - target.to(action_flat.device)).norm(dim=-1, keepdim=True)
        succ_logit = 5.0 - 10.0 * d
        horizon_logits = {}
        for j, h in enumerate(horizons):
            horizon_logits[h] = {
                "success": succ_logit + (noise * j),
                "drop": torch.full_like(succ_logit, -5.0),
            }
        return {
            "horizon_logits": horizon_logits,
            "recovery_logits": succ_logit,
        }

    return fn


class TestRSSMReranker(unittest.TestCase):
    def _cfg(self, **kw):
        base = dict(enabled=True, start_after_frames=100, uncertainty_threshold=0.4)
        base.update(kw)
        return RSSMRerankConfig(**base)

    def test_disabled_fallback(self):
        r = RSSMReranker(RSSMRerankConfig(enabled=False))
        cand = torch.randn(3, 5, 3).clamp(-1, 1)
        out = r.score_candidates(
            belief_flat=torch.randn(3, 8), cand_actions=cand,
            predict_fn=_predict_fn_near(torch.zeros(3)), frames=10**9,
        )
        self.assertFalse(out["active"])
        self.assertEqual(out["fallback_reason"], RERANK_DISABLED)

    def test_before_start_fallback(self):
        r = RSSMReranker(self._cfg(start_after_frames=10**9))
        cand = torch.randn(2, 4, 3).clamp(-1, 1)
        out = r.score_candidates(
            belief_flat=torch.randn(2, 8), cand_actions=cand,
            predict_fn=_predict_fn_near(torch.zeros(3)), frames=10,
        )
        self.assertEqual(out["fallback_reason"], RERANK_BEFORE_START)

    def test_missing_rssm_fallback(self):
        r = RSSMReranker(self._cfg())
        cand = torch.randn(2, 4, 3).clamp(-1, 1)
        out = r.score_candidates(
            belief_flat=None, cand_actions=cand, predict_fn=None, frames=500,
        )
        self.assertFalse(out["active"])
        self.assertEqual(out["fallback_reason"], RERANK_MODEL_UNAVAILABLE)
        # falls back to actor column
        self.assertTrue(torch.allclose(out["best_action"], cand[:, 0, :]))

    def test_action_score_shape_and_best(self):
        target = torch.tensor([0.6, 0.6, 0.6])
        r = RSSMReranker(self._cfg())
        B, K, A = 4, 6, 3
        cand = torch.randn(B, K, A).clamp(-1, 1)
        # plant target as candidate 3
        cand[:, 3, :] = target
        out = r.score_candidates(
            belief_flat=torch.randn(B, 8), cand_actions=cand,
            predict_fn=_predict_fn_near(target), frames=500,
        )
        self.assertTrue(out["active"])
        self.assertEqual(out["fallback_reason"], RERANK_OK)
        self.assertEqual(tuple(out["score"].shape), (B, K))
        self.assertTrue(torch.all(out["best_idx"] == 3))

    def test_uncertainty_gate(self):
        # large per-horizon disagreement -> high uncertainty -> fallback
        target = torch.zeros(3)
        r = RSSMReranker(self._cfg(uncertainty_threshold=0.01))
        cand = torch.randn(3, 5, 3).clamp(-1, 1)
        out = r.score_candidates(
            belief_flat=torch.randn(3, 8), cand_actions=cand,
            predict_fn=_predict_fn_near(target, noise=5.0), frames=500,
        )
        self.assertEqual(out["fallback_reason"], RERANK_HIGH_UNCERTAINTY)
        self.assertFalse(out["active"])

    def test_no_grad_through_planner(self):
        target = torch.zeros(3)
        r = RSSMReranker(self._cfg())
        cand = torch.randn(2, 4, 3).clamp(-1, 1).requires_grad_(True)
        belief = torch.randn(2, 8, requires_grad=True)
        out = r.score_candidates(
            belief_flat=belief, cand_actions=cand,
            predict_fn=_predict_fn_near(target), frames=500,
        )
        self.assertFalse(out["score"].requires_grad)
        self.assertFalse(out["best_action"].requires_grad)


if __name__ == "__main__":
    unittest.main()
