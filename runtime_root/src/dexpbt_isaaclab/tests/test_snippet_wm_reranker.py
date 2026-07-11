"""Tests for snippet WM reranker."""

from __future__ import annotations

import unittest

import torch

from dexpbt_isaaclab.world_model.snippet_wm_reranker import (
    SnippetWMRerankConfig,
    effective_alpha_wm,
    score_snippet_candidates,
)


class TestSnippetWMReranker(unittest.TestCase):
    def test_alpha_zero_uses_priority(self):
        cfg = SnippetWMRerankConfig(enabled=True)
        pri = torch.tensor([[1.0, 5.0, 2.0]])
        idx, _ = score_snippet_candidates(
            priority=pri,
            p_recovery=torch.zeros(1, 3),
            p_success=torch.zeros(1, 3),
            p_drop=torch.zeros(1, 3),
            uncertainty=torch.zeros(1, 3),
            action_delta=torch.zeros(1, 3),
            alpha_wm=0.0,
            cfg=cfg,
        )
        self.assertEqual(int(idx.item()), 1)

    def test_alpha_positive_can_change(self):
        cfg = SnippetWMRerankConfig(enabled=True, w_rec=1.0, score_cap=1.0)
        pri = torch.tensor([[1.0, 1.0, 1.0]])
        p_rec = torch.tensor([[0.1, 0.9, 0.2]])
        idx, diag = score_snippet_candidates(
            priority=pri,
            p_recovery=p_rec,
            p_success=torch.zeros(1, 3),
            p_drop=torch.zeros(1, 3),
            uncertainty=torch.zeros(1, 3),
            action_delta=torch.zeros(1, 3),
            alpha_wm=1.0,
            cfg=cfg,
        )
        self.assertEqual(int(idx.item()), 1)
        self.assertGreaterEqual(diag["snippet_wm_rerank/wm_term_mean"], 0.0)

    def test_effective_alpha_when_disabled(self):
        self.assertEqual(effective_alpha_wm(wm_calibration_enable=False, alpha_wm=0.9), 0.0)
