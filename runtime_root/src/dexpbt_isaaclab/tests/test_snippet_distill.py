"""Tests for snippet distillation (first-step only)."""

from __future__ import annotations

import unittest

import torch
import torch.nn as nn

from dexpbt_isaaclab.world_model.snippet_distill import (
    SnippetDistillConfig,
    compute_snippet_distill_loss,
)


class _MuHead(nn.Module):
    def __init__(self, f=8, a=3):
        super().__init__()
        self.lin = nn.Linear(f, a)

    def forward(self, x):
        return self.lin(x)


class TestSnippetDistill(unittest.TestCase):
    def test_loss_first_step_only(self):
        head = _MuHead()
        cfg = SnippetDistillConfig(enabled=True, weight=1.0)
        feat = torch.randn(4, 8, requires_grad=False)
        teacher = torch.randn(4, 3)
        loss, m = compute_snippet_distill_loss(
            mu_fn=head,
            feat_start=feat,
            teacher_first_action=teacher,
            mask=torch.ones(4),
            cfg=cfg,
        )
        self.assertGreater(float(m["snippet_distill/loss"]), 0.0)
        loss.backward()
        self.assertIsNotNone(head.lin.weight.grad)

    def test_mu_only_grad(self):
        head = _MuHead()
        critic = nn.Linear(8, 1)
        cfg = SnippetDistillConfig(enabled=True, weight=1.0)
        feat = torch.randn(2, 8)
        teacher = torch.randn(2, 3)
        loss, _ = compute_snippet_distill_loss(
            mu_fn=head,
            feat_start=feat,
            teacher_first_action=teacher,
            mask=torch.ones(2),
            cfg=cfg,
        )
        loss.backward()
        self.assertIsNotNone(head.lin.weight.grad)
        self.assertIsNone(critic.weight.grad)
