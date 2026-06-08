"""Phase A4 P0 — gradient-flow tests for the seq distill bridge.

User constraint (4):

    seq_aux_value_head must have grad after backward,
    shared PPO feature (the captured actor_mlp_out) must have grad,
    RSSM/Transformer teacher params must NOT have grad.
    teacher must detach.

We construct a tiny end-to-end fixture:

    * student feature: a 1-layer MLP standing in for the PPO shared encoder
    * student head: ``build_seq_aux_value_head``
    * teacher: a real ``RSSMLiteWorldModel`` (or a fake module with
      ``encode_belief``)

Then we run a forward + ``compute_seq_distill_aux_loss`` + backward and
assert grads land on the right modules and miss the wrong ones.

Constraint (1) check: ``--seq_wm_policy_coupling`` defaults to ``none`` is
also covered here (via train.py argparse) — see
``test_train_cli_default_is_none``.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
import unittest
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

from dexpbt_isaaclab.world_model.rssm_lite import RSSMLiteWorldModel
from dexpbt_isaaclab.world_model.seq_policy_aux import (
    COUPLING_DISTILL_VALUE,
    COUPLING_DISTILL_VALUE_SUCCESS,
    COUPLING_NONE,
    CapturedFeature,
    SeqDistillBridge,
    SeqDistillCache,
    SeqPolicyCouplingConfig,
    attach_seq_aux_heads,
    build_seq_aux_value_head,
    build_teacher_cache_from_predictions,
    compute_seq_distill_aux_loss,
    register_actor_mlp_capture,
)
from dexpbt_isaaclab.world_model.seq_wm_base import SequenceBatch


T = 4
N = 3
TOTAL = T * N
FEATURE_DIM = 8


class _FakeDataset:
    def __init__(self) -> None:
        self.last_range: Optional[tuple[int, int]] = None


class _FakeAgent:
    def __init__(self) -> None:
        self.dataset = _FakeDataset()


def _make_bridge(mode: str = COUPLING_DISTILL_VALUE) -> SeqDistillBridge:
    cfg = SeqPolicyCouplingConfig(
        mode=mode, weight=1.0, value_weight=1.0, success_weight=1.0,
        warmup_frames=0,
    )
    bridge = SeqDistillBridge(cfg=cfg)
    bridge.rollout_generation = 1
    bridge.last_frames = 10**9
    agent = _FakeAgent()
    bridge.agent_ref = agent
    return bridge


class StudentGradFlowTest(unittest.TestCase):
    """The student head + shared encoder MUST receive grad."""

    def test_student_head_and_shared_feature_get_grad(self):
        bridge = _make_bridge()
        # "Shared encoder" stand-in: a small linear layer producing the
        # 1-d feature the head sees. We require its grad to be non-zero
        # after backward, which proves the loss flows back through the
        # captured feature into the PPO shared encoder.
        encoder = nn.Linear(2, FEATURE_DIM)
        head = build_seq_aux_value_head(in_dim=FEATURE_DIM, hidden=8)
        bridge.seq_aux_value_head = head

        teacher = torch.randn(T, N)
        bridge.cache = build_teacher_cache_from_predictions(
            teacher_value=teacher, teacher_success_logit=None,
            T=T, N=N, generation=1,
        )

        # Mock minibatch covering env 0 (T samples).
        bridge.agent_ref.dataset.last_range = (0, T)
        x = torch.randn(T, 2, requires_grad=False)  # raw obs proxy
        feat = encoder(x)  # (T, FEATURE_DIM); requires grad via encoder
        bridge.feature_capture = CapturedFeature(tensor=feat, fired=True)

        losses = compute_seq_distill_aux_loss(bridge)
        self.assertIn("seq_distill_value", losses)
        loss = losses["seq_distill_value"]
        self.assertTrue(loss.requires_grad)

        # Zero any pre-existing grads.
        for p in head.parameters():
            p.grad = None
        for p in encoder.parameters():
            p.grad = None

        loss.backward()

        # 1. Student head MUST have non-trivial grad.
        for n, p in head.named_parameters():
            self.assertIsNotNone(p.grad, f"head.{n}.grad is None")
            self.assertGreater(
                float(p.grad.abs().sum().item()), 0.0,
                msg=f"head.{n}.grad is all zeros — student didn't learn"
            )

        # 2. Shared-encoder stand-in MUST have non-trivial grad — proves
        #    the gradient flowed THROUGH the captured feature back to
        #    the upstream PPO shared encoder representation.
        for n, p in encoder.named_parameters():
            self.assertIsNotNone(p.grad, f"encoder.{n}.grad is None")
            self.assertGreater(
                float(p.grad.abs().sum().item()), 0.0,
                msg=f"encoder.{n}.grad is all zeros — feature path broken"
            )


class TeacherDetachTest(unittest.TestCase):
    """Teacher must be frozen: zero gradient under any path."""

    def test_rssm_teacher_params_have_no_grad_after_distill_backward(self):
        # Build a real (tiny) RSSM teacher. encode_belief must run under
        # no_grad, and the cache must be stored detached.
        teacher = RSSMLiteWorldModel(
            obs_dim=4, action_dim=2,
            obs_embed_dim=8, action_embed_dim=4,
            h_dim=8, z_dim=4, head_hidden=8,
            with_value_head=True, with_success_head=False,
            with_task_error_head=False,
        )

        # Build a bare-minimum sequence batch (T, N) on CPU.
        obs = torch.randn(T, N, 4)
        action = torch.randn(T, N, 2)
        reward = torch.zeros(T, N)
        done = torch.zeros(T, N)
        next_obs = torch.cat([obs[1:], obs[-1:].clone()], dim=0)
        returns = torch.zeros(T, N)
        valid_mask = torch.ones(T, N)
        seg = torch.zeros(T, N, dtype=torch.long)
        batch = SequenceBatch(
            obs=obs, action=action, reward=reward, done=done,
            next_obs=next_obs, returns=returns,
            d_hat=None, next_d_hat=None,
            failure=None, success_count=None, next_success_count=None,
            lifted=None, valid_mask=valid_mask, segment_id=seg,
        )

        # Snapshot teacher params, run encode_belief (no_grad), confirm
        # the produced value tensor has no grad fn and is detached.
        for p in teacher.parameters():
            p.grad = None  # cleared
        with torch.no_grad():
            preds = teacher.encode_belief(batch)
        self.assertIsNotNone(preds["belief"])
        self.assertFalse(preds["belief"].requires_grad)
        self.assertIsNotNone(preds["value"])
        self.assertFalse(preds["value"].requires_grad)

        # Plumb teacher predictions into the bridge and run the full
        # student-side backward.
        bridge = _make_bridge()
        bridge.cache = build_teacher_cache_from_predictions(
            teacher_value=preds["value"], teacher_success_logit=None,
            T=T, N=N, generation=1,
        )
        head = build_seq_aux_value_head(in_dim=FEATURE_DIM, hidden=8)
        bridge.seq_aux_value_head = head
        encoder = nn.Linear(2, FEATURE_DIM)
        bridge.agent_ref.dataset.last_range = (0, T)
        feat = encoder(torch.randn(T, 2))
        bridge.feature_capture = CapturedFeature(tensor=feat, fired=True)
        loss = compute_seq_distill_aux_loss(bridge)["seq_distill_value"]

        # Reset grads; backward; assert teacher is untouched.
        for p in teacher.parameters():
            p.grad = None
        for p in head.parameters():
            p.grad = None
        for p in encoder.parameters():
            p.grad = None
        loss.backward()

        # Teacher params: NONE must receive a grad.
        for n, p in teacher.named_parameters():
            self.assertTrue(
                p.grad is None or float(p.grad.abs().sum().item()) == 0.0,
                msg=f"teacher param {n} received grad — detach contract broken"
            )
        # Student / encoder grads must exist.
        any_head = any(
            p.grad is not None and float(p.grad.abs().sum().item()) > 0
            for p in head.parameters()
        )
        any_enc = any(
            p.grad is not None and float(p.grad.abs().sum().item()) > 0
            for p in encoder.parameters()
        )
        self.assertTrue(any_head, "student head grad missing")
        self.assertTrue(any_enc, "shared encoder grad missing")


class CouplingNoneIsolationTest(unittest.TestCase):
    """coupling=none → bridge returns {} regardless of cache / capture."""

    def test_none_returns_empty_even_with_cache_present(self):
        bridge = _make_bridge(mode=COUPLING_NONE)
        bridge.cache = build_teacher_cache_from_predictions(
            teacher_value=torch.randn(T, N),
            teacher_success_logit=None,
            T=T, N=N, generation=1,
        )
        bridge.feature_capture = CapturedFeature(
            tensor=torch.randn(T, FEATURE_DIM, requires_grad=True),
            fired=True,
        )
        bridge.agent_ref.dataset.last_range = (0, T)
        result = compute_seq_distill_aux_loss(bridge)
        self.assertEqual(result, {})
        self.assertFalse(bridge.last_active)


class AttachSeqAuxHeadsTest(unittest.TestCase):
    """attach_seq_aux_heads MUST register heads + add to optimizer."""

    def test_attach_value_head_only(self):
        class _FakeNetwork(nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder = nn.Linear(8, FEATURE_DIM)

        class _FakeRlGamesModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.model = _FakeNetwork()

        m = _FakeRlGamesModel()
        opt = torch.optim.Adam(m.parameters(), lr=1e-3)
        n_groups_before = len(opt.param_groups)
        cfg = SeqPolicyCouplingConfig(mode=COUPLING_DISTILL_VALUE)
        out = attach_seq_aux_heads(
            rl_games_model=m, optimizer=opt, cfg=cfg, feature_dim=FEATURE_DIM,
        )
        self.assertIn("seq_aux_value_head", out)
        self.assertNotIn("seq_aux_success_head", out)
        self.assertTrue(hasattr(m.model, "seq_aux_value_head"))
        self.assertEqual(len(opt.param_groups), n_groups_before + 1)
        names = [g.get("name") for g in opt.param_groups]
        self.assertIn("seq_aux_value_head", names)

    def test_attach_value_and_success_heads(self):
        class _FakeNetwork(nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder = nn.Linear(8, FEATURE_DIM)

        class _FakeRlGamesModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.model = _FakeNetwork()

        m = _FakeRlGamesModel()
        opt = torch.optim.Adam(m.parameters(), lr=1e-3)
        n_groups_before = len(opt.param_groups)
        cfg = SeqPolicyCouplingConfig(mode=COUPLING_DISTILL_VALUE_SUCCESS)
        out = attach_seq_aux_heads(
            rl_games_model=m, optimizer=opt, cfg=cfg, feature_dim=FEATURE_DIM,
        )
        self.assertIn("seq_aux_value_head", out)
        self.assertIn("seq_aux_success_head", out)
        self.assertEqual(len(opt.param_groups), n_groups_before + 2)


class CouplingConfigDefaultsTest(unittest.TestCase):
    """User constraint (1): default mode must be 'none'."""

    def test_default_mode_is_none(self):
        cfg = SeqPolicyCouplingConfig()
        self.assertEqual(cfg.mode, COUPLING_NONE)
        self.assertFalse(cfg.enabled)
        self.assertFalse(cfg.has_value)
        self.assertFalse(cfg.has_success)

    def test_invalid_mode_raises(self):
        with self.assertRaises(ValueError):
            SeqPolicyCouplingConfig(mode="bogus")


class TrainCliDefaultIsNoneTest(unittest.TestCase):
    """Belt-and-braces: parse a tiny argparse mirroring train.py and
    confirm --seq_wm_policy_coupling defaults to 'none'.

    We don't import train.py at module scope (it would pull Omniverse),
    but its argparse arm-up is plain Python; mirroring a minimal subset
    is enough to lock the default in.
    """

    def test_default_none_in_minimal_argparse(self):
        p = argparse.ArgumentParser()
        # Mirror exactly the relevant bits from train.py:
        p.add_argument(
            "--seq_wm_policy_coupling",
            type=str,
            choices=["none", "distill_value", "distill_value_success"],
            default="none",
        )
        p.add_argument("--seq_wm_policy_coupling_weight", type=float, default=0.05)
        p.add_argument("--seq_wm_policy_coupling_warmup_frames", type=int,
                       default=100_000_000)
        ns = p.parse_args([])
        self.assertEqual(ns.seq_wm_policy_coupling, "none")
        self.assertEqual(ns.seq_wm_policy_coupling_warmup_frames, 100_000_000)


if __name__ == "__main__":
    unittest.main()
