"""Phase A4 P1 (v2 coupling) -- sub-tower wiring + grad routing tests.

The v2 mode `critic_distill_value_norm` introduces a private
`seq_aux_critic_mlp` sub-tower stacked between the captured shared
encoder feature (`actor_mlp_out`) and `seq_aux_value_head`. The legacy
P0 `distill_value` path remains the actor-direct distill (head consumes
`actor_mlp_out` directly) and is preserved as negative ablation only.

This file pins:
  * sub-tower attachment is v2-only and registers an optimizer param group,
  * v2 mode requires `feature_source=shared_to_aux_critic`,
  * after a v2 backward, gradients land on (sub-tower, value head, encoder)
    but never on the teacher,
  * v2 mode without an attached sub-tower fails fast (no fallback),
  * legacy `distill_value` does NOT create a sub-tower and still produces
    grad on `actor_mlp` (the P0 invariant),
  * config validation rejects forbidden combinations (e.g. v2 mode +
    feature_source=actor, or feature_source=shared_to_aux_critic on a
    legacy mode).
"""

from __future__ import annotations

import unittest
from typing import Optional

import torch
import torch.nn as nn

from dexpbt_isaaclab.world_model.rssm_lite import RSSMLiteWorldModel
from dexpbt_isaaclab.world_model.seq_policy_aux import (
    COUPLING_CRITIC_DISTILL_VALUE_NORM,
    COUPLING_DISTILL_VALUE,
    COUPLING_NONE,
    FEATURE_SOURCE_ACTOR,
    FEATURE_SOURCE_SHARED_TO_AUX_CRITIC,
    GATE_NONE,
    VALUE_NORM_NONE,
    VALUE_NORM_ROLLOUT_ZSCORE,
    CapturedFeature,
    SeqDistillBridge,
    SeqPolicyCouplingConfig,
    attach_seq_aux_heads,
    build_teacher_cache_from_predictions,
    compute_seq_distill_aux_loss,
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


class _FakeNetwork(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Linear(8, FEATURE_DIM)


class _FakeRlGamesModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = _FakeNetwork()


def _v2_cfg(**overrides) -> SeqPolicyCouplingConfig:
    base = dict(
        mode=COUPLING_CRITIC_DISTILL_VALUE_NORM,
        weight=1.0,
        value_weight=1.0,
        success_weight=0.0,
        warmup_frames=0,
        feature_source=FEATURE_SOURCE_SHARED_TO_AUX_CRITIC,
        teacher_value_norm=VALUE_NORM_ROLLOUT_ZSCORE,
        gate=GATE_NONE,  # tests below isolate gate behavior elsewhere
        ramp_frames=0,
    )
    base.update(overrides)
    return SeqPolicyCouplingConfig(**base)


def _v2_bridge(value_norm: str = VALUE_NORM_ROLLOUT_ZSCORE) -> SeqDistillBridge:
    cfg = _v2_cfg(teacher_value_norm=value_norm)
    bridge = SeqDistillBridge(cfg=cfg)
    bridge.rollout_generation = 1
    bridge.last_frames = 10**9
    agent = _FakeAgent()
    bridge.agent_ref = agent
    return bridge


class V2ConfigValidationTest(unittest.TestCase):
    """Plan section 8 -- explicit validation, no silent fallback."""

    def test_v2_requires_shared_to_aux_critic(self):
        with self.assertRaisesRegex(ValueError, "feature_source=shared_to_aux_critic"):
            SeqPolicyCouplingConfig(
                mode=COUPLING_CRITIC_DISTILL_VALUE_NORM,
                feature_source=FEATURE_SOURCE_ACTOR,
                teacher_value_norm=VALUE_NORM_ROLLOUT_ZSCORE,
            )

    def test_v2_requires_rollout_zscore(self):
        with self.assertRaisesRegex(ValueError, "teacher_value_norm=rollout_zscore"):
            SeqPolicyCouplingConfig(
                mode=COUPLING_CRITIC_DISTILL_VALUE_NORM,
                feature_source=FEATURE_SOURCE_SHARED_TO_AUX_CRITIC,
                teacher_value_norm=VALUE_NORM_NONE,
            )

    def test_legacy_mode_rejects_shared_to_aux_critic(self):
        with self.assertRaisesRegex(ValueError, "is reserved"):
            SeqPolicyCouplingConfig(
                mode=COUPLING_DISTILL_VALUE,
                feature_source=FEATURE_SOURCE_SHARED_TO_AUX_CRITIC,
            )

    def test_alias_resolves_to_canonical_v2(self):
        cfg = SeqPolicyCouplingConfig(
            mode="shared_aux_critic_distill_value_norm",
            feature_source=FEATURE_SOURCE_SHARED_TO_AUX_CRITIC,
            teacher_value_norm=VALUE_NORM_ROLLOUT_ZSCORE,
        )
        self.assertEqual(cfg.mode, COUPLING_CRITIC_DISTILL_VALUE_NORM)
        self.assertTrue(cfg.is_v2)

    def test_v2_warning_when_gate_disabled(self):
        cfg = SeqPolicyCouplingConfig(
            mode=COUPLING_CRITIC_DISTILL_VALUE_NORM,
            feature_source=FEATURE_SOURCE_SHARED_TO_AUX_CRITIC,
            teacher_value_norm=VALUE_NORM_ROLLOUT_ZSCORE,
            gate=GATE_NONE,
        )
        self.assertTrue(cfg.gate_user_overridden)


class V2AttachSubtowerTest(unittest.TestCase):
    """Sub-tower must register module + optimizer group iff v2."""

    def test_legacy_distill_value_does_not_attach_subtower(self):
        m = _FakeRlGamesModel()
        opt = torch.optim.Adam(m.parameters(), lr=1e-3)
        cfg = SeqPolicyCouplingConfig(mode=COUPLING_DISTILL_VALUE)
        out = attach_seq_aux_heads(
            rl_games_model=m, optimizer=opt, cfg=cfg, feature_dim=FEATURE_DIM,
        )
        self.assertNotIn("seq_aux_critic_mlp", out)
        self.assertFalse(hasattr(m.model, "seq_aux_critic_mlp"))
        names = [g.get("name") for g in opt.param_groups]
        self.assertNotIn("seq_aux_critic_mlp", names)

    def test_v2_attaches_subtower_and_optimizer_group(self):
        m = _FakeRlGamesModel()
        opt = torch.optim.Adam(m.parameters(), lr=1e-3)
        n_groups_before = len(opt.param_groups)
        cfg = _v2_cfg()
        out = attach_seq_aux_heads(
            rl_games_model=m, optimizer=opt, cfg=cfg, feature_dim=FEATURE_DIM,
        )
        self.assertIn("seq_aux_critic_mlp", out)
        self.assertIn("seq_aux_value_head", out)
        self.assertTrue(hasattr(m.model, "seq_aux_critic_mlp"))
        self.assertTrue(hasattr(m.model, "seq_aux_value_head"))
        self.assertEqual(len(opt.param_groups), n_groups_before + 2)
        names = [g.get("name") for g in opt.param_groups]
        self.assertIn("seq_aux_critic_mlp", names)
        self.assertIn("seq_aux_value_head", names)

    def test_v2_subtower_output_dim_matches_value_head_input(self):
        m = _FakeRlGamesModel()
        opt = torch.optim.Adam(m.parameters(), lr=1e-3)
        cfg = _v2_cfg(aux_critic_subtower_hidden=64)
        attach_seq_aux_heads(
            rl_games_model=m, optimizer=opt, cfg=cfg, feature_dim=FEATURE_DIM,
        )
        x = torch.randn(2, FEATURE_DIM)
        sub_out = m.model.seq_aux_critic_mlp(x)
        head_in = m.model.seq_aux_value_head(sub_out)
        self.assertEqual(sub_out.shape[-1], FEATURE_DIM)
        self.assertEqual(head_in.shape, (2, 1))


class V2GradFlowTest(unittest.TestCase):
    """Sub-tower + value head + shared encoder must receive grad in v2.
    Teacher is detached; legacy mode keeps its own actor-direct grad path.
    """

    def test_v2_grad_lands_on_subtower_and_encoder(self):
        m = _FakeRlGamesModel()
        opt = torch.optim.Adam(m.parameters(), lr=1e-3)
        cfg = _v2_cfg()
        heads = attach_seq_aux_heads(
            rl_games_model=m, optimizer=opt, cfg=cfg, feature_dim=FEATURE_DIM,
        )

        bridge = SeqDistillBridge(cfg=cfg)
        bridge.rollout_generation = 1
        bridge.last_frames = 10**9
        bridge.agent_ref = _FakeAgent()
        bridge.seq_aux_value_head = heads["seq_aux_value_head"]
        bridge.seq_aux_critic_mlp = heads["seq_aux_critic_mlp"]

        teacher = torch.randn(T, N)
        bridge.cache = build_teacher_cache_from_predictions(
            teacher_value=teacher, teacher_success_logit=None,
            T=T, N=N, generation=1,
            teacher_value_norm=VALUE_NORM_ROLLOUT_ZSCORE,
            returns_for_corr=None,
        )

        # Stand-in for the upstream PPO encoder.
        encoder = nn.Linear(2, FEATURE_DIM)
        bridge.agent_ref.dataset.last_range = (0, T)
        feat = encoder(torch.randn(T, 2))
        bridge.feature_capture = CapturedFeature(tensor=feat, fired=True)

        for p in m.model.seq_aux_critic_mlp.parameters():
            p.grad = None
        for p in m.model.seq_aux_value_head.parameters():
            p.grad = None
        for p in encoder.parameters():
            p.grad = None

        losses = compute_seq_distill_aux_loss(bridge)
        self.assertIn("seq_distill_value", losses)
        losses["seq_distill_value"].backward()

        # Sub-tower receives grad.
        any_sub = any(
            p.grad is not None and float(p.grad.abs().sum().item()) > 0.0
            for p in m.model.seq_aux_critic_mlp.parameters()
        )
        self.assertTrue(any_sub, "seq_aux_critic_mlp grad missing")

        # Value head receives grad.
        any_head = any(
            p.grad is not None and float(p.grad.abs().sum().item()) > 0.0
            for p in m.model.seq_aux_value_head.parameters()
        )
        self.assertTrue(any_head, "seq_aux_value_head grad missing")

        # Encoder receives grad (loss flowed shared encoder -> sub-tower).
        any_enc = any(
            p.grad is not None and float(p.grad.abs().sum().item()) > 0.0
            for p in encoder.parameters()
        )
        self.assertTrue(any_enc, "shared encoder grad missing in v2")

    def test_v2_teacher_params_have_no_grad(self):
        teacher = RSSMLiteWorldModel(
            obs_dim=4, action_dim=2,
            obs_embed_dim=8, action_embed_dim=4,
            h_dim=8, z_dim=4, head_hidden=8,
            with_value_head=True, with_success_head=False,
            with_task_error_head=False,
        )
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

        for p in teacher.parameters():
            p.grad = None
        with torch.no_grad():
            preds = teacher.encode_belief(batch)
        self.assertIsNotNone(preds["value"])
        self.assertFalse(preds["value"].requires_grad)

        m = _FakeRlGamesModel()
        opt = torch.optim.Adam(m.parameters(), lr=1e-3)
        cfg = _v2_cfg()
        heads = attach_seq_aux_heads(
            rl_games_model=m, optimizer=opt, cfg=cfg, feature_dim=FEATURE_DIM,
        )
        bridge = SeqDistillBridge(cfg=cfg)
        bridge.rollout_generation = 1
        bridge.last_frames = 10**9
        bridge.agent_ref = _FakeAgent()
        bridge.seq_aux_value_head = heads["seq_aux_value_head"]
        bridge.seq_aux_critic_mlp = heads["seq_aux_critic_mlp"]
        bridge.cache = build_teacher_cache_from_predictions(
            teacher_value=preds["value"], teacher_success_logit=None,
            T=T, N=N, generation=1,
            teacher_value_norm=VALUE_NORM_ROLLOUT_ZSCORE,
        )

        encoder = nn.Linear(2, FEATURE_DIM)
        bridge.agent_ref.dataset.last_range = (0, T)
        feat = encoder(torch.randn(T, 2))
        bridge.feature_capture = CapturedFeature(tensor=feat, fired=True)
        losses = compute_seq_distill_aux_loss(bridge)
        for p in teacher.parameters():
            p.grad = None
        losses["seq_distill_value"].backward()

        for n, p in teacher.named_parameters():
            self.assertTrue(
                p.grad is None or float(p.grad.abs().sum().item()) == 0.0,
                msg=f"teacher param {n} received grad in v2 -- detach broken",
            )


class V2FailFastNoFallbackTest(unittest.TestCase):
    """v2 must NEVER silently delegate to the legacy actor-direct path."""

    def test_v2_without_subtower_attached_raises(self):
        bridge = _v2_bridge()
        bridge.seq_aux_critic_mlp = None  # simulating misconfiguration
        bridge.seq_aux_value_head = nn.Linear(FEATURE_DIM, 1)
        teacher = torch.randn(T, N)
        bridge.cache = build_teacher_cache_from_predictions(
            teacher_value=teacher, teacher_success_logit=None,
            T=T, N=N, generation=1,
            teacher_value_norm=VALUE_NORM_ROLLOUT_ZSCORE,
        )
        bridge.agent_ref.dataset.last_range = (0, T)
        bridge.feature_capture = CapturedFeature(
            tensor=torch.randn(T, FEATURE_DIM), fired=True,
        )
        with self.assertRaisesRegex(RuntimeError, "seq_aux_critic_mlp"):
            compute_seq_distill_aux_loss(bridge)


class LegacyDistillValueGradFlowTest(unittest.TestCase):
    """Legacy P0 path keeps its actor-direct grad behavior intact."""

    def test_legacy_distill_value_no_subtower_grad(self):
        m = _FakeRlGamesModel()
        opt = torch.optim.Adam(m.parameters(), lr=1e-3)
        cfg = SeqPolicyCouplingConfig(
            mode=COUPLING_DISTILL_VALUE, weight=1.0, warmup_frames=0,
        )
        heads = attach_seq_aux_heads(
            rl_games_model=m, optimizer=opt, cfg=cfg, feature_dim=FEATURE_DIM,
        )
        self.assertNotIn("seq_aux_critic_mlp", heads)

        bridge = SeqDistillBridge(cfg=cfg)
        bridge.rollout_generation = 1
        bridge.last_frames = 10**9
        bridge.agent_ref = _FakeAgent()
        bridge.seq_aux_value_head = heads["seq_aux_value_head"]
        # IMPORTANT: do NOT attach a sub-tower for legacy.
        bridge.seq_aux_critic_mlp = None

        teacher = torch.randn(T, N)
        bridge.cache = build_teacher_cache_from_predictions(
            teacher_value=teacher, teacher_success_logit=None,
            T=T, N=N, generation=1,
        )
        encoder = nn.Linear(2, FEATURE_DIM)
        bridge.agent_ref.dataset.last_range = (0, T)
        feat = encoder(torch.randn(T, 2))
        bridge.feature_capture = CapturedFeature(tensor=feat, fired=True)

        for p in m.model.seq_aux_value_head.parameters():
            p.grad = None
        for p in encoder.parameters():
            p.grad = None
        losses = compute_seq_distill_aux_loss(bridge)
        self.assertIn("seq_distill_value", losses)
        losses["seq_distill_value"].backward()

        # Legacy: head + encoder receive grad. No sub-tower exists at all.
        any_head = any(
            p.grad is not None and float(p.grad.abs().sum().item()) > 0.0
            for p in m.model.seq_aux_value_head.parameters()
        )
        any_enc = any(
            p.grad is not None and float(p.grad.abs().sum().item()) > 0.0
            for p in encoder.parameters()
        )
        self.assertTrue(any_head)
        self.assertTrue(any_enc)


class CouplingNoneStillIsolatedTest(unittest.TestCase):
    """Adding v2 must not break the existing coupling=none isolation."""

    def test_none_returns_empty_with_v2_field_defaults(self):
        cfg = SeqPolicyCouplingConfig()
        self.assertEqual(cfg.mode, COUPLING_NONE)
        self.assertEqual(cfg.feature_source, FEATURE_SOURCE_ACTOR)
        self.assertEqual(cfg.teacher_value_norm, VALUE_NORM_NONE)
        self.assertEqual(cfg.gate, GATE_NONE)
        self.assertEqual(cfg.ramp_frames, 0)
        self.assertFalse(cfg.enabled)
        self.assertFalse(cfg.is_v2)


if __name__ == "__main__":
    unittest.main()
