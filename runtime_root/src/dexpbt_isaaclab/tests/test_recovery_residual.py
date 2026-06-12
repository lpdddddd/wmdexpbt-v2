"""Unit tests for the EIR-MPC Recovery Residual Head (Phase B)."""

from __future__ import annotations

import unittest

import torch
import torch.nn as nn

from dexpbt_isaaclab.world_model.recovery_residual import (
    RecoveryResidualConfig,
    RecoveryResidualHead,
    RecoveryResidualBridge,
    attach_recovery_residual,
    compute_recovery_residual_loss,
    compute_rollout_residual_action,
    APPLY_OK,
    APPLY_AUX_MISSING,
    APPLY_AUX_STALE,
    APPLY_BEFORE_WARMUP,
    APPLY_KILL_SWITCH_OFF,
)


def _cfg(**kw):
    base = dict(enabled=True, feature_dim=8, hidden_dim=16, action_dim=3)
    base.update(kw)
    return RecoveryResidualConfig(**base)


class TestResidualHead(unittest.TestCase):
    def test_forward_shape(self):
        head = RecoveryResidualHead(_cfg())
        feat = torch.randn(5, 8)
        delta_raw, gate_logit = head(feat)
        self.assertEqual(tuple(delta_raw.shape), (5, 3))
        self.assertEqual(tuple(gate_logit.shape), (5, 1))
        out = head.recovery_action(feat, torch.zeros(5, 3), alpha=0.5)
        self.assertEqual(tuple(out["a_rec"].shape), (5, 3))
        self.assertTrue(torch.all(out["a_rec"] <= 1.0) and torch.all(out["a_rec"] >= -1.0))

    def test_loss_updates_residual_params(self):
        cfg = _cfg(distill_weight=1.0, floor_distill_weight=1.0, hard_floor_distill_weight=1.0)
        bridge = RecoveryResidualBridge(cfg=cfg, head=RecoveryResidualHead(cfg))
        opt = torch.optim.SGD(bridge.head.parameters(), lr=0.1)
        feat = torch.randn(6, 8)
        mu = torch.zeros(6, 3)
        a_star = torch.full((6, 3), 0.5)
        before = bridge.head.delta_head.weight.detach().clone()
        out = compute_recovery_residual_loss(
            bridge, feat_e=feat, mu_live_e=mu, a_star=a_star, tol_mean=0.015
        )
        self.assertIn("eir_residual_total", out)
        opt.zero_grad()
        out["eir_residual_total"].backward()
        self.assertIsNotNone(bridge.head.delta_head.weight.grad)
        opt.step()
        after = bridge.head.delta_head.weight.detach()
        self.assertFalse(torch.allclose(before, after))

    def test_loss_no_grad_to_main_actor_when_mu_detached(self):
        # constraint 19.1: residual distill must not touch the main actor mu.
        cfg = _cfg(mu_base_detach=True, distill_weight=1.0)
        bridge = RecoveryResidualBridge(cfg=cfg, head=RecoveryResidualHead(cfg))
        actor_mu = nn.Linear(8, 3)
        feat = torch.randn(6, 8)
        mu_live = actor_mu(feat)  # depends on actor params
        a_star = torch.full((6, 3), 0.4)
        out = compute_recovery_residual_loss(
            bridge, feat_e=feat, mu_live_e=mu_live, a_star=a_star, tol_mean=0.01
        )
        out["eir_residual_total"].backward()
        # main actor receives no gradient
        self.assertTrue(actor_mu.weight.grad is None or float(actor_mu.weight.grad.abs().sum()) == 0.0)
        # residual head does receive gradient
        self.assertIsNotNone(bridge.head.delta_head.weight.grad)
        self.assertGreater(float(bridge.head.delta_head.weight.grad.abs().sum()), 0.0)

    def test_disabled_residual_returns_empty(self):
        cfg = _cfg(enabled=False)
        bridge = RecoveryResidualBridge(cfg=cfg, head=RecoveryResidualHead(cfg))
        out = compute_recovery_residual_loss(
            bridge, feat_e=torch.randn(4, 8), mu_live_e=torch.zeros(4, 3),
            a_star=torch.zeros(4, 3), tol_mean=0.01,
        )
        self.assertEqual(out, {})

    def test_attach_adds_optimizer_group(self):
        cfg = _cfg()
        opt = torch.optim.Adam([torch.nn.Parameter(torch.zeros(2))], lr=1e-3)
        n_before = len(opt.param_groups)
        head = attach_recovery_residual(optimizer=opt, cfg=cfg)
        self.assertEqual(len(opt.param_groups), n_before + 1)
        self.assertEqual(opt.param_groups[-1]["name"], "eir_residual")
        self.assertIsInstance(head, RecoveryResidualHead)

    def test_checkpoint_roundtrip(self):
        cfg = _cfg()
        head = RecoveryResidualHead(cfg)
        head2 = RecoveryResidualHead(cfg)
        head2.load_state_dict(head.state_dict())
        feat = torch.randn(3, 8)
        d1, g1 = head(feat)
        d2, g2 = head2(feat)
        self.assertTrue(torch.allclose(d1, d2))
        self.assertTrue(torch.allclose(g1, g2))


class TestRolloutApply(unittest.TestCase):
    def _bridge(self, **kw):
        params = dict(apply_to_rollout=True, apply_after_frames=100, gate_confidence_threshold=0.0)
        params.update(kw)
        cfg = _cfg(**params)
        return RecoveryResidualBridge(cfg=cfg, head=RecoveryResidualHead(cfg))

    def test_apply_active_when_aux_available(self):
        # constraint 19.3: residual apply is active when aux present + floor + warmup.
        bridge = self._bridge()
        B = 5
        feat = torch.randn(B, 8)
        mu = torch.zeros(B, 3)
        # rows 0,1 floor near-success; others not
        d_hat = torch.tensor([0.01, 0.01, 0.5, 0.5, 0.5])
        tol = torch.tensor([0.015, 0.015, 0.05, 0.05, 0.05])
        lifted = torch.tensor([1.0, 1.0, 1.0, 0.0, 1.0])
        bridge.set_aux_snapshot(d_hat=d_hat, tol=tol, lifted=lifted, frame=500)
        res = compute_rollout_residual_action(bridge, feat=feat, mu=mu, frames=500)
        self.assertEqual(res["fallback_reason"], APPLY_OK)
        self.assertTrue(bool(res["apply_mask"][0]) and bool(res["apply_mask"][1]))
        self.assertFalse(bool(res["apply_mask"][2]))
        self.assertFalse(bool(res["apply_mask"][3]))
        # only gated rows changed
        changed = (res["action"] != mu).any(dim=-1)
        self.assertTrue(torch.equal(changed, res["apply_mask"]))

    def test_apply_self_disables_when_aux_missing(self):
        bridge = self._bridge()
        feat = torch.randn(4, 8)
        mu = torch.randn(4, 3).clamp(-1, 1)
        bridge.set_aux_snapshot(d_hat=None, tol=None, lifted=None, frame=500)
        res = compute_rollout_residual_action(bridge, feat=feat, mu=mu, frames=500)
        self.assertEqual(res["fallback_reason"], APPLY_AUX_MISSING)
        self.assertTrue(torch.allclose(res["action"], mu))
        self.assertEqual(res["apply_fraction"], 0.0)

    def test_apply_self_disables_when_aux_stale_by_frame_age(self):
        bridge = self._bridge(apply_max_aux_age_frames=1000)
        feat = torch.randn(3, 8)
        mu = torch.zeros(3, 3)
        bridge.set_aux_snapshot(
            d_hat=torch.full((3,), 0.01), tol=torch.full((3,), 0.015),
            lifted=torch.ones(3), frame=100,
        )
        res = compute_rollout_residual_action(bridge, feat=feat, mu=mu, frames=5000)
        self.assertEqual(res["fallback_reason"], APPLY_AUX_STALE)
        self.assertTrue(torch.allclose(res["action"], mu))

    def test_apply_self_disables_when_aux_generation_mismatch(self):
        bridge = self._bridge(apply_max_aux_age_frames=10**9)
        feat = torch.randn(3, 8)
        mu = torch.zeros(3, 3)
        bridge.set_aux_snapshot(
            d_hat=torch.full((3,), 0.01), tol=torch.full((3,), 0.015),
            lifted=torch.ones(3), frame=500, generation=7,
        )
        res = compute_rollout_residual_action(
            bridge, feat=feat, mu=mu, frames=500, rollout_generation=8,
        )
        self.assertEqual(res["fallback_reason"], APPLY_AUX_STALE)
        self.assertTrue(torch.allclose(res["action"], mu))

    def test_apply_not_aux_missing_when_fresh_aux_after_warmup(self):
        bridge = self._bridge()
        feat = torch.randn(3, 8)
        mu = torch.zeros(3, 3)
        bridge.set_aux_snapshot(
            d_hat=torch.full((3,), 0.01), tol=torch.full((3,), 0.015),
            lifted=torch.ones(3), frame=500, generation=3,
        )
        res = compute_rollout_residual_action(
            bridge, feat=feat, mu=mu, frames=500, rollout_generation=3,
        )
        self.assertNotEqual(res["fallback_reason"], APPLY_AUX_MISSING)
        self.assertNotEqual(res["fallback_reason"], APPLY_AUX_STALE)

    def test_apply_blocked_before_warmup(self):
        bridge = self._bridge(apply_after_frames=1_000_000)
        feat = torch.randn(3, 8)
        mu = torch.zeros(3, 3)
        bridge.set_aux_snapshot(
            d_hat=torch.full((3,), 0.01), tol=torch.full((3,), 0.015),
            lifted=torch.ones(3), frame=10,
        )
        res = compute_rollout_residual_action(bridge, feat=feat, mu=mu, frames=10)
        self.assertEqual(res["fallback_reason"], APPLY_BEFORE_WARMUP)
        self.assertTrue(torch.allclose(res["action"], mu))

    def test_apply_kill_switch_off(self):
        cfg = _cfg(apply_to_rollout=False)
        bridge = RecoveryResidualBridge(cfg=cfg, head=RecoveryResidualHead(cfg))
        feat = torch.randn(3, 8)
        mu = torch.zeros(3, 3)
        res = compute_rollout_residual_action(bridge, feat=feat, mu=mu, frames=10**9)
        self.assertEqual(res["fallback_reason"], APPLY_KILL_SWITCH_OFF)
        self.assertTrue(torch.allclose(res["action"], mu))


if __name__ == "__main__":
    unittest.main()
