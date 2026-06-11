"""FloorCritic (Real-target) unit tests.

Covers (see results/FLOORCRITIC_IMPLEMENTATION_PLAN.md, section G):
  1. default-off: enabled=False is a no-op (lambda=0, loss={}).
  2. target builder: near_success / success / dhat_improve / drop_risk
     correctness; no done-boundary crossing; horizon-overflow masking;
     valid_fraction + pos_rate.
  3. gradient (Phase 0 detached): FC loss gives NO grad to mu / actor_mlp /
     backbone; aux tower+heads DO get grad.
  4. gradient (Phase 1 shared): FC loss gives NO grad to mu head, but DOES
     give grad to actor_mlp backbone; aux heads get grad.
  5. grad-ratio / auto-scale behaviour.
  6. lambda(tolerance) stage activation.
  7. feature_source switching + critic_feature degradation.
  8. integration-ish smoke: compute_floorcritic_loss returns a grad-bearing
     tensor aligned to a fake PPODataset.last_range; logging snapshot filled.

Uses a tiny mock rl_games-like model (shared backbone) so the tests do not
require Isaac Sim / rl_games to be importable.
"""

from __future__ import annotations

import unittest
from pathlib import Path

import torch
import torch.nn as nn

from dexpbt_isaaclab.world_model.floorcritic import (
    FEATURE_SOURCE_DETACHED,
    FEATURE_SOURCE_SHARED,
    FloorCriticBridge,
    FloorCriticCapture,
    FloorCriticConfig,
    FloorCriticHeads,
    FloorCriticTargetBuilder,
    attach_floorcritic_heads,
    compute_floorcritic_loss,
    compute_grad_audit,
)


# ---------------------------------------------------------------------------
# Mock rl_games-like model (shared backbone: actor_mlp -> mu/value heads)
# ---------------------------------------------------------------------------


class _MockA2CNetwork(nn.Module):
    def __init__(self, obs_dim=12, feat_dim=16, act_dim=4):
        super().__init__()
        self.actor_mlp = nn.Sequential(nn.Linear(obs_dim, feat_dim), nn.ELU())
        self.critic_mlp = nn.Sequential()  # empty, like separate=False
        self.mu = nn.Linear(feat_dim, act_dim)
        self.value = nn.Linear(feat_dim, 1)

    def forward(self, obs):
        feat = self.actor_mlp(obs)
        return self.mu(feat), self.value(feat), feat


class _MockModel(nn.Module):
    def __init__(self, **kw):
        super().__init__()
        self.a2c_network = _MockA2CNetwork(**kw)

    def forward(self, obs):
        return self.a2c_network(obs)


class _MockDataset:
    def __init__(self, last_range):
        self.last_range = last_range


class _MockAgent:
    def __init__(self, dataset):
        self.dataset = dataset


def _install_capture(model, capture):
    a2c = model.a2c_network

    def hook(_m, _i, out):
        capture.tensor = out.to(torch.float32)
        capture.fired = True

    return a2c.actor_mlp.register_forward_hook(hook)


# ---------------------------------------------------------------------------
# 1. default-off
# ---------------------------------------------------------------------------


class TestDefaultOff(unittest.TestCase):
    def test_disabled_lambda_zero(self):
        cfg = FloorCriticConfig(enabled=False)
        for tol in (0.5, 0.04, 0.02, 0.012, 0.005):
            self.assertEqual(cfg.lambda_for_tol(tol), 0.0)

    def test_disabled_loss_empty(self):
        cfg = FloorCriticConfig(enabled=False)
        bridge = FloorCriticBridge(cfg=cfg)
        self.assertEqual(compute_floorcritic_loss(bridge), {})

    def test_default_config_is_phase0_safe(self):
        cfg = FloorCriticConfig()
        self.assertFalse(cfg.enabled)
        self.assertEqual(cfg.feature_source, FEATURE_SOURCE_DETACHED)
        self.assertTrue(cfg.should_detach)
        self.assertTrue(cfg.actor_detach)
        self.assertFalse(cfg.update_actor)


# ---------------------------------------------------------------------------
# 2. target builder
# ---------------------------------------------------------------------------


class TestTargetBuilder(unittest.TestCase):
    def _builder(self, **over):
        base = dict(
            enabled=True,
            horizons=(2,),
            near_success_mult=1.5,
            dhat_improve_margin=0.005,
            drop_dhat_margin=0.01,
        )
        base.update(over)
        return FloorCriticTargetBuilder(FloorCriticConfig(**base))

    def test_success_target(self):
        # T=4, N=1. success_cnt increases at t=2 -> for h=2, t=0 sees increase
        # at t=2 (within k=1..2), t=1 sees t=2 or t=3.
        T, N = 4, 1
        d_hat = torch.ones(T, N, 1) * 0.1
        succ = torch.tensor([[0.0], [0.0], [1.0], [1.0]]).reshape(T, N, 1)
        tol = torch.ones(T, N, 1) * 0.01
        fail = torch.zeros(T, N, 1)
        lifted = torch.ones(T, N, 1)
        dones = torch.zeros(T, N)
        cache = self._builder().build(
            d_hat=d_hat, success_cnt=succ, success_tol=tol,
            failure=fail, lifted=lifted, dones=dones, generation=1,
        )
        key = "success_h2"
        # layout flat[n*T + t]; N=1 so flat[t].
        tgt = cache.targets_flat[key]
        mask = cache.masks_flat[key]
        # valid for t where t+2 < T -> t in {0,1}.
        self.assertEqual(float(mask[0]), 1.0)
        self.assertEqual(float(mask[1]), 1.0)
        self.assertEqual(float(mask[2]), 0.0)
        self.assertEqual(float(mask[3]), 0.0)
        # t=0: success at t=2 (k=2) -> 1; t=1: success at t=2 (k=1) -> 1.
        self.assertEqual(float(tgt[0]), 1.0)
        self.assertEqual(float(tgt[1]), 1.0)

    def test_near_success_target(self):
        T, N = 4, 1
        # d_hat dips to 0.012 at t=1; tol=0.01 -> near threshold=0.015.
        d_hat = torch.tensor([[0.05], [0.012], [0.05], [0.05]]).reshape(T, N, 1)
        succ = torch.zeros(T, N, 1)
        tol = torch.ones(T, N, 1) * 0.01
        dones = torch.zeros(T, N)
        cache = self._builder().build(
            d_hat=d_hat, success_cnt=succ, success_tol=tol,
            failure=torch.zeros(T, N, 1), lifted=torch.ones(T, N, 1),
            dones=dones, generation=1,
        )
        tgt = cache.targets_flat["near_success_h2"]
        # t=0: future k=1..2 -> d_hat[1]=0.012 <= 0.015 -> 1.
        self.assertEqual(float(tgt[0]), 1.0)

    def test_dhat_improve_target(self):
        T, N = 4, 1
        d_hat = torch.tensor([[0.10], [0.05], [0.10], [0.10]]).reshape(T, N, 1)
        cache = self._builder().build(
            d_hat=d_hat, success_cnt=torch.zeros(T, N, 1),
            success_tol=torch.ones(T, N, 1) * 0.01,
            failure=torch.zeros(T, N, 1), lifted=torch.ones(T, N, 1),
            dones=torch.zeros(T, N), generation=1,
        )
        tgt = cache.targets_flat["dhat_improve_h2"]
        # t=0: min future d_hat=0.05 <= 0.10-0.005 -> 1.
        self.assertEqual(float(tgt[0]), 1.0)

    def test_drop_risk_failure(self):
        T, N = 4, 1
        fail = torch.tensor([[0.0], [1.0], [0.0], [0.0]]).reshape(T, N, 1)
        cache = self._builder().build(
            d_hat=torch.ones(T, N, 1) * 0.1, success_cnt=torch.zeros(T, N, 1),
            success_tol=torch.ones(T, N, 1) * 0.01,
            failure=fail, lifted=torch.ones(T, N, 1),
            dones=torch.zeros(T, N), generation=1,
        )
        tgt = cache.targets_flat["drop_risk_h2"]
        self.assertEqual(float(tgt[0]), 1.0)  # failure at t=1 within k=1..2

    def test_drop_risk_lifted_drop(self):
        T, N = 4, 1
        # lifted 1 -> 0 transition.
        lifted = torch.tensor([[1.0], [0.0], [1.0], [1.0]]).reshape(T, N, 1)
        cache = self._builder().build(
            d_hat=torch.ones(T, N, 1) * 0.1, success_cnt=torch.zeros(T, N, 1),
            success_tol=torch.ones(T, N, 1) * 0.01,
            failure=torch.zeros(T, N, 1), lifted=lifted,
            dones=torch.zeros(T, N), generation=1,
        )
        tgt = cache.targets_flat["drop_risk_h2"]
        self.assertEqual(float(tgt[0]), 1.0)

    def test_done_boundary_masks_target(self):
        T, N = 4, 1
        dones = torch.tensor([[0.0], [1.0], [0.0], [0.0]]).reshape(T, N)
        cache = self._builder().build(
            d_hat=torch.ones(T, N, 1) * 0.1, success_cnt=torch.zeros(T, N, 1),
            success_tol=torch.ones(T, N, 1) * 0.01,
            failure=torch.zeros(T, N, 1), lifted=torch.ones(T, N, 1),
            dones=dones, generation=1,
        )
        mask = cache.masks_flat["near_success_h2"]
        # t=0 horizon covers t=0,1 where done at t=1 -> masked out.
        self.assertEqual(float(mask[0]), 0.0)

    def test_horizon_overflow_mask(self):
        T, N = 3, 1
        cache = self._builder(horizons=(4,)).build(
            d_hat=torch.ones(T, N, 1) * 0.1, success_cnt=torch.zeros(T, N, 1),
            success_tol=torch.ones(T, N, 1) * 0.01,
            failure=torch.zeros(T, N, 1), lifted=torch.ones(T, N, 1),
            dones=torch.zeros(T, N), generation=1,
        )
        mask = cache.masks_flat["near_success_h4"]
        # No t satisfies t+4 < 3 -> all masked.
        self.assertEqual(float(mask.sum()), 0.0)

    def test_pos_rate_and_valid_fraction_recorded(self):
        T, N = 4, 2
        cache = self._builder().build(
            d_hat=torch.ones(T, N, 1) * 0.1, success_cnt=torch.zeros(T, N, 1),
            success_tol=torch.ones(T, N, 1) * 0.01,
            failure=torch.zeros(T, N, 1), lifted=torch.ones(T, N, 1),
            dones=torch.zeros(T, N), generation=1,
        )
        self.assertIn("near_success_h2", cache.pos_rate)
        self.assertIn("near_success_h2", cache.valid_fraction)
        self.assertIn("near_success_h2", cache.degenerate)


# ---------------------------------------------------------------------------
# 3 + 4. gradient routing
# ---------------------------------------------------------------------------


def _fake_targets_masks(keys, mb, device):
    targets = {k: (torch.rand(mb, device=device) > 0.5).float() for k in keys}
    masks = {k: torch.ones(mb, device=device) for k in keys}
    return targets, masks


class TestGradientRouting(unittest.TestCase):
    def _setup(self, feature_source):
        torch.manual_seed(0)
        cfg = FloorCriticConfig(
            enabled=True, feature_source=feature_source, horizons=(2,)
        )
        model = _MockModel()
        heads = FloorCriticHeads(cfg)
        cap = FloorCriticCapture()
        handle = _install_capture(model, cap)
        obs = torch.randn(5, 12)
        _mu, _v, _feat = model(obs)  # fires hook -> cap.tensor (5, feat_dim)
        keys = cfg.head_keys()
        targets, masks = _fake_targets_masks(keys, 5, obs.device)
        return cfg, model, heads, cap, targets, masks, handle

    def test_detached_blocks_backbone_and_actor(self):
        cfg, model, heads, cap, targets, masks, handle = self._setup(
            FEATURE_SOURCE_DETACHED
        )
        # feature_dim mismatch: heads expect cfg.feature_dim(256) but mock
        # feat is 16 -> rebuild heads at the mock dim.
        cfg.feature_dim = cap.tensor.shape[-1]
        heads = FloorCriticHeads(cfg)
        audit = compute_grad_audit(
            rl_games_model=model, heads=heads, capture=cap, cfg=cfg,
            targets=targets, masks=masks,
        )
        self.assertLessEqual(audit["actor_head_grad_from_fc"], 1e-9)
        self.assertLessEqual(audit["shared_backbone_grad_from_fc"], 1e-9)
        self.assertLessEqual(audit["value_head_grad_from_fc"], 1e-9)
        self.assertGreater(audit["aux_head_grad_norm"], 0.0)
        self.assertEqual(audit["actor_grad_blocked"], 1.0)

    def test_shared_updates_backbone_not_actor(self):
        cfg, model, heads, cap, targets, masks, handle = self._setup(
            FEATURE_SOURCE_SHARED
        )
        cfg.feature_dim = cap.tensor.shape[-1]
        heads = FloorCriticHeads(cfg)
        audit = compute_grad_audit(
            rl_games_model=model, heads=heads, capture=cap, cfg=cfg,
            targets=targets, masks=masks,
        )
        # mu head never connected -> 0 grad from FC.
        self.assertLessEqual(audit["actor_head_grad_from_fc"], 1e-9)
        self.assertEqual(audit["actor_grad_blocked"], 1.0)
        # backbone (actor_mlp) DOES receive grad in shared mode.
        self.assertGreater(audit["shared_backbone_grad_from_fc"], 0.0)
        self.assertGreater(audit["aux_head_grad_norm"], 0.0)


# ---------------------------------------------------------------------------
# 5. grad-ratio / auto-scale
# ---------------------------------------------------------------------------


class TestAutoScale(unittest.TestCase):
    def test_auto_scale_caps_ratio(self):
        cfg = FloorCriticConfig(
            enabled=True, feature_source=FEATURE_SOURCE_SHARED,
            horizons=(2,), max_grad_ratio=0.1, auto_scale_by_grad_ratio=True,
            floor_tol=0.02, hard_floor_tol=0.012, start_tol=0.04, weight=1.0,
        )
        cfg.feature_dim = 16
        bridge = FloorCriticBridge(cfg=cfg)
        bridge.heads = FloorCriticHeads(cfg)
        # Tiny PPO grad EMA -> large ratio -> auto-scale kicks in.
        bridge.ppo_grad_ema = 1e-6
        # Build a cache with tol in the floor band so lambda>0.
        builder = FloorCriticTargetBuilder(cfg)
        T, N = 6, 4
        cache = builder.build(
            d_hat=torch.rand(T, N, 1), success_cnt=torch.zeros(T, N, 1),
            success_tol=torch.ones(T, N, 1) * 0.03,
            failure=torch.zeros(T, N, 1), lifted=torch.ones(T, N, 1),
            dones=torch.zeros(T, N), generation=1,
        )
        bridge.cache = cache
        # Fake capture aligned to a minibatch [0, mb).
        mb = 8
        bridge.capture.tensor = torch.randn(mb, 16, requires_grad=True)
        bridge.capture.fired = True
        bridge.agent_ref = _MockAgent(_MockDataset((0, mb)))
        out = compute_floorcritic_loss(bridge)
        self.assertIn("floorcritic_total", out)
        # ratio was huge -> factor < 1.
        self.assertLess(bridge.last_auto_scale_factor, 1.0)
        self.assertGreater(bridge.last_fc_grad_ratio, cfg.max_grad_ratio)

    def test_detached_no_autoscale(self):
        cfg = FloorCriticConfig(
            enabled=True, feature_source=FEATURE_SOURCE_DETACHED,
            horizons=(2,), weight=1.0,
        )
        cfg.feature_dim = 16
        bridge = FloorCriticBridge(cfg=cfg)
        bridge.heads = FloorCriticHeads(cfg)
        builder = FloorCriticTargetBuilder(cfg)
        T, N = 6, 4
        bridge.cache = builder.build(
            d_hat=torch.rand(T, N, 1), success_cnt=torch.zeros(T, N, 1),
            success_tol=torch.ones(T, N, 1) * 0.03,
            failure=torch.zeros(T, N, 1), lifted=torch.ones(T, N, 1),
            dones=torch.zeros(T, N), generation=1,
        )
        mb = 8
        bridge.capture.tensor = torch.randn(mb, 16, requires_grad=True)
        bridge.capture.fired = True
        bridge.agent_ref = _MockAgent(_MockDataset((0, mb)))
        compute_floorcritic_loss(bridge)
        self.assertEqual(bridge.last_auto_scale_factor, 1.0)


# ---------------------------------------------------------------------------
# 6. lambda activation
# ---------------------------------------------------------------------------


class TestLambdaActivation(unittest.TestCase):
    def test_stage_tiers(self):
        cfg = FloorCriticConfig(
            enabled=True, start_tol=0.04, floor_tol=0.02, hard_floor_tol=0.012,
            weight=0.005, floor_weight=0.01, hard_floor_weight=0.02,
        )
        self.assertEqual(cfg.lambda_for_tol(0.05), 0.0)
        self.assertEqual(cfg.lambda_for_tol(0.03), 0.005)
        # tol == floor_tol(0.02): falls in (hard_floor, floor] -> floor_weight.
        self.assertEqual(cfg.lambda_for_tol(0.02), 0.01)
        self.assertEqual(cfg.lambda_for_tol(0.015), 0.01)
        # tol == hard_floor_tol(0.012): falls in tol<=hard_floor -> hard_floor_weight.
        self.assertEqual(cfg.lambda_for_tol(0.012), 0.02)
        self.assertEqual(cfg.lambda_for_tol(0.01), 0.02)

    def test_boundaries(self):
        cfg = FloorCriticConfig(
            enabled=True, start_tol=0.04, floor_tol=0.02, hard_floor_tol=0.012,
            weight=0.005, floor_weight=0.01, hard_floor_weight=0.02,
        )
        # tol == start_tol (0.04): not > start_tol -> base weight.
        self.assertEqual(cfg.lambda_for_tol(0.04), 0.005)
        # just above start_tol -> 0.
        self.assertEqual(cfg.lambda_for_tol(0.0401), 0.0)


# ---------------------------------------------------------------------------
# 7. feature_source switching
# ---------------------------------------------------------------------------


class TestFeatureSource(unittest.TestCase):
    def test_critic_feature_degrades_to_shared(self):
        cfg = FloorCriticConfig(enabled=True, feature_source="critic_feature")
        self.assertEqual(cfg.effective_feature_source, FEATURE_SOURCE_SHARED)
        self.assertFalse(cfg.should_detach)
        self.assertEqual(cfg.feature_source_id, 2)

    def test_detached_vs_shared_ids(self):
        self.assertEqual(
            FloorCriticConfig(feature_source=FEATURE_SOURCE_DETACHED).feature_source_id, 0
        )
        self.assertEqual(
            FloorCriticConfig(feature_source=FEATURE_SOURCE_SHARED).feature_source_id, 1
        )

    def test_invalid_feature_source_raises(self):
        with self.assertRaises(ValueError):
            FloorCriticConfig(feature_source="bogus")


# ---------------------------------------------------------------------------
# 8. integration-ish smoke + optimizer param group
# ---------------------------------------------------------------------------


class TestIntegrationSmoke(unittest.TestCase):
    def test_attach_adds_param_group(self):
        cfg = FloorCriticConfig(enabled=True)
        cfg.feature_dim = 16
        dummy = nn.Linear(4, 4)
        opt = torch.optim.Adam(dummy.parameters(), lr=1e-3)
        n_before = len(opt.param_groups)
        heads = attach_floorcritic_heads(optimizer=opt, cfg=cfg)
        self.assertEqual(len(opt.param_groups), n_before + 1)
        self.assertEqual(opt.param_groups[-1]["name"], "floorcritic_heads")
        self.assertGreater(len(list(heads.parameters())), 0)

    def test_loss_grad_bearing_and_logging(self):
        cfg = FloorCriticConfig(
            enabled=True, feature_source=FEATURE_SOURCE_DETACHED,
            horizons=(2,), weight=0.005,
        )
        cfg.feature_dim = 16
        bridge = FloorCriticBridge(cfg=cfg)
        bridge.heads = FloorCriticHeads(cfg)
        builder = FloorCriticTargetBuilder(cfg)
        T, N = 6, 4
        bridge.cache = builder.build(
            d_hat=torch.rand(T, N, 1), success_cnt=torch.zeros(T, N, 1),
            success_tol=torch.ones(T, N, 1) * 0.03,
            failure=torch.zeros(T, N, 1), lifted=torch.ones(T, N, 1),
            dones=torch.zeros(T, N), generation=1,
        )
        mb = 8
        bridge.capture.tensor = torch.randn(mb, 16, requires_grad=True)
        bridge.capture.fired = True
        bridge.agent_ref = _MockAgent(_MockDataset((0, mb)))
        out = compute_floorcritic_loss(bridge)
        self.assertIn("floorcritic_total", out)
        loss = out["floorcritic_total"]
        self.assertTrue(loss.requires_grad)
        self.assertEqual(bridge.last_active, 1.0)
        self.assertGreater(bridge.last_lambda, 0.0)
        self.assertTrue(len(bridge.last_pos_rate) > 0)
        self.assertTrue(bridge.last_actor_grad_blocked)

    def test_inactive_when_tol_above_start(self):
        cfg = FloorCriticConfig(enabled=True, horizons=(2,), start_tol=0.04)
        cfg.feature_dim = 16
        bridge = FloorCriticBridge(cfg=cfg)
        bridge.heads = FloorCriticHeads(cfg)
        builder = FloorCriticTargetBuilder(cfg)
        T, N = 6, 4
        bridge.cache = builder.build(
            d_hat=torch.rand(T, N, 1), success_cnt=torch.zeros(T, N, 1),
            success_tol=torch.ones(T, N, 1) * 0.075,  # above start_tol
            failure=torch.zeros(T, N, 1), lifted=torch.ones(T, N, 1),
            dones=torch.zeros(T, N), generation=1,
        )
        bridge.capture.tensor = torch.randn(8, 16, requires_grad=True)
        bridge.capture.fired = True
        bridge.agent_ref = _MockAgent(_MockDataset((0, 8)))
        out = compute_floorcritic_loss(bridge)
        self.assertEqual(out, {})
        self.assertEqual(bridge.last_lambda, 0.0)


# ---------------------------------------------------------------------------
# 9. CLI / launcher / plan presence
# ---------------------------------------------------------------------------


class TestCliAndArtifacts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root = Path(__file__).resolve().parents[1]
        cls.train_text = (
            root / "dexpbt_isaaclab" / "scripts" / "train.py"
        ).read_text()
        cls.launch_text = (
            root / "dexpbt_isaaclab" / "scripts" / "launch_pbt.py"
        ).read_text()

    def test_train_declares_flags(self):
        for flag in [
            "--wm_floorcritic_enable",
            "--wm_floorcritic_feature_source",
            "--wm_floorcritic_max_grad_ratio",
            "--wm_floorcritic_auto_scale_by_grad_ratio",
            "--wm_floorcritic_weight",
            "--wm_floorcritic_floor_weight",
            "--wm_floorcritic_hard_floor_weight",
            "--wm_floorcritic_horizons",
        ]:
            self.assertIn(flag, self.train_text, f"missing {flag}")

    def test_launch_pbt_has_floorcritic_booleans(self):
        for flag in [
            '"--wm_floorcritic_enable"',
            '"--no-wm_floorcritic_enable"',
            '"--wm_floorcritic_actor_detach"',
        ]:
            self.assertIn(flag, self.launch_text, f"missing {flag}")


if __name__ == "__main__":
    unittest.main()
