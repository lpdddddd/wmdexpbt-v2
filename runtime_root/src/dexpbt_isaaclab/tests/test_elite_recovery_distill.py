"""ERD-MPC unit tests."""

from __future__ import annotations

import unittest

import torch
import torch.nn as nn

from dexpbt_isaaclab.world_model.elite_recovery_distill import (
    BLOCK_OK,
    BLOCK_LOW_MARGIN,
    BLOCK_TOL_ABOVE_START,
    BLOCK_WARMUP_FRAMES,
    ERDActionScorer,
    ERDBridge,
    ERDCapture,
    ERDConfig,
    ERDTargetBuilder,
    attach_erd_modules,
    compute_erd_grad_audit,
    compute_erd_loss,
    generate_candidate_actions,
    register_erd_capture,
)


class _MockA2CNetwork(nn.Module):
    def __init__(self, obs_dim=12, feat_dim=16, act_dim=4):
        super().__init__()
        self.actor_mlp = nn.Sequential(nn.Linear(obs_dim, feat_dim), nn.Tanh())
        self.mu = nn.Linear(feat_dim, act_dim)
        self.mu_act = nn.Identity()
        self.value = nn.Linear(feat_dim, 1)

    def forward(self, obs):
        feat = self.actor_mlp(obs)
        return self.mu_act(self.mu(feat)), self.value(feat)


class _MockModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.a2c_network = _MockA2CNetwork()

    def forward(self, obs):
        return self.a2c_network(obs)


class _Dataset:
    def __init__(self, last_range=(0, 4)):
        self.last_range = last_range


class _Agent:
    def __init__(self, model, last_range=(0, 4), frame=0):
        self.model = model
        self.dataset = _Dataset(last_range)
        self.frame = frame


class _FlatActionScorer(nn.Module):
    def __init__(self, cfg: ERDConfig):
        super().__init__()
        self.cfg = cfg
        self.bias = nn.Parameter(torch.zeros(()))

    def forward(self, feat, action):
        ref = action.sum(dim=-1) * 0.0 + feat.sum(dim=-1) * 0.0 + self.bias
        out = {}
        for h in self.cfg.horizons:
            out[f"success_h{h}"] = ref
            out[f"improve_h{h}"] = ref
            out[f"drop_h{h}"] = ref - 10.0
        return out


def _cache(cfg: ERDConfig, T=6, N=2, action_dim=4):
    d_hat = torch.tensor(
        [
            [0.20, 0.20],
            [0.18, 0.25],
            [0.10, 0.30],
            [0.08, 0.35],
            [0.30, 0.20],
            [0.31, 0.19],
        ],
        dtype=torch.float32,
    ).view(T, N, 1)
    success = torch.zeros(T, N, 1)
    success[3:, 0] = 1.0
    success[5:, 1] = 1.0
    tol = torch.full((T, N, 1), 0.01)
    failure = torch.zeros(T, N, 1)
    failure[4, 1] = 1.0
    lifted = torch.ones(T, N, 1)
    lifted[4:, 1] = 0.0
    dones = torch.zeros(T, N)
    actions = torch.linspace(-0.5, 0.5, T * N * action_dim).view(T, N, action_dim)
    return ERDTargetBuilder(cfg).build(
        d_hat=d_hat,
        success_cnt=success,
        success_tol=tol,
        failure=failure,
        lifted=lifted,
        dones=dones,
        actions=actions,
        generation=1,
    )


class TestERDTargets(unittest.TestCase):
    def test_success_improve_drop_targets_and_flatten(self):
        cfg = ERDConfig(enabled=True, horizons=(2,), action_dim=4)
        cache = _cache(cfg)
        self.assertEqual(cache.actions_flat.shape, (12, 4))
        self.assertIn("success_h2", cache.targets_flat)
        self.assertIn("improve_h2", cache.targets_flat)
        self.assertIn("drop_h2", cache.targets_flat)
        self.assertGreater(cache.pos_rate["improve_h2"], 0.0)
        self.assertGreater(cache.pos_rate["drop_h2"], 0.0)
        # flat[n*T+t] -> n=0, t=1 sees future success at t=3 inside h=2.
        self.assertEqual(cache.targets_flat["success_h2"][1].item(), 1.0)

    def test_cache_actions_flat_clamped_to_env_usable_space(self):
        cfg = ERDConfig(enabled=True, horizons=(2,), action_dim=4)
        T, N = 5, 1
        d = torch.ones(T, N, 1) * 0.01
        s = torch.zeros(T, N, 1)
        tol = torch.ones(T, N, 1) * 0.01
        dones = torch.zeros(T, N)
        actions = torch.tensor(
            [[[-2.0, -1.2, 1.2, 2.0]]] * T,
            dtype=torch.float32,
        )
        cache = ERDTargetBuilder(cfg).build(
            d_hat=d,
            success_cnt=s,
            success_tol=tol,
            failure=torch.zeros_like(d),
            lifted=torch.ones_like(d),
            dones=dones,
            actions=actions,
            generation=1,
        )
        self.assertTrue(torch.all(cache.actions_flat <= 1.0))
        self.assertTrue(torch.all(cache.actions_flat >= -1.0))
        self.assertTrue(
            torch.allclose(cache.actions_flat[0], torch.tensor([-1.0, -1.0, 1.0, 1.0]))
        )

    def test_done_boundary_and_overflow_mask(self):
        cfg = ERDConfig(enabled=True, horizons=(2,), action_dim=4)
        T, N = 5, 1
        d = torch.ones(T, N, 1)
        s = torch.zeros(T, N, 1)
        tol = torch.ones(T, N, 1) * 0.01
        dones = torch.zeros(T, N)
        dones[1, 0] = 1.0
        actions = torch.zeros(T, N, 4)
        cache = ERDTargetBuilder(cfg).build(
            d_hat=d,
            success_cnt=s,
            success_tol=tol,
            failure=torch.zeros_like(d),
            lifted=torch.ones_like(d),
            dones=dones,
            actions=actions,
            generation=1,
        )
        mask = cache.masks_flat["success_h2"]
        self.assertEqual(mask[0].item(), 0.0)
        self.assertEqual(mask[-1].item(), 0.0)


class TestCandidateActions(unittest.TestCase):
    def test_candidates_include_mean_and_are_clamped_and_delta_capped(self):
        torch.manual_seed(0)
        cfg = ERDConfig(
            enabled=True,
            candidate_k=8,
            action_dim=4,
            action_noise_std=10.0,
            max_action_delta=0.2,
        )
        mu = torch.tensor([[0.9, -0.9, 0.0, 0.5], [2.0, -2.0, 0.1, -0.1]])
        c = generate_candidate_actions(mu, cfg, tol_mean=0.01)
        self.assertEqual(c.shape, (2, 8, 4))
        self.assertTrue(torch.all(c <= 1.0))
        self.assertTrue(torch.all(c >= -1.0))
        self.assertTrue(torch.allclose(c[:, 0], mu.detach().clamp(-1.0, 1.0)))
        delta = (c[:, 1:] - c[:, :1]).abs()
        self.assertLessEqual(float(delta.max().item()), 0.20001)


class TestGradientRouting(unittest.TestCase):
    def test_scorer_and_distill_gradient_routes(self):
        torch.manual_seed(0)
        cfg = ERDConfig(enabled=True, horizons=(2,), feature_dim=16, action_dim=4)
        model = _MockModel()
        scorer = ERDActionScorer(cfg)
        obs = torch.randn(6, 12)
        action = torch.randn(6, 4).clamp(-1, 1)
        target = torch.ones(6)
        mask = torch.ones(6)
        audit = compute_erd_grad_audit(
            rl_games_model=model,
            scorer=scorer,
            cfg=cfg,
            obs=obs,
            action=action,
            target=target,
            mask=mask,
        )
        self.assertGreater(audit["scorer_path_scorer_grad"], 0.0)
        self.assertLessEqual(audit["scorer_path_mu_grad"], 1e-9)
        self.assertLessEqual(audit["scorer_path_actor_mlp_grad"], 1e-9)
        self.assertLessEqual(audit["distill_path_scorer_grad"], 1e-9)
        self.assertGreater(audit["distill_path_mu_grad"], 0.0)
        self.assertGreater(audit["distill_path_actor_mlp_grad"], 0.0)
        self.assertEqual(audit["best_action_requires_grad"], 0.0)


class TestComputeLoss(unittest.TestCase):
    def _setup(self, cfg: ERDConfig, frame=300_000_000):
        torch.manual_seed(0)
        model = _MockModel()
        cap = ERDCapture()
        register_erd_capture(model, cap)
        obs = torch.randn(4, 12)
        model(obs)
        scorer = ERDActionScorer(cfg)
        bridge = ERDBridge(cfg=cfg, scorer=scorer, capture=cap)
        bridge.agent_ref = _Agent(model, (0, 4), frame=frame)
        bridge.cache = _cache(cfg, T=6, N=2, action_dim=4)
        bridge.ppo_grad_ema = 1000.0
        return bridge

    def test_disabled_returns_empty(self):
        bridge = ERDBridge(cfg=ERDConfig(enabled=False))
        self.assertEqual(compute_erd_loss(bridge), {})

    def test_warmup_blocks_distill_but_scorer_trains(self):
        cfg = ERDConfig(
            enabled=True,
            horizons=(2,),
            feature_dim=16,
            action_dim=4,
            start_after_frames=1_000_000_000,
            scorer_warmup_iters=0,
        )
        bridge = self._setup(cfg, frame=100)
        loss = compute_erd_loss(bridge, frames=100)
        self.assertIn("erd_total", loss)
        self.assertEqual(bridge.last_scorer_active, 1.0)
        self.assertEqual(bridge.last_distill_active, 0.0)
        self.assertEqual(bridge.last_distill_block_reason, BLOCK_WARMUP_FRAMES)

    def test_tol_above_start_blocks_distill(self):
        cfg = ERDConfig(
            enabled=True,
            horizons=(2,),
            feature_dim=16,
            action_dim=4,
            start_after_frames=0,
            scorer_warmup_iters=0,
            start_tol=0.005,
        )
        bridge = self._setup(cfg)
        loss = compute_erd_loss(bridge, frames=300_000_000)
        self.assertIn("erd_total", loss)
        self.assertEqual(bridge.last_distill_active, 0.0)
        self.assertEqual(bridge.last_distill_block_reason, BLOCK_TOL_ABOVE_START)

    def test_distill_blocks_on_margin_safely(self):
        cfg = ERDConfig(
            enabled=True,
            horizons=(2,),
            feature_dim=16,
            action_dim=4,
            start_after_frames=0,
            scorer_warmup_iters=0,
            min_target_valid_fraction=0.0,
            min_risky_fraction=0.0,
            min_score_margin=100.0,
            max_drop_prob_for_teacher=1.0,
        )
        bridge = self._setup(cfg)
        loss = compute_erd_loss(bridge, frames=300_000_000)
        self.assertIn("erd_total", loss)
        self.assertEqual(bridge.last_distill_active, 0.0)
        self.assertEqual(bridge.last_distill_block_reason, BLOCK_LOW_MARGIN)

    def test_attach_adds_optimizer_group(self):
        cfg = ERDConfig(enabled=True, feature_dim=16, action_dim=4)
        opt = torch.optim.Adam(_MockModel().parameters(), lr=1e-3)
        before = len(opt.param_groups)
        scorer = attach_erd_modules(optimizer=opt, cfg=cfg)
        self.assertIsInstance(scorer, ERDActionScorer)
        self.assertEqual(len(opt.param_groups), before + 1)

    def test_compute_erd_loss_distill_active_has_actor_grad(self):
        torch.manual_seed(0)
        cfg = ERDConfig(
            enabled=True,
            horizons=(2,),
            feature_dim=16,
            action_dim=4,
            candidate_k=1,
            scorer_loss_weight=0.0,
            distill_loss_weight=1.0,
            floor_distill_loss_weight=1.0,
            hard_floor_distill_loss_weight=1.0,
            start_after_frames=0,
            scorer_warmup_iters=0,
            min_target_valid_fraction=0.0,
            min_risky_fraction=0.0,
            min_score_margin=-1.0,
            max_drop_prob_for_teacher=1.0,
            auto_scale_by_grad_ratio=False,
        )
        model = _MockModel()
        with torch.no_grad():
            model.a2c_network.mu.weight.fill_(0.05)
            model.a2c_network.mu.bias.fill_(1.5)
        cap = ERDCapture()
        register_erd_capture(model, cap)
        obs = torch.randn(4, 12)
        model(obs)
        scorer = _FlatActionScorer(cfg)
        bridge = ERDBridge(cfg=cfg, scorer=scorer, capture=cap)
        bridge.agent_ref = _Agent(model, (0, 4), frame=0)
        bridge.cache = _cache(cfg, T=6, N=2, action_dim=4)
        bridge.ppo_grad_ema = 1000.0

        loss = compute_erd_loss(bridge, frames=0)
        self.assertIn("erd_total", loss)
        self.assertEqual(bridge.last_distill_active, 1.0)
        self.assertEqual(bridge.last_distill_block_reason, BLOCK_OK)
        self.assertGreater(bridge.last_raw_mu_clip_fraction, 0.0)

        loss["erd_total"].backward()

        def grad_norm(params):
            total = 0.0
            for p in params:
                if p.grad is not None:
                    total += float(p.grad.detach().norm().item() ** 2)
            return total ** 0.5

        self.assertGreater(grad_norm(model.a2c_network.mu.parameters()), 0.0)
        self.assertGreater(grad_norm(model.a2c_network.actor_mlp.parameters()), 0.0)
        self.assertLessEqual(grad_norm(scorer.parameters()), 1e-9)

    def test_raw_mu_clip_logging_and_delta_norm_use_clipped_base(self):
        cfg = ERDConfig(
            enabled=True,
            horizons=(2,),
            feature_dim=16,
            action_dim=4,
            candidate_k=1,
            scorer_loss_weight=0.01,
            distill_loss_weight=1.0,
            floor_distill_loss_weight=1.0,
            hard_floor_distill_loss_weight=1.0,
            start_after_frames=0,
            scorer_warmup_iters=0,
            min_target_valid_fraction=0.0,
            min_risky_fraction=0.0,
            min_score_margin=-1.0,
            max_drop_prob_for_teacher=1.0,
            auto_scale_by_grad_ratio=False,
        )
        model = _MockModel()
        with torch.no_grad():
            model.a2c_network.mu.weight.fill_(0.01)
            model.a2c_network.mu.bias.fill_(2.0)
        cap = ERDCapture()
        register_erd_capture(model, cap)
        model(torch.randn(4, 12))
        bridge = ERDBridge(cfg=cfg, scorer=_FlatActionScorer(cfg), capture=cap)
        bridge.agent_ref = _Agent(model, (0, 4), frame=0)
        bridge.cache = _cache(cfg, T=6, N=2, action_dim=4)

        loss = compute_erd_loss(bridge, frames=0)
        self.assertIn("erd_total", loss)
        self.assertGreater(bridge.last_raw_mu_abs_mean, 1.0)
        self.assertEqual(bridge.last_raw_mu_clip_fraction, 1.0)
        # K=1 picks the clipped actor mean; delta is measured from the clipped base,
        # not from raw out-of-bounds mu.
        self.assertLessEqual(bridge.last_action_delta_norm, 1e-9)


if __name__ == "__main__":
    unittest.main()

