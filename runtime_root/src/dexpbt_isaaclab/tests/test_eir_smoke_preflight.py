"""Smoke preflight checks for EIR-MPC wiring (run before smoke launcher).

These are fast unit/integration assertions that the smoke run should satisfy
without needing a live Isaac Sim session.
"""

from __future__ import annotations

import types
import unittest

import torch
import torch.nn as nn

from dexpbt_isaaclab.world_model.elite_recovery_bank import (
    EliteRecoveryBank,
    EliteRecoveryBankConfig,
    FEATURE_SOURCE_ACTOR_MLP,
    FEATURE_SOURCE_FALLBACK,
)
from dexpbt_isaaclab.world_model.elite_recovery_distill import (
    ERDConfig,
    ERDTargetBuilder,
    ERDActionScorer,
    ERDCapture,
    register_erd_capture,
)
from dexpbt_isaaclab.world_model.eir_mpc import (
    EIRBridge,
    EIRConfig,
    compute_eir_loss,
    SRC_RSSM,
)
from dexpbt_isaaclab.world_model.recovery_residual import (
    RecoveryResidualConfig,
    RecoveryResidualHead,
    RecoveryResidualBridge,
    attach_recovery_residual,
    compute_rollout_residual_action,
    APPLY_AUX_MISSING,
    APPLY_BEFORE_WARMUP,
)
from dexpbt_isaaclab.world_model.rssm_mppi_planner import (
    RSSMReranker,
    RSSMRerankConfig,
    RERANK_MODEL_UNAVAILABLE,
)


class _A2C(nn.Module):
    def __init__(self, obs_dim=12, feat_dim=16, act_dim=4):
        super().__init__()
        self.actor_mlp = nn.Sequential(nn.Linear(obs_dim, feat_dim), nn.Tanh())
        self.mu = nn.Linear(feat_dim, act_dim)
        self.mu_act = nn.Identity()

    def forward(self, obs):
        f = self.actor_mlp(obs)
        return self.mu_act(self.mu(f)), torch.zeros(obs.shape[0], 1)


class _Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.a2c_network = _A2C()


class _Agent:
    def __init__(self, model, last_range, frame=0):
        self.model = model
        self.dataset = types.SimpleNamespace(last_range=last_range)
        self.frame = frame


def _elite_rollout_cache(*, T=6, N=2, action_dim=4):
    erd_cfg = ERDConfig(enabled=True, horizons=(2,), action_dim=action_dim, feature_dim=16)
    d_hat = torch.full((T, N, 1), 0.015)
    success = torch.zeros(T, N, 1)
    success[3:, 0] = 1.0
    tol = torch.full((T, N, 1), 0.01)
    lifted = torch.ones(T, N, 1)
    dones = torch.zeros(T, N)
    actions = torch.linspace(-0.4, 0.4, T * N * action_dim).view(T, N, action_dim)
    cache = ERDTargetBuilder(erd_cfg).build(
        d_hat=d_hat, success_cnt=success, success_tol=tol, failure=torch.zeros(T, N, 1),
        lifted=lifted, dones=dones, actions=actions, generation=1,
    )
    return erd_cfg, cache


class TestEIRSmokePreflight(unittest.TestCase):
    def test_rollout_apply_clamps_modified_mus_and_actions(self):
        mus = torch.tensor([[1.5, -1.2, 0.0]])
        actions = torch.tensor([[1.5, -1.2, 0.0]])
        delta = torch.tensor([[-0.8, 0.5, 0.0]])
        out_mus = (mus + delta).clamp(-1.0, 1.0)
        out_actions = (actions + delta).clamp(-1.0, 1.0)
        self.assertTrue(torch.all(out_mus <= 1.0) and torch.all(out_mus >= -1.0))
        self.assertTrue(torch.all(out_actions <= 1.0) and torch.all(out_actions >= -1.0))
        self.assertAlmostEqual(float(out_mus[0, 0]), 0.7)
        self.assertAlmostEqual(float(out_mus[0, 1]), -0.7)

    def test_eir_residual_head_in_optimizer(self):
        cfg = RecoveryResidualConfig(enabled=True, feature_dim=8, hidden_dim=8, action_dim=3)
        opt = torch.optim.Adam([nn.Parameter(torch.zeros(2))], lr=1e-3)
        attach_recovery_residual(optimizer=opt, cfg=cfg)
        names = [g.get("name") for g in opt.param_groups]
        self.assertIn("eir_residual", names)

    def test_eir_bank_updates_after_eligible_rollout(self):
        erd_cfg, cache = _elite_rollout_cache()
        bank = EliteRecoveryBank(
            EliteRecoveryBankConfig(
                enabled=True, bank_size=64, feature_dim=16, action_dim=4, horizons=(2,),
            )
        )
        self.assertEqual(bank.size, 0)
        feats = torch.randn(cache.actions_flat.shape[0], 16)
        bank.update_from_cache(cache, feats, source_stats={"source_nsucc_mean": 10.0})
        self.assertGreater(bank.size, 0)

    def test_bank_feature_source_tracking(self):
        bridge = EIRBridge(
            eir_cfg=EIRConfig(enabled=True), erd_cfg=ERDConfig(enabled=True),
        )
        bridge.record_bank_feature_source(FEATURE_SOURCE_ACTOR_MLP)
        bridge.record_bank_feature_source(FEATURE_SOURCE_FALLBACK)
        self.assertAlmostEqual(bridge.bank_feature_source_actor_fraction, 0.5)
        self.assertAlmostEqual(bridge.bank_feature_source_fallback_fraction, 0.5)

    def test_eir_disables_erd_raw_mu_distill_path(self):
        erd_cfg, cache = _elite_rollout_cache()
        B = cache.actions_flat.shape[0]
        eir_cfg = EIRConfig(
            enabled=True, action_dim=4, feature_dim=16, horizons=(2,),
            local_candidate_k=4, elite_candidate_k=0, archive_candidate_k=0,
            min_score_margin=-1.0, min_elite_margin=-1.0, max_drop_prob_for_teacher=1.0,
            start_after_frames=0, scorer_warmup_iters=0, min_target_valid_fraction=0.0,
            min_risky_fraction=0.0,
        )
        model = _Model()
        scorer = ERDActionScorer(erd_cfg)
        capture = ERDCapture()
        register_erd_capture(model, capture)
        res_cfg = RecoveryResidualConfig(
            enabled=True, feature_dim=16, hidden_dim=16, action_dim=4, mu_base_detach=True,
        )
        res_bridge = RecoveryResidualBridge(cfg=res_cfg, head=RecoveryResidualHead(res_cfg))
        agent = _Agent(model, last_range=(0, B), frame=0)
        bridge = EIRBridge(
            eir_cfg=eir_cfg, erd_cfg=erd_cfg, scorer=scorer, residual_bridge=res_bridge,
            capture=capture, agent_ref=agent, cache=cache,
        )
        model.a2c_network(torch.randn(B, 12))
        out = compute_eir_loss(bridge, frames=0)
        self.assertIn("eir_total", out)
        out["eir_total"].backward()
        mu_grad = sum(
            float(p.grad.abs().sum()) for p in model.a2c_network.mu.parameters() if p.grad is not None
        )
        self.assertEqual(mu_grad, 0.0)

    def test_rssm_reranker_fallback_without_belief_provider(self):
        reranker = RSSMReranker(RSSMRerankConfig(enabled=True, start_after_frames=0))
        cand = torch.randn(2, 5, 3).clamp(-1, 1)
        out = reranker.score_candidates(
            belief_flat=None,
            cand_actions=cand,
            predict_fn=lambda b, a: {"horizon_logits": {2: {"success": a.norm(dim=-1, keepdim=True), "drop": a.norm(dim=-1, keepdim=True)}}},
            frames=1000,
        )
        self.assertFalse(out["active"])
        self.assertEqual(out["fallback_reason"], RERANK_MODEL_UNAVAILABLE)

    def test_rssm_reranker_does_not_add_candidate_without_belief(self):
        erd_cfg, cache = _elite_rollout_cache()
        B = cache.actions_flat.shape[0]
        eir_cfg = EIRConfig(
            enabled=True, action_dim=4, feature_dim=16, horizons=(2,),
            local_candidate_k=4, elite_candidate_k=0, archive_candidate_k=0,
            min_score_margin=-1.0, min_elite_margin=-1.0, max_drop_prob_for_teacher=1.0,
            start_after_frames=0, scorer_warmup_iters=0, min_target_valid_fraction=0.0,
            min_risky_fraction=0.0,
        )
        model = _Model()
        scorer = ERDActionScorer(erd_cfg)
        capture = ERDCapture()
        register_erd_capture(model, capture)
        bridge = EIRBridge(
            eir_cfg=eir_cfg, erd_cfg=erd_cfg, scorer=scorer, capture=capture,
            agent_ref=_Agent(model, last_range=(0, B)), cache=cache,
        )
        bridge.rssm_reranker = RSSMReranker(RSSMRerankConfig(enabled=True, start_after_frames=0))
        bridge.rssm_predict_fn = lambda b, a: {
            "horizon_logits": {2: {"success": a.norm(dim=-1, keepdim=True), "drop": a.norm(dim=-1, keepdim=True)}},
        }
        bridge.rssm_belief_provider = None
        model.a2c_network(torch.randn(B, 12))
        compute_eir_loss(bridge, frames=1000)
        self.assertEqual(bridge.last_rssm_candidate_added, 0.0)

    def test_rssm_imagine_step_available_on_rssm_backbone(self):
        # full MPPI requires get_state_from_belief + imagine_step on the model.
        from dexpbt_isaaclab.world_model.rssm_lite import RSSMLiteWorldModel
        m = RSSMLiteWorldModel(
            obs_dim=12, action_dim=4, obs_embed_dim=16, action_embed_dim=8,
            h_dim=12, z_dim=4, head_hidden=16,
        )
        self.assertTrue(hasattr(m, "get_state_from_belief"))
        self.assertTrue(hasattr(m, "imagine_step"))
        self.assertTrue(m.supports_prior_rollout)
        belief = torch.randn(3, m.belief_dim)
        ns, preds = m.imagine_step(belief, torch.zeros(3, 4))
        self.assertEqual(tuple(ns.belief.shape), (3, m.belief_dim))
        self.assertIn("prior_std_mean", preds)

    def test_mppi_planner_safe_fallback_without_belief(self):
        from dexpbt_isaaclab.world_model.rssm_mppi_planner import (
            RSSMMPPIConfig, RSSMMPPIPlanner, MPPI_NO_BELIEF,
        )
        planner = RSSMMPPIPlanner(RSSMMPPIConfig(enabled=True, mode="full", start_after_frames=0))
        actor = torch.zeros(4, 4)
        out = planner.plan(
            belief_flat=None, actor_action=actor,
            candidate_actions=torch.randn(4, 2, 4).clamp(-1, 1),
            rssm_model=None, frames=1000,
        )
        self.assertFalse(out["active"])
        # no belief / no model => safe fallback, actor action returned unchanged
        self.assertTrue(torch.allclose(out["best_action"], actor))

    def test_residual_apply_not_aux_missing_after_warmup_with_fresh_aux(self):
        cfg = RecoveryResidualConfig(
            enabled=True, feature_dim=8, hidden_dim=8, action_dim=3,
            apply_to_rollout=True, apply_after_frames=100, gate_confidence_threshold=0.0,
            apply_max_aux_age_frames=10**9,
        )
        bridge = RecoveryResidualBridge(cfg=cfg, head=RecoveryResidualHead(cfg))
        feat = torch.randn(3, 8)
        mu = torch.zeros(3, 3)
        bridge.set_aux_snapshot(
            d_hat=torch.full((3,), 0.01), tol=torch.full((3,), 0.015),
            lifted=torch.ones(3), frame=500, generation=2,
        )
        res = compute_rollout_residual_action(
            bridge, feat=feat, mu=mu, frames=500, rollout_generation=2,
        )
        self.assertNotEqual(res["fallback_reason"], APPLY_AUX_MISSING)
        self.assertNotEqual(res["fallback_reason"], APPLY_BEFORE_WARMUP)

    def test_eir_mppi_stoch_preflight_belief_cache_and_planner(self):
        from dexpbt_isaaclab.world_model.rssm_belief_provider import (
            build_rssm_belief_cache,
            BELIEF_OK,
        )
        from dexpbt_isaaclab.world_model.rssm_lite import RSSMLiteWorldModel
        from dexpbt_isaaclab.world_model.rssm_mppi_planner import (
            RSSMMPPIConfig,
            RSSMMPPIPlanner,
        )

        T, N, OBS, ACT = 6, 4, 12, 4
        td = {
            "obses": torch.randn(T, N, OBS),
            "actions": torch.randn(T, N, ACT).clamp(-1, 1),
            "rewards": torch.randn(T, N, 1),
            "dones": torch.zeros(T, N, 1),
            "aux_d_hat": torch.rand(T, N, 1),
            "aux_failure": torch.zeros(T, N, 1),
            "aux_lifted": torch.ones(T, N, 1),
            "aux_success_cnt": torch.zeros(T, N, 1),
            "aux_success_tolerance": torch.full((T, N, 1), 0.02),
            "aux_picked": torch.zeros(T, N, 1),
            "aux_dropped": torch.zeros(T, N, 1),
            "aux_closest_d_hat": torch.rand(T, N, 1),
        }
        exp_buf = types.SimpleNamespace(tensor_dict=td)
        model = RSSMLiteWorldModel(
            obs_dim=OBS, action_dim=ACT, obs_embed_dim=32, action_embed_dim=16,
            h_dim=24, z_dim=8, head_hidden=32, aux_dim=12, aux_embed_dim=16,
        )
        self.assertEqual(model.aux_dim, 12)
        cache = build_rssm_belief_cache(
            seq_wm_trainer=types.SimpleNamespace(model=model, device=torch.device("cpu")),
            experience_buffer=exp_buf,
            returns_full=torch.randn(T, N),
            last_obs=torch.randn(N, OBS),
            last_d_hat=torch.rand(N, 1),
            last_success_count=torch.zeros(N, 1),
            frames=1000,
            generation=1,
            device=torch.device("cpu"),
        )
        print("[EIR-MPPI preflight] belief_cache_valid expected under task_aux_inputs=True")
        print("[EIR-MPPI preflight] stochastic_prior_rollout expected after warmup")
        self.assertTrue(cache.valid)
        self.assertEqual(cache.fallback_reason, BELIEF_OK)
        self.assertIsNotNone(cache.belief_flat)
        planner = RSSMMPPIPlanner(
            RSSMMPPIConfig(
                enabled=True,
                mode="full",
                stochastic_rollouts=True,
                num_latent_samples=4,
                start_after_frames=0,
            )
        )
        self.assertTrue(planner.cfg.stochastic_rollouts)
        self.assertEqual(planner.cfg.num_latent_samples, 4)


if __name__ == "__main__":
    unittest.main()
