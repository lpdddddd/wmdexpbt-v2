"""Unit tests for EIR-MPC candidate generation + reranking (Phase A)."""

from __future__ import annotations

import types
import unittest

import torch

from dexpbt_isaaclab.world_model.elite_recovery_distill import ERDConfig
from dexpbt_isaaclab.world_model.elite_recovery_bank import (
    EliteRecoveryBank,
    EliteRecoveryBankConfig,
)
from dexpbt_isaaclab.world_model import eir_mpc
from dexpbt_isaaclab.world_model.eir_mpc import (
    EIRConfig,
    generate_eir_candidates,
    rerank_candidates,
    SRC_ACTOR,
    SRC_GAUSSIAN,
    SRC_ELITE_LOCAL,
    SRC_RSSM,
)


class _ProximityScorer:
    """Mock action scorer: success logit high when action is near ``target``."""

    def __init__(self, target, horizons=(4, 8)):
        self.target = target
        self.horizons = horizons

    def __call__(self, feat, action):
        # feat [BK,F], action [BK,A]
        d = (action - self.target.to(action.device)).norm(dim=-1)
        succ = 5.0 - 10.0 * d
        zeros = torch.zeros_like(succ)
        out = {}
        for h in self.horizons:
            out[f"success_h{h}"] = succ
            out[f"improve_h{h}"] = zeros
            out[f"drop_h{h}"] = zeros - 5.0  # low drop prob
        return out


def _make_cache(actions, d_hat, tol, lifted, targets):
    cache = types.SimpleNamespace()
    cache.actions_flat = actions.clamp(-1, 1)
    cache.d_hat_flat = d_hat.reshape(-1)
    cache.success_tol_flat = tol.reshape(-1)
    cache.lifted_flat = lifted.reshape(-1)
    cache.targets_flat = targets
    cache.action_dim = actions.shape[-1]
    return cache


class TestCandidateGeneration(unittest.TestCase):
    def test_local_only_when_no_bank(self):
        cfg = EIRConfig(enabled=True, action_dim=3, feature_dim=4, local_candidate_k=16)
        mu = torch.zeros(5, 3)
        out = generate_eir_candidates(mu, cfg=cfg, bank=None, feat_r=None)
        # actor(1) + gaussian(16)
        self.assertEqual(tuple(out["cand"].shape), (5, 17, 3))
        self.assertEqual(tuple(out["source_id"].shape), (5, 17))
        self.assertTrue(torch.all(out["source_id"][:, 0] == SRC_ACTOR))
        self.assertTrue(torch.all(out["source_id"][:, 1:] == SRC_GAUSSIAN))
        # candidate 0 is clipped actor mean
        self.assertTrue(torch.allclose(out["cand"][:, 0, :], mu.clamp(-1, 1)))

    def test_local_plus_elite_concat_and_metadata(self):
        bank = _filled_bank(target=torch.tensor([0.7, 0.7, 0.7]))
        cfg = EIRConfig(
            enabled=True, action_dim=3, feature_dim=4,
            local_candidate_k=8, elite_candidate_k=4, archive_candidate_k=0,
        )
        mu = torch.zeros(3, 3)
        feat_r = torch.randn(3, 4)
        out = generate_eir_candidates(mu, cfg=cfg, bank=bank, feat_r=feat_r)
        # actor(1) + gaussian(8) + elite(4)
        self.assertEqual(tuple(out["cand"].shape), (3, 13, 3))
        self.assertEqual(tuple(out["source_id"].shape), (3, 13))
        self.assertTrue(torch.any(out["source_id"] == SRC_ELITE_LOCAL))


class TestReranking(unittest.TestCase):
    def test_best_source_is_elite_when_elite_matches_target(self):
        target = torch.tensor([0.7, 0.7, 0.7])
        bank = _filled_bank(target=target)
        cfg = EIRConfig(
            enabled=True, action_dim=3, feature_dim=4,
            local_candidate_k=8, elite_candidate_k=8, archive_candidate_k=0,
            min_score_margin=0.0, min_elite_margin=0.0,
        )
        erd_cfg = ERDConfig(enabled=True, horizons=(4, 8), action_dim=3, feature_dim=4)
        # actor mean far from target so elite candidate should win
        mu = torch.full((4, 3), -0.8)
        feat_r = torch.randn(4, 4)
        out = generate_eir_candidates(mu, cfg=cfg, bank=bank, feat_r=feat_r)
        scorer = _ProximityScorer(target)
        res = rerank_candidates(
            cand=out["cand"], source_id=out["source_id"], priority=out["priority"],
            similarity=out["similarity"], scorer=scorer, feat_r=feat_r,
            cfg=cfg, erd_cfg=erd_cfg,
        )
        self.assertEqual(tuple(res["best_action"].shape), (4, 3))
        self.assertGreater(res["best_source_frac"]["elite"], 0.0)
        # best action should be close to target (elite memory), not actor mean
        self.assertLess(float((res["best_action"] - target).norm(dim=-1).mean()), 0.3)

    def test_best_source_frac_sums_to_one(self):
        cfg = EIRConfig(enabled=True, action_dim=3, feature_dim=4, local_candidate_k=8)
        erd_cfg = ERDConfig(enabled=True, horizons=(4, 8), action_dim=3, feature_dim=4)
        mu = torch.zeros(6, 3)
        feat_r = torch.randn(6, 4)
        out = generate_eir_candidates(mu, cfg=cfg, bank=None, feat_r=feat_r)
        scorer = _ProximityScorer(torch.zeros(3))
        res = rerank_candidates(
            cand=out["cand"], source_id=out["source_id"], priority=out["priority"],
            similarity=out["similarity"], scorer=scorer, feat_r=feat_r,
            cfg=cfg, erd_cfg=erd_cfg,
        )
        total = sum(res["best_source_frac"].values())
        self.assertAlmostEqual(total, 1.0, places=5)

    def test_eligibility_requires_margin(self):
        # actor mean already at target -> no candidate beats margin -> few eligible
        target = torch.zeros(3)
        cfg = EIRConfig(
            enabled=True, action_dim=3, feature_dim=4, local_candidate_k=8,
            elite_candidate_k=0, archive_candidate_k=0, min_score_margin=0.5,
        )
        erd_cfg = ERDConfig(enabled=True, horizons=(4, 8), action_dim=3, feature_dim=4)
        mu = torch.zeros(5, 3)
        feat_r = torch.randn(5, 4)
        out = generate_eir_candidates(mu, cfg=cfg, bank=None, feat_r=feat_r)
        scorer = _ProximityScorer(target)
        res = rerank_candidates(
            cand=out["cand"], source_id=out["source_id"], priority=out["priority"],
            similarity=out["similarity"], scorer=scorer, feat_r=feat_r,
            cfg=cfg, erd_cfg=erd_cfg,
        )
        # with a strict 0.5 margin and actor already optimal, none eligible
        self.assertEqual(int(res["eligible"].sum().item()), 0)


import torch.nn as nn
from dexpbt_isaaclab.world_model.elite_recovery_distill import (
    ERDActionScorer,
    ERDCapture,
    ERDTargetBuilder,
    register_erd_capture,
)
from dexpbt_isaaclab.world_model.recovery_residual import (
    RecoveryResidualConfig,
    RecoveryResidualHead,
    RecoveryResidualBridge,
)
from dexpbt_isaaclab.world_model.eir_mpc import EIRBridge, compute_eir_loss, EIR_OK


class _A2C(nn.Module):
    def __init__(self, obs_dim=12, feat_dim=16, act_dim=4):
        super().__init__()
        self.actor_mlp = nn.Sequential(nn.Linear(obs_dim, feat_dim), nn.Tanh())
        self.mu = nn.Linear(feat_dim, act_dim)
        self.mu_act = nn.Identity()
        self.value = nn.Linear(feat_dim, 1)

    def forward(self, obs):
        f = self.actor_mlp(obs)
        return self.mu_act(self.mu(f)), self.value(f)


class _Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.a2c_network = _A2C()


class _DS:
    def __init__(self, last_range):
        self.last_range = last_range


class _Agent:
    def __init__(self, model, last_range, frame=0):
        self.model = model
        self.dataset = _DS(last_range)
        self.frame = frame


def _build_eir_cache(cfg_h=(2,), T=6, N=2, action_dim=4):
    erd_cfg = ERDConfig(enabled=True, horizons=cfg_h, action_dim=action_dim, feature_dim=16)
    d_hat = torch.full((T, N, 1), 0.015)
    success = torch.zeros(T, N, 1)
    success[3:, 0] = 1.0
    tol = torch.full((T, N, 1), 0.01)
    failure = torch.zeros(T, N, 1)
    lifted = torch.ones(T, N, 1)
    dones = torch.zeros(T, N)
    actions = torch.linspace(-0.4, 0.4, T * N * action_dim).view(T, N, action_dim)
    cache = ERDTargetBuilder(erd_cfg).build(
        d_hat=d_hat, success_cnt=success, success_tol=tol, failure=failure,
        lifted=lifted, dones=dones, actions=actions, generation=1,
    )
    return erd_cfg, cache


class TestComputeEIRLoss(unittest.TestCase):
    def _setup(self, *, residual_enabled=True):
        erd_cfg, cache = _build_eir_cache()
        B = cache.actions_flat.shape[0]  # T*N = 12
        eir_cfg = EIRConfig(
            enabled=True, action_dim=4, feature_dim=16, horizons=(2,),
            local_candidate_k=8, elite_candidate_k=8, archive_candidate_k=0,
            min_score_margin=-1.0, min_elite_margin=-1.0, max_drop_prob_for_teacher=1.0,
            start_after_frames=0, scorer_warmup_iters=0, min_target_valid_fraction=0.0,
            min_risky_fraction=0.0, start_tol=0.04, floor_tol=0.02, hard_floor_tol=0.012,
        )
        model = _Model()
        scorer = ERDActionScorer(erd_cfg)
        capture = ERDCapture()
        handle = register_erd_capture(model, capture)
        res_cfg = RecoveryResidualConfig(
            enabled=residual_enabled, feature_dim=16, hidden_dim=16, action_dim=4,
            distill_weight=1.0, floor_distill_weight=1.0, hard_floor_distill_weight=1.0,
        )
        res_bridge = RecoveryResidualBridge(cfg=res_cfg, head=RecoveryResidualHead(res_cfg))
        bank = EliteRecoveryBank(
            EliteRecoveryBankConfig(enabled=True, bank_size=64, feature_dim=16, action_dim=4, horizons=(2,))
        )
        agent = _Agent(model, last_range=(0, B), frame=0)
        bridge = EIRBridge(
            eir_cfg=eir_cfg, erd_cfg=erd_cfg, scorer=scorer, residual_bridge=res_bridge,
            bank=bank, capture=capture, agent_ref=agent, cache=cache,
        )
        # fire capture: forward actor_mlp over B obs rows
        obs = torch.randn(B, 12)
        model.a2c_network(obs)
        return bridge, model, scorer, res_bridge

    def test_eir_total_returns_dict_and_logs(self):
        bridge, *_ = self._setup()
        out = compute_eir_loss(bridge, frames=0)
        self.assertIn("eir_total", out)
        self.assertEqual(bridge.last_block_reason, EIR_OK)
        self.assertEqual(bridge.last_scorer_active, 1.0)
        self.assertEqual(bridge.last_residual_active, 1.0)
        self.assertIn("actor", bridge.last_best_source_frac)

    def test_scorer_and_residual_grad_separation_no_actor_grad(self):
        # constraint 19.1: EIR loss must not touch main actor mu / actor_mlp.
        bridge, model, scorer, res_bridge = self._setup()
        out = compute_eir_loss(bridge, frames=0)
        out["eir_total"].backward()
        # scorer trained
        sc_grad = sum(
            float(p.grad.abs().sum()) for p in scorer.parameters() if p.grad is not None
        )
        self.assertGreater(sc_grad, 0.0)
        # residual head trained
        rh_grad = sum(
            float(p.grad.abs().sum()) for p in res_bridge.head.parameters() if p.grad is not None
        )
        self.assertGreater(rh_grad, 0.0)
        # main actor mu + actor_mlp receive NO gradient
        mu_grad = sum(
            float(p.grad.abs().sum()) for p in model.a2c_network.mu.parameters() if p.grad is not None
        )
        mlp_grad = sum(
            float(p.grad.abs().sum()) for p in model.a2c_network.actor_mlp.parameters() if p.grad is not None
        )
        self.assertEqual(mu_grad, 0.0)
        self.assertEqual(mlp_grad, 0.0)

    def test_disabled_returns_empty(self):
        bridge, *_ = self._setup()
        bridge.eir_cfg.enabled = False
        self.assertEqual(compute_eir_loss(bridge, frames=0), {})

    def test_rssm_reranker_contributes_candidate_when_active(self):
        # Phase E wire-in: an active reranker with a live belief + predict_fn
        # folds a model-preferred action into the candidate pool as SRC_RSSM.
        from dexpbt_isaaclab.world_model.rssm_mppi_planner import (
            RSSMReranker, RSSMRerankConfig,
        )
        from dexpbt_isaaclab.world_model.eir_mpc import SRC_RSSM

        bridge, model, *_ = self._setup()
        bridge.rssm_reranker = RSSMReranker(
            RSSMRerankConfig(enabled=True, start_after_frames=0, uncertainty_threshold=1.0)
        )

        def _predict_fn(belief_flat, action_flat):
            d = action_flat.norm(dim=-1, keepdim=True)
            succ = 5.0 - 10.0 * d
            return {
                "horizon_logits": {2: {"success": succ, "drop": torch.full_like(succ, -5.0)}},
                "recovery_logits": succ,
            }

        bridge.rssm_predict_fn = _predict_fn
        bridge.rssm_belief_provider = lambda feat_r: torch.zeros(feat_r.shape[0], 8)

        seen = {}
        import dexpbt_isaaclab.world_model.eir_mpc as _eir
        orig_rerank = _eir.rerank_candidates

        def _spy(**kw):
            seen["source_id"] = kw["source_id"]
            return orig_rerank(**kw)

        _eir.rerank_candidates = _spy
        try:
            out = compute_eir_loss(bridge, frames=1000)
        finally:
            _eir.rerank_candidates = orig_rerank
        self.assertEqual(bridge.last_rssm_candidate_added, 1.0)
        self.assertIn("source_id", seen)
        self.assertTrue(bool((seen["source_id"] == SRC_RSSM).any()))

    def test_rssm_reranker_no_candidate_without_belief(self):
        # predict_fn set but no per-row belief -> reranker self-disables, pool
        # is unchanged (documented safe fallback).
        from dexpbt_isaaclab.world_model.rssm_mppi_planner import (
            RSSMReranker, RSSMRerankConfig,
        )

        bridge, *_ = self._setup()
        bridge.rssm_reranker = RSSMReranker(
            RSSMRerankConfig(enabled=True, start_after_frames=0)
        )
        bridge.rssm_predict_fn = lambda b, a: {
            "horizon_logits": {2: {"success": a.norm(dim=-1, keepdim=True), "drop": a.norm(dim=-1, keepdim=True)}},
        }
        bridge.rssm_belief_provider = None  # no belief available
        out = compute_eir_loss(bridge, frames=1000)
        self.assertEqual(bridge.last_rssm_candidate_added, 0.0)


class TestComputeEIRLossWithMPPI(unittest.TestCase):
    """Full H-step MPPI integration into compute_eir_loss."""

    def _setup_with_mppi(self, *, planner_active=True, belief_valid=True, ensemble=False):
        from dexpbt_isaaclab.world_model.rssm_lite import RSSMState
        from dexpbt_isaaclab.world_model.rssm_mppi_planner import (
            RSSMMPPIConfig, RSSMMPPIPlanner,
        )
        from dexpbt_isaaclab.world_model.rssm_belief_provider import RSSMBeliefCache, BELIEF_OK

        base = TestComputeEIRLoss()
        bridge, model, scorer, res_bridge = base._setup()
        B = bridge.cache.actions_flat.shape[0]
        belief_dim = 6

        class _ToyModel:
            def get_state_from_belief(self, belief):
                return RSSMState(belief=belief, deter=belief[:, :3], stoch=belief[:, 3:])

            def imagine_step(self, state, action, *, deterministic=True, sample_temperature=1.0, **kwargs):
                if not isinstance(state, RSSMState):
                    state = self.get_state_from_belief(state)
                bel = state.belief
                Bn = bel.shape[0]
                # reward larger actions (so non-actor sequences win when active)
                mag = action.norm(dim=-1)
                succ = torch.sigmoid(3.0 * mag) if planner_active else torch.zeros(Bn)
                preds = {
                    "success": succ,
                    "drop": torch.zeros(Bn),
                    "recovery": succ,
                    "value": mag,
                    "task_error_delta": torch.zeros(Bn),
                    "prior_std_mean": torch.zeros(Bn),
                    "imagine_mode": "prior_rollout",
                }
                return RSSMState(belief=bel + 0.01 * mag[:, None], deter=bel[:, :3], stoch=bel[:, 3:]), preds

        cfg = RSSMMPPIConfig(
            enabled=True, mode="full", start_after_frames=0, horizon=3,
            num_sequences=16, num_elite=4, cem_iters=1,
            min_sequence_margin=(-1.0 if planner_active else 1e6),
            uncertainty_threshold=1.0,
            use_model_ensemble=bool(ensemble),
            ensemble_min_valid_models=2,
        )
        bridge.rssm_planner = RSSMMPPIPlanner(cfg)
        bridge.rssm_model = _ToyModel()
        belief_flat_ensemble = None
        if ensemble:
            bridge.rssm_models = [bridge.rssm_model, _ToyModel(), _ToyModel()]
            belief_flat_ensemble = [torch.randn(B, belief_dim) for _ in range(3)]
        if belief_valid:
            bridge.rssm_belief_cache = RSSMBeliefCache(
                belief_flat=torch.randn(B, belief_dim), state_flat=None,
                frame=0, generation=0, rows=B, belief_dim=belief_dim,
                valid=True, fallback_reason=BELIEF_OK, source="test",
                belief_flat_ensemble=belief_flat_ensemble,
                ensemble_size=(3 if ensemble else 1),
                ensemble_valid_size=(3 if ensemble else 1),
            )
        else:
            bridge.rssm_belief_cache = RSSMBeliefCache.invalid(3)
        return bridge

    def test_passes_belief_and_appends_rssm_candidate_when_active(self):
        from dexpbt_isaaclab.world_model.eir_mpc import SRC_RSSM
        import dexpbt_isaaclab.world_model.eir_mpc as _eir

        bridge = self._setup_with_mppi(planner_active=True, belief_valid=True)
        seen = {}
        orig = _eir.rerank_candidates

        def _spy(**kw):
            seen["source_id"] = kw["source_id"]
            return orig(**kw)

        _eir.rerank_candidates = _spy
        try:
            compute_eir_loss(bridge, frames=1000)
        finally:
            _eir.rerank_candidates = orig
        self.assertEqual(bridge.last_mppi_active, 1.0)
        self.assertEqual(bridge.last_mppi_belief_valid, 1.0)
        self.assertEqual(bridge.last_rssm_candidate_added, 1.0)
        self.assertTrue(bool((seen["source_id"] == SRC_RSSM).any()))

    def test_no_crash_and_no_candidate_when_mppi_inactive(self):
        bridge = self._setup_with_mppi(planner_active=False, belief_valid=True)
        out = compute_eir_loss(bridge, frames=1000)
        # planner ran (belief valid) but low margin -> inactive, no candidate
        self.assertEqual(bridge.last_rssm_candidate_added, 0.0)
        self.assertEqual(bridge.last_mppi_active, 0.0)

    def test_no_crash_when_belief_invalid(self):
        bridge = self._setup_with_mppi(planner_active=True, belief_valid=False)
        # belief cache invalid -> planner gets None belief -> inactive, no crash
        out = compute_eir_loss(bridge, frames=1000)
        self.assertEqual(bridge.last_rssm_candidate_added, 0.0)

    def test_compute_eir_loss_passes_ensemble_beliefs_to_mppi(self):
        bridge = self._setup_with_mppi(planner_active=True, belief_valid=True, ensemble=True)
        compute_eir_loss(bridge, frames=1000)
        # ensemble path actually engaged (>=2 valid members).
        self.assertEqual(bridge.last_mppi_use_model_ensemble, 1.0)
        self.assertGreaterEqual(bridge.last_mppi_ensemble_valid_size, 2.0)

    def test_compute_eir_loss_ensemble_rssm_candidate_added(self):
        from dexpbt_isaaclab.world_model.eir_mpc import SRC_RSSM
        import dexpbt_isaaclab.world_model.eir_mpc as _eir

        bridge = self._setup_with_mppi(planner_active=True, belief_valid=True, ensemble=True)
        seen = {}
        orig = _eir.rerank_candidates

        def _spy(**kw):
            seen["source_id"] = kw["source_id"]
            return orig(**kw)

        _eir.rerank_candidates = _spy
        try:
            compute_eir_loss(bridge, frames=1000)
        finally:
            _eir.rerank_candidates = orig
        self.assertEqual(bridge.last_mppi_active, 1.0)
        self.assertEqual(bridge.last_rssm_candidate_added, 1.0)
        self.assertTrue(bool((seen["source_id"] == SRC_RSSM).any()))

    def test_compute_eir_loss_ensemble_fallback_no_crash(self):
        # ensemble requested but belief cache lacks ensemble beliefs -> single
        # model fallback, must not crash and must not engage ensemble path.
        bridge = self._setup_with_mppi(planner_active=True, belief_valid=True, ensemble=True)
        bridge.rssm_belief_cache.belief_flat_ensemble = None
        compute_eir_loss(bridge, frames=1000)
        self.assertEqual(bridge.last_mppi_use_model_ensemble, 0.0)


def _filled_bank(*, target):
    cfg = EliteRecoveryBankConfig(
        enabled=True, bank_size=64, feature_dim=4, action_dim=3,
        horizons=(4, 8), tol_threshold=0.02, hard_tol_threshold=0.012,
        near_success_mult=3.0, min_priority=0.0,
    )
    bank = EliteRecoveryBank(cfg)
    n = 16
    actions = target.expand(n, 3) + 0.01 * torch.randn(n, 3)
    actions = actions.clamp(-1, 1)
    d_hat = torch.full((n,), 0.02)
    tol = torch.full((n,), 0.015)
    lifted = torch.ones(n)
    targets = {
        "success_h4": torch.ones(n),
        "improve_h4": torch.zeros(n),
        "drop_h4": torch.zeros(n),
        "success_h8": torch.zeros(n),
        "improve_h8": torch.zeros(n),
        "drop_h8": torch.zeros(n),
    }
    feats = torch.randn(n, 4)
    cache = _make_cache(actions, d_hat, tol, lifted, targets)
    bank.update_from_cache(cache, feats, source_stats={"source_nsucc_mean": 10.0})
    return bank


class TestStageForTol(unittest.TestCase):
    def test_stage_for_tol_returns_mid_when_guidance_enabled(self):
        cfg = EIRConfig(
            enabled=True, guidance_enable=True, guidance_start_tol=0.04,
            floor_tol=0.02, hard_floor_tol=0.012,
        )
        self.assertEqual(eir_mpc._stage_for_tol(cfg, 0.03), 3)   # mid
        self.assertEqual(eir_mpc._stage_for_tol(cfg, 0.015), 1)  # floor
        self.assertEqual(eir_mpc._stage_for_tol(cfg, 0.01), 2)   # hard
        self.assertIsNone(eir_mpc._stage_for_tol(cfg, 0.05))     # above guidance band

    def test_stage_for_tol_no_mid_when_guidance_disabled(self):
        cfg = EIRConfig(enabled=True, guidance_enable=False, floor_tol=0.02, hard_floor_tol=0.012)
        self.assertIsNone(eir_mpc._stage_for_tol(cfg, 0.03))
        self.assertEqual(eir_mpc._stage_for_tol(cfg, 0.015), 1)


class TestRSSMSourceBonus(unittest.TestCase):
    """[B,K]-aligned, source-gated RSSM rerank bonus."""

    def _setup(self, B=2):
        target = torch.tensor([0.5, 0.5, 0.5])
        # actor col closer to target than rssm col -> actor wins without bonus
        actor = torch.full((B, 1, 3), 0.45)
        rssm = torch.full((B, 1, 3), 0.40)
        cand = torch.cat([actor, rssm], dim=1)  # [B,2,3]
        source_id = torch.tensor([[SRC_ACTOR, SRC_RSSM]] * B, dtype=torch.long)
        priority = torch.zeros(B, 2)
        similarity = torch.zeros(B, 2)
        feat_r = torch.randn(B, 4)
        cfg = EIRConfig(
            enabled=True, action_dim=3, feature_dim=4,
            min_score_margin=0.0, min_elite_margin=0.0,
        )
        erd_cfg = ERDConfig(enabled=True, horizons=(4, 8), action_dim=3, feature_dim=4)
        scorer = _ProximityScorer(target)
        mask = source_id == SRC_RSSM
        return dict(
            cand=cand, source_id=source_id, priority=priority, similarity=similarity,
            feat_r=feat_r, cfg=cfg, erd_cfg=erd_cfg, scorer=scorer, mask=mask, B=B,
        )

    def _margin_bk(self, mask, value):
        return torch.where(mask, torch.full(mask.shape, float(value)), torch.zeros(mask.shape))

    def test_rssm_source_bonus_applied_only_to_rssm_candidate(self):
        s = self._setup()
        base = rerank_candidates(
            cand=s["cand"], source_id=s["source_id"], priority=s["priority"],
            similarity=s["similarity"], scorer=s["scorer"], feat_r=s["feat_r"],
            cfg=s["cfg"], erd_cfg=s["erd_cfg"], rssm_source_bonus=0.0,
        )
        # without bonus, actor (closer) wins
        self.assertEqual(base["best_source_frac"]["actor"], 1.0)
        boosted = rerank_candidates(
            cand=s["cand"], source_id=s["source_id"], priority=s["priority"],
            similarity=s["similarity"], scorer=s["scorer"], feat_r=s["feat_r"],
            cfg=s["cfg"], erd_cfg=s["erd_cfg"], rssm_source_bonus=1.0,
            rssm_bonus_margin_scale=1.0, rssm_plan_margin_bk=self._margin_bk(s["mask"], 1.0),
        )
        # large bonus flips the winner to the RSSM candidate
        self.assertEqual(boosted["best_source_frac"]["rssm"], 1.0)
        # actor column score is untouched by the bonus
        self.assertAlmostEqual(base["actor_score_mean"], boosted["actor_score_mean"], places=5)

    def test_rssm_source_bonus_uses_planner_margin(self):
        s = self._setup()
        zero_margin = rerank_candidates(
            cand=s["cand"], source_id=s["source_id"], priority=s["priority"],
            similarity=s["similarity"], scorer=s["scorer"], feat_r=s["feat_r"],
            cfg=s["cfg"], erd_cfg=s["erd_cfg"], rssm_source_bonus=0.5,
            rssm_plan_margin_bk=self._margin_bk(s["mask"], 0.0),
        )
        hi_margin = rerank_candidates(
            cand=s["cand"], source_id=s["source_id"], priority=s["priority"],
            similarity=s["similarity"], scorer=s["scorer"], feat_r=s["feat_r"],
            cfg=s["cfg"], erd_cfg=s["erd_cfg"], rssm_source_bonus=0.5,
            rssm_plan_margin_bk=self._margin_bk(s["mask"], 1.0),
        )
        self.assertAlmostEqual(zero_margin["rssm_source_bonus_mean"], 0.0, places=6)
        self.assertGreater(hi_margin["rssm_source_bonus_mean"], 0.0)

    def test_rssm_source_bonus_penalizes_uncertainty(self):
        s = self._setup()
        no_unc = rerank_candidates(
            cand=s["cand"], source_id=s["source_id"], priority=s["priority"],
            similarity=s["similarity"], scorer=s["scorer"], feat_r=s["feat_r"],
            cfg=s["cfg"], erd_cfg=s["erd_cfg"], rssm_source_bonus=0.5,
            rssm_plan_margin_bk=self._margin_bk(s["mask"], 1.0),
            rssm_bonus_uncertainty_penalty=0.1,
            rssm_plan_uncertainty_bk=self._margin_bk(s["mask"], 0.0),
        )
        hi_unc = rerank_candidates(
            cand=s["cand"], source_id=s["source_id"], priority=s["priority"],
            similarity=s["similarity"], scorer=s["scorer"], feat_r=s["feat_r"],
            cfg=s["cfg"], erd_cfg=s["erd_cfg"], rssm_source_bonus=0.5,
            rssm_plan_margin_bk=self._margin_bk(s["mask"], 1.0),
            rssm_bonus_uncertainty_penalty=0.1,
            rssm_plan_uncertainty_bk=self._margin_bk(s["mask"], 2.0),
        )
        self.assertGreater(no_unc["rssm_source_bonus_mean"], hi_unc["rssm_source_bonus_mean"])

    def test_rssm_source_bonus_small_and_bounded(self):
        s = self._setup()
        bonus_val = 0.05
        # margin clamped to [0,1], so the positive bonus never exceeds bonus_val
        res = rerank_candidates(
            cand=s["cand"], source_id=s["source_id"], priority=s["priority"],
            similarity=s["similarity"], scorer=s["scorer"], feat_r=s["feat_r"],
            cfg=s["cfg"], erd_cfg=s["erd_cfg"], rssm_source_bonus=bonus_val,
            rssm_bonus_margin_scale=1.0, rssm_plan_margin_bk=self._margin_bk(s["mask"], 5.0),
        )
        self.assertLessEqual(res["rssm_source_bonus_mean"], bonus_val + 1e-6)

    def test_actor_candidate_score_unchanged(self):
        s = self._setup()
        base = rerank_candidates(
            cand=s["cand"], source_id=s["source_id"], priority=s["priority"],
            similarity=s["similarity"], scorer=s["scorer"], feat_r=s["feat_r"],
            cfg=s["cfg"], erd_cfg=s["erd_cfg"], rssm_source_bonus=0.0,
        )
        boosted = rerank_candidates(
            cand=s["cand"], source_id=s["source_id"], priority=s["priority"],
            similarity=s["similarity"], scorer=s["scorer"], feat_r=s["feat_r"],
            cfg=s["cfg"], erd_cfg=s["erd_cfg"], rssm_source_bonus=1.0,
            rssm_plan_margin_bk=self._margin_bk(s["mask"], 1.0),
        )
        self.assertAlmostEqual(base["actor_score_mean"], boosted["actor_score_mean"], places=5)


if __name__ == "__main__":
    unittest.main()
