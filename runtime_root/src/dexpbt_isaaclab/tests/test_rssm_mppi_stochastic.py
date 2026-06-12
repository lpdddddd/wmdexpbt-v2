"""Tests for stochastic risk-sensitive RSSM-MPPI planner."""

from __future__ import annotations

import unittest
from unittest import mock

import torch

from dexpbt_isaaclab.world_model.rssm_lite import RSSMState
from dexpbt_isaaclab.world_model.rssm_mppi_planner import (
    RSSMMPPIConfig,
    RSSMMPPIPlanner,
    _tail_mean,
)


A, D = 3, 6


class _ToyRSSM:
    """Toy RSSM whose per-step score depends on action norm; stochastic mode
    injects sample-dependent noise into success/drop."""

    def __init__(self, target, base_unc=0.05):
        self.target = target
        self.base_unc = float(base_unc)
        self.last_deterministic = True

    def get_state_from_belief(self, belief):
        return RSSMState(belief=belief, deter=belief[:, :D // 2], stoch=belief[:, D // 2:])

    def imagine_step(self, state, action, *, deterministic=True, sample_temperature=1.0, **kwargs):
        self.last_deterministic = bool(deterministic)
        if not isinstance(state, RSSMState):
            state = self.get_state_from_belief(state)
        belief = state.belief
        b = belief.shape[0]
        dev = belief.device
        d = (action - self.target.to(action.device)).norm(dim=-1)
        succ = torch.sigmoid(5.0 - 10.0 * d)
        drop = torch.sigmoid(-5.0 + 8.0 * d)
        if not deterministic:
            g = torch.randn(b, device=dev)
            succ = (succ + 0.35 * g * float(sample_temperature)).clamp(0.0, 1.0)
            drop = (drop + 0.35 * g.abs() * float(sample_temperature)).clamp(0.0, 1.0)
            mode = "stochastic_prior_rollout"
        else:
            mode = "prior_rollout"
        shift = action.mean(dim=-1, keepdim=True)
        next_belief = belief + 0.01 * shift
        preds = {
            "success": succ,
            "drop": drop,
            "recovery": succ.clone(),
            "value": -d,
            "task_error_delta": d * 0.1,
            "prior_std_mean": torch.full((b,), self.base_unc, device=dev),
            "imagine_mode": mode,
        }
        return RSSMState(
            belief=next_belief, deter=next_belief[:, :D // 2], stoch=next_belief[:, D // 2:]
        ), preds


def _cfg(**kw):
    base = dict(
        enabled=True,
        mode="full",
        start_after_frames=0,
        horizon=4,
        num_sequences=8,
        num_elite=4,
        cem_iters=1,
        uncertainty_threshold=10.0,
        min_sequence_margin=-1.0,
        include_elite_sequence=False,
    )
    base.update(kw)
    return RSSMMPPIConfig(**base)


class TestTailMean(unittest.TestCase):
    def test_top_tail(self):
        x = torch.tensor([[[1.0, 2.0, 10.0, 3.0]]])
        out = _tail_mean(x, alpha=0.25, largest=True)
        self.assertAlmostEqual(float(out.item()), 10.0)

    def test_bottom_tail(self):
        x = torch.tensor([[[5.0, 1.0, 2.0, 3.0]]])
        out = _tail_mean(x, alpha=0.25, largest=False)
        self.assertAlmostEqual(float(out.item()), 1.0)


class TestStochasticMPPI(unittest.TestCase):
    def _inputs(self, b=4):
        belief = torch.randn(b, D)
        actor = torch.zeros(b, A)
        cand = torch.randn(b, 2, A).clamp(-1, 1)
        return belief, actor, cand

    def test_mppi_stochastic_sequence_shape(self):
        planner = RSSMMPPIPlanner(_cfg(stochastic_rollouts=True, num_latent_samples=4))
        belief, actor, cand = self._inputs()
        out = planner.plan(
            belief_flat=belief,
            actor_action=actor,
            candidate_actions=cand,
            rssm_model=_ToyRSSM(torch.tensor([0.8, 0.0, 0.0])),
            frames=1000,
        )
        self.assertTrue(out["active"])
        self.assertEqual(tuple(out["best_sequence"].shape), (4, 4, A))
        self.assertEqual(planner.last_stochastic_rollouts, 1.0)
        self.assertEqual(planner.last_num_latent_samples, 4.0)

    def test_mppi_stochastic_uses_imagine_step_deterministic_false(self):
        model = _ToyRSSM(torch.zeros(A))
        planner = RSSMMPPIPlanner(_cfg(stochastic_rollouts=True, num_latent_samples=3))
        belief, actor, cand = self._inputs(b=2)
        planner.plan(
            belief_flat=belief,
            actor_action=actor,
            candidate_actions=cand,
            rssm_model=model,
            frames=1000,
        )
        self.assertFalse(model.last_deterministic)

    def test_mppi_stochastic_score_samples_shape(self):
        planner = RSSMMPPIPlanner(_cfg(stochastic_rollouts=True, num_latent_samples=5))
        belief, actor, cand = self._inputs(b=3)
        seq = actor[:, None, None, :].expand(3, 2, 4, A).clone()
        out = planner._rollout_and_score(
            belief_flat=belief,
            seq=seq,
            actor_action=actor,
            rssm_model=_ToyRSSM(torch.zeros(A)),
        )
        self.assertEqual(tuple(out["score"].shape), (3, 2))
        self.assertEqual(tuple(out["score_std"].shape), (3, 2))
        self.assertEqual(tuple(out["drop_cvar"].shape), (3, 2))

    def test_mppi_risk_beta_penalizes_high_variance_sequence(self):
        target = torch.tensor([0.9, 0.0, 0.0])
        model = _ToyRSSM(target)
        planner = RSSMMPPIPlanner(
            _cfg(
                stochastic_rollouts=True,
                num_latent_samples=16,
                num_sequences=0,
                include_elite_sequence=False,
                score_aggregation="mean_minus_std",
                risk_beta=2.0,
                drop_cvar_weight=0.0,
            )
        )
        b = 1
        actor = torch.zeros(b, A)
        stable = actor[:, None, None, :].expand(b, 1, 4, A).clone()
        risky = torch.full((b, 1, 4, A), 0.95)
        seq = torch.cat([stable, risky], dim=1)
        out = planner._rollout_and_score(
            belief_flat=torch.randn(b, D),
            seq=seq,
            actor_action=actor,
            rssm_model=model,
        )
        self.assertGreater(float(out["score"][0, 0].item()), float(out["score"][0, 1].item()))

    def test_mppi_drop_cvar_penalizes_tail_drop_sequence(self):
        class _DropSpikeRSSM(_ToyRSSM):
            def imagine_step(self, state, action, *, deterministic=True, sample_temperature=1.0, **kwargs):
                ns, preds = super().imagine_step(
                    state, action, deterministic=deterministic, sample_temperature=sample_temperature
                )
                if not deterministic and action[:, 0].mean() > 0.5:
                    spike = torch.rand_like(preds["drop"]) > 0.6
                    preds["drop"] = torch.where(spike, torch.ones_like(preds["drop"]), preds["drop"] * 0.1)
                return ns, preds

        planner = RSSMMPPIPlanner(
            _cfg(
                stochastic_rollouts=True,
                num_latent_samples=12,
                num_sequences=0,
                include_elite_sequence=False,
                score_aggregation="mean_minus_std",
                risk_beta=0.0,
                drop_cvar_weight=3.0,
            )
        )
        b = 1
        actor = torch.zeros(b, A)
        low = actor[:, None, None, :].expand(b, 1, 4, A).clone()
        high = torch.full((b, 1, 4, A), 0.9)
        seq = torch.cat([low, high], dim=1)
        out = planner._rollout_and_score(
            belief_flat=torch.randn(b, D),
            seq=seq,
            actor_action=actor,
            rssm_model=_DropSpikeRSSM(torch.zeros(A)),
        )
        self.assertGreater(float(out["score"][0, 0].item()), float(out["score"][0, 1].item()))

    def test_mppi_stochastic_falls_back_to_deterministic_when_num_samples_1(self):
        planner = RSSMMPPIPlanner(_cfg(stochastic_rollouts=True, num_latent_samples=1))
        model = _ToyRSSM(torch.zeros(A))
        belief, actor, cand = self._inputs(b=2)
        planner.plan(
            belief_flat=belief,
            actor_action=actor,
            candidate_actions=cand,
            rssm_model=model,
            frames=1000,
        )
        self.assertEqual(planner.last_stochastic_rollouts, 0.0)
        self.assertTrue(model.last_deterministic)

    def test_mppi_chunking_matches_no_chunking_small_case(self):
        # Deterministic rollouts: chunking must not change scores (RNG batching differs in stochastic mode).
        cfg_small = _cfg(stochastic_rollouts=False, max_rollout_batch=1000)
        cfg_chunk = _cfg(stochastic_rollouts=False, max_rollout_batch=2)
        planner_a = RSSMMPPIPlanner(cfg_small)
        planner_b = RSSMMPPIPlanner(cfg_chunk)
        belief = torch.randn(2, D)
        actor = torch.zeros(2, A)
        seq = torch.randn(2, 3, 4, A).clamp(-1, 1)
        model = _ToyRSSM(torch.zeros(A))
        out_a = planner_a._rollout_and_score(
            belief_flat=belief, seq=seq, actor_action=actor, rssm_model=model
        )
        out_b = planner_b._rollout_and_score(
            belief_flat=belief, seq=seq, actor_action=actor, rssm_model=model
        )
        self.assertTrue(torch.allclose(out_a["score"], out_b["score"], atol=1e-5))

    def test_mppi_no_grad_stochastic(self):
        planner = RSSMMPPIPlanner(_cfg(stochastic_rollouts=True, num_latent_samples=4))
        belief = torch.randn(2, D, requires_grad=True)
        actor = torch.zeros(2, A, requires_grad=True)
        seq = actor[:, None, None, :].expand(2, 2, 4, A).clone()
        with mock.patch.object(planner, "_flat_rollout_scores", wraps=planner._flat_rollout_scores) as spy:
            out = planner._rollout_and_score(
                belief_flat=belief,
                seq=seq,
                actor_action=actor,
                rssm_model=_ToyRSSM(torch.zeros(A)),
            )
            self.assertFalse(out["score"].requires_grad)
            self.assertFalse(spy.call_args.kwargs["belief_flat"].requires_grad)


if __name__ == "__main__":
    unittest.main()
