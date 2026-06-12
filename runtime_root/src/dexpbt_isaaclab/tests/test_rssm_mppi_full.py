"""Tests for the full H-step RSSM-MPPI planner (EIR-MPC)."""

from __future__ import annotations

import unittest

import torch

from dexpbt_isaaclab.world_model.rssm_lite import RSSMState
from dexpbt_isaaclab.world_model.rssm_mppi_planner import (
    RSSMMPPIConfig,
    RSSMMPPIPlanner,
    MPPI_OK,
    MPPI_DISABLED,
    MPPI_BEFORE_START,
    MPPI_NO_BELIEF,
    MPPI_NO_RSSM_MODEL,
    MPPI_HIGH_UNCERTAINTY,
    MPPI_LOW_MARGIN,
)

A, D = 3, 6


class _ToyRSSM:
    """Toy model with a real action-conditioned prior + heads that reward
    actions near ``target`` (success high, drop low). Belief advances by a
    deterministic linear map so sequences differ by action."""

    def __init__(self, target, uncertainty=0.0):
        self.target = target
        self._unc = float(uncertainty)

    def get_state_from_belief(self, belief):
        return RSSMState(belief=belief, deter=belief[:, :D // 2], stoch=belief[:, D // 2:])

    def imagine_step(self, state, action, *, deterministic=True, sample_temperature=1.0, **kwargs):
        if not isinstance(state, RSSMState):
            state = self.get_state_from_belief(state)
        belief = state.belief
        B = belief.shape[0]
        dev = belief.device
        d = (action - self.target.to(action.device)).norm(dim=-1)  # [B]
        succ = torch.sigmoid(5.0 - 10.0 * d)
        drop = torch.sigmoid(-5.0 + 0.0 * d)
        rec = succ.clone()
        value = -d
        task_delta = d * 0.1  # positive delta = worse; planner rewards -sum
        # advance belief deterministically by action mean (keeps it action-dep)
        shift = action.mean(dim=-1, keepdim=True)
        next_belief = belief + 0.01 * shift
        preds = {
            "success": succ, "drop": drop, "recovery": rec,
            "value": value, "task_error_delta": task_delta,
            "prior_std_mean": torch.full((B,), self._unc, device=dev),
            "imagine_mode": "prior_rollout",
        }
        return RSSMState(belief=next_belief, deter=next_belief[:, :D // 2], stoch=next_belief[:, D // 2:]), preds


def _cfg(**kw):
    base = dict(
        enabled=True, mode="full", start_after_frames=0, horizon=4,
        num_sequences=32, num_elite=4, cem_iters=1,
        uncertainty_threshold=0.4, min_sequence_margin=-1.0,
    )
    base.update(kw)
    return RSSMMPPIConfig(**base)


class TestMPPIPlanner(unittest.TestCase):
    def _inputs(self, B=5, K=4):
        belief = torch.randn(B, D)
        actor = torch.zeros(B, A)
        cand = torch.randn(B, K, A).clamp(-1, 1)
        return belief, actor, cand

    def test_disabled_fallback(self):
        planner = RSSMMPPIPlanner(_cfg(enabled=False))
        belief, actor, cand = self._inputs()
        out = planner.plan(belief_flat=belief, actor_action=actor,
                           candidate_actions=cand, rssm_model=_ToyRSSM(torch.zeros(A)), frames=1000)
        self.assertFalse(out["active"])
        self.assertEqual(out["fallback_reason"], MPPI_DISABLED)

    def test_before_start_fallback(self):
        planner = RSSMMPPIPlanner(_cfg(start_after_frames=10**9))
        belief, actor, cand = self._inputs()
        out = planner.plan(belief_flat=belief, actor_action=actor,
                           candidate_actions=cand, rssm_model=_ToyRSSM(torch.zeros(A)), frames=10)
        self.assertEqual(out["fallback_reason"], MPPI_BEFORE_START)

    def test_no_belief_fallback(self):
        planner = RSSMMPPIPlanner(_cfg())
        _b, actor, cand = self._inputs()
        out = planner.plan(belief_flat=None, actor_action=actor,
                           candidate_actions=cand, rssm_model=_ToyRSSM(torch.zeros(A)), frames=1000)
        self.assertEqual(out["fallback_reason"], MPPI_NO_BELIEF)
        self.assertTrue(torch.allclose(out["best_action"], actor))

    def test_no_rssm_model_fallback(self):
        planner = RSSMMPPIPlanner(_cfg())
        belief, actor, cand = self._inputs()
        out = planner.plan(belief_flat=belief, actor_action=actor,
                           candidate_actions=cand, rssm_model=object(), frames=1000)
        self.assertEqual(out["fallback_reason"], MPPI_NO_RSSM_MODEL)

    def test_sequence_and_rollout_shape(self):
        cfg = _cfg(horizon=4, num_sequences=16)
        planner = RSSMMPPIPlanner(cfg)
        belief, actor, cand = self._inputs(B=5, K=3)
        out = planner.plan(belief_flat=belief, actor_action=actor,
                           candidate_actions=cand, rssm_model=_ToyRSSM(torch.zeros(A)), frames=1000)
        self.assertTrue(out["active"])
        self.assertEqual(tuple(out["best_sequence"].shape), (5, 4, A))
        self.assertEqual(tuple(out["best_action"].shape), (5, A))
        self.assertEqual(tuple(out["score_margin"].shape), (5,))

    def test_selects_best_sequence_toward_target(self):
        # target action favored; best_action should move toward target vs actor=0
        target = torch.tensor([0.8, 0.8, 0.8])
        planner = RSSMMPPIPlanner(_cfg(num_sequences=128, action_noise_std=0.4))
        B = 6
        belief = torch.randn(B, D)
        actor = torch.zeros(B, A)
        cand = torch.zeros(B, 1, A)
        out = planner.plan(belief_flat=belief, actor_action=actor,
                           candidate_actions=cand, rssm_model=_ToyRSSM(target), frames=1000)
        self.assertTrue(out["active"])
        # best first action should be closer to target than the actor (0)
        d_best = (out["best_action"] - target).norm(dim=-1)
        d_actor = (actor - target).norm(dim=-1)
        self.assertTrue(bool((d_best < d_actor).float().mean() > 0.5))

    def test_actor_baseline_and_margin(self):
        planner = RSSMMPPIPlanner(_cfg(num_sequences=64, action_noise_std=0.3))
        belief, actor, cand = self._inputs(B=4)
        out = planner.plan(belief_flat=belief, actor_action=actor,
                           candidate_actions=cand, rssm_model=_ToyRSSM(torch.tensor([0.7, 0.7, 0.7])), frames=1000)
        # best score >= actor score (best includes actor column)
        self.assertTrue(bool((out["best_score"] >= out["actor_score"] - 1e-5).all()))
        self.assertTrue(torch.allclose(out["score_margin"], out["best_score"] - out["actor_score"], atol=1e-4))

    def test_low_margin_fallback(self):
        # huge min margin -> never accepted
        planner = RSSMMPPIPlanner(_cfg(min_sequence_margin=1e6))
        belief, actor, cand = self._inputs()
        out = planner.plan(belief_flat=belief, actor_action=actor,
                           candidate_actions=cand, rssm_model=_ToyRSSM(torch.tensor([0.7, 0.7, 0.7])), frames=1000)
        self.assertFalse(out["active"])
        self.assertEqual(out["fallback_reason"], MPPI_LOW_MARGIN)

    def test_high_uncertainty_fallback(self):
        planner = RSSMMPPIPlanner(_cfg(uncertainty_threshold=0.1))
        belief, actor, cand = self._inputs()
        out = planner.plan(belief_flat=belief, actor_action=actor,
                           candidate_actions=cand, rssm_model=_ToyRSSM(torch.zeros(A), uncertainty=0.9), frames=1000)
        self.assertFalse(out["active"])
        self.assertEqual(out["fallback_reason"], MPPI_HIGH_UNCERTAINTY)

    def test_cem_iters_runs(self):
        planner = RSSMMPPIPlanner(_cfg(cem_iters=3, num_sequences=32, num_elite=6))
        belief, actor, cand = self._inputs(B=4)
        out = planner.plan(belief_flat=belief, actor_action=actor,
                           candidate_actions=cand, rssm_model=_ToyRSSM(torch.tensor([0.6, 0.6, 0.6])), frames=1000)
        self.assertTrue(out["active"])
        self.assertEqual(out["fallback_reason"], MPPI_OK)

    def test_no_grad_through_rssm(self):
        planner = RSSMMPPIPlanner(_cfg())
        belief = torch.randn(4, D, requires_grad=True)
        actor = torch.zeros(4, A, requires_grad=True)
        cand = torch.randn(4, 2, A, requires_grad=True).clamp(-1, 1)
        out = planner.plan(belief_flat=belief, actor_action=actor,
                           candidate_actions=cand, rssm_model=_ToyRSSM(torch.zeros(A)), frames=1000)
        self.assertFalse(out["best_action"].requires_grad)
        self.assertFalse(out["best_score"].requires_grad)


if __name__ == "__main__":
    unittest.main()
