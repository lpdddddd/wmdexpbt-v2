"""Tests for RSSM-lite prior-only ``imagine_step`` (EIR-MPC full MPPI)."""

from __future__ import annotations

import unittest

import torch
import torch.nn as nn

from dexpbt_isaaclab.world_model.rssm_lite import (
    RSSMLiteWorldModel,
    RSSMState,
    IMAGINE_PRIOR_ROLLOUT,
    IMAGINE_STOCHASTIC_PRIOR_ROLLOUT,
    IMAGINE_FIXED_BELIEF_FALLBACK,
)


def _model(**kw):
    base = dict(
        obs_dim=12, action_dim=4, obs_embed_dim=32, action_embed_dim=16,
        h_dim=24, z_dim=8, head_hidden=32,
        with_value_head=True, with_success_head=True, with_task_error_head=True,
        with_horizon_heads=True, horizon_steps=(2, 4), with_recovery_head=True,
        recovery_horizon=4,
    )
    base.update(kw)
    return RSSMLiteWorldModel(**base)


class TestImagineStep(unittest.TestCase):
    def test_get_state_from_belief_split(self):
        m = _model()
        belief = torch.randn(5, m.belief_dim)
        st = m.get_state_from_belief(belief)
        self.assertIsInstance(st, RSSMState)
        self.assertEqual(tuple(st.deter.shape), (5, m.h_dim))
        self.assertEqual(tuple(st.stoch.shape), (5, m.z_dim))

    def test_imagine_step_shape(self):
        m = _model()
        belief = torch.randn(7, m.belief_dim)
        action = torch.randn(7, 4).clamp(-1, 1)
        next_state, preds = m.imagine_step(belief, action)
        self.assertEqual(tuple(next_state.belief.shape), (7, m.belief_dim))
        self.assertIn("prior_std_mean", preds)
        self.assertEqual(tuple(preds["prior_std_mean"].shape), (7,))

    def test_imagine_step_no_grad(self):
        m = _model()
        belief = torch.randn(4, m.belief_dim, requires_grad=True)
        action = torch.randn(4, 4, requires_grad=True).clamp(-1, 1)
        next_state, preds = m.imagine_step(belief, action)
        self.assertFalse(next_state.belief.requires_grad)
        for k, v in preds.items():
            if isinstance(v, torch.Tensor):
                self.assertFalse(v.requires_grad, f"{k} should be detached")

    def test_imagine_step_predicts_required_terms(self):
        m = _model()
        belief = torch.randn(6, m.belief_dim)
        action = torch.randn(6, 4).clamp(-1, 1)
        _ns, preds = m.imagine_step(belief, action)
        # value + success + drop + recovery available with these heads
        self.assertIn("value", preds)
        self.assertIn("success", preds)
        self.assertIn("drop", preds)
        self.assertIn("recovery", preds)
        self.assertEqual(preds.get("imagine_mode"), IMAGINE_PRIOR_ROLLOUT)

    def test_imagine_step_advances_belief_with_prior(self):
        m = _model()
        belief = torch.randn(3, m.belief_dim)
        a1 = torch.full((3, 4), 0.3)
        a2 = torch.full((3, 4), -0.7)
        ns1, _ = m.imagine_step(belief, a1)
        ns2, _ = m.imagine_step(belief, a2)
        # different actions -> different next belief (prior is action-conditioned)
        self.assertFalse(torch.allclose(ns1.belief, ns2.belief))
        # belief should actually move from the start
        self.assertFalse(torch.allclose(ns1.belief, belief))

    def test_imagine_step_deterministic_default_unchanged(self):
        m = _model()
        belief = torch.randn(4, m.belief_dim)
        action = torch.randn(4, 4).clamp(-1, 1)
        ns1, p1 = m.imagine_step(belief, action)
        ns2, p2 = m.imagine_step(belief, action, deterministic=True)
        self.assertEqual(p1.get("imagine_mode"), IMAGINE_PRIOR_ROLLOUT)
        self.assertEqual(p2.get("imagine_mode"), IMAGINE_PRIOR_ROLLOUT)
        self.assertTrue(torch.allclose(ns1.belief, ns2.belief))

    def test_imagine_step_stochastic_shape(self):
        m = _model()
        belief = torch.randn(5, m.belief_dim)
        action = torch.randn(5, 4).clamp(-1, 1)
        ns, preds = m.imagine_step(belief, action, deterministic=False, sample_temperature=1.0)
        self.assertEqual(tuple(ns.belief.shape), (5, m.belief_dim))
        self.assertEqual(preds.get("imagine_mode"), IMAGINE_STOCHASTIC_PRIOR_ROLLOUT)

    def test_imagine_step_stochastic_changes_latent_with_fixed_prior_std(self):
        m = _model()
        belief = torch.randn(3, m.belief_dim)
        action = torch.randn(3, 4).clamp(-1, 1)
        torch.manual_seed(0)
        ns1, _ = m.imagine_step(belief, action, deterministic=False, sample_temperature=1.0)
        torch.manual_seed(1)
        ns2, _ = m.imagine_step(belief, action, deterministic=False, sample_temperature=1.0)
        self.assertFalse(torch.allclose(ns1.belief, ns2.belief))

    def test_imagine_step_no_grad_stochastic(self):
        m = _model()
        belief = torch.randn(4, m.belief_dim, requires_grad=True)
        action = torch.randn(4, 4, requires_grad=True).clamp(-1, 1)
        ns, preds = m.imagine_step(belief, action, deterministic=False)
        self.assertFalse(ns.belief.requires_grad)
        self.assertFalse(preds["prior_std_mean"].requires_grad)

    def test_imagine_step_fixed_belief_fallback_if_no_prior(self):
        # A stub model without gru/prior_net but with the heads -> fallback.
        m = _model()

        class _NoPrior:
            belief_dim = m.belief_dim
            h_dim = m.h_dim
            z_dim = m.z_dim
            heads = m.heads
            supports_prior_rollout = False
            get_state_from_belief = RSSMLiteWorldModel.get_state_from_belief
            imagine_step = RSSMLiteWorldModel.imagine_step

        stub = _NoPrior()
        belief = torch.randn(4, m.belief_dim)
        action = torch.randn(4, 4).clamp(-1, 1)
        next_state, preds = _NoPrior.imagine_step(stub, belief, action)
        self.assertEqual(preds.get("imagine_mode"), IMAGINE_FIXED_BELIEF_FALLBACK)
        # belief unchanged in fallback
        self.assertTrue(torch.allclose(next_state.belief, belief))


if __name__ == "__main__":
    unittest.main()
