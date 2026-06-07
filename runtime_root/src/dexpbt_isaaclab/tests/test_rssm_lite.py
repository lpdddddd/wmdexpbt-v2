"""
Tests for RSSMLiteWorldModel.

Covered:
* Belief shape (T, B, belief_dim).
* All loss terms finite; backward runs without NaN.
* Prior/posterior std stay above std_floor (no degeneracy at init).
* KL is finite on a randomly sampled batch.
* Hidden reset on done: h_t differs across two batches that only differ in
  done_{t-1}.
* Loss-weights and beta_kl plumbing.
"""

from __future__ import annotations

import unittest

import torch

from dexpbt_isaaclab.world_model.rssm_lite import RSSMLiteWorldModel
from dexpbt_isaaclab.world_model.seq_wm_base import SequenceBatch


def _make_batch(T: int, B: int, obs_dim: int, action_dim: int,
                *, dones: torch.Tensor | None = None,
                seed: int = 0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    obs = torch.randn(T, B, obs_dim, generator=g)
    next_obs = torch.randn(T, B, obs_dim, generator=g)
    action = torch.randn(T, B, action_dim, generator=g)
    reward = torch.randn(T, B, generator=g)
    returns = torch.randn(T, B, generator=g)
    done = torch.zeros(T, B) if dones is None else dones.float()
    d_hat = torch.randn(T, B, generator=g)
    next_d_hat = torch.randn(T, B, generator=g)
    success = torch.zeros(T, B)
    next_success = torch.zeros(T, B)
    valid = torch.ones(T, B)
    seg = torch.zeros(T, B, dtype=torch.long)
    # Recompute valid from done to be consistent.
    if (done.sum() > 0):
        prev_cum = torch.zeros_like(done)
        prev_cum[1:] = torch.cumsum(done[:-1], dim=0)
        valid = (prev_cum < 0.5).float()
        seg = prev_cum.long()

    return SequenceBatch(
        obs=obs, action=action, reward=reward, done=done,
        next_obs=next_obs, returns=returns,
        d_hat=d_hat, next_d_hat=next_d_hat,
        success_count=success, next_success_count=next_success,
        valid_mask=valid, segment_id=seg,
    )


class RSSMShapeAndForwardTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.T, self.B, self.obs_dim, self.action_dim = 8, 6, 12, 4
        self.model = RSSMLiteWorldModel(
            obs_dim=self.obs_dim, action_dim=self.action_dim,
            obs_embed_dim=32, action_embed_dim=16,
            h_dim=24, z_dim=8, head_hidden=32, free_nats=1.0,
        )

    def test_belief_shape(self):
        batch = _make_batch(self.T, self.B, self.obs_dim, self.action_dim)
        belief = self.model.encode_sequence(batch)
        self.assertEqual(tuple(belief.shape),
                         (self.T, self.B, self.model.belief_dim))
        self.assertEqual(self.model.belief_dim, 24 + 8)

    def test_compute_aux_loss_finite_and_backward(self):
        batch = _make_batch(self.T, self.B, self.obs_dim, self.action_dim, seed=1)
        out = self.model.compute_aux_loss(batch, beta_kl=0.1)
        for name, v in out.items():
            self.assertTrue(torch.isfinite(v).all().item(),
                            f"loss term {name} is not finite")
        out["loss_total"].backward()
        # Some param has a non-zero grad.
        any_grad = False
        for p in self.model.parameters():
            if p.grad is not None and p.grad.abs().sum().item() > 0:
                any_grad = True
                break
        self.assertTrue(any_grad)

    def test_prior_posterior_std_above_floor(self):
        batch = _make_batch(self.T, self.B, self.obs_dim, self.action_dim, seed=2)
        _ = self.model.encode_sequence(batch)
        prior_std = self.model._last_prior_std
        post_std = self.model._last_post_std
        self.assertGreaterEqual(float(prior_std.min().item()), self.model.std_floor - 1e-6)
        self.assertGreaterEqual(float(post_std.min().item()), self.model.std_floor - 1e-6)
        # Not collapsed to floor everywhere.
        self.assertGreater(float(prior_std.mean().item()), self.model.std_floor)
        self.assertGreater(float(post_std.mean().item()), self.model.std_floor)


class RSSMHiddenResetOnDoneTest(unittest.TestCase):
    """A done at step t-1 must zero h, z so the t-th GRU step starts fresh."""

    def test_belief_differs_when_done_t_minus_1_changes(self):
        torch.manual_seed(123)
        T, B, D, A = 4, 1, 6, 3
        model = RSSMLiteWorldModel(
            obs_dim=D, action_dim=A, obs_embed_dim=16,
            action_embed_dim=8, h_dim=12, z_dim=4, head_hidden=16,
        )
        batch_no_done = _make_batch(T, B, D, A, seed=99)
        # Inject a done at t=1 in one batch only; everything else identical.
        batch_with_done = _make_batch(T, B, D, A, seed=99)
        batch_with_done.done[1, 0] = 1.0
        # Recompute valid_mask consistent with the done.
        batch_with_done.valid_mask[2:, 0] = 0.0
        batch_with_done.segment_id[2:, 0] = 1

        # We need deterministic reparam to make beliefs comparable. Do that
        # by seeding torch's global RNG before each forward and comparing
        # h-only (h is deterministic given (h_prev, z_prev, a_prev)).
        torch.manual_seed(7)
        b1 = model.encode_sequence(batch_no_done)
        h1 = b1[..., :model.h_dim]

        torch.manual_seed(7)
        b2 = model.encode_sequence(batch_with_done)
        h2 = b2[..., :model.h_dim]

        # h[0] and h[1] must match (same inputs up to t=1, no reset yet).
        self.assertTrue(torch.allclose(h1[0], h2[0], atol=1e-5))
        self.assertTrue(torch.allclose(h1[1], h2[1], atol=1e-5))
        # h[2] must DIFFER (h reset to zero in batch_with_done at t=2 entry).
        self.assertFalse(torch.allclose(h1[2], h2[2], atol=1e-4),
                         "h[2] should differ when done[1]=1 forces reset")


class RSSMLossWeightsTest(unittest.TestCase):
    """Setting lambda_value=0 should null out value contribution."""

    def test_value_weight_zero_disables_value_loss(self):
        torch.manual_seed(0)
        model = RSSMLiteWorldModel(
            obs_dim=8, action_dim=3, obs_embed_dim=16, action_embed_dim=8,
            h_dim=12, z_dim=4, head_hidden=16,
        )
        batch = _make_batch(T=4, B=4, obs_dim=8, action_dim=3, seed=42)
        weights_w_value = {"state": 1.0, "reward": 1.0, "done": 0.2,
                           "success": 1.0, "task_error": 1.0, "value": 1.0}
        weights_no_value = dict(weights_w_value); weights_no_value["value"] = 0.0
        torch.manual_seed(0); a = model.compute_aux_loss(batch, loss_weights=weights_w_value, beta_kl=0.0)
        torch.manual_seed(0); b = model.compute_aux_loss(batch, loss_weights=weights_no_value, beta_kl=0.0)
        # The two totals should differ unless L_value happens to be exactly 0
        # (it won't on random init: returns are non-trivial).
        self.assertFalse(
            torch.allclose(a["loss_total"], b["loss_total"], atol=1e-6),
            "weight_value=0 should change loss_total"
        )


if __name__ == "__main__":
    unittest.main()
