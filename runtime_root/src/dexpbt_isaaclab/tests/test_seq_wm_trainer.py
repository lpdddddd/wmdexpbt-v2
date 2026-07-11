"""
Tests for SeqWMTrainer.

Covered:
* One step() updates the model and returns finite metrics.
* state_dict / load_state_dict round-trip preserves both model and optimizer.
* updates_per_ppo_epoch>1 increments global_step accordingly.
* "Actor model" (a separate nn.Module not registered with the trainer) has
  parameters untouched after step()  (proxy for "PPO actor weights are not
  modified by the sequence-WM trainer").
* beta_kl warmup linearly interpolates between start and end.
"""

from __future__ import annotations

import copy
import unittest

import torch
import torch.nn as nn

from dexpbt_isaaclab.world_model.rssm_lite import RSSMLiteWorldModel
from dexpbt_isaaclab.world_model.seq_wm_trainer import (
    SeqWMConfig,
    SeqWMTrainer,
    _beta_kl_at,
)
from dexpbt_isaaclab.world_model.wm_aux_buffer import register_seq_aux_fields


class _FakeBuffer:
    """Mimics rl_games ExperienceBuffer for the sequence trainer's needs."""

    def __init__(self, T, N, obs_dim, action_dim, device="cpu"):
        self.tensor_dict: dict[str, torch.Tensor] = {}
        self.obs_base_shape = (T, N)
        self.device = device
        self.tensor_dict["obses"] = torch.randn(T, N, obs_dim)
        self.tensor_dict["actions"] = torch.randn(T, N, action_dim)
        self.tensor_dict["rewards"] = torch.randn(T, N, 1)
        self.tensor_dict["dones"] = torch.zeros(T, N).to(torch.uint8)

    def update_data(self, name, index, val):
        stored = self.tensor_dict[name]
        if val.ndim < stored.ndim - 1:
            val = val.unsqueeze(-1)
        stored[index].copy_(val)


def _build_setup(*, T=8, N=12, obs_dim=8, action_dim=3, B=8, seed=0):
    torch.manual_seed(seed)
    buf = _FakeBuffer(T, N, obs_dim, action_dim)
    register_seq_aux_fields(buf)
    # Fill aux d_hat with random noise so they're not all zero.
    buf.tensor_dict["aux_d_hat"].uniform_(-1, 1)
    last_obs = torch.randn(N, obs_dim)
    returns_full = torch.randn(T, N, 1)
    last_d_hat = torch.randn(N)

    model = RSSMLiteWorldModel(
        obs_dim=obs_dim, action_dim=action_dim, obs_embed_dim=16,
        action_embed_dim=8, h_dim=16, z_dim=4, head_hidden=16,
    )
    cfg = SeqWMConfig(
        seq_len=T, seq_batch_size=B, updates_per_ppo_epoch=1,
        lr=1e-3, lambda_total=0.05, lambda_value=0.1,
        beta_kl_warmup_frames=1_000_000,  # short warmup for test
    )
    trainer = SeqWMTrainer(model=model, cfg=cfg, device="cpu",
                           strategy="rssm_belief_value")
    return buf, last_obs, returns_full, last_d_hat, trainer


class SeqWMTrainerStepTest(unittest.TestCase):
    def test_step_updates_model_and_returns_metrics(self):
        buf, last_obs, returns_full, last_d_hat, trainer = _build_setup()
        # Snapshot model params before.
        before = {k: v.detach().clone() for k, v in trainer.model.named_parameters()}
        metrics = trainer.step(
            experience_buffer=buf, frames=0,
            last_obs=last_obs, returns_full=returns_full, last_d_hat=last_d_hat,
        )
        # Metrics finite.
        self.assertGreater(metrics["loss_total"], 0.0)
        self.assertTrue(metrics["grad_norm"] >= 0.0)
        self.assertEqual(metrics["global_step"], 1)
        # Some param changed.
        any_changed = False
        for k, v in trainer.model.named_parameters():
            if not torch.allclose(v.detach(), before[k]):
                any_changed = True
                break
        self.assertTrue(any_changed, "no parameter was updated after step()")

    def test_actor_module_untouched(self):
        """A dummy actor module that's NOT in the trainer must keep its
        parameters byte-identical after step() runs."""
        buf, last_obs, returns_full, last_d_hat, trainer = _build_setup(seed=7)
        actor = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 3))
        actor_before = copy.deepcopy(actor.state_dict())
        # Run trainer; actor is unrelated.
        trainer.step(
            experience_buffer=buf, frames=0,
            last_obs=last_obs, returns_full=returns_full, last_d_hat=last_d_hat,
        )
        for k, v in actor.state_dict().items():
            self.assertTrue(torch.equal(v, actor_before[k]),
                            f"actor param {k} changed after seq WM step")

    def test_updates_per_ppo_epoch(self):
        buf, last_obs, returns_full, last_d_hat, trainer = _build_setup(seed=3)
        trainer.cfg.updates_per_ppo_epoch = 3
        trainer.step(
            experience_buffer=buf, frames=0,
            last_obs=last_obs, returns_full=returns_full, last_d_hat=last_d_hat,
        )
        self.assertEqual(trainer.global_step, 3)


class SeqWMTrainerCheckpointTest(unittest.TestCase):
    def test_state_dict_roundtrip(self):
        buf, last_obs, returns_full, last_d_hat, trainer = _build_setup(seed=11)
        trainer.step(experience_buffer=buf, frames=0,
                     last_obs=last_obs, returns_full=returns_full,
                     last_d_hat=last_d_hat)
        trainer.step(experience_buffer=buf, frames=10_000,
                     last_obs=last_obs, returns_full=returns_full,
                     last_d_hat=last_d_hat)
        sd = trainer.state_dict()
        self.assertEqual(sd["global_step"], 2)
        self.assertEqual(sd["strategy"], "rssm_belief_value")
        self.assertIn("model_state_dict", sd)
        self.assertIn("optimizer_state_dict", sd)

        # Build a fresh trainer + load.
        torch.manual_seed(0)
        fresh_model = RSSMLiteWorldModel(
            obs_dim=8, action_dim=3, obs_embed_dim=16, action_embed_dim=8,
            h_dim=16, z_dim=4, head_hidden=16,
        )
        fresh = SeqWMTrainer(model=fresh_model, cfg=trainer.cfg, device="cpu",
                              strategy="rssm_belief_value")
        fresh.load_state_dict(sd)
        # Same params after load.
        for (k1, v1), (k2, v2) in zip(trainer.model.named_parameters(),
                                       fresh.model.named_parameters()):
            self.assertEqual(k1, k2)
            self.assertTrue(torch.allclose(v1.detach(), v2.detach(), atol=1e-7))
        self.assertEqual(fresh.global_step, 2)


class BetaKLWarmupTest(unittest.TestCase):
    def test_linear_warmup(self):
        cfg = SeqWMConfig(beta_kl_start=0.05, beta_kl_end=0.20,
                          beta_kl_warmup_frames=200_000_000)
        self.assertAlmostEqual(_beta_kl_at(0, cfg), 0.05, places=5)
        self.assertAlmostEqual(_beta_kl_at(100_000_000, cfg), 0.125, places=5)
        self.assertAlmostEqual(_beta_kl_at(200_000_000, cfg), 0.20, places=5)
        # Beyond clamps to end.
        self.assertAlmostEqual(_beta_kl_at(500_000_000, cfg), 0.20, places=5)

    def test_warmup_zero_returns_end(self):
        cfg = SeqWMConfig(beta_kl_start=0.05, beta_kl_end=0.20,
                          beta_kl_warmup_frames=0)
        self.assertAlmostEqual(_beta_kl_at(0, cfg), 0.20, places=5)


if __name__ == "__main__":
    unittest.main()
