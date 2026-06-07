import unittest

import torch
import torch.nn as nn

from dexpbt_isaaclab.world_model.wm_config import WMConfig
from dexpbt_isaaclab.world_model.wm_trainer import WMTrainer, get_strategy


class _FakeBuffer:
    def __init__(self, obs_dim: int, action_dim: int, batch: int):
        self.size = batch
        self._obs = torch.randn(batch, obs_dim)
        self._action = torch.randn(batch, action_dim)
        self._zeros3 = torch.zeros(batch, 3)
        self._zeros1 = torch.zeros(batch, 1)

    def sample(self, batch_size: int):
        return {
            "obs_t": self._obs[:batch_size],
            "action_t": self._action[:batch_size],
            "object_pos_t": self._zeros3[:batch_size],
            "object_linvel_t": self._zeros3[:batch_size],
            "object_angvel_t": self._zeros3[:batch_size],
            "d_hat_t": self._zeros1[:batch_size],
        }


class _SharedEncoderModel(nn.Module):
    def __init__(self, obs_dim: int, hidden_dim: int, action_dim: int):
        super().__init__()
        self.shared = nn.Linear(obs_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, 1)
        self.mu = nn.Linear(hidden_dim, action_dim)

    def encode(self, obs: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.shared(obs))


class ValueImagGradientPathTest(unittest.TestCase):
    def test_value_imag_gradient_path(self):
        torch.manual_seed(0)
        obs_dim, action_dim, hidden_dim, batch_size = 16, 4, 8, 32

        shared_model = _SharedEncoderModel(obs_dim, hidden_dim, action_dim)
        wm_module = nn.Identity()
        wm_optimizer = torch.optim.Adam([torch.zeros(1, requires_grad=True)], lr=1e-3)
        buffer = _FakeBuffer(obs_dim, action_dim, batch_size)
        trainer = WMTrainer(
            wm_module=wm_module,
            wm_optimizer=wm_optimizer,
            wm_buffer=buffer,
            config=WMConfig(obs_dim=obs_dim, action_dim=action_dim, batch_size=batch_size),
            strategy=get_strategy("value_imag_h1"),
            ppo_model=None,
            ppo_optimizer=None,
            tensorboard_writer=None,
            device="cpu",
        )
        trainer._value_imag_enabled = True
        trainer.aux_value_head = nn.Linear(hidden_dim, 1)
        trainer._normalize_obs = lambda obs: obs.to(torch.float32)

        def _get_encoder_features(obs, detach=False):
            h = shared_model.encode(obs)
            return h.detach() if detach else h

        trainer._get_encoder_features = _get_encoder_features
        trainer._rollout_imagined_return = (
            lambda obs_raw, action_t, object_pos_t, object_linvel_t, object_angvel_t, d_hat_t, horizon, gamma:
            (
                torch.ones(obs_raw.shape[0], dtype=torch.float32),
                obs_raw.to(torch.float32),
                torch.zeros(obs_raw.shape[0], dtype=torch.float32),
                torch.ones(obs_raw.shape[0], dtype=torch.float32),
            )
        )

        for p in list(shared_model.parameters()) + list(trainer.aux_value_head.parameters()):
            if p.grad is not None:
                p.grad.zero_()

        loss_dict = trainer.compute_value_imag_loss_dict()
        self.assertIsNotNone(loss_dict)
        loss = loss_dict["value_imag"]
        loss.backward()

        aux_grad = sum(float(p.grad.norm().item()) for p in trainer.aux_value_head.parameters() if p.grad is not None)
        shared_grad = sum(float(p.grad.norm().item()) for p in shared_model.shared.parameters() if p.grad is not None)
        actor_head_grad = sum(float(p.grad.norm().item()) for p in shared_model.mu.parameters() if p.grad is not None)

        self.assertGreater(aux_grad, 0.0, "aux_value_head should receive non-zero gradients")
        self.assertGreater(shared_grad, 0.0, "shared encoder should receive non-zero gradients")
        self.assertEqual(actor_head_grad, 0.0, "actor-specific head must not be directly updated by value_imag loss")


if __name__ == "__main__":
    unittest.main()
