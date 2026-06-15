"""Tests for cross-profile PBT checkpoint restore (optimizer param groups)."""

from __future__ import annotations

import unittest

import torch

from dexpbt_isaaclab.scripts.train import (
    _CheckpointSafeRunner,
    _align_optimizer_state_to_agent,
)


class TestCrossProfileCheckpointRestore(unittest.TestCase):
    def test_align_optimizer_state_drops_donor_only_groups(self):
        class _Agent:
            pass

        base = torch.nn.Linear(2, 2)
        agent = _Agent()
        agent.optimizer = torch.optim.Adam(base.parameters(), lr=1e-3)

        donor_base = torch.nn.Linear(2, 2)
        donor_eir = torch.nn.Linear(2, 2)
        donor_opt = torch.optim.Adam(
            [
                {"params": donor_base.parameters(), "lr": 1e-3},
                {"params": donor_eir.parameters(), "lr": 1e-3, "name": "eir_scorer"},
            ]
        )
        ckpt_opt = donor_opt.state_dict()
        aligned = _align_optimizer_state_to_agent(agent, ckpt_opt)
        self.assertEqual(len(aligned["param_groups"]), 1)
        agent.optimizer.load_state_dict(aligned)

    def test_checkpoint_safe_runner_class_exists_for_ppo_only_path(self):
        from rl_games.torch_runner import Runner

        self.assertTrue(issubclass(_CheckpointSafeRunner, Runner))


if __name__ == "__main__":
    unittest.main()
