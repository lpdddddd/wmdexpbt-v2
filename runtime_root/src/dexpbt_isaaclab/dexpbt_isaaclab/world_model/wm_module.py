"""
3-member ensemble World Model MLP.
Predicts delta object state, reward, done, failure logits given (obs, action).
"""

import torch
import torch.nn as nn
from typing import Optional


class _WorldModelMember(nn.Module):
    """
    Single member of the ensemble. Predicts deltas (not absolute states).

    Input:  obs (B, obs_dim) + action (B, action_dim)  → concat (B, obs_dim+action_dim)
    Output:
        object_pos_delta    (B, 3)
        d_hat_delta         (B, 1)
        linvel_delta        (B, 3)
        angvel_delta        (B, 3)
        reward              (B, 1)
        done_logits         (B, 1)
        failure_logits      (B, 1)
    """

    def __init__(self, obs_dim: int, action_dim: int, hidden_sizes: list[int]):
        super().__init__()
        in_dim = obs_dim + action_dim
        layers = []
        prev_dim = in_dim
        for h in hidden_sizes:
            layers.extend([nn.Linear(prev_dim, h), nn.Tanh()])
            prev_dim = h
        self.encoder = nn.Sequential(*layers)
        # 7 prediction heads
        self.head_pos = nn.Linear(prev_dim, 3)
        self.head_dhat = nn.Linear(prev_dim, 1)
        self.head_linvel = nn.Linear(prev_dim, 3)
        self.head_angvel = nn.Linear(prev_dim, 3)
        self.head_reward = nn.Linear(prev_dim, 1)
        self.head_done = nn.Linear(prev_dim, 1)
        self.head_failure = nn.Linear(prev_dim, 1)

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> dict[str, torch.Tensor]:
        x = torch.cat([obs, action], dim=-1)
        h = self.encoder(x)
        return {
            "object_pos_delta": self.head_pos(h),
            "d_hat_delta":     self.head_dhat(h),
            "linvel_delta":    self.head_linvel(h),
            "angvel_delta":    self.head_angvel(h),
            "reward":          self.head_reward(h),
            "done_logits":     self.head_done(h),
            "failure_logits":  self.head_failure(h),
        }


class WorldModel(nn.Module):
    """
    3-member ensemble WorldModel.

    forward_mean(): forward all members, return mean prediction (for training/targets)
    forward_all():  forward all members, return list of per-member predictions (for uncertainty)
    """

    def __init__(self, obs_dim: int, action_dim: int,
                 hidden_sizes: list[int] | None = None,
                 ensemble_size: int = 3):
        super().__init__()
        if hidden_sizes is None:
            hidden_sizes = [512, 256, 128]
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.ensemble_size = ensemble_size

        self.members = nn.ModuleList([
            _WorldModelMember(obs_dim, action_dim, hidden_sizes)
            for _ in range(ensemble_size)
        ])

    def forward_mean(self, obs: torch.Tensor, action: torch.Tensor) -> dict[str, torch.Tensor]:
        """Forward all members, return mean (detached for target, non-detached for training)."""
        preds = [m(obs, action) for m in self.members]
        # mean across ensemble members
        keys = list(preds[0].keys())
        result = {}
        for k in keys:
            stacked = torch.stack([p[k] for p in preds], dim=0)   # (E, B, ...)
            result[k] = stacked.mean(dim=0)
        return result

    def forward_all(self, obs: torch.Tensor, action: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Forward all members, return stacked predictions.
        Returns dict with each value shape (E, B, ...).
        Used for uncertainty computation.
        """
        preds = [m(obs, action) for m in self.members]
        keys = list(preds[0].keys())
        result = {}
        for k in keys:
            result[k] = torch.stack([p[k] for p in preds], dim=0)   # (E, B, ...)
        return result

    def reset_parameters(self):
        for m in self.members:
            for p in m.parameters():
                if p.dim() >= 2:
                    nn.init.orthogonal_(p, gain=1.0)
                elif p.dim() == 1:
                    nn.init.zeros_(p)


def compute_uncertainty(wm_module: WorldModel, obs: torch.Tensor,
                       action: torch.Tensor) -> tuple[torch.Tensor, dict]:
    """
    Compute per-sample ensemble variance.
    Returns (primary_unc, variance_dict) where both are on device.
    primary_unc shape: (B,)
    """
    all_preds = wm_module.forward_all(obs, action)  # (E, B, *)
    variance = {}
    for key in ["object_pos_delta", "d_hat_delta", "linvel_delta", "angvel_delta"]:
        var = all_preds[key].var(dim=0)   # (B, dim)
        variance[key] = var
    # Primary uncertainty: mean variance across position/dhat/velocity dims
    primary_unc = torch.stack([
        variance["object_pos_delta"].mean(-1),
        variance["d_hat_delta"].mean(-1),
        variance["linvel_delta"].mean(-1),
        variance["angvel_delta"].mean(-1),
    ]).mean(dim=0)  # (B,)
    return primary_unc, variance