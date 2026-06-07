"""
Shared prediction heads for sequence world models.

These heads are reused by both RSSMLiteWorldModel and TransformerLiteWorldModel.
Each head is a small MLP. Inputs are the belief b_t (shape (..., belief_dim))
and, for transition heads, the action a_t (shape (..., action_dim)).

All heads operate on flattened (T*B, ...) tensors so callers should reshape
batch tensors before invocation; the heads do not assume a (T, B) layout.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def _mlp(in_dim: int, hidden: int, out_dim: int, n_hidden: int = 2) -> nn.Sequential:
    layers: list[nn.Module] = []
    prev = in_dim
    for _ in range(n_hidden):
        layers.append(nn.Linear(prev, hidden))
        layers.append(nn.ELU())
        prev = hidden
    layers.append(nn.Linear(prev, out_dim))
    return nn.Sequential(*layers)


class PredictionHeads(nn.Module):
    """Bundle of prediction heads shared by RSSM-lite and Transformer-lite.

    Args:
        belief_dim: dimensionality of belief b_t (e.g. 320 for RSSM, 256 for Trans).
        action_dim: env action dim (23 for KukaAllegro).
        obs_dim: env obs dim (used only for next_obs_delta head output dim).
        hidden_dim: MLP hidden width.
        n_hidden: MLP depth (number of hidden layers; out layer is on top).
        with_value_head: include V_seq(b_t) for *_belief_value strategies.
        with_success_head: include success BCE head.
        with_task_error_head: include task_error_delta head.
    """

    def __init__(
        self,
        *,
        belief_dim: int,
        action_dim: int,
        obs_dim: int,
        hidden_dim: int = 256,
        n_hidden: int = 2,
        with_value_head: bool = True,
        with_success_head: bool = True,
        with_task_error_head: bool = True,
    ) -> None:
        super().__init__()
        self.belief_dim = int(belief_dim)
        self.action_dim = int(action_dim)
        self.obs_dim = int(obs_dim)
        self.hidden_dim = int(hidden_dim)

        ba_in = self.belief_dim + self.action_dim
        # Transition heads: (b_t, a_t) -> target_t
        self.next_obs_delta_head = _mlp(ba_in, hidden_dim, obs_dim, n_hidden)
        self.reward_head = _mlp(ba_in, hidden_dim, 1, n_hidden)
        self.done_head = _mlp(ba_in, hidden_dim, 1, n_hidden)

        self.task_error_delta_head: nn.Module | None = (
            _mlp(ba_in, hidden_dim, 1, n_hidden) if with_task_error_head else None
        )
        self.success_head: nn.Module | None = (
            _mlp(ba_in, hidden_dim, 1, n_hidden) if with_success_head else None
        )

        # Value head only depends on belief (not action), and is the only
        # head whose output is used as a *target prediction of returns_t*.
        self.value_head: nn.Module | None = (
            _mlp(self.belief_dim, hidden_dim, 1, n_hidden) if with_value_head else None
        )

    # -- Forward helpers (caller flattens (T, B, ...) -> (T*B, ...)) --

    def predict_transition(
        self,
        belief_flat: torch.Tensor,
        action_flat: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Run the (b_t, a_t)-conditioned heads.

        Returns a dict with keys: 'next_obs_delta', 'reward', 'done_logits',
        and optionally 'task_error_delta', 'success_logits'. Each value has
        shape matching the head output (e.g. reward => (N, 1)).
        """
        ba = torch.cat([belief_flat, action_flat], dim=-1)
        out: dict[str, torch.Tensor] = {
            "next_obs_delta": self.next_obs_delta_head(ba),
            "reward": self.reward_head(ba),
            "done_logits": self.done_head(ba),
        }
        if self.task_error_delta_head is not None:
            out["task_error_delta"] = self.task_error_delta_head(ba)
        if self.success_head is not None:
            out["success_logits"] = self.success_head(ba)
        return out

    def predict_value(self, belief_flat: torch.Tensor) -> torch.Tensor | None:
        """Run V_seq(b_t). Returns None if value head is disabled.

        The caller decides whether to detach belief_flat before calling
        (controlled by --seq_wm_value_detach_belief).
        """
        if self.value_head is None:
            return None
        return self.value_head(belief_flat)

    @property
    def has_value_head(self) -> bool:
        return self.value_head is not None


__all__ = ["PredictionHeads"]
