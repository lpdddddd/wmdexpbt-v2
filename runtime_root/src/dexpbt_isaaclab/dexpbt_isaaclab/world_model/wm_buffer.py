"""
Ring-buffer for world model transitions.
Stores: (obs_t, action_t, reward_t, obs_tp1, done_t, failure_t,
         object_pos_t, object_pos_tp1,
         object_linvel_t, object_linvel_tp1,
         object_angvel_t, object_angvel_tp1,
         d_hat_t, d_hat_tp1)
"""

import torch
from collections import OrderedDict
from dataclasses import dataclass


@dataclass
class WMTransition:
    """Single transition fields."""
    obs_t: torch.Tensor           # (N, obs_dim)
    action_t: torch.Tensor       # (N, action_dim)
    reward_t: torch.Tensor       # (N, 1)
    obs_tp1: torch.Tensor         # (N, obs_dim)
    done_t: torch.Tensor          # (N, 1)
    failure_t: torch.Tensor      # (N, 1)
    object_pos_t: torch.Tensor   # (N, 3)
    object_pos_tp1: torch.Tensor # (N, 3)
    object_linvel_t: torch.Tensor # (N, 3)
    object_linvel_tp1: torch.Tensor
    object_angvel_t: torch.Tensor # (N, 3)
    object_angvel_tp1: torch.Tensor
    d_hat_t: torch.Tensor        # (N, 1)
    d_hat_tp1: torch.Tensor      # (N, 1)


class WMBuffer:
    """
    Ring buffer with add() and sample().
    Uses torch tensors internally for GPU speed.

    add(): concatenates new transitions to existing buffer (prepend + truncate).
    sample(): returns flat tensors of all stored transitions.
    """

    def __init__(self, capacity: int, obs_dim: int, action_dim: int, device: str = "cuda:0"):
        self.capacity = capacity
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.device = device

        self._buf = OrderedDict([
            ("obs_t",           torch.zeros(capacity, obs_dim, device=device)),
            ("action_t",        torch.zeros(capacity, action_dim, device=device)),
            ("reward_t",        torch.zeros(capacity, 1, device=device)),
            ("obs_tp1",         torch.zeros(capacity, obs_dim, device=device)),
            ("done_t",          torch.zeros(capacity, 1, device=device)),
            ("failure_t",       torch.zeros(capacity, 1, device=device)),
            ("object_pos_t",    torch.zeros(capacity, 3, device=device)),
            ("object_pos_tp1",  torch.zeros(capacity, 3, device=device)),
            ("object_linvel_t", torch.zeros(capacity, 3, device=device)),
            ("object_linvel_tp1", torch.zeros(capacity, 3, device=device)),
            ("object_angvel_t", torch.zeros(capacity, 3, device=device)),
            ("object_angvel_tp1", torch.zeros(capacity, 3, device=device)),
            ("d_hat_t",         torch.zeros(capacity, 1, device=device)),
            ("d_hat_tp1",       torch.zeros(capacity, 1, device=device)),
        ])
        self._size = 0
        self._head = 0   # next write position

    @property
    def size(self) -> int:
        return self._size

    def add(self, *,
            obs_t: torch.Tensor,
            action_t: torch.Tensor,
            reward_t: torch.Tensor,
            obs_tp1: torch.Tensor,
            done_t: torch.Tensor,
            failure_t: torch.Tensor,
            object_pos_t: torch.Tensor,
            object_pos_tp1: torch.Tensor,
            object_linvel_t: torch.Tensor,
            object_linvel_tp1: torch.Tensor,
            object_angvel_t: torch.Tensor,
            object_angvel_tp1: torch.Tensor,
            d_hat_t: torch.Tensor,
            d_hat_tp1: torch.Tensor,
            add_mask: torch.Tensor | None = None,
    ) -> None:
        """
        Add transitions to buffer.
        add_mask: (N,) bool tensor. Only rows where add_mask=True are written.
                  If None, all rows are written.
        """
        N = obs_t.shape[0]
        if add_mask is None:
            add_mask = torch.ones(N, dtype=torch.bool, device=self.device)

        n_to_add = add_mask.sum().item()
        if n_to_add == 0:
            return

        # Collect tensors in the EXACT same order as self._buf keys
        # (Python dict preserves insertion order; order matters for the loop below)
        tensor_list = [
            ("obs_t",          obs_t),
            ("action_t",       action_t),
            ("reward_t",       reward_t),
            ("obs_tp1",        obs_tp1),
            ("done_t",         done_t),
            ("failure_t",      failure_t),
            ("object_pos_t",   object_pos_t),
            ("object_pos_tp1", object_pos_tp1),
            ("object_linvel_t",object_linvel_t),
            ("object_linvel_tp1", object_linvel_tp1),
            ("object_angvel_t",object_angvel_t),
            ("object_angvel_tp1", object_angvel_tp1),
            ("d_hat_t",        d_hat_t),
            ("d_hat_tp1",      d_hat_tp1),
        ]

        indices = torch.where(add_mask)[0].long()
        positions = (self._head + torch.arange(len(indices), device=self.device, dtype=torch.long)) % self.capacity

        for key, src in tensor_list:
            self._buf[key][positions] = src[indices]

        # Update size/head AFTER writing all fields (once per add() call)
        self._size = min(self.capacity, self._size + n_to_add)
        self._head = (self._head + n_to_add) % self.capacity

    def sample(self, batch_size: int | None = None) -> dict[str, torch.Tensor]:
        """Return all stored transitions as a flat dict."""
        if self._size == 0:
            raise RuntimeError("WMBuffer is empty")
        actual = self._size if batch_size is None else min(batch_size, self._size)
        if batch_size is not None and batch_size <= self._size:
            indices = torch.randperm(self._size, device=self.device)[:batch_size]
        else:
            indices = torch.arange(self._size, device=self.device)
        return {key: buf[indices] for key, buf in self._buf.items()}

    def clear(self) -> None:
        self._size = 0
        self._head = 0