"""
Sequence sampler over the PPO ExperienceBuffer.

Pulls (T, B) windows from per-step (H, N) tensors:

    H = horizon_length (PPO rollout length, default 16 in this project)
    N = num_actors      (per-policy num_envs, e.g. 8192)
    T = seq_len         (<= H; default 16, may be 8 to halve cost)
    B = seq_batch_size  (default 1024)

The sampler:

* Selects B (env, start_t) pairs uniformly without replacement on env when
  possible.
* Slices obs / action / reward / done / returns into (T, B, ...) windows.
* Builds next_obs[t, b]:
    - for t such that start_t + t + 1 < H: from obs_full[start_t+t+1, env]
    - for t such that start_t + t + 1 == H: from last_obs[env]
* Computes valid_mask[t, b] (cross-episode mask):
    - valid_mask[0] = 1
    - valid_mask[t] = 1 iff no done occurred in done_window[0..t-1]
  This is the contract used by:
    - state_delta loss (mask out crossings)
    - task_error_delta loss (same)
    - RSSM hidden reset (segment boundaries)
* Emits segment_id[t, b] = cumulative_done[<t] so backbones can reset
  hidden state at segment boundaries.
* Forwards aux scalar fields (d_hat, failure, success_count, lifted) through
  the same gather, both at time t and at time t+1 where applicable.
"""

from __future__ import annotations

from typing import Optional

import torch

from .seq_wm_base import SequenceBatch


def _maybe_squeeze_last(t: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    """Squeeze trailing dim if it's 1; otherwise pass through."""
    if t is None:
        return None
    if t.ndim >= 1 and t.shape[-1] == 1:
        return t.squeeze(-1)
    return t


def _gather_seq(
    full: Optional[torch.Tensor],
    t_idx: torch.Tensor,        # (T, B) long
    e_idx: torch.Tensor,        # (T, B) long
) -> Optional[torch.Tensor]:
    """Gather (T, B[, D]) from (H, N[, D]) via fancy indexing."""
    if full is None:
        return None
    # full[t_idx, e_idx] broadcasts naturally.
    return full[t_idx, e_idx]


def _gather_next(
    full: Optional[torch.Tensor],     # (H, N[, D])
    last_step: Optional[torch.Tensor], # (N[, D]) — value of "step H" (post-rollout)
    t_idx: torch.Tensor,              # (T, B)
    e_idx: torch.Tensor,              # (T, B)
    H: int,
) -> Optional[torch.Tensor]:
    """Gather "next" version at index (t_idx + 1), falling back to last_step
    when t_idx + 1 == H.

    full and last_step may both be None — returns None.
    full alone is None but last_step is not None — returns None (we don't
    invent a value mid-window without the underlying series).
    """
    if full is None:
        return None
    nxt = t_idx + 1
    in_range = nxt < H
    # safe gather: clamp to H-1 then overwrite with last_step where needed
    safe_idx = nxt.clamp(max=H - 1)
    out = full[safe_idx, e_idx]
    if last_step is not None:
        # broadcast last_step[e_idx] onto positions where nxt == H
        last_at_e = last_step[e_idx]
        # in_range has shape (T, B); make broadcastable to (..., D)
        if out.ndim > in_range.ndim:
            mask = in_range.unsqueeze(-1).expand_as(out)
        else:
            mask = in_range
        out = torch.where(mask, out, last_at_e)
    return out


class SequenceSampler:
    """Sample (T, B) sequence windows from (H, N) PPO rollout tensors."""

    def __init__(
        self,
        *,
        seq_len: int,
        seq_batch_size: int,
        device: torch.device | str = "cuda",
        generator: Optional[torch.Generator] = None,
    ) -> None:
        if seq_len <= 0:
            raise ValueError(f"seq_len must be > 0, got {seq_len}")
        if seq_batch_size <= 0:
            raise ValueError(f"seq_batch_size must be > 0, got {seq_batch_size}")
        self.seq_len = int(seq_len)
        self.seq_batch_size = int(seq_batch_size)
        self.device = torch.device(device)
        self._gen = generator  # may be None -> default RNG

    # ----- index sampling -----

    def _sample_indices(
        self,
        H: int,
        N: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (t_idx, e_idx) of shape (T, B) on self.device."""
        T = self.seq_len
        B = self.seq_batch_size
        if T > H:
            raise ValueError(
                f"seq_len ({T}) cannot exceed horizon_length ({H}); pass a "
                f"smaller --seq_wm_seq_len or extend the PPO rollout."
            )
        # env indices: uniform with replacement (B may be > N).
        env_idx = torch.randint(
            low=0,
            high=N,
            size=(B,),
            device=self.device,
            generator=self._gen,
        )
        # start_t: uniform in [0, H-T].
        if H == T:
            start_t = torch.zeros(B, dtype=torch.long, device=self.device)
        else:
            start_t = torch.randint(
                low=0,
                high=H - T + 1,
                size=(B,),
                device=self.device,
                generator=self._gen,
            )
        # Build (T, B) index grids.
        t_off = torch.arange(T, device=self.device).unsqueeze(1)         # (T, 1)
        t_idx = start_t.unsqueeze(0) + t_off                              # (T, B)
        e_idx = env_idx.unsqueeze(0).expand(T, B).contiguous()           # (T, B)
        return t_idx, e_idx

    # ----- valid mask + segment id -----

    @staticmethod
    def _build_mask_and_segment(
        done_window: torch.Tensor,  # (T, B) float in {0,1}
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (valid_mask, segment_id), both (T, B).

        valid_mask[t, b] = 1 iff cumulative done in [0..t-1] is 0.
        segment_id[t, b] = sum of dones in [0..t-1].
        """
        T = done_window.shape[0]
        if T == 1:
            valid = torch.ones_like(done_window)
            seg = torch.zeros_like(done_window, dtype=torch.long)
            return valid, seg
        # Cumulative dones strictly before t:
        #   prev_done_cum[0] = 0
        #   prev_done_cum[t] = sum(done[0..t-1])
        prev_done_cum = torch.zeros_like(done_window)
        prev_done_cum[1:] = torch.cumsum(done_window[:-1], dim=0)
        valid = (prev_done_cum < 0.5).to(done_window.dtype)
        seg = prev_done_cum.long()
        return valid, seg

    # ----- main entry -----

    def sample(
        self,
        *,
        obs_full: torch.Tensor,       # (H, N, obs_dim)
        action_full: torch.Tensor,    # (H, N, action_dim)
        reward_full: torch.Tensor,    # (H, N) or (H, N, 1)
        done_full: torch.Tensor,      # (H, N) or (H, N, 1) float/uint8
        returns_full: torch.Tensor,   # (H, N) or (H, N, 1)
        last_obs: torch.Tensor,       # (N, obs_dim)  obs at step H
        d_hat_full: Optional[torch.Tensor] = None,        # (H, N) or (H, N, 1)
        last_d_hat: Optional[torch.Tensor] = None,        # (N,) or (N, 1)
        failure_full: Optional[torch.Tensor] = None,
        success_count_full: Optional[torch.Tensor] = None,
        last_success_count: Optional[torch.Tensor] = None,
        lifted_full: Optional[torch.Tensor] = None,
    ) -> SequenceBatch:
        # Shape sanity & normalisation.
        if obs_full.ndim != 3:
            raise ValueError(f"obs_full must be (H, N, D); got {obs_full.shape}")
        H, N, D_obs = obs_full.shape
        if action_full.shape[:2] != (H, N):
            raise ValueError(
                f"action_full leading dims {action_full.shape[:2]} != obs leading {(H, N)}"
            )
        # Normalise (H, N, 1) -> (H, N) for scalar fields.
        reward_full = _maybe_squeeze_last(reward_full)
        done_full   = _maybe_squeeze_last(done_full).to(obs_full.dtype)
        returns_full = _maybe_squeeze_last(returns_full)
        d_hat_full = _maybe_squeeze_last(d_hat_full)
        failure_full = _maybe_squeeze_last(failure_full)
        success_count_full = _maybe_squeeze_last(success_count_full)
        lifted_full = _maybe_squeeze_last(lifted_full)
        last_d_hat = _maybe_squeeze_last(last_d_hat) if last_d_hat is not None else None
        last_success_count = (
            _maybe_squeeze_last(last_success_count)
            if last_success_count is not None else None
        )

        if last_obs.shape != (N, D_obs):
            raise ValueError(
                f"last_obs must have shape (N, D_obs)=({N},{D_obs}); got {last_obs.shape}"
            )

        # Build (T, B) index grids.
        t_idx, e_idx = self._sample_indices(H, N)
        T, B = t_idx.shape
        assert T == self.seq_len and B == self.seq_batch_size

        # Gather windowed tensors.
        obs_w     = _gather_seq(obs_full,     t_idx, e_idx)   # (T, B, D_obs)
        action_w  = _gather_seq(action_full,  t_idx, e_idx)   # (T, B, D_a)
        reward_w  = _gather_seq(reward_full,  t_idx, e_idx)   # (T, B)
        done_w    = _gather_seq(done_full,    t_idx, e_idx)   # (T, B)
        returns_w = _gather_seq(returns_full, t_idx, e_idx)   # (T, B)
        d_hat_w   = _gather_seq(d_hat_full,   t_idx, e_idx)
        failure_w = _gather_seq(failure_full, t_idx, e_idx)
        success_w = _gather_seq(success_count_full, t_idx, e_idx)
        lifted_w  = _gather_seq(lifted_full,  t_idx, e_idx)

        next_obs_w = _gather_next(obs_full, last_obs, t_idx, e_idx, H)
        next_d_hat_w = _gather_next(d_hat_full, last_d_hat, t_idx, e_idx, H)
        next_success_w = _gather_next(success_count_full, last_success_count, t_idx, e_idx, H)

        valid_mask, segment_id = self._build_mask_and_segment(done_w)

        return SequenceBatch(
            obs=obs_w,
            action=action_w,
            reward=reward_w,
            done=done_w,
            next_obs=next_obs_w,
            returns=returns_w,
            d_hat=d_hat_w,
            next_d_hat=next_d_hat_w,
            failure=failure_w,
            success_count=success_w,
            next_success_count=next_success_w,
            lifted=lifted_w,
            valid_mask=valid_mask,
            segment_id=segment_id,
            meta={
                "horizon_length": int(H),
                "num_actors": int(N),
                "obs_dim": int(D_obs),
                "seq_len": int(T),
                "seq_batch_size": int(B),
            },
        )


__all__ = ["SequenceSampler"]
