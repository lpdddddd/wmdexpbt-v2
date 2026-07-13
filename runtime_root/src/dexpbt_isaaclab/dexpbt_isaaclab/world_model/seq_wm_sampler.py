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


def build_rollout_aux_scalars(
    *,
    T: int,
    N: int,
    d_hat: Optional[torch.Tensor],
    next_d_hat: Optional[torch.Tensor],
    success_count: Optional[torch.Tensor],
    next_success_count: Optional[torch.Tensor],
    success_tolerance: Optional[torch.Tensor],
    lifted: Optional[torch.Tensor],
    failure: Optional[torch.Tensor],
    picked: Optional[torch.Tensor] = None,
    dropped: Optional[torch.Tensor] = None,
    closest_d_hat: Optional[torch.Tensor] = None,
    device: torch.device | str,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Build ``(T, N, aux_dim)`` task aux scalars for a full rollout batch.

    Mirrors the field order assembled in ``SequenceSampler.sample`` so RSSM
    posterior inputs match the seq-WM training path when
    ``--seq_wm_use_task_aux_inputs`` is enabled.
    """
    aux_components: list[torch.Tensor] = []

    def _push(x: Optional[torch.Tensor]) -> None:
        if x is None:
            aux_components.append(
                torch.zeros(T, N, 1, device=device, dtype=dtype)
            )
        else:
            if x.ndim == 2:
                aux_components.append(x.unsqueeze(-1).to(dtype=dtype))
            else:
                aux_components.append(x.to(dtype=dtype))

    _push(d_hat)
    _push(next_d_hat)
    if d_hat is not None and next_d_hat is not None:
        _push(next_d_hat - d_hat)
    else:
        _push(None)
    _push(success_count)
    _push(next_success_count)
    if success_count is not None and next_success_count is not None:
        _push(next_success_count - success_count)
    else:
        _push(None)
    _push(success_tolerance)
    _push(lifted)
    _push(failure)
    _push(picked)
    _push(dropped)
    _push(closest_d_hat)
    return torch.cat(aux_components, dim=-1)


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
        # Phase A4 P4 (2026-06-09): replay prioritization knobs (Part F).
        # Defaults preserve uniform sampling. When prioritize=True the
        # sampler computes per-(env, start_t) weights from ``failure``,
        # ``success_count`` deltas, and per-step success_tolerance, then
        # uses torch.multinomial to pick prioritized indices for the
        # non-uniform fraction. ``uniform_floor`` of indices stay uniform.
        prioritize_floor_recovery: bool = False,
        floor_weight: float = 2.0,
        hard_floor_weight: float = 3.0,
        positive_slope_weight: float = 2.0,
        drop_near_success_weight: float = 1.5,
        uniform_floor: float = 0.3,
        floor_tol_threshold: float = 0.02,
        hard_floor_tol_threshold: float = 0.012,
    ) -> None:
        if seq_len <= 0:
            raise ValueError(f"seq_len must be > 0, got {seq_len}")
        if seq_batch_size <= 0:
            raise ValueError(f"seq_batch_size must be > 0, got {seq_batch_size}")
        self.seq_len = int(seq_len)
        self.seq_batch_size = int(seq_batch_size)
        self.device = torch.device(device)
        self._gen = generator  # may be None -> default RNG
        self.prioritize_floor_recovery = bool(prioritize_floor_recovery)
        self.floor_weight = float(floor_weight)
        self.hard_floor_weight = float(hard_floor_weight)
        self.positive_slope_weight = float(positive_slope_weight)
        self.drop_near_success_weight = float(drop_near_success_weight)
        self.uniform_floor = float(min(1.0, max(0.0, uniform_floor)))
        self.floor_tol_threshold = float(floor_tol_threshold)
        self.hard_floor_tol_threshold = float(hard_floor_tol_threshold)
        self.last_replay_metrics: dict[str, float] = {}

    # ----- index sampling -----

    @staticmethod
    def _validate_env_index_mask(
        env_index_mask: Optional[torch.Tensor],
        N: int,
        device: torch.device,
    ) -> Optional[torch.Tensor]:
        """Return bool ``(N,)`` on ``device``, or ``None`` if unrestricted.

        Raises ``ValueError`` on shape / empty-mask errors (caller fail-closed).
        """
        if env_index_mask is None:
            return None
        if not isinstance(env_index_mask, torch.Tensor):
            raise ValueError(
                f"env_index_mask must be a Tensor or None, got {type(env_index_mask)}"
            )
        m = env_index_mask.detach().to(device=device)
        if m.ndim != 1 or int(m.shape[0]) != int(N):
            raise ValueError(
                f"env_index_mask must have shape (N,)={(N,)}; got {tuple(m.shape)}"
            )
        m = m.bool()
        if int(m.sum().item()) <= 0:
            raise ValueError("env_index_mask selects zero env slots")
        return m

    def _sample_env_indices(
        self,
        N: int,
        B: int,
        env_index_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Sample ``B`` env indices, optionally restricted to ``env_index_mask``."""
        if env_index_mask is None:
            return torch.randint(
                low=0,
                high=N,
                size=(B,),
                device=self.device,
                generator=self._gen,
            )
        allowed = env_index_mask.nonzero(as_tuple=False).reshape(-1)
        pick = torch.randint(
            low=0,
            high=int(allowed.numel()),
            size=(B,),
            device=self.device,
            generator=self._gen,
        )
        return allowed[pick].long()

    def _sample_indices(
        self,
        H: int,
        N: int,
        env_index_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (t_idx, e_idx) of shape (T, B) on self.device."""
        T = self.seq_len
        B = self.seq_batch_size
        if T > H:
            raise ValueError(
                f"seq_len ({T}) cannot exceed horizon_length ({H}); pass a "
                f"smaller --seq_wm_seq_len or extend the PPO rollout."
            )
        mask = self._validate_env_index_mask(env_index_mask, N, self.device)
        env_idx = self._sample_env_indices(N, B, mask)
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
        # Phase A4 P4 (2026-06-09): task-aware aux fields (Part B).
        picked_full: Optional[torch.Tensor] = None,
        dropped_full: Optional[torch.Tensor] = None,
        closest_d_hat_full: Optional[torch.Tensor] = None,
        success_tolerance_full: Optional[torch.Tensor] = None,
        # Optional pre-computed aux scalar tensor (T-only); when provided,
        # already shape (H, N, A) and bypasses the field-by-field assemble.
        aux_scalars_full: Optional[torch.Tensor] = None,
        # Held-out / train partition: bool (N,) selecting allowed env slots.
        # None preserves legacy unrestricted sampling.
        env_index_mask: Optional[torch.Tensor] = None,
        return_source_ids: bool = False,
        partition_worker_rank: int = 0,
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
        picked_full = _maybe_squeeze_last(picked_full)
        dropped_full = _maybe_squeeze_last(dropped_full)
        closest_d_hat_full = _maybe_squeeze_last(closest_d_hat_full)
        success_tolerance_full = _maybe_squeeze_last(success_tolerance_full)
        last_d_hat = _maybe_squeeze_last(last_d_hat) if last_d_hat is not None else None
        last_success_count = (
            _maybe_squeeze_last(last_success_count)
            if last_success_count is not None else None
        )

        if last_obs.shape != (N, D_obs):
            raise ValueError(
                f"last_obs must have shape (N, D_obs)=({N},{D_obs}); got {last_obs.shape}"
            )

        # Build (T, B) index grids. Optionally apply replay prioritization
        # weights for a fraction (1 - uniform_floor) of the batch.
        t_idx, e_idx = self._sample_indices_prioritized(
            H,
            N,
            success_tolerance_full=success_tolerance_full,
            failure_full=failure_full,
            success_count_full=success_count_full,
            d_hat_full=d_hat_full,
            env_index_mask=env_index_mask,
        )
        T, B = t_idx.shape
        assert T == self.seq_len and B == self.seq_batch_size
        start_t = t_idx[0].detach().clone()
        env_ids = e_idx[0].detach().clone()

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
        picked_w  = _gather_seq(picked_full,  t_idx, e_idx)
        dropped_w = _gather_seq(dropped_full, t_idx, e_idx)
        closest_w = _gather_seq(closest_d_hat_full, t_idx, e_idx)
        succtol_w = _gather_seq(success_tolerance_full, t_idx, e_idx)

        next_obs_w = _gather_next(obs_full, last_obs, t_idx, e_idx, H)
        next_d_hat_w = _gather_next(d_hat_full, last_d_hat, t_idx, e_idx, H)
        next_success_w = _gather_next(success_count_full, last_success_count, t_idx, e_idx, H)

        valid_mask, segment_id = self._build_mask_and_segment(done_w)

        # Phase A4 P4 (2026-06-09): build aux_scalars tensor (Part B).
        aux_scalars_w: Optional[torch.Tensor] = None
        if aux_scalars_full is not None:
            aux_scalars_w = _gather_seq(aux_scalars_full, t_idx, e_idx)
        else:
            aux_components: list[torch.Tensor] = []
            # Order is stable: d_hat, next_d_hat, delta, success_cnt,
            # next_success_cnt, success_delta, success_tolerance, lifted,
            # failure, picked, dropped, closest_d_hat. Missing fields
            # contribute a zero column so model.aux_dim can be audited.
            def _push(x: Optional[torch.Tensor]):
                if x is None:
                    aux_components.append(
                        torch.zeros(T, B, 1, device=obs_full.device, dtype=obs_full.dtype)
                    )
                else:
                    if x.ndim == 2:
                        aux_components.append(x.unsqueeze(-1).to(obs_full.dtype))
                    else:
                        aux_components.append(x.to(obs_full.dtype))
            _push(d_hat_w)
            _push(next_d_hat_w)
            if d_hat_w is not None and next_d_hat_w is not None:
                _push(next_d_hat_w - d_hat_w)
            else:
                _push(None)
            _push(success_w)
            _push(next_success_w)
            if success_w is not None and next_success_w is not None:
                _push(next_success_w - success_w)
            else:
                _push(None)
            _push(succtol_w)
            _push(lifted_w)
            _push(failure_w)
            _push(picked_w)
            _push(dropped_w)
            _push(closest_w)
            aux_scalars_w = torch.cat(aux_components, dim=-1)

        meta: dict = {
            "horizon_length": int(H),
            "num_actors": int(N),
            "obs_dim": int(D_obs),
            "seq_len": int(T),
            "seq_batch_size": int(B),
            "replay_prioritized": bool(self.prioritize_floor_recovery),
            "aux_dim": int(aux_scalars_w.shape[-1])
            if aux_scalars_w is not None else 0,
            "env_indices": env_ids.detach().cpu(),
            "start_t": start_t.detach().cpu(),
            "partition_restricted": bool(env_index_mask is not None),
        }
        if return_source_ids:
            # (B, 4): worker_rank, env_index, sequence_start, rollout_generation
            # generation is 0 for single-rollout experience buffers.
            wr = int(partition_worker_rank)
            src = torch.stack(
                [
                    torch.full((B,), wr, dtype=torch.long),
                    env_ids.detach().cpu().long(),
                    start_t.detach().cpu().long(),
                    torch.zeros(B, dtype=torch.long),
                ],
                dim=1,
            )
            meta["source_ids"] = src
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
            aux_scalars=aux_scalars_w,
            success_tolerance_seq=succtol_w,
            picked=picked_w,
            dropped=dropped_w,
            closest_d_hat=closest_w,
            meta=meta,
        )

    def _sample_indices_prioritized(
        self,
        H: int,
        N: int,
        *,
        success_tolerance_full: Optional[torch.Tensor],
        failure_full: Optional[torch.Tensor],
        success_count_full: Optional[torch.Tensor],
        d_hat_full: Optional[torch.Tensor],
        env_index_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Phase A4 P4 (2026-06-09): replay prioritization (Part F).

        When ``self.prioritize_floor_recovery`` is False this falls back
        to the legacy uniform sampler.

        Otherwise the (env, start_t) probability is built from a 0..1
        weight per (start_t, env) bin, scaled by per-feature multipliers.
        ``self.uniform_floor`` of the batch is reserved for legacy
        uniform draws so the prior over the data does not collapse.
        """
        if not self.prioritize_floor_recovery:
            return self._sample_indices(H, N, env_index_mask=env_index_mask)
        T = self.seq_len
        B = self.seq_batch_size
        if T > H:
            return self._sample_indices(H, N, env_index_mask=env_index_mask)
        mask = self._validate_env_index_mask(env_index_mask, N, self.device)
        n_uniform = max(1, int(round(self.uniform_floor * B)))
        n_prio = max(0, B - n_uniform)
        # Uniform pool first (re-use legacy logic):
        t_idx_u, e_idx_u = self._sample_indices(H, N, env_index_mask=mask)
        if n_uniform == B or n_prio == 0:
            return t_idx_u, e_idx_u
        # Build per-(t_start, env) weights. We work on shape (H_eff, N)
        # where H_eff = H - T + 1 to ensure the window fits.
        H_eff = max(1, H - T + 1)
        device = self.device
        weights = torch.ones(H_eff, N, device=device, dtype=torch.float32)
        if mask is not None:
            # Zero disallowed env columns so multinomial never picks them.
            weights = weights * mask.to(device=device, dtype=torch.float32).unsqueeze(0)
        # Floor stage from per-step success_tolerance averaged over
        # window. Missing -> uniform.
        if success_tolerance_full is not None:
            tol = success_tolerance_full.to(device=device, dtype=torch.float32)
            if tol.ndim == 3:
                tol = tol.squeeze(-1)
            # average tolerance over the (T)-length window starting at t.
            cum = tol.cumsum(dim=0)
            for ts in range(H_eff):
                end = ts + T
                window_sum = (
                    cum[end - 1] - (cum[ts - 1] if ts >= 1 else 0.0)
                )
                window_mean = window_sum / max(T, 1)
                hard_mask = (window_mean <= self.hard_floor_tol_threshold).float()
                floor_mask = (
                    (window_mean <= self.floor_tol_threshold).float() - hard_mask
                ).clamp_min(0.0)
                weights[ts] = (
                    weights[ts]
                    + (self.hard_floor_weight - 1.0) * hard_mask
                    + (self.floor_weight - 1.0) * floor_mask
                )
                if mask is not None:
                    weights[ts] = weights[ts] * mask.to(dtype=weights.dtype)
        # Drop-near-success: failure_full sees a positive within window
        # AND success_count rises within same window.
        if failure_full is not None and success_count_full is not None:
            fail = failure_full.to(device=device, dtype=torch.float32)
            sc = success_count_full.to(device=device, dtype=torch.float32)
            if fail.ndim == 3:
                fail = fail.squeeze(-1)
            if sc.ndim == 3:
                sc = sc.squeeze(-1)
            for ts in range(H_eff):
                end = ts + T
                fail_in_window = (fail[ts:end].sum(dim=0) > 0.5).float()
                sc_rise = (sc[end - 1] - sc[ts]).clamp_min(0.0)
                near_success = ((sc_rise > 0.5) & (fail_in_window > 0.5)).float()
                weights[ts] = (
                    weights[ts] + (self.drop_near_success_weight - 1.0) * near_success
                )
                if mask is not None:
                    weights[ts] = weights[ts] * mask.to(dtype=weights.dtype)
        # Positive slope: d_hat trends down within the window.
        if d_hat_full is not None:
            d_hat = d_hat_full.to(device=device, dtype=torch.float32)
            if d_hat.ndim == 3:
                d_hat = d_hat.squeeze(-1)
            for ts in range(H_eff):
                end = ts + T
                slope = (d_hat[ts] - d_hat[end - 1]).clamp_min(0.0)
                weights[ts] = (
                    weights[ts] + (self.positive_slope_weight - 1.0) * (slope > 1e-6).float()
                )
                if mask is not None:
                    weights[ts] = weights[ts] * mask.to(dtype=weights.dtype)
        # Sample prioritized indices via 1D multinomial.
        flat = weights.reshape(-1)
        flat_sum = float(flat.sum().item())
        if flat_sum <= 0.0:
            # Mask wiped all mass — fall back to restricted uniform.
            return self._sample_indices(H, N, env_index_mask=mask)
        flat = flat / (flat.sum() + 1e-9)
        idx = torch.multinomial(flat, num_samples=n_prio, replacement=True)
        ts_p = (idx // N).long()
        e_p = (idx % N).long()
        # Build (T, B) for prioritized portion.
        t_off = torch.arange(T, device=device).unsqueeze(1)
        t_idx_p = ts_p.unsqueeze(0) + t_off
        e_idx_p = e_p.unsqueeze(0).expand(T, n_prio).contiguous()
        # Concatenate uniform + prioritized; shuffle so layout is mixed.
        t_idx = torch.cat([t_idx_u[:, : n_uniform], t_idx_p], dim=1)
        e_idx = torch.cat([e_idx_u[:, : n_uniform], e_idx_p], dim=1)
        perm = torch.randperm(B, device=device)
        self.last_replay_metrics = {
            "uniform_fraction": float(n_uniform / max(B, 1)),
            "prio_fraction": float(n_prio / max(B, 1)),
            "weights_max": float(weights.max().item()),
            "weights_mean": float(weights.mean().item()),
        }
        return t_idx[:, perm], e_idx[:, perm]


__all__ = ["SequenceSampler", "build_rollout_aux_scalars"]
