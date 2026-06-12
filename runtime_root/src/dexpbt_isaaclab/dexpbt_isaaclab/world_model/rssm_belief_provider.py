"""RSSM belief cache for EIR-MPC full H-step MPPI (Phase: full MPPI).

The H-step planner needs a per-rollout-row belief ``b_t`` to seed its prior
latent rollout. Encoding the RSSM per PPO minibatch would be wasteful and would
re-encode the same rollout many times, so we build the belief ONCE per rollout
here (deterministic full-rollout ``encode_sequence``, posterior mean, no random
sampler) and flatten it to the same ``flat[n*T + t] = belief[t, n]`` convention
as the EIR target cache / elite-bank features so the planner can index a belief
for any risky minibatch row.

Everything is detached (no grad into the world model). On any failure the cache
is returned with ``valid=False`` and a ``fallback_reason`` so the planner self-
disables cleanly instead of crashing the run.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import torch


BELIEF_OK = 0
BELIEF_DISABLED = 1
BELIEF_NO_SEQ_WM = 2
BELIEF_NOT_RSSM = 3
BELIEF_BUILD_BATCH_FAILED = 4
BELIEF_ENCODE_FAILED = 5
BELIEF_SHAPE_MISMATCH = 6
BELIEF_STALE = 7
BELIEF_NO_HEADS = 8


@dataclass
class RSSMBeliefCache:
    belief_flat: Optional[torch.Tensor]
    state_flat: Optional[Any]
    frame: int
    generation: int
    rows: int
    belief_dim: int
    valid: bool
    fallback_reason: int
    source: str

    @staticmethod
    def invalid(reason: int, *, frame: int = 0, generation: int = -1, source: str = "none") -> "RSSMBeliefCache":
        return RSSMBeliefCache(
            belief_flat=None, state_flat=None, frame=int(frame),
            generation=int(generation), rows=0, belief_dim=0,
            valid=False, fallback_reason=int(reason), source=source,
        )

    def slice_rows(self, start: int, end: int) -> Optional[torch.Tensor]:
        """Return ``belief_flat[start:end]`` or None when unavailable/misaligned."""
        if not self.valid or self.belief_flat is None:
            return None
        if start < 0 or end > int(self.belief_flat.shape[0]) or end <= start:
            return None
        return self.belief_flat[start:end]


def _squeeze_last1(t: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    if t is None:
        return None
    if t.ndim >= 1 and t.shape[-1] == 1:
        return t.squeeze(-1)
    return t


def _flatten_tn(x: torch.Tensor) -> torch.Tensor:
    """``(T, N, ...) -> (N*T, ...)`` with ``flat[n*T + t] = x[t, n]``."""
    T = x.shape[0]
    N = x.shape[1]
    rest = x.shape[2:]
    # (T, N, ...) -> (N, T, ...) -> (N*T, ...)
    return x.permute(1, 0, *range(2, x.dim())).reshape(N * T, *rest)


def build_rssm_belief_cache(
    *,
    seq_wm_trainer: Any,
    experience_buffer: Any,
    returns_full: Optional[torch.Tensor],
    last_obs: Optional[torch.Tensor],
    last_d_hat: Optional[torch.Tensor],
    last_success_count: Optional[torch.Tensor],
    frames: int,
    generation: int,
    device: Any,
    enabled: bool = True,
) -> RSSMBeliefCache:
    """Encode the current rollout into a flattened belief cache.

    Reuses the deterministic full-rollout ``SequenceBatch`` construction from
    ``SeqWMTrainer.compute_teacher_cache`` (no random sampling), then runs
    ``model.encode_sequence(..., deterministic=True)`` under ``no_grad`` and
    flattens with ``flat[n*T+t] = belief[t, n]``.
    """
    if not enabled:
        return RSSMBeliefCache.invalid(BELIEF_DISABLED, frame=frames, generation=generation)
    if seq_wm_trainer is None:
        return RSSMBeliefCache.invalid(BELIEF_NO_SEQ_WM, frame=frames, generation=generation)
    model = getattr(seq_wm_trainer, "model", None)
    if model is None or not hasattr(model, "encode_sequence"):
        return RSSMBeliefCache.invalid(BELIEF_NO_SEQ_WM, frame=frames, generation=generation)
    # Only RSSM-family backbones expose belief = concat(h, z) + a prior
    # transition usable by imagine_step. Transformer-lite is encode-only.
    if not hasattr(model, "get_state_from_belief"):
        return RSSMBeliefCache.invalid(BELIEF_NOT_RSSM, frame=frames, generation=generation)
    heads = getattr(model, "heads", None)
    if heads is None:
        return RSSMBeliefCache.invalid(BELIEF_NO_HEADS, frame=frames, generation=generation)

    try:
        from .seq_wm_base import SequenceBatch
        from .wm_aux_buffer import read_aux_full

        td = experience_buffer.tensor_dict
        obs_full = td["obses"]
        action_full = td["actions"]
        reward_full = td["rewards"]
        done_full = td["dones"].to(obs_full.dtype)
        T = int(obs_full.shape[0])
        N = int(obs_full.shape[1])

        d_hat_full = read_aux_full(experience_buffer, "aux_d_hat")
        failure_full = read_aux_full(experience_buffer, "aux_failure")
        lifted_full = read_aux_full(experience_buffer, "aux_lifted")
        success_count_full = read_aux_full(experience_buffer, "aux_success_cnt")
        succtol_full = read_aux_full(experience_buffer, "aux_success_tolerance")
        picked_full = read_aux_full(experience_buffer, "aux_picked")
        dropped_full = read_aux_full(experience_buffer, "aux_dropped")
        closest_full = read_aux_full(experience_buffer, "aux_closest_d_hat")

        reward_2d = _squeeze_last1(reward_full)
        done_2d = _squeeze_last1(done_full)
        returns_2d = _squeeze_last1(returns_full) if returns_full is not None else torch.zeros(T, N, device=obs_full.device, dtype=obs_full.dtype)
        d_hat_2d = _squeeze_last1(d_hat_full)
        failure_2d = _squeeze_last1(failure_full)
        lifted_2d = _squeeze_last1(lifted_full)
        success_2d = _squeeze_last1(success_count_full)
        succtol_2d = _squeeze_last1(succtol_full)
        picked_2d = _squeeze_last1(picked_full)
        dropped_2d = _squeeze_last1(dropped_full)
        closest_2d = _squeeze_last1(closest_full)
        last_d_hat_1d = _squeeze_last1(last_d_hat) if last_d_hat is not None else None
        last_success_1d = _squeeze_last1(last_success_count) if last_success_count is not None else None

        next_obs = torch.empty_like(obs_full)
        next_obs[:-1] = obs_full[1:]
        next_obs[-1] = (last_obs.to(obs_full.dtype) if last_obs is not None else obs_full[-1])

        def _build_next(full, last_val):
            if full is None:
                return None
            out = torch.empty_like(full)
            out[:-1] = full[1:]
            out[-1] = last_val.to(full.dtype) if last_val is not None else full[-1]
            return out

        next_d_hat = _build_next(d_hat_2d, last_d_hat_1d)
        next_success = _build_next(success_2d, last_success_1d)

        valid_mask = torch.ones(T, N, dtype=obs_full.dtype, device=obs_full.device)
        if T >= 2:
            cum = torch.cumsum(done_2d[:-1], dim=0)
            valid_mask[1:] = (cum < 0.5).to(obs_full.dtype)
        segment_id = torch.zeros(T, N, dtype=torch.long, device=obs_full.device)
        if T >= 2:
            segment_id[1:] = torch.cumsum(done_2d[:-1], dim=0).long()

        aux_scalars = None
        success_tolerance_seq = None
        aux_dim = int(getattr(model, "aux_dim", 0) or 0)
        if aux_dim > 0:
            from .seq_wm_sampler import build_rollout_aux_scalars

            aux_scalars = build_rollout_aux_scalars(
                T=T,
                N=N,
                d_hat=d_hat_2d,
                next_d_hat=next_d_hat,
                success_count=success_2d,
                next_success_count=next_success,
                success_tolerance=succtol_2d,
                lifted=lifted_2d,
                failure=failure_2d,
                picked=picked_2d,
                dropped=dropped_2d,
                closest_d_hat=closest_2d,
                device=obs_full.device,
                dtype=obs_full.dtype,
            )
            if int(aux_scalars.shape[-1]) != aux_dim:
                print(
                    f"[EIR-belief] aux scalar dim {aux_scalars.shape[-1]} != model.aux_dim {aux_dim}"
                )
                return RSSMBeliefCache.invalid(
                    BELIEF_SHAPE_MISMATCH, frame=frames, generation=generation
                )
            success_tolerance_seq = succtol_2d

        batch = SequenceBatch(
            obs=obs_full, action=action_full, reward=reward_2d, done=done_2d,
            next_obs=next_obs, returns=returns_2d, d_hat=d_hat_2d,
            next_d_hat=next_d_hat, failure=failure_2d, success_count=success_2d,
            next_success_count=next_success, lifted=lifted_2d,
            valid_mask=valid_mask, segment_id=segment_id,
            aux_scalars=aux_scalars,
            success_tolerance_seq=success_tolerance_seq,
            meta={"horizon_length": T, "num_actors": N, "seq_len": T, "seq_batch_size": N},
        ).to(device)
    except Exception as exc:  # pragma: no cover - defensive
        print(f"[EIR-belief] build batch failed: {exc}")
        return RSSMBeliefCache.invalid(BELIEF_BUILD_BATCH_FAILED, frame=frames, generation=generation)

    try:
        was_training = model.training
        model.eval()
        try:
            with torch.no_grad():
                belief_seq = model.encode_sequence(batch, deterministic=True)
        finally:
            if was_training:
                model.train()
    except TypeError:
        try:
            with torch.no_grad():
                belief_seq = model.encode_sequence(batch)
        except Exception as exc:  # pragma: no cover
            print(f"[EIR-belief] encode failed: {exc}")
            return RSSMBeliefCache.invalid(BELIEF_ENCODE_FAILED, frame=frames, generation=generation)
    except Exception as exc:  # pragma: no cover
        print(f"[EIR-belief] encode failed: {exc}")
        return RSSMBeliefCache.invalid(BELIEF_ENCODE_FAILED, frame=frames, generation=generation)

    if belief_seq is None or belief_seq.dim() != 3 or belief_seq.shape[0] != T or belief_seq.shape[1] != N:
        return RSSMBeliefCache.invalid(BELIEF_SHAPE_MISMATCH, frame=frames, generation=generation)

    belief_flat = _flatten_tn(belief_seq.detach()).contiguous()
    if int(belief_flat.shape[0]) != T * N:
        return RSSMBeliefCache.invalid(BELIEF_SHAPE_MISMATCH, frame=frames, generation=generation)

    return RSSMBeliefCache(
        belief_flat=belief_flat,
        state_flat=None,
        frame=int(frames),
        generation=int(generation),
        rows=int(T * N),
        belief_dim=int(belief_flat.shape[-1]),
        valid=True,
        fallback_reason=BELIEF_OK,
        source="rssm_encode_sequence",
    )


__all__ = [
    "RSSMBeliefCache",
    "build_rssm_belief_cache",
    "BELIEF_OK",
    "BELIEF_DISABLED",
    "BELIEF_NO_SEQ_WM",
    "BELIEF_NOT_RSSM",
    "BELIEF_BUILD_BATCH_FAILED",
    "BELIEF_ENCODE_FAILED",
    "BELIEF_SHAPE_MISMATCH",
    "BELIEF_STALE",
    "BELIEF_NO_HEADS",
]
