"""
Sequence World Model — base interface and SequenceBatch dataclass.

This module defines the abstract surface that RSSM-lite and Transformer-lite
backbones must implement. It is *purely auxiliary*: the backbones only see
sequences sampled from the existing PPO ExperienceBuffer (plus aux scalars
written by the patched play_steps_rnn loop) — they never feed back into the
actor's loss path or into the policy observation.

Target alignment convention (must match SequenceSampler exactly):
    obs_t        = s_t                                 from buffer.obses[t]
    action_t     = a_t                                 from buffer.actions[t]
    reward_t     = r_t  (env returns this with done_t) from buffer.rewards[t]
    done_t       = 1[episode ended at step t]          from buffer.dones[t]
    next_obs_t   = s_{t+1}  (= obs[t+1] inside window) virtual; targets only
    aux.d_hat_t  = d_hat at obs_t                      from aux_tensor_dict
    returns_t    = R_t (GAE return)                    from batch_dict["returns"]

Heads predict TIME-t targets from belief b_t and action a_t:
    next_obs_delta_head(b_t, a_t)  ↔  next_obs_t - obs_t
    reward_head(b_t, a_t)          ↔  reward_t
    done_head(b_t, a_t)            ↔  done_t
    success_head(b_t, a_t)         ↔  (success_count_{t+1} > success_count_t)
    task_error_delta_head(b_t,a_t) ↔  aux.d_hat_{t+1} - aux.d_hat_t
    value_head(b_t)                ↔  returns_t      (only *_belief_value)

valid_mask[t, b]:
    1 iff env b has not seen any done_{<t} *within the sampled window*.
    Loss for state_delta and task_error_delta is masked by valid_mask AND by
    (1 - done_t) (next_obs / next_d_hat is undefined when episode ends at t).
    reward_t / done_t / value_t losses keep done_t rows.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

import torch


@dataclass
class SequenceBatch:
    """One sampled sequence batch ready for the world model.

    Shapes use convention (T, B, ...) where T = seq_len, B = seq_batch_size.

    Required tensors come from the PPO ExperienceBuffer + aux_tensor_dict.
    Optional tensors may be None if the env / pipeline doesn't provide them
    (e.g. on environments without a `success` counter); the backbones must
    handle missing fields gracefully.
    """

    obs: torch.Tensor          # (T, B, obs_dim)        s_t
    action: torch.Tensor       # (T, B, action_dim)     a_t
    reward: torch.Tensor       # (T, B)                 r_t  (NOT (T,B,1))
    done: torch.Tensor         # (T, B) float           done_t in {0,1}
    next_obs: torch.Tensor     # (T, B, obs_dim)        s_{t+1}; for t=T-1
                               #                        equals last buffered obs

    # GAE return per step (PPO computes this in batch_dict["returns"]).
    returns: torch.Tensor      # (T, B)

    # Auxiliary signals (some optional). All shapes (T, B) unless noted.
    d_hat: Optional[torch.Tensor] = None         # (T, B)   task error at obs_t
    next_d_hat: Optional[torch.Tensor] = None    # (T, B)   task error at obs_{t+1}
    failure: Optional[torch.Tensor] = None       # (T, B)   failure flag
    success_count: Optional[torch.Tensor] = None # (T, B)   per-env success counter
    next_success_count: Optional[torch.Tensor] = None  # (T, B)
    lifted: Optional[torch.Tensor] = None        # (T, B)   picked / lifted flag

    # Cross-episode mask. valid_mask[t, b] = 1 iff env b has not crossed any
    # done in steps [0..t-1] within this window. Used to:
    #   - suppress state_delta / task_error_delta loss for the first step after a
    #     done (i.e. wherever valid_mask=0), and
    #   - signal the RSSM/Transformer to reset its hidden state at those points.
    # Shape: (T, B). Element t=0 is always 1.
    valid_mask: torch.Tensor = None  # type: ignore[assignment]

    # Segment ids per (t, b), increasing by 1 on each done. Useful for
    # vectorised hidden-state reset in RSSM. Shape: (T, B), dtype long.
    segment_id: torch.Tensor = None  # type: ignore[assignment]

    # Phase A4 P4 (2026-06-09): task-aware aux scalars (Part B).
    # ``aux_scalars[t,b,:]`` is a small concatenation of normalised task
    # signals (d_hat, next_d_hat, delta_d_hat, success_count, next_success_
    # count, success_delta, success_tolerance, lifted, failure, ...). The
    # field is None when --seq_wm_use_task_aux_inputs is False. Shape:
    # (T, B, aux_dim). The model is responsible for handling None
    # gracefully (default forward keeps the legacy path).
    aux_scalars: Optional[torch.Tensor] = None
    # Per-step success_tolerance (T, B). Used by floor-aware loss
    # weighting in compute_aux_loss; None disables the feature.
    success_tolerance_seq: Optional[torch.Tensor] = None
    # Per-step picked / dropped flags (Part B). Optional; not consumed
    # by the legacy loss computation, only by Phase-A4 P4 horizon /
    # recovery / replay-prio code paths when those features are enabled.
    picked: Optional[torch.Tensor] = None
    dropped: Optional[torch.Tensor] = None
    closest_d_hat: Optional[torch.Tensor] = None

    # Bookkeeping (not consumed by losses).
    meta: dict = field(default_factory=dict)

    @property
    def T(self) -> int:
        return self.obs.shape[0]

    @property
    def B(self) -> int:
        return self.obs.shape[1]

    @property
    def obs_dim(self) -> int:
        return self.obs.shape[-1]

    @property
    def action_dim(self) -> int:
        return self.action.shape[-1]

    @property
    def device(self) -> torch.device:
        return self.obs.device

    def to(self, device: torch.device | str) -> "SequenceBatch":
        """Return a SequenceBatch with all tensors moved to `device`."""
        def _mv(t: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
            return t if t is None else t.to(device)
        return SequenceBatch(
            obs=_mv(self.obs),
            action=_mv(self.action),
            reward=_mv(self.reward),
            done=_mv(self.done),
            next_obs=_mv(self.next_obs),
            returns=_mv(self.returns),
            d_hat=_mv(self.d_hat),
            next_d_hat=_mv(self.next_d_hat),
            failure=_mv(self.failure),
            success_count=_mv(self.success_count),
            next_success_count=_mv(self.next_success_count),
            lifted=_mv(self.lifted),
            valid_mask=_mv(self.valid_mask),
            segment_id=_mv(self.segment_id),
            aux_scalars=_mv(self.aux_scalars),
            success_tolerance_seq=_mv(self.success_tolerance_seq),
            picked=_mv(self.picked),
            dropped=_mv(self.dropped),
            closest_d_hat=_mv(self.closest_d_hat),
            meta=dict(self.meta),
        )


class SequenceWorldModelBase(torch.nn.Module, ABC):
    """Abstract base class for sequence-modelling auxiliary world models.

    Subclasses (RSSMLiteWorldModel, TransformerLiteWorldModel) implement
    ``encode_sequence`` to turn a SequenceBatch into per-timestep beliefs
    ``b_t`` of shape (T, B, belief_dim), then share PredictionHeads to compute
    losses against the canonical targets.

    The base class deliberately does NOT own an optimizer — that lives in
    SeqWMTrainer, so we can clip / schedule / log centrally.
    """

    @abstractmethod
    def encode_sequence(self, batch: SequenceBatch) -> torch.Tensor:
        """Run the backbone and return belief tensor of shape (T, B, belief_dim).

        Subclasses MUST honour ``batch.valid_mask`` / ``batch.segment_id``:
        the belief at any (t, b) where a previous step in the same window had
        done=1 must NOT carry forward state from before that done.
        """

    @abstractmethod
    def compute_aux_loss(self, batch: SequenceBatch) -> dict[str, torch.Tensor]:
        """Run encode_sequence + heads + losses, return per-term loss dict.

        Required keys: 'loss_total'.
        Optional keys (all scalar tensors): 'loss_state', 'loss_reward',
        'loss_done', 'loss_success', 'loss_task_error', 'loss_value', 'loss_kl'.
        Loss names match the wm_seq/* TensorBoard scalars.
        """

    def get_belief(self, batch: SequenceBatch) -> torch.Tensor:
        """Public accessor for belief; default just calls encode_sequence."""
        return self.encode_sequence(batch)

    @torch.no_grad()
    def encode_belief(
        self, batch: SequenceBatch
    ) -> dict[str, Optional[torch.Tensor]]:
        """Detached teacher forward used by the PPO distillation bridge.

        Runs ``encode_sequence`` under ``no_grad``, then runs the
        ``value_head`` / ``success_head`` (when present) on the resulting
        belief and returns detached predictions. The result is suitable as
        a frozen distillation target for a PPO-side student head.

        Returns a dict with:
            ``belief``        Tensor[T, B, belief_dim] (detached)
            ``value``         Tensor[T, B] or None     (only if has_value_head)
            ``success_logit`` Tensor[T, B] or None     (only if has_success_head)
        Subclasses may override (e.g. to skip reparameterisation noise) but
        the default implementation is correct for both RSSM-lite and
        Transformer-lite because their ``encode_sequence`` signatures match
        the base contract and their ``heads`` bundle exposes the same
        ``predict_value`` / ``success_head`` accessors.
        """
        was_training = self.training
        self.eval()
        try:
            belief = self.encode_sequence(batch)
        finally:
            if was_training:
                self.train()
        belief = belief.detach()

        T, B = belief.shape[0], belief.shape[1]
        belief_flat = belief.reshape(T * B, -1)
        value = None
        success_logit = None
        heads = getattr(self, "heads", None)
        if heads is not None:
            if getattr(heads, "has_value_head", False):
                v = heads.predict_value(belief_flat)
                if v is not None:
                    value = v.reshape(T, B).detach()
            success_head = getattr(heads, "success_head", None)
            if success_head is not None:
                action_flat = batch.action.reshape(T * B, -1)
                ba = torch.cat([belief_flat, action_flat], dim=-1)
                logits = success_head(ba).squeeze(-1)
                success_logit = logits.reshape(T, B).detach()
        return {
            "belief": belief,
            "value": value,
            "success_logit": success_logit,
        }

    @abstractmethod
    def report_metrics(self) -> dict[str, float]:
        """Return scalar metrics to log under wm_rssm/* or wm_trans/*.

        Called at most once per WM training step. Should NOT include any
        tensor that requires grad; return Python floats only.
        """


__all__ = ["SequenceBatch", "SequenceWorldModelBase"]
