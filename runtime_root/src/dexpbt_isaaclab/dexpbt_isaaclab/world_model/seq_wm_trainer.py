"""
Trainer for sequence world model strategies (RSSM-lite, Transformer-lite).

Sits beside the existing single-step ``WMTrainer`` (wm_trainer.py). The two
trainers do not share state: a run uses one or the other depending on the
active strategy. Both run after the PPO epoch's gradient step, but the
sequence trainer pulls its data straight from the rl_games ExperienceBuffer
(plus a returns tensor, computed inside ``train_epoch``), without going
through the legacy ``WMBuffer``.

Public API:
    SeqWMConfig             dataclass with all training hyperparameters
    SeqWMTrainer            owns the SequenceWorldModelBase + optimizer
                            and exposes a single ``step()`` method that
                            consumes an ExperienceBuffer-like object.

Saving / loading:
    ``state_dict()`` returns a dict suitable for ``torch.save``.
    ``load_state_dict(d)`` restores it. The same dict can be embedded under
    ``wm_seq_state_dict`` inside the existing wm_latest.pt checkpoint
    schema (see Phase A4 wiring).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional

import torch
import torch.nn.utils as nn_utils

from .seq_policy_aux import SeqDistillCache, build_teacher_cache_from_predictions
from .seq_wm_base import SequenceBatch, SequenceWorldModelBase
from .seq_wm_sampler import SequenceSampler
from .wm_aux_buffer import read_aux_full


@dataclass
class SeqWMConfig:
    """All hyperparameters that drive SeqWMTrainer behaviour.

    Maps roughly 1-1 to the --seq_wm_* CLI flags introduced in Phase A4.
    """

    # Sampler shapes.
    seq_len: int = 16
    seq_batch_size: int = 1024
    updates_per_ppo_epoch: int = 1

    # Optimization.
    lr: float = 1e-3
    weight_decay: float = 0.0
    grad_clip: float = 100.0

    # Loss weights (per-term, scaled by lambda_total in the trainer).
    lambda_total: float = 0.05
    lambda_state: float = 1.0
    lambda_reward: float = 1.0
    lambda_done: float = 0.2
    lambda_success: float = 1.0
    lambda_task_error: float = 1.0
    lambda_value: float = 0.1

    # belief_value-only knobs.
    value_detach_belief: bool = False  # plan revision: default false

    # KL warmup (RSSM only — Transformer ignores).
    beta_kl_start: float = 0.05
    beta_kl_end: float = 0.20
    beta_kl_warmup_frames: int = 200_000_000
    free_nats: float = 1.0  # informational; the model already owns this

    # Logging / checkpointing.
    log_every_n_steps: int = 50
    seq_wm_save_every_n_steps: int = 200

    # Misc.
    seed: int = 0


def _beta_kl_at(frames: int, cfg: SeqWMConfig) -> float:
    """Linear warmup from `beta_kl_start` to `beta_kl_end` over
    `beta_kl_warmup_frames`. Constant before/after."""
    if cfg.beta_kl_warmup_frames <= 0:
        return cfg.beta_kl_end
    f = max(0.0, min(1.0, frames / float(cfg.beta_kl_warmup_frames)))
    return cfg.beta_kl_start + f * (cfg.beta_kl_end - cfg.beta_kl_start)


class SeqWMTrainer:
    """Drives a SequenceWorldModelBase on data from a PPO ExperienceBuffer."""

    def __init__(
        self,
        *,
        model: SequenceWorldModelBase,
        cfg: SeqWMConfig,
        device: torch.device | str = "cuda",
        sampler: Optional[SequenceSampler] = None,
        strategy: str = "rssm_belief_value",
    ) -> None:
        self.model = model
        self.cfg = cfg
        self.device = torch.device(device)
        self.strategy = str(strategy)

        self.model.to(self.device)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
        )
        self.sampler = sampler or SequenceSampler(
            seq_len=cfg.seq_len,
            seq_batch_size=cfg.seq_batch_size,
            device=self.device,
        )
        self.global_step = 0
        self.last_metrics: dict[str, float] = {}

    # ----- public API -----

    @torch.no_grad()
    def _build_loss_weights(self) -> dict[str, float]:
        c = self.cfg
        return {
            "state": c.lambda_state,
            "reward": c.lambda_reward,
            "done": c.lambda_done,
            "success": c.lambda_success,
            "task_error": c.lambda_task_error,
            "value": c.lambda_value,
        }

    def step(
        self,
        *,
        experience_buffer: Any,
        frames: int,
        last_obs: torch.Tensor,
        returns_full: torch.Tensor,
        last_d_hat: Optional[torch.Tensor] = None,
        last_success_count: Optional[torch.Tensor] = None,
    ) -> dict[str, float]:
        """One sequence-WM update from the current PPO ExperienceBuffer.

        Args:
            experience_buffer: rl_games-like buffer with .tensor_dict containing
                'obses', 'actions', 'rewards', 'dones' shaped (T, N, ...) and
                optional aux fields registered via wm_aux_buffer.register_seq_aux_fields.
            frames: total environment frames seen by this policy so far
                (used for beta_kl warmup).
            last_obs: (N, obs_dim) — the obs at "step H" (post-rollout).
            returns_full: (T, N) or (T, N, 1) — GAE returns from this rollout.
            last_d_hat: (N,) optional, used as the "step H" d_hat.
            last_success_count: (N,) optional.

        Returns:
            dict of scalar metrics (suitable for direct TB logging under wm_seq/*).
        """
        td = experience_buffer.tensor_dict
        obs_full = td["obses"]
        action_full = td["actions"]
        reward_full = td["rewards"]
        done_full = td["dones"].to(obs_full.dtype)

        d_hat_full = read_aux_full(experience_buffer, "aux_d_hat")
        failure_full = read_aux_full(experience_buffer, "aux_failure")
        lifted_full = read_aux_full(experience_buffer, "aux_lifted")
        success_count_full = read_aux_full(experience_buffer, "aux_success_cnt")

        cumulative_metrics: dict[str, float] = {}

        for _update_i in range(self.cfg.updates_per_ppo_epoch):
            batch = self.sampler.sample(
                obs_full=obs_full,
                action_full=action_full,
                reward_full=reward_full,
                done_full=done_full,
                returns_full=returns_full,
                last_obs=last_obs.to(obs_full.dtype),
                d_hat_full=d_hat_full,
                last_d_hat=last_d_hat,
                failure_full=failure_full,
                success_count_full=success_count_full,
                last_success_count=last_success_count,
                lifted_full=lifted_full,
            )
            batch = batch.to(self.device)
            beta = _beta_kl_at(frames, self.cfg)

            # Sub-models that use beta_kl accept it; others ignore.
            kwargs: dict[str, Any] = {
                "loss_weights": self._build_loss_weights(),
                "beta_kl": beta,
                "value_detach_belief": self.cfg.value_detach_belief,
            }
            losses = self.model.compute_aux_loss(batch, **kwargs)

            scaled = self.cfg.lambda_total * losses["loss_total"]
            self.optimizer.zero_grad(set_to_none=True)
            scaled.backward()
            grad_norm = nn_utils.clip_grad_norm_(
                self.model.parameters(), self.cfg.grad_clip
            )
            self.optimizer.step()
            self.global_step += 1

            metrics = self.model.report_metrics()
            metrics.update({
                "lambda_total": float(self.cfg.lambda_total),
                "scaled_loss_total": float(scaled.detach().item()),
                "grad_norm": float(grad_norm.detach().item()),
                "frames": int(frames),
                "global_step": int(self.global_step),
                "seq_len": int(self.cfg.seq_len),
                "seq_batch_size": int(self.cfg.seq_batch_size),
                "active_strategy_id": float(hash(self.strategy) % 1_000_003),
            })
            # Accumulate (for last-update visibility, just keep the latest).
            cumulative_metrics = metrics

        self.last_metrics = cumulative_metrics
        return cumulative_metrics

    # ----- teacher cache for PPO distillation (Phase A4 P0) -----

    def compute_teacher_cache(
        self,
        *,
        experience_buffer: Any,
        last_obs: torch.Tensor,
        returns_full: torch.Tensor,
        last_d_hat: Optional[torch.Tensor] = None,
        last_success_count: Optional[torch.Tensor] = None,
        generation: int,
        teacher_value_norm: str = "none",
        teacher_value_norm_min_std: float = 1e-3,
        compute_teacher_return_corr: bool = False,
    ) -> SeqDistillCache:
        """Build a frozen ``(T*N,)`` teacher target cache for the current rollout.

        Runs ``self.model.encode_belief`` on a deterministic ``(T, N)``
        ``SequenceBatch`` covering the entire rollout (no random sampling).
        Output values are detached and laid out to match rl_games'
        ``swap_and_flatten01``: ``flat[n * T + t] == teacher[t, n]``.

        Returns a ``SeqDistillCache`` whose ``generation`` field equals
        the supplied ``generation`` argument; the bridge cross-checks it
        against the live rollout generation in ``compute_seq_distill_aux_loss``
        and raises if they disagree.

        Caller must ensure ``returns_full`` is shaped ``(T, N)`` (squeezed)
        so the SequenceBatch contract holds.
        """
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

        def _squeeze(t):
            if t is None:
                return None
            if t.ndim >= 1 and t.shape[-1] == 1:
                return t.squeeze(-1)
            return t

        reward_2d = _squeeze(reward_full)
        done_2d = _squeeze(done_full)
        returns_2d = _squeeze(returns_full)
        d_hat_2d = _squeeze(d_hat_full)
        failure_2d = _squeeze(failure_full)
        lifted_2d = _squeeze(lifted_full)
        success_2d = _squeeze(success_count_full)
        last_d_hat_1d = _squeeze(last_d_hat) if last_d_hat is not None else None
        last_success_1d = (
            _squeeze(last_success_count) if last_success_count is not None else None
        )

        # next_* fields for the deterministic full-rollout batch.
        # next_obs[T-1] uses last_obs (post-rollout step H), exactly like
        # the random sampler when start_t + T == H. For t < T-1, next_obs[t]
        # = obs_full[t+1].
        next_obs = torch.empty_like(obs_full)
        next_obs[:-1] = obs_full[1:]
        next_obs[-1] = last_obs.to(obs_full.dtype)

        def _build_next_scalar(full, last_val):
            if full is None:
                return None
            out = torch.empty_like(full)
            out[:-1] = full[1:]
            if last_val is not None:
                out[-1] = last_val.to(full.dtype)
            else:
                out[-1] = full[-1]
            return out

        next_d_hat = _build_next_scalar(d_hat_2d, last_d_hat_1d)
        next_success = _build_next_scalar(success_2d, last_success_1d)

        # Per-step valid_mask / segment_id for the full rollout.
        valid_mask = torch.ones(T, N, dtype=obs_full.dtype, device=obs_full.device)
        if T >= 2:
            cum = torch.cumsum(done_2d[:-1], dim=0)
            valid_mask[1:] = (cum < 0.5).to(obs_full.dtype)
        segment_id = torch.zeros(T, N, dtype=torch.long, device=obs_full.device)
        if T >= 2:
            segment_id[1:] = torch.cumsum(done_2d[:-1], dim=0).long()

        batch = SequenceBatch(
            obs=obs_full,
            action=action_full,
            reward=reward_2d,
            done=done_2d,
            next_obs=next_obs,
            returns=returns_2d,
            d_hat=d_hat_2d,
            next_d_hat=next_d_hat,
            failure=failure_2d,
            success_count=success_2d,
            next_success_count=next_success,
            lifted=lifted_2d,
            valid_mask=valid_mask,
            segment_id=segment_id,
            meta={
                "horizon_length": T,
                "num_actors": N,
                "obs_dim": int(obs_full.shape[-1]),
                "seq_len": T,
                "seq_batch_size": N,
            },
        ).to(self.device)

        with torch.no_grad():
            preds = self.model.encode_belief(batch)
        teacher_value = preds.get("value")
        teacher_success_logit = preds.get("success_logit")

        returns_for_corr = (
            returns_2d
            if (compute_teacher_return_corr and teacher_value is not None)
            else None
        )

        cache = build_teacher_cache_from_predictions(
            teacher_value=teacher_value,
            teacher_success_logit=teacher_success_logit,
            T=T,
            N=N,
            generation=int(generation),
            teacher_value_norm=str(teacher_value_norm),
            teacher_value_norm_min_std=float(teacher_value_norm_min_std),
            returns_for_corr=returns_for_corr,
        )
        return cache

    # ----- save / load -----

    def state_dict(self) -> dict[str, Any]:
        return {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "global_step": int(self.global_step),
            "strategy": self.strategy,
            "cfg": dict(self.cfg.__dict__),
        }

    def load_state_dict(self, d: Mapping[str, Any], strict: bool = True) -> None:
        self.model.load_state_dict(d["model_state_dict"], strict=strict)
        if "optimizer_state_dict" in d:
            self.optimizer.load_state_dict(d["optimizer_state_dict"])
        self.global_step = int(d.get("global_step", 0))


__all__ = ["SeqWMConfig", "SeqWMTrainer", "_beta_kl_at"]
