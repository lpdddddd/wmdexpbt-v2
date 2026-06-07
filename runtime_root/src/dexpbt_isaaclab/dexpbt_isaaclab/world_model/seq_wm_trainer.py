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
