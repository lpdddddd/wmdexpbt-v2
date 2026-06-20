"""Online Elite-AWR buffer, loss, and sidecar checkpoint helpers (v17).

Uses sequence-aware recurrent recomputation. When a stored sequence starts at
episode reset, logprob follows the same rl_games recurrent convention with
zero initial RNN state. Otherwise the prefix inside the stored segment is
burn-in approximation, not exact PPO hidden reconstruction.
"""

from __future__ import annotations

import copy
import math
import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Optional

import torch
import torch.nn.functional as F


class DoneReason(str, Enum):
    SUCCESS = "done_by_success"
    TIMEOUT = "done_by_timeout"
    FAILURE = "done_by_failure_or_drop"
    UNKNOWN = "done_reason_unknown"


@dataclass
class EliteAWRConfig:
    enable: bool = False
    loss_weight: float = 0.0
    start_after_frames: int = 800_000_000
    ramp_frames: int = 300_000_000
    min_buffer_size: int = 50_000
    buffer_capacity: int = 400_000
    batch_steps: int = 4096
    minibatch_max_sequences: int = 128
    update_every: int = 4
    max_episode_len: int = 200
    min_burnin_steps: int = 32
    min_loss_steps_per_entry: int = 16
    weight_mode: str = "exp_quality"
    beta: float = 1.0
    weight_max: float = 5.0
    quality_min: float = 1.5
    delta_metric: str = "mean"
    delta_soft: float = 0.20
    delta_mid: float = 0.45
    delta_hard: float = 0.80
    delta_mid_weight: float = 0.5
    delta_far_weight: float = 0.2
    loss_cap: float = 0.03
    kl_gate_enable: bool = False
    kl_soft: float = 0.05
    kl_hard: float = 0.10
    ref_update_interval_frames: int = 200_000_000
    local_sample_frac: float = 0.7
    inherited_sample_frac: float = 0.2
    broadcast_sample_frac: float = 0.1
    copy_top_k_steps: int = 50_000
    broadcast_top_k_steps: int = 10_000
    broadcast_min_source_nsucc: float = 35.0
    pause_on_regression: bool = True
    regression_drop_threshold: float = 4.0
    regression_pause_frames: int = 100_000_000
    sidecar_format_version: int = 1
    collection_mode: str = "rollout_buffer"
    collect_after_frames: int = 600_000_000
    collect_every: int = 4
    score_temperature: float = 1.0
    entropy_floor_eps: float = 0.5
    score_beta_quality: float = 1.0
    score_beta_similarity: float = 0.5
    score_beta_skill_delta: float = 0.25
    delta_gate_gamma: float = 0.05


@dataclass
class EliteAWREntry:
    obs: torch.Tensor
    action: torch.Tensor
    valid_mask: torch.Tensor
    loss_mask: torch.Tensor
    quality_per_step: torch.Tensor
    tolerance_per_step: torch.Tensor
    episode_quality: float
    episode_tolerance_min: float
    future_success: bool
    future_success_within_100: bool
    lifted_frac: float
    no_drop: bool
    future_improvement: float
    had_drop: bool
    starts_at_episode_reset: bool
    truncated_prefix: bool
    original_episode_len: int
    stored_seq_len: int
    loss_start_idx: int
    source_policy: int
    source_profile: str
    source_nsucc: Optional[float]
    source_retention: Optional[float]
    source_type: str
    env_frames: int
    done_reason: str
    info_missing: dict[str, bool] = field(default_factory=dict)

    @property
    def num_valid_steps(self) -> int:
        return int(self.valid_mask.sum().item())

    @property
    def num_loss_steps(self) -> int:
        return int(self.loss_mask.sum().item())


def awr_active_from_args(args_cli) -> bool:
    return bool(getattr(args_cli, "awr_enable", False)) and float(
        getattr(args_cli, "awr_loss_weight", 0.0) or 0.0
    ) > 0.0


def pbt_should_keep_local_progress_counters(
    *,
    source_frame: int | None,
    target_frame: int,
    enable_frame_gap_reset: bool,
    frame_gap_reset_threshold: int,
) -> tuple[bool, int]:
    """Return whether to keep local epoch/frame instead of inheriting checkpoint progress.

    ``enable_frame_gap_reset`` enables the large-gap mechanism; it does not force reset
    on every restore. Progress is kept locally only when the frame gap exceeds threshold.
    """
    gap = abs(int(source_frame) - int(target_frame)) if source_frame is not None else 0
    if not enable_frame_gap_reset or source_frame is None:
        return False, gap
    return gap > int(frame_gap_reset_threshold), gap


def build_elite_awr_config_from_args(args_cli) -> EliteAWRConfig:
    batch_steps = getattr(args_cli, "awr_batch_steps", None)
    if batch_steps is None:
        batch_steps = getattr(args_cli, "awr_batch_size", 4096)
    copy_steps = getattr(args_cli, "awr_copy_top_k_steps", None)
    if copy_steps is None:
        copy_steps = getattr(args_cli, "awr_copy_top_k", 50_000)
    broadcast_steps = getattr(args_cli, "awr_broadcast_top_k_steps", None)
    if broadcast_steps is None:
        broadcast_steps = getattr(args_cli, "awr_broadcast_top_k", 10_000)
    collection_mode = str(
        getattr(args_cli, "awr_collection_mode", "rollout_buffer") or "rollout_buffer"
    )
    min_loss_raw = getattr(args_cli, "awr_min_loss_steps_per_entry", None)
    if min_loss_raw is None:
        min_loss_steps = 4 if collection_mode == "rollout_buffer" else 16
    else:
        min_loss_steps = int(min_loss_raw)
    return EliteAWRConfig(
        enable=bool(getattr(args_cli, "awr_enable", False)),
        loss_weight=float(getattr(args_cli, "awr_loss_weight", 0.0) or 0.0),
        start_after_frames=int(getattr(args_cli, "awr_start_after_frames", 800_000_000)),
        ramp_frames=int(getattr(args_cli, "awr_ramp_frames", 300_000_000)),
        min_buffer_size=int(getattr(args_cli, "awr_min_buffer_size", 50_000)),
        buffer_capacity=int(getattr(args_cli, "awr_buffer_capacity", 400_000)),
        batch_steps=int(batch_steps),
        minibatch_max_sequences=int(
            getattr(args_cli, "awr_minibatch_max_sequences", 128)
        ),
        update_every=int(getattr(args_cli, "awr_update_every", 4)),
        max_episode_len=int(getattr(args_cli, "awr_max_episode_len", 200)),
        min_burnin_steps=int(getattr(args_cli, "awr_min_burnin_steps", 32)),
        min_loss_steps_per_entry=min_loss_steps,
        weight_mode=str(getattr(args_cli, "awr_weight_mode", "exp_quality")),
        beta=float(getattr(args_cli, "awr_beta", 1.0)),
        weight_max=float(getattr(args_cli, "awr_weight_max", 5.0)),
        quality_min=float(getattr(args_cli, "awr_quality_min", 1.5)),
        delta_metric=str(getattr(args_cli, "awr_delta_metric", "mean") or "mean"),
        delta_soft=float(getattr(args_cli, "awr_delta_soft", 0.20)),
        delta_mid=float(getattr(args_cli, "awr_delta_mid", 0.45)),
        delta_hard=float(getattr(args_cli, "awr_delta_hard", 0.80)),
        delta_mid_weight=float(getattr(args_cli, "awr_delta_mid_weight", 0.5)),
        delta_far_weight=float(getattr(args_cli, "awr_delta_far_weight", 0.2)),
        loss_cap=float(getattr(args_cli, "awr_loss_cap", 0.03)),
        kl_gate_enable=bool(getattr(args_cli, "awr_kl_gate_enable", False)),
        kl_soft=float(getattr(args_cli, "awr_kl_soft", 0.05)),
        kl_hard=float(getattr(args_cli, "awr_kl_hard", 0.10)),
        ref_update_interval_frames=int(
            getattr(args_cli, "awr_ref_update_interval_frames", 200_000_000)
        ),
        local_sample_frac=float(getattr(args_cli, "awr_local_sample_frac", 0.7)),
        inherited_sample_frac=float(
            getattr(args_cli, "awr_inherited_sample_frac", 0.2)
        ),
        broadcast_sample_frac=float(
            getattr(args_cli, "awr_broadcast_sample_frac", 0.1)
        ),
        copy_top_k_steps=int(copy_steps),
        broadcast_top_k_steps=int(broadcast_steps),
        broadcast_min_source_nsucc=float(
            getattr(args_cli, "awr_broadcast_min_source_nsucc", 35.0)
        ),
        pause_on_regression=bool(getattr(args_cli, "awr_pause_on_regression", True)),
        regression_drop_threshold=float(
            getattr(args_cli, "awr_regression_drop_threshold", 4.0)
        ),
        regression_pause_frames=int(
            getattr(args_cli, "awr_regression_pause_frames", 100_000_000)
        ),
        collection_mode=str(
            getattr(args_cli, "awr_collection_mode", "rollout_buffer") or "rollout_buffer"
        ),
        collect_after_frames=int(
            getattr(args_cli, "awr_collect_after_frames", 600_000_000)
        ),
        collect_every=int(getattr(args_cli, "awr_collect_every", 4)),
        score_temperature=float(getattr(args_cli, "awr_score_temperature", 1.0)),
        entropy_floor_eps=float(getattr(args_cli, "awr_entropy_floor_eps", 0.5)),
        score_beta_quality=float(getattr(args_cli, "awr_score_beta_quality", 1.0)),
        score_beta_similarity=float(getattr(args_cli, "awr_score_beta_similarity", 0.5)),
        score_beta_skill_delta=float(
            getattr(args_cli, "awr_score_beta_skill_delta", 0.25)
        ),
        delta_gate_gamma=float(getattr(args_cli, "awr_delta_gate_gamma", 0.05)),
    )


def compute_elite_awr_quality(
    *,
    future_success_within_100: bool,
    lifted_frac: float,
    no_drop: bool,
    future_improvement: float,
    tolerance_min: float,
    source_nsucc: Optional[float],
    had_drop: bool,
) -> float:
    quality = 1.0
    quality += 2.0 * float(future_success_within_100)
    quality += 1.0 * float(lifted_frac >= 0.5)
    quality += 1.0 * float(no_drop)
    quality += 1.0 * float(future_improvement > 0.0)
    quality += 0.5 * float(tolerance_min <= 0.06)
    quality += 1.0 * float(tolerance_min <= 0.04)
    if source_nsucc is not None:
        quality += 0.5 * float(source_nsucc >= 30.0)
        quality += 1.0 * float(source_nsucc >= 35.0)
    quality -= 2.0 * float(had_drop)
    return float(max(0.0, min(6.0, quality)))


def _quality_sample_weight(quality: torch.Tensor, cfg: EliteAWRConfig) -> torch.Tensor:
    if cfg.weight_mode == "quality":
        return quality.clamp(0.0, cfg.weight_max)
    if cfg.weight_mode == "exp_quality":
        return torch.exp((quality - 2.0) / max(cfg.beta, 1e-6)).clamp(
            0.0, cfg.weight_max
        )
    return quality.clamp(0.0, cfg.weight_max)


def _delta_gate(delta: torch.Tensor, cfg: EliteAWRConfig) -> torch.Tensor:
    w = torch.ones_like(delta)
    w = torch.where(delta <= cfg.delta_soft, w, torch.full_like(w, cfg.delta_mid_weight))
    w = torch.where(delta <= cfg.delta_mid, w, torch.full_like(w, cfg.delta_far_weight))
    w = torch.where(delta <= cfg.delta_hard, w, torch.zeros_like(w))
    return w


_DELTA_METRIC_CODES = {
    "linf": 0.0,
    "mean": 1.0,
    "rms": 2.0,
    "p90": 3.0,
}


def _delta_metric_code(metric: str) -> float:
    return float(_DELTA_METRIC_CODES.get(str(metric or "mean").strip().lower(), 1.0))


def _compute_action_delta(action: torch.Tensor, mu: torch.Tensor, metric: str) -> torch.Tensor:
    diff = (action - mu).abs()
    m = str(metric or "mean").strip().lower()
    if m == "linf":
        return diff.amax(dim=-1)
    if m == "rms":
        return torch.sqrt((diff * diff).mean(dim=-1))
    if m == "p90":
        return torch.quantile(diff, 0.9, dim=-1)
    return diff.mean(dim=-1)


def _zero_rnn_states(agent, num_seqs: int, device: torch.device):
    template = agent.model.get_default_rnn_state()
    if template is None:
        return None
    out = []
    for state in template:
        z = torch.zeros(
            (state.shape[0], num_seqs, state.shape[2]),
            device=device,
            dtype=state.dtype,
        )
        out.append(z)
    return out if len(out) == 1 else tuple(out)


def eval_policy_log_prob_sequence(
    agent,
    obs_tb: torch.Tensor,
    action_tb: torch.Tensor,
    *,
    seq_length: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Recurrent logprob recompute for (T, B, *) sequences.

    Uses same rl_games recurrent logprob convention when sequence starts at
    reset; burn-in approximation otherwise (caller controls loss_mask).
    """
    t_steps, batch = obs_tb.shape[:2]
    device = obs_tb.device
    obs_flat = agent._preproc_obs(obs_tb.reshape(t_steps * batch, -1))
    act_flat = action_tb.reshape(t_steps * batch, -1)
    batch_dict: dict[str, Any] = {
        "is_train": True,
        "prev_actions": act_flat,
        "obs": obs_flat,
        "seq_length": int(seq_length),
    }
    if getattr(agent, "is_rnn", False):
        batch_dict["rnn_states"] = _zero_rnn_states(agent, batch, device)
    with torch.amp.autocast(
        "cuda",
        enabled=getattr(agent, "mixed_precision", False),
        dtype=torch.bfloat16,
    ):
        res = agent.model(batch_dict)
    neglogp = res["prev_neglogp"].reshape(t_steps, batch)
    logp = -neglogp
    mu = res["mus"].reshape(t_steps, batch, -1)
    return logp, mu


def _effective_lambda(cfg: EliteAWRConfig, frames: int) -> float:
    if frames < cfg.start_after_frames:
        return 0.0
    if cfg.ramp_frames <= 0:
        return cfg.loss_weight
    ramp = min(1.0, (frames - cfg.start_after_frames) / float(cfg.ramp_frames))
    return cfg.loss_weight * ramp


def build_sequence_entry_from_episode(
    pending: list[dict[str, Any]],
    *,
    cfg: EliteAWRConfig,
    source_policy: int,
    source_profile: str,
    source_nsucc: Optional[float],
    source_retention: Optional[float],
    source_type: str,
    env_frames: int,
    done_reason: str,
) -> Optional[EliteAWREntry]:
    if not pending:
        return None
    episode_len = len(pending)
    if episode_len <= cfg.max_episode_len:
        segment = pending
        starts_at_reset = True
        truncated_prefix = False
        loss_start_idx = 0
        for i, step in enumerate(segment):
            if step.get("quality_step_pass", False):
                loss_start_idx = i
                break
    else:
        segment = pending[-cfg.max_episode_len :]
        starts_at_reset = False
        truncated_prefix = True
        loss_start_idx = min(cfg.min_burnin_steps, len(segment) - 1)
        for i in range(loss_start_idx, len(segment)):
            if segment[i].get("quality_step_pass", False):
                loss_start_idx = i
                break

    obs = torch.stack([s["obs"] for s in segment], dim=0)
    action = torch.stack([s["action"] for s in segment], dim=0)
    if not torch.isfinite(obs).all() or not torch.isfinite(action).all():
        return None

    stored_seq_len = len(segment)
    valid_mask = torch.ones(stored_seq_len, dtype=torch.bool)
    loss_mask = torch.zeros(stored_seq_len, dtype=torch.bool)
    if truncated_prefix:
        burn_end = min(cfg.min_burnin_steps, stored_seq_len)
        loss_mask[burn_end:] = True
    else:
        loss_mask[loss_start_idx:] = True
    for i, step in enumerate(segment):
        if not step.get("quality_step_pass", True):
            loss_mask[i] = False

    num_loss = int(loss_mask.sum().item())
    if num_loss < cfg.min_loss_steps_per_entry:
        return None

    tol_vals = [float(s.get("tolerance", 0.075)) for s in segment]
    tolerance_per_step = torch.tensor(tol_vals, dtype=torch.float32)
    tolerance_min = float(min(tol_vals)) if tol_vals else 0.075

    future_success_within_100 = any(s.get("future_success_within_100", False) for s in segment)
    future_success = any(s.get("future_success", False) for s in segment)
    lifted_frac = sum(1 for s in segment if s.get("lifted", False)) / max(stored_seq_len, 1)
    no_drop = all(not s.get("drop", False) for s in segment)
    had_drop = any(s.get("drop", False) for s in segment)
    future_improvement = max(float(s.get("future_improvement", 0.0)) for s in segment)

    episode_quality = compute_elite_awr_quality(
        future_success_within_100=future_success_within_100,
        lifted_frac=lifted_frac,
        no_drop=no_drop,
        future_improvement=future_improvement,
        tolerance_min=tolerance_min,
        source_nsucc=source_nsucc,
        had_drop=had_drop,
    )
    if episode_quality < cfg.quality_min:
        return None

    quality_per_step = torch.full((stored_seq_len,), episode_quality, dtype=torch.float32)
    info_missing = pending[0].get("info_missing", {})

    return EliteAWREntry(
        obs=obs.cpu(),
        action=action.cpu(),
        valid_mask=valid_mask,
        loss_mask=loss_mask,
        quality_per_step=quality_per_step,
        tolerance_per_step=tolerance_per_step,
        episode_quality=episode_quality,
        episode_tolerance_min=tolerance_min,
        future_success=future_success,
        future_success_within_100=future_success_within_100,
        lifted_frac=lifted_frac,
        no_drop=no_drop,
        future_improvement=future_improvement,
        had_drop=had_drop,
        starts_at_episode_reset=starts_at_reset,
        truncated_prefix=truncated_prefix,
        original_episode_len=episode_len,
        stored_seq_len=stored_seq_len,
        loss_start_idx=int(loss_start_idx),
        source_policy=source_policy,
        source_profile=source_profile,
        source_nsucc=source_nsucc,
        source_retention=source_retention,
        source_type=source_type,
        env_frames=env_frames,
        done_reason=done_reason,
        info_missing=dict(info_missing),
    )


class EliteAWRBuffer:
    SIDECAR_FORMAT_VERSION = 1

    def __init__(self, cfg: EliteAWRConfig, *, policy_idx: int = 0) -> None:
        self.cfg = cfg
        self.policy_idx = int(policy_idx)
        self.local: list[EliteAWREntry] = []
        self.inherited: list[EliteAWREntry] = []
        self.broadcast: list[EliteAWREntry] = []
        self._total_steps = 0
        self.telemetry: dict[str, float] = {}

    def _pool(self, source_type: str) -> list[EliteAWREntry]:
        if source_type == "inherited":
            return self.inherited
        if source_type == "broadcast":
            return self.broadcast
        return self.local

    def total_steps(self) -> int:
        return self._total_steps

    def num_sequences(self) -> int:
        return len(self.local) + len(self.inherited) + len(self.broadcast)

    def insert(self, entry: EliteAWREntry) -> bool:
        pool = self._pool(entry.source_type)
        steps = entry.num_valid_steps
        if self._total_steps + steps > self.cfg.buffer_capacity:
            self._evict_steps(steps)
        pool.append(entry)
        self._total_steps += steps
        return True

    def _evict_steps(self, incoming_steps: int) -> None:
        while self._total_steps + incoming_steps > self.cfg.buffer_capacity:
            removed = self._pop_lowest_quality()
            if removed is None:
                break
            self._total_steps -= removed.num_valid_steps

    def _pop_lowest_quality(self) -> Optional[EliteAWREntry]:
        best_pool: list[EliteAWREntry] | None = None
        best_idx = -1
        best_q = float("inf")
        for pool in (self.local, self.inherited, self.broadcast):
            for i, e in enumerate(pool):
                if e.episode_quality < best_q:
                    best_q = e.episode_quality
                    best_pool = pool
                    best_idx = i
        if best_pool is None or best_idx < 0:
            return None
        return best_pool.pop(best_idx)

    def sample_batch(self, device: torch.device) -> Optional[dict[str, Any]]:
        cfg = self.cfg
        if self._total_steps < cfg.min_buffer_size:
            return None
        pools = {
            "local": list(self.local),
            "inherited": list(self.inherited),
            "broadcast": list(self.broadcast),
        }
        available = {k: v for k, v in pools.items() if v}
        if not available:
            return None

        t_pad = min(cfg.max_episode_len, max(e.stored_seq_len for e in self.all_entries()))
        k_ep = max(1, cfg.batch_steps // max(t_pad, 1))
        if cfg.minibatch_max_sequences > 0:
            k_ep = min(k_ep, cfg.minibatch_max_sequences)

        targets = {
            "local": int(round(k_ep * cfg.local_sample_frac)),
            "inherited": int(round(k_ep * cfg.inherited_sample_frac)),
            "broadcast": int(round(k_ep * cfg.broadcast_sample_frac)),
        }
        total_target = sum(targets.values())
        if total_target < k_ep:
            targets["local"] += k_ep - total_target
        elif total_target > k_ep:
            targets["local"] = max(0, targets["local"] - (total_target - k_ep))

        picked: list[EliteAWREntry] = []
        for name, count in targets.items():
            pool = available.get(name) or []
            if not pool and available:
                pool = max(available.values(), key=len)
            if not pool:
                continue
            ranked = sorted(pool, key=lambda e: e.episode_quality, reverse=True)
            picked.extend(ranked[: max(count, 0)])

        if not picked:
            return None
        if len(picked) > k_ep:
            picked = sorted(picked, key=lambda e: e.episode_quality, reverse=True)[:k_ep]

        t_pad = min(cfg.max_episode_len, max(e.stored_seq_len for e in picked))
        b = len(picked)
        obs = torch.zeros(t_pad, b, picked[0].obs.shape[-1], device=device)
        action = torch.zeros(t_pad, b, picked[0].action.shape[-1], device=device)
        valid_mask = torch.zeros(t_pad, b, dtype=torch.bool, device=device)
        loss_mask = torch.zeros(t_pad, b, dtype=torch.bool, device=device)
        quality = torch.zeros(t_pad, b, device=device)
        meta = {
            "starts_at_reset_frac": 0.0,
            "truncated_prefix_frac": 0.0,
            "original_episode_len_mean": 0.0,
            "stored_seq_len_mean": 0.0,
            "loss_steps_mean": 0.0,
            "burnin_steps_mean": 0.0,
        }
        starts = truncated = orig_lens = stored_lens = loss_steps = burnin = 0.0
        for j, entry in enumerate(picked):
            slen = entry.stored_seq_len
            obs[:slen, j] = entry.obs.to(device)
            action[:slen, j] = entry.action.to(device)
            valid_mask[:slen, j] = entry.valid_mask.to(device)
            loss_mask[:slen, j] = entry.loss_mask.to(device)
            quality[:slen, j] = entry.quality_per_step.to(device)
            starts += float(entry.starts_at_episode_reset)
            truncated += float(entry.truncated_prefix)
            orig_lens += float(entry.original_episode_len)
            stored_lens += float(entry.stored_seq_len)
            loss_steps += float(entry.num_loss_steps)
            if entry.truncated_prefix:
                burnin += float(min(cfg.min_burnin_steps, entry.stored_seq_len))
        inv = 1.0 / max(b, 1)
        meta.update(
            {
                "starts_at_reset_frac": starts * inv,
                "truncated_prefix_frac": truncated * inv,
                "original_episode_len_mean": orig_lens * inv,
                "stored_seq_len_mean": stored_lens * inv,
                "loss_steps_mean": loss_steps * inv,
                "burnin_steps_mean": burnin * inv,
            }
        )
        valid_count = int(valid_mask.sum().item())
        padded_count = t_pad * b
        return {
            "obs": obs,
            "action": action,
            "valid_mask": valid_mask,
            "loss_mask": loss_mask,
            "quality": quality,
            "seq_length": t_pad,
            "batch_sequences": b,
            "batch_valid_steps": valid_count,
            "batch_padded_steps": padded_count,
            "batch_pad_frac": 1.0 - (valid_count / max(padded_count, 1)),
            **meta,
        }

    def all_entries(self) -> list[EliteAWREntry]:
        return self.local + self.inherited + self.broadcast

    def top_entries_by_steps(self, step_budget: int, *, source_type: str | None = None) -> list[EliteAWREntry]:
        pools: list[EliteAWREntry] = []
        if source_type is None:
            pools = self.all_entries()
        else:
            pools = list(self._pool(source_type))
        ranked = sorted(pools, key=lambda e: e.episode_quality, reverse=True)
        out: list[EliteAWREntry] = []
        steps = 0
        for entry in ranked:
            if steps + entry.num_valid_steps > step_budget:
                continue
            out.append(entry)
            steps += entry.num_valid_steps
            if steps >= step_budget:
                break
        return out

    def top_entries_by_steps_global(
        self,
        step_budget: int,
        *,
        source_types: tuple[str, ...] = ("local", "inherited"),
    ) -> list[EliteAWREntry]:
        candidates: list[EliteAWREntry] = []
        for source_type in source_types:
            candidates.extend(self._pool(source_type))
        ranked = sorted(candidates, key=lambda e: e.episode_quality, reverse=True)
        out: list[EliteAWREntry] = []
        steps = 0
        for entry in ranked:
            if steps + entry.num_valid_steps > step_budget:
                continue
            out.append(entry)
            steps += entry.num_valid_steps
            if steps >= step_budget:
                break
        return out

    def merge_entries(self, entries: list[EliteAWREntry], *, source_type: str) -> int:
        loaded = 0
        for entry in entries:
            e = copy.deepcopy(entry)
            e.source_type = source_type
            self.insert(e)
            loaded += e.num_valid_steps
        return loaded

    def sidecar_payload(self) -> dict[str, Any]:
        """Full-buffer payload for debug only; checkpoint/copy must use top-k."""
        return {
            "format_version": self.SIDECAR_FORMAT_VERSION,
            "policy_idx": self.policy_idx,
            "local": self.local,
            "inherited": self.inherited,
            "broadcast": self.broadcast,
            "total_steps": self._total_steps,
            "meta": {
                "num_sequences": self.num_sequences(),
                "num_steps": self.total_steps(),
                "step_budget": self.total_steps(),
                "selected_from_total_steps": self._total_steps,
                "full_buffer": True,
            },
        }

    def sidecar_payload_topk(
        self,
        *,
        copy_step_budget: int,
        broadcast_step_budget: int = 0,
    ) -> dict[str, Any]:
        copy_selected = self.top_entries_by_steps_global(
            copy_step_budget, source_types=("local", "inherited")
        )
        local = [e for e in copy_selected if e.source_type == "local"]
        inherited = [e for e in copy_selected if e.source_type == "inherited"]
        broadcast: list[EliteAWREntry] = []
        if broadcast_step_budget > 0:
            broadcast = self.top_entries_by_steps_global(
                broadcast_step_budget, source_types=("broadcast",)
            )
        selected = copy_selected + broadcast
        num_steps = sum(e.num_valid_steps for e in selected)
        return {
            "format_version": self.SIDECAR_FORMAT_VERSION,
            "policy_idx": self.policy_idx,
            "local": local,
            "inherited": inherited,
            "broadcast": broadcast,
            "total_steps": num_steps,
            "meta": {
                "num_sequences": len(selected),
                "num_steps": num_steps,
                "step_budget": int(copy_step_budget),
                "broadcast_step_budget": int(broadcast_step_budget),
                "selected_from_total_steps": int(self._total_steps),
                "full_buffer": False,
            },
        }

    def load_sidecar_payload(
        self,
        payload: Mapping[str, Any],
        *,
        target_pool: str = "as_saved",
    ) -> int:
        if target_pool == "as_saved":
            self.local = list(payload.get("local") or [])
            self.inherited = list(payload.get("inherited") or [])
            self.broadcast = list(payload.get("broadcast") or [])
            meta = payload.get("meta") or {}
            self._total_steps = int(payload.get("total_steps") or meta.get("num_steps") or 0)
            if self._total_steps <= 0:
                self._total_steps = sum(e.num_valid_steps for e in self.all_entries())
            return self._total_steps

        entries: list[EliteAWREntry] = []
        for key in ("local", "inherited", "broadcast"):
            entries.extend(list(payload.get(key) or []))
        if not entries:
            return 0
        loaded = self.merge_entries(entries, source_type=target_pool)
        return loaded


def _sidecar_meta_from_payload(
    buffer: EliteAWRBuffer,
    payload: Mapping[str, Any],
    *,
    save_failed: float,
    size_mb: float,
) -> dict[str, Any]:
    meta = dict(payload.get("meta") or {})
    return {
        "num_sequences": int(meta.get("num_sequences", 0)),
        "num_steps": int(meta.get("num_steps", payload.get("total_steps", 0))),
        "source_policy": int(payload.get("policy_idx", buffer.policy_idx)),
        "format_version": int(
            payload.get("format_version", EliteAWRBuffer.SIDECAR_FORMAT_VERSION)
        ),
        "step_budget": int(meta.get("step_budget", 0)),
        "selected_from_total_steps": int(meta.get("selected_from_total_steps", 0)),
        "full_buffer": bool(meta.get("full_buffer", False)),
        "size_mb": float(size_mb),
        "save_failed": float(save_failed),
    }


def save_awr_sidecar(buffer: EliteAWRBuffer, path: Path) -> tuple[bool, dict[str, Any]]:
    """Deprecated alias: saves top-k sidecar, not the full buffer."""
    return save_awr_sidecar_topk(
        buffer,
        path,
        step_budget=buffer.cfg.copy_top_k_steps,
        broadcast_step_budget=0,
    )


def save_awr_sidecar_topk(
    buffer: EliteAWRBuffer,
    path: Path,
    *,
    step_budget: int,
    broadcast_step_budget: int = 0,
) -> tuple[bool, dict[str, Any]]:
    payload = buffer.sidecar_payload_topk(
        copy_step_budget=int(step_budget),
        broadcast_step_budget=int(broadcast_step_budget),
    )
    meta = _sidecar_meta_from_payload(buffer, payload, save_failed=1.0, size_mb=0.0)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        torch.save(payload, tmp)
        tmp.replace(path)
        meta["size_mb"] = path.stat().st_size / (1024 * 1024)
        meta["save_failed"] = 0.0
        return True, meta
    except Exception:
        meta["save_failed"] = 1.0
        return False, meta


def load_awr_sidecar(
    buffer: EliteAWRBuffer,
    path: Path,
    *,
    source_type: str = "local",
) -> int:
    """Load sidecar entries into ``local`` (default) or ``inherited`` pool."""
    if not path.is_file():
        return 0
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        return 0
    if source_type == "inherited":
        return buffer.load_sidecar_payload(payload, target_pool="inherited")
    if source_type == "broadcast":
        return buffer.load_sidecar_payload(payload, target_pool="broadcast")
    return buffer.load_sidecar_payload(payload, target_pool="as_saved")


def resolve_awr_sidecar_path(ref: str, checkpoint_path: str | Path, workspace: str | Path | None = None) -> Path:
    ref_path = Path(ref)
    if ref_path.is_file():
        return ref_path
    ckpt = Path(checkpoint_path)
    candidates: list[Path] = []
    if workspace is not None:
        candidates.append(Path(workspace) / ref)
    for base in (ckpt.parent, ckpt.parent.parent, ckpt.parent.parent.parent):
        candidates.append(base / ref)
        ws_parent = base
        for _ in range(4):
            candidates.append(ws_parent / ref)
            if ws_parent.parent == ws_parent:
                break
            ws_parent = ws_parent.parent
    seen: set[Path] = set()
    for path in candidates:
        if path in seen:
            continue
        seen.add(path)
        if path.is_file():
            return path
    if workspace is not None:
        return Path(workspace) / ref
    return ckpt.parent.parent / ref


def consume_awr_sidecar_from_checkpoint(
    awr_bridge: EliteAWRBridge,
    checkpoint_path: str | Path,
    *,
    workspace: str | Path | None = None,
) -> int:
    """Load source checkpoint sidecar ref into target inherited pool."""
    try:
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except Exception:
        awr_bridge.stats["awr/buffer_inherit_loaded_steps"] = 0.0
        awr_bridge.stats["awr/sidecar_restore_success"] = 0.0
        return 0
    if not isinstance(payload, dict):
        awr_bridge.stats["awr/buffer_inherit_loaded_steps"] = 0.0
        awr_bridge.stats["awr/sidecar_restore_success"] = 0.0
        return 0

    ref = payload.get("awr_buffer_ref")
    meta = payload.get("awr_buffer_ref_meta") or {}
    if not isinstance(ref, str) or not ref:
        awr_bridge.stats["awr/buffer_inherit_loaded_steps"] = 0.0
        awr_bridge.stats["awr/sidecar_restore_success"] = 0.0
        return 0

    sidecar_path = resolve_awr_sidecar_path(ref, checkpoint_path, workspace=workspace)
    loaded = load_awr_sidecar(
        awr_bridge.buffer, sidecar_path, source_type="inherited"
    )
    awr_bridge.stats["awr/buffer_inherit_loaded_steps"] = float(loaded)
    awr_bridge.stats["awr/sidecar_restore_success"] = float(loaded > 0)
    awr_bridge.stats["awr/sidecar_source_policy"] = float(meta.get("source_policy", -1))
    awr_bridge.stats["awr/inherit_enabled"] = 1.0
    awr_bridge.stats["awr/broadcast_enabled"] = 0.0
    awr_bridge.stats["awr/buffer_broadcast_loaded_steps"] = 0.0
    return loaded


def _standardize(x: torch.Tensor) -> torch.Tensor:
    flat = x.detach().reshape(-1)
    if flat.numel() <= 1:
        return torch.zeros_like(x)
    mean = flat.mean()
    std = flat.std(unbiased=False).clamp(min=1e-6)
    return (x - mean) / std


def _apply_entropy_floor(
    weights: torch.Tensor,
    *,
    eps: float,
) -> tuple[torch.Tensor, float]:
    """INV-9: enforce minimum normalized entropy or fall back to uniform."""
    flat = weights.reshape(-1)
    active = flat > 0
    if not bool(active.any()):
        return weights, 0.0
    w = flat[active]
    w = w / w.sum().clamp(min=1e-8)
    entropy = -(w * torch.log(w.clamp(min=1e-12))).sum()
    h_max = math.log(max(int(active.sum().item()), 1))
    h_norm = float((entropy / max(h_max, 1e-8)).item()) if h_max > 0 else 1.0
    if h_norm < float(eps):
        uniform = torch.zeros_like(flat)
        uniform[active] = 1.0 / max(int(active.sum().item()), 1)
        out = uniform.reshape_as(weights)
        return out, 1.0
    return weights, 0.0


def _compute_awr_sample_weights(
    *,
    quality: torch.Tensor,
    delta_w: torch.Tensor,
    loss_mask: torch.Tensor,
    cfg: EliteAWRConfig,
    skill_delta: float = 0.0,
    similarity: torch.Tensor | None = None,
) -> tuple[torch.Tensor, float]:
    region = loss_mask.bool()
    if not bool(region.any()):
        return torch.zeros_like(quality), 0.0
    q = quality[region]
    d = delta_w[region]
    if similarity is None:
        sim = torch.zeros_like(q)
    else:
        sim = similarity[region]
    score = (
        cfg.score_beta_quality * _standardize(q)
        + cfg.score_beta_similarity * _standardize(sim)
        + cfg.score_beta_skill_delta * float(skill_delta)
        + torch.log(d.clamp(min=0.0) + cfg.delta_gate_gamma)
    )
    temp = max(float(cfg.score_temperature), 1e-6)
    w_active = torch.softmax(score / temp, dim=0)
    w = torch.zeros_like(quality)
    w[region] = w_active
    w, uniform_applied = _apply_entropy_floor(
        w * loss_mask.float(), eps=cfg.entropy_floor_eps
    )
    return w, uniform_applied


def _finite_float(x: float, default: float = 0.0) -> float:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return default
    return v if math.isfinite(v) else default


def _tensor_quantile(x: torch.Tensor, q: float) -> float:
    flat = x.detach().reshape(-1)
    if flat.numel() == 0:
        return 0.0
    qv = torch.quantile(flat, q)
    return _finite_float(float(qv.item()))


def _action_flatten_candidates(tensor: torch.Tensor) -> list[tuple[str, torch.Tensor]]:
    act_dim = int(tensor.shape[-1])
    out: list[tuple[str, torch.Tensor]] = []
    if tensor.ndim == 1:
        out.append(("flat", tensor.reshape(-1, 1)))
        return out
    if tensor.ndim == 2:
        out.append(("direct", tensor.reshape(-1, act_dim)))
        return out
    if tensor.ndim == 3:
        out.append(("direct_TNA", tensor.reshape(-1, act_dim)))
        out.append(("transpose_TN_NT", tensor.transpose(0, 1).reshape(-1, act_dim)))
    return out


_ACTION_ORDER_CODES = {
    "direct:direct": 1.0,
    "direct:direct_TNA": 2.0,
    "direct:transpose_TN_NT": 3.0,
    "direct_TNA:direct": 4.0,
    "direct_TNA:direct_TNA": 5.0,
    "direct_TNA:transpose_TN_NT": 6.0,
    "transpose_TN_NT:direct": 7.0,
    "transpose_TN_NT:direct_TNA": 8.0,
    "transpose_TN_NT:transpose_TN_NT": 9.0,
}


def _action_match_order_code(captured_name: str, ppo_name: str) -> float:
    return float(_ACTION_ORDER_CODES.get(f"{captured_name}:{ppo_name}", float(hash(f"{captured_name}:{ppo_name}") % 10_000)))


def _compute_awr_loss_health_stats(
    *,
    logp: torch.Tensor,
    gate_delta: torch.Tensor,
    linf_delta: torch.Tensor,
    delta_mean: torch.Tensor,
    quality: torch.Tensor,
    loss_mask: torch.Tensor,
    delta_w: torch.Tensor,
    w: torch.Tensor,
    cfg: EliteAWRConfig,
) -> dict[str, float]:
    stats: dict[str, float] = {}
    loss_region = loss_mask.bool()
    region_count = float(loss_region.float().sum().item())
    stats["awr/loss_region_steps"] = _finite_float(region_count)
    if region_count <= 0.0:
        stats["awr/active_frac"] = 0.0
        stats["awr/delta_gate_drop_frac"] = 0.0
        stats["awr/delta_gate_full_frac"] = 0.0
        stats["awr/delta_gate_mid_frac"] = 0.0
        stats["awr/delta_gate_far_frac"] = 0.0
        stats["awr/nonzero_weight_frac"] = 0.0
        stats["awr/effective_nonzero_weighted_frac"] = 0.0
        stats["awr/delta_linf_mean"] = 0.0
        stats["awr/delta_linf_p90"] = 0.0
        stats["awr/delta_mean_mean"] = 0.0
        stats["awr/delta_mean_p90"] = 0.0
        stats["awr/delta_gate_mean"] = 0.0
        stats["awr/delta_gate_p90"] = 0.0
        return stats

    active_frac = float((w > 0).float().sum().item()) / max(region_count, 1.0)
    stats["awr/active_frac"] = _finite_float(active_frac)
    stats["awr/logp_mean"] = _finite_float(float(logp[loss_region].mean().item()))
    stats["awr/logp_p10"] = _tensor_quantile(logp[loss_region], 0.10)
    stats["awr/logp_p90"] = _tensor_quantile(logp[loss_region], 0.90)
    stats["awr/delta_linf_mean"] = _finite_float(float(linf_delta[loss_region].mean().item()))
    stats["awr/delta_linf_p90"] = _tensor_quantile(linf_delta[loss_region], 0.90)
    stats["awr/delta_mean_mean"] = _finite_float(float(delta_mean[loss_region].mean().item()))
    stats["awr/delta_mean_p90"] = _tensor_quantile(delta_mean[loss_region], 0.90)
    stats["awr/delta_gate_mean"] = _finite_float(float(gate_delta[loss_region].mean().item()))
    stats["awr/delta_gate_p90"] = _tensor_quantile(gate_delta[loss_region], 0.90)
    stats["awr/quality_mean"] = _finite_float(float(quality[loss_region].mean().item()))
    stats["awr/quality_p90"] = _tensor_quantile(quality[loss_region], 0.90)
    stats["awr/weight_mean"] = _finite_float(float(w[loss_region].mean().item()))
    stats["awr/weight_p90"] = _tensor_quantile(w[loss_region], 0.90)
    stats["awr/nonzero_weight_frac"] = _finite_float(
        float((w[loss_region] > 0).float().sum().item()) / max(region_count, 1.0)
    )
    stats["awr/effective_nonzero_weighted_frac"] = _finite_float(
        float(((delta_w > 0) & (w > 0) & loss_region).float().sum().item())
        / max(region_count, 1.0)
    )

    full = ((gate_delta <= cfg.delta_soft) & loss_region).float().sum() / max(region_count, 1.0)
    mid = (
        (gate_delta > cfg.delta_soft) & (gate_delta <= cfg.delta_mid) & loss_region
    ).float().sum() / max(region_count, 1.0)
    far = (
        (gate_delta > cfg.delta_mid) & (gate_delta <= cfg.delta_hard) & loss_region
    ).float().sum() / max(region_count, 1.0)
    drop = ((delta_w == 0) & loss_region).float().sum() / max(region_count, 1.0)
    stats["awr/delta_gate_full_frac"] = _finite_float(float(full))
    stats["awr/delta_gate_mid_frac"] = _finite_float(float(mid))
    stats["awr/delta_gate_far_frac"] = _finite_float(float(far))
    stats["awr/delta_gate_drop_frac"] = _finite_float(float(drop))
    return stats


def awr_should_run_builder_this_rollout(bridge: "EliteAWRBridge", cfg: EliteAWRConfig) -> bool:
    """Return True when this rollout should run the AWR builder (collect_every gate)."""
    collect_every = max(int(getattr(cfg, "collect_every", 1) or 1), 1)
    stats = getattr(bridge, "stats", None)
    if not isinstance(stats, dict):
        stats = {}
        bridge.stats = stats
    stats["awr/collect_every"] = float(collect_every)
    counter = int(getattr(bridge, "_awr_collect_counter", 0) or 0) + 1
    bridge._awr_collect_counter = counter
    stats["awr/collect_counter"] = float(counter)
    if collect_every <= 1:
        stats["awr/collection_skipped_collect_every"] = 0.0
        return True
    if counter % collect_every != 0:
        stats["awr/collection_skipped_collect_every"] = 1.0
        return False
    stats["awr/collection_skipped_collect_every"] = 0.0
    return True


# v17.2 debug hotfix: runtime fuse for slow rollout-buffer builder (vectorization deferred).
AWR_BUILDER_FUSE_SLOW_MS = 500.0
AWR_BUILDER_FUSE_PAUSE_FRAMES = 100_000_000
AWR_BUILDER_SKIP_REASON_FUSE = 7201.0


def awr_builder_collection_paused(bridge: "EliteAWRBridge", frames: int) -> bool:
    """Return True while rollout-builder collection is paused by fuse."""
    try:
        fuse_until = int(getattr(bridge, "_awr_builder_fuse_until_frame", 0) or 0)
        return int(frames) < fuse_until
    except Exception:
        return False


def awr_record_builder_fuse_skip(bridge: "EliteAWRBridge", frames: int) -> None:
    """Record fuse skip telemetry only; must never interrupt training."""
    del frames
    try:
        stats = getattr(bridge, "stats", None)
        if not isinstance(stats, dict):
            return
        fuse_until = int(getattr(bridge, "_awr_builder_fuse_until_frame", 0) or 0)
        stats["awr/builder_fuse_active"] = 1.0
        stats["awr/builder_skipped"] = 1.0
        stats["awr/builder_skip_reason_code"] = AWR_BUILDER_SKIP_REASON_FUSE
        stats["awr/builder_fuse_until_frame"] = float(fuse_until)
    except Exception:
        return


def awr_apply_builder_run(
    bridge: "EliteAWRBridge",
    frames: int,
    builder_time_ms: float,
    builder_stats: Mapping[str, float],
) -> None:
    """Merge builder stats and arm fuse after consecutive slow builder runs."""
    try:
        stats = getattr(bridge, "stats", None)
        if not isinstance(stats, dict):
            stats = {}
            bridge.stats = stats
        stats.update(dict(builder_stats))
        stats["awr/rollout_builder_time_ms"] = float(builder_time_ms)
        stats["awr/rollout_builder_entries"] = float(
            builder_stats.get("awr/rollout_builder_entries", 0.0)
        )
        stats["awr/rollout_builder_selected_steps"] = float(
            builder_stats.get("awr/rollout_builder_selected_steps", 0.0)
        )
        stats["awr/builder_fuse_active"] = 0.0
        stats["awr/builder_skipped"] = 0.0

        slow_count = int(getattr(bridge, "_awr_builder_slow_count", 0) or 0)
        if float(builder_time_ms) > AWR_BUILDER_FUSE_SLOW_MS:
            slow_count += 1
            bridge._awr_builder_slow_count = slow_count
            if slow_count >= 2:
                bridge._awr_builder_fuse_until_frame = (
                    int(frames) + AWR_BUILDER_FUSE_PAUSE_FRAMES
                )
                bridge._awr_builder_slow_count = 0
                stats["awr/builder_fuse_until_frame"] = float(
                    bridge._awr_builder_fuse_until_frame
                )
        else:
            bridge._awr_builder_slow_count = 0
    except Exception:
        return


class EliteAWRBridge:
    def __init__(
        self,
        *,
        buffer: EliteAWRBuffer,
        cfg: EliteAWRConfig,
        agent_ref: Any = None,
    ) -> None:
        self.buffer = buffer
        self.cfg = cfg
        self.agent_ref = agent_ref
        self.stats: dict[str, float] = {}
        self._update_counter = 0
        self._disabled = False
        self._disable_reason = ""
        self._ref_snapshot: dict[str, torch.Tensor] | None = None
        self._ref_update_frame = 0
        self._pause_until_frame = 0
        self._loss_cap_hits = 0
        self._loss_cap_total = 0
        self._action_checked = False
        self._pending_action_check: list[tuple[torch.Tensor, torch.Tensor]] = []
        self._awr_builder_slow_count = 0
        self._awr_builder_fuse_until_frame = 0
        self._awr_collect_counter = 0
        self.stats["awr/broadcast_enabled"] = 0.0
        self.stats["awr/buffer_broadcast_loaded_steps"] = 0.0
        self.stats["awr/inherit_enabled"] = 1.0

    @property
    def disabled(self) -> bool:
        return self._disabled

    def disable(self, reason: str) -> None:
        self._disabled = True
        self._disable_reason = reason
        self.stats["awr/enabled"] = 0.0
        self.stats["awr/disabled_reason"] = float(hash(reason) % 10_000)

    def record_action_convention_unavailable(self, reason: str = "unavailable") -> None:
        if self._action_checked:
            return
        self.stats["awr/action_convention_checked"] = 0.0
        self.stats["awr/action_convention_skip_reason"] = float(hash(reason) % 10_000)
        self._action_checked = True

    def maybe_check_action_convention(
        self,
        captured: torch.Tensor,
        ppo_actions: torch.Tensor,
        *,
        horizon_length: int | None = None,
        num_actors: int | None = None,
    ) -> None:
        del horizon_length, num_actors
        if self._action_checked or self._disabled:
            return
        captured = captured.detach()
        ppo_actions = ppo_actions.detach()
        device = ppo_actions.device
        captured = captured.to(device=device, dtype=ppo_actions.dtype)

        cap_cands = _action_flatten_candidates(captured)
        ppo_cands = _action_flatten_candidates(ppo_actions)
        if not cap_cands or not ppo_cands:
            self.record_action_convention_unavailable("empty_candidates")
            return

        best: tuple[float, float, str, str, torch.Tensor, torch.Tensor] | None = None
        alt_means: list[float] = []
        for c_name, c_flat in cap_cands:
            for p_name, p_flat in ppo_cands:
                p_flat = p_flat.to(device=device, dtype=ppo_actions.dtype)
                c_try = c_flat.to(device=device, dtype=ppo_actions.dtype)
                if c_try.shape != p_flat.shape:
                    if c_try.numel() != p_flat.numel():
                        continue
                    try:
                        c_try = c_try.reshape_as(p_flat)
                    except RuntimeError:
                        continue
                err = (c_try - p_flat).abs()
                if not torch.isfinite(err).all():
                    continue
                mean_err = float(err.mean().item())
                max_err = float(err.max().item())
                alt_means.append(mean_err)
                if best is None or mean_err < best[0] or (
                    math.isclose(mean_err, best[0]) and max_err < best[1]
                ):
                    best = (mean_err, max_err, c_name, p_name, c_try, p_flat)

        self.stats["awr/action_convention_checked"] = 1.0
        self._action_checked = True
        if best is None:
            self.disable("action_convention_mismatch")
            self.stats["awr/action_convention_mismatch"] = 1.0
            self.stats["awr/action_match_error_mean"] = float("inf")
            self.stats["awr/action_match_error_max"] = float("inf")
            return

        mean_err, max_err, c_name, p_name, best_flat, ppo_flat = best
        self.stats["awr/action_capture_min"] = float(best_flat.min().item())
        self.stats["awr/action_capture_max"] = float(best_flat.max().item())
        self.stats["awr/action_capture_mean_abs"] = float(best_flat.abs().mean().item())
        self.stats["awr/ppo_buffer_action_min"] = float(ppo_flat.min().item())
        self.stats["awr/ppo_buffer_action_max"] = float(ppo_flat.max().item())
        self.stats["awr/action_match_error_mean"] = mean_err
        self.stats["awr/action_match_error_max"] = max_err
        self.stats["awr/action_match_order_code"] = _action_match_order_code(c_name, p_name)
        if alt_means:
            alt = sorted(alt_means)
            self.stats["awr/action_match_error_alt_mean"] = float(
                alt[1] if len(alt) > 1 else alt[0]
            )

        mean_tol = 1e-4
        max_tol = 1e-3
        if mean_err > mean_tol or max_err > max_tol:
            self.disable("action_convention_mismatch")
            self.stats["awr/action_convention_mismatch"] = 1.0

    def _maybe_update_ref(self, frames: int, agent) -> None:
        if self._ref_snapshot is None or (
            frames - self._ref_update_frame >= self.cfg.ref_update_interval_frames
        ):
            self._ref_snapshot = {
                k: v.detach().cpu().clone()
                for k, v in agent.model.state_dict().items()
            }
            self._ref_update_frame = frames

    def compute_aux_loss(self) -> dict[str, torch.Tensor]:
        if self._disabled:
            return {}
        agent = self.agent_ref
        if agent is None:
            return {}
        frames = int(getattr(agent, "frame", 0) or 0)
        self.stats["awr/enabled"] = 1.0
        self.stats["awr/config_enable_flag"] = float(self.cfg.enable)
        self.stats["awr/loss_weight"] = float(self.cfg.loss_weight)
        self.stats["awr/kl_gate_enabled"] = float(self.cfg.kl_gate_enable)
        self.stats["awr/kl_telemetry_only"] = float(not self.cfg.kl_gate_enable)
        self.stats["awr/broadcast_enabled"] = 0.0
        self.stats["awr/buffer_broadcast_loaded_steps"] = 0.0
        if frames < self.cfg.start_after_frames:
            self.stats["awr/skipped_not_started"] = 1.0
            return {}
        if frames < self._pause_until_frame:
            self.stats["awr/skipped_regression_pause"] = 1.0
            return {}
        self._update_counter += 1
        if self._update_counter % max(self.cfg.update_every, 1) != 0:
            self.stats["awr/skipped_update_every"] = 1.0
            return {}
        if self.buffer.total_steps() < self.cfg.min_buffer_size:
            self.stats["awr/skipped_min_buffer"] = 1.0
            return {}

        device = agent.ppo_device
        batch = self.buffer.sample_batch(device)
        if batch is None:
            self.stats["awr/sample_count"] = 0.0
            return {}

        loss, stats = compute_elite_awr_loss(agent, batch, self.cfg, frames=frames, bridge=self)
        self.stats.update(stats)
        if loss is None:
            return {}
        return {"awr_total": loss}


def compute_elite_awr_loss(
    agent,
    batch: dict[str, Any],
    cfg: EliteAWRConfig,
    *,
    frames: int,
    bridge: EliteAWRBridge | None = None,
) -> tuple[Optional[torch.Tensor], dict[str, float]]:
    stats: dict[str, float] = {}
    eff_lambda = _effective_lambda(cfg, frames)
    stats["awr/effective_lambda"] = float(eff_lambda)
    if eff_lambda <= 0.0:
        return None, stats

    obs = batch["obs"]
    action = batch["action"]
    loss_mask = batch["loss_mask"]
    quality = batch["quality"]
    seq_length = int(batch["seq_length"])
    loss_region = loss_mask.bool()
    if not bool(loss_region.any()):
        stats["awr/active_frac"] = 0.0
        stats.update(_compute_awr_loss_health_stats(
            logp=torch.zeros_like(loss_mask, dtype=torch.float32),
            gate_delta=torch.zeros_like(loss_mask, dtype=torch.float32),
            linf_delta=torch.zeros_like(loss_mask, dtype=torch.float32),
            delta_mean=torch.zeros_like(loss_mask, dtype=torch.float32),
            quality=quality,
            loss_mask=loss_mask,
            delta_w=torch.zeros_like(loss_mask, dtype=torch.float32),
            w=torch.zeros_like(loss_mask, dtype=torch.float32),
            cfg=cfg,
        ))
        return None, stats

    logp, mu = eval_policy_log_prob_sequence(
        agent, obs, action, seq_length=seq_length
    )
    delta_linf = (action - mu).abs().amax(dim=-1)
    delta_mean = (action - mu).abs().mean(dim=-1)
    gate_delta = _compute_action_delta(action, mu, cfg.delta_metric)
    delta_w = _delta_gate(gate_delta, cfg)
    similarity = -gate_delta
    _pbt_raw = getattr(agent, "_pbt_skill_delta", None)
    if isinstance(_pbt_raw, (int, float)):
        skill_delta = float(_pbt_raw)
    else:
        skill_delta = 0.0
    skill_delta = max(-1.0, min(1.0, skill_delta))
    w, uniform_applied = _compute_awr_sample_weights(
        quality=quality,
        delta_w=delta_w,
        loss_mask=loss_mask,
        cfg=cfg,
        skill_delta=skill_delta,
        similarity=similarity,
    )
    stats["awr/skill_delta_coupling"] = float(skill_delta)
    stats["awr/entropy_floor_uniform"] = float(uniform_applied)
    stats["awr/delta_metric_code"] = _delta_metric_code(cfg.delta_metric)
    stats["awr/delta_gate_metric_code"] = stats["awr/delta_metric_code"]
    stats["awr/sample_weight_quality_beta"] = float(cfg.score_beta_quality)
    stats["awr/sample_weight_similarity_beta"] = float(cfg.score_beta_similarity)
    sim_enabled = 1.0
    stats["awr/sample_weight_similarity_enabled"] = sim_enabled
    if bool(loss_region.any()):
        sim_region = similarity[loss_region]
        stats["awr/sample_weight_similarity_mean"] = _finite_float(
            float(sim_region.mean().item())
        )
        stats["awr/sample_weight_similarity_p90"] = _tensor_quantile(sim_region, 0.90)
    stats.update(
        _compute_awr_loss_health_stats(
            logp=logp,
            gate_delta=gate_delta,
            linf_delta=delta_linf,
            delta_mean=delta_mean,
            quality=quality,
            loss_mask=loss_mask,
            delta_w=delta_w,
            w=w,
            cfg=cfg,
        )
    )
    w_sum = w.sum().clamp(min=1e-6)
    awr_loss = -(w * logp).sum() / w_sum

    stats["awr/kl_gate_enabled"] = float(cfg.kl_gate_enable)
    stats["awr/kl_telemetry_only"] = float(not cfg.kl_gate_enable)
    kl_scale = 1.0
    if cfg.kl_gate_enable:
        stats["awr/kl_gate_skip_count"] = 0.0
    if bridge is not None:
        bridge._maybe_update_ref(frames, agent)
    weighted = eff_lambda * kl_scale * awr_loss
    capped = weighted.clamp(max=cfg.loss_cap)
    if bridge is not None:
        bridge._loss_cap_total += 1
        if float(capped.item()) >= cfg.loss_cap - 1e-8:
            bridge._loss_cap_hits += 1
        stats["awr/loss_cap_hit"] = float(
            bridge._loss_cap_hits == bridge._loss_cap_total
        )
        stats["awr/loss_cap_hit_frac"] = bridge._loss_cap_hits / max(
            bridge._loss_cap_total, 1
        )

    stats["awr/loss_raw"] = float(awr_loss.detach().item())
    stats["awr/loss_weighted"] = float(capped.detach().item())
    stats["awr/batch_sequences"] = float(batch.get("batch_sequences", 0))
    stats["awr/batch_valid_steps"] = float(batch.get("batch_valid_steps", 0))
    stats["awr/batch_padded_steps"] = float(batch.get("batch_padded_steps", 0))
    stats["awr/batch_pad_frac"] = float(batch.get("batch_pad_frac", 0.0))
    stats["awr/starts_at_reset_frac"] = float(batch.get("starts_at_reset_frac", 0.0))
    stats["awr/truncated_prefix_frac"] = float(batch.get("truncated_prefix_frac", 0.0))
    stats["awr/original_episode_len_mean"] = float(
        batch.get("original_episode_len_mean", 0.0)
    )
    stats["awr/stored_seq_len_mean"] = float(batch.get("stored_seq_len_mean", 0.0))
    stats["awr/loss_steps_mean"] = float(batch.get("loss_steps_mean", 0.0))
    stats["awr/burnin_steps_mean"] = float(batch.get("burnin_steps_mean", 0.0))
    valid = loss_mask.float()
    stats["awr/loss_mask_frac"] = float(valid.mean().item())
    return capped, stats


def patch_model_get_aux_loss_for_awr(model, awr_bridge):
    """Chain Online Elite-AWR aux loss after any existing get_aux_loss hook."""
    old_get_aux_loss = getattr(model, "get_aux_loss", None)

    def patched_get_aux_loss(*args, **kwargs):
        out: dict = {}
        if old_get_aux_loss is not None:
            old = old_get_aux_loss(*args, **kwargs)
            if isinstance(old, dict):
                out.update(old)
            elif old is not None:
                out["prev_aux"] = old
        if awr_bridge is None or awr_bridge.disabled:
            return out if out else None
        awr_out = awr_bridge.compute_aux_loss()
        if isinstance(awr_out, dict):
            out.update(awr_out)
        elif awr_out is not None:
            out["awr_total"] = awr_out
        return out if out else None

    model.get_aux_loss = patched_get_aux_loss
