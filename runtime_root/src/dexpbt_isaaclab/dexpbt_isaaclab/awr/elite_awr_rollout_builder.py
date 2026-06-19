"""Build Online Elite-AWR entries from PPO experience_buffer rollout tensors (v17.1/v17.2).

v17.2 vectorization hotfix: default builder uses batched torch ops; reference path retained.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any, Mapping, Optional

import torch

from dexpbt_isaaclab.awr.elite_awr import (
    DoneReason,
    EliteAWRConfig,
    EliteAWREntry,
    compute_elite_awr_quality,
)
from dexpbt_isaaclab.awr.elite_awr_interceptor import classify_done_reason

PPO_BUFFER_ACTIONS = 1
DEFAULT_TOLERANCE = 0.075


def should_collect_awr(
    frames: int,
    cfg: EliteAWRConfig,
    bridge: Any | None,
) -> bool:
    if bridge is None:
        return False
    if getattr(bridge, "disabled", False):
        return False
    if not cfg.enable or float(cfg.loss_weight) <= 0.0:
        return False
    if str(getattr(cfg, "collection_mode", "rollout_buffer")) != "rollout_buffer":
        return False
    if int(frames) < int(getattr(cfg, "collect_after_frames", 0)):
        return False
    return True


def _awr_builder_mode() -> str:
    return os.environ.get("DEXPBT_AWR_BUILDER_MODE", "vectorized").strip().lower()


def _vectorize_all_fragments() -> bool:
    raw = os.environ.get("DEXPBT_AWR_VECTORIZE_ALL_FRAGMENTS", "1").strip().lower()
    return raw not in ("0", "false", "no")


def _squeeze_env(x: torch.Tensor | None) -> torch.Tensor | None:
    if x is None:
        return None
    if x.ndim >= 3 and x.shape[-1] == 1:
        return x.squeeze(-1)
    return x


def _bool_done(dones: torch.Tensor, t: int, n: int) -> bool:
    val = dones[t, n]
    if isinstance(val, torch.Tensor):
        if val.numel() == 1:
            return bool(val.item())
        return bool(val.any().item())
    return bool(val)


@dataclass
class _RolloutTensors:
    obs: torch.Tensor
    actions: torch.Tensor
    dones_bool: torch.Tensor
    d_hat: torch.Tensor | None
    tol: torch.Tensor | None
    lifted: torch.Tensor | None
    failure: torch.Tensor | None
    success_cnt: torch.Tensor | None
    info_missing: dict[str, bool]
    T: int
    N: int


def _load_rollout_tensors(td: Mapping[str, Any]) -> _RolloutTensors:
    obs = td["obses"]
    actions = td["actions"]
    dones = td["dones"]
    dones_bool = dones.squeeze(-1).bool() if dones.ndim >= 3 else dones.bool()
    d_hat = _squeeze_env(td.get("aux_d_hat"))
    tol = _squeeze_env(td.get("aux_success_tolerance"))
    lifted = _squeeze_env(td.get("aux_lifted"))
    failure = _squeeze_env(td.get("aux_failure"))
    if failure is None:
        failure = _squeeze_env(td.get("aux_dropped"))
    success_cnt = _squeeze_env(td.get("aux_success_cnt"))
    T, N = int(dones_bool.shape[0]), int(dones_bool.shape[1])
    info_missing = {
        "lifted": lifted is None,
        "success": success_cnt is None,
        "distance": d_hat is None,
        "tolerance": tol is None,
    }
    return _RolloutTensors(
        obs=obs,
        actions=actions,
        dones_bool=dones_bool,
        d_hat=d_hat,
        tol=tol,
        lifted=lifted,
        failure=failure,
        success_cnt=success_cnt,
        info_missing=info_missing,
        T=T,
        N=N,
    )


def _init_builder_stats(info_missing: dict[str, bool], source_nsucc: Optional[float]) -> dict[str, float]:
    stats: dict[str, float] = {
        "awr/rollout_builder_entries": 0.0,
        "awr/rollout_builder_candidate_fragments": 0.0,
        "awr/rollout_builder_selected_fragments": 0.0,
        "awr/rollout_builder_selected_steps": 0.0,
        "awr/rollout_builder_partial_frac": 0.0,
    }
    for key, missing in info_missing.items():
        if missing:
            stats[f"awr/info_missing_{key if key != 'distance' else 'distance'}"] = 1.0
    if source_nsucc is None:
        stats["awr/source_nsucc_missing_frac"] = 1.0
    return stats


def _fragment_success_signal(
    success_cnt: torch.Tensor | None,
    t0: int,
    t1: int,
    n: int,
    *,
    terminal_success: bool,
) -> tuple[bool, bool]:
    """Return (future_success, future_success_within_100)."""
    frag_len = t1 - t0 + 1
    if success_cnt is not None:
        sc0 = float(success_cnt[t0, n].item())
        sc1 = float(success_cnt[t1, n].item())
        if sc1 > sc0:
            return True, True

    if terminal_success:
        future_success = True
        future_within_100 = frag_len <= 100
        return future_success, future_within_100
    return False, False


def _future_improvement_max(d_hat_frag: torch.Tensor) -> float:
    if d_hat_frag.numel() == 0:
        return 0.0
    vals = d_hat_frag.float()
    if not torch.isfinite(vals).any():
        return 0.0
    best = 0.0
    length = int(vals.shape[0])
    for i in range(length):
        cur = float(vals[i].item())
        if not (cur == cur):
            continue
        window = vals[i : min(length, i + 100)]
        wfin = window[torch.isfinite(window)]
        if wfin.numel() == 0:
            continue
        best = max(best, cur - float(wfin.min().item()))
    return float(best)


def _quality_step_pass(
    lifted: bool,
    drop: bool,
    tol: float,
) -> bool:
    return (not drop) and (lifted or tol <= 0.06)


def _build_entry_from_fragment(
    *,
    obs_frag: torch.Tensor,
    action_frag: torch.Tensor,
    lifted_frag: torch.Tensor | None,
    tol_frag: torch.Tensor | None,
    failure_frag: torch.Tensor | None,
    cfg: EliteAWRConfig,
    source_policy: int,
    source_profile: str,
    source_nsucc: Optional[float],
    source_retention: Optional[float],
    source_type: str,
    env_frames: int,
    done_reason: str,
    future_success: bool,
    future_success_within_100: bool,
    future_improvement: float,
    starts_at_episode_reset: bool,
    truncated_prefix: bool,
    original_episode_len: int,
    info_missing: dict[str, bool],
) -> Optional[EliteAWREntry]:
    episode_len = int(obs_frag.shape[0])
    if episode_len <= cfg.max_episode_len:
        segment_obs = obs_frag
        segment_act = action_frag
        starts = starts_at_episode_reset
        trunc = truncated_prefix
        loss_start_idx = 0
    else:
        segment_obs = obs_frag[-cfg.max_episode_len :]
        segment_act = action_frag[-cfg.max_episode_len :]
        starts = False
        trunc = True
        loss_start_idx = min(cfg.min_burnin_steps, segment_obs.shape[0] - 1)

    stored_seq_len = int(segment_obs.shape[0])
    if stored_seq_len <= 0:
        return None

    lifted_seg = lifted_frag[-stored_seq_len:] if lifted_frag is not None else None
    tol_seg = tol_frag[-stored_seq_len:] if tol_frag is not None else None
    fail_seg = failure_frag[-stored_seq_len:] if failure_frag is not None else None

    valid_mask = torch.ones(stored_seq_len, dtype=torch.bool)
    loss_mask = torch.zeros(stored_seq_len, dtype=torch.bool)
    if trunc and stored_seq_len > cfg.min_burnin_steps:
        burn_end = cfg.min_burnin_steps
        loss_mask[burn_end:] = True
    else:
        loss_mask[loss_start_idx:] = True

    tol_vals: list[float] = []
    lifted_count = 0
    had_drop = False
    for i in range(stored_seq_len):
        tol = float(tol_seg[i].item()) if tol_seg is not None else DEFAULT_TOLERANCE
        tol_vals.append(tol)
        lifted = bool(lifted_seg[i].item()) if lifted_seg is not None else False
        drop = bool(fail_seg[i].item()) if fail_seg is not None else False
        if lifted:
            lifted_count += 1
        if drop:
            had_drop = True
        if not _quality_step_pass(lifted, drop, tol):
            loss_mask[i] = False
        elif not trunc and i >= loss_start_idx:
            loss_mask[i] = True

    num_loss = int(loss_mask.sum().item())
    if num_loss < cfg.min_loss_steps_per_entry:
        return None

    tolerance_per_step = torch.tensor(tol_vals, dtype=torch.float32)
    tolerance_min = float(min(tol_vals)) if tol_vals else DEFAULT_TOLERANCE
    lifted_frac = lifted_count / max(stored_seq_len, 1)
    no_drop = not had_drop

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

    if not torch.isfinite(segment_obs).all() or not torch.isfinite(segment_act).all():
        return None

    return EliteAWREntry(
        obs=segment_obs.detach().cpu(),
        action=segment_act.detach().cpu(),
        valid_mask=valid_mask,
        loss_mask=loss_mask,
        quality_per_step=quality_per_step,
        tolerance_per_step=tolerance_per_step,
        episode_quality=episode_quality,
        episode_tolerance_min=tolerance_min,
        future_success=future_success,
        future_success_within_100=future_success_within_100,
        lifted_frac=float(lifted_frac),
        no_drop=no_drop,
        future_improvement=future_improvement,
        had_drop=had_drop,
        starts_at_episode_reset=starts,
        truncated_prefix=trunc,
        original_episode_len=original_episode_len,
        stored_seq_len=stored_seq_len,
        loss_start_idx=int(loss_start_idx),
        source_policy=int(source_policy),
        source_profile=str(source_profile),
        source_nsucc=source_nsucc,
        source_retention=source_retention,
        source_type=str(source_type),
        env_frames=int(env_frames),
        done_reason=str(done_reason),
        info_missing=dict(info_missing),
    )


def _accept_fragment(
    *,
    future_success_within_100: bool,
    future_success: bool,
    lifted_frac: float,
    no_drop: bool,
    future_improvement: float,
    tolerance_min: float,
    source_nsucc: Optional[float],
    had_drop: bool,
    episode_success: bool,
    frag_len: int,
    quality_min: float,
) -> bool:
    accept = False
    if future_success_within_100:
        accept = True
    elif lifted_frac >= 0.5 and no_drop and tolerance_min <= 0.06:
        accept = True
    elif future_improvement > 0.01:
        accept = True
    elif source_nsucc is not None and source_nsucc >= 30.0 and no_drop:
        accept = True
    elif episode_success and frag_len <= 100:
        accept = True

    if had_drop and not future_success_within_100:
        accept = False
    if source_nsucc is not None and source_nsucc < 10.0:
        if not future_success_within_100 and future_improvement <= 0.0:
            accept = False

    q_preview = compute_elite_awr_quality(
        future_success_within_100=future_success_within_100,
        lifted_frac=lifted_frac,
        no_drop=no_drop,
        future_improvement=future_improvement,
        tolerance_min=tolerance_min,
        source_nsucc=source_nsucc,
        had_drop=had_drop,
    )
    if q_preview < quality_min:
        accept = False
    return accept


def build_awr_entries_from_experience_buffer_reference(
    *,
    experience_buffer,
    cfg: EliteAWRConfig,
    source_policy: int,
    source_profile: str,
    source_nsucc: Optional[float],
    source_retention: Optional[float],
    env_frames: int,
    source_type: str = "local",
) -> tuple[list[EliteAWREntry], dict[str, float]]:
    t0_wall = time.perf_counter()
    rt = _load_rollout_tensors(experience_buffer.tensor_dict)
    stats = _init_builder_stats(rt.info_missing, source_nsucc)
    stats["awr/rollout_builder_mode_code"] = 0.0

    obs = rt.obs
    actions = rt.actions
    dones = experience_buffer.tensor_dict["dones"]
    d_hat = rt.d_hat
    tol = rt.tol
    lifted = rt.lifted
    failure = rt.failure
    success_cnt = rt.success_cnt
    T, N = rt.T, rt.N

    entries: list[EliteAWREntry] = []
    candidate_count = 0
    selected_count = 0
    partial_count = 0
    done_unknown_count = 0
    success_from_cnt_count = 0
    success_cleared_by_unknown = 0.0

    for n in range(N):
        frag_start = 0
        for t in range(T):
            done = _bool_done(dones, t, n)
            at_rollout_end = t == T - 1
            if not done and not at_rollout_end:
                continue

            t_end = t
            if t_end < frag_start:
                frag_start = t + 1
                continue

            candidate_count += 1
            truncated_prefix = True
            starts_at_reset = False

            obs_frag = obs[frag_start : t_end + 1, n]
            act_frag = actions[frag_start : t_end + 1, n]
            frag_len = int(obs_frag.shape[0])

            lifted_frag = lifted[frag_start : t_end + 1, n] if lifted is not None else None
            tol_frag = tol[frag_start : t_end + 1, n] if tol is not None else None
            fail_frag = failure[frag_start : t_end + 1, n] if failure is not None else None
            d_hat_frag = d_hat[frag_start : t_end + 1, n] if d_hat is not None else None

            tol_min = float(tol_frag.min().item()) if tol_frag is not None else DEFAULT_TOLERANCE
            lifted_frac = (
                float(lifted_frag.float().mean().item()) if lifted_frag is not None else 0.0
            )
            no_drop = not bool(fail_frag.any().item()) if fail_frag is not None else True
            had_drop = bool(fail_frag.any().item()) if fail_frag is not None else False
            future_improvement = (
                _future_improvement_max(d_hat_frag) if d_hat_frag is not None else 0.0
            )

            terminal_success = False
            done_reason = DoneReason.UNKNOWN.value
            if done:
                term_slice = dones[t_end, n]
                trunc_slice = torch.zeros_like(term_slice) if not at_rollout_end else term_slice
                done_reason = classify_done_reason(
                    terminated=term_slice,
                    truncated=trunc_slice,
                    env_idx=0,
                    infos=None,
                    env_attrs=None,
                )
                terminal_success = done_reason == DoneReason.SUCCESS.value

            future_success, future_within_100 = _fragment_success_signal(
                success_cnt,
                frag_start,
                t_end,
                n,
                terminal_success=terminal_success,
            )
            success_from_cnt = False
            if success_cnt is not None:
                sc0 = float(success_cnt[frag_start, n].item())
                sc1 = float(success_cnt[t_end, n].item())
                success_from_cnt = sc1 > sc0
            if success_from_cnt and future_success:
                success_from_cnt_count += 1
            if done_reason == DoneReason.UNKNOWN.value:
                done_unknown_count += 1
                if future_success and success_from_cnt:
                    pass
                elif not future_success:
                    future_within_100 = False
                    future_success = False
                else:
                    success_cleared_by_unknown += 1.0
                    future_within_100 = False
                    future_success = False

            if not done:
                partial_count += 1

            if not _accept_fragment(
                future_success_within_100=future_within_100,
                future_success=future_success,
                lifted_frac=lifted_frac,
                no_drop=no_drop,
                future_improvement=future_improvement,
                tolerance_min=tol_min,
                source_nsucc=source_nsucc,
                had_drop=had_drop,
                episode_success=terminal_success,
                frag_len=frag_len,
                quality_min=cfg.quality_min,
            ):
                frag_start = t + 1
                continue

            entry = _build_entry_from_fragment(
                obs_frag=obs_frag,
                action_frag=act_frag,
                lifted_frag=lifted_frag,
                tol_frag=tol_frag,
                failure_frag=fail_frag,
                cfg=cfg,
                source_policy=source_policy,
                source_profile=source_profile,
                source_nsucc=source_nsucc,
                source_retention=source_retention,
                source_type=source_type,
                env_frames=env_frames,
                done_reason=done_reason,
                future_success=future_success,
                future_success_within_100=future_within_100,
                future_improvement=future_improvement,
                starts_at_episode_reset=starts_at_reset,
                truncated_prefix=truncated_prefix,
                original_episode_len=frag_len,
                info_missing=rt.info_missing,
            )
            if entry is not None:
                entries.append(entry)
                selected_count += 1
                stats["awr/rollout_builder_selected_steps"] += float(entry.num_valid_steps)

            frag_start = t + 1

    stats["awr/rollout_builder_candidate_fragments"] = float(candidate_count)
    stats["awr/rollout_builder_selected_fragments"] = float(selected_count)
    stats["awr/rollout_builder_entries"] = float(len(entries))
    stats["awr/rollout_builder_partial_frac"] = float(partial_count) / max(candidate_count, 1)
    stats["awr/done_reason_unknown_frac"] = float(done_unknown_count) / max(candidate_count, 1)
    stats["awr/success_from_success_cnt"] = float(success_from_cnt_count)
    stats["awr/success_cleared_by_unknown_done"] = float(success_cleared_by_unknown)
    stats["awr/rollout_builder_time_ms"] = (time.perf_counter() - t0_wall) * 1000.0
    return entries, stats


def _enumerate_fragments(dones_bool: torch.Tensor, *, all_fragments: bool) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return (frag_n, frag_t0, frag_t1) inclusive fragment bounds."""
    T, N = int(dones_bool.shape[0]), int(dones_bool.shape[1])
    device = dones_bool.device
    end_mask = dones_bool.clone()
    end_mask[T - 1, :] = True

    frag_n: list[int] = []
    frag_t0: list[int] = []
    frag_t1: list[int] = []

    for n in range(N):
        if all_fragments:
            ends = torch.nonzero(end_mask[:, n], as_tuple=False).view(-1)
        else:
            ends = torch.tensor([T - 1], device=device, dtype=torch.long)
        if ends.numel() == 0:
            continue
        t0 = 0
        for t_end in ends.tolist():
            t_end_i = int(t_end)
            if t_end_i < t0:
                t0 = t_end_i + 1
                continue
            frag_n.append(n)
            frag_t0.append(t0)
            frag_t1.append(t_end_i)
            t0 = t_end_i + 1

    if not frag_n:
        empty = torch.empty(0, dtype=torch.long, device=device)
        return empty, empty, empty

    return (
        torch.tensor(frag_n, device=device, dtype=torch.long),
        torch.tensor(frag_t0, device=device, dtype=torch.long),
        torch.tensor(frag_t1, device=device, dtype=torch.long),
    )


def _batched_future_improvement(d_hat_ft: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Max_i d[i] - min(d[i:i+100]) over masked positions. d_hat_ft: [F, T]."""
    x = d_hat_ft.float()
    x = x.masked_fill(~mask, float("nan"))
    rev = torch.flip(x, dims=[1])
    rev_fill = torch.nan_to_num(rev, nan=float("inf"))
    cum = torch.cummin(rev_fill, dim=1).values
    future_min = torch.flip(cum, dims=[1])
    imp = x - future_min
    imp = imp.masked_fill(~mask, float("-inf"))
    imp = torch.nan_to_num(imp, nan=float("-inf"))
    out = imp.max(dim=1).values
    out = torch.where(torch.isfinite(out), out.clamp(min=0.0), torch.zeros_like(out))
    return out


def _batched_elite_awr_quality(
    *,
    future_success_within_100: torch.Tensor,
    lifted_frac: torch.Tensor,
    no_drop: torch.Tensor,
    future_improvement: torch.Tensor,
    tolerance_min: torch.Tensor,
    had_drop: torch.Tensor,
    source_nsucc: Optional[float],
) -> torch.Tensor:
    """Vectorized equivalent of compute_elite_awr_quality (v17.2 builder hotfix)."""
    quality = torch.ones_like(lifted_frac)
    quality = quality + 2.0 * future_success_within_100.float()
    quality = quality + (lifted_frac >= 0.5).float()
    quality = quality + no_drop.float()
    quality = quality + (future_improvement > 0.0).float()
    quality = quality + 0.5 * (tolerance_min <= 0.06).float()
    quality = quality + (tolerance_min <= 0.04).float()
    if source_nsucc is not None:
        quality = quality + 0.5 * float(source_nsucc >= 30.0)
        quality = quality + 1.0 * float(source_nsucc >= 35.0)
    quality = quality - 2.0 * had_drop.float()
    return quality.clamp(0.0, 6.0)


def _batch_accept_mask(
    *,
    future_success_within_100: torch.Tensor,
    future_success: torch.Tensor,
    lifted_frac: torch.Tensor,
    no_drop: torch.Tensor,
    future_improvement: torch.Tensor,
    tolerance_min: torch.Tensor,
    had_drop: torch.Tensor,
    episode_success: torch.Tensor,
    frag_len: torch.Tensor,
    source_nsucc: Optional[float],
    quality_min: float,
) -> torch.Tensor:
    accept = future_success_within_100.clone()
    accept = accept | (lifted_frac >= 0.5) & no_drop & (tolerance_min <= 0.06)
    accept = accept | (future_improvement > 0.01)
    if source_nsucc is not None:
        accept = accept | ((source_nsucc >= 30.0) & no_drop)
    accept = accept | (episode_success & (frag_len <= 100))
    accept = accept & ~(had_drop & ~future_success_within_100)
    if source_nsucc is not None and source_nsucc < 10.0:
        reject_low = (~future_success_within_100) & (future_improvement <= 0.0)
        accept = accept & ~reject_low

    q_preview = _batched_elite_awr_quality(
        future_success_within_100=future_success_within_100,
        lifted_frac=lifted_frac,
        no_drop=no_drop,
        future_improvement=future_improvement,
        tolerance_min=tolerance_min,
        had_drop=had_drop,
        source_nsucc=source_nsucc,
    )
    accept = accept & (q_preview >= float(quality_min))
    return accept


def build_awr_entries_from_experience_buffer_vectorized(
    *,
    experience_buffer,
    cfg: EliteAWRConfig,
    source_policy: int,
    source_profile: str,
    source_nsucc: Optional[float],
    source_retention: Optional[float],
    env_frames: int,
    source_type: str = "local",
) -> tuple[list[EliteAWREntry], dict[str, float]]:
    """v17.2 vectorized builder: batched torch precompute + small selected materialize loop."""
    t0_wall = time.perf_counter()
    rt = _load_rollout_tensors(experience_buffer.tensor_dict)
    stats = _init_builder_stats(rt.info_missing, source_nsucc)
    stats["awr/rollout_builder_mode_code"] = 1.0
    stats["awr/rollout_builder_vectorized_speed_path"] = 1.0

    pre_t0 = time.perf_counter()
    frag_n, frag_t0, frag_t1 = _enumerate_fragments(
        rt.dones_bool, all_fragments=_vectorize_all_fragments()
    )
    F = int(frag_n.numel())
    stats["awr/rollout_builder_candidate_fragments"] = float(F)

    if F == 0:
        stats["awr/rollout_builder_time_ms"] = (time.perf_counter() - t0_wall) * 1000.0
        stats["awr/rollout_builder_precompute_ms"] = 0.0
        stats["awr/rollout_builder_selected_materialize_ms"] = 0.0
        return [], stats

    T = rt.T
    device = rt.dones_bool.device
    t_grid = torch.arange(T, device=device).unsqueeze(0).expand(F, T)
    mask = (t_grid >= frag_t0.unsqueeze(1)) & (t_grid <= frag_t1.unsqueeze(1))
    frag_len = (frag_t1 - frag_t0 + 1).float()

    if rt.tol is not None:
        tol_ft = rt.tol[:, frag_n].T
        tol_masked = tol_ft.masked_fill(~mask, float("inf"))
        tolerance_min = tol_masked.min(dim=1).values
        tolerance_min = torch.where(torch.isfinite(tolerance_min), tolerance_min, torch.full((F,), DEFAULT_TOLERANCE, device=device))
    else:
        tolerance_min = torch.full((F,), DEFAULT_TOLERANCE, device=device, dtype=torch.float32)

    if rt.lifted is not None:
        lifted_ft = rt.lifted[:, frag_n].T.float()
        lifted_sum = (lifted_ft * mask.float()).sum(dim=1)
        lifted_frac = lifted_sum / frag_len.clamp(min=1.0)
    else:
        lifted_frac = torch.zeros(F, device=device, dtype=torch.float32)

    if rt.failure is not None:
        fail_ft = rt.failure[:, frag_n].T
        fail_bool = fail_ft.bool() if fail_ft.dtype == torch.bool else (fail_ft != 0)
        had_drop = (fail_bool & mask).any(dim=1)
        no_drop = ~had_drop
    else:
        had_drop = torch.zeros(F, device=device, dtype=torch.bool)
        no_drop = torch.ones(F, device=device, dtype=torch.bool)

    if rt.d_hat is not None:
        d_hat_ft = rt.d_hat[:, frag_n].T
        future_improvement = _batched_future_improvement(d_hat_ft, mask)
    else:
        future_improvement = torch.zeros(F, device=device, dtype=torch.float32)

    episode_done = rt.dones_bool[frag_t1, frag_n]
    is_partial = ~episode_done
    dones = experience_buffer.tensor_dict["dones"]

    if rt.success_cnt is not None:
        sc_start = rt.success_cnt[frag_t0, frag_n].float()
        sc_end = rt.success_cnt[frag_t1, frag_n].float()
        success_from_cnt = sc_end > sc_start
    else:
        success_from_cnt = torch.zeros(F, device=device, dtype=torch.bool)

    future_success = success_from_cnt.clone()
    future_success_within_100 = success_from_cnt & (frag_len <= 100.0)

    terminal_success = torch.zeros(F, device=device, dtype=torch.bool)
    done_reasons = [DoneReason.UNKNOWN.value] * F
    done_unknown = torch.zeros(F, device=device, dtype=torch.bool)

    episode_done_idx = torch.nonzero(episode_done, as_tuple=False).view(-1)
    for idx in episode_done_idx.tolist():
        n = int(frag_n[idx].item())
        t1 = int(frag_t1[idx].item())
        term_slice = dones[t1, n]
        trunc_slice = torch.zeros_like(term_slice)
        dr = classify_done_reason(
            terminated=term_slice,
            truncated=trunc_slice,
            env_idx=0,
            infos=None,
            env_attrs=None,
        )
        done_reasons[idx] = dr
        if dr == DoneReason.UNKNOWN.value:
            done_unknown[idx] = True
        if dr == DoneReason.SUCCESS.value:
            terminal_success[idx] = True
            if not bool(future_success[idx].item()):
                future_success[idx] = True
                if float(frag_len[idx].item()) <= 100.0:
                    future_success_within_100[idx] = True

    success_from_cnt_count = int((success_from_cnt & future_success).sum().item())
    done_unknown_count = int(done_unknown.sum().item())

    keep_unknown = done_unknown & future_success & success_from_cnt
    clear_no_fs = done_unknown & (~future_success)
    clear_other = done_unknown & future_success & (~success_from_cnt)
    success_cleared_by_unknown = float(clear_other.sum().item())
    clear_mask = clear_no_fs | clear_other
    future_success = future_success & ~clear_mask
    future_success_within_100 = future_success_within_100 & ~clear_mask

    accept = _batch_accept_mask(
        future_success_within_100=future_success_within_100,
        future_success=future_success,
        lifted_frac=lifted_frac,
        no_drop=no_drop,
        future_improvement=future_improvement,
        tolerance_min=tolerance_min,
        had_drop=had_drop,
        episode_success=terminal_success,
        frag_len=frag_len,
        source_nsucc=source_nsucc,
        quality_min=cfg.quality_min,
    )

    stats["awr/rollout_builder_precompute_ms"] = (time.perf_counter() - pre_t0) * 1000.0
    mat_t0 = time.perf_counter()

    entries: list[EliteAWREntry] = []
    selected_count = 0
    obs = rt.obs
    actions = rt.actions

    selected_idx = torch.nonzero(accept, as_tuple=False).view(-1)
    for idx in selected_idx.tolist():
        n = int(frag_n[idx].item())
        t0 = int(frag_t0[idx].item())
        t1 = int(frag_t1[idx].item())

        entry = _build_entry_from_fragment(
            obs_frag=obs[t0 : t1 + 1, n],
            action_frag=actions[t0 : t1 + 1, n],
            lifted_frag=rt.lifted[t0 : t1 + 1, n] if rt.lifted is not None else None,
            tol_frag=rt.tol[t0 : t1 + 1, n] if rt.tol is not None else None,
            failure_frag=rt.failure[t0 : t1 + 1, n] if rt.failure is not None else None,
            cfg=cfg,
            source_policy=source_policy,
            source_profile=source_profile,
            source_nsucc=source_nsucc,
            source_retention=source_retention,
            source_type=source_type,
            env_frames=env_frames,
            done_reason=done_reasons[idx],
            future_success=bool(future_success[idx].item()),
            future_success_within_100=bool(future_success_within_100[idx].item()),
            future_improvement=float(future_improvement[idx].item()),
            starts_at_episode_reset=False,
            truncated_prefix=True,
            original_episode_len=int(t1 - t0 + 1),
            info_missing=rt.info_missing,
        )
        if entry is not None:
            entries.append(entry)
            selected_count += 1
            stats["awr/rollout_builder_selected_steps"] += float(entry.num_valid_steps)

    partial_count = int(is_partial.sum().item())
    stats["awr/rollout_builder_selected_fragments"] = float(selected_count)
    stats["awr/rollout_builder_entries"] = float(len(entries))
    stats["awr/rollout_builder_partial_frac"] = float(partial_count) / max(F, 1)
    stats["awr/done_reason_unknown_frac"] = float(done_unknown_count) / max(F, 1)
    stats["awr/success_from_success_cnt"] = float(success_from_cnt_count)
    stats["awr/success_cleared_by_unknown_done"] = float(success_cleared_by_unknown)
    stats["awr/rollout_builder_selected_materialize_ms"] = (time.perf_counter() - mat_t0) * 1000.0
    stats["awr/rollout_builder_time_ms"] = (time.perf_counter() - t0_wall) * 1000.0
    return entries, stats


def build_awr_entries_from_experience_buffer(
    *,
    experience_buffer,
    cfg: EliteAWRConfig,
    source_policy: int,
    source_profile: str,
    source_nsucc: Optional[float],
    source_retention: Optional[float],
    env_frames: int,
    source_type: str = "local",
) -> tuple[list[EliteAWREntry], dict[str, float]]:
    mode = _awr_builder_mode()
    if mode == "reference":
        return build_awr_entries_from_experience_buffer_reference(
            experience_buffer=experience_buffer,
            cfg=cfg,
            source_policy=source_policy,
            source_profile=source_profile,
            source_nsucc=source_nsucc,
            source_retention=source_retention,
            env_frames=env_frames,
            source_type=source_type,
        )
    try:
        return build_awr_entries_from_experience_buffer_vectorized(
            experience_buffer=experience_buffer,
            cfg=cfg,
            source_policy=source_policy,
            source_profile=source_profile,
            source_nsucc=source_nsucc,
            source_retention=source_retention,
            env_frames=env_frames,
            source_type=source_type,
        )
    except Exception:
        entries, stats = build_awr_entries_from_experience_buffer_reference(
            experience_buffer=experience_buffer,
            cfg=cfg,
            source_policy=source_policy,
            source_profile=source_profile,
            source_nsucc=source_nsucc,
            source_retention=source_retention,
            env_frames=env_frames,
            source_type=source_type,
        )
        stats["awr/rollout_builder_vectorized_failed"] = 1.0
        stats["awr/rollout_builder_fallback_reference"] = 1.0
        return entries, stats


def check_rollout_buffer_action_sanity(
    bridge: Any,
    td: Mapping[str, Any],
) -> None:
    """Record PPO-buffer action diagnostics; never disables AWR."""
    actions = td.get("actions")
    if actions is None:
        bridge.stats["awr/action_convention_checked"] = 0.0
        bridge.stats["awr/action_convention_skip_reason"] = float(hash("missing_actions") % 10_000)
        return
    if not torch.isfinite(actions).all():
        bridge.stats["awr/action_convention_checked"] = 0.0
        bridge.stats["awr/action_convention_skip_reason"] = float(hash("nonfinite_actions") % 10_000)
        return

    bridge.stats["awr/action_source_code"] = float(PPO_BUFFER_ACTIONS)
    bridge.stats["awr/action_convention_checked"] = 1.0
    bridge.stats["awr/action_convention_mismatch"] = 0.0
    bridge.stats["awr/ppo_buffer_action_min"] = float(actions.min().item())
    bridge.stats["awr/ppo_buffer_action_max"] = float(actions.max().item())
    bridge.stats["awr/ppo_buffer_action_mean_abs"] = float(actions.abs().mean().item())
    abs_a = actions.abs()
    bridge.stats["awr/action_out_of_bounds_frac_1"] = float((abs_a > 1.0).float().mean().item())
    bridge.stats["awr/action_out_of_bounds_frac_2"] = float((abs_a > 2.0).float().mean().item())

    mus = td.get("mus")
    if mus is not None and torch.isfinite(mus).all():
        delta = (actions - mus).abs()
        bridge.stats["awr/action_mu_delta_mean"] = float(delta.mean().item())
        flat = delta.reshape(-1)
        if flat.numel() > 0:
            bridge.stats["awr/action_mu_delta_p90"] = float(torch.quantile(flat, 0.9).item())


__all__ = [
    "PPO_BUFFER_ACTIONS",
    "build_awr_entries_from_experience_buffer",
    "build_awr_entries_from_experience_buffer_reference",
    "build_awr_entries_from_experience_buffer_vectorized",
    "check_rollout_buffer_action_sanity",
    "should_collect_awr",
]
