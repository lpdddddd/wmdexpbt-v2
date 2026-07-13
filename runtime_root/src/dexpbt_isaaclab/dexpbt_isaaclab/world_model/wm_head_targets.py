"""Shared WM head supervision targets (training + calibration).

Single source of truth for horizon success/drop/valid masks and soft-OR
recovery targets. Extracted from the RSSM training path so calibration
cannot drift into a second label definition.
"""

from __future__ import annotations

from typing import Optional

import torch

from .seq_wm_base import SequenceBatch


def build_horizon_valid_mask(batch: SequenceBatch, horizon: int) -> torch.Tensor:
    """Return mask ``(T, B)`` where a horizon-H target is computable.

    For step ``t`` we need the next ``horizon`` steps to stay inside the
    sequence without a done crossing. Matches RSSM training exactly.
    """
    T, B = batch.T, batch.B
    h = max(1, int(horizon))
    not_done = 1.0 - batch.done.float()
    in_seg = batch.valid_mask.float()
    ok = batch.valid_mask.new_zeros(T, B)
    for t in range(T):
        if t + h >= T:
            continue
        window = not_done[t : t + h]
        seg = in_seg[t]
        ok[t] = seg * window.prod(dim=0)
    return ok


def build_horizon_success_target(batch: SequenceBatch, horizon: int) -> torch.Tensor:
    """Float ``(T, B)``: 1.0 if ``success_count`` strictly increases over ``[t, t+H]``.

    Requires ``batch.success_count`` and ``batch.next_success_count``. When either
    is missing, returns all zeros (callers that need fail-closed must check
    ``has_horizon_success_sources``).
    """
    T, B = batch.T, batch.B
    device = batch.device
    h = max(1, int(horizon))
    sc = batch.success_count
    nsc = batch.next_success_count
    if sc is None or nsc is None:
        return torch.zeros(T, B, device=device, dtype=batch.obs.dtype)
    target = torch.zeros(T, B, device=device, dtype=batch.obs.dtype)
    for t in range(T):
        t_end = min(T - 1, t + h - 1)
        inc = (nsc[t_end] - sc[t]).clamp_min(0)
        target[t] = (inc > 0.5).to(target.dtype)
    return target


def build_horizon_drop_target(batch: SequenceBatch, horizon: int) -> torch.Tensor:
    """Float ``(T, B)``: 1.0 if any drop/failure within ``[t, t+H-1]``.

    Uses only ``batch.failure`` / ``batch.dropped``. Does **not** treat done or
    d_hat worsen as drop (those belong to other heads / masks).
    """
    T, B = batch.T, batch.B
    device = batch.device
    h = max(1, int(horizon))
    target = torch.zeros(T, B, device=device, dtype=batch.obs.dtype)
    if batch.failure is None and batch.dropped is None:
        return target
    fail_seq = batch.failure if batch.failure is not None else None
    drop_seq = batch.dropped if batch.dropped is not None else None
    for t in range(T):
        t_end = min(T, t + h)
        window_fail = (
            fail_seq[t:t_end].sum(dim=0)
            if fail_seq is not None
            else torch.zeros(B, device=device, dtype=batch.obs.dtype)
        )
        window_drop = (
            drop_seq[t:t_end].sum(dim=0)
            if drop_seq is not None
            else torch.zeros(B, device=device, dtype=batch.obs.dtype)
        )
        target[t] = ((window_fail + window_drop) > 0.5).to(target.dtype)
    return target


def build_recovery_target_soft_or(
    batch: SequenceBatch,
    *,
    horizon: int,
    d_hat_margin: float,
    d_hat_tol_min_steps: int,
    floor_tol_threshold: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Soft-OR recovery target + mask (matches RSSM recovery head training).

    ``target[t,b] = 1`` if ANY of:
      * success_count strictly increases over ``horizon``
      * d_hat improves by ``d_hat_margin``
      * d_hat <= success_tolerance for >= ``d_hat_tol_min_steps`` steps

    Drop / failure are **not** used to clear recovery positives. Mask is
    horizon-valid AND (when available) ``success_tolerance <= floor_tol_threshold``.
    """
    T, B = batch.T, batch.B
    device = batch.device
    h = max(1, int(horizon))
    target = torch.zeros(T, B, device=device, dtype=batch.obs.dtype)

    success_inc = build_horizon_success_target(batch, h)
    target = torch.maximum(target, success_inc)

    if batch.d_hat is not None:
        for t in range(T):
            t_end = min(T - 1, t + h - 1)
            margin_ok = (batch.d_hat[t] - batch.d_hat[t_end]) >= float(d_hat_margin)
            target[t] = torch.maximum(target[t], margin_ok.to(target.dtype))

    if (
        batch.d_hat is not None
        and batch.success_tolerance_seq is not None
        and int(d_hat_tol_min_steps) > 0
    ):
        for t in range(T):
            t_end = min(T, t + h)
            tol_seg = batch.success_tolerance_seq[t:t_end]
            d_seg = batch.d_hat[t:t_end]
            under_tol = (d_seg <= tol_seg).float().sum(dim=0)
            target[t] = torch.maximum(
                target[t],
                (under_tol >= float(d_hat_tol_min_steps)).to(target.dtype),
            )

    valid_h = build_horizon_valid_mask(batch, h)
    mask = valid_h
    if batch.success_tolerance_seq is not None:
        tol_at_t = batch.success_tolerance_seq
        in_floor = (tol_at_t <= float(floor_tol_threshold)).to(mask.dtype)
        mask = mask * in_floor
    return target, mask


def has_horizon_success_sources(batch: SequenceBatch) -> bool:
    return batch.success_count is not None and batch.next_success_count is not None


def has_horizon_drop_sources(batch: SequenceBatch) -> bool:
    """Drop head is defined when at least one of failure/dropped is present."""
    return batch.failure is not None or batch.dropped is not None


def has_recovery_sources(batch: SequenceBatch) -> bool:
    """Recovery soft-OR needs success counters and/or d_hat."""
    if has_horizon_success_sources(batch):
        return True
    return batch.d_hat is not None
