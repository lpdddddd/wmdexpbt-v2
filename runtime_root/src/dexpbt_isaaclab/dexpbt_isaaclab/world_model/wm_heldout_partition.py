"""Stable env-slot held-out partition for WM train vs calibration.

Partition is deterministic across processes and restarts:

    is_heldout(env) <=> stable_u32(seed, worker_rank, env) / 2**32 < fraction

Uses an explicit integer mix (not Python ``hash()``). Does not hash
observations, rewards, or other float content.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import torch

from .wm_head_probs import (
    FAILURE_INSUFFICIENT_HELDOUT_SAMPLES,
    FAILURE_INVALID_PARTITION,
)


def stable_u32(seed: int, worker_rank: int, env_index: int) -> int:
    """Deterministic 32-bit mix of (seed, worker_rank, env_index)."""
    x = (
        (int(seed) & 0xFFFFFFFF)
        ^ ((int(worker_rank) & 0xFFFFFFFF) * 0x9E3779B1 & 0xFFFFFFFF)
        ^ ((int(env_index) & 0xFFFFFFFF) * 0x85EBCA6B & 0xFFFFFFFF)
    ) & 0xFFFFFFFF
    # Finalise avalanche (Murmur-inspired).
    x ^= x >> 16
    x = (x * 0x7FEB352D) & 0xFFFFFFFF
    x ^= x >> 15
    x = (x * 0x846CA68B) & 0xFFFFFFFF
    x ^= x >> 16
    return int(x & 0xFFFFFFFF)


def stable_unit_interval(seed: int, worker_rank: int, env_index: int) -> float:
    """Map (seed, rank, env) to ``[0, 1)``."""
    return float(stable_u32(seed, worker_rank, env_index)) / float(1 << 32)


@dataclass(frozen=True)
class EnvPartitionResult:
    train_mask: torch.Tensor  # (N,) bool
    heldout_mask: torch.Tensor  # (N,) bool
    valid: bool
    diagnostics: dict[str, float]
    failure_reason_id: float = 0.0


def validate_heldout_fraction(fraction: float) -> tuple[bool, str]:
    """Return ``(ok, message)``. ``0`` is allowed (partition disabled)."""
    try:
        f = float(fraction)
    except (TypeError, ValueError):
        return False, "heldout_fraction is not a float"
    if f < 0.0 or f >= 1.0:
        return False, f"heldout_fraction must satisfy 0 <= f < 1, got {f}"
    return True, ""


def build_env_partition_masks(
    num_envs: int,
    *,
    heldout_fraction: float,
    partition_seed: int,
    worker_rank: int,
    device: torch.device | str = "cpu",
) -> EnvPartitionResult:
    """Build mutually exclusive train / held-out env masks.

    When ``heldout_fraction == 0``: all envs are train; heldout is empty but
    ``valid=True`` and ``heldout_enabled=0`` (legacy full-replay mode).

    When ``0 < fraction < 1``: both sides must be non-empty, else invalid.
    """
    device_t = torch.device(device)
    n = int(num_envs)
    diag: dict[str, float] = {
        "wm_cal/heldout_enabled": 0.0,
        "wm_cal/heldout_fraction_configured": float(heldout_fraction),
        "wm_cal/heldout_env_count": 0.0,
        "wm_cal/train_env_count": float(max(n, 0)),
        "wm_cal/heldout_env_fraction_actual": 0.0,
        "wm_cal/partition_overlap_count": 0.0,
        "wm_cal/partition_seed": float(int(partition_seed)),
        "wm_cal/partition_worker_rank": float(int(worker_rank)),
        "wm_cal/partition_num_envs": float(max(n, 0)),
    }
    empty = torch.zeros(max(n, 0), dtype=torch.bool, device=device_t)
    ok_frac, msg = validate_heldout_fraction(heldout_fraction)
    if not ok_frac:
        diag["wm_cal/partition_error"] = 1.0
        msg_acc = 0
        for ch in msg:
            msg_acc = (msg_acc * 131 + ord(ch)) & 0xFFFFFFFF
        diag["wm_cal/partition_error_msg_hash"] = float(
            stable_u32(int(partition_seed), 0, int(msg_acc))
        )
        return EnvPartitionResult(
            train_mask=empty.clone(),
            heldout_mask=empty.clone(),
            valid=False,
            diagnostics=diag,
            failure_reason_id=FAILURE_INVALID_PARTITION,
        )
    if n <= 0:
        diag["wm_cal/partition_error"] = 1.0
        return EnvPartitionResult(
            train_mask=empty,
            heldout_mask=empty.clone(),
            valid=False,
            diagnostics=diag,
            failure_reason_id=FAILURE_INVALID_PARTITION,
        )

    frac = float(heldout_fraction)
    if frac == 0.0:
        train = torch.ones(n, dtype=torch.bool, device=device_t)
        held = torch.zeros(n, dtype=torch.bool, device=device_t)
        diag["wm_cal/train_env_count"] = float(n)
        diag["wm_cal/heldout_env_count"] = 0.0
        diag["wm_cal/partition_overlap_count"] = 0.0
        return EnvPartitionResult(
            train_mask=train,
            heldout_mask=held,
            valid=True,
            diagnostics=diag,
            failure_reason_id=0.0,
        )

    held_list = [
        stable_unit_interval(int(partition_seed), int(worker_rank), i) < frac
        for i in range(n)
    ]
    held = torch.tensor(held_list, dtype=torch.bool, device=device_t)
    train = ~held
    overlap = int((train & held).sum().item())
    n_held = int(held.sum().item())
    n_train = int(train.sum().item())
    diag.update(
        {
            "wm_cal/heldout_enabled": 1.0,
            "wm_cal/heldout_env_count": float(n_held),
            "wm_cal/train_env_count": float(n_train),
            "wm_cal/heldout_env_fraction_actual": float(n_held) / float(max(n, 1)),
            "wm_cal/partition_overlap_count": float(overlap),
        }
    )
    if overlap != 0 or n_held <= 0 or n_train <= 0:
        diag["wm_cal/partition_error"] = 1.0
        return EnvPartitionResult(
            train_mask=train,
            heldout_mask=held,
            valid=False,
            diagnostics=diag,
            failure_reason_id=FAILURE_INVALID_PARTITION,
        )
    return EnvPartitionResult(
        train_mask=train,
        heldout_mask=held,
        valid=True,
        diagnostics=diag,
        failure_reason_id=0.0,
    )


def partition_cfg_from_calibration(
    cal_cfg: Any,
    *,
    worker_rank: int = 0,
) -> dict[str, Any]:
    """Extract partition knobs from ``WMCalibrationConfig`` (or duck-typed)."""
    enabled = bool(getattr(cal_cfg, "enabled", False))
    frac = float(getattr(cal_cfg, "heldout_fraction", 0.0) or 0.0)
    seed = int(getattr(cal_cfg, "heldout_partition_seed", 0) or 0)
    return {
        "calibration_enabled": enabled,
        "heldout_fraction": frac if enabled else 0.0,
        "partition_seed": seed,
        "worker_rank": int(worker_rank),
    }


def masks_for_wm_update(
    num_envs: int,
    *,
    calibration_enabled: bool,
    heldout_fraction: float,
    partition_seed: int,
    worker_rank: int,
    device: torch.device | str = "cpu",
) -> tuple[Optional[torch.Tensor], EnvPartitionResult]:
    """Return train ``env_index_mask`` for WM gradient updates, or ``None``.

    ``None`` means unrestricted sampling (held-out disabled / cal off).
    """
    if not calibration_enabled or float(heldout_fraction) <= 0.0:
        # Build diagnostics-only result with full train.
        part = build_env_partition_masks(
            num_envs,
            heldout_fraction=0.0,
            partition_seed=partition_seed,
            worker_rank=worker_rank,
            device=device,
        )
        return None, part
    part = build_env_partition_masks(
        num_envs,
        heldout_fraction=heldout_fraction,
        partition_seed=partition_seed,
        worker_rank=worker_rank,
        device=device,
    )
    if not part.valid:
        return None, part
    return part.train_mask, part


__all__ = [
    "FAILURE_INVALID_PARTITION",
    "FAILURE_INSUFFICIENT_HELDOUT_SAMPLES",
    "EnvPartitionResult",
    "stable_u32",
    "stable_unit_interval",
    "validate_heldout_fraction",
    "build_env_partition_masks",
    "partition_cfg_from_calibration",
    "masks_for_wm_update",
]
