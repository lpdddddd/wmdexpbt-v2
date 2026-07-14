"""Tolerance-aware EliteSnippetBank merge helpers (Experiment B)."""

from __future__ import annotations

import math
from typing import Any

import torch


def log_rel_delta(tol_a: torch.Tensor, tol_b: float, *, eps: float = 1e-8) -> torch.Tensor:
    tb = float(tol_b)
    if not math.isfinite(tb) or tb <= 0.0:
        # fail-closed sentinel: mark all incompatible
        return torch.full_like(tol_a, float("inf"))
    return torch.abs(torch.log((tol_a.float() + eps) / (tb + eps)))


def filter_state_by_tolerance(
    state: dict[str, Any],
    *,
    query_tol: float,
    max_log_gap: float = 0.20,
    eps: float = 1e-8,
) -> tuple[dict[str, Any], dict[str, float]]:
    """Return a filtered bank state keeping only log-relative compatible rows."""
    diag: dict[str, float] = {
        "incoming_count": 0.0,
        "incoming_compatible_count": 0.0,
        "incoming_compatible_frac": 0.0,
        "target_query_tol": float(query_tol) if math.isfinite(float(query_tol)) else float("nan"),
        "broadcast_rejected_incompatible": 0.0,
        "delta_eps_min": float("nan"),
        "delta_eps_med": float("nan"),
        "delta_eps_max": float("nan"),
        "incoming_tol_min": float("nan"),
        "incoming_tol_med": float("nan"),
        "incoming_tol_max": float("nan"),
        "invalid_tol_fail_close": 0.0,
    }
    n = int(state.get("size", 0) or 0)
    diag["incoming_count"] = float(n)
    if n <= 0:
        return {"schema_version": state.get("schema_version", 2), "size": 0}, diag
    if not math.isfinite(float(query_tol)) or float(query_tol) <= 0.0:
        diag["invalid_tol_fail_close"] = 1.0
        diag["broadcast_rejected_incompatible"] = 1.0
        return {"schema_version": state.get("schema_version", 2), "size": 0}, diag

    tol = state.get("success_tol")
    if tol is None:
        diag["invalid_tol_fail_close"] = 1.0
        diag["broadcast_rejected_incompatible"] = 1.0
        return {"schema_version": state.get("schema_version", 2), "size": 0}, diag
    tol_t = tol.float()[:n]
    if not torch.isfinite(tol_t).all() or (tol_t <= 0).any():
        diag["invalid_tol_fail_close"] = 1.0
        diag["broadcast_rejected_incompatible"] = 1.0
        return {"schema_version": state.get("schema_version", 2), "size": 0}, diag

    delta = log_rel_delta(tol_t, float(query_tol), eps=eps)
    diag["incoming_tol_min"] = float(tol_t.min())
    diag["incoming_tol_med"] = float(tol_t.median())
    diag["incoming_tol_max"] = float(tol_t.max())
    diag["delta_eps_min"] = float(delta.min())
    diag["delta_eps_med"] = float(delta.median())
    diag["delta_eps_max"] = float(delta.max())
    mask = delta <= float(max_log_gap)
    c = int(mask.sum().item())
    diag["incoming_compatible_count"] = float(c)
    diag["incoming_compatible_frac"] = float(c) / float(n) if n else 0.0
    if c <= 0:
        diag["broadcast_rejected_incompatible"] = 1.0
        return {"schema_version": state.get("schema_version", 2), "size": 0}, diag

    idx = torch.nonzero(mask, as_tuple=False).reshape(-1)
    out: dict[str, Any] = {
        "schema_version": int(state.get("schema_version", 2)),
        "size": int(c),
    }
    for k, v in state.items():
        if k in ("schema_version", "size", "add_count", "seen_count", "retrieve_calls", "retrieve_hits"):
            continue
        if torch.is_tensor(v):
            out[k] = v[:n][idx].clone()
    return out, diag


def local_count_in_bank(bank: Any, policy_idx: int) -> int:
    if bank is None or int(getattr(bank, "size", 0)) <= 0:
        return 0
    src = bank.scalars["source_policy_idx"][: bank.size]
    return int((src == float(policy_idx)).sum().item())
