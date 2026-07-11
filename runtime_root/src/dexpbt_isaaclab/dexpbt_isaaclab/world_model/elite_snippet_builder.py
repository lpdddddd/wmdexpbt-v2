"""Build H-step recovery snippets from ERD target caches."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import torch


@dataclass
class SnippetBatch:
    feat_start: torch.Tensor
    actions: torch.Tensor
    d_hat: torch.Tensor
    success_tol: torch.Tensor
    lifted: torch.Tensor
    picked: torch.Tensor
    dropped: torch.Tensor
    closest_d_hat: torch.Tensor
    success_h: torch.Tensor
    improve_h: torch.Tensor
    drop_h: torch.Tensor
    recovery_label: torch.Tensor
    s0: torch.Tensor
    s1: torch.Tensor
    i0: torch.Tensor
    d0: torch.Tensor
    d1: torch.Tensor
    # Progress-conditioned refinement context (defaults to neutral zeros so old
    # callers / caches without a progress config keep working unchanged).
    normalized_error: Optional[torch.Tensor] = None
    recent_progress: Optional[torch.Tensor] = None
    oscillation: Optional[torch.Tensor] = None
    progress_history_valid: Optional[torch.Tensor] = None
    normalized_improvement: Optional[torch.Tensor] = None


class SnippetTargetBuilder:
    def __init__(self, *, horizon: int = 4, horizons: tuple[int, ...] = (4, 8)) -> None:
        self.horizon = max(1, int(horizon))
        self.horizons = tuple(int(h) for h in horizons if int(h) > 0) or (self.horizon,)

    def build(
        self,
        cache: Any,
        feature_flat: torch.Tensor,
        *,
        picked_flat: Optional[torch.Tensor] = None,
        dropped_flat: Optional[torch.Tensor] = None,
    ) -> SnippetBatch:
        T, N = int(getattr(cache, "T", 0)), int(getattr(cache, "N", 0))
        H = self.horizon
        if T <= 0 or N <= 0:
            return self._empty(feature_flat)
        n_flat = T * N
        # Rollout cache tensors usually live on GPU; bank features are CPU-resident
        # (_eir_extract_rollout_features returns .cpu()). Index everything on the
        # feature device to avoid mixed-device advanced indexing errors.
        dev = feature_flat.device
        actions = cache.actions_flat.reshape(n_flat, -1).to(device=dev)
        A = int(actions.shape[-1])
        d_hat = cache.d_hat_flat.reshape(n_flat).to(device=dev)
        tol = cache.success_tol_flat.reshape(n_flat).to(device=dev)
        lifted = (
            cache.lifted_flat.reshape(n_flat).to(device=dev)
            if cache.lifted_flat is not None
            else torch.ones(n_flat, device=dev)
        )
        picked = (
            picked_flat.reshape(n_flat).to(device=dev)
            if picked_flat is not None
            else torch.zeros(n_flat, device=dev)
        )
        dropped = (
            dropped_flat.reshape(n_flat).to(device=dev)
            if dropped_flat is not None
            else torch.zeros(n_flat, device=dev)
        )
        targets = getattr(cache, "targets_flat", {}) or {}

        def tgt(key: str) -> torch.Tensor:
            t = targets.get(key)
            return torch.zeros(n_flat, device=dev) if t is None else t.reshape(n_flat).to(device=dev)

        h0, h1 = int(self.horizons[0]), int(self.horizons[-1])
        s0, i0, d0 = tgt(f"success_h{h0}"), tgt(f"improve_h{h0}"), tgt(f"drop_h{h0}")
        s1, d1 = tgt(f"success_h{h1}"), tgt(f"drop_h{h1}")
        starts = [n * T + t for n in range(N) for t in range(T) if t + H <= T]
        if not starts:
            return self._empty(feature_flat)
        idx = torch.tensor(starts, dtype=torch.long, device=dev)
        m = int(idx.shape[0])
        act_snip = torch.stack([actions[idx + h] for h in range(H)], dim=1)
        closest = torch.stack([d_hat[idx + h] for h in range(H)], dim=1).min(dim=1).values
        recovery = ((s0[idx] > 0.5) | (s1[idx] > 0.5) | (i0[idx] > 0.5)).float()

        # Progress context: read pre-computed flat fields when the cache carries
        # them (progress config enabled); otherwise fall back to neutral zeros.
        def ctx(name: str) -> torch.Tensor:
            t = getattr(cache, name, None)
            if t is None:
                return torch.zeros(m, device=dev)
            return t.reshape(n_flat).to(device=dev)[idx]

        norm_err = ctx("normalized_error_flat")
        recent_prog = ctx("recent_progress_flat")
        osc = ctx("oscillation_flat")
        hist_valid = ctx("progress_history_valid_flat")
        # Normalized improvement over the snippet horizon: (d_t - min_j d_{t+j}) / eps_t.
        eps = 1e-6
        norm_improve = (d_hat[idx] - closest) / (tol[idx].clamp_min(0.0) + eps).clamp_min(eps)
        return SnippetBatch(
            feat_start=feature_flat[idx], actions=act_snip.clamp(-1.0, 1.0),
            d_hat=d_hat[idx], success_tol=tol[idx], lifted=lifted[idx],
            picked=picked[idx], dropped=dropped[idx], closest_d_hat=closest,
            success_h=s1[idx], improve_h=i0[idx], drop_h=d1[idx], recovery_label=recovery,
            s0=s0[idx], s1=s1[idx], i0=i0[idx], d0=d0[idx], d1=d1[idx],
            normalized_error=norm_err, recent_progress=recent_prog, oscillation=osc,
            progress_history_valid=hist_valid, normalized_improvement=norm_improve,
        )

    @staticmethod
    def _empty(feature_flat: torch.Tensor) -> SnippetBatch:
        z1 = feature_flat[:0]
        z0 = z1.new_zeros(0)
        return SnippetBatch(
            feat_start=z1, actions=z1.new_zeros(0, 1, 1), d_hat=z0, success_tol=z0,
            lifted=z0, picked=z0, dropped=z0, closest_d_hat=z0, success_h=z0,
            improve_h=z0, drop_h=z0, recovery_label=z0, s0=z0, s1=z0, i0=z0, d0=z0, d1=z0,
            normalized_error=z0, recent_progress=z0, oscillation=z0,
            progress_history_valid=z0, normalized_improvement=z0,
        )
