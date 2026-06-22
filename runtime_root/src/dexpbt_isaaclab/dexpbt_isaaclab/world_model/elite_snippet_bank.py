"""Elite snippet bank for CSER-v0."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import torch

from .elite_recovery_bank import EliteRecoveryBank, EliteRecoveryBankConfig, STAGE_MID

SNIPPET_SCHEMA_VERSION = 1

_SNIPPET_SCALAR_COLUMNS = (
    "d_hat",
    "success_tol",
    "lifted",
    "picked",
    "dropped",
    "closest_d_hat",
    "success_h",
    "improve_h",
    "drop_h",
    "recovery_label",
    "source_policy_idx",
    "source_nsucc_mean",
    "source_nsucc_max",
    "frame",
    "priority_score",
    "bucket_id",
    "horizon",
)


@dataclass
class EliteSnippetBankConfig:
    enabled: bool = False
    bank_size: int = 50_000
    feature_dim: int = 256
    action_dim: int = 23
    horizon: int = 4
    horizons: tuple[int, ...] = (4, 8)
    sample_pool: int = 4096
    knn_topk: int = 32
    tol_threshold: float = 0.02
    hard_tol_threshold: float = 0.012
    near_success_mult: float = 3.0
    min_priority: float = 1.0
    guidance_enable: bool = False
    guidance_bank_tol: float = 0.04
    guidance_mid_near_success_mult: float = 1.5
    guidance_mid_priority_scale: float = 0.5
    recent_fraction: float = 0.2
    top_fraction: float = 0.6
    diversity_fraction: float = 0.2

    def __post_init__(self) -> None:
        self.horizon = max(1, int(self.horizon))
        self.horizons = tuple(int(h) for h in self.horizons if int(h) > 0) or (self.horizon,)
        self.bank_size = max(1, int(self.bank_size))
        self.sample_pool = max(1, int(self.sample_pool))
        self.knn_topk = max(1, int(self.knn_topk))

    @property
    def h_label(self) -> int:
        return int(self.horizon)


class EliteSnippetBank:
    def __init__(self, cfg: EliteSnippetBankConfig) -> None:
        self.cfg = cfg
        self.capacity = int(cfg.bank_size)
        self.H = int(cfg.h_label)
        self.A = int(cfg.action_dim)
        self.F = int(cfg.feature_dim)
        self.feature = torch.zeros(self.capacity, self.F, dtype=torch.float32)
        self.actions = torch.zeros(self.capacity, self.H, self.A, dtype=torch.float32)
        self.scalars = {n: torch.zeros(self.capacity, dtype=torch.float32) for n in _SNIPPET_SCALAR_COLUMNS}
        self.size = 0
        self.add_count = 0
        self.seen_count = 0
        self.retrieve_calls = 0
        self.retrieve_hits = 0
        self.last_update_diagnostics: dict[str, float] = {}
        self._mask_helper = EliteRecoveryBank(
            EliteRecoveryBankConfig(
                enabled=True,
                bank_size=1,
                feature_dim=self.F,
                action_dim=self.A,
                horizons=cfg.horizons,
                tol_threshold=cfg.tol_threshold,
                hard_tol_threshold=cfg.hard_tol_threshold,
                near_success_mult=cfg.near_success_mult,
                min_priority=cfg.min_priority,
                guidance_enable=cfg.guidance_enable,
                guidance_bank_tol=cfg.guidance_bank_tol,
                guidance_mid_near_success_mult=cfg.guidance_mid_near_success_mult,
                guidance_mid_priority_scale=cfg.guidance_mid_priority_scale,
            )
        )

    def update_from_builder_batch(self, batch, *, source_stats=None, frame: int = 0) -> dict[str, float]:
        self.last_update_diagnostics = {}
        if not self.cfg.enabled or batch.feat_start.shape[0] == 0:
            return self.metrics()
        n = int(batch.feat_start.shape[0])
        dev, f32 = torch.device("cpu"), torch.float32
        lifted = batch.lifted.detach().to(dev, f32).reshape(-1)[:n]
        tol = batch.success_tol.detach().to(dev, f32).reshape(-1)[:n]
        d_hat = batch.d_hat.detach().to(dev, f32).reshape(-1)[:n]
        elite = self._mask_helper._elite_mask(
            lifted=lifted, tol=tol, d_hat=d_hat,
            s0=batch.s0[:n].to(dev, f32), s1=batch.s1[:n].to(dev, f32),
            i0=batch.i0[:n].to(dev, f32), d0=batch.d0[:n].to(dev, f32),
        )
        self.seen_count += n
        if not elite.any():
            self.last_update_diagnostics["update_inserted_rows"] = 0.0
            return self.metrics()
        idx = elite.nonzero(as_tuple=False).reshape(-1)
        super_elite = self._mask_helper._super_elite_mask(tol=tol, d_hat=d_hat, s0=batch.s0.to(dev, f32), d1=batch.d1.to(dev, f32))[idx]
        src = source_stats or {}
        src_mean, src_max, src_idx = float(src.get("source_nsucc_mean", 0)), float(src.get("source_nsucc_max", 0)), float(src.get("source_policy_idx", -1))
        ones = torch.ones(idx.shape[0], device=dev, dtype=f32)
        priority = self._mask_helper.compute_priority(
            s0=batch.s0[idx].to(dev, f32), s1=batch.s1[idx].to(dev, f32),
            i0=batch.i0[idx].to(dev, f32), i1=batch.i0[idx].to(dev, f32),
            d0=batch.d0[idx].to(dev, f32), d1=batch.d1[idx].to(dev, f32),
            source_nsucc_mean=ones * src_mean, source_nsucc_max=ones * src_max, super_elite=super_elite,
        )
        if self.cfg.guidance_enable:
            mid = self._mask_helper._row_stage_ids(tol)[idx] == float(STAGE_MID)
            if mid.any():
                priority = torch.where(mid, priority * float(self.cfg.guidance_mid_priority_scale), priority)
        keep = priority >= float(self.cfg.min_priority)
        if not keep.any():
            self.last_update_diagnostics["update_inserted_rows"] = 0.0
            return self.metrics()
        idx, priority = idx[keep], priority[keep]
        m = int(idx.shape[0])
        stage = self._mask_helper._row_stage_ids(tol)[idx]
        bucket_id = stage * 2.0 + ((batch.s0[idx] > 0.5) | (batch.s1[idx] > 0.5)).float()
        new = {
            "feature": batch.feat_start[idx].detach().to(dev, f32),
            "actions": batch.actions[idx].detach().to(dev, f32),
            "d_hat": d_hat[idx], "success_tol": tol[idx], "lifted": lifted[idx],
            "picked": batch.picked[idx].to(dev, f32), "dropped": batch.dropped[idx].to(dev, f32),
            "closest_d_hat": batch.closest_d_hat[idx].to(dev, f32),
            "success_h": batch.success_h[idx].to(dev, f32), "improve_h": batch.improve_h[idx].to(dev, f32),
            "drop_h": batch.drop_h[idx].to(dev, f32), "recovery_label": batch.recovery_label[idx].to(dev, f32),
            "source_policy_idx": ones[:m] * src_idx, "source_nsucc_mean": ones[:m] * src_mean,
            "source_nsucc_max": ones[:m] * src_max, "frame": ones[:m] * float(frame),
            "priority_score": priority, "bucket_id": bucket_id, "horizon": ones[:m] * float(self.H),
        }
        self._insert(new)
        self.add_count += m
        self.last_update_diagnostics["update_inserted_rows"] = float(m)
        return self.metrics()

    @property
    def retrieval_hit_rate(self) -> float:
        return float(self.retrieve_hits) / float(max(self.retrieve_calls, 1))

    def _current_block(self):
        s = self.size
        block = {"feature": self.feature[:s], "actions": self.actions[:s]}
        for n in _SNIPPET_SCALAR_COLUMNS:
            block[n] = self.scalars[n][:s]
        return block

    def _store_block(self, block):
        n = min(int(block["feature"].shape[0]), self.capacity)
        self.feature[:n], self.actions[:n] = block["feature"][:n], block["actions"][:n]
        for name in _SNIPPET_SCALAR_COLUMNS:
            self.scalars[name][:n] = block[name][:n]
        self.size = n

    def _insert(self, new):
        cur = self._current_block()
        combined = {k: torch.cat([cur[k], new[k]], dim=0) for k in new}
        if int(combined["feature"].shape[0]) <= self.capacity:
            self._store_block(combined)
        else:
            keep = self._mask_helper._select_keep_indices(combined, self.capacity)
            self._store_block({k: v[keep] for k, v in combined.items()})

    def retrieve_candidates(self, feat, *, k: int, stage=None):
        B, H, A, k = int(feat.shape[0]), self.H, self.A, max(1, int(k))
        dev = feat.device
        self.retrieve_calls += B
        if self.size == 0:
            return torch.zeros(B, k, H, A, device=dev), torch.zeros(B, k, device=dev), torch.zeros(B, k, device=dev), False
        pool_idx = torch.arange(self.size)
        if stage is not None:
            bucket = self.scalars["bucket_id"][: self.size]
            sel = (torch.div(bucket, 2.0, rounding_mode="floor") == float(stage)).nonzero(as_tuple=False).reshape(-1)
            if sel.numel() > 0:
                pool_idx = sel
        if pool_idx.numel() > self.cfg.sample_pool:
            pool_idx = pool_idx[torch.argsort(self.scalars["priority_score"][pool_idx], descending=True)[: self.cfg.sample_pool]]
        pf = self.feature[pool_idx].to(device=dev, dtype=feat.dtype)
        pa = self.actions[pool_idx].to(device=dev, dtype=feat.dtype)
        pp = self.scalars["priority_score"][pool_idx].to(device=dev, dtype=feat.dtype)
        kk = min(k, int(pool_idx.shape[0]))
        sim = torch.nn.functional.normalize(feat, dim=-1) @ torch.nn.functional.normalize(pf, dim=-1).t()
        top_sim, top_idx = torch.topk(sim, kk, dim=1)
        ga, gp = pa[top_idx], pp[top_idx]
        self.retrieve_hits += int((top_sim > 0).any(dim=1).sum().item())
        if kk < k:
            pad = k - kk
            ga = torch.cat([ga, ga[:, -1:, :, :].expand(B, pad, H, A)], 1)
            top_sim = torch.cat([top_sim, top_sim[:, -1:].expand(B, pad)], 1)
            gp = torch.cat([gp, gp[:, -1:].expand(B, pad)], 1)
        return ga, top_sim, gp, True

    def merge(self, other_state, *, max_entries=None):
        other = self._block_from_state(other_state)
        m = int(other["feature"].shape[0])
        if m == 0:
            return 0
        if max_entries is not None and m > int(max_entries):
            o = torch.argsort(other["priority_score"], descending=True)[: int(max_entries)]
            other = {k: v[o] for k, v in other.items()}
        self._insert(other)
        return int(other["feature"].shape[0])

    def metrics(self):
        out = {"elite_snippet/bank_size": float(self.size), "elite_snippet/inserted_rows": float(self.add_count), "elite_snippet/retrieval_hit_rate": float(self.retrieval_hit_rate)}
        if self.size > 0:
            out["elite_snippet/priority_mean"] = float(self.scalars["priority_score"][: self.size].mean())
        return out

    def top_entries_state(self, n: int):
        s, n = self.size, min(int(n), self.size)
        if n <= 0:
            return {"schema_version": SNIPPET_SCHEMA_VERSION, "size": 0}
        order = torch.argsort(self.scalars["priority_score"][:s], descending=True)[:n]
        st = {"schema_version": SNIPPET_SCHEMA_VERSION, "size": n, "feature": self.feature[:s][order].clone(), "actions": self.actions[:s][order].clone()}
        for name in _SNIPPET_SCALAR_COLUMNS:
            st[name] = self.scalars[name][:s][order].clone()
        return st

    def state_dict(self):
        s = self.size
        st = {"schema_version": SNIPPET_SCHEMA_VERSION, "size": s, "add_count": self.add_count, "seen_count": self.seen_count,
              "retrieve_calls": self.retrieve_calls, "retrieve_hits": self.retrieve_hits,
              "feature": self.feature[:s].clone(), "actions": self.actions[:s].clone()}
        for name in _SNIPPET_SCALAR_COLUMNS:
            st[name] = self.scalars[name][:s].clone()
        return st

    def load_state_dict(self, state):
        if int(state.get("schema_version", 0)) != SNIPPET_SCHEMA_VERSION:
            raise ValueError("snippet bank schema mismatch")
        block = self._block_from_state(state)
        self.add_count = int(state.get("add_count", 0))
        self.seen_count = int(state.get("seen_count", 0))
        self.retrieve_calls = int(state.get("retrieve_calls", 0))
        self.retrieve_hits = int(state.get("retrieve_hits", 0))
        self.size = 0
        if block["feature"].shape[0] > 0:
            self._store_block(block)

    def _block_from_state(self, state):
        n = int(state.get("size", 0))
        if n <= 0:
            empty = torch.zeros(0, self.F)
            return {
                "feature": empty,
                "actions": torch.zeros(0, self.H, self.A),
                **{name: torch.zeros(0) for name in _SNIPPET_SCALAR_COLUMNS},
            }
        block = {"feature": state["feature"].float()[:n], "actions": state["actions"].float()[:n]}
        for name in _SNIPPET_SCALAR_COLUMNS:
            block[name] = state[name].float()[:n]
        return block
