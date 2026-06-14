"""Elite Recovery Bank for EIR-MPC.

A tensor-backed memory of real, high-quality recovery transitions harvested from
PPO rollouts in floor / hard-floor near-success states. Each entry stores the
actor feature (or a fallback retrieval key), the env-usable clipped action, the
task-error / tolerance context, the horizon success / improve / drop labels, the
source policy statistics and a scalar priority. The bank supports priority
reservoir insertion, cosine-similarity candidate retrieval, priority-preserving
merge (for PBT broadcast) and ``state_dict`` / ``load_state_dict`` round-trips.

Flatten convention matches ``ERDTargetBuilder``: ``flat[n * T + t] = x[t, n]`` so
that features, actions and targets line up row-for-row.

Storage lives on CPU by default (pinned-friendly); retrieval samples a bounded
pool and runs cosine similarity on that subset so per-call cost stays bounded
regardless of bank size.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import torch


# Candidate / feature source provenance.
FEATURE_SOURCE_ACTOR_MLP = 0
FEATURE_SOURCE_FALLBACK = 1

# Floor-stage bucket ids: stage in {0:none, 1:floor, 2:hard, 3:mid} x success
# flag. STAGE_MID is appended as id 3 (NOT a renumber) so the legacy floor/hard
# bucket ids (2/3 and 4/5 via ``stage*2+is_success``) keep their exact meaning;
# the new mid buckets are 6/7. This preserves old checkpoints' bucket semantics.
STAGE_NONE = 0
STAGE_FLOOR = 1
STAGE_HARD = 2
STAGE_MID = 3


@dataclass
class EliteRecoveryBankConfig:
    enabled: bool = False
    bank_size: int = 500_000
    feature_dim: int = 256
    action_dim: int = 23
    horizons: tuple[int, ...] = (4, 8)

    sample_pool: int = 4096
    use_feature_knn: bool = True
    knn_topk: int = 32

    tol_threshold: float = 0.02
    hard_tol_threshold: float = 0.012
    near_success_mult: float = 3.0
    min_priority: float = 1.0

    # Stage-aware guidance (opt-in). When ``guidance_enable`` is False the bank
    # collects exactly as before (floor-only at ``tol_threshold``). When True it
    # additionally collects a weaker "mid" band ``tol_threshold < tol <=
    # guidance_bank_tol`` with a tighter near-success multiplier and a scaled
    # priority so mid samples never outrank true floor/hard elites.
    guidance_enable: bool = False
    guidance_bank_tol: float = 0.04
    guidance_mid_near_success_mult: float = 1.5
    guidance_mid_priority_scale: float = 0.5

    recent_fraction: float = 0.2
    top_fraction: float = 0.6
    diversity_fraction: float = 0.2

    def __post_init__(self) -> None:
        self.horizons = tuple(int(h) for h in self.horizons if int(h) > 0)
        if not self.horizons:
            self.horizons = (4, 8)
        self.bank_size = max(1, int(self.bank_size))
        self.sample_pool = max(1, int(self.sample_pool))
        self.knn_topk = max(1, int(self.knn_topk))

    @property
    def h0(self) -> int:
        return int(self.horizons[0])

    @property
    def h1(self) -> int:
        return int(self.horizons[-1])


# Scalar (1-D) columns stored per entry, beyond feature[F] and action[A].
_SCALAR_COLUMNS = (
    "d_hat",
    "success_tol",
    "lifted",
    "success_h0",
    "improve_h0",
    "drop_h0",
    "success_h1",
    "improve_h1",
    "drop_h1",
    "source_policy_idx",
    "source_nsucc_mean",
    "source_nsucc_max",
    "frame",
    "priority_score",
    "bucket_id",
)


def _get_target(cache: Any, key: str, n: int, device, dtype) -> torch.Tensor:
    targets = getattr(cache, "targets_flat", {}) or {}
    t = targets.get(key)
    if t is None:
        return torch.zeros(n, device=device, dtype=dtype)
    return t.to(device=device, dtype=dtype).reshape(-1)[:n]


class EliteRecoveryBank:
    """Tensor-backed priority reservoir of elite recovery transitions."""

    def __init__(
        self,
        cfg: EliteRecoveryBankConfig,
        device: torch.device | str = "cpu",
    ) -> None:
        self.cfg = cfg
        self.device = torch.device(device)
        cap = int(cfg.bank_size)
        self.capacity = cap
        self.size = 0
        self.add_count = 0
        self.seen_count = 0
        self.feature = torch.zeros(cap, int(cfg.feature_dim), dtype=torch.float32)
        self.action = torch.zeros(cap, int(cfg.action_dim), dtype=torch.float32)
        self.scalars: dict[str, torch.Tensor] = {
            name: torch.zeros(cap, dtype=torch.float32) for name in _SCALAR_COLUMNS
        }
        # Per-update insertion diagnostics (observation only; does not affect the
        # insert rules). Populated by ``update_from_cache`` and surfaced via
        # ``update_diagnostics()`` so we can see *why* the bank stays empty.
        self.last_update_diagnostics: dict[str, float] = {}
        self._reset_update_diagnostics()

    # ----- diagnostics ----------------------------------------------------
    _UPDATE_DIAG_KEYS = (
        "update_seen_rows",
        "update_risky_rows",
        "update_near_success_rows",
        "update_lifted_rows",
        "update_priority_positive_rows",
        "update_inserted_rows",
        "reject_reason_tol",
        "reject_reason_not_lifted",
        "reject_reason_low_priority",
        "reject_reason_no_feature",
        "reject_reason_no_action",
        # stage-aware (observation only)
        "update_mid_stage_rows",
        "update_floor_stage_rows",
        "update_hard_stage_rows",
        "update_inserted_mid_rows",
        "update_inserted_floor_rows",
        "update_inserted_hard_rows",
        "reject_reason_not_guidance_stage",
    )

    def _reset_update_diagnostics(self) -> None:
        self.last_update_diagnostics = {k: 0.0 for k in self._UPDATE_DIAG_KEYS}

    def update_diagnostics(self) -> dict[str, float]:
        return dict(self.last_update_diagnostics)

    # ----- elite selection ------------------------------------------------
    def _row_stage_ids(self, tol: torch.Tensor) -> torch.Tensor:
        """Per-row stage id with precedence hard > floor > mid > none.

        STAGE_MID is only assigned when ``guidance_enable`` is True; otherwise
        rows above ``tol_threshold`` stay STAGE_NONE (legacy behaviour, so the
        floor-only collection gate is preserved bit-for-bit when guidance off).
        """
        cfg = self.cfg
        stage = torch.full_like(tol, float(STAGE_NONE))
        if bool(cfg.guidance_enable):
            mid = tol <= float(cfg.guidance_bank_tol)
            stage = torch.where(mid, torch.full_like(tol, float(STAGE_MID)), stage)
        floor = tol <= float(cfg.tol_threshold)
        stage = torch.where(floor, torch.full_like(tol, float(STAGE_FLOOR)), stage)
        hard = tol <= float(cfg.hard_tol_threshold)
        stage = torch.where(hard, torch.full_like(tol, float(STAGE_HARD)), stage)
        return stage

    def _near_success_mult_for_stage(self, stage: torch.Tensor) -> torch.Tensor:
        cfg = self.cfg
        mult = torch.full_like(stage, float(cfg.near_success_mult))
        if bool(cfg.guidance_enable):
            mult = torch.where(
                stage == float(STAGE_MID),
                torch.full_like(stage, float(cfg.guidance_mid_near_success_mult)),
                mult,
            )
        return mult

    def _elite_mask(
        self,
        *,
        lifted: torch.Tensor,
        tol: torch.Tensor,
        d_hat: torch.Tensor,
        s0: torch.Tensor,
        s1: torch.Tensor,
        i0: torch.Tensor,
        d0: torch.Tensor,
    ) -> torch.Tensor:
        stage = self._row_stage_ids(tol)
        mult = self._near_success_mult_for_stage(stage)
        cond = lifted > 0.5
        cond = cond & (stage > float(STAGE_NONE))
        cond = cond & (d_hat <= (mult * tol).clamp_min(1e-6))
        cond = cond & ((s0 > 0.5) | (s1 > 0.5) | (i0 > 0.5))
        cond = cond & (d0 <= 0.5)
        return cond

    def _super_elite_mask(
        self,
        *,
        tol: torch.Tensor,
        d_hat: torch.Tensor,
        s0: torch.Tensor,
        d1: torch.Tensor,
    ) -> torch.Tensor:
        cfg = self.cfg
        cond = tol <= float(cfg.hard_tol_threshold)
        cond = cond & (d_hat <= 2.0 * tol)
        cond = cond & (s0 > 0.5)
        cond = cond & (d1 <= 0.5)
        return cond

    @staticmethod
    def compute_priority(
        *,
        s0: torch.Tensor,
        s1: torch.Tensor,
        i0: torch.Tensor,
        i1: torch.Tensor,
        d0: torch.Tensor,
        d1: torch.Tensor,
        source_nsucc_mean: torch.Tensor,
        source_nsucc_max: torch.Tensor,
        super_elite: torch.Tensor,
    ) -> torch.Tensor:
        priority = (
            2.0 * s0
            + 1.2 * s1
            + 0.8 * i0
            + 0.4 * i1
            - 2.0 * d0
            - 1.0 * d1
            + 0.03 * source_nsucc_mean
            + 0.01 * source_nsucc_max
        )
        priority = priority + super_elite.float() * 1.5  # hard_floor_bonus
        return priority

    # ----- update ---------------------------------------------------------
    def update_from_cache(
        self,
        cache: Any,
        feature_flat: torch.Tensor,
        *,
        source_stats: Optional[dict[str, float]] = None,
        frame: int = 0,
    ) -> dict[str, float]:
        """Insert elite rows from a rollout target cache.

        ``feature_flat`` MUST be ``[T*N, feature_dim]`` aligned to the same
        ``flat[n*T+t]`` convention as the cache (actor_mlp features or a fallback
        retrieval key produced by the caller). It is never assumed to come from
        the PPO minibatch capture hook.
        """
        self._reset_update_diagnostics()
        if not self.cfg.enabled:
            return self.metrics()
        actions = getattr(cache, "actions_flat", None)
        d_hat = getattr(cache, "d_hat_flat", None)
        tol = getattr(cache, "success_tol_flat", None)
        if actions is None or d_hat is None or tol is None:
            self.last_update_diagnostics["reject_reason_no_action"] = 1.0
            return self.metrics()
        n = int(actions.shape[0])
        if n == 0:
            return self.metrics()
        if feature_flat is None or int(feature_flat.shape[0]) != n:
            self.last_update_diagnostics["reject_reason_no_feature"] = 1.0
            raise ValueError(
                f"[EIR-bank] feature_flat shape {None if feature_flat is None else tuple(feature_flat.shape)} "
                f"does not match cache rows {n}"
            )
        if int(feature_flat.shape[-1]) != int(self.cfg.feature_dim):
            raise ValueError(
                f"[EIR-bank] feature_dim mismatch feature={int(feature_flat.shape[-1])} "
                f"cfg={int(self.cfg.feature_dim)}"
            )

        dev = torch.device("cpu")
        f32 = torch.float32
        feature_flat = feature_flat.detach().to(device=dev, dtype=f32)
        actions = actions.detach().to(device=dev, dtype=f32)
        d_hat = d_hat.detach().to(device=dev, dtype=f32).reshape(-1)[:n]
        tol = tol.detach().to(device=dev, dtype=f32).reshape(-1)[:n]
        lifted_flat = getattr(cache, "lifted_flat", None)
        if lifted_flat is not None:
            lifted = lifted_flat.detach().to(device=dev, dtype=f32).reshape(-1)[:n]
        else:
            lifted = torch.ones(n, device=dev, dtype=f32)

        h0, h1 = self.cfg.h0, self.cfg.h1
        s0 = _get_target(cache, f"success_h{h0}", n, dev, f32)
        i0 = _get_target(cache, f"improve_h{h0}", n, dev, f32)
        d0 = _get_target(cache, f"drop_h{h0}", n, dev, f32)
        s1 = _get_target(cache, f"success_h{h1}", n, dev, f32)
        i1 = _get_target(cache, f"improve_h{h1}", n, dev, f32)
        d1 = _get_target(cache, f"drop_h{h1}", n, dev, f32)

        # ---- per-row diagnostics (observation only; gate rules unchanged) ----
        lifted_mask = lifted > 0.5
        tol_mask = tol <= float(self.cfg.tol_threshold)
        near_success_mask = d_hat <= (float(self.cfg.near_success_mult) * tol).clamp_min(1e-6)
        diag = self.last_update_diagnostics
        diag["update_seen_rows"] = float(n)
        diag["update_lifted_rows"] = float(lifted_mask.sum().item())
        diag["update_risky_rows"] = float(tol_mask.sum().item())
        diag["update_near_success_rows"] = float(near_success_mask.sum().item())
        diag["reject_reason_not_lifted"] = float((~lifted_mask).sum().item())
        # rows that were lifted but fail the tolerance (risky) staging gate
        diag["reject_reason_tol"] = float((lifted_mask & (~tol_mask)).sum().item())
        # stage-aware row counts (observation only; precedence hard>floor>mid>none)
        stage_all = self._row_stage_ids(tol)
        diag["update_mid_stage_rows"] = float((stage_all == float(STAGE_MID)).sum().item())
        diag["update_floor_stage_rows"] = float((stage_all == float(STAGE_FLOOR)).sum().item())
        diag["update_hard_stage_rows"] = float((stage_all == float(STAGE_HARD)).sum().item())
        diag["reject_reason_not_guidance_stage"] = float(
            (lifted_mask & (stage_all == float(STAGE_NONE))).sum().item()
        )

        elite = self._elite_mask(
            lifted=lifted, tol=tol, d_hat=d_hat, s0=s0, s1=s1, i0=i0, d0=d0
        )
        self.seen_count += n
        if not bool(elite.any()):
            return self.metrics()

        idx = elite.nonzero(as_tuple=False).reshape(-1)
        super_elite = self._super_elite_mask(tol=tol, d_hat=d_hat, s0=s0, d1=d1)[idx]
        src = source_stats or {}
        src_idx = float(src.get("source_policy_idx", -1.0))
        src_mean = float(src.get("source_nsucc_mean", 0.0))
        src_max = float(src.get("source_nsucc_max", 0.0))

        m = int(idx.shape[0])
        ones = torch.ones(m, device=dev, dtype=f32)
        priority = self.compute_priority(
            s0=s0[idx], s1=s1[idx], i0=i0[idx], i1=i1[idx], d0=d0[idx], d1=d1[idx],
            source_nsucc_mean=ones * src_mean,
            source_nsucc_max=ones * src_max,
            super_elite=super_elite,
        )
        # mid-stage samples are weaker guidance: scale their priority down so they
        # never outrank true floor/hard elites (and may drop below min_priority).
        if bool(self.cfg.guidance_enable):
            stage_pi = self._row_stage_ids(tol)[idx]
            mid_pi = stage_pi == float(STAGE_MID)
            if bool(mid_pi.any()):
                priority = torch.where(
                    mid_pi,
                    priority * float(self.cfg.guidance_mid_priority_scale),
                    priority,
                )
        keep = priority >= float(self.cfg.min_priority)
        diag["update_priority_positive_rows"] = float(keep.sum().item())
        diag["reject_reason_low_priority"] = float((~keep).sum().item())
        if not bool(keep.any()):
            return self.metrics()
        idx = idx[keep]
        priority = priority[keep]
        m = int(idx.shape[0])

        # bucket id: stage * 2 + is_success. ``_row_stage_ids`` returns STAGE_MID
        # (=3 -> buckets 6/7) for the mid band, keeping floor/hard buckets intact.
        stage = self._row_stage_ids(tol)[idx]
        is_success = ((s0[idx] > 0.5) | (s1[idx] > 0.5)).float()
        bucket_id = stage * 2.0 + is_success
        diag["update_inserted_mid_rows"] = float((stage == float(STAGE_MID)).sum().item())
        diag["update_inserted_floor_rows"] = float((stage == float(STAGE_FLOOR)).sum().item())
        diag["update_inserted_hard_rows"] = float((stage == float(STAGE_HARD)).sum().item())

        new = {
            "feature": feature_flat[idx],
            "action": actions[idx],
            "d_hat": d_hat[idx],
            "success_tol": tol[idx],
            "lifted": lifted[idx],
            "success_h0": s0[idx],
            "improve_h0": i0[idx],
            "drop_h0": d0[idx],
            "success_h1": s1[idx],
            "improve_h1": i1[idx],
            "drop_h1": d1[idx],
            "source_policy_idx": ones[:m] * src_idx,
            "source_nsucc_mean": ones[:m] * src_mean,
            "source_nsucc_max": ones[:m] * src_max,
            "frame": ones[:m] * float(frame),
            "priority_score": priority,
            "bucket_id": bucket_id,
        }
        self._insert(new)
        self.add_count += m
        self.last_update_diagnostics["update_inserted_rows"] = float(m)
        return self.metrics()

    def _current_block(self) -> dict[str, torch.Tensor]:
        s = self.size
        block = {"feature": self.feature[:s], "action": self.action[:s]}
        for name in _SCALAR_COLUMNS:
            block[name] = self.scalars[name][:s]
        return block

    def _store_block(self, block: dict[str, torch.Tensor]) -> None:
        n = int(block["feature"].shape[0])
        n = min(n, self.capacity)
        self.feature[:n] = block["feature"][:n]
        self.action[:n] = block["action"][:n]
        for name in _SCALAR_COLUMNS:
            self.scalars[name][:n] = block[name][:n]
        self.size = n

    def _insert(self, new: dict[str, torch.Tensor]) -> None:
        cur = self._current_block()
        combined = {
            k: torch.cat([cur[k], new[k]], dim=0) for k in new.keys()
        }
        total = int(combined["feature"].shape[0])
        if total <= self.capacity:
            self._store_block(combined)
            return
        keep_idx = self._select_keep_indices(combined, self.capacity)
        trimmed = {k: v[keep_idx] for k, v in combined.items()}
        self._store_block(trimmed)

    def _select_keep_indices(
        self, block: dict[str, torch.Tensor], cap: int
    ) -> torch.Tensor:
        cfg = self.cfg
        total = int(block["priority_score"].shape[0])
        if total <= cap:
            return torch.arange(total)
        priority = block["priority_score"]
        frame = block["frame"]
        source = block["source_policy_idx"]

        n_top = min(cap, int(cfg.top_fraction * cap))
        n_recent = min(cap - n_top, int(cfg.recent_fraction * cap))
        n_div = cap - n_top - n_recent

        kept = torch.zeros(total, dtype=torch.bool)
        # top by priority
        top_order = torch.argsort(priority, descending=True)
        if n_top > 0:
            kept[top_order[:n_top]] = True
        # recent by frame among not-yet-kept
        if n_recent > 0:
            remaining = (~kept).nonzero(as_tuple=False).reshape(-1)
            rec_order = remaining[torch.argsort(frame[remaining], descending=True)]
            kept[rec_order[:n_recent]] = True
        # diversity: round-robin across source policies among not-yet-kept
        if n_div > 0:
            remaining = (~kept).nonzero(as_tuple=False).reshape(-1)
            if remaining.numel() > 0:
                div_sel = self._round_robin_by_source(
                    remaining, source[remaining], priority[remaining], n_div
                )
                kept[div_sel] = True
        # safeguard fill by priority if rounding left us short
        deficit = cap - int(kept.sum().item())
        if deficit > 0:
            remaining = (~kept).nonzero(as_tuple=False).reshape(-1)
            fill_order = remaining[torch.argsort(priority[remaining], descending=True)]
            kept[fill_order[:deficit]] = True
        keep_idx = kept.nonzero(as_tuple=False).reshape(-1)
        # if we somehow overshoot, trim by priority
        if int(keep_idx.shape[0]) > cap:
            order = torch.argsort(priority[keep_idx], descending=True)
            keep_idx = keep_idx[order[:cap]]
        return keep_idx

    @staticmethod
    def _round_robin_by_source(
        indices: torch.Tensor,
        source: torch.Tensor,
        priority: torch.Tensor,
        n_pick: int,
    ) -> torch.Tensor:
        if n_pick <= 0 or indices.numel() == 0:
            return indices[:0]
        order = torch.argsort(priority, descending=True)
        indices = indices[order]
        source = source[order]
        buckets: dict[float, list[int]] = {}
        for local_pos in range(int(indices.shape[0])):
            key = float(source[local_pos].item())
            buckets.setdefault(key, []).append(int(indices[local_pos].item()))
        picked: list[int] = []
        keys = list(buckets.keys())
        ptr = {k: 0 for k in keys}
        while len(picked) < n_pick:
            progressed = False
            for k in keys:
                if ptr[k] < len(buckets[k]):
                    picked.append(buckets[k][ptr[k]])
                    ptr[k] += 1
                    progressed = True
                    if len(picked) >= n_pick:
                        break
            if not progressed:
                break
        return torch.tensor(picked, dtype=torch.long)

    # ----- retrieval ------------------------------------------------------
    def retrieve_candidates(
        self,
        feat: torch.Tensor,
        *,
        k: int,
        stage: Optional[int] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, bool]:
        """Return ``(actions[B,k,A], sim[B,k], priority[B,k], valid)``.

        Samples up to ``sample_pool`` top-priority entries (optionally restricted
        to a floor-stage ``stage``), then cosine-similarity top-k per query row.
        Falls back to ``valid=False`` zero tensors when the bank is empty.
        """
        B = int(feat.shape[0])
        A = int(self.cfg.action_dim)
        k = max(1, int(k))
        dev = feat.device
        if self.size == 0:
            return (
                torch.zeros(B, k, A, device=dev, dtype=feat.dtype),
                torch.zeros(B, k, device=dev, dtype=feat.dtype),
                torch.zeros(B, k, device=dev, dtype=feat.dtype),
                False,
            )
        pool_idx = torch.arange(self.size)
        if stage is not None:
            bucket = self.scalars["bucket_id"][: self.size]
            stage_of = torch.div(bucket, 2.0, rounding_mode="floor")
            sel = (stage_of == float(stage)).nonzero(as_tuple=False).reshape(-1)
            if sel.numel() > 0:
                pool_idx = sel
        # cap pool to sample_pool by priority
        if int(pool_idx.shape[0]) > int(self.cfg.sample_pool):
            pr = self.scalars["priority_score"][pool_idx]
            top = torch.argsort(pr, descending=True)[: int(self.cfg.sample_pool)]
            pool_idx = pool_idx[top]
        pool_feat = self.feature[pool_idx].to(device=dev, dtype=feat.dtype)
        pool_act = self.action[pool_idx].to(device=dev, dtype=feat.dtype)
        pool_pri = self.scalars["priority_score"][pool_idx].to(device=dev, dtype=feat.dtype)
        P = int(pool_idx.shape[0])
        kk = min(k, P)

        fn = torch.nn.functional.normalize(feat, dim=-1, eps=1e-6)
        pn = torch.nn.functional.normalize(pool_feat, dim=-1, eps=1e-6)
        sim = fn @ pn.t()  # [B, P]
        top_sim, top_idx = torch.topk(sim, kk, dim=1)
        gathered_act = pool_act[top_idx]  # [B, kk, A]
        gathered_pri = pool_pri[top_idx]  # [B, kk]

        if kk < k:
            pad = k - kk
            gathered_act = torch.cat(
                [gathered_act, gathered_act[:, -1:, :].expand(B, pad, A)], dim=1
            )
            top_sim = torch.cat([top_sim, top_sim[:, -1:].expand(B, pad)], dim=1)
            gathered_pri = torch.cat(
                [gathered_pri, gathered_pri[:, -1:].expand(B, pad)], dim=1
            )
        return gathered_act, top_sim, gathered_pri, True

    # ----- merge / broadcast ---------------------------------------------
    def merge(self, other_state: dict[str, Any], *, max_entries: Optional[int] = None) -> int:
        """Merge entries from another bank state dict, priority-preserving.

        Returns the number of entries actually folded in (capped by
        ``max_entries`` and bank capacity, keeping highest priority).
        """
        other = self._block_from_state(other_state)
        m = int(other["feature"].shape[0])
        if m == 0:
            return 0
        if max_entries is not None and m > int(max_entries):
            order = torch.argsort(other["priority_score"], descending=True)[: int(max_entries)]
            other = {key: val[order] for key, val in other.items()}
            m = int(other["feature"].shape[0])
        self._insert(other)
        return m

    # ----- metrics --------------------------------------------------------
    def metrics(self) -> dict[str, float]:
        s = self.size
        out: dict[str, float] = {
            "elite_bank_size": float(s),
            "elite_bank_add_count": float(self.add_count),
            "elite_acceptance_rate": float(self.add_count / max(self.seen_count, 1)),
            "elite_priority_mean": 0.0,
            "elite_priority_max": 0.0,
        }
        if s > 0:
            pr = self.scalars["priority_score"][:s]
            out["elite_priority_mean"] = float(pr.mean().item())
            out["elite_priority_max"] = float(pr.max().item())
            bucket = self.scalars["bucket_id"][:s]
            # buckets 0..7: stage(0=none,1=floor,2=hard,3=mid) * 2 + is_success
            for b in range(8):
                out[f"elite_bucket_{b}"] = float((bucket == float(b)).sum().item())
        return out

    def per_policy_contribution(self) -> dict[int, int]:
        s = self.size
        if s == 0:
            return {}
        src = self.scalars["source_policy_idx"][:s]
        out: dict[int, int] = {}
        for v in src.unique().tolist():
            out[int(v)] = int((src == v).sum().item())
        return out

    # ----- (de)serialization ---------------------------------------------
    def top_entries_state(self, n: int) -> dict[str, Any]:
        s = self.size
        n = min(int(n), s)
        if n <= 0:
            return self._empty_state()
        pr = self.scalars["priority_score"][:s]
        order = torch.argsort(pr, descending=True)[:n]
        state: dict[str, Any] = {
            "cfg": self._cfg_dict(),
            "size": int(n),
            "feature": self.feature[:s][order].clone(),
            "action": self.action[:s][order].clone(),
        }
        for name in _SCALAR_COLUMNS:
            state[name] = self.scalars[name][:s][order].clone()
        return state

    def state_dict(self) -> dict[str, Any]:
        s = self.size
        state: dict[str, Any] = {
            "cfg": self._cfg_dict(),
            "size": int(s),
            "add_count": int(self.add_count),
            "seen_count": int(self.seen_count),
            "feature": self.feature[:s].clone(),
            "action": self.action[:s].clone(),
        }
        for name in _SCALAR_COLUMNS:
            state[name] = self.scalars[name][:s].clone()
        return state

    def load_state_dict(self, state: dict[str, Any]) -> None:
        block = self._block_from_state(state)
        # reset and store (trim to capacity)
        self.size = 0
        if int(block["feature"].shape[0]) == 0:
            self.add_count = int(state.get("add_count", 0))
            self.seen_count = int(state.get("seen_count", 0))
            return
        keep_idx = self._select_keep_indices(block, self.capacity) if int(
            block["feature"].shape[0]
        ) > self.capacity else torch.arange(int(block["feature"].shape[0]))
        self._store_block({k: v[keep_idx] for k, v in block.items()})
        self.add_count = int(state.get("add_count", self.size))
        self.seen_count = int(state.get("seen_count", self.size))

    def _block_from_state(self, state: dict[str, Any]) -> dict[str, torch.Tensor]:
        s = int(state.get("size", 0))
        feature = state.get("feature")
        if feature is None or s == 0:
            return self._empty_block()
        feature = feature[:s].to(torch.float32)
        action = state["action"][:s].to(torch.float32)
        # adapt feature width to current cfg if needed
        if int(feature.shape[-1]) != int(self.cfg.feature_dim):
            fd = int(self.cfg.feature_dim)
            if int(feature.shape[-1]) > fd:
                feature = feature[:, :fd]
            else:
                pad = fd - int(feature.shape[-1])
                feature = torch.cat([feature, torch.zeros(s, pad)], dim=1)
        block = {"feature": feature, "action": action}
        for name in _SCALAR_COLUMNS:
            v = state.get(name)
            block[name] = (
                v[:s].to(torch.float32) if v is not None else torch.zeros(s)
            )
        return block

    def _empty_block(self) -> dict[str, torch.Tensor]:
        block = {
            "feature": torch.zeros(0, int(self.cfg.feature_dim)),
            "action": torch.zeros(0, int(self.cfg.action_dim)),
        }
        for name in _SCALAR_COLUMNS:
            block[name] = torch.zeros(0)
        return block

    def _empty_state(self) -> dict[str, Any]:
        state: dict[str, Any] = {"cfg": self._cfg_dict(), "size": 0}
        state["feature"] = torch.zeros(0, int(self.cfg.feature_dim))
        state["action"] = torch.zeros(0, int(self.cfg.action_dim))
        for name in _SCALAR_COLUMNS:
            state[name] = torch.zeros(0)
        return state

    def _cfg_dict(self) -> dict[str, Any]:
        return {
            "bank_size": int(self.cfg.bank_size),
            "feature_dim": int(self.cfg.feature_dim),
            "action_dim": int(self.cfg.action_dim),
            "horizons": list(self.cfg.horizons),
        }
