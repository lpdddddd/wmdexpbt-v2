"""Light RSSM reranking for EIR-MPC (Phase E).

The RSSM (`rssm_lite.py`) has no prior-only latent rollout and no single-state
encoder, so a true H-step MPPI is out of scope. This module ships the LIGHT
version: at a FIXED belief (taken from the posterior over the recent recorded
window via ``encode_sequence``), it re-scores candidate first-actions with the
trained prediction heads (`heads.predict_transition(belief_flat, action_flat)`
plus the recovery / value heads). The best-scoring action is offered back into
the EIR candidate pool as an ``rssm`` source.

Everything runs under ``torch.no_grad`` (no gradient through the world model).
The reranker self-disables and reports a ``fallback_reason`` when the model is
unavailable, before the warmup frame, or when predictive uncertainty is high --
in which case the caller falls back to elite / scorer reranking.

A true H-step latent rollout would require adding a prior-only ``imagine_step``
to ``rssm_lite.py``; that is intentionally deferred.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional

import torch


# fallback reasons
RERANK_OK = 0
RERANK_DISABLED = 1
RERANK_BEFORE_START = 2
RERANK_MODEL_UNAVAILABLE = 3
RERANK_NO_HEADS = 4
RERANK_HIGH_UNCERTAINTY = 5


@dataclass
class RSSMRerankConfig:
    enabled: bool = False
    start_after_frames: int = 800_000_000
    horizon: int = 4
    num_sequences: int = 64
    noise_std: float = 0.10
    elite_frac: float = 0.2
    uncertainty_threshold: float = 0.4
    min_margin: float = 0.0

    w_success: float = 1.0
    w_recovery: float = 0.8
    w_drop: float = 2.0
    w_value: float = 0.2


def _extract_head_probs(preds: dict[str, Any], device, dtype) -> dict[str, torch.Tensor]:
    """Pull mean success/drop probs (over horizons) and recovery prob from a
    ``predict_transition`` output dict. Returns probabilities in [0,1]."""
    out: dict[str, torch.Tensor] = {}
    succ_list: list[torch.Tensor] = []
    drop_list: list[torch.Tensor] = []
    horizon_logits = preds.get("horizon_logits")
    if isinstance(horizon_logits, dict):
        for _h, head_out in horizon_logits.items():
            if "success" in head_out:
                succ_list.append(torch.sigmoid(head_out["success"]).reshape(-1))
            if "drop" in head_out:
                drop_list.append(torch.sigmoid(head_out["drop"]).reshape(-1))
    if succ_list:
        out["success_stack"] = torch.stack(succ_list, dim=0)  # [H, N]
        out["success"] = out["success_stack"].mean(dim=0)
    if drop_list:
        out["drop"] = torch.stack(drop_list, dim=0).mean(dim=0)
    rec = preds.get("recovery_logits")
    if rec is not None:
        out["recovery"] = torch.sigmoid(rec).reshape(-1)
    return out


class RSSMReranker:
    def __init__(self, cfg: RSSMRerankConfig) -> None:
        self.cfg = cfg
        self.last_active = 0.0
        self.last_best_score = 0.0
        self.last_actor_score = 0.0
        self.last_score_margin = 0.0
        self.last_uncertainty = 0.0
        self.last_best_first_action_delta = 0.0
        self.last_fallback_reason = RERANK_DISABLED

    def score_candidates(
        self,
        *,
        belief_flat: Optional[torch.Tensor],
        cand_actions: torch.Tensor,
        predict_fn: Optional[Callable[[torch.Tensor, torch.Tensor], dict[str, Any]]],
        value_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        frames: int = 0,
    ) -> dict[str, Any]:
        """Score ``cand_actions[B,K,A]`` at ``belief_flat[B,D]``.

        Returns dict: ``score[B,K]``, ``best_action[B,A]``, ``best_idx[B]``,
        ``uncertainty[B]``, ``active`` (bool), ``fallback_reason`` (int).
        Candidate column 0 is treated as the actor action for margin.
        """
        cfg = self.cfg
        B, K, A = cand_actions.shape
        dev = cand_actions.device
        dtype = cand_actions.dtype

        def _inactive(reason: int) -> dict[str, Any]:
            self.last_active = 0.0
            self.last_fallback_reason = int(reason)
            return {
                "score": torch.zeros(B, K, device=dev, dtype=dtype),
                "best_action": cand_actions[:, 0, :],
                "best_idx": torch.zeros(B, dtype=torch.long, device=dev),
                "uncertainty": torch.zeros(B, device=dev, dtype=dtype),
                "active": False,
                "fallback_reason": int(reason),
            }

        if not cfg.enabled:
            return _inactive(RERANK_DISABLED)
        if int(frames) < int(cfg.start_after_frames):
            return _inactive(RERANK_BEFORE_START)
        if predict_fn is None or belief_flat is None:
            return _inactive(RERANK_MODEL_UNAVAILABLE)

        with torch.no_grad():
            belief_rep = belief_flat[:, None, :].expand(B, K, belief_flat.shape[-1]).reshape(B * K, -1)
            cand_flat = cand_actions.reshape(B * K, A)
            try:
                preds = predict_fn(belief_rep, cand_flat)
            except Exception:
                return _inactive(RERANK_MODEL_UNAVAILABLE)
            probs = _extract_head_probs(preds, dev, dtype)
            if "success" not in probs and "recovery" not in probs:
                return _inactive(RERANK_NO_HEADS)

            succ = probs.get("success", torch.zeros(B * K, device=dev, dtype=dtype)).view(B, K)
            drop = probs.get("drop", torch.zeros(B * K, device=dev, dtype=dtype)).view(B, K)
            rec = probs.get("recovery", torch.zeros(B * K, device=dev, dtype=dtype)).view(B, K)

            val_term = torch.zeros(B, K, device=dev, dtype=dtype)
            if value_fn is not None:
                try:
                    v = value_fn(belief_rep).reshape(B, K)
                    v_min = v.min(dim=1, keepdim=True).values
                    v_max = v.max(dim=1, keepdim=True).values
                    val_term = (v - v_min) / (v_max - v_min + 1e-6)
                except Exception:
                    val_term = torch.zeros(B, K, device=dev, dtype=dtype)

            score = (
                cfg.w_success * succ
                + cfg.w_recovery * rec
                - cfg.w_drop * drop
                + cfg.w_value * val_term
            )

            # uncertainty proxy: std of success prob across horizons (per row,
            # averaged over candidates). High => unreliable, fall back.
            if "success_stack" in probs and int(probs["success_stack"].shape[0]) >= 2:
                ss = probs["success_stack"].view(-1, B, K)  # [H, B, K]
                uncertainty = ss.std(dim=0).mean(dim=1)  # [B]
            else:
                # single horizon (or none): no cross-horizon disagreement signal.
                uncertainty = torch.zeros(B, device=dev, dtype=dtype)

            best_idx = score.argmax(dim=1)
            rows = torch.arange(B, device=dev)
            best_action = cand_actions[rows, best_idx]
            best_score = score[rows, best_idx]
            actor_score = score[:, 0]
            margin = best_score - actor_score
            first_delta = (best_action - cand_actions[:, 0, :]).norm(dim=-1)

            self.last_best_score = float(best_score.mean().item())
            self.last_actor_score = float(actor_score.mean().item())
            self.last_score_margin = float(margin.mean().item())
            self.last_uncertainty = float(uncertainty.mean().item())
            self.last_best_first_action_delta = float(first_delta.mean().item())

            if float(uncertainty.mean().item()) > float(cfg.uncertainty_threshold):
                self.last_active = 0.0
                self.last_fallback_reason = RERANK_HIGH_UNCERTAINTY
                return {
                    "score": score,
                    "best_action": cand_actions[:, 0, :],
                    "best_idx": torch.zeros(B, dtype=torch.long, device=dev),
                    "uncertainty": uncertainty,
                    "active": False,
                    "fallback_reason": RERANK_HIGH_UNCERTAINTY,
                }

            self.last_active = 1.0
            self.last_fallback_reason = RERANK_OK
            return {
                "score": score,
                "best_action": best_action,
                "best_idx": best_idx,
                "uncertainty": uncertainty,
                "active": True,
                "fallback_reason": RERANK_OK,
            }
