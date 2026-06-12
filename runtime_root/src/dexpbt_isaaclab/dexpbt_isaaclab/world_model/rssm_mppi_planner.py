"""RSSM reranking + full H-step MPPI for EIR-MPC.

Two modes share this module:

* LIGHT (``RSSMReranker``): at a FIXED belief, rescore candidate first-actions
  with the trained prediction heads. Single-step, no latent rollout. Kept as a
  fallback and for backward compatibility.

* FULL (``RSSMMPPIPlanner``): a genuine H-step model-predictive action-sequence
  search. Given a per-row belief ``b_t`` (from ``rssm_belief_provider``) it
  samples ``K`` action sequences of length ``H``, rolls them forward through the
  RSSM *prior* transition (``model.imagine_step``), scores each sequence with the
  success/drop/recovery/value/task-error heads, optionally refines with CEM, and
  offers the best sequence's first action back into the EIR candidate pool as an
  ``rssm`` source. The EIR reranker still chooses among actor / gaussian / elite
  / archive / rssm, so MPPI never unilaterally overrides elite/scorer.

Everything runs under ``torch.no_grad`` (no gradient through the world model).
Both modes self-disable with a ``fallback_reason`` when the model / belief is
unavailable, before the warmup frame, on high uncertainty, or on low margin.

If the backbone lacks a real prior transition, ``imagine_step`` degrades to a
fixed-belief rollout and the planner logs ``imagine_mode=fixed_belief_fallback``
so the run is still informative.
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


# ======================================================================
# Full H-step MPPI planner
# ======================================================================

# MPPI fallback reasons.
MPPI_OK = 0
MPPI_DISABLED = 1
MPPI_BEFORE_START = 2
MPPI_NO_BELIEF = 3
MPPI_NO_RSSM_MODEL = 4
MPPI_NO_IMAGINE_STEP = 5
MPPI_HIGH_UNCERTAINTY = 6
MPPI_LOW_MARGIN = 7
MPPI_EXCEPTION = 8

_MPPI_FALLBACK_NAMES = {
    MPPI_OK: "ok", MPPI_DISABLED: "disabled", MPPI_BEFORE_START: "before_start",
    MPPI_NO_BELIEF: "no_belief", MPPI_NO_RSSM_MODEL: "no_rssm_model",
    MPPI_NO_IMAGINE_STEP: "no_imagine_step", MPPI_HIGH_UNCERTAINTY: "high_uncertainty",
    MPPI_LOW_MARGIN: "low_margin", MPPI_EXCEPTION: "exception",
}


@dataclass
class RSSMMPPIConfig:
    enabled: bool = False
    mode: str = "full"  # off | light | full
    start_after_frames: int = 800_000_000

    horizon: int = 4
    num_sequences: int = 64
    num_elite: int = 8
    cem_iters: int = 1

    action_noise_std: float = 0.10
    action_noise_decay: float = 0.85
    max_action_delta: float = 0.5
    smooth_noise: bool = True

    include_actor_sequence: bool = True
    include_elite_sequence: bool = True
    include_archive_sequence: bool = True

    uncertainty_threshold: float = 0.4
    min_sequence_margin: float = 0.02

    w_success: float = 1.0
    w_recovery: float = 0.8
    w_drop: float = 2.0
    w_value: float = 0.2
    w_task_improve: float = 0.5
    w_uncertainty: float = 0.2
    w_action_smoothness: float = 0.05
    w_action_delta: float = 0.02

    fail_fast: bool = False

    def __post_init__(self) -> None:
        self.horizon = max(1, int(self.horizon))
        self.num_sequences = max(1, int(self.num_sequences))
        self.num_elite = max(1, min(int(self.num_elite), int(self.num_sequences)))
        self.cem_iters = max(1, int(self.cem_iters))


class RSSMMPPIPlanner:
    """Full H-step model-predictive action-sequence search over RSSM prior."""

    def __init__(self, cfg: RSSMMPPIConfig) -> None:
        self.cfg = cfg
        self.last_active = 0.0
        self.last_fallback_reason = MPPI_DISABLED
        self.last_mode = cfg.mode
        self.last_imagine_mode = "none"
        self.last_best_score = 0.0
        self.last_actor_score = 0.0
        self.last_score_margin = 0.0
        self.last_uncertainty = 0.0
        self.last_best_first_action_delta = 0.0
        self.last_low_margin_fraction = 0.0
        self.last_high_uncertainty_fraction = 0.0
        self.last_num_sequences = 0
        self.last_horizon = 0

    # ---- helpers --------------------------------------------------------
    def _inactive(self, reason: int, B: int, A: int, actor_action: torch.Tensor, dev, dtype) -> dict:
        self.last_active = 0.0
        self.last_fallback_reason = int(reason)
        H = int(self.cfg.horizon)
        return {
            "active": False,
            "fallback_reason": int(reason),
            "best_action": actor_action,
            "best_sequence": actor_action[:, None, :].expand(B, H, A).contiguous(),
            "best_score": torch.zeros(B, device=dev, dtype=dtype),
            "actor_score": torch.zeros(B, device=dev, dtype=dtype),
            "score_margin": torch.zeros(B, device=dev, dtype=dtype),
            "uncertainty": torch.zeros(B, device=dev, dtype=dtype),
            "best_first_action_delta": torch.zeros(B, device=dev, dtype=dtype),
            "mode": self.cfg.mode,
            "imagine_mode": "none",
        }

    def _build_sequences(
        self,
        actor_action: torch.Tensor,        # [B, A]
        candidate_actions: torch.Tensor,   # [B, K, A]
        mean_seq: torch.Tensor,            # [B, H, A] gaussian mean
        std: float,
    ) -> tuple[torch.Tensor, int]:
        """Return ``(seq[B, Ktot, H, A], actor_index)``. Column 0 is the actor."""
        cfg = self.cfg
        B, A = actor_action.shape
        H = int(cfg.horizon)
        dev, dtype = actor_action.device, actor_action.dtype
        blocks: list[torch.Tensor] = []

        # actor repeated sequence (always index 0)
        actor_seq = actor_action[:, None, :].expand(B, H, A).unsqueeze(1)  # [B,1,H,A]
        blocks.append(actor_seq)

        # candidate-derived repeated sequences (elite/archive/local first-actions)
        if cfg.include_elite_sequence and candidate_actions is not None and candidate_actions.shape[1] > 0:
            cand_seq = candidate_actions[:, :, None, :].expand(
                B, candidate_actions.shape[1], H, A
            )
            blocks.append(cand_seq)

        # sampled gaussian sequences around the (CEM) mean
        K = int(cfg.num_sequences)
        if K > 0:
            noise = torch.randn(B, K, H, A, device=dev, dtype=dtype) * float(std)
            if cfg.smooth_noise and H > 1:
                noise = torch.cumsum(noise, dim=2) / (H ** 0.5)
            noise = noise.clamp(-float(cfg.max_action_delta), float(cfg.max_action_delta))
            sampled = (mean_seq[:, None, :, :] + noise).clamp(-1.0, 1.0)
            blocks.append(sampled)

        seq = torch.cat(blocks, dim=1).clamp(-1.0, 1.0)  # [B, Ktot, H, A]
        return seq, 0

    @torch.no_grad()
    def _rollout_and_score(
        self, *, belief_flat: torch.Tensor, seq: torch.Tensor, actor_action: torch.Tensor, rssm_model
    ) -> dict:
        """Roll each [B,Ktot,H,A] sequence through the prior and score it."""
        cfg = self.cfg
        B, K, H, A = seq.shape
        dev, dtype = seq.device, seq.dtype
        D = int(belief_flat.shape[-1])

        belief_rep = belief_flat[:, None, :].expand(B, K, D).reshape(B * K, D)
        state = rssm_model.get_state_from_belief(belief_rep)
        seq_bk = seq.reshape(B * K, H, A)

        succ_steps: list[torch.Tensor] = []
        drop_steps: list[torch.Tensor] = []
        rec_steps: list[torch.Tensor] = []
        value_steps: list[torch.Tensor] = []
        task_delta_steps: list[torch.Tensor] = []
        unc_steps: list[torch.Tensor] = []
        imagine_mode = "none"

        for h in range(H):
            action_h = seq_bk[:, h, :]
            state, preds = rssm_model.imagine_step(state, action_h)
            imagine_mode = str(preds.get("imagine_mode", imagine_mode))
            zeros = torch.zeros(B * K, device=dev, dtype=dtype)
            succ_steps.append(preds.get("success", zeros).reshape(B * K).to(dtype))
            drop_steps.append(preds.get("drop", zeros).reshape(B * K).to(dtype))
            rec_steps.append(preds.get("recovery", zeros).reshape(B * K).to(dtype))
            value_steps.append(preds.get("value", zeros).reshape(B * K).to(dtype))
            task_delta_steps.append(preds.get("task_error_delta", zeros).reshape(B * K).to(dtype))
            unc_steps.append(preds.get("prior_std_mean", zeros).reshape(B * K).to(dtype))

        succ = torch.stack(succ_steps, dim=0)        # [H, B*K]
        drop = torch.stack(drop_steps, dim=0)
        rec = torch.stack(rec_steps, dim=0)
        value = torch.stack(value_steps, dim=0)
        task_delta = torch.stack(task_delta_steps, dim=0)
        unc = torch.stack(unc_steps, dim=0)

        succ_max = succ.max(dim=0).values
        drop_max = drop.max(dim=0).values
        rec_mean = rec.mean(dim=0)
        value_last = value[-1]
        # task improvement: predicted cumulative decrease in d_hat
        task_improve = -task_delta.sum(dim=0)
        unc_mean = unc.mean(dim=0)

        # action regularizers
        actor_rep = actor_action[:, None, :].expand(B, K, A).reshape(B * K, A)
        a_delta = ((seq_bk - actor_rep[:, None, :]) ** 2).sum(dim=-1).sum(dim=1)  # [B*K]
        if H > 1:
            smooth = ((seq_bk[:, 1:, :] - seq_bk[:, :-1, :]) ** 2).sum(dim=-1).sum(dim=1)
        else:
            smooth = torch.zeros(B * K, device=dev, dtype=dtype)

        score = (
            cfg.w_success * succ_max
            + cfg.w_recovery * rec_mean
            - cfg.w_drop * drop_max
            + cfg.w_value * value_last
            + cfg.w_task_improve * task_improve
            - cfg.w_uncertainty * unc_mean
            - cfg.w_action_smoothness * smooth
            - cfg.w_action_delta * a_delta
        ).reshape(B, K)
        unc_bk = unc_mean.reshape(B, K)
        return {"score": score, "uncertainty": unc_bk, "imagine_mode": imagine_mode}

    # ---- main entry -----------------------------------------------------
    @torch.no_grad()
    def plan(
        self,
        *,
        belief_flat: Optional[torch.Tensor],
        actor_action: torch.Tensor,
        candidate_actions: torch.Tensor,
        rssm_model,
        frames: int,
    ) -> dict:
        cfg = self.cfg
        B, A = actor_action.shape
        dev, dtype = actor_action.device, actor_action.dtype
        self.last_mode = cfg.mode
        self.last_num_sequences = 0
        self.last_horizon = int(cfg.horizon)

        if not cfg.enabled or cfg.mode == "off":
            return self._inactive(MPPI_DISABLED, B, A, actor_action, dev, dtype)
        if int(frames) < int(cfg.start_after_frames):
            return self._inactive(MPPI_BEFORE_START, B, A, actor_action, dev, dtype)
        if belief_flat is None:
            return self._inactive(MPPI_NO_BELIEF, B, A, actor_action, dev, dtype)
        if rssm_model is None or not hasattr(rssm_model, "get_state_from_belief"):
            return self._inactive(MPPI_NO_RSSM_MODEL, B, A, actor_action, dev, dtype)
        if not hasattr(rssm_model, "imagine_step"):
            return self._inactive(MPPI_NO_IMAGINE_STEP, B, A, actor_action, dev, dtype)
        if int(belief_flat.shape[0]) != B:
            return self._inactive(MPPI_NO_BELIEF, B, A, actor_action, dev, dtype)

        try:
            belief_flat = belief_flat.to(device=dev, dtype=dtype)
            H = int(cfg.horizon)
            mean_seq = actor_action[:, None, :].expand(B, H, A).contiguous()
            std = float(cfg.action_noise_std)
            last = None
            seq = None
            for it in range(int(cfg.cem_iters)):
                seq, _actor_idx = self._build_sequences(
                    actor_action, candidate_actions, mean_seq, std
                )
                last = self._rollout_and_score(
                    belief_flat=belief_flat, seq=seq, actor_action=actor_action,
                    rssm_model=rssm_model,
                )
                if it < int(cfg.cem_iters) - 1:
                    # CEM refit: take top num_elite, update mean over their steps,
                    # decay std, then resample on the next iter.
                    score = last["score"]
                    K = int(score.shape[1])
                    ne = min(int(cfg.num_elite), K)
                    elite_idx = torch.topk(score, ne, dim=1).indices  # [B, ne]
                    rows = torch.arange(B, device=dev)[:, None].expand(B, ne)
                    elite_seq = seq[rows, elite_idx]  # [B, ne, H, A]
                    mean_seq = elite_seq.mean(dim=1)  # [B, H, A]
                    std = std * float(cfg.action_noise_decay)

            score = last["score"]              # [B, Ktot]
            unc = last["uncertainty"]          # [B, Ktot]
            self.last_imagine_mode = last["imagine_mode"]
            self.last_num_sequences = int(seq.shape[1])

            actor_score = score[:, 0]
            best_idx = score.argmax(dim=1)
            rows = torch.arange(B, device=dev)
            best_score = score[rows, best_idx]
            best_seq = seq[rows, best_idx]            # [B, H, A]
            best_action = best_seq[:, 0, :]
            margin = best_score - actor_score
            best_unc = unc[rows, best_idx]
            first_delta = (best_action - actor_action).norm(dim=-1)

            self.last_best_score = float(best_score.mean().item())
            self.last_actor_score = float(actor_score.mean().item())
            self.last_score_margin = float(margin.mean().item())
            self.last_uncertainty = float(best_unc.mean().item())
            self.last_best_first_action_delta = float(first_delta.mean().item())
            self.last_high_uncertainty_fraction = float(
                (best_unc > float(cfg.uncertainty_threshold)).float().mean().item()
            )
            self.last_low_margin_fraction = float(
                (margin < float(cfg.min_sequence_margin)).float().mean().item()
            )

            mean_unc = float(best_unc.mean().item())
            mean_margin = float(margin.mean().item())
            # uncertainty gate (only meaningful with a real prior rollout; in
            # fixed-belief fallback prior_std is 0 so this never blocks).
            if mean_unc > float(cfg.uncertainty_threshold):
                self.last_active = 0.0
                self.last_fallback_reason = MPPI_HIGH_UNCERTAINTY
                out = self._inactive(MPPI_HIGH_UNCERTAINTY, B, A, actor_action, dev, dtype)
                out.update({
                    "uncertainty": best_unc, "score_margin": margin,
                    "best_score": best_score, "actor_score": actor_score,
                    "imagine_mode": self.last_imagine_mode,
                })
                return out
            if mean_margin < float(cfg.min_sequence_margin):
                self.last_active = 0.0
                self.last_fallback_reason = MPPI_LOW_MARGIN
                out = self._inactive(MPPI_LOW_MARGIN, B, A, actor_action, dev, dtype)
                out.update({
                    "uncertainty": best_unc, "score_margin": margin,
                    "best_score": best_score, "actor_score": actor_score,
                    "best_action": best_action, "best_sequence": best_seq,
                    "best_first_action_delta": first_delta,
                    "imagine_mode": self.last_imagine_mode,
                })
                return out

            self.last_active = 1.0
            self.last_fallback_reason = MPPI_OK
            return {
                "active": True,
                "fallback_reason": MPPI_OK,
                "best_action": best_action,
                "best_sequence": best_seq,
                "best_score": best_score,
                "actor_score": actor_score,
                "score_margin": margin,
                "uncertainty": best_unc,
                "best_first_action_delta": first_delta,
                "mode": cfg.mode,
                "imagine_mode": self.last_imagine_mode,
            }
        except Exception as exc:  # pragma: no cover - defensive
            if cfg.fail_fast:
                raise
            print(f"[EIR-MPPI] plan exception (fallback): {exc}")
            return self._inactive(MPPI_EXCEPTION, B, A, actor_action, dev, dtype)
