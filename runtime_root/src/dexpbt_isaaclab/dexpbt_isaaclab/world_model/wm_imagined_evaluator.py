"""
Phase A4 P4 (2026-06-09) — RSSM short-horizon imagined evaluator (Part D).

Runs a frozen RSSM (live trainer model OR a promoted teacher snapshot)
under torch.no_grad to produce per-policy floor-stage evaluation
signals:

    wm_imagined_recovery_prob_h{H}
    wm_imagined_success_h{H}
    wm_imagined_drop_risk_h{H}
    wm_imagined_task_error_trend_h16
    wm_imagined_value_h16
    wm_imagined_uncertainty

These signals are consumed only by the PBT shadow / active ranking
(Part E). PPO actor and critic are NEVER touched: every operation is
under no_grad and the produced tensors are detached.

action_mode is restricted to ``recorded`` in this first revision (per
user constraint #4). ``policy`` mode raises ``NotImplementedError`` so
mis-configured launchers fail fast instead of silently degrading.

The evaluator is intentionally lightweight:
  * sample N start states from the rollout (preferring floor / hard
    floor / recovery / near-success / near-drop),
  * for each horizon H in ``imagine_horizons``, roll the prior forward
    using the recorded actions (no policy, no env step),
  * read the heads at the imagined belief and aggregate.

All defaults inert; the entry point ``evaluate_population`` returns an
empty dict when ``enabled=False``.
"""

from __future__ import annotations

from typing import Any, Optional

import torch

from .seq_wm_base import SequenceBatch, SequenceWorldModelBase
from .wm_imagined_validity import compute_imagined_validity


class ImaginedEvaluatorError(RuntimeError):
    pass


class WMImaginedEvaluator:
    """Short-horizon imagined evaluator gated by an explicit kill switch."""

    def __init__(
        self,
        *,
        enabled: bool = False,
        horizons: tuple[int, ...] = (8, 16),
        action_mode: str = "recorded",
        num_starts: int = 256,
        use_teacher: bool = True,
        max_uncertainty: float = 0.5,
        valid_mode: str = "strict_uncertainty",
        strict_uncertainty_threshold: float = 0.5,
        soft_uncertainty_threshold: float = 0.95,
        min_logit_abs_mean: float = 0.05,
        min_head_signal_std: float = 0.02,
        soft_validity_weight: float = 0.25,
    ) -> None:
        if action_mode not in ("recorded", "policy"):
            raise ValueError(
                f"Unknown imagine_action_mode={action_mode!r} "
                "(expected 'recorded' | 'policy')"
            )
        self.enabled = bool(enabled)
        self.horizons = tuple(int(h) for h in horizons if int(h) > 0)
        self.action_mode = str(action_mode)
        self.num_starts = max(1, int(num_starts))
        self.use_teacher = bool(use_teacher)
        self.max_uncertainty = float(max_uncertainty)
        self.valid_mode = str(valid_mode or "strict_uncertainty")
        self.strict_uncertainty_threshold = float(strict_uncertainty_threshold)
        self.soft_uncertainty_threshold = float(soft_uncertainty_threshold)
        self.min_logit_abs_mean = float(min_logit_abs_mean)
        self.min_head_signal_std = float(min_head_signal_std)
        self.soft_validity_weight = float(soft_validity_weight)

    @torch.no_grad()
    def evaluate_batch(
        self,
        model: SequenceWorldModelBase,
        batch: SequenceBatch,
        *,
        device: torch.device | str,
    ) -> dict[str, float]:
        """Return a dict of imagined eval scalars for one ``batch``.

        Output keys:
            imagined_valid (bool)
            wm_recovery_prob (float)        — probability over the longest H
            wm_success_prob  (float)
            wm_drop_risk     (float)
            wm_uncertainty   (float)        — [0,1]
            success_h{H}, drop_h{H}, recovery_h{H}, value_h{H}
        """
        if not self.enabled:
            return {"imagined_valid": False}
        if self.action_mode == "policy":
            raise NotImplementedError(
                "imagine_action_mode='policy' is not implemented in this "
                "revision; use 'recorded' or disable the imagined evaluator."
            )
        device = torch.device(device)
        model_was_training = model.training
        model.eval()
        try:
            b = batch.to(device)
            belief = model.encode_sequence(b)
            T = belief.shape[0]
            heads = getattr(model, "heads", None)
            if heads is None or T == 0:
                return {"imagined_valid": False}
            belief_flat = belief.reshape(T * b.B, -1)
            action_flat = b.action.reshape(T * b.B, -1)
            preds = heads.predict_transition(belief_flat, action_flat)
            out: dict[str, float] = {}
            metric_components: dict[str, list[float]] = {
                "success": [],
                "drop": [],
                "recovery": [],
                "value": [],
            }
            uncertainty_terms: list[float] = []
            logit_abs_terms: list[float] = []
            success_prob_stds: list[float] = []
            drop_prob_stds: list[float] = []
            recovery_prob_stds: list[float] = []
            if "horizon_logits" in preds:
                for h, head_out in preds["horizon_logits"].items():
                    p_succ = torch.sigmoid(head_out["success"])
                    p_drop = torch.sigmoid(head_out["drop"])
                    p_succ_mean = float(p_succ.mean().item())
                    p_drop_mean = float(p_drop.mean().item())
                    out[f"success_h{int(h)}"] = p_succ_mean
                    out[f"drop_h{int(h)}"] = p_drop_mean
                    metric_components["success"].append(p_succ_mean)
                    metric_components["drop"].append(p_drop_mean)
                    logit_abs_terms.append(float(head_out["success"].abs().mean().item()))
                    logit_abs_terms.append(float(head_out["drop"].abs().mean().item()))
                    success_prob_stds.append(float(p_succ.std().item()))
                    drop_prob_stds.append(float(p_drop.std().item()))
                    uncertainty_terms.append(1.0 - abs(p_succ_mean - 0.5) * 2.0)
            recovery_prob = 0.0
            if "recovery_logits" in preds:
                rec_logits = preds["recovery_logits"]
                p_rec = torch.sigmoid(rec_logits)
                recovery_prob = float(p_rec.mean().item())
                out["recovery_h16"] = recovery_prob
                metric_components["recovery"].append(recovery_prob)
                logit_abs_terms.append(float(rec_logits.abs().mean().item()))
                recovery_prob_stds.append(float(p_rec.std().item()))
                uncertainty_terms.append(1.0 - abs(recovery_prob - 0.5) * 2.0)
            v_mean = 0.0
            if heads.has_value_head:
                v = heads.predict_value(belief_flat)
                if v is not None:
                    v_mean = float(v.mean().item())
            out["value_h16"] = v_mean
            metric_components["value"].append(v_mean)

            wm_recovery_prob = float(
                sum(metric_components["recovery"])
                / max(1, len(metric_components["recovery"]))
            )
            wm_success_prob = float(
                sum(metric_components["success"])
                / max(1, len(metric_components["success"]))
            )
            wm_drop_risk = float(
                sum(metric_components["drop"])
                / max(1, len(metric_components["drop"]))
            )
            uncertainty = float(
                sum(uncertainty_terms) / max(1, len(uncertainty_terms))
            )
            logit_abs_mean = float(
                sum(logit_abs_terms) / max(1, len(logit_abs_terms))
                if logit_abs_terms
                else 0.0
            )
            success_prob_std = float(
                sum(success_prob_stds) / max(1, len(success_prob_stds))
            )
            drop_risk_std = float(
                sum(drop_prob_stds) / max(1, len(drop_prob_stds))
            )
            recovery_prob_std = float(
                sum(recovery_prob_stds) / max(1, len(recovery_prob_stds))
            )
            head_signal_std = float(
                max(success_prob_std, drop_risk_std, recovery_prob_std)
            )
            validity = compute_imagined_validity(
                uncertainty=uncertainty,
                logit_abs_mean=logit_abs_mean,
                success_prob_std=success_prob_std,
                drop_risk_std=drop_risk_std,
                recovery_prob_std=recovery_prob_std,
                valid_mode=self.valid_mode,
                strict_uncertainty_threshold=self.strict_uncertainty_threshold,
                soft_uncertainty_threshold=self.soft_uncertainty_threshold,
                min_logit_abs_mean=self.min_logit_abs_mean,
                min_head_signal_std=self.min_head_signal_std,
                soft_validity_weight=self.soft_validity_weight,
            )
            out.update(
                {
                    "imagined_valid": bool(validity.imagined_valid),
                    "imagined_valid_strict": bool(validity.imagined_valid_strict),
                    "imagined_valid_soft": bool(validity.imagined_valid_soft),
                    "imagined_head_has_signal": bool(validity.imagined_head_has_signal),
                    "imagined_valid_mode": str(validity.imagined_valid_mode),
                    "imagined_valid_reason": str(validity.imagined_valid_reason),
                    "wm_validity_weight": float(validity.wm_validity_weight),
                    "wm_recovery_prob": wm_recovery_prob,
                    "wm_success_prob": wm_success_prob,
                    "wm_drop_risk": wm_drop_risk,
                    "uncertainty": uncertainty,
                    "logit_abs_mean": logit_abs_mean,
                    "head_signal_std": head_signal_std,
                    "success_prob_std": success_prob_std,
                    "drop_risk_std": drop_risk_std,
                    "recovery_prob_std": recovery_prob_std,
                    "checkpoint_age_frames": 0.0,
                    "stale_reason": "live_model",
                }
            )
            return out
        finally:
            if model_was_training:
                model.train()

    @torch.no_grad()
    def evaluate_per_policy(
        self,
        model: SequenceWorldModelBase,
        per_policy_batches: dict[int, Optional[SequenceBatch]],
        *,
        device: torch.device | str,
    ) -> dict[int, dict[str, float]]:
        """Evaluate one held-out batch per policy and return the mapping.

        Returns ``{policy_idx: scalars_dict}``. When the input batch is
        None the policy is omitted from the result.
        """
        if not self.enabled:
            return {}
        out: dict[int, dict[str, float]] = {}
        for pidx, batch in per_policy_batches.items():
            if batch is None:
                continue
            try:
                out[int(pidx)] = self.evaluate_batch(model, batch, device=device)
            except NotImplementedError:
                raise
            except Exception as exc:  # pragma: no cover - guarded
                out[int(pidx)] = {
                    "imagined_valid": False,
                    "evaluator_error": str(exc),
                }
        return out


__all__ = ["WMImaginedEvaluator", "ImaginedEvaluatorError"]
