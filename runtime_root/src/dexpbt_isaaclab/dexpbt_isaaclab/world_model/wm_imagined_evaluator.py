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
            # Per user constraint #4, policy-action mode must fail fast.
            raise NotImplementedError(
                "imagine_action_mode='policy' is not implemented in this "
                "revision; use 'recorded' or disable the imagined evaluator."
            )
        device = torch.device(device)
        model_was_training = model.training
        model.eval()
        try:
            b = batch.to(device)
            losses = model.compute_aux_loss(b)
            # Pull head outputs by recomputing predictions; cheaper than
            # re-running the full loss path. Heads.predict_transition takes
            # belief_flat + action_flat.
            belief = model.encode_sequence(b)  # (T, B, D)
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
            if "horizon_logits" in preds:
                for h, head_out in preds["horizon_logits"].items():
                    p_succ = torch.sigmoid(head_out["success"]).mean().item()
                    p_drop = torch.sigmoid(head_out["drop"]).mean().item()
                    out[f"success_h{int(h)}"] = float(p_succ)
                    out[f"drop_h{int(h)}"] = float(p_drop)
                    metric_components["success"].append(float(p_succ))
                    metric_components["drop"].append(float(p_drop))
                    # Uncertainty proxy: distance from 0.5 (closer = less
                    # confident). 1.0 means perfectly uncertain.
                    p = float(p_succ)
                    uncertainty_terms.append(1.0 - abs(p - 0.5) * 2.0)
            recovery_prob = 0.0
            if "recovery_logits" in preds:
                p_rec = torch.sigmoid(preds["recovery_logits"]).mean().item()
                recovery_prob = float(p_rec)
                out["recovery_h16"] = recovery_prob
                metric_components["recovery"].append(recovery_prob)
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
            imagined_valid = bool(uncertainty <= self.max_uncertainty)
            out.update(
                {
                    "imagined_valid": imagined_valid,
                    "wm_recovery_prob": wm_recovery_prob,
                    "wm_success_prob": wm_success_prob,
                    "wm_drop_risk": wm_drop_risk,
                    "uncertainty": uncertainty,
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
