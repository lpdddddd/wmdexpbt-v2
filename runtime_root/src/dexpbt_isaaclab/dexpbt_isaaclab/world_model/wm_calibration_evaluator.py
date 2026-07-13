"""WM calibration evaluator for CSER-v0."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import torch

from .wm_calibration_metrics import (
    auc_binary,
    clamp01,
    ece,
    pos_rate,
    spearman_rank,
    uncertainty_error_corr,
)


@dataclass
class WMCalibrationConfig:
    enabled: bool = False
    eval_interval_frames: int = 50_000_000
    min_samples: int = 512
    heldout_fraction: float = 0.2
    alpha_min: float = 0.0
    alpha_max: float = 1.0
    gate_auc_recovery: float = 0.55
    gate_ece_max: float = 0.15
    gate_rank_corr_min: float = 0.1
    ema_beta: float = 0.9
    pass_threshold: float = 0.5
    eval_tol_mode: str = "hybrid"
    eval_tol_threshold: float = 0.05
    eval_tol_quantile: float = 0.5
    eval_tol_margin: float = 1.1
    eval_tol_cap: float = 0.06
    eval_tol_min_samples_for_stats: int = 32
    # Must match SnippetWMRerankConfig score weights (injected at wiring time).
    w_rec: float = 0.3
    w_success: float = 0.2
    w_drop: float = 0.4
    w_unc: float = 0.1
    score_cap: float = 0.25

    def apply_snippet_rerank_weights(self, rerank_cfg: Any) -> None:
        """Copy score weights from the live snippet reranker config."""
        if rerank_cfg is None:
            return
        self.w_rec = float(getattr(rerank_cfg, "w_rec", self.w_rec))
        self.w_success = float(getattr(rerank_cfg, "w_success", self.w_success))
        self.w_drop = float(getattr(rerank_cfg, "w_drop", self.w_drop))
        self.w_unc = float(getattr(rerank_cfg, "w_unc", self.w_unc))
        self.score_cap = float(getattr(rerank_cfg, "score_cap", self.score_cap))


@dataclass
class WMCalibrationState:
    alpha_wm: float = 0.0
    ema_raw: float = 0.0
    calibration_pass: bool = False
    last_eval_frame: int = 0
    last_metrics: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "alpha_wm": float(self.alpha_wm),
            "ema_raw": float(self.ema_raw),
            "calibration_pass": bool(self.calibration_pass),
            "last_eval_frame": int(self.last_eval_frame),
            "last_metrics": dict(self.last_metrics),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "WMCalibrationState":
        st = cls()
        st.alpha_wm = float(d.get("alpha_wm", 0.0))
        st.ema_raw = float(d.get("ema_raw", 0.0))
        st.calibration_pass = bool(d.get("calibration_pass", False))
        st.last_eval_frame = int(d.get("last_eval_frame", 0))
        lm = d.get("last_metrics")
        st.last_metrics = dict(lm) if isinstance(lm, dict) else {}
        return st


def _as_tensor(values: list[float] | torch.Tensor) -> torch.Tensor:
    if isinstance(values, torch.Tensor):
        return values.float().reshape(-1)
    return torch.tensor([float(v) for v in values], dtype=torch.float32)


class WMCalibrationEvaluator:
    def __init__(self, cfg: WMCalibrationConfig) -> None:
        self.cfg = cfg

    def apply_snippet_rerank_weights(self, rerank_cfg: Any) -> None:
        """Delegate to config so callers can sync from the live reranker."""
        self.cfg.apply_snippet_rerank_weights(rerank_cfg)

    def evaluate_batch(self, batch: dict[str, list[float]]) -> dict[str, float]:
        from .snippet_wm_reranker import (
            compute_realized_wm_utility,
            compute_snippet_wm_term,
        )

        p_rec = batch.get("p_recovery", [])
        y_rec = batch.get("y_recovery", [])
        p_drop = batch.get("p_drop", [])
        y_drop = batch.get("y_drop", [])
        p_succ = batch.get("p_success", [])
        y_succ = batch.get("y_success", [])
        unc = batch.get("uncertainty", [])
        err = batch.get("error", [])
        improve = batch.get("y_improve", [])

        cfg = self.cfg
        p_rec_t = _as_tensor(p_rec)
        p_succ_t = _as_tensor(p_succ)
        p_drop_t = _as_tensor(p_drop)
        unc_t = _as_tensor(unc) if unc else torch.zeros_like(p_rec_t)
        y_rec_t = _as_tensor(y_rec)
        y_succ_t = _as_tensor(y_succ)
        y_drop_t = _as_tensor(y_drop)

        if p_rec_t.numel() > 0 and unc_t.numel() == p_rec_t.numel():
            predicted_wm_term = compute_snippet_wm_term(
                p_recovery=p_rec_t,
                p_success=p_succ_t if p_succ_t.numel() == p_rec_t.numel() else torch.zeros_like(p_rec_t),
                p_drop=p_drop_t if p_drop_t.numel() == p_rec_t.numel() else torch.zeros_like(p_rec_t),
                uncertainty=unc_t,
                w_rec=float(cfg.w_rec),
                w_success=float(cfg.w_success),
                w_drop=float(cfg.w_drop),
                w_unc=float(cfg.w_unc),
                score_cap=float(cfg.score_cap),
            )
            realized_utility = compute_realized_wm_utility(
                y_recovery=y_rec_t if y_rec_t.numel() == p_rec_t.numel() else torch.zeros_like(p_rec_t),
                y_success=y_succ_t if y_succ_t.numel() == p_rec_t.numel() else torch.zeros_like(p_rec_t),
                y_drop=y_drop_t if y_drop_t.numel() == p_rec_t.numel() else torch.zeros_like(p_rec_t),
                w_rec=float(cfg.w_rec),
                w_success=float(cfg.w_success),
                w_drop=float(cfg.w_drop),
            )
            pred_list = [float(x) for x in predicted_wm_term.tolist()]
            util_list = [float(x) for x in realized_utility.tolist()]
            rank_combined = spearman_rank(pred_list, util_list)
            combined_score_mean = float(predicted_wm_term.mean().item()) if predicted_wm_term.numel() else 0.0
            realized_utility_mean = float(realized_utility.mean().item()) if realized_utility.numel() else 0.0
        else:
            # Legacy override path (tests may pass wm_recovery_score / y_improve).
            pred_list = list(batch.get("wm_recovery_score", p_rec))
            util_list = list(improve) if improve else list(y_rec)
            rank_combined = spearman_rank(pred_list, util_list)
            combined_score_mean = float(sum(pred_list) / max(len(pred_list), 1)) if pred_list else 0.0
            realized_utility_mean = float(sum(util_list) / max(len(util_list), 1)) if util_list else 0.0

        # Recovery ranking diagnostic (prefer y_recovery; fall back to y_improve).
        rank_recovery = spearman_rank(p_rec, y_rec if y_rec else improve)

        metrics = {
            "wm_cal/auc_recovery": auc_binary(p_rec, y_rec),
            "wm_cal/auc_drop": auc_binary(p_drop, y_drop),
            "wm_cal/auc_success": auc_binary(p_succ, y_succ),
            "wm_cal/ece_recovery": ece(p_rec, y_rec),
            "wm_cal/ece_success": ece(p_succ, y_succ),
            "wm_cal/ece_drop": ece(p_drop, y_drop),
            "wm_cal/uncertainty_error_corr": uncertainty_error_corr(unc, err),
            "wm_cal/rank_corr_combined": float(rank_combined),
            "wm_cal/rank_corr_recovery": float(rank_recovery),
            # Legacy key now points at the combined ranking used by gate_rank.
            "wm_cal/rank_corr": float(rank_combined),
            "wm_cal/combined_score_mean": float(combined_score_mean),
            "wm_cal/realized_utility_mean": float(realized_utility_mean),
            "wm_cal/n_samples": float(len(p_rec)),
            "wm_cal/recovery_pos_rate": pos_rate(y_rec),
            "wm_cal/drop_pos_rate": pos_rate(y_drop),
            "wm_cal/success_pos_rate": pos_rate(y_succ),
            "wm_cal/insufficient_samples": float(batch.get("insufficient_samples", 0.0)),
            "wm_cal/missing_head": float(batch.get("missing_head", 0.0)),
            # Legacy aliases consumed by existing dashboards.
            "wm_cal/AUC_recovery": auc_binary(p_rec, y_rec),
            "wm_cal/ECE": ece(p_rec, y_rec),
            "wm_cal/eval_samples": float(len(p_rec)),
        }
        return metrics

    def compute_alpha(
        self, metrics: dict[str, float], state: WMCalibrationState
    ) -> WMCalibrationState:
        cfg = self.cfg
        auc_r = float(
            metrics.get("wm_cal/auc_recovery", metrics.get("wm_cal/AUC_recovery", 0.0))
        )
        ece_v = float(
            metrics.get("wm_cal/ece_recovery", metrics.get("wm_cal/ECE", 1.0))
        )
        rank_c = float(
            metrics.get(
                "wm_cal/rank_corr_combined",
                metrics.get("wm_cal/rank_corr", 0.0),
            )
        )
        gate_rec = clamp01((auc_r - cfg.gate_auc_recovery) / max(1e-6, 1.0 - cfg.gate_auc_recovery))
        gate_ece = clamp01((cfg.gate_ece_max - ece_v) / max(1e-6, cfg.gate_ece_max))
        gate_rank = clamp01((rank_c - cfg.gate_rank_corr_min) / max(1e-6, 1.0 - cfg.gate_rank_corr_min))
        # Recovery AUC/ECE remain base quality gates; combined ranking constrains
        # alpha when success/drop spoil the actual reranker score ordering.
        raw = gate_rec * gate_ece * gate_rank
        ema = cfg.ema_beta * float(state.ema_raw) + (1.0 - cfg.ema_beta) * raw
        alpha = clamp01(ema)
        alpha = max(float(cfg.alpha_min), min(float(cfg.alpha_max), alpha))
        state.ema_raw = float(ema)
        state.alpha_wm = float(alpha)
        state.calibration_pass = bool(raw >= float(cfg.pass_threshold))
        state.last_metrics = dict(metrics)
        metrics["wm_cal/alpha_wm"] = float(alpha)
        metrics["wm_cal/calibration_pass"] = 1.0 if state.calibration_pass else 0.0
        metrics["wm_cal/gate_recovery_auc"] = float(gate_rec)
        metrics["wm_cal/gate_recovery_ece"] = float(gate_ece)
        metrics["wm_cal/gate_combined_rank"] = float(gate_rank)
        return state
