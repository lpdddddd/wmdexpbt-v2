"""WM calibration evaluator for CSER-v0."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .wm_calibration_metrics import auc_binary, clamp01, ece, spearman_rank, uncertainty_error_corr


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


class WMCalibrationEvaluator:
    def __init__(self, cfg: WMCalibrationConfig) -> None:
        self.cfg = cfg

    def evaluate_batch(self, batch: dict[str, list[float]]) -> dict[str, float]:
        p_rec = batch.get("p_recovery", [])
        y_rec = batch.get("y_recovery", [])
        p_drop = batch.get("p_drop", [])
        y_drop = batch.get("y_drop", [])
        p_succ = batch.get("p_success", [])
        y_succ = batch.get("y_success", [])
        unc = batch.get("uncertainty", [])
        err = batch.get("error", [])
        wm_score = batch.get("wm_recovery_score", [])
        improve = batch.get("y_improve", [])
        return {
            "wm_cal/AUC_recovery": auc_binary(p_rec, y_rec),
            "wm_cal/AUC_drop": auc_binary(p_drop, y_drop),
            "wm_cal/AUC_success": auc_binary(p_succ, y_succ),
            "wm_cal/ECE": ece(p_rec, y_rec),
            "wm_cal/uncertainty_error_corr": uncertainty_error_corr(unc, err),
            "wm_cal/rank_corr": spearman_rank(wm_score, improve),
            "wm_cal/eval_samples": float(len(p_rec)),
        }

    def compute_alpha(
        self, metrics: dict[str, float], state: WMCalibrationState
    ) -> WMCalibrationState:
        cfg = self.cfg
        auc_r = float(metrics.get("wm_cal/AUC_recovery", 0.0))
        ece_v = float(metrics.get("wm_cal/ECE", 1.0))
        rank_c = float(metrics.get("wm_cal/rank_corr", 0.0))
        gate_rec = clamp01((auc_r - cfg.gate_auc_recovery) / max(1e-6, 1.0 - cfg.gate_auc_recovery))
        gate_ece = clamp01((cfg.gate_ece_max - ece_v) / max(1e-6, cfg.gate_ece_max))
        gate_rank = clamp01((rank_c - cfg.gate_rank_corr_min) / max(1e-6, 1.0 - cfg.gate_rank_corr_min))
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
        return state
