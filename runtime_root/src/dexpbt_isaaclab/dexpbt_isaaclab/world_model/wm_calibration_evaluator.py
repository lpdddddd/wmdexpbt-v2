"""WM calibration evaluator for CSER-v0."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import math

import torch

from .wm_calibration_metrics import (
    auc_binary,
    clamp01,
    ece,
    pos_rate,
    spearman_rank,
    uncertainty_error_corr,
)
from .wm_head_probs import (
    FAILURE_INSUFFICIENT_SAMPLES,
    FAILURE_MISSING_HEAD,
    FAILURE_OK,
    FAILURE_SHAPE_MISMATCH,
    FAILURE_NONFINITE,
)


@dataclass
class WMCalibrationConfig:
    enabled: bool = False
    eval_interval_frames: int = 50_000_000
    min_samples: int = 512
    # Env-slot held-out fraction for WM train vs calibration partition.
    # 0.0 disables partition (legacy: both paths see all envs).
    heldout_fraction: float = 0.2
    # Deterministic partition seed (combined with worker_rank + env_index).
    heldout_partition_seed: int = 0
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
    w_rec: float = 0.3
    w_success: float = 0.2
    w_drop: float = 0.4
    w_unc: float = 0.1
    score_cap: float = 0.25

    def apply_snippet_rerank_weights(self, rerank_cfg: Any) -> None:
        if rerank_cfg is None:
            return
        self.w_rec = float(getattr(rerank_cfg, "w_rec", self.w_rec))
        self.w_success = float(getattr(rerank_cfg, "w_success", self.w_success))
        self.w_drop = float(getattr(rerank_cfg, "w_drop", self.w_drop))
        self.w_unc = float(getattr(rerank_cfg, "w_unc", self.w_unc))
        self.score_cap = float(getattr(rerank_cfg, "score_cap", self.score_cap))

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": bool(self.enabled),
            "eval_interval_frames": int(self.eval_interval_frames),
            "min_samples": int(self.min_samples),
            "heldout_fraction": float(self.heldout_fraction),
            "heldout_partition_seed": int(self.heldout_partition_seed),
            "alpha_min": float(self.alpha_min),
            "alpha_max": float(self.alpha_max),
            "gate_auc_recovery": float(self.gate_auc_recovery),
            "gate_ece_max": float(self.gate_ece_max),
            "gate_rank_corr_min": float(self.gate_rank_corr_min),
            "ema_beta": float(self.ema_beta),
            "pass_threshold": float(self.pass_threshold),
        }


@dataclass
class WMCalibrationState:
    alpha_wm: float = 0.0
    ema_raw: float = 0.0
    calibration_pass: bool = False
    last_eval_frame: int = 0
    last_metrics: dict[str, float] = field(default_factory=dict)
    # Persisted partition knobs so restart keeps the same env-slot split.
    heldout_fraction: float = 0.2
    heldout_partition_seed: int = 0
    partition_worker_rank: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "alpha_wm": float(self.alpha_wm),
            "ema_raw": float(self.ema_raw),
            "calibration_pass": bool(self.calibration_pass),
            "last_eval_frame": int(self.last_eval_frame),
            "last_metrics": dict(self.last_metrics),
            "heldout_fraction": float(self.heldout_fraction),
            "heldout_partition_seed": int(self.heldout_partition_seed),
            "partition_worker_rank": int(self.partition_worker_rank),
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
        st.heldout_fraction = float(d.get("heldout_fraction", st.heldout_fraction))
        st.heldout_partition_seed = int(
            d.get("heldout_partition_seed", st.heldout_partition_seed)
        )
        st.partition_worker_rank = int(
            d.get("partition_worker_rank", st.partition_worker_rank)
        )
        return st


def _as_cpu_float1d(values: Any) -> tuple[torch.Tensor, bool]:
    """Detach to CPU float1d. Returns ``(tensor, ok)``; never raises on bad input."""
    empty = torch.empty(0, dtype=torch.float32)
    if values is None:
        return empty, False
    try:
        if isinstance(values, torch.Tensor):
            t = values.detach().float().reshape(-1).cpu()
            if int(t.numel()) == 0:
                return empty, False
            if not bool(torch.isfinite(t).all()):
                return empty, False
            return t, True
        seq = list(values)
        if len(seq) == 0:
            return empty, False
        out: list[float] = []
        for v in seq:
            if isinstance(v, (str, bytes, bytearray)):
                return empty, False
            if isinstance(v, bool):
                # bool is int subclass; allow as 0/1 label.
                out.append(float(v))
                continue
            if isinstance(v, (int, float)):
                fv = float(v)
                if not math.isfinite(fv):
                    return empty, False
                out.append(fv)
                continue
            # Reject nested objects / tensors-in-list mixed types.
            if isinstance(v, torch.Tensor):
                if v.numel() != 1 or not bool(torch.isfinite(v.float()).all()):
                    return empty, False
                out.append(float(v.detach().float().reshape(-1)[0].item()))
                continue
            return empty, False
        return torch.tensor(out, dtype=torch.float32), True
    except (TypeError, ValueError, RuntimeError):
        return empty, False


def apply_calibration_outcome(
    state: WMCalibrationState,
    *,
    cfg: WMCalibrationConfig,
    frames: int,
    metrics: dict[str, float],
    mode: str,
    failure_reason_id: float | None = None,
) -> WMCalibrationState:
    """Unified alpha transitions: structural zero / insufficient EMA decay."""
    mode_norm = str(mode or "").strip().lower()
    metrics = dict(metrics)
    if mode_norm == "structural":
        # Unconditional hard zero — ignore alpha_min so stale trust cannot linger.
        state.alpha_wm = 0.0
        state.ema_raw = 0.0
        state.calibration_pass = False
        metrics["wm_cal/alpha_wm"] = 0.0
        metrics["wm_cal/calibration_pass"] = 0.0
        if failure_reason_id is not None:
            metrics["wm_cal/failure_reason_id"] = float(failure_reason_id)
        else:
            metrics.setdefault("wm_cal/failure_reason_id", FAILURE_MISSING_HEAD)
        metrics["wm_cal/eval_valid"] = 0.0
    elif mode_norm in ("insufficient", "insufficient_heldout"):
        raw = 0.0
        ema = float(cfg.ema_beta) * float(state.ema_raw) + (1.0 - float(cfg.ema_beta)) * raw
        alpha = clamp01(ema)
        alpha = max(float(cfg.alpha_min), min(float(cfg.alpha_max), alpha))
        state.ema_raw = float(ema)
        state.alpha_wm = float(alpha)
        state.calibration_pass = False
        metrics["wm_cal/alpha_wm"] = float(alpha)
        metrics["wm_cal/calibration_pass"] = 0.0
        metrics["wm_cal/insufficient_samples"] = 1.0
        if mode_norm == "insufficient_heldout":
            metrics["wm_cal/insufficient_heldout_samples"] = 1.0
        # Always authoritative — do not keep pred_diag's OK=0 via setdefault.
        if failure_reason_id is not None:
            metrics["wm_cal/failure_reason_id"] = float(failure_reason_id)
        elif mode_norm == "insufficient_heldout":
            from .wm_head_probs import FAILURE_INSUFFICIENT_HELDOUT_SAMPLES

            metrics["wm_cal/failure_reason_id"] = FAILURE_INSUFFICIENT_HELDOUT_SAMPLES
        else:
            metrics["wm_cal/failure_reason_id"] = FAILURE_INSUFFICIENT_SAMPLES
        metrics["wm_cal/eval_valid"] = 0.0
    state.last_eval_frame = int(frames)
    state.last_metrics = dict(metrics)
    return state


class WMCalibrationEvaluator:
    def __init__(self, cfg: WMCalibrationConfig) -> None:
        self.cfg = cfg

    def apply_snippet_rerank_weights(self, rerank_cfg: Any) -> None:
        self.cfg.apply_snippet_rerank_weights(rerank_cfg)

    def evaluate_batch(self, batch: dict[str, Any]) -> dict[str, float]:
        from .snippet_wm_reranker import (
            compute_realized_wm_utility,
            compute_snippet_wm_term,
        )

        cfg = self.cfg
        required = (
            "p_recovery",
            "y_recovery",
            "p_success",
            "y_success",
            "p_drop",
            "y_drop",
            "uncertainty",
        )

        def _invalid(reason: float, **extra: float) -> dict[str, float]:
            out = {
                "wm_cal/auc_recovery": 0.0,
                "wm_cal/auc_drop": 0.0,
                "wm_cal/auc_success": 0.0,
                "wm_cal/ece_recovery": 1.0,
                "wm_cal/ece_success": 1.0,
                "wm_cal/ece_drop": 1.0,
                "wm_cal/uncertainty_error_corr": 0.0,
                "wm_cal/combined_abs_error_mean": 0.0,
                "wm_cal/recovery_abs_error_mean": 0.0,
                "wm_cal/recovery_uncertainty_error_corr": 0.0,
                "wm_cal/rank_corr_combined": 0.0,
                "wm_cal/rank_corr_recovery": 0.0,
                "wm_cal/rank_corr": 0.0,
                "wm_cal/combined_score_mean": 0.0,
                "wm_cal/realized_utility_mean": 0.0,
                "wm_cal/n_samples": 0.0,
                "wm_cal/recovery_pos_rate": 0.0,
                "wm_cal/drop_pos_rate": 0.0,
                "wm_cal/success_pos_rate": 0.0,
                "wm_cal/insufficient_samples": float(batch.get("insufficient_samples", 0.0) or 0.0),
                "wm_cal/missing_head": float(batch.get("missing_head", 0.0) or 0.0),
                "wm_cal/AUC_recovery": 0.0,
                "wm_cal/ECE": 1.0,
                "wm_cal/eval_samples": 0.0,
                "wm_cal/eval_valid": 0.0,
                "wm_cal/batch_invalid": 1.0,
                "wm_cal/failure_reason_id": float(reason),
            }
            out.update(extra)
            return out

        tensors: dict[str, torch.Tensor] = {}
        for k in required:
            t, ok = _as_cpu_float1d(batch.get(k))
            if not ok:
                return _invalid(FAILURE_NONFINITE, **{"wm_cal/invalid_input": 1.0})
            tensors[k] = t
        n = int(tensors["p_recovery"].numel())

        if n <= 0:
            return _invalid(FAILURE_MISSING_HEAD)
        if any(int(tensors[k].numel()) != n for k in required):
            return _invalid(FAILURE_SHAPE_MISMATCH, **{"wm_cal/shape_mismatch": 1.0, "wm_cal/n_samples": float(n)})
        stacked = torch.stack([tensors[k] for k in required], dim=0)
        if not bool(torch.isfinite(stacked).all()):
            return _invalid(FAILURE_NONFINITE, **{"wm_cal/nonfinite_prediction": 1.0, "wm_cal/n_samples": float(n)})

        p_rec_t = tensors["p_recovery"]
        p_succ_t = tensors["p_success"]
        p_drop_t = tensors["p_drop"]
        unc_t = tensors["uncertainty"]
        y_rec_t = tensors["y_recovery"]
        y_succ_t = tensors["y_success"]
        y_drop_t = tensors["y_drop"]

        predicted_wm_term = compute_snippet_wm_term(
            p_recovery=p_rec_t,
            p_success=p_succ_t,
            p_drop=p_drop_t,
            uncertainty=unc_t,
            w_rec=float(cfg.w_rec),
            w_success=float(cfg.w_success),
            w_drop=float(cfg.w_drop),
            w_unc=float(cfg.w_unc),
            score_cap=float(cfg.score_cap),
        )
        realized_utility = compute_realized_wm_utility(
            y_recovery=y_rec_t,
            y_success=y_succ_t,
            y_drop=y_drop_t,
            w_rec=float(cfg.w_rec),
            w_success=float(cfg.w_success),
            w_drop=float(cfg.w_drop),
        )
        combined_err = (predicted_wm_term - realized_utility).abs()
        recovery_err = (p_rec_t - y_rec_t).abs()
        pred_list = predicted_wm_term.tolist()
        util_list = realized_utility.tolist()
        p_rec = p_rec_t.tolist()
        y_rec = y_rec_t.tolist()
        p_succ = p_succ_t.tolist()
        y_succ = y_succ_t.tolist()
        p_drop = p_drop_t.tolist()
        y_drop = y_drop_t.tolist()
        unc = unc_t.tolist()
        comb_err = combined_err.tolist()
        rec_err = recovery_err.tolist()

        rank_combined = spearman_rank(pred_list, util_list)
        rank_recovery = spearman_rank(p_rec, y_rec)

        return {
            "wm_cal/auc_recovery": auc_binary(p_rec, y_rec),
            "wm_cal/auc_drop": auc_binary(p_drop, y_drop),
            "wm_cal/auc_success": auc_binary(p_succ, y_succ),
            "wm_cal/ece_recovery": ece(p_rec, y_rec),
            "wm_cal/ece_success": ece(p_succ, y_succ),
            "wm_cal/ece_drop": ece(p_drop, y_drop),
            "wm_cal/uncertainty_error_corr": uncertainty_error_corr(unc, comb_err),
            "wm_cal/combined_abs_error_mean": float(combined_err.mean().item()),
            "wm_cal/recovery_abs_error_mean": float(recovery_err.mean().item()),
            "wm_cal/recovery_uncertainty_error_corr": uncertainty_error_corr(unc, rec_err),
            "wm_cal/rank_corr_combined": float(rank_combined),
            "wm_cal/rank_corr_recovery": float(rank_recovery),
            "wm_cal/rank_corr": float(rank_combined),
            "wm_cal/combined_score_mean": float(predicted_wm_term.mean().item()),
            "wm_cal/realized_utility_mean": float(realized_utility.mean().item()),
            "wm_cal/n_samples": float(n),
            "wm_cal/recovery_pos_rate": pos_rate(y_rec),
            "wm_cal/drop_pos_rate": pos_rate(y_drop),
            "wm_cal/success_pos_rate": pos_rate(y_succ),
            "wm_cal/insufficient_samples": float(batch.get("insufficient_samples", 0.0) or 0.0),
            "wm_cal/missing_head": float(batch.get("missing_head", 0.0) or 0.0),
            "wm_cal/AUC_recovery": auc_binary(p_rec, y_rec),
            "wm_cal/ECE": ece(p_rec, y_rec),
            "wm_cal/eval_samples": float(n),
            "wm_cal/eval_valid": 1.0,
            "wm_cal/batch_invalid": 0.0,
            "wm_cal/failure_reason_id": FAILURE_OK,
        }

    def compute_alpha(
        self, metrics: dict[str, float], state: WMCalibrationState
    ) -> WMCalibrationState:
        cfg = self.cfg
        if float(metrics.get("wm_cal/batch_invalid", 0.0)) > 0.5:
            return apply_calibration_outcome(
                state,
                cfg=cfg,
                frames=int(state.last_eval_frame),
                metrics=metrics,
                mode="structural",
            )
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
        metrics["wm_cal/eval_valid"] = 1.0
        metrics["wm_cal/failure_reason_id"] = FAILURE_OK
        return state
