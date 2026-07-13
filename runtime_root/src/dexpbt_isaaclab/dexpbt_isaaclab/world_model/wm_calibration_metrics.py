"""Pure calibration metrics for CSER-v0 WM evaluator."""

from __future__ import annotations

import math
from typing import Any, Optional

import torch

from .seq_wm_base import SequenceBatch


def clamp01(x: float) -> float:
    if not math.isfinite(float(x)):
        return 0.0
    return max(0.0, min(1.0, float(x)))


def auc_binary(probs: list[float], labels: list[float]) -> float:
    if not probs or len(probs) != len(labels):
        return 0.0
    pairs = sorted(zip(probs, labels), key=lambda x: x[0])
    n_pos = sum(1 for _, y in pairs if y > 0.5)
    n_neg = len(pairs) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5
    rank_sum = 0.0
    for i, (_, y) in enumerate(pairs, start=1):
        if y > 0.5:
            rank_sum += i
    auc = (rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return clamp01(float(auc))


def ece(probs: list[float], labels: list[float], n_bins: int = 10) -> float:
    if not probs or len(probs) != len(labels):
        return 1.0
    bins = [[] for _ in range(max(1, int(n_bins)))]
    for p, y in zip(probs, labels):
        idx = min(int(p * n_bins), n_bins - 1) if p < 1.0 else n_bins - 1
        bins[idx].append((float(p), float(y)))
    total = 0.0
    n = float(len(probs))
    for bucket in bins:
        if not bucket:
            continue
        conf = sum(p for p, _ in bucket) / len(bucket)
        acc = sum(y for _, y in bucket) / len(bucket)
        total += (len(bucket) / n) * abs(conf - acc)
    return float(total)


def spearman_rank(x: list[float], y: list[float]) -> float:
    if not x or len(x) != len(y) or len(x) < 2:
        return 0.0
    n = len(x)

    def _rank(vals):
        order = sorted(range(n), key=lambda i: vals[i])
        ranks = [0.0] * n
        for r, i in enumerate(order, start=1):
            ranks[i] = float(r)
        return ranks

    rx, ry = _rank(x), _rank(y)
    mx = sum(rx) / n
    my = sum(ry) / n
    num = sum((rx[i] - mx) * (ry[i] - my) for i in range(n))
    den_x = math.sqrt(sum((rx[i] - mx) ** 2 for i in range(n)))
    den_y = math.sqrt(sum((ry[i] - my) ** 2 for i in range(n)))
    if den_x <= 1e-12 or den_y <= 1e-12:
        return 0.0
    return float(num / (den_x * den_y))


def uncertainty_error_corr(uncertainty: list[float], errors: list[float]) -> float:
    if not uncertainty or len(uncertainty) != len(errors):
        return 0.0
    n = len(uncertainty)
    mu_u = sum(uncertainty) / n
    mu_e = sum(errors) / n
    num = sum((uncertainty[i] - mu_u) * (errors[i] - mu_e) for i in range(n))
    den_u = math.sqrt(sum((uncertainty[i] - mu_u) ** 2 for i in range(n)))
    den_e = math.sqrt(sum((errors[i] - mu_e) ** 2 for i in range(n)))
    if den_u <= 1e-12 or den_e <= 1e-12:
        return 0.0
    return float(num / (den_u * den_e))


def pos_rate(labels: list[float]) -> float:
    if not labels:
        return 0.0
    return float(sum(1 for y in labels if float(y) > 0.5) / len(labels))


def prob_uncertainty(probs: torch.Tensor) -> torch.Tensor:
    """Uncertainty proxy in [0, 1]: high when probability is near 0.5.

    Thin alias of ``binary_head_uncertainty`` for backward compatibility.
    Prefer ``combined_head_uncertainty`` when recovery/success/drop are all used.
    """
    from .snippet_wm_reranker import binary_head_uncertainty

    return binary_head_uncertainty(probs)


def resolve_wm_calibration_eval_tol(
    tol: torch.Tensor,
    *,
    valid: torch.Tensor,
    future_ok: torch.Tensor,
    floor_tol_threshold: float,
    mode: str = "hybrid",
    fixed_eval_tol: float = 0.05,
    quantile: float = 0.5,
    margin: float = 1.1,
    cap: float = 0.06,
    min_samples_for_stats: int = 32,
) -> tuple[float, dict[str, float]]:
    """Resolve tolerance cutoff for calibration ``eval_mask`` (A / B / hybrid)."""
    mode_norm = str(mode or "hybrid").strip().lower()
    mode_ids = {"fixed": 0.0, "adaptive_quantile": 1.0, "hybrid": 2.0}
    floor = float(floor_tol_threshold)
    fixed_t = max(floor, float(fixed_eval_tol))

    def _adaptive_tol() -> float:
        stats_mask = (valid > 0.5) & (future_ok > 0.5)
        n_stats = int(stats_mask.sum().item())
        if n_stats >= int(min_samples_for_stats):
            vals = tol[stats_mask]
        else:
            vals = tol.reshape(-1)
        if int(vals.numel()) <= 0:
            q = floor
        else:
            q = float(vals.quantile(float(quantile)).item())
        adaptive = max(floor, q * float(margin))
        return min(adaptive, float(cap))

    if mode_norm == "fixed":
        eval_tol = fixed_t
    elif mode_norm == "adaptive_quantile":
        eval_tol = _adaptive_tol()
    elif mode_norm == "hybrid":
        eval_tol = min(max(fixed_t, _adaptive_tol()), float(cap))
    else:
        eval_tol = fixed_t
        mode_norm = "fixed"

    diag = {
        "wm_cal/eval_tol_mode_id": float(mode_ids.get(mode_norm, 2.0)),
        "wm_cal/eval_tol_threshold": float(eval_tol),
        "wm_cal/eval_tol_fixed": float(fixed_t),
        "wm_cal/eval_tol_cap": float(cap),
    }
    stats_mask = (valid > 0.5) & (future_ok > 0.5)
    if int(stats_mask.sum().item()) > 0:
        diag["wm_cal/eval_tol_quantile_value"] = float(
            tol[stats_mask].quantile(float(quantile)).item()
        )
    return float(eval_tol), diag


def resolve_calibration_horizon(
    horizon_steps: tuple[int, ...] | list[int],
    *,
    default: int = 4,
) -> int:
    steps = [int(h) for h in horizon_steps if int(h) > 0]
    if not steps:
        return int(default)
    at_least = [h for h in steps if h >= int(default)]
    if at_least:
        return min(at_least)
    return min(steps)


def build_wm_calibration_targets(
    batch: SequenceBatch,
    *,
    horizon: int,
    improve_margin: float,
    floor_tol_threshold: float,
    eval_tol_mode: str = "hybrid",
    eval_tol_threshold: float = 0.05,
    eval_tol_quantile: float = 0.5,
    eval_tol_margin: float = 1.1,
    eval_tol_cap: float = 0.06,
    eval_tol_min_samples_for_stats: int = 32,
) -> tuple[dict[str, torch.Tensor], dict[str, float]]:
    """Build flat calibration labels aligned with ``belief.reshape(T * B, -1)``."""
    t_len, b_len = batch.T, batch.B
    device = batch.device
    dtype = torch.float32
    n_flat = t_len * b_len
    z = torch.zeros(n_flat, device=device, dtype=dtype)
    empty_targets = {
        "eval_mask": torch.zeros(n_flat, device=device, dtype=torch.bool),
        "y_recovery": z.clone(),
        "y_success": z.clone(),
        "y_drop": z.clone(),
        "y_improve": z.clone(),
    }
    empty_diag = {"wm_cal/eval_tol_threshold": float(floor_tol_threshold)}
    if batch.d_hat is None:
        return empty_targets, empty_diag

    d_hat = batch.d_hat
    tol = batch.success_tolerance_seq
    if tol is None:
        tol = torch.full((t_len, b_len), float(floor_tol_threshold), device=device, dtype=d_hat.dtype)
    valid = batch.valid_mask
    if valid is None:
        valid = torch.ones(t_len, b_len, device=device, dtype=d_hat.dtype)
    seg = batch.segment_id
    if seg is None:
        seg = torch.zeros(t_len, b_len, device=device, dtype=torch.long)

    h_steps = max(1, int(horizon))
    min_future = d_hat.clone()
    future_ok = torch.zeros(t_len, b_len, device=device, dtype=dtype)
    y_drop = torch.zeros(t_len, b_len, device=device, dtype=dtype)

    for h in range(1, h_steps + 1):
        if h >= t_len:
            break
        same_seg = seg[h:, :] == seg[:-h, :]
        step_ok = (valid[h:, :] > 0.5) & (valid[:-h, :] > 0.5) & same_seg
        future_ok[:-h, :] = torch.where(
            step_ok,
            torch.ones_like(future_ok[:-h, :]),
            future_ok[:-h, :],
        )
        min_future[:-h, :] = torch.where(
            step_ok,
            torch.minimum(min_future[:-h, :], d_hat[h:, :]),
            min_future[:-h, :],
        )
        if batch.dropped is not None:
            drop_h = (batch.dropped[h:, :] > 0.5) & step_ok
            y_drop[:-h, :] = torch.where(drop_h, torch.ones_like(y_drop[:-h, :]), y_drop[:-h, :])
        if batch.failure is not None:
            fail_h = (batch.failure[h:, :] > 0.5) & step_ok
            y_drop[:-h, :] = torch.where(fail_h, torch.ones_like(y_drop[:-h, :]), y_drop[:-h, :])
        done_h = (batch.done[h:, :] > 0.5) & step_ok
        y_drop[:-h, :] = torch.where(done_h, torch.ones_like(y_drop[:-h, :]), y_drop[:-h, :])
        worsen = (d_hat[h:, :] > d_hat[:-h, :] + float(improve_margin)) & step_ok
        y_drop[:-h, :] = torch.where(worsen, torch.ones_like(y_drop[:-h, :]), y_drop[:-h, :])

    y_success = ((min_future <= tol) & (future_ok > 0.5)).float()
    y_improve = (((d_hat - min_future) >= float(improve_margin)) & (future_ok > 0.5)).float()
    y_recovery = ((y_success > 0.5) | (y_improve > 0.5)) & (y_drop < 0.5)
    y_recovery = y_recovery.float()
    eval_tol, tol_diag = resolve_wm_calibration_eval_tol(
        tol,
        valid=valid,
        future_ok=future_ok,
        floor_tol_threshold=float(floor_tol_threshold),
        mode=str(eval_tol_mode),
        fixed_eval_tol=float(eval_tol_threshold),
        quantile=float(eval_tol_quantile),
        margin=float(eval_tol_margin),
        cap=float(eval_tol_cap),
        min_samples_for_stats=int(eval_tol_min_samples_for_stats),
    )
    eval_mask = (tol <= float(eval_tol)) & (future_ok > 0.5)
    tol_diag["wm_cal/eval_mask_frac"] = float(eval_mask.float().mean().item())

    def _flat(x: torch.Tensor) -> torch.Tensor:
        return x.reshape(t_len * b_len)

    targets = {
        "eval_mask": _flat(eval_mask).bool(),
        "y_recovery": _flat(y_recovery),
        "y_success": _flat(y_success),
        "y_drop": _flat(y_drop),
        "y_improve": _flat(y_improve),
    }
    return targets, tol_diag


def extract_wm_calibration_predictions(
    heads: Any,
    *,
    belief_flat: torch.Tensor,
    action_flat: torch.Tensor,
    horizon: int,
) -> tuple[Optional[dict[str, torch.Tensor]], dict[str, float]]:
    """Return WM head probabilities for calibration eval or None when heads missing."""
    from .snippet_wm_reranker import combined_head_uncertainty

    diag: dict[str, float] = {"wm_cal/missing_head": 0.0}
    if heads is None or not bool(getattr(heads, "has_recovery_head", False)):
        diag["wm_cal/missing_head"] = 1.0
        return None, diag

    preds = heads.predict_transition(belief_flat, action_flat)
    rec = preds.get("recovery_logits")
    if rec is None:
        diag["wm_cal/missing_head"] = 1.0
        return None, diag

    p_rec = torch.sigmoid(rec).reshape(-1)
    p_succ = torch.zeros_like(p_rec)
    p_drop = torch.zeros_like(p_rec)
    h_logits = preds.get("horizon_logits")
    if isinstance(h_logits, dict) and h_logits:
        h_use = int(horizon)
        if h_use not in h_logits:
            h_use = min(h_logits.keys(), key=lambda x: abs(int(x) - int(horizon)))
        head_out = h_logits.get(h_use)
        if isinstance(head_out, dict):
            if "success" in head_out:
                p_succ = torch.sigmoid(head_out["success"]).reshape(-1)
            else:
                diag["wm_cal/missing_head"] = 1.0
            if "drop" in head_out:
                p_drop = torch.sigmoid(head_out["drop"]).reshape(-1)
            else:
                diag["wm_cal/missing_head"] = 1.0
        else:
            diag["wm_cal/missing_head"] = 1.0
    else:
        diag["wm_cal/missing_head"] = 1.0

    valid = (
        torch.isfinite(p_rec)
        & torch.isfinite(p_succ)
        & torch.isfinite(p_drop)
    )
    p_rec_safe = torch.where(valid, p_rec, torch.zeros_like(p_rec))
    p_succ_safe = torch.where(valid, p_succ, torch.zeros_like(p_succ))
    p_drop_safe = torch.where(valid, p_drop, torch.zeros_like(p_drop))
    unc = combined_head_uncertainty(p_rec_safe, p_succ_safe, p_drop_safe)
    unc = torch.where(valid, unc, torch.ones_like(unc))
    return {
        "p_recovery": p_rec_safe,
        "p_success": p_succ_safe,
        "p_drop": p_drop_safe,
        "uncertainty": unc,
        "valid_mask": valid,
    }, diag
