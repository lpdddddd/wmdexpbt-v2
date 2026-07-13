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


def _to_float_list(values) -> list[float]:
    if values is None:
        return []
    if isinstance(values, torch.Tensor):
        t = values.detach().float().reshape(-1).cpu()
        if t.numel() == 0:
            return []
        out = []
        for v in t.tolist():
            fv = float(v)
            if not math.isfinite(fv):
                return []  # signal nonfinite via empty + caller checks
            out.append(fv)
        # Mark nonfinite by sentinel handled by callers: we filter below.
        return out
    try:
        seq = list(values)
    except TypeError:
        return []
    out = []
    for v in seq:
        try:
            fv = float(v)
        except (TypeError, ValueError):
            return []
        if not math.isfinite(fv):
            return []
        out.append(fv)
    return out


def _average_ranks(vals: list[float]) -> list[float]:
    """1-based average ranks with ties (permutation-invariant)."""
    n = len(vals)
    order = sorted(range(n), key=lambda i: vals[i])
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and vals[order[j + 1]] == vals[order[i]]:
            j += 1
        avg = 0.5 * (float(i + 1) + float(j + 1))
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def auc_binary(probs, labels) -> float:
    """Mann-Whitney AUC with average ranks for tied predictions.

    All-tied predictions with both classes present return 0.5. Empty /
    length-mismatch / nonfinite inputs return 0.0. Single-class labels return 0.5.
    Result is invariant to joint permutation of (probs, labels).
    """
    p = _to_float_list(probs)
    y = _to_float_list(labels)
    if not p or len(p) != len(y):
        # Distinguish nonfinite-in-tensor (empty from _to_float_list on bad) vs true empty.
        if isinstance(probs, torch.Tensor) and probs.numel() > 0:
            flat = probs.detach().float().reshape(-1)
            if flat.numel() and not bool(torch.isfinite(flat).all()):
                return 0.0
        if isinstance(labels, torch.Tensor) and labels.numel() > 0:
            flat = labels.detach().float().reshape(-1)
            if flat.numel() and not bool(torch.isfinite(flat).all()):
                return 0.0
        if not p or not y or len(p) != len(y):
            return 0.0
    n_pos = sum(1 for v in y if v > 0.5)
    n_neg = len(y) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5
    ranks = _average_ranks(p)
    rank_sum = sum(ranks[i] for i in range(len(y)) if y[i] > 0.5)
    auc = (rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return clamp01(float(auc))


def ece(probs, labels, n_bins: int = 10) -> float:
    p = _to_float_list(probs)
    y = _to_float_list(labels)
    if not p or len(p) != len(y):
        return 1.0
    bins = [[] for _ in range(max(1, int(n_bins)))]
    for pv, yv in zip(p, y):
        idx = min(int(pv * n_bins), n_bins - 1) if pv < 1.0 else n_bins - 1
        bins[idx].append((float(pv), float(yv)))
    total = 0.0
    n = float(len(p))
    for bucket in bins:
        if not bucket:
            continue
        conf = sum(a for a, _ in bucket) / len(bucket)
        acc = sum(b for _, b in bucket) / len(bucket)
        total += (len(bucket) / n) * abs(conf - acc)
    return float(total)


def spearman_rank(x, y) -> float:
    """Spearman rho with tie-aware average ranks; constant series -> 0.

    Empty / length mismatch / nonfinite / n<2 -> 0. Invariant to joint
    permutation of (x, y).
    """
    xs = _to_float_list(x)
    ys = _to_float_list(y)
    if not xs or len(xs) != len(ys) or len(xs) < 2:
        return 0.0
    n = len(xs)
    rx = _average_ranks(xs)
    ry = _average_ranks(ys)
    mx = sum(rx) / n
    my = sum(ry) / n
    num = sum((rx[i] - mx) * (ry[i] - my) for i in range(n))
    den_x = math.sqrt(sum((rx[i] - mx) ** 2 for i in range(n)))
    den_y = math.sqrt(sum((ry[i] - my) ** 2 for i in range(n)))
    if den_x <= 1e-12 or den_y <= 1e-12:
        return 0.0
    return float(num / (den_x * den_y))


def uncertainty_error_corr(uncertainty, errors) -> float:
    u = uncertainty if isinstance(uncertainty, list) else None
    e = errors if isinstance(errors, list) else None
    if u is None:
        if isinstance(uncertainty, torch.Tensor):
            if not bool(torch.isfinite(uncertainty.detach().float().reshape(-1)).all()):
                return 0.0
            u = uncertainty.detach().float().reshape(-1).cpu().tolist()
        else:
            try:
                u = [float(x) for x in uncertainty]
            except TypeError:
                return 0.0
    if e is None:
        if isinstance(errors, torch.Tensor):
            if not bool(torch.isfinite(errors.detach().float().reshape(-1)).all()):
                return 0.0
            e = errors.detach().float().reshape(-1).cpu().tolist()
        else:
            try:
                e = [float(x) for x in errors]
            except TypeError:
                return 0.0
    if not u or len(u) != len(e):
        return 0.0
    n = len(u)
    mu_u = sum(u) / n
    mu_e = sum(e) / n
    num = sum((u[i] - mu_u) * (e[i] - mu_e) for i in range(n))
    den_u = math.sqrt(sum((u[i] - mu_u) ** 2 for i in range(n)))
    den_e = math.sqrt(sum((e[i] - mu_e) ** 2 for i in range(n)))
    if den_u <= 1e-12 or den_e <= 1e-12:
        return 0.0
    return float(num / (den_u * den_e))


def pos_rate(labels) -> float:
    if labels is None:
        return 0.0
    if isinstance(labels, torch.Tensor):
        vals = labels.detach().float().reshape(-1)
        if int(vals.numel()) == 0:
            return 0.0
        return float((vals > 0.5).float().mean().item())
    try:
        seq = list(labels)
    except TypeError:
        return 0.0
    if len(seq) == 0:
        return 0.0
    return float(sum(1 for y in seq if float(y) > 0.5) / len(seq))


def prob_uncertainty(probs: torch.Tensor) -> torch.Tensor:
    """Uncertainty proxy in [0, 1]: high when probability is near 0.5.

    Thin alias of ``binary_head_uncertainty`` for backward compatibility.
    Prefer ``combined_head_uncertainty`` when recovery/success/drop are all used.
    """
    from .wm_head_probs import binary_head_uncertainty

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
    horizon_steps: tuple[int, ...] | list[int] | None = None,
    recovery_horizon: int,
    improve_margin: float,
    recovery_d_hat_tol_min_steps: int = 0,
    floor_tol_threshold: float,
    recovery_shadow_tol_threshold: float | None = None,
    # Legacy unused kwargs kept for call-site compat (eval_tol filters dropped;
    # recovery floor mask is the authoritative stage gate).
    horizon: int | None = None,
    eval_tol_mode: str = "hybrid",
    eval_tol_threshold: float = 0.05,
    eval_tol_quantile: float = 0.5,
    eval_tol_margin: float = 1.1,
    eval_tol_cap: float = 0.06,
    eval_tol_min_samples_for_stats: int = 32,
) -> tuple[dict[str, torch.Tensor], dict[str, float]]:
    """Build flat calibration labels from the **same** training target builders.

    Horizon semantics
    -----------------
    - ``y_success`` / ``y_drop``: mean over configured ``horizon_steps`` of
      ``build_horizon_success_target`` / ``build_horizon_drop_target``
      (identical to RSSM head training labels, then averaged like runtime probs).
    - ``y_recovery`` / recovery mask: ``build_recovery_target_soft_or`` with
      ``recovery_horizon`` (not the primary success horizon). Drop does **not**
      clear recovery positives; done / d_hat-worsen are **not** drop labels.
    - ``eval_mask``: intersection of all horizon valid masks AND recovery mask.
      Prediction validity is applied later by the trainer.
    - Missing required target sources → empty eval_mask +
      ``wm_cal/missing_target_source=1`` (fail-closed; no fake all-zero labels).

    Note: env-slot held-out partition is applied by ``SeqWMTrainer`` /
    ``SequenceSampler`` (not here). This builder only forms labels.
    """
    from .wm_head_probs import (
        FAILURE_MISSING_TARGET_SOURCE,
        normalize_horizon_steps,
    )
    from .wm_head_targets import (
        build_horizon_drop_target,
        build_horizon_success_target,
        build_horizon_valid_mask,
        build_recovery_target_soft_or,
        has_horizon_drop_sources,
        has_horizon_success_sources,
        has_recovery_sources,
    )

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
        "recovery_mask": torch.zeros(n_flat, device=device, dtype=torch.bool),
    }
    # Legacy eval_tol kwargs ignored (recovery floor mask is authoritative).
    _ = (eval_tol_mode, eval_tol_threshold, eval_tol_quantile, eval_tol_margin,
         eval_tol_cap, eval_tol_min_samples_for_stats, horizon)

    if horizon_steps is None:
        steps = normalize_horizon_steps((4, 8))
    else:
        steps = normalize_horizon_steps(horizon_steps)

    diag: dict[str, float] = {
        "wm_cal/eval_tol_threshold": float(floor_tol_threshold),
        "wm_cal/horizon_count": float(len(steps)),
        "wm_cal/recovery_horizon": float(max(1, int(recovery_horizon))),
        "wm_cal/missing_target_source": 0.0,
        "wm_cal/failure_reason_id": 0.0,
    }

    if not has_horizon_success_sources(batch):
        diag["wm_cal/missing_target_source"] = 1.0
        diag["wm_cal/failure_reason_id"] = FAILURE_MISSING_TARGET_SOURCE
        diag["wm_cal/missing_success_source"] = 1.0
        return empty_targets, diag
    if not has_horizon_drop_sources(batch):
        diag["wm_cal/missing_target_source"] = 1.0
        diag["wm_cal/failure_reason_id"] = FAILURE_MISSING_TARGET_SOURCE
        diag["wm_cal/missing_drop_source"] = 1.0
        return empty_targets, diag
    if not has_recovery_sources(batch):
        diag["wm_cal/missing_target_source"] = 1.0
        diag["wm_cal/failure_reason_id"] = FAILURE_MISSING_TARGET_SOURCE
        diag["wm_cal/missing_recovery_source"] = 1.0
        return empty_targets, diag

    succ_stack = []
    drop_stack = []
    mask_stack = []
    for h in steps:
        succ_stack.append(build_horizon_success_target(batch, h).float())
        drop_stack.append(build_horizon_drop_target(batch, h).float())
        mask_stack.append(build_horizon_valid_mask(batch, h).float())

    y_success = torch.stack(succ_stack, dim=0).mean(dim=0)
    y_drop = torch.stack(drop_stack, dim=0).mean(dim=0)
    horizon_mask_all = torch.stack(mask_stack, dim=0).amin(dim=0)

    shadow = (
        float(floor_tol_threshold)
        if recovery_shadow_tol_threshold is None
        else float(recovery_shadow_tol_threshold)
    )
    y_recovery, recovery_mask_f = build_recovery_target_soft_or(
        batch,
        horizon=int(max(1, recovery_horizon)),
        d_hat_margin=float(improve_margin),
        d_hat_tol_min_steps=int(recovery_d_hat_tol_min_steps),
        floor_tol_threshold=shadow,
    )
    # Diagnostic only: d_hat margin component (not used for alpha gates).
    y_improve = torch.zeros_like(y_recovery)
    if batch.d_hat is not None:
        h_rec = int(max(1, recovery_horizon))
        for t in range(t_len):
            t_end = min(t_len - 1, t + h_rec - 1)
            y_improve[t] = (
                (batch.d_hat[t] - batch.d_hat[t_end]) >= float(improve_margin)
            ).to(y_improve.dtype)

    eval_mask = (horizon_mask_all > 0.5) & (recovery_mask_f > 0.5)
    diag["wm_cal/eval_mask_frac"] = float(eval_mask.float().mean().item())
    diag["wm_cal/recovery_mask_frac"] = float((recovery_mask_f > 0.5).float().mean().item())

    def _flat(x: torch.Tensor) -> torch.Tensor:
        return x.reshape(t_len * b_len)

    targets = {
        "eval_mask": _flat(eval_mask).bool(),
        "y_recovery": _flat(y_recovery.float()),
        "y_success": _flat(y_success.float()),
        "y_drop": _flat(y_drop.float()),
        "y_improve": _flat(y_improve.float()),
        "recovery_mask": _flat(recovery_mask_f > 0.5).bool(),
    }
    return targets, diag


def extract_wm_calibration_predictions(
    heads: Any,
    *,
    belief_flat: torch.Tensor,
    action_flat: torch.Tensor,
    horizon: int | None = None,
    horizon_steps: tuple[int, ...] | list[int] | None = None,
) -> tuple[Optional[dict[str, torch.Tensor]], dict[str, float]]:
    """Return WM head probabilities for calibration via shared extractor.

    Required heads: recovery + success/drop for every configured horizon.
    Missing / shape / horizon mismatch => ``None`` (structural failure).
    Per-row nonfinite => valid_mask=False (excluded from metrics by caller).
    """
    from .wm_head_probs import (
        FAILURE_MISSING_HEAD,
        extract_required_wm_head_probs,
        normalize_horizon_steps,
    )

    if horizon_steps is None:
        steps = normalize_horizon_steps((int(horizon),) if horizon is not None else (4, 8))
    else:
        steps = normalize_horizon_steps(horizon_steps)

    diag: dict[str, float] = {
        "wm_cal/missing_head": 0.0,
        "wm_cal/failure_reason_id": 0.0,
        "wm_cal/eval_valid": 0.0,
    }
    if heads is None or not bool(getattr(heads, "has_recovery_head", False)):
        diag["wm_cal/missing_head"] = 1.0
        diag["wm_cal/failure_reason_id"] = FAILURE_MISSING_HEAD
        return None, diag

    try:
        preds = heads.predict_transition(belief_flat, action_flat)
    except Exception:
        from .wm_head_probs import FAILURE_PREDICT_EXCEPTION
        diag["wm_cal/predict_exception"] = 1.0
        diag["wm_cal/failure_reason_id"] = FAILURE_PREDICT_EXCEPTION
        return None, diag

    n = int(belief_flat.shape[0]) if isinstance(belief_flat, torch.Tensor) else None
    result = extract_required_wm_head_probs(
        preds,
        horizon_steps=steps,
        expected_batch=n,
        device=belief_flat.device if isinstance(belief_flat, torch.Tensor) else None,
        dtype=torch.float32,
    )
    diag.update(result.diagnostics)
    if not result.structural_ok:
        # Do not return zero-filled "valid" predictions for missing required heads.
        return None, diag
    # If caller previously set missing_head while still returning tensors, that
    # was unsafe; structural_ok False already returned None above.
    if float(diag.get("wm_cal/missing_head", 0.0)) > 0.5:
        return None, diag

    return {
        "p_recovery": result.p_recovery,
        "p_success": result.p_success,
        "p_drop": result.p_drop,
        "uncertainty": result.uncertainty,
        "valid_mask": result.valid_mask,
    }, diag
