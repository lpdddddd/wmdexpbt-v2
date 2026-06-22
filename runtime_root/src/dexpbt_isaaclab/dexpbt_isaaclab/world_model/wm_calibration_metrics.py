"""Pure calibration metrics for CSER-v0 WM evaluator."""

from __future__ import annotations

import math


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
