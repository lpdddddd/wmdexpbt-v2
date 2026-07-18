#!/usr/bin/env python3
"""CCRS analysis: R(k, tau) tables, D/F/C decomposition, paired bootstrap.

Definitions (tolerances in meters):

  R(k, tau) = P(error <= tau | policy checkpoint k)
            ~ fixed-tolerance success rate under identical reset seeds.

For a curriculum transition with pre checkpoint p (live tol a) and post
checkpoint q (live tol b, b < a):

  D = R(p, a) - R(q, b)                      # observed live drop
  F = 0.5 * [(R(p,a) - R(q,a)) + (R(p,b) - R(q,b))]   # policy-update effect
  C = 0.5 * [(R(p,a) - R(p,b)) + (R(q,a) - R(q,b))]   # criterion effect

  Identity: D = F + C (exact, by construction).

Paired bootstrap: because every (checkpoint, tolerance) cell is evaluated
with the SAME reset seed set, we resample SEEDS (not episodes across cells)
so that each bootstrap replicate re-computes all four cells R(p,a), R(p,b),
R(q,a), R(q,b) from the same resampled seed multiset. Unpaired data
(mismatched seed sets) is rejected loudly.

Success definition: episode success = (nsucc > 0) under the fixed tolerance,
consistent with eval_protocol.make_result_row. A per-seed cell value is the
success_rate of that seed's episodes; R is the mean over seeds.
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

METRIC_SUCCESS_RATE = "success_rate"
METRIC_NSUCC_MEAN = "nsucc_mean"


# ---------------------------------------------------------------------------
# Result loading / indexing
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CellKey:
    worker_id: int
    checkpoint_iteration: int
    fixed_tolerance: float

    def rounded(self) -> "CellKey":
        return CellKey(self.worker_id, self.checkpoint_iteration,
                       round(self.fixed_tolerance, 6))


class CCRSTable:
    """Index of per-seed results: (worker, iter, tol) -> {seed: row}."""

    def __init__(self) -> None:
        self.cells: dict[CellKey, dict[int, dict]] = {}
        self.meta: dict[CellKey, dict] = {}

    @staticmethod
    def from_jsonl(path: Path) -> "CCRSTable":
        t = CCRSTable()
        if not path.is_file():
            return t
        with path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                t.add_row(row)
        return t

    def add_row(self, row: dict) -> None:
        if row.get("status") != "ok":
            return
        key = CellKey(
            int(row["worker_id"]),
            int(row["checkpoint_iteration"]),
            round(float(row["fixed_tolerance"]), 6),
        )
        seed = int(row["reset_seed"])
        self.cells.setdefault(key, {})[seed] = row
        self.meta.setdefault(key, {
            "checkpoint_frame": row.get("checkpoint_frame"),
            "checkpoint_path": row.get("checkpoint_path"),
        })

    def seeds(self, key: CellKey) -> set[int]:
        return set(self.cells.get(key.rounded(), {}).keys())

    def per_seed_metric(self, key: CellKey, metric: str = METRIC_SUCCESS_RATE) -> dict[int, float]:
        out = {}
        for seed, row in self.cells.get(key.rounded(), {}).items():
            v = row.get(metric)
            if v is not None and not (isinstance(v, float) and math.isnan(v)):
                out[seed] = float(v)
        return out

    def R(self, key: CellKey, metric: str = METRIC_SUCCESS_RATE,
          seeds: Optional[Iterable[int]] = None) -> float:
        """Mean metric over seeds (optionally restricted / with multiplicity)."""
        per_seed = self.per_seed_metric(key, metric)
        if seeds is None:
            vals = list(per_seed.values())
        else:
            vals = [per_seed[s] for s in seeds if s in per_seed]
        if not vals:
            return float("nan")
        return sum(vals) / len(vals)

    def common_seeds(self, keys: Iterable[CellKey]) -> set[int]:
        keys = list(keys)
        if not keys:
            return set()
        s = self.seeds(keys[0])
        for k in keys[1:]:
            s &= self.seeds(k)
        return s

    def is_empty(self) -> bool:
        return not self.cells


# ---------------------------------------------------------------------------
# D/F/C decomposition
# ---------------------------------------------------------------------------

@dataclass
class DFCResult:
    worker_id: int
    pre_iteration: int
    post_iteration: int
    tol_pre: float     # a (meters)
    tol_post: float    # b (meters), b < a
    R_pa: float
    R_pb: float
    R_qa: float
    R_qb: float
    D: float
    F: float
    C: float
    identity_residual: float          # |D - (F + C)|, must be ~0
    n_seeds: int
    ci: dict = field(default_factory=dict)  # bootstrap CIs per component


def dfc_from_R(R_pa: float, R_pb: float, R_qa: float, R_qb: float) -> tuple[float, float, float]:
    D = R_pa - R_qb
    F = 0.5 * ((R_pa - R_qa) + (R_pb - R_qb))
    C = 0.5 * ((R_pa - R_pb) + (R_qa - R_qb))
    return D, F, C


def paired_bootstrap_ci(
    per_seed_cells: dict[str, dict[int, float]],
    stat_fn: Callable[[dict[str, float]], dict[str, float]],
    n_boot: int = 2000,
    seed: int = 1234,
    alpha: float = 0.05,
) -> dict[str, tuple[float, float]]:
    """Percentile bootstrap over SEEDS shared by all cells.

    per_seed_cells: name -> {seed: value}. All cells must share the same seed
    set; raises ValueError otherwise (unpaired data must not silently pass).
    stat_fn: maps {name: mean_value_over_resampled_seeds} to named statistics.
    Returns {stat_name: (lo, hi)}.
    """
    names = list(per_seed_cells.keys())
    if not names:
        return {}
    seed_sets = [set(per_seed_cells[n].keys()) for n in names]
    common = set.intersection(*seed_sets)
    union = set.union(*seed_sets)
    if common != union:
        raise ValueError(
            f"unpaired seed sets across cells: common={sorted(common)} "
            f"union={sorted(union)}; refusing to bootstrap unpaired data"
        )
    seeds = sorted(common)
    if len(seeds) < 2:
        raise ValueError(f"need >=2 shared seeds for bootstrap, got {len(seeds)}")

    rng = random.Random(seed)
    samples: dict[str, list[float]] = {}
    for _ in range(n_boot):
        draw = [seeds[rng.randrange(len(seeds))] for _ in seeds]
        means = {
            n: sum(per_seed_cells[n][s] for s in draw) / len(draw)
            for n in names
        }
        stats = stat_fn(means)
        for k, v in stats.items():
            samples.setdefault(k, []).append(v)

    out: dict[str, tuple[float, float]] = {}
    for k, vals in samples.items():
        vals.sort()
        lo_i = int((alpha / 2) * (len(vals) - 1))
        hi_i = int((1 - alpha / 2) * (len(vals) - 1))
        out[k] = (vals[lo_i], vals[hi_i])
    return out


def compute_dfc(
    table: CCRSTable,
    worker_id: int,
    pre_iteration: int,
    post_iteration: int,
    tol_pre: float,
    tol_post: float,
    metric: str = METRIC_SUCCESS_RATE,
    n_boot: int = 2000,
) -> DFCResult:
    """D/F/C for one transition using evaluated cells at tolerances a and b.

    Note: a and b must both exist in the sweep tolerance grid (the evaluator
    snaps live tolerances to the nearest grid value at planning time).
    """
    if not tol_post < tol_pre:
        raise ValueError(f"expected tol_post < tol_pre, got {tol_post} >= {tol_pre}")
    kpa = CellKey(worker_id, pre_iteration, tol_pre)
    kpb = CellKey(worker_id, pre_iteration, tol_post)
    kqa = CellKey(worker_id, post_iteration, tol_pre)
    kqb = CellKey(worker_id, post_iteration, tol_post)
    cells = {
        "pa": table.per_seed_metric(kpa, metric),
        "pb": table.per_seed_metric(kpb, metric),
        "qa": table.per_seed_metric(kqa, metric),
        "qb": table.per_seed_metric(kqb, metric),
    }
    missing = [n for n, c in cells.items() if not c]
    if missing:
        raise KeyError(
            f"missing evaluated cells {missing} for worker={worker_id} "
            f"iters=({pre_iteration},{post_iteration}) tols=({tol_pre},{tol_post})"
        )
    common = set.intersection(*(set(c.keys()) for c in cells.values()))
    if not common:
        raise ValueError("no shared seeds across the four D/F/C cells")
    seeds = sorted(common)
    R_pa = sum(cells["pa"][s] for s in seeds) / len(seeds)
    R_pb = sum(cells["pb"][s] for s in seeds) / len(seeds)
    R_qa = sum(cells["qa"][s] for s in seeds) / len(seeds)
    R_qb = sum(cells["qb"][s] for s in seeds) / len(seeds)
    D, F, C = dfc_from_R(R_pa, R_pb, R_qa, R_qb)

    ci: dict = {}
    if len(seeds) >= 2:
        restricted = {n: {s: c[s] for s in seeds} for n, c in cells.items()}

        def _stat(means: dict[str, float]) -> dict[str, float]:
            d, f, c = dfc_from_R(means["pa"], means["pb"], means["qa"], means["qb"])
            return {"D": d, "F": f, "C": c}

        ci = paired_bootstrap_ci(restricted, _stat, n_boot=n_boot)

    return DFCResult(
        worker_id=worker_id,
        pre_iteration=pre_iteration,
        post_iteration=post_iteration,
        tol_pre=tol_pre,
        tol_post=tol_post,
        R_pa=R_pa, R_pb=R_pb, R_qa=R_qa, R_qb=R_qb,
        D=D, F=F, C=C,
        identity_residual=abs(D - (F + C)),
        n_seeds=len(seeds),
        ci=ci,
    )


# ---------------------------------------------------------------------------
# Capability curves & pairwise comparison
# ---------------------------------------------------------------------------

@dataclass
class CurvePoint:
    tolerance: float
    value: float
    ci_lo: float
    ci_hi: float
    n_seeds: int


def capability_curve(
    table: CCRSTable,
    worker_id: int,
    checkpoint_iteration: int,
    tolerances: Iterable[float],
    metric: str = METRIC_SUCCESS_RATE,
    n_boot: int = 2000,
) -> list[CurvePoint]:
    pts: list[CurvePoint] = []
    for tol in tolerances:
        key = CellKey(worker_id, checkpoint_iteration, tol)
        per_seed = table.per_seed_metric(key, metric)
        if not per_seed:
            continue
        vals = list(per_seed.values())
        mean = sum(vals) / len(vals)
        if len(vals) >= 2:
            ci = paired_bootstrap_ci(
                {"x": per_seed}, lambda m: {"x": m["x"]}, n_boot=n_boot)
            lo, hi = ci["x"]
        else:
            lo = hi = float("nan")
        pts.append(CurvePoint(tol, mean, lo, hi, len(vals)))
    pts.sort(key=lambda p: -p.tolerance)
    return pts


@dataclass
class PairwiseDiff:
    tolerance: float
    diff: float       # value(A) - value(B)
    ci_lo: float
    ci_hi: float
    n_seeds: int
    significant: bool  # CI excludes 0


def curve_difference(
    table: CCRSTable,
    a: tuple[int, int],
    b: tuple[int, int],
    tolerances: Iterable[float],
    metric: str = METRIC_SUCCESS_RATE,
    n_boot: int = 2000,
) -> list[PairwiseDiff]:
    """Paired per-tolerance difference between checkpoints A=(worker,iter), B."""
    out: list[PairwiseDiff] = []
    for tol in tolerances:
        ka = CellKey(a[0], a[1], tol)
        kb = CellKey(b[0], b[1], tol)
        ca = table.per_seed_metric(ka, metric)
        cb = table.per_seed_metric(kb, metric)
        common = sorted(set(ca) & set(cb))
        if not common:
            continue
        diff = sum(ca[s] - cb[s] for s in common) / len(common)
        if len(common) >= 2:
            restricted = {
                "a": {s: ca[s] for s in common},
                "b": {s: cb[s] for s in common},
            }
            ci = paired_bootstrap_ci(
                restricted, lambda m: {"d": m["a"] - m["b"]}, n_boot=n_boot)
            lo, hi = ci["d"]
        else:
            lo = hi = float("nan")
        sig = (not math.isnan(lo)) and (lo > 0 or hi < 0)
        out.append(PairwiseDiff(tol, diff, lo, hi, len(common), sig))
    out.sort(key=lambda p: -p.tolerance)
    return out


# ---------------------------------------------------------------------------
# Pre-registered decision criteria (frozen BEFORE seeing sweep results)
# ---------------------------------------------------------------------------

# Minimum absolute success-rate difference considered practically meaningful.
PREREG_MIN_EFFECT = 0.05
# Tolerance regions (meters) for the p1-vs-p4 crossing test.
PREREG_HIGH_PRECISION_TOLS = [0.020, 0.019, 0.018]
PREREG_MID_PRECISION_TOLS = [0.023, 0.022, 0.021]


def _sig_better(diffs: list[PairwiseDiff], tols: list[float], sign: int) -> bool:
    """True if at ANY tol in tols the diff is significant with given sign and
    |effect| >= PREREG_MIN_EFFECT."""
    for d in diffs:
        if any(abs(d.tolerance - t) < 1e-9 for t in tols):
            if d.significant and sign * d.diff >= PREREG_MIN_EFFECT:
                return True
    return False


def preregistered_verdicts(
    p4_minus_p1: list[PairwiseDiff],
    p5_peak_minus_tip: list[PairwiseDiff],
) -> dict[str, Any]:
    """Apply the pre-registered A/B/C/D criteria.

    p4_minus_p1: curve_difference(A=p4_peak_ckpt, B=p1_peak_ckpt).
    p5_peak_minus_tip: curve_difference(A=p5_peak, B=p5_tip); positive diff
    means peak better than tip (capability loss at tip).
    """
    # A: crossing — p4 better at <=0.020, p1 better around 0.022
    p4_wins_high = _sig_better(p4_minus_p1, PREREG_HIGH_PRECISION_TOLS, sign=+1)
    p1_wins_mid = _sig_better(p4_minus_p1, PREREG_MID_PRECISION_TOLS, sign=-1)
    crossing = p4_wins_high and p1_wins_mid

    # B: p4 dominates on most tolerances
    sig_cells = [d for d in p4_minus_p1 if d.significant and abs(d.diff) >= PREREG_MIN_EFFECT]
    p4_dominant = (
        len(sig_cells) > 0
        and all(d.diff > 0 for d in sig_cells)
        and len([d for d in p4_minus_p1 if d.diff > 0]) >= math.ceil(0.7 * len(p4_minus_p1))
    ) if p4_minus_p1 else False

    # C: p1 still better at fixed 0.020
    c_cells = [d for d in p4_minus_p1 if abs(d.tolerance - 0.020) < 1e-9]
    p1_better_at_020 = bool(
        c_cells and c_cells[0].significant and c_cells[0].diff <= -PREREG_MIN_EFFECT
    )

    # D: p5 forgetting vs reclassification
    p5_sig_drop = [d for d in p5_peak_minus_tip
                   if d.significant and d.diff >= PREREG_MIN_EFFECT]
    if p5_peak_minus_tip:
        if p5_sig_drop:
            p5_verdict = "real_policy_forgetting"
        else:
            p5_verdict = "criterion_reclassification_dominant"
    else:
        p5_verdict = "insufficient_data"

    if crossing:
        ab_verdict = "A_capability_curve_crossing"
    elif p4_dominant:
        ab_verdict = "B_p4_globally_stronger"
    elif p1_better_at_020:
        ab_verdict = "C_p1_still_better_at_high_precision"
    else:
        ab_verdict = "inconclusive"

    return {
        "prereg_min_effect": PREREG_MIN_EFFECT,
        "high_precision_tols_m": PREREG_HIGH_PRECISION_TOLS,
        "mid_precision_tols_m": PREREG_MID_PRECISION_TOLS,
        "p4_wins_high_precision": p4_wins_high,
        "p1_wins_mid_precision": p1_wins_mid,
        "verdict_p1_p4": ab_verdict,
        "verdict_p5": p5_verdict,
        "n_significant_p4_minus_p1": len(sig_cells),
        "n_significant_p5_drop": len(p5_sig_drop),
    }
