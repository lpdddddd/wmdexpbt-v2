#!/usr/bin/env python3
"""CCRS analysis: Q(k, tau) tables, D/F/C decomposition, paired bootstrap.

Primary capability surface (tolerances in meters):

  Q(k, tau) = E[min(nsucc, M) | checkpoint k, fixed tolerance tau] / M
            with M = max_consecutive_successes (default 50).

Estimated by the mean per-sample normalized_nsucc over identical reset
seeds AND identical per-env initial states (verified via reset
fingerprints, not just equal seed integers).

Auxiliary metrics: nsucc_mean (raw capability) and
success_any_rate = P(nsucc > 0) (saturates for strong policies; never used
for primary verdicts; it is NOT P(error <= tau)).

For a curriculum transition with pre checkpoint p (live tol a) and post
checkpoint q (live tol b, b < a):

  D = Q(p,a) - Q(q,b)                       # observed live drop
  F = 0.5 * [(Q(p,a) - Q(q,a)) + (Q(p,b) - Q(q,b))]   # policy-update effect
  C = 0.5 * [(Q(p,a) - Q(p,b)) + (Q(q,a) - Q(q,b))]   # criterion effect

  Identity: D = F + C (exact, by construction).

Two D/F/C modes:
- exact transition mode (compute_dfc): a and b are the REAL transition
  tolerances from tolerance_transitions.csv; missing exact cells raise
  (reported as missing, never silently mapped to a grid);
- grid-projected mode (grid_projected_dfc): explicit compatibility mode
  that snaps to the sweep grid; results carry mode="grid_projected" and
  must not be presented as exact D/F/C.

Pairing & bootstrap: every (checkpoint, tolerance) cell is evaluated with
the SAME reset seed set and the SAME per-env initial states. Sample-level
pairing uses sample_id = (reset_seed, env_id). Before any paired analysis
the per-env reset fingerprints of all involved cells are compared; on any
mismatch the analysis is marked reset_pairing_invalid, paired bootstrap is
refused, and no significance conclusion is produced.

Hierarchical paired bootstrap: outer resample over seeds, inner resample
over env_ids within each drawn seed; every replicate applies the SAME
resampled sample_id multiset to all cells.
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

from .eval_protocol import SCHEMA_VERSION

METRIC_NORMALIZED_NSUCC = "normalized_nsucc"   # PRIMARY (Q)
METRIC_NSUCC_MEAN = "nsucc"                    # per-sample raw nsucc
METRIC_SUCCESS_ANY = "success_any"             # auxiliary only

PRIMARY_METRIC = METRIC_NORMALIZED_NSUCC
AUXILIARY_METRICS = (METRIC_SUCCESS_ANY,)

# Tolerance match precision for exact transition cells (meters).
EXACT_TOL_DECIMALS = 6


class ResetPairingError(ValueError):
    """Raised when paired analysis is attempted on unpaired/mismatched data."""


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
                       round(self.fixed_tolerance, EXACT_TOL_DECIMALS))


def _sample_metric_value(sample: dict, metric: str) -> Optional[float]:
    if metric == METRIC_NORMALIZED_NSUCC:
        v = sample.get("normalized_nsucc")
    elif metric == METRIC_NSUCC_MEAN:
        v = sample.get("nsucc")
    elif metric == METRIC_SUCCESS_ANY:
        n = sample.get("nsucc")
        v = None if n is None else (1.0 if float(n) > 0 else 0.0)
    else:
        v = sample.get(metric)
    if v is None:
        return None
    v = float(v)
    return None if math.isnan(v) else v


class CCRSTable:
    """Index of per-unit results: (worker, iter, tol) -> {seed: row}.

    Rows must be schema v2 (status=ok); older schemas are rejected at load.
    """

    def __init__(self) -> None:
        self.cells: dict[CellKey, dict[int, dict]] = {}
        self.meta: dict[CellKey, dict] = {}
        self.rejected_rows = 0

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
            self.rejected_rows += 1
            return
        if int(row.get("schema_version", -1)) != SCHEMA_VERSION:
            self.rejected_rows += 1
            return
        key = CellKey(
            int(row["worker_id"]),
            int(row["checkpoint_iteration"]),
            round(float(row["fixed_tolerance"]), EXACT_TOL_DECIMALS),
        )
        seed = int(row["reset_seed"])
        self.cells.setdefault(key, {})[seed] = row
        self.meta.setdefault(key, {
            "checkpoint_frame": row.get("checkpoint_frame"),
            "checkpoint_path": row.get("checkpoint_path"),
        })

    def seeds(self, key: CellKey) -> set[int]:
        return set(self.cells.get(key.rounded(), {}).keys())

    # -- sample-level access ------------------------------------------------

    def per_sample_metric(
        self, key: CellKey, metric: str = PRIMARY_METRIC,
    ) -> dict[tuple[int, int], float]:
        """{(seed, env_id): value} across all seeds of the cell."""
        out: dict[tuple[int, int], float] = {}
        for seed, row in self.cells.get(key.rounded(), {}).items():
            for s in row.get("samples", []):
                v = _sample_metric_value(s, metric)
                if v is not None:
                    out[(seed, int(s["env_id"]))] = v
        return out

    def per_seed_metric(self, key: CellKey,
                        metric_field: str = "normalized_nsucc") -> dict[int, float]:
        """Seed-level aggregate values (from the stored aggregate fields)."""
        out = {}
        for seed, row in self.cells.get(key.rounded(), {}).items():
            v = row.get(metric_field)
            if v is not None and not (isinstance(v, float) and math.isnan(v)):
                out[seed] = float(v)
        return out

    def fingerprints(self, key: CellKey) -> dict[int, dict[int, str]]:
        """{seed: {env_id: fingerprint}}."""
        out: dict[int, dict[int, str]] = {}
        for seed, row in self.cells.get(key.rounded(), {}).items():
            fp = row.get("per_env_reset_fingerprint") or {}
            out[seed] = {int(k): str(v) for k, v in fp.items()}
        return out

    def Q(self, key: CellKey, metric: str = PRIMARY_METRIC) -> float:
        """Point estimate: mean over all paired samples of the cell."""
        vals = list(self.per_sample_metric(key, metric).values())
        if not vals:
            return float("nan")
        return sum(vals) / len(vals)

    def is_empty(self) -> bool:
        return not self.cells


# ---------------------------------------------------------------------------
# Reset pairing validation (fingerprints, not just seed integers)
# ---------------------------------------------------------------------------

@dataclass
class PairingCheck:
    valid: bool
    n_shared_samples: int
    mismatched_samples: list[str] = field(default_factory=list)
    reason: str = ""


def check_reset_pairing(table: CCRSTable, keys: Iterable[CellKey]) -> PairingCheck:
    """Verify that all cells share identical per-env initial states.

    For every (seed, env_id) present in ALL cells, the reset fingerprints
    must be identical across cells. Any mismatch invalidates pairing.
    """
    keys = [k.rounded() for k in keys]
    if not keys:
        return PairingCheck(False, 0, reason="no cells")
    fps = {k: table.fingerprints(k) for k in keys}
    common_seeds = set.intersection(*(set(f.keys()) for f in fps.values()))
    if not common_seeds:
        return PairingCheck(False, 0, reason="no shared seeds")
    mismatched: list[str] = []
    n_shared = 0
    for seed in sorted(common_seeds):
        env_sets = [set(fps[k].get(seed, {}).keys()) for k in keys]
        common_envs = set.intersection(*env_sets) if env_sets else set()
        for env_id in sorted(common_envs):
            n_shared += 1
            vals = {fps[k][seed][env_id] for k in keys}
            if len(vals) != 1:
                mismatched.append(f"seed_{seed}_env_{env_id:03d}")
    if n_shared == 0:
        return PairingCheck(False, 0, reason="no shared (seed, env_id) samples")
    if mismatched:
        return PairingCheck(False, n_shared, mismatched,
                            reason="reset fingerprints differ across cells")
    return PairingCheck(True, n_shared)


def _restrict_to_shared(
    cells: dict[str, dict[tuple[int, int], float]],
) -> dict[str, dict[tuple[int, int], float]]:
    """Restrict all cells to their shared sample_id set; strict: raises when
    the shared set is empty."""
    ids = set.intersection(*(set(c.keys()) for c in cells.values()))
    if not ids:
        raise ResetPairingError("no shared sample ids across cells")
    return {n: {k: c[k] for k in ids} for n, c in cells.items()}


# ---------------------------------------------------------------------------
# Hierarchical paired bootstrap (seeds outer, env_ids inner)
# ---------------------------------------------------------------------------

def hierarchical_paired_bootstrap(
    cells: dict[str, dict[tuple[int, int], float]],
    stat_fn: Callable[[dict[str, float]], dict[str, float]],
    n_boot: int = 2000,
    seed: int = 1234,
    alpha: float = 0.05,
) -> dict[str, tuple[float, float]]:
    """Percentile bootstrap over paired samples keyed by (seed, env_id).

    All cells must have the SAME sample_id set (strict pairing; raises
    ResetPairingError otherwise). Each replicate:
      1. resamples seeds with replacement (outer level);
      2. within each drawn seed, resamples env_ids with replacement (inner);
      3. applies the same resampled sample multiset to every cell.
    stat_fn maps {cell_name: replicate_mean} -> named statistics.
    """
    names = list(cells.keys())
    if not names:
        return {}
    id_sets = [set(cells[n].keys()) for n in names]
    common = set.intersection(*id_sets)
    union = set.union(*id_sets)
    if common != union:
        raise ResetPairingError(
            f"unpaired sample sets across cells: |common|={len(common)} "
            f"|union|={len(union)}; refusing to bootstrap unpaired data"
        )
    if len(common) < 2:
        raise ResetPairingError(
            f"need >=2 shared paired samples for bootstrap, got {len(common)}"
        )
    by_seed: dict[int, list[tuple[int, int]]] = {}
    for (s, e) in common:
        by_seed.setdefault(s, []).append((s, e))
    seeds = sorted(by_seed.keys())

    rng = random.Random(seed)
    samples_acc: dict[str, list[float]] = {}
    for _ in range(n_boot):
        drawn: list[tuple[int, int]] = []
        for _ in seeds:  # outer: seeds with replacement
            s = seeds[rng.randrange(len(seeds))]
            pool = by_seed[s]
            for _ in pool:  # inner: env_ids with replacement within seed
                drawn.append(pool[rng.randrange(len(pool))])
        means = {
            n: sum(cells[n][sid] for sid in drawn) / len(drawn)
            for n in names
        }
        stats = stat_fn(means)
        for k, v in stats.items():
            samples_acc.setdefault(k, []).append(v)

    out: dict[str, tuple[float, float]] = {}
    for k, vals in samples_acc.items():
        vals.sort()
        lo_i = int((alpha / 2) * (len(vals) - 1))
        hi_i = int((1 - alpha / 2) * (len(vals) - 1))
        out[k] = (vals[lo_i], vals[hi_i])
    return out


# Backwards-compatible name used by capability_curve for single-cell CIs.
def paired_bootstrap_ci(
    cells: dict[str, dict],
    stat_fn: Callable[[dict[str, float]], dict[str, float]],
    n_boot: int = 2000,
    seed: int = 1234,
    alpha: float = 0.05,
) -> dict[str, tuple[float, float]]:
    """Wrapper accepting either {(seed,env_id): v} or legacy {seed: v} keys."""
    def _norm(c: dict) -> dict[tuple[int, int], float]:
        out = {}
        for k, v in c.items():
            if isinstance(k, tuple):
                out[(int(k[0]), int(k[1]))] = float(v)
            else:
                out[(int(k), 0)] = float(v)
        return out
    return hierarchical_paired_bootstrap(
        {n: _norm(c) for n, c in cells.items()},
        stat_fn, n_boot=n_boot, seed=seed, alpha=alpha,
    )


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
    Q_pa: float
    Q_pb: float
    Q_qa: float
    Q_qb: float
    D: float
    F: float
    C: float
    identity_residual: float           # |D - (F + C)|, must be ~0
    n_samples: int
    mode: str = "exact"                # "exact" | "grid_projected"
    metric: str = PRIMARY_METRIC
    reset_pairing_valid: bool = True
    lineage_changed: Optional[bool] = None
    transition_kind: str = "within_lineage"   # or "lineage_replacement_transition"
    ci: dict = field(default_factory=dict)    # bootstrap CIs per component


def dfc_from_Q(Q_pa: float, Q_pb: float, Q_qa: float, Q_qb: float) -> tuple[float, float, float]:
    D = Q_pa - Q_qb
    F = 0.5 * ((Q_pa - Q_qa) + (Q_pb - Q_qb))
    C = 0.5 * ((Q_pa - Q_pb) + (Q_qa - Q_qb))
    return D, F, C


def compute_dfc(
    table: CCRSTable,
    worker_id: int,
    pre_iteration: int,
    post_iteration: int,
    tol_pre: float,
    tol_post: float,
    metric: str = PRIMARY_METRIC,
    n_boot: int = 2000,
    mode: str = "exact",
    lineage_changed: Optional[bool] = None,
) -> DFCResult:
    """D/F/C for one transition using cells evaluated at EXACTLY a and b.

    exact mode: tolerances must exist exactly (to EXACT_TOL_DECIMALS) in the
    evaluated results. Missing cells raise KeyError mentioning
    "missing exact transition cells" — never silently snapped to a grid.
    Fingerprint pairing is validated; on mismatch the point estimates are
    still returned but reset_pairing_valid=False and NO CI is produced.
    """
    if not tol_post < tol_pre:
        raise ValueError(f"expected tol_post < tol_pre, got {tol_post} >= {tol_pre}")
    kpa = CellKey(worker_id, pre_iteration, tol_pre)
    kpb = CellKey(worker_id, pre_iteration, tol_post)
    kqa = CellKey(worker_id, post_iteration, tol_pre)
    kqb = CellKey(worker_id, post_iteration, tol_post)
    cells = {
        "pa": table.per_sample_metric(kpa, metric),
        "pb": table.per_sample_metric(kpb, metric),
        "qa": table.per_sample_metric(kqa, metric),
        "qb": table.per_sample_metric(kqb, metric),
    }
    missing = [n for n, c in cells.items() if not c]
    if missing:
        raise KeyError(
            f"missing exact transition cells {missing} for worker={worker_id} "
            f"iters=({pre_iteration},{post_iteration}) "
            f"tols=({tol_pre},{tol_post}) mode={mode}"
        )
    pairing = check_reset_pairing(table, [kpa, kpb, kqa, kqb])

    shared = _restrict_to_shared(cells)
    n = len(next(iter(shared.values())))
    Q_pa = sum(shared["pa"].values()) / n
    Q_pb = sum(shared["pb"].values()) / n
    Q_qa = sum(shared["qa"].values()) / n
    Q_qb = sum(shared["qb"].values()) / n
    D, F, C = dfc_from_Q(Q_pa, Q_pb, Q_qa, Q_qb)

    ci: dict = {}
    if pairing.valid:
        def _stat(means: dict[str, float]) -> dict[str, float]:
            d, f, c = dfc_from_Q(means["pa"], means["pb"], means["qa"], means["qb"])
            return {"D": d, "F": f, "C": c}
        try:
            ci = hierarchical_paired_bootstrap(shared, _stat, n_boot=n_boot)
        except ResetPairingError:
            pairing = PairingCheck(False, pairing.n_shared_samples,
                                   reason="bootstrap pairing failed")

    return DFCResult(
        worker_id=worker_id,
        pre_iteration=pre_iteration,
        post_iteration=post_iteration,
        tol_pre=tol_pre,
        tol_post=tol_post,
        Q_pa=Q_pa, Q_pb=Q_pb, Q_qa=Q_qa, Q_qb=Q_qb,
        D=D, F=F, C=C,
        identity_residual=abs(D - (F + C)),
        n_samples=n,
        mode=mode,
        metric=metric,
        reset_pairing_valid=pairing.valid,
        lineage_changed=lineage_changed,
        transition_kind=("lineage_replacement_transition" if lineage_changed
                         else "within_lineage"),
        ci=ci,
    )


def grid_projected_dfc(
    table: CCRSTable,
    worker_id: int,
    pre_iteration: int,
    post_iteration: int,
    tol_pre_real: float,
    tol_post_real: float,
    grid: Iterable[float],
    metric: str = PRIMARY_METRIC,
    n_boot: int = 2000,
    max_snap_distance: float = 0.0021,
    lineage_changed: Optional[bool] = None,
) -> DFCResult:
    """Compatibility D/F/C on the sweep grid (NOT exact).

    Snaps the real transition tolerances to the nearest grid value; result
    carries mode="grid_projected" and must not be used for primary
    scientific conclusions. Raises when snap distance exceeds
    max_snap_distance.
    """
    grid = sorted(float(g) for g in grid)
    if not grid:
        raise ValueError("empty grid")

    def _snap(tol: float) -> float:
        g = min(grid, key=lambda x: abs(x - tol))
        if abs(g - tol) > max_snap_distance:
            raise ValueError(
                f"grid snap distance {abs(g - tol):.4f} exceeds "
                f"{max_snap_distance} for tol={tol}"
            )
        return g

    a, b = _snap(tol_pre_real), _snap(tol_post_real)
    if not b < a:
        raise ValueError(
            f"grid projection collapsed transition ({tol_pre_real}->{tol_post_real} "
            f"both snap near {a}); refusing"
        )
    res = compute_dfc(
        table, worker_id, pre_iteration, post_iteration,
        tol_pre=a, tol_post=b, metric=metric, n_boot=n_boot,
        mode="grid_projected", lineage_changed=lineage_changed,
    )
    return res


# ---------------------------------------------------------------------------
# Capability curves & pairwise comparison
# ---------------------------------------------------------------------------

@dataclass
class CurvePoint:
    tolerance: float
    value: float          # PRIMARY: normalized_nsucc (Q)
    ci_lo: float
    ci_hi: float
    n_samples: int
    metric: str = PRIMARY_METRIC
    nsucc_mean: float = float("nan")
    success_any_rate: float = float("nan")


def capability_curve(
    table: CCRSTable,
    worker_id: int,
    checkpoint_iteration: int,
    tolerances: Iterable[float],
    metric: str = PRIMARY_METRIC,
    n_boot: int = 2000,
) -> list[CurvePoint]:
    pts: list[CurvePoint] = []
    for tol in tolerances:
        key = CellKey(worker_id, checkpoint_iteration, tol)
        per_sample = table.per_sample_metric(key, metric)
        if not per_sample:
            continue
        vals = list(per_sample.values())
        mean = sum(vals) / len(vals)
        if len(vals) >= 2:
            try:
                ci = hierarchical_paired_bootstrap(
                    {"x": per_sample}, lambda m: {"x": m["x"]}, n_boot=n_boot)
                lo, hi = ci["x"]
            except ResetPairingError:
                lo = hi = float("nan")
        else:
            lo = hi = float("nan")
        pts.append(CurvePoint(
            tolerance=tol, value=mean, ci_lo=lo, ci_hi=hi,
            n_samples=len(vals), metric=metric,
            nsucc_mean=table.Q(key, METRIC_NSUCC_MEAN),
            success_any_rate=table.Q(key, METRIC_SUCCESS_ANY),
        ))
    pts.sort(key=lambda p: -p.tolerance)
    return pts


@dataclass
class PairwiseDiff:
    tolerance: float
    diff: float           # value(A) - value(B) on PRIMARY metric
    ci_lo: float
    ci_hi: float
    n_samples: int
    significant: bool     # CI excludes 0 (only when pairing valid)
    metric: str = PRIMARY_METRIC
    reset_pairing_valid: bool = True
    pairing_reason: str = ""


def curve_difference(
    table: CCRSTable,
    a: tuple[int, int],
    b: tuple[int, int],
    tolerances: Iterable[float],
    metric: str = PRIMARY_METRIC,
    n_boot: int = 2000,
) -> list[PairwiseDiff]:
    """Paired per-tolerance difference between checkpoints A=(worker,iter), B.

    Pairing is validated via reset fingerprints per tolerance; on mismatch
    the diff is reported WITHOUT CI and marked reset_pairing_valid=False
    (never significant).
    """
    out: list[PairwiseDiff] = []
    for tol in tolerances:
        ka = CellKey(a[0], a[1], tol)
        kb = CellKey(b[0], b[1], tol)
        ca = table.per_sample_metric(ka, metric)
        cb = table.per_sample_metric(kb, metric)
        common = sorted(set(ca) & set(cb))
        if not common:
            continue
        diff = sum(ca[s] - cb[s] for s in common) / len(common)
        pairing = check_reset_pairing(table, [ka, kb])
        if pairing.valid:
            restricted = {
                "a": {s: ca[s] for s in common},
                "b": {s: cb[s] for s in common},
            }
            try:
                ci = hierarchical_paired_bootstrap(
                    restricted, lambda m: {"d": m["a"] - m["b"]}, n_boot=n_boot)
                lo, hi = ci["d"]
            except ResetPairingError as e:
                pairing = PairingCheck(False, len(common), reason=str(e))
                lo = hi = float("nan")
        else:
            lo = hi = float("nan")
        sig = pairing.valid and (not math.isnan(lo)) and (lo > 0 or hi < 0)
        out.append(PairwiseDiff(
            tolerance=tol, diff=diff, ci_lo=lo, ci_hi=hi,
            n_samples=len(common), significant=sig, metric=metric,
            reset_pairing_valid=pairing.valid,
            pairing_reason=pairing.reason,
        ))
    out.sort(key=lambda p: -p.tolerance)
    return out


# ---------------------------------------------------------------------------
# Pre-registered decision criteria (frozen BEFORE seeing sweep results)
# ---------------------------------------------------------------------------

# Minimum absolute normalized_nsucc difference considered practically
# meaningful: 0.05 (= ~2.5 raw nsucc at M=50).
PREREG_MIN_EFFECT = 0.05
# Tolerance regions (meters) for the p1-vs-p4 crossing test.
PREREG_HIGH_PRECISION_TOLS = [0.020, 0.019, 0.018]
PREREG_MID_PRECISION_TOLS = [0.023, 0.022, 0.021]


def _sig_better(diffs: list[PairwiseDiff], tols: list[float], sign: int) -> bool:
    """True if at ANY tol in tols the diff is significant with given sign,
    |effect| >= PREREG_MIN_EFFECT, and reset pairing is valid."""
    for d in diffs:
        if any(abs(d.tolerance - t) < 1e-9 for t in tols):
            if (d.reset_pairing_valid and d.significant
                    and sign * d.diff >= PREREG_MIN_EFFECT):
                return True
    return False


def preregistered_verdicts(
    p4_minus_p1: list[PairwiseDiff],
    p5_peak_minus_tip: list[PairwiseDiff],
) -> dict[str, Any]:
    """Apply the pre-registered A/B/C/D criteria on normalized_nsucc diffs.

    success_any_rate is auxiliary and deliberately not consulted here.
    If reset pairing is invalid for all cells, verdicts degrade to
    "reset_pairing_invalid" and no significance claims are made.
    """
    pairing_ok_p14 = any(d.reset_pairing_valid for d in p4_minus_p1)
    pairing_ok_p5 = any(d.reset_pairing_valid for d in p5_peak_minus_tip)

    # A: crossing — p4 better at <=0.020, p1 better around 0.021-0.023
    p4_wins_high = _sig_better(p4_minus_p1, PREREG_HIGH_PRECISION_TOLS, sign=+1)
    p1_wins_mid = _sig_better(p4_minus_p1, PREREG_MID_PRECISION_TOLS, sign=-1)
    crossing = p4_wins_high and p1_wins_mid

    # B: p4 dominates on most tolerances
    valid_cells = [d for d in p4_minus_p1 if d.reset_pairing_valid]
    sig_cells = [d for d in valid_cells
                 if d.significant and abs(d.diff) >= PREREG_MIN_EFFECT]
    p4_dominant = (
        len(sig_cells) > 0
        and all(d.diff > 0 for d in sig_cells)
        and len([d for d in valid_cells if d.diff > 0])
        >= math.ceil(0.7 * len(valid_cells))
    ) if valid_cells else False

    # C: p1 still better at fixed 0.020
    c_cells = [d for d in valid_cells if abs(d.tolerance - 0.020) < 1e-9]
    p1_better_at_020 = bool(
        c_cells and c_cells[0].significant and c_cells[0].diff <= -PREREG_MIN_EFFECT
    )

    # D: p5 forgetting vs reclassification
    p5_valid = [d for d in p5_peak_minus_tip if d.reset_pairing_valid]
    p5_sig_drop = [d for d in p5_valid
                   if d.significant and d.diff >= PREREG_MIN_EFFECT]
    if not p5_peak_minus_tip:
        p5_verdict = "insufficient_data"
    elif not pairing_ok_p5:
        p5_verdict = "reset_pairing_invalid"
    elif p5_sig_drop:
        p5_verdict = "real_policy_forgetting"
    else:
        p5_verdict = "criterion_reclassification_dominant"

    if p4_minus_p1 and not pairing_ok_p14:
        ab_verdict = "reset_pairing_invalid"
    elif crossing:
        ab_verdict = "A_capability_curve_crossing"
    elif p4_dominant:
        ab_verdict = "B_p4_globally_stronger"
    elif p1_better_at_020:
        ab_verdict = "C_p1_still_better_at_high_precision"
    else:
        ab_verdict = "inconclusive"

    return {
        "metric": PRIMARY_METRIC,
        "prereg_min_effect": PREREG_MIN_EFFECT,
        "prereg_min_effect_nsucc_at_M50": PREREG_MIN_EFFECT * 50,
        "high_precision_tols_m": PREREG_HIGH_PRECISION_TOLS,
        "mid_precision_tols_m": PREREG_MID_PRECISION_TOLS,
        "reset_pairing_valid_p1_p4": pairing_ok_p14,
        "reset_pairing_valid_p5": pairing_ok_p5,
        "p4_wins_high_precision": p4_wins_high,
        "p1_wins_mid_precision": p1_wins_mid,
        "verdict_p1_p4": ab_verdict,
        "verdict_p5": p5_verdict,
        "n_significant_p4_minus_p1": len(sig_cells),
        "n_significant_p5_drop": len(p5_sig_drop),
    }
