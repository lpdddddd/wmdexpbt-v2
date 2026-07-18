#!/usr/bin/env python3
"""CCRS capability-frontier analysis (offline, no Isaac, no GPU).

Consumes existing sweep results (ccrs_results.jsonl with per-sample data and
reset fingerprints) and produces:

1. Named pairwise checkpoint comparisons (e.g. p1 tip vs p5 peak) on the
   fixed tolerance grid, with strict fingerprint pairing, hierarchical
   paired bootstrap CIs, practical-threshold flags and paired-sample
   completeness checks.
2. Best-checkpoint capability envelope: per fixed tolerance, the
   numerically best and second-best checkpoints among a given set, with the
   paired CI of their difference. "Numerically best" and "practically
   significantly best" are reported as SEPARATE fields — a checkpoint can
   lead numerically without the lead being statistically or practically
   meaningful.
3. Curve-crossing assessment between two checkpoints: A significantly +
   practically better in one tolerance region while B is in another.
4. Figures: best_checkpoint_envelope.png, peak_tip_temporal_differences.png.

All comparisons use the PRIMARY metric normalized_nsucc (Q); raw nsucc_mean
is attached as context. Practical threshold: PREREG_MIN_EFFECT (0.05
normalized_nsucc ~= 2.5 raw nsucc at M=50).

CLI:
  python -m dexpbt_isaaclab.ccrs.frontier \
    --results results/analysis/ccrs_base_retention/ccrs_results.jsonl \
    --out-dir results/analysis/ccrs_base_retention [--n-boot 2000]
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional

from .eval_protocol import ANALYSIS_PROTOCOL_VERSION, DEFAULT_TOLERANCES_M
from .analysis import (
    CCRSTable,
    CellKey,
    METRIC_NSUCC_MEAN,
    PREREG_MIN_EFFECT,
    PRIMARY_METRIC,
    PairingCheck,
    ResetPairingError,
    check_reset_pairing,
    hierarchical_paired_bootstrap,
)


@dataclass
class ComparisonPoint:
    """A vs B at one fixed tolerance (paired samples)."""
    tolerance: float
    Q_a: float
    Q_b: float
    nsucc_a: float
    nsucc_b: float
    diff: float                     # Q_a - Q_b
    ci_lo: float
    ci_hi: float
    significant: bool               # CI excludes 0 AND pairing valid
    practical: bool                 # significant AND |diff| >= PREREG_MIN_EFFECT
    n_paired_samples: int
    expected_paired_samples: int
    paired_samples_complete: bool
    reset_pairing_valid: bool
    pairing_reason: str = ""


@dataclass
class Comparison:
    name: str
    a_label: str
    b_label: str
    a: tuple[int, int]              # (worker_id, iteration)
    b: tuple[int, int]
    points: list[ComparisonPoint] = field(default_factory=list)


def compare_checkpoints(
    table: CCRSTable,
    a: tuple[int, int],
    b: tuple[int, int],
    tolerances: Iterable[float],
    n_boot: int = 2000,
    expected_paired_samples: Optional[int] = None,
) -> list[ComparisonPoint]:
    """Strictly paired A-vs-B comparison at each tolerance.

    Fingerprint pairing is validated per tolerance; on mismatch no CI and no
    significance are produced. expected_paired_samples (e.g. seeds x envs)
    drives the completeness flag; when None it defaults to the observed max
    across tolerances.
    """
    tols = list(tolerances)
    per_tol: list[dict] = []
    for tol in tols:
        ka = CellKey(a[0], a[1], tol)
        kb = CellKey(b[0], b[1], tol)
        ca = table.per_sample_metric(ka, PRIMARY_METRIC)
        cb = table.per_sample_metric(kb, PRIMARY_METRIC)
        common = sorted(set(ca) & set(cb))
        if not common:
            continue
        per_tol.append(dict(tol=tol, ka=ka, kb=kb, ca=ca, cb=cb, common=common))
    if expected_paired_samples is None:
        expected_paired_samples = max((len(d["common"]) for d in per_tol), default=0)

    out: list[ComparisonPoint] = []
    for d in per_tol:
        tol, ka, kb, ca, cb, common = (d["tol"], d["ka"], d["kb"],
                                       d["ca"], d["cb"], d["common"])
        q_a = sum(ca[s] for s in common) / len(common)
        q_b = sum(cb[s] for s in common) / len(common)
        na = table.per_sample_metric(ka, METRIC_NSUCC_MEAN)
        nb = table.per_sample_metric(kb, METRIC_NSUCC_MEAN)
        ns_a = sum(na[s] for s in common) / len(common)
        ns_b = sum(nb[s] for s in common) / len(common)
        diff = q_a - q_b

        pairing = check_reset_pairing(table, [ka, kb])
        lo = hi = float("nan")
        if pairing.valid:
            try:
                ci = hierarchical_paired_bootstrap(
                    {"a": {s: ca[s] for s in common},
                     "b": {s: cb[s] for s in common}},
                    lambda m: {"d": m["a"] - m["b"]}, n_boot=n_boot)
                lo, hi = ci["d"]
            except ResetPairingError as e:
                pairing = PairingCheck(False, len(common), reason=str(e))
        sig = (pairing.valid and not math.isnan(lo) and (lo > 0 or hi < 0))
        out.append(ComparisonPoint(
            tolerance=tol,
            Q_a=q_a, Q_b=q_b, nsucc_a=ns_a, nsucc_b=ns_b,
            diff=diff, ci_lo=lo, ci_hi=hi,
            significant=sig,
            practical=bool(sig and abs(diff) >= PREREG_MIN_EFFECT),
            n_paired_samples=len(common),
            expected_paired_samples=int(expected_paired_samples),
            paired_samples_complete=(len(common) == int(expected_paired_samples)),
            reset_pairing_valid=pairing.valid,
            pairing_reason=pairing.reason,
        ))
    out.sort(key=lambda p: -p.tolerance)
    return out


# ---------------------------------------------------------------------------
# Best-checkpoint envelope
# ---------------------------------------------------------------------------

@dataclass
class EnvelopePoint:
    """Per-tolerance ranking of checkpoints on Q (numerical vs practical)."""
    tolerance: float
    best: tuple[int, int]           # numerically best (worker, iteration)
    best_label: str
    best_Q: float
    best_nsucc: float
    second: Optional[tuple[int, int]]
    second_label: str
    second_Q: float
    second_nsucc: float
    diff_best_second: float
    ci_lo: float
    ci_hi: float
    significant: bool
    practical: bool                 # best lead is significant AND >= threshold
    reset_pairing_valid: bool


def best_checkpoint_envelope(
    table: CCRSTable,
    checkpoints: dict[tuple[int, int], str],
    tolerances: Iterable[float],
    n_boot: int = 1000,
) -> list[EnvelopePoint]:
    """For each tolerance, rank `checkpoints` ({(w, it): label}) by Q and
    test whether the numerical leader beats the runner-up with a paired CI.
    """
    out: list[EnvelopePoint] = []
    for tol in tolerances:
        scored: list[tuple[float, tuple[int, int]]] = []
        for key in checkpoints:
            q = table.Q(CellKey(key[0], key[1], tol))
            if not math.isnan(q):
                scored.append((q, key))
        if len(scored) < 2:
            continue
        scored.sort(key=lambda x: -x[0])
        (q1, k1), (q2, k2) = scored[0], scored[1]
        pts = compare_checkpoints(table, k1, k2, [tol], n_boot=n_boot)
        if pts:
            p = pts[0]
            lo, hi, sig, prac, pv = (p.ci_lo, p.ci_hi, p.significant,
                                     p.practical, p.reset_pairing_valid)
            diff = p.diff
        else:
            lo = hi = float("nan")
            sig = prac = pv = False
            diff = q1 - q2
        out.append(EnvelopePoint(
            tolerance=tol,
            best=k1, best_label=checkpoints[k1],
            best_Q=q1, best_nsucc=table.Q(CellKey(*k1, tol), METRIC_NSUCC_MEAN),
            second=k2, second_label=checkpoints[k2],
            second_Q=q2, second_nsucc=table.Q(CellKey(*k2, tol), METRIC_NSUCC_MEAN),
            diff_best_second=diff, ci_lo=lo, ci_hi=hi,
            significant=sig, practical=prac, reset_pairing_valid=pv,
        ))
    out.sort(key=lambda p: -p.tolerance)
    return out


# ---------------------------------------------------------------------------
# Crossing assessment
# ---------------------------------------------------------------------------

def assess_crossing(
    points: list[ComparisonPoint],
    region_b_better: list[float],
    region_a_better: list[float],
) -> dict:
    """Crossing between A and B (points = A - B):

    numerical crossing: sign of the diff flips between the two regions;
    practical crossing: B practically better (diff <= -threshold, sig) at
    some tol in region_b_better AND A practically better (diff >= +threshold,
    sig) at some tol in region_a_better. Reported separately.
    """
    def _in(t: float, region: list[float]) -> bool:
        return any(abs(t - r) < 1e-9 for r in region)

    b_num = [p for p in points if _in(p.tolerance, region_b_better) and p.diff < 0]
    a_num = [p for p in points if _in(p.tolerance, region_a_better) and p.diff > 0]
    b_prac = [p for p in b_num if p.practical]
    a_prac = [p for p in a_num if p.practical]
    pairing_ok = all(p.reset_pairing_valid for p in points) if points else False
    return {
        "region_b_better_tols_m": region_b_better,
        "region_a_better_tols_m": region_a_better,
        "numerical_crossing": bool(b_num and a_num),
        "practical_crossing": bool(pairing_ok and b_prac and a_prac),
        "b_better_numerical_tols": [p.tolerance for p in b_num],
        "a_better_numerical_tols": [p.tolerance for p in a_num],
        "b_better_practical_tols": [p.tolerance for p in b_prac],
        "a_better_practical_tols": [p.tolerance for p in a_prac],
        "reset_pairing_valid": pairing_ok,
    }


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def fig_envelope(
    table: CCRSTable,
    checkpoints: dict[tuple[int, int], str],
    envelope: list[EnvelopePoint],
    out_path: Path,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    tols = [e.tolerance for e in envelope]
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    for key, label in sorted(checkpoints.items()):
        xs, ys = [], []
        for tol in tols:
            q = table.Q(CellKey(key[0], key[1], tol))
            if not math.isnan(q):
                xs.append(tol * 100)
                ys.append(q)
        bold = any(e.best == key for e in envelope)
        ax.plot(xs, ys, marker="o", markersize=3,
                lw=2.2 if bold else 0.9,
                alpha=1.0 if bold else 0.35,
                label=label if bold else None,
                zorder=3 if bold else 1)
    for e in envelope:
        marker = "*" if e.practical else ("^" if e.significant else ".")
        ax.scatter(e.tolerance * 100, e.best_Q, s=160 if e.practical else 70,
                   marker=marker, color="k", zorder=5)
        ax.annotate(e.best_label, (e.tolerance * 100, e.best_Q),
                    xytext=(0, 9), textcoords="offset points",
                    ha="center", fontsize=7)
    ax.set_xlabel("fixed tolerance (cm)")
    ax.set_ylabel("Q = normalized nsucc")
    ax.set_title("Best-checkpoint capability envelope\n"
                 "(★ lead is significant + practical; ▲ significant only; "
                 "· numerical only)")
    ax.invert_xaxis()
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="lower left")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_peak_tip_differences(
    comparisons: list[Comparison],
    out_path: Path,
) -> None:
    """Tip - peak (temporal) ΔQ per worker across the tolerance grid."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8.5, 5))
    ax.axhline(0, color="k", lw=0.8)
    ax.axhspan(-PREREG_MIN_EFFECT, PREREG_MIN_EFFECT, color="gray", alpha=0.12,
               label=f"|ΔQ| < {PREREG_MIN_EFFECT} (below practical threshold)")
    for comp in comparisons:
        xs = [p.tolerance * 100 for p in comp.points]
        ys = [p.diff for p in comp.points]
        line, = ax.plot(xs, ys, marker="o", label=comp.name)
        ok = [not math.isnan(p.ci_lo) for p in comp.points]
        if all(ok):
            ax.fill_between(xs, [p.ci_lo for p in comp.points],
                            [p.ci_hi for p in comp.points],
                            color=line.get_color(), alpha=0.15)
        for p in comp.points:
            if p.practical:
                ax.annotate("*", (p.tolerance * 100, p.diff), fontsize=13,
                            ha="center", color=line.get_color())
    ax.set_xlabel("fixed tolerance (cm)")
    ax.set_ylabel("ΔQ = Q(tip) − Q(peak)")
    ax.set_title("Peak → tip temporal capability change per worker\n"
                 "(* = significant + practical; negative = forgetting)")
    ax.invert_xaxis()
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

# The 12 key checkpoints of the base-retention 5B run.
BASE_RETENTION_CHECKPOINTS: dict[tuple[int, int], str] = {
    (0, 160): "p0 tip",
    (1, 149): "p1 peak",
    (1, 160): "p1 tip",
    (2, 96): "p2 peak",
    (2, 109): "p2 target",
    (2, 161): "p2 tip",
    (3, 159): "p3 peak",
    (3, 160): "p3 tip",
    (4, 141): "p4 peak",
    (4, 160): "p4 tip",
    (5, 117): "p5 peak",
    (5, 160): "p5 tip",
}

NAMED_COMPARISONS: list[tuple[str, tuple[int, int], tuple[int, int]]] = [
    ("p1 tip vs p1 peak", (1, 160), (1, 149)),
    ("p1 tip vs p5 peak", (1, 160), (5, 117)),
    ("p4 tip vs p4 peak", (4, 160), (4, 141)),
    ("p3 tip vs p3 peak", (3, 160), (3, 159)),
    ("p5 tip vs p5 peak", (5, 160), (5, 117)),
]


def run(results_jsonl: Path, out_dir: Path, n_boot: int = 2000,
        expected_paired_samples: int = 2048) -> dict:
    table = CCRSTable.from_jsonl(results_jsonl)
    if table.is_empty():
        raise SystemExit(f"no usable rows in {results_jsonl}")
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    tols = DEFAULT_TOLERANCES_M

    comparisons: list[Comparison] = []
    for name, a, b in NAMED_COMPARISONS:
        pts = compare_checkpoints(table, a, b, tols, n_boot=n_boot,
                                  expected_paired_samples=expected_paired_samples)
        comparisons.append(Comparison(
            name=name,
            a_label=BASE_RETENTION_CHECKPOINTS.get(a, str(a)),
            b_label=BASE_RETENTION_CHECKPOINTS.get(b, str(b)),
            a=a, b=b, points=pts,
        ))

    envelope = best_checkpoint_envelope(
        table, BASE_RETENTION_CHECKPOINTS, tols, n_boot=n_boot)

    # crossing: A = p1 tip, B = p5 peak; B region 2.8-2.1 cm, A region 2.0-1.8 cm
    p1tip_vs_p5peak = next(c for c in comparisons if c.name == "p1 tip vs p5 peak")
    crossing = assess_crossing(
        p1tip_vs_p5peak.points,
        region_b_better=[0.028, 0.025, 0.023, 0.022, 0.021],
        region_a_better=[0.020, 0.019, 0.018],
    )

    fig_envelope(table, BASE_RETENTION_CHECKPOINTS, envelope,
                 fig_dir / "best_checkpoint_envelope.png")
    temporal = [c for c in comparisons if c.name != "p1 tip vs p5 peak"]
    fig_peak_tip_differences(temporal,
                             fig_dir / "peak_tip_temporal_differences.png")

    result = {
        "analysis_protocol_version": ANALYSIS_PROTOCOL_VERSION,
        "primary_metric": PRIMARY_METRIC,
        "practical_threshold": PREREG_MIN_EFFECT,
        "expected_paired_samples": expected_paired_samples,
        "n_boot": n_boot,
        "comparisons": [
            {
                "name": c.name, "a": list(c.a), "b": list(c.b),
                "a_label": c.a_label, "b_label": c.b_label,
                "points": [dataclasses.asdict(p) for p in c.points],
            } for c in comparisons
        ],
        "envelope": [
            {**dataclasses.asdict(e),
             "best": list(e.best),
             "second": list(e.second) if e.second else None}
            for e in envelope
        ],
        "crossing_p1tip_vs_p5peak": crossing,
        "figures": ["best_checkpoint_envelope.png",
                    "peak_tip_temporal_differences.png"],
    }
    out_path = out_dir / "frontier_analysis.json"
    tmp = out_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(result, indent=2))
    tmp.replace(out_path)

    # merge into ccrs_summary.json (non-destructive)
    summary_path = out_dir / "ccrs_summary.json"
    if summary_path.is_file():
        try:
            summary = json.loads(summary_path.read_text())
        except json.JSONDecodeError:
            summary = {}
        summary["frontier"] = {
            "analysis_protocol_version": ANALYSIS_PROTOCOL_VERSION,
            "crossing_p1tip_vs_p5peak": crossing,
            "envelope_best_per_tolerance": {
                f"{e.tolerance:.3f}": {
                    "best": e.best_label, "best_Q": e.best_Q,
                    "best_nsucc": e.best_nsucc,
                    "second": e.second_label,
                    "lead_significant": e.significant,
                    "lead_practical": e.practical,
                } for e in envelope
            },
        }
        tmp = summary_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(summary, indent=2, default=str))
        tmp.replace(summary_path)
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description="CCRS capability-frontier analysis (offline).")
    ap.add_argument("--results", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--expected-paired-samples", type=int, default=2048)
    args = ap.parse_args()
    r = run(Path(args.results), Path(args.out_dir), n_boot=args.n_boot,
            expected_paired_samples=args.expected_paired_samples)
    print(f"[ccrs.frontier] {len(r['comparisons'])} comparisons, "
          f"{len(r['envelope'])} envelope points -> "
          f"{Path(args.out_dir) / 'frontier_analysis.json'}")


if __name__ == "__main__":
    main()
