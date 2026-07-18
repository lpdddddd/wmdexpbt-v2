#!/usr/bin/env python3
"""Final analysis for the p1 fixed-precision extension (offline, no GPU).

Combines, read-only:
- the NEW extension results (0.016/0.014/0.012/0.010 m) from
  results/analysis/ccrs_p1_precision_extension/extension_results.jsonl;
- the EXISTING p1 tip 0.018 m cells from
  results/analysis/ccrs_base_retention/ccrs_results.jsonl (never re-run,
  never modified).

Outputs (into the extension dir):
- precision_extension_analysis.json — per-tolerance nsucc_mean + 95% CI,
  normalized_nsucc + CI, terminal-error quantiles, dropped rate, sample
  counts, and the decision label;
- P1_PRECISION_REPORT.md — human-readable table + verdict;
- figures/p1_precision_capability_curve.png — capability curve
  (1.8 → 1.0 cm).

Decision label (pre-registered, on nsucc_mean at fixed 0.010 m):
- TARGET_REACHED:  nsucc_mean >= 40
- NEAR_TARGET:     30 <= nsucc_mean < 40
- PRECISION_LIMIT: nsucc_mean < 30
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from dexpbt_isaaclab.ccrs.analysis import (
    CCRSTable,
    CellKey,
    METRIC_NORMALIZED_NSUCC,
    METRIC_NSUCC_MEAN,
    ResetPairingError,
    hierarchical_paired_bootstrap,
)
from dexpbt_isaaclab.ccrs.eval_protocol import quantiles
from dexpbt_isaaclab.ccrs.precision_extension import (
    EXTENSION_TOLERANCES_M,
    P1_TIP_ITERATION,
    P1_TIP_WORKER_ID,
)

# Analysis grid: existing 0.018 cell + the four new tolerances.
ANALYSIS_TOLERANCES_M = [0.018] + list(EXTENSION_TOLERANCES_M)

# Pre-registered decision thresholds on nsucc_mean at fixed 0.010 m.
DECISION_TOLERANCE_M = 0.010
TARGET_NSUCC = 40.0
NEAR_TARGET_NSUCC = 30.0


def decision_label(nsucc_mean_at_1cm: float) -> str:
    if math.isnan(nsucc_mean_at_1cm):
        return "NO_DATA"
    if nsucc_mean_at_1cm >= TARGET_NSUCC:
        return "TARGET_REACHED"
    if nsucc_mean_at_1cm >= NEAR_TARGET_NSUCC:
        return "NEAR_TARGET"
    return "PRECISION_LIMIT"


def _cell_ci(table: CCRSTable, key: CellKey, metric: str,
             n_boot: int = 2000) -> tuple[float, float]:
    """Hierarchical bootstrap CI of a single cell's per-sample mean."""
    samples = table.per_sample_metric(key, metric)
    if len(samples) < 2:
        return float("nan"), float("nan")
    try:
        ci = hierarchical_paired_bootstrap(
            {"c": samples}, lambda m: {"v": m["c"]}, n_boot=n_boot)
        return ci["v"]
    except ResetPairingError:
        return float("nan"), float("nan")


def _terminal_stats(table: CCRSTable, key: CellKey) -> dict:
    errs: list[float] = []
    drops: list[bool] = []
    n_total = 0
    for _seed, row in table.cells.get(key.rounded(), {}).items():
        for s in row.get("samples", []):
            n_total += 1
            if s.get("terminal_error_valid") and s.get("terminal_error") is not None:
                v = float(s["terminal_error"])
                if not math.isnan(v):
                    errs.append(v)
            if s.get("dropped") is not None:
                drops.append(bool(s["dropped"]))
    qs = quantiles(sorted(errs), [0.10, 0.25, 0.50, 0.75, 0.90]) if errs else {}
    return {
        "n_samples": n_total,
        "terminal_error_n_valid": len(errs),
        "terminal_error_quantiles": qs,
        "dropped_rate": (sum(drops) / len(drops)) if drops else float("nan"),
    }


def analyze(extension_jsonl: Path, base_jsonl: Path, out_dir: Path,
            n_boot: int = 2000) -> dict:
    table = CCRSTable.from_jsonl(extension_jsonl)
    # merge ONLY the p1 tip rows from the base sweep (read-only source)
    base = CCRSTable.from_jsonl(base_jsonl)
    for key, seed_rows in base.cells.items():
        if (key.worker_id == P1_TIP_WORKER_ID
                and key.checkpoint_iteration == P1_TIP_ITERATION):
            for row in seed_rows.values():
                table.add_row(row)

    points = []
    for tol in ANALYSIS_TOLERANCES_M:
        key = CellKey(P1_TIP_WORKER_ID, P1_TIP_ITERATION, tol)
        samples_ns = table.per_sample_metric(key, METRIC_NSUCC_MEAN)
        ns_mean = (sum(samples_ns.values()) / len(samples_ns)
                   if samples_ns else float("nan"))
        ns_lo, ns_hi = _cell_ci(table, key, METRIC_NSUCC_MEAN, n_boot)
        q_mean = table.Q(key, METRIC_NORMALIZED_NSUCC)
        q_lo, q_hi = _cell_ci(table, key, METRIC_NORMALIZED_NSUCC, n_boot)
        pt = {
            "tolerance_m": tol,
            "tolerance_cm": round(tol * 100, 2),
            "source": ("ccrs_base_retention (reused, not re-run)"
                       if abs(tol - 0.018) < 1e-9 else "extension"),
            "nsucc_mean": ns_mean,
            "nsucc_mean_ci95": [ns_lo, ns_hi],
            "normalized_nsucc": q_mean,
            "normalized_nsucc_ci95": [q_lo, q_hi],
            **_terminal_stats(table, key),
        }
        points.append(pt)

    at_1cm = next((p for p in points
                   if abs(p["tolerance_m"] - DECISION_TOLERANCE_M) < 1e-9),
                  None)
    ns_1cm = at_1cm["nsucc_mean"] if at_1cm else float("nan")
    label = decision_label(ns_1cm)

    result = {
        "checkpoint": {"worker_id": P1_TIP_WORKER_ID,
                       "iteration": P1_TIP_ITERATION},
        "analysis_tolerances_m": ANALYSIS_TOLERANCES_M,
        "decision": {
            "tolerance_m": DECISION_TOLERANCE_M,
            "nsucc_mean_at_1cm": ns_1cm,
            "target_reached_threshold": TARGET_NSUCC,
            "near_target_threshold": NEAR_TARGET_NSUCC,
            "label": label,
        },
        "points": points,
        "n_boot": n_boot,
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / "precision_extension_analysis.json"
    tmp = out_json.with_suffix(".tmp")
    tmp.write_text(json.dumps(result, indent=2))
    tmp.replace(out_json)

    _write_report(result, out_dir / "P1_PRECISION_REPORT.md")
    _fig_curve(points, out_dir / "figures" / "p1_precision_capability_curve.png")
    return result


def _fmt(v: float, nd: int = 3) -> str:
    return "n/a" if (isinstance(v, float) and math.isnan(v)) else f"{v:.{nd}f}"


def _write_report(result: dict, path: Path) -> None:
    d = result["decision"]
    lines = [
        "# p1 tip fixed-precision extension — final analysis",
        "",
        f"Checkpoint: p{P1_TIP_WORKER_ID} it{P1_TIP_ITERATION} (5B frames). "
        "0.018 cm cells reused read-only from ccrs_base_retention; "
        "0.016–0.010 m cells from the extension sweep. Same protocol "
        "(policy_only, deterministic, 512 envs, 512 first episodes, "
        "paired resets, M=50).",
        "",
        "| tol (cm) | nsucc_mean [95% CI] | normalized_nsucc [95% CI] "
        "| terminal err q50 | dropped rate | n | source |",
        "|---|---|---|---|---|---|---|",
    ]
    for p in result["points"]:
        q50 = p["terminal_error_quantiles"].get("q50", float("nan")) \
            if p["terminal_error_quantiles"] else float("nan")
        lines.append(
            f"| {p['tolerance_cm']:.1f} "
            f"| {_fmt(p['nsucc_mean'], 1)} "
            f"[{_fmt(p['nsucc_mean_ci95'][0], 1)}, "
            f"{_fmt(p['nsucc_mean_ci95'][1], 1)}] "
            f"| {_fmt(p['normalized_nsucc'])} "
            f"[{_fmt(p['normalized_nsucc_ci95'][0])}, "
            f"{_fmt(p['normalized_nsucc_ci95'][1])}] "
            f"| {_fmt(q50)} | {_fmt(p['dropped_rate'])} "
            f"| {p['n_samples']} | {p['source']} |"
        )
    lines += [
        "",
        "## Decision (pre-registered)",
        "",
        f"- nsucc_mean at fixed {d['tolerance_m'] * 100:.1f} cm: "
        f"**{_fmt(d['nsucc_mean_at_1cm'], 1)}**",
        f"- label: **{d['label']}**",
        f"  (TARGET_REACHED >= {d['target_reached_threshold']:.0f}; "
        f"NEAR_TARGET >= {d['near_target_threshold']:.0f}; "
        "PRECISION_LIMIT below)",
        "",
        "Figure: `figures/p1_precision_capability_curve.png`.",
        "",
        "Single checkpoint, single population run; the 4 evaluation seeds "
        "quantify evaluation uncertainty only.",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text("\n".join(lines))
    tmp.replace(path)


def _fig_curve(points: list[dict], out_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    pts = [p for p in points if not math.isnan(p["nsucc_mean"])]
    if not pts:
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    xs = [p["tolerance_cm"] for p in pts]
    ys = [p["nsucc_mean"] for p in pts]
    los = [p["nsucc_mean_ci95"][0] for p in pts]
    his = [p["nsucc_mean_ci95"][1] for p in pts]
    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.plot(xs, ys, marker="o", color="tab:blue", label="p1 tip nsucc_mean")
    if all(not math.isnan(v) for v in los + his):
        ax.fill_between(xs, los, his, color="tab:blue", alpha=0.18,
                        label="95% CI (hierarchical bootstrap)")
    ax.axhline(TARGET_NSUCC, color="tab:green", ls="--", lw=1,
               label=f"TARGET_REACHED ({TARGET_NSUCC:.0f})")
    ax.axhline(NEAR_TARGET_NSUCC, color="tab:orange", ls="--", lw=1,
               label=f"NEAR_TARGET ({NEAR_TARGET_NSUCC:.0f})")
    ax.set_xlabel("fixed tolerance (cm)")
    ax.set_ylabel("nsucc_mean (M=50 cap)")
    ax.set_title("p1 tip fixed-precision capability curve (1.8 → 1.0 cm)")
    ax.invert_xaxis()
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="p1 precision extension final analysis (offline).")
    ap.add_argument("--extension-results", required=True,
                    help="extension_results.jsonl from the extension sweep")
    ap.add_argument("--base-results", required=True,
                    help="ccrs_base_retention/ccrs_results.jsonl (read-only)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--n-boot", type=int, default=2000)
    args = ap.parse_args()
    r = analyze(Path(args.extension_results), Path(args.base_results),
                Path(args.out_dir), n_boot=args.n_boot)
    d = r["decision"]
    print(f"[precision_analysis] nsucc_mean@1cm={d['nsucc_mean_at_1cm']} "
          f"label={d['label']} -> {Path(args.out_dir)}")


if __name__ == "__main__":
    main()
