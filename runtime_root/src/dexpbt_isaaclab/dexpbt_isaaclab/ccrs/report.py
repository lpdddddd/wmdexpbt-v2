#!/usr/bin/env python3
"""CCRS report generation: figures + CCRS_REPORT.md + ccrs_summary.json.

Consumes ccrs_results.jsonl (per-unit raw rows), the checkpoint manifest and
tolerance_transitions.csv, and produces:

  figures/capability_curve_w{i}.png     fixed-tol curve per worker/checkpoint
  figures/ccrs_heatmap.png              checkpoint x tolerance heatmap
  figures/p1_p4_difference.png          paired diff with CI
  figures/p5_peak_vs_tip.png            p5 peak/tip curves + diff
  figures/dfc_decomposition.png         D/F/C bars per transition
  figures/precision_retention_frontier.png
  CCRS_REPORT.md                        human report (criteria pre-registered)
  ccrs_summary.json                     machine-readable summary

With no/partial data it degrades gracefully: writes a summary stating which
sections are missing and skips those figures — it never fabricates results.

CLI:
  python -m dexpbt_isaaclab.ccrs.report \
    --results results/analysis/ccrs_base_retention/ccrs_results.jsonl \
    --manifest results/analysis/ccrs_base_retention/checkpoint_manifest.csv \
    --transitions results/analysis/ccrs_base_retention/tolerance_transitions.csv \
    --out-dir results/analysis/ccrs_base_retention
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import math
from pathlib import Path
from typing import Any, Optional

from .analysis import (
    CCRSTable,
    CellKey,
    CurvePoint,
    DFCResult,
    METRIC_SUCCESS_RATE,
    PairwiseDiff,
    capability_curve,
    compute_dfc,
    curve_difference,
    preregistered_verdicts,
)


def _grid_tolerances(table: CCRSTable) -> list[float]:
    tols = sorted({k.fixed_tolerance for k in table.cells}, reverse=True)
    return tols


def _checkpoints_by_role(manifest_csv: Path) -> dict[str, list[dict]]:
    """role -> manifest rows (role matched by prefix)."""
    out: dict[str, list[dict]] = {}
    if not manifest_csv.is_file():
        return out
    with manifest_csv.open() as f:
        for r in csv.DictReader(f):
            for role in (r.get("roles") or "").split(";"):
                role = role.strip()
                if role:
                    out.setdefault(role.split("@")[0], []).append(r)
    return out


def _find_ckpt(manifest_roles: dict[str, list[dict]], worker: int, role: str) -> Optional[dict]:
    for r in manifest_roles.get(role, []):
        if int(r["worker_id"]) == worker:
            return r
    return None


def _load_transitions(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    with path.open() as f:
        return list(csv.DictReader(f))


def _nearest_grid(tol: float, grid: list[float]) -> Optional[float]:
    if not grid:
        return None
    best = min(grid, key=lambda g: abs(g - tol))
    return best if abs(best - tol) <= 0.0021 else None


# ---------------------------------------------------------------------------
# Figures (matplotlib imported lazily; skipped gracefully if unavailable)
# ---------------------------------------------------------------------------

def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def fig_capability_curves(table: CCRSTable, fig_dir: Path, tols: list[float]) -> list[str]:
    plt = _plt()
    written = []
    workers = sorted({k.worker_id for k in table.cells})
    for w in workers:
        iters = sorted({k.checkpoint_iteration for k in table.cells if k.worker_id == w})
        if not iters:
            continue
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for it in iters:
            pts = capability_curve(table, w, it, tols)
            if not pts:
                continue
            xs = [p.tolerance * 100 for p in pts]
            ys = [p.value for p in pts]
            ax.plot(xs, ys, marker="o", label=f"iter {it}")
            los = [p.ci_lo for p in pts]
            his = [p.ci_hi for p in pts]
            if not any(math.isnan(v) for v in los + his):
                ax.fill_between(xs, los, his, alpha=0.15)
        ax.set_xlabel("fixed tolerance (cm)")
        ax.set_ylabel("R(k, tau) = fixed-tol success rate")
        ax.set_title(f"Capability curve — worker p{w}")
        ax.invert_xaxis()
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
        out = fig_dir / f"capability_curve_w{w}.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        written.append(out.name)
    return written


def fig_heatmap(table: CCRSTable, fig_dir: Path, tols: list[float]) -> Optional[str]:
    plt = _plt()
    keys = sorted(table.cells.keys(), key=lambda k: (k.worker_id, k.checkpoint_iteration))
    ckpts = sorted({(k.worker_id, k.checkpoint_iteration) for k in keys})
    if not ckpts or not tols:
        return None
    import numpy as np
    grid = np.full((len(ckpts), len(tols)), np.nan)
    for i, (w, it) in enumerate(ckpts):
        for j, tol in enumerate(tols):
            v = table.R(CellKey(w, it, tol))
            grid[i, j] = v
    fig, ax = plt.subplots(figsize=(1.2 + 0.8 * len(tols), 0.6 + 0.35 * len(ckpts)))
    im = ax.imshow(grid, aspect="auto", cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks(range(len(tols)))
    ax.set_xticklabels([f"{t*100:.1f}cm" for t in tols], rotation=45, ha="right")
    ax.set_yticks(range(len(ckpts)))
    ax.set_yticklabels([f"p{w}@it{it}" for w, it in ckpts], fontsize=7)
    fig.colorbar(im, label="R(k, tau)")
    ax.set_title("CCRS heatmap (fixed-tolerance success rate)")
    out = fig_dir / "ccrs_heatmap.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out.name


def fig_pair_diff(diffs: list[PairwiseDiff], title: str, fname: str, fig_dir: Path,
                  label: str) -> Optional[str]:
    if not diffs:
        return None
    plt = _plt()
    xs = [d.tolerance * 100 for d in diffs]
    ys = [d.diff for d in diffs]
    los = [d.ci_lo for d in diffs]
    his = [d.ci_hi for d in diffs]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.axhline(0, color="k", lw=0.8)
    ax.plot(xs, ys, marker="o", label=label)
    if not any(math.isnan(v) for v in los + his):
        yerr_lo = [y - lo for y, lo in zip(ys, los)]
        yerr_hi = [hi - y for y, hi in zip(ys, his)]
        ax.errorbar(xs, ys, yerr=[yerr_lo, yerr_hi], fmt="none", capsize=3)
    for d in diffs:
        if d.significant:
            ax.annotate("*", (d.tolerance * 100, d.diff), fontsize=14, ha="center")
    ax.set_xlabel("fixed tolerance (cm)")
    ax.set_ylabel("Δ success rate")
    ax.set_title(title)
    ax.invert_xaxis()
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    out = fig_dir / fname
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out.name


def fig_dfc(dfc_rows: list[DFCResult], fig_dir: Path) -> Optional[str]:
    if not dfc_rows:
        return None
    plt = _plt()
    labels = [f"p{r.worker_id} it{r.pre_iteration}->{r.post_iteration}\n"
              f"{r.tol_pre*100:.2f}->{r.tol_post*100:.2f}cm" for r in dfc_rows]
    import numpy as np
    x = np.arange(len(dfc_rows))
    w = 0.27
    fig, ax = plt.subplots(figsize=(1.5 + 1.1 * len(dfc_rows), 4.5))
    ax.bar(x - w, [r.D for r in dfc_rows], w, label="D (observed drop)")
    ax.bar(x, [r.F for r in dfc_rows], w, label="F (policy effect)")
    ax.bar(x + w, [r.C for r in dfc_rows], w, label="C (criterion effect)")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel("Δ success rate")
    ax.set_title("D/F/C decomposition per tolerance transition")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    out = fig_dir / "dfc_decomposition.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out.name


def fig_frontier(table: CCRSTable, fig_dir: Path, tols: list[float]) -> Optional[str]:
    """Precision-retention frontier: per checkpoint, tightest tolerance with
    R >= 0.5 (x) vs R at the loosest grid tolerance (y)."""
    if table.is_empty() or not tols:
        return None
    plt = _plt()
    pts = []
    for (w, it) in sorted({(k.worker_id, k.checkpoint_iteration) for k in table.cells}):
        loosest = max(tols)
        r_loose = table.R(CellKey(w, it, loosest))
        tight = None
        for tol in sorted(tols):  # ascending = tightest first
            r = table.R(CellKey(w, it, tol))
            if not math.isnan(r) and r >= 0.5:
                tight = tol
                break
        if tight is not None and not math.isnan(r_loose):
            pts.append((w, it, tight, r_loose))
    if not pts:
        return None
    fig, ax = plt.subplots(figsize=(6.5, 5))
    for w, it, tight, r_loose in pts:
        ax.scatter(tight * 100, r_loose, s=40)
        ax.annotate(f"p{w}@{it}", (tight * 100, r_loose), fontsize=7,
                    xytext=(3, 3), textcoords="offset points")
    ax.set_xlabel("tightest tolerance with R>=0.5 (cm)  [lower = more precise]")
    ax.set_ylabel(f"R at loosest tol ({max(tols)*100:.1f}cm)  [higher = better retention]")
    ax.set_title("Precision–retention frontier")
    ax.grid(alpha=0.3)
    out = fig_dir / "precision_retention_frontier.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out.name


# ---------------------------------------------------------------------------
# Main report
# ---------------------------------------------------------------------------

PREREG_BLOCK = """\
## Pre-registered decision criteria (frozen before sweep results)

Effect threshold: |Δ success rate| >= 0.05 with 95% paired-bootstrap CI
excluding 0. Tolerances in meters (cm shown for readability).

- **A (crossing)**: p4(peak) significantly better than p1(peak) at
  τ ∈ {0.020, 0.019, 0.018} AND p1 significantly better at
  τ ∈ {0.023, 0.022, 0.021} → capability curve crossing; population has a
  stable-capability vs high-precision division of labor → proceed to
  Anchor-Scout CD-ReST validation.
- **B (p4 dominant)**: p4 better on ≥70% of grid tolerances and all
  significant cells favor p4 → p4 is globally stronger; continue fine
  curriculum from p4 rather than adding cross-worker transfer mechanisms.
- **C (p1 still better at 0.020 m)**: p4's low live tolerance does not prove
  high-precision capability; audit dynamic curriculum, sampling variance and
  evaluation criteria.
- **D (p5 peak vs tip)**: significant fixed-tolerance drop (peak - tip) →
  real policy forgetting; otherwise (live nsucc drop but flat fixed-tol
  capability) → criterion reclassification dominant.
"""


def generate(
    results_jsonl: Path,
    manifest_csv: Path,
    transitions_csv: Path,
    out_dir: Path,
    n_boot: int = 2000,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(exist_ok=True)

    table = CCRSTable.from_jsonl(results_jsonl)
    roles = _checkpoints_by_role(manifest_csv)
    transitions = _load_transitions(transitions_csv)
    tols = _grid_tolerances(table)

    summary: dict[str, Any] = {
        "results_jsonl": str(results_jsonl),
        "n_cells": len(table.cells),
        "grid_tolerances_m": tols,
        "sections": {},
        "missing": [],
    }
    figures: list[str] = []
    md: list[str] = ["# CCRS Report — base-retention 5B", ""]
    md.append(PREREG_BLOCK)

    if table.is_empty():
        summary["missing"].append("no evaluation results yet (ccrs_results.jsonl empty)")
        md.append("\n**No evaluation data found.** Run the sweep first; this "
                  "report was generated in dry mode and contains no results.\n")
        (out_dir / "CCRS_REPORT.md").write_text("\n".join(md))
        (out_dir / "ccrs_summary.json").write_text(json.dumps(summary, indent=2))
        return summary

    # 1/2: curves + heatmap + frontier
    figures += fig_capability_curves(table, fig_dir, tols)
    hm = fig_heatmap(table, fig_dir, tols)
    if hm:
        figures.append(hm)
    fr = fig_frontier(table, fig_dir, tols)
    if fr:
        figures.append(fr)

    # 3: p1 vs p4 peak comparison
    p1_peak = _find_ckpt(roles, 1, "peak") or _find_ckpt(roles, 1, "target")
    p4_peak = _find_ckpt(roles, 4, "peak") or _find_ckpt(roles, 4, "target")
    p4_minus_p1: list[PairwiseDiff] = []
    if p1_peak and p4_peak:
        p4_minus_p1 = curve_difference(
            table,
            a=(4, int(p4_peak["iteration"])),
            b=(1, int(p1_peak["iteration"])),
            tolerances=tols, n_boot=n_boot,
        )
        f = fig_pair_diff(p4_minus_p1, "p4(peak) − p1(peak) fixed-tolerance success rate",
                          "p1_p4_difference.png", fig_dir, "p4 − p1")
        if f:
            figures.append(f)
        summary["sections"]["p4_minus_p1"] = [dataclasses.asdict(d) for d in p4_minus_p1]
    else:
        summary["missing"].append("p1/p4 peak checkpoints not found in manifest")

    # 4: p5 peak vs tip
    p5_peak = _find_ckpt(roles, 5, "peak")
    p5_tip = _find_ckpt(roles, 5, "tip")
    p5_diffs: list[PairwiseDiff] = []
    if p5_peak and p5_tip:
        p5_diffs = curve_difference(
            table,
            a=(5, int(p5_peak["iteration"])),
            b=(5, int(p5_tip["iteration"])),
            tolerances=tols, n_boot=n_boot,
        )
        f = fig_pair_diff(p5_diffs, "p5 peak − tip fixed-tolerance success rate",
                          "p5_peak_vs_tip.png", fig_dir, "peak − tip")
        if f:
            figures.append(f)
        summary["sections"]["p5_peak_minus_tip"] = [dataclasses.asdict(d) for d in p5_diffs]
    else:
        summary["missing"].append("p5 peak/tip checkpoints not found in manifest")

    # 5: D/F/C per transition (snap live tols to grid)
    dfc_rows: list[DFCResult] = []
    for t in transitions:
        try:
            a = _nearest_grid(float(t["pre_tolerance"]), tols)
            b = _nearest_grid(float(t["post_tolerance"]), tols)
            if a is None or b is None or not b < a:
                continue
            dfc_rows.append(compute_dfc(
                table,
                worker_id=int(t["worker_id"]),
                pre_iteration=int(t["pre_iteration"]),
                post_iteration=int(t["post_iteration"]),
                tol_pre=a, tol_post=b, n_boot=n_boot,
            ))
        except (KeyError, ValueError):
            continue
    if dfc_rows:
        f = fig_dfc(dfc_rows, fig_dir)
        if f:
            figures.append(f)
        summary["sections"]["dfc"] = [dataclasses.asdict(r) for r in dfc_rows]
        bad = [r for r in dfc_rows if r.identity_residual > 1e-9]
        if bad:
            summary["missing"].append(f"D=F+C identity violated in {len(bad)} rows (BUG)")
    else:
        summary["missing"].append("no D/F/C transitions computable (missing paired cells)")

    # 6: pre-registered verdicts
    verdicts = preregistered_verdicts(p4_minus_p1, p5_diffs)
    summary["verdicts"] = verdicts

    # ---- markdown body ----
    md.append("\n## Verdicts (auto-applied pre-registered criteria)\n")
    md.append(f"- p1 vs p4: **{verdicts['verdict_p1_p4']}**")
    md.append(f"- p5 peak→tip: **{verdicts['verdict_p5']}**")
    md.append("")
    if p4_minus_p1:
        md.append("## p4 − p1 (fixed-tolerance success rate)\n")
        md.append("| tol (m) | tol (cm) | Δ | 95% CI | sig |")
        md.append("|---|---|---|---|---|")
        for d in p4_minus_p1:
            md.append(f"| {d.tolerance:.3f} | {d.tolerance*100:.1f} | {d.diff:+.3f} "
                      f"| [{d.ci_lo:+.3f}, {d.ci_hi:+.3f}] | {'yes' if d.significant else 'no'} |")
        md.append("")
    if p5_diffs:
        md.append("## p5 peak − tip (fixed-tolerance success rate)\n")
        md.append("| tol (m) | tol (cm) | Δ | 95% CI | sig |")
        md.append("|---|---|---|---|---|")
        for d in p5_diffs:
            md.append(f"| {d.tolerance:.3f} | {d.tolerance*100:.1f} | {d.diff:+.3f} "
                      f"| [{d.ci_lo:+.3f}, {d.ci_hi:+.3f}] | {'yes' if d.significant else 'no'} |")
        md.append("")
    if dfc_rows:
        md.append("## D/F/C decomposition\n")
        md.append("| worker | iters | tol a→b (m) | D | F | C | n_seeds |")
        md.append("|---|---|---|---|---|---|---|")
        for r in dfc_rows:
            md.append(f"| p{r.worker_id} | {r.pre_iteration}→{r.post_iteration} "
                      f"| {r.tol_pre:.4f}→{r.tol_post:.4f} "
                      f"| {r.D:+.3f} | {r.F:+.3f} | {r.C:+.3f} | {r.n_seeds} |")
        md.append("")
    if summary["missing"]:
        md.append("## Missing / degraded sections\n")
        for m in summary["missing"]:
            md.append(f"- {m}")
        md.append("")
    if figures:
        md.append("## Figures\n")
        for f in figures:
            md.append(f"![{f}](figures/{f})")
        md.append("")

    (out_dir / "CCRS_REPORT.md").write_text("\n".join(md))
    (out_dir / "ccrs_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    summary["figures"] = figures
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate CCRS report from sweep results.")
    ap.add_argument("--results", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--transitions", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--n-boot", type=int, default=2000)
    args = ap.parse_args()
    s = generate(
        results_jsonl=Path(args.results),
        manifest_csv=Path(args.manifest),
        transitions_csv=Path(args.transitions),
        out_dir=Path(args.out_dir),
        n_boot=args.n_boot,
    )
    print(f"[ccrs.report] cells={s['n_cells']} missing={len(s['missing'])} "
          f"-> {Path(args.out_dir) / 'CCRS_REPORT.md'}")


if __name__ == "__main__":
    main()
