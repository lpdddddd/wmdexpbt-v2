#!/usr/bin/env python3
"""CCRS report generation: figures + CCRS_REPORT.md + ccrs_summary.json.

Consumes ccrs_results.jsonl (per-unit raw rows, schema v2), the checkpoint
manifest and tolerance_transitions.csv, and produces:

  figures/capability_curve_w{i}.png     fixed-tol Q curve per worker/ckpt
  figures/ccrs_heatmap.png              checkpoint x tolerance heatmap (Q)
  figures/p1_p4_difference.png          paired Q diff with CI
  figures/p5_peak_vs_tip.png            p5 peak/tip Q diff
  figures/dfc_decomposition.png         exact D/F/C bars per transition
  figures/precision_retention_frontier.png
  CCRS_REPORT.md                        human report (criteria pre-registered)
  ccrs_summary.json                     machine-readable summary

Primary metric everywhere: normalized_nsucc (Q). nsucc_mean and
success_any_rate are attached as context columns only.

D/F/C: exact transition mode uses the REAL pre/post tolerances from
tolerance_transitions.csv; transitions whose exact cells were not evaluated
are listed under "missing exact transition cells". A clearly labelled
grid_projected_dfc section can be produced additionally but never replaces
exact results. lineage_replacement transitions are reported separately from
within-lineage transitions.

With no/partial data the report degrades gracefully and never fabricates
results. Paired significance is only reported when reset fingerprints
match; otherwise the section is marked reset_pairing_invalid.

CLI:
  python -m dexpbt_isaaclab.ccrs.report \
    --results results/analysis/ccrs_base_retention/ccrs_results.jsonl \
    --manifest results/analysis/ccrs_base_retention/checkpoint_manifest.csv \
    --transitions results/analysis/ccrs_base_retention/tolerance_transitions.csv \
    --out-dir results/analysis/ccrs_base_retention [--grid-projected]
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import math
from pathlib import Path
from typing import Any, Optional

from .eval_protocol import DEFAULT_TOLERANCES_M
from .analysis import (
    CCRSTable,
    CellKey,
    DFCResult,
    PRIMARY_METRIC,
    PairwiseDiff,
    capability_curve,
    compute_dfc,
    curve_difference,
    grid_projected_dfc,
    preregistered_verdicts,
)


def _grid_tolerances(table: CCRSTable) -> list[float]:
    """Grid tolerances present in results (standard grid only; exact
    transition tolerances are handled separately)."""
    evaluated = {k.fixed_tolerance for k in table.cells}
    return sorted(
        (t for t in DEFAULT_TOLERANCES_M
         if any(abs(t - e) < 1e-9 for e in evaluated)),
        reverse=True,
    )


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


def _tr_lineage_changed(t: dict) -> Optional[bool]:
    v = str(t.get("lineage_changed", "")).strip().lower()
    if v in ("true", "1"):
        return True
    if v in ("false", "0"):
        return False
    return None


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
        ax.set_ylabel("Q(k, tau) = normalized nsucc")
        ax.set_title(f"Capability curve (normalized_nsucc) — worker p{w}")
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
            grid[i, j] = table.Q(CellKey(w, it, tol))
    fig, ax = plt.subplots(figsize=(1.2 + 0.8 * len(tols), 0.6 + 0.35 * len(ckpts)))
    im = ax.imshow(grid, aspect="auto", cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks(range(len(tols)))
    ax.set_xticklabels([f"{t*100:.1f}cm" for t in tols], rotation=45, ha="right")
    ax.set_yticks(range(len(ckpts)))
    ax.set_yticklabels([f"p{w}@it{it}" for w, it in ckpts], fontsize=7)
    fig.colorbar(im, label="Q(k, tau) = normalized nsucc")
    ax.set_title("CCRS heatmap (normalized_nsucc)")
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
        elif not d.reset_pairing_valid:
            ax.annotate("!", (d.tolerance * 100, d.diff), fontsize=11,
                        ha="center", color="red")
    ax.set_xlabel("fixed tolerance (cm)")
    ax.set_ylabel("Δ normalized_nsucc")
    ax.set_title(title)
    ax.invert_xaxis()
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    out = fig_dir / fname
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out.name


def fig_dfc(dfc_rows: list[DFCResult], fig_dir: Path,
            fname: str = "dfc_decomposition.png",
            title: str = "Exact D/F/C decomposition per tolerance transition"
            ) -> Optional[str]:
    if not dfc_rows:
        return None
    plt = _plt()
    labels = []
    for r in dfc_rows:
        tag = " [LR]" if r.transition_kind == "lineage_replacement_transition" else ""
        labels.append(f"p{r.worker_id} it{r.pre_iteration}->{r.post_iteration}{tag}\n"
                      f"{r.tol_pre*100:.2f}->{r.tol_post*100:.2f}cm")
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
    ax.set_ylabel("Δ normalized_nsucc")
    ax.set_title(title + " ([LR] = lineage replacement)")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    out = fig_dir / fname
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out.name


def fig_frontier(table: CCRSTable, fig_dir: Path, tols: list[float]) -> Optional[str]:
    """Precision-retention frontier on Q: per checkpoint, tightest tolerance
    with Q >= 0.5 (x) vs Q at the loosest grid tolerance (y)."""
    if table.is_empty() or not tols:
        return None
    plt = _plt()
    pts = []
    for (w, it) in sorted({(k.worker_id, k.checkpoint_iteration) for k in table.cells}):
        loosest = max(tols)
        q_loose = table.Q(CellKey(w, it, loosest))
        tight = None
        for tol in sorted(tols):  # ascending = tightest first
            q = table.Q(CellKey(w, it, tol))
            if not math.isnan(q) and q >= 0.5:
                tight = tol
                break
        if tight is not None and not math.isnan(q_loose):
            pts.append((w, it, tight, q_loose))
    if not pts:
        return None
    fig, ax = plt.subplots(figsize=(6.5, 5))
    for w, it, tight, q_loose in pts:
        ax.scatter(tight * 100, q_loose, s=40)
        ax.annotate(f"p{w}@{it}", (tight * 100, q_loose), fontsize=7,
                    xytext=(3, 3), textcoords="offset points")
    ax.set_xlabel("tightest tolerance with Q>=0.5 (cm)  [lower = more precise]")
    ax.set_ylabel(f"Q at loosest tol ({max(tols)*100:.1f}cm)  [higher = better retention]")
    ax.set_title("Precision–retention frontier (normalized_nsucc)")
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

PRIMARY metric: normalized_nsucc Q(k, τ) = E[min(nsucc, M)]/M, M=50.
Effect threshold: |Δ normalized_nsucc| >= 0.05 (≈ 2.5 raw nsucc at M=50)
with 95% hierarchical paired-bootstrap CI excluding 0, computed ONLY when
reset fingerprints of the compared cells match exactly. success_any_rate
(P(nsucc>0)) is auxiliary and never used for these verdicts.
Tolerances in meters (cm shown for readability).

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
- **D (p5 peak vs tip)**: significant fixed-tolerance Q drop (peak - tip) →
  real policy forgetting; otherwise (live nsucc drop but flat fixed-tol Q)
  → criterion reclassification dominant.
- If reset pairing is invalid for a comparison, its verdict degrades to
  reset_pairing_invalid and NO significance conclusion is produced.
"""


def _diff_table_md(diffs: list[PairwiseDiff], table: CCRSTable,
                   a: tuple[int, int], b: tuple[int, int]) -> list[str]:
    from .analysis import METRIC_NSUCC_MEAN, METRIC_SUCCESS_ANY
    md = ["| tol (m) | tol (cm) | ΔQ | 95% CI | sig | pairing "
          "| nsuccA/B | anyA/B |",
          "|---|---|---|---|---|---|---|---|"]
    for d in diffs:
        ka = CellKey(a[0], a[1], d.tolerance)
        kb = CellKey(b[0], b[1], d.tolerance)
        ns_a = table.Q(ka, METRIC_NSUCC_MEAN)
        ns_b = table.Q(kb, METRIC_NSUCC_MEAN)
        any_a = table.Q(ka, METRIC_SUCCESS_ANY)
        any_b = table.Q(kb, METRIC_SUCCESS_ANY)
        ci = (f"[{d.ci_lo:+.3f}, {d.ci_hi:+.3f}]"
              if not math.isnan(d.ci_lo) else "-")
        pairing = "ok" if d.reset_pairing_valid else "INVALID"
        md.append(
            f"| {d.tolerance:.4f} | {d.tolerance*100:.2f} | {d.diff:+.3f} "
            f"| {ci} | {'yes' if d.significant else 'no'} | {pairing} "
            f"| {ns_a:.1f}/{ns_b:.1f} | {any_a:.2f}/{any_b:.2f} |"
        )
    return md


def generate(
    results_jsonl: Path,
    manifest_csv: Path,
    transitions_csv: Path,
    out_dir: Path,
    n_boot: int = 2000,
    include_grid_projected: bool = False,
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
        "primary_metric": PRIMARY_METRIC,
        "auxiliary_metrics": ["nsucc_mean", "success_any_rate"],
        "n_cells": len(table.cells),
        "n_rejected_rows": table.rejected_rows,
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

    # 1/2: curves + heatmap + frontier (grid tolerances, primary metric)
    figures += fig_capability_curves(table, fig_dir, tols)
    hm = fig_heatmap(table, fig_dir, tols)
    if hm:
        figures.append(hm)
    fr = fig_frontier(table, fig_dir, tols)
    if fr:
        figures.append(fr)

    # 3: p1 vs p4 peak comparison (primary metric)
    p1_peak = _find_ckpt(roles, 1, "peak") or _find_ckpt(roles, 1, "target")
    p4_peak = _find_ckpt(roles, 4, "peak") or _find_ckpt(roles, 4, "target")
    p4_minus_p1: list[PairwiseDiff] = []
    a_key = b_key = None
    if p1_peak and p4_peak:
        a_key = (4, int(p4_peak["iteration"]))
        b_key = (1, int(p1_peak["iteration"]))
        p4_minus_p1 = curve_difference(table, a=a_key, b=b_key,
                                       tolerances=tols, n_boot=n_boot)
        f = fig_pair_diff(p4_minus_p1,
                          "p4(peak) − p1(peak), Δ normalized_nsucc",
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
    p5a = p5b = None
    if p5_peak and p5_tip:
        p5a = (5, int(p5_peak["iteration"]))
        p5b = (5, int(p5_tip["iteration"]))
        p5_diffs = curve_difference(table, a=p5a, b=p5b,
                                    tolerances=tols, n_boot=n_boot)
        f = fig_pair_diff(p5_diffs, "p5 peak − tip, Δ normalized_nsucc",
                          "p5_peak_vs_tip.png", fig_dir, "peak − tip")
        if f:
            figures.append(f)
        summary["sections"]["p5_peak_minus_tip"] = [dataclasses.asdict(d) for d in p5_diffs]
    else:
        summary["missing"].append("p5 peak/tip checkpoints not found in manifest")

    # 5: D/F/C per transition — EXACT tolerances only (never snapped)
    dfc_rows: list[DFCResult] = []
    missing_exact: list[str] = []
    pairing_invalid_dfc: list[str] = []
    for t in transitions:
        try:
            a = float(t["pre_tolerance"])
            b = float(t["post_tolerance"])
            if not b < a:
                continue
            res = compute_dfc(
                table,
                worker_id=int(t["worker_id"]),
                pre_iteration=int(t["pre_iteration"]),
                post_iteration=int(t["post_iteration"]),
                tol_pre=a, tol_post=b, n_boot=n_boot,
                mode="exact",
                lineage_changed=_tr_lineage_changed(t),
            )
            dfc_rows.append(res)
            if not res.reset_pairing_valid:
                pairing_invalid_dfc.append(
                    f"p{t['worker_id']} it{t['pre_iteration']}->{t['post_iteration']}")
        except KeyError:
            missing_exact.append(
                f"p{t.get('worker_id')} it{t.get('pre_iteration')}->"
                f"{t.get('post_iteration')} tol {t.get('pre_tolerance')}->"
                f"{t.get('post_tolerance')}")
        except (ValueError, TypeError):
            continue
    if missing_exact:
        summary["missing"].append(
            f"missing exact transition cells for {len(missing_exact)} transitions")
        summary["missing_exact_transitions"] = missing_exact
    if pairing_invalid_dfc:
        summary["missing"].append(
            f"reset_pairing_invalid for {len(pairing_invalid_dfc)} D/F/C transitions")
    within = [r for r in dfc_rows if r.transition_kind == "within_lineage"]
    lineage_repl = [r for r in dfc_rows
                    if r.transition_kind == "lineage_replacement_transition"]
    if dfc_rows:
        f = fig_dfc(dfc_rows, fig_dir)
        if f:
            figures.append(f)
        summary["sections"]["dfc_exact"] = [dataclasses.asdict(r) for r in dfc_rows]
        bad = [r for r in dfc_rows if r.identity_residual > 1e-9]
        if bad:
            summary["missing"].append(f"D=F+C identity violated in {len(bad)} rows (BUG)")
    elif transitions:
        summary["missing"].append(
            "no exact D/F/C computable (no transitions have exact cells evaluated)")

    # 5b: optional grid-projected D/F/C (compatibility, clearly labelled)
    gp_rows: list[DFCResult] = []
    if include_grid_projected and tols:
        for t in transitions:
            try:
                gp_rows.append(grid_projected_dfc(
                    table,
                    worker_id=int(t["worker_id"]),
                    pre_iteration=int(t["pre_iteration"]),
                    post_iteration=int(t["post_iteration"]),
                    tol_pre_real=float(t["pre_tolerance"]),
                    tol_post_real=float(t["post_tolerance"]),
                    grid=tols, n_boot=n_boot,
                    lineage_changed=_tr_lineage_changed(t),
                ))
            except (KeyError, ValueError, TypeError):
                continue
        if gp_rows:
            summary["sections"]["grid_projected_dfc"] = [
                dataclasses.asdict(r) for r in gp_rows]

    # 6: pre-registered verdicts (primary metric only)
    verdicts = preregistered_verdicts(p4_minus_p1, p5_diffs)
    summary["verdicts"] = verdicts

    # ---- markdown body ----
    md.append("\n## Verdicts (auto-applied pre-registered criteria, "
              "normalized_nsucc)\n")
    md.append(f"- p1 vs p4: **{verdicts['verdict_p1_p4']}**")
    md.append(f"- p5 peak→tip: **{verdicts['verdict_p5']}**")
    md.append("")
    if p4_minus_p1 and a_key and b_key:
        md.append("## p4 − p1 (Δ normalized_nsucc; nsucc_mean and "
                  "success_any_rate shown as context only)\n")
        md += _diff_table_md(p4_minus_p1, table, a_key, b_key)
        md.append("")
    if p5_diffs and p5a and p5b:
        md.append("## p5 peak − tip (Δ normalized_nsucc)\n")
        md += _diff_table_md(p5_diffs, table, p5a, p5b)
        md.append("")

    def _dfc_table(rows: list[DFCResult]) -> list[str]:
        out = ["| worker | iters | tol a→b (m) | D | F | C | n | pairing |",
               "|---|---|---|---|---|---|---|---|"]
        for r in rows:
            out.append(
                f"| p{r.worker_id} | {r.pre_iteration}→{r.post_iteration} "
                f"| {r.tol_pre:.4f}→{r.tol_post:.4f} "
                f"| {r.D:+.3f} | {r.F:+.3f} | {r.C:+.3f} | {r.n_samples} "
                f"| {'ok' if r.reset_pairing_valid else 'INVALID'} |")
        return out

    if within:
        md.append("## Exact D/F/C — within-lineage transitions "
                  "(policy forgetting analysis)\n")
        md += _dfc_table(within)
        md.append("")
    if lineage_repl:
        md.append("## Exact D/F/C — lineage_replacement_transitions\n")
        md.append("F here includes PBT copy/replacement effects and must NOT "
                  "be read as ordinary policy update or forgetting.\n")
        md += _dfc_table(lineage_repl)
        md.append("")
    if missing_exact:
        md.append("## Missing exact transition cells\n")
        for m in missing_exact:
            md.append(f"- {m}")
        md.append("")
    if gp_rows:
        md.append("## grid_projected_dfc (compatibility only — NOT exact, "
                  "not for primary conclusions)\n")
        md += _dfc_table(gp_rows)
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
    ap.add_argument("--grid-projected", action="store_true",
                    help="Additionally emit grid_projected_dfc (labelled)")
    args = ap.parse_args()
    s = generate(
        results_jsonl=Path(args.results),
        manifest_csv=Path(args.manifest),
        transitions_csv=Path(args.transitions),
        out_dir=Path(args.out_dir),
        n_boot=args.n_boot,
        include_grid_projected=args.grid_projected,
    )
    print(f"[ccrs.report] cells={s['n_cells']} missing={len(s['missing'])} "
          f"-> {Path(args.out_dir) / 'CCRS_REPORT.md'}")


if __name__ == "__main__":
    main()
