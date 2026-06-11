#!/usr/bin/env python3
"""Analyze an ERD-MPC run and write markdown/json reports."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

try:
    from tensorboard.backend.event_processing import event_accumulator

    _TB_OK = True
except Exception:  # pragma: no cover
    _TB_OK = False


ERD_TAGS = [
    "erd/active",
    "erd/scorer_active",
    "erd/distill_active",
    "erd/distill_block_reason",
    "erd/risky_fraction",
    "erd/scorer_loss",
    "erd/distill_loss",
    "erd/total_loss",
    "erd/best_score",
    "erd/current_actor_score",
    "erd/score_margin",
    "erd/best_success_prob",
    "erd/best_improve_prob",
    "erd/best_drop_prob",
    "erd/actor_success_prob",
    "erd/actor_drop_prob",
    "erd/action_delta_norm",
    "erd/distill_grad_ratio",
    "erd/auto_scale_factor",
    "erd/target_valid_fraction",
]


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    out = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return out


def _stats(values: list[float]) -> dict[str, float]:
    vals = [float(v) for v in values if v == v]
    if not vals:
        return {"n": 0}
    return {
        "n": len(vals),
        "mean": statistics.fmean(vals),
        "min": min(vals),
        "max": max(vals),
        "last": vals[-1],
    }


def _collect_tb(summary_dirs: list[Path]) -> dict[str, list[tuple[int, float]]]:
    merged: dict[str, list[tuple[int, float]]] = {}
    if not _TB_OK:
        return merged
    for d in summary_dirs:
        for ev in sorted(d.glob("events.out.tfevents.*")):
            try:
                ea = event_accumulator.EventAccumulator(
                    str(ev), size_guidance={"scalars": 0}
                )
                ea.Reload()
            except Exception:
                continue
            for tag in ea.Tags().get("scalars", []):
                if not (tag in ERD_TAGS or tag.startswith("erd/target_pos_rate_")):
                    continue
                merged.setdefault(tag, [])
                for s in ea.Scalars(tag):
                    merged[tag].append((int(s.step), float(s.value)))
    for v in merged.values():
        v.sort(key=lambda x: x[0])
    return merged


def _policy_summary(run_dir: Path, pid: int) -> dict[str, Any]:
    metrics = run_dir / "dexpbt_8gpu_ws" / "metrics" / f"training_curve_policy_{pid:02d}.jsonl"
    rows = _read_jsonl(metrics)
    out: dict[str, Any] = {"policy": pid, "curve_points": len(rows)}
    if rows:
        means = [float(r.get("nsucc_mean", 0.0)) for r in rows]
        maxs = [float(r.get("nsucc_max", 0.0)) for r in rows]
        tols = [float(r.get("success_tolerance", 1.0)) for r in rows]
        out.update(
            nsucc_peak=max(means),
            nsucc_end=means[-1],
            nsucc_max_peak=max(maxs),
            nsucc_max_end=maxs[-1],
            frames_end=int(rows[-1].get("env_frames", 0)),
            tol_end=tols[-1],
            tol_min=min(tols),
            collapse_count=sum(1 for r in rows if r.get("early_collapse_triggered")),
            rollback_count=max(int(r.get("rollback_count", 0) or 0) for r in rows),
        )
    summaries = sorted((run_dir / f"policy_{pid:02d}" / "rl_games").glob("attempt_*/summaries"))
    tb = _collect_tb(summaries)
    out["tb_available"] = bool(tb)
    for tag in ERD_TAGS:
        vals = [v for _, v in tb.get(tag, [])]
        out[tag.split("/", 1)[1]] = _stats(vals)
    out["scorer_active_points"] = sum(1 for _, v in tb.get("erd/scorer_active", []) if v > 0)
    out["distill_active_points"] = sum(1 for _, v in tb.get("erd/distill_active", []) if v > 0)
    out["pos_rate_last"] = {
        tag.split("erd/target_pos_rate_", 1)[1]: vals[-1][1]
        for tag, vals in tb.items()
        if tag.startswith("erd/target_pos_rate_") and vals
    }
    return out


def analyze(run_dir: Path, baseline: Path | None = None) -> dict[str, Any]:
    policies = [_policy_summary(run_dir, pid) for pid in range(6)]
    latest = [p.get("nsucc_end") for p in policies if "nsucc_end" in p]
    peaks = [p.get("nsucc_peak") for p in policies if "nsucc_peak" in p]
    rec_notes = []
    if sum(p.get("distill_active_points", 0) for p in policies) == 0:
        rec_notes.append("ERD distillation never activated; inspect warmup/gates.")
    margins = [
        p.get("score_margin", {}).get("last", 0.0)
        for p in policies
        if p.get("score_margin", {}).get("n", 0)
    ]
    if margins and max(margins) <= 1e-6:
        rec_notes.append("Score margin is near zero; candidate search may be useless.")
    ratios = [
        p.get("distill_grad_ratio", {}).get("max", 0.0)
        for p in policies
        if p.get("distill_grad_ratio", {}).get("n", 0)
    ]
    if ratios and max(ratios) > 0.2:
        rec_notes.append("Grad ratio exceeded configured cap; reduce distill weights.")
    report: dict[str, Any] = {
        "run_dir": str(run_dir),
        "tb_available": _TB_OK,
        "per_policy": policies,
        "population": {
            "latest_mean": statistics.fmean(latest) if latest else 0.0,
            "latest_median": statistics.median(latest) if latest else 0.0,
            "latest_max": max(latest) if latest else 0.0,
            "peak_mean_avg": statistics.fmean(peaks) if peaks else 0.0,
            "peak_mean_max": max(peaks) if peaks else 0.0,
            "exceeds_30": sum(1 for v in latest if v >= 30.0),
            "exceeds_40": sum(1 for v in latest if v >= 40.0),
            "exceeds_45": sum(1 for v in latest if v >= 45.0),
        },
        "recommendation": {"notes": rec_notes, "continue": not rec_notes},
    }
    return report


def _write_md(report: dict[str, Any], out_dir: Path) -> None:
    pop = report["population"]
    lines = [
        "# ERD-MPC Run Report",
        "",
        f"- run_dir: `{report['run_dir']}`",
        f"- tensorboard_reader_available: {report['tb_available']}",
        "",
        "## Population",
        f"- latest mean/median/max: {pop['latest_mean']:.2f} / {pop['latest_median']:.2f} / {pop['latest_max']:.2f}",
        f"- peak mean avg/max: {pop['peak_mean_avg']:.2f} / {pop['peak_mean_max']:.2f}",
        f"- latest policies >=30/40/45: {pop['exceeds_30']} / {pop['exceeds_40']} / {pop['exceeds_45']}",
        "",
        "## Recommendation",
        f"- continue: **{report['recommendation']['continue']}**",
    ]
    for note in report["recommendation"]["notes"]:
        lines.append(f"  - note: {note}")
    lines += ["", "## Per-policy"]
    for p in report["per_policy"]:
        lines += [
            "",
            f"### policy_{p['policy']:02d}",
            f"- frames_end: {p.get('frames_end', 0)}",
            f"- nsucc peak/end: {p.get('nsucc_peak', 0.0):.2f} / {p.get('nsucc_end', 0.0):.2f}",
            f"- tol end/min: {p.get('tol_end', 1.0):.4f} / {p.get('tol_min', 1.0):.4f}",
            f"- scorer_active_points: {p.get('scorer_active_points', 0)}",
            f"- distill_active_points: {p.get('distill_active_points', 0)}",
            f"- risky_fraction: {p.get('risky_fraction', {})}",
            f"- score_margin: {p.get('score_margin', {})}",
            f"- action_delta_norm: {p.get('action_delta_norm', {})}",
            f"- distill_grad_ratio: {p.get('distill_grad_ratio', {})}",
            f"- auto_scale_factor: {p.get('auto_scale_factor', {})}",
            f"- pos_rate_last: {p.get('pos_rate_last', {})}",
        ]
    (out_dir / "ERD_MPC_RUN_REPORT.md").write_text("\n".join(lines))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--out", default=None)
    ap.add_argument("--baseline", default=None)
    args = ap.parse_args()
    run_dir = Path(args.run_dir).resolve()
    out_dir = Path(args.out).resolve() if args.out else run_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    report = analyze(run_dir, Path(args.baseline).resolve() if args.baseline else None)
    (out_dir / "ERD_MPC_RUN_REPORT.json").write_text(json.dumps(report, indent=2))
    _write_md(report, out_dir)
    print(f"[analyze_erd_mpc_run] wrote report to {out_dir}")


if __name__ == "__main__":
    main()

