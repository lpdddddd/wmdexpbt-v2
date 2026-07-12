#!/usr/bin/env python3
"""Post-hoc analysis for Progress-Conditioned CSER-DexPBT runs.

Reads a run directory and emits CSV + JSON + Markdown diagnostics focused on the
progress_refine method's key questions:

  * nsucc_mean / nsucc_max over frames (per policy + population)
  * population mean vs bottom-slot mean
  * curriculum tolerance + normalized curriculum progress
  * gate distribution over training (mean / p50 / p90 / active_frac)
  * active states binned by normalized error (proximity structure)
  * progress + oscillation distribution
  * retrieval valid rate, teacher confidence, WM fallback rate
  * snippet loss contribution
  * post-shrink retention, collapse + rescue events
  * comparison against cser_dexpbt_conservative and cser_dexpbt_2cm_bridge runs

All inputs are optional: missing files are skipped, never fatal.  TensorBoard
scalars are used when the ``tensorboard`` package is available; otherwise the
JSONL training curves + adaptive_precision sidecars drive the analysis.

Usage:
    python analyze_progress_refine.py --run-dir <RUN> [--out-dir <OUT>] \
        [--compare-conservative <RUN>] [--compare-bridge <RUN>]
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Optional


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    for line in path.read_text(errors="ignore").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except Exception:
            continue
    return rows


def _ws(run_dir: Path) -> Path:
    cand = run_dir / "dexpbt_8gpu_ws"
    return cand if cand.exists() else run_dir


def _metrics_dir(run_dir: Path) -> Path:
    return _ws(run_dir) / "metrics"


def _training_curves(run_dir: Path) -> dict[int, list[dict[str, Any]]]:
    md = _metrics_dir(run_dir)
    out: dict[int, list[dict[str, Any]]] = {}
    if not md.exists():
        return out
    for p in sorted(md.glob("training_curve_policy_*.jsonl")):
        try:
            idx = int(p.stem.split("_")[-1])
        except Exception:
            continue
        out[idx] = _read_jsonl(p)
    return out


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    k = max(0, min(len(s) - 1, int(round(q * (len(s) - 1)))))
    return float(s[k])


def _tb_scalars(run_dir: Path) -> dict[str, list[tuple[int, float]]]:
    """Best-effort TensorBoard scalar extraction (optional dependency)."""
    try:
        from tensorboard.backend.event_processing.event_accumulator import (
            EventAccumulator,
        )
    except Exception:
        return {}
    out: dict[str, list[tuple[int, float]]] = {}
    for ev in _ws(run_dir).rglob("events.out.tfevents.*"):
        try:
            acc = EventAccumulator(str(ev.parent))
            acc.Reload()
            for tag in acc.Tags().get("scalars", []):
                if not (tag.startswith("progress_refine/") or tag.startswith("adaptive_consolidation/") or tag.startswith("snippet_retrieval/")):
                    continue
                out.setdefault(tag, [])
                for e in acc.Scalars(tag):
                    out[tag].append((int(e.step), float(e.value)))
        except Exception:
            continue
    return out


def _population_summary(curves: dict[int, list[dict[str, Any]]]) -> dict[str, Any]:
    last_means: list[float] = []
    per_policy: dict[str, Any] = {}
    for idx, rows in curves.items():
        if not rows:
            continue
        last = rows[-1]
        nm = float(last.get("nsucc_mean", 0.0) or 0.0)
        nx = float(last.get("nsucc_max", 0.0) or 0.0)
        tol = float(last.get("success_tolerance", 0.0) or 0.0)
        last_means.append(nm)
        per_policy[str(idx)] = {
            "last_frame": int(last.get("env_frames", 0) or 0),
            "nsucc_mean": nm,
            "nsucc_max": nx,
            "success_tolerance": tol,
            "target_success_tolerance": float(last.get("target_success_tolerance", 0.0) or 0.0),
            "peak_nsucc_mean": max((float(r.get("nsucc_mean", 0.0) or 0.0) for r in rows), default=0.0),
            "best_tolerance": min(
                (float(r.get("success_tolerance", 1.0) or 1.0) for r in rows), default=1.0
            ),
        }
    pop_mean = sum(last_means) / len(last_means) if last_means else 0.0
    bottom_mean = min(last_means) if last_means else 0.0
    return {
        "population_mean_nsucc": pop_mean,
        "bottom_slot_nsucc": bottom_mean,
        "n_policies": len(per_policy),
        "per_policy": per_policy,
    }


def _curriculum_progress(eps_init: float, eps_target: float, eps_cur: float) -> float:
    if eps_init <= 0 or eps_target <= 0 or eps_cur <= 0:
        return 0.0
    denom = math.log((eps_init + 1e-6) / (eps_target + 1e-6))
    if abs(denom) < 1e-9:
        return 0.0
    c = math.log((eps_init + 1e-6) / (eps_cur + 1e-6)) / denom
    return float(min(1.0, max(0.0, c)))


def _tb_series_stats(series: list[tuple[int, float]]) -> dict[str, float]:
    if not series:
        return {"mean": 0.0, "last": 0.0, "p50": 0.0, "p90": 0.0, "n": 0.0}
    vals = [v for _, v in series]
    return {
        "mean": float(sum(vals) / len(vals)),
        "last": float(series[-1][1]),
        "p50": _percentile(vals, 0.5),
        "p90": _percentile(vals, 0.9),
        "n": float(len(vals)),
    }


def analyze(run_dir: Path, out_dir: Path, compares: dict[str, Path]) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    curves = _training_curves(run_dir)
    tb = _tb_scalars(run_dir)

    summary: dict[str, Any] = {"run_dir": str(run_dir)}
    summary["population"] = _population_summary(curves)

    # Per-policy curve CSV.
    curve_csv = out_dir / "progress_refine_curves.csv"
    with curve_csv.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(
            ["policy", "env_frames", "nsucc_mean", "nsucc_max",
             "success_tolerance", "target_tolerance", "curriculum_progress"]
        )
        for idx, rows in sorted(curves.items()):
            eps_init = max((float(r.get("success_tolerance", 0.0) or 0.0) for r in rows), default=0.075)
            for r in rows:
                tol = float(r.get("success_tolerance", 0.0) or 0.0)
                tgt = float(r.get("target_success_tolerance", 0.0) or 0.0)
                w.writerow([
                    idx, int(r.get("env_frames", 0) or 0),
                    float(r.get("nsucc_mean", 0.0) or 0.0),
                    float(r.get("nsucc_max", 0.0) or 0.0),
                    tol, tgt, _curriculum_progress(eps_init, tgt or 0.018, tol),
                ])

    # Gate / retrieval / teacher / adaptive-consolidation series from TB.
    gate_keys = [
        "progress_refine/gate_mean", "progress_refine/gate_p50", "progress_refine/gate_p90",
        "progress_refine/active_frac", "progress_refine/history_valid_frac",
        "progress_refine/normalized_error_mean", "progress_refine/normalized_error_active_mean",
        "progress_refine/recent_progress_mean", "progress_refine/oscillation_mean",
        "progress_refine/teacher_confidence_mean", "progress_refine/retrieval_valid_frac",
        "progress_refine/retrieval_budget_frac", "progress_refine/effective_sample_size",
        "progress_refine/loss", "progress_refine/fallback_frac", "progress_refine/nonfinite_count",
        "snippet_retrieval/tolerance_weight_mean", "snippet_retrieval/log_tolerance_gap_mean",
        "snippet_retrieval/no_compatible_frac",
        "adaptive_consolidation/curriculum_progress", "adaptive_consolidation/stage",
        "adaptive_consolidation/local_retention", "adaptive_consolidation/probation_active",
        "adaptive_consolidation/shrink_blocked", "adaptive_consolidation/hard_collapse_streak",
        "adaptive_consolidation/hard_collapse_adaptive_only",
        "adaptive_consolidation/hard_collapse_legacy_only",
        "adaptive_consolidation/hard_collapse_both",
        "adaptive_consolidation/rollback_target_source",
        "adaptive_consolidation/rollback_log_gap",
        "adaptive_consolidation/rollback_fallback_used",
        "adaptive_consolidation/rollback_freeze_only_count",
        "adaptive_consolidation/rescue_live_selected",
        "adaptive_consolidation/rescue_archive_selected",
        "adaptive_consolidation/rescue_ordinary_fallback_selected",
        "adaptive_consolidation/rescue_no_source",
        "adaptive_consolidation/rescue_burnin_blocked",
        "adaptive_consolidation/rescue_burnin_bypassed",
        "adaptive_consolidation/rescue_emergency_bypass",
        "adaptive_consolidation/rescue_executed_count",
        "adaptive_consolidation/rescue_quota_used",
    ]
    gate_stats = {k: _tb_series_stats(tb.get(k, [])) for k in gate_keys}
    summary["gate_and_retrieval"] = gate_stats

    # Adaptive precision sidecars (probation / retention / rescue).
    ap_dir = _ws(run_dir) / "adaptive_precision"
    adaptive: dict[str, Any] = {}
    if ap_dir.exists():
        for p in sorted(ap_dir.glob("*.json")):
            try:
                data = json.loads(p.read_text())
            except Exception:
                continue
            prob = data.get("probation", {})
            adaptive[p.stem] = {
                "schema_version": int(data.get("schema_version", 0) or 0),
                "probation_active": bool(prob.get("active", False)),
                "probation_stable_ticks": int(prob.get("stable_ticks", 0)),
                "hard_collapse_streak": int(prob.get("hard_collapse_streak", 0)),
                "degradation_streak": int(data.get("degradation_streak", 0)),
                "rescue_executed": int(data.get("rescue_executed", 0)),
                "rescue_executed_for_lineage": int(
                    data.get("rescue_executed_for_lineage", 0) or 0
                ),
                "last_update_frame": int(data.get("last_update_frame", -1) or -1),
                "freeze_only_streak": int(data.get("freeze_only_streak", 0) or 0),
                "retention_bands": len((data.get("retention", {}) or {}).get("states", {})),
            }
    summary["adaptive_precision"] = adaptive

    # Collapse / rescue events.
    safety = _read_jsonl(_metrics_dir(run_dir) / "curriculum_safety_events.jsonl")
    copy_events = _read_jsonl(_metrics_dir(run_dir) / "pbt_copy_events.jsonl")
    rescue_types: dict[str, int] = {}
    for e in copy_events:
        et = str(e.get("event_type", ""))
        if "rescue" in et or "adaptive" in et:
            rescue_types[et] = rescue_types.get(et, 0) + 1
    trigger_sources: dict[str, int] = {}
    rollback_sources: dict[str, int] = {}
    for e in safety + copy_events:
        ts = e.get("adaptive_probation_hard_collapse_trigger_source") or e.get(
            "hard_collapse_trigger_source"
        )
        if ts:
            trigger_sources[str(ts)] = trigger_sources.get(str(ts), 0) + 1
        rs = e.get("adaptive_rollback_target_source") or e.get("rollback_target_source")
        if rs:
            rollback_sources[str(rs)] = rollback_sources.get(str(rs), 0) + 1
    summary["events"] = {
        "curriculum_safety_events": len(safety),
        "pbt_copy_events": len(copy_events),
        "rescue_events": sum(1 for e in copy_events if "rescue" in str(e.get("event_type", ""))),
        "rescue_event_types": rescue_types,
        "hard_collapse_trigger_sources": trigger_sources,
        "rollback_target_sources": rollback_sources,
        "adaptive_executed": rescue_types.get("adaptive_degradation_rescue_executed", 0),
        "adaptive_rolled_back": rescue_types.get(
            "adaptive_degradation_rescue_rolled_back", 0
        ),
    }

    # Sanity checks the reviewer asked about.
    gm = gate_stats["progress_refine/gate_mean"]
    ne_all = gate_stats["progress_refine/normalized_error_mean"]["mean"]
    ne_active = gate_stats["progress_refine/normalized_error_active_mean"]["mean"]
    loss_series = tb.get("progress_refine/loss", [])
    loss_nonzero = any(abs(v) > 0 for _, v in loss_series)
    changed = tb.get("snippet_wm_rerank/changed_frac", [])
    summary["checks"] = {
        "gate_concentrates_near_boundary": bool(ne_active <= ne_all + 1e-9) if ne_all else None,
        "gate_mean": gm.get("mean", 0.0),
        "loss_nonzero_at_some_point": bool(loss_nonzero) if loss_series else None,
        "wm_changed_selection_frac_mean": _tb_series_stats(changed).get("mean", 0.0) if changed else None,
        "retrieval_valid_frac_mean": gate_stats["progress_refine/retrieval_valid_frac"].get("mean", 0.0),
        "fallback_frac_mean": gate_stats["progress_refine/fallback_frac"].get("mean", 0.0),
        "nonfinite_total": gate_stats["progress_refine/nonfinite_count"].get("mean", 0.0),
    }

    # Comparisons.
    comparisons: dict[str, Any] = {}
    for label, cdir in compares.items():
        if cdir is None:
            continue
        ccurves = _training_curves(Path(cdir))
        comparisons[label] = _population_summary(ccurves)
    summary["comparisons"] = comparisons

    (out_dir / "progress_refine_summary.json").write_text(json.dumps(summary, indent=2))

    # Markdown summary.
    md = ["# Progress-Conditioned CSER-DexPBT analysis", "", f"Run: `{run_dir}`", ""]
    pop = summary["population"]
    md += [
        "## Population",
        f"- policies: {pop['n_policies']}",
        f"- population mean nsucc: {pop['population_mean_nsucc']:.2f}",
        f"- bottom slot nsucc: {pop['bottom_slot_nsucc']:.2f}",
        "",
        "## Gate & retrieval (TB means)",
    ]
    for k in gate_keys:
        st = gate_stats[k]
        md.append(f"- {k}: mean={st['mean']:.4f} last={st['last']:.4f} p90={st['p90']:.4f}")
    md += ["", "## Checks"]
    for k, v in summary["checks"].items():
        md.append(f"- {k}: {v}")
    if comparisons:
        md += ["", "## Comparisons (population mean nsucc)"]
        md.append(f"- this run: {pop['population_mean_nsucc']:.2f}")
        for label, cs in comparisons.items():
            md.append(f"- {label}: {cs['population_mean_nsucc']:.2f}")
    (out_dir / "progress_refine_summary.md").write_text("\n".join(md) + "\n")
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--compare-conservative", default=None)
    ap.add_argument("--compare-bridge", default=None)
    args = ap.parse_args()
    run_dir = Path(args.run_dir)
    out_dir = Path(args.out_dir) if args.out_dir else run_dir / "progress_refine_analysis"
    compares = {
        "cser_dexpbt_conservative": args.compare_conservative,
        "cser_dexpbt_2cm_bridge": args.compare_bridge,
    }
    summary = analyze(run_dir, out_dir, {k: v for k, v in compares.items() if v})
    print(json.dumps(summary.get("checks", {}), indent=2))
    print(f"[analyze_progress_refine] wrote {out_dir}")


if __name__ == "__main__":
    main()
