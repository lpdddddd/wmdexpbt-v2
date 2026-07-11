#!/usr/bin/env python3
"""Phase A4 P3 (2026-06-09) -- 3-way comparison for fastcurr_inherit 800M.

Compare three RSSM coupling=none 800M runs:

  1. fastcurr_inherit  -- adaptive_fast + source_safe + promising guard
  2. fastcurr (legacy) -- adaptive_fast only (commit 002f008 baseline)
  3. reference         -- normal curriculum (P2.1 smoke 800M)

Reports curriculum / PBT inheritance / promising-protection metrics and
prints an acceptance verdict per plan section G:

  * primary curriculum: median final tolerance <= --target-tolerance OR
    >= 4/6 policies <= --target-tolerance
  * primary safety: end_mean median >= --safety-end-mean (default 25) AND
    no unmitigated collapse
  * churn: median attempts/policy reduced vs fastcurr
  * outliers: median tolerance excluding high-attempt outliers improves

Inputs:
  --fastcurr_inherit RUN_DIR   adaptive_fast + source_safe run
  --fastcurr RUN_DIR           adaptive_fast-only baseline run
  --reference RUN_DIR          normal-curriculum reference run
  --target-tolerance FLOAT     final-tolerance acceptance (default 0.04)
  --target-frames INT          env-frame budget (default 800M)
  --safety-end-mean FLOAT      end_mean safety threshold (default 25)
  --high-attempt-threshold INT min attempts to flag a policy as outlier (5)
  --out-json PATH              optional JSON dump path

Outputs:
  Markdown report on stdout + JSON dump at
  ``$FASTCURR_INHERIT_RUN_DIR/rssm_fastcurr_inherit_800m_compare.json``.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Optional


def _load_curves(run_dir: Path) -> dict[int, list[dict[str, Any]]]:
    metrics_dir = run_dir / "dexpbt_8gpu_ws" / "metrics"
    rows: dict[int, list[dict[str, Any]]] = defaultdict(list)
    if not metrics_dir.is_dir():
        return rows
    for fp in sorted(metrics_dir.glob("training_curve_policy_*.jsonl")):
        for line in fp.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            pid = r.get("policy_idx")
            if pid is None:
                continue
            rows[int(pid)].append(r)
    for pid in rows:
        rows[pid].sort(key=lambda r: int(r.get("env_frames", 0)))
    return rows


def _load_copy_events(run_dir: Path) -> list[dict[str, Any]]:
    path = run_dir / "dexpbt_8gpu_ws" / "metrics" / "pbt_copy_events.jsonl"
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def _per_policy_summary(
    curves: dict[int, list[dict]],
    target_frames: int,
) -> dict[int, dict]:
    out: dict[int, dict] = {}
    for pid, rows in curves.items():
        rows = [r for r in rows if int(r.get("env_frames", 0)) <= target_frames]
        if not rows:
            continue
        means = [float(r.get("nsucc_mean", 0) or 0) for r in rows]
        maxes = [float(r.get("nsucc_max", 0) or 0) for r in rows]
        tols = [
            float(r.get("success_tolerance", float("nan")) or float("nan"))
            for r in rows
        ]

        def first_at(thr: float) -> Optional[int]:
            for r in rows:
                if float(r.get("nsucc_mean", 0) or 0) >= thr:
                    return int(r.get("env_frames", 0))
            return None

        def tol_at(frame_budget: int) -> Optional[float]:
            best = None
            for r in rows:
                if int(r.get("env_frames", 0)) <= frame_budget:
                    best = float(
                        r.get("success_tolerance", float("nan")) or float("nan")
                    )
            return best

        shrinks: list[tuple[int, float, float, float, str]] = []
        prev_tol = None
        for r in rows:
            cur_tol = float(
                r.get("success_tolerance", float("nan")) or float("nan")
            )
            if (
                prev_tol is not None
                and not math.isnan(prev_tol)
                and not math.isnan(cur_tol)
                and cur_tol + 1e-9 < prev_tol
            ):
                factor = cur_tol / max(prev_tol, 1e-12)
                reason = str(r.get("curriculum_shrink_reason", ""))
                shrinks.append(
                    (int(r.get("env_frames", 0)), float(prev_tol), float(cur_tol), float(factor), reason)
                )
            prev_tol = cur_tol

        last_row = rows[-1]
        attempts = int(
            last_row.get("self_restart_count", 0)
            + last_row.get("cross_restart_count", 0)
            + last_row.get("recovery_restart_count", 0)
            + 1  # initial attempt
        )
        out[pid] = {
            "n_rows": len(rows),
            "final_frames": int(last_row.get("env_frames", 0)),
            "peak_mean": max(means) if means else float("nan"),
            "peak_max": max(maxes) if maxes else float("nan"),
            "end_mean": means[-1] if means else float("nan"),
            "end_max": maxes[-1] if maxes else float("nan"),
            "retention": (means[-1] / max(max(means), 1e-9)) if means else float("nan"),
            "first_nsucc_10": first_at(10.0),
            "first_nsucc_20": first_at(20.0),
            "first_nsucc_30": first_at(30.0),
            "tolerance_at_200M": tol_at(200_000_000),
            "tolerance_at_400M": tol_at(400_000_000),
            "tolerance_at_600M": tol_at(600_000_000),
            "tolerance_at_800M": tol_at(target_frames),
            "final_tolerance": tols[-1] if tols else float("nan"),
            "min_tolerance": (
                min(t for t in tols if not math.isnan(t))
                if any(not math.isnan(t) for t in tols)
                else float("nan")
            ),
            "shrinks": shrinks,
            "shrink_count": len(shrinks),
            "rollback_count": int(last_row.get("rollback_count", 0) or 0),
            "cross_restart_count": int(
                last_row.get("cross_restart_count", 0) or 0
            ),
            "self_restart_count": int(
                last_row.get("self_restart_count", 0) or 0
            ),
            "recovery_restart_count": int(
                last_row.get("recovery_restart_count", 0) or 0
            ),
            "attempts": attempts,
            "speed_mode": str(last_row.get("curriculum_speed_mode", "")),
            "inherit_mode": str(
                last_row.get("pbt_curriculum_inherit_mode", "reset")
            ),
            "inherit_count": int(
                last_row.get("curriculum_inherited_tolerance_count", 0) or 0
            ),
            "reset_count": int(
                last_row.get("curriculum_reset_to_initial_tolerance_count", 0)
                or 0
            ),
            "missing_meta_count": int(
                last_row.get(
                    "curriculum_missing_source_curriculum_metadata_count", 0
                )
                or 0
            ),
            "promising_protect_until": int(
                last_row.get("promising_protect_until", 0) or 0
            ),
        }
    return out


def _summary(by_pid: dict[int, dict], key: str) -> dict[str, Any]:
    vals = [
        v[key]
        for v in by_pid.values()
        if v.get(key) is not None
        and not (isinstance(v.get(key), float) and math.isnan(v[key]))
    ]
    if not vals:
        return {"median": None, "min": None, "max": None, "n": 0}
    return {
        "median": float(statistics.median(vals)),
        "min": float(min(vals)),
        "max": float(max(vals)),
        "n": len(vals),
    }


def _count_above(by_pid: dict[int, dict], key: str, thr: float) -> int:
    return sum(
        1 for v in by_pid.values()
        if v.get(key) is not None
        and not (isinstance(v[key], float) and math.isnan(v[key]))
        and v[key] >= thr
    )


def _count_below_eq(by_pid: dict[int, dict], key: str, thr: float) -> int:
    return sum(
        1 for v in by_pid.values()
        if v.get(key) is not None
        and not (isinstance(v[key], float) and math.isnan(v[key]))
        and v[key] <= thr
    )


def _shrink_factor_histogram(by_pid: dict[int, dict]) -> dict[str, int]:
    buckets: Counter[str] = Counter()
    for v in by_pid.values():
        for _, _, _, factor, _ in v["shrinks"]:
            if abs(factor - 0.90) < 0.01:
                buckets["strong_~0.90"] += 1
            elif abs(factor - 0.925) < 0.01:
                buckets["mid_~0.925"] += 1
            elif abs(factor - 0.95) < 0.01:
                buckets["low_~0.95"] += 1
            else:
                buckets[f"other_~{factor:.3f}"] += 1
    return dict(buckets)


def _scrape_engineering_signals(run_dir: Path) -> dict[str, int]:
    sigkill = 0
    runtime_err = 0
    for fp in sorted(run_dir.glob("policy_*/train.log")):
        try:
            txt = fp.read_text(errors="replace")
        except Exception:
            continue
        sigkill += len(re.findall(r"exited with code 137", txt))
        runtime_err += len(re.findall(r"RuntimeError|Traceback", txt))
    return {
        "sigkill_137_total": sigkill,
        "runtime_error_total": runtime_err,
    }


def _scrape_fps(run_dir: Path) -> dict[str, float]:
    fps_step: list[float] = []
    fps_total: list[float] = []
    for fp in sorted(run_dir.glob("policy_*/train.log")):
        try:
            txt = fp.read_text(errors="replace")
        except Exception:
            continue
        steps = [int(x) for x in re.findall(r"fps step:\s+(\d+)", txt)]
        totals = [int(x) for x in re.findall(r"fps total:\s+(\d+)", txt)]
        if steps:
            fps_step.append(sum(steps[-50:]) / max(1, len(steps[-50:])))
        if totals:
            fps_total.append(sum(totals[-50:]) / max(1, len(totals[-50:])))
    return {
        "fps_step_mean": float(sum(fps_step) / max(1, len(fps_step)))
        if fps_step
        else float("nan"),
        "fps_total_mean": float(sum(fps_total) / max(1, len(fps_total)))
        if fps_total
        else float("nan"),
    }


def _summarise_run(run_dir: Path, target_frames: int, label: str) -> dict[str, Any]:
    curves = _load_curves(run_dir)
    by_pid = _per_policy_summary(curves, target_frames)
    copy_events = _load_copy_events(run_dir)
    inheritance_reasons = Counter(
        str(e.get("inheritance_reason", "")) for e in copy_events
    )
    recovery_consolidation_sources = Counter(
        str(e.get("recovery_consolidation_source", "")) for e in copy_events
    )
    protected_count = sum(
        1 for e in copy_events if bool(e.get("protected_by_promising_gate"))
    )
    inherited_tol_events = sum(
        1
        for e in copy_events
        if bool(e.get("source_safe_for_curriculum"))
    )
    return {
        "label": label,
        "run_dir": str(run_dir),
        "policies": sorted(by_pid.keys()),
        "n_policies": len(by_pid),
        "rows_loaded": sum(len(v) for v in curves.values()),
        "by_pid": by_pid,
        "peak_mean": _summary(by_pid, "peak_mean"),
        "end_mean": _summary(by_pid, "end_mean"),
        "retention": _summary(by_pid, "retention"),
        "tolerance_at_200M": _summary(by_pid, "tolerance_at_200M"),
        "tolerance_at_400M": _summary(by_pid, "tolerance_at_400M"),
        "tolerance_at_600M": _summary(by_pid, "tolerance_at_600M"),
        "tolerance_at_800M": _summary(by_pid, "tolerance_at_800M"),
        "final_tolerance": _summary(by_pid, "final_tolerance"),
        "min_tolerance": _summary(by_pid, "min_tolerance"),
        "shrink_count": _summary(by_pid, "shrink_count"),
        "rollback_count": _summary(by_pid, "rollback_count"),
        "cross_restart_count": _summary(by_pid, "cross_restart_count"),
        "recovery_restart_count": _summary(by_pid, "recovery_restart_count"),
        "attempts": _summary(by_pid, "attempts"),
        "count_peak_above_30": _count_above(by_pid, "peak_mean", 30.0),
        "count_peak_above_35": _count_above(by_pid, "peak_mean", 35.0),
        "count_end_above_25": _count_above(by_pid, "end_mean", 25.0),
        "count_final_tol_le_004": _count_below_eq(by_pid, "final_tolerance", 0.04),
        "count_final_tol_le_005": _count_below_eq(by_pid, "final_tolerance", 0.05),
        "shrink_factor_histogram": _shrink_factor_histogram(by_pid),
        "engineering": _scrape_engineering_signals(run_dir),
        "fps": _scrape_fps(run_dir),
        "copy_events_total": len(copy_events),
        "inheritance_reason_counts": dict(inheritance_reasons),
        "recovery_consolidation_source_counts": dict(
            recovery_consolidation_sources
        ),
        "copy_events_protected_count": protected_count,
        "copy_events_inherited_tolerance_count": inherited_tol_events,
    }


def _verdict(
    inherit: dict,
    fastcurr: dict,
    reference: dict,
    *,
    target_tol: float,
    safety_end_mean: float,
    high_attempt_threshold: int,
) -> dict[str, Any]:
    notes: list[str] = []

    # Primary curriculum: median <= target OR >= 4/6 policies <= target.
    inh_final = inherit["final_tolerance"]["median"]
    inh_count_le_target = sum(
        1 for v in inherit["by_pid"].values()
        if v.get("final_tolerance") is not None
        and not math.isnan(float(v["final_tolerance"]))
        and float(v["final_tolerance"]) <= target_tol + 1e-9
    )
    primary_curriculum = (
        inh_final is not None and inh_final <= target_tol + 1e-9
    ) or inh_count_le_target >= 4
    notes.append(
        f"primary curriculum: median final tol "
        f"{inh_final if inh_final is not None else 'n/a'} (target<={target_tol}); "
        f"count<=target={inh_count_le_target}/{inherit['n_policies']} -- "
        f"{'PASS' if primary_curriculum else 'FAIL'}"
    )

    # Primary safety.
    inh_end = inherit["end_mean"]["median"]
    collapsed = []
    for pid, info in inherit["by_pid"].items():
        peak = info.get("peak_mean", float("nan"))
        end = info.get("end_mean", float("nan"))
        if (
            peak is not None
            and not math.isnan(peak)
            and peak >= 10.0
            and end is not None
            and not math.isnan(end)
            and (end / max(peak, 1e-9)) < 0.5
        ):
            collapsed.append(pid)
    primary_safety = (
        inh_end is not None and inh_end >= safety_end_mean and not collapsed
    )
    notes.append(
        f"primary safety: end_mean median "
        f"{inh_end if inh_end is not None else 'n/a'} >= {safety_end_mean}; "
        f"collapsed={collapsed} -- {'PASS' if primary_safety else 'FAIL'}"
    )

    # Churn: attempts median lower than fastcurr.
    inh_att = inherit["attempts"]["median"]
    fc_att = fastcurr["attempts"]["median"]
    churn_pass = (
        inh_att is not None
        and fc_att is not None
        and inh_att <= fc_att
    )
    notes.append(
        f"churn: attempts median inherit={inh_att} vs fastcurr={fc_att} "
        f"-- {'PASS' if churn_pass else 'NOT MET'}"
    )

    # Outlier-excluded median tolerance.
    def _median_excl_outliers(rs: dict) -> Optional[float]:
        vals = []
        for v in rs["by_pid"].values():
            att = int(v.get("attempts", 0) or 0)
            tol = v.get("final_tolerance")
            if (
                tol is not None
                and not math.isnan(float(tol))
                and att < high_attempt_threshold
            ):
                vals.append(float(tol))
        return float(statistics.median(vals)) if vals else None

    inh_excl = _median_excl_outliers(inherit)
    fc_excl = _median_excl_outliers(fastcurr)
    outlier_pass = (
        inh_excl is not None
        and fc_excl is not None
        and inh_excl <= fc_excl + 1e-9
    )
    notes.append(
        f"outlier-excluded median final tol: inherit={inh_excl} "
        f"fastcurr={fc_excl} (attempts<{high_attempt_threshold}) -- "
        f"{'PASS' if outlier_pass else 'NOT MET'}"
    )

    if primary_curriculum and primary_safety and churn_pass and outlier_pass:
        overall = "PASS"
    elif primary_safety and (primary_curriculum or churn_pass or outlier_pass):
        overall = "MARGINAL"
    else:
        overall = "FAIL"

    return {
        "overall": overall,
        "primary_curriculum_pass": primary_curriculum,
        "primary_safety_pass": primary_safety,
        "churn_pass": churn_pass,
        "outlier_pass": outlier_pass,
        "notes": notes,
        "collapsed_policies": collapsed,
        "target_tolerance": target_tol,
        "safety_end_mean": safety_end_mean,
        "high_attempt_threshold": high_attempt_threshold,
        "inh_final_tol_median": inh_final,
        "inh_count_le_target": inh_count_le_target,
        "outlier_excluded_inherit": inh_excl,
        "outlier_excluded_fastcurr": fc_excl,
    }


def _row(name: str, val: Any) -> str:
    if val is None:
        return f"| {name} | n/a | n/a | n/a | 0 |"
    if isinstance(val, dict):
        med = val.get("median")
        mn = val.get("min")
        mx = val.get("max")
        n = val.get("n", 0)

        def fmt(x):
            return "n/a" if x is None else f"{x:.4f}"

        return f"| {name} | {fmt(med)} | {fmt(mn)} | {fmt(mx)} | {n} |"
    return f"| {name} | {val} | | | |"


def _emit_markdown(
    inherit: dict,
    fastcurr: dict,
    reference: dict,
    verdict: dict,
    args,
) -> str:
    lines: list[str] = []
    lines.append("# RSSM fastcurr_inherit vs fastcurr vs reference -- 800M comparison\n")
    lines.append(f"target_frames={args.target_frames}")
    lines.append(f"target_tolerance={args.target_tolerance}")
    lines.append(f"safety_end_mean={args.safety_end_mean}")
    lines.append(f"high_attempt_threshold={args.high_attempt_threshold}")
    lines.append("")

    lines.append("## Verdict")
    lines.append(f"- overall: **{verdict['overall']}**")
    lines.append(f"- primary curriculum: {verdict['primary_curriculum_pass']}")
    lines.append(f"- primary safety: {verdict['primary_safety_pass']}")
    lines.append(f"- churn pass: {verdict['churn_pass']}")
    lines.append(f"- outlier pass: {verdict['outlier_pass']}")
    for note in verdict["notes"]:
        lines.append(f"  - {note}")
    if verdict["collapsed_policies"]:
        lines.append(f"  - collapsed policies: {verdict['collapsed_policies']}")
    lines.append("")

    lines.append("## Tolerance trajectory")
    for label, run in (
        ("fastcurr_inherit", inherit),
        ("fastcurr", fastcurr),
        ("reference", reference),
    ):
        lines.append(f"### {label}")
        lines.append("| frame budget | median | min | max | n |")
        lines.append("|---|---|---|---|---|")
        for k in (
            "tolerance_at_200M",
            "tolerance_at_400M",
            "tolerance_at_600M",
            "tolerance_at_800M",
        ):
            lines.append(_row(k, run[k]))
        lines.append(_row("final_tolerance", run["final_tolerance"]))
        lines.append(_row("min_tolerance", run["min_tolerance"]))
        lines.append("")

    lines.append("## Population stats (median / min / max / n_policies)")
    for label, run in (
        ("fastcurr_inherit", inherit),
        ("fastcurr", fastcurr),
        ("reference", reference),
    ):
        lines.append(f"### {label}")
        lines.append("| metric | median | min | max | n |")
        lines.append("|---|---|---|---|---|")
        for k in (
            "peak_mean",
            "end_mean",
            "retention",
            "shrink_count",
            "rollback_count",
            "cross_restart_count",
            "recovery_restart_count",
            "attempts",
        ):
            lines.append(_row(k, run[k]))
        lines.append("")

    lines.append("## Counts")
    lines.append(
        f"- count peak_mean >= 30: inherit={inherit['count_peak_above_30']} "
        f"fastcurr={fastcurr['count_peak_above_30']} "
        f"ref={reference['count_peak_above_30']}"
    )
    lines.append(
        f"- count peak_mean >= 35: inherit={inherit['count_peak_above_35']} "
        f"fastcurr={fastcurr['count_peak_above_35']} "
        f"ref={reference['count_peak_above_35']}"
    )
    lines.append(
        f"- count end_mean >= 25: inherit={inherit['count_end_above_25']} "
        f"fastcurr={fastcurr['count_end_above_25']} "
        f"ref={reference['count_end_above_25']}"
    )
    lines.append(
        f"- count final_tol <= 0.04: inherit={inherit['count_final_tol_le_004']} "
        f"fastcurr={fastcurr['count_final_tol_le_004']} "
        f"ref={reference['count_final_tol_le_004']}"
    )
    lines.append(
        f"- count final_tol <= 0.05: inherit={inherit['count_final_tol_le_005']} "
        f"fastcurr={fastcurr['count_final_tol_le_005']} "
        f"ref={reference['count_final_tol_le_005']}"
    )
    lines.append("")

    lines.append("## Shrink-factor histogram")
    for label, run in (
        ("fastcurr_inherit", inherit),
        ("fastcurr", fastcurr),
        ("reference", reference),
    ):
        lines.append(f"- {label}: {run['shrink_factor_histogram'] or '(none)'}")
    lines.append("")

    lines.append("## Inheritance / promising audit (fastcurr_inherit)")
    lines.append(
        f"- copy events total: {inherit['copy_events_total']}"
    )
    lines.append(
        f"- inheritance_reason histogram: {inherit['inheritance_reason_counts']}"
    )
    lines.append(
        f"- recovery_consolidation_source histogram: "
        f"{inherit['recovery_consolidation_source_counts']}"
    )
    lines.append(
        f"- copy events with source_safe_for_curriculum=True: "
        f"{inherit['copy_events_inherited_tolerance_count']}"
    )
    lines.append(
        f"- copy events flagged protected_by_promising_gate (audit only): "
        f"{inherit['copy_events_protected_count']}"
    )
    lines.append("")

    lines.append("## Engineering / safety")
    for label, run in (
        ("fastcurr_inherit", inherit),
        ("fastcurr", fastcurr),
        ("reference", reference),
    ):
        lines.append(
            f"- {label}: engineering={run['engineering']}, fps={run['fps']}"
        )
    lines.append("")

    lines.append("## Notes for acceptance reading")
    lines.append(
        "- If overall=FAIL but inheritance_reason histogram shows mostly "
        "missing_source_tolerance / source_unstable_legacy, archive metadata "
        "enrichment may not have propagated; rerun with the inherit launcher "
        "after upgrading the existing fastcurr archives or only count "
        "post-fix copy events."
    )
    lines.append(
        "- If collapsed_policies is non-empty, disable "
        "--pbt_protect_promising_workers first and rerun before tuning "
        "inheritance thresholds."
    )
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--fastcurr_inherit", required=True, type=Path,
        help="adaptive_fast + source_safe + promising run",
    )
    p.add_argument(
        "--fastcurr", required=True, type=Path,
        help="adaptive_fast-only baseline run",
    )
    p.add_argument(
        "--reference", required=True, type=Path,
        help="normal-curriculum reference run",
    )
    p.add_argument("--target-frames", type=int, default=800_000_000)
    p.add_argument("--target-tolerance", type=float, default=0.04)
    p.add_argument("--safety-end-mean", type=float, default=25.0)
    p.add_argument("--high-attempt-threshold", type=int, default=5)
    p.add_argument("--out-json", type=Path, default=None)
    args = p.parse_args()

    inh = _summarise_run(args.fastcurr_inherit, args.target_frames, "fastcurr_inherit")
    fc = _summarise_run(args.fastcurr, args.target_frames, "fastcurr")
    ref = _summarise_run(args.reference, args.target_frames, "reference")
    verdict = _verdict(
        inh,
        fc,
        ref,
        target_tol=args.target_tolerance,
        safety_end_mean=args.safety_end_mean,
        high_attempt_threshold=args.high_attempt_threshold,
    )

    md = _emit_markdown(inh, fc, ref, verdict, args)
    print(md)

    out = args.out_json or (
        args.fastcurr_inherit / "rssm_fastcurr_inherit_800m_compare.json"
    )
    payload = {
        "fastcurr_inherit": inh,
        "fastcurr": fc,
        "reference": ref,
        "verdict": verdict,
        "target_frames": args.target_frames,
        "target_tolerance": args.target_tolerance,
        "safety_end_mean": args.safety_end_mean,
        "high_attempt_threshold": args.high_attempt_threshold,
    }
    out.write_text(json.dumps(payload, indent=2, default=str))
    print(f"\nJSON dumped to: {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
