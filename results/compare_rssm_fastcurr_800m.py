#!/usr/bin/env python3
"""Phase A4 P2 (2026-06-08) -- adaptive_fast vs normal curriculum 800M comparison.

Reads two run directories (fastcurr + reference) and emits a curriculum-focused
markdown report:

  * peak / end nsucc_mean (median / min / max)
  * time-to-nsucc_mean threshold (median across policies)
  * tolerance trajectory at 200 / 400 / 600 / 800M env frames
  * final tolerance + #shrinks per policy + shrink-factor histogram
  * collapse-after-shrink count (rollback within consolidation window)
  * rollback / cross-restart / recovery counts
  * OOM (SIGKILL 137) / RuntimeError counts (best-effort, scrapes train.log)
  * FPS

Acceptance verdict (per plan section 8):
  * primary curriculum goal: final tolerance median <= --target-tolerance (0.04 default)
  * primary performance safety: end_mean median >= 15 AND no unmitigated collapse
    (no policy with peak >= 10 AND end / peak < 0.5)
  * stronger-success: final tolerance median <= target AND peak_mean median >= 20
    AND retention median >= 0.7

Inputs:
  --fastcurr RUN_DIR     : fastcurr run dir (must contain
                           dexpbt_8gpu_ws/metrics/training_curve_policy_*.jsonl)
  --reference RUN_DIR    : reference run (same shape).
  --reference-is-strict  : pass when the reference uses curriculum_speed_mode=normal
                           AND the same commit/config as the fastcurr launcher.
  --target-tolerance     : final-tolerance acceptance threshold (default 0.04).
  --target-frames        : env-frame budget (default 800M).

Outputs:
  Markdown report on stdout + JSON dump at
  ``$FASTCURR_RUN_DIR/rssm_fastcurr_800m_compare.json``.
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


def _per_policy_summary(curves: dict[int, list[dict]], target_frames: int) -> dict[int, dict]:
    out: dict[int, dict] = {}
    for pid, rows in curves.items():
        rows = [r for r in rows if int(r.get("env_frames", 0)) <= target_frames]
        if not rows:
            continue
        means = [float(r.get("nsucc_mean", 0) or 0) for r in rows]
        maxes = [float(r.get("nsucc_max", 0) or 0) for r in rows]
        tols = [float(r.get("success_tolerance", float("nan")) or float("nan")) for r in rows]

        # Time to nsucc_mean threshold
        def first_at(thr: float) -> Optional[int]:
            for r in rows:
                if float(r.get("nsucc_mean", 0) or 0) >= thr:
                    return int(r.get("env_frames", 0))
            return None

        # Tolerance at frame budgets (closest preceding sample)
        def tol_at(frame_budget: int) -> Optional[float]:
            best = None
            for r in rows:
                if int(r.get("env_frames", 0)) <= frame_budget:
                    best = float(r.get("success_tolerance", float("nan")) or float("nan"))
            return best

        # Shrink event detection: tolerance strictly decreased between adjacent
        # JSONL rows. Record (env_frames, pre, post, factor).
        shrinks: list[tuple[int, float, float, float]] = []
        prev_tol = None
        for r in rows:
            cur_tol = float(r.get("success_tolerance", float("nan")) or float("nan"))
            if prev_tol is not None and not math.isnan(prev_tol) and not math.isnan(cur_tol):
                if cur_tol + 1e-9 < prev_tol:
                    factor = cur_tol / max(prev_tol, 1e-12)
                    shrinks.append(
                        (int(r.get("env_frames", 0)), float(prev_tol), float(cur_tol), float(factor))
                    )
            prev_tol = cur_tol

        # Collapse-after-shrink: a rollback_count increment within
        # `curriculum_adaptive_consolidation_frames` (or 100M default) of a
        # shrink event. We use rollback_count delta across all rows.
        rollback_increments = 0
        prev_rb = 0
        rollback_after_shrink = 0
        for r in rows:
            rb = int(r.get("rollback_count", 0) or 0)
            if rb > prev_rb:
                rollback_increments += rb - prev_rb
                # Was a shrink seen within the last 100M frames?
                rb_frame = int(r.get("env_frames", 0))
                cons_frames = int(
                    r.get("curriculum_adaptive_consolidation_frames", 100_000_000)
                    or 100_000_000
                )
                for s_frame, _, _, _ in shrinks:
                    if 0 <= rb_frame - s_frame <= cons_frames:
                        rollback_after_shrink += 1
                        break
                prev_rb = rb

        last_row = rows[-1]
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
            "min_tolerance": min(t for t in tols if not math.isnan(t)) if any(not math.isnan(t) for t in tols) else float("nan"),
            "shrinks": shrinks,
            "shrink_count": len(shrinks),
            "rollback_count": int(last_row.get("rollback_count", 0) or 0),
            "rollback_after_shrink": rollback_after_shrink,
            "cross_restart_count": int(last_row.get("cross_restart_count", 0) or 0),
            "self_restart_count": int(last_row.get("self_restart_count", 0) or 0),
            "recovery_restart_count": int(last_row.get("recovery_restart_count", 0) or 0),
            "speed_mode": str(last_row.get("curriculum_speed_mode", "")),
            "adaptive_fast_enabled": bool(last_row.get("curriculum_adaptive_fast_enabled", False)),
        }
    return out


def _summary(by_pid: dict[int, dict], key: str) -> dict[str, Any]:
    vals = [v[key] for v in by_pid.values() if v.get(key) is not None and not (isinstance(v.get(key), float) and math.isnan(v[key]))]
    if not vals:
        return {"median": None, "min": None, "max": None, "n": 0}
    return {
        "median": float(statistics.median(vals)),
        "min": float(min(vals)),
        "max": float(max(vals)),
        "n": len(vals),
    }


def _time_to_threshold(by_pid: dict[int, dict], key: str) -> Optional[int]:
    times = [v[key] for v in by_pid.values() if v.get(key) is not None]
    if not times:
        return None
    return int(statistics.median(times))


def _count_above(by_pid: dict[int, dict], key: str, thr: float) -> int:
    return sum(
        1 for v in by_pid.values()
        if v.get(key) is not None
        and not (isinstance(v[key], float) and math.isnan(v[key]))
        and v[key] >= thr
    )


def _shrink_factor_histogram(by_pid: dict[int, dict]) -> dict[str, int]:
    """Bucket factors by tier label (strong=0.90, mid=0.925, low=0.95, other)."""
    buckets: Counter[str] = Counter()
    for v in by_pid.values():
        for _, _, _, factor in v["shrinks"]:
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
    """Best-effort: count SIGKILL(137) / RuntimeError / fail-fast hits in train.log."""
    sigkill = 0
    runtime_err = 0
    fail_fast = 0
    for fp in sorted(run_dir.glob("policy_*/train.log")):
        try:
            txt = fp.read_text(errors="replace")
        except Exception:
            continue
        sigkill += len(re.findall(r"exited with code 137", txt))
        runtime_err += len(re.findall(r"RuntimeError|Traceback", txt))
        fail_fast += len(re.findall(r"fail-fast|FailFast", txt))
    return {
        "sigkill_137_total": sigkill,
        "runtime_error_total": runtime_err,
        "fail_fast_total": fail_fast,
    }


def _scrape_fps(run_dir: Path) -> dict[str, float]:
    """Mean of last 50 'fps total' / 'fps step' lines per policy, then averaged."""
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
        "fps_step_mean": float(sum(fps_step) / max(1, len(fps_step))) if fps_step else float("nan"),
        "fps_total_mean": float(sum(fps_total) / max(1, len(fps_total))) if fps_total else float("nan"),
    }


def _summarise_run(run_dir: Path, target_frames: int, label: str) -> dict[str, Any]:
    curves = _load_curves(run_dir)
    by_pid = _per_policy_summary(curves, target_frames)
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
        "rollback_after_shrink": _summary(by_pid, "rollback_after_shrink"),
        "cross_restart_count": _summary(by_pid, "cross_restart_count"),
        "recovery_restart_count": _summary(by_pid, "recovery_restart_count"),
        "count_peak_above_30": _count_above(by_pid, "peak_mean", 30.0),
        "count_peak_above_35": _count_above(by_pid, "peak_mean", 35.0),
        "count_end_above_15": _count_above(by_pid, "end_mean", 15.0),
        "time_to_nsucc_10": _time_to_threshold(by_pid, "first_nsucc_10"),
        "time_to_nsucc_20": _time_to_threshold(by_pid, "first_nsucc_20"),
        "time_to_nsucc_30": _time_to_threshold(by_pid, "first_nsucc_30"),
        "shrink_factor_histogram": _shrink_factor_histogram(by_pid),
        "engineering": _scrape_engineering_signals(run_dir),
        "fps": _scrape_fps(run_dir),
    }


def _verdict(fastcurr: dict, reference: dict, target_tol: float) -> dict[str, Any]:
    notes: list[str] = []
    primary_curriculum = False
    primary_safety = False
    stronger = False

    fc_final_tol = fastcurr["final_tolerance"]["median"]
    if fc_final_tol is not None and fc_final_tol <= target_tol + 1e-9:
        primary_curriculum = True
        notes.append(
            f"primary curriculum: final tolerance median {fc_final_tol:.4f} "
            f"<= target {target_tol:.4f} -- PASS"
        )
    elif fc_final_tol is not None:
        notes.append(
            f"primary curriculum: final tolerance median {fc_final_tol:.4f} "
            f"> target {target_tol:.4f} -- FAIL"
        )
    else:
        notes.append("primary curriculum: no fastcurr tolerance data -- FAIL")

    fc_end = fastcurr["end_mean"]["median"]
    # Detect unmitigated collapse: any policy with peak >= 10 AND end/peak < 0.5
    collapsed = []
    for pid, info in fastcurr["by_pid"].items():
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
    if fc_end is not None and fc_end >= 15.0 and not collapsed:
        primary_safety = True
        notes.append(
            f"primary safety: end_mean median {fc_end:.2f} >= 15 AND "
            f"no collapse -- PASS"
        )
    else:
        notes.append(
            f"primary safety: end_mean median "
            f"{fc_end if fc_end is not None else 'n/a'} "
            f"vs threshold 15; collapsed_policies={collapsed} -- "
            f"{'FAIL' if (fc_end is None or fc_end < 15.0 or collapsed) else 'PASS'}"
        )

    fc_peak = fastcurr["peak_mean"]["median"]
    fc_ret = fastcurr["retention"]["median"]
    if (
        primary_curriculum
        and fc_peak is not None
        and fc_peak >= 20.0
        and fc_ret is not None
        and fc_ret >= 0.7
    ):
        stronger = True
        notes.append(
            f"stronger success: peak_mean median {fc_peak:.2f} >= 20 AND "
            f"retention median {fc_ret:.2f} >= 0.7 -- PASS"
        )
    else:
        notes.append(
            f"stronger success: peak_mean={fc_peak} retention={fc_ret} -- "
            f"{'PASS' if stronger else 'NOT MET'}"
        )

    if primary_curriculum and primary_safety:
        overall = "PASS"
    elif primary_curriculum or primary_safety:
        overall = "MARGINAL"
    else:
        overall = "FAIL"

    return {
        "overall": overall,
        "primary_curriculum_pass": primary_curriculum,
        "primary_safety_pass": primary_safety,
        "stronger_success_pass": stronger,
        "notes": notes,
        "collapsed_policies": collapsed,
        "target_tolerance": target_tol,
    }


def _row(name: str, val: dict | int | float | None) -> str:
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


def _emit_markdown(fastcurr: dict, reference: dict, verdict: dict, args) -> str:
    lines: list[str] = []
    lines.append("# RSSM adaptive_fast curriculum vs reference -- 800M comparison\n")
    lines.append(f"target_frames={args.target_frames}")
    lines.append(f"target_tolerance={args.target_tolerance}")
    lines.append(f"reference_is_strict={'true' if args.reference_is_strict else 'false'}")
    if not args.reference_is_strict:
        lines.append("")
        lines.append(
            "> **Caveat**: reference is the legacy fixed RSSM 800M run and "
            "may differ from the fastcurr commit/config (e.g. control-state "
            "schema, P2.1 stability flags). For a strict commit/config-"
            "equivalent comparison run "
            "start_regrasping_fixed_rssm_belief_value_normalcurr_800m.sh."
        )
    lines.append("")

    lines.append("## Verdict")
    lines.append(f"- overall: **{verdict['overall']}**")
    lines.append(f"- primary curriculum (final tolerance <= target): {verdict['primary_curriculum_pass']}")
    lines.append(f"- primary safety (end_mean >= 15 AND no collapse): {verdict['primary_safety_pass']}")
    lines.append(f"- stronger success bonus: {verdict['stronger_success_pass']}")
    for note in verdict["notes"]:
        lines.append(f"  - {note}")
    if verdict["collapsed_policies"]:
        lines.append(f"  - collapsed policies: {verdict['collapsed_policies']}")
    lines.append("")

    lines.append("## Tolerance trajectory (fastcurr)")
    lines.append("| frame budget | median | min | max | n |")
    lines.append("|---|---|---|---|---|")
    for k in ("tolerance_at_200M", "tolerance_at_400M", "tolerance_at_600M", "tolerance_at_800M"):
        lines.append(_row(k, fastcurr[k]))
    lines.append(_row("final_tolerance", fastcurr["final_tolerance"]))
    lines.append(_row("min_tolerance", fastcurr["min_tolerance"]))
    lines.append("")

    lines.append("## Tolerance trajectory (reference)")
    lines.append("| frame budget | median | min | max | n |")
    lines.append("|---|---|---|---|---|")
    for k in ("tolerance_at_200M", "tolerance_at_400M", "tolerance_at_600M", "tolerance_at_800M"):
        lines.append(_row(k, reference[k]))
    lines.append(_row("final_tolerance", reference["final_tolerance"]))
    lines.append(_row("min_tolerance", reference["min_tolerance"]))
    lines.append("")

    lines.append("## Population stats (median / min / max / n_policies)")
    lines.append("")
    lines.append("### fastcurr")
    lines.append("| metric | median | min | max | n |")
    lines.append("|---|---|---|---|---|")
    for k in ("peak_mean", "end_mean", "retention"):
        lines.append(_row(k, fastcurr[k]))
    lines.append("")
    lines.append("### reference")
    lines.append("| metric | median | min | max | n |")
    lines.append("|---|---|---|---|---|")
    for k in ("peak_mean", "end_mean", "retention"):
        lines.append(_row(k, reference[k]))
    lines.append("")

    lines.append("## Time-to-threshold (median across policies that reached it)")
    lines.append("| threshold | fastcurr (env_frames) | reference (env_frames) |")
    lines.append("|---|---|---|")
    for k, label in (
        ("time_to_nsucc_10", "nsucc>=10"),
        ("time_to_nsucc_20", "nsucc>=20"),
        ("time_to_nsucc_30", "nsucc>=30"),
    ):
        lines.append(f"| {label} | {fastcurr[k]} | {reference[k]} |")
    lines.append("")

    lines.append("## Population reach")
    lines.append(f"- count peak_mean >= 30: fastcurr={fastcurr['count_peak_above_30']} ref={reference['count_peak_above_30']}")
    lines.append(f"- count peak_mean >= 35: fastcurr={fastcurr['count_peak_above_35']} ref={reference['count_peak_above_35']}")
    lines.append(f"- count end_mean >= 15: fastcurr={fastcurr['count_end_above_15']} ref={reference['count_end_above_15']}")
    lines.append("")

    lines.append("## Shrink event statistics (fastcurr)")
    lines.append(f"- median shrinks per policy: {fastcurr['shrink_count']['median']}")
    lines.append(
        f"- shrink-factor histogram: {fastcurr['shrink_factor_histogram'] or '(no shrink events)'}"
    )
    lines.append(
        f"- median rollback_after_shrink per policy: {fastcurr['rollback_after_shrink']['median']}"
    )
    lines.append("")
    lines.append("## Shrink event statistics (reference)")
    lines.append(f"- median shrinks per policy: {reference['shrink_count']['median']}")
    lines.append(
        f"- shrink-factor histogram: {reference['shrink_factor_histogram'] or '(no shrink events)'}"
    )
    lines.append(
        f"- median rollback_after_shrink per policy: {reference['rollback_after_shrink']['median']}"
    )
    lines.append("")

    lines.append("## PBT event totals (last row per policy)")
    lines.append(f"- fastcurr: cross={fastcurr['cross_restart_count']}, recovery={fastcurr['recovery_restart_count']}, rollback={fastcurr['rollback_count']}")
    lines.append(f"- reference: cross={reference['cross_restart_count']}, recovery={reference['recovery_restart_count']}, rollback={reference['rollback_count']}")
    lines.append("")

    lines.append("## Engineering / safety")
    lines.append(f"- fastcurr: engineering={fastcurr['engineering']}, fps={fastcurr['fps']}")
    lines.append(f"- reference: engineering={reference['engineering']}, fps={reference['fps']}")
    lines.append("")

    lines.append("## Notes for adaptive_fast acceptance reading")
    lines.append("- Inspect TB tags `curriculum/adaptive_shrink_factor`, "
                 "`curriculum/retention_used_for_shrink`, `curriculum/blocked_by_*`, "
                 "`curriculum/pre_shrink_tolerance`, `curriculum/post_shrink_tolerance` "
                 "for the fastcurr run directly (no aggregation here).")
    lines.append("- Fail conditions (plan section 8): primary curriculum AND primary "
                 "safety both miss -> archive adaptive_fast as a curriculum negative "
                 "ablation; keep normal mode as the main 3B candidate.")
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--fastcurr", required=True, type=Path, help="fastcurr run dir")
    p.add_argument("--reference", required=True, type=Path,
                   help="reference run dir (curriculum_speed_mode=normal)")
    p.add_argument(
        "--reference-is-strict", action="store_true",
        help="pass when reference uses the same commit and same P2.1 config "
        "as the fastcurr launcher",
    )
    p.add_argument("--target-frames", type=int, default=800_000_000)
    p.add_argument("--target-tolerance", type=float, default=0.04)
    p.add_argument("--out-json", type=Path, default=None)
    args = p.parse_args()

    fc = _summarise_run(args.fastcurr, args.target_frames, label="fastcurr")
    ref = _summarise_run(args.reference, args.target_frames, label="reference")
    verdict = _verdict(fc, ref, args.target_tolerance)

    md = _emit_markdown(fc, ref, verdict, args)
    print(md)

    out = args.out_json or (args.fastcurr / "rssm_fastcurr_800m_compare.json")
    payload = {
        "fastcurr": fc,
        "reference": ref,
        "verdict": verdict,
        "target_frames": args.target_frames,
        "target_tolerance": args.target_tolerance,
        "reference_is_strict": bool(args.reference_is_strict),
    }
    out.write_text(json.dumps(payload, indent=2, default=str))
    print(f"\nJSON dumped to: {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
