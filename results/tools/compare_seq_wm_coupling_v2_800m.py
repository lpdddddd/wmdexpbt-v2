#!/usr/bin/env python3
"""Phase A4 P1 -- v2-vs-coupling=none 800M smoke comparison.

Reads two run directories (v2 + reference) and emits a population-level
comparison + acceptance verdict. Reference can be either:

  * a fresh `coupling=none` strict-control run (recommended), OR
  * the existing fixed RSSM 800M smoke (e.g.
    regrasping_fixed_rssm_belief_value_p2_1_smoke_800m_*).

The header line flags `reference_is_strict=true|false` so any caveat
about config drift is visible in every report. When the reference is
not strict, a one-paragraph caveat is emitted at the top.

Inputs:
  --v2 RUN_DIR          : v2 run directory (must contain
                          dexpbt_8gpu_ws/metrics/training_curve_policy_*.jsonl)
  --reference RUN_DIR   : reference run (same shape).
  --reference-is-strict : pass when the reference uses the same commit
                          and config block as the v2 launcher.
  --target-frames       : env-frame budget to evaluate at (default 800M).

Outputs:
  Markdown report on stdout + JSON dump at
  ``$V2_RUN_DIR/seq_wm_v2_800m_compare.json``.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional


def _load_curves(run_dir: Path) -> list[dict[str, Any]]:
    metrics_dir = run_dir / "dexpbt_8gpu_ws" / "metrics"
    rows: list[dict[str, Any]] = []
    for fp in sorted(metrics_dir.glob("training_curve_policy_*.jsonl")):
        for line in fp.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def _per_policy_peaks(curves: list[dict], target_frames: int) -> dict[int, dict]:
    by_pid: dict[int, list[dict]] = defaultdict(list)
    for r in curves:
        pid = int(r.get("policy_idx", -1))
        if pid < 0:
            continue
        if int(r.get("env_frames", 0)) > target_frames:
            continue
        by_pid[pid].append(r)
    out: dict[int, dict] = {}
    for pid, rows in by_pid.items():
        rows.sort(key=lambda r: int(r.get("env_frames", 0)))
        nsucc_means = [float(r.get("nsucc_mean", 0) or 0) for r in rows]
        nsucc_maxes = [float(r.get("nsucc_max", 0) or 0) for r in rows]
        out[pid] = {
            "rows": rows,
            "peak_mean": max(nsucc_means) if nsucc_means else float("nan"),
            "peak_max": max(nsucc_maxes) if nsucc_maxes else float("nan"),
            "end_mean": nsucc_means[-1] if nsucc_means else float("nan"),
            "end_max": nsucc_maxes[-1] if nsucc_maxes else float("nan"),
        }
    return out


def _summary(by_pid: dict[int, dict], key: str) -> dict[str, float]:
    vals = [v[key] for v in by_pid.values() if not math.isnan(v.get(key, float("nan")))]
    if not vals:
        return {"median": float("nan"), "min": float("nan"), "max": float("nan"),
                "n": 0}
    return {
        "median": float(statistics.median(vals)),
        "min": float(min(vals)),
        "max": float(max(vals)),
        "n": len(vals),
    }


def _time_to_threshold(by_pid: dict[int, dict], thr: float) -> Optional[int]:
    """Return median across policies of the first env_frame at which
    nsucc_mean >= thr. Policies that never reach thr contribute None.
    """
    times: list[int] = []
    for pid, info in by_pid.items():
        rows = info["rows"]
        first = None
        for r in rows:
            if float(r.get("nsucc_mean", 0) or 0) >= thr:
                first = int(r.get("env_frames", 0))
                break
        if first is not None:
            times.append(first)
    if not times:
        return None
    return int(statistics.median(times))


def _count_above(by_pid: dict[int, dict], thr: float) -> int:
    return sum(
        1 for v in by_pid.values()
        if not math.isnan(v.get("peak_mean", float("nan")))
        and v["peak_mean"] >= thr
    )


def _retention(by_pid: dict[int, dict]) -> dict[str, float]:
    """end_mean / peak_mean per policy; report median + min."""
    ratios = []
    for v in by_pid.values():
        peak = v.get("peak_mean", float("nan"))
        end = v.get("end_mean", float("nan"))
        if not (math.isnan(peak) or math.isnan(end) or peak <= 1e-9):
            ratios.append(end / peak)
    if not ratios:
        return {"median": float("nan"), "min": float("nan"), "n": 0}
    return {
        "median": float(statistics.median(ratios)),
        "min": float(min(ratios)),
        "n": len(ratios),
    }


def _pbt_event_counts(curves: list[dict]) -> dict[str, int]:
    return {
        "self_restart_total": sum(int(r.get("self_restart_count", 0) or 0)
                                  for r in curves[-1:] if r),
        "cross_restart_total": sum(int(r.get("cross_restart_count", 0) or 0)
                                   for r in curves[-1:] if r),
        "rollback_total": sum(int(r.get("rollback_count", 0) or 0)
                              for r in curves[-1:] if r),
        "recovery_total": sum(int(r.get("recovery_restart_count", 0) or 0)
                              for r in curves[-1:] if r),
    }


def _verdict(v2: dict, ref: dict) -> dict[str, Any]:
    primary_pass = False
    primary_notes = []

    v2_peak = v2["peak_mean"]["median"]
    ref_peak = ref["peak_mean"]["median"]
    if not math.isnan(v2_peak) and not math.isnan(ref_peak):
        ratio_peak = v2_peak / max(ref_peak, 1e-9)
        primary_notes.append(
            f"peak_mean median ratio v2/ref = {ratio_peak:.3f} (need >=1.05)"
        )
        if ratio_peak >= 1.05:
            primary_pass = True

    v2_end = v2["end_mean"]["median"]
    ref_end = ref["end_mean"]["median"]
    if not math.isnan(v2_end) and not math.isnan(ref_end):
        ratio_end = v2_end / max(ref_end, 1e-9)
        primary_notes.append(
            f"end_mean median ratio v2/ref = {ratio_end:.3f} (need >=1.05)"
        )
        if ratio_end >= 1.05:
            primary_pass = True

    secondary_pass = False
    secondary_notes = []
    v2_t20 = v2["time_to_nsucc_20"]
    ref_t20 = ref["time_to_nsucc_20"]
    if v2_t20 is not None and ref_t20 is not None and ref_t20 > 0:
        ratio_t20 = v2_t20 / ref_t20
        secondary_notes.append(
            f"time-to-nsucc20 ratio v2/ref = {ratio_t20:.3f} (need <=0.85)"
        )
        if ratio_t20 <= 0.85:
            secondary_pass = True
    v2_above35 = v2["count_peak_above_35"]
    ref_above35 = ref["count_peak_above_35"]
    secondary_notes.append(
        f"policies peak>=35: v2={v2_above35} ref={ref_above35} (need v2>ref)"
    )
    if v2_above35 > ref_above35:
        secondary_pass = True

    overall = "PASS" if (primary_pass and secondary_pass) else (
        "MARGINAL" if primary_pass or secondary_pass else "FAIL"
    )
    return {
        "primary_pass": primary_pass,
        "secondary_pass": secondary_pass,
        "overall": overall,
        "primary_notes": primary_notes,
        "secondary_notes": secondary_notes,
    }


def _summarise_run(run_dir: Path, target_frames: int, label: str) -> dict[str, Any]:
    curves = _load_curves(run_dir)
    by_pid = _per_policy_peaks(curves, target_frames)
    summary = {
        "label": label,
        "run_dir": str(run_dir),
        "policies": sorted(by_pid.keys()),
        "n_policies": len(by_pid),
        "rows_loaded": len(curves),
        "peak_mean": _summary(by_pid, "peak_mean"),
        "peak_max": _summary(by_pid, "peak_max"),
        "end_mean": _summary(by_pid, "end_mean"),
        "end_max": _summary(by_pid, "end_max"),
        "retention": _retention(by_pid),
        "time_to_nsucc_10": _time_to_threshold(by_pid, 10.0),
        "time_to_nsucc_20": _time_to_threshold(by_pid, 20.0),
        "time_to_nsucc_30": _time_to_threshold(by_pid, 30.0),
        "count_peak_above_30": _count_above(by_pid, 30.0),
        "count_peak_above_35": _count_above(by_pid, 35.0),
        "pbt_events": _pbt_event_counts(curves),
    }
    return summary


def _emit_markdown(v2: dict, ref: dict, verdict: dict, args) -> str:
    lines: list[str] = []
    lines.append("# Sequence WM v2 vs coupling=none -- 800M comparison\n")
    lines.append(f"target_frames={args.target_frames}")
    lines.append(f"reference_is_strict={'true' if args.reference_is_strict else 'false'}")
    if not args.reference_is_strict:
        lines.append("")
        lines.append("> **Caveat**: reference is the legacy fixed RSSM 800M run "
                     "and may differ from the v2 commit/config (e.g. RSSM "
                     "hyperparameters, curriculum block, max_consecutive_successes, "
                     "PBT thresholds). Treat the comparison as indicative; for a "
                     "strict commit/config-equivalent comparison run "
                     "start_regrasping_fixed_rssm_belief_value_no_policy_coupling_800m.sh.")
    lines.append("")
    lines.append("## Verdict")
    lines.append(f"- overall: **{verdict['overall']}**")
    lines.append(f"- primary acceptance pass: {verdict['primary_pass']}")
    for n in verdict["primary_notes"]:
        lines.append(f"  - {n}")
    lines.append(f"- secondary acceptance pass: {verdict['secondary_pass']}")
    for n in verdict["secondary_notes"]:
        lines.append(f"  - {n}")
    lines.append("")

    def _row(name, val):
        if isinstance(val, dict):
            return (f"| {name} | "
                    f"{val.get('median', float('nan')):.3f} | "
                    f"{val.get('min', float('nan')):.3f} | "
                    f"{val.get('max', float('nan')):.3f} | "
                    f"{val.get('n', 0)} |")
        return f"| {name} | {val} | | | |"

    lines.append("## Population stats (median / min / max / n_policies)")
    lines.append("")
    lines.append("### v2")
    lines.append("| metric | median | min | max | n |")
    lines.append("|---|---|---|---|---|")
    for k in ["peak_mean", "peak_max", "end_mean", "end_max", "retention"]:
        lines.append(_row(k, v2[k]))
    lines.append("")
    lines.append("### reference")
    lines.append("| metric | median | min | max | n |")
    lines.append("|---|---|---|---|---|")
    for k in ["peak_mean", "peak_max", "end_mean", "end_max", "retention"]:
        lines.append(_row(k, ref[k]))
    lines.append("")
    lines.append("## Time-to-threshold (median across policies that reached threshold)")
    lines.append("| threshold | v2 (env_frames) | reference (env_frames) |")
    lines.append("|---|---|---|")
    for k, label in [("time_to_nsucc_10", "nsucc>=10"),
                     ("time_to_nsucc_20", "nsucc>=20"),
                     ("time_to_nsucc_30", "nsucc>=30")]:
        lines.append(f"| {label} | {v2[k]} | {ref[k]} |")
    lines.append("")
    lines.append("## Population reach")
    lines.append(f"- count peak_mean >= 30: v2={v2['count_peak_above_30']} ref={ref['count_peak_above_30']}")
    lines.append(f"- count peak_mean >= 35: v2={v2['count_peak_above_35']} ref={ref['count_peak_above_35']}")
    lines.append("")
    lines.append("## PBT event totals (last row per policy)")
    lines.append(f"- v2: {v2['pbt_events']}")
    lines.append(f"- ref: {ref['pbt_events']}")
    lines.append("")
    lines.append("## Notes for v2 acceptance reading")
    lines.append("- Inspect TB tags `seq_policy_aux/teacher_return_corr`, "
                 "`seq_policy_aux/gate_pass`, `seq_policy_aux/effective_weight`, "
                 "`seq_policy_aux/aux_critic_subtower_grad_norm` for the v2 run "
                 "directly (no aggregation here).")
    lines.append("- v2 fail conditions (plan section 11): primary AND secondary "
                 "miss -> archive v2 as a second negative ablation alongside "
                 "legacy distill_value; do NOT start 3B.")
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--v2", required=True, type=Path, help="v2 run dir")
    p.add_argument("--reference", required=True, type=Path,
                   help="reference (coupling=none) run dir")
    p.add_argument(
        "--reference-is-strict", action="store_true",
        help="pass when reference uses the same commit/config as v2 launcher",
    )
    p.add_argument("--target-frames", type=int, default=800_000_000)
    p.add_argument("--out-json", type=Path, default=None)
    args = p.parse_args()

    v2 = _summarise_run(args.v2, args.target_frames, label="v2")
    ref = _summarise_run(args.reference, args.target_frames, label="reference")
    verdict = _verdict(v2, ref)

    md = _emit_markdown(v2, ref, verdict, args)
    print(md)

    out = args.out_json or (args.v2 / "seq_wm_v2_800m_compare.json")
    payload = {
        "v2": v2, "reference": ref, "verdict": verdict,
        "target_frames": args.target_frames,
        "reference_is_strict": bool(args.reference_is_strict),
    }
    out.write_text(json.dumps(payload, indent=2, default=str))
    print(f"\nJSON dumped to: {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
