#!/usr/bin/env python3
"""Generate an 800M-frame P1 smoke acceptance report.

Extends generate_smoke_report_cbwm.py with P1-specific criteria:
  A1-A3: nsucc_mean peak / no collapse
  B1-B2: mastery shrink / collapse freeze events
  C1-C2: stable-source tier usage / CF healthy gate

Usage:
  python generate_smoke_report_p1.py /path/to/regrasping_cbwm_smoke_800m_p1_YYYYMMDD_HHMMSS
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

# Reuse base report helpers from CB-WM smoke report.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from generate_smoke_report_cbwm import (  # noqa: E402
    FRAMES_PER_EPOCH,
    _detect_pre_post_degradation,
    _eligible_source_stats,
    _grep_logs,
    _load_cf_results,
    _load_manifests,
    _load_training_curves,
    _restart_counts,
    generate_report as generate_base_report,
)


def _load_curriculum_control(ws: Path) -> list[dict[str, Any]]:
    ctrl_dir = ws / "curriculum_control"
    rows: list[dict[str, Any]] = []
    if not ctrl_dir.is_dir():
        return rows
    for fp in sorted(ctrl_dir.glob("*.json")):
        try:
            data = json.loads(fp.read_text())
            data["_path"] = str(fp)
            rows.append(data)
        except json.JSONDecodeError:
            continue
    return rows


def _nsucc_at_frame(curves: list[dict], target_m: float, window_m: float = 40.0) -> dict[str, Any]:
    """Best policy nsucc_mean near target frame (default 480M ±40M)."""
    target = target_m * 1_000_000
    window = window_m * 1_000_000
    best: dict[int, float] = {}
    for pt in curves:
        ef = int(pt.get("env_frames", 0))
        if abs(ef - target) > window:
            continue
        pid = int(pt.get("policy_idx", -1))
        nm = float(pt.get("nsucc_mean", 0) or 0)
        best[pid] = max(best.get(pid, 0.0), nm)
    if not best:
        return {"found": False, "best_policy": -1, "peak_nsucc_mean": float("nan")}
    pid, peak = max(best.items(), key=lambda x: x[1])
    return {"found": True, "best_policy": pid, "peak_nsucc_mean": peak, "by_policy": best}


def _nsucc_at_end(curves: list[dict]) -> dict[str, Any]:
    last_by_policy: dict[int, dict] = {}
    for pt in curves:
        pid = int(pt.get("policy_idx", -1))
        ef = int(pt.get("env_frames", 0))
        prev = last_by_policy.get(pid)
        if prev is None or ef >= int(prev.get("env_frames", 0)):
            last_by_policy[pid] = pt
    if not last_by_policy:
        return {"best_policy": -1, "nsucc_mean": float("nan")}
    pid, pt = max(
        last_by_policy.items(),
        key=lambda x: float(x[1].get("nsucc_mean", 0) or 0),
    )
    return {
        "best_policy": pid,
        "nsucc_mean": float(pt.get("nsucc_mean", 0) or 0),
        "env_frames": int(pt.get("env_frames", 0)),
    }


def _p1_tier_stats(curves: list[dict]) -> dict[str, int]:
    """Aggregate PBT source-tier usage across the population.

    Primary path reads the cumulative ``source_tier_<label>_count`` fields
    written per tick. When those fields are absent or all zero (older runs
    or truncated curves), fall back to counting the per-tick
    ``last_source_tier_label`` so this matches the D-section's tier
    distribution computed in ``generate_smoke_report_p2.py``.
    """

    keys = (
        "strict_stable_live",
        "usable_live",
        "relaxed_live",
        "best_stable_archive",
        "archive",
        "skipped",
    )
    totals: dict[str, int] = {k: 0 for k in keys}
    for pt in curves:
        for key in keys:
            try:
                totals[key] += int(pt.get(f"source_tier_{key}_count", 0) or 0)
            except (TypeError, ValueError):
                continue

    if sum(totals.values()) == 0:
        # Fallback: count last_source_tier_label per tick.
        for pt in curves:
            label = str(pt.get("last_source_tier_label") or "").strip()
            if not label or label.lower() == "none":
                continue
            if label not in totals:
                totals[label] = 0
            totals[label] += 1
    return dict(totals)


def _p1_curriculum_events(curves: list[dict], ctrl: list[dict], run_dir: Path) -> dict[str, Any]:
    mastery_shrink_logs = _grep_logs(run_dir, r"mastery.*shrank|curriculum_shrink|last_mastered_tolerance")
    collapse_logs = _grep_logs(run_dir, r"collapse_detected|collapse_freeze_active")
    any_allow_shrink = any(c.get("allow_shrink") for c in ctrl)
    any_collapse_freeze = any(c.get("collapse_freeze_active") for c in ctrl)
    mastery_consec_max = max((int(c.get("mastery_consecutive_count", 0)) for c in ctrl), default=0)
    return {
        "mastery_shrink_log_lines": len(mastery_shrink_logs),
        "collapse_log_lines": len(collapse_logs),
        "control_allow_shrink_seen": any_allow_shrink,
        "control_collapse_freeze_seen": any_collapse_freeze,
        "max_mastery_consecutive_count": mastery_consec_max,
    }


def _evaluate_p1_criteria(
    curves: list[dict],
    ctrl: list[dict],
    run_dir: Path,
    tier_stats: dict[str, int],
    cf_results: list[dict],
) -> list[dict[str, Any]]:
    peak_480 = _nsucc_at_frame(curves, target_m=480.0)
    end = _nsucc_at_end(curves)
    events = _p1_curriculum_events(curves, ctrl, run_dir)
    degradation = _detect_pre_post_degradation(curves)

    peak_val = float(peak_480.get("peak_nsucc_mean", float("nan")))
    end_val = float(end.get("nsucc_mean", float("nan")))

    a1 = peak_480.get("found") and peak_val >= 3.5
    a2 = end_val >= 2.0 if end_val == end_val else False
    a3 = (not degradation["has_pre_post_degradation"]) or (
        end_val >= 0.5 * peak_val if peak_val == peak_val and peak_val > 0 else False
    )

    b1 = (
        events["mastery_shrink_log_lines"] > 0
        or events["control_allow_shrink_seen"]
        or events["max_mastery_consecutive_count"] >= 2
    )
    b2 = events["collapse_log_lines"] > 0 or events["control_collapse_freeze_seen"]

    stable_tier_used = (
        tier_stats.get("strict_stable_live", 0)
        + tier_stats.get("usable_live", 0)
        + tier_stats.get("best_stable_archive", 0)
    ) > 0
    c1 = stable_tier_used
    cf_skip_stable = len(_grep_logs(run_dir, r"skip trigger: no policy passed health gates|no stable source"))
    c2 = len(cf_results) == 0 or cf_skip_stable > 0 or stable_tier_used

    return [
        {"id": "A1", "name": "Peak nsucc_mean @480M±40M ≥ 3.5", "pass": bool(a1),
         "detail": f"peak={peak_val:.3f} policy={peak_480.get('best_policy')}"},
        {"id": "A2", "name": "Best policy nsucc_mean @end ≥ 2.0", "pass": bool(a2),
         "detail": f"end={end_val:.3f} policy={end.get('best_policy')}"},
        {"id": "A3", "name": "No monotonic collapse (end ≥ 0.5×peak)", "pass": bool(a3),
         "detail": f"degradation={degradation['has_pre_post_degradation']}"},
        {"id": "B1", "name": "Mastery shrink / gate activity", "pass": bool(b1),
         "detail": str(events)},
        {"id": "B2", "name": "Collapse freeze detected (if collapse)", "pass": bool(b2),
         "detail": f"collapse_logs={events['collapse_log_lines']}"},
        {"id": "C1", "name": "Stable-source tier used", "pass": bool(c1),
         "detail": str(tier_stats)},
        {"id": "C2", "name": "CF gated on stable source or skipped", "pass": bool(c2),
         "detail": f"cf_blocks={len(cf_results)} cf_skip_logs={cf_skip_stable}"},
    ]


def generate_p1_report(
    run_dir: Path,
    frame_range: tuple[int, int] | None = None,
) -> str:
    ws = run_dir / "dexpbt_8gpu_ws"
    curves = _load_training_curves(ws, frame_range=frame_range)
    ctrl = _load_curriculum_control(ws)
    cf_results = _load_cf_results(ws)
    tier_stats = _p1_tier_stats(curves)
    criteria = _evaluate_p1_criteria(curves, ctrl, run_dir, tier_stats, cf_results)

    base = generate_base_report(run_dir, frame_range=frame_range)
    now = datetime.now().strftime("%Y-%m-%d %H:%M CST")
    passed = sum(1 for c in criteria if c["pass"])
    total = len(criteria)

    lines = [
        f"# P1 Smoke Acceptance Report (800M)",
        "",
        f"**Run ID:** `{run_dir.name}`",
        f"**Generated:** {now}",
        f"**Overall:** {passed}/{total} criteria passed",
        "",
        "## P1 acceptance criteria",
        "",
        "| ID | Criterion | Pass | Detail |",
        "|----|-----------|------|--------|",
    ]
    for c in criteria:
        mark = "PASS" if c["pass"] else "FAIL"
        lines.append(f"| {c['id']} | {c['name']} | **{mark}** | `{c['detail'][:80]}` |")

    lines += [
        "",
        "## P1 tier breakdown",
        "",
    ]
    for k, v in sorted(tier_stats.items()):
        lines.append(f"- `{k}`: {v}")

    lines += [
        "",
        "## Curriculum control snapshots",
        "",
        f"Control files found: **{len(ctrl)}**",
        "",
    ]
    for c in ctrl[:8]:
        lines.append(
            f"- `{Path(c['_path']).name}`: allow_shrink={c.get('allow_shrink')} "
            f"mastery_consec={c.get('mastery_consecutive_count')} "
            f"collapse_freeze={c.get('collapse_freeze_active')}"
        )

    lines += [
        "",
        "---",
        "",
        "## Base CB-WM smoke report (Patches A-J)",
        "",
        base,
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate P1 800M smoke report")
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("-o", "--output", type=Path, default=None)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    if not run_dir.is_dir():
        raise SystemExit(f"Run dir not found: {run_dir}")
    report = generate_p1_report(run_dir)
    out = args.output or (run_dir / "SMOKE_P1_REPORT.md")
    out.write_text(report)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
