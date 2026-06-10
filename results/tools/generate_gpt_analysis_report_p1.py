#!/usr/bin/env python3
"""Generate GPT_ANALYSIS_DATA_REPORT for P1 formal run."""

from __future__ import annotations

import json
import os
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any


RESULTS_ROOT = Path("/root/autodl-tmp/wmdexpbt/results")
RUN_DIR = Path(
    os.environ.get(
        "P1_RUN_DIR",
        str(RESULTS_ROOT / "regrasping_cbwm_p1_20260601_083719"),
    )
)
OLD_RUN_DIR = Path(
    os.environ.get(
        "P1_OLD_RUN_DIR",
        str(RESULTS_ROOT / "regrasping_cbwm_20260601_002817"),
    )
)
SWEEP_DIR = OLD_RUN_DIR / "tolerance_sweep"
WS = RUN_DIR / "dexpbt_8gpu_ws"
METRICS = WS / "metrics"
CURR_CTRL = WS / "curriculum_control"
FRAMES_PER_EPOCH = 8192 * 16


def load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    if not path.is_file():
        return rows
    for line in path.read_text(errors="replace").splitlines():
        line = line.strip()
        if line:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def grep_logs(pattern: str) -> list[str]:
    hits: list[str] = []
    rx = re.compile(pattern)
    for log in sorted(RUN_DIR.rglob("*.log")):
        try:
            for ln in log.read_text(errors="replace").splitlines():
                if rx.search(ln):
                    hits.append(ln)
        except OSError:
            continue
    return hits


def count_attempts(policy_idx: int) -> int:
    f = RUN_DIR / f"policy_{policy_idx:02d}/train.log"
    if not f.is_file():
        return 0
    return f.read_text(errors="replace").count("experiment_name attempt_")


def live_frames(policy_idx: int) -> int:
    f = RUN_DIR / f"policy_{policy_idx:02d}/train.log"
    if not f.is_file():
        return 0
    best = 0
    for m in re.finditer(r"frames: (\d+)", f.read_text(errors="replace")):
        best = max(best, int(m.group(1)))
    return best


def summarize_policy(pid: int, curves: list[dict]) -> dict[str, Any]:
    pts = [p for p in curves if int(p.get("policy_idx", -1)) == pid]
    if not pts:
        pts = load_jsonl(METRICS / f"training_curve_policy_{pid:02d}.jsonl")
    if not pts:
        return {"policy_idx": pid}

    # Dedupe by env_frames keeping last
    by_frame: dict[int, dict] = {}
    for p in pts:
        ef = int(p.get("env_frames", 0))
        by_frame[ef] = p
    pts = sorted(by_frame.values(), key=lambda x: int(x.get("env_frames", 0)))

    means = [float(p.get("nsucc_mean", 0) or 0) for p in pts]
    peak_pt = max(pts, key=lambda x: float(x.get("nsucc_mean", 0) or 0))
    last = pts[-1]
    cur = json.loads((CURR_CTRL / f"{pid:03d}.json").read_text()) if (CURR_CTRL / f"{pid:03d}.json").is_file() else "{}"

    tol = last.get("success_tolerance")
    if tol is None:
        tol = (last.get("curriculum") or {}).get("success_tolerance")

    return {
        "policy_idx": pid,
        "env_frames_M": live_frames(pid) / 1e6,
        "last_tick_M": int(last.get("env_frames", 0)) / 1e6,
        "pbt_ticks": len(pts),
        "peak_nsucc_mean": float(peak_pt.get("nsucc_mean", 0) or 0),
        "peak_at_M": int(peak_pt.get("env_frames", 0)) / 1e6,
        "peak_at_480M_window": round(
            max(
                (float(p.get("nsucc_mean", 0) or 0) for p in pts if 440 <= int(p.get("env_frames", 0)) / 1e6 <= 520),
                default=0.0,
            ),
            3,
        ),
        "peak_nsucc_max": float(peak_pt.get("nsucc_max", 0) or 0),
        "last_nsucc_mean": float(last.get("nsucc_mean", 0) or 0),
        "last_nsucc_max": float(last.get("nsucc_max", 0) or 0),
        "last_tolerance": tol,
        "cross_restart_count": int(last.get("cross_restart_count", 0)),
        "recovery_restart_count": int(last.get("recovery_restart_count", 0)),
        "attempts": count_attempts(pid),
        "curriculum_control": cur,
        "tick_series": [
            {
                "env_M": round(int(p.get("env_frames", 0)) / 1e6, 1),
                "nsucc_mean": round(float(p.get("nsucc_mean", 0) or 0), 3),
                "nsucc_max": round(float(p.get("nsucc_max", 0) or 0), 1),
                "tolerance": p.get("success_tolerance") or (p.get("curriculum") or {}).get("success_tolerance"),
            }
            for p in pts
        ],
    }


def count_source_tiers() -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for ln in grep_logs(r"source_selection tier="):
        m = re.search(r"tier=(\S+)", ln)
        if m:
            counts[m.group(1)] += 1
    return dict(counts)


def build_report() -> dict[str, Any]:
    all_curves: list[dict] = []
    for pid in range(8):
        all_curves.extend(load_jsonl(METRICS / f"training_curve_policy_{pid:02d}.jsonl"))

    policies = [summarize_policy(pid, all_curves) for pid in range(8)]

    # p0 @480M: pick best mean in [440,520]M window (avoid post-restart frame resets)
    p0_pts = policies[0].get("tick_series", [])
    window = [p for p in p0_pts if 440 <= p.get("env_M", 0) <= 520 and p.get("nsucc_mean", 0) > 0.5]
    p0_480 = max(window, key=lambda x: x.get("nsucc_mean", 0)) if window else None
    if p0_480 is None:
        p0_480 = next((p for p in p0_pts if abs(p.get("env_M", 0) - 480) < 40), None)

    tier_counts = count_source_tiers()
    collapse_logs = len(grep_logs(r"collapse_freeze|collapse_detected"))
    cf_resets = len(grep_logs(r"\[CF\] reset agent counters"))

    sweep_p0 = {}
    sweep_p6 = {}
    if (SWEEP_DIR / "p0_480M_sweep.json").is_file():
        sweep_p0 = json.loads((SWEEP_DIR / "p0_480M_sweep.json").read_text())
    if (SWEEP_DIR / "p6_280M_sweep.json").is_file():
        sweep_p6 = json.loads((SWEEP_DIR / "p6_280M_sweep.json").read_text())

    return {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "run_id": RUN_DIR.name,
        "run_dir": str(RUN_DIR),
        "comparison_run_id": OLD_RUN_DIR.name,
        "stopped": (RUN_DIR / "STOPPED.txt").read_text().strip() if (RUN_DIR / "STOPPED.txt").is_file() else "",
        "config": {
            "patch": "P1_curriculum_stability",
            "num_policies": 6,
            "num_envs": 8192,
            "max_iterations": 38147,
            "target_env_frames_B": 5.0,
            "pbt_start_after_M": 200,
            "pbt_interval_steps_M": 20,
            "curriculum_mode": "mastery",
            "mastery_nsucc_mean": 4.0,
            "mastery_nsucc_max": 20.0,
            "mastery_consecutive_windows": 2,
            "curriculum_shrink_factor": 0.95,
            "collapse_drop_ratio": 0.5,
            "collapse_patience": 2,
            "collapse_freeze_env_frames_M": 200,
            "pbt_source_require_stable_mean": True,
            "pbt_allow_best_nsucc_archive_recovery": False,
            "cf_source_min_nsucc_mean": 3.0,
            "cf_source_min_nsucc_max": 20.0,
        },
        "pbt_operational_stats": {
            "source_tier_log_counts": tier_counts,
            "collapse_related_log_lines": collapse_logs,
            "cf_counter_reset_log_lines": cf_resets,
            "total_training_curve_points": len(all_curves),
        },
        "comparison_at_480M": {
            "old_formal_run_p0_peak_mean": 4.27,
            "p1_run_p0_at_480M": p0_480,
            "p1_run_p0_peak_in_440_520M_window": policies[0].get("peak_at_480M_window"),
        },
        "tolerance_sweep_task1": {
            "p0_480M": sweep_p0.get("analysis"),
            "p6_280M": sweep_p6.get("analysis"),
        },
        "policies": policies,
        "diagnosis": {
            "outcome": "population_collapse",
            "peak_better_than_old_at_480M": bool(p0_480 and p0_480.get("nsucc_mean", 0) > 4.27),
            "final_all_policies_mean_near_zero": all(
                p.get("last_nsucc_mean", 1) < 0.5 for p in policies if p.get("tick_series")
            ),
            "mastery_gate_never_sustained": all(
                (p.get("curriculum_control") or {}).get("mastery_consecutive_count", 0) < 2
                for p in policies
            ),
            "collapse_freeze_activated": sum(
                1 for p in policies if (p.get("curriculum_control") or {}).get("collapse_freeze_active")
            ),
            "hypotheses": [
                "allow_shrink=True persists entire 20M PBT window after brief mastery pass → many env-side shrinks per window",
                "PBT restart inherits collapsed source tolerance (~0.02) propagating tight tol to bottom workers",
                "collapse_freeze triggers after mean already near zero (too late)",
                "mastery gate requires max>=20 (sparse env spike) + mean>=4 for 2 consecutive PBT windows — hard to sustain while tolerance tightens",
            ],
        },
    }


def render_md(data: dict[str, Any]) -> str:
    lines = [
        "# DexPBT CB-WM P1 — Data Report for External Analysis",
        "",
        f"**Generated:** {data['generated_at']}",
        f"**Run ID:** `{data['run_id']}`",
        f"**Run dir:** `{data['run_dir']}`",
        f"**Comparison (P0 formal, no P1):** `{data['comparison_run_id']}`",
        "",
        f"**Stop note:** {data.get('stopped', 'N/A')}",
        "",
        "## 1. Experiment Context",
        "",
        "Formal 6-GPU run with **P1 curriculum/stability patch** (mastery gate, collapse freeze, stable-source cascade, best_stable archive, no best_nsucc recovery).",
        "Stopped manually after population-wide nsucc_mean collapse.",
        "",
        "### P1 config (vs legacy)",
        "",
        "| Parameter | P1 value | Legacy (old formal) |",
        "| --- | --- | --- |",
        "| curriculum_mode | mastery | legacy (mean>3 shrink) |",
        "| curriculum_shrink_factor | 0.95 | 0.9 |",
        "| mastery_nsucc_mean | 4.0 | N/A |",
        "| mastery_consecutive_windows | 2 | N/A |",
        "| collapse_freeze | 200M env frames | N/A |",
        "| pbt_source_require_stable_mean | True | False |",
        "| best_nsucc recovery | **disabled** | enabled |",
        "| cf_source_min_nsucc_mean | 3.0 | 1.5 |",
        "",
        "## 2. PBT / P1 Operational Stats",
        "",
    ]
    stats = data["pbt_operational_stats"]
    lines.append("| Metric | Value |")
    lines.append("| --- | --- |")
    for k, v in stats.items():
        lines.append(f"| {k} | {v} |")
    lines.append("")
    lines.append("**Source tier selection (log line counts):**")
    for k, v in sorted((stats.get("source_tier_log_counts") or {}).items(), key=lambda x: -x[1]):
        lines.append(f"- `{k}`: {v}")
    lines.append("")

    lines += [
        "## 3. Per-Policy Summary",
        "",
        "| p | live_M | peak_mean | peak@M | last_mean | last_tol | xrst | rec | attempts | collapse_freeze | mastery_consec |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for p in data["policies"]:
        cc = p.get("curriculum_control") or {}
        lines.append(
            f"| {p['policy_idx']} | {p.get('env_frames_M', 0):.0f} | "
            f"{p.get('peak_nsucc_mean', 0):.2f} | {p.get('peak_at_M', 0):.0f} | "
            f"{p.get('last_nsucc_mean', 0):.2f} | {p.get('last_tolerance', '?')} | "
            f"{p.get('cross_restart_count', 0)} | {p.get('recovery_restart_count', 0)} | "
            f"{p.get('attempts', 0)} | {cc.get('collapse_freeze_active')} | "
            f"{cc.get('mastery_consecutive_count', 0)} |"
        )
    lines.append("")

    cmp480 = data.get("comparison_at_480M", {})
    lines += [
        "## 4. 480M Comparison (key validation point)",
        "",
        f"| Metric | Old formal (no P1) | P1 run p0 |",
        f"| --- | --- | --- |",
        f"| peak/mean @480M (440-520M window) | {cmp480.get('old_formal_run_p0_peak_mean', '?')} | "
        f"{cmp480.get('p1_run_p0_peak_in_440_520M_window', 'N/A')} |",
        "",
    ]

    lines += ["## 5. Policy 0 — Full PBT Tick Series", ""]
    lines.append("| env_M | nsucc_mean | nsucc_max | tolerance |")
    lines.append("| --- | --- | --- | --- |")
    for t in data["policies"][0].get("tick_series", []):
        lines.append(
            f"| {t['env_M']} | {t['nsucc_mean']} | {t['nsucc_max']} | {t.get('tolerance', '?')} |"
        )
    lines.append("")

    lines += ["## 6. Task 1 — Fixed Tolerance Eval (pre-P1 run checkpoints)", ""]
    sweep = data.get("tolerance_sweep_task1", {})
    if sweep.get("p0_480M"):
        lines.append("**p0@480M (old formal ckpt):**")
        lines.append(f"- best_tolerance: {sweep['p0_480M'].get('best_tolerance')}")
        lines.append(f"- best_nsucc_mean (fair eval): {sweep['p0_480M'].get('best_nsucc_mean')}")
        lines.append("")
    if sweep.get("p6_280M"):
        lines.append("**p6@280M:** fair-eval nsucc_mean=0 at all tolerances (training mean was 5.46)")
        lines.append("")

    diag = data.get("diagnosis", {})
    lines += [
        "## 7. Diagnosis Summary",
        "",
        f"**Outcome:** {diag.get('outcome')}",
        "",
        "| Check | Result |",
        "| --- | --- |",
        f"| Peak @480M better than old? | {diag.get('peak_better_than_old_at_480M')} |",
        f"| Final population mean≈0? | {diag.get('final_all_policies_mean_near_zero')} |",
        f"| Mastery consecutive≥2 sustained? | {not diag.get('mastery_gate_never_sustained')} |",
        f"| Policies with collapse_freeze active | {diag.get('collapse_freeze_activated')} / 8 |",
        "",
        "### Hypotheses for next patch",
        "",
    ]
    for i, h in enumerate(diag.get("hypotheses", []), 1):
        lines.append(f"{i}. {h}")
    lines.append("")

    lines += [
        "## 8. Artifact Paths",
        "",
        f"- JSON: `{data['run_dir']}/GPT_ANALYSIS_DATA_REPORT.json`",
        f"- MD: `{data['run_dir']}/GPT_ANALYSIS_DATA_REPORT.md`",
        f"- Training curves: `{data['run_dir']}/dexpbt_8gpu_ws/metrics/training_curve_policy_*.jsonl`",
        f"- Curriculum control: `{data['run_dir']}/dexpbt_8gpu_ws/curriculum_control/*.json`",
        f"- Tolerance sweeps: `{SWEEP_DIR}/`",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    data = build_report()
    json_path = RUN_DIR / "GPT_ANALYSIS_DATA_REPORT.json"
    md_path = RUN_DIR / "GPT_ANALYSIS_DATA_REPORT.md"
    json_path.write_text(json.dumps(data, indent=2))
    md_path.write_text(render_md(data))
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
