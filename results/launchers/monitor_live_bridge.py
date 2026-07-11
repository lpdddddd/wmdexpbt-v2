#!/usr/bin/env python3
"""Live bridge monitor for 6-policy continuation runs."""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path
from typing import Any


HEADERS = [
    "timestamp",
    "policy_idx",
    "population_frame",
    "success_tolerance",
    "nsucc_mean",
    "nsucc_max",
    "same_level_peak",
    "same_level_retention",
    "bridge_level",
    "bridge_probation_active",
    "probation_bad_streak",
    "shrink_token",
    "forced_tolerance",
    "rollback_reason",
    "late_rescue_executed",
    "source_policy_if_copied",
]


def _last_jsonl(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    last = None
    try:
        with path.open() as fh:
            for line in fh:
                line = line.strip()
                if line:
                    last = json.loads(line)
    except Exception:
        return last
    return last if isinstance(last, dict) else None


def _read_control(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text())
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _f(x: Any, default: float = float("nan")) -> float:
    try:
        v = float(x)
        return v if math.isfinite(v) else default
    except Exception:
        return default


def collect_row(run_dir: Path, policy_idx: int) -> dict[str, Any]:
    ws = run_dir / "dexpbt_8gpu_ws"
    curve = ws / "metrics" / f"training_curve_policy_{policy_idx:02d}.jsonl"
    last = _last_jsonl(curve) or {}
    ctrl = _read_control(ws / "curriculum_control" / f"{policy_idx:03d}.json")
    return {
        "timestamp": time.time(),
        "policy_idx": policy_idx,
        "population_frame": int(
            last.get("env_frames")
            or last.get("population_frame")
            or last.get("frame")
            or 0
        ),
        "success_tolerance": _f(
            last.get("success_tolerance", ctrl.get("forced_success_tolerance"))
        ),
        "nsucc_mean": _f(last.get("nsucc_mean")),
        "nsucc_max": _f(last.get("nsucc_max")),
        "same_level_peak": _f(
            last.get("same_tol_peak", ctrl.get("same_tol_peak_nsucc_mean"))
        ),
        "same_level_retention": _f(
            last.get("same_tol_retention", ctrl.get("same_tol_retention"))
        ),
        "bridge_level": ctrl.get("bridge_level_id", last.get("bridge_level_id", "")),
        "bridge_probation_active": bool(
            ctrl.get("bridge_probation_active", False)
        ),
        "probation_bad_streak": int(
            ctrl.get("bridge_probation_bad_streak", 0) or 0
        ),
        "shrink_token": bool(ctrl.get("shrink_token", False)),
        "forced_tolerance": _f(ctrl.get("forced_success_tolerance")),
        "rollback_reason": str(ctrl.get("rollback_reason", "") or ""),
        "late_rescue_executed": int(
            ctrl.get("late_rescue_executed_count", 0) or 0
        ),
        "source_policy_if_copied": ctrl.get("source_policy_idx_for_copy", ""),
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    means = [r["nsucc_mean"] for r in rows if math.isfinite(r["nsucc_mean"])]
    tols = [r["success_tolerance"] for r in rows if math.isfinite(r["success_tolerance"])]
    means_sorted = sorted(means)

    def _count_le(thr: float) -> int:
        return sum(1 for t in tols if t <= thr)

    return {
        "mean_nsucc": float(sum(means) / len(means)) if means else float("nan"),
        "median_nsucc": (
            float(means_sorted[len(means_sorted) // 2]) if means_sorted else float("nan")
        ),
        "max_nsucc": float(max(means)) if means else float("nan"),
        "min_nsucc": float(min(means)) if means else float("nan"),
        "workers_le_020": _count_le(0.020),
        "workers_le_0195": _count_le(0.0195),
        "workers_le_019": _count_le(0.019),
        "workers_le_0185": _count_le(0.0185),
        "workers_rollback": sum(1 for r in rows if r.get("rollback_reason")),
        "n_policies": len(rows),
    }


def format_terminal(rows: list[dict[str, Any]], summary: dict[str, Any]) -> str:
    pf = max((r["population_frame"] for r in rows), default=0)
    lines = [f"PF={pf / 1e9:.2f}B"]
    for r in rows:
        lines.append(
            f"p{r['policy_idx']} tol={r['success_tolerance']:.4f} "
            f"nsucc={r['nsucc_mean']:.1f} ret={r['same_level_retention']:.2f}"
        )
    lines.append(
        f"population mean={summary['mean_nsucc']:.1f} "
        f"workers<=.020: {summary['workers_le_020']}/{summary['n_policies']}"
    )
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--interval-s", type=float, default=60.0)
    ap.add_argument("--num-policies", type=int, default=6)
    args = ap.parse_args()
    run_dir = Path(args.run_dir)
    csv_path = run_dir / "live_bridge_summary.csv"
    jsonl_path = run_dir / "live_bridge_summary.jsonl"
    write_header = not csv_path.exists()
    with csv_path.open("a", newline="") as csv_fh, jsonl_path.open("a") as jl_fh:
        writer = csv.DictWriter(csv_fh, fieldnames=HEADERS)
        if write_header:
            writer.writeheader()
            csv_fh.flush()
        while True:
            rows = [
                collect_row(run_dir, i) for i in range(int(args.num_policies))
            ]
            summary = summarize(rows)
            for row in rows:
                writer.writerow(row)
                payload = dict(row)
                payload["summary"] = summary
                jl_fh.write(json.dumps(payload) + "\n")
            csv_fh.flush()
            jl_fh.flush()
            print(format_terminal(rows, summary), flush=True)
            print("---", flush=True)
            time.sleep(float(args.interval_s))


if __name__ == "__main__":
    main()
