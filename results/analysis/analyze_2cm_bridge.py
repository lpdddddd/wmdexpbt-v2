#!/usr/bin/env python3
"""2cm bridge diagnostics: contrasts same-bin retention vs naive cross-tol.

Consumes per-policy training curves, curriculum control JSON, curriculum safety
events and pbt copy/rescue events. Missing inputs are skipped, never fatal.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def _load_jsonl(path: Path):
    rows = []
    if not path.exists():
        return rows
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except Exception:
            continue
    return rows


LADDER = [0.025, 0.0235, 0.022, 0.021, 0.0205, 0.020, 0.0195, 0.019, 0.0185, 0.018]


def _first_cross(rows, tol):
    for r in rows:
        cur = r.get("curriculum") or r
        t = cur.get("success_tolerance")
        if t is not None and float(t) <= tol + 1e-9:
            return int(r.get("frame", cur.get("frame", -1)) or -1)
    return -1


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run_dir", required=True)
    p.add_argument("--out_dir", default=None)
    args = p.parse_args()
    run = Path(args.run_dir).resolve()
    ws = run / "dexpbt_8gpu_ws"
    metrics = ws / "metrics"
    out = Path(args.out_dir) if args.out_dir else run / "bridge_analysis"
    out.mkdir(parents=True, exist_ok=True)

    safety = _load_jsonl(metrics / "curriculum_safety_events.jsonl")
    copies = _load_jsonl(metrics / "pbt_copy_events.jsonl")
    rescues = [c for c in copies if c.get("event_type") == "late_degradation_rescue"]

    worker_rows = []
    same_tol_rows = []
    for curve in sorted(metrics.glob("training_curve_policy_*.jsonl")):
        pidx = curve.stem.split("_")[-1]
        rows = _load_jsonl(curve)
        rec = {"policy": pidx}
        for tol in (0.025, 0.022, 0.0205, 0.020):
            rec[f"first_frame_tol_{tol}"] = _first_cross(rows, tol)
        worker_rows.append(rec)
        for r in rows:
            cur = r.get("curriculum") or r
            same_tol_rows.append({
                "policy": pidx,
                "frame": r.get("frame", cur.get("frame", -1)),
                "success_tolerance": cur.get("success_tolerance"),
                "nsucc_mean": r.get("nsucc_mean", cur.get("mean_prev_episode_successes")),
                "same_tol_retention": r.get("same_tol_retention"),
                "same_tol_retention_valid": r.get("same_tol_retention_valid"),
                "naive_cross_tol_retention": r.get("nsucc_mean_drop_ratio"),
            })

    def _write_csv(name, rows):
        if not rows:
            return
        keys = sorted({k for r in rows for k in r})
        with open(out / name, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)

    _write_csv("bridge_worker_summary.csv", worker_rows)
    _write_csv("same_tol_retention.csv", same_tol_rows)
    _write_csv("rollback_events.csv", [s for s in safety if "rollback" in str(s.get("event", ""))])
    _write_csv("late_rescue_events.csv", rescues)
    (out / "bridge_summary.json").write_text(json.dumps({
        "workers": worker_rows,
        "n_rollback_events": len([s for s in safety if "rollback" in str(s.get("event", ""))]),
        "n_late_rescue_events": len(rescues),
        "ladder": LADDER,
    }, indent=2))
    print(f"[analyze-2cm] wrote outputs -> {out}")


if __name__ == "__main__":
    main()
