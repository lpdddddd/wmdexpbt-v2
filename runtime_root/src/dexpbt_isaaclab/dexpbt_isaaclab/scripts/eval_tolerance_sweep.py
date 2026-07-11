#!/usr/bin/env python3
"""Fixed success_tolerance sweep for a single checkpoint.

Runs eval_fair once per tolerance value and merges results into CSV/JSON.
Use to locate the tolerance band where nsucc_mean begins to collapse.

Example:
  python -m dexpbt_isaaclab.scripts.eval_tolerance_sweep \
    --task DexPBT-Regrasping-KukaAllegro-Direct-v0 \
    --checkpoint_path /path/to/000024.pth \
    --label p0_480M \
    --tolerance_list 0.04,0.03,0.02,0.015,0.01 \
    --workspace /path/to/dexpbt_8gpu_ws \
    --num_envs 512 --num_episodes 512 \
    --output_csv /tmp/sweep.csv --output_json /tmp/sweep.json
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import tempfile
from pathlib import Path


def _parse_tolerances(raw: str) -> list[float]:
    out: list[float] = []
    for part in raw.split(","):
        part = part.strip()
        if part:
            out.append(float(part))
    if not out:
        raise ValueError("tolerance_list must contain at least one value")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Fixed tolerance sweep via eval_fair.")
    parser.add_argument("--task", type=str, required=True)
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--workspace", type=str, required=True,
                        help="PBT workspace (required by eval_fair even with checkpoint_override)")
    parser.add_argument("--label", type=str, default="sweep",
                        help="Tag stored in method_name / output rows")
    parser.add_argument("--meta_json", type=str, default=None,
                        help="Optional meta JSON for checkpoint tolerance reference")
    parser.add_argument("--tolerance_list", type=str,
                        default="0.04,0.03,0.02,0.015,0.01")
    parser.add_argument("--num_envs", type=int, default=512)
    parser.add_argument("--num_episodes", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_steps", type=int, default=200_000)
    parser.add_argument("--output_csv", type=str, required=True)
    parser.add_argument("--output_json", type=str, required=True)
    parser.add_argument("--headless", action="store_true", default=True)
    args = parser.parse_args()

    ckpt = Path(args.checkpoint_path).resolve()
    if not ckpt.is_file():
        print(f"[tolerance_sweep] checkpoint not found: {ckpt}", file=sys.stderr)
        sys.exit(2)

    tolerances = _parse_tolerances(args.tolerance_list)
    meta_tol = None
    if args.meta_json:
        meta = json.loads(Path(args.meta_json).read_text())
        curr = meta.get("curriculum") or {}
        meta_tol = curr.get("success_tolerance")

    rows: list[dict] = []
    with tempfile.TemporaryDirectory(prefix="tol_sweep_") as tmpdir:
        for tol in tolerances:
            tmp_csv = Path(tmpdir) / f"sweep_{tol:.4f}.csv"
            cmd = [
                sys.executable, "-m", "dexpbt_isaaclab.scripts.eval_fair",
                "--workspace", str(Path(args.workspace).resolve()),
                "--method_name", f"{args.label}@tol={tol:.4f}",
                "--task", args.task,
                "--checkpoint_rule", "best_r_meta",
                "--checkpoint_override", str(ckpt),
                "--eval_tolerance_mode", "fixed",
                "--fixed_success_tolerance", str(tol),
                "--num_envs", str(args.num_envs),
                "--num_episodes", str(args.num_episodes),
                "--seed", str(args.seed),
                "--max_steps", str(args.max_steps),
                "--output_csv", str(tmp_csv),
                "--headless",
            ]
            print(f"[tolerance_sweep] running tol={tol:.4f} ...")
            proc = subprocess.run(cmd, capture_output=False)
            if proc.returncode != 0:
                print(f"[tolerance_sweep] eval_fair failed for tol={tol}", file=sys.stderr)
                sys.exit(proc.returncode)
            if not tmp_csv.is_file():
                print(f"[tolerance_sweep] missing output {tmp_csv}", file=sys.stderr)
                sys.exit(3)
            with open(tmp_csv, newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    row["sweep_label"] = args.label
                    row["requested_tolerance"] = tol
                    row["checkpoint_training_tolerance"] = meta_tol if meta_tol is not None else ""
                    rows.append(row)

    out_csv = Path(args.output_csv).resolve()
    out_json = Path(args.output_json).resolve()
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_json.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = sorted({k for r in rows for k in r.keys()})
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    summary = {
        "label": args.label,
        "checkpoint_path": str(ckpt),
        "meta_json": args.meta_json,
        "checkpoint_training_tolerance": meta_tol,
        "tolerance_list": tolerances,
        "rows": rows,
    }
    means = [
        (float(r["requested_tolerance"]), float(r["nsucc_mean"]))
        for r in rows
        if r.get("nsucc_mean") not in ("", "nan", "NaN")
    ]
    if means:
        best_tol, best_mean = max(means, key=lambda x: x[1])
        collapse_tol = None
        for tol, mean in sorted(means):
            if best_mean > 0 and mean < best_mean * 0.5:
                collapse_tol = tol
                break
        summary["analysis"] = {
            "best_tolerance": best_tol,
            "best_nsucc_mean": best_mean,
            "first_collapse_below_50pct_tolerance": collapse_tol,
        }

    out_json.write_text(json.dumps(summary, indent=2))
    print(f"[tolerance_sweep] wrote {len(rows)} rows -> {out_csv}")
    print(f"[tolerance_sweep] wrote summary -> {out_json}")
    if "analysis" in summary:
        print(json.dumps(summary["analysis"], indent=2))


if __name__ == "__main__":
    main()
