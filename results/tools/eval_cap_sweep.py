#!/usr/bin/env python3
"""Eval-only cap sweep wrapper around eval_fair.py.

Runs the same fair-eval protocol with multiple
``max_consecutive_successes`` caps so we can probe the upper tail of
nsucc that the training cap (=50) hides.

NOT used during training. Pure post-hoc evaluation on saved checkpoints.

Usage:
  python results/eval_cap_sweep.py \
    --run-dir results/regrasping_fixed_h1_p2_1_formal_5b_<DATE> \
    --caps 50 100 200 \
    --episodes 256 \
    --num-envs 512

Outputs:
  <run-dir>/EVAL_CAP_SWEEP.json    - aggregated results per (policy, cap).
  <run-dir>/eval_cap_sweep/cap_<N>.csv - raw eval_fair CSV per cap.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path
from statistics import mean, median
from typing import Any


WMDEXPBT_ROOT = Path("/root/autodl-tmp/wmdexpbt")
DEFAULT_TASK = "DexPBT-Regrasping-KukaAllegro-Direct-v0"


def _run_one_cap(
    run_dir: Path,
    cap: int,
    *,
    task: str,
    method_name: str,
    num_envs: int,
    num_episodes: int,
    seed: int,
    eval_tolerance_mode: str,
    fixed_success_tolerance: float,
    checkpoint_rule: str,
    extra_eval_args: list[str],
    out_subdir: str = "eval_cap_sweep",
) -> Path:
    """Spawn eval_fair.py for one cap; return the output CSV path."""

    out_dir = run_dir / out_subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / f"cap_{cap}.csv"
    workspace = run_dir / "dexpbt_8gpu_ws"
    if not workspace.is_dir():
        raise SystemExit(f"workspace not found: {workspace}")

    cmd = [
        sys.executable,
        "-m",
        "dexpbt_isaaclab.scripts.eval_fair",
        "--workspace",
        str(workspace),
        "--method_name",
        f"{method_name}_cap{cap}",
        "--task",
        task,
        "--checkpoint_rule",
        checkpoint_rule,
        "--eval_tolerance_mode",
        eval_tolerance_mode,
        "--fixed_success_tolerance",
        str(fixed_success_tolerance),
        "--num_envs",
        str(num_envs),
        "--num_episodes",
        str(num_episodes),
        "--seed",
        str(seed),
        "--output_csv",
        str(out_csv),
        "--max_consecutive_successes_override",
        str(cap),
        "--headless",
    ]
    kit_args = os.environ.get("DEXPBT_KIT_ARGS")
    if kit_args:
        cmd.extend(["--kit_args", kit_args])
    cmd += list(extra_eval_args)

    cwd = WMDEXPBT_ROOT / "runtime_root/src/dexpbt_isaaclab"
    print(f"[eval_cap_sweep] cap={cap} -> {out_csv}")
    print(f"[eval_cap_sweep] cwd={cwd}")
    print(f"[eval_cap_sweep] cmd: {' '.join(shlex.quote(c) for c in cmd)}")
    subprocess.run(cmd, check=True, cwd=str(cwd))
    return out_csv


def _percentiles(values: list[float], qs: list[float]) -> dict[str, float]:
    if not values:
        return {f"p{int(q*100)}": float("nan") for q in qs}
    sorted_vals = sorted(values)
    n = len(sorted_vals)
    out: dict[str, float] = {}
    for q in qs:
        if n == 1:
            out[f"p{int(q*100)}"] = sorted_vals[0]
            continue
        # nearest-rank percentile (no interpolation, simple + reproducible).
        rank = max(0, min(n - 1, int(round(q * (n - 1)))))
        out[f"p{int(q*100)}"] = sorted_vals[rank]
    return out


def _summarize_csv(csv_path: Path, cap: int) -> list[dict[str, Any]]:
    """Each eval_fair CSV has aggregate rows per policy. We re-emit the
    aggregate plus a `cap_hits` flag if nsucc_max == cap."""

    rows: list[dict] = []
    if not csv_path.is_file():
        return rows
    with csv_path.open() as f:
        reader = csv.DictReader(f)
        for r in reader:
            try:
                nsucc_max = float(r.get("nsucc_max", float("nan")))
                nsucc_mean = float(r.get("nsucc_mean", float("nan")))
                nsucc_median = float(r.get("nsucc_median", float("nan")))
                nsucc_std = float(r.get("nsucc_std", float("nan")))
            except ValueError:
                nsucc_max = nsucc_mean = nsucc_median = nsucc_std = float("nan")
            rows.append(
                {
                    "cap": cap,
                    "policy_idx": int(r.get("policy_idx", -1)),
                    "checkpoint_iteration": int(r.get("checkpoint_iteration", -1)),
                    "env_frames": int(r.get("env_frames", -1)),
                    "tolerance_applied": float(r.get("tolerance_applied", float("nan"))),
                    "nsucc_mean": nsucc_mean,
                    "nsucc_median": nsucc_median,
                    "nsucc_max": nsucc_max,
                    "nsucc_std": nsucc_std,
                    "cap_hit": bool(
                        nsucc_max == nsucc_max
                        and nsucc_max >= float(cap) - 1e-6
                    ),
                }
            )
    return rows


def _aggregate(per_policy_rows: list[dict]) -> dict[str, Any]:
    means = [r["nsucc_mean"] for r in per_policy_rows
             if r["nsucc_mean"] == r["nsucc_mean"]]
    maxes = [r["nsucc_max"] for r in per_policy_rows
             if r["nsucc_max"] == r["nsucc_max"]]
    cap_hits = sum(1 for r in per_policy_rows if r["cap_hit"])
    return {
        "n_policies": len(per_policy_rows),
        "mean_nsucc_mean": mean(means) if means else float("nan"),
        "median_nsucc_mean": median(means) if means else float("nan"),
        "max_nsucc_max": max(maxes) if maxes else float("nan"),
        "cap_hit_count": cap_hits,
        "cap_hit_fraction": (
            cap_hits / len(per_policy_rows)
            if per_policy_rows
            else float("nan")
        ),
        # eval_fair's CSV exposes nsucc_mean/median/max per policy; the per-
        # episode array isn't stored, so percentile-of-per-episode is not
        # available here. We expose percentiles ACROSS POLICIES instead.
        **_percentiles(means, [0.5, 0.75, 0.9, 0.99]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Eval-only cap sweep wrapper")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--caps", type=int, nargs="+", default=[50, 100, 200])
    parser.add_argument("--episodes", type=int, default=256)
    parser.add_argument("--num-envs", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--task", type=str, default=DEFAULT_TASK)
    parser.add_argument("--method-name", type=str, default="fixed_h1")
    parser.add_argument(
        "--checkpoint-rule",
        type=str,
        default="best_r_meta",
        choices=["best_r_meta", "last", "best_train_reward"],
    )
    parser.add_argument(
        "--eval-tolerance-mode",
        type=str,
        default="fixed",
        choices=["checkpoint", "fixed"],
    )
    parser.add_argument("--fixed-success-tolerance", type=float, default=0.01)
    parser.add_argument(
        "--extra-eval-args",
        nargs=argparse.REMAINDER,
        default=[],
        help=(
            "Extra args appended to each eval_fair invocation, e.g. "
            "--policy_indices 0,2,5"
        ),
    )
    parser.add_argument("--skip-run", action="store_true",
                        help="Re-aggregate from existing CSVs without running eval_fair")
    parser.add_argument(
        "--out-subdir",
        type=str,
        default="eval_cap_sweep",
        help=(
            "Subdirectory under --run-dir for cap_<N>.csv outputs. Default "
            "'eval_cap_sweep'. For pinned-checkpoint Track A use "
            "'eval_cap_sweep_pinned' so the outputs do not collide."
        ),
    )
    parser.add_argument(
        "--out-json-name",
        type=str,
        default=None,
        help=(
            "Optional aggregated JSON filename (placed under --run-dir). "
            "Defaults to EVAL_CAP_SWEEP.json or EVAL_CAP_SWEEP_PINNED.json "
            "(when --out-subdir contains 'pinned')."
        ),
    )
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    if not run_dir.is_dir():
        raise SystemExit(f"run dir not found: {run_dir}")

    summary: dict[str, Any] = {
        "run_dir": str(run_dir),
        "task": args.task,
        "checkpoint_rule": args.checkpoint_rule,
        "eval_tolerance_mode": args.eval_tolerance_mode,
        "fixed_success_tolerance": args.fixed_success_tolerance,
        "num_envs": args.num_envs,
        "num_episodes": args.episodes,
        "seed": args.seed,
        "caps": args.caps,
        "out_subdir": args.out_subdir,
        "extra_eval_args": list(args.extra_eval_args),
        "per_cap": {},
    }
    for cap in args.caps:
        csv_path = run_dir / args.out_subdir / f"cap_{cap}.csv"
        if not args.skip_run:
            csv_path = _run_one_cap(
                run_dir,
                cap,
                task=args.task,
                method_name=args.method_name,
                num_envs=args.num_envs,
                num_episodes=args.episodes,
                seed=args.seed,
                eval_tolerance_mode=args.eval_tolerance_mode,
                fixed_success_tolerance=args.fixed_success_tolerance,
                checkpoint_rule=args.checkpoint_rule,
                extra_eval_args=args.extra_eval_args,
                out_subdir=args.out_subdir,
            )
        rows = _summarize_csv(csv_path, cap)
        agg = _aggregate(rows)
        summary["per_cap"][str(cap)] = {
            "csv": str(csv_path),
            "rows": rows,
            "aggregate": agg,
        }

    if args.out_json_name:
        out_json = run_dir / args.out_json_name
    elif "pinned" in args.out_subdir:
        out_json = run_dir / "EVAL_CAP_SWEEP_PINNED.json"
    else:
        out_json = run_dir / "EVAL_CAP_SWEEP.json"
    out_json.write_text(json.dumps(summary, indent=2))
    print(f"\n[eval_cap_sweep] wrote {out_json}")
    # Brief stdout summary table.
    print("\ncap | n_policies | mean_nsucc_mean | max_nsucc_max | cap_hits")
    for cap_str, block in summary["per_cap"].items():
        a = block["aggregate"]
        print(
            f"{cap_str:>3} | {a['n_policies']:>10} | "
            f"{a['mean_nsucc_mean']:>15.3f} | "
            f"{a['max_nsucc_max']:>13.3f} | "
            f"{a['cap_hit_count']:>8}"
        )


if __name__ == "__main__":
    main()
