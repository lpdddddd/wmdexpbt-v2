#!/usr/bin/env python3
"""Eval-only nsucc percentile reporter.

Computes p50/p75/p90/p99 of per-episode ``prev_episode_successes`` from
saved checkpoints. Does NOT modify env, observer, or training; the env's
existing ``prev_episode_successes`` tensor is the source of truth. The
training cap (``max_consecutive_successes=50``) stays the default; this
script can optionally probe higher caps via the eval_fair override flag.

Two operating modes:

  1. ``--mode aggregate``: read the eval_fair CSVs already produced by
     ``eval_cap_sweep.py`` and report nsucc percentiles ACROSS POLICIES
     for each cap. No new rollouts. (Fast / read-only.)

  2. ``--mode rollout``: spawn ``eval_fair`` with --num_envs and
     --num_episodes you specify, then post-process the resulting CSV the
     same way. eval_fair already aggregates per-policy mean/median/max;
     this script does not currently capture the per-episode array, so
     across-policy percentiles are what we report.

If you need true per-episode percentiles (within a single policy),
modify ``eval_fair._run_one_eval`` to emit the raw ``episode_nsucc``
array. That is intentionally out of scope here so we keep training
behavior bit-identical to the smoke run.

Usage:
  python results/eval_nsucc_percentile.py \
    --run-dir results/regrasping_fixed_h1_p2_1_formal_5b_<DATE> \
    --caps 50 100 200 \
    --mode aggregate
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean, median
from typing import Any


def _load_csv_rows(csv_path: Path) -> list[dict[str, Any]]:
    if not csv_path.is_file():
        return []
    with csv_path.open() as f:
        return list(csv.DictReader(f))


def _percentiles(values: list[float], qs: list[float]) -> dict[str, float]:
    if not values:
        return {f"p{int(q * 100)}": float("nan") for q in qs}
    sorted_vals = sorted(values)
    n = len(sorted_vals)
    out: dict[str, float] = {}
    for q in qs:
        rank = max(0, min(n - 1, int(round(q * (n - 1)))))
        out[f"p{int(q * 100)}"] = sorted_vals[rank]
    return out


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    means = []
    maxes = []
    medians = []
    for r in rows:
        try:
            m = float(r.get("nsucc_mean", float("nan")))
            mx = float(r.get("nsucc_max", float("nan")))
            mid = float(r.get("nsucc_median", float("nan")))
            if m == m:
                means.append(m)
            if mx == mx:
                maxes.append(mx)
            if mid == mid:
                medians.append(mid)
        except ValueError:
            continue
    return {
        "n_policies": len(rows),
        "mean_nsucc_mean": mean(means) if means else float("nan"),
        "median_nsucc_mean": median(means) if means else float("nan"),
        "max_nsucc_max": max(maxes) if maxes else float("nan"),
        "median_nsucc_median": median(medians) if medians else float("nan"),
        "across_policies_pct_of_mean": _percentiles(
            means, [0.5, 0.75, 0.9, 0.99]
        ),
        "across_policies_pct_of_max": _percentiles(
            maxes, [0.5, 0.75, 0.9, 0.99]
        ),
    }


def _aggregate_mode(
    run_dir: Path,
    caps: list[int],
    input_subdir: str = "eval_cap_sweep",
) -> dict[str, Any]:
    sweep_dir = run_dir / input_subdir
    summary: dict[str, Any] = {
        "run_dir": str(run_dir),
        "mode": "aggregate",
        "input_subdir": input_subdir,
        "caps": caps,
        "per_cap": {},
    }
    for cap in caps:
        csv_path = sweep_dir / f"cap_{cap}.csv"
        rows = _load_csv_rows(csv_path)
        agg = _aggregate(rows)
        summary["per_cap"][str(cap)] = {
            "csv": str(csv_path),
            "n_rows": len(rows),
            "aggregate": agg,
        }
    return summary


def _rollout_mode(args: argparse.Namespace) -> dict[str, Any]:
    """Drive eval_cap_sweep.py and then aggregate. We delegate the
    actual rollout to eval_cap_sweep so that all the eval_fair plumbing
    lives in one place.
    """

    import subprocess
    import sys

    cmd = [
        sys.executable,
        str(Path(__file__).parent / "eval_cap_sweep.py"),
        "--run-dir",
        str(args.run_dir),
        "--caps",
        *[str(c) for c in args.caps],
        "--episodes",
        str(args.episodes),
        "--num-envs",
        str(args.num_envs),
        "--seed",
        str(args.seed),
        "--task",
        args.task,
        "--checkpoint-rule",
        args.checkpoint_rule,
        "--eval-tolerance-mode",
        args.eval_tolerance_mode,
        "--fixed-success-tolerance",
        str(args.fixed_success_tolerance),
        "--out-subdir",
        args.input_subdir,
    ]
    print(f"[eval_nsucc_percentile] dispatching: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    return _aggregate_mode(
        args.run_dir, args.caps, args.input_subdir
    ) | {"mode": "rollout"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Eval-only nsucc percentile reporter")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--mode",
        type=str,
        choices=["aggregate", "rollout"],
        default="aggregate",
    )
    parser.add_argument("--caps", type=int, nargs="+", default=[50, 100, 200])
    parser.add_argument("--episodes", type=int, default=256)
    parser.add_argument("--num-envs", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--task", type=str,
                        default="DexPBT-Regrasping-KukaAllegro-Direct-v0")
    parser.add_argument("--checkpoint-rule", type=str, default="best_r_meta",
                        choices=["best_r_meta", "last", "best_train_reward"])
    parser.add_argument("--eval-tolerance-mode", type=str, default="fixed",
                        choices=["checkpoint", "fixed"])
    parser.add_argument("--fixed-success-tolerance", type=float, default=0.01)
    parser.add_argument(
        "--input-subdir",
        type=str,
        default="eval_cap_sweep",
        help=(
            "Subdirectory under --run-dir containing cap_<N>.csv. Default "
            "'eval_cap_sweep'; use 'eval_cap_sweep_pinned' for the pinned "
            "Track A sweep so the unpinned outputs are not overwritten."
        ),
    )
    parser.add_argument("-o", "--output", type=Path, default=None)
    args = parser.parse_args()
    args.run_dir = args.run_dir.resolve()
    if not args.run_dir.is_dir():
        raise SystemExit(f"run dir not found: {args.run_dir}")

    if args.mode == "rollout":
        summary = _rollout_mode(args)
    else:
        summary = _aggregate_mode(args.run_dir, args.caps, args.input_subdir)

    out = args.output or (args.run_dir / "EVAL_NSUCC_PERCENTILE.json")
    out.write_text(json.dumps(summary, indent=2))
    print(f"\n[eval_nsucc_percentile] wrote {out}")
    print("\ncap | n_policies | mean | median | max_max | p50 | p75 | p90 | p99")
    for cap_str, block in summary["per_cap"].items():
        a = block["aggregate"]
        pcts = a.get("across_policies_pct_of_mean", {})
        print(
            f"{cap_str:>3} | {a['n_policies']:>10} | "
            f"{a['mean_nsucc_mean']:>5.2f} | "
            f"{a['median_nsucc_mean']:>6.2f} | "
            f"{a['max_nsucc_max']:>7.2f} | "
            f"{pcts.get('p50', float('nan')):>5.2f} | "
            f"{pcts.get('p75', float('nan')):>5.2f} | "
            f"{pcts.get('p90', float('nan')):>5.2f} | "
            f"{pcts.get('p99', float('nan')):>5.2f}"
        )


if __name__ == "__main__":
    main()
