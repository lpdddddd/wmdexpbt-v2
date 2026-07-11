#!/usr/bin/env python3
"""Aggregate per-policy training curve jsonl files into Fig.4-style curves.

Reads `<workspace>/metrics/training_curve_policy_*.jsonl` (written by
pbt_observer at every PBT tick, see M5) and produces:

  <output_dir>/training_curve_policy.csv         (long-format per-policy points)
  <output_dir>/training_curve_population.csv     (population aggregate per
                                                  env_frames bucket)
  <output_dir>/training_curve_population.png     (best/mean N_succ + tolerance)

This intentionally does NOT depend on Isaac Sim. It is a pure post-processing
script meant to be run on the same node or anywhere with read access to the
workspace.

Usage:
  python -m dexpbt_isaaclab.scripts.analyze_training_curve \
    --workspace /path/to/run/dexpbt_8gpu_ws \
    --output_dir /path/to/run/analysis
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path


def _load_points(workspace: Path) -> list[dict]:
    metrics_dir = workspace / "metrics"
    if not metrics_dir.is_dir():
        return []
    rows: list[dict] = []
    for fp in sorted(metrics_dir.glob("training_curve_policy_*.jsonl")):
        with open(fp) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except Exception:
                    continue
    return rows


def _write_policy_csv(rows: list[dict], out_path: Path) -> None:
    if not rows:
        return
    fieldnames = [
        "env_frames", "policy_idx", "pbt_iteration",
        "r_meta", "r_meta_window_size", "nsucc_mean", "nsucc_max",
        "success_tolerance", "target_success_tolerance", "initial_tolerance",
        "frame_since_restart", "last_curriculum_update", "restart_count",
        "tier",
        "checkpoint", "timestamp",
    ]
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in sorted(rows, key=lambda x: (x.get("env_frames", 0), x.get("policy_idx", 0))):
            w.writerow({k: r.get(k, "") for k in fieldnames})


def _bucket_population(rows: list[dict], bucket_steps: int) -> dict[int, dict]:
    """Bucket points by env_frames // bucket_steps so peers with slightly
    different tick boundaries align in the population table."""
    buckets: dict[int, list[dict]] = defaultdict(list)
    for r in rows:
        if "env_frames" not in r:
            continue
        b = int(r["env_frames"] // bucket_steps) * bucket_steps
        buckets[b].append(r)
    aggregated: dict[int, dict] = {}
    for b, pts in buckets.items():
        if not pts:
            continue
        # Keep only one row per policy per bucket — take the latest within the bucket
        latest_by_policy: dict[int, dict] = {}
        for p in pts:
            pid = int(p.get("policy_idx", -1))
            prev = latest_by_policy.get(pid)
            if prev is None or int(p.get("env_frames", 0)) > int(prev.get("env_frames", 0)):
                latest_by_policy[pid] = p
        unique_pts = list(latest_by_policy.values())

        def _valid_floats(values: list) -> list[float]:
            return [float(v) for v in values if v is not None and not (isinstance(v, float) and math.isnan(v))]

        r_metas = _valid_floats([p.get("r_meta") for p in unique_pts])
        nsucc_mean = _valid_floats([p.get("nsucc_mean") for p in unique_pts])
        nsucc_max = _valid_floats([p.get("nsucc_max") for p in unique_pts])
        tols = _valid_floats([p.get("success_tolerance") for p in unique_pts])
        target_tols = _valid_floats([p.get("target_success_tolerance") for p in unique_pts])
        target_tol = min(target_tols) if target_tols else float("nan")

        best_idx = -1
        if r_metas:
            best_idx_in_list = max(range(len(unique_pts)), key=lambda i: float(unique_pts[i].get("r_meta", float("-inf"))))
            best_idx = int(unique_pts[best_idx_in_list].get("policy_idx", -1))

        at_target = sum(1 for t in tols if not math.isnan(target_tol) and t <= target_tol * 1.0001)

        aggregated[b] = {
            "env_frames": b,
            "num_policies": len(unique_pts),
            "best_r_meta": max(r_metas) if r_metas else float("nan"),
            "mean_r_meta": (sum(r_metas) / len(r_metas)) if r_metas else float("nan"),
            "best_nsucc_max": max(nsucc_max) if nsucc_max else float("nan"),
            "best_nsucc_mean": max(nsucc_mean) if nsucc_mean else float("nan"),
            "mean_nsucc_mean": (sum(nsucc_mean) / len(nsucc_mean)) if nsucc_mean else float("nan"),
            # Backward-compatible aliases
            "best_nsucc": max(nsucc_max) if nsucc_max else float("nan"),
            "mean_nsucc": (sum(nsucc_mean) / len(nsucc_mean)) if nsucc_mean else float("nan"),
            "best_success_tolerance": min(tols) if tols else float("nan"),
            "mean_success_tolerance": (sum(tols) / len(tols)) if tols else float("nan"),
            "best_policy_idx": best_idx,
            "num_policies_at_target_tolerance": at_target,
            "target_success_tolerance": target_tol,
        }
    return aggregated


def _write_population_csv(agg: dict[int, dict], out_path: Path) -> None:
    if not agg:
        return
    fieldnames = [
        "env_frames", "num_policies",
        "best_nsucc_max", "mean_nsucc_mean",
        "best_nsucc_mean",
        "best_nsucc", "mean_nsucc",
        "best_r_meta", "mean_r_meta",
        "best_success_tolerance", "mean_success_tolerance",
        "best_policy_idx",
        "num_policies_at_target_tolerance",
        "target_success_tolerance",
    ]
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for b in sorted(agg.keys()):
            w.writerow(agg[b])


def _maybe_plot(agg: dict[int, dict], out_path: Path) -> None:
    if not agg:
        return
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        print("[curve] matplotlib not available, skipping PNG")
        return

    xs = sorted(agg.keys())
    best_n = [agg[b]["best_nsucc_max"] for b in xs]
    mean_n = [agg[b]["mean_nsucc_mean"] for b in xs]
    best_tol = [agg[b]["best_success_tolerance"] for b in xs]

    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(xs, best_n, label="best N_succ", color="C0", linewidth=2)
    ax1.plot(xs, mean_n, label="mean N_succ", color="C0", linestyle="--", linewidth=1.5)
    ax1.set_xlabel("env transitions")
    ax1.set_ylabel("N_succ per episode")
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(xs, best_tol, label="best ε (tightest tolerance)", color="C3", linewidth=1.5)
    ax2.set_ylabel("success_tolerance ε", color="C3")
    ax2.tick_params(axis="y", labelcolor="C3")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")
    plt.title("DexPBT training-time curve (Fig.4 style)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--bucket_steps", type=int, default=20_000_000,
                        help="Align peer points within this many env frames "
                        "(default = pbt_interval_steps)")
    args = parser.parse_args()

    workspace = Path(args.workspace).resolve()
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = _load_points(workspace)
    if not rows:
        print(f"[curve] No training_curve_policy_*.jsonl found under {workspace}/metrics")
        return

    _write_policy_csv(rows, out_dir / "training_curve_policy.csv")
    agg = _bucket_population(rows, args.bucket_steps)
    _write_population_csv(agg, out_dir / "training_curve_population.csv")
    _maybe_plot(agg, out_dir / "training_curve_population.png")
    print(f"[curve] wrote training_curve_policy.csv ({len(rows)} points), "
          f"training_curve_population.csv ({len(agg)} buckets), and PNG to {out_dir}")


if __name__ == "__main__":
    main()
