#!/usr/bin/env python3
"""Aggregate Figure 4 single-arm suite results and emit summary plots."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing import event_accumulator


TASKS = ["regrasping", "throw", "reorientation"]
METHODS = ["ppo", "dexpbt"]


def load_scalar_series(run_dir: Path, tag: str) -> tuple[np.ndarray, np.ndarray] | None:
    event_files = sorted(run_dir.glob("**/events.out.tfevents.*"))
    if not event_files:
        return None
    latest = event_files[-1]
    accumulator = event_accumulator.EventAccumulator(str(latest))
    accumulator.Reload()
    tags = accumulator.Tags().get("scalars", [])
    if tag not in tags:
        return None
    scalars = accumulator.Scalars(tag)
    steps = np.array([scalar.step for scalar in scalars], dtype=np.float64)
    values = np.array([scalar.value for scalar in scalars], dtype=np.float64)
    return steps, values


def interpolate_curves(curves: list[tuple[np.ndarray, np.ndarray]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    max_step = max(curve[0][-1] for curve in curves)
    grid = np.unique(np.concatenate([curve[0] for curve in curves] + [np.array([max_step])]))
    aligned = []
    for steps, values in curves:
        aligned.append(np.interp(grid, steps, values))
    data = np.stack(aligned, axis=0)
    return grid, data.mean(axis=0), data.max(axis=0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate Figure 4 single-arm suite results.")
    parser.add_argument("--results-root", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    args = parser.parse_args()

    results_root = Path(args.results_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict[str, object]] = []
    figure, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()

    for task_idx, task_name in enumerate(TASKS):
        ax = axes[task_idx]
        for method_name in METHODS:
            method_root = results_root / task_name / method_name
            if not method_root.exists():
                continue
            run_dirs = sorted([path for path in method_root.iterdir() if path.is_dir()])
            curves = []
            tolerance_curves = []
            for run_dir in run_dirs:
                run_path = run_dir
                if method_name == "ppo":
                    candidate = run_dir / "run"
                    if candidate.exists():
                        run_path = candidate
                series = load_scalar_series(run_path, "Episode/successes")
                tol_series = load_scalar_series(run_path, "Episode/success_tolerance")
                if series is None:
                    continue
                curves.append(series)
                if tol_series is not None:
                    tolerance_curves.append(tol_series)
                summary_rows.append(
                    {
                        "task": task_name,
                        "method": method_name,
                        "run": run_dir.name,
                        "final_step": int(series[0][-1]),
                        "final_successes": float(series[1][-1]),
                        "best_successes": float(np.max(series[1])),
                        "final_success_tolerance": float(tol_series[1][-1]) if tol_series is not None else None,
                    }
                )
            if not curves:
                continue
            grid, mean_curve, best_curve = interpolate_curves(curves)
            ax.plot(grid, best_curve, label=f"{method_name} best")
            ax.plot(grid, mean_curve, linestyle="--", label=f"{method_name} mean")

        ax.set_title(task_name)
        ax.set_xlabel("env steps")
        ax.set_ylabel("successes")
        ax.legend()

    # fourth panel: reorientation tolerance
    tol_ax = axes[3]
    task_name = "reorientation"
    for method_name in METHODS:
        method_root = results_root / task_name / method_name
        if not method_root.exists():
            continue
        run_dirs = sorted([path for path in method_root.iterdir() if path.is_dir()])
        curves = []
        for run_dir in run_dirs:
            run_path = run_dir
            if method_name == "ppo":
                candidate = run_dir / "run"
                if candidate.exists():
                    run_path = candidate
            tol_series = load_scalar_series(run_path, "Episode/success_tolerance")
            if tol_series is not None:
                curves.append(tol_series)
        if not curves:
            continue
        grid, mean_curve, best_curve = interpolate_curves(curves)
        tol_ax.plot(grid, best_curve, label=f"{method_name} best")
        tol_ax.plot(grid, mean_curve, linestyle="--", label=f"{method_name} mean")
    tol_ax.set_title("reorientation tolerance")
    tol_ax.set_xlabel("env steps")
    tol_ax.set_ylabel("success tolerance")
    tol_ax.legend()

    figure.tight_layout()
    figure.savefig(output_dir / "figure4_single_arm.png", dpi=180)

    with open(output_dir / "summary.json", "w") as file_obj:
        json.dump(summary_rows, file_obj, indent=2)

    with open(output_dir / "summary.csv", "w", newline="") as file_obj:
        writer = csv.DictWriter(
            file_obj,
            fieldnames=[
                "task",
                "method",
                "run",
                "final_step",
                "final_successes",
                "best_successes",
                "final_success_tolerance",
            ],
        )
        writer.writeheader()
        writer.writerows(summary_rows)


if __name__ == "__main__":
    main()
