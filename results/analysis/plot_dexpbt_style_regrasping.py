#!/usr/bin/env python3
"""Plot DexPBT-style regrasping learning curves from training_curve JSONL metrics.

Usage:
  python results/analysis/plot_dexpbt_style_regrasping.py \\
    --run-dir /path/to/regrasping_* \\
    --out-dir /path/to/run/figures \\
    --title Regrasping
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import warnings
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


FRAME_KEYS = ("population_frame", "env_frames", "frame")
NSUCC_KEYS = ("nsucc_mean", "mean_prev_episode_successes")
TOL_KEYS = ("success_tolerance", "tolerance")


def _sf(value: Any) -> float | None:
    try:
        x = float(value)
        if math.isnan(x) or math.isinf(x):
            return None
        return x
    except (TypeError, ValueError):
        return None


def _pick(row: dict[str, Any], keys: tuple[str, ...]) -> float | None:
    for key in keys:
        if key in row:
            val = _sf(row[key])
            if val is not None:
                return val
    return None


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                warnings.warn(f"{path}:{line_no}: skip bad json ({exc})")
    return rows


def discover_curve_files(metrics_dir: Path) -> list[tuple[int, Path]]:
    found: dict[int, Path] = {}
    pattern = re.compile(r"training_curve_policy_(\d+)\.jsonl$")
    for fp in sorted(metrics_dir.glob("training_curve_policy_*.jsonl")):
        match = pattern.match(fp.name)
        if not match:
            continue
        idx = int(match.group(1))
        found[idx] = fp
    return sorted(found.items())


def load_worker_curves(metrics_dir: Path) -> dict[int, list[tuple[float, float]]]:
    curves: dict[int, list[tuple[float, float]]] = {}
    files = discover_curve_files(metrics_dir)
    if not files:
        return curves

    for policy_idx, path in files:
        points: list[tuple[float, float]] = []
        for row in read_jsonl(path):
            frame = _pick(row, FRAME_KEYS)
            nsucc = _pick(row, NSUCC_KEYS)
            if frame is None or nsucc is None:
                continue
            frame_b = frame / 1e9
            if frame_b < 0:
                continue
            points.append((frame_b, nsucc))
        if points:
            points.sort(key=lambda x: x[0])
            curves[policy_idx] = points
        else:
            warnings.warn(f"{path}: no usable points")

    return curves


def forward_fill(values: np.ndarray) -> np.ndarray:
    """Carry the last observed value forward; leading NaNs stay NaN."""
    out = values.copy()
    last = np.nan
    for idx in range(len(out)):
        if not np.isnan(out[idx]):
            last = out[idx]
        elif not np.isnan(last):
            out[idx] = last
    return out


def bin_worker_curves(
    curves: dict[int, list[tuple[float, float]]],
    *,
    bin_size: float,
) -> tuple[np.ndarray, dict[int, np.ndarray]]:
    if not curves:
        raise RuntimeError("No worker curves to bin")

    max_frame = max(pt[0] for pts in curves.values() for pt in pts)
    bin_edges = np.arange(0.0, max_frame + bin_size, bin_size)
    bin_centers = bin_edges[:-1] + bin_size / 2.0

    binned: dict[int, np.ndarray] = {}
    for policy_idx, points in curves.items():
        values = np.full(len(bin_centers), np.nan, dtype=float)
        for bin_idx in range(len(bin_centers)):
            lo = bin_edges[bin_idx]
            hi = bin_edges[bin_idx + 1]
            in_bin = [ns for fr, ns in points if lo <= fr < hi]
            if in_bin:
                values[bin_idx] = in_bin[-1]
        values = forward_fill(values)
        binned[policy_idx] = values

    return bin_centers, binned


def aggregate_population(
    bin_centers: np.ndarray,
    binned: dict[int, np.ndarray],
) -> dict[str, np.ndarray]:
    if not binned:
        raise RuntimeError("No binned worker data")

    stacked = np.vstack(list(binned.values()))
    present = np.sum(~np.isnan(stacked), axis=0)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        worker_min = np.nanmin(stacked, axis=0)
        worker_max = np.nanmax(stacked, axis=0)
        worker_mean = np.nanmean(stacked, axis=0)

    worker_min[present == 0] = np.nan
    worker_max[present == 0] = np.nan
    worker_mean[present == 0] = np.nan
    best_worker = worker_max.copy()

    return {
        "frame_b": bin_centers,
        "worker_min": worker_min,
        "worker_max": worker_max,
        "worker_mean": worker_mean,
        "best_worker": best_worker,
        "num_workers_present": present.astype(int),
    }


def rolling_smooth(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return values.copy()

    out = np.full_like(values, np.nan, dtype=float)
    for idx in range(len(values)):
        lo = max(0, idx - window // 2)
        hi = min(len(values), idx + window // 2 + 1)
        chunk = values[lo:hi]
        chunk = chunk[~np.isnan(chunk)]
        if chunk.size:
            out[idx] = float(np.mean(chunk))
    return out


def smooth_stats(stats: dict[str, np.ndarray], window: int) -> dict[str, np.ndarray]:
    out = dict(stats)
    for key in ("worker_min", "worker_max", "worker_mean", "best_worker"):
        out[key] = rolling_smooth(stats[key], window)
    return out


def infer_x_max(max_frame_b: float) -> float:
    if max_frame_b >= 4.0:
        return 5.0
    if max_frame_b >= 4.8:
        return 5.0
    return math.ceil(max_frame_b * 2.0) / 2.0


def infer_y_max(stats: dict[str, np.ndarray], default: float) -> float:
    peak = float(np.nanmax(stats["best_worker"]))
    if peak <= 0:
        return default
    # Round up to next 5 with a little headroom.
    return max(default, math.ceil((peak + 2.0) / 5.0) * 5.0)


def write_stats_csv(path: Path, stats: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            [
                "frame_b",
                "worker_min",
                "worker_max",
                "worker_mean",
                "best_worker",
                "num_workers_present",
            ]
        )
        for idx in range(len(stats["frame_b"])):
            row = [
                stats["frame_b"][idx],
                stats["worker_min"][idx],
                stats["worker_max"][idx],
                stats["worker_mean"][idx],
                stats["best_worker"][idx],
                stats["num_workers_present"][idx],
            ]
            writer.writerow(
                "" if (isinstance(v, float) and np.isnan(v)) else v for v in row
            )


def apply_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 11,
            "axes.labelsize": 14,
            "axes.titlesize": 15,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "legend.fontsize": 10,
            "axes.linewidth": 1.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def plot_dexpbt_style(
    stats: dict[str, np.ndarray],
    *,
    title: str,
    x_max: float,
    y_max: float,
    show_legend: bool,
    plot_workers: bool,
    worker_curves: dict[int, list[tuple[float, float]]] | None,
    mark_stages: bool,
    figsize: tuple[float, float],
) -> tuple[plt.Figure, plt.Axes]:
    apply_style()
    fig, ax = plt.subplots(figsize=figsize)

    x = stats["frame_b"]
    y_min = stats["worker_min"]
    y_max_vals = stats["worker_max"]
    y_mean = stats["worker_mean"]
    y_best = stats["best_worker"]

    if plot_workers and worker_curves:
        for points in worker_curves.values():
            xs = [p[0] for p in points]
            ys = [p[1] for p in points]
            ax.plot(xs, ys, color="#5B8CC0", alpha=0.18, linewidth=0.8, zorder=1)

    valid = ~np.isnan(y_min) & ~np.isnan(y_max_vals)
    ax.fill_between(
        x[valid],
        y_min[valid],
        y_max_vals[valid],
        color="#66BB77",
        alpha=0.22,
        linewidth=0,
        zorder=2,
    )

    ax.plot(
        x,
        y_best,
        color="#0B5EA8",
        linewidth=2.6,
        linestyle="-",
        zorder=4,
        label="Best worker",
    )
    ax.plot(
        x,
        y_mean,
        color="#18A64A",
        linewidth=2.0,
        linestyle="-.",
        marker=None,
        zorder=3,
        label="Population mean",
    )

    if mark_stages:
        for stage_x in (2.2, 3.5):
            ax.axvline(stage_x, color="0.5", linestyle="--", linewidth=1.0, alpha=0.6)

    ax.set_xlim(0, x_max)
    ax.set_ylim(0, y_max)
    x_tick_step = 1 if x_max <= 6 else 2
    ax.set_xticks(list(range(0, int(x_max) + 1, x_tick_step)))
    y_tick_step = 10 if y_max <= 60 else 20
    ax.set_yticks(list(range(0, int(y_max) + 1, y_tick_step)))
    ax.minorticks_on()
    ax.tick_params(axis="both", which="major", direction="in", length=5, width=1.0)
    ax.tick_params(axis="both", which="minor", direction="in", length=2.5, width=0.8)
    ax.grid(True, which="major", color="0.85", linewidth=0.5, alpha=0.8)
    ax.grid(True, which="minor", color="0.92", linewidth=0.35, alpha=0.55)

    ax.set_title(title, fontweight="bold", pad=10)
    ax.set_xlabel("Env. steps", labelpad=6)
    ax.set_ylabel("Num. successes", labelpad=6)
    ax.text(
        0.98,
        -0.10,
        r"$\times 10^9$",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=11,
    )

    if show_legend:
        ax.legend(frameon=False, loc="lower right")

    fig.subplots_adjust(left=0.14, right=0.98, bottom=0.14, top=0.92)
    return fig, ax


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot DexPBT-style regrasping learning curves."
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Run directory containing dexpbt_8gpu_ws/metrics/",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory (default: <run-dir>/figures)",
    )
    parser.add_argument("--title", type=str, default="Regrasping")
    parser.add_argument("--bin-size", type=float, default=0.04)
    parser.add_argument("--smooth-window", type=int, default=1)
    parser.add_argument("--x-max", type=float, default=None)
    parser.add_argument("--y-max", type=float, default=None)
    parser.add_argument("--fig-width", type=float, default=7.2)
    parser.add_argument("--fig-height", type=float, default=6.4)
    parser.add_argument("--legend", action="store_true")
    parser.add_argument("--plot-workers", action="store_true")
    parser.add_argument("--mark-stages", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.expanduser().resolve()
    metrics_dir = run_dir / "dexpbt_8gpu_ws" / "metrics"
    out_dir = (args.out_dir or (run_dir / "figures")).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if not metrics_dir.is_dir():
        raise FileNotFoundError(f"Metrics directory not found: {metrics_dir}")

    worker_curves = load_worker_curves(metrics_dir)
    if not worker_curves:
        raise RuntimeError(f"No training curves found under {metrics_dir}")

    missing = [i for i in range(6) if i not in worker_curves]
    for idx in missing:
        print(f"warning: missing training curve for policy {idx}")

    print(f"loaded {len(worker_curves)} worker curves from {metrics_dir}")

    bin_centers, binned = bin_worker_curves(worker_curves, bin_size=args.bin_size)
    stats = aggregate_population(bin_centers, binned)
    if args.smooth_window > 1:
        stats = smooth_stats(stats, args.smooth_window)

    max_frame_b = float(np.nanmax(stats["frame_b"]))
    x_max = args.x_max if args.x_max is not None else infer_x_max(max_frame_b)
    y_max = args.y_max if args.y_max is not None else infer_y_max(stats, 50.0)

    fig, _ax = plot_dexpbt_style(
        stats,
        title=args.title,
        x_max=x_max,
        y_max=y_max,
        show_legend=args.legend,
        plot_workers=args.plot_workers,
        worker_curves=worker_curves,
        mark_stages=args.mark_stages,
        figsize=(args.fig_width, args.fig_height),
    )

    pdf_path = out_dir / "dexpbt_style_regrasping.pdf"
    png_path = out_dir / "dexpbt_style_regrasping.png"
    csv_path = out_dir / "dexpbt_style_regrasping_stats.csv"

    fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.08)
    fig.savefig(png_path, dpi=400, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    write_stats_csv(csv_path, stats)

    print(f"wrote {pdf_path}")
    print(f"wrote {png_path}")
    print(f"wrote {csv_path}")


if __name__ == "__main__":
    main()
