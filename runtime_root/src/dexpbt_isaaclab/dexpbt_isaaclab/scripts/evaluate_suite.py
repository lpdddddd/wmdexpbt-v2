"""Evaluate all checkpoints in a Figure 4 suite results directory.

Walks the results tree, finds the latest checkpoint for each task/method/seed,
runs evaluate_checkpoint.py for each, and produces a consolidated summary CSV.

Usage:
    python -m dexpbt_isaaclab.scripts.evaluate_suite \
        --results-root results/figure4_single_arm_suite_20260410_143535 \
        --output-dir results/eval_summary
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

TASK_TO_GYM_ID = {
    "regrasping": "DexPBT-Regrasping-KukaAllegro-Direct-v0",
    "throw": "DexPBT-Throw-KukaAllegro-Direct-v0",
    "reorientation": "DexPBT-Reorientation-KukaAllegro-Direct-v0",
}

TASKS = ["regrasping", "throw", "reorientation"]
METHODS = ["ppo", "dexpbt"]


def find_latest_checkpoint(seed_dir: Path) -> Path | None:
    """Find the last checkpoint (highest epoch) across all attempt dirs."""
    rl_games_dir = seed_dir / "rl_games"
    if not rl_games_dir.exists():
        return None

    best_ckpt: Path | None = None
    best_epoch = -1

    for nn_dir in rl_games_dir.rglob("nn"):
        for pth in nn_dir.glob("last_*.pth"):
            name = pth.stem
            parts = name.split("_ep_")
            if len(parts) < 2:
                continue
            try:
                epoch = int(parts[1].split("_")[0])
            except (ValueError, IndexError):
                continue
            if epoch > best_epoch:
                best_epoch = epoch
                best_ckpt = pth

    return best_ckpt


def run_eval(
    task_gym_id: str,
    checkpoint: Path,
    num_envs: int,
    num_episodes: int,
    output_json: Path,
    video: bool = False,
    video_length: int = 500,
) -> dict | None:
    """Run evaluate_checkpoint.py as a subprocess and return results."""
    cmd = [
        sys.executable, "-m", "dexpbt_isaaclab.scripts.evaluate_checkpoint",
        "--task", task_gym_id,
        "--checkpoint", str(checkpoint),
        "--num_envs", str(num_envs),
        "--num_episodes", str(num_episodes),
        "--output_json", str(output_json),
        "--headless",
    ]
    if video:
        cmd += ["--video", "--video_length", str(video_length)]

    print(f"\n{'='*60}")
    print(f"  Evaluating: {checkpoint.parent.parent.parent.name} / {checkpoint.name}")
    print(f"  Command: {' '.join(cmd)}")
    print(f"{'='*60}\n")

    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        print(f"[WARN] Evaluation failed for {checkpoint} (exit code {result.returncode})")
        return None

    if output_json.exists():
        with open(output_json) as f:
            return json.load(f)
    return None


def main():
    parser = argparse.ArgumentParser(description="Evaluate all checkpoints in a suite.")
    parser.add_argument("--results-root", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--num-envs", type=int, default=512)
    parser.add_argument("--num-episodes", type=int, default=512)
    parser.add_argument("--video", action="store_true", default=False,
                        help="Also record video for each checkpoint (uses 4 envs).")
    parser.add_argument("--video-length", type=int, default=500)
    parser.add_argument("--tasks", nargs="+", default=None,
                        help="Restrict to specific tasks (default: all).")
    parser.add_argument("--methods", nargs="+", default=None,
                        help="Restrict to specific methods (default: all).")
    args = parser.parse_args()

    results_root = Path(args.results_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    tasks = args.tasks or TASKS
    methods = args.methods or METHODS

    all_results: list[dict] = []

    for task_name in tasks:
        gym_id = TASK_TO_GYM_ID.get(task_name)
        if gym_id is None:
            print(f"[WARN] Unknown task: {task_name}, skipping.")
            continue

        for method in methods:
            method_dir = results_root / task_name / method
            if not method_dir.exists():
                print(f"[INFO] No directory for {task_name}/{method}, skipping.")
                continue

            seed_dirs = sorted([d for d in method_dir.iterdir() if d.is_dir()])
            for seed_dir in seed_dirs:
                ckpt = find_latest_checkpoint(seed_dir)
                if ckpt is None:
                    print(f"[WARN] No checkpoint found in {seed_dir}, skipping.")
                    continue

                eval_json = output_dir / f"{task_name}_{method}_{seed_dir.name}_eval.json"
                eval_result = run_eval(
                    task_gym_id=gym_id,
                    checkpoint=ckpt,
                    num_envs=args.num_envs,
                    num_episodes=args.num_episodes,
                    output_json=eval_json,
                    video=args.video,
                    video_length=args.video_length,
                )

                row = {
                    "task": task_name,
                    "method": method,
                    "seed": seed_dir.name,
                    "checkpoint": str(ckpt),
                }
                if eval_result:
                    row.update({
                        "num_episodes": eval_result.get("num_episodes", 0),
                        "success_tolerance": eval_result.get("success_tolerance"),
                        "nsucc_mean": eval_result.get("nsucc_mean"),
                        "nsucc_std": eval_result.get("nsucc_std"),
                        "nsucc_median": eval_result.get("nsucc_median"),
                        "nsucc_min": eval_result.get("nsucc_min"),
                        "nsucc_max": eval_result.get("nsucc_max"),
                    })
                else:
                    row.update({
                        "num_episodes": 0,
                        "success_tolerance": None,
                        "nsucc_mean": None,
                        "nsucc_std": None,
                        "nsucc_median": None,
                        "nsucc_min": None,
                        "nsucc_max": None,
                    })
                all_results.append(row)

    # Write consolidated CSV
    csv_path = output_dir / "eval_summary.csv"
    fieldnames = [
        "task", "method", "seed", "num_episodes",
        "success_tolerance", "nsucc_mean", "nsucc_std",
        "nsucc_median", "nsucc_min", "nsucc_max", "checkpoint",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_results)

    # Write JSON
    json_path = output_dir / "eval_summary.json"
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2)

    # Print summary table
    print("\n" + "=" * 100)
    print("  SUITE EVALUATION SUMMARY")
    print("=" * 100)
    print(f"  {'Task':<16} {'Method':<8} {'Seed':<12} {'Episodes':<10} {'Tolerance':<11} "
          f"{'Mean':<8} {'Std':<8} {'Median':<8} {'Min':<6} {'Max':<6}")
    print("-" * 100)
    for r in all_results:
        tol = f"{r['success_tolerance']:.4f}" if r.get("success_tolerance") is not None else "N/A"
        mean = f"{r['nsucc_mean']:.2f}" if r.get("nsucc_mean") is not None else "N/A"
        std = f"{r['nsucc_std']:.2f}" if r.get("nsucc_std") is not None else "N/A"
        med = f"{r['nsucc_median']:.1f}" if r.get("nsucc_median") is not None else "N/A"
        mn = f"{r['nsucc_min']:.0f}" if r.get("nsucc_min") is not None else "N/A"
        mx = f"{r['nsucc_max']:.0f}" if r.get("nsucc_max") is not None else "N/A"
        print(f"  {r['task']:<16} {r['method']:<8} {r['seed']:<12} "
              f"{r['num_episodes']:<10} {tol:<11} {mean:<8} {std:<8} {med:<8} {mn:<6} {mx:<6}")
    print("=" * 100)
    print(f"\n[INFO] Summary saved to: {csv_path}")
    print(f"[INFO] Full results:     {json_path}")


if __name__ == "__main__":
    main()
