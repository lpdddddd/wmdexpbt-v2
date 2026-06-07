#!/usr/bin/env python3
"""
Evaluate DexPBT Figure 4 single-arm suite checkpoints and record videos.

Paper methodology: deterministic evaluation at final checkpoint,
measuring Nsucc (number of consecutive successful steps) per episode,
using target success_tolerance (ε* = 0.01 for regrasping/reorientation,
ε* = 0.075 for throw).


Usage (full suite, stats only):
    python -m dexpbt_isaaclab.scripts.eval_all_checkpoints \
        --results-root results/figure4_single_arm_suite_20260410_143535 \
        --output-dir results/eval_results

Usage (with videos):
    python -m dexpbt_isaaclab.scripts.eval_all_checkpoints \
        --results-root results/figure4_single_arm_suite_20260410_143535 \
        --output-dir results/eval_results \
        --video \
        --num-video-envs 4 \
        --video-length 300

Usage (specific tasks/methods):
    python -m dexpbt_isaaclab.scripts.eval_all_checkpoints \
        --results-root results/figure4_single_arm_suite_20260410_143535 \
        --output-dir results/eval_results \
        --tasks regrasping throw reorientation \
        --methods ppo dexpbt
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Tasks and method configuration
# ---------------------------------------------------------------------------

TASK_TO_GYM_ID = {
    "regrasping": "DexPBT-Regrasping-KukaAllegro-Direct-v0",
    "throw": "DexPBT-Throw-KukaAllegro-Direct-v0",
    "reorientation": "DexPBT-Reorientation-KukaAllegro-Direct-v0",
}

TASKS = ["regrasping", "throw", "reorientation"]
METHODS = ["ppo", "dexpbt"]

# Target success tolerances per paper:
# - regrasping: ε* = 0.01 (1 cm)
# - reorientation: ε* = 0.01 (1 cm)
# - throw: ε* = 0.075 (7.5 cm)
TASK_TARGET_TOLERANCE = {
    "regrasping": 0.01,
    "throw": 0.075,
    "reorientation": 0.01,
}


def find_all_checkpoints(
    results_root: Path,
    task: str,
    method: str,
) -> list[tuple[str, Path]]:
    """Find all (label, checkpoint_path) for a given task/method.

    Returns sorted by label so runs are reproducible.
    """
    method_dir = results_root / task / method
    if not method_dir.exists():
        return []

    items: list[tuple[str, Path]] = []

    if method == "ppo":
        # Direct seed dirs
        for seed_dir in sorted(method_dir.iterdir()):
            if not seed_dir.is_dir():
                continue
            ckpt = _find_latest_checkpoint(seed_dir)
            if ckpt:
                items.append((seed_dir.name, ckpt))
    else:
        # DexPBT: policy_* dirs
        for pol_dir in sorted(method_dir.iterdir()):
            if not pol_dir.is_dir():
                continue
            if not pol_dir.name.startswith("policy_"):
                continue
            ckpt = _find_latest_checkpoint(pol_dir)
            if ckpt:
                items.append((pol_dir.name, ckpt))

    return items


def _find_latest_checkpoint(seed_dir: Path) -> Optional[Path]:
    """Find checkpoint with highest reward across all attempt subdirs.

    For PBT runs, the last checkpoint may be a replaced (worse) policy,
    so we pick the checkpoint with the highest reward recorded in the filename.
    For non-PBT (PPO baseline), this still picks the best reward checkpoint.
    """
    best: Optional[Path] = None
    best_reward = -float("inf")
    for nn_dir in seed_dir.rglob("nn"):
        for pth in nn_dir.glob("last_DexPBTKukaAllegroLSTMPPO_ep_*.pth"):
            name = pth.stem
            # Parse: last_DexPBTKukaAllegroLSTMPPO_ep_XXXXX_rew_YYYYY.pth
            parts = name.split("_ep_")
            if len(parts) < 2:
                continue
            ep_parts = parts[1].split("_rew_")
            if len(ep_parts) < 2:
                continue
            try:
                rew_str = ep_parts[1].strip("_")
                reward = float(rew_str)
            except (ValueError, IndexError):
                continue
            if reward > best_reward:
                best_reward = reward
                best = pth
    return best


# ---------------------------------------------------------------------------
# Subprocess evaluation (runs in isaaclab311 environment)
# ---------------------------------------------------------------------------

_EVAL_SCRIPT = "dexpbt_isaaclab.scripts._eval_single_ckpt"


def run_eval_subprocess(
    task_gym_id: str,
    checkpoint: Path,
    num_envs: int,
    num_episodes: int,
    output_json: Path,
    video: bool = False,
    video_length: int = 300,
    video_dir: Optional[Path] = None,
    seed: int = 42,
) -> Optional[dict]:
    """Run _eval_single_ckpt.py as a subprocess via isaaclab311 python."""
    cmd = [
        "python", "-m", _EVAL_SCRIPT,
        "--task", task_gym_id,
        "--checkpoint", str(checkpoint),
        "--num_envs", str(num_envs),
        "--num_episodes", str(num_episodes),
        "--output_json", str(output_json),
        "--seed", str(seed),
        "--headless",
    ]
    if video:
        cmd += ["--video", "--video_length", str(video_length)]
        if video_dir:
            cmd += ["--video_dir", str(video_dir)]

    env = os.environ.copy()
    # Propagate Vulkan setup from supervisor script
    vulkan = "/root/autodl-tmp/vulkan_sdk_1.4.304.0/x86_64"
    if Path(vulkan).exists():
        env["VULKAN_SDK"] = vulkan
        env["PATH"] = f"{vulkan}/bin:{env.get('PATH', '')}"
        env["LD_LIBRARY_PATH"] = f"{vulkan}/lib:{env.get('LD_LIBRARY_PATH', '')}"
        env["VK_LAYER_PATH"] = f"{vulkan}/share/vulkan/explicit_layer.d"
        env["VK_ICD_FILENAMES"] = "/root/autodl-tmp/nvidia_egl_icd.json"
    env["XDG_RUNTIME_DIR"] = "/tmp/runtime-root"
    env["TORCHDYNAMO_DISABLE"] = "1"
    env["PYTHONUNBUFFERED"] = "1"

    print(f"\n{'='*60}")
    print(f"  TASK={task_gym_id}  CHECKPOINT={checkpoint.name}")
    print(f"  CMD: {' '.join(cmd)}")
    print(f"{'='*60}")

    # CWD must not contain a dexpbt_isaaclab/ directory, otherwise sys.path[0]=''
    # resolves it as a namespace package (skipping __init__.py and gym.register).
    result = subprocess.run(cmd, env=env, capture_output=False, cwd="/tmp")
    if result.returncode != 0:
        print(f"[WARN] Evaluation failed (exit {result.returncode})")
        return None

    if output_json.exists():
        with open(output_json) as f:
            return json.load(f)
    return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate all checkpoints in a Figure 4 suite."
    )
    parser.add_argument(
        "--results-root", type=str,
        default="/root/autodl-tmp/dexpbt_isaaclab/results/figure4_single_arm_suite_20260410_143535",
    )
    parser.add_argument(
        "--output-dir", type=str,
        default="/root/autodl-tmp/dexpbt_isaaclab/results/eval_results",
    )
    parser.add_argument(
        "--num-envs", type=int, default=512,
        help="Number of envs for stats collection (default: 512).",
    )
    parser.add_argument(
        "--num-episodes", type=int, default=512,
        help="Minimum episodes to collect before reporting (default: 512).",
    )
    parser.add_argument(
        "--video", action="store_true", default=False,
        help="Also record videos.",
    )
    parser.add_argument(
        "--num-video-envs", type=int, default=4,
        help="Number of envs for video recording (default: 4).",
    )
    parser.add_argument(
        "--video-length", type=int, default=300,
        help="Number of steps per video episode (default: 300).",
    )
    parser.add_argument(
        "--tasks", nargs="+", default=None,
        help="Restrict to specific tasks (default: all three).",
    )
    parser.add_argument(
        "--methods", nargs="+", default=None,
        help="Restrict to specific methods (default: both ppo and dexpbt).",
    )
    parser.add_argument(
        "--skip-stats", action="store_true", default=False,
        help="Skip stats collection (only record videos).",
    )
    args = parser.parse_args()

    results_root = Path(args.results_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    video_base_dir = output_dir / "videos"
    if args.video:
        video_base_dir.mkdir(parents=True, exist_ok=True)

    tasks = args.tasks or TASKS
    methods = args.methods or METHODS

    all_results: list[dict] = []
    video_records: list[dict] = []

    total_jobs = 0
    for task in tasks:
        for method in methods:
            items = find_all_checkpoints(results_root, task, method)
            total_jobs += len(items)

    job_idx = 0
    for task in tasks:
        gym_id = TASK_TO_GYM_ID[task]
        target_tol = TASK_TARGET_TOLERANCE[task]

        for method in methods:
            items = find_all_checkpoints(results_root, task, method)
            if not items:
                print(f"[INFO] No checkpoints found for {task}/{method}, skipping.")
                continue

            print(f"\n{'#'*70}")
            print(f"# {task.upper()} / {method.upper()} — {len(items)} checkpoints")
            print(f"#'*70")

            for label, ckpt in items:
                job_idx += 1
                print(f"\n[{job_idx}/{total_jobs}] {task}/{method}/{label}")

                # --- Stats evaluation ---
                if not args.skip_stats:
                    stats_json = output_dir / f"stats_{task}_{method}_{label}.json"
                    stats_result = run_eval_subprocess(
                        task_gym_id=gym_id,
                        checkpoint=ckpt,
                        num_envs=args.num_envs,
                        num_episodes=args.num_episodes,
                        output_json=stats_json,
                        video=False,
                    )

                    row = {
                        "task": task,
                        "method": method,
                        "label": label,
                        "checkpoint": str(ckpt),
                        "target_tolerance": target_tol,
                    }
                    if stats_result:
                        row.update({
                            "num_episodes": stats_result.get("num_episodes", 0),
                            "success_tolerance": stats_result.get("success_tolerance"),
                            "nsucc_mean": stats_result.get("nsucc_mean"),
                            "nsucc_std": stats_result.get("nsucc_std"),
                            "nsucc_median": stats_result.get("nsucc_median"),
                            "nsucc_min": stats_result.get("nsucc_min"),
                            "nsucc_max": stats_result.get("nsucc_max"),
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

                # --- Video recording ---
                if args.video:
                    vid_subdir = video_base_dir / task / method / label
                    vid_subdir.mkdir(parents=True, exist_ok=True)
                    vid_json = vid_subdir / "result.json"
                    vid_result = run_eval_subprocess(
                        task_gym_id=gym_id,
                        checkpoint=ckpt,
                        num_envs=args.num_video_envs,
                        num_episodes=args.num_video_envs,
                        output_json=vid_json,
                        video=True,
                        video_length=args.video_length,
                        video_dir=vid_subdir,
                    )
                    if vid_result:
                        video_records.append({
                            "task": task,
                            "method": method,
                            "label": label,
                            "video_subdir": str(vid_subdir),
                        })

    # -------------------------------------------------------------------------
    # Write results
    # -------------------------------------------------------------------------
    if all_results:
        csv_path = output_dir / "eval_summary.csv"
        fieldnames = [
            "task", "method", "label", "num_episodes",
            "target_tolerance", "success_tolerance",
            "nsucc_mean", "nsucc_std", "nsucc_median",
            "nsucc_min", "nsucc_max", "checkpoint",
        ]
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_results)

        json_path = output_dir / "eval_summary.json"
        with open(json_path, "w") as f:
            json.dump(all_results, f, indent=2)

    if video_records:
        vid_manifest_path = output_dir / "video_manifest.json"
        with open(vid_manifest_path, "w") as f:
            json.dump(video_records, f, indent=2)

    # -------------------------------------------------------------------------
    # Print summary table
    # -------------------------------------------------------------------------
    print("\n" + "=" * 110)
    print("  EVALUATION SUMMARY")
    print("=" * 110)
    print(f"  {'Task':<16} {'Method':<8} {'Label':<12} {'Ep':<6} "
          f"{'ε*':<6} {'ε':<7} {'N_succ Mean':<12} {'Std':<8} {'Med':<8} {'Min':<5} {'Max':<5}")
    print("-" * 110)
    for r in all_results:
        ep = r.get("num_episodes", 0)
        tt = f"{r.get('target_tolerance', 0):.3f}"
        te = f"{r.get('success_tolerance', 0):.4f}" if r.get("success_tolerance") is not None else "N/A"
        mn = f"{r.get('nsucc_mean'):.2f}" if r.get("nsucc_mean") is not None else "N/A"
        sd = f"{r.get('nsucc_std'):.2f}" if r.get("nsucc_std") is not None else "N/A"
        md = f"{r.get('nsucc_median'):.1f}" if r.get("nsucc_median") is not None else "N/A"
        mi = f"{r.get('nsucc_min'):.0f}" if r.get("nsucc_min") is not None else "N/A"
        mx = f"{r.get('nsucc_max'):.0f}" if r.get("nsucc_max") is not None else "N/A"
        print(f"  {r['task']:<16} {r['method']:<8} {r['label']:<12} {ep:<6} "
              f"{tt:<6} {te:<7} {mn:<12} {sd:<8} {md:<8} {mi:<5} {mx:<5}")
    print("=" * 110)
    print(f"\n[INFO] Results: {output_dir}")
    if all_results:
        print(f"[INFO] CSV:     {csv_path}")
        print(f"[INFO] JSON:    {json_path}")
    if args.video and video_records:
        print(f"[INFO] Videos:  {video_base_dir}")
        print(f"[INFO] Manifest: {vid_manifest_path}")


if __name__ == "__main__":
    main()
