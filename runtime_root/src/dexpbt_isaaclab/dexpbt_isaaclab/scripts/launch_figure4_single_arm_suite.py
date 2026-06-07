#!/usr/bin/env python3
"""Launch the full Figure 4 single-arm suite in a queued manner."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch the full Figure 4 single-arm suite.")
    parser.add_argument("--results-root", type=str, required=True)
    parser.add_argument("--num-envs", type=int, default=8192)
    parser.add_argument("--max-iterations", type=int, default=38147)
    parser.add_argument("--gpus", type=str, default="0,1")
    parser.add_argument("--ppo-seeds", type=int, nargs="*", default=None)
    parser.add_argument("--extra", nargs="*", default=[])
    args = parser.parse_args()

    results_root = Path(args.results_root).resolve()
    results_root.mkdir(parents=True, exist_ok=True)

    ppo_cmd = [
        sys.executable,
        "-m",
        "dexpbt_isaaclab.scripts.launch_single_arm_ppo_suite",
        f"--results-root={results_root}",
        f"--num-envs={args.num_envs}",
        f"--max-iterations={args.max_iterations}",
        f"--gpus={args.gpus}",
    ]
    if args.ppo_seeds is not None:
        ppo_cmd.append("--seeds")
        ppo_cmd.extend(str(seed) for seed in args.ppo_seeds)
    ppo_cmd.extend(args.extra)

    dexpbt_cmd = [
        sys.executable,
        "-m",
        "dexpbt_isaaclab.scripts.launch_single_arm_dexpbt_suite",
        f"--results-root={results_root}",
        f"--num-envs={args.num_envs}",
        f"--max-iterations={args.max_iterations}",
        f"--gpus={args.gpus}",
    ]
    dexpbt_cmd.extend(args.extra)

    with open(results_root / "ppo_suite.log", "ab") as ppo_log:
        ppo_result = subprocess.run(ppo_cmd, stdout=ppo_log, stderr=subprocess.STDOUT, check=False)
    if ppo_result.returncode != 0:
        raise SystemExit(f"PPO suite failed with exit code {ppo_result.returncode}")
    with open(results_root / "dexpbt_suite.log", "ab") as dexpbt_log:
        dexpbt_result = subprocess.run(dexpbt_cmd, stdout=dexpbt_log, stderr=subprocess.STDOUT, check=False)
    if dexpbt_result.returncode != 0:
        raise SystemExit(f"DexPBT suite failed with exit code {dexpbt_result.returncode}")


if __name__ == "__main__":
    main()
