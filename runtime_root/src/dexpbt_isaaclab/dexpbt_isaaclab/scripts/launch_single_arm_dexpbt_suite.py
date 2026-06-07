#!/usr/bin/env python3
"""Launch the Figure 4 single-arm DexPBT suite."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys


TASKS = {
    "regrasping": "DexPBT-Regrasping-KukaAllegro-Direct-v0",
    "throw": "DexPBT-Throw-KukaAllegro-Direct-v0",
    "reorientation": "DexPBT-Reorientation-KukaAllegro-Direct-v0",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch |P|=8 DexPBT runs for Figure 4.")
    parser.add_argument("--results-root", type=str, required=True)
    parser.add_argument("--num-policies", type=int, default=6)
    parser.add_argument("--num-envs", type=int, default=8192)
    parser.add_argument("--max-iterations", type=int, default=38147)
    parser.add_argument("--gpus", type=str, default="0")
    parser.add_argument("--tasks", type=str, nargs="*", default=list(TASKS.keys()))
    parser.add_argument("--pbt-interval-steps", type=int, default=20_000_000)
    parser.add_argument("--pbt-start-after", type=int, default=200_000_000)
    parser.add_argument("--pbt-initial-delay", type=int, default=50_000_000)
    parser.add_argument("--mutation-rate", type=float, default=0.2)
    parser.add_argument("--change-min", type=float, default=1.1)
    parser.add_argument("--change-max", type=float, default=1.5)
    parser.add_argument("--extra", nargs="*", default=[])
    args = parser.parse_args()

    results_root = Path(args.results_root).resolve()
    results_root.mkdir(parents=True, exist_ok=True)

    for task_name in args.tasks:
        task_id = TASKS[task_name]
        task_root = results_root / task_name / "dexpbt"
        task_root.mkdir(parents=True, exist_ok=True)
        log_path = task_root / "launcher.log"
        cmd = [
            sys.executable,
            "-m",
            "dexpbt_isaaclab.scripts.launch_pbt",
            f"--task={task_id}",
            f"--num-policies={args.num_policies}",
            f"--num-envs={args.num_envs}",
            f"--max-iterations={args.max_iterations}",
            f"--train-root={task_root}",
            f"--workspace={task_name}_ws",
            f"--gpus={args.gpus}",
            f"--pbt-interval-steps={args.pbt_interval_steps}",
            f"--pbt-start-after={args.pbt_start_after}",
            f"--pbt-initial-delay={args.pbt_initial_delay}",
            f"--mutation-rate={args.mutation_rate}",
            f"--change-min={args.change_min}",
            f"--change-max={args.change_max}",
        ]
        cmd.extend(args.extra)
        print(f"[DexPBT] task={task_name}: {' '.join(cmd)}", flush=True)
        with open(log_path, "ab") as log_file:
            result = subprocess.run(cmd, stdout=log_file, stderr=subprocess.STDOUT, check=False)
        if result.returncode != 0:
            raise SystemExit(f"DexPBT launcher failed for task={task_name} with exit code {result.returncode}")


if __name__ == "__main__":
    main()
