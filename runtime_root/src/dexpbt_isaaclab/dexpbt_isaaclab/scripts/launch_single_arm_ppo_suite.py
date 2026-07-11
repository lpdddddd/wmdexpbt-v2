#!/usr/bin/env python3
"""Launch the Figure 4 single-arm PPO baseline suite."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys
import time


TASKS = {
    "regrasping": "DexPBT-Regrasping-KukaAllegro-Direct-v0",
    "throw": "DexPBT-Throw-KukaAllegro-Direct-v0",
    "reorientation": "DexPBT-Reorientation-KukaAllegro-Direct-v0",
}

DEFAULT_SEEDS = [11, 22, 33, 44, 55, 66, 77, 88]
SUPERVISOR = Path(__file__).resolve().parents[2] / "run_train_supervisor.sh"


def _read_status(run_dir: Path) -> dict[str, str]:
    status_path = run_dir / "status.txt"
    if not status_path.is_file():
        return {}
    status: dict[str, str] = {}
    for line in status_path.read_text().splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            status[key] = value
    return status


def _pid_is_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _run_is_active(run_dir: Path) -> bool:
    pid_path = run_dir / "pid.txt"
    if not pid_path.is_file():
        return False
    try:
        pid = int(pid_path.read_text().strip())
    except ValueError:
        return False
    return _pid_is_alive(pid)


def _run_is_complete(run_dir: Path) -> bool:
    status = _read_status(run_dir)
    return status.get("status") == "stopped" and status.get("exit_code") == "0"


def _active_supervisor_count(root: Path, ignore_pids: set[int] | None = None) -> int:
    ignore_pids = ignore_pids or set()
    count = 0
    for pid_path in root.glob("**/pid.txt"):
        try:
            pid = int(pid_path.read_text().strip())
        except ValueError:
            continue
        if pid in ignore_pids:
            continue
        if _pid_is_alive(pid):
            count += 1
    return count


def _poll_running(running: list[tuple[subprocess.Popen, str, object]]) -> list[tuple[subprocess.Popen, str, object]]:
    still_running: list[tuple[subprocess.Popen, str, object]] = []
    for proc, label, log_file in running:
        exit_code = proc.poll()
        if exit_code is None:
            still_running.append((proc, label, log_file))
            continue
        log_file.close()
        if exit_code != 0:
            raise RuntimeError(f"{label} failed with exit code {exit_code}")
    return still_running


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch 8-seed PPO baselines for Figure 4.")
    parser.add_argument("--results-root", type=str, required=True)
    parser.add_argument("--num-envs", type=int, default=8192)
    parser.add_argument("--max-iterations", type=int, default=38147)
    parser.add_argument("--gpus", type=str, default="0", help="Comma-separated GPU ids")
    parser.add_argument("--seeds", type=int, nargs="*", default=DEFAULT_SEEDS)
    parser.add_argument("--tasks", type=str, nargs="*", default=list(TASKS.keys()))
    parser.add_argument("--extra", nargs="*", default=[])
    args = parser.parse_args()

    gpu_ids = [gpu.strip() for gpu in args.gpus.split(",") if gpu.strip()]
    if not gpu_ids:
        raise ValueError("At least one GPU id must be provided via --gpus")

    results_root = Path(args.results_root).resolve()
    results_root.mkdir(parents=True, exist_ok=True)

    launch_index = 0
    running: list[tuple[subprocess.Popen, str, object]] = []
    for task_name in args.tasks:
        task_id = TASKS[task_name]
        task_root = results_root / task_name / "ppo"
        task_root.mkdir(parents=True, exist_ok=True)
        for seed in args.seeds:
            run_dir = task_root / f"seed_{seed:03d}"
            if _run_is_complete(run_dir):
                print(f"[PPO] skip completed task={task_name} seed={seed}", flush=True)
                continue
            if _run_is_active(run_dir):
                print(f"[PPO] skip active task={task_name} seed={seed}", flush=True)
                continue

            while True:
                running = _poll_running(running)
                running_pids = {proc.pid for proc, _, _ in running}
                active_count = _active_supervisor_count(results_root, ignore_pids=running_pids)
                if len(running) + active_count < len(gpu_ids):
                    break
                time.sleep(5)
            gpu = gpu_ids[launch_index % len(gpu_ids)]
            launch_index += 1
            run_dir.mkdir(parents=True, exist_ok=True)
            log_path = run_dir / "train.log"
            cmd = [
                "bash",
                str(SUPERVISOR),
                str(run_dir),
                task_id,
                "--device=cuda:0",
                "--headless",
                f"--seed={seed}",
                f"--num_envs={args.num_envs}",
                f"--max_iterations={args.max_iterations}",
            ]
            cmd.extend(args.extra)
            print(f"[PPO] task={task_name} seed={seed} gpu={gpu}: {' '.join(cmd)}", flush=True)
            env = dict(os.environ)
            env["CUDA_VISIBLE_DEVICES"] = gpu
            log_file = open(log_path, "ab")
            proc = subprocess.Popen(
                cmd,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                env=env,
                start_new_session=True,
            )
            running.append((proc, f"PPO task={task_name} seed={seed}", log_file))

    while running:
        running = _poll_running(running)
        if running:
            time.sleep(10)


if __name__ == "__main__":
    main()
