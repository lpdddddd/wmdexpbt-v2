"""Evaluate a trained DexPBT checkpoint and optionally record video.

Usage (stats only):
    python -m dexpbt_isaaclab.scripts.evaluate_checkpoint \
        --task DexPBT-Regrasping-KukaAllegro-Direct-v0 \
        --checkpoint results/.../nn/last_DexPBT_ep_38147_rew_....pth \
        --num_envs 512 --headless

Usage (video):
    python -m dexpbt_isaaclab.scripts.evaluate_checkpoint \
        --task DexPBT-Regrasping-KukaAllegro-Direct-v0 \
        --checkpoint results/.../nn/last_DexPBT_ep_38147_rew_....pth \
        --num_envs 4 --video --video_length 500 --headless
"""

import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Evaluate a DexPBT checkpoint.")
parser.add_argument("--task", type=str, required=True)
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--num_envs", type=int, default=512)
parser.add_argument("--num_episodes", type=int, default=512,
                    help="Minimum number of completed episodes to collect before reporting.")
parser.add_argument("--deterministic", action="store_true", default=True)
parser.add_argument("--stochastic", action="store_true", default=False,
                    help="Use stochastic actions (overrides --deterministic).")
parser.add_argument("--video", action="store_true", default=False)
parser.add_argument("--video_length", type=int, default=500,
                    help="Number of steps to record when --video is set.")
parser.add_argument("--video_dir", type=str, default=None,
                    help="Output directory for MP4 files (defaults to <checkpoint_dir>/videos).")
parser.add_argument("--output_json", type=str, default=None,
                    help="Path to write eval results JSON (defaults to <checkpoint_dir>/eval_results.json).")
parser.add_argument("--seed", type=int, default=42)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

if args_cli.video:
    args_cli.enable_cameras = True

sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ---------------------------------------------------------------------------
# Imports that require Omniverse runtime
# ---------------------------------------------------------------------------

import json
import math
import os
from importlib import resources
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
import yaml
from rl_games.common import env_configurations, vecenv
from rl_games.torch_runner import Runner

from isaaclab_rl.rl_games import RlGamesGpuEnv, RlGamesVecEnvWrapper

import dexpbt_isaaclab  # noqa: F401


def _load_agent_cfg() -> dict:
    cfg_file = resources.files("dexpbt_isaaclab.agents").joinpath("rl_games_ppo_cfg.yaml")
    with open(str(cfg_file)) as f:
        return yaml.safe_load(f)


def _load_env_cfg(task_name: str):
    spec = gym.spec(task_name)
    entry = spec.kwargs["env_cfg_entry_point"]
    module_path, cls_name = entry.rsplit(":", 1)
    import importlib
    mod = importlib.import_module(module_path)
    cfg_cls = getattr(mod, cls_name)
    return cfg_cls()


def main():
    env_cfg = _load_env_cfg(args_cli.task)
    agent_cfg = _load_agent_cfg()

    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.seed
    agent_cfg["params"]["seed"] = args_cli.seed
    agent_cfg["params"]["config"]["torch_compile"] = False

    checkpoint_path = os.path.abspath(args_cli.checkpoint)
    checkpoint_dir = os.path.dirname(os.path.dirname(checkpoint_path))

    render_mode = "rgb_array" if args_cli.video else None
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=render_mode)

    if args_cli.video:
        video_dir = args_cli.video_dir or os.path.join(checkpoint_dir, "videos")
        os.makedirs(video_dir, exist_ok=True)
        video_kwargs = {
            "video_folder": video_dir,
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print(f"[INFO] Recording video to: {video_dir}")
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    rl_device = agent_cfg["params"]["config"]["device"]
    clip_obs = agent_cfg["params"]["env"].get("clip_observations", math.inf)
    clip_actions = agent_cfg["params"]["env"].get("clip_actions", math.inf)

    env = RlGamesVecEnvWrapper(env, rl_device, clip_obs, clip_actions)

    vecenv.register(
        "IsaacRlgWrapper",
        lambda config_name, num_actors, **kwargs: RlGamesGpuEnv(config_name, num_actors, **kwargs),
    )
    env_configurations.register("rlgpu", {"vecenv_type": "IsaacRlgWrapper", "env_creator": lambda **kwargs: env})

    agent_cfg["params"]["load_checkpoint"] = True
    agent_cfg["params"]["load_path"] = checkpoint_path
    agent_cfg["params"]["config"]["num_actors"] = env.unwrapped.num_envs
    agent_cfg["params"]["config"]["player"] = {
        "deterministic": not args_cli.stochastic,
        "games_num": 100000,
        "print_stats": False,
    }

    print(f"[INFO] Loading checkpoint: {checkpoint_path}")
    print(f"[INFO] Task: {args_cli.task}  |  Envs: {args_cli.num_envs}  |  Deterministic: {not args_cli.stochastic}")

    runner = Runner()
    runner.load(agent_cfg)
    agent = runner.create_player()
    agent.restore(checkpoint_path)
    agent.reset()

    raw_env = env.unwrapped
    num_envs = raw_env.num_envs

    obs = env.reset()
    if isinstance(obs, dict):
        obs = obs["obs"]
    _ = agent.get_batch_size(obs, 1)
    if agent.is_rnn:
        agent.init_rnn()

    episode_nsucc: list[float] = []
    step_count = 0
    # When video mode: cap at video_length steps but also allow early exit
    # at num_episodes. When not video: allow up to 100k steps.
    max_steps = (args_cli.video_length + 500) if args_cli.video else 100_000

    print(f"[INFO] Collecting {args_cli.num_episodes} episodes (max {max_steps} steps) ...")

    while simulation_app.is_running() and step_count < max_steps:
        with torch.inference_mode():
            obs_t = agent.obs_to_torch(obs)
            actions = agent.get_action(obs_t, is_deterministic=not args_cli.stochastic)
            obs, _, dones, extras = env.step(actions)

        step_count += 1

        done_mask = dones.bool()
        if done_mask.any():
            done_indices = done_mask.nonzero(as_tuple=False).squeeze(-1)
            nsucc_vals = raw_env.prev_episode_successes[done_indices].cpu().numpy()
            episode_nsucc.extend(nsucc_vals.tolist())

        if len(episode_nsucc) >= args_cli.num_episodes:
            break

        if agent.is_rnn and agent.states is not None and len(dones) > 0:
            new_states = []
            for s in agent.states:
                s_new = s.clone()
                s_new[:, dones.bool(), :] = 0.0
                new_states.append(s_new)
            agent.states = tuple(new_states)

        if step_count % 500 == 0:
            print(f"  step {step_count}: {len(episode_nsucc)} episodes collected")

    success_tol = float(raw_env.success_tolerance)

    if episode_nsucc:
        arr = np.array(episode_nsucc)
        results = {
            "task": args_cli.task,
            "checkpoint": checkpoint_path,
            "num_envs": num_envs,
            "num_episodes": len(arr),
            "steps_run": step_count,
            "deterministic": not args_cli.stochastic,
            "success_tolerance": success_tol,
            "nsucc_mean": float(arr.mean()),
            "nsucc_std": float(arr.std()),
            "nsucc_median": float(np.median(arr)),
            "nsucc_min": float(arr.min()),
            "nsucc_max": float(arr.max()),
        }
    else:
        results = {
            "task": args_cli.task,
            "checkpoint": checkpoint_path,
            "num_envs": num_envs,
            "num_episodes": 0,
            "steps_run": step_count,
            "deterministic": not args_cli.stochastic,
            "success_tolerance": success_tol,
            "nsucc_mean": None,
            "nsucc_std": None,
            "nsucc_median": None,
            "nsucc_min": None,
            "nsucc_max": None,
        }

    print("\n" + "=" * 60)
    print("  EVALUATION RESULTS")
    print("=" * 60)
    print(f"  Task:              {results['task']}")
    print(f"  Checkpoint:        {Path(results['checkpoint']).name}")
    print(f"  Episodes:          {results['num_episodes']}")
    print(f"  Steps run:         {results['steps_run']}")
    print(f"  Deterministic:     {results['deterministic']}")
    print(f"  Success tolerance: {results['success_tolerance']:.4f}")
    if results["nsucc_mean"] is not None:
        print(f"  Nsucc mean:        {results['nsucc_mean']:.2f}")
        print(f"  Nsucc std:         {results['nsucc_std']:.2f}")
        print(f"  Nsucc median:      {results['nsucc_median']:.1f}")
        print(f"  Nsucc min:         {results['nsucc_min']:.0f}")
        print(f"  Nsucc max:         {results['nsucc_max']:.0f}")
    else:
        print("  (no episodes completed)")
    print("=" * 60 + "\n")

    output_json = args_cli.output_json
    if output_json is None:
        output_json = os.path.join(checkpoint_dir, "eval_results.json")
    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    with open(output_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[INFO] Results saved to: {output_json}")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
