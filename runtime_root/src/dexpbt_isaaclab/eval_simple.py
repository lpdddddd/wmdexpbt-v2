#!/usr/bin/env python3
"""
Simplified evaluation script - skips WM patching and just loads the checkpoint.
"""

from __future__ import annotations
import argparse
import json
import os
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# AppLauncher must come first
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Evaluate DexPBT checkpoint.")
parser.add_argument("--task", type=str, required=True)
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--num_envs", type=int, default=512)
parser.add_argument("--num_episodes", type=int, default=512)
parser.add_argument("--output_json", type=str, required=True)
parser.add_argument("--seed", type=int, default=42)

from isaaclab.app import AppLauncher
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ---------------------------------------------------------------------------
# Runtime imports
# ---------------------------------------------------------------------------
import gymnasium as gym
import numpy as np
import torch
import yaml
from rl_games.common import env_configurations, vecenv
from rl_games.torch_runner import Runner

from isaaclab_rl.rl_games import RlGamesGpuEnv, RlGamesVecEnvWrapper
import dexpbt_isaaclab


def _load_agent_cfg() -> dict:
    import importlib.resources as resources
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

    print(f"[INFO] Creating environment...")
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)

    rl_device = agent_cfg["params"]["config"]["device"]
    clip_obs = agent_cfg["params"]["env"].get("clip_observations", float("inf"))
    clip_actions = agent_cfg["params"]["env"].get("clip_actions", float("inf"))

    env = RlGamesVecEnvWrapper(env, rl_device, clip_obs, clip_actions)

    vecenv.register(
        "IsaacRlgWrapper",
        lambda config_name, num_actors, **kw: RlGamesGpuEnv(config_name, num_actors, **kw),
    )
    env_configurations.register(
        "rlgpu",
        {"vecenv_type": "IsaacRlgWrapper", "env_creator": lambda **kw: env},
    )

    agent_cfg["params"]["load_checkpoint"] = True
    agent_cfg["params"]["load_path"] = checkpoint_path
    agent_cfg["params"]["config"]["num_actors"] = env.unwrapped.num_envs
    agent_cfg["params"]["config"]["player"] = {
        "deterministic": True,
        "games_num": 100000,
        "print_stats": False,
    }

    print(f"[INFO] Loading checkpoint: {checkpoint_path}")

    runner = Runner()
    runner.load(agent_cfg)
    agent = runner.create_player()
    agent.reset()

    # Try to restore checkpoint, handling aux_value_head issue
    try:
        agent.restore(checkpoint_path)
    except RuntimeError as e:
        if "aux_value_head" in str(e):
            print(f"[WARN] aux_value_head mismatch, trying strict=False restore...")
            # Load state dict manually, skip aux_value_head keys
            checkpoint = torch.load(checkpoint_path, map_location=rl_device)
            if 'model' in checkpoint:
                state_dict = checkpoint['model']
                # Remove aux_value_head keys if present
                filtered_state_dict = {k: v for k, v in state_dict.items() if not k.startswith('aux_value_head')}
                agent.model.load_state_dict(filtered_state_dict, strict=False)
                print(f"[INFO] Loaded checkpoint (skipped aux_value_head keys)")
            else:
                raise e
        else:
            raise e

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
    max_steps = 2_000_000

    print(f"[INFO] Collecting >= {args_cli.num_episodes} episodes...")

    while simulation_app.is_running() and step_count < max_steps:
        with torch.no_grad():
            obs_t = agent.obs_to_torch(obs)
            actions = agent.get_action(obs_t, is_deterministic=True)
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
            for s in agent.states:
                s[:, dones.bool(), :] = 0.0

        if step_count % 500 == 0:
            print(f"  step {step_count}: {len(episode_nsucc)} episodes")

    success_tol = float(raw_env.success_tolerance)

    if episode_nsucc:
        arr = np.array(episode_nsucc)
        results = {
            "task": args_cli.task,
            "checkpoint": checkpoint_path,
            "num_envs": num_envs,
            "num_episodes": len(arr),
            "steps_run": step_count,
            "deterministic": True,
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
            "deterministic": True,
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
    print(f"  Checkpoint:        {Path(results['checkpoint']).name}")
    print(f"  Episodes:          {results['num_episodes']}")
    print(f"  Success tolerance: {results['success_tolerance']:.4f}")
    if results["nsucc_mean"] is not None:
        print(f"  Nsucc mean:        {results['nsucc_mean']:.2f}")
        print(f"  Nsucc std:         {results['nsucc_std']:.2f}")
        print(f"  Nsucc median:      {results['nsucc_median']:.1f}")
    print("=" * 60 + "\n")

    output_json = Path(args_cli.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[INFO] Results saved to: {output_json}")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()