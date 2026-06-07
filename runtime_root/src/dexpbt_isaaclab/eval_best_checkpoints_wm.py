#!/usr/bin/env python3
"""
Evaluate DexPBT checkpoints that were trained with World Model (WM) strategy.
Handles aux_value_head weights properly.

Usage:
    python eval_best_checkpoints_wm.py --checkpoint /path/to/checkpoint.pth --output_json /path/to/results.json --num_envs 512 --num_episodes 512
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
parser = argparse.ArgumentParser(description="Evaluate DexPBT WM checkpoint.")
parser.add_argument("--task", type=str, required=True)
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--num_envs", type=int, default=512)
parser.add_argument("--num_episodes", type=int, default=512)
parser.add_argument("--output_json", type=str, required=True)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--wm_strategy", type=str, default="value_imag_h1")

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

# Import WM related modules
from dexpbt_isaaclab.world_model import get_strategy, WMTrainer


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


def _patch_network_for_wm(model, wm_trainer):
    """Monkey-patch the network to include WM aux_value_head."""
    # Add aux_value_head to model
    wm_trainer._add_aux_value_head()

    # Store reference to wm_trainer in model
    model.wm_trainer = wm_trainer

    # Patch get_aux_loss to return WM loss
    original_get_aux_loss = model.get_aux_loss

    def patched_get_aux_loss(*args, **kwargs):
        if hasattr(model, 'wm_trainer') and model.wm_trainer is not None:
            wm_loss_dict = model.wm_trainer.compute_value_imag_loss_dict()
            if wm_loss_dict:
                return wm_loss_dict.get('value_imag_loss', torch.tensor(0.0, device='cuda'))
        return torch.tensor(0.0, device='cuda') if original_get_aux_loss is None else original_get_aux_loss(*args, **kwargs)

    model.get_aux_loss = patched_get_aux_loss


def main():
    env_cfg = _load_env_cfg(args_cli.task)
    agent_cfg = _load_agent_cfg()

    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.seed
    agent_cfg["params"]["seed"] = args_cli.seed
    agent_cfg["params"]["config"]["torch_compile"] = False

    checkpoint_path = os.path.abspath(args_cli.checkpoint)
    checkpoint_dir = os.path.dirname(os.path.dirname(checkpoint_path))

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
    print(f"[INFO] WM Strategy: {args_cli.wm_strategy}")

    # Initialize WM trainer
    strategy = get_strategy(args_cli.wm_strategy)
    wm_trainer = WMTrainer(
        strategy=strategy,
        observer=env.unwrapped,
        device=rl_device,
    )

    # Load checkpoint first to get the model
    runner = Runner()
    runner.load(agent_cfg)
    agent = runner.create_player()
    agent.reset()

    # Patch network for WM (adds aux_value_head before restore)
    if strategy.use_value_imag:
        _patch_network_for_wm(agent.model, wm_trainer)

    # Restore checkpoint
    agent.restore(checkpoint_path)

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

    print(f"[INFO] Collecting ≥{args_cli.num_episodes} episodes (max {max_steps} steps) ...")

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
    print(f"  Task:              {results['task']}")
    print(f"  Checkpoint:        {Path(results['checkpoint']).name}")
    print(f"  Episodes:          {results['num_episodes']}")
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

    output_json = Path(args_cli.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[INFO] Results saved to: {output_json}")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()