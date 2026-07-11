"""Subprocess evaluation script called by eval_all_checkpoints.py.

This script is run via `python -m dexpbt_isaaclab.scripts._eval_single_ckpt`
in the isaaclab311 conda environment. It loads a checkpoint, runs deterministic
evaluation, collects Nsucc statistics, and optionally records videos.

Usage:
    python -m dexpbt_isaaclab.scripts._eval_single_ckpt \
        --task DexPBT-Regrasping-KukaAllegro-Direct-v0 \
        --checkpoint /path/to/checkpoint.pth \
        --num_envs 512 \
        --num_episodes 512 \
        --output_json /path/to/results.json \
        --seed 42 \
        [--video --video_length 300 --video_dir /path/to/videos]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# AppLauncher must come first; all Omniverse/Isaac Lab imports follow
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Evaluate a single DexPBT checkpoint.")
parser.add_argument("--task", type=str, required=True)
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--num_envs", type=int, default=512)
parser.add_argument("--num_episodes", type=int, default=512)
parser.add_argument("--output_json", type=str, required=True)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--video", action="store_true", default=False)
parser.add_argument("--video_length", type=int, default=300)
parser.add_argument("--video_dir", type=str, default=None)

from isaaclab.app import AppLauncher  # noqa: E402
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

if args_cli.video:
    args_cli.enable_cameras = True

sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ---------------------------------------------------------------------------
# Force-enable offscreen rendering for video (rgb_array requires PARTIAL_RENDERING)
# The AppLauncher chain should set this, but we double-check / override here.
# ---------------------------------------------------------------------------
if args_cli.video:
    import carb
    carb.settings.get_settings().set_bool("/isaaclab/render/offscreen", True)

# ---------------------------------------------------------------------------
# Runtime imports
# ---------------------------------------------------------------------------
import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402
from rl_games.common import env_configurations, vecenv  # noqa: E402
from rl_games.torch_runner import Runner  # noqa: E402

from isaaclab_rl.rl_games import RlGamesGpuEnv, RlGamesVecEnvWrapper  # noqa: E402
import dexpbt_isaaclab  # noqa: F401


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
        print(f"[VIDEO] Recording to: {video_dir}")
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

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
        "deterministic": True,  # paper uses deterministic evaluation
        "games_num": 100000,
        "print_stats": False,
    }

    print(f"[INFO] Loading: {checkpoint_path}")
    print(f"[INFO] Task={args_cli.task}  |  Envs={args_cli.num_envs}  |  Deterministic=True")

    runner = Runner()
    runner.load(agent_cfg)
    agent = runner.create_player()

    # ---------------------------------------------------------------------
    # WM checkpoint compatibility: if the checkpoint contains aux_value_head
    # keys (added by value_imag_h1 strategy), inject a matching layer into
    # the agent's model before restore().
    # ---------------------------------------------------------------------
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model_state = ckpt.get("model", ckpt)
    has_aux = any(k.startswith("aux_value_head.") for k in model_state.keys())

    if has_aux:
        print("[INFO] Checkpoint has aux_value_head — injecting placeholder")
        am = agent.model
        if not hasattr(am, "aux_value_head"):
            shared_dim = 256  # safe default; probing skipped for simplicity
            am.aux_value_head = torch.nn.Sequential(
                torch.nn.Linear(shared_dim, 128),
                torch.nn.Tanh(),
                torch.nn.Linear(128, 1),
            )
            print(f"[INFO] aux_value_head injected (shared_dim={shared_dim})")

        # Monkey-patch load_state_dict to use strict=False so aux_value_head
        # extra keys don't cause a RuntimeError.
        orig_load = am.load_state_dict

        def _load_strict_false(state_dict, strict=True):
            return orig_load(state_dict, strict=False)

        am.load_state_dict = _load_strict_false

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
    max_steps = args_cli.video_length if args_cli.video else 2_000_000

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

        if len(episode_nsucc) >= args_cli.num_episodes and not args_cli.video:
            break

        # Reset RNN states for done envs
        if agent.is_rnn and agent.states is not None and len(dones) > 0:
            for s in agent.states:
                s[:, dones.bool(), :] = 0.0

        if step_count % 500 == 0 and not args_cli.video:
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
    print("  RESULTS")
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
    print(f"[INFO] Saved to: {output_json}")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
