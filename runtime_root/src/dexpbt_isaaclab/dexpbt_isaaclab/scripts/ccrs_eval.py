#!/usr/bin/env python3
"""CCRS single-unit evaluator: one checkpoint x one fixed tolerance x one seed.

Runs a frozen policy (rl_games player, deterministic by default) in the task
env with success_tolerance FORCED to a fixed value for the entire rollout.
No optimizer step, no PBT, no curriculum update, no mutation can occur:
this process never constructs a trainer/optimizer — only a player — and the
env-side curriculum trigger is disabled exactly as in eval_fair.py.

policy_only mode (default, main CCRS results):
  - pure rl_games player action = actor forward pass only
  - no snippet retrieval, no WM rerank, no MPPI, no residual action,
    no actor distillation (those subsystems live in the training observer
    stack and are simply never instantiated here; asserted at runtime).

system_on mode: reserved, currently rejected at startup (do not mix with
policy-only results).

One unit per process by design: Isaac Sim is not reliable across env.close()
re-creation in one process (empirically hangs; see 2026-07-15 eval notes).
Resume/skip: if the per-unit raw JSON already exists and is valid, the
process exits 0 immediately (checked BEFORE Isaac boot).

Usage:
  python -m dexpbt_isaaclab.scripts.ccrs_eval \
    --task DexPBT-Regrasping-KukaAllegro-Direct-v0 \
    --run-id regrasping_..._5b_20260716_130631 \
    --worker-id 1 --checkpoint-iteration 149 --checkpoint-frame 4120000000 \
    --checkpoint-path /path/to/dexpbt_8gpu_ws/001/000149.pth \
    --fixed-tolerance 0.020 --reset-seed 10001 \
    --num-envs 512 --num-episodes 512 \
    --raw-dir results/analysis/ccrs_base_retention/raw \
    --headless
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

# --- protocol import must not require Isaac ---
from dexpbt_isaaclab.ccrs.eval_protocol import (
    EVALUATION_MODE_POLICY_ONLY,
    EVALUATION_MODE_SYSTEM_ON,
    EvalUnit,
    is_done,
    make_result_row,
    result_path,
)

parser = argparse.ArgumentParser(description="CCRS single-unit fixed-tolerance eval.")
parser.add_argument("--task", type=str, required=True)
parser.add_argument("--run-id", type=str, required=True)
parser.add_argument("--worker-id", type=int, required=True)
parser.add_argument("--checkpoint-iteration", type=int, required=True)
parser.add_argument("--checkpoint-frame", type=int, required=True)
parser.add_argument("--checkpoint-path", type=str, required=True)
parser.add_argument("--fixed-tolerance", type=float, required=True,
                    help="Fixed success tolerance in METERS (e.g. 0.020 = 2 cm)")
parser.add_argument("--reset-seed", type=int, required=True)
parser.add_argument("--num-envs", type=int, default=512)
parser.add_argument("--num-episodes", type=int, default=512)
parser.add_argument("--deterministic", type=str, default="true", choices=["true", "false"])
parser.add_argument("--max-steps", type=int, default=200_000)
parser.add_argument("--max-consecutive-successes", type=int, default=50)
parser.add_argument("--evaluation-mode", type=str, default=EVALUATION_MODE_POLICY_ONLY,
                    choices=[EVALUATION_MODE_POLICY_ONLY, EVALUATION_MODE_SYSTEM_ON])
parser.add_argument("--raw-dir", type=str, required=True)
parser.add_argument("--force", action="store_true", help="Re-run even if unit is done")


def _build_unit(a: argparse.Namespace) -> EvalUnit:
    return EvalUnit(
        run_id=a.run_id,
        worker_id=a.worker_id,
        checkpoint_iteration=a.checkpoint_iteration,
        checkpoint_frame=a.checkpoint_frame,
        checkpoint_path=a.checkpoint_path,
        fixed_tolerance=a.fixed_tolerance,
        reset_seed=a.reset_seed,
        num_envs=a.num_envs,
        num_episodes=a.num_episodes,
        evaluation_mode=a.evaluation_mode,
        deterministic=(a.deterministic == "true"),
        max_steps=a.max_steps,
        max_consecutive_successes=a.max_consecutive_successes,
    )


def _git_sha() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(Path(__file__).resolve().parent),
            capture_output=True, text=True, check=True,
        ).stdout.strip()
    except Exception:
        return "unknown"


def _ckpt_hash(path: Path, head_bytes: int = 4 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    h.update(str(path.stat().st_size).encode())
    with path.open("rb") as f:
        h.update(f.read(head_bytes))
    return h.hexdigest()[:16]


# ---------------------------------------------------------------------------
# Pre-Isaac validation and resume check (cheap exit paths)
# ---------------------------------------------------------------------------
_pre_args, _hydra_args = parser.parse_known_args()

if _pre_args.evaluation_mode == EVALUATION_MODE_SYSTEM_ON:
    print("[ccrs_eval] system_on mode is reserved and not implemented; refusing "
          "to run so it cannot be mixed with policy_only results.", file=sys.stderr)
    sys.exit(3)

if not (0.001 <= _pre_args.fixed_tolerance <= 0.2):
    print(f"[ccrs_eval] fixed tolerance {_pre_args.fixed_tolerance} out of meter "
          "range [0.001, 0.2]; pass meters (0.020 = 2 cm)", file=sys.stderr)
    sys.exit(3)

_unit = _build_unit(_pre_args)
_raw_dir = Path(_pre_args.raw_dir).resolve()
_raw_dir.mkdir(parents=True, exist_ok=True)
_out_path = result_path(_raw_dir, _unit)

if not _pre_args.force and is_done(_raw_dir, _unit):
    print(f"[ccrs_eval] unit already done, skipping: {_out_path.name}")
    sys.exit(0)

_ckpt = Path(_pre_args.checkpoint_path)
if not _ckpt.is_file():
    print(f"[ccrs_eval] checkpoint not found: {_ckpt}", file=sys.stderr)
    sys.exit(2)

# ---------------------------------------------------------------------------
# Isaac boot (only after all cheap checks pass)
# ---------------------------------------------------------------------------
from isaaclab.app import AppLauncher  # noqa: E402

AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402
from rl_games.common import env_configurations, vecenv  # noqa: E402
from rl_games.torch_runner import Runner  # noqa: E402

from isaaclab_rl.rl_games import RlGamesGpuEnv, RlGamesVecEnvWrapper  # noqa: E402
import dexpbt_isaaclab  # noqa: F401, E402  (register envs)

# Reuse the proven loaders from eval_fair instead of duplicating them.
from dexpbt_isaaclab.scripts.eval_fair import (  # noqa: E402
    _filtered_restore,
    _load_agent_cfg,
    _load_env_cfg,
)


def run_unit(unit: EvalUnit) -> dict:
    env_cfg = _load_env_cfg(args_cli.task)
    agent_cfg = _load_agent_cfg()

    env_cfg.scene.num_envs = unit.num_envs
    env_cfg.seed = unit.reset_seed
    env_cfg.max_consecutive_successes = int(unit.max_consecutive_successes)
    agent_cfg["params"]["seed"] = unit.reset_seed
    agent_cfg["params"]["config"]["torch_compile"] = False

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)

    # Freeze tolerance BEFORE any rollout; disable env-side curriculum trigger.
    raw_env = env.unwrapped
    assert hasattr(raw_env, "success_tolerance"), "env has no success_tolerance"
    raw_env.success_tolerance = float(unit.fixed_tolerance)
    raw_env.initial_tolerance = float(unit.fixed_tolerance)
    raw_env.target_tolerance = float(unit.fixed_tolerance)
    raw_env.last_curriculum_update = 10 ** 9  # curriculum can never fire
    tol_at_start = float(raw_env.success_tolerance)

    rl_device = agent_cfg["params"]["config"]["device"]
    clip_obs = agent_cfg["params"]["env"].get("clip_observations", float("inf"))
    clip_actions = agent_cfg["params"]["env"].get("clip_actions", float("inf"))
    env = RlGamesVecEnvWrapper(env, rl_device, clip_obs, clip_actions)

    vecenv.register(
        "IsaacRlgWrapper",
        lambda config_name, num_actors, **kw: RlGamesGpuEnv(config_name, num_actors, **kw),
    )
    env_configurations.register(
        "rlgpu", {"vecenv_type": "IsaacRlgWrapper", "env_creator": lambda **kw: env}
    )

    agent_cfg["params"]["load_checkpoint"] = True
    agent_cfg["params"]["load_path"] = str(unit.checkpoint_path)
    agent_cfg["params"]["config"]["num_actors"] = env.unwrapped.num_envs
    agent_cfg["params"]["config"]["player"] = {
        "deterministic": unit.deterministic,
        "games_num": 10 ** 9,
        "print_stats": False,
    }

    runner = Runner()
    runner.load(agent_cfg)
    agent = runner.create_player()  # player only: no optimizer exists
    agent.reset()
    _filtered_restore(agent, Path(unit.checkpoint_path), rl_device)

    # policy_only guarantees (test-checkable invariants):
    assert not hasattr(agent, "optimizer"), "player must not own an optimizer"
    agent.model.eval()
    for p in agent.model.parameters():
        p.requires_grad_(False)

    raw_env = env.unwrapped
    obs = env.reset()
    if isinstance(obs, dict):
        obs = obs["obs"]
    _ = agent.get_batch_size(obs, 1)
    if agent.is_rnn:
        agent.init_rnn()

    episode_nsucc: list[float] = []
    episode_final_error: list[float] = []
    step = 0
    while simulation_app.is_running() and step < unit.max_steps:
        with torch.no_grad():
            obs_t = agent.obs_to_torch(obs)
            actions = agent.get_action(obs_t, is_deterministic=unit.deterministic)
            obs, reward, dones, extras = env.step(actions)
        step += 1
        dm = dones.bool()
        if dm.any():
            idx = dm.nonzero(as_tuple=False).squeeze(-1)
            ns_vals = raw_env.prev_episode_successes[idx].detach().cpu().numpy()
            episode_nsucc.extend(ns_vals.tolist())
            if hasattr(raw_env, "object_pos") and hasattr(raw_env, "goal_pos"):
                err = (raw_env.object_pos[idx] - raw_env.goal_pos[idx]).norm(dim=-1)
                episode_final_error.extend(err.detach().cpu().numpy().tolist())
            else:
                episode_final_error.extend([float("nan")] * len(idx))
        if agent.is_rnn and agent.states is not None and len(dones) > 0:
            for s in agent.states:
                s[:, dones.bool(), :] = 0.0
        if len(episode_nsucc) >= unit.num_episodes:
            break

    tol_at_end = float(raw_env.success_tolerance)
    env.close()

    return make_result_row(
        unit=unit,
        episode_nsucc=episode_nsucc[: unit.num_episodes],
        episode_final_error=episode_final_error[: unit.num_episodes],
        git_sha=_git_sha(),
        checkpoint_hash=_ckpt_hash(Path(unit.checkpoint_path)),
        keep_per_episode=True,
        applied_tolerance_start=tol_at_start,
        applied_tolerance_end=tol_at_end,
        extra={"steps_used": step},
    )


def main() -> None:
    row = run_unit(_unit)
    tmp = _out_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(row))
    tmp.replace(_out_path)
    brief = {k: row[k] for k in (
        "worker_id", "checkpoint_iteration", "fixed_tolerance", "reset_seed",
        "num_episodes", "success_rate", "nsucc_mean",
    )}
    print(f"[ccrs_eval] done: {json.dumps(brief)}")
    print(f"[ccrs_eval] wrote {_out_path}")


if __name__ == "__main__":
    main()
    simulation_app.close()
