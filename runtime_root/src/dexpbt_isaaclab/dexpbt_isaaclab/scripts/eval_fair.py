#!/usr/bin/env python3
"""Unified fair-comparison evaluation for DexPBT baseline and CB-WM-DexPBT.

Goals (compare with eval_simple / eval_best_checkpoints_wm / eval_all_checkpoints):
1. SAME deterministic protocol, SAME num_envs / num_episodes for baseline & ours.
2. Explicit `eval_tolerance_mode`:
   - `checkpoint`: use the success_tolerance recorded in the workspace meta
     at the time the checkpoint was saved (faithful to training).
   - `fixed`: force success_tolerance = --fixed_success_tolerance (e.g. 0.01
     to match the paper's eps*).
3. Explicit `checkpoint_rule`:
   - `best_r_meta`: workspace meta with maximum `objective` field
     (this is the only metric pbt_observer uses for replacement decisions).
   - `last`: latest PBT tick.
   - `best_train_reward`: rl_games filename `_rew_XXX.pth` heuristic
     (deprecated; kept for parity with old reports).
4. Same script handles baseline (no WM keys in checkpoint) and ours
   (aux_value_head extra keys); we restore with strict=False on filtered dict.

Usage (typical):
  python -m dexpbt_isaaclab.scripts.eval_fair \
    --workspace /path/to/run/dexpbt_8gpu_ws \
    --method_name baseline \
    --task DexPBT-Regrasping-KukaAllegro-Direct-v0 \
    --checkpoint_rule best_r_meta \
    --eval_tolerance_mode fixed --fixed_success_tolerance 0.01 \
    --num_envs 512 --num_episodes 512 \
    --output_csv /tmp/baseline_fair.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Optional

# AppLauncher / argparse plumbing identical to other scripts: must register
# launcher args first, then app must be created BEFORE any Isaac imports.
parser = argparse.ArgumentParser(description="Fair DexPBT evaluation.")
parser.add_argument("--workspace", type=str, required=True,
                    help="PBT workspace dir containing 000/, 001/, ... and metrics/")
parser.add_argument("--method_name", type=str, required=True,
                    help="Free-form tag stored in CSV (e.g. baseline, cb_wm_dexpbt)")
parser.add_argument("--task", type=str, required=True)
parser.add_argument("--checkpoint_rule", type=str,
                    choices=["best_r_meta", "last", "best_train_reward"],
                    default="best_r_meta")
parser.add_argument("--eval_tolerance_mode", type=str,
                    choices=["checkpoint", "fixed"], default="fixed")
parser.add_argument("--fixed_success_tolerance", type=float, default=0.01)
parser.add_argument("--num_envs", type=int, default=512)
parser.add_argument("--num_episodes", type=int, default=512)
parser.add_argument("--deterministic", type=str, default="true",
                    choices=["true", "false"])
parser.add_argument("--policy_indices", type=str, default="all",
                    help="'all' or comma-separated like '0,1,3'")
parser.add_argument("--output_csv", type=str, required=True)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--max_steps", type=int, default=200_000,
                    help="Per-policy hard cap on env steps")
parser.add_argument("--checkpoint_override", type=str, default=None,
                    help="If set, ignore workspace lookup and eval this single .pth")
parser.add_argument(
    "--max_consecutive_successes_override",
    type=int,
    default=None,
    help=(
        "Eval-only override for env_cfg.max_consecutive_successes. "
        "Leave unset to use the env default (50). Only affects the eval "
        "rollout in this process; training defaults are not modified."
    ),
)
# Track A pinned-checkpoint extension (2026-06-06): per-policy PBT-tick
# override map. Format: `0=108,1=137,...`. Iterations refer to the
# {iter:06d}.json meta filename under {workspace}/{idx:03d}/. When set
# for a given policy, --checkpoint_rule is ignored for that policy.
parser.add_argument(
    "--policy_checkpoint_iterations",
    type=str,
    default=None,
    help=(
        "Optional per-policy PBT-tick override, e.g. "
        "'0=108,1=137,2=131,3=135,4=83,5=114'. Falls back to "
        "--checkpoint_rule for any policy not listed."
    ),
)
parser.add_argument(
    "--policy_expected_frames",
    type=str,
    default=None,
    help=(
        "Optional per-policy expected env_frames map, e.g. "
        "'0=2160000000,1=2740000000,...'. When --policy_checkpoint_iterations "
        "is in use, the loaded meta's `frame` field is compared to this "
        "expected value; a WARNING is printed when |delta| exceeds "
        "10M frames. Mismatch only warns, does not abort eval."
    ),
)

from isaaclab.app import AppLauncher
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import numpy as np
import torch
import yaml
from rl_games.common import env_configurations, vecenv
from rl_games.torch_runner import Runner

from isaaclab_rl.rl_games import RlGamesGpuEnv, RlGamesVecEnvWrapper
import dexpbt_isaaclab  # noqa: F401  (register envs)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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
    return getattr(mod, cls_name)()


def _load_workspace_metas(workspace: Path, policy_idx: int) -> list[dict]:
    """Load all .json meta files for one policy under the PBT workspace.

    pbt_observer writes one meta per PBT tick at `{workspace}/{idx:03d}/{iter:06d}.json`.
    Each meta has fields: objective, frame, iteration, params, checkpoint,
    curriculum (after M1).
    """
    policy_dir = workspace / f"{policy_idx:03d}"
    metas: list[dict] = []
    if not policy_dir.is_dir():
        return metas
    for p in sorted(policy_dir.glob("*.json")):
        try:
            metas.append(json.loads(p.read_text()))
        except Exception:
            continue
    return metas


def _pick_checkpoint(
    workspace: Path,
    policy_idx: int,
    rule: str,
) -> tuple[Optional[Path], Optional[dict]]:
    """Returns (checkpoint_path, meta_dict_or_None)."""
    metas = _load_workspace_metas(workspace, policy_idx)
    if not metas:
        # Fall back to best_train_reward over rl_games attempts
        if rule == "best_train_reward":
            return _find_best_train_reward(workspace, policy_idx), None
        return None, None

    if rule == "best_r_meta":
        best = max(metas, key=lambda m: float(m.get("objective", float("-inf"))))
        return Path(best["checkpoint"]), best
    if rule == "last":
        last = max(metas, key=lambda m: int(m.get("iteration", -1)))
        return Path(last["checkpoint"]), last
    if rule == "best_train_reward":
        return _find_best_train_reward(workspace, policy_idx), None
    return None, None


def _parse_idx_int_map(spec: Optional[str]) -> dict[int, int]:
    """Parse `0=108,1=137,...` into {0: 108, 1: 137, ...}; empty/None -> {}."""
    if not spec:
        return {}
    out: dict[int, int] = {}
    for tok in str(spec).split(","):
        tok = tok.strip()
        if not tok:
            continue
        if "=" not in tok:
            raise ValueError(
                f"invalid idx=value entry {tok!r}; expected e.g. 0=108"
            )
        k, v = tok.split("=", 1)
        out[int(k.strip())] = int(v.strip())
    return out


def _pick_pinned_checkpoint(
    workspace: Path,
    policy_idx: int,
    iteration: int,
) -> tuple[Optional[Path], Optional[dict]]:
    """Load the meta + checkpoint for an explicit (policy_idx, PBT-tick) pin.

    Returns (None, None) if the meta JSON is missing or unparseable so the
    main loop can decide whether to skip / fall back.
    """
    policy_dir = workspace / f"{policy_idx:03d}"
    meta_path = policy_dir / f"{int(iteration):06d}.json"
    if not meta_path.is_file():
        print(
            f"[fair_eval] WARN: pinned meta not found for policy_idx={policy_idx} "
            f"iteration={iteration}: {meta_path}"
        )
        return None, None
    try:
        meta = json.loads(meta_path.read_text())
    except Exception as e:
        print(
            f"[fair_eval] WARN: failed to parse pinned meta {meta_path}: {e}"
        )
        return None, None
    ckpt_str = meta.get("checkpoint")
    if not ckpt_str:
        print(
            f"[fair_eval] WARN: pinned meta {meta_path} has no `checkpoint` field"
        )
        return None, meta
    return Path(ckpt_str), meta


def _find_best_train_reward(workspace: Path, policy_idx: int) -> Optional[Path]:
    """Legacy heuristic: highest `_rew_` number in rl_games filenames."""
    # supervisor RUN_DIR is workspace.parent; per-policy dir is `policy_{idx:02d}`
    run_dir = workspace.parent
    policy_dir = run_dir / f"policy_{policy_idx:02d}"
    best: Optional[Path] = None
    best_rew = float("-inf")
    for pth in policy_dir.rglob("last_DexPBTKukaAllegroLSTMPPO_ep_*.pth"):
        name = pth.stem
        try:
            rew = float(name.split("_rew_")[1].strip("_"))
        except (IndexError, ValueError):
            continue
        if rew > best_rew:
            best_rew = rew
            best = pth
    return best


def _resolve_eval_tolerance(
    mode: str,
    fixed: float,
    meta: Optional[dict],
) -> float:
    if mode == "fixed":
        return float(fixed)
    if mode == "checkpoint":
        if meta is None:
            print("[fair_eval] WARN: --eval_tolerance_mode=checkpoint but no meta found; falling back to fixed")
            return float(fixed)
        curr = meta.get("curriculum") or {}
        st = curr.get("success_tolerance")
        if st is None:
            print("[fair_eval] WARN: meta has no curriculum.success_tolerance; falling back to fixed")
            return float(fixed)
        return float(st)
    return float(fixed)


def _checkpoint_has_running_norm(checkpoint_path: Path) -> bool:
    """True if checkpoint state dict contains running-normalization keys."""
    try:
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except Exception:
        return False
    if isinstance(ckpt, dict):
        state = ckpt.get("model") or ckpt.get("state_dict") or ckpt
        if isinstance(state, dict):
            for key in state.keys():
                if "running_mean" in key or "running_std" in key or "running_var" in key:
                    return True
    return False


def _filtered_restore(agent, checkpoint_path: Path, device: str) -> None:
    """Restore an rl_games agent, tolerating aux_value_head keys from WM runs."""
    try:
        agent.restore(str(checkpoint_path))
        return
    except RuntimeError as e:
        if "aux_value_head" not in str(e):
            raise
    print("[fair_eval] aux_value_head detected; loading with strict=False filter")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model_sd = ckpt.get("model", ckpt)
    filtered = {k: v for k, v in model_sd.items() if not k.startswith("aux_value_head")}
    agent.model.load_state_dict(filtered, strict=False)


def _run_one_eval(
    task: str,
    checkpoint_path: Path,
    eval_tolerance: float,
    num_envs: int,
    num_episodes: int,
    deterministic: bool,
    seed: int,
    max_steps: int,
    max_consecutive_successes_override: Optional[int] = None,
) -> dict:
    env_cfg = _load_env_cfg(task)
    agent_cfg = _load_agent_cfg()

    env_cfg.scene.num_envs = num_envs
    env_cfg.seed = seed
    if max_consecutive_successes_override is not None:
        env_cfg.max_consecutive_successes = int(max_consecutive_successes_override)
    agent_cfg["params"]["seed"] = seed
    agent_cfg["params"]["config"]["torch_compile"] = False

    env = gym.make(task, cfg=env_cfg, render_mode=None)

    # CRITICAL: override env-side curriculum state BEFORE any rollout.
    raw_env = env.unwrapped
    if hasattr(raw_env, "success_tolerance"):
        raw_env.success_tolerance = float(eval_tolerance)
        # also bump initial so curriculum doesn't tighten further during eval
        raw_env.initial_tolerance = float(eval_tolerance)
        raw_env.target_tolerance = float(eval_tolerance)
        raw_env.last_curriculum_update = 10**9  # disable curriculum updates during eval

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
    agent_cfg["params"]["load_path"] = str(checkpoint_path)
    agent_cfg["params"]["config"]["num_actors"] = env.unwrapped.num_envs
    agent_cfg["params"]["config"]["player"] = {
        "deterministic": deterministic,
        "games_num": 10 ** 9,
        "print_stats": False,
    }

    runner = Runner()
    runner.load(agent_cfg)
    agent = runner.create_player()
    agent.reset()
    _filtered_restore(agent, checkpoint_path, rl_device)

    raw_env = env.unwrapped
    obs = env.reset()
    if isinstance(obs, dict):
        obs = obs["obs"]
    _ = agent.get_batch_size(obs, 1)
    if agent.is_rnn:
        agent.init_rnn()

    episode_nsucc: list[float] = []
    episode_reward: list[float] = []
    episode_object_err: list[float] = []
    episode_dropped: list[float] = []
    cumul_reward = torch.zeros(env.unwrapped.num_envs, device=rl_device)
    step = 0
    while simulation_app.is_running() and step < max_steps:
        with torch.no_grad():
            obs_t = agent.obs_to_torch(obs)
            actions = agent.get_action(obs_t, is_deterministic=deterministic)
            obs, reward, dones, extras = env.step(actions)
        step += 1
        if isinstance(reward, torch.Tensor):
            cumul_reward = cumul_reward + reward.float().view(-1)
        dm = dones.bool()
        if dm.any():
            idx = dm.nonzero(as_tuple=False).squeeze(-1)
            ns_vals = raw_env.prev_episode_successes[idx].detach().cpu().numpy()
            episode_nsucc.extend(ns_vals.tolist())
            if hasattr(raw_env, "object_pos") and hasattr(raw_env, "goal_pos"):
                err = (raw_env.object_pos[idx] - raw_env.goal_pos[idx]).norm(dim=-1)
                episode_object_err.extend(err.detach().cpu().numpy().tolist())
            episode_reward.extend(cumul_reward[idx].detach().cpu().numpy().tolist())
            cumul_reward[idx] = 0.0
            # consider a drop when object_pos.z fell below 0.1 -- matches env termination
            if hasattr(raw_env, "object_pos"):
                dropped = (raw_env.object_pos[idx, 2] < 0.1).float()
                episode_dropped.extend(dropped.detach().cpu().numpy().tolist())
        if agent.is_rnn and agent.states is not None and len(dones) > 0:
            for s in agent.states:
                s[:, dones.bool(), :] = 0.0
        if len(episode_nsucc) >= num_episodes:
            break

    arr = np.array(episode_nsucc) if episode_nsucc else np.zeros(0)
    rew_arr = np.array(episode_reward) if episode_reward else np.zeros(0)
    err_arr = np.array(episode_object_err) if episode_object_err else np.zeros(0)
    drop_arr = np.array(episode_dropped) if episode_dropped else np.zeros(0)

    final_tol = float(getattr(raw_env, "success_tolerance", eval_tolerance))
    target_tol = float(getattr(raw_env, "target_tolerance", eval_tolerance))

    if arr.size:
        nsucc_mean = float(arr.mean())
        nsucc_median = float(np.median(arr))
        nsucc_max = float(arr.max())
        nsucc_std = float(arr.std())
        success_rate = float((arr > 0).mean())
    else:
        nsucc_mean = nsucc_median = nsucc_max = nsucc_std = success_rate = float("nan")

    # r_meta replicated from dexpbt_utils.tolerance_successes_objective.
    # We compute the average per-episode r_meta for direct comparison with training.
    try:
        init_tol = float(getattr(raw_env, "initial_tolerance", eval_tolerance))
        span = init_tol - target_tol
        if abs(span) < 1e-9:
            tol_obj = 1.0
        else:
            tol_obj = (init_tol - final_tol) / span
        if final_tol > target_tol:
            r_meta_eval = float((arr * 0.01 + tol_obj).mean()) if arr.size else float("nan")
        else:
            r_meta_eval = float((arr + tol_obj).mean()) if arr.size else float("nan")
    except Exception:
        r_meta_eval = float("nan")

    # Capture num_envs BEFORE close() — DexPBTKukaAllegroEnv.num_envs reads
    # self.scene which is destroyed by close().
    try:
        actual_num_envs = int(env.unwrapped.num_envs)
    except Exception:
        actual_num_envs = int(num_envs)

    env.close()

    return {
        "num_envs": actual_num_envs,
        "num_episodes": int(arr.size),
        "deterministic": bool(deterministic),
        "success_tolerance": final_tol,
        "target_success_tolerance": target_tol,
        "nsucc_mean": nsucc_mean,
        "nsucc_median": nsucc_median,
        "nsucc_max": nsucc_max,
        "nsucc_std": nsucc_std,
        "success_rate": success_rate,
        "r_meta_eval": r_meta_eval,
        "reward_mean": float(rew_arr.mean()) if rew_arr.size else float("nan"),
        "drop_rate": float(drop_arr.mean()) if drop_arr.size else float("nan"),
        "object_error_mean": float(err_arr.mean()) if err_arr.size else float("nan"),
    }


def main() -> None:
    workspace = Path(args_cli.workspace).resolve()
    deterministic = args_cli.deterministic == "true"
    output_csv = Path(args_cli.output_csv).resolve()
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    if args_cli.checkpoint_override is not None:
        # Single-checkpoint shortcut
        targets = [(int(-1), Path(args_cli.checkpoint_override).resolve(), None)]
    else:
        if args_cli.policy_indices == "all":
            indices = list(range(32))  # cap; metas with no checkpoint are skipped
        else:
            indices = [int(s.strip()) for s in args_cli.policy_indices.split(",") if s.strip()]
        # Track A pinned-checkpoint extension (2026-06-06).
        pinned_iters = _parse_idx_int_map(args_cli.policy_checkpoint_iterations)
        expected_frames = _parse_idx_int_map(args_cli.policy_expected_frames)
        targets = []
        for idx in indices:
            if idx in pinned_iters:
                pin_iter = int(pinned_iters[idx])
                ckpt, meta = _pick_pinned_checkpoint(workspace, idx, pin_iter)
                if ckpt is None or not ckpt.exists():
                    print(
                        f"[fair_eval] WARN: skipping policy_idx={idx} pinned at "
                        f"iter={pin_iter} (checkpoint missing)"
                    )
                    continue
                # Frame-mismatch check (constraint 2 / plan §1.2).
                if meta is not None and idx in expected_frames:
                    json_frames = int(meta.get("frame", -1))
                    exp_frames = int(expected_frames[idx])
                    delta = abs(json_frames - exp_frames)
                    print(
                        f"[pin] policy={idx} tick={pin_iter} "
                        f"json_env_frames={json_frames} "
                        f"expected_frames={exp_frames} delta={delta}"
                    )
                    if delta > 10_000_000:
                        print(
                            f"[pin] WARNING: frame mismatch for policy={idx} "
                            f"|json_env_frames - expected_frames|={delta} "
                            f"exceeds 10M; eval continues regardless."
                        )
                elif meta is not None:
                    json_frames = int(meta.get("frame", -1))
                    print(
                        f"[pin] policy={idx} tick={pin_iter} "
                        f"json_env_frames={json_frames} "
                        f"expected_frames=<unset>"
                    )
                targets.append((idx, ckpt, meta))
                continue
            ckpt, meta = _pick_checkpoint(workspace, idx, args_cli.checkpoint_rule)
            if ckpt is None or not ckpt.exists():
                continue
            targets.append((idx, ckpt, meta))

    if not targets:
        print(f"[fair_eval] No checkpoints found under workspace={workspace}")
        sys.exit(2)

    rows: list[dict] = []
    for idx, ckpt, meta in targets:
        eval_tol = _resolve_eval_tolerance(
            args_cli.eval_tolerance_mode,
            args_cli.fixed_success_tolerance,
            meta,
        )
        print(f"\n[fair_eval] policy_idx={idx}  ckpt={ckpt}  eval_tol={eval_tol}")
        result = _run_one_eval(
            task=args_cli.task,
            checkpoint_path=ckpt,
            eval_tolerance=eval_tol,
            num_envs=args_cli.num_envs,
            num_episodes=args_cli.num_episodes,
            deterministic=deterministic,
            seed=args_cli.seed,
            max_steps=args_cli.max_steps,
            max_consecutive_successes_override=args_cli.max_consecutive_successes_override,
        )
        row = {
            "method_name": args_cli.method_name,
            "policy_idx": int(idx),
            "checkpoint_path": str(ckpt),
            "checkpoint_rule": args_cli.checkpoint_rule,
            "checkpoint_iteration": int((meta or {}).get("iteration", -1)),
            "env_frames": int((meta or {}).get("frame", -1)),
            "eval_tolerance_mode": args_cli.eval_tolerance_mode,
            "tolerance_applied": float(eval_tol),
            "running_norm_loaded": _checkpoint_has_running_norm(ckpt),
            "max_consecutive_successes_applied": int(
                args_cli.max_consecutive_successes_override
            ) if args_cli.max_consecutive_successes_override is not None else 50,
            **result,
        }
        rows.append(row)
        print(json.dumps(row, indent=2))

    fieldnames = sorted({k for r in rows for k in r.keys()})
    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    print(f"\n[fair_eval] wrote {len(rows)} rows -> {output_csv}")


if __name__ == "__main__":
    main()
    simulation_app.close()
