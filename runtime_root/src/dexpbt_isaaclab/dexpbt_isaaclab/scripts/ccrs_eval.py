#!/usr/bin/env python3
"""CCRS single-unit evaluator: one checkpoint x one fixed tolerance x one seed.

Runs a frozen policy (rl_games player, deterministic by default) in the task
env with success_tolerance FORCED to a fixed value for the entire rollout.
No optimizer step, no PBT, no curriculum update, no mutation can occur:
this process never constructs a trainer/optimizer — only a player — and the
env-side curriculum trigger is disabled exactly as in eval_fair.py.

v2 paired-reset protocol:
- requires num_episodes <= num_envs;
- collects ONLY the first completed episode of each env_id < num_episodes
  (FirstEpisodeCollector), so the sample_id set (seed_XXXX_env_YYY) is
  identical across all checkpoints/tolerances sharing a reset seed;
- after the initial env.reset() and BEFORE the first action, records a
  per-env reset fingerprint (object/goal pose + robot joint positions,
  fixed-precision quantized) as pairing evidence;
- terminal error is taken from an eval-only pre-reset snapshot hook wrapped
  around env._reset_idx (the buffers read after step() are post-reset for
  done envs and must NOT be used). If the hook cannot attach,
  terminal_error_valid=false and terminal metrics are NaN;
- status=ok ONLY when all requested episodes were collected; otherwise
  status=incomplete with missing_env_ids and stop_reason.

policy_only mode (default, main CCRS results): pure rl_games player forward
pass; no snippet retrieval, no WM rerank, no MPPI, no residual action, no
actor distillation (those subsystems live in the training observer stack and
are never instantiated here; asserted at runtime).

system_on mode: reserved, rejected at startup.

One unit per process by design: Isaac Sim is not reliable across env.close()
re-creation in one process. Resume/skip: if the per-unit raw JSON exists and
passes the strict is_done() validation, the process exits 0 before Isaac boot.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

# --- protocol imports must not require Isaac ---
from dexpbt_isaaclab.ccrs.eval_protocol import (
    EVALUATION_MODE_POLICY_ONLY,
    EVALUATION_MODE_SYSTEM_ON,
    EvalUnit,
    git_dirty,
    is_done,
    make_result_row,
    result_path,
    source_sha256,
    write_result_atomic,
)
from dexpbt_isaaclab.ccrs.rollout_collect import (
    FirstEpisodeCollector,
    TerminalSnapshotHook,
    extract_reset_fingerprints,
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
parser.add_argument("--num-episodes", type=int, default=512,
                    help="Must be <= num-envs (paired-reset protocol)")
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


def _code_provenance() -> dict:
    """git_dirty + hash of the evaluator sources actually executing.

    git_sha alone is insufficient: this sweep can legitimately run from a
    dirty worktree, in which case HEAD does not identify the code. The
    source hash covers ccrs_eval.py plus the ccrs protocol modules it
    depends on for semantics.
    """
    here = Path(__file__).resolve()
    ccrs_dir = here.parent.parent / "ccrs"
    sources = [here,
               ccrs_dir / "eval_protocol.py",
               ccrs_dir / "rollout_collect.py"]
    from dexpbt_isaaclab.ccrs.eval_protocol import ANALYSIS_PROTOCOL_VERSION
    return {
        "git_dirty": git_dirty(here.parent),
        "evaluator_source_sha256": source_sha256(*sources),
        "analysis_protocol_version": ANALYSIS_PROTOCOL_VERSION,
    }


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

if _pre_args.num_episodes > _pre_args.num_envs:
    print(f"[ccrs_eval] paired-reset protocol requires num_episodes <= num_envs "
          f"(got {_pre_args.num_episodes} > {_pre_args.num_envs})", file=sys.stderr)
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
import torch  # noqa: E402
import yaml  # noqa: E402
from rl_games.common import env_configurations, vecenv  # noqa: E402
from rl_games.torch_runner import Runner  # noqa: E402

from isaaclab_rl.rl_games import RlGamesGpuEnv, RlGamesVecEnvWrapper  # noqa: E402
import dexpbt_isaaclab  # noqa: F401, E402  (register envs)


# NOTE: these three helpers mirror eval_fair.py. We cannot import eval_fair:
# its module top level builds its own argparse parser with different required
# args and boots its own AppLauncher, so importing it inside another script
# aborts on argument parsing.

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


def _filtered_restore(agent, checkpoint_path: Path, device: str) -> None:
    """Restore an rl_games agent, tolerating aux_value_head keys from WM runs."""
    try:
        agent.restore(str(checkpoint_path))
        return
    except RuntimeError as e:
        if "aux_value_head" not in str(e):
            raise
    print("[ccrs_eval] aux_value_head detected; loading with strict=False filter")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model_sd = ckpt.get("model", ckpt)
    filtered = {k: v for k, v in model_sd.items() if not k.startswith("aux_value_head")}
    agent.model.load_state_dict(filtered, strict=False)


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

    # --- reset fingerprints: AFTER initial reset, BEFORE the first action ---
    eligible = list(range(unit.num_episodes))
    per_env_fingerprint = extract_reset_fingerprints(raw_env, eligible)

    # --- eval-only pre-reset terminal snapshot hook (default off; installed
    # here explicitly, never in training code paths) ---
    hook = TerminalSnapshotHook(raw_env)
    hook_ok = hook.install()
    if not hook_ok:
        print("[ccrs_eval] WARN: terminal snapshot hook unavailable; "
              "terminal_error will be marked invalid")

    _ = agent.get_batch_size(obs, 1)
    if agent.is_rnn:
        agent.init_rnn()

    collector = FirstEpisodeCollector(
        reset_seed=unit.reset_seed,
        num_episodes=unit.num_episodes,
        max_consecutive_successes=unit.max_consecutive_successes,
    )

    step = 0
    max_steps_reached = False
    while simulation_app.is_running() and not collector.complete:
        if step >= unit.max_steps:
            max_steps_reached = True
            break
        with torch.no_grad():
            obs_t = agent.obs_to_torch(obs)
            actions = agent.get_action(obs_t, is_deterministic=unit.deterministic)
            obs, reward, dones, extras = env.step(actions)
        step += 1
        dm = dones.bool()
        if dm.any():
            idx = dm.nonzero(as_tuple=False).squeeze(-1)
            ns_vals = raw_env.prev_episode_successes[idx].detach().cpu().numpy()
            for env_id, ns in zip(idx.detach().cpu().numpy().tolist(),
                                  ns_vals.tolist()):
                terr, tvalid, dropped = hook.take(int(env_id))
                collector.offer(
                    env_id=int(env_id),
                    nsucc=float(ns),
                    terminal_error=terr,
                    terminal_error_valid=tvalid,
                    dropped=dropped,
                )
        if agent.is_rnn and agent.states is not None and len(dones) > 0:
            for s in agent.states:
                s[:, dones.bool(), :] = 0.0

    tol_at_end = float(raw_env.success_tolerance)
    hook.uninstall()
    env.close()

    return make_result_row(
        unit=unit,
        samples=collector.samples,
        per_env_fingerprint=per_env_fingerprint,
        git_sha=_git_sha(),
        checkpoint_hash=_ckpt_hash(Path(unit.checkpoint_path)),
        max_steps_reached=max_steps_reached,
        applied_tolerance_start=tol_at_start,
        applied_tolerance_end=tol_at_end,
        extra={"steps_used": step,
               "terminal_snapshot_hook_installed": bool(hook_ok),
               **_code_provenance()},
    )


def main() -> None:
    row = run_unit(_unit)
    out = write_result_atomic(_raw_dir, _unit, row)
    brief = {k: row[k] for k in (
        "status", "worker_id", "checkpoint_iteration", "fixed_tolerance",
        "reset_seed", "actual_num_episodes", "normalized_nsucc", "nsucc_mean",
        "success_any_rate", "terminal_error_valid",
    )}
    print(f"[ccrs_eval] finished: {json.dumps(brief)}")
    print(f"[ccrs_eval] wrote {out}")
    if row["status"] != "ok":
        print(f"[ccrs_eval] unit INCOMPLETE: missing={row['missing_env_ids'][:10]}"
              f"{'...' if len(row['missing_env_ids']) > 10 else ''} "
              f"stop_reason={row['stop_reason']}", file=sys.stderr)
        sys.exit(4)


if __name__ == "__main__":
    main()
    simulation_app.close()
