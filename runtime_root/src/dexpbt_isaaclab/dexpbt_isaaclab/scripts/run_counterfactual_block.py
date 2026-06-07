#!/usr/bin/env python
"""
Standalone Counterfactual Block Runner.

Polls for cf_block_request.json, launches children per strategy,
waits for completion, writes cf_block_result.json.

Usage:
    python -m dexpbt_isaaclab.scripts.run_counterfactual_block \
        --workspace /path/to/workspace \
        --poll_interval 30

Run in background:
    nohup python -m dexpbt_isaaclab.scripts.run_counterfactual_block \
        --workspace ... > cf_runner.log 2>&1 &
"""

import argparse
import json
import os
import subprocess
import sys
import time
import math
from pathlib import Path

# Add isaaclab to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Import the reporting r_meta decomposition so we can detect low_signal
# blocks where the apparent r_meta change is purely tolerance-driven.
try:  # pragma: no cover - import guard for cf-runner standalone mode
    from dexpbt_isaaclab.dexpbt_utils import split_r_meta_components
except Exception:  # pragma: no cover
    split_r_meta_components = None  # type: ignore


MAX_PARALLEL = 2           # max strategies running simultaneously
CF_GPUS = [6, 7]           # GPU 6-7 dedicated to CF children
CF_NUM_ENVS = 4096         # lower than main training (8192)
CF_HORIZON = 16
TIMEOUT_SEC = 30 * 60      # default; overridden per child below

# Patch 3 / 4 (rev 2, 2026-05-30): CF child training / eval defaults.
# These can all be overridden per-block via the request payload.
DEFAULT_CF_TRAIN_FRAMES = 20_000_000
DEFAULT_CF_MIN_EVAL_S = 30.0
DEFAULT_CF_RESET_EPOCH = True
DEFAULT_CF_DELTA_MODE = "learning_improvement"  # or "raw_after"
DEFAULT_CF_LOW_SIGNAL_ACTION = "skip"  # or "downweight"
DEFAULT_CF_LOW_SIGNAL_WEIGHT = 0.25


def build_param_overrides(params: dict) -> list[str]:
    """Convert params dict to CLI overrides for train.py."""
    overrides = []
    for key, val in params.items():
        if key in ("gamma", "e_clip", "kl_threshold", "learning_rate", "critic_coef"):
            overrides.extend([f"--{key}", str(val)])
    return overrides


def parse_status_file(path: Path) -> dict:
    """Parse status.txt for r_meta result."""
    if not path.exists():
        return {}
    try:
        # Prefer JSON if available.
        text = path.read_text().strip()
        if text.startswith("{") and text.endswith("}"):
            data = json.loads(text)
            return data if isinstance(data, dict) else {}
        lines = path.read_text().strip().split("\n")
        result = {}
        for line in lines:
            if "=" in line:
                k, v = line.strip().split("=", 1)
                try:
                    result[k.strip()] = float(v.strip())
                except ValueError:
                    result[k.strip()] = v.strip()
        return result
    except Exception:
        return {}


def _steps_to_iterations(cf_steps: int, num_envs: int, horizon: int = 16) -> int:
    """Convert target env-steps to PPO epochs (max_iterations).

    Patch 3 (rev 2, 2026-05-30): the source argument is now interpreted as
    `cf_train_frames` -- the number of env frames the child should train
    -- not the legacy `cf_steps`. Caller (launch_child) selects which.
    """
    denom = max(1, int(num_envs) * int(horizon))
    return max(1, int(math.ceil(float(cf_steps) / float(denom))))


def _request_int(request: dict, key: str, default: int) -> int:
    v = request.get(key, default)
    try:
        return int(v)
    except Exception:
        return int(default)


def _request_float(request: dict, key: str, default: float) -> float:
    v = request.get(key, default)
    try:
        return float(v)
    except Exception:
        return float(default)


def _request_bool(request: dict, key: str, default: bool) -> bool:
    v = request.get(key, default)
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        return v.strip().lower() in ("1", "true", "yes", "y", "on")
    try:
        return bool(int(v))
    except Exception:
        return bool(default)


def _safe_float(v, default: float) -> float:
    """Coerce status / result values to float, mapping None / non-numeric to default.

    Patch 3 + 4 (rev 2, 2026-05-30): status.txt may serialize r_meta_before
    as JSON null (which becomes Python None after `parse_status_file`).
    Without this guard, `float(None)` crashes in `run_cf_block`'s delta_w
    computation. Mirrors the test-side helper in
    test_cf_learning_improvement.py.
    """
    if v is None:
        return float(default)
    try:
        return float(v)
    except (TypeError, ValueError):
        return float(default)


def launch_child(strategy: str, block_id: int, batch_idx: int,
                 request: dict, gpu_id: int) -> tuple[subprocess.Popen, Path]:
    """Launch one CF child process.

    Patch 3 (rev 2, 2026-05-30):
      * `cf_train_frames` (not `cf_steps`) drives `max_iterations` so the
        child actually trains for a meaningful budget rather than exiting
        immediately after restoring a high-epoch source checkpoint.
      * `--cf_reset_epoch_counter` is forwarded to train.py so the child
        resets `agent.frame` / `agent.epoch_num` to 0 after restore.
      * `--cf_min_eval_s` and `--cf_train_frames` are forwarded so the
        child writes its target into status.txt for the runner's
        invalid/low_signal validation.
    """
    child_dir = request["workspace_dir"] / f"cf_block_{block_id}" / f"batch_{batch_idx}" / f"child_{strategy}"
    child_dir.mkdir(parents=True, exist_ok=True)

    train_root = child_dir / "rl_games"
    train_root.mkdir(parents=True, exist_ok=True)

    num_envs = _request_int(request, "cf_num_envs", CF_NUM_ENVS)
    cf_horizon = _request_int(request, "cf_horizon", CF_HORIZON)
    cf_train_frames = _request_int(
        request, "cf_train_frames", DEFAULT_CF_TRAIN_FRAMES
    )
    cf_reset_epoch = _request_bool(
        request, "cf_reset_epoch_counter", DEFAULT_CF_RESET_EPOCH
    )
    cf_min_eval_s = _request_float(request, "cf_min_eval_s", DEFAULT_CF_MIN_EVAL_S)
    max_iterations = _steps_to_iterations(cf_train_frames, num_envs, horizon=cf_horizon)
    source_curriculum = dict(request.get("source_curriculum", {}))
    source_tolerance = float(request.get("source_tolerance", source_curriculum.get("success_tolerance", 0.075)))

    cmd = [
        sys.executable, "-u", "-m", "dexpbt_isaaclab.scripts.train",
        "--task", request["env_task"],
        "--train_root", str(train_root),
        "--experiment_name", f"cf_{strategy}_b{batch_idx}",
        "--num_envs", str(num_envs),
        "--max_iterations", str(max_iterations),
        "--headless",
        "--enable_wm",
        "--wm_strategy", strategy,
        "--wm_ensemble_size", "3",
        "--wm_update_every_n_epochs", "2",
        "--observer_mode", "default",
        "--checkpoint", request["source_checkpoint"],
        "--wm_load_mode", "reset" if strategy == "none" else "auto",
        "--success_tolerance_resume", str(source_tolerance),
        "--status_output_path", str(child_dir / "status.txt"),
        # Patch 3 (rev 2): explicit child-side budget / behaviour.
        "--cf_train_frames", str(cf_train_frames),
        "--cf_min_eval_s", str(cf_min_eval_s),
    ]
    if cf_reset_epoch:
        cmd.append("--cf_reset_epoch_counter")
    if "initial_tolerance" in source_curriculum:
        cmd.extend(["--initial_tolerance_resume", str(source_curriculum["initial_tolerance"])])
    if "target_tolerance" in source_curriculum:
        cmd.extend(["--target_tolerance_resume", str(source_curriculum["target_tolerance"])])
    if "last_curriculum_update" in source_curriculum:
        cmd.extend(["--last_curriculum_update_resume", str(source_curriculum["last_curriculum_update"])])
    if "frame_since_restart" in source_curriculum:
        cmd.extend(["--frame_since_restart_resume", str(source_curriculum["frame_since_restart"])])
    cmd.extend(build_param_overrides(request.get("source_params", {})))

    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    log_file = open(child_dir / "train.log", "ab")
    proc = subprocess.Popen(
        cmd,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL,
        start_new_session=True,
    )
    return proc, child_dir


def _classify_child_result(status: dict, request: dict) -> dict:
    """Annotate a CF child result with `invalid` / `low_signal` flags.

    Patch 3 + 4 (rev 2, 2026-05-30) + Patch C (rev 3, 2026-05-30):

      * `invalid` -- the child didn't actually train. Triggered by missing
        fields, wall-time below `cf_min_eval_s`, no PPO samples, or trained
        frames below `cf_invalid_frames_ratio * cf_train_frames` (default
        0.25 since the previous 0.5 threshold combined with the
        `child_train_frames=0` bug masked perfectly valid 600-second CF
        runs in regrasping_cbwm_20260530_160513).
      * `low_signal` -- nsucc never moved off zero and the apparent
        `r_meta` change is fully explained by tolerance progress. This
        block carries no real selection signal between strategies, so
        the runner downstream either skips it or downweights it.

    `invalid_reason` is now an enum-style string for downstream filters:
      "missing_r_meta" | "elapsed_too_short" | "no_samples"
      | "frame_count_missing" | "insufficient_frames"
    Plus block-level reasons set by `run_cf_block`:
      "none_baseline_invalid" | "none_missing"
    """
    cf_min_eval_s = _request_float(request, "cf_min_eval_s", DEFAULT_CF_MIN_EVAL_S)
    cf_train_frames = _request_int(
        request, "cf_train_frames", DEFAULT_CF_TRAIN_FRAMES
    )
    cf_invalid_frames_ratio = _request_float(
        request, "cf_invalid_frames_ratio", 0.25
    )

    elapsed_s = float(status.get("elapsed_s", 0.0) or 0.0)
    child_train_frames = int(status.get("child_train_frames", 0) or 0)
    samples = int(status.get("r_meta_samples", 0) or 0)

    invalid = False
    invalid_reason = None
    if "r_meta_after" not in status and "r_meta" not in status:
        invalid, invalid_reason = True, "missing_r_meta"
    elif elapsed_s < cf_min_eval_s:
        invalid = True
        invalid_reason = "elapsed_too_short"
    elif samples < 2:
        invalid = True
        invalid_reason = "no_samples"
    elif child_train_frames == 0:
        # Distinct from "insufficient_frames": elapsed_s and samples both
        # cleared, but the writer never produced a positive frame count.
        # In Patch C this should be a true outlier; surface the
        # frame_estimation_method to aid diagnosis.
        invalid = True
        invalid_reason = "frame_count_missing"
    elif child_train_frames < cf_invalid_frames_ratio * cf_train_frames:
        invalid = True
        invalid_reason = "insufficient_frames"

    low_signal = False
    low_signal_reason = None
    if not invalid:
        nsucc_before = float(
            status.get("nsucc_before",
                       status.get("mean_prev_episode_successes", 0.0)) or 0.0
        )
        nsucc_after = float(
            status.get("nsucc_after",
                       status.get("mean_prev_episode_successes", 0.0)) or 0.0
        )
        if abs(nsucc_before) < 1e-9 and abs(nsucc_after) < 1e-9:
            # Compare success-component, NOT raw r_meta, so a change that
            # is entirely tolerance progress doesn't fool us.
            r_meta_before = float(
                status.get("r_meta_before",
                           status.get("r_meta", 0.0)) or 0.0
            )
            r_meta_after = float(
                status.get("r_meta_after",
                           status.get("r_meta", 0.0)) or 0.0
            )
            tol_before = float(
                status.get("success_tolerance_before",
                           status.get("success_tolerance", 0.0)) or 0.0
            )
            tol_after = float(
                status.get("success_tolerance_after",
                           status.get("success_tolerance", 0.0)) or 0.0
            )
            init_tol = float(request.get("source_curriculum", {}).get(
                "initial_tolerance", tol_after) or tol_after)
            target_tol = float(request.get("source_curriculum", {}).get(
                "target_tolerance", tol_after) or tol_after)

            if split_r_meta_components is not None:
                _, _, suc_before = split_r_meta_components(
                    tol_before, init_tol, target_tol, nsucc_before
                )
                _, _, suc_after = split_r_meta_components(
                    tol_after, init_tol, target_tol, nsucc_after
                )
                if abs(suc_after - suc_before) < 1e-6:
                    low_signal = True
                    low_signal_reason = (
                        "nsucc_before=0,nsucc_after=0,"
                        "success_component_unchanged"
                    )
            else:  # pragma: no cover - fallback when import failed
                # Without the decomposition helper we still flag if both
                # nsucc are zero and the raw r_meta delta is small.
                if abs(r_meta_after - r_meta_before) < 1e-6:
                    low_signal = True
                    low_signal_reason = "nsucc=0_and_r_meta_delta~0"

    out = dict(status)
    out["invalid"] = bool(invalid)
    out["invalid_reason"] = invalid_reason
    out["low_signal"] = bool(low_signal)
    out["low_signal_reason"] = low_signal_reason
    return out


def _cf_source_health_ok(request: dict) -> tuple[bool, str]:
    """Runner-side defence: do not launch children on unhealthy sources."""
    min_mean = float(request.get("cf_source_min_nsucc_mean", 0.0) or 0.0)
    min_max = float(request.get("cf_source_min_nsucc_max", 0.0) or 0.0)
    nsucc_mean = float(request.get("source_nsucc_mean", 0.0) or 0.0)
    nsucc_max = float(request.get("source_nsucc_max", 0.0) or 0.0)
    if request.get("cf_require_nsucc_gt_zero", True) and nsucc_mean <= 0.0:
        return False, "source_nsucc_mean<=0"
    if nsucc_mean < min_mean:
        return False, f"source_nsucc_mean<{min_mean}"
    if nsucc_max < min_max:
        return False, f"source_nsucc_max<{min_max}"
    return True, "ok"


def run_cf_block(request: dict) -> dict:
    """Run one CF block: launch children, collect results, return delta_w.

    Patch 3 + 4 (rev 2, 2026-05-30): validates each child result against
    `cf_min_eval_s` and `cf_train_frames` (invalid), and against the
    tolerance-only-r_meta failure mode (low_signal). Both flags are
    surfaced in the result JSON so the posterior selector (and TB) can
    filter accordingly. delta_w supports two modes:

      * raw_after (legacy): delta_w[s] = r_meta_after[s] - r_meta_after[none]
      * learning_improvement (default): per-child learning increment,
        relative to `none`. Falls back to raw_after if any child lacks a
        valid `r_meta_before` snapshot.
    """
    block_id = request["block_id"]
    ok, reason = _cf_source_health_ok(request)
    if not ok:
        print(f"[CF] skip block {block_id}: unhealthy source ({reason})")
        return {
            "results": {},
            "delta_w": {},
            "posterior_weights": {},
            "invalid_block": True,
            "invalid_block_reason": reason,
            "all_strategies_tied": False,
            "cf_delta_mode": request.get("cf_delta_mode"),
        }

    strategies = request["strategies"]

    # Set workspace_dir from request if not present
    workspace = Path(request.get("_workspace", "/root/autodl-tmp/dexpbt_isaaclab/results/cf_workspace"))
    request["workspace_dir"] = workspace

    print(f"[CF] Block {block_id}: strategies={strategies}, checkpoint={request['source_checkpoint'][:80]}")

    cf_delta_mode = str(request.get("cf_delta_mode", DEFAULT_CF_DELTA_MODE))
    cf_low_signal_action = str(
        request.get("cf_low_signal_action", DEFAULT_CF_LOW_SIGNAL_ACTION)
    )
    cf_low_signal_weight = float(
        request.get("cf_low_signal_weight", DEFAULT_CF_LOW_SIGNAL_WEIGHT)
    )

    results: dict[str, dict] = {}

    max_parallel = _request_int(request, "cf_max_parallel_children", MAX_PARALLEL)
    gpus = [
        int(x)
        for x in str(
            request.get("cf_cuda_visible_devices", ",".join(str(g) for g in CF_GPUS))
        ).split(",")
        if x.strip()
    ]
    if not gpus:
        gpus = list(CF_GPUS)
    # Batch strategies
    batches = [strategies[i:i + max_parallel]
               for i in range(0, len(strategies), max_parallel)]

    cf_train_frames = _request_int(
        request, "cf_train_frames", DEFAULT_CF_TRAIN_FRAMES
    )

    for batch_idx, batch in enumerate(batches):
        print(f"[CF] Batch {batch_idx}/{len(batches)-1}: {batch}")
        procs = []

        for i, strategy in enumerate(batch):
            gpu_id = gpus[i % len(gpus)]
            proc, child_dir = launch_child(strategy, block_id, batch_idx, request, gpu_id)
            procs.append((proc, strategy, child_dir, gpu_id))
            print(f"[CF]   Launched {strategy} on GPU {gpu_id}, log={child_dir}/train.log")

        # Wait for batch to complete (with timeout)
        for proc, strategy, child_dir, gpu_id in procs:
            child_timeout = int(request.get("child_timeout_sec", TIMEOUT_SEC))
            if child_timeout <= 0:
                num_envs = _request_int(request, "cf_num_envs", CF_NUM_ENVS)
                cf_horizon = _request_int(request, "cf_horizon", CF_HORIZON)
                max_iterations = _steps_to_iterations(
                    cf_train_frames, num_envs, horizon=cf_horizon
                )
                child_timeout = max(TIMEOUT_SEC, max_iterations * 180)
            try:
                exit_code = proc.wait(timeout=child_timeout)
            except subprocess.TimeoutExpired:
                print(f"[CF]   TIMEOUT — killing {strategy} on GPU {gpu_id}")
                proc.kill()
                exit_code = -1

            if exit_code != 0:
                print(f"[CF]   {strategy} FAILED (exit={exit_code})")
                # Surface an invalid stub so downstream can warn explicitly.
                results[strategy] = {
                    "r_meta": float("nan"),
                    "r_meta_before": float("nan"),
                    "r_meta_after": float("nan"),
                    "nsucc_before": float("nan"),
                    "nsucc_after": float("nan"),
                    "elapsed_s": 0.0,
                    "child_train_frames": 0,
                    "r_meta_samples": 0,
                    "exit_code": int(exit_code),
                    "invalid": True,
                    "invalid_reason": f"exit_code={exit_code}",
                    "low_signal": False,
                    "low_signal_reason": None,
                    "gpu": gpu_id,
                }
                continue

            status_path = child_dir / "status.txt"
            if status_path.exists():
                status = parse_status_file(status_path)
                classified = _classify_child_result(status, request)
                r_meta_after = float(
                    classified.get("r_meta_after",
                                   classified.get("r_meta", 0.0))
                )
                results[strategy] = {
                    # Patch 3 (rev 2): keep legacy `r_meta` key = after value
                    # so older readers continue to function unchanged.
                    "r_meta": r_meta_after,
                    "r_meta_before": float(
                        classified.get("r_meta_before", float("nan"))
                    ),
                    "r_meta_after": r_meta_after,
                    "nsucc_before": float(
                        classified.get("nsucc_before", float("nan"))
                    ),
                    "nsucc_after": float(
                        classified.get("nsucc_after", float("nan"))
                    ),
                    "success_tolerance_before": float(
                        classified.get("success_tolerance_before", float("nan"))
                    ),
                    "success_tolerance_after": float(
                        classified.get("success_tolerance_after", float("nan"))
                    ),
                    "elapsed_s": float(classified.get("elapsed_s", 0.0) or 0.0),
                    "child_train_frames": int(
                        classified.get("child_train_frames", 0) or 0
                    ),
                    # Patch C (rev 3): expose which signal produced the
                    # frame count so analyzers can spot estimation chains
                    # falling back to samples-based math.
                    "frame_estimation_method": str(
                        classified.get("frame_estimation_method", "unknown")
                    ),
                    "frames_ratio": (
                        float(classified.get("child_train_frames", 0) or 0)
                        / float(max(1, cf_train_frames))
                    ),
                    "r_meta_samples": int(
                        classified.get("r_meta_samples", 0) or 0
                    ),
                    "invalid": bool(classified["invalid"]),
                    "invalid_reason": classified["invalid_reason"],
                    "low_signal": bool(classified["low_signal"]),
                    "low_signal_reason": classified["low_signal_reason"],
                    "steps": cf_train_frames,
                    "cf_num_envs": _request_int(request, "cf_num_envs", CF_NUM_ENVS),
                    "cf_horizon": _request_int(request, "cf_horizon", CF_HORIZON),
                    "gpu": gpu_id,
                }
                print(
                    f"[CF]   {strategy} done: r_meta_after={r_meta_after}, "
                    f"invalid={classified['invalid']} "
                    f"low_signal={classified['low_signal']} "
                    f"elapsed_s={classified.get('elapsed_s')}"
                )
            else:
                print(f"[CF]   {strategy}: no status.txt found")
                results[strategy] = {
                    "r_meta": float("nan"),
                    "r_meta_before": float("nan"),
                    "r_meta_after": float("nan"),
                    "elapsed_s": 0.0,
                    "child_train_frames": 0,
                    "invalid": True,
                    "invalid_reason": "no_status_txt",
                    "low_signal": False,
                    "low_signal_reason": None,
                    "gpu": gpu_id,
                }

    # ------------------------------------------------------------------
    # delta_w computation (Patch 4, rev 2, 2026-05-30).
    # ------------------------------------------------------------------
    none_result = results.get("none")
    invalid_block = False
    invalid_block_reason = None
    delta_w: dict[str, float] = {}
    posterior_weights: dict[str, float] = {}

    # Patch I (2026-05-31): pick which metric drives `delta_w`. Default
    # is the legacy r_meta-based delta; `nsucc` uses the count of mean
    # successes (ground-truth signal) which is the only thing that
    # survives unchanged through curriculum tightening.
    cf_delta_metric = str(request.get("cf_delta_metric", "r_meta"))
    if cf_delta_metric not in ("r_meta", "nsucc"):
        print(
            f"[CF] WARNING: unknown cf_delta_metric={cf_delta_metric!r}; "
            "falling back to r_meta"
        )
        cf_delta_metric = "r_meta"
    # Diagnostic per-strategy deltas (always present in result JSON, even
    # when delta_w uses a different metric). Helps analyzers cross-check.
    nsucc_deltas: dict[str, float] = {}
    r_meta_deltas: dict[str, float] = {}

    if none_result is None or none_result.get("invalid"):
        invalid_block = True
        invalid_block_reason = (
            "none_missing" if none_result is None else "none_baseline_invalid"
        )
        print(f"[CF] WARNING: invalid block ({invalid_block_reason}) -- no delta_w")
    else:
        if cf_delta_mode == "learning_improvement":
            # Each child needs a valid before snapshot. Fallback to raw_after
            # if not, but flag the block so analyzers can spot the regression.
            if cf_delta_metric == "nsucc":
                need_before = any(
                    math.isnan(float(r.get("nsucc_before", float("nan"))))
                    for r in results.values()
                    if not r.get("invalid")
                )
                before_key = "nsucc_before"
            else:
                need_before = any(
                    math.isnan(float(r.get("r_meta_before", float("nan"))))
                    for r in results.values()
                    if not r.get("invalid")
                )
                before_key = "r_meta_before"
            if need_before:
                print(
                    f"[CF] WARNING: {before_key} missing for at least one "
                    "valid strategy; falling back to raw_after delta_w"
                )
                cf_delta_mode = "raw_after"

        none_r_before = _safe_float(none_result.get("r_meta_before"), float("nan"))
        none_r_after = _safe_float(none_result.get("r_meta_after"), float("nan"))
        none_n_before = _safe_float(none_result.get("nsucc_before"), float("nan"))
        none_n_after = _safe_float(none_result.get("nsucc_after"), float("nan"))
        for s, r in results.items():
            if r.get("invalid"):
                continue
            r_after = _safe_float(
                r.get("r_meta_after"),
                _safe_float(r.get("r_meta"), 0.0),
            )
            n_after = _safe_float(r.get("nsucc_after"), 0.0)
            r_before = _safe_float(r.get("r_meta_before"), r_after)
            n_before = _safe_float(r.get("nsucc_before"), n_after)
            # Always compute both deltas for diagnostics.
            if cf_delta_mode == "learning_improvement":
                child_r_delta = r_after - r_before
                none_r_delta = none_r_after - none_r_before
                r_meta_deltas[s] = float(child_r_delta - none_r_delta)
                child_n_delta = n_after - n_before
                none_n_delta = none_n_after - none_n_before
                nsucc_deltas[s] = float(child_n_delta - none_n_delta)
            else:  # raw_after
                r_meta_deltas[s] = float(r_after - none_r_after)
                nsucc_deltas[s] = float(n_after - none_n_after)
            # Use the chosen metric for posterior selector.
            if cf_delta_metric == "nsucc":
                delta_w[s] = nsucc_deltas[s]
            else:
                delta_w[s] = r_meta_deltas[s]

            if r.get("low_signal"):
                if cf_low_signal_action == "skip":
                    delta_w.pop(s, None)
                    posterior_weights[s] = 0.0
                elif cf_low_signal_action == "downweight":
                    posterior_weights[s] = float(cf_low_signal_weight)
                else:
                    posterior_weights[s] = 1.0
            else:
                posterior_weights[s] = 1.0

    # Detect "all strategies tied" condition (the historic failure mode at
    # cf_block_85 where every strategy reported r_meta=0.541).
    after_values = [
        float(r.get("r_meta_after", float("nan")))
        for r in results.values()
        if not r.get("invalid")
    ]
    after_values = [v for v in after_values if not math.isnan(v)]
    all_strategies_tied = bool(
        after_values
        and max(after_values) - min(after_values) < 1e-9
    )

    if all_strategies_tied:
        print(
            f"[CF] WARNING: all strategies tied at "
            f"r_meta_after={after_values[0]:.6f}; "
            f"posterior will see zero signal"
        )

    return {
        "results": results,
        "delta_w": delta_w,
        "cf_delta_mode": cf_delta_mode,
        # Patch I (2026-05-31): also surface the per-strategy raw deltas
        # for both metrics so analyzers / posterior can audit which signal
        # is driving the selector (and rerun with the other metric if
        # needed). `delta_w` itself uses `cf_delta_metric`.
        "cf_delta_metric": cf_delta_metric,
        "r_meta_delta": r_meta_deltas,
        "nsucc_delta": nsucc_deltas,
        "cf_low_signal_action": cf_low_signal_action,
        "cf_low_signal_weight": cf_low_signal_weight,
        "posterior_weights": posterior_weights,
        "invalid_block": invalid_block,
        "invalid_block_reason": invalid_block_reason,
        "all_strategies_tied": all_strategies_tied,
        "cf_num_envs": _request_int(request, "cf_num_envs", CF_NUM_ENVS),
        "cf_train_frames": cf_train_frames,
        # Keep `cf_steps` for backward compat in the result JSON.
        "cf_steps": _request_int(request, "cf_steps", cf_train_frames),
        "cf_horizon": _request_int(request, "cf_horizon", CF_HORIZON),
        "source_num_envs": _request_int(request, "source_num_envs", 0),
    }


def _write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        fh.write(json.dumps(payload, indent=2))
        fh.flush()
        try:
            os.fsync(fh.fileno())
        except OSError:
            pass
    tmp.rename(path)


def _write_runner_heartbeat(workspace: Path) -> None:
    hb_path = workspace / "counterfactual" / "runner_heartbeat.json"
    _write_json_atomic(
        hb_path,
        {
            "timestamp": time.time(),
            "pid": os.getpid(),
            "status": "alive",
        },
    )


def main():
    parser = argparse.ArgumentParser(description="Counterfactual Block Runner")
    parser.add_argument("--workspace", type=str, required=True,
                        help="Workspace directory containing cf_block_request.json")
    parser.add_argument("--poll_interval", type=int, default=30,
                        help="Seconds between polling for new requests")
    parser.add_argument("--heartbeat_interval", type=int, default=30,
                        help="Seconds between heartbeat writes")
    parser.add_argument("--cf_num_envs", type=int, default=CF_NUM_ENVS)
    parser.add_argument("--cf_max_parallel_children", type=int, default=MAX_PARALLEL)
    parser.add_argument("--cf_cuda_visible_devices", type=str, default="4,5")
    parser.add_argument("--cf_horizon", type=int, default=CF_HORIZON)
    args = parser.parse_args()

    workspace = Path(args.workspace)
    request_path = workspace / "cf_block_request.json"
    result_path = workspace / "cf_block_result.json"

    print(f"[CF] Counterfactual Block Runner started.")
    print(f"[CF] Workspace: {workspace}")
    print(f"[CF] Polling every {args.poll_interval}s")
    print(f"[CF] Heartbeat every {args.heartbeat_interval}s")
    last_heartbeat = 0.0

    while True:
        now = time.time()
        if now - last_heartbeat >= max(1, int(args.heartbeat_interval)):
            _write_runner_heartbeat(workspace)
            last_heartbeat = now
        if request_path.exists():
            # Consume request
            try:
                request = json.loads(request_path.read_text())
                request_path.unlink()
            except Exception as e:
                print(f"[CF] Failed to read request: {e}")
                time.sleep(args.poll_interval)
                continue

            block_id = request.get("block_id", 0)
            print(f"\n{'='*60}")
            print(f"[CF] Processing block {block_id}")
            print(f"{'='*60}")

            # Set workspace in request
            request["_workspace"] = str(workspace)
            request.setdefault("cf_num_envs", int(args.cf_num_envs))
            request.setdefault("cf_max_parallel_children", int(args.cf_max_parallel_children))
            request.setdefault("cf_cuda_visible_devices", str(args.cf_cuda_visible_devices))
            request.setdefault("cf_horizon", int(args.cf_horizon))

            t0 = time.time()
            result = run_cf_block(request)
            elapsed = time.time() - t0

            print(f"\n[CF] Block {block_id} complete in {elapsed:.0f}s")
            print(f"[CF] Results: {result.get('results', {})}")
            print(f"[CF] Delta_w: {result.get('delta_w', {})}")

            # Write result atomically.
            # Patch 3 + 4 (rev 2, 2026-05-30): include the full validation
            # surface so the posterior selector and TB observer can act on it.
            result_data = {
                "block_id": block_id,
                "results": result.get("results", {}),
                "delta_w": result.get("delta_w", {}),
                "posterior_weights": result.get("posterior_weights", {}),
                "cf_delta_mode": result.get("cf_delta_mode"),
                "cf_low_signal_action": result.get("cf_low_signal_action"),
                "cf_low_signal_weight": result.get("cf_low_signal_weight"),
                "invalid_block": bool(result.get("invalid_block", False)),
                "invalid_block_reason": result.get("invalid_block_reason"),
                "all_strategies_tied": bool(result.get("all_strategies_tied", False)),
                "cf_num_envs": result.get("cf_num_envs"),
                "cf_train_frames": result.get("cf_train_frames"),
                "cf_steps": result.get("cf_steps"),
                "cf_horizon": result.get("cf_horizon"),
                "source_num_envs": result.get("source_num_envs"),
                "timestamp": time.time(),
            }
            _write_json_atomic(result_path, result_data)
            print(f"[CF] Wrote result → {result_path}")

        time.sleep(args.poll_interval)


if __name__ == "__main__":
    main()