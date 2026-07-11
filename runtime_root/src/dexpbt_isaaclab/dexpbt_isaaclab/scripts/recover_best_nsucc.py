#!/usr/bin/env python3
"""Patch D (rev 3, 2026-05-30): offline peak-checkpoint eval helper.

The Patch 5 `best_nsucc/` and `best_r_meta/` archives accumulate peak
DexPBT checkpoints that survive ranking collapse, but they are NOT
wired into PBT source selection (by design -- see plan / audit report).
This script provides an offline way to:

  1. Inspect the archive and surface the highest-scoring checkpoint by
     a configurable criterion (`nsucc_mean` or `nsucc_max`).
  2. Print the matching metadata in machine-readable JSON.
  3. Optionally delegate to `dexpbt_isaaclab.scripts.eval_fair` to run
     a real evaluation on that checkpoint (deterministic protocol,
     same num_envs / num_episodes as production eval).

The picker logic is in `_pick_best_from_archive()` and is intentionally
pure (no Isaac Sim imports), so it can be unit-tested without bringing
up the Omniverse stack.

Usage:

  # 1. Just inspect what the archive thinks is best:
  python -m dexpbt_isaaclab.scripts.recover_best_nsucc \
      --workspace /path/to/run/dexpbt_8gpu_ws

  # 2. Inspect and run a real eval via eval_fair:
  python -m dexpbt_isaaclab.scripts.recover_best_nsucc \
      --workspace /path/to/run/dexpbt_8gpu_ws \
      --run_eval \
      --task DexPBT-Regrasping-KukaAllegro-Direct-v0 \
      --num_envs 8192 --num_episodes 256

  # 3. Switch archive / criterion:
  python -m dexpbt_isaaclab.scripts.recover_best_nsucc \
      --workspace ... --archive best_r_meta --criterion nsucc_max
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


_VALID_ARCHIVES = ("best_nsucc", "best_r_meta")
_VALID_CRITERIA = ("nsucc_mean", "nsucc_max", "r_meta_raw")


def _read_meta(path: Path) -> dict[str, Any] | None:
    """Read one archive json file, returning None on parse error."""
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def _pick_best_from_archive(
    archive_dir: Path,
    criterion: str = "nsucc_mean",
) -> dict[str, Any]:
    """Return the highest-scoring archive entry under `criterion`.

    Args:
      archive_dir: Path to `best_nsucc/` or `best_r_meta/`.
      criterion: One of `nsucc_mean`, `nsucc_max`, `r_meta_raw`.

    Raises:
      ValueError: archive doesn't exist, criterion is unknown, or no
        valid entries were found.

    The selection ignores `_index.json` (which always reflects the
    *latest* peak written, not the highest one), instead enumerating
    every iter_XXXX_policyYYY_*.json file and picking by criterion.
    """
    if criterion not in _VALID_CRITERIA:
        raise ValueError(
            f"Unknown criterion={criterion!r}; expected one of {_VALID_CRITERIA}"
        )
    if not archive_dir.is_dir():
        raise ValueError(f"Archive directory not found: {archive_dir}")

    candidates: list[tuple[float, Path, dict]] = []
    for f in archive_dir.glob("*.json"):
        if f.name == "_index.json":
            continue
        meta = _read_meta(f)
        if meta is None:
            continue
        try:
            score = float(meta.get(criterion, 0.0) or 0.0)
        except Exception:
            continue
        candidates.append((score, f, meta))

    if not candidates:
        raise ValueError(
            f"No valid entries under {archive_dir} for criterion={criterion!r}"
        )

    candidates.sort(key=lambda x: x[0], reverse=True)
    score, path, meta = candidates[0]
    # Surface the path and chosen criterion for downstream tooling.
    return {
        "archive_dir": str(archive_dir),
        "criterion": criterion,
        "criterion_value": score,
        "archive_file": str(path),
        "meta": meta,
    }


def _summary_payload(pick: dict[str, Any]) -> dict[str, Any]:
    """Project the chosen archive entry to a flat, machine-readable summary."""
    meta = pick["meta"]
    return {
        "archive_dir": pick["archive_dir"],
        "archive_file": pick["archive_file"],
        "criterion": pick["criterion"],
        "criterion_value": pick["criterion_value"],
        "checkpoint_path": meta.get("checkpoint"),
        "policy_idx": meta.get("policy_idx"),
        "frame": meta.get("frame"),
        "iteration": meta.get("iteration"),
        "saved_nsucc_mean": meta.get("nsucc_mean"),
        "saved_nsucc_max": meta.get("nsucc_max"),
        "saved_r_meta_raw": meta.get("r_meta_raw"),
        "saved_pbt_ranking_score": meta.get(
            "pbt_ranking_score", meta.get("objective")
        ),
        "saved_success_tolerance": (
            (meta.get("curriculum") or {}).get("success_tolerance")
        ),
        "saved_target_tolerance": (
            (meta.get("curriculum") or {}).get("target_tolerance")
        ),
        "saved_initial_tolerance": (
            (meta.get("curriculum") or {}).get("initial_tolerance")
        ),
    }


def _build_eval_fair_command(
    summary: dict[str, Any],
    *,
    task: str,
    num_envs: int,
    num_episodes: int,
    deterministic: str,
    output_csv: str,
    method_name: str,
    eval_tolerance_mode: str,
    fixed_success_tolerance: float,
) -> list[str]:
    """Compose the `eval_fair` CLI for a single-checkpoint eval."""
    return [
        sys.executable, "-u", "-m", "dexpbt_isaaclab.scripts.eval_fair",
        "--workspace", str(Path(summary["archive_dir"]).parent),
        "--method_name", method_name,
        "--task", task,
        "--checkpoint_rule", "best_r_meta",
        "--eval_tolerance_mode", eval_tolerance_mode,
        "--fixed_success_tolerance", str(fixed_success_tolerance),
        "--num_envs", str(num_envs),
        "--num_episodes", str(num_episodes),
        "--deterministic", deterministic,
        "--output_csv", output_csv,
        "--checkpoint_override", str(summary["checkpoint_path"]),
        "--headless",
    ]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Patch D: read best_nsucc / best_r_meta archive and "
        "(optionally) run an eval_fair pass on the peak checkpoint."
    )
    parser.add_argument(
        "--workspace", type=str, required=True,
        help="PBT workspace directory (contains best_nsucc/ / best_r_meta/).",
    )
    parser.add_argument(
        "--archive", type=str, default="best_nsucc",
        choices=_VALID_ARCHIVES,
        help="Which archive to read (default: best_nsucc).",
    )
    parser.add_argument(
        "--criterion", type=str, default="nsucc_mean",
        choices=_VALID_CRITERIA,
        help="Score by which to pick the 'best' archive entry.",
    )
    parser.add_argument(
        "--output_json", type=str, default=None,
        help="Optional path to write the JSON summary. If omitted, the "
        "summary is printed to stdout.",
    )
    parser.add_argument(
        "--run_eval", action="store_true",
        help="If set, invoke eval_fair on the chosen checkpoint after "
        "printing the summary.",
    )
    # eval_fair pass-through options (only used with --run_eval).
    parser.add_argument(
        "--task", type=str,
        default="DexPBT-Regrasping-KukaAllegro-Direct-v0",
    )
    parser.add_argument("--num_envs", type=int, default=8192)
    parser.add_argument("--num_episodes", type=int, default=256)
    parser.add_argument(
        "--deterministic", type=str, default="true",
        choices=["true", "false"],
    )
    parser.add_argument("--output_csv", type=str, default="/tmp/peak_eval.csv")
    parser.add_argument(
        "--method_name", type=str, default="recover_best_nsucc",
    )
    parser.add_argument(
        "--eval_tolerance_mode", type=str, default="checkpoint",
        choices=["checkpoint", "fixed"],
        help="`checkpoint` reuses the success_tolerance the archive entry "
        "was saved with (faithful evaluation). `fixed` forces "
        "--fixed_success_tolerance.",
    )
    parser.add_argument(
        "--fixed_success_tolerance", type=float, default=0.01,
    )
    args = parser.parse_args()

    archive_dir = Path(args.workspace).resolve() / args.archive
    pick = _pick_best_from_archive(archive_dir, criterion=args.criterion)
    summary = _summary_payload(pick)

    if args.output_json:
        Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output_json).write_text(json.dumps(summary, indent=2))
        print(f"[recover_best_nsucc] Wrote summary -> {args.output_json}")
    print(json.dumps(summary, indent=2))

    if not args.run_eval:
        return 0

    ckpt = summary.get("checkpoint_path")
    if not ckpt or not Path(ckpt).is_file():
        print(
            f"[recover_best_nsucc] ERROR: checkpoint path is missing or "
            f"not a file: {ckpt!r}",
            file=sys.stderr,
        )
        return 2

    cmd = _build_eval_fair_command(
        summary,
        task=args.task,
        num_envs=args.num_envs,
        num_episodes=args.num_episodes,
        deterministic=args.deterministic,
        output_csv=args.output_csv,
        method_name=args.method_name,
        eval_tolerance_mode=args.eval_tolerance_mode,
        fixed_success_tolerance=args.fixed_success_tolerance,
    )
    print("[recover_best_nsucc] Running eval_fair:")
    print("  " + " ".join(cmd))
    return subprocess.call(cmd, env=dict(os.environ))


if __name__ == "__main__":
    raise SystemExit(main())
