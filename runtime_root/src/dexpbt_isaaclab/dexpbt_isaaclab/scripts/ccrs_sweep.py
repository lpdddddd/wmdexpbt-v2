#!/usr/bin/env python3
"""CCRS sweep scheduler: run ccrs_eval units sequentially (per shard).

No Isaac import here — each unit runs in its own subprocess (Isaac cannot
survive env.close() + re-create in one process). Completed units are skipped
via the per-unit raw JSON marker, so the sweep is resumable and can be
sharded across GPUs by --shard-index/--num-shards (units are interleaved
deterministically).

Plan-only mode (--plan) prints the unit table and exits without touching
any GPU. This is the default safety behaviour unless --execute is passed.

Usage (plan only, safe):
  python -m dexpbt_isaaclab.scripts.ccrs_sweep \
    --manifest results/analysis/ccrs_base_retention/checkpoint_manifest.csv \
    --roles tip,peak,target,transition \
    --out-dir results/analysis/ccrs_base_retention \
    --task DexPBT-Regrasping-KukaAllegro-Direct-v0 \
    --plan

Execution (one shard, pinned to one GPU by CUDA_VISIBLE_DEVICES):
  CUDA_VISIBLE_DEVICES=0 python -m dexpbt_isaaclab.scripts.ccrs_sweep \
    ... --execute --shard-index 0 --num-shards 6
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path

from dexpbt_isaaclab.ccrs.eval_protocol import (
    DEFAULT_SEEDS,
    DEFAULT_TOLERANCES_M,
    EvalUnit,
    build_units,
    is_done,
    result_path,
    aggregate_raw_dir,
)


def load_manifest_checkpoints(manifest_csv: Path, roles_filter: list[str]) -> list[dict]:
    """Select manifest rows whose roles intersect the filter; dedupe by (worker, iter)."""
    rows: list[dict] = []
    seen: set[tuple[int, int]] = set()
    with manifest_csv.open() as f:
        for r in csv.DictReader(f):
            roles = [x for x in (r.get("roles") or "").split(";") if x]
            if roles_filter and not any(
                any(rf == role or role.startswith(rf) for rf in roles_filter)
                for role in roles
            ):
                continue
            if r.get("notes") and "CHECKPOINT_MISSING" in r["notes"]:
                continue
            key = (int(r["worker_id"]), int(r["iteration"]))
            if key in seen:
                continue
            seen.add(key)
            rows.append({
                "worker_id": int(r["worker_id"]),
                "iteration": int(r["iteration"]),
                "frame": int(r["frame"]),
                "checkpoint_path": r["checkpoint_path"],
                "run_id": r["run_id"],
            })
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description="CCRS sweep scheduler (subprocess driver).")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--task", required=True)
    ap.add_argument("--roles", default="tip,peak,target,transition",
                    help="Comma list of role prefixes to include from manifest")
    ap.add_argument("--tolerances", default=",".join(str(t) for t in DEFAULT_TOLERANCES_M),
                    help="Comma list of fixed tolerances in METERS")
    ap.add_argument("--seeds", default=",".join(str(s) for s in DEFAULT_SEEDS))
    ap.add_argument("--num-envs", type=int, default=512)
    ap.add_argument("--num-episodes", type=int, default=512)
    ap.add_argument("--deterministic", default="true", choices=["true", "false"])
    ap.add_argument("--max-steps", type=int, default=200_000)
    ap.add_argument("--shard-index", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--plan", action="store_true", help="Print plan and exit (no GPU)")
    ap.add_argument("--execute", action="store_true",
                    help="Actually run units (required to launch anything)")
    ap.add_argument("--aggregate-only", action="store_true",
                    help="Only merge existing raw results into jsonl/csv")
    ap.add_argument("--per-unit-timeout-s", type=int, default=3600)
    args = ap.parse_args()

    out_dir = Path(args.out_dir).resolve()
    raw_dir = out_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    if args.aggregate_only:
        n = aggregate_raw_dir(raw_dir, out_dir / "ccrs_results.jsonl", out_dir / "ccrs_results.csv")
        print(f"[ccrs_sweep] aggregated {n} rows")
        return

    ckpts = load_manifest_checkpoints(
        Path(args.manifest), [r.strip() for r in args.roles.split(",") if r.strip()]
    )
    if not ckpts:
        print("[ccrs_sweep] no checkpoints selected from manifest", file=sys.stderr)
        sys.exit(2)
    run_id = ckpts[0]["run_id"]
    tols = [float(t) for t in args.tolerances.split(",") if t.strip()]
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]

    units = build_units(
        run_id=run_id,
        checkpoints=ckpts,
        tolerances_m=tols,
        seeds=seeds,
        num_envs=args.num_envs,
        num_episodes=args.num_episodes,
        deterministic=(args.deterministic == "true"),
        max_steps=args.max_steps,
    )
    shard = [u for i, u in enumerate(units) if i % args.num_shards == args.shard_index]
    done = [u for u in shard if is_done(raw_dir, u)]
    todo = [u for u in shard if not is_done(raw_dir, u)]

    print(f"[ccrs_sweep] checkpoints={len(ckpts)} tolerances={len(tols)} seeds={len(seeds)}")
    print(f"[ccrs_sweep] total units={len(units)}  shard {args.shard_index}/{args.num_shards}: "
          f"{len(shard)} units ({len(done)} done, {len(todo)} todo)")

    if args.plan or not args.execute:
        for u in todo[:20]:
            print(f"  TODO w{u.worker_id} it{u.checkpoint_iteration} "
                  f"tol={u.fixed_tolerance:.4f} seed={u.reset_seed}")
        if len(todo) > 20:
            print(f"  ... and {len(todo) - 20} more")
        if not args.execute:
            print("[ccrs_sweep] plan only (pass --execute to run)")
        return

    failures = 0
    for i, u in enumerate(todo):
        cmd = [
            sys.executable, "-m", "dexpbt_isaaclab.scripts.ccrs_eval",
            "--task", args.task,
            "--run-id", u.run_id,
            "--worker-id", str(u.worker_id),
            "--checkpoint-iteration", str(u.checkpoint_iteration),
            "--checkpoint-frame", str(u.checkpoint_frame),
            "--checkpoint-path", u.checkpoint_path,
            "--fixed-tolerance", str(u.fixed_tolerance),
            "--reset-seed", str(u.reset_seed),
            "--num-envs", str(u.num_envs),
            "--num-episodes", str(u.num_episodes),
            "--deterministic", "true" if u.deterministic else "false",
            "--max-steps", str(u.max_steps),
            "--raw-dir", str(raw_dir),
            "--headless",
        ]
        t0 = time.time()
        print(f"[ccrs_sweep] ({i + 1}/{len(todo)}) w{u.worker_id} it{u.checkpoint_iteration} "
              f"tol={u.fixed_tolerance:.4f} seed={u.reset_seed}")
        try:
            proc = subprocess.run(cmd, timeout=args.per_unit_timeout_s)
            rc = proc.returncode
        except subprocess.TimeoutExpired:
            rc = -9
            print("[ccrs_sweep] unit TIMEOUT", file=sys.stderr)
        dt = time.time() - t0
        if rc != 0 or not is_done(raw_dir, u):
            failures += 1
            print(f"[ccrs_sweep] unit FAILED rc={rc} ({dt:.0f}s)", file=sys.stderr)
        else:
            print(f"[ccrs_sweep] unit ok ({dt:.0f}s)")

    n = aggregate_raw_dir(raw_dir, out_dir / "ccrs_results.jsonl", out_dir / "ccrs_results.csv")
    print(f"[ccrs_sweep] shard finished: {len(todo) - failures}/{len(todo)} ok; "
          f"aggregated {n} rows total")
    if failures:
        sys.exit(1)


if __name__ == "__main__":
    main()
