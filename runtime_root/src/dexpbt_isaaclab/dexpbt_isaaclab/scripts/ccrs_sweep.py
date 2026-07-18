#!/usr/bin/env python3
"""CCRS sweep scheduler: run ccrs_eval units sequentially (per shard).

No Isaac import here — each unit runs in its own subprocess (Isaac cannot
survive env.close() + re-create in one process). Completed units are skipped
via the strict per-unit raw JSON validation (is_done), so the sweep is
resumable and can be sharded across GPUs by --shard-index/--num-shards
(units are interleaved deterministically).

Unit sources:
1. capability grid: manifest checkpoints (role filter) x --tolerances grid;
2. exact transitions (--transitions CSV): for every transition, the pre and
   post checkpoints are evaluated at EXACTLY the real pre/post tolerances
   (a, b) — required for exact D/F/C, never snapped to the grid.

Plan-only mode (--plan) prints the unit table and exits without touching
any GPU. This is the default safety behaviour unless --execute is passed.

Usage (plan only, safe):
  python -m dexpbt_isaaclab.scripts.ccrs_sweep \
    --manifest results/analysis/ccrs_base_retention/checkpoint_manifest.csv \
    --roles tip,peak,target \
    --transitions results/analysis/ccrs_base_retention/tolerance_transitions.csv \
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
import subprocess
import sys
import time
from pathlib import Path

from dexpbt_isaaclab.ccrs.eval_protocol import (
    DEFAULT_SEEDS,
    DEFAULT_TOLERANCES_M,
    build_units,
    is_done,
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


def load_exact_transition_checkpoints(
    manifest_csv: Path,
    transitions_csv: Path,
    workers_filter: list[int] | None = None,
) -> list[dict]:
    """Checkpoint rows for exact D/F/C cells.

    For every transition (pre_iter, post_iter, a, b): both checkpoints get a
    per-checkpoint "tolerances" list containing exactly {a, b}. Tolerances
    are accumulated when a checkpoint participates in several transitions.
    """
    if not transitions_csv.is_file():
        return []
    # (worker, iter) -> manifest row (for checkpoint_path / frame / run_id)
    all_rows: dict[tuple[int, int], dict] = {}
    with manifest_csv.open() as f:
        for r in csv.DictReader(f):
            if r.get("notes") and "CHECKPOINT_MISSING" in r["notes"]:
                continue
            all_rows[(int(r["worker_id"]), int(r["iteration"]))] = r

    acc: dict[tuple[int, int], dict] = {}
    with transitions_csv.open() as f:
        for t in csv.DictReader(f):
            w = int(t["worker_id"])
            if workers_filter and w not in workers_filter:
                continue
            a = float(t["pre_tolerance"])
            b = float(t["post_tolerance"])
            for it in (int(t["pre_iteration"]), int(t["post_iteration"])):
                mrow = all_rows.get((w, it))
                if mrow is None:
                    print(f"[ccrs_sweep] WARN: transition checkpoint p{w} it{it} "
                          "not in manifest; exact cells will be missing",
                          file=sys.stderr)
                    continue
                key = (w, it)
                if key not in acc:
                    acc[key] = {
                        "worker_id": w,
                        "iteration": it,
                        "frame": int(mrow["frame"]),
                        "checkpoint_path": mrow["checkpoint_path"],
                        "run_id": mrow["run_id"],
                        "tolerances": [],
                    }
                for tol in (a, b):
                    if not any(abs(tol - x) < 1e-9 for x in acc[key]["tolerances"]):
                        acc[key]["tolerances"].append(tol)
    return list(acc.values())


def main() -> None:
    ap = argparse.ArgumentParser(description="CCRS sweep scheduler (subprocess driver).")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--task", required=True)
    ap.add_argument("--roles", default="tip,peak,target",
                    help="Comma list of role prefixes to include from manifest "
                         "for the capability grid")
    ap.add_argument("--transitions", default=None,
                    help="tolerance_transitions.csv for exact D/F/C cells "
                         "(pre/post checkpoints at real a/b tolerances)")
    ap.add_argument("--transition-workers", default=None,
                    help="Comma list of worker ids to restrict exact "
                         "transition units (e.g. 1,4,5)")
    ap.add_argument("--tolerances", default=",".join(str(t) for t in DEFAULT_TOLERANCES_M),
                    help="Comma list of grid tolerances in METERS")
    ap.add_argument("--seeds", default=",".join(str(s) for s in DEFAULT_SEEDS))
    ap.add_argument("--num-envs", type=int, default=512)
    ap.add_argument("--num-episodes", type=int, default=512,
                    help="Must be <= num-envs (paired-reset protocol)")
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

    if args.num_episodes > args.num_envs:
        print(f"[ccrs_sweep] paired-reset protocol requires num_episodes <= "
              f"num_envs (got {args.num_episodes} > {args.num_envs})",
              file=sys.stderr)
        sys.exit(2)

    grid_ckpts = load_manifest_checkpoints(
        Path(args.manifest), [r.strip() for r in args.roles.split(",") if r.strip()]
    )
    trans_ckpts: list[dict] = []
    if args.transitions:
        wf = ([int(x) for x in args.transition_workers.split(",") if x.strip()]
              if args.transition_workers else None)
        trans_ckpts = load_exact_transition_checkpoints(
            Path(args.manifest), Path(args.transitions), wf)

    if not grid_ckpts and not trans_ckpts:
        print("[ccrs_sweep] no checkpoints selected", file=sys.stderr)
        sys.exit(2)
    run_id = (grid_ckpts or trans_ckpts)[0]["run_id"]
    tols = [float(t) for t in args.tolerances.split(",") if t.strip()]
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]

    units = build_units(
        run_id=run_id,
        checkpoints=grid_ckpts + trans_ckpts,  # trans rows carry own tolerances
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

    print(f"[ccrs_sweep] grid_ckpts={len(grid_ckpts)} exact_transition_ckpts="
          f"{len(trans_ckpts)} grid_tolerances={len(tols)} seeds={len(seeds)}")
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
            "--max-consecutive-successes", str(u.max_consecutive_successes),
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
