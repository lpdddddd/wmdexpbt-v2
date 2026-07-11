#!/usr/bin/env python3
"""Fixed-tolerance A/B/C evaluation driver for the 2cm bridge study.

Wraps ``eval_fair.py`` with identical controlled settings across checkpoints.
If checkpoints are missing locally it prints exact runnable commands instead of
failing, and never fabricates results.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

EVAL_MODULE = "dexpbt_isaaclab.scripts.eval_fair"


def build_cmd(workspace, ckpt, tol, out_csv, args):
    return [
        sys.executable, "-m", EVAL_MODULE,
        "--workspace", str(workspace),
        "--method_name", f"p{args.policy_idx:02d}_tol{tol}",
        "--task", args.task,
        "--eval_tolerance_mode", "fixed",
        "--fixed_success_tolerance", str(tol),
        "--num_envs", str(args.num_envs),
        "--num_episodes", str(args.num_episodes),
        "--seed", str(args.seed),
        "--deterministic", "true",
        "--max_consecutive_successes_override", "50",
        "--checkpoint_override", str(ckpt),
        "--output_csv", str(out_csv),
    ]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--workspace", required=True)
    p.add_argument("--task", default="DexPBT-Regrasping-KukaAllegro-Direct-v0")
    p.add_argument("--policy_idx", type=int, default=5)
    p.add_argument("--iterations", default="131,140,145")
    p.add_argument("--fixed_tolerances", default="0.020")
    p.add_argument("--sweep", default="")
    p.add_argument("--num_envs", type=int, default=512)
    p.add_argument("--num_episodes", type=int, default=512)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out_dir", default=None)
    p.add_argument("--dry_run", action="store_true")
    args = p.parse_args()

    ws = Path(args.workspace).resolve()
    out_dir = Path(args.out_dir) if args.out_dir else ws.parent / "eval_abc"
    out_dir.mkdir(parents=True, exist_ok=True)
    iters = [int(x) for x in args.iterations.split(",") if x.strip()]
    tols = [float(x) for x in args.fixed_tolerances.split(",") if x.strip()]
    if args.sweep:
        tols = sorted(set(tols + [float(x) for x in args.sweep.split(",") if x.strip()]), reverse=True)

    planned, missing = [], []
    for it in iters:
        ckpt = ws / f"{args.policy_idx:03d}" / f"{it:06d}.pth"
        for tol in tols:
            out_csv = out_dir / f"p{args.policy_idx:02d}_iter{it}_tol{tol}.csv"
            cmd = build_cmd(ws, ckpt, tol, out_csv, args)
            planned.append({"iter": it, "tol": tol, "checkpoint": str(ckpt),
                            "output_csv": str(out_csv), "cmd": " ".join(cmd), "exists": ckpt.exists()})
            if not ckpt.exists():
                missing.append(str(ckpt))

    (out_dir / "fixed_tol_eval_comparison.json").write_text(json.dumps(planned, indent=2))
    if missing:
        print("[fixed-eval] missing checkpoints; run these once available:")
        for row in planned:
            print(row["cmd"])
        return
    if args.dry_run:
        for row in planned:
            print(row["cmd"])
        return
    for row in planned:
        print(f"[fixed-eval] {row['cmd']}")
        subprocess.run(row["cmd"].split(), check=False)
    print(f"[fixed-eval] wrote plan -> {out_dir/'fixed_tol_eval_comparison.json'}")


if __name__ == "__main__":
    main()
