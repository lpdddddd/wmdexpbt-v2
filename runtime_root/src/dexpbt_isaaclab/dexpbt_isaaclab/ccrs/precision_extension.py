#!/usr/bin/env python3
"""p1 fixed-precision extension: plan/launch 16 eval units down to 1 cm.

Goal: decide whether the p1 tip checkpoint (worker 1, iteration 160, 5B
frames) of the completed base-retention run already has high-nsucc
capability at a FIXED 1 cm tolerance.

Design (everything identical to the original CCRS protocol):
- checkpoint resolved from the existing CCRS manifest (never guessed);
- NEW tolerances only: 0.016 / 0.014 / 0.012 / 0.010 m — the existing
  0.018 m results are reused read-only and never re-run;
- the original 4 seeds (10001..10004), policy_only, deterministic,
  512 envs, 512 first episodes, frozen curriculum, paired reset
  fingerprints, max_consecutive_successes=50, terminal snapshot hook;
- one subprocess per unit via dexpbt_isaaclab.scripts.ccrs_eval (unchanged:
  no optimizer / PBT / curriculum update / snippet / WM rerank / MPPI);
- results go to a SEPARATE directory (results/analysis/
  ccrs_p1_precision_extension/) so the original ccrs_base_retention outputs
  are never modified;
- resumable: completed units are skipped via the strict is_done check;
- shards (--execute) only run units and write raw JSON — they never
  aggregate. Aggregation happens exactly once via --aggregate-only, called
  by the launcher after all shards exited successfully (avoids concurrent
  writes to extension_results.jsonl/csv + aggregation_provenance.json).

Default mode is --plan (dry run): builds the unit manifest, runs the full
preflight checklist, writes extension_manifest.csv + DRY_RUN_CHECKLIST.md
and exits WITHOUT touching any GPU. --execute is required to launch.
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

from dexpbt_isaaclab.ccrs.eval_protocol import (
    DEFAULT_MAX_CONSECUTIVE_SUCCESSES,
    DEFAULT_SEEDS,
    DEFAULT_TOLERANCES_M,
    EVALUATION_MODE_POLICY_ONLY,
    EvalUnit,
    METRIC_PROTOCOL_VERSION,
    RESET_PROTOCOL_VERSION,
    aggregate_raw_dir,
    build_units,
    is_done,
)

# Target checkpoint: p1 tip of the completed base-retention 5B run.
P1_TIP_WORKER_ID = 1
P1_TIP_ITERATION = 160
P1_TIP_FRAME = 5_000_000_000

# NEW fixed tolerances (meters). 0.018 is deliberately absent: those cells
# already exist in ccrs_base_retention and must not be re-run.
EXTENSION_TOLERANCES_M = [0.016, 0.014, 0.012, 0.010]

EXPECTED_NUM_UNITS = len(EXTENSION_TOLERANCES_M) * len(DEFAULT_SEEDS)  # 16

# Estimated wall time per unit, from the completed 736-unit sweep logs
# (mean ~125 s per unit on one RTX-class GPU, one Isaac subprocess each).
EST_SECONDS_PER_UNIT = 150

MANIFEST_COLUMNS = [
    "run_id", "worker_id", "checkpoint_iteration", "checkpoint_frame",
    "checkpoint_path", "fixed_tolerance_m", "fixed_tolerance_cm",
    "reset_seed", "evaluation_mode", "deterministic", "num_envs",
    "num_episodes", "max_steps", "max_consecutive_successes",
    "reset_protocol_version", "metric_protocol_version",
    "unit_key", "config_hash",
]


def resolve_p1_tip_checkpoint(manifest_csv: Path) -> dict:
    """Resolve the p1 tip checkpoint row from the existing CCRS manifest.

    Never guesses paths: requires exactly one manifest row with
    worker_id=1, iteration=160 that is not marked CHECKPOINT_MISSING.
    """
    matches = []
    with manifest_csv.open() as f:
        for r in csv.DictReader(f):
            if (int(r["worker_id"]) == P1_TIP_WORKER_ID
                    and int(r["iteration"]) == P1_TIP_ITERATION):
                if r.get("notes") and "CHECKPOINT_MISSING" in r["notes"]:
                    continue
                matches.append(r)
    if len(matches) != 1:
        raise ValueError(
            f"expected exactly 1 manifest row for p{P1_TIP_WORKER_ID} "
            f"it{P1_TIP_ITERATION}, found {len(matches)} in {manifest_csv}"
        )
    r = matches[0]
    if int(r["frame"]) != P1_TIP_FRAME:
        raise ValueError(
            f"p1 tip frame mismatch: manifest says {r['frame']}, "
            f"expected {P1_TIP_FRAME}"
        )
    return {
        "worker_id": int(r["worker_id"]),
        "iteration": int(r["iteration"]),
        "frame": int(r["frame"]),
        "checkpoint_path": r["checkpoint_path"],
        "run_id": r["run_id"],
        "roles": r.get("roles", ""),
        "file_size_bytes": int(r.get("file_size_bytes", -1)),
    }


def build_extension_units(ckpt: dict) -> list[EvalUnit]:
    """16 units: 4 new tolerances x 4 original seeds, original protocol."""
    return build_units(
        run_id=ckpt["run_id"],
        checkpoints=[ckpt],
        tolerances_m=EXTENSION_TOLERANCES_M,
        seeds=DEFAULT_SEEDS,
        num_envs=512,
        num_episodes=512,
        evaluation_mode=EVALUATION_MODE_POLICY_ONLY,
        deterministic=True,
        max_steps=200_000,
        max_consecutive_successes=DEFAULT_MAX_CONSECUTIVE_SUCCESSES,
    )


@dataclass
class PreflightResult:
    ok: bool
    checks: list[tuple[str, bool, str]] = field(default_factory=list)

    def add(self, name: str, passed: bool, detail: str = "") -> None:
        self.checks.append((name, passed, detail))
        if not passed:
            self.ok = False


def preflight(
    units: list[EvalUnit],
    ckpt: dict,
    out_dir: Path,
    base_ccrs_dir: Path,
    eval_script: Path | None = None,
) -> PreflightResult:
    """All launch-blocking validations. Pure CPU: no GPU, no Isaac import."""
    res = PreflightResult(ok=True)

    p = Path(ckpt["checkpoint_path"])
    res.add("checkpoint_exists", p.is_file(), str(p))
    if p.is_file() and ckpt.get("file_size_bytes", -1) > 0:
        res.add("checkpoint_size_matches_manifest",
                p.stat().st_size == ckpt["file_size_bytes"],
                f"{p.stat().st_size} vs manifest {ckpt['file_size_bytes']}")

    res.add("unit_count_is_16", len(units) == EXPECTED_NUM_UNITS,
            f"{len(units)} units")

    tols = sorted({u.fixed_tolerance for u in units}, reverse=True)
    res.add("tolerances_are_new_grid",
            tols == sorted(EXTENSION_TOLERANCES_M, reverse=True),
            f"{tols}")
    overlap = [t for t in tols
               if any(abs(t - g) < 1e-9 for g in DEFAULT_TOLERANCES_M)]
    res.add("no_overlap_with_existing_grid_incl_0018", not overlap,
            f"overlap={overlap}")
    res.add("tolerances_in_meters",
            all(0.001 <= t <= 0.2 for t in tols),
            "meter-range sanity (0.001..0.2)")

    seeds = sorted({u.reset_seed for u in units})
    res.add("seeds_match_original_ccrs", seeds == sorted(DEFAULT_SEEDS),
            f"{seeds}")

    res.add("all_policy_only",
            all(u.evaluation_mode == EVALUATION_MODE_POLICY_ONLY for u in units))
    res.add("all_deterministic", all(u.deterministic for u in units))
    res.add("envs_512_episodes_512",
            all(u.num_envs == 512 and u.num_episodes == 512 for u in units))
    res.add("max_consecutive_successes_50",
            all(u.max_consecutive_successes == 50 for u in units))
    res.add("reset_protocol_matches_original",
            all(u.reset_protocol_version == RESET_PROTOCOL_VERSION
                and u.metric_protocol_version == METRIC_PROTOCOL_VERSION
                for u in units),
            f"{RESET_PROTOCOL_VERSION} / {METRIC_PROTOCOL_VERSION}")
    res.add("single_checkpoint_p1_tip",
            all(u.worker_id == P1_TIP_WORKER_ID
                and u.checkpoint_iteration == P1_TIP_ITERATION
                and u.checkpoint_frame == P1_TIP_FRAME for u in units))
    res.add("unit_keys_unique",
            len({u.unit_key() for u in units}) == len(units))

    out_res = out_dir.resolve()
    base_res = base_ccrs_dir.resolve()
    res.add("output_dir_separate_from_base_ccrs",
            out_res != base_res and base_res not in out_res.parents
            and not str(out_res).startswith(str(base_res) + "/"),
            f"{out_res} vs {base_res}")

    # Evaluator source invariants: the unchanged ccrs_eval is policy-only
    # with a frozen curriculum and the v2 reset protocol wiring.
    if eval_script is None:
        eval_script = (Path(__file__).resolve().parent.parent
                       / "scripts" / "ccrs_eval.py")
    src = eval_script.read_text() if eval_script.is_file() else ""
    res.add("eval_script_exists", bool(src), str(eval_script))
    for required in (
        "last_curriculum_update = 10 ** 9",
        'assert not hasattr(agent, "optimizer")',
        "FirstEpisodeCollector",
        "extract_reset_fingerprints",
        "TerminalSnapshotHook",
    ):
        res.add(f"eval_src_has[{required[:40]}]", required in src)
    for forbidden in ("PbtObserver", "optimizer.step", "loss.backward"):
        res.add(f"eval_src_lacks[{forbidden}]", forbidden not in src)
    return res


def write_manifest(units: list[EvalUnit], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(MANIFEST_COLUMNS)
        for u in units:
            w.writerow([
                u.run_id, u.worker_id, u.checkpoint_iteration,
                u.checkpoint_frame, u.checkpoint_path,
                f"{u.fixed_tolerance:.4f}", f"{u.fixed_tolerance * 100:.2f}",
                u.reset_seed, u.evaluation_mode, u.deterministic,
                u.num_envs, u.num_episodes, u.max_steps,
                u.max_consecutive_successes,
                u.reset_protocol_version, u.metric_protocol_version,
                u.unit_key(), u.config_hash(),
            ])
    tmp.replace(path)


def write_checklist(res: PreflightResult, units: list[EvalUnit],
                    ckpt: dict, out_dir: Path, num_gpus: int,
                    path: Path) -> None:
    per_gpu = (len(units) + num_gpus - 1) // num_gpus
    est_min = per_gpu * EST_SECONDS_PER_UNIT / 60
    lines = [
        "# p1 fixed-precision extension — dry-run checklist",
        "",
        f"- checkpoint: `{ckpt['checkpoint_path']}`",
        f"  (p{ckpt['worker_id']} it{ckpt['iteration']} frame {ckpt['frame']},"
        f" roles: {ckpt.get('roles', '?')})",
        f"- units: {len(units)} = {len(EXTENSION_TOLERANCES_M)} tolerances "
        f"x {len(DEFAULT_SEEDS)} seeds",
        f"- new tolerances (m): {EXTENSION_TOLERANCES_M} "
        "(0.018 reused read-only from ccrs_base_retention, NOT re-run)",
        f"- seeds: {DEFAULT_SEEDS}",
        "- protocol: policy_only, deterministic, 512 envs, 512 first "
        "episodes, frozen curriculum, paired reset fingerprints, "
        "max_consecutive_successes=50, terminal snapshot hook, "
        f"{RESET_PROTOCOL_VERSION} / {METRIC_PROTOCOL_VERSION}",
        f"- results dir: `{out_dir}` (original ccrs_base_retention outputs "
        "untouched)",
        f"- planned GPUs: {num_gpus}; estimated wall time ≈ "
        f"{per_gpu} units/GPU x {EST_SECONDS_PER_UNIT}s ≈ {est_min:.0f} min",
        "",
        "## Preflight checks",
        "",
        "| check | status | detail |",
        "|---|---|---|",
    ]
    for name, passed, detail in res.checks:
        lines.append(f"| {name} | {'PASS' if passed else '**FAIL**'} "
                     f"| {detail} |")
    lines += [
        "",
        f"**Overall: {'ALL CHECKS PASSED' if res.ok else 'FAILED'}** "
        "(GPU launch requires --execute and all checks passing).",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text("\n".join(lines))
    tmp.replace(path)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="p1 fixed-precision extension to 1 cm (16 units).")
    ap.add_argument("--manifest", required=True,
                    help="existing ccrs_base_retention checkpoint_manifest.csv")
    ap.add_argument("--out-dir", required=True,
                    help="results/analysis/ccrs_p1_precision_extension")
    ap.add_argument("--base-ccrs-dir",
                    default="results/analysis/ccrs_base_retention",
                    help="original CCRS results dir (protected, read-only)")
    ap.add_argument("--task", default="DexPBT-Regrasping-KukaAllegro-Direct-v0")
    ap.add_argument("--num-gpus", type=int, default=2,
                    help="planned GPU count (for the time estimate + launcher)")
    ap.add_argument("--shard-index", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--plan", action="store_true",
                    help="dry run: manifest + checklist only (default)")
    ap.add_argument("--execute", action="store_true",
                    help="actually run units (blocked unless preflight passes)")
    ap.add_argument("--aggregate-only", action="store_true")
    ap.add_argument("--per-unit-timeout-s", type=int, default=3600)
    args = ap.parse_args()

    out_dir = Path(args.out_dir).resolve()
    raw_dir = out_dir / "raw"

    if args.aggregate_only:
        n = aggregate_raw_dir(raw_dir, out_dir / "extension_results.jsonl",
                              out_dir / "extension_results.csv")
        print(f"[precision_extension] aggregated {n} rows")
        return

    ckpt = resolve_p1_tip_checkpoint(Path(args.manifest))
    units = build_extension_units(ckpt)
    res = preflight(units, ckpt, out_dir, Path(args.base_ccrs_dir))

    write_manifest(units, out_dir / "extension_manifest.csv")
    write_checklist(res, units, ckpt, out_dir, args.num_gpus,
                    out_dir / "DRY_RUN_CHECKLIST.md")

    for name, passed, detail in res.checks:
        print(f"  [{'ok' if passed else 'FAIL'}] {name}"
              + (f"  ({detail})" if detail and not passed else ""))
    done = [u for u in units if is_done(raw_dir, u)]
    print(f"[precision_extension] {len(units)} units "
          f"({len(done)} done, {len(units) - len(done)} todo); "
          f"manifest + checklist written to {out_dir}")

    if not args.execute:
        print("[precision_extension] plan only (pass --execute to run)")
        return
    if not res.ok:
        print("[precision_extension] preflight FAILED — refusing to launch",
              file=sys.stderr)
        sys.exit(2)

    shard = [u for i, u in enumerate(units)
             if i % args.num_shards == args.shard_index]
    todo = [u for u in shard if not is_done(raw_dir, u)]
    raw_dir.mkdir(parents=True, exist_ok=True)
    print(f"[precision_extension] shard {args.shard_index}/{args.num_shards}: "
          f"{len(shard)} units, {len(todo)} todo")
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
        print(f"[precision_extension] ({i + 1}/{len(todo)}) "
              f"tol={u.fixed_tolerance:.4f} seed={u.reset_seed}")
        try:
            rc = subprocess.run(cmd, timeout=args.per_unit_timeout_s).returncode
        except subprocess.TimeoutExpired:
            rc = -9
            print("[precision_extension] unit TIMEOUT", file=sys.stderr)
        dt = time.time() - t0
        if rc != 0 or not is_done(raw_dir, u):
            failures += 1
            print(f"[precision_extension] unit FAILED rc={rc} ({dt:.0f}s)",
                  file=sys.stderr)
        else:
            print(f"[precision_extension] unit ok ({dt:.0f}s)")
    # Shards NEVER aggregate: two shards would race on
    # extension_results.jsonl/csv + aggregation_provenance.json. The
    # launcher's main process runs --aggregate-only exactly once after ALL
    # shards exited successfully; raw JSONs stay for resume on failure.
    print(f"[precision_extension] shard finished: "
          f"{len(todo) - failures}/{len(todo)} ok "
          "(aggregation deferred to the launcher's single --aggregate-only)")
    if failures:
        sys.exit(1)


if __name__ == "__main__":
    main()
