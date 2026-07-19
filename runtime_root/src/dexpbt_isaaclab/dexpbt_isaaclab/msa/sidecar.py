"""MSA probe sidecar: production wiring of MSAController to a live workspace.

Launched by the MSA / MSA_PERMUTED_ROUTE arm launchers next to the training
supervisor. It is the ONLY writer of msa_* state and of
forced_success_tolerance during probes; it never touches policy weights,
optimizer, replay, PBT lineage or any other curriculum field.

Wiring:
- worker views      <- newest tick meta JSON in <workspace>/<policy>/
                       (worker-local monotonic frame ledger) + live tau from
                       curriculum_control/<policy>.json / tick meta;
- snapshot(w, tag)  <- freeze the newest step-boundary checkpoint of worker w
                       read-only under <workspace>/msa_snapshots/;
- evaluate(...)     <- eval_bridge.EvalRequest -> one ccrs_eval subprocess
                       per seed (policy-only, deterministic, no writes),
                       results read back per-seed (fail closed on gaps);
- set_forced_tolerance <- writes forced_success_tolerance into the worker's
                       curriculum control JSON via the existing atomic
                       writer (the env already honors this field);
- train_frames      <- poll the tick metas until the worker's ledger
                       advanced by exactly the packet size.

The heavy work happens in child processes; this process is pure python.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from dexpbt_isaaclab.msa.controller import MSAController, WorkerView
from dexpbt_isaaclab.msa.decision import SeedEvalResult
from dexpbt_isaaclab.msa.eval_bridge import (
    EvalRequest,
    FORMAL_NUM_SEEDS,
    SnapshotSpec,
    build_seed_manifest,
    load_seed_results,
)

POLL_S = 30.0


def _newest_meta(policy_dir: Path) -> dict | None:
    metas = sorted(policy_dir.glob("0*.json"))
    for m in reversed(metas):
        try:
            return json.loads(m.read_text())
        except (json.JSONDecodeError, OSError):
            continue
    return None


def _worker_views(workspace: Path, num_policies: int) -> list[WorkerView]:
    views = []
    for w in range(num_policies):
        meta = _newest_meta(workspace / f"{w:03d}")
        if meta is None:
            continue
        cur = meta.get("curriculum", {})
        views.append(WorkerView(
            worker_id=w,
            lineage_id=meta.get("lineage_id"),
            active_tau=float(cur.get("success_tolerance", float("nan"))),
            frame=int(meta.get("frame", 0)),
            source_checkpoint_id=str(meta.get("checkpoint", "")),
        ))
    return views


def build_controller(workspace: Path, *, task: str, rho_threshold: float,
                     permuted_route: bool, num_seeds: int,
                     eval_num_envs: int, python_exe: str) -> MSAController:
    manifest_id, seeds = build_seed_manifest(num_seeds)
    snap_dir = workspace / "msa_snapshots"
    raw_dir = workspace / "msa_eval_raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    control_dir = workspace / "curriculum_control"

    snapshots: dict[str, SnapshotSpec] = {}

    def snapshot(worker_id: int, tag: str) -> str:
        ckpts = sorted((workspace / f"{worker_id:03d}").glob("0*.pth"))
        if not ckpts:
            raise FileNotFoundError(f"no checkpoint for worker {worker_id}")
        meta = _newest_meta(workspace / f"{worker_id:03d}") or {}
        snap_id = f"w{worker_id}_{tag}_{int(time.time())}"
        spec = SnapshotSpec.freeze(ckpts[-1], snap_id, worker_id,
                                   int(meta.get("frame", 0)), snap_dir)
        snapshots[snap_id] = spec
        return snap_id

    def evaluate(worker_id: int, snapshot_id: str, tau: float) -> SeedEvalResult:
        spec = snapshots[snapshot_id]
        req = EvalRequest(snapshot=spec, tau=tau, manifest_id=manifest_id,
                          seeds=seeds, task=task,
                          run_id=f"msa_{workspace.parent.name}",
                          raw_dir=str(raw_dir), num_envs=eval_num_envs,
                          num_episodes=eval_num_envs)
        for cmd in req.build_commands(python_exe):
            subprocess.run(cmd, check=True)
        return load_seed_results(raw_dir, spec, tau, manifest_id, seeds)

    def set_forced_tolerance(worker_id: int, tol: float) -> None:
        # go through the existing atomic control writer; only this one field
        from dexpbt_isaaclab.pbt.pbt_curriculum_p1 import (
            CurriculumControlState,
            read_curriculum_control,
            write_curriculum_control,
        )
        path = control_dir / f"{worker_id:03d}.json"
        control = read_curriculum_control(path) or CurriculumControlState()
        control.forced_success_tolerance = float(tol)
        control.msa_enabled = True
        write_curriculum_control(path, control)

    def train_frames(worker_id: int, n_frames: int, tau: float) -> int:
        meta = _newest_meta(workspace / f"{worker_id:03d}") or {}
        start = int(meta.get("frame", 0))
        target = start + n_frames
        while True:
            meta = _newest_meta(workspace / f"{worker_id:03d}") or {}
            frame = int(meta.get("frame", 0))
            if frame >= target:
                # ledger is tick-quantized; charge exactly the packet
                return start + n_frames
            time.sleep(POLL_S)

    return MSAController(
        state_dir=control_dir, evaluate=evaluate, snapshot=snapshot,
        retention_threshold=rho_threshold,
        evaluator_manifest_id=manifest_id,
        permuted_route=permuted_route,
        set_forced_tolerance=set_forced_tolerance,
        train_frames=train_frames)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--workspace", required=True)
    ap.add_argument("--arm", required=True,
                    choices=["MSA", "MSA_PERMUTED_ROUTE"])
    ap.add_argument("--task", default="DexPBT-Regrasping-KukaAllegro-Direct-v0")
    ap.add_argument("--rho-threshold", type=float, required=True)
    ap.add_argument("--permuted-route", action="store_true")
    ap.add_argument("--num-policies", type=int, default=6)
    ap.add_argument("--num-seeds", type=int, default=FORMAL_NUM_SEEDS)
    ap.add_argument("--eval-num-envs", type=int, default=512)
    ap.add_argument("--once", action="store_true",
                    help="run at most one probe cycle then exit")
    args = ap.parse_args(argv)

    workspace = Path(args.workspace)
    permuted = args.permuted_route or args.arm == "MSA_PERMUTED_ROUTE"
    ctl = build_controller(
        workspace, task=args.task, rho_threshold=args.rho_threshold,
        permuted_route=permuted, num_seeds=args.num_seeds,
        eval_num_envs=args.eval_num_envs, python_exe=sys.executable)

    while True:
        views = _worker_views(workspace, args.num_policies)
        due = ctl.due_workers(views)
        if due:
            result = ctl.run_cycle([v for v in views
                                    if v.worker_id in due])
            out = workspace / "msa_cycles"
            out.mkdir(exist_ok=True)
            (out / f"{result.intervention_id}.json").write_text(
                json.dumps({
                    "intervention_id": result.intervention_id,
                    "due_workers": result.due_workers,
                    "l_mid_by_worker": result.l_mid_by_worker,
                    "panel_stat": result.panel_stat,
                    "retention_by_worker": result.retention_by_worker,
                    "promoted": result.promoted,
                    "first_blocking_gate": result.first_blocking_gate,
                }, indent=2, default=str))
        if args.once:
            return 0
        time.sleep(POLL_S)


if __name__ == "__main__":
    raise SystemExit(main())
