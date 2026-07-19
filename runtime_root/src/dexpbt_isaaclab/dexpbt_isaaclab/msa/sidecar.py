"""MSA probe sidecar: production wiring of MSAController to a live workspace.

Launched by the MSA / MSA_PERMUTED_ROUTE arm launchers next to the training
supervisors. It is the ONLY writer of msa_* control fields (through the
locked update_curriculum_control API) and never touches policy weights,
optimizer, replay, PBT lineage or any observer-owned curriculum field.

Wiring:
- worker views        <- newest COMMITTED tick (LATEST.commit marker in
                         <workspace>/<policy>/, never a bare newest-file
                         glob) + live tau from the committed tick meta;
- hold/release        <- SIGSTOP/SIGCONT of the worker's training process
                         tree (supervisor pid from policy_NN/pid.txt); a
                         held worker cannot advance the policy while its
                         snapshot is being evaluated (phase barrier);
- snapshot(w, tag)    <- freeze the newest committed checkpoint of worker w
                         read-only under <workspace>/msa_snapshots/; the tag
                         embeds the intervention id, so a resumed cycle
                         reuses the existing frozen file (idempotent);
- evaluate(...)       <- eval_bridge.EvalRequest -> one ccrs_eval subprocess
                         per seed pinned to --eval-device (REQUIRED flag;
                         never defaults onto the trainers' cuda:0), results
                         read back per-seed with canonical tolerances
                         (fail closed on gaps);
- pin_tolerance       <- msa_forced_success_tolerance via locked update
                         (env-side pin: no shrink can fire during a packet);
- commit_tolerance    <- pin -> wait until the live committed tick reports
                         the tolerance -> sync curriculum_resume_snapshot ->
                         clear the pin. Durable across restarts (promotion
                         and stay-revert both go through this);
- train_until         <- PARALLEL barrier polling all listed workers'
                         committed tick ledgers; returns OBSERVED frames;
- save/load state     <- msa_provenance / msa_state_machine / msa_cycle
                         inside CurriculumControlState (single source of
                         truth, rides the existing resume chain).

On startup the sidecar checks for a persisted in-flight cycle and resumes
it idempotently (completed segments are skipped via provenance, snapshots
and eval results are reused); a cycle marked FAILED is not resumed
automatically.

The heavy work happens in child processes; this process is pure python.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

from dexpbt_isaaclab.msa.controller import (
    MSAController,
    PROBE_PACKET_FRAMES,
    WorkerView,
)
from dexpbt_isaaclab.msa.decision import SeedEvalResult
from dexpbt_isaaclab.msa.eval_bridge import (
    EvalRequest,
    FORMAL_NUM_SEEDS,
    PILOT_NUM_SEEDS,
    SnapshotSpec,
    build_seed_manifest,
    load_seed_results,
)

POLL_S = 30.0
COMMIT_TOLERANCE_ATOL = 1e-6


# ---------------------------------------------------------------------------
# committed-tick discovery (LATEST.commit marker, never newest-file glob)
# ---------------------------------------------------------------------------

def latest_committed_meta(policy_dir: Path) -> Optional[dict]:
    """The meta JSON of the newest COMMITTED tick of one worker.

    Uses the LATEST.commit marker written by the observer's staging-commit
    barrier; checkpoint (.pth) and meta (.json) are derived from the SAME
    committed prefix so they can never come from two different ticks.
    Falls back to the newest .pth/.json PAIR if no marker exists (early in
    a run), still requiring both files of one prefix.
    """
    marker = policy_dir / "LATEST.commit"
    prefixes: list[str] = []
    if marker.is_file():
        try:
            commit_dir = str(json.loads(marker.read_text())["commit_dir"])
            prefixes = [commit_dir.split(".", 1)[0]]
        except (json.JSONDecodeError, KeyError, OSError):
            prefixes = []
    if not prefixes:
        prefixes = sorted(
            (p.stem for p in policy_dir.glob("0*.pth")
             if (policy_dir / f"{p.stem}.json").is_file()),
            reverse=True)
    for prefix in prefixes:
        meta_path = policy_dir / f"{prefix}.json"
        ckpt_path = policy_dir / f"{prefix}.pth"
        if not (meta_path.is_file() and ckpt_path.is_file()):
            continue
        try:
            meta = json.loads(meta_path.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        meta["_committed_prefix"] = prefix
        meta["_committed_ckpt"] = str(ckpt_path)
        return meta
    return None


def _worker_views(workspace: Path, num_policies: int) -> list[WorkerView]:
    views = []
    for w in range(num_policies):
        meta = latest_committed_meta(workspace / f"{w:03d}")
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


# ---------------------------------------------------------------------------
# hold / release: SIGSTOP/SIGCONT the training process tree
# ---------------------------------------------------------------------------

def _descendants(pid: int) -> list[int]:
    out = []
    try:
        kids = subprocess.run(["pgrep", "-P", str(pid)],
                              capture_output=True, text=True)
        for line in kids.stdout.split():
            k = int(line)
            out.append(k)
            out.extend(_descendants(k))
    except (ValueError, OSError):
        pass
    return out


class WorkerHolds:
    """Pause/resume worker training process trees (idempotent).

    NOTE: correctness of SIGSTOP against a live Isaac trainer is NOT yet
    GPU-verified; the launch checklist requires a supervised first hold.
    """

    def __init__(self, run_dir: Path) -> None:
        self.run_dir = Path(run_dir)
        self._held: dict[int, list[int]] = {}

    def _pids(self, worker_id: int) -> list[int]:
        pid_file = self.run_dir / f"policy_{worker_id:02d}" / "pid.txt"
        if not pid_file.is_file():
            raise FileNotFoundError(f"no pid file for worker {worker_id}: "
                                    f"{pid_file}")
        root = int(pid_file.read_text().strip())
        return [root] + _descendants(root)

    def hold(self, worker_ids: list[int]) -> None:
        for w in worker_ids:
            if w in self._held:
                continue
            pids = self._pids(w)
            # stop children first so the supervisor cannot respawn mid-hold
            for pid in reversed(pids):
                try:
                    os.kill(pid, signal.SIGSTOP)
                except ProcessLookupError:
                    pass
            self._held[w] = pids

    def release(self, worker_ids: list[int]) -> None:
        for w in worker_ids:
            pids = self._held.pop(w, None)
            if pids is None:
                continue
            for pid in pids:
                try:
                    os.kill(pid, signal.SIGCONT)
                except ProcessLookupError:
                    pass

    def release_all(self) -> None:
        self.release(sorted(self._held.keys()))


# ---------------------------------------------------------------------------
# controller wiring
# ---------------------------------------------------------------------------

def build_controller(workspace: Path, *, task: str, rho_threshold: float,
                     permuted_route: bool, num_seeds: int,
                     eval_num_envs: int, eval_device: str,
                     python_exe: str,
                     packet_frames: int = PROBE_PACKET_FRAMES,
                     ) -> tuple[MSAController, WorkerHolds]:
    manifest_id, seeds = build_seed_manifest(num_seeds)
    snap_dir = workspace / "msa_snapshots"
    raw_dir = workspace / "msa_eval_raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    control_dir = workspace / "curriculum_control"
    run_dir = workspace.parent
    holds = WorkerHolds(run_dir)

    snapshots: dict[str, SnapshotSpec] = {}

    def snapshot(worker_id: int, tag: str) -> str:
        """Freeze the newest COMMITTED checkpoint; idempotent per tag."""
        snap_id = f"w{worker_id}_{tag}"
        existing = snap_dir / f"{snap_id}.pth"
        meta = latest_committed_meta(workspace / f"{worker_id:03d}")
        if meta is None:
            raise FileNotFoundError(
                f"no committed checkpoint for worker {worker_id}")
        if existing.is_file():
            spec = SnapshotSpec(snapshot_id=snap_id, worker_id=worker_id,
                                frame=int(meta.get("frame", 0)),
                                path=str(existing))
        else:
            spec = SnapshotSpec.freeze(
                Path(meta["_committed_ckpt"]), snap_id, worker_id,
                int(meta.get("frame", 0)), snap_dir)
        snapshots[snap_id] = spec
        return snap_id

    def evaluate(worker_id: int, snapshot_id: str, tau: float) -> SeedEvalResult:
        spec = snapshots[snapshot_id]
        req = EvalRequest(snapshot=spec, tau=tau, manifest_id=manifest_id,
                          seeds=seeds, task=task,
                          run_id=f"msa_{run_dir.name}",
                          raw_dir=str(raw_dir), eval_device=eval_device,
                          num_envs=eval_num_envs,
                          num_episodes=eval_num_envs)
        env = req.build_env()
        for cmd in req.build_commands(python_exe):
            # ccrs_eval exits 0 immediately when the unit is already done,
            # so a resumed cycle never re-runs completed evals.
            subprocess.run(cmd, check=True, env=env)
        return load_seed_results(raw_dir, spec, tau, manifest_id, seeds)

    def _control_path(worker_id: int) -> Path:
        return control_dir / f"{worker_id:03d}.json"

    def pin_tolerance(worker_id: int, tol: float) -> None:
        from dexpbt_isaaclab.pbt.pbt_curriculum_p1 import (
            update_curriculum_control,
        )

        def _mutate(control):
            control.msa_enabled = True
            control.msa_forced_success_tolerance = float(tol)

        update_curriculum_control(_control_path(worker_id), _mutate)

    def commit_tolerance(worker_id: int, tol: float,
                         max_wait_s: float = 3600.0) -> None:
        """Durably make tol the worker's live tolerance, then clear the pin.

        pin -> wait for a committed tick reporting the tolerance -> sync
        curriculum_resume_snapshot (so a supervisor crash-restart resumes
        the SAME tolerance) -> clear the pin. The env keeps the value after
        the pin clears because the mastery curriculum only ever shrinks
        from the current live tolerance.

        MUST be called with the worker RUNNING (the controller releases
        holds before committing); a bounded wait guards against a dead
        trainer during crash cleanup.
        """
        from dexpbt_isaaclab.pbt.pbt_curriculum_p1 import (
            update_curriculum_control,
            write_curriculum_resume_snapshot,
        )

        pin_tolerance(worker_id, tol)
        policy_ws = workspace / f"{worker_id:03d}"
        deadline = time.monotonic() + float(max_wait_s)
        while True:
            meta = latest_committed_meta(policy_ws)
            cur = (meta or {}).get("curriculum", {})
            live = float(cur.get("success_tolerance", float("nan")))
            if abs(live - float(tol)) <= COMMIT_TOLERANCE_ATOL:
                break
            if time.monotonic() > deadline:
                raise TimeoutError(
                    f"worker {worker_id}: live tolerance did not reach "
                    f"{tol} within {max_wait_s}s (pin left in place so the "
                    "next tick still applies it)")
            time.sleep(POLL_S)
        snapshot_cur = dict(cur)
        snapshot_cur["success_tolerance"] = float(tol)
        snapshot_cur["last_curriculum_update"] = int(
            cur.get("frame_since_restart", cur.get("last_curriculum_update", 0)))
        write_curriculum_resume_snapshot(
            run_dir / f"policy_{worker_id:02d}",
            curriculum=snapshot_cur,
            env_frames=int((meta or {}).get("frame", 0)),
            policy_idx=worker_id,
        )

        def _clear(control):
            control.msa_forced_success_tolerance = 0.0

        update_curriculum_control(_control_path(worker_id), _clear)

    def train_until(targets: dict[int, int]) -> dict[int, int]:
        """PARALLEL barrier over committed tick ledgers; observed frames."""
        observed: dict[int, int] = {}
        pending = dict(targets)
        while pending:
            for w in list(pending.keys()):
                meta = latest_committed_meta(workspace / f"{w:03d}") or {}
                frame = int(meta.get("frame", 0))
                if frame >= pending[w]:
                    observed[w] = frame          # OBSERVED, never start+n
                    del pending[w]
            if pending:
                time.sleep(POLL_S)
        return observed

    def save_state(worker_id: int, prov: Optional[dict],
                   machine: Optional[dict], cycle: Optional[dict]) -> None:
        from dexpbt_isaaclab.pbt.pbt_curriculum_p1 import (
            update_curriculum_control,
        )

        def _mutate(control):
            control.msa_enabled = True
            if prov is not None:
                control.msa_provenance = prov
            if machine is not None:
                control.msa_state_machine = machine
            if cycle is not None:
                control.msa_cycle = cycle

        update_curriculum_control(_control_path(worker_id), _mutate)

    def load_state(worker_id: int):
        from dexpbt_isaaclab.pbt.pbt_curriculum_p1 import (
            read_curriculum_control,
        )
        control = read_curriculum_control(_control_path(worker_id))
        if control is None:
            return None, None, None
        return (control.msa_provenance, control.msa_state_machine,
                control.msa_cycle)

    ctl = MSAController(
        evaluate=evaluate, snapshot=snapshot,
        retention_threshold=rho_threshold,
        evaluator_manifest_id=manifest_id,
        pin_tolerance=pin_tolerance,
        commit_tolerance=commit_tolerance,
        hold_workers=holds.hold,
        release_workers=holds.release,
        train_until=train_until,
        save_state=save_state,
        load_state=load_state,
        packet_frames=packet_frames,
        permuted_route=permuted_route)
    return ctl, holds


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--workspace", required=True)
    ap.add_argument("--arm", required=True,
                    choices=["MSA", "MSA_PERMUTED_ROUTE"])
    ap.add_argument("--task", default="DexPBT-Regrasping-KukaAllegro-Direct-v0")
    ap.add_argument("--rho-threshold", type=float, required=True)
    ap.add_argument("--permuted-route", action="store_true")
    ap.add_argument("--num-policies", type=int, default=6)
    ap.add_argument("--num-seeds", type=int, default=PILOT_NUM_SEEDS,
                    help=f"pilot default {PILOT_NUM_SEEDS}; formal "
                         f"{FORMAL_NUM_SEEDS} must be requested explicitly")
    ap.add_argument("--eval-num-envs", type=int, default=512)
    ap.add_argument("--eval-device", required=True,
                    help="GPU for the eval subprocesses (e.g. cuda:7). "
                         "REQUIRED: never defaults onto the trainers' GPUs")
    ap.add_argument("--once", action="store_true",
                    help="run at most one probe cycle then exit")
    args = ap.parse_args(argv)

    workspace = Path(args.workspace)
    permuted = args.permuted_route or args.arm == "MSA_PERMUTED_ROUTE"
    ctl, holds = build_controller(
        workspace, task=args.task, rho_threshold=args.rho_threshold,
        permuted_route=permuted, num_seeds=args.num_seeds,
        eval_num_envs=args.eval_num_envs, eval_device=args.eval_device,
        python_exe=sys.executable)

    def _write_cycle_summary(result) -> None:
        out = workspace / "msa_cycles"
        out.mkdir(exist_ok=True)
        (out / f"{result.intervention_id}.json").write_text(
            json.dumps({
                "intervention_id": result.intervention_id,
                "due_workers": result.due_workers,
                "tau_by_worker": result.tau_by_worker,
                "tau_plus_by_worker": result.tau_plus_by_worker,
                "l_mid_by_worker": result.l_mid_by_worker,
                "panel_stat": result.panel_stat,
                "retention_by_worker": result.retention_by_worker,
                "promoted": result.promoted,
                "first_blocking_gate": result.first_blocking_gate,
                "resumed": result.resumed,
            }, indent=2, default=str))

    try:
        while True:
            views = _worker_views(workspace, args.num_policies)
            all_ids = [v.worker_id for v in views]
            pending = ctl.pending_cycle(all_ids)
            if pending is not None:
                print(f"[msa-sidecar] resuming in-flight cycle "
                      f"{pending.get('intervention_id')}")
                result = ctl.run_cycle(views, resume_cycle=pending)
                _write_cycle_summary(result)
            else:
                due = ctl.due_workers(views)
                if due:
                    result = ctl.run_cycle(views)
                    _write_cycle_summary(result)
            if args.once:
                return 0
            time.sleep(POLL_S)
    finally:
        # never leave a worker stopped if the sidecar dies
        holds.release_all()


if __name__ == "__main__":
    raise SystemExit(main())
