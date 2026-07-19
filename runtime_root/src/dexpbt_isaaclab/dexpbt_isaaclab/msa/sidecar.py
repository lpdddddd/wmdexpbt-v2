"""MSA probe sidecar: production wiring of MSAController to a live workspace.

Launched by the MSA / MSA_PERMUTED_ROUTE arm launchers next to the training
supervisors. It is the ONLY writer of msa_* control fields (through the
locked update_curriculum_control API) and never touches policy weights,
optimizer, replay, PBT lineage or any observer-owned curriculum field.

Readiness barrier: the sidecar REQUIRES --readiness-marker (the
FIRST_TICK_VALIDATED.json written by continuation_guard once ALL SIX
workers produced a validated first continuation tick) AND
--hold-smoke-marker (the HOLD_SMOKE_PASSED.json written only by a fully
successful run_hold_smoke.sh). Until BOTH markers exist — and worker
views are exactly {0..5} — the sidecar neither starts nor resumes any
probe cycle and performs no holds (it cannot race the hold smoke over
MSA_HOLDS.json). If the guard fails the arm (GUARD_FAILED) or the smoke
fails (HOLD_SMOKE_FAILED.json), the sidecar releases all holds and exits.

Cycle coordination: while a cycle is in flight (from before the first
PRE_EVAL hold, through eval/probe/DECIDE and the durable COMMITTING
phase, until DONE or failure cleanup completes) the controller keeps
<run_dir>/MSA_BUSY.json in place via set_busy; the continuation guard
defers +100M park/cap actions while it exists and the hold watchdog
treats a dead sidecar with a busy marker as fatal even with no holds.

Wiring:
- worker views        <- newest COMMITTED tick (LATEST.commit marker in
                         <workspace>/<policy>/, never a bare newest-file
                         glob) + live tau from the committed tick meta;
                         refresh_view re-reads AFTER a hold so packet start
                         frames can never be one tick stale;
- hold/release        <- SIGSTOP/SIGCONT of the worker's training process
                         tree; hold VERIFIES the processes reached the
                         stopped state; release resumes children before the
                         supervisor/root; held PIDs are persisted to
                         MSA_HOLDS.json so the external hold watchdog (and
                         the guard's stop paths) can SIGCONT them if this
                         process dies;
- snapshot(w, tag)    <- freeze the newest committed checkpoint of worker w
                         read-only under <workspace>/msa_snapshots/ and
                         record the frozen frame in <snap>.meta.json; a
                         resumed cycle reuses the existing frozen file AND
                         its recorded frame (idempotent);
- evaluate(...)       <- eval_bridge.EvalRequest -> ccrs_eval subprocesses
                         with BOUNDED per-device concurrency over
                         --eval-devices (at most one Isaac process per GPU,
                         seeds round-robin, fail-closed cancellation);
- pin_tolerance       <- msa_forced_success_tolerance via locked update
                         (env-side pin: no shrink can fire during a packet);
- commit_tolerance    <- pin -> ONE bounded wait (--commit-max-wait-s) for a
                         live committed tick reporting the tolerance ->
                         sync curriculum_resume_snapshot -> clear the pin.
                         On timeout raises controller.CommitTimeout; the
                         main loop then writes COMMIT_TIMEOUT.json,
                         releases all holds and stops the whole arm
                         (cleanup never waits a second full timeout);
- train_until         <- PARALLEL barrier polling all listed workers'
                         committed tick ledgers; returns OBSERVED frames;
- save/load state     <- msa_provenance / msa_state_machine / msa_cycle
                         inside CurriculumControlState (single source of
                         truth, rides the existing resume chain).

--hold-smoke-only: a standalone single-worker hold smoke (wait for
readiness, hold one worker, verify its committed frame is frozen for
--hold-smoke-seconds, release, verify the next committed tick appears).
No tolerance writes, no cycle, no eval; result in HOLD_SMOKE_RESULT.json;
release is guaranteed in a finally block.

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
    CommitTimeout,
    CycleFailure,
    MSAController,
    PROBE_PACKET_FRAMES,
    WorkerView,
)
from dexpbt_isaaclab.msa.decision import SeedEvalResult
from dexpbt_isaaclab.msa.eval_bridge import (
    EvalRequest,
    FORMAL_NUM_SEEDS,
    SMOKE_NUM_SEEDS,
    SnapshotSpec,
    build_seed_manifest,
    load_seed_results,
    run_seed_jobs,
)

POLL_S = 30.0
COMMIT_TOLERANCE_ATOL = 1e-6
DEFAULT_COMMIT_MAX_WAIT_S = 3600.0
HOLDS_FILE = "MSA_HOLDS.json"
BUSY_FILE = "MSA_BUSY.json"
SMOKE_PASSED_FILE = "HOLD_SMOKE_PASSED.json"
SMOKE_FAILED_FILE = "HOLD_SMOKE_FAILED.json"
NUM_WORKERS = 6


def _atomic_write_json(path: Path, payload: dict) -> None:
    tmp = path.with_suffix(f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(payload, indent=2, default=str))
    os.replace(tmp, path)


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


def _view_from_meta(worker_id: int, meta: dict) -> WorkerView:
    cur = meta.get("curriculum", {})
    return WorkerView(
        worker_id=worker_id,
        lineage_id=meta.get("lineage_id"),
        active_tau=float(cur.get("success_tolerance", float("nan"))),
        frame=int(meta.get("frame", 0)),
        source_checkpoint_id=str(meta.get("checkpoint", "")),
    )


def _worker_views(workspace: Path, num_policies: int) -> list[WorkerView]:
    views = []
    for w in range(num_policies):
        meta = latest_committed_meta(workspace / f"{w:03d}")
        if meta is None:
            continue
        views.append(_view_from_meta(w, meta))
    return views


def readiness_ok(marker_path: Path, views: list[WorkerView],
                 num_policies: int = NUM_WORKERS) -> tuple[bool, str]:
    """Gate for starting/resuming ANY probe cycle: the guard's readiness
    marker must exist AND the committed worker views must be exactly
    {0, .., num_policies-1} (a launcher that brought up only a subset must
    never trigger a partial-panel cycle)."""
    if not marker_path.is_file():
        return False, f"readiness marker missing: {marker_path}"
    try:
        payload = json.loads(marker_path.read_text())
        marker_ids = sorted(int(w["worker_id"]) for w in payload["workers"])
    except (json.JSONDecodeError, KeyError, TypeError, OSError) as exc:
        return False, f"readiness marker unreadable: {exc}"
    if marker_ids != list(range(num_policies)):
        return False, (f"readiness marker covers workers {marker_ids}, "
                       f"expected {list(range(num_policies))}")
    ids = sorted(v.worker_id for v in views)
    if ids != list(range(num_policies)):
        return False, (f"committed worker views {ids} != expected "
                       f"{list(range(num_policies))} (partial launch?)")
    return True, ""


def cycle_gate_ok(readiness_marker: Path, smoke_marker: Path,
                  views: list[WorkerView],
                  num_policies: int = NUM_WORKERS) -> tuple[bool, str]:
    """The FULL gate for starting/resuming any probe cycle: the guard's
    6/6 readiness marker AND the manually-run hold smoke's PASSED marker.
    Until both exist the sidecar idles (it never touches MSA_HOLDS.json in
    that state, so it cannot race the hold smoke)."""
    ready, why = readiness_ok(readiness_marker, views, num_policies)
    if not ready:
        return False, why
    if not smoke_marker.is_file():
        return False, (f"hold smoke gate not passed: {smoke_marker} "
                       "missing (run run_hold_smoke.sh first)")
    return True, ""


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


def _proc_state(pid: int) -> Optional[str]:
    """The one-letter /proc state ('T' = stopped) or None if gone."""
    try:
        text = (Path("/proc") / str(pid) / "stat").read_text()
    except (FileNotFoundError, ProcessLookupError, OSError):
        return None
    # field 3 comes after the parenthesized comm, which may contain spaces
    return text.rsplit(")", 1)[1].split()[0]


def read_holds_file(run_dir: Path) -> dict[int, list[int]]:
    path = Path(run_dir) / HOLDS_FILE
    if not path.is_file():
        return {}
    try:
        raw = json.loads(path.read_text())
        return {int(w): [int(p) for p in pids] for w, pids in raw.items()}
    except (json.JSONDecodeError, ValueError, OSError):
        return {}


class WorkerHolds:
    """Pause/resume worker training process trees (idempotent).

    - hold() SIGSTOPs children first (so the supervisor cannot respawn
      mid-hold) and VERIFIES every surviving PID reached the stopped
      state; a failed verification releases and raises (fail closed);
    - release() SIGCONTs children BEFORE the supervisor/root, so the
      supervisor never observes a still-stopped child on wakeup;
    - held PIDs are persisted to <run_dir>/MSA_HOLDS.json on every hold
      and the record is removed on release, so the external hold watchdog
      (and the guard's stop paths) can SIGCONT them if the sidecar dies
      without running its finally block.

    NOTE: correctness of SIGSTOP against a live Isaac trainer is NOT yet
    GPU-verified; run the --hold-smoke-only mode first (see checklist).
    """

    STOP_VERIFY_TIMEOUT_S = 10.0

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

    def _persist(self) -> None:
        path = self.run_dir / HOLDS_FILE
        if not self._held:
            try:
                path.unlink()
            except FileNotFoundError:
                pass
            return
        tmp = path.with_suffix(f".tmp.{os.getpid()}")
        tmp.write_text(json.dumps(
            {str(w): pids for w, pids in sorted(self._held.items())},
            indent=2))
        os.replace(tmp, path)

    def _verify_stopped(self, pids: list[int]) -> None:
        deadline = time.monotonic() + self.STOP_VERIFY_TIMEOUT_S
        pending = list(pids)
        while pending:
            pending = [p for p in pending
                       if _proc_state(p) not in (None, "T", "t")]
            if not pending:
                return
            if time.monotonic() > deadline:
                raise RuntimeError(
                    f"hold verification failed: pids {pending} did not "
                    "reach stopped state")
            time.sleep(0.2)

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
            try:
                self._verify_stopped(pids)
            except RuntimeError:
                for pid in reversed(pids):      # fail closed: undo the hold
                    try:
                        os.kill(pid, signal.SIGCONT)
                    except ProcessLookupError:
                        pass
                raise
            self._held[w] = pids
            self._persist()

    def release(self, worker_ids: list[int]) -> None:
        for w in worker_ids:
            pids = self._held.pop(w, None)
            if pids is None:
                continue
            # children BEFORE the supervisor/root (pids = [root, *children])
            for pid in reversed(pids):
                try:
                    os.kill(pid, signal.SIGCONT)
                except ProcessLookupError:
                    pass
            self._persist()

    def release_all(self) -> None:
        self.release(sorted(self._held.keys()))


# ---------------------------------------------------------------------------
# controller wiring
# ---------------------------------------------------------------------------

def build_controller(workspace: Path, *, task: str, rho_threshold: float,
                     permuted_route: bool, num_seeds: int,
                     eval_num_envs: int, eval_devices: list[str],
                     python_exe: str,
                     packet_frames: int = PROBE_PACKET_FRAMES,
                     commit_max_wait_s: float = DEFAULT_COMMIT_MAX_WAIT_S,
                     ) -> tuple[MSAController, WorkerHolds]:
    manifest_id, seeds = build_seed_manifest(num_seeds)
    snap_dir = workspace / "msa_snapshots"
    raw_dir = workspace / "msa_eval_raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    control_dir = workspace / "curriculum_control"
    run_dir = workspace.parent
    holds = WorkerHolds(run_dir)
    if not eval_devices:
        raise ValueError("--eval-devices is required (e.g. cuda:6,cuda:7)")

    snapshots: dict[str, SnapshotSpec] = {}

    def snapshot(worker_id: int, tag: str) -> tuple[str, int]:
        """Freeze the newest COMMITTED checkpoint; idempotent per tag.
        Returns (snapshot_id, frame RECORDED at freeze time) — a reused
        snapshot keeps its original frame from <snap>.meta.json, never a
        re-read of the live ledger."""
        snap_id = f"w{worker_id}_{tag}"
        existing = snap_dir / f"{snap_id}.pth"
        side_meta = snap_dir / f"{snap_id}.meta.json"
        if existing.is_file() and side_meta.is_file():
            frozen = json.loads(side_meta.read_text())
            spec = SnapshotSpec(snapshot_id=snap_id, worker_id=worker_id,
                                frame=int(frozen["frame"]),
                                path=str(existing))
        else:
            meta = latest_committed_meta(workspace / f"{worker_id:03d}")
            if meta is None:
                raise FileNotFoundError(
                    f"no committed checkpoint for worker {worker_id}")
            spec = SnapshotSpec.freeze(
                Path(meta["_committed_ckpt"]), snap_id, worker_id,
                int(meta.get("frame", 0)), snap_dir)
            side_meta.write_text(json.dumps({
                "frame": spec.frame,
                "committed_prefix": meta.get("_committed_prefix", ""),
                "source": meta.get("_committed_ckpt", "")}))
        snapshots[snap_id] = spec
        return snap_id, spec.frame

    def evaluate(worker_id: int, snapshot_id: str, tau: float) -> SeedEvalResult:
        spec = snapshots[snapshot_id]
        req = EvalRequest(snapshot=spec, tau=tau, manifest_id=manifest_id,
                          seeds=seeds, task=task,
                          run_id=f"msa_{run_dir.name}",
                          raw_dir=str(raw_dir),
                          eval_device=eval_devices[0],
                          num_envs=eval_num_envs,
                          num_episodes=eval_num_envs)
        # ccrs_eval exits 0 immediately when the unit is already done, so a
        # resumed cycle never re-runs completed evals. Bounded concurrency:
        # at most one Isaac process per eval device, fail closed.
        run_seed_jobs(req.build_commands(python_exe), eval_devices)
        return load_seed_results(raw_dir, spec, tau, manifest_id, seeds)

    def refresh_view(worker_id: int) -> WorkerView:
        meta = latest_committed_meta(workspace / f"{worker_id:03d}")
        if meta is None:
            raise FileNotFoundError(
                f"refresh_view: no committed tick for worker {worker_id}")
        return _view_from_meta(worker_id, meta)

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
                         max_wait_s: float = commit_max_wait_s) -> None:
        """Durably make tol the worker's live tolerance, then clear the pin.

        pin -> ONE bounded wait for a committed tick reporting the
        tolerance -> sync curriculum_resume_snapshot (so a supervisor
        crash-restart resumes the SAME tolerance) -> clear the pin. The env
        keeps the value after the pin clears because the mastery curriculum
        only ever shrinks from the current live tolerance.

        MUST be called with the worker RUNNING (the controller releases
        holds before committing). On timeout raises CommitTimeout — the
        pin stays in place (the next tick would still apply it) and the
        MAIN LOOP stops the arm; cleanup never re-waits.
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
                raise CommitTimeout(
                    worker_id, tol,
                    last_frame=int((meta or {}).get("frame", -1)),
                    last_tau=live, max_wait_s=float(max_wait_s))
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

    def set_busy(info: Optional[dict]) -> None:
        """Maintain <run_dir>/MSA_BUSY.json: exists from before the first
        hold of a cycle until the cycle is DONE or its failure cleanup
        finished. The guard defers park/cap actions and the watchdog treats
        a dead sidecar as fatal while it exists."""
        busy_path = run_dir / BUSY_FILE
        if info is None:
            try:
                busy_path.unlink()
            except FileNotFoundError:
                pass
            return
        _atomic_write_json(busy_path, dict(info))

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
        refresh_view=refresh_view,
        set_busy=set_busy,
        packet_frames=packet_frames,
        permuted_route=permuted_route)
    return ctl, holds


# ---------------------------------------------------------------------------
# hold smoke (single worker, no tolerance writes, no cycle, no eval)
# ---------------------------------------------------------------------------

def hold_smoke(run_dir: Path, workspace: Path, worker_id: int,
               hold_seconds: float, readiness_marker: Path,
               resume_timeout_s: float = 5400.0,
               readiness_timeout_s: float = 24 * 3600.0,
               poll_s: float = POLL_S) -> int:
    """Standalone SIGSTOP/SIGCONT validation of ONE worker.

    wait readiness -> record committed frame -> hold -> wait hold_seconds
    -> confirm the committed frame did NOT advance -> release -> confirm
    the NEXT committed tick appears within resume_timeout_s. Writes
    HOLD_SMOKE_RESULT.json; release is guaranteed (finally).

    Gate semantics: HOLD_SMOKE_PASSED.json is written ATOMICALLY, and ONLY
    when (a) the frame was completely frozen during the hold, (b) the next
    committed tick appeared after release AND (c) residual_holds is empty.
    Any other outcome writes HOLD_SMOKE_FAILED.json, guarantees every PID
    was SIGCONTed (release_all in the finally) and fail-closed stops the
    whole arm. The PASSED marker can never be written on a failure path."""
    result: dict = {"worker_id": int(worker_id),
                    "hold_seconds": float(hold_seconds),
                    "ok": False, "timestamp": time.time()}
    out_path = run_dir / "HOLD_SMOKE_RESULT.json"
    holds = WorkerHolds(run_dir)
    policy_ws = workspace / f"{worker_id:03d}"
    try:
        deadline = time.monotonic() + readiness_timeout_s
        while not readiness_marker.is_file():
            if time.monotonic() > deadline:
                result["error"] = (f"readiness marker never appeared: "
                                   f"{readiness_marker}")
                return 2
            time.sleep(poll_s)
        meta = latest_committed_meta(policy_ws)
        if meta is None:
            result["error"] = f"no committed tick for worker {worker_id}"
            return 2
        frame_before = int(meta.get("frame", -1))
        result["frame_before_hold"] = frame_before
        holds.hold([worker_id])
        result["held"] = True
        time.sleep(float(hold_seconds))
        meta2 = latest_committed_meta(policy_ws) or {}
        frame_during = int(meta2.get("frame", -1))
        result["frame_after_hold_window"] = frame_during
        if frame_during != frame_before:
            result["error"] = (f"frame advanced during hold: "
                               f"{frame_before} -> {frame_during}")
            return 2
        holds.release([worker_id])
        result["released"] = True
        resume_deadline = time.monotonic() + float(resume_timeout_s)
        while True:
            meta3 = latest_committed_meta(policy_ws) or {}
            frame_after = int(meta3.get("frame", -1))
            if frame_after > frame_before:
                result["next_committed_frame"] = frame_after
                break
            if time.monotonic() > resume_deadline:
                result["error"] = (f"no new committed tick within "
                                   f"{resume_timeout_s}s after release")
                return 2
            time.sleep(poll_s)
        result["ok"] = True
        return 0
    finally:
        holds.release_all()          # guaranteed release, success or not
        result["residual_holds"] = {
            str(w): pids for w, pids in read_holds_file(run_dir).items()}
        if result["ok"] and result["residual_holds"]:
            result["ok"] = False
            result["error"] = (f"residual holds after release: "
                               f"{result['residual_holds']}")
        out_path.write_text(json.dumps(result, indent=2))
        if result["ok"]:
            _atomic_write_json(run_dir / SMOKE_PASSED_FILE, {
                "worker_id": int(worker_id),
                "frame_before_hold": result.get("frame_before_hold"),
                "next_committed_frame": result.get("next_committed_frame"),
                "hold_seconds": float(hold_seconds),
                "timestamp": time.time(),
            })
            try:      # a pass supersedes an older (relaunched-arm) failure
                (run_dir / SMOKE_FAILED_FILE).unlink()
            except FileNotFoundError:
                pass
        else:
            # NEVER write the PASSED marker on any failure path; record
            # the failure and fail-closed stop the whole arm (all PIDs
            # were already SIGCONTed by release_all above).
            _atomic_write_json(run_dir / SMOKE_FAILED_FILE, dict(result))
            from dexpbt_isaaclab.msa.continuation_guard import stop_arm
            stop_arm(run_dir,
                     f"hold smoke failed: {result.get('error', 'unknown')}",
                     marker_name=None)


# ---------------------------------------------------------------------------
# main loop
# ---------------------------------------------------------------------------

def _stop_markers_present(run_dir: Path) -> Optional[str]:
    for name in ("GUARD_FAILED", "MSA_SIDECAR_DIED.json",
                 "COMMIT_TIMEOUT.json", SMOKE_FAILED_FILE,
                 "STATE_FIDELITY_FAILED.json",
                 "ARM_COMPLETE.json"):
        if (run_dir / name).is_file():
            return name
    return None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--workspace", required=True)
    ap.add_argument("--arm", required=True,
                    choices=["MSA", "MSA_PERMUTED_ROUTE",
                             "MSA_STATEFUL", "MSA_PERMUTED_STATEFUL"])
    ap.add_argument("--task", default="DexPBT-Regrasping-KukaAllegro-Direct-v0")
    ap.add_argument("--rho-threshold", type=float, default=None)
    ap.add_argument("--permuted-route", action="store_true")
    ap.add_argument("--num-policies", type=int, default=NUM_WORKERS)
    ap.add_argument("--num-seeds", type=int, default=SMOKE_NUM_SEEDS,
                    help=f"execution-smoke default {SMOKE_NUM_SEEDS} "
                         "(machinery validation only, NOT a method "
                         "conclusion); pilot 16 must be set explicitly; "
                         f"formal {FORMAL_NUM_SEEDS} is formal-only")
    ap.add_argument("--eval-num-envs", type=int, default=512)
    ap.add_argument("--eval-devices", default="",
                    help="comma list of eval GPUs (e.g. cuda:6,cuda:7). "
                         "REQUIRED for cycles: at most one Isaac process "
                         "per device; never defaults onto the trainers' "
                         "cuda:0")
    ap.add_argument("--readiness-marker", required=True,
                    help="FIRST_TICK_VALIDATED.json written by "
                         "continuation_guard; no cycle starts/resumes "
                         "before it exists")
    ap.add_argument("--hold-smoke-marker", default="",
                    help="HOLD_SMOKE_PASSED.json written by the hold "
                         "smoke; no cycle starts/resumes before it exists "
                         "(REQUIRED unless --hold-smoke-only)")
    ap.add_argument("--commit-max-wait-s", type=float,
                    default=DEFAULT_COMMIT_MAX_WAIT_S,
                    help="single bounded wait for a durable tolerance "
                         "commit; on timeout the arm is stopped")
    ap.add_argument("--once", action="store_true",
                    help="run at most one probe cycle then exit")
    ap.add_argument("--hold-smoke-only", action="store_true",
                    help="single-worker hold smoke; no tolerance writes, "
                         "no cycle, no eval")
    ap.add_argument("--hold-smoke-worker", type=int, default=0)
    ap.add_argument("--hold-smoke-seconds", type=float, default=60.0)
    args = ap.parse_args(argv)

    workspace = Path(args.workspace)
    run_dir = workspace.parent
    readiness_marker = Path(args.readiness_marker)

    if args.hold_smoke_only:
        return hold_smoke(run_dir, workspace, args.hold_smoke_worker,
                          args.hold_smoke_seconds, readiness_marker)

    if args.rho_threshold is None:
        ap.error("--rho-threshold is required (unless --hold-smoke-only)")
    eval_devices = [d for d in str(args.eval_devices).split(",") if d.strip()]
    if not eval_devices:
        ap.error("--eval-devices is required (unless --hold-smoke-only)")
    if not str(args.hold_smoke_marker).strip():
        ap.error("--hold-smoke-marker is required (unless --hold-smoke-only)")
    smoke_marker = Path(args.hold_smoke_marker)

    permuted = args.permuted_route or args.arm == "MSA_PERMUTED_ROUTE"
    ctl, holds = build_controller(
        workspace, task=args.task, rho_threshold=args.rho_threshold,
        permuted_route=permuted, num_seeds=args.num_seeds,
        eval_num_envs=args.eval_num_envs, eval_devices=eval_devices,
        python_exe=sys.executable,
        commit_max_wait_s=args.commit_max_wait_s)

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

    def _stop_arm_commit_timeout(err: CommitTimeout) -> None:
        # promotion is NEVER claimed here; the marker records every
        # worker's durable-commit ack state at the moment of the timeout.
        _atomic_write_json(run_dir / "COMMIT_TIMEOUT.json", {
            "worker_id": err.worker_id,
            "target_tau": err.target_tau,
            "last_observed_frame": err.last_frame,
            "last_observed_tau": err.last_tau,
            "max_wait_s": err.max_wait_s,
            "commit_acks_by_worker": getattr(err, "commit_acks", {}),
            "time": time.time(),
        })
        holds.release_all()
        from dexpbt_isaaclab.msa.continuation_guard import stop_arm
        stop_arm(run_dir, f"commit timeout: {err}", marker_name=None)

    try:
        while True:
            marker = _stop_markers_present(run_dir)
            if marker:
                print(f"[msa-sidecar] stop marker {marker} present; "
                      "releasing holds and exiting")
                return 0 if marker == "ARM_COMPLETE.json" else 2
            views = _worker_views(workspace, args.num_policies)
            # DOUBLE gate: 6/6 readiness marker AND hold-smoke PASSED
            # marker. While waiting the sidecar performs no holds at all,
            # so it can start early without racing the manual hold smoke.
            ready, why = cycle_gate_ok(readiness_marker, smoke_marker,
                                       views, args.num_policies)
            if not ready:
                print(f"[msa-sidecar] waiting for gate: {why}")
                time.sleep(POLL_S)
                continue
            all_ids = [v.worker_id for v in views]
            try:
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
            except CycleFailure as err:
                if isinstance(err.__cause__, CommitTimeout):
                    _stop_arm_commit_timeout(err.__cause__)
                    return 4
                raise
            if args.once:
                return 0
            time.sleep(POLL_S)
    finally:
        # never leave a worker stopped if the sidecar dies (the external
        # hold watchdog covers the cases where this block never runs)
        holds.release_all()


if __name__ == "__main__":
    raise SystemExit(main())
