"""MSA probe sidecar: production wiring of MSAController to a live workspace.

Launched by the MSA / MSA_PERMUTED_ROUTE arm launchers next to the training
supervisors. It is the ONLY writer of msa_* control fields (through the
locked update_curriculum_control API) and never touches policy weights,
optimizer, replay, PBT lineage or any observer-owned curriculum field.

Readiness barrier + hold handoff: the guard SIGSTOPs every worker at its
validated first continuation tick and records it in READINESS_HOLDS.json;
FIRST_TICK_VALIDATED.json appears only when all six are frozen at their
own boundary. The sidecar then ADOPTS those holds (verifies every PID is
still stopped and every worker's committed frame equals the validated
frame recorded in the marker, imports them into WorkerHolds and atomically
converts READINESS_HOLDS.json -> MSA_HOLDS.json) and starts PRE_EVAL from
exactly that frozen boundary — the workers are never released between
validation and the first PRE snapshot, so no worker trains past its
matched intervention boundary. Any adoption failure (PID no longer
stopped, frame != validated frame) SIGCONTs everything and fail-closed
stops the arm.

Hold smoke gate: the sidecar REQUIRES --hold-smoke-certificate — the
HOLD_SMOKE_CERTIFICATE.json produced by the DISPOSABLE
start_stateful_hold_smoke.sh run (same code / Isaac task / 8192 envs /
the shared eval GPU). That disposable run also proves SHARED-GPU EVAL
COEXISTENCE: while the throwaway trainer is SIGSTOPped (still holding its
VRAM) it launches one real policy-only eval cell on the SAME GPU and
requires it to start and exit cleanly (no CUDA OOM / Isaac startup
failure). The certificate must have eval_coexistence_ok=true and match
the current git commit, task, wm profile and GPU environment, else the
sidecar stops the arm fail-closed at startup. The in-arm run_hold_smoke.sh (--hold-smoke-only)
remains for EXECUTION SMOKE only: it releases the target worker and waits
for an extra committed tick, which breaks the matched intervention
boundary, so it must NEVER be used inside the formal four-arm comparison.
If the guard fails the arm (GUARD_FAILED) or a smoke fails
(HOLD_SMOKE_FAILED.json), the sidecar releases all holds and exits.

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

from dexpbt_isaaclab.msa.budget import load_reservation, write_no_new_cycle
from dexpbt_isaaclab.msa.controller import (
    CommitTimeout,
    CycleFailure,
    MSAController,
    PROBE_PACKET_FRAMES,
    WorkerView,
)
from dexpbt_isaaclab.msa.decision import (
    PROMOTION_MODE_GLOBAL,
    PROMOTION_MODE_SELECTIVE,
    SeedEvalResult,
)
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
READINESS_HOLDS_FILE = "READINESS_HOLDS.json"
BUSY_FILE = "MSA_BUSY.json"
SMOKE_PASSED_FILE = "HOLD_SMOKE_PASSED.json"
SMOKE_FAILED_FILE = "HOLD_SMOKE_FAILED.json"
CERTIFICATE_FILE = "HOLD_SMOKE_CERTIFICATE.json"
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
    6/6 readiness marker AND the hold-smoke gate file (either the
    execution-smoke PASSED marker or a validated disposable-smoke
    certificate — the caller decides which path applies and validates the
    certificate contents at startup). Until both exist the sidecar idles
    (it never touches MSA_HOLDS.json in that state, so it cannot race a
    hold smoke)."""
    ready, why = readiness_ok(readiness_marker, views, num_policies)
    if not ready:
        return False, why
    if not smoke_marker.is_file():
        return False, (f"hold smoke gate not passed: {smoke_marker} "
                       "missing (run the disposable hold smoke first)")
    return True, ""


# ---------------------------------------------------------------------------
# hold-smoke certificate (disposable GPU smoke, outside the formal arms)
# ---------------------------------------------------------------------------

def certificate_ok(cert: dict, *, git_commit: str, task: str, profile: str,
                   gpu_names: dict[str, str],
                   driver_version: str) -> tuple[bool, str]:
    """Validate a HOLD_SMOKE_CERTIFICATE.json against the CURRENT arm
    environment. The certificate is produced by the DISPOSABLE
    start_stateful_hold_smoke.sh run (same code, Isaac task, 8192 envs,
    shared eval GPU) — the formal arm never runs a hold smoke itself
    (an in-arm smoke releases the worker for an extra committed tick and
    breaks the matched intervention boundary).

    Must match: ok flag, the shared-GPU eval coexistence proof
    (eval_coexistence_ok), git commit, task, wm profile, GPU model of
    EVERY eval device and driver version. UUID / CUDA version are recorded
    for the audit trail but only the model+driver must match (any
    same-model GPU on the same driver behaves identically for SIGSTOP
    semantics and for running an eval cell beside a frozen trainer)."""
    if not cert.get("ok"):
        return False, "certificate has ok != true"
    if cert.get("eval_coexistence_ok") is not True:
        return False, ("certificate missing eval_coexistence_ok=true "
                       "(the shared-GPU eval-beside-frozen-trainer preflight "
                       "was not proven)")
    if str(cert.get("git_commit", "")) != str(git_commit):
        return False, (f"certificate git commit {cert.get('git_commit')!r} "
                       f"!= current {git_commit!r}")
    if str(cert.get("task", "")) != str(task):
        return False, (f"certificate task {cert.get('task')!r} != "
                       f"arm task {task!r}")
    if str(cert.get("wm_profile", "")) != str(profile):
        return False, (f"certificate profile {cert.get('wm_profile')!r} != "
                       f"arm profile {profile!r}")
    gpu = cert.get("gpu") or {}
    cert_name = str(gpu.get("name", ""))
    cert_driver = str(gpu.get("driver_version", ""))
    if not cert_name:
        return False, "certificate has no GPU model"
    for idx, name in sorted(gpu_names.items()):
        if str(name) != cert_name:
            return False, (f"eval GPU {idx} model {name!r} != certified "
                           f"model {cert_name!r}")
    if cert_driver != str(driver_version):
        return False, (f"driver {driver_version!r} != certified "
                       f"{cert_driver!r}")
    return True, ""


def _current_git_commit(repo_dir: Path) -> str:
    out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(repo_dir),
                         capture_output=True, text=True)
    if out.returncode != 0:
        raise RuntimeError(f"git rev-parse failed in {repo_dir}: "
                           f"{out.stderr.strip()}")
    return out.stdout.strip()


def _gpu_env(indices: list[str]) -> tuple[dict[str, str], str, str]:
    """(name per requested GPU index, driver version, CUDA version) from
    nvidia-smi. Raises when a requested index is not present — the gate
    must fail closed on a machine that does not match the certificate."""
    out = subprocess.run(
        ["nvidia-smi",
         "--query-gpu=index,name,uuid,driver_version",
         "--format=csv,noheader"],
        capture_output=True, text=True)
    if out.returncode != 0:
        raise RuntimeError(f"nvidia-smi failed: {out.stderr.strip()}")
    rows = {}
    for line in out.stdout.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 4:
            rows[parts[0]] = {"name": parts[1], "uuid": parts[2],
                              "driver_version": parts[3]}
    names: dict[str, str] = {}
    driver = ""
    for idx in indices:
        if idx not in rows:
            raise RuntimeError(f"eval GPU index {idx} not present "
                               f"(available: {sorted(rows)})")
        names[idx] = rows[idx]["name"]
        driver = rows[idx]["driver_version"]
    cuda = ""
    smi = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
    for token in smi.stdout.split():
        if token.startswith("12.") or token.startswith("11."):
            cuda = token
            break
    return names, driver, cuda


def _gpu_memory_used(index: str) -> Optional[int]:
    """Used VRAM (MiB) on one GPU index, or None if unavailable."""
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used",
         "--format=csv,noheader,nounits", "-i", str(index)],
        capture_output=True, text=True)
    if out.returncode != 0:
        return None
    try:
        return int(out.stdout.strip().splitlines()[0])
    except (ValueError, IndexError):
        return None


def _gpu_uuid_map() -> dict[str, str]:
    """GPU index -> UUID from nvidia-smi (audit trail only)."""
    smi = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"],
        capture_output=True, text=True)
    uuid_by_idx: dict[str, str] = {}
    for line in smi.stdout.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 2:
            uuid_by_idx[parts[0]] = parts[1]
    return uuid_by_idx


def _pid_running(pid: int) -> bool:
    """Alive and not a zombie (zombies are reap-pending, never survivors)."""
    return _proc_state(pid) not in (None, "Z", "X", "x")


def _run_bounded_logged(cmd: list[str], env: dict, log_path: Path,
                        gpu_index: Optional[str], timeout_s: float,
                        term_grace_s: float = 20.0,
                        poll_s: float = 2.0) -> dict:
    """Run one child with NO PIPE deadlock and a HARD wall timeout.

    - stdout+stderr append to log_path (a file, not an unconsumed PIPE:
      an Isaac process writing more than the ~64KB pipe buffer would
      otherwise block forever with nobody reading);
    - start_new_session=True: the child gets its OWN process group, so a
      timeout can terminate the ENTIRE evaluator tree with killpg;
    - on timeout: SIGTERM the process group -> bounded grace ->
      SIGKILL the process group -> verify no survivor (every pid ever
      enumerated under the child); survivors are reported, never ignored;
    - samples the GPU's used VRAM (peak) while the child runs if
      gpu_index is given.

    Returns {exit_code, timed_out, duration_s, log_path, error_tail,
    survivor_pids, gpu_memory_peak}.
    """
    result: dict = {"exit_code": None, "timed_out": False,
                    "duration_s": 0.0, "log_path": str(log_path),
                    "error_tail": "", "survivor_pids": [],
                    "gpu_memory_peak": None}
    start = time.monotonic()
    known: set[int] = set()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "ab") as logf:
        proc = subprocess.Popen(cmd, env=env, stdout=logf,
                                stderr=subprocess.STDOUT,
                                start_new_session=True)
        known.add(proc.pid)
        deadline = start + float(timeout_s)
        while proc.poll() is None:
            known.update(_descendants(proc.pid))
            if gpu_index is not None:
                m = _gpu_memory_used(gpu_index)
                if m is not None:
                    result["gpu_memory_peak"] = max(
                        result["gpu_memory_peak"] or 0, m)
            if time.monotonic() > deadline:
                result["timed_out"] = True
                break
            time.sleep(max(0.05, float(poll_s)))
    if result["timed_out"]:
        known.update(_descendants(proc.pid))
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except (ProcessLookupError, PermissionError, OSError):
            pass
        grace_end = time.monotonic() + max(0.5, float(term_grace_s))
        while time.monotonic() < grace_end:
            if proc.poll() is not None and \
                    not any(_pid_running(p) for p in known):
                break
            time.sleep(0.2)
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError, OSError):
            pass
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            pass
        verify_end = time.monotonic() + 10.0
        survivors = [p for p in sorted(known) if _pid_running(p)]
        while survivors and time.monotonic() < verify_end:
            time.sleep(0.2)
            survivors = [p for p in sorted(known) if _pid_running(p)]
        result["survivor_pids"] = survivors
    else:
        proc.wait()
    result["exit_code"] = proc.poll()
    result["duration_s"] = round(time.monotonic() - start, 2)
    try:
        tail = Path(log_path).read_bytes()[-4000:]
        result["error_tail"] = tail.decode("utf-8", "replace")[-2000:]
    except OSError:
        pass
    return result


def _run_coexistence_eval(policy_ws: Path, meta: dict, worker_id: int,
                          gpu_index: str, task: str, run_dir: Path,
                          eval_num_envs: int, python_exe: str,
                          timeout_s: float = 1800.0) -> dict:
    """Launch ONE real policy-only eval cell (N=1) on the SAME GPU as the
    frozen trainer, through the production eval_bridge, with bounded
    wall time and full process-group cleanup (_run_bounded_logged).
    Proves the eval process can start and run beside a SIGSTOPped trainer
    that is still holding its VRAM — no CUDA OOM, no Isaac startup
    failure, no hung evaluator left behind."""
    from dexpbt_isaaclab.msa.eval_bridge import (
        EvalRequest,
        SnapshotSpec,
        build_seed_manifest,
    )
    manifest_id, seeds = build_seed_manifest(1)
    snap_dir = run_dir / "cert_eval_snapshot"
    raw_dir = run_dir / "cert_eval_raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    spec = SnapshotSpec.freeze(
        Path(meta["_committed_ckpt"]), f"cert_w{worker_id}", worker_id,
        int(meta.get("frame", 0)), snap_dir)
    tau = float((meta.get("curriculum") or {}).get("success_tolerance", 0.02))
    req = EvalRequest(snapshot=spec, tau=tau, manifest_id=manifest_id,
                      seeds=seeds, task=task,
                      run_id=f"cert_{run_dir.name}", raw_dir=str(raw_dir),
                      eval_device=f"cuda:{gpu_index}",
                      num_envs=int(eval_num_envs),
                      num_episodes=int(eval_num_envs))
    cmd = req.build_commands(python_exe)[0]
    env = req.build_env()
    return _run_bounded_logged(cmd, env, run_dir / "cert_eval.log",
                               str(gpu_index), timeout_s)


def _tree_shape(root_pid: int) -> list[dict]:
    """[{pid, comm}] for the worker tree — recorded in the certificate so
    a later reader can see WHAT was actually SIGSTOPped."""
    shape = []
    for pid in [root_pid] + _descendants(root_pid):
        try:
            comm = (Path("/proc") / str(pid) / "comm").read_text().strip()
        except OSError:
            comm = "?"
        shape.append({"pid": int(pid), "comm": comm})
    return shape


def certificate_smoke(run_dir: Path, workspace: Path, worker_id: int,
                      hold_seconds: float, cert_path: Path, *,
                      task: str, profile: str, gpu_index: str,
                      repo_dir: Path, eval_num_envs: int = 512,
                      eval_timeout_s: float = 1800.0,
                      python_exe: Optional[str] = None,
                      first_tick_timeout_s: float = 4 * 3600.0,
                      resume_timeout_s: float = 5400.0,
                      poll_s: float = POLL_S) -> int:
    """DISPOSABLE hold smoke: validates SIGSTOP/SIGCONT against a real
    Isaac trainer AND the shared-GPU eval coexistence path OUTSIDE the
    formal arms, then writes HOLD_SMOKE_CERTIFICATE.json. Produces NO
    method results; the disposable run is torn down by its launcher after.

    wait first committed tick -> record tree shape + VRAM -> hold ->
    verify the committed frame is frozen -> WHILE the trainer is frozen
    (still holding its VRAM) launch ONE real policy-only eval cell (N=1,
    eval_num_envs) on the SAME GPU with a HARD wall timeout
    (eval_timeout_s; on timeout the WHOLE evaluator process group is
    terminated and verified gone) and require it to exit cleanly (no CUDA
    OOM, no Isaac startup failure, no survivor) -> release -> verify the
    next committed tick appears. ok=true only when BOTH the hold
    behaviour AND the shared-GPU eval succeeded; the trainer hold is
    released in the finally block on EVERY path (success, failure,
    exception, timeout)."""
    holds = WorkerHolds(run_dir)
    policy_ws = workspace / f"{worker_id:03d}"
    python_exe = python_exe or sys.executable
    cert: dict = {"ok": False, "eval_coexistence_ok": False,
                  "task": str(task), "wm_profile": str(profile),
                  "hold_seconds": float(hold_seconds),
                  "worker_id": int(worker_id),
                  "eval_device": f"cuda:{gpu_index}",
                  "eval_num_envs": int(eval_num_envs),
                  "eval_timeout_s": float(eval_timeout_s),
                  "eval_exit_code": None, "eval_error": "",
                  "eval_duration_s": None, "eval_timed_out": False,
                  "eval_log_path": "", "eval_error_tail": "",
                  "eval_survivor_pids": [],
                  "gpu_memory_before_hold": None,
                  "gpu_memory_while_trainer_frozen": None,
                  "gpu_memory_during_eval": None,
                  "disposable_run_dir": str(run_dir),
                  "timestamp": time.time()}
    try:
        cert["git_commit"] = _current_git_commit(repo_dir)
        names, driver, cuda = _gpu_env([str(gpu_index)])
        cert["gpu"] = {"index": str(gpu_index),
                       "name": names[str(gpu_index)],
                       "uuid": _gpu_uuid_map().get(str(gpu_index), ""),
                       "driver_version": driver,
                       "cuda_version": cuda}
        deadline = time.monotonic() + first_tick_timeout_s
        while latest_committed_meta(policy_ws) is None:
            if time.monotonic() > deadline:
                cert["error"] = ("no committed tick within "
                                 f"{first_tick_timeout_s}s")
                return 2
            time.sleep(poll_s)
        meta = latest_committed_meta(policy_ws) or {}
        frame_before = int(meta.get("frame", -1))
        cert["frame_before_hold"] = frame_before
        cert["gpu_memory_before_hold"] = _gpu_memory_used(gpu_index)
        pid_file = run_dir / f"policy_{worker_id:02d}" / "pid.txt"
        cert["worker_tree"] = _tree_shape(int(pid_file.read_text().strip()))
        holds.hold([worker_id])
        time.sleep(float(hold_seconds))
        frame_during = int((latest_committed_meta(policy_ws) or {})
                           .get("frame", -1))
        cert["frame_after_hold_window"] = frame_during
        if frame_during != frame_before:
            cert["error"] = (f"frame advanced during hold: {frame_before} "
                             f"-> {frame_during}")
            return 2
        # SHARED-GPU COEXISTENCE: while the trainer is still SIGSTOPped and
        # holding its VRAM, run one real eval cell on the SAME GPU with a
        # hard wall timeout and full process-group cleanup.
        cert["gpu_memory_while_trainer_frozen"] = _gpu_memory_used(gpu_index)
        try:
            res = _run_coexistence_eval(
                policy_ws, meta, worker_id, str(gpu_index), task, run_dir,
                eval_num_envs, python_exe, timeout_s=float(eval_timeout_s))
            cert["eval_exit_code"] = res["exit_code"]
            cert["eval_duration_s"] = res["duration_s"]
            cert["eval_timed_out"] = bool(res["timed_out"])
            cert["eval_log_path"] = res["log_path"]
            cert["eval_survivor_pids"] = list(res["survivor_pids"])
            cert["gpu_memory_during_eval"] = res["gpu_memory_peak"]
            if res["timed_out"]:
                cert["eval_error"] = (f"eval timed out after "
                                      f"{eval_timeout_s}s (process group "
                                      "terminated)")
                cert["eval_error_tail"] = res["error_tail"]
                cert["error"] = "shared-GPU eval coexistence timed out"
                return 2
            if res["survivor_pids"]:
                cert["eval_error"] = (f"evaluator survivors after "
                                      f"termination: {res['survivor_pids']}")
                cert["eval_error_tail"] = res["error_tail"]
                cert["error"] = "shared-GPU eval left survivors"
                return 2
            if res["exit_code"] != 0:
                cert["eval_error"] = f"eval exited {res['exit_code']}"
                cert["eval_error_tail"] = res["error_tail"]
                cert["error"] = "shared-GPU eval coexistence failed"
                return 2
            cert["eval_coexistence_ok"] = True
        except Exception as exc:                         # noqa: BLE001
            cert["eval_error"] = f"{type(exc).__name__}: {exc}"
            cert["error"] = "shared-GPU eval coexistence raised"
            return 2
        holds.release([worker_id])
        resume_deadline = time.monotonic() + float(resume_timeout_s)
        while True:
            frame_after = int((latest_committed_meta(policy_ws) or {})
                              .get("frame", -1))
            if frame_after > frame_before:
                cert["next_committed_frame"] = frame_after
                break
            if time.monotonic() > resume_deadline:
                cert["error"] = (f"no new committed tick within "
                                 f"{resume_timeout_s}s after release")
                return 2
            time.sleep(poll_s)
        cert["ok"] = True
        return 0
    except Exception as exc:                             # noqa: BLE001
        cert["error"] = f"{type(exc).__name__}: {exc}"
        return 2
    finally:
        holds.release_all()
        residual = read_holds_file(run_dir)
        if cert["ok"] and residual:
            cert["ok"] = False
            cert["error"] = f"residual holds after release: {residual}"
        # ok requires the coexistence proof too (defensive: never certify
        # without it even if a future edit reorders the success path)
        if cert["ok"] and not cert.get("eval_coexistence_ok"):
            cert["ok"] = False
            cert["error"] = "eval coexistence not proven"
        _atomic_write_json(cert_path, cert)


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


def read_readiness_holds(run_dir: Path) -> dict[int, dict]:
    """READINESS_HOLDS.json written by the continuation guard: one record
    per worker frozen at its validated first continuation tick
    ({worker_id, root_pid, pids, validated_frame, validated_tolerance,
    lineage_id, hold_timestamp})."""
    path = Path(run_dir) / READINESS_HOLDS_FILE
    if not path.is_file():
        return {}
    try:
        raw = json.loads(path.read_text())
        return {int(w): dict(rec) for w, rec in raw.items()}
    except (json.JSONDecodeError, ValueError, TypeError, OSError):
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

    def adopt(self, records: dict[int, dict]) -> list[int]:
        """Adopt readiness holds created by the continuation guard WITHOUT
        releasing the workers: verify every recorded PID is still in the
        stopped state, import them as this instance's held PIDs and persist
        them to MSA_HOLDS.json (the watchdog/guard then treat them exactly
        like sidecar-created holds). Raises on any PID that is dead or
        running — the caller must fail closed."""
        for w, rec in sorted(records.items()):
            pids = [int(p) for p in rec.get("pids", [])]
            if not pids:
                raise RuntimeError(f"worker {w}: readiness hold record has "
                                   "no PIDs")
            bad = [p for p in pids if _proc_state(p) not in ("T", "t")]
            if bad:
                raise RuntimeError(
                    f"worker {w}: readiness-held pids {bad} are no longer "
                    "in the stopped state (dead or resumed)")
        for w, rec in sorted(records.items()):
            self._held[int(w)] = [int(p) for p in rec["pids"]]
        self._persist()
        return sorted(int(w) for w in records)


def adopt_readiness_holds(holds: WorkerHolds, run_dir: Path, workspace: Path,
                          readiness_marker: Path) -> list[int]:
    """Seamless guard -> sidecar hold handoff for the FIRST MSA cycle.

    The guard froze every worker at its validated first continuation tick
    and left READINESS_HOLDS.json behind (MSA arms only). Before starting
    PRE_EVAL the sidecar:

    1. verifies every recorded PID is STILL stopped;
    2. verifies every worker's newest COMMITTED frame equals the
       validated_frame recorded in FIRST_TICK_VALIDATED.json — the PRE
       snapshot must come from exactly that matched intervention boundary;
    3. imports the holds into WorkerHolds (persisting MSA_HOLDS.json) and
       atomically removes READINESS_HOLDS.json.

    Any failure SIGCONTs EVERY readiness-held PID (children before root)
    and raises — the caller must stop the whole arm (the workers were
    never our holds to keep)."""
    records = read_readiness_holds(run_dir)
    if not records:
        return []
    try:
        payload = json.loads(readiness_marker.read_text())
        validated_frame = {int(r["worker_id"]): int(r["validated_frame"])
                           for r in payload["workers"]}
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
        raise RuntimeError(f"readiness marker unreadable during hold "
                           f"adoption: {exc}") from exc
    try:
        for w, rec in sorted(records.items()):
            want = validated_frame.get(int(w))
            if want is None:
                raise RuntimeError(
                    f"worker {w} is readiness-held but absent from "
                    f"{readiness_marker.name}")
            rec_frame = int(rec.get("validated_frame", -1))
            if rec_frame != want:
                raise RuntimeError(
                    f"worker {w}: readiness hold frame {rec_frame} != "
                    f"marker validated frame {want}")
            meta = latest_committed_meta(workspace / f"{w:03d}")
            frame = int((meta or {}).get("frame", -1))
            if frame != want:
                raise RuntimeError(
                    f"worker {w}: committed frame {frame} != validated "
                    f"boundary {want} — the PRE snapshot would not come "
                    "from the matched intervention boundary")
        holds.adopt(records)
    except Exception:
        # fail closed: the workers must never be left frozen by a failed
        # handoff — SIGCONT everything (children before supervisor/root).
        for rec in records.values():
            for pid in reversed([int(p) for p in rec.get("pids", [])]):
                try:
                    os.kill(pid, signal.SIGCONT)
                except (ProcessLookupError, OSError):
                    pass
        try:
            (Path(run_dir) / READINESS_HOLDS_FILE).unlink()
        except FileNotFoundError:
            pass
        raise
    # conversion: MSA_HOLDS.json was persisted above; removing the guard's
    # file completes the atomic ownership transfer (if we die between the
    # two writes, both files describe the SAME stopped pids and every stop
    # path SIGCONTs the union).
    try:
        (Path(run_dir) / READINESS_HOLDS_FILE).unlink()
    except FileNotFoundError:
        pass
    return sorted(int(w) for w in records)


# ---------------------------------------------------------------------------
# controller wiring
# ---------------------------------------------------------------------------

def build_controller(workspace: Path, *, task: str, rho_threshold: float,
                     permuted_route: bool, num_seeds: int,
                     eval_num_envs: int, eval_devices: list[str],
                     python_exe: str,
                     packet_frames: int = PROBE_PACKET_FRAMES,
                     commit_max_wait_s: float = DEFAULT_COMMIT_MAX_WAIT_S,
                     promotion_mode: str = PROMOTION_MODE_GLOBAL,
                     population_worker_ids: Optional[list[int]] = None,
                     cap_target_by_worker: Optional[dict[int, int]] = None,
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
        a dead sidecar as fatal while it exists.

        When a budget reservation exists for this run, its per-worker
        ceilings are merged into the busy payload so the continuation
        guard can enforce "never advance past the reserved ceiling while
        busy" without itself parsing CYCLE_BUDGET_RESERVATION.json."""
        busy_path = run_dir / BUSY_FILE
        if info is None:
            try:
                busy_path.unlink()
            except FileNotFoundError:
                pass
            return
        payload = dict(info)
        reservation = load_reservation(run_dir)
        if reservation is not None:
            payload["reserved_ceiling_by_worker"] = {
                str(k): int(v)
                for k, v in reservation.reserved_ceiling_by_worker.items()}
        _atomic_write_json(busy_path, payload)

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
        permuted_route=permuted_route,
        promotion_mode=promotion_mode,
        population_worker_ids=population_worker_ids,
        cap_target_by_worker=cap_target_by_worker,
        run_dir=run_dir)
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
                             "MSA_STATEFUL", "MSA_PERMUTED_STATEFUL",
                             "MSA_SELECTIVE_STATEFUL"])
    ap.add_argument("--task", default="DexPBT-Regrasping-KukaAllegro-Direct-v0")
    ap.add_argument("--rho-threshold", type=float, default=None)
    ap.add_argument("--permuted-route", action="store_true")
    ap.add_argument("--num-policies", type=int, default=NUM_WORKERS)
    ap.add_argument("--promotion-mode", default=PROMOTION_MODE_GLOBAL,
                    choices=[PROMOTION_MODE_GLOBAL, PROMOTION_MODE_SELECTIVE],
                    help="global_panel: single panel-wide promotion gate "
                         "(bit-for-bit the original path); "
                         "worker_local_selective: per-worker eligibility + "
                         "budget-gated admission (MSA_SELECTIVE_STATEFUL)")
    ap.add_argument("--population-size", type=int, default=NUM_WORKERS,
                    help="population size for worker_local_selective's "
                         "min-anchors accounting; defaults to the full "
                         "6-worker panel")
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
                    help="HOLD_SMOKE_PASSED.json written by the in-arm "
                         "hold smoke — EXECUTION SMOKE ONLY (it releases "
                         "the worker for an extra committed tick and "
                         "breaks the matched intervention boundary); "
                         "formal arms must use --hold-smoke-certificate")
    ap.add_argument("--hold-smoke-certificate", default="",
                    help="HOLD_SMOKE_CERTIFICATE.json produced by the "
                         "DISPOSABLE start_stateful_hold_smoke.sh; must "
                         "match the current git commit, task, wm profile "
                         "and GPU environment (fail-closed at startup)")
    ap.add_argument("--wm-profile", default="",
                    help="the arm's wm strategy profile (certificate "
                         "matching; REQUIRED with "
                         "--hold-smoke-certificate)")
    ap.add_argument("--certificate-smoke", action="store_true",
                    help="disposable single-worker hold smoke producing "
                         "HOLD_SMOKE_CERTIFICATE.json; no readiness "
                         "marker, no tolerance writes, no cycle, no eval")
    ap.add_argument("--cert-out", default="",
                    help="certificate output path (--certificate-smoke)")
    ap.add_argument("--cert-gpu-index", default="4",
                    help="GPU index the disposable smoke worker trains on")
    ap.add_argument("--cert-eval-num-envs", type=int, default=512,
                    help="env count for the shared-GPU coexistence eval "
                         "cell run beside the frozen trainer "
                         "(--certificate-smoke)")
    ap.add_argument("--cert-eval-timeout-s", type=float, default=1800.0,
                    help="hard wall timeout for the coexistence eval cell; "
                         "on expiry the whole evaluator process group is "
                         "SIGTERM/SIGKILLed and the smoke fails closed "
                         "(--certificate-smoke)")
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

    if args.certificate_smoke:
        if not str(args.cert_out).strip():
            ap.error("--cert-out is required with --certificate-smoke")
        if not str(args.wm_profile).strip():
            ap.error("--wm-profile is required with --certificate-smoke")
        return certificate_smoke(
            run_dir, workspace, args.hold_smoke_worker,
            args.hold_smoke_seconds, Path(args.cert_out),
            task=args.task, profile=args.wm_profile,
            gpu_index=str(args.cert_gpu_index),
            eval_num_envs=args.cert_eval_num_envs,
            eval_timeout_s=args.cert_eval_timeout_s,
            repo_dir=Path(__file__).resolve().parent)

    if args.hold_smoke_only:
        return hold_smoke(run_dir, workspace, args.hold_smoke_worker,
                          args.hold_smoke_seconds, readiness_marker)

    if args.rho_threshold is None:
        ap.error("--rho-threshold is required (unless --hold-smoke-only)")
    eval_devices = [d for d in str(args.eval_devices).split(",") if d.strip()]
    if not eval_devices:
        ap.error("--eval-devices is required (unless --hold-smoke-only)")
    have_marker = bool(str(args.hold_smoke_marker).strip())
    have_cert = bool(str(args.hold_smoke_certificate).strip())
    if have_marker == have_cert:
        ap.error("exactly ONE of --hold-smoke-marker (execution smoke "
                 "only) or --hold-smoke-certificate (formal arms) is "
                 "required")
    if have_cert:
        # fail-closed certificate validation at STARTUP: the disposable
        # smoke must have certified this exact commit/task/profile/GPU env.
        from dexpbt_isaaclab.msa.continuation_guard import stop_arm
        cert_path = Path(args.hold_smoke_certificate)
        try:
            if not str(args.wm_profile).strip():
                raise RuntimeError("--wm-profile is required with "
                                   "--hold-smoke-certificate")
            cert = json.loads(cert_path.read_text())
            from dexpbt_isaaclab.msa.eval_bridge import device_index
            indices = [device_index(d) for d in eval_devices]
            gpu_names, driver, _cuda = _gpu_env(indices)
            git_commit = _current_git_commit(Path(__file__).resolve().parent)
            ok, why = certificate_ok(
                cert, git_commit=git_commit, task=args.task,
                profile=args.wm_profile, gpu_names=gpu_names,
                driver_version=driver)
            if not ok:
                raise RuntimeError(why)
        except Exception as exc:                         # noqa: BLE001
            print(f"[msa-sidecar] hold smoke certificate INVALID: {exc}")
            stop_arm(run_dir,
                     f"hold smoke certificate invalid: {exc}",
                     "GUARD_FAILED")
            return 2
        print(f"[msa-sidecar] hold smoke certificate validated: "
              f"{cert_path}")
        smoke_marker = cert_path
    else:
        smoke_marker = Path(args.hold_smoke_marker)

    permuted = args.permuted_route or args.arm == "MSA_PERMUTED_ROUTE"
    population_worker_ids = list(range(int(args.population_size)))
    cap_target_by_worker: Optional[dict[int, int]] = None
    arm_config_path = run_dir / "ARM_CONFIG.json"
    if arm_config_path.is_file():
        # worker-local hard cap = baseline_tip_frame + CONTINUATION_CAP —
        # the SAME matched cap the continuation guard enforces (imported
        # here, not at module scope, to avoid the sidecar<->guard import
        # cycle: continuation_guard imports marker helpers from sidecar).
        from dexpbt_isaaclab.msa.continuation_guard import (
            CONTINUATION_CAP_FRAMES,
        )
        arm_cfg = json.loads(arm_config_path.read_text())
        resume_checkpoints = arm_cfg.get("resume_checkpoints") or {}
        cap_target_by_worker = {
            int(key.lstrip("p")): int(entry["frame"]) + CONTINUATION_CAP_FRAMES
            for key, entry in resume_checkpoints.items()
        }
    ctl, holds = build_controller(
        workspace, task=args.task, rho_threshold=args.rho_threshold,
        permuted_route=permuted, num_seeds=args.num_seeds,
        eval_num_envs=args.eval_num_envs, eval_devices=eval_devices,
        python_exe=sys.executable,
        commit_max_wait_s=args.commit_max_wait_s,
        promotion_mode=args.promotion_mode,
        population_worker_ids=population_worker_ids,
        cap_target_by_worker=cap_target_by_worker)

    def _write_cycle_summary(result) -> None:
        out = workspace / "msa_cycles"
        out.mkdir(exist_ok=True)
        payload = {
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
            "promotion_mode": result.promotion_mode,
        }
        # optional worker_local_selective / budget fields: only present
        # (non-None) when the cycle actually used that mode / gate.
        for field_name in ("selected_for_promotion", "anchor_workers",
                          "eligible_by_worker", "rejection_reason_by_worker",
                          "budget_admission", "final_tol_by_worker"):
            value = getattr(result, field_name, None)
            if value is not None:
                payload[field_name] = value
        (out / f"{result.intervention_id}.json").write_text(
            json.dumps(payload, indent=2, default=str))

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
            # HOLD HANDOFF: the guard left every worker frozen at its
            # validated boundary (READINESS_HOLDS.json). Adopt those holds
            # (verify stopped state + committed frame == validated frame,
            # convert to MSA_HOLDS.json) and start PRE_EVAL from EXACTLY
            # that boundary — the workers are never released in between.
            if (run_dir / READINESS_HOLDS_FILE).is_file():
                try:
                    adopted = adopt_readiness_holds(
                        holds, run_dir, workspace, readiness_marker)
                    print(f"[msa-sidecar] adopted readiness holds for "
                          f"workers {adopted}; first PRE_EVAL starts from "
                          "the frozen validated boundary")
                except Exception as exc:                 # noqa: BLE001
                    from dexpbt_isaaclab.msa.continuation_guard import (
                        stop_arm,
                    )
                    print(f"[msa-sidecar] readiness hold adoption FAILED: "
                          f"{exc}")
                    stop_arm(run_dir,
                             f"readiness hold adoption failed: {exc}",
                             "GUARD_FAILED")
                    return 2
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
                        if result.first_blocking_gate == "all_budget_ineligible":
                            # NO busy marker was ever set (the controller
                            # refuses admission before the first hold): this
                            # is NOT a CycleFailure, just "wait for the guard
                            # to park these workers at their cap". Record it
                            # for observability and keep polling.
                            write_no_new_cycle(run_dir, {
                                "intervention_id": result.intervention_id,
                                "reason": "all_budget_ineligible",
                                "budget_admission": result.budget_admission,
                            })
                            print("[msa-sidecar] all due workers are "
                                  "budget-ineligible for a new cycle "
                                  f"({result.intervention_id}); idling for "
                                  "the continuation guard to park them")
                        else:
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
