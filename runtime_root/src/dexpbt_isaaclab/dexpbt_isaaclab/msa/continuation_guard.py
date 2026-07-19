"""Continuation guard for ALL base_retention_msa_probe arms.

Fail-closed jobs, shared by BASE_CONTINUATION / GATE_RELAX / MSA /
MSA_PERMUTED_ROUTE (a bare --checkpoint warm start does NOT restore the
curriculum tolerance by itself, and an rl_games epoch reset could silently
re-run ~5B frames):

1. --verify-only (BEFORE launch): validate the bootstrap manifest in
   ARM_CONFIG.json against disk — checkpoint exists, sha256 head-hash
   matches, worker mapping / lineage / frame / tolerance fields present.
   Non-zero exit MUST abort the launcher.

2. guard loop (AFTER launch):
   - SIX-WORKER READINESS BARRIER: only after ALL SIX workers have produced
     their first continuation committed tick AND every one passed the
     checkpoint/frame/lineage/tolerance validation, the guard atomically
     writes FIRST_TICK_VALIDATED.json (worker ids, validated frames,
     tolerances, lineages, checkpoint identities, timestamp). The MSA
     sidecar refuses to start/resume any probe cycle until this marker
     exists. ANY worker failing first-tick validation stops the WHOLE arm
     (GUARD_FAILED) and the readiness marker is never written.
   - MATCHED +100M CAP: target[w] = baseline[w] + 100M. Each worker is
     individually parked (its process tree stopped) at its FIRST committed
     frame >= target[w]; the observed frame is recorded. An observed frame
     beyond target + one tick of slack fails the arm (a worker must never
     silently run to ~140M+ before anyone notices). When all six are
     parked, ARM_COMPLETE.json is written (NORMAL completion, not a
     failure) and the guard stops the sidecar, the watchdog and every
     remaining worker process tree.
   Regardless of the resume runner's max_iterations semantics (rl_games
   max_epochs on a warm start), this worker-local cap is the FINAL
   authority on the continuation budget.

Every stop path first SIGCONTs any PIDs recorded in MSA_HOLDS.json (a
SIGTERM to a SIGSTOPped process would be queued, not delivered) and then
terminates the FULL process tree of each worker, never just the supervisor.

Pure decision logic lives in check_first_tick / cap_action /
readiness_entry so it is unit testable without processes; the loop and
the stop actions are thin.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from dexpbt_isaaclab.msa.sidecar import (
    HOLDS_FILE,
    _descendants,
    latest_committed_meta,
    read_holds_file,
)

CONTINUATION_CAP_FRAMES = 100_000_000
TICK_SLACK_FRAMES = 20_000_000
TOLERANCE_ATOL = 1e-6
POLL_S = 60.0
READINESS_MARKER = "FIRST_TICK_VALIDATED.json"
NUM_WORKERS = 6


def ckpt_head_hash(path: Path, head_bytes: int = 4 * 1024 * 1024) -> str:
    """size + first-4MB sha256 (same scheme as the CCRS manifest)."""
    h = hashlib.sha256()
    h.update(str(path.stat().st_size).encode())
    with path.open("rb") as f:
        h.update(f.read(head_bytes))
    return h.hexdigest()


# ---------------------------------------------------------------------------
# bootstrap manifest verification (pre-launch)
# ---------------------------------------------------------------------------

def verify_bootstrap(config: dict) -> list[str]:
    """Return a list of failures (empty == pass) for one ARM_CONFIG dict."""
    failures: list[str] = []
    resume = config.get("resume_checkpoints") or {}
    if len(resume) != NUM_WORKERS:
        failures.append(f"expected {NUM_WORKERS} resume checkpoints, "
                        f"got {len(resume)}")
    for key, entry in sorted(resume.items()):
        wid = int(key.lstrip("p"))
        path = Path(str(entry.get("path", "")))
        if not path.is_file():
            failures.append(f"{key}: checkpoint missing: {path}")
            continue
        want = str(entry.get("sha256_head", ""))
        if not want:
            failures.append(f"{key}: manifest has no sha256_head")
        elif ckpt_head_hash(path) != want:
            failures.append(f"{key}: checkpoint hash mismatch for {path}")
        # worker mapping: workspace path embeds the worker directory
        if f"/{wid:03d}/" not in str(path):
            failures.append(
                f"{key}: checkpoint path {path} does not belong to worker "
                f"{wid:03d} (mapping error)")
        for field_name in ("frame", "live_tolerance", "lineage_id",
                           "iteration"):
            if entry.get(field_name) in (None, ""):
                failures.append(f"{key}: manifest field {field_name} empty")
        tol = float(entry.get("live_tolerance", 0.0) or 0.0)
        if not (0.001 <= tol <= 0.2):
            failures.append(f"{key}: live_tolerance {tol} outside sane "
                            "meter range")
    return failures


# ---------------------------------------------------------------------------
# first-tick, readiness and cap checks (pure)
# ---------------------------------------------------------------------------

@dataclass
class GuardVerdict:
    ok: bool
    reason: str = ""


def check_first_tick(entry: dict, meta: dict,
                     atol: float = TOLERANCE_ATOL) -> GuardVerdict:
    """Validate the FIRST committed continuation tick of one worker against
    its bootstrap manifest entry."""
    base_frame = int(entry["frame"])
    want_tol = float(entry["live_tolerance"])
    want_lineage = str(entry.get("lineage_id") or "")
    frame = int(meta.get("frame", -1))
    cur = meta.get("curriculum") or {}
    live_tol = float(cur.get("success_tolerance", float("nan")))
    lineage = str(meta.get("lineage_id") or "")
    if frame < base_frame:
        return GuardVerdict(False,
                            f"first tick frame {frame} < 5B baseline "
                            f"{base_frame} (epoch/frame ledger reset?)")
    if not (abs(live_tol - want_tol) <= atol):
        return GuardVerdict(False,
                            f"first tick tolerance {live_tol!r} != manifest "
                            f"tip tolerance {want_tol!r} (atol {atol}); "
                            "tolerance resume failed")
    if want_lineage and lineage and lineage != want_lineage:
        return GuardVerdict(False,
                            f"lineage {lineage} != manifest {want_lineage}")
    return GuardVerdict(True)


def readiness_entry(worker_id: int, entry: dict, meta: dict) -> dict:
    """One worker's validated line in FIRST_TICK_VALIDATED.json."""
    cur = meta.get("curriculum") or {}
    return {
        "worker_id": int(worker_id),
        "validated_frame": int(meta.get("frame", -1)),
        "validated_tolerance": float(
            cur.get("success_tolerance", float("nan"))),
        "lineage_id": str(meta.get("lineage_id") or ""),
        "checkpoint_identity": {
            "resume_checkpoint": str(entry.get("path", "")),
            "sha256_head": str(entry.get("sha256_head", "")),
            "committed_prefix": str(meta.get("_committed_prefix", "")),
        },
    }


def write_readiness_marker(run_dir: Path, records: dict[int, dict]) -> Path:
    """Atomically write FIRST_TICK_VALIDATED.json — only with 6/6 records."""
    if sorted(records.keys()) != list(range(NUM_WORKERS)):
        raise ValueError(
            f"readiness marker requires ALL {NUM_WORKERS} validated "
            f"workers, got {sorted(records.keys())}")
    marker = run_dir / READINESS_MARKER
    payload = {
        "workers": [records[w] for w in sorted(records)],
        "timestamp": time.time(),
        "num_workers": NUM_WORKERS,
    }
    tmp = marker.with_suffix(f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(payload, indent=2))
    os.replace(tmp, marker)                     # atomic on POSIX
    return marker


def cap_action(entry: dict, meta: dict,
               cap_frames: int = CONTINUATION_CAP_FRAMES,
               slack: int = TICK_SLACK_FRAMES) -> str:
    """Matched +100M cap decision for one worker's committed frame:

    'run'  — below target, keep training;
    'park' — first committed frame >= target (within one tick of slack):
             stop THIS worker safely, record the observed frame;
    'fail' — beyond target + slack: the worker overshot a full tick past
             its budget (must never silently run to ~140M), stop the arm.
    """
    base_frame = int(entry["frame"])
    frame = int(meta.get("frame", -1))
    target = base_frame + int(cap_frames)
    if frame > target + int(slack):
        return "fail"
    if frame >= target:
        return "park"
    return "run"


# ---------------------------------------------------------------------------
# stop actions (SIGCONT-before-terminate, full process trees)
# ---------------------------------------------------------------------------

def _sigcont_held_pids(run_dir: Path) -> None:
    """SIGCONT every PID recorded in MSA_HOLDS.json (a stopped process
    queues SIGTERM without acting on it)."""
    for pids in read_holds_file(run_dir).values():
        for pid in reversed(pids):              # children first, root last
            try:
                os.kill(int(pid), signal.SIGCONT)
            except (ProcessLookupError, OSError):
                pass


def _terminate_tree(root_pid: int) -> None:
    for pid in [root_pid] + _descendants(root_pid):
        try:
            os.kill(pid, signal.SIGTERM)
        except (ProcessLookupError, OSError):
            pass


def _worker_root_pid(run_dir: Path, worker_id: int) -> Optional[int]:
    pid_file = run_dir / f"policy_{worker_id:02d}" / "pid.txt"
    if not pid_file.is_file():
        return None
    try:
        return int(pid_file.read_text().strip())
    except ValueError:
        return None


def stop_worker(run_dir: Path, worker_id: int) -> None:
    """Safely park ONE worker: SIGCONT its held PIDs, then terminate its
    full process tree."""
    _sigcont_held_pids(run_dir)
    pid = _worker_root_pid(run_dir, worker_id)
    if pid is not None:
        _terminate_tree(pid)


def stop_arm(run_dir: Path, reason: str,
             marker_name: Optional[str] = None,
             num_workers: int = NUM_WORKERS) -> None:
    """Stop the WHOLE arm: SIGCONT held PIDs first, write the marker (if
    any), kill sidecar + watchdog, terminate every worker's FULL tree."""
    _sigcont_held_pids(run_dir)
    if marker_name:
        (run_dir / marker_name).write_text(
            json.dumps({"reason": reason, "time": time.time()}, indent=2))
    for pid_name in ("msa_sidecar.pid", "hold_watchdog.pid"):
        pf = run_dir / pid_name
        if pf.is_file():
            try:
                _terminate_tree(int(pf.read_text().strip()))
            except (ValueError, OSError):
                pass
    for w in range(num_workers):
        pid = _worker_root_pid(run_dir, w)
        if pid is not None:
            _terminate_tree(pid)


# ---------------------------------------------------------------------------
# guard loop
# ---------------------------------------------------------------------------

def guard_loop(run_dir: Path, workspace: Path, config: dict,
               cap_frames: int = CONTINUATION_CAP_FRAMES,
               poll_s: float = POLL_S,
               max_iterations: Optional[int] = None) -> int:
    """0 = ARM_COMPLETE (all six parked at +100M); 2 = fail-closed stop."""
    resume = config["resume_checkpoints"]
    validated: dict[int, dict] = {}
    marker_written = (run_dir / READINESS_MARKER).is_file()
    parked: dict[int, dict] = {}
    it = 0
    while True:
        it += 1
        for key, entry in sorted(resume.items()):
            wid = int(key.lstrip("p"))
            meta = latest_committed_meta(workspace / f"{wid:03d}")
            if meta is None:
                continue
            if int(meta.get("frame", -1)) <= int(entry["frame"]):
                continue     # not a continuation tick yet
            if wid not in validated:
                verdict = check_first_tick(entry, meta)
                if not verdict.ok:
                    # readiness marker is NEVER written on any failure
                    stop_arm(run_dir, f"worker {wid}: {verdict.reason}",
                             "GUARD_FAILED")
                    return 2
                validated[wid] = readiness_entry(wid, entry, meta)
            if wid in parked:
                continue
            action = cap_action(entry, meta, cap_frames=cap_frames)
            if action == "fail":
                stop_arm(
                    run_dir,
                    f"worker {wid}: observed frame {meta.get('frame')} "
                    f"exceeds target + one tick of slack "
                    f"(baseline {entry['frame']} + {cap_frames} "
                    f"+ {TICK_SLACK_FRAMES})", "GUARD_FAILED")
                return 2
            if action == "park":
                parked[wid] = {
                    "worker_id": wid,
                    "baseline_frame": int(entry["frame"]),
                    "target_frame": int(entry["frame"]) + int(cap_frames),
                    "observed_frame": int(meta.get("frame", -1)),
                    "parked_at": time.time(),
                }
                stop_worker(run_dir, wid)
                print(f"[guard] worker {wid} parked at "
                      f"{parked[wid]['observed_frame']} "
                      f"(target {parked[wid]['target_frame']})")
        if not marker_written and len(validated) == len(resume):
            write_readiness_marker(run_dir, validated)
            marker_written = True
            print(f"[guard] {READINESS_MARKER} written "
                  f"({len(validated)}/{len(resume)} workers validated)")
        if len(parked) == len(resume):
            # NORMAL completion — never failure semantics
            (run_dir / "ARM_COMPLETE.json").write_text(json.dumps({
                "reason": "all six workers reached their worker-local "
                          "+100M continuation target",
                "workers": [parked[w] for w in sorted(parked)],
                "cap_frames": int(cap_frames),
                "time": time.time(),
            }, indent=2))
            stop_arm(run_dir, "ARM_COMPLETE", marker_name=None)
            return 0
        if max_iterations is not None and it >= max_iterations:
            return 0
        time.sleep(poll_s)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True, help="ARM_CONFIG.json path")
    ap.add_argument("--verify-only", action="store_true",
                    help="pre-launch bootstrap verification, then exit")
    ap.add_argument("--run-dir", default="")
    ap.add_argument("--workspace", default="")
    ap.add_argument("--cap-frames", type=int, default=CONTINUATION_CAP_FRAMES)
    args = ap.parse_args(argv)

    config = json.loads(Path(args.config).read_text())
    failures = verify_bootstrap(config)
    if failures:
        for f in failures:
            print(f"BOOTSTRAP FAIL: {f}")
        return 2
    print("BOOTSTRAP OK: 6/6 checkpoints verified (hash, mapping, fields)")
    if args.verify_only:
        return 0
    if not args.run_dir or not args.workspace:
        print("guard loop needs --run-dir and --workspace")
        return 2
    return guard_loop(Path(args.run_dir), Path(args.workspace), config,
                      cap_frames=args.cap_frames)


if __name__ == "__main__":
    raise SystemExit(main())
