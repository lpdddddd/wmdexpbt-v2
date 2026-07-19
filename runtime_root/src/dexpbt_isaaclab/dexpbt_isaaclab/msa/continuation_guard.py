"""Continuation guard for ALL base_retention_msa_probe stateful arms.

Fail-closed jobs, shared by BASE_STATEFUL / GATE_RELAX_STATEFUL /
MSA_STATEFUL / MSA_PERMUTED_STATEFUL:

1. --verify-only (BEFORE launch): validate the bootstrap manifest in
   ARM_CONFIG.json against disk — checkpoint exists, sha256 head-hash
   matches, worker mapping / lineage / frame / tolerance fields present.
   Non-zero exit MUST abort the launcher.

2. guard loop (AFTER launch):
   - SIX-WORKER READINESS BARRIER + STALL REPRODUCTION GATE: each worker's
     FIRST continuation committed tick (one full 80M committed interval on
     the restored state) is validated:
       * frame continues past the 5B baseline (no epoch/ledger reset);
       * lineage unchanged;
       * RESUME_APPLIED.json (written by train.py when the per-policy
         resume flags were applied to the env) confirms the tolerance
         resume STRUCTURALLY — the guard no longer requires the live
         first-tick tolerance to still equal the manifest tip, because a
         live curriculum may legally move within its own gates;
       * the live tolerance stays within ONE mastery shrink of the tip
         (>= 0.95 * tip): more than one shrink inside a single committed
         interval is the legacy-fallback cascade signature and fails;
       * the restored control JSON was not clobbered back to defaults
         (last-mastered, peak, freeze horizons — verify_control_fidelity).
     The arm-level STALL REPRODUCTION GATE then requires >= 4 of 6 workers
     to hold EXACTLY their source tolerance through the interval. If a
     majority broke out of the source stall band on their own, the restore
     did not reproduce the stalled regime: STATE_FIDELITY_FAILED.json is
     written and the whole arm stops — MSA must not start.
     Only when every check passes is FIRST_TICK_VALIDATED.json written
     (atomically, 6/6 only) with the per-worker reproduction record.
   - MATCHED WORKER-LOCAL CAP: target[w] = baseline[w] + cap (default
     6 committed ticks = 480M at the REAL stage-3 cadence of 80M/tick;
     the base 20M pbt_interval is NOT the committed quantum). Each worker
     is individually parked at its FIRST committed frame >= target[w];
     overshoot beyond one full tick of slack (80M) fails the arm. All six
     parked -> ARM_COMPLETE.json (normal completion).
   - MSA COORDINATION: while <run_dir>/MSA_BUSY.json exists park/cap
     actions are deferred as cap_pending; stop_worker SIGCONTs ONLY the
     target worker's held PIDs; only stop_arm releases all holds.

Every stop path SIGCONTs held PIDs first, then terminates each FULL
process tree with escalation: SIGTERM -> 20 s grace -> re-enumerate the
tree -> SIGKILL survivors -> verify exit. (The 2026-07-19 smoke showed
Isaac trainers surviving a bare SIGTERM for minutes.)

Pure decision logic lives in check_first_tick / verify_control_fidelity /
stall_reproduction_gate / cap_action / readiness_entry so it is unit
testable without processes; the loop and the stop actions are thin.
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
    BUSY_FILE,
    HOLDS_FILE,
    _descendants,
    latest_committed_meta,
    read_holds_file,
)

# REAL committed cadence: base pbt_interval 20M x stage-3 interval mult 4.0
# (pbt_consolidation_schedule.effective_tick_interval, SELECTION stage for
# every frame >= 3.5B — a 5B continuation is always stage 3). Confirmed
# from source-run metas (80M deltas from 3.48B on) and from the 2026-07-19
# smoke (first commit at baseline+80M).
COMMITTED_TICK_FRAMES = 80_000_000
CONTINUATION_CAP_TICKS = 6
CONTINUATION_CAP_FRAMES = CONTINUATION_CAP_TICKS * COMMITTED_TICK_FRAMES
TICK_SLACK_FRAMES = COMMITTED_TICK_FRAMES
TOLERANCE_ATOL = 1e-6
# One legitimate mastery shrink per committed interval is possible when the
# source state was already about to shrink; TWO shrinks inside one interval
# can only come from the ungated legacy fallback (0.9x per curriculum
# window) — the exact 2026-07-19 failure signature.
MAX_SHRINK_FACTOR_PER_INTERVAL = 0.95
POLL_S = 60.0
READINESS_MARKER = "FIRST_TICK_VALIDATED.json"
FIDELITY_FAILED_MARKER = "STATE_FIDELITY_FAILED.json"
BUSY_MARKER = BUSY_FILE
NUM_WORKERS = 6
MIN_REPRODUCED_WORKERS = 4
TERM_GRACE_S = 20.0
KILL_VERIFY_S = 10.0


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
# first-tick, fidelity, reproduction, readiness and cap checks (pure)
# ---------------------------------------------------------------------------

@dataclass
class GuardVerdict:
    ok: bool
    reason: str = ""
    # True when the live first-tick tolerance is EXACTLY the source tip
    # tolerance — the worker reproduced its stall through one full
    # committed interval on the restored state.
    reproduced: bool = False


def check_first_tick(entry: dict, meta: dict,
                     resume_applied: Optional[dict] = None,
                     atol: float = TOLERANCE_ATOL) -> GuardVerdict:
    """Validate the FIRST committed continuation tick of one worker.

    Tolerance-resume success is judged STRUCTURALLY from RESUME_APPLIED
    (written by train.py at env creation), not by requiring the live
    tolerance to still equal the manifest tip: a state-faithful restore
    runs a live curriculum whose own gates may legally move it. The live
    value must stay within ONE mastery shrink of the tip; anything lower
    is the ungated legacy-fallback cascade and fails the arm.
    """
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
    if resume_applied is None:
        return GuardVerdict(False,
                            "RESUME_APPLIED.json missing — cannot confirm "
                            "the tolerance resume was applied to the env")
    applied_tol = float((resume_applied.get("applied") or {})
                        .get("success_tolerance", float("nan")))
    if not (abs(applied_tol - want_tol) <= 1e-9):
        return GuardVerdict(False,
                            f"RESUME_APPLIED success_tolerance "
                            f"{applied_tol!r} != manifest tip {want_tol!r}; "
                            "tolerance resume failed")
    floor = MAX_SHRINK_FACTOR_PER_INTERVAL * want_tol - atol
    if not (floor <= live_tol <= want_tol + atol):
        return GuardVerdict(False,
                            f"first tick tolerance {live_tol!r} outside "
                            f"[{floor:.9f}, {want_tol + atol:.9f}] — more "
                            "than one shrink in a single committed interval "
                            "(ungated legacy-curriculum cascade signature)")
    if want_lineage and lineage and lineage != want_lineage:
        return GuardVerdict(False,
                            f"lineage {lineage} != manifest {want_lineage}")
    return GuardVerdict(True, reproduced=abs(live_tol - want_tol) <= atol)


def verify_control_fidelity(control, seed: dict,
                            atol: float = TOLERANCE_ATOL) -> list[str]:
    """After the first observer tick, the restored gate state must still be
    live — not clobbered back to dataclass defaults by an unseeded
    observer. Returns violations (empty == faithful).

    `control` is a CurriculumControlState (or None); `seed` is the
    bootstrap's observer_state_seed payload for the same worker.
    """
    if control is None:
        return ["curriculum control JSON missing at first tick"]
    violations: list[str] = []
    src_lm = float(seed["last_mastered_tolerance"])
    if abs(float(control.last_mastered_tolerance) - src_lm) > atol:
        violations.append(
            f"last_mastered_tolerance {control.last_mastered_tolerance!r} "
            f"!= source {src_lm!r} (default-clobber signature is 0.075)")
    if float(control.recent_peak_nsucc_mean) < \
            float(seed["recent_peak_nsucc_mean"]) - 1e-6:
        violations.append(
            f"recent_peak_nsucc_mean {control.recent_peak_nsucc_mean!r} "
            f"fell below source peak {seed['recent_peak_nsucc_mean']!r} "
            "(peak map was reset)")
    src_safe = float(seed["last_mastered_safe_tolerance"])
    if src_safe > 1e-9 and \
            abs(float(control.curriculum_last_mastered_tolerance)
                - src_safe) > atol:
        violations.append(
            "curriculum_last_mastered_tolerance "
            f"{control.curriculum_last_mastered_tolerance!r} != source "
            f"{src_safe!r} (safe rollback target lost)")
    if int(control.collapse_freeze_until_env_frames) < \
            int(seed["collapse_freeze_until"]):
        violations.append(
            f"collapse_freeze_until_env_frames "
            f"{control.collapse_freeze_until_env_frames} < source "
            f"{seed['collapse_freeze_until']} (freeze horizon lost)")
    return violations


def stall_reproduction_gate(records: dict[int, dict],
                            min_reproduced: int = MIN_REPRODUCED_WORKERS,
                            ) -> GuardVerdict:
    """Arm-level gate over the six per-worker readiness records.

    Every worker already passed check_first_tick (at most one legal
    shrink) and verify_control_fidelity. The arm additionally requires a
    MAJORITY (>= min_reproduced of 6) to hold EXACTLY the source tip
    tolerance through the full reproduction interval; if most workers
    broke out of the source stall band on their own, we did NOT restore
    the stalled regime and MSA must not start on top of it.
    """
    reproduced = [w for w, r in sorted(records.items())
                  if r.get("stall_reproduced")]
    if len(reproduced) < min_reproduced:
        moved = {w: records[w].get("validated_tolerance")
                 for w in sorted(records) if w not in reproduced}
        return GuardVerdict(
            False,
            f"stall reproduction FAILED: only {len(reproduced)}/"
            f"{len(records)} workers held their source tolerance through "
            f"the reproduction interval (need >= {min_reproduced}); "
            f"moved: {moved}")
    return GuardVerdict(True, reproduced=True)


def readiness_entry(worker_id: int, entry: dict, meta: dict,
                    verdict: Optional[GuardVerdict] = None) -> dict:
    """One worker's validated line in FIRST_TICK_VALIDATED.json."""
    cur = meta.get("curriculum") or {}
    rec = {
        "worker_id": int(worker_id),
        "validated_frame": int(meta.get("frame", -1)),
        "validated_tolerance": float(
            cur.get("success_tolerance", float("nan"))),
        "source_tolerance": float(entry.get("live_tolerance", float("nan"))),
        "lineage_id": str(meta.get("lineage_id") or ""),
        "checkpoint_identity": {
            "resume_checkpoint": str(entry.get("path", "")),
            "sha256_head": str(entry.get("sha256_head", "")),
            "committed_prefix": str(meta.get("_committed_prefix", "")),
        },
    }
    if verdict is not None:
        rec["stall_reproduced"] = bool(verdict.reproduced)
    return rec


def write_readiness_marker(run_dir: Path, records: dict[int, dict],
                           reproduction: Optional[dict] = None) -> Path:
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
    if reproduction is not None:
        payload["stall_reproduction"] = reproduction
    tmp = marker.with_suffix(f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(payload, indent=2))
    os.replace(tmp, marker)                     # atomic on POSIX
    return marker


def cap_action(entry: dict, meta: dict,
               cap_frames: int = CONTINUATION_CAP_FRAMES,
               slack: int = TICK_SLACK_FRAMES) -> str:
    """Matched worker-local cap decision on one committed frame:

    'run'  — below target, keep training;
    'park' — first committed frame >= target (within one tick of slack):
             stop THIS worker safely, record the observed frame;
    'fail' — beyond target + slack: the worker overshot a full committed
             tick past its budget, stop the arm.
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
# stop actions (SIGCONT -> SIGTERM -> grace -> re-enumerate -> SIGKILL)
# ---------------------------------------------------------------------------

def _sigcont_held_pids(run_dir: Path) -> None:
    """SIGCONT every PID recorded in MSA_HOLDS.json (a stopped process
    queues SIGTERM without acting on it). ONLY for whole-arm stops."""
    for pids in read_holds_file(run_dir).values():
        for pid in reversed(pids):              # children first, root last
            try:
                os.kill(int(pid), signal.SIGCONT)
            except (ProcessLookupError, OSError):
                pass


def _sigcont_worker_pids(run_dir: Path, worker_id: int) -> None:
    """SIGCONT ONLY the pids MSA_HOLDS.json records for one worker —
    a single-worker park must never wake the other held workers."""
    for pid in reversed(read_holds_file(run_dir).get(int(worker_id), [])):
        try:
            os.kill(int(pid), signal.SIGCONT)
        except (ProcessLookupError, OSError):
            pass


def _alive(pid: int) -> bool:
    """True only for RUNNING processes: a zombie (terminated but not yet
    reaped by its parent) still answers kill(pid, 0), so /proc state is
    checked to avoid treating reap-pending children as SIGKILL survivors."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except OSError:
        return True
    try:
        with open(f"/proc/{pid}/stat") as f:
            state = f.read().rsplit(")", 1)[1].split()[0]
        return state not in ("Z", "X", "x")
    except OSError:
        return False


def _signal_all(pids: list[int], sig: int) -> None:
    for pid in pids:
        try:
            os.kill(pid, sig)
        except (ProcessLookupError, OSError):
            pass


def _terminate_tree(root_pid: int,
                    grace_s: float = TERM_GRACE_S,
                    verify_s: float = KILL_VERIFY_S) -> list[int]:
    """Escalating full-tree termination. Returns PIDs still alive at the
    end (empty == verified dead).

    SIGCONT (a stopped process only queues signals) -> SIGTERM the whole
    tree -> grace window -> RE-ENUMERATE the tree (children spawned or
    missed between enumeration and delivery — the 2026-07-19 smoke showed
    Isaac trainers surviving SIGTERM) -> SIGKILL survivors -> verify.
    """
    def tree() -> list[int]:
        return [root_pid] + _descendants(root_pid)

    pids = tree()
    _signal_all(pids, signal.SIGCONT)
    _signal_all(pids, signal.SIGTERM)
    deadline = time.monotonic() + max(0.0, grace_s)
    while time.monotonic() < deadline:
        if not any(_alive(p) for p in pids):
            break
        time.sleep(0.5)
    survivors = sorted({p for p in (pids + tree()) if _alive(p)})
    if survivors:
        _signal_all(survivors, signal.SIGCONT)
        _signal_all(survivors, signal.SIGKILL)
        deadline = time.monotonic() + max(0.0, verify_s)
        while time.monotonic() < deadline:
            survivors = [p for p in survivors if _alive(p)]
            if not survivors:
                break
            time.sleep(0.5)
    left = [p for p in survivors if _alive(p)]
    if left:
        print(f"[guard] WARNING: pids still alive after SIGKILL: {left}")
    return left


def _worker_root_pid(run_dir: Path, worker_id: int) -> Optional[int]:
    pid_file = run_dir / f"policy_{worker_id:02d}" / "pid.txt"
    if not pid_file.is_file():
        return None
    try:
        return int(pid_file.read_text().strip())
    except ValueError:
        return None


def stop_worker(run_dir: Path, worker_id: int) -> None:
    """Safely park ONE worker: SIGCONT ONLY that worker's held PIDs
    (never the whole panel's), then terminate its full process tree with
    SIGTERM -> grace -> SIGKILL escalation."""
    _sigcont_worker_pids(run_dir, worker_id)
    pid = _worker_root_pid(run_dir, worker_id)
    if pid is not None:
        _terminate_tree(pid)


def stop_arm(run_dir: Path, reason: str,
             marker_name: Optional[str] = None,
             num_workers: int = NUM_WORKERS,
             marker_payload: Optional[dict] = None) -> None:
    """Stop the WHOLE arm: SIGCONT held PIDs first, write the marker (if
    any), kill sidecar + watchdog, terminate every worker's FULL tree
    (escalating to SIGKILL after the grace window)."""
    _sigcont_held_pids(run_dir)
    if marker_name:
        payload = {"reason": reason, "time": time.time()}
        if marker_payload:
            payload.update(marker_payload)
        (run_dir / marker_name).write_text(json.dumps(payload, indent=2))
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

def _load_resume_applied(run_dir: Path, worker_id: int) -> Optional[dict]:
    p = run_dir / f"policy_{worker_id:02d}" / "RESUME_APPLIED.json"
    if not p.is_file():
        return None
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def _load_observer_seed(workspace: Path, worker_id: int) -> Optional[dict]:
    p = workspace / "observer_state_seed" / f"{worker_id:03d}.json"
    if not p.is_file():
        return None
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def guard_loop(run_dir: Path, workspace: Path, config: dict,
               cap_frames: int = CONTINUATION_CAP_FRAMES,
               poll_s: float = POLL_S,
               max_iterations: Optional[int] = None) -> int:
    """0 = ARM_COMPLETE (all six parked at their cap); 2 = fail-closed."""
    resume = config["resume_checkpoints"]
    state_faithful = bool(config.get("state_faithful"))
    validated: dict[int, dict] = {}
    marker_written = (run_dir / READINESS_MARKER).is_file()
    parked: dict[int, dict] = {}
    cap_pending: set[int] = set()
    it = 0
    while True:
        it += 1
        # an in-flight MSA cycle (eval holds, probe packets, DECIDE,
        # durable COMMITTING) — park/cap actions are DEFERRED while it
        # exists: stopping or SIGCONTing workers mid-cycle would corrupt
        # the phase barrier. The committed meta is re-read on every
        # iteration, so the park/cap verdict after the busy marker clears
        # is always based on fresh frames.
        busy = (run_dir / BUSY_MARKER).is_file()
        for key, entry in sorted(resume.items()):
            wid = int(key.lstrip("p"))
            meta = latest_committed_meta(workspace / f"{wid:03d}")
            if meta is None:
                continue
            if int(meta.get("frame", -1)) <= int(entry["frame"]):
                continue     # not a continuation tick yet
            if wid not in validated:
                resume_applied = (_load_resume_applied(run_dir, wid)
                                  if state_faithful else
                                  {"applied": {"success_tolerance":
                                               float(entry["live_tolerance"])}})
                verdict = check_first_tick(entry, meta, resume_applied)
                if not verdict.ok:
                    # readiness marker is NEVER written on any failure
                    stop_arm(run_dir, f"worker {wid}: {verdict.reason}",
                             "GUARD_FAILED")
                    return 2
                if state_faithful:
                    seed = _load_observer_seed(workspace, wid)
                    if seed is None:
                        stop_arm(run_dir,
                                 f"worker {wid}: observer state seed "
                                 "missing — state-faithful arm launched "
                                 "without bootstrap", "GUARD_FAILED")
                        return 2
                    from dexpbt_isaaclab.pbt.pbt_curriculum_p1 import (
                        read_curriculum_control,
                    )
                    control = read_curriculum_control(
                        workspace / "curriculum_control" / f"{wid:03d}.json")
                    violations = verify_control_fidelity(control, seed)
                    if violations:
                        stop_arm(
                            run_dir,
                            f"worker {wid}: restored control state was "
                            f"overwritten by defaults: {violations}",
                            FIDELITY_FAILED_MARKER,
                            marker_payload={"worker_id": wid,
                                            "violations": violations})
                        return 2
                validated[wid] = readiness_entry(wid, entry, meta, verdict)
            if wid in parked:
                continue
            action = cap_action(entry, meta, cap_frames=cap_frames)
            if action != "run" and busy:
                if wid not in cap_pending:
                    cap_pending.add(wid)
                    print(f"[guard] worker {wid} reached its cap "
                          f"(frame {meta.get('frame')}) during an MSA "
                          "cycle; cap_pending until the busy marker "
                          "clears")
                continue
            cap_pending.discard(wid)
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
            reproduction = None
            if state_faithful:
                gate = stall_reproduction_gate(validated)
                reproduction = {
                    "passed": gate.ok,
                    "min_reproduced": MIN_REPRODUCED_WORKERS,
                    "reproduced_workers": [
                        w for w in sorted(validated)
                        if validated[w].get("stall_reproduced")],
                }
                if not gate.ok:
                    stop_arm(run_dir, gate.reason, FIDELITY_FAILED_MARKER,
                             marker_payload={
                                 "stall_reproduction": reproduction,
                                 "workers": [validated[w]
                                             for w in sorted(validated)]})
                    return 2
            write_readiness_marker(run_dir, validated, reproduction)
            marker_written = True
            print(f"[guard] {READINESS_MARKER} written "
                  f"({len(validated)}/{len(resume)} workers validated"
                  + (", stall reproduction gate PASSED" if state_faithful
                     else "") + ")")
        if len(parked) == len(resume):
            # NORMAL completion — never failure semantics
            (run_dir / "ARM_COMPLETE.json").write_text(json.dumps({
                "reason": "all six workers reached their worker-local "
                          f"+{cap_frames} continuation target",
                "workers": [parked[w] for w in sorted(parked)],
                "cap_frames": int(cap_frames),
                "committed_tick_frames": COMMITTED_TICK_FRAMES,
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
