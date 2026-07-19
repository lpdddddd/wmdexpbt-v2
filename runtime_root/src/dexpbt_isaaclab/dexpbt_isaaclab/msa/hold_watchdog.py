"""External hold watchdog for the MSA probe arms.

Runs as its OWN process (launched by every stateful arm launcher, next to
but independent of the guard and the sidecar), so hold recovery never
depends on the guard's or sidecar's `finally` block surviving a SIGKILL /
OOM / power event. It covers BOTH freeze owners across the whole arm
lifecycle:

READINESS PHASE (the 6/6 barrier and the MSA pre-adoption window). Before
FIRST_TICK_VALIDATED.json the CONTINUATION GUARD freezes validated workers
into READINESS_HOLDS.json; for MSA arms the holds stay after the marker
until the sidecar adopts them. While READINESS_HOLDS.json is non-empty the
watchdog NEVER idle-exits:
  - if the guard dies mid-barrier (marker not yet written): SIGCONT every
    readiness-held pid, write GUARD_DIED_DURING_READINESS.json, stop the
    arm;
  - if the sidecar dies after the marker but before adoption completes
    (READINESS_HOLDS.json still present): same — release the frozen
    workers and stop the arm.

POST-ADOPTION / CYCLE PHASE. Once the readiness holds are gone the sidecar
owns MSA_HOLDS.json / MSA_BUSY.json:
  - sidecar DEAD with MSA_HOLDS.json non-empty (workers SIGSTOPped
    mid-phase): SIGCONT them, write MSA_SIDECAR_DIED.json, stop the arm;
  - sidecar DEAD with MSA_BUSY.json present even with EMPTY holds (death
    during the durable COMMITTING phase, workers running but commits
    un-acked): equally unrecoverable — write MSA_SIDECAR_DIED.json and
    stop the arm;
  - sidecar dead with NO readiness holds, NO MSA holds and NO busy marker:
    exit quietly (idle_exit).

FAIL CLOSED. A holds file that exists but is unreadable/partially written
is NEVER treated as the empty set: if its owner is already dead the arm is
stopped rather than silently abandoned. A pid file that has not appeared
yet is tolerated for a bounded startup grace; only after the grace, with
readiness holds present and the owner pid still absent, is it a failure.

CALLER SAFE. stop_arm skips os.getpid(), so the watchdog calling it never
kills itself before the workers are down.

All decision logic lives in check_once() with an injectable stop action so
it is unit-testable with real (dummy) processes and without killing
anything real.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import time
from pathlib import Path
from typing import Callable, Optional

from dexpbt_isaaclab.msa.sidecar import (
    BUSY_FILE,
    HOLDS_FILE,
    READINESS_HOLDS_FILE,
)

POLL_S = 20.0
DIED_MARKER = "MSA_SIDECAR_DIED.json"
READINESS_DIED_MARKER = "GUARD_DIED_DURING_READINESS.json"
READINESS_MARKER = "FIRST_TICK_VALIDATED.json"
STARTUP_GRACE_S = 120.0


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:      # exists, owned by someone else
        return True


def _pid_from(run_dir: Path, name: str) -> Optional[int]:
    pf = Path(run_dir) / name
    if not pf.is_file():
        return None
    try:
        return int(pf.read_text().strip())
    except (ValueError, OSError):
        return None


def sidecar_pid(run_dir: Path) -> Optional[int]:
    return _pid_from(run_dir, "msa_sidecar.pid")


def _read_holds_strict(run_dir: Path) -> tuple[dict[int, list[int]], bool]:
    """(MSA holds, corrupt?). Absent -> ({}, False); present but
    unreadable/wrong-shape -> ({}, True) so the caller fails closed
    instead of treating a truncated write as no holds."""
    path = Path(run_dir) / HOLDS_FILE
    if not path.is_file():
        return {}, False
    try:
        raw = json.loads(path.read_text())
        if not isinstance(raw, dict):
            return {}, True
        return {int(w): [int(p) for p in pids]
                for w, pids in raw.items()}, False
    except (json.JSONDecodeError, ValueError, TypeError, OSError):
        return {}, True


def _read_readiness_strict(run_dir: Path) -> tuple[dict[int, dict], bool]:
    """(readiness holds, corrupt?). Same fail-closed contract as
    _read_holds_strict; each record must be a dict carrying 'pids'."""
    path = Path(run_dir) / READINESS_HOLDS_FILE
    if not path.is_file():
        return {}, False
    try:
        raw = json.loads(path.read_text())
        if not isinstance(raw, dict):
            return {}, True
        out: dict[int, dict] = {}
        for w, rec in raw.items():
            if not isinstance(rec, dict) or "pids" not in rec:
                return {}, True
            out[int(w)] = rec
        return out, False
    except (json.JSONDecodeError, ValueError, TypeError, OSError):
        return {}, True


def _read_busy(run_dir: Path) -> tuple[Optional[dict], bool]:
    path = Path(run_dir) / BUSY_FILE
    if not path.is_file():
        return None, False
    try:
        return json.loads(path.read_text()), False
    except (json.JSONDecodeError, OSError):
        return {"unreadable": True}, True


def _sigcont_pids(pids: list[int]) -> list[int]:
    done = []
    for p in reversed(pids):                 # children first, root last
        try:
            os.kill(int(p), signal.SIGCONT)
            done.append(int(p))
        except (ProcessLookupError, OSError):
            pass
    return done


def _default_stop_arm(run_dir: Path, reason: str) -> None:
    from dexpbt_isaaclab.msa.continuation_guard import stop_arm
    stop_arm(run_dir, reason, marker_name=None)


def _stop_readiness(run_dir: Path, readiness: dict[int, dict],
                    *, phase: str, dead_pid: Optional[int],
                    marker_present: bool, corrupt: bool,
                    stop_arm_fn: Callable[[Path, str], None]) -> str:
    """Release the readiness-frozen workers and stop the whole arm because
    the freeze owner (guard before the marker, sidecar in the pre-adoption
    window) died. Returns 'released_readiness'."""
    released: dict[int, list[int]] = {}
    for w, rec in sorted(readiness.items()):
        released[w] = _sigcont_pids([int(p) for p in rec.get("pids", [])])
    (run_dir / READINESS_DIED_MARKER).write_text(json.dumps({
        "phase": phase,
        "dead_pid": dead_pid,
        "marker_present": bool(marker_present),
        "readiness_holds_corrupt": bool(corrupt),
        "holds_found": {str(w): rec.get("pids", [])
                        for w, rec in sorted(readiness.items())},
        "sigcont_delivered": {str(w): p for w, p in sorted(released.items())},
        "time": time.time(),
        "action": "released readiness-frozen workers, stopping the whole "
                  "arm (freeze owner died before the matched boundary was "
                  "handed off)",
    }, indent=2))
    stop_arm_fn(run_dir, f"{phase}: freeze owner {dead_pid} died with "
                         f"readiness holds (corrupt={corrupt})")
    return "released_readiness"


def _stop_sidecar(run_dir: Path, pid: Optional[int],
                  holds: dict[int, list[int]], busy: Optional[dict],
                  *, corrupt: bool,
                  stop_arm_fn: Callable[[Path, str], None]) -> str:
    released: dict[int, list[int]] = {}
    for w, pids in sorted(holds.items()):
        released[w] = _sigcont_pids(list(pids))
    (run_dir / DIED_MARKER).write_text(json.dumps({
        "sidecar_pid": pid,
        "holds_found": {str(w): pids for w, pids in sorted(holds.items())},
        "sigcont_delivered": {str(w): p for w, p in sorted(released.items())},
        "busy_marker": busy,
        "holds_corrupt": bool(corrupt),
        "time": time.time(),
        "action": "released held workers, stopping the whole arm "
                  "(no uncontrolled continuation training)",
    }, indent=2))
    stop_arm_fn(run_dir, f"sidecar {pid} died mid-cycle "
                         f"(holds={sorted(holds)}, busy={busy is not None}, "
                         f"corrupt={corrupt})")
    return "released"


def check_once(run_dir: Path,
               stop_arm_fn: Optional[Callable[[Path, str], None]] = None,
               *, grace_expired: bool = True) -> str:
    """One watchdog decision. Returns:

    'ok'                 — the relevant owner is alive (or its pid file has
                           not appeared yet and the startup grace has not
                           elapsed): keep watching;
    'idle_exit'          — no readiness holds, sidecar dead, NO MSA holds
                           and NO busy marker: exit quietly;
    'released'           — sidecar dead in the cycle phase with MSA holds
                           OR a busy marker (or an unreadable holds/busy
                           file): SIGCONTed the held pids, wrote
                           MSA_SIDECAR_DIED.json, stopped the arm;
    'released_readiness' — the guard/sidecar died while READINESS_HOLDS
                           was still active (barrier or pre-adoption
                           window): SIGCONTed the readiness-frozen pids,
                           wrote GUARD_DIED_DURING_READINESS.json, stopped
                           the arm.

    `grace_expired` gates ONLY the pure startup race (an owner pid file
    that has not been written yet); it defaults True so unit tests that
    stage a dead owner get a deterministic decision.
    """
    run_dir = Path(run_dir)
    if stop_arm_fn is None:
        stop_arm_fn = _default_stop_arm

    guard_pid = _pid_from(run_dir, "continuation_guard.pid")
    side_pid = sidecar_pid(run_dir)
    guard_alive = guard_pid is not None and _pid_alive(guard_pid)
    side_alive = side_pid is not None and _pid_alive(side_pid)

    readiness, r_corrupt = _read_readiness_strict(run_dir)
    holds, h_corrupt = _read_holds_strict(run_dir)
    busy, b_corrupt = _read_busy(run_dir)
    marker_present = (run_dir / READINESS_MARKER).is_file()

    # READINESS PHASE: readiness holds present (or unreadable) — the
    # barrier / pre-adoption window is active and we must NEVER idle-exit.
    in_readiness = bool(readiness) or r_corrupt
    if in_readiness:
        # before the marker the GUARD owns the freeze; after it (MSA
        # pre-adoption window) the SIDECAR must adopt the holds
        if not marker_present:
            owner_pid, owner_alive = guard_pid, guard_alive
            phase = "guard died during readiness barrier"
        else:
            owner_pid, owner_alive = side_pid, side_alive
            phase = "sidecar died before readiness hold adoption"
        if owner_alive:
            return "ok"
        if owner_pid is None and not grace_expired:
            return "ok"                      # startup race: pid not written
        return _stop_readiness(
            run_dir, readiness, phase=phase, dead_pid=owner_pid,
            marker_present=marker_present, corrupt=r_corrupt,
            stop_arm_fn=stop_arm_fn)

    # CYCLE PHASE: no readiness holds. Monitor the sidecar.
    if side_pid is None or side_alive:
        return "ok"
    # sidecar dead
    if h_corrupt or b_corrupt:
        # unreadable holds/busy on a dead sidecar: fail closed
        return _stop_sidecar(run_dir, side_pid, holds, busy,
                             corrupt=True, stop_arm_fn=stop_arm_fn)
    if not holds and busy is None:
        return "idle_exit"
    return _stop_sidecar(run_dir, side_pid, holds, busy,
                         corrupt=False, stop_arm_fn=stop_arm_fn)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--poll-s", type=float, default=POLL_S)
    ap.add_argument("--startup-grace-s", type=float, default=STARTUP_GRACE_S)
    args = ap.parse_args(argv)
    run_dir = Path(args.run_dir)
    start = time.monotonic()
    while True:
        grace_expired = (time.monotonic() - start) > args.startup_grace_s
        state = check_once(run_dir, grace_expired=grace_expired)
        if state == "released":
            print("[hold-watchdog] sidecar died with active holds/busy: "
                  "released workers, stopped arm")
            return 3
        if state == "released_readiness":
            print("[hold-watchdog] freeze owner died during the readiness "
                  "barrier: released frozen workers, stopped arm")
            return 3
        if state == "idle_exit":
            print("[hold-watchdog] owner exited cleanly (no holds); done")
            return 0
        time.sleep(args.poll_s)


if __name__ == "__main__":
    raise SystemExit(main())
