"""External hold watchdog for the MSA probe sidecar.

Runs as its OWN process (launched by the arm launcher, next to but
independent of the sidecar), so hold recovery never depends on the
sidecar's `finally` block surviving a SIGKILL / OOM / power event.

Loop:
- read <run_dir>/msa_sidecar.pid and probe the sidecar process;
- while the sidecar is alive: keep watching;
- if the sidecar is DEAD and <run_dir>/MSA_HOLDS.json is NON-empty
  (workers were left SIGSTOPped mid-phase):
    1. SIGCONT every recorded PID that is still alive (children first,
       supervisor/root last);
    2. write MSA_SIDECAR_DIED.json (the holds it found + timestamp);
    3. stop the WHOLE arm (uncontrolled continuation training after a
       mid-cycle sidecar death would corrupt the probe's dose accounting);
- if the sidecar is DEAD and <run_dir>/MSA_BUSY.json exists — EVEN with no
  recorded holds (e.g. death during the durable COMMITTING phase, when
  every worker is running but un-acked commits are pending) — the arm is
  equally unrecoverable without operator review: write
  MSA_SIDECAR_DIED.json and stop the whole arm;
- only if the sidecar is dead with NO holds and NO busy marker does the
  watchdog exit quietly (a clean --once exit or a normal shutdown).

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

from dexpbt_isaaclab.msa.sidecar import BUSY_FILE, read_holds_file

POLL_S = 20.0
DIED_MARKER = "MSA_SIDECAR_DIED.json"


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:      # exists, owned by someone else
        return True


def sidecar_pid(run_dir: Path) -> Optional[int]:
    pf = Path(run_dir) / "msa_sidecar.pid"
    if not pf.is_file():
        return None
    try:
        return int(pf.read_text().strip())
    except ValueError:
        return None


def check_once(run_dir: Path,
               stop_arm_fn: Optional[Callable[[Path, str], None]] = None,
               ) -> str:
    """One watchdog decision. Returns:

    'ok'          — sidecar alive (or no pid file yet), keep watching;
    'idle_exit'   — sidecar dead, NO holds recorded and NO busy marker:
                    exit quietly;
    'released'    — sidecar dead with recorded holds OR with the
                    MSA_BUSY.json cycle marker (even if holds are empty,
                    e.g. mid-COMMITTING): SIGCONTed all live held PIDs
                    (children before root), wrote MSA_SIDECAR_DIED.json
                    and stopped the arm.
    """
    run_dir = Path(run_dir)
    pid = sidecar_pid(run_dir)
    if pid is None or _pid_alive(pid):
        return "ok"
    holds = read_holds_file(run_dir)
    busy_path = run_dir / BUSY_FILE
    busy: Optional[dict] = None
    if busy_path.is_file():
        try:
            busy = json.loads(busy_path.read_text())
        except (json.JSONDecodeError, OSError):
            busy = {"unreadable": True}
    if not holds and busy is None:
        return "idle_exit"
    released: dict[int, list[int]] = {}
    for w, pids in sorted(holds.items()):
        done = []
        for p in reversed(pids):             # children first, root last
            try:
                os.kill(int(p), signal.SIGCONT)
                done.append(int(p))
            except (ProcessLookupError, OSError):
                pass
        released[w] = done
    (run_dir / DIED_MARKER).write_text(json.dumps({
        "sidecar_pid": pid,
        "holds_found": {str(w): pids for w, pids in sorted(holds.items())},
        "sigcont_delivered": {str(w): p for w, p in sorted(released.items())},
        "busy_marker": busy,
        "time": time.time(),
        "action": "released held workers, stopping the whole arm "
                  "(no uncontrolled continuation training)",
    }, indent=2))
    if stop_arm_fn is None:
        from dexpbt_isaaclab.msa.continuation_guard import stop_arm

        def stop_arm_fn(rd, reason):
            stop_arm(rd, reason, marker_name=None)
    stop_arm_fn(run_dir, f"sidecar {pid} died mid-cycle "
                         f"(holds={sorted(holds)}, busy={busy is not None})")
    return "released"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--poll-s", type=float, default=POLL_S)
    args = ap.parse_args(argv)
    run_dir = Path(args.run_dir)
    while True:
        state = check_once(run_dir)
        if state == "released":
            print("[hold-watchdog] sidecar died with active holds: "
                  "released workers, stopped arm")
            return 3
        if state == "idle_exit":
            print("[hold-watchdog] sidecar exited cleanly (no holds); done")
            return 0
        time.sleep(args.poll_s)


if __name__ == "__main__":
    raise SystemExit(main())
