"""Continuation guard for ALL base_retention_msa_probe arms.

Two jobs, both fail-closed, shared by BASE_CONTINUATION / GATE_RELAX / MSA /
MSA_PERMUTED_ROUTE (a bare --checkpoint warm start does NOT restore the
curriculum tolerance by itself, and an rl_games epoch reset could silently
re-run ~5B frames):

1. --verify-only (BEFORE launch): validate the bootstrap manifest in
   ARM_CONFIG.json against disk — checkpoint exists, sha256 head-hash
   matches, worker mapping / lineage / frame / tolerance fields present.
   Non-zero exit MUST abort the launcher.

2. guard loop (AFTER launch): for every worker,
   - FIRST committed tick: live tolerance must equal the manifest tip
     tolerance within 1e-6, lineage_id must match, frame must be >= the
     5B baseline. Any mismatch stops the whole arm (fail closed).
   - HARD CAP: a worker whose ledger exceeds baseline + cap (+1 tick of
     slack) stops the whole arm — the continuation budget is worker-local
     +100M frames, never a silent extra 5B.

Pure decision logic lives in check_first_tick / check_cap so it is unit
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

from dexpbt_isaaclab.msa.sidecar import latest_committed_meta

CONTINUATION_CAP_FRAMES = 100_000_000
TICK_SLACK_FRAMES = 20_000_000
TOLERANCE_ATOL = 1e-6
POLL_S = 60.0


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
    if len(resume) != 6:
        failures.append(f"expected 6 resume checkpoints, got {len(resume)}")
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
# first-tick and cap checks (pure)
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


def check_cap(entry: dict, meta: dict,
              cap_frames: int = CONTINUATION_CAP_FRAMES,
              slack: int = TICK_SLACK_FRAMES) -> GuardVerdict:
    """Worker-local hard frame cap: baseline + cap (+ one tick of slack)."""
    base_frame = int(entry["frame"])
    frame = int(meta.get("frame", -1))
    limit = base_frame + int(cap_frames) + int(slack)
    if frame > limit:
        return GuardVerdict(False,
                            f"frame {frame} exceeds hard cap {limit} "
                            f"(baseline {base_frame} + {cap_frames} "
                            f"+ slack {slack})")
    return GuardVerdict(True)


# ---------------------------------------------------------------------------
# stop action + loop
# ---------------------------------------------------------------------------

def _stop_run(run_dir: Path, reason: str, marker_name: str) -> None:
    (run_dir / marker_name).write_text(
        json.dumps({"reason": reason, "time": time.time()}, indent=2))
    for pid_file in sorted(run_dir.glob("policy_*/pid.txt")):
        try:
            os.kill(int(pid_file.read_text().strip()), signal.SIGTERM)
        except (ValueError, OSError):
            pass


def guard_loop(run_dir: Path, workspace: Path, config: dict,
               cap_frames: int = CONTINUATION_CAP_FRAMES,
               poll_s: float = POLL_S,
               max_iterations: Optional[int] = None) -> int:
    """0 = all workers finished under the cap; 2 = fail-closed stop."""
    resume = config["resume_checkpoints"]
    first_tick_ok: set[int] = set()
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
            if wid not in first_tick_ok:
                verdict = check_first_tick(entry, meta)
                if not verdict.ok:
                    _stop_run(run_dir, f"worker {wid}: {verdict.reason}",
                              "GUARD_FAILED")
                    return 2
                first_tick_ok.add(wid)
            verdict = check_cap(entry, meta, cap_frames=cap_frames)
            if not verdict.ok:
                _stop_run(run_dir, f"worker {wid}: {verdict.reason}",
                          "CAP_REACHED")
                return 2
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
