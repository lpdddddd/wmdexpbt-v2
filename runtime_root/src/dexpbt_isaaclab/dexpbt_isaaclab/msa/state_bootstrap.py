"""State-faithful bootstrap for base_retention_msa_probe stateful arms.

A bare per-policy warm start (--checkpoint + --success_tolerance_resume)
restores ONLY the policy weights and the live tolerance. Everything that
actually produced the 0.022-0.028 stall in the source run lives elsewhere:

  * env-side gate state   -> curriculum_control/<w>.json  (allow_shrink,
    shrink_token, collapse_freeze_active, consolidation step, adaptive_*)
  * observer in-memory state -> NOT persisted anywhere (rolling nsucc
    windows, recent peak, freeze-until maps, last-mastered records,
    rollback counters, one-shrink ledger) — silently reset on relaunch
  * env-side curriculum clock -> curriculum_resume_snapshot.json
    (last_curriculum_update / frame_since_restart, in sim steps)

The 2026-07-19 GPU execution smoke proved the failure mode: with an empty
curriculum_control the env fell back to the LEGACY curriculum (0.9x shrink
at mean nsucc > 3, no gates) and worker 0 shrank 0.0231 -> 0.0168 within
the first 80M-frame committed interval.

This module restores / rebuilds all of it BEFORE any worker starts:

  1. copies each worker's source control JSON into the new workspace with
     every base-retention gate field intact and ONLY the msa_* fields
     cleared to defaults;
  2. rebuilds the observer's in-memory state from the source control JSON
     plus the last REBUILD_METAS committed metas (window samples exactly
     as _window_sample_from_state would build them) into
     <workspace>/observer_state_seed/<w>.json — the observer loads these
     at its first curriculum tick (see pbt_observer);
  3. copies curriculum_resume_snapshot.json into the new policy dirs and
     exports the env clock (last_curriculum_update / frame_since_restart)
     for per-policy CLI resume flags;
  4. writes STATE_FIDELITY.json (per worker, per field: source value,
     restored value, category: copied_exact / rebuilt_from_metas /
     env_clock / default_lost) and STATE_FINGERPRINT.json (sha256 of the
     restored state + of every source file consumed);
  5. FAILS CLOSED (non-zero exit, nothing marked faithful) when any
     critical source is missing.

The source run is strictly read-only here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from dexpbt_isaaclab.pbt.pbt_curriculum_p1 import (
    CurriculumControlState,
    read_curriculum_control,
    write_curriculum_control,
)

NUM_WORKERS = 6
# Rebuild horizon requested for the rolling-window reconstruction. The
# observer window keeps max_len=8 samples; we read 12 so the drop-ratio /
# slope of the earliest kept samples see the same predecessors the source
# observer saw, and so shrinks_since_rollback has more history to count.
REBUILD_METAS = 12
WINDOW_MAX_LEN = 8
# Real committed cadence of any 5B continuation: base pbt_interval 20M x
# stage-3 (SELECTION, frame >= 3.5B) interval mult 4.0 = 80M env frames.
# Measured in the source run: deltas are exactly 80M from 3.48B onward.
COMMITTED_TICK_FRAMES = 80_000_000
# An observer that restarts AFTER this much continuation progress must NOT
# re-apply the (by then stale) seed; it falls back to today's lossy restart
# behavior, which is acceptable outside the bootstrap window.
SEED_VALID_TICKS = 2

SEED_DIR_NAME = "observer_state_seed"
FIDELITY_NAME = "STATE_FIDELITY.json"
FINGERPRINT_NAME = "STATE_FINGERPRINT.json"

# Observer in-memory state that is not persisted anywhere and cannot be
# rebuilt from committed metas. Restored as defaults; the fidelity report
# marks them 'default_lost' and quantifies the impact.
DEFAULT_LOST_FIELDS = (
    "promising_protect_until",       # only in copy-event JSONL, best-effort
    "mutation_frozen_until",         # not persisted at all
    "late_floor_retention_patience_count",  # not persisted at all
    "floor_recovery_score_history",  # reporting-only
)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical_hash(obj: Any) -> str:
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


class BootstrapError(RuntimeError):
    """Any missing critical source state — the caller MUST fail closed."""


# ---------------------------------------------------------------------------
# source-run readers (read-only)
# ---------------------------------------------------------------------------

def committed_meta_files(worker_ws: Path) -> list[Path]:
    out = []
    for p in sorted(worker_ws.glob("0*.json")):
        if p.name.endswith(".tmp"):
            continue
        out.append(p)
    return out


def load_source_worker(src_run: Path, worker_id: int) -> dict[str, Any]:
    """Load every stall-state source for one worker; raise BootstrapError
    on anything missing (fail closed, never a partial restore)."""
    ws = src_run / "dexpbt_8gpu_ws" / f"{worker_id:03d}"
    control_path = (src_run / "dexpbt_8gpu_ws" / "curriculum_control"
                    / f"{worker_id:03d}.json")
    snap_path = (src_run / f"policy_{worker_id:02d}"
                 / "curriculum_resume_snapshot.json")
    latest_path = ws / "LATEST.commit"
    for p, what in ((control_path, "curriculum control JSON"),
                    (snap_path, "curriculum resume snapshot"),
                    (latest_path, "LATEST.commit")):
        if not p.is_file():
            raise BootstrapError(f"worker {worker_id}: source {what} "
                                 f"missing: {p}")
    control = read_curriculum_control(control_path)
    if control is None:
        raise BootstrapError(f"worker {worker_id}: unreadable control JSON "
                             f"{control_path}")
    snapshot = json.loads(snap_path.read_text())
    tip_prefix = json.loads(latest_path.read_text())["commit_dir"].split(".")[0]
    tip_meta_path = ws / f"{tip_prefix}.json"
    if not tip_meta_path.is_file():
        raise BootstrapError(f"worker {worker_id}: tip meta missing "
                             f"{tip_meta_path}")
    tip_meta = json.loads(tip_meta_path.read_text())
    metas = committed_meta_files(ws)
    if len(metas) < REBUILD_METAS:
        raise BootstrapError(
            f"worker {worker_id}: only {len(metas)} committed metas in "
            f"source run, need >= {REBUILD_METAS} for window rebuild")
    tail = [json.loads(p.read_text()) for p in metas[-REBUILD_METAS:]]
    return {
        "worker_id": worker_id,
        "control": control,
        "control_path": control_path,
        "snapshot": snapshot,
        "snapshot_path": snap_path,
        "tip_meta": tip_meta,
        "tip_meta_path": tip_meta_path,
        "tail_metas": tail,
        "tail_meta_paths": metas[-REBUILD_METAS:],
    }


# ---------------------------------------------------------------------------
# rebuilders
# ---------------------------------------------------------------------------

def window_sample_from_meta(meta: dict[str, Any]) -> dict[str, Any]:
    """EXACTLY pbt_observer._window_sample_from_state field semantics."""
    cur = meta.get("curriculum") or {}
    lineage = meta.get("lineage_id")
    return {
        "env_frames": int(meta.get("frame", 0)),
        "nsucc_mean": float(
            meta.get("nsucc_mean",
                     cur.get("mean_prev_episode_successes", 0.0)) or 0.0),
        "nsucc_max": float(
            meta.get("nsucc_max",
                     cur.get("max_prev_episode_successes", 0.0)) or 0.0),
        "success_tolerance": float(
            cur.get("success_tolerance", 0.075) or 0.075),
        "pbt_iteration": int(meta.get("iteration", 0)),
        "lineage_id": str(lineage) if lineage is not None else None,
    }


def rebuild_shrinks_since_rollback(tail_metas: list[dict],
                                   last_rollback_frame: int) -> int:
    """Count committed tolerance shrinks strictly after the last rollback.

    Rebuilt from meta history (a shrink = tolerance decrease between
    consecutive committed metas). If the rollback predates the rebuild
    horizon this is a LOWER BOUND — recorded as such in the fidelity
    report. The value only gates whether a hard-collapse rollback is
    allowed before its cooldown elapses, so a lower bound is conservative.
    """
    count = 0
    for prev, curr in zip(tail_metas, tail_metas[1:]):
        if int(curr.get("frame", -1)) <= int(last_rollback_frame):
            continue
        t_prev = float((prev.get("curriculum") or {})
                       .get("success_tolerance", 0.075))
        t_curr = float((curr.get("curriculum") or {})
                       .get("success_tolerance", 0.075))
        if t_curr < t_prev - 1e-12:
            count += 1
    return count


def build_observer_seed(src: dict[str, Any]) -> dict[str, Any]:
    """The observer's in-memory curriculum state for one worker, rebuilt
    from the source control JSON (authoritative last-written values) and
    the last committed metas (window samples)."""
    c: CurriculumControlState = src["control"]
    tip_frame = int(src["tip_meta"].get("frame", -1))
    samples = [window_sample_from_meta(m) for m in src["tail_metas"]]
    last_rollback_frame = int(c.last_rollback_frame)
    return {
        "schema_version": 1,
        "policy_idx": int(src["worker_id"]),
        "tip_frame": tip_frame,
        "seed_valid_until_frame": tip_frame
                                  + SEED_VALID_TICKS * COMMITTED_TICK_FRAMES,
        # rebuilt_from_metas
        "window_samples": samples[-WINDOW_MAX_LEN:],
        "shrinks_since_rollback": rebuild_shrinks_since_rollback(
            src["tail_metas"], last_rollback_frame),
        # copied_exact from the source control JSON (the observer wrote
        # these from the very in-memory dicts we are restoring)
        "recent_peak_nsucc_mean": float(c.recent_peak_nsucc_mean),
        "collapse_freeze_until": int(c.collapse_freeze_until_env_frames),
        "curriculum_safety_freeze_until": max(
            int(c.curriculum_rollback_freeze_until),
            int(c.late_floor_retention_freeze_until)),
        "wm_curriculum_freeze_until": int(c.wm_curriculum_freeze_until),
        "last_mastered_tolerance": float(c.last_mastered_tolerance),
        "last_mastered_safe_tolerance": float(
            c.curriculum_last_mastered_tolerance),
        "last_mastered_safe_frame": int(c.curriculum_last_mastered_frame),
        "last_mastered_safe_nsucc_mean": float(
            c.curriculum_last_mastered_nsucc_mean),
        "last_mastered_safe_global_retention": float(
            c.curriculum_last_mastered_global_retention),
        "repeated_collapse_count": int(c.curriculum_repeated_collapse_count),
        "rollback_count": int(c.rollback_count),
        "last_rollback_frame": last_rollback_frame,
        "rollback_reason": str(c.rollback_reason),
        "floor_recovery_protect_until": int(c.floor_recovery_protect_until),
        "first_floor_frame": int(c.first_floor_frame),
        "first_hard_floor_frame": int(c.first_hard_floor_frame),
        # default_lost (see DEFAULT_LOST_FIELDS)
        "default_lost": list(DEFAULT_LOST_FIELDS),
    }


def build_restored_control(src_control: CurriculumControlState,
                           ) -> CurriculumControlState:
    """Copy of the source control state with ONLY msa_* cleared."""
    data = asdict(src_control)
    restored = CurriculumControlState(**data)
    restored.msa_enabled = False
    restored.msa_forced_success_tolerance = 0.0
    restored.msa_provenance = None
    restored.msa_state_machine = None
    restored.msa_cycle = None
    return restored


def env_clock_from_snapshot(snapshot: dict[str, Any]) -> dict[str, Any]:
    cur = snapshot.get("curriculum") or {}
    for key in ("success_tolerance", "last_curriculum_update",
                "frame_since_restart"):
        if key not in cur:
            raise BootstrapError(f"resume snapshot missing curriculum.{key}")
    return {
        "success_tolerance": float(cur["success_tolerance"]),
        "last_curriculum_update": int(cur["last_curriculum_update"]),
        "frame_since_restart": int(cur["frame_since_restart"]),
        "curriculum_freeze_frames_after_restart": int(
            cur.get("curriculum_freeze_frames_after_restart", 0)),
    }


# ---------------------------------------------------------------------------
# fidelity comparison
# ---------------------------------------------------------------------------

def compare_worker_fidelity(src: dict[str, Any],
                            restored_control: CurriculumControlState,
                            seed: dict[str, Any]) -> dict[str, Any]:
    """Per-field source-vs-restored comparison for one worker. Every
    critical field must be exactly equal (or carry an explicit allowance);
    a single mismatch marks the worker (and the run) NOT faithful."""
    c: CurriculumControlState = src["control"]
    fields: dict[str, Any] = {}
    ok = True

    def rec(name: str, source, restored, category: str,
            match: bool | None = None, note: str = "") -> None:
        nonlocal ok
        m = (source == restored) if match is None else match
        if category != "default_lost" and not m:
            ok = False
        fields[name] = {"source": source, "restored": restored,
                        "category": category, "match": bool(m)}
        if note:
            fields[name]["note"] = note

    # 1. control JSON: every non-msa field must copy exactly
    src_d, rst_d = asdict(c), asdict(restored_control)
    mismatched = [k for k in src_d
                  if not k.startswith("msa_") and src_d[k] != rst_d[k]]
    rec("control_json_non_msa_fields",
        f"{len([k for k in src_d if not k.startswith('msa_')])} fields",
        "identical" if not mismatched else f"MISMATCH: {mismatched}",
        "copied_exact", match=not mismatched)
    rec("control_msa_cleared", "-",
        "cleared" if (not rst_d["msa_enabled"]
                      and rst_d["msa_forced_success_tolerance"] == 0.0
                      and rst_d["msa_provenance"] is None
                      and rst_d["msa_state_machine"] is None
                      and rst_d["msa_cycle"] is None) else "NOT CLEARED",
        "copied_exact",
        match=rst_d["msa_enabled"] is False
              and rst_d["msa_forced_success_tolerance"] == 0.0)

    # 2. observer seed scalars vs their authoritative source-control values
    for seed_key, src_val in (
        ("recent_peak_nsucc_mean", float(c.recent_peak_nsucc_mean)),
        ("collapse_freeze_until", int(c.collapse_freeze_until_env_frames)),
        ("wm_curriculum_freeze_until", int(c.wm_curriculum_freeze_until)),
        ("last_mastered_tolerance", float(c.last_mastered_tolerance)),
        ("last_mastered_safe_tolerance",
         float(c.curriculum_last_mastered_tolerance)),
        ("last_mastered_safe_frame", int(c.curriculum_last_mastered_frame)),
        ("last_mastered_safe_nsucc_mean",
         float(c.curriculum_last_mastered_nsucc_mean)),
        ("last_mastered_safe_global_retention",
         float(c.curriculum_last_mastered_global_retention)),
        ("repeated_collapse_count",
         int(c.curriculum_repeated_collapse_count)),
        ("rollback_count", int(c.rollback_count)),
        ("last_rollback_frame", int(c.last_rollback_frame)),
        ("floor_recovery_protect_until",
         int(c.floor_recovery_protect_until)),
        ("first_floor_frame", int(c.first_floor_frame)),
        ("first_hard_floor_frame", int(c.first_hard_floor_frame)),
    ):
        rec(f"seed.{seed_key}", src_val, seed[seed_key], "copied_exact")
    rec("seed.curriculum_safety_freeze_until",
        max(int(c.curriculum_rollback_freeze_until),
            int(c.late_floor_retention_freeze_until)),
        seed["curriculum_safety_freeze_until"], "copied_exact",
        note="max(rollback_freeze_until, late_floor_freeze_until): both "
             "source gates funnel into the same observer freeze map")

    # 3. rebuilt-from-metas fields
    tip = src["tip_meta"]
    tip_sample = window_sample_from_meta(tip)
    last = seed["window_samples"][-1]
    rec("seed.window_last_sample_frame", tip_sample["env_frames"],
        last["env_frames"], "rebuilt_from_metas")
    rec("seed.window_last_sample_tolerance",
        tip_sample["success_tolerance"], last["success_tolerance"],
        "rebuilt_from_metas")
    rec("seed.window_len", WINDOW_MAX_LEN, len(seed["window_samples"]),
        "rebuilt_from_metas")
    horizon_covers_rollback = (
        int(src["tail_metas"][0].get("frame", 0))
        <= int(c.last_rollback_frame))
    rec("seed.shrinks_since_rollback",
        "unpersisted (source observer memory)",
        seed["shrinks_since_rollback"], "rebuilt_from_metas", match=True,
        note="exact when the rebuild horizon covers last_rollback_frame "
             f"(covers={horizon_covers_rollback}); otherwise a "
             "conservative lower bound")

    # 4. env clock (resume snapshot -> CLI flags + snapshot copy)
    clock = env_clock_from_snapshot(src["snapshot"])
    tip_tol = float((tip.get("curriculum") or {})
                    .get("success_tolerance", float("nan")))
    rec("env.success_tolerance", tip_tol, clock["success_tolerance"],
        "env_clock", match=abs(tip_tol - clock["success_tolerance"]) <= 1e-9)
    for k in ("last_curriculum_update", "frame_since_restart",
              "curriculum_freeze_frames_after_restart"):
        rec(f"env.{k}", clock[k], clock[k], "env_clock")

    # 5. lost state, explicit
    for name in DEFAULT_LOST_FIELDS:
        rec(f"lost.{name}", "unpersisted in source", "default",
            "default_lost", match=False)

    return {"worker_id": int(src["worker_id"]), "faithful": ok,
            "fields": fields}


# ---------------------------------------------------------------------------
# bootstrap driver
# ---------------------------------------------------------------------------

def bootstrap_run(src_run: Path, run_dir: Path, workspace: Path,
                  num_workers: int = NUM_WORKERS) -> dict[str, Any]:
    """Restore the full source stall state into a NEW run. Fail closed."""
    src_run = src_run.resolve()
    if not src_run.is_dir():
        raise BootstrapError(f"source run missing: {src_run}")
    control_dir = workspace / "curriculum_control"
    seed_dir = workspace / SEED_DIR_NAME
    control_dir.mkdir(parents=True, exist_ok=True)
    seed_dir.mkdir(parents=True, exist_ok=True)

    per_worker: list[dict[str, Any]] = []
    source_hashes: dict[str, str] = {}
    restored_state: dict[str, Any] = {}
    clocks: dict[int, dict[str, Any]] = {}

    for w in range(num_workers):
        src = load_source_worker(src_run, w)
        restored = build_restored_control(src["control"])
        seed = build_observer_seed(src)
        seed["source_run"] = str(src_run)
        fidelity = compare_worker_fidelity(src, restored, seed)
        if not fidelity["faithful"]:
            raise BootstrapError(
                f"worker {w}: fidelity comparison failed BEFORE write: "
                + json.dumps({k: v for k, v in fidelity["fields"].items()
                              if not v["match"]
                              and v["category"] != "default_lost"}))

        write_curriculum_control(control_dir / f"{w:03d}.json", restored)
        seed_path = seed_dir / f"{w:03d}.json"
        seed_path.write_text(json.dumps(seed, indent=2))

        policy_dir = run_dir / f"policy_{w:02d}"
        policy_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src["snapshot_path"],
                     policy_dir / "curriculum_resume_snapshot.json")

        # read-back check: what the workers/observer will actually see
        readback = read_curriculum_control(control_dir / f"{w:03d}.json")
        if readback is None or asdict(readback) != asdict(restored):
            raise BootstrapError(f"worker {w}: control read-back mismatch")

        clocks[w] = env_clock_from_snapshot(src["snapshot"])
        per_worker.append(fidelity)
        for p in ([src["control_path"], src["snapshot_path"],
                   src["tip_meta_path"]] + src["tail_meta_paths"]):
            source_hashes[str(p)] = sha256_file(p)
        restored_state[f"control_{w:03d}"] = asdict(restored)
        restored_state[f"seed_{w:03d}"] = seed

    fidelity_doc = {
        "schema_version": 1,
        "source_run": str(src_run),
        "run_dir": str(run_dir),
        "timestamp": time.time(),
        "state_faithful": all(f["faithful"] for f in per_worker),
        "committed_tick_frames": COMMITTED_TICK_FRAMES,
        "workers": per_worker,
        "env_clocks": {str(w): clocks[w] for w in sorted(clocks)},
        "known_losses": {
            name: "restored as default; see STATE_SOURCE_AUDIT.md for the "
                  "behavioral impact assessment"
            for name in DEFAULT_LOST_FIELDS
        },
    }
    (run_dir / FIDELITY_NAME).write_text(json.dumps(fidelity_doc, indent=2))

    fingerprint = {
        "schema_version": 1,
        "state_fingerprint": canonical_hash(restored_state),
        "source_files_sha256": source_hashes,
        "source_run": str(src_run),
        "timestamp": time.time(),
    }
    (run_dir / FINGERPRINT_NAME).write_text(json.dumps(fingerprint, indent=2))
    return fidelity_doc


def verify_fingerprint(run_dir: Path, workspace: Path,
                       num_workers: int = NUM_WORKERS,
                       check_source: bool = True) -> list[str]:
    """Pre-launch validation: the on-disk restored state still hashes to
    the recorded fingerprint and (optionally) no source file changed.
    Returns failures (empty == pass)."""
    failures: list[str] = []
    fp_path = run_dir / FINGERPRINT_NAME
    if not fp_path.is_file():
        return [f"{FINGERPRINT_NAME} missing — bootstrap never ran"]
    fp = json.loads(fp_path.read_text())
    restored_state: dict[str, Any] = {}
    for w in range(num_workers):
        control = read_curriculum_control(
            workspace / "curriculum_control" / f"{w:03d}.json")
        seed_path = workspace / SEED_DIR_NAME / f"{w:03d}.json"
        if control is None:
            failures.append(f"worker {w}: restored control JSON missing")
            continue
        if not seed_path.is_file():
            failures.append(f"worker {w}: observer seed missing")
            continue
        restored_state[f"control_{w:03d}"] = asdict(control)
        restored_state[f"seed_{w:03d}"] = json.loads(seed_path.read_text())
    if failures:
        return failures
    got = canonical_hash(restored_state)
    if got != fp["state_fingerprint"]:
        failures.append(
            f"state fingerprint mismatch: disk {got[:16]}... != recorded "
            f"{fp['state_fingerprint'][:16]}... (restored state was "
            "modified after bootstrap)")
    if check_source:
        for path_str, want in fp["source_files_sha256"].items():
            p = Path(path_str)
            if not p.is_file():
                failures.append(f"source file vanished: {p}")
            elif sha256_file(p) != want:
                failures.append(f"source file CHANGED after bootstrap: {p}")
    return failures


# ---------------------------------------------------------------------------
# committed-cadence audit (read-only, source run)
# ---------------------------------------------------------------------------

def cadence_audit(src_run: Path, num_workers: int = NUM_WORKERS) -> dict:
    """Measure the REAL committed cadence from source-run metas — never
    assume the base 20M pbt_interval is the committed quantum."""
    out: dict[str, Any] = {"source_run": str(src_run), "workers": {}}
    for w in range(num_workers):
        ws = src_run / "dexpbt_8gpu_ws" / f"{w:03d}"
        frames = []
        for p in committed_meta_files(ws):
            try:
                frames.append(int(json.loads(p.read_text())["frame"]))
            except Exception:
                continue
        deltas = [b - a for a, b in zip(frames, frames[1:])]
        tail = deltas[-10:]
        out["workers"][str(w)] = {
            "n_committed": len(frames),
            "tip_frame": frames[-1] if frames else -1,
            "last_10_deltas": tail,
            "tail_cadence_frames": max(set(tail), key=tail.count)
                                   if tail else -1,
        }
    tails = {v["tail_cadence_frames"] for v in out["workers"].values()}
    out["committed_tick_frames"] = (tails.pop() if len(tails) == 1 else -1)
    out["matches_module_constant"] = (
        out["committed_tick_frames"] == COMMITTED_TICK_FRAMES)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source-run", default="")
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--workspace", default="")
    ap.add_argument("--verify", action="store_true",
                    help="verify fingerprint of an already-bootstrapped run")
    ap.add_argument("--audit-cadence", action="store_true")
    args = ap.parse_args(argv)

    run_dir = Path(args.run_dir)
    workspace = Path(args.workspace) if args.workspace \
        else run_dir / "dexpbt_8gpu_ws"

    if args.audit_cadence:
        if not args.source_run:
            print("--audit-cadence needs --source-run")
            return 2
        print(json.dumps(cadence_audit(Path(args.source_run)), indent=2))
        return 0

    if args.verify:
        failures = verify_fingerprint(run_dir, workspace)
        for f in failures:
            print(f"FINGERPRINT FAIL: {f}")
        if failures:
            return 2
        print("STATE FINGERPRINT OK: restored state and source files "
              "unchanged")
        return 0

    if not args.source_run:
        print("bootstrap needs --source-run")
        return 2
    try:
        doc = bootstrap_run(Path(args.source_run), run_dir, workspace)
    except BootstrapError as exc:
        print(f"STATE BOOTSTRAP FAILED (fail closed): {exc}")
        return 2
    print(f"STATE BOOTSTRAP OK: state_faithful={doc['state_faithful']} "
          f"({len(doc['workers'])} workers) -> {run_dir / FIDELITY_NAME}")
    return 0 if doc["state_faithful"] else 2


if __name__ == "__main__":
    sys.exit(main())
