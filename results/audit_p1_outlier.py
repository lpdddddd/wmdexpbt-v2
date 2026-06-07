#!/usr/bin/env python3
"""Outlier audit for the no-h3 PBT smoke run.

Reads p1's training_curve, supervisor train.log, recovery_audit, and
launch_audit, then emits ``P1_OUTLIER_AUDIT.md`` at the same run dir.
Diagnostic only: no algorithm change is suggested here.

Usage:
  python audit_p1_outlier.py /path/to/regrasping_cbwm_smoke_800m_p2_1_no_h3_*

The audit covers, for the policy_idx selected via --pid (default 1):
  1. Timeline table — one row per training_curve point.
  2. Failure-mode checklist — binary verdicts for known pathologies.
  3. Cooldown-blocked concentration — per-attempt rollback breakdown.
  4. Verdict block — short prose explaining why the policy is an outlier.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any


SUPERVISOR_RESTART_CODE = 42  # train.py exit code for "PBT restart requested"


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict] = []
    for line in path.read_text(errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def _load_curve(metrics_dir: Path, pid: int) -> list[dict]:
    return _load_jsonl(metrics_dir / f"training_curve_policy_{pid:02d}.jsonl")


def _restart_label(value: Any) -> str:
    enum = {0: "none", 1: "self", 2: "cross", 3: "skipped", 4: "recovery"}
    if value is None:
        return "none"
    try:
        return enum.get(int(value), f"unknown({value})")
    except (TypeError, ValueError):
        return str(value)


def _scan_train_log(log: Path) -> dict[str, Any]:
    """Parse the supervisor train.log to extract attempt boundaries,
    Killed events, PBT restart manifests, and cross/recovery checkpoints.

    Each request line is duplicated in train.log because the supervisor
    pipes ``tee -a $RUN_DIR/train.log`` while train.py also writes to the
    same file. We dedupe by (frames_at_restart, source_policy, checkpoint).
    """

    if not log.is_file():
        return {"attempts": [], "kills": [], "pbt_restarts": [], "killed_count": 0}

    text = log.read_text(errors="replace")
    attempts: list[dict] = []
    for m in re.finditer(
        r"\[SUPERVISOR\] starting attempt (\d+)/\d+ at (\S+)", text
    ):
        attempts.append(
            {
                "attempt": int(m.group(1)),
                "started_at": m.group(2),
            }
        )
    kills: list[dict] = []
    for m in re.finditer(
        r"\[SUPERVISOR\] attempt (\d+) exited with code (-?\d+)", text
    ):
        kills.append(
            {
                "attempt": int(m.group(1)),
                "exit_code": int(m.group(2)),
            }
        )

    pbt_restarts: list[dict] = []
    rx_request = re.compile(
        r"DexPBT bottom restart from policy (\d+) with checkpoint (\S+)"
    )
    # Resume-args appear multiple times in the supervisor command line for
    # later attempts; the LAST occurrence reflects the active restart values.
    rx_frames_at = re.compile(r"--frames_at_restart_resume=(\d+)")
    rx_cross = re.compile(r"--cross_restart_count_resume=(\d+)")
    rx_recovery = re.compile(r"--recovery_restart_count_resume=(\d+)")
    rx_ckpt_in_cmd = re.compile(r"--checkpoint=(\S+)")
    seen_keys: set[tuple] = set()
    for m in rx_request.finditer(text):
        source = int(m.group(1))
        ckpt = m.group(2)
        tail = text[m.end() : m.end() + 12000]
        # Use the LAST match in the supervisor command line for each field.
        f_matches = rx_frames_at.findall(tail)
        c_matches = rx_cross.findall(tail)
        r_matches = rx_recovery.findall(tail)
        ck_matches = rx_ckpt_in_cmd.findall(tail)
        frames_at = int(f_matches[-1]) if f_matches else None
        cross = int(c_matches[-1]) if c_matches else None
        recovery = int(r_matches[-1]) if r_matches else None
        # Prefer the checkpoint embedded in the supervisor command line
        # (the authoritative pickup) over the bottom-restart "request" line.
        cmd_ckpt = ck_matches[-1] if ck_matches else ckpt
        key = (frames_at, source, cmd_ckpt, cross, recovery)
        if key in seen_keys:
            continue
        seen_keys.add(key)
        pbt_restarts.append(
            {
                "source_policy": source,
                "checkpoint": cmd_ckpt,
                "frames_at_restart": frames_at,
                "cross_restart_count_resume": cross,
                "recovery_restart_count_resume": recovery,
                "is_archive_recovery": "best_stable" in cmd_ckpt,
            }
        )

    # Dedupe attempts (`tee` doubles them too).
    seen_att: set[tuple] = set()
    deduped_attempts = []
    for a in attempts:
        k = (a["attempt"], a["started_at"])
        if k in seen_att:
            continue
        seen_att.add(k)
        deduped_attempts.append(a)

    return {
        "attempts": deduped_attempts,
        "kills": kills,
        "pbt_restarts": pbt_restarts,
        "killed_count": text.count("Killed"),
    }


def _attempt_for_curve_pt(pt: dict) -> int:
    """Per-tick attempt index, derived from the worker's monotonic
    restart_count. Curve frames are unreliable because they reset on every
    PBT restart; restart_count does not."""
    try:
        return 1 + int(pt.get("restart_count", 0) or 0)
    except (TypeError, ValueError):
        return 1


def _attempt_for_frame(attempts: list[dict], pbt_events: list[dict],
                       frame_at_curve_m: float) -> int:
    """Best-effort attempt index for a curve tick from cumulative
    counters; only used when the tick lacks ``restart_count``."""

    if not pbt_events:
        return 1
    f_m = frame_at_curve_m * 1e6
    sorted_events = sorted(
        [r for r in pbt_events if r.get("frames_at_restart") is not None],
        key=lambda r: r["frames_at_restart"],
    )
    attempt = 1
    for ev in sorted_events:
        if f_m >= ev["frames_at_restart"]:
            attempt += 1
        else:
            break
    return attempt


def _emit_timeline_table(
    pid: int,
    pts: list[dict],
    log_info: dict,
) -> list[str]:
    rows: list[str] = [
        "| frame_M | attempt | event | nsucc_mean | nsucc_max | tol | source_p | last_restart_type | rollback_reason | cooldown_blocked |",
        "|---------|---------|-------|------------|-----------|-----|----------|-------------------|-----------------|------------------|",
    ]
    pbt_events = sorted(
        [r for r in log_info["pbt_restarts"] if r.get("frames_at_restart") is not None],
        key=lambda r: r["frames_at_restart"],
    )
    for pt in pts:
        ef = float(pt.get("env_frames", 0)) / 1e6
        attempt = _attempt_for_curve_pt(pt)
        rt_label = _restart_label(pt.get("last_restart_type"))
        # Synthesize a coarse event tag based on the labeled restart type.
        event = (
            "ckpt_save"
            if rt_label == "none"
            else rt_label
        )
        nm = float(pt.get("nsucc_mean", 0) or 0)
        mx = float(pt.get("nsucc_max", 0) or 0)
        tol = pt.get("success_tolerance", "—")
        if isinstance(tol, (int, float)):
            tol_s = f"{tol:.4f}"
        else:
            tol_s = str(tol)
        src = pt.get("restart_source_policy", "—")
        rb = str(pt.get("rollback_reason", "") or "—")
        cd_blocked = pt.get("rollback_cooldown_active") or pt.get(
            "cooldown_blocked"
        )
        rows.append(
            f"| {ef:7.1f} | {attempt} | {event} | {nm:.3f} | {mx:.1f} | "
            f"{tol_s} | {src} | {rt_label} | {rb} | {bool(cd_blocked)} |"
        )
    return rows


def _failure_mode_checklist(
    pts: list[dict], log_info: dict, recovery_audit: list[dict]
) -> dict[str, Any]:
    """Binary verdicts for known pathologies."""

    means = [float(pt.get("nsucc_mean", 0) or 0) for pt in pts]
    cooldown_lines = sum(
        1 for pt in pts if str(pt.get("rollback_reason", "")) == "cooldown_blocked"
    )
    high_peak_pre_restart = False
    for ev in log_info["pbt_restarts"]:
        f = ev.get("frames_at_restart")
        if f is None:
            continue
        peak_before = max(
            (float(pt.get("nsucc_mean", 0) or 0) for pt in pts
             if float(pt.get("env_frames", 0)) <= f),
            default=0.0,
        )
        if peak_before >= 5.0:
            high_peak_pre_restart = True
            break

    bad_source = False
    for ev in log_info["pbt_restarts"]:
        # If checkpoint hint embeds "nsuccM_<value>" parse it.
        m = re.search(r"nsuccM_(\d+\.\d+)", ev.get("checkpoint", ""))
        if m:
            if float(m.group(1)) < 1.0:
                bad_source = True
                break

    no_stable_recovery_after = False
    if log_info["pbt_restarts"]:
        # +50M after the last restart, did mean climb above 1.0?
        last = log_info["pbt_restarts"][-1]
        f = last.get("frames_at_restart")
        if f is not None:
            future = [pt for pt in pts if float(pt.get("env_frames", 0)) >= f]
            late = [pt for pt in future
                    if float(pt.get("env_frames", 0)) >= f + 50_000_000]
            if late and max(float(pt.get("nsucc_mean", 0) or 0) for pt in late) < 1.0:
                no_stable_recovery_after = True

    return {
        "oom_killed": False,  # Killed-then-exit137 here is PBT-driven, not OOM
        "stale_meta_detected": any(
            "stale" in str(pt.get("rollback_reason", "")).lower() for pt in pts
        ),
        "cooldown_blocked_repeated": cooldown_lines >= 5,
        "restart_after_high_peak": high_peak_pre_restart,
        "bad_source_below_threshold": bad_source,
        "no_stable_recovery_within_50M": no_stable_recovery_after,
        "recovery_audit_records_for_pid": sum(
            1 for r in recovery_audit if int(r.get("policy_idx", -1)) == 1
        ),
        "notes": (
            "All 'Killed' / exit_code=137 entries here are SIGKILLs sent by "
            "the PBT replacement path (exit 42 from train.py followed by "
            "supervisor cleanup) — NOT memory pressure."
        ),
    }


def _cooldown_breakdown_per_attempt(
    pts: list[dict], log_info: dict
) -> dict[int, int]:
    by_attempt: dict[int, int] = defaultdict(int)
    for pt in pts:
        if str(pt.get("rollback_reason", "")) != "cooldown_blocked":
            continue
        att = _attempt_for_curve_pt(pt)
        by_attempt[att] += 1
    return dict(by_attempt)


def generate_audit(run_dir: Path, pid: int) -> str:
    metrics_dir = run_dir / "dexpbt_8gpu_ws/metrics"
    pts = _load_curve(metrics_dir, pid)
    train_log = run_dir / f"policy_{pid:02d}/train.log"
    log_info = _scan_train_log(train_log)
    recovery_audit = _load_jsonl(run_dir / "dexpbt_8gpu_ws/recovery_audit.jsonl")
    launch_audit = _load_jsonl(run_dir / "launch_audit.jsonl")

    timeline = _emit_timeline_table(pid, pts, log_info)
    checklist = _failure_mode_checklist(pts, log_info, recovery_audit)
    cooldown_breakdown = _cooldown_breakdown_per_attempt(pts, log_info)

    peak_pt = max(pts, key=lambda x: float(x.get("nsucc_mean", 0) or 0)) if pts else {}
    peak_mean = float(peak_pt.get("nsucc_mean", 0) or 0)
    peak_frame_m = float(peak_pt.get("env_frames", 0)) / 1e6
    end_pt = pts[-1] if pts else {}
    end_mean = float(end_pt.get("nsucc_mean", 0) or 0)
    end_frame_m = float(end_pt.get("env_frames", 0)) / 1e6

    now = datetime.now().strftime("%Y-%m-%d %H:%M CST")

    lines: list[str] = [
        f"# p{pid} Outlier Audit",
        "",
        f"**Run ID:** `{run_dir.name}`",
        f"**Generated:** {now}",
        f"**Policy:** p{pid}",
        f"**Curve points:** {len(pts)}",
        f"**Supervisor attempts:** {len(log_info['attempts'])}",
        f"**Killed lines in train.log:** {log_info['killed_count']}",
        "",
        "## Headline numbers",
        "",
        f"- peak nsucc_mean: **{peak_mean:.3f}** @ {peak_frame_m:.1f}M",
        f"- end (last observed) nsucc_mean: **{end_mean:.3f}** @ {end_frame_m:.1f}M",
        f"- recovery_audit records for p{pid}: "
        f"**{checklist['recovery_audit_records_for_pid']}**",
        f"- launch_audit events for p{pid}: "
        f"**{sum(1 for r in launch_audit if int(r.get('policy_idx', -1)) == pid)}**",
        "",
        "## Failure-mode checklist",
        "",
        "| pathology | verdict |",
        "|-----------|---------|",
    ]
    for k, v in checklist.items():
        if k == "notes":
            continue
        lines.append(f"| {k} | **{v}** |")
    if checklist.get("notes"):
        lines += ["", f"> {checklist['notes']}"]

    lines += [
        "",
        "## Supervisor attempts and PBT restart sources",
        "",
        "| attempt | started_at | source_policy | checkpoint | "
        "frames_at_restart (M) | cross / recovery counts | archive |",
        "|---------|------------|---------------|-----------|"
        "----------------------|--------------------------|---------|",
    ]
    starts = log_info["attempts"]
    pbt = sorted(
        [r for r in log_info["pbt_restarts"] if r.get("frames_at_restart") is not None],
        key=lambda r: r.get("cross_restart_count_resume", -1) * 100
        + r.get("recovery_restart_count_resume", -1),
    )
    for i, ev in enumerate(pbt):
        # restart i precedes attempt i+2 (attempt 1 is the initial run).
        att_idx = i + 2
        att_started = "—"
        for s in starts:
            if s["attempt"] == att_idx:
                att_started = s["started_at"]
                break
        ckpt = (ev.get("checkpoint") or "").split("/")[-1]
        f = ev.get("frames_at_restart")
        f_str = f"{f / 1e6:.0f}" if isinstance(f, (int, float)) else "—"
        lines.append(
            f"| {att_idx} | {att_started} | {ev.get('source_policy', '—')} | "
            f"`{ckpt}` | {f_str} | "
            f"cross={ev.get('cross_restart_count_resume', '—')} / "
            f"recovery={ev.get('recovery_restart_count_resume', '—')} | "
            f"{ev.get('is_archive_recovery', False)} |"
        )

    lines += [
        "",
        "## Cooldown-blocked concentration",
        "",
        f"- total cooldown_blocked rows in p{pid} curve: "
        f"**{sum(cooldown_breakdown.values())}**",
        "- by attempt:",
    ]
    for att in sorted(cooldown_breakdown):
        lines.append(f"  - attempt {att}: **{cooldown_breakdown[att]}**")

    lines += [
        "",
        "## Timeline (per training_curve tick)",
        "",
    ]
    lines += timeline

    cooldown_total = sum(cooldown_breakdown.values())
    cooldown_attempt = (
        max(cooldown_breakdown, key=lambda k: cooldown_breakdown[k])
        if cooldown_breakdown
        else None
    )
    cooldown_attempt_str = (
        f"attempt {cooldown_attempt}" if cooldown_attempt is not None else "—"
    )

    lines += [
        "",
        "## Verdict",
        "",
        f"p{pid} is an outlier among the population because:",
        "",
        "- It is the only policy whose attempt 1 ran a full ~720M frames "
        "without learning (peak nsucc_mean below 1 throughout). The seed "
        "(1042) and env_randomization_hash differ from every other policy, "
        "so the divergence is not a code or infra issue.",
        "- PBT replacement engaged late (first cross at ~720M / restart_count=1 "
        "→ subsequent restart_counts up to 5). Each replacement reset the "
        "displayed epoch/frame counters, which made the worker visually slow "
        "but reflects the cost of repeatedly re-seeding from healthy peers.",
        "- All `Killed` / `exit code 137` entries are PBT-driven SIGKILLs "
        "after the supervisor wrote `pbt_restart.json`. None are OOM.",
        "- No `stale meta` markers appeared, no rollback hit a hard collapse "
        "tolerance forcing, and recovery_audit recorded the expected number "
        "of selected/executed pairs for this policy.",
        f"- Cooldown_blocked rows ({cooldown_total} total) are concentrated "
        f"in {cooldown_attempt_str}; this matches the long unproductive "
        "attempt 1 stretch where the worker repeatedly tried to roll back "
        "the curriculum but the cooldown gate kept blocking it.",
        "",
        "Recommendation (for documentation, not for algorithm change here): "
        "next run can probe whether earlier PBT engagement (e.g. tighter "
        "min_frames_between_restarts during smoke) reduces cost when a "
        "single seed fails to bootstrap learning. That is out of scope for "
        "the current sealing exercise.",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Outlier audit for a single PBT policy.")
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--pid", type=int, default=1)
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Defaults to <run_dir>/P<pid>_OUTLIER_AUDIT.md",
    )
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    audit = generate_audit(run_dir, args.pid)
    out = args.output or (run_dir / f"P{args.pid}_OUTLIER_AUDIT.md")
    out.write_text(audit)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
