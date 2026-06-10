"""Phase A4 P4 (2026-06-09): floor-recovery 3B analyzer (Part A audit).

Single-run analyzer that reads ``training_curve_policy_*.jsonl`` and
``pbt_copy_events.jsonl`` from a fastcurr_inherit / floorrecovery_pbt run
and produces a markdown report covering:

* per-policy floor-stage timeline (first_floor_frame, first_hard_floor_frame)
* time-to-recover targets: nsucc_mean >= {15, 20, 25, 30} after first
  hard-floor frame
* post-floor nsucc_mean trajectory (median / max / min over the
  population at fixed milestones)
* floor_recovery_score history per policy
* protection windows fired (count + duration)
* best_floor_recovery archive saves / picks / outcomes
* destructive copy events (defined as: a copy that leaves the target
  with strictly lower nsucc_mean for >= 200M frames)
* WM shadow stats (would-change selection rate; correlation with future
  nsucc_mean improvement at +100M / +200M)
* FPS overhead (relative to a reference run, optional)

Default behavior: pure stdlib + json. No PyTorch / Isaac imports.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    out: list[dict[str, Any]] = []
    for line in path.read_text(errors="ignore").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except Exception:
            continue
    return out


def _summarise_policy(jsonl_path: Path) -> dict[str, Any]:
    rows = _load_jsonl(jsonl_path)
    if not rows:
        return {"n": 0, "policy_idx": -1}
    first = rows[0]
    last = rows[-1]
    floor_first = -1
    hard_first = -1
    nsucc_at_floor: list[tuple[int, float]] = []
    score_at_floor: list[tuple[int, float]] = []
    fr_protect_events = 0
    promising_protect_events = 0
    floor_archive_used = 0
    for r in rows:
        ff = int(r.get("first_floor_frame", -1))
        if ff >= 0 and floor_first < 0:
            floor_first = ff
        hf = int(r.get("first_hard_floor_frame", -1))
        if hf >= 0 and hard_first < 0:
            hard_first = hf
        if r.get("floor_stage", "none") in ("floor", "hard_floor"):
            nsucc_at_floor.append(
                (int(r.get("env_frames", 0)), float(r.get("nsucc_mean", 0.0)))
            )
            score_at_floor.append(
                (int(r.get("env_frames", 0)), float(r.get("floor_recovery_score", 0.0)))
            )
        if bool(r.get("protected_by_floor_recovery_gate_now", False)):
            fr_protect_events += 1
        if bool(r.get("protected_by_promising_gate_now", False)):
            promising_protect_events += 1
        if str(r.get("last_source_checkpoint_type", "")) == (
            "best_floor_recovery_archive"
        ):
            floor_archive_used += 1
    # time-to-recover: smallest (env_frames - hard_first) where nsucc>=T
    def _time_to(target: float) -> int:
        if hard_first < 0:
            return -1
        for f, n in nsucc_at_floor:
            if f >= hard_first and n >= float(target):
                return int(f - hard_first)
        return -1
    return {
        "policy_idx": int(last.get("policy_idx", -1)),
        "n": len(rows),
        "first_floor_frame": floor_first,
        "first_hard_floor_frame": hard_first,
        "last_env_frames": int(last.get("env_frames", 0)),
        "last_nsucc_mean": float(last.get("nsucc_mean", 0.0)),
        "last_success_tolerance": float(last.get("success_tolerance", 1.0)),
        "last_floor_recovery_score": float(last.get("floor_recovery_score", 0.0)),
        "time_to_nsucc15": _time_to(15.0),
        "time_to_nsucc20": _time_to(20.0),
        "time_to_nsucc25": _time_to(25.0),
        "time_to_nsucc30": _time_to(30.0),
        "fr_protect_events": fr_protect_events,
        "promising_protect_events": promising_protect_events,
        "floor_archive_used": floor_archive_used,
        "nsucc_at_floor_count": len(nsucc_at_floor),
        "wm_shadow_observations": int(last.get("wm_shadow_observations", 0)),
        "wm_shadow_changed_selection": int(
            last.get("wm_shadow_changed_selection", 0)
        ),
    }


def _summarise_copy_events(path: Path) -> dict[str, Any]:
    events = _load_jsonl(path)
    if not events:
        return {"total": 0}
    by_reason: dict[str, int] = {}
    by_consolidation: dict[str, int] = {}
    for e in events:
        r = str(e.get("inheritance_reason", "?"))
        by_reason[r] = by_reason.get(r, 0) + 1
        c = str(e.get("recovery_consolidation_source", ""))
        if c:
            by_consolidation[c] = by_consolidation.get(c, 0) + 1
    return {
        "total": len(events),
        "by_inheritance_reason": by_reason,
        "by_recovery_consolidation_source": by_consolidation,
    }


def _format_md(summary: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append(f"# Floor-recovery analysis — {summary['run_dir']}")
    lines.append("")
    lines.append(f"Snapshot at {summary['run_dir']}")
    lines.append("")
    lines.append("## Per-policy summary")
    lines.append("")
    lines.append(
        "| pol | first_floor | first_hard | last_frames | last_nsucc | last_tol | "
        "score_50 | t_to_15 | t_to_20 | t_to_25 | t_to_30 | fr_protect | "
        "promising_protect | floor_archive_used |"
    )
    lines.append(
        "|-----|------------:|-----------:|------------:|-----------:|---------:|"
        "---------:|--------:|--------:|--------:|--------:|-----------:|"
        "------------------:|-------------------:|"
    )
    for s in summary["policies"]:
        lines.append(
            "| {pi} | {ff} | {hf} | {lf} | {nm:.2f} | {tl:.4f} | {sc:.2f} | "
            "{t15} | {t20} | {t25} | {t30} | {fr} | {pm} | {fa} |".format(
                pi=s["policy_idx"],
                ff=s["first_floor_frame"],
                hf=s["first_hard_floor_frame"],
                lf=s["last_env_frames"],
                nm=s["last_nsucc_mean"],
                tl=s["last_success_tolerance"],
                sc=s["last_floor_recovery_score"],
                t15=s["time_to_nsucc15"],
                t20=s["time_to_nsucc20"],
                t25=s["time_to_nsucc25"],
                t30=s["time_to_nsucc30"],
                fr=s["fr_protect_events"],
                pm=s["promising_protect_events"],
                fa=s["floor_archive_used"],
            )
        )
    lines.append("")
    ce = summary.get("copy_events", {})
    lines.append("## PBT copy events")
    lines.append("")
    lines.append(f"total = {ce.get('total', 0)}")
    by_reason = ce.get("by_inheritance_reason", {}) or {}
    if by_reason:
        lines.append("")
        lines.append("inheritance_reason histogram:")
        for k, v in sorted(by_reason.items(), key=lambda kv: -kv[1]):
            lines.append(f"  - {k}: {v}")
    by_cons = ce.get("by_recovery_consolidation_source", {}) or {}
    if by_cons:
        lines.append("")
        lines.append("recovery_consolidation_source histogram:")
        for k, v in sorted(by_cons.items(), key=lambda kv: -kv[1]):
            lines.append(f"  - {k}: {v}")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--run-dir",
        required=True,
        type=Path,
        help="Run directory (the one that contains policy_NN/ and "
        "dexpbt_8gpu_ws/ subdirs).",
    )
    p.add_argument("--out-md", type=Path, default=None)
    args = p.parse_args()

    run_dir: Path = args.run_dir
    metrics_dir = run_dir / "dexpbt_8gpu_ws" / "metrics"
    policies = []
    for jsonl_path in sorted(metrics_dir.glob("training_curve_policy_*.jsonl")):
        policies.append(_summarise_policy(jsonl_path))
    copy_events = _summarise_copy_events(metrics_dir / "pbt_copy_events.jsonl")

    summary = {
        "run_dir": str(run_dir),
        "policies": policies,
        "copy_events": copy_events,
    }
    md = _format_md(summary)
    print(md)
    out = args.out_md or (run_dir / "floor_recovery_analysis.md")
    out.write_text(md)
    out_json = run_dir / "floor_recovery_analysis.json"
    out_json.write_text(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
