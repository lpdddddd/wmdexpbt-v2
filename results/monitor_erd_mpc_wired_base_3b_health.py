#!/usr/bin/env python3
"""Early health monitor for ERD-MPC wired-base 3B runs.

Usage:
  python monitor_erd_mpc_wired_base_3b_health.py RUN_DIR [--milestone startup|50M|100M|200M] [--out DIR]

Checks startup wiring, ERD/Wired TB signals, training curves, and early-stop criteria.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import subprocess
from pathlib import Path
from typing import Any

try:
    from tensorboard.backend.event_processing import event_accumulator

    _TB_OK = True
except Exception:
    _TB_OK = False

from analyze_erd_mpc_run import (  # noqa: E402
    BLOCK_REASON_NAMES,
    ERD_TAGS,
    _block_reason_hist,
    _read_jsonl,
    _stats,
    _top_block_reason,
)

MILESTONE_FRAMES = {
    "startup": 0,
    "50M": 50_000_000,
    "100M": 100_000_000,
    "200M": 200_000_000,
}

WIRED_PREFIXES = (
    "wm_teacher/",
    "wm_imagine/",
    "wm_runtime_wiring/",
    "pbt/wm_changed_source_selection",
    "pbt/wm_shadow_changed_selection",
    "floor_recovery_score_wm",
    "wm_rssm/recovery_pos_rate",
    "wm_seq/loss_horizon_success_h4",
    "wm_seq/loss_horizon_success_h8",
    "wm_seq/loss_drop_risk_h4",
    "wm_seq/loss_drop_risk_h8",
)

ERD_POS_TAGS = [
    "erd/target_pos_rate_success_h4",
    "erd/target_pos_rate_improve_h4",
    "erd/target_pos_rate_drop_h4",
    "erd/target_pos_rate_success_h8",
    "erd/target_pos_rate_improve_h8",
    "erd/target_pos_rate_drop_h8",
]


def _collect_tb(summary_dirs: list[Path]) -> dict[str, list[tuple[int, float]]]:
    merged: dict[str, list[tuple[int, float]]] = {}
    if not _TB_OK:
        return merged
    for d in summary_dirs:
        for ev in sorted(d.glob("events.out.tfevents.*")):
            try:
                ea = event_accumulator.EventAccumulator(
                    str(ev), size_guidance={"scalars": 0}
                )
                ea.Reload()
            except Exception:
                continue
            for tag in ea.Tags().get("scalars", []):
                keep = (
                    tag in ERD_TAGS
                    or tag in ERD_POS_TAGS
                    or tag.startswith("erd/target_pos_rate_")
                    or any(tag.startswith(p) for p in WIRED_PREFIXES)
                )
                if not keep:
                    continue
                merged.setdefault(tag, [])
                for s in ea.Scalars(tag):
                    merged[tag].append((int(s.step), float(s.value)))
    for v in merged.values():
        v.sort(key=lambda x: x[0])
    return merged


def _policy_startup(run_dir: Path, pid: int) -> dict[str, Any]:
    pdir = run_dir / f"policy_{pid:02d}"
    log = pdir / "train.log"
    out: dict[str, Any] = {
        "policy": pid,
        "dir_exists": pdir.is_dir(),
        "train_log_bytes": log.stat().st_size if log.exists() else 0,
        "first_epoch": False,
        "startup_hang": False,
        "erd_enabled_log": False,
        "teacher_log": False,
        "imagine_log": False,
        "status": (pdir / "status.txt").read_text().strip()
        if (pdir / "status.txt").exists()
        else "",
    }
    if not log.exists():
        return out
    text = log.read_text(errors="replace")
    out["first_epoch"] = "epoch: 1/" in text or "epoch: 2/" in text
    out["startup_hang"] = "startup_hang" in text or "reason=startup_hang" in text
    out["erd_enabled_log"] = "[ERD] enabled:" in text
    out["teacher_log"] = bool(
        re.search(r"TeacherManager|iterative_teacher|wm_teacher", text, re.I)
    )
    out["imagine_log"] = bool(
        re.search(r"ImaginedEvaluator|imagined_eval|wm_imagine", text, re.I)
    )
    m = re.findall(r"epoch: (\d+)/(\d+).*?frames: (\d+)", text)
    if m:
        ep, max_ep, frames = m[-1]
        out["latest_epoch"] = int(ep)
        out["max_epoch"] = int(max_ep)
        out["latest_frames_log"] = int(frames)
    return out


def _curve_summary(run_dir: Path, pid: int) -> dict[str, Any]:
    path = (
        run_dir
        / "dexpbt_8gpu_ws"
        / "metrics"
        / f"training_curve_policy_{pid:02d}.jsonl"
    )
    rows = _read_jsonl(path)
    if not rows:
        return {"policy": pid, "curve_points": 0}
    means = [float(r.get("nsucc_mean", 0.0)) for r in rows]
    maxs = [float(r.get("nsucc_max", 0.0)) for r in rows]
    tols = [float(r.get("success_tolerance", 1.0)) for r in rows]
    last = rows[-1]
    peak = max(means) if means else 0.0
    end = means[-1] if means else 0.0
    return {
        "policy": pid,
        "curve_points": len(rows),
        "frames_end": int(last.get("env_frames", 0)),
        "nsucc_end": end,
        "nsucc_peak": peak,
        "nsucc_max_peak": max(maxs) if maxs else 0.0,
        "tol_end": tols[-1],
        "tol_min": min(tols) if tols else 1.0,
        "retention": end / peak if peak > 1e-9 else 0.0,
        "restart_count": int(last.get("restart_count", 0) or 0),
        "rollback_count": int(last.get("rollback_count", 0) or 0),
        "collapse_count": sum(1 for r in rows if r.get("early_collapse_triggered")),
    }


def _tb_policy_summary(tb: dict[str, list[tuple[int, float]]], pid: int) -> dict[str, Any]:
    out: dict[str, Any] = {"policy": pid, "tb_tags": 0}
    for tag in ERD_TAGS + ERD_POS_TAGS:
        vals = [v for _, v in tb.get(tag, [])]
        if vals:
            out[tag.split("/", 1)[1]] = _stats(vals)
            out["tb_tags"] += 1
    block_vals = [v for _, v in tb.get("erd/distill_block_reason", [])]
    hist = _block_reason_hist(block_vals)
    out["distill_block_reason_histogram"] = hist
    out["top_block_reason"] = _top_block_reason(hist)
    out["scorer_active_points"] = sum(
        1 for _, v in tb.get("erd/scorer_active", []) if v > 0
    )
    out["distill_active_points"] = sum(
        1 for _, v in tb.get("erd/distill_active", []) if v > 0
    )
    wired_hits = {}
    for prefix in WIRED_PREFIXES:
        hits = [tag for tag in tb if tag.startswith(prefix) or tag == prefix]
        if hits:
            wired_hits[prefix.rstrip("/")] = len(hits)
    out["wired_tag_families"] = wired_hits
    return out


def _evaluate_alerts(
    *,
    milestone: str,
    startup: list[dict[str, Any]],
    curves: list[dict[str, Any]],
    tb_by_policy: list[dict[str, Any]],
) -> list[str]:
    alerts: list[str] = []
    target_frames = MILESTONE_FRAMES.get(milestone, 0)

    if milestone == "startup":
        if not all(s.get("dir_exists") for s in startup):
            alerts.append("Not all 6 policy directories exist yet (stagger may still be warming).")
        if any(s.get("startup_hang") for s in startup):
            alerts.append("startup_hang detected in at least one policy.")
        launched = [s for s in startup if s.get("train_log_bytes", 0) > 0]
        if launched and not all(s.get("erd_enabled_log") for s in launched):
            alerts.append("ERD wiring log missing on one or more launched policies.")
        if launched and not all(s.get("first_epoch") for s in launched):
            alerts.append("At least one launched policy has not reached first epoch yet.")
        return alerts

    pop_frames = max((c.get("frames_end", 0) for c in curves), default=0)
    if target_frames and pop_frames < target_frames * 0.8:
        alerts.append(
            f"Population max frames {pop_frames:,} below milestone {milestone} target."
        )

    if all((t.get("scorer_active_points", 0) == 0 for t in tb_by_policy if t.get("tb_tags"))):
        if tb_by_policy and any(t.get("tb_tags") for t in tb_by_policy):
            alerts.append("erd/scorer_active is 0 across all policies with TB data.")

    if target_frames >= 200_000_000:
        distill_pts = sum(t.get("distill_active_points", 0) for t in tb_by_policy)
        if distill_pts == 0:
            reasons = [
                t.get("top_block_reason")
                for t in tb_by_policy
                if t.get("top_block_reason") not in (None, "warmup_frames", "warmup_iters")
            ]
            if reasons:
                alerts.append(
                    "At/after 200M: distill_active still 0 and top block reasons are "
                    f"{sorted(set(reasons))} (not warmup)."
                )

    for t in tb_by_policy:
        pid = t.get("policy")
        raw_clip = t.get("raw_mu_clip_fraction", {})
        if raw_clip.get("max", 0.0) > 0.3:
            alerts.append(f"p{pid:02d}: raw_mu_clip_fraction max={raw_clip.get('max'):.3f} > 0.3")
        cand_clip = t.get("candidate_clip_fraction", {})
        if cand_clip.get("mean", 0.0) > 0.2:
            alerts.append(
                f"p{pid:02d}: candidate_clip_fraction mean={cand_clip.get('mean'):.3f} high"
            )
        ratio = t.get("distill_grad_ratio", {})
        scale = t.get("auto_scale_factor", {})
        if ratio.get("mean", 0.0) > 0.2 and scale.get("mean", 1.0) < 0.3:
            alerts.append(
                f"p{pid:02d}: distill_grad_ratio capped (mean ratio={ratio.get('mean'):.3f}, "
                f"auto_scale={scale.get('mean'):.3f})"
            )
        for key in ("success_h4", "improve_h4", "drop_h4", "success_h8", "improve_h8", "drop_h8"):
            pr = t.get(f"target_pos_rate_{key}", {})
            if pr.get("n", 0) and (
                pr.get("mean", 0.5) <= 1e-3 or pr.get("mean", 0.5) >= 1.0 - 1e-3
            ):
                alerts.append(
                    f"p{pid:02d}: target_pos_rate_{key} degenerate mean={pr.get('mean'):.4f}"
                )
        wired = t.get("wired_tag_families", {})
        if t.get("tb_tags", 0) > 20 and not wired:
            alerts.append(f"p{pid:02d}: wired stack TB families all absent.")

    ends = [c.get("nsucc_end", 0.0) for c in curves if c.get("curve_points")]
    if ends and statistics.fmean(ends) < 3.0 and target_frames >= 50_000_000:
        alerts.append(f"Population nsucc_end mean low: {statistics.fmean(ends):.2f}")

    return alerts


def build_report(run_dir: Path, milestone: str = "startup") -> dict[str, Any]:
    run_dir = run_dir.resolve()
    startup = [_policy_startup(run_dir, pid) for pid in range(6)]
    curves = [_curve_summary(run_dir, pid) for pid in range(6)]
    tb_by_policy = []
    for pid in range(6):
        summary_dirs = sorted(
            (run_dir / f"policy_{pid:02d}" / "rl_games").glob("attempt_*/summaries")
        )
        tb_by_policy.append(_tb_policy_summary(_collect_tb(summary_dirs), pid))

    pop_frames = max((c.get("frames_end", 0) for c in curves), default=0)
    pop_end = [c.get("nsucc_end", 0.0) for c in curves if c.get("curve_points")]
    alerts = _evaluate_alerts(
        milestone=milestone,
        startup=startup,
        curves=curves,
        tb_by_policy=tb_by_policy,
    )

    proc_count = 0
    try:
        proc_count = int(
            subprocess.check_output(
                [
                    "bash",
                    "-lc",
                    f"pgrep -af {run_dir.name} | grep -v pgrep | wc -l",
                ],
                text=True,
            ).strip()
            or 0
        )
    except Exception:
        pass

    return {
        "run_dir": str(run_dir),
        "milestone": milestone,
        "target_frames": MILESTONE_FRAMES.get(milestone, pop_frames),
        "population_frames_max": pop_frames,
        "population_nsucc_end_mean": statistics.fmean(pop_end) if pop_end else 0.0,
        "population_nsucc_end_max": max(pop_end) if pop_end else 0.0,
        "process_count": proc_count,
        "tb_available": _TB_OK,
        "startup": startup,
        "curves": curves,
        "tb_by_policy": tb_by_policy,
        "alerts": alerts,
        "continue_run": len(alerts) == 0 or milestone == "startup",
    }


def _write_md(report: dict[str, Any], out_path: Path) -> None:
    lines = [
        f"# ERD-MPC Wired-Base Health Report ({report['milestone']})",
        "",
        f"- run_dir: `{report['run_dir']}`",
        f"- population_frames_max: {report['population_frames_max']:,}",
        f"- population_nsucc_end_mean/max: {report['population_nsucc_end_mean']:.2f} / {report['population_nsucc_end_max']:.2f}",
        f"- process_count: {report['process_count']}",
        f"- tb_available: {report['tb_available']}",
        f"- continue_run: **{report['continue_run']}**",
        "",
        "## Alerts",
    ]
    if report["alerts"]:
        for a in report["alerts"]:
            lines.append(f"- {a}")
    else:
        lines.append("- none")
    lines += ["", "## Startup"]
    for s in report["startup"]:
        lines.append(
            f"- p{s['policy']:02d}: dir={s['dir_exists']} log={s['train_log_bytes']}B "
            f"first_epoch={s.get('first_epoch')} erd={s.get('erd_enabled_log')} "
            f"hang={s.get('startup_hang')} status={s.get('status') or '-'}"
        )
    lines += ["", "## Curves"]
    for c in report["curves"]:
        if not c.get("curve_points"):
            lines.append(f"- p{c['policy']:02d}: no curve yet")
            continue
        lines.append(
            f"- p{c['policy']:02d}: frames={c['frames_end']:,} nsucc={c['nsucc_end']:.2f}/"
            f"{c['nsucc_peak']:.2f} tol={c['tol_end']:.4f} restarts={c['restart_count']}"
        )
    lines += ["", "## ERD TB snapshot"]
    for t in report["tb_by_policy"]:
        if not t.get("tb_tags"):
            lines.append(f"- p{t['policy']:02d}: no TB yet")
            continue
        lines.append(
            f"- p{t['policy']:02d}: scorer_pts={t.get('scorer_active_points',0)} "
            f"distill_pts={t.get('distill_active_points',0)} "
            f"top_block={t.get('top_block_reason')} "
            f"score_margin={t.get('score_margin', {})} "
            f"raw_mu_clip={t.get('raw_mu_clip_fraction', {})}"
        )
    out_path.write_text("\n".join(lines))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--milestone", default="startup", choices=list(MILESTONE_FRAMES))
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    run_dir = Path(args.run_dir).resolve()
    out_dir = Path(args.out).resolve() if args.out else run_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    report = build_report(run_dir, milestone=args.milestone)
    json_path = out_dir / f"ERD_MPC_WIRED_HEALTH_{args.milestone}.json"
    md_path = out_dir / f"ERD_MPC_WIRED_HEALTH_{args.milestone}.md"
    json_path.write_text(json.dumps(report, indent=2))
    _write_md(report, md_path)
    print(f"[monitor] wrote {md_path}")
    if report["alerts"]:
        print("[monitor] ALERTS:")
        for a in report["alerts"]:
            print(f"  - {a}")


if __name__ == "__main__":
    main()
