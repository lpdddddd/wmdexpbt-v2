#!/usr/bin/env python3
"""Strategy Zoo v1.5 elite-heavy distill-clip validation analyzer.

Unlike the v1.4 analyzer, this report focuses on *pipeline activation* for the
clip-fix workers (p1 fast_v2 / p5 strong_v2): does the elite bank fill, are
teachers selected, does the soft activation gate open, and does the clip path
finally produce a nonzero ``actor_distill_active_fraction`` / ``weighted_loss``.
It evaluates the v1.5 hard-success criteria and classifies the failure mode when
they are not met.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

FRAME_MARKS = (500_000_000, 1_000_000_000)
TOP3_FRAME = 1_000_000_000

CLIP_PROFILES = (
    "elite_bank_actor_distill_fast_v2",
    "elite_bank_actor_distill_strong_v2",
)
ELITE_PROFILES = (
    "elite_bank_actor_distill",
    "elite_bank_actor_distill_fast_v2",
    "elite_bank_actor_distill_strong_v2",
)

# Full pipeline-activation tag set requested for v1.5 clip validation.
TB_TAGS = [
    "eir/bank_elite_bank_size",
    "eir/bank_elite_bank_add_count",
    "eir/best_source_frac_elite",
    "eir/teacher_selected_below_004",
    "eir/best_source_frac_elite_below_004",
    "eir/actor_distill_nonzero_below_004",
    "eir/actor_distill_activation_nonzero_frac",
    "eir/actor_distill_active_fraction",
    "eir/actor_distill_weighted_loss",
    "eir/actor_distill_reject_large_delta",
    "eir/actor_distill_clip_large_delta",
    "eir/actor_distill_reject_low_margin",
    "eir/actor_distill_reject_bad_source",
    "eir/actor_distill_reject_tol",
    "eir/actor_distill_reject_high_uncertainty",
    "eir/actor_distill_reject_uncertainty_missing",
    "eir/actor_distill_delta_linf_mean_all",
    "eir/elite_candidate_gate_weight",
]

# Hard-success metrics (per the v1.5 spec): for p1 OR p5 all must be > 0.
HARD_SUCCESS_KEYS = (
    "bank_elite_bank_size",
    "best_source_frac_elite",
    "actor_distill_activation_nonzero_frac",
    "actor_distill_clip_large_delta",
    "actor_distill_active_fraction",
    "actor_distill_weighted_loss",
)

RESULTS_DOCS = Path(__file__).resolve().parent / "docs"


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.is_file():
        return rows
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return rows


def _safe_float(v: Any) -> float | None:
    try:
        x = float(v)
        if math.isnan(x) or math.isinf(x):
            return None
        return x
    except (TypeError, ValueError):
        return None


def _scalar_stats(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"last": None, "mean": None, "late_mean": None, "max": None}
    late_n = max(1, len(values) // 4)
    late = values[-late_n:]
    return {
        "last": values[-1],
        "mean": sum(values) / len(values),
        "late_mean": sum(late) / len(late),
        "max": max(values),
    }


def _tb_available() -> bool:
    try:
        from tensorboard.backend.event_processing.event_accumulator import (  # noqa: F401
            EventAccumulator,
        )

        return True
    except Exception:
        return False


def _find_policy_event_files(run_dir: Path) -> dict[int, list[Path]]:
    """Return all event files per policy (across attempts), newest first."""
    out: dict[int, list[Path]] = {}
    for policy_dir in sorted(run_dir.glob("policy_*")):
        try:
            pid = int(policy_dir.name.split("_", 1)[1])
        except (IndexError, ValueError):
            continue
        candidates: list[Path] = []
        for attempt in sorted(policy_dir.glob("rl_games/attempt_*")):
            summaries = attempt / "summaries"
            if not summaries.is_dir():
                continue
            candidates.extend(summaries.glob("events.out.tfevents.*"))
        if candidates:
            out[pid] = sorted(candidates, key=lambda p: p.stat().st_mtime, reverse=True)
    return out


def _load_tb_scalars(event_path: Path, tags: list[str]) -> dict[str, list[float]]:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    acc = EventAccumulator(str(event_path), size_guidance={"scalars": 0})
    acc.Reload()
    available = set(acc.Tags().get("scalars", []))
    out: dict[str, list[float]] = {}
    for tag in tags:
        if tag not in available:
            continue
        out[tag] = [float(e.value) for e in acc.Scalars(tag)]
    return out


def _merge_attempt_scalars(event_paths: list[Path], tags: list[str]) -> dict[str, list[float]]:
    """Pick, per tag, the attempt event file with the most samples.

    Cross-restart attempts each carry their own distill telemetry; we surface the
    richest series per tag so a late attempt does not hide an earlier active one.
    """
    best: dict[str, list[float]] = {}
    for ev in event_paths:
        try:
            scal = _load_tb_scalars(ev, tags)
        except Exception:
            continue
        for tag, vals in scal.items():
            if len(vals) > len(best.get(tag, [])):
                best[tag] = vals
    return best


def _tb_section(run_dir: Path) -> dict[str, Any]:
    tb_ok = _tb_available()
    section: dict[str, Any] = {"tb_available": tb_ok, "policies": {}}
    if not tb_ok:
        section["tb_error"] = "tensorboard package unavailable"
        return section
    event_files = _find_policy_event_files(run_dir)
    section["event_files"] = {str(k): [str(p) for p in v] for k, v in event_files.items()}
    for pid, ev_paths in event_files.items():
        scalars = _merge_attempt_scalars(ev_paths, TB_TAGS)
        policy_stats: dict[str, Any] = {}
        for tag, vals in scalars.items():
            short = tag.replace("eir/", "")
            policy_stats[short] = _scalar_stats(vals)
        section["policies"][pid] = policy_stats
    return section


def _parse_run_config(run_dir: Path) -> dict[str, str]:
    cfg: dict[str, str] = {}
    path = run_dir / "RUN_CONFIG.txt"
    if not path.is_file():
        return cfg
    for line in path.read_text().splitlines():
        if "=" not in line:
            continue
        k, v = line.split("=", 1)
        cfg[k.strip()] = v.strip()
    return cfg


def _policy_profiles(run_dir: Path, cfg: dict[str, str]) -> dict[int, str]:
    profiles: dict[int, str] = {}
    for i in range(16):
        key = f"policy_{i:02d}_profile"
        if key in cfg:
            profiles[i] = cfg[key]
    lineage = run_dir / "dexpbt_8gpu_ws" / "policy_lineage.json"
    if lineage.is_file():
        try:
            data = json.loads(lineage.read_text())
            for pid_s, st in (data.get("policies") or {}).items():
                profiles[int(pid_s)] = str(st.get("wm_strategy_profile_name", "unknown"))
        except Exception:
            pass
    return profiles


def _load_curves(run_dir: Path) -> dict[int, list[dict[str, Any]]]:
    ws = run_dir / "dexpbt_8gpu_ws"
    metrics = ws / "metrics"
    out: dict[int, list[dict[str, Any]]] = {}
    if not metrics.is_dir():
        return out
    for path in sorted(metrics.glob("training_curve_policy_*.jsonl")):
        stem = path.stem.replace("training_curve_policy_", "")
        try:
            pid = int(stem)
        except ValueError:
            continue
        out[pid] = _read_jsonl(path)
    return out


def _same_frame_table(
    curves: dict[int, list[dict[str, Any]]],
    profiles: dict[int, str],
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for mark in FRAME_MARKS:
        out[str(mark)] = {}
        for pid, pts in curves.items():
            prof = profiles.get(pid, f"p{pid}")
            best = None
            for r in pts:
                f = _safe_float(r.get("env_frames"))
                if f is None or f > mark:
                    break
                best = r
            if best is None:
                continue
            out[str(mark)][prof] = {
                "policy_idx": pid,
                "nsucc_mean": _safe_float(best.get("nsucc_mean")),
                "success_tolerance": _safe_float(best.get("success_tolerance")),
            }
    return out


def _profile_copy_table(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "dexpbt_8gpu_ws" / "metrics" / "profile_copy_events.jsonl"
    events = _read_jsonl(path)
    reject_reasons: dict[str, int] = defaultdict(int)
    transitions: list[dict[str, Any]] = []
    for ev in events:
        rr = str(ev.get("reject_reason") or ev.get("decision") or "")
        if rr:
            reject_reasons[rr] += 1
        if ev.get("accepted"):
            transitions.append(
                {
                    "source_policy": ev.get("source_policy"),
                    "target_policy": ev.get("target_policy"),
                    "source_profile": ev.get("source_profile"),
                    "target_old_profile": ev.get("target_old_profile"),
                    "target_new_profile": ev.get("target_new_profile"),
                }
            )
    return {
        "event_count": len(events),
        "attempt_count": sum(1 for e in events if e.get("attempted")),
        "accept_count": sum(1 for e in events if e.get("accepted")),
        "reject_reason_counts": dict(reject_reasons),
        "transitions": transitions,
        "events_tail": events[-10:],
    }


def _lineage_table(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "dexpbt_8gpu_ws" / "policy_lineage.json"
    if not path.is_file():
        return {"present": False}
    data = json.loads(path.read_text())
    policies = data.get("policies") or {}
    unique = sorted({str(v.get("wm_strategy_profile_name")) for v in policies.values()})
    return {
        "present": True,
        "unique_profiles": unique,
        "profile_diversity_count": len(unique),
        "policies": policies,
    }


def _population_section(
    curves: dict[int, list[dict[str, Any]]],
    profiles: dict[int, str],
    frame: int = TOP3_FRAME,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for pid, pts in curves.items():
        best = None
        for r in pts:
            f = _safe_float(r.get("env_frames"))
            if f is None or f > frame:
                break
            best = r
        if best is None:
            continue
        ns = _safe_float(best.get("nsucc_mean"))
        if ns is None:
            continue
        rows.append(
            {
                "policy_idx": pid,
                "profile": profiles.get(pid, "unknown"),
                "nsucc_mean": ns,
                "env_frames": _safe_float(best.get("env_frames")),
            }
        )
    rows.sort(key=lambda x: x["nsucc_mean"], reverse=True)
    top3 = rows[:3]
    return {
        "frame": frame,
        "rows": rows,
        "top3": top3,
        "top3_nsucc_mean": (
            sum(r["nsucc_mean"] for r in top3) / len(top3) if top3 else None
        ),
    }


def _pipeline_table(tb: dict[str, Any], profiles: dict[int, str]) -> dict[str, Any]:
    """Per clip-profile (p1/p5) full pipeline-activation telemetry."""
    out: dict[str, Any] = {}
    policy_tb = tb.get("policies") or {}
    for prof in ELITE_PROFILES:
        pids = [p for p, pr in profiles.items() if pr == prof]
        for pid in pids:
            stats = policy_tb.get(pid) or policy_tb.get(str(pid)) or {}
            key = f"p{pid}_{prof}"
            row: dict[str, Any] = {"policy_idx": pid, "profile": prof}
            for tag in (t.replace("eir/", "") for t in TB_TAGS):
                row[tag] = stats.get(tag, {"last": None, "mean": None, "late_mean": None, "max": None})
            out[key] = row
    return out


def _metric_positive(stat: dict[str, Any] | None) -> bool:
    if not stat:
        return False
    for k in ("max", "last", "late_mean", "mean"):
        v = _safe_float(stat.get(k))
        if v is not None and v > 0.0:
            return True
    return False


def _evaluate_hard_success(
    tb: dict[str, Any], profiles: dict[int, str]
) -> dict[str, Any]:
    """v1.5 hard-success: for p1 OR p5 all HARD_SUCCESS_KEYS > 0. Classify failure."""
    policy_tb = tb.get("policies") or {}
    per_worker: dict[str, Any] = {}
    any_pass = False
    for prof in CLIP_PROFILES:
        pids = [p for p, pr in profiles.items() if pr == prof]
        for pid in pids:
            stats = policy_tb.get(pid) or policy_tb.get(str(pid)) or {}
            checks = {k: _metric_positive(stats.get(k)) for k in HARD_SUCCESS_KEYS}
            passed = all(checks.values())
            any_pass = any_pass or passed
            per_worker[f"p{pid}_{prof}"] = {
                "checks": checks,
                "passed": passed,
                "failure_class": _classify_failure(stats) if not passed else None,
            }
    return {
        "hard_success": any_pass,
        "criteria": list(HARD_SUCCESS_KEYS),
        "per_worker": per_worker,
    }


def _classify_failure(stats: dict[str, Any]) -> str:
    bank = _metric_positive(stats.get("bank_elite_bank_size"))
    teacher = _metric_positive(stats.get("best_source_frac_elite"))
    activation = _metric_positive(stats.get("actor_distill_activation_nonzero_frac"))
    active_frac = _metric_positive(stats.get("actor_distill_active_fraction"))
    clip = _metric_positive(stats.get("actor_distill_clip_large_delta"))
    loss = _metric_positive(stats.get("actor_distill_weighted_loss"))
    if not bank:
        return "bank_size=0: not entering bank/guidance zone; raise bank_tol or inspect curriculum/tol."
    if bank and not teacher:
        return "bank>0 but best_source_frac_elite=0: candidate retrieval/reranker issue."
    if teacher and activation and not active_frac:
        return "teacher+activation>0 but active_fraction=0: mask still rejecting; inspect reject_large_delta/low_margin/bad_source/tol."
    if clip and not active_frac:
        return "clip_large_delta>0 but active_fraction=0: clip mode not bypassing reject mask."
    if active_frac and not loss:
        return "active_fraction>0 but weighted_loss=0: loss computation/weighting bug."
    return "partial: some pipeline stages active; see per-metric stats."


def analyze(run_dir: Path) -> dict[str, Any]:
    cfg = _parse_run_config(run_dir)
    profiles = _policy_profiles(run_dir, cfg)
    curves = _load_curves(run_dir)
    tb = _tb_section(run_dir)
    return {
        "run_dir": str(run_dir),
        "config_version": cfg.get("config_version", cfg.get("profile", "")),
        "clip_validation_bank_tol_p1_p5": cfg.get("clip_validation_bank_tol_p1_p5"),
        "tb": tb,
        "pipeline_activation": _pipeline_table(tb, profiles),
        "hard_success": _evaluate_hard_success(tb, profiles),
        "profile_copy": _profile_copy_table(run_dir),
        "lineage": _lineage_table(run_dir),
        "same_frame_nsucc": _same_frame_table(curves, profiles),
        "population": _population_section(curves, profiles),
    }


def _fmt(stat: dict[str, Any] | None) -> str:
    if not stat:
        return "n/a"
    return (
        f"last={stat.get('last')} mean={stat.get('mean')} "
        f"late_mean={stat.get('late_mean')} max={stat.get('max')}"
    )


def _render_md(report: dict[str, Any]) -> str:
    lines = [
        "# Strategy Zoo v1.5 Elite-Heavy Clip Validation Report",
        "",
        f"Run: `{report.get('run_dir', '')}`",
        f"Config: `{report.get('config_version', '')}`",
        f"clip_validation_bank_tol_p1_p5: {report.get('clip_validation_bank_tol_p1_p5')}",
        f"tb_available: {(report.get('tb') or {}).get('tb_available')}",
        "",
        "## Hard-success (clip validation)",
        "",
        f"**hard_success = {report.get('hard_success', {}).get('hard_success')}**",
        f"criteria (p1 OR p5, all > 0): {report.get('hard_success', {}).get('criteria')}",
        "",
    ]
    for worker, info in sorted((report.get("hard_success") or {}).get("per_worker", {}).items()):
        lines.append(f"### {worker}")
        lines.append(f"- passed: {info.get('passed')}")
        lines.append(f"- checks: {info.get('checks')}")
        if info.get("failure_class"):
            lines.append(f"- failure_class: {info.get('failure_class')}")
        lines.append("")
    lines.append("## Pipeline activation (clip + elite workers)")
    lines.append("")
    key_metrics = [
        "bank_elite_bank_size",
        "bank_elite_bank_add_count",
        "best_source_frac_elite",
        "teacher_selected_below_004",
        "actor_distill_activation_nonzero_frac",
        "actor_distill_active_fraction",
        "actor_distill_weighted_loss",
        "actor_distill_reject_large_delta",
        "actor_distill_clip_large_delta",
        "actor_distill_reject_low_margin",
        "actor_distill_reject_bad_source",
        "actor_distill_reject_tol",
    ]
    for worker, row in sorted((report.get("pipeline_activation") or {}).items()):
        lines.append(f"### {worker}")
        for m in key_metrics:
            lines.append(f"- {m}: {_fmt(row.get(m))}")
        lines.append("")
    pc = report.get("profile_copy") or {}
    pop = report.get("population") or {}
    lines.extend(
        [
            "## Profile copy",
            "",
            f"events={pc.get('event_count')} attempts={pc.get('attempt_count')} accepts={pc.get('accept_count')}",
            f"reject_reason_counts={pc.get('reject_reason_counts')}",
            "",
            "## Population @1B",
            "",
            f"top3_nsucc_mean={pop.get('top3_nsucc_mean')}",
            f"top3={[(r.get('profile'), round(r.get('nsucc_mean'), 1)) for r in (pop.get('top3') or [])]}",
            f"profile_diversity_count={(report.get('lineage') or {}).get('profile_diversity_count')}",
            "",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze Strategy Zoo v1.5 elite-heavy clip validation run"
    )
    parser.add_argument("run_dir", type=Path)
    parser.add_argument(
        "--out-json",
        type=Path,
        default=RESULTS_DOCS / "STRATEGY_ZOO_V15_CLIP_VALIDATION_REPORT.json",
    )
    parser.add_argument(
        "--out-md",
        type=Path,
        default=RESULTS_DOCS / "STRATEGY_ZOO_V15_CLIP_VALIDATION_REPORT.md",
    )
    args = parser.parse_args()
    report = analyze(args.run_dir.resolve())
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, indent=2))
    args.out_md.write_text(_render_md(report))
    print(f"Wrote {args.out_json}")
    print(f"Wrote {args.out_md}")


if __name__ == "__main__":
    main()
