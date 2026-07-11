#!/usr/bin/env python3
"""Strategy Zoo v1.4 elite distill rescue + selector copy fix analyzer."""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

FRAME_MARKS = (500_000_000, 1_000_000_000, 1_500_000_000, 2_000_000_000)
ELITE_PROFILES = (
    "elite_bank_actor_distill_fast_v2",
    "elite_bank_actor_distill_fast",
    "elite_bank_actor_distill",
    "elite_bank_actor_distill_strong_v2",
    "elite_bank_actor_distill_strong",
)
SELECTOR_PROFILES = ("wm_pbt_source_selector_v2", "wm_pbt_source_selector")

TB_TAGS = [
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
    "eir/bank_elite_bank_add_count",
    "eir/bank_elite_bank_size",
    "eir/elite_candidate_gate_weight",
    "eir/actor_distill_delta_linf_mean_all",
]

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
        return {"last": None, "mean": None, "late_mean": None}
    late_n = max(1, len(values) // 4)
    late = values[-late_n:]
    return {
        "last": values[-1],
        "mean": sum(values) / len(values),
        "late_mean": sum(late) / len(late),
    }


def _tb_available() -> bool:
    try:
        from tensorboard.backend.event_processing.event_accumulator import (  # noqa: F401
            EventAccumulator,
        )

        return True
    except Exception:
        return False


def _find_policy_event_files(run_dir: Path) -> dict[int, Path]:
    out: dict[int, Path] = {}
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
            for ev in summaries.glob("events.out.tfevents.*"):
                candidates.append(ev)
        if candidates:
            out[pid] = max(candidates, key=lambda p: p.stat().st_mtime)
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


def _tb_section(run_dir: Path) -> dict[str, Any]:
    tb_ok = _tb_available()
    section: dict[str, Any] = {"tb_available": tb_ok, "policies": {}}
    if not tb_ok:
        section["tb_error"] = "tensorboard package unavailable"
        return section
    event_files = _find_policy_event_files(run_dir)
    section["event_files"] = {str(k): str(v) for k, v in event_files.items()}
    for pid, ev_path in event_files.items():
        try:
            scalars = _load_tb_scalars(ev_path, TB_TAGS)
        except Exception as exc:
            section["policies"][pid] = {"error": str(exc)}
            continue
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
            }
    return out


def _profile_copy_table(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "dexpbt_8gpu_ws" / "metrics" / "profile_copy_events.jsonl"
    events = _read_jsonl(path)
    reject_reasons: dict[str, int] = defaultdict(int)
    transitions: list[dict[str, Any]] = []
    top1_override_count = 0
    for ev in events:
        rr = str(ev.get("reject_reason") or ev.get("decision") or "")
        if rr:
            reject_reasons[rr] += 1
        if ev.get("top1_override_used"):
            top1_override_count += 1
        if ev.get("accepted"):
            transitions.append(
                {
                    "source_policy": ev.get("source_policy"),
                    "target_policy": ev.get("target_policy"),
                    "source_profile": ev.get("source_profile"),
                    "target_old_profile": ev.get("target_old_profile"),
                    "target_new_profile": ev.get("target_new_profile"),
                    "top1_override_used": ev.get("top1_override_used"),
                }
            )
    return {
        "event_count": len(events),
        "attempt_count": sum(1 for e in events if e.get("attempted")),
        "accept_count": sum(1 for e in events if e.get("accepted")),
        "reject_reason_counts": dict(reject_reasons),
        "top1_override_used_count": top1_override_count,
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
    frame: int = 2_000_000_000,
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
            }
        )
    rows.sort(key=lambda x: x["nsucc_mean"], reverse=True)
    top3 = rows[:3]
    return {
        "top3": top3,
        "top3_nsucc_mean": (
            sum(r["nsucc_mean"] for r in top3) / len(top3) if top3 else None
        ),
    }


def _elite_rescue_table(
    tb: dict[str, Any],
    profiles: dict[int, str],
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    policy_tb = tb.get("policies") or {}
    for prof in ELITE_PROFILES:
        pid = next((p for p, pr in profiles.items() if pr == prof), None)
        row: dict[str, Any] = {"policy_idx": pid, "profile": prof}
        if pid is None or pid not in policy_tb:
            row["tb_missing"] = True
            out[prof] = row
            continue
        stats = policy_tb[pid]
        for key in (
            "actor_distill_activation_nonzero_frac",
            "actor_distill_active_fraction",
            "actor_distill_weighted_loss",
            "actor_distill_reject_large_delta",
            "actor_distill_clip_large_delta",
            "actor_distill_reject_low_margin",
            "actor_distill_reject_bad_source",
            "actor_distill_reject_tol",
        ):
            row[key] = stats.get(key, {"last": None, "mean": None, "late_mean": None})
        out[prof] = row
    return out


def _elite_bank_table(
    tb: dict[str, Any],
    profiles: dict[int, str],
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    policy_tb = tb.get("policies") or {}
    for prof in ELITE_PROFILES:
        pid = next((p for p, pr in profiles.items() if pr == prof), None)
        if pid is None or pid not in policy_tb:
            continue
        stats = policy_tb[pid]
        out[prof] = {
            "policy_idx": pid,
            "best_source_frac_elite": stats.get("best_source_frac_elite"),
            "teacher_selected_below_004": stats.get("teacher_selected_below_004"),
            "bank_elite_bank_add_count_last": (stats.get("bank_elite_bank_add_count") or {}).get(
                "last"
            ),
            "bank_elite_bank_size_last": (stats.get("bank_elite_bank_size") or {}).get("last"),
        }
    return out


def analyze(run_dir: Path) -> dict[str, Any]:
    cfg = _parse_run_config(run_dir)
    profiles = _policy_profiles(run_dir, cfg)
    curves = _load_curves(run_dir)
    tb = _tb_section(run_dir)
    return {
        "run_dir": str(run_dir),
        "config_version": cfg.get("config_version", cfg.get("profile", "")),
        "tb": tb,
        "elite_distill_rescue": _elite_rescue_table(tb, profiles),
        "elite_bank": _elite_bank_table(tb, profiles),
        "profile_copy": _profile_copy_table(run_dir),
        "lineage": _lineage_table(run_dir),
        "same_frame_nsucc": _same_frame_table(curves, profiles),
        "population": _population_section(curves, profiles),
    }


def _render_md(report: dict[str, Any]) -> str:
    lines = [
        "# Strategy Zoo v1.4 Elite Rescue Report",
        "",
        f"Run: `{report.get('run_dir', '')}`",
        f"Config: `{report.get('config_version', '')}`",
        f"tb_available: {(report.get('tb') or {}).get('tb_available')}",
        "",
        "## Elite distill rescue",
        "",
    ]
    for prof, row in sorted((report.get("elite_distill_rescue") or {}).items()):
        af = (row.get("actor_distill_active_fraction") or {}).get("last")
        wl = (row.get("actor_distill_weighted_loss") or {}).get("last")
        lines.append(f"### {prof} (p{row.get('policy_idx')})")
        lines.append(f"- active_fraction last={af}")
        lines.append(f"- weighted_loss last={wl}")
        lines.append("")
    pc = report.get("profile_copy") or {}
    lines.extend(
        [
            "## Profile copy",
            "",
            f"attempts={pc.get('attempt_count')} accepts={pc.get('accept_count')}",
            f"top1_override_used_count={pc.get('top1_override_used_count')}",
            f"reject_reason_counts={pc.get('reject_reason_counts')}",
            "",
            "## Population @2B",
            "",
            f"top3_nsucc_mean={(report.get('population') or {}).get('top3_nsucc_mean')}",
            "",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze Strategy Zoo v1.4 elite rescue run")
    parser.add_argument("run_dir", type=Path)
    parser.add_argument(
        "--out-json",
        type=Path,
        default=RESULTS_DOCS / "STRATEGY_ZOO_V14_ELITE_RESCUE_REPORT.json",
    )
    parser.add_argument(
        "--out-md",
        type=Path,
        default=RESULTS_DOCS / "STRATEGY_ZOO_V14_ELITE_RESCUE_REPORT.md",
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
