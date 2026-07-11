#!/usr/bin/env python3
"""Strategy Zoo v1.3 elite-centered performance-debug run analyzer.

Produces same-frame nsucc checkpoints, throughput vs p0, elite-distill metrics
for p1/p4/p5, WM source selector metrics for p2/p3, profile-copy events,
lineage, and population summary.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any


FRAME_MARKS = (500_000_000, 1_000_000_000, 1_500_000_000, 2_000_000_000)
ELITE_PROFILES = (
    "elite_bank_actor_distill_fast",
    "elite_bank_actor_distill",
    "elite_bank_actor_distill_strong",
)
SELECTOR_PROFILES = ("wm_pbt_source_selector_v2", "wm_pbt_source_selector")


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


def _mean_field(rows: list[dict], key: str) -> float | None:
    vals = [_safe_float(r.get(key)) for r in rows]
    vals = [v for v in vals if v is not None]
    if not vals:
        return None
    return sum(vals) / len(vals)


def _throughput_table(
    curves: dict[int, list[dict[str, Any]]],
    profiles: dict[int, str],
) -> dict[str, Any]:
    rows: dict[str, Any] = {}
    ref_tp = None
    for pid in sorted(curves):
        prof = profiles.get(pid, f"p{pid}")
        pts = curves[pid]
        if len(pts) < 2:
            continue
        t0 = _safe_float(pts[0].get("timestamp"))
        t1 = _safe_float(pts[-1].get("timestamp"))
        f0 = _safe_float(pts[0].get("env_frames"))
        f1 = _safe_float(pts[-1].get("env_frames"))
        if None in (t0, t1, f0, f1) or t1 <= t0 or f1 <= f0:
            continue
        tp = (f1 - f0) / (t1 - t0)
        nsucc = _safe_float(pts[-1].get("nsucc_mean"))
        if prof == "ppo_only":
            ref_tp = tp
        rows[prof] = {
            "policy_idx": pid,
            "throughput_frames_per_s": tp,
            "nsucc_last": nsucc,
            "throughput_ratio_vs_p0": None,
        }
    if ref_tp and ref_tp > 0:
        for prof, row in rows.items():
            row["throughput_ratio_vs_p0"] = row["throughput_frames_per_s"] / ref_tp
    return rows


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


def _top_reason(rows: list[dict], key: str) -> str | None:
    counts: dict[str, int] = defaultdict(int)
    for r in rows:
        val = str(r.get(key) or "")
        if val:
            counts[val] += 1
    if not counts:
        return None
    return max(counts, key=counts.get)


def _elite_distill_section(
    curves: dict[int, list[dict[str, Any]]],
    profiles: dict[int, str],
    throughput: dict[str, Any],
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for prof in ELITE_PROFILES:
        pid = next((p for p, pr in profiles.items() if pr == prof), None)
        rows = curves.get(pid, []) if pid is not None else []
        out[prof] = {
            "policy_idx": pid,
            "throughput_ratio_vs_p0": (throughput.get(prof) or {}).get(
                "throughput_ratio_vs_p0"
            ),
            "best_source_frac_elite_mean": _mean_field(rows, "best_source_frac_elite"),
            "teacher_selected_above_004_mean": _mean_field(
                rows, "teacher_selected_above_004"
            ),
            "teacher_selected_below_004_mean": _mean_field(
                rows, "teacher_selected_below_004"
            ),
            "best_source_frac_elite_below_004_mean": _mean_field(
                rows, "best_source_frac_elite_below_004"
            ),
            "actor_distill_nonzero_below_004_mean": _mean_field(
                rows, "actor_distill_nonzero_below_004"
            ),
            "actor_distill_activation_nonzero_frac_mean": _mean_field(
                rows, "actor_distill_activation_nonzero_frac"
            ),
            "actor_distill_weighted_loss_mean": _mean_field(
                rows, "actor_distill_weighted_loss"
            ),
            "bank_elite_bank_add_count_last": (
                _safe_float(rows[-1].get("bank_elite_bank_add_count"))
                if rows
                else None
            ),
        }
    return out


def _selector_section(
    curves: dict[int, list[dict[str, Any]]],
    profiles: dict[int, str],
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for prof in SELECTOR_PROFILES:
        pid = next((p for p, pr in profiles.items() if pr == prof), None)
        rows = curves.get(pid, []) if pid is not None else []
        wm_used = [1.0 if r.get("wm_score_used") else 0.0 for r in rows]
        out[prof] = {
            "policy_idx": pid,
            "wm_score_used_frac": (sum(wm_used) / len(wm_used)) if wm_used else None,
            "wm_score_effective_weight_mean": _mean_field(
                rows, "wm_score_effective_weight"
            ),
            "wm_shadow_changed_selection_mean": _mean_field(
                rows, "wm_shadow_changed_selection"
            ),
            "wm_fallback_reason_top": _top_reason(rows, "wm_fallback_reason"),
        }
    return out


def _profile_copy_table(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "dexpbt_8gpu_ws" / "metrics" / "profile_copy_events.jsonl"
    events = _read_jsonl(path)
    reject_reasons: dict[str, int] = defaultdict(int)
    copy_pairs: dict[str, int] = defaultdict(int)
    for ev in events:
        rr = str(ev.get("reject_reason") or ev.get("decision") or "")
        if rr:
            reject_reasons[rr] += 1
        if ev.get("accepted"):
            src = str(ev.get("source_profile", ""))
            tgt = str(ev.get("target_new_profile", ""))
            if src and tgt:
                copy_pairs[f"{src} -> {tgt}"] += 1
    return {
        "event_count": len(events),
        "attempt_count": sum(1 for e in events if e.get("attempted")),
        "accept_count": sum(1 for e in events if e.get("accepted")),
        "reject_reason_counts": dict(reject_reasons),
        "source_to_target_counts": dict(copy_pairs),
        "events_tail": events[-10:],
    }


def _lineage_table(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "dexpbt_8gpu_ws" / "policy_lineage.json"
    if not path.is_file():
        return {"present": False}
    try:
        data = json.loads(path.read_text())
    except Exception as exc:
        return {"present": True, "error": str(exc)}
    policies = data.get("policies") or {}
    unique = sorted({str(v.get("wm_strategy_profile_name")) for v in policies.values()})
    return {
        "present": True,
        "updated_at": data.get("updated_at"),
        "policies": policies,
        "unique_profiles": unique,
        "profile_diversity_count": len(unique),
        "generations": {
            pid: int(v.get("wm_strategy_profile_generation", 0))
            for pid, v in policies.items()
        },
    }


def _population_section(
    curves: dict[int, list[dict[str, Any]]],
    profiles: dict[int, str],
    frame: int = 2_000_000_000,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    restart_total = 0
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
        restart_total += int(best.get("restart_count") or 0)
        rows.append(
            {
                "policy_idx": pid,
                "profile": profiles.get(pid, "unknown"),
                "nsucc_mean": ns,
                "retention": _safe_float(best.get("source_retention")),
            }
        )
    rows.sort(key=lambda x: x["nsucc_mean"], reverse=True)
    top3 = rows[:3]
    top3_mean = (
        sum(r["nsucc_mean"] for r in top3) / len(top3) if top3 else None
    )
    return {
        "top1": top3[0] if top3 else None,
        "top3": top3,
        "top3_nsucc_mean": top3_mean,
        "restart_count_total": restart_total,
    }


def analyze(run_dir: Path) -> dict[str, Any]:
    cfg = _parse_run_config(run_dir)
    profiles = _policy_profiles(run_dir, cfg)
    curves = _load_curves(run_dir)
    throughput = _throughput_table(curves, profiles)
    return {
        "run_dir": str(run_dir),
        "config_version": cfg.get("config_version", cfg.get("profile", "")),
        "same_frame_nsucc": _same_frame_table(curves, profiles),
        "throughput_vs_p0": throughput,
        "elite_distill_p1_p4_p5": _elite_distill_section(curves, profiles, throughput),
        "source_selector_p2_p3": _selector_section(curves, profiles),
        "profile_copy": _profile_copy_table(run_dir),
        "lineage": _lineage_table(run_dir),
        "population": _population_section(curves, profiles),
    }


def _render_md(report: dict[str, Any]) -> str:
    lines = [
        "# Strategy Zoo v1.3 Elite Debug Report",
        "",
        f"Run: `{report.get('run_dir', '')}`",
        f"Config: `{report.get('config_version', '')}`",
        "",
        "## Same-frame nsucc",
        "",
    ]
    for mark, table in sorted((report.get("same_frame_nsucc") or {}).items()):
        lines.append(f"### @ {int(mark) // 1_000_000_000}B frames")
        for prof, row in sorted(table.items()):
            ns = row.get("nsucc_mean")
            lines.append(
                f"- **{prof}** (p{row.get('policy_idx')}): nsucc={ns:.2f}"
                if ns is not None
                else f"- **{prof}** (p{row.get('policy_idx')}): nsucc=n/a"
            )
        lines.append("")
    lines.extend(["## Throughput vs p0", ""])
    for prof, row in sorted((report.get("throughput_vs_p0") or {}).items()):
        ratio = row.get("throughput_ratio_vs_p0")
        lines.append(
            f"- **{prof}**: ratio={ratio:.3f} nsucc_last={row.get('nsucc_last')}"
            if ratio is not None
            else f"- **{prof}**: nsucc_last={row.get('nsucc_last')}"
        )
    lines.extend(["", "## Profile copy", ""])
    pc = report.get("profile_copy") or {}
    lines.append(
        f"attempts={pc.get('attempt_count')} accepts={pc.get('accept_count')} "
        f"reject_reason_counts={pc.get('reject_reason_counts')}"
    )
    lines.append(f"source_to_target={pc.get('source_to_target_counts')}")
    pop = report.get("population") or {}
    lines.extend(["", "## Population @2B", ""])
    lines.append(f"top3_nsucc_mean={pop.get('top3_nsucc_mean')}")
    for row in pop.get("top3") or []:
        lines.append(
            f"- p{row['policy_idx']} {row['profile']}: nsucc={row['nsucc_mean']:.2f} "
            f"retention={row.get('retention')}"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze Strategy Zoo v1.3 elite-centered debug run"
    )
    parser.add_argument("run_dir", type=Path, help="Run directory under results/runs/")
    parser.add_argument(
        "--out-json",
        type=Path,
        default=None,
        help="Write JSON report (default: results/docs/STRATEGY_ZOO_V13_ELITE_DEBUG_REPORT.json)",
    )
    parser.add_argument(
        "--out-md",
        type=Path,
        default=None,
        help="Write markdown report (default: results/docs/STRATEGY_ZOO_V13_ELITE_DEBUG_REPORT.md)",
    )
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    report = analyze(run_dir)
    repo = Path(__file__).resolve().parents[1]
    out_json = args.out_json or repo / "docs" / "STRATEGY_ZOO_V13_ELITE_DEBUG_REPORT.json"
    out_md = args.out_md or repo / "docs" / "STRATEGY_ZOO_V13_ELITE_DEBUG_REPORT.md"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2))
    out_md.write_text(_render_md(report))
    print(f"Wrote {out_json}")
    print(f"Wrote {out_md}")


if __name__ == "__main__":
    main()
