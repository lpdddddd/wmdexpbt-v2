#!/usr/bin/env python3
"""Phase A4 P4 wired (2026-06-10) — diagnose WM runtime wiring health.

Given a run dir, this script answers:

  1. Was the iterative TeacherManager actually instantiated?
  2. How many teacher eval attempts? Promotions? Rejections (with
     reasons)? Did any policy ever get a non-None promoted teacher?
  3. Was the WMImaginedEvaluator actually instantiated? Valid count vs
     invalid count? Top invalid_reason values?
  4. Did the per-policy ``wm_imagined/<pidx>.json`` mailbox actually
     exist on disk? Most-recent ages?
  5. How often did the PBT observer record ``wm_changed_source_selection``
     (active actual change)? ``wm_shadow_changed_selection`` (would-change)?
     ``wm_score_used`` (per-tick fallback miss-rate)?
  6. Per-policy horizon / recovery pos_rate distribution at the latest
     point.
  7. Did WM score ever feed into the floor_recovery_score? Look at
     ``floor_recovery_score_wm`` deltas vs ``floor_recovery_score_real``.
  8. If active mode NEVER actually changed source selection, surface
     the most common ``wm_fallback_reason`` strings.

Outputs ``wm_runtime_wiring_report.md`` and ``wm_runtime_wiring_report.json``
inside the run dir (or wherever ``--out`` points).
"""

from __future__ import annotations

import argparse
import collections
import json
import statistics
from pathlib import Path
from typing import Any


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    out: list[dict[str, Any]] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return None


def _summarize_policy(jsonl_path: Path) -> dict[str, Any]:
    rows = _read_jsonl(jsonl_path)
    if not rows:
        return {"n": 0}
    last = rows[-1]
    counts = collections.Counter(
        str(r.get("wm_fallback_reason", "feature_off")) for r in rows
    )
    wm_used = sum(1 for r in rows if r.get("wm_score_used"))
    shadow_changed = max(
        (int(r.get("wm_shadow_changed_selection", 0) or 0) for r in rows),
        default=0,
    )
    real_score_last = float(last.get("floor_recovery_score_real", 0.0))
    wm_score_last = float(last.get("floor_recovery_score_wm", 0.0))
    return {
        "n": len(rows),
        "last_frames": int(last.get("env_frames", 0)),
        "last_tolerance": float(last.get("success_tolerance", 1.0)),
        "last_nsucc_mean": float(last.get("nsucc_mean", 0.0) or 0.0),
        "last_floor_stage": str(last.get("floor_stage", "none")),
        "wm_score_used_count": int(wm_used),
        "wm_score_used_fraction": float(wm_used) / max(1, len(rows)),
        "wm_shadow_changed_selection_max": int(shadow_changed),
        "fallback_reasons": dict(counts),
        "floor_recovery_score_real_last": real_score_last,
        "floor_recovery_score_wm_last": wm_score_last,
        "wm_score_delta_last": wm_score_last - real_score_last,
    }


def _summarize_teacher_index(idx_path: Path) -> dict[str, Any]:
    data = _read_json(idx_path)
    if not data:
        return {"present": False}
    audit = data.get("audit", {}) or {}
    cur = data.get("current", {}) or {}
    return {
        "present": True,
        "current_frame": int(cur.get("frame", 0)),
        "current_score": float(cur.get("score", 0.0)),
        "current_recovery_pos_rate": float(
            ((cur.get("metrics") or {}).get("recovery_pos_rate", 0.0))
        ),
        "promotions": int(audit.get("promotions", 0)),
        "rejections": int(audit.get("rejections", 0)),
        "last_promotion_reason": str(audit.get("last_promotion_reason", "")),
        "last_rejection_reason": str(audit.get("last_rejection_reason", "")),
    }


def analyze(run_dir: Path) -> dict[str, Any]:
    workspace = run_dir / "dexpbt_8gpu_ws"
    metrics_dir = workspace / "metrics"
    teachers_dir = workspace / "world_model_teachers"
    wm_imagined_dir = workspace / "wm_imagined"

    per_policy: dict[int, dict[str, Any]] = {}
    for jsonl in sorted(metrics_dir.glob("training_curve_policy_*.jsonl")):
        try:
            idx = int(jsonl.stem.split("_")[-1])
        except ValueError:
            continue
        per_policy[idx] = _summarize_policy(jsonl)

    teachers: dict[int, dict[str, Any]] = {}
    if teachers_dir.exists():
        for sub in sorted(teachers_dir.iterdir()):
            if not sub.is_dir():
                continue
            try:
                idx = int(sub.name.split("_")[-1])
            except ValueError:
                continue
            teachers[idx] = _summarize_teacher_index(
                sub / "teacher_index.json"
            )
    wm_imagined: dict[int, dict[str, Any]] = {}
    if wm_imagined_dir.exists():
        for path in sorted(wm_imagined_dir.glob("*.json")):
            try:
                idx = int(path.stem)
            except ValueError:
                continue
            data = _read_json(path) or {}
            wm_imagined[idx] = {
                "imagined_valid": bool(data.get("imagined_valid", False)),
                "invalid_reason": str(data.get("invalid_reason", "")),
                "frames": int(data.get("frames", 0)),
                "uncertainty": float(data.get("uncertainty", 1.0)),
                "wm_recovery_prob": float(
                    data.get("wm_recovery_prob", 0.0)
                ),
                "wm_success_prob": float(
                    data.get("wm_success_prob", 0.0)
                ),
                "wm_drop_risk": float(data.get("wm_drop_risk", 0.0)),
                "teacher_step": float(data.get("teacher_step", -1.0)),
                "horizon_pos_rate": data.get("horizon_pos_rate", {}) or {},
            }

    teacher_promoted_total = sum(
        int(t.get("promotions", 0)) for t in teachers.values()
    )
    teacher_rejected_total = sum(
        int(t.get("rejections", 0)) for t in teachers.values()
    )
    imagined_valid_total = sum(
        1 for v in wm_imagined.values() if v.get("imagined_valid")
    )
    imagined_total = len(wm_imagined)

    fallback_reasons_global: dict[str, int] = collections.Counter()
    wm_used_total = 0
    shadow_changed_max = 0
    for s in per_policy.values():
        for k, v in (s.get("fallback_reasons") or {}).items():
            fallback_reasons_global[k] += int(v)
        wm_used_total += int(s.get("wm_score_used_count", 0))
        shadow_changed_max = max(
            shadow_changed_max,
            int(s.get("wm_shadow_changed_selection_max", 0)),
        )

    summary = {
        "run_dir": str(run_dir),
        "workspace": str(workspace),
        "wm_imagined_dir_exists": wm_imagined_dir.exists(),
        "teachers_dir_exists": teachers_dir.exists(),
        "per_policy": per_policy,
        "teachers": teachers,
        "wm_imagined": wm_imagined,
        "wm_imagined_valid_count": imagined_valid_total,
        "wm_imagined_total": imagined_total,
        "teacher_promoted_total": teacher_promoted_total,
        "teacher_rejected_total": teacher_rejected_total,
        "wm_score_used_total": wm_used_total,
        "wm_shadow_changed_selection_max": shadow_changed_max,
        "fallback_reason_histogram": dict(fallback_reasons_global),
    }

    # Verdict
    verdict_parts: list[str] = []
    if not wm_imagined_dir.exists():
        verdict_parts.append("WM imagined evaluator: NOT WIRED at runtime.")
    elif imagined_total == 0:
        verdict_parts.append("WM imagined evaluator: dir exists but no per-policy files.")
    else:
        verdict_parts.append(
            f"WM imagined evaluator: {imagined_valid_total}/{imagined_total} "
            "valid entries on disk."
        )
    if not teachers_dir.exists() or not teachers:
        verdict_parts.append("Iterative teacher: NOT WIRED at runtime.")
    else:
        verdict_parts.append(
            f"Iterative teacher: {teacher_promoted_total} promotions, "
            f"{teacher_rejected_total} rejections across "
            f"{len(teachers)} policy directories."
        )
    if wm_used_total == 0:
        verdict_parts.append(
            "Active WM ranking: NEVER engaged "
            f"(top fallback reason: {fallback_reasons_global.most_common(1)})."
        )
    else:
        verdict_parts.append(
            f"Active WM ranking: engaged in {wm_used_total} "
            "training-curve points."
        )
    summary["verdict"] = " | ".join(verdict_parts)
    return summary


def render_markdown(summary: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append(f"# WM runtime wiring report — `{Path(summary['run_dir']).name}`")
    lines.append("")
    lines.append(f"- Workspace: `{summary['workspace']}`")
    lines.append(
        f"- `wm_imagined/` dir present: **{summary['wm_imagined_dir_exists']}**"
    )
    lines.append(
        f"- `world_model_teachers/` dir present: **{summary['teachers_dir_exists']}**"
    )
    lines.append("")
    lines.append("## Verdict")
    lines.append("")
    lines.append(summary["verdict"])
    lines.append("")
    lines.append("## Imagined evaluator (per-policy disk mailbox)")
    lines.append("")
    if summary["wm_imagined"]:
        lines.append(
            "| pidx | imagined_valid | uncertainty | wm_recovery | wm_success | wm_drop | teacher_step | reason |"
        )
        lines.append(
            "|-----:|:--:|---:|---:|---:|---:|---:|---|"
        )
        for pidx in sorted(summary["wm_imagined"].keys()):
            v = summary["wm_imagined"][pidx]
            lines.append(
                f"| {pidx} | {v['imagined_valid']} | {v['uncertainty']:.3f} | "
                f"{v['wm_recovery_prob']:.3f} | {v['wm_success_prob']:.3f} | "
                f"{v['wm_drop_risk']:.3f} | {v['teacher_step']:.0f} | "
                f"{v['invalid_reason']} |"
            )
    else:
        lines.append("_No `wm_imagined/<pidx>.json` files written. The runtime wiring is OFF or never reached the disk mailbox._")
    lines.append("")
    lines.append("## Teachers (per-policy index.json)")
    lines.append("")
    if summary["teachers"]:
        lines.append(
            "| pidx | present | promotions | rejections | last_reason | current_score | rec_pos |"
        )
        lines.append(
            "|-----:|:-:|---:|---:|---|---:|---:|"
        )
        for pidx in sorted(summary["teachers"].keys()):
            t = summary["teachers"][pidx]
            if not t.get("present"):
                lines.append(f"| {pidx} | False | – | – | – | – | – |")
                continue
            lines.append(
                f"| {pidx} | True | {t['promotions']} | {t['rejections']} | "
                f"{t.get('last_promotion_reason') or t.get('last_rejection_reason') or '-'} | "
                f"{t['current_score']:.4f} | {t['current_recovery_pos_rate']:.3f} |"
            )
    else:
        lines.append("_No `world_model_teachers/policy_<pidx>/teacher_index.json` files. TeacherManager NOT wired._")
    lines.append("")
    lines.append("## PBT observer side: WM score usage per policy")
    lines.append("")
    if summary["per_policy"]:
        lines.append(
            "| pidx | last_frames | last_tol | last_nsucc | floor_stage | wm_used / n | shadow_changed_max | score_real | score_wm | Δ |"
        )
        lines.append("|-----:|---:|---:|---:|---|---:|---:|---:|---:|---:|")
        for pidx in sorted(summary["per_policy"].keys()):
            s = summary["per_policy"][pidx]
            n = max(1, int(s.get("n", 0)))
            lines.append(
                f"| {pidx} | {s.get('last_frames', 0)/1e6:.0f}M | "
                f"{s.get('last_tolerance', 0):.4f} | "
                f"{s.get('last_nsucc_mean', 0):.1f} | "
                f"{s.get('last_floor_stage', 'none')} | "
                f"{s.get('wm_score_used_count', 0)}/{n} | "
                f"{s.get('wm_shadow_changed_selection_max', 0)} | "
                f"{s.get('floor_recovery_score_real_last', 0):.2f} | "
                f"{s.get('floor_recovery_score_wm_last', 0):.2f} | "
                f"{s.get('wm_score_delta_last', 0):+.2f} |"
            )
    else:
        lines.append("_No `training_curve_policy_*.jsonl` rows found._")
    lines.append("")
    lines.append("## Fallback reason histogram (per-tick)")
    lines.append("")
    hist = summary.get("fallback_reason_histogram", {}) or {}
    if not hist:
        lines.append("_No reasons recorded — observer never reached the WM ranking branch._")
    else:
        for reason, count in sorted(hist.items(), key=lambda kv: -kv[1]):
            lines.append(f"- `{reason}`: {count}")
    lines.append("")
    lines.append("## Aggregate counters")
    lines.append("")
    lines.append(
        f"- Imagined evaluator valid count: **{summary['wm_imagined_valid_count']}** of **{summary['wm_imagined_total']}** mailbox entries"
    )
    lines.append(
        f"- Teacher promotions: **{summary['teacher_promoted_total']}**, rejections: **{summary['teacher_rejected_total']}**"
    )
    lines.append(
        f"- Per-policy ``wm_score_used`` ticks (sum across policies): **{summary['wm_score_used_total']}**"
    )
    lines.append(
        f"- Max ``wm_shadow_changed_selection`` across policies: **{summary['wm_shadow_changed_selection_max']}**"
    )
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Diagnose RSSM iterative-stack runtime wiring (TeacherManager, "
            "WMImaginedEvaluator, active/shadow PBT ranking, replay)."
        )
    )
    ap.add_argument("--run-dir", required=True, type=Path,
                    help="Path to a regrasping_*_run dir.")
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="Where to write the report (default: run-dir).")
    ap.add_argument("--md-name", type=str,
                    default="wm_runtime_wiring_report.md")
    ap.add_argument("--json-name", type=str,
                    default="wm_runtime_wiring_report.json")
    ns = ap.parse_args()

    run_dir = ns.run_dir.resolve()
    out_dir = (ns.out_dir or run_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = analyze(run_dir)
    md = render_markdown(summary)

    md_path = out_dir / ns.md_name
    json_path = out_dir / ns.json_name
    md_path.write_text(md)
    json_path.write_text(json.dumps(summary, indent=2))
    print(f"wrote {md_path}")
    print(f"wrote {json_path}")
    print(summary["verdict"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
