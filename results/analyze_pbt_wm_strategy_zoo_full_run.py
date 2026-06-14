#!/usr/bin/env python3
"""PBT-WM Strategy Zoo v1.0 run analyzer.

Aggregates per-profile metrics from training_curve JSONL + RUN_CONFIG and
answers the nine strategy-zoo evaluation questions.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from collections import defaultdict
from pathlib import Path
from typing import Any


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with open(path) as fh:
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


def _stats(vals: list[float]) -> dict[str, Any]:
    if not vals:
        return {"n": 0, "last": None, "peak": None, "mean": None}
    return {
        "n": len(vals),
        "last": vals[-1],
        "peak": max(vals),
        "mean": sum(vals) / len(vals),
    }


def _parse_run_config(run_dir: Path) -> dict[str, str]:
    cfg: dict[str, str] = {}
    p = run_dir / "RUN_CONFIG.txt"
    if not p.exists():
        return cfg
    for line in p.read_text().splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            cfg[k.strip()] = v.strip()
    return cfg


def _detect_ws(run_dir: Path, ws_name: str | None) -> Path:
    if ws_name:
        return run_dir / ws_name
    cand = run_dir / "dexpbt_8gpu_ws"
    if (cand / "metrics").is_dir():
        return cand
    for child in sorted(run_dir.iterdir()) if run_dir.is_dir() else []:
        if (child / "metrics").is_dir():
            return child
    return cand


def analyze_run(run_dir: Path, *, ws_name: str | None = None, num_policies: int = 6) -> dict[str, Any]:
    run_dir = run_dir.resolve()
    ws = _detect_ws(run_dir, ws_name)
    metrics = ws / "metrics"
    run_cfg = _parse_run_config(run_dir)

    policy_profiles: dict[int, str] = {}
    for i in range(num_policies):
        key = f"policy_{i:02d}_profile"
        if key in run_cfg:
            policy_profiles[i] = run_cfg[key]
        else:
            policy_profiles[i] = "unknown"

    by_profile: dict[str, dict[str, Any]] = defaultdict(dict)
    all_curves: dict[int, list[dict]] = {}

    for pid in range(num_policies):
        prof = policy_profiles.get(pid, "unknown")
        rows = _read_jsonl(metrics / f"training_curve_policy_{pid:02d}.jsonl")
        all_curves[pid] = rows
        if not rows:
            by_profile[prof] = {"policy_idx": pid, "n_points": 0}
            continue

        nsucc = [_safe_float(r.get("nsucc_mean")) for r in rows]
        nsucc = [x for x in nsucc if x is not None]
        rmeta = [_safe_float(r.get("r_meta_raw", r.get("r_meta"))) for r in rows]
        rmeta = [x for x in rmeta if x is not None]
        tol = [_safe_float(r.get("success_tolerance")) for r in rows]
        tol = [x for x in tol if x is not None]

        peak_idx = nsucc.index(max(nsucc)) if nsucc else 0
        post_peak = nsucc[peak_idx:] if nsucc else []
        retention = (
            post_peak[-1] / post_peak[0]
            if len(post_peak) > 1 and post_peak[0] > 1e-9
            else None
        )

        shrink_blocks = sum(
            1 for r in rows if r.get("wm_curriculum_blocked_shrink")
        )
        profile_copies = sum(
            1
            for r in rows
            if str(r.get("pbt_wm_profile_copy_decision", "")) == "copied"
        )
        wm_score_used = sum(1 for r in rows if r.get("wm_score_used"))
        wm_fallback = sum(
            1
            for r in rows
            if str(r.get("wm_fallback_reason", "ok")) not in ("ok", "feature_off", "")
        )

        actor_distill_frac = [
            _safe_float(r.get("profile_actor_distill_enabled")) for r in rows
        ]
        mppi_frac = [_safe_float(r.get("profile_rssm_mppi_enabled")) for r in rows]

        by_profile[prof] = {
            "policy_idx": pid,
            "n_points": len(rows),
            "nsucc_mean": _stats(nsucc),
            "r_meta": _stats(rmeta),
            "success_tolerance": _stats(tol),
            "nsucc_retention_after_peak": retention,
            "restart_count_last": int(rows[-1].get("restart_count", 0)),
            "shrink_count": sum(
                1 for r in rows if r.get("curriculum_shrink_reason")
            ),
            "collapse_freeze_count": sum(
                1 for r in rows if r.get("collapse_freeze_active")
            ),
            "wm_curriculum_blocked_shrink_count": shrink_blocks,
            "pbt_profile_copy_count": profile_copies,
            "wm_score_used_ticks": wm_score_used,
            "wm_fallback_ticks": wm_fallback,
            "profile_actor_distill_enabled_frac": _stats(
                [x for x in actor_distill_frac if x is not None]
            ),
            "profile_rssm_mppi_enabled_frac": _stats(
                [x for x in mppi_frac if x is not None]
            ),
            "residual_apply_enabled_last": bool(
                rows[-1].get("profile_residual_apply_enabled", False)
            ),
        }

    profiles_sorted_stability = sorted(
        by_profile.items(),
        key=lambda kv: (
            -(kv[1].get("nsucc_retention_after_peak") or 0),
            -(kv[1].get("nsucc_mean", {}).get("last") or 0),
        ),
    )
    profiles_sorted_peak = sorted(
        by_profile.items(),
        key=lambda kv: -(kv[1].get("nsucc_mean", {}).get("peak") or 0),
    )

    diversity_rows = [
        r.get("pbt_wm_profile_diversity_count")
        for curves in all_curves.values()
        for r in curves
        if r.get("pbt_wm_profile_diversity_count") is not None
    ]
    diversity_min = min(diversity_rows) if diversity_rows else None

    questions = {
        "most_stable_profile": profiles_sorted_stability[0][0]
        if profiles_sorted_stability
        else None,
        "highest_peak_profile": profiles_sorted_peak[0][0]
        if profiles_sorted_peak
        else None,
        "wm_curriculum_reduces_collapse": (
            by_profile.get("wm_curriculum_controller", {}).get(
                "wm_curriculum_blocked_shrink_count", 0
            )
            > 0
        ),
        "wm_pbt_selector_changes_ranking": (
            by_profile.get("wm_pbt_source_selector", {}).get("wm_score_used_ticks", 0)
            > 0
        ),
        "rssm_mppi_actor_distill_has_teacher_path": bool(
            by_profile.get("rssm_mppi_actor_distill", {}).get(
                "profile_rssm_mppi_enabled_frac", {}
            ).get("last")
        ),
        "elite_bank_vs_planner_stability": {
            "elite_retention": by_profile.get("elite_bank_actor_distill", {}).get(
                "nsucc_retention_after_peak"
            ),
            "planner_retention": by_profile.get(
                "conservative_rssm_planner_distill", {}
            ).get("nsucc_retention_after_peak"),
        },
        "conservative_planner_no_rollout_apply": not any(
            v.get("residual_apply_enabled_last")
            for v in by_profile.values()
        ),
        "profile_diversity_collapse": (
            diversity_min is not None and diversity_min < 3
        ),
    }

    return {
        "run_dir": str(run_dir),
        "workspace": str(ws),
        "run_config": run_cfg,
        "policy_profiles": policy_profiles,
        "by_profile": dict(by_profile),
        "questions": questions,
    }


def write_reports(report: dict[str, Any], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "PBT_WM_STRATEGY_ZOO_FULL_REPORT.json"
    md_path = out_dir / "PBT_WM_STRATEGY_ZOO_FULL_REPORT.md"
    json_path.write_text(json.dumps(report, indent=2))

    lines = [
        "# PBT-WM Strategy Zoo Full Report",
        "",
        f"Run: `{report.get('run_dir')}`",
        "",
        "## Per-profile summary",
        "",
    ]
    for prof, data in sorted(report.get("by_profile", {}).items()):
        ns = data.get("nsucc_mean", {})
        lines.append(f"### `{prof}` (policy {data.get('policy_idx')})")
        lines.append(
            f"- latest nsucc_mean: {ns.get('last')} | peak: {ns.get('peak')}"
        )
        lines.append(
            f"- retention after peak: {data.get('nsucc_retention_after_peak')}"
        )
        lines.append(
            f"- wm curriculum blocked shrink: "
            f"{data.get('wm_curriculum_blocked_shrink_count', 0)}"
        )
        lines.append(
            f"- profile copies: {data.get('pbt_profile_copy_count', 0)}"
        )
        lines.append(
            f"- residual apply (last): {data.get('residual_apply_enabled_last')}"
        )
        lines.append("")

    q = report.get("questions", {})
    lines.extend(
        [
            "## Questions",
            "",
            f"1. Most stable profile: **{q.get('most_stable_profile')}**",
            f"2. Highest peak profile: **{q.get('highest_peak_profile')}**",
            f"3. WM curriculum controller active blocks: **{q.get('wm_curriculum_reduces_collapse')}**",
            f"4. WM PBT selector used WM score: **{q.get('wm_pbt_selector_changes_ranking')}**",
            f"5. RSSM-MPPI actor distill path enabled: **{q.get('rssm_mppi_actor_distill_has_teacher_path')}**",
            f"6. Elite vs planner retention: **{q.get('elite_bank_vs_planner_stability')}**",
            f"7. No rollout residual apply: **{q.get('conservative_planner_no_rollout_apply')}**",
            f"8. Profile diversity collapse (<3): **{q.get('profile_diversity_collapse')}**",
            "",
        ]
    )
    md_path.write_text("\n".join(lines))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--ws", type=str, default=None)
    ap.add_argument("--num-policies", type=int, default=6)
    args = ap.parse_args()
    report = analyze_run(
        args.run_dir, ws_name=args.ws, num_policies=args.num_policies
    )
    out = args.out or args.run_dir
    write_reports(report, out)
    print(f"Wrote {out / 'PBT_WM_STRATEGY_ZOO_FULL_REPORT.json'}")


if __name__ == "__main__":
    main()
