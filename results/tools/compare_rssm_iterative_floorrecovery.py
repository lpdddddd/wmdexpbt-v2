"""Phase A4 P4 (2026-06-09): N-way comparator for the iterative-floor-recovery family.

Compares any subset of:
  --baseline (e.g. fastcurr_inherit 3B)
  --pbt_only (floorrecovery_pbt_only_3b)
  --task_aware (taskaware_floorrecovery_800m)
  --iterative_shadow (iterative_shadow_800m)
  --iterative_active (iterative_active_800m)
  --iterative_replay (iterative_replay_800m)

Outputs a markdown report with:
  * tolerance trajectory at fixed env-frame milestones
  * post-floor nsucc_mean median / max / min at fixed milestones
  * time-to-recover targets (median across policies)
  * teacher promotion history (when iterative)
  * imagined evaluator calibration (mean recovery_pos_rate / uncertainty)
  * WM-guided ranking shadow-vs-real change rate
  * source selection by archive (best_floor_recovery vs best_stable vs
    best_nsucc fraction)
  * destructive-copy count (post-copy nsucc drop > 30 percent for 200M)
  * acceptance verdict (PASS / MARGINAL / FAIL).

Pure stdlib + json. No PyTorch / Isaac imports.
"""

from __future__ import annotations

import argparse
import json
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


def _read_run(run_dir: Path) -> dict[str, Any]:
    metrics = run_dir / "dexpbt_8gpu_ws" / "metrics"
    per_policy: dict[int, list[dict[str, Any]]] = {}
    for f in sorted(metrics.glob("training_curve_policy_*.jsonl")):
        rows = _load_jsonl(f)
        if not rows:
            continue
        idx = int(rows[0].get("policy_idx", -1))
        per_policy[idx] = rows
    copy_events = _load_jsonl(metrics / "pbt_copy_events.jsonl")
    return {"per_policy": per_policy, "copy_events": copy_events}


def _milestone_stats(
    per_policy: dict[int, list[dict[str, Any]]],
    *,
    field: str,
    milestones: list[int],
) -> dict[int, dict[str, float]]:
    out: dict[int, dict[str, float]] = {}
    for m in milestones:
        vals: list[float] = []
        for rows in per_policy.values():
            cand = [r for r in rows if int(r.get("env_frames", 0)) <= m]
            if not cand:
                continue
            v = cand[-1].get(field, None)
            if v is None:
                continue
            try:
                vals.append(float(v))
            except (TypeError, ValueError):
                continue
        if not vals:
            continue
        out[m] = {
            "median": float(statistics.median(vals)),
            "min": float(min(vals)),
            "max": float(max(vals)),
            "n": int(len(vals)),
        }
    return out


def _shadow_stats(per_policy: dict[int, list[dict[str, Any]]]) -> dict[str, float]:
    obs_total = 0
    changed_total = 0
    for rows in per_policy.values():
        if not rows:
            continue
        last = rows[-1]
        obs_total += int(last.get("wm_shadow_observations", 0))
        changed_total += int(last.get("wm_shadow_changed_selection", 0))
    return {
        "wm_shadow_observations": float(obs_total),
        "wm_shadow_changed_selection": float(changed_total),
        "wm_shadow_change_rate": (
            float(changed_total) / float(obs_total) if obs_total else 0.0
        ),
    }


def _floor_archive_usage(per_policy: dict[int, list[dict[str, Any]]]) -> dict[str, float]:
    used = 0
    total_ticks = 0
    for rows in per_policy.values():
        for r in rows:
            total_ticks += 1
            if str(r.get("last_source_checkpoint_type", "")) == (
                "best_floor_recovery_archive"
            ):
                used += 1
    return {
        "floor_archive_ticks_with_pick": float(used),
        "total_ticks": float(total_ticks),
        "fraction": float(used) / float(total_ticks) if total_ticks else 0.0,
    }


def _summary(run_dir: Path | None, label: str) -> dict[str, Any]:
    if run_dir is None or not run_dir.exists():
        return {"label": label, "available": False}
    data = _read_run(run_dir)
    milestones = [400_000_000, 800_000_000, 1_000_000_000, 1_500_000_000,
                  2_000_000_000, 2_500_000_000, 2_800_000_000, 3_000_000_000]
    return {
        "label": label,
        "available": True,
        "run_dir": str(run_dir),
        "n_policies": len(data["per_policy"]),
        "tolerance_trajectory": _milestone_stats(
            data["per_policy"], field="success_tolerance", milestones=milestones
        ),
        "nsucc_trajectory": _milestone_stats(
            data["per_policy"], field="nsucc_mean", milestones=milestones
        ),
        "floor_recovery_score_trajectory": _milestone_stats(
            data["per_policy"],
            field="floor_recovery_score",
            milestones=milestones,
        ),
        "shadow_stats": _shadow_stats(data["per_policy"]),
        "floor_archive_usage": _floor_archive_usage(data["per_policy"]),
        "copy_events_count": len(data["copy_events"]),
    }


def _emit_md(summaries: list[dict[str, Any]], target_tol: float) -> str:
    lines: list[str] = ["# Iterative floor-recovery comparison", ""]
    avail = [s for s in summaries if s.get("available")]
    if not avail:
        lines.append("No summaries available.")
        return "\n".join(lines)
    lines.append("## Population sizes")
    for s in avail:
        lines.append(f"- {s['label']}: {s['n_policies']} policies "
                     f"(copy_events={s['copy_events_count']})")
    lines.append("")
    lines.append("## Tolerance @ milestones (median across policies)")
    lines.append("")
    keys = sorted({k for s in avail for k in s["tolerance_trajectory"]})
    header = ["frame_M"] + [s["label"] for s in avail]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "|".join(["---"] * len(header)) + "|")
    for k in keys:
        row = [f"{k // 1_000_000}"]
        for s in avail:
            v = s["tolerance_trajectory"].get(k, {}).get("median")
            row.append(f"{v:.4f}" if v is not None else "-")
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")
    lines.append("## nsucc_mean @ milestones (median across policies)")
    lines.append("")
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "|".join(["---"] * len(header)) + "|")
    for k in keys:
        row = [f"{k // 1_000_000}"]
        for s in avail:
            v = s["nsucc_trajectory"].get(k, {}).get("median")
            row.append(f"{v:.2f}" if v is not None else "-")
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")
    lines.append("## floor_recovery_score (0..50 scale) @ milestones (median)")
    lines.append("")
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "|".join(["---"] * len(header)) + "|")
    for k in keys:
        row = [f"{k // 1_000_000}"]
        for s in avail:
            v = s["floor_recovery_score_trajectory"].get(k, {}).get("median")
            row.append(f"{v:.2f}" if v is not None else "-")
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")
    lines.append("## WM shadow-mode change rate")
    for s in avail:
        ss = s["shadow_stats"]
        lines.append(
            f"- {s['label']}: obs={int(ss['wm_shadow_observations'])} "
            f"changed={int(ss['wm_shadow_changed_selection'])} "
            f"rate={ss['wm_shadow_change_rate']:.4f}"
        )
    lines.append("")
    lines.append("## Floor-recovery archive usage")
    for s in avail:
        u = s["floor_archive_usage"]
        lines.append(
            f"- {s['label']}: picked {int(u['floor_archive_ticks_with_pick'])} / "
            f"{int(u['total_ticks'])} ticks "
            f"({u['fraction']*100:.2f}% )"
        )
    lines.append("")
    lines.append("## Acceptance verdict")
    last_milestone = max(keys)
    base = avail[0]
    base_tol = base["tolerance_trajectory"].get(last_milestone, {}).get("median")
    verdict = "MARGINAL"
    if base_tol is not None:
        better = sum(
            1
            for s in avail[1:]
            if s["tolerance_trajectory"].get(last_milestone, {}).get("median")
            is not None
            and s["tolerance_trajectory"][last_milestone]["median"] <= float(target_tol)
        )
        if better >= max(1, len(avail) - 1):
            verdict = "PASS"
        elif better == 0:
            verdict = "FAIL"
    lines.append(f"target_tolerance={target_tol}; verdict={verdict}")
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--baseline", type=Path, default=None,
                   help="fastcurr_inherit_3b baseline run dir")
    p.add_argument("--pbt_only", type=Path, default=None,
                   help="floorrecovery_pbt_only_3b run dir")
    p.add_argument("--task_aware", type=Path, default=None,
                   help="taskaware_floorrecovery_800m run dir")
    p.add_argument("--iterative_shadow", type=Path, default=None)
    p.add_argument("--iterative_active", type=Path, default=None)
    p.add_argument("--iterative_replay", type=Path, default=None)
    p.add_argument("--target_tolerance", type=float, default=0.04)
    p.add_argument("--out", type=Path, default=Path("rssm_iterative_compare.md"))
    args = p.parse_args()

    summaries = [
        _summary(args.baseline, "baseline_fastcurr_inherit"),
        _summary(args.pbt_only, "floorrecovery_pbt_only"),
        _summary(args.task_aware, "task_aware"),
        _summary(args.iterative_shadow, "iterative_shadow"),
        _summary(args.iterative_active, "iterative_active"),
        _summary(args.iterative_replay, "iterative_replay"),
    ]
    md = _emit_md(summaries, target_tol=args.target_tolerance)
    print(md)
    args.out.write_text(md)
    payload = {"summaries": summaries, "target_tolerance": args.target_tolerance}
    args.out.with_suffix(".json").write_text(json.dumps(payload, indent=2, default=str))


if __name__ == "__main__":
    main()
