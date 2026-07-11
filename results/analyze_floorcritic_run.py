#!/usr/bin/env python3
"""Analyze a FloorCritic run: engineering health + (Phase 1) does-it-hurt.

Given a run dir, produces ``FLOORCRITIC_RUN_REPORT.md`` and
``FLOORCRITIC_RUN_REPORT.json`` answering (see
FLOORCRITIC_IMPLEMENTATION_PLAN.md, section I):

  1. nsucc_mean peak / end per policy
  2. tolerance trajectory
  3. entry frames for tol <= 0.04 / 0.02 / 0.012 / 0.01
  4. FloorCritic active frames (wm_floorcritic/active == 1)
  5. FloorCritic loss curves (total + per head, summary stats)
  6. target pos_rate per head
  7. actor_grad_blocked: was it ALWAYS true?
  8. critic / aux head grad norm + shared-backbone grad-from-fc
  9. comparison vs a baseline run (--baseline), if given
 10. recommendation: continue to 1.5B / 3B, or fall back

FloorCritic scalars are read from rl_games TB event files under
``policy_XX/rl_games/attempt_*/summaries/``. nsucc / tolerance come from the
PBT observer ``dexpbt_8gpu_ws/metrics/training_curve_policy_XX.jsonl``.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any, Optional

try:
    from tensorboard.backend.event_processing import event_accumulator

    _TB_OK = True
except Exception:  # pragma: no cover
    _TB_OK = False


FC_TAGS = [
    "wm_floorcritic/active",
    "wm_floorcritic/lambda",
    "wm_floorcritic/loss_total",
    "wm_floorcritic/feature_source_id",
    "wm_floorcritic/target_valid_fraction",
    "wm_floorcritic/fc_grad_ratio_to_ppo_grad",
    "wm_floorcritic/auto_scale_factor",
    "wm_floorcritic/shared_backbone_grad_from_fc",
    "wm_floorcritic/actor_head_grad_from_fc",
    "wm_floorcritic/value_head_grad_from_fc",
    "wm_floorcritic/actor_grad_blocked",
    "wm_floorcritic/aux_head_grad_norm",
    "wm_floorcritic/critic_grad_norm",
]
TOL_THRESHOLDS = [0.04, 0.02, 0.012, 0.01]


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    out = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def _collect_tb_scalars(summary_dirs: list[Path]) -> dict[str, list[tuple[float, float]]]:
    """Return {tag: [(step, value), ...]} merged across attempt dirs."""
    merged: dict[str, list[tuple[float, float]]] = {}
    if not _TB_OK:
        return merged
    for d in summary_dirs:
        try:
            ea = event_accumulator.EventAccumulator(
                str(d), size_guidance={"scalars": 0}
            )
            ea.Reload()
            avail = set(ea.Tags().get("scalars", []))
        except Exception:
            continue
        for tag in FC_TAGS:
            if tag not in avail:
                continue
            try:
                evs = ea.Scalars(tag)
            except Exception:
                continue
            merged.setdefault(tag, []).extend(
                (float(e.step), float(e.value)) for e in evs
            )
    for tag in merged:
        merged[tag].sort(key=lambda x: x[0])
    return merged


def _stats(vals: list[float]) -> dict[str, float]:
    if not vals:
        return {"n": 0}
    return {
        "n": len(vals),
        "mean": float(statistics.fmean(vals)),
        "min": float(min(vals)),
        "max": float(max(vals)),
        "last": float(vals[-1]),
    }


def _policy_summary(run_dir: Path, pid: int) -> dict[str, Any]:
    ws = run_dir / "dexpbt_8gpu_ws"
    curve = ws / "metrics" / f"training_curve_policy_{pid:02d}.jsonl"
    rows = _read_jsonl(curve)
    out: dict[str, Any] = {"policy": pid, "curve_points": len(rows)}

    if rows:
        nsucc = [float(r.get("nsucc_mean", 0.0) or 0.0) for r in rows]
        tol = [float(r.get("success_tolerance", 1.0)) for r in rows]
        frames = [int(r.get("env_frames", 0)) for r in rows]
        out["nsucc_peak"] = max(nsucc)
        out["nsucc_end"] = nsucc[-1]
        out["tol_end"] = tol[-1]
        out["tol_min"] = min(tol)
        out["frames_end"] = frames[-1]
        crossings = {}
        for thr in TOL_THRESHOLDS:
            hit = next(
                (frames[i] for i in range(len(tol)) if tol[i] <= thr), None
            )
            crossings[f"tol<={thr}"] = hit
        out["tol_crossings"] = crossings

    # FloorCritic TB scalars.
    summary_dirs = sorted(
        (run_dir / f"policy_{pid:02d}" / "rl_games").glob(
            "attempt_*/summaries"
        )
    )
    tb = _collect_tb_scalars(summary_dirs)
    out["tb_available"] = bool(tb)
    if tb:
        active = [v for _, v in tb.get("wm_floorcritic/active", [])]
        out["fc_active_points"] = int(sum(1 for v in active if v >= 0.5))
        out["fc_total_points"] = len(active)
        agb = [v for _, v in tb.get("wm_floorcritic/actor_grad_blocked", [])]
        out["actor_grad_blocked_always"] = bool(agb) and all(
            v >= 0.5 for v in agb
        )
        out["actor_grad_blocked_violations"] = int(
            sum(1 for v in agb if v < 0.5)
        )
        out["loss_total"] = _stats(
            [v for _, v in tb.get("wm_floorcritic/loss_total", []) if v > 0]
        )
        out["lambda"] = _stats(
            [v for _, v in tb.get("wm_floorcritic/lambda", [])]
        )
        out["fc_grad_ratio"] = _stats(
            [v for _, v in tb.get("wm_floorcritic/fc_grad_ratio_to_ppo_grad", [])]
        )
        out["auto_scale_factor"] = _stats(
            [v for _, v in tb.get("wm_floorcritic/auto_scale_factor", [])]
        )
        out["shared_backbone_grad_from_fc"] = _stats(
            [v for _, v in tb.get("wm_floorcritic/shared_backbone_grad_from_fc", [])]
        )
        out["actor_head_grad_from_fc"] = _stats(
            [v for _, v in tb.get("wm_floorcritic/actor_head_grad_from_fc", [])]
        )
        out["aux_head_grad_norm"] = _stats(
            [v for _, v in tb.get("wm_floorcritic/aux_head_grad_norm", [])]
        )
        out["target_valid_fraction"] = _stats(
            [v for _, v in tb.get("wm_floorcritic/target_valid_fraction", [])]
        )
        # per-head loss + pos_rate tags discovered dynamically.
        head_losses = {}
        head_posrates = {}
        for d in summary_dirs:
            if not _TB_OK:
                break
            try:
                ea = event_accumulator.EventAccumulator(
                    str(d), size_guidance={"scalars": 0}
                )
                ea.Reload()
                for tag in ea.Tags().get("scalars", []):
                    if tag.startswith("wm_floorcritic/loss_") and tag != "wm_floorcritic/loss_total":
                        vals = [float(e.value) for e in ea.Scalars(tag)]
                        head_losses.setdefault(tag, []).extend(vals)
                    elif tag.startswith("wm_floorcritic/pos_rate_"):
                        vals = [float(e.value) for e in ea.Scalars(tag)]
                        head_posrates.setdefault(tag, []).extend(vals)
            except Exception:
                continue
        out["head_loss_last"] = {
            k.split("/")[-1]: (v[-1] if v else 0.0)
            for k, v in head_losses.items()
        }
        out["head_pos_rate_last"] = {
            k.split("/")[-1]: (v[-1] if v else 0.0)
            for k, v in head_posrates.items()
        }
        # degeneracy flag: any head with pos_rate ~0 or ~1 throughout.
        degenerate = {}
        for k, v in head_posrates.items():
            if v:
                mx = max(v)
                mn = min(v)
                degenerate[k.split("/")[-1]] = bool(mx <= 1e-6 or mn >= 1.0 - 1e-6)
        out["head_pos_rate_degenerate"] = degenerate
    return out


def _recommendation(per_policy: list[dict[str, Any]],
                    baseline: Optional[dict[str, Any]]) -> dict[str, Any]:
    rec: dict[str, Any] = {}
    # actor_grad_blocked must hold everywhere FC ran.
    violations = sum(
        int(p.get("actor_grad_blocked_violations", 0) or 0) for p in per_policy
    )
    rec["actor_grad_blocked_ok"] = violations == 0
    # any FC activity at all?
    active = sum(int(p.get("fc_active_points", 0) or 0) for p in per_policy)
    rec["fc_ever_active"] = active > 0
    # nsucc peak across population.
    peaks = [p.get("nsucc_peak", 0.0) for p in per_policy if "nsucc_peak" in p]
    rec["nsucc_peak_max"] = max(peaks) if peaks else 0.0
    rec["nsucc_end_max"] = max(
        (p.get("nsucc_end", 0.0) for p in per_policy if "nsucc_end" in p),
        default=0.0,
    )
    # decision heuristic.
    notes = []
    proceed = True
    if not rec["actor_grad_blocked_ok"]:
        proceed = False
        notes.append("actor_grad_blocked violated -> investigate before scaling.")
    if not rec["fc_ever_active"]:
        notes.append(
            "FloorCritic never activated (tol never <= start_tol within run); "
            "engineering-only result."
        )
    if baseline is not None:
        b_peak = baseline.get("nsucc_peak_max", 0.0)
        if rec["nsucc_peak_max"] + 1e-6 < 0.9 * b_peak:
            proceed = False
            notes.append(
                f"nsucc peak {rec['nsucc_peak_max']:.1f} < 0.9 * baseline "
                f"{b_peak:.1f} -> FloorCritic may be hurting; lower weight or "
                f"fall back to actor_mlp_out_detached."
            )
    rec["recommend_proceed_to_next_scale"] = proceed
    rec["notes"] = notes
    return rec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=str)
    ap.add_argument("--num-policies", type=int, default=6)
    ap.add_argument("--baseline", type=str, default=None,
                    help="Optional baseline run dir for nsucc comparison.")
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    run_dir = Path(args.run_dir).resolve()
    out_dir = Path(args.out).resolve() if args.out else run_dir

    per_policy = [
        _policy_summary(run_dir, pid) for pid in range(args.num_policies)
    ]

    baseline_summary = None
    if args.baseline:
        b_dir = Path(args.baseline).resolve()
        b_pp = [_policy_summary(b_dir, pid) for pid in range(args.num_policies)]
        baseline_summary = {
            "run_dir": str(b_dir),
            "nsucc_peak_max": max(
                (p.get("nsucc_peak", 0.0) for p in b_pp if "nsucc_peak" in p),
                default=0.0,
            ),
            "nsucc_end_max": max(
                (p.get("nsucc_end", 0.0) for p in b_pp if "nsucc_end" in p),
                default=0.0,
            ),
        }

    rec = _recommendation(per_policy, baseline_summary)
    report = {
        "run_dir": str(run_dir),
        "tb_available": _TB_OK,
        "per_policy": per_policy,
        "baseline": baseline_summary,
        "recommendation": rec,
    }

    (out_dir / "FLOORCRITIC_RUN_REPORT.json").write_text(
        json.dumps(report, indent=2)
    )

    # Markdown.
    lines = [
        "# FloorCritic Run Report",
        "",
        f"- run_dir: `{run_dir}`",
        f"- tensorboard_reader_available: {_TB_OK}",
        "",
        "## Recommendation",
        f"- actor_grad_blocked_ok: **{rec['actor_grad_blocked_ok']}**",
        f"- fc_ever_active: **{rec['fc_ever_active']}**",
        f"- nsucc_peak_max: **{rec['nsucc_peak_max']:.2f}**",
        f"- nsucc_end_max: **{rec['nsucc_end_max']:.2f}**",
        f"- recommend_proceed_to_next_scale: **{rec['recommend_proceed_to_next_scale']}**",
    ]
    for n in rec["notes"]:
        lines.append(f"  - note: {n}")
    if baseline_summary:
        lines += [
            "",
            "## Baseline comparison",
            f"- baseline run: `{baseline_summary['run_dir']}`",
            f"- baseline nsucc_peak_max: {baseline_summary['nsucc_peak_max']:.2f}",
            f"- baseline nsucc_end_max: {baseline_summary['nsucc_end_max']:.2f}",
        ]
    lines += ["", "## Per-policy", ""]
    for p in per_policy:
        pid = p["policy"]
        lines.append(f"### policy_{pid:02d}")
        lines.append(f"- curve_points: {p.get('curve_points', 0)}")
        if "nsucc_peak" in p:
            lines.append(
                f"- nsucc peak/end: {p['nsucc_peak']:.2f} / {p['nsucc_end']:.2f}"
            )
            lines.append(
                f"- tol end/min: {p['tol_end']:.4f} / {p['tol_min']:.4f}"
            )
            lines.append(f"- tol_crossings: {p.get('tol_crossings', {})}")
        lines.append(f"- tb_available: {p.get('tb_available', False)}")
        if p.get("tb_available"):
            lines.append(
                f"- fc_active_points: {p.get('fc_active_points', 0)}/"
                f"{p.get('fc_total_points', 0)}"
            )
            lines.append(
                f"- actor_grad_blocked_always: "
                f"**{p.get('actor_grad_blocked_always', False)}** "
                f"(violations={p.get('actor_grad_blocked_violations', 0)})"
            )
            lines.append(f"- loss_total: {p.get('loss_total', {})}")
            lines.append(f"- lambda: {p.get('lambda', {})}")
            lines.append(f"- fc_grad_ratio: {p.get('fc_grad_ratio', {})}")
            lines.append(f"- auto_scale_factor: {p.get('auto_scale_factor', {})}")
            lines.append(
                f"- shared_backbone_grad_from_fc: "
                f"{p.get('shared_backbone_grad_from_fc', {})}"
            )
            lines.append(
                f"- actor_head_grad_from_fc: {p.get('actor_head_grad_from_fc', {})}"
            )
            lines.append(f"- aux_head_grad_norm: {p.get('aux_head_grad_norm', {})}")
            lines.append(
                f"- target_valid_fraction: {p.get('target_valid_fraction', {})}"
            )
            lines.append(f"- head_pos_rate_last: {p.get('head_pos_rate_last', {})}")
            lines.append(
                f"- head_pos_rate_degenerate: "
                f"{p.get('head_pos_rate_degenerate', {})}"
            )
        lines.append("")

    (out_dir / "FLOORCRITIC_RUN_REPORT.md").write_text("\n".join(lines))
    print(f"[analyze_floorcritic_run] wrote report to {out_dir}")
    print(f"  recommend_proceed_to_next_scale="
          f"{rec['recommend_proceed_to_next_scale']} "
          f"actor_grad_blocked_ok={rec['actor_grad_blocked_ok']} "
          f"fc_ever_active={rec['fc_ever_active']}")


if __name__ == "__main__":
    main()
