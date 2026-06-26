#!/usr/bin/env python3
"""Post-hoc CSER-DexPBT mechanism analyzer for formal runs.

Reads training curves, PBT copy events, curriculum safety, EIR broadcast,
curriculum control JSON, workspace policy meta, and optional TensorBoard.

Usage:
  python results/analysis/analyze_cser_mechanisms.py \\
    --run-dir /path/to/regrasping_cser_dexpbt_conservative_5b_* \\
    --out-dir /path/to/run/analysis_cser_mechanisms
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import warnings
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any


PBT_TICK_FRAMES = 20_000_000


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def _sf(v: Any) -> float | None:
    try:
        x = float(v)
        if math.isnan(x) or math.isinf(x):
            return None
        return x
    except (TypeError, ValueError):
        return None


def _frames_b(frames: float | int | None) -> float | None:
    if frames is None:
        return None
    return float(frames) / 1e9


def _mean(vals: list[float]) -> float | None:
    return statistics.mean(vals) if vals else None


def _load_curves(metrics_dir: Path) -> dict[int, list[dict[str, Any]]]:
    out: dict[int, list[dict[str, Any]]] = {}
    for fp in sorted(metrics_dir.glob("training_curve_policy_*.jsonl")):
        stem = fp.stem.replace("training_curve_policy_", "")
        try:
            idx = int(stem)
        except ValueError:
            continue
        out[idx] = _read_jsonl(fp)
    return out


def _nearest_curve_point(
    pts: list[dict[str, Any]], target_frames: int
) -> dict[str, Any] | None:
    if not pts:
        return None
    return min(pts, key=lambda p: abs(int(p.get("env_frames", 0)) - target_frames))


def _worker_nsucc_tol_summary(
    curves: dict[int, list[dict[str, Any]]],
    copy_events: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    cross_by_policy: Counter[int] = Counter()
    for e in copy_events:
        if str(e.get("tier_top_level") or e.get("tier")) == "bottom":
            cross_by_policy[int(e.get("policy_idx", -1))] += 1

    rows: list[dict[str, Any]] = []
    for idx in sorted(curves):
        pts = curves[idx]
        if not pts:
            continue
        final = pts[-1]
        peak_pt = max(pts, key=lambda p: _sf(p.get("nsucc_mean")) or -1.0)
        peak_nsucc = _sf(peak_pt.get("nsucc_mean")) or 0.0
        final_nsucc = _sf(final.get("nsucc_mean")) or 0.0
        retention = (final_nsucc / peak_nsucc) if peak_nsucc > 0 else None
        rows.append(
            {
                "policy_idx": idx,
                "env_frames_final": int(final.get("env_frames", 0)),
                "env_frames_final_b": _frames_b(final.get("env_frames")),
                "nsucc_final": final_nsucc,
                "nsucc_peak": peak_nsucc,
                "nsucc_peak_frame_b": _frames_b(peak_pt.get("env_frames")),
                "success_tolerance_final": _sf(final.get("success_tolerance")),
                "success_tolerance_peak": _sf(peak_pt.get("success_tolerance")),
                "peak_to_final_drop": peak_nsucc - final_nsucc,
                "peak_to_final_retention": retention,
                "cross_restart_count": int(final.get("cross_restart_count", 0)),
                "total_restart_count": int(final.get("restart_count", 0)),
                "bottom_copy_events": cross_by_policy.get(idx, 0),
            }
        )
    return rows


def _copy_events_summary(copy_events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for e in copy_events:
        rows.append(
            {
                "env_frames": int(e.get("env_frames", 0)),
                "env_frames_b": _frames_b(e.get("env_frames")),
                "policy_idx": int(e.get("policy_idx", -1)),
                "pbt_iteration": int(e.get("pbt_iteration", 0)),
                "tier": e.get("tier"),
                "tier_top_level": e.get("tier_top_level"),
                "source_policy_idx": e.get("source_policy_idx"),
                "source_checkpoint_type": e.get("source_checkpoint_type"),
                "source_success_tolerance": _sf(e.get("source_success_tolerance")),
                "source_nsucc_mean": _sf(e.get("source_nsucc_mean")),
                "source_nsucc_max": _sf(e.get("source_nsucc_max")),
                "source_retention": _sf(e.get("source_retention")),
                "source_safe_for_curriculum": e.get("source_safe_for_curriculum"),
                "target_tolerance_before_copy": _sf(e.get("target_tolerance_before_copy")),
                "target_tolerance_after_copy": _sf(e.get("target_tolerance_after_copy")),
                "inherited_tolerance": _sf(e.get("inherited_tolerance")),
                "inheritance_reason": e.get("inheritance_reason"),
                "strong_live_inherit_exact": e.get("strong_live_inherit_exact"),
                "recovery_consolidation_frames_selected": e.get(
                    "recovery_consolidation_frames_selected"
                ),
            }
        )
    return rows


def _p00_rescue_window(
    curves: dict[int, list[dict[str, Any]]],
    copy_events: list[dict[str, Any]],
    *,
    policy_idx: int = 0,
    frame_min_b: float = 3.5,
    frame_max_b: float = 4.0,
    tick_window: int = 5,
) -> list[dict[str, Any]]:
    pts = curves.get(policy_idx, [])
    frame_min = int(frame_min_b * 1e9)
    frame_max = int(frame_max_b * 1e9)
    rescue_events = [
        e
        for e in copy_events
        if int(e.get("policy_idx", -1)) == policy_idx
        and frame_min <= int(e.get("env_frames", 0)) <= frame_max
        and str(e.get("tier_top_level") or "") == "bottom"
    ]
    if not rescue_events:
        rescue_events = [
            e
            for e in copy_events
            if int(e.get("policy_idx", -1)) == policy_idx
            and str(e.get("tier_top_level") or "") == "bottom"
        ]
        if rescue_events:
            rescue_events = [max(rescue_events, key=lambda e: int(e.get("env_frames", 0)))]

    rows: list[dict[str, Any]] = []
    for e in rescue_events:
        event_frame = int(e.get("env_frames", 0))
        tick_frames = [
            event_frame + (i - tick_window) * PBT_TICK_FRAMES
            for i in range(2 * tick_window + 1)
        ]
        tick_context: dict[str, Any] = {}
        for i, tf in enumerate(tick_frames):
            pt = _nearest_curve_point(pts, tf)
            label = f"tick_{i - tick_window:+d}"
            if pt:
                tick_context[f"{label}_frame_b"] = _frames_b(pt.get("env_frames"))
                tick_context[f"{label}_nsucc_mean"] = _sf(pt.get("nsucc_mean"))
                tick_context[f"{label}_success_tolerance"] = _sf(
                    pt.get("success_tolerance")
                )
                tick_context[f"{label}_pbt_iteration"] = int(pt.get("pbt_iteration", 0))
            else:
                tick_context[f"{label}_frame_b"] = _frames_b(tf)
                tick_context[f"{label}_nsucc_mean"] = None
                tick_context[f"{label}_success_tolerance"] = None
                tick_context[f"{label}_pbt_iteration"] = None

        rows.append(
            {
                "policy_idx": policy_idx,
                "event_frame": event_frame,
                "event_frame_b": _frames_b(event_frame),
                "source_policy_idx": e.get("source_policy_idx"),
                "source_checkpoint_type": e.get("source_checkpoint_type"),
                "source_success_tolerance": _sf(e.get("source_success_tolerance")),
                "source_nsucc_mean": _sf(e.get("source_nsucc_mean")),
                "source_nsucc_max": _sf(e.get("source_nsucc_max")),
                "source_retention": _sf(e.get("source_retention")),
                "target_tolerance_before_copy": _sf(e.get("target_tolerance_before_copy")),
                "target_tolerance_after_copy": _sf(e.get("target_tolerance_after_copy")),
                "inherited_tolerance": _sf(e.get("inherited_tolerance")),
                "inheritance_reason": e.get("inheritance_reason"),
                "source_safe_for_curriculum": e.get("source_safe_for_curriculum"),
                "recovery_consolidation_frames_selected": e.get(
                    "recovery_consolidation_frames_selected"
                ),
                "strong_live_inherit_exact": e.get("strong_live_inherit_exact"),
                **tick_context,
            }
        )
    return rows


def _curriculum_safety_reason_counts(
    safety_events: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    counts: Counter[str] = Counter()
    for e in safety_events:
        reason = (
            e.get("block_reason")
            or e.get("reason")
            or e.get("event")
            or "unknown"
        )
        counts[str(reason)] += 1
    return [{"block_reason": k, "count": v} for k, v in counts.most_common()]


def _broadcast_summary(broadcast_events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    pair_counts: Counter[tuple[Any, Any]] = Counter()
    event_counts: Counter[str] = Counter()
    entry_totals: Counter[str] = Counter()
    for e in broadcast_events:
        ev = str(e.get("event") or "unknown")
        event_counts[ev] += 1
        src = e.get("source_policy_idx", e.get("source_idx"))
        tgt = e.get("target_policy_idx", e.get("policy_idx", e.get("consumer_policy_idx")))
        pair_counts[(src, tgt)] += 1
        entries = e.get("entries") or e.get("num_entries") or e.get("bank_entries")
        if entries is not None:
            entry_totals[ev] += int(entries)

    rows: list[dict[str, Any]] = [
        {"metric": "broadcast_events", "value": event_counts.get("broadcast", 0)},
        {"metric": "consume_events", "value": event_counts.get("consume", 0)},
        {"metric": "total_events", "value": sum(event_counts.values())},
    ]
    for (src, tgt), cnt in pair_counts.most_common(20):
        rows.append(
            {
                "metric": "source_target_pair",
                "source_policy_idx": src,
                "target_policy_idx": tgt,
                "count": cnt,
            }
        )
    for ev, total in entry_totals.items():
        rows.append({"metric": f"{ev}_entries_total", "value": total})
    return rows


def _find_tensorboard_dirs(run_dir: Path) -> list[Path]:
    dirs: set[Path] = set()
    ws = run_dir / "dexpbt_8gpu_ws"
    if ws.is_dir():
        for idx in range(32):
            pol = ws / f"{idx:03d}"
            if not pol.is_dir():
                continue
            for child in pol.glob("events.out.tfevents.*"):
                dirs.add(child.parent)
            for child in pol.glob("**/events.out.tfevents.*"):
                dirs.add(child.parent)
                break
        metrics = ws / "metrics"
        if metrics.is_dir():
            for child in metrics.glob("events.out.tfevents.*"):
                dirs.add(child.parent)
    return sorted(dirs)


def _read_tb_scalar_summary(
    run_dir: Path,
    tag_specs: list[tuple[str, str]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], bool, list[str]]:
    """Return (wm_rows, snippet_rows, missing_tb, warnings)."""
    warnings_out: list[str] = []
    tb_dirs = _find_tensorboard_dirs(run_dir)
    if not tb_dirs:
        warnings_out.append("No TensorBoard event files found under run directory.")
        empty = [
            {"tag": tag, "missing_tensorboard_metrics": True, "last_value": None}
            for _, tag in tag_specs
        ]
        return empty[:4], empty[4:], True, warnings_out

    try:
        from tensorboard.backend.event_processing.event_accumulator import (  # type: ignore
            EventAccumulator,
        )
    except ImportError:
        warnings_out.append("tensorboard package unavailable; skipping TB metrics.")
        empty = [
            {"tag": tag, "missing_tensorboard_metrics": True, "last_value": None}
            for _, tag in tag_specs
        ]
        return empty[:4], empty[4:], True, warnings_out

    scalar_values: dict[str, float | None] = {tag: None for _, tag in tag_specs}
    for tb_dir in tb_dirs:
        try:
            acc = EventAccumulator(str(tb_dir))
            acc.Reload()
            available = set(acc.Tags().get("scalars", []))
            for _, tag in tag_specs:
                if tag not in available:
                    continue
                events = acc.Scalars(tag)
                if events:
                    scalar_values[tag] = float(events[-1].value)
        except Exception as exc:
            warnings_out.append(f"TB read failed for {tb_dir}: {exc}")

    wm_tags = [t for k, t in tag_specs if k == "wm"]
    snip_tags = [t for k, t in tag_specs if k == "snippet"]
    any_found = any(scalar_values[t] is not None for t in scalar_values)
    missing = not any_found

    def _rows(tags: list[str]) -> list[dict[str, Any]]:
        return [
            {
                "tag": tag,
                "last_value": scalar_values[tag],
                "missing_tensorboard_metrics": scalar_values[tag] is None,
            }
            for tag in tags
        ]

    return _rows(wm_tags), _rows(snip_tags), missing, warnings_out


def _load_policy_meta(run_dir: Path) -> dict[int, dict[str, Any]]:
    ws = run_dir / "dexpbt_8gpu_ws"
    out: dict[int, dict[str, Any]] = {}
    if not ws.is_dir():
        return out
    for idx in range(6):
        pol_dir = ws / f"{idx:03d}"
        if not pol_dir.is_dir():
            continue
        json_files = sorted(pol_dir.glob("*.json"), key=lambda p: p.stat().st_mtime)
        if not json_files:
            continue
        meta = _read_json(json_files[-1])
        if meta:
            out[idx] = meta
    return out


def _load_curriculum_controls(run_dir: Path) -> dict[int, dict[str, Any]]:
    ctrl_dir = run_dir / "dexpbt_8gpu_ws" / "curriculum_control"
    out: dict[int, dict[str, Any]] = {}
    if not ctrl_dir.is_dir():
        return out
    for fp in sorted(ctrl_dir.glob("*.json")):
        try:
            idx = int(fp.stem)
        except ValueError:
            continue
        data = _read_json(fp)
        if data:
            out[idx] = data
    return out


def _broadcast_near_rescue(
    broadcast_events: list[dict[str, Any]],
    rescue_frame: int | None,
    *,
    window_frames: int = 100_000_000,
) -> dict[str, Any]:
    if rescue_frame is None:
        return {"aligned_broadcast": 0, "aligned_consume": 0}
    aligned_b = aligned_c = 0
    for e in broadcast_events:
        ef = int(e.get("env_frames", e.get("frame", 0)) or 0)
        if abs(ef - rescue_frame) > window_frames:
            continue
        ev = str(e.get("event") or "")
        if ev == "broadcast":
            aligned_b += 1
        elif ev == "consume":
            aligned_c += 1
    return {
        "rescue_frame": rescue_frame,
        "window_frames": window_frames,
        "aligned_broadcast": aligned_b,
        "aligned_consume": aligned_c,
    }


def _render_summary_md(
    *,
    run_dir: Path,
    out_dir: Path,
    run_config: dict[str, str],
    worker_rows: list[dict[str, Any]],
    p00_rescue: list[dict[str, Any]],
    safety_counts: list[dict[str, Any]],
    wm_summary: list[dict[str, Any]],
    snippet_summary: list[dict[str, Any]],
    missing_tb: bool,
    tb_warnings: list[str],
    broadcast_align: dict[str, Any],
    policy_meta: dict[int, dict[str, Any]],
) -> str:
    pop_final = _mean([r["nsucc_final"] for r in worker_rows])
    pop_peak = _mean([r["nsucc_peak"] for r in worker_rows])

    p00_event = p00_rescue[0] if p00_rescue else {}
    p00_meta = policy_meta.get(0, {})
    p00_curriculum = (p00_meta.get("curriculum") or {}) if p00_meta else {}

    inherit_reason = p00_event.get("inheritance_reason", "n/a")
    source_safe = p00_event.get("source_safe_for_curriculum")
    tol_after_copy = p00_event.get("target_tolerance_after_copy")
    tol_before_copy = p00_event.get("target_tolerance_before_copy")
    meta_tol = _sf(p00_curriculum.get("success_tolerance"))
    p00_final = next((r for r in worker_rows if r["policy_idx"] == 0), {})

    wm_nonzero = [
        r for r in wm_summary if r.get("last_value") not in (None, 0.0)
    ]
    snip_nonzero = [
        r for r in snippet_summary if r.get("last_value") not in (None, 0.0)
    ]

    late_strict = [
        r
        for r in worker_rows
        if r["policy_idx"] in (1, 3)
        and (r.get("success_tolerance_final") or 1.0) <= 0.025
        and (r.get("peak_to_final_drop") or 0) >= 8.0
    ]

    lines = [
        "# CSER Mechanism Analysis Summary",
        "",
        f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"**Run dir:** `{run_dir}`",
        f"**Output dir:** `{out_dir}`",
        "",
        "## Population",
        "",
        f"- Population final nsucc mean: **{pop_final:.2f}**" if pop_final else "- Population final nsucc mean: n/a",
        f"- Population peak nsucc mean: **{pop_peak:.2f}**" if pop_peak else "- Population peak nsucc mean: n/a",
        "",
        "## Q1: Why does p00 stay at tol=0.075 after ~3.72B copy rescue?",
        "",
    ]

    if p00_event:
        lines += [
            f"- Rescue copy @ **{p00_event.get('event_frame_b', '?'):.2f}B** from p{p00_event.get('source_policy_idx')} "
            f"({p00_event.get('source_checkpoint_type')}).",
            f"- `inheritance_reason`: **{inherit_reason}**",
            f"- `source_safe_for_curriculum`: **{source_safe}**",
            f"- Copy event `target_tolerance_before_copy`: **{tol_before_copy}**, "
            f"`target_tolerance_after_copy`: **{tol_after_copy}**",
            f"- Final workspace meta `curriculum.success_tolerance`: **{meta_tol}**",
            f"- Final training-curve tol: **{p00_final.get('success_tolerance_final')}**",
        ]
        if inherit_reason in ("source_unstable_legacy", "missing_source_tolerance"):
            lines.append(
                "- Root cause: inheritance fell back to legacy `last_mastered`/initial "
                f"({tol_before_copy}) because source metadata failed source_safe gates."
            )
        elif meta_tol and meta_tol >= 0.07 and tol_after_copy and tol_after_copy < 0.04:
            lines.append(
                "- Root cause: copy event computed a stricter inherited tolerance, but "
                "the env/meta snapshot still reports the loose initial tolerance (0.075). "
                "Strong-live exact inherit + env resume sync is the intended patch."
            )
        elif tol_after_copy and tol_after_copy >= 0.07:
            lines.append(
                "- Root cause: inherited tolerance remained at/near initial 0.075 "
                "(legacy last_mastered or relax-one-level from a loose target)."
            )
        else:
            lines.append(
                "- Inherited tolerance was computed below 0.04; check env-side apply "
                "vs curve logging for a resume mismatch."
            )
    else:
        lines.append("- No p00 bottom copy event found in 3.5–4.0B window.")

    lines += [
        "",
        "## Q2: WM calibration / snippet rerank visible?",
        "",
    ]
    if missing_tb:
        lines.append("- **TensorBoard metrics missing** (`missing_tensorboard_metrics=true`).")
        for w in tb_warnings:
            lines.append(f"- Warning: {w}")
    else:
        lines.append(
            f"- WM calibration tags with non-zero last values: **{len(wm_nonzero)}** / {len(wm_summary)}"
        )
        lines.append(
            f"- Snippet / consolidation tags with non-zero last values: **{len(snip_nonzero)}** / {len(snippet_summary)}"
        )

    lines += [
        "",
        "## Q3: Late-stage p01/p03 retention gap vs strict tolerance?",
        "",
    ]
    if late_strict:
        for r in late_strict:
            lines.append(
                f"- p{r['policy_idx']:02d}: peak {r['nsucc_peak']:.1f} → final {r['nsucc_final']:.1f} "
                f"(drop {r['peak_to_final_drop']:.1f}), final tol {r['success_tolerance_final']:.4f}"
            )
        lines.append(
            "- Pattern consistent with late-floor strict tolerance (≤0.025) plus "
            "continued shrink/mutation pressure; late-floor retention gate targets this."
        )
    else:
        lines.append("- No p01/p03 late-drop alert under strict-tol threshold in this run.")

    lines += [
        "",
        "## Curriculum safety block_reason distribution",
        "",
        "| reason | count |",
        "|--------|-------|",
    ]
    for row in safety_counts[:12]:
        lines.append(f"| {row['block_reason']} | {row['count']} |")

    lines += [
        "",
        "## Broadcast alignment around p00 rescue",
        "",
        f"- Aligned broadcast within ±100M frames: **{broadcast_align.get('aligned_broadcast', 0)}**",
        f"- Aligned consume within ±100M frames: **{broadcast_align.get('aligned_consume', 0)}**",
        "",
        "## Output files",
        "",
        "- `worker_nsucc_tol_summary.csv`",
        "- `copy_events_summary.csv`",
        "- `p00_rescue_window.csv`",
        "- `curriculum_safety_reason_counts.csv`",
        "- `broadcast_summary.csv`",
        "- `wm_calibration_summary.csv`",
        "- `snippet_summary.csv`",
        "",
    ]
    if run_config:
        lines += ["## Run config", ""]
        for k in ("method", "config_version", "commit", "target_frames"):
            if k in run_config:
                lines.append(f"- `{k}`: {run_config[k]}")
        lines.append("")
    return "\n".join(lines)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("missing_tensorboard_metrics=true\n" if "summary" in path.name else "")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _read_run_config(run_dir: Path) -> dict[str, str]:
    cfg: dict[str, str] = {}
    path = run_dir / "RUN_CONFIG.txt"
    if not path.is_file():
        return cfg
    for line in path.read_text(encoding="utf-8").splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            cfg[k.strip()] = v.strip()
    return cfg


def analyze_run(run_dir: Path, out_dir: Path) -> dict[str, Any]:
    run_dir = run_dir.resolve()
    metrics_dir = run_dir / "dexpbt_8gpu_ws" / "metrics"
    out_dir.mkdir(parents=True, exist_ok=True)

    missing_inputs: list[str] = []
    if not metrics_dir.is_dir():
        missing_inputs.append(str(metrics_dir))

    curves = _load_curves(metrics_dir) if metrics_dir.is_dir() else {}
    if metrics_dir.is_dir() and not curves:
        missing_inputs.append("training_curve_policy_*.jsonl")

    copy_events = _read_jsonl(metrics_dir / "pbt_copy_events.jsonl") if metrics_dir.is_dir() else []
    safety_events = _read_jsonl(metrics_dir / "curriculum_safety_events.jsonl") if metrics_dir.is_dir() else []
    broadcast_events = _read_jsonl(metrics_dir / "eir_broadcast_events.jsonl") if metrics_dir.is_dir() else []

    run_config = _read_run_config(run_dir)
    worker_rows = _worker_nsucc_tol_summary(curves, copy_events)
    copy_summary = _copy_events_summary(copy_events)
    p00_rescue = _p00_rescue_window(curves, copy_events, policy_idx=0)
    safety_counts = _curriculum_safety_reason_counts(safety_events)
    broadcast_rows = _broadcast_summary(broadcast_events)
    policy_meta = _load_policy_meta(run_dir)
    curriculum_controls = _load_curriculum_controls(run_dir)

    tb_tag_specs = [
        ("wm", "wm_cal/alpha_wm"),
        ("wm", "wm_cal/auc_recovery"),
        ("wm", "wm_cal/ece_recovery"),
        ("wm", "wm_cal/rank_corr"),
        ("snippet", "snippet_distill/loss"),
        ("snippet", "snippet_distill/active_frac"),
        ("snippet", "snippet_wm_rerank/changed_frac"),
        ("snippet", "snippet_wm_rerank/wm_term_mean"),
        ("snippet", "pbt_consolidation/stage"),
        ("snippet", "pbt_consolidation/restart_blocked_count"),
    ]
    wm_summary, snippet_summary, missing_tb, tb_warnings = _read_tb_scalar_summary(
        run_dir, tb_tag_specs
    )

    rescue_frame = int(p00_rescue[0]["event_frame"]) if p00_rescue else None
    broadcast_align = _broadcast_near_rescue(broadcast_events, rescue_frame)

    _write_csv(out_dir / "worker_nsucc_tol_summary.csv", worker_rows)
    _write_csv(out_dir / "copy_events_summary.csv", copy_summary)
    _write_csv(out_dir / "p00_rescue_window.csv", p00_rescue)
    _write_csv(out_dir / "curriculum_safety_reason_counts.csv", safety_counts)
    _write_csv(out_dir / "broadcast_summary.csv", broadcast_rows)
    _write_csv(out_dir / "wm_calibration_summary.csv", wm_summary)
    _write_csv(out_dir / "snippet_summary.csv", snippet_summary)

    summary_md = _render_summary_md(
        run_dir=run_dir,
        out_dir=out_dir,
        run_config=run_config,
        worker_rows=worker_rows,
        p00_rescue=p00_rescue,
        safety_counts=safety_counts,
        wm_summary=wm_summary,
        snippet_summary=snippet_summary,
        missing_tb=missing_tb,
        tb_warnings=tb_warnings,
        broadcast_align=broadcast_align,
        policy_meta=policy_meta,
    )
    (out_dir / "summary.md").write_text(summary_md, encoding="utf-8")

    summary = {
        "run_dir": str(run_dir),
        "generated_at": datetime.now().isoformat(),
        "missing_inputs": missing_inputs,
        "missing_tensorboard_metrics": missing_tb,
        "tb_warnings": tb_warnings,
        "run_config": run_config,
        "worker_summary": worker_rows,
        "p00_rescue": p00_rescue,
        "broadcast_alignment": broadcast_align,
        "curriculum_control_policies": sorted(curriculum_controls.keys()),
        "policy_meta_policies": sorted(policy_meta.keys()),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    if missing_inputs and not curves:
        raise FileNotFoundError(
            "Required inputs missing: " + ", ".join(missing_inputs)
        )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="CSER-DexPBT post-hoc mechanism analyzer")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    summary = analyze_run(args.run_dir, args.out_dir)
    print(f"Wrote analysis to {args.out_dir.resolve()}")
    if summary.get("missing_tensorboard_metrics"):
        print("missing_tensorboard_metrics=true")
    p00 = summary.get("p00_rescue") or []
    if p00:
        print(
            f"p00 rescue inheritance_reason={p00[0].get('inheritance_reason')} "
            f"source_safe={p00[0].get('source_safe_for_curriculum')}"
        )


if __name__ == "__main__":
    main()
