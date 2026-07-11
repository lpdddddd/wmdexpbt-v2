#!/usr/bin/env python3
"""Generate a 100M-frame smoke report for CB-WM DexPBT runs (Patches A-J).

Collects the nine metric groups requested for smoke acceptance:
  1. source_fallback count
  2. self_restart / cross_restart count
  3. skip replacement count
  4. eligible source count
  5. best_nsucc / mean_nsucc
  6. CF child_train_frames
  7. posterior_update_count
  8. wm_posterior mean / n
  9. nsucc pre-post degradation (先升后降)

Usage:
  python generate_smoke_report_cbwm.py /path/to/regrasping_cbwm_smoke_YYYYMMDD_HHMMSS
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any


FRAMES_PER_EPOCH = 8192 * 16  # num_envs * horizon_length


def _load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    if not path.is_file():
        return rows
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def _glob_json(run_dir: Path, pattern: str) -> list[Path]:
    return sorted(run_dir.rglob(pattern))


def _grep_logs(run_dir: Path, pattern: str) -> list[str]:
    hits: list[str] = []
    rx = re.compile(pattern)
    for log in sorted(run_dir.rglob("*.log")):
        try:
            text = log.read_text(errors="replace")
        except OSError:
            continue
        for line in text.splitlines():
            if rx.search(line):
                hits.append(line)
    return hits


def _load_training_curves(
    ws: Path,
    frame_range: tuple[int, int] | None = None,
) -> list[dict]:
    """Load all per-policy training_curve JSONL rows.

    Track C (2026-06-06): optional `frame_range=(min_frames, max_frames)` —
    when set, only points whose `env_frames` fall within
    [min_frames, max_frames] (inclusive) are returned. Used by the report
    `--compare-to` flow to restrict a long fixed_h1 5B run to its 0–3B
    segment for fair comparison with a 3B Track C run.
    """
    metrics = ws / "metrics"
    rows: list[dict] = []
    if not metrics.is_dir():
        return rows
    for fp in sorted(metrics.glob("training_curve_policy_*.jsonl")):
        rows.extend(_load_jsonl(fp))
    if frame_range is not None:
        lo, hi = int(frame_range[0]), int(frame_range[1])
        rows = [
            r for r in rows
            if lo <= int(r.get("env_frames", -1)) <= hi
        ]
    return rows


def _load_manifests(run_dir: Path) -> list[dict]:
    out: list[dict] = []
    for fp in _glob_json(run_dir, "pbt_restart.json"):
        try:
            out.append(json.loads(fp.read_text()))
        except json.JSONDecodeError:
            continue
    return out


def _load_cf_results(ws: Path) -> list[dict]:
    cf_dir = ws / "counterfactual"
    out: list[dict] = []
    if not cf_dir.is_dir():
        return out
    for fp in sorted(cf_dir.glob("cf_block_result*.json")):
        try:
            out.append(json.loads(fp.read_text()))
        except json.JSONDecodeError:
            continue
    return out


def _epoch_frames_from_logs(run_dir: Path) -> dict[int, int]:
    """Best-effort live frame estimate per policy from epoch: N in logs."""
    frames: dict[int, int] = {}
    for i in range(8):
        policy_dir = run_dir / f"policy_{i:02d}"
        if not policy_dir.is_dir():
            continue
        best_epoch = 0
        for log in policy_dir.rglob("*.log"):
            try:
                text = log.read_text(errors="replace")
            except OSError:
                continue
            for m in re.finditer(r"epoch:\s*(\d+)", text):
                best_epoch = max(best_epoch, int(m.group(1)))
        if best_epoch > 0:
            frames[i] = best_epoch * FRAMES_PER_EPOCH
    return frames


def _count_fallback(manifests: list[dict], curves: list[dict]) -> int:
    n = 0
    for m in manifests:
        reason = str(m.get("skip_reason", "") or m.get("source_tier_label", ""))
        if "fallback" in reason.lower():
            n += 1
    # Also count from tier labels in manifests
    for m in manifests:
        if m.get("source_tier_label") in ("strict_live",) and m.get("skip_reason") in (
            "strict_fallback",
            "legacy_low_nsucc_fallback",
        ):
            n += 1
    # Log grep backup
    return n


def _count_skips(manifests: list[dict], curves: list[dict], run_dir: Path) -> int:
    skip_logs = _grep_logs(
        run_dir,
        r"skip restart|skip self-restart|source_selection_skipped|action=skip|no eligible source",
    )
    # Prefer cumulative counter from last training_curve point per policy
    by_policy: dict[int, dict] = {}
    for pt in curves:
        pid = int(pt.get("policy_idx", -1))
        if pid < 0:
            continue
        ef = int(pt.get("env_frames", 0))
        prev = by_policy.get(pid)
        if prev is None or ef >= int(prev.get("env_frames", 0)):
            by_policy[pid] = pt
    tier_skipped = sum(int(p.get("source_tier_skipped_count", 0)) for p in by_policy.values())
    if tier_skipped > 0:
        return tier_skipped
    return len(skip_logs)


def _restart_counts(manifests: list[dict], curves: list[dict]) -> dict[str, int]:
    """Max cumulative self/cross restart counters across policies."""
    self_max = 0
    cross_max = 0
    recovery_max = 0
    for pt in curves:
        self_max = max(self_max, int(pt.get("self_restart_count", 0)))
        cross_max = max(cross_max, int(pt.get("cross_restart_count", 0)))
        recovery_max = max(recovery_max, int(pt.get("recovery_restart_count", 0)))
    # Manifest tail as fallback
    for m in manifests:
        self_max = max(self_max, int(m.get("self_restart_count", 0)))
        cross_max = max(cross_max, int(m.get("cross_restart_count", 0)))
        recovery_max = max(recovery_max, int(m.get("recovery_restart_count", 0)))
    return {
        "self_restart": self_max,
        "cross_restart": cross_max,
        "recovery_restart": recovery_max,
    }


def _eligible_source_stats(curves: list[dict]) -> dict[str, Any]:
    eligible_ticks = sum(1 for pt in curves if pt.get("eligible_as_source"))
    ineligible_ticks = sum(1 for pt in curves if pt.get("eligible_as_source") is False)
    by_reason: dict[str, int] = defaultdict(int)
    for pt in curves:
        if not pt.get("eligible_as_source"):
            by_reason[str(pt.get("eligible_source_reason", "unknown"))] += 1
    # Population eligible count at last tick bucket
    last_by_policy: dict[int, dict] = {}
    for pt in curves:
        pid = int(pt.get("policy_idx", -1))
        ef = int(pt.get("env_frames", 0))
        prev = last_by_policy.get(pid)
        if prev is None or ef >= int(prev.get("env_frames", 0)):
            last_by_policy[pid] = pt
    live_eligible = sum(1 for p in last_by_policy.values() if p.get("eligible_as_source"))
    return {
        "eligible_ticks_total": eligible_ticks,
        "ineligible_ticks_total": ineligible_ticks,
        "live_eligible_policies_at_end": live_eligible,
        "ineligible_reasons": dict(by_reason),
    }


def _nsucc_population(curves: list[dict]) -> dict[str, float]:
    if not curves:
        return {"best_nsucc_max": float("nan"), "mean_nsucc_mean": float("nan")}
    nsucc_max_vals = [
        float(pt["nsucc_max"]) for pt in curves if pt.get("nsucc_max") is not None
    ]
    nsucc_mean_vals = [
        float(pt["nsucc_mean"]) for pt in curves if pt.get("nsucc_mean") is not None
    ]
    # Peak across all points
    best_max = max(nsucc_max_vals) if nsucc_max_vals else float("nan")
    # Last tick per policy, then population mean
    last_by_policy: dict[int, dict] = {}
    for pt in curves:
        pid = int(pt.get("policy_idx", -1))
        ef = int(pt.get("env_frames", 0))
        prev = last_by_policy.get(pid)
        if prev is None or ef >= int(prev.get("env_frames", 0)):
            last_by_policy[pid] = pt
    live_means = [
        float(p["nsucc_mean"])
        for p in last_by_policy.values()
        if p.get("nsucc_mean") is not None
    ]
    live_maxs = [
        float(p["nsucc_max"])
        for p in last_by_policy.values()
        if p.get("nsucc_max") is not None
    ]
    return {
        "peak_nsucc_max_all_ticks": best_max,
        "live_mean_nsucc_mean": sum(live_means) / len(live_means) if live_means else float("nan"),
        "live_mean_nsucc_max": sum(live_maxs) / len(live_maxs) if live_maxs else float("nan"),
        "live_best_nsucc_max": max(live_maxs) if live_maxs else float("nan"),
    }


def _cf_stats(cf_results: list[dict], run_dir: Path) -> dict[str, Any]:
    child_frames: list[int] = []
    posterior_updates = 0
    for r in cf_results:
        for strat, payload in (r.get("children") or {}).items():
            if isinstance(payload, dict):
                ctf = int(payload.get("child_train_frames", 0) or 0)
                if ctf > 0:
                    child_frames.append(ctf)
        if r.get("posterior_updated"):
            posterior_updates += 1
        if r.get("delta_w"):
            posterior_updates = max(posterior_updates, len(_grep_logs(run_dir, r"posterior update")))
    # Log-based posterior update count
    log_updates = len(_grep_logs(run_dir, r"posterior update|posterior_update_count|Applied CF posterior"))
    posterior_updates = max(posterior_updates, log_updates)
    return {
        "cf_blocks_completed": len(cf_results),
        "child_train_frames_list": child_frames,
        "child_train_frames_min": min(child_frames) if child_frames else 0,
        "child_train_frames_max": max(child_frames) if child_frames else 0,
        "child_train_frames_mean": (
            sum(child_frames) / len(child_frames) if child_frames else 0.0
        ),
        "posterior_update_count": posterior_updates,
    }


def _posterior_state(ws: Path) -> dict[str, Any]:
    path = ws / "wm_posterior_latest.json"
    if not path.is_file():
        return {"found": False}
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError:
        return {"found": False, "error": "invalid_json"}
    strategies = data.get("posterior") or data.get("strategies") or data
    rows: dict[str, dict] = {}
    if isinstance(strategies, dict):
        for k, v in strategies.items():
            if isinstance(v, dict):
                rows[k] = {"mean": v.get("mean"), "n": v.get("n")}
    return {"found": True, "path": str(path), "strategies": rows}


def _detect_pre_post_degradation(curves: list[dict]) -> dict[str, Any]:
    """Flag policies whose nsucc_mean peaks early then drops >30% at end."""
    by_policy: dict[int, list[dict]] = defaultdict(list)
    for pt in curves:
        by_policy[int(pt.get("policy_idx", -1))].append(pt)
    degraded: list[dict] = []
    for pid, pts in sorted(by_policy.items()):
        if len(pts) < 2:
            continue
        pts = sorted(pts, key=lambda x: int(x.get("env_frames", 0)))
        means = [float(p.get("nsucc_mean", 0) or 0) for p in pts]
        peak = max(means)
        final = means[-1]
        peak_idx = means.index(peak)
        # Peak in first 60% of ticks, final at least 30% below peak, peak > 0.5
        if peak > 0.5 and peak_idx < len(means) * 0.6 and final < peak * 0.7:
            degraded.append(
                {
                    "policy_idx": pid,
                    "peak_nsucc_mean": peak,
                    "final_nsucc_mean": final,
                    "peak_frame": int(pts[peak_idx].get("env_frames", 0)),
                    "final_frame": int(pts[-1].get("env_frames", 0)),
                }
            )
    return {
        "degraded_policies": degraded,
        "has_pre_post_degradation": len(degraded) > 0,
    }


def generate_report(
    run_dir: Path,
    frame_range: tuple[int, int] | None = None,
) -> str:
    ws = run_dir / "dexpbt_8gpu_ws"
    curves = _load_training_curves(ws, frame_range=frame_range)
    manifests = _load_manifests(run_dir)
    cf_results = _load_cf_results(ws)
    live_frames = _epoch_frames_from_logs(run_dir)

    fallback_manifest = sum(
        1
        for m in manifests
        if str(m.get("skip_reason", "")).endswith("fallback")
        or m.get("skip_reason") in ("strict_fallback", "legacy_low_nsucc_fallback")
    )
    fallback_logs = len(
        _grep_logs(run_dir, r"strict_fallback|legacy_low_nsucc_fallback|source_fallback")
    )

    restarts = _restart_counts(manifests, curves)
    skips = _count_skips(manifests, curves, run_dir)
    eligible = _eligible_source_stats(curves)
    nsucc = _nsucc_population(curves)
    cf = _cf_stats(cf_results, run_dir)
    posterior = _posterior_state(ws)
    degradation = _detect_pre_post_degradation(curves)

    max_live_frame = max(live_frames.values()) if live_frames else 0
    run_id = run_dir.name
    now = datetime.now().strftime("%Y-%m-%d %H:%M CST")

    lines = [
        f"# CB-WM DexPBT Smoke Report (Patches A-J)",
        "",
        f"**Run ID:** `{run_id}`",
        f"**Generated:** {now}",
        f"**Run dir:** `{run_dir}`",
        f"**Max live frames (log est.):** {max_live_frame / 1e6:.2f}M",
        f"**Training curve points:** {len(curves)}",
        f"**PBT restart manifests:** {len(manifests)}",
        "",
        "## Smoke schedule overrides",
        "",
    ]
    smoke_cfg = run_dir / "SMOKE_CONFIG.txt"
    if smoke_cfg.is_file():
        lines.append("```")
        lines.append(smoke_cfg.read_text().strip())
        lines.append("```")
    else:
        lines.append("(no SMOKE_CONFIG.txt — may be a full run dir)")

    lines += [
        "",
        "## 1. source_fallback count",
        "",
        f"- Manifest fallback restarts: **{fallback_manifest}**",
        f"- Log lines matching fallback: **{fallback_logs}**",
        "",
        "## 2. self_restart / cross_restart count",
        "",
        f"- self_restart (max across policies): **{restarts['self_restart']}**",
        f"- cross_restart (max across policies): **{restarts['cross_restart']}**",
        f"- recovery_restart (archive, Patch G): **{restarts['recovery_restart']}**",
        "",
        "## 3. skip replacement count",
        "",
        f"- skip events (tier_skipped counter or log grep): **{skips}**",
        "",
        "## 4. eligible source count",
        "",
        f"- eligible PBT ticks (sum): **{eligible['eligible_ticks_total']}**",
        f"- ineligible PBT ticks (sum): **{eligible['ineligible_ticks_total']}**",
        f"- live eligible policies at end: **{eligible['live_eligible_policies_at_end']}**",
        "",
        "Ineligible reasons:",
        "",
    ]
    for reason, cnt in sorted(eligible["ineligible_reasons"].items(), key=lambda x: -x[1]):
        lines.append(f"- `{reason}`: {cnt}")

    lines += [
        "",
        "## 5. best_nsucc / mean_nsucc",
        "",
        f"- peak nsucc_max (all ticks): **{nsucc.get('peak_nsucc_max_all_ticks', float('nan')):.3f}**",
        f"- live best nsucc_max: **{nsucc.get('live_best_nsucc_max', float('nan')):.3f}**",
        f"- live mean nsucc_mean: **{nsucc.get('live_mean_nsucc_mean', float('nan')):.3f}**",
        f"- live mean nsucc_max: **{nsucc.get('live_mean_nsucc_max', float('nan')):.3f}**",
        "",
        "### Per-policy live frames (epoch est.)",
        "",
    ]
    for pid in range(8):
        fr = live_frames.get(pid, 0)
        lines.append(f"- policy_{pid:02d}: {fr / 1e6:.2f}M frames")

    lines += [
        "",
        "## 6. CF child_train_frames",
        "",
        f"- CF blocks completed: **{cf['cf_blocks_completed']}**",
        f"- child_train_frames (min/mean/max): **{cf['child_train_frames_min']} / "
        f"{cf['child_train_frames_mean']:.0f} / {cf['child_train_frames_max']}**",
        "",
    ]
    if cf["child_train_frames_list"]:
        lines.append("Per-child values:")
        for v in cf["child_train_frames_list"][:20]:
            lines.append(f"- {v:,}")
        if len(cf["child_train_frames_list"]) > 20:
            lines.append(f"- ... ({len(cf['child_train_frames_list'])} total)")

    lines += [
        "",
        "## 7. posterior_update_count",
        "",
        f"- **{cf['posterior_update_count']}**",
        "",
        "## 8. wm_posterior mean / n",
        "",
    ]
    if posterior.get("found"):
        lines.append(f"Source: `{posterior.get('path', '')}`")
        lines.append("")
        lines.append("| strategy | mean | n |")
        lines.append("|----------|------|---|")
        for strat, st in sorted((posterior.get("strategies") or {}).items()):
            lines.append(f"| {strat} | {st.get('mean', '?'):.4f} | {st.get('n', '?')} |")
    else:
        lines.append("*(wm_posterior_latest.json not found yet)*")

    lines += [
        "",
        "## 9. nsucc 先升后降 (pre-post degradation)",
        "",
        f"**Detected:** {'YES' if degradation['has_pre_post_degradation'] else 'NO'}",
        "",
    ]
    if degradation["degraded_policies"]:
        lines.append("| policy | peak nsucc_mean | final nsucc_mean | peak frame | final frame |")
        lines.append("|--------|-----------------|------------------|------------|-------------|")
        for d in degradation["degraded_policies"]:
            lines.append(
                f"| p{d['policy_idx']:02d} | {d['peak_nsucc_mean']:.3f} | "
                f"{d['final_nsucc_mean']:.3f} | {d['peak_frame']/1e6:.1f}M | "
                f"{d['final_frame']/1e6:.1f}M |"
            )
    else:
        lines.append("No clear pre-post degradation pattern in training_curve (or insufficient PBT ticks).")

    lines += [
        "",
        "## Tier breakdown (Patch G, last tick per policy)",
        "",
    ]
    last_by_policy: dict[int, dict] = {}
    for pt in curves:
        pid = int(pt.get("policy_idx", -1))
        ef = int(pt.get("env_frames", 0))
        prev = last_by_policy.get(pid)
        if prev is None or ef >= int(prev.get("env_frames", 0)):
            last_by_policy[pid] = pt
    tier_totals: dict[str, int] = defaultdict(int)
    for p in last_by_policy.values():
        for k in ("strict_live", "relaxed_live", "archive", "skipped"):
            tier_totals[k] += int(p.get(f"source_tier_{k}_count", 0))
    for k, v in sorted(tier_totals.items()):
        lines.append(f"- `{k}`: {v}")

    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate CB-WM smoke report")
    parser.add_argument("run_dir", type=Path, help="Path to smoke run directory")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output markdown path (default: RUN_DIR/SMOKE_RUN_DATA_REPORT.md)",
    )
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    if not run_dir.is_dir():
        raise SystemExit(f"Run dir not found: {run_dir}")
    report = generate_report(run_dir)
    out = args.output or (run_dir / "SMOKE_RUN_DATA_REPORT.md")
    out.write_text(report)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
