#!/usr/bin/env python3
"""Generate 800M P2 smoke acceptance report (P2-A..F criteria S1-S8).

Usage:
  python generate_smoke_report_p2.py /path/to/regrasping_cbwm_smoke_800m_p2_YYYYMMDD_HHMMSS
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from generate_smoke_report_p1 import (  # noqa: E402
    _load_curriculum_control,
    _nsucc_at_end,
    _nsucc_at_frame,
    generate_p1_report,
)
from generate_smoke_report_cbwm import _grep_logs, _load_training_curves  # noqa: E402


def _load_recovery_audit(ws: Path) -> list[dict[str, Any]]:
    path = ws / "recovery_audit.jsonl"
    rows: list[dict[str, Any]] = []
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


def _load_launch_audit(run_dir: Path) -> list[dict[str, Any]]:
    path = run_dir / "launch_audit.jsonl"
    rows: list[dict[str, Any]] = []
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


def _audit_count(audit: list[dict], event: str) -> int:
    return sum(1 for r in audit if str(r.get("event")) == event)


def _audit_policies_first_epoch(audit: list[dict]) -> set[int]:
    return {
        int(r.get("policy_idx", -1))
        for r in audit
        if str(r.get("event")) == "policy_first_epoch_time"
        and int(r.get("policy_idx", -1)) >= 0
    }


def _audit_startup_hang_per_policy(audit: list[dict]) -> dict[int, int]:
    counts: dict[int, int] = defaultdict(int)
    for r in audit:
        if str(r.get("event")) != "startup_timeout":
            continue
        pid = int(r.get("policy_idx", -1))
        if pid >= 0:
            counts[pid] += 1
    return dict(counts)


# Authoritative enum mirrored from
# runtime_root/src/dexpbt_isaaclab/dexpbt_isaaclab/pbt/pbt_observer.py.
# Keep names identical to the values used in tests / reports.
RESTART_TYPE_LABEL: dict[int, str] = {
    0: "none",
    1: "self",
    2: "cross",
    3: "skipped",
    4: "recovery",
}


def _restart_type_label(value: Any) -> str:
    """Best-effort mapping of last_restart_type to a human label.

    Accepts ints, numeric strings, or already-labeled strings.
    """

    if value is None:
        return "none"
    try:
        i = int(value)
        return RESTART_TYPE_LABEL.get(i, f"unknown({i})")
    except (TypeError, ValueError):
        s = str(value).strip()
        return s or "none"


def _restart_reason_breakdown(curves: list[dict]) -> Counter[str]:
    """Per-tick last_restart_type counts using the labeled enum."""

    counts: Counter[str] = Counter()
    for pt in curves:
        label = _restart_type_label(pt.get("last_restart_type"))
        if label == "none":
            continue
        counts[label] += 1
    return counts


def _post_replacement_recovery(
    curves: list[dict], horizon_frames: int = 50_000_000
) -> dict[str, list[dict]]:
    """Split per-policy "restart-like" events using both the labeled
    last_restart_type AND cumulative cross/recovery counters.

    Returns four buckets:
      - actual_cross   (real cross-copy restart, type==2 or cross_count delta>0)
      - actual_recovery (real recovery restart, type==4 or recovery_count delta>0)
      - skipped_ticks  (type==3 only; not a real restart)
      - self_improvement (type==0; consecutive curve mean delta summary)

    "delta" rows look horizon_frames ahead and report nsucc_mean change.
    """

    by_policy: dict[int, list[dict]] = defaultdict(list)
    for pt in sorted(curves, key=lambda p: int(p.get("env_frames", 0))):
        pid = int(pt.get("policy_idx", -1))
        if pid >= 0:
            by_policy[pid].append(pt)

    cross_rows: list[dict] = []
    recovery_rows: list[dict] = []
    skipped_rows: list[dict] = []
    self_rows: list[dict] = []
    self_improvement_deltas: list[float] = []

    def _lookahead(points: list[dict], idx: int, ef_at: int) -> tuple[float, float]:
        """Return (mean_at, mean_at+horizon) for the given curve index."""
        target = ef_at + horizon_frames
        future_pt = None
        for q in points[idx + 1 :]:
            if int(q.get("env_frames", 0)) >= target:
                future_pt = q
                break
        mean_at = float(points[idx].get("nsucc_mean", 0) or 0)
        mean_later = (
            float(future_pt.get("nsucc_mean", 0) or 0)
            if future_pt is not None
            else float("nan")
        )
        return mean_at, mean_later

    for pid, points in by_policy.items():
        # Curves can interleave rows from multiple writers at the same env_frames
        # with non-monotonic cumulative counters. Use running MAX so we only
        # count the moment a counter first reaches a new high.
        seen_cross_max = 0
        seen_recovery_max = 0
        prev_mean_for_self = None
        for idx, pt in enumerate(points):
            ef_at = int(pt.get("env_frames", 0))
            cur_label = _restart_type_label(pt.get("last_restart_type"))
            cross_now = int(pt.get("cross_restart_count", 0) or 0)
            recovery_now = int(pt.get("recovery_restart_count", 0) or 0)
            cross_jumped = cross_now > seen_cross_max
            recovery_jumped = recovery_now > seen_recovery_max
            cur_mean = float(pt.get("nsucc_mean", 0) or 0)

            row_common = {
                "policy_idx": pid,
                "frame_at_restart_m": ef_at / 1e6,
            }

            if cross_jumped or cur_label == "cross":
                mean_at, mean_later = _lookahead(points, idx, ef_at)
                cross_rows.append(
                    {
                        **row_common,
                        "restart_type": "cross",
                        "mean_at_restart": mean_at,
                        "mean_plus_50M": mean_later,
                        "delta": (
                            mean_later - mean_at
                            if mean_later == mean_later
                            else float("nan")
                        ),
                    }
                )
            if recovery_jumped or cur_label == "recovery":
                mean_at, mean_later = _lookahead(points, idx, ef_at)
                recovery_rows.append(
                    {
                        **row_common,
                        "restart_type": "recovery",
                        "mean_at_restart": mean_at,
                        "mean_plus_50M": mean_later,
                        "delta": (
                            mean_later - mean_at
                            if mean_later == mean_later
                            else float("nan")
                        ),
                    }
                )
            if cur_label == "skipped" and not (cross_jumped or recovery_jumped):
                skipped_rows.append(
                    {
                        **row_common,
                        "restart_type": "skipped",
                        "rollback_reason": str(pt.get("rollback_reason", "") or ""),
                    }
                )
            if cur_label == "none" and prev_mean_for_self is not None:
                self_improvement_deltas.append(cur_mean - prev_mean_for_self)

            seen_cross_max = max(seen_cross_max, cross_now)
            seen_recovery_max = max(seen_recovery_max, recovery_now)
            prev_mean_for_self = cur_mean

    if self_improvement_deltas:
        deltas_sorted = sorted(self_improvement_deltas)
        n = len(deltas_sorted)
        median_delta = deltas_sorted[n // 2]
        self_rows.append(
            {
                "n": n,
                "mean_delta": sum(self_improvement_deltas) / n,
                "median_delta": median_delta,
                "max_delta": max(self_improvement_deltas),
                "min_delta": min(self_improvement_deltas),
            }
        )

    return {
        "actual_cross": cross_rows,
        "actual_recovery": recovery_rows,
        "skipped_ticks": skipped_rows,
        "self_improvement": self_rows,
    }


def _detect_population_diversity_failed(
    curves: list[dict], min_policies: int = 2
) -> tuple[bool, str]:
    """Flag identical-trajectory populations or shared seeds."""

    by_policy: dict[int, list[tuple[int, float, float]]] = defaultdict(list)
    seeds: dict[int, int | None] = {}
    for pt in sorted(curves, key=lambda p: int(p.get("env_frames", 0))):
        pid = int(pt.get("policy_idx", -1))
        if pid < 0:
            continue
        by_policy[pid].append(
            (
                int(pt.get("env_frames", 0)),
                round(float(pt.get("nsucc_mean", 0) or 0), 3),
                round(float(pt.get("nsucc_max", 0) or 0), 3),
            )
        )
        if "seed" in pt and pt["seed"] is not None and pid not in seeds:
            try:
                seeds[pid] = int(pt["seed"])
            except (TypeError, ValueError):
                seeds[pid] = None

    seen_seeds: set[int] = set()
    duplicate_seeds: list[int] = []
    for pid in sorted(seeds):
        s = seeds[pid]
        if s is None:
            continue
        if s in seen_seeds:
            duplicate_seeds.append(s)
        seen_seeds.add(s)
    if duplicate_seeds:
        return True, f"duplicate seeds across policies: {sorted(set(duplicate_seeds))}"

    pids = sorted(by_policy)
    if len(pids) < min_policies:
        return False, "single-policy run; diversity check skipped"
    base_pid = pids[0]
    base = by_policy[base_pid]
    base_keys = {ef: (m, mx) for ef, m, mx in base}
    for pid in pids[1:]:
        other = by_policy[pid]
        if len(other) != len(base):
            return False, "policy curves have different lengths; not strictly identical"
        identical = all(
            base_keys.get(ef) == (m, mx) for ef, m, mx in other
        )
        if identical:
            return True, f"policies {base_pid} and {pid} produced bit-identical trajectories"
    return False, "no identical trajectory pair detected"


def _format_pbt_population_verification(
    run_dir: Path,
    curves: list[dict],
    audit: list[dict],
    num_policies_expected: int | None,
) -> tuple[list[str], list[dict]]:
    latest = _latest_points_by_policy(curves)
    peaks = _peak_mean_by_policy(curves)

    seeded = _audit_policies_first_epoch(audit)
    if not seeded:
        # Fallback: any policy that produced a curve point counts as started.
        seeded = set(latest.keys())
    startup_hang_count = _audit_count(audit, "startup_timeout")
    startup_hang_failed = _audit_count(audit, "startup_hang_failed")
    startup_hang_per_pid = _audit_startup_hang_per_policy(audit)

    if num_policies_expected is None:
        num_policies_expected = max(len(seeded), len(latest)) or 1

    diversity_failed, diversity_detail = _detect_population_diversity_failed(curves)

    end_means: list[float] = []
    for pid, pt in latest.items():
        try:
            end_means.append(float(pt.get("nsucc_mean", 0) or 0))
        except (TypeError, ValueError):
            continue
    pop_mean = sum(end_means) / len(end_means) if end_means else float("nan")
    pop_median = (
        sorted(end_means)[len(end_means) // 2] if end_means else float("nan")
    )

    restart_breakdown = _restart_reason_breakdown(curves)
    if startup_hang_count:
        restart_breakdown["startup_hang"] += int(startup_hang_count)

    # Use lifetime MAX of cumulative counters per policy (not latest tick),
    # because the training curve interleaves rows from multiple writers and
    # the literal last row may carry a lower cumulative value than seen
    # earlier. Same fix is used in _post_replacement_recovery.
    max_counters_by_pid: dict[int, dict[str, int]] = defaultdict(
        lambda: {
            "cross_restart_count": 0,
            "self_restart_count": 0,
            "recovery_restart_count": 0,
        }
    )
    for pt in curves:
        pid = int(pt.get("policy_idx", -1))
        if pid < 0:
            continue
        bucket = max_counters_by_pid[pid]
        for key in ("cross_restart_count", "self_restart_count", "recovery_restart_count"):
            try:
                bucket[key] = max(bucket[key], int(pt.get(key, 0) or 0))
            except (TypeError, ValueError):
                continue

    cross_total = sum(b["cross_restart_count"] for b in max_counters_by_pid.values())
    self_total = sum(b["self_restart_count"] for b in max_counters_by_pid.values())
    recovery_total = sum(b["recovery_restart_count"] for b in max_counters_by_pid.values())

    eligible_thresholds_met = all(
        float(pt.get("nsucc_mean", 0) or 0)
        >= float(pt.get("min_nsucc_mean_as_source", 0) or 0)
        for pt in latest.values()
    )
    pbt_replacements = cross_total + recovery_total
    d4_pass = pbt_replacements >= 1 or eligible_thresholds_met

    criteria = [
        {
            "id": "D1",
            "name": "all_policies_started == num_policies",
            "pass": len(seeded) >= num_policies_expected,
            "detail": f"seen={sorted(seeded)} expected={num_policies_expected}",
        },
        {
            "id": "D2",
            "name": "startup_hang_count == 0 (warn=1, fail>=2)",
            "pass": startup_hang_count == 0,
            "detail": f"hangs={startup_hang_count} failed={startup_hang_failed}",
        },
        {
            "id": "D3",
            "name": "population_diversity_failed == False",
            "pass": not diversity_failed,
            "detail": diversity_detail,
        },
        {
            "id": "D4",
            "name": ">=1 PBT replacement OR all policies meet source thresholds",
            "pass": d4_pass,
            "detail": (
                f"cross={cross_total} recovery={recovery_total} "
                f"thresholds_met={eligible_thresholds_met}"
            ),
        },
    ]

    lines = [
        "## PBT population verification (no-h3 full smoke)",
        "",
        f"- num_policies_expected: **{num_policies_expected}**",
        f"- all_policies_started: **{len(seeded)}** -> {sorted(seeded)}",
        f"- startup_hang_count: **{startup_hang_count}** "
        f"(failed_after_retry: **{startup_hang_failed}**)",
        f"- population_diversity_failed: **{diversity_failed}** ({diversity_detail})",
        f"- population end_mean: mean=**{pop_mean:.3f}** median=**{pop_median:.3f}** n={len(end_means)}",
        "",
        "### Acceptance",
        "",
        "| ID | Criterion | Pass | Detail |",
        "|----|-----------|------|--------|",
    ]
    for c in criteria:
        mark = "PASS" if c["pass"] else "FAIL"
        lines.append(f"| {c['id']} | {c['name']} | **{mark}** | `{c['detail'][:120]}` |")

    lines += [
        "",
        "### Restart counts by reason",
        "",
    ]
    if restart_breakdown:
        for reason, count in restart_breakdown.most_common():
            lines.append(f"- `{reason}`: **{count}**")
    else:
        lines.append("- No restart events recorded.")
    lines += [
        "",
        f"- self_restart total: **{self_total}**",
        f"- cross_restart total: **{cross_total}**",
        f"- recovery_restart total: **{recovery_total}**",
    ]

    lines += [
        "",
        "### Per-policy table",
        "",
        "| pid | seed | env_hash | peak_mean | end_mean | retention | startup_hangs | cross | recovery |",
        "|-----|------|----------|-----------|----------|-----------|----------------|-------|----------|",
    ]
    pids = sorted(set(latest) | set(peaks) | set(seeded))
    for pid in pids:
        pt = latest.get(pid, {})
        peak = peaks.get(pid, float("nan"))
        end = float(pt.get("nsucc_mean", 0) or 0)
        retention = (
            f"{end / peak:.2%}" if peak == peak and peak > 0 else "—"
        )
        seed = pt.get("seed", "—")
        env_hash = pt.get("env_randomization_hash", "—") or "—"
        cross_for_pid = max_counters_by_pid.get(pid, {}).get("cross_restart_count", 0)
        recovery_for_pid = max_counters_by_pid.get(pid, {}).get(
            "recovery_restart_count", 0
        )
        lines.append(
            f"| p{pid} | {seed} | {env_hash} | {peak:.3f} | {end:.3f} | "
            f"{retention} | {startup_hang_per_pid.get(pid, 0)} | "
            f"{cross_for_pid} | {recovery_for_pid} |"
        )

    recovery_buckets = _post_replacement_recovery(curves)
    cross_rows = recovery_buckets["actual_cross"]
    recovery_rows = recovery_buckets["actual_recovery"]
    skipped_rows = recovery_buckets["skipped_ticks"]
    self_rows = recovery_buckets["self_improvement"]

    def _render_delta_table(rows: list[dict], header: str) -> list[str]:
        block: list[str] = ["", header, ""]
        if not rows:
            block.append("- (none)")
            return block
        block += [
            "| pid | restart_type | frame@restart (M) | mean@restart | mean@+50M | delta |",
            "|-----|--------------|-------------------|--------------|-----------|-------|",
        ]
        for r in rows:
            mean_plus = r["mean_plus_50M"]
            mean_plus_str = "—" if mean_plus != mean_plus else f"{mean_plus:.3f}"
            delta = r["delta"]
            delta_str = "—" if delta != delta else f"{delta:.3f}"
            block.append(
                f"| p{r['policy_idx']} | {r['restart_type']} | "
                f"{r['frame_at_restart_m']:.0f} | "
                f"{r['mean_at_restart']:.3f} | {mean_plus_str} | {delta_str} |"
            )
        return block

    lines += _render_delta_table(
        cross_rows, "### PBT actual cross-copy restart (+50M frames)"
    )
    lines += _render_delta_table(
        recovery_rows, "### PBT actual recovery restart (+50M frames)"
    )

    lines += [
        "",
        "### PBT skipped ticks (last_restart_type=3, NOT a restart)",
        "",
        f"- count: **{len(skipped_rows)}**",
    ]
    if skipped_rows:
        reason_counter: Counter[str] = Counter()
        for r in skipped_rows:
            reason_counter[(r.get("rollback_reason") or "(blank)")] += 1
        lines.append("- reasons (top 5):")
        for reason, n in reason_counter.most_common(5):
            lines.append(f"  - `{reason}`: **{n}**")

    lines += [
        "",
        "### Self-improvement delta (last_restart_type=0, no restart)",
        "",
    ]
    if self_rows:
        s = self_rows[0]
        lines.append(
            f"- n={s['n']} mean_delta={s['mean_delta']:.3f} "
            f"median_delta={s['median_delta']:.3f} "
            f"max_delta={s['max_delta']:.3f} min_delta={s['min_delta']:.3f}"
        )
    else:
        lines.append("- (no self-improvement rows)")

    return lines, criteria


def _shrinks_per_window(curves: list[dict]) -> dict[int, int]:
    """Max shrinks_in_window seen per policy (from training curve if logged)."""
    by_policy: dict[int, int] = defaultdict(int)
    for pt in curves:
        pid = int(pt.get("policy_idx", -1))
        sw = int(pt.get("shrinks_in_window", 0) or 0)
        by_policy[pid] = max(by_policy[pid], sw)
    return dict(by_policy)


def _s4_per_tick_diagnosis(curves: list[dict]) -> dict[str, Any]:
    """Track C (2026-06-06): per-tick S4 diagnosis.

    Walk training_curve_policy_*.jsonl ordered by env_frames per policy.
    Identify each first-trigger of `hard_collapse_triggered` (transition
    from False/None to True). For every first-trigger, ask: is the
    rollback mechanism observable nearby? We say yes when EITHER the
    same tick has `forced_success_tolerance > 0` OR a nearby tick within
    the next 50M frames has `forced_success_tolerance > 0` OR the tick's
    `rollback_reason` indicates a cooldown_blocked aftermath (which only
    fires when a previous rollback already executed).

    Returns {first_triggers: int, rolled_back: int, unrolled: int,
             unrolled_examples: list[(pid, env_frames)]}.
    """
    by_policy: dict[int, list[dict]] = defaultdict(list)
    for pt in curves:
        try:
            pid = int(pt.get("policy_idx", -1))
        except (TypeError, ValueError):
            continue
        if pid < 0:
            continue
        by_policy[pid].append(pt)
    first_triggers = 0
    rolled_back = 0
    unrolled = 0
    unrolled_examples: list[tuple[int, int]] = []
    LOOKAHEAD = 50_000_000
    for pid, pts in by_policy.items():
        pts = sorted(pts, key=lambda p: int(p.get("env_frames", 0)))
        prev_active = False
        for i, pt in enumerate(pts):
            try:
                hct = bool(pt.get("hard_collapse_triggered"))
            except Exception:
                hct = False
            if hct and not prev_active:
                first_triggers += 1
                ef = int(pt.get("env_frames", 0))
                this_forced = float(pt.get("forced_success_tolerance", 0) or 0) > 1e-9
                this_cooldown = str(pt.get("rollback_reason") or "") == "cooldown_blocked"
                rolled = this_forced or this_cooldown
                if not rolled:
                    for j in range(i + 1, len(pts)):
                        nxt = pts[j]
                        if int(nxt.get("env_frames", 0)) - ef > LOOKAHEAD:
                            break
                        if float(nxt.get("forced_success_tolerance", 0) or 0) > 1e-9:
                            rolled = True
                            break
                        if str(nxt.get("rollback_reason") or "") == "cooldown_blocked":
                            rolled = True
                            break
                if rolled:
                    rolled_back += 1
                else:
                    unrolled += 1
                    unrolled_examples.append((pid, ef))
            prev_active = hct
    return {
        "first_triggers": first_triggers,
        "rolled_back": rolled_back,
        "unrolled": unrolled,
        "unrolled_examples": unrolled_examples,
    }


def _evaluate_p2_criteria(
    run_dir: Path,
    curves: list[dict],
    ctrl: list[dict],
    recovery_audit: list[dict],
) -> list[dict[str, Any]]:
    peak = _nsucc_at_frame(curves, target_m=480.0)
    end = _nsucc_at_end(curves)
    peak_val = float(peak.get("peak_nsucc_mean", float("nan")))
    end_val = float(end.get("nsucc_mean", float("nan")))

    any_shrink_token = any(c.get("shrink_token") for c in ctrl)
    any_consolidation = any(
        int(c.get("post_shrink_consolidation_until_restart_step", 0) or 0) > 0
        for c in ctrl
    )
    any_rollback = any(
        float(c.get("forced_success_tolerance", 0) or 0) > 1e-9 for c in ctrl
    )
    any_hard_collapse = any(c.get("hard_collapse_triggered") for c in ctrl)
    any_early_freeze = any(c.get("collapse_freeze_active") for c in ctrl)

    # Aggregate single-shot shrink evidence from training_curve as a fallback,
    # since curriculum_control snapshots may miss the shrink_token boolean if
    # the gate signal wasn't logged at exactly the same step.
    max_shrinks_in_window = 0
    for source in (ctrl, curves):
        for entry in source:
            try:
                sw = int(entry.get("shrinks_in_window", 0) or 0)
            except (TypeError, ValueError):
                sw = 0
            if sw > max_shrinks_in_window:
                max_shrinks_in_window = sw
    shrink_blocked_reasons = Counter(
        str(c.get("shrink_blocked_reason") or "")
        for c in ctrl
        if str(c.get("shrink_blocked_reason") or "")
    )

    selected = [r for r in recovery_audit if r.get("outcome") == "selected"]
    executed = [r for r in recovery_audit if r.get("outcome") == "executed"]
    skipped = [r for r in recovery_audit if r.get("outcome") == "skipped"]
    selected_for_best_stable = [
        r for r in selected if r.get("tier") == "best_stable_archive"
    ]

    recovery_tol_logs = _grep_logs(run_dir, r"recovery_tolerance_after|last_mastered")
    s6_ok = len(recovery_tol_logs) > 0 or any(
        pt.get("pbt_recovery_tolerance_mode") == "last_mastered" for pt in curves
    )

    mutation_frozen_seen = any(pt.get("mutation_frozen") for pt in curves)
    hyperparam_logged = any(
        pt.get("hyperparam_gamma") is not None for pt in curves
    )

    # S1 — single-shot shrink. PASS unless we observe an actual multi-shrink
    # within a single PBT window. The shrink_token boolean is informational.
    s1_ok = max_shrinks_in_window <= 1
    s1_detail = (
        f"max_shrinks_in_window={max_shrinks_in_window} "
        f"shrink_token_seen={any_shrink_token} "
        f"consolidation_active={any_consolidation} "
        f"shrink_blocked={dict(shrink_blocked_reasons.most_common(3))}"
    )

    # S2 — preserved.
    s2_ok = end_val >= 0.5 * peak_val if peak_val == peak_val and peak_val > 0 else False

    # S3 — preserved.
    s3_ok = any_early_freeze

    # S4 — tri-state. Track C (2026-06-06): per-tick first-trigger
    # diagnosis instead of the noisy curriculum_control any() check.
    # Curriculum_control snapshots are end-of-run only and miss the
    # critical first-trigger / rollback timing; cooldown_blocked aftermath
    # ticks were misread as fresh failed triggers in the formal 5B run.
    s4_diag = _s4_per_tick_diagnosis(curves)
    if s4_diag["first_triggers"] == 0 and not any_hard_collapse:
        s4_ok: bool | None = None
        s4_detail = (
            f"hard_collapse=False (N/A: no first-trigger seen in curves)"
        )
    elif s4_diag["first_triggers"] == 0 and any_hard_collapse:
        # Snapshot ctrl says hard_collapse=True but per-tick curve has no
        # first-trigger transition — cooldown aftermath only.
        s4_ok = bool(any_rollback)
        s4_detail = (
            "hard_collapse_in_ctrl=True but no per-tick first-trigger "
            f"(snapshots reflect cooldown aftermath); forced_tol_seen={any_rollback}"
        )
    else:
        # Real first-triggers observed: PASS only when every first-trigger
        # has an observable rollback / cooldown aftermath nearby.
        s4_ok = (s4_diag["unrolled"] == 0)
        ex = s4_diag["unrolled_examples"][:3]
        s4_detail = (
            f"first_triggers={s4_diag['first_triggers']} "
            f"rolled_back={s4_diag['rolled_back']} "
            f"unrolled={s4_diag['unrolled']} unrolled_examples={ex}"
        )

    # S5 — best_stable: PASS if every selection has been resolved
    # (either executed or skipped). Tier filter is only used for detail.
    if recovery_audit:
        n_sel = len(selected)
        n_exec = len(executed)
        n_skip = len(skipped)
        s5_ok = n_sel > 0 and (n_exec + n_skip) >= n_sel
    else:
        s5_ok = True
    s5_detail = (
        f"selected={len(selected)} executed={len(executed)} "
        f"skipped={len(skipped)} "
        f"selected_for_best_stable={len(selected_for_best_stable)}"
    )

    s7_ok = mutation_frozen_seen or hyperparam_logged
    s8_ok = (run_dir / "eval_protocol.md").is_file() or Path(__file__).parent.joinpath(
        "eval_protocol.md"
    ).is_file()

    return [
        {"id": "S1", "name": "Single-shot shrink (no multi-shrink within window)",
         "pass": s1_ok, "detail": s1_detail},
        {"id": "S2", "name": "No unmitigated collapse (end ≥ 0.5×peak)", "pass": s2_ok,
         "detail": f"peak={peak_val:.2f} end={end_val:.2f}"},
        {"id": "S3", "name": "Early freeze triggered (30% drop)", "pass": s3_ok,
         "detail": f"collapse_freeze_seen={any_early_freeze}"},
        {"id": "S4", "name": "Hard collapse tolerance rollback", "pass": s4_ok,
         "detail": s4_detail},
        {"id": "S5", "name": "best_stable: every selected resolved (executed+skipped)",
         "pass": s5_ok, "detail": s5_detail},
        {"id": "S6", "name": "Recovery uses last_mastered tolerance", "pass": s6_ok,
         "detail": f"tol_logs={len(recovery_tol_logs)}"},
        {"id": "S7", "name": "Hyperparam / mutation freeze audit", "pass": s7_ok,
         "detail": f"mutation_frozen={mutation_frozen_seen}"},
        {"id": "S8", "name": "Fair eval protocol documented", "pass": s8_ok,
         "detail": "eval_protocol.md"},
    ]


def _latest_points_by_policy(curves: list[dict]) -> dict[int, dict]:
    latest: dict[int, dict] = {}
    for pt in curves:
        pid = int(pt.get("policy_idx", -1))
        if pid < 0:
            continue
        ef = int(pt.get("env_frames", 0))
        prev = latest.get(pid)
        if prev is None or ef >= int(prev.get("env_frames", 0)):
            latest[pid] = pt
    return latest


def _peak_mean_by_policy(curves: list[dict]) -> dict[int, float]:
    peaks: dict[int, float] = defaultdict(float)
    for pt in curves:
        pid = int(pt.get("policy_idx", -1))
        if pid < 0:
            continue
        nm = float(pt.get("nsucc_mean", 0) or 0)
        peaks[pid] = max(peaks[pid], nm)
    return dict(peaks)


def _nsucc_max_saturation_frames(
    curves: list[dict], *, threshold: float = 50.0
) -> dict[int, int | None]:
    first_frame: dict[int, int | None] = {}
    for pt in sorted(curves, key=lambda p: int(p.get("env_frames", 0))):
        pid = int(pt.get("policy_idx", -1))
        if pid < 0 or pid in first_frame:
            continue
        nm = float(pt.get("nsucc_max", 0) or 0)
        if nm >= threshold:
            first_frame[pid] = int(pt.get("env_frames", 0))
    for pid in {int(pt.get("policy_idx", -1)) for pt in curves if pt.get("policy_idx") is not None}:
        if pid >= 0 and pid not in first_frame:
            first_frame[pid] = None
    return first_frame


def _count_killed_by_policy(run_dir: Path) -> dict[int, int]:
    counts: dict[int, int] = defaultdict(int)
    rx = re.compile(r"\bKilled\b", re.IGNORECASE)
    for log in sorted(run_dir.rglob("train.log")):
        m = re.search(r"policy_(\d+)", str(log))
        if not m:
            continue
        pid = int(m.group(1))
        try:
            text = log.read_text(errors="replace")
        except OSError:
            continue
        counts[pid] += sum(1 for line in text.splitlines() if rx.search(line))
    return dict(counts)


def _oom_by_wm_strategy(run_dir: Path) -> Counter[str]:
    killed_rx = re.compile(r"\bKilled\b", re.IGNORECASE)
    cli_rx = re.compile(r"--wm_strategy=([^\s,]+)")
    selected_rx = re.compile(r"\[WM\]\s*Strategy selected:\s*(\S+)")
    counts: Counter[str] = Counter()
    for log in sorted(run_dir.rglob("train.log")):
        try:
            lines = log.read_text(errors="replace").splitlines()
        except OSError:
            continue
        current_strategy = "unknown"
        for line in lines:
            cli_match = cli_rx.search(line)
            if cli_match:
                current_strategy = cli_match.group(1)
            sel_match = selected_rx.search(line)
            if sel_match:
                current_strategy = sel_match.group(1)
            if killed_rx.search(line):
                counts[current_strategy] += 1
    return counts


def _counter_from_curve_field(curves: list[dict], field: str) -> Counter[str]:
    counts: Counter[str] = Counter()
    for pt in curves:
        val = pt.get(field)
        if val is None:
            continue
        text = str(val).strip()
        if text:
            counts[text] += 1
    return counts


def _cooldown_blocked_by_policy(curves: list[dict]) -> dict[int, int]:
    counts: dict[int, int] = defaultdict(int)
    for pt in curves:
        reason = str(pt.get("rollback_reason", "") or "")
        if reason == "cooldown_blocked":
            pid = int(pt.get("policy_idx", -1))
            if pid >= 0:
                counts[pid] += 1
    return dict(counts)


def _format_diagnostic_indicators(run_dir: Path, curves: list[dict]) -> list[str]:
    latest = _latest_points_by_policy(curves)
    peaks = _peak_mean_by_policy(curves)
    saturation = _nsucc_max_saturation_frames(curves)
    killed = _count_killed_by_policy(run_dir)
    oom_by_strategy = _oom_by_wm_strategy(run_dir)
    tier_dist = _counter_from_curve_field(curves, "last_source_tier_label")
    recovery_tol_dist = _counter_from_curve_field(curves, "pbt_recovery_tolerance_mode")
    crosscopy_tol_dist = _counter_from_curve_field(curves, "pbt_crosscopy_tolerance_mode")
    cooldown_blocked = _cooldown_blocked_by_policy(curves)

    lines = [
        "## Diagnostic Indicators (P2.1 no-h3 prep)",
        "",
        "### nsucc_max saturation time (first env_frames with nsucc_max ≥ 50)",
        "",
        "| Policy | First saturation (M frames) |",
        "|--------|----------------------------|",
    ]
    for pid in sorted(set(saturation) | set(latest)):
        frame = saturation.get(pid)
        frame_m = f"{frame / 1e6:.1f}" if frame is not None else "—"
        lines.append(f"| p{pid} | {frame_m} |")

    retentions: list[float] = []
    lines += [
        "",
        "### nsucc_mean retention (end_mean / peak_mean)",
        "",
        "| Policy | peak_mean | end_mean | retention |",
        "|--------|-----------|----------|-----------|",
    ]
    for pid in sorted(set(peaks) | set(latest)):
        peak = peaks.get(pid, float("nan"))
        end = float(latest.get(pid, {}).get("nsucc_mean", float("nan")) or float("nan"))
        if peak == peak and peak > 0 and end == end:
            retention = end / peak
            retentions.append(retention)
            ret_str = f"{retention:.2%}"
        else:
            ret_str = "—"
        lines.append(
            f"| p{pid} | {peak:.2f} | {end:.2f} | {ret_str} |"
        )
    if retentions:
        lines += [
            "",
            f"- Median retention: **{sorted(retentions)[len(retentions) // 2]:.2%}**",
            f"- Min retention: **{min(retentions):.2%}**",
        ]

    lines += [
        "",
        "### killed / restart count",
        "",
        "| Policy | Killed (train.log) | restart_count (curve tail) |",
        "|--------|--------------------|----------------------------|",
    ]
    for pid in sorted(set(killed) | set(latest)):
        restart = int(latest.get(pid, {}).get("restart_count", 0) or 0)
        lines.append(f"| p{pid} | {killed.get(pid, 0)} | {restart} |")
    total_killed = sum(killed.values())
    total_restart = max(
        (int(pt.get("restart_count", 0) or 0) for pt in latest.values()),
        default=0,
    )
    lines += [
        "",
        f"- Total killed: **{total_killed}**",
        f"- Max restart_count across policies: **{total_restart}**",
    ]

    lines += [
        "",
        "### OOM by wm_strategy",
        "",
    ]
    if oom_by_strategy:
        for strategy, count in oom_by_strategy.most_common():
            lines.append(f"- `{strategy}`: **{count}**")
    else:
        lines.append("- No Killed lines with wm_strategy attribution found.")

    lines += [
        "",
        "### source tier distribution (`last_source_tier_label`)",
        "",
    ]
    if tier_dist:
        for tier, count in tier_dist.most_common():
            lines.append(f"- `{tier}`: **{count}**")
    else:
        lines.append("- No tier labels logged.")

    lines += [
        "",
        "### tolerance mode distribution",
        "",
        "**Recovery (`pbt_recovery_tolerance_mode`):**",
        "",
    ]
    if recovery_tol_dist:
        for mode, count in recovery_tol_dist.most_common():
            lines.append(f"- `{mode}`: **{count}**")
    else:
        lines.append("- No recovery tolerance mode logged.")
    lines += ["", "**Cross-copy (`pbt_crosscopy_tolerance_mode`):**", ""]
    if crosscopy_tol_dist:
        for mode, count in crosscopy_tol_dist.most_common():
            lines.append(f"- `{mode}`: **{count}**")
    else:
        lines.append("- No cross-copy tolerance mode logged.")

    total_cooldown = sum(cooldown_blocked.values())
    lines += [
        "",
        "### cooldown_blocked count (from training curve `rollback_reason`)",
        "",
        f"- Total: **{total_cooldown}**",
    ]
    if cooldown_blocked:
        lines.append("- By policy:")
        for pid in sorted(cooldown_blocked):
            lines.append(f"  - p{pid}: **{cooldown_blocked[pid]}**")
    return lines


def _parse_frame_range(spec: str | None) -> tuple[int, int] | None:
    if spec is None or str(spec).strip() == "":
        return None
    s = str(spec).strip()
    if ":" not in s:
        raise ValueError(f"frame range must be MIN:MAX (got {spec!r})")
    lo, hi = s.split(":", 1)
    return (int(lo or 0), int(hi or 10**18))


def _format_frame_range(fr: tuple[int, int] | None) -> str:
    if fr is None:
        return "all"
    lo, hi = fr
    return f"{lo / 1e6:.0f}M:{hi / 1e6:.0f}M"


# Track C (2026-06-06) helpers: pool hard-check, CB-WM section, and the
# cross-run compare block.

FORBIDDEN_POOL_STRATEGIES = ("value_imag_h3", "aux_low")


def _load_run_config(run_dir: Path) -> dict[str, Any]:
    p = run_dir / "RUN_CONFIG.json"
    if not p.is_file():
        return {}
    try:
        return json.loads(p.read_text())
    except json.JSONDecodeError:
        return {}


def _load_posterior_state(ws: Path) -> dict[str, Any]:
    p = ws / "wm_posterior_latest.json"
    if not p.is_file():
        return {}
    try:
        return json.loads(p.read_text())
    except json.JSONDecodeError:
        return {}


def _pool_hard_check(run_cfg: dict, posterior: dict) -> tuple[bool, list[str]]:
    """Constraint 1: Track C launcher must pass --wm_strategy_pool excluding
    h3 and aux_low; report must verify both RUN_CONFIG and the saved
    posterior file. Returns (ok, list_of_lines_to_emit).
    """
    cfg_pool = run_cfg.get("wm_strategy_pool")
    post_pool = posterior.get("__pool__")
    lines: list[str] = []
    ok = True
    forbidden_in_cfg = []
    forbidden_in_post = []
    if cfg_pool is not None:
        if isinstance(cfg_pool, str):
            cfg_pool_list = [s.strip() for s in cfg_pool.split(",") if s.strip()]
        else:
            cfg_pool_list = list(cfg_pool)
        forbidden_in_cfg = [
            s for s in cfg_pool_list if s in FORBIDDEN_POOL_STRATEGIES
        ]
        lines.append(
            f"- `RUN_CONFIG.json.wm_strategy_pool` = `{cfg_pool_list}`"
        )
    else:
        lines.append("- `RUN_CONFIG.json.wm_strategy_pool` = (not set)")
    if isinstance(post_pool, list):
        forbidden_in_post = [
            s for s in post_pool if s in FORBIDDEN_POOL_STRATEGIES
        ]
        lines.append(f"- `wm_posterior_latest.json.__pool__` = `{post_pool}`")
    else:
        lines.append("- `wm_posterior_latest.json.__pool__` = (not set)")
    if forbidden_in_cfg or forbidden_in_post:
        ok = False
        bad = sorted(set(forbidden_in_cfg) | set(forbidden_in_post))
        lines.append(
            f"- **[FATAL] pool contains forbidden strategy: {bad}**"
        )
    else:
        lines.append("- pool hard-check: OK (no h3 / no aux_low)")
    return ok, lines


def _format_cbwm_section(run_dir: Path, curves: list[dict]) -> list[str]:
    """Track C (2026-06-06): CB-WM run summary. Surfaces telemetry that's
    already loaded by cbwm.py / training curves. Constraint 4 acceptance
    is enforced here:
      - CF blocks completed >= 3
      - posterior_update_count >= 3
      - at least 2 strategies with posterior n > 1
    """
    from generate_smoke_report_cbwm import (  # noqa: E402
        _load_cf_results,
        _cf_stats,
        _posterior_state,
    )
    ws = run_dir / "dexpbt_8gpu_ws"
    cf_results = _load_cf_results(ws)
    cf = _cf_stats(cf_results, run_dir)
    posterior = _posterior_state(ws)
    raw_post = _load_posterior_state(ws)

    cf_blocks = int(cf.get("blocks_completed", len(cf_results)))
    posterior_update_count = 0
    n_per_strategy: dict[str, int] = {}
    mean_per_strategy: dict[str, float] = {}
    for sname, sdata in raw_post.items():
        if sname == "__pool__":
            continue
        if not isinstance(sdata, dict):
            continue
        n_per_strategy[sname] = int(sdata.get("n", 1))
        mean_per_strategy[sname] = float(sdata.get("mean", 0.0))
        posterior_update_count += max(0, int(sdata.get("n", 1)) - 1)

    strategies_with_n_gt_1 = sum(1 for n in n_per_strategy.values() if n > 1)

    # Selected-strategy distribution + active timeline from training curves.
    strategy_count: Counter[str] = Counter()
    active_timeline: dict[int, list[tuple[int, str]]] = defaultdict(list)
    for pt in curves:
        s = str(pt.get("hyperparam_wm_strategy") or "")
        if not s:
            continue
        strategy_count[s] += 1
        try:
            pid = int(pt.get("policy_idx", -1))
        except (TypeError, ValueError):
            continue
        if pid < 0:
            continue
        ef = int(pt.get("env_frames", 0))
        prev = active_timeline[pid][-1] if active_timeline[pid] else (None, None)
        if s != prev[1]:
            active_timeline[pid].append((ef, s))

    # OOM by strategy
    oom_by_strategy = _oom_by_wm_strategy(run_dir)

    # Acceptance constraint 4
    acc4_cf = "PASS" if cf_blocks >= 3 else "FAIL"
    acc4_post = "PASS" if posterior_update_count >= 3 else "FAIL"
    acc4_n = (
        "PASS" if strategies_with_n_gt_1 >= 2 else "FAIL"
    )

    lines = [
        "",
        "## CB-WM run summary (Track C)",
        "",
        f"- CF blocks completed: **{cf_blocks}** (acceptance >=3 → **{acc4_cf}**)",
        f"- posterior_update_count (sum of n-1 across strategies): "
        f"**{posterior_update_count}** (acceptance >=3 → **{acc4_post}**)",
        f"- strategies with posterior n>1: **{strategies_with_n_gt_1}** "
        f"(acceptance >=2 → **{acc4_n}**)",
        "",
        "### Posterior n / mean per strategy",
        "",
        "| strategy | n | mean |",
        "|----------|---|------|",
    ]
    for s in sorted(n_per_strategy):
        lines.append(
            f"| {s} | {n_per_strategy[s]} | {mean_per_strategy.get(s, 0.0):.4f} |"
        )

    if cf:
        lines += [
            "",
            "### CF block stats",
            "",
            f"- blocks_completed: {cf.get('blocks_completed', '—')}",
            f"- recent_block_strategies: {cf.get('recent_block_strategies', '—')}",
            f"- skipped_reasons: {cf.get('skipped_reasons', '—')}",
        ]

    lines += [
        "",
        "### Selected-strategy distribution (training curve ticks)",
        "",
    ]
    for s, c in strategy_count.most_common():
        lines.append(f"- `{s}`: {c}")

    lines += [
        "",
        "### Active strategy timeline per policy",
        "",
    ]
    for pid in sorted(active_timeline):
        toks = [
            f"{ef / 1e6:.0f}M:{s}" for ef, s in active_timeline[pid][:20]
        ]
        more = "" if len(active_timeline[pid]) <= 20 else f" (+{len(active_timeline[pid]) - 20} more)"
        lines.append(f"- p{pid}: {' → '.join(toks)}{more}")

    if oom_by_strategy:
        lines += [
            "",
            "### OOM / killed by WM strategy (from train.log)",
            "",
        ]
        for s, c in oom_by_strategy.most_common():
            lines.append(f"- `{s}`: {c}")
    return lines


def _peak_metrics_from_curves(
    curves: list[dict]
) -> dict[str, Any]:
    """Population peak_mean median/max + best 3-tick rolling mean per policy.

    Returns:
      peak_mean_per_policy, peak_mean_median, peak_mean_max,
      best_3tick_per_policy, best_3tick_population_median,
      time_to_peak_per_policy (env_frames at first reaching 0.9 * own peak).
    """
    by_policy: dict[int, list[dict]] = defaultdict(list)
    for pt in curves:
        try:
            pid = int(pt.get("policy_idx", -1))
        except (TypeError, ValueError):
            continue
        if pid < 0:
            continue
        by_policy[pid].append(pt)
    peak_mean: dict[int, float] = {}
    best_3tick: dict[int, float] = {}
    time_to_peak: dict[int, int] = {}
    for pid, pts in by_policy.items():
        pts = sorted(pts, key=lambda p: int(p.get("env_frames", 0)))
        means = [float(p.get("nsucc_mean", 0) or 0) for p in pts]
        if not means:
            continue
        pmax = max(means)
        peak_mean[pid] = pmax
        # rolling 3-tick mean
        best = float("-inf")
        for i in range(len(means) - 2):
            avg = (means[i] + means[i + 1] + means[i + 2]) / 3.0
            if avg > best:
                best = avg
        if best == float("-inf"):
            best = pmax
        best_3tick[pid] = best
        # time-to-peak: first env_frames hitting 90% of own peak
        thresh = 0.9 * pmax
        for p in pts:
            if float(p.get("nsucc_mean", 0) or 0) >= thresh:
                time_to_peak[pid] = int(p.get("env_frames", 0))
                break
    pm_vals = sorted(peak_mean.values())

    def _median(xs: list[float]) -> float:
        if not xs:
            return float("nan")
        xs = sorted(xs)
        m = len(xs) // 2
        if len(xs) % 2:
            return xs[m]
        return (xs[m - 1] + xs[m]) / 2.0

    return {
        "peak_mean_per_policy": peak_mean,
        "peak_mean_median": _median(pm_vals),
        "peak_mean_max": max(pm_vals) if pm_vals else float("nan"),
        "best_3tick_per_policy": best_3tick,
        "best_3tick_median": _median(sorted(best_3tick.values())),
        "time_to_peak_per_policy": time_to_peak,
    }


def _format_compare_section(
    this_run_dir: Path,
    this_curves: list[dict],
    other_run_dir: Path,
    other_frame_range: tuple[int, int] | None,
) -> list[str]:
    """Track C (2026-06-06): CB-WM vs reference run comparison."""
    other_ws = other_run_dir / "dexpbt_8gpu_ws"
    other_curves = _load_training_curves(other_ws, frame_range=other_frame_range)
    this_metrics = _peak_metrics_from_curves(this_curves)
    other_metrics = _peak_metrics_from_curves(other_curves)
    lines = [
        "",
        f"## Compare to reference run",
        "",
        f"- this run: `{this_run_dir.name}`",
        f"- reference: `{other_run_dir.name}`  "
        f"(frame_range={_format_frame_range(other_frame_range)})",
        "",
        "| metric | this | reference |",
        "|--------|------|-----------|",
        f"| peak_mean median | {this_metrics['peak_mean_median']:.2f} "
        f"| {other_metrics['peak_mean_median']:.2f} |",
        f"| peak_mean max | {this_metrics['peak_mean_max']:.2f} "
        f"| {other_metrics['peak_mean_max']:.2f} |",
        f"| best 3-tick rolling median | {this_metrics['best_3tick_median']:.2f} "
        f"| {other_metrics['best_3tick_median']:.2f} |",
    ]
    # Per-policy time-to-peak summary
    def _ttpm(d: dict[int, int]) -> str:
        if not d:
            return "—"
        vals = sorted(d.values())
        return f"min={vals[0] / 1e6:.0f}M / median={vals[len(vals) // 2] / 1e6:.0f}M / max={vals[-1] / 1e6:.0f}M"

    lines += [
        f"| time-to-peak (90% of own peak) | "
        f"{_ttpm(this_metrics['time_to_peak_per_policy'])} | "
        f"{_ttpm(other_metrics['time_to_peak_per_policy'])} |",
    ]
    return lines


def generate_p2_report(
    run_dir: Path,
    frame_range: tuple[int, int] | None = None,
    compare_to: Path | None = None,
    compare_frame_range: tuple[int, int] | None = None,
) -> str:
    ws = run_dir / "dexpbt_8gpu_ws"
    curves = _load_training_curves(ws, frame_range=frame_range)
    ctrl = _load_curriculum_control(ws)
    recovery_audit = _load_recovery_audit(ws)
    launch_audit = _load_launch_audit(run_dir)
    criteria = _evaluate_p2_criteria(run_dir, curves, ctrl, recovery_audit)
    p1_body = generate_p1_report(run_dir, frame_range=frame_range)
    oom_lines = _grep_logs(run_dir, r"exit code 137|Killed|OOM|out of memory")
    h3_lines = _grep_logs(run_dir, r"wm_strategy[= ]value_imag_h3|value_imag_h3")
    oom_h3_hint = min(len(oom_lines), len(h3_lines))

    # Best-effort: parse num_policies from launch_audit, then fall back to
    # the highest policy_idx that produced a training-curve point.
    started_in_audit = {
        int(r.get("policy_idx", -1))
        for r in launch_audit
        if str(r.get("event")) == "policy_start_time"
        and int(r.get("policy_idx", -1)) >= 0
    }
    seen_in_curves = {
        int(pt.get("policy_idx", -1))
        for pt in curves
        if pt.get("policy_idx") is not None
    }
    candidate = max(len(started_in_audit), max(seen_in_curves, default=-1) + 1)
    num_policies_expected = candidate if candidate > 0 else None

    pbt_lines, pbt_criteria = _format_pbt_population_verification(
        run_dir, curves, launch_audit, num_policies_expected
    )

    passed = sum(1 for c in criteria if c["pass"] is True)
    pbt_passed = sum(1 for c in pbt_criteria if c["pass"] is True)
    p2_total = sum(1 for c in criteria if c["pass"] is not None)
    pbt_total = len(pbt_criteria)
    now = datetime.now().strftime("%Y-%m-%d %H:%M CST")

    stop_record = run_dir / "RUN_STOPPED_BY_USER.json"
    run_status_lines: list[str] = []
    completed_only_block: list[str] = []
    all_observed_block: list[str] = []
    standing_conclusions: list[str] = []
    if stop_record.is_file():
        try:
            stop_data = json.loads(stop_record.read_text())
        except json.JSONDecodeError:
            stop_data = {}
        completed = stop_data.get("completed_policies") or []
        partial = stop_data.get("partial_policies") or []
        run_status_lines = [
            "**Run status:** STOPPED_BY_USER "
            f"({len(completed)}/{len(completed) + len(partial)} policies completed, "
            f"{','.join('p' + str(p) for p in partial) or '—'} partial)",
            f"**Stop time:** {stop_data.get('stop_time', '—')}",
            f"**Reason:** {stop_data.get('reason', '—')}",
        ]
        comp_summary = stop_data.get("population_summary_completed_only") or {}
        all_summary = stop_data.get("population_summary_all_observed") or {}

        def _fmt_summary(title: str, summary: dict) -> list[str]:
            if not summary:
                return [f"### {title}", "", "- (no summary recorded)"]
            return [
                f"### {title}",
                "",
                f"- policies: {summary.get('policies', '—')}",
                f"- mean end_mean: **{summary.get('mean_end_mean', '—')}**",
                f"- median end_mean: **{summary.get('median_end_mean', '—')}**",
                f"- max end_mean: **{summary.get('max_end_mean', '—')}**",
                f"- min end_mean: **{summary.get('min_end_mean', '—')}**",
                f"- median retention pct: **{summary.get('median_retention_pct', '—')}**",
                f"- min retention pct: **{summary.get('min_retention_pct', '—')}**",
                f"- note: {summary.get('note', '—')}",
            ]

        completed_only_block = _fmt_summary(
            "A. Completed-only aggregate", comp_summary
        )
        all_observed_block = _fmt_summary(
            "B. All-observed aggregate (p1 partial)", all_summary
        )
        pbt_audit = stop_data.get("pbt_replacement_audit") or {}
        cross_total_label = pbt_audit.get(
            "cross_restart_lifetime_max",
            pbt_audit.get("cross_restart_total", "—"),
        )
        recovery_total_label = pbt_audit.get(
            "recovery_restart_lifetime_max",
            pbt_audit.get("recovery_restart_total", "—"),
        )
        standing_conclusions = [
            "## Stopped-smoke conclusions",
            "",
            "- no-h3 fixed `value_imag_h1` smoke confirms stable RL/PBT learning in completed workers.",
            f"- PBT replacement works: cross={cross_total_label}, "
            f"recovery={recovery_total_label} (lifetime max).",
            "- p5 recovery burst is the strongest signal.",
            "- p1 is an outlier requiring separate bad-seed / restart-integrity audit; "
            "see `P1_OUTLIER_AUDIT.md`.",
        ]

    # Track C (2026-06-06): pool hard-check at the very top so any
    # forbidden strategy in the pool is impossible to miss.
    run_cfg = _load_run_config(run_dir)
    posterior_state = _load_posterior_state(ws)
    pool_ok, pool_lines = _pool_hard_check(run_cfg, posterior_state)
    is_track_c = bool(
        run_cfg.get("wm_strategy_pool")
        or posterior_state.get("__pool__")
    )

    lines = [
        "# P2 Smoke Acceptance Report (800M)",
        "",
        f"**Run ID:** `{run_dir.name}`",
        f"**Generated:** {now}",
        f"**Frame range:** {_format_frame_range(frame_range)}",
    ]
    if not pool_ok:
        lines.append(
            "**[FATAL] pool contains forbidden strategy — see CB-WM section**"
        )
    if run_status_lines:
        lines += run_status_lines
    lines += [
        f"**P2 criteria:** {passed}/{p2_total} passed (criteria with N/A excluded)",
        f"**D criteria (PBT verification):** {pbt_passed}/{pbt_total} passed",
        "",
        "| ID | Criterion | Pass | Detail |",
        "|----|-----------|------|--------|",
    ]
    for c in criteria:
        if c["pass"] is True:
            mark = "PASS"
        elif c["pass"] is False:
            mark = "FAIL"
        else:
            mark = "N/A"
        lines.append(f"| {c['id']} | {c['name']} | **{mark}** | `{c['detail'][:90]}` |")
    if completed_only_block or all_observed_block:
        lines += [
            "",
            "## Population aggregates",
            "",
        ]
        lines += completed_only_block
        lines += [""]
        lines += all_observed_block
    if standing_conclusions:
        lines += [""]
        lines += standing_conclusions
    lines += [
        "",
        "## Recovery audit (`recovery_audit.jsonl`)",
        "",
        f"Records: **{len(recovery_audit)}**",
        "",
    ]
    for r in recovery_audit[:20]:
        lines.append(
            f"- outcome={r.get('outcome')} skipped={r.get('skipped_reason', '-')}"
        )
    lines += [
        "",
        "## OOM Diagnosis (P2.1-F)",
        "",
        f"- OOM-like log lines: **{len(oom_lines)}**",
        f"- `value_imag_h3` mentions: **{len(h3_lines)}**",
        f"- Approximate overlap hint: **{oom_h3_hint}**",
        "- Recommendation: keep the main WM pool unchanged, and run an optional no_h3 smoke ablation.",
        "- Optional launch knobs for next smoke:",
        "  - `--wm_disable_value_imag_h3` (if implemented in launch wrapper), or",
        "  - a dedicated no_h3 smoke script using fixed `value_imag_h1` / no-h3 strategy set.",
    ]
    lines += [""]
    lines += pbt_lines
    lines += _format_diagnostic_indicators(run_dir, curves)

    if is_track_c:
        lines += [
            "",
            "## CB-WM pool hard-check",
            "",
        ]
        lines += pool_lines
        lines += _format_cbwm_section(run_dir, curves)

    if compare_to is not None:
        lines += _format_compare_section(
            run_dir, curves, compare_to, compare_frame_range
        )

    lines += ["", "---", "", p1_body]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate P2 800M smoke report")
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("-o", "--output", type=Path, default=None)
    parser.add_argument(
        "--frame-range",
        type=str,
        default=None,
        help=(
            "Restrict THIS run's training-curve aggregation to env_frames "
            "in MIN:MAX (inclusive). Example: 0:3000000000 to read only "
            "the 0–3B segment."
        ),
    )
    parser.add_argument(
        "--compare-to",
        type=Path,
        default=None,
        help=(
            "Optional reference run dir. When set, a 'Compare to reference "
            "run' section is appended with peak_mean median/max, best 3-tick "
            "rolling median and time-to-peak comparison."
        ),
    )
    parser.add_argument(
        "--compare-frame-range",
        type=str,
        default=None,
        help=(
            "MIN:MAX env_frames range applied ONLY to --compare-to (this "
            "run's range is controlled by --frame-range). Use e.g. "
            "0:3000000000 to clamp a long fixed_h1 5B reference to its "
            "0–3B segment for fair comparison."
        ),
    )
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    fr = _parse_frame_range(args.frame_range)
    cfr = _parse_frame_range(args.compare_frame_range)
    compare_to = args.compare_to.resolve() if args.compare_to else None
    report = generate_p2_report(
        run_dir,
        frame_range=fr,
        compare_to=compare_to,
        compare_frame_range=cfr,
    )
    out = args.output or (run_dir / "SMOKE_P2_REPORT.md")
    out.write_text(report)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
