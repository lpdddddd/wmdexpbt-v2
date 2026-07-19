"""Read-only 3B-5B first-blocker audit for a completed base-retention run.

Answers: at each potential shrink opportunity between start_frame and
end_frame, which gate blocked FIRST (in the runtime precedence order)?

Sources (never written to):
- <run>/dexpbt_8gpu_ws/metrics/curriculum_safety_events.jsonl
    frame-tagged block events emitted by the observer:
      late_floor_retention_block, danger_zone_block, real_signal_block
      (global retention / real-signal safety), repeated_collapse_freeze_only,
      repeated_collapse_rollback, pre_floor_peak_retention_guard.
- <run>/dexpbt_8gpu_ws/metrics/training_curve_policy_XX.jsonl
    periodic state snapshots: success_tolerance, shrink_token,
    curriculum_blocked_by_consolidation / _mastery_gate / _collapse /
    _recovery, collapse_freeze_until_env_frames, env_frames.

An "opportunity" is one snapshot interval in which success_tolerance did
NOT decrease. The first blocker is attributed with the runtime precedence
(env-side mastery shrink order, then observer-side gates):

  post_shrink_consolidation > rollback/collapse freeze > interval/recovery
  > mastery_gate > pre_floor_peak_guard > late_floor_retention_gate
  > danger_zone > global_retention > other

Pure python + json; safe to run anywhere; makes zero writes to the run dir.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

BLOCKER_ORDER = [
    "post_shrink_consolidation",
    "rollback_collapse_freeze",
    "interval",
    "mastery_gate",
    "pre_floor_peak_guard",
    "late_floor_retention_gate",
    "danger_zone",
    "global_retention",
    "other",
]

_EVENT_TO_BLOCKER = {
    "pre_floor_peak_retention_guard": "pre_floor_peak_guard",
    "late_floor_retention_block": "late_floor_retention_gate",
    "danger_zone_block": "danger_zone",
    "real_signal_block": "global_retention",
    "repeated_collapse_freeze_only": "rollback_collapse_freeze",
    "repeated_collapse_rollback": "rollback_collapse_freeze",
}


def _load_events(run_dir: Path) -> list[dict]:
    p = run_dir / "dexpbt_8gpu_ws" / "metrics" / "curriculum_safety_events.jsonl"
    events = []
    if p.is_file():
        for line in p.read_text().splitlines():
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return events


def _load_curve(run_dir: Path, policy_idx: int) -> list[dict]:
    p = (run_dir / "dexpbt_8gpu_ws" / "metrics"
         / f"training_curve_policy_{policy_idx:02d}.jsonl")
    rows = []
    if p.is_file():
        for line in p.read_text().splitlines():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    rows.sort(key=lambda r: r.get("env_frames", 0))
    return rows


_SHRINK_REASON_TO_BLOCKER = {
    # authoritative per-tick field written by the observer
    "blocked_consolidation": "post_shrink_consolidation",
    "blocked_collapse": "rollback_collapse_freeze",
    "blocked_recovery": "interval",
    "blocked_mastery_gate": "mastery_gate",
    "blocked_low_nsucc": "other",
    "pre_floor_peak_retention_guard": "pre_floor_peak_guard",
}

_SAFETY_REASON_TO_BLOCKER = {
    "real_signal_risk": "global_retention",
    "danger_zone_not_stable": "danger_zone",
    "danger_zone_insufficient_stability": "danger_zone",
}


def _first_blocker(prev: dict, curr: dict, window_events: list[dict]) -> str:
    """Attribute the first blocker for one no-shrink interval.

    Primary source: the observer's per-tick `curriculum_shrink_reason`.
    `curriculum_safety_blocked` is sub-attributed via
    `curriculum_safety_block_reason`; a generic `curriculum_safety_freeze_
    active` (freeze armed at an earlier tick) is resolved with the
    frame-tagged gate events inside the window (late_floor / danger /
    global-retention / pre-floor), falling back to the freeze itself.
    """
    reason = str(curr.get("curriculum_shrink_reason", ""))
    if reason in _SHRINK_REASON_TO_BLOCKER:
        return _SHRINK_REASON_TO_BLOCKER[reason]
    if reason == "curriculum_safety_blocked":
        sub = str(curr.get("curriculum_safety_block_reason", ""))
        if sub in _SAFETY_REASON_TO_BLOCKER:
            return _SAFETY_REASON_TO_BLOCKER[sub]
        # freeze_active: find which gate armed it via window events
        evs = Counter(_EVENT_TO_BLOCKER.get(e.get("event"), "other")
                      for e in window_events)
        for blocker in BLOCKER_ORDER:
            if evs.get(blocker):
                return blocker
        return "rollback_collapse_freeze"

    # Fallbacks for rows without the field (older schema).
    if curr.get("curriculum_blocked_by_consolidation"):
        return "post_shrink_consolidation"
    if (curr.get("curriculum_blocked_by_collapse")
            or curr.get("collapse_freeze_active")
            or curr.get("rollback_cooldown_active")
            or curr.get("collapse_rollback_active")):
        return "rollback_collapse_freeze"
    if curr.get("curriculum_blocked_by_recovery"):
        return "interval"
    if curr.get("curriculum_blocked_by_mastery_gate"):
        return "mastery_gate"
    evs = Counter(_EVENT_TO_BLOCKER.get(e.get("event"), "other")
                  for e in window_events)
    for blocker in BLOCKER_ORDER:
        if evs.get(blocker):
            return blocker
    if not curr.get("shrink_token", False):
        return "mastery_gate"
    return "other"


def audit_run(run_dir: Path, start_frame: int, end_frame: int,
              num_policies: int = 6) -> dict:
    events = _load_events(run_dir)
    events_by_policy: dict[int, list[dict]] = defaultdict(list)
    for e in events:
        events_by_policy[int(e.get("policy_idx", -1))].append(e)

    per_policy: dict[str, dict] = {}
    total = Counter()
    event_total = Counter()
    for pol in range(num_policies):
        rows = [r for r in _load_curve(run_dir, pol)
                if start_frame <= int(r.get("env_frames", -1)) <= end_frame]
        counts = Counter()
        opportunities = 0
        shrinks = 0
        for prev, curr in zip(rows, rows[1:]):
            lo = int(prev.get("env_frames", 0))
            hi = int(curr.get("env_frames", 0))
            tol_prev = float(prev.get("success_tolerance", float("nan")))
            tol_curr = float(curr.get("success_tolerance", float("nan")))
            if tol_curr < tol_prev - 1e-9:
                shrinks += 1
                continue
            opportunities += 1
            window_events = [e for e in events_by_policy.get(pol, [])
                             if lo < int(e.get("env_frames", -1)) <= hi]
            counts[_first_blocker(prev, curr, window_events)] += 1
        ev_counts = Counter(
            _EVENT_TO_BLOCKER.get(e.get("event"), "other")
            for e in events_by_policy.get(pol, [])
            if start_frame <= int(e.get("env_frames", -1)) <= end_frame)
        per_policy[f"p{pol}"] = {
            "no_shrink_intervals": opportunities,
            "shrink_intervals": shrinks,
            "first_blocker_counts": dict(counts),
            "gate_block_event_counts": dict(ev_counts),
        }
        total.update(counts)
        event_total.update(ev_counts)

    dominant = total.most_common(1)[0][0] if total else "none"
    return {
        "run_dir": str(run_dir),
        "window_frames": [start_frame, end_frame],
        "num_policies": num_policies,
        "per_policy": per_policy,
        "total_first_blocker_counts": dict(total),
        "total_gate_block_event_counts": dict(event_total),
        "dominant_first_blocker": dominant,
        "note": ("read-only audit; attribution combines snapshot "
                 "curriculum_blocked_by_* flags (env precedence) with "
                 "frame-tagged curriculum_safety_events (observer gates); "
                 "intervals are training_curve snapshots (~50M frames)"),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--start-frame", type=int, default=3_000_000_000)
    ap.add_argument("--end-frame", type=int, default=5_000_000_000)
    ap.add_argument("--num-policies", type=int, default=6)
    ap.add_argument("--out", required=True, help="output JSON path "
                    "(must NOT be inside the run dir)")
    args = ap.parse_args(argv)

    run_dir = Path(args.run_dir).resolve()
    out = Path(args.out).resolve()
    if str(out).startswith(str(run_dir)):
        raise SystemExit("refusing to write inside the audited run dir")

    result = audit_run(run_dir, args.start_frame, args.end_frame,
                       args.num_policies)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.parent / (out.name + ".tmp")
    tmp.write_text(json.dumps(result, indent=2))
    tmp.replace(out)
    print(json.dumps(result["total_first_blocker_counts"], indent=2))
    print(f"dominant_first_blocker={result['dominant_first_blocker']}")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
