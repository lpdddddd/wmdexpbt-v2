#!/usr/bin/env python3
"""Readonly mid-run (~1B) audit: NOW vs JUL12 Profile 22."""
from __future__ import annotations

import json
import math
import statistics as st
from collections import Counter
from pathlib import Path

NOW = Path(
    "/root/autodl-tmp/wmdexpbt/results/runs/"
    "regrasping_cser_dexpbt_progress_refine_5b_20260714_092846"
)
JUL = Path(
    "/root/autodl-tmp/wmdexpbt/results/runs/"
    "regrasping_cser_dexpbt_progress_refine_5b_20260712_114254"
)
OUT = Path("/root/autodl-tmp/wmdexpbt/results/analysis/profile22_midrun_1b_audit")
FRAMES = [200_000_000, 400_000_000, 600_000_000, 800_000_000, 1_000_000_000]
CAP = 1_050_000_000


def load_jsonl(p: Path):
    if not p.exists():
        return []
    return [json.loads(l) for l in p.open() if l.strip()]


def curves(run: Path):
    out = {}
    for i in range(6):
        rows = load_jsonl(run / "dexpbt_8gpu_ws/metrics" / f"training_curve_policy_{i:02d}.jsonl")
        out[i] = {int(r["env_frames"]): r for r in rows}
    return out


def pop(rows):
    if not rows:
        return None
    ns = [float(r["nsucc_mean"]) for _, r in rows]
    ts = [float(r["success_tolerance"]) for _, r in rows]
    rm = []
    for _, r in rows:
        try:
            fv = float(r.get("r_meta"))
            if math.isfinite(fv):
                rm.append(fv)
        except Exception:
            pass
    return {
        "n": len(rows),
        "per_nsucc": {i: float(r["nsucc_mean"]) for i, r in rows},
        "per_nmax": {i: float(r.get("nsucc_max") or 0) for i, r in rows},
        "per_tol": {i: float(r["success_tolerance"]) for i, r in rows},
        "per_rmeta": {
            i: (
                float(r["r_meta"])
                if r.get("r_meta") is not None and math.isfinite(float(r.get("r_meta")))
                else None
            )
            for i, r in rows
        },
        "best": max(ns),
        "med": st.median(ns),
        "mean": sum(ns) / len(ns),
        "tol_min": min(ts),
        "tol_med": st.median(ts),
        "rmeta_med": st.median(rm) if rm else None,
    }


def at(by_policy, frame):
    return [(i, m[frame]) for i, m in by_policy.items() if frame in m]


def shrinks_to(by_policy, upto):
    out = {}
    for i, m in by_policy.items():
        frames = sorted(f for f in m if f <= upto)
        if not frames:
            continue
        cnt = 0
        factors = []
        for fa, fb in zip(frames, frames[1:]):
            ta = float(m[fa]["success_tolerance"])
            tb = float(m[fb]["success_tolerance"])
            if tb < ta - 1e-12:
                cnt += 1
                factors.append(tb / ta if ta > 0 else None)
        last = m[frames[-1]]
        out[i] = {
            "curve_frame": frames[-1],
            "shrinks": cnt,
            "factors": sorted({round(f, 4) for f in factors if f}),
            "restart_count": int(last.get("restart_count") or 0),
            "weight_copy_count": int(last.get("weight_copy_count") or 0),
            "cross_restart_count": int(last.get("cross_restart_count") or 0),
            "self_restart_count": int(last.get("self_restart_count") or 0),
            "self_restart_blocked_count": int(last.get("self_restart_blocked_count") or 0),
            "mid_self_mutation_count": int(last.get("mid_self_mutation_count") or 0),
            "recovery_restart_count": int(last.get("recovery_restart_count") or 0),
            "allow_mid_self_mutation": bool(last.get("allow_mid_self_mutation")),
            "disable_self_restart": bool(last.get("disable_self_restart")),
            "source_tier_strict_live_count": int(last.get("source_tier_strict_live_count") or 0),
            "source_tier_relaxed_live_count": int(last.get("source_tier_relaxed_live_count") or 0),
            "source_tier_archive_count": int(last.get("source_tier_archive_count") or 0),
            "source_tier_skipped_count": int(last.get("source_tier_skipped_count") or 0),
            "best_stable_recovery_executed_count": int(
                last.get("best_stable_recovery_executed_count") or 0
            ),
            "tier_last": last.get("tier"),
            "nsucc_last": float(last.get("nsucc_mean") or 0),
            "tol_last": float(last.get("success_tolerance") or 0),
            "profile_name": last.get("wm_strategy_profile_name"),
            "profile_id": last.get("wm_strategy_profile_id"),
        }
    return out


def tier_stats(by_policy, upto):
    tiers = Counter()
    longest = {}
    visits = {i: Counter() for i in range(6)}
    for i, m in by_policy.items():
        seq = [(f, m[f].get("tier")) for f in sorted(f for f in m if f <= upto)]
        for f, t in seq:
            tiers[t] += 1
            visits[i][t] += 1
        longest_span = 0
        run_start = None
        prev = None
        for f, t in seq:
            if t == "bottom":
                if run_start is None:
                    run_start = f
                prev = f
            else:
                if run_start is not None and prev is not None:
                    longest_span = max(longest_span, prev - run_start)
                run_start = None
                prev = None
        if run_start is not None and prev is not None:
            longest_span = max(longest_span, prev - run_start)
        longest[i] = longest_span
    return {
        "tiers": dict(tiers),
        "visits": {i: dict(visits[i]) for i in visits},
        "longest_bottom_span_frames": longest,
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    now_c = curves(NOW)
    jul_c = curves(JUL)

    tip = []
    for i in range(6):
        ddir = NOW / "dexpbt_8gpu_ws" / f"{i:03d}"
        files = sorted(
            [p for p in ddir.glob("*.json") if p.name[0].isdigit()],
            key=lambda p: p.stat().st_mtime,
        )
        if not files:
            continue
        d = json.loads(files[-1].read_text())
        cur = d.get("curriculum") if isinstance(d.get("curriculum"), dict) else {}
        tip.append(
            {
                "p": i,
                "frame": int(d.get("frame") or 0),
                "nsucc": float(d.get("nsucc_mean") or 0),
                "nmax": float(d.get("nsucc_max") or 0),
                "tol": float(cur.get("success_tolerance") or d.get("success_tolerance") or 0),
            }
        )

    same = {}
    for fr in FRAMES:
        same[str(fr)] = {
            "NOW": pop(at(now_c, fr)),
            "JUL12": pop(at(jul_c, fr)),
        }

    # PBT ticks
    skip = Counter()
    should = 0
    total = 0
    per_should = Counter()
    for i in range(6):
        for d in load_jsonl(NOW / "dexpbt_8gpu_ws/metrics" / f"pbt_tick_debug_policy_{i:02d}.jsonl"):
            ef = int(d.get("env_frames") or d.get("population_frame") or 0)
            if ef > CAP:
                continue
            total += 1
            if d.get("should_tick"):
                should += 1
                per_should[i] += 1
                skip["TICK_LABEL_" + str(d.get("skip_reason") or "")] += 1
            else:
                skip[str(d.get("skip_reason") or "none")] += 1

    copy_events = []
    for e in load_jsonl(NOW / "dexpbt_8gpu_ws/metrics/pbt_copy_events.jsonl"):
        if int(e.get("env_frames") or 0) <= CAP:
            copy_events.append(e)

    rescue_side = {}
    for i in range(6):
        p = NOW / "dexpbt_8gpu_ws/adaptive_precision" / f"{i:03d}.json"
        if p.exists():
            d = json.loads(p.read_text())
            rescue_side[i] = {
                "rescue_executed": d.get("rescue_executed"),
                "rescue_attempt_count": d.get("rescue_attempt_count"),
                "rescue_selected_count": d.get("rescue_selected_count"),
                "rescue_blocked_count": d.get("rescue_blocked_count"),
                "rescue_last_reason": d.get("rescue_last_reason"),
                "probation_active": (d.get("probation") or {}).get("active"),
            }

    eir = [
        e
        for e in load_jsonl(NOW / "dexpbt_8gpu_ws/metrics/eir_broadcast_events.jsonl")
        if int(e.get("frame") or 0) <= 1_000_000_000
    ]

    # TB proxy from NOW itself
    tb = {"note": "from current NOW TensorBoard", "policies": {}}
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

        want_sub = [
            "elite_snippet/bank_size",
            "elite_snippet/bank_size_mid",
            "elite_snippet/bank_size_hard",
            "elite_snippet/bank_size_floor",
            "elite_snippet/retrieval_hit_rate",
            "progress_refine/active_frac",
            "progress_refine/effective_sample_size",
            "progress_refine/gate_mean",
            "progress_refine/history_valid_frac",
            "progress_refine/retrieval_valid_frac",
            "progress_refine/loss",
            "snippet_distill/active_frac",
            "snippet_distill/loss",
            "snippet_retrieval/no_compatible_frac",
            "snippet_wm_rerank/changed_frac",
            "snippet_wm_rerank/changed_frac_overall",
            "snippet_wm_rerank/fallback_frac",
            "snippet_wm_rerank/fallback_reason_id",
            "snippet_wm_rerank/row_active_frac",
            "snippet_wm_rerank/row_fallback_frac",
            "eir/actor_distill_active_fraction",
            "eir/actor_distill_loss",
            "losses/a_loss",
            "adaptive_consolidation/curriculum_progress",
        ]
        for i in range(6):
            evs = list((NOW / f"policy_{i:02d}").rglob("events.out.tfevents*"))
            if not evs:
                continue
            ea = EventAccumulator(str(evs[0].parent), size_guidance={"scalars": 20000})
            ea.Reload()
            tags = set(ea.Tags().get("scalars", []))
            series = {}
            for t in want_sub:
                if t not in tags:
                    continue
                sc = ea.Scalars(t)
                # restrict approx to <=1B: steps often = iterations; use last 30% as "~1B vicinity" if tip~1B
                vals = [(s.step, s.value) for s in sc]
                if not vals:
                    continue
                # use last 15% as late/~1B window for current mid-run
                late = vals[int(0.85 * len(vals)) :] or vals
                series[t] = {
                    "n": len(vals),
                    "last": vals[-1][1],
                    "late_mean": sum(v for _, v in late) / len(late),
                    "max": max(v for _, v in vals),
                    "nonzero_frac": sum(1 for _, v in vals if abs(v) > 1e-12) / len(vals),
                }
            tb["policies"][i] = {"series": series, "n_tags": len(tags)}
            if i >= 2:
                break
    except Exception as e:
        tb["error"] = str(e)

    payload = {
        "tip": tip,
        "same_frame": same,
        "counters_le_1B": {
            "NOW": shrinks_to(now_c, 1_000_000_000),
            "JUL12": shrinks_to(jul_c, 1_000_000_000),
        },
        "tiers_le_1B": {
            "NOW": tier_stats(now_c, 1_000_000_000),
            "JUL12": tier_stats(jul_c, 1_000_000_000),
        },
        "pbt_ticks_le_1.05B": {
            "tick_lines": total,
            "should_tick": should,
            "per_policy_should_tick": dict(per_should),
            "reasons": dict(skip.most_common()),
        },
        "copy_events_le_1.05B": copy_events,
        "rescue_side": rescue_side,
        "eir_broadcast_le_1B": {
            "counts": dict(Counter(e.get("event") for e in eir)),
            "n": len(eir),
            "entry_sizes": [e.get("entries") for e in eir],
        },
        "tb": tb,
    }
    (OUT / "audit_midrun_raw.json").write_text(json.dumps(payload, indent=2, default=str))

    # compact stdout
    print("TIP", tip)
    for fr in FRAMES:
        print("\nFRAME", fr)
        for name in ("NOW", "JUL12"):
            s = same[str(fr)][name]
            if not s:
                print(name, "MISSING")
                continue
            print(
                name,
                f"n={s['n']} best/med/mean={s['best']:.2f}/{s['med']:.2f}/{s['mean']:.2f}",
                f"tol={s['tol_min']:.4f}/{s['tol_med']:.4f}",
                "nsucc",
                s["per_nsucc"],
            )
    print("\nCOUNTERS NOW", json.dumps(payload["counters_le_1B"]["NOW"], indent=2))
    print("COUNTERS JUL", json.dumps(payload["counters_le_1B"]["JUL12"], indent=2))
    print("TIERS", json.dumps(payload["tiers_le_1B"], indent=2))
    print("PBT", json.dumps(payload["pbt_ticks_le_1.05B"], indent=2))
    print("COPY", len(copy_events), copy_events)
    print("RESCUE", json.dumps(rescue_side, indent=2))
    print("EIR", payload["eir_broadcast_le_1B"]["counts"], "n", payload["eir_broadcast_le_1B"]["n"])
    for i, pdata in tb.get("policies", {}).items():
        print(f"\nTB p{i}")
        for k, v in sorted(pdata.get("series", {}).items()):
            print(f"  {k}: late={v['late_mean']:.6g} last={v['last']:.6g} nz={v['nonzero_frac']:.3f}")


if __name__ == "__main__":
    main()
