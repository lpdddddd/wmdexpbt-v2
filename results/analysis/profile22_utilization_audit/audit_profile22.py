#!/usr/bin/env python3
"""Read-only Profile 22 utilization audit for progress_refine_5b_20260712_114254."""
from __future__ import annotations

import json
import math
import statistics as st
from collections import Counter, defaultdict
from pathlib import Path

RUN = Path(
    "/root/autodl-tmp/wmdexpbt/results/runs/"
    "regrasping_cser_dexpbt_progress_refine_5b_20260712_114254"
)
PROXY_TB_RUN = Path(
    "/root/autodl-tmp/wmdexpbt/results/runs/"
    "regrasping_cser_dexpbt_progress_refine_5b_20260714_092846"
)
OUT = Path("/root/autodl-tmp/wmdexpbt/results/analysis/profile22_utilization_audit")
MET = RUN / "dexpbt_8gpu_ws" / "metrics"


def load_jsonl(path: Path):
    rows = []
    if not path.exists():
        return rows
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def finite(x):
    try:
        v = float(x)
        return math.isfinite(v)
    except Exception:
        return False


def audit_pbt():
    # tick debug
    skip = Counter()
    should = 0
    total = 0
    per_policy_ticks = Counter()
    for i in range(6):
        for d in load_jsonl(MET / f"pbt_tick_debug_policy_{i:02d}.jsonl"):
            total += 1
            if d.get("should_tick"):
                should += 1
                per_policy_ticks[i] += 1
            skip[str(d.get("skip_reason") or ("TICK" if d.get("should_tick") else "none"))] += 1

    # curves: tier occupancy, restarts, mutations
    tier_counts = Counter()
    bottom_rows = 0
    source_as = Counter()
    target_copy = Counter()
    policy_stats = {}
    for i in range(6):
        rows = load_jsonl(MET / f"training_curve_policy_{i:02d}.jsonl")
        tiers = Counter(r.get("tier") for r in rows)
        for t, c in tiers.items():
            tier_counts[t] += c
        bottom_rows += tiers.get("bottom", 0)
        last = rows[-1] if rows else {}
        # tol shrinks
        tols = [(int(r["env_frames"]), float(r["success_tolerance"]), float(r["nsucc_mean"]), float(r.get("nsucc_max") or 0)) for r in rows]
        shrinks = []
        for a, b in zip(tols, tols[1:]):
            if b[1] < a[1] - 1e-12:
                shrinks.append(
                    {
                        "from_frame": a[0],
                        "to_frame": b[0],
                        "tol_from": a[1],
                        "tol_to": b[1],
                        "factor": (b[1] / a[1]) if a[1] > 0 else None,
                        "nsucc_at_from": a[2],
                        "nmax_at_from": a[3],
                        "nsucc_at_to": b[2],
                    }
                )
        policy_stats[i] = {
            "n_curve_pts": len(rows),
            "last_frame": int(last.get("env_frames") or 0),
            "last_nsucc": float(last.get("nsucc_mean") or 0),
            "last_tol": float(last.get("success_tolerance") or 0),
            "tiers": dict(tiers),
            "shrink_count": len(shrinks),
            "shrinks": shrinks,
            "restart_count": int(last.get("restart_count") or 0),
            "weight_copy_count": int(last.get("weight_copy_count") or 0),
            "cross_restart_count": int(last.get("cross_restart_count") or 0),
            "self_restart_count": int(last.get("self_restart_count") or 0),
            "self_restart_blocked_count": int(last.get("self_restart_blocked_count") or 0),
            "mid_self_mutation_count": int(last.get("mid_self_mutation_count") or 0),
            "recovery_restart_count": int(last.get("recovery_restart_count") or 0),
            "rollback_count": int(last.get("rollback_count") or 0),
            "best_stable_recovery_executed": int(last.get("best_stable_recovery_executed_count") or 0),
            "source_tier_strict": int(last.get("source_tier_strict_live_count") or 0),
            "source_tier_relaxed": int(last.get("source_tier_relaxed_live_count") or 0),
            "source_tier_archive": int(last.get("source_tier_archive_count") or 0),
            "source_tier_skipped": int(last.get("source_tier_skipped_count") or 0),
            "wm_score_used_last": bool(last.get("wm_score_used")),
            "wm_score_effective_weight_last": float(last.get("wm_score_effective_weight") or 0),
            "wm_shadow_changed": int(last.get("wm_shadow_changed_selection") or 0),
            "wm_shadow_obs": int(last.get("wm_shadow_observations") or 0),
            "allow_mid_self_mutation": bool(last.get("allow_mid_self_mutation")),
            "disable_self_restart": bool(last.get("disable_self_restart")),
            "profile_actor_distill_enabled": bool(last.get("profile_actor_distill_enabled")),
            "profile_elite_bank_enabled": bool(last.get("profile_elite_bank_enabled")),
            "mutation_frozen_last": bool(last.get("mutation_frozen")),
            "pbt_wm_profile_copy_enable": bool(last.get("pbt_wm_profile_copy_enable")),
        }

    # copy events
    copy_events = load_jsonl(MET / "pbt_copy_events.jsonl")
    executed = []
    selected = []
    for e in copy_events:
        et = e.get("event_type") or e.get("tier") or "copy_record"
        if et == "adaptive_degradation_rescue_executed" or (
            e.get("tier") and e.get("source_policy_idx") is not None
        ):
            if et == "adaptive_degradation_rescue_executed":
                executed.append(e)
            elif "tier" in e and "source_policy_idx" in e:
                executed.append(e)  # inheritance record also counts as applied copy
        if et == "adaptive_degradation_rescue_selected":
            selected.append(e)
        if e.get("source_policy") is not None:
            source_as[int(e["source_policy"])] += 1
        if e.get("source_policy_idx") is not None:
            source_as[int(e["source_policy_idx"])] += 1
        if e.get("target_policy") is not None:
            target_copy[int(e["target_policy"])] += 1
        if e.get("policy_idx") is not None and et.endswith("executed"):
            target_copy[int(e["policy_idx"])] += 1

    # unique executed cross-copies by env_frames+target
    uniq_exec = {}
    for e in copy_events:
        if e.get("event_type") == "adaptive_degradation_rescue_executed":
            key = (e.get("env_frames"), e.get("target_policy"), e.get("source_policy"))
            uniq_exec[key] = e
        elif e.get("tier") and e.get("source_policy_idx") is not None:
            key = (e.get("env_frames"), e.get("policy_idx"), e.get("source_policy_idx"))
            uniq_exec.setdefault(("copy",) + key, e)

    # adaptive sidecars
    rescue_side = {}
    for i in range(6):
        p = RUN / "dexpbt_8gpu_ws" / "adaptive_precision" / f"{i:03d}.json"
        if p.exists():
            rescue_side[i] = json.loads(p.read_text())

    # eir broadcast = bank merge activity
    eir = load_jsonl(MET / "eir_broadcast_events.jsonl")
    eir_by = Counter(e.get("event") for e in eir)

    # bottom opportunities ≈ curve points where tier==bottom (40M cadence) — proxy
    # Better: each policy's should_tick while bottom. Not directly logged.
    # Use bottom_rows as opportunity proxy (population-level curve samples).
    n_executed_copies = len(
        [e for e in copy_events if e.get("event_type") == "adaptive_degradation_rescue_executed"]
    )
    # also count distinct inheritance records
    n_inheritance = len([e for e in copy_events if e.get("tier") and "source_policy_idx" in e])

    return {
        "tick_total_lines": total,
        "should_tick_count": should,
        "per_policy_should_ticks": dict(per_policy_ticks),
        "skip_reason": dict(skip.most_common()),
        "tier_counts_over_curve_pts": dict(tier_counts),
        "bottom_curve_pts": bottom_rows,
        "copy_events_raw": len(copy_events),
        "n_rescue_executed_events": n_executed_copies,
        "n_inheritance_records": n_inheritance,
        "n_rescue_selected": len(selected),
        "source_as_counts": dict(source_as),
        "target_copy_counts": dict(target_copy),
        "policy_stats": policy_stats,
        "rescue_side": {
            i: {
                "rescue_executed": s.get("rescue_executed"),
                "rescue_attempt_count": s.get("rescue_attempt_count"),
                "rescue_selected_count": s.get("rescue_selected_count"),
                "rescue_blocked_count": s.get("rescue_blocked_count"),
                "rescue_last_reason": s.get("rescue_last_reason"),
                "probation_active": (s.get("probation") or {}).get("active"),
            }
            for i, s in rescue_side.items()
        },
        "eir_broadcast": dict(eir_by),
        "eir_events": len(eir),
        "eir_entry_sizes": [e.get("entries") for e in eir if e.get("entries") is not None][:20],
        "copy_events_full": copy_events,
    }


def retention_after_shrink(policy_i: int, shrinks, rows, horizons=(50e6, 100e6, 150e6, 200e6)):
    """For each shrink, measure nsucc retention after horizons."""
    # index by frame
    by_f = {int(r["env_frames"]): float(r["nsucc_mean"]) for r in rows}
    frames = sorted(by_f)
    out = []
    for sh in shrinks:
        base_f = sh["to_frame"]
        base_n = by_f.get(base_f, sh["nsucc_at_to"])
        # peak before shrink in last 80M
        pre = [by_f[f] for f in frames if sh["from_frame"] - 80e6 <= f <= sh["from_frame"]]
        peak = max(pre) if pre else sh["nsucc_at_from"]
        rec = {"shrink_to_frame": base_f, "tol_to": sh["tol_to"], "nsucc_at_shrink": base_n, "peak_before": peak}
        for h in horizons:
            target = base_f + h
            # nearest
            cand = [f for f in frames if abs(f - target) <= 20e6]
            if not cand:
                rec[f"ret_at_{int(h/1e6)}M"] = None
                continue
            f = min(cand, key=lambda x: abs(x - target))
            n = by_f[f]
            rec[f"ret_at_{int(h/1e6)}M"] = (n / peak) if peak > 1e-6 else None
            rec[f"nsucc_at_{int(h/1e6)}M"] = n
        out.append(rec)
    return out


def audit_curriculum(pbt):
    curr = {}
    for i in range(6):
        rows = load_jsonl(MET / f"training_curve_policy_{i:02d}.jsonl")
        shrinks = pbt["policy_stats"][i]["shrinks"]
        # adaptive-fast factors from events when shrink happened
        factors = [s["factor"] for s in shrinks if s.get("factor")]
        # mastery: nsucc_mean>=4 and nmax>=20 at shrink time
        mastery_like = sum(
            1 for s in shrinks if s["nsucc_at_from"] >= 4.0 and s["nmax_at_from"] >= 20.0
        )
        # dwell time per tol (quantize)
        dwell = Counter()
        for a, b in zip(rows, rows[1:] + [rows[-1]]):
            tol = round(float(a["success_tolerance"]), 4)
            dt = max(0, int(b["env_frames"]) - int(a["env_frames"]))
            dwell[tol] += dt
        # collapse/rollback markers
        rollbacks = sum(1 for r in rows if r.get("curriculum_rollback_triggered") or r.get("hard_collapse_triggered"))
        hard = sum(1 for r in rows if r.get("hard_collapse_triggered"))
        # post-shrink retention
        rets = retention_after_shrink(i, shrinks, rows)
        # mean retention at 100M
        r100 = [r["ret_at_100M"] for r in rets if r.get("ret_at_100M") is not None]
        # shrink-collapse cycle heuristic: within 200M after shrink, nsucc < 0.5 * peak
        cycles = 0
        for r in rets:
            vals = [r.get(f"ret_at_{h}M") for h in (50, 100, 150, 200)]
            vals = [v for v in vals if v is not None]
            if vals and min(vals) < 0.5:
                cycles += 1
        curr[i] = {
            "shrink_count": len(shrinks),
            "shrink_factors": factors,
            "unique_factors": sorted({round(f, 4) for f in factors if f}),
            "mastery_like_shrinks": mastery_like,
            "dwell_tol_frames": {str(k): v for k, v in sorted(dwell.items(), key=lambda x: -x[1])[:12]},
            "hard_collapse_curve_flags": hard,
            "rollback_triggered_flags": rollbacks,
            "post_shrink_retention": rets,
            "mean_ret_100M": (sum(r100) / len(r100)) if r100 else None,
            "shrink_then_drop_cycles": cycles,
            "tol_trajectory": [(int(r["env_frames"]), float(r["success_tolerance"]), float(r["nsucc_mean"])) for r in rows[:: max(1, len(rows)//20)]],
        }
    return curr


def audit_wm_from_curves(pbt):
    out = {}
    for i in range(6):
        rows = load_jsonl(MET / f"training_curve_policy_{i:02d}.jsonl")
        used = sum(1 for r in rows if r.get("wm_score_used"))
        valid = sum(1 for r in rows if r.get("wm_imagined_valid"))
        eff_w = [float(r.get("wm_score_effective_weight") or 0) for r in rows]
        shadow_chg = int(rows[-1].get("wm_shadow_changed_selection") or 0) if rows else 0
        shadow_obs = int(rows[-1].get("wm_shadow_observations") or 0) if rows else 0
        out[i] = {
            "curve_pts": len(rows),
            "wm_score_used_pts": used,
            "wm_imagined_valid_pts": valid,
            "wm_score_used_frac": used / len(rows) if rows else 0,
            "mean_eff_weight": sum(eff_w) / len(eff_w) if eff_w else 0,
            "zero_eff_weight_frac": sum(1 for w in eff_w if abs(w) < 1e-12) / len(eff_w) if eff_w else 1,
            "shadow_changed_final": shadow_chg,
            "shadow_obs_final": shadow_obs,
            "shadow_change_rate": (shadow_chg / shadow_obs) if shadow_obs else None,
        }
    return out


def sample_tb_proxy():
    """Sample TB from Jul14 same-profile run; Jul12 TB deleted."""
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except Exception as e:
        return {"error": str(e)}

    tags_of_interest = [
        "eir/snippet_distill_loss",
        "eir/snippet_distill_weighted_loss",
        "eir/snippet_distill_active_fraction",
        "eir/snippet_distill_nonzero_frac",
        "eir/snippet_progress_gate_mean",
        "eir/snippet_progress_gate_active_frac",
        "eir/snippet_ess",
        "eir/snippet_retrieval_valid_frac",
        "eir/snippet_retrieval_attempted",
        "eir/snippet_bank_size",
        "eir/bank_elite_bank_size",
        "eir/actor_distill_loss",
        "eir/actor_distill_weighted_loss",
        "eir/actor_distill_active_fraction",
        "eir/actor_distill_fallback_with_snippet_used",
        "losses/a_loss",
        "losses/actor_loss",
        "eir/wm_alpha",
        "eir/snippet_wm_alpha",
        "eir/snippet_alpha_wm",
        "eir/wm_calibration_pass",
        "eir/snippet_wm_rerank_changed_frac",
        "eir/snippet_teacher_accepted_frac",
        "eir/snippet_no_compatible_frac",
        "progress_refinement/gate_mean",
        "progress_refinement/active_fraction",
        "progress_refinement/ess",
    ]
    # discover actual tags on one policy
    results = {"note": "Jul12 TB deleted; proxy from Jul14 same Profile 22 run", "policies": {}}
    for i in range(6):
        evs = list((PROXY_TB_RUN / f"policy_{i:02d}").rglob("events.out.tfevents*"))
        if not evs:
            continue
        ea = EventAccumulator(str(evs[0].parent), size_guidance={"scalars": 10000})
        ea.Reload()
        tags = set(ea.Tags().get("scalars", []))
        # fuzzy match
        matched = {}
        for want in tags_of_interest:
            if want in tags:
                matched[want] = want
            else:
                # find similar
                cands = [t for t in tags if want.split("/")[-1] in t]
                if cands:
                    matched[want] = sorted(cands)[0]
        # also collect all snippet/wm related
        snip = sorted(
            [
                t
                for t in tags
                if any(
                    s in t.lower()
                    for s in (
                        "snippet",
                        "progress_ref",
                        "bank_elite",
                        "alpha_wm",
                        "wm_calib",
                        "actor_distill",
                        "a_loss",
                        "actor_loss",
                    )
                )
            ]
        )
        prefer = [
            t
            for t in snip
            if any(
                s in t.lower()
                for s in (
                    "snippet_distill",
                    "progress_gate",
                    "ess",
                    "retrieval",
                    "alpha",
                    "bank_size",
                    "teacher",
                    "compatible",
                    "rerank",
                    "actor_distill_loss",
                    "actor_distill_active",
                    "a_loss",
                    "calib",
                )
            )
        ]
        series = {}
        for logical, actual in matched.items():
            try:
                sc = ea.Scalars(actual)
                if not sc:
                    continue
                vals = [s.value for s in sc]
                late = vals[int(0.8 * len(vals)) :] or vals
                series[logical] = {
                    "actual_tag": actual,
                    "n": len(vals),
                    "mean_all": sum(vals) / len(vals),
                    "mean_late": sum(late) / len(late),
                    "max": max(vals),
                    "last": vals[-1],
                    "nonzero_frac": sum(1 for v in vals if abs(v) > 1e-12) / len(vals),
                }
            except Exception as e:
                series[logical] = {"error": str(e), "actual_tag": actual}
        for t in prefer[:50]:
            if any(series.get(k, {}).get("actual_tag") == t for k in series):
                continue
            try:
                sc = ea.Scalars(t)
                vals = [s.value for s in sc]
                late = vals[int(0.8 * len(vals)) :] or vals
                series[t] = {
                    "actual_tag": t,
                    "n": len(vals),
                    "mean_all": sum(vals) / len(vals),
                    "mean_late": sum(late) / len(late),
                    "max": max(vals),
                    "last": vals[-1],
                    "nonzero_frac": sum(1 for v in vals if abs(v) > 1e-12) / len(vals),
                }
            except Exception:
                pass
        results["policies"][i] = {
            "n_tags": len(tags),
            "snippet_related_tags": snip[:100],
            "prefer_tags": prefer[:60],
            "series": series,
        }
        if i >= 1:  # deep-parse p0 and p1 only
            break
    return results


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    pbt = audit_pbt()
    curr = audit_curriculum(pbt)
    wm = audit_wm_from_curves(pbt)
    tb = sample_tb_proxy()

    # rates
    bottom_opp = pbt["bottom_curve_pts"]  # proxy
    executed = pbt["n_rescue_executed_events"]
    # Better opportunities: sum of should_ticks / 6 * (bottom share)? Use bottom share of tiers
    tier_total = sum(pbt["tier_counts_over_curve_pts"].values()) or 1
    bottom_share = pbt["tier_counts_over_curve_pts"].get("bottom", 0) / tier_total
    estimated_bottom_ticks = int(round(pbt["should_tick_count"] * bottom_share))
    copy_rate_proxy = executed / estimated_bottom_ticks if estimated_bottom_ticks else None

    summary = {
        "run": str(RUN),
        "pbt": pbt,
        "curriculum": curr,
        "wm_curves": wm,
        "tb_proxy": tb,
        "derived": {
            "executed_cross_copies": executed,
            "bottom_curve_pts": bottom_opp,
            "estimated_bottom_ticks": estimated_bottom_ticks,
            "copy_execution_rate_proxy": copy_rate_proxy,
            "bottom_tier_share": bottom_share,
            "sum_self_restarts": sum(p["self_restart_count"] for p in pbt["policy_stats"].values()),
            "sum_self_restart_blocked": sum(p["self_restart_blocked_count"] for p in pbt["policy_stats"].values()),
            "sum_mid_mutation_attempts": sum(p["mid_self_mutation_count"] for p in pbt["policy_stats"].values()),
            "sum_weight_copies": sum(p["weight_copy_count"] for p in pbt["policy_stats"].values()),
            "sum_rollbacks": sum(p["rollback_count"] for p in pbt["policy_stats"].values()),
        },
    }
    # strip huge shrink lists detail for json size? keep
    (OUT / "audit_raw.json").write_text(json.dumps(summary, indent=2, default=str))
    print(json.dumps({
        "derived": summary["derived"],
        "skip_reason": pbt["skip_reason"],
        "tiers": pbt["tier_counts_over_curve_pts"],
        "rescue_side": pbt["rescue_side"],
        "eir_broadcast": pbt["eir_broadcast"],
        "per_policy": {i: {k: v for k, v in s.items() if k != "shrinks"} for i, s in pbt["policy_stats"].items()},
        "tb_policies": list((tb.get("policies") or {}).keys()),
        "tb_p0_series_keys": list((((tb.get("policies") or {}).get(0) or {}).get("series") or {}).keys())[:40],
    }, indent=2, default=str))


if __name__ == "__main__":
    main()
