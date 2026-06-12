#!/usr/bin/env python3
"""EIR-MPC run analyzer (Phase F, full).

Reads per-policy ``training_curve`` JSONL, TensorBoard ``eir/*`` / ``eir_mppi/*``
/ ``eir_pbt/*`` scalars and the ``eir_broadcast_events.jsonl`` audit log, then
emits a JSON + Markdown report covering:

  * performance: population mean/median/max, per-policy latest/peak nsucc,
    count of latest nsucc_mean >= 35, peak retention.
  * elite bank: size, add count, acceptance rate, priority, per-bucket counts.
  * candidates: candidate-source + best-source fractions, score margin (vs the
    ERD ~0.005 baseline), elite similarity / action delta.
  * residual: loss, gate, delta, apply fraction / active frames / fallback
    reason histogram.
  * RSSM rerank: active fraction, score margin, uncertainty, fallback histogram.
  * PBT broadcast: counts, source/target, entries, and post-broadcast 50M / 100M
    nsucc_mean improvement for broadcast targets.
  * failure classification + a continue / stop / patch recommendation.

Usage:
  python analyze_eir_mpc_run.py RUN_DIR [--out DIR] [--num-policies N] [--ws NAME]
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
from collections import defaultdict

EIR_SCALAR_PREFIXES = ("eir/", "eir_mppi/", "eir_pbt/")

RESIDUAL_APPLY_FALLBACK = {
    0: "ok", 1: "aux_missing", 2: "aux_stale", 3: "before_warmup",
    4: "kill_switch_off", 5: "gate_closed", 6: "not_floor_stage",
}
RSSM_FALLBACK = {
    0: "ok", 1: "disabled", 2: "before_start", 3: "model_unavailable",
    4: "no_heads", 5: "high_uncertainty",
}
BUCKET_NAMES = {
    0: "none/improve", 1: "none/success", 2: "floor/improve",
    3: "floor/success", 4: "hard/improve", 5: "hard/success",
}


def _read_jsonl(path):
    rows = []
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        rows.append(json.loads(line))
                    except Exception:
                        pass
    except Exception:
        pass
    return rows


def _safe(v):
    try:
        x = float(v)
        return None if (math.isnan(x) or math.isinf(x)) else x
    except Exception:
        return None


def _series_stats(xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return {"n": 0, "mean": None, "min": None, "max": None, "last": None, "median": None}
    s = sorted(xs)
    mid = s[len(s) // 2] if len(s) % 2 else 0.5 * (s[len(s) // 2 - 1] + s[len(s) // 2])
    return {
        "n": len(xs), "mean": sum(xs) / len(xs), "min": min(xs), "max": max(xs),
        "last": xs[-1], "median": mid,
    }


def _detect_ws(run_dir, ws_name):
    if ws_name:
        return os.path.join(run_dir, ws_name)
    # common default; else first dir containing a metrics/ folder
    cand = os.path.join(run_dir, "dexpbt_8gpu_ws")
    if os.path.isdir(os.path.join(cand, "metrics")):
        return cand
    for name in sorted(os.listdir(run_dir)) if os.path.isdir(run_dir) else []:
        p = os.path.join(run_dir, name)
        if os.path.isdir(os.path.join(p, "metrics")):
            return p
    return cand


def _detect_num_policies(metrics_dir):
    n = 0
    for pid in range(64):
        if os.path.exists(os.path.join(metrics_dir, f"training_curve_policy_{pid:02d}.jsonl")):
            n = pid + 1
    return max(n, 1)


def _collect_tb(run_dir, pid):
    """Return ``{tag: [(frame, value)]}`` for EIR scalars of one policy."""
    out = defaultdict(list)
    try:
        from tensorboard.backend.event_processing import event_accumulator
    except Exception:
        return out, False
    pattern = os.path.join(
        run_dir, f"policy_{pid:02d}", "rl_games", "attempt_*", "summaries",
        "events.out.tfevents.*",
    )
    for ev in sorted(glob.glob(pattern)):
        try:
            ea = event_accumulator.EventAccumulator(ev, size_guidance={"scalars": 0})
            ea.Reload()
            for tag in ea.Tags().get("scalars", []):
                if tag.startswith(EIR_SCALAR_PREFIXES):
                    for s in ea.Scalars(tag):
                        v = _safe(s.value)
                        if v is not None:
                            out[tag].append((int(s.step), v))
        except Exception:
            pass
    return out, True


def _hist_from_steps(steps_vals, names):
    hist = defaultdict(int)
    for _f, v in steps_vals:
        try:
            hist[names.get(int(round(v)), str(int(round(v))))] += 1
        except Exception:
            pass
    return dict(hist)


def _improvement_after(curve, frame, horizon):
    """nsucc_mean improvement from ``frame`` to ``frame+horizon`` for one curve
    of (env_frames, nsucc_mean) points. Returns None if insufficient coverage."""
    pts = [(f, v) for f, v in curve if f is not None and v is not None]
    if not pts:
        return None
    before = [v for f, v in pts if f <= frame]
    after = [v for f, v in pts if frame < f <= frame + horizon]
    if not before or not after:
        return None
    return (sum(after) / len(after)) - before[-1]


def analyze(run_dir, out_dir, ws_name=None, num_policies=None):
    ws = _detect_ws(run_dir, ws_name)
    metrics_dir = os.path.join(ws, "metrics")
    npol = num_policies or _detect_num_policies(metrics_dir)

    per_policy = {}
    pop_latest_mean = []
    pop_peak_mean = []
    curves = {}  # pid -> [(env_frames, nsucc_mean)]
    for pid in range(npol):
        jl = os.path.join(metrics_dir, f"training_curve_policy_{pid:02d}.jsonl")
        rows = _read_jsonl(jl)
        curve = [(_safe(r.get("env_frames")), _safe(r.get("nsucc_mean"))) for r in rows]
        curves[pid] = curve
        nsucc = [v for _f, v in curve if v is not None]
        tb, tb_ok = _collect_tb(run_dir, pid)
        key_tags = {tag: _series_stats([v for _f, v in xs]) for tag, xs in tb.items()}
        latest_mean = nsucc[-1] if nsucc else None
        peak_mean = max(nsucc) if nsucc else None
        if latest_mean is not None:
            pop_latest_mean.append(latest_mean)
        if peak_mean is not None:
            pop_peak_mean.append(peak_mean)
        retention = (latest_mean / peak_mean) if (peak_mean and peak_mean > 1e-9) else None
        # bucket counts (last value per bucket tag)
        buckets = {}
        for b, bname in BUCKET_NAMES.items():
            st = key_tags.get(f"eir/bank_elite_bucket_{b}")
            if st and st.get("last") is not None:
                buckets[bname] = st["last"]
        per_policy[f"policy_{pid:02d}"] = {
            "frames_end": curve[-1][0] if curve else None,
            "latest_nsucc_mean": latest_mean,
            "peak_nsucc_mean": peak_mean,
            "retention": retention,
            "residual_apply_fallback_hist": _hist_from_steps(
                tb.get("eir/residual_apply_fallback_reason", []), RESIDUAL_APPLY_FALLBACK
            ),
            "rssm_fallback_hist": _hist_from_steps(
                tb.get("eir_mppi/fallback_reason", []), RSSM_FALLBACK
            ),
            "bank_buckets": buckets,
            "eir_tags": key_tags,
            "tb_reader_available": tb_ok,
        }

    def _agg(tag, field="last"):
        vals = []
        for p in per_policy.values():
            st = p["eir_tags"].get(tag)
            if st and st.get(field) is not None:
                vals.append(st[field])
        return sum(vals) / len(vals) if vals else None

    # ---- broadcast events + post-broadcast improvement ----
    bc_events = _read_jsonl(os.path.join(metrics_dir, "eir_broadcast_events.jsonl"))
    broadcasts = [e for e in bc_events if e.get("event") == "broadcast"]
    consumes = [e for e in bc_events if e.get("event") == "consume" and e.get("consumed")]
    post_improve = []
    for e in consumes:
        tgt = e.get("target")
        frame = _safe(e.get("frame"))
        if tgt is None or frame is None or int(tgt) not in curves:
            continue
        post_improve.append({
            "target": int(tgt),
            "source": e.get("source"),
            "frame": int(frame),
            "improve_50m": _improvement_after(curves[int(tgt)], frame, 50_000_000),
            "improve_100m": _improvement_after(curves[int(tgt)], frame, 100_000_000),
        })
    imp50 = [d["improve_50m"] for d in post_improve if d["improve_50m"] is not None]
    imp100 = [d["improve_100m"] for d in post_improve if d["improve_100m"] is not None]

    latest_ge_35 = sum(1 for v in pop_latest_mean if v >= 35.0)
    summary = {
        "run_dir": run_dir, "workspace": ws, "num_policies": npol,
        # performance
        "population_latest_mean_nsucc": (sum(pop_latest_mean) / len(pop_latest_mean)) if pop_latest_mean else None,
        "population_peak_mean_nsucc": (sum(pop_peak_mean) / len(pop_peak_mean)) if pop_peak_mean else None,
        "population_latest_max_nsucc": max(pop_latest_mean) if pop_latest_mean else None,
        "count_latest_nsucc_ge_35": latest_ge_35,
        # bank
        "eir_active_mean": _agg("eir/active"),
        "elite_bank_size_mean": _agg("eir/elite_bank_size"),
        "elite_acceptance_rate_mean": _agg("eir/bank_elite_acceptance_rate"),
        "elite_priority_mean": _agg("eir/bank_elite_priority_mean", "mean"),
        "bank_feature_source_mean": _agg("eir/bank_feature_source", "mean"),
        # candidates
        "score_margin_median_of_means": _agg("eir/score_margin", "mean"),
        "best_source_frac_elite_mean": _agg("eir/best_source_frac_elite"),
        "best_source_frac_archive_mean": _agg("eir/best_source_frac_archive"),
        "best_source_frac_rssm_mean": _agg("eir/best_source_frac_rssm"),
        "best_source_frac_gaussian_mean": _agg("eir/best_source_frac_gaussian"),
        "best_source_frac_actor_mean": _agg("eir/best_source_frac_actor"),
        "elite_similarity_mean": _agg("eir/elite_similarity_mean", "mean"),
        # residual
        "residual_active_mean": _agg("eir/residual_active"),
        "residual_loss_mean": _agg("eir/residual_loss", "mean"),
        "residual_gate_mean": _agg("eir/residual_gate_mean", "mean"),
        "residual_apply_fraction_mean": _agg("eir/residual_apply_fraction", "mean"),
        "residual_apply_active_max": _agg("eir/residual_apply_active", "max"),
        # rssm
        "rssm_active_mean": _agg("eir_mppi/active"),
        "rssm_score_margin_mean": _agg("eir_mppi/score_margin", "mean"),
        "rssm_uncertainty_mean": _agg("eir_mppi/uncertainty", "mean"),
        # broadcast
        "broadcast_count_max": _agg("eir_pbt/broadcast_count", "max"),
        "bank_merge_count_max": _agg("eir_pbt/bank_merge_count", "max"),
        "broadcast_events": len(broadcasts),
        "broadcast_consumed_events": len(consumes),
        "broadcast_post_improve_50m_mean": (sum(imp50) / len(imp50)) if imp50 else None,
        "broadcast_post_improve_100m_mean": (sum(imp100) / len(imp100)) if imp100 else None,
    }

    elite_plus_archive = (summary["best_source_frac_elite_mean"] or 0.0) + (summary["best_source_frac_archive_mean"] or 0.0)
    criteria = {
        "best_source_elite_archive_gt_0.2": elite_plus_archive > 0.2,
        "score_margin_ge_0.02": (summary["score_margin_median_of_means"] or 0.0) >= 0.02,
        "residual_apply_observed": bool((summary["residual_apply_active_max"] or 0.0) > 0.0),
        "elite_bank_nonempty": bool((summary["elite_bank_size_mean"] or 0.0) > 0.0),
        "broadcast_fired": len(broadcasts) > 0,
        "broadcast_helped_targets": bool(summary["broadcast_post_improve_100m_mean"] and summary["broadcast_post_improve_100m_mean"] > 0.0),
        "latest_ge_35_count_increased": latest_ge_35 > 0,
    }
    summary["criteria"] = criteria

    # ---- failure classification + recommendation ----
    failures = []
    if not criteria["elite_bank_nonempty"]:
        failures.append("elite_bank_empty: no elite transitions harvested (check risky gate / tol staging)")
    if not criteria["best_source_elite_archive_gt_0.2"]:
        failures.append("low_elite_best_source: best action rarely from elite/archive (candidate margin too weak)")
    if not criteria["score_margin_ge_0.02"]:
        failures.append("low_score_margin: rerank margin still near ERD baseline (~0.005)")
    if (summary["eir_active_mean"] or 0.0) <= 0.0:
        failures.append("eir_inactive: aux loss never fired (warmup/valid-fraction/risky gate blocking)")
    if criteria["broadcast_fired"] and not criteria["broadcast_helped_targets"]:
        failures.append("broadcast_no_lift: payloads delivered but targets did not improve within 100M")

    if not failures and (criteria["best_source_elite_archive_gt_0.2"] and criteria["score_margin_ge_0.02"]):
        recommendation = "continue"
    elif (summary["eir_active_mean"] or 0.0) <= 0.0 or not criteria["elite_bank_nonempty"]:
        recommendation = "stop_and_patch"
    else:
        recommendation = "patch"
    summary["failures"] = failures
    summary["recommendation"] = recommendation

    os.makedirs(out_dir, exist_ok=True)
    report = {"summary": summary, "broadcast_post_improve": post_improve, "per_policy": per_policy}
    with open(os.path.join(out_dir, "EIR_MPC_RUN_REPORT.json"), "w") as f:
        json.dump(report, f, indent=2)

    lines = ["# EIR-MPC Run Report", "", f"- run_dir: `{run_dir}`",
             f"- workspace: `{ws}` ({npol} policies)", ""]
    lines.append(f"## Recommendation: **{recommendation}**")
    if failures:
        lines.append("")
        lines.append("### Failures")
        for fmsg in failures:
            lines.append(f"- {fmsg}")
    lines += ["", "## Summary"]
    for k, v in summary.items():
        if k in ("criteria", "failures", "recommendation"):
            continue
        lines.append(f"- {k}: {v}")
    lines += ["", "## Success criteria"]
    for k, v in criteria.items():
        lines.append(f"- {k}: {v}")
    if post_improve:
        lines += ["", "## Broadcast post-improvement"]
        for d in post_improve:
            lines.append(
                f"- target {d['target']} <- source {d['source']} @ {d['frame']}: "
                f"+50M={d['improve_50m']}, +100M={d['improve_100m']}"
            )
    with open(os.path.join(out_dir, "EIR_MPC_RUN_REPORT.md"), "w") as f:
        f.write("\n".join(lines))
    print(json.dumps(summary, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--out", default=None)
    ap.add_argument("--num-policies", type=int, default=None)
    ap.add_argument("--ws", default=None, help="workspace subdir name (auto-detected if omitted)")
    args = ap.parse_args()
    analyze(args.run_dir, args.out or args.run_dir, ws_name=args.ws, num_policies=args.num_policies)


if __name__ == "__main__":
    main()
