#!/usr/bin/env python3
"""Readonly root-cause audit for Profile 22 run 20260714_092846 (p2 focus)."""
from __future__ import annotations

import csv
import json
import math
import statistics as st
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

RUN = Path(
    "/root/autodl-tmp/wmdexpbt/results/runs/"
    "regrasping_cser_dexpbt_progress_refine_5b_20260714_092846"
)
OUT = Path("/root/autodl-tmp/wmdexpbt/results/analysis/profile22_root_cause")
WS = RUN / "dexpbt_8gpu_ws"
METRICS = WS / "metrics"
INTERVAL = 20_000_000
SKILL_DELTA_WINDOW = 6
SKILL_DELTA_REF = 5.0
SKILL_DELTA_IMPROVE_EPS = 0.05
PBT_STUCK_NSUCC_MEAN = 1.0
LOG_GAP = 0.20
EPS = 1e-8
N_TOP, N_MID, N_BOT = 1, 3, 2  # n=6


def load_jsonl(p: Path) -> list[dict[str, Any]]:
    if not p.exists():
        return []
    return [json.loads(l) for l in p.open() if l.strip()]


def compute_skill_delta(samples: list[dict[str, Any]]) -> tuple[float | None, str]:
    if len(samples) < SKILL_DELTA_WINDOW:
        return None, "short_window"
    xs = samples[-SKILL_DELTA_WINDOW:]
    lineage_ids = {s.get("lineage_id") for s in xs}
    if None in lineage_ids or len(lineage_ids) != 1:
        return None, f"lineage_invalid:{lineage_ids}"
    tol_bins = {round(float(s["success_tolerance"]), 6) for s in xs}
    if len(tol_bins) != 1:
        return None, f"tol_bins:{sorted(tol_bins)}"
    pf = np.array([float(s["env_frames"]) for s in xs], dtype=np.float64)
    y = np.array([float(s["nsucc_mean"]) for s in xs], dtype=np.float64)
    alpha = 2.0 / (len(xs) + 1.0)
    ema = np.empty_like(y)
    ema[0] = y[0]
    for i in range(1, len(y)):
        ema[i] = alpha * y[i] + (1.0 - alpha) * ema[i - 1]
    pf_var = float(np.var(pf))
    if pf_var <= 1e-12:
        return None, "pf_var0"
    slope = float(np.cov(pf, ema, bias=True)[0, 1] / pf_var)
    raw = slope * float(INTERVAL)
    return float(max(-1.0, min(1.0, raw / SKILL_DELTA_REF))), "ok"


def target_gate(nsucc: float, delta: float | None, delta_why: str) -> tuple[bool, str]:
    if delta is None:
        return False, f"skill_delta_invalid:{delta_why}"
    if float(delta) >= SKILL_DELTA_IMPROVE_EPS:
        return False, "skill_delta_improving"
    if not (nsucc < PBT_STUCK_NSUCC_MEAN and float(delta) <= 0.0):
        return False, "not_stuck"
    return True, "ok"


def tier_of(rank: int) -> str:
    if rank < N_TOP:
        return "top"
    if rank < N_TOP + N_MID:
        return "mid"
    return "bottom"


def load_curves() -> dict[int, list[dict[str, Any]]]:
    out: dict[int, list[dict[str, Any]]] = {}
    for i in range(6):
        out[i] = load_jsonl(METRICS / f"training_curve_policy_{i:02d}.jsonl")
    return out


def load_workspace_series(policy_idx: int) -> list[dict[str, Any]]:
    d = WS / f"{policy_idx:03d}"
    rows = []
    for p in sorted(d.glob("*.json")):
        if "staging" in p.name or p.name.endswith(".tmp"):
            continue
        try:
            data = json.loads(p.read_text())
        except Exception:
            continue
        if not isinstance(data, dict) or data.get("frame") is None:
            continue
        cur = data.get("curriculum") or {}
        rows.append(
            {
                "env_frames": int(data["frame"]),
                "nsucc_mean": float(data.get("nsucc_mean") or 0.0),
                "nsucc_max": float(data.get("nsucc_max") or 0.0),
                "success_tolerance": float(cur.get("success_tolerance") or 0.075),
                "lineage_id": data.get("lineage_id"),
                "r_meta": data.get("r_meta"),
                "json_name": p.name,
            }
        )
    return rows


def nearest_row(rows: list[dict[str, Any]], frame: int) -> dict[str, Any] | None:
    if not rows:
        return None
    return min(rows, key=lambda r: abs(int(r["env_frames"]) - int(frame)))


def rank_population(curves: dict[int, list[dict[str, Any]]], frame: int) -> list[int]:
    scores = {}
    for i, rows in curves.items():
        r = nearest_row(rows, frame)
        if r is None:
            continue
        # Prefer logged ranking score when same-frame exact exists.
        exact = next((x for x in rows if int(x["env_frames"]) == int(frame)), None)
        src = exact or r
        score = src.get("pbt_ranking_score")
        if score is None or not math.isfinite(float(score)):
            # success_aware_r_meta proxy: prefer r_meta then nsucc
            rm = src.get("r_meta")
            nm = float(src.get("nsucc_mean") or 0.0)
            score = float(rm) if rm is not None and math.isfinite(float(rm)) else nm
        scores[i] = float(score)
    return sorted(scores.keys(), key=lambda k: scores[k], reverse=True)


def build_p2_tick_replay(curves: dict[int, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    p2_json = load_workspace_series(2)
    apc = json.loads((WS / "adaptive_precision/002.json").read_text())
    lineage = apc.get("adaptive_lineage_id") or (p2_json[0].get("lineage_id") if p2_json else None)

    ticks = load_jsonl(METRICS / "pbt_tick_debug_policy_02.jsonl")
    should = [t for t in ticks if t.get("should_tick")]
    hist: list[dict[str, Any]] = []
    # Grow history from workspace JSON (observed commit samples).
    json_by_frame = {int(r["env_frames"]): r for r in p2_json}

    # First continuous bottom span from curve.
    curve2 = curves[2]
    first_bottom = next(
        (int(r["env_frames"]) for r in curve2 if r.get("tier") == "bottom"), None
    )
    # longest bottom span end: last consecutive bottom from first_bottom tip
    bottom_end = None
    if first_bottom is not None:
        in_span = False
        start = first_bottom
        for r in curve2:
            f = int(r["env_frames"])
            if f < first_bottom:
                continue
            if r.get("tier") == "bottom":
                if not in_span:
                    start = f
                    in_span = True
                bottom_end = f
            else:
                if in_span:
                    break

    rows_out: list[dict[str, Any]] = []
    for t in should:
        ef = int(t["env_frames"])
        # Match nearest curve / json for state (reconstructed alignment).
        c2 = nearest_row(curve2, ef)
        j2 = nearest_row(p2_json, ef)
        if j2 is not None and int(j2["env_frames"]) not in {int(s["env_frames"]) for s in hist}:
            hist.append(
                {
                    "env_frames": int(j2["env_frames"]),
                    "nsucc_mean": float(j2["nsucc_mean"]),
                    "success_tolerance": float(j2["success_tolerance"]),
                    "lineage_id": j2.get("lineage_id") or lineage,
                }
            )
        elif c2 is not None and int(c2["env_frames"]) not in {int(s["env_frames"]) for s in hist}:
            hist.append(
                {
                    "env_frames": int(c2["env_frames"]),
                    "nsucc_mean": float(c2["nsucc_mean"]),
                    "success_tolerance": float(c2["success_tolerance"]),
                    "lineage_id": lineage,
                }
            )

        sorted_pols = rank_population(curves, ef)
        rank = sorted_pols.index(2) if 2 in sorted_pols else -1
        tier_recon = tier_of(rank) if rank >= 0 else "unknown"
        tier_obs = (c2 or {}).get("tier") if c2 else None
        nsucc = float((j2 or c2 or {}).get("nsucc_mean") or 0.0)
        nmax = float((j2 or c2 or {}).get("nsucc_max") or 0.0)
        tol = float((j2 or c2 or {}).get("success_tolerance") or 0.075)
        peak = float((c2 or {}).get("curriculum_recent_peak_nsucc_mean_long") or 0.0)
        gret = (c2 or {}).get("curriculum_global_retention")
        prom = (c2 or {}).get("protected_by_promising_gate_now")
        prom_until = (c2 or {}).get("promising_protect_until")
        progress = (c2 or {}).get("curriculum_progress")
        if progress is None and peak and tol:
            # not observable exact curriculum_progress on all rows
            progress = None

        delta, delta_why = compute_skill_delta(hist)
        allowed, gate_reason = target_gate(nsucc, delta, delta_why)

        # Exploration stage via curriculum_progress threshold 0.5 (observed TB ~0.3)
        # Use APC / curve proxy: Progress Refine schedule mode curriculum_progress.
        stage = "exploration"  # reconstructed default unless progress>=0.5 observed
        if isinstance(progress, (int, float)) and float(progress) >= 0.5:
            stage = "consolidation_or_later"

        # Source candidates from population at same aligned frame (reconstructed)
        sources = []
        for pid in sorted_pols:
            if pid == 2:
                continue
            pr = nearest_row(curves[pid], ef)
            if pr is None:
                continue
            s_ns = float(pr.get("nsucc_mean") or 0.0)
            s_nm = float(pr.get("nsucc_max") or 0.0)
            eligible = s_ns >= 3.0  # stable_source_min_mean reconstructed
            # live source also needs skill_delta >= 0.02 — not fully observable without that worker's window
            sources.append(
                {
                    "policy": pid,
                    "nsucc_mean": s_ns,
                    "nsucc_max": s_nm,
                    "tier": tier_of(sorted_pols.index(pid)),
                    "stable_mean_proxy_ok": eligible,
                    "promising": bool(pr.get("protected_by_promising_gate_now")),
                }
            )

        in_bottom_span = (
            first_bottom is not None
            and bottom_end is not None
            and first_bottom <= ef <= bottom_end + INTERVAL
            and (tier_obs == "bottom" or tier_recon == "bottom")
        )

        final_decision = "n/a_not_bottom"
        source_cascade_called = False
        if tier_obs == "bottom" or tier_recon == "bottom":
            if prom:
                final_decision = "early_return:promising_protection"
            elif stage != "exploration":
                # consolidation gate — exploratory schedule allows
                final_decision = "check_consolidation_then_target_gate"
            elif not allowed:
                final_decision = f"early_return:target_gate:{gate_reason}"
                source_cascade_called = False
            else:
                final_decision = "would_enter_source_selection"
                source_cascade_called = True

        evidence = {
            "tier": "observed" if tier_obs is not None else "reconstructed",
            "skill_delta": "reconstructed",
            "target_gate": "reconstructed",
            "should_tick": "observed",
            "promising": "observed" if prom is not None else "not_observable",
            "source_eligibility_skill_delta": "not_observable",
            "restart": "observed_counters",
        }

        rows_out.append(
            {
                "env_frames": ef,
                "population_frame": t.get("population_frame"),
                "pbt_iteration": t.get("self_pbt_iteration"),
                "should_tick": True,
                "rank_reconstructed": rank,
                "tier_observed": tier_obs,
                "tier_reconstructed": tier_recon,
                "in_bottom_span": in_bottom_span,
                "nsucc_mean": nsucc,
                "nsucc_max": nmax,
                "recent_peak": peak,
                "global_retention": gret,
                "success_tolerance": tol,
                "curriculum_progress": progress,
                "consolidation_stage_recon": stage,
                "protected_by_promising_observed": prom,
                "promising_protect_until": prom_until,
                "skill_delta_recon": delta,
                "skill_delta_invalid_reason": delta_why if delta is None else "",
                "target_gate_allowed_recon": allowed,
                "target_gate_reason_recon": gate_reason,
                "ordinary_source_cascade_called_recon": source_cascade_called,
                "candidate_sources_proxy": json.dumps(sources),
                "n_stable_mean_proxy_sources": sum(
                    1 for s in sources if s["stable_mean_proxy_ok"]
                ),
                "final_decision_recon": final_decision,
                "restart_attempted_observed": 0,
                "restart_committed_observed": 0,
                "mutation_applied_observed": 0,
                "weight_copy_count_curve": (c2 or {}).get("weight_copy_count"),
                "source_tier_skipped_count": (c2 or {}).get("source_tier_skipped_count"),
                "evidence": json.dumps(evidence),
                "bottom_span_start": first_bottom,
                "bottom_span_end": bottom_end,
            }
        )
    return rows_out


def audit_bank() -> dict[str, Any]:
    import torch

    cks = sorted((WS / "002").glob("*.pth"))
    latest = cks[-1]
    obj = torch.load(latest, map_location="cpu", weights_only=False)
    b = obj["eir_elite_snippet_bank"]
    size = int(b["size"])
    tol = b["success_tol"][:size].numpy()
    src = b["source_policy_idx"][:size].numpy().astype(int)
    pri = b["priority_score"][:size].numpy()
    bucket = b["bucket_id"][:size].numpy()
    frame = b["frame"][:size].numpy()
    nerr = b["normalized_error"][:size].numpy()
    nimp = b["normalized_improvement"][:size].numpy()
    succ = b["success_h"][:size].numpy()
    rec = b["recovery_label"][:size].numpy()
    drop = b["drop_h"][:size].numpy()

    # query tol from latest curve tip
    curve2 = load_jsonl(METRICS / "training_curve_policy_02.jsonl")
    tip = curve2[-1]
    qtol = float(tip["success_tolerance"])
    delta = np.abs(np.log((tol + EPS) / (qtol + EPS)))
    compat = delta <= LOG_GAP

    # tolerance hist
    tol_hist_path = OUT / "p2_bank_tolerance_hist.csv"
    # round bins
    rounded = np.round(tol, 6)
    uniq, cnts = np.unique(rounded, return_counts=True)
    with tol_hist_path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(
            [
                "success_tolerance",
                "count",
                "frac",
                "delta_eps_vs_query",
                "compat_gap_le_0.20",
            ]
        )
        for u, c in zip(uniq, cnts):
            dlt = abs(math.log((float(u) + EPS) / (qtol + EPS)))
            w.writerow([float(u), int(c), float(c) / size, dlt, int(dlt <= LOG_GAP)])

    src_hist_path = OUT / "p2_bank_source_hist.csv"
    with src_hist_path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(
            [
                "source_policy",
                "count",
                "frac",
                "tol_min",
                "tol_med",
                "tol_max",
                "local_vs_broadcast",
            ]
        )
        for s in sorted(set(src.tolist())):
            m = src == s
            provenance = "local" if int(s) == 2 else "broadcast_or_other"
            w.writerow(
                [
                    int(s),
                    int(m.sum()),
                    float(m.mean()),
                    float(tol[m].min()),
                    float(np.median(tol[m])),
                    float(tol[m].max()),
                    provenance,
                ]
            )

    # priority top slices
    order = np.argsort(-pri)
    def slice_stats(frac: float) -> dict[str, Any]:
        k = max(1, int(round(size * frac)))
        idx = order[:k]
        return {
            "k": k,
            "tol_med": float(np.median(tol[idx])),
            "tol_min": float(tol[idx].min()),
            "tol_max": float(tol[idx].max()),
            "source_counts": {int(a): int(b) for a, b in zip(*np.unique(src[idx], return_counts=True))},
        }

    # bucket x tol crosstab
    cross = Counter()
    for bu, to in zip(bucket, rounded):
        cross[(float(bu), float(to))] += 1

    eir_events = load_jsonl(METRICS / "eir_broadcast_events.jsonl")
    bc_to_p2 = [
        e
        for e in eir_events
        if int(e.get("target_policy", e.get("target", -1))) == 2
        or str(e.get("event", "")).lower().find("broadcast") >= 0
    ]

    return {
        "checkpoint": str(latest),
        "bank_size": size,
        "schema_version": int(b.get("schema_version", -1)),
        "query_tolerance_tip": qtol,
        "query_frame": int(tip["env_frames"]),
        "tol_min": float(tol.min()),
        "tol_med": float(np.median(tol)),
        "tol_max": float(tol.max()),
        "unique_tol": int(len(uniq)),
        "source_hist": {int(a): int(b) for a, b in zip(*np.unique(src, return_counts=True))},
        "local_count": int((src == 2).sum()),
        "broadcast_like_count": int((src != 2).sum()),
        "compat_frac": float(compat.mean()),
        "compat_count": int(compat.sum()),
        "delta_eps_med": float(np.median(delta)),
        "delta_eps_min": float(delta.min()),
        "delta_eps_max": float(delta.max()),
        "priority_top_1pct": slice_stats(0.01),
        "priority_top_5pct": slice_stats(0.05),
        "priority_top_10pct": slice_stats(0.10),
        "bucket_tol_cross_top": [
            {"bucket_id": a, "tol": b, "count": c}
            for (a, b), c in cross.most_common(20)
        ],
        "normalized_error_med": float(np.median(nerr)),
        "success_h_mean": float(succ.mean()),
        "recovery_label_mean": float(rec.mean()),
        "drop_h_mean": float(drop.mean()),
        "insert_frame_min": float(frame.min()),
        "insert_frame_max": float(frame.max()),
        "eir_broadcast_events_n": len(eir_events),
        "eir_events_mentioning_p2_n": len(bc_to_p2),
        "hypothesis": "broadcast_low_tol_fill",
    }


def summarize_replay(rows: list[dict[str, Any]]) -> dict[str, Any]:
    bottom_opps = [
        r
        for r in rows
        if r.get("in_bottom_span")
        and (r.get("tier_observed") == "bottom" or r.get("tier_reconstructed") == "bottom")
    ]
    reasons = Counter(r["target_gate_reason_recon"] for r in bottom_opps)
    finals = Counter(r["final_decision_recon"] for r in bottom_opps)
    cascade = sum(1 for r in bottom_opps if r["ordinary_source_cascade_called_recon"])
    return {
        "should_tick_total": len(rows),
        "bottom_opportunities": len(bottom_opps),
        "bottom_span_start": bottom_opps[0]["bottom_span_start"] if bottom_opps else None,
        "bottom_span_end": bottom_opps[0]["bottom_span_end"] if bottom_opps else None,
        "target_gate_reason_counts": dict(reasons),
        "final_decision_counts": dict(finals),
        "source_cascade_entered": cascade,
        "any_restart": any(int(r.get("restart_committed_observed") or 0) for r in rows),
    }


def maybe_tb_p2() -> dict[str, Any]:
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except Exception as e:
        return {"error": str(e)}
    tbs = list((RUN / "policy_02").rglob("events.out.tfevents*"))
    if not tbs:
        return {"error": "no_tb"}
    ea = EventAccumulator(str(tbs[0].parent), size_guidance={"scalars": 0})
    ea.Reload()
    tags = set(ea.Tags().get("scalars", []))
    out: dict[str, Any] = {"tags_present": sorted(t for t in tags if any(k in t for k in ("skill_delta", "snippet", "progress_refine", "belief", "alpha")))}

    def series(tag: str) -> list[tuple[int, float]]:
        if tag not in tags:
            return []
        return [(int(e.step), float(e.value)) for e in ea.Scalars(tag)]

    for tag in [
        "pbt/skill_delta",
        "pbt/skill_delta_valid",
        "pbt/skill_delta_copy_allowed",
        "snippet_retrieval/no_compatible_frac",
        "snippet_wm_rerank/fallback_reason_id",
        "snippet_wm_rerank/changed_frac",
        "progress_refine/active_frac",
        "progress_refine/effective_sample_size",
        "eir/mppi_belief_valid",
    ]:
        s = series(tag)
        if not s:
            out[tag] = None
            continue
        vals = [v for _, v in s]
        out[tag] = {
            "n": len(s),
            "first": s[0],
            "last": s[-1],
            "mean": float(sum(vals) / len(vals)),
            "allowed_or_nonzero": sum(1 for v in vals if v > 0.5),
            "points": s if tag.startswith("pbt/") else s[:: max(1, len(s) // 20)],
        }
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    curves = load_curves()
    replay = build_p2_tick_replay(curves)
    csv_path = OUT / "p2_pbt_tick_replay.csv"
    if replay:
        with csv_path.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(replay[0].keys()))
            w.writeheader()
            for r in replay:
                w.writerow(r)

    bank = audit_bank()
    summary = summarize_replay(replay)
    tip = {i: curves[i][-1] for i in range(6) if curves[i]}
    apc = json.loads((WS / "adaptive_precision/002.json").read_text())

    # Adaptive rescue observability
    rescue = {
        "rescue_attempt_count": apc.get("rescue_attempt_count"),
        "rescue_selected_count": apc.get("rescue_selected_count"),
        "rescue_blocked_count": apc.get("rescue_blocked_count"),
        "rescue_executed": apc.get("rescue_executed"),
        "rescue_last_reason": apc.get("rescue_last_reason"),
        "degradation_streak": apc.get("degradation_streak"),
        "observed": True,
    }

    tb = maybe_tb_p2()

    raw = {
        "run": str(RUN),
        "tip_nsucc": {i: float(tip[i]["nsucc_mean"]) for i in tip},
        "tip_tier": {i: tip[i].get("tier") for i in tip},
        "tip_tol": {i: float(tip[i]["success_tolerance"]) for i in tip},
        "pbt_replay_summary": summary,
        "adaptive_rescue_sidecar": rescue,
        "bank": bank,
        "tb_p2": tb,
        "code_refs": {
            "ordinary_target_gate": "pbt_control_plane.py:pbt_copy_allowed_for_target",
            "stuck_def": "nsucc_mean < 1.0 AND skill_delta <= 0",
            "skill_delta": "pbt_curriculum_p1.py:compute_skill_delta",
            "belief_need": "train.py:_need_belief = rssm_planner or rssm_reranker",
            "belief_missing": "snippet_wm_reranker.py:_REASON_BELIEF_MISSING=2",
            "alpha_zero": "snippet_wm_reranker.py:_REASON_ALPHA_ZERO=11",
            "compat": "elite_snippet_bank.py:retrieve_candidates_batched log_relative gap 0.20",
        },
    }
    (OUT / "root_cause_raw.json").write_text(json.dumps(raw, indent=2, default=str))
    print(json.dumps({"summary": summary, "bank_compat": bank["compat_frac"], "bank_sources": bank["source_hist"], "tb_keys": list((tb or {}).keys())[:15]}, indent=2))


if __name__ == "__main__":
    main()
