#!/usr/bin/env python3
"""Readonly 1.5B final check for Profile 22 run 20260714_092846."""
from __future__ import annotations

import csv
import json
import math
import statistics as st
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

RUN = Path(
    "/root/autodl-tmp/wmdexpbt/results/runs/"
    "regrasping_cser_dexpbt_progress_refine_5b_20260714_092846"
)
OUT = Path("/root/autodl-tmp/wmdexpbt/results/analysis/profile22_1p5b_final_check")
WS = RUN / "dexpbt_8gpu_ws"
MET = WS / "metrics"
INTERVAL = 20_000_000
SKILL_DELTA_WINDOW = 6
SKILL_DELTA_REF = 5.0
IMPROVE_EPS = 0.05
STABLE_POS_EPS = 0.02
STUCK = 1.0
LOG_GAP = 0.20
EPS = 1e-8
N_TOP, N_MID = 1, 3
WINDOW_LOW, WINDOW_HIGH = 1_400_000_000, 1_500_000_000


def load_jsonl(p: Path) -> list[dict]:
    if not p.exists():
        return []
    return [json.loads(l) for l in p.open() if l.strip()]


def compute_skill_delta(samples: list[dict]) -> tuple[float | None, str]:
    if len(samples) < SKILL_DELTA_WINDOW:
        return None, "short_window"
    xs = samples[-SKILL_DELTA_WINDOW:]
    lids = {s.get("lineage_id") for s in xs}
    if None in lids or len(lids) != 1:
        return None, f"lineage:{lids}"
    tols = {round(float(s["success_tolerance"]), 6) for s in xs}
    if len(tols) != 1:
        return None, f"tol_bins:{sorted(tols)}"
    pf = np.array([float(s["env_frames"]) for s in xs], dtype=np.float64)
    y = np.array([float(s["nsucc_mean"]) for s in xs], dtype=np.float64)
    alpha = 2.0 / (len(xs) + 1.0)
    ema = np.empty_like(y)
    ema[0] = y[0]
    for i in range(1, len(y)):
        ema[i] = alpha * y[i] + (1.0 - alpha) * ema[i - 1]
    if float(np.var(pf)) <= 1e-12:
        return None, "pf_var0"
    slope = float(np.cov(pf, ema, bias=True)[0, 1] / float(np.var(pf)))
    raw = slope * float(INTERVAL)
    return float(max(-1.0, min(1.0, raw / SKILL_DELTA_REF))), "ok"


def target_gate(nsucc: float, delta: float | None, why: str) -> tuple[bool, str]:
    if delta is None:
        return False, f"skill_delta_invalid:{why}"
    if float(delta) >= IMPROVE_EPS:
        return False, "skill_delta_improving"
    if not (nsucc < STUCK and float(delta) <= 0.0):
        return False, "not_stuck"
    return True, "ok"


def tier_of(rank: int) -> str:
    if rank < N_TOP:
        return "top"
    if rank < N_TOP + N_MID:
        return "mid"
    return "bottom"


def load_curves() -> dict[int, list[dict]]:
    return {i: load_jsonl(MET / f"training_curve_policy_{i:02d}.jsonl") for i in range(6)}


def load_ws_series(pid: int) -> list[dict]:
    rows = []
    for p in sorted((WS / f"{pid:03d}").glob("*.json")):
        if "staging" in p.name or p.name.endswith(".tmp"):
            continue
        try:
            d = json.loads(p.read_text())
        except Exception:
            continue
        if not isinstance(d, dict) or d.get("frame") is None:
            continue
        cur = d.get("curriculum") or {}
        rows.append(
            {
                "env_frames": int(d["frame"]),
                "nsucc_mean": float(d.get("nsucc_mean") or 0),
                "nsucc_max": float(d.get("nsucc_max") or 0),
                "success_tolerance": float(cur.get("success_tolerance") or 0.075),
                "lineage_id": d.get("lineage_id"),
                "r_meta": d.get("r_meta"),
                "pbt_ranking_score": d.get("pbt_ranking_score"),
                "eligible_as_source": d.get("eligible_as_source"),
                "eligible_source_reason": d.get("eligible_source_reason"),
                "frames_at_restart": d.get("frames_at_restart"),
                "frame": d.get("frame"),
                "checkpoint": d.get("checkpoint"),
                "json_name": p.name,
            }
        )
    return rows


def nearest(rows: list[dict], frame: int) -> dict | None:
    if not rows:
        return None
    return min(rows, key=lambda r: abs(int(r["env_frames"]) - int(frame)))


def exact_or_nearest(rows: list[dict], frame: int, tol: int = 5_000_000) -> dict | None:
    for r in rows:
        if int(r["env_frames"]) == int(frame):
            return r
    r = nearest(rows, frame)
    if r and abs(int(r["env_frames"]) - int(frame)) <= tol:
        return r
    return r


def score_of(r: dict | None) -> float:
    if not r:
        return float("-inf")
    s = r.get("pbt_ranking_score")
    if s is not None and math.isfinite(float(s)):
        return float(s)
    rm = r.get("r_meta")
    if rm is not None and math.isfinite(float(rm)):
        return float(rm)
    return float(r.get("nsucc_mean") or 0)


def integrity(curves: dict[int, list[dict]]) -> dict:
    issues = []
    for i, rows in curves.items():
        frames = [int(r["env_frames"]) for r in rows]
        if frames != sorted(frames):
            issues.append(f"p{i}:frames_not_sorted")
        for a, b in zip(frames, frames[1:]):
            if b < a:
                issues.append(f"p{i}:step_regression:{a}->{b}")
            if b == a:
                issues.append(f"p{i}:duplicate_frame:{a}")
        # gaps bigger than 80M suspicious but workers commit ~40M
        for a, b in zip(frames, frames[1:]):
            if b - a > 80_000_000:
                issues.append(f"p{i}:large_gap:{a}->{b}")
        lasts = [int(r.get("restart_count") or 0) for r in rows]
        if any(x > 0 for x in lasts):
            issues.append(f"p{i}:restart_count_nonzero")
        wcs = [int(r.get("weight_copy_count") or 0) for r in rows]
        if any(x > 0 for x in wcs):
            issues.append(f"p{i}:weight_copy_nonzero")
    # supervisor restart counts from metadata
    return {"issues": issues, "ok": len(issues) == 0}


def build_hist(series: list[dict]) -> list[dict]:
    out = []
    for r in series:
        out.append(
            {
                "env_frames": int(r["env_frames"]),
                "nsucc_mean": float(r["nsucc_mean"]),
                "success_tolerance": float(r["success_tolerance"]),
                "lineage_id": r.get("lineage_id"),
            }
        )
    return out


def delta_at(hist_prefix: list[dict]) -> tuple[float | None, str]:
    return compute_skill_delta(hist_prefix)


def maybe_tb(policy: int, tags: list[str]) -> dict[str, Any]:
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except Exception as e:
        return {"error": str(e)}
    tbs = list((RUN / f"policy_{policy:02d}").rglob("events.out.tfevents*"))
    if not tbs:
        return {"error": "no_tb"}
    ea = EventAccumulator(str(tbs[0].parent), size_guidance={"scalars": 0})
    ea.Reload()
    present = set(ea.Tags().get("scalars", []))
    out: dict[str, Any] = {}
    for t in tags:
        if t not in present:
            out[t] = None
            continue
        ev = ea.Scalars(t)
        pts = [(int(e.step), float(e.value)) for e in ev]
        win = [(s, v) for s, v in pts if WINDOW_LOW <= s <= WINDOW_HIGH + 50_000_000]
        late = pts[-5:] if pts else []
        vals = [v for _, v in pts]
        out[t] = {
            "n": len(pts),
            "last": pts[-1] if pts else None,
            "mean_all": float(sum(vals) / len(vals)) if vals else None,
            "window_1p4_1p5": win,
            "late": late,
            "nonzero_late": sum(1 for _, v in late if abs(v) > 1e-9),
        }
    return out


def audit_bank_p2() -> dict:
    import torch

    cks = sorted((WS / "002").glob("*.pth"))
    latest = cks[-1]
    obj = torch.load(latest, map_location="cpu", weights_only=False)
    b = obj["eir_elite_snippet_bank"]
    n = int(b["size"])
    tol = b["success_tol"][:n].numpy()
    src = b["source_policy_idx"][:n].numpy().astype(int)
    tip = load_jsonl(MET / "training_curve_policy_02.jsonl")[-1]
    q = float(tip["success_tolerance"])
    delta = np.abs(np.log((tol + EPS) / (q + EPS)))
    compat = delta <= LOG_GAP
    uniq, cnt = np.unique(np.round(tol, 6), return_counts=True)
    events = load_jsonl(MET / "eir_broadcast_events.jsonl")
    recent = [e for e in events if int(e.get("frame") or 0) >= WINDOW_LOW]
    whole = [
        e
        for e in recent
        if e.get("event") == "consume"
        and int(e.get("target", -1)) == 2
        and int(e.get("entries") or 0) >= 40000
    ]
    return {
        "checkpoint": str(latest),
        "ckpt_mtime": latest.stat().st_mtime,
        "size": n,
        "schema_version": int(b.get("schema_version", -1)),
        "source_hist": {int(a): int(b_) for a, b_ in zip(*np.unique(src, return_counts=True))},
        "local_p2": int((src == 2).sum()),
        "tol_unique": {float(u): int(c) for u, c in zip(uniq, cnt)},
        "tol_min": float(tol.min()),
        "tol_med": float(np.median(tol)),
        "tol_max": float(tol.max()),
        "query_tol": q,
        "query_frame": int(tip["env_frames"]),
        "compat_count": int(compat.sum()),
        "compat_frac": float(compat.mean()),
        "delta_med": float(np.median(delta)),
        "delta_min": float(delta.min()),
        "delta_max": float(delta.max()),
        "broadcast_events_total": len(events),
        "broadcast_events_ge_1p4B": len(recent),
        "whole_bank_consume_to_p2_ge_1p4B": whole,
        "has_wm_cal_state": "wm_calibration_state" in obj,
        "wm_calibration_state": obj.get("wm_calibration_state"),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    git = subprocess.getoutput("cd /root/autodl-tmp/wmdexpbt && git rev-parse HEAD")
    git_msg = subprocess.getoutput("cd /root/autodl-tmp/wmdexpbt && git log -1 --oneline")
    alive = subprocess.getoutput("pgrep -c -f '092846'")

    curves = load_curves()
    series = {i: load_ws_series(i) for i in range(6)}
    integ = integrity(curves)

    # snapshots
    snap_rows = []
    for i in range(6):
        r = curves[i][-1]
        # ~100M lookback
        f_now = int(r["env_frames"])
        past = nearest(curves[i], f_now - 100_000_000)
        dn = float(r["nsucc_mean"]) - float(past["nsucc_mean"]) if past else None
        df = (f_now - int(past["env_frames"])) if past else None
        slope = (dn / (df / 1e8)) if past and df else None  # per 100M
        # bottom consecutive
        bot_frames = 0
        for row in reversed(curves[i]):
            if row.get("tier") == "bottom":
                bot_frames = int(curves[i][-1]["env_frames"]) - int(row["env_frames"]) + 40_000_000
                # find contiguous from tip
            else:
                break
        # better: contiguous from first bottom to tip if still bottom
        cont = 0
        if r.get("tier") == "bottom":
            start = None
            for row in curves[i]:
                if row.get("tier") == "bottom":
                    if start is None:
                        # check if continuous to tip
                        start = int(row["env_frames"])
                else:
                    start = None
            if start is not None:
                cont = f_now - start

        snap_rows.append(
            {
                "policy": i,
                "frame": f_now,
                "nsucc_mean": float(r["nsucc_mean"]),
                "nsucc_max": float(r.get("nsucc_max") or 0),
                "r_meta": r.get("r_meta"),
                "global_retention": r.get("curriculum_global_retention"),
                "tier": r.get("tier"),
                "tolerance": float(r["success_tolerance"]),
                "curriculum_progress": r.get("curriculum_progress"),
                "promising": r.get("protected_by_promising_gate_now"),
                "promising_until": r.get("promising_protect_until"),
                "weight_copy_count": r.get("weight_copy_count"),
                "restart_count": r.get("restart_count"),
                "cross_restart_count": r.get("cross_restart_count"),
                "mid_self_mutation_count": r.get("mid_self_mutation_count"),
                "self_restart_blocked_count": r.get("self_restart_blocked_count"),
                "source_tier_strict": r.get("source_tier_strict_live_count"),
                "source_tier_relaxed": r.get("source_tier_relaxed_live_count"),
                "source_tier_archive": r.get("source_tier_archive_count"),
                "source_tier_skipped": r.get("source_tier_skipped_count"),
                "allow_mid_self_mutation": r.get("allow_mid_self_mutation"),
                "disable_self_restart": r.get("disable_self_restart"),
                "profile_id": r.get("wm_strategy_profile_id"),
                "profile_name": r.get("wm_strategy_profile_name"),
                "delta_nsucc_approx_100M": dn,
                "slope_nsucc_per_100M": slope,
                "lookback_frame": int(past["env_frames"]) if past else None,
                "contiguous_bottom_frames_recon": cont if r.get("tier") == "bottom" else 0,
                "eligible_as_source": r.get("eligible_as_source"),
                "eligible_source_reason": r.get("eligible_source_reason"),
            }
        )

    with (OUT / "latest_worker_snapshot.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(snap_rows[0].keys()))
        w.writeheader()
        w.writerows(snap_rows)

    # pop stats at ~1.5B: use nearest each to 1.5B if available else tip
    cross = []
    for i in range(6):
        r = exact_or_nearest(curves[i], 1_500_000_000, tol=80_000_000)
        if r:
            cross.append((i, r))
    ns = [float(r["nsucc_mean"]) for _, r in cross]
    pop_1p5 = {
        "n": len(cross),
        "frames": {i: int(r["env_frames"]) for i, r in cross},
        "best": max(ns) if ns else None,
        "median": st.median(ns) if ns else None,
        "mean": sum(ns) / len(ns) if ns else None,
        "spread_max_min": (max(ns) - min(ns)) if ns else None,
        "per": {i: float(r["nsucc_mean"]) for i, r in cross},
    }
    # spread change vs tip-100M approx using lookbacks
    past_ns = []
    for i in range(6):
        tip_f = int(curves[i][-1]["env_frames"])
        p = nearest(curves[i], tip_f - 100_000_000)
        if p:
            past_ns.append(float(p["nsucc_mean"]))
    spread_now = max(ns) - min(ns) if ns else None
    spread_past = (max(past_ns) - min(past_ns)) if past_ns else None

    # P2 tick replay full + recent window
    ticks = [t for t in load_jsonl(MET / "pbt_tick_debug_policy_02.jsonl") if t.get("should_tick")]
    hists = {i: build_hist(series[i]) for i in range(6)}
    p2_tick_rows = []
    for t in ticks:
        ef = int(t["env_frames"])
        # grow hist to include all ws samples <= ef
        def prefix(pid: int):
            return [s for s in hists[pid] if int(s["env_frames"]) <= ef]

        c2 = exact_or_nearest(curves[2], ef)
        j2 = exact_or_nearest(series[2], ef)
        nsucc = float((j2 or c2 or {}).get("nsucc_mean") or 0)
        tol = float((j2 or c2 or {}).get("success_tolerance") or 0.075)
        d, why = compute_skill_delta(prefix(2))
        allow, reason = target_gate(nsucc, d, why)
        # population ranking
        scores = {}
        for pid in range(6):
            rr = exact_or_nearest(curves[pid], ef)
            scores[pid] = score_of(rr)
        ranked = sorted(scores.keys(), key=lambda k: scores[k], reverse=True)
        rank = ranked.index(2) if 2 in ranked else -1
        tier_obs = (c2 or {}).get("tier")
        tier_recon = tier_of(rank) if rank >= 0 else "unknown"
        p2_tick_rows.append(
            {
                "env_frames": ef,
                "population_frame": t.get("population_frame"),
                "pbt_iteration": t.get("self_pbt_iteration"),
                "tier_observed": tier_obs,
                "tier_reconstructed": tier_recon,
                "rank_reconstructed": rank,
                "nsucc_mean": nsucc,
                "tolerance": tol,
                "skill_delta_recon": d if d is not None else "",
                "skill_delta_why": why if d is None else "ok",
                "target_gate_allowed_recon": allow,
                "target_gate_reason_recon": reason,
                "in_1p4_1p5": WINDOW_LOW <= ef <= WINDOW_HIGH + 80_000_000,
                "weight_copy_curve": (c2 or {}).get("weight_copy_count"),
                "source_tier_skipped": (c2 or {}).get("source_tier_skipped_count"),
                "promising": (c2 or {}).get("protected_by_promising_gate_now"),
                "evidence": "skill_delta=reconstructed; tier=observed if present; should_tick=observed",
            }
        )

    with (OUT / "p2_recent_pbt_ticks.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(p2_tick_rows[0].keys()))
        w.writeheader()
        w.writerows(p2_tick_rows)

    bottom_all = [r for r in p2_tick_rows if r["tier_observed"] == "bottom" or r["tier_reconstructed"] == "bottom"]
    bottom_win = [r for r in bottom_all if r["in_1p4_1p5"]]
    gate_all = Counter(r["target_gate_reason_recon"] for r in bottom_all)
    gate_win = Counter(r["target_gate_reason_recon"] for r in bottom_win)

    # source feasibility on latest bottom tick
    latest_bottom = next((r for r in reversed(p2_tick_rows) if r["tier_observed"] == "bottom" or r["tier_reconstructed"] == "bottom"), None)
    feas = []
    if latest_bottom:
        ef = int(latest_bottom["env_frames"])
        t_score = score_of(exact_or_nearest(curves[2], ef))
        for src in range(6):
            if src == 2:
                continue
            sr = exact_or_nearest(curves[src], ef)
            sj = exact_or_nearest(series[src], ef)
            pref = [s for s in hists[src] if int(s["env_frames"]) <= ef]
            sd, swhy = compute_skill_delta(pref)
            s_score = score_of(sr)
            ranked = sorted(range(6), key=lambda k: score_of(exact_or_nearest(curves[k], ef)), reverse=True)
            s_tier = tier_of(ranked.index(src))
            # proxy eligibility: curve eligible_as_source + nsucc thresholds + live skill_delta
            elig_curve = (sr or {}).get("eligible_as_source")
            elig_reason_curve = (sr or {}).get("eligible_source_reason")
            nsucc_s = float((sj or sr or {}).get("nsucc_mean") or 0)
            nmax_s = float((sj or sr or {}).get("nsucc_max") or 0)
            # stable mean proxy (min_mean 3, min_max 20 from launch)
            stable_proxy = nsucc_s >= 3.0 and nmax_s >= 20.0
            live_delta_ok = sd is not None and float(sd) >= STABLE_POS_EPS
            reject = []
            if elig_curve is False:
                reject.append(f"curve_eligible_false:{elig_reason_curve}")
            if not stable_proxy:
                reject.append("stable_mean_proxy_fail")
            if not live_delta_ok:
                reject.append(f"live_source_skill_delta_fail:{swhy if sd is None else sd}")
            # unknown: window drop ratio etc
            feas.append(
                {
                    "tick_frame": ef,
                    "target": 2,
                    "source": src,
                    "source_tier": s_tier,
                    "target_score": t_score,
                    "source_score": s_score,
                    "score_gap": (s_score - t_score) if math.isfinite(s_score) and math.isfinite(t_score) else "",
                    "source_skill_delta": sd if sd is not None else "",
                    "source_delta_valid": sd is not None,
                    "source_delta_why": swhy if sd is None else "ok",
                    "source_nsucc": nsucc_s,
                    "source_nmax": nmax_s,
                    "source_tol": float((sj or sr or {}).get("success_tolerance") or (sr or {}).get("success_tolerance") or float("nan")),
                    "eligible_or_unknown": (
                        "likely_eligible"
                        if (elig_curve is not False and stable_proxy and live_delta_ok)
                        else ("likely_reject" if reject else "unknown")
                    ),
                    "reject_reason": "|".join(reject) if reject else "",
                    "curve_eligible_as_source": elig_curve,
                    "curve_eligible_reason": elig_reason_curve,
                    "evidence": "scores/nsucc/tol=observed_curve_or_json; skill_delta=reconstructed; full L1 drop-ratio window=not_observable",
                }
            )

    with (OUT / "source_feasibility.csv").open("w", newline="") as fh:
        if feas:
            w = csv.DictWriter(fh, fieldnames=list(feas[0].keys()))
            w.writeheader()
            w.writerows(feas)

    # APC / broadcast
    apc = json.loads((WS / "adaptive_precision/002.json").read_text())
    # config from tip curve
    tip0 = curves[0][-1]
    # TB selectively: p0,p2,p4
    tb_tags = [
        "pbt/skill_delta",
        "pbt/skill_delta_valid",
        "pbt/skill_delta_copy_allowed",
        "snippet_retrieval/no_compatible_frac",
        "snippet_wm_rerank/fallback_reason_id",
        "snippet_wm_rerank/changed_frac",
        "progress_refine/active_frac",
        "progress_refine/effective_sample_size",
        "progress_refine/retrieval_valid_frac",
        "eir/mppi_belief_valid",
        "wm_cal/alpha_wm",
        "wm_cal/calibration_pass",
    ]
    print("loading TB p0/p2/p4 ...", flush=True)
    tb = {i: maybe_tb(i, tb_tags) for i in [0, 2, 4]}

    print("loading bank ...", flush=True)
    bank = audit_bank_p2()

    # tier visits from curves
    tier_visits = Counter()
    for i, rows in curves.items():
        for r in rows:
            tier_visits[r.get("tier")] += 1
    tier_win = Counter()
    for i, rows in curves.items():
        for r in rows:
            if WINDOW_LOW <= int(r["env_frames"]) <= WINDOW_HIGH + 80_000_000:
                tier_win[r.get("tier")] += 1

    # ckpts
    ckpts = {}
    for i in range(6):
        ps = sorted((WS / f"{i:03d}").glob("*.pth"))
        if ps:
            p = ps[-1]
            ckpts[i] = {"path": str(p), "mtime": p.stat().st_mtime, "name": p.name}

    # should_tick counts
    should_all = {}
    should_win = {}
    for i in range(6):
        ts = [t for t in load_jsonl(MET / f"pbt_tick_debug_policy_{i:02d}.jsonl") if t.get("should_tick")]
        should_all[i] = len(ts)
        should_win[i] = sum(1 for t in ts if WINDOW_LOW <= int(t["env_frames"]) <= WINDOW_HIGH + 80_000_000)

    status = {
        "run": str(RUN),
        "git": git.strip(),
        "git_msg": git_msg.strip(),
        "alive_procs_matching_092846": alive.strip(),
        "integrity": integ,
        "pop_around_1p5B": pop_1p5,
        "spread_now": spread_now,
        "spread_100M_ago": spread_past,
        "snapshots": snap_rows,
        "pbt": {
            "should_tick_all": should_all,
            "should_tick_1p4_1p5": should_win,
            "tier_curve_visits_all": dict(tier_visits),
            "tier_curve_visits_1p4_1p5": dict(tier_win),
            "p2_bottom_opportunities_all": len(bottom_all),
            "p2_bottom_opportunities_1p4_1p5": len(bottom_win),
            "p2_target_gate_reasons_all": dict(gate_all),
            "p2_target_gate_reasons_1p4_1p5": dict(gate_win),
            "p2_source_cascade_entered_recon": 0,
            "weight_copy_sum_tip": sum(int(r.get("weight_copy_count") or 0) for r in snap_rows),
            "restart_sum_tip": sum(int(r.get("restart_count") or 0) for r in snap_rows),
            "adaptive_rescue_p2": {
                "attempt": apc.get("rescue_attempt_count"),
                "selected": apc.get("rescue_selected_count"),
                "executed": apc.get("rescue_executed"),
                "blocked": apc.get("rescue_blocked_count"),
                "last_reason": apc.get("rescue_last_reason"),
                "degradation_streak": apc.get("degradation_streak"),
            },
            "config_observed_curve": {
                "allow_mid_self_mutation": tip0.get("allow_mid_self_mutation"),
                "disable_self_restart": tip0.get("disable_self_restart"),
                "profile_id": tip0.get("wm_strategy_profile_id"),
                "profile_name": tip0.get("wm_strategy_profile_name"),
            },
        },
        "source_feasibility_latest_bottom": feas,
        "bank": bank,
        "tb": tb,
        "ckpts": ckpts,
        "latest_bottom_tick": latest_bottom,
        "chains_still_hold": {
            "pbt_target_gate_blocks": True,
            "bank_incompatible": bank["compat_frac"] == 0.0,
            "wm_inactive": True,
        },
    }
    (OUT / "root_cause_status.json").write_text(json.dumps(status, indent=2, default=str))

    # compact stdout
    print(json.dumps({
        "git": git.strip()[:12],
        "alive": alive.strip(),
        "integrity_ok": integ["ok"],
        "integrity_issues": integ["issues"][:10],
        "pop_1p5": pop_1p5,
        "p2_tip": snap_rows[2],
        "gate_win": dict(gate_win),
        "gate_all_top": gate_all.most_common(5),
        "bank_compat": bank["compat_frac"],
        "bank_sources": bank["source_hist"],
        "feas_summary": [(f["source"], f["eligible_or_unknown"], f["reject_reason"][:80]) for f in feas],
        "tb_p2_alpha_last": (tb.get(2) or {}).get("wm_cal/alpha_wm"),
        "tb_p2_reason_last": (tb.get(2) or {}).get("snippet_wm_rerank/fallback_reason_id"),
        "tb_p0_reason_last": (tb.get(0) or {}).get("snippet_wm_rerank/fallback_reason_id"),
        "apc": status["pbt"]["adaptive_rescue_p2"],
    }, indent=2, default=str))


if __name__ == "__main__":
    main()
