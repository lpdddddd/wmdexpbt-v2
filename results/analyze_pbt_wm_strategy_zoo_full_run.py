#!/usr/bin/env python3
"""PBT-WM Strategy Zoo v1.0 run analyzer.

Aggregates per-profile metrics from training_curve JSONL + RUN_CONFIG and
answers the nine strategy-zoo evaluation questions.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from collections import defaultdict
from pathlib import Path
from typing import Any


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return rows


def _safe_float(v: Any) -> float | None:
    try:
        x = float(v)
        if math.isnan(x) or math.isinf(x):
            return None
        return x
    except (TypeError, ValueError):
        return None


def _stats(vals: list[float]) -> dict[str, Any]:
    if not vals:
        return {"n": 0, "last": None, "peak": None, "mean": None}
    return {
        "n": len(vals),
        "last": vals[-1],
        "peak": max(vals),
        "mean": sum(vals) / len(vals),
    }


TOL_BINS = ("gt_006", "005_006", "004_005", "le_004")
EXPOSURE_BINS = ("003_004", "002_003", "le_002")


def _exposure_bin(tol: float | None) -> str | None:
    if tol is None:
        return None
    if tol <= 0.02:
        return "le_002"
    if tol <= 0.03:
        return "002_003"
    if tol <= 0.04:
        return "003_004"
    return None


def _tol_bin(tol: float | None) -> str:
    if tol is None:
        return "le_004"
    if tol > 0.06:
        return "gt_006"
    if tol > 0.05:
        return "005_006"
    if tol > 0.04:
        return "004_005"
    return "le_004"


def _by_tol_bin_stats(rows: list[dict]) -> dict[str, dict[str, Any]]:
    """Aggregate per-tol-bin WM activation / curriculum stats for one policy."""
    out: dict[str, dict[str, Any]] = {}
    for b in TOL_BINS:
        out[b] = {
            "n": 0,
            "actor_distill_active_fraction": None,
            "actor_distill_activation_weight_mean": None,
            "wm_score_used_fraction": None,
            "wm_curriculum_blocked_shrink_count": 0,
            "nsucc_mean_latest": None,
            "collapse_count": 0,
        }
    acc: dict[str, dict[str, list]] = {b: defaultdict(list) for b in TOL_BINS}
    for r in rows:
        tol_raw = r.get("actor_distill_tol_bin")
        if tol_raw in TOL_BINS:
            b = str(tol_raw)
        else:
            b = _tol_bin(_safe_float(r.get("success_tolerance")))
        out[b]["n"] += 1
        adf = _safe_float(
            r.get("actor_distill_activation_nonzero_frac",
                  r.get("profile_actor_distill_enabled"))
        )
        if adf is not None:
            acc[b]["adf"].append(adf)
        awm = _safe_float(r.get("actor_distill_activation_weight_mean"))
        if awm is not None:
            acc[b]["awm"].append(awm)
        acc[b]["wm_used"].append(1.0 if r.get("wm_score_used") else 0.0)
        if r.get("wm_curriculum_blocked_shrink"):
            out[b]["wm_curriculum_blocked_shrink_count"] += 1
        ns = _safe_float(r.get("nsucc_mean"))
        if ns is not None:
            acc[b]["nsucc"].append(ns)
        if (
            r.get("collapse_freeze_active")
            or r.get("early_collapse_triggered")
            or r.get("hard_collapse_triggered")
        ):
            out[b]["collapse_count"] += 1
    for b in TOL_BINS:
        if acc[b]["adf"]:
            out[b]["actor_distill_active_fraction"] = sum(acc[b]["adf"]) / len(acc[b]["adf"])
        if acc[b]["awm"]:
            out[b]["actor_distill_activation_weight_mean"] = sum(acc[b]["awm"]) / len(acc[b]["awm"])
        if acc[b]["wm_used"]:
            out[b]["wm_score_used_fraction"] = sum(acc[b]["wm_used"]) / len(acc[b]["wm_used"])
        if acc[b]["nsucc"]:
            out[b]["nsucc_mean_latest"] = acc[b]["nsucc"][-1]
    return out


    return out


def _tol_exposure_stats(rows: list[dict]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for b in EXPOSURE_BINS:
        out[b] = {
            "ticks_by_tol_bin": 0,
            "frames_by_tol_bin": 0,
            "nsucc_mean_by_tol_bin": None,
            "min_nsucc_by_tol_bin": None,
            "retention_by_tol_bin": None,
            "collapse_count_by_tol_bin": 0,
        }
    acc: dict[str, dict[str, list]] = {
        b: {"nsucc": [], "frames": []} for b in EXPOSURE_BINS
    }
    for r in rows:
        tol = _safe_float(r.get("success_tolerance"))
        b = _exposure_bin(tol)
        if b is None:
            continue
        out[b]["ticks_by_tol_bin"] += 1
        out[b]["frames_by_tol_bin"] += int(r.get("env_frames", 0) or 0)
        ns = _safe_float(r.get("nsucc_mean"))
        if ns is not None:
            acc[b]["nsucc"].append(ns)
        if r.get("collapse_freeze_active") or r.get("hard_collapse_triggered"):
            out[b]["collapse_count_by_tol_bin"] += 1
    for b in EXPOSURE_BINS:
        if acc[b]["nsucc"]:
            out[b]["nsucc_mean_by_tol_bin"] = sum(acc[b]["nsucc"]) / len(
                acc[b]["nsucc"]
            )
            out[b]["min_nsucc_by_tol_bin"] = min(acc[b]["nsucc"])
            peak = max(acc[b]["nsucc"])
            if peak > 1e-9:
                out[b]["retention_by_tol_bin"] = acc[b]["nsucc"][-1] / peak
    return out


def _shrink_safety_stats(rows: list[dict]) -> dict[str, Any]:
    blocked_reason_counts: dict[str, int] = defaultdict(int)
    rollback_reason_counts: dict[str, int] = defaultdict(int)
    last_mastered_trajectory: list[dict[str, Any]] = []
    for r in rows:
        reason = str(r.get("curriculum_safety_block_reason") or "")
        if r.get("curriculum_safety_blocked_shrink") and reason:
            blocked_reason_counts[reason] += 1
        rb = str(r.get("curriculum_rollback_reason") or r.get("rollback_reason") or "")
        if r.get("curriculum_rollback_triggered") and rb:
            rollback_reason_counts[rb] += 1
        lm = _safe_float(r.get("curriculum_last_mastered_tolerance"))
        if lm is not None and lm > 0:
            last_mastered_trajectory.append(
                {
                    "env_frames": int(r.get("env_frames", 0) or 0),
                    "tolerance": lm,
                    "nsucc_mean": _safe_float(r.get("curriculum_last_mastered_nsucc_mean")),
                }
            )
    return {
        "shrink_count_total": sum(
            1 for r in rows if r.get("curriculum_shrink_reason")
        ),
        "strong_shrink_count": sum(
            1 for r in rows if r.get("curriculum_shrink_reason") == "strong"
        ),
        "strong_shrink_blocked_by_global_retention": sum(
            1
            for r in rows
            if not r.get("curriculum_strong_shrink_allowed", True)
            and r.get("curriculum_strong_shrink_block_reason")
            == "low_global_retention_or_unstable"
        ),
        "strong_shrink_blocked_by_danger_zone": sum(
            1
            for r in rows
            if not r.get("curriculum_strong_shrink_allowed", True)
            and r.get("curriculum_strong_shrink_block_reason") == "danger_zone"
        ),
        "blocked_shrink_count": sum(
            1 for r in rows if r.get("curriculum_safety_blocked_shrink")
        ),
        "blocked_shrink_reason_counts": dict(blocked_reason_counts),
        "danger_zone_block_count": sum(
            1 for r in rows if r.get("curriculum_danger_blocked_shrink")
        ),
        "rollback_count": sum(
            1 for r in rows if r.get("curriculum_rollback_triggered")
        ),
        "rollback_reason_counts": dict(rollback_reason_counts),
        "last_mastered_tolerance_trajectory": last_mastered_trajectory[-20:],
    }


def _causal_diagnosis(
    by_profile: dict[str, dict[str, Any]],
    all_curves: dict[int, list[dict]],
    policy_profiles: dict[int, str],
) -> dict[str, Any]:
    def _exposure(prof: str, b: str) -> int:
        pid = next(
            (p for p, pr in policy_profiles.items() if pr == prof),
            None,
        )
        if pid is None:
            return 0
        return (
            by_profile.get(prof, {})
            .get("tol_exposure", {})
            .get(b, {})
            .get("ticks_by_tol_bin", 0)
        )

    wm_profiles = [
        p
        for p in by_profile
        if p not in ("ppo_only", "unknown")
    ]
    p0_le002 = _exposure("ppo_only", "le_002")
    wm_le002 = max(_exposure(p, "le_002") for p in wm_profiles) if wm_profiles else 0
    danger_blocks = sum(
        v.get("shrink_safety", {}).get("danger_zone_block_count", 0)
        for v in by_profile.values()
    )
    real_fallback_blocks = sum(
        v.get("shrink_safety", {}).get("blocked_shrink_reason_counts", {}).get(
            "real_signal_risk", 0
        )
        for v in by_profile.values()
    )
    imagined_invalid_frac = {}
    for prof, data in by_profile.items():
        pid = data.get("policy_idx")
        rows = all_curves.get(pid, [])
        if rows:
            imagined_invalid_frac[prof] = sum(
                1 for r in rows if not r.get("wm_imagined_valid")
            ) / len(rows)
    return {
        "wm_workers_enter_tol_le_003_earlier_than_ppo": wm_le002 > p0_le002,
        "danger_zone_blocked_shrink_nonzero": danger_blocks > 0,
        "strong_shrink_blocked_on_low_global_retention": any(
            v.get("shrink_safety", {}).get(
                "strong_shrink_blocked_by_global_retention", 0
            )
            > 0
            for v in by_profile.values()
        ),
        "rollback_to_last_mastered_occurred": any(
            v.get("shrink_safety", {}).get("rollback_count", 0) > 0
            for v in by_profile.values()
        ),
        "ppo_high_retention_survivorship_bias": p0_le002 <= 3 and wm_le002 >= 5,
        "wm_imagined_still_mostly_invalid": all(
            v > 0.5 for v in imagined_invalid_frac.values()
        )
        if imagined_invalid_frac
        else None,
        "real_fallback_took_over_when_imagined_invalid": real_fallback_blocks > 0,
        "exposure_ticks_le_002": {"ppo_only": p0_le002, "wm_max": wm_le002},
        "imagined_invalid_fraction_by_profile": imagined_invalid_frac,
    }


def _wm_imagined_head_diagnostic(
    all_curves: dict[int, list[dict]],
    policy_profiles: dict[int, str],
) -> dict[str, Any]:
    out: dict[str, Any] = {"by_policy": {}, "by_tol_bin": {}}
    for pid, rows in all_curves.items():
        prof = policy_profiles.get(pid, "unknown")
        if not rows:
            continue
        valid_frac = sum(1 for r in rows if r.get("wm_imagined_valid")) / len(rows)
        unc = [_safe_float(r.get("wm_imagined_uncertainty")) for r in rows]
        unc = [x for x in unc if x is not None]
        succ = [_safe_float(r.get("wm_imagined_success_prob")) for r in rows]
        succ = [x for x in succ if x is not None]
        drop = [_safe_float(r.get("wm_imagined_drop_risk")) for r in rows]
        drop = [x for x in drop if x is not None]
        rec = [_safe_float(r.get("wm_imagined_recovery_prob")) for r in rows]
        rec = [x for x in rec if x is not None]
        counterfactual = sum(
            1
            for r in rows
            if not r.get("wm_imagined_valid")
            and r.get("curriculum_safety_real_risk")
        )
        out["by_policy"][str(pid)] = {
            "profile": prof,
            "imagined_valid_fraction": valid_frac,
            "success_prob": _stats(succ),
            "drop_risk": _stats(drop),
            "recovery_prob": _stats(rec),
            "uncertainty": _stats(unc),
            "uncertainty_le_050_frac": (
                sum(1 for x in unc if x <= 0.5) / len(unc) if unc else None
            ),
            "uncertainty_le_070_frac": (
                sum(1 for x in unc if x <= 0.7) / len(unc) if unc else None
            ),
            "uncertainty_le_090_frac": (
                sum(1 for x in unc if x <= 0.9) / len(unc) if unc else None
            ),
            "uncertainty_le_095_frac": (
                sum(1 for x in unc if x <= 0.95) / len(unc) if unc else None
            ),
            "counterfactual_would_block_by_real_fallback": counterfactual,
        }
    for b in TOL_BINS + list(EXPOSURE_BINS):
        sub = []
        for rows in all_curves.values():
            for r in rows:
                tol = _safe_float(r.get("success_tolerance"))
                if b in EXPOSURE_BINS:
                    if _exposure_bin(tol) != b:
                        continue
                elif _tol_bin(tol) != b:
                    continue
                sub.append(r)
        if sub:
            out["by_tol_bin"][b] = {
                "imagined_valid_fraction": sum(
                    1 for r in sub if r.get("wm_imagined_valid")
                )
                / len(sub),
                "n": len(sub),
            }
    return out


def _parse_run_config(run_dir: Path) -> dict[str, str]:
    cfg: dict[str, str] = {}
    p = run_dir / "RUN_CONFIG.txt"
    if not p.exists():
        return cfg
    for line in p.read_text().splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            cfg[k.strip()] = v.strip()
    return cfg


def _detect_ws(run_dir: Path, ws_name: str | None) -> Path:
    if ws_name:
        return run_dir / ws_name
    cand = run_dir / "dexpbt_8gpu_ws"
    if (cand / "metrics").is_dir():
        return cand
    for child in sorted(run_dir.iterdir()) if run_dir.is_dir() else []:
        if (child / "metrics").is_dir():
            return child
    return cand


def analyze_run(run_dir: Path, *, ws_name: str | None = None, num_policies: int = 6) -> dict[str, Any]:
    run_dir = run_dir.resolve()
    ws = _detect_ws(run_dir, ws_name)
    metrics = ws / "metrics"
    run_cfg = _parse_run_config(run_dir)

    policy_profiles: dict[int, str] = {}
    for i in range(num_policies):
        key = f"policy_{i:02d}_profile"
        if key in run_cfg:
            policy_profiles[i] = run_cfg[key]
        else:
            policy_profiles[i] = "unknown"

    by_profile: dict[str, dict[str, Any]] = defaultdict(dict)
    all_curves: dict[int, list[dict]] = {}

    for pid in range(num_policies):
        prof = policy_profiles.get(pid, "unknown")
        rows = _read_jsonl(metrics / f"training_curve_policy_{pid:02d}.jsonl")
        all_curves[pid] = rows
        if not rows:
            by_profile[prof] = {"policy_idx": pid, "n_points": 0}
            continue

        nsucc = [_safe_float(r.get("nsucc_mean")) for r in rows]
        nsucc = [x for x in nsucc if x is not None]
        rmeta = [_safe_float(r.get("r_meta_raw", r.get("r_meta"))) for r in rows]
        rmeta = [x for x in rmeta if x is not None]
        tol = [_safe_float(r.get("success_tolerance")) for r in rows]
        tol = [x for x in tol if x is not None]

        peak_idx = nsucc.index(max(nsucc)) if nsucc else 0
        post_peak = nsucc[peak_idx:] if nsucc else []
        retention = (
            post_peak[-1] / post_peak[0]
            if len(post_peak) > 1 and post_peak[0] > 1e-9
            else None
        )

        shrink_blocks = sum(
            1 for r in rows if r.get("wm_curriculum_blocked_shrink")
        )
        profile_copies = sum(
            1
            for r in rows
            if str(r.get("pbt_wm_profile_copy_decision", "")) == "copied"
        )
        wm_score_used = sum(1 for r in rows if r.get("wm_score_used"))
        wm_fallback = sum(
            1
            for r in rows
            if str(r.get("wm_fallback_reason", "ok")) not in ("ok", "feature_off", "")
        )

        actor_distill_frac = [
            _safe_float(r.get("profile_actor_distill_enabled")) for r in rows
        ]
        mppi_frac = [_safe_float(r.get("profile_rssm_mppi_enabled")) for r in rows]

        by_profile[prof] = {
            "policy_idx": pid,
            "n_points": len(rows),
            "nsucc_mean": _stats(nsucc),
            "r_meta": _stats(rmeta),
            "success_tolerance": _stats(tol),
            "nsucc_retention_after_peak": retention,
            "restart_count_last": int(rows[-1].get("restart_count", 0)),
            "shrink_count": sum(
                1 for r in rows if r.get("curriculum_shrink_reason")
            ),
            "collapse_freeze_count": sum(
                1 for r in rows if r.get("collapse_freeze_active")
            ),
            "wm_curriculum_blocked_shrink_count": shrink_blocks,
            "pbt_profile_copy_count": profile_copies,
            "wm_score_used_ticks": wm_score_used,
            "wm_fallback_ticks": wm_fallback,
            "profile_actor_distill_enabled_frac": _stats(
                [x for x in actor_distill_frac if x is not None]
            ),
            "profile_rssm_mppi_enabled_frac": _stats(
                [x for x in mppi_frac if x is not None]
            ),
            "residual_apply_enabled_last": bool(
                rows[-1].get("profile_residual_apply_enabled", False)
            ),
            "by_tol_bin": _by_tol_bin_stats(rows),
            "tol_exposure": _tol_exposure_stats(rows),
            "shrink_safety": _shrink_safety_stats(rows),
        }

    profiles_sorted_stability = sorted(
        by_profile.items(),
        key=lambda kv: (
            -(kv[1].get("nsucc_retention_after_peak") or 0),
            -(kv[1].get("nsucc_mean", {}).get("last") or 0),
        ),
    )
    profiles_sorted_peak = sorted(
        by_profile.items(),
        key=lambda kv: -(kv[1].get("nsucc_mean", {}).get("peak") or 0),
    )

    diversity_rows = [
        r.get("pbt_wm_profile_diversity_count")
        for curves in all_curves.values()
        for r in curves
        if r.get("pbt_wm_profile_diversity_count") is not None
    ]
    diversity_min = min(diversity_rows) if diversity_rows else None

    questions = {
        "most_stable_profile": profiles_sorted_stability[0][0]
        if profiles_sorted_stability
        else None,
        "highest_peak_profile": profiles_sorted_peak[0][0]
        if profiles_sorted_peak
        else None,
        "wm_curriculum_reduces_collapse": (
            by_profile.get("wm_curriculum_controller", {}).get(
                "wm_curriculum_blocked_shrink_count", 0
            )
            > 0
        ),
        "wm_pbt_selector_changes_ranking": (
            by_profile.get("wm_pbt_source_selector", {}).get("wm_score_used_ticks", 0)
            > 0
        ),
        "rssm_mppi_actor_distill_has_teacher_path": bool(
            by_profile.get("rssm_mppi_actor_distill", {}).get(
                "profile_rssm_mppi_enabled_frac", {}
            ).get("last")
        ),
        "elite_bank_vs_planner_stability": {
            "elite_retention": by_profile.get("elite_bank_actor_distill", {}).get(
                "nsucc_retention_after_peak"
            ),
            "planner_retention": by_profile.get(
                "conservative_rssm_planner_distill", {}
            ).get("nsucc_retention_after_peak"),
        },
        "conservative_planner_no_rollout_apply": not any(
            v.get("residual_apply_enabled_last")
            for v in by_profile.values()
        ),
        "profile_diversity_collapse": (
            diversity_min is not None and diversity_min < 3
        ),
    }

    # --- Soft Activation Gate v1 questions (by-tol-bin) ---
    def _bin(prof: str, b: str) -> dict:
        return by_profile.get(prof, {}).get("by_tol_bin", {}).get(b, {})

    def _nonzero_above_004(prof: str) -> bool:
        # WM has nonzero-but-small activation while tol > 0.04?
        for b in ("gt_006", "005_006", "004_005"):
            w = _bin(prof, b).get("actor_distill_activation_weight_mean")
            if w is not None and w > 0.0:
                return True
        return False

    def _activation_increases_as_tol_shrinks(prof: str) -> bool:
        seq = []
        for b in ("gt_006", "005_006", "004_005", "le_004"):
            w = _bin(prof, b).get("actor_distill_activation_weight_mean")
            if w is not None:
                seq.append(w)
        return len(seq) >= 2 and seq[-1] >= seq[0]

    soft_gate_questions = {
        "soft_gate_nonzero_above_tol_004": {
            prof: _nonzero_above_004(prof)
            for prof in (
                "rssm_mppi_actor_distill",
                "elite_bank_actor_distill",
                "conservative_rssm_planner_distill",
            )
        },
        "activation_increases_as_tol_shrinks": {
            prof: _activation_increases_as_tol_shrinks(prof)
            for prof in (
                "rssm_mppi_actor_distill",
                "elite_bank_actor_distill",
                "conservative_rssm_planner_distill",
            )
        },
        "wm_curriculum_blocks_before_floor": {
            prof: any(
                _bin(prof, b).get("wm_curriculum_blocked_shrink_count", 0) > 0
                for b in ("gt_006", "005_006", "004_005")
            )
            for prof in ("wm_curriculum_controller",)
        },
        "wm_source_used_by_tol_bin": {
            b: _bin("wm_pbt_source_selector", b).get("wm_score_used_fraction")
            for b in TOL_BINS
        },
    }
    questions["soft_activation_gate"] = soft_gate_questions

    retention_diagnosis = {
        "tol_exposure_by_profile": {
            prof: data.get("tol_exposure", {}) for prof, data in by_profile.items()
        },
        "shrink_safety_by_profile": {
            prof: data.get("shrink_safety", {}) for prof, data in by_profile.items()
        },
        "causal_diagnosis": _causal_diagnosis(
            by_profile, all_curves, policy_profiles
        ),
    }
    wm_imagined_diagnostic = _wm_imagined_head_diagnostic(
        all_curves, policy_profiles
    )

    return {
        "run_dir": str(run_dir),
        "workspace": str(ws),
        "run_config": run_cfg,
        "policy_profiles": policy_profiles,
        "by_profile": dict(by_profile),
        "questions": questions,
        "retention_collapse_diagnosis": retention_diagnosis,
        "wm_imagined_head_diagnostic": wm_imagined_diagnostic,
    }


def write_reports(report: dict[str, Any], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "PBT_WM_STRATEGY_ZOO_FULL_REPORT.json"
    md_path = out_dir / "PBT_WM_STRATEGY_ZOO_FULL_REPORT.md"
    json_path.write_text(json.dumps(report, indent=2))

    lines = [
        "# PBT-WM Strategy Zoo Full Report",
        "",
        f"Run: `{report.get('run_dir')}`",
        "",
        "## Profile semantics",
        "",
        "`ppo_only` is the PPO/PBT anchor under the shared strategy-zoo "
        "PBT/curriculum scaffold. It disables WM/EIR/action-level intervention, "
        "but it is **not** an exact original DexPBT reproduction.",
        "",
        "## Per-profile summary",
        "",
    ]
    for prof, data in sorted(report.get("by_profile", {}).items()):
        ns = data.get("nsucc_mean", {})
        lines.append(f"### `{prof}` (policy {data.get('policy_idx')})")
        lines.append(
            f"- latest nsucc_mean: {ns.get('last')} | peak: {ns.get('peak')}"
        )
        lines.append(
            f"- retention after peak: {data.get('nsucc_retention_after_peak')}"
        )
        lines.append(
            f"- wm curriculum blocked shrink: "
            f"{data.get('wm_curriculum_blocked_shrink_count', 0)}"
        )
        lines.append(
            f"- profile copies: {data.get('pbt_profile_copy_count', 0)}"
        )
        lines.append(
            f"- residual apply (last): {data.get('residual_apply_enabled_last')}"
        )
        btb = data.get("by_tol_bin", {})
        if btb:
            lines.append("- by tol bin (active_frac / activation_w / wm_score_used / blocked / collapse):")
            for b in TOL_BINS:
                s = btb.get(b, {})
                lines.append(
                    f"    - `{b}` (n={s.get('n', 0)}): "
                    f"adf={s.get('actor_distill_active_fraction')} "
                    f"aw={s.get('actor_distill_activation_weight_mean')} "
                    f"wm_used={s.get('wm_score_used_fraction')} "
                    f"blocked={s.get('wm_curriculum_blocked_shrink_count', 0)} "
                    f"collapse={s.get('collapse_count', 0)}"
                )
        lines.append("")

    q = report.get("questions", {})
    sag = q.get("soft_activation_gate", {})
    lines.extend(
        [
            "## Questions",
            "",
            f"1. Most stable profile: **{q.get('most_stable_profile')}**",
            f"2. Highest peak profile: **{q.get('highest_peak_profile')}**",
            f"3. WM curriculum controller active blocks: **{q.get('wm_curriculum_reduces_collapse')}**",
            f"4. WM PBT selector used WM score: **{q.get('wm_pbt_selector_changes_ranking')}**",
            f"5. RSSM-MPPI actor distill path enabled: **{q.get('rssm_mppi_actor_distill_has_teacher_path')}**",
            f"6. Elite vs planner retention: **{q.get('elite_bank_vs_planner_stability')}**",
            f"7. No rollout residual apply: **{q.get('conservative_planner_no_rollout_apply')}**",
            f"8. Profile diversity collapse (<3): **{q.get('profile_diversity_collapse')}**",
            "",
            "## Soft Activation Gate v1",
            "",
            f"- WM has nonzero activation while tol > 0.04: **{sag.get('soft_gate_nonzero_above_tol_004')}**",
            f"- Activation increases as tol shrinks: **{sag.get('activation_increases_as_tol_shrinks')}**",
            f"- WM curriculum blocks before floor stage: **{sag.get('wm_curriculum_blocks_before_floor')}**",
            f"- WM source score used by tol bin: **{sag.get('wm_source_used_by_tol_bin')}**",
            "",
            "Residual rollout apply remains HARD-gated (not relaxed by the soft gate).",
            "",
        ]
    )
    md_path.write_text("\n".join(lines))

    diag = report.get("retention_collapse_diagnosis", {})
    diag_json = out_dir / "PBT_WM_RETENTION_COLLAPSE_DIAGNOSIS.json"
    diag_md = out_dir / "PBT_WM_RETENTION_COLLAPSE_DIAGNOSIS.md"
    diag_json.write_text(json.dumps(diag, indent=2))
    causal = diag.get("causal_diagnosis", {})
    diag_lines = [
        "# PBT-WM Retention Collapse Diagnosis",
        "",
        f"Run: `{report.get('run_dir')}`",
        "",
        "## Causal diagnosis (8 questions)",
        "",
        f"1. WM workers enter tol<=0.03 earlier than ppo_only: **{causal.get('wm_workers_enter_tol_le_003_earlier_than_ppo')}**",
        f"2. Danger zone blocked_shrink nonzero: **{causal.get('danger_zone_blocked_shrink_nonzero')}**",
        f"3. Strong shrink blocked on low global retention: **{causal.get('strong_shrink_blocked_on_low_global_retention')}**",
        f"4. Rollback to last mastered occurred: **{causal.get('rollback_to_last_mastered_occurred')}**",
        f"5. p0 survivorship bias (<=2 ticks at <=0.02): **{causal.get('ppo_high_retention_survivorship_bias')}**",
        f"6. WM imagined still mostly invalid: **{causal.get('wm_imagined_still_mostly_invalid')}**",
        f"7. Real fallback took over when imagined invalid: **{causal.get('real_fallback_took_over_when_imagined_invalid')}**",
        f"8. Exposure ticks at <=0.02: **{causal.get('exposure_ticks_le_002')}**",
        "",
    ]
    diag_md.write_text("\n".join(diag_lines))

    wm_diag = report.get("wm_imagined_head_diagnostic", {})
    wm_json = out_dir / "WM_IMAGINED_HEAD_DIAGNOSTIC.json"
    wm_md = out_dir / "WM_IMAGINED_HEAD_DIAGNOSTIC.md"
    wm_json.write_text(json.dumps(wm_diag, indent=2))
    wm_lines = [
        "# WM Imagined Head Diagnostic",
        "",
        f"Run: `{report.get('run_dir')}`",
        "",
        "## Per-policy summary",
        "",
    ]
    for pid, data in sorted(wm_diag.get("by_policy", {}).items()):
        wm_lines.append(
            f"- policy {pid} ({data.get('profile')}): "
            f"valid_frac={data.get('imagined_valid_fraction'):.3f} "
            f"unc_mean={data.get('uncertainty', {}).get('mean')} "
            f"counterfactual_real_block={data.get('counterfactual_would_block_by_real_fallback')}"
        )
    wm_md.write_text("\n".join(wm_lines))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--ws", type=str, default=None)
    ap.add_argument("--num-policies", type=int, default=6)
    args = ap.parse_args()
    report = analyze_run(
        args.run_dir, ws_name=args.ws, num_policies=args.num_policies
    )
    out = args.out or args.run_dir
    write_reports(report, out)
    print(f"Wrote {out / 'PBT_WM_STRATEGY_ZOO_FULL_REPORT.json'}")


if __name__ == "__main__":
    main()
