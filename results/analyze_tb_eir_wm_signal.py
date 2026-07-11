#!/usr/bin/env python3
"""TensorBoard + JSONL diagnostic report for WM head / EIR / curriculum signals."""
from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import subprocess
import warnings
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with open(path, encoding="utf-8") as fh:
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


def _percentile(vals: list[float], p: float) -> float | None:
    if not vals:
        return None
    s = sorted(vals)
    if len(s) == 1:
        return s[0]
    k = (len(s) - 1) * p / 100.0
    lo = int(math.floor(k))
    hi = int(math.ceil(k))
    if lo == hi:
        return s[lo]
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def _scalar_summary(vals: list[float]) -> dict[str, Any]:
    if not vals:
        return {
            "n": 0,
            "mean": None,
            "median": None,
            "p90": None,
            "max": None,
            "last": None,
        }
    return {
        "n": len(vals),
        "mean": statistics.fmean(vals),
        "median": statistics.median(vals),
        "p90": _percentile(vals, 90),
        "max": max(vals),
        "last": vals[-1],
    }


def _fraction_in_ranges(vals: list[float], lo: float, hi: float) -> float | None:
    if not vals:
        return None
    return sum(1 for v in vals if lo <= v <= hi) / len(vals)


def _fraction_below(vals: list[float], thr: float) -> float | None:
    if not vals:
        return None
    return sum(1 for v in vals if v < thr) / len(vals)


def _fraction_above(vals: list[float], thr: float) -> float | None:
    if not vals:
        return None
    return sum(1 for v in vals if v > thr) / len(vals)


def _head_loss_verdict(
    too_low: float | None, healthy: float | None, too_high: float | None
) -> str:
    if too_low is None and healthy is None and too_high is None:
        return "no_data"
    tl = too_low or 0.0
    h = healthy or 0.0
    th = too_high or 0.0
    if h >= 0.5:
        return "healthy"
    if tl >= 0.5:
        return "too_weak"
    if th >= 0.25:
        return "may_dominate"
    if h >= 0.25:
        return "mixed_healthy"
    return "uncertain"


def _eir_chain_verdict(
    cand_frac: float | None, sel_frac: float | None, distill_mean: float | None
) -> str:
    cf = cand_frac or 0.0
    sf = sel_frac or 0.0
    dm = distill_mean or 0.0
    if cf <= 0.01:
        return "case3_no_candidates"
    if sf <= 0.01:
        return "case1_pool_no_selection"
    if dm <= 0.01:
        return "case2_selected_no_distill"
    return "case4_active"


# ---------------------------------------------------------------------------
# Event file discovery
# ---------------------------------------------------------------------------

def find_event_files(run_dir: Path) -> tuple[list[str], list[str]]:
    """Return (event_paths, warnings)."""
    warns: list[str] = []
    paths: set[str] = set()
    search_roots = [
        run_dir,
        Path("/root/autodl-tmp/wmdexpbt/runtime_root/outputs"),
        Path("/root/autodl-tmp/wmdexpbt/results/runs"),
    ]
    run_token = run_dir.name
    for root in search_roots:
        if not root.exists():
            continue
        try:
            for p in root.rglob("events.out.tfevents.*"):
                sp = str(p.resolve())
                if root == run_dir or run_token in sp:
                    paths.add(sp)
        except OSError as exc:
            warns.append(f"scan failed for {root}: {exc}")

    # Also run find for robustness (user-requested pattern)
    for pattern in [
        f'find "{run_dir}" -name "events.out.tfevents.*" 2>/dev/null',
        'find /root/autodl-tmp/wmdexpbt/runtime_root/outputs -name "events.out.tfevents.*" 2>/dev/null',
        f'find /root/autodl-tmp/wmdexpbt/results/runs -path "*{run_token[-6:]}*" -name "events.out.tfevents.*" 2>/dev/null',
    ]:
        try:
            out = subprocess.check_output(pattern, shell=True, text=True)
            for line in out.splitlines():
                line = line.strip()
                if line:
                    paths.add(str(Path(line).resolve()))
        except subprocess.CalledProcessError:
            pass

    return sorted(paths), warns


def _policy_from_event_path(path: str) -> int | None:
    m = re.search(r"policy_(\d+)", path)
    if m:
        return int(m.group(1))
    return None


def group_events_by_policy(event_files: list[str]) -> dict[int, list[str]]:
    grouped: dict[int, list[str]] = defaultdict(list)
    for p in event_files:
        idx = _policy_from_event_path(p)
        if idx is not None:
            grouped[idx].append(p)
    for idx in grouped:
        grouped[idx] = sorted(grouped[idx])
    return dict(grouped)


# ---------------------------------------------------------------------------
# TensorBoard reading
# ---------------------------------------------------------------------------

TB_AVAILABLE = False
try:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    TB_AVAILABLE = True
except ImportError:
    pass


def _resolve_tag(requested: str, available: set[str], fuzzy: list[str]) -> tuple[str | None, str | None]:
    """Return (matched_tag, match_mode)."""
    if requested in available:
        return requested, "exact"
    req_tail = requested.split("/")[-1]
    for tag in available:
        if tag.endswith(req_tail) or tag == requested:
            return tag, "suffix"
    for pat in fuzzy:
        matches = sorted(t for t in available if pat in t.lower())
        if len(matches) == 1:
            return matches[0], f"fuzzy:{pat}"
        if len(matches) > 1:
            # prefer shortest / most specific
            best = min(matches, key=lambda t: (len(t), t))
            return best, f"fuzzy_multi:{pat}"
    return None, None


TAG_ALIASES: dict[str, list[str]] = {
    "weighted_head_loss_fraction": [
        "seq_wm_head/weighted_head_loss_fraction",
        "wm_rssm/seq_wm_head/head_loss_weighted_fraction",
        "wm_rssm/head_loss_weighted_fraction",
    ],
    "success_label_pos_frac": [
        "seq_wm_head/success_label_pos_frac",
        "wm_rssm/seq_wm_head/success_label_pos_frac",
        "wm_rssm/success_label_pos_frac",
    ],
    "drop_label_pos_frac": [
        "seq_wm_head/drop_label_pos_frac",
        "wm_rssm/seq_wm_head/drop_label_pos_frac",
        "wm_rssm/drop_label_pos_frac",
    ],
    "recovery_label_pos_frac": [
        "seq_wm_head/recovery_label_pos_frac",
        "wm_rssm/seq_wm_head/recovery_label_pos_frac",
        "wm_rssm/recovery_label_pos_frac",
    ],
    "success_logit_abs_mean": [
        "seq_wm_head/success_logit_abs_mean",
        "wm_rssm/seq_wm_head/success_logit_abs_mean",
    ],
    "drop_logit_abs_mean": [
        "seq_wm_head/drop_logit_abs_mean",
        "wm_rssm/seq_wm_head/drop_logit_abs_mean",
    ],
    "recovery_logit_abs_mean": [
        "seq_wm_head/recovery_logit_abs_mean",
        "wm_rssm/seq_wm_head/recovery_logit_abs_mean",
    ],
    "success_loss_weighted": [
        "seq_wm_head/success_loss_weighted",
        "wm_rssm/seq_wm_head/success_loss_weighted",
    ],
    "drop_loss_weighted": [
        "seq_wm_head/drop_loss_weighted",
        "wm_rssm/seq_wm_head/drop_loss_weighted",
    ],
    "recovery_loss_weighted": [
        "seq_wm_head/recovery_loss_weighted",
        "wm_rssm/seq_wm_head/recovery_loss_weighted",
    ],
    "loss_total": [
        "seq_wm/loss_total",
        "wm_rssm/loss_total",
    ],
    "loss_dynamics_or_recon": [
        "seq_wm/loss_dynamics_or_recon",
        "seq_wm_head/loss_dynamics_or_recon",
        "wm_rssm/seq_wm_head/loss_dynamics_or_recon",
    ],
}

EIR_EXACT_TAGS = [
    "eir/elite_candidate_gate_weight",
    "eir/rssm_candidate_gate_weight",
    "eir/elite_candidate_retrieval_allowed",
    "eir/rssm_candidate_retrieval_allowed",
    "eir/elite_candidate_count",
    "eir/archive_candidate_count",
    "eir/rssm_candidate_added",
    "eir/elite_teacher_candidate_available_above_004",
    "eir/rssm_teacher_candidate_available_above_004",
    "eir/any_teacher_candidate_available_above_004",
    "eir/teacher_candidate_available_above_004",
    "eir/teacher_selected_above_004",
    "eir/best_source_frac_elite",
    "eir/best_source_frac_archive",
    "eir/best_source_frac_rssm",
    "eir/teacher_margin_above_004",
    "eir/actor_distill_activation_nonzero_frac",
    "eir/actor_distill_activation_weight_mean",
]

EIR_FUZZY = [
    "candidate_gate",
    "teacher_candidate",
    "teacher_selected",
    "best_source_frac",
    "actor_distill",
    "rssm_candidate",
    "elite_candidate",
]

PBT_WM_TAGS = [
    "pbt/wm_score_used",
    "pbt/wm_source_activation_weight",
    "pbt/wm_source_activation_nonzero",
    "pbt/wm_source_unc_weight",
    "pbt/wm_source_time_weight",
    "pbt/wm_shadow_observations",
    "pbt/wm_shadow_changed_selection",
    "pbt/wm_changed_source_selection",
]

WM_HEAD_REQUESTED = list(TAG_ALIASES.keys())


def _build_tag_map(
    requested_tags: list[str],
    available: set[str],
    fuzzy_patterns: list[str] | None = None,
    allow_fuzzy: bool = True,
) -> dict[str, str]:
    tag_map: dict[str, str] = {}
    fuzzy = fuzzy_patterns or []
    for req in requested_tags:
        matched = None
        if req in available:
            tag_map[req] = req
            continue
        aliases = TAG_ALIASES.get(req, [req])
        for alias in aliases:
            matched, _ = _resolve_tag(alias, available, [] if not allow_fuzzy else fuzzy)
            if matched:
                break
        if not matched and allow_fuzzy:
            matched, _ = _resolve_tag(req, available, fuzzy)
        tag_map[req] = matched or ""
    return tag_map


def load_policy_tb_bundle(
    event_paths: list[str],
) -> tuple[dict[str, list[float]], dict[str, str], list[str]]:
    """Load all WM/EIR/PBT scalars: discover tags globally, read once per file."""
    load_warns: list[str] = []
    if not TB_AVAILABLE:
        return {}, {}, ["tensorboard package unavailable"]
    if not event_paths:
        return {}, {}, ["no event files for policy"]

    available: set[str] = set()
    for ep in event_paths:
        try:
            ea = EventAccumulator(ep, size_guidance={"scalars": 0})
            ea.Reload()
            available.update(ea.Tags().get("scalars", []))
        except Exception as exc:
            load_warns.append(f"tag scan failed {ep}: {exc}")

    wm_map = _build_tag_map(WM_HEAD_REQUESTED, available, allow_fuzzy=False)
    eir_map = _build_tag_map(EIR_EXACT_TAGS, available, EIR_FUZZY, allow_fuzzy=False)
    pbt_map = _build_tag_map(PBT_WM_TAGS, available, allow_fuzzy=False)
    logical_to_actual = {**wm_map, **eir_map, **pbt_map}

    merged: dict[str, list[tuple[int, float]]] = defaultdict(list)
    for ep in event_paths:
        try:
            ea = EventAccumulator(ep, size_guidance={"scalars": 0})
            ea.Reload()
            tags = set(ea.Tags().get("scalars", []))
            for logical, actual in logical_to_actual.items():
                if not actual or actual not in tags:
                    continue
                for ev in ea.Scalars(actual):
                    merged[logical].append((int(ev.step), float(ev.value)))
        except Exception as exc:
            load_warns.append(f"scalar read failed {ep}: {exc}")

    out: dict[str, list[float]] = {}
    for logical, pts in merged.items():
        pts.sort(key=lambda x: x[0])
        out[logical] = [v for _, v in pts]
    return out, logical_to_actual, load_warns


def _parse_run_config(run_dir: Path) -> dict[str, str]:
    cfg: dict[str, str] = {}
    path = run_dir / "RUN_CONFIG.txt"
    if not path.exists():
        return cfg
    for line in path.read_text(encoding="utf-8").splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            cfg[k.strip()] = v.strip()
    return cfg


def _git_commit(run_dir: Path, run_cfg: dict[str, str]) -> str:
    commit = run_cfg.get("commit", "unknown")
    if commit and commit != "unknown":
        return commit
    try:
        root = run_dir
        while root != root.parent:
            if (root / ".git").exists():
                out = subprocess.check_output(
                    ["git", "-C", str(root), "rev-parse", "HEAD"],
                    text=True,
                ).strip()
                return out[:12] if out else "unknown"
            root = root.parent
    except Exception:
        pass
    return "unknown"


# ---------------------------------------------------------------------------
# JSONL aggregations
# ---------------------------------------------------------------------------

def aggregate_training_curve(rows: list[dict]) -> dict[str, Any]:
    if not rows:
        return {}
    last = rows[-1]
    profile = last.get("wm_strategy_profile_name", "unknown")
    frames = int(last.get("env_frames", 0))

    def _count_bool(key: str) -> int:
        return sum(1 for r in rows if r.get(key))

    def _count_truthy(key: str) -> int:
        return sum(1 for r in rows if r.get(key) in (True, 1, 1.0))

    def _vals(key: str) -> list[float]:
        return [v for r in rows if (v := _safe_float(r.get(key))) is not None]

    fallback = Counter(str(r.get("wm_fallback_reason", "")) for r in rows if r.get("wm_fallback_reason"))
    gate_mode = Counter(str(r.get("wm_ranking_uncertainty_gate_mode", "")) for r in rows if r.get("wm_ranking_uncertainty_gate_mode"))

    strict_valid = _count_bool("wm_imagined_valid")  # jsonl uses combined valid
    soft_valid = sum(
        1 for r in rows
        if _safe_float(r.get("wm_validity_weight")) == 0.25
        or r.get("wm_score_used_soft") in (True, 1, 1.0)
    )
    strict_used = sum(
        1 for r in rows
        if r.get("wm_score_used") and (
            _safe_float(r.get("wm_validity_weight")) == 1.0
            or r.get("wm_score_used_strict") in (True, 1, 1.0)
        )
    )
    soft_used = sum(
        1 for r in rows
        if r.get("wm_score_used") and (
            _safe_float(r.get("wm_validity_weight")) == 0.25
            or r.get("wm_score_used_soft") in (True, 1, 1.0)
        )
    )

    block_reasons = Counter(
        str(r.get("curriculum_safety_block_reason", ""))
        for r in rows
        if r.get("curriculum_safety_block_reason")
    )
    wm_ctrl_reasons = Counter(
        str(r.get("wm_curriculum_controller_reason", ""))
        for r in rows
        if r.get("wm_curriculum_controller_reason")
    )

    retention = _vals("curriculum_safety_global_retention")
    slope = _vals("curriculum_safety_recent_slope")
    ret_summary = _scalar_summary(retention)
    if retention:
        ret_summary["min"] = min(retention)
    slope_summary = _scalar_summary(slope)
    if slope:
        slope_summary["min"] = min(slope)

    return {
        "profile": profile,
        "frames": frames,
        "n_ticks": len(rows),
        "cross_restart_count": last.get("cross_restart_count"),
        "restart_count": last.get("restart_count"),
        "wm_score_used_count": _count_bool("wm_score_used"),
        "wm_score_used_strict_count": strict_used,
        "wm_score_used_soft_count": soft_used,
        "strict_valid_count": strict_valid,
        "soft_valid_count": soft_valid,
        "wm_score_effective_weight": _scalar_summary(_vals("wm_score_effective_weight")),
        "wm_source_activation_weight": _scalar_summary(_vals("wm_source_activation_weight")),
        "fallback_reason_counts": dict(fallback),
        "gate_mode_counts": dict(gate_mode),
        "curriculum_safety_blocked_shrink_count": _count_bool("curriculum_safety_blocked_shrink"),
        "curriculum_safety_real_risk_count": _count_bool("curriculum_safety_real_risk"),
        "curriculum_safety_wm_risk_count": _count_bool("curriculum_safety_wm_risk"),
        "curriculum_safety_danger_zone_count": _count_bool("curriculum_safety_danger_zone"),
        "curriculum_safety_repeated_collapse_count": sum(
            int(r.get("curriculum_safety_repeated_collapse_count") or 0) for r in rows
        ),
        "curriculum_rollback_count": _count_bool("curriculum_rollback_triggered"),
        "wm_curriculum_blocked_shrink_count": _count_bool("wm_curriculum_blocked_shrink"),
        "block_reason_counts": dict(block_reasons),
        "wm_curriculum_controller_reason_counts": dict(wm_ctrl_reasons),
        "global_retention": ret_summary,
        "recent_slope": slope_summary,
        "nsucc_mean_last": _safe_float(last.get("nsucc_mean")),
        "nsucc_max_last": _safe_float(last.get("nsucc_max")),
    }


def aggregate_curriculum_events(path: Path, policy_idx: int) -> dict[str, Any]:
    rows = _read_jsonl(path)
    pol_rows = [r for r in rows if int(r.get("policy_idx", -1)) == policy_idx]
    return {
        "total_events": len(pol_rows),
        "event_counts": dict(Counter(str(r.get("event", "")) for r in pol_rows)),
    }


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------


def analyze_run(run_dir: Path) -> dict[str, Any]:
    run_dir = run_dir.resolve()
    metrics_dir = run_dir / "dexpbt_8gpu_ws" / "metrics"
    run_cfg = _parse_run_config(run_dir)
    commit = _git_commit(run_dir, run_cfg)

    event_files, scan_warns = find_event_files(run_dir)
    by_policy = group_events_by_policy(event_files)

    policy_profiles: dict[str, str] = {}
    for i in range(6):
        key = f"policy_{i:02d}_profile"
        if key in run_cfg:
            policy_profiles[str(i)] = run_cfg[key]

    frame_coverage: dict[str, int] = {}
    training_curves: dict[str, dict] = {}
    for i in range(6):
        curve_path = metrics_dir / f"training_curve_policy_{i:02d}.jsonl"
        rows = _read_jsonl(curve_path)
        training_curves[str(i)] = aggregate_training_curve(rows)
        if rows:
            frame_coverage[str(i)] = int(rows[-1].get("env_frames", 0))

    missing_tags: list[str] = []
    tb_warnings: list[str] = scan_warns
    wm_head_loss: dict[str, Any] = {}
    eir_teacher_supply: dict[str, Any] = {}
    wm_source_selector: dict[str, Any] = {}
    tb_tag_matches: dict[str, dict[str, str]] = {}

    for i in range(6):
        pol_key = str(i)
        ev_paths = by_policy.get(i, [])
        scalars, tag_map, warns = load_policy_tb_bundle(ev_paths)
        tb_warnings.extend(warns)
        tb_tag_matches[pol_key] = tag_map

        whlf = scalars.get("weighted_head_loss_fraction", [])
        head_entry = {
            "profile": policy_profiles.get(pol_key, training_curves.get(pol_key, {}).get("profile")),
            "frames": frame_coverage.get(pol_key),
            "event_files": ev_paths,
            "weighted_head_loss_fraction": _scalar_summary(whlf),
            "too_low_fraction": _fraction_below(whlf, 0.01),
            "healthy_fraction": _fraction_in_ranges(whlf, 0.05, 0.30),
            "too_high_fraction": _fraction_above(whlf, 0.40),
            "verdict": _head_loss_verdict(
                _fraction_below(whlf, 0.01),
                _fraction_in_ranges(whlf, 0.05, 0.30),
                _fraction_above(whlf, 0.40),
            ),
            "matched_tags": {
                k: tag_map[k]
                for k in WM_HEAD_REQUESTED
                if tag_map.get(k)
            },
            "extra_scalars": {
                k: _scalar_summary(scalars[k])
                for k in scalars
                if k in WM_HEAD_REQUESTED
                and k != "weighted_head_loss_fraction"
                and scalars[k]
            },
        }
        if not whlf:
            missing_tags.append(f"p{pol_key}:weighted_head_loss_fraction")
        wm_head_loss[pol_key] = head_entry

        eir_scalars = {k: scalars.get(k, []) for k in EIR_EXACT_TAGS}
        cand_key = "eir/any_teacher_candidate_available_above_004"
        if not eir_scalars.get(cand_key):
            for alt in (
                "eir/teacher_candidate_available_above_004",
                "eir/elite_teacher_candidate_available_above_004",
            ):
                if scalars.get(alt):
                    eir_scalars[cand_key] = scalars[alt]
                    break

        cand = eir_scalars.get(cand_key, [])
        sel = eir_scalars.get("eir/teacher_selected_above_004", [])
        distill = eir_scalars.get("eir/actor_distill_activation_nonzero_frac", [])
        distill_w = eir_scalars.get("eir/actor_distill_activation_weight_mean", [])

        elite_gate = eir_scalars.get("eir/elite_candidate_gate_weight", [])
        rssm_gate = eir_scalars.get("eir/rssm_candidate_gate_weight", [])
        elite_allowed = eir_scalars.get("eir/elite_candidate_retrieval_allowed", [])
        rssm_allowed = eir_scalars.get("eir/rssm_candidate_retrieval_allowed", [])

        eir_entry = {
            "profile": policy_profiles.get(pol_key),
            "event_files": ev_paths,
            "matched_tags": {k: tag_map[k] for k in EIR_EXACT_TAGS if tag_map.get(k)},
            "candidate_gate_weight": {
                "elite_mean": _scalar_summary(elite_gate).get("mean"),
                "elite_last": _scalar_summary(elite_gate).get("last"),
                "rssm_mean": _scalar_summary(rssm_gate).get("mean"),
                "rssm_last": _scalar_summary(rssm_gate).get("last"),
            },
            "retrieval_allowed_fraction": {
                "elite": _fraction_above(elite_allowed, 0.5),
                "rssm": _fraction_above(rssm_allowed, 0.5),
            },
            "candidate_count_mean": _scalar_summary(
                eir_scalars.get("eir/elite_candidate_count", [])
                + eir_scalars.get("eir/archive_candidate_count", [])
            ).get("mean"),
            "candidate_available_frac": _fraction_above(cand, 0.5),
            "teacher_selected_frac": _fraction_above(sel, 0.5),
            "actor_distill_nonzero_mean": _scalar_summary(distill).get("mean"),
            "actor_distill_nonzero_last": _scalar_summary(distill).get("last"),
            "actor_distill_weight_mean": _scalar_summary(distill_w).get("mean"),
            "actor_distill_weight_last": _scalar_summary(distill_w).get("last"),
            "best_source_frac": {
                "elite_mean": _scalar_summary(eir_scalars.get("eir/best_source_frac_elite", [])).get("mean"),
                "archive_mean": _scalar_summary(eir_scalars.get("eir/best_source_frac_archive", [])).get("mean"),
                "rssm_mean": _scalar_summary(eir_scalars.get("eir/best_source_frac_rssm", [])).get("mean"),
            },
            "verdict": _eir_chain_verdict(
                _fraction_above(cand, 0.5),
                _fraction_above(sel, 0.5),
                _scalar_summary(distill).get("mean"),
            ),
            "scalar_summaries": {
                k: _scalar_summary(v) for k, v in eir_scalars.items() if v
            },
        }
        if not any(eir_scalars.values()) and pol_key in ("1", "4", "5"):
            missing_tags.append(f"p{pol_key}:eir/* (no EIR scalars in TB)")
        eir_teacher_supply[pol_key] = eir_entry

        pbt_scalars = {k: scalars.get(k, []) for k in PBT_WM_TAGS}
        tc = training_curves.get(pol_key, {})
        wm_used_tb = pbt_scalars.get("pbt/wm_score_used", [])
        wm_source_selector[pol_key] = {
            "profile": policy_profiles.get(pol_key),
            "from_training_curve": {
                k: tc.get(k)
                for k in (
                    "wm_score_used_count",
                    "wm_score_used_strict_count",
                    "wm_score_used_soft_count",
                    "strict_valid_count",
                    "soft_valid_count",
                    "wm_score_effective_weight",
                    "wm_source_activation_weight",
                    "fallback_reason_counts",
                    "gate_mode_counts",
                )
            },
            "tb_wm_score_used_fraction": _fraction_above(wm_used_tb, 0.5),
            "tb_wm_source_activation_weight": _scalar_summary(
                pbt_scalars.get("pbt/wm_source_activation_weight", [])
            ),
            "tb_matched_tags": {k: tag_map[k] for k in PBT_WM_TAGS if tag_map.get(k)},
            "verdict": (
                "sustained_active"
                if (tc.get("wm_score_used_count") or 0) >= 10
                and (_scalar_summary(_vals_from_tc(tc, "wm_score_effective_weight")).get("mean") or 0) > 0.05
                else (
                    "episodic"
                    if (tc.get("wm_score_used_count") or 0) > 0
                    else "inactive"
                )
            ),
        }

    # p2 curriculum deep dive
    p2_curve = training_curves.get("2", {})
    p2_events = aggregate_curriculum_events(
        metrics_dir / "curriculum_safety_events.jsonl", 2
    )
    p2_curriculum = {
        **p2_curve,
        "curriculum_safety_events": p2_events,
        "interpretation": _p2_interpretation(p2_curve, p2_events),
    }

    verdicts = {
        "head_loss_scale": _overall_head_verdict(wm_head_loss),
        "eir_teacher_supply": {
            pol: eir_teacher_supply[pol]["verdict"] for pol in ("1", "4", "5")
        },
        "p3_wm_source_selector": wm_source_selector.get("3", {}).get("verdict", "unknown"),
        "p2_curriculum_driver": p2_curriculum.get("interpretation", "unknown"),
    }

    recommendations = _build_recommendations(
        wm_head_loss, eir_teacher_supply, wm_source_selector, p2_curriculum, verdicts
    )

    return {
        "run_dir": str(run_dir),
        "commit": commit,
        "tb_available": TB_AVAILABLE,
        "event_files": event_files,
        "event_files_by_policy": {str(k): v for k, v in by_policy.items()},
        "policy_profiles": policy_profiles,
        "frame_coverage": frame_coverage,
        "wm_head_loss": wm_head_loss,
        "wm_source_selector": wm_source_selector,
        "eir_teacher_supply": eir_teacher_supply,
        "p2_curriculum": p2_curriculum,
        "training_curves_summary": training_curves,
        "verdicts": verdicts,
        "missing_tags": sorted(set(missing_tags)),
        "tb_tag_matches": tb_tag_matches,
        "warnings": sorted(set(tb_warnings)),
        "recommendations": recommendations,
    }


def _vals_from_tc(tc: dict, key: str) -> list[float]:
    block = tc.get(key)
    if isinstance(block, dict):
        m = block.get("mean")
        return [m] if m is not None else []
    return []


def _overall_head_verdict(wm_head_loss: dict) -> str:
    verdicts = [v.get("verdict") for v in wm_head_loss.values() if v.get("verdict") not in (None, "no_data")]
    if not verdicts:
        return "no_data"
    healthy = sum(1 for v in verdicts if v == "healthy")
    if healthy >= 4:
        return "mostly_healthy"
    weak = sum(1 for v in verdicts if v == "too_weak")
    if weak >= 4:
        return "mostly_too_weak"
    return "mixed"


def _p2_interpretation(p2: dict, events: dict) -> str:
    real = p2.get("curriculum_safety_real_risk_count", 0)
    wm = p2.get("curriculum_safety_wm_risk_count", 0)
    danger = p2.get("curriculum_safety_danger_zone_count", 0)
    wm_block = p2.get("wm_curriculum_blocked_shrink_count", 0)
    real_events = events.get("event_counts", {}).get("real_signal_block", 0)
    restarts = p2.get("cross_restart_count", 0)

    if real >= wm and real_events >= wm:
        driver = "real_risk_dominant"
    elif wm > real * 1.5:
        driver = "wm_risk_dominant"
    else:
        driver = "mixed_real_and_wm"

    if danger > wm_block:
        driver += "_with_danger_zone"
    if restarts >= 3:
        driver += "_high_restarts"
    return driver


def _build_recommendations(
    wm_head: dict,
    eir: dict,
    wm_src: dict,
    p2: dict,
    verdicts: dict,
) -> list[str]:
    recs: list[str] = []
    if verdicts.get("head_loss_scale") == "mostly_healthy":
        recs.append("Continue run; WM head loss scale is mostly in the 0.05–0.30 healthy band.")
    elif verdicts.get("head_loss_scale") == "mostly_too_weak":
        recs.append("Consider increasing head loss lambdas; weighted_head_loss_fraction is mostly <0.01.")
    elif verdicts.get("head_loss_scale") == "mixed":
        recs.append("Head loss scale is policy-dependent; review per-policy TB before changing global lambdas.")

    for pol, label in (("1", "p1 rssm_mppi"), ("4", "p4 elite_bank"), ("5", "p5 conservative")):
        v = eir.get(pol, {}).get("verdict", "")
        if v == "case3_no_candidates":
            recs.append(f"{label}: candidate retrieval inactive — check bank/planner availability and gate weights.")
        elif v == "case1_pool_no_selection":
            recs.append(f"{label}: candidates enter pool but reranker rarely selects — review teacher margin / reranker.")
        elif v == "case2_selected_no_distill":
            recs.append(f"{label}: teachers selected but actor_distill signal near zero — review distill mask/loss gate.")
        elif v == "case4_active":
            recs.append(f"{label}: EIR teacher supply + actor_distill path active (TB confirmed).")

    p3v = verdicts.get("p3_wm_source_selector")
    if p3v == "sustained_active":
        recs.append("p3 WM source selector is sustained-active; no ranking-weight hotfix needed.")
    elif p3v == "episodic":
        recs.append("p3 WM ranking is episodic; monitor fallback_reason_counts and teacher promotions.")
    else:
        recs.append("p3 WM source selector inactive in JSONL — investigate wm_score_mode / imagined validity.")

    interp = p2.get("interpretation", "")
    if "real_risk_dominant" in interp:
        recs.append("p2 retention/restarts driven mainly by real-signal curriculum safety, not strict WM risk.")
    elif "wm_risk_dominant" in interp:
        recs.append("p2 issues driven mainly by WM-risk blocks — review wm_curriculum thresholds or soft_valid usage.")
    else:
        recs.append("p2 curriculum blocks are mixed real+WM; tune controller only after separating block_reason_counts.")

    if not recs:
        recs.append("Continue run and re-run this diagnostic after all policies reach 3B frames.")
    return recs


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------

def _fmt(v: Any, digits: int = 4) -> str:
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:.{digits}f}"
    if isinstance(v, dict):
        return json.dumps(v, ensure_ascii=False)
    return str(v)


def render_markdown(data: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# TB EIR/WM Signal Diagnosis\n")
    lines.append("## 0. Run metadata\n")
    lines.append(f"- **Run path:** `{data['run_dir']}`")
    lines.append(f"- **Commit SHA:** `{data.get('commit', 'unknown')}`")
    lines.append(f"- **TensorBoard available:** {data.get('tb_available')}")
    lines.append(f"- **Event files found:** {len(data.get('event_files', []))}")
    lines.append("\n### Event files read (by policy)\n")
    for pol in sorted(data.get("event_files_by_policy", {}), key=lambda x: int(x)):
        files = data["event_files_by_policy"][pol]
        lines.append(f"- **p{pol}** ({len(files)} files):")
        for f in files:
            lines.append(f"  - `{f}`")
    lines.append("\n### Policy / profile mapping\n")
    lines.append("| Policy | Profile | Frames |")
    lines.append("|--------|---------|--------|")
    for pol in sorted(data.get("policy_profiles", {}), key=lambda x: int(x)):
        prof = data["policy_profiles"][pol]
        frames = data.get("frame_coverage", {}).get(pol, 0)
        lines.append(f"| p{pol} | {prof} | {frames/1e9:.2f}B |")

    lines.append("\n## 1. Executive summary\n")
    for rec in _executive_bullets(data):
        lines.append(f"- {rec}")

    lines.append("\n## 2. WM head loss scale\n")
    lines.append(
        "| Policy | Profile | Frames | WHLF mean | last | max | healthy_frac | too_low_frac | too_high_frac | Verdict |"
    )
    lines.append("|--------|---------|--------|-----------|------|-----|--------------|--------------|---------------|---------|")
    for pol in sorted(data.get("wm_head_loss", {}), key=lambda x: int(x)):
        h = data["wm_head_loss"][pol]
        s = h.get("weighted_head_loss_fraction", {})
        lines.append(
            f"| p{pol} | {h.get('profile','')} | {(h.get('frames') or 0)/1e9:.2f}B "
            f"| {_fmt(s.get('mean'))} | {_fmt(s.get('last'))} | {_fmt(s.get('max'))} "
            f"| {_fmt(h.get('healthy_fraction'))} | {_fmt(h.get('too_low_fraction'))} "
            f"| {_fmt(h.get('too_high_fraction'))} | {h.get('verdict')} |"
        )

    lines.append("\n## 3. WM imagined / source selector signal\n")
    lines.append(
        "| Policy | strict_valid | soft_valid | wm_score_used | strict_used | soft_used | eff_w mean | eff_w max | fallback top | Verdict |"
    )
    lines.append("|--------|--------------|------------|---------------|-------------|-----------|------------|-----------|--------------|---------|")
    for pol in sorted(data.get("wm_source_selector", {}), key=lambda x: int(x)):
        tc = data["wm_source_selector"][pol].get("from_training_curve", {})
        fb = tc.get("fallback_reason_counts", {})
        fb_top = sorted(fb.items(), key=lambda x: -x[1])[:3]
        fb_str = ", ".join(f"{k}:{v}" for k, v in fb_top) if fb_top else "—"
        ew = tc.get("wm_score_effective_weight") or {}
        lines.append(
            f"| p{pol} | {tc.get('strict_valid_count','—')} | {tc.get('soft_valid_count','—')} "
            f"| {tc.get('wm_score_used_count','—')} | {tc.get('wm_score_used_strict_count','—')} "
            f"| {tc.get('wm_score_used_soft_count','—')} | {_fmt(ew.get('mean'))} | {_fmt(ew.get('max'))} "
            f"| {fb_str} | {data['wm_source_selector'][pol].get('verdict')} |"
        )

    lines.append("\n**p3 detail:** WM ranking uses strict path when `wm_validity_weight=1.0`; "
                 "`high_uncertainty` fallback should be absent post-hotfix. "
                 "Check `gate_mode_counts` and `fallback_reason_counts` in JSON.\n")

    lines.append("\n## 4. EIR teacher supply chain\n")
    lines.append(
        "| Policy | Profile | cand_avail_frac | cand_count_mean | teacher_sel_frac | distill_nonzero mean | distill_w mean | Verdict |"
    )
    lines.append("|--------|---------|-----------------|-----------------|------------------|----------------------|----------------|---------|")
    for pol in ("0", "1", "2", "3", "4", "5"):
        e = data.get("eir_teacher_supply", {}).get(pol, {})
        lines.append(
            f"| p{pol} | {e.get('profile','—')} | {_fmt(e.get('candidate_available_frac'))} "
            f"| {_fmt(e.get('candidate_count_mean'))} | {_fmt(e.get('teacher_selected_frac'))} "
            f"| {_fmt(e.get('actor_distill_nonzero_mean'))} | {_fmt(e.get('actor_distill_weight_mean'))} "
            f"| {e.get('verdict','—')} |"
        )

    lines.append("\n## 5. p2 curriculum controller diagnosis\n")
    p2 = data.get("p2_curriculum", {})
    lines.append(f"- **Interpretation:** {p2.get('interpretation')}")
    lines.append(f"- **cross_restart_count:** {p2.get('cross_restart_count')}")
    lines.append(f"- **block_reason_counts:** `{p2.get('block_reason_counts')}`")
    lines.append(f"- **real_risk ticks:** {p2.get('curriculum_safety_real_risk_count')}")
    lines.append(f"- **wm_risk ticks:** {p2.get('curriculum_safety_wm_risk_count')}")
    lines.append(f"- **danger_zone ticks:** {p2.get('curriculum_safety_danger_zone_count')}")
    lines.append(f"- **wm_curriculum_blocked_shrink ticks:** {p2.get('wm_curriculum_blocked_shrink_count')}")
    lines.append(f"- **rollback_count:** {p2.get('curriculum_rollback_count')}")
    gr = p2.get("global_retention") or {}
    sl = p2.get("recent_slope") or {}
    lines.append(
        f"- **global_retention mean/min:** {_fmt(gr.get('mean'))} / {_fmt(gr.get('min') if 'min' in gr else None)}"
    )
    lines.append(f"- **recent_slope mean:** {_fmt(sl.get('mean'))}")
    lines.append(f"- **curriculum_safety_events (p2):** `{p2.get('curriculum_safety_events')}`")

    lines.append("\n## 6. Actionable conclusions\n")
    for rec in data.get("recommendations", []):
        lines.append(f"- {rec}")

    lines.append("\n## 7. Missing data / caveats\n")
    if data.get("missing_tags"):
        lines.append("**Missing TB tags / series:**")
        for t in data["missing_tags"]:
            lines.append(f"- `{t}`")
    else:
        lines.append("- No critical missing tags.")
    if data.get("warnings"):
        lines.append("\n**Warnings:**")
        for w in data["warnings"][:20]:
            lines.append(f"- {w}")
        if len(data["warnings"]) > 20:
            lines.append(f"- … and {len(data['warnings']) - 20} more")
    lines.append(
        "\n*Note: EIR scalars log only on actor-distill profiles (p1/p4/p5). "
        "p0/p2/p3 may show `case3` or empty EIR TB by design.*"
    )
    return "\n".join(lines) + "\n"


def _executive_bullets(data: dict) -> list[str]:
    bullets: list[str] = []
    v = data.get("verdicts", {})
    bullets.append(f"WM head loss scale: **{v.get('head_loss_scale', 'unknown')}** (TB `head_loss_weighted_fraction`).")
    for pol, name in (("1", "p1"), ("4", "p4"), ("5", "p5")):
        ev = data.get("eir_teacher_supply", {}).get(pol, {}).get("verdict", "n/a")
        bullets.append(f"{name} EIR chain: **{ev}**.")
    p3 = data.get("wm_source_selector", {}).get("3", {})
    tc = p3.get("from_training_curve", {})
    bullets.append(
        f"p3 WM source selector: **{p3.get('verdict')}** "
        f"({tc.get('wm_score_used_count', 0)} ranking ticks, "
        f"eff_w mean={_fmt((tc.get('wm_score_effective_weight') or {}).get('mean'))})."
    )
    bullets.append(f"p2 curriculum driver: **{data.get('p2_curriculum', {}).get('interpretation')}**.")
    fb = tc.get("fallback_reason_counts", {}) if False else p3.get("from_training_curve", {}).get("fallback_reason_counts", {})
    hi_unc = fb.get("high_uncertainty", 0)
    bullets.append(
        f"p3 legacy `high_uncertainty` fallback count in JSONL: **{hi_unc}** "
        f"(post-hotfix should be 0)."
    )
    bullets.append(
        f"Safety patch evidence: p2 real_signal_block events = "
        f"{data.get('p2_curriculum', {}).get('curriculum_safety_events', {}).get('event_counts', {}).get('real_signal_block', 0)}."
    )
    bullets.append(f"TensorBoard event files scanned: **{len(data.get('event_files', []))}** across run_dir (+ outputs fallback).")
    return bullets


def main() -> None:
    parser = argparse.ArgumentParser(description="TB/EIR/WM signal diagnosis")
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    args = parser.parse_args()

    print(f"Analyzing {args.run_dir} ...")
    data = analyze_run(args.run_dir)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.out_dir / "TB_EIR_WM_SIGNAL_DIAGNOSIS.json"
    md_path = args.out_dir / "TB_EIR_WM_SIGNAL_DIAGNOSIS.md"

    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, ensure_ascii=False)

    md = render_markdown(data)
    md_path.write_text(md, encoding="utf-8")

    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
