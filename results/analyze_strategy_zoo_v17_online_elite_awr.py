#!/usr/bin/env python3
"""Analyze Strategy Zoo v17 Online Elite-AWR runs."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any


PPO_BUFFER_ACTIONS = 1
V13_REFERENCE_FPS = {
    0: 95000.0,
    1: 90000.0,
    2: 95000.0,
    3: 90000.0,
    4: 90000.0,
    5: 90000.0,
}
FRAME_MARKS = (
    ("0.5B", 500_000_000),
    ("1.0B", 1_000_000_000),
    ("1.5B", 1_500_000_000),
    ("2.0B", 2_000_000_000),
)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.is_file():
        return rows
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def _nearest_row(rows: list[dict[str, Any]], target_frames: int) -> dict[str, Any] | None:
    if not rows:
        return None
    return min(rows, key=lambda r: abs(int(r.get("env_frames", 0)) - target_frames))


def _stat(values: list[float]) -> dict[str, float]:
    vals = [v for v in values if math.isfinite(v)]
    if not vals:
        return {"last": float("nan"), "mean": float("nan"), "late_mean": float("nan")}
    tail = vals[-max(1, len(vals) // 5) :]
    return {
        "last": float(vals[-1]),
        "mean": float(sum(vals) / len(vals)),
        "late_mean": float(sum(tail) / len(tail)),
    }


def _get(row: dict[str, Any], *keys: str, default: float = float("nan")) -> float:
    for key in keys:
        if key in row:
            try:
                return float(row[key])
            except (TypeError, ValueError):
                return default
    return default


def _load_sidecar_meta(run_dir: Path, policy_idx: int) -> dict[str, Any]:
    ws = run_dir / "dexpbt_8gpu_ws" / f"{policy_idx:03d}"
    for ckpt_dir in (ws / "nn", ws):
        for ckpt in sorted(ckpt_dir.glob("*.pth"), reverse=True):
            try:
                import torch

                payload = torch.load(ckpt, map_location="cpu", weights_only=False)
            except Exception:
                continue
            if isinstance(payload, dict) and payload.get("awr_buffer_ref_meta"):
                return dict(payload["awr_buffer_ref_meta"])
    sidecar = (
        run_dir
        / "dexpbt_8gpu_ws"
        / "awr_buffers"
        / f"policy_{policy_idx:02d}_awr_topk.pt"
    )
    if sidecar.is_file():
        return {"sidecar_path": str(sidecar), "size_mb": sidecar.stat().st_size / (1024 * 1024)}
    return {}


def analyze_run(run_dir: Path) -> dict[str, Any]:
    metrics = run_dir / "dexpbt_8gpu_ws" / "metrics"
    curves: dict[int, list[dict[str, Any]]] = {}
    for path in sorted(metrics.glob("training_curve_policy_*.jsonl")):
        pid = int(path.stem.split("_")[-1])
        curves[pid] = _read_jsonl(path)

    profile_copy = _read_jsonl(metrics / "profile_copy_events.jsonl")
    lineage_path = metrics / "policy_lineage.jsonl"
    lineage = _read_jsonl(lineage_path)

    same_frame: dict[str, Any] = {}
    for label, frames in FRAME_MARKS:
        per_policy = {}
        nsucc_vals = []
        for pid, rows in sorted(curves.items()):
            row = _nearest_row(rows, frames)
            if row is None:
                continue
            nsucc = float(row.get("nsucc_mean", float("nan")))
            per_policy[f"p{pid}"] = nsucc
            if math.isfinite(nsucc):
                nsucc_vals.append(nsucc)
        ranked = sorted(nsucc_vals, reverse=True)
        same_frame[label] = {
            "target_env_frames": frames,
            "per_policy_nsucc_mean": per_policy,
            "top1_mean": ranked[0] if ranked else float("nan"),
            "top3_mean": float(sum(ranked[:3]) / min(3, len(ranked))) if ranked else float("nan"),
        }

    awr_table: dict[str, Any] = {}
    for pid, rows in sorted(curves.items()):
        last = rows[-1] if rows else {}
        profile = str(last.get("wm_strategy_profile_name") or last.get("hyperparam_wm_strategy_profile") or "")
        awr_active = _get(last, "awr_enabled", "awr/enabled", default=0.0) > 0.5
        sidecar_meta = _load_sidecar_meta(run_dir, pid)
        awr_table[f"p{pid}"] = {
            "profile": profile,
            "awr_active": awr_active,
            "awr_loss_weight": _get(last, "awr_loss_weight", "awr/loss_weight"),
            "buffer_size": _get(last, "awr_buffer_sequences", "awr/buffer_sequences"),
            "buffer_steps": _get(last, "awr_buffer_steps", "awr/buffer_steps"),
            "sidecar_steps": float(sidecar_meta.get("num_steps", float("nan"))),
            "sidecar_save_failed": float(sidecar_meta.get("save_failed", float("nan"))),
            "action_convention_checked": _get(
                last, "awr_action_convention_checked", "awr/action_convention_checked", default=0.0
            ),
            "action_match_error_mean": _get(
                last, "awr_action_match_error_mean", "awr/action_match_error_mean"
            ),
            "action_match_error_max": _get(
                last, "awr_action_match_error_max", "awr/action_match_error_max"
            ),
            "action_match_order_code": _get(
                last, "awr_action_match_order_code", "awr/action_match_order_code", default=0.0
            ),
            "quality_mean": _stat(
                [_get(r, "awr_quality_mean", "awr/quality_mean") for r in rows]
            ),
            "delta_gate_health": {
                "full_frac": _stat(
                    [_get(r, "awr_delta_gate_full_frac", "awr/delta_gate_full_frac") for r in rows]
                ),
                "mid_frac": _stat(
                    [_get(r, "awr_delta_gate_mid_frac", "awr/delta_gate_mid_frac") for r in rows]
                ),
                "far_frac": _stat(
                    [_get(r, "awr_delta_gate_far_frac", "awr/delta_gate_far_frac") for r in rows]
                ),
                "drop_frac": _stat(
                    [_get(r, "awr_delta_gate_drop_frac", "awr/delta_gate_drop_frac") for r in rows]
                ),
            },
            "local_inherited_status": {
                "buffer_steps": _get(last, "awr_buffer_steps", "awr/buffer_steps"),
                "inherit_loaded_steps": _get(
                    last, "awr_buffer_inherit_loaded_steps", "awr/buffer_inherit_loaded_steps", default=0.0
                ),
                "sidecar_steps": float(sidecar_meta.get("num_steps", float("nan"))),
            },
            "action_convention_result": {
                "checked": _get(
                    last, "awr_action_convention_checked", "awr/action_convention_checked", default=0.0
                ),
                "mismatch": _get(
                    last, "awr_action_convention_mismatch", "awr/action_convention_mismatch", default=0.0
                ),
                "order_code": _get(
                    last, "awr_action_match_order_code", "awr/action_match_order_code", default=0.0
                ),
            },
            "awr_weighted_loss": _stat(
                [_get(r, "awr_loss_weighted", "awr/loss_weighted") for r in rows]
            ),
            "awr_active_frac": _stat(
                [_get(r, "awr_active_frac", "awr/enabled", default=0.0) for r in rows]
            ),
            "delta_gate_drop_frac": _stat(
                [_get(r, "awr_delta_gate_drop_frac", "awr/delta_gate_drop_frac") for r in rows]
            ),
            "kl_gate_enabled": _get(last, "awr_kl_gate_enabled", "awr/kl_gate_enabled", default=0.0),
            "kl_telemetry_only": _get(last, "awr_kl_telemetry_only", "awr/kl_telemetry_only", default=1.0),
            "kl_skip_count": _get(last, "awr_kl_gate_skip_count", "awr/kl_gate_skip_count", default=0.0),
            "loss_cap_hit_frac": _stat(
                [_get(r, "awr_loss_cap_hit_frac", "awr/loss_cap_hit_frac") for r in rows]
            ),
            "buffer_inherit_loaded_steps": _get(
                last, "awr_buffer_inherit_loaded_steps", "awr/buffer_inherit_loaded_steps", default=0.0
            ),
            "buffer_broadcast_loaded_steps": _get(
                last, "awr_buffer_broadcast_loaded_steps", "awr/buffer_broadcast_loaded_steps", default=0.0
            ),
            "broadcast_enabled": _get(last, "awr_broadcast_enabled", "awr/broadcast_enabled", default=0.0),
            "collection_mode": str(
                last.get("awr_collection_mode")
                or last.get("hyperparam_awr_collection_mode")
                or "unknown"
            ),
            "rollout_builder_entries": _get(
                last, "awr_rollout_builder_entries", "awr/rollout_builder_entries", default=0.0
            ),
            "rollout_builder_selected_steps": _get(
                last,
                "awr_rollout_builder_selected_steps",
                "awr/rollout_builder_selected_steps",
                default=0.0,
            ),
            "rollout_builder_time_ms": _get(
                last, "awr_rollout_builder_time_ms", "awr/rollout_builder_time_ms", default=0.0
            ),
            "skipped_collect_not_started": _get(
                last, "awr_skipped_collect_not_started", "awr/skipped_collect_not_started", default=0.0
            ),
            "action_source_code": _get(
                last, "awr_action_source_code", "awr/action_source_code", default=0.0
            ),
            "action_out_of_bounds_frac_1": _get(
                last, "awr_action_out_of_bounds_frac_1", "awr/action_out_of_bounds_frac_1", default=0.0
            ),
            "ppo_buffer_action_min": _get(
                last, "awr_ppo_buffer_action_min", "awr/ppo_buffer_action_min"
            ),
            "ppo_buffer_action_max": _get(
                last, "awr_ppo_buffer_action_max", "awr/ppo_buffer_action_max"
            ),
            "fps_last": _get(last, "fps", "train/fps", default=float("nan")),
        }

    v13_health: dict[str, Any] = {
        "elite_bank": {},
        "best_source_frac_elite": {},
        "teacher_selected_below_004": {},
        "profile_copy_events": len(profile_copy),
        "lineage_events": len(lineage),
    }
    for pid, rows in sorted(curves.items()):
        last = rows[-1] if rows else {}
        v13_health["elite_bank"][f"p{pid}"] = {
            "size": _get(last, "eir_elite_bank_size", "eir/elite_bank_size", default=0.0),
            "add_count": _get(last, "eir_elite_bank_add_count", "eir/elite_bank_add_count", default=0.0),
        }
        v13_health["best_source_frac_elite"][f"p{pid}"] = _get(
            last, "eir_best_source_frac_elite", "eir/best_source_frac_elite"
        )
        v13_health["teacher_selected_below_004"][f"p{pid}"] = _get(
            last, "eir_teacher_selected_below_004", "eir/teacher_selected_below_004"
        )

    separation: dict[str, Any] = {}
    for pid, rows in sorted(curves.items()):
        separation[f"p{pid}"] = {
            "eir": {
                "actor_distill_active_fraction": _stat(
                    [
                        _get(r, "eir_actor_distill_active_fraction", "eir/actor_distill_active_fraction")
                        for r in rows
                    ]
                ),
                "actor_distill_weighted_loss": _stat(
                    [
                        _get(
                            r,
                            "eir_actor_distill_weighted_loss",
                            "eir/actor_distill_weighted_loss",
                        )
                        for r in rows
                    ]
                ),
                "actor_distill_reject_large_delta": _stat(
                    [
                        _get(
                            r,
                            "eir_actor_distill_reject_large_delta",
                            "eir/actor_distill_reject_large_delta",
                        )
                        for r in rows
                    ]
                ),
            },
            "awr": {
                "weighted_loss": _stat(
                    [_get(r, "awr_loss_weighted", "awr/loss_weighted") for r in rows]
                ),
                "active_frac": _stat(
                    [_get(r, "awr_active_frac", "awr/enabled", default=0.0) for r in rows]
                ),
                "delta_gate_drop_frac": _stat(
                    [_get(r, "awr_delta_gate_drop_frac", "awr/delta_gate_drop_frac") for r in rows]
                ),
                "quality_mean": _stat(
                    [_get(r, "awr_quality_mean", "awr/quality_mean") for r in rows]
                ),
            },
        }

    awr_health_summary = {
        "local_inherited_status": {
            pid: row.get("local_inherited_status", {})
            for pid, row in awr_table.items()
        },
        "action_convention": {
            pid: row.get("action_convention_result", {})
            for pid, row in awr_table.items()
        },
        "delta_gate": {
            pid: row.get("delta_gate_health", {})
            for pid, row in awr_table.items()
        },
        "weighted_loss": {
            pid: row.get("awr_weighted_loss", {})
            for pid, row in awr_table.items()
        },
        "separation_vs_eir_actor_distill": separation,
    }

    v17_1_health: dict[str, Any] = {"checks": {}, "per_policy": {}}
    for pid_key, row in awr_table.items():
        pid = int(pid_key.lstrip("p"))
        fps = float(row.get("fps_last", float("nan")))
        ref_fps = V13_REFERENCE_FPS.get(pid, float("nan"))
        awr_on = bool(row.get("awr_active"))
        perf_ok = (not awr_on) or (not math.isfinite(fps)) or (not math.isfinite(ref_fps)) or fps >= 0.7 * ref_fps
        mismatch = float(row.get("action_convention_result", {}).get("mismatch", 0.0))
        conv_ok = mismatch < 0.5
        source_code = float(row.get("action_source_code", 0.0))
        source_ok = (not awr_on) or source_code == float(PPO_BUFFER_ACTIONS)
        v17_1_health["per_policy"][pid_key] = {
            "awr_collection_mode": row.get("collection_mode"),
            "rollout_builder_entries": row.get("rollout_builder_entries"),
            "rollout_builder_selected_steps": row.get("rollout_builder_selected_steps"),
            "rollout_builder_time_ms": row.get("rollout_builder_time_ms"),
            "skipped_collect_not_started": row.get("skipped_collect_not_started"),
            "action_source_code": source_code,
            "action_out_of_bounds_frac_1": row.get("action_out_of_bounds_frac_1"),
            "ppo_buffer_action_min": row.get("ppo_buffer_action_min"),
            "ppo_buffer_action_max": row.get("ppo_buffer_action_max"),
            "fps_by_policy": fps,
            "v13_reference_fps": ref_fps,
            "collection_performance_ok": perf_ok,
            "action_convention_ok": conv_ok,
            "action_source_ok": source_ok,
        }
    v17_1_health["checks"] = {
        "collection_performance_failed": any(
            not block.get("collection_performance_ok", True)
            for block in v17_1_health["per_policy"].values()
        ),
        "action_convention_failed": any(
            not block.get("action_convention_ok", True)
            for block in v17_1_health["per_policy"].values()
        ),
        "action_source_failed": any(
            not block.get("action_source_ok", True)
            for block in v17_1_health["per_policy"].values()
        ),
    }

    return {
        "run_dir": str(run_dir),
        "same_frame_nsucc": same_frame,
        "awr_active_table": awr_table,
        "awr_health_summary": awr_health_summary,
        "v17_1_rollout_buffer_health": v17_1_health,
        "v13_mechanism_health": v13_health,
        "separation_block": separation,
        "run_config_notes": {
            "awr_kl_gate_enable": False,
            "awr_kl_telemetry_only": True,
            "awr_broadcast_enabled": False,
            "awr_inherit_enabled": True,
            "awr_collection_mode": "rollout_buffer",
        },
    }


def _render_markdown(report: dict[str, Any]) -> str:
    lines = ["# Strategy Zoo v17 Online Elite-AWR Report", ""]
    lines.append(f"Run: `{report['run_dir']}`")
    lines.append("")
    lines.append("## A. Same-frame nsucc")
    for label, block in report["same_frame_nsucc"].items():
        lines.append(f"### {label}")
        for pid, val in sorted(block.get("per_policy_nsucc_mean", {}).items()):
            lines.append(f"- {pid}: {val:.3f}" if math.isfinite(val) else f"- {pid}: n/a")
        lines.append(f"- top1 mean: {block.get('top1_mean')}")
        lines.append(f"- top3 mean: {block.get('top3_mean')}")
        lines.append("")
    lines.append("## B. AWR active table")
    for pid, row in report["awr_active_table"].items():
        lines.append(f"### {pid}")
        for key, val in row.items():
            lines.append(f"- {key}: {val}")
        lines.append("")
    lines.append("## B2. AWR health summary")
    summary = report.get("awr_health_summary", {})
    lines.append("### Local/inherited status")
    lines.append(str(summary.get("local_inherited_status", {})))
    lines.append("### Action convention check result")
    lines.append(str(summary.get("action_convention", {})))
    lines.append("### Delta gate health")
    lines.append(str(summary.get("delta_gate", {})))
    lines.append("### Weighted loss health")
    lines.append(str(summary.get("weighted_loss", {})))
    lines.append("")
    lines.append("## B3. v17.1 rollout-buffer health")
    v171 = report.get("v17_1_rollout_buffer_health", {})
    lines.append(f"- checks: {v171.get('checks', {})}")
    for pid, block in sorted(v171.get("per_policy", {}).items()):
        lines.append(f"### {pid}")
        for key, val in block.items():
            lines.append(f"- {key}: {val}")
        lines.append("")
    lines.append("")
    lines.append("## C. v13 mechanism health")
    lines.append(f"- profile_copy_events: {report['v13_mechanism_health']['profile_copy_events']}")
    lines.append(f"- lineage_events: {report['v13_mechanism_health']['lineage_events']}")
    lines.append("")
    lines.append("## D. Separation block (EIR actor_distill vs AWR)")
    for pid, block in report["separation_block"].items():
        lines.append(f"### {pid}")
        lines.append(f"- EIR actor_distill: {block['eir']}")
        lines.append(f"- AWR: {block['awr']}")
        lines.append("")
    notes = report.get("run_config_notes", {})
    lines.append("## Run config notes")
    lines.append(f"- awr_kl_gate_enable: {notes.get('awr_kl_gate_enable')}")
    lines.append(f"- awr_kl_telemetry_only: {notes.get('awr_kl_telemetry_only')}")
    lines.append(f"- awr_broadcast_enabled: {notes.get('awr_broadcast_enabled')}")
    lines.append(f"- awr_inherit_enabled: {notes.get('awr_inherit_enabled')}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=str, required=True)
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/docs",
    )
    args = parser.parse_args()
    run_dir = Path(args.run_dir)
    report = analyze_run(run_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "STRATEGY_ZOO_V17_ONLINE_ELITE_AWR_REPORT.json"
    md_path = out_dir / "STRATEGY_ZOO_V17_ONLINE_ELITE_AWR_REPORT.md"
    json_path.write_text(json.dumps(report, indent=2))
    md_path.write_text(_render_markdown(report))
    print(f"wrote {json_path}")
    print(f"wrote {md_path}")


if __name__ == "__main__":
    main()
