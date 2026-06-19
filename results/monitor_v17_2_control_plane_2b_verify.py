#!/usr/bin/env python3
"""2B control-plane verification monitor (observe-only, fail-fast optional).

Reads training_curve, pbt_copy_events, train.log, checkpoint meta, nvidia-smi.
Does NOT modify training code.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

FPS_RE = re.compile(
    r"fps step:\s*(\d+)\s+fps step and policy inference:\s*(\d+)\s+fps total:\s*(\d+).*?frames:\s*(\d+)"
)

PROFILES = [
    "ppo_only",
    "elite_bank_actor_distill_fast_awr",
    "wm_pbt_source_selector_v2",
    "wm_pbt_source_selector_awr",
    "elite_bank_actor_distill_awr",
    "elite_bank_actor_distill_strong_awr",
]

FAIL_FAST_COPY_ZERO_AFTER_FRAMES = 300_000_000
FAIL_FAST_FRAME_SPREAD = 800_000_000
FAIL_FAST_FPS_RATIO = 5.0
CURVE_STALE_GAP_FRAMES = 100_000_000


def _resolve_population_frames(policies: list[dict[str, Any]]) -> dict[str, Any]:
    """v17.2 debug hotfix: prefer train.log frames over stale training_curve frames."""
    curve_frames = [p["env_frames_curve"] for p in policies if p["env_frames_curve"] > 0]
    log_frames = [p["fps_frames_log"] for p in policies if p["fps_frames_log"] > 0]
    if log_frames:
        all_frames = log_frames
        frame_source = "log"
    elif curve_frames:
        all_frames = curve_frames
        frame_source = "curve"
    else:
        all_frames = []
        frame_source = "none"
    spread = (max(all_frames) - min(all_frames)) if len(all_frames) >= 2 else 0
    max_frame = max(all_frames) if all_frames else 0
    curve_stale = False
    stale_gap_frames = 0
    if log_frames and curve_frames:
        stale_gap_frames = max(log_frames) - max(curve_frames)
        curve_stale = stale_gap_frames > CURVE_STALE_GAP_FRAMES
    return {
        "all_frames": all_frames,
        "frame_source": frame_source,
        "frame_spread": spread,
        "max_env_frames": max_frame,
        "curve_stale": curve_stale,
        "stale_gap_frames": stale_gap_frames,
        "curve_frames": curve_frames,
        "log_frames": log_frames,
    }


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    out: list[dict[str, Any]] = []
    for line in path.read_text(errors="replace").splitlines():
        line = line.strip()
        if line:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def _latest_meta_lineage(run_dir: Path, policy_idx: int) -> dict[str, Any]:
    ws = run_dir / "dexpbt_8gpu_ws" / f"{policy_idx:03d}"
    for d in (ws / "nn", ws):
        if not d.is_dir():
            continue
        for meta in sorted(d.glob("*.json"), reverse=True):
            if meta.name.endswith(".tmp") or meta.name == "LATEST.commit":
                continue
            try:
                data = json.loads(meta.read_text())
            except Exception:
                continue
            if isinstance(data, dict) and data.get("lineage_id"):
                return {
                    "lineage_id": str(data.get("lineage_id")),
                    "population_frame": int(data.get("population_frame", data.get("frame", 0)) or 0),
                    "lineage_start_pf": data.get("lineage_start_pf"),
                    "weight_age_frame": data.get("weight_age_frame"),
                    "meta_path": str(meta),
                }
    return {}


def _parse_fps(log_path: Path) -> dict[str, Any]:
    if not log_path.is_file():
        return {"fps_total": float("nan"), "frames": 0, "epoch": 0}
    best = None
    for line in log_path.read_text(errors="replace").splitlines():
        m = FPS_RE.search(line)
        if m:
            best = {
                "fps_step": int(m.group(1)),
                "fps_infer": int(m.group(2)),
                "fps_total": int(m.group(3)),
                "frames": int(m.group(4)),
            }
    return best or {"fps_total": float("nan"), "frames": 0}


def _gpu_stats() -> list[dict[str, Any]]:
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,utilization.gpu,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=10,
        )
    except Exception:
        return []
    rows = []
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 4:
            rows.append(
                {
                    "gpu": int(parts[0]),
                    "util_pct": float(parts[1]),
                    "mem_used_mb": float(parts[2]),
                    "mem_total_mb": float(parts[3]),
                }
            )
    return rows


def _policy_status(run_dir: Path, pid: int) -> dict[str, Any]:
    pdir = run_dir / f"policy_{pid:02d}"
    log = pdir / "train.log"
    metrics = run_dir / "dexpbt_8gpu_ws" / "metrics"
    curve_path = metrics / f"training_curve_policy_{pid:02d}.jsonl"
    curves = _read_jsonl(curve_path)
    last = curves[-1] if curves else {}
    fps = _parse_fps(log)
    lineage = _latest_meta_lineage(run_dir, pid)

    def g(key: str, default: float = float("nan")) -> float:
        if key in last:
            try:
                return float(last[key])
            except (TypeError, ValueError):
                pass
        alt = key.replace("awr_", "awr/")
        if alt in last:
            try:
                return float(last[alt])
            except (TypeError, ValueError):
                pass
        return default

    return {
        "policy": pid,
        "profile": PROFILES[pid] if pid < len(PROFILES) else "unknown",
        "status": (pdir / "status.txt").read_text().strip()
        if (pdir / "status.txt").is_file()
        else "",
        "train_log_kb": log.stat().st_size / 1024 if log.is_file() else 0,
        "env_frames_curve": int(last.get("env_frames", fps.get("frames", 0)) or 0),
        "pbt_iteration": int(last.get("pbt_iteration", 0) or 0),
        "lineage_id": lineage.get("lineage_id", ""),
        "population_frame_meta": lineage.get("population_frame"),
        "target_frame_after_restore": g("pbt_copy_target_frame_before_restore", 0.0),
        "fps_total": float(fps.get("fps_total", float("nan"))),
        "fps_frames_log": int(fps.get("frames", 0) or 0),
        "awr_enabled": g("awr_enabled", g("awr/enabled", 0.0)),
        "awr_loss_weighted": g("awr_loss_weighted", g("awr/loss_weighted", 0.0)),
        "awr_effective_lambda": g("awr_effective_lambda", g("awr/effective_lambda", 0.0)),
        "awr_buffer_steps": g("awr_buffer_steps", g("awr/buffer_steps", 0.0)),
        "awr_buffer_sequences": g("awr_buffer_sequences", g("awr/buffer_sequences", 0.0)),
        "awr_active_frac": g("awr_active_frac", g("awr/active_frac", 0.0)),
        "awr_action_convention_mismatch": g(
            "awr_action_convention_mismatch", g("awr/action_convention_mismatch", 0.0)
        ),
        "awr_rollout_builder_time_ms": g(
            "awr_rollout_builder_time_ms", g("awr/rollout_builder_time_ms", 0.0)
        ),
        "disabled_flags": {
            "awr_disabled": g("awr_enabled", 1.0) < 0.5 and pid in (1, 3, 4, 5),
            "action_convention_mismatch": g("awr_action_convention_mismatch", 0.0) > 0.5,
            "startup_hang": "startup_hang" in log.read_text(errors="replace")[-5000:]
            if log.is_file()
            else False,
        },
    }


def _copy_event_count(run_dir: Path) -> int:
    path = run_dir / "dexpbt_8gpu_ws" / "metrics" / "pbt_copy_events.jsonl"
    return len(_read_jsonl(path))


def _baseline_fps_path(run_dir: Path) -> Path:
    return run_dir / "verify_fps_baseline.json"


def _load_baselines(run_dir: Path) -> dict[int, float]:
    p = _baseline_fps_path(run_dir)
    if not p.is_file():
        return {}
    try:
        raw = json.loads(p.read_text())
        return {int(k): float(v) for k, v in raw.items()}
    except Exception:
        return {}


def _save_baselines(run_dir: Path, baselines: dict[int, float]) -> None:
    _baseline_fps_path(run_dir).write_text(json.dumps(baselines, indent=2))


def evaluate(run_dir: Path, *, milestone: str = "periodic") -> dict[str, Any]:
    run_dir = Path(run_dir).resolve()
    now = datetime.now(timezone.utc).astimezone().isoformat()
    policies = [_policy_status(run_dir, i) for i in range(6)]
    frame_info = _resolve_population_frames(policies)
    spread = int(frame_info["frame_spread"])
    all_frames = frame_info["all_frames"]
    copy_count = _copy_event_count(run_dir)
    gpu = _gpu_stats()

    baselines = _load_baselines(run_dir)
    for p in policies:
        pid = int(p["policy"])
        fps = float(p["fps_total"])
        if math.isfinite(fps) and fps > 1000:
            if pid not in baselines:
                baselines[pid] = fps
    _save_baselines(run_dir, baselines)

    awr_active_policies = [
        p for p in policies if p["awr_enabled"] > 0.5 or p["profile"].endswith("_awr")
    ]
    awr_loss_nonzero = sum(
        1 for p in awr_active_policies if p["awr_loss_weighted"] > 1e-9
    )
    awr_active_frac_mean = (
        sum(p["awr_active_frac"] for p in awr_active_policies) / max(len(awr_active_policies), 1)
    )

    fail_reasons: list[str] = []
    max_frame = int(frame_info["max_env_frames"])
    for p in policies:
        pid = int(p["policy"])
        fps = float(p["fps_total"])
        base = baselines.get(pid)
        if base and math.isfinite(fps) and fps > 0 and fps < base / FAIL_FAST_FPS_RATIO:
            fail_reasons.append(
                f"p{pid} FPS collapse: {fps:.0f} vs baseline {base:.0f} (>{FAIL_FAST_FPS_RATIO}x drop)"
            )
        if p["disabled_flags"].get("action_convention_mismatch"):
            fail_reasons.append(f"p{pid} AWR disabled_by_convention")
        builder_ms = float(p["awr_rollout_builder_time_ms"])
        fps_val = float(p["fps_total"])
        if math.isfinite(builder_ms) and builder_ms > 500 and math.isfinite(fps_val) and fps_val < 30000:
            fail_reasons.append(
                f"p{pid} suspected interceptor CPU bottleneck: builder_ms={builder_ms:.1f}, fps={fps_val:.0f}"
            )

    if max_frame >= FAIL_FAST_COPY_ZERO_AFTER_FRAMES and copy_count == 0:
        fail_reasons.append(
            f"PBT copy count still 0 at max_frame={max_frame} (>={FAIL_FAST_COPY_ZERO_AFTER_FRAMES})"
        )
    if spread > FAIL_FAST_FRAME_SPREAD:
        fail_reasons.append(f"frame spread {spread} > {FAIL_FAST_FRAME_SPREAD}")

    report = {
        "timestamp": now,
        "milestone": milestone,
        "run_dir": str(run_dir),
        "commit": (run_dir / "RUN_CONFIG.txt").read_text(errors="replace").split("commit=")[-1].split()[0]
        if (run_dir / "RUN_CONFIG.txt").is_file()
        else "",
        "policies": policies,
        "aggregate": {
            "frame_spread": spread,
            "max_env_frames": max_frame,
            "frame_source": frame_info["frame_source"],
            "curve_stale": frame_info["curve_stale"],
            "stale_gap_frames": frame_info["stale_gap_frames"],
            "pbt_copy_event_count": copy_count,
            "awr_active_fraction_mean": awr_active_frac_mean,
            "awr_loss_nonzero_policies": awr_loss_nonzero,
            "gpu": gpu,
        },
        "fail_fast": {
            "triggered": bool(fail_reasons),
            "reasons": fail_reasons,
        },
    }
    return report


def _format_report(report: dict[str, Any]) -> str:
    lines = [
        f"=== 2B Control-Plane Verify Report [{report.get('milestone')}] {report.get('timestamp')} ===",
        f"run_dir: {report.get('run_dir')}",
        f"commit: {report.get('commit')}",
        "",
        f"frame_spread: {report['aggregate']['frame_spread']}",
        f"max_env_frames: {report['aggregate']['max_env_frames']}",
        f"frame_source: {report['aggregate'].get('frame_source', 'unknown')}",
        f"curve_stale: {report['aggregate'].get('curve_stale', False)}",
        f"stale_gap_frames: {report['aggregate'].get('stale_gap_frames', 0)}",
        f"pbt_copy_events: {report['aggregate']['pbt_copy_event_count']}",
        f"awr_active_fraction_mean: {report['aggregate']['awr_active_fraction_mean']:.4f}",
        "",
        "per-worker:",
    ]
    for p in report["policies"]:
        lid = f"{str(p['lineage_id'])[:8]}..." if p["lineage_id"] else "n/a"
        lines.append(
            f"  p{p['policy']} ({p['profile']}): frames={p['env_frames_curve']} "
            f"pbt_iter={p['pbt_iteration']} fps_total={p['fps_total']} "
            f"awr_en={p['awr_enabled']:.1f} awr_loss={p['awr_loss_weighted']:.2e} "
            f"lambda={p['awr_effective_lambda']:.2e} buf={int(p['awr_buffer_steps'])} "
            f"lineage={lid}"
        )
    if report["aggregate"]["gpu"]:
        lines.append("")
        lines.append("GPU:")
        for g in report["aggregate"]["gpu"]:
            lines.append(
                f"  gpu{g['gpu']}: util={g['util_pct']:.0f}% mem={g['mem_used_mb']:.0f}/{g['mem_total_mb']:.0f}MB"
            )
    if report["fail_fast"]["triggered"]:
        lines.append("")
        lines.append("FAIL_FAST:")
        for r in report["fail_fast"]["reasons"]:
            lines.append(f"  - {r}")
    lines.append("")
    return "\n".join(lines)


def _stop_run(run_dir: Path) -> None:
    pidfile = run_dir / "nohup.pid"
    if pidfile.is_file():
        pid = int(pidfile.read_text().strip())
        try:
            os.killpg(os.getpgid(pid), signal.SIGTERM)
        except ProcessLookupError:
            try:
                os.kill(pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
    mon = run_dir / "verify_monitor.pid"
    if mon.is_file():
        try:
            os.kill(int(mon.read_text().strip()), signal.SIGTERM)
        except (ProcessLookupError, ValueError):
            pass


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--milestone", default="periodic")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--loop", action="store_true")
    ap.add_argument("--interval-s", type=int, default=1800)
    ap.add_argument("--fail-fast", action="store_true")
    args = ap.parse_args()

    def once() -> int:
        report = evaluate(args.run_dir, milestone=args.milestone)
        text = _format_report(report)
        print(text)
        out_dir = args.out or (args.run_dir / "verify_reports")
        out_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = out_dir / f"verify_{args.milestone}_{stamp}.json"
        out_path.write_text(json.dumps(report, indent=2))
        txt_path = out_dir / f"verify_{args.milestone}_{stamp}.txt"
        txt_path.write_text(text)
        if args.fail_fast and report["fail_fast"]["triggered"]:
            print("FAIL_FAST triggered — stopping run", file=sys.stderr)
            _stop_run(args.run_dir)
            return 2
        return 0

    if not args.loop:
        raise SystemExit(once())

    while True:
        rc = once()
        if rc != 0:
            return rc
        time.sleep(max(args.interval_s, 60))


if __name__ == "__main__":
    raise SystemExit(main())
