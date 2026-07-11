#!/usr/bin/env python3
"""Select per-policy continuation checkpoints and fork a writable workspace."""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
from pathlib import Path
from typing import Any


def _finite(x: Any) -> float | None:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(v):
        return None
    return v


def _load_meta(path: Path) -> dict[str, Any] | None:
    try:
        data = json.loads(path.read_text())
    except Exception:
        return None
    return data if isinstance(data, dict) else None


def select_policy_checkpoint(
    policy_dir: Path,
    *,
    tol_lo: float = 0.020,
    tol_hi: float = 0.026,
    min_mean_primary: float = 25.0,
    min_mean_fallback: float = 20.0,
) -> dict[str, Any]:
    candidates: list[dict[str, Any]] = []
    for meta_path in sorted(policy_dir.glob("*.json")):
        if meta_path.name.startswith("_"):
            continue
        ckpt = meta_path.with_suffix(".pth")
        if not ckpt.is_file():
            continue
        meta = _load_meta(meta_path)
        if meta is None:
            continue
        cur = meta.get("curriculum") if isinstance(meta.get("curriculum"), dict) else {}
        tol = _finite(cur.get("success_tolerance"))
        mean = _finite(meta.get("nsucc_mean"))
        nmax = _finite(meta.get("nsucc_max"))
        frame_raw = meta.get("population_frame", meta.get("frame"))
        try:
            frame = int(frame_raw)
        except (TypeError, ValueError):
            continue
        if tol is None or mean is None or nmax is None or frame < 0:
            continue
        if not (tol_lo <= tol <= tol_hi):
            continue
        try:
            iteration = int(meta.get("iteration", int(meta_path.stem)))
        except Exception:
            iteration = int(meta_path.stem)
        candidates.append(
            {
                "meta_path": str(meta_path),
                "checkpoint": str(ckpt.resolve()),
                "iteration": iteration,
                "source_population_frame": frame,
                "success_tolerance": tol,
                "nsucc_mean": mean,
                "nsucc_max": nmax,
                "initial_tolerance": _finite(cur.get("initial_tolerance")),
                "target_tolerance_source": _finite(cur.get("target_tolerance")),
                "last_curriculum_update": int(cur.get("last_curriculum_update") or 0),
                "frame_since_restart": int(cur.get("frame_since_restart") or 0),
                "seed_in_meta": meta.get("seed"),
            }
        )
    if not candidates:
        raise RuntimeError(f"no checkpoint candidates in {policy_dir}")

    def _rank(c: dict[str, Any]) -> tuple:
        # Prefer stricter (lower) tol near 2cm, then higher mean, then later frame.
        return (
            float(c["success_tolerance"]),
            -float(c["nsucc_mean"]),
            -int(c["source_population_frame"]),
        )

    primary = [c for c in candidates if c["nsucc_mean"] >= min_mean_primary]
    fallback = [c for c in candidates if c["nsucc_mean"] >= min_mean_fallback]
    warning = False
    if primary:
        pool = primary
        reason = (
            f"strictest_tol_in_[{tol_lo},{tol_hi}]_with_mean>={min_mean_primary}"
        )
    elif fallback:
        pool = fallback
        reason = (
            f"relaxed_mean>={min_mean_fallback}_strictest_tol_in_[{tol_lo},{tol_hi}]"
        )
        warning = True
    else:
        pool = candidates
        reason = f"best_mean_in_[{tol_lo},{tol_hi}]_below_mean_{min_mean_fallback}"
        warning = True
    best = sorted(pool, key=_rank)[0]
    best["selection_reason"] = reason
    best["source_checkpoint_quality_warning"] = warning
    best["num_candidates"] = len(candidates)
    return best


def resolve_base_seed(src_run: Path) -> int:
    cfg = src_run / "RUN_CONFIG.txt"
    if cfg.is_file():
        for line in cfg.read_text().splitlines():
            if line.startswith("base_seed="):
                return int(line.split("=", 1)[1].strip())
    # Fall back to launch_master / known launcher convention.
    master = src_run / "launch_master.log"
    if master.is_file():
        text = master.read_text(errors="ignore")
        for token in text.split():
            if token.startswith("--seed="):
                return int(token.split("=", 1)[1])
            if token == "--seed":
                # next token not easily available; continue
                pass
    raise RuntimeError(
        f"could not determine base_seed from {src_run}; refusing to guess"
    )


def fork_workspace(src_ws: Path, dst_ws: Path) -> dict[str, Any]:
    """Copy mutable JSON state; hardlink immutable archives/checkpoints where possible."""
    dst_ws.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {"copied": [], "hardlinked_dirs": [], "skipped": []}

    # Mutable control / lineage / posterior: real copies.
    # Do NOT copy historical training curves — they would poison live monitors
    # with source end-of-run frames/tolerances.
    for name in (
        "curriculum_control",
        "policy_lineage.json",
        "wm_posterior_latest.json",
        "wm_imagined",
        "eir_broadcast",
    ):
        src = src_ws / name
        dst = dst_ws / name
        if not src.exists():
            report["skipped"].append(name)
            continue
        if src.is_dir():
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(src, dst)
        else:
            shutil.copy2(src, dst)
        report["copied"].append(name)

    # Fresh metrics directory for the continuation run.
    metrics_dst = dst_ws / "metrics"
    if metrics_dst.exists():
        shutil.rmtree(metrics_dst)
    metrics_dst.mkdir(parents=True, exist_ok=True)
    report["copied"].append("metrics (empty)")

    # Large immutable archives / teachers: prefer hardlink tree (same FS).
    for name in (
        "best",
        "best_nsucc",
        "best_stable",
        "best_floor_recovery",
        "best_r_meta",
        "world_model_teachers",
    ):
        src = src_ws / name
        dst = dst_ws / name
        if not src.exists():
            report["skipped"].append(name)
            continue
        if dst.exists():
            shutil.rmtree(dst)
        try:
            # cp -al creates hardlinked tree without writing into source.
            subprocess_cp_al = shutil.which("cp")
            if subprocess_cp_al is None:
                raise RuntimeError("cp not found")
            import subprocess

            subprocess.check_call(["cp", "-a", "--link", str(src), str(dst)])
            report["hardlinked_dirs"].append(name)
        except Exception:
            # Fallback to copy (slow / disk heavy).
            shutil.copytree(src, dst)
            report["copied"].append(name + " (full_copy_fallback)")

    # Do NOT copy live policy dirs 000-005 or restart_* (new run writes fresh).
    for name in ("000", "001", "002", "003", "004", "005", "restart_p2"):
        report["skipped"].append(f"live_or_tmp:{name}")
        (dst_ws / name).mkdir(parents=True, exist_ok=True)

    # Ensure metrics exists even if source lacked it.
    (dst_ws / "metrics").mkdir(parents=True, exist_ok=True)
    return report


def build_manifest(
    *,
    src_run: Path,
    dst_run: Path,
    additional_frames: int,
    target_tol: float,
    base_seed: int | None = None,
    num_policies: int = 6,
) -> dict[str, Any]:
    src_ws = src_run / "dexpbt_8gpu_ws"
    if not src_ws.is_dir():
        raise FileNotFoundError(src_ws)
    if base_seed is None:
        base_seed = resolve_base_seed(src_run)

    policies: list[dict[str, Any]] = []
    for idx in range(num_policies):
        policy_dir = src_ws / f"{idx:03d}"
        if not policy_dir.is_dir():
            raise FileNotFoundError(policy_dir)
        picked = select_policy_checkpoint(policy_dir)
        source_frame = int(picked["source_population_frame"])
        target_frames = source_frame + int(additional_frames)
        init_tol = picked.get("initial_tolerance")
        if init_tol is None:
            init_tol = 0.075
        entry = {
            "policy_idx": idx,
            "checkpoint": picked["checkpoint"],
            "source_iteration": int(picked["iteration"]),
            "source_population_frame": source_frame,
            "target_population_frames": int(target_frames),
            "success_tolerance_resume": float(picked["success_tolerance"]),
            "initial_tolerance_resume": float(init_tol),
            "target_tolerance_resume": float(target_tol),
            "last_curriculum_update_resume": int(
                picked.get("last_curriculum_update") or 0
            ),
            "frame_since_restart_resume": int(picked.get("frame_since_restart") or 0),
            # train.py uses effective_seed = base_seed + 1000 * policy_idx
            "seed": int(base_seed),
            "effective_seed": int(base_seed) + 1000 * idx,
            "source_nsucc_mean": float(picked["nsucc_mean"]),
            "source_nsucc_max": float(picked["nsucc_max"]),
            "selection_reason": picked["selection_reason"],
            "source_checkpoint_quality_warning": bool(
                picked["source_checkpoint_quality_warning"]
            ),
            "num_candidates": int(picked["num_candidates"]),
            "meta_path": picked["meta_path"],
        }
        policies.append(entry)

    fork_report = fork_workspace(src_ws, dst_run / "dexpbt_8gpu_ws")
    manifest = {
        "source_run": str(src_run.resolve()),
        "destination_run": str(dst_run.resolve()),
        "additional_frames": int(additional_frames),
        "target_tolerance": float(target_tol),
        "base_seed": int(base_seed),
        "seed_rule": "effective_seed = base_seed + 1000 * policy_idx",
        "num_policies": int(num_policies),
        "workspace_fork": fork_report,
        "policies": policies,
    }
    return manifest


def write_manifest(manifest: dict[str, Any], run_dir: Path) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    json_path = run_dir / "resume_manifest.json"
    txt_path = run_dir / "resume_manifest.txt"
    json_path.write_text(json.dumps(manifest, indent=2))
    lines = [
        f"source_run={manifest['source_run']}",
        f"destination_run={manifest['destination_run']}",
        f"base_seed={manifest['base_seed']}",
        f"seed_rule={manifest['seed_rule']}",
        f"additional_frames={manifest['additional_frames']}",
        f"target_tolerance={manifest['target_tolerance']}",
        "",
    ]
    for p in manifest["policies"]:
        lines.extend(
            [
                f"policy={p['policy_idx']:02d}",
                f"  checkpoint={p['checkpoint']}",
                f"  iteration={p['source_iteration']}",
                f"  source_frame={p['source_population_frame']}",
                f"  target_frame={p['target_population_frames']}",
                f"  success_tolerance={p['success_tolerance_resume']:.8f}",
                f"  target_tolerance={p['target_tolerance_resume']}",
                f"  nsucc_mean={p['source_nsucc_mean']:.4f}",
                f"  nsucc_max={p['source_nsucc_max']:.4f}",
                f"  seed_base={p['seed']} effective_seed={p['effective_seed']}",
                f"  reason={p['selection_reason']}",
                f"  quality_warning={p['source_checkpoint_quality_warning']}",
                "",
            ]
        )
    txt_path.write_text("\n".join(lines))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src-run", required=True)
    ap.add_argument("--dst-run", required=True)
    ap.add_argument("--additional-frames", type=int, default=1_500_000_000)
    ap.add_argument("--target-tol", type=float, default=0.018)
    ap.add_argument("--base-seed", type=int, default=None)
    ap.add_argument("--num-policies", type=int, default=6)
    args = ap.parse_args()
    manifest = build_manifest(
        src_run=Path(args.src_run),
        dst_run=Path(args.dst_run),
        additional_frames=int(args.additional_frames),
        target_tol=float(args.target_tol),
        base_seed=args.base_seed,
        num_policies=int(args.num_policies),
    )
    write_manifest(manifest, Path(args.dst_run))
    print(json.dumps({"ok": True, "policies": len(manifest["policies"])}, indent=2))


if __name__ == "__main__":
    main()
