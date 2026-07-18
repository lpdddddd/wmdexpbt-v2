#!/usr/bin/env python3
"""CCRS fixed-tolerance evaluation protocol (pure python, no Isaac import).

Defines the evaluation unit (checkpoint x tolerance x seed), config hashing,
completion/resume bookkeeping, and result-row schema. The GPU entry point
(dexpbt_isaaclab.scripts.ccrs_eval) imports this module so unit tests can
validate the protocol without Isaac.

Units: all tolerances in meters (m). cm shown in reports is m * 100.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional

DEFAULT_TOLERANCES_M = [0.028, 0.025, 0.023, 0.022, 0.021, 0.020, 0.019, 0.018]
DEFAULT_SEEDS = [10001, 10002, 10003, 10004]

EVALUATION_MODE_POLICY_ONLY = "policy_only"
EVALUATION_MODE_SYSTEM_ON = "system_on"  # reserved; not mixed with main results

SCHEMA_VERSION = 1


@dataclass(frozen=True)
class EvalUnit:
    """One independently runnable evaluation: checkpoint x tolerance x seed."""
    run_id: str
    worker_id: int
    checkpoint_iteration: int
    checkpoint_frame: int
    checkpoint_path: str
    fixed_tolerance: float          # meters, frozen for the whole rollout
    reset_seed: int
    num_envs: int
    num_episodes: int
    evaluation_mode: str = EVALUATION_MODE_POLICY_ONLY
    deterministic: bool = True
    max_steps: int = 200_000
    max_consecutive_successes: int = 50

    def unit_key(self) -> str:
        """Stable identity for resume/skip bookkeeping."""
        payload = (
            f"{self.run_id}|w{self.worker_id}|it{self.checkpoint_iteration}"
            f"|tol{self.fixed_tolerance:.6f}|seed{self.reset_seed}"
            f"|mode{self.evaluation_mode}|det{int(self.deterministic)}"
            f"|envs{self.num_envs}|eps{self.num_episodes}"
        )
        return hashlib.sha256(payload.encode()).hexdigest()[:16]

    def config_hash(self) -> str:
        d = dataclasses.asdict(self)
        return hashlib.sha256(
            json.dumps(d, sort_keys=True).encode()
        ).hexdigest()[:16]


def build_units(
    run_id: str,
    checkpoints: Iterable[dict],
    tolerances_m: Iterable[float],
    seeds: Iterable[int],
    num_envs: int,
    num_episodes: int,
    evaluation_mode: str = EVALUATION_MODE_POLICY_ONLY,
    deterministic: bool = True,
    max_steps: int = 200_000,
    max_consecutive_successes: int = 50,
) -> list[EvalUnit]:
    """Cross product checkpoints x tolerances x seeds.

    `checkpoints` rows need: worker_id, iteration, frame, checkpoint_path.
    """
    units: list[EvalUnit] = []
    for ck in checkpoints:
        for tol in tolerances_m:
            if not (0.001 <= float(tol) <= 0.2):
                raise ValueError(
                    f"tolerance {tol} out of sane meter range [0.001, 0.2]; "
                    "tolerances must be meters, not cm"
                )
            for seed in seeds:
                units.append(EvalUnit(
                    run_id=run_id,
                    worker_id=int(ck["worker_id"]),
                    checkpoint_iteration=int(ck["iteration"]),
                    checkpoint_frame=int(ck["frame"]),
                    checkpoint_path=str(ck["checkpoint_path"]),
                    fixed_tolerance=float(tol),
                    reset_seed=int(seed),
                    num_envs=int(num_envs),
                    num_episodes=int(num_episodes),
                    evaluation_mode=evaluation_mode,
                    deterministic=deterministic,
                    max_steps=int(max_steps),
                    max_consecutive_successes=int(max_consecutive_successes),
                ))
    return units


def result_path(raw_dir: Path, unit: EvalUnit) -> Path:
    """Per-unit raw result JSON path (also the resume/skip marker)."""
    return raw_dir / (
        f"w{unit.worker_id:02d}_it{unit.checkpoint_iteration:06d}"
        f"_tol{unit.fixed_tolerance:.4f}_seed{unit.reset_seed}"
        f"_{unit.evaluation_mode}_{unit.unit_key()}.json"
    )


def is_done(raw_dir: Path, unit: EvalUnit) -> bool:
    p = result_path(raw_dir, unit)
    if not p.is_file():
        return False
    try:
        d = json.loads(p.read_text())
        return d.get("status") == "ok" and d.get("schema_version") == SCHEMA_VERSION
    except Exception:
        return False


def quantiles(sorted_vals: list[float], qs: Iterable[float]) -> dict[str, float]:
    """Simple linear-interpolation quantiles on pre-sorted values."""
    out: dict[str, float] = {}
    n = len(sorted_vals)
    for q in qs:
        key = f"q{int(round(q * 100))}"
        if n == 0:
            out[key] = float("nan")
            continue
        pos = q * (n - 1)
        lo = int(math.floor(pos))
        hi = int(math.ceil(pos))
        frac = pos - lo
        out[key] = float(sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac)
    return out


def wilson_ci(successes: int, total: int, z: float = 1.959964) -> tuple[float, float]:
    """95% Wilson score interval for a binomial proportion."""
    if total <= 0:
        return (float("nan"), float("nan"))
    p = successes / total
    denom = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denom
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def mean_ci95(vals: list[float]) -> tuple[float, float, float]:
    """(mean, lo, hi) normal-approx 95% CI for a mean."""
    n = len(vals)
    if n == 0:
        return (float("nan"),) * 3
    m = sum(vals) / n
    if n == 1:
        return (m, float("nan"), float("nan"))
    var = sum((v - m) ** 2 for v in vals) / (n - 1)
    half = 1.959964 * math.sqrt(var / n)
    return (m, m - half, m + half)


def make_result_row(
    unit: EvalUnit,
    episode_nsucc: list[float],
    episode_final_error: list[float],
    git_sha: str,
    checkpoint_hash: str,
    extra: Optional[dict[str, Any]] = None,
    keep_per_episode: bool = True,
    applied_tolerance_start: Optional[float] = None,
    applied_tolerance_end: Optional[float] = None,
) -> dict[str, Any]:
    """Aggregate one finished unit into the raw result row.

    Success definition for CCRS R(k, tau): an episode counts as success when
    it achieved at least one success under the FIXED tolerance tau
    (nsucc > 0). nsucc-based metrics are also kept for continuity with
    training curves.
    """
    n = len(episode_nsucc)
    n_succ = sum(1 for v in episode_nsucc if v > 0)
    sr_lo, sr_hi = wilson_ci(n_succ, n)
    ns_mean, ns_lo, ns_hi = mean_ci95(episode_nsucc)
    errs = sorted(float(e) for e in episode_final_error if not math.isnan(e))
    qs = quantiles(errs, [0.10, 0.25, 0.50, 0.75, 0.90])

    if applied_tolerance_start is not None and applied_tolerance_end is not None:
        drift = abs(applied_tolerance_start - applied_tolerance_end)
        if drift > 1e-9:
            raise RuntimeError(
                f"fixed tolerance drifted during eval: "
                f"{applied_tolerance_start} -> {applied_tolerance_end}"
            )

    row: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "ok",
        # identity
        "run_id": unit.run_id,
        "worker_id": unit.worker_id,
        "checkpoint_iteration": unit.checkpoint_iteration,
        "checkpoint_frame": unit.checkpoint_frame,
        "checkpoint_path": unit.checkpoint_path,
        "checkpoint_hash": checkpoint_hash,
        "git_sha": git_sha,
        "config_hash": unit.config_hash(),
        "unit_key": unit.unit_key(),
        # protocol
        "fixed_tolerance": unit.fixed_tolerance,
        "fixed_tolerance_cm": unit.fixed_tolerance * 100.0,
        "reset_seed": unit.reset_seed,
        "evaluation_mode": unit.evaluation_mode,
        "deterministic": unit.deterministic,
        "num_envs": unit.num_envs,
        "max_consecutive_successes": unit.max_consecutive_successes,
        # raw metrics
        "num_episodes": n,
        "num_success": n_succ,
        "success_rate": (n_succ / n) if n else float("nan"),
        "success_rate_ci95_lo": sr_lo,
        "success_rate_ci95_hi": sr_hi,
        "nsucc_mean": ns_mean,
        "nsucc_mean_ci95_lo": ns_lo,
        "nsucc_mean_ci95_hi": ns_hi,
        "nsucc_std": (
            math.sqrt(sum((v - ns_mean) ** 2 for v in episode_nsucc) / (n - 1))
            if n > 1 else float("nan")
        ),
        "final_error_mean": (sum(errs) / len(errs)) if errs else float("nan"),
        "final_error_median": qs["q50"],
        "final_error_q10": qs["q10"],
        "final_error_q25": qs["q25"],
        "final_error_q50": qs["q50"],
        "final_error_q75": qs["q75"],
        "final_error_q90": qs["q90"],
    }
    if keep_per_episode:
        row["per_episode_nsucc"] = [float(v) for v in episode_nsucc]
        row["per_episode_final_error"] = [float(e) for e in episode_final_error]
    if extra:
        row.update(extra)
    return row


CSV_COLUMNS = [
    "run_id", "worker_id", "checkpoint_iteration", "checkpoint_frame",
    "fixed_tolerance", "fixed_tolerance_cm", "reset_seed", "evaluation_mode",
    "deterministic", "num_envs", "num_episodes", "num_success",
    "success_rate", "success_rate_ci95_lo", "success_rate_ci95_hi",
    "nsucc_mean", "nsucc_mean_ci95_lo", "nsucc_mean_ci95_hi", "nsucc_std",
    "final_error_mean", "final_error_median",
    "final_error_q10", "final_error_q25", "final_error_q50",
    "final_error_q75", "final_error_q90",
    "git_sha", "config_hash", "checkpoint_hash", "checkpoint_path",
]


def aggregate_raw_dir(raw_dir: Path, jsonl_out: Path, csv_out: Path) -> int:
    """Collect all per-unit raw jsons into ccrs_results.jsonl / .csv.

    Per-episode arrays are kept in the jsonl but not the csv.
    Returns number of rows written.
    """
    import csv as _csv
    rows = []
    for p in sorted(raw_dir.glob("*.json")):
        try:
            d = json.loads(p.read_text())
        except Exception:
            continue
        if d.get("status") == "ok" and d.get("schema_version") == SCHEMA_VERSION:
            rows.append(d)
    jsonl_out.parent.mkdir(parents=True, exist_ok=True)
    with jsonl_out.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    with csv_out.open("w", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    return len(rows)
