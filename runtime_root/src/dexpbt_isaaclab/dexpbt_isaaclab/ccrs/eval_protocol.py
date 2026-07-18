#!/usr/bin/env python3
"""CCRS fixed-tolerance evaluation protocol (pure python, no Isaac import).

Defines the evaluation unit (checkpoint x tolerance x seed), config hashing,
completion/resume bookkeeping, reset fingerprinting, per-env sample schema
and the result-row schema. The GPU entry point
(dexpbt_isaaclab.scripts.ccrs_eval) imports this module so unit tests can
validate the protocol without Isaac.

Protocol versions
-----------------
SCHEMA_VERSION = 2
  v1: episodes collected in completion order, success = (nsucc > 0),
      terminal error read post-reset (unreliable). DEPRECATED.
  v2: paired resets (first episode per env, num_episodes <= num_envs),
      per-env samples with sample_id + reset fingerprint, primary metric
      normalized_nsucc, terminal error via eval-only pre-reset snapshot
      hook with explicit validity flag, strict incomplete/resume checks.

Metrics (METRIC_PROTOCOL v2)
----------------------------
- normalized_nsucc  (PRIMARY, "Q"):
      Q(k, tau) = E[min(nsucc, M) | ckpt k, fixed tol tau] / M,
      M = max_consecutive_successes (default 50).
- nsucc_mean / nsucc_std: raw capability metrics (kept for continuity).
- success_any_rate = P(nsucc > 0): auxiliary ONLY, saturates near 1.0 for
  strong policies and must not be used for primary verdicts. It is NOT
  P(error <= tau).

Units: all tolerances in meters (m). cm shown in reports is m * 100.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional

DEFAULT_TOLERANCES_M = [0.028, 0.025, 0.023, 0.022, 0.021, 0.020, 0.019, 0.018]
DEFAULT_SEEDS = [10001, 10002, 10003, 10004]
DEFAULT_MAX_CONSECUTIVE_SUCCESSES = 50

EVALUATION_MODE_POLICY_ONLY = "policy_only"
EVALUATION_MODE_SYSTEM_ON = "system_on"  # reserved; not mixed with main results

SCHEMA_VERSION = 2
RESET_PROTOCOL_VERSION = "v2_first_episode_per_env"
METRIC_PROTOCOL_VERSION = "v2_normalized_nsucc"
# Version of the downstream analysis contract (frontier/comparisons). Recorded
# in every raw row so results can be matched to the analysis code they were
# interpreted with.
ANALYSIS_PROTOCOL_VERSION = "v2.1_frontier"

# Fixed-point quantization step for reset fingerprints (meters / radians /
# quaternion components). Coarse enough to absorb float serialization noise,
# fine enough that genuinely different initial states cannot collide.
FINGERPRINT_QUANT = 1e-5

METRIC_PRIMARY = "normalized_nsucc"


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
    max_consecutive_successes: int = DEFAULT_MAX_CONSECUTIVE_SUCCESSES
    reset_protocol_version: str = RESET_PROTOCOL_VERSION
    metric_protocol_version: str = METRIC_PROTOCOL_VERSION

    def __post_init__(self) -> None:
        if self.num_episodes > self.num_envs:
            raise ValueError(
                f"paired-reset protocol requires num_episodes <= num_envs, "
                f"got episodes={self.num_episodes} > envs={self.num_envs}"
            )

    def checkpoint_identity(self) -> str:
        """Stable checkpoint identity: worker/iteration/frame/path."""
        return (
            f"w{self.worker_id}|it{self.checkpoint_iteration}"
            f"|fr{self.checkpoint_frame}|{self.checkpoint_path}"
        )

    def unit_key(self) -> str:
        """Stable identity for resume/skip bookkeeping.

        Covers checkpoint identity (incl. frame + path), tolerance, seed,
        rollout limits and protocol versions so that any protocol change
        invalidates old results.
        """
        payload = (
            f"{self.run_id}|{self.checkpoint_identity()}"
            f"|tol{self.fixed_tolerance:.6f}|seed{self.reset_seed}"
            f"|mode{self.evaluation_mode}|det{int(self.deterministic)}"
            f"|envs{self.num_envs}|eps{self.num_episodes}"
            f"|steps{self.max_steps}|mcs{self.max_consecutive_successes}"
            f"|rp{self.reset_protocol_version}|mp{self.metric_protocol_version}"
        )
        return hashlib.sha256(payload.encode()).hexdigest()[:16]

    def config_hash(self) -> str:
        d = dataclasses.asdict(self)
        return hashlib.sha256(
            json.dumps(d, sort_keys=True).encode()
        ).hexdigest()[:16]


def make_sample_id(reset_seed: int, env_id: int) -> str:
    return f"seed_{reset_seed}_env_{env_id:03d}"


# ---------------------------------------------------------------------------
# Code provenance (recorded per raw row so results remain attributable even
# when the evaluator ran from a dirty worktree)
# ---------------------------------------------------------------------------

def git_dirty(repo_dir: Path) -> Optional[bool]:
    """True when the worktree has uncommitted tracked changes; None when git
    status cannot be determined."""
    import subprocess
    try:
        out = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=str(repo_dir), capture_output=True, text=True, check=True,
        ).stdout
        return bool(out.strip())
    except Exception:
        return None


def source_sha256(*paths: Path) -> str:
    """SHA-256 over the concatenated contents of the given source files
    (sorted by name for stability). Identifies the ACTUAL evaluation code
    regardless of git state."""
    h = hashlib.sha256()
    for p in sorted(paths, key=lambda x: str(x)):
        h.update(str(p.name).encode())
        h.update(Path(p).read_bytes())
    return h.hexdigest()[:16]


@dataclass
class EpisodeSample:
    """First completed episode of one env under the paired-reset protocol."""
    sample_id: str
    reset_seed: int
    env_id: int
    nsucc: float
    normalized_nsucc: float
    terminal_error: float          # NaN when not valid
    terminal_error_valid: bool
    dropped: Optional[bool]        # None when not valid

    @staticmethod
    def create(
        reset_seed: int,
        env_id: int,
        nsucc: float,
        max_consecutive_successes: int,
        terminal_error: float = float("nan"),
        terminal_error_valid: bool = False,
        dropped: Optional[bool] = None,
    ) -> "EpisodeSample":
        m = float(max_consecutive_successes)
        return EpisodeSample(
            sample_id=make_sample_id(reset_seed, env_id),
            reset_seed=int(reset_seed),
            env_id=int(env_id),
            nsucc=float(nsucc),
            normalized_nsucc=min(float(nsucc), m) / m,
            terminal_error=(float(terminal_error) if terminal_error_valid
                            else float("nan")),
            terminal_error_valid=bool(terminal_error_valid),
            dropped=(bool(dropped) if terminal_error_valid and dropped is not None
                     else None),
        )


def normalized_nsucc_mean(nsucc_values: Iterable[float],
                          max_consecutive_successes: int) -> float:
    """Q estimator: mean(min(nsucc, M)) / M."""
    vals = [float(v) for v in nsucc_values]
    if not vals:
        return float("nan")
    m = float(max_consecutive_successes)
    return sum(min(v, m) for v in vals) / (len(vals) * m)


# ---------------------------------------------------------------------------
# Reset fingerprints
# ---------------------------------------------------------------------------

def quantize_values(values: Iterable[float],
                    quant: float = FINGERPRINT_QUANT) -> list[int]:
    """Fixed-precision quantization to kill float serialization noise."""
    return [int(round(float(v) / quant)) for v in values]


def fingerprint_env_state(field_map: Mapping[str, Iterable[float]],
                          quant: float = FINGERPRINT_QUANT) -> str:
    """Hash one env's initial task state.

    field_map: field name -> flat iterable of floats for ONE env
    (e.g. object_pos, object_rot, goal_pos, goal_rot, joint_pos).
    Field names are part of the hash so adding fields changes fingerprints
    loudly instead of silently.
    """
    h = hashlib.sha256()
    for name in sorted(field_map.keys()):
        h.update(name.encode())
        h.update(json.dumps(quantize_values(field_map[name], quant)).encode())
    return h.hexdigest()[:16]


def aggregate_fingerprint(per_env: Mapping[int, str]) -> str:
    """Order-independent digest over {env_id: fingerprint}."""
    h = hashlib.sha256()
    for env_id in sorted(per_env.keys()):
        h.update(f"{env_id}:{per_env[env_id]};".encode())
    return h.hexdigest()[:16]


# ---------------------------------------------------------------------------
# Result persistence / resume
# ---------------------------------------------------------------------------

def result_path(raw_dir: Path, unit: EvalUnit) -> Path:
    """Per-unit raw result JSON path (also the resume/skip marker)."""
    return raw_dir / (
        f"w{unit.worker_id:02d}_it{unit.checkpoint_iteration:06d}"
        f"_tol{unit.fixed_tolerance:.4f}_seed{unit.reset_seed}"
        f"_{unit.evaluation_mode}_{unit.unit_key()}.json"
    )


def is_done(raw_dir: Path, unit: EvalUnit) -> bool:
    """Strict resume check. A unit counts as done ONLY when the stored row:

    - parses, status == "ok",
    - schema_version == SCHEMA_VERSION,
    - config_hash matches this exact EvalUnit,
    - actual_num_episodes == requested_num_episodes,
    - evaluation_mode / fixed_tolerance / checkpoint identity match,
    - reset & metric protocol versions match.

    Note: result_path() embeds unit_key, so protocol/config changes already
    map to a different file; the field checks below defend against manual
    renames or partially written files.
    """
    p = result_path(raw_dir, unit)
    if not p.is_file():
        return False
    try:
        d = json.loads(p.read_text())
    except Exception:
        return False
    try:
        return bool(
            d.get("status") == "ok"
            and int(d.get("schema_version", -1)) == SCHEMA_VERSION
            and d.get("config_hash") == unit.config_hash()
            and int(d.get("actual_num_episodes", -1)) == int(unit.num_episodes)
            and int(d.get("requested_num_episodes", -2)) == int(unit.num_episodes)
            and d.get("evaluation_mode") == unit.evaluation_mode
            and abs(float(d.get("fixed_tolerance", float("nan")))
                    - unit.fixed_tolerance) < 1e-12
            and d.get("checkpoint_path") == unit.checkpoint_path
            and int(d.get("checkpoint_frame", -1)) == int(unit.checkpoint_frame)
            and int(d.get("checkpoint_iteration", -1)) == int(unit.checkpoint_iteration)
            and d.get("reset_protocol_version") == unit.reset_protocol_version
            and d.get("metric_protocol_version") == unit.metric_protocol_version
        )
    except (TypeError, ValueError):
        return False


def write_result_atomic(raw_dir: Path, unit: EvalUnit, row: dict) -> Path:
    """Write raw result via temp file + atomic rename."""
    out = result_path(raw_dir, unit)
    tmp = out.with_suffix(".tmp")
    tmp.write_text(json.dumps(row))
    tmp.replace(out)
    return out


# ---------------------------------------------------------------------------
# Unit building
# ---------------------------------------------------------------------------

def _validate_tolerance_m(tol: float) -> None:
    if not (0.001 <= float(tol) <= 0.2):
        raise ValueError(
            f"tolerance {tol} out of sane meter range [0.001, 0.2]; "
            "tolerances must be meters, not cm"
        )


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
    max_consecutive_successes: int = DEFAULT_MAX_CONSECUTIVE_SUCCESSES,
) -> list[EvalUnit]:
    """Cross product checkpoints x tolerances x seeds.

    `checkpoints` rows need: worker_id, iteration, frame, checkpoint_path.
    A checkpoint row may carry its own "tolerances" list (used for exact
    transition cells); when present it REPLACES the global grid for that
    checkpoint.
    """
    units: list[EvalUnit] = []
    global_tols = [float(t) for t in tolerances_m]
    for ck in checkpoints:
        tols = [float(t) for t in ck.get("tolerances", global_tols)]
        for tol in tols:
            _validate_tolerance_m(tol)
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
    # dedupe (checkpoint may appear in both grid and exact roles)
    seen: set[str] = set()
    out: list[EvalUnit] = []
    for u in units:
        k = u.unit_key()
        if k not in seen:
            seen.add(k)
            out.append(u)
    return out


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------

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
    samples: list[EpisodeSample],
    per_env_fingerprint: Mapping[int, str],
    git_sha: str,
    checkpoint_hash: str,
    max_steps_reached: bool = False,
    extra: Optional[dict[str, Any]] = None,
    applied_tolerance_start: Optional[float] = None,
    applied_tolerance_end: Optional[float] = None,
) -> dict[str, Any]:
    """Aggregate one finished (or incomplete) unit into the raw result row.

    status = "ok" ONLY when the number of collected per-env samples equals
    the requested num_episodes; otherwise "incomplete" with diagnostics.
    """
    if applied_tolerance_start is not None and applied_tolerance_end is not None:
        drift = abs(applied_tolerance_start - applied_tolerance_end)
        if drift > 1e-9:
            raise RuntimeError(
                f"fixed tolerance drifted during eval: "
                f"{applied_tolerance_start} -> {applied_tolerance_end}"
            )

    # per-env dedupe guard (protocol: exactly one sample per env)
    env_ids = [s.env_id for s in samples]
    if len(env_ids) != len(set(env_ids)):
        raise ValueError("duplicate env_id in samples; first-episode-per-env "
                         "protocol violated")

    n_req = int(unit.num_episodes)
    n_act = len(samples)
    complete = (n_act == n_req)
    expected_ids = set(range(n_req))
    missing_env_ids = sorted(expected_ids - set(env_ids))

    m = float(unit.max_consecutive_successes)
    nsuccs = [s.nsucc for s in samples]
    norm_vals = [s.normalized_nsucc for s in samples]
    n_any = sum(1 for v in nsuccs if v > 0)

    q_mean, q_lo, q_hi = mean_ci95(norm_vals)
    ns_mean, ns_lo, ns_hi = mean_ci95(nsuccs)
    any_lo, any_hi = wilson_ci(n_any, n_act)

    terminal_valid = bool(samples) and all(s.terminal_error_valid for s in samples)
    errs = sorted(s.terminal_error for s in samples
                  if s.terminal_error_valid and not math.isnan(s.terminal_error))
    qs = quantiles(errs, [0.10, 0.25, 0.50, 0.75, 0.90])
    drops = [s.dropped for s in samples if s.dropped is not None]

    row: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "ok" if complete else "incomplete",
        "reset_protocol_version": unit.reset_protocol_version,
        "metric_protocol_version": unit.metric_protocol_version,
        "analysis_protocol_version": ANALYSIS_PROTOCOL_VERSION,
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
        "max_steps": unit.max_steps,
        "max_consecutive_successes": unit.max_consecutive_successes,
        # completeness
        "requested_num_episodes": n_req,
        "actual_num_episodes": n_act,
        "num_episodes": n_act,
        "max_steps_reached": bool(max_steps_reached),
        "missing_env_ids": missing_env_ids,
        "stop_reason": ("all_episodes_collected" if complete
                        else ("max_steps_reached" if max_steps_reached
                              else "unknown_early_stop")),
        # PRIMARY metric (Q)
        "normalized_nsucc": q_mean,
        "normalized_nsucc_ci95_lo": q_lo,
        "normalized_nsucc_ci95_hi": q_hi,
        # raw capability metrics
        "nsucc_mean": ns_mean,
        "nsucc_mean_ci95_lo": ns_lo,
        "nsucc_mean_ci95_hi": ns_hi,
        "nsucc_std": (
            math.sqrt(sum((v - ns_mean) ** 2 for v in nsuccs) / (n_act - 1))
            if n_act > 1 else float("nan")
        ),
        # auxiliary (NOT for primary verdicts; NOT P(error<=tau))
        "success_any_rate": (n_any / n_act) if n_act else float("nan"),
        "success_any_ci95_lo": any_lo,
        "success_any_ci95_hi": any_hi,
        "num_success_any": n_any,
        # terminal error (valid only with pre-reset snapshot hook)
        "terminal_error_valid": terminal_valid,
        "terminal_error_mean": (sum(errs) / len(errs)) if (terminal_valid and errs)
                               else float("nan"),
        "terminal_error_median": qs["q50"] if terminal_valid else float("nan"),
        "terminal_error_q10": qs["q10"] if terminal_valid else float("nan"),
        "terminal_error_q25": qs["q25"] if terminal_valid else float("nan"),
        "terminal_error_q50": qs["q50"] if terminal_valid else float("nan"),
        "terminal_error_q75": qs["q75"] if terminal_valid else float("nan"),
        "terminal_error_q90": qs["q90"] if terminal_valid else float("nan"),
        "dropped_rate": (sum(1 for d in drops if d) / len(drops))
                        if (terminal_valid and drops) else float("nan"),
        # reset pairing evidence
        "reset_fingerprint": aggregate_fingerprint(per_env_fingerprint),
        "per_env_reset_fingerprint": {str(k): v for k, v in
                                      sorted(per_env_fingerprint.items())},
        "sample_ids": [s.sample_id for s in
                       sorted(samples, key=lambda x: x.env_id)],
        # per-sample raw data (re-aggregatable)
        "samples": [dataclasses.asdict(s) for s in
                    sorted(samples, key=lambda x: x.env_id)],
    }
    if extra:
        row.update(extra)
    return row


CSV_COLUMNS = [
    "run_id", "worker_id", "checkpoint_iteration", "checkpoint_frame",
    "fixed_tolerance", "fixed_tolerance_cm", "reset_seed", "evaluation_mode",
    "deterministic", "num_envs",
    "requested_num_episodes", "actual_num_episodes", "status",
    "normalized_nsucc", "normalized_nsucc_ci95_lo", "normalized_nsucc_ci95_hi",
    "nsucc_mean", "nsucc_mean_ci95_lo", "nsucc_mean_ci95_hi", "nsucc_std",
    "success_any_rate", "success_any_ci95_lo", "success_any_ci95_hi",
    "terminal_error_valid", "terminal_error_mean", "terminal_error_median",
    "terminal_error_q10", "terminal_error_q25", "terminal_error_q50",
    "terminal_error_q75", "terminal_error_q90", "dropped_rate",
    "reset_fingerprint", "max_consecutive_successes",
    "reset_protocol_version", "metric_protocol_version",
    "git_sha", "config_hash", "checkpoint_hash", "checkpoint_path",
]


def aggregate_raw_dir(raw_dir: Path, jsonl_out: Path, csv_out: Path) -> int:
    """Collect all per-unit raw jsons into ccrs_results.jsonl / .csv.

    Only status=ok rows with the CURRENT schema version are aggregated;
    old-schema and incomplete rows are excluded (never silently mixed in).
    Per-sample arrays are kept in the jsonl but not the csv.
    Returns number of rows written.
    """
    import csv as _csv
    rows = []
    skipped_old_schema = 0
    skipped_incomplete = 0
    for p in sorted(raw_dir.glob("*.json")):
        try:
            d = json.loads(p.read_text())
        except Exception:
            continue
        if int(d.get("schema_version", -1)) != SCHEMA_VERSION:
            skipped_old_schema += 1
            continue
        if d.get("status") != "ok":
            skipped_incomplete += 1
            continue
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
    if skipped_old_schema or skipped_incomplete:
        print(f"[ccrs.aggregate] skipped {skipped_old_schema} old-schema and "
              f"{skipped_incomplete} incomplete raw files")
    return len(rows)
