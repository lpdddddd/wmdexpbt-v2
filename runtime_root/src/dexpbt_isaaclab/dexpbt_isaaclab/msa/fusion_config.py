"""MSA-aware DexPBT fusion configuration.

Internal tolerance units are METERS (same as the rest of DexPBT / MSA pilot).
Centimeters appear only at the config boundary; conversion happens HERE only.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields
from typing import Any, Mapping, Optional

# Pilot-aligned defaults (meters unless named *_cm).
DEFAULT_BUILD_THRESHOLD_M = 0.030          # 3.0 cm
DEFAULT_STRICT_THRESHOLD_M = 0.020         # 2.0 cm
DEFAULT_TARGET_FLOOR_M = 0.010             # tau_min / pilot TAU_FLOOR
DEFAULT_TIGHTEN_RATIO = 0.95               # gamma (pilot tau_plus)
DEFAULT_PREFILTER_EVAL_N = 4
DEFAULT_STRICT_EVAL_N = 16
DEFAULT_PROBE_BUDGET_FRAMES = 80_000_000    # one committed tick (pilot packet+tick)
DEFAULT_CYCLE_INTERVAL_FRAMES = 100_000_000
DEFAULT_HOLD_COOLDOWN_FRAMES = 80_000_000
DEFAULT_MAX_CONSECUTIVE_FAILURES = 3
DEFAULT_RETENTION_MIN = 0.88               # pilot rho threshold
DEFAULT_BOOTSTRAP_SAMPLES = 10_000         # pilot BOOTSTRAP_REPEATS
DEFAULT_CONFIDENCE_LEVEL = 0.05            # pilot PANEL_QUANTILE (lower)
DEFAULT_MIN_ANCHOR_WORKERS = 1
MSA_CHECKPOINT_VERSION = "msa_fusion_v1"
EVIDENCE_VERSION = "msa_fusion_v1"


def cm_to_m(cm: float) -> float:
    """Single conversion site: centimeters -> meters."""
    return float(cm) * 0.01


def m_to_cm(meters: float) -> float:
    return float(meters) * 100.0


@dataclass
class MSAFusionConfig:
    """Fusion-profile MSA controller knobs (all thresholds in meters)."""

    msa_integrated_enabled: bool = False
    msa_build_threshold_m: float = DEFAULT_BUILD_THRESHOLD_M
    msa_strict_threshold_m: float = DEFAULT_STRICT_THRESHOLD_M
    msa_target_floor_m: float = DEFAULT_TARGET_FLOOR_M
    msa_tighten_ratio: float = DEFAULT_TIGHTEN_RATIO
    msa_prefilter_eval_n: int = DEFAULT_PREFILTER_EVAL_N
    msa_strict_eval_n: int = DEFAULT_STRICT_EVAL_N
    msa_probe_budget_frames: int = DEFAULT_PROBE_BUDGET_FRAMES
    msa_cycle_interval_frames: int = DEFAULT_CYCLE_INTERVAL_FRAMES
    msa_hold_cooldown_frames: int = DEFAULT_HOLD_COOLDOWN_FRAMES
    msa_max_consecutive_failures: int = DEFAULT_MAX_CONSECUTIVE_FAILURES
    msa_retention_min: float = DEFAULT_RETENTION_MIN
    msa_bootstrap_samples: int = DEFAULT_BOOTSTRAP_SAMPLES
    msa_confidence_level: float = DEFAULT_CONFIDENCE_LEVEL
    msa_require_population_guard: bool = True
    msa_min_anchor_workers: int = DEFAULT_MIN_ANCHOR_WORKERS
    msa_fail_closed: bool = True
    msa_checkpoint_version: str = MSA_CHECKPOINT_VERSION
    msa_evidence_version: str = EVIDENCE_VERSION
    msa_inprocess_hook_enable: bool = False
    msa_fake_decision_injector_enable: bool = False  # test-only; formal profiles False

    def validate(self) -> None:
        if self.msa_build_threshold_m <= self.msa_strict_threshold_m:
            raise ValueError(
                "msa_build_threshold_m must be > msa_strict_threshold_m "
                f"(got {self.msa_build_threshold_m} vs "
                f"{self.msa_strict_threshold_m})")
        if self.msa_strict_threshold_m < self.msa_target_floor_m:
            raise ValueError(
                "msa_strict_threshold_m must be >= msa_target_floor_m")
        if not (0.0 < self.msa_tighten_ratio < 1.0):
            raise ValueError("msa_tighten_ratio must be in (0, 1)")
        if self.msa_prefilter_eval_n < 1 or self.msa_strict_eval_n < 1:
            raise ValueError("eval N must be >= 1")
        if self.msa_probe_budget_frames <= 0:
            raise ValueError("msa_probe_budget_frames must be > 0")
        if self.msa_hold_cooldown_frames < 0:
            raise ValueError("msa_hold_cooldown_frames must be >= 0")
        if self.msa_max_consecutive_failures < 1:
            raise ValueError("msa_max_consecutive_failures must be >= 1")
        if not (0.0 < self.msa_retention_min <= 1.5):
            raise ValueError("msa_retention_min out of sane range")
        if not (0.0 < self.msa_confidence_level < 0.5):
            raise ValueError("msa_confidence_level must be in (0, 0.5)")
        if self.msa_min_anchor_workers < 1:
            raise ValueError("msa_min_anchor_workers must be >= 1")
        if self.msa_fake_decision_injector_enable and not self.msa_inprocess_hook_enable:
            raise ValueError(
                "msa_fake_decision_injector_enable requires "
                "msa_inprocess_hook_enable (test-only path)")

    def stage_for_tolerance(self, tau_m: float) -> str:
        """Return BUILD | MID | STRICT for a live tolerance in meters."""
        t = float(tau_m)
        if t > self.msa_build_threshold_m:
            return "BUILD"
        if t > self.msa_strict_threshold_m:
            return "MID"
        return "STRICT"

    def probe_target(self, tau_m: float) -> float:
        """tau_plus = max(gamma * tau, floor) — pilot formula."""
        return max(self.msa_target_floor_m,
                   float(self.msa_tighten_ratio) * float(tau_m))

    def eval_n_for_tolerance(self, tau_m: float) -> int:
        stage = self.stage_for_tolerance(tau_m)
        if stage == "STRICT":
            return int(self.msa_strict_eval_n)
        if stage == "MID":
            return int(self.msa_prefilter_eval_n)
        return 0  # BUILD: no paired N eval by default

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def config_hash(self) -> str:
        import hashlib
        blob = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode()).hexdigest()[:16]


def fusion_config_from_mapping(raw: Mapping[str, Any] | None) -> MSAFusionConfig:
    """Build config from profile table / CLI dict.

    Accepts either meter keys (`msa_build_threshold_m`) or cm keys
    (`msa_build_threshold_cm`); cm wins conversion at this boundary only.
    """
    raw = dict(raw or {})
    known = {f.name for f in fields(MSAFusionConfig)}
    kwargs: dict[str, Any] = {}

    # cm -> m (single conversion site)
    cm_map = {
        "msa_build_threshold_cm": "msa_build_threshold_m",
        "msa_strict_threshold_cm": "msa_strict_threshold_m",
        "msa_target_floor_cm": "msa_target_floor_m",
    }
    for cm_key, m_key in cm_map.items():
        if cm_key in raw and raw[cm_key] is not None:
            kwargs[m_key] = cm_to_m(float(raw[cm_key]))

    for k, v in raw.items():
        if k in known and k not in kwargs:
            kwargs[k] = v

    cfg = MSAFusionConfig(**{k: v for k, v in kwargs.items() if k in known})
    cfg.validate()
    return cfg


def print_effective_config(cfg: MSAFusionConfig, *, sink=print) -> None:
    d = cfg.to_dict()
    d["msa_build_threshold_cm"] = m_to_cm(cfg.msa_build_threshold_m)
    d["msa_strict_threshold_cm"] = m_to_cm(cfg.msa_strict_threshold_m)
    d["msa_target_floor_cm"] = m_to_cm(cfg.msa_target_floor_m)
    d["config_hash"] = cfg.config_hash()
    sink("[msa-fusion] effective config:")
    sink(json.dumps(d, indent=2, sort_keys=True))
