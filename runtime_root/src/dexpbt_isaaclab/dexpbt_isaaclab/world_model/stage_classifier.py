"""
Stage classifier for the stage-aware Bayesian selector.

Given per-policy training-state features, returns one of four stage labels
that the posterior buckets sub-counts under.

Priority (high to low):
    recovery > early > acquisition > mastery

Recovery is triggered for any restart whose ``last_restart_type`` is in
RECOVERY_TRIGGERS (set widened per plan revision: cross_copy, recovery,
oom_relaunch, manual_restart, startup_retry — all the cases where the
policy's optimizer / curriculum state is fresh and we want to gate strategy
choice differently).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


# Frozen tuple to make membership checks tiny.
RECOVERY_TRIGGERS: frozenset[str] = frozenset({
    "cross_copy",
    "recovery",
    "oom_relaunch",
    "manual_restart",
    "startup_retry",
})

VALID_STAGES: tuple[str, ...] = ("recovery", "early", "acquisition", "mastery")


@dataclass
class StageFeatures:
    """Inputs to ``infer_stage``.

    Mirrors the payload produced by ``DexPBTPbtAlgoObserver`` when writing a
    CF request. All fields default to neutral values so a partial dict still
    classifies sensibly.
    """

    env_frames: int = 0
    nsucc_mean: float = 0.0
    nsucc_max: float = 0.0
    success_tolerance: float = 0.075
    frames_since_restart: int = 10**18  # default: "long ago / no recent restart"
    last_restart_type: str = "none"
    policy_idx: int = 0

    # Constants drive the thresholds; expose as kwargs for tests.
    early_frames_threshold: int = 800_000_000
    early_nsucc_threshold: float = 3.0
    acquisition_nsucc_threshold: float = 15.0
    recovery_frames_window: int = 100_000_000

    @classmethod
    def from_dict(cls, d: dict) -> "StageFeatures":
        # Allow constants to be overridden too, but only consume fields we know.
        kwargs = {}
        for f in cls.__dataclass_fields__:
            if f in d:
                kwargs[f] = d[f]
        return cls(**kwargs)


def infer_stage(features: StageFeatures) -> str:
    """Return one of {'recovery', 'early', 'acquisition', 'mastery'}."""
    # 1. Recovery (highest priority): recent restart whose type is in the
    #    recovery trigger set.
    if (features.frames_since_restart < features.recovery_frames_window
            and features.last_restart_type in RECOVERY_TRIGGERS):
        return "recovery"

    # 2. Early: low frames OR low nsucc.
    if (features.env_frames < features.early_frames_threshold
            or features.nsucc_mean < features.early_nsucc_threshold):
        return "early"

    # 3. Acquisition: middle band of nsucc.
    if features.nsucc_mean < features.acquisition_nsucc_threshold:
        return "acquisition"

    # 4. Mastery.
    return "mastery"


def stage_features_payload(
    *,
    env_frames: int,
    nsucc_mean: float,
    nsucc_max: float,
    success_tolerance: float,
    frames_since_restart: int,
    last_restart_type: str,
    policy_idx: int,
) -> dict:
    """Serialisable dict embedded in CF requests / results."""
    return {
        "env_frames": int(env_frames),
        "nsucc_mean": float(nsucc_mean),
        "nsucc_max": float(nsucc_max),
        "success_tolerance": float(success_tolerance),
        "frames_since_restart": int(frames_since_restart),
        "last_restart_type": str(last_restart_type),
        "policy_idx": int(policy_idx),
    }


__all__ = [
    "RECOVERY_TRIGGERS",
    "VALID_STAGES",
    "StageFeatures",
    "infer_stage",
    "stage_features_payload",
]
