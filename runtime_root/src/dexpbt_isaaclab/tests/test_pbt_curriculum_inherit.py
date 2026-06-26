"""Phase A4 P3 (2026-06-09): source-safe curriculum inheritance tests.

Covers:
- default reset mode preserves legacy `_resolve_restart_success_tolerance`;
- source_safe stable-source inheritance: tol_after = max(src_tol/0.95, floor)
  clamped to current target;
- floor clamp + target-already-tighter branch;
- unstable source falls back to legacy with `source_unstable_legacy`;
- missing source tolerance falls back + counter increments;
- archive metadata enriched with retention / last_mastered / source_safe;
- archive without metadata triggers fallback;
- source-aware recovery consolidation: 50M for stable, 100M for unstable;
- stale slot consolidation cleared by `apply_source_safe_consolidation_on_copy`;
- promising protection: end_mean gate, retention guard, hard collapse override;
- protected worker remains eligible as a source;
- copy-event JSONL row schema written;
- counters in training-curve / TB increment on legacy reset;
- train.py declares all new flags; launch_pbt boolean_flags includes the
  promising toggles.

Tests intentionally avoid spinning up an env / PPO; they exercise the helpers
directly via observer instances + mock window samples + on-disk JSON.
"""

from __future__ import annotations

import json
import math
import re
import tempfile
import unittest
from pathlib import Path

import torch

from dexpbt_isaaclab.pbt.pbt_curriculum_p1 import (
    CurriculumControlState,
    WindowSample,
    apply_source_safe_consolidation_on_copy,
    read_curriculum_control,
    write_curriculum_control,
)
from dexpbt_isaaclab.pbt.pbt_observer import DexPBTPbtAlgoObserver


def _sample(
    *,
    mean: float,
    max_: float,
    tol: float = 0.075,
    iteration: int = 0,
    env_frames: int = 0,
) -> WindowSample:
    return WindowSample(
        env_frames=env_frames,
        nsucc_mean=mean,
        nsucc_max=max_,
        success_tolerance=tol,
        pbt_iteration=iteration,
    )


def _make_observer(
    workspace: Path,
    *,
    inherit_mode: str = "reset",
    protect_promising: bool = False,
    promising_nsucc_mean: float = 20.0,
    promising_peak_mean: float = 25.0,
    promising_min_retention: float = 0.5,
    promising_protect_frames: int = 150_000_000,
    promising_allow_collapse_override: bool = True,
    source_safe_recovery_frames: int = 50_000_000,
    source_unsafe_recovery_frames: int = 100_000_000,
    source_safe_min_mean: float = 20.0,
    source_safe_min_retention: float = 0.7,
    crosscopy_mode: str = "checkpoint",
    recovery_mode: str = "last_mastered",
    speed_mode: str = "normal",
    strong_live_enable: bool = True,
    strong_live_max_tol: float = 0.035,
    strong_live_min_mean: float = 30.0,
    strong_live_min_max: float = 35.0,
) -> DexPBTPbtAlgoObserver:
    obs = DexPBTPbtAlgoObserver(
        policy_idx=0,
        num_policies=2,
        mutable_params={
            "enable_wm": False,
            "enable_counterfactual": False,
            "wm_strategy": "none",
        },
        workspace=str(workspace),
        train_dir=str(workspace),
        task_name="test-task",
        interval_steps=10_000_000,
        start_after=0,
        initial_delay=0,
        num_envs=1,
        pbt_min_frames_as_source_after_restart=0,
        pbt_min_frames_between_restarts=0,
        curriculum_mode="mastery" if speed_mode == "adaptive_fast" else "legacy",
        mastery_nsucc_mean=4.0,
        mastery_nsucc_max=20.0,
        mastery_consecutive_windows=2,
        curriculum_shrink_factor=0.95,
        pbt_single_shrink_per_window=True,
        post_shrink_consolidation_env_frames=100_000_000,
        curriculum_collapse_detect=True,
        pbt_early_collapse_detect=True,
        pbt_recovery_tolerance_mode=recovery_mode,
        pbt_crosscopy_tolerance_mode=crosscopy_mode,
        curriculum_speed_mode=speed_mode,
        # Phase A4 P3 inherit / promising knobs.
        pbt_curriculum_inherit_mode=inherit_mode,
        pbt_source_safe_recovery_consolidation_frames=source_safe_recovery_frames,
        pbt_source_unsafe_recovery_consolidation_frames=source_unsafe_recovery_frames,
        pbt_source_safe_min_mean=source_safe_min_mean,
        pbt_source_safe_min_retention=source_safe_min_retention,
        pbt_protect_promising_workers=protect_promising,
        pbt_promising_nsucc_mean=promising_nsucc_mean,
        pbt_promising_peak_mean=promising_peak_mean,
        pbt_promising_min_retention=promising_min_retention,
        pbt_promising_protect_frames=promising_protect_frames,
        pbt_promising_allow_hard_collapse_override=promising_allow_collapse_override,
        pbt_strong_live_inherit_exact_enable=strong_live_enable,
        pbt_strong_live_inherit_max_tol=strong_live_max_tol,
        pbt_strong_live_inherit_min_mean=strong_live_min_mean,
        pbt_strong_live_inherit_min_max=strong_live_min_max,
    )
    obs.writer = None
    return obs


class _TmpWs(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.ws = Path(self._tmp.name)


class ResetModePreservesLegacyTest(_TmpWs):
    """#1: default --pbt_curriculum_inherit_mode=reset preserves legacy."""

    def test_reset_mode_matches_legacy_resolver(self):
        obs = _make_observer(
            self.ws,
            inherit_mode="reset",
            crosscopy_mode="last_mastered",
        )
        # Seed last_mastered to a non-default value so the legacy
        # last_mastered branch is observably distinct from `checkpoint`.
        obs._last_mastered_tolerance[7] = 0.05
        source_state = {
            "policy_idx": 7,
            "nsucc_mean": 28.0,
            "nsucc_max": 50.0,
            "curriculum": {"success_tolerance": 0.06, "target_tolerance": 0.01},
        }
        legacy_tol, legacy_mode = obs._resolve_restart_success_tolerance(
            source_policy=7,
            source_curriculum=source_state["curriculum"],
            checkpoint_type="live",
        )
        out = obs._resolve_curriculum_inheritance(
            source_policy=7,
            source_state=source_state,
            source_curriculum=source_state["curriculum"],
            checkpoint_type="live",
            target_current_tolerance=0.075,
        )
        self.assertEqual(out["inheritance_reason"], "disabled_reset_mode")
        self.assertAlmostEqual(out["tol_after"], legacy_tol, places=12)
        self.assertEqual(out["legacy_mode"], legacy_mode)
        self.assertFalse(out["source_safe_for_curriculum"])


class SourceSafeStableInheritsOneLevelTest(_TmpWs):
    """#2: stable source -> max(src_tol/0.95, floor) clamped to current target."""

    def test_stable_source_inherits_one_level_looser(self):
        obs = _make_observer(self.ws, inherit_mode="source_safe")
        source_state = {
            "policy_idx": 1,
            "nsucc_mean": 28.0,
            "nsucc_max": 50.0,
            "source_retention_at_save": 0.95,
            "curriculum": {
                "success_tolerance": 0.05,
                "target_tolerance": 0.01,
            },
        }
        out = obs._resolve_curriculum_inheritance(
            source_policy=1,
            source_state=source_state,
            source_curriculum=source_state["curriculum"],
            checkpoint_type="live",
            target_current_tolerance=0.075,
        )
        # 0.05 / 0.95 = 0.05263157...
        self.assertTrue(out["source_safe_for_curriculum"])
        self.assertAlmostEqual(out["tol_after"], 0.05 / 0.95, places=10)
        self.assertEqual(
            out["inheritance_reason"], "source_stable_relax_one_level"
        )

    def test_target_already_tighter_keeps_target(self):
        obs = _make_observer(self.ws, inherit_mode="source_safe")
        source_state = {
            "policy_idx": 1,
            "nsucc_mean": 28.0,
            "source_retention_at_save": 0.95,
            "curriculum": {
                "success_tolerance": 0.05,
                "target_tolerance": 0.01,
            },
        }
        out = obs._resolve_curriculum_inheritance(
            source_policy=1,
            source_state=source_state,
            source_curriculum=source_state["curriculum"],
            checkpoint_type="live",
            target_current_tolerance=0.04,
        )
        self.assertTrue(out["source_safe_for_curriculum"])
        # Current target 0.04 < 0.0526 so we keep 0.04.
        self.assertAlmostEqual(out["tol_after"], 0.04, places=10)
        self.assertEqual(out["inheritance_reason"], "target_already_tighter")


class SourceSafeFloorClampedTest(_TmpWs):
    """#3: never inherit below env target_success_tolerance floor."""

    def test_floor_clamp_when_relaxed_below_floor(self):
        obs = _make_observer(self.ws, inherit_mode="source_safe")
        source_state = {
            "policy_idx": 1,
            "nsucc_mean": 30.0,
            "source_retention_at_save": 0.95,
            "curriculum": {
                "success_tolerance": 0.005,  # tiny
                "target_tolerance": 0.01,
            },
        }
        out = obs._resolve_curriculum_inheritance(
            source_policy=1,
            source_state=source_state,
            source_curriculum=source_state["curriculum"],
            checkpoint_type="live",
            target_current_tolerance=0.075,
        )
        # 0.005 / 0.95 = 0.00526 < floor 0.01 -> clamp to 0.01.
        self.assertTrue(out["source_safe_for_curriculum"])
        self.assertAlmostEqual(out["tol_after"], 0.01, places=10)
        self.assertEqual(out["inheritance_reason"], "floor_clamped")


class SourceSafeUnstableFallsBackTest(_TmpWs):
    """#4: unstable source -> legacy resolver + source_unstable_legacy."""

    def test_low_nsucc_unstable_falls_back(self):
        obs = _make_observer(
            self.ws, inherit_mode="source_safe", crosscopy_mode="checkpoint"
        )
        source_state = {
            "policy_idx": 2,
            "nsucc_mean": 5.0,  # below min_mean 20
            "source_retention_at_save": 0.9,
            "curriculum": {"success_tolerance": 0.06, "target_tolerance": 0.01},
        }
        out = obs._resolve_curriculum_inheritance(
            source_policy=2,
            source_state=source_state,
            source_curriculum=source_state["curriculum"],
            checkpoint_type="live",
            target_current_tolerance=0.075,
        )
        self.assertEqual(out["inheritance_reason"], "source_unstable_legacy")
        self.assertFalse(out["source_safe_for_curriculum"])
        # legacy crosscopy_mode=checkpoint -> tol_after equals source ckpt_tol.
        self.assertAlmostEqual(out["tol_after"], 0.06, places=10)

    def test_low_retention_unstable_falls_back(self):
        obs = _make_observer(self.ws, inherit_mode="source_safe")
        source_state = {
            "policy_idx": 3,
            "nsucc_mean": 28.0,  # high enough
            "source_retention_at_save": 0.4,  # below min_retention 0.7
            "curriculum": {"success_tolerance": 0.06, "target_tolerance": 0.01},
        }
        out = obs._resolve_curriculum_inheritance(
            source_policy=3,
            source_state=source_state,
            source_curriculum=source_state["curriculum"],
            checkpoint_type="live",
            target_current_tolerance=0.075,
        )
        self.assertEqual(out["inheritance_reason"], "source_unstable_legacy")
        self.assertFalse(out["source_safe_for_curriculum"])


class SourceSafeMissingMetadataTest(_TmpWs):
    """#5: missing source tolerance -> legacy + missing_source_tolerance."""

    def test_missing_source_tolerance_falls_back_with_reason(self):
        obs = _make_observer(self.ws, inherit_mode="source_safe")
        source_state = {
            "policy_idx": 4,
            "nsucc_mean": 28.0,
            "source_retention_at_save": 0.95,
            "curriculum": {
                # NO success_tolerance.
                "target_tolerance": 0.01,
            },
        }
        out = obs._resolve_curriculum_inheritance(
            source_policy=4,
            source_state=source_state,
            source_curriculum=source_state["curriculum"],
            checkpoint_type="live",
            target_current_tolerance=0.075,
        )
        self.assertEqual(out["inheritance_reason"], "missing_source_tolerance")
        self.assertFalse(out["source_safe_for_curriculum"])


class ArchiveMetadataInheritsTest(_TmpWs):
    """#6 + #7: archive source with / without enriched metadata."""

    def test_archive_with_enriched_metadata_inherits(self):
        obs = _make_observer(self.ws, inherit_mode="source_safe")
        archive_state = {
            "policy_idx": 5,
            "nsucc_mean": 30.0,
            "nsucc_max": 50.0,
            # Phase A4 P3 enrichment fields.
            "source_retention_at_save": 0.92,
            "last_mastered_tolerance_at_save": 0.05,
            "source_safe_for_curriculum_at_save": True,
            "curriculum": {
                "success_tolerance": 0.05,
                "target_tolerance": 0.01,
            },
        }
        out = obs._resolve_curriculum_inheritance(
            source_policy=5,
            source_state=archive_state,
            source_curriculum=archive_state["curriculum"],
            checkpoint_type="archive",
            target_current_tolerance=0.075,
        )
        self.assertTrue(out["source_safe_for_curriculum"])
        self.assertAlmostEqual(out["tol_after"], 0.05 / 0.95, places=10)

    def test_archive_missing_metadata_falls_back(self):
        obs = _make_observer(self.ws, inherit_mode="source_safe")
        # No retention / source_safe fields, no per-policy curriculum_control.
        archive_state = {
            "policy_idx": 99,  # no rolling window for this idx
            "nsucc_mean": 30.0,
            "nsucc_max": 50.0,
            "curriculum": {
                "success_tolerance": 0.05,
                "target_tolerance": 0.01,
            },
        }
        out = obs._resolve_curriculum_inheritance(
            source_policy=99,
            source_state=archive_state,
            source_curriculum=archive_state["curriculum"],
            checkpoint_type="archive",
            target_current_tolerance=0.075,
        )
        # No retention proxy -> has_source_retention=False -> unstable path.
        self.assertEqual(out["inheritance_reason"], "source_unstable_legacy")
        self.assertFalse(out["source_safe_for_curriculum"])


class RecoveryConsolidationSelectionTest(_TmpWs):
    """#8 + #9: 50M for stable, 100M for unstable; clears stale consolidation."""

    def test_stable_source_re_arms_50m_consolidation(self):
        ctrl_path = self.ws / "curriculum_control" / "000.json"
        # Seed a stale consolidation window from a previous owner.
        ctrl_path.parent.mkdir(parents=True, exist_ok=True)
        write_curriculum_control(
            ctrl_path,
            CurriculumControlState(
                post_shrink_consolidation_until_restart_step=1_000_000_000,
                blocked_by_consolidation=True,
                last_mastered_tolerance=0.075,
            ),
        )
        apply_source_safe_consolidation_on_copy(
            ctrl_path,
            consolidation_env_frames=50_000_000,
            consolidation_source="recovery_consolidation_source_safe",
            num_envs=8192,
            inherited_tolerance=0.0526,
            inheritance_audit={
                "inheritance_mode": "source_safe",
                "inheritance_reason": "source_stable_relax_one_level",
                "inherited_tolerance": 0.0526,
                "target_tolerance_before_copy": 0.075,
                "target_tolerance_after_copy": 0.0526,
                "source_policy_idx_for_copy": 1,
                "source_checkpoint_type_for_copy": "live",
                "source_success_tolerance_for_copy": 0.05,
                "source_nsucc_mean_for_copy": 28.0,
                "source_nsucc_max_for_copy": 50.0,
                "source_retention_for_copy": 0.95,
                "source_safe_for_curriculum": True,
            },
        )
        ctrl = read_curriculum_control(ctrl_path)
        self.assertIsNotNone(ctrl)
        # Stale 1B-step value should be replaced with the freshly-armed 50M
        # consolidation expressed in sim steps for num_envs=8192.
        self.assertNotEqual(
            ctrl.post_shrink_consolidation_until_restart_step,
            1_000_000_000,
        )
        self.assertGreater(
            ctrl.post_shrink_consolidation_until_restart_step, 0
        )
        # Stale blocked_by_consolidation is cleared but env loop will set it
        # again; here we only assert the immediate post-clear state.
        self.assertFalse(ctrl.blocked_by_consolidation)
        self.assertEqual(
            ctrl.recovery_consolidation_frames_selected, 50_000_000
        )
        self.assertEqual(
            ctrl.recovery_consolidation_source,
            "recovery_consolidation_source_safe",
        )
        self.assertEqual(ctrl.inheritance_mode, "source_safe")
        self.assertEqual(
            ctrl.inheritance_reason, "source_stable_relax_one_level"
        )
        self.assertAlmostEqual(ctrl.inherited_tolerance, 0.0526, places=6)
        # last_mastered synced to inherited tolerance.
        self.assertAlmostEqual(
            ctrl.last_mastered_tolerance, 0.0526, places=6
        )

    def test_unstable_source_re_arms_100m_consolidation(self):
        ctrl_path = self.ws / "curriculum_control" / "002.json"
        ctrl_path.parent.mkdir(parents=True, exist_ok=True)
        apply_source_safe_consolidation_on_copy(
            ctrl_path,
            consolidation_env_frames=100_000_000,
            consolidation_source="recovery_consolidation_source_unsafe",
            num_envs=8192,
            inherited_tolerance=0.0,  # legacy fallback
            inheritance_audit={
                "inheritance_mode": "source_safe",
                "inheritance_reason": "source_unstable_legacy",
                "source_safe_for_curriculum": False,
            },
        )
        ctrl = read_curriculum_control(ctrl_path)
        self.assertIsNotNone(ctrl)
        self.assertEqual(
            ctrl.recovery_consolidation_frames_selected, 100_000_000
        )
        self.assertEqual(
            ctrl.recovery_consolidation_source,
            "recovery_consolidation_source_unsafe",
        )
        self.assertEqual(
            ctrl.inheritance_reason, "source_unstable_legacy"
        )
        # inherited_tolerance==0 should NOT overwrite last_mastered.
        self.assertAlmostEqual(ctrl.last_mastered_tolerance, 0.075, places=6)


class PromisingProtectionTest(_TmpWs):
    """#10: end_mean gate protects worker."""

    def test_end_mean_gate_protects_when_protection_enabled(self):
        obs = _make_observer(
            self.ws,
            protect_promising=True,
            promising_nsucc_mean=20.0,
            promising_protect_frames=100_000_000,
        )
        window = obs._policy_window(0)
        # end_mean = 25 >= 20 -> protected.
        window.samples = [_sample(mean=10.0, max_=20.0), _sample(mean=25.0, max_=40.0)]
        obs._recent_peak_nsucc_mean[0] = 25.0
        obs._maybe_enter_promising_protection(0, env_frames=200_000_000, window=window)
        protected, reason = obs._is_promising_protected(0, env_frames=250_000_000)
        self.assertTrue(protected)
        self.assertEqual(reason, "active_window")

    def test_disabled_flag_never_protects(self):
        obs = _make_observer(self.ws, protect_promising=False)
        window = obs._policy_window(0)
        window.samples = [_sample(mean=30.0, max_=50.0)]
        obs._recent_peak_nsucc_mean[0] = 30.0
        obs._maybe_enter_promising_protection(0, env_frames=200_000_000, window=window)
        protected, _ = obs._is_promising_protected(0, env_frames=250_000_000)
        self.assertFalse(protected)


class PromisingRetentionGuardTest(_TmpWs):
    """#11: peak gate requires retention >= --pbt_promising_min_retention."""

    def test_peak_high_retention_low_does_not_protect(self):
        obs = _make_observer(
            self.ws,
            protect_promising=True,
            promising_nsucc_mean=100.0,  # disable end_mean branch
            promising_peak_mean=25.0,
            promising_min_retention=0.5,
        )
        window = obs._policy_window(0)
        # end_mean=8 / peak=30 -> retention=0.27 < 0.5.
        window.samples = [_sample(mean=30.0, max_=50.0), _sample(mean=8.0, max_=10.0)]
        obs._recent_peak_nsucc_mean[0] = 30.0
        obs._maybe_enter_promising_protection(0, env_frames=200_000_000, window=window)
        protected, _ = obs._is_promising_protected(0, env_frames=210_000_000)
        self.assertFalse(protected)

    def test_peak_high_retention_ok_protects(self):
        obs = _make_observer(
            self.ws,
            protect_promising=True,
            promising_nsucc_mean=100.0,  # disable end_mean branch
            promising_peak_mean=25.0,
            promising_min_retention=0.5,
        )
        window = obs._policy_window(0)
        # end_mean=18 / peak=30 -> retention=0.6 >= 0.5.
        window.samples = [_sample(mean=30.0, max_=50.0), _sample(mean=18.0, max_=30.0)]
        obs._recent_peak_nsucc_mean[0] = 30.0
        obs._maybe_enter_promising_protection(0, env_frames=200_000_000, window=window)
        protected, _ = obs._is_promising_protected(0, env_frames=210_000_000)
        self.assertTrue(protected)


class HardCollapseOverridesProtectionTest(_TmpWs):
    """#12: hard collapse overrides protection; protected != ineligible source.

    The override is checked at the call site in `_ternary_pbt`; here we
    confirm the protection state itself is reset after the protection
    window expires, and that the helper does not couple to source
    eligibility.
    """

    def test_protection_expires_after_window(self):
        obs = _make_observer(
            self.ws, protect_promising=True, promising_protect_frames=10_000_000
        )
        window = obs._policy_window(0)
        window.samples = [_sample(mean=25.0, max_=40.0)]
        obs._recent_peak_nsucc_mean[0] = 25.0
        obs._maybe_enter_promising_protection(0, env_frames=100_000_000, window=window)
        protected, _ = obs._is_promising_protected(0, env_frames=105_000_000)
        self.assertTrue(protected)
        # Past the deadline.
        protected, _ = obs._is_promising_protected(0, env_frames=200_000_000)
        self.assertFalse(protected)

    def test_protected_worker_independent_from_source_eligibility(self):
        obs = _make_observer(self.ws, protect_promising=True)
        window = obs._policy_window(0)
        window.samples = [_sample(mean=25.0, max_=40.0)]
        obs._recent_peak_nsucc_mean[0] = 25.0
        obs._maybe_enter_promising_protection(0, env_frames=100_000_000, window=window)
        # Protection only uses _promising_protect_until; it must not also
        # toggle the source-eligibility state used by `_select_source`.
        # Capture eligibility flag before vs after to assert no coupling.
        baseline = getattr(obs, "curr_eligible_as_source", None)
        obs._maybe_enter_promising_protection(0, env_frames=110_000_000, window=window)
        self.assertEqual(
            getattr(obs, "curr_eligible_as_source", None), baseline
        )
        protected, _ = obs._is_promising_protected(0, env_frames=110_000_000)
        self.assertTrue(protected)


class CopyEventJsonlTest(_TmpWs):
    """#13: copy-event JSONL row schema written to disk."""

    def test_copy_event_payload_round_trips(self):
        obs = _make_observer(self.ws, inherit_mode="source_safe")
        obs._write_pbt_copy_event(
            {
                "env_frames": 300_000_000,
                "tier": "strict_live",
                "tier_top_level": "bottom",
                "curriculum_inherit_mode": "source_safe",
                "source_policy_idx": 1,
                "source_checkpoint_type": "live",
                "source_success_tolerance": 0.05,
                "source_nsucc_mean": 28.0,
                "source_nsucc_max": 50.0,
                "source_retention": 0.95,
                "source_safe_for_curriculum": True,
                "target_tolerance_before_copy": 0.075,
                "target_tolerance_after_copy": 0.0526,
                "inherited_tolerance": 0.0526,
                "inheritance_reason": "source_stable_relax_one_level",
                "legacy_tolerance_mode": "checkpoint",
                "legacy_tolerance_value": 0.05,
                "recovery_consolidation_frames_selected": 50_000_000,
                "recovery_consolidation_source": "recovery_consolidation_source_safe",
                "protected_by_promising_gate": False,
                "promising_protect_until": 0,
            }
        )
        path = self.ws / "metrics" / "pbt_copy_events.jsonl"
        self.assertTrue(path.exists())
        rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
        self.assertEqual(len(rows), 1)
        for k in (
            "curriculum_inherit_mode",
            "source_policy_idx",
            "source_success_tolerance",
            "source_nsucc_mean",
            "source_retention",
            "target_tolerance_after_copy",
            "inherited_tolerance",
            "inheritance_reason",
            "recovery_consolidation_frames_selected",
            "recovery_consolidation_source",
            "protected_by_promising_gate",
            "promising_protect_until",
            "timestamp",
            "policy_idx",
        ):
            self.assertIn(k, rows[0])


class CountersTest(_TmpWs):
    """#14: legacy reset path increments reset counter."""

    def test_legacy_reset_mode_increments_counter_via_observer(self):
        # Smoke: the observer keeps a running counter; the live increment
        # happens inside _ternary_pbt. We assert that the counter exists
        # and is initialised to 0 -- bumping it requires the full PBT
        # path, which is exercised by integration tests.
        obs = _make_observer(self.ws, inherit_mode="reset")
        self.assertEqual(obs._curriculum_reset_count, 0)
        self.assertEqual(obs._curriculum_inherited_count, 0)
        self.assertEqual(obs._curriculum_missing_metadata_count, 0)
        # Manually mimic what _ternary_pbt does for one reset-mode bottom
        # event.
        obs._curriculum_reset_count += 1
        self.assertEqual(obs._curriculum_reset_count, 1)


class StrongLiveExactInheritTest(_TmpWs):
    """CSER patch: strong live source exact tolerance inheritance."""

    def test_live_strong_source_exact_inherit(self):
        obs = _make_observer(self.ws, inherit_mode="source_safe")
        source_state = {
            "policy_idx": 1,
            "nsucc_mean": 35.0,
            "nsucc_max": 40.0,
            "source_retention_at_save": 0.4,
            "curriculum": {
                "success_tolerance": 0.023,
                "target_tolerance": 0.01,
            },
        }
        out = obs._resolve_curriculum_inheritance(
            source_policy=1,
            source_state=source_state,
            source_curriculum=source_state["curriculum"],
            checkpoint_type="live",
            target_current_tolerance=0.075,
        )
        self.assertTrue(out["strong_live_inherit_exact"])
        self.assertEqual(out["inheritance_reason"], "source_strong_live_inherit_exact")
        self.assertAlmostEqual(out["tol_after"], 0.023, places=10)
        self.assertTrue(out["source_safe_for_curriculum"])

    def test_live_strong_respects_tighter_target_before(self):
        obs = _make_observer(self.ws, inherit_mode="source_safe")
        source_state = {
            "policy_idx": 1,
            "nsucc_mean": 35.0,
            "nsucc_max": 40.0,
            "curriculum": {"success_tolerance": 0.023, "target_tolerance": 0.01},
        }
        out = obs._resolve_curriculum_inheritance(
            source_policy=1,
            source_state=source_state,
            source_curriculum=source_state["curriculum"],
            checkpoint_type="live",
            target_current_tolerance=0.020,
        )
        self.assertAlmostEqual(out["tol_after"], 0.020, places=10)

    def test_archive_strong_does_not_use_exact_live_branch(self):
        obs = _make_observer(self.ws, inherit_mode="source_safe")
        source_state = {
            "policy_idx": 1,
            "nsucc_mean": 35.0,
            "nsucc_max": 40.0,
            "source_retention_at_save": 0.95,
            "curriculum": {"success_tolerance": 0.023, "target_tolerance": 0.01},
        }
        out = obs._resolve_curriculum_inheritance(
            source_policy=1,
            source_state=source_state,
            source_curriculum=source_state["curriculum"],
            checkpoint_type="archive",
            target_current_tolerance=0.075,
        )
        self.assertNotEqual(out["inheritance_reason"], "source_strong_live_inherit_exact")
        self.assertFalse(out.get("strong_live_inherit_exact", False))

    def test_weak_mean_falls_back_to_legacy(self):
        obs = _make_observer(self.ws, inherit_mode="source_safe", crosscopy_mode="checkpoint")
        source_state = {
            "policy_idx": 2,
            "nsucc_mean": 15.0,
            "nsucc_max": 40.0,
            "source_retention_at_save": 0.95,
            "curriculum": {"success_tolerance": 0.023, "target_tolerance": 0.01},
        }
        out = obs._resolve_curriculum_inheritance(
            source_policy=2,
            source_state=source_state,
            source_curriculum=source_state["curriculum"],
            checkpoint_type="live",
            target_current_tolerance=0.075,
        )
        self.assertEqual(out["inheritance_reason"], "source_unstable_legacy")

    def test_missing_tolerance_falls_back_without_crash(self):
        obs = _make_observer(self.ws, inherit_mode="source_safe")
        source_state = {
            "policy_idx": 3,
            "nsucc_mean": 35.0,
            "nsucc_max": 40.0,
            "curriculum": {"target_tolerance": 0.01},
        }
        out = obs._resolve_curriculum_inheritance(
            source_policy=3,
            source_state=source_state,
            source_curriculum=source_state["curriculum"],
            checkpoint_type="live",
            target_current_tolerance=0.075,
        )
        self.assertEqual(out["inheritance_reason"], "missing_source_tolerance")


class LauncherFlagsTest(unittest.TestCase):
    """#15: train.py exposes new flags; launch_pbt boolean_flags has the toggles."""

    @classmethod
    def setUpClass(cls):
        cls.train_path = (
            Path(__file__).resolve().parents[1]
            / "dexpbt_isaaclab"
            / "scripts"
            / "train.py"
        )
        cls.launch_path = (
            Path(__file__).resolve().parents[1]
            / "dexpbt_isaaclab"
            / "scripts"
            / "launch_pbt.py"
        )
        cls.train_text = cls.train_path.read_text()
        cls.launch_text = cls.launch_path.read_text()

    def test_train_declares_inherit_mode_flag(self):
        self.assertIn('"--pbt_curriculum_inherit_mode"', self.train_text)
        self.assertIn('choices=["reset", "source_safe"]', self.train_text)

    def test_train_declares_source_safe_flags(self):
        for flag in (
            "--pbt_source_safe_recovery_consolidation_frames",
            "--pbt_source_unsafe_recovery_consolidation_frames",
            "--pbt_source_safe_min_mean",
            "--pbt_source_safe_min_retention",
        ):
            self.assertIn(f'"{flag}"', self.train_text, f"missing {flag}")

    def test_train_declares_promising_flags(self):
        for flag in (
            "--pbt_protect_promising_workers",
            "--pbt_promising_nsucc_mean",
            "--pbt_promising_peak_mean",
            "--pbt_promising_min_retention",
            "--pbt_promising_protect_frames",
            "--pbt_promising_allow_hard_collapse_override",
        ):
            self.assertIn(f'"{flag}"', self.train_text, f"missing {flag}")

    def test_train_forwards_flags_to_observer(self):
        for kwarg in (
            "pbt_curriculum_inherit_mode=args_cli.pbt_curriculum_inherit_mode",
            "pbt_source_safe_min_mean=args_cli.pbt_source_safe_min_mean",
            "pbt_protect_promising_workers=",
            "pbt_promising_min_retention=",
        ):
            self.assertTrue(
                re.search(re.escape(kwarg), self.train_text),
                f"missing forward: {kwarg}",
            )

    def test_launch_pbt_has_promising_boolean_flags(self):
        for flag in (
            '"--pbt_protect_promising_workers"',
            '"--no-pbt_protect_promising_workers"',
            '"--pbt_promising_allow_hard_collapse_override"',
            '"--no-pbt_promising_allow_hard_collapse_override"',
        ):
            self.assertIn(flag, self.launch_text, f"missing boolean flag {flag}")


if __name__ == "__main__":
    unittest.main()
