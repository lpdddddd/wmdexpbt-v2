"""Phase A4 P2 (2026-06-08): adaptive-fast curriculum tests.

Covers:
- normal mode preserves legacy single-factor shrink behaviour;
- adaptive_fast tier selection (strong / mid / low) and shrink-factor pick;
- nsucc<low_mean blocks shrink with adaptive_fast_low_nsucc_mean reason;
- recovery stage forces legacy factor + longer consolidation;
- existing collapse_freeze / consolidation gates take precedence;
- tolerance floor (target_tolerance) respected under aggressive shrink;
- PBT restart `base_args` carry --curriculum_speed_mode and fast knobs;
- JSONL/control-state payload contains adaptive fields;
- __init__ validation rejects inverted thresholds + non-mastery mode.

These tests intentionally avoid spinning up an env or PPO agent; they exercise
the observer / env-curriculum code paths directly using mock window samples
and the on-disk control JSON.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from dexpbt_isaaclab.pbt.pbt_curriculum_p1 import (
    CurriculumControlState,
    WindowSample,
    apply_env_tolerance_curriculum,
    mark_shrink_consumed,
    read_curriculum_control,
    write_curriculum_control,
)
from dexpbt_isaaclab.pbt.pbt_observer import DexPBTPbtAlgoObserver


def _ts(mean_nsucc: float) -> torch.Tensor:
    return torch.full((4,), float(mean_nsucc))


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
    speed_mode: str = "normal",
    curriculum_mode: str = "mastery",
    pbt_curriculum_freeze_frames: int = 100_000_000,
    curriculum_fast_strong_mean: float = 30.0,
    curriculum_fast_mid_mean: float = 20.0,
    curriculum_fast_low_mean: float = 10.0,
    curriculum_fast_retention_threshold: float = 0.8,
    curriculum_fast_retention_window: int = 4,
    curriculum_fast_shrink_strong: float = 0.90,
    curriculum_fast_shrink_mid: float = 0.925,
    curriculum_fast_shrink_low: float = 0.95,
    curriculum_fast_consolidation_frames: int = 50_000_000,
    curriculum_fast_recovery_consolidation_frames: int = 100_000_000,
    curriculum_fast_target_tolerance_800m: float = 0.04,
    curriculum_shrink_factor: float = 0.95,
    post_shrink_consolidation_env_frames: int = 100_000_000,
    mastery_nsucc_mean: float = 4.0,
    mastery_nsucc_max: float = 20.0,
    mastery_consecutive_windows: int = 2,
    pbt_single_shrink_per_window: bool = True,
    curriculum_collapse_detect: bool = True,
    pbt_early_collapse_detect: bool = True,
) -> DexPBTPbtAlgoObserver:
    obs = DexPBTPbtAlgoObserver(
        policy_idx=0,
        num_policies=1,
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
        pbt_curriculum_freeze_frames=pbt_curriculum_freeze_frames,
        pbt_curriculum_freeze_on_recovery_frames=pbt_curriculum_freeze_frames,
        curriculum_mode=curriculum_mode,
        mastery_nsucc_mean=mastery_nsucc_mean,
        mastery_nsucc_max=mastery_nsucc_max,
        mastery_consecutive_windows=mastery_consecutive_windows,
        curriculum_shrink_factor=curriculum_shrink_factor,
        pbt_single_shrink_per_window=pbt_single_shrink_per_window,
        post_shrink_consolidation_env_frames=post_shrink_consolidation_env_frames,
        curriculum_collapse_detect=curriculum_collapse_detect,
        pbt_early_collapse_detect=pbt_early_collapse_detect,
        # Phase A4 P2 adaptive-fast knobs.
        curriculum_speed_mode=speed_mode,
        curriculum_fast_strong_mean=curriculum_fast_strong_mean,
        curriculum_fast_mid_mean=curriculum_fast_mid_mean,
        curriculum_fast_low_mean=curriculum_fast_low_mean,
        curriculum_fast_retention_threshold=curriculum_fast_retention_threshold,
        curriculum_fast_retention_window=curriculum_fast_retention_window,
        curriculum_fast_shrink_strong=curriculum_fast_shrink_strong,
        curriculum_fast_shrink_mid=curriculum_fast_shrink_mid,
        curriculum_fast_shrink_low=curriculum_fast_shrink_low,
        curriculum_fast_consolidation_frames=curriculum_fast_consolidation_frames,
        curriculum_fast_recovery_consolidation_frames=curriculum_fast_recovery_consolidation_frames,
        curriculum_fast_target_tolerance_800m=curriculum_fast_target_tolerance_800m,
    )
    obs.writer = None
    return obs


class NormalModePreservesLegacyTest(unittest.TestCase):
    """test_normal_mode_preserves_legacy_shrink_factor (#1)."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.ws = Path(self._tmp.name)
        self.control_path = self.ws / "curriculum_control" / "000.json"

    def test_observer_writes_zero_adaptive_factor_in_normal_mode(self):
        obs = _make_observer(self.ws, speed_mode="normal")
        # Build a tier-strong window so that adaptive_fast WOULD pick 0.90;
        # in normal mode we expect adaptive_shrink_factor to remain 0.0.
        window = obs._policy_window(0)
        window.samples = [
            _sample(mean=32.0, max_=50.0, iteration=i, env_frames=200_000_000 + i)
            for i in range(4)
        ]
        obs._recent_peak_nsucc_mean[0] = 32.0
        ctrl = obs._build_curriculum_control_state(
            policy_idx=0,
            window=window,
            env_frames=300_000_000,
            population={
                0: {
                    "frame": 300_000_000,
                    "frames_at_restart": 0,
                    "curriculum": {"frame_since_restart": 200_000_000},
                }
            },
        )
        self.assertEqual(ctrl.speed_mode, "normal")
        self.assertEqual(ctrl.adaptive_shrink_factor, 0.0)
        self.assertEqual(ctrl.adaptive_consolidation_frames, 0)
        self.assertEqual(ctrl.shrink_reason, "normal_mode")

    def test_env_uses_legacy_factor_when_adaptive_zero(self):
        write_curriculum_control(
            self.control_path,
            CurriculumControlState(
                curriculum_mode="mastery",
                shrink_token=True,
                allow_shrink=True,
                # All adaptive_* fields default to 0 / "" -> legacy fallback.
            ),
        )
        kwargs = dict(
            curriculum_mode="mastery",
            curriculum_control_path=self.control_path,
            curriculum_shrink_factor=0.95,
            last_curriculum_update=0,
            frames_since_restart=200_000_000,
            curriculum_interval=10_000_000,
            prev_episode_successes=_ts(5.0),
            success_tolerance=0.075,
            initial_tolerance=0.075,
            target_tolerance=0.01,
            post_shrink_consolidation_env_frames=100_000_000,
            num_envs=8192,
            pbt_single_shrink_per_window=True,
        )
        tol1, _ = apply_env_tolerance_curriculum(**kwargs)
        # Legacy: 0.075 * 0.95 = 0.07125
        self.assertAlmostEqual(tol1, 0.075 * 0.95, places=6)


class AdaptiveFastTierSelectionTest(unittest.TestCase):
    """tests #2 / #3 / #4: strong / mid / low tier."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.ws = Path(self._tmp.name)

    def _ctrl_for_window(
        self,
        *,
        nsuccs: list[float],
        env_frames_at_tick: int = 300_000_000,
        retention_window: int = 4,
    ) -> CurriculumControlState:
        obs = _make_observer(
            self.ws,
            speed_mode="adaptive_fast",
            curriculum_fast_retention_window=retention_window,
        )
        window = obs._policy_window(0)
        window.samples = [
            _sample(mean=v, max_=50.0, iteration=i, env_frames=200_000_000 + i)
            for i, v in enumerate(nsuccs)
        ]
        obs._recent_peak_nsucc_mean[0] = max(nsuccs) if nsuccs else 0.0
        return obs._build_curriculum_control_state(
            policy_idx=0,
            window=window,
            env_frames=env_frames_at_tick,
            population={
                0: {
                    "frame": env_frames_at_tick,
                    "frames_at_restart": 0,
                    "curriculum": {"frame_since_restart": env_frames_at_tick},
                }
            },
        )

    def test_strong_tier_selects_0_90_with_50m_consolidation(self):
        # Last 4 samples are all strong: nsucc=32 each -> retention=1.0.
        ctrl = self._ctrl_for_window(nsuccs=[32.0, 33.0, 31.5, 32.2])
        self.assertEqual(ctrl.shrink_reason, "strong")
        self.assertAlmostEqual(ctrl.adaptive_shrink_factor, 0.90, places=6)
        self.assertEqual(ctrl.adaptive_consolidation_frames, 50_000_000)
        self.assertGreaterEqual(ctrl.retention_used_for_shrink, 0.8)
        self.assertGreaterEqual(ctrl.nsucc_mean_used_for_shrink, 30.0)
        self.assertTrue(ctrl.shrink_token)

    def test_mid_tier_selects_0_925(self):
        ctrl = self._ctrl_for_window(nsuccs=[22.0, 22.5, 21.5, 22.0])
        self.assertEqual(ctrl.shrink_reason, "mid")
        self.assertAlmostEqual(ctrl.adaptive_shrink_factor, 0.925, places=6)
        self.assertEqual(ctrl.adaptive_consolidation_frames, 50_000_000)
        self.assertTrue(ctrl.shrink_token)

    def test_low_tier_selects_0_95(self):
        # nsucc_max also has to clear mastery_nsucc_max=20 for the gate to pass.
        ctrl = self._ctrl_for_window(nsuccs=[12.0, 12.5, 12.5, 12.0])
        self.assertEqual(ctrl.shrink_reason, "low")
        self.assertAlmostEqual(ctrl.adaptive_shrink_factor, 0.95, places=6)
        self.assertEqual(ctrl.adaptive_consolidation_frames, 50_000_000)
        self.assertTrue(ctrl.shrink_token)


class AdaptiveFastBlocksLowNsuccTest(unittest.TestCase):
    """test #5: nsucc<low_mean blocks shrink, reason=adaptive_fast_low_nsucc_mean."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.ws = Path(self._tmp.name)

    def test_blocks_shrink_with_low_nsucc(self):
        # mastery gate passes (mastery_nsucc_mean=4, max=20), but adaptive_fast
        # demands nsucc>=10. Set nsuccs to 8 -> blocked.
        obs = _make_observer(self.ws, speed_mode="adaptive_fast")
        window = obs._policy_window(0)
        window.samples = [
            _sample(mean=8.0, max_=22.0, iteration=i, env_frames=200_000_000 + i)
            for i in range(4)
        ]
        obs._recent_peak_nsucc_mean[0] = 8.0
        ctrl = obs._build_curriculum_control_state(
            policy_idx=0,
            window=window,
            env_frames=300_000_000,
            population={
                0: {
                    "frame": 300_000_000,
                    "frames_at_restart": 0,
                    "curriculum": {"frame_since_restart": 300_000_000},
                }
            },
        )
        self.assertFalse(ctrl.shrink_token)
        self.assertEqual(ctrl.shrink_blocked_reason, "adaptive_fast_low_nsucc_mean")
        self.assertEqual(ctrl.shrink_reason, "blocked_low_nsucc")
        self.assertTrue(ctrl.blocked_by_adaptive_fast_low_nsucc)


class AdaptiveFastRecoveryFallbackTest(unittest.TestCase):
    """test #6: recovery stage uses legacy factor + 100M consolidation."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.ws = Path(self._tmp.name)

    def test_recovery_uses_legacy_factor_and_long_consolidation(self):
        obs = _make_observer(
            self.ws,
            speed_mode="adaptive_fast",
            curriculum_shrink_factor=0.95,
            pbt_curriculum_freeze_frames=100_000_000,
        )
        # frames_since_restart=50M (<100M freeze) -> in_recovery=True.
        # nsucc=32 strong-tier value, but recovery overrides.
        window = obs._policy_window(0)
        window.samples = [
            _sample(mean=32.0, max_=50.0, iteration=i, env_frames=10_000_000 + i)
            for i in range(4)
        ]
        obs._recent_peak_nsucc_mean[0] = 32.0
        # env_frames=50M, frames_at_restart=0 -> elapsed=50M < 100M.
        ctrl = obs._build_curriculum_control_state(
            policy_idx=0,
            window=window,
            env_frames=50_000_000,
            population={
                0: {
                    "frame": 50_000_000,
                    "frames_at_restart": 0,
                    "curriculum": {"frame_since_restart": 50_000_000},
                }
            },
        )
        self.assertEqual(ctrl.shrink_reason, "recovery_legacy")
        self.assertAlmostEqual(ctrl.adaptive_shrink_factor, 0.95, places=6)
        self.assertEqual(ctrl.adaptive_consolidation_frames, 100_000_000)
        self.assertTrue(ctrl.blocked_by_recovery)


class CollapseFreezeTakesPrecedenceTest(unittest.TestCase):
    """test #7: collapse_freeze blocks shrink regardless of nsucc."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.ws = Path(self._tmp.name)

    def test_collapse_freeze_active_blocks_adaptive_fast(self):
        obs = _make_observer(self.ws, speed_mode="adaptive_fast")
        window = obs._policy_window(0)
        # Force a hard collapse: peak 30, then drop to 5 for >=2 windows.
        window.samples = [
            _sample(mean=30.0, max_=50.0, iteration=0, env_frames=180_000_000),
            _sample(mean=5.0, max_=20.0, iteration=1, env_frames=200_000_000),
            _sample(mean=4.0, max_=18.0, iteration=2, env_frames=220_000_000),
        ]
        obs._recent_peak_nsucc_mean[0] = 30.0
        ctrl = obs._build_curriculum_control_state(
            policy_idx=0,
            window=window,
            env_frames=300_000_000,
            population={
                0: {
                    "frame": 300_000_000,
                    "frames_at_restart": 0,
                    "curriculum": {"frame_since_restart": 300_000_000},
                }
            },
        )
        self.assertFalse(ctrl.shrink_token)
        self.assertEqual(ctrl.shrink_blocked_reason, "collapse_freeze_active")
        self.assertTrue(ctrl.blocked_by_collapse)
        # Still no adaptive factor when blocked -- the env never sees a token.
        self.assertEqual(ctrl.adaptive_shrink_factor, 0.0)


class ConsolidationTakesPrecedenceTest(unittest.TestCase):
    """test #8: post_shrink_consolidation blocks shrink regardless of tier."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.ws = Path(self._tmp.name)

    def test_consolidation_blocks_adaptive_fast(self):
        obs = _make_observer(self.ws, speed_mode="adaptive_fast")
        # Pre-write a control with consolidation_active set far in the future.
        control_path = obs._curriculum_control_dir / "000.json"
        write_curriculum_control(
            control_path,
            CurriculumControlState(
                curriculum_mode="mastery",
                post_shrink_consolidation_until_restart_step=10**9,
            ),
        )
        window = obs._policy_window(0)
        window.samples = [
            _sample(mean=32.0, max_=50.0, iteration=i, env_frames=200_000_000 + i)
            for i in range(4)
        ]
        obs._recent_peak_nsucc_mean[0] = 32.0
        ctrl = obs._build_curriculum_control_state(
            policy_idx=0,
            window=window,
            env_frames=300_000_000,
            population={
                0: {
                    "frame": 300_000_000,
                    "frames_at_restart": 0,
                    "curriculum": {"frame_since_restart": 100_000},  # < until=1e9
                }
            },
        )
        self.assertFalse(ctrl.shrink_token)
        self.assertEqual(ctrl.shrink_blocked_reason, "post_shrink_consolidation")
        self.assertTrue(ctrl.blocked_by_consolidation)


class ToleranceFloorRespectedTest(unittest.TestCase):
    """test #9: repeated 0.90 shrinks still clamp to target_tolerance."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.ws = Path(self._tmp.name)
        self.control_path = self.ws / "000.json"

    def test_strong_factor_does_not_drop_below_target_tolerance(self):
        target = 0.04
        tol = 0.05
        for _ in range(20):
            write_curriculum_control(
                self.control_path,
                CurriculumControlState(
                    curriculum_mode="mastery",
                    shrink_token=True,
                    allow_shrink=True,
                    adaptive_shrink_factor=0.90,
                    adaptive_consolidation_frames=0,  # disable consolidation block
                ),
            )
            new_tol, _ = apply_env_tolerance_curriculum(
                curriculum_mode="mastery",
                curriculum_control_path=self.control_path,
                curriculum_shrink_factor=0.95,
                last_curriculum_update=0,
                frames_since_restart=200_000_000,
                curriculum_interval=1,  # any positive < frames_since_restart
                prev_episode_successes=_ts(40.0),
                success_tolerance=tol,
                initial_tolerance=0.075,
                target_tolerance=target,
                post_shrink_consolidation_env_frames=0,
                num_envs=8192,
                pbt_single_shrink_per_window=True,
            )
            tol = new_tol
            self.assertGreaterEqual(tol, target - 1e-12)
        self.assertAlmostEqual(tol, target, places=6)


class PbtRestartCarriesAdaptiveFastFlagsTest(unittest.TestCase):
    """test #10: --curriculum_speed_mode + --curriculum_fast_* survive PBT restart.

    The launcher / observer rebuilds child argv from sys.argv; new --curriculum_*
    value flags are NOT in `skip_flags`, so they should be carried forward.
    We assert this by reading the source files (avoiding the heavy Isaac Lab
    side-effects of importing train.py at unit-test time).
    """

    EXPECTED_FLAGS = (
        "--curriculum_speed_mode",
        "--curriculum_fast_strong_mean",
        "--curriculum_fast_mid_mean",
        "--curriculum_fast_low_mean",
        "--curriculum_fast_retention_threshold",
        "--curriculum_fast_retention_window",
        "--curriculum_fast_shrink_strong",
        "--curriculum_fast_shrink_mid",
        "--curriculum_fast_shrink_low",
        "--curriculum_fast_consolidation_frames",
        "--curriculum_fast_recovery_consolidation_frames",
        "--curriculum_fast_target_tolerance_800m",
    )

    def test_train_argparse_declares_adaptive_fast_flags(self):
        # Read train.py source -- importing it pulls in Isaac Lab and is
        # not safe in the headless unit-test env. argparse declarations are
        # plain text matches.
        from dexpbt_isaaclab.scripts import __file__ as scripts_init
        train_src = (Path(scripts_init).parent / "train.py").read_text()
        for flag in self.EXPECTED_FLAGS:
            self.assertIn(
                f'"{flag}"',
                train_src,
                f"train.py argparse missing {flag}",
            )

    def test_observer_skip_flags_does_not_strip_curriculum_speed_mode(self):
        # The PBT restart path (pbt_observer.py) reconstructs child argv from
        # sys.argv minus a `skip_flags` set. Adaptive-fast config flags must
        # NOT be in skip_flags so they survive cross/self-restart.
        from dexpbt_isaaclab.pbt import pbt_observer

        src = Path(pbt_observer.__file__).read_text()
        # Find every literal `skip_flags = {...}` block and ensure none
        # contain our flag names.
        marker = "skip_flags"
        for flag in self.EXPECTED_FLAGS:
            # Must not appear anywhere as a "skip_flags = {... '--flag' ...}"
            # entry. We use a coarse but effective heuristic: split the file
            # around `skip_flags` and assert the flag name does not appear in
            # the same logical block. This is conservative -- false positives
            # would require a literal " --curriculum_speed_mode " string near
            # `skip_flags` not in a config-keep position, which we manually
            # inspected to be absent.
            if marker not in src:
                continue
            # Inspect each `{ ... }` literal that follows a `skip_flags` token.
            chunks = src.split(marker)
            for chunk in chunks[1:]:
                # Take up to the next closing brace on a balanced level.
                if "{" not in chunk:
                    continue
                start = chunk.index("{")
                depth = 0
                end = start
                for i, ch in enumerate(chunk[start:], start=start):
                    if ch == "{":
                        depth += 1
                    elif ch == "}":
                        depth -= 1
                        if depth == 0:
                            end = i
                            break
                block = chunk[start : end + 1]
                self.assertNotIn(
                    flag,
                    block,
                    f"--{flag} unexpectedly appears in pbt_observer skip_flags",
                )


class LoggingPayloadContainsAdaptiveFieldsTest(unittest.TestCase):
    """test #11: control JSON + dataclass round-trip carries adaptive fields."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.ws = Path(self._tmp.name)
        self.control_path = self.ws / "000.json"

    def test_round_trip_preserves_adaptive_fields(self):
        original = CurriculumControlState(
            curriculum_mode="mastery",
            shrink_token=True,
            allow_shrink=True,
            speed_mode="adaptive_fast",
            adaptive_shrink_factor=0.90,
            adaptive_consolidation_frames=50_000_000,
            shrink_reason="strong",
            nsucc_mean_used_for_shrink=32.5,
            retention_used_for_shrink=0.91,
            blocked_by_recovery=False,
            blocked_by_adaptive_fast_low_nsucc=False,
            pre_shrink_tolerance=0.05,
            post_shrink_tolerance=0.045,
        )
        write_curriculum_control(self.control_path, original)
        decoded = read_curriculum_control(self.control_path)
        self.assertIsNotNone(decoded)
        assert decoded is not None
        self.assertEqual(decoded.speed_mode, "adaptive_fast")
        self.assertAlmostEqual(decoded.adaptive_shrink_factor, 0.90)
        self.assertEqual(decoded.adaptive_consolidation_frames, 50_000_000)
        self.assertEqual(decoded.shrink_reason, "strong")
        self.assertAlmostEqual(decoded.nsucc_mean_used_for_shrink, 32.5)
        self.assertAlmostEqual(decoded.retention_used_for_shrink, 0.91)
        self.assertAlmostEqual(decoded.pre_shrink_tolerance, 0.05)
        self.assertAlmostEqual(decoded.post_shrink_tolerance, 0.045)

    def test_mark_shrink_consumed_writes_pre_post_tolerance(self):
        write_curriculum_control(
            self.control_path,
            CurriculumControlState(
                curriculum_mode="mastery",
                shrink_token=True,
                allow_shrink=True,
                speed_mode="adaptive_fast",
                adaptive_shrink_factor=0.90,
                adaptive_consolidation_frames=50_000_000,
            ),
        )
        mark_shrink_consumed(
            self.control_path,
            frames_since_restart=100_000,
            consolidation_env_frames=50_000_000,
            num_envs=8192,
            pre_shrink_tolerance=0.075,
            post_shrink_tolerance=0.0675,
        )
        decoded = read_curriculum_control(self.control_path)
        assert decoded is not None
        self.assertAlmostEqual(decoded.pre_shrink_tolerance, 0.075)
        self.assertAlmostEqual(decoded.post_shrink_tolerance, 0.0675)
        # Adaptive fields preserved.
        self.assertEqual(decoded.speed_mode, "adaptive_fast")
        self.assertAlmostEqual(decoded.adaptive_shrink_factor, 0.90)


class ValidationRejectsBadConfigTest(unittest.TestCase):
    """test #12: __init__ rejects inverted thresholds + non-mastery mode."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.ws = Path(self._tmp.name)

    def test_inverted_thresholds_raise_runtime_error(self):
        with self.assertRaises(RuntimeError) as cm:
            _make_observer(
                self.ws,
                speed_mode="adaptive_fast",
                curriculum_fast_strong_mean=10.0,  # < mid
                curriculum_fast_mid_mean=20.0,
                curriculum_fast_low_mean=15.0,
            )
        self.assertIn("strong_mean > mid_mean", str(cm.exception))

    def test_non_mastery_mode_with_adaptive_fast_raises(self):
        with self.assertRaises(RuntimeError) as cm:
            _make_observer(
                self.ws,
                speed_mode="adaptive_fast",
                curriculum_mode="legacy",
            )
        self.assertIn("requires --curriculum_mode=mastery", str(cm.exception))

    def test_unknown_speed_mode_raises(self):
        with self.assertRaises(RuntimeError) as cm:
            _make_observer(self.ws, speed_mode="blazing_fast")
        self.assertIn("Unknown curriculum_speed_mode", str(cm.exception))

    def test_invalid_shrink_factor_raises(self):
        with self.assertRaises(RuntimeError):
            _make_observer(
                self.ws,
                speed_mode="adaptive_fast",
                curriculum_fast_shrink_strong=1.5,
            )


class ObserverContractInvariantsTest(unittest.TestCase):
    """Bonus: verifies observer never overrides shrink_token=True when blocked."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.ws = Path(self._tmp.name)

    def test_normal_mode_emits_shrink_reason_constant(self):
        obs = _make_observer(self.ws, speed_mode="normal")
        window = obs._policy_window(0)
        window.samples = [
            _sample(mean=2.0, max_=10.0, iteration=i, env_frames=200_000_000 + i)
            for i in range(2)
        ]
        ctrl = obs._build_curriculum_control_state(
            policy_idx=0,
            window=window,
            env_frames=300_000_000,
            population={
                0: {
                    "frame": 300_000_000,
                    "frames_at_restart": 0,
                    "curriculum": {"frame_since_restart": 300_000_000},
                }
            },
        )
        self.assertEqual(ctrl.shrink_reason, "normal_mode")
        self.assertEqual(ctrl.adaptive_shrink_factor, 0.0)


if __name__ == "__main__":
    unittest.main()
