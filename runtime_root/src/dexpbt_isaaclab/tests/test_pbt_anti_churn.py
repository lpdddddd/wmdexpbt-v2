"""Patch 2 (rev 2, 2026-05-30): PBT anti-churn / N_adapt gates.

Reproduces and pins down the two independent burn-in gates that prevent
PBT thrash:

  * source-cooldown   -> `_is_eligible_as_source`
  * elimination-cooldown -> `_is_eligible_for_elimination`

Also exercises `_frames_since_restart`'s field-fallback chain
(frame - frames_at_restart -> curriculum.frame_since_restart -> 0)
and the `_restart_count` resume hook that previously stayed at 0 and
disabled the 50M `initial_delay` between PBT ticks.
"""

import tempfile
import unittest
from pathlib import Path

from dexpbt_isaaclab.pbt.pbt_observer import DexPBTPbtAlgoObserver


def _make_observer(
    workspace: Path,
    *,
    min_frames_as_source: int = 50_000_000,
    min_frames_between_restarts: int = 50_000_000,
    num_envs: int = 1,
    restart_count_resume: int | None = None,
    frames_at_restart_resume: int | None = None,
    last_mutation_frame_resume: int | None = None,
) -> DexPBTPbtAlgoObserver:
    # Patch A (rev 3, 2026-05-30): the new nsucc-threshold source gate is
    # OFF in this file so the Patch 2 anti-churn tests exercise *only* the
    # frame-cooldown gate they were originally designed for. Patch A's
    # nsucc-gate behaviour is covered by `test_pbt_source_nsucc_gate.py`.
    return DexPBTPbtAlgoObserver(
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
        num_envs=num_envs,
        pbt_min_frames_as_source_after_restart=min_frames_as_source,
        pbt_min_frames_between_restarts=min_frames_between_restarts,
        pbt_min_nsucc_mean_as_source=0.0,
        pbt_min_nsucc_max_as_source=0.0,
        restart_count_resume=restart_count_resume,
        frames_at_restart_resume=frames_at_restart_resume,
        last_mutation_frame_resume=last_mutation_frame_resume,
    )


def _state(
    *,
    frame_since_restart: int | None = None,
    nsucc_mean: float = 0.0,
    frame: int = 0,
    frames_at_restart: int | None = None,
) -> dict:
    s = {
        "policy_idx": 0,
        "frame": frame,
        "nsucc_mean": nsucc_mean,
        "curriculum": {},
    }
    if frame_since_restart is not None:
        s["curriculum"]["frame_since_restart"] = frame_since_restart
    if frames_at_restart is not None:
        s["frames_at_restart"] = frames_at_restart
    return s


class FramesSinceRestartFallbackTest(unittest.TestCase):
    """`_frames_since_restart` picks the most reliable source available."""

    def test_prefers_frame_minus_frames_at_restart_over_curriculum(self):
        # curriculum.frame_since_restart counts sim steps; PBT gates use env
        # frames, so the observer delta must win when both are present.
        s = _state(frame_since_restart=42_000, frame=100_000_000,
                   frames_at_restart=10_000_000)
        self.assertEqual(
            DexPBTPbtAlgoObserver._frames_since_restart(s), 90_000_000
        )

    def test_falls_back_to_curriculum_when_observer_fields_missing(self):
        s = _state(frame_since_restart=42_000_000)
        self.assertEqual(
            DexPBTPbtAlgoObserver._frames_since_restart(s), 42_000_000
        )

    def test_falls_back_to_frame_minus_frames_at_restart(self):
        s = _state(frame=120_000_000, frames_at_restart=50_000_000)
        self.assertEqual(
            DexPBTPbtAlgoObserver._frames_since_restart(s), 70_000_000
        )

    def test_returns_zero_when_no_info_available(self):
        s = _state()
        self.assertEqual(DexPBTPbtAlgoObserver._frames_since_restart(s), 0)

    def test_clamps_negative_fallback_to_zero(self):
        # If frame < frames_at_restart for whatever reason, we should not
        # report a negative fsr.
        s = _state(frame=10_000_000, frames_at_restart=20_000_000)
        self.assertEqual(DexPBTPbtAlgoObserver._frames_since_restart(s), 0)

    def test_handles_none_state(self):
        self.assertEqual(DexPBTPbtAlgoObserver._frames_since_restart(None), 0)

    def test_num_envs_8192_live_run_scenario_uses_env_frames_not_steps(self):
        """Regression for regrasping_cbwm_20260531_143643 unit bug.

        curriculum.frame_since_restart counts sim steps (~339k) while
        algo.frame is env_frames (~2.78B). PBT gates must use the latter.
        """
        s = _state(
            frame=2_780_037_120,
            frames_at_restart=0,
            frame_since_restart=339_376,
        )
        fsr = DexPBTPbtAlgoObserver._frames_since_restart(s)
        self.assertEqual(fsr, 2_780_037_120)
        self.assertAlmostEqual(
            s["frame"] / s["curriculum"]["frame_since_restart"], 8192, delta=1
        )


class EligibleAsSourceTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.workspace = Path(self._tmp.name)

    def test_num_envs_8192_passes_anti_churn_when_never_restarted(self):
        obs = _make_observer(self.workspace, num_envs=8192)
        s = _state(
            frame=2_780_037_120,
            frames_at_restart=0,
            frame_since_restart=339_376,
            nsucc_mean=0.0,
        )
        self.assertTrue(obs._is_eligible_for_elimination(s, env_frames=2_780_037_120))

    def test_blocked_when_fsr_below_min_frames(self):
        obs = _make_observer(self.workspace)
        s = _state(frame_since_restart=10_000_000, nsucc_mean=1.5)
        self.assertFalse(obs._is_eligible_as_source(s, env_frames=200_000_000))

    def test_blocked_when_nsucc_zero_even_if_fsr_long_enough(self):
        # Patch A (rev 3): the nsucc gate is now driven by explicit
        # `pbt_min_nsucc_mean_as_source` / `pbt_min_nsucc_max_as_source`
        # thresholds (see test_pbt_source_nsucc_gate.py). This test pins
        # down the Patch 2 frame-cooldown gate in isolation: configure
        # the gate explicitly so the nsucc=0 worker is rejected here for
        # documentation reasons.
        obs = DexPBTPbtAlgoObserver(
            policy_idx=0,
            num_policies=1,
            mutable_params={
                "enable_wm": False,
                "enable_counterfactual": False,
                "wm_strategy": "none",
            },
            workspace=str(self.workspace),
            train_dir=str(self.workspace),
            task_name="test-task",
            interval_steps=10_000_000,
            start_after=0,
            initial_delay=0,
            num_envs=1,
            pbt_min_frames_as_source_after_restart=50_000_000,
            pbt_min_frames_between_restarts=50_000_000,
            pbt_min_nsucc_mean_as_source=1.0,
            pbt_min_nsucc_max_as_source=5.0,
        )
        s = _state(frame_since_restart=200_000_000, nsucc_mean=0.0)
        self.assertFalse(obs._is_eligible_as_source(s, env_frames=300_000_000))

    def test_allowed_when_both_gates_pass(self):
        # Patch A (rev 3): the helper turns off the nsucc gate so this
        # test exercises only the frame-cooldown gate. nsucc_mean must be
        # > the gate-disabled threshold (0.0). Use 2.0 to be safe.
        obs = _make_observer(self.workspace)
        s = _state(frame_since_restart=60_000_000, nsucc_mean=2.0)
        self.assertTrue(obs._is_eligible_as_source(s, env_frames=300_000_000))

    def test_min_frames_as_source_is_configurable(self):
        # Patch A (rev 3): nsucc gate disabled in helper; only verify
        # the frame-cooldown threshold is respected.
        obs = _make_observer(self.workspace, min_frames_as_source=10_000_000)
        s = _state(frame_since_restart=15_000_000, nsucc_mean=2.0)
        self.assertTrue(obs._is_eligible_as_source(s, env_frames=300_000_000))

    def test_none_state_treated_as_ineligible(self):
        obs = _make_observer(self.workspace)
        self.assertFalse(
            obs._is_eligible_as_source(None, env_frames=300_000_000)
        )


class EligibleForEliminationTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.workspace = Path(self._tmp.name)

    def test_blocked_within_burnin_window(self):
        obs = _make_observer(self.workspace)
        s = _state(frame_since_restart=10_000_000, nsucc_mean=0.0)
        self.assertFalse(
            obs._is_eligible_for_elimination(s, env_frames=200_000_000)
        )

    def test_allowed_after_burnin_even_if_nsucc_zero(self):
        # Elimination cooldown is INDEPENDENT of nsucc -- a bad worker
        # must still be killable once it has run long enough that there is
        # genuinely new evidence.
        obs = _make_observer(self.workspace)
        s = _state(frame_since_restart=200_000_000, nsucc_mean=0.0)
        self.assertTrue(
            obs._is_eligible_for_elimination(s, env_frames=300_000_000)
        )

    def test_independent_of_source_cooldown(self):
        """If only source gate is set, elimination can still trigger."""
        obs = _make_observer(
            self.workspace,
            min_frames_as_source=100_000_000,
            min_frames_between_restarts=10_000_000,
        )
        s = _state(frame_since_restart=20_000_000, nsucc_mean=0.0)
        # Source: blocked (20M < 100M); Elimination: allowed (20M >= 10M).
        self.assertFalse(obs._is_eligible_as_source(s, env_frames=300_000_000))
        self.assertTrue(
            obs._is_eligible_for_elimination(s, env_frames=300_000_000)
        )

    def test_none_state_treated_as_eligible(self):
        # No meta yet means there's nothing to protect; the worker has
        # never restarted from PBT and may be replaced normally.
        obs = _make_observer(self.workspace)
        self.assertTrue(
            obs._is_eligible_for_elimination(None, env_frames=300_000_000)
        )


class RestartResumeHookTest(unittest.TestCase):
    """Patch 2: __init__ must rebuild anti-churn state from resume args."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.workspace = Path(self._tmp.name)

    def test_defaults_when_no_resume(self):
        obs = _make_observer(self.workspace)
        self.assertEqual(obs._restart_count, 0)
        self.assertEqual(obs._frames_at_restart, 0)
        self.assertEqual(obs.last_mutation_frame, 0)

    def test_resume_args_take_effect(self):
        obs = _make_observer(
            self.workspace,
            restart_count_resume=7,
            frames_at_restart_resume=300_000_000,
            last_mutation_frame_resume=300_000_000,
        )
        self.assertEqual(obs._restart_count, 7)
        self.assertEqual(obs._frames_at_restart, 300_000_000)
        self.assertEqual(obs.last_mutation_frame, 300_000_000)

    def test_resume_args_isolated_from_anti_churn_config(self):
        # Different env_frames; should still report correct fsr from
        # observer's own state when reading own snapshot.
        obs = _make_observer(
            self.workspace,
            restart_count_resume=2,
            frames_at_restart_resume=200_000_000,
            last_mutation_frame_resume=200_000_000,
            min_frames_as_source=50_000_000,
            min_frames_between_restarts=50_000_000,
        )
        # If we synthesize a state at frame=260M with frames_at_restart=200M,
        # the helper should derive fsr=60M and clear both gates.
        s = _state(frame=260_000_000, frames_at_restart=200_000_000,
                   nsucc_mean=1.0)
        self.assertEqual(
            DexPBTPbtAlgoObserver._frames_since_restart(s), 60_000_000
        )
        self.assertTrue(obs._is_eligible_as_source(s, env_frames=260_000_000))
        self.assertTrue(
            obs._is_eligible_for_elimination(s, env_frames=260_000_000)
        )


class TopPolicyEligibilityFilterTest(unittest.TestCase):
    """End-to-end shape: applying the gate to a sorted top-N list."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.workspace = Path(self._tmp.name)

    def test_collapse_scenario_filters_out_freshly_restarted_workers(self):
        # Patch 2 (frame-cooldown) end-to-end: three "top" workers, only
        # one has frames_since_restart past the burn-in. The two freshly
        # restarted ones must be filtered out by the frame-cooldown gate
        # alone (independent of the Patch A nsucc gate, which is covered
        # by test_pbt_source_nsucc_gate.py).
        obs = _make_observer(self.workspace)
        states = {
            0: _state(frame_since_restart=10_000_000, nsucc_mean=2.5),
            1: _state(frame_since_restart=10_000_000, nsucc_mean=2.5),
            2: _state(frame_since_restart=80_000_000, nsucc_mean=2.5),
        }
        eligible = [
            idx
            for idx, s in states.items()
            if obs._is_eligible_as_source(s, env_frames=300_000_000)
        ]
        self.assertEqual(eligible, [2])

    def test_fallback_when_no_one_eligible(self):
        # If no top worker passes the gate, the production code falls back
        # to the original top-N list to avoid stalling PBT entirely. This
        # test ensures the helpers don't accidentally promote ineligible
        # workers; the fallback policy itself is exercised in pbt logic.
        obs = _make_observer(self.workspace)
        states = {
            0: _state(frame_since_restart=10_000_000, nsucc_mean=0.0),
            1: _state(frame_since_restart=10_000_000, nsucc_mean=0.0),
        }
        eligible = [
            idx
            for idx, s in states.items()
            if obs._is_eligible_as_source(s, env_frames=300_000_000)
        ]
        self.assertEqual(eligible, [])


if __name__ == "__main__":
    unittest.main()
