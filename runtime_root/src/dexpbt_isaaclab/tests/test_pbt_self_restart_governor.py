"""Patch F (2026-05-31): mid-tier self-restart governor.

Pins down the four-stage gate added to `_eligible_for_self_restart` and
the new accounting that `_ternary_pbt` performs around self/cross/skip
decisions. Smoke run 20260530_200619 produced 100% self-restart traffic
because the legacy DexPBT mid branch unconditionally relaunched the
mid worker on every PBT tick; Patch F flips this default OFF.

What we cover here:

1. `_eligible_for_self_restart`
     - disabled-by-default master switch
     - lifetime count cap (`max_self_restarts_per_policy`)
     - cooldown between consecutive self-restarts
     - legacy escape hatch `allow_mid_self_mutation`

2. Resume hooks
     - `self_restart_count_resume` / `last_self_restart_frame_resume`
       / `cross_restart_count_resume` are honoured by `__init__`

3. End-to-end `_ternary_pbt` shape (with all heavy I/O mocked)
     - default config: mid tier returns without relaunch and bumps the
       blocked counter
     - legacy config: mid tier triggers exactly once then is blocked
       by the lifetime cap on the second tick
     - bottom tier with eligible top: increments cross counter, sets
       `_last_restart_type=2`
"""

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from dexpbt_isaaclab.pbt.pbt_observer import DexPBTPbtAlgoObserver


def _make_observer(
    workspace: Path,
    *,
    disable_self_restart: bool = True,
    allow_mid_self_mutation: bool = False,
    max_self_restarts_per_policy: int = 1,
    min_frames_between_self_restarts: int = 100_000_000,
    self_restart_count_resume: int | None = None,
    last_self_restart_frame_resume: int | None = None,
    cross_restart_count_resume: int | None = None,
    num_policies: int = 3,
    policy_idx: int = 0,
) -> DexPBTPbtAlgoObserver:
    obs = DexPBTPbtAlgoObserver(
        policy_idx=policy_idx,
        num_policies=num_policies,
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
        pbt_min_nsucc_mean_as_source=0.0,
        pbt_min_nsucc_max_as_source=0.0,
        pbt_disable_self_restart=disable_self_restart,
        pbt_allow_mid_self_mutation=allow_mid_self_mutation,
        pbt_max_self_restarts_per_policy=max_self_restarts_per_policy,
        pbt_min_frames_between_self_restarts=min_frames_between_self_restarts,
        self_restart_count_resume=self_restart_count_resume,
        last_self_restart_frame_resume=last_self_restart_frame_resume,
        cross_restart_count_resume=cross_restart_count_resume,
    )
    obs.writer = None
    return obs


class EligibleForSelfRestartTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.workspace = Path(self._tmp.name)

    def test_disabled_by_default(self):
        obs = _make_observer(self.workspace)
        ok, reason = obs._eligible_for_self_restart(env_frames=10**9)
        self.assertFalse(ok)
        self.assertEqual(reason, "disabled_by_default")

    def test_legacy_escape_hatch_re_enables_path(self):
        obs = _make_observer(
            self.workspace,
            disable_self_restart=True,  # master switch still True
            allow_mid_self_mutation=True,  # escape hatch flips behaviour
            max_self_restarts_per_policy=3,
            min_frames_between_self_restarts=10_000_000,
        )
        ok, reason = obs._eligible_for_self_restart(env_frames=300_000_000)
        self.assertTrue(ok, reason)
        self.assertEqual(reason, "ok")

    def test_lifetime_cap_blocks_after_max_count(self):
        obs = _make_observer(
            self.workspace,
            disable_self_restart=False,
            max_self_restarts_per_policy=2,
            min_frames_between_self_restarts=0,
            self_restart_count_resume=2,
        )
        ok, reason = obs._eligible_for_self_restart(env_frames=500_000_000)
        self.assertFalse(ok)
        self.assertIn("self_restart_count", reason)

    def test_cooldown_blocks_too_soon(self):
        obs = _make_observer(
            self.workspace,
            disable_self_restart=False,
            allow_mid_self_mutation=True,
            max_self_restarts_per_policy=5,
            min_frames_between_self_restarts=50_000_000,
            self_restart_count_resume=1,
            last_self_restart_frame_resume=200_000_000,
        )
        ok, reason = obs._eligible_for_self_restart(env_frames=220_000_000)
        self.assertFalse(ok)
        self.assertIn("frames_since_self_restart", reason)

    def test_cooldown_passes_after_window(self):
        obs = _make_observer(
            self.workspace,
            disable_self_restart=False,
            allow_mid_self_mutation=True,
            max_self_restarts_per_policy=5,
            min_frames_between_self_restarts=50_000_000,
            self_restart_count_resume=1,
            last_self_restart_frame_resume=200_000_000,
        )
        ok, reason = obs._eligible_for_self_restart(env_frames=260_000_000)
        self.assertTrue(ok, reason)

    def test_first_attempt_passes_cooldown(self):
        # default _last_self_restart_frame should be deeply negative so
        # the very first self-restart never trips the cooldown.
        obs = _make_observer(
            self.workspace,
            disable_self_restart=False,
            allow_mid_self_mutation=True,
            max_self_restarts_per_policy=5,
            min_frames_between_self_restarts=100_000_000,
        )
        ok, _ = obs._eligible_for_self_restart(env_frames=10_000_000)
        self.assertTrue(ok)


class ResumeHookTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.workspace = Path(self._tmp.name)

    def test_defaults_when_no_resume(self):
        obs = _make_observer(self.workspace)
        self.assertEqual(obs._self_restart_count, 0)
        self.assertEqual(obs._cross_restart_count, 0)
        # _last_self_restart_frame must be deeply negative so the cooldown
        # gate never erroneously blocks the very first attempt.
        self.assertLess(obs._last_self_restart_frame, -10**12)

    def test_resume_args_take_effect(self):
        obs = _make_observer(
            self.workspace,
            self_restart_count_resume=3,
            last_self_restart_frame_resume=200_000_000,
            cross_restart_count_resume=2,
        )
        self.assertEqual(obs._self_restart_count, 3)
        self.assertEqual(obs._cross_restart_count, 2)
        self.assertEqual(obs._last_self_restart_frame, 200_000_000)


def _state(
    *,
    policy_idx: int = 0,
    frame: int = 300_000_000,
    nsucc_mean: float = 5.0,
    nsucc_max: float = 20.0,
    pbt_ranking_score: float = 100.0,
    iteration: int = 10,
):
    return {
        "policy_idx": policy_idx,
        "frame": frame,
        "iteration": iteration,
        "objective": pbt_ranking_score,
        "pbt_ranking_score": pbt_ranking_score,
        "r_meta_raw": pbt_ranking_score,
        "nsucc_mean": nsucc_mean,
        "checkpoint": "/tmp/dummy.pth",
        "curriculum": {
            "frame_since_restart": 200_000_000,
            "mean_prev_episode_successes": nsucc_mean,
            "max_prev_episode_successes": nsucc_max,
            "success_tolerance": 0.05,
        },
        "params": {},
    }


class TernaryPbtMidBranchTest(unittest.TestCase):
    """`_ternary_pbt` mid-tier branch with Patch F governor active."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.workspace = Path(self._tmp.name)

    def _patched(self, obs):
        # Heavy I/O paths get mocked so the test stays unit-scope.
        return patch.multiple(
            obs,
            _maybe_save_best_policy=lambda *a, **kw: None,
            _restart_process_with_new_params=lambda *a, **kw: None,
        )

    def _call_ternary(self, obs, population, env_frames):
        # _ternary_pbt expects `sorted_policies` ranked best->worst by
        # objective. Re-create that ordering here.
        objectives = {
            i: float(s["pbt_ranking_score"]) for i, s in population.items()
        }
        sorted_policies = sorted(
            objectives, key=lambda k: objectives[k], reverse=True
        )
        obs._ternary_pbt(
            sorted_policies=sorted_policies,
            objectives=objectives,
            population=population,
            env_frames=env_frames,
        )

    def test_default_blocks_mid_self_restart(self):
        obs = _make_observer(self.workspace, num_policies=3, policy_idx=0)
        # Build a 3-policy world where idx=0 ranks 2nd (mid tier under
        # the default ternary split: top=1, mid=1, bottom=1).
        population = {
            0: _state(policy_idx=0, pbt_ranking_score=50.0),  # mid
            1: _state(policy_idx=1, pbt_ranking_score=100.0),  # top
            2: _state(policy_idx=2, pbt_ranking_score=10.0),  # bottom
        }
        with self._patched(obs):
            self._call_ternary(obs, population, env_frames=300_000_000)
        # Mid worker was blocked: no self-restart was recorded, blocked
        # counter went up, restart_type is "skipped".
        self.assertEqual(obs._self_restart_count, 0)
        self.assertEqual(obs._self_restart_blocked_count, 1)
        self.assertEqual(obs._last_restart_type, 3)
        # The mid-tier branch was entered (mutation was attempted).
        self.assertEqual(obs._mid_self_mutation_count, 1)

    def test_legacy_mode_allows_then_caps(self):
        obs = _make_observer(
            self.workspace,
            num_policies=3,
            policy_idx=0,
            disable_self_restart=False,
            allow_mid_self_mutation=True,
            max_self_restarts_per_policy=1,
            min_frames_between_self_restarts=10_000_000,
        )
        population = {
            0: _state(policy_idx=0, pbt_ranking_score=50.0),
            1: _state(policy_idx=1, pbt_ranking_score=100.0),
            2: _state(policy_idx=2, pbt_ranking_score=10.0),
        }
        with self._patched(obs):
            # 1st PBT tick: mid path is allowed exactly once.
            self._call_ternary(obs, population, env_frames=300_000_000)
            self.assertEqual(obs._self_restart_count, 1)
            self.assertEqual(obs._last_restart_type, 1)

            # 2nd PBT tick (well past cooldown): lifetime cap kicks in.
            self._call_ternary(obs, population, env_frames=500_000_000)
        self.assertEqual(obs._self_restart_count, 1)  # still 1, not 2
        self.assertEqual(obs._self_restart_blocked_count, 1)
        self.assertEqual(obs._last_restart_type, 3)

    def test_bottom_branch_bumps_cross_counter(self):
        obs = _make_observer(self.workspace, num_policies=3, policy_idx=2)
        population = {
            0: _state(policy_idx=0, pbt_ranking_score=100.0),  # top
            1: _state(policy_idx=1, pbt_ranking_score=50.0),  # mid
            2: _state(policy_idx=2, pbt_ranking_score=10.0),  # bottom
        }
        with self._patched(obs):
            self._call_ternary(obs, population, env_frames=300_000_000)
        self.assertEqual(obs._cross_restart_count, 1)
        self.assertEqual(obs._last_restart_type, 2)
        self.assertEqual(obs._self_restart_count, 0)


if __name__ == "__main__":
    unittest.main()
