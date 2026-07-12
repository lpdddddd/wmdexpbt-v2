"""Patch A (rev 3, 2026-05-30): PBT source-eligibility nsucc gate.

Pins down the failure mode observed in regrasping_cbwm_20260530_160513
where `policy_1` reported nsucc_mean=0.17 but r_meta_raw=0.94 (driven by
tolerance progress in `tolerance_successes_objective`) and was still
selected as PBT #1 source despite essentially zero success signal.

These tests validate:

  1. `_eligible_as_source_with_reason` rejects low-nsucc workers and
     returns a stable reason string.
  2. healthy workers (nsucc_mean >= 1, nsucc_max >= 5) pass when their
     restart cooldown also clears.
  3. `_ternary_pbt` skips replacement (does NOT call restart) when no
     candidate passes, instead of silently falling back to a bad source.
  4. legacy fallback path is preserved behind
     `pbt_allow_low_nsucc_source_fallback=True` for ablation / rollback.
  5. `_write_training_curve_point` carries the eligibility reason.
"""

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from dexpbt_isaaclab.pbt.pbt_observer import DexPBTPbtAlgoObserver


def _make_observer(
    workspace: Path,
    *,
    num_policies: int = 4,
    min_frames_as_source: int = 50_000_000,
    min_frames_between_restarts: int = 50_000_000,
    min_nsucc_mean_as_source: float = 1.0,
    min_nsucc_max_as_source: float = 5.0,
    allow_low_nsucc_source_fallback: bool = False,
    policy_idx: int = 0,
) -> DexPBTPbtAlgoObserver:
    return DexPBTPbtAlgoObserver(
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
        pbt_min_frames_as_source_after_restart=min_frames_as_source,
        pbt_min_frames_between_restarts=min_frames_between_restarts,
        pbt_min_nsucc_mean_as_source=min_nsucc_mean_as_source,
        pbt_min_nsucc_max_as_source=min_nsucc_max_as_source,
        pbt_allow_low_nsucc_source_fallback=allow_low_nsucc_source_fallback,
    )


def _state(
    *,
    policy_idx: int = 0,
    nsucc_mean: float = 0.0,
    nsucc_max: float = 0.0,
    r_meta_raw: float = 0.0,
    pbt_ranking_score: float | None = None,
    frame: int = 200_000_000,
    frame_since_restart: int = 200_000_000,
    frames_at_restart: int = 0,
    checkpoint: str = "/tmp/dummy.pth",
    params: dict | None = None,
) -> dict:
    if pbt_ranking_score is None:
        pbt_ranking_score = r_meta_raw
    return {
        "policy_idx": policy_idx,
        "iteration": 10,
        "frame": frame,
        "frames_at_restart": frames_at_restart,
        "objective": pbt_ranking_score,
        "r_meta_raw": r_meta_raw,
        "pbt_ranking_score": pbt_ranking_score,
        "nsucc_mean": nsucc_mean,
        "nsucc_max": nsucc_max,
        "curriculum": {"frame_since_restart": frame_since_restart},
        "checkpoint": checkpoint,
        "params": params or {"gamma": 0.99},
    }


class EligibleAsSourceWithReasonTest(unittest.TestCase):
    """Patch A: gate must reject low-nsucc workers regardless of r_meta_raw."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.obs = _make_observer(Path(self._tmp.name))

    def test_policy_1_collapse_case_not_eligible(self):
        """nsucc_mean=0.17, nsucc_max=2, r_raw=0.94 (the exact failure mode
        observed in regrasping_cbwm_20260530_160513) must NOT be eligible.
        """
        s = _state(nsucc_mean=0.17, nsucc_max=2, r_meta_raw=0.94)
        ok, reason = self.obs._eligible_as_source_with_reason(s, env_frames=300_000_000)
        self.assertFalse(ok)
        self.assertTrue(reason.startswith("nsucc_mean<"),
                        msg=f"unexpected reason: {reason!r}")

    def test_healthy_policy_eligible(self):
        """nsucc_mean=2.18, nsucc_max=31 (the policy_3 healthy case) must pass."""
        s = _state(nsucc_mean=2.18, nsucc_max=31, r_meta_raw=0.32)
        ok, reason = self.obs._eligible_as_source_with_reason(s, env_frames=300_000_000)
        self.assertTrue(ok, msg=f"reason={reason!r}")
        self.assertEqual(reason, "ok")

    def test_nsucc_max_below_threshold_not_eligible(self):
        """High mean alone is not enough; max must clear too (anti one-off luck)."""
        s = _state(nsucc_mean=1.5, nsucc_max=3, r_meta_raw=0.40)
        ok, reason = self.obs._eligible_as_source_with_reason(s, env_frames=300_000_000)
        self.assertFalse(ok)
        self.assertTrue(reason.startswith("nsucc_max<"),
                        msg=f"unexpected reason: {reason!r}")

    def test_short_fsr_not_eligible(self):
        """Anti-churn gate still takes precedence over nsucc gate."""
        s = _state(
            nsucc_mean=3.0,
            nsucc_max=20,
            r_meta_raw=0.5,
            frame=60_000_000,
            frames_at_restart=50_000_000,
            frame_since_restart=10_000_000,
        )
        ok, reason = self.obs._eligible_as_source_with_reason(s, env_frames=300_000_000)
        self.assertFalse(ok)
        self.assertTrue(reason.startswith("frames_since_restart<"),
                        msg=f"unexpected reason: {reason!r}")

    def test_nsucc_zero_collapsed_not_eligible(self):
        """Exact collapse case (policy_5: nsucc=0, r_raw=0.94)."""
        s = _state(nsucc_mean=0.0, nsucc_max=0, r_meta_raw=0.94)
        ok, reason = self.obs._eligible_as_source_with_reason(s, env_frames=300_000_000)
        self.assertFalse(ok)

    def test_none_state_not_eligible(self):
        ok, reason = self.obs._eligible_as_source_with_reason(None, env_frames=0)
        self.assertFalse(ok)
        self.assertEqual(reason, "no_state")

    def test_threshold_is_configurable(self):
        """A higher mean threshold should reject more workers."""
        obs2 = _make_observer(Path(self._tmp.name),
                              min_nsucc_mean_as_source=3.0)
        s = _state(nsucc_mean=2.18, nsucc_max=31, r_meta_raw=0.32)
        ok, reason = obs2._eligible_as_source_with_reason(s, env_frames=300_000_000)
        self.assertFalse(ok)
        self.assertTrue(reason.startswith("nsucc_mean<3.0"),
                        msg=f"unexpected reason: {reason!r}")

    def test_bool_wrapper_matches_tuple_form(self):
        """`_is_eligible_as_source` must agree with the (bool, reason) helper."""
        s = _state(nsucc_mean=0.17, nsucc_max=2, r_meta_raw=0.94)
        ok_bool = self.obs._is_eligible_as_source(s, env_frames=300_000_000)
        ok_tuple, _ = self.obs._eligible_as_source_with_reason(s, env_frames=300_000_000)
        self.assertEqual(ok_bool, ok_tuple)


class StateNsuccMaxTest(unittest.TestCase):
    """`_state_nsucc_max` falls back through state -> curriculum -> 0."""

    def test_reads_state_field(self):
        self.assertEqual(
            DexPBTPbtAlgoObserver._state_nsucc_max(
                {"nsucc_max": 12.0, "curriculum": {}}
            ),
            12.0,
        )

    def test_falls_back_to_curriculum(self):
        self.assertEqual(
            DexPBTPbtAlgoObserver._state_nsucc_max(
                {"curriculum": {"max_prev_episode_successes": 42.0}}
            ),
            42.0,
        )

    def test_zero_when_missing(self):
        self.assertEqual(
            DexPBTPbtAlgoObserver._state_nsucc_max({"curriculum": {}}), 0.0
        )

    def test_handles_none(self):
        self.assertEqual(DexPBTPbtAlgoObserver._state_nsucc_max(None), 0.0)

    def test_handles_nan_safely(self):
        self.assertEqual(
            DexPBTPbtAlgoObserver._state_nsucc_max({"nsucc_max": float("nan")}),
            0.0,
        )


class TernaryPbtNoSourceSkipsReplacementTest(unittest.TestCase):
    """When NO candidate passes the gate AND fallback is disabled (default),
    `_ternary_pbt` must NOT call `_restart_process_with_new_params`.
    """

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.workspace = Path(self._tmp.name)
        self.ckpt = self.workspace / "dummy.pth"
        self.ckpt.write_text("dummy")
        # Bottom-tier worker is the test subject (policy_3 / rank 3 of 4)
        self.obs = _make_observer(
            self.workspace, num_policies=4, policy_idx=3
        )
        # `writer` is set in `after_init` by rl_games; in unit tests we
        # bypass that and just disable TB writes.
        self.obs.writer = None
        # Permit elimination so the only thing stopping the restart is the
        # source-gate logic we want to exercise.
        self.obs.min_frames_between_restarts = 0

    def _seed_skill_delta(self, obs, population, *, env_frames: int, target_idx: int):
        from dexpbt_isaaclab.pbt.pbt_curriculum_p1 import WindowSample

        step = int(obs.interval_steps)
        for idx, st in population.items():
            mean = float(st.get("nsucc_mean", 0.0))
            w = obs._policy_window(idx)
            w.samples = []
            if int(idx) == int(target_idx):
                seq = [0.0] * 6
            else:
                # Live source gate needs stable positive skill_delta even when
                # absolute nsucc is low (legacy fallback path).
                seq = [0.2 * i for i in range(6)]
            for i, m in enumerate(seq):
                w.samples.append(
                    WindowSample(
                        env_frames=int(env_frames) - (5 - i) * step,
                        nsucc_mean=float(m),
                        nsucc_max=float(m) + 5.0,
                        success_tolerance=0.05,
                        pbt_iteration=i,
                        lineage_id="same",
                    )
                )

    def _build_population_all_unhealthy(self) -> dict:
        """policy_0/1 high r_meta but nsucc=0.17 (collapse leaders);
        policy_2/3 fresh / no signal.
        """
        ck = str(self.ckpt)
        return {
            0: _state(policy_idx=0, nsucc_mean=0.17, nsucc_max=2,
                      r_meta_raw=0.94, pbt_ranking_score=0.94, checkpoint=ck),
            1: _state(policy_idx=1, nsucc_mean=0.00, nsucc_max=0,
                      r_meta_raw=0.94, pbt_ranking_score=0.05, checkpoint=ck),
            2: _state(policy_idx=2, nsucc_mean=0.30, nsucc_max=3,
                      r_meta_raw=0.10, pbt_ranking_score=0.10, checkpoint=ck),
            3: _state(policy_idx=3, nsucc_mean=0.00, nsucc_max=0,
                      r_meta_raw=0.00, pbt_ranking_score=0.00,
                      frame=55_000_000, frames_at_restart=50_000_000,
                      frame_since_restart=5_000_000, checkpoint=ck),
        }

    def test_default_skips_replacement_no_restart(self):
        pop = self._build_population_all_unhealthy()
        self._seed_skill_delta(self.obs, pop, env_frames=300_000_000, target_idx=3)
        sorted_pols = sorted(pop.keys(), key=lambda k: -pop[k]["objective"])
        objs = {k: pop[k]["objective"] for k in pop}

        with mock.patch.object(
            self.obs, "_restart_process_with_new_params"
        ) as m_restart, mock.patch.object(
            self.obs, "_maybe_save_best_policy"
        ):
            self.obs._ternary_pbt(sorted_pols, objs, pop, env_frames=300_000_000)

        m_restart.assert_not_called()

    def test_legacy_fallback_restored_when_flag_on(self):
        """`pbt_allow_low_nsucc_source_fallback=True` must restore the old
        behaviour where a bad source is re-admitted and replacement runs.
        """
        obs_legacy = _make_observer(
            self.workspace, num_policies=4, policy_idx=3,
            allow_low_nsucc_source_fallback=True,
        )
        obs_legacy.writer = None
        obs_legacy.min_frames_between_restarts = 0

        pop = self._build_population_all_unhealthy()
        self._seed_skill_delta(obs_legacy, pop, env_frames=300_000_000, target_idx=3)
        sorted_pols = sorted(pop.keys(), key=lambda k: -pop[k]["objective"])
        objs = {k: pop[k]["objective"] for k in pop}

        with mock.patch.object(
            obs_legacy, "_restart_process_with_new_params", return_value=True
        ) as m_restart, mock.patch.object(
            obs_legacy, "_maybe_save_best_policy"
        ):
            obs_legacy._ternary_pbt(sorted_pols, objs, pop, env_frames=300_000_000)

        # With legacy fallback ON, a bad source gets picked and restart fires.
        m_restart.assert_called_once()

    def test_healthy_candidate_uses_normal_path(self):
        """When at least one top candidate is eligible, replacement proceeds
        normally (no skip).

        Note: with num_policies=4, n_top=1, so only the rank-1 policy is
        considered as a source. We must therefore promote the healthy
        worker to rank-1 (highest ranking_score) for this test.
        """
        pop = self._build_population_all_unhealthy()
        # Replace policy_0 (the original rank-1, collapsed) with a healthy
        # worker so the rank-1 candidate now passes the nsucc gate.
        pop[0] = _state(
            policy_idx=0,
            nsucc_mean=2.5,
            nsucc_max=20,
            r_meta_raw=0.50,
            pbt_ranking_score=0.50,
            checkpoint=str(self.ckpt),
        )
        self._seed_skill_delta(self.obs, pop, env_frames=300_000_000, target_idx=3)
        sorted_pols = sorted(pop.keys(), key=lambda k: -pop[k]["objective"])
        objs = {k: pop[k]["objective"] for k in pop}
        # Sanity: rank 1 should be policy_0 (highest ranking_score).
        self.assertEqual(sorted_pols[0], 0)

        with mock.patch.object(
            self.obs, "_restart_process_with_new_params", return_value=True
        ) as m_restart, mock.patch.object(
            self.obs, "_maybe_save_best_policy"
        ):
            self.obs._ternary_pbt(sorted_pols, objs, pop, env_frames=300_000_000)

        m_restart.assert_called_once()


class TrainingCurveCarriesEligibilityReasonTest(unittest.TestCase):
    """`_write_training_curve_point` must persist the eligibility reason
    so audits can grep why a worker was filtered out at a given tick.
    """

    def test_curve_writes_eligible_source_reason(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            obs = _make_observer(Path(tmpdir))
            obs.curr_eligible_source_reason = "nsucc_mean<1.0"
            obs.curr_eligible_as_source = False
            obs.pbt_iteration = 5

            own_state = _state(nsucc_mean=0.17, nsucc_max=2, r_meta_raw=0.94)
            obs._write_training_curve_point(
                own_state,
                env_frames=300_000_000,
                curriculum={"mean_prev_episode_successes": 0.17,
                            "max_prev_episode_successes": 2.0,
                            "success_tolerance": 0.014,
                            "target_tolerance": 0.01,
                            "initial_tolerance": 0.075,
                            "frame_since_restart": 200_000_000,
                            "last_curriculum_update": 0},
                tier="top",
            )

            curve_path = (Path(tmpdir) / "metrics"
                          / f"training_curve_policy_{obs.policy_idx:02d}.jsonl")
            self.assertTrue(curve_path.exists())
            lines = curve_path.read_text().splitlines()
            self.assertEqual(len(lines), 1)
            pt = json.loads(lines[0])

            self.assertIn("eligible_source_reason", pt)
            self.assertEqual(pt["eligible_source_reason"], "nsucc_mean<1.0")
            self.assertIn("min_nsucc_mean_as_source", pt)
            self.assertEqual(pt["min_nsucc_mean_as_source"], 1.0)
            self.assertIn("min_nsucc_max_as_source", pt)
            self.assertEqual(pt["min_nsucc_max_as_source"], 5.0)
            self.assertIn("allow_low_nsucc_source_fallback", pt)
            self.assertFalse(pt["allow_low_nsucc_source_fallback"])


if __name__ == "__main__":
    unittest.main()
