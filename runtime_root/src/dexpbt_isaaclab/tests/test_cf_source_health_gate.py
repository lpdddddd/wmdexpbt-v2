"""Patch I (2026-05-31): CF source health gate + nsucc-based delta.

Smoke run 20260530_200619 showed CF blocks consistently producing
`low_signal` because the apparent top worker (chosen by `objective`)
had `nsucc_mean` near zero -- the high objective was driven by
tolerance progress alone, so every child necessarily ended up tied at
the same near-zero raw r_meta. Patch I addresses this two ways:

  1. `CounterfactualManager.check_trigger` consults
     `cf_source_min_nsucc_mean`, `cf_source_min_nsucc_max`, and
     `cf_source_min_ranking_score` thresholds, skipping the trigger
     when the would-be source fails any of them.

  2. `cf_delta_metric=nsucc` switches the posterior-feeding `delta_w`
     from `r_meta_after - r_meta_before` to `nsucc_after - nsucc_before`
     so the selector reacts to actual success counts.

Tests here pin down both behaviours in isolation.
"""

import tempfile
import unittest
from pathlib import Path

from dexpbt_isaaclab.world_model.wm_counterfactual import CounterfactualManager


def _mgr(workspace: Path, **kw) -> CounterfactualManager:
    defaults = dict(
        workspace=str(workspace),
        strategies=["none", "aux_low"],
        interval_gens=1,
        env_task="test-task",
        num_envs=4096,
        source_num_envs=8192,
        max_parallel_children=1,
        cf_cuda_visible_devices="0",
        cf_horizon=16,
        cf_steps=1_000,
        cf_train_frames=1_000,
        cf_min_eval_s=0.0,
        cf_require_nsucc_gt_zero=False,
    )
    defaults.update(kw)
    return CounterfactualManager(**defaults)


class CheckTriggerHealthGateTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.workspace = Path(self._tmp.name)

    def test_passes_when_gate_disabled(self):
        m = _mgr(self.workspace)
        ok = m.check_trigger(
            pbt_generation=10,
            source_policy_idx=0,
            source_r_meta=0.5,
            top_checkpoint=str(self.workspace / "fake.pth"),
            top_params={},
            top_tolerance=0.05,
            top_curriculum={"target_tolerance": 0.01},
            source_nsucc_mean=0.0,
            source_nsucc_max=0.0,
            source_ranking_score=0.05,
        )
        self.assertTrue(ok)
        self.assertEqual(m.skipped_source_health_gate_count, 0)

    def test_blocks_below_nsucc_mean_gate(self):
        m = _mgr(self.workspace, cf_source_min_nsucc_mean=1.5)
        ok = m.check_trigger(
            pbt_generation=10,
            source_policy_idx=0,
            source_r_meta=0.5,
            top_checkpoint=str(self.workspace / "fake.pth"),
            top_params={},
            top_tolerance=0.05,
            top_curriculum={"target_tolerance": 0.01},
            source_nsucc_mean=1.0,
            source_nsucc_max=20.0,
            source_ranking_score=0.5,
        )
        self.assertFalse(ok)
        self.assertEqual(m.skipped_source_health_gate_count, 1)

    def test_blocks_below_nsucc_max_gate(self):
        m = _mgr(self.workspace, cf_source_min_nsucc_max=10.0)
        ok = m.check_trigger(
            pbt_generation=10,
            source_policy_idx=0,
            source_r_meta=0.5,
            top_checkpoint=str(self.workspace / "fake.pth"),
            top_params={},
            top_tolerance=0.05,
            top_curriculum={"target_tolerance": 0.01},
            source_nsucc_mean=3.0,
            source_nsucc_max=5.0,
            source_ranking_score=0.5,
        )
        self.assertFalse(ok)
        self.assertEqual(m.skipped_source_health_gate_count, 1)

    def test_blocks_below_ranking_score_gate(self):
        m = _mgr(self.workspace, cf_source_min_ranking_score=0.3)
        ok = m.check_trigger(
            pbt_generation=10,
            source_policy_idx=0,
            source_r_meta=0.1,
            top_checkpoint=str(self.workspace / "fake.pth"),
            top_params={},
            top_tolerance=0.05,
            top_curriculum={"target_tolerance": 0.01},
            source_nsucc_mean=3.0,
            source_nsucc_max=20.0,
            source_ranking_score=0.1,
        )
        self.assertFalse(ok)
        self.assertEqual(m.skipped_source_health_gate_count, 1)

    def test_passes_when_all_gates_met(self):
        m = _mgr(
            self.workspace,
            cf_source_min_nsucc_mean=1.5,
            cf_source_min_nsucc_max=10.0,
            cf_source_min_ranking_score=0.3,
        )
        ok = m.check_trigger(
            pbt_generation=10,
            source_policy_idx=0,
            source_r_meta=0.5,
            top_checkpoint=str(self.workspace / "fake.pth"),
            top_params={},
            top_tolerance=0.05,
            top_curriculum={"target_tolerance": 0.01},
            source_nsucc_mean=2.0,
            source_nsucc_max=15.0,
            source_ranking_score=0.5,
        )
        self.assertTrue(ok)

    def test_falls_back_to_curriculum_fields(self):
        # When the explicit scalars are None, the manager pulls from
        # top_curriculum.mean_prev_episode_successes / max_prev_episode_successes.
        m = _mgr(self.workspace, cf_source_min_nsucc_mean=1.5)
        ok = m.check_trigger(
            pbt_generation=10,
            source_policy_idx=0,
            source_r_meta=0.5,
            top_checkpoint=str(self.workspace / "fake.pth"),
            top_params={},
            top_tolerance=0.05,
            top_curriculum={
                "target_tolerance": 0.01,
                "mean_prev_episode_successes": 0.5,
                "max_prev_episode_successes": 1.0,
            },
            source_nsucc_mean=None,
            source_nsucc_max=None,
            source_ranking_score=None,
        )
        self.assertFalse(ok)


class CfDeltaMetricRequestPayloadTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.workspace = Path(self._tmp.name)

    def test_default_metric_is_r_meta(self):
        m = _mgr(self.workspace)
        m.check_trigger(
            pbt_generation=10,
            source_policy_idx=0,
            source_r_meta=0.5,
            top_checkpoint=str(self.workspace / "fake.pth"),
            top_params={},
            top_tolerance=0.05,
            top_curriculum={"target_tolerance": 0.01},
            source_nsucc_mean=1.0,
            source_nsucc_max=10.0,
            source_ranking_score=0.5,
        )
        import json
        payload = json.loads((self.workspace / "cf_block_request.json").read_text())
        self.assertEqual(payload["cf_delta_metric"], "r_meta")
        # Health gate values are also surfaced in the request.
        self.assertIn("cf_source_min_nsucc_mean", payload)
        self.assertIn("cf_source_min_nsucc_max", payload)
        self.assertIn("cf_source_min_ranking_score", payload)

    def test_nsucc_metric_propagates_to_request(self):
        m = _mgr(self.workspace, cf_delta_metric="nsucc")
        m.check_trigger(
            pbt_generation=10,
            source_policy_idx=0,
            source_r_meta=0.5,
            top_checkpoint=str(self.workspace / "fake.pth"),
            top_params={},
            top_tolerance=0.05,
            top_curriculum={"target_tolerance": 0.01},
            source_nsucc_mean=1.0,
            source_nsucc_max=10.0,
            source_ranking_score=0.5,
        )
        import json
        payload = json.loads((self.workspace / "cf_block_request.json").read_text())
        self.assertEqual(payload["cf_delta_metric"], "nsucc")

    def test_unknown_metric_raises(self):
        with self.assertRaises(ValueError):
            _mgr(self.workspace, cf_delta_metric="garbage")


if __name__ == "__main__":
    unittest.main()
