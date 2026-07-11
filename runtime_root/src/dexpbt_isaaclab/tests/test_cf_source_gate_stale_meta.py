"""P0: CF source gate rejects stale meta and unhealthy sources."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from dexpbt_isaaclab.pbt.pbt_observer import DexPBTPbtAlgoObserver
from dexpbt_isaaclab.scripts.run_counterfactual_block import _cf_source_health_ok, run_cf_block
from dexpbt_isaaclab.world_model.wm_counterfactual import CounterfactualManager


class CFSourceGateStaleMetaTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.workspace = Path(self._tmp.name)

    def _obs(self) -> DexPBTPbtAlgoObserver:
        obs = DexPBTPbtAlgoObserver(
            policy_idx=0,
            num_policies=2,
            mutable_params={
                "enable_wm": True,
                "enable_counterfactual": True,
                "wm_strategy": "value_imag_h1",
                "cf_source_min_nsucc_mean": 1.5,
                "cf_source_min_nsucc_max": 5.0,
            },
            workspace=str(self.workspace),
            train_dir=str(self.workspace),
            task_name="test-task",
            interval_steps=20_000_000,
            start_after=0,
            initial_delay=0,
            num_envs=8192,
        )
        obs.writer = None
        obs.pbt_iteration = 10
        return obs

    def _state(self, *, frame: int, nsucc_mean: float, nsucc_max: float) -> dict:
        return {
            "policy_idx": 1,
            "frame": frame,
            "objective": 0.5,
            "pbt_ranking_score": 0.5,
            "nsucc_mean": nsucc_mean,
            "nsucc_max": nsucc_max,
            "checkpoint": str(self.workspace / "src.pth"),
            "curriculum": {
                "success_tolerance": 0.05,
                "mean_prev_episode_successes": nsucc_mean,
                "max_prev_episode_successes": nsucc_max,
            },
            "params": {},
        }

    def test_cf_source_gate_blocks_zero_nsucc(self):
        obs = self._obs()
        (self.workspace / "src.pth").write_text("x")
        population = {
            0: None,
            1: self._state(frame=200_000_000, nsucc_mean=0.0, nsucc_max=0.0),
        }
        with patch.object(CounterfactualManager, "check_trigger", return_value=False) as chk:
            obs._check_cf_trigger(population, env_frames=200_000_000)
            chk.assert_not_called()

    def test_cf_source_gate_allows_healthy_source(self):
        obs = self._obs()
        (self.workspace / "src.pth").write_text("x")
        population = {
            1: self._state(frame=190_000_000, nsucc_mean=2.0, nsucc_max=10.0),
        }
        with patch.object(CounterfactualManager, "check_trigger", return_value=True) as chk:
            obs._check_cf_trigger(population, env_frames=200_000_000)
            chk.assert_called_once()
            kw = chk.call_args.kwargs
            self.assertEqual(kw["source_checkpoint_type"], "live")
            self.assertEqual(kw["source_meta_frame"], 190_000_000)

    def test_cf_source_gate_rejects_stale_meta(self):
        obs = self._obs()
        (self.workspace / "src.pth").write_text("x")
        population = {
            1: self._state(frame=100_000_000, nsucc_mean=3.0, nsucc_max=20.0),
        }
        with patch.object(CounterfactualManager, "check_trigger", return_value=False) as chk:
            obs._check_cf_trigger(population, env_frames=200_000_000)
            chk.assert_not_called()
            self.assertEqual(obs._cf_skipped_stale_meta_count, 1)

    def test_cf_request_records_source_checkpoint_type(self):
        with tempfile.TemporaryDirectory() as td:
            ws = Path(td)
            mgr = CounterfactualManager(
                workspace=ws,
                strategies=["none", "aux_low"],
                cf_source_min_nsucc_mean=1.5,
                cf_source_min_nsucc_max=5.0,
            )
            (ws / "ckpt.pth").write_text("x")
            fired = mgr.check_trigger(
                pbt_generation=5,
                source_policy_idx=1,
                source_r_meta=0.5,
                top_checkpoint=str(ws / "ckpt.pth"),
                top_params={},
                top_tolerance=0.05,
                source_nsucc_mean=2.0,
                source_nsucc_max=10.0,
                source_meta_frame=180_000_000,
                source_checkpoint_type="live",
            )
            self.assertTrue(fired)
            req = json.loads(mgr.request_path.read_text())
            self.assertEqual(req["source_checkpoint_type"], "live")
            self.assertEqual(req["source_meta_frame"], 180_000_000)

    def test_runner_rechecks_unhealthy_source(self):
        ok, reason = _cf_source_health_ok(
            {
                "source_nsucc_mean": 0.0,
                "source_nsucc_max": 0.0,
                "cf_source_min_nsucc_mean": 1.5,
                "cf_source_min_nsucc_max": 5.0,
            }
        )
        self.assertFalse(ok)
        result = run_cf_block(
            {
                "block_id": 1,
                "strategies": ["none"],
                "source_checkpoint": "/tmp/x",
                "source_nsucc_mean": 0.0,
                "source_nsucc_max": 0.0,
                "cf_source_min_nsucc_mean": 1.5,
                "cf_source_min_nsucc_max": 5.0,
            }
        )
        self.assertTrue(result["invalid_block"])


if __name__ == "__main__":
    unittest.main()
