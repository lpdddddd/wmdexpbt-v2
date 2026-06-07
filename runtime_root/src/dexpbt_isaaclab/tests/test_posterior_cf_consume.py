"""P0: posterior consumes CF results without PBT restart."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from dexpbt_isaaclab.pbt.pbt_observer import DexPBTPbtAlgoObserver
from dexpbt_isaaclab.world_model.wm_counterfactual import CounterfactualManager


def _result(block_id: int) -> dict:
    return {
        "block_id": block_id,
        "delta_w": {"aux_low": 0.1, "none": 0.0},
        "posterior_weights": {"aux_low": 1.0},
        "invalid_block": False,
        "all_strategies_tied": False,
        "cf_delta_mode": "nsucc",
        "results": {
            "none": {"invalid": False},
            "aux_low": {"invalid": False},
            "aux_mid": {"invalid": False},
            "value_imag_h1": {"invalid": False},
            "value_imag_h3": {"invalid": False},
        },
        "timestamp": 1.0,
    }


class PosteriorConsumeWithoutRestartTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.workspace = Path(self._tmp.name)

    def _make_obs(self, policy_idx: int = 0) -> DexPBTPbtAlgoObserver:
        obs = DexPBTPbtAlgoObserver(
            policy_idx=policy_idx,
            num_policies=2,
            mutable_params={
                "enable_wm": True,
                "enable_counterfactual": True,
                "wm_strategy": "value_imag_h1",
                "cf_source_min_nsucc_mean": 0.0,
                "cf_source_min_nsucc_max": 0.0,
            },
            workspace=str(self.workspace),
            train_dir=str(self.workspace / "train"),
            task_name="test-task",
            interval_steps=10_000_000,
            start_after=0,
            initial_delay=0,
            num_envs=8192,
        )
        obs.writer = None
        obs.algo = type("A", (), {"frame": 50_000_000})()
        return obs

    def test_posterior_consumes_cf_result_without_restart(self):
        obs = self._make_obs(policy_idx=0)
        assert obs._cf_manager is not None
        assert obs._posterior_selector is not None

        obs._cf_manager.result_path.write_text(json.dumps(_result(5)))
        self.assertTrue(obs._maybe_consume_counterfactual_result(50_000_000))
        self.assertEqual(obs._last_consumed_cf_block_id, 5)
        self.assertEqual(obs._cf_posterior_update_count, 1)
        self.assertFalse(obs._cf_manager.result_path.exists())
        data = json.loads(obs._posterior_selector.shared_path.read_text())
        self.assertGreaterEqual(int(data["aux_low"]["n"]), 2)

    def test_posterior_does_not_double_consume_same_block(self):
        obs = self._make_obs(policy_idx=0)
        obs._cf_manager.result_path.write_text(json.dumps(_result(7)))
        self.assertTrue(obs._maybe_consume_counterfactual_result(50_000_000))
        obs._cf_manager.result_path.write_text(json.dumps(_result(7)))
        self.assertFalse(obs._maybe_consume_counterfactual_result(60_000_000))
        self.assertEqual(obs._cf_posterior_update_count, 1)

    def test_only_policy0_updates_posterior(self):
        obs = self._make_obs(policy_idx=1)
        obs._cf_manager.result_path.write_text(json.dumps(_result(3)))
        self.assertFalse(obs._maybe_consume_counterfactual_result(50_000_000))
        self.assertEqual(obs._cf_posterior_update_count, 0)


class CounterfactualManagerPeekTest(unittest.TestCase):
    def test_peek_skips_consumed_block_ids(self):
        with tempfile.TemporaryDirectory() as td:
            ws = Path(td)
            mgr = CounterfactualManager(workspace=ws, strategies=["none", "aux_low"])
            mgr.result_path.write_text(json.dumps(_result(2)))
            self.assertIsNone(mgr.peek_unconsumed_result(last_consumed_block_id=2))
            self.assertIsNotNone(mgr.peek_unconsumed_result(last_consumed_block_id=1))


if __name__ == "__main__":
    unittest.main()
