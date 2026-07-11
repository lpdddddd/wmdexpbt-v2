"""Patch I (2026-05-31): nsucc-based delta_w for CF posterior.

When `cf_delta_metric=nsucc`, `run_cf_block` must compute the
posterior-feeding `delta_w` from `nsucc_after - nsucc_before` instead
of `r_meta_after - r_meta_before`. The result JSON should also surface
both per-strategy delta dicts (`r_meta_delta`, `nsucc_delta`) so an
analyzer can cross-check signal quality.

We drive `run_cf_block` directly with a pre-baked request dict, and
monkey-patch `launch_child` so we don't need to spin up real Isaac Sim
subprocesses. Each child is hand-crafted to expose nsucc_before /
nsucc_after / r_meta_before / r_meta_after fields the way a real
status.txt would.
"""

import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from dexpbt_isaaclab.scripts import run_counterfactual_block as cf_mod


def _status_payload(
    *,
    r_meta_before: float,
    r_meta_after: float,
    nsucc_before: float,
    nsucc_after: float,
    elapsed_s: float = 60.0,
    r_meta_samples: int = 20,
) -> dict:
    return {
        "r_meta": r_meta_after,
        "r_meta_before": r_meta_before,
        "r_meta_after": r_meta_after,
        "nsucc_before": nsucc_before,
        "nsucc_after": nsucc_after,
        "mean_prev_episode_successes": nsucc_after,
        "elapsed_s": elapsed_s,
        "r_meta_samples": r_meta_samples,
        "frame": 20_000_000,
        "child_train_frames": 20_000_000,
        "num_envs": 4096,
        "horizon_length": 16,
        "iteration": 100,
    }


def _make_request(
    workspace: Path,
    *,
    cf_delta_metric: str,
    cf_delta_mode: str = "learning_improvement",
) -> dict:
    ckpt = workspace / "source.pth"
    ckpt.write_text("dummy")
    return {
        "_workspace": str(workspace),
        "block_id": 1,
        "source_policy_idx": 0,
        "source_r_meta": 0.5,
        "source_checkpoint": str(ckpt),
        "source_params": {
            "gamma": 0.99,
            "kl_threshold": 0.01,
            "grad_norm": 1.0,
            "critic_coef": 4.0,
            "bounds_loss_coef": 1e-4,
            "e_clip": 0.1,
            "mini_epochs": 2,
            "learning_rate": 1e-4,
            "alpha_reach": 30.0,
            "alpha_pick": 20.0,
            "r_picked": 300.0,
            "alpha_targ": 200.0,
            "r_success": 1000.0,
            "kuka_actions_penalty_scale": 3e-3,
            "allegro_actions_penalty_scale": 3e-4,
            "wm_strategy": "value_imag_h3",
            "wm_ensemble_size": 3,
            "enable_wm": False,
            "enable_counterfactual": False,
        },
        "source_tolerance": 0.05,
        "source_target_tolerance": 0.01,
        "source_curriculum": {
            "target_tolerance": 0.01,
            "initial_tolerance": 0.075,
        },
        "source_env_frames": 200_000_000,
        "source_nsucc_mean": 3.0,
        "source_nsucc_max": 20.0,
        "source_ranking_score": 0.5,
        "strategies": ["none", "aux_low"],
        "env_task": "test-task",
        "num_envs": 4096,
        "source_num_envs": 8192,
        "cf_num_envs": 4096,
        "cf_horizon": 16,
        "cf_max_parallel_children": 1,
        "cf_cuda_visible_devices": "0",
        "cf_steps": 1_000,
        "cf_train_frames": 1_000_000,
        "cf_reset_epoch_counter": True,
        "cf_min_eval_s": 0.0,
        "cf_delta_mode": cf_delta_mode,
        "cf_low_signal_action": "skip",
        "cf_low_signal_weight": 0.25,
        "cf_delta_metric": cf_delta_metric,
        "cf_source_min_nsucc_mean": 0.0,
        "cf_source_min_nsucc_max": 0.0,
        "cf_source_min_ranking_score": -1.0,
        "timestamp": 0.0,
    }


class CfDeltaMetricAggregationTest(unittest.TestCase):
    """Drive `run_cf_block` end-to-end with mocked `launch_child`."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.workspace = Path(self._tmp.name)

    def _stub_launch_child(self, strategy_statuses):
        """Build a `launch_child` replacement that pre-writes the status
        file and returns a tuple `(MockProc, child_dir)` matching the
        production signature. The runner consumes proc.wait() and then
        reads status.txt from child_dir.
        """

        class _MockProc:
            def __init__(self):
                self.returncode = 0

            def wait(self, timeout=None):  # noqa: ARG002
                return 0

            def kill(self):
                return None

        def fake_launch_child(strategy, block_id, batch_idx, request, gpu_id):
            child_dir = (
                Path(request["workspace_dir"])
                / f"cf_block_{block_id}"
                / f"batch_{batch_idx}"
                / f"child_{strategy}"
            )
            child_dir.mkdir(parents=True, exist_ok=True)
            status = strategy_statuses[strategy]
            # status.txt uses `key=value` lines; ints/floats serialize fine.
            lines = []
            for k, v in status.items():
                if isinstance(v, float) and math.isnan(v):
                    lines.append(f"{k}=null")
                else:
                    lines.append(f"{k}={v}")
            (child_dir / "status.txt").write_text("\n".join(lines))
            return _MockProc(), child_dir

        return fake_launch_child

    def _run_with_metric(self, metric: str) -> dict:
        # Each child gets a realistic before/after pair so both r_meta and
        # nsucc deltas have non-trivial structure.
        strategy_statuses = {
            "none": _status_payload(
                r_meta_before=0.40, r_meta_after=0.42,
                nsucc_before=2.0, nsucc_after=2.1,
            ),
            "aux_low": _status_payload(
                r_meta_before=0.41, r_meta_after=0.50,
                nsucc_before=2.0, nsucc_after=4.0,
            ),
        }
        with patch.object(
            cf_mod, "launch_child", self._stub_launch_child(strategy_statuses)
        ):
            request = _make_request(self.workspace, cf_delta_metric=metric)
            result = cf_mod.run_cf_block(request)
        return result

    def test_r_meta_metric_drives_delta_w(self):
        result = self._run_with_metric("r_meta")
        # learning_improvement: child_delta - none_delta.
        # none: 0.42 - 0.40 = 0.02
        # aux_low: 0.50 - 0.41 = 0.09; delta_w = 0.09 - 0.02 = 0.07
        self.assertAlmostEqual(result["delta_w"]["aux_low"], 0.07, places=4)
        # r_meta_delta should match delta_w exactly.
        self.assertAlmostEqual(result["r_meta_delta"]["aux_low"], 0.07, places=4)
        # nsucc_delta is reported for diagnostics:
        # none nsucc delta = 0.1; aux_low nsucc delta = 2.0; diff = 1.9
        self.assertAlmostEqual(result["nsucc_delta"]["aux_low"], 1.9, places=4)
        # Result also tags the chosen metric.
        self.assertEqual(result["cf_delta_metric"], "r_meta")

    def test_nsucc_metric_drives_delta_w(self):
        result = self._run_with_metric("nsucc")
        # nsucc deltas as above: aux_low = 1.9
        self.assertAlmostEqual(result["delta_w"]["aux_low"], 1.9, places=4)
        self.assertAlmostEqual(result["nsucc_delta"]["aux_low"], 1.9, places=4)
        # r_meta delta unchanged at 0.07 for diagnostics.
        self.assertAlmostEqual(result["r_meta_delta"]["aux_low"], 0.07, places=4)
        self.assertEqual(result["cf_delta_metric"], "nsucc")

    def test_none_strategy_has_zero_delta_w(self):
        # By construction, `none` is its own baseline -> delta_w[none] = 0.
        for metric in ("r_meta", "nsucc"):
            result = self._run_with_metric(metric)
            self.assertAlmostEqual(result["delta_w"]["none"], 0.0, places=6)

    def test_unknown_metric_falls_back_to_r_meta(self):
        # The runner accepts unknown metrics defensively (with a warning).
        request = _make_request(self.workspace, cf_delta_metric="garbage")
        strategy_statuses = {
            "none": _status_payload(
                r_meta_before=0.40, r_meta_after=0.42,
                nsucc_before=2.0, nsucc_after=2.1,
            ),
            "aux_low": _status_payload(
                r_meta_before=0.41, r_meta_after=0.50,
                nsucc_before=2.0, nsucc_after=4.0,
            ),
        }
        with patch.object(
            cf_mod, "launch_child", self._stub_launch_child(strategy_statuses)
        ):
            result = cf_mod.run_cf_block(request)
        # Falls back to r_meta delta:
        self.assertAlmostEqual(result["delta_w"]["aux_low"], 0.07, places=4)
        self.assertEqual(result["cf_delta_metric"], "r_meta")


if __name__ == "__main__":
    unittest.main()
