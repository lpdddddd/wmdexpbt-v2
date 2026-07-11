"""v17.3 AWR rescue patch tests."""

from __future__ import annotations

import math
import unittest
from pathlib import Path
from unittest import mock

import torch

from dexpbt_isaaclab.awr.elite_awr import (
    EliteAWRBridge,
    EliteAWRBuffer,
    EliteAWRConfig,
    _compute_action_delta,
    _compute_awr_sample_weights,
    _delta_gate,
    awr_should_run_builder_this_rollout,
    compute_elite_awr_loss,
)


class TestV173AWRRescue(unittest.TestCase):
    def test_sample_weight_similarity_not_quality_duplicate(self):
        quality = torch.tensor([1.0, 2.0, 3.0, 4.0])
        similarity = torch.tensor([4.0, 3.0, 2.0, 1.0])
        loss_mask = torch.ones(4, dtype=torch.bool)
        delta_w = torch.ones(4)

        cfg_sim = EliteAWRConfig(
            score_beta_quality=0.0,
            score_beta_similarity=1.0,
            score_beta_skill_delta=0.0,
        )
        w_sim, _, _ = _compute_awr_sample_weights(
            quality=quality,
            delta_w=delta_w,
            loss_mask=loss_mask,
            cfg=cfg_sim,
            skill_delta=0.0,
            similarity=similarity,
        )

        cfg_q = EliteAWRConfig(
            score_beta_quality=1.0,
            score_beta_similarity=0.0,
            score_beta_skill_delta=0.0,
        )
        w_q, _, _ = _compute_awr_sample_weights(
            quality=quality,
            delta_w=delta_w,
            loss_mask=loss_mask,
            cfg=cfg_q,
            skill_delta=0.0,
            similarity=similarity,
        )

        self.assertGreater(w_sim[0].item(), w_sim[3].item())
        self.assertGreater(w_q[3].item(), w_q[0].item())

    def test_delta_metric_mean_less_brittle_than_linf(self):
        action = torch.zeros(1, 1, 23)
        mu = torch.zeros(1, 1, 23)
        action[0, 0, 0] = 2.0

        delta_linf = _compute_action_delta(action, mu, "linf")
        delta_mean = _compute_action_delta(action, mu, "mean")
        cfg = EliteAWRConfig(
            delta_soft=0.20,
            delta_mid=0.45,
            delta_hard=0.80,
        )

        w_linf = _delta_gate(delta_linf, cfg)
        w_mean = _delta_gate(delta_mean, cfg)

        self.assertEqual(w_linf.item(), 0.0)
        self.assertGreater(w_mean.item(), 0.0)
        self.assertLess(delta_mean.item(), cfg.delta_hard)

    def test_delta_metric_linf_backward_compatible(self):
        action = torch.randn(2, 3, 5)
        mu = torch.randn(2, 3, 5)
        old = (action - mu).abs().amax(dim=-1)
        new = _compute_action_delta(action, mu, "linf")
        self.assertTrue(torch.allclose(old, new))

    def test_collect_every_skips_builder(self):
        cfg = EliteAWRConfig(collect_every=4)
        bridge = EliteAWRBridge(
            buffer=EliteAWRBuffer(cfg, policy_idx=1),
            cfg=cfg,
        )
        results = [
            awr_should_run_builder_this_rollout(bridge, cfg)
            for _ in range(4)
        ]
        self.assertEqual(results, [False, False, False, True])
        self.assertEqual(bridge.stats["awr/collect_every"], 4.0)
        self.assertEqual(bridge.stats["awr/collect_counter"], 4.0)
        self.assertEqual(bridge.stats["awr/collection_skipped_collect_every"], 0.0)

    def test_collect_every_one_runs_every_rollout(self):
        cfg = EliteAWRConfig(collect_every=1)
        bridge = EliteAWRBridge(
            buffer=EliteAWRBuffer(cfg, policy_idx=1),
            cfg=cfg,
        )
        for _ in range(3):
            self.assertTrue(awr_should_run_builder_this_rollout(bridge, cfg))
            self.assertEqual(bridge.stats["awr/collection_skipped_collect_every"], 0.0)

    def test_v17_3_launcher_contains_rescue_args(self):
        repo_root = Path(__file__).resolve().parents[4]
        launcher = (
            repo_root
            / "results"
            / "launchers"
            / "start_regrasping_pbt_wm_strategy_zoo_v17_3_awr_rescue_900m_smoke.sh"
        )
        self.assertTrue(launcher.is_file(), msg=str(launcher))
        text = launcher.read_text()
        for needle in (
            "awr_delta_metric mean",
            "awr_collect_every 4",
            "awr_max_episode_len 320",
            "awr_start_after_frames 500000000",
            "awr_ramp_frames 150000000",
            "awr_collect_after_frames 400000000",
            "DEXPBT_AWR_BUILDER_MODE=vectorized",
        ):
            self.assertIn(needle, text)

    def test_delta_gate_zero_masks_weight(self):
        quality = torch.tensor([1.0, 2.0, 3.0, 4.0])
        similarity = torch.tensor([4.0, 3.0, 2.0, 1.0])
        loss_mask = torch.ones(4, dtype=torch.bool)
        delta_w = torch.tensor([1.0, 0.0, 0.0, 0.0])
        cfg = EliteAWRConfig(
            score_beta_quality=1.0,
            score_beta_similarity=0.0,
            score_beta_skill_delta=0.0,
        )
        w, _, stats = _compute_awr_sample_weights(
            quality=quality,
            delta_w=delta_w,
            loss_mask=loss_mask,
            cfg=cfg,
            skill_delta=0.0,
            similarity=similarity,
        )
        self.assertGreater(w[0].item(), 0.0)
        self.assertEqual(w[1].item(), 0.0)
        self.assertEqual(w[2].item(), 0.0)
        self.assertEqual(w[3].item(), 0.0)
        self.assertEqual(stats["awr/delta_gate_applied_to_weights"], 1.0)
        self.assertEqual(stats["awr/pre_gate_nonzero_weight_frac"], 1.0)
        self.assertEqual(stats["awr/post_gate_nonzero_weight_frac"], 0.25)

    def test_delta_gate_soft_scales_weight(self):
        quality = torch.tensor([1.0, 2.0, 3.0, 4.0])
        similarity = torch.tensor([4.0, 3.0, 2.0, 1.0])
        loss_mask = torch.ones(4, dtype=torch.bool)
        delta_w = torch.tensor([1.0, 0.5, 0.2, 0.0])
        cfg = EliteAWRConfig(
            score_beta_quality=1.0,
            score_beta_similarity=0.0,
            score_beta_skill_delta=0.0,
            entropy_floor_eps=0.99,
        )
        w, _, stats = _compute_awr_sample_weights(
            quality=quality,
            delta_w=delta_w,
            loss_mask=loss_mask,
            cfg=cfg,
            skill_delta=0.0,
            similarity=similarity,
        )
        self.assertEqual(w[3].item(), 0.0)
        self.assertLess(
            stats["awr/post_gate_nonzero_weight_frac"],
            stats["awr/pre_gate_nonzero_weight_frac"],
        )
        self.assertEqual(stats["awr/post_gate_nonzero_weight_frac"], 0.75)

    def test_entropy_floor_does_not_reactivate_zero_delta_gate(self):
        quality = torch.tensor([10.0, 1.0, 2.0, 1.0])
        similarity = torch.tensor([1.0, 1.0, 1.0, 1.0])
        loss_mask = torch.ones(4, dtype=torch.bool)
        delta_w = torch.tensor([1.0, 0.0, 1.0, 0.0])
        cfg = EliteAWRConfig(
            score_beta_quality=1.0,
            score_beta_similarity=0.0,
            score_beta_skill_delta=0.0,
            entropy_floor_eps=0.99,
        )
        w, _, _ = _compute_awr_sample_weights(
            quality=quality,
            delta_w=delta_w,
            loss_mask=loss_mask,
            cfg=cfg,
            skill_delta=0.0,
            similarity=similarity,
        )
        self.assertGreater(w[0].item(), 0.0)
        self.assertGreater(w[2].item(), 0.0)
        self.assertEqual(w[1].item(), 0.0)
        self.assertEqual(w[3].item(), 0.0)

    def test_weight_stats_present(self):
        quality = torch.tensor([1.0, 2.0, 3.0, 4.0])
        similarity = torch.tensor([4.0, 3.0, 2.0, 1.0])
        loss_mask = torch.ones(4, dtype=torch.bool)
        delta_w = torch.tensor([1.0, 0.5, 0.2, 0.0])
        cfg = EliteAWRConfig(start_after_frames=0, loss_weight=0.001)
        _, _, weight_stats = _compute_awr_sample_weights(
            quality=quality,
            delta_w=delta_w,
            loss_mask=loss_mask,
            cfg=cfg,
            similarity=similarity,
        )
        for key in (
            "awr/delta_gate_applied_to_weights",
            "awr/pre_gate_nonzero_weight_frac",
            "awr/post_gate_nonzero_weight_frac",
        ):
            self.assertIn(key, weight_stats)

        agent = mock.Mock()
        agent.ppo_device = torch.device("cpu")
        agent.frame = 1_000_000_000
        agent.is_rnn = False
        agent.mixed_precision = False
        agent._preproc_obs = lambda x: x

        class _Net(torch.nn.Module):
            def forward(self, input_dict):
                t = input_dict["prev_actions"].shape[0]
                return {
                    "prev_neglogp": torch.ones(t),
                    "mus": torch.zeros(t, input_dict["prev_actions"].shape[-1]),
                }

            def get_default_rnn_state(self):
                return None

        agent.model = _Net()
        batch = {
            "obs": torch.zeros(4, 2, 4),
            "action": torch.zeros(4, 2, 2),
            "loss_mask": torch.ones(4, 2, dtype=torch.bool),
            "quality": torch.full((4, 2), 3.0),
            "seq_length": 4,
        }
        _, stats = compute_elite_awr_loss(agent, batch, cfg, frames=agent.frame)
        for key in (
            "awr/delta_gate_applied_to_weights",
            "awr/pre_gate_nonzero_weight_frac",
            "awr/post_gate_nonzero_weight_frac",
        ):
            self.assertIn(key, stats)

    def test_awr_should_run_builder_exported(self):
        from dexpbt_isaaclab.awr import awr_should_run_builder_this_rollout as exported

        self.assertIs(exported, awr_should_run_builder_this_rollout)

    def test_v17_3_launcher_monitor_interval_fast(self):
        repo_root = Path(__file__).resolve().parents[4]
        launcher = (
            repo_root
            / "results"
            / "launchers"
            / "start_regrasping_pbt_wm_strategy_zoo_v17_3_awr_rescue_900m_smoke.sh"
        )
        text = launcher.read_text()
        self.assertIn("monitor_interval_s=600", text)
        self.assertIn("--interval-s 600", text)
        self.assertNotIn("monitor_interval_s=1800", text)
        self.assertNotIn("--interval-s 1800", text)


if __name__ == "__main__":
    unittest.main()
