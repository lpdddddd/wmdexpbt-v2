#!/usr/bin/env python3
"""Unit tests for v17.1 rollout-buffer Elite-AWR builder."""

from __future__ import annotations

import argparse
import tempfile
import unittest
from pathlib import Path

import torch

from dexpbt_isaaclab.awr.elite_awr import (
    EliteAWRBridge,
    EliteAWRBuffer,
    EliteAWRConfig,
    awr_active_from_args,
    build_elite_awr_config_from_args,
    load_awr_sidecar,
    pbt_should_keep_local_progress_counters,
    save_awr_sidecar_topk,
)
from dexpbt_isaaclab.awr.elite_awr_rollout_builder import (
    PPO_BUFFER_ACTIONS,
    build_awr_entries_from_experience_buffer,
    check_rollout_buffer_action_sanity,
    should_collect_awr,
)
from dexpbt_isaaclab.world_model.wm_strategy_profiles import apply_wm_strategy_profile


class _FakeBuffer:
    def __init__(self, tensor_dict: dict[str, torch.Tensor]) -> None:
        self.tensor_dict = tensor_dict


def _make_rollout(
    T: int = 8,
    N: int = 4,
    *,
    raw_actions: bool = True,
) -> _FakeBuffer:
    obs_dim = 6
    act_dim = 2
    actions = torch.randn(T, N, act_dim)
    if raw_actions:
        actions = actions * 3.0
    dones = torch.zeros(T, N, 1, dtype=torch.bool)
    dones[min(T - 1, 3), 0, 0] = True
    if N > 1:
        dones[min(T - 1, 7), 1, 0] = True
    success_cnt = torch.zeros(T, N, 1)
    success_cnt[min(T - 1, 3), 0, 0] = 1.0
    return _FakeBuffer(
        {
            "obses": torch.randn(T, N, obs_dim),
            "actions": actions,
            "dones": dones,
            "aux_d_hat": torch.linspace(0.2, 0.05, T).unsqueeze(-1).unsqueeze(-1).expand(T, N, 1),
            "aux_success_tolerance": torch.full((T, N, 1), 0.05),
            "aux_lifted": torch.ones(T, N, 1, dtype=torch.bool),
            "aux_failure": torch.zeros(T, N, 1, dtype=torch.bool),
            "aux_success_cnt": success_cnt,
            "mus": torch.zeros(T, N, act_dim),
        }
    )


class TestEliteAWRRolloutBuilder(unittest.TestCase):
    def test_build_entries_action_equals_ppo_buffer(self):
        cfg = EliteAWRConfig(
            quality_min=0.0,
            min_loss_steps_per_entry=2,
            min_burnin_steps=1,
            collect_after_frames=0,
        )
        T, N = 8, 1
        actions = torch.randn(T, N, 2) * 3.0
        dones = torch.zeros(T, N, 1, dtype=torch.bool)
        dones[-1, 0, 0] = True
        td = {
            "obses": torch.randn(T, N, 6),
            "actions": actions,
            "dones": dones,
            "aux_d_hat": torch.linspace(0.2, 0.05, T).view(T, N, 1),
            "aux_success_tolerance": torch.full((T, N, 1), 0.05),
            "aux_lifted": torch.ones(T, N, 1, dtype=torch.bool),
            "aux_failure": torch.zeros(T, N, 1, dtype=torch.bool),
            "aux_success_cnt": torch.zeros(T, N, 1),
        }
        td["aux_success_cnt"][-1, 0, 0] = 1.0
        entries, stats = build_awr_entries_from_experience_buffer(
            experience_buffer=_FakeBuffer(td),
            cfg=cfg,
            source_policy=1,
            source_profile="elite_bank_actor_distill_awr",
            source_nsucc=30.0,
            source_retention=0.8,
            env_frames=1_000,
        )
        self.assertGreaterEqual(stats["awr/rollout_builder_entries"], 1.0)
        self.assertGreater(len(entries), 0)
        entry = entries[0]
        self.assertTrue(torch.allclose(entry.action, actions[:, 0].cpu()))

    def test_no_env_clamped_actions_used(self):
        cfg = EliteAWRConfig(quality_min=0.0, min_loss_steps_per_entry=2, collect_after_frames=0)
        exp = _make_rollout(raw_actions=True)
        entries, _ = build_awr_entries_from_experience_buffer(
            experience_buffer=exp,
            cfg=cfg,
            source_policy=1,
            source_profile="x",
            source_nsucc=None,
            source_retention=None,
            env_frames=0,
        )
        self.assertTrue(any(float(e.action.abs().max()) > 1.0 for e in entries))

    def test_collect_after_frames_gate(self):
        cfg = EliteAWRConfig(
            enable=True,
            loss_weight=0.001,
            collect_after_frames=600_000_000,
        )
        bridge = EliteAWRBridge(buffer=EliteAWRBuffer(cfg), cfg=cfg)
        self.assertFalse(should_collect_awr(100, cfg, bridge))
        self.assertTrue(should_collect_awr(700_000_000, cfg, bridge))

    def test_disabled_bridge_skips(self):
        cfg = EliteAWRConfig(collect_after_frames=0)
        bridge = EliteAWRBridge(buffer=EliteAWRBuffer(cfg), cfg=cfg)
        bridge.disable("test")
        self.assertFalse(should_collect_awr(1_000_000_000, cfg, bridge))

    def test_selected_entries_only_cpu_materialized(self):
        cfg = EliteAWRConfig(quality_min=0.0, min_loss_steps_per_entry=2, collect_after_frames=0)
        exp = _make_rollout(T=8, N=4)
        entries, stats = build_awr_entries_from_experience_buffer(
            experience_buffer=exp,
            cfg=cfg,
            source_policy=1,
            source_profile="x",
            source_nsucc=None,
            source_retention=None,
            env_frames=0,
        )
        self.assertGreater(stats["awr/rollout_builder_selected_fragments"], 0.0)
        self.assertGreater(len(entries), 0)
        for entry in entries:
            self.assertFalse(entry.obs.is_cuda)
            self.assertFalse(entry.action.is_cuda)
            self.assertEqual(entry.action.dtype, exp.tensor_dict["actions"].dtype)

    def test_missing_aux_neutral_no_crash(self):
        cfg = EliteAWRConfig(quality_min=0.0, min_loss_steps_per_entry=2, collect_after_frames=0)
        td = {
            "obses": torch.randn(4, 2, 3),
            "actions": torch.randn(4, 2, 2) * 2.0,
            "dones": torch.zeros(4, 2, 1, dtype=torch.bool),
        }
        td["dones"][3, 0, 0] = True
        entries, stats = build_awr_entries_from_experience_buffer(
            experience_buffer=_FakeBuffer(td),
            cfg=cfg,
            source_policy=1,
            source_profile="x",
            source_nsucc=None,
            source_retention=None,
            env_frames=0,
        )
        self.assertIsInstance(entries, list)
        self.assertGreaterEqual(stats.get("awr/info_missing_lifted", 0.0), 1.0)

    def test_future_improvement_from_d_hat_window(self):
        cfg = EliteAWRConfig(quality_min=0.0, min_loss_steps_per_entry=2, collect_after_frames=0)
        T, N = 6, 1
        d_hat = torch.tensor([0.5, 0.4, 0.3, 0.2, 0.1, 0.05], dtype=torch.float32).view(T, N, 1)
        td = {
            "obses": torch.randn(T, N, 3),
            "actions": torch.randn(T, N, 2),
            "dones": torch.zeros(T, N, 1, dtype=torch.bool),
            "aux_d_hat": d_hat,
            "aux_success_tolerance": torch.full((T, N, 1), 0.05),
            "aux_lifted": torch.ones(T, N, 1, dtype=torch.bool),
        }
        td["dones"][-1, 0, 0] = True
        entries, _ = build_awr_entries_from_experience_buffer(
            experience_buffer=_FakeBuffer(td),
            cfg=cfg,
            source_policy=1,
            source_profile="x",
            source_nsucc=None,
            source_retention=None,
            env_frames=0,
        )
        self.assertGreater(len(entries), 0)
        self.assertGreater(entries[0].future_improvement, 0.0)

    def test_source_nsucc_missing_neutral(self):
        cfg = EliteAWRConfig(quality_min=0.0, min_loss_steps_per_entry=2, collect_after_frames=0)
        _, stats = build_awr_entries_from_experience_buffer(
            experience_buffer=_make_rollout(),
            cfg=cfg,
            source_policy=1,
            source_profile="x",
            source_nsucc=None,
            source_retention=None,
            env_frames=0,
        )
        self.assertEqual(stats.get("awr/source_nsucc_missing_frac", 0.0), 1.0)

    def test_awr_active_profiles(self):
        for profile, active in (
            ("ppo_only", False),
            ("wm_pbt_source_selector_v2", False),
            ("elite_bank_actor_distill_fast_awr", True),
            ("wm_pbt_source_selector_awr", True),
            ("elite_bank_actor_distill_awr", True),
            ("elite_bank_actor_distill_strong_awr", True),
        ):
            ns = argparse.Namespace(wm_strategy_profile=profile)
            apply_wm_strategy_profile(ns, authoritative=True)
            self.assertEqual(awr_active_from_args(ns), active, profile)

    def test_rollout_mode_builder_gate_after_collect(self):
        cfg = EliteAWRConfig(
            enable=True,
            loss_weight=0.001,
            collect_after_frames=100,
            collection_mode="rollout_buffer",
        )
        bridge = EliteAWRBridge(buffer=EliteAWRBuffer(cfg), cfg=cfg)
        self.assertFalse(should_collect_awr(50, cfg, bridge))
        self.assertTrue(should_collect_awr(150, cfg, bridge))

    def test_sidecar_saves_rollout_built_entries(self):
        cfg = EliteAWRConfig(copy_top_k_steps=50, quality_min=0.0, min_loss_steps_per_entry=2, collect_after_frames=0)
        entries, _ = build_awr_entries_from_experience_buffer(
            experience_buffer=_make_rollout(),
            cfg=cfg,
            source_policy=4,
            source_profile="elite_bank_actor_distill_awr",
            source_nsucc=30.0,
            source_retention=0.8,
            env_frames=1000,
        )
        buf = EliteAWRBuffer(cfg, policy_idx=4)
        for entry in entries:
            buf.insert(entry)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "policy_04_awr_topk.pt"
            ok, meta = save_awr_sidecar_topk(buf, path, step_budget=50)
            self.assertTrue(ok)
            buf2 = EliteAWRBuffer(cfg, policy_idx=4)
            loaded = load_awr_sidecar(buf2, path, source_type="local")
            self.assertGreater(loaded, 0)

    def test_out_of_bounds_frac_does_not_disable_awr(self):
        cfg = EliteAWRConfig()
        bridge = EliteAWRBridge(buffer=EliteAWRBuffer(cfg), cfg=cfg)
        td = {"actions": torch.tensor([[[3.0, -2.5]]]), "mus": torch.zeros(1, 1, 2)}
        check_rollout_buffer_action_sanity(bridge, td)
        self.assertFalse(bridge.disabled)
        self.assertEqual(bridge.stats["awr/action_convention_mismatch"], 0.0)
        self.assertGreater(bridge.stats["awr/action_out_of_bounds_frac_1"], 0.0)
        self.assertEqual(bridge.stats["awr/action_source_code"], float(PPO_BUFFER_ACTIONS))

    def test_pbt_restore_large_frame_gap_keeps_target_progress(self):
        should_reset, gap = pbt_should_keep_local_progress_counters(
            source_frame=1_980_000_000,
            target_frame=200_000_000,
            reset_progress_counters=True,
            frame_gap_reset_threshold=500_000_000,
        )
        self.assertTrue(should_reset)
        self.assertGreater(gap, 500_000_000)
        agent_frame = 200_000_000
        agent_epoch = 500
        if should_reset:
            restored_frame = agent_frame
            restored_epoch = agent_epoch
        else:
            restored_frame = 1_980_000_000
            restored_epoch = 12000
        self.assertEqual(restored_frame, 200_000_000)
        self.assertEqual(restored_epoch, 500)

    def test_build_config_includes_collection_fields(self):
        ns = argparse.Namespace(
            awr_enable=True,
            awr_loss_weight=0.001,
            awr_collection_mode="rollout_buffer",
            awr_collect_after_frames=123,
        )
        cfg = build_elite_awr_config_from_args(ns)
        self.assertEqual(cfg.collection_mode, "rollout_buffer")
        self.assertEqual(cfg.collect_after_frames, 123)


if __name__ == "__main__":
    unittest.main()
