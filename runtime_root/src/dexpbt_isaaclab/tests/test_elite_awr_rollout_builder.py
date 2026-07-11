#!/usr/bin/env python3
"""Unit tests for v17.1 rollout-buffer Elite-AWR builder."""

from __future__ import annotations

import argparse
import os
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

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
    AWR_BUILDER_MAX_ENTRIES_PER_ROLLOUT,
    AWR_BUILDER_SELECTED_CAP_REASON_ENTRIES,
    PPO_BUFFER_ACTIONS,
    _enumerate_fragments,
    _enumerate_fragments_batched,
    _fragment_success_signal,
    build_awr_entries_from_experience_buffer,
    build_awr_entries_from_experience_buffer_reference,
    build_awr_entries_from_experience_buffer_vectorized,
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

    def test_pbt_progress_reset_only_when_frame_gap_exceeds_threshold(self):
        keep, gap = pbt_should_keep_local_progress_counters(
            source_frame=1_980_000_000,
            target_frame=200_000_000,
            enable_frame_gap_reset=True,
            frame_gap_reset_threshold=500_000_000,
        )
        self.assertTrue(keep)
        self.assertGreater(gap, 500_000_000)

        keep_small, gap_small = pbt_should_keep_local_progress_counters(
            source_frame=220_000_000,
            target_frame=200_000_000,
            enable_frame_gap_reset=True,
            frame_gap_reset_threshold=500_000_000,
        )
        self.assertFalse(keep_small)
        self.assertLess(gap_small, 500_000_000)

    def test_pbt_progress_reset_disabled_flag_inherits_progress(self):
        keep, _ = pbt_should_keep_local_progress_counters(
            source_frame=1_980_000_000,
            target_frame=200_000_000,
            enable_frame_gap_reset=False,
            frame_gap_reset_threshold=500_000_000,
        )
        self.assertFalse(keep)

    def test_pbt_progress_reset_missing_source_frame_inherits_progress(self):
        keep, gap = pbt_should_keep_local_progress_counters(
            source_frame=None,
            target_frame=200_000_000,
            enable_frame_gap_reset=True,
            frame_gap_reset_threshold=500_000_000,
        )
        self.assertFalse(keep)
        self.assertEqual(gap, 0)

    def test_rollout_buffer_default_min_loss_steps_is_4(self):
        ns = argparse.Namespace(
            awr_enable=True,
            awr_loss_weight=0.001,
            awr_collection_mode="rollout_buffer",
            awr_min_loss_steps_per_entry=None,
        )
        cfg = build_elite_awr_config_from_args(ns)
        self.assertEqual(cfg.min_loss_steps_per_entry, 4)

    def test_interceptor_default_min_loss_steps_remains_16(self):
        ns = argparse.Namespace(
            awr_enable=True,
            awr_loss_weight=0.001,
            awr_collection_mode="interceptor",
            awr_min_loss_steps_per_entry=None,
        )
        cfg = build_elite_awr_config_from_args(ns)
        self.assertEqual(cfg.min_loss_steps_per_entry, 16)

    def test_rollout_builder_accepts_short_fragment_with_4_loss_steps(self):
        T, N = 16, 1
        cfg = EliteAWRConfig(
            quality_min=0.0,
            min_loss_steps_per_entry=4,
            min_burnin_steps=32,
            collect_after_frames=0,
        )
        lifted = torch.zeros(T, N, 1, dtype=torch.bool)
        lifted[-4:, 0, 0] = True
        success_cnt = torch.zeros(T, N, 1)
        success_cnt[-1, 0, 0] = 1.0
        td = {
            "obses": torch.randn(T, N, 3),
            "actions": torch.randn(T, N, 2),
            "dones": torch.zeros(T, N, 1, dtype=torch.bool),
            "aux_success_tolerance": torch.full((T, N, 1), 0.05),
            "aux_lifted": lifted,
            "aux_failure": torch.zeros(T, N, 1, dtype=torch.bool),
            "aux_success_cnt": success_cnt,
        }
        td["dones"][-1, 0, 0] = True
        entries, stats = build_awr_entries_from_experience_buffer(
            experience_buffer=_FakeBuffer(td),
            cfg=cfg,
            source_policy=1,
            source_profile="x",
            source_nsucc=None,
            source_retention=None,
            env_frames=0,
        )
        self.assertGreater(len(entries), 0)
        self.assertGreaterEqual(stats["awr/rollout_builder_selected_fragments"], 1.0)
        self.assertGreaterEqual(int(entries[0].loss_mask.sum()), 4)

    def test_v17_1_launchers_use_min_loss_steps_4(self):
        repo = Path(__file__).resolve().parents[4]
        for name in (
            "start_regrasping_pbt_wm_strategy_zoo_v17_1_online_elite_awr_smoke.sh",
            "start_regrasping_pbt_wm_strategy_zoo_v17_1_online_elite_awr_2b.sh",
        ):
            text = (repo / "results/launchers" / name).read_text()
            self.assertIn("awr_min_loss_steps_per_entry=4", text)
            self.assertIn("--awr_min_loss_steps_per_entry 4", text)

    def test_success_cnt_success_not_cleared_by_unknown_done(self):
        T, N = 8, 1
        success_cnt = torch.zeros(T, N, 1)
        success_cnt[-1, 0, 0] = 1.0
        td = {
            "obses": torch.randn(T, N, 3),
            "actions": torch.randn(T, N, 2),
            "dones": torch.zeros(T, N, 1, dtype=torch.bool),
            "aux_success_tolerance": torch.full((T, N, 1), 0.05),
            "aux_lifted": torch.ones(T, N, 1, dtype=torch.bool),
            "aux_failure": torch.zeros(T, N, 1, dtype=torch.bool),
            "aux_success_cnt": success_cnt,
        }
        td["dones"][-1, 0, 0] = True
        cfg = EliteAWRConfig(quality_min=0.0, min_loss_steps_per_entry=4, collect_after_frames=0)
        _, stats = build_awr_entries_from_experience_buffer(
            experience_buffer=_FakeBuffer(td),
            cfg=cfg,
            source_policy=1,
            source_profile="x",
            source_nsucc=None,
            source_retention=None,
            env_frames=0,
        )
        self.assertGreaterEqual(stats.get("awr/success_from_success_cnt", 0.0), 1.0)
        self.assertEqual(stats.get("awr/success_cleared_by_unknown_done", -1.0), 0.0)

    def test_unknown_done_without_success_remains_non_success(self):
        future_success, future_within_100 = _fragment_success_signal(
            torch.zeros(4, 1),
            0,
            3,
            0,
            terminal_success=False,
        )
        self.assertFalse(future_success)
        self.assertFalse(future_within_100)

    def test_fragment_success_signal_uses_success_count_delta(self):
        success_cnt = torch.zeros(4, 1)
        success_cnt[3, 0] = 1.0
        future_success, future_within_100 = _fragment_success_signal(
            success_cnt,
            0,
            3,
            0,
            terminal_success=False,
        )
        self.assertTrue(future_success)
        self.assertTrue(future_within_100)

        flat_cnt = torch.ones(4, 1)
        fs2, fw2 = _fragment_success_signal(
            flat_cnt,
            0,
            3,
            0,
            terminal_success=False,
        )
        self.assertFalse(fs2)
        self.assertFalse(fw2)

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


def _fragment_triplets_equal(
    a: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    b: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> bool:
    def _sort(n, t0, t1):
        if n.numel() == 0:
            return n, t0, t1
        key = n.to(torch.long) * 1_000_000 + t0.to(torch.long) * 1_000 + t1.to(torch.long)
        order = torch.argsort(key)
        return n[order], t0[order], t1[order]

    an, at0, at1 = _sort(*a)
    bn, bt0, bt1 = _sort(*b)
    return (
        torch.equal(an, bn)
        and torch.equal(at0, bt0)
        and torch.equal(at1, bt1)
    )


def _entry_signature(entry) -> tuple:
    return (
        entry.stored_seq_len,
        int(entry.loss_mask.sum()),
        bool(entry.future_success_within_100),
        bool(entry.had_drop),
        bool(entry.no_drop),
        round(entry.episode_quality, 4),
        round(float(entry.action.abs().sum()), 4),
    )


def _entry_signature_soft(entry) -> tuple:
    """Semantic signature without action tensor (burn-in diffs vs reference)."""
    return _entry_signature(entry)[:-1]


class TestBatchedFragmentEnumeration(unittest.TestCase):
    def _assert_enum_match(self, dones_bool: torch.Tensor, *, all_fragments: bool = True) -> None:
        ref = _enumerate_fragments(dones_bool, all_fragments=all_fragments)
        bat = _enumerate_fragments_batched(dones_bool, all_fragments=all_fragments)
        self.assertTrue(
            _fragment_triplets_equal(ref, bat),
            msg=f"ref={tuple(x.tolist() for x in ref)} bat={tuple(x.tolist() for x in bat)}",
        )

    def test_batched_matches_reference_random_multi_done(self):
        torch.manual_seed(17)
        T, N = 16, 32
        dones = torch.rand(T, N) < 0.12
        dones[-1, :] = True
        self._assert_enum_match(dones)

    def test_batched_matches_reference_no_done(self):
        T, N = 16, 32
        dones = torch.zeros(T, N, dtype=torch.bool)
        self._assert_enum_match(dones)

    def test_batched_matches_reference_done_at_final_step(self):
        T, N = 16, 32
        dones = torch.zeros(T, N, dtype=torch.bool)
        dones[-1, :] = True
        self._assert_enum_match(dones)

    def test_final_fragment_only_semantics(self):
        """Bugfix: final-only mode returns [last_done+1, T-1], not [0, T-1]."""
        T, N = 8, 3
        dones = torch.zeros(T, N, dtype=torch.bool)
        dones[3, 0] = True
        dones[7, 2] = True
        frag_n, frag_t0, frag_t1 = _enumerate_fragments_batched(dones, all_fragments=False)
        expected = {
            0: (4, 7),
            1: (0, 7),
            2: (7, 7),
        }
        for n in range(N):
            idx = int((frag_n == n).nonzero(as_tuple=True)[0].item())
            self.assertEqual(int(frag_t0[idx].item()), expected[n][0], msg=f"env{n} t0")
            self.assertEqual(int(frag_t1[idx].item()), expected[n][1], msg=f"env{n} t1")


class TestVectorizedRolloutBuilder(unittest.TestCase):
    def _cfg(self, **kwargs) -> EliteAWRConfig:
        base = dict(
            quality_min=0.0,
            min_loss_steps_per_entry=2,
            min_burnin_steps=1,
            collect_after_frames=0,
        )
        base.update(kwargs)
        return EliteAWRConfig(**base)

    def _rich_rollout(self, T: int = 16, N: int = 8) -> _FakeBuffer:
        act_dim = 2
        obs_dim = 6
        actions = torch.randn(T, N, act_dim) * 2.5
        dones = torch.zeros(T, N, 1, dtype=torch.bool)
        dones[T // 2, 0, 0] = True
        dones[T - 1, :, 0] = True
        success_cnt = torch.zeros(T, N, 1)
        success_cnt[T // 2, 0, 0] = 1.0
        success_cnt[T - 1, 1, 0] = 2.0
        success_cnt[0, 1, 0] = 1.0
        return _FakeBuffer(
            {
                "obses": torch.randn(T, N, obs_dim),
                "actions": actions,
                "dones": dones,
                "aux_d_hat": torch.linspace(0.3, 0.05, T).view(T, 1, 1).expand(T, N, 1),
                "aux_success_tolerance": torch.full((T, N, 1), 0.05),
                "aux_lifted": torch.ones(T, N, 1, dtype=torch.bool),
                "aux_failure": torch.zeros(T, N, 1, dtype=torch.bool),
                "aux_success_cnt": success_cnt,
            }
        )

    def test_reference_vs_vectorized_equivalence_smoke(self):
        cfg = self._cfg(min_loss_steps_per_entry=4, min_burnin_steps=0)
        exp = self._rich_rollout(T=16, N=8)
        kwargs = dict(
            experience_buffer=exp,
            cfg=cfg,
            source_policy=1,
            source_profile="elite_bank_actor_distill_awr",
            source_nsucc=30.0,
            source_retention=0.8,
            env_frames=600_000_000,
        )
        ref_entries, ref_stats = build_awr_entries_from_experience_buffer_reference(**kwargs)
        vec_entries, vec_stats = build_awr_entries_from_experience_buffer_vectorized(**kwargs)

        self.assertEqual(
            ref_stats["awr/rollout_builder_candidate_fragments"],
            vec_stats["awr/rollout_builder_candidate_fragments"],
        )
        self.assertGreater(len(vec_entries), 0)
        ref_steps = float(ref_stats["awr/rollout_builder_selected_steps"])
        vec_steps = float(vec_stats["awr/rollout_builder_selected_steps"])
        if ref_steps > 0:
            self.assertLessEqual(abs(vec_steps - ref_steps) / ref_steps, 0.10)
        self.assertEqual(vec_stats.get("awr/success_cleared_by_unknown_done", -1.0), 0.0)
        self.assertEqual(vec_stats.get("awr/done_reason_classification_disabled", 0.0), 1.0)
        self.assertEqual(vec_stats.get("awr/success_signal_source_code", 0.0), 1.0)

        ref_sigs = sorted(_entry_signature_soft(e) for e in ref_entries)
        vec_sigs = sorted(_entry_signature_soft(e) for e in vec_entries)
        overlap = len(set(ref_sigs) & set(vec_sigs))
        self.assertGreaterEqual(overlap, min(len(ref_sigs), len(vec_sigs)) // 2)

    def test_reference_vs_vectorized_signature_multiset(self):
        cfg = self._cfg(min_loss_steps_per_entry=4, min_burnin_steps=0)
        exp = self._rich_rollout(T=16, N=8)
        kwargs = dict(
            experience_buffer=exp,
            cfg=cfg,
            source_policy=1,
            source_profile="elite_bank_actor_distill_awr",
            source_nsucc=30.0,
            source_retention=0.8,
            env_frames=600_000_000,
        )
        ref_entries, _ = build_awr_entries_from_experience_buffer_reference(**kwargs)
        vec_entries, vec_stats = build_awr_entries_from_experience_buffer_vectorized(**kwargs)
        self.assertGreater(len(vec_entries), 0)
        ref_multiset = sorted(_entry_signature_soft(e) for e in ref_entries)
        vec_multiset = sorted(_entry_signature_soft(e) for e in vec_entries)
        self.assertGreaterEqual(len(vec_multiset), 1)
        shared = set(ref_multiset) & set(vec_multiset)
        self.assertGreater(len(shared), 0)
        self.assertEqual(vec_stats["awr/rollout_builder_selected_fragments_before_cap"], vec_stats["awr/rollout_builder_selected_fragments_after_cap"])

    def test_loss_mask_no_burn_for_short_non_truncated_fragment(self):
        T, N = 16, 1
        cfg = EliteAWRConfig(
            quality_min=0.0,
            min_loss_steps_per_entry=4,
            min_burnin_steps=4,
            max_episode_len=200,
            collect_after_frames=0,
        )
        td = {
            "obses": torch.randn(T, N, 3),
            "actions": torch.randn(T, N, 2),
            "dones": torch.zeros(T, N, 1, dtype=torch.bool),
            "aux_success_tolerance": torch.full((T, N, 1), 0.05),
            "aux_lifted": torch.ones(T, N, 1, dtype=torch.bool),
            "aux_failure": torch.zeros(T, N, 1, dtype=torch.bool),
            "aux_success_cnt": torch.zeros(T, N, 1),
        }
        td["dones"][-1, 0, 0] = True
        vec_entries, _ = build_awr_entries_from_experience_buffer_vectorized(
            experience_buffer=_FakeBuffer(td),
            cfg=cfg,
            source_policy=1,
            source_profile="x",
            source_nsucc=None,
            source_retention=None,
            env_frames=0,
        )
        ref_entries, _ = build_awr_entries_from_experience_buffer_reference(
            experience_buffer=_FakeBuffer(td),
            cfg=cfg,
            source_policy=1,
            source_profile="x",
            source_nsucc=None,
            source_retention=None,
            env_frames=0,
        )
        self.assertGreater(len(vec_entries), 0)
        self.assertFalse(vec_entries[0].truncated_prefix)
        self.assertEqual(vec_entries[0].loss_start_idx, 0)
        self.assertEqual(int(vec_entries[0].loss_mask.sum()), T)
        self.assertEqual(int(ref_entries[0].loss_mask.sum()), T - cfg.min_burnin_steps)

    def test_loss_mask_truncated_tail_with_burn_in(self):
        T, N = 16, 1
        cfg = EliteAWRConfig(
            quality_min=0.0,
            min_loss_steps_per_entry=4,
            min_burnin_steps=4,
            max_episode_len=8,
            collect_after_frames=0,
        )
        td = {
            "obses": torch.randn(T, N, 3),
            "actions": torch.randn(T, N, 2),
            "dones": torch.zeros(T, N, 1, dtype=torch.bool),
            "aux_success_tolerance": torch.full((T, N, 1), 0.05),
            "aux_lifted": torch.ones(T, N, 1, dtype=torch.bool),
            "aux_failure": torch.zeros(T, N, 1, dtype=torch.bool),
            "aux_success_cnt": torch.zeros(T, N, 1),
        }
        td["dones"][-1, 0, 0] = True
        kwargs = dict(
            experience_buffer=_FakeBuffer(td),
            cfg=cfg,
            source_policy=1,
            source_profile="x",
            source_nsucc=None,
            source_retention=None,
            env_frames=0,
        )
        ref_entries, _ = build_awr_entries_from_experience_buffer_reference(**kwargs)
        vec_entries, _ = build_awr_entries_from_experience_buffer_vectorized(**kwargs)
        self.assertGreater(len(vec_entries), 0)
        ve = vec_entries[0]
        re = ref_entries[0]
        self.assertEqual(ve.stored_seq_len, 8)
        self.assertTrue(ve.truncated_prefix)
        self.assertEqual(ve.loss_start_idx, cfg.min_burnin_steps)
        self.assertEqual(int(ve.loss_mask.sum()), int(re.loss_mask.sum()))
        self.assertEqual(ve.stored_seq_len, re.stored_seq_len)

    def test_failure_rejects_without_success_cnt_delta(self):
        T, N = 16, 1
        failure = torch.zeros(T, N, 1, dtype=torch.bool)
        failure[5:, 0, 0] = True
        td = {
            "obses": torch.randn(T, N, 3),
            "actions": torch.randn(T, N, 2),
            "dones": torch.zeros(T, N, 1, dtype=torch.bool),
            "aux_success_tolerance": torch.full((T, N, 1), 0.05),
            "aux_lifted": torch.ones(T, N, 1, dtype=torch.bool),
            "aux_failure": failure,
            "aux_success_cnt": torch.zeros(T, N, 1),
        }
        td["dones"][-1, 0, 0] = True
        cfg = self._cfg(min_loss_steps_per_entry=4, quality_min=0.0)
        entries, _ = build_awr_entries_from_experience_buffer_vectorized(
            experience_buffer=_FakeBuffer(td),
            cfg=cfg,
            source_policy=1,
            source_profile="x",
            source_nsucc=30.0,
            source_retention=None,
            env_frames=0,
        )
        self.assertEqual(len(entries), 0)

    def test_failure_kept_when_success_cnt_delta_true(self):
        T, N = 16, 1
        success_cnt = torch.zeros(T, N, 1)
        success_cnt[-1, 0, 0] = 1.0
        failure = torch.zeros(T, N, 1, dtype=torch.bool)
        failure[10, 0, 0] = True
        td = {
            "obses": torch.randn(T, N, 3),
            "actions": torch.randn(T, N, 2),
            "dones": torch.zeros(T, N, 1, dtype=torch.bool),
            "aux_success_tolerance": torch.full((T, N, 1), 0.05),
            "aux_lifted": torch.ones(T, N, 1, dtype=torch.bool),
            "aux_failure": failure,
            "aux_success_cnt": success_cnt,
        }
        td["dones"][-1, 0, 0] = True
        cfg = self._cfg(min_loss_steps_per_entry=4, quality_min=0.0)
        entries, stats = build_awr_entries_from_experience_buffer_vectorized(
            experience_buffer=_FakeBuffer(td),
            cfg=cfg,
            source_policy=1,
            source_profile="x",
            source_nsucc=None,
            source_retention=None,
            env_frames=0,
        )
        self.assertGreater(len(entries), 0)
        self.assertTrue(entries[0].had_drop)
        self.assertTrue(entries[0].future_success_within_100)
        self.assertGreaterEqual(stats["awr/success_from_success_cnt"], 1.0)

    def test_selected_cap_limits_materialize(self):
        T, N = 16, 4096
        cfg = self._cfg(min_loss_steps_per_entry=2, quality_min=0.0)
        exp = self._rich_rollout(T=T, N=N)
        with mock.patch(
            "dexpbt_isaaclab.awr.elite_awr_rollout_builder.AWR_BUILDER_MAX_ENTRIES_PER_ROLLOUT",
            32,
        ), mock.patch(
            "dexpbt_isaaclab.awr.elite_awr_rollout_builder.AWR_BUILDER_MAX_SELECTED_STEPS_PER_ROLLOUT",
            1_000_000,
        ):
            _, stats = build_awr_entries_from_experience_buffer_vectorized(
                experience_buffer=exp,
                cfg=cfg,
                source_policy=1,
                source_profile="x",
                source_nsucc=30.0,
                source_retention=0.8,
                env_frames=0,
            )
        self.assertGreater(stats["awr/rollout_builder_selected_fragments_before_cap"], 32.0)
        self.assertEqual(stats["awr/rollout_builder_selected_fragments_after_cap"], 32.0)
        self.assertEqual(stats["awr/rollout_builder_selected_cap_applied"], 1.0)
        self.assertEqual(
            stats["awr/rollout_builder_selected_cap_reason_code"],
            float(AWR_BUILDER_SELECTED_CAP_REASON_ENTRIES),
        )

    def test_vectorized_done_and_success_stats(self):
        cfg = self._cfg(min_loss_steps_per_entry=4)
        _, stats = build_awr_entries_from_experience_buffer_vectorized(
            experience_buffer=self._rich_rollout(T=16, N=4),
            cfg=cfg,
            source_policy=1,
            source_profile="x",
            source_nsucc=30.0,
            source_retention=None,
            env_frames=0,
        )
        self.assertEqual(stats["awr/done_reason_classification_disabled"], 1.0)
        self.assertEqual(stats["awr/success_signal_source_code"], 1.0)
        self.assertIn("awr/done_fragment_frac", stats)
        self.assertEqual(stats["awr/success_cleared_by_unknown_done"], 0.0)

    def test_vectorized_missing_success_cnt_sets_stats(self):
        cfg = self._cfg()
        td = {
            "obses": torch.randn(4, 1, 3),
            "actions": torch.randn(4, 1, 2),
            "dones": torch.zeros(4, 1, 1, dtype=torch.bool),
            "aux_success_tolerance": torch.full((4, 1, 1), 0.05),
            "aux_lifted": torch.ones(4, 1, 1, dtype=torch.bool),
        }
        td["dones"][-1, 0, 0] = True
        _, stats = build_awr_entries_from_experience_buffer_vectorized(
            experience_buffer=_FakeBuffer(td),
            cfg=cfg,
            source_policy=1,
            source_profile="x",
            source_nsucc=None,
            source_retention=None,
            env_frames=0,
        )
        self.assertEqual(stats.get("awr/success_signal_missing", 0.0), 1.0)
        self.assertEqual(stats.get("awr/info_missing_success", 0.0), 1.0)

    def test_vectorized_success_cnt_delta_preserved(self):
        T, N = 16, 1
        success_cnt = torch.zeros(T, N, 1)
        success_cnt[-1, 0, 0] = 1.0
        td = {
            "obses": torch.randn(T, N, 3),
            "actions": torch.randn(T, N, 2),
            "dones": torch.zeros(T, N, 1, dtype=torch.bool),
            "aux_success_tolerance": torch.full((T, N, 1), 0.05),
            "aux_lifted": torch.ones(T, N, 1, dtype=torch.bool),
            "aux_failure": torch.zeros(T, N, 1, dtype=torch.bool),
            "aux_success_cnt": success_cnt,
        }
        td["dones"][-1, 0, 0] = True
        cfg = self._cfg(min_loss_steps_per_entry=4)
        _, stats = build_awr_entries_from_experience_buffer_vectorized(
            experience_buffer=_FakeBuffer(td),
            cfg=cfg,
            source_policy=1,
            source_profile="x",
            source_nsucc=None,
            source_retention=None,
            env_frames=0,
        )
        self.assertGreaterEqual(stats.get("awr/success_from_success_cnt", 0.0), 1.0)
        self.assertEqual(stats.get("awr/success_cleared_by_unknown_done", -1.0), 0.0)

    def test_vectorized_missing_aux_neutral(self):
        cfg = self._cfg()
        td = {
            "obses": torch.randn(4, 2, 3),
            "actions": torch.randn(4, 2, 2) * 2.0,
            "dones": torch.zeros(4, 2, 1, dtype=torch.bool),
        }
        td["dones"][3, 0, 0] = True
        entries, stats = build_awr_entries_from_experience_buffer_vectorized(
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
        self.assertEqual(stats.get("awr/rollout_builder_mode_code", 0.0), 1.0)

    def test_builder_mode_env_reference(self):
        cfg = self._cfg()
        exp = _make_rollout(T=8, N=2)
        with mock.patch.dict(os.environ, {"DEXPBT_AWR_BUILDER_MODE": "reference"}):
            _, stats = build_awr_entries_from_experience_buffer(
                experience_buffer=exp,
                cfg=cfg,
                source_policy=1,
                source_profile="x",
                source_nsucc=None,
                source_retention=None,
                env_frames=0,
            )
        self.assertEqual(stats.get("awr/rollout_builder_mode_code", -1.0), 0.0)

    def test_builder_mode_env_vectorized(self):
        cfg = self._cfg()
        exp = _make_rollout(T=8, N=2)
        with mock.patch.dict(os.environ, {"DEXPBT_AWR_BUILDER_MODE": "vectorized"}):
            _, stats = build_awr_entries_from_experience_buffer(
                experience_buffer=exp,
                cfg=cfg,
                source_policy=1,
                source_profile="x",
                source_nsucc=None,
                source_retention=None,
                env_frames=0,
            )
        self.assertEqual(stats.get("awr/rollout_builder_mode_code", -1.0), 1.0)
        self.assertEqual(stats.get("awr/rollout_builder_vectorized_speed_path", 0.0), 1.0)

    def test_vectorized_exception_falls_back_to_reference(self):
        cfg = self._cfg()
        exp = _make_rollout(T=8, N=2)
        with mock.patch.dict(os.environ, {"DEXPBT_AWR_BUILDER_MODE": "vectorized"}):
            with mock.patch(
                "dexpbt_isaaclab.awr.elite_awr_rollout_builder.build_awr_entries_from_experience_buffer_vectorized",
                side_effect=RuntimeError("boom"),
            ):
                entries, stats = build_awr_entries_from_experience_buffer(
                    experience_buffer=exp,
                    cfg=cfg,
                    source_policy=1,
                    source_profile="x",
                    source_nsucc=None,
                    source_retention=None,
                    env_frames=0,
                )
        self.assertIsInstance(entries, list)
        self.assertEqual(stats.get("awr/rollout_builder_vectorized_failed", 0.0), 1.0)
        self.assertEqual(stats.get("awr/rollout_builder_fallback_reference", 0.0), 1.0)

    def test_vectorized_benchmark_faster_than_reference(self):
        cfg = self._cfg(min_loss_steps_per_entry=4)
        exp = self._rich_rollout(T=16, N=1024)
        kwargs = dict(
            experience_buffer=exp,
            cfg=cfg,
            source_policy=1,
            source_profile="x",
            source_nsucc=30.0,
            source_retention=0.8,
            env_frames=600_000_000,
        )
        t0 = time.perf_counter()
        build_awr_entries_from_experience_buffer_reference(**kwargs)
        ref_ms = (time.perf_counter() - t0) * 1000.0
        t1 = time.perf_counter()
        build_awr_entries_from_experience_buffer_vectorized(**kwargs)
        vec_ms = (time.perf_counter() - t1) * 1000.0
        print(f"[benchmark] reference={ref_ms:.2f}ms vectorized={vec_ms:.2f}ms ratio={ref_ms/max(vec_ms,1e-6):.1f}x")
        self.assertLess(vec_ms, ref_ms)

    def test_vectorized_benchmark_n8192_print_only(self):
        cfg = self._cfg(min_loss_steps_per_entry=4, quality_min=99.0)
        exp = self._rich_rollout(T=16, N=8192)
        kwargs = dict(
            experience_buffer=exp,
            cfg=cfg,
            source_policy=1,
            source_profile="x",
            source_nsucc=30.0,
            source_retention=0.8,
            env_frames=600_000_000,
        )
        t0 = time.perf_counter()
        build_awr_entries_from_experience_buffer_reference(**kwargs)
        ref_ms = (time.perf_counter() - t0) * 1000.0
        t1 = time.perf_counter()
        _, vec_stats = build_awr_entries_from_experience_buffer_vectorized(**kwargs)
        vec_ms = (time.perf_counter() - t1) * 1000.0
        print(
            f"[benchmark N=8192] reference={ref_ms:.2f}ms vectorized={vec_ms:.2f}ms "
            f"pre={vec_stats.get('awr/rollout_builder_precompute_ms', 0):.2f}ms "
            f"ratio={ref_ms/max(vec_ms,1e-6):.1f}x"
        )


class TestAwrBuilderFuseImports(unittest.TestCase):
    def test_train_lazy_import_builder_fuse_helpers(self):
        from dexpbt_isaaclab.awr.elite_awr import (
            awr_apply_builder_run,
            awr_builder_collection_paused,
            awr_record_builder_fuse_skip,
        )

        self.assertTrue(callable(awr_apply_builder_run))
        self.assertTrue(callable(awr_builder_collection_paused))
        self.assertTrue(callable(awr_record_builder_fuse_skip))

    def test_awr_package_exports_builder_fuse_helpers(self):
        import dexpbt_isaaclab.awr as awr_pkg

        for name in (
            "awr_apply_builder_run",
            "awr_builder_collection_paused",
            "awr_record_builder_fuse_skip",
            "AWR_BUILDER_FUSE_SLOW_MS",
            "AWR_BUILDER_FUSE_PAUSE_FRAMES",
            "AWR_BUILDER_SKIP_REASON_FUSE",
        ):
            self.assertTrue(hasattr(awr_pkg, name), msg=name)

    def test_record_builder_fuse_skip_never_raises(self):
        from dexpbt_isaaclab.awr.elite_awr import awr_record_builder_fuse_skip

        awr_record_builder_fuse_skip(SimpleNamespace(stats=None), 600_000_000)
        bridge = SimpleNamespace(stats={"awr/builder_fuse_active": 0.0})
        awr_record_builder_fuse_skip(bridge, 600_000_000)
        self.assertEqual(bridge.stats["awr/builder_skipped"], 1.0)


if __name__ == "__main__":
    unittest.main()
