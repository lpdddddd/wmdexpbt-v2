"""Unit tests for Strategy Zoo v1.7 Online Elite-AWR module."""

from __future__ import annotations

import argparse
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch

from dexpbt_isaaclab.awr.elite_awr import (
    EliteAWRBridge,
    EliteAWRBuffer,
    EliteAWRConfig,
    EliteAWREntry,
    awr_active_from_args,
    build_elite_awr_config_from_args,
    build_sequence_entry_from_episode,
    compute_elite_awr_quality,
    load_awr_sidecar,
    patch_model_get_aux_loss_for_awr,
    save_awr_sidecar,
)
from dexpbt_isaaclab.awr.elite_awr_interceptor import DoneReason, classify_done_reason
from dexpbt_isaaclab.world_model.wm_strategy_profiles import (
    PROFILE_IDS,
    PROFILE_TABLES,
    apply_wm_strategy_profile,
    profile_capability_flags,
)


def _pending(episode_len: int, *, quality_from: int = 0) -> list[dict]:
    steps = []
    for i in range(episode_len):
        steps.append(
            {
                "obs": torch.randn(4),
                "action": torch.randn(2),
                "lifted": i >= quality_from,
                "drop": False,
                "tolerance": 0.05,
                "quality_step_pass": i >= quality_from,
                "future_success_within_100": i >= max(0, episode_len - 10),
                "future_improvement": 0.1,
                "info_missing": {},
            }
        )
    return steps


def _make_entry(stored_len: int, *, truncated: bool, starts_at_reset: bool) -> EliteAWREntry:
    cfg = EliteAWRConfig(min_loss_steps_per_entry=4, min_burnin_steps=8, quality_min=0.0)
    pending = _pending(stored_len if starts_at_reset else stored_len + 50, quality_from=stored_len // 4)
    if truncated:
        pending = pending[-stored_len:]
    entry = build_sequence_entry_from_episode(
        pending,
        cfg=cfg,
        source_policy=1,
        source_profile="elite_bank_actor_distill_awr",
        source_nsucc=30.0,
        source_retention=0.8,
        source_type="local",
        env_frames=1000,
        done_reason=DoneReason.SUCCESS.value,
    )
    assert entry is not None
    return entry


class TestEliteAWR(unittest.TestCase):
    def test_sequence_entry_from_episode_start_bitmatch_flag(self):
        cfg = EliteAWRConfig(max_episode_len=200, min_loss_steps_per_entry=4, quality_min=0.0)
        entry = build_sequence_entry_from_episode(
            _pending(80, quality_from=40),
            cfg=cfg,
            source_policy=1,
            source_profile="x",
            source_nsucc=None,
            source_retention=None,
            source_type="local",
            env_frames=0,
            done_reason=DoneReason.SUCCESS.value,
        )
        self.assertIsNotNone(entry)
        assert entry is not None
        self.assertTrue(entry.starts_at_episode_reset)
        self.assertFalse(entry.truncated_prefix)

    def test_sequence_entry_truncated_prefix(self):
        cfg = EliteAWRConfig(
            max_episode_len=200,
            min_burnin_steps=32,
            min_loss_steps_per_entry=16,
            quality_min=0.0,
        )
        entry = build_sequence_entry_from_episode(
            _pending(300, quality_from=250),
            cfg=cfg,
            source_policy=1,
            source_profile="x",
            source_nsucc=None,
            source_retention=None,
            source_type="local",
            env_frames=0,
            done_reason=DoneReason.SUCCESS.value,
        )
        self.assertIsNotNone(entry)
        assert entry is not None
        self.assertFalse(entry.starts_at_episode_reset)
        self.assertTrue(entry.truncated_prefix)
        burn_end = min(cfg.min_burnin_steps, entry.stored_seq_len)
        self.assertFalse(entry.loss_mask[:burn_end].any())
        self.assertGreaterEqual(int(entry.loss_mask.sum()), cfg.min_loss_steps_per_entry)

    def test_awr_batch_steps_not_num_sequences(self):
        cfg = EliteAWRConfig(batch_steps=4096, max_episode_len=200, min_buffer_size=1)
        buf = EliteAWRBuffer(cfg, policy_idx=1)
        for _ in range(30):
            buf.insert(_make_entry(200, truncated=False, starts_at_reset=True))
        batch = buf.sample_batch(torch.device("cpu"))
        self.assertIsNotNone(batch)
        assert batch is not None
        self.assertLessEqual(batch["batch_sequences"], 4096 // 200 + 1)
        self.assertNotEqual(batch["batch_sequences"], 4096)

    def test_awr_sidecar_checkpoint(self):
        cfg = EliteAWRConfig(min_buffer_size=1)
        buf = EliteAWRBuffer(cfg, policy_idx=4)
        buf.insert(_make_entry(32, truncated=False, starts_at_reset=True))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "policy_04_awr_topk.pt"
            ok, meta = save_awr_sidecar(buf, path)
            self.assertTrue(ok)
            self.assertEqual(meta["save_failed"], 0.0)
            main_ckpt = {"awr_buffer_ref": path.name, "awr_buffer_ref_meta": meta}
            self.assertNotIn("awr_buffer_topk", main_ckpt)
            self.assertIn("awr_buffer_ref", main_ckpt)
            buf2 = EliteAWRBuffer(cfg, policy_idx=4)
            loaded = load_awr_sidecar(buf2, path)
            self.assertGreater(loaded, 0)
            self.assertEqual(buf2.num_sequences(), 1)

    def test_awr_sidecar_step_budget(self):
        cfg = EliteAWRConfig(copy_top_k_steps=50, broadcast_top_k_steps=20)
        buf = EliteAWRBuffer(cfg, policy_idx=1)
        for q in (6.0, 5.0, 4.0, 3.0):
            e = _make_entry(20, truncated=False, starts_at_reset=True)
            e.episode_quality = q
            buf.insert(e)
        top = buf.top_entries_by_steps(cfg.copy_top_k_steps)
        steps = sum(x.num_valid_steps for x in top)
        self.assertLessEqual(steps, cfg.copy_top_k_steps)
        btop = buf.top_entries_by_steps(cfg.broadcast_top_k_steps)
        bsteps = sum(x.num_valid_steps for x in btop)
        self.assertLessEqual(bsteps, cfg.broadcast_top_k_steps)

    def test_action_convention_sanity(self):
        cfg = EliteAWRConfig()
        buf = EliteAWRBuffer(cfg, policy_idx=0)
        bridge = EliteAWRBridge(buffer=buf, cfg=cfg)
        a = torch.tensor([[0.1, -0.2], [0.3, 0.4]])
        bridge.maybe_check_action_convention(a, a.clone())
        self.assertFalse(bridge.disabled)
        bridge2 = EliteAWRBridge(buffer=buf, cfg=cfg)
        bridge2.maybe_check_action_convention(a, a + 1.0)
        self.assertTrue(bridge2.disabled)

    def test_aux_loss_chain_preserves_eir(self):
        model = mock.Mock()
        model.get_aux_loss = lambda: {"eir_total": torch.tensor(1.0)}
        bridge = EliteAWRBridge(
            buffer=EliteAWRBuffer(EliteAWRConfig(min_buffer_size=10**9), policy_idx=0),
            cfg=EliteAWRConfig(),
        )
        patch_model_get_aux_loss_for_awr(model, bridge)
        out = model.get_aux_loss()
        self.assertIn("eir_total", out)

    def test_awr_active_requires_loss_weight(self):
        args = argparse.Namespace(awr_enable=True, awr_loss_weight=0.0)
        self.assertFalse(awr_active_from_args(args))
        args.awr_loss_weight = 0.001
        self.assertTrue(awr_active_from_args(args))

    def test_p0_p2_no_awr(self):
        for profile in ("ppo_only", "wm_pbt_source_selector_v2"):
            ns = argparse.Namespace(wm_strategy_profile=profile)
            for dest in {k for t in PROFILE_TABLES.values() for k in t}:
                if not hasattr(ns, dest):
                    setattr(ns, dest, None)
            apply_wm_strategy_profile(ns, authoritative=True)
            self.assertFalse(awr_active_from_args(ns))

    def test_p3_lambda_conservative(self):
        ns = argparse.Namespace(wm_strategy_profile="wm_pbt_source_selector_awr")
        for dest in {k for t in PROFILE_TABLES.values() for k in t}:
            if not hasattr(ns, dest):
                setattr(ns, dest, None)
        apply_wm_strategy_profile(ns, authoritative=True)
        self.assertAlmostEqual(ns.awr_loss_weight, 0.0003)

    def test_loss_cap_hit_frac(self):
        cfg = EliteAWRConfig(loss_cap=0.001, loss_weight=1.0, start_after_frames=0)
        buf = EliteAWRBuffer(cfg, policy_idx=0)
        bridge = EliteAWRBridge(buffer=buf, cfg=cfg)
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
        bridge.agent_ref = agent
        batch = {
            "obs": torch.zeros(4, 2, 4),
            "action": torch.zeros(4, 2, 2),
            "loss_mask": torch.ones(4, 2, dtype=torch.bool),
            "quality": torch.full((4, 2), 3.0),
            "seq_length": 4,
            "batch_sequences": 2,
            "batch_valid_steps": 8,
            "batch_padded_steps": 8,
            "batch_pad_frac": 0.0,
        }
        from dexpbt_isaaclab.awr.elite_awr import compute_elite_awr_loss

        for _ in range(3):
            compute_elite_awr_loss(agent, batch, cfg, frames=agent.frame, bridge=bridge)
        self.assertGreater(bridge._loss_cap_hits, 0)
        self.assertGreater(bridge._loss_cap_hits / bridge._loss_cap_total, 0.0)

    def test_source_nsucc_missing_neutral(self):
        with_nsucc = compute_elite_awr_quality(
            future_success_within_100=False,
            lifted_frac=1.0,
            no_drop=True,
            future_improvement=0.0,
            tolerance_min=0.05,
            source_nsucc=40.0,
            had_drop=False,
        )
        missing = compute_elite_awr_quality(
            future_success_within_100=False,
            lifted_frac=1.0,
            no_drop=True,
            future_improvement=0.0,
            tolerance_min=0.05,
            source_nsucc=None,
            had_drop=False,
        )
        self.assertLess(missing, with_nsucc)
        self.assertGreater(missing, 0.0)

    def test_done_reason_unknown(self):
        reason = classify_done_reason(
            terminated=torch.tensor([True]),
            truncated=torch.tensor([False]),
            env_idx=0,
            infos={},
            env_attrs=None,
        )
        self.assertEqual(reason, DoneReason.UNKNOWN.value)

    def test_v17_awr_profiles_registered(self):
        self.assertEqual(PROFILE_IDS["elite_bank_actor_distill_fast_awr"], 14)
        self.assertEqual(PROFILE_IDS["wm_pbt_source_selector_awr"], 15)
        caps = profile_capability_flags(
            argparse.Namespace(
                wm_strategy_profile="elite_bank_actor_distill_awr",
                awr_enable=True,
                awr_loss_weight=0.0007,
            )
        )
        self.assertTrue(caps["profile_awr_enabled"])

    def test_build_config_batch_steps_alias(self):
        ns = argparse.Namespace(awr_batch_size=2048, awr_copy_top_k=1000, awr_broadcast_top_k=500)
        cfg = build_elite_awr_config_from_args(ns)
        self.assertEqual(cfg.batch_steps, 2048)
        self.assertEqual(cfg.copy_top_k_steps, 1000)


if __name__ == "__main__":
    unittest.main()
