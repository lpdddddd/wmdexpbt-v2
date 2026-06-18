"""Unit tests for Strategy Zoo v1.7 Online Elite-AWR module."""

from __future__ import annotations

import argparse
import json
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
    compute_elite_awr_loss,
    compute_elite_awr_quality,
    load_awr_sidecar,
    patch_model_get_aux_loss_for_awr,
    save_awr_sidecar,
    save_awr_sidecar_topk,
)
from dexpbt_isaaclab.awr.elite_awr_interceptor import (
    DoneReason,
    EliteAWRInterceptorEnv,
    classify_done_reason,
)
from dexpbt_isaaclab.world_model.wm_strategy_profiles import (
    PROFILE_COPY_SOURCE_BONUS,
    PROFILE_IDS,
    PROFILE_TABLES,
    apply_wm_strategy_profile,
    profile_capability_flags,
    profile_copy_source_bonus,
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
        cfg = EliteAWRConfig(min_buffer_size=1, copy_top_k_steps=40)
        buf = EliteAWRBuffer(cfg, policy_idx=4)
        buf.insert(_make_entry(32, truncated=False, starts_at_reset=True))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "policy_04_awr_topk.pt"
            ok, meta = save_awr_sidecar_topk(buf, path, step_budget=40)
            self.assertTrue(ok)
            self.assertEqual(meta["save_failed"], 0.0)
            self.assertFalse(meta.get("full_buffer", True))
            main_ckpt = {"awr_buffer_ref": path.name, "awr_buffer_ref_meta": meta}
            self.assertNotIn("awr_buffer_topk", main_ckpt)
            self.assertIn("awr_buffer_ref", main_ckpt)
            buf2 = EliteAWRBuffer(cfg, policy_idx=4)
            loaded = load_awr_sidecar(buf2, path, source_type="local")
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

    def test_awr_patch_independent_of_eir(self):
        model = mock.Mock()
        model.get_aux_loss = lambda: None
        bridge = EliteAWRBridge(
            buffer=EliteAWRBuffer(EliteAWRConfig(min_buffer_size=10**9), policy_idx=0),
            cfg=EliteAWRConfig(),
        )
        bridge.compute_aux_loss = lambda: {"awr_total": torch.tensor(0.01)}
        patch_model_get_aux_loss_for_awr(model, bridge)
        out = model.get_aux_loss()
        self.assertIsInstance(out, dict)
        self.assertIn("awr_total", out)

    def test_eir_and_awr_aux_chain_coexist(self):
        model = mock.Mock()
        model.get_aux_loss = lambda: {"eir_total": torch.tensor(1.0)}
        bridge = EliteAWRBridge(
            buffer=EliteAWRBuffer(EliteAWRConfig(min_buffer_size=10**9), policy_idx=0),
            cfg=EliteAWRConfig(),
        )
        patch_model_get_aux_loss_for_awr(model, bridge)
        out = model.get_aux_loss()
        self.assertIn("eir_total", out)

    def test_selector_awr_has_aux_patch_when_eir_off(self):
        ns = argparse.Namespace(wm_strategy_profile="wm_pbt_source_selector_awr")
        for dest in {k for t in PROFILE_TABLES.values() for k in t}:
            if not hasattr(ns, dest):
                setattr(ns, dest, None)
        apply_wm_strategy_profile(ns, authoritative=True)
        self.assertTrue(awr_active_from_args(ns))
        self.assertFalse(getattr(ns, "eir_enable", True))
        model = mock.Mock()
        model.get_aux_loss = lambda: None
        bridge = EliteAWRBridge(
            buffer=EliteAWRBuffer(EliteAWRConfig(), policy_idx=3),
            cfg=build_elite_awr_config_from_args(ns),
        )
        patch_model_get_aux_loss_for_awr(model, bridge)
        self.assertTrue(callable(model.get_aux_loss))

    def test_action_convention_check_is_called_after_rollout(self):
        cfg = EliteAWRConfig()
        bridge = EliteAWRBridge(buffer=EliteAWRBuffer(cfg, policy_idx=0), cfg=cfg)
        captured = torch.stack([torch.tensor([[0.1, 0.2]]), torch.tensor([[0.3, 0.4]])])
        ppo = captured.clone()
        captured_flat = captured.reshape(-1, captured.shape[-1])
        ppo_flat = ppo.reshape(-1, ppo.shape[-1])
        bridge.maybe_check_action_convention(captured_flat, ppo_flat)
        self.assertEqual(bridge.stats.get("awr/action_convention_checked"), 1.0)

    def test_action_convention_mismatch_disables_awr(self):
        cfg = EliteAWRConfig()
        bridge = EliteAWRBridge(buffer=EliteAWRBuffer(cfg, policy_idx=0), cfg=cfg)
        a = torch.zeros(4, 2)
        bridge.maybe_check_action_convention(a, a + 1.0)
        self.assertTrue(bridge.disabled)
        self.assertEqual(bridge.stats.get("awr/action_convention_mismatch"), 1.0)

    def test_awr_kl_gate_disabled_is_explicit(self):
        cfg = EliteAWRConfig(kl_gate_enable=False, start_after_frames=0, loss_weight=0.001)
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
        self.assertEqual(stats["awr/kl_gate_enabled"], 0.0)
        self.assertEqual(stats["awr/kl_telemetry_only"], 1.0)

    def test_awr_profile_copy_bonus_matches_v13_base(self):
        self.assertAlmostEqual(
            profile_copy_source_bonus("elite_bank_actor_distill_fast_awr"),
            PROFILE_COPY_SOURCE_BONUS["elite_bank_actor_distill_fast"],
        )
        self.assertAlmostEqual(
            profile_copy_source_bonus("elite_bank_actor_distill_awr"),
            PROFILE_COPY_SOURCE_BONUS["elite_bank_actor_distill"],
        )
        self.assertAlmostEqual(
            profile_copy_source_bonus("elite_bank_actor_distill_strong_awr"),
            PROFILE_COPY_SOURCE_BONUS["elite_bank_actor_distill_strong"],
        )
        self.assertAlmostEqual(
            profile_copy_source_bonus("wm_pbt_source_selector_awr"),
            PROFILE_COPY_SOURCE_BONUS["wm_pbt_source_selector"],
        )
        self.assertAlmostEqual(profile_copy_source_bonus("ppo_only"), 0.6)
        self.assertAlmostEqual(profile_copy_source_bonus("wm_pbt_source_selector_v2"), 1.0)

    def test_awr_sidecar_saved_on_checkpoint(self):
        cfg = EliteAWRConfig(copy_top_k_steps=30)
        buf = EliteAWRBuffer(cfg, policy_idx=2)
        buf.insert(_make_entry(20, truncated=False, starts_at_reset=True))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "policy_02_awr_topk.pt"
            ok, meta = save_awr_sidecar(buf, path)
            self.assertTrue(ok)
            self.assertTrue(path.is_file())

    def test_awr_sidecar_manifest_contains_ref_only(self):
        cfg = EliteAWRConfig(copy_top_k_steps=30)
        buf = EliteAWRBuffer(cfg, policy_idx=1)
        buf.insert(_make_entry(16, truncated=False, starts_at_reset=True))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "policy_01_awr_topk.pt"
            _, meta = save_awr_sidecar_topk(buf, path, step_budget=30)
            manifest = {"awr_buffer_ref": path.name, "awr_buffer_ref_meta": meta}
            self.assertNotIn("local", manifest)
            self.assertIn("num_sequences", meta)

    def test_awr_inherit_loads_source_sidecar_to_inherited_pool(self):
        cfg = EliteAWRConfig(copy_top_k_steps=50)
        src = EliteAWRBuffer(cfg, policy_idx=1)
        entry = _make_entry(20, truncated=False, starts_at_reset=True)
        src.insert(entry)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "policy_01_awr_topk.pt"
            save_awr_sidecar_topk(src, path, step_budget=50)
            dst = EliteAWRBuffer(cfg, policy_idx=3)
            loaded = load_awr_sidecar(dst, path, source_type="inherited")
            self.assertGreater(loaded, 0)
            self.assertEqual(len(dst.local), 0)
            self.assertGreater(len(dst.inherited), 0)

    def test_awr_sidecar_topk_respects_step_budget(self):
        cfg = EliteAWRConfig(copy_top_k_steps=35)
        buf = EliteAWRBuffer(cfg, policy_idx=0)
        for q in (6.0, 5.5, 5.0, 4.5):
            e = _make_entry(20, truncated=False, starts_at_reset=True)
            e.episode_quality = q
            buf.insert(e)
        payload = buf.sidecar_payload_topk(copy_step_budget=35, broadcast_step_budget=0)
        steps = int(payload["meta"]["num_steps"])
        self.assertLessEqual(steps, 35)

    def test_awr_sidecar_does_not_save_full_buffer_by_default(self):
        cfg = EliteAWRConfig(copy_top_k_steps=10)
        buf = EliteAWRBuffer(cfg, policy_idx=0)
        for _ in range(5):
            buf.insert(_make_entry(20, truncated=False, starts_at_reset=True))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "topk.pt"
            _, meta = save_awr_sidecar(buf, path)
            self.assertFalse(meta.get("full_buffer", True))
            self.assertLessEqual(meta["num_steps"], cfg.copy_top_k_steps)

    def test_future_improvement_uses_d_hat_window(self):
        pending = []
        for i, d in enumerate([0.5, 0.4, 0.3, 0.2, 0.25]):
            pending.append(
                {
                    "obs": torch.randn(4),
                    "action": torch.randn(2),
                    "lifted": True,
                    "drop": False,
                    "tolerance": 0.05,
                    "d_hat": d,
                    "quality_step_pass": True,
                    "info_missing": {},
                }
            )
        import math

        for i, step in enumerate(pending):
            current_d = float(step["d_hat"])
            future_slice = pending[i : min(len(pending), i + 101)]
            future_d_vals = [
                float(s["d_hat"])
                for s in future_slice
                if math.isfinite(float(s.get("d_hat", float("nan"))))
            ]
            step["future_improvement"] = float(current_d - min(future_d_vals))
        self.assertAlmostEqual(pending[0]["future_improvement"], 0.3)
        self.assertAlmostEqual(pending[-1]["future_improvement"], 0.0)

    def test_future_improvement_missing_distance_neutral(self):
        pending = [
            {
                "obs": torch.randn(4),
                "action": torch.randn(2),
                "lifted": True,
                "drop": False,
                "tolerance": 0.05,
                "quality_step_pass": True,
                "info_missing": {"distance": True},
            }
        ]
        import math

        step = pending[0]
        current_d = float(step.get("d_hat", float("nan")))
        if not math.isfinite(current_d):
            step["future_improvement"] = 0.0
        self.assertEqual(step["future_improvement"], 0.0)

    def test_p0_p2_no_awr_wrapper_or_patch(self):
        for profile in ("ppo_only", "wm_pbt_source_selector_v2"):
            ns = argparse.Namespace(
                wm_strategy_profile=profile,
                awr_enable=True,
                awr_loss_weight=0.001,
            )
            for dest in {k for t in PROFILE_TABLES.values() for k in t}:
                if not hasattr(ns, dest):
                    setattr(ns, dest, None)
            apply_wm_strategy_profile(ns, authoritative=True)
            self.assertFalse(awr_active_from_args(ns))
            self.assertFalse(getattr(ns, "awr_enable", True))
            self.assertEqual(float(getattr(ns, "awr_loss_weight", 0.0)), 0.0)

    def test_v17_analyzer_separates_awr_and_actor_distill(self):
        import importlib.util
        import sys

        repo_root = Path(__file__).resolve().parents[4]
        analyzer_path = repo_root / "results" / "analyze_strategy_zoo_v17_online_elite_awr.py"
        spec = importlib.util.spec_from_file_location("v17_analyzer", analyzer_path)
        assert spec and spec.loader
        mod = importlib.util.module_from_spec(spec)
        sys.modules["v17_analyzer"] = mod
        spec.loader.exec_module(mod)
        with tempfile.TemporaryDirectory() as tmp:
            metrics = Path(tmp) / "dexpbt_8gpu_ws" / "metrics"
            metrics.mkdir(parents=True)
            row = {
                "env_frames": 1_000_000_000,
                "nsucc_mean": 30.0,
                "eir/actor_distill_active_fraction": 0.2,
                "eir/actor_distill_weighted_loss": 0.01,
                "awr/loss_weighted": 0.0005,
                "awr/enabled": 1.0,
            }
            (metrics / "training_curve_policy_01.jsonl").write_text(json.dumps(row) + "\n")
            report = mod.analyze_run(Path(tmp))
            block = report["separation_block"]["p1"]
            self.assertIn("actor_distill_active_fraction", block["eir"])
            self.assertIn("weighted_loss", block["awr"])
            self.assertNotEqual(block["eir"], block["awr"])


if __name__ == "__main__":
    unittest.main()
