"""Tests for EliteSnippetBank."""

from __future__ import annotations

import types
import unittest

import torch

from dexpbt_isaaclab.world_model.elite_snippet_bank import (
    SNIPPET_SCHEMA_VERSION,
    EliteSnippetBank,
    EliteSnippetBankConfig,
)
from dexpbt_isaaclab.world_model.elite_snippet_builder import SnippetBatch, SnippetTargetBuilder


class TestEliteSnippetBank(unittest.TestCase):
    def _batch(self, n=4, h=4, a=3, f=8):
        return SnippetBatch(
            feat_start=torch.randn(n, f),
            actions=torch.randn(n, h, a).clamp(-1, 1),
            d_hat=torch.full((n,), 0.02),
            success_tol=torch.full((n,), 0.015),
            lifted=torch.ones(n),
            picked=torch.zeros(n),
            dropped=torch.zeros(n),
            closest_d_hat=torch.full((n,), 0.02),
            success_h=torch.tensor([1.0, 0.0, 1.0, 0.0]),
            improve_h=torch.tensor([0.0, 1.0, 0.0, 0.0]),
            drop_h=torch.zeros(n),
            recovery_label=torch.tensor([1.0, 1.0, 1.0, 0.0]),
            s0=torch.tensor([1.0, 0.0, 1.0, 0.0]),
            s1=torch.tensor([1.0, 0.0, 0.0, 0.0]),
            i0=torch.tensor([0.0, 1.0, 0.0, 0.0]),
            d0=torch.zeros(n),
            d1=torch.zeros(n),
        )

    def test_insert_and_shape(self):
        cfg = EliteSnippetBankConfig(enabled=True, bank_size=20, feature_dim=8, action_dim=3, horizon=4)
        bank = EliteSnippetBank(cfg)
        bank.update_from_builder_batch(self._batch(), source_stats={"source_nsucc_mean": 10.0})
        self.assertGreater(bank.size, 0)
        acts, sim, pri, ok = bank.retrieve_candidates(torch.randn(2, 8), k=2)
        self.assertTrue(ok)
        self.assertEqual(tuple(acts.shape), (2, 2, 4, 3))

    def test_state_dict_roundtrip(self):
        cfg = EliteSnippetBankConfig(enabled=True, bank_size=20, feature_dim=8, action_dim=3, horizon=4)
        bank = EliteSnippetBank(cfg)
        bank.update_from_builder_batch(self._batch())
        sd = bank.state_dict()
        self.assertEqual(int(sd["schema_version"]), SNIPPET_SCHEMA_VERSION)
        bank2 = EliteSnippetBank(cfg)
        bank2.load_state_dict(sd)
        self.assertEqual(bank2.size, bank.size)

    def test_builder_horizon(self):
        t, n, a, h = 6, 2, 3, 4
        cache = types.SimpleNamespace(
            T=t,
            N=n,
            actions_flat=torch.randn(t * n, a),
            d_hat_flat=torch.full((t * n,), 0.02),
            success_tol_flat=torch.full((t * n,), 0.015),
            lifted_flat=torch.ones(t * n),
            targets_flat={
                "success_h4": torch.ones(t * n),
                "improve_h4": torch.zeros(t * n),
                "drop_h4": torch.zeros(t * n),
                "success_h8": torch.zeros(t * n),
                "drop_h8": torch.zeros(t * n),
            },
        )
        batch = SnippetTargetBuilder(horizon=h, horizons=(4, 8)).build(
            cache, torch.randn(t * n, 8)
        )
        self.assertGreater(batch.actions.shape[0], 0)
        self.assertEqual(batch.actions.shape[1], h)

    def test_builder_gpu_cache_cpu_features(self):
        if not torch.cuda.is_available():
            self.skipTest("cuda required")
        t, n, a, h = 6, 2, 3, 4
        cache = types.SimpleNamespace(
            T=t,
            N=n,
            actions_flat=torch.randn(t * n, a, device="cuda"),
            d_hat_flat=torch.full((t * n,), 0.02, device="cuda"),
            success_tol_flat=torch.full((t * n,), 0.015, device="cuda"),
            lifted_flat=torch.ones(t * n, device="cuda"),
            targets_flat={
                "success_h4": torch.ones(t * n, device="cuda"),
                "improve_h4": torch.zeros(t * n, device="cuda"),
                "drop_h4": torch.zeros(t * n, device="cuda"),
                "success_h8": torch.zeros(t * n, device="cuda"),
                "drop_h8": torch.zeros(t * n, device="cuda"),
            },
        )
        batch = SnippetTargetBuilder(horizon=h, horizons=(4, 8)).build(
            cache, torch.randn(t * n, 8)
        )
        self.assertGreater(batch.actions.shape[0], 0)
        self.assertEqual(batch.feat_start.device.type, "cpu")

    def test_update_from_gpu_batch(self):
        if not torch.cuda.is_available():
            self.skipTest("cuda required")
        cfg = EliteSnippetBankConfig(enabled=True, bank_size=20, feature_dim=8, action_dim=3, horizon=4)
        bank = EliteSnippetBank(cfg)
        batch = self._batch()
        gpu_batch = SnippetBatch(
            feat_start=batch.feat_start.cuda(),
            actions=batch.actions.cuda(),
            d_hat=batch.d_hat.cuda(),
            success_tol=batch.success_tol.cuda(),
            lifted=batch.lifted.cuda(),
            picked=batch.picked.cuda(),
            dropped=batch.dropped.cuda(),
            closest_d_hat=batch.closest_d_hat.cuda(),
            success_h=batch.success_h.cuda(),
            improve_h=batch.improve_h.cuda(),
            drop_h=batch.drop_h.cuda(),
            recovery_label=batch.recovery_label.cuda(),
            s0=batch.s0.cuda(),
            s1=batch.s1.cuda(),
            i0=batch.i0.cuda(),
            d0=batch.d0.cuda(),
            d1=batch.d1.cuda(),
        )
        bank.update_from_builder_batch(gpu_batch, source_stats={"source_nsucc_mean": 10.0})
        self.assertGreater(bank.size, 0)
