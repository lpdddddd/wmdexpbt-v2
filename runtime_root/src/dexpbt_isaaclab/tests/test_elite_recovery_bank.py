"""Unit tests for the EIR-MPC Elite Recovery Bank (Phase A)."""

from __future__ import annotations

import types
import unittest

import torch

from dexpbt_isaaclab.world_model.elite_recovery_bank import (
    EliteRecoveryBank,
    EliteRecoveryBankConfig,
    STAGE_FLOOR,
    STAGE_HARD,
    STAGE_MID,
)


def _make_cache(*, actions, d_hat, tol, lifted, targets):
    """Build a minimal ERD-like target cache object for the bank."""
    n = actions.shape[0]
    cache = types.SimpleNamespace()
    cache.actions_flat = actions.clamp(-1.0, 1.0)
    cache.d_hat_flat = d_hat.reshape(-1)
    cache.success_tol_flat = tol.reshape(-1)
    cache.lifted_flat = lifted.reshape(-1) if lifted is not None else None
    cache.targets_flat = targets
    cache.T = 1
    cache.N = n
    cache.action_dim = actions.shape[-1]
    return cache


def _zeros(n):
    return torch.zeros(n)


class TestEliteBank(unittest.TestCase):
    def _cfg(self, **kw):
        base = dict(
            enabled=True,
            bank_size=100,
            feature_dim=4,
            action_dim=3,
            horizons=(4, 8),
            tol_threshold=0.02,
            hard_tol_threshold=0.012,
            near_success_mult=3.0,
            min_priority=1.0,
        )
        base.update(kw)
        return EliteRecoveryBankConfig(**base)

    def _elite_cache(self):
        # 6 rows; elite rows are {0 (success), 4 (improve)}.
        n = 6
        actions = torch.zeros(n, 3)
        for i in range(n):
            actions[i] = float(i) / 10.0
        d_hat = torch.tensor([0.02, 0.02, 0.02, 0.02, 0.02, 0.30])
        tol = torch.tensor([0.015, 0.015, 0.05, 0.015, 0.015, 0.015])
        lifted = torch.tensor([1.0, 0.0, 1.0, 1.0, 1.0, 1.0])
        targets = {
            "success_h4": torch.tensor([1.0, 1.0, 1.0, 1.0, 0.0, 1.0]),
            "improve_h4": torch.tensor([0.0, 0.0, 0.0, 0.0, 1.0, 0.0]),
            "drop_h4": torch.tensor([0.0, 0.0, 0.0, 1.0, 0.0, 0.0]),
            "success_h8": _zeros(n),
            "improve_h8": _zeros(n),
            "drop_h8": _zeros(n),
        }
        feats = torch.arange(n * 4, dtype=torch.float32).reshape(n, 4)
        cache = _make_cache(
            actions=actions, d_hat=d_hat, tol=tol, lifted=lifted, targets=targets
        )
        return cache, feats

    def test_disabled_bank_noop(self):
        bank = EliteRecoveryBank(self._cfg(enabled=False))
        cache, feats = self._elite_cache()
        bank.update_from_cache(cache, feats, source_stats={"source_nsucc_mean": 10.0})
        self.assertEqual(bank.size, 0)

    def test_update_diagnostics_smoke(self):
        # Observation-only diagnostics must populate and stay consistent with the
        # actual inserts (does not change insert behaviour).
        bank = EliteRecoveryBank(self._cfg())
        cache, feats = self._elite_cache()
        bank.update_from_cache(
            cache,
            feats,
            source_stats={"source_policy_idx": 3, "source_nsucc_mean": 10.0, "source_nsucc_max": 20.0},
            frame=1234,
        )
        diag = bank.update_diagnostics()
        self.assertEqual(diag["update_seen_rows"], 6.0)
        self.assertEqual(diag["update_inserted_rows"], float(bank.size))
        # lifted rows: 5 of 6 are lifted
        self.assertEqual(diag["update_lifted_rows"], 5.0)
        # diagnostics must reset between updates
        empty = _make_cache(
            actions=torch.zeros(2, 3),
            d_hat=torch.full((2,), 0.30),
            tol=torch.full((2,), 0.05),
            lifted=torch.zeros(2),
            targets={k: torch.zeros(2) for k in (
                "success_h4", "improve_h4", "drop_h4",
                "success_h8", "improve_h8", "drop_h8",
            )},
        )
        bank.update_from_cache(empty, torch.zeros(2, 4))
        diag2 = bank.update_diagnostics()
        self.assertEqual(diag2["update_seen_rows"], 2.0)
        self.assertEqual(diag2["update_inserted_rows"], 0.0)
        self.assertEqual(diag2["reject_reason_not_lifted"], 2.0)

    def test_update_adds_only_elite(self):
        bank = EliteRecoveryBank(self._cfg())
        cache, feats = self._elite_cache()
        bank.update_from_cache(
            cache,
            feats,
            source_stats={"source_policy_idx": 3, "source_nsucc_mean": 10.0, "source_nsucc_max": 20.0},
            frame=1234,
        )
        # rows 0 and 4 satisfy elite condition and clear min_priority
        self.assertEqual(bank.size, 2)
        # alignment: stored features equal input feature rows for elite indices
        stored = bank.feature[: bank.size]
        self.assertTrue(torch.allclose(stored[0], feats[0]))
        self.assertTrue(torch.allclose(stored[1], feats[4]))

    def test_priority_computation(self):
        one = torch.ones(1)
        zero = torch.zeros(1)
        pr = EliteRecoveryBank.compute_priority(
            s0=one, s1=zero, i0=zero, i1=zero, d0=zero, d1=zero,
            source_nsucc_mean=zero, source_nsucc_max=zero, super_elite=torch.zeros(1, dtype=torch.bool),
        )
        self.assertAlmostEqual(float(pr.item()), 2.0, places=5)
        pr2 = EliteRecoveryBank.compute_priority(
            s0=one, s1=one, i0=one, i1=zero, d0=zero, d1=zero,
            source_nsucc_mean=torch.tensor([10.0]), source_nsucc_max=torch.tensor([20.0]),
            super_elite=torch.ones(1, dtype=torch.bool),
        )
        # 2 + 1.2 + 0.8 + 0.3 + 0.2 + 1.5(super) = 6.0
        self.assertAlmostEqual(float(pr2.item()), 6.0, places=5)

    def test_max_size_enforced(self):
        bank = EliteRecoveryBank(self._cfg(bank_size=3, min_priority=0.0))
        # 5 elite rows with distinct priorities via success/improve patterns
        n = 5
        actions = torch.randn(n, 3).clamp(-1, 1)
        d_hat = torch.full((n,), 0.02)
        tol = torch.full((n,), 0.015)
        lifted = torch.ones(n)
        targets = {
            "success_h4": torch.tensor([1.0, 1.0, 0.0, 0.0, 0.0]),
            "improve_h4": torch.tensor([0.0, 0.0, 1.0, 1.0, 1.0]),
            "drop_h4": torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0]),
            "success_h8": torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0]),
            "improve_h8": torch.tensor([0.0, 0.0, 1.0, 0.0, 0.0]),
            "drop_h8": torch.tensor([0.0, 0.0, 0.0, 0.0, 1.0]),
        }
        feats = torch.randn(n, 4)
        cache = _make_cache(actions=actions, d_hat=d_hat, tol=tol, lifted=lifted, targets=targets)
        bank.update_from_cache(cache, feats, source_stats={})
        self.assertEqual(bank.size, 3)
        # top 3 priorities should be retained
        kept = sorted(bank.scalars["priority_score"][: bank.size].tolist(), reverse=True)
        self.assertGreaterEqual(kept[0], kept[1])
        self.assertGreaterEqual(kept[1], kept[2])
        # highest-priority entry (s0+s1) must survive
        self.assertAlmostEqual(max(kept), 2.0 + 1.2, places=4)

    def test_state_dict_roundtrip(self):
        bank = EliteRecoveryBank(self._cfg())
        cache, feats = self._elite_cache()
        bank.update_from_cache(cache, feats, source_stats={"source_nsucc_mean": 10.0, "source_nsucc_max": 20.0})
        sd = bank.state_dict()
        bank2 = EliteRecoveryBank(self._cfg())
        bank2.load_state_dict(sd)
        self.assertEqual(bank2.size, bank.size)
        self.assertTrue(torch.allclose(bank2.feature[: bank2.size], bank.feature[: bank.size]))
        self.assertTrue(torch.allclose(bank2.action[: bank2.size], bank.action[: bank.size]))

    def test_merge_keeps_high_priority(self):
        bank = EliteRecoveryBank(self._cfg(bank_size=3, min_priority=0.0))
        cache, feats = self._elite_cache()
        bank.update_from_cache(cache, feats, source_stats={"source_nsucc_mean": 10.0, "source_nsucc_max": 20.0})
        other = EliteRecoveryBank(self._cfg(bank_size=10, min_priority=0.0))
        # craft a very high priority entry in other
        n = 2
        actions = torch.randn(n, 3).clamp(-1, 1)
        d_hat = torch.full((n,), 0.005)
        tol = torch.full((n,), 0.01)
        lifted = torch.ones(n)
        targets = {
            "success_h4": torch.tensor([1.0, 1.0]),
            "improve_h4": torch.tensor([1.0, 1.0]),
            "drop_h4": torch.zeros(n),
            "success_h8": torch.tensor([1.0, 1.0]),
            "improve_h8": torch.tensor([1.0, 1.0]),
            "drop_h8": torch.zeros(n),
        }
        feats2 = torch.randn(n, 4)
        c2 = _make_cache(actions=actions, d_hat=d_hat, tol=tol, lifted=lifted, targets=targets)
        other.update_from_cache(c2, feats2, source_stats={"source_nsucc_mean": 30.0, "source_nsucc_max": 50.0})
        high = other.scalars["priority_score"][: other.size].max().item()
        bank.merge(other.state_dict(), max_entries=50)
        self.assertLessEqual(bank.size, 3)
        self.assertAlmostEqual(
            bank.scalars["priority_score"][: bank.size].max().item(), high, places=4
        )

    def test_retrieval_shape_B_K_A(self):
        bank = EliteRecoveryBank(self._cfg(min_priority=0.0))
        n = 12
        actions = torch.randn(n, 3).clamp(-1, 1)
        d_hat = torch.full((n,), 0.02)
        tol = torch.full((n,), 0.015)
        lifted = torch.ones(n)
        targets = {
            "success_h4": torch.ones(n),
            "improve_h4": torch.zeros(n),
            "drop_h4": torch.zeros(n),
            "success_h8": torch.zeros(n),
            "improve_h8": torch.zeros(n),
            "drop_h8": torch.zeros(n),
        }
        feats = torch.randn(n, 4)
        cache = _make_cache(actions=actions, d_hat=d_hat, tol=tol, lifted=lifted, targets=targets)
        bank.update_from_cache(cache, feats, source_stats={})
        q = torch.randn(2, 4)
        acts, sim, pri, valid = bank.retrieve_candidates(q, k=5)
        self.assertTrue(valid)
        self.assertEqual(tuple(acts.shape), (2, 5, 3))
        self.assertEqual(tuple(sim.shape), (2, 5))
        self.assertEqual(tuple(pri.shape), (2, 5))

    def test_retrieval_empty_bank_fallback(self):
        bank = EliteRecoveryBank(self._cfg())
        q = torch.randn(3, 4)
        acts, sim, pri, valid = bank.retrieve_candidates(q, k=4)
        self.assertFalse(valid)
        self.assertEqual(tuple(acts.shape), (3, 4, 3))
        self.assertTrue(torch.allclose(acts, torch.zeros_like(acts)))

    def test_feature_dim_mismatch_raises(self):
        bank = EliteRecoveryBank(self._cfg(feature_dim=4))
        cache, _ = self._elite_cache()
        bad = torch.randn(6, 8)  # wrong width
        with self.assertRaises(ValueError):
            bank.update_from_cache(cache, bad, source_stats={})

    def test_fallback_feature_key_width_accepted(self):
        # 19.2: a fallback retrieval key of correct width is accepted as feature.
        bank = EliteRecoveryBank(self._cfg(feature_dim=4))
        cache, _ = self._elite_cache()
        fallback_key = torch.randn(6, 4)
        bank.update_from_cache(cache, fallback_key, source_stats={"source_nsucc_mean": 10.0})
        self.assertEqual(bank.size, 2)
        self.assertTrue(torch.allclose(bank.feature[0], fallback_key[0]))


class TestStageAwareGuidanceBank(unittest.TestCase):
    """Stage-aware (mid-band) collection. STAGE_MID=3 -> buckets 6/7."""

    def _cfg(self, **kw):
        base = dict(
            enabled=True,
            bank_size=100,
            feature_dim=4,
            action_dim=3,
            horizons=(4, 8),
            tol_threshold=0.02,
            hard_tol_threshold=0.012,
            near_success_mult=3.0,
            min_priority=0.5,
            guidance_enable=True,
            guidance_bank_tol=0.04,
            guidance_mid_near_success_mult=1.5,
            guidance_mid_priority_scale=0.5,
        )
        base.update(kw)
        return EliteRecoveryBankConfig(**base)

    def _rows(self, tols):
        """One success row per tol value; near-success d_hat=0.01, lifted."""
        n = len(tols)
        actions = torch.zeros(n, 3)
        d_hat = torch.full((n,), 0.01)
        tol = torch.tensor(tols, dtype=torch.float32)
        lifted = torch.ones(n)
        targets = {
            "success_h4": torch.ones(n),
            "improve_h4": torch.zeros(n),
            "drop_h4": torch.zeros(n),
            "success_h8": torch.zeros(n),
            "improve_h8": torch.zeros(n),
            "drop_h8": torch.zeros(n),
        }
        feats = torch.arange(n * 4, dtype=torch.float32).reshape(n, 4)
        cache = _make_cache(
            actions=actions, d_hat=d_hat, tol=tol, lifted=lifted, targets=targets
        )
        return cache, feats

    def test_elite_bank_collects_mid_stage_when_guidance_enabled(self):
        bank = EliteRecoveryBank(self._cfg())
        cache, feats = self._rows([0.03])  # mid band: 0.02 < 0.03 <= 0.04
        bank.update_from_cache(cache, feats, source_stats={"source_nsucc_mean": 10.0})
        self.assertEqual(bank.size, 1)
        bucket = bank.scalars["bucket_id"][: bank.size]
        # mid success -> STAGE_MID*2 + 1 = 7
        self.assertEqual(int(bucket[0].item()), STAGE_MID * 2 + 1)

    def test_elite_bank_does_not_collect_mid_stage_when_guidance_disabled(self):
        bank = EliteRecoveryBank(self._cfg(guidance_enable=False))
        cache, feats = self._rows([0.03])
        bank.update_from_cache(cache, feats, source_stats={"source_nsucc_mean": 10.0})
        self.assertEqual(bank.size, 0)
        # but a floor row is still collected with guidance off
        bank2 = EliteRecoveryBank(self._cfg(guidance_enable=False))
        cache2, feats2 = self._rows([0.015])
        bank2.update_from_cache(cache2, feats2, source_stats={"source_nsucc_mean": 10.0})
        self.assertEqual(bank2.size, 1)

    def test_elite_bank_mid_priority_scaled_below_floor(self):
        bank = EliteRecoveryBank(self._cfg())
        # row 0 floor (tol=0.015), row 1 mid (tol=0.03); identical labels/source
        cache, feats = self._rows([0.015, 0.03])
        bank.update_from_cache(cache, feats, source_stats={"source_nsucc_mean": 10.0})
        self.assertEqual(bank.size, 2)
        bucket = bank.scalars["bucket_id"][: bank.size]
        pri = bank.scalars["priority_score"][: bank.size]
        floor_pri = float(pri[bucket == float(STAGE_FLOOR * 2 + 1)][0].item())
        mid_pri = float(pri[bucket == float(STAGE_MID * 2 + 1)][0].item())
        self.assertAlmostEqual(mid_pri, 0.5 * floor_pri, places=5)

    def test_elite_bank_stage_bucket_ids_are_distinct(self):
        bank = EliteRecoveryBank(self._cfg())
        # hard (0.01), floor (0.015), mid (0.03)
        cache, feats = self._rows([0.01, 0.015, 0.03])
        bank.update_from_cache(cache, feats, source_stats={"source_nsucc_mean": 10.0})
        self.assertEqual(bank.size, 3)
        buckets = set(int(b) for b in bank.scalars["bucket_id"][: bank.size].tolist())
        self.assertEqual(
            buckets,
            {STAGE_HARD * 2 + 1, STAGE_FLOOR * 2 + 1, STAGE_MID * 2 + 1},
        )

    def test_elite_bank_diagnostics_stage_counts(self):
        bank = EliteRecoveryBank(self._cfg())
        cache, feats = self._rows([0.01, 0.015, 0.03])
        bank.update_from_cache(cache, feats, source_stats={"source_nsucc_mean": 10.0})
        diag = bank.update_diagnostics()
        self.assertEqual(diag["update_hard_stage_rows"], 1.0)
        self.assertEqual(diag["update_floor_stage_rows"], 1.0)
        self.assertEqual(diag["update_mid_stage_rows"], 1.0)
        self.assertEqual(diag["update_inserted_hard_rows"], 1.0)
        self.assertEqual(diag["update_inserted_floor_rows"], 1.0)
        self.assertEqual(diag["update_inserted_mid_rows"], 1.0)
        self.assertEqual(diag["reject_reason_not_guidance_stage"], 0.0)

    def test_elite_bank_above_guidance_tol_rejected(self):
        bank = EliteRecoveryBank(self._cfg())
        cache, feats = self._rows([0.05])  # above guidance_bank_tol=0.04
        bank.update_from_cache(cache, feats, source_stats={"source_nsucc_mean": 10.0})
        self.assertEqual(bank.size, 0)
        diag = bank.update_diagnostics()
        self.assertEqual(diag["reject_reason_not_guidance_stage"], 1.0)


if __name__ == "__main__":
    unittest.main()
