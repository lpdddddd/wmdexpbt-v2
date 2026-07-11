"""Phase A4 P1 (v2 coupling) -- teacher value normalization + corr gate.

Targets the v2 mode safeguards introduced in `seq_policy_aux.py`:

  * `apply_rollout_zscore` standardizes the (T, N) teacher tensor.
  * `compute_teacher_return_corr` measures Pearson(teacher_value, returns).
  * `build_teacher_cache_from_predictions` populates `cache.meta` with all
    diagnostic stats.
  * `compute_seq_distill_aux_loss` blocks the rollout when corr is below
    the threshold (or invalid) -- gate_blocked is NOT a fail-fast.
"""

from __future__ import annotations

import unittest
from typing import Optional

import torch
import torch.nn as nn

from dexpbt_isaaclab.world_model.seq_policy_aux import (
    COUPLING_CRITIC_DISTILL_VALUE_NORM,
    FEATURE_SOURCE_SHARED_TO_AUX_CRITIC,
    GATE_NONE,
    GATE_TEACHER_CORR,
    VALUE_NORM_ROLLOUT_ZSCORE,
    CapturedFeature,
    SeqDistillBridge,
    SeqPolicyCouplingConfig,
    apply_rollout_zscore,
    build_seq_aux_critic_subtower,
    build_seq_aux_value_head,
    build_teacher_cache_from_predictions,
    compute_seq_distill_aux_loss,
    compute_teacher_return_corr,
)


T = 4
N = 3
FEATURE_DIM = 8


class _FakeDataset:
    def __init__(self) -> None:
        self.last_range: Optional[tuple[int, int]] = None


class _FakeAgent:
    def __init__(self) -> None:
        self.dataset = _FakeDataset()


def _v2_cfg(**overrides) -> SeqPolicyCouplingConfig:
    base = dict(
        mode=COUPLING_CRITIC_DISTILL_VALUE_NORM,
        weight=1.0,
        value_weight=1.0,
        success_weight=0.0,
        warmup_frames=0,
        feature_source=FEATURE_SOURCE_SHARED_TO_AUX_CRITIC,
        teacher_value_norm=VALUE_NORM_ROLLOUT_ZSCORE,
        gate=GATE_TEACHER_CORR,
        teacher_corr_threshold=0.2,
        ramp_frames=0,
    )
    base.update(overrides)
    return SeqPolicyCouplingConfig(**base)


def _v2_bridge_with_heads(cfg: SeqPolicyCouplingConfig) -> SeqDistillBridge:
    bridge = SeqDistillBridge(cfg=cfg)
    bridge.rollout_generation = 1
    bridge.last_frames = 10**9
    bridge.agent_ref = _FakeAgent()
    bridge.seq_aux_value_head = build_seq_aux_value_head(in_dim=FEATURE_DIM)
    bridge.seq_aux_critic_mlp = build_seq_aux_critic_subtower(
        in_dim=FEATURE_DIM, hidden=FEATURE_DIM
    )
    return bridge


class RolloutZScoreTest(unittest.TestCase):
    def test_zscore_produces_unit_stats(self):
        torch.manual_seed(0)
        teacher = torch.randn(T, N) * 3.7 + 11.5
        normed, mean, std, too_small = apply_rollout_zscore(
            teacher, min_std=1e-6
        )
        self.assertFalse(too_small)
        self.assertAlmostEqual(float(normed.mean().item()), 0.0, places=5)
        self.assertAlmostEqual(
            float(normed.std(unbiased=False).item()), 1.0, places=4
        )
        # raw stats reported back exactly
        self.assertAlmostEqual(mean, float(teacher.mean().item()), places=5)
        self.assertAlmostEqual(
            std, float(teacher.std(unbiased=False).item()), places=5
        )

    def test_degenerate_std_returns_raw_and_flag(self):
        teacher = torch.full((T, N), 7.0)
        out, mean, std, too_small = apply_rollout_zscore(
            teacher, min_std=1e-3
        )
        self.assertTrue(too_small)
        self.assertEqual(std, 0.0)
        # Raw tensor returned unchanged.
        self.assertTrue(torch.equal(out, teacher))


class TeacherReturnCorrTest(unittest.TestCase):
    def test_perfect_linear_relationship(self):
        torch.manual_seed(1)
        returns = torch.randn(T, N)
        teacher = 2.5 * returns + 1.7
        corr, valid = compute_teacher_return_corr(teacher, returns)
        self.assertTrue(valid)
        self.assertAlmostEqual(corr, 1.0, places=5)

    def test_inverse_linear_relationship(self):
        torch.manual_seed(2)
        returns = torch.randn(T, N)
        teacher = -3.0 * returns
        corr, valid = compute_teacher_return_corr(teacher, returns)
        self.assertTrue(valid)
        self.assertAlmostEqual(corr, -1.0, places=5)

    def test_constant_teacher_is_invalid(self):
        torch.manual_seed(3)
        returns = torch.randn(T, N)
        teacher = torch.zeros(T, N)
        corr, valid = compute_teacher_return_corr(teacher, returns)
        self.assertFalse(valid)
        self.assertEqual(corr, 0.0)

    def test_constant_returns_is_invalid(self):
        teacher = torch.randn(T, N)
        returns = torch.zeros(T, N)
        corr, valid = compute_teacher_return_corr(teacher, returns)
        self.assertFalse(valid)
        self.assertEqual(corr, 0.0)

    def test_shape_mismatch_raises(self):
        teacher = torch.randn(T, N)
        returns = torch.randn(T + 1, N)
        with self.assertRaises(ValueError):
            compute_teacher_return_corr(teacher, returns)


class CacheMetaPopulatedTest(unittest.TestCase):
    """`build_teacher_cache_from_predictions` -> cache.meta is the diag log."""

    def test_meta_has_all_v2_diagnostic_keys(self):
        torch.manual_seed(4)
        teacher = torch.randn(T, N)
        returns = teacher * 0.9 + torch.randn(T, N) * 0.05
        cache = build_teacher_cache_from_predictions(
            teacher_value=teacher,
            teacher_success_logit=None,
            T=T, N=N, generation=1,
            teacher_value_norm=VALUE_NORM_ROLLOUT_ZSCORE,
            teacher_value_norm_min_std=1e-3,
            returns_for_corr=returns,
        )
        self.assertIn("teacher_value_mean", cache.meta)
        self.assertIn("teacher_value_std", cache.meta)
        self.assertIn("teacher_value_norm_enabled", cache.meta)
        self.assertIn("teacher_value_norm_std_too_small", cache.meta)
        self.assertIn("teacher_return_corr", cache.meta)
        self.assertIn("teacher_return_corr_valid", cache.meta)
        self.assertTrue(cache.meta["teacher_value_norm_enabled"])
        self.assertTrue(cache.meta["teacher_return_corr_valid"])
        self.assertGreater(cache.meta["teacher_return_corr"], 0.5)
        # post-norm flat tensor: mean ~ 0, std ~ 1
        flat = cache.teacher_value_flat
        self.assertAlmostEqual(float(flat.mean().item()), 0.0, places=4)
        self.assertAlmostEqual(
            float(flat.std(unbiased=False).item()), 1.0, places=4
        )

    def test_meta_records_degenerate_std(self):
        teacher = torch.full((T, N), 5.0)
        cache = build_teacher_cache_from_predictions(
            teacher_value=teacher,
            teacher_success_logit=None,
            T=T, N=N, generation=1,
            teacher_value_norm=VALUE_NORM_ROLLOUT_ZSCORE,
            teacher_value_norm_min_std=1e-3,
            returns_for_corr=torch.randn(T, N),
        )
        self.assertTrue(cache.meta["teacher_value_norm_std_too_small"])
        # Cache still has flat tensor (raw values), so the gate path
        # downstream can block deterministically without RuntimeError.
        self.assertIsNotNone(cache.teacher_value_flat)


class TeacherCorrGateBehaviorTest(unittest.TestCase):
    """gate=teacher_corr blocks low/invalid corr; gate=none never blocks."""

    def _setup(self, cfg: SeqPolicyCouplingConfig, teacher: torch.Tensor,
               returns: torch.Tensor) -> SeqDistillBridge:
        bridge = _v2_bridge_with_heads(cfg)
        bridge.cache = build_teacher_cache_from_predictions(
            teacher_value=teacher, teacher_success_logit=None,
            T=T, N=N, generation=1,
            teacher_value_norm=cfg.teacher_value_norm,
            returns_for_corr=returns,
        )
        bridge.agent_ref.dataset.last_range = (0, T)
        bridge.feature_capture = CapturedFeature(
            tensor=torch.randn(T, FEATURE_DIM, requires_grad=True), fired=True,
        )
        return bridge

    def test_high_corr_passes(self):
        torch.manual_seed(10)
        returns = torch.randn(T, N)
        teacher = 1.5 * returns + 0.1 * torch.randn(T, N)
        bridge = self._setup(_v2_cfg(), teacher, returns)
        result = compute_seq_distill_aux_loss(bridge)
        self.assertIn("seq_distill_value", result)
        self.assertTrue(bridge.last_gate_pass)
        self.assertFalse(bridge.last_gate_blocked)
        self.assertEqual(bridge.gate_blocked_count, 0)

    def test_low_corr_blocks(self):
        torch.manual_seed(11)
        returns = torch.randn(T, N)
        # Teacher independent of returns → corr ~ 0.
        teacher = torch.randn(T, N)
        bridge = self._setup(
            _v2_cfg(teacher_corr_threshold=0.99), teacher, returns,
        )
        result = compute_seq_distill_aux_loss(bridge)
        self.assertEqual(result, {})
        self.assertTrue(bridge.last_gate_blocked)
        self.assertFalse(bridge.last_gate_pass)
        self.assertEqual(bridge.gate_blocked_count, 1)
        self.assertFalse(bridge.last_active)

    def test_invalid_corr_blocks(self):
        teacher = torch.full((T, N), 3.0)  # constant -> corr invalid
        returns = torch.randn(T, N)
        bridge = self._setup(_v2_cfg(), teacher, returns)
        result = compute_seq_distill_aux_loss(bridge)
        self.assertEqual(result, {})
        self.assertTrue(bridge.last_gate_blocked)

    def test_gate_none_never_blocks(self):
        # gate=none on v2 emits a warning at config build but does not block
        torch.manual_seed(12)
        returns = torch.randn(T, N)
        teacher = torch.randn(T, N)  # corr ~ 0
        cfg = _v2_cfg(gate=GATE_NONE, teacher_corr_threshold=0.99)
        bridge = self._setup(cfg, teacher, returns)
        result = compute_seq_distill_aux_loss(bridge)
        self.assertIn("seq_distill_value", result)
        self.assertFalse(bridge.last_gate_blocked)


class DegenerateStdGateInteractionTest(unittest.TestCase):
    """Degenerate std + gate=teacher_corr -> blocks via invalid corr (no fail-fast)."""

    def test_degenerate_std_gates_off_safely(self):
        torch.manual_seed(13)
        teacher = torch.full((T, N), 4.0)
        returns = torch.randn(T, N)
        cfg = _v2_cfg(gate=GATE_TEACHER_CORR)
        bridge = _v2_bridge_with_heads(cfg)
        bridge.cache = build_teacher_cache_from_predictions(
            teacher_value=teacher, teacher_success_logit=None,
            T=T, N=N, generation=1,
            teacher_value_norm=VALUE_NORM_ROLLOUT_ZSCORE,
            returns_for_corr=returns,
        )
        bridge.agent_ref.dataset.last_range = (0, T)
        bridge.feature_capture = CapturedFeature(
            tensor=torch.randn(T, FEATURE_DIM), fired=True,
        )
        result = compute_seq_distill_aux_loss(bridge)
        self.assertEqual(result, {})
        self.assertTrue(bridge.last_gate_blocked)
        self.assertEqual(bridge.gate_blocked_count, 1)


if __name__ == "__main__":
    unittest.main()
