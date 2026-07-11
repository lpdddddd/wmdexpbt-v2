"""Phase A4 P1 (v2 coupling) -- warmup + post-warmup linear ramp.

Plan section 7: pre-warmup -> {} (warmup_active=1, effective_weight=0).
Post-warmup, ramp_factor goes linearly 0 -> 1 over `ramp_frames`. Once
ramp >= 1 the loss is fully scaled by `weight`. Gate blocking composes
multiplicatively: gate_blocked -> effective_weight=0 even at full ramp.
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
    build_seq_aux_critic_subtower,
    build_seq_aux_value_head,
    build_teacher_cache_from_predictions,
    compute_seq_distill_aux_loss,
)


T = 4
N = 3
FEATURE_DIM = 8

WARMUP_FRAMES = 100
RAMP_FRAMES = 100


class _FakeDataset:
    def __init__(self) -> None:
        self.last_range: Optional[tuple[int, int]] = None


class _FakeAgent:
    def __init__(self) -> None:
        self.dataset = _FakeDataset()


def _make_bridge(
    *,
    gate: str = GATE_NONE,
    weight: float = 0.05,
    teacher_value: Optional[torch.Tensor] = None,
    returns: Optional[torch.Tensor] = None,
) -> SeqDistillBridge:
    cfg = SeqPolicyCouplingConfig(
        mode=COUPLING_CRITIC_DISTILL_VALUE_NORM,
        weight=weight,
        value_weight=1.0,
        success_weight=0.0,
        warmup_frames=WARMUP_FRAMES,
        ramp_frames=RAMP_FRAMES,
        feature_source=FEATURE_SOURCE_SHARED_TO_AUX_CRITIC,
        teacher_value_norm=VALUE_NORM_ROLLOUT_ZSCORE,
        gate=gate,
        teacher_corr_threshold=0.2,
    )
    bridge = SeqDistillBridge(cfg=cfg)
    bridge.rollout_generation = 1
    bridge.agent_ref = _FakeAgent()
    bridge.seq_aux_value_head = build_seq_aux_value_head(in_dim=FEATURE_DIM)
    bridge.seq_aux_critic_mlp = build_seq_aux_critic_subtower(
        in_dim=FEATURE_DIM, hidden=FEATURE_DIM
    )
    if teacher_value is None:
        torch.manual_seed(7)
        teacher_value = torch.randn(T, N)
    bridge.cache = build_teacher_cache_from_predictions(
        teacher_value=teacher_value, teacher_success_logit=None,
        T=T, N=N, generation=1,
        teacher_value_norm=VALUE_NORM_ROLLOUT_ZSCORE,
        returns_for_corr=returns,
    )
    bridge.agent_ref.dataset.last_range = (0, T)
    bridge.feature_capture = CapturedFeature(
        tensor=torch.randn(T, FEATURE_DIM, requires_grad=True), fired=True,
    )
    return bridge


class WarmupTest(unittest.TestCase):
    def test_pre_warmup_returns_empty(self):
        bridge = _make_bridge()
        bridge.last_frames = WARMUP_FRAMES - 1
        result = compute_seq_distill_aux_loss(bridge)
        self.assertEqual(result, {})
        self.assertTrue(bridge.last_warmup_active)
        self.assertFalse(bridge.last_active)
        self.assertEqual(bridge.last_effective_weight, 0.0)

    def test_at_warmup_with_ramp_returns_empty(self):
        # frames == warmup -> ramp_factor == 0 -> still {}.
        bridge = _make_bridge()
        bridge.last_frames = WARMUP_FRAMES
        result = compute_seq_distill_aux_loss(bridge)
        self.assertEqual(result, {})
        self.assertEqual(bridge.last_ramp_factor, 0.0)


class RampScheduleTest(unittest.TestCase):
    def test_half_ramp_gives_half_effective_weight(self):
        bridge = _make_bridge(weight=0.1)
        bridge.last_frames = WARMUP_FRAMES + RAMP_FRAMES // 2
        result = compute_seq_distill_aux_loss(bridge)
        self.assertIn("seq_distill_value", result)
        self.assertAlmostEqual(bridge.last_ramp_factor, 0.5, places=5)
        self.assertAlmostEqual(
            bridge.last_effective_weight, 0.05, places=5
        )

    def test_full_ramp_gives_full_weight(self):
        bridge = _make_bridge(weight=0.05)
        bridge.last_frames = WARMUP_FRAMES + RAMP_FRAMES
        result = compute_seq_distill_aux_loss(bridge)
        self.assertIn("seq_distill_value", result)
        self.assertAlmostEqual(bridge.last_ramp_factor, 1.0, places=5)
        self.assertAlmostEqual(
            bridge.last_effective_weight, 0.05, places=5
        )

    def test_post_full_ramp_clamps_to_one(self):
        bridge = _make_bridge(weight=0.07)
        bridge.last_frames = WARMUP_FRAMES + RAMP_FRAMES * 100
        result = compute_seq_distill_aux_loss(bridge)
        self.assertIn("seq_distill_value", result)
        self.assertAlmostEqual(bridge.last_ramp_factor, 1.0, places=5)
        self.assertAlmostEqual(
            bridge.last_effective_weight, 0.07, places=5
        )

    def test_ramp_frames_zero_means_immediate(self):
        cfg = SeqPolicyCouplingConfig(
            mode=COUPLING_CRITIC_DISTILL_VALUE_NORM,
            weight=0.05, warmup_frames=WARMUP_FRAMES, ramp_frames=0,
            feature_source=FEATURE_SOURCE_SHARED_TO_AUX_CRITIC,
            teacher_value_norm=VALUE_NORM_ROLLOUT_ZSCORE,
            gate=GATE_NONE,
        )
        bridge = SeqDistillBridge(cfg=cfg)
        bridge.rollout_generation = 1
        bridge.agent_ref = _FakeAgent()
        bridge.seq_aux_value_head = build_seq_aux_value_head(in_dim=FEATURE_DIM)
        bridge.seq_aux_critic_mlp = build_seq_aux_critic_subtower(
            in_dim=FEATURE_DIM, hidden=FEATURE_DIM
        )
        torch.manual_seed(8)
        bridge.cache = build_teacher_cache_from_predictions(
            teacher_value=torch.randn(T, N), teacher_success_logit=None,
            T=T, N=N, generation=1,
            teacher_value_norm=VALUE_NORM_ROLLOUT_ZSCORE,
        )
        bridge.agent_ref.dataset.last_range = (0, T)
        bridge.feature_capture = CapturedFeature(
            tensor=torch.randn(T, FEATURE_DIM), fired=True,
        )
        # 1 frame past warmup, ramp_frames==0 -> ramp_factor=1.
        bridge.last_frames = WARMUP_FRAMES + 1
        result = compute_seq_distill_aux_loss(bridge)
        self.assertIn("seq_distill_value", result)
        self.assertAlmostEqual(bridge.last_ramp_factor, 1.0, places=5)
        self.assertAlmostEqual(bridge.last_effective_weight, 0.05, places=5)


class GateAndRampCompositionTest(unittest.TestCase):
    """gate_blocked -> {} regardless of ramp_factor."""

    def test_gate_blocks_during_warmup_returns_empty(self):
        torch.manual_seed(9)
        # uncorrelated teacher / returns -> gate would block IF reached.
        teacher = torch.randn(T, N)
        returns = torch.randn(T, N)
        bridge = _make_bridge(
            gate=GATE_TEACHER_CORR, teacher_value=teacher, returns=returns,
        )
        bridge.cfg.teacher_corr_threshold = 0.99
        bridge.last_frames = WARMUP_FRAMES - 1  # warmup wins first
        result = compute_seq_distill_aux_loss(bridge)
        self.assertEqual(result, {})
        # warmup gate triggers before the corr gate is checked.
        self.assertTrue(bridge.last_warmup_active)

    def test_gate_blocks_after_warmup_zeroes_weight(self):
        torch.manual_seed(9)
        teacher = torch.randn(T, N)
        returns = torch.randn(T, N)
        bridge = _make_bridge(
            gate=GATE_TEACHER_CORR, teacher_value=teacher, returns=returns,
        )
        bridge.cfg.teacher_corr_threshold = 0.99
        bridge.last_frames = WARMUP_FRAMES + RAMP_FRAMES * 5
        result = compute_seq_distill_aux_loss(bridge)
        self.assertEqual(result, {})
        self.assertTrue(bridge.last_gate_blocked)
        self.assertEqual(bridge.last_effective_weight, 0.0)
        self.assertFalse(bridge.last_active)


if __name__ == "__main__":
    unittest.main()
