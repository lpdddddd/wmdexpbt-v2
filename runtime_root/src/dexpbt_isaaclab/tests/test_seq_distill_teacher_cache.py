"""Phase A4 P0 — teacher cache + minibatch alignment tests.

Mirrors rl_games' ``swap_and_flatten01`` layout exactly:

    teacher: shape (T, N) per-step value
    flat:    shape (T*N,) where  flat[n * T + t] == teacher[t, n]

A PPO RNN minibatch with ``last_range = (start, end)`` slices the flat
dataset contiguously, so the teacher targets for that minibatch are
``flat[start:end]``. This file pins that contract with very small T, N.

Constraints from the user:
* Use ``T=4, N=3, teacher_flat = arange(T*N)`` to make off-by-one shifts
  visible by inspection.
* ``compute_seq_distill_aux_loss`` must fail-fast when:
    - the bridge has no live agent reference,
    - ``agent.dataset.last_range`` is missing / malformed,
    - the captured ``actor_mlp_out`` shape disagrees with the slice,
    - the teacher cache generation does not match the live rollout.
"""

from __future__ import annotations

import unittest
from dataclasses import dataclass
from typing import Optional

import torch

from dexpbt_isaaclab.world_model.seq_policy_aux import (
    CapturedFeature,
    COUPLING_DISTILL_VALUE,
    COUPLING_NONE,
    SeqDistillBridge,
    SeqPolicyCouplingConfig,
    build_seq_aux_value_head,
    build_teacher_cache_from_predictions,
    compute_seq_distill_aux_loss,
)


T = 4
N = 3
TOTAL = T * N  # 12


def _teacher_for_arange_flat() -> torch.Tensor:
    """Return teacher_value (T, N) such that the flat layout is arange(T*N)."""
    # flat[n * T + t] = n * T + t. Build teacher s.t. teacher[t, n] = n*T + t.
    flat = torch.arange(TOTAL, dtype=torch.float32)
    teacher = flat.view(N, T).transpose(0, 1).contiguous()  # (T, N)
    return teacher


class _FakeDataset:
    """Bare minimum stand-in for rl_games PPODataset.last_range."""

    def __init__(self) -> None:
        self.last_range: Optional[tuple[int, int]] = None


class _FakeAgent:
    def __init__(self) -> None:
        self.dataset = _FakeDataset()


class TeacherCacheLayoutTest(unittest.TestCase):
    def test_flat_matches_arange(self):
        teacher = _teacher_for_arange_flat()
        cache = build_teacher_cache_from_predictions(
            teacher_value=teacher,
            teacher_success_logit=None,
            T=T, N=N, generation=1,
        )
        self.assertIsNotNone(cache.teacher_value_flat)
        self.assertEqual(int(cache.teacher_value_flat.numel()), TOTAL)
        # The whole point: this MUST equal arange(T*N).
        self.assertTrue(torch.equal(
            cache.teacher_value_flat,
            torch.arange(TOTAL, dtype=torch.float32),
        ))

    def test_per_index_alignment_matches_swap_and_flatten01(self):
        """flat[n*T + t] == teacher[t, n] for every (t, n) pair.

        This pins the layout contract used by rl_games swap_and_flatten01.
        """
        teacher = torch.arange(TOTAL, dtype=torch.float32).view(T, N) * 1.5 + 7.0
        cache = build_teacher_cache_from_predictions(
            teacher_value=teacher,
            teacher_success_logit=None,
            T=T, N=N, generation=2,
        )
        flat = cache.teacher_value_flat
        for n in range(N):
            for t in range(T):
                idx = n * T + t
                self.assertAlmostEqual(
                    float(flat[idx].item()),
                    float(teacher[t, n].item()),
                    places=5,
                )

    def test_minibatch_slice_returns_correct_window(self):
        """Simulate a PPO minibatch: last_range=(start, end) → flat[start:end]
        must equal the teacher samples for envs [gstart, gend) × all T steps.
        """
        teacher = _teacher_for_arange_flat()
        cache = build_teacher_cache_from_predictions(
            teacher_value=teacher,
            teacher_success_logit=None,
            T=T, N=N, generation=3,
        )
        # Pretend num_games_batch=1, seq_length=T → minibatch_size = T.
        for env_idx in range(N):
            start = env_idx * T
            end = start + T
            slice_ = cache.teacher_value_flat[start:end]
            # Environment env_idx contributes [t=0..T-1] which in the
            # arange flat layout is values [env_idx*T .. env_idx*T+T).
            expected = torch.arange(start, end, dtype=torch.float32)
            self.assertTrue(torch.equal(slice_, expected))

    def test_shape_mismatch_raises(self):
        bad_teacher = torch.zeros(T + 1, N)  # wrong T
        with self.assertRaises(ValueError):
            build_teacher_cache_from_predictions(
                teacher_value=bad_teacher,
                teacher_success_logit=None,
                T=T, N=N, generation=4,
            )


class MinibatchSliceFailFastTest(unittest.TestCase):
    """compute_seq_distill_aux_loss must NEVER silently fall back."""

    def _make_bridge(self, has_value_head: bool = True, mode: str = COUPLING_DISTILL_VALUE) -> SeqDistillBridge:
        cfg = SeqPolicyCouplingConfig(
            mode=mode,
            weight=1.0,
            value_weight=1.0,
            success_weight=0.0,
            warmup_frames=0,
        )
        bridge = SeqDistillBridge(cfg=cfg)
        bridge.rollout_generation = 1
        cache = build_teacher_cache_from_predictions(
            teacher_value=_teacher_for_arange_flat(),
            teacher_success_logit=None,
            T=T, N=N, generation=1,
        )
        bridge.cache = cache
        if has_value_head:
            bridge.seq_aux_value_head = build_seq_aux_value_head(in_dim=8)
        bridge.last_frames = 1_000_000_000  # past warmup
        # Capture the right "feature" shape for one minibatch:
        bridge.feature_capture = CapturedFeature(tensor=torch.randn(T, 8), fired=True)
        agent = _FakeAgent()
        bridge.agent_ref = agent
        agent.dataset.last_range = (0, T)  # first env's window
        return bridge

    def test_missing_agent_ref_raises(self):
        bridge = self._make_bridge()
        bridge.agent_ref = None
        with self.assertRaisesRegex(RuntimeError, "agent_ref is None"):
            compute_seq_distill_aux_loss(bridge)

    def test_missing_last_range_raises(self):
        bridge = self._make_bridge()
        bridge.agent_ref.dataset.last_range = None
        with self.assertRaisesRegex(RuntimeError, "last_range missing"):
            compute_seq_distill_aux_loss(bridge)

    def test_malformed_last_range_raises(self):
        bridge = self._make_bridge()
        bridge.agent_ref.dataset.last_range = (5,)  # not 2-tuple
        with self.assertRaisesRegex(RuntimeError, "last_range missing or malformed"):
            compute_seq_distill_aux_loss(bridge)

    def test_invalid_range_raises(self):
        bridge = self._make_bridge()
        bridge.agent_ref.dataset.last_range = (5, 5)
        with self.assertRaisesRegex(RuntimeError, "invalid minibatch range"):
            compute_seq_distill_aux_loss(bridge)

    def test_feature_capture_not_fired_raises(self):
        bridge = self._make_bridge()
        bridge.feature_capture = CapturedFeature(tensor=None, fired=False)
        with self.assertRaisesRegex(RuntimeError, "did not capture features"):
            compute_seq_distill_aux_loss(bridge)

    def test_shape_mismatch_raises(self):
        bridge = self._make_bridge()
        # last_range expects 4 samples, but capture has 5
        bridge.feature_capture = CapturedFeature(tensor=torch.randn(5, 8), fired=True)
        with self.assertRaisesRegex(
            RuntimeError, "captured feature batch .* != minibatch slice size"
        ):
            compute_seq_distill_aux_loss(bridge)

    def test_generation_mismatch_raises(self):
        bridge = self._make_bridge()
        bridge.rollout_generation = 99  # cache.generation is 1
        with self.assertRaisesRegex(RuntimeError, "generation .* does not match"):
            compute_seq_distill_aux_loss(bridge)

    def test_warmup_returns_empty(self):
        bridge = self._make_bridge()
        bridge.last_frames = 0
        bridge.cfg.warmup_frames = 10_000_000
        result = compute_seq_distill_aux_loss(bridge)
        self.assertEqual(result, {})
        self.assertTrue(bridge.last_warmup_active)
        self.assertFalse(bridge.last_active)

    def test_coupling_none_returns_empty(self):
        bridge = self._make_bridge(mode=COUPLING_NONE)
        result = compute_seq_distill_aux_loss(bridge)
        self.assertEqual(result, {})
        self.assertFalse(bridge.last_active)

    def test_happy_path_returns_value_loss(self):
        bridge = self._make_bridge()
        result = compute_seq_distill_aux_loss(bridge)
        self.assertIn("seq_distill_value", result)
        self.assertTrue(bridge.last_active)
        self.assertEqual(bridge.last_indices, (0, T))


class ArrangedSliceCorrectnessTest(unittest.TestCase):
    """End-to-end: targets read inside compute_seq_distill_aux_loss must
    equal the swap_and_flatten01 slice (the whole point of the patch)."""

    def test_target_matches_arange_slice(self):
        cfg = SeqPolicyCouplingConfig(
            mode=COUPLING_DISTILL_VALUE,
            weight=1.0,
            value_weight=1.0,
            success_weight=0.0,
            warmup_frames=0,
        )
        bridge = SeqDistillBridge(cfg=cfg)
        bridge.rollout_generation = 7
        bridge.cache = build_teacher_cache_from_predictions(
            teacher_value=_teacher_for_arange_flat(),
            teacher_success_logit=None,
            T=T, N=N, generation=7,
        )
        # Make the head a no-op identity so V_aux == feat[:, 0] (we just
        # want to check the target slice, not learn anything).
        head = torch.nn.Linear(1, 1)
        with torch.no_grad():
            head.weight.fill_(1.0)
            head.bias.fill_(0.0)
        bridge.seq_aux_value_head = head

        # Stage two minibatches: env 0 then env 2.
        agent = _FakeAgent()
        bridge.agent_ref = agent
        bridge.last_frames = 10**9  # past warmup

        for env_idx in (0, 2):
            start = env_idx * T
            end = start + T
            agent.dataset.last_range = (start, end)
            # Feature is just a 1-d feature so head(feat) reduces to feat
            # We want V_aux to equal expected so loss = 0 confirms slice ok.
            expected_slice = bridge.cache.teacher_value_flat[start:end]
            feat = expected_slice.unsqueeze(-1).clone()
            bridge.feature_capture = CapturedFeature(tensor=feat, fired=True)
            losses = compute_seq_distill_aux_loss(bridge)
            self.assertIn("seq_distill_value", losses)
            self.assertAlmostEqual(
                float(losses["seq_distill_value"].detach().item()),
                0.0, places=5,
                msg=f"expected zero distill loss for env_idx={env_idx} when "
                    f"V_aux == teacher_slice; mismatch implies wrong slice.",
            )


if __name__ == "__main__":
    unittest.main()
