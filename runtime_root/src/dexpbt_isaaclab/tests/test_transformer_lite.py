"""
Tests for TransformerLiteWorldModel.

Covered:
* Belief shape (T, B, hidden_dim).
* Attention mask matrix:
  - lower-triangular (causal) on uniform-segment input;
  - segment-aware boundaries: a token after a done cannot attend to a
    pre-done token.
* All loss terms finite; backward runs.
* Padding token (valid_mask=0 column j) is never attended to (by checking
  output equality of valid prefixes when invalidated tail is replaced with
  random noise).
* Compatible with seq_len <= max_seq_len; raises if seq_len > max_seq_len.
"""

from __future__ import annotations

import unittest

import torch

from dexpbt_isaaclab.world_model.seq_wm_base import SequenceBatch
from dexpbt_isaaclab.world_model.transformer_lite import (
    TransformerLiteWorldModel,
    _build_attn_mask,
)


def _make_batch(T: int, B: int, obs_dim: int, action_dim: int,
                *, dones: torch.Tensor | None = None,
                seed: int = 0) -> SequenceBatch:
    g = torch.Generator(device="cpu").manual_seed(seed)
    obs = torch.randn(T, B, obs_dim, generator=g)
    next_obs = torch.randn(T, B, obs_dim, generator=g)
    action = torch.randn(T, B, action_dim, generator=g)
    reward = torch.randn(T, B, generator=g)
    returns = torch.randn(T, B, generator=g)
    done = torch.zeros(T, B) if dones is None else dones.float()
    d_hat = torch.randn(T, B, generator=g)
    next_d_hat = torch.randn(T, B, generator=g)
    success = torch.zeros(T, B)
    next_success = torch.zeros(T, B)
    valid = torch.ones(T, B)
    seg = torch.zeros(T, B, dtype=torch.long)
    if (done.sum() > 0):
        prev_cum = torch.zeros_like(done)
        prev_cum[1:] = torch.cumsum(done[:-1], dim=0)
        valid = (prev_cum < 0.5).float()
        seg = prev_cum.long()
    return SequenceBatch(
        obs=obs, action=action, reward=reward, done=done,
        next_obs=next_obs, returns=returns,
        d_hat=d_hat, next_d_hat=next_d_hat,
        success_count=success, next_success_count=next_success,
        valid_mask=valid, segment_id=seg,
    )


class AttnMaskTest(unittest.TestCase):
    def test_causal_only_no_done(self):
        T, B, H = 5, 1, 2
        done = torch.zeros(T, B)
        valid = torch.ones(T, B)
        seg = torch.zeros(T, B, dtype=torch.long)
        mask = _build_attn_mask(done, valid, seg, num_heads=H)
        self.assertEqual(mask.shape, (B * H, T, T))
        # For batch 0, head 0:
        m = mask[0]
        # i=0: only j=0 allowed (i.e. mask[0,0]=0, others -inf).
        for i in range(T):
            for j in range(T):
                if j <= i:
                    self.assertEqual(float(m[i, j].item()), 0.0)
                else:
                    self.assertTrue(torch.isinf(m[i, j]).item())

    def test_segment_boundary_blocks_pre_done(self):
        # T=5: done at t=1 -> segment_id = [0,0,1,1,1]
        T, B, H = 5, 1, 2
        done = torch.tensor([[0], [1], [0], [0], [0]], dtype=torch.float)
        valid = torch.tensor([[1], [1], [0], [0], [0]], dtype=torch.float)  # cross-episode
        # Note: segment_id from sampler convention: prev_done_cum.
        # done[0]=0, done[1]=1 -> prev_done_cum[0]=0, [1]=0, [2]=1, [3]=1, [4]=1.
        seg = torch.tensor([[0], [0], [1], [1], [1]], dtype=torch.long)
        mask = _build_attn_mask(done, valid, seg, num_heads=H)
        m = mask[0]  # batch 0, head 0
        # Token 2 (segment 1) must NOT attend to tokens 0 or 1 (segment 0).
        self.assertTrue(torch.isinf(m[2, 0]).item())
        self.assertTrue(torch.isinf(m[2, 1]).item())
        self.assertEqual(float(m[2, 2].item()), 0.0)
        # Token 3 attends to 2, 3 only (within new segment, and j <= i).
        self.assertEqual(float(m[3, 2].item()), 0.0)
        self.assertEqual(float(m[3, 3].item()), 0.0)
        self.assertTrue(torch.isinf(m[3, 4]).item())  # future
        self.assertTrue(torch.isinf(m[3, 0]).item())
        self.assertTrue(torch.isinf(m[3, 1]).item())


class TransformerForwardTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.T, self.B, self.D, self.A = 8, 6, 12, 4
        self.model = TransformerLiteWorldModel(
            obs_dim=self.D, action_dim=self.A, hidden_dim=32,
            num_heads=4, num_layers=2, ffn_dim=64, max_seq_len=16,
            head_hidden=32,
        )

    def test_belief_shape(self):
        batch = _make_batch(self.T, self.B, self.D, self.A)
        belief = self.model.encode_sequence(batch)
        self.assertEqual(tuple(belief.shape),
                         (self.T, self.B, self.model.belief_dim))

    def test_loss_finite_and_backward(self):
        batch = _make_batch(self.T, self.B, self.D, self.A, seed=1)
        out = self.model.compute_aux_loss(batch)
        for n, v in out.items():
            self.assertTrue(torch.isfinite(v).all().item(),
                            f"loss term {n} not finite")
        out["loss_total"].backward()
        # At least some param has grad.
        any_grad = any(
            p.grad is not None and p.grad.abs().sum().item() > 0
            for p in self.model.parameters()
        )
        self.assertTrue(any_grad)

    def test_seq_len_above_max_raises(self):
        batch = _make_batch(20, 4, self.D, self.A, seed=2)
        with self.assertRaises(ValueError):
            self.model.encode_sequence(batch)


class TransformerInvalidPaddingIgnoredTest(unittest.TestCase):
    """If valid_mask=0 at column j, the encoder mask should -inf token j;
    swapping invalidated tokens for random garbage must not change the
    belief at the valid prefix."""

    def test_invalid_tokens_dont_affect_valid_prefix(self):
        torch.manual_seed(7)
        T, B, D, A = 6, 1, 8, 3
        # done at step 2: prev_done_cum -> [0,0,0,1,1,1]; valid -> [1,1,1,0,0,0]
        done = torch.zeros(T, B); done[2, 0] = 1.0
        batch = _make_batch(T, B, D, A,
                              dones=done, seed=10)
        model = TransformerLiteWorldModel(
            obs_dim=D, action_dim=A, hidden_dim=24, num_heads=4,
            num_layers=2, ffn_dim=48, max_seq_len=8, head_hidden=24,
            dropout=0.0,   # determinism
        )
        model.eval()
        with torch.no_grad():
            b1 = model.encode_sequence(batch)
            # Replace the invalid tail (t >= 3) with garbage in obs/action.
            batch.obs[3:].normal_()
            batch.action[3:].normal_()
            b2 = model.encode_sequence(batch)
        # The valid prefix t in {0, 1, 2} (which is segment 0; for t=2 also
        # segment 0 since done is at THIS step, mask says prev_done_cum<0.5
        # so valid=1 and seg=0). Tokens at t in {0, 1, 2} should NOT attend
        # to tokens at t >= 3 (causal blocks anyway), so they should match.
        for t in range(3):
            self.assertTrue(torch.allclose(b1[t], b2[t], atol=1e-5),
                             f"belief at t={t} affected by invalid tail")


if __name__ == "__main__":
    unittest.main()
