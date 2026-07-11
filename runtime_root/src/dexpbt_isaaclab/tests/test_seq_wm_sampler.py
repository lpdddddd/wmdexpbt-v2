"""
Tests for dexpbt_isaaclab.world_model.seq_wm_sampler.SequenceSampler.

Covered:
* Output shapes for seq_len ∈ {8, 16}, including aux fields and masks.
* seq_len > H raises ValueError.
* obs / action / reward / done values gathered from (H, N) match window
  positions exactly (no off-by-one inside the sampler — see test_seq_wm_target_alignment.py
  for the contract end-to-end).
* valid_mask = 0 strictly AFTER any done within the window.
* segment_id increments by 1 per done.
* next_obs[t < T-1, b] == obs[t+1, b] within window.
* next_obs[t = T-1, b] == last_obs[env_idx_b] when start_t + T == H.
* Aux fields are forwarded with correct (T, B) shape.
"""

from __future__ import annotations

import unittest

import torch

from dexpbt_isaaclab.world_model.seq_wm_sampler import SequenceSampler


def _make_random_buffers(
    H: int, N: int, obs_dim: int, action_dim: int, device: str = "cpu",
    seed: int = 0,
):
    g = torch.Generator(device=device).manual_seed(seed)
    obs_full = torch.randn(H, N, obs_dim, generator=g, device=device)
    action_full = torch.randn(H, N, action_dim, generator=g, device=device)
    reward_full = torch.randn(H, N, 1, generator=g, device=device)
    returns_full = torch.randn(H, N, 1, generator=g, device=device)
    last_obs = torch.randn(N, obs_dim, generator=g, device=device)
    # dones: zero by default; specific tests will fill known patterns.
    done_full = torch.zeros(H, N, device=device)
    d_hat_full = torch.randn(H, N, 1, generator=g, device=device)
    last_d_hat = torch.randn(N, generator=g, device=device)
    return dict(
        obs_full=obs_full,
        action_full=action_full,
        reward_full=reward_full,
        done_full=done_full,
        returns_full=returns_full,
        last_obs=last_obs,
        d_hat_full=d_hat_full,
        last_d_hat=last_d_hat,
    )


class SequenceSamplerShapeTest(unittest.TestCase):
    def test_shapes_seq_len_16(self):
        H, N, obs_dim, action_dim = 16, 32, 92, 23
        buffers = _make_random_buffers(H, N, obs_dim, action_dim)
        sampler = SequenceSampler(seq_len=16, seq_batch_size=24, device="cpu")
        batch = sampler.sample(**buffers)
        self.assertEqual(batch.obs.shape, (16, 24, obs_dim))
        self.assertEqual(batch.action.shape, (16, 24, action_dim))
        self.assertEqual(batch.reward.shape, (16, 24))
        self.assertEqual(batch.done.shape, (16, 24))
        self.assertEqual(batch.returns.shape, (16, 24))
        self.assertEqual(batch.next_obs.shape, (16, 24, obs_dim))
        self.assertEqual(batch.valid_mask.shape, (16, 24))
        self.assertEqual(batch.segment_id.shape, (16, 24))
        # aux scalar fields should have (T, B) shape (squeezed).
        self.assertEqual(batch.d_hat.shape, (16, 24))
        self.assertEqual(batch.next_d_hat.shape, (16, 24))

    def test_shapes_seq_len_8(self):
        H, N, obs_dim, action_dim = 16, 32, 92, 23
        buffers = _make_random_buffers(H, N, obs_dim, action_dim, seed=1)
        sampler = SequenceSampler(seq_len=8, seq_batch_size=12, device="cpu")
        batch = sampler.sample(**buffers)
        self.assertEqual(batch.obs.shape, (8, 12, obs_dim))
        self.assertEqual(batch.next_obs.shape, (8, 12, obs_dim))

    def test_seq_len_too_long_raises(self):
        buffers = _make_random_buffers(H=16, N=8, obs_dim=4, action_dim=3)
        sampler = SequenceSampler(seq_len=32, seq_batch_size=4, device="cpu")
        with self.assertRaises(ValueError):
            sampler.sample(**buffers)


class SequenceSamplerGatherTest(unittest.TestCase):
    """When seq_len == H, start_t is forced to 0, so the gather reduces to
    selecting B env columns and stacking them. We can verify exactly."""

    def test_gather_when_T_equals_H(self):
        H, N, obs_dim, action_dim = 16, 16, 5, 3
        buffers = _make_random_buffers(H, N, obs_dim, action_dim, seed=42)
        sampler = SequenceSampler(seq_len=16, seq_batch_size=8, device="cpu")
        # Drive RNG explicitly.
        torch.manual_seed(0)
        batch = sampler.sample(**buffers)

        # Re-derive (start_t=0; env_idx) from the meta.
        # We can verify: for some b, batch.obs[t, b, :] is in obs_full[t, e, :]
        # for some single e. Easier: assert each (T, b) column equals
        # obs_full[:, e, :] for SOME e by exact comparison.
        T, B, D = batch.obs.shape
        for b in range(B):
            # Find the env index whose full-window obs matches column b exactly.
            obs_col = batch.obs[:, b, :]                       # (T, D)
            diffs = (buffers["obs_full"] - obs_col.unsqueeze(1)).abs().sum(
                dim=(0, 2)
            )                                                  # (N,)
            best_env = int(diffs.argmin().item())
            self.assertLess(diffs[best_env].item(), 1e-6,
                            f"column {b} obs not exact match to any env")
            # Action / reward must come from the SAME env.
            self.assertTrue(
                torch.allclose(batch.action[:, b, :],
                               buffers["action_full"][:, best_env, :])
            )
            self.assertTrue(
                torch.allclose(batch.reward[:, b],
                               buffers["reward_full"][:, best_env, 0])
            )

    def test_next_obs_uses_last_obs_when_T_equals_H(self):
        H, N, obs_dim, action_dim = 16, 16, 5, 3
        buffers = _make_random_buffers(H, N, obs_dim, action_dim, seed=7)
        sampler = SequenceSampler(seq_len=16, seq_batch_size=8, device="cpu")
        batch = sampler.sample(**buffers)
        T = batch.T
        # For t < T-1, next_obs[t,b] must equal obs[t+1, b].
        self.assertTrue(torch.allclose(batch.next_obs[:T-1], batch.obs[1:T]))
        # For t == T-1, next_obs[T-1, b] must equal last_obs[env_b]; we
        # verify by matching against last_obs over all envs (the env_idx
        # used for column b is the one whose obs[0] matches batch.obs[0,b]).
        for b in range(batch.B):
            obs_first = batch.obs[0, b]
            diffs = (buffers["obs_full"][0] - obs_first).abs().sum(dim=-1)
            env_b = int(diffs.argmin().item())
            self.assertTrue(
                torch.allclose(batch.next_obs[T-1, b], buffers["last_obs"][env_b])
            )


class SequenceSamplerMaskTest(unittest.TestCase):
    def test_valid_mask_zero_after_done(self):
        # Construct deterministic dones: env 0 has a single done at step 5,
        # env 1 has dones at steps 3 and 10, env 2 has none.
        H, N, obs_dim, action_dim = 16, 4, 3, 2
        buffers = _make_random_buffers(H, N, obs_dim, action_dim, seed=11)
        buffers["done_full"][:] = 0.0
        buffers["done_full"][5, 0] = 1.0
        buffers["done_full"][3, 1] = 1.0
        buffers["done_full"][10, 1] = 1.0
        # env 3: no dones. env 2: no dones.

        # Force seq_len == H so windows are deterministic per env.
        sampler = SequenceSampler(seq_len=H, seq_batch_size=8, device="cpu")
        batch = sampler.sample(**buffers)

        # For each column b, identify env then check valid_mask & segment_id.
        for b in range(batch.B):
            obs_first = batch.obs[0, b]
            diffs = (buffers["obs_full"][0] - obs_first).abs().sum(dim=-1)
            env_b = int(diffs.argmin().item())
            done_col = buffers["done_full"][:, env_b]
            # valid_mask[0]==1; valid_mask[t] for t>0 is 1 iff cumulative
            # done over [0..t-1] is 0.
            for t in range(1, H):
                cum = float(done_col[:t].sum().item())
                want_valid = 1.0 if cum < 0.5 else 0.0
                self.assertAlmostEqual(
                    float(batch.valid_mask[t, b].item()), want_valid,
                    places=6,
                    msg=f"valid_mask[t={t}, b={b}, env={env_b}] expected {want_valid}",
                )
                want_seg = int(cum)
                self.assertEqual(
                    int(batch.segment_id[t, b].item()), want_seg,
                    f"segment_id[t={t}, b={b}, env={env_b}] expected {want_seg}",
                )
            # First step always valid.
            self.assertEqual(float(batch.valid_mask[0, b].item()), 1.0)
            self.assertEqual(int(batch.segment_id[0, b].item()), 0)


class SequenceSamplerOptionalAuxTest(unittest.TestCase):
    """Sampler must tolerate missing aux fields (no d_hat, no failure)."""

    def test_no_aux_fields(self):
        H, N, obs_dim, action_dim = 16, 8, 4, 2
        buffers = _make_random_buffers(H, N, obs_dim, action_dim, seed=2)
        # Drop d_hat / last_d_hat.
        buffers["d_hat_full"] = None
        buffers["last_d_hat"] = None
        sampler = SequenceSampler(seq_len=16, seq_batch_size=4, device="cpu")
        batch = sampler.sample(**buffers)
        self.assertIsNone(batch.d_hat)
        self.assertIsNone(batch.next_d_hat)
        self.assertIsNone(batch.failure)
        self.assertIsNone(batch.success_count)
        self.assertIsNone(batch.lifted)
        # Mandatory fields still present.
        self.assertEqual(batch.obs.shape, (16, 4, obs_dim))
        self.assertEqual(batch.next_obs.shape, (16, 4, obs_dim))


if __name__ == "__main__":
    unittest.main()
