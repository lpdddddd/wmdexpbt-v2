"""
Strict off-by-one alignment tests for SequenceSampler.

We build (H, N) tensors whose entries encode (step_index, env_index, field)
unambiguously, then assert that the gathered (T, B) batch satisfies the
plan's target-alignment contract:

    obs[t, b]      = s_t                from obs_full[start_t + t, env_b]
    action[t, b]   = a_t                from action_full[start_t + t, env_b]
    reward[t, b]   = r_t                from reward_full[start_t + t, env_b]
    done[t, b]     = done_t              from done_full[start_t + t, env_b]
    d_hat[t, b]    = d_hat_t             from d_hat_full[start_t + t, env_b]
    next_obs[t,b]  = s_{t+1}             obs_full[start_t + t + 1, env_b]
                                          OR last_obs[env_b] when t+1 == H
    next_d_hat[t,b]= d_hat_{t+1}         (same rule, with last_d_hat)

In particular, reward[t, b] must NOT equal reward_full[t-1, env_b] or
reward_full[t+1, env_b]: this catches accidental shifts.

We also verify the effective state-delta loss mask
    eff = valid_mask * (1 - done)
flips to 0 exactly at done steps and at all steps after a done within the
window (cross-episode protection).
"""

from __future__ import annotations

import unittest

import torch

from dexpbt_isaaclab.world_model.seq_wm_sampler import SequenceSampler


def _encode_step_env(step: int, env: int, salt: float) -> float:
    """Deterministic distinct float per (step, env, field-salt)."""
    # Avoid collisions across all H, N, salts used in tests.
    return float(step) * 1000.0 + float(env) + salt * 1e-3


def _build_aligned_buffers(H: int, N: int, obs_dim: int, action_dim: int,
                            done_envs: dict[int, list[int]] | None = None):
    """Construct (H, N, *) tensors with unambiguous entries.

    obs[t, n, k]      = encode(t, n, salt=obs+k)
    action[t, n, k]   = encode(t, n, salt=act+k)
    reward[t, n, 0]   = encode(t, n, salt=rew)     # (H, N, 1)
    returns[t, n, 0]  = encode(t, n, salt=ret)
    d_hat[t, n, 0]    = encode(t, n, salt=dh)
    done[t, n]        = 1 if (t in done_envs.get(n, [])) else 0
    last_obs[n, k]    = encode(H, n, salt=obs+k)   # 'step H' value
    last_d_hat[n]     = encode(H, n, salt=dh)
    """
    obs = torch.zeros(H, N, obs_dim)
    action = torch.zeros(H, N, action_dim)
    reward = torch.zeros(H, N, 1)
    returns = torch.zeros(H, N, 1)
    d_hat = torch.zeros(H, N, 1)
    done = torch.zeros(H, N)
    for t in range(H):
        for n in range(N):
            for k in range(obs_dim):
                obs[t, n, k] = _encode_step_env(t, n, salt=10 + k)
            for k in range(action_dim):
                action[t, n, k] = _encode_step_env(t, n, salt=200 + k)
            reward[t, n, 0] = _encode_step_env(t, n, salt=300)
            returns[t, n, 0] = _encode_step_env(t, n, salt=400)
            d_hat[t, n, 0] = _encode_step_env(t, n, salt=500)
    if done_envs:
        for n, steps in done_envs.items():
            for t in steps:
                done[t, n] = 1.0

    last_obs = torch.zeros(N, obs_dim)
    last_d_hat = torch.zeros(N)
    for n in range(N):
        for k in range(obs_dim):
            last_obs[n, k] = _encode_step_env(H, n, salt=10 + k)
        last_d_hat[n] = _encode_step_env(H, n, salt=500)

    return dict(
        obs_full=obs, action_full=action, reward_full=reward,
        returns_full=returns, d_hat_full=d_hat, done_full=done,
        last_obs=last_obs, last_d_hat=last_d_hat,
    )


def _recover_env_idx(batch_obs_first: torch.Tensor, obs_full_first_step: torch.Tensor) -> int:
    """Given batch.obs[0, b, :], find the env index in obs_full[0, :, :] it
    came from (exact match)."""
    diffs = (obs_full_first_step - batch_obs_first).abs().sum(dim=-1)
    return int(diffs.argmin().item())


def _recover_env_idx_window(
    batch_obs_col: torch.Tensor,    # (T, D)
    obs_full: torch.Tensor,         # (H, N, D)
    start_t: int,
) -> int:
    """Given a column of batch.obs and the window start, find the env idx."""
    T = batch_obs_col.shape[0]
    target = obs_full[start_t:start_t + T, :, :]   # (T, N, D)
    diffs = (target - batch_obs_col.unsqueeze(1)).abs().sum(dim=(0, 2))
    return int(diffs.argmin().item())


class TargetAlignmentH16T16(unittest.TestCase):
    """When seq_len == H, start_t = 0 for all columns."""

    def test_per_step_alignment(self):
        H, N, D, A = 16, 8, 3, 2
        # env 1 has done at step 4; env 5 has dones at 2 and 9.
        bufs = _build_aligned_buffers(H, N, D, A,
                                       done_envs={1: [4], 5: [2, 9]})
        sampler = SequenceSampler(seq_len=H, seq_batch_size=8, device="cpu")
        batch = sampler.sample(**bufs)
        T = batch.T
        B = batch.B

        for b in range(B):
            env_b = _recover_env_idx(batch.obs[0, b], bufs["obs_full"][0])
            for t in range(T):
                # obs[t,b] == obs_full[t, env_b]
                self.assertTrue(torch.allclose(
                    batch.obs[t, b], bufs["obs_full"][t, env_b]
                ), f"obs mismatch t={t} b={b} env={env_b}")
                # action[t,b] == action_full[t, env_b]
                self.assertTrue(torch.allclose(
                    batch.action[t, b], bufs["action_full"][t, env_b]
                ))
                # reward[t,b] == reward_full[t, env_b, 0]
                self.assertAlmostEqual(
                    float(batch.reward[t, b].item()),
                    float(bufs["reward_full"][t, env_b, 0].item()),
                    places=4,
                )
                # done[t,b] == done_full[t, env_b]
                self.assertEqual(
                    float(batch.done[t, b].item()),
                    float(bufs["done_full"][t, env_b].item()),
                )
                # d_hat[t,b] == d_hat_full[t, env_b, 0]
                self.assertAlmostEqual(
                    float(batch.d_hat[t, b].item()),
                    float(bufs["d_hat_full"][t, env_b, 0].item()),
                    places=4,
                )

    def test_no_off_by_one_in_reward(self):
        """reward[t, b] must NOT equal reward_full[t-1] or [t+1] (env_b)."""
        H, N, D, A = 16, 4, 2, 2
        bufs = _build_aligned_buffers(H, N, D, A)
        sampler = SequenceSampler(seq_len=H, seq_batch_size=4, device="cpu")
        batch = sampler.sample(**bufs)
        for b in range(batch.B):
            env_b = _recover_env_idx(batch.obs[0, b], bufs["obs_full"][0])
            for t in range(1, batch.T - 1):
                got = float(batch.reward[t, b].item())
                want = float(bufs["reward_full"][t, env_b, 0].item())
                shifted_left = float(bufs["reward_full"][t - 1, env_b, 0].item())
                shifted_right = float(bufs["reward_full"][t + 1, env_b, 0].item())
                self.assertNotAlmostEqual(got, shifted_left, places=4,
                                          msg=f"reward[{t},{b}] = reward_full[t-1] (off-by-one)")
                self.assertNotAlmostEqual(got, shifted_right, places=4,
                                          msg=f"reward[{t},{b}] = reward_full[t+1] (off-by-one)")
                self.assertAlmostEqual(got, want, places=4)

    def test_next_obs_alignment(self):
        H, N, D, A = 16, 4, 3, 2
        bufs = _build_aligned_buffers(H, N, D, A)
        sampler = SequenceSampler(seq_len=H, seq_batch_size=4, device="cpu")
        batch = sampler.sample(**bufs)
        # T == H here, so for t < T-1: next_obs[t,b] == obs_full[t+1, env_b]
        # for t == T-1: next_obs[T-1, b] == last_obs[env_b]
        for b in range(batch.B):
            env_b = _recover_env_idx(batch.obs[0, b], bufs["obs_full"][0])
            for t in range(batch.T - 1):
                self.assertTrue(torch.allclose(
                    batch.next_obs[t, b], bufs["obs_full"][t + 1, env_b]
                ), f"next_obs t={t} b={b}")
            self.assertTrue(torch.allclose(
                batch.next_obs[batch.T - 1, b], bufs["last_obs"][env_b]
            ), f"next_obs t=T-1 b={b} should equal last_obs")

    def test_next_d_hat_alignment(self):
        H, N, D, A = 16, 4, 2, 2
        bufs = _build_aligned_buffers(H, N, D, A)
        sampler = SequenceSampler(seq_len=H, seq_batch_size=4, device="cpu")
        batch = sampler.sample(**bufs)
        for b in range(batch.B):
            env_b = _recover_env_idx(batch.obs[0, b], bufs["obs_full"][0])
            for t in range(batch.T - 1):
                self.assertAlmostEqual(
                    float(batch.next_d_hat[t, b].item()),
                    float(bufs["d_hat_full"][t + 1, env_b, 0].item()),
                    places=4,
                )
            self.assertAlmostEqual(
                float(batch.next_d_hat[batch.T - 1, b].item()),
                float(bufs["last_d_hat"][env_b].item()),
                places=4,
            )


class TargetAlignmentH16T8(unittest.TestCase):
    """When seq_len < H, start_t may be > 0."""

    def test_window_aligned_with_random_start(self):
        H, N, D, A = 16, 8, 3, 2
        bufs = _build_aligned_buffers(H, N, D, A)
        sampler = SequenceSampler(seq_len=8, seq_batch_size=12, device="cpu")
        batch = sampler.sample(**bufs)
        T = batch.T
        for b in range(batch.B):
            # Find env index from the full window match (start_t unknown).
            # We try every possible start_t in [0, H-T].
            obs_col = batch.obs[:, b, :]  # (T, D)
            best = None
            for s in range(H - T + 1):
                target = bufs["obs_full"][s:s + T]   # (T, N, D)
                diff = (target - obs_col.unsqueeze(1)).abs().sum(dim=(0, 2))
                m = int(diff.argmin().item())
                if diff[m] < 1e-6:
                    best = (s, m)
                    break
            self.assertIsNotNone(best, f"could not align column b={b}")
            start_t, env_b = best
            # All t in [0, T) must match obs_full[start_t + t, env_b]
            for t in range(T):
                self.assertTrue(torch.allclose(
                    batch.obs[t, b], bufs["obs_full"][start_t + t, env_b]
                ))
                self.assertAlmostEqual(
                    float(batch.reward[t, b].item()),
                    float(bufs["reward_full"][start_t + t, env_b, 0].item()),
                    places=4,
                )
                self.assertEqual(
                    float(batch.done[t, b].item()),
                    float(bufs["done_full"][start_t + t, env_b].item()),
                )


class EffectiveStateLossMask(unittest.TestCase):
    """state_delta / task_error_delta loss should use eff = valid * (1 - done).

    The effective mask flips to 0:
      - at the done step itself, AND
      - at every step strictly AFTER the first done in the window.
    """

    def test_eff_mask_zero_at_done_and_after(self):
        H, N, D, A = 16, 4, 2, 2
        bufs = _build_aligned_buffers(H, N, D, A,
                                       done_envs={2: [5], 3: [3, 9]})
        sampler = SequenceSampler(seq_len=H, seq_batch_size=8, device="cpu")
        batch = sampler.sample(**bufs)
        eff = batch.valid_mask * (1.0 - batch.done)

        for b in range(batch.B):
            env_b = _recover_env_idx(batch.obs[0, b], bufs["obs_full"][0])
            done_col = bufs["done_full"][:, env_b]
            seen_done = False
            for t in range(batch.T):
                cum_before = float(done_col[:t].sum().item())
                want_valid = 1.0 if cum_before < 0.5 else 0.0
                want_eff = want_valid * (1.0 - float(done_col[t].item()))
                self.assertAlmostEqual(
                    float(eff[t, b].item()), want_eff, places=6,
                    msg=f"eff mask wrong t={t} b={b} env={env_b}",
                )


if __name__ == "__main__":
    unittest.main()
