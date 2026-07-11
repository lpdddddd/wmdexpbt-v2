"""Tests for SeqAuxCollector + SeqAuxInterceptorEnv."""

from __future__ import annotations

import unittest

import torch

from dexpbt_isaaclab.world_model.seq_aux_collector import (
    SeqAuxCollector,
    SeqAuxInterceptorEnv,
)


class _FakeEnv:
    """Mimics the env step contract enough for SeqAuxInterceptorEnv tests."""

    def __init__(self, num_envs: int):
        self.num_envs = num_envs
        self.lifted_object = torch.zeros(num_envs)
        self.successes = torch.zeros(num_envs)
        self._t = 0

    def step(self, actions):
        # Build a deterministic infos["wm"] dict.
        N = self.num_envs
        d_hat = torch.full((N, 1), float(self._t))
        failure = torch.zeros(N, 1)
        infos = {"wm": {"d_hat": d_hat, "failure": failure}}
        # Update env attrs the collector reads via env_attrs_provider.
        self.lifted_object = (torch.arange(N) % 2 == 0).float()
        self.successes = torch.full((N,), float(self._t))
        obs = torch.zeros(N, 4)
        rew = torch.zeros(N)
        term = torch.zeros(N)
        trunc = torch.zeros(N)
        self._t += 1
        return obs, rew, term, trunc, infos

    def reset(self):
        return torch.zeros(self.num_envs, 4), {}


def _env_attrs_provider(env: _FakeEnv) -> dict:
    return {"lifted_object": env.lifted_object, "successes": env.successes}


class CollectorWindowTest(unittest.TestCase):
    def test_full_window_shape(self):
        c = SeqAuxCollector(num_envs=4, horizon_length=8, device="cpu")
        for k, v in c.full_window().items():
            self.assertEqual(v.shape[:2], (8, 4))

    def test_step_writes_correct_index(self):
        c = SeqAuxCollector(num_envs=3, horizon_length=4, device="cpu")
        c.reset_window()
        env = _FakeEnv(3)
        wrap = SeqAuxInterceptorEnv(env, c, env_attrs_provider=_env_attrs_provider)
        for _ in range(4):
            wrap.step(torch.zeros(3, 1))
        full = c.full_window()
        # aux_d_hat[t, n, 0] == t (per fake env contract).
        for t in range(4):
            for n in range(3):
                self.assertAlmostEqual(
                    float(full["aux_d_hat"][t, n, 0].item()), float(t),
                )
        # aux_lifted: even envs -> 1.0, odd envs -> 0.0.
        self.assertTrue(torch.allclose(
            full["aux_lifted"][-1].view(-1),
            (torch.arange(3) % 2 == 0).float(),
        ))

    def test_record_terminal_updates_last_only(self):
        c = SeqAuxCollector(num_envs=2, horizon_length=3, device="cpu")
        env = _FakeEnv(2)
        wrap = SeqAuxInterceptorEnv(env, c, env_attrs_provider=_env_attrs_provider)
        wrap.step(torch.zeros(2, 1))
        wrap.step(torch.zeros(2, 1))
        wrap.step(torch.zeros(2, 1))
        last_before = {k: v.clone() for k, v in c.last_step().items()}
        # Build fresh infos with d_hat = 99.
        N = 2
        infos = {"wm": {"d_hat": torch.full((N, 1), 99.0),
                          "failure": torch.zeros(N, 1)}}
        c.record_terminal(infos, env_attrs={
            "lifted_object": torch.ones(N),
            "successes": torch.zeros(N),
        })
        last_after = c.last_step()
        # Window storage must NOT have changed (record_terminal only updates last_).
        self.assertTrue(torch.equal(
            c.full_window()["aux_d_hat"], c.full_window()["aux_d_hat"]
        ))
        # last["aux_d_hat"] now equals 99.
        self.assertTrue(torch.allclose(
            last_after["aux_d_hat"].view(-1),
            torch.full((N,), 99.0)
        ))
        # The pre-terminal last value at index N for d_hat was self._t - 1 = 2.
        self.assertTrue(torch.allclose(
            last_before["aux_d_hat"].view(-1), torch.full((N,), 2.0)
        ))

    def test_reset_window_starts_at_zero(self):
        c = SeqAuxCollector(num_envs=2, horizon_length=3, device="cpu")
        env = _FakeEnv(2)
        wrap = SeqAuxInterceptorEnv(env, c, env_attrs_provider=_env_attrs_provider)
        for _ in range(3):
            wrap.step(torch.zeros(2, 1))
        c.reset_window()
        # Step once after reset; that value must land at index 0 (overwrites).
        wrap.step(torch.zeros(2, 1))
        # The fake env returns d_hat = self._t which is now 3 (env has its
        # own counter); index 0 of storage should equal 3.0.
        self.assertAlmostEqual(
            float(c.full_window()["aux_d_hat"][0, 0, 0].item()), 3.0,
        )


if __name__ == "__main__":
    unittest.main()
