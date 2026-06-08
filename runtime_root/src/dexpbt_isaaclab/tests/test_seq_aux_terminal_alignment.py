"""Phase A4 P0 Task C — terminal aux alignment end-to-end test.

Production path under audit:

    SeqAuxCollector.on_step writes both ``storage[t]`` and ``_last`` on
    every env step, but ``record_terminal`` was *never* called in
    production. That left ``_last`` set to whatever the very last
    ``on_step`` saw, with no positive confirmation that the value
    corresponds to the post-rollout (step H) snapshot.

The Phase A4 P0 fix:

    1. ``SeqAuxInterceptorEnv`` now retains ``last_infos`` and
       ``last_env_attrs`` from its most recent ``step()`` call.
    2. The patched ``play_steps_rnn`` in ``train.py`` calls
       ``record_terminal(infos=wrap.last_infos, env_attrs=wrap.last_env_attrs)``
       after the rollout returns.

What this test pins:

    * ``wrap.last_infos`` / ``wrap.last_env_attrs`` are populated on each
      step (None before the first step).
    * ``record_terminal`` updates ``_last`` (consistent with whatever the
      wrap saw) but does NOT touch ``storage``.
    * If production ``record_terminal`` is fed a *different* infos than
      the in-loop on_step (e.g. a fresh env-attrs read after the
      rollout), ``_last`` reflects the fresh value while storage stays
      put: this is the "no off-by-one" guarantee.
"""

from __future__ import annotations

import unittest

import torch

from dexpbt_isaaclab.world_model.seq_aux_collector import (
    SeqAuxCollector,
    SeqAuxInterceptorEnv,
)


class _CountingEnv:
    """Each step returns infos["wm"]["d_hat"] = self._t (then increments)."""

    def __init__(self, num_envs: int):
        self.num_envs = num_envs
        self.lifted_object = torch.zeros(num_envs)
        self.successes = torch.zeros(num_envs)
        self._t = 0

    def step(self, actions):
        N = self.num_envs
        d_hat = torch.full((N, 1), float(self._t))
        infos = {"wm": {"d_hat": d_hat, "failure": torch.zeros(N, 1)}}
        self.successes = torch.full((N,), float(self._t))
        self._t += 1
        return torch.zeros(N, 4), torch.zeros(N), torch.zeros(N), torch.zeros(N), infos

    def reset(self):
        return torch.zeros(self.num_envs, 4), {}


def _env_attrs_provider(env: _CountingEnv) -> dict:
    return {"lifted_object": env.lifted_object, "successes": env.successes}


class WrapRetainsLastInfosTest(unittest.TestCase):
    def test_starts_none(self):
        N = 2
        c = SeqAuxCollector(num_envs=N, horizon_length=3, device="cpu")
        env = _CountingEnv(N)
        wrap = SeqAuxInterceptorEnv(env, c, env_attrs_provider=_env_attrs_provider)
        self.assertIsNone(wrap.last_infos)
        self.assertIsNone(wrap.last_env_attrs)

    def test_updates_on_every_step(self):
        N = 2
        c = SeqAuxCollector(num_envs=N, horizon_length=3, device="cpu")
        env = _CountingEnv(N)
        wrap = SeqAuxInterceptorEnv(env, c, env_attrs_provider=_env_attrs_provider)
        for expected in range(3):
            wrap.step(torch.zeros(N, 1))
            self.assertIsNotNone(wrap.last_infos)
            self.assertEqual(
                float(wrap.last_infos["wm"]["d_hat"][0, 0].item()),
                float(expected),
            )
            self.assertIsNotNone(wrap.last_env_attrs)


class RecordTerminalUsesWrapInfosTest(unittest.TestCase):
    """Production play_steps_rnn flow:

    for _ in range(H):
        wrap.step(...)                            # storage filled, _last set
    collector.record_terminal(
        infos=wrap.last_infos,
        env_attrs=wrap.last_env_attrs,
    )                                             # P0 P0 patch
    """

    def test_record_terminal_does_not_touch_storage(self):
        H, N = 4, 2
        c = SeqAuxCollector(num_envs=N, horizon_length=H, device="cpu")
        c.reset_window()
        env = _CountingEnv(N)
        wrap = SeqAuxInterceptorEnv(env, c, env_attrs_provider=_env_attrs_provider)

        for _ in range(H):
            wrap.step(torch.zeros(N, 1))

        storage_before = {k: v.clone() for k, v in c.full_window().items()}
        c.record_terminal(infos=wrap.last_infos, env_attrs=wrap.last_env_attrs)
        for k, v in storage_before.items():
            self.assertTrue(
                torch.equal(c.full_window()[k], v),
                msg=f"record_terminal mutated storage[{k}]",
            )

    def test_record_terminal_updates_last_consistent_with_last_infos(self):
        H, N = 4, 2
        c = SeqAuxCollector(num_envs=N, horizon_length=H, device="cpu")
        c.reset_window()
        env = _CountingEnv(N)
        wrap = SeqAuxInterceptorEnv(env, c, env_attrs_provider=_env_attrs_provider)

        for _ in range(H):
            wrap.step(torch.zeros(N, 1))

        # Build expected post-record value from the wrap snapshot.
        expected = float(wrap.last_infos["wm"]["d_hat"][0, 0].item())
        c.record_terminal(infos=wrap.last_infos, env_attrs=wrap.last_env_attrs)
        last = c.last_step()
        for n in range(N):
            self.assertEqual(float(last["aux_d_hat"][n, 0].item()), expected)

    def test_record_terminal_with_fresh_infos_diverges_from_storage(self):
        """When called with infos DIFFERENT from the last on_step (e.g. a
        post-rollout env-attrs read), _last must diverge from storage[H-1].
        This proves the patch can correct any off-by-one if env semantics
        change later — it's not a no-op safety net.
        """
        H, N = 4, 2
        c = SeqAuxCollector(num_envs=N, horizon_length=H, device="cpu")
        c.reset_window()
        env = _CountingEnv(N)
        wrap = SeqAuxInterceptorEnv(env, c, env_attrs_provider=_env_attrs_provider)
        for _ in range(H):
            wrap.step(torch.zeros(N, 1))
        storage_h_minus_one = c.full_window()["aux_d_hat"][H - 1, 0, 0].clone()

        # Pretend a post-rollout state read produces 99.0 — this could be
        # because the env exposes a fresher computation that isn't visible
        # via on_step (e.g. derived from current obs only, not infos).
        N_ = N
        fresh_infos = {
            "wm": {
                "d_hat": torch.full((N_, 1), 99.0),
                "failure": torch.zeros(N_, 1),
            }
        }
        fresh_env_attrs = {
            "lifted_object": torch.ones(N_),
            "successes": torch.zeros(N_),
        }
        c.record_terminal(infos=fresh_infos, env_attrs=fresh_env_attrs)
        last = c.last_step()
        # _last == 99 (fresh), storage[H-1] still == H-1 (= 3 here).
        for n in range(N):
            self.assertEqual(float(last["aux_d_hat"][n, 0].item()), 99.0)
        self.assertEqual(
            float(c.full_window()["aux_d_hat"][H - 1, 0, 0].item()),
            float(storage_h_minus_one.item()),
        )


if __name__ == "__main__":
    unittest.main()
