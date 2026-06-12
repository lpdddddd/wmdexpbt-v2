"""Tests for the EIR-MPC RSSM belief cache (full MPPI seed)."""

from __future__ import annotations

import types
import unittest

import torch
import torch.nn as nn

from dexpbt_isaaclab.world_model.rssm_lite import RSSMLiteWorldModel
from dexpbt_isaaclab.world_model.rssm_belief_provider import (
    build_rssm_belief_cache,
    RSSMBeliefCache,
    BELIEF_OK,
    BELIEF_NOT_RSSM,
    BELIEF_DISABLED,
)


T, N, OBS, ACT = 6, 4, 12, 4


def _exp_buffer(*, with_task_aux=False):
    td = {
        "obses": torch.randn(T, N, OBS),
        "actions": torch.randn(T, N, ACT).clamp(-1, 1),
        "rewards": torch.randn(T, N, 1),
        "dones": torch.zeros(T, N, 1),
        "aux_d_hat": torch.rand(T, N, 1),
        "aux_failure": torch.zeros(T, N, 1),
        "aux_lifted": torch.ones(T, N, 1),
        "aux_success_cnt": torch.zeros(T, N, 1),
    }
    if with_task_aux:
        td.update(
            {
                "aux_success_tolerance": torch.full((T, N, 1), 0.02),
                "aux_picked": torch.zeros(T, N, 1),
                "aux_dropped": torch.zeros(T, N, 1),
                "aux_closest_d_hat": torch.rand(T, N, 1),
            }
        )
    return types.SimpleNamespace(tensor_dict=td)


def _rssm(*, aux_dim=0):
    return RSSMLiteWorldModel(
        obs_dim=OBS, action_dim=ACT, obs_embed_dim=32, action_embed_dim=16,
        h_dim=24, z_dim=8, head_hidden=32, aux_dim=aux_dim, aux_embed_dim=16,
    )


def _trainer(model):
    return types.SimpleNamespace(model=model, device=torch.device("cpu"))


class TestBeliefCache(unittest.TestCase):
    def test_build_shape(self):
        model = _rssm()
        cache = build_rssm_belief_cache(
            seq_wm_trainer=_trainer(model), experience_buffer=_exp_buffer(),
            returns_full=torch.randn(T, N), last_obs=torch.randn(N, OBS),
            last_d_hat=torch.rand(N, 1), last_success_count=torch.zeros(N, 1),
            frames=1000, generation=2, device=torch.device("cpu"),
        )
        self.assertTrue(cache.valid)
        self.assertEqual(cache.fallback_reason, BELIEF_OK)
        self.assertEqual(cache.rows, T * N)
        self.assertEqual(int(cache.belief_flat.shape[0]), T * N)
        self.assertEqual(cache.belief_dim, model.belief_dim)

    def test_build_with_task_aux_scalars(self):
        model = _rssm(aux_dim=12)
        cache = build_rssm_belief_cache(
            seq_wm_trainer=_trainer(model),
            experience_buffer=_exp_buffer(with_task_aux=True),
            returns_full=torch.randn(T, N),
            last_obs=torch.randn(N, OBS),
            last_d_hat=torch.rand(N, 1),
            last_success_count=torch.zeros(N, 1),
            frames=1000,
            generation=2,
            device=torch.device("cpu"),
        )
        self.assertTrue(cache.valid)
        self.assertEqual(cache.fallback_reason, BELIEF_OK)

    def test_rssm_belief_cache_with_aux_dim_12_valid(self):
        model = RSSMLiteWorldModel(
            obs_dim=OBS, action_dim=ACT, obs_embed_dim=32, action_embed_dim=16,
            h_dim=24, z_dim=8, head_hidden=32, aux_dim=12, aux_embed_dim=16,
            with_horizon_heads=True, with_recovery_head=True,
        )
        cache = build_rssm_belief_cache(
            seq_wm_trainer=_trainer(model),
            experience_buffer=_exp_buffer(with_task_aux=True),
            returns_full=torch.randn(T, N),
            last_obs=torch.randn(N, OBS),
            last_d_hat=torch.rand(N, 1),
            last_success_count=torch.zeros(N, 1),
            frames=1000,
            generation=2,
            device=torch.device("cpu"),
        )
        self.assertTrue(cache.valid)
        self.assertEqual(cache.fallback_reason, BELIEF_OK)
        self.assertEqual(int(cache.belief_flat.shape[0]), T * N)
        self.assertEqual(cache.belief_dim, model.belief_dim)

    def test_rssm_belief_cache_flatten_alignment_still_holds_with_aux(self):
        class _StubModel(nn.Module):
            belief_dim = 3

            def get_state_from_belief(self, b):
                return b

            class _Heads:
                pass

            heads = _Heads()
            aux_dim = 12

            def encode_sequence(self, batch, deterministic=False):
                T_, N_ = batch.T, batch.B
                assert batch.aux_scalars is not None
                assert batch.aux_scalars.shape[-1] == 12
                out = torch.zeros(T_, N_, self.belief_dim)
                for t in range(T_):
                    for n in range(N_):
                        out[t, n, 0] = float(n * T_ + t)
                return out

        model = _StubModel()
        cache = build_rssm_belief_cache(
            seq_wm_trainer=_trainer(model),
            experience_buffer=_exp_buffer(with_task_aux=True),
            returns_full=torch.randn(T, N),
            last_obs=torch.randn(N, OBS),
            last_d_hat=torch.rand(N, 1),
            last_success_count=torch.zeros(N, 1),
            frames=0,
            generation=0,
            device=torch.device("cpu"),
        )
        self.assertTrue(cache.valid)
        for t in range(T):
            for n in range(N):
                self.assertAlmostEqual(
                    float(cache.belief_flat[n * T + t, 0]), float(n * T + t)
                )

    def test_flatten_alignment(self):
        # Stub model whose encode_sequence writes belief_seq[t,n,0]=n*T+t so we
        # can verify flat[n*T+t] picks the right row.
        class _StubModel(nn.Module):
            belief_dim = 3

            def get_state_from_belief(self, b):  # marks it RSSM-like
                return b

            class _Heads:
                pass

            heads = _Heads()

            def encode_sequence(self, batch, deterministic=False):
                T_, N_ = batch.T, batch.B
                out = torch.zeros(T_, N_, self.belief_dim)
                for t in range(T_):
                    for n in range(N_):
                        out[t, n, 0] = float(n * T_ + t)
                return out

        model = _StubModel()
        cache = build_rssm_belief_cache(
            seq_wm_trainer=_trainer(model), experience_buffer=_exp_buffer(),
            returns_full=torch.randn(T, N), last_obs=torch.randn(N, OBS),
            last_d_hat=None, last_success_count=None,
            frames=0, generation=0, device=torch.device("cpu"),
        )
        self.assertTrue(cache.valid)
        for t in range(T):
            for n in range(N):
                self.assertAlmostEqual(
                    float(cache.belief_flat[n * T + t, 0]), float(n * T + t)
                )

    def test_fallback_on_non_rssm(self):
        # model without get_state_from_belief -> NOT_RSSM
        class _Enc(nn.Module):
            heads = object()

            def encode_sequence(self, batch, deterministic=False):
                return torch.randn(batch.T, batch.B, 5)

        cache = build_rssm_belief_cache(
            seq_wm_trainer=_trainer(_Enc()), experience_buffer=_exp_buffer(),
            returns_full=torch.randn(T, N), last_obs=torch.randn(N, OBS),
            last_d_hat=None, last_success_count=None,
            frames=0, generation=0, device=torch.device("cpu"),
        )
        self.assertFalse(cache.valid)
        self.assertEqual(cache.fallback_reason, BELIEF_NOT_RSSM)
        self.assertIsNone(cache.slice_rows(0, 4))

    def test_disabled(self):
        cache = build_rssm_belief_cache(
            seq_wm_trainer=_trainer(_rssm()), experience_buffer=_exp_buffer(),
            returns_full=torch.randn(T, N), last_obs=torch.randn(N, OBS),
            last_d_hat=None, last_success_count=None,
            frames=0, generation=0, device=torch.device("cpu"), enabled=False,
        )
        self.assertFalse(cache.valid)
        self.assertEqual(cache.fallback_reason, BELIEF_DISABLED)

    def test_no_grad(self):
        model = _rssm()
        cache = build_rssm_belief_cache(
            seq_wm_trainer=_trainer(model), experience_buffer=_exp_buffer(),
            returns_full=torch.randn(T, N), last_obs=torch.randn(N, OBS),
            last_d_hat=None, last_success_count=None,
            frames=0, generation=0, device=torch.device("cpu"),
        )
        self.assertTrue(cache.valid)
        self.assertFalse(cache.belief_flat.requires_grad)

    def test_slice_rows_bounds(self):
        model = _rssm()
        cache = build_rssm_belief_cache(
            seq_wm_trainer=_trainer(model), experience_buffer=_exp_buffer(),
            returns_full=torch.randn(T, N), last_obs=torch.randn(N, OBS),
            last_d_hat=None, last_success_count=None,
            frames=0, generation=0, device=torch.device("cpu"),
        )
        self.assertIsNotNone(cache.slice_rows(0, N))
        self.assertEqual(int(cache.slice_rows(0, N).shape[0]), N)
        self.assertIsNone(cache.slice_rows(0, T * N + 1))  # out of bounds


if __name__ == "__main__":
    unittest.main()
