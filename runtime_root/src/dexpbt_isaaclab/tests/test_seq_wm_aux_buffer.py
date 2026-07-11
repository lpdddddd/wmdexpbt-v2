"""
Tests for dexpbt_isaaclab.world_model.wm_aux_buffer helpers.

We avoid importing rl_games (heavy dep) by faking a buffer object that
exposes the same .tensor_dict / .obs_base_shape / .device / .update_data
contract.
"""

from __future__ import annotations

import unittest

import torch

from dexpbt_isaaclab.world_model.wm_aux_buffer import (
    DEFAULT_SEQ_AUX_FIELDS,
    extract_step_aux,
    insert_step_aux,
    read_aux_full,
    register_seq_aux_fields,
)


class _FakeBuffer:
    """Minimal stand-in for rl_games ExperienceBuffer."""

    def __init__(self, T: int, N: int, device: str = "cpu"):
        self.tensor_dict: dict[str, torch.Tensor] = {}
        self.obs_base_shape = (T, N)
        self.device = device

    def update_data(self, name: str, index: int, val: torch.Tensor) -> None:
        # Mirror rl_games' update_data semantics: write `val` at row `index`.
        stored = self.tensor_dict[name]
        if val.ndim < stored.ndim - 1:
            val = val.unsqueeze(-1)
        stored[index].copy_(val)


class RegisterFieldsTest(unittest.TestCase):
    def test_default_fields_allocate_with_correct_shape(self):
        T, N = 16, 8
        buf = _FakeBuffer(T, N)
        new_keys = register_seq_aux_fields(buf)
        self.assertEqual(set(new_keys), set(DEFAULT_SEQ_AUX_FIELDS.keys()))
        for k, shape in DEFAULT_SEQ_AUX_FIELDS.items():
            self.assertIn(k, buf.tensor_dict)
            self.assertEqual(tuple(buf.tensor_dict[k].shape), (T, N) + shape)
            self.assertEqual(buf.tensor_dict[k].dtype, torch.float32)

    def test_register_idempotent(self):
        buf = _FakeBuffer(16, 8)
        register_seq_aux_fields(buf)
        new_keys_2 = register_seq_aux_fields(buf)
        self.assertEqual(new_keys_2, [], "second call should re-register nothing")

    def test_custom_fields(self):
        buf = _FakeBuffer(8, 4)
        new_keys = register_seq_aux_fields(buf, fields={"foo": (3,), "bar": (1,)})
        self.assertIn("foo", buf.tensor_dict)
        self.assertEqual(buf.tensor_dict["foo"].shape, (8, 4, 3))
        self.assertEqual(buf.tensor_dict["bar"].shape, (8, 4, 1))


class ExtractInsertReadTest(unittest.TestCase):
    def test_extract_from_infos_wm(self):
        N = 6
        infos = {
            "wm": {
                "d_hat": torch.arange(N, dtype=torch.float32).unsqueeze(-1),  # (N,1)
                "failure": torch.zeros(N, 1),
            },
            "log": {"successes": 0.42},  # not per-env, ignored
        }
        env_attrs = {
            "lifted_object": torch.tensor([1, 0, 1, 0, 1, 0], dtype=torch.bool),
            "successes": torch.tensor([0, 1, 1, 2, 0, 0], dtype=torch.long),
        }
        aux = extract_step_aux(infos, device="cpu", env_attrs=env_attrs)
        self.assertIn("aux_d_hat", aux)
        self.assertEqual(aux["aux_d_hat"].shape, (N, 1))
        self.assertTrue(torch.allclose(
            aux["aux_d_hat"].view(-1), torch.arange(N, dtype=torch.float32)
        ))
        self.assertEqual(aux["aux_failure"].shape, (N, 1))
        self.assertEqual(aux["aux_lifted"].shape, (N, 1))
        self.assertEqual(aux["aux_success_cnt"].shape, (N, 1))
        # lifted_object must round-trip through float.
        self.assertTrue(torch.equal(
            aux["aux_lifted"].view(-1).bool(),
            env_attrs["lifted_object"],
        ))

    def test_extract_handles_missing_wm(self):
        # No "wm" in infos -> nothing extracted from infos side, but
        # env_attrs-based fields still work.
        env_attrs = {"lifted_object": torch.zeros(3)}
        aux = extract_step_aux({}, device="cpu", env_attrs=env_attrs)
        self.assertIn("aux_lifted", aux)
        self.assertNotIn("aux_d_hat", aux)
        self.assertNotIn("aux_failure", aux)

    def test_insert_step_aux_writes_correct_row(self):
        T, N = 4, 3
        buf = _FakeBuffer(T, N)
        register_seq_aux_fields(buf)
        # Step t=2: write known values for d_hat and failure.
        d_hat = torch.tensor([[0.1], [0.2], [0.3]])
        fail = torch.tensor([[1.0], [0.0], [1.0]])
        insert_step_aux(buf, 2, {"aux_d_hat": d_hat, "aux_failure": fail})
        self.assertTrue(torch.allclose(buf.tensor_dict["aux_d_hat"][2], d_hat))
        self.assertTrue(torch.allclose(buf.tensor_dict["aux_failure"][2], fail))
        # Other rows untouched (still zero).
        self.assertTrue(torch.all(buf.tensor_dict["aux_d_hat"][0] == 0))
        self.assertTrue(torch.all(buf.tensor_dict["aux_d_hat"][3] == 0))

    def test_insert_unknown_field_silently_skipped(self):
        T, N = 2, 2
        buf = _FakeBuffer(T, N)
        register_seq_aux_fields(buf, fields={"aux_d_hat": (1,)})
        insert_step_aux(buf, 0, {
            "aux_d_hat": torch.ones(N, 1),
            "aux_does_not_exist": torch.ones(N, 1),
        })
        # No exception. Known field updated, unknown ignored.
        self.assertTrue(torch.all(buf.tensor_dict["aux_d_hat"][0] == 1.0))
        self.assertNotIn("aux_does_not_exist", buf.tensor_dict)

    def test_read_aux_full_returns_none_for_missing(self):
        buf = _FakeBuffer(4, 2)
        self.assertIsNone(read_aux_full(buf, "aux_d_hat"))
        register_seq_aux_fields(buf)
        t = read_aux_full(buf, "aux_d_hat")
        self.assertIsNotNone(t)
        self.assertEqual(t.shape, (4, 2, 1))


if __name__ == "__main__":
    unittest.main()
