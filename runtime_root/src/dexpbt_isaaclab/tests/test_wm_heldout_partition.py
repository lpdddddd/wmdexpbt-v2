"""Tests for stable WM held-out env-slot partition."""

from __future__ import annotations

import unittest

import torch

from dexpbt_isaaclab.world_model.rssm_lite import RSSMLiteWorldModel
from dexpbt_isaaclab.world_model.seq_wm_sampler import SequenceSampler
from dexpbt_isaaclab.world_model.seq_wm_trainer import SeqWMConfig, SeqWMTrainer
from dexpbt_isaaclab.world_model.wm_aux_buffer import register_seq_aux_fields
from dexpbt_isaaclab.world_model.wm_calibration_evaluator import (
    WMCalibrationConfig,
    WMCalibrationEvaluator,
    WMCalibrationState,
    apply_calibration_outcome,
)
from dexpbt_isaaclab.world_model.wm_head_probs import (
    FAILURE_INSUFFICIENT_HELDOUT_SAMPLES,
    FAILURE_INVALID_PARTITION,
)
from dexpbt_isaaclab.world_model.wm_heldout_partition import (
    build_env_partition_masks,
    stable_u32,
    stable_unit_interval,
)


class TestStablePartition(unittest.TestCase):
    def test_partition_deterministic_same_seed_rank_nenv(self):
        a = build_env_partition_masks(
            64, heldout_fraction=0.2, partition_seed=7, worker_rank=3
        )
        b = build_env_partition_masks(
            64, heldout_fraction=0.2, partition_seed=7, worker_rank=3
        )
        self.assertTrue(torch.equal(a.heldout_mask, b.heldout_mask))
        self.assertTrue(torch.equal(a.train_mask, b.train_mask))
        self.assertTrue(a.valid)

    def test_checkpoint_restart_same_partition(self):
        part = build_env_partition_masks(
            32, heldout_fraction=0.2, partition_seed=11, worker_rank=2
        )
        st = WMCalibrationState(
            heldout_fraction=0.2,
            heldout_partition_seed=11,
            partition_worker_rank=2,
        )
        st2 = WMCalibrationState.from_dict(st.to_dict())
        part2 = build_env_partition_masks(
            32,
            heldout_fraction=st2.heldout_fraction,
            partition_seed=st2.heldout_partition_seed,
            worker_rank=st2.partition_worker_rank,
        )
        self.assertTrue(torch.equal(part.heldout_mask, part2.heldout_mask))

    def test_different_worker_rank_not_identical(self):
        a = build_env_partition_masks(
            128, heldout_fraction=0.2, partition_seed=0, worker_rank=0
        )
        b = build_env_partition_masks(
            128, heldout_fraction=0.2, partition_seed=0, worker_rank=1
        )
        self.assertFalse(torch.equal(a.heldout_mask, b.heldout_mask))

    def test_train_heldout_mutex_and_union(self):
        part = build_env_partition_masks(
            40, heldout_fraction=0.25, partition_seed=1, worker_rank=0
        )
        self.assertTrue(part.valid)
        self.assertEqual(int((part.train_mask & part.heldout_mask).sum().item()), 0)
        self.assertTrue(bool((part.train_mask | part.heldout_mask).all()))
        self.assertEqual(part.diagnostics["wm_cal/partition_overlap_count"], 0.0)
        self.assertGreater(part.diagnostics["wm_cal/heldout_env_count"], 0.0)
        self.assertGreater(part.diagnostics["wm_cal/train_env_count"], 0.0)

    def test_hash_not_python_builtin(self):
        # Stable across calls; independent of PYTHONHASHSEED.
        self.assertEqual(stable_u32(1, 2, 3), stable_u32(1, 2, 3))
        self.assertNotEqual(stable_u32(1, 2, 3), stable_u32(1, 3, 2))
        u = stable_unit_interval(0, 0, 0)
        self.assertGreaterEqual(u, 0.0)
        self.assertLess(u, 1.0)

    def test_heldout_fraction_zero_legacy(self):
        part = build_env_partition_masks(
            16, heldout_fraction=0.0, partition_seed=0, worker_rank=0
        )
        self.assertTrue(part.valid)
        self.assertEqual(part.diagnostics["wm_cal/heldout_enabled"], 0.0)
        self.assertTrue(bool(part.train_mask.all()))
        self.assertFalse(bool(part.heldout_mask.any()))

    def test_invalid_fraction_structural(self):
        part = build_env_partition_masks(
            16, heldout_fraction=1.0, partition_seed=0, worker_rank=0
        )
        self.assertFalse(part.valid)
        self.assertEqual(part.failure_reason_id, FAILURE_INVALID_PARTITION)
        st = WMCalibrationState(alpha_wm=0.9, ema_raw=0.9, calibration_pass=True)
        cfg = WMCalibrationConfig(enabled=True, alpha_min=0.1)
        st = apply_calibration_outcome(
            st,
            cfg=cfg,
            frames=1,
            metrics=dict(part.diagnostics),
            mode="structural",
            failure_reason_id=FAILURE_INVALID_PARTITION,
        )
        self.assertEqual(st.alpha_wm, 0.0)
        self.assertEqual(st.ema_raw, 0.0)
        self.assertEqual(
            st.last_metrics["wm_cal/failure_reason_id"], FAILURE_INVALID_PARTITION
        )

    def test_insufficient_heldout_ema_decay(self):
        cfg = WMCalibrationConfig(enabled=True, ema_beta=0.9, alpha_min=0.0)
        st = WMCalibrationState(alpha_wm=0.8, ema_raw=0.8)
        st = apply_calibration_outcome(
            st,
            cfg=cfg,
            frames=5,
            metrics={},
            mode="insufficient_heldout",
            failure_reason_id=FAILURE_INSUFFICIENT_HELDOUT_SAMPLES,
        )
        self.assertAlmostEqual(st.ema_raw, 0.9 * 0.8, places=6)
        self.assertEqual(
            st.last_metrics["wm_cal/failure_reason_id"],
            FAILURE_INSUFFICIENT_HELDOUT_SAMPLES,
        )


class TestSamplerPartitionMask(unittest.TestCase):
    def _buffers(self, H=8, N=20, obs_dim=4, action_dim=2, seed=0):
        g = torch.Generator().manual_seed(seed)
        return dict(
            obs_full=torch.randn(H, N, obs_dim, generator=g),
            action_full=torch.randn(H, N, action_dim, generator=g),
            reward_full=torch.randn(H, N, generator=g),
            done_full=torch.zeros(H, N),
            returns_full=torch.randn(H, N, generator=g),
            last_obs=torch.randn(N, obs_dim, generator=g),
        )

    def test_train_mask_excludes_heldout_ids(self):
        part = build_env_partition_masks(
            20, heldout_fraction=0.3, partition_seed=3, worker_rank=1
        )
        sampler = SequenceSampler(seq_len=8, seq_batch_size=32, device="cpu")
        torch.manual_seed(0)
        batch = sampler.sample(
            **self._buffers(),
            env_index_mask=part.train_mask,
            return_source_ids=True,
            partition_worker_rank=1,
        )
        env_ids = batch.meta["env_indices"].long()
        self.assertFalse(bool(part.heldout_mask[env_ids].any()))
        self.assertTrue(bool(part.train_mask[env_ids].all()))

    def test_heldout_mask_excludes_train_ids(self):
        part = build_env_partition_masks(
            20, heldout_fraction=0.3, partition_seed=3, worker_rank=1
        )
        sampler = SequenceSampler(seq_len=8, seq_batch_size=32, device="cpu")
        torch.manual_seed(1)
        batch = sampler.sample(
            **self._buffers(),
            env_index_mask=part.heldout_mask,
            return_source_ids=True,
        )
        env_ids = batch.meta["env_indices"].long()
        self.assertFalse(bool(part.train_mask[env_ids].any()))
        self.assertTrue(bool(part.heldout_mask[env_ids].all()))

    def test_none_mask_legacy_behavior(self):
        buffers = self._buffers(N=10)
        sampler = SequenceSampler(seq_len=8, seq_batch_size=4, device="cpu")
        torch.manual_seed(2)
        b1 = sampler.sample(**buffers)
        torch.manual_seed(2)
        b2 = sampler.sample(**buffers, env_index_mask=None)
        self.assertTrue(torch.allclose(b1.obs, b2.obs))

    def test_bad_mask_shape_raises(self):
        sampler = SequenceSampler(seq_len=8, seq_batch_size=4, device="cpu")
        with self.assertRaises(ValueError):
            sampler.sample(**self._buffers(N=10), env_index_mask=torch.ones(7).bool())

    def test_empty_mask_raises(self):
        sampler = SequenceSampler(seq_len=8, seq_batch_size=4, device="cpu")
        with self.assertRaises(ValueError):
            sampler.sample(
                **self._buffers(N=10),
                env_index_mask=torch.zeros(10).bool(),
            )


class TestTrainerPartitionWiring(unittest.TestCase):
    def _setup(self, *, N=32, heldout_fraction=0.25, seed=5, rank=0):
        torch.manual_seed(0)
        T, obs_dim, action_dim, B = 8, 6, 3, 16

        class _Buf:
            def __init__(self):
                self.obs_base_shape = (T, N)
                self.device = "cpu"
                self.tensor_dict = {
                    "obses": torch.randn(T, N, obs_dim),
                    "actions": torch.randn(T, N, action_dim),
                    "rewards": torch.randn(T, N, 1),
                    "dones": torch.zeros(T, N, dtype=torch.uint8),
                }

            def update_data(self, name, index, val):
                stored = self.tensor_dict[name]
                if val.ndim < stored.ndim - 1:
                    val = val.unsqueeze(-1)
                stored[index].copy_(val)

        from dexpbt_isaaclab.world_model.wm_aux_buffer import (
            EXTENDED_SEQ_AUX_FIELDS,
            register_seq_aux_fields,
        )

        buf = _Buf()
        register_seq_aux_fields(buf, fields=EXTENDED_SEQ_AUX_FIELDS)
        buf.tensor_dict["aux_d_hat"].uniform_(0.01, 0.04)
        buf.tensor_dict["aux_success_tolerance"].fill_(0.015)
        buf.tensor_dict["aux_success_cnt"].zero_()
        buf.tensor_dict["aux_failure"].zero_()
        # Create success increments for some envs so calibration can label.
        buf.tensor_dict["aux_success_cnt"][4:, :] = 1.0

        model = RSSMLiteWorldModel(
            obs_dim=obs_dim,
            action_dim=action_dim,
            obs_embed_dim=16,
            action_embed_dim=8,
            h_dim=16,
            z_dim=4,
            head_hidden=16,
            with_horizon_heads=True,
            horizon_steps=(4,),
            with_recovery_head=True,
            recovery_horizon=4,
        )
        cfg = SeqWMConfig(
            seq_len=T,
            seq_batch_size=B,
            horizon_steps=(4,),
            use_horizon_heads=True,
            use_recovery_head=True,
            recovery_horizon=4,
            floor_tol_threshold=0.02,
            recovery_shadow_tol_threshold=0.05,
        )
        trainer = SeqWMTrainer(model=model, cfg=cfg, device="cpu")
        cal = WMCalibrationEvaluator(
            WMCalibrationConfig(
                enabled=True,
                heldout_fraction=heldout_fraction,
                heldout_partition_seed=seed,
                min_samples=4,
                eval_interval_frames=0,
            )
        )
        trainer.configure_wm_partition(evaluator=cal, worker_rank=rank)
        last_obs = torch.randn(N, obs_dim)
        returns = torch.randn(T, N)
        return trainer, buf, last_obs, returns, cal

    def test_train_and_cal_source_ids_disjoint(self):
        trainer, buf, last_obs, returns, cal = self._setup()
        part = build_env_partition_masks(
            32, heldout_fraction=0.25, partition_seed=5, worker_rank=0
        )
        metrics = trainer.step(
            experience_buffer=buf,
            frames=0,
            last_obs=last_obs,
            returns_full=returns,
        )
        self.assertEqual(metrics["wm_cal/partition_overlap_count"], 0.0)
        train_ids = set(
            int(x)
            for x in trainer.sampler.sample(
                obs_full=buf.tensor_dict["obses"],
                action_full=buf.tensor_dict["actions"],
                reward_full=buf.tensor_dict["rewards"],
                done_full=buf.tensor_dict["dones"].float(),
                returns_full=returns,
                last_obs=last_obs,
                env_index_mask=part.train_mask,
                return_source_ids=True,
            ).meta["env_indices"].tolist()
        )
        cal_ids = set(
            int(x)
            for x in trainer.sampler.sample(
                obs_full=buf.tensor_dict["obses"],
                action_full=buf.tensor_dict["actions"],
                reward_full=buf.tensor_dict["rewards"],
                done_full=buf.tensor_dict["dones"].float(),
                returns_full=returns,
                last_obs=last_obs,
                env_index_mask=part.heldout_mask,
                return_source_ids=True,
            ).meta["env_indices"].tolist()
        )
        self.assertEqual(train_ids & cal_ids, set())

    def test_ensemble_members_use_train_partition(self):
        trainer, buf, last_obs, returns, cal = self._setup()
        member = SeqWMTrainer(
            model=RSSMLiteWorldModel(
                obs_dim=6,
                action_dim=3,
                obs_embed_dim=16,
                action_embed_dim=8,
                h_dim=16,
                z_dim=4,
                head_hidden=16,
            ),
            cfg=trainer.cfg,
            device="cpu",
        )
        member.configure_wm_partition(evaluator=cal, worker_rank=0)
        m0 = trainer.step(
            experience_buffer=buf, frames=0, last_obs=last_obs, returns_full=returns
        )
        m1 = member.step(
            experience_buffer=buf, frames=0, last_obs=last_obs, returns_full=returns
        )
        self.assertEqual(m0["wm_cal/heldout_enabled"], 1.0)
        self.assertEqual(m1["wm_cal/heldout_enabled"], 1.0)
        self.assertEqual(m0["wm_cal/partition_overlap_count"], 0.0)
        self.assertEqual(m1["wm_cal/partition_overlap_count"], 0.0)

    def test_cal_no_fallback_to_train(self):
        trainer, buf, last_obs, returns, cal = self._setup(heldout_fraction=0.25)
        # Force tiny held-out eval by raising min_samples hugely.
        cal.cfg.min_samples = 10_000_000
        st = WMCalibrationState(alpha_wm=0.7, ema_raw=0.7)
        st = trainer.run_wm_calibration_eval(
            experience_buffer=buf,
            frames=1,
            last_obs=last_obs,
            returns_full=returns,
            evaluator=cal,
            state=st,
        )
        self.assertEqual(
            st.last_metrics["wm_cal/failure_reason_id"],
            FAILURE_INSUFFICIENT_HELDOUT_SAMPLES,
        )
        # No train fallback: heldout_enabled stays on.
        self.assertEqual(st.last_metrics.get("wm_cal/heldout_enabled"), 1.0)
        self.assertLess(st.ema_raw, 0.7)

    def test_ppo_buffer_untouched_by_partition(self):
        trainer, buf, last_obs, returns, cal = self._setup()
        before = buf.tensor_dict["obses"].clone()
        trainer.step(
            experience_buffer=buf, frames=0, last_obs=last_obs, returns_full=returns
        )
        self.assertTrue(torch.equal(before, buf.tensor_dict["obses"]))
        self.assertEqual(tuple(buf.tensor_dict["obses"].shape), (8, 32, 6))


if __name__ == "__main__":
    unittest.main()
