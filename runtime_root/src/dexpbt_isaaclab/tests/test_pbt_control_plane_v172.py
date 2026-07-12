"""v17.2 hardened control-plane: snapshot commit, restore, skill_delta, clocks."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from dexpbt_isaaclab.pbt.pbt_control_plane import (
    PbtClockState,
    PbtStateSnapshot,
    assert_pbt_clock_invariants,
    derive_lineage_frame,
    derive_pbt_iteration,
    pbt_copy_allowed_for_live_source,
    pbt_copy_allowed_for_target,
    pbt_pure_restore_clocks,
    pbt_scheduler_sync_clock,
    quantize_population_frame,
    snapshot_commit_barrier,
    validate_snapshot_commit,
)
from dexpbt_isaaclab.pbt.pbt_curriculum_p1 import (
    RollingNsuccWindow,
    WindowSample,
    compute_skill_delta,
)
from dexpbt_isaaclab.awr.elite_awr import (
    EliteAWRConfig,
    _apply_entropy_floor,
    _compute_awr_sample_weights,
)


class TestSkillDeltaGating(unittest.TestCase):
    def _sample(self, pf: int, nsucc: float, lineage: str, tol: float = 0.075) -> WindowSample:
        return WindowSample(
            env_frames=pf,
            nsucc_mean=nsucc,
            nsucc_max=nsucc,
            success_tolerance=tol,
            pbt_iteration=pf // 20_000_000,
            lineage_id=lineage,
        )

    def test_invalid_when_lineage_changes(self):
        window = RollingNsuccWindow(max_len=8)
        for i in range(6):
            lid = "a" if i < 3 else "b"
            window.append(self._sample(i * 20_000_000, float(i), lid))
        self.assertIsNone(
            compute_skill_delta(window, window_len=6, interval_steps=20_000_000)
        )

    def test_stuck_gate_requires_low_nsucc_and_nonpositive_delta(self):
        window = RollingNsuccWindow(max_len=8)
        for i in range(6):
            window.append(self._sample(i * 20_000_000, 5.0, "same"))
        allowed, reason, _ = pbt_copy_allowed_for_target(
            window, interval_steps=20_000_000
        )
        self.assertFalse(allowed)
        self.assertEqual(reason, "not_stuck")

    def test_improving_gate_blocks_copy(self):
        window = RollingNsuccWindow(max_len=8)
        for i in range(6):
            window.append(self._sample(i * 20_000_000, float(i), "same"))
        allowed, reason, _ = pbt_copy_allowed_for_target(
            window, interval_steps=20_000_000
        )
        self.assertFalse(allowed)
        self.assertEqual(reason, "skill_delta_improving")

    def test_live_source_requires_stable_positive_delta(self):
        window = RollingNsuccWindow(max_len=8)
        for i in range(6):
            window.append(self._sample(i * 20_000_000, 0.2, "same"))
        ok, reason = pbt_copy_allowed_for_live_source(
            window, interval_steps=20_000_000
        )
        self.assertFalse(ok)
        self.assertEqual(reason, "source_skill_delta_invalid_or_unstable")

    def test_adaptive_rescue_target_allows_unavailable_delta(self):
        from dexpbt_isaaclab.pbt.pbt_control_plane import (
            pbt_copy_allowed_for_adaptive_rescue_target,
        )

        window = RollingNsuccWindow(max_len=8)
        ok, reason, delta = pbt_copy_allowed_for_adaptive_rescue_target(
            window, interval_steps=20_000_000, hard_collapse=False
        )
        self.assertTrue(ok)
        self.assertEqual(reason, "skill_delta_unavailable_ok")
        self.assertIsNone(delta)

    def test_adaptive_rescue_target_blocks_improving(self):
        from dexpbt_isaaclab.pbt.pbt_control_plane import (
            pbt_copy_allowed_for_adaptive_rescue_target,
        )

        window = RollingNsuccWindow(max_len=8)
        for i in range(6):
            window.append(self._sample(i * 20_000_000, float(i), "same"))
        ok, reason, _ = pbt_copy_allowed_for_adaptive_rescue_target(
            window, interval_steps=20_000_000, hard_collapse=False
        )
        self.assertFalse(ok)
        self.assertEqual(reason, "skill_delta_improving")

    def test_adaptive_rescue_hard_collapse_bypasses_improving(self):
        from dexpbt_isaaclab.pbt.pbt_control_plane import (
            pbt_copy_allowed_for_adaptive_rescue_target,
        )

        window = RollingNsuccWindow(max_len=8)
        for i in range(6):
            window.append(self._sample(i * 20_000_000, float(i), "same"))
        ok, reason, _ = pbt_copy_allowed_for_adaptive_rescue_target(
            window, interval_steps=20_000_000, hard_collapse=True
        )
        self.assertTrue(ok)
        self.assertEqual(reason, "adaptive_hard_collapse")


class TestClockDerivation(unittest.TestCase):
    def test_quantized_iteration(self):
        tick = 20_000_000
        pf = quantize_population_frame(45_000_001, tick)
        self.assertEqual(pf, 40_000_000)
        self.assertEqual(derive_pbt_iteration(pf, tick), 2)
        self.assertEqual(derive_lineage_frame(60_000_000, 40_000_000), 20_000_000)

    def test_scheduler_sync_is_single_writer(self):
        algo = SimpleNamespace(frame=45_000_001)
        pf = pbt_scheduler_sync_clock(
            algo,
            tick=20_000_000,
            lineage_id="lineage-a",
            lineage_start_pf=20_000_000,
            weight_age_frame=5_000_000,
        )
        self.assertEqual(pf, 40_000_000)
        self.assertEqual(algo.frame, 40_000_000)
        self.assertEqual(algo._pbt_population_frame, 40_000_000)
        algo._pbt_derived_iteration = derive_pbt_iteration(pf, 20_000_000)
        assert_pbt_clock_invariants(algo, tick=20_000_000)


class TestPureRestore(unittest.TestCase):
    def test_copy_restore_continues_target_clock(self):
        agent = SimpleNamespace(frame=0, epoch_num=0)
        ckpt = {"frame": 80_000_000, "epoch": 100}
        snap = pbt_pure_restore_clocks(
            agent,
            ckpt,
            meta=None,
            target_frame_before_restore=40_000_000,
            restart_source_policy=2,
        )
        self.assertEqual(snap.clocks.population_frame, 40_000_000)
        self.assertEqual(agent.frame, 40_000_000)
        self.assertTrue(getattr(agent, "_pbt_restored_from_pbt_copy", False))
        self.assertEqual(int(getattr(agent, "_pbt_copy_reset_progress_due_to_frame_gap", 1)), 0)

    def test_copy_without_target_fails_fast(self):
        agent = SimpleNamespace(frame=0)
        with self.assertRaises(RuntimeError):
            pbt_pure_restore_clocks(
                agent,
                {"frame": 1},
                meta=None,
                target_frame_before_restore=None,
                restart_source_policy=1,
            )

    def test_meta_snapshot_restore(self):
        agent = SimpleNamespace(frame=0, epoch_num=0)
        meta = PbtStateSnapshot(
            clocks=PbtClockState(60_000_000, "lid", 40_000_000, 55_000_000)
        ).to_meta_fields()
        snap = pbt_pure_restore_clocks(
            agent,
            {"frame": 10},
            meta=meta,
            target_frame_before_restore=None,
            restart_source_policy=None,
        )
        self.assertEqual(snap.clocks.population_frame, 60_000_000)
        self.assertEqual(agent._pbt_lineage_id, "lid")


class TestSnapshotCommitBarrier(unittest.TestCase):
    def test_atomic_commit_and_validate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model_src = root / "model.tmp"
            meta_src = root / "meta.tmp"
            publish_model = root / "model.pth"
            publish_meta = root / "meta.json"
            torch.save({"model": {}}, model_src)
            meta_src.write_text(json.dumps({"frame": 40_000_000}))
            commit_dir = snapshot_commit_barrier(
                staging_dir=root / "000001.staging",
                model_src=model_src,
                meta_src=meta_src,
                publish_model=publish_model,
                publish_meta=publish_meta,
                manifest={
                    "commit_id": "000001",
                    "population_frame": 40_000_000,
                    "lineage_id": "lineage-a",
                    "lineage_start_pf": 20_000_000,
                    "weight_age_frame": 35_000_000,
                },
                tick=20_000_000,
            )
            ok, reason = validate_snapshot_commit(commit_dir, tick=20_000_000)
            self.assertTrue(ok, reason)
            self.assertTrue(publish_model.is_file())
            self.assertTrue(publish_meta.is_file())
            latest = publish_meta.parent / "LATEST.commit"
            self.assertTrue(latest.is_file())

    def test_validate_rejects_corrupt_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            staging = root / "bad.staging"
            staging.mkdir()
            (staging / "MANIFEST").write_text("{not json")
            ok, reason = validate_snapshot_commit(staging, tick=20_000_000)
            self.assertFalse(ok)
            self.assertEqual(reason, "manifest_corrupt")


class TestAwrEntropyFloor(unittest.TestCase):
    def test_entropy_floor_falls_back_to_uniform(self):
        w = torch.tensor([0.97, 0.02, 0.01, 0.0])
        out, applied = _apply_entropy_floor(w, eps=0.5)
        self.assertEqual(applied, 1.0)
        active = out[out > 0]
        self.assertAlmostEqual(float(active.sum().item()), 1.0, places=5)
        self.assertAlmostEqual(float(active.std(unbiased=False).item()), 0.0, places=5)

    def test_apply_entropy_floor_helper(self):
        w = torch.tensor([0.99, 0.01, 0.0, 0.0])
        out, applied = _apply_entropy_floor(w, eps=0.5)
        self.assertEqual(applied, 1.0)
        active = out[out > 0]
        self.assertAlmostEqual(float(active.mean().item()), 0.5, places=5)


if __name__ == "__main__":
    unittest.main()
