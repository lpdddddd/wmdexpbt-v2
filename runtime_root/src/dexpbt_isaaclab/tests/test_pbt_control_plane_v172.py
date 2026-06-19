"""v17.2 control-plane correctness: snapshot commit, skill_delta gating, frame restore."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import torch

from dexpbt_isaaclab.pbt.pbt_curriculum_p1 import (
    RollingNsuccWindow,
    WindowSample,
    compute_skill_delta,
)
from dexpbt_isaaclab.pbt.pbt_observer import (
    snapshot_commit_barrier,
    validate_snapshot_commit,
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

    def test_invalid_when_curriculum_shifts(self):
        window = RollingNsuccWindow(max_len=8)
        for i in range(6):
            tol = 0.075 if i < 3 else 0.05
            window.append(self._sample(i * 20_000_000, float(i), "same", tol=tol))
        self.assertIsNone(
            compute_skill_delta(window, window_len=6, interval_steps=20_000_000)
        )

    def test_valid_improving_window_is_bounded(self):
        window = RollingNsuccWindow(max_len=8)
        for i in range(6):
            window.append(self._sample(i * 20_000_000, float(i), "same"))
        delta = compute_skill_delta(window, window_len=6, interval_steps=20_000_000)
        self.assertIsNotNone(delta)
        assert delta is not None
        self.assertGreaterEqual(delta, -1.0)
        self.assertLessEqual(delta, 1.0)
        self.assertGreater(delta, 0.0)


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
                },
                tick=20_000_000,
            )
            ok, reason = validate_snapshot_commit(commit_dir, tick=20_000_000)
            self.assertTrue(ok, reason)
            self.assertTrue(publish_model.is_file())
            self.assertTrue(publish_meta.is_file())

    def test_validate_rejects_corrupt_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            staging = root / "bad.staging"
            staging.mkdir()
            (staging / "MANIFEST").write_text("{not json")
            ok, reason = validate_snapshot_commit(staging, tick=20_000_000)
            self.assertFalse(ok)
            self.assertEqual(reason, "manifest_corrupt")


if __name__ == "__main__":
    unittest.main()
