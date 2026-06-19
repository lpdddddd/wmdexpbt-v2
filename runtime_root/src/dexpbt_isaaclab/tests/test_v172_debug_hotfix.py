"""v17.2 debug hotfix tests: monitor spread, AWR builder fuse, PBT tick scheduler."""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from dexpbt_isaaclab.awr.elite_awr import (
    AWR_BUILDER_FUSE_PAUSE_FRAMES,
    AWR_BUILDER_FUSE_SLOW_MS,
    AWR_BUILDER_SKIP_REASON_FUSE,
    EliteAWRBridge,
    EliteAWRBuffer,
    EliteAWRConfig,
    awr_apply_builder_run,
    awr_builder_collection_paused,
    awr_record_builder_fuse_skip,
)
from dexpbt_isaaclab.pbt.pbt_control_plane import (
    evaluate_pbt_tick_schedule,
    read_latest_commit_info,
)


def _load_monitor_module():
    repo_root = Path(__file__).resolve().parents[4]
    monitor_path = repo_root / "results" / "monitor_v17_2_control_plane_2b_verify.py"
    spec = importlib.util.spec_from_file_location("monitor_v172_debug", monitor_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class TestMonitorFrameSpread(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.monitor = _load_monitor_module()

    def test_prefers_log_frames_when_curve_stale(self):
        policies = [
            {"env_frames_curve": 200_000_000, "fps_frames_log": 1_780_000_000},
            {"env_frames_curve": 200_000_000, "fps_frames_log": 670_000_000},
        ]
        info = self.monitor._resolve_population_frames(policies)
        self.assertEqual(info["frame_source"], "log")
        self.assertGreater(info["frame_spread"], 1_000_000_000)
        self.assertTrue(info["curve_stale"])
        self.assertGreater(info["stale_gap_frames"], 100_000_000)

    def test_falls_back_to_curve_when_log_missing(self):
        policies = [
            {"env_frames_curve": 220_000_000, "fps_frames_log": 0},
            {"env_frames_curve": 200_000_000, "fps_frames_log": 0},
        ]
        info = self.monitor._resolve_population_frames(policies)
        self.assertEqual(info["frame_source"], "curve")
        self.assertEqual(info["frame_spread"], 20_000_000)
        self.assertFalse(info["curve_stale"])


class TestAwrBuilderFuse(unittest.TestCase):
    def _bridge(self) -> EliteAWRBridge:
        cfg = EliteAWRConfig(enable=True, loss_weight=0.001)
        return EliteAWRBridge(buffer=EliteAWRBuffer(cfg), cfg=cfg)

    def test_two_slow_runs_arm_fuse_and_skip_collection(self):
        bridge = self._bridge()
        slow_ms = AWR_BUILDER_FUSE_SLOW_MS + 1.0
        frames = 600_000_000

        awr_apply_builder_run(bridge, frames, slow_ms, {"awr/rollout_builder_entries": 0.0})
        self.assertEqual(getattr(bridge, "_awr_builder_slow_count", 0), 1)
        self.assertFalse(awr_builder_collection_paused(bridge, frames))

        awr_apply_builder_run(bridge, frames + 1, slow_ms, {"awr/rollout_builder_entries": 0.0})
        fuse_until = int(getattr(bridge, "_awr_builder_fuse_until_frame", 0))
        self.assertEqual(fuse_until, frames + 1 + AWR_BUILDER_FUSE_PAUSE_FRAMES)
        self.assertTrue(awr_builder_collection_paused(bridge, frames + 2))

        awr_record_builder_fuse_skip(bridge, frames + 2)
        self.assertEqual(bridge.stats["awr/builder_fuse_active"], 1.0)
        self.assertEqual(bridge.stats["awr/builder_skipped"], 1.0)
        self.assertEqual(bridge.stats["awr/builder_skip_reason_code"], AWR_BUILDER_SKIP_REASON_FUSE)

    def test_fast_run_resets_slow_count(self):
        bridge = self._bridge()
        awr_apply_builder_run(
            bridge,
            600_000_000,
            AWR_BUILDER_FUSE_SLOW_MS + 10.0,
            {"awr/rollout_builder_entries": 0.0},
        )
        awr_apply_builder_run(
            bridge,
            600_131_072,
            AWR_BUILDER_FUSE_SLOW_MS - 1.0,
            {"awr/rollout_builder_entries": 1.0},
        )
        self.assertEqual(getattr(bridge, "_awr_builder_slow_count", 0), 0)

    def test_apply_builder_run_never_disables_bridge(self):
        bridge = self._bridge()
        awr_apply_builder_run(
            bridge,
            600_000_000,
            AWR_BUILDER_FUSE_SLOW_MS + 100.0,
            {"awr/rollout_builder_entries": 0.0, "awr/rollout_builder_time_ms": 999.0},
        )
        awr_apply_builder_run(
            bridge,
            600_000_001,
            AWR_BUILDER_FUSE_SLOW_MS + 100.0,
            {"awr/rollout_builder_entries": 0.0, "awr/rollout_builder_time_ms": 999.0},
        )
        self.assertFalse(bridge.disabled)
        self.assertTrue(awr_builder_collection_paused(bridge, 600_000_002))


class TestPbtTickScheduler(unittest.TestCase):
    def test_derived_iteration_advances_with_raw_frame_despite_frozen_commit(self):
        should_tick, derived, reason = evaluate_pbt_tick_schedule(
            raw_frame=250_000_000,
            self_pbt_iteration=10,
            last_mutation_frame=200_000_000,
            start_after=200_000_000,
            initial_delay=50_000_000,
            interval_steps=20_000_000,
        )
        self.assertTrue(should_tick)
        self.assertEqual(derived, 12)
        self.assertEqual(reason, "")

    def test_existing_latest_commit_does_not_block_next_interval(self):
        with tempfile.TemporaryDirectory() as tmp:
            ws = Path(tmp)
            commit_dir = ws / "000012.staging.commit"
            commit_dir.mkdir(parents=True)
            (commit_dir / "MANIFEST").write_text(
                json.dumps(
                    {
                        "pbt_iteration": 10,
                        "population_frame": 200_000_000,
                    }
                )
            )
            (ws / "LATEST.commit").write_text(
                json.dumps({"commit_id": "abc", "commit_dir": commit_dir.name})
            )
            latest = read_latest_commit_info(ws)
            self.assertTrue(latest["latest_commit_exists"])
            self.assertEqual(latest["latest_commit_iteration"], 10)

            should_tick, derived, _ = evaluate_pbt_tick_schedule(
                raw_frame=250_000_000,
                self_pbt_iteration=10,
                last_mutation_frame=200_000_000,
                start_after=200_000_000,
                initial_delay=50_000_000,
                interval_steps=20_000_000,
            )
            self.assertTrue(should_tick)
            self.assertEqual(derived, 12)

    def test_initial_delay_blocks_until_enough_frames_since_last_tick(self):
        should_tick, _, reason = evaluate_pbt_tick_schedule(
            raw_frame=220_000_000,
            self_pbt_iteration=10,
            last_mutation_frame=200_000_000,
            start_after=200_000_000,
            initial_delay=50_000_000,
            interval_steps=20_000_000,
        )
        self.assertFalse(should_tick)
        self.assertEqual(reason, "initial_delay")

        should_tick, derived, reason = evaluate_pbt_tick_schedule(
            raw_frame=250_000_000,
            self_pbt_iteration=10,
            last_mutation_frame=200_000_000,
            start_after=200_000_000,
            initial_delay=50_000_000,
            interval_steps=20_000_000,
        )
        self.assertTrue(should_tick)
        self.assertEqual(derived, 12)
        self.assertEqual(reason, "")


if __name__ == "__main__":
    unittest.main()
