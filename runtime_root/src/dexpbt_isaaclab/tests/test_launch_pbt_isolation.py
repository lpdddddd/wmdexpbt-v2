"""Unit tests for launch_pbt isolation + audit helpers."""

from __future__ import annotations

import json
import os
import tempfile
import threading
import time
import unittest
from pathlib import Path

from dexpbt_isaaclab.scripts.launch_pbt import (
    DEFAULT_FIRST_EPOCH_MARKERS,
    _audit,
    _clean_policy_locks,
    _prepare_policy_ov_root,
    _StartupWatchdog,
)


class PreparePolicyOvRootTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self._tmp.name)
        self.runtime_root = self.tmp_path / "runtime_root"
        self.shared_v2 = self.runtime_root / "cache" / "ov" / "data" / "exts" / "v2"
        self.shared_v2.mkdir(parents=True)
        self.shared_client = self.runtime_root / "cache" / "ov_client" / "client"
        self.shared_client.mkdir(parents=True)
        # Two extension dirs we expect to be symlinked.
        for name in ("omni.kit.foo-1.0.0", "omni.kit.bar-2.3.4"):
            (self.shared_v2 / name).mkdir()
        # Drop a cache_db.json + _cache.json that should be copied.
        (self.shared_v2 / "cache_db.json").write_text('{"shared":true}')
        (self.shared_client / "_cache.json").write_text('{"shared_client":true}')

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _policy_dir(self) -> Path:
        d = self.tmp_path / "run" / "policy_00"
        d.mkdir(parents=True)
        return d

    def test_symlink_tree_shape(self) -> None:
        policy_dir = self._policy_dir()
        xdg_data, xdg_cache = _prepare_policy_ov_root(policy_dir, self.runtime_root)

        per_v2 = xdg_data / "ov" / "data" / "exts" / "v2"
        self.assertTrue((per_v2 / "index").is_dir())
        # Extension dirs are symlinked, not copied.
        for name in ("omni.kit.foo-1.0.0", "omni.kit.bar-2.3.4"):
            link = per_v2 / name
            self.assertTrue(link.is_symlink(), f"{name} not symlink")
            self.assertEqual(link.resolve(), (self.shared_v2 / name).resolve())
        # cache_db.json copied (writable, not symlink).
        cdb = per_v2 / "cache_db.json"
        self.assertTrue(cdb.is_file() and not cdb.is_symlink())
        self.assertIn("shared", cdb.read_text())
        # _cache.json under per-policy ov client.
        client = xdg_cache / "ov" / "client" / "_cache.json"
        self.assertTrue(client.is_file() and not client.is_symlink())

    def test_idempotent_symlink_creation(self) -> None:
        policy_dir = self._policy_dir()
        _prepare_policy_ov_root(policy_dir, self.runtime_root)
        # Second call should not raise.
        xdg_data, _ = _prepare_policy_ov_root(policy_dir, self.runtime_root)
        per_v2 = xdg_data / "ov" / "data" / "exts" / "v2"
        self.assertTrue((per_v2 / "omni.kit.foo-1.0.0").is_symlink())

    def test_missing_shared_cache_db_warns_only(self) -> None:
        # Remove the shared cache_db.json -> per-policy file should be absent
        # but the function must not raise.
        (self.shared_v2 / "cache_db.json").unlink()
        policy_dir = self._policy_dir()
        xdg_data, _ = _prepare_policy_ov_root(policy_dir, self.runtime_root)
        cdb = xdg_data / "ov" / "data" / "exts" / "v2" / "cache_db.json"
        self.assertFalse(cdb.exists(), "cache_db.json should not be fabricated")

    def test_missing_shared_client_cache_warns_only(self) -> None:
        (self.shared_client / "_cache.json").unlink()
        policy_dir = self._policy_dir()
        _, xdg_cache = _prepare_policy_ov_root(policy_dir, self.runtime_root)
        client = xdg_cache / "ov" / "client" / "_cache.json"
        self.assertFalse(client.exists(), "_cache.json should not be fabricated")


class CleanPolicyLocksTest(unittest.TestCase):
    def test_locks_and_flag_removed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            policy_dir = Path(tmp) / "policy_03"
            policy_dir.mkdir()
            data_dir = policy_dir / ".xdg_data"
            (data_dir / "ov" / "data" / "exts" / "v2" / "index").mkdir(parents=True)
            (data_dir / "ov" / "data" / "exts" / "v2" / "index" / "registry.lock").write_text("x")
            (data_dir / "ov" / "data" / "exts" / "v2" / "kvdb.lock").write_text("x")
            (data_dir / "stale.lock").write_text("x")
            (policy_dir / "startup_hang.flag").write_text("{}")
            (policy_dir / "train_pid.txt").write_text("12345")

            _clean_policy_locks(policy_dir)

            self.assertFalse(
                (data_dir / "ov" / "data" / "exts" / "v2" / "index" / "registry.lock").exists()
            )
            self.assertFalse(
                (data_dir / "ov" / "data" / "exts" / "v2" / "kvdb.lock").exists()
            )
            self.assertFalse((data_dir / "stale.lock").exists())
            self.assertFalse((policy_dir / "startup_hang.flag").exists())
            self.assertFalse((policy_dir / "train_pid.txt").exists())


class AuditFormatTest(unittest.TestCase):
    def test_audit_writes_jsonl(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_root = Path(tmp)
            _audit(run_root, 1, "policy_start_time", attempt=1, gpu="3")
            _audit(run_root, 1, "policy_first_epoch_time", elapsed_s=42.5)
            audit_path = run_root / "launch_audit.jsonl"
            lines = audit_path.read_text().splitlines()
            self.assertEqual(len(lines), 2)
            row1 = json.loads(lines[0])
            row2 = json.loads(lines[1])
            self.assertEqual(row1["event"], "policy_start_time")
            self.assertEqual(row1["policy_idx"], 1)
            self.assertEqual(row2["elapsed_s"], 42.5)
            self.assertIn("ts", row1)


class WatchdogMatcherTest(unittest.TestCase):
    """Verify the watchdog flips warmed when any default marker appears."""

    def test_marker_match_sets_warmed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            policy_dir = Path(tmp) / "policy_00"
            policy_dir.mkdir()
            run_root = Path(tmp)
            train_log = policy_dir / "train.log"
            # Pre-populate with text that matches one of the default markers.
            train_log.write_text("noise...\nfps step: 12345\n")

            class _Dummy:
                returncode: int | None = None

                def poll(self):
                    return None

            warmed: set[int] = set()
            wd = _StartupWatchdog(
                run_root=run_root,
                policy_dir=policy_dir,
                policy_idx=0,
                markers=DEFAULT_FIRST_EPOCH_MARKERS,
                timeout_s=0.5,
                max_restarts=0,
                proc=_Dummy(),  # type: ignore[arg-type]
                warmed_set=warmed,
                warmed_lock=threading.Lock(),
            )
            wd._poll_interval = 0.05
            t = threading.Thread(target=wd.run, daemon=True)
            t.start()
            t.join(timeout=2.0)
            self.assertIn(0, warmed)


if __name__ == "__main__":
    unittest.main()
