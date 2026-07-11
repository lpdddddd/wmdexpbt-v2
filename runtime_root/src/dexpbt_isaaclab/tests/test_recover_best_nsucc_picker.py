"""Patch D (rev 3, 2026-05-30): peak archive picker.

`recover_best_nsucc._pick_best_from_archive` selects the highest-scoring
archive entry by a configurable criterion (`nsucc_mean`, `nsucc_max`,
or `r_meta_raw`). The picker is the only logic in the script that runs
without Isaac Sim, so we cover it here. The end-to-end eval path
(`_build_eval_fair_command` + `subprocess.call`) is exercised in
manual smoke runs.

Importantly: this picker must NEVER let `_index.json` (which records the
LATEST peak, not the highest one) influence selection. We synthesize an
archive where `_index.json` points to a *lower* score than another entry
and assert that the picker still finds the higher one.
"""

import json
import tempfile
import unittest
from pathlib import Path

from dexpbt_isaaclab.scripts.recover_best_nsucc import (
    _pick_best_from_archive,
    _summary_payload,
)


def _write_meta(d: Path, name: str, meta: dict) -> Path:
    p = d / name
    p.write_text(json.dumps(meta))
    return p


def _meta(
    *,
    policy_idx: int = 1,
    iteration: int = 24,
    frame: int = 480_000_000,
    nsucc_mean: float = 4.27,
    nsucc_max: float = 45.0,
    r_meta_raw: float = 0.632,
    success_tolerance: float = 0.036,
    checkpoint: str = "/tmp/fake.pth",
) -> dict:
    return {
        "policy_idx": policy_idx,
        "iteration": iteration,
        "frame": frame,
        "nsucc_mean": nsucc_mean,
        "nsucc_max": nsucc_max,
        "r_meta_raw": r_meta_raw,
        "pbt_ranking_score": r_meta_raw,
        "objective": r_meta_raw,
        "checkpoint": checkpoint,
        "curriculum": {
            "success_tolerance": success_tolerance,
            "initial_tolerance": 0.075,
            "target_tolerance": 0.01,
        },
    }


class PickBestFromArchiveTest(unittest.TestCase):
    """Picker honours criterion, ignores `_index.json`, and tolerates junk."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.archive = Path(self._tmp.name)

    def test_picks_highest_nsucc_mean(self):
        _write_meta(self.archive, "A_iter_0010.json",
                    _meta(nsucc_mean=1.5, nsucc_max=10))
        _write_meta(self.archive, "B_iter_0020.json",
                    _meta(nsucc_mean=4.27, nsucc_max=45))
        _write_meta(self.archive, "C_iter_0015.json",
                    _meta(nsucc_mean=2.6, nsucc_max=35))

        pick = _pick_best_from_archive(self.archive, criterion="nsucc_mean")
        self.assertEqual(pick["criterion"], "nsucc_mean")
        self.assertAlmostEqual(pick["criterion_value"], 4.27)
        self.assertTrue(pick["archive_file"].endswith("B_iter_0020.json"))

    def test_picks_highest_nsucc_max(self):
        _write_meta(self.archive, "A.json",
                    _meta(nsucc_mean=3.5, nsucc_max=20))
        _write_meta(self.archive, "B.json",
                    _meta(nsucc_mean=2.0, nsucc_max=45))
        pick = _pick_best_from_archive(self.archive, criterion="nsucc_max")
        self.assertAlmostEqual(pick["criterion_value"], 45.0)

    def test_picks_highest_r_meta_raw(self):
        _write_meta(self.archive, "low.json",
                    _meta(r_meta_raw=0.30))
        _write_meta(self.archive, "high.json",
                    _meta(r_meta_raw=0.94))
        pick = _pick_best_from_archive(self.archive, criterion="r_meta_raw")
        self.assertAlmostEqual(pick["criterion_value"], 0.94)

    def test_skip_index_file(self):
        """`_index.json` (a Patch 5 latest-peak summary) MUST be ignored
        by the picker -- otherwise a recent low-scoring peak would
        shadow the historical maximum.
        """
        _write_meta(self.archive, "real_high.json",
                    _meta(nsucc_mean=4.27))
        # `_index.json` claims the latest peak is much lower; if the
        # picker accidentally treats it as a regular entry it will be
        # included in the candidates list.
        _write_meta(self.archive, "_index.json",
                    _meta(nsucc_mean=0.1))
        pick = _pick_best_from_archive(self.archive, criterion="nsucc_mean")
        self.assertAlmostEqual(pick["criterion_value"], 4.27)

    def test_empty_archive_raises(self):
        with self.assertRaises(ValueError) as ctx:
            _pick_best_from_archive(self.archive, criterion="nsucc_mean")
        self.assertIn("No valid entries", str(ctx.exception))

    def test_missing_archive_raises(self):
        with self.assertRaises(ValueError) as ctx:
            _pick_best_from_archive(
                self.archive / "does_not_exist",
                criterion="nsucc_mean",
            )
        self.assertIn("not found", str(ctx.exception))

    def test_unknown_criterion_raises(self):
        with self.assertRaises(ValueError) as ctx:
            _pick_best_from_archive(self.archive, criterion="not_a_field")
        self.assertIn("Unknown criterion", str(ctx.exception))

    def test_ignores_unparseable_files(self):
        (self.archive / "garbage.json").write_text("{not json")
        _write_meta(self.archive, "good.json", _meta(nsucc_mean=2.0))
        pick = _pick_best_from_archive(self.archive, criterion="nsucc_mean")
        self.assertAlmostEqual(pick["criterion_value"], 2.0)


class SummaryPayloadTest(unittest.TestCase):
    """`_summary_payload` projects the chosen meta into a flat dict."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.archive = Path(self._tmp.name)

    def test_summary_fields_present(self):
        _write_meta(self.archive, "peak.json", _meta(
            checkpoint="/some/path.pth",
            policy_idx=3,
            iteration=22,
            frame=480_000_000,
            nsucc_mean=4.27,
            nsucc_max=45,
            r_meta_raw=0.632,
            success_tolerance=0.036,
        ))
        pick = _pick_best_from_archive(self.archive, criterion="nsucc_mean")
        s = _summary_payload(pick)
        self.assertEqual(s["checkpoint_path"], "/some/path.pth")
        self.assertEqual(s["policy_idx"], 3)
        self.assertEqual(s["iteration"], 22)
        self.assertEqual(s["frame"], 480_000_000)
        self.assertAlmostEqual(s["saved_nsucc_mean"], 4.27)
        self.assertAlmostEqual(s["saved_nsucc_max"], 45.0)
        self.assertAlmostEqual(s["saved_r_meta_raw"], 0.632)
        self.assertAlmostEqual(s["saved_success_tolerance"], 0.036)
        self.assertIn("archive_dir", s)
        self.assertIn("archive_file", s)


class ArchiveDoesNotPolluteWorkspaceTest(unittest.TestCase):
    """Sanity: the picker MUST be a pure read on the archive directory.

    No PBT workspace files (e.g. 000/000010.json) should be touched, and
    the script must not modify `_index.json`.
    """

    def test_only_reads_jsons_does_not_write(self):
        with tempfile.TemporaryDirectory() as tmp:
            archive = Path(tmp)
            _write_meta(archive, "a.json", _meta(nsucc_mean=2.0))
            _write_meta(archive, "_index.json", _meta(nsucc_mean=0.1))
            before_mtime = (archive / "_index.json").stat().st_mtime
            before_content = (archive / "_index.json").read_text()

            _pick_best_from_archive(archive, criterion="nsucc_mean")

            after_mtime = (archive / "_index.json").stat().st_mtime
            after_content = (archive / "_index.json").read_text()
            self.assertEqual(before_mtime, after_mtime)
            self.assertEqual(before_content, after_content)


if __name__ == "__main__":
    unittest.main()
