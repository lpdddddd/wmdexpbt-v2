"""Patch 5 (rev 2, 2026-05-30): peak-protected best archives.

Guards the property that `best_r_meta/` and `best_nsucc/` are append-only,
indexed by reporting metrics (NOT the ranking score), and survive when the
ranking score later collapses (the failure mode observed at 1.9B frames in
regrasping_cbwm_20260530_090008 where `best/` got polluted with iter
81--90 entries that all had nsucc=0).
"""

import json
import tempfile
import unittest
from pathlib import Path

from dexpbt_isaaclab.pbt.pbt_observer import DexPBTPbtAlgoObserver


def _make_observer(workspace: Path) -> DexPBTPbtAlgoObserver:
    """Construct an observer without triggering Isaac/WM init."""
    return DexPBTPbtAlgoObserver(
        policy_idx=0,
        num_policies=1,
        mutable_params={
            "enable_wm": False,
            "enable_counterfactual": False,
            "wm_strategy": "none",
        },
        workspace=str(workspace),
        train_dir=str(workspace),
        task_name="test-task",
        # Disable cooldowns / curriculum gating; this test only exercises the
        # best-archive logic.
        interval_steps=10_000_000,
        start_after=0,
        initial_delay=0,
        num_envs=1,
    )


def _state(
    *,
    iteration: int,
    frame: int,
    policy_idx: int,
    r_meta_raw: float,
    nsucc_mean: float,
    nsucc_max: float,
    ranking_score: float | None = None,
    checkpoint: Path,
) -> dict:
    """Build a fake `_save_workspace_checkpoint` state dict for testing."""
    return {
        "policy_idx": policy_idx,
        "iteration": iteration,
        "frame": frame,
        "r_meta_raw": r_meta_raw,
        "pbt_ranking_score": (
            ranking_score if ranking_score is not None else r_meta_raw
        ),
        "objective": ranking_score if ranking_score is not None else r_meta_raw,
        "ranking_metric_mode": "success_aware_r_meta",
        "nsucc_zero_cap": 0.05,
        "nsucc_mean": nsucc_mean,
        "nsucc_max": nsucc_max,
        "params": {},
        "checkpoint": str(checkpoint.resolve()),
        "curriculum": {},
        "experiment_name": "unit-test",
        "timestamp": 0.0,
    }


class BestStrictBetterPredicates(unittest.TestCase):
    def test_r_meta_strict_better_first_entry(self):
        s = _state(
            iteration=1, frame=1, policy_idx=0,
            r_meta_raw=0.1, nsucc_mean=0.0, nsucc_max=0.0,
            checkpoint=Path("/dev/null"),
        )
        self.assertTrue(
            DexPBTPbtAlgoObserver._is_strictly_better_r_meta(s, None)
        )

    def test_r_meta_strict_better_requires_strict_gt(self):
        a = _state(
            iteration=1, frame=1, policy_idx=0,
            r_meta_raw=0.2, nsucc_mean=0.0, nsucc_max=0.0,
            checkpoint=Path("/dev/null"),
        )
        b = _state(
            iteration=2, frame=2, policy_idx=0,
            r_meta_raw=0.2, nsucc_mean=10.0, nsucc_max=99.0,
            checkpoint=Path("/dev/null"),
        )
        # nsucc has improved but r_meta_raw is tied -> NOT better on r_meta axis
        self.assertFalse(
            DexPBTPbtAlgoObserver._is_strictly_better_r_meta(b, a)
        )

    def test_nsucc_strict_lex_mean_first(self):
        a = _state(
            iteration=1, frame=1, policy_idx=0,
            r_meta_raw=0.5, nsucc_mean=1.0, nsucc_max=10.0,
            checkpoint=Path("/dev/null"),
        )
        b = _state(
            iteration=2, frame=2, policy_idx=0,
            r_meta_raw=0.5, nsucc_mean=1.5, nsucc_max=0.0,
            checkpoint=Path("/dev/null"),
        )
        # mean improved, max regressed -> still strictly better on (mean, max)
        self.assertTrue(
            DexPBTPbtAlgoObserver._is_strictly_better_nsucc(b, a)
        )

    def test_nsucc_strict_falls_back_to_max_on_mean_tie(self):
        a = _state(
            iteration=1, frame=1, policy_idx=0,
            r_meta_raw=0.5, nsucc_mean=1.5, nsucc_max=10.0,
            checkpoint=Path("/dev/null"),
        )
        b = _state(
            iteration=2, frame=2, policy_idx=0,
            r_meta_raw=0.5, nsucc_mean=1.5, nsucc_max=11.0,
            checkpoint=Path("/dev/null"),
        )
        self.assertTrue(
            DexPBTPbtAlgoObserver._is_strictly_better_nsucc(b, a)
        )

    def test_nsucc_not_better_on_full_tie(self):
        a = _state(
            iteration=1, frame=1, policy_idx=0,
            r_meta_raw=0.5, nsucc_mean=1.0, nsucc_max=10.0,
            checkpoint=Path("/dev/null"),
        )
        b = _state(
            iteration=2, frame=2, policy_idx=0,
            r_meta_raw=0.5, nsucc_mean=1.0, nsucc_max=10.0,
            checkpoint=Path("/dev/null"),
        )
        self.assertFalse(
            DexPBTPbtAlgoObserver._is_strictly_better_nsucc(b, a)
        )


class BestArchiveFlowTest(unittest.TestCase):
    """End-to-end test of `_update_best_archives` with a real temp dir."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.workspace = Path(self._tmp.name)
        self.obs = _make_observer(self.workspace)

        # Manually create the archive dirs (normally created in after_init).
        self.obs.best_r_meta_workspace.mkdir(parents=True, exist_ok=True)
        self.obs.best_nsucc_workspace.mkdir(parents=True, exist_ok=True)

        # Fake checkpoint payload referenced by every state.
        self.ckpt_dir = self.workspace / "ckpts"
        self.ckpt_dir.mkdir()
        self.ckpt_a = self.ckpt_dir / "a.pth"
        self.ckpt_a.write_bytes(b"a-weights")
        self.ckpt_b = self.ckpt_dir / "b.pth"
        self.ckpt_b.write_bytes(b"b-weights")
        self.ckpt_c = self.ckpt_dir / "c.pth"
        self.ckpt_c.write_bytes(b"c-weights")

    def _count_pth(self, archive_dir: Path) -> int:
        return len(list(archive_dir.glob("*.pth")))

    def _read_index(self, archive_dir: Path) -> dict:
        return json.loads((archive_dir / "_index.json").read_text())

    def test_peak_r_meta_then_collapse_keeps_peak(self):
        # Tick 1: real DexPBT peak from policy_3 reference numbers.
        peak = _state(
            iteration=28, frame=560_000_000, policy_idx=3,
            r_meta_raw=0.427, nsucc_mean=2.76, nsucc_max=38.0,
            ranking_score=0.427,
            checkpoint=self.ckpt_a,
        )
        self.obs._update_best_archives(peak)
        # Tick 2: post-collapse latest state where r_meta_raw equals the
        # tolerance term alone (nsucc=0). With the new semantics this must
        # NOT pollute the peak archives.
        collapsed = _state(
            iteration=85, frame=1_700_000_000, policy_idx=0,
            r_meta_raw=0.541, nsucc_mean=0.0, nsucc_max=0.0,
            ranking_score=0.05,
            checkpoint=self.ckpt_b,
        )
        self.obs._update_best_archives(collapsed)

        # r_meta peak: collapsed has higher r_meta_raw -> archive should
        # accept it (we never lose history; both peaks coexist).
        self.assertEqual(self._count_pth(self.obs.best_r_meta_workspace), 2)
        # But nsucc peak archive must still point to iter 28 (collapsed
        # cannot replace a non-zero nsucc peak).
        nsucc_index = self._read_index(self.obs.best_nsucc_workspace)
        self.assertEqual(nsucc_index["iteration"], 28)
        self.assertEqual(nsucc_index["nsucc_mean"], 2.76)
        self.assertEqual(nsucc_index["nsucc_max"], 38.0)
        # And only ONE file in nsucc archive (collapsed did not qualify).
        self.assertEqual(self._count_pth(self.obs.best_nsucc_workspace), 1)

    def test_archive_is_append_only(self):
        s1 = _state(
            iteration=1, frame=10_000_000, policy_idx=3,
            r_meta_raw=0.1, nsucc_mean=0.5, nsucc_max=5.0,
            checkpoint=self.ckpt_a,
        )
        s2 = _state(
            iteration=2, frame=20_000_000, policy_idx=3,
            r_meta_raw=0.2, nsucc_mean=1.0, nsucc_max=10.0,
            checkpoint=self.ckpt_b,
        )
        s3 = _state(
            iteration=3, frame=30_000_000, policy_idx=3,
            r_meta_raw=0.3, nsucc_mean=2.0, nsucc_max=20.0,
            checkpoint=self.ckpt_c,
        )
        for s in (s1, s2, s3):
            self.obs._update_best_archives(s)

        # All three were strictly better than predecessor on both metrics.
        self.assertEqual(self._count_pth(self.obs.best_r_meta_workspace), 3)
        self.assertEqual(self._count_pth(self.obs.best_nsucc_workspace), 3)
        # Index now points to the most recent (and best) state.
        rmeta_index = self._read_index(self.obs.best_r_meta_workspace)
        self.assertEqual(rmeta_index["iteration"], 3)

    def test_archive_skips_when_not_strictly_better(self):
        s1 = _state(
            iteration=1, frame=10_000_000, policy_idx=3,
            r_meta_raw=0.5, nsucc_mean=2.0, nsucc_max=20.0,
            checkpoint=self.ckpt_a,
        )
        s2 = _state(
            iteration=2, frame=20_000_000, policy_idx=3,
            r_meta_raw=0.5, nsucc_mean=2.0, nsucc_max=20.0,
            checkpoint=self.ckpt_b,
        )
        self.obs._update_best_archives(s1)
        self.obs._update_best_archives(s2)
        self.assertEqual(self._count_pth(self.obs.best_r_meta_workspace), 1)
        self.assertEqual(self._count_pth(self.obs.best_nsucc_workspace), 1)

    def test_index_reload_on_construction(self):
        """A fresh observer should rebuild its in-memory peak state from disk."""
        peak = _state(
            iteration=28, frame=560_000_000, policy_idx=3,
            r_meta_raw=0.427, nsucc_mean=2.76, nsucc_max=38.0,
            checkpoint=self.ckpt_a,
        )
        self.obs._update_best_archives(peak)
        # Construct another observer over the same workspace.
        obs2 = _make_observer(self.workspace)
        obs2._best_r_meta_state = obs2._load_best_index(obs2.best_r_meta_workspace)
        obs2._best_nsucc_state = obs2._load_best_index(obs2.best_nsucc_workspace)
        self.assertIsNotNone(obs2._best_r_meta_state)
        self.assertEqual(
            obs2._best_r_meta_state["iteration"], 28
        )
        # A new strictly-worse state must NOT replace the on-disk peak.
        worse = _state(
            iteration=29, frame=600_000_000, policy_idx=3,
            r_meta_raw=0.0, nsucc_mean=0.0, nsucc_max=0.0,
            checkpoint=self.ckpt_b,
        )
        self.assertFalse(
            DexPBTPbtAlgoObserver._is_strictly_better_r_meta(
                worse, obs2._best_r_meta_state
            )
        )
        self.assertFalse(
            DexPBTPbtAlgoObserver._is_strictly_better_nsucc(
                worse, obs2._best_nsucc_state
            )
        )


class PeakFilenameSchemaTest(unittest.TestCase):
    def test_filename_contains_required_fields(self):
        s = _state(
            iteration=28, frame=560_000_000, policy_idx=3,
            r_meta_raw=0.427, nsucc_mean=2.76, nsucc_max=38.0,
            checkpoint=Path("/dev/null"),
        )
        with tempfile.TemporaryDirectory() as tmp:
            obs = _make_observer(Path(tmp))
            name = obs._peak_filename("r_meta", s)
        for token in ("iter_0028", "policy003", "frame_00560M", "nsuccM_002.760",
                      "nsuccX_0038.0", "rmeta_000.4270"):
            self.assertIn(token, name, msg=f"missing token {token!r} in {name!r}")


if __name__ == "__main__":
    unittest.main()
