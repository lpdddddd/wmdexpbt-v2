"""CCRS toolkit tests (no Isaac, no GPU) — schema v2 paired-reset protocol.

Behavioral coverage (user-required, 2026-07-18 fix round):
 1. normalized_nsucc computed correctly (Q = mean(min(nsucc,M))/M);
 2. success_any_rate is auxiliary only (not used in verdicts);
 3. capability curve defaults to normalized_nsucc;
 4. D/F/C defaults to normalized_nsucc;
 5. same seed + env_id + fingerprint can be paired;
 6. fingerprint mismatch -> paired analysis refuses / marks invalid;
 7. evaluator records only the FIRST episode per env;
 8. episode completion order does not affect sample_id pairing;
 9. insufficient episodes -> status=incomplete;
10. incomplete results are NOT skipped by resume;
11. config hash change -> old results not skipped;
12. checkpoint identity change -> old results not skipped;
13. terminal_error without a reliable source -> marked invalid;
14. exact transitions use real a/b tolerances;
15. missing exact cells -> D/F/C refused (KeyError);
16. lineage change identified and excluded from within-lineage analysis;
17. old-schema results never mix into new aggregation;
18. report fabricates nothing on empty data.

Plus regression tests kept from v1 (tolerance freeze/drift, meters-only
units, policy-only source-level invariants, atomic writes, stats helpers).
Source-string checks are retained ONLY as secondary evidence; reset
consistency and eval correctness are covered by the behavioral tests above.
"""

from __future__ import annotations

import json
import math
import random
import unittest
from pathlib import Path
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dexpbt_isaaclab.ccrs.eval_protocol import (  # noqa: E402
    DEFAULT_TOLERANCES_M,
    EVALUATION_MODE_POLICY_ONLY,
    EpisodeSample,
    EvalUnit,
    METRIC_PRIMARY,
    RESET_PROTOCOL_VERSION,
    SCHEMA_VERSION,
    aggregate_fingerprint,
    aggregate_raw_dir,
    build_units,
    fingerprint_env_state,
    is_done,
    make_result_row,
    make_sample_id,
    mean_ci95,
    normalized_nsucc_mean,
    quantiles,
    quantize_values,
    result_path,
    wilson_ci,
    write_result_atomic,
)
from dexpbt_isaaclab.ccrs.rollout_collect import (  # noqa: E402
    FINGERPRINT_FIELDS,
    FirstEpisodeCollector,
    TerminalSnapshotHook,
    extract_reset_fingerprints,
)
from dexpbt_isaaclab.ccrs.analysis import (  # noqa: E402
    CCRSTable,
    CellKey,
    PRIMARY_METRIC,
    PairwiseDiff,
    ResetPairingError,
    capability_curve,
    check_reset_pairing,
    compute_dfc,
    curve_difference,
    dfc_from_Q,
    grid_projected_dfc,
    hierarchical_paired_bootstrap,
    preregistered_verdicts,
)
from dexpbt_isaaclab.ccrs import manifest as ccrs_manifest  # noqa: E402
from dexpbt_isaaclab.ccrs.report import generate as report_generate  # noqa: E402


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _mk_unit(**kw) -> EvalUnit:
    base = dict(
        run_id="run_x", worker_id=1, checkpoint_iteration=149,
        checkpoint_frame=4_120_000_000,
        checkpoint_path="/tmp/ckpt.pth",
        fixed_tolerance=0.020, reset_seed=10001,
        num_envs=16, num_episodes=8,
    )
    base.update(kw)
    return EvalUnit(**base)


def _mk_samples(unit: EvalUnit, nsuccs=None, valid_terminal=False):
    nsuccs = nsuccs if nsuccs is not None else [float(i % 5) for i in range(unit.num_episodes)]
    return [
        EpisodeSample.create(
            reset_seed=unit.reset_seed, env_id=i, nsucc=ns,
            max_consecutive_successes=unit.max_consecutive_successes,
            terminal_error=0.01 * (i + 1) if valid_terminal else float("nan"),
            terminal_error_valid=valid_terminal,
            dropped=False if valid_terminal else None,
        )
        for i, ns in enumerate(nsuccs)
    ]


def _mk_fps(unit: EvalUnit, salt: str = "") -> dict[int, str]:
    return {
        i: fingerprint_env_state({"object_pos": [0.1 * i, 0.2, 0.3],
                                  "salt": [float(len(salt))]})
        for i in range(unit.num_episodes)
    }


def _mk_row(unit: EvalUnit, nsuccs=None, valid_terminal=False, fps=None, **kw):
    samples = _mk_samples(unit, nsuccs, valid_terminal)
    return make_result_row(
        unit=unit, samples=samples,
        per_env_fingerprint=fps if fps is not None else _mk_fps(unit),
        git_sha="deadbeef", checkpoint_hash="cafebabe", **kw,
    )


def _table_row(worker, it, tol, seed, nsuccs, fps=None, frame=0, m=50):
    """Build a schema-v2 style result row for CCRSTable tests."""
    samples = [
        {"sample_id": make_sample_id(seed, i), "reset_seed": seed, "env_id": i,
         "nsucc": float(ns), "normalized_nsucc": min(float(ns), m) / m,
         "terminal_error": float("nan"), "terminal_error_valid": False,
         "dropped": None}
        for i, ns in enumerate(nsuccs)
    ]
    fps = fps if fps is not None else {
        i: f"fp_{seed}_{i}" for i in range(len(nsuccs))
    }
    return {
        "status": "ok", "schema_version": SCHEMA_VERSION,
        "worker_id": worker, "checkpoint_iteration": it,
        "fixed_tolerance": tol, "reset_seed": seed,
        "normalized_nsucc": sum(s["normalized_nsucc"] for s in samples) / len(samples),
        "nsucc_mean": sum(s["nsucc"] for s in samples) / len(samples),
        "success_any_rate": sum(1 for s in samples if s["nsucc"] > 0) / len(samples),
        "checkpoint_frame": frame, "checkpoint_path": "x",
        "samples": samples,
        "per_env_reset_fingerprint": {str(k): v for k, v in fps.items()},
    }


# ---------------------------------------------------------------------------
# 1/2: metrics
# ---------------------------------------------------------------------------

class TestNormalizedNsucc(unittest.TestCase):
    def test_normalized_nsucc_formula(self):
        # mean(min(nsucc, 50))/50; 60 clips to 50
        vals = [0.0, 25.0, 50.0, 60.0]
        expect = (0 + 25 + 50 + 50) / (4 * 50)
        self.assertAlmostEqual(normalized_nsucc_mean(vals, 50), expect)
        s = EpisodeSample.create(reset_seed=1, env_id=0, nsucc=60.0,
                                 max_consecutive_successes=50)
        self.assertAlmostEqual(s.normalized_nsucc, 1.0)
        s2 = EpisodeSample.create(reset_seed=1, env_id=1, nsucc=20.0,
                                  max_consecutive_successes=50)
        self.assertAlmostEqual(s2.normalized_nsucc, 0.4)

    def test_row_metrics_no_saturation(self):
        # nsucc 30..45 saturates success_any (all 1.0) but NOT normalized_nsucc
        u = _mk_unit(num_episodes=4, num_envs=8)
        row = _mk_row(u, nsuccs=[30.0, 35.0, 40.0, 45.0])
        self.assertAlmostEqual(row["success_any_rate"], 1.0)  # saturated
        self.assertAlmostEqual(row["normalized_nsucc"], 37.5 / 50)  # informative
        self.assertAlmostEqual(row["nsucc_mean"], 37.5)
        self.assertEqual(row["metric_protocol_version"], "v2_normalized_nsucc")

    def test_success_any_is_not_error_probability(self):
        u = _mk_unit(num_episodes=2, num_envs=4)
        row = _mk_row(u, nsuccs=[1.0, 0.0])
        # named success_any_rate, not success_rate / P(error<=tau)
        self.assertIn("success_any_rate", row)
        self.assertNotIn("success_rate", row)

    def test_primary_metric_constant(self):
        self.assertEqual(METRIC_PRIMARY, "normalized_nsucc")
        self.assertEqual(PRIMARY_METRIC, "normalized_nsucc")


class TestAuxiliaryMetricNotInVerdicts(unittest.TestCase):
    def test_verdicts_use_normalized_nsucc_only(self):
        v = preregistered_verdicts([], [])
        self.assertEqual(v["metric"], "normalized_nsucc")
        self.assertAlmostEqual(v["prereg_min_effect"], 0.05)
        self.assertAlmostEqual(v["prereg_min_effect_nsucc_at_M50"], 2.5)


# ---------------------------------------------------------------------------
# 3/4: defaults use normalized_nsucc
# ---------------------------------------------------------------------------

class TestPrimaryMetricDefaults(unittest.TestCase):
    def _table(self):
        t = CCRSTable()
        for seed in (1, 2):
            t.add_row(_table_row(1, 100, 0.022, seed, [40.0, 30.0, 20.0, 10.0]))
        return t

    def test_capability_curve_default_metric(self):
        t = self._table()
        pts = capability_curve(t, 1, 100, [0.022], n_boot=50)
        self.assertEqual(len(pts), 1)
        self.assertEqual(pts[0].metric, "normalized_nsucc")
        self.assertAlmostEqual(pts[0].value, 25.0 / 50)
        # context metrics attached
        self.assertAlmostEqual(pts[0].nsucc_mean, 25.0)
        self.assertAlmostEqual(pts[0].success_any_rate, 1.0)

    def test_dfc_default_metric(self):
        t = CCRSTable()
        for seed in (1, 2):
            for it, tol, ns in [(100, 0.023, 45.0), (100, 0.020, 40.0),
                                (101, 0.023, 42.0), (101, 0.020, 25.0)]:
                t.add_row(_table_row(1, it, tol, seed, [ns, ns - 5]))
        res = compute_dfc(t, 1, 100, 101, tol_pre=0.023, tol_post=0.020, n_boot=50)
        self.assertEqual(res.metric, "normalized_nsucc")
        self.assertEqual(res.mode, "exact")
        self.assertLess(res.identity_residual, 1e-12)


# ---------------------------------------------------------------------------
# 5/6: fingerprint pairing
# ---------------------------------------------------------------------------

class TestResetPairing(unittest.TestCase):
    def _paired_table(self, mismatch=False):
        t = CCRSTable()
        for (w, it) in [(4, 141), (1, 149)]:
            for seed in (1, 2):
                fps = {i: f"fp_{seed}_{i}" for i in range(4)}
                if mismatch and w == 1:
                    fps[2] = "fp_DIFFERENT"
                t.add_row(_table_row(w, it, 0.020, seed,
                                     [30.0 + w, 20.0, 10.0, 5.0], fps=fps))
        return t

    def test_matching_fingerprints_pairable(self):
        t = self._paired_table()
        chk = check_reset_pairing(t, [CellKey(4, 141, 0.020), CellKey(1, 149, 0.020)])
        self.assertTrue(chk.valid)
        self.assertEqual(chk.n_shared_samples, 8)  # 2 seeds x 4 envs
        diffs = curve_difference(t, a=(4, 141), b=(1, 149), tolerances=[0.020],
                                 n_boot=50)
        self.assertTrue(diffs[0].reset_pairing_valid)
        self.assertFalse(math.isnan(diffs[0].ci_lo))

    def test_fingerprint_mismatch_rejected(self):
        t = self._paired_table(mismatch=True)
        chk = check_reset_pairing(t, [CellKey(4, 141, 0.020), CellKey(1, 149, 0.020)])
        self.assertFalse(chk.valid)
        self.assertIn("seed_1_env_002", chk.mismatched_samples)
        diffs = curve_difference(t, a=(4, 141), b=(1, 149), tolerances=[0.020],
                                 n_boot=50)
        self.assertFalse(diffs[0].reset_pairing_valid)
        self.assertFalse(diffs[0].significant)      # no significance conclusion
        self.assertTrue(math.isnan(diffs[0].ci_lo))  # no paired bootstrap
        # verdicts degrade to reset_pairing_invalid
        v = preregistered_verdicts(diffs, [])
        self.assertEqual(v["verdict_p1_p4"], "reset_pairing_invalid")

    def test_seed_integer_equality_is_not_sufficient(self):
        """Same reset_seed but different fingerprints must NOT pair."""
        t = self._paired_table(mismatch=True)
        rows_a = t.cells[CellKey(4, 141, 0.020)]
        rows_b = t.cells[CellKey(1, 149, 0.020)]
        self.assertEqual(set(rows_a.keys()), set(rows_b.keys()))  # same seeds
        chk = check_reset_pairing(t, [CellKey(4, 141, 0.020), CellKey(1, 149, 0.020)])
        self.assertFalse(chk.valid)

    def test_fingerprint_quantization_stability(self):
        a = fingerprint_env_state({"object_pos": [0.10000000001, 0.2, 0.3]})
        b = fingerprint_env_state({"object_pos": [0.10000000002, 0.2, 0.3]})
        self.assertEqual(a, b)  # sub-quant noise ignored
        c = fingerprint_env_state({"object_pos": [0.101, 0.2, 0.3]})
        self.assertNotEqual(a, c)  # real difference detected
        self.assertEqual(quantize_values([0.02]), [2000])

    def test_extract_fingerprints_from_env_like(self):
        import torch
        class FakeEnv:
            object_pos = torch.tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
            object_rot = torch.tensor([[1.0, 0, 0, 0], [1.0, 0, 0, 0]])
            goal_pos = torch.tensor([[0.0, 0.0, 0.5], [0.0, 0.1, 0.5]])
            goal_rot = torch.tensor([[1.0, 0, 0, 0], [1.0, 0, 0, 0]])
            dof_pos = torch.tensor([[0.1] * 5, [0.2] * 5])
        fps = extract_reset_fingerprints(FakeEnv(), [0, 1])
        self.assertEqual(len(fps), 2)
        self.assertNotEqual(fps[0], fps[1])
        # missing field -> loud failure, no silent degradation
        class BadEnv:
            object_pos = torch.zeros(1, 3)
        with self.assertRaises(AttributeError):
            extract_reset_fingerprints(BadEnv(), [0])

    def test_aggregate_fingerprint_order_independent(self):
        a = aggregate_fingerprint({0: "x", 1: "y"})
        b = aggregate_fingerprint({1: "y", 0: "x"})
        self.assertEqual(a, b)


# ---------------------------------------------------------------------------
# 7/8: first-episode-per-env collector, order independence
# ---------------------------------------------------------------------------

class TestFirstEpisodeCollector(unittest.TestCase):
    def test_only_first_episode_recorded(self):
        c = FirstEpisodeCollector(reset_seed=7, num_episodes=4,
                                  max_consecutive_successes=50)
        self.assertTrue(c.offer(0, nsucc=10.0))
        self.assertFalse(c.offer(0, nsucc=99.0))   # second episode ignored
        self.assertTrue(c.offer(5 % 4, nsucc=3.0) or True)
        vals = {s.env_id: s.nsucc for s in c.samples}
        self.assertEqual(vals[0], 10.0)            # first value kept

    def test_env_ids_beyond_num_episodes_ignored(self):
        c = FirstEpisodeCollector(reset_seed=7, num_episodes=2,
                                  max_consecutive_successes=50)
        self.assertFalse(c.offer(2, nsucc=1.0))
        self.assertFalse(c.offer(5, nsucc=1.0))
        self.assertTrue(c.offer(1, nsucc=1.0))
        self.assertEqual(c.missing_env_ids, [0])
        self.assertFalse(c.complete)
        self.assertTrue(c.offer(0, nsucc=2.0))
        self.assertTrue(c.complete)

    def test_completion_order_does_not_affect_sample_ids(self):
        orders = [[0, 1, 2, 3], [3, 1, 0, 2]]
        id_sets = []
        for order in orders:
            c = FirstEpisodeCollector(reset_seed=9, num_episodes=4,
                                      max_consecutive_successes=50)
            for env_id in order:
                c.offer(env_id, nsucc=float(env_id))
            ids = sorted(s.sample_id for s in c.samples)
            by_id = {s.sample_id: s.nsucc for s in c.samples}
            id_sets.append((ids, by_id))
        self.assertEqual(id_sets[0][0], id_sets[1][0])   # same sample_id set
        self.assertEqual(id_sets[0][1], id_sets[1][1])   # same values per id

    def test_num_episodes_must_not_exceed_num_envs(self):
        with self.assertRaises(ValueError):
            _mk_unit(num_envs=4, num_episodes=8)


# ---------------------------------------------------------------------------
# 9/10/11/12: incomplete status + strict resume
# ---------------------------------------------------------------------------

class TestIncompleteAndResume(unittest.TestCase):
    def test_incomplete_status_and_diagnostics(self):
        u = _mk_unit(num_episodes=4, num_envs=8)
        samples = _mk_samples(u, nsuccs=[1.0, 2.0])[:2]  # only env 0,1
        row = make_result_row(u, samples, _mk_fps(u), "sha", "ck",
                              max_steps_reached=True)
        self.assertEqual(row["status"], "incomplete")
        self.assertEqual(row["requested_num_episodes"], 4)
        self.assertEqual(row["actual_num_episodes"], 2)
        self.assertTrue(row["max_steps_reached"])
        self.assertEqual(row["missing_env_ids"], [2, 3])
        self.assertEqual(row["stop_reason"], "max_steps_reached")

    def test_incomplete_not_skipped_by_resume(self):
        with tempfile.TemporaryDirectory() as td:
            raw = Path(td)
            u = _mk_unit(num_episodes=4, num_envs=8)
            samples = _mk_samples(u)[:2]
            row = make_result_row(u, samples, _mk_fps(u), "sha", "ck",
                                  max_steps_reached=True)
            write_result_atomic(raw, u, row)
            self.assertFalse(is_done(raw, u))     # incomplete must re-run

    def test_complete_row_is_done(self):
        with tempfile.TemporaryDirectory() as td:
            raw = Path(td)
            u = _mk_unit(num_episodes=4, num_envs=8)
            write_result_atomic(raw, u, _mk_row(u, nsuccs=[1, 2, 3, 4]))
            self.assertTrue(is_done(raw, u))

    def test_config_change_invalidates_resume(self):
        with tempfile.TemporaryDirectory() as td:
            raw = Path(td)
            u = _mk_unit(num_episodes=4, num_envs=8, max_steps=1000)
            write_result_atomic(raw, u, _mk_row(u, nsuccs=[1, 2, 3, 4]))
            self.assertTrue(is_done(raw, u))
            # max_steps changes config hash AND unit key -> not done
            u2 = _mk_unit(num_episodes=4, num_envs=8, max_steps=2000)
            self.assertFalse(is_done(raw, u2))
            # even a manually renamed file must fail field validation
            row = _mk_row(u, nsuccs=[1, 2, 3, 4])
            (raw / result_path(raw, u2).name).write_text(json.dumps(row))
            self.assertFalse(is_done(raw, u2))

    def test_checkpoint_identity_change_invalidates_resume(self):
        with tempfile.TemporaryDirectory() as td:
            raw = Path(td)
            u = _mk_unit()
            row = _mk_row(u, nsuccs=list(range(u.num_episodes)))
            write_result_atomic(raw, u, row)
            for delta in (dict(checkpoint_frame=999),
                          dict(checkpoint_path="/tmp/other.pth"),
                          dict(checkpoint_iteration=150)):
                u2 = _mk_unit(**delta)
                self.assertFalse(is_done(raw, u2), delta)
                (raw / result_path(raw, u2).name).write_text(json.dumps(row))
                self.assertFalse(is_done(raw, u2), delta)

    def test_unit_key_covers_required_fields(self):
        base = _mk_unit()
        variants = [
            _mk_unit(checkpoint_frame=1), _mk_unit(checkpoint_path="/y.pth"),
            _mk_unit(max_steps=1), _mk_unit(max_consecutive_successes=10),
        ]
        for v in variants:
            self.assertNotEqual(base.unit_key(), v.unit_key())
        # protocol versions are part of the key (frozen dataclass -> rebuild)
        import dataclasses as dc
        v2 = dc.replace(base, reset_protocol_version="v1_legacy")
        self.assertNotEqual(base.unit_key(), v2.unit_key())

    def test_atomic_write_no_tmp_left(self):
        with tempfile.TemporaryDirectory() as td:
            raw = Path(td)
            u = _mk_unit(num_episodes=2, num_envs=4)
            write_result_atomic(raw, u, _mk_row(u, nsuccs=[1, 2]))
            self.assertEqual(len(list(raw.glob("*.tmp"))), 0)
            self.assertTrue(result_path(raw, u).is_file())

    def test_corrupt_result_not_done(self):
        with tempfile.TemporaryDirectory() as td:
            raw = Path(td)
            u = _mk_unit(num_episodes=2, num_envs=4)
            result_path(raw, u).write_text("{not json")
            self.assertFalse(is_done(raw, u))


# ---------------------------------------------------------------------------
# 13: terminal error validity
# ---------------------------------------------------------------------------

class TestTerminalError(unittest.TestCase):
    def test_invalid_without_reliable_source(self):
        u = _mk_unit(num_episodes=2, num_envs=4)
        row = _mk_row(u, nsuccs=[1.0, 2.0], valid_terminal=False)
        self.assertFalse(row["terminal_error_valid"])
        self.assertTrue(math.isnan(row["terminal_error_mean"]))
        self.assertTrue(math.isnan(row["terminal_error_median"]))
        self.assertTrue(math.isnan(row["dropped_rate"]))
        for s in row["samples"]:
            self.assertFalse(s["terminal_error_valid"])
            self.assertTrue(math.isnan(s["terminal_error"]))
            self.assertIsNone(s["dropped"])

    def test_valid_with_snapshot(self):
        u = _mk_unit(num_episodes=2, num_envs=4)
        row = _mk_row(u, nsuccs=[1.0, 2.0], valid_terminal=True)
        self.assertTrue(row["terminal_error_valid"])
        self.assertFalse(math.isnan(row["terminal_error_mean"]))
        self.assertAlmostEqual(row["dropped_rate"], 0.0)

    def test_sample_forced_invalid_when_flag_false(self):
        s = EpisodeSample.create(reset_seed=1, env_id=0, nsucc=1.0,
                                 max_consecutive_successes=50,
                                 terminal_error=0.42, terminal_error_valid=False)
        self.assertTrue(math.isnan(s.terminal_error))  # value discarded

    def test_terminal_snapshot_hook_pre_reset(self):
        """Hook must capture PRE-reset state even though _reset_idx mutates
        the buffers (this is exactly the v1 bug scenario)."""
        import torch

        class FakeEnv:
            num_envs = 3
            object_pos = torch.tensor([[0., 0., 0.5], [0., 0., 0.05], [1., 0., 0.5]])
            goal_pos = torch.zeros(3, 3)

            def _reset_idx(self, env_ids):
                # simulate post-reset buffer refresh (what the real env does)
                for i in (env_ids.tolist() if hasattr(env_ids, "tolist") else env_ids):
                    self.object_pos[i] = torch.tensor([9.0, 9.0, 9.0])

        env = FakeEnv()
        hook = TerminalSnapshotHook(env)
        self.assertTrue(hook.install())
        env._reset_idx([1])  # done env 1: pre-reset error = 0.05 (z<0.1 -> dropped)
        err, valid, dropped = hook.take(1)
        self.assertTrue(valid)
        self.assertAlmostEqual(err, 0.05, places=6)
        self.assertTrue(dropped)
        # env 0 was not reset -> no terminal data
        err0, valid0, dropped0 = hook.take(0)
        self.assertFalse(valid0)
        self.assertTrue(math.isnan(err0))
        self.assertIsNone(dropped0)
        # buffers were indeed mutated post-hook (v1 read would see 9.0)
        self.assertAlmostEqual(float(env.object_pos[1][0]), 9.0)
        hook.uninstall()

    def test_hook_default_off_and_unavailable_env(self):
        class NoBuffersEnv:
            pass
        hook = TerminalSnapshotHook(NoBuffersEnv())
        self.assertFalse(hook.available)
        self.assertFalse(hook.install())
        self.assertFalse(hook.installed)

    def test_training_code_not_modified_by_hook(self):
        env_src = (Path(__file__).resolve().parents[1]
                   / "dexpbt_isaaclab" / "dexpbt_kuka_allegro_env.py").read_text()
        self.assertNotIn("TerminalSnapshotHook", env_src)
        self.assertNotIn("ccrs", env_src.lower())


# ---------------------------------------------------------------------------
# 14/15/16: exact transitions, missing cells, lineage
# ---------------------------------------------------------------------------

class TestExactTransitions(unittest.TestCase):
    def _table_exact(self):
        t = CCRSTable()
        a, b = 0.0249, 0.0237   # real (non-grid) transition tolerances
        for seed in (1, 2):
            for it, tol, ns in [(146, a, 44.0), (146, b, 40.0),
                                (147, a, 43.0), (147, b, 36.0)]:
                t.add_row(_table_row(1, it, tol, seed, [ns, ns - 4.0]))
        return t, a, b

    def test_exact_uses_real_tolerances(self):
        t, a, b = self._table_exact()
        res = compute_dfc(t, 1, 146, 147, tol_pre=a, tol_post=b, n_boot=50)
        self.assertEqual(res.mode, "exact")
        self.assertAlmostEqual(res.tol_pre, 0.0249)
        self.assertAlmostEqual(res.tol_post, 0.0237)
        self.assertLess(res.identity_residual, 1e-12)
        self.assertIn("D", res.ci)

    def test_missing_exact_cells_refused(self):
        t, a, b = self._table_exact()
        # ask for grid tolerances that were never evaluated
        with self.assertRaises(KeyError) as ctx:
            compute_dfc(t, 1, 146, 147, tol_pre=0.025, tol_post=0.023)
        self.assertIn("missing exact transition cells", str(ctx.exception))

    def test_grid_projected_labelled(self):
        t = CCRSTable()
        for seed in (1, 2):
            for it, tol, ns in [(146, 0.025, 44.0), (146, 0.023, 40.0),
                                (147, 0.025, 43.0), (147, 0.023, 36.0)]:
                t.add_row(_table_row(1, it, tol, seed, [ns, ns - 4.0]))
        res = grid_projected_dfc(
            t, 1, 146, 147, tol_pre_real=0.0249, tol_post_real=0.0237,
            grid=[0.028, 0.025, 0.023, 0.020], n_boot=50)
        self.assertEqual(res.mode, "grid_projected")   # cannot be confused with exact
        self.assertAlmostEqual(res.tol_pre, 0.025)
        self.assertAlmostEqual(res.tol_post, 0.023)

    def test_grid_projection_collapse_refused(self):
        t, a, b = self._table_exact()
        with self.assertRaises(ValueError):
            grid_projected_dfc(t, 1, 146, 147, tol_pre_real=0.0241,
                               tol_post_real=0.0239, grid=[0.024], n_boot=10)

    def test_lineage_transition_kinds(self):
        t, a, b = self._table_exact()
        r1 = compute_dfc(t, 1, 146, 147, tol_pre=a, tol_post=b,
                         n_boot=10, lineage_changed=False)
        self.assertEqual(r1.transition_kind, "within_lineage")
        r2 = compute_dfc(t, 1, 146, 147, tol_pre=a, tol_post=b,
                         n_boot=10, lineage_changed=True)
        self.assertEqual(r2.transition_kind, "lineage_replacement_transition")

    def test_dfc_identity_random(self):
        rng = random.Random(0)
        for _ in range(200):
            vals = [rng.random() for _ in range(4)]
            D, F, C = dfc_from_Q(*vals)
            self.assertAlmostEqual(D, F + C, places=12)


class TestManifestLineage(unittest.TestCase):
    def _mk_run(self, td: Path, lineage_change=True):
        ws = td / "run_x" / "dexpbt_8gpu_ws" / "001"
        ws.mkdir(parents=True)
        specs = [
            (10, 200_000_000, 0.075, 5.0, "lin-A"),
            (12, 240_000_000, 0.070, 8.0, "lin-A"),
            (14, 280_000_000, 0.022, 20.0, "lin-B" if lineage_change else "lin-A"),
            (16, 320_000_000, 0.021, 22.0, "lin-B" if lineage_change else "lin-A"),
        ]
        for it, frame, tol, ns, lin in specs:
            ckpt = ws / f"{it:06d}.pth"
            ckpt.write_bytes(b"PK\x03\x04fake")
            meta = {
                "policy_idx": 1, "iteration": it, "frame": frame,
                "nsucc_mean": ns, "lineage_id": lin,
                "curriculum": {"success_tolerance": tol},
                "checkpoint": str(ckpt),
            }
            (ws / f"{it:06d}.json").write_text(json.dumps(meta))
        return td / "run_x"

    def test_lineage_changed_detected(self):
        with tempfile.TemporaryDirectory() as td:
            run = self._mk_run(Path(td), lineage_change=True)
            rows, trans = ccrs_manifest.build_manifest(
                run_dir=run, repo_root=Path(td), targets={}, torch_check="none")
            # 0.070->0.022 crosses lineage A->B; 0.022->0.021 stays within B
            t1 = [t for t in trans if abs(t.post_tolerance - 0.022) < 1e-9][0]
            self.assertTrue(t1.lineage_changed)
            self.assertEqual(t1.pre_lineage_id, "lin-A")
            self.assertEqual(t1.post_lineage_id, "lin-B")
            t2 = [t for t in trans if abs(t.post_tolerance - 0.021) < 1e-9][0]
            self.assertFalse(t2.lineage_changed)

    def test_lineage_unchanged(self):
        with tempfile.TemporaryDirectory() as td:
            run = self._mk_run(Path(td), lineage_change=False)
            rows, trans = ccrs_manifest.build_manifest(
                run_dir=run, repo_root=Path(td), targets={}, torch_check="none")
            self.assertTrue(all(t.lineage_changed is False for t in trans))

    def test_manifest_core_fields(self):
        with tempfile.TemporaryDirectory() as td:
            run = self._mk_run(Path(td))
            rows, trans = ccrs_manifest.build_manifest(
                run_dir=run, repo_root=Path(td),
                targets={1: [240_000_000]}, torch_check="none")
            by_iter = {r.iteration: r for r in rows}
            self.assertEqual(by_iter[12].frame, 240_000_000)
            self.assertEqual(by_iter[12].frame_deviation, 0)
            self.assertAlmostEqual(by_iter[14].live_tolerance, 0.022)
            out = Path(td) / "out"
            csv_path, md_path = ccrs_manifest.write_outputs(rows, trans, out, run)
            header = csv_path.read_text().splitlines()[0]
            for col in ("run_id", "git_commit", "training_commit", "worker_id",
                        "frame", "checkpoint_path", "live_tolerance", "live_nsucc",
                        "historical_peak_nsucc", "lineage_id", "policy_arch"):
                self.assertIn(col, header)
            theader = (out / "tolerance_transitions.csv").read_text().splitlines()[0]
            for col in ("pre_lineage_id", "post_lineage_id", "lineage_changed"):
                self.assertIn(col, theader)


# ---------------------------------------------------------------------------
# 17: schema versioning
# ---------------------------------------------------------------------------

class TestSchemaVersioning(unittest.TestCase):
    def test_old_schema_excluded_from_aggregation(self):
        with tempfile.TemporaryDirectory() as td:
            raw = Path(td) / "raw"
            raw.mkdir()
            u = _mk_unit(num_episodes=2, num_envs=4)
            good = _mk_row(u, nsuccs=[1, 2])
            write_result_atomic(raw, u, good)
            # v1-style row (old schema, old metric)
            old = dict(good)
            old["schema_version"] = 1
            (raw / "old_v1_result.json").write_text(json.dumps(old))
            n = aggregate_raw_dir(raw, Path(td) / "r.jsonl", Path(td) / "r.csv")
            self.assertEqual(n, 1)

    def test_old_schema_not_done(self):
        with tempfile.TemporaryDirectory() as td:
            raw = Path(td)
            u = _mk_unit(num_episodes=2, num_envs=4)
            row = _mk_row(u, nsuccs=[1, 2])
            row["schema_version"] = 1
            write_result_atomic(raw, u, row)
            self.assertFalse(is_done(raw, u))

    def test_table_rejects_old_schema(self):
        t = CCRSTable()
        r = _table_row(1, 100, 0.02, 1, [1.0])
        r["schema_version"] = 1
        t.add_row(r)
        self.assertTrue(t.is_empty())
        self.assertEqual(t.rejected_rows, 1)

    def test_schema_version_bumped(self):
        self.assertGreaterEqual(SCHEMA_VERSION, 2)
        self.assertEqual(RESET_PROTOCOL_VERSION, "v2_first_episode_per_env")


# ---------------------------------------------------------------------------
# 18: report empty-data behaviour + synthetic end-to-end
# ---------------------------------------------------------------------------

class TestReport(unittest.TestCase):
    def test_report_dry_mode_no_fabrication(self):
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "out"
            s = report_generate(
                results_jsonl=Path(td) / "missing.jsonl",
                manifest_csv=Path(td) / "missing.csv",
                transitions_csv=Path(td) / "missing_t.csv",
                out_dir=out,
            )
            self.assertEqual(s["n_cells"], 0)
            self.assertEqual(s["primary_metric"], "normalized_nsucc")
            self.assertTrue(any("no evaluation results" in m for m in s["missing"]))
            md = (out / "CCRS_REPORT.md").read_text()
            self.assertIn("No evaluation data found", md)
            self.assertIn("Pre-registered decision criteria", md)
            self.assertIn("normalized_nsucc", md)
            self.assertTrue((out / "ccrs_summary.json").is_file())

    def test_report_with_synthetic_data(self):
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "out"
            rows = []
            for seed in (1, 2, 3):
                for (w, it, tol, ns) in [
                    (1, 149, 0.022, 40.0), (1, 149, 0.020, 30.0),
                    (4, 141, 0.022, 30.0), (4, 141, 0.020, 38.0),
                ]:
                    rows.append(_table_row(w, it, tol, seed,
                                           [ns, ns - 2, ns + 2, ns - 1],
                                           frame=it * 10**7))
            res = Path(td) / "r.jsonl"
            res.write_text("\n".join(json.dumps(r) for r in rows))
            man = Path(td) / "m.csv"
            man.write_text(
                "run_id,git_commit,worker_id,iteration,frame,checkpoint_path,"
                "live_tolerance,live_nsucc,historical_peak_nsucc,lineage_id,"
                "policy_arch,roles,notes\n"
                "r,x,1,149,4120000000,x,0.0237,43.4,43.4,l,arch,peak,\n"
                "r,x,4,141,3480000000,x,0.0214,40.2,40.2,l,arch,peak,\n"
            )
            tr = Path(td) / "t.csv"
            tr.write_text(
                "worker_id,pre_iteration,post_iteration,pre_frame,post_frame,"
                "pre_tolerance,post_tolerance,pre_nsucc,post_nsucc,"
                "pre_lineage_id,post_lineage_id,lineage_changed\n"
                "1,149,150,4120000000,4200000000,0.0299,0.0288,43.4,38.4,l,l,False\n"
            )
            s = report_generate(res, man, tr, out, n_boot=100)
            self.assertGreater(s["n_cells"], 0)
            self.assertIn("verdicts", s)
            # the transition (0.0299/0.0288) was never evaluated -> missing exact
            self.assertTrue(any("missing exact transition cells" in m
                                for m in s["missing"]))
            md = (out / "CCRS_REPORT.md").read_text()
            self.assertIn("normalized_nsucc", md)
            self.assertIn("p4 − p1", md)


# ---------------------------------------------------------------------------
# regression: v1 invariants kept
# ---------------------------------------------------------------------------

class TestFixedToleranceInvariant(unittest.TestCase):
    def test_tolerance_drift_raises(self):
        u = _mk_unit(num_episodes=2, num_envs=4)
        with self.assertRaises(RuntimeError):
            _mk_row(u, nsuccs=[1, 2], applied_tolerance_start=0.020,
                    applied_tolerance_end=0.019)

    def test_tolerance_stable_ok(self):
        u = _mk_unit(num_episodes=2, num_envs=4)
        row = _mk_row(u, nsuccs=[1, 2], applied_tolerance_start=0.020,
                      applied_tolerance_end=0.020)
        self.assertEqual(row["status"], "ok")

    def test_tolerance_unit_is_meters(self):
        with self.assertRaises(ValueError):
            build_units(
                run_id="r", checkpoints=[dict(worker_id=0, iteration=1, frame=1,
                                              checkpoint_path="x")],
                tolerances_m=[2.0],  # cm passed as meters
                seeds=[1], num_envs=4, num_episodes=4,
            )
        row = _mk_row(_mk_unit(num_episodes=2, num_envs=4), nsuccs=[1, 2])
        self.assertAlmostEqual(row["fixed_tolerance_cm"], row["fixed_tolerance"] * 100)

    def test_duplicate_env_id_rejected(self):
        u = _mk_unit(num_episodes=2, num_envs=4)
        s = _mk_samples(u, nsuccs=[1.0, 2.0])
        dup = s + [s[0]]
        with self.assertRaises(ValueError):
            make_result_row(u, dup, _mk_fps(u), "sha", "ck")


class TestBuildUnitsExactTolerances(unittest.TestCase):
    def test_per_checkpoint_tolerances_override_grid(self):
        cks = [
            dict(worker_id=1, iteration=10, frame=1, checkpoint_path="a"),
            dict(worker_id=1, iteration=12, frame=2, checkpoint_path="b",
                 tolerances=[0.0249, 0.0237]),
        ]
        units = build_units("r", cks, tolerances_m=[0.028, 0.025],
                            seeds=[1], num_envs=4, num_episodes=4)
        by_it = {}
        for u in units:
            by_it.setdefault(u.checkpoint_iteration, []).append(u.fixed_tolerance)
        self.assertEqual(sorted(by_it[10]), [0.025, 0.028])
        self.assertEqual(sorted(by_it[12]), [0.0237, 0.0249])

    def test_dedupe(self):
        ck = dict(worker_id=1, iteration=10, frame=1, checkpoint_path="a")
        units = build_units("r", [ck, dict(ck)], tolerances_m=[0.02],
                            seeds=[1], num_envs=4, num_episodes=4)
        self.assertEqual(len(units), 1)


class TestHierarchicalBootstrap(unittest.TestCase):
    def test_unpaired_rejected(self):
        cells = {"a": {(1, 0): 0.5, (1, 1): 0.6}, "b": {(1, 0): 0.4, (2, 0): 0.5}}
        with self.assertRaises(ResetPairingError):
            hierarchical_paired_bootstrap(cells, lambda m: {"d": m["a"] - m["b"]},
                                          n_boot=10)

    def test_single_sample_rejected(self):
        cells = {"a": {(1, 0): 0.5}, "b": {(1, 0): 0.4}}
        with self.assertRaises(ResetPairingError):
            hierarchical_paired_bootstrap(cells, lambda m: {"d": m["a"] - m["b"]},
                                          n_boot=10)

    def test_ci_brackets_truth(self):
        rng = random.Random(1)
        a, b = {}, {}
        for s in (1, 2, 3):
            for e in range(20):
                a[(s, e)] = 0.7 + 0.02 * rng.random()
                b[(s, e)] = 0.5 + 0.02 * rng.random()
        ci = hierarchical_paired_bootstrap(
            {"a": a, "b": b}, lambda m: {"d": m["a"] - m["b"]}, n_boot=400)
        lo, hi = ci["d"]
        self.assertGreater(lo, 0.15)
        self.assertLess(hi, 0.25)


class TestStatsHelpers(unittest.TestCase):
    def test_wilson(self):
        lo, hi = wilson_ci(50, 100)
        self.assertTrue(0.40 < lo < 0.5 < hi < 0.60)
        self.assertTrue(math.isnan(wilson_ci(0, 0)[0]))

    def test_quantiles(self):
        q = quantiles([1.0, 2.0, 3.0, 4.0, 5.0], [0.5, 0.25])
        self.assertAlmostEqual(q["q50"], 3.0)
        self.assertAlmostEqual(q["q25"], 2.0)

    def test_mean_ci(self):
        m, lo, hi = mean_ci95([1.0, 2.0, 3.0])
        self.assertAlmostEqual(m, 2.0)
        self.assertTrue(lo < 2.0 < hi)


class TestVerdicts(unittest.TestCase):
    def _diff(self, tol, d, lo, hi, pairing=True):
        return PairwiseDiff(tolerance=tol, diff=d, ci_lo=lo, ci_hi=hi,
                            n_samples=100,
                            significant=pairing and (lo > 0 or hi < 0),
                            reset_pairing_valid=pairing)

    def test_crossing_verdict(self):
        p4_minus_p1 = [
            self._diff(0.022, -0.10, -0.15, -0.05),
            self._diff(0.020, +0.10, +0.05, +0.15),
        ]
        v = preregistered_verdicts(p4_minus_p1, [])
        self.assertEqual(v["verdict_p1_p4"], "A_capability_curve_crossing")

    def test_p4_dominant_verdict(self):
        p4_minus_p1 = [
            self._diff(0.023, +0.08, +0.02, +0.14),
            self._diff(0.021, +0.09, +0.03, +0.15),
            self._diff(0.019, +0.12, +0.06, +0.18),
        ]
        v = preregistered_verdicts(p4_minus_p1, [])
        self.assertEqual(v["verdict_p1_p4"], "B_p4_globally_stronger")

    def test_p5_forgetting_vs_reclassification(self):
        drop = [self._diff(0.021, +0.20, +0.10, +0.30)]
        flat = [self._diff(0.021, +0.01, -0.03, +0.05)]
        self.assertEqual(preregistered_verdicts([], drop)["verdict_p5"],
                         "real_policy_forgetting")
        self.assertEqual(preregistered_verdicts([], flat)["verdict_p5"],
                         "criterion_reclassification_dominant")

    def test_pairing_invalid_verdicts(self):
        bad = [self._diff(0.020, +0.30, float("nan"), float("nan"), pairing=False)]
        v = preregistered_verdicts(bad, bad)
        self.assertEqual(v["verdict_p1_p4"], "reset_pairing_invalid")
        self.assertEqual(v["verdict_p5"], "reset_pairing_invalid")


class TestProvenance(unittest.TestCase):
    def test_source_sha256_stable_and_sensitive(self):
        from dexpbt_isaaclab.ccrs.eval_protocol import source_sha256
        with tempfile.TemporaryDirectory() as td:
            a = Path(td) / "a.py"
            b = Path(td) / "b.py"
            a.write_text("x = 1\n")
            b.write_text("y = 2\n")
            h1 = source_sha256(a, b)
            h2 = source_sha256(b, a)          # order-independent
            self.assertEqual(h1, h2)
            b.write_text("y = 3\n")           # content change -> new hash
            self.assertNotEqual(h1, source_sha256(a, b))

    def test_git_dirty_detection(self):
        import subprocess
        from dexpbt_isaaclab.ccrs.eval_protocol import git_dirty
        with tempfile.TemporaryDirectory() as td:
            repo = Path(td)
            subprocess.run(["git", "init", "-q"], cwd=td, check=True)
            f = repo / "f.txt"
            f.write_text("clean")
            subprocess.run(["git", "add", "f.txt"], cwd=td, check=True)
            subprocess.run(
                ["git", "-c", "user.name=t", "-c", "user.email=t@t",
                 "commit", "-qm", "init"], cwd=td, check=True)
            self.assertFalse(git_dirty(repo))
            f.write_text("dirty")             # tracked modification
            self.assertTrue(git_dirty(repo))
        # non-repo -> None (unknown), not a crash
        with tempfile.TemporaryDirectory() as td2:
            self.assertIsNone(git_dirty(Path(td2)))

    def test_result_row_carries_analysis_protocol_version(self):
        from dexpbt_isaaclab.ccrs.eval_protocol import ANALYSIS_PROTOCOL_VERSION
        u = _mk_unit(num_episodes=2, num_envs=4)
        row = _mk_row(u, nsuccs=[1.0, 2.0])
        self.assertEqual(row["analysis_protocol_version"], ANALYSIS_PROTOCOL_VERSION)

    def test_ccrs_eval_emits_provenance_fields(self):
        src = (Path(__file__).resolve().parents[1]
               / "dexpbt_isaaclab" / "scripts" / "ccrs_eval.py").read_text()
        for fieldname in ("git_dirty", "evaluator_source_sha256",
                          "analysis_protocol_version"):
            self.assertIn(fieldname, src)
        self.assertIn("_code_provenance()", src)


class TestFrontier(unittest.TestCase):
    def _table(self, crossing=True):
        """A: better at tight tols; B: better at loose tols (if crossing)."""
        t = CCRSTable()
        for seed in (1, 2):
            for tol, qa, qb in [(0.028, 30.0, 40.0), (0.021, 32.0, 36.0),
                                (0.019, 38.0, 25.0), (0.018, 40.0, 20.0)]:
                if not crossing:
                    qa = qb + 5.0
                t.add_row(_table_row(1, 160, tol, seed, [qa, qa - 2, qa + 2, qa - 1]))
                t.add_row(_table_row(5, 117, tol, seed, [qb, qb - 2, qb + 2, qb - 1]))
        return t

    def test_compare_checkpoints_fields(self):
        from dexpbt_isaaclab.ccrs.frontier import compare_checkpoints
        t = self._table()
        pts = compare_checkpoints(t, (1, 160), (5, 117),
                                  [0.028, 0.018], n_boot=200,
                                  expected_paired_samples=8)
        self.assertEqual(len(pts), 2)
        for p in pts:
            self.assertTrue(p.reset_pairing_valid)
            self.assertTrue(p.paired_samples_complete)
            self.assertEqual(p.n_paired_samples, 8)
            self.assertFalse(math.isnan(p.ci_lo))
        by_tol = {round(p.tolerance, 3): p for p in pts}
        self.assertLess(by_tol[0.028].diff, 0)   # B better loose
        self.assertGreater(by_tol[0.018].diff, 0)  # A better tight
        self.assertTrue(by_tol[0.018].practical)

    def test_incomplete_paired_samples_flagged(self):
        from dexpbt_isaaclab.ccrs.frontier import compare_checkpoints
        t = self._table()
        pts = compare_checkpoints(t, (1, 160), (5, 117), [0.028],
                                  n_boot=50, expected_paired_samples=2048)
        self.assertFalse(pts[0].paired_samples_complete)

    def test_crossing_assessment(self):
        from dexpbt_isaaclab.ccrs.frontier import assess_crossing, compare_checkpoints
        t = self._table(crossing=True)
        pts = compare_checkpoints(t, (1, 160), (5, 117),
                                  [0.028, 0.021, 0.019, 0.018], n_boot=200,
                                  expected_paired_samples=8)
        c = assess_crossing(pts, region_b_better=[0.028, 0.021],
                            region_a_better=[0.019, 0.018])
        self.assertTrue(c["numerical_crossing"])
        self.assertTrue(c["practical_crossing"])
        c2 = assess_crossing(
            compare_checkpoints(self._table(crossing=False), (1, 160), (5, 117),
                                [0.028, 0.021, 0.019, 0.018], n_boot=100,
                                expected_paired_samples=8),
            region_b_better=[0.028, 0.021], region_a_better=[0.019, 0.018])
        self.assertFalse(c2["numerical_crossing"])
        self.assertFalse(c2["practical_crossing"])

    def test_envelope_numerical_vs_practical_separate(self):
        from dexpbt_isaaclab.ccrs.frontier import best_checkpoint_envelope
        t = self._table()
        env = best_checkpoint_envelope(
            t, {(1, 160): "p1 tip", (5, 117): "p5 peak"},
            [0.028, 0.018], n_boot=200)
        by_tol = {round(e.tolerance, 3): e for e in env}
        self.assertEqual(by_tol[0.028].best_label, "p5 peak")
        self.assertEqual(by_tol[0.018].best_label, "p1 tip")
        # numerical best is always defined; practical is a separate flag
        for e in env:
            self.assertIsInstance(e.practical, bool)
            self.assertGreaterEqual(e.diff_best_second, 0.0)


class TestEvaluatorSourceInvariants(unittest.TestCase):
    """Source-level checks: secondary evidence only (see module docstring)."""

    SRC = (Path(__file__).resolve().parents[1]
           / "dexpbt_isaaclab" / "scripts" / "ccrs_eval.py").read_text()

    def test_no_optimizer_or_pbt_paths(self):
        for forbidden in ("create_agent(", "torch.optim", "PbtObserver",
                          "pbt_observer", "optimizer.step", "loss.backward"):
            self.assertNotIn(forbidden, self.SRC, forbidden)
        self.assertIn("create_player()", self.SRC)
        self.assertIn("requires_grad_(False)", self.SRC)
        self.assertIn('assert not hasattr(agent, "optimizer")', self.SRC)

    def test_policy_only_no_wm_snippet_mppi_residual(self):
        code_lines = [ln for ln in self.SRC.splitlines()
                      if not ln.strip().startswith("#")]
        in_doc = False
        code = []
        for ln in code_lines:
            s = ln.strip()
            if s.startswith('"""') or s.endswith('"""'):
                if not (s.startswith('"""') and s.endswith('"""') and len(s) > 3):
                    in_doc = not in_doc
                continue
            if not in_doc:
                code.append(ln)
        joined = "\n".join(code).lower()
        for forbidden in ("snippet", "mppi", "wm_rerank", "distill",
                          "elitebank", "seq_wm", "world_model"):
            self.assertNotIn(forbidden, joined,
                             f"policy-only eval must not reference {forbidden}")

    def test_curriculum_frozen_in_eval(self):
        self.assertIn("last_curriculum_update = 10 ** 9", self.SRC)
        self.assertIn("initial_tolerance = float(unit.fixed_tolerance)", self.SRC)
        self.assertIn("target_tolerance = float(unit.fixed_tolerance)", self.SRC)

    def test_system_on_refused(self):
        self.assertIn("system_on mode is reserved", self.SRC)

    def test_first_episode_and_fingerprint_wired(self):
        self.assertIn("FirstEpisodeCollector", self.SRC)
        self.assertIn("extract_reset_fingerprints", self.SRC)
        self.assertIn("TerminalSnapshotHook", self.SRC)
        # fingerprints taken before first action
        idx_fp = self.SRC.index("extract_reset_fingerprints(raw_env")
        idx_act = self.SRC.index("agent.get_action(obs_t")
        self.assertLess(idx_fp, idx_act)


if __name__ == "__main__":
    unittest.main(verbosity=2)
