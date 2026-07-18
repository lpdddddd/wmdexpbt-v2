"""CCRS toolkit tests (no Isaac, no GPU).

Covers the preparation-phase requirements:
1. fixed tolerance cannot drift during eval (make_result_row raises);
2. evaluator performs no optimizer/PBT work (source-level invariants);
3. same seed -> same unit identity across checkpoints/tolerances (env seed
   is fully determined by reset_seed, asserted at protocol level);
4. manifest correctly identifies worker/frame/tolerance/transitions;
5. done units are skipped (resume);
6. synthetic CCRS data satisfies D = F + C exactly;
7. missing checkpoints / unpaired seeds -> loud errors or explicit notes;
8. policy-only mode: system_on refused, eval path has no training imports;
9. tolerance units are meters everywhere; cm only derived for display.
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
    EvalUnit,
    aggregate_raw_dir,
    build_units,
    is_done,
    make_result_row,
    mean_ci95,
    quantiles,
    result_path,
    wilson_ci,
)
from dexpbt_isaaclab.ccrs.analysis import (  # noqa: E402
    CCRSTable,
    CellKey,
    compute_dfc,
    curve_difference,
    dfc_from_R,
    paired_bootstrap_ci,
    preregistered_verdicts,
    PairwiseDiff,
)
from dexpbt_isaaclab.ccrs import manifest as ccrs_manifest  # noqa: E402
from dexpbt_isaaclab.ccrs.report import generate as report_generate  # noqa: E402


def _mk_unit(**kw) -> EvalUnit:
    base = dict(
        run_id="run_x", worker_id=1, checkpoint_iteration=149,
        checkpoint_frame=4_120_000_000,
        checkpoint_path="/tmp/ckpt.pth",
        fixed_tolerance=0.020, reset_seed=10001,
        num_envs=8, num_episodes=16,
    )
    base.update(kw)
    return EvalUnit(**base)


def _mk_row(unit: EvalUnit, nsucc=None, err=None, **kw):
    nsucc = nsucc if nsucc is not None else [1.0, 0.0, 2.0, 3.0]
    err = err if err is not None else [0.01, 0.03, 0.005, 0.002]
    return make_result_row(
        unit=unit, episode_nsucc=nsucc, episode_final_error=err,
        git_sha="deadbeef", checkpoint_hash="cafebabe", **kw,
    )


class TestFixedToleranceInvariant(unittest.TestCase):
    def test_tolerance_drift_raises(self):
        u = _mk_unit()
        with self.assertRaises(RuntimeError):
            _mk_row(u, applied_tolerance_start=0.020, applied_tolerance_end=0.019)

    def test_tolerance_stable_ok(self):
        u = _mk_unit()
        row = _mk_row(u, applied_tolerance_start=0.020, applied_tolerance_end=0.020)
        self.assertEqual(row["status"], "ok")
        self.assertAlmostEqual(row["fixed_tolerance"], 0.020)

    def test_tolerance_unit_is_meters(self):
        # cm passed as if meters must be rejected by build_units
        with self.assertRaises(ValueError):
            build_units(
                run_id="r", checkpoints=[dict(worker_id=0, iteration=1, frame=1,
                                              checkpoint_path="x")],
                tolerances_m=[2.0],  # 2.0 "m" = someone passed cm
                seeds=[1], num_envs=4, num_episodes=4,
            )
        row = _mk_row(_mk_unit())
        self.assertAlmostEqual(row["fixed_tolerance_cm"], row["fixed_tolerance"] * 100)


class TestEvaluatorSafety(unittest.TestCase):
    """Source-level guarantees for ccrs_eval (cannot boot Isaac in unit tests)."""

    SRC = (Path(__file__).resolve().parents[1]
           / "dexpbt_isaaclab" / "scripts" / "ccrs_eval.py").read_text()

    def test_no_optimizer_or_pbt_paths(self):
        # never constructs a trainer / optimizer / observer
        for forbidden in ("create_agent(", "torch.optim", "PbtObserver",
                          "pbt_observer", "optimizer.step", "loss.backward"):
            self.assertNotIn(forbidden, self.SRC, forbidden)
        # player-only path present, gradients disabled
        self.assertIn("create_player()", self.SRC)
        self.assertIn("requires_grad_(False)", self.SRC)
        self.assertIn('assert not hasattr(agent, "optimizer")', self.SRC)

    def test_policy_only_no_wm_snippet_mppi_residual(self):
        for forbidden in ("snippet", "SnippetBank", "mppi", "MPPI",
                          "residual", "wm_rerank", "distill", "EliteBank",
                          "seq_wm", "world_model"):
            # allowed only in comments/docstring; strip them crudely
            code_lines = [
                ln for ln in self.SRC.splitlines()
                if not ln.strip().startswith("#")
            ]
            in_doc = False
            code = []
            for ln in code_lines:
                s = ln.strip()
                if s.startswith('"""') or s.endswith('"""'):
                    in_doc = not in_doc if not (s.startswith('"""') and s.endswith('"""') and len(s) > 3) else in_doc
                    continue
                if not in_doc:
                    code.append(ln)
            self.assertNotIn(forbidden.lower(), "\n".join(code).lower(),
                             f"policy-only eval must not reference {forbidden}")

    def test_curriculum_frozen_in_eval(self):
        # env-side curriculum disabled the same way eval_fair does
        self.assertIn("last_curriculum_update = 10 ** 9", self.SRC)
        self.assertIn("initial_tolerance = float(unit.fixed_tolerance)", self.SRC)
        self.assertIn("target_tolerance = float(unit.fixed_tolerance)", self.SRC)

    def test_system_on_refused(self):
        self.assertIn("system_on mode is reserved", self.SRC)


class TestSeedIdentity(unittest.TestCase):
    def test_same_seed_same_reset_identity(self):
        """Reset conditions are determined solely by reset_seed: two units at
        the same seed but different checkpoint/tolerance must carry the same
        env seed; different seeds must differ."""
        u1 = _mk_unit(fixed_tolerance=0.020, checkpoint_iteration=149)
        u2 = _mk_unit(fixed_tolerance=0.018, checkpoint_iteration=160)
        self.assertEqual(u1.reset_seed, u2.reset_seed)
        # unit identity differs (different cells) but seed component equal
        self.assertNotEqual(u1.unit_key(), u2.unit_key())
        u3 = _mk_unit(reset_seed=10002)
        self.assertNotEqual(u1.unit_key(), u3.unit_key())

    def test_unit_key_stable(self):
        self.assertEqual(_mk_unit().unit_key(), _mk_unit().unit_key())


class TestResumeSkip(unittest.TestCase):
    def test_done_and_skip(self):
        with tempfile.TemporaryDirectory() as td:
            raw = Path(td)
            u = _mk_unit()
            self.assertFalse(is_done(raw, u))
            row = _mk_row(u)
            result_path(raw, u).write_text(json.dumps(row))
            self.assertTrue(is_done(raw, u))
            # corrupt -> not done
            result_path(raw, u).write_text("{not json")
            self.assertFalse(is_done(raw, u))

    def test_aggregate(self):
        with tempfile.TemporaryDirectory() as td:
            raw = Path(td) / "raw"
            raw.mkdir()
            for seed in (1, 2):
                u = _mk_unit(reset_seed=seed)
                result_path(raw, u).write_text(json.dumps(_mk_row(u)))
            n = aggregate_raw_dir(raw, Path(td) / "r.jsonl", Path(td) / "r.csv")
            self.assertEqual(n, 2)
            lines = (Path(td) / "r.jsonl").read_text().strip().splitlines()
            self.assertEqual(len(lines), 2)
            # per-episode arrays preserved in jsonl
            self.assertIn("per_episode_nsucc", json.loads(lines[0]))


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


class TestDFC(unittest.TestCase):
    def test_identity_exact(self):
        rng = random.Random(0)
        for _ in range(200):
            vals = [rng.random() for _ in range(4)]
            D, F, C = dfc_from_R(*vals)
            self.assertAlmostEqual(D, F + C, places=12)

    def _table(self, seeds=(1, 2, 3, 4)):
        """Synthetic table: R(p,.) strong, R(q,.) weaker at tight tol."""
        t = CCRSTable()
        truth = {
            (1, 100, 0.023): 0.9, (1, 100, 0.020): 0.8,   # p (pre)
            (1, 101, 0.023): 0.85, (1, 101, 0.020): 0.55,  # q (post)
        }
        for (w, it, tol), r in truth.items():
            for s in seeds:
                jitter = 0.01 * ((s % 3) - 1)
                t.add_row({
                    "status": "ok", "worker_id": w, "checkpoint_iteration": it,
                    "fixed_tolerance": tol, "reset_seed": s,
                    "success_rate": r + jitter, "nsucc_mean": 10 * (r + jitter),
                    "checkpoint_frame": it * 10, "checkpoint_path": "x",
                })
        return t

    def test_compute_dfc_and_identity(self):
        t = self._table()
        res = compute_dfc(t, worker_id=1, pre_iteration=100, post_iteration=101,
                          tol_pre=0.023, tol_post=0.020, n_boot=200)
        self.assertLess(res.identity_residual, 1e-12)
        self.assertAlmostEqual(res.D, res.R_pa - res.R_qb, places=12)
        self.assertEqual(res.n_seeds, 4)
        self.assertIn("D", res.ci)

    def test_missing_cells_raise(self):
        t = self._table()
        with self.assertRaises(KeyError):
            compute_dfc(t, worker_id=1, pre_iteration=100, post_iteration=999,
                        tol_pre=0.023, tol_post=0.020)

    def test_tol_order_enforced(self):
        t = self._table()
        with self.assertRaises(ValueError):
            compute_dfc(t, worker_id=1, pre_iteration=100, post_iteration=101,
                        tol_pre=0.020, tol_post=0.023)


class TestPairedBootstrap(unittest.TestCase):
    def test_unpaired_seeds_rejected(self):
        cells = {"a": {1: 0.5, 2: 0.6}, "b": {1: 0.4, 3: 0.5}}
        with self.assertRaises(ValueError):
            paired_bootstrap_ci(cells, lambda m: {"d": m["a"] - m["b"]}, n_boot=10)

    def test_single_seed_rejected(self):
        cells = {"a": {1: 0.5}, "b": {1: 0.4}}
        with self.assertRaises(ValueError):
            paired_bootstrap_ci(cells, lambda m: {"d": m["a"] - m["b"]}, n_boot=10)

    def test_ci_brackets_truth(self):
        rng = random.Random(1)
        a = {s: 0.7 + 0.02 * rng.random() for s in range(12)}
        b = {s: 0.5 + 0.02 * rng.random() for s in range(12)}
        ci = paired_bootstrap_ci({"a": a, "b": b},
                                 lambda m: {"d": m["a"] - m["b"]}, n_boot=500)
        lo, hi = ci["d"]
        self.assertTrue(lo < 0.21 < hi or (0.15 < lo and hi < 0.25))
        self.assertGreater(lo, 0.1)


class TestCurveDifferenceAndVerdicts(unittest.TestCase):
    def _diff(self, tol, d, lo, hi):
        return PairwiseDiff(tolerance=tol, diff=d, ci_lo=lo, ci_hi=hi,
                            n_seeds=4, significant=(lo > 0 or hi < 0))

    def test_crossing_verdict(self):
        p4_minus_p1 = [
            self._diff(0.022, -0.10, -0.15, -0.05),  # p1 wins mid
            self._diff(0.020, +0.10, +0.05, +0.15),  # p4 wins high-precision
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

    def test_curve_difference_paired(self):
        t = CCRSTable()
        for s in (1, 2, 3):
            for (w, it, r) in [(4, 141, 0.8), (1, 149, 0.6)]:
                t.add_row({"status": "ok", "worker_id": w,
                           "checkpoint_iteration": it, "fixed_tolerance": 0.020,
                           "reset_seed": s, "success_rate": r + 0.01 * s,
                           "nsucc_mean": 1.0, "checkpoint_frame": 0,
                           "checkpoint_path": "x"})
        diffs = curve_difference(t, a=(4, 141), b=(1, 149), tolerances=[0.020],
                                 n_boot=100)
        self.assertEqual(len(diffs), 1)
        self.assertAlmostEqual(diffs[0].diff, 0.2, places=6)


class TestManifest(unittest.TestCase):
    def _mk_run(self, td: Path):
        """Synthetic workspace with metas + fake ckpts + one shrink transition."""
        ws = td / "run_x" / "dexpbt_8gpu_ws" / "001"
        ws.mkdir(parents=True)
        specs = [
            (10, 200_000_000, 0.075, 5.0),
            (12, 240_000_000, 0.070, 8.0),
            (14, 280_000_000, 0.022, 20.0),   # shrink into band
            (16, 320_000_000, 0.022, 22.0),
        ]
        for it, frame, tol, ns in specs:
            ckpt = ws / f"{it:06d}.pth"
            ckpt.write_bytes(b"PK\x03\x04fake")  # not a valid zip archive
            meta = {
                "policy_idx": 1, "iteration": it, "frame": frame,
                "nsucc_mean": ns, "lineage_id": "lin-1",
                "curriculum": {"success_tolerance": tol},
                "checkpoint": str(ckpt),
            }
            (ws / f"{it:06d}.json").write_text(json.dumps(meta))
        # a missing-checkpoint meta
        meta = {
            "policy_idx": 1, "iteration": 18, "frame": 360_000_000,
            "nsucc_mean": 23.0, "lineage_id": "lin-1",
            "curriculum": {"success_tolerance": 0.021},
            "checkpoint": str(ws / "000018.pth"),  # never written
        }
        (ws / "000018.json").write_text(json.dumps(meta))
        return td / "run_x"

    def test_manifest_fields_and_transitions(self):
        with tempfile.TemporaryDirectory() as td:
            run = self._mk_run(Path(td))
            rows, trans = ccrs_manifest.build_manifest(
                run_dir=run, repo_root=Path(td),
                targets={1: [240_000_000]},
                band_lo=0.018, band_hi=0.0285, torch_check="none",
            )
            by_iter = {r.iteration: r for r in rows}
            # target hit exactly
            self.assertIn(12, by_iter)
            self.assertEqual(by_iter[12].frame, 240_000_000)
            self.assertEqual(by_iter[12].frame_deviation, 0)
            self.assertIn("target", by_iter[12].roles)
            # tip = iter 18 (missing ckpt -> flagged, not fabricated)
            self.assertIn(18, by_iter)
            self.assertIn("CHECKPOINT_MISSING", by_iter[18].notes)
            # tolerance recorded in meters
            self.assertAlmostEqual(by_iter[14].live_tolerance, 0.022)
            # transitions detected: 0.070->0.022 and 0.022->0.021 (both in band)
            self.assertEqual(len(trans), 2)
            self.assertAlmostEqual(trans[0].post_tolerance, 0.022)
            self.assertEqual(trans[0].pre_iteration, 12)
            self.assertEqual(trans[0].post_iteration, 14)
            self.assertAlmostEqual(trans[1].post_tolerance, 0.021)
            # peak role = iter 18 (nsucc 23) even though ckpt missing
            self.assertIn("peak", by_iter[18].roles)

    def test_write_outputs(self):
        with tempfile.TemporaryDirectory() as td:
            run = self._mk_run(Path(td))
            rows, trans = ccrs_manifest.build_manifest(
                run_dir=run, repo_root=Path(td), targets={}, torch_check="none")
            out = Path(td) / "out"
            csv_path, md_path = ccrs_manifest.write_outputs(rows, trans, out, run)
            self.assertTrue(csv_path.is_file())
            self.assertTrue(md_path.is_file())
            header = csv_path.read_text().splitlines()[0]
            for col in ("run_id", "git_commit", "worker_id", "frame",
                        "checkpoint_path", "live_tolerance", "live_nsucc",
                        "historical_peak_nsucc", "lineage_id", "policy_arch"):
                self.assertIn(col, header)


class TestReportEmptyData(unittest.TestCase):
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
            self.assertTrue(any("no evaluation results" in m for m in s["missing"]))
            md = (out / "CCRS_REPORT.md").read_text()
            self.assertIn("No evaluation data found", md)
            # pre-registered criteria are present even in dry mode
            self.assertIn("Pre-registered decision criteria", md)
            self.assertTrue((out / "ccrs_summary.json").is_file())

    def test_report_with_synthetic_data(self):
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "out"
            # minimal synthetic results + manifest
            rows = []
            for s_ in (1, 2, 3):
                for (w, it, tol, r) in [
                    (1, 149, 0.022, 0.8), (1, 149, 0.020, 0.6),
                    (4, 141, 0.022, 0.6), (4, 141, 0.020, 0.75),
                ]:
                    rows.append({
                        "status": "ok", "schema_version": 1, "worker_id": w,
                        "checkpoint_iteration": it, "fixed_tolerance": tol,
                        "reset_seed": s_, "success_rate": r + 0.01 * s_,
                        "nsucc_mean": r * 40, "checkpoint_frame": it,
                        "checkpoint_path": "x",
                    })
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
            tr.write_text("worker_id,pre_iteration,post_iteration,pre_frame,"
                          "post_frame,pre_tolerance,post_tolerance,pre_nsucc,post_nsucc\n")
            s = report_generate(res, man, tr, out, n_boot=100)
            self.assertGreater(s["n_cells"], 0)
            self.assertIn("verdicts", s)
            self.assertTrue((out / "CCRS_REPORT.md").is_file())
            md = (out / "CCRS_REPORT.md").read_text()
            self.assertIn("p4 − p1", md)


if __name__ == "__main__":
    unittest.main(verbosity=2)
