"""Tests for the p1 fixed-precision extension (CPU only, no Isaac, no GPU).

Covers:
- checkpoint resolution from the existing CCRS manifest (never guessed);
- 16-unit plan: 4 new tolerances x 4 original seeds, no 0.018 duplicate,
  meters units, original protocol (policy_only / deterministic / 512x512 /
  M=50 / v2 reset+metric protocol);
- preflight failure paths (missing checkpoint, out-dir inside the protected
  ccrs_base_retention dir, wrong unit count);
- decision label thresholds (TARGET_REACHED / NEAR_TARGET /
  PRECISION_LIMIT);
- offline analysis merging the read-only 0.018 base cells with the new
  extension cells, without modifying the base results file;
- launcher run-control (fail-closed): default/plan never launch, unknown
  args rejected with usage, preflight non-zero rc or FAIL output aborts
  before any GPU work, shards never aggregate (single --aggregate-only by
  the launcher), resume skips completed units, base results untouched.
"""

from __future__ import annotations

import csv
import json
import math
import os
import subprocess
import unittest
from pathlib import Path
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dexpbt_isaaclab.ccrs.eval_protocol import (  # noqa: E402
    DEFAULT_SEEDS,
    DEFAULT_TOLERANCES_M,
    EVALUATION_MODE_POLICY_ONLY,
    METRIC_PROTOCOL_VERSION,
    RESET_PROTOCOL_VERSION,
    SCHEMA_VERSION,
    make_sample_id,
)
from dexpbt_isaaclab.ccrs.precision_extension import (  # noqa: E402
    EXPECTED_NUM_UNITS,
    EXTENSION_TOLERANCES_M,
    P1_TIP_FRAME,
    P1_TIP_ITERATION,
    P1_TIP_WORKER_ID,
    build_extension_units,
    preflight,
    resolve_p1_tip_checkpoint,
    write_manifest,
)
from dexpbt_isaaclab.ccrs.precision_analysis import (  # noqa: E402
    ANALYSIS_TOLERANCES_M,
    DECISION_TOLERANCE_M,
    analyze,
    decision_label,
)

MANIFEST_FIELDS = [
    "run_id", "worker_id", "iteration", "frame", "checkpoint_path",
    "roles", "notes", "file_size_bytes",
]


def _write_manifest_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _p1_row(ckpt_path: str, frame: int = P1_TIP_FRAME, notes: str = "") -> dict:
    return {
        "run_id": "run_x", "worker_id": P1_TIP_WORKER_ID,
        "iteration": P1_TIP_ITERATION, "frame": frame,
        "checkpoint_path": ckpt_path, "roles": "tip", "notes": notes,
        "file_size_bytes": -1,
    }


class TestResolveCheckpoint(unittest.TestCase):
    def test_resolves_single_row(self):
        with tempfile.TemporaryDirectory() as td:
            m = Path(td) / "m.csv"
            _write_manifest_csv(m, [
                _p1_row("/tmp/p1.pth"),
                {**_p1_row("/tmp/p5.pth"), "worker_id": 5, "iteration": 117},
            ])
            ck = resolve_p1_tip_checkpoint(m)
            self.assertEqual(ck["checkpoint_path"], "/tmp/p1.pth")
            self.assertEqual(ck["worker_id"], P1_TIP_WORKER_ID)
            self.assertEqual(ck["frame"], P1_TIP_FRAME)

    def test_missing_row_raises(self):
        with tempfile.TemporaryDirectory() as td:
            m = Path(td) / "m.csv"
            _write_manifest_csv(m, [
                {**_p1_row("/tmp/p5.pth"), "worker_id": 5, "iteration": 117},
            ])
            with self.assertRaises(ValueError):
                resolve_p1_tip_checkpoint(m)

    def test_duplicate_rows_raise(self):
        with tempfile.TemporaryDirectory() as td:
            m = Path(td) / "m.csv"
            _write_manifest_csv(m, [_p1_row("/tmp/a.pth"),
                                    _p1_row("/tmp/b.pth")])
            with self.assertRaises(ValueError):
                resolve_p1_tip_checkpoint(m)

    def test_checkpoint_missing_note_excluded(self):
        with tempfile.TemporaryDirectory() as td:
            m = Path(td) / "m.csv"
            _write_manifest_csv(
                m, [_p1_row("/tmp/gone.pth", notes="CHECKPOINT_MISSING")])
            with self.assertRaises(ValueError):
                resolve_p1_tip_checkpoint(m)

    def test_frame_mismatch_raises(self):
        with tempfile.TemporaryDirectory() as td:
            m = Path(td) / "m.csv"
            _write_manifest_csv(m, [_p1_row("/tmp/p1.pth", frame=4_000_000_000)])
            with self.assertRaises(ValueError):
                resolve_p1_tip_checkpoint(m)


def _ckpt_dict(path: str) -> dict:
    return {"worker_id": P1_TIP_WORKER_ID, "iteration": P1_TIP_ITERATION,
            "frame": P1_TIP_FRAME, "checkpoint_path": path,
            "run_id": "run_x", "roles": "tip", "file_size_bytes": -1}


class TestExtensionUnits(unittest.TestCase):
    def setUp(self):
        self.units = build_extension_units(_ckpt_dict("/tmp/p1.pth"))

    def test_exactly_16_units(self):
        self.assertEqual(len(self.units), EXPECTED_NUM_UNITS)
        self.assertEqual(len(self.units), 16)
        self.assertEqual(len({u.unit_key() for u in self.units}), 16)

    def test_no_0018_duplicate_and_meters(self):
        tols = sorted({u.fixed_tolerance for u in self.units})
        self.assertEqual(tols, sorted(EXTENSION_TOLERANCES_M))
        self.assertNotIn(0.018, tols)
        for t in tols:
            self.assertTrue(0.001 <= t <= 0.2, f"{t} not in meter range")
            self.assertNotIn(t, DEFAULT_TOLERANCES_M)

    def test_original_seeds(self):
        self.assertEqual(sorted({u.reset_seed for u in self.units}),
                         sorted(DEFAULT_SEEDS))

    def test_protocol_identical_to_original_ccrs(self):
        for u in self.units:
            self.assertEqual(u.evaluation_mode, EVALUATION_MODE_POLICY_ONLY)
            self.assertTrue(u.deterministic)
            self.assertEqual(u.num_envs, 512)
            self.assertEqual(u.num_episodes, 512)
            self.assertEqual(u.max_consecutive_successes, 50)
            self.assertEqual(u.reset_protocol_version, RESET_PROTOCOL_VERSION)
            self.assertEqual(u.metric_protocol_version, METRIC_PROTOCOL_VERSION)
            self.assertEqual(u.worker_id, P1_TIP_WORKER_ID)
            self.assertEqual(u.checkpoint_iteration, P1_TIP_ITERATION)
            self.assertEqual(u.checkpoint_frame, P1_TIP_FRAME)

    def test_manifest_written_with_16_rows(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "extension_manifest.csv"
            write_manifest(self.units, p)
            rows = list(csv.DictReader(p.open()))
            self.assertEqual(len(rows), 16)
            self.assertEqual({r["fixed_tolerance_m"] for r in rows},
                             {"0.0160", "0.0140", "0.0120", "0.0100"})


class TestPreflight(unittest.TestCase):
    def _run(self, ckpt_path: Path, out_rel="ext", base_rel="base"):
        with tempfile.TemporaryDirectory() as td:
            out_dir = Path(td) / out_rel
            base_dir = Path(td) / base_rel
            base_dir.mkdir(parents=True, exist_ok=True)
            units = build_extension_units(_ckpt_dict(str(ckpt_path)))
            return preflight(units, _ckpt_dict(str(ckpt_path)),
                             out_dir, base_dir)

    def test_passes_with_existing_checkpoint(self):
        with tempfile.TemporaryDirectory() as td:
            ck = Path(td) / "p1.pth"
            ck.write_bytes(b"x" * 128)
            res = self._run(ck)
            failed = [n for n, ok, _ in res.checks if not ok]
            self.assertTrue(res.ok, f"failed: {failed}")

    def test_fails_when_checkpoint_missing(self):
        res = self._run(Path("/nonexistent/p1.pth"))
        self.assertFalse(res.ok)
        by_name = {n: ok for n, ok, _ in res.checks}
        self.assertFalse(by_name["checkpoint_exists"])

    def test_fails_when_out_dir_inside_base_ccrs(self):
        with tempfile.TemporaryDirectory() as td:
            ck = Path(td) / "p1.pth"
            ck.write_bytes(b"x")
            units = build_extension_units(_ckpt_dict(str(ck)))
            base = Path(td) / "ccrs_base_retention"
            base.mkdir()
            res = preflight(units, _ckpt_dict(str(ck)),
                            base / "sub", base)
            by_name = {n: ok for n, ok, _ in res.checks}
            self.assertFalse(by_name["output_dir_separate_from_base_ccrs"])
            self.assertFalse(res.ok)

    def test_fails_on_wrong_unit_count(self):
        with tempfile.TemporaryDirectory() as td:
            ck = Path(td) / "p1.pth"
            ck.write_bytes(b"x")
            units = build_extension_units(_ckpt_dict(str(ck)))[:10]
            res = preflight(units, _ckpt_dict(str(ck)),
                            Path(td) / "ext", Path(td) / "base")
            by_name = {n: ok for n, ok, _ in res.checks}
            self.assertFalse(by_name["unit_count_is_16"])
            self.assertFalse(res.ok)

    def test_fails_on_0018_contamination(self):
        with tempfile.TemporaryDirectory() as td:
            ck = Path(td) / "p1.pth"
            ck.write_bytes(b"x")
            from dexpbt_isaaclab.ccrs.eval_protocol import build_units
            units = build_units(
                run_id="run_x", checkpoints=[_ckpt_dict(str(ck))],
                tolerances_m=[0.018] + EXTENSION_TOLERANCES_M[1:],
                seeds=DEFAULT_SEEDS, num_envs=512, num_episodes=512)
            res = preflight(units, _ckpt_dict(str(ck)),
                            Path(td) / "ext", Path(td) / "base")
            by_name = {n: ok for n, ok, _ in res.checks}
            self.assertFalse(by_name["no_overlap_with_existing_grid_incl_0018"])
            self.assertFalse(res.ok)


class TestDecisionLabel(unittest.TestCase):
    def test_thresholds(self):
        self.assertEqual(decision_label(45.0), "TARGET_REACHED")
        self.assertEqual(decision_label(40.0), "TARGET_REACHED")
        self.assertEqual(decision_label(39.99), "NEAR_TARGET")
        self.assertEqual(decision_label(30.0), "NEAR_TARGET")
        self.assertEqual(decision_label(29.99), "PRECISION_LIMIT")
        self.assertEqual(decision_label(0.0), "PRECISION_LIMIT")
        self.assertEqual(decision_label(float("nan")), "NO_DATA")

    def test_single_recommendation_per_label(self):
        from dexpbt_isaaclab.ccrs.precision_analysis import (
            drop_onset_tolerance, recommendation)
        pts = [
            {"tolerance_m": 0.018, "nsucc_mean": 40.8},
            {"tolerance_m": 0.016, "nsucc_mean": 37.0},
            {"tolerance_m": 0.014, "nsucc_mean": 28.8},  # -22% -> onset
            {"tolerance_m": 0.012, "nsucc_mean": 14.8},
            {"tolerance_m": 0.010, "nsucc_mean": 4.7},
        ]
        self.assertEqual(recommendation("TARGET_REACHED", pts)["action"],
                         "STOP")
        self.assertEqual(recommendation("NEAR_TARGET", pts)["action"],
                         "PRECISION_CONTINUATION")
        rec = recommendation("PRECISION_LIMIT", pts)
        self.assertEqual(rec["action"], "SENTINEL_GRID_RETENTION")
        self.assertAlmostEqual(rec["drop_onset_tolerance_m"], 0.014)
        self.assertAlmostEqual(drop_onset_tolerance(pts), 0.014)
        # flat curve -> no onset, still a single recommendation
        flat = [{"tolerance_m": t, "nsucc_mean": 20.0}
                for t in (0.018, 0.014, 0.010)]
        rec2 = recommendation("PRECISION_LIMIT", flat)
        self.assertEqual(rec2["action"], "SENTINEL_GRID_RETENTION")
        self.assertIsNone(rec2["drop_onset_tolerance_m"])


def _result_row(worker, it, tol, seed, nsuccs, m=50):
    samples = [
        {"sample_id": make_sample_id(seed, i), "reset_seed": seed, "env_id": i,
         "nsucc": float(ns), "normalized_nsucc": min(float(ns), m) / m,
         "terminal_error": 0.005 * (i + 1), "terminal_error_valid": True,
         "dropped": (i % 4 == 0)}
        for i, ns in enumerate(nsuccs)
    ]
    return {
        "status": "ok", "schema_version": SCHEMA_VERSION,
        "worker_id": worker, "checkpoint_iteration": it,
        "fixed_tolerance": tol, "reset_seed": seed,
        "normalized_nsucc": sum(s["normalized_nsucc"] for s in samples) / len(samples),
        "nsucc_mean": sum(s["nsucc"] for s in samples) / len(samples),
        "checkpoint_frame": P1_TIP_FRAME, "checkpoint_path": "x",
        "samples": samples,
        "per_env_reset_fingerprint": {str(i): f"fp_{seed}_{i}"
                                      for i in range(len(nsuccs))},
    }


class TestOfflineAnalysis(unittest.TestCase):
    def _write_jsonl(self, path: Path, rows: list[dict]) -> None:
        with path.open("w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")

    def test_analysis_merges_base_0018_readonly(self):
        with tempfile.TemporaryDirectory() as td:
            base = Path(td) / "base.jsonl"
            ext = Path(td) / "ext.jsonl"
            out = Path(td) / "out"
            # base sweep: p1 tip at 0.018 (must be reused) and an unrelated
            # p5 row (must be ignored)
            base_rows = [
                _result_row(1, 160, 0.018, s, [45.0, 43.0, 44.0, 46.0])
                for s in (10001, 10002)
            ] + [_result_row(5, 117, 0.018, 10001, [50.0] * 4)]
            self._write_jsonl(base, base_rows)
            base_bytes = base.read_bytes()
            # extension: high nsucc down to 1.2cm, 35 at 1cm -> NEAR_TARGET
            ext_rows = []
            for tol, ns in [(0.016, 44.0), (0.014, 42.0),
                            (0.012, 40.0), (0.010, 35.0)]:
                for s in (10001, 10002):
                    ext_rows.append(_result_row(
                        1, 160, tol, s, [ns, ns - 1, ns + 1, ns]))
            self._write_jsonl(ext, ext_rows)

            r = analyze(ext, base, out, n_boot=100)

            self.assertEqual(base.read_bytes(), base_bytes,
                             "base results file must not be modified")
            self.assertEqual([p["tolerance_m"] for p in r["points"]],
                             ANALYSIS_TOLERANCES_M)
            by_tol = {p["tolerance_m"]: p for p in r["points"]}
            self.assertAlmostEqual(by_tol[0.018]["nsucc_mean"], 44.5)
            self.assertIn("reused", by_tol[0.018]["source"])
            self.assertEqual(by_tol[0.016]["source"], "extension")
            self.assertAlmostEqual(by_tol[0.010]["nsucc_mean"], 35.0)
            self.assertEqual(r["decision"]["label"], "NEAR_TARGET")
            self.assertAlmostEqual(r["decision"]["tolerance_m"],
                                   DECISION_TOLERANCE_M)
            # CIs, terminal stats, dropped rate present
            self.assertFalse(math.isnan(by_tol[0.010]["nsucc_mean_ci95"][0]))
            self.assertIn("q50", by_tol[0.010]["terminal_error_quantiles"])
            self.assertAlmostEqual(by_tol[0.010]["dropped_rate"], 0.25)
            # outputs written
            self.assertTrue((out / "precision_extension_analysis.json").is_file())
            self.assertTrue((out / "P1_PRECISION_REPORT.md").is_file())
            self.assertTrue(
                (out / "figures" / "p1_precision_capability_curve.png").is_file())

    def test_target_reached_label(self):
        with tempfile.TemporaryDirectory() as td:
            base = Path(td) / "base.jsonl"
            ext = Path(td) / "ext.jsonl"
            self._write_jsonl(base, [])
            rows = []
            for tol in EXTENSION_TOLERANCES_M:
                for s in (10001, 10002):
                    rows.append(_result_row(1, 160, tol, s, [45.0] * 4))
            self._write_jsonl(ext, rows)
            r = analyze(ext, base, Path(td) / "out", n_boot=50)
            self.assertEqual(r["decision"]["label"], "TARGET_REACHED")

    def test_precision_limit_and_missing_data(self):
        with tempfile.TemporaryDirectory() as td:
            base = Path(td) / "base.jsonl"
            ext = Path(td) / "ext.jsonl"
            self._write_jsonl(base, [])
            rows = [_result_row(1, 160, 0.010, 10001, [10.0] * 4)]
            self._write_jsonl(ext, rows)
            r = analyze(ext, base, Path(td) / "out", n_boot=50)
            self.assertEqual(r["decision"]["label"], "PRECISION_LIMIT")
            # 1.8-1.2cm have no data -> NaN, reported not fabricated
            by_tol = {p["tolerance_m"]: p for p in r["points"]}
            self.assertTrue(math.isnan(by_tol[0.016]["nsucc_mean"]))


REPO_ROOT = Path(__file__).resolve().parents[4]
LAUNCHER = (REPO_ROOT / "results" / "analysis"
            / "ccrs_p1_precision_extension" / "run_extension.sh")
BASE_CCRS_DIR = REPO_ROOT / "results" / "analysis" / "ccrs_base_retention"


def _write_done_raw(raw_dir: Path, units) -> None:
    """Write valid completed raw rows for all units (is_done => True)."""
    from dexpbt_isaaclab.ccrs.eval_protocol import (
        EpisodeSample, make_result_row, write_result_atomic)
    raw_dir.mkdir(parents=True, exist_ok=True)
    for u in units:
        samples = [
            EpisodeSample.create(
                reset_seed=u.reset_seed, env_id=i, nsucc=40.0,
                max_consecutive_successes=u.max_consecutive_successes,
                terminal_error=0.005, terminal_error_valid=True,
                dropped=False)
            for i in range(u.num_episodes)
        ]
        row = make_result_row(
            unit=u, samples=samples,
            per_env_fingerprint={i: f"fp_{u.reset_seed}_{i}"
                                 for i in range(u.num_episodes)},
            git_sha="deadbeef", checkpoint_hash="cafebabe")
        write_result_atomic(raw_dir, u, row)


class TestLauncherRunControl(unittest.TestCase):
    """Behavioral fail-closed tests against the REAL launcher script.

    Every invocation here uses plan mode, a broken manifest, or an unknown
    argument — none of these paths can reach the shard-launch section, so
    no GPU is ever touched by this test class.
    """

    def _run(self, args: list[str], manifest: Path | None = None,
             out: Path | None = None) -> subprocess.CompletedProcess:
        env = dict(os.environ)
        if manifest is not None:
            env["PRECISION_EXT_MANIFEST"] = str(manifest)
        if out is not None:
            env["PRECISION_EXT_OUT"] = str(out)
        return subprocess.run(
            ["bash", str(LAUNCHER), *args],
            env=env, capture_output=True, text=True, timeout=120)

    def test_default_is_plan_and_never_launches(self):
        with tempfile.TemporaryDirectory() as td:
            p = self._run([], out=Path(td))
            self.assertEqual(p.returncode, 0, p.stderr)
            self.assertIn("ALL CHECKS PASSED", p.stdout)
            self.assertIn("plan only", p.stdout)
            self.assertNotIn("shard 0 ->", p.stdout)

    def test_plan_arg_never_launches(self):
        with tempfile.TemporaryDirectory() as td:
            p = self._run(["plan"], out=Path(td))
            self.assertEqual(p.returncode, 0, p.stderr)
            self.assertIn("plan only", p.stdout)
            self.assertNotIn("shard 0 ->", p.stdout)

    def test_unknown_arg_exits_with_usage(self):
        p = self._run(["banana"])
        self.assertEqual(p.returncode, 64)
        self.assertIn("usage:", p.stderr)
        self.assertNotIn("shard 0 ->", p.stdout)

    def test_execute_aborts_on_preflight_nonzero_rc(self):
        # nonexistent manifest -> checkpoint resolution error -> plan
        # subprocess exits non-zero -> execute must abort before any launch
        with tempfile.TemporaryDirectory() as td:
            p = self._run(["execute"],
                          manifest=Path(td) / "missing.csv", out=Path(td))
            self.assertNotEqual(p.returncode, 0)
            self.assertIn("preflight ABORT", p.stderr)
            self.assertNotIn("shard 0 ->", p.stdout)
            self.assertNotIn("ALL CHECKS PASSED", p.stdout)

    def test_execute_aborts_on_preflight_fail_output(self):
        # manifest resolves but points at a missing checkpoint file ->
        # plan rc=0 with FAIL lines -> execute must abort
        with tempfile.TemporaryDirectory() as td:
            m = Path(td) / "m.csv"
            _write_manifest_csv(m, [_p1_row(str(Path(td) / "gone.pth"))])
            p = self._run(["execute"], manifest=m, out=Path(td) / "out")
            self.assertNotEqual(p.returncode, 0)
            self.assertIn("preflight ABORT", p.stderr)
            self.assertIn("FAIL", p.stderr)
            self.assertNotIn("shard 0 ->", p.stdout)

    def test_single_aggregate_only_call_in_launcher(self):
        src = LAUNCHER.read_text()
        self.assertEqual(src.count("--aggregate-only"), 1)
        # aggregation must come after both-shards-succeeded gating
        self.assertLess(src.index("shards failed"), src.index("--aggregate-only"))
        self.assertIn("set -euo pipefail", src)


class TestShardNoAggregateAndResume(unittest.TestCase):
    def _prepared_dirs(self, td: Path):
        """Temp manifest + checkpoint + out dir with ALL 16 units done."""
        ck = td / "p1.pth"
        ck.write_bytes(b"x" * 256)
        m = td / "m.csv"
        _write_manifest_csv(m, [_p1_row(str(ck))])
        out = td / "ext_out"
        base = td / "base_ccrs"
        base.mkdir()
        sentinel = base / "ccrs_results.jsonl"
        sentinel.write_text('{"untouched": true}\n')
        units = build_extension_units(resolve_p1_tip_checkpoint(m))
        _write_done_raw(out / "raw", units)
        return m, out, base, sentinel, units

    def _run_module(self, args: list[str]) -> subprocess.CompletedProcess:
        return subprocess.run(
            [sys.executable, "-m", "dexpbt_isaaclab.ccrs.precision_extension",
             *args],
            cwd=str(Path(__file__).resolve().parents[1]),
            capture_output=True, text=True, timeout=120)

    def test_resume_skips_done_units_and_shard_does_not_aggregate(self):
        with tempfile.TemporaryDirectory() as td:
            m, out, base, sentinel, units = self._prepared_dirs(Path(td))
            for u in units:
                from dexpbt_isaaclab.ccrs.eval_protocol import is_done
                self.assertTrue(is_done(out / "raw", u))
            # both shards execute: all units already done -> nothing runs,
            # exit 0, and NO aggregation output is produced by shards
            for shard in (0, 1):
                p = self._run_module([
                    "--manifest", str(m), "--out-dir", str(out),
                    "--base-ccrs-dir", str(base),
                    "--execute", "--shard-index", str(shard),
                    "--num-shards", "2"])
                self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
                self.assertIn("0 todo", p.stdout)
                self.assertIn("aggregation deferred", p.stdout)
            self.assertFalse((out / "extension_results.jsonl").exists(),
                             "shards must not aggregate")
            self.assertFalse((out / "extension_results.csv").exists())
            self.assertFalse((out / "aggregation_provenance.json").exists())
            # single aggregate-only call produces the merged results once
            p = self._run_module([
                "--manifest", str(m), "--out-dir", str(out),
                "--base-ccrs-dir", str(base), "--aggregate-only"])
            self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
            self.assertIn("aggregated 16 rows", p.stdout)
            self.assertTrue((out / "extension_results.jsonl").is_file())
            # original base results dir untouched throughout
            self.assertEqual(sentinel.read_text(), '{"untouched": true}\n')
            self.assertEqual([f.name for f in base.iterdir()],
                             ["ccrs_results.jsonl"])

    def test_execute_source_has_no_aggregate_in_shard_path(self):
        src = (Path(__file__).resolve().parents[1] / "dexpbt_isaaclab"
               / "ccrs" / "precision_extension.py").read_text()
        # aggregate_raw_dir appears in imports and exactly one call site,
        # inside the --aggregate-only branch
        call_sites = [ln for ln in src.splitlines()
                      if "aggregate_raw_dir(" in ln and "import" not in ln]
        self.assertEqual(len(call_sites), 1, call_sites)
        self.assertLess(src.index("args.aggregate_only"),
                        src.index(call_sites[0].strip()))


if __name__ == "__main__":
    unittest.main(verbosity=2)
