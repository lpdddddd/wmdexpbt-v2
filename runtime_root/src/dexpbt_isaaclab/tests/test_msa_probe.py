"""Unit tests for the Matched Sentinel Advancement (MSA) probe toolkit.

Covers the original 15 required behaviors plus the pilot-integration fixes:
heterogeneous per-worker tolerances, parallel 20M committed-tick exposure,
eval phase barriers, persisted/resumable cycles, crash cleanup, locked
control updates, canonical tolerance precision, continuation guard and
durable promotion commits. Pure CPU; no Isaac, no GPU.
"""

from __future__ import annotations

import json
import os
import tempfile
import threading
import unittest
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dexpbt_isaaclab.msa import decision as dec
from dexpbt_isaaclab.msa import dry_run as dr
from dexpbt_isaaclab.msa.controller import CycleFailure, PROBE_PACKET_FRAMES
from dexpbt_isaaclab.msa.eval_bridge import (
    EvalRequest,
    PILOT_NUM_SEEDS,
    SnapshotSpec,
    build_seed_manifest,
    canonical_tau,
    load_seed_results,
)
from dexpbt_isaaclab.msa.provenance import (
    EXPOSURE_WINDOW_FRAMES,
    ExposureSegment,
    MSAProvenance,
    probe_due,
    tau_plus_of,
)
from dexpbt_isaaclab.msa.state_machine import (
    IllegalTransition,
    MSAPhase,
    MSAProbeStateMachine,
)

MSA_DIR = Path(__file__).resolve().parents[1] / "dexpbt_isaaclab" / "msa"
SRC_DIR = Path(__file__).resolve().parents[1]
LAUNCHER_DIR = Path("/root/autodl-tmp/wmdexpbt/results/launchers/msa_probe")
TICK = dr.TICK_FRAMES


def _prov_with_exposure(worker=1, tau_plus=0.0223, start=4_950_000_000,
                        end=4_970_000_000):
    p = MSAProvenance(worker_id=worker)
    p.record_exposure(ExposureSegment("common_probe", tau_plus, start, end))
    return p


class TestDueLogic(unittest.TestCase):
    # J1: tau=0.010 -> never probe (no tighter sentinel exists)
    def test_tau_at_floor_never_due(self):
        self.assertFalse(probe_due(None, 5_000_000_000, 0.010))
        self.assertFalse(probe_due(MSAProvenance(worker_id=0),
                                   5_000_000_000, 0.010))

    def test_tau_plus_floor_clamp(self):
        self.assertAlmostEqual(tau_plus_of(0.0235), 0.0235 * 0.95)
        self.assertAlmostEqual(tau_plus_of(0.0102), 0.010)  # clamped
        self.assertAlmostEqual(tau_plus_of(0.010), 0.010)

    # J2: resume without provenance -> immediately due
    def test_missing_provenance_immediately_due(self):
        self.assertTrue(probe_due(None, 5_000_000_000, 0.0235))

    def test_from_dict_none_stays_none(self):
        self.assertIsNone(MSAProvenance.from_dict(None))

    # J3: at most one probe per 100M worker-local frames
    def test_probe_at_most_once_per_window(self):
        tau = 0.0235
        tp = tau_plus_of(tau)
        prov = _prov_with_exposure(tau_plus=tp, start=4_950_000_000,
                                   end=4_970_000_000)
        # exposure inside the window -> not due
        self.assertFalse(probe_due(prov, 5_000_000_000, tau))
        # window has fully passed the exposure -> due again
        self.assertTrue(probe_due(prov, 4_970_000_000 + EXPOSURE_WINDOW_FRAMES + 1,
                                  tau))

    def test_floor_worker_never_due(self):
        self.assertFalse(probe_due(None, 5_000_000_000, 0.0235,
                                   is_floor_worker=True))

    def test_tighter_exposure_counts(self):
        tau = 0.0235
        prov = _prov_with_exposure(tau_plus=0.015)  # tighter than tau_plus
        self.assertFalse(probe_due(prov, 5_000_000_000, tau))

    def test_intervention_record_replaced_not_duplicated(self):
        from dexpbt_isaaclab.msa.provenance import InterventionRecord
        p = MSAProvenance(worker_id=0)
        rec = InterventionRecord(intervention_id="i1", worker_id=0,
                                 lineage_id="l", source_checkpoint_id="c",
                                 active_tau=0.023, proposed_tau=0.0219,
                                 due_frame=5_000_000_000,
                                 evaluator_manifest_id="m")
        rec.final_decision = "abort"
        p.record_intervention(rec)
        rec2 = InterventionRecord(intervention_id="i1", worker_id=0,
                                  lineage_id="l", source_checkpoint_id="c",
                                  active_tau=0.023, proposed_tau=0.0219,
                                  due_frame=5_000_000_000,
                                  evaluator_manifest_id="m")
        rec2.final_decision = "stay"
        p.record_intervention(rec2)
        self.assertEqual(len(p.interventions), 1)
        self.assertEqual(p.interventions[0]["final_decision"], "stay")


class TestStateMachine(unittest.TestCase):
    def test_full_legal_cycle(self):
        m = MSAProbeStateMachine()
        for nxt in [MSAPhase.PRE_EVAL, MSAPhase.COMMON_PROBE, MSAPhase.MID_EVAL,
                    MSAPhase.ROUTE_SECOND_HALF, MSAPhase.SECOND_PROBE,
                    MSAPhase.POST_EVAL, MSAPhase.DECIDE, MSAPhase.PROMOTE,
                    MSAPhase.NORMAL]:
            m.transition(nxt)
        self.assertEqual(m.phase, MSAPhase.NORMAL)
        self.assertEqual(m.cycle_index, 1)

    def test_illegal_jump_raises(self):
        m = MSAProbeStateMachine()
        with self.assertRaises(IllegalTransition):
            m.transition(MSAPhase.DECIDE)
        m.transition(MSAPhase.PRE_EVAL)
        with self.assertRaises(IllegalTransition):
            m.transition(MSAPhase.SECOND_PROBE)

    def test_abort_from_any_phase(self):
        m = MSAProbeStateMachine()
        m.transition(MSAPhase.PRE_EVAL)
        m.transition(MSAPhase.COMMON_PROBE)
        m.abort("test crash")
        self.assertEqual(m.phase, MSAPhase.NORMAL)
        self.assertIn("ABORT", m.history[-1])

    def test_no_implicit_freeze_flags(self):
        # the machine is a single enum, not a pile of booleans
        src = (MSA_DIR / "state_machine.py").read_text()
        self.assertNotIn("freeze_flag", src)
        m = MSAProbeStateMachine()
        self.assertTrue(m.training_allowed())
        m.transition(MSAPhase.PRE_EVAL)
        self.assertFalse(m.training_allowed())

    def test_serialization_roundtrip(self):
        m = MSAProbeStateMachine()
        m.transition(MSAPhase.PRE_EVAL, "test")
        m2 = MSAProbeStateMachine.from_dict(m.to_dict())
        self.assertEqual(m2.phase, MSAPhase.PRE_EVAL)
        self.assertEqual(m2.history, m.history)


class TestEvaluatorIsolation(unittest.TestCase):
    # J4: evaluator does not modify live state
    def test_snapshot_frozen_read_only(self):
        import os
        import stat as st
        with tempfile.TemporaryDirectory() as td:
            src = Path(td) / "ckpt.pth"
            src.write_bytes(b"weights")
            spec = SnapshotSpec.freeze(src, "w0_pre", 0, 5_000_000_000,
                                       Path(td) / "snaps")
            spec.assert_immutable()
            mode = os.stat(spec.path).st_mode
            self.assertEqual(mode & (st.S_IWUSR | st.S_IWGRP | st.S_IWOTH), 0)
            self.assertEqual(Path(spec.path).read_bytes(), b"weights")
            if os.geteuid() != 0:   # root bypasses permission bits
                with self.assertRaises(PermissionError):
                    Path(spec.path).write_bytes(b"tampered")

    def test_writable_snapshot_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "snap.pth"
            p.write_bytes(b"w")
            spec = SnapshotSpec("s", 0, 0, str(p))
            with self.assertRaises(PermissionError):
                spec.assert_immutable()

    @staticmethod
    def _code_identifiers(path: Path) -> set[str]:
        """All identifiers in actual code (docstrings/comments excluded)."""
        import ast
        names: set[str] = set()
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Name):
                names.add(node.id)
            elif isinstance(node, ast.Attribute):
                names.add(node.attr)
            elif isinstance(node, ast.FunctionDef):
                names.add(node.name)
        return names

    def test_sidecar_never_writes_live_training_state(self):
        for mod in ("sidecar.py", "controller.py", "eval_bridge.py"):
            names = self._code_identifiers(MSA_DIR / mod)
            for banned in ("load_state_dict", "optimizer", "replay_buffer",
                           "reload_checkpoint", "reset_optimizer",
                           "mutate", "restart_policy"):
                self.assertNotIn(banned, names, f"{banned} used in {mod}")

    def test_only_msa_fields_are_written(self):
        # the sidecar's control write path touches ONLY the msa_* fields
        # (the observer-owned forced_success_tolerance is never written)
        src = (MSA_DIR / "sidecar.py").read_text()
        import re
        writes = re.findall(r"control\.(\w+)\s*=", src)
        self.assertEqual(
            sorted(set(writes)),
            ["msa_cycle", "msa_enabled", "msa_forced_success_tolerance",
             "msa_provenance", "msa_state_machine"])

    def test_eval_device_required(self):
        with tempfile.TemporaryDirectory() as td:
            src = Path(td) / "c.pth"
            src.write_bytes(b"w")
            spec = SnapshotSpec.freeze(src, "w0_pre", 0, 5_000_000_000,
                                       Path(td) / "snaps")
            mid, seeds = build_seed_manifest(2)
            req = EvalRequest(snapshot=spec, tau=0.0219, manifest_id=mid,
                              seeds=seeds, task="T", run_id="r",
                              raw_dir=td)  # no eval_device
            with self.assertRaises(ValueError):
                req.build_commands()
            req2 = EvalRequest(snapshot=spec, tau=0.0219, manifest_id=mid,
                               seeds=seeds, task="T", run_id="r",
                               raw_dir=td, eval_device="cuda:7")
            env = req2.build_env(base_env={})
            self.assertEqual(env["CUDA_VISIBLE_DEVICES"], "7")
            cmds = req2.build_commands()
            self.assertEqual(len(cmds), 2)


class TestPairedEvalContracts(unittest.TestCase):
    # J5: pre/mid tolerance or seed manifest mismatch fails closed
    def test_manifest_mismatch_fails_closed(self):
        mid_a, seeds = build_seed_manifest(8)
        mid_b, _ = build_seed_manifest(8, base=999_000)
        a = dec.SeedEvalResult(mid_a, 0.0223, seeds, [0.5] * 8)
        b = dec.SeedEvalResult(mid_b, 0.0223, list(range(999_001, 999_009)),
                               [0.6] * 8)
        with self.assertRaises(dec.ManifestMismatch):
            dec.l_mid(a, b)

    def test_tolerance_mismatch_fails_closed(self):
        mid, seeds = build_seed_manifest(8)
        a = dec.SeedEvalResult(mid, 0.0223, seeds, [0.5] * 8)
        b = dec.SeedEvalResult(mid, 0.0235, seeds, [0.6] * 8)
        with self.assertRaises(dec.ManifestMismatch):
            dec.l_mid(a, b)

    def test_seed_order_mismatch_fails_closed(self):
        mid, seeds = build_seed_manifest(8)
        a = dec.SeedEvalResult(mid, 0.0223, seeds, [0.5] * 8)
        b = dec.SeedEvalResult(mid, 0.0223, list(reversed(seeds)), [0.6] * 8)
        with self.assertRaises(dec.ManifestMismatch):
            dec.l_mid(a, b)

    # J11: rho only compares the same tau (cross-tolerance is impossible)
    def test_rho_same_tau_only(self):
        mid, seeds = build_seed_manifest(8)
        pre = dec.SeedEvalResult(mid, 0.0235, seeds, [0.5] * 8)
        post_wrong = dec.SeedEvalResult(mid, 0.0223, seeds, [0.4] * 8)
        with self.assertRaises(dec.ManifestMismatch):
            dec.same_tau_retention(pre, post_wrong)

    # J10: p_pre < 0.01 -> retention unavailable
    def test_low_p_pre_retention_unavailable(self):
        mid, seeds = build_seed_manifest(8)
        pre = dec.SeedEvalResult(mid, 0.0235, seeds, [0.005] * 8)
        post = dec.SeedEvalResult(mid, 0.0235, seeds, [0.5] * 8)
        r = dec.same_tau_retention(pre, post)
        self.assertTrue(r.retention_unavailable)
        self.assertIsNone(r.rho)
        # and it blocks promotion
        panel = {"lower_quantile": 0.05}
        d = dec.decide_promotion(panel, {1: r}, 0.88)
        self.assertFalse(d.promote)
        self.assertEqual(d.first_blocking_gate, "retention_unavailable_w1")


class TestCanonicalTolerance(unittest.TestCase):
    """Regression tests for the .6f-write / 1e-9-read precision bug."""

    FULL_PRECISION_TAU = tau_plus_of(0.023112487509668322)  # 0.02195686...

    def test_canonical_roundtrip(self):
        tau = self.FULL_PRECISION_TAU
        s = canonical_tau(tau)
        # the canonical string parses back to a float whose canonical form
        # is the SAME string (stable fixed point)
        self.assertEqual(canonical_tau(float(s)), s)
        # and it retains more than 6 decimals of information
        self.assertNotEqual(float(s), round(tau, 6))

    def test_non_six_decimal_tau_matches_own_request(self):
        tau = self.FULL_PRECISION_TAU
        with tempfile.TemporaryDirectory() as td:
            src = Path(td) / "c.pth"
            src.write_bytes(b"w")
            spec = SnapshotSpec.freeze(src, "w0_pre", 0, 5_000_000_000,
                                       Path(td) / "snaps")
            mid, seeds = build_seed_manifest(3)
            raw = Path(td) / "raw"
            raw.mkdir()
            # simulate ccrs_eval output: argparse parses the canonical CLI
            # string to float and stores that float in the raw row
            stored = float(canonical_tau(tau))
            for i, seed in enumerate(seeds):
                (raw / f"u{i}.json").write_text(json.dumps({
                    "status": "ok", "checkpoint_path": spec.path,
                    "fixed_tolerance": stored, "reset_seed": seed,
                    "normalized_nsucc": 0.5}))
            res = load_seed_results(raw, spec, tau, mid, seeds)
            self.assertEqual(len(res.success), 3)
            self.assertEqual(canonical_tau(res.tau), canonical_tau(tau))

    def test_six_decimal_rounding_no_longer_matches_silently(self):
        # the OLD bug: request wrote f"{tau:.6f}", results stored the
        # rounded value, reader wanted 1e-9. Now a 6-decimal-rounded raw
        # row must NOT match a full-precision request.
        tau = self.FULL_PRECISION_TAU
        with tempfile.TemporaryDirectory() as td:
            src = Path(td) / "c.pth"
            src.write_bytes(b"w")
            spec = SnapshotSpec.freeze(src, "w0_pre", 0, 5_000_000_000,
                                       Path(td) / "snaps")
            mid, seeds = build_seed_manifest(2)
            raw = Path(td) / "raw"
            raw.mkdir()
            for i, seed in enumerate(seeds):
                (raw / f"u{i}.json").write_text(json.dumps({
                    "status": "ok", "checkpoint_path": spec.path,
                    "fixed_tolerance": round(tau, 6),   # legacy rounding
                    "reset_seed": seed, "normalized_nsucc": 0.5}))
            with self.assertRaises(FileNotFoundError):
                load_seed_results(raw, spec, tau, mid, seeds)

    def test_ccrs_unit_key_distinguishes_close_taus(self):
        from dexpbt_isaaclab.ccrs.eval_protocol import EvalUnit
        kw = dict(run_id="r", worker_id=0, checkpoint_iteration=1,
                  checkpoint_frame=5_000_000_000, checkpoint_path="/x.pth",
                  reset_seed=1, num_envs=8, num_episodes=8,
                  evaluation_mode="policy_only")
        a = EvalUnit(fixed_tolerance=0.021956863134184906, **kw)
        b = EvalUnit(fixed_tolerance=0.021956863, **kw)  # same at 6 decimals
        self.assertNotEqual(a.unit_key(), b.unit_key())
        from dexpbt_isaaclab.ccrs.eval_protocol import result_path
        self.assertNotEqual(result_path(Path("/tmp"), a),
                            result_path(Path("/tmp"), b))


class TestRouting(unittest.TestCase):
    # J7: top-3 deterministic with stable tie-break
    def test_top3_stable_tie_break(self):
        l_mids = {0: 0.06, 1: 0.10, 2: 0.02, 3: 0.08, 4: 0.06, 5: 0.06}
        r = dec.route_second_half(l_mids, top_k=3)
        self.assertEqual(r.order, [1, 3, 0, 4, 5, 2])
        self.assertEqual(r.recipients, [1, 3, 0])
        # deterministic across calls
        self.assertEqual(dec.route_second_half(l_mids, top_k=3).recipients,
                         [1, 3, 0])

    # J8: permuted route keeps the total dose identical
    def test_permuted_route_same_dose(self):
        l_mids = {w: 0.01 * w for w in range(6)}
        normal = dec.route_second_half(l_mids, top_k=3, permuted=False)
        perm = dec.route_second_half(l_mids, top_k=3, permuted=True)
        self.assertEqual(len(normal.recipients), len(perm.recipients))
        self.assertEqual(perm.arm, "MSA_PERMUTED_ROUTE")
        self.assertEqual(perm.permutation_seed, dec.PERMUTED_ROUTE_SEED)
        # same multiset of L_mid values, remapped identities
        self.assertEqual(sorted(perm.permuted_assignment.values()),
                         sorted(l_mids.values()))
        # deterministic for the pre-registered seed
        again = dec.route_second_half(l_mids, top_k=3, permuted=True)
        self.assertEqual(perm.recipients, again.recipients)


class TestBootstrap(unittest.TestCase):
    # J12: bootstrap deterministic
    def test_bootstrap_deterministic(self):
        diffs = {w: [0.01 * ((i % 7) - 3) + 0.02 for i in range(32)]
                 for w in range(6)}
        a = dec.panel_bootstrap_lower_quantile(diffs, n_boot=500)
        b = dec.panel_bootstrap_lower_quantile(diffs, n_boot=500)
        self.assertEqual(a["lower_quantile"], b["lower_quantile"])
        self.assertEqual(a["prng_seed"], dec.BOOTSTRAP_SEED)

    def test_bootstrap_is_labeled_criterion_local_conditional(self):
        diffs = {w: [0.02, 0.03, 0.01, 0.02] for w in range(6)}
        s = dec.panel_bootstrap_lower_quantile(diffs, n_boot=50)
        self.assertIn("NOT a population CI", s["note"])
        self.assertIn("criterion-local", s["note"])

    def test_negative_panel_blocks_promotion(self):
        diffs = {w: [-0.05, -0.04, -0.06, -0.05] for w in range(6)}
        s = dec.panel_bootstrap_lower_quantile(diffs, n_boot=200)
        mid, seeds = build_seed_manifest(4)
        pre = dec.SeedEvalResult(mid, 0.0235, seeds, [0.5] * 4)
        post = dec.SeedEvalResult(mid, 0.0235, seeds, [0.5] * 4)
        ret = {w: dec.same_tau_retention(pre, post) for w in range(6)}
        d = dec.decide_promotion(s, ret, 0.88)
        self.assertFalse(d.promote)
        self.assertEqual(d.first_blocking_gate, "panel_lower_quantile")

    def test_cross_tolerance_gate_names_rejected(self):
        s = {"lower_quantile": 0.05}
        mid, seeds = build_seed_manifest(4)
        pre = dec.SeedEvalResult(mid, 0.0235, seeds, [0.5] * 4)
        post = dec.SeedEvalResult(mid, 0.0235, seeds, [0.5] * 4)
        ret = {0: dec.same_tau_retention(pre, post)}
        with self.assertRaises(ValueError):
            dec.decide_promotion(s, ret, 0.88, extra_safety_gates={
                "historical_peak_retention": True})


class TestHeterogeneousDryRunCycle(unittest.TestCase):
    """Full probe cycle: 6 workers with DIFFERENT live tolerances, parallel
    tick-quantized trainer, phase-barrier holds, persisted state."""

    @classmethod
    def setUpClass(cls):
        cls.world, cls.ctl, cls.result = dr.run_dry_run()

    # heterogeneous tau: every worker probed at its OWN tau / tau_plus
    def test_worker_local_tolerances(self):
        for w, tau in dr.DRY_RUN_TAUS.items():
            self.assertEqual(canonical_tau(self.result.tau_by_worker[w]),
                             canonical_tau(tau))
            self.assertEqual(canonical_tau(self.result.tau_plus_by_worker[w]),
                             canonical_tau(tau_plus_of(tau)))
        # six distinct taus in the panel
        self.assertEqual(len(set(self.result.tau_by_worker.values())), 6)
        # FakeWorld.evaluate raises on any foreign-tolerance eval, so a
        # completed cycle proves all evals were criterion-local.

    # common exposure is PARALLEL: exactly one 20M packet per worker (a
    # serial per-worker loop would advance everyone once per call -> 6x)
    def test_common_exposure_parallel_exactly_one_packet(self):
        recs = self.result.records
        for w in range(6):
            segs = [s for s in recs[w]["segments"]
                    if s["segment_type"] == "common_probe"]
            self.assertEqual(len(segs), 1)
            self.assertEqual(
                segs[0]["segment_end_frame"] - segs[0]["segment_start_frame"],
                PROBE_PACKET_FRAMES)
        # REAL stage-3 committed cadence (never assume the base 20M
        # pbt_interval is the committed quantum)
        self.assertEqual(PROBE_PACKET_FRAMES, 80_000_000)

    # top-3 second half is parallel too; non-recipients trained at their tau
    def test_second_half_parallel_and_non_recipients_reverted(self):
        recipients = set(self.result.routing.recipients)
        for w in range(6):
            plus = self.world.trained_frames_at_tol.get((w, "tau_plus"), 0)
            base = self.world.trained_frames_at_tol.get((w, "tau"), 0)
            if w in recipients:
                self.assertEqual(plus, 2 * TICK)   # common + second
                self.assertEqual(base, 0)
            else:
                self.assertEqual(plus, TICK)       # common only
                # ran in parallel at their OWN tau during the second packet
                self.assertEqual(base, TICK)

    def test_non_recipient_repinned_to_own_tau_after_mid(self):
        recipients = set(self.result.routing.recipients)
        for w in range(6):
            pins = [t for wid, t in self.world.pin_log if wid == w]
            tau = float(canonical_tau(dr.DRY_RUN_TAUS[w]))
            tau_plus = float(canonical_tau(tau_plus_of(dr.DRY_RUN_TAUS[w])))
            self.assertEqual(pins[0], tau_plus)    # common probe pin
            if w not in recipients:
                # revert pin right after routing (before the second packet)
                self.assertIn(tau, pins[1:])

    # frames are OBSERVED committed ticks, ledger includes probe frames
    def test_observed_frames_and_ledger(self):
        recipients = set(self.result.routing.recipients)
        for w in range(6):
            expect = 2 * TICK   # everyone advanced during both packets
            self.assertEqual(self.world.trained_frames[w], expect)
            self.assertEqual(self.world.frame[w], 5_000_000_000 + expect)
        # provenance tau_plus segments account exactly for the probe dose
        for w in range(6):
            prov = self.ctl.load_provenance(w)
            total = sum(s["segment_end_frame"] - s["segment_start_frame"]
                        for s in prov.exposures)
            want = 2 * TICK if w in recipients else TICK
            self.assertEqual(total, want)

    # eval phase barrier: FakeWorld asserts snapshot/eval happen only while
    # held and that held workers never advance; reaching DONE proves it.
    def test_frames_do_not_advance_during_eval(self):
        self.assertEqual(self.world.state_store[0]["msa_cycle"]["status"],
                         "DONE")
        # every snapshot/eval passed the held-assertion in the fake world
        self.assertGreater(len(self.world.eval_calls), 0)

    # J9: q_post never enters the common-dose L_mid
    def test_q_post_not_in_common_dose(self):
        recs = self.result.records
        recipients = set(self.result.routing.recipients)
        for w in range(6):
            if w in recipients:
                self.assertIn("q_post_mean", recs[w])
            else:
                self.assertNotIn("q_post_mean", recs[w])
        # L_mid == exactly one packet of learning (never two)
        for w, lm in self.result.l_mid_by_worker.items():
            self.assertAlmostEqual(lm, self.world.learn_rate[w], delta=1e-9)

    # J13: no optimizer/PBT lineage reset during/after the probe
    def test_no_resets(self):
        self.assertEqual(self.world.reset_calls, [])

    def test_promotion_commits_worker_local_tau_plus_and_clears_pins(self):
        self.assertTrue(self.result.promoted)
        for w, tau in dr.DRY_RUN_TAUS.items():
            tau_plus = float(canonical_tau(tau_plus_of(tau)))
            self.assertEqual(self.world.live_tolerance[w], tau_plus)
            # resume snapshot synced -> survives a restart
            self.assertEqual(self.world.resume_snapshot_tolerance[w],
                             tau_plus)
            # forced override cleared
            self.assertEqual(self.world.pinned_tolerance[w], 0.0)

    def test_no_residual_holds(self):
        self.assertEqual(self.world.held, set())

    def test_state_persisted_roundtrip(self):
        for w in range(6):
            slot = self.world.state_store[w]
            prov = MSAProvenance.from_dict(slot["msa_provenance"])
            self.assertEqual(prov.worker_id, w)
            self.assertEqual(len(prov.interventions), 1)
            m = MSAProbeStateMachine.from_dict(slot["msa_state_machine"])
            self.assertEqual(m.phase, MSAPhase.NORMAL)
            self.assertEqual(slot["msa_cycle"]["status"], "DONE")
            self.assertEqual(slot["msa_cycle"]["packet_frames"],
                             PROBE_PACKET_FRAMES)

    def test_provenance_records_complete(self):
        rec = self.result.records[1]
        for field in ("intervention_id", "worker_id", "lineage_id",
                      "source_checkpoint_id", "active_tau", "proposed_tau",
                      "due_frame", "evaluator_manifest_id", "snapshot_pre_id",
                      "snapshot_mid_id", "snapshot_post_id",
                      "recipient_status", "first_blocking_gate",
                      "final_decision"):
            self.assertIn(field, rec)
            self.assertNotEqual(rec[field], "", f"{field} empty")

    def test_permuted_arm_same_dose_different_mapping(self):
        world2, _, result2 = dr.run_dry_run(permuted=True)
        self.assertEqual(sum(world2.trained_frames.values()),
                         sum(self.world.trained_frames.values()))
        self.assertEqual(len(result2.routing.recipients),
                         len(self.result.routing.recipients))

    def test_stay_decision_restores_worker_local_tau(self):
        # negative learning -> panel gate blocks -> everyone back at OWN tau
        world, ctl, views = dr.make_world(
            learn_rates={w: -0.05 for w in range(6)})
        result = ctl.run_cycle(views)
        self.assertFalse(result.promoted)
        self.assertEqual(result.first_blocking_gate, "panel_lower_quantile")
        for w, tau in dr.DRY_RUN_TAUS.items():
            self.assertEqual(world.live_tolerance[w],
                             float(canonical_tau(tau)))
            self.assertEqual(world.pinned_tolerance[w], 0.0)
            self.assertEqual(world.resume_snapshot_tolerance[w],
                             float(canonical_tau(tau)))


class TestCrashRecovery(unittest.TestCase):
    """Any-phase crash: pins cleared, own tau restored, FAILED recorded."""

    def _crashed_world(self, crash_on: str) -> dr.FakeWorld:
        world, ctl, views = dr.make_world(crash_on=crash_on)
        with self.assertRaises(CycleFailure):
            ctl.run_cycle(views)
        return world

    def _assert_clean(self, world: dr.FakeWorld, failed_phase: str):
        for w, tau in world.taus.items():
            self.assertEqual(world.live_tolerance[w],
                             float(canonical_tau(tau)),
                             f"worker {w} left at wrong tolerance")
            self.assertEqual(world.pinned_tolerance[w], 0.0,
                             f"worker {w} pin not cleared")
        self.assertEqual(world.held, set(), "holds not released")
        cycle = world.state_store[0]["msa_cycle"]
        self.assertEqual(cycle["status"], "FAILED")
        self.assertEqual(cycle["failed_phase"], failed_phase)
        self.assertIn("InjectedCrash", cycle["error"])
        m = MSAProbeStateMachine.from_dict(
            world.state_store[0]["msa_state_machine"])
        self.assertEqual(m.phase, MSAPhase.NORMAL)
        self.assertIn("ABORT", m.history[-1])
        prov = MSAProvenance.from_dict(
            world.state_store[0]["msa_provenance"])
        self.assertEqual(prov.interventions[-1]["final_decision"], "abort")

    def test_crash_in_pre_eval(self):
        self._assert_clean(self._crashed_world("pre"), "PRE_EVAL")

    def test_crash_in_common_probe(self):
        self._assert_clean(self._crashed_world("common_train"),
                           "COMMON_PROBE")

    def test_crash_in_mid_eval(self):
        self._assert_clean(self._crashed_world("mid"), "MID_EVAL")

    def test_crash_in_post_eval(self):
        self._assert_clean(self._crashed_world("post"), "POST_EVAL")

    def test_failed_cycle_not_auto_resumed(self):
        world = self._crashed_world("mid")
        _, ctl2, _ = dr.make_world(world=world)
        self.assertIsNone(ctl2.pending_cycle(list(world.taus.keys())))


class TestRestartResume(unittest.TestCase):
    """Hard-kill (no cleanup) -> restart -> resume from persisted state;
    completed segments are NEVER re-trained."""

    def test_resume_after_hard_kill_in_mid_eval(self):
        world, ctl, views = dr.make_world(crash_on="mid")
        world.crash_hard = True
        with self.assertRaises(dr.HardKill):
            ctl.run_cycle(views)
        # common packet already trained + persisted
        frames_after_kill = dict(world.trained_frames)
        for w in range(6):
            self.assertEqual(frames_after_kill[w], TICK)
        # restart: fresh controller over the SAME persistent state store
        world.crash_hard = False
        world2, ctl2, views2 = dr.make_world(world=world, crash_on=None)
        pending = ctl2.pending_cycle(list(world.taus.keys()))
        self.assertIsNotNone(pending)
        self.assertEqual(pending["status"], "RUNNING")
        result = ctl2.run_cycle(views2, resume_cycle=pending)
        self.assertTrue(result.resumed)
        self.assertEqual(result.intervention_id,
                         pending["intervention_id"])
        # the common packet was NOT re-trained: only the second packet's
        # tick was added on top
        for w in range(6):
            self.assertEqual(world.trained_frames[w], 2 * TICK)
        # exactly one common_probe segment per worker in provenance
        for w in range(6):
            prov = ctl2.load_provenance(w)
            commons = [s for s in prov.exposures
                       if s["segment_type"] == "common_probe"
                       and s["intervention_id"] == result.intervention_id]
            self.assertEqual(len(commons), 1)
        # cycle completes and state returns to NORMAL
        self.assertEqual(world.state_store[0]["msa_cycle"]["status"], "DONE")
        m = MSAProbeStateMachine.from_dict(
            world.state_store[0]["msa_state_machine"])
        self.assertEqual(m.phase, MSAPhase.NORMAL)
        # snapshots were reused, not re-frozen (same ids, no duplicates)
        pre_snaps = [k for k in world.snapshots if k.endswith("_pre")]
        self.assertEqual(len(pre_snaps), 6)


class TestConcurrentControlUpdates(unittest.TestCase):
    """update_curriculum_control + write_curriculum_control_preserving_msa:
    no lost updates between the MSA sidecar and the observer."""

    def test_no_lost_updates(self):
        from dexpbt_isaaclab.pbt.pbt_curriculum_p1 import (
            CurriculumControlState,
            read_curriculum_control,
            update_curriculum_control,
            write_curriculum_control_preserving_msa,
        )
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "000.json"

            n_each = 25
            errors: list[Exception] = []

            def msa_writer():
                try:
                    for i in range(n_each):
                        def _mut(state, i=i):
                            state.msa_enabled = True
                            cyc = dict(state.msa_cycle or {})
                            cyc["counter"] = int(cyc.get("counter", 0)) + 1
                            state.msa_cycle = cyc
                        update_curriculum_control(path, _mut)
                except Exception as e:   # pragma: no cover
                    errors.append(e)

            def observer_writer():
                try:
                    for i in range(n_each):
                        # observer rebuilds the FULL state from scratch
                        fresh = CurriculumControlState(
                            recent_peak_nsucc_mean=float(i))
                        write_curriculum_control_preserving_msa(path, fresh)
                except Exception as e:   # pragma: no cover
                    errors.append(e)

            threads = [threading.Thread(target=msa_writer),
                       threading.Thread(target=observer_writer)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()
            self.assertEqual(errors, [])
            final = read_curriculum_control(path)
            # every MSA increment survived the observer's full-state writes
            self.assertTrue(final.msa_enabled)
            self.assertEqual(final.msa_cycle["counter"], n_each)

    def test_mark_shrink_consumed_preserves_msa_fields(self):
        from dexpbt_isaaclab.pbt.pbt_curriculum_p1 import (
            mark_shrink_consumed,
            read_curriculum_control,
            update_curriculum_control,
        )
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "000.json"

            def _mut(state):
                state.msa_enabled = True
                state.msa_provenance = {"worker_id": 0, "exposures": []}
            update_curriculum_control(path, _mut)
            mark_shrink_consumed(path, frames_since_restart=1000,
                                 consolidation_env_frames=50_000_000,
                                 num_envs=8192)
            final = read_curriculum_control(path)
            self.assertTrue(final.msa_enabled)
            self.assertEqual(final.msa_provenance["worker_id"], 0)
            self.assertFalse(final.shrink_token)


class TestEnvPinBehavior(unittest.TestCase):
    """msa_forced_success_tolerance: env-side pin semantics + inertness."""

    def _apply(self, control, success_tolerance=0.023):
        import torch

        from dexpbt_isaaclab.pbt import pbt_curriculum_p1 as mod
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "000.json"
            mod.write_curriculum_control(path, control)
            return mod.apply_env_tolerance_curriculum(
                last_curriculum_update=0,
                frames_since_restart=10_000_000,
                curriculum_interval=1_000,
                prev_episode_successes=torch.ones(8),
                success_tolerance=success_tolerance,
                initial_tolerance=0.075,
                target_tolerance=0.01,
                curriculum_mode="mastery",
                curriculum_control_path=path,
                curriculum_shrink_factor=0.95,
                post_shrink_consolidation_env_frames=0,
                num_envs=8,
            )

    def test_msa_disabled_pin_is_inert(self):
        from dexpbt_isaaclab.pbt.pbt_curriculum_p1 import CurriculumControlState
        control = CurriculumControlState(
            msa_enabled=False, msa_forced_success_tolerance=0.0219,
            allow_shrink=False)
        tol, _ = self._apply(control)
        self.assertEqual(tol, 0.023)   # untouched: mastery gate blocks

    def test_msa_enabled_pin_applies_and_blocks_shrink(self):
        from dexpbt_isaaclab.pbt.pbt_curriculum_p1 import CurriculumControlState
        control = CurriculumControlState(
            msa_enabled=True, msa_forced_success_tolerance=0.0219,
            allow_shrink=True, shrink_token=True)
        tol, _ = self._apply(control)
        self.assertEqual(tol, 0.0219)  # pinned, no shrink from the pin

    def test_msa_pin_has_priority_over_observer_forced(self):
        from dexpbt_isaaclab.pbt.pbt_curriculum_p1 import CurriculumControlState
        control = CurriculumControlState(
            msa_enabled=True, msa_forced_success_tolerance=0.0219,
            forced_success_tolerance=0.05)
        tol, _ = self._apply(control)
        self.assertEqual(tol, 0.0219)


class TestContinuationGuard(unittest.TestCase):
    """First-tick validation (structured RESUME_APPLIED + one-shrink band
    + lineage/frame) and the worker-local committed-tick cap."""

    TIP_TOL = 0.023112487509668322
    ENTRY = {"frame": 5_000_000_000, "live_tolerance": TIP_TOL,
             "lineage_id": "lin-1", "iteration": 160}
    RESUME_OK = {"applied": {"success_tolerance": TIP_TOL}}

    def test_first_tick_pass(self):
        from dexpbt_isaaclab.msa.continuation_guard import check_first_tick
        meta = {"frame": 5_080_000_000, "lineage_id": "lin-1",
                "curriculum": {"success_tolerance": self.TIP_TOL}}
        v = check_first_tick(self.ENTRY, meta, self.RESUME_OK)
        self.assertTrue(v.ok)
        # held EXACTLY the source tolerance -> stall reproduced
        self.assertTrue(v.reproduced)

    def test_first_tick_missing_resume_applied_fails_closed(self):
        from dexpbt_isaaclab.msa.continuation_guard import check_first_tick
        meta = {"frame": 5_080_000_000, "lineage_id": "lin-1",
                "curriculum": {"success_tolerance": self.TIP_TOL}}
        v = check_first_tick(self.ENTRY, meta, None)
        self.assertFalse(v.ok)
        self.assertIn("RESUME_APPLIED", v.reason)

    def test_first_tick_resume_applied_wrong_tolerance_fails(self):
        from dexpbt_isaaclab.msa.continuation_guard import check_first_tick
        meta = {"frame": 5_080_000_000, "lineage_id": "lin-1",
                "curriculum": {"success_tolerance": self.TIP_TOL}}
        v = check_first_tick(self.ENTRY, meta,
                             {"applied": {"success_tolerance": 0.075}})
        self.assertFalse(v.ok)
        self.assertIn("tolerance resume failed", v.reason)

    def test_first_tick_one_legal_shrink_passes_not_reproduced(self):
        # a live curriculum may legally shrink ONCE inside the interval;
        # the worker passes first-tick validation but does NOT count as
        # having reproduced the stall.
        from dexpbt_isaaclab.msa.continuation_guard import check_first_tick
        meta = {"frame": 5_080_000_000, "lineage_id": "lin-1",
                "curriculum": {"success_tolerance": 0.95 * self.TIP_TOL}}
        v = check_first_tick(self.ENTRY, meta, self.RESUME_OK)
        self.assertTrue(v.ok)
        self.assertFalse(v.reproduced)

    def test_first_tick_legacy_cascade_fails_closed(self):
        # two+ shrinks in one committed interval — the exact 2026-07-19
        # legacy-fallback signature (0.0231 -> 0.0168) — must fail.
        from dexpbt_isaaclab.msa.continuation_guard import check_first_tick
        meta = {"frame": 5_080_000_000, "lineage_id": "lin-1",
                "curriculum": {"success_tolerance": 0.016849}}
        v = check_first_tick(self.ENTRY, meta, self.RESUME_OK)
        self.assertFalse(v.ok)
        self.assertIn("legacy", v.reason)

    def test_first_tick_tolerance_reset_fails_closed(self):
        from dexpbt_isaaclab.msa.continuation_guard import check_first_tick
        meta = {"frame": 5_080_000_000, "lineage_id": "lin-1",
                "curriculum": {"success_tolerance": 0.075}}  # reset to initial
        v = check_first_tick(self.ENTRY, meta, self.RESUME_OK)
        self.assertFalse(v.ok)

    def test_first_tick_frame_reset_fails_closed(self):
        from dexpbt_isaaclab.msa.continuation_guard import check_first_tick
        meta = {"frame": 20_000_000, "lineage_id": "lin-1",
                "curriculum": {"success_tolerance": self.TIP_TOL}}
        v = check_first_tick(self.ENTRY, meta, self.RESUME_OK)
        self.assertFalse(v.ok)
        self.assertIn("baseline", v.reason)

    def test_first_tick_lineage_mismatch_fails_closed(self):
        from dexpbt_isaaclab.msa.continuation_guard import check_first_tick
        meta = {"frame": 5_080_000_000, "lineage_id": "OTHER",
                "curriculum": {"success_tolerance": self.TIP_TOL}}
        self.assertFalse(
            check_first_tick(self.ENTRY, meta, self.RESUME_OK).ok)

    def test_worker_local_cap_at_real_cadence(self):
        from dexpbt_isaaclab.msa.continuation_guard import (
            COMMITTED_TICK_FRAMES,
            CONTINUATION_CAP_FRAMES,
            TICK_SLACK_FRAMES,
            cap_action,
        )
        # budget = 6 committed ticks at the REAL 80M stage-3 cadence; the
        # base 20M pbt_interval is NOT the committed quantum.
        self.assertEqual(COMMITTED_TICK_FRAMES, 80_000_000)
        self.assertEqual(CONTINUATION_CAP_FRAMES, 480_000_000)
        self.assertEqual(TICK_SLACK_FRAMES, 80_000_000)
        target = 5_000_000_000 + CONTINUATION_CAP_FRAMES
        # below target: keep running
        self.assertEqual(cap_action(self.ENTRY, {"frame": target - 1}),
                         "run")
        # first frame >= target: park THIS worker (normal completion path)
        self.assertEqual(cap_action(self.ENTRY, {"frame": target}), "park")
        self.assertEqual(cap_action(self.ENTRY,
                                    {"frame": target + 80_000_000}),
                         "park")   # within one tick of slack
        # beyond target + slack: fail — a worker must never silently
        # overshoot a full committed tick past its budget
        self.assertEqual(cap_action(self.ENTRY,
                                    {"frame": target + 80_000_001}),
                         "fail")

    def test_bootstrap_verify_hash_and_mapping(self):
        from dexpbt_isaaclab.msa.continuation_guard import (
            ckpt_head_hash,
            verify_bootstrap,
        )
        with tempfile.TemporaryDirectory() as td:
            entries = {}
            for w in range(6):
                d = Path(td) / f"{w:03d}"
                d.mkdir()
                p = d / "000160.pth"
                p.write_bytes(b"ckpt-%d" % w)
                entries[f"p{w}"] = {
                    "path": str(p), "frame": 5_000_000_000,
                    "live_tolerance": 0.023, "lineage_id": f"l{w}",
                    "iteration": 160, "sha256_head": ckpt_head_hash(p)}
            self.assertEqual(verify_bootstrap(
                {"resume_checkpoints": entries}), [])
            # tampered checkpoint -> hash failure
            Path(entries["p2"]["path"]).write_bytes(b"TAMPERED")
            fails = verify_bootstrap({"resume_checkpoints": entries})
            self.assertTrue(any("hash mismatch" in f for f in fails))


class TestOriginalProfileUnchanged(unittest.TestCase):
    # J15: original base-retention behavior is untouched
    def test_control_state_defaults_inert(self):
        from dexpbt_isaaclab.pbt.pbt_curriculum_p1 import CurriculumControlState
        s = CurriculumControlState()
        self.assertFalse(s.msa_enabled)
        self.assertEqual(s.msa_forced_success_tolerance, 0.0)
        self.assertIsNone(s.msa_provenance)
        self.assertIsNone(s.msa_state_machine)
        self.assertIsNone(s.msa_cycle)

    def test_old_control_json_still_reads(self):
        from dexpbt_isaaclab.pbt.pbt_curriculum_p1 import (
            CurriculumControlState,
            read_curriculum_control,
            write_curriculum_control,
        )
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "000.json"
            # simulate a pre-MSA control JSON (no msa_* keys)
            legacy = CurriculumControlState().to_json()
            for k in ("msa_enabled", "msa_forced_success_tolerance",
                      "msa_provenance", "msa_state_machine", "msa_cycle"):
                legacy.pop(k)
            p.write_text(json.dumps(legacy))
            state = read_curriculum_control(p)
            self.assertIsNotNone(state)
            self.assertFalse(state.msa_enabled)
            self.assertIsNone(state.msa_provenance)
            # roundtrip through the writer keeps working
            write_curriculum_control(p, state)
            self.assertFalse(read_curriculum_control(p).msa_enabled)

    def test_msa_disabled_curriculum_behavior_unchanged(self):
        """With msa_enabled=False the mastery curriculum path is bit-for-bit
        the pre-MSA behavior (shrink still allowed when gates pass)."""
        import torch

        from dexpbt_isaaclab.pbt import pbt_curriculum_p1 as mod
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "000.json"
            mod.write_curriculum_control(
                path, mod.CurriculumControlState(allow_shrink=True,
                                                 shrink_token=True))
            tol, _ = mod.apply_env_tolerance_curriculum(
                last_curriculum_update=0,
                frames_since_restart=10_000_000,
                curriculum_interval=1_000,
                prev_episode_successes=torch.ones(8),
                success_tolerance=0.023,
                initial_tolerance=0.075,
                target_tolerance=0.01,
                curriculum_mode="mastery",
                curriculum_control_path=path,
                curriculum_shrink_factor=0.95,
                post_shrink_consolidation_env_frames=0,
                num_envs=8,
            )
            self.assertAlmostEqual(tol, 0.023 * 0.95)

    def test_base_continuation_launcher_keeps_original_gates(self):
        if not LAUNCHER_DIR.exists():
            self.skipTest("arm launchers not generated in this checkout")
        base = (LAUNCHER_DIR / "start_base_stateful.sh").read_text()
        relax = (LAUNCHER_DIR / "start_gate_relax_stateful.sh").read_text()
        self.assertIn("--late_floor_retention_min_retention 0.88", base)
        self.assertIn("--late_floor_retention_min_retention 0.75", relax)
        # single-flag diff: everything else identical
        diff = [
            (a, b) for a, b in zip(base.splitlines(), relax.splitlines())
            if a != b and "GATE_RELAX" not in a + b
            and "gate_relax" not in a + b and "BASE_STATEFUL" not in a + b
            and "base_stateful" not in a + b
        ]
        self.assertEqual(len(diff), 1, diff)
        self.assertIn("late_floor_retention_min_retention", diff[0][0])
        # all arms refuse to run without the explicit safety flag
        for sh in LAUNCHER_DIR.glob("start_*.sh"):
            self.assertIn("REFUSING TO LAUNCH", sh.read_text())

    def test_gate_relax_goes_through_dedicated_profile(self):
        """Profile expansion is authoritative: a CLI-only override of
        late_floor_retention_min_retention would be silently reverted to the
        profile table value. GATE_RELAX must therefore select the dedicated
        cser_dexpbt_conservative_gate_relax profile; all other arms keep the
        original profile."""
        from dexpbt_isaaclab.world_model.wm_strategy_profiles import (
            PROFILE_IDS,
            PROFILE_TABLES,
        )
        table = PROFILE_TABLES["cser_dexpbt_conservative_gate_relax"]
        base = PROFILE_TABLES["cser_dexpbt_conservative"]
        self.assertEqual(table["late_floor_retention_min_retention"], 0.75)
        # exactly one key differs from the base profile
        diff = {k for k in table if table[k] != base.get(k)}
        self.assertEqual(diff, {"late_floor_retention_min_retention"})
        self.assertEqual(set(base) | {"late_floor_retention_min_retention"},
                         set(table) | set(base))
        self.assertIn("cser_dexpbt_conservative_gate_relax", PROFILE_IDS)
        # base profile table itself is untouched
        self.assertEqual(base["late_floor_retention_min_retention"], 0.88)

        if not LAUNCHER_DIR.exists():
            self.skipTest("arm launchers not generated in this checkout")
        relax = (LAUNCHER_DIR / "start_gate_relax_stateful.sh").read_text()
        self.assertIn('--profile-per-policy "'
                      + ",".join(["cser_dexpbt_conservative_gate_relax"] * 6)
                      + '"', relax)
        for other in ("start_base_stateful.sh", "start_msa_stateful.sh",
                      "start_msa_permuted_stateful.sh"):
            text = (LAUNCHER_DIR / other).read_text()
            self.assertIn('--profile-per-policy "'
                          + ",".join(["cser_dexpbt_conservative"] * 6)
                          + '"', text)
            self.assertNotIn("gate_relax", text)

    def test_launchers_have_guard_tolerance_resume_and_eval_device(self):
        if not LAUNCHER_DIR.exists():
            self.skipTest("arm launchers not generated in this checkout")
        for sh in LAUNCHER_DIR.glob("start_*.sh"):
            if sh.name in ("run_hold_smoke.sh",
                           "start_stateful_hold_smoke.sh"):
                continue
            text = sh.read_text()
            # STATE-FAITHFUL bootstrap before anything launches (fail
            # closed) + fingerprint validation
            self.assertIn("state_bootstrap", text)
            self.assertIn("REFUSING TO LAUNCH: state bootstrap failed", text)
            self.assertIn("--verify", text)
            self.assertIn("state fingerprint validation failed", text)
            # fail-closed bootstrap verification before any launch
            self.assertIn("continuation_guard", text)
            self.assertIn("--verify-only", text)
            self.assertIn("REFUSING TO LAUNCH: bootstrap verification",
                          text)
            # explicit tolerance + env-clock resume (a bare --checkpoint
            # resets tolerance AND restart-step-relative gates)
            self.assertIn("--tolerance-resume-per-policy", text)
            self.assertIn("--curriculum-clock-resume-per-policy", text)
            # hard cap guard loop at the real cadence budget
            self.assertIn("continuation_guard.log", text)
            self.assertIn("--cap-frames 480000000", text)
        for arm in ("start_msa_stateful.sh",
                    "start_msa_permuted_stateful.sh"):
            text = (LAUNCHER_DIR / arm).read_text()
            # bounded multi-device eval, never the trainers' cuda:0
            self.assertIn("--eval-devices", text)
            # SIX-GPU machine: eval SHARES cuda:4,cuda:5 with trainers
            # 4/5 (SIGSTOPped during eval barriers, VRAM stays allocated)
            self.assertIn("cuda:4,cuda:5", text)
            self.assertNotIn("cuda:6,cuda:7", text)
            self.assertNotIn("trainers then share 0-3", text)
            # N=4 execution-smoke default; pilot N=16 explicit only
            self.assertIn('--num-seeds "${MSA_NUM_SEEDS:-4}"', text)
            # 6/6 readiness barrier gate for the sidecar
            self.assertIn("--readiness-marker", text)
            self.assertIn("FIRST_TICK_VALIDATED.json", text)
            # hold smoke gate: DISPOSABLE certificate, validated at sidecar
            # startup; the launcher refuses to start without the file
            self.assertIn("--hold-smoke-certificate", text)
            self.assertIn("HOLD_SMOKE_CERTIFICATE.json", text)
            self.assertIn("--wm-profile", text)
            self.assertIn("hold smoke certificate missing", text)
            # bounded durable-commit wait
            self.assertIn("--commit-max-wait-s", text)
            # external hold watchdog independent of the sidecar
            self.assertIn("hold_watchdog", text)
        # the in-arm hold smoke is EXECUTION SMOKE ONLY (it advances the
        # released worker one extra tick past the matched boundary) and
        # exercises the shared-GPU path via worker 4
        smoke = (LAUNCHER_DIR / "run_hold_smoke.sh").read_text()
        self.assertIn("--hold-smoke-only", smoke)
        self.assertIn("EXECUTION SMOKE ONLY", smoke)
        self.assertIn("FORBIDDEN in the formal four-arm comparison", smoke)
        self.assertIn('WORKER="${2:-4}"', smoke)
        self.assertIn("--hold-smoke-worker", smoke)
        self.assertIn("HOLD_SMOKE_RESULT.json", smoke)
        # DISPOSABLE certificate smoke launcher (one throwaway worker on
        # the shared eval GPU; produces the certificate the formal MSA
        # launchers require)
        disp = (LAUNCHER_DIR / "start_stateful_hold_smoke.sh").read_text()
        self.assertIn("--certificate-smoke", disp)
        self.assertIn("HOLD_SMOKE_CERTIFICATE.json", disp)
        self.assertIn('SMOKE_GPU="${SMOKE_GPU:-4}"', disp)
        self.assertIn("--num-envs 8192", disp)
        self.assertIn("--wm-profile", disp)
        self.assertIn("stop_arm", disp)   # disposable run torn down

    def test_six_gpu_metadata_recorded(self):
        # NEW TEST 8: the MSA launchers record the ACTUAL eval devices
        # (default cuda:4,cuda:5 on this six-GPU machine) into run
        # metadata, with the shared-GPU semantics spelled out.
        if not LAUNCHER_DIR.exists():
            self.skipTest("arm launchers not generated in this checkout")
        for arm in ("start_msa_stateful.sh",
                    "start_msa_permuted_stateful.sh"):
            text = (LAUNCHER_DIR / arm).read_text()
            self.assertIn(
                'MSA_EVAL_DEVICES_ACTUAL="${MSA_EVAL_DEVICES:-cuda:4,cuda:5}"',
                text)
            self.assertIn('EVAL_DEVICES.json', text)
            self.assertIn('"eval_devices": "$MSA_EVAL_DEVICES_ACTUAL"', text)
            self.assertIn('"trainer_gpus": "$DEXPBT_GPUS"', text)
            self.assertIn("SIGSTOPped during eval barriers", text)
            self.assertIn("keep their VRAM", text)


class TestAudit(unittest.TestCase):
    def test_audit_synthetic_run(self):
        from dexpbt_isaaclab.msa.audit import audit_run
        with tempfile.TemporaryDirectory() as td:
            run = Path(td)
            metrics = run / "dexpbt_8gpu_ws" / "metrics"
            metrics.mkdir(parents=True)
            rows = [
                {"policy_idx": 0, "env_frames": 3_000_000_000,
                 "success_tolerance": 0.025,
                 "curriculum_shrink_reason": "blocked_consolidation"},
                {"policy_idx": 0, "env_frames": 3_050_000_000,
                 "success_tolerance": 0.025,
                 "curriculum_shrink_reason": "blocked_collapse"},
                {"policy_idx": 0, "env_frames": 3_100_000_000,
                 "success_tolerance": 0.024,   # shrank: not an opportunity
                 "curriculum_shrink_reason": "mid"},
                {"policy_idx": 0, "env_frames": 3_150_000_000,
                 "success_tolerance": 0.024,
                 "curriculum_shrink_reason": "curriculum_safety_blocked",
                 "curriculum_safety_block_reason": "curriculum_safety_freeze_active"},
            ]
            with open(metrics / "training_curve_policy_00.jsonl", "w") as f:
                for r in rows:
                    f.write(json.dumps(r) + "\n")
            with open(metrics / "curriculum_safety_events.jsonl", "w") as f:
                f.write(json.dumps({
                    "event": "late_floor_retention_block", "policy_idx": 0,
                    "env_frames": 3_130_000_000}) + "\n")
            out = audit_run(run, 3_000_000_000, 5_000_000_000, num_policies=1)
            counts = out["per_policy"]["p0"]["first_blocker_counts"]
            self.assertEqual(counts.get("rollback_collapse_freeze"), 1)
            self.assertEqual(counts.get("late_floor_retention_gate"), 1)
            self.assertEqual(out["per_policy"]["p0"]["shrink_intervals"], 1)


class TestArmConfigs(unittest.TestCase):
    def test_arm_configs_matched_and_charged(self):
        if not LAUNCHER_DIR.exists():
            self.skipTest("arm configs not generated in this checkout")
        cfgs = {p.stem.replace(".config", ""): json.loads(p.read_text())
                for p in LAUNCHER_DIR.glob("*.config.json")}
        self.assertEqual(len(cfgs), 4)
        seeds = {c["base_seed"] for c in cfgs.values()}
        self.assertEqual(len(seeds), 1)          # matched seeds
        resumes = {json.dumps(c["resume_checkpoints"], sort_keys=True)
                   for c in cfgs.values()}
        self.assertEqual(len(resumes), 1)        # same six 5B checkpoints
        for c in cfgs.values():
            self.assertTrue(c["frame_ledger"]["monotonic"])
            self.assertTrue(c["frame_ledger"]["probe_frames_charged"])
            for entry in c["resume_checkpoints"].values():
                self.assertTrue(entry.get("sha256_head"),
                                "bootstrap hash missing")
        self.assertTrue(
            cfgs["msa_permuted_stateful"]["msa"]["permuted_route"])
        self.assertFalse(cfgs["msa_stateful"]["msa"]["permuted_route"])
        # one committed tick at the REAL 80M stage-3 cadence everywhere
        for name in ("msa_stateful", "msa_permuted_stateful"):
            self.assertEqual(cfgs[name]["msa"]["packet_frames"], 80_000_000)
            self.assertEqual(cfgs[name]["msa"]["eval_num_seeds_pilot"],
                             PILOT_NUM_SEEDS)
        # same probe dose/eval budget between MSA and its negative control
        for k in ("packet_frames", "top_k_recipients",
                  "eval_num_seeds_pilot", "eval_seed_manifest_id_pilot"):
            self.assertEqual(cfgs["msa_stateful"]["msa"][k],
                             cfgs["msa_permuted_stateful"]["msa"][k])
        # state-faithful metadata on every arm
        for c in cfgs.values():
            self.assertTrue(c["state_faithful"])
            self.assertIn("source_run", c)
            self.assertEqual(c["committed_tick_frames"], 80_000_000)
            self.assertEqual(c["continuation_frames_per_worker"],
                             480_000_000)
            self.assertEqual(sum(c["budget_breakdown_ticks"].values()),
                             c["continuation_ticks_per_worker"])


def _mk_workspace(root: Path, frames: dict[int, int], tol: float = 0.023,
                  prefix: str = "000161") -> Path:
    """Synthetic committed-tick workspace (newest .pth/.json pair)."""
    ws = Path(root) / "dexpbt_8gpu_ws"
    for w, frame in frames.items():
        d = ws / f"{w:03d}"
        d.mkdir(parents=True, exist_ok=True)
        (d / f"{prefix}.pth").write_bytes(b"ckpt")
        (d / f"{prefix}.json").write_text(json.dumps({
            "frame": int(frame), "lineage_id": f"lin{w}",
            "curriculum": {"success_tolerance": tol}}))
    return ws


def _guard_config(tol: float = 0.023,
                  baseline: int = 5_000_000_000) -> dict:
    return {"resume_checkpoints": {
        f"p{w}": {"frame": baseline, "live_tolerance": tol,
                  "lineage_id": f"lin{w}", "iteration": 160,
                  "path": f"/x/{w:03d}/000160.pth", "sha256_head": "h"}
        for w in range(6)}}


def _reap(proc) -> None:
    try:
        proc.kill()
    except Exception:
        pass
    try:
        proc.wait(timeout=5)
    except Exception:
        pass


def _spawn_worker_procs(testcase, run_dir: Path, workers=range(6),
                        cmd=("sleep", "300")) -> dict:
    """One REAL 'trainer' process per worker with a pid.txt, so the guard's
    readiness barrier has an actual process tree to SIGSTOP."""
    import subprocess
    procs = {}
    for w in workers:
        proc = subprocess.Popen(list(cmd))
        testcase.addCleanup(_reap, proc)
        d = Path(run_dir) / f"policy_{w:02d}"
        d.mkdir(parents=True, exist_ok=True)
        (d / "pid.txt").write_text(str(proc.pid))
        procs[w] = proc
    return procs


def _write_marker_stub(run_dir: Path,
                       frame: int = 5_080_000_000,
                       tol: float = 0.023) -> None:
    """A pre-existing FIRST_TICK_VALIDATED.json (readiness completed in an
    earlier guard lifetime) so cap-focused tests skip the barrier."""
    (Path(run_dir) / "FIRST_TICK_VALIDATED.json").write_text(json.dumps({
        "workers": [{"worker_id": w, "validated_frame": frame,
                     "validated_tolerance": tol, "lineage_id": f"lin{w}",
                     "checkpoint_identity": {}} for w in range(6)],
        "timestamp": 0.0, "num_workers": 6}))


def _spawn_stopped(testcase, cmd=("sleep", "120")):
    """A real process SIGSTOPped into the 'T' state."""
    import os as _os
    import signal as _sg
    import subprocess
    import time as _t

    from dexpbt_isaaclab.msa.sidecar import _proc_state
    p = subprocess.Popen(list(cmd))
    testcase.addCleanup(_reap, p)
    _os.kill(p.pid, _sg.SIGSTOP)
    dl = _t.monotonic() + 5
    while _proc_state(p.pid) not in ("T", "t") and _t.monotonic() < dl:
        _t.sleep(0.02)
    return p


def _spawn_running(testcase, cmd=("sleep", "120")):
    import subprocess
    p = subprocess.Popen(list(cmd))
    testcase.addCleanup(_reap, p)
    return p


def _dead_pid() -> int:
    import subprocess
    d = subprocess.Popen(["true"])
    d.wait()
    return d.pid


def _write_readiness_holds_file(run_dir: Path, procs: dict,
                                frame: int = 5_080_000_000,
                                tol: float = 0.023) -> None:
    (Path(run_dir) / "READINESS_HOLDS.json").write_text(json.dumps({
        str(w): {"worker_id": w, "root_pid": p.pid, "pids": [p.pid],
                 "validated_frame": frame, "validated_tolerance": tol,
                 "lineage_id": f"lin{w}", "hold_timestamp": 0.0}
        for w, p in procs.items()}))


# a caller process that registers its OWN pid as a control pid, invokes a
# stop path, then writes a completion marker — used to prove stop_arm is
# caller-safe (it must NOT kill the caller before the arm is down).
_CALLER_SCRIPT = """
import os, sys, json
from pathlib import Path
run_dir = Path(sys.argv[1]); pid_name = sys.argv[2]; via = sys.argv[3]
(run_dir / pid_name).write_text(str(os.getpid()))
if via == "watchdog":
    from dexpbt_isaaclab.msa.hold_watchdog import check_once
    check_once(run_dir)                       # real stop_arm inside
else:
    from dexpbt_isaaclab.msa.continuation_guard import stop_arm
    stop_arm(run_dir, "self-invoked stop", marker_name="GUARD_FAILED")
(run_dir / "CALLER_DONE.json").write_text(json.dumps({"pid": os.getpid()}))
"""


def _run_caller(run_dir: Path, pid_name: str, via: str):
    import os as _os
    import subprocess
    env = dict(_os.environ)
    env["PYTHONPATH"] = str(SRC_DIR) + _os.pathsep + env.get("PYTHONPATH", "")
    return subprocess.run([sys.executable, "-c", _CALLER_SCRIPT,
                           str(run_dir), pid_name, via],
                          env=env, capture_output=True, text=True,
                          timeout=90)


class TestReadinessBarrier(unittest.TestCase):
    """6/6 first-tick readiness: the guard writes FIRST_TICK_VALIDATED.json
    only after ALL SIX workers validated; the sidecar refuses to start or
    resume a cycle before the marker exists and views are exactly {0..5}."""

    def test_readiness_marker_requires_6_of_6(self):
        from dexpbt_isaaclab.msa.continuation_guard import (
            write_readiness_marker,
        )
        with tempfile.TemporaryDirectory() as td:
            partial = {w: {"worker_id": w} for w in range(3)}
            with self.assertRaises(ValueError):
                write_readiness_marker(Path(td), partial)
            self.assertFalse(
                (Path(td) / "FIRST_TICK_VALIDATED.json").exists())

    def test_marker_contents_and_atomic_write(self):
        from dexpbt_isaaclab.msa.continuation_guard import (
            READINESS_MARKER,
            readiness_entry,
            write_readiness_marker,
        )
        with tempfile.TemporaryDirectory() as td:
            entry = {"path": "/x/000/000160.pth", "sha256_head": "abc",
                     "frame": 5_000_000_000, "live_tolerance": 0.023,
                     "lineage_id": "lin0"}
            meta = {"frame": 5_020_000_000, "lineage_id": "lin0",
                    "_committed_prefix": "000161",
                    "curriculum": {"success_tolerance": 0.023}}
            records = {w: readiness_entry(w, entry, meta) for w in range(6)}
            marker = write_readiness_marker(Path(td), records)
            self.assertEqual(marker.name, READINESS_MARKER)
            payload = json.loads(marker.read_text())
            self.assertEqual(len(payload["workers"]), 6)
            self.assertIn("timestamp", payload)
            w0 = payload["workers"][0]
            self.assertEqual(w0["validated_frame"], 5_020_000_000)
            self.assertEqual(w0["validated_tolerance"], 0.023)
            self.assertEqual(w0["lineage_id"], "lin0")
            self.assertEqual(
                w0["checkpoint_identity"]["resume_checkpoint"],
                "/x/000/000160.pth")
            self.assertEqual(
                w0["checkpoint_identity"]["committed_prefix"], "000161")
            # no leftover tmp files (atomic os.replace)
            self.assertEqual(list(Path(td).glob("*.tmp.*")), [])

    def test_guard_loop_partial_workers_no_marker(self):
        # NEW TESTS 1+2: an early worker is FROZEN at its validated tick
        # (its frame cannot advance) and before 6/6 there is NO validated
        # worker left unfrozen.
        from dexpbt_isaaclab.msa.continuation_guard import (
            READINESS_MARKER,
            guard_loop,
            read_readiness_holds,
        )
        from dexpbt_isaaclab.msa.sidecar import _proc_state
        with tempfile.TemporaryDirectory() as td:
            run_dir = Path(td)
            # only 3 of 6 workers produced a continuation tick
            frames = {w: 5_020_000_000 for w in range(3)}
            frames.update({w: 5_000_000_000 for w in range(3, 6)})
            ws = _mk_workspace(run_dir, frames)
            procs = _spawn_worker_procs(self, run_dir)
            rc = guard_loop(run_dir, ws, _guard_config(), max_iterations=1)
            self.assertEqual(rc, 0)   # still watching, nothing failed
            self.assertFalse((run_dir / READINESS_MARKER).exists())
            self.assertFalse((run_dir / "GUARD_FAILED").exists())
            holds = read_readiness_holds(run_dir)
            # every VALIDATED worker is frozen ('T'), nobody else is
            self.assertEqual(sorted(holds), [0, 1, 2])
            for w in range(3):
                self.assertIn(_proc_state(procs[w].pid), ("T", "t"))
                self.assertEqual(holds[w]["validated_frame"], 5_020_000_000)
                self.assertIn("validated_tolerance", holds[w])
                self.assertIn("lineage_id", holds[w])
                self.assertIn("hold_timestamp", holds[w])
                self.assertIn(procs[w].pid,
                              [int(p) for p in holds[w]["pids"]])
            for w in range(3, 6):    # not yet validated -> still running
                self.assertNotIn(_proc_state(procs[w].pid), ("T", "t"))

    def test_early_frozen_worker_makes_no_progress(self):
        # NEW TEST 1 (progress check): a counter process stands in for a
        # trainer; after the readiness freeze its output stops advancing.
        import time as _time

        from dexpbt_isaaclab.msa.continuation_guard import guard_loop
        from dexpbt_isaaclab.msa.sidecar import _proc_state
        with tempfile.TemporaryDirectory() as td:
            run_dir = Path(td)
            frames = {0: 5_020_000_000}
            frames.update({w: 5_000_000_000 for w in range(1, 6)})
            ws = _mk_workspace(run_dir, frames)
            counter = run_dir / "counter.txt"
            procs = _spawn_worker_procs(
                self, run_dir, workers=[0],
                cmd=("bash", "-c",
                     f'i=0; while :; do i=$((i+1)); echo $i > "{counter}";'
                     ' sleep 0.02; done'))
            deadline = _time.monotonic() + 5
            while not counter.exists() and _time.monotonic() < deadline:
                _time.sleep(0.05)
            rc = guard_loop(run_dir, ws, _guard_config(), max_iterations=1)
            self.assertEqual(rc, 0)
            self.assertIn(_proc_state(procs[0].pid), ("T", "t"))
            v1 = counter.read_text()
            _time.sleep(0.5)
            self.assertEqual(counter.read_text(), v1,
                             "frozen worker still made progress")
            # cleanup: release so no stopped child outlives the test
            from dexpbt_isaaclab.msa.continuation_guard import (
                release_readiness_holds,
            )
            release_readiness_holds(run_dir)

    def test_guard_loop_writes_marker_only_after_all_six(self):
        # NEW TEST 3: with 6/6 ready on a non-MSA arm all six workers are
        # released SIMULTANEOUSLY and the readiness holds file is gone.
        from dexpbt_isaaclab.msa.continuation_guard import (
            READINESS_HOLDS_FILE,
            READINESS_MARKER,
            guard_loop,
        )
        from dexpbt_isaaclab.msa.sidecar import _proc_state
        with tempfile.TemporaryDirectory() as td:
            run_dir = Path(td)
            ws = _mk_workspace(run_dir,
                               {w: 5_020_000_000 for w in range(6)})
            procs = _spawn_worker_procs(self, run_dir)
            rc = guard_loop(run_dir, ws, _guard_config(), max_iterations=1,
                            msa_handoff=False)
            self.assertEqual(rc, 0)
            payload = json.loads((run_dir / READINESS_MARKER).read_text())
            self.assertEqual(
                sorted(w["worker_id"] for w in payload["workers"]),
                list(range(6)))
            # simultaneous release: nobody stays frozen, record removed
            self.assertFalse((run_dir / READINESS_HOLDS_FILE).exists())
            for w in range(6):
                self.assertNotIn(_proc_state(procs[w].pid), ("T", "t"))

    def test_guard_loop_msa_handoff_keeps_workers_frozen(self):
        # MSA arms: after the marker the workers STAY frozen at their
        # validated boundaries for the sidecar to adopt.
        from dexpbt_isaaclab.msa.continuation_guard import (
            READINESS_MARKER,
            guard_loop,
            read_readiness_holds,
        )
        from dexpbt_isaaclab.msa.sidecar import _proc_state
        with tempfile.TemporaryDirectory() as td:
            run_dir = Path(td)
            ws = _mk_workspace(run_dir,
                               {w: 5_020_000_000 for w in range(6)})
            procs = _spawn_worker_procs(self, run_dir)
            config = dict(_guard_config())
            config["arm"] = "MSA_STATEFUL"    # handoff derived from config
            rc = guard_loop(run_dir, ws, config, max_iterations=1)
            self.assertEqual(rc, 0)
            self.assertTrue((run_dir / READINESS_MARKER).exists())
            holds = read_readiness_holds(run_dir)
            self.assertEqual(sorted(holds), list(range(6)))
            for w in range(6):
                self.assertIn(_proc_state(procs[w].pid), ("T", "t"))

    def test_guard_loop_any_failure_stops_arm_without_marker(self):
        from dexpbt_isaaclab.msa.continuation_guard import (
            READINESS_MARKER,
            guard_loop,
        )
        from dexpbt_isaaclab.msa.sidecar import _proc_state
        with tempfile.TemporaryDirectory() as td:
            run_dir = Path(td)
            ws = _mk_workspace(run_dir,
                               {w: 5_020_000_000 for w in range(6)})
            # worker 4's tolerance reset to the 0.075 initial value
            bad = ws / "004" / "000161.json"
            meta = json.loads(bad.read_text())
            meta["curriculum"]["success_tolerance"] = 0.075
            bad.write_text(json.dumps(meta))
            procs = _spawn_worker_procs(self, run_dir)
            rc = guard_loop(run_dir, ws, _guard_config(), max_iterations=1,
                            msa_handoff=False)
            self.assertEqual(rc, 2)
            self.assertTrue((run_dir / "GUARD_FAILED").exists())
            self.assertFalse((run_dir / READINESS_MARKER).exists())
            # the already-frozen workers (0-3) were SIGCONTed and the whole
            # arm terminated: nobody may be left SIGSTOPped
            for w in range(6):
                self.assertNotIn(_proc_state(procs[w].pid), ("T", "t"))

    def test_sidecar_readiness_gate(self):
        from dexpbt_isaaclab.msa.continuation_guard import (
            readiness_entry,
            write_readiness_marker,
        )
        from dexpbt_isaaclab.msa.sidecar import WorkerView, readiness_ok
        with tempfile.TemporaryDirectory() as td:
            marker = Path(td) / "FIRST_TICK_VALIDATED.json"

            def _views(n):
                return [WorkerView(worker_id=w, lineage_id=f"l{w}",
                                   active_tau=0.023, frame=5_020_000_000,
                                   source_checkpoint_id="c")
                        for w in range(n)]

            # no marker -> not ready even with all six views
            ok, why = readiness_ok(marker, _views(6))
            self.assertFalse(ok)
            self.assertIn("marker missing", why)
            entry = {"path": "p", "sha256_head": "h"}
            meta = {"frame": 5_020_000_000, "lineage_id": "l",
                    "curriculum": {"success_tolerance": 0.023}}
            write_readiness_marker(
                Path(td), {w: readiness_entry(w, entry, meta)
                           for w in range(6)})
            # marker present but only 4 workers visible -> not ready
            ok, why = readiness_ok(marker, _views(4))
            self.assertFalse(ok)
            self.assertIn("partial launch", why)
            # marker + exactly {0..5} -> ready
            ok, why = readiness_ok(marker, _views(6))
            self.assertTrue(ok, why)


class TestHoldRaceStartFrames(unittest.TestCase):
    """A tick committed between due detection and the PRE_EVAL hold ran at
    tau and must NEVER be counted as tau_plus exposure; packet start frames
    and snapshot frames come from the POST-hold committed read."""

    def test_stale_tick_not_counted_as_tau_plus(self):
        world, ctl, views = dr.make_world()
        # worker 2 commits ONE more tick (at its own tau) after the views
        # were taken but before the controller holds
        world._advance_one_tick(2)
        self.assertEqual(world.frame[2], 5_000_000_000 + TICK)
        result = ctl.run_cycle(views)     # views still carry the stale frame
        seg = [s for s in result.records[2]["segments"]
               if s["segment_type"] == "common_probe"][0]
        # packet start = post-hold frame, so the stale tick is EXCLUDED
        self.assertEqual(seg["segment_start_frame"], 5_000_000_000 + TICK)
        self.assertEqual(
            seg["segment_end_frame"] - seg["segment_start_frame"], TICK)
        # the stale tick is accounted at tau, never tau_plus
        recipients = set(result.routing.recipients)
        want_plus = 2 * TICK if 2 in recipients else TICK
        self.assertEqual(
            world.trained_frames_at_tol.get((2, "tau_plus"), 0), want_plus)
        # persisted start_frames match the post-hold read
        cycle = world.state_store[2]["msa_cycle"]
        self.assertEqual(int(cycle["start_frames"]["2"]),
                         5_000_000_000 + TICK)

    def test_snapshot_frame_equals_packet_start_frame(self):
        world, ctl, views = dr.make_world()
        world._advance_one_tick(5)
        result = ctl.run_cycle(views)
        iid = result.intervention_id
        for w in range(6):
            start = int(world.state_store[w]["msa_cycle"]
                        ["start_frames"][str(w)])
            self.assertEqual(world.snapshots[f"w{w}_{iid}_pre"][3], start)
            mid = int(world.state_store[w]["msa_cycle"]
                      ["mid_frames"][str(w)])
            self.assertEqual(world.snapshots[f"w{w}_{iid}_mid"][3], mid)


class TestWorkerHoldsSafety(unittest.TestCase):
    """SIGSTOP verification, children-before-root release order, and
    MSA_HOLDS.json persistence for the external watchdog."""

    def _spawn_sleeper(self, run_dir: Path, worker_id: int = 0):
        import subprocess
        proc = subprocess.Popen(["sleep", "60"])
        d = run_dir / f"policy_{worker_id:02d}"
        d.mkdir(parents=True, exist_ok=True)
        (d / "pid.txt").write_text(str(proc.pid))
        self.addCleanup(proc.kill)
        return proc

    def test_hold_verifies_stopped_and_persists(self):
        from dexpbt_isaaclab.msa.sidecar import (
            WorkerHolds,
            _proc_state,
            read_holds_file,
        )
        with tempfile.TemporaryDirectory() as td:
            run_dir = Path(td)
            proc = self._spawn_sleeper(run_dir)
            holds = WorkerHolds(run_dir)
            holds.hold([0])
            try:
                self.assertIn(_proc_state(proc.pid), ("T", "t"))
                rec = read_holds_file(run_dir)
                self.assertIn(0, rec)
                self.assertIn(proc.pid, rec[0])
            finally:
                holds.release([0])
            self.assertNotIn(_proc_state(proc.pid), ("T", "t"))
            # record removed after a normal release
            self.assertEqual(read_holds_file(run_dir), {})

    def test_release_order_children_before_root(self):
        from unittest import mock

        from dexpbt_isaaclab.msa.sidecar import WorkerHolds
        with tempfile.TemporaryDirectory() as td:
            holds = WorkerHolds(Path(td))
            # root=111, children deeper in the tree later in the list
            holds._held = {0: [111, 222, 333]}
            delivered = []

            def _fake_kill(pid, sig):
                delivered.append((pid, sig))

            with mock.patch("os.kill", _fake_kill):
                holds.release([0])
            import signal as sg
            self.assertEqual(delivered, [(333, sg.SIGCONT),
                                         (222, sg.SIGCONT),
                                         (111, sg.SIGCONT)])

    def test_hold_verification_failure_fails_closed(self):
        from unittest import mock

        from dexpbt_isaaclab.msa.sidecar import WorkerHolds
        with tempfile.TemporaryDirectory() as td:
            run_dir = Path(td)
            proc = self._spawn_sleeper(run_dir)
            holds = WorkerHolds(run_dir)
            holds.STOP_VERIFY_TIMEOUT_S = 0.3
            # simulate a process that never reaches the stopped state
            with mock.patch("dexpbt_isaaclab.msa.sidecar._proc_state",
                            return_value="R"):
                with self.assertRaises(RuntimeError):
                    holds.hold([0])
            # fail-closed: nothing recorded as held, SIGCONT was sent back
            self.assertEqual(holds._held, {})
            self.assertNotIn(proc.pid, [p for ps in holds._held.values()
                                        for p in ps])


class TestHoldWatchdog(unittest.TestCase):
    """External watchdog: a dead sidecar with recorded holds -> SIGCONT all
    held PIDs, MSA_SIDECAR_DIED.json, stop the whole arm."""

    def test_sidecar_alive_is_ok(self):
        import os as _os

        from dexpbt_isaaclab.msa.hold_watchdog import check_once
        with tempfile.TemporaryDirectory() as td:
            (Path(td) / "continuation_guard.pid").write_text(
                str(_os.getpid()))
            (Path(td) / "msa_sidecar.pid").write_text(str(_os.getpid()))
            self.assertEqual(check_once(Path(td)), "ok")

    def test_dead_sidecar_no_holds_keeps_watching_until_arm_complete(self):
        # guard lifetime supervision: a dead sidecar with nothing frozen is
        # NOT idle any more — the guard is still supervised. Only
        # ARM_COMPLETE.json (normal completion) allows a quiet exit.
        from dexpbt_isaaclab.msa.hold_watchdog import check_once
        with tempfile.TemporaryDirectory() as td:
            run_dir = Path(td)
            guard = _spawn_running(self)
            (run_dir / "continuation_guard.pid").write_text(str(guard.pid))
            (run_dir / "msa_sidecar.pid").write_text(str(_dead_pid()))
            self.assertEqual(check_once(run_dir), "ok")
            (run_dir / "ARM_COMPLETE.json").write_text("{}")
            self.assertEqual(check_once(run_dir), "idle_exit")

    def test_dead_sidecar_with_holds_releases_and_stops(self):
        import os as _os
        import signal as sg
        import subprocess
        import time as _time

        from dexpbt_isaaclab.msa.hold_watchdog import check_once
        from dexpbt_isaaclab.msa.sidecar import _proc_state
        with tempfile.TemporaryDirectory() as td:
            run_dir = Path(td)
            guard = _spawn_running(self)              # guard still alive
            (run_dir / "continuation_guard.pid").write_text(str(guard.pid))
            held = subprocess.Popen(["sleep", "60"])
            self.addCleanup(held.kill)
            _os.kill(held.pid, sg.SIGSTOP)
            deadline = _time.monotonic() + 5
            while (_proc_state(held.pid) not in ("T", "t")
                   and _time.monotonic() < deadline):
                _time.sleep(0.05)
            (run_dir / "MSA_HOLDS.json").write_text(
                json.dumps({"0": [held.pid]}))
            dead = subprocess.Popen(["true"])
            dead.wait()
            (run_dir / "msa_sidecar.pid").write_text(str(dead.pid))
            stops = []
            state = check_once(run_dir,
                               stop_arm_fn=lambda rd, r: stops.append(r))
            self.assertEqual(state, "released")
            # the held worker was SIGCONTed back to life
            deadline = _time.monotonic() + 5
            while (_proc_state(held.pid) in ("T", "t")
                   and _time.monotonic() < deadline):
                _time.sleep(0.05)
            self.assertNotIn(_proc_state(held.pid), ("T", "t"))
            marker = json.loads(
                (run_dir / "MSA_SIDECAR_DIED.json").read_text())
            self.assertEqual(marker["holds_found"]["0"], [held.pid])
            # the whole arm was stopped (no uncontrolled training)
            self.assertEqual(len(stops), 1)


class TestMatchedCap(unittest.TestCase):
    """target[w] = baseline + 6 committed ticks (480M at the real 80M
    cadence); per-worker park; ARM_COMPLETE.json on 6/6; overshoot beyond
    one tick of slack fails the arm."""

    CAP = 480_000_000

    def test_all_six_parked_writes_arm_complete(self):
        from dexpbt_isaaclab.msa.continuation_guard import guard_loop
        with tempfile.TemporaryDirectory() as td:
            run_dir = Path(td)
            ws = _mk_workspace(
                run_dir, {w: 5_000_000_000 + self.CAP for w in range(6)})
            _write_marker_stub(run_dir)   # readiness completed earlier
            rc = guard_loop(run_dir, ws, _guard_config(), max_iterations=1)
            self.assertEqual(rc, 0)
            payload = json.loads(
                (run_dir / "ARM_COMPLETE.json").read_text())
            self.assertEqual(len(payload["workers"]), 6)
            for rec in payload["workers"]:
                self.assertEqual(rec["target_frame"],
                                 5_000_000_000 + self.CAP)
                self.assertEqual(rec["observed_frame"],
                                 rec["target_frame"])
            # normal completion: NEVER failure semantics
            self.assertFalse((run_dir / "GUARD_FAILED").exists())
            self.assertFalse((run_dir / "CAP_REACHED").exists())

    def test_overshoot_beyond_one_tick_fails_the_arm(self):
        from dexpbt_isaaclab.msa.continuation_guard import guard_loop
        with tempfile.TemporaryDirectory() as td:
            run_dir = Path(td)
            frames = {w: 5_000_000_000 + self.CAP for w in range(6)}
            # more than one full committed tick past target: silent overrun
            frames[3] = 5_000_000_000 + self.CAP + 80_000_001
            ws = _mk_workspace(run_dir, frames)
            _write_marker_stub(run_dir)   # readiness completed earlier
            rc = guard_loop(run_dir, ws, _guard_config(), max_iterations=1)
            self.assertEqual(rc, 2)
            self.assertTrue((run_dir / "GUARD_FAILED").exists())
            self.assertFalse((run_dir / "ARM_COMPLETE.json").exists())
            reason = json.loads(
                (run_dir / "GUARD_FAILED").read_text())["reason"]
            self.assertIn("worker 3", reason)
            self.assertIn("slack", reason)


class TestCommitTimeoutBounded(unittest.TestCase):
    """A durable-commit timeout is raised ONCE; crash cleanup restores the
    original-tau pin and releases holds WITHOUT a second bounded wait."""

    def test_commit_timeout_single_bounded_wait(self):
        from dexpbt_isaaclab.msa.controller import CommitTimeout
        world, ctl, views = dr.make_world(crash_on="commit_timeout")
        with self.assertRaises(CycleFailure) as cm:
            ctl.run_cycle(views)
        cause = cm.exception.__cause__
        self.assertIsInstance(cause, CommitTimeout)
        self.assertEqual(cause.max_wait_s, 1800.0)
        # ONE bounded commit attempt PER WORKER (the six waits run in
        # parallel), and the cleanup never re-waits: exactly 6 calls.
        self.assertEqual(world.commit_calls, 6)
        # the ack table (all false — nothing durably committed) rides on
        # the timeout for COMMIT_TIMEOUT.json
        self.assertEqual(getattr(cause, "commit_acks", None),
                         {str(w): False for w in world.taus})
        # holds cleared; every worker back on an original-tau PIN (the
        # durable revert is intentionally skipped after a timeout)
        self.assertEqual(world.held, set())
        for w, tau in world.taus.items():
            self.assertEqual(world.live_tolerance[w],
                             float(canonical_tau(tau)))
            self.assertEqual(world.pinned_tolerance[w],
                             float(canonical_tau(tau)))
        # cycle marked FAILED with the commit context preserved
        cycle = world.state_store[0]["msa_cycle"]
        self.assertEqual(cycle["status"], "FAILED")
        self.assertIn("CommitTimeout", cycle["error"])


class TestEvalScheduler(unittest.TestCase):
    """Bounded per-device concurrency: seeds round-robin over the eval
    devices, at most ONE process per device, fail-closed cancellation."""

    def test_round_robin_and_max_one_per_device(self):
        from dexpbt_isaaclab.msa.eval_bridge import run_seed_jobs
        lock = threading.Lock()
        active: dict[str, int] = {}
        max_active: dict[str, int] = {}
        ran: list[tuple[int, str]] = []

        def runner(i, cmd, dev):
            import time as _time
            with lock:
                active[dev] = active.get(dev, 0) + 1
                max_active[dev] = max(max_active.get(dev, 0), active[dev])
                ran.append((i, dev))
            _time.sleep(0.01)
            with lock:
                active[dev] -= 1
            return 0

        cmds = [["cmd", str(i)] for i in range(8)]
        run_seed_jobs(cmds, ["cuda:6", "cuda:7"], runner=runner)
        self.assertEqual(len(ran), 8)
        # never two Isaac processes on one GPU
        self.assertEqual(max_active, {"6": 1, "7": 1})
        # deterministic static round-robin: job i -> device idx[i % 2]
        for i, dev in ran:
            self.assertEqual(dev, "6" if i % 2 == 0 else "7")

    def test_failure_cancels_remaining_and_raises(self):
        from dexpbt_isaaclab.msa.eval_bridge import run_seed_jobs
        ran = []

        def runner(i, cmd, dev):
            ran.append(i)
            return 1 if i == 0 else 0     # first job on device 0 fails

        cmds = [["cmd", str(i)] for i in range(8)]
        with self.assertRaises(RuntimeError) as cm:
            run_seed_jobs(cmds, ["cuda:6", "cuda:7"], runner=runner)
        self.assertIn("fail closed", str(cm.exception))
        # jobs queued AFTER the failure on the same device never ran
        self.assertNotIn(2, ran)
        self.assertNotIn(4, ran)
        self.assertNotIn(6, ran)

    def test_device_validation(self):
        from dexpbt_isaaclab.msa.eval_bridge import (
            device_index,
            run_seed_jobs,
        )
        self.assertEqual(device_index("cuda:7"), "7")
        self.assertEqual(device_index("6"), "6")
        with self.assertRaises(ValueError):
            device_index("")
        with self.assertRaises(ValueError):
            run_seed_jobs([["c"]], [])
        with self.assertRaises(ValueError):
            run_seed_jobs([["c"]], ["cuda:6", "6"])   # duplicate GPU


class TestEvalCostEstimate(unittest.TestCase):
    """True per-cycle workload: 27 eval cells (PRE 12 + MID 6 + POST 9)."""

    def test_27_cells_and_wall_times(self):
        from dexpbt_isaaclab.msa.eval_bridge import (
            EVAL_CELLS_PER_CYCLE,
            eval_cost_estimate,
        )
        self.assertEqual(EVAL_CELLS_PER_CYCLE,
                         {"pre": 12, "mid": 6, "post": 9, "total": 27})
        smoke = eval_cost_estimate(4, 2)
        self.assertEqual(smoke["estimated_process_launches"], 108)
        self.assertEqual(smoke["estimated_wall_time_hours"], 5.4)
        pilot = eval_cost_estimate(16, 2)
        self.assertEqual(pilot["estimated_process_launches"], 432)
        self.assertEqual(pilot["estimated_wall_time_hours"], 21.6)
        formal = eval_cost_estimate(256, 2)
        self.assertEqual(formal["estimated_process_launches"], 6912)
        self.assertEqual(formal["estimated_wall_time_hours"], 345.6)

    def test_cost_estimate_file_generated(self):
        if not LAUNCHER_DIR.exists():
            self.skipTest("arm launchers not generated in this checkout")
        payload = json.loads(
            (LAUNCHER_DIR / "EVAL_COST_ESTIMATE.json").read_text())
        seeds = {e["seeds"]: e for e in payload["estimates"]}
        self.assertEqual(set(seeds), {4, 16, 256})
        for e in payload["estimates"]:
            self.assertEqual(e["eval_cells"]["total"], 27)
            self.assertEqual(e["devices"], 2)


class TestHoldSmoke(unittest.TestCase):
    """--hold-smoke-only: readiness-gated single-worker hold validation
    with guaranteed release and HOLD_SMOKE_RESULT.json."""

    def test_hold_smoke_end_to_end(self):
        import subprocess
        import time as _time

        from dexpbt_isaaclab.msa.sidecar import (
            _proc_state,
            hold_smoke,
            read_holds_file,
        )
        with tempfile.TemporaryDirectory() as td:
            run_dir = Path(td)
            ws = _mk_workspace(run_dir, {0: 5_020_000_000})
            proc = subprocess.Popen(["sleep", "60"])
            self.addCleanup(proc.kill)
            d = run_dir / "policy_00"
            d.mkdir()
            (d / "pid.txt").write_text(str(proc.pid))
            marker = run_dir / "FIRST_TICK_VALIDATED.json"
            marker.write_text(json.dumps({
                "workers": [{"worker_id": w} for w in range(6)]}))

            # a "trainer" that commits the next tick shortly after release
            def _later_tick():
                _time.sleep(0.8)
                (ws / "000" / "000162.pth").write_bytes(b"ckpt")
                (ws / "000" / "000162.json").write_text(json.dumps({
                    "frame": 5_040_000_000, "lineage_id": "lin0",
                    "curriculum": {"success_tolerance": 0.023}}))

            t = threading.Thread(target=_later_tick, daemon=True)
            t.start()
            rc = hold_smoke(run_dir, ws, 0, hold_seconds=0.3,
                            readiness_marker=marker,
                            resume_timeout_s=10.0, poll_s=0.1)
            t.join()
            self.assertEqual(rc, 0)
            result = json.loads(
                (run_dir / "HOLD_SMOKE_RESULT.json").read_text())
            self.assertTrue(result["ok"])
            self.assertEqual(result["frame_before_hold"], 5_020_000_000)
            self.assertEqual(result["frame_after_hold_window"],
                             5_020_000_000)
            self.assertEqual(result["next_committed_frame"], 5_040_000_000)
            # release guaranteed: process running, no residual holds
            self.assertNotIn(_proc_state(proc.pid), ("T", "t"))
            self.assertEqual(read_holds_file(run_dir), {})
            self.assertEqual(result["residual_holds"], {})
            # gate marker: PASSED written atomically, FAILED absent
            passed = json.loads(
                (run_dir / "HOLD_SMOKE_PASSED.json").read_text())
            self.assertEqual(passed["worker_id"], 0)
            self.assertEqual(passed["next_committed_frame"], 5_040_000_000)
            self.assertFalse((run_dir / "HOLD_SMOKE_FAILED.json").exists())
            self.assertEqual(list(run_dir.glob("*.tmp.*")), [])

    def test_hold_smoke_failure_stops_arm_never_writes_passed(self):
        import subprocess
        import time

        from dexpbt_isaaclab.msa.sidecar import _proc_state, hold_smoke
        with tempfile.TemporaryDirectory() as td:
            run_dir = Path(td)
            # workspace has NO committed tick for worker 0 -> smoke fails
            ws = run_dir / "dexpbt_8gpu_ws"
            (ws / "000").mkdir(parents=True)
            marker = run_dir / "FIRST_TICK_VALIDATED.json"
            marker.write_text(json.dumps({
                "workers": [{"worker_id": w} for w in range(6)]}))
            # a live "formal sidecar" waiting at the gate: the fail-closed
            # stop must terminate it
            sidecar_proc = subprocess.Popen(["sleep", "60"])
            self.addCleanup(sidecar_proc.kill)
            (run_dir / "msa_sidecar.pid").write_text(str(sidecar_proc.pid))
            rc = hold_smoke(run_dir, ws, 0, hold_seconds=0.1,
                            readiness_marker=marker,
                            resume_timeout_s=1.0, poll_s=0.05)
            self.assertEqual(rc, 2)
            # PASSED is NEVER written on a failure path
            self.assertFalse((run_dir / "HOLD_SMOKE_PASSED.json").exists())
            failed = json.loads(
                (run_dir / "HOLD_SMOKE_FAILED.json").read_text())
            self.assertFalse(failed["ok"])
            self.assertIn("no committed tick", failed["error"])
            self.assertEqual(failed["residual_holds"], {})
            # the arm was fail-closed stopped (waiting sidecar SIGTERMed)
            deadline = time.monotonic() + 5
            while (_proc_state(sidecar_proc.pid) not in (None, "Z")
                   and time.monotonic() < deadline):
                time.sleep(0.05)
            self.assertIn(_proc_state(sidecar_proc.pid), (None, "Z"))


class TestHoldSmokeGate(unittest.TestCase):
    """The formal sidecar's cycle gate requires BOTH the 6/6 readiness
    marker AND HOLD_SMOKE_PASSED.json before any cycle starts/resumes."""

    def _views(self, n=6):
        from dexpbt_isaaclab.msa.sidecar import WorkerView
        return [WorkerView(worker_id=w, lineage_id=f"l{w}",
                           active_tau=0.023, frame=5_020_000_000,
                           source_checkpoint_id="c") for w in range(n)]

    def test_no_smoke_marker_gate_closed(self):
        from dexpbt_isaaclab.msa.continuation_guard import (
            readiness_entry,
            write_readiness_marker,
        )
        from dexpbt_isaaclab.msa.sidecar import cycle_gate_ok
        with tempfile.TemporaryDirectory() as td:
            readiness = Path(td) / "FIRST_TICK_VALIDATED.json"
            smoke = Path(td) / "HOLD_SMOKE_PASSED.json"
            entry = {"path": "p", "sha256_head": "h"}
            meta = {"frame": 5_020_000_000, "lineage_id": "l",
                    "curriculum": {"success_tolerance": 0.023}}
            write_readiness_marker(
                Path(td), {w: readiness_entry(w, entry, meta)
                           for w in range(6)})
            # readiness satisfied, hold smoke NOT run -> gate stays closed
            ok, why = cycle_gate_ok(readiness, smoke, self._views(6))
            self.assertFalse(ok)
            self.assertIn("hold smoke gate", why)
            # smoke PASSED -> gate opens
            smoke.write_text(json.dumps({"worker_id": 0}))
            ok, why = cycle_gate_ok(readiness, smoke, self._views(6))
            self.assertTrue(ok, why)

    def test_smoke_marker_alone_not_sufficient(self):
        from dexpbt_isaaclab.msa.sidecar import cycle_gate_ok
        with tempfile.TemporaryDirectory() as td:
            readiness = Path(td) / "FIRST_TICK_VALIDATED.json"
            smoke = Path(td) / "HOLD_SMOKE_PASSED.json"
            smoke.write_text(json.dumps({"worker_id": 0}))
            ok, why = cycle_gate_ok(readiness, smoke, self._views(6))
            self.assertFalse(ok)
            self.assertIn("readiness marker missing", why)

    def test_smoke_failed_is_a_sidecar_stop_marker(self):
        from dexpbt_isaaclab.msa.sidecar import _stop_markers_present
        with tempfile.TemporaryDirectory() as td:
            self.assertIsNone(_stop_markers_present(Path(td)))
            (Path(td) / "HOLD_SMOKE_FAILED.json").write_text("{}")
            self.assertEqual(_stop_markers_present(Path(td)),
                             "HOLD_SMOKE_FAILED.json")


class TestCommittingPhase(unittest.TestCase):
    """Durable commits happen in an explicit persisted COMMITTING state:
    DONE/NORMAL only after ALL SIX workers acked; a crash mid-COMMITTING
    resumes with commits ONLY (no re-eval, no re-training)."""

    def test_committing_persisted_before_any_commit(self):
        """Crash after DECIDE, before the FIRST durable commit: restart
        completes only the commits."""
        world, ctl, views = dr.make_world()
        world.crash_hard = True
        world.crash_commit_workers = set(range(6))   # every commit dies
        with self.assertRaises(dr.HardKill):
            ctl.run_cycle(views)
        # persisted state: COMMITTING with the full decision context and
        # an all-false ack table; NOT DONE, machine NOT back to NORMAL
        cycle = world.state_store[0]["msa_cycle"]
        self.assertEqual(cycle["status"], "COMMITTING")
        self.assertIn(cycle["decision"], ("promote", "stay"))
        self.assertEqual(len(cycle["final_tol_by_worker"]), 6)
        self.assertEqual(cycle["commit_acks_by_worker"],
                         {str(w): False for w in range(6)})
        self.assertNotEqual(
            world.state_store[0]["msa_state_machine"]["phase"], "NORMAL")
        self.assertEqual(world.commit_log, [])
        # busy marker survives the hard kill (watchdog/guard signal)
        self.assertIsNotNone(world.busy)
        evals_before = len(world.eval_calls)
        trained_before = dict(world.trained_frames)

        # restart: un-crash the world, resume via pending_cycle
        world.crash_hard = False
        world.crash_commit_workers = set()
        world2, ctl2, views2 = dr.make_world(world=world)
        pending = ctl2.pending_cycle(list(range(6)))
        self.assertIsNotNone(pending)
        self.assertEqual(pending["status"], "COMMITTING")
        result = ctl2.run_cycle(views2, resume_cycle=pending)
        self.assertTrue(result.resumed)
        # commits ONLY: no re-eval, no re-training
        self.assertEqual(len(world.eval_calls), evals_before)
        self.assertEqual(world.trained_frames, trained_before)
        self.assertEqual(sorted(w for w, _ in world.commit_log),
                         list(range(6)))
        cycle = world.state_store[0]["msa_cycle"]
        self.assertEqual(cycle["status"], "DONE")
        self.assertEqual(cycle["commit_acks_by_worker"],
                         {str(w): True for w in range(6)})
        self.assertEqual(ctl2.machine.phase, MSAPhase.NORMAL)
        self.assertIsNone(world.busy)
        # committed tolerances match the persisted final_tol_by_worker
        for w in range(6):
            self.assertEqual(world.live_tolerance[w],
                             float(cycle["final_tol_by_worker"][str(w)]))
            self.assertEqual(world.pinned_tolerance[w], 0.0)

    def test_partial_acks_resume_commits_only_unacked(self):
        """Crash after SOME workers acked: restart commits ONLY the
        un-acked workers (idempotent for the rest)."""
        world, ctl, views = dr.make_world()
        world.crash_hard = True
        world.crash_commit_workers = {4, 5}
        with self.assertRaises(dr.HardKill):
            ctl.run_cycle(views)
        cycle = world.state_store[0]["msa_cycle"]
        self.assertEqual(cycle["status"], "COMMITTING")
        acks = cycle["commit_acks_by_worker"]
        self.assertEqual({w for w in range(6) if acks[str(w)]},
                         {0, 1, 2, 3})
        self.assertEqual(sorted(w for w, _ in world.commit_log),
                         [0, 1, 2, 3])
        calls_before = world.commit_calls

        world.crash_hard = False
        world.crash_commit_workers = set()
        world2, ctl2, views2 = dr.make_world(world=world)
        pending = ctl2.pending_cycle(list(range(6)))
        self.assertEqual(pending["status"], "COMMITTING")
        result = ctl2.run_cycle(views2, resume_cycle=pending)
        self.assertTrue(result.resumed)
        # exactly TWO new commit attempts: workers 4 and 5 only
        self.assertEqual(world.commit_calls - calls_before, 2)
        self.assertEqual(sorted(w for w, _ in world.commit_log),
                         list(range(6)))
        cycle = world.state_store[0]["msa_cycle"]
        self.assertEqual(cycle["status"], "DONE")
        self.assertEqual(cycle["commit_acks_by_worker"],
                         {str(w): True for w in range(6)})
        self.assertIsNone(world.busy)
        for w in range(6):
            self.assertEqual(world.pinned_tolerance[w], 0.0)

    def test_done_never_written_before_all_acks(self):
        """The success path itself: every persisted DONE snapshot must
        carry a full-true ack table (no DONE-before-durable-commit)."""
        world, ctl, views = dr.make_world()
        result = ctl.run_cycle(views)
        self.assertTrue(result.promoted or result.first_blocking_gate)
        for w in range(6):
            cycle = world.state_store[w]["msa_cycle"]
            self.assertEqual(cycle["status"], "DONE")
            self.assertEqual(cycle["commit_acks_by_worker"],
                             {str(x): True for x in range(6)})


class TestBusyMarkerLifecycle(unittest.TestCase):
    """MSA_BUSY.json (in-memory mirror in FakeWorld): set BEFORE the first
    hold, kept through eval/probe/DECIDE/COMMITTING, cleared only at DONE
    or after failure cleanup; a hard kill leaves it in place."""

    def test_busy_set_before_first_hold_and_cleared_at_done(self):
        world, ctl, views = dr.make_world()
        orig_hold = world.hold_workers
        busy_at_hold = []

        def _hold(ids):
            busy_at_hold.append(world.busy is not None)
            orig_hold(ids)

        ctl._hold = _hold
        ctl.run_cycle(views)
        # every hold (PRE/MID/POST barriers) happened under a busy marker
        self.assertTrue(busy_at_hold)
        self.assertTrue(all(busy_at_hold))
        # cleared only after full DONE
        self.assertIsNone(world.busy)
        # the marker carried the phase progression, incl. COMMITTING
        phases = [b["phase"] for b in world.busy_log if b]
        self.assertIn("PRE_EVAL", phases)
        statuses = [b["status"] for b in world.busy_log if b]
        self.assertIn("COMMITTING", statuses)
        self.assertEqual(world.busy_log[-1], None)

    def test_soft_failure_cleanup_clears_busy(self):
        world, ctl, views = dr.make_world(crash_on="mid")
        with self.assertRaises(CycleFailure):
            ctl.run_cycle(views)
        self.assertIsNone(world.busy)     # cleanup completed -> cleared

    def test_hard_kill_leaves_busy_for_watchdog(self):
        world, ctl, views = dr.make_world(crash_on="mid")
        world.crash_hard = True
        with self.assertRaises(dr.HardKill):
            ctl.run_cycle(views)
        self.assertIsNotNone(world.busy)  # watchdog/guard must see it


class TestGuardMSACoordination(unittest.TestCase):
    """While MSA_BUSY.json exists the guard defers park/cap (cap_pending):
    it must not stop workers or SIGCONT holds mid-cycle; after the busy
    marker clears it re-reads committed metas and parks normally."""

    def _stopped_sleeper(self, run_dir: Path, worker_id: int):
        import os as _os
        import signal as sg
        import subprocess
        import time as _time

        from dexpbt_isaaclab.msa.sidecar import _proc_state
        proc = subprocess.Popen(["sleep", "60"])
        self.addCleanup(proc.kill)
        _os.kill(proc.pid, sg.SIGSTOP)
        deadline = _time.monotonic() + 5
        while (_proc_state(proc.pid) not in ("T", "t")
               and _time.monotonic() < deadline):
            _time.sleep(0.05)
        d = run_dir / f"policy_{worker_id:02d}"
        d.mkdir(parents=True, exist_ok=True)
        (d / "pid.txt").write_text(str(proc.pid))
        return proc

    def test_cap_during_hold_defers_and_touches_nobody(self):
        from dexpbt_isaaclab.msa.continuation_guard import guard_loop
        from dexpbt_isaaclab.msa.sidecar import _proc_state
        with tempfile.TemporaryDirectory() as td:
            run_dir = Path(td)
            # all six at their cap target, one even past the slack —
            # but an MSA cycle is in flight (busy marker + held worker)
            frames = {w: 5_000_000_000 + 480_000_000 for w in range(6)}
            frames[3] = 5_000_000_000 + 480_000_000 + 80_000_001
            ws = _mk_workspace(run_dir, frames)
            _write_marker_stub(run_dir)   # readiness completed earlier
            (run_dir / "MSA_BUSY.json").write_text(json.dumps(
                {"intervention_id": "msa_x", "phase": "MID_EVAL",
                 "due_workers": list(range(6))}))
            held = self._stopped_sleeper(run_dir, 0)
            (run_dir / "MSA_HOLDS.json").write_text(
                json.dumps({"0": [held.pid]}))
            rc = guard_loop(run_dir, ws, _guard_config(), max_iterations=1)
            self.assertEqual(rc, 0)
            # cap deferred: no completion, no failure, worker STILL held
            self.assertFalse((run_dir / "ARM_COMPLETE.json").exists())
            self.assertFalse((run_dir / "GUARD_FAILED").exists())
            self.assertIn(_proc_state(held.pid), ("T", "t"))

    def test_cap_waits_for_busy_clear_then_parks(self):
        from dexpbt_isaaclab.msa.continuation_guard import guard_loop
        with tempfile.TemporaryDirectory() as td:
            run_dir = Path(td)
            frames = {w: 5_000_000_000 + 480_000_000 for w in range(6)}
            ws = _mk_workspace(run_dir, frames)
            _write_marker_stub(run_dir)   # readiness completed earlier
            busy = run_dir / "MSA_BUSY.json"
            busy.write_text(json.dumps({"phase": "PROMOTE",
                                        "status": "COMMITTING"}))
            # COMMITTING in flight: nothing parks
            rc = guard_loop(run_dir, ws, _guard_config(), max_iterations=1)
            self.assertEqual(rc, 0)
            self.assertFalse((run_dir / "ARM_COMPLETE.json").exists())
            # sidecar finished the cycle and cleared the marker: the guard
            # re-reads committed metas and applies the normal park/cap
            busy.unlink()
            rc = guard_loop(run_dir, ws, _guard_config(), max_iterations=1)
            self.assertEqual(rc, 0)
            payload = json.loads(
                (run_dir / "ARM_COMPLETE.json").read_text())
            self.assertEqual(len(payload["workers"]), 6)

    def test_stop_worker_sigconts_only_target(self):
        from dexpbt_isaaclab.msa.continuation_guard import stop_worker
        from dexpbt_isaaclab.msa.sidecar import _proc_state
        with tempfile.TemporaryDirectory() as td:
            run_dir = Path(td)
            w0 = self._stopped_sleeper(run_dir, 0)
            w1 = self._stopped_sleeper(run_dir, 1)
            (run_dir / "MSA_HOLDS.json").write_text(json.dumps(
                {"0": [w0.pid], "1": [w1.pid]}))
            stop_worker(run_dir, 0)
            import time as _time

            deadline = _time.monotonic() + 5
            while (_proc_state(w0.pid) not in (None, "Z")
                   and _time.monotonic() < deadline):
                _time.sleep(0.05)
            # target worker was SIGCONTed then terminated ...
            self.assertIn(_proc_state(w0.pid), (None, "Z"))
            # ... while worker 1's held process was NEVER SIGCONTed
            self.assertIn(_proc_state(w1.pid), ("T", "t"))


class TestSidecarHoldAdoption(unittest.TestCase):
    """MSA arms: the guard leaves READINESS_HOLDS.json behind (workers
    frozen at their validated boundaries); the sidecar ADOPTS them —
    verifying stopped state + committed frame == validated frame — and
    atomically converts the record to MSA_HOLDS.json. Any failure releases
    everything and stops the arm."""

    FRAME = 5_080_000_000

    def _handoff_run(self, td: Path):
        """A run where the guard already completed the MSA handoff: six
        frozen sleepers, READINESS_HOLDS.json + FIRST_TICK_VALIDATED.json
        at the same validated frames."""
        from dexpbt_isaaclab.msa.continuation_guard import guard_loop
        run_dir = Path(td)
        ws = _mk_workspace(run_dir, {w: self.FRAME for w in range(6)})
        procs = _spawn_worker_procs(self, run_dir)
        config = dict(_guard_config())
        config["arm"] = "MSA_STATEFUL"
        rc = guard_loop(run_dir, ws, config, max_iterations=1)
        self.assertEqual(rc, 0)
        return run_dir, ws, procs

    def test_adopt_seamless(self):
        # NEW TEST 4: the sidecar adopts the guard's readiness holds
        # without ever releasing a worker.
        from dexpbt_isaaclab.msa.sidecar import (
            READINESS_HOLDS_FILE,
            WorkerHolds,
            _proc_state,
            adopt_readiness_holds,
            read_holds_file,
            read_readiness_holds,
        )
        with tempfile.TemporaryDirectory() as td:
            run_dir, ws, procs = self._handoff_run(Path(td))
            before = read_readiness_holds(run_dir)
            holds = WorkerHolds(run_dir)
            adopted = adopt_readiness_holds(
                holds, run_dir, ws,
                run_dir / "FIRST_TICK_VALIDATED.json")
            self.assertEqual(adopted, list(range(6)))
            # converted: MSA_HOLDS.json carries the SAME pids, the guard's
            # readiness record is gone, and nobody was ever released
            self.assertFalse((run_dir / READINESS_HOLDS_FILE).exists())
            msa_holds = read_holds_file(run_dir)
            for w in range(6):
                self.assertEqual(msa_holds[w],
                                 [int(p) for p in before[w]["pids"]])
                self.assertIn(_proc_state(procs[w].pid), ("T", "t"))
            # WorkerHolds internal state imported: release works normally
            holds.release_all()
            for w in range(6):
                self.assertNotIn(_proc_state(procs[w].pid), ("T", "t"))

    def test_pre_snapshot_frame_equals_validated_frame(self):
        # NEW TEST 5: adoption verifies the newest COMMITTED frame equals
        # the validated frame in FIRST_TICK_VALIDATED.json — the frozen
        # boundary the first PRE snapshot must come from. A mismatch (the
        # worker somehow committed past its validated tick) fails closed.
        from dexpbt_isaaclab.msa.sidecar import (
            WorkerHolds,
            _proc_state,
            adopt_readiness_holds,
        )
        with tempfile.TemporaryDirectory() as td:
            run_dir, ws, procs = self._handoff_run(Path(td))
            # worker 3 commits ONE more tick past its validated boundary
            d = ws / "003"
            (d / "000162.pth").write_bytes(b"ckpt")
            (d / "000162.json").write_text(json.dumps({
                "frame": self.FRAME + 80_000_000, "lineage_id": "lin3",
                "curriculum": {"success_tolerance": 0.023}}))
            holds = WorkerHolds(run_dir)
            with self.assertRaises(RuntimeError) as cm:
                adopt_readiness_holds(
                    holds, run_dir, ws,
                    run_dir / "FIRST_TICK_VALIDATED.json")
            self.assertIn("validated boundary", str(cm.exception))
            self.assertIn("PRE snapshot", str(cm.exception))
            # fail closed: EVERYONE released (never left frozen by a
            # failed handoff)
            for w in range(6):
                self.assertNotIn(_proc_state(procs[w].pid), ("T", "t"))

    def test_adopt_failure_releases_all_and_stops_arm(self):
        # NEW TEST 6: a readiness-held PID that died (or resumed) fails
        # the adoption; everything is released; the sidecar's caller then
        # stops the whole arm (GUARD_FAILED path in main()).
        from dexpbt_isaaclab.msa.sidecar import (
            READINESS_HOLDS_FILE,
            WorkerHolds,
            _proc_state,
            adopt_readiness_holds,
            read_holds_file,
        )
        with tempfile.TemporaryDirectory() as td:
            run_dir, ws, procs = self._handoff_run(Path(td))
            procs[2].kill()               # held worker 2 died under hold
            procs[2].wait(timeout=5)
            holds = WorkerHolds(run_dir)
            with self.assertRaises(RuntimeError) as cm:
                adopt_readiness_holds(
                    holds, run_dir, ws,
                    run_dir / "FIRST_TICK_VALIDATED.json")
            self.assertIn("no longer in the stopped state",
                          str(cm.exception))
            # nothing adopted, nobody left frozen
            self.assertEqual(read_holds_file(run_dir), {})
            self.assertFalse((run_dir / READINESS_HOLDS_FILE).exists())
            for w in range(6):
                if w == 2:
                    continue
                self.assertNotIn(_proc_state(procs[w].pid), ("T", "t"))


class TestTerminateTreeOrder(unittest.TestCase):
    """Children-before-root termination with escalation, robust against a
    supervisor that respawns its children."""

    def test_kill_order_children_before_root(self):
        # NEW TEST 7a: the signalling order is deepest descendants FIRST,
        # supervisor/root LAST — for SIGTERM and SIGKILL alike.
        import subprocess
        import time as _time

        from dexpbt_isaaclab.msa.continuation_guard import _kill_order
        sup = subprocess.Popen(["bash", "-c", "sleep 300 & wait"])
        self.addCleanup(_reap, sup)
        deadline = _time.monotonic() + 5
        order = _kill_order(sup.pid)
        while len(order) < 2 and _time.monotonic() < deadline:
            _time.sleep(0.05)
            order = _kill_order(sup.pid)
        self.assertGreaterEqual(len(order), 2)
        self.assertEqual(order[-1], sup.pid)      # root LAST
        for child in order[:-1]:                  # children strictly first
            self.assertNotEqual(child, sup.pid)

    def test_supervisor_respawning_child_is_fully_stopped(self):
        # NEW TEST 7b: a supervisor that ignores SIGTERM and RESPAWNS its
        # child after every kill — the escalation must re-enumerate,
        # SIGKILL children-first/root-last and verify NO survivor remains
        # (including a child respawned mid-escalation).
        import subprocess
        import time as _time

        from dexpbt_isaaclab.msa.continuation_guard import (
            _alive,
            _terminate_tree,
        )
        sup = subprocess.Popen(
            ["bash", "-c", 'trap "" TERM; while :; do sleep 300; done'])
        self.addCleanup(_reap, sup)
        deadline = _time.monotonic() + 5
        from dexpbt_isaaclab.msa.continuation_guard import _kill_order
        while len(_kill_order(sup.pid)) < 2 \
                and _time.monotonic() < deadline:
            _time.sleep(0.05)
        seen = set(_kill_order(sup.pid))
        left = _terminate_tree(sup.pid, grace_s=1.0, verify_s=8.0)
        self.assertEqual(left, [])                # verified: no survivor
        sup.wait(timeout=5)                       # reap the zombie root
        for pid in seen:
            self.assertFalse(_alive(pid),
                             f"pid {pid} survived the escalation")


class TestHoldSmokeCertificate(unittest.TestCase):
    """The formal MSA sidecar requires the DISPOSABLE hold-smoke
    certificate and validates commit/task/profile/GPU env at startup."""

    CERT = {
        "ok": True,
        "eval_coexistence_ok": True,
        "eval_device": "cuda:4",
        "eval_num_envs": 512,
        "eval_exit_code": 0,
        "git_commit": "abc123",
        "task": "DexPBT-Regrasping-KukaAllegro-Direct-v0",
        "wm_profile": "cser_dexpbt_conservative",
        "gpu": {"index": "4", "name": "NVIDIA vGPU-32GB",
                "uuid": "GPU-x", "driver_version": "550.54.15",
                "cuda_version": "12.4"},
        "worker_tree": [{"pid": 1, "comm": "bash"}],
        "frame_before_hold": 5_000_000_000,
        "next_committed_frame": 5_080_000_000,
        "timestamp": 0.0,
    }
    ENV = dict(git_commit="abc123",
               task="DexPBT-Regrasping-KukaAllegro-Direct-v0",
               profile="cser_dexpbt_conservative",
               gpu_names={"4": "NVIDIA vGPU-32GB", "5": "NVIDIA vGPU-32GB"},
               driver_version="550.54.15")

    def test_matching_certificate_passes(self):
        from dexpbt_isaaclab.msa.sidecar import certificate_ok
        ok, why = certificate_ok(dict(self.CERT), **self.ENV)
        self.assertTrue(ok, why)

    def test_mismatches_fail_closed(self):
        from dexpbt_isaaclab.msa.sidecar import certificate_ok
        cases = [
            ({"ok": False}, "ok != true"),
            ({"eval_coexistence_ok": False}, "eval_coexistence_ok"),
            ({"git_commit": "OTHER"}, "git commit"),
            ({"task": "OtherTask-v0"}, "task"),
            ({"wm_profile": "some_other_profile"}, "profile"),
            ({"gpu": {**self.CERT["gpu"], "name": "NVIDIA A100"}},
             "model"),
            ({"gpu": {**self.CERT["gpu"], "driver_version": "999.99"}},
             "driver"),
        ]
        for patch, needle in cases:
            cert = dict(self.CERT)
            cert.update(patch)
            ok, why = certificate_ok(cert, **self.ENV)
            self.assertFalse(ok, f"expected failure for {patch}")
            self.assertIn(needle, why)

    def test_certificate_without_coexistence_proof_rejected(self):
        # P1: an OLD certificate (produced before the shared-GPU eval
        # coexistence smoke existed) has no eval_coexistence_ok field at
        # all — the formal sidecar must reject it.
        from dexpbt_isaaclab.msa.sidecar import certificate_ok
        cert = dict(self.CERT)
        del cert["eval_coexistence_ok"]
        ok, why = certificate_ok(cert, **self.ENV)
        self.assertFalse(ok)
        self.assertIn("eval_coexistence_ok", why)

    def test_eval_gpu_model_mismatch_fails(self):
        # certificate was produced on GPU 4; if ANY current eval device
        # has a different model the gate must fail closed.
        from dexpbt_isaaclab.msa.sidecar import certificate_ok
        env = dict(self.ENV)
        env["gpu_names"] = {"4": "NVIDIA vGPU-32GB", "5": "NVIDIA T4"}
        ok, why = certificate_ok(dict(self.CERT), **env)
        self.assertFalse(ok)
        self.assertIn("eval GPU 5", why)


class TestWatchdogBusyNoHolds(unittest.TestCase):
    """Sidecar death with busy=true but EMPTY holds (e.g. mid-COMMITTING,
    all workers running) must still write MSA_SIDECAR_DIED.json and stop
    the arm — never a quiet idle_exit."""

    def test_dead_sidecar_busy_no_holds_stops_arm(self):
        import subprocess

        from dexpbt_isaaclab.msa.hold_watchdog import check_once
        with tempfile.TemporaryDirectory() as td:
            run_dir = Path(td)
            guard = _spawn_running(self)              # guard still alive
            (run_dir / "continuation_guard.pid").write_text(str(guard.pid))
            dead = subprocess.Popen(["true"])
            dead.wait()
            (run_dir / "msa_sidecar.pid").write_text(str(dead.pid))
            (run_dir / "MSA_BUSY.json").write_text(json.dumps(
                {"intervention_id": "msa_x", "phase": "PROMOTE",
                 "status": "COMMITTING", "due_workers": list(range(6))}))
            stops = []
            state = check_once(run_dir,
                               stop_arm_fn=lambda rd, r: stops.append(r))
            self.assertEqual(state, "released")
            marker = json.loads(
                (run_dir / "MSA_SIDECAR_DIED.json").read_text())
            self.assertEqual(marker["holds_found"], {})
            self.assertEqual(marker["busy_marker"]["status"], "COMMITTING")
            self.assertEqual(len(stops), 1)

    def test_dead_sidecar_no_busy_no_holds_never_stops_arm(self):
        # nothing frozen and no busy marker: never a stop — with a live
        # guard the watchdog keeps watching; after ARM_COMPLETE it idles.
        from dexpbt_isaaclab.msa.hold_watchdog import check_once
        with tempfile.TemporaryDirectory() as td:
            run_dir = Path(td)
            guard = _spawn_running(self)
            (run_dir / "continuation_guard.pid").write_text(str(guard.pid))
            (run_dir / "msa_sidecar.pid").write_text(str(_dead_pid()))
            calls = []
            self.assertEqual(
                check_once(run_dir, stop_arm_fn=lambda rd, r:
                           calls.append(r)), "ok")
            self.assertEqual(calls, [])
            (run_dir / "ARM_COMPLETE.json").write_text("{}")
            self.assertEqual(
                check_once(run_dir, stop_arm_fn=lambda rd, r:
                           calls.append(r)), "idle_exit")
            self.assertEqual(calls, [])


class TestStopArmCallerSafe(unittest.TestCase):
    """P0-1: stop_arm is caller-safe — invoked from the sidecar / watchdog
    (whose pid IS a control pid), it stops the workers and the OTHER
    control process first and NEVER kills the caller before returning."""

    def _assert_workers_dead(self, procs):
        from dexpbt_isaaclab.msa.sidecar import _proc_state
        for w, p in procs.items():
            self.assertIn(_proc_state(p.pid), (None, "Z"),
                          f"worker {w} not terminated")

    def _assert_gone(self, pid):
        from dexpbt_isaaclab.msa.sidecar import _proc_state
        deadline = __import__("time").monotonic() + 5
        while (_proc_state(pid) not in (None, "Z")
               and __import__("time").monotonic() < deadline):
            __import__("time").sleep(0.05)
        self.assertIn(_proc_state(pid), (None, "Z"))

    def test_caller_is_sidecar_kills_workers_and_all_siblings(self):
        # TEST A: caller registers itself as msa_sidecar.pid; the sibling
        # hold_watchdog AND continuation_guard MUST both be stopped.
        with tempfile.TemporaryDirectory() as td:
            run_dir = Path(td)
            procs = _spawn_worker_procs(self, run_dir)
            watchdog = _spawn_running(self)           # the watchdog sibling
            (run_dir / "hold_watchdog.pid").write_text(str(watchdog.pid))
            guard = _spawn_running(self)              # the guard sibling
            (run_dir / "continuation_guard.pid").write_text(str(guard.pid))
            res = _run_caller(run_dir, "msa_sidecar.pid", "sidecar")
            self.assertEqual(res.returncode, 0, res.stderr)
            # the caller ran PAST stop_arm (it was never killed)
            self.assertTrue((run_dir / "CALLER_DONE.json").exists())
            self.assertTrue((run_dir / "GUARD_FAILED").exists())
            self._assert_workers_dead(procs)
            # every sibling control process was terminated
            self._assert_gone(watchdog.pid)
            self._assert_gone(guard.pid)
            # stale ownership cleared
            self.assertFalse((run_dir / "MSA_HOLDS.json").exists())
            self.assertFalse((run_dir / "READINESS_HOLDS.json").exists())

    def test_caller_is_watchdog_kills_workers_and_all_siblings(self):
        # TEST B: caller registers itself as hold_watchdog.pid; the
        # sibling msa_sidecar AND continuation_guard MUST be stopped.
        with tempfile.TemporaryDirectory() as td:
            run_dir = Path(td)
            procs = _spawn_worker_procs(self, run_dir)
            sidecar = _spawn_running(self)            # the sidecar sibling
            (run_dir / "msa_sidecar.pid").write_text(str(sidecar.pid))
            guard = _spawn_running(self)              # the guard sibling
            (run_dir / "continuation_guard.pid").write_text(str(guard.pid))
            res = _run_caller(run_dir, "hold_watchdog.pid", "guard")
            self.assertEqual(res.returncode, 0, res.stderr)
            self.assertTrue((run_dir / "CALLER_DONE.json").exists())
            self._assert_workers_dead(procs)
            self._assert_gone(sidecar.pid)
            self._assert_gone(guard.pid)

    def test_caller_is_guard_kills_workers_and_all_siblings(self):
        # TEST C (new sibling set): caller registers itself as
        # continuation_guard.pid — stop_arm must skip the caller and stop
        # the sidecar and watchdog siblings.
        with tempfile.TemporaryDirectory() as td:
            run_dir = Path(td)
            procs = _spawn_worker_procs(self, run_dir)
            sidecar = _spawn_running(self)
            (run_dir / "msa_sidecar.pid").write_text(str(sidecar.pid))
            watchdog = _spawn_running(self)
            (run_dir / "hold_watchdog.pid").write_text(str(watchdog.pid))
            res = _run_caller(run_dir, "continuation_guard.pid", "guard")
            self.assertEqual(res.returncode, 0, res.stderr)
            self.assertTrue((run_dir / "CALLER_DONE.json").exists())
            self._assert_workers_dead(procs)
            self._assert_gone(sidecar.pid)
            self._assert_gone(watchdog.pid)

    def test_watchdog_check_once_stop_is_caller_safe(self):
        # TEST D / P0-2 #6: the REAL watchdog path (check_once -> default
        # stop_arm) invoked from a process registered as hold_watchdog.pid
        # tears the arm down (including the guard sibling) and still
        # reaches its completion marker.
        with tempfile.TemporaryDirectory() as td:
            run_dir = Path(td)
            procs = _spawn_worker_procs(self, run_dir)
            guard = _spawn_running(self)              # guard alive
            (run_dir / "continuation_guard.pid").write_text(str(guard.pid))
            (run_dir / "msa_sidecar.pid").write_text(str(_dead_pid()))
            (run_dir / "MSA_HOLDS.json").write_text(
                json.dumps({"0": [procs[0].pid]}))
            res = _run_caller(run_dir, "hold_watchdog.pid", "watchdog")
            self.assertEqual(res.returncode, 0, res.stderr)
            self.assertTrue((run_dir / "CALLER_DONE.json").exists())
            self.assertTrue((run_dir / "MSA_SIDECAR_DIED.json").exists())
            self._assert_workers_dead(procs)
            self._assert_gone(guard.pid)


class TestReadinessWatchdog(unittest.TestCase):
    """P0-2: the external watchdog covers the readiness barrier and the MSA
    pre-adoption window (READINESS_HOLDS.json), never idle-exits while
    readiness holds exist, and fails closed on a corrupt holds file."""

    def _stop_capture(self):
        calls = []
        return calls, (lambda rd, r: calls.append(r))

    def test_guard_death_partial_readiness_releases_and_stops(self):
        # TEST 1: guard dies mid-barrier with a PARTIAL readiness set ->
        # SIGCONT the frozen workers, CONTINUATION_GUARD_DIED (plus the
        # readiness-specific GUARD_DIED_DURING_READINESS record), stop.
        from dexpbt_isaaclab.msa.hold_watchdog import check_once
        from dexpbt_isaaclab.msa.sidecar import _proc_state
        with tempfile.TemporaryDirectory() as td:
            run_dir = Path(td)
            frozen = {w: _spawn_stopped(self) for w in range(3)}
            _write_readiness_holds_file(run_dir, frozen)
            (run_dir / "continuation_guard.pid").write_text(str(_dead_pid()))
            calls, stop = self._stop_capture()
            state = check_once(run_dir, stop_arm_fn=stop)
            self.assertEqual(state, "guard_died")
            marker = json.loads(
                (run_dir / "GUARD_DIED_DURING_READINESS.json").read_text())
            self.assertIn("guard died", marker["phase"])
            self.assertTrue(
                (run_dir / "CONTINUATION_GUARD_DIED.json").exists())
            self.assertEqual(len(calls), 1)
            for p in frozen.values():
                deadline = __import__("time").monotonic() + 5
                while (_proc_state(p.pid) in ("T", "t")
                       and __import__("time").monotonic() < deadline):
                    __import__("time").sleep(0.05)
                self.assertNotIn(_proc_state(p.pid), ("T", "t"))

    def test_sidecar_death_before_adoption_releases_and_stops(self):
        # TEST 2: 6/6 marker written, readiness holds still present (MSA
        # arm awaiting adoption), sidecar dies -> release all + stop.
        from dexpbt_isaaclab.msa.hold_watchdog import check_once
        with tempfile.TemporaryDirectory() as td:
            run_dir = Path(td)
            frozen = {w: _spawn_stopped(self) for w in range(6)}
            _write_readiness_holds_file(run_dir, frozen)
            _write_marker_stub(run_dir)               # FIRST_TICK_VALIDATED
            guard = _spawn_running(self)              # guard still alive
            (run_dir / "continuation_guard.pid").write_text(str(guard.pid))
            (run_dir / "msa_sidecar.pid").write_text(str(_dead_pid()))
            calls, stop = self._stop_capture()
            state = check_once(run_dir, stop_arm_fn=stop)
            self.assertEqual(state, "released_readiness")
            marker = json.loads(
                (run_dir / "GUARD_DIED_DURING_READINESS.json").read_text())
            self.assertIn("sidecar died before readiness", marker["phase"])
            self.assertTrue(marker["marker_present"])
            self.assertEqual(len(calls), 1)

    def test_readiness_present_never_idle(self):
        # TEST 3: while readiness holds exist the watchdog never idle-exits
        # (owner alive -> ok; owner dead -> released_readiness).
        from dexpbt_isaaclab.msa.hold_watchdog import check_once
        with tempfile.TemporaryDirectory() as td:
            run_dir = Path(td)
            frozen = {w: _spawn_stopped(self) for w in range(6)}
            _write_readiness_holds_file(run_dir, frozen)
            guard = _spawn_running(self)
            (run_dir / "continuation_guard.pid").write_text(str(guard.pid))
            _, stop = self._stop_capture()
            self.assertEqual(check_once(run_dir, stop_arm_fn=stop), "ok")
            # guard dies: still not idle — fatal guard death
            guard.kill()
            guard.wait(timeout=5)
            state = check_once(run_dir, stop_arm_fn=stop)
            self.assertNotEqual(state, "idle_exit")
            self.assertEqual(state, "guard_died")

    def test_corrupt_readiness_fails_closed(self):
        # TEST 4: an unreadable READINESS_HOLDS.json is NEVER treated as
        # the empty set — with a dead/absent guard past grace it fails
        # closed; before the grace it waits.
        from dexpbt_isaaclab.msa.hold_watchdog import check_once
        with tempfile.TemporaryDirectory() as td:
            run_dir = Path(td)
            (run_dir / "READINESS_HOLDS.json").write_text("{ not json")
            _, stop = self._stop_capture()
            # guard pid not written yet, still in startup grace -> wait
            self.assertEqual(
                check_once(run_dir, stop_arm_fn=stop, grace_expired=False),
                "ok")
            # grace elapsed, guard never came up -> fail closed
            state = check_once(run_dir, stop_arm_fn=stop, grace_expired=True)
            self.assertEqual(state, "guard_died")
            marker = json.loads(
                (run_dir / "GUARD_DIED_DURING_READINESS.json").read_text())
            self.assertTrue(marker["readiness_holds_corrupt"])
            marker2 = json.loads(
                (run_dir / "CONTINUATION_GUARD_DIED.json").read_text())
            self.assertTrue(marker2["readiness_holds_corrupt"])

    def test_normal_adoption_no_false_action(self):
        # TEST 5: after adoption (readiness gone, MSA_HOLDS present) with a
        # LIVE sidecar the watchdog is quiet — no release, no markers.
        from dexpbt_isaaclab.msa.hold_watchdog import check_once
        with tempfile.TemporaryDirectory() as td:
            run_dir = Path(td)
            held = _spawn_stopped(self)
            (run_dir / "MSA_HOLDS.json").write_text(
                json.dumps({"0": [held.pid]}))
            sidecar = _spawn_running(self)
            (run_dir / "msa_sidecar.pid").write_text(str(sidecar.pid))
            guard = _spawn_running(self)
            (run_dir / "continuation_guard.pid").write_text(str(guard.pid))
            _write_marker_stub(run_dir)
            calls, stop = self._stop_capture()
            self.assertEqual(check_once(run_dir, stop_arm_fn=stop), "ok")
            self.assertEqual(calls, [])
            self.assertFalse((run_dir / "MSA_SIDECAR_DIED.json").exists())
            self.assertFalse(
                (run_dir / "GUARD_DIED_DURING_READINESS.json").exists())
            # the adopted worker was never disturbed
            from dexpbt_isaaclab.msa.sidecar import _proc_state
            self.assertIn(_proc_state(held.pid), ("T", "t"))

    def test_guard_alive_during_barrier_is_ok(self):
        # TEST 6: guard alive mid-barrier (no marker yet, readiness holds
        # accumulating) -> ok, nobody released.
        from dexpbt_isaaclab.msa.hold_watchdog import check_once
        from dexpbt_isaaclab.msa.sidecar import _proc_state
        with tempfile.TemporaryDirectory() as td:
            run_dir = Path(td)
            frozen = {w: _spawn_stopped(self) for w in range(2)}
            _write_readiness_holds_file(run_dir, frozen)
            guard = _spawn_running(self)
            (run_dir / "continuation_guard.pid").write_text(str(guard.pid))
            calls, stop = self._stop_capture()
            self.assertEqual(check_once(run_dir, stop_arm_fn=stop), "ok")
            self.assertEqual(calls, [])
            for p in frozen.values():
                self.assertIn(_proc_state(p.pid), ("T", "t"))


class TestCoexistenceEvalBounded(unittest.TestCase):
    """Patch A: the coexistence eval never uses an unconsumed PIPE (log
    file instead), has a HARD wall timeout that terminates the whole
    evaluator process group, and the trainer hold is released on the
    timeout path too."""

    def test_large_output_does_not_deadlock(self):
        # a child writing far more than the ~64KB pipe buffer must finish:
        # with an unconsumed PIPE this would block forever.
        from dexpbt_isaaclab.msa.sidecar import _run_bounded_logged
        with tempfile.TemporaryDirectory() as td:
            log = Path(td) / "eval.log"
            code = ("import sys\n"
                    "sys.stdout.write('x' * 2_000_000)\n"
                    "sys.stderr.write('e' * 2_000_000)\n"
                    "print('done')\n")
            res = _run_bounded_logged(
                [sys.executable, "-c", code], dict(os.environ), log,
                gpu_index=None, timeout_s=120.0, poll_s=0.05)
            self.assertEqual(res["exit_code"], 0)
            self.assertFalse(res["timed_out"])
            self.assertEqual(res["survivor_pids"], [])
            self.assertGreater(log.stat().st_size, 3_900_000)
            self.assertEqual(res["log_path"], str(log))
            self.assertIn("done", res["error_tail"])

    def test_timeout_terminates_whole_process_group(self):
        # a hung evaluator WITH a child must be SIGTERM/SIGKILLed as a
        # process group and leave no survivor.
        from dexpbt_isaaclab.msa.sidecar import (
            _pid_running,
            _run_bounded_logged,
        )
        with tempfile.TemporaryDirectory() as td:
            log = Path(td) / "eval.log"
            t0 = __import__("time").monotonic()
            res = _run_bounded_logged(
                ["bash", "-c", "sleep 300 & echo child_spawned; "
                               "exec sleep 300"],
                dict(os.environ), log, gpu_index=None, timeout_s=1.0,
                term_grace_s=2.0, poll_s=0.1)
            elapsed = __import__("time").monotonic() - t0
            self.assertTrue(res["timed_out"])
            self.assertEqual(res["survivor_pids"], [])
            self.assertNotEqual(res["exit_code"], 0)
            self.assertLess(elapsed, 30.0)     # bounded, never infinite
            # log captured pre-kill output, proving the redirect worked
            self.assertIn("child_spawned", res["error_tail"])
            # nothing from the group survives (re-verify independently)
            for p in res["survivor_pids"]:
                self.assertFalse(_pid_running(p))

    def test_timeout_releases_trainer_hold_and_fails_certificate(self):
        # certificate_smoke: an eval timeout must (a) fail the certificate
        # (ok=false, eval_coexistence_ok=false, eval_timed_out=true) and
        # (b) STILL release the trainer hold in the finally block.
        from dexpbt_isaaclab.msa import sidecar as sc
        from dexpbt_isaaclab.msa.sidecar import _proc_state, read_holds_file
        with tempfile.TemporaryDirectory() as td:
            run_dir = Path(td)
            ws = _mk_workspace(run_dir, {0: 5_020_000_000})
            procs = _spawn_worker_procs(self, run_dir, workers=[0])
            fake_res = {"exit_code": None, "timed_out": True,
                        "duration_s": 1800.0,
                        "log_path": str(run_dir / "cert_eval.log"),
                        "error_tail": "isaac hung", "survivor_pids": [],
                        "gpu_memory_peak": 12345}
            patches = {
                "_current_git_commit": lambda repo: "testcommit",
                "_gpu_env": lambda idxs: ({"4": "FakeGPU"}, "550", "12.4"),
                "_gpu_uuid_map": lambda: {"4": "GPU-fake"},
                "_gpu_memory_used": lambda idx: None,
                "_run_coexistence_eval":
                    lambda *a, **k: dict(fake_res),
            }
            saved = {k: getattr(sc, k) for k in patches}
            for k, v in patches.items():
                setattr(sc, k, v)
            try:
                cert_path = run_dir / "CERT.json"
                rc = sc.certificate_smoke(
                    run_dir, ws, 0, 0.3, cert_path,
                    task="T-v0", profile="prof", gpu_index="4",
                    repo_dir=run_dir, eval_num_envs=8,
                    eval_timeout_s=1800.0)
            finally:
                for k, v in saved.items():
                    setattr(sc, k, v)
            self.assertEqual(rc, 2)
            cert = json.loads(cert_path.read_text())
            self.assertFalse(cert["ok"])
            self.assertFalse(cert["eval_coexistence_ok"])
            self.assertTrue(cert["eval_timed_out"])
            self.assertEqual(cert["eval_duration_s"], 1800.0)
            self.assertEqual(cert["eval_survivor_pids"], [])
            self.assertIn("timed out", cert["error"])
            self.assertIn("isaac hung", cert["eval_error_tail"])
            # the trainer hold WAS released despite the timeout
            deadline = __import__("time").monotonic() + 5
            while (_proc_state(procs[0].pid) in ("T", "t")
                   and __import__("time").monotonic() < deadline):
                __import__("time").sleep(0.05)
            self.assertNotIn(_proc_state(procs[0].pid), ("T", "t"))
            self.assertEqual(read_holds_file(run_dir), {})


class TestGuardLifetimeWatchdog(unittest.TestCase):
    """Patch B: the watchdog supervises continuation_guard.pid for the
    WHOLE arm lifetime — guard death before ARM_COMPLETE.json is fatal in
    every phase; guard exit after ARM_COMPLETE is never reported."""

    def _stop_capture(self):
        calls = []
        return calls, (lambda rd, r: calls.append(r))

    def test_base_arm_guard_death_after_readiness_stops_arm(self):
        # BASE/GATE arm shape: readiness done (marker present, holds
        # released, NO sidecar) — the guard dying mid-continuation must
        # stop the arm and record the full state.
        from dexpbt_isaaclab.msa.hold_watchdog import check_once
        with tempfile.TemporaryDirectory() as td:
            run_dir = Path(td)
            _mk_workspace(run_dir, {w: 5_100_000_000 + w for w in range(6)})
            _write_marker_stub(run_dir)
            (run_dir / "continuation_guard.pid").write_text(str(_dead_pid()))
            calls, stop = self._stop_capture()
            state = check_once(run_dir, stop_arm_fn=stop)
            self.assertEqual(state, "guard_died")
            self.assertEqual(len(calls), 1)
            marker = json.loads(
                (run_dir / "CONTINUATION_GUARD_DIED.json").read_text())
            self.assertTrue(marker["readiness_marker_present"])
            self.assertEqual(marker["msa_holds"], {})
            self.assertIsNone(marker["msa_busy"])
            self.assertIn("guard_pid", marker)
            self.assertIn("time", marker)
            for w in range(6):
                self.assertEqual(marker["worker_frames"][str(w)],
                                 5_100_000_000 + w)

    def test_msa_adoption_guard_death_releases_and_stops(self):
        # MSA arm mid-cycle: sidecar ALIVE and holding a worker — guard
        # death is still fatal; the union of holds is SIGCONTed.
        from dexpbt_isaaclab.msa.hold_watchdog import check_once
        from dexpbt_isaaclab.msa.sidecar import _proc_state
        with tempfile.TemporaryDirectory() as td:
            run_dir = Path(td)
            _write_marker_stub(run_dir)
            held = _spawn_stopped(self)
            (run_dir / "MSA_HOLDS.json").write_text(
                json.dumps({"2": [held.pid]}))
            sidecar = _spawn_running(self)
            (run_dir / "msa_sidecar.pid").write_text(str(sidecar.pid))
            (run_dir / "continuation_guard.pid").write_text(str(_dead_pid()))
            calls, stop = self._stop_capture()
            state = check_once(run_dir, stop_arm_fn=stop)
            self.assertEqual(state, "guard_died")
            self.assertEqual(len(calls), 1)
            marker = json.loads(
                (run_dir / "CONTINUATION_GUARD_DIED.json").read_text())
            self.assertEqual(marker["msa_holds"]["2"], [held.pid])
            # the held worker was SIGCONTed back to life
            deadline = __import__("time").monotonic() + 5
            while (_proc_state(held.pid) in ("T", "t")
                   and __import__("time").monotonic() < deadline):
                __import__("time").sleep(0.05)
            self.assertNotIn(_proc_state(held.pid), ("T", "t"))

    def test_msa_busy_guard_death_fails_closed(self):
        # guard death while MSA_BUSY.json exists (holds empty, e.g. the
        # COMMITTING window) is equally fatal — never a quiet pass.
        from dexpbt_isaaclab.msa.hold_watchdog import check_once
        with tempfile.TemporaryDirectory() as td:
            run_dir = Path(td)
            _write_marker_stub(run_dir)
            sidecar = _spawn_running(self)
            (run_dir / "msa_sidecar.pid").write_text(str(sidecar.pid))
            (run_dir / "MSA_BUSY.json").write_text(json.dumps(
                {"intervention_id": "msa_x", "status": "COMMITTING"}))
            (run_dir / "continuation_guard.pid").write_text(str(_dead_pid()))
            calls, stop = self._stop_capture()
            state = check_once(run_dir, stop_arm_fn=stop)
            self.assertEqual(state, "guard_died")
            self.assertEqual(len(calls), 1)
            marker = json.loads(
                (run_dir / "CONTINUATION_GUARD_DIED.json").read_text())
            self.assertEqual(marker["msa_busy"]["status"], "COMMITTING")

    def test_arm_complete_guard_exit_is_not_reported(self):
        # NORMAL completion: after ARM_COMPLETE.json the guard's exit is
        # expected — the watchdog idles out with no marker and no stop.
        from dexpbt_isaaclab.msa.hold_watchdog import check_once
        with tempfile.TemporaryDirectory() as td:
            run_dir = Path(td)
            _write_marker_stub(run_dir)
            (run_dir / "ARM_COMPLETE.json").write_text("{}")
            (run_dir / "continuation_guard.pid").write_text(str(_dead_pid()))
            calls, stop = self._stop_capture()
            self.assertEqual(check_once(run_dir, stop_arm_fn=stop),
                             "idle_exit")
            self.assertEqual(calls, [])
            self.assertFalse(
                (run_dir / "CONTINUATION_GUARD_DIED.json").exists())
            # a MISSING guard pid file after ARM_COMPLETE is fine too
            (run_dir / "continuation_guard.pid").unlink()
            self.assertEqual(check_once(run_dir, stop_arm_fn=stop),
                             "idle_exit")
            self.assertEqual(calls, [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
