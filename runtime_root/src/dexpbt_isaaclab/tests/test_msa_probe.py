"""Unit tests for the Matched Sentinel Advancement (MSA) probe toolkit.

Covers the original 15 required behaviors plus the pilot-integration fixes:
heterogeneous per-worker tolerances, parallel 20M committed-tick exposure,
eval phase barriers, persisted/resumable cycles, crash cleanup, locked
control updates, canonical tolerance precision, continuation guard and
durable promotion commits. Pure CPU; no Isaac, no GPU.
"""

from __future__ import annotations

import json
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
        self.assertEqual(PROBE_PACKET_FRAMES, 20_000_000)

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
    """First-tick tolerance/lineage/frame validation + hard +100M cap."""

    ENTRY = {"frame": 5_000_000_000, "live_tolerance": 0.023112487509668322,
             "lineage_id": "lin-1", "iteration": 160}

    def test_first_tick_pass(self):
        from dexpbt_isaaclab.msa.continuation_guard import check_first_tick
        meta = {"frame": 5_020_000_000, "lineage_id": "lin-1",
                "curriculum": {"success_tolerance": 0.023112487509668322}}
        self.assertTrue(check_first_tick(self.ENTRY, meta).ok)

    def test_first_tick_tolerance_mismatch_fails_closed(self):
        from dexpbt_isaaclab.msa.continuation_guard import check_first_tick
        meta = {"frame": 5_020_000_000, "lineage_id": "lin-1",
                "curriculum": {"success_tolerance": 0.075}}  # reset to initial
        v = check_first_tick(self.ENTRY, meta)
        self.assertFalse(v.ok)
        self.assertIn("tolerance", v.reason)

    def test_first_tick_frame_reset_fails_closed(self):
        from dexpbt_isaaclab.msa.continuation_guard import check_first_tick
        meta = {"frame": 20_000_000, "lineage_id": "lin-1",
                "curriculum": {"success_tolerance": 0.023112487509668322}}
        v = check_first_tick(self.ENTRY, meta)
        self.assertFalse(v.ok)
        self.assertIn("baseline", v.reason)

    def test_first_tick_lineage_mismatch_fails_closed(self):
        from dexpbt_isaaclab.msa.continuation_guard import check_first_tick
        meta = {"frame": 5_020_000_000, "lineage_id": "OTHER",
                "curriculum": {"success_tolerance": 0.023112487509668322}}
        self.assertFalse(check_first_tick(self.ENTRY, meta).ok)

    def test_hard_cap_100m(self):
        from dexpbt_isaaclab.msa.continuation_guard import (
            CONTINUATION_CAP_FRAMES,
            check_cap,
        )
        self.assertEqual(CONTINUATION_CAP_FRAMES, 100_000_000)
        ok_meta = {"frame": 5_000_000_000 + 100_000_000 + 20_000_000}
        self.assertTrue(check_cap(self.ENTRY, ok_meta).ok)
        over = {"frame": 5_000_000_000 + 100_000_000 + 20_000_001}
        v = check_cap(self.ENTRY, over)
        self.assertFalse(v.ok)
        self.assertIn("hard cap", v.reason)

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
        base = (LAUNCHER_DIR / "start_base_continuation.sh").read_text()
        relax = (LAUNCHER_DIR / "start_gate_relax.sh").read_text()
        self.assertIn("--late_floor_retention_min_retention 0.88", base)
        self.assertIn("--late_floor_retention_min_retention 0.75", relax)
        # single-flag diff: everything else identical
        diff = [
            (a, b) for a, b in zip(base.splitlines(), relax.splitlines())
            if a != b and "GATE_RELAX" not in a + b
            and "gate_relax" not in a + b and "BASE_CONTINUATION" not in a + b
            and "base_continuation" not in a + b
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
        relax = (LAUNCHER_DIR / "start_gate_relax.sh").read_text()
        self.assertIn('--profile-per-policy "'
                      + ",".join(["cser_dexpbt_conservative_gate_relax"] * 6)
                      + '"', relax)
        for other in ("start_base_continuation.sh", "start_msa.sh",
                      "start_msa_permuted_route.sh"):
            text = (LAUNCHER_DIR / other).read_text()
            self.assertIn('--profile-per-policy "'
                          + ",".join(["cser_dexpbt_conservative"] * 6)
                          + '"', text)
            self.assertNotIn("gate_relax", text)

    def test_launchers_have_guard_tolerance_resume_and_eval_device(self):
        if not LAUNCHER_DIR.exists():
            self.skipTest("arm launchers not generated in this checkout")
        for sh in LAUNCHER_DIR.glob("start_*.sh"):
            text = sh.read_text()
            # fail-closed bootstrap verification before any launch
            self.assertIn("continuation_guard", text)
            self.assertIn("--verify-only", text)
            self.assertIn("REFUSING TO LAUNCH: bootstrap verification",
                          text)
            # explicit tolerance resume (a bare --checkpoint resets tol)
            self.assertIn("--tolerance-resume-per-policy", text)
            # hard cap guard loop
            self.assertIn("continuation_guard.log", text)
        for arm in ("start_msa.sh", "start_msa_permuted_route.sh"):
            text = (LAUNCHER_DIR / arm).read_text()
            self.assertIn("--eval-device", text)
            self.assertIn("--num-seeds", text)


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
        self.assertTrue(cfgs["msa_permuted_route"]["msa"]["permuted_route"])
        self.assertFalse(cfgs["msa"]["msa"]["permuted_route"])
        # 20M committed-tick packet everywhere
        for name in ("msa", "msa_permuted_route"):
            self.assertEqual(cfgs[name]["msa"]["packet_frames"], 20_000_000)
            self.assertEqual(cfgs[name]["msa"]["eval_num_seeds_pilot"],
                             PILOT_NUM_SEEDS)
        # same probe dose/eval budget between MSA and its negative control
        for k in ("packet_frames", "top_k_recipients",
                  "eval_num_seeds_pilot", "eval_seed_manifest_id_pilot"):
            self.assertEqual(cfgs["msa"]["msa"][k],
                             cfgs["msa_permuted_route"]["msa"][k])


if __name__ == "__main__":
    unittest.main(verbosity=2)
