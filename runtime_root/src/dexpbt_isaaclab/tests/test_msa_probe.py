"""Unit tests for the Matched Sentinel Advancement (MSA) probe toolkit.

Covers the 15 required behaviors (spec section J) plus state-machine and
arm-generation invariants. Pure CPU; no Isaac, no GPU.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dexpbt_isaaclab.msa import decision as dec
from dexpbt_isaaclab.msa import dry_run as dr
from dexpbt_isaaclab.msa.eval_bridge import SnapshotSpec, build_seed_manifest
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


def _prov_with_exposure(worker=1, tau_plus=0.0223, start=4_950_000_000,
                        end=4_958_000_000):
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
                                   end=4_958_000_000)
        # exposure inside the window -> not due
        self.assertFalse(probe_due(prov, 5_000_000_000, tau))
        # window has fully passed the exposure -> due again
        self.assertTrue(probe_due(prov, 4_958_000_000 + EXPOSURE_WINDOW_FRAMES + 1,
                                  tau))

    def test_floor_worker_never_due(self):
        self.assertFalse(probe_due(None, 5_000_000_000, 0.0235,
                                   is_floor_worker=True))

    def test_tighter_exposure_counts(self):
        tau = 0.0235
        prov = _prov_with_exposure(tau_plus=0.015)  # tighter than tau_plus
        self.assertFalse(probe_due(prov, 5_000_000_000, tau))


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

    def test_only_forced_tolerance_field_is_written(self):
        # the sidecar's control write path touches forced_success_tolerance
        # and msa_* only
        src = (MSA_DIR / "sidecar.py").read_text()
        import re
        writes = re.findall(r"control\.(\w+)\s*=", src)
        self.assertEqual(sorted(set(writes)),
                         ["forced_success_tolerance", "msa_enabled"])


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

    def test_bootstrap_is_labeled_conditional(self):
        diffs = {w: [0.02, 0.03, 0.01, 0.02] for w in range(6)}
        s = dec.panel_bootstrap_lower_quantile(diffs, n_boot=50)
        self.assertIn("NOT a population CI", s["note"])

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


class TestDryRunCycle(unittest.TestCase):
    """Full probe cycle against the deterministic fake world."""

    def setUp(self):
        self.td = tempfile.TemporaryDirectory()
        self.world, self.ctl, self.result = dr.run_dry_run(Path(self.td.name))

    def tearDown(self):
        self.td.cleanup()

    # J6: every worker's first half is exactly 8M
    def test_common_probe_is_8m_for_all(self):
        recs = self.result.records
        for w in range(6):
            segs = [s for s in recs[w]["segments"]
                    if s["segment_type"] == "common_probe"]
            self.assertEqual(len(segs), 1)
            self.assertEqual(
                segs[0]["segment_end_frame"] - segs[0]["segment_start_frame"],
                8_000_000)

    # J9: q_post never enters the common-dose L_mid
    def test_q_post_not_in_common_dose(self):
        recs = self.result.records
        recipients = set(self.result.routing.recipients)
        for w in range(6):
            if w in recipients:
                self.assertIn("q_post_mean", recs[w])
            else:
                self.assertNotIn("q_post_mean", recs[w])
        # L_mid for recipients was computed BEFORE the second packet:
        # re-derive from pre/mid means in the fake world (learn_rate * 1
        # packet), not 2 packets.
        for w, lm in self.result.l_mid_by_worker.items():
            self.assertAlmostEqual(lm, self.world.learn_rate[w], delta=1e-9)

    # J13: no optimizer/PBT lineage reset during/after the probe
    def test_no_resets(self):
        self.assertEqual(self.world.reset_calls, [])

    # J14: frame ledger includes all probe frames
    def test_frame_ledger_includes_probe_frames(self):
        recipients = set(self.result.routing.recipients)
        for w in range(6):
            expect = 16_000_000 if w in recipients else 8_000_000
            self.assertEqual(self.world.trained_frames[w], expect)
            self.assertEqual(self.world.frame[w], 5_000_000_000 + expect)
        # provenance segments account for every trained frame
        for w in range(6):
            prov = self.ctl.load_provenance(w)
            total = sum(s["segment_end_frame"] - s["segment_start_frame"]
                        for s in prov.exposures)
            self.assertEqual(total, self.world.trained_frames[w])

    def test_forced_tolerance_sequence(self):
        # tau_plus during probe, back to tau after; promotion re-applies
        # tau_plus permanently at the end (this dry run promotes)
        tau_plus = dr.tau_plus_of(dr.DRY_RUN_TAU)
        by_worker = {}
        for w, tol in self.world.forced_tol_log:
            by_worker.setdefault(w, []).append(round(tol, 9))
        for w in range(6):
            seq = by_worker[w]
            self.assertEqual(seq[0], round(tau_plus, 9))       # common probe
            self.assertIn(round(dr.DRY_RUN_TAU, 9), seq)       # revert to tau
            self.assertEqual(seq[-1], round(tau_plus, 9))      # promotion

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

    def test_second_probe_deterministic_recipients(self):
        # learn rates 1>3>(0==5 tie -> 0)>4>2
        self.assertEqual(self.result.routing.recipients, [1, 3, 0])

    def test_permuted_arm_same_dose_different_mapping(self):
        with tempfile.TemporaryDirectory() as td2:
            world2, _, result2 = dr.run_dry_run(Path(td2), permuted=True)
        self.assertEqual(sum(world2.trained_frames.values()),
                         sum(self.world.trained_frames.values()))
        self.assertEqual(len(result2.routing.recipients),
                         len(self.result.routing.recipients))


class TestOriginalProfileUnchanged(unittest.TestCase):
    # J15: original base-retention behavior is untouched
    def test_control_state_defaults_inert(self):
        from dexpbt_isaaclab.pbt.pbt_curriculum_p1 import CurriculumControlState
        s = CurriculumControlState()
        self.assertFalse(s.msa_enabled)
        self.assertIsNone(s.msa_provenance)
        self.assertIsNone(s.msa_state_machine)

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
            for k in ("msa_enabled", "msa_provenance", "msa_state_machine"):
                legacy.pop(k)
            p.write_text(json.dumps(legacy))
            state = read_curriculum_control(p)
            self.assertIsNotNone(state)
            self.assertFalse(state.msa_enabled)
            self.assertIsNone(state.msa_provenance)
            # roundtrip through the writer keeps working
            write_curriculum_control(p, state)
            self.assertFalse(read_curriculum_control(p).msa_enabled)

    def test_env_curriculum_does_not_branch_on_msa(self):
        import inspect

        from dexpbt_isaaclab.pbt import pbt_curriculum_p1 as mod
        for fn in (mod.apply_env_tolerance_curriculum,
                   mod.dexpbt_tolerance_curriculum_mastery,
                   mod.mark_shrink_consumed):
            self.assertNotIn("msa", inspect.getsource(fn))

    def test_base_continuation_launcher_keeps_original_gates(self):
        launcher_dir = Path("/root/autodl-tmp/wmdexpbt/results/launchers/msa_probe")
        if not launcher_dir.exists():
            self.skipTest("arm launchers not generated in this checkout")
        base = (launcher_dir / "start_base_continuation.sh").read_text()
        relax = (launcher_dir / "start_gate_relax.sh").read_text()
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
        for sh in launcher_dir.glob("start_*.sh"):
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

        launcher_dir = Path("/root/autodl-tmp/wmdexpbt/results/launchers/msa_probe")
        if not launcher_dir.exists():
            self.skipTest("arm launchers not generated in this checkout")
        relax = (launcher_dir / "start_gate_relax.sh").read_text()
        self.assertIn('--profile-per-policy "'
                      + ",".join(["cser_dexpbt_conservative_gate_relax"] * 6)
                      + '"', relax)
        for other in ("start_base_continuation.sh", "start_msa.sh",
                      "start_msa_permuted_route.sh"):
            text = (launcher_dir / other).read_text()
            self.assertIn('--profile-per-policy "'
                          + ",".join(["cser_dexpbt_conservative"] * 6)
                          + '"', text)
            self.assertNotIn("gate_relax", text)


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
        cfg_dir = Path("/root/autodl-tmp/wmdexpbt/results/launchers/msa_probe")
        if not cfg_dir.exists():
            self.skipTest("arm configs not generated in this checkout")
        cfgs = {p.stem.replace(".config", ""): json.loads(p.read_text())
                for p in cfg_dir.glob("*.config.json")}
        self.assertEqual(len(cfgs), 4)
        seeds = {c["base_seed"] for c in cfgs.values()}
        self.assertEqual(len(seeds), 1)          # matched seeds
        resumes = {json.dumps(c["resume_checkpoints"], sort_keys=True)
                   for c in cfgs.values()}
        self.assertEqual(len(resumes), 1)        # same six 5B checkpoints
        for c in cfgs.values():
            self.assertTrue(c["frame_ledger"]["monotonic"])
            self.assertTrue(c["frame_ledger"]["probe_frames_charged"])
        self.assertTrue(cfgs["msa_permuted_route"]["msa"]["permuted_route"])
        self.assertFalse(cfgs["msa"]["msa"]["permuted_route"])
        # same probe dose/eval budget between MSA and its negative control
        for k in ("packet_frames", "top_k_recipients", "eval_num_seeds",
                  "eval_seed_manifest_id"):
            self.assertEqual(cfgs["msa"]["msa"][k],
                             cfgs["msa_permuted_route"]["msa"][k])


if __name__ == "__main__":
    unittest.main(verbosity=2)
