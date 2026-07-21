"""Production MSA fusion protocol / transaction / adapter / PBT gate tests."""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dexpbt_isaaclab.msa.fake_decision import ENV_GUARD, injector_allowed
from dexpbt_isaaclab.msa.fusion_adapter import (
    sidecar_summary_to_commands,
    try_complete_cycle,
)
from dexpbt_isaaclab.msa.fusion_config import fusion_config_from_mapping
from dexpbt_isaaclab.msa.fusion_protocol import (
    WorkerAck,
    WorkerCommand,
    claim_cycle_result,
    claim_worker_command,
    list_pending_commands,
    validate_command_for_worker,
    write_worker_ack,
    write_worker_command,
)
from dexpbt_isaaclab.msa.fusion_transaction import (
    fingerprint_pair,
    restore_anchor_into_workspace,
    save_committed_anchor,
    save_probe_anchor,
    verify_anchor_fingerprint,
)
from dexpbt_isaaclab.msa.pbt_eligibility import (
    can_be_exploit_source,
    can_be_exploit_target,
    donor_eligible,
    recipient_after_pbt_copy,
)
from dexpbt_isaaclab.msa.worker_state import (
    MODE_BUILD,
    MODE_HOLD,
    MODE_PROBE,
    WorkerMSAState,
)
from dexpbt_isaaclab.world_model.wm_strategy_profiles import PROFILE_TABLES


def _cfg():
    return fusion_config_from_mapping({"msa_integrated_enabled": True})


def _write_fake_ckpt(ws: Path, stem: str = "000001", tol: float = 0.02):
    ws.mkdir(parents=True, exist_ok=True)
    pth = ws / f"{stem}.pth"
    meta = ws / f"{stem}.json"
    pth.write_bytes(b"WEIGHTS_" + stem.encode() + os.urandom(16))
    meta.write_text(json.dumps({
        "frame": 1000,
        "curriculum": {"success_tolerance": tol},
    }))
    return pth, meta


class TestFakeInjectorFormalProfile(unittest.TestCase):
    def test_formal_profile_disables_injector(self):
        os.environ.pop(ENV_GUARD, None)
        t = PROFILE_TABLES["msa_aware_cser_dexpbt_conservative"]
        self.assertFalse(t.get("msa_fake_decision_injector_enable", False))
        self.assertFalse(injector_allowed(False))
        self.assertFalse(injector_allowed(True))  # env guard off


class TestProtocolExactlyOnce(unittest.TestCase):
    def test_claim_once_and_per_worker_commands(self):
        with tempfile.TemporaryDirectory() as td:
            summary = {
                "intervention_id": "cyc1",
                "due_workers": [0, 1],
                "selected_for_promotion": [0],
                "anchor_workers": [1, 2, 3, 4, 5],
                "tau_by_worker": {"0": 0.02, "1": 0.02},
                "tau_plus_by_worker": {"0": 0.019, "1": 0.019},
                "final_tol_by_worker": {"0": 0.019, "1": 0.02},
                "promoted": True,
                "l_mid_by_worker": {},
                "retention_by_worker": {},
                "rejection_reason_by_worker": {"1": "L_mid_lower_quantile<=0"},
                "eligible_by_worker": {"0": True, "1": False},
            }
            claimed, cmds = sidecar_summary_to_commands(summary, run_dir=td)
            self.assertTrue(claimed)
            self.assertEqual(len(cmds), 2)
            claimed2, cmds2 = sidecar_summary_to_commands(summary, run_dir=td)
            self.assertFalse(claimed2)
            self.assertEqual(cmds2, [])

            c0 = claim_worker_command(td, 0, "cyc1")
            self.assertIsNotNone(c0)
            self.assertEqual(c0.action, "advance")
            # Idempotent re-claim by same worker.
            c0b = claim_worker_command(td, 0, "cyc1")
            self.assertIsNotNone(c0b)

            # Worker 1 cannot claim worker 0's identity via wrong id.
            pending1 = list_pending_commands(td, 1)
            self.assertEqual(pending1, ["cyc1"])
            c1 = claim_worker_command(td, 1, "cyc1")
            self.assertEqual(c1.action, "rollback")

            write_worker_ack(td, WorkerAck(
                cycle_id="cyc1", worker_id=0, action_applied="advance",
                ok=True, mode_after=MODE_BUILD, tolerance_after=0.019,
            ))
            write_worker_ack(td, WorkerAck(
                cycle_id="cyc1", worker_id=1, action_applied="rollback",
                ok=True, mode_after=MODE_HOLD, tolerance_after=0.02,
            ))
            self.assertTrue(try_complete_cycle(td, "cyc1", [0, 1]))

    def test_stale_command_rejected(self):
        cmd = WorkerCommand(
            cycle_id="c", transaction_id="txn_c", worker_id=0,
            action="advance", expected_mode="PROBE",
        )
        ok, reason = validate_command_for_worker(
            cmd, worker_id=0, current_mode=MODE_BUILD,
            current_tolerance=0.02, cycle_id="",
        )
        self.assertFalse(ok)
        self.assertIn("advance_rejected_mode", reason)

    def test_wrong_worker_rejected(self):
        cmd = WorkerCommand(worker_id=2, action="hold", cycle_id="c")
        ok, reason = validate_command_for_worker(
            cmd, worker_id=0, current_mode=MODE_PROBE,
            current_tolerance=0.02, cycle_id="c",
        )
        self.assertFalse(ok)
        self.assertEqual(reason, "worker_id_mismatch")

    def test_cycle_id_mismatch_rejected(self):
        cmd = WorkerCommand(
            cycle_id="c1", transaction_id="txn_c1", worker_id=0,
            action="advance", expected_mode="PROBE",
        )
        ok, reason = validate_command_for_worker(
            cmd, worker_id=0, current_mode=MODE_PROBE,
            current_tolerance=0.02, cycle_id="c2",
        )
        self.assertFalse(ok)
        self.assertEqual(reason, "cycle_id_mismatch")

    def test_matching_txn_form_accepted(self):
        cmd = WorkerCommand(
            cycle_id="c1", transaction_id="txn_c1", worker_id=0,
            action="advance", expected_mode="PROBE",
            expected_tolerance=0.02,
        )
        ok, reason = validate_command_for_worker(
            cmd, worker_id=0, current_mode=MODE_PROBE,
            current_tolerance=0.02, cycle_id="c1",
        )
        self.assertTrue(ok)
        self.assertEqual(reason, "ok")


class TestTransactionCheckpoint(unittest.TestCase):
    def test_save_verify_restore_commit(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            ws = td / "ws" / "000"
            _write_fake_ckpt(ws, "000010", tol=0.022)
            rec = save_probe_anchor(
                td, 0, "cycA", ws, tolerance=0.022, frame=5000,
            )
            ok, fp = verify_anchor_fingerprint(rec)
            self.assertTrue(ok)
            self.assertEqual(fp, rec["fingerprint"])

            # Mutate live workspace then restore.
            _write_fake_ckpt(ws, "000011", tol=0.019)
            restored = restore_anchor_into_workspace(rec, ws)
            self.assertTrue(Path(restored["checkpoint"]).is_file())
            meta = json.loads(Path(restored["meta"]).read_text())
            self.assertAlmostEqual(
                float(meta["curriculum"]["success_tolerance"]), 0.022,
            )

            committed = save_committed_anchor(
                td, 0, "cycA", ws, tolerance=0.019, frame=6000,
            )
            self.assertTrue(Path(committed["anchor_pth"]).is_file())

    def test_fingerprint_mismatch_detected(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            ws = td / "ws" / "000"
            _write_fake_ckpt(ws)
            rec = save_probe_anchor(td, 0, "cycB", ws, tolerance=0.02)
            # Corrupt anchor weights.
            Path(rec["anchor_pth"]).write_bytes(b"CORRUPTED")
            ok, reason = verify_anchor_fingerprint(rec)
            self.assertFalse(ok)
            self.assertEqual(reason, "fingerprint_mismatch")


class TestPBTAuthoritative(unittest.TestCase):
    def test_probe_blocked_as_donor_and_recipient(self):
        cfg = _cfg()
        probe = WorkerMSAState(
            worker_id=0, mode=MODE_PROBE, transaction_open=True,
            current_tolerance=0.02, anchor_tolerance=0.02,
            anchor_checkpoint="a.pth", committed_stage="STRICT",
        )
        build = WorkerMSAState(
            worker_id=1, mode=MODE_BUILD, current_tolerance=0.02,
            anchor_tolerance=0.02, anchor_checkpoint="b.pth",
            committed_stage="STRICT",
        )
        self.assertFalse(can_be_exploit_source(probe))
        self.assertFalse(can_be_exploit_target(probe))
        d = donor_eligible(
            probe, recipient_stage="STRICT",
            population=[probe, build], cfg=cfg,
        )
        self.assertFalse(d.eligible)

    def test_copy_resets_evidence_not_stage_claim(self):
        cfg = _cfg()
        donor = WorkerMSAState(
            worker_id=0, mode=MODE_BUILD, current_tolerance=0.02,
            committed_stage="STRICT", cycle_id="old",
            pass_evidence={"x": 1}, consecutive_passes=2,
        )
        recip = WorkerMSAState(
            worker_id=1, mode=MODE_BUILD, current_tolerance=0.025,
            committed_stage="MID", pass_evidence={"y": 1},
        )
        out = recipient_after_pbt_copy(recip, donor, cfg=cfg)
        self.assertEqual(out.pass_evidence, {})
        self.assertEqual(out.cycle_id, "")
        self.assertEqual(out.worker_id, 1)
        self.assertEqual(out.mode, MODE_BUILD)


class TestToleranceOwnership(unittest.TestCase):
    def test_fusion_profile_proposal_only_flag(self):
        from dexpbt_isaaclab.pbt.pbt_curriculum_p1 import (
            CurriculumControlState,
            apply_env_tolerance_curriculum,
            read_curriculum_control,
            write_curriculum_control,
        )
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "000.json"
            write_curriculum_control(path, CurriculumControlState(
                allow_shrink=True, shrink_token=True,
                msa_integrated_enabled=True,
            ))
            new_tol, _ = apply_env_tolerance_curriculum(
                curriculum_mode="mastery",
                curriculum_control_path=path,
                curriculum_shrink_factor=0.95,
                last_curriculum_update=0,
                frames_since_restart=10**9,
                curriculum_interval=1,
                prev_episode_successes=[1.0] * 10,
                success_tolerance=0.025,
                initial_tolerance=0.05,
                target_tolerance=0.01,
                pbt_single_shrink_per_window=True,
            )
            self.assertAlmostEqual(new_tol, 0.025)
            ctrl = read_curriculum_control(path)
            self.assertGreater(float(ctrl.msa_tolerance_proposal or 0.0), 0.0)


class TestEvalFixtureUsesRealDecision(unittest.TestCase):
    def test_fixture_selects_advance_and_hold(self):
        from dexpbt_isaaclab.msa.eval_fixture import build_fixture_summary
        summary = build_fixture_summary(
            cycle_id="fix1",
            advance_workers=[0],
            hold_workers=[1],
        )
        self.assertIn(0, summary["selected_for_promotion"])
        self.assertNotIn(1, summary["selected_for_promotion"])
        self.assertTrue(summary["promoted"])


class TestObserverCallSitesPresent(unittest.TestCase):
    def test_observer_has_production_msa_methods(self):
        from dexpbt_isaaclab.pbt.pbt_observer import DexPBTPbtAlgoObserver
        for name in (
            "_msa_fusion_after_steps",
            "_msa_enter_probe_with_anchor",
            "_msa_apply_worker_command",
            "_msa_donor_gate",
            "_msa_recipient_gate",
            "_msa_request_rollback_restart",
        ):
            self.assertTrue(hasattr(DexPBTPbtAlgoObserver, name), name)


class TestAdapterNoFakeDependency(unittest.TestCase):
    def test_production_commands_not_test_only(self):
        with tempfile.TemporaryDirectory() as td:
            summary = {
                "intervention_id": "prod1",
                "due_workers": [0],
                "selected_for_promotion": [0],
                "tau_by_worker": {"0": 0.02},
                "tau_plus_by_worker": {"0": 0.019},
                "final_tol_by_worker": {"0": 0.019},
                "promoted": True,
            }
            _, cmds = sidecar_summary_to_commands(summary, run_dir=td)
            self.assertTrue(cmds[0].production)
            self.assertFalse(cmds[0].test_only)


if __name__ == "__main__":
    unittest.main()
