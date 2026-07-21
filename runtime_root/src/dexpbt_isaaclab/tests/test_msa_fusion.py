"""Unit + integration tests for MSA-aware DexPBT fusion.

Pure CPU; no Isaac / no GPU. Covers config, stages, metrics gates, PBT
eligibility, serialization/migration, WM advance guard, and fake-trainer
state-machine flows.
"""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dexpbt_isaaclab.msa.fake_decision import (
    ENV_GUARD,
    injector_allowed,
    write_fake_decisions,
)
from dexpbt_isaaclab.msa.fusion_config import (
    MSAFusionConfig,
    cm_to_m,
    fusion_config_from_mapping,
    m_to_cm,
)
from dexpbt_isaaclab.msa.fusion_hook import (
    MSAFusionHook,
    wm_signals_must_not_advance,
)
from dexpbt_isaaclab.msa.pbt_eligibility import (
    can_be_exploit_source,
    can_be_exploit_target,
    count_reliable_anchors,
    donor_eligible,
    population_has_min_anchors,
    recipient_after_pbt_copy,
)
from dexpbt_isaaclab.msa.worker_state import (
    MODE_BUILD,
    MODE_HOLD,
    MODE_PROBE,
    WorkerMSAState,
    apply_advance,
    apply_rollback_hold,
    load_worker_state,
    migrate_worker_state,
    save_worker_state,
)
from dexpbt_isaaclab.pbt.pbt_curriculum_p1 import CurriculumControlState
from dexpbt_isaaclab.world_model.wm_strategy_profiles import PROFILE_TABLES


def _cfg(**kwargs) -> MSAFusionConfig:
    base = {"msa_integrated_enabled": True}
    base.update(kwargs)
    return fusion_config_from_mapping(base)


class TestUnitsCmAndStages(unittest.TestCase):
    def test_cm_m_roundtrip(self):
        self.assertAlmostEqual(cm_to_m(3.0), 0.03)
        self.assertAlmostEqual(m_to_cm(0.02), 2.0)

    def test_stage_split(self):
        cfg = _cfg()
        self.assertEqual(cfg.stage_for_tolerance(0.035), "BUILD")
        self.assertEqual(cfg.stage_for_tolerance(0.025), "MID")
        self.assertEqual(cfg.stage_for_tolerance(0.020), "STRICT")
        self.assertEqual(cfg.stage_for_tolerance(0.015), "STRICT")

    def test_probe_target_and_floor(self):
        cfg = _cfg()
        self.assertAlmostEqual(cfg.probe_target(0.020), 0.019)
        self.assertAlmostEqual(cfg.probe_target(0.0105), 0.010)  # floor

    def test_config_validation(self):
        with self.assertRaises(ValueError):
            fusion_config_from_mapping({
                "msa_build_threshold_cm": 1.0,
                "msa_strict_threshold_cm": 2.0,
            })
        with self.assertRaises(ValueError):
            fusion_config_from_mapping({"msa_tighten_ratio": 1.5})

    def test_profile_table_has_fusion(self):
        t = PROFILE_TABLES["msa_aware_cser_dexpbt_conservative"]
        self.assertTrue(t["msa_integrated_enabled"])
        self.assertFalse(t["msa_fake_decision_injector_enable"])
        # Old conservative profile unchanged.
        old = PROFILE_TABLES["cser_dexpbt_conservative"]
        self.assertFalse(old.get("msa_integrated_enabled", False))


class TestAdvanceGates(unittest.TestCase):
    def test_apply_advance_rollback_cooldown(self):
        cfg = _cfg()
        st = WorkerMSAState(
            worker_id=1,
            mode=MODE_PROBE,
            current_tolerance=0.02,
            anchor_tolerance=0.02,
            transaction_open=True,
            pending_target_tolerance=0.019,
        )
        ok = apply_advance(
            st, new_tol=0.019, frame=100, anchor_checkpoint="ck",
            metrics={"l_lcb": 0.1, "rho": 0.95}, cfg=cfg,
        )
        self.assertEqual(ok.mode, MODE_BUILD)
        self.assertAlmostEqual(ok.current_tolerance, 0.019)
        self.assertEqual(ok.advance_count, 1)
        self.assertEqual(ok.pass_evidence, {})

        st2 = WorkerMSAState(
            worker_id=2,
            mode=MODE_PROBE,
            current_tolerance=0.02,
            anchor_tolerance=0.02,
            transaction_open=True,
        )
        bad = apply_rollback_hold(
            st2, frame=100, cooldown_frames=50,
            metrics={"l_lcb": -0.1}, reason="fail",
        )
        self.assertEqual(bad.mode, MODE_HOLD)
        self.assertEqual(bad.hold_until_frame, 150)
        self.assertEqual(bad.consecutive_failures, 1)
        self.assertEqual(bad.rollback_count, 1)

    def test_hook_advance_requires_l_and_rho(self):
        with tempfile.TemporaryDirectory() as td:
            hook = MSAFusionHook(td, _cfg(), num_workers=2)
            st = hook.load_or_migrate(0, current_tolerance=0.02)
            st = hook.start_probe(st, frame=10)
            # Fail: L_LCB <= 0
            st = hook.apply_decision_from_metrics(
                st, frame=20, l_lcb=-0.01, rho=1.0, rho_available=True,
                promote=True, reason="neg_l",
            )
            self.assertEqual(st.mode, MODE_HOLD)
            st = hook.tick_hold_exit(st, frame=10**12)
            st = hook.start_probe(st, frame=30)
            st = hook.apply_decision_from_metrics(
                st, frame=40, l_lcb=0.1, rho=0.5, rho_available=True,
                promote=True, reason="low_rho",
            )
            self.assertEqual(st.mode, MODE_HOLD)
            st = hook.tick_hold_exit(st, frame=10**12)
            st = hook.start_probe(st, frame=50)
            st = hook.apply_decision_from_metrics(
                st, frame=60, l_lcb=0.1, rho=0.95, rho_available=True,
                promote=True, reason="ok",
                anchor_checkpoint="a.pth",
            )
            self.assertEqual(st.mode, MODE_BUILD)
            self.assertAlmostEqual(st.current_tolerance, 0.019)

    def test_wm_cannot_trigger_advance(self):
        self.assertTrue(wm_signals_must_not_advance({
            "wm_recovery_prob": 0.99,
            "predicted_success": 1.0,
        }))
        self.assertFalse(wm_signals_must_not_advance({
            "l_lcb": 0.1, "rho": 0.9,
        }))


class TestPBTEligibility(unittest.TestCase):
    def test_probe_not_donor_or_target(self):
        st = WorkerMSAState(worker_id=0, mode=MODE_PROBE, transaction_open=True,
                            anchor_tolerance=0.02, anchor_checkpoint="x")
        self.assertFalse(can_be_exploit_source(st))
        self.assertFalse(can_be_exploit_target(st))

    def test_donor_stage_and_anchors(self):
        cfg = _cfg(msa_min_anchor_workers=1)
        donor = WorkerMSAState(
            worker_id=0, mode=MODE_BUILD, committed_stage="STRICT",
            current_tolerance=0.02, anchor_tolerance=0.02,
            anchor_checkpoint="d.pth",
        )
        peer = WorkerMSAState(
            worker_id=1, mode=MODE_BUILD, committed_stage="STRICT",
            current_tolerance=0.02, anchor_tolerance=0.02,
            anchor_checkpoint="p.pth",
        )
        d = donor_eligible(
            donor, recipient_stage="STRICT", population=[donor, peer], cfg=cfg,
        )
        self.assertTrue(d.eligible)
        far = donor_eligible(
            donor, recipient_stage="BUILD", population=[donor, peer],
            cfg=cfg, allow_adjacent=True,
        )
        # STRICT vs BUILD are not adjacent (diff=2)
        self.assertFalse(far.eligible)

    def test_evidence_reset_on_pbt_copy(self):
        cfg = _cfg()
        donor = WorkerMSAState(
            worker_id=0, mode=MODE_BUILD, current_tolerance=0.02,
            anchor_tolerance=0.02, cycle_id="c1",
            pass_evidence={"x": 1}, consecutive_passes=3,
            decision_metrics={"l_lcb": 0.2},
        )
        recip = WorkerMSAState(
            worker_id=1, mode=MODE_BUILD, current_tolerance=0.025,
            anchor_tolerance=0.025, pass_evidence={"old": 1},
        )
        out = recipient_after_pbt_copy(recip, donor, cfg=cfg)
        self.assertEqual(out.worker_id, 1)
        self.assertEqual(out.pass_evidence, {})
        self.assertEqual(out.cycle_id, "")
        self.assertEqual(out.consecutive_passes, 0)
        self.assertEqual(out.mode, MODE_BUILD)
        self.assertIn("pbt_copy", out.decision_reason)

    def test_min_anchors(self):
        cfg = _cfg(msa_min_anchor_workers=2)
        pop = [
            WorkerMSAState(worker_id=0, mode=MODE_BUILD, anchor_tolerance=0.02,
                           anchor_checkpoint="a"),
            WorkerMSAState(worker_id=1, mode=MODE_PROBE, transaction_open=True,
                           anchor_tolerance=0.02, anchor_checkpoint="b"),
        ]
        self.assertEqual(count_reliable_anchors(pop), 1)
        self.assertFalse(population_has_min_anchors(pop, cfg))


class TestSerializationMigration(unittest.TestCase):
    def test_atomic_save_reload(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "000.json"
            st = WorkerMSAState(worker_id=0, mode=MODE_BUILD,
                                current_tolerance=0.03, anchor_tolerance=0.03)
            save_worker_state(path, st)
            loaded = load_worker_state(path)
            self.assertIsNotNone(loaded)
            self.assertEqual(loaded.mode, MODE_BUILD)
            self.assertAlmostEqual(loaded.current_tolerance, 0.03)

    def test_old_checkpoint_migration(self):
        st, logs = migrate_worker_state(
            None, worker_id=3, current_tolerance=0.022, anchor_checkpoint="tip",
            cfg=_cfg(),
        )
        self.assertEqual(st.mode, MODE_BUILD)
        self.assertEqual(st.pass_evidence, {})
        self.assertTrue(any("migrate to BUILD" in x for x in logs))
        st2, logs2 = migrate_worker_state(
            {"worker_id": 3, "mode": "WEIRD", "evidence_version": "old"},
            worker_id=3, current_tolerance=0.022, cfg=_cfg(),
        )
        self.assertEqual(st2.mode, MODE_BUILD)
        self.assertTrue(any("invalid mode" in x for x in logs2))


class TestCurriculumProposalOnly(unittest.TestCase):
    def test_fusion_shrink_is_proposal_only(self):
        from dexpbt_isaaclab.pbt.pbt_curriculum_p1 import (
            apply_env_tolerance_curriculum,
            read_curriculum_control,
            write_curriculum_control,
        )

        with tempfile.TemporaryDirectory() as td:
            ctrl_path = Path(td) / "000.json"
            write_curriculum_control(
                ctrl_path,
                CurriculumControlState(
                    allow_shrink=True,
                    shrink_token=True,
                    msa_integrated_enabled=True,
                    msa_tolerance_proposal=0.0,
                ),
            )
            new_tol, _ = apply_env_tolerance_curriculum(
                curriculum_mode="mastery",
                curriculum_control_path=ctrl_path,
                curriculum_shrink_factor=0.95,
                last_curriculum_update=0,
                frames_since_restart=10**9,
                curriculum_interval=1,
                prev_episode_successes=[1.0] * 10,
                success_tolerance=0.025,
                initial_tolerance=0.05,
                target_tolerance=0.01,
                curriculum_freeze_frames_after_restart=0,
                post_shrink_consolidation_env_frames=0,
                num_envs=1,
                pbt_single_shrink_per_window=True,
            )
            self.assertAlmostEqual(new_tol, 0.025)
            refreshed = read_curriculum_control(ctrl_path)
            self.assertGreater(float(refreshed.msa_tolerance_proposal or 0.0), 0.0)


class TestFakeInjectorGuard(unittest.TestCase):
    def test_env_guard(self):
        self.assertFalse(injector_allowed(True))
        os.environ[ENV_GUARD] = "1"
        try:
            self.assertTrue(injector_allowed(True))
            self.assertFalse(injector_allowed(False))
            with tempfile.TemporaryDirectory() as td:
                p = write_fake_decisions(
                    td, advance_workers=[0], hold_workers=[1],
                )
                doc = json.loads(p.read_text())
                self.assertTrue(doc["test_only"])
                self.assertEqual(doc["decisions"]["0"]["action"], "advance")
        finally:
            os.environ.pop(ENV_GUARD, None)


class TestIntegrationFlows(unittest.TestCase):
    def test_build_probe_advance(self):
        with tempfile.TemporaryDirectory() as td:
            hook = MSAFusionHook(td, _cfg(), num_workers=2)
            st = hook.load_or_migrate(0, current_tolerance=0.025)
            self.assertEqual(st.mode, MODE_BUILD)
            self.assertTrue(hook.readiness_allows_probe(st, frame=10**9))
            st = hook.start_probe(st, frame=100)
            self.assertEqual(st.mode, MODE_PROBE)
            st = hook.apply_decision_from_metrics(
                st, frame=200, l_lcb=0.05, rho=0.95, rho_available=True,
                promote=True, reason="ok", anchor_checkpoint="n.pth",
            )
            self.assertEqual(st.mode, MODE_BUILD)
            self.assertLess(st.current_tolerance, 0.025)

    def test_build_probe_rollback_hold(self):
        with tempfile.TemporaryDirectory() as td:
            hook = MSAFusionHook(td, _cfg(msa_hold_cooldown_frames=1000))
            st = hook.load_or_migrate(1, current_tolerance=0.02)
            st = hook.start_probe(st, frame=10)
            st = hook.apply_decision_from_metrics(
                st, frame=20, l_lcb=-0.2, rho=0.5, rho_available=True,
                promote=False, reason="fail",
            )
            self.assertEqual(st.mode, MODE_HOLD)
            self.assertAlmostEqual(st.current_tolerance, st.anchor_tolerance)
            self.assertFalse(hook.readiness_allows_probe(st, frame=21))

    def test_interrupt_resume_probe(self):
        with tempfile.TemporaryDirectory() as td:
            hook = MSAFusionHook(td, _cfg())
            st = hook.load_or_migrate(0, current_tolerance=0.02)
            st = hook.start_probe(st, frame=10, cycle_id="cycA")
            hook.save(st)
            reloaded = load_worker_state(Path(td) / "msa_worker_state" / "000.json")
            self.assertEqual(reloaded.mode, MODE_PROBE)
            self.assertTrue(reloaded.transaction_open)
            self.assertEqual(reloaded.cycle_id, "cycA")

    def test_eval_failure_fail_closed(self):
        with tempfile.TemporaryDirectory() as td:
            hook = MSAFusionHook(td, _cfg(msa_fail_closed=True))
            st = hook.load_or_migrate(0, current_tolerance=0.02)
            st = hook.start_probe(st, frame=1)
            # Missing rho_available -> not advance.
            st = hook.apply_decision_from_metrics(
                st, frame=2, l_lcb=0.5, rho=1.0, rho_available=False,
                promote=True, reason="eval_incomplete",
            )
            self.assertEqual(st.mode, MODE_HOLD)

    def test_pbt_msa_conflict_probe_blocks(self):
        st = WorkerMSAState(worker_id=0, mode=MODE_PROBE, transaction_open=True)
        self.assertFalse(can_be_exploit_target(st))
        self.assertFalse(can_be_exploit_source(st))

    def test_independent_worker_tolerances(self):
        with tempfile.TemporaryDirectory() as td:
            hook = MSAFusionHook(td, _cfg(), num_workers=2)
            a = hook.load_or_migrate(0, current_tolerance=0.03)
            b = hook.load_or_migrate(1, current_tolerance=0.018)
            a = hook.start_probe(a, frame=1)
            a = hook.apply_decision_from_metrics(
                a, frame=2, l_lcb=0.1, rho=0.95, rho_available=True,
                promote=True, reason="a", anchor_checkpoint="a",
            )
            self.assertNotAlmostEqual(a.current_tolerance, b.current_tolerance)

    def test_no_promote_safe_stop(self):
        with tempfile.TemporaryDirectory() as td:
            hook = MSAFusionHook(td, _cfg(), num_workers=2)
            states = {}
            for w in (0, 1):
                st = hook.load_or_migrate(w, current_tolerance=0.02)
                st = hook.start_probe(st, frame=1)
                states[w] = st
            doc = {
                "intervention_id": "none_pass",
                "decisions": {
                    "0": {"action": "hold", "l_lcb": -0.1, "rho": 0.5,
                          "rho_available": True},
                    "1": {"action": "hold", "l_lcb": -0.2, "rho": 0.4,
                          "rho_available": True},
                },
            }
            states = hook.apply_sidecar_decision_doc(doc, states, frame_by_worker={0: 2, 1: 2})
            self.assertTrue(all(s.mode == MODE_HOLD for s in states.values()))
            # Anchors preserved (no forced tighten).
            self.assertTrue(all(
                abs(s.current_tolerance - s.anchor_tolerance) < 1e-12
                for s in states.values()
            ))

    def test_old_profile_path_unchanged_flag(self):
        ctrl = CurriculumControlState(msa_integrated_enabled=False)
        self.assertFalse(ctrl.msa_integrated_enabled)

    def test_sidecar_decision_consume(self):
        with tempfile.TemporaryDirectory() as td:
            os.environ[ENV_GUARD] = "1"
            try:
                write_fake_decisions(td, advance_workers=[0], hold_workers=[1])
                hook = MSAFusionHook(td, _cfg(), num_workers=2)
                states = {
                    0: hook.load_or_migrate(0, current_tolerance=0.02),
                    1: hook.load_or_migrate(1, current_tolerance=0.02),
                }
                for w in states:
                    states[w] = hook.start_probe(states[w], frame=1)
                doc = hook.consume_sidecar_decision_file()
                self.assertIsNotNone(doc)
                states = hook.apply_sidecar_decision_doc(
                    doc, states, frame_by_worker={0: 5, 1: 5},
                )
                self.assertEqual(states[0].mode, MODE_BUILD)
                self.assertEqual(states[1].mode, MODE_HOLD)
            finally:
                os.environ.pop(ENV_GUARD, None)

    def test_fingerprint_commit_rollback(self):
        st = WorkerMSAState(
            worker_id=0, mode=MODE_PROBE, current_tolerance=0.02,
            anchor_tolerance=0.02, anchor_checkpoint="pre.pth",
            anchor_fingerprint="fp_pre", transaction_open=True,
            pending_target_tolerance=0.019,
        )
        advanced = apply_advance(
            st, new_tol=0.019, frame=1, anchor_checkpoint="post.pth",
            metrics={}, cfg=_cfg(),
        )
        self.assertEqual(advanced.anchor_checkpoint, "post.pth")
        st2 = WorkerMSAState(
            worker_id=1, mode=MODE_PROBE, current_tolerance=0.019,
            anchor_tolerance=0.02, anchor_checkpoint="pre.pth",
            transaction_open=True,
        )
        rolled = apply_rollback_hold(st2, frame=1, cooldown_frames=1)
        self.assertAlmostEqual(rolled.current_tolerance, 0.02)
        self.assertEqual(rolled.anchor_checkpoint, "pre.pth")


if __name__ == "__main__":
    unittest.main()
