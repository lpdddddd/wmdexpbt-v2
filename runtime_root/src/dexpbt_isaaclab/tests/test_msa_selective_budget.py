"""Budget admission + worker-local selective promotion + real-data replay."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from dexpbt_isaaclab.msa.budget import (
    admit_cycle_budget,
    clear_reservation,
    load_reservation,
    worst_case_remaining_cycle_dose,
    write_no_new_cycle,
    write_reservation,
    NO_NEW_CYCLE_MARKER,
    RESERVATION_MARKER,
)
from dexpbt_isaaclab.msa.decision import (
    PROMOTION_MODE_GLOBAL,
    PROMOTION_MODE_SELECTIVE,
    RetentionResult,
    decide_promotion,
    decide_selective_promotion,
    panel_bootstrap_lower_quantile,
    worker_bootstrap_lower_quantile,
)
from dexpbt_isaaclab.msa import dry_run as dr


ANALYSIS = Path(
    "/root/autodl-tmp/wmdexpbt/results/analysis/msa_stateful_n16_20260720")
LAUNCHER_DIR = Path(
    "/root/autodl-tmp/wmdexpbt/results/launchers/msa_probe")


class TestBudgetDoseFormula(unittest.TestCase):
    def test_worst_case_dose_is_derived_not_guessed(self):
        # common=(packet+tick) + second=(packet+tick) + commit=tick
        # with packet==tick==80M → 400M
        self.assertEqual(worst_case_remaining_cycle_dose(), 400_000_000)
        self.assertEqual(
            worst_case_remaining_cycle_dose(packet_frames=80_000_000,
                                           committed_tick_frames=80_000_000),
            2 * (80_000_000 + 80_000_000) + 80_000_000)

    def test_admit_filters_ineligible_without_busy(self):
        tip = 5_000_000_000
        cap = tip + 480_000_000
        # w0 has only 100M remaining (<400M); w1 has 450M (>=400M)
        adm = admit_cycle_budget(
            intervention_id="msa_t",
            due_workers=[0, 1],
            latest_frame_by_worker={0: tip + 380_000_000,
                                    1: tip + 30_000_000},
            cap_target_by_worker={0: cap, 1: cap},
        )
        self.assertEqual(adm.budget_eligible_workers, [1])
        self.assertEqual(adm.budget_ineligible_workers, [0])
        self.assertEqual(adm.required_reserve, 400_000_000)

    def test_all_ineligible(self):
        tip = 5_000_000_000
        cap = tip + 480_000_000
        adm = admit_cycle_budget(
            intervention_id="msa_t",
            due_workers=[0, 1, 2],
            latest_frame_by_worker={w: tip + 400_000_000 for w in range(3)},
            cap_target_by_worker={w: cap for w in range(3)},
        )
        self.assertEqual(adm.budget_eligible_workers, [])
        self.assertEqual(sorted(adm.budget_ineligible_workers), [0, 1, 2])


class TestBudgetControllerIntegration(unittest.TestCase):
    TIP = 5_000_000_000
    CAP = 480_000_000
    TICK = 80_000_000
    RESERVE = 400_000_000

    def test_insufficient_budget_skips_pre_eval_and_busy(self):
        tip = self.TIP
        caps = {w: tip + self.CAP for w in range(6)}
        from dexpbt_isaaclab.msa.controller import WorkerView
        with tempfile.TemporaryDirectory() as td:
            run_dir = Path(td)
            world, ctl, views = dr.make_world(
                cap_target_by_worker=caps,
                run_dir=run_dir,
            )
            for i in range(6):
                world.frame[i] = tip + 450_000_000
            views2 = [
                WorkerView(worker_id=i, lineage_id=f"lin_{i}",
                           active_tau=world.taus[i],
                           frame=world.frame[i],
                           source_checkpoint_id=f"ws/{i:03d}/000160.pth")
                for i in range(6)
            ]
            result = ctl.run_cycle(views2)
            self.assertEqual(result.first_blocking_gate,
                             "all_budget_ineligible")
            self.assertFalse(result.promoted)
            self.assertIsNone(world.busy)
            self.assertFalse((run_dir / RESERVATION_MARKER).exists())
            self.assertTrue((run_dir / NO_NEW_CYCLE_MARKER).exists())
            self.assertEqual(world.held, set())
            self.assertEqual(ctl.machine.phase.value, "NORMAL")
            self.assertFalse(world.snapshots)
            self.assertFalse(world.eval_calls)
            self.assertFalse(world.pin_log)

    def test_no_new_cycle_marker_helper(self):
        with tempfile.TemporaryDirectory() as td:
            run_dir = Path(td)
            write_no_new_cycle(run_dir, {"reason": "all_budget_ineligible",
                                         "due_workers": [0, 1]})
            self.assertTrue((run_dir / NO_NEW_CYCLE_MARKER).exists())
            body = json.loads((run_dir / NO_NEW_CYCLE_MARKER).read_text())
            self.assertEqual(body["reason"], "all_budget_ineligible")


class TestFrozenAdmissionBarrier(unittest.TestCase):
    """Admission must use post-hold frozen frames, not stale due-detection."""

    TIP = 5_000_000_000
    CAP = 480_000_000
    TICK = 80_000_000

    def _views_at(self, world, frame: int):
        from dexpbt_isaaclab.msa.controller import WorkerView
        return [
            WorkerView(worker_id=i, lineage_id=f"lin_{i}",
                       active_tau=world.taus[i], frame=frame,
                       source_checkpoint_id=f"ws/{i:03d}/000160.pth")
            for i in range(6)
        ]

    def test_admission_uses_post_hold_frame_when_tick_slips(self):
        tip = self.TIP
        caps = {w: tip + self.CAP for w in range(6)}
        stale = tip + self.TICK          # remaining == 400M if used
        live = tip + 2 * self.TICK       # remaining == 320M after hold
        with tempfile.TemporaryDirectory() as td:
            run_dir = Path(td)
            world, ctl, _ = dr.make_world(
                cap_target_by_worker=caps, run_dir=run_dir)
            for i in range(6):
                world.frame[i] = live
            result = ctl.run_cycle(self._views_at(world, stale))
            self.assertEqual(result.first_blocking_gate,
                             "all_budget_ineligible")
            rem = result.budget_admission["remaining_at_admission"]
            for w in range(6):
                self.assertEqual(int(rem[str(w)]),
                                 self.CAP - 2 * self.TICK)
            self.assertFalse((run_dir / RESERVATION_MARKER).exists())
            self.assertTrue((run_dir / NO_NEW_CYCLE_MARKER).exists())
            self.assertIsNone(world.busy)
            self.assertEqual(world.held, set())

    def test_exact_400m_remaining_becomes_ineligible_after_slip(self):
        tip = self.TIP
        caps = {w: tip + self.CAP for w in range(6)}
        # stale view: exactly 400M remaining → would admit on stale frame
        stale = tip + self.TICK
        # hold refresh: +80M slipped → 320M remaining → refuse
        live = tip + 2 * self.TICK
        with tempfile.TemporaryDirectory() as td:
            run_dir = Path(td)
            world, ctl, _ = dr.make_world(
                cap_target_by_worker=caps, run_dir=run_dir)
            for i in range(6):
                world.frame[i] = live
            # prove stale frame alone would be eligible
            from dexpbt_isaaclab.msa.budget import admit_cycle_budget
            stale_adm = admit_cycle_budget(
                intervention_id="stale",
                due_workers=list(range(6)),
                latest_frame_by_worker={w: stale for w in range(6)},
                cap_target_by_worker=caps)
            self.assertEqual(stale_adm.budget_eligible_workers,
                             list(range(6)))
            result = ctl.run_cycle(self._views_at(world, stale))
            self.assertEqual(result.first_blocking_gate,
                             "all_budget_ineligible")
            self.assertEqual(
                result.budget_admission["budget_eligible_workers"], [])

    def test_all_ineligible_leaves_no_cycle_artifacts(self):
        tip = self.TIP
        caps = {w: tip + self.CAP for w in range(6)}
        with tempfile.TemporaryDirectory() as td:
            run_dir = Path(td)
            world, ctl, views = dr.make_world(
                cap_target_by_worker=caps, run_dir=run_dir)
            for i in range(6):
                world.frame[i] = tip + 450_000_000
            views2 = self._views_at(world, world.frame[0])
            result = ctl.run_cycle(views2)
            self.assertEqual(result.first_blocking_gate,
                             "all_budget_ineligible")
            self.assertIsNone(world.busy)
            self.assertEqual(world.held, set())
            self.assertFalse((run_dir / RESERVATION_MARKER).exists())
            self.assertTrue((run_dir / NO_NEW_CYCLE_MARKER).exists())
            self.assertFalse(world.snapshots)
            self.assertFalse(world.eval_calls)
            self.assertFalse(world.pin_log)
            self.assertEqual(ctl.machine.history, [])
            # hold then release happened (admission barrier), never PRE_EVAL
            hold_ops = [op for op, _ in world.hold_log]
            self.assertIn("hold", hold_ops)
            self.assertIn("release", hold_ops)


class TestFullCycleDose(unittest.TestCase):
    TIP = 5_000_000_000
    CAP = 480_000_000
    TICK = 80_000_000

    def test_observed_cycle_dose_includes_commit_tick(self):
        tip = self.TIP
        caps = {w: tip + self.CAP for w in range(6)}
        with tempfile.TemporaryDirectory() as td:
            run_dir = Path(td)
            world, ctl, views = dr.make_world(
                cap_target_by_worker=caps, run_dir=run_dir)
            result = ctl.run_cycle(views)
            self.assertNotEqual(result.first_blocking_gate,
                                "all_budget_ineligible")
            cycle = world.state_store[0]["msa_cycle"]
            for w in result.due_workers:
                start = int(cycle["start_frames"][str(w)])
                final = int(result.final_committed_frame_by_worker[w])
                dose = int(result.observed_cycle_dose_by_worker[w])
                self.assertEqual(dose, final - start)
                self.assertEqual(
                    int(cycle["final_committed_frame_by_worker"][str(w)]),
                    final)
                self.assertEqual(
                    int(cycle["observed_cycle_dose_by_worker"][str(w)]), dose)
                # probe packets (2 ticks) + commit confirmation (1 tick)
                self.assertEqual(dose, 3 * self.TICK)
                self.assertGreater(dose, 2 * self.TICK)

    def test_final_frame_over_reserved_ceiling_fails_closed(self):
        from dexpbt_isaaclab.msa.controller import CycleFailure
        tip = self.TIP
        caps = {w: tip + self.CAP for w in range(6)}
        with tempfile.TemporaryDirectory() as td:
            run_dir = Path(td)
            world, ctl, views = dr.make_world(
                cap_target_by_worker=caps, run_dir=run_dir)
            real = world.commit_tolerance

            def overshoot(wid, tol):
                real(wid, tol)
                # after a normal ~240M dose, ceiling is tip+400M; push past it
                world.frame[wid] += 200_000_000

            world.commit_tolerance = overshoot
            ctl._commit = overshoot
            with self.assertRaises(CycleFailure) as ctx:
                ctl.run_cycle(views)
            self.assertIn("reserved ceiling", str(ctx.exception))
            self.assertIsNone(world.busy)
            self.assertEqual(world.held, set())
            self.assertFalse((run_dir / RESERVATION_MARKER).exists())

    def test_committing_resume_records_full_final_frame_dose(self):
        tip = self.TIP
        caps = {w: tip + self.CAP for w in range(6)}
        with tempfile.TemporaryDirectory() as td:
            run_dir = Path(td)
            world, ctl, views = dr.make_world(
                cap_target_by_worker=caps,
                run_dir=run_dir,
                crash_commit_workers={2},
                crash_hard=True,
            )
            with self.assertRaises(dr.HardKill):
                ctl.run_cycle(views)
            cycle = world.state_store[0]["msa_cycle"]
            self.assertEqual(cycle["status"], "COMMITTING")
            start_frames = {int(k): int(v)
                            for k, v in cycle["start_frames"].items()}
            world.crash_hard = False
            world.crash_commit_workers = set()
            world2, ctl2, views2 = dr.make_world(
                world=world, cap_target_by_worker=caps, run_dir=run_dir)
            pending = ctl2.pending_cycle(list(range(6)))
            result = ctl2.run_cycle(views2, resume_cycle=pending)
            self.assertTrue(result.resumed)
            self.assertIsNotNone(result.final_committed_frame_by_worker)
            self.assertIsNotNone(result.observed_cycle_dose_by_worker)
            done = world.state_store[0]["msa_cycle"]
            self.assertEqual(done["status"], "DONE")
            for w in result.due_workers:
                final = int(result.final_committed_frame_by_worker[w])
                dose = int(result.observed_cycle_dose_by_worker[w])
                self.assertEqual(dose, final - start_frames[w])
                self.assertEqual(dose, 3 * self.TICK)
                self.assertEqual(
                    int(done["final_committed_frame_by_worker"][str(w)]),
                    final)


class TestGlobalPanelDecisionUnchanged(unittest.TestCase):
    def test_global_panel_decision_replay_unchanged(self):
        path = ANALYSIS / "SELECTIVE_REPLAY.json"
        if not path.is_file():
            raise unittest.SkipTest(f"missing {path}")
        replay = json.loads(path.read_text())
        for rid, gate, promote in (
            ("5fa1236a", "panel_lower_quantile", False),
            ("af1468d9", "same_tau_retention_w5", False),
        ):
            c = next(r for r in replay if r["intervention_id"].endswith(rid))
            paired = {int(w): info["paired_diffs"]
                      for w, info in c["workers"].items()}
            retention = {}
            for w, info in c["workers"].items():
                wi = int(w)
                retention[wi] = RetentionResult(
                    tau=info["tau"], p_pre=info["p_pre"],
                    p_post=info["p_post"], rho=info["rho"],
                    retention_unavailable=(info["p_pre"] is None
                                           or info["p_pre"] < 0.01))
            panel = panel_bootstrap_lower_quantile(paired)
            glob = decide_promotion(panel, retention, 0.88)
            self.assertEqual(glob.promote, promote)
            self.assertEqual(glob.first_blocking_gate, gate)
            self.assertEqual(c["global_panel"]["promoted"], promote)
            self.assertEqual(c["global_panel"]["first_blocking_gate"], gate)


class TestSelectiveDecisionUnit(unittest.TestCase):
    def _ret(self, rho, p_pre=0.5, p_post=None):
        p_post = p_post if p_post is not None else rho * p_pre
        return RetentionResult(tau=0.02, p_pre=p_pre, p_post=p_post,
                               rho=rho, retention_unavailable=False)

    def test_weak_worker_does_not_veto_two_strong(self):
        # Strong w0/w1: positive L_mid diffs; weak w5: positive L_mid but low rho
        paired = {
            0: [0.2] * 16,
            1: [0.18] * 16,
            5: [0.1] * 16,
        }
        retention = {
            0: self._ret(1.26),
            1: self._ret(1.36),
            5: self._ret(0.75),
        }
        tau = {0: 0.024, 1: 0.020, 5: 0.020}
        tau_plus = {w: 0.95 * tau[w] for w in tau}
        # global would fail on w5
        panel = panel_bootstrap_lower_quantile(paired)
        glob = decide_promotion(panel, retention, 0.88)
        self.assertFalse(glob.promote)
        self.assertEqual(glob.first_blocking_gate, "same_tau_retention_w5")
        # selective promotes w0/w1, rejects w5
        sel = decide_selective_promotion(
            paired, retention, tau, tau_plus, 0.88,
            population_worker_ids=list(range(6)), panel_stat=panel)
        self.assertEqual(sel.selected_for_promotion, [0, 1])
        self.assertEqual(sel.rejection_reason_by_worker[5],
                         "same_tau_retention")
        self.assertGreaterEqual(len(sel.anchor_workers), 3)
        self.assertEqual(sel.final_tol_by_worker[0], tau_plus[0])
        self.assertEqual(sel.final_tol_by_worker[5], tau[5])

    def test_lmid_ci_nonpositive_rejects(self):
        paired = {0: [-0.1] * 16, 1: [0.2] * 16}
        retention = {0: self._ret(1.1), 1: self._ret(1.1)}
        tau = {0: 0.02, 1: 0.02}
        tau_plus = {0: 0.019, 1: 0.019}
        sel = decide_selective_promotion(
            paired, retention, tau, tau_plus, 0.88,
            population_worker_ids=list(range(6)))
        self.assertFalse(sel.eligible_by_worker[0])
        self.assertEqual(sel.rejection_reason_by_worker[0],
                         "L_mid_lower_quantile<=0")
        self.assertEqual(sel.selected_for_promotion, [1])

    def test_rho_below_threshold_rejects(self):
        paired = {0: [0.2] * 16}
        retention = {0: self._ret(0.87)}
        tau = {0: 0.02}; tau_plus = {0: 0.019}
        sel = decide_selective_promotion(
            paired, retention, tau, tau_plus, 0.88,
            population_worker_ids=list(range(6)))
        self.assertFalse(sel.eligible_by_worker[0])
        self.assertEqual(sel.rejection_reason_by_worker[0],
                         "same_tau_retention")

    def test_low_p_pre_rejects(self):
        paired = {0: [0.2] * 16}
        retention = {0: RetentionResult(
            tau=0.02, p_pre=0.005, p_post=0.01, rho=None,
            retention_unavailable=True, reason="p_pre low")}
        tau = {0: 0.02}; tau_plus = {0: 0.019}
        sel = decide_selective_promotion(
            paired, retention, tau, tau_plus, 0.88,
            population_worker_ids=list(range(6)))
        self.assertFalse(sel.eligible_by_worker[0])
        self.assertIn("p_pre", sel.rejection_reason_by_worker[0])

    def test_top3_tiebreak_and_min_anchors(self):
        # 6 eligible with distinct L_mid; must keep >=3 anchors → max 3 promote
        paired = {w: [0.3 - 0.01 * w] * 16 for w in range(6)}
        retention = {w: self._ret(1.0 + 0.01 * w) for w in range(6)}
        tau = {w: 0.02 for w in range(6)}
        tau_plus = {w: 0.019 for w in range(6)}
        sel = decide_selective_promotion(
            paired, retention, tau, tau_plus, 0.88,
            population_worker_ids=list(range(6)))
        self.assertEqual(len(sel.selected_for_promotion), 3)
        self.assertEqual(len(sel.anchor_workers), 3)
        # deterministic: highest L_mid_lo first → w0, w1, w2
        self.assertEqual(sel.selected_for_promotion, [0, 1, 2])

    def test_global_mode_unchanged_negative_panel(self):
        paired = {w: [-0.1] * 8 for w in range(6)}
        retention = {w: self._ret(1.0) for w in range(6)}
        panel = panel_bootstrap_lower_quantile(paired)
        d = decide_promotion(panel, retention, 0.88)
        self.assertFalse(d.promote)
        self.assertEqual(d.first_blocking_gate, "panel_lower_quantile")


class TestSelectiveDryRun(unittest.TestCase):
    def test_mixed_promote_stay_durable_commit(self):
        world, ctl, result = dr.run_selective_dry_run()
        self.assertEqual(ctl.promotion_mode, PROMOTION_MODE_SELECTIVE)
        self.assertIsNotNone(result.selected_for_promotion)
        selected = list(result.selected_for_promotion or [])
        # Selected workers end at tau_plus; other due workers stay at tau
        for w in selected:
            self.assertAlmostEqual(
                world.live_tolerance[w],
                result.tau_plus_by_worker[w], places=9)
        for w in result.due_workers:
            if w not in selected:
                self.assertAlmostEqual(
                    world.live_tolerance[w],
                    result.tau_by_worker[w], places=9)
        self.assertGreaterEqual(len(result.anchor_workers or []), 3)

    def test_partial_commit_resume_selective(self):
        world, ctl, views = dr.make_world(
            promotion_mode=PROMOTION_MODE_SELECTIVE,
            crash_commit_workers={2},
            crash_hard=True,
        )
        with self.assertRaises(dr.HardKill):
            ctl.run_cycle(views)
        cycle = world.state_store[0]["msa_cycle"]
        self.assertEqual(cycle["status"], "COMMITTING")
        world.crash_hard = False
        world.crash_commit_workers = set()
        world2, ctl2, views2 = dr.make_world(
            world=world, promotion_mode=PROMOTION_MODE_SELECTIVE)
        pending = ctl2.pending_cycle(list(range(6)))
        self.assertIsNotNone(pending)
        self.assertEqual(pending["status"], "COMMITTING")
        result = ctl2.run_cycle(views2, resume_cycle=pending)
        self.assertTrue(result.resumed)
        # all due workers eventually acked
        final = world.state_store[0]["msa_cycle"]
        self.assertEqual(final["status"], "DONE")
        self.assertTrue(all(final["commit_acks_by_worker"][str(w)]
                            for w in final["due_workers"]))


class TestRealDataReplay(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = ANALYSIS / "SELECTIVE_REPLAY.json"
        if not path.is_file():
            raise unittest.SkipTest(f"missing {path}")
        cls.replay = json.loads(path.read_text())

    def test_cycle1_global_false_selective_promotes_w2_w3(self):
        c1 = next(r for r in self.replay
                  if r["intervention_id"].endswith("5fa1236a"))
        self.assertFalse(c1["global_panel"]["promoted"])
        self.assertEqual(c1["global_panel"]["first_blocking_gate"],
                         "panel_lower_quantile")
        paired = {int(w): info["paired_diffs"]
                  for w, info in c1["workers"].items()}
        retention = {}
        tau = {}; tau_plus = {}
        for w, info in c1["workers"].items():
            wi = int(w)
            retention[wi] = RetentionResult(
                tau=info["tau"], p_pre=info["p_pre"], p_post=info["p_post"],
                rho=info["rho"],
                retention_unavailable=(info["p_pre"] is None
                                       or info["p_pre"] < 0.01))
            tau[wi] = info["tau"]; tau_plus[wi] = info["tau_plus"]
        panel = panel_bootstrap_lower_quantile(paired)
        glob = decide_promotion(panel, retention, 0.88)
        self.assertFalse(glob.promote)
        sel = decide_selective_promotion(
            paired, retention, tau, tau_plus, 0.88,
            population_worker_ids=list(range(6)), panel_stat=panel)
        self.assertEqual(sorted(sel.selected_for_promotion), [2, 3])
        # w5 fails at least one individual gate (L_mid CI and/or rho)
        self.assertFalse(sel.eligible_by_worker[5])
        self.assertIn(sel.rejection_reason_by_worker[5],
                      ("L_mid_lower_quantile<=0", "same_tau_retention",
                       "p_pre<0.01_or_unavailable"))

    def test_cycle2_global_false_selective_promotes_w0_w1_rejects_w5(self):
        c2 = next(r for r in self.replay
                  if r["intervention_id"].endswith("af1468d9"))
        self.assertFalse(c2["global_panel"]["promoted"])
        self.assertEqual(c2["global_panel"]["first_blocking_gate"],
                         "same_tau_retention_w5")
        paired = {int(w): info["paired_diffs"]
                  for w, info in c2["workers"].items()}
        retention = {}
        tau = {}; tau_plus = {}
        for w, info in c2["workers"].items():
            wi = int(w)
            retention[wi] = RetentionResult(
                tau=info["tau"], p_pre=info["p_pre"], p_post=info["p_post"],
                rho=info["rho"], retention_unavailable=False)
            tau[wi] = info["tau"]; tau_plus[wi] = info["tau_plus"]
        panel = panel_bootstrap_lower_quantile(paired)
        glob = decide_promotion(panel, retention, 0.88)
        self.assertFalse(glob.promote)
        self.assertEqual(glob.first_blocking_gate, "same_tau_retention_w5")
        sel = decide_selective_promotion(
            paired, retention, tau, tau_plus, 0.88,
            population_worker_ids=list(range(6)), panel_stat=panel)
        self.assertEqual(sorted(sel.selected_for_promotion), [0, 1])
        self.assertEqual(sel.rejection_reason_by_worker[5],
                         "same_tau_retention")
        self.assertNotIn(5, sel.selected_for_promotion)


class TestArmConfigsUnchanged(unittest.TestCase):
    def test_original_msa_stateful_is_global_panel(self):
        from dexpbt_isaaclab.msa.arms import arm_config, TipCheckpoint
        tips = [
            TipCheckpoint(worker_id=i, iteration=160, frame=5_000_000_000,
                          checkpoint_path=f"/tmp/{i}.pth",
                          live_tolerance=0.02, lineage_id=f"lin{i}")
            for i in range(6)
        ]
        cfg = arm_config("MSA_STATEFUL", tips, "/tmp/src")
        self.assertNotEqual(cfg["msa"].get("promotion_mode"),
                            PROMOTION_MODE_SELECTIVE)
        sel = arm_config("MSA_SELECTIVE_STATEFUL", tips, "/tmp/src")
        self.assertEqual(sel["msa"]["promotion_mode"],
                         PROMOTION_MODE_SELECTIVE)
        self.assertEqual(sel["msa"]["max_promote_per_cycle"], 3)
        self.assertEqual(sel["msa"]["min_population_anchors"], 3)


class TestGuardReservationOverrun(unittest.TestCase):
    def test_busy_over_reserved_ceiling_fails(self):
        from dexpbt_isaaclab.msa.continuation_guard import cap_action
        # baseline helper still works
        entry = {"frame": 5_000_000_000}
        meta = {"frame": 5_000_000_000 + 480_000_000 + 1}
        # without slack exceed → park at exact target handled elsewhere
        self.assertEqual(
            cap_action(entry, {"frame": 5_000_000_000 + 480_000_000}),
            "park")
        self.assertEqual(
            cap_action(entry, {"frame": 5_000_000_000 + 480_000_000
                                       + 80_000_000 + 1}),
            "fail")


class TestCompletionOrderingAndNoExtraCommit(unittest.TestCase):
    """Dose validation must finish before DONE/NORMAL; cleanup after
    durable commits must never emit an extra confirmation tick."""

    TIP = 5_000_000_000
    CAP = 480_000_000
    TICK = 80_000_000

    def _caps(self):
        tip = self.TIP
        return {w: tip + self.CAP for w in range(6)}

    def _install_overshoot(self, world, ctl, extra: int = 200_000_000):
        real = world.commit_tolerance

        def overshoot(wid, tol):
            real(wid, tol)
            world.frame[wid] += int(extra)

        world.commit_tolerance = overshoot
        ctl._commit = overshoot

    def test_dose_fail_stays_committing_until_cleanup_no_extra_tick(self):
        from dexpbt_isaaclab.msa.controller import CycleFailure
        from dexpbt_isaaclab.msa.state_machine import MSAPhase
        with tempfile.TemporaryDirectory() as td:
            run_dir = Path(td)
            world, ctl, views = dr.make_world(
                cap_target_by_worker=self._caps(), run_dir=run_dir)
            seen = []
            frames_at_validate = {}
            real_fin = ctl._finalize_cycle_dose

            def wrap(due, start_frames, cycle, admission):
                seen.append((str(cycle.get("status")),
                             ctl.machine.phase.value,
                             int(world.commit_calls)))
                frames_at_validate.update(
                    {w: int(world.frame[w]) for w in due})
                return real_fin(due, start_frames, cycle, admission)

            ctl._finalize_cycle_dose = wrap
            self._install_overshoot(world, ctl)
            with self.assertRaises(CycleFailure) as ctx:
                ctl.run_cycle(views)
            self.assertTrue(seen)
            status, phase, commits_at_val = seen[0]
            self.assertEqual(status, "COMMITTING")
            self.assertIn(phase, (MSAPhase.DECIDE.value,
                                  MSAPhase.PROMOTE.value))
            self.assertEqual(commits_at_val, 6)
            self.assertEqual(world.commit_calls, 6)
            for w, fr in frames_at_validate.items():
                self.assertEqual(int(world.frame[w]), fr)
            cycle = world.state_store[0]["msa_cycle"]
            self.assertEqual(cycle["status"], "FAILED")
            self.assertTrue(cycle.get("durable_commits_completed"))
            self.assertFalse(cycle.get("rollback_claimed", True))
            self.assertTrue(ctx.exception.durable_commits_completed)
            self.assertEqual(
                cycle["commit_acks_by_worker"],
                {str(w): True for w in range(6)})
            self.assertIn("final_committed_frame_by_worker", cycle)
            self.assertIn("observed_cycle_dose_by_worker", cycle)

    def test_stay_decision_dose_fail_no_repeat_commit(self):
        from dexpbt_isaaclab.msa import controller as cmod
        from dexpbt_isaaclab.msa.controller import CycleFailure
        from dexpbt_isaaclab.msa.decision import PromotionDecision
        with tempfile.TemporaryDirectory() as td:
            run_dir = Path(td)
            world, ctl, views = dr.make_world(
                cap_target_by_worker=self._caps(), run_dir=run_dir)

            def force_stay(panel_stat, retention_by_worker,
                           retention_threshold, extra_safety_gates=None):
                return PromotionDecision(
                    promote=False,
                    first_blocking_gate="injected_stay",
                    panel_lower_quantile=float(
                        panel_stat.get("lower_quantile", 0.0)),
                    retention={w: vars(r)
                               for w, r in retention_by_worker.items()},
                )

            self._install_overshoot(world, ctl)
            frames_snap = {}
            real_fin = ctl._finalize_cycle_dose

            def wrap(due, start_frames, cycle, admission):
                frames_snap.update({w: int(world.frame[w]) for w in due})
                return real_fin(due, start_frames, cycle, admission)

            ctl._finalize_cycle_dose = wrap
            orig = cmod.decide_promotion
            cmod.decide_promotion = force_stay
            try:
                with self.assertRaises(CycleFailure):
                    ctl.run_cycle(views)
            finally:
                cmod.decide_promotion = orig
            cycle = world.state_store[0]["msa_cycle"]
            self.assertEqual(cycle["decision"], "stay")
            self.assertEqual(world.commit_calls, 6)
            for w, fr in frames_snap.items():
                self.assertEqual(int(world.frame[w]), fr)
            self.assertFalse(cycle.get("rollback_claimed", True))
            self.assertTrue(cycle.get("durable_commits_completed"))

    def test_selective_promote_dose_fail_no_false_rollback(self):
        from dexpbt_isaaclab.msa.controller import CycleFailure
        with tempfile.TemporaryDirectory() as td:
            run_dir = Path(td)
            world, ctl, views = dr.make_world(
                promotion_mode=PROMOTION_MODE_SELECTIVE,
                cap_target_by_worker=self._caps(), run_dir=run_dir)
            self._install_overshoot(world, ctl)
            frames_snap = {}
            real_fin = ctl._finalize_cycle_dose

            def wrap(due, start_frames, cycle, admission):
                frames_snap.update({w: int(world.frame[w]) for w in due})
                return real_fin(due, start_frames, cycle, admission)

            ctl._finalize_cycle_dose = wrap
            with self.assertRaises(CycleFailure) as ctx:
                ctl.run_cycle(views)
            cycle = world.state_store[0]["msa_cycle"]
            self.assertEqual(cycle["decision"], "promote")
            self.assertTrue(cycle.get("selected_for_promotion"))
            self.assertEqual(world.commit_calls, 6)
            for w, fr in frames_snap.items():
                self.assertEqual(int(world.frame[w]), fr)
            self.assertFalse(cycle.get("rollback_claimed", True))
            self.assertIn("applied_final_tol_by_worker", cycle)
            self.assertTrue(ctx.exception.durable_commits_completed)

    def test_committing_resume_dose_fail_no_recommit_acked(self):
        from dexpbt_isaaclab.msa.controller import CycleFailure
        with tempfile.TemporaryDirectory() as td:
            run_dir = Path(td)
            world, ctl, views = dr.make_world(
                cap_target_by_worker=self._caps(),
                run_dir=run_dir,
                crash_commit_workers={4, 5},
                crash_hard=True,
            )
            with self.assertRaises(dr.HardKill):
                ctl.run_cycle(views)
            cycle = world.state_store[0]["msa_cycle"]
            self.assertEqual(cycle["status"], "COMMITTING")
            acked = {w for w in range(6)
                     if cycle["commit_acks_by_worker"][str(w)]}
            self.assertEqual(acked, {0, 1, 2, 3})
            commits_after_kill = int(world.commit_calls)
            frames_acked = {w: int(world.frame[w]) for w in acked}

            world.crash_hard = False
            world.crash_commit_workers = set()
            world2, ctl2, views2 = dr.make_world(
                world=world, cap_target_by_worker=self._caps(),
                run_dir=run_dir)
            self._install_overshoot(world, ctl2)
            pending = ctl2.pending_cycle(list(range(6)))
            with self.assertRaises(CycleFailure):
                ctl2.run_cycle(views2, resume_cycle=pending)
            self.assertEqual(world.commit_calls - commits_after_kill, 2)
            for w in acked:
                self.assertEqual(int(world.frame[w]), frames_acked[w])
            done = world.state_store[0]["msa_cycle"]
            self.assertEqual(done["status"], "FAILED")
            self.assertTrue(done.get("durable_commits_completed"))
            self.assertFalse(done.get("rollback_claimed", True))

    def test_success_path_done_only_after_dose_validation(self):
        from dexpbt_isaaclab.msa.state_machine import MSAPhase
        with tempfile.TemporaryDirectory() as td:
            run_dir = Path(td)
            world, ctl, views = dr.make_world(
                cap_target_by_worker=self._caps(), run_dir=run_dir)
            seen = []
            real_fin = ctl._finalize_cycle_dose

            def wrap(due, start_frames, cycle, admission):
                seen.append((str(cycle.get("status")),
                             ctl.machine.phase.value))
                out = real_fin(due, start_frames, cycle, admission)
                self.assertEqual(cycle.get("status"), "COMMITTING")
                self.assertNotEqual(ctl.machine.phase, MSAPhase.NORMAL)
                return out

            ctl._finalize_cycle_dose = wrap
            result = ctl.run_cycle(views)
            self.assertTrue(seen)
            self.assertEqual(seen[0][0], "COMMITTING")
            self.assertIn(seen[0][1], (MSAPhase.DECIDE.value,
                                       MSAPhase.PROMOTE.value))
            cycle = world.state_store[0]["msa_cycle"]
            self.assertEqual(cycle["status"], "DONE")
            self.assertEqual(ctl.machine.phase, MSAPhase.NORMAL)
            self.assertIsNotNone(result.observed_cycle_dose_by_worker)


class TestCycleFailureStopsArm(unittest.TestCase):
    def test_cycle_failure_writes_marker_and_calls_stop_arm(self):
        from dexpbt_isaaclab.msa.controller import CycleFailure
        from dexpbt_isaaclab.msa.sidecar import handle_cycle_failure
        with tempfile.TemporaryDirectory() as td:
            run_dir = Path(td)
            stops = []
            released = []

            err = CycleFailure(
                "injected cycle failure",
                cycle={
                    "intervention_id": "msa_test",
                    "status": "FAILED",
                    "failed_phase": "COMMITTING",
                    "commit_acks_by_worker": {"0": True},
                    "final_tol_by_worker": {"0": 0.02},
                    "final_committed_frame_by_worker": {"0": 1},
                    "observed_cycle_dose_by_worker": {"0": 1},
                    "budget_admission": {"required_reserve": 400_000_000},
                    "durable_commits_completed": True,
                    "rollback_claimed": False,
                },
                durable_commits_completed=True,
            )
            err.__cause__ = RuntimeError("reserved ceiling")

            def stop(rd, reason, marker_name=None, marker_payload=None,
                     num_workers=6):
                stops.append({
                    "run_dir": Path(rd),
                    "reason": reason,
                    "marker_name": marker_name,
                })

            handle_cycle_failure(
                run_dir, err,
                release_holds_fn=lambda: released.append(True),
                stop_arm_fn=stop,
                num_policies=6)
            self.assertTrue((run_dir / "MSA_CYCLE_FAILED.json").is_file())
            body = json.loads(
                (run_dir / "MSA_CYCLE_FAILED.json").read_text())
            self.assertEqual(body["intervention_id"], "msa_test")
            self.assertEqual(body["cause_type"], "RuntimeError")
            self.assertIn("reserved ceiling", body["cause_message"])
            self.assertTrue(body["durable_commits_completed"])
            self.assertFalse(body["rollback_claimed"])
            self.assertEqual(released, [True])
            self.assertEqual(len(stops), 1)
            self.assertEqual(stops[0]["run_dir"], run_dir)
            self.assertIsNone(stops[0]["marker_name"])

    def test_commit_timeout_path_unchanged_marker(self):
        """CommitTimeout still uses COMMIT_TIMEOUT.json (regression)."""
        from dexpbt_isaaclab.msa.controller import CommitTimeout, CycleFailure
        world, ctl, views = dr.make_world(crash_on="commit_timeout")
        with self.assertRaises(CycleFailure) as cm:
            ctl.run_cycle(views)
        cause = cm.exception.__cause__
        self.assertIsInstance(cause, CommitTimeout)
        self.assertEqual(world.commit_calls, 6)
        cycle = world.state_store[0]["msa_cycle"]
        self.assertEqual(cycle["status"], "FAILED")
        self.assertFalse(cycle.get("durable_commits_completed", False))


if __name__ == "__main__":
    unittest.main(verbosity=2)
