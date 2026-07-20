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
    def test_insufficient_budget_skips_pre_eval_and_busy(self):
        tip = 5_080_000_000
        caps = {w: tip + 480_000_000 for w in range(6)}
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
            self.assertTrue((run_dir / RESERVATION_MARKER).exists())

    def test_no_new_cycle_marker_helper(self):
        with tempfile.TemporaryDirectory() as td:
            run_dir = Path(td)
            write_no_new_cycle(run_dir, {"reason": "all_budget_ineligible",
                                         "due_workers": [0, 1]})
            self.assertTrue((run_dir / NO_NEW_CYCLE_MARKER).exists())
            body = json.loads((run_dir / NO_NEW_CYCLE_MARKER).read_text())
            self.assertEqual(body["reason"], "all_budget_ineligible")


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


if __name__ == "__main__":
    unittest.main(verbosity=2)
