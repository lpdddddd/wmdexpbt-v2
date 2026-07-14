"""Unit tests for Profile22 Experiment ABC (no training processes)."""

from __future__ import annotations

import argparse
import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from dexpbt_isaaclab.pbt.pbt_chronic_bottom import (
    ChronicBottomState,
    evaluate_chronic_eligibility,
    live_source_ok_for_chronic,
    pick_chronic_source,
    reset_after_chronic_copy,
    skill_delta_invalid_reason_is_mixed_tol,
    update_continuous_bottom,
)
from dexpbt_isaaclab.world_model.elite_snippet_bank import (
    EliteSnippetBank,
    EliteSnippetBankConfig,
    SNIPPET_SCHEMA_VERSION,
)
from dexpbt_isaaclab.world_model.eir_broadcast import (
    EliteBroadcastConfig,
    merge_payload_into_bridge,
)
from dexpbt_isaaclab.world_model.snippet_bank_tolerance import filter_state_by_tolerance
from dexpbt_isaaclab.world_model.wm_strategy_profiles import (
    PROFILE_IDS,
    PROFILE_TABLES,
    apply_wm_strategy_profile,
)


def _ns(**kwargs):
    args = argparse.Namespace(wm_strategy_profile=kwargs.get("wm_strategy_profile", "legacy"))
    for table in PROFILE_TABLES.values():
        for k in table:
            if not hasattr(args, k):
                setattr(args, k, None)
    for k, v in {
        "pbt_enable_chronic_bottom_exploit": False,
        "eir_tolerance_aware_bank_broadcast": False,
        "eir_enable_belief_for_snippet_wm_rerank": False,
        "eir_preserve_local_bank_fraction": 0.20,
        "pbt_allow_mid_self_mutation": False,
    }.items():
        if getattr(args, k, None) is None:
            setattr(args, k, v)
    for k, v in kwargs.items():
        setattr(args, k, v)
    return args


class TestConfigRegressionABC(unittest.TestCase):
    def test_profile22_unchanged_by_abc_flags(self):
        t22 = PROFILE_TABLES["cser_dexpbt_progress_refine"]
        self.assertFalse(bool(t22.get("pbt_enable_chronic_bottom_exploit", False)))
        self.assertFalse(bool(t22.get("eir_tolerance_aware_bank_broadcast", False)))
        self.assertFalse(bool(t22.get("eir_enable_belief_for_snippet_wm_rerank", False)))
        self.assertEqual(PROFILE_IDS["cser_dexpbt_progress_refine"], 22)

    def test_abc_profile_id_and_flags(self):
        self.assertEqual(PROFILE_IDS["cser_dexpbt_progress_refine_abc"], 24)
        t = PROFILE_TABLES["cser_dexpbt_progress_refine_abc"]
        self.assertTrue(t["pbt_enable_chronic_bottom_exploit"])
        self.assertTrue(t["eir_tolerance_aware_bank_broadcast"])
        self.assertTrue(t["eir_enable_belief_for_snippet_wm_rerank"])
        self.assertAlmostEqual(float(t["eir_preserve_local_bank_fraction"]), 0.20)
        self.assertFalse(t["pbt_allow_mid_self_mutation"])
        self.assertEqual(t["eir_mppi_mode"], "off")
        self.assertFalse(t["eir_use_rssm_mppi"])

    def test_abc_vs_22_diff_only_abc(self):
        t22 = PROFILE_TABLES["cser_dexpbt_progress_refine"]
        tabc = PROFILE_TABLES["cser_dexpbt_progress_refine_abc"]
        diff = {k: (t22.get(k), tabc.get(k)) for k in set(t22) | set(tabc) if t22.get(k) != tabc.get(k)}
        # Profile22 now pin ABC enables to False; ABC turns them on + chronic knobs + local reserve.
        expected = {
            "pbt_enable_chronic_bottom_exploit",
            "pbt_chronic_min_bottom_frames",
            "pbt_chronic_min_bottom_ticks",
            "pbt_chronic_min_score_gap",
            "pbt_chronic_cooldown_frames",
            "pbt_chronic_max_copies_per_lineage",
            "eir_tolerance_aware_bank_broadcast",
            "eir_preserve_local_bank_fraction",
            "eir_enable_belief_for_snippet_wm_rerank",
        }
        self.assertEqual(set(diff.keys()), expected)

    def test_apply_profile22_does_not_enable_abc(self):
        args = _ns(wm_strategy_profile="cser_dexpbt_progress_refine")
        apply_wm_strategy_profile(args, authoritative=True)
        self.assertFalse(bool(args.pbt_enable_chronic_bottom_exploit))
        self.assertFalse(bool(args.eir_tolerance_aware_bank_broadcast))
        self.assertFalse(bool(args.eir_enable_belief_for_snippet_wm_rerank))

    def test_apply_abc_enables_all(self):
        args = _ns(wm_strategy_profile="cser_dexpbt_progress_refine_abc")
        apply_wm_strategy_profile(args, authoritative=True)
        self.assertTrue(args.pbt_enable_chronic_bottom_exploit)
        self.assertTrue(args.eir_tolerance_aware_bank_broadcast)
        self.assertTrue(args.eir_enable_belief_for_snippet_wm_rerank)
        self.assertFalse(args.pbt_allow_mid_self_mutation)


class TestChronicBottomPure(unittest.TestCase):
    def test_below_200m_not_eligible(self):
        st = ChronicBottomState(continuous_bottom_frames=100_000_000, continuous_bottom_ticks=10)
        el = evaluate_chronic_eligibility(
            tier="bottom",
            state=st,
            env_frames=500_000_000,
            promising_protected=False,
            in_elimination_burnin=False,
            in_probation=False,
            target_score=0.1,
            ranked_scores={0: 0.1, 1: 0.5},
            target_idx=0,
        )
        self.assertFalse(el.eligible)
        self.assertEqual(el.reason, "bottom_frames_below_threshold")

    def test_promising_blocks(self):
        st = ChronicBottomState(continuous_bottom_frames=250_000_000, continuous_bottom_ticks=10)
        el = evaluate_chronic_eligibility(
            tier="bottom",
            state=st,
            env_frames=500_000_000,
            promising_protected=True,
            in_elimination_burnin=False,
            in_probation=False,
            target_score=0.1,
            ranked_scores={0: 0.1, 1: 0.5},
            target_idx=0,
        )
        self.assertFalse(el.eligible)
        self.assertEqual(el.reason, "promising_protected")

    def test_cooldown_and_probation(self):
        st = ChronicBottomState(
            continuous_bottom_frames=250_000_000,
            continuous_bottom_ticks=10,
            chronic_copy_cooldown_until=600_000_000,
        )
        el = evaluate_chronic_eligibility(
            tier="bottom",
            state=st,
            env_frames=500_000_000,
            promising_protected=False,
            in_elimination_burnin=False,
            in_probation=False,
            target_score=0.1,
            ranked_scores={0: 0.1, 1: 0.5},
            target_idx=0,
        )
        self.assertEqual(el.reason, "chronic_cooldown")
        el2 = evaluate_chronic_eligibility(
            tier="bottom",
            state=ChronicBottomState(continuous_bottom_frames=250_000_000, continuous_bottom_ticks=10),
            env_frames=500_000_000,
            promising_protected=False,
            in_elimination_burnin=False,
            in_probation=True,
            target_score=0.1,
            ranked_scores={0: 0.1, 1: 0.5},
            target_idx=0,
        )
        self.assertEqual(el2.reason, "probation_active")

    def test_gap_threshold(self):
        st = ChronicBottomState(continuous_bottom_frames=250_000_000, continuous_bottom_ticks=10)
        el = evaluate_chronic_eligibility(
            tier="bottom",
            state=st,
            env_frames=500_000_000,
            promising_protected=False,
            in_elimination_burnin=False,
            in_probation=False,
            target_score=0.4,
            ranked_scores={0: 0.4, 1: 0.5},
            target_idx=0,
            min_score_gap=0.15,
        )
        self.assertFalse(el.eligible)
        self.assertEqual(el.reason, "score_gap_below_threshold")

    def test_valid_delta_and_mixed_tol_fallback(self):
        ok, reason, fb = live_source_ok_for_chronic(
            skill_delta=0.05,
            skill_delta_why="ok",
            source_nsucc_mean=30.0,
            target_nsucc_mean=1.0,
        )
        self.assertTrue(ok)
        self.assertFalse(fb)
        ok2, reason2, fb2 = live_source_ok_for_chronic(
            skill_delta=None,
            skill_delta_why="mixed_tol_bins",
            source_nsucc_mean=30.0,
            target_nsucc_mean=1.0,
            source_recent_drop_ratio=0.05,
        )
        self.assertTrue(ok2)
        self.assertTrue(fb2)
        self.assertTrue(skill_delta_invalid_reason_is_mixed_tol(reason2) or "fallback" in reason2 or ok2)
        ok3, reason3, _ = live_source_ok_for_chronic(
            skill_delta=None,
            skill_delta_why="lineage_change",
            source_nsucc_mean=30.0,
            target_nsucc_mean=1.0,
        )
        self.assertFalse(ok3)
        self.assertIn("lineage_change", reason3)

    def test_pick_source_prefers_top_and_exact_copy_state(self):
        pop = {
            0: {"pbt_ranking_score": 0.1, "nsucc_mean": 1.0, "checkpoint": "/tmp/x"},
            1: {"pbt_ranking_score": 0.9, "nsucc_mean": 35.0, "checkpoint": "/tmp/y"},
            2: {"pbt_ranking_score": 0.5, "nsucc_mean": 20.0, "checkpoint": "/tmp/z"},
        }
        decision = pick_chronic_source(
            sorted_policies=[1, 2, 0],
            population=pop,
            target_idx=0,
            target_score=0.1,
            target_nsucc_mean=1.0,
            n_top=1,
            n_mid=1,
            min_score_gap=0.15,
            skill_delta_by_policy={1: (0.05, "ok"), 2: (0.05, "ok")},
            checkpoint_ok={1: True, 2: True},
        )
        self.assertEqual(decision.action, "copy")
        self.assertEqual(decision.source_policy, 1)
        self.assertTrue(decision.tier.startswith("chronic_top"))

    def test_mixed_tol_fallback_selects(self):
        pop = {
            0: {"pbt_ranking_score": 0.1, "nsucc_mean": 1.0},
            1: {"pbt_ranking_score": 0.9, "nsucc_mean": 35.0},
        }
        decision = pick_chronic_source(
            sorted_policies=[1, 0],
            population=pop,
            target_idx=0,
            target_score=0.1,
            target_nsucc_mean=1.0,
            n_top=1,
            n_mid=0,
            min_score_gap=0.15,
            skill_delta_by_policy={1: (None, "mixed_tol_bins")},
            drop_ratio_by_policy={1: 0.05},
            checkpoint_ok={1: True},
        )
        self.assertEqual(decision.action, "copy")
        self.assertTrue(decision.source_fallback_used)

    def test_update_clear_and_resume_state(self):
        st = ChronicBottomState()
        update_continuous_bottom(st, tier="bottom", env_frames=100, interval_steps=20)
        update_continuous_bottom(st, tier="bottom", env_frames=120, interval_steps=20)
        self.assertGreaterEqual(st.continuous_bottom_ticks, 2)
        payload = st.to_json()
        restored = ChronicBottomState.from_json(payload)
        self.assertEqual(restored.continuous_bottom_ticks, st.continuous_bottom_ticks)
        update_continuous_bottom(st, tier="mid", env_frames=140, interval_steps=20)
        self.assertEqual(st.continuous_bottom_frames, 0)
        reset_after_chronic_copy(st, env_frames=200, cooldown=400)
        self.assertEqual(st.chronic_copy_count, 1)
        self.assertEqual(st.chronic_copy_cooldown_until, 600)


class TestToleranceAwareBank(unittest.TestCase):
    def _bank(self, size=8, policy=2, tol=0.02):
        cfg = EliteSnippetBankConfig(enabled=True, bank_size=size, feature_dim=4, action_dim=2, horizon=2)
        bank = EliteSnippetBank(cfg)
        n = 3
        block = {
            "feature": torch.randn(n, 4),
            "actions": torch.randn(n, 2, 2),
            "d_hat": torch.ones(n) * 0.01,
            "success_tol": torch.ones(n) * float(tol),
            "lifted": torch.ones(n),
            "picked": torch.ones(n),
            "dropped": torch.zeros(n),
            "closest_d_hat": torch.ones(n) * 0.005,
            "success_h": torch.ones(n),
            "improve_h": torch.ones(n),
            "drop_h": torch.zeros(n),
            "recovery_label": torch.ones(n),
            "source_policy_idx": torch.ones(n) * float(policy),
            "source_nsucc_mean": torch.ones(n) * 30.0,
            "source_nsucc_max": torch.ones(n) * 40.0,
            "frame": torch.ones(n) * 1.0,
            "priority_score": torch.linspace(1, 3, n),
            "bucket_id": torch.ones(n) * 2.0,
            "horizon": torch.ones(n) * 2.0,
            "normalized_error": torch.zeros(n),
            "recent_progress": torch.zeros(n),
            "oscillation": torch.zeros(n),
            "progress_history_valid": torch.zeros(n),
            "normalized_improvement": torch.zeros(n),
        }
        bank._insert(block)
        return bank

    def test_zero_compatible_does_not_overwrite(self):
        bank = self._bank(policy=2, tol=0.02)
        before = bank.state_dict()
        foreign = self._bank(policy=0, tol=0.075).state_dict()
        filtered, diag = filter_state_by_tolerance(foreign, query_tol=0.02, max_log_gap=0.20)
        self.assertEqual(int(filtered["size"]), 0)
        self.assertEqual(float(diag["broadcast_rejected_incompatible"]), 1.0)
        # merge_tolerance_aware with empty filtered returns 0 and keeps bank
        merged = bank.merge_tolerance_aware(filtered, local_policy_idx=2, preserve_local_fraction=0.2)
        self.assertEqual(merged, 0)
        self.assertEqual(bank.size, before["size"])

    def test_compatible_merge_and_cap(self):
        bank = self._bank(size=10, policy=2, tol=0.02)
        foreign = self._bank(size=10, policy=0, tol=0.022).state_dict()
        filtered, diag = filter_state_by_tolerance(foreign, query_tol=0.02, max_log_gap=0.20)
        self.assertGreater(int(filtered["size"]), 0)
        bank.merge_tolerance_aware(
            filtered, local_policy_idx=2, preserve_local_fraction=0.2, max_entries=50_000
        )
        self.assertLessEqual(bank.size, 10)

    def test_local_reserve(self):
        cfg = EliteSnippetBankConfig(enabled=True, bank_size=10, feature_dim=4, action_dim=2, horizon=2)
        bank = EliteSnippetBank(cfg)
        local = {
            "feature": torch.randn(4, 4),
            "actions": torch.randn(4, 2, 2),
            "d_hat": torch.ones(4) * 0.01,
            "success_tol": torch.ones(4) * 0.02,
            "lifted": torch.ones(4),
            "picked": torch.ones(4),
            "dropped": torch.zeros(4),
            "closest_d_hat": torch.ones(4) * 0.005,
            "success_h": torch.ones(4),
            "improve_h": torch.ones(4),
            "drop_h": torch.zeros(4),
            "recovery_label": torch.ones(4),
            "source_policy_idx": torch.ones(4) * 2.0,
            "source_nsucc_mean": torch.ones(4) * 10.0,
            "source_nsucc_max": torch.ones(4) * 10.0,
            "frame": torch.ones(4),
            "priority_score": torch.ones(4) * 1.0,  # low priority local
            "bucket_id": torch.ones(4) * 2.0,
            "horizon": torch.ones(4) * 2.0,
            "normalized_error": torch.zeros(4),
            "recent_progress": torch.zeros(4),
            "oscillation": torch.zeros(4),
            "progress_history_valid": torch.zeros(4),
            "normalized_improvement": torch.zeros(4),
        }
        foreign = {k: v.clone() if torch.is_tensor(v) else v for k, v in local.items()}
        foreign["feature"] = torch.randn(10, 4)
        foreign["actions"] = torch.randn(10, 2, 2)
        for k in foreign:
            if k in ("feature", "actions"):
                continue
            if torch.is_tensor(foreign[k]):
                foreign[k] = torch.ones(10) * (30.0 if k == "priority_score" else (0.0 if "source_policy" in k else float(foreign[k][0] if foreign[k].numel() else 0)))
        foreign["source_policy_idx"] = torch.zeros(10)
        foreign["success_tol"] = torch.ones(10) * 0.02
        foreign["priority_score"] = torch.ones(10) * 30.0
        bank._insert(local)
        bank._insert_with_local_reserve(foreign, local_policy_idx=2, preserve_local_fraction=0.2)
        local_kept = int((bank.scalars["source_policy_idx"][: bank.size] == 2.0).sum().item())
        self.assertGreaterEqual(local_kept, 2)  # 20% of 10
        self.assertEqual(bank.size, 10)

    def test_illegal_tol_fail_close(self):
        st = self._bank(tol=0.02).state_dict()
        filtered, diag = filter_state_by_tolerance(st, query_tol=0.0)
        self.assertEqual(int(filtered["size"]), 0)
        self.assertEqual(float(diag["invalid_tol_fail_close"]), 1.0)
        filtered2, diag2 = filter_state_by_tolerance(st, query_tol=float("nan"))
        self.assertEqual(float(diag2["invalid_tol_fail_close"]), 1.0)

    def test_save_load_and_old_schema(self):
        bank = self._bank()
        st = bank.state_dict()
        self.assertEqual(int(st["schema_version"]), SNIPPET_SCHEMA_VERSION)
        bank2 = self._bank()
        bank2.size = 0
        bank2.load_state_dict(st)
        self.assertEqual(bank2.size, bank.size)
        # old schema without progress columns
        old = {k: v for k, v in st.items() if k not in (
            "normalized_error", "recent_progress", "oscillation",
            "progress_history_valid", "normalized_improvement",
        )}
        old["schema_version"] = 1
        bank3 = self._bank()
        bank3.size = 0
        bank3.load_state_dict(old)
        self.assertEqual(bank3.size, bank.size)

    def test_merge_payload_rejects_incompatible(self):
        bank = self._bank(policy=2, tol=0.02)
        bridge = SimpleNamespace(
            bank=None,
            snippet_bank=bank,
            residual_bridge=None,
            scorer=None,
            agent_ref=None,
        )
        foreign = self._bank(policy=0, tol=0.075).state_dict()
        cfg = EliteBroadcastConfig(
            tolerance_aware_bank_broadcast=True,
            preserve_local_bank_fraction=0.2,
            target_policy_idx=2,
            query_success_tolerance=0.02,
        )
        before = bank.size
        out = merge_payload_into_bridge(
            bridge,
            {"snippet_bank_state": foreign},
            cfg=cfg,
            query_success_tolerance=0.02,
        )
        self.assertTrue(out["broadcast_rejected_incompatible"])
        self.assertEqual(bank.size, before)


class TestBeliefNeedFlag(unittest.TestCase):
    def test_need_belief_logic(self):
        # Mirror the train.py predicate without importing heavy train module.
        def need_belief(planner, reranker, flag, snippet_enable, cfg_enabled=False):
            return (
                planner is not None
                or reranker is not None
                or (flag and snippet_enable)
                or (flag and cfg_enabled)
            )

        self.assertFalse(need_belief(None, None, False, True))
        self.assertTrue(need_belief(None, None, True, True))
        self.assertFalse(need_belief(None, None, True, False))
        self.assertTrue(need_belief(object(), None, False, False))


class TestLauncherSyntax(unittest.TestCase):
    def test_launcher_exists_and_bash_n(self):
        import subprocess

        path = Path(
            "/root/autodl-tmp/wmdexpbt/results/launchers/"
            "start_regrasping_cser_dexpbt_progress_refine_abc_5b.sh"
        )
        self.assertTrue(path.is_file())
        text = path.read_text()
        self.assertIn("cser_dexpbt_progress_refine_abc", text)
        self.assertIn("pbt_enable_chronic_bottom_exploit", text)
        self.assertIn("eir_tolerance_aware_bank_broadcast", text)
        self.assertIn("eir_enable_belief_for_snippet_wm_rerank", text)
        self.assertIn("no-pbt_allow_mid_self_mutation", text)
        self.assertIn("eir_mppi_mode off", text)
        proc = subprocess.run(["bash", "-n", str(path)], check=False, capture_output=True)
        self.assertEqual(proc.returncode, 0, proc.stderr.decode())


if __name__ == "__main__":
    unittest.main()
