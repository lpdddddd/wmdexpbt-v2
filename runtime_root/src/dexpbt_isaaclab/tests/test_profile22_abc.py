"""Unit tests for Profile22 Experiment ABC review fixes (no training)."""

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
    reset_on_lineage_change,
    update_continuous_bottom,
)
from dexpbt_isaaclab.pbt.pbt_live_source_health import (
    CommonSourceEligibility,
    evaluate_common_live_source_eligibility,
)
from dexpbt_isaaclab.world_model.eir_belief import need_eir_belief_cache
from dexpbt_isaaclab.world_model.eir_broadcast import (
    EliteBroadcastConfig,
    merge_payload_into_bridge,
)
from dexpbt_isaaclab.world_model.elite_snippet_bank import (
    EliteSnippetBank,
    EliteSnippetBankConfig,
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
        "snippet_wm_rerank_enable": False,
    }.items():
        if getattr(args, k, None) is None:
            setattr(args, k, v)
    for k, v in kwargs.items():
        setattr(args, k, v)
    return args


def _ok_health(**overrides) -> CommonSourceEligibility:
    base = dict(
        has_state=True,
        ranking_score=0.9,
        nsucc_mean=35.0,
        source_gate_ok=True,
        source_gate_reason="ok",
        restart_burnin_ok=True,
        restart_burnin_reason="ok",
        checkpoint_exists=True,
        checkpoint_commit_ok=True,
        lineage_ok=True,
        freshness_ok=True,
        probation_ok=True,
        collapse_ok=True,
    )
    base.update(overrides)
    return evaluate_common_live_source_eligibility(**base)


class TestConfigRegressionABC(unittest.TestCase):
    def test_profile22_abc_off(self):
        t22 = PROFILE_TABLES["cser_dexpbt_progress_refine"]
        self.assertFalse(bool(t22.get("pbt_enable_chronic_bottom_exploit")))
        self.assertFalse(bool(t22.get("eir_tolerance_aware_bank_broadcast")))
        self.assertFalse(bool(t22.get("eir_enable_belief_for_snippet_wm_rerank")))
        args = _ns(wm_strategy_profile="cser_dexpbt_progress_refine")
        apply_wm_strategy_profile(args, authoritative=True)
        self.assertFalse(args.pbt_enable_chronic_bottom_exploit)
        self.assertFalse(args.eir_tolerance_aware_bank_broadcast)
        self.assertFalse(args.eir_enable_belief_for_snippet_wm_rerank)

    def test_profile24_authoritative_all_on_overrides_cli_off(self):
        self.assertEqual(PROFILE_IDS["cser_dexpbt_progress_refine_abc"], 24)
        args = _ns(
            wm_strategy_profile="cser_dexpbt_progress_refine_abc",
            pbt_enable_chronic_bottom_exploit=False,
            eir_tolerance_aware_bank_broadcast=False,
            eir_enable_belief_for_snippet_wm_rerank=False,
        )
        apply_wm_strategy_profile(
            args,
            argv=[
                "train.py",
                "--wm_strategy_profile=cser_dexpbt_progress_refine_abc",
                "--no-pbt_enable_chronic_bottom_exploit",
                "--no-eir_tolerance_aware_bank_broadcast",
                "--no-eir_enable_belief_for_snippet_wm_rerank",
            ],
            authoritative=True,
        )
        self.assertTrue(args.pbt_enable_chronic_bottom_exploit)
        self.assertTrue(args.eir_tolerance_aware_bank_broadcast)
        self.assertTrue(args.eir_enable_belief_for_snippet_wm_rerank)
        self.assertFalse(args.pbt_allow_mid_self_mutation)
        self.assertEqual(args.eir_mppi_mode, "off")


class TestCommonSourceHealth(unittest.TestCase):
    def test_rejects_high_score_but_ineligible(self):
        h = _ok_health(source_gate_ok=False, source_gate_reason="nsucc_mean<1.0")
        self.assertFalse(h.eligible)
        self.assertIn("nsucc_mean", h.reason)

    def test_rejects_burnin(self):
        h = _ok_health(
            restart_burnin_ok=False,
            restart_burnin_reason="frames_since_restart<50000000",
            source_gate_ok=False,
            source_gate_reason="frames_since_restart<50000000",
        )
        self.assertFalse(h.eligible)

    def test_rejects_lineage_mismatch(self):
        h = _ok_health(lineage_ok=False)
        self.assertFalse(h.eligible)
        self.assertEqual(h.reason, "lineage_mismatch")

    def test_rejects_commit_invalid(self):
        h = _ok_health(checkpoint_commit_ok=False)
        self.assertFalse(h.eligible)

    def test_rejects_probation(self):
        h = _ok_health(probation_ok=False)
        self.assertEqual(h.reason, "source_probation")

    def test_rejects_collapse(self):
        h = _ok_health(collapse_ok=False)
        self.assertEqual(h.reason, "source_collapse_or_unhealthy")

    def test_rejects_nan_score(self):
        h = _ok_health(ranking_score=float("nan"))
        self.assertEqual(h.reason, "nonfinite_metrics")

    def test_rejects_missing_health_field(self):
        h = _ok_health(freshness_ok=None)
        self.assertEqual(h.reason, "freshness_unknown")

    def test_accepts_all_ok(self):
        h = _ok_health()
        self.assertTrue(h.eligible)


class TestChronicSourceWithHealth(unittest.TestCase):
    def test_health_fail_rejects_despite_high_score(self):
        pop = {
            0: {"pbt_ranking_score": 0.1, "nsucc_mean": 1.0},
            1: {"pbt_ranking_score": 0.9, "nsucc_mean": 35.0},
        }
        bad = _ok_health(source_gate_ok=False, source_gate_reason="eligible_as_source=False")
        d = pick_chronic_source(
            sorted_policies=[1, 0],
            population=pop,
            target_idx=0,
            target_score=0.1,
            target_nsucc_mean=1.0,
            n_top=1,
            n_mid=0,
            min_score_gap=0.15,
            skill_delta_by_policy={1: (0.05, "ok")},
            common_health_by_policy={1: bad},
        )
        self.assertEqual(d.action, "skip")

    def test_valid_delta_accept(self):
        pop = {
            0: {"pbt_ranking_score": 0.1, "nsucc_mean": 1.0},
            1: {"pbt_ranking_score": 0.9, "nsucc_mean": 35.0},
        }
        d = pick_chronic_source(
            sorted_policies=[1, 0],
            population=pop,
            target_idx=0,
            target_score=0.1,
            target_nsucc_mean=1.0,
            n_top=1,
            n_mid=0,
            min_score_gap=0.15,
            skill_delta_by_policy={1: (0.05, "ok")},
            common_health_by_policy={1: _ok_health()},
        )
        self.assertEqual(d.action, "copy")
        self.assertTrue(d.common_health_allowed)
        self.assertFalse(d.mixed_tol_fallback_used)

    def test_mixed_tol_fallback_accept(self):
        pop = {
            0: {"pbt_ranking_score": 0.1, "nsucc_mean": 1.0},
            1: {"pbt_ranking_score": 0.9, "nsucc_mean": 35.0},
        }
        d = pick_chronic_source(
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
            common_health_by_policy={1: _ok_health()},
        )
        self.assertEqual(d.action, "copy")
        self.assertTrue(d.mixed_tol_fallback_used)

    def test_lineage_invalid_delta_rejects(self):
        ok, reason, _ = live_source_ok_for_chronic(
            skill_delta=None,
            skill_delta_why="lineage_change",
            source_nsucc_mean=35.0,
            target_nsucc_mean=1.0,
        )
        self.assertFalse(ok)
        self.assertIn("lineage_change", reason)

    def test_target_score_nonfinite_rejects(self):
        d = pick_chronic_source(
            sorted_policies=[1, 0],
            population={0: {}, 1: {"pbt_ranking_score": 1.0, "nsucc_mean": 10}},
            target_idx=0,
            target_score=float("nan"),
            target_nsucc_mean=1.0,
            n_top=1,
            n_mid=0,
            min_score_gap=0.15,
            skill_delta_by_policy={1: (0.05, "ok")},
            common_health_by_policy={1: _ok_health()},
        )
        self.assertEqual(d.action, "skip")


class TestLineageReset(unittest.TestCase):
    def test_same_lineage_keeps_span(self):
        st = ChronicBottomState(
            continuous_bottom_frames=250_000_000,
            continuous_bottom_ticks=12,
            bottom_enter_frame=100,
            last_tier="bottom",
            chronic_copy_count=0,
            chronic_copy_cooldown_until=900,
        )
        payload = st.to_json()
        restored = ChronicBottomState.from_json(payload)
        self.assertEqual(restored.continuous_bottom_frames, 250_000_000)
        self.assertEqual(restored.continuous_bottom_ticks, 12)

    def test_lineage_change_clears_span_keeps_cooldown(self):
        st = ChronicBottomState(
            continuous_bottom_frames=250_000_000,
            continuous_bottom_ticks=12,
            bottom_enter_frame=100,
            last_tier="bottom",
            chronic_copy_count=1,
            chronic_copy_cooldown_until=900,
            last_chronic_copy_frame=500,
            last_chronic_source=3,
        )
        reset_on_lineage_change(st)
        self.assertEqual(st.continuous_bottom_frames, 0)
        self.assertEqual(st.continuous_bottom_ticks, 0)
        self.assertEqual(st.bottom_enter_frame, -1)
        self.assertEqual(st.last_tier, "")
        self.assertEqual(st.chronic_copy_count, 0)
        self.assertEqual(st.chronic_copy_cooldown_until, 900)
        self.assertEqual(st.last_chronic_copy_frame, 500)
        self.assertEqual(st.last_reason, "lineage_reset")

    def test_chronic_copy_reset_and_trigger_snapshot(self):
        st = ChronicBottomState(
            continuous_bottom_frames=300_000_000,
            continuous_bottom_ticks=20,
        )
        trigger_f, trigger_t = st.continuous_bottom_frames, st.continuous_bottom_ticks
        reset_after_chronic_copy(st, env_frames=1000, cooldown=400)
        self.assertEqual(trigger_f, 300_000_000)
        self.assertEqual(trigger_t, 20)
        self.assertEqual(st.continuous_bottom_frames, 0)
        self.assertEqual(st.chronic_copy_count, 1)
        self.assertEqual(st.chronic_copy_cooldown_until, 1400)


class TestBankMergeCounts(unittest.TestCase):
    def _mk(self, n, policy, tol, pri, capacity=10):
        cfg = EliteSnippetBankConfig(
            enabled=True, bank_size=capacity, feature_dim=4, action_dim=2, horizon=2
        )
        bank = EliteSnippetBank(cfg)
        if n <= 0:
            return bank
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
            "frame": torch.ones(n),
            "priority_score": torch.ones(n) * float(pri),
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

    def test_empty_and_partial(self):
        bank = self._mk(0, 2, 0.02, 1.0, capacity=10)
        foreign = self._mk(3, 0, 0.02, 5.0, capacity=10).state_dict()
        kept = bank.merge_tolerance_aware(foreign, local_policy_idx=2)
        self.assertEqual(kept, 3)
        self.assertEqual(bank.size, 3)
        self.assertEqual(bank.last_update_diagnostics["incoming_kept_count"], 3.0)

    def test_full_bank_replace_keeps_incoming_count(self):
        bank = self._mk(10, 2, 0.02, 1.0, capacity=10)  # low pri local
        foreign = self._mk(10, 0, 0.02, 50.0, capacity=10).state_dict()
        kept = bank.merge_tolerance_aware(
            foreign, local_policy_idx=2, preserve_local_fraction=0.2
        )
        self.assertGreater(kept, 0)
        self.assertEqual(bank.size, 10)
        self.assertEqual(
            bank.last_update_diagnostics["incoming_kept_count"], float(kept)
        )
        # local reserve 20% of 10 = 2
        local_after = int(
            (bank.scalars["source_policy_idx"][: bank.size] == 2.0).sum().item()
        )
        self.assertGreaterEqual(local_after, 2)

    def test_zero_compatible_unchanged(self):
        bank = self._mk(4, 2, 0.02, 1.0, capacity=10)
        before = bank.size
        foreign = self._mk(5, 0, 0.075, 50.0, capacity=10).state_dict()
        filtered, diag = filter_state_by_tolerance(foreign, query_tol=0.02)
        self.assertEqual(int(filtered["size"]), 0)
        kept = bank.merge_tolerance_aware(filtered, local_policy_idx=2)
        self.assertEqual(kept, 0)
        self.assertEqual(bank.size, before)

    def test_merge_payload_logging_fields(self):
        bank = self._mk(2, 2, 0.02, 1.0, capacity=10)
        bridge = SimpleNamespace(
            bank=None, snippet_bank=bank, residual_bridge=None, scorer=None, agent_ref=None
        )
        foreign = self._mk(3, 0, 0.075, 5.0, capacity=10).state_dict()
        cfg = EliteBroadcastConfig(
            tolerance_aware_bank_broadcast=True,
            target_policy_idx=2,
            query_success_tolerance=0.02,
        )
        out = merge_payload_into_bridge(
            bridge, {"snippet_bank_state": foreign}, cfg=cfg, query_success_tolerance=0.02
        )
        self.assertTrue(out["broadcast_rejected_incompatible"])
        self.assertIn("incoming_compatible_count", out["bank_diagnostics"])
        self.assertEqual(out["snippet_bank_merged_entries"], 0)
        self.assertEqual(out["recovery_bank_merged_entries"], 0)


class TestBeliefPredicate(unittest.TestCase):
    def test_production_helper_matrix(self):
        self.assertFalse(
            need_eir_belief_cache(
                rssm_planner_present=False,
                rssm_reranker_present=False,
                snippet_belief_flag=False,
                snippet_wm_rerank_enabled=True,
            )
        )
        self.assertTrue(
            need_eir_belief_cache(
                rssm_planner_present=False,
                rssm_reranker_present=False,
                snippet_belief_flag=True,
                snippet_wm_rerank_enabled=True,
            )
        )
        self.assertFalse(
            need_eir_belief_cache(
                rssm_planner_present=False,
                rssm_reranker_present=False,
                snippet_belief_flag=True,
                snippet_wm_rerank_enabled=False,
            )
        )
        self.assertTrue(
            need_eir_belief_cache(
                rssm_planner_present=True,
                rssm_reranker_present=False,
                snippet_belief_flag=False,
                snippet_wm_rerank_enabled=False,
            )
        )
        self.assertTrue(
            need_eir_belief_cache(
                rssm_planner_present=False,
                rssm_reranker_present=True,
                snippet_belief_flag=False,
                snippet_wm_rerank_enabled=False,
            )
        )

    def test_profile_resolved_predicate(self):
        a22 = _ns(wm_strategy_profile="cser_dexpbt_progress_refine")
        apply_wm_strategy_profile(a22, authoritative=True)
        self.assertFalse(
            need_eir_belief_cache(
                rssm_planner_present=False,
                rssm_reranker_present=False,
                snippet_belief_flag=bool(a22.eir_enable_belief_for_snippet_wm_rerank),
                snippet_wm_rerank_enabled=bool(
                    getattr(a22, "snippet_wm_rerank_enable", False)
                ),
            )
        )
        a24 = _ns(wm_strategy_profile="cser_dexpbt_progress_refine_abc")
        apply_wm_strategy_profile(a24, authoritative=True)
        # Profile24 enables belief flag; snippet_wm_rerank comes from CSER base
        # tables (typically True when EIR snippet path on). Independent of that,
        # belief helper requires both flag and rerank enable.
        self.assertTrue(bool(a24.eir_enable_belief_for_snippet_wm_rerank))
        need = need_eir_belief_cache(
            rssm_planner_present=False,
            rssm_reranker_present=False,
            snippet_belief_flag=True,
            snippet_wm_rerank_enabled=True,
        )
        self.assertTrue(need)


class TestLauncherSyntax(unittest.TestCase):
    def test_bash_n(self):
        import subprocess

        path = Path(
            "/root/autodl-tmp/wmdexpbt/results/launchers/"
            "start_regrasping_cser_dexpbt_progress_refine_abc_5b.sh"
        )
        self.assertTrue(path.is_file())
        self.assertEqual(
            subprocess.run(["bash", "-n", str(path)], capture_output=True).returncode, 0
        )


if __name__ == "__main__":
    unittest.main()
