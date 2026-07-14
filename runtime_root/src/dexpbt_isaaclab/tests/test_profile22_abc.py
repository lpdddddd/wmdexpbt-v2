"""Unit tests for Profile22 Experiment ABC review fixes (no training)."""

from __future__ import annotations

import argparse
import json
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


class TestBeliefIntOrMissing(unittest.TestCase):
    def test_preserves_zero(self):
        from dexpbt_isaaclab.world_model.eir_belief import int_or_missing

        self.assertEqual(int_or_missing(0), 0)
        self.assertEqual(int_or_missing(None), -1)
        self.assertEqual(int_or_missing(3), 3)


class TestBankForeignEvictionWithLocalReflux(unittest.TestCase):
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

    def test_incoming_local_rows_are_not_counted_as_foreign(self):
        bank = self._mk(8, 0, 0.02, 1.0, capacity=10)  # existing foreign
        incoming = self._mk(4, 2, 0.02, 50.0, capacity=10).state_dict()  # local reflux
        kept = bank.merge_tolerance_aware(
            incoming, local_policy_idx=2, preserve_local_fraction=0.2
        )
        d = bank.last_update_diagnostics
        self.assertEqual(int(d["incoming_kept_count"]), kept)
        self.assertEqual(int(d["incoming_local_kept"]), kept)
        self.assertEqual(int(d["incoming_foreign_kept"]), 0)
        self.assertEqual(int(d["foreign_evicted_count"]), max(0, 8 - int(d["existing_foreign_kept"])))


class TestSnapshotWorkspaceHealth(unittest.TestCase):
    TICK = 200_000_000

    def _publish(self, peer_ws: Path, *, frame: int, lineage: str, prefix: str):
        from dexpbt_isaaclab.pbt.pbt_control_plane import snapshot_commit_barrier

        peer_ws.mkdir(parents=True, exist_ok=True)
        model_src = peer_ws / f"{prefix}.model.tmp"
        meta_src = peer_ws / f"{prefix}.meta.tmp"
        publish_model = peer_ws / f"{prefix}.pth"
        publish_meta = peer_ws / f"{prefix}.json"
        torch.save({"w": torch.arange(4).float()}, model_src)
        state = {
            "policy_idx": 1,
            "checkpoint": str(publish_model.resolve()),
            "lineage_id": lineage,
            "population_frame": int(frame),
            "frame": int(frame),
            "objective": 0.95,
            "pbt_ranking_score": 0.95,
            "nsucc_mean": 42.0,
            "iteration": int(frame // self.TICK),
            "params": {},
        }
        meta_src.write_text(json.dumps(state))
        commit_dir = snapshot_commit_barrier(
            staging_dir=peer_ws / f"{prefix}.staging",
            model_src=model_src,
            meta_src=meta_src,
            publish_model=publish_model,
            publish_meta=publish_meta,
            manifest={
                "population_frame": int(frame),
                "lineage_id": lineage,
                "lineage_start_pf": 0,
                "weight_age_frame": int(frame),
                "policy_idx": 1,
                "pbt_iteration": int(frame // self.TICK),
            },
            tick=self.TICK,
        )
        state = json.loads(publish_meta.read_text())
        return state, commit_dir

    def test_legal_snapshot_without_completion_marker_passes(self):
        from dexpbt_isaaclab.pbt.pbt_live_source_health import (
            inspect_peer_snapshot_health,
        )

        with tempfile.TemporaryDirectory() as tmp:
            peer = Path(tmp) / "001"
            state, _ = self._publish(
                peer, frame=400_000_000, lineage="L1", prefix="iter0002"
            )
            self.assertFalse((peer / "population_frame_complete.json").exists())
            h = inspect_peer_snapshot_health(
                policy_workspace=peer, state=state, tick=self.TICK
            )
            self.assertTrue(h.snapshot_commit_valid, h.snapshot_commit_reason)
            self.assertTrue(h.lineage_valid)
            self.assertTrue(h.freshness_valid, h.freshness_reason)
            self.assertTrue(h.final_health_allowed, h.final_health_reason)
            common = evaluate_common_live_source_eligibility(
                has_state=True,
                ranking_score=0.95,
                nsucc_mean=42.0,
                source_gate_ok=True,
                source_gate_reason="ok",
                restart_burnin_ok=True,
                restart_burnin_reason="ok",
                checkpoint_exists=h.checkpoint_exists,
                checkpoint_commit_ok=h.checkpoint_commit_ok,
                lineage_ok=h.lineage_ok,
                freshness_ok=h.freshness_ok,
                probation_ok=True,
                collapse_ok=True,
            )
            self.assertTrue(common.eligible, common.reason)

    def test_completion_marker_neither_required_nor_substitute(self):
        from dexpbt_isaaclab.pbt.pbt_live_source_health import (
            inspect_peer_snapshot_health,
        )

        with tempfile.TemporaryDirectory() as tmp:
            peer = Path(tmp) / "001"
            state, commit_dir = self._publish(
                peer, frame=200_000_000, lineage="L1", prefix="iter0001"
            )
            (peer / "population_frame_complete.json").write_text(
                json.dumps({"complete": True})
            )
            h_ok = inspect_peer_snapshot_health(
                policy_workspace=peer, state=state, tick=self.TICK
            )
            self.assertTrue(h_ok.final_health_allowed)
            # Marker cannot rescue a destroyed commit.
            import shutil

            shutil.rmtree(commit_dir)
            h_bad = inspect_peer_snapshot_health(
                policy_workspace=peer, state=state, tick=self.TICK
            )
            self.assertFalse(h_bad.final_health_allowed)
            self.assertIn("commit_dir", h_bad.final_health_reason)

    def test_corrupt_latest_commit_rejected(self):
        from dexpbt_isaaclab.pbt.pbt_live_source_health import (
            inspect_peer_snapshot_health,
        )

        with tempfile.TemporaryDirectory() as tmp:
            peer = Path(tmp) / "001"
            state, _ = self._publish(
                peer, frame=200_000_000, lineage="L1", prefix="iter0001"
            )
            (peer / "LATEST.commit").write_text("{not-json")
            h = inspect_peer_snapshot_health(
                policy_workspace=peer, state=state, tick=self.TICK
            )
            self.assertFalse(h.final_health_allowed)
            self.assertEqual(h.snapshot_commit_reason, "latest_commit_corrupt")

    def test_missing_commit_dir_rejected(self):
        from dexpbt_isaaclab.pbt.pbt_live_source_health import (
            inspect_peer_snapshot_health,
        )

        with tempfile.TemporaryDirectory() as tmp:
            peer = Path(tmp) / "001"
            state, commit_dir = self._publish(
                peer, frame=200_000_000, lineage="L1", prefix="iter0001"
            )
            import shutil

            shutil.rmtree(commit_dir)
            h = inspect_peer_snapshot_health(
                policy_workspace=peer, state=state, tick=self.TICK
            )
            self.assertFalse(h.final_health_allowed)
            self.assertFalse(h.commit_dir_exists)

    def test_missing_manifest_rejected(self):
        from dexpbt_isaaclab.pbt.pbt_live_source_health import (
            inspect_peer_snapshot_health,
        )

        with tempfile.TemporaryDirectory() as tmp:
            peer = Path(tmp) / "001"
            state, commit_dir = self._publish(
                peer, frame=200_000_000, lineage="L1", prefix="iter0001"
            )
            (commit_dir / "MANIFEST").unlink()
            h = inspect_peer_snapshot_health(
                policy_workspace=peer, state=state, tick=self.TICK
            )
            self.assertFalse(h.snapshot_commit_valid)
            self.assertFalse(h.final_health_allowed)

    def test_model_hash_mismatch_rejected(self):
        from dexpbt_isaaclab.pbt.pbt_live_source_health import (
            inspect_peer_snapshot_health,
        )

        with tempfile.TemporaryDirectory() as tmp:
            peer = Path(tmp) / "001"
            state, commit_dir = self._publish(
                peer, frame=200_000_000, lineage="L1", prefix="iter0001"
            )
            torch.save({"w": torch.ones(4)}, commit_dir / "model.pth")
            h = inspect_peer_snapshot_health(
                policy_workspace=peer, state=state, tick=self.TICK
            )
            self.assertFalse(h.snapshot_commit_valid)
            self.assertIn("member_corrupt", h.snapshot_commit_reason)

    def test_meta_hash_mismatch_rejected(self):
        from dexpbt_isaaclab.pbt.pbt_live_source_health import (
            inspect_peer_snapshot_health,
        )

        with tempfile.TemporaryDirectory() as tmp:
            peer = Path(tmp) / "001"
            state, commit_dir = self._publish(
                peer, frame=200_000_000, lineage="L1", prefix="iter0001"
            )
            (commit_dir / "meta.json").write_text(json.dumps({"tampered": True}))
            h = inspect_peer_snapshot_health(
                policy_workspace=peer, state=state, tick=self.TICK
            )
            self.assertFalse(h.snapshot_commit_valid)
            self.assertIn("member_corrupt", h.snapshot_commit_reason)

    def test_lineage_mismatch_rejected(self):
        from dexpbt_isaaclab.pbt.pbt_live_source_health import (
            inspect_peer_snapshot_health,
        )

        with tempfile.TemporaryDirectory() as tmp:
            peer = Path(tmp) / "001"
            state, _ = self._publish(
                peer, frame=200_000_000, lineage="L1", prefix="iter0001"
            )
            state = dict(state)
            state["lineage_id"] = "OTHER"
            h = inspect_peer_snapshot_health(
                policy_workspace=peer, state=state, tick=self.TICK
            )
            self.assertTrue(h.snapshot_commit_valid)
            self.assertFalse(h.lineage_valid)
            self.assertFalse(h.final_health_allowed)

    def test_frame_mismatch_rejected(self):
        from dexpbt_isaaclab.pbt.pbt_live_source_health import (
            inspect_peer_snapshot_health,
        )

        with tempfile.TemporaryDirectory() as tmp:
            peer = Path(tmp) / "001"
            state, _ = self._publish(
                peer, frame=200_000_000, lineage="L1", prefix="iter0001"
            )
            state = dict(state)
            state["population_frame"] = 400_000_000
            state["frame"] = 400_000_000
            h = inspect_peer_snapshot_health(
                policy_workspace=peer, state=state, tick=self.TICK
            )
            self.assertEqual(h.freshness_reason, "population_frame_mismatch")
            self.assertFalse(h.final_health_allowed)

    def test_checkpoint_missing_rejected(self):
        from dexpbt_isaaclab.pbt.pbt_live_source_health import (
            inspect_peer_snapshot_health,
        )

        with tempfile.TemporaryDirectory() as tmp:
            peer = Path(tmp) / "001"
            state, _ = self._publish(
                peer, frame=200_000_000, lineage="L1", prefix="iter0001"
            )
            Path(state["checkpoint"]).unlink()
            h = inspect_peer_snapshot_health(
                policy_workspace=peer, state=state, tick=self.TICK
            )
            self.assertFalse(h.checkpoint_exists)
            self.assertFalse(h.final_health_allowed)

    def test_latest_points_to_newer_commit_only(self):
        from dexpbt_isaaclab.pbt.pbt_live_source_health import (
            inspect_peer_snapshot_health,
        )

        with tempfile.TemporaryDirectory() as tmp:
            peer = Path(tmp) / "001"
            state_old, old_dir = self._publish(
                peer, frame=200_000_000, lineage="L1", prefix="iter0001"
            )
            state_new, new_dir = self._publish(
                peer, frame=400_000_000, lineage="L1", prefix="iter0002"
            )
            self.assertTrue(old_dir.is_dir())
            self.assertTrue(new_dir.is_dir())
            latest = json.loads((peer / "LATEST.commit").read_text())
            self.assertEqual(latest["commit_dir"], new_dir.name)
            h_new = inspect_peer_snapshot_health(
                policy_workspace=peer, state=state_new, tick=self.TICK
            )
            self.assertTrue(h_new.final_health_allowed)
            # Old state vs LATEST(new) → frame/identity fail-close.
            h_stale = inspect_peer_snapshot_health(
                policy_workspace=peer, state=state_old, tick=self.TICK
            )
            self.assertFalse(h_stale.final_health_allowed)

    def test_high_score_invalid_snapshot_not_selectable(self):
        from dexpbt_isaaclab.pbt.pbt_live_source_health import (
            inspect_peer_snapshot_health,
        )

        with tempfile.TemporaryDirectory() as tmp:
            peer = Path(tmp) / "001"
            state, commit_dir = self._publish(
                peer, frame=200_000_000, lineage="L1", prefix="iter0001"
            )
            (commit_dir / "MANIFEST").unlink()
            h = inspect_peer_snapshot_health(
                policy_workspace=peer, state=state, tick=self.TICK
            )
            common = evaluate_common_live_source_eligibility(
                has_state=True,
                ranking_score=9.9,
                nsucc_mean=99.0,
                source_gate_ok=True,
                source_gate_reason="ok",
                restart_burnin_ok=True,
                restart_burnin_reason="ok",
                checkpoint_exists=h.checkpoint_exists,
                checkpoint_commit_ok=h.checkpoint_commit_ok,
                lineage_ok=h.lineage_ok,
                freshness_ok=h.freshness_ok,
                probation_ok=True,
                collapse_ok=True,
            )
            self.assertFalse(common.eligible)
            decision = pick_chronic_source(
                sorted_policies=[1, 0],
                population={
                    0: {"objective": 0.01, "nsucc_mean": 1.0},
                    1: {
                        **state,
                        "objective": 9.9,
                        "pbt_ranking_score": 9.9,
                        "nsucc_mean": 99.0,
                    },
                },
                target_idx=0,
                target_score=0.01,
                target_nsucc_mean=1.0,
                n_top=1,
                n_mid=0,
                min_score_gap=0.0,
                skill_delta_by_policy={1: (0.05, "ok")},
                drop_ratio_by_policy={1: 0.0},
                common_health_by_policy={1: common},
            )
            self.assertEqual(decision.action, "skip")
            self.assertEqual(decision.source_policy, -1)

    def test_legal_snapshot_enables_chronic_skill_and_fallback_paths(self):
        from dexpbt_isaaclab.pbt.pbt_live_source_health import (
            inspect_peer_snapshot_health,
        )

        with tempfile.TemporaryDirectory() as tmp:
            peer = Path(tmp) / "001"
            state, _ = self._publish(
                peer, frame=200_000_000, lineage="L1", prefix="iter0001"
            )
            h = inspect_peer_snapshot_health(
                policy_workspace=peer, state=state, tick=self.TICK
            )
            common = evaluate_common_live_source_eligibility(
                has_state=True,
                ranking_score=0.95,
                nsucc_mean=42.0,
                source_gate_ok=True,
                source_gate_reason="ok",
                restart_burnin_ok=True,
                restart_burnin_reason="ok",
                checkpoint_exists=h.checkpoint_exists,
                checkpoint_commit_ok=h.checkpoint_commit_ok,
                lineage_ok=h.lineage_ok,
                freshness_ok=h.freshness_ok,
                probation_ok=True,
                collapse_ok=True,
            )
            self.assertTrue(common.eligible)
            ok_delta, _, fallback = live_source_ok_for_chronic(
                skill_delta=0.05,
                skill_delta_why="ok",
                source_nsucc_mean=42.0,
                target_nsucc_mean=1.0,
                source_recent_drop_ratio=0.0,
            )
            self.assertTrue(ok_delta)
            self.assertFalse(fallback)
            ok_fb, reason, fallback2 = live_source_ok_for_chronic(
                skill_delta=None,
                skill_delta_why="mixed_tol_bins",
                source_nsucc_mean=42.0,
                target_nsucc_mean=1.0,
                source_recent_drop_ratio=0.0,
            )
            self.assertTrue(ok_fb, reason)
            self.assertTrue(fallback2)
            decision = pick_chronic_source(
                sorted_policies=[1, 0],
                population={
                    0: {"objective": 0.01, "nsucc_mean": 1.0},
                    1: {
                        **state,
                        "objective": 0.95,
                        "pbt_ranking_score": 0.95,
                        "nsucc_mean": 42.0,
                    },
                },
                target_idx=0,
                target_score=0.01,
                target_nsucc_mean=1.0,
                n_top=1,
                n_mid=0,
                min_score_gap=0.0,
                skill_delta_by_policy={1: (0.05, "ok")},
                drop_ratio_by_policy={1: 0.0},
                common_health_by_policy={1: common},
            )
            self.assertEqual(decision.action, "copy")
            self.assertEqual(decision.source_policy, 1)


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
