"""Tests for soft EIR candidate-retrieval gating."""

from __future__ import annotations

import unittest
from types import SimpleNamespace

import torch

from dexpbt_isaaclab.world_model.eir_candidate_gate import (
    CandidateGateConfig,
    allow_retrieval,
    elite_candidate_weight,
    rssm_candidate_weight,
)
from dexpbt_isaaclab.world_model.eir_mpc import (
    EIRConfig,
    SRC_ELITE_ARCHIVE,
    SRC_ELITE_LOCAL,
    generate_eir_candidates,
)
from dexpbt_isaaclab.world_model.wm_strategy_profiles import (
    PROFILE_TABLES,
    apply_wm_strategy_profile,
)


class _MockBank:
    size = 8

    def retrieve_candidates(self, feat, *, k: int, stage=None):
        B = int(feat.shape[0])
        acts = torch.zeros(B, k, 4)
        sim = torch.ones(B, k)
        pri = torch.ones(B, k)
        return acts, sim, pri, True


class TestCandidateGateHelpers(unittest.TestCase):
    def test_elite_gate_weight_increases_as_tol_tightens(self):
        cfg = CandidateGateConfig(mode="profile_soft")
        w_loose = elite_candidate_weight(cfg, 0.055)
        w_tight = elite_candidate_weight(cfg, 0.045)
        self.assertGreaterEqual(w_tight, w_loose)
        self.assertEqual(elite_candidate_weight(
            CandidateGateConfig(mode="hard_tol"), 0.060
        ), 1.0)

    def test_conservative_rssm_later_gate(self):
        cfg = CandidateGateConfig(
            mode="profile_soft",
            profile_name="conservative_rssm_planner_distill",
            conservative_rssm_tol_start=0.055,
            conservative_rssm_tol_full=0.035,
        )
        self.assertFalse(allow_retrieval(
            rssm_candidate_weight(cfg, 0.060), 0.05
        ))
        self.assertTrue(allow_retrieval(
            rssm_candidate_weight(cfg, 0.045), 0.05
        ))


class TestGenerateEIRCandidatesGating(unittest.TestCase):
    def _elite_cfg(self) -> EIRConfig:
        args = SimpleNamespace(wm_strategy_profile="elite_bank_actor_distill")
        apply_wm_strategy_profile(args)
        return EIRConfig(
            enabled=True,
            action_dim=4,
            feature_dim=8,
            elite_candidate_k=int(args.eir_elite_candidate_k),
            archive_candidate_k=int(args.eir_archive_candidate_k),
            candidate_gate_mode=str(args.eir_candidate_gate_mode),
            elite_candidate_tol_start=float(args.eir_elite_candidate_tol_start),
            elite_candidate_tol_full=float(args.eir_elite_candidate_tol_full),
            candidate_min_weight_for_retrieval=float(
                args.eir_candidate_min_weight_for_retrieval
            ),
            wm_strategy_profile_name="elite_bank_actor_distill",
        )

    def test_elite_profile_retrieves_at_tol_060(self):
        cfg = self._elite_cfg()
        bank = _MockBank()
        archive = _MockBank()
        mu = torch.zeros(2, 4)
        feat = torch.randn(2, 8)
        out = generate_eir_candidates(
            mu, cfg=cfg, bank=bank, archive_bank=archive,
            feat_r=feat, tol_mean=0.060,
        )
        src = out["source_id"].unique().tolist()
        self.assertIn(SRC_ELITE_LOCAL, src)
        self.assertIn(SRC_ELITE_ARCHIVE, src)
        self.assertGreaterEqual(float(out["elite_candidate_gate_weight"]), 0.05)

    def test_rssm_profile_gate_at_tol_055(self):
        args = SimpleNamespace(wm_strategy_profile="rssm_mppi_actor_distill")
        apply_wm_strategy_profile(args)
        cfg = EIRConfig(
            enabled=True,
            action_dim=4,
            feature_dim=8,
            candidate_gate_mode=str(args.eir_candidate_gate_mode),
            rssm_candidate_tol_start=float(args.eir_rssm_candidate_tol_start),
            rssm_candidate_tol_full=float(args.eir_rssm_candidate_tol_full),
            candidate_min_weight_for_retrieval=float(
                args.eir_candidate_min_weight_for_retrieval
            ),
            wm_strategy_profile_name="rssm_mppi_actor_distill",
        )
        gate = cfg.candidate_gate_config()
        w = rssm_candidate_weight(gate, 0.055)
        self.assertTrue(allow_retrieval(w, gate.min_weight_for_retrieval))

    def test_light_rssm_profile_gate_tols(self):
        args = SimpleNamespace(wm_strategy_profile="light_rssm_mppi_actor_distill")
        apply_wm_strategy_profile(args)
        cfg = CandidateGateConfig(
            mode="profile_soft",
            profile_name="light_rssm_mppi_actor_distill",
            rssm_tol_start=float(args.eir_rssm_candidate_tol_start),
            rssm_tol_full=float(args.eir_rssm_candidate_tol_full),
        )
        w = rssm_candidate_weight(cfg, 0.060)
        self.assertGreater(w, 0.0)

    def test_light_conservative_profile_gate_tols(self):
        args = SimpleNamespace(
            wm_strategy_profile="light_conservative_rssm_planner_distill"
        )
        apply_wm_strategy_profile(args)
        cfg = CandidateGateConfig(
            mode="profile_soft",
            profile_name="light_conservative_rssm_planner_distill",
            conservative_rssm_tol_start=float(
                args.eir_conservative_rssm_candidate_tol_start
            ),
            conservative_rssm_tol_full=float(
                args.eir_conservative_rssm_candidate_tol_full
            ),
        )
        self.assertFalse(allow_retrieval(rssm_candidate_weight(cfg, 0.060), 0.05))
        self.assertTrue(allow_retrieval(rssm_candidate_weight(cfg, 0.040), 0.05))

    def test_all_zoo_profiles_keep_residual_apply_off(self):
        for profile, table in PROFILE_TABLES.items():
            if not table:
                continue
            self.assertFalse(
                table.get("eir_residual_apply_to_rollout", False),
                msg=f"{profile} must keep residual rollout apply off",
            )

    def test_teacher_candidate_available_above_004_counts_rssm_gate(self):
        tol = 0.055
        tol_above_004 = tol > 0.04
        elite_w = 0.0
        rssm_w = rssm_candidate_weight(
            CandidateGateConfig(mode="profile_soft"), tol
        )
        min_w = 0.05
        elite_avail = tol_above_004 and allow_retrieval(elite_w, min_w)
        rssm_avail = tol_above_004 and allow_retrieval(rssm_w, min_w)
        any_avail = elite_avail or rssm_avail
        self.assertFalse(elite_avail)
        self.assertTrue(rssm_avail)
        self.assertTrue(any_avail)
        self.assertEqual(float(any_avail), float(rssm_avail))
        self.assertEqual(float(any_avail), 1.0)


if __name__ == "__main__":
    unittest.main()
