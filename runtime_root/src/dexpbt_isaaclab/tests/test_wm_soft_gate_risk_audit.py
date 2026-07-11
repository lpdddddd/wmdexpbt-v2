"""Risk-audit unit tests for PBT-WM Soft Activation Gate v1."""

from __future__ import annotations

import unittest
from types import SimpleNamespace

import torch

from dexpbt_isaaclab.world_model import eir_mpc
from dexpbt_isaaclab.world_model.eir_mpc import (
    EIRConfig,
    EIR_TOL_ABOVE_START,
    SRC_ELITE_ARCHIVE,
    SRC_ELITE_LOCAL,
    compute_actor_distill_activation,
    compute_eir_loss,
    generate_eir_candidates,
)
from dexpbt_isaaclab.world_model.wm_activation_gate import (
    WMActivationConfig,
    blend_wm_ranking_score,
    compute_wm_activation,
    compute_wm_pbt_source_activation_weight,
)


from dexpbt_isaaclab.world_model.wm_strategy_profiles import apply_wm_strategy_profile
from tests.test_eir_mpc import TestComputeEIRLoss


class _MockBank:
    size = 8

    def __init__(self) -> None:
        self.calls: list[tuple[float | None, int]] = []

    def retrieve_candidates(self, feat, *, k: int, stage=None):
        self.calls.append((stage, int(k)))
        B = int(feat.shape[0])
        A = 4
        acts = torch.zeros(B, k, A)
        sim = torch.ones(B, k)
        pri = torch.ones(B, k)
        return acts, sim, pri, True


class TestHardTolConsistency(unittest.TestCase):
    def test_hard_tol_time_ramp_differs_between_modules(self):
        """Risk A: wm_activation_gate hard_tol uses time ramp; eir actor path step-on."""
        cfg = WMActivationConfig(
            gate_mode="hard_tol",
            hard_tol=0.04,
            start_after_frames=100,
            ramp_frames=100,
        )
        wm = compute_wm_activation(
            cfg=cfg, frames=150, tol=0.03, margin=0.1, uncertainty=0.1, source="rssm"
        )
        self.assertGreater(wm.time_weight, 0.0)
        self.assertLess(wm.time_weight, 1.0)

        act = compute_actor_distill_activation(
            cfg=cfg,
            frames=150,
            tol_mean=0.03,
            margin=torch.tensor([0.1]),
            best_source=torch.tensor([eir_mpc.SRC_RSSM]),
            best_unc=None,
        )
        self.assertEqual(act["time_weight"], 1.0)


class TestCandidatePoolTolGate(unittest.TestCase):
    TOLS = (0.065, 0.055, 0.045, 0.035)

    def _elite_cfg(self) -> EIRConfig:
        args = SimpleNamespace(wm_strategy_profile="elite_bank_actor_distill")
        apply_wm_strategy_profile(args)
        return EIRConfig(
            enabled=True,
            action_dim=4,
            feature_dim=8,
            elite_candidate_k=int(args.eir_elite_candidate_k),
            archive_candidate_k=int(args.eir_archive_candidate_k),
            guidance_enable=bool(args.eir_guidance_enable),
            guidance_start_tol=float(args.eir_guidance_start_tol),
            floor_tol=0.02,
            hard_floor_tol=0.012,
        )

    def test_stage_and_elite_retrieval_above_start_tol(self):
        cfg = self._elite_cfg()
        bank = _MockBank()
        archive = _MockBank()
        mu = torch.zeros(2, 4)
        feat = torch.randn(2, 8)

        for tol in self.TOLS:
            with self.subTest(tol=tol):
                stage = eir_mpc._stage_for_tol(cfg, tol)
                if tol <= cfg.guidance_start_tol:
                    self.assertEqual(stage, 3)
                else:
                    self.assertIsNone(stage)
                bank.calls.clear()
                archive.calls.clear()
                out = generate_eir_candidates(
                    mu,
                    cfg=cfg,
                    bank=bank,
                    archive_bank=archive,
                    feat_r=feat,
                    tol_mean=tol,
                    stage=stage,
                )
                src = out["source_id"].unique().tolist()
                self.assertIn(SRC_ELITE_LOCAL, src)
                self.assertIn(SRC_ELITE_ARCHIVE, src)
                self.assertEqual(len(bank.calls), 1)
                self.assertEqual(len(archive.calls), 1)

    def test_soft_actor_distill_not_blocked_by_outer_start_tol(self):
        bridge, model, scorer, _ = TestComputeEIRLoss()._setup()
        bridge.scorer = scorer
        bridge.eir_cfg.actor_distill_enable = True
        bridge.eir_cfg.actor_distill_gate_mode = "soft_tol"
        bridge.eir_cfg.actor_distill_activation_enable = True
        bridge.eir_cfg.actor_distill_start_after_frames = 0
        bridge.eir_cfg.start_after_frames = 0
        bridge.eir_cfg.scorer_warmup_iters = 0
        bridge.scorer_warmup_iters_done = 999
        bridge.last_target_valid_fraction = 1.0
        bridge.last_nsucc_mean = -1.0
        bridge.cache.tol_mean = 0.055
        bridge.eir_cfg.min_risky_fraction = 0.0
        compute_eir_loss(bridge, frames=600_000_000)
        self.assertNotEqual(bridge.last_block_reason, EIR_TOL_ABOVE_START)


class TestWMPBTSourceActivation(unittest.TestCase):
    def test_before_start_after_frames_zero_weight(self):
        w, t, u, *_ = compute_wm_pbt_source_activation_weight(
            enable=True,
            gate_mode="confidence_only",
            frames=100,
            start_after_frames=300_000_000,
            ramp_frames=300_000_000,
            wm_valid=True,
            wm_uncertainty=0.1,
            max_uncertainty=0.5,
        )
        self.assertEqual(w, 0.0)
        self.assertEqual(t, 0.0)

    def test_blend_leaves_real_score_when_weight_zero(self):
        real, wm = 40.0, 48.0
        self.assertEqual(blend_wm_ranking_score(real, wm, 0.0), real)

    def test_blend_applies_partial_wm_delta(self):
        real, wm = 40.0, 50.0
        self.assertAlmostEqual(blend_wm_ranking_score(real, wm, 0.5), 45.0)

    def test_invalid_eval_yields_zero_activation(self):
        w, *_ = compute_wm_pbt_source_activation_weight(
            enable=True,
            gate_mode="confidence_only",
            frames=400_000_000,
            start_after_frames=300_000_000,
            ramp_frames=300_000_000,
            wm_valid=False,
            wm_uncertainty=0.1,
            max_uncertainty=0.5,
        )
        self.assertEqual(w, 0.0)


class TestProfileAuthoritative(unittest.TestCase):
    def test_ppo_only_overrides_restart_wm_flags(self):
        args = SimpleNamespace(
            wm_strategy_profile="ppo_only",
            enable_wm=True,
            wm_strategy="rssm_belief_value",
            eir_enable=True,
        )
        apply_wm_strategy_profile(args, authoritative=True)
        self.assertFalse(args.enable_wm)
        self.assertEqual(args.wm_strategy, "none")
        self.assertFalse(args.eir_enable)

    def test_rssm_profile_overrides_conflicting_restart_flags(self):
        args = SimpleNamespace(
            wm_strategy_profile="rssm_mppi_actor_distill",
            eir_enable=False,
            wm_strategy="none",
        )
        apply_wm_strategy_profile(args, authoritative=True)
        self.assertTrue(args.eir_enable)
        self.assertEqual(args.wm_strategy, "rssm_belief_value")



class TestCrossProfileCheckpointRestore(unittest.TestCase):
    def test_align_optimizer_state_drops_donor_only_groups(self):
        import torch
        from dexpbt_isaaclab.scripts.train import _align_optimizer_state_to_agent

        class _Agent:
            pass

        base = torch.nn.Linear(2, 2)
        agent = _Agent()
        agent.optimizer = torch.optim.Adam(base.parameters(), lr=1e-3)

        donor_base = torch.nn.Linear(2, 2)
        donor_eir = torch.nn.Linear(2, 2)
        donor_opt = torch.optim.Adam(
            [
                {"params": donor_base.parameters(), "lr": 1e-3},
                {"params": donor_eir.parameters(), "lr": 1e-3, "name": "eir_scorer"},
            ]
        )
        ckpt_opt = donor_opt.state_dict()
        aligned = _align_optimizer_state_to_agent(agent, ckpt_opt)
        self.assertEqual(len(aligned["param_groups"]), 1)
        agent.optimizer.load_state_dict(aligned)

if __name__ == "__main__":
    unittest.main()
