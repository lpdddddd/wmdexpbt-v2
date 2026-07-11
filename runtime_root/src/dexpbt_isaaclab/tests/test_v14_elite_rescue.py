"""Unit tests for Strategy Zoo v1.4 elite distill rescue and selector copy fix."""

from __future__ import annotations

import argparse
import random
import unittest
from pathlib import Path

import torch

from dexpbt_isaaclab.pbt.pbt_profile_copy import ProfileCopyConfig, decide_profile_copy
from dexpbt_isaaclab.world_model.eir_mpc import build_actor_distill_mask
from dexpbt_isaaclab.world_model.eir_mpc import SRC_ELITE_LOCAL
from dexpbt_isaaclab.world_model.wm_strategy_profiles import (
    PROFILE_IDS,
    PROFILE_TABLES,
    apply_wm_strategy_profile,
    profile_capability_flags,
)

REPO_ROOT = Path(__file__).resolve().parents[4]
V14_LAUNCHER = (
    REPO_ROOT
    / "results/launchers/start_regrasping_pbt_wm_strategy_zoo_v14_elite_rescue_2b_debug.sh"
)
V14_ANALYZER = REPO_ROOT / "results/analyze_strategy_zoo_v14_elite.py"

ZOO_PROFILES_V14 = [
    "ppo_only",
    "elite_bank_actor_distill_fast_v2",
    "wm_pbt_source_selector_v2",
    "wm_pbt_source_selector",
    "elite_bank_actor_distill",
    "elite_bank_actor_distill_strong_v2",
]

RSSM_PROFILES = (
    "rssm_mppi_actor_distill",
    "light_rssm_mppi_actor_distill",
    "conservative_rssm_planner_distill",
    "light_conservative_rssm_planner_distill",
)


def _expand(profile: str):
    args = argparse.Namespace(wm_strategy_profile=profile)
    for dest in {k for t in PROFILE_TABLES.values() for k in t}:
        if not hasattr(args, dest):
            setattr(args, dest, None)
    apply_wm_strategy_profile(
        args,
        argv=["train.py", f"--wm_strategy_profile={profile}"],
        authoritative=(profile != "legacy"),
    )
    return args, profile_capability_flags(args)


def _population(
    scores: dict[int, float],
    frames: dict[int, int] | None = None,
) -> dict[int, dict]:
    pop: dict[int, dict] = {}
    for pid, score in scores.items():
        st: dict = {"pbt_ranking_score": float(score)}
        if frames and pid in frames:
            st["frame"] = int(frames[pid])
        pop[pid] = st
    return pop


class TestV14EliteProfiles(unittest.TestCase):
    def test_elite_fast_v2_clip_mode(self):
        self.assertIn("elite_bank_actor_distill_fast_v2", PROFILE_TABLES)
        self.assertEqual(PROFILE_IDS["elite_bank_actor_distill_fast_v2"], 12)
        args, caps = _expand("elite_bank_actor_distill_fast_v2")
        self.assertEqual(args.eir_actor_distill_large_delta_mode, "clip")
        self.assertFalse(args.eir_residual_apply_to_rollout)
        self.assertFalse(args.eir_use_rssm_mppi)
        self.assertFalse(caps["profile_rssm_mppi_enabled"])

    def test_elite_strong_v2_clip_mode(self):
        self.assertIn("elite_bank_actor_distill_strong_v2", PROFILE_TABLES)
        self.assertEqual(PROFILE_IDS["elite_bank_actor_distill_strong_v2"], 13)
        args, caps = _expand("elite_bank_actor_distill_strong_v2")
        self.assertEqual(args.eir_actor_distill_large_delta_mode, "clip")
        self.assertFalse(args.eir_residual_apply_to_rollout)
        self.assertFalse(args.eir_use_rssm_mppi)
        self.assertFalse(caps["profile_rssm_mppi_enabled"])

    def test_build_actor_distill_mask_clip_mode(self):
        eligible = torch.ones(2, dtype=torch.bool)
        teacher = torch.tensor([[2.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]])
        mu_r = torch.zeros(2, 4)
        mask, rej = build_actor_distill_mask(
            eligible=eligible,
            best_source=torch.tensor([SRC_ELITE_LOCAL, SRC_ELITE_LOCAL]),
            margin=torch.tensor([0.5, 0.5]),
            teacher=teacher,
            mu_r=mu_r,
            best_idx=torch.zeros(2, dtype=torch.long),
            allowed_source_ids=(SRC_ELITE_LOCAL,),
            min_margin=0.0,
            max_delta=0.35,
            tol_mean=0.01,
            tol_threshold=0.04,
            start_after_frames=0,
            frames=100,
            rssm_unc_bk=None,
            uncertainty_threshold=2.0,
            large_delta_mode="clip",
        )
        self.assertTrue(bool(mask[0]))
        self.assertEqual(rej["reject_large_delta"], 0)
        self.assertEqual(rej["clip_large_delta"], 1)

    def test_teacher_for_loss_clip_logic(self):
        mu_r = torch.zeros(1, 4)
        teacher = torch.tensor([[2.0, -2.0, 0.1, 0.0]])
        clip = 0.35
        raw_delta = teacher - mu_r
        clipped = raw_delta.clamp(-clip, clip)
        teacher_for_loss = (mu_r + clipped).detach()
        self.assertAlmostEqual(float(teacher_for_loss[0, 0]), 0.35)
        self.assertAlmostEqual(float(teacher_for_loss[0, 1]), -0.35)

    def test_source_min_gates_block_copy(self):
        cfg = ProfileCopyConfig(
            enable=True,
            after_frames=0,
            source_min_nsucc_mean=20.0,
            source_min_frames=600_000_000,
            bottom_k=2,
            top_k=2,
        )
        pop = _population({0: 40.0, 1: 10.0}, frames={0: 1_000_000_000, 1: 100_000_000})
        low_nsucc = decide_profile_copy(
            cfg=cfg,
            env_frames=700_000_000,
            target_policy_idx=1,
            source_policy_idx=0,
            target_profile_name="ppo_only",
            source_profile_name="wm_pbt_source_selector",
            source_generation=0,
            diversity_count=4,
            profiles_set={"ppo_only", "wm_pbt_source_selector", "elite_bank_actor_distill"},
            population=pop,
            source_meta={},
            target_meta={},
            source_checkpoint_type="live",
            source_frames=1_000_000_000,
        )
        self.assertEqual(low_nsucc.reject_reason, "source_nsucc_too_low")
        low_frames = decide_profile_copy(
            cfg=cfg,
            env_frames=700_000_000,
            target_policy_idx=1,
            source_policy_idx=0,
            target_profile_name="ppo_only",
            source_profile_name="wm_pbt_source_selector",
            source_generation=0,
            diversity_count=4,
            profiles_set={"ppo_only", "wm_pbt_source_selector", "elite_bank_actor_distill"},
            population=_population({0: 40.0, 1: 10.0}, frames={0: 100_000_000, 1: 100_000_000}),
            source_meta={},
            target_meta={},
            source_checkpoint_type="live",
            source_frames=100_000_000,
            window_nsucc={0: 30.0, 1: 10.0},
        )
        self.assertEqual(low_frames.reject_reason, "source_frames_too_low")

    def test_top1_override_passes_target_protected(self):
        cfg = ProfileCopyConfig(
            enable=True,
            after_frames=0,
            top1_override_enable=True,
            top1_override_after_frames=900_000_000,
            top1_override_min_nsucc_gap=5.0,
            top1_override_min_diversity=2,
            top1_override_allow_target_protected=True,
            bottom_k=2,
            top_k=2,
            base_prob=1.0,
        )
        pop = _population({0: 40.0, 1: 10.0})
        out = decide_profile_copy(
            cfg=cfg,
            env_frames=950_000_000,
            target_policy_idx=1,
            source_policy_idx=0,
            target_profile_name="ppo_only",
            source_profile_name="wm_pbt_source_selector",
            source_generation=0,
            diversity_count=3,
            profiles_set={"ppo_only", "wm_pbt_source_selector", "elite_bank_actor_distill"},
            population=pop,
            source_meta={},
            target_meta={},
            source_checkpoint_type="live",
            target_protected_until=10**12,
            window_nsucc={0: 30.0, 1: 20.0},
            rng=random.Random(0),
        )
        self.assertTrue(out.top1_override_used)
        self.assertNotEqual(out.reject_reason, "target_protected")
        self.assertTrue(out.accepted or out.decision == "rng_skip")

    def test_top1_override_relaxes_diversity_to_two_not_one(self):
        cfg = ProfileCopyConfig(
            enable=True,
            after_frames=0,
            keep_diversity_min_profiles=3,
            top1_override_enable=True,
            top1_override_after_frames=900_000_000,
            top1_override_min_nsucc_gap=5.0,
            top1_override_min_diversity=2,
            bottom_k=2,
            top_k=2,
            base_prob=1.0,
        )
        pop = _population({0: 40.0, 1: 10.0})
        profiles_set = {
            "ppo_only",
            "wm_pbt_source_selector",
            "elite_bank_actor_distill",
            "elite_bank_actor_distill_fast",
        }
        out = decide_profile_copy(
            cfg=cfg,
            env_frames=950_000_000,
            target_policy_idx=1,
            source_policy_idx=0,
            target_profile_name="ppo_only",
            source_profile_name="wm_pbt_source_selector",
            source_generation=0,
            diversity_count=4,
            profiles_set=profiles_set,
            population=pop,
            source_meta={},
            target_meta={},
            source_checkpoint_type="live",
            window_nsucc={0: 30.0, 1: 20.0},
            rng=random.Random(0),
        )
        self.assertNotEqual(out.reject_reason, "diversity_min")
        cfg_one = ProfileCopyConfig(
            enable=True,
            after_frames=0,
            keep_diversity_min_profiles=3,
            top1_override_enable=True,
            top1_override_after_frames=900_000_000,
            top1_override_min_nsucc_gap=5.0,
            top1_override_min_diversity=1,
            bottom_k=2,
            top_k=2,
        )
        out_one = decide_profile_copy(
            cfg=cfg_one,
            env_frames=950_000_000,
            target_policy_idx=1,
            source_policy_idx=0,
            target_profile_name="ppo_only",
            source_profile_name="wm_pbt_source_selector",
            source_generation=0,
            diversity_count=4,
            profiles_set=profiles_set,
            population=pop,
            source_meta={},
            target_meta={},
            source_checkpoint_type="live",
            window_nsucc={0: 30.0, 1: 20.0},
        )
        self.assertEqual(out_one.diversity_after_copy, 3)

    def test_v14_launcher_profile_list(self):
        self.assertTrue(V14_LAUNCHER.is_file())
        text = V14_LAUNCHER.read_text()
        line = next(l for l in text.splitlines() if l.startswith("PROFILE_LIST="))
        profiles = line.split("=", 1)[1].strip().strip('"').split(",")
        self.assertEqual(profiles, ZOO_PROFILES_V14)

    def test_v14_launcher_no_rssm_performance_profile(self):
        text = V14_LAUNCHER.read_text()
        for prof in RSSM_PROFILES:
            self.assertNotIn(prof, text)

    def test_analyzer_output_defaults_to_results_docs(self):
        self.assertTrue(V14_ANALYZER.is_file())
        text = V14_ANALYZER.read_text()
        self.assertIn("STRATEGY_ZOO_V14_ELITE_RESCUE_REPORT", text)
        self.assertIn("RESULTS_DOCS", text)
        self.assertNotIn("wmdexpbt/docs/", text)


if __name__ == "__main__":
    unittest.main()
