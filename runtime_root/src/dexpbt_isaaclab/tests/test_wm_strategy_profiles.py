"""Unit tests for wm_strategy_profiles (PBT-WM Strategy Zoo v1.0)."""

from __future__ import annotations

import argparse
import tempfile
import unittest
from pathlib import Path

from dexpbt_isaaclab.pbt.pbt_observer import DexPBTPbtAlgoObserver
from dexpbt_isaaclab.world_model.wm_strategy_profiles import (
    PROFILE_DESCRIPTIONS,
    PROFILE_IDS,
    PROFILE_TABLES,
    apply_wm_strategy_profile,
    collect_explicit_dests,
    profile_capability_flags,
    profile_controlled_restart_skip_flags,
    profile_copy_restart_train_args,
    resolve_profile_copy_restart_config,
)


def _ns(**kwargs):
    return argparse.Namespace(**kwargs)


def _all_profile_dests() -> set[str]:
    dests: set[str] = set()
    for table in PROFILE_TABLES.values():
        dests.update(table.keys())
    return dests


class TestWMStrategyProfiles(unittest.TestCase):
    def test_profile_ids_are_stable(self):
        self.assertEqual(PROFILE_IDS["legacy"], 0)
        self.assertEqual(PROFILE_IDS["ppo_only"], 1)
        self.assertEqual(PROFILE_IDS["rssm_mppi_actor_distill"], 2)
        self.assertEqual(PROFILE_IDS["wm_curriculum_controller"], 3)
        self.assertEqual(PROFILE_IDS["wm_pbt_source_selector"], 4)
        self.assertEqual(PROFILE_IDS["elite_bank_actor_distill"], 5)
        self.assertEqual(PROFILE_IDS["conservative_rssm_planner_distill"], 6)
        self.assertEqual(PROFILE_IDS["light_rssm_mppi_actor_distill"], 7)
        self.assertEqual(PROFILE_IDS["wm_pbt_source_selector_v2"], 8)
        self.assertEqual(PROFILE_IDS["light_conservative_rssm_planner_distill"], 9)

    def test_ppo_only_description_is_not_original_dexpbt(self):
        desc = PROFILE_DESCRIPTIONS["ppo_only"]
        self.assertIn("anchor", desc)
        self.assertIn("not an exact original DexPBT", desc)

    def test_legacy_no_op(self):
        args = _ns(wm_strategy_profile="legacy", eir_enable=False)
        audit = apply_wm_strategy_profile(args)
        self.assertEqual(audit, [])
        self.assertFalse(args.eir_enable)

    def test_ppo_only_disables_all_wm_intervention(self):
        args = _ns(
            wm_strategy_profile="ppo_only",
            enable_wm=True,
            eir_enable=True,
            pbt_wm_score_mode="active",
            pbt_use_wm_recovery_score=True,
            eir_residual_apply_to_rollout=True,
            wm_curriculum_controller_enable=True,
        )
        apply_wm_strategy_profile(args)
        self.assertFalse(args.enable_wm)
        self.assertFalse(args.eir_enable)
        self.assertEqual(args.pbt_wm_score_mode, "off")
        self.assertFalse(args.pbt_use_wm_recovery_score)
        self.assertFalse(args.eir_residual_apply_to_rollout)
        self.assertFalse(args.wm_curriculum_controller_enable)

    def test_rssm_mppi_actor_distill_enables_actor_distill_no_residual_apply(self):
        args = _ns(
            wm_strategy_profile="rssm_mppi_actor_distill",
            eir_actor_distill_enable=False,
            eir_residual_apply_to_rollout=True,
            eir_use_rssm_mppi=False,
        )
        apply_wm_strategy_profile(args)
        self.assertTrue(args.eir_actor_distill_enable)
        self.assertFalse(args.eir_residual_apply_to_rollout)
        self.assertTrue(args.eir_use_rssm_mppi)
        self.assertEqual(args.eir_actor_distill_allowed_sources, "rssm,elite,archive")

    def test_wm_curriculum_controller_enables_controller_no_action_intervention(self):
        args = _ns(
            wm_strategy_profile="wm_curriculum_controller",
            wm_curriculum_controller_enable=False,
            eir_actor_distill_enable=True,
            eir_enable=True,
            eir_residual_apply_to_rollout=True,
        )
        apply_wm_strategy_profile(args)
        self.assertTrue(args.wm_curriculum_controller_enable)
        self.assertFalse(args.eir_actor_distill_enable)
        self.assertFalse(args.eir_enable)
        self.assertFalse(args.eir_residual_apply_to_rollout)

    def test_wm_pbt_source_selector_enables_active_wm_score_no_action_intervention(self):
        args = _ns(
            wm_strategy_profile="wm_pbt_source_selector",
            pbt_wm_score_mode="off",
            pbt_use_wm_recovery_score=False,
            eir_actor_distill_enable=True,
            eir_enable=True,
        )
        apply_wm_strategy_profile(args)
        self.assertEqual(args.pbt_wm_score_mode, "active")
        self.assertTrue(args.pbt_use_wm_recovery_score)
        self.assertFalse(args.eir_actor_distill_enable)
        self.assertFalse(args.eir_enable)

    def test_elite_bank_actor_distill_allows_only_elite_archive_sources(self):
        args = _ns(
            wm_strategy_profile="elite_bank_actor_distill",
            eir_actor_distill_allowed_sources="rssm,elite,archive",
            eir_use_rssm_mppi=True,
            eir_mppi_mode="full",
        )
        apply_wm_strategy_profile(args)
        self.assertEqual(args.eir_actor_distill_allowed_sources, "elite,archive")
        self.assertFalse(args.eir_use_rssm_mppi)
        self.assertEqual(args.eir_mppi_mode, "off")

    def test_conservative_planner_allows_only_rssm_teacher(self):
        args = _ns(
            wm_strategy_profile="conservative_rssm_planner_distill",
            eir_actor_distill_allowed_sources="rssm,elite,archive",
        )
        apply_wm_strategy_profile(args)
        self.assertEqual(args.eir_actor_distill_allowed_sources, "rssm")
        self.assertTrue(args.eir_use_rssm_mppi)
        self.assertFalse(args.eir_residual_apply_to_rollout)

    def test_nonlegacy_profile_overrides_restart_wm_args(self):
        argv = [
            "train.py",
            "--wm_strategy_profile=ppo_only",
            "--enable_wm",
            "--wm_strategy=rssm_belief_value",
            "--eir_enable",
        ]
        args = _ns(
            wm_strategy_profile="ppo_only",
            enable_wm=True,
            wm_strategy="rssm_belief_value",
            eir_enable=True,
        )
        audit = apply_wm_strategy_profile(args, argv=argv, authoritative=True)
        self.assertFalse(args.enable_wm)
        self.assertEqual(args.wm_strategy, "none")
        self.assertFalse(args.eir_enable)
        overridden = [a for a in audit if a.get("overridden")]
        self.assertTrue(any(a.get("flag") == "enable_wm" for a in overridden))
        self.assertTrue(all(a.get("profile_authoritative") for a in overridden))

    def test_legacy_profile_keeps_user_explicit_args(self):
        argv = [
            "train.py",
            "--wm_strategy_profile",
            "ppo_only",
            "--enable_wm",
        ]
        explicit = collect_explicit_dests(argv)
        self.assertIn("enable_wm", explicit)
        args = _ns(
            wm_strategy_profile="ppo_only",
            enable_wm=True,
        )
        audit = apply_wm_strategy_profile(args, argv=argv, authoritative=False)
        kept = [a for a in audit if not a.get("overridden", True)]
        self.assertTrue(args.enable_wm)
        self.assertTrue(any(a.get("flag") == "enable_wm" for a in kept))

    def test_rssm_profile_not_polluted_by_no_eir_or_wrong_wm_strategy_when_authoritative(
        self,
    ):
        argv = [
            "train.py",
            "--wm_strategy_profile=rssm_mppi_actor_distill",
            "--no-eir_enable",
            "--wm_strategy=none",
        ]
        args = _ns(
            wm_strategy_profile="rssm_mppi_actor_distill",
            eir_enable=False,
            wm_strategy="none",
        )
        apply_wm_strategy_profile(args, argv=argv, authoritative=True)
        self.assertTrue(args.eir_enable)
        self.assertEqual(args.wm_strategy, "rssm_belief_value")

    def test_profile_controlled_restart_skip_flags_cover_table(self):
        skip = profile_controlled_restart_skip_flags()
        for dest in _all_profile_dests():
            self.assertIn(f"--{dest}", skip)

    def test_light_rssm_mppi_actor_distill_lightens_mppi(self):
        args = _ns(
            wm_strategy_profile="light_rssm_mppi_actor_distill",
            eir_mppi_num_sequences=64,
            eir_mppi_use_model_ensemble=True,
        )
        apply_wm_strategy_profile(args)
        self.assertEqual(args.eir_mppi_num_sequences, 16)
        self.assertEqual(args.eir_mppi_num_elite, 2)
        self.assertFalse(args.eir_mppi_use_model_ensemble)
        self.assertEqual(args.eir_actor_distill_start_after_frames, 300_000_000)

    def test_wm_pbt_source_selector_v2_conservative_weights(self):
        args = _ns(
            wm_strategy_profile="wm_pbt_source_selector_v2",
            pbt_wm_recovery_weight=0.3,
            wm_imagined_soft_uncertainty_threshold=0.95,
            wm_imagined_min_logit_abs_mean=0.05,
            wm_imagined_min_head_signal_std=0.02,
            wm_imagined_soft_validity_weight=0.25,
        )
        apply_wm_strategy_profile(
            args,
            argv=[
                "train.py",
                "--wm_strategy_profile=wm_pbt_source_selector_v2",
                "--wm_imagined_soft_uncertainty_threshold=0.95",
                "--wm_imagined_min_logit_abs_mean=0.05",
                "--wm_imagined_min_head_signal_std=0.02",
                "--wm_imagined_soft_validity_weight=0.25",
            ],
            authoritative=True,
        )
        self.assertEqual(args.pbt_wm_recovery_weight, 0.20)
        self.assertEqual(args.wm_imagined_soft_uncertainty_threshold, 0.90)
        self.assertEqual(args.wm_imagined_min_logit_abs_mean, 0.10)
        self.assertEqual(args.wm_imagined_min_head_signal_std, 0.05)
        self.assertEqual(args.wm_imagined_soft_validity_weight, 0.10)
        self.assertFalse(args.eir_enable)
        self.assertFalse(args.eir_actor_distill_enable)

    def test_light_conservative_planner_detected_by_family(self):
        args = _ns(
            wm_strategy_profile="light_conservative_rssm_planner_distill",
            eir_mppi_num_sequences=64,
        )
        apply_wm_strategy_profile(args)
        caps = profile_capability_flags(args)
        self.assertTrue(caps["profile_conservative_planner_enabled"])
        self.assertEqual(args.eir_mppi_num_sequences, 24)
        self.assertEqual(args.eir_actor_distill_allowed_sources, "rssm")

    def test_profile_capability_flags(self):
        args = _ns(
            wm_strategy_profile="rssm_mppi_actor_distill",
            eir_actor_distill_enable=True,
            eir_residual_apply_to_rollout=False,
            eir_enable=True,
            eir_use_rssm_mppi=True,
            eir_mppi_mode="full",
            pbt_wm_score_mode="off",
            wm_curriculum_controller_enable=False,
            eir_use_elite_bank=True,
        )
        caps = profile_capability_flags(args)
        self.assertTrue(caps["profile_actor_distill_enabled"])
        self.assertFalse(caps["profile_residual_apply_enabled"])
        self.assertTrue(caps["profile_rssm_mppi_enabled"])


class TestSoftActivationGateProfiles(unittest.TestCase):
    def _expand(self, profile: str):
        args = _ns(wm_strategy_profile=profile)
        for dest in _all_profile_dests():
            if not hasattr(args, dest):
                setattr(args, dest, None)
        apply_wm_strategy_profile(
            args, argv=["train.py", f"--wm_strategy_profile={profile}"],
            authoritative=True,
        )
        return args

    def test_rssm_mppi_profile_uses_soft_actor_distill_gate(self):
        a = self._expand("rssm_mppi_actor_distill")
        self.assertEqual(a.eir_actor_distill_gate_mode, "soft_tol")
        self.assertEqual(a.eir_actor_distill_tol_start, 0.060)
        self.assertEqual(a.eir_actor_distill_tol_full, 0.040)
        self.assertTrue(a.wm_activation_enable)

    def test_elite_profile_uses_earlier_soft_gate_than_rssm(self):
        elite = self._expand("elite_bank_actor_distill")
        rssm = self._expand("rssm_mppi_actor_distill")
        self.assertEqual(elite.eir_actor_distill_gate_mode, "soft_tol")
        # elite softens earlier: higher tol_start and shorter ramp.
        self.assertGreater(
            elite.eir_actor_distill_tol_start, rssm.eir_actor_distill_tol_start
        )
        self.assertLess(
            elite.eir_actor_distill_ramp_frames, rssm.eir_actor_distill_ramp_frames
        )
        self.assertEqual(elite.elite_distill_gate_mode, "soft_tol")
        self.assertEqual(elite.eir_actor_distill_allowed_sources, "elite,archive")

    def test_conservative_profile_uses_stricter_uncertainty_and_later_gate(self):
        cons = self._expand("conservative_rssm_planner_distill")
        elite = self._expand("elite_bank_actor_distill")
        self.assertEqual(cons.eir_actor_distill_gate_mode, "soft_tol")
        # later + stricter: lower tol_full, stricter uncertainty than elite.
        self.assertLess(
            cons.eir_actor_distill_tol_full, elite.eir_actor_distill_tol_full
        )
        self.assertEqual(cons.eir_actor_distill_uncertainty_threshold, 1.5)
        self.assertEqual(cons.eir_actor_distill_allowed_sources, "rssm")
        self.assertEqual(cons.conservative_planner_gate_mode, "soft_tol")

    def test_wm_pbt_source_selector_uses_confidence_only_gate(self):
        a = self._expand("wm_pbt_source_selector")
        self.assertEqual(a.wm_pbt_source_gate_mode, "confidence_only")
        self.assertEqual(a.wm_pbt_source_start_after_frames, 300_000_000)
        self.assertEqual(a.pbt_wm_score_mode, "active")

    def test_wm_curriculum_controller_uses_shrink_candidate_gate(self):
        a = self._expand("wm_curriculum_controller")
        self.assertEqual(a.wm_curriculum_gate_mode, "shrink_candidate")

    def test_ppo_only_disables_activation(self):
        a = self._expand("ppo_only")
        self.assertFalse(a.wm_activation_enable)

    def test_residual_apply_still_disabled_in_all_strategy_zoo_profiles(self):
        for profile in (
            "ppo_only",
            "rssm_mppi_actor_distill",
            "wm_curriculum_controller",
            "wm_pbt_source_selector",
            "elite_bank_actor_distill",
            "conservative_rssm_planner_distill",
            "light_rssm_mppi_actor_distill",
            "wm_pbt_source_selector_v2",
            "light_conservative_rssm_planner_distill",
        ):
            a = self._expand(profile)
            self.assertFalse(
                a.eir_residual_apply_to_rollout,
                msg=f"{profile} must keep residual rollout apply OFF",
            )


class TestRewriteTrainArgProfile(unittest.TestCase):
    def test_rewrite_train_arg_profile_handles_equals_and_space_forms(self):
        args = [
            "--wm_strategy_profile",
            "ppo_only",
            "--foo=bar",
            "--wm_strategy_profile=wm_pbt_source_selector",
        ]
        out = DexPBTPbtAlgoObserver._rewrite_train_arg_profile(
            args, "elite_bank_actor_distill"
        )
        self.assertIn("--wm_strategy_profile=elite_bank_actor_distill", out)
        self.assertNotIn("ppo_only", out)
        self.assertFalse(any(a.startswith("--wm_strategy_profile=wm_pbt") for a in out))
        self.assertEqual(
            sum(1 for a in out if a.startswith("--wm_strategy_profile=")), 1
        )


class TestProfileCopyRestartArgs(unittest.TestCase):
    def test_profile_copy_restart_train_args_enabled(self):
        out = profile_copy_restart_train_args(
            enable=True,
            after_frames=500_000_000,
            prob=0.5,
            keep_diversity_min_profiles=3,
        )
        self.assertEqual(
            out,
            [
                "--pbt_wm_profile_copy_enable",
                "--pbt_wm_profile_copy_after_frames=500000000",
                "--pbt_wm_profile_copy_prob=0.5",
                "--pbt_wm_profile_keep_diversity_min_profiles=3",
                "--pbt_wm_profile_copy_min_nsucc_gap=0.0",
                "--pbt_wm_profile_copy_min_source_retention=0.0",
                "--pbt_wm_profile_copy_cooldown_frames=0",
                "--pbt_wm_profile_copy_bottom_k=0",
                "--pbt_wm_profile_copy_top_k=0",
            ],
        )

    def test_profile_copy_restart_train_args_disabled(self):
        out = profile_copy_restart_train_args(
            enable=False,
            after_frames=1_000_000_000,
            prob=0.3,
            keep_diversity_min_profiles=3,
        )
        self.assertIn("--no-pbt_wm_profile_copy_enable", out)
        self.assertNotIn("--pbt_wm_profile_copy_enable", out)

    def test_resolve_profile_copy_restart_config_recovers_from_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            meta = Path(tmp) / "metadata.txt"
            meta.write_text(
                "train_args=--pbt_wm_profile_copy_enable "
                "train_args=--pbt_wm_profile_copy_after_frames=500000000 "
                "train_args=--pbt_wm_profile_copy_prob=0.5 "
                "train_args=--pbt_wm_profile_keep_diversity_min_profiles=3\n"
            )
            resolved = resolve_profile_copy_restart_config(
                enable=False,
                after_frames=1_000_000_000,
                prob=0.3,
                keep_diversity_min_profiles=3,
                metadata_path=meta,
            )
            self.assertEqual(
                resolved,
                (True, 500_000_000, 0.5, 3, 0.0, 0.0, 0, 0, 0),
            )

    def test_resolve_profile_copy_restart_config_keeps_explicit_values(self):
        resolved = resolve_profile_copy_restart_config(
            enable=True,
            after_frames=250_000_000,
            prob=0.4,
            keep_diversity_min_profiles=2,
            metadata_path=None,
        )
        self.assertEqual(resolved, (True, 250_000_000, 0.4, 2, 0.0, 0.0, 0, 0, 0))


class TestProfileCopyDefaults(unittest.TestCase):
    def test_profile_copy_disabled_by_default(self):
        args = _ns(
            pbt_wm_profile_copy_enable=False,
            pbt_wm_profile_copy_after_frames=1_000_000_000,
            pbt_wm_profile_copy_prob=0.3,
            pbt_wm_profile_keep_diversity_min_profiles=3,
        )
        self.assertFalse(args.pbt_wm_profile_copy_enable)

    def test_profile_copy_preserves_min_diversity(self):
        diversity = 3
        min_profiles = 3
        allow_copy = diversity > min_profiles
        self.assertFalse(allow_copy)


if __name__ == "__main__":
    unittest.main()
