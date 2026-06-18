"""Unit tests for Strategy Zoo v1.2 profile-copy logic."""

from __future__ import annotations

import json
import random
import tempfile
import unittest
from pathlib import Path

from dexpbt_isaaclab.pbt.pbt_observer import DexPBTPbtAlgoObserver
from dexpbt_isaaclab.pbt.pbt_profile_copy import (
    ProfileCopyConfig,
    compute_weighted_copy_prob,
    decide_profile_copy,
)
from dexpbt_isaaclab.world_model.wm_strategy_profiles import (
    profile_copy_restart_train_args,
    profile_copy_source_bonus,
    profile_controlled_restart_skip_flags,
    resolve_profile_copy_restart_config,
)

REPO_ROOT = Path(__file__).resolve().parents[4]
V12_LAUNCHER = (
    REPO_ROOT
    / "results/launchers/start_regrasping_pbt_wm_strategy_zoo_v12_2b_debug.sh"
)


def _population(
    *,
    target_nsucc: float = 10.0,
    source_nsucc: float = 20.0,
    source_retention: float = 0.9,
) -> dict[int, dict]:
    return {
        0: {
            "pbt_ranking_score": target_nsucc,
            "nsucc_mean": target_nsucc,
            "wm_strategy_profile_name": "ppo_only",
            "wm_strategy_profile_generation": 0,
        },
        1: {
            "pbt_ranking_score": source_nsucc,
            "nsucc_mean": source_nsucc,
            "wm_strategy_profile_name": "elite_bank_actor_distill",
            "wm_strategy_profile_generation": 2,
            "source_retention_at_save": source_retention,
        },
    }


class TestProfileCopyEligibility(unittest.TestCase):
    def test_nsucc_gap_rejects_weak_source(self):
        cfg = ProfileCopyConfig(
            enable=True,
            after_frames=100,
            base_prob=0.7,
            keep_diversity_min_profiles=1,
            min_nsucc_gap=3.0,
            bottom_k=2,
            top_k=2,
        )
        out = decide_profile_copy(
            cfg=cfg,
            env_frames=500,
            target_policy_idx=0,
            source_policy_idx=1,
            target_profile_name="ppo_only",
            source_profile_name="elite_bank_actor_distill",
            source_generation=2,
            diversity_count=4,
            profiles_set={"ppo_only", "elite_bank_actor_distill", "wm_pbt_source_selector"},
            population=_population(target_nsucc=15.0, source_nsucc=16.0),
            source_meta={"has_source_retention": True, "source_retention": 0.9},
            target_meta={"has_source_retention": True, "source_retention": 0.6},
            window_nsucc={0: 15.0, 1: 16.0},
            source_checkpoint_type="live",
        )
        self.assertEqual(out.decision, "nsucc_gap")

    def test_retention_low_rejects(self):
        cfg = ProfileCopyConfig(
            enable=True,
            after_frames=100,
            min_source_retention=0.8,
            bottom_k=2,
            top_k=2,
        )
        out = decide_profile_copy(
            cfg=cfg,
            env_frames=500,
            target_policy_idx=0,
            source_policy_idx=1,
            target_profile_name="ppo_only",
            source_profile_name="elite_bank_actor_distill",
            source_generation=2,
            diversity_count=4,
            profiles_set={"ppo_only", "elite_bank_actor_distill"},
            population=_population(),
            source_meta={"has_source_retention": True, "source_retention": 0.5},
            target_meta={},
            source_checkpoint_type="live",
        )
        self.assertEqual(out.decision, "retention_low")

    def test_cooldown_rejects(self):
        cfg = ProfileCopyConfig(
            enable=True,
            after_frames=100,
            cooldown_frames=300,
            bottom_k=2,
            top_k=2,
        )
        out = decide_profile_copy(
            cfg=cfg,
            env_frames=350,
            target_policy_idx=0,
            source_policy_idx=1,
            target_profile_name="ppo_only",
            source_profile_name="elite_bank_actor_distill",
            source_generation=2,
            diversity_count=4,
            profiles_set={"ppo_only", "elite_bank_actor_distill"},
            population=_population(),
            source_meta={"has_source_retention": True, "source_retention": 0.9},
            target_meta={},
            source_checkpoint_type="live",
            last_profile_copy_frame=100,
        )
        self.assertEqual(out.decision, "cooldown")

    def test_bottom_k_and_top_k_gating(self):
        cfg = ProfileCopyConfig(
            enable=True,
            after_frames=100,
            bottom_k=1,
            top_k=1,
        )
        pop = {
            0: {"pbt_ranking_score": 30.0, "wm_strategy_profile_name": "ppo_only"},
            1: {"pbt_ranking_score": 20.0, "wm_strategy_profile_name": "elite_bank_actor_distill"},
            2: {"pbt_ranking_score": 10.0, "wm_strategy_profile_name": "wm_pbt_source_selector"},
        }
        out = decide_profile_copy(
            cfg=cfg,
            env_frames=500,
            target_policy_idx=0,
            source_policy_idx=1,
            target_profile_name="ppo_only",
            source_profile_name="elite_bank_actor_distill",
            source_generation=2,
            diversity_count=3,
            profiles_set={"ppo_only", "elite_bank_actor_distill", "wm_pbt_source_selector"},
            population=pop,
            source_meta={"has_source_retention": True, "source_retention": 0.9},
            target_meta={},
            source_checkpoint_type="live",
        )
        self.assertEqual(out.decision, "target_not_bottom")

    def test_diversity_guard_rejects_when_after_below_keep_min(self):
        cfg = ProfileCopyConfig(
            enable=True,
            after_frames=100,
            keep_diversity_min_profiles=3,
            bottom_k=2,
            top_k=2,
        )
        out = decide_profile_copy(
            cfg=cfg,
            env_frames=500,
            target_policy_idx=0,
            source_policy_idx=1,
            target_profile_name="ppo_only",
            source_profile_name="elite_bank_actor_distill",
            source_generation=2,
            diversity_count=2,
            profiles_set={"ppo_only", "elite_bank_actor_distill"},
            population=_population(),
            source_meta={"has_source_retention": True, "source_retention": 0.9},
            target_meta={},
            source_checkpoint_type="live",
        )
        self.assertEqual(out.decision, "diversity_min")
        self.assertEqual(out.diversity_after_copy, 1)

    def test_diversity_after_equals_keep_min_allows_eligibility(self):
        cfg = ProfileCopyConfig(
            enable=True,
            after_frames=100,
            keep_diversity_min_profiles=2,
            bottom_k=2,
            top_k=2,
            base_prob=0.0,
        )
        out = decide_profile_copy(
            cfg=cfg,
            env_frames=500,
            target_policy_idx=0,
            source_policy_idx=1,
            target_profile_name="ppo_only",
            source_profile_name="elite_bank_actor_distill",
            source_generation=2,
            diversity_count=3,
            profiles_set={
                "ppo_only",
                "elite_bank_actor_distill",
                "wm_pbt_source_selector",
            },
            population=_population(),
            source_meta={"has_source_retention": True, "source_retention": 0.9},
            target_meta={},
            source_checkpoint_type="live",
        )
        self.assertEqual(out.diversity_after_copy, 2)
        self.assertNotEqual(out.decision, "diversity_min")
        self.assertTrue(out.eligible)

    def test_retention_low_when_source_has_no_retention_metadata(self):
        cfg = ProfileCopyConfig(
            enable=True,
            after_frames=100,
            min_source_retention=0.8,
            keep_diversity_min_profiles=1,
            bottom_k=2,
            top_k=2,
        )
        out = decide_profile_copy(
            cfg=cfg,
            env_frames=500,
            target_policy_idx=0,
            source_policy_idx=1,
            target_profile_name="ppo_only",
            source_profile_name="elite_bank_actor_distill",
            source_generation=2,
            diversity_count=3,
            profiles_set={
                "ppo_only",
                "elite_bank_actor_distill",
                "wm_pbt_source_selector",
            },
            population=_population(),
            source_meta={},
            target_meta={},
            source_checkpoint_type="live",
        )
        self.assertEqual(out.decision, "retention_low")
        self.assertEqual(out.reject_reason, "retention_low")

    def test_same_profile_copy_not_accepted(self):
        cfg = ProfileCopyConfig(
            enable=True,
            after_frames=100,
            keep_diversity_min_profiles=1,
            bottom_k=2,
            top_k=2,
            base_prob=1.0,
        )
        out = decide_profile_copy(
            cfg=cfg,
            env_frames=500,
            target_policy_idx=0,
            source_policy_idx=1,
            target_profile_name="elite_bank_actor_distill",
            source_profile_name="elite_bank_actor_distill",
            source_generation=2,
            diversity_count=3,
            profiles_set={
                "ppo_only",
                "elite_bank_actor_distill",
                "wm_pbt_source_selector",
            },
            population=_population(),
            source_meta={"has_source_retention": True, "source_retention": 0.9},
            target_meta={},
            source_checkpoint_type="live",
        )
        self.assertEqual(out.decision, "same_profile")
        self.assertEqual(out.reject_reason, "same_profile")
        self.assertFalse(out.accepted)
        self.assertFalse(out.attempted)
        self.assertEqual(out.new_profile_name, "elite_bank_actor_distill")

    def test_source_bonus_and_prob_clip(self):
        p = compute_weighted_copy_prob(
            base_prob=0.7,
            source_profile="elite_bank_actor_distill",
            target_need_factor=1.25,
            prob_clip_max=0.9,
        )
        self.assertAlmostEqual(p, 0.9)
        self.assertAlmostEqual(profile_copy_source_bonus("ppo_only"), 0.6)
        self.assertAlmostEqual(profile_copy_source_bonus("wm_pbt_source_selector"), 1.1)
        self.assertAlmostEqual(profile_copy_source_bonus("elite_bank_actor_distill"), 1.3)
        self.assertGreater(
            profile_copy_source_bonus("elite_bank_actor_distill"),
            profile_copy_source_bonus("light_rssm_mppi_actor_distill"),
        )


class TestProfileCopyCooldownResume(unittest.TestCase):
    def test_profile_copy_cooldown_frame_resume(self):
        obs = object.__new__(DexPBTPbtAlgoObserver)
        DexPBTPbtAlgoObserver.__init__(
            obs,
            policy_idx=0,
            num_policies=6,
            mutable_params={},
            workspace="/tmp/ws",
            train_dir="/tmp/ws/rl_games",
            task_name="test",
            interval_steps=1,
            start_after=0,
            initial_delay=0,
            last_profile_copy_frame_resume=123_456_789,
        )
        self.assertEqual(obs._last_profile_copy_frame, 123_456_789)

    def test_restart_train_args_include_last_profile_copy_frame_resume(self):
        skip_flags = {
            "--last_profile_copy_frame_resume",
        }
        stale_argv = [
            "--gamma=0.99",
            "--last_profile_copy_frame_resume=1",
        ]
        filtered = [
            a
            for a in stale_argv
            if a.split("=")[0] not in skip_flags
        ]
        last_frame = 456_789_012
        restored = filtered + [
            f"--last_profile_copy_frame_resume={int(last_frame)}"
        ]
        self.assertIn(
            "--last_profile_copy_frame_resume=456789012",
            restored,
        )
        self.assertNotIn("--last_profile_copy_frame_resume=1", restored)


class TestProfileCopyRestartSurvival(unittest.TestCase):
    def test_profile_copy_flags_survive_restart_rewrite(self):
        skip = profile_controlled_restart_skip_flags()
        for flag in (
            "--pbt_wm_profile_copy_min_nsucc_gap",
            "--pbt_wm_profile_copy_min_source_retention",
            "--pbt_wm_profile_copy_cooldown_frames",
            "--pbt_wm_profile_copy_bottom_k",
            "--pbt_wm_profile_copy_top_k",
        ):
            self.assertIn(flag, skip)
        reinject = profile_copy_restart_train_args(
            enable=True,
            after_frames=300_000_000,
            prob=0.7,
            keep_diversity_min_profiles=3,
            min_nsucc_gap=3.0,
            min_source_retention=0.8,
            cooldown_frames=300_000_000,
            bottom_k=2,
            top_k=2,
        )
        argv = [
            "--gamma=0.99",
            "--pbt_wm_profile_copy_enable",
            "--pbt_wm_profile_copy_after_frames=300000000",
        ]
        skip_set = set(profile_controlled_restart_skip_flags())
        filtered = [
            a
            for a in argv
            if a.split("=")[0] not in skip_set
        ]
        restored = filtered + reinject
        self.assertIn("--pbt_wm_profile_copy_min_nsucc_gap=3.0", restored)
        self.assertIn("--pbt_wm_profile_copy_bottom_k=2", restored)

    def test_resolve_profile_copy_restart_config_v12_flags(self):
        with tempfile.TemporaryDirectory() as tmp:
            meta = Path(tmp) / "metadata.txt"
            meta.write_text(
                "train_args=--pbt_wm_profile_copy_enable "
                "train_args=--pbt_wm_profile_copy_after_frames=300000000 "
                "train_args=--pbt_wm_profile_copy_prob=0.7 "
                "train_args=--pbt_wm_profile_keep_diversity_min_profiles=3 "
                "train_args=--pbt_wm_profile_copy_min_nsucc_gap=3.0 "
                "train_args=--pbt_wm_profile_copy_min_source_retention=0.8 "
                "train_args=--pbt_wm_profile_copy_cooldown_frames=300000000 "
                "train_args=--pbt_wm_profile_copy_bottom_k=2 "
                "train_args=--pbt_wm_profile_copy_top_k=2\n"
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
                (True, 300_000_000, 0.7, 3, 3.0, 0.8, 300_000_000, 2, 2),
            )


class TestProfileCopyEvents(unittest.TestCase):
    def test_profile_copy_reject_reason_logged(self):
        cfg = ProfileCopyConfig(enable=True, after_frames=1_000_000)
        out = decide_profile_copy(
            cfg=cfg,
            env_frames=100,
            target_policy_idx=0,
            source_policy_idx=1,
            target_profile_name="ppo_only",
            source_profile_name="elite_bank_actor_distill",
            source_generation=2,
            diversity_count=4,
            profiles_set={"ppo_only", "elite_bank_actor_distill"},
            population=_population(),
            source_meta={},
            target_meta={},
            source_checkpoint_type="live",
        )
        self.assertEqual(out.decision, "before_after_frames")
        self.assertEqual(out.reject_reason, "before_after_frames")

    def test_profile_copy_lineage_written(self):
        with tempfile.TemporaryDirectory() as td:
            ws = Path(td)
            obs = object.__new__(DexPBTPbtAlgoObserver)
            obs.workspace = ws
            obs.policy_idx = 2
            obs.pbt_iteration = 5
            obs.wm_strategy_profile_name = "elite_bank_actor_distill"
            obs.wm_strategy_profile_id = 5
            obs._wm_strategy_profile_generation = 3
            obs._wm_strategy_profile_parent = 4
            obs._wm_strategy_profile_copied_from_source = True
            obs._update_policy_lineage(600_000_000)
            path = ws / "policy_lineage.json"
            self.assertTrue(path.is_file())
            data = json.loads(path.read_text())
            self.assertIn("002", data["policies"])
            self.assertEqual(
                data["policies"]["002"]["wm_strategy_profile_name"],
                "elite_bank_actor_distill",
            )

    def test_profile_copy_event_retention_diagnostic_fields(self):
        with tempfile.TemporaryDirectory() as td:
            ws = Path(td)
            obs = object.__new__(DexPBTPbtAlgoObserver)
            obs.workspace = ws
            obs.policy_idx = 1
            obs.pbt_iteration = 2
            source_meta = {"has_source_retention": False, "source_retention": float("nan")}
            target_meta = {"has_source_retention": True, "source_retention": 0.75}
            obs._write_profile_copy_event(
                {
                    "decision": "retention_low",
                    "reject_reason": "retention_low",
                    "source_has_retention": bool(
                        source_meta.get("has_source_retention", False)
                    ),
                    "target_has_retention": bool(
                        target_meta.get("has_source_retention", False)
                    ),
                    "source_retention": (
                        float(source_meta["source_retention"])
                        if source_meta.get("has_source_retention")
                        else None
                    ),
                    "target_retention": (
                        float(target_meta["source_retention"])
                        if target_meta.get("has_source_retention")
                        else None
                    ),
                }
            )
            path = ws / "metrics" / "profile_copy_events.jsonl"
            event = json.loads(path.read_text().strip())
            self.assertFalse(event["source_has_retention"])
            self.assertIsNone(event["source_retention"])
            self.assertTrue(event["target_has_retention"])
            self.assertAlmostEqual(event["target_retention"], 0.75)

    def test_profile_copy_rng_deterministic(self):
        cfg = ProfileCopyConfig(
            enable=True,
            after_frames=100,
            base_prob=0.5,
            keep_diversity_min_profiles=2,
            bottom_k=2,
            top_k=2,
        )
        kwargs = dict(
            cfg=cfg,
            env_frames=500,
            target_policy_idx=0,
            source_policy_idx=1,
            target_profile_name="ppo_only",
            source_profile_name="elite_bank_actor_distill",
            source_generation=2,
            diversity_count=4,
            profiles_set={"ppo_only", "elite_bank_actor_distill", "wm_pbt_source_selector"},
            population=_population(),
            source_meta={"has_source_retention": True, "source_retention": 0.9},
            target_meta={},
            source_checkpoint_type="live",
            rng=random.Random(501),
        )
        a = decide_profile_copy(**kwargs)
        b = decide_profile_copy(**kwargs)
        self.assertEqual(a.decision, b.decision)


class TestV12Launcher(unittest.TestCase):
    def test_profile_copy_enabled_in_v12_launcher(self):
        self.assertTrue(V12_LAUNCHER.is_file(), f"missing {V12_LAUNCHER}")
        text = V12_LAUNCHER.read_text()
        self.assertIn("--pbt_wm_profile_copy_enable", text)
        self.assertIn("--pbt_wm_profile_copy_after_frames 300000000", text)
        self.assertIn("--pbt_wm_profile_copy_prob 0.7", text)
        self.assertIn("--pbt_wm_profile_copy_min_nsucc_gap 3.0", text)
        self.assertIn("strategy_zoo_v1.2_performance_debug", text)


if __name__ == "__main__":
    unittest.main()
