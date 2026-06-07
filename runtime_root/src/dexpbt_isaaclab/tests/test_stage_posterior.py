"""
Tests for Phase A4 stage-aware posterior:

* stage_classifier.infer_stage covers recovery / early / acquisition / mastery
  with documented priorities and trigger sets.
* SharedPosteriorSelector additive schema:
  - legacy file (no __stage_posterior__) loads cleanly, stage_posterior empty
  - update_batch_stage updates BOTH global and stage entries
  - save/load round-trip preserves both
  - sample_stage falls back PER-STRATEGY when n < min_stage_n
  - none baseline mean is fixed at 0 in stage entries too
"""

from __future__ import annotations

import json
import random
import tempfile
import unittest
from pathlib import Path

from dexpbt_isaaclab.world_model.stage_classifier import (
    RECOVERY_TRIGGERS,
    StageFeatures,
    VALID_STAGES,
    infer_stage,
    stage_features_payload,
)
from dexpbt_isaaclab.world_model.wm_posterior_selector import (
    BayesianPosteriorSelector,
    PosteriorState,
    SharedPosteriorSelector,
    STAGES,
)


class StageClassifierTest(unittest.TestCase):
    def test_recovery_priority_over_early(self):
        # Even with low frames, recent restart of trigger type -> recovery.
        f = StageFeatures(
            env_frames=10_000_000,
            nsucc_mean=0.0,
            frames_since_restart=50_000_000,
            last_restart_type="cross_copy",
        )
        self.assertEqual(infer_stage(f), "recovery")

    def test_recovery_skipped_when_old_restart(self):
        # frames_since_restart >= 100M -> recovery NOT triggered.
        f = StageFeatures(
            env_frames=10_000_000,
            nsucc_mean=0.0,
            frames_since_restart=200_000_000,
            last_restart_type="cross_copy",
        )
        self.assertEqual(infer_stage(f), "early")

    def test_recovery_skipped_when_unknown_trigger(self):
        f = StageFeatures(
            env_frames=10_000_000,
            nsucc_mean=0.0,
            frames_since_restart=10,
            last_restart_type="some_random_label",
        )
        self.assertEqual(infer_stage(f), "early")

    def test_early_by_low_frames(self):
        f = StageFeatures(env_frames=500_000_000, nsucc_mean=10.0)
        self.assertEqual(infer_stage(f), "early")

    def test_early_by_low_nsucc(self):
        f = StageFeatures(env_frames=2_000_000_000, nsucc_mean=2.0)
        self.assertEqual(infer_stage(f), "early")

    def test_acquisition_band(self):
        f = StageFeatures(env_frames=2_000_000_000, nsucc_mean=10.0)
        self.assertEqual(infer_stage(f), "acquisition")

    def test_mastery(self):
        f = StageFeatures(env_frames=3_000_000_000, nsucc_mean=20.0)
        self.assertEqual(infer_stage(f), "mastery")

    def test_recovery_triggers_set_complete(self):
        for t in ("cross_copy", "recovery", "oom_relaunch",
                  "manual_restart", "startup_retry"):
            self.assertIn(t, RECOVERY_TRIGGERS)
        self.assertNotIn("none", RECOVERY_TRIGGERS)

    def test_payload_shape(self):
        p = stage_features_payload(
            env_frames=1, nsucc_mean=2.0, nsucc_max=3.0,
            success_tolerance=0.05, frames_since_restart=100,
            last_restart_type="cross_copy", policy_idx=2,
        )
        self.assertEqual(set(p.keys()), {
            "env_frames", "nsucc_mean", "nsucc_max", "success_tolerance",
            "frames_since_restart", "last_restart_type", "policy_idx",
        })

    def test_valid_stages_match_posterior(self):
        # Stage labels must match posterior.STAGES
        self.assertEqual(set(VALID_STAGES), set(STAGES))


class StagePosteriorAdditiveSchemaTest(unittest.TestCase):
    """Legacy schema (no __stage_posterior__) must still load."""

    def test_legacy_schema_loads_clean(self):
        legacy = {
            "none":     {"n": 1, "mean": 0.0, "M2": 1.0, "last_update_step": 0},
            "aux_mid":  {"n": 5, "mean": 0.2, "M2": 1.5, "last_update_step": 100},
            "__pool__": ["none", "aux_mid"],
        }
        sel = BayesianPosteriorSelector(strategies=["none", "aux_mid"])
        sel.load_state(legacy)
        self.assertEqual(sel.posterior["aux_mid"].n, 5)
        # stage_posterior must remain empty (per-stage dicts initialized empty).
        for stage in STAGES:
            self.assertEqual(sel.stage_posterior.get(stage, {}), {})


class StageUpdateAndSampleTest(unittest.TestCase):
    def setUp(self):
        random.seed(0)
        self.sel = BayesianPosteriorSelector(
            strategies=["none", "aux_mid", "rssm_belief_value", "trans_belief_value"]
        )

    def test_update_batch_stage_updates_both(self):
        self.sel.update_batch_stage(
            delta_w={"none": 0.0, "aux_mid": 0.5, "rssm_belief_value": 1.0},
            stage="early", step=100,
        )
        # Global state moved.
        self.assertGreater(self.sel.posterior["aux_mid"].n, 1)
        # Stage state created and matches.
        self.assertIn("aux_mid", self.sel.stage_posterior["early"])
        self.assertGreater(
            self.sel.stage_posterior["early"]["aux_mid"].n, 1
        )

    def test_none_mean_fixed_in_stage(self):
        for _ in range(5):
            self.sel.update_batch_stage(
                delta_w={"none": 0.7, "aux_mid": 0.3},  # delta_w["none"] is "noise"
                stage="acquisition", step=10,
            )
        self.assertEqual(
            float(self.sel.stage_posterior["acquisition"]["none"].mean), 0.0
        )
        # global "none" mean also pinned at 0.
        self.assertEqual(float(self.sel.posterior["none"].mean), 0.0)

    def test_sample_stage_per_strategy_fallback(self):
        # Build stage posterior where rssm_belief_value has many samples,
        # but trans_belief_value has zero stage samples.
        for _ in range(10):
            self.sel.update_batch_stage(
                delta_w={"none": 0.0, "aux_mid": -0.3, "rssm_belief_value": 1.0},
                stage="acquisition", step=42,
            )
        # And the GLOBAL trans_belief_value has been updated separately.
        self.sel.update_batch(
            delta_w={"trans_belief_value": 5.0}, step=42,
        )
        # min_stage_n=3 means rssm/aux_mid use stage; trans falls back to global.
        # Sample many times; trans has very high global mean, so it should
        # win frequently if global fallback is in play.
        import collections
        winners = collections.Counter()
        for _ in range(200):
            winners[self.sel.sample_stage("acquisition", min_stage_n=3)] += 1
        # trans_belief_value must appear in winners (proves fallback wired).
        self.assertGreater(winners["trans_belief_value"], 0)
        # Fallback counter must reflect at least one fallback for trans.
        self.assertGreater(
            self.sel.fallback_counts.get(("acquisition", "trans_belief_value"), 0),
            0,
        )

    def test_sample_stage_global_context_short_circuits(self):
        # context='global' must always use the global path -> no fallback inc.
        self.sel.update_batch(
            delta_w={"none": 0.0, "aux_mid": 0.1, "rssm_belief_value": 0.2,
                     "trans_belief_value": 0.3},
            step=1,
        )
        before = dict(self.sel.fallback_counts)
        for _ in range(20):
            self.sel.sample_stage("acquisition", context="global")
        self.assertEqual(self.sel.fallback_counts, before)


class StageRoundtripTest(unittest.TestCase):
    """save() then load_or_init() must restore both global and stage state."""

    def test_shared_save_load_with_stage(self):
        with tempfile.TemporaryDirectory() as td:
            shared_path = Path(td) / "wm_posterior_latest.json"
            writer = SharedPosteriorSelector(
                shared_path=shared_path,
                strategies=["none", "aux_mid", "rssm_belief_value"],
            )
            writer.load_or_init(policy_idx=0)
            writer.update_and_save_stage(
                delta_w={"none": 0.0, "aux_mid": 0.4, "rssm_belief_value": 1.5},
                stage="early", step=200,
            )
            writer.update_and_save_stage(
                delta_w={"none": 0.0, "aux_mid": -0.2, "rssm_belief_value": 0.5},
                stage="early", step=300,
            )
            writer.update_and_save_stage(
                delta_w={"none": 0.0, "aux_mid": 0.1, "rssm_belief_value": -0.5},
                stage="acquisition", step=400,
            )

            # Inspect on-disk schema.
            data = json.loads(shared_path.read_text())
            self.assertIn("__stage_posterior__", data)
            self.assertIn("early", data["__stage_posterior__"])
            self.assertIn("acquisition", data["__stage_posterior__"])
            self.assertIn("rssm_belief_value", data["__stage_posterior__"]["early"])

            # Reader path.
            reader = SharedPosteriorSelector(
                shared_path=shared_path,
                strategies=["none", "aux_mid", "rssm_belief_value"],
            )
            reader.load_or_init(policy_idx=1)
            self.assertEqual(
                reader.stage_posterior["early"]["rssm_belief_value"].n,
                writer.stage_posterior["early"]["rssm_belief_value"].n,
            )
            self.assertAlmostEqual(
                reader.stage_posterior["early"]["rssm_belief_value"].mean,
                writer.stage_posterior["early"]["rssm_belief_value"].mean,
                places=6,
            )
            # none mean stays 0.
            self.assertEqual(
                float(reader.stage_posterior["early"]["none"].mean), 0.0
            )

    def test_save_omits_stage_when_empty(self):
        with tempfile.TemporaryDirectory() as td:
            shared_path = Path(td) / "wm_posterior_latest.json"
            writer = SharedPosteriorSelector(
                shared_path=shared_path,
                strategies=["none", "aux_mid"],
            )
            writer.load_or_init(policy_idx=0)
            # Only global update, no stage update.
            writer.update_and_save(
                delta_w={"none": 0.0, "aux_mid": 0.5}, step=100
            )
            data = json.loads(shared_path.read_text())
            self.assertNotIn("__stage_posterior__", data,
                             "stage block should not appear when empty")


if __name__ == "__main__":
    unittest.main()
