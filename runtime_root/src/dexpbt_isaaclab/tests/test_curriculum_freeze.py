"""Patch H (2026-05-31): post-restart curriculum freeze.

Smoke run 20260530_200619 showed peaks at moderate tolerance
(success_tolerance ~0.04-0.06) followed by collapse, because the
curriculum kept tightening epsilon on every interval even for
freshly-restarted workers that had not yet stabilised. Patch H adds a
post-restart freeze window during which the curriculum keeps the
inherited tolerance untouched.

Tests pin down:

  1. `dexpbt_tolerance_curriculum` honours
     `curriculum_freeze_frames_after_restart`:
       - while frozen, tolerance is unchanged regardless of mean Nsucc
       - `last_curriculum_update` is advanced to `frames_since_restart`
       - immediately after the freeze, the regular `curriculum_interval`
         gate still controls when the next tighten can fire.
  2. `freeze=0` recovers the legacy unfrozen behaviour exactly.
  3. The pbt observer writes the freeze hook into train_args on every
     restart, including the longer recovery freeze on archive recoveries.
"""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from dexpbt_isaaclab.dexpbt_utils import dexpbt_tolerance_curriculum
from dexpbt_isaaclab.pbt.pbt_observer import DexPBTPbtAlgoObserver


def _ts(mean_nsucc: float) -> torch.Tensor:
    # Tensor whose mean matches the requested scalar; the curriculum
    # only consults .mean().
    return torch.full((4,), float(mean_nsucc))


class ToleranceCurriculumFreezeTest(unittest.TestCase):
    def test_freeze_zero_recovers_legacy_behaviour(self):
        # mean Nsucc > 3, interval already elapsed -> legacy path tightens.
        tol, lcu = dexpbt_tolerance_curriculum(
            last_curriculum_update=0,
            frames_since_restart=100_000_000,
            curriculum_interval=10_000_000,
            prev_episode_successes=_ts(5.0),
            success_tolerance=0.075,
            initial_tolerance=0.075,
            target_tolerance=0.01,
            curriculum_freeze_frames_after_restart=0,
        )
        # 0.9 * 0.075 = 0.0675
        self.assertAlmostEqual(tol, 0.0675, places=6)
        self.assertEqual(lcu, 100_000_000)

    def test_freeze_holds_tolerance(self):
        # Inside the freeze window (50M < 100M) -> tolerance unchanged.
        tol, lcu = dexpbt_tolerance_curriculum(
            last_curriculum_update=0,
            frames_since_restart=50_000_000,
            curriculum_interval=10_000_000,
            prev_episode_successes=_ts(5.0),
            success_tolerance=0.06,
            initial_tolerance=0.075,
            target_tolerance=0.01,
            curriculum_freeze_frames_after_restart=100_000_000,
        )
        self.assertEqual(tol, 0.06)
        # `last_curriculum_update` must advance to frames_since_restart so
        # the immediately-post-freeze tick is still gated by interval.
        self.assertEqual(lcu, 50_000_000)

    def test_freeze_does_not_tighten_even_when_nsucc_high(self):
        tol, _ = dexpbt_tolerance_curriculum(
            last_curriculum_update=0,
            frames_since_restart=20_000_000,
            curriculum_interval=1_000_000,
            prev_episode_successes=_ts(50.0),
            success_tolerance=0.05,
            initial_tolerance=0.075,
            target_tolerance=0.01,
            curriculum_freeze_frames_after_restart=100_000_000,
        )
        self.assertEqual(tol, 0.05)

    def test_freeze_clears_then_normal_path_resumes(self):
        # First tick post-freeze: interval-gate still applies.
        tol, lcu = dexpbt_tolerance_curriculum(
            last_curriculum_update=99_000_000,  # advanced during freeze
            frames_since_restart=100_000_001,  # 1 frame past freeze
            curriculum_interval=10_000_000,  # but interval not elapsed
            prev_episode_successes=_ts(5.0),
            success_tolerance=0.06,
            initial_tolerance=0.075,
            target_tolerance=0.01,
            curriculum_freeze_frames_after_restart=100_000_000,
        )
        # Not yet eligible to tighten (interval gate); but freeze has cleared.
        self.assertEqual(tol, 0.06)
        self.assertEqual(lcu, 99_000_000)

        # Now well past interval -> tightening fires.
        tol, lcu = dexpbt_tolerance_curriculum(
            last_curriculum_update=99_000_000,
            frames_since_restart=200_000_000,
            curriculum_interval=10_000_000,
            prev_episode_successes=_ts(5.0),
            success_tolerance=0.06,
            initial_tolerance=0.075,
            target_tolerance=0.01,
            curriculum_freeze_frames_after_restart=100_000_000,
        )
        self.assertAlmostEqual(tol, 0.054, places=6)
        self.assertEqual(lcu, 200_000_000)


class ObserverWritesFreezeHookTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.workspace = Path(self._tmp.name)

    def _make(self, **kw):
        # train_dir.parent is where the manifest lives, so use a subdir of
        # workspace so we can scope `rglob` to find it.
        run_dir = self.workspace / "run"
        run_dir.mkdir(parents=True, exist_ok=True)
        td = run_dir / "rl_games"
        td.mkdir(parents=True, exist_ok=True)
        self._run_dir = run_dir
        defaults = dict(
            policy_idx=0,
            num_policies=2,
            mutable_params={
                "enable_wm": False,
                "enable_counterfactual": False,
                "wm_strategy": "none",
            },
            workspace=str(self.workspace),
            train_dir=str(td),
            task_name="test-task",
            interval_steps=10_000_000,
            start_after=0,
            initial_delay=0,
            num_envs=1,
            pbt_min_frames_as_source_after_restart=0,
            pbt_min_frames_between_restarts=0,
            pbt_min_nsucc_mean_as_source=0.0,
            pbt_min_nsucc_max_as_source=0.0,
            pbt_curriculum_freeze_frames=120_000_000,
            pbt_curriculum_freeze_on_recovery_frames=250_000_000,
        )
        defaults.update(kw)
        obs = DexPBTPbtAlgoObserver(**defaults)
        obs.writer = None
        return obs

    def _captured_train_args(self, obs, *, checkpoint_type: str):
        """Run a stripped-down version of `_restart_process_with_new_params`
        far enough to collect the `train_args` list, without firing exec.

        We patch out the three side-effecting calls the function makes
        between train_args assembly and process kill:
          * shutil.copy2 -> would copy the source ckpt to a working path
          * `_rewrite_checkpoint` -> would torch.load a real `.pth`
          * os.kill -> would terminate the test process
        """
        import json
        import os
        import shutil as _shutil
        import signal as _signal

        ckpt = self.workspace / "fake.pth"
        ckpt.write_text("noop")
        obs._last_source_checkpoint_type = checkpoint_type
        # Stub the SIGKILL-via-raise_signal and os._exit() so the function
        # returns rather than terminating the test process.
        with patch.object(_shutil, "copy2", lambda *a, **kw: None), \
             patch.object(obs, "_rewrite_checkpoint", lambda *a, **kw: None), \
             patch.object(_signal, "raise_signal", lambda *_a: None), \
             patch.object(os, "kill", side_effect=SystemExit(0)), \
             patch.object(os, "_exit", side_effect=SystemExit(0)):
            try:
                obs._restart_process_with_new_params(
                    new_params={
                        "gamma": 0.99,
                        "kl_threshold": 0.01,
                        "grad_norm": 1.0,
                        "critic_coef": 4.0,
                        "bounds_loss_coef": 1e-4,
                        "e_clip": 0.1,
                        "mini_epochs": 2,
                        "learning_rate": 1e-4,
                        "alpha_reach": 30.0,
                        "alpha_pick": 20.0,
                        "r_picked": 300.0,
                        "alpha_targ": 200.0,
                        "r_success": 1000.0,
                        "kuka_actions_penalty_scale": 3e-3,
                        "allegro_actions_penalty_scale": 3e-4,
                        "wm_strategy": "value_imag_h3",
                        "wm_ensemble_size": 3,
                        "enable_wm": False,
                        "enable_counterfactual": False,
                    },
                    checkpoint_path=str(ckpt),
                    source_policy=0,
                    env_frames=300_000_000,
                    tier="bottom",
                    source_curriculum={
                        "success_tolerance": 0.05,
                        "initial_tolerance": 0.075,
                        "target_tolerance": 0.01,
                        "frame_since_restart": 200_000_000,
                        "last_curriculum_update": 0,
                    },
                )
            except SystemExit:
                pass

        manifests = sorted(
            self.workspace.rglob("pbt_restart.json"),
            key=lambda p: p.stat().st_mtime,
        )
        self.assertTrue(manifests, "no pbt_restart.json written")
        return json.loads(manifests[-1].read_text())

    def test_writes_regular_freeze_for_cross_copy(self):
        obs = self._make()
        manifest = self._captured_train_args(obs, checkpoint_type="live")
        train_args = manifest.get("train_args", [])
        match = [a for a in train_args
                 if a.startswith("--curriculum_freeze_steps_after_restart_resume=")]
        self.assertEqual(len(match), 1, f"missing freeze hook in {train_args}")
        value = int(match[0].split("=", 1)[1])
        self.assertEqual(value, 120_000_000)

    def test_writes_longer_freeze_for_archive_recovery(self):
        obs = self._make()
        manifest = self._captured_train_args(obs, checkpoint_type="archive")
        train_args = manifest.get("train_args", [])
        match = [a for a in train_args
                 if a.startswith("--curriculum_freeze_steps_after_restart_resume=")]
        self.assertEqual(len(match), 1)
        value = int(match[0].split("=", 1)[1])
        self.assertEqual(value, 250_000_000)


if __name__ == "__main__":
    unittest.main()
