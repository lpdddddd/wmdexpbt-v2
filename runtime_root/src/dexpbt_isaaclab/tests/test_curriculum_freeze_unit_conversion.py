"""P0: curriculum freeze env-frame -> simulation-step conversion (Scheme A)."""

import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from dexpbt_isaaclab.dexpbt_utils import dexpbt_tolerance_curriculum, env_frames_to_sim_steps
from dexpbt_isaaclab.pbt.pbt_observer import DexPBTPbtAlgoObserver


def _ts(mean_nsucc: float) -> torch.Tensor:
    return torch.full((4,), float(mean_nsucc))


class EnvFramesToSimStepsTest(unittest.TestCase):
    def test_100m_env_frames_at_8192_envs(self):
        steps = env_frames_to_sim_steps(100_000_000, 8192)
        self.assertEqual(steps, 12_208)
        self.assertEqual(steps, math.ceil(100_000_000 / 8192))

    def test_freeze_holds_for_converted_steps_not_env_frames(self):
        freeze_steps = env_frames_to_sim_steps(100_000_000, 8192)
        tol, lcu = dexpbt_tolerance_curriculum(
            last_curriculum_update=0,
            frames_since_restart=12_207,
            curriculum_interval=3000,
            prev_episode_successes=_ts(50.0),
            success_tolerance=0.06,
            initial_tolerance=0.075,
            target_tolerance=0.01,
            curriculum_freeze_frames_after_restart=freeze_steps,
        )
        self.assertEqual(tol, 0.06)
        self.assertEqual(lcu, 12_207)

        tol2, _ = dexpbt_tolerance_curriculum(
            last_curriculum_update=0,
            frames_since_restart=12_208,
            curriculum_interval=3000,
            prev_episode_successes=_ts(50.0),
            success_tolerance=0.06,
            initial_tolerance=0.075,
            target_tolerance=0.01,
            curriculum_freeze_frames_after_restart=freeze_steps,
        )
        self.assertAlmostEqual(tol2, 0.054, places=3)


class ObserverFreezeStepsInjectionTest(unittest.TestCase):
    def test_observer_converts_recovery_freeze_for_8192_envs(self):
        with tempfile.TemporaryDirectory() as td:
            workspace = Path(td)
            run_dir = workspace / "run"
            run_dir.mkdir()
            ckpt = workspace / "fake.pth"
            ckpt.write_text("noop")

            obs = DexPBTPbtAlgoObserver(
                policy_idx=0,
                num_policies=2,
                mutable_params={
                    "enable_wm": False,
                    "enable_counterfactual": False,
                    "wm_strategy": "none",
                },
                workspace=str(workspace),
                train_dir=str(run_dir),
                task_name="test-task",
                interval_steps=10_000_000,
                start_after=0,
                initial_delay=0,
                num_envs=8192,
                pbt_curriculum_freeze_frames=100_000_000,
                pbt_curriculum_freeze_on_recovery_frames=200_000_000,
            )
            obs.writer = None
            obs._last_source_checkpoint_type = "archive"

            import json
            import os
            import shutil as _shutil
            import signal as _signal

            with patch.object(_shutil, "copyfile", lambda *a, **kw: None), \
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
                            "wm_strategy": "none",
                            "wm_ensemble_size": 3,
                            "enable_wm": False,
                            "enable_counterfactual": False,
                        },
                        checkpoint_path=str(ckpt),
                        source_policy=0,
                        env_frames=300_000_000,
                        tier="bottom",
                        source_curriculum={"success_tolerance": 0.036},
                    )
                except SystemExit:
                    pass

            manifests = sorted(workspace.rglob("pbt_restart.json"))
            self.assertTrue(manifests, "no pbt_restart.json written")
            manifest = json.loads(manifests[-1].read_text())
            match = [
                a for a in manifest["train_args"]
                if a.startswith("--curriculum_freeze_steps_after_restart_resume=")
            ]
            self.assertEqual(len(match), 1)
            steps = int(match[0].split("=", 1)[1])
            self.assertEqual(steps, env_frames_to_sim_steps(200_000_000, 8192))


if __name__ == "__main__":
    unittest.main()
