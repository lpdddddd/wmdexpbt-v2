"""P0: archive recovery executes after fsr fix; freeze uses env-frame conversion."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from dexpbt_isaaclab.dexpbt_utils import env_frames_to_sim_steps
from dexpbt_isaaclab.pbt.pbt_observer import DexPBTPbtAlgoObserver


def _state(*, frame: int, frames_at_restart: int = 0) -> dict:
    return {
        "policy_idx": 6,
        "frame": frame,
        "frames_at_restart": frames_at_restart,
        "restart_count": 0,
        "nsucc_mean": 0.0,
        "nsucc_max": 0.0,
        "curriculum": {"frame_since_restart": frame // 8192},
        "checkpoint": "/tmp/x.pth",
        "params": {},
    }


class ArchiveRecoveryExecutesTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.workspace = Path(self._tmp.name)

    def _obs(self) -> DexPBTPbtAlgoObserver:
        obs = DexPBTPbtAlgoObserver(
            policy_idx=6,
            num_policies=8,
            mutable_params={
                "enable_wm": False,
                "enable_counterfactual": False,
                "wm_strategy": "none",
            },
            workspace=str(self.workspace),
            train_dir=str(self.workspace / "run"),
            task_name="test-task",
            interval_steps=20_000_000,
            start_after=0,
            initial_delay=0,
            num_envs=8192,
            pbt_min_frames_between_restarts=50_000_000,
            pbt_enable_best_nsucc_recovery=True,
            pbt_recovery_min_nsucc_mean=2.0,
            pbt_recovery_min_nsucc_max=8.0,
            pbt_curriculum_freeze_on_recovery_frames=200_000_000,
        )
        obs.writer = None
        return obs

    def test_elimination_gate_passes_after_fsr_fix(self):
        obs = self._obs()
        own = _state(frame=2_780_037_120, frames_at_restart=0)
        self.assertTrue(obs._is_eligible_for_elimination(own, env_frames=2_780_037_120))

    def test_archive_recovery_freeze_uses_env_frames(self):
        steps = env_frames_to_sim_steps(200_000_000, 8192)
        self.assertEqual(steps, 24_415)

        obs = self._obs()
        ckpt = self.workspace / "peak.pth"
        ckpt.write_text("noop")
        archive_dir = self.workspace / "best_nsucc"
        archive_dir.mkdir()
        meta = {
            "policy_idx": 0,
            "nsucc_mean": 4.27,
            "nsucc_max": 45.0,
            "frame": 480_000_000,
            "iteration": 24,
            "curriculum": {"success_tolerance": 0.036},
            "checkpoint": str(ckpt),
        }
        (archive_dir / "peak.json").write_text(json.dumps(meta))
        (archive_dir / "_index.json").write_text(json.dumps(meta))
        (archive_dir / "peak.pth").write_bytes(b"x")

        obs._last_source_checkpoint_type = "archive"
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
                    env_frames=600_000_000,
                    tier="bottom",
                    source_curriculum=meta["curriculum"],
                )
            except SystemExit:
                pass

        manifests = sorted(self.workspace.rglob("pbt_restart.json"))
        self.assertTrue(manifests)
        manifest = json.loads(manifests[-1].read_text())
        match = [
            a for a in manifest["train_args"]
            if a.startswith("--curriculum_freeze_steps_after_restart_resume=")
        ]
        self.assertEqual(int(match[0].split("=", 1)[1]), steps)


if __name__ == "__main__":
    unittest.main()
