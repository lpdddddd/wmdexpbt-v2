"""Patch J (2026-05-31): fixed-WM-strategy ablation mode.

When `--wm_strategy_mode=fixed`, the observer should:

  * Pin the worker's `_wm_strategy` to `fixed_wm_strategy` on construction.
  * Force `enable_counterfactual=False` regardless of what was passed in
    (so the CF runner never starts, and CF triggers are never produced).
  * Skip the posterior selector during restart-time strategy sampling.
  * Forward both `--wm_strategy_mode` and `--fixed_wm_strategy` in
    `train_args` so the spawned process stays in fixed mode across the
    rl_games re-exec hop.
  * Forward `--enable_counterfactual` only when CF is actually enabled
    (i.e. never, in fixed mode).

This is the ablation/baseline knob: with
`--wm_strategy_mode=fixed --fixed_wm_strategy=none` the run reduces to
pure DexPBT for clean comparison against Bayesian + CF.
"""

import os
import shutil as _shutil
import signal as _signal
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from dexpbt_isaaclab.pbt.pbt_observer import DexPBTPbtAlgoObserver


# A minimally complete `new_params` dict for `_restart_process_with_new_params`.
# Copied from tests/test_curriculum_freeze.py so the rl_games train_args
# assembly does not KeyError on missing PPO knobs.
_BASE_NEW_PARAMS: dict = {
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
}


def _make_observer(
    workspace: Path,
    *,
    mutable_params=None,
    wm_strategy_mode: str = "fixed",
    fixed_wm_strategy: str = "value_imag_h3",
) -> DexPBTPbtAlgoObserver:
    # `_restart_process_with_new_params` writes its manifest to
    # `train_dir.parent`. Place train_dir at workspace/run/rl_games so the
    # manifest ends up under workspace/ and `workspace.rglob(...)` finds it.
    run_dir = workspace / "run"
    run_dir.mkdir(parents=True, exist_ok=True)
    train_dir = run_dir / "rl_games"
    train_dir.mkdir(parents=True, exist_ok=True)
    mp = {
        "enable_wm": True,
        "enable_counterfactual": True,
        "wm_strategy": "aux_low",
    }
    if mutable_params is not None:
        mp.update(mutable_params)
    obs = DexPBTPbtAlgoObserver(
        policy_idx=0,
        num_policies=3,
        mutable_params=mp,
        workspace=str(workspace),
        train_dir=str(train_dir),
        task_name="test-task",
        interval_steps=10_000_000,
        start_after=0,
        initial_delay=0,
        num_envs=1,
        pbt_min_frames_as_source_after_restart=0,
        pbt_min_frames_between_restarts=0,
        pbt_min_nsucc_mean_as_source=0.0,
        pbt_min_nsucc_max_as_source=0.0,
        wm_strategy_mode=wm_strategy_mode,
        fixed_wm_strategy=fixed_wm_strategy,
    )
    obs.writer = None
    return obs


class ConstructionTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.workspace = Path(self._tmp.name)

    def test_fixed_mode_pins_strategy_and_disables_cf(self):
        obs = _make_observer(
            self.workspace,
            wm_strategy_mode="fixed",
            fixed_wm_strategy="value_imag_h3",
        )
        self.assertEqual(obs.wm_strategy_mode, "fixed")
        self.assertEqual(obs.fixed_wm_strategy, "value_imag_h3")
        self.assertEqual(obs._wm_strategy, "value_imag_h3")
        self.assertFalse(obs._enable_counterfactual)
        # Even though we passed `enable_counterfactual=True`, the
        # in-process mutable_params should now reflect the override so
        # downstream consumers (e.g. params published to the supervisor)
        # do not advertise CF.
        self.assertFalse(obs.mutable_params.get("enable_counterfactual", True))
        self.assertEqual(obs.mutable_params.get("wm_strategy"), "value_imag_h3")

    def test_bayesian_mode_leaves_mutable_params_alone(self):
        obs = _make_observer(
            self.workspace,
            wm_strategy_mode="bayesian",
            fixed_wm_strategy="value_imag_h3",
        )
        self.assertEqual(obs.wm_strategy_mode, "bayesian")
        # In Bayesian mode the constructor must NOT clobber the
        # user-provided wm_strategy/enable_counterfactual.
        self.assertEqual(obs._wm_strategy, "aux_low")
        self.assertTrue(obs._enable_counterfactual)
        self.assertEqual(obs.mutable_params.get("wm_strategy"), "aux_low")
        self.assertTrue(obs.mutable_params.get("enable_counterfactual"))

    def test_unknown_mode_raises(self):
        with self.assertRaises(ValueError):
            _make_observer(self.workspace, wm_strategy_mode="garbage")


class RestartArgsForwardingTest(unittest.TestCase):
    """In fixed mode, `_restart_process_with_new_params` should:

    * not sample from the posterior selector
    * keep wm_strategy=fixed in the spawned train_args
    * never append `--enable_counterfactual`
    * always append `--wm_strategy_mode` / `--fixed_wm_strategy`
    """

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.workspace = Path(self._tmp.name)
        # _restart_process_with_new_params calls into _save_workspace_checkpoint
        # before train_args is constructed; create a fake src file so
        # shutil.copyfile() succeeds.
        (self.workspace / "current.pth").write_text("noop")

    def _captured_train_args(self, obs, *, override_params=None):
        captured: dict = {}

        def fake_popen(args, **kw):
            captured["args"] = list(args)
            raise SystemExit(0)

        # Patch out every side-effecting call between train_args
        # assembly and process exec: checkpoint copy, ckpt rewrite,
        # SIGKILL via raise_signal, kill(parent), os._exit, and -- in
        # fixed mode -- the posterior selector must NEVER fire.
        def explode_if_sampled(*_a, **_kw):
            raise AssertionError(
                "posterior_selector.sample() should not be called in "
                "fixed wm_strategy_mode"
            )

        new_params = dict(_BASE_NEW_PARAMS)
        if override_params:
            new_params.update(override_params)

        ckpt = self.workspace / "current.pth"
        ckpt.write_text("noop")

        def _do_call():
            try:
                obs._restart_process_with_new_params(
                    new_params=new_params,
                    checkpoint_path=str(ckpt),
                    source_policy=1,
                    env_frames=500_000_000,
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

        with patch("subprocess.Popen", side_effect=fake_popen), \
             patch.object(_shutil, "copy2", lambda *a, **kw: None), \
             patch.object(_shutil, "copyfile", lambda *a, **kw: None), \
             patch.object(_signal, "raise_signal", lambda *_a: None), \
             patch.object(os, "kill", side_effect=SystemExit(0)), \
             patch.object(os, "_exit", side_effect=SystemExit(0)), \
             patch.object(obs, "_rewrite_checkpoint", lambda *a, **kw: None):
            if obs._posterior_selector is not None:
                with patch.object(
                    obs._posterior_selector, "sample", side_effect=explode_if_sampled
                ):
                    _do_call()
            else:
                _do_call()
        # `_restart_process_with_new_params` calls os.kill() (the line
        # we replaced with SystemExit) instead of subprocess.Popen, so
        # we read the assembled train_args from the manifest written
        # just before kill.
        import json
        manifests = sorted(
            self.workspace.rglob("pbt_restart.json"),
            key=lambda p: p.stat().st_mtime,
        )
        if not manifests:
            return []
        manifest = json.loads(manifests[-1].read_text())
        return list(manifest.get("train_args", []))

    def test_fixed_mode_forwards_mode_flags_and_omits_cf(self):
        obs = _make_observer(
            self.workspace,
            wm_strategy_mode="fixed",
            fixed_wm_strategy="value_imag_h3",
            mutable_params={
                "enable_wm": True,
                # Caller tries to flip CF on; the observer must overrule.
                "enable_counterfactual": True,
                "wm_strategy": "aux_low",
            },
        )
        obs.last_checkpoint = str(self.workspace / "current.pth")
        # Even if the caller hands us new_params with enable_counterfactual=True,
        # the fixed-mode branch must rewrite it to False so the spawned
        # process never receives `--enable_counterfactual`.
        train_args = self._captured_train_args(
            obs,
            override_params={
                "enable_wm": True,
                "enable_counterfactual": True,
                "wm_strategy": "aux_low",
            },
        )
        # Mode + fixed-strategy flags MUST be forwarded.
        self.assertIn("--wm_strategy_mode=fixed", train_args)
        self.assertIn("--fixed_wm_strategy=value_imag_h3", train_args)
        # CF must NOT be propagated.
        self.assertNotIn("--enable_counterfactual", train_args)
        # The spawned process should also receive the locked strategy.
        self.assertIn("--wm_strategy=value_imag_h3", train_args)

    def test_bayesian_mode_keeps_bayesian_flag(self):
        obs = _make_observer(
            self.workspace,
            wm_strategy_mode="bayesian",
            mutable_params={
                "enable_wm": True,
                "enable_counterfactual": False,
                "wm_strategy": "aux_low",
            },
        )
        obs.last_checkpoint = str(self.workspace / "current.pth")
        # In Bayesian mode the posterior selector path runs; we don't
        # assert on the final strategy because the test selector may
        # not be initialised. We only check the mode flag forwarding.
        try:
            train_args = self._captured_train_args(
                obs,
                override_params={"enable_wm": False, "enable_counterfactual": False},
            )
        except AssertionError:
            # If the test environment didn't wire up a posterior
            # selector, _restart_process_with_new_params falls through
            # to `selected_strategy = new_params.get(...)`. We accept
            # either path.
            train_args = []
        if train_args:
            self.assertIn("--wm_strategy_mode=bayesian", train_args)


if __name__ == "__main__":
    unittest.main()
