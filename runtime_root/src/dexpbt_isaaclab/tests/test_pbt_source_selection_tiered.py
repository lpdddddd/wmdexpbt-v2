"""Patch G (2026-05-31): tiered source selection + best_nsucc recovery.

The previous Patch A behaviour was binary (strict-or-skip): when all
top-N workers failed the strict nsucc gate, the bottom worker simply
skipped its PBT replacement. Smoke run 20260530_200619 showed this
produced too many "skip" ticks once the population co-degraded — no
cross-copy could ever fire, so good ideas couldn't propagate.

Patch G introduces the cascade L1 strict -> L2 relaxed -> L3 archive
-> L4 skip in `_select_source`. Tests here pin down:

  1. Each tier's eligibility helpers in isolation.
  2. `_select_best_nsucc_recovery_source` correctly reads the archive,
     applies health gates, applies cooldown, and rejects missing pths.
  3. `_select_source` cascade ordering produces the right tier label.
  4. Resume hooks for recovery accounting take effect.
  5. End-to-end `_ternary_pbt` bottom branch transitions through tiers.
"""

import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from dexpbt_isaaclab.pbt.pbt_observer import DexPBTPbtAlgoObserver


def _make_observer(
    workspace: Path,
    *,
    source_selection_mode: str = "tiered",
    min_nsucc_mean_as_source: float = 1.0,
    min_nsucc_max_as_source: float = 5.0,
    relaxed_nsucc_mean_as_source: float = 0.5,
    relaxed_nsucc_max_as_source: float = 2.0,
    enable_best_nsucc_recovery: bool = True,
    recovery_min_nsucc_mean: float = 2.0,
    recovery_min_nsucc_max: float = 10.0,
    recovery_cooldown_frames: int = 200_000_000,
    min_frames_as_source_after_restart: int = 0,
    min_frames_between_restarts: int = 0,
    recovery_restart_count_resume: int | None = None,
    last_recovery_frame_resume: int | None = None,
    num_policies: int = 3,
    policy_idx: int = 2,
) -> DexPBTPbtAlgoObserver:
    obs = DexPBTPbtAlgoObserver(
        policy_idx=policy_idx,
        num_policies=num_policies,
        mutable_params={
            "enable_wm": False,
            "enable_counterfactual": False,
            "wm_strategy": "none",
        },
        workspace=str(workspace),
        train_dir=str(workspace),
        task_name="test-task",
        interval_steps=10_000_000,
        start_after=0,
        initial_delay=0,
        num_envs=1,
        pbt_min_frames_as_source_after_restart=min_frames_as_source_after_restart,
        pbt_min_frames_between_restarts=min_frames_between_restarts,
        pbt_min_nsucc_mean_as_source=min_nsucc_mean_as_source,
        pbt_min_nsucc_max_as_source=min_nsucc_max_as_source,
        pbt_source_selection_mode=source_selection_mode,
        pbt_relaxed_nsucc_mean_as_source=relaxed_nsucc_mean_as_source,
        pbt_relaxed_nsucc_max_as_source=relaxed_nsucc_max_as_source,
        pbt_enable_best_nsucc_recovery=enable_best_nsucc_recovery,
        pbt_recovery_min_nsucc_mean=recovery_min_nsucc_mean,
        pbt_recovery_min_nsucc_max=recovery_min_nsucc_max,
        pbt_recovery_cooldown_frames=recovery_cooldown_frames,
        recovery_restart_count_resume=recovery_restart_count_resume,
        last_recovery_frame_resume=last_recovery_frame_resume,
    )
    obs.writer = None
    return obs


def _state(
    *,
    policy_idx: int = 0,
    frame: int = 300_000_000,
    nsucc_mean: float = 5.0,
    nsucc_max: float = 20.0,
    pbt_ranking_score: float = 100.0,
    frame_since_restart: int = 200_000_000,
    frames_at_restart: int = 0,
    checkpoint: str | None = None,
):
    return {
        "policy_idx": policy_idx,
        "frame": frame,
        "frames_at_restart": frames_at_restart,
        "iteration": 10,
        "objective": pbt_ranking_score,
        "pbt_ranking_score": pbt_ranking_score,
        "r_meta_raw": pbt_ranking_score,
        "nsucc_mean": nsucc_mean,
        "nsucc_max": nsucc_max,
        "checkpoint": checkpoint or "/tmp/dummy.pth",
        "curriculum": {
            "frame_since_restart": frame_since_restart,
            "mean_prev_episode_successes": nsucc_mean,
            "max_prev_episode_successes": nsucc_max,
            "success_tolerance": 0.05,
        },
        "params": {},
    }


def _seed_skill_delta_window(
    obs,
    policy_idx: int,
    *,
    nsucc_seq: list[float],
    env_frames: int,
    lineage: str = "same",
    tol: float = 0.05,
):
    """Seed a contiguous same-lineage window for skill_delta gates."""
    from dexpbt_isaaclab.pbt.pbt_curriculum_p1 import WindowSample

    w = obs._policy_window(policy_idx)
    w.samples = []
    n = len(nsucc_seq)
    step = int(obs.interval_steps)
    for i, mean in enumerate(nsucc_seq):
        w.samples.append(
            WindowSample(
                env_frames=int(env_frames) - (n - 1 - i) * step,
                nsucc_mean=float(mean),
                nsucc_max=float(mean) + 5.0,
                success_tolerance=float(tol),
                pbt_iteration=i,
                lineage_id=lineage,
            )
        )
    return w


def _seed_bottom_skill_delta_for_ternary(
    obs,
    population: dict,
    *,
    env_frames: int,
    target_idx: int,
):
    """Target stuck (nsucc<1, nonpositive delta); live sources stably improving."""
    for idx, st in population.items():
        mean = float(st.get("nsucc_mean", 0.0))
        if int(idx) == int(target_idx):
            # Flat zero → stuck + nonpositive delta.
            _seed_skill_delta_window(
                obs, idx, nsucc_seq=[0.0] * 6, env_frames=env_frames
            )
        elif mean > 0.0:
            # Strictly increasing → stable positive skill_delta for live source.
            seq = [max(0.0, mean - 1.0 + 0.2 * i) for i in range(6)]
            _seed_skill_delta_window(obs, idx, nsucc_seq=seq, env_frames=env_frames)
        else:
            _seed_skill_delta_window(
                obs, idx, nsucc_seq=[0.0] * 6, env_frames=env_frames
            )


def _write_archive_peak(
    workspace: Path,
    *,
    policy_idx: int,
    iteration: int,
    frame: int,
    nsucc_mean: float,
    nsucc_max: float,
    r_meta_raw: float = 0.0,
    task_name: str = "test-task",
) -> Path:
    """Mimic what `_save_peak_archive('nsucc', state)` would have written."""
    archive_dir = workspace / "best_nsucc"
    archive_dir.mkdir(parents=True, exist_ok=True)
    prefix = (
        f"{task_name}_best_nsucc"
        f"_iter_{iteration:04d}_policy{policy_idx:03d}"
        f"_frame_{frame // 1_000_000:05d}M"
        f"_nsuccM_{max(0.0, nsucc_mean):07.3f}_nsuccX_{max(0.0, nsucc_max):06.1f}"
        f"_rmeta_{max(0.0, r_meta_raw):08.4f}"
    )
    pth = archive_dir / f"{prefix}.pth"
    pth.write_text("dummy_ckpt_bytes")
    state = {
        "policy_idx": policy_idx,
        "iteration": iteration,
        "frame": frame,
        "nsucc_mean": nsucc_mean,
        "nsucc_max": nsucc_max,
        "r_meta_raw": r_meta_raw,
        "checkpoint": str(pth),
        "params": {},
        "curriculum": {
            "success_tolerance": 0.05,
            "mean_prev_episode_successes": nsucc_mean,
            "max_prev_episode_successes": nsucc_max,
        },
    }
    (archive_dir / f"{prefix}.json").write_text(json.dumps(state, indent=2))
    (archive_dir / "_index.json").write_text(json.dumps(state, indent=2))
    return pth


class RelaxedGateTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.workspace = Path(self._tmp.name)

    def test_passes_under_relaxed_thresholds(self):
        obs = _make_observer(self.workspace)
        s = _state(nsucc_mean=0.7, nsucc_max=3.0)
        ok, reason = obs._eligible_as_source_relaxed_with_reason(
            s, env_frames=300_000_000
        )
        self.assertTrue(ok, reason)

    def test_blocks_below_relaxed_mean(self):
        obs = _make_observer(self.workspace)
        s = _state(nsucc_mean=0.1, nsucc_max=10.0)
        ok, reason = obs._eligible_as_source_relaxed_with_reason(
            s, env_frames=300_000_000
        )
        self.assertFalse(ok)
        self.assertIn("relaxed_nsucc_mean", reason)

    def test_blocks_below_relaxed_max(self):
        obs = _make_observer(self.workspace)
        s = _state(nsucc_mean=5.0, nsucc_max=1.0)
        ok, reason = obs._eligible_as_source_relaxed_with_reason(
            s, env_frames=300_000_000
        )
        self.assertFalse(ok)
        self.assertIn("relaxed_nsucc_max", reason)

    def test_frame_cooldown_still_applies(self):
        # Anti-churn frame-cooldown is a precondition for L2 as well.
        obs = _make_observer(
            self.workspace,
            min_frames_as_source_after_restart=50_000_000,
        )
        s = _state(
            frame=60_000_000,
            frames_at_restart=50_000_000,
            frame_since_restart=10_000_000,
            nsucc_mean=5.0,
            nsucc_max=20.0,
        )
        ok, reason = obs._eligible_as_source_relaxed_with_reason(
            s, env_frames=300_000_000
        )
        self.assertFalse(ok)
        self.assertIn("frames_since_restart", reason)

    def test_handles_none_state(self):
        obs = _make_observer(self.workspace)
        ok, reason = obs._eligible_as_source_relaxed_with_reason(
            None, env_frames=300_000_000
        )
        self.assertFalse(ok)
        self.assertEqual(reason, "no_state")


class ArchiveRecoveryPickerTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.workspace = Path(self._tmp.name)

    def test_disabled_returns_reason(self):
        obs = _make_observer(self.workspace, enable_best_nsucc_recovery=False)
        synth, reason = obs._select_best_nsucc_recovery_source(env_frames=300_000_000)
        self.assertIsNone(synth)
        self.assertEqual(reason, "recovery_disabled")

    def test_empty_archive(self):
        obs = _make_observer(self.workspace)
        synth, reason = obs._select_best_nsucc_recovery_source(env_frames=300_000_000)
        self.assertIsNone(synth)
        self.assertEqual(reason, "archive_empty")

    def test_below_health_gate(self):
        _write_archive_peak(
            self.workspace,
            policy_idx=0,
            iteration=5,
            frame=200_000_000,
            nsucc_mean=1.0,  # below default recovery_min_nsucc_mean=2.0
            nsucc_max=20.0,
        )
        obs = _make_observer(self.workspace)
        synth, reason = obs._select_best_nsucc_recovery_source(env_frames=300_000_000)
        self.assertIsNone(synth)
        self.assertIn("archive_nsucc_mean", reason)

    def test_picks_archive_when_healthy(self):
        peak_pth = _write_archive_peak(
            self.workspace,
            policy_idx=0,
            iteration=5,
            frame=200_000_000,
            nsucc_mean=3.0,
            nsucc_max=25.0,
        )
        obs = _make_observer(self.workspace)
        synth, reason = obs._select_best_nsucc_recovery_source(env_frames=300_000_000)
        self.assertEqual(reason, "ok")
        self.assertIsNotNone(synth)
        self.assertEqual(synth["checkpoint"], str(peak_pth))
        self.assertEqual(synth["_source_checkpoint_type"], "archive")

    def test_cooldown_blocks_replay(self):
        _write_archive_peak(
            self.workspace,
            policy_idx=0,
            iteration=5,
            frame=200_000_000,
            nsucc_mean=3.0,
            nsucc_max=25.0,
        )
        obs = _make_observer(
            self.workspace,
            recovery_cooldown_frames=100_000_000,
            last_recovery_frame_resume=400_000_000,
        )
        synth, reason = obs._select_best_nsucc_recovery_source(env_frames=420_000_000)
        self.assertIsNone(synth)
        self.assertEqual(reason, "recovery_cooldown")

    def test_missing_pth_returns_reason(self):
        _write_archive_peak(
            self.workspace,
            policy_idx=0,
            iteration=5,
            frame=200_000_000,
            nsucc_mean=3.0,
            nsucc_max=25.0,
        )
        # Delete every .pth file leaving the _index.json behind.
        for p in (self.workspace / "best_nsucc").glob("*.pth"):
            p.unlink()
        obs = _make_observer(self.workspace)
        synth, reason = obs._select_best_nsucc_recovery_source(env_frames=300_000_000)
        self.assertIsNone(synth)
        self.assertEqual(reason, "archive_pth_missing")


class SelectSourceCascadeTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.workspace = Path(self._tmp.name)

    def _decision(self, obs, population, env_frames=300_000_000, n_top=1):
        sorted_pol = sorted(
            population.keys(),
            key=lambda k: population[k]["pbt_ranking_score"],
            reverse=True,
        )
        return obs._select_source(
            sorted_policies=sorted_pol,
            population=population,
            env_frames=env_frames,
            n_top=n_top,
        )

    def test_l1_strict_fires_when_any_eligible(self):
        obs = _make_observer(self.workspace)
        population = {
            0: _state(policy_idx=0, pbt_ranking_score=100.0,
                      nsucc_mean=5.0, nsucc_max=25.0),
            1: _state(policy_idx=1, pbt_ranking_score=50.0,
                      nsucc_mean=0.0, nsucc_max=0.0),
            2: _state(policy_idx=2, pbt_ranking_score=10.0,
                      nsucc_mean=0.0, nsucc_max=0.0),
        }
        d = self._decision(obs, population)
        self.assertEqual(d["tier"], "strict_live")
        self.assertEqual(d["source_policy"], 0)
        self.assertEqual(d["checkpoint_type"], "live")

    def test_l2_relaxed_when_strict_fails(self):
        obs = _make_observer(self.workspace)
        # top idx=0 fails strict but passes relaxed (nsucc_mean=0.7);
        # bottom idx=2 is the worker doing the selection.
        population = {
            0: _state(policy_idx=0, pbt_ranking_score=100.0,
                      nsucc_mean=0.1, nsucc_max=0.0),  # strict + relaxed fail
            1: _state(policy_idx=1, pbt_ranking_score=50.0,
                      nsucc_mean=0.7, nsucc_max=3.0),  # relaxed pass
            2: _state(policy_idx=2, pbt_ranking_score=10.0,
                      nsucc_mean=0.0, nsucc_max=0.0),
        }
        d = self._decision(obs, population)
        self.assertEqual(d["tier"], "relaxed_live")
        self.assertEqual(d["source_policy"], 1)
        self.assertEqual(d["checkpoint_type"], "live")

    def test_l3_archive_when_no_live_eligible(self):
        # Pre-seed archive with a healthy peak.
        _write_archive_peak(
            self.workspace,
            policy_idx=0,
            iteration=5,
            frame=200_000_000,
            nsucc_mean=3.0,
            nsucc_max=25.0,
        )
        obs = _make_observer(self.workspace)
        # Everyone failed both gates.
        population = {
            0: _state(policy_idx=0, pbt_ranking_score=100.0,
                      nsucc_mean=0.0, nsucc_max=0.0),
            1: _state(policy_idx=1, pbt_ranking_score=50.0,
                      nsucc_mean=0.0, nsucc_max=0.0),
            2: _state(policy_idx=2, pbt_ranking_score=10.0,
                      nsucc_mean=0.0, nsucc_max=0.0),
        }
        d = self._decision(obs, population)
        self.assertEqual(d["tier"], "archive")
        self.assertEqual(d["checkpoint_type"], "archive")
        self.assertEqual(d["source_policy"], 0)

    def test_l4_skip_when_everything_fails(self):
        obs = _make_observer(self.workspace)
        population = {
            0: _state(policy_idx=0, pbt_ranking_score=100.0,
                      nsucc_mean=0.0, nsucc_max=0.0),
            1: _state(policy_idx=1, pbt_ranking_score=50.0,
                      nsucc_mean=0.0, nsucc_max=0.0),
            2: _state(policy_idx=2, pbt_ranking_score=10.0,
                      nsucc_mean=0.0, nsucc_max=0.0),
        }
        d = self._decision(obs, population)
        self.assertEqual(d["tier"], "skipped")
        self.assertEqual(d["action"], "skip")

    def test_strict_mode_does_not_consult_l2_or_l3(self):
        _write_archive_peak(
            self.workspace,
            policy_idx=0,
            iteration=5,
            frame=200_000_000,
            nsucc_mean=3.0,
            nsucc_max=25.0,
        )
        obs = _make_observer(self.workspace, source_selection_mode="strict")
        population = {
            0: _state(policy_idx=0, pbt_ranking_score=100.0,
                      nsucc_mean=0.1, nsucc_max=0.0),
            1: _state(policy_idx=1, pbt_ranking_score=50.0,
                      nsucc_mean=0.7, nsucc_max=3.0),
            2: _state(policy_idx=2, pbt_ranking_score=10.0,
                      nsucc_mean=0.0, nsucc_max=0.0),
        }
        d = self._decision(obs, population)
        self.assertEqual(d["tier"], "skipped")  # strict-or-skip


class TernaryPbtBottomBranchTieredTest(unittest.TestCase):
    """End-to-end through `_ternary_pbt`."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.workspace = Path(self._tmp.name)
        self.ckpt = self.workspace / "dummy.pth"
        self.ckpt.write_text("dummy_ckpt_bytes")

    def _patched(self, obs):
        return patch.multiple(
            obs,
            _maybe_save_best_policy=lambda *a, **kw: None,
            _restart_process_with_new_params=lambda *a, **kw: True,
        )

    def _call_ternary(self, obs, population, env_frames):
        objectives = {
            i: float(s["pbt_ranking_score"]) for i, s in population.items()
        }
        sorted_pol = sorted(
            objectives, key=lambda k: objectives[k], reverse=True
        )
        obs._ternary_pbt(
            sorted_policies=sorted_pol,
            objectives=objectives,
            population=population,
            env_frames=env_frames,
        )

    def test_strict_live_path_increments_cross_counter(self):
        obs = _make_observer(self.workspace, num_policies=3, policy_idx=2)
        ck = str(self.ckpt)
        population = {
            0: _state(policy_idx=0, pbt_ranking_score=100.0,
                      nsucc_mean=5.0, nsucc_max=25.0, checkpoint=ck),
            1: _state(policy_idx=1, pbt_ranking_score=50.0,
                      nsucc_mean=0.0, nsucc_max=0.0, checkpoint=ck),
            2: _state(policy_idx=2, pbt_ranking_score=10.0,
                      nsucc_mean=0.0, nsucc_max=0.0, checkpoint=ck),
        }
        _seed_bottom_skill_delta_for_ternary(
            obs, population, env_frames=300_000_000, target_idx=2
        )
        with self._patched(obs):
            self._call_ternary(obs, population, env_frames=300_000_000)
        self.assertEqual(obs._cross_restart_count, 1)
        self.assertEqual(obs._recovery_restart_count, 0)
        self.assertEqual(obs._source_tier_counts["strict_live"], 1)
        self.assertEqual(obs._last_source_tier_label, "strict_live")
        self.assertEqual(obs._last_source_checkpoint_type, "live")
        self.assertEqual(obs._last_restart_type, 2)

    def test_archive_path_increments_recovery_counter(self):
        _write_archive_peak(
            self.workspace,
            policy_idx=0,
            iteration=5,
            frame=200_000_000,
            nsucc_mean=3.0,
            nsucc_max=25.0,
        )
        obs = _make_observer(self.workspace, num_policies=3, policy_idx=2)
        ck = str(self.ckpt)
        population = {
            0: _state(policy_idx=0, pbt_ranking_score=100.0,
                      nsucc_mean=0.0, nsucc_max=0.0, checkpoint=ck),
            1: _state(policy_idx=1, pbt_ranking_score=50.0,
                      nsucc_mean=0.0, nsucc_max=0.0, checkpoint=ck),
            2: _state(policy_idx=2, pbt_ranking_score=10.0,
                      nsucc_mean=0.0, nsucc_max=0.0, checkpoint=ck),
        }
        _seed_bottom_skill_delta_for_ternary(
            obs, population, env_frames=300_000_000, target_idx=2
        )
        with self._patched(obs):
            self._call_ternary(obs, population, env_frames=300_000_000)
        self.assertEqual(obs._recovery_restart_count, 1)
        self.assertEqual(obs._cross_restart_count, 0)
        self.assertEqual(obs._source_tier_counts["archive"], 1)
        self.assertEqual(obs._last_source_tier_label, "archive")
        self.assertEqual(obs._last_source_checkpoint_type, "archive")
        self.assertEqual(obs._last_recovery_frame, 300_000_000)
        # restart_type=4 marks recovery (distinct from self/cross/skipped).
        self.assertEqual(obs._last_restart_type, 4)

    def test_l4_skip_does_not_increment_any_counter(self):
        obs = _make_observer(self.workspace, num_policies=3, policy_idx=2)
        ck = str(self.ckpt)
        population = {
            0: _state(policy_idx=0, pbt_ranking_score=100.0,
                      nsucc_mean=0.0, nsucc_max=0.0, checkpoint=ck),
            1: _state(policy_idx=1, pbt_ranking_score=50.0,
                      nsucc_mean=0.0, nsucc_max=0.0, checkpoint=ck),
            2: _state(policy_idx=2, pbt_ranking_score=10.0,
                      nsucc_mean=0.0, nsucc_max=0.0, checkpoint=ck),
        }
        _seed_bottom_skill_delta_for_ternary(
            obs, population, env_frames=300_000_000, target_idx=2
        )
        with self._patched(obs):
            self._call_ternary(obs, population, env_frames=300_000_000)
        self.assertEqual(obs._cross_restart_count, 0)
        self.assertEqual(obs._recovery_restart_count, 0)
        self.assertEqual(obs._source_tier_counts["skipped"], 1)
        self.assertEqual(obs._last_source_tier_label, "skipped")
        self.assertEqual(obs._last_restart_type, 3)


class ResumeHookTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.workspace = Path(self._tmp.name)

    def test_defaults_when_no_resume(self):
        obs = _make_observer(self.workspace)
        self.assertEqual(obs._recovery_restart_count, 0)
        self.assertLess(obs._last_recovery_frame, -10**12)

    def test_resume_args_take_effect(self):
        obs = _make_observer(
            self.workspace,
            recovery_restart_count_resume=3,
            last_recovery_frame_resume=500_000_000,
        )
        self.assertEqual(obs._recovery_restart_count, 3)
        self.assertEqual(obs._last_recovery_frame, 500_000_000)


if __name__ == "__main__":
    unittest.main()
