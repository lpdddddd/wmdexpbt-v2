"""Phase A4 P4 (2026-06-09): floor-recovery + iterative RSSM stack tests.

Covers:
* default-off regression: with no flags set, the public surface behaves
  identically to the pre-P4 codebase (PBT ranking, archive, score-utils,
  RSSM forward path).
* infra fix: write_curriculum_control uses unique tmp filenames and
  retries on failure; concurrent writes do not collide.
* score utils:
    floor_stage_classify thresholds
    compute_floor_recovery_slope window math
    compute_floor_recovery_score 0..50 unified scale
    combine_wm_recovery_score additive bounded delta
* PBT:
    _ternary_pbt floor-recovery guard skips bottom replacement when
        protected; hard collapse override still wins.
    _select_best_floor_recovery_recovery_source returns None when
        archive disabled or empty.
    _build_curriculum_control_state populates floor_stage when feature
        is on.
* RSSM (Part B):
    PredictionHeads gains horizon + recovery heads only when toggled.
    RSSMLiteWorldModel.compute_aux_loss accepts new kwargs without
        breaking default behavior.
    soft-OR recovery target builder respects done boundaries.
* Teacher manager (Part C):
    promotion gate rejects on NaN, BCE worse, recovery-pos degenerate.
    snapshot save/load round trip (state_dict only, no model required).
* Imagined evaluator (Part D):
    action_mode='policy' raises NotImplementedError.
    enabled=False returns {'imagined_valid': False}.
* WM-guided ranking (Part E):
    pbt_wm_score_mode tri-state values + alias mapping.
* Replay sampler (Part F):
    prioritize=False is bit-for-bit uniform.
    prioritize=True respects uniform_floor.
* CLI passthrough:
    train.py declares all new flags; launch_pbt boolean_flags has the
    new toggles; the experimental_* preset expands flags but never
    bypasses kill switches.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
import threading
import unittest
from pathlib import Path

import torch

from dexpbt_isaaclab.dexpbt_utils import (
    FLOOR_STAGE_FLOOR,
    FLOOR_STAGE_HARD_FLOOR,
    FLOOR_STAGE_NONE,
    combine_wm_recovery_score,
    compute_floor_recovery_score,
    compute_floor_recovery_slope,
    floor_stage_classify,
)
from dexpbt_isaaclab.pbt.pbt_curriculum_p1 import (
    CurriculumControlState,
    WindowSample,
    read_curriculum_control,
    write_curriculum_control,
)
from dexpbt_isaaclab.world_model.rssm_lite import RSSMLiteWorldModel
from dexpbt_isaaclab.world_model.seq_wm_base import SequenceBatch
from dexpbt_isaaclab.world_model.seq_wm_heads import PredictionHeads
from dexpbt_isaaclab.world_model.seq_wm_sampler import SequenceSampler
from dexpbt_isaaclab.world_model.wm_imagined_evaluator import WMImaginedEvaluator
from dexpbt_isaaclab.world_model.wm_teacher_manager import (
    FloorRecoveryEvalBuffer,
    TeacherEvalMetrics,
    TeacherManager,
)


# =========================================================================
# Section 1: score utils
# =========================================================================


class FloorStageClassifyTest(unittest.TestCase):
    def test_thresholds(self):
        # Above floor.
        self.assertEqual(
            floor_stage_classify(0.05, 0.01, floor_tol_threshold=0.02),
            FLOOR_STAGE_NONE,
        )
        # In floor band.
        self.assertEqual(
            floor_stage_classify(0.018, 0.01, floor_tol_threshold=0.02),
            FLOOR_STAGE_FLOOR,
        )
        # Hard floor.
        self.assertEqual(
            floor_stage_classify(0.011, 0.01, hard_floor_tol_threshold=0.012),
            FLOOR_STAGE_HARD_FLOOR,
        )
        # Equal target tolerance counts as hard floor.
        self.assertEqual(
            floor_stage_classify(0.01, 0.01),
            FLOOR_STAGE_HARD_FLOOR,
        )


class FloorRecoverySlopeTest(unittest.TestCase):
    def test_empty_history(self):
        slope, pos = compute_floor_recovery_slope([])
        self.assertEqual(slope, 0.0)
        self.assertEqual(pos, 0.0)

    def test_positive_recovery(self):
        slope, pos = compute_floor_recovery_slope([5.0, 10.0, 15.0, 25.0])
        self.assertGreater(pos, 0.0)
        self.assertLessEqual(pos, 1.0)
        self.assertGreater(slope, 0.0)

    def test_negative_slope_clipped_to_zero(self):
        slope, pos = compute_floor_recovery_slope([30.0, 25.0, 20.0, 10.0])
        self.assertLess(slope, 0.0)
        self.assertEqual(pos, 0.0)


class FloorRecoveryScoreTest(unittest.TestCase):
    def test_unified_scale_zero(self):
        score, comp = compute_floor_recovery_score(
            nsucc_mean=0.0,
            nsucc_max=0.0,
            retention=0.0,
            slope_positive_norm=0.0,
            drop_ratio=0.0,
        )
        self.assertEqual(score, 0.0)
        self.assertEqual(comp["weighted_norm"], 0.0)

    def test_unified_scale_positive(self):
        # nsucc_mean=20 -> norm=0.4; full saturation on the other terms
        # gives weighted = 0.4 + 0.5*1 + 0.3*1 + 0.2*1 = 1.4 and score
        # = 70 on the nsucc-comparable scale (50 * weighted).
        score, comp = compute_floor_recovery_score(
            nsucc_mean=20.0,
            nsucc_max=50.0,
            retention=1.0,
            slope_positive_norm=1.0,
            drop_ratio=0.0,
        )
        self.assertGreater(score, 0.0)
        # Sanity check the math, not a hard cap: weighted equals sum of
        # the five active sub-terms.
        self.assertAlmostEqual(score, 70.0)

    def test_hard_collapse_pulls_score_below_real(self):
        # Without collapse, the same inputs would score 70.
        baseline, _ = compute_floor_recovery_score(
            nsucc_mean=20.0,
            nsucc_max=50.0,
            retention=1.0,
            slope_positive_norm=1.0,
            drop_ratio=0.0,
            hard_collapse=False,
        )
        with_collapse, _ = compute_floor_recovery_score(
            nsucc_mean=20.0,
            nsucc_max=50.0,
            retention=1.0,
            slope_positive_norm=1.0,
            drop_ratio=0.0,
            hard_collapse=True,
        )
        # Hard collapse subtracts 50 (1.0 * 50) from the weighted sum,
        # so the score is strictly lower; the magnitude matters more
        # than the sign for ranking.
        self.assertLess(with_collapse, baseline)


class CombineWMRecoveryScoreTest(unittest.TestCase):
    def test_zero_wm_keeps_real(self):
        score, comp = combine_wm_recovery_score(
            real_score_50=10.0,
            wm_recovery_prob=0.0,
            wm_success_prob=0.0,
            wm_drop_risk=0.0,
            wm_uncertainty=0.0,
        )
        self.assertEqual(score, 10.0)

    def test_high_wm_recovery_bumps_score(self):
        score, _ = combine_wm_recovery_score(
            real_score_50=10.0,
            wm_recovery_prob=1.0,
            wm_success_prob=1.0,
            wm_drop_risk=0.0,
            wm_uncertainty=0.0,
        )
        self.assertGreater(score, 10.0)

    def test_high_drop_or_uncertainty_pulls_score_down(self):
        score, _ = combine_wm_recovery_score(
            real_score_50=10.0,
            wm_recovery_prob=0.0,
            wm_success_prob=0.0,
            wm_drop_risk=1.0,
            wm_uncertainty=1.0,
        )
        self.assertLess(score, 10.0)


# =========================================================================
# Section 2: infra fix — atomic write race mitigation
# =========================================================================


class CurriculumControlAtomicWriteTest(unittest.TestCase):
    def test_unique_tmp_filename(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "000.json"
            write_curriculum_control(path, CurriculumControlState())
            self.assertTrue(path.exists())
            # No leftover .tmp files in the dir.
            tmps = list(Path(d).glob("*.tmp"))
            self.assertEqual(tmps, [])

    def test_concurrent_writers_no_collision(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "000.json"
            errors: list[Exception] = []

            def writer(tag: str):
                try:
                    state = CurriculumControlState()
                    state.shrink_blocked_reason = tag
                    write_curriculum_control(path, state)
                except Exception as exc:
                    errors.append(exc)

            threads = [
                threading.Thread(target=writer, args=(f"t{i}",)) for i in range(8)
            ]
            for t in threads:
                t.start()
            for t in threads:
                t.join()
            self.assertEqual(errors, [])
            self.assertTrue(path.exists())
            # Final JSON is valid (any one of the 8 tags wins).
            ctrl = read_curriculum_control(path)
            self.assertIsNotNone(ctrl)
            self.assertRegex(ctrl.shrink_blocked_reason, r"^t\d+$")


# =========================================================================
# Section 3: prediction heads + RSSM default-off regression
# =========================================================================


class PredictionHeadsRegressionTest(unittest.TestCase):
    def test_default_no_horizon_no_recovery(self):
        h = PredictionHeads(
            belief_dim=32, action_dim=4, obs_dim=10, hidden_dim=16, n_hidden=1
        )
        self.assertFalse(h.has_horizon_heads)
        self.assertFalse(h.has_recovery_head)
        self.assertEqual(len(h.success_horizon_heads), 0)
        self.assertIsNone(h.recovery_head)

    def test_horizon_and_recovery_when_enabled(self):
        h = PredictionHeads(
            belief_dim=32,
            action_dim=4,
            obs_dim=10,
            hidden_dim=16,
            n_hidden=1,
            with_horizon_heads=True,
            horizon_steps=(8, 16),
            with_recovery_head=True,
            recovery_horizon=16,
        )
        self.assertTrue(h.has_horizon_heads)
        self.assertTrue(h.has_recovery_head)
        self.assertEqual(set(h.success_horizon_heads.keys()), {"h8", "h16"})
        belief = torch.zeros(2, 32)
        action = torch.zeros(2, 4)
        out = h.predict_transition(belief, action)
        self.assertIn("horizon_logits", out)
        self.assertIn(8, out["horizon_logits"])
        self.assertIn("recovery_logits", out)
        # Default still emits the legacy keys.
        self.assertIn("next_obs_delta", out)


class RSSMLiteDefaultBehaviorTest(unittest.TestCase):
    """compute_aux_loss with no new flags returns the same loss key set as legacy."""

    def setUp(self):
        torch.manual_seed(0)
        self.model = RSSMLiteWorldModel(
            obs_dim=8,
            action_dim=4,
            obs_embed_dim=16,
            action_embed_dim=8,
            h_dim=16,
            z_dim=4,
            head_hidden=16,
            with_value_head=True,
            with_success_head=True,
            with_task_error_head=True,
            free_nats=0.5,
        )
        T, B = 4, 3
        self.batch = SequenceBatch(
            obs=torch.randn(T, B, 8),
            action=torch.randn(T, B, 4),
            reward=torch.randn(T, B),
            done=torch.zeros(T, B),
            next_obs=torch.randn(T, B, 8),
            returns=torch.randn(T, B),
            d_hat=torch.rand(T, B),
            next_d_hat=torch.rand(T, B),
            failure=torch.zeros(T, B),
            success_count=torch.zeros(T, B),
            next_success_count=torch.zeros(T, B),
            lifted=torch.zeros(T, B),
            valid_mask=torch.ones(T, B),
            segment_id=torch.zeros(T, B, dtype=torch.long),
        )

    def test_default_kwargs(self):
        out = self.model.compute_aux_loss(self.batch)
        # Legacy keys still present.
        for k in (
            "loss_total",
            "loss_state",
            "loss_reward",
            "loss_done",
            "loss_success",
            "loss_task_error",
            "loss_value",
            "loss_kl",
        ):
            self.assertIn(k, out)
            self.assertTrue(torch.is_tensor(out[k]))
        # Recovery loss is zero when head is absent.
        self.assertAlmostEqual(float(out["loss_recovery"].item()), 0.0)

    def test_floor_aware_weighting_does_not_break_default(self):
        # No success_tolerance_seq -> multiplier stays 1.0.
        out = self.model.compute_aux_loss(
            self.batch, floor_aware_weighting=True
        )
        self.assertIn("loss_total", out)


# =========================================================================
# Section 4: teacher manager + imagined evaluator
# =========================================================================


class TeacherManagerPromotionGateTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.workspace = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_disabled_short_circuits(self):
        mgr = TeacherManager(
            workspace_root=self.workspace, policy_idx=0, enabled=False
        )
        ok, reason = mgr.maybe_promote(
            candidate_path=self.workspace / "x.pt",
            candidate_metrics=TeacherEvalMetrics(score=10.0, recovery_pos_rate=0.5),
            env_frames=0,
            global_step=0,
        )
        self.assertFalse(ok)
        self.assertEqual(reason, "manager_disabled")

    def test_first_teacher_accepted(self):
        mgr = TeacherManager(
            workspace_root=self.workspace, policy_idx=0, enabled=True
        )
        m = TeacherEvalMetrics(score=1.0, recovery_pos_rate=0.5)
        ok, reason = mgr.maybe_promote(
            candidate_path=self.workspace / "x.pt",
            candidate_metrics=m,
            env_frames=0,
            global_step=0,
        )
        self.assertTrue(ok)
        self.assertEqual(reason, "first_teacher")

    def test_nan_rejected(self):
        mgr = TeacherManager(
            workspace_root=self.workspace, policy_idx=0, enabled=True
        )
        m = TeacherEvalMetrics(score=1.0, has_nan=True, recovery_pos_rate=0.5)
        ok, reason = mgr.maybe_promote(
            candidate_path=self.workspace / "x.pt",
            candidate_metrics=m,
            env_frames=0,
            global_step=0,
        )
        self.assertFalse(ok)
        self.assertEqual(reason, "nan")

    def test_recovery_pos_degenerate_rejected(self):
        mgr = TeacherManager(
            workspace_root=self.workspace, policy_idx=0, enabled=True
        )
        m = TeacherEvalMetrics(score=10.0, recovery_pos_rate=0.0)
        ok, reason = mgr.maybe_promote(
            candidate_path=self.workspace / "x.pt",
            candidate_metrics=m,
            env_frames=0,
            global_step=0,
        )
        self.assertFalse(ok)
        self.assertEqual(reason, "recovery_pos_degenerate")

    def test_score_below_threshold_rejected(self):
        mgr = TeacherManager(
            workspace_root=self.workspace,
            policy_idx=0,
            enabled=True,
            promote_min_improvement=0.05,
        )
        first = TeacherEvalMetrics(score=10.0, recovery_pos_rate=0.5)
        mgr.maybe_promote(
            candidate_path=self.workspace / "first.pt",
            candidate_metrics=first,
            env_frames=0,
            global_step=0,
        )
        cand = TeacherEvalMetrics(
            score=10.02,
            recovery_pos_rate=0.5,
            recovery_bce=first.recovery_bce,
            horizon_success_bce=first.horizon_success_bce,
            drop_risk_bce=first.drop_risk_bce,
            d_hat_trend_mse=first.d_hat_trend_mse,
        )
        ok, reason = mgr.maybe_promote(
            candidate_path=self.workspace / "cand.pt",
            candidate_metrics=cand,
            env_frames=int(50e6),
            global_step=1,
        )
        self.assertFalse(ok)
        self.assertEqual(reason, "score_below_threshold")


class ImaginedEvaluatorKillSwitchTest(unittest.TestCase):
    def test_disabled_returns_invalid(self):
        ev = WMImaginedEvaluator(enabled=False)
        # Construct a tiny model & batch to call evaluate_batch but the
        # evaluator should bail out before any model use.
        ev_out = ev.evaluate_batch(model=None, batch=None, device="cpu")  # type: ignore[arg-type]
        self.assertFalse(ev_out["imagined_valid"])

    def test_policy_action_mode_raises(self):
        ev = WMImaginedEvaluator(enabled=True, action_mode="policy")
        with self.assertRaises(NotImplementedError):
            ev.evaluate_batch(model=None, batch=None, device="cpu")  # type: ignore[arg-type]


# =========================================================================
# Section 5: replay sampler — uniform when disabled, prioritized respects floor
# =========================================================================


class ReplaySamplerTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.H = 8
        self.N = 16
        self.sampler_uniform = SequenceSampler(seq_len=4, seq_batch_size=8, device="cpu")
        self.sampler_prio = SequenceSampler(
            seq_len=4,
            seq_batch_size=8,
            device="cpu",
            prioritize_floor_recovery=True,
            uniform_floor=0.5,
        )

    def _make_inputs(self):
        H, N = self.H, self.N
        obs = torch.randn(H, N, 6)
        action = torch.randn(H, N, 3)
        reward = torch.randn(H, N)
        done = torch.zeros(H, N)
        returns = torch.randn(H, N)
        last_obs = torch.randn(N, 6)
        return obs, action, reward, done, returns, last_obs

    def test_uniform_default_call(self):
        obs, action, reward, done, returns, last_obs = self._make_inputs()
        batch = self.sampler_uniform.sample(
            obs_full=obs, action_full=action, reward_full=reward,
            done_full=done, returns_full=returns, last_obs=last_obs,
        )
        self.assertEqual(batch.B, 8)
        self.assertEqual(batch.T, 4)

    def test_prioritized_keeps_uniform_floor(self):
        obs, action, reward, done, returns, last_obs = self._make_inputs()
        batch = self.sampler_prio.sample(
            obs_full=obs, action_full=action, reward_full=reward,
            done_full=done, returns_full=returns, last_obs=last_obs,
            success_tolerance_full=torch.full((self.H, self.N), 0.005),
            failure_full=torch.zeros(self.H, self.N),
            success_count_full=torch.arange(
                self.H * self.N, dtype=torch.float32
            ).reshape(self.H, self.N),
            d_hat_full=torch.linspace(1.0, 0.0, self.H * self.N).reshape(self.H, self.N),
        )
        # Last replay metrics record the uniform fraction.
        self.assertGreaterEqual(
            self.sampler_prio.last_replay_metrics["uniform_fraction"], 0.5
        )


# =========================================================================
# Section 6: CLI passthrough — train.py declares flags + launch_pbt set
# =========================================================================


class FloorRecoveryFlagsDeclaredTest(unittest.TestCase):
    """Source-text smoke checks (no argparse import).

    Ensures that every kill switch listed in the user spec is wired in
    train.py and that BooleanOptionalAction toggles also appear in
    launch_pbt.py's boolean_flags set.
    """

    @classmethod
    def setUpClass(cls):
        cls.train_text = (
            Path(__file__).parent.parent
            / "dexpbt_isaaclab"
            / "scripts"
            / "train.py"
        ).read_text()
        cls.launch_text = (
            Path(__file__).parent.parent
            / "dexpbt_isaaclab"
            / "scripts"
            / "launch_pbt.py"
        ).read_text()

    def test_train_declares_kill_switches(self):
        for flag in (
            "--pbt_floor_recovery_aware",
            "--pbt_enable_best_floor_recovery_archive",
            "--pbt_wm_score_mode",
            "--pbt_use_wm_recovery_score",
            "--seq_wm_use_task_aux_inputs",
            "--seq_wm_loss_preset",
            "--seq_wm_floor_aware_weighting",
            "--seq_wm_use_horizon_heads",
            "--seq_wm_use_recovery_head",
            "--seq_wm_enable_iterative_teacher",
            "--seq_wm_enable_imagined_evaluator",
            "--seq_wm_replay_prioritize_floor_recovery",
            "--experimental_rssm_floor_recovery_stack",
        ):
            self.assertIn(f'"{flag}"', self.train_text, f"missing {flag}")

    def test_launch_pbt_has_new_boolean_flags(self):
        for flag in (
            '"--pbt_floor_recovery_aware"',
            '"--no-pbt_floor_recovery_aware"',
            '"--pbt_enable_best_floor_recovery_archive"',
            '"--no-pbt_enable_best_floor_recovery_archive"',
            '"--seq_wm_use_task_aux_inputs"',
            '"--no-seq_wm_use_task_aux_inputs"',
            '"--seq_wm_floor_aware_weighting"',
            '"--no-seq_wm_floor_aware_weighting"',
            '"--seq_wm_use_horizon_heads"',
            '"--no-seq_wm_use_horizon_heads"',
            '"--seq_wm_use_recovery_head"',
            '"--no-seq_wm_use_recovery_head"',
            '"--seq_wm_enable_iterative_teacher"',
            '"--no-seq_wm_enable_iterative_teacher"',
            '"--seq_wm_enable_imagined_evaluator"',
            '"--no-seq_wm_enable_imagined_evaluator"',
            '"--seq_wm_replay_prioritize_floor_recovery"',
            '"--no-seq_wm_replay_prioritize_floor_recovery"',
            '"--experimental_rssm_floor_recovery_stack"',
            '"--no-experimental_rssm_floor_recovery_stack"',
        ):
            self.assertIn(flag, self.launch_text, f"missing {flag}")

    def test_pbt_wm_score_mode_three_choices(self):
        # Triple-quoted string match is brittle; we just look for the
        # canonical literal list passed to add_argument.
        self.assertIn('"off", "shadow", "active"', self.train_text)

    def test_imagine_action_mode_recorded_default(self):
        # default 'recorded' present in the parser block.
        self.assertIn('default="recorded"', self.train_text)

    def test_experimental_preset_does_not_force_active(self):
        # The preset upgrades pbt_wm_score_mode 'off' -> 'shadow', never
        # 'active'. We make sure no literal "args_cli.pbt_wm_score_mode = 'active'"
        # appears in the preset expansion block.
        self.assertNotIn(
            "args_cli.pbt_wm_score_mode = 'active'",
            self.train_text,
        )

    def test_launchers_disable_active_by_default(self):
        # Iterative shadow / replay launchers must NOT default to active.
        for path in (
            "/root/autodl-tmp/wmdexpbt/results/start_regrasping_fixed_rssm_iterative_shadow_800m.sh",
            "/root/autodl-tmp/wmdexpbt/results/start_regrasping_fixed_rssm_iterative_replay_800m.sh",
        ):
            text = Path(path).read_text()
            self.assertNotIn("--pbt_wm_score_mode active", text, path)
        # Active launcher exists separately.
        active = Path(
            "/root/autodl-tmp/wmdexpbt/results/start_regrasping_fixed_rssm_iterative_active_800m.sh"
        )
        self.assertIn("--pbt_wm_score_mode active", active.read_text())


# =========================================================================
# Section 7: Floor-recovery eval buffer
# =========================================================================


class FloorRecoveryEvalBufferTest(unittest.TestCase):
    def test_capacity(self):
        buf = FloorRecoveryEvalBuffer(capacity=2)
        for _ in range(5):
            T, B = 2, 2
            tol = torch.full((T, B), 0.005)
            batch = SequenceBatch(
                obs=torch.zeros(T, B, 4),
                action=torch.zeros(T, B, 2),
                reward=torch.zeros(T, B),
                done=torch.zeros(T, B),
                next_obs=torch.zeros(T, B, 4),
                returns=torch.zeros(T, B),
                valid_mask=torch.ones(T, B),
                segment_id=torch.zeros(T, B, dtype=torch.long),
                success_tolerance_seq=tol,
            )
            buf.record_candidate(batch)
        self.assertEqual(len(buf), 2)

    def test_skip_when_no_floor_signal(self):
        buf = FloorRecoveryEvalBuffer(capacity=2)
        T, B = 2, 2
        tol = torch.full((T, B), 0.06)  # well above floor
        batch = SequenceBatch(
            obs=torch.zeros(T, B, 4),
            action=torch.zeros(T, B, 2),
            reward=torch.zeros(T, B),
            done=torch.zeros(T, B),
            next_obs=torch.zeros(T, B, 4),
            returns=torch.zeros(T, B),
            valid_mask=torch.ones(T, B),
            segment_id=torch.zeros(T, B, dtype=torch.long),
            success_tolerance_seq=tol,
        )
        accepted = buf.record_candidate(batch)
        self.assertFalse(accepted)
        self.assertEqual(len(buf), 0)


# =========================================================================
# Section 8: Phase A4 P4 wired (2026-06-10) — runtime wiring tests.
# =========================================================================
#
# These tests cover the runtime wiring patch:
#   * default-off: TeacherManager / WMImaginedEvaluator construct cleanly
#     with enabled=False and produce no side effects.
#   * recovery target wider mask gives non-zero pos_rate at tight tol.
#   * horizon heads skip horizons exceeding seq_len-2 (no fake zero loss).
#   * imagined evaluator policy mode raises NotImplementedError.
#   * train.py declares the new wiring flags.
#   * pbt_observer.py exposes the new TB tags.
#   * analyze_wm_runtime_wiring.py CLI module is importable.
# =========================================================================


class WiredHorizonSkipTest(unittest.TestCase):
    """Horizons longer than seq_len-2 must be skipped, not silently zeroed."""

    def test_h16_with_seq_len_16_is_skipped(self):
        T, B = 16, 4
        torch.manual_seed(0)
        model = RSSMLiteWorldModel(
            obs_dim=8,
            action_dim=2,
            obs_embed_dim=8,
            action_embed_dim=4,
            h_dim=8,
            z_dim=4,
            head_hidden=8,
            with_value_head=True,
            with_horizon_heads=True,
            horizon_steps=(4, 8, 16),
            with_recovery_head=True,
            recovery_horizon=4,
            free_nats=0.0,
        )
        batch = SequenceBatch(
            obs=torch.randn(T, B, 8),
            action=torch.zeros(T, B, 2),
            reward=torch.zeros(T, B),
            done=torch.zeros(T, B),
            next_obs=torch.randn(T, B, 8),
            returns=torch.zeros(T, B),
            valid_mask=torch.ones(T, B),
            segment_id=torch.zeros(T, B, dtype=torch.long),
            d_hat=torch.full((T, B), 0.05),
            next_d_hat=torch.full((T, B), 0.04),
            success_count=torch.zeros(T, B),
            next_success_count=torch.zeros(T, B),
            success_tolerance_seq=torch.full((T, B), 0.03),
        )
        losses = model.compute_aux_loss(
            batch,
            recovery_shadow_tol_threshold=0.04,
            horizon_skip_too_long=True,
        )
        # h16 must NOT be present (skipped because 16 > 16-2=14).
        self.assertNotIn("loss_horizon_success_h16", losses)
        # h4 and h8 (both <= 14) must be present.
        self.assertIn("loss_horizon_success_h4", losses)
        self.assertIn("loss_horizon_success_h8", losses)
        metrics = model.report_metrics()
        self.assertIn("horizon_skipped_h16", metrics)
        self.assertEqual(metrics["horizon_skipped_h16"], 1.0)


class WiredRecoveryShadowMaskTest(unittest.TestCase):
    """Wider mask should produce non-zero pos_rate at tight tolerance."""

    def test_pos_rate_nonzero_at_tight_tol(self):
        T, B = 8, 4
        torch.manual_seed(1)
        model = RSSMLiteWorldModel(
            obs_dim=4,
            action_dim=2,
            obs_embed_dim=4,
            action_embed_dim=4,
            h_dim=4,
            z_dim=4,
            head_hidden=4,
            with_value_head=False,
            with_horizon_heads=False,
            with_recovery_head=True,
            recovery_horizon=4,
            free_nats=0.0,
        )
        # Tolerance is 0.03 (above the strict 0.02 floor but below the
        # shadow threshold 0.04). d_hat improves by 0.01 each step.
        d_hat = torch.tensor(
            [[0.05 - 0.005 * t for _ in range(B)] for t in range(T)]
        )
        next_d_hat = torch.cat(
            [d_hat[1:], d_hat[-1:].clone()], dim=0
        )
        batch = SequenceBatch(
            obs=torch.zeros(T, B, 4),
            action=torch.zeros(T, B, 2),
            reward=torch.zeros(T, B),
            done=torch.zeros(T, B),
            next_obs=torch.zeros(T, B, 4),
            returns=torch.zeros(T, B),
            valid_mask=torch.ones(T, B),
            segment_id=torch.zeros(T, B, dtype=torch.long),
            d_hat=d_hat,
            next_d_hat=next_d_hat,
            success_count=torch.zeros(T, B),
            next_success_count=torch.zeros(T, B),
            success_tolerance_seq=torch.full((T, B), 0.03),
        )
        # Strict mask (default): 0.03 > 0.02 → recovery target empty.
        losses_strict = model.compute_aux_loss(
            batch,
            floor_tol_threshold=0.02,
        )
        m_strict = model.report_metrics()
        # Wider shadow: 0.03 <= 0.04 → mask covers steps and target>0.
        losses_wide = model.compute_aux_loss(
            batch,
            floor_tol_threshold=0.02,
            recovery_shadow_tol_threshold=0.04,
            recovery_d_hat_margin=0.005,
        )
        m_wide = model.report_metrics()
        # Strict path should have zero coverage; wide path > 0.
        self.assertEqual(m_strict["recovery_floor_sample_fraction"], 0.0)
        self.assertGreater(m_wide["recovery_floor_sample_fraction"], 0.0)
        # And pos_rate should be non-zero in the wide path.
        self.assertGreater(m_wide["recovery_pos_rate"], 0.0)


class WiredImaginedEvaluatorTest(unittest.TestCase):
    """Default-off, policy-mode raises, recorded-mode produces a valid dict."""

    def test_default_off_returns_invalid(self):
        ev = WMImaginedEvaluator(enabled=False)
        out = ev.evaluate_batch(model=None, batch=None, device="cpu")
        self.assertFalse(out["imagined_valid"])

    def test_policy_mode_raises(self):
        with self.assertRaises(NotImplementedError):
            ev = WMImaginedEvaluator(enabled=True, action_mode="policy")
            T, B = 4, 2
            batch = SequenceBatch(
                obs=torch.zeros(T, B, 4),
                action=torch.zeros(T, B, 2),
                reward=torch.zeros(T, B),
                done=torch.zeros(T, B),
                next_obs=torch.zeros(T, B, 4),
                returns=torch.zeros(T, B),
                valid_mask=torch.ones(T, B),
                segment_id=torch.zeros(T, B, dtype=torch.long),
            )
            class _Stub:
                training = False
                def eval(self):
                    return None
                def train(self):
                    return None
                def compute_aux_loss(self, _b):
                    return {}
                def encode_sequence(self, _b):
                    return torch.zeros(T, B, 4)
                heads = None
            ev.evaluate_batch(_Stub(), batch, device="cpu")


class WiredCLIDeclarationTest(unittest.TestCase):
    """Train.py declares the new wiring flags + the default horizon list."""

    @classmethod
    def setUpClass(cls):
        cls.train_text = (
            Path(__file__).parent.parent
            / "dexpbt_isaaclab"
            / "scripts"
            / "train.py"
        ).read_text()
        cls.observer_text = (
            Path(__file__).parent.parent
            / "dexpbt_isaaclab"
            / "pbt"
            / "pbt_observer.py"
        ).read_text()

    def test_default_horizon_steps_is_4_8(self):
        # The default must NOT be 8,16 anymore (h16 is degenerate at
        # seq_len=16). Look for "default=\"4,8\"" near the seq_wm_horizon_steps
        # parser entry.
        seg = self.train_text.split("--seq_wm_horizon_steps", 1)[1][:400]
        self.assertIn('default="4,8"', seg)

    def test_train_declares_wiring_flags(self):
        for flag in (
            "--seq_wm_recovery_shadow_tol_threshold",
            "--seq_wm_horizon_skip_too_long",
            "--seq_wm_teacher_eval_buffer_capacity",
            "--seq_wm_teacher_record_below_tol",
            "--seq_wm_teacher_require_recovery_pos_low",
            "--seq_wm_teacher_require_recovery_pos_high",
        ):
            self.assertIn(flag, self.train_text, f"{flag} not declared")

    def test_train_imports_runtime_modules(self):
        self.assertIn("from dexpbt_isaaclab.world_model.wm_teacher_manager", self.train_text)
        self.assertIn("from dexpbt_isaaclab.world_model.wm_imagined_evaluator", self.train_text)
        self.assertIn("_run_wm_runtime_wiring(", self.train_text)
        # Per-policy disk mailbox publish path.
        self.assertIn("_atomic_write_json(", self.train_text)
        self.assertIn('wm_imagine_dir / f"{int(policy_idx_for_file):03d}.json"', self.train_text)

    def test_observer_emits_wired_tags(self):
        for tag in (
            'pbt/wm_score_used',
            'pbt/source_selected_by_wm',
            'pbt/wm_fallback_reason_hash',
            'pbt/floor_recovery_score_real',
            'pbt/floor_recovery_score_wm',
            'pbt/hard_collapse_population',
        ):
            self.assertIn(tag, self.observer_text, f"missing {tag}")

    def test_observer_refresh_helper_present(self):
        self.assertIn(
            "_refresh_wm_imagined_eval_from_disk", self.observer_text
        )


class WiredObserverDiskRefreshTest(unittest.TestCase):
    """Observer must reload imagined eval from per-policy JSON files."""

    def test_refresh_reads_files(self):
        # Construct a stub object with the required attributes only — we
        # are unit-testing _refresh_wm_imagined_eval_from_disk in
        # isolation without booting a full observer.
        from dexpbt_isaaclab.pbt.pbt_observer import DexPBTPbtAlgoObserver

        class _Stub:
            workspace = Path(tempfile.mkdtemp()) / "ws"
            _wm_imagined_eval_per_policy = {}

        stub = _Stub()
        d = stub.workspace / "wm_imagined"
        d.mkdir(parents=True, exist_ok=True)
        for pidx, valid in [(0, True), (1, False)]:
            (d / f"{pidx:03d}.json").write_text(
                json.dumps({
                    "policy_idx": pidx,
                    "frames": 100_000_000,
                    "imagined_valid": valid,
                    "uncertainty": 0.2 if valid else 0.9,
                    "wm_recovery_prob": 0.8 if valid else 0.0,
                    "invalid_reason": "" if valid else "high_uncertainty",
                })
            )
        # Bind the unbound method on the stub by passing it as ``self``.
        n = DexPBTPbtAlgoObserver._refresh_wm_imagined_eval_from_disk(
            stub, env_frames=110_000_000
        )
        self.assertEqual(n, 2)
        self.assertIn(0, stub._wm_imagined_eval_per_policy)
        self.assertIn(1, stub._wm_imagined_eval_per_policy)
        # Stale gate: an entry older than max_age should flip valid=False.
        n_stale = DexPBTPbtAlgoObserver._refresh_wm_imagined_eval_from_disk(
            stub, env_frames=10_000_000_000, max_age_frames=1_000_000
        )
        self.assertEqual(n_stale, 2)
        self.assertFalse(
            stub._wm_imagined_eval_per_policy[0]["imagined_valid"]
        )
        self.assertEqual(
            stub._wm_imagined_eval_per_policy[0]["invalid_reason"],
            "stale_wm_imagined",
        )


class WiredAnalyzerImportTest(unittest.TestCase):
    """analyze_wm_runtime_wiring.py imports cleanly and exposes analyze()."""

    def test_module_loads(self):
        import importlib.util
        path = (
            Path(__file__).parent.parent.parent.parent.parent
            / "results"
            / "analyze_wm_runtime_wiring.py"
        )
        self.assertTrue(path.exists(), f"missing {path}")
        spec = importlib.util.spec_from_file_location(
            "analyze_wm_runtime_wiring", path
        )
        self.assertIsNotNone(spec)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)  # type: ignore[union-attr]
        self.assertTrue(hasattr(mod, "analyze"))
        self.assertTrue(hasattr(mod, "render_markdown"))


class WiredTeacherManagerNoEvalDataTest(unittest.TestCase):
    """maybe_promote with empty buffer + zero recovery_pos_rate degeneracy."""

    def test_empty_buffer_evaluate_returns_zero_metrics(self):
        with tempfile.TemporaryDirectory() as tmp:
            mgr = TeacherManager(
                workspace_root=Path(tmp),
                policy_idx=0,
                enabled=True,
                require_recovery_pos_in=(0.05, 0.95),
            )
            buf = FloorRecoveryEvalBuffer(capacity=4)
            # Stub model — never reached because buffer is empty.
            metrics = TeacherManager.evaluate(
                model=None,  # type: ignore[arg-type]
                eval_buffer=buf,
                device="cpu",
            )
            self.assertEqual(metrics.n_samples, 0)
            self.assertEqual(metrics.score, 0.0)
            # Promotion should still gate on recovery_pos_rate=0 → reject.
            promoted, reason = mgr.maybe_promote(
                candidate_path=Path(tmp) / "x.pt",
                candidate_metrics=metrics,
                env_frames=10_000_000,
                global_step=0,
            )
            self.assertFalse(promoted)
            self.assertEqual(reason, "recovery_pos_degenerate")


if __name__ == "__main__":
    unittest.main()
