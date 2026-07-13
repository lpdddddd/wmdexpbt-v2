"""Unit tests for tie-aware calibration metrics and shared head extraction."""

from __future__ import annotations

import math
import random
import unittest

import torch

from dexpbt_isaaclab.world_model.snippet_wm_reranker import (
    SnippetWMRerankConfig,
    compute_snippet_wm_term,
    wm_head_probs_for_candidates,
)
from dexpbt_isaaclab.world_model.wm_calibration_evaluator import (
    WMCalibrationConfig,
    WMCalibrationEvaluator,
    WMCalibrationState,
    apply_calibration_outcome,
)
from dexpbt_isaaclab.world_model.wm_calibration_metrics import (
    auc_binary,
    spearman_rank,
)
from dexpbt_isaaclab.world_model.wm_head_probs import (
    combined_head_uncertainty,
    extract_required_wm_head_probs,
)


class TestTieAwareMetrics(unittest.TestCase):
    def test_spearman_average_ranks_for_ties(self):
        # x has ties at 1.0; average ranks should be used.
        x = [1.0, 1.0, 2.0, 3.0]
        y = [0.0, 1.0, 2.0, 3.0]
        rho = spearman_rank(x, y)
        # Hand: ranks_x = [1.5, 1.5, 3, 4], ranks_y = [1,2,3,4]
        rx = [1.5, 1.5, 3.0, 4.0]
        ry = [1.0, 2.0, 3.0, 4.0]
        mx, my = sum(rx) / 4, sum(ry) / 4
        num = sum((rx[i] - mx) * (ry[i] - my) for i in range(4))
        den = math.sqrt(sum((rx[i] - mx) ** 2 for i in range(4))) * math.sqrt(
            sum((ry[i] - my) ** 2 for i in range(4))
        )
        self.assertAlmostEqual(rho, num / den, places=6)

    def test_spearman_permutation_invariant(self):
        x = [0.1, 0.2, 0.2, 0.9, 0.5]
        y = [0.0, 1.0, 0.0, 1.0, 1.0]
        base = spearman_rank(x, y)
        idx = list(range(len(x)))
        random.seed(0)
        for _ in range(20):
            random.shuffle(idx)
            xp = [x[i] for i in idx]
            yp = [y[i] for i in idx]
            self.assertAlmostEqual(spearman_rank(xp, yp), base, places=6)

    def test_constant_spearman_zero(self):
        self.assertEqual(spearman_rank([1.0, 1.0, 1.0], [0.0, 1.0, 0.5]), 0.0)
        self.assertEqual(spearman_rank([0.0, 1.0, 0.5], [2.0, 2.0, 2.0]), 0.0)

    def test_auc_tied_predictions(self):
        # All predictions tied with both classes -> 0.5
        self.assertAlmostEqual(auc_binary([0.4, 0.4, 0.4, 0.4], [0, 1, 0, 1]), 0.5, places=6)
        # Classic separable
        self.assertGreater(auc_binary([0.1, 0.2, 0.8, 0.9], [0, 0, 1, 1]), 0.9)

    def test_auc_permutation_invariant(self):
        p = [0.1, 0.5, 0.5, 0.9, 0.2]
        y = [0.0, 1.0, 0.0, 1.0, 0.0]
        base = auc_binary(p, y)
        idx = list(range(len(p)))
        random.seed(1)
        for _ in range(20):
            random.shuffle(idx)
            self.assertAlmostEqual(
                auc_binary([p[i] for i in idx], [y[i] for i in idx]), base, places=6
            )

    def test_clipped_wm_term_rank_permutation_invariant(self):
        cfg = SnippetWMRerankConfig(score_cap=0.05, w_rec=0.3, w_success=0.2, w_drop=0.4, w_unc=0.1)
        p_rec = torch.tensor([0.9, 0.8, 0.1, 0.2, 0.95, 0.05])
        p_succ = torch.tensor([0.9, 0.7, 0.2, 0.1, 0.8, 0.15])
        p_drop = torch.tensor([0.1, 0.2, 0.8, 0.7, 0.05, 0.9])
        unc = combined_head_uncertainty(p_rec, p_succ, p_drop)
        pred = compute_snippet_wm_term(
            p_recovery=p_rec,
            p_success=p_succ,
            p_drop=p_drop,
            uncertainty=unc,
            w_rec=cfg.w_rec,
            w_success=cfg.w_success,
            w_drop=cfg.w_drop,
            w_unc=cfg.w_unc,
            score_cap=cfg.score_cap,
        )
        util = cfg.w_rec * torch.tensor([1.0, 1.0, 0.0, 0.0, 1.0, 0.0]) + cfg.w_success * torch.tensor(
            [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
        ) - cfg.w_drop * torch.tensor([0.0, 0.0, 1.0, 1.0, 0.0, 1.0])
        base = spearman_rank(pred.tolist(), util.tolist())
        idx = list(range(pred.numel()))
        random.seed(2)
        for _ in range(15):
            random.shuffle(idx)
            self.assertAlmostEqual(
                spearman_rank([pred[i].item() for i in idx], [util[i].item() for i in idx]),
                base,
                places=6,
            )


class TestHorizonAlignment(unittest.TestCase):
    def test_runtime_calibration_horizon_mean_identity(self):
        n = 4
        # Distinct logits so H4 vs H8 means differ from either alone.
        preds = {
            "recovery_logits": torch.linspace(-2, 2, n),
            "horizon_logits": {
                8: {
                    "success": torch.linspace(-1, 3, n),
                    "drop": torch.linspace(2, -2, n),
                },
                4: {
                    "success": torch.linspace(3, -1, n),
                    "drop": torch.linspace(-2, 2, n),
                },
            },
        }
        steps = (4, 8)
        cal = extract_required_wm_head_probs(preds, horizon_steps=steps, expected_batch=n)

        def _predict(belief_rep, cand_flat):
            m = belief_rep.shape[0]
            # Expand single-row style: reuse first n then tile — here B*K = n with B=n,K=1
            return {
                "recovery_logits": preds["recovery_logits"],
                "horizon_logits": {
                    4: {
                        "success": preds["horizon_logits"][4]["success"],
                        "drop": preds["horizon_logits"][4]["drop"],
                    },
                    8: {
                        "success": preds["horizon_logits"][8]["success"],
                        "drop": preds["horizon_logits"][8]["drop"],
                    },
                },
            }

        belief = torch.zeros(n, 3)
        acts = torch.zeros(n, 1, 2)
        p_rec, p_succ, p_drop, unc, valid, _ = wm_head_probs_for_candidates(
            belief_flat=belief,
            cand_first_actions=acts,
            predict_fn=_predict,
            horizon_steps=steps,
        )
        self.assertTrue(bool(valid.all()))
        self.assertTrue(torch.allclose(p_rec.reshape(-1), cal.p_recovery, atol=1e-5))
        self.assertTrue(torch.allclose(p_succ.reshape(-1), cal.p_success, atol=1e-5))
        self.assertTrue(torch.allclose(p_drop.reshape(-1), cal.p_drop, atol=1e-5))
        self.assertTrue(torch.allclose(unc.reshape(-1), cal.uncertainty, atol=1e-5))
        # Mean of 4 and 8 differs from horizon-4-only.
        only4 = extract_required_wm_head_probs(preds, horizon_steps=(4,), expected_batch=n)
        self.assertFalse(torch.allclose(cal.p_success, only4.p_success))
        cfg = SnippetWMRerankConfig()
        term_rt = compute_snippet_wm_term(
            p_recovery=p_rec.reshape(-1),
            p_success=p_succ.reshape(-1),
            p_drop=p_drop.reshape(-1),
            uncertainty=unc.reshape(-1),
            w_rec=cfg.w_rec,
            w_success=cfg.w_success,
            w_drop=cfg.w_drop,
            w_unc=cfg.w_unc,
            score_cap=cfg.score_cap,
        )
        term_cal = compute_snippet_wm_term(
            p_recovery=cal.p_recovery,
            p_success=cal.p_success,
            p_drop=cal.p_drop,
            uncertainty=cal.uncertainty,
            w_rec=cfg.w_rec,
            w_success=cfg.w_success,
            w_drop=cfg.w_drop,
            w_unc=cfg.w_unc,
            score_cap=cfg.score_cap,
        )
        self.assertTrue(torch.allclose(term_rt, term_cal, atol=1e-5))


class TestFailureAlphaTransitions(unittest.TestCase):
    def test_structural_clears_stale_alpha(self):
        cfg = WMCalibrationConfig(enabled=True, ema_beta=0.9, alpha_max=1.0)
        st = WMCalibrationState(alpha_wm=0.8, ema_raw=0.8, calibration_pass=True)
        st = apply_calibration_outcome(
            st,
            cfg=cfg,
            frames=100,
            metrics={"wm_cal/missing_head": 1.0},
            mode="structural",
        )
        self.assertEqual(st.alpha_wm, 0.0)
        self.assertEqual(st.ema_raw, 0.0)
        self.assertFalse(st.calibration_pass)
        self.assertEqual(st.last_eval_frame, 100)

    def test_insufficient_decays_ema(self):
        cfg = WMCalibrationConfig(enabled=True, ema_beta=0.9, alpha_max=1.0)
        st = WMCalibrationState(alpha_wm=0.8, ema_raw=0.8)
        st = apply_calibration_outcome(
            st,
            cfg=cfg,
            frames=200,
            metrics={},
            mode="insufficient",
        )
        self.assertAlmostEqual(st.ema_raw, 0.9 * 0.8, places=6)
        self.assertAlmostEqual(st.alpha_wm, 0.9 * 0.8, places=6)
        self.assertFalse(st.calibration_pass)

    def test_length_mismatch_no_silent_zero(self):
        ev = WMCalibrationEvaluator(WMCalibrationConfig(enabled=True))
        metrics = ev.evaluate_batch(
            {
                "p_recovery": [0.1, 0.9],
                "y_recovery": [0.0, 1.0],
                "p_success": [0.1],  # mismatch
                "y_success": [0.0, 1.0],
                "p_drop": [0.1, 0.2],
                "y_drop": [0.0, 1.0],
                "uncertainty": [0.1, 0.2],
            }
        )
        self.assertEqual(metrics["wm_cal/batch_invalid"], 1.0)
        self.assertEqual(metrics["wm_cal/rank_corr_combined"], 0.0)

    def test_tensor_input_no_ambiguous_truth(self):
        ev = WMCalibrationEvaluator(WMCalibrationConfig(enabled=True))
        metrics = ev.evaluate_batch(
            {
                "p_recovery": torch.tensor([0.1, 0.9, 0.2, 0.8]),
                "y_recovery": torch.tensor([0.0, 1.0, 0.0, 1.0]),
                "p_success": torch.tensor([0.2, 0.8, 0.3, 0.7]),
                "y_success": torch.tensor([0.0, 1.0, 0.0, 1.0]),
                "p_drop": torch.tensor([0.8, 0.1, 0.7, 0.2]),
                "y_drop": torch.tensor([1.0, 0.0, 1.0, 0.0]),
                "uncertainty": torch.tensor([0.1, 0.1, 0.1, 0.1]),
            }
        )
        self.assertEqual(metrics["wm_cal/batch_invalid"], 0.0)
        self.assertEqual(metrics["wm_cal/eval_valid"], 1.0)

    def test_structural_zeros_even_when_alpha_min_positive(self):
        cfg = WMCalibrationConfig(
            enabled=True, ema_beta=0.9, alpha_min=0.25, alpha_max=1.0
        )
        st = WMCalibrationState(alpha_wm=0.8, ema_raw=0.8, calibration_pass=True)
        st = apply_calibration_outcome(
            st,
            cfg=cfg,
            frames=50,
            metrics={"wm_cal/missing_head": 1.0, "wm_cal/failure_reason_id": 1.0},
            mode="structural",
        )
        self.assertEqual(st.alpha_wm, 0.0)
        self.assertEqual(st.ema_raw, 0.0)
        self.assertFalse(st.calibration_pass)
        self.assertEqual(st.last_metrics["wm_cal/alpha_wm"], 0.0)

    def test_illegal_evaluator_input_safe_fail(self):
        from dexpbt_isaaclab.world_model.wm_calibration_evaluator import _as_cpu_float1d
        from dexpbt_isaaclab.world_model.wm_head_probs import FAILURE_NONFINITE

        for bad in (
            ["x", 0.1],
            [object()],
            [float("nan"), 0.2],
            [float("inf"), 0.0],
            {"a": 1},
        ):
            t, ok = _as_cpu_float1d(bad)
            self.assertFalse(ok)
            self.assertEqual(int(t.numel()), 0)
        ev = WMCalibrationEvaluator(WMCalibrationConfig(enabled=True))
        metrics = ev.evaluate_batch(
            {
                "p_recovery": ["bad", 0.1],
                "y_recovery": [0.0, 1.0],
                "p_success": [0.1, 0.2],
                "y_success": [0.0, 1.0],
                "p_drop": [0.1, 0.2],
                "y_drop": [0.0, 1.0],
                "uncertainty": [0.1, 0.2],
            }
        )
        self.assertEqual(metrics["wm_cal/batch_invalid"], 1.0)
        self.assertEqual(metrics["wm_cal/failure_reason_id"], FAILURE_NONFINITE)


class TestUnifiedHeadTargets(unittest.TestCase):
    """Training labels == calibration labels (shared wm_head_targets)."""

    def _batch(
        self,
        *,
        t: int = 10,
        b: int = 2,
        success_bump_at: int | None = 5,
        fail_at: int | None = None,
        d0: float = 0.04,
        d_end: float = 0.02,
        tol: float = 0.015,
        done_at: int | None = None,
        drop_sources: bool = True,
        success_sources: bool = True,
    ):
        from dexpbt_isaaclab.world_model.seq_wm_base import SequenceBatch

        d_hat = torch.full((t, b), float(d0))
        d_hat[-1, :] = float(d_end)
        done = torch.zeros(t, b)
        valid = torch.ones(t, b)
        seg = torch.zeros(t, b, dtype=torch.long)
        if done_at is not None:
            done[done_at, :] = 1.0
            valid[done_at + 1 :, :] = 0.0
            seg[done_at + 1 :, :] = 1
        sc = torch.zeros(t, b) if success_sources else None
        nsc = torch.zeros(t, b) if success_sources else None
        if success_sources and success_bump_at is not None:
            nsc[success_bump_at:, :] = 1.0
        failure = torch.zeros(t, b) if drop_sources else None
        dropped = torch.zeros(t, b) if drop_sources else None
        if drop_sources and fail_at is not None:
            failure[fail_at, :] = 1.0
        return SequenceBatch(
            obs=torch.zeros(t, b, 4),
            action=torch.zeros(t, b, 3),
            reward=torch.zeros(t, b),
            done=done,
            next_obs=torch.zeros(t, b, 4),
            returns=torch.zeros(t, b),
            d_hat=d_hat,
            success_tolerance_seq=torch.full((t, b), float(tol)),
            valid_mask=valid,
            segment_id=seg,
            dropped=dropped,
            failure=failure,
            success_count=sc,
            next_success_count=nsc,
        )

    def test_success_target_elementwise_equal(self):
        from dexpbt_isaaclab.world_model.rssm_lite import RSSMLiteWorldModel
        from dexpbt_isaaclab.world_model.wm_calibration_metrics import (
            build_wm_calibration_targets,
        )
        from dexpbt_isaaclab.world_model.wm_head_targets import (
            build_horizon_success_target,
        )

        batch = self._batch(success_bump_at=3)
        train = RSSMLiteWorldModel._build_horizon_success_target(batch, 4)
        shared = build_horizon_success_target(batch, 4)
        self.assertTrue(torch.equal(train, shared))
        tgt, _ = build_wm_calibration_targets(
            batch,
            horizon_steps=(4,),
            recovery_horizon=4,
            improve_margin=0.005,
            floor_tol_threshold=0.02,
            recovery_shadow_tol_threshold=0.05,
        )
        self.assertTrue(
            torch.allclose(tgt["y_success"].reshape(batch.T, batch.B), train.float())
        )

    def test_drop_target_elementwise_equal(self):
        from dexpbt_isaaclab.world_model.rssm_lite import RSSMLiteWorldModel
        from dexpbt_isaaclab.world_model.wm_calibration_metrics import (
            build_wm_calibration_targets,
        )
        from dexpbt_isaaclab.world_model.wm_head_targets import build_horizon_drop_target

        batch = self._batch(fail_at=2, success_bump_at=None)
        train = RSSMLiteWorldModel._build_horizon_drop_target(batch, 4)
        shared = build_horizon_drop_target(batch, 4)
        self.assertTrue(torch.equal(train, shared))
        tgt, _ = build_wm_calibration_targets(
            batch,
            horizon_steps=(4,),
            recovery_horizon=4,
            improve_margin=0.005,
            floor_tol_threshold=0.02,
            recovery_shadow_tol_threshold=0.05,
        )
        self.assertTrue(
            torch.allclose(tgt["y_drop"].reshape(batch.T, batch.B), train.float())
        )

    def test_recovery_target_and_mask_elementwise_equal(self):
        from dexpbt_isaaclab.world_model.rssm_lite import RSSMLiteWorldModel
        from dexpbt_isaaclab.world_model.wm_calibration_metrics import (
            build_wm_calibration_targets,
        )
        from dexpbt_isaaclab.world_model.wm_head_targets import (
            build_recovery_target_soft_or,
        )

        batch = self._batch(success_bump_at=2, d0=0.03, d_end=0.01)
        model = RSSMLiteWorldModel(
            obs_dim=4,
            action_dim=3,
            obs_embed_dim=8,
            action_embed_dim=4,
            h_dim=8,
            z_dim=4,
            head_hidden=8,
            with_recovery_head=True,
            recovery_horizon=4,
        )
        train_y, train_m = model._build_recovery_target_soft_or(
            batch,
            horizon=4,
            d_hat_margin=0.005,
            d_hat_tol_min_steps=0,
            floor_tol_threshold=0.05,
        )
        shared_y, shared_m = build_recovery_target_soft_or(
            batch,
            horizon=4,
            d_hat_margin=0.005,
            d_hat_tol_min_steps=0,
            floor_tol_threshold=0.05,
        )
        self.assertTrue(torch.equal(train_y, shared_y))
        self.assertTrue(torch.equal(train_m, shared_m))
        tgt, _ = build_wm_calibration_targets(
            batch,
            horizon_steps=(4,),
            recovery_horizon=4,
            improve_margin=0.005,
            floor_tol_threshold=0.02,
            recovery_shadow_tol_threshold=0.05,
        )
        self.assertTrue(
            torch.allclose(tgt["y_recovery"].reshape(batch.T, batch.B), train_y.float())
        )
        self.assertTrue(
            torch.equal(
                tgt["recovery_mask"].reshape(batch.T, batch.B), (train_m > 0.5)
            )
        )

    def test_recovery_uses_recovery_horizon_not_primary(self):
        from dexpbt_isaaclab.world_model.wm_calibration_metrics import (
            build_wm_calibration_targets,
        )
        from dexpbt_isaaclab.world_model.wm_head_targets import (
            build_recovery_target_soft_or,
        )

        # Success increment only visible under horizon>=6 (nsc rises at t=5).
        batch = self._batch(t=12, success_bump_at=5)
        y_short, _ = build_recovery_target_soft_or(
            batch,
            horizon=4,
            d_hat_margin=10.0,  # disable margin path
            d_hat_tol_min_steps=0,
            floor_tol_threshold=0.05,
        )
        y_long, _ = build_recovery_target_soft_or(
            batch,
            horizon=8,
            d_hat_margin=10.0,
            d_hat_tol_min_steps=0,
            floor_tol_threshold=0.05,
        )
        # At t=0, H=4 sees nsc[3]-sc[0]=0; H=8 sees nsc[7]-sc[0]=1.
        self.assertEqual(float(y_short[0, 0].item()), 0.0)
        self.assertEqual(float(y_long[0, 0].item()), 1.0)
        tgt, diag = build_wm_calibration_targets(
            batch,
            horizon_steps=(4,),
            recovery_horizon=8,
            improve_margin=10.0,
            floor_tol_threshold=0.02,
            recovery_shadow_tol_threshold=0.05,
        )
        self.assertEqual(diag["wm_cal/recovery_horizon"], 8.0)
        self.assertEqual(float(tgt["y_recovery"].reshape(batch.T, batch.B)[0, 0]), 1.0)

    def test_drop_does_not_clear_recovery_positive(self):
        from dexpbt_isaaclab.world_model.wm_calibration_metrics import (
            build_wm_calibration_targets,
        )

        batch = self._batch(success_bump_at=2, fail_at=1)
        tgt, _ = build_wm_calibration_targets(
            batch,
            horizon_steps=(4,),
            recovery_horizon=4,
            improve_margin=0.005,
            floor_tol_threshold=0.02,
            recovery_shadow_tol_threshold=0.05,
        )
        y = tgt["y_recovery"].reshape(batch.T, batch.B)
        d = tgt["y_drop"].reshape(batch.T, batch.B)
        # Row 0 has both drop and success-inc recovery.
        self.assertGreater(float(d[0, 0].item()), 0.5)
        self.assertGreater(float(y[0, 0].item()), 0.5)

    def test_done_and_dhat_worsen_not_calibration_only_drop(self):
        from dexpbt_isaaclab.world_model.rssm_lite import RSSMLiteWorldModel
        from dexpbt_isaaclab.world_model.wm_calibration_metrics import (
            build_wm_calibration_targets,
        )

        # Worsening d_hat + done, but no failure/dropped flags.
        batch = self._batch(
            success_bump_at=None, fail_at=None, d0=0.01, d_end=0.08, done_at=2
        )
        train = RSSMLiteWorldModel._build_horizon_drop_target(batch, 4)
        tgt, _ = build_wm_calibration_targets(
            batch,
            horizon_steps=(4,),
            recovery_horizon=4,
            improve_margin=0.005,
            floor_tol_threshold=0.02,
            recovery_shadow_tol_threshold=0.05,
        )
        cal = tgt["y_drop"].reshape(batch.T, batch.B)
        self.assertTrue(torch.equal(train, cal))
        self.assertEqual(float(cal.max().item()), 0.0)

    def test_multi_horizon_mean_matches_stack_mean(self):
        from dexpbt_isaaclab.world_model.wm_calibration_metrics import (
            build_wm_calibration_targets,
        )
        from dexpbt_isaaclab.world_model.wm_head_targets import (
            build_horizon_success_target,
        )

        batch = self._batch(t=12, success_bump_at=5)
        s4 = build_horizon_success_target(batch, 4)
        s8 = build_horizon_success_target(batch, 8)
        expected = torch.stack([s4, s8], dim=0).mean(dim=0)
        tgt, _ = build_wm_calibration_targets(
            batch,
            horizon_steps=(4, 8),
            recovery_horizon=4,
            improve_margin=0.005,
            floor_tol_threshold=0.02,
            recovery_shadow_tol_threshold=0.05,
        )
        self.assertTrue(
            torch.allclose(tgt["y_success"].reshape(batch.T, batch.B), expected)
        )
        # At t=0 mean is 0.5 (H4=0, H8=1).
        self.assertAlmostEqual(float(expected[0, 0].item()), 0.5, places=5)

    def test_combined_mask_is_intersection(self):
        from dexpbt_isaaclab.world_model.wm_calibration_metrics import (
            build_wm_calibration_targets,
        )
        from dexpbt_isaaclab.world_model.wm_head_targets import (
            build_horizon_valid_mask,
            build_recovery_target_soft_or,
        )

        batch = self._batch(t=12, success_bump_at=1)
        m4 = build_horizon_valid_mask(batch, 4)
        m8 = build_horizon_valid_mask(batch, 8)
        _, rec_m = build_recovery_target_soft_or(
            batch,
            horizon=4,
            d_hat_margin=0.005,
            d_hat_tol_min_steps=0,
            floor_tol_threshold=0.05,
        )
        expected = (torch.stack([m4, m8], dim=0).amin(dim=0) > 0.5) & (rec_m > 0.5)
        tgt, _ = build_wm_calibration_targets(
            batch,
            horizon_steps=(4, 8),
            recovery_horizon=4,
            improve_margin=0.005,
            floor_tol_threshold=0.02,
            recovery_shadow_tol_threshold=0.05,
        )
        self.assertTrue(
            torch.equal(tgt["eval_mask"].reshape(batch.T, batch.B), expected)
        )

    def test_missing_target_source_fail_closed_clears_alpha(self):
        from dexpbt_isaaclab.world_model.wm_calibration_metrics import (
            build_wm_calibration_targets,
        )
        from dexpbt_isaaclab.world_model.wm_head_probs import (
            FAILURE_MISSING_TARGET_SOURCE,
        )

        batch = self._batch(success_sources=False)
        tgt, diag = build_wm_calibration_targets(
            batch,
            horizon_steps=(4,),
            recovery_horizon=4,
            improve_margin=0.005,
            floor_tol_threshold=0.02,
            recovery_shadow_tol_threshold=0.05,
        )
        self.assertEqual(diag["wm_cal/missing_target_source"], 1.0)
        self.assertEqual(diag["wm_cal/failure_reason_id"], FAILURE_MISSING_TARGET_SOURCE)
        self.assertEqual(int(tgt["eval_mask"].sum().item()), 0)
        cfg = WMCalibrationConfig(enabled=True, alpha_min=0.2)
        st = WMCalibrationState(alpha_wm=0.7, ema_raw=0.7, calibration_pass=True)
        st = apply_calibration_outcome(
            st,
            cfg=cfg,
            frames=1,
            metrics=dict(diag),
            mode="structural",
        )
        self.assertEqual(st.alpha_wm, 0.0)
        self.assertEqual(st.ema_raw, 0.0)

    def test_heldout_fraction_marked_disabled(self):
        from dexpbt_isaaclab.world_model.wm_calibration_metrics import (
            build_wm_calibration_targets,
        )
        from dexpbt_isaaclab.world_model.wm_heldout_partition import (
            build_env_partition_masks,
        )

        batch = self._batch()
        _, diag = build_wm_calibration_targets(
            batch,
            horizon_steps=(4,),
            recovery_horizon=4,
            improve_margin=0.005,
            floor_tol_threshold=0.02,
            recovery_shadow_tol_threshold=0.05,
        )
        # Label builder does not own partition telemetry.
        self.assertNotIn("wm_cal/heldout_fraction_enabled", diag)
        # Fraction 0 => partition disabled (all train).
        part = build_env_partition_masks(
            16, heldout_fraction=0.0, partition_seed=0, worker_rank=0
        )
        self.assertEqual(part.diagnostics["wm_cal/heldout_enabled"], 0.0)
        self.assertTrue(bool(part.train_mask.all()))
        self.assertFalse(bool(part.heldout_mask.any()))
        self.assertGreater(WMCalibrationConfig().heldout_fraction, 0.0)


class TestCalibrationDeterminismAndTelemetry(unittest.TestCase):
    def _trainer_and_batch(self):
        from dexpbt_isaaclab.world_model.rssm_lite import RSSMLiteWorldModel
        from dexpbt_isaaclab.world_model.seq_wm_base import SequenceBatch
        from dexpbt_isaaclab.world_model.seq_wm_trainer import SeqWMConfig, SeqWMTrainer

        torch.manual_seed(0)
        t, b, obs_dim, act_dim = 10, 4, 6, 3
        model = RSSMLiteWorldModel(
            obs_dim=obs_dim,
            action_dim=act_dim,
            obs_embed_dim=16,
            action_embed_dim=8,
            h_dim=16,
            z_dim=4,
            head_hidden=16,
            with_horizon_heads=True,
            horizon_steps=(4,),
            with_recovery_head=True,
            recovery_horizon=4,
        )
        cfg = SeqWMConfig(
            seq_len=t,
            seq_batch_size=b,
            horizon_steps=(4,),
            use_horizon_heads=True,
            use_recovery_head=True,
            recovery_horizon=4,
            floor_tol_threshold=0.02,
            recovery_shadow_tol_threshold=0.05,
            recovery_d_hat_margin=0.005,
        )
        trainer = SeqWMTrainer(model=model, cfg=cfg, device="cpu", strategy="rssm_belief_value")
        sc = torch.zeros(t, b)
        nsc = torch.zeros(t, b)
        nsc[2:, :] = 1.0
        batch = SequenceBatch(
            obs=torch.randn(t, b, obs_dim),
            action=torch.randn(t, b, act_dim),
            reward=torch.zeros(t, b),
            done=torch.zeros(t, b),
            next_obs=torch.randn(t, b, obs_dim),
            returns=torch.zeros(t, b),
            d_hat=torch.full((t, b), 0.03),
            success_tolerance_seq=torch.full((t, b), 0.015),
            valid_mask=torch.ones(t, b),
            segment_id=torch.zeros(t, b, dtype=torch.long),
            dropped=torch.zeros(t, b),
            failure=torch.zeros(t, b),
            success_count=sc,
            next_success_count=nsc,
        )
        return trainer, batch

    def test_deterministic_calibration_twice_identical(self):
        trainer, batch = self._trainer_and_batch()
        trainer.model.train()
        was = trainer.model.training

        class _FixedSampler:
            def sample(self, **kwargs):
                return batch

        trainer.sampler = _FixedSampler()
        ev = WMCalibrationEvaluator(
            WMCalibrationConfig(
                enabled=True,
                min_samples=4,
                eval_interval_frames=0,
                alpha_min=0.0,
                ema_beta=0.0,
                gate_auc_recovery=0.0,
                gate_ece_max=1.0,
                gate_rank_corr_min=-1.0,
                heldout_fraction=0.0,
            )
        )

        class _Buf:
            tensor_dict = {
                "obses": batch.obs,
                "actions": batch.action,
                "rewards": batch.reward.unsqueeze(-1),
                "dones": batch.done.to(torch.uint8),
            }

        st1 = WMCalibrationState()
        st1 = trainer.run_wm_calibration_eval(
            experience_buffer=_Buf(),
            frames=1,
            last_obs=batch.obs[-1],
            returns_full=batch.returns.unsqueeze(-1),
            evaluator=ev,
            state=st1,
        )
        m1 = dict(st1.last_metrics)
        p1 = (
            m1.get("wm_cal/combined_score_mean"),
            m1.get("wm_cal/auc_recovery"),
            m1.get("wm_cal/alpha_wm"),
            m1.get("wm_cal/rank_corr_combined"),
        )
        st2 = WMCalibrationState()
        st2 = trainer.run_wm_calibration_eval(
            experience_buffer=_Buf(),
            frames=1,
            last_obs=batch.obs[-1],
            returns_full=batch.returns.unsqueeze(-1),
            evaluator=ev,
            state=st2,
        )
        m2 = dict(st2.last_metrics)
        p2 = (
            m2.get("wm_cal/combined_score_mean"),
            m2.get("wm_cal/auc_recovery"),
            m2.get("wm_cal/alpha_wm"),
            m2.get("wm_cal/rank_corr_combined"),
        )
        self.assertEqual(p1, p2)
        self.assertEqual(trainer.model.training, was)

    def test_calibration_restores_training_state(self):
        trainer, batch = self._trainer_and_batch()
        trainer.model.train()

        class _FixedSampler:
            def sample(self, **kwargs):
                return batch

        trainer.sampler = _FixedSampler()

        class _Buf:
            tensor_dict = {
                "obses": batch.obs,
                "actions": batch.action,
                "rewards": batch.reward.unsqueeze(-1),
                "dones": batch.done.to(torch.uint8),
            }

        ev = WMCalibrationEvaluator(
            WMCalibrationConfig(
                enabled=True, min_samples=4, eval_interval_frames=0, heldout_fraction=0.0
            )
        )
        trainer.run_wm_calibration_eval(
            experience_buffer=_Buf(),
            frames=1,
            last_obs=batch.obs[-1],
            returns_full=batch.returns.unsqueeze(-1),
            evaluator=ev,
            state=WMCalibrationState(),
        )
        self.assertTrue(trainer.model.training)

        trainer.model.eval()
        trainer.run_wm_calibration_eval(
            experience_buffer=_Buf(),
            frames=10_000_000,
            last_obs=batch.obs[-1],
            returns_full=batch.returns.unsqueeze(-1),
            evaluator=ev,
            state=WMCalibrationState(last_eval_frame=0),
        )
        self.assertFalse(trainer.model.training)

    def test_insufficient_reason_not_overwritten(self):
        from dexpbt_isaaclab.world_model.wm_head_probs import (
            FAILURE_INSUFFICIENT_SAMPLES,
            FAILURE_OK,
        )

        cfg = WMCalibrationConfig(enabled=True, ema_beta=0.9)
        st = WMCalibrationState(alpha_wm=0.5, ema_raw=0.5)
        metrics = {
            "wm_cal/failure_reason_id": FAILURE_OK,  # noisy prior
            "wm_cal/n_samples": 2.0,
        }
        st = apply_calibration_outcome(
            st,
            cfg=cfg,
            frames=3,
            metrics=metrics,
            mode="insufficient",
        )
        # setdefault must keep authoritative id if already written by caller.
        metrics2 = {
            "wm_cal/failure_reason_id": FAILURE_INSUFFICIENT_SAMPLES,
            "wm_cal/n_samples": 2.0,
        }
        st2 = WMCalibrationState(alpha_wm=0.5, ema_raw=0.5)
        st2 = apply_calibration_outcome(
            st2, cfg=cfg, frames=3, metrics=metrics2, mode="insufficient"
        )
        self.assertEqual(
            st2.last_metrics["wm_cal/failure_reason_id"], FAILURE_INSUFFICIENT_SAMPLES
        )
        # Even if only setdefault path: after apply, reason is insufficient.
        self.assertEqual(
            st.last_metrics["wm_cal/failure_reason_id"], FAILURE_INSUFFICIENT_SAMPLES
        )

    def test_telemetry_counts_only_target_subset(self):
        trainer, batch = self._trainer_and_batch()

        class _FixedSampler:
            def sample(self, **kwargs):
                return batch

        trainer.sampler = _FixedSampler()

        class _Buf:
            tensor_dict = {
                "obses": batch.obs,
                "actions": batch.action,
                "rewards": batch.reward.unsqueeze(-1),
                "dones": batch.done.to(torch.uint8),
            }

        # Force some nonfinite preds so invalid_prediction_count > 0 on subset.
        heads = trainer.model.heads
        orig = heads.predict_transition

        def _nan_some(belief_flat, action_flat):
            out = orig(belief_flat, action_flat)
            n = belief_flat.shape[0]
            succ = out["horizon_logits"][4]["success"].clone()
            succ[0] = float("nan")
            out["horizon_logits"][4]["success"] = succ
            return out

        heads.predict_transition = _nan_some
        ev = WMCalibrationEvaluator(
            WMCalibrationConfig(
                enabled=True,
                min_samples=100000,
                eval_interval_frames=0,
                heldout_fraction=0.0,
            )
        )
        st = trainer.run_wm_calibration_eval(
            experience_buffer=_Buf(),
            frames=1,
            last_obs=batch.obs[-1],
            returns_full=batch.returns.unsqueeze(-1),
            evaluator=ev,
            state=WMCalibrationState(),
        )
        m = st.last_metrics
        target_n = float(m["wm_cal/target_eval_count"])
        valid_n = float(m["wm_cal/valid_prediction_count"])
        invalid_n = float(m["wm_cal/invalid_prediction_count"])
        self.assertAlmostEqual(valid_n + invalid_n, target_n, places=5)
        self.assertAlmostEqual(
            float(m["wm_cal/valid_prediction_frac"]),
            valid_n / max(target_n, 1.0),
            places=5,
        )
        self.assertIn("wm_cal/batch_valid_prediction_count", m)
        self.assertEqual(
            m["wm_cal/failure_reason_id"],
            __import__(
                "dexpbt_isaaclab.world_model.wm_head_probs",
                fromlist=["FAILURE_INSUFFICIENT_SAMPLES"],
            ).FAILURE_INSUFFICIENT_SAMPLES,
        )

    def test_predict_exception_not_missing_head(self):
        from dexpbt_isaaclab.world_model.wm_calibration_metrics import (
            extract_wm_calibration_predictions,
        )
        from dexpbt_isaaclab.world_model.wm_head_probs import (
            FAILURE_PREDICT_EXCEPTION,
        )

        class _Boom:
            has_recovery_head = True

            def predict_transition(self, *a, **k):
                raise RuntimeError("boom")

        preds, diag = extract_wm_calibration_predictions(
            _Boom(),
            belief_flat=torch.zeros(2, 4),
            action_flat=torch.zeros(2, 3),
            horizon_steps=(4,),
        )
        self.assertIsNone(preds)
        self.assertEqual(diag["wm_cal/failure_reason_id"], FAILURE_PREDICT_EXCEPTION)
        self.assertEqual(diag.get("wm_cal/missing_head", 0.0), 0.0)


if __name__ == "__main__":
    unittest.main()
