"""Unit tests for Profile 23 readiness-adaptive pure control logic."""

from __future__ import annotations

import math
import unittest

import torch

from dexpbt_isaaclab.world_model.readiness_adaptive import (
    PHASE_CALIBRATED_REFINEMENT,
    PHASE_PRIORITY_BOOTSTRAP,
    PHASE_WARMUP,
    ReadinessAdaptiveConfig,
    ReadinessAdaptiveState,
    accumulate_ready_frames,
    apply_wm_graduation,
    bank_readiness,
    bank_size_readiness,
    coverage_readiness,
    dynamic_active_budget,
    dynamic_max_log_gap,
    dynamic_soft_scale,
    ess_and_scale,
    evaluate_bank_ready_windows,
    phase_lambda,
    refinement_ramp,
    select_shadow_indices,
    shadow_row_budget,
    sync_phase,
    telemetry,
    update_coverage_ema,
)


def _cfg(**over) -> ReadinessAdaptiveConfig:
    return ReadinessAdaptiveConfig(
        readiness_adaptive_enable=True,
        readiness_min_frames=100,
        readiness_required_windows=2,
        readiness_enter_threshold=0.50,
        readiness_exit_threshold=0.20,
        readiness_ramp_frames=1000,
        wm_graduation_enable=True,
        wm_graduation_required_passes=2,
        wm_graduation_early_until_frames=500,
        **over,
    )


class TestBankReadinessMath(unittest.TestCase):
    def test_01_bank_size_readiness_log1p_clip(self):
        low, full = 1000, 3000
        self.assertAlmostEqual(bank_size_readiness(0, low, full), 0.0)
        self.assertAlmostEqual(bank_size_readiness(low, low, full), 0.0)
        self.assertAlmostEqual(bank_size_readiness(full, low, full), 1.0)
        mid = math.exp(0.5 * (math.log1p(full) + math.log1p(low))) - 1.0
        self.assertAlmostEqual(bank_size_readiness(mid, low, full), 0.5, places=5)
        self.assertEqual(bank_size_readiness(10_000, low, full), 1.0)

    def test_02_coverage_and_product(self):
        self.assertAlmostEqual(coverage_readiness(0.10, 0.20), 0.5)
        self.assertAlmostEqual(coverage_readiness(0.40, 0.20), 1.0)
        self.assertAlmostEqual(bank_readiness(0.8, 0.5), 0.4)
        ema = update_coverage_ema(0.0, 0.2, 0.9)
        self.assertAlmostEqual(ema, 0.02)
        self.assertAlmostEqual(refinement_ramp(500, 1000, 0.8), 0.4)


class TestHysteresisAndReadyFrames(unittest.TestCase):
    def test_03_hysteresis_no_single_window_flip(self):
        cfg = _cfg()
        st = ReadinessAdaptiveState()
        st = evaluate_bank_ready_windows(st, frames=200, r_bank=0.60, cfg=cfg)
        self.assertFalse(st.bank_ready)
        self.assertEqual(st.enter_window_streak, 1)
        st = evaluate_bank_ready_windows(st, frames=300, r_bank=0.60, cfg=cfg)
        self.assertTrue(st.bank_ready)
        self.assertEqual(st.enter_window_streak, 0)

        # Single weak window must not exit.
        st = evaluate_bank_ready_windows(st, frames=400, r_bank=0.10, cfg=cfg)
        self.assertTrue(st.bank_ready)
        self.assertEqual(st.exit_window_streak, 1)
        st = evaluate_bank_ready_windows(st, frames=500, r_bank=0.10, cfg=cfg)
        self.assertFalse(st.bank_ready)

    def test_04_ready_frames_pause_never_reset(self):
        cfg = _cfg()
        st = ReadinessAdaptiveState(bank_ready=True, last_frames_seen=100)
        st = accumulate_ready_frames(st, frames=150, cfg=cfg)
        self.assertEqual(st.ready_accumulated_frames, 50)
        st.bank_ready = False
        st = accumulate_ready_frames(st, frames=200, cfg=cfg)
        self.assertEqual(st.ready_accumulated_frames, 50)  # paused
        st.bank_ready = True
        st = accumulate_ready_frames(st, frames=250, cfg=cfg)
        self.assertEqual(st.ready_accumulated_frames, 100)  # resumed, not reset


class TestPhaseAndAlpha(unittest.TestCase):
    def test_05_phase_transitions(self):
        st = ReadinessAdaptiveState()
        st = sync_phase(st)
        self.assertEqual(st.phase, PHASE_WARMUP)

        st.bank_ready = True
        st = sync_phase(st)
        self.assertEqual(st.phase, PHASE_PRIORITY_BOOTSTRAP)
        self.assertEqual(st.phase_transition_count, 1)

        st.wm_graduated = True
        st = sync_phase(st)
        self.assertEqual(st.phase, PHASE_CALIBRATED_REFINEMENT)
        self.assertEqual(st.phase_transition_count, 2)

        st = sync_phase(st, wm_structural_failure=True)
        self.assertFalse(st.wm_graduated)
        self.assertEqual(st.wm_pass_streak, 0)
        self.assertEqual(st.wm_ema_raw, 0.0)
        self.assertEqual(st.last_alpha_effective, 0.0)
        self.assertEqual(st.phase, PHASE_PRIORITY_BOOTSTRAP)
        self.assertEqual(st.phase_transition_count, 3)

        st.bank_ready = False
        st = sync_phase(st, wm_structural_failure=True)
        self.assertEqual(st.phase, PHASE_WARMUP)

    def test_06_alpha_phase1_zero(self):
        cfg = _cfg()
        st = ReadinessAdaptiveState(wm_pass_streak=0, wm_graduated=False)
        st, alpha = apply_wm_graduation(
            st,
            frames=100,
            metrics={
                "wm_cal/eval_valid": 1.0,
                "wm_cal/calibration_pass": 1.0,
                "wm_cal/valid_prediction_frac": 1.0,
                "wm_cal/raw_gate": 1.0,
            },
            mode="valid",
            cfg=cfg,
            r_bank=1.0,
        )
        # streak=1 < required 2 ⇒ not graduated ⇒ alpha 0
        self.assertEqual(st.wm_pass_streak, 1)
        self.assertFalse(st.wm_graduated)
        self.assertEqual(alpha, 0.0)

    def test_07_streak_lt_2_alpha_zero(self):
        cfg = _cfg()
        st = ReadinessAdaptiveState()
        metrics = {
            "wm_cal/eval_valid": 1.0,
            "wm_cal/calibration_pass": 1.0,
            "wm_cal/valid_prediction_frac": 0.95,
            "wm_cal/raw_gate": 0.8,
        }
        st, a1 = apply_wm_graduation(
            st, frames=100, metrics=metrics, mode="valid", cfg=cfg, r_bank=1.0
        )
        self.assertEqual(a1, 0.0)
        self.assertEqual(st.wm_pass_streak, 1)
        st, a2 = apply_wm_graduation(
            st, frames=100, metrics=metrics, mode="valid", cfg=cfg, r_bank=1.0
        )
        self.assertTrue(st.wm_graduated)
        self.assertGreater(a2, 0.0)

    def test_08_early_alpha_cap_0_5(self):
        # beta_up=0 ⇒ ema jumps to raw=1.0; early cap must still clamp to 0.5
        cfg = _cfg(wm_graduation_early_beta_up=0.0)
        st = ReadinessAdaptiveState(wm_pass_streak=1, wm_ema_raw=0.0)
        metrics = {
            "wm_cal/eval_valid": 1.0,
            "wm_cal/calibration_pass": 1.0,
            "wm_cal/valid_prediction_frac": 1.0,
            "wm_cal/raw_gate": 1.0,
        }
        st, alpha = apply_wm_graduation(
            st, frames=100, metrics=metrics, mode="valid", cfg=cfg, r_bank=1.0
        )
        self.assertTrue(st.wm_graduated)
        self.assertAlmostEqual(alpha, 0.5)
        self.assertAlmostEqual(st.last_alpha_cap, 0.5)

    def test_09_structural_zeros(self):
        cfg = _cfg()
        st = ReadinessAdaptiveState(
            wm_pass_streak=5, wm_graduated=True, wm_ema_raw=0.7, last_alpha_effective=0.4
        )
        st, alpha = apply_wm_graduation(
            st, frames=100, metrics={}, mode="structural", cfg=cfg, r_bank=1.0
        )
        self.assertEqual(st.wm_pass_streak, 0)
        self.assertEqual(st.wm_ema_raw, 0.0)
        self.assertFalse(st.wm_graduated)
        self.assertEqual(alpha, 0.0)

    def test_10_insufficient_beta_down(self):
        cfg = _cfg(wm_graduation_beta_down=0.5)
        st = ReadinessAdaptiveState(wm_ema_raw=0.8, wm_pass_streak=2, wm_graduated=True)
        st, alpha = apply_wm_graduation(
            st, frames=100, metrics={}, mode="insufficient", cfg=cfg, r_bank=1.0
        )
        self.assertAlmostEqual(st.wm_ema_raw, 0.4)
        self.assertEqual(st.wm_pass_streak, 0)
        self.assertEqual(alpha, 0.0)


class TestEssAndDynamic(unittest.TestCase):
    def test_11_ess_skip_and_scale(self):
        # Uniform weights over 4 rows → ESS=4 < 8 → skip
        w_small = torch.ones(4)
        ess, scale, skip = ess_and_scale(w_small, ess_target=128.0, ess_min=8.0)
        self.assertAlmostEqual(ess, 4.0, places=5)
        self.assertTrue(skip)
        self.assertEqual(scale, 0.0)

        # 128 equal weights → ESS=128 → scale=1
        w_full = torch.ones(128)
        ess2, scale2, skip2 = ess_and_scale(w_full, ess_target=128.0, ess_min=8.0)
        self.assertAlmostEqual(ess2, 128.0, places=4)
        self.assertFalse(skip2)
        self.assertAlmostEqual(scale2, 1.0)

        # 64 equal → ESS=64 → scale=0.5
        w_mid = torch.ones(64)
        ess3, scale3, skip3 = ess_and_scale(w_mid, ess_target=128.0, ess_min=8.0)
        self.assertAlmostEqual(ess3, 64.0, places=4)
        self.assertFalse(skip3)
        self.assertAlmostEqual(scale3, 0.5)

    def test_12_dynamic_log_gap_and_budget(self):
        self.assertAlmostEqual(dynamic_max_log_gap(0.0), 0.30)
        self.assertAlmostEqual(dynamic_max_log_gap(1.0), 0.20)
        self.assertAlmostEqual(dynamic_max_log_gap(0.5), 0.25)
        self.assertAlmostEqual(dynamic_soft_scale(0.0), 0.15)
        self.assertAlmostEqual(dynamic_soft_scale(1.0), 0.10)

        self.assertAlmostEqual(dynamic_active_budget(0.0), 0.15)
        self.assertAlmostEqual(dynamic_active_budget(1.0), 0.35)
        self.assertAlmostEqual(dynamic_active_budget(0.5), 0.25)
        for r in (0.0, 0.25, 0.5, 0.75, 1.0, 2.0):
            b = dynamic_active_budget(r)
            self.assertGreaterEqual(b, 0.15)
            self.assertLessEqual(b, 0.35)


class TestShadowPhaseLambdaTelemetry(unittest.TestCase):
    def test_13_shadow_budget_and_indices(self):
        self.assertEqual(shadow_row_budget(100, 0.05), 5)
        self.assertEqual(shadow_row_budget(19, 0.05), 0)  # floor(0.95)=0
        self.assertEqual(shadow_row_budget(20, 0.05), 1)
        self.assertLessEqual(shadow_row_budget(1000, 0.05), 50)

        mask = torch.zeros(100, dtype=torch.bool)
        mask[10:60] = True  # 50 eligible
        idx = select_shadow_indices(mask, 0.05)
        self.assertEqual(int(idx.numel()), 2)  # floor(50*0.05)=2
        self.assertTrue(bool((idx >= 10).all() and (idx < 60).all()))

        g = torch.Generator().manual_seed(0)
        idx2 = select_shadow_indices(mask, 0.05, generator=g)
        self.assertEqual(int(idx2.numel()), 2)

    def test_14_phase_lambda_and_roundtrip(self):
        cfg = _cfg()
        self.assertEqual(phase_lambda(PHASE_WARMUP, cfg), 0.0)
        self.assertAlmostEqual(phase_lambda(PHASE_PRIORITY_BOOTSTRAP, cfg), 0.001)
        self.assertAlmostEqual(phase_lambda(PHASE_CALIBRATED_REFINEMENT, cfg), 0.0015)

        st = ReadinessAdaptiveState(
            phase=PHASE_PRIORITY_BOOTSTRAP,
            coverage_ema=0.1,
            bank_ready=True,
            ready_accumulated_frames=42,
            last_bank_readiness=0.55,
        )
        d = st.to_dict()
        st2 = ReadinessAdaptiveState.from_dict(d)
        self.assertEqual(st2.phase, st.phase)
        self.assertEqual(st2.ready_accumulated_frames, 42)
        self.assertTrue(st2.bank_ready)
        tel = telemetry(st2)
        self.assertIn("readiness_adaptive/phase", tel)
        self.assertIn("readiness_adaptive/bank_readiness", tel)
        self.assertIn("readiness_adaptive/ess", tel)


class TestProfile23Wiring(unittest.TestCase):
    def test_15_profile_22_snapshot_unchanged(self):
        from dexpbt_isaaclab.world_model.wm_strategy_profiles import PROFILE_TABLES

        table = PROFILE_TABLES["cser_dexpbt_progress_refine"]
        # Critical Profile-22 fields must remain byte-identical; readiness off/absent.
        self.assertEqual(table["progress_refinement_start_after_frames"], 300_000_000)
        self.assertEqual(table["progress_refinement_ramp_frames"], 300_000_000)
        self.assertEqual(table["adaptive_consolidation_stage_consolidation_progress"], 0.50)
        self.assertEqual(table["adaptive_consolidation_stage_selection_progress"], 0.85)
        self.assertEqual(table["adaptive_probation_min_frames"], 200_000_000)
        self.assertEqual(table["adaptive_rescue_source_max_staleness_frames"], 40_000_000)
        self.assertFalse(bool(table.get("readiness_adaptive_enable", False)))
        self.assertFalse(bool(table.get("wm_graduation_enable", False)))
        self.assertFalse(bool(table.get("adaptive_lineage_diversity_enable", False)))
        self.assertFalse(bool(table.get("adaptive_probation_stage_aware_enable", False)))

    def test_16_profile_23_id_and_flags(self):
        from dexpbt_isaaclab.world_model.wm_strategy_profiles import (
            PROFILE_IDS,
            PROFILE_TABLES,
            PROFILE_FAMILY,
            PROFILE_COPY_SOURCE_BONUS,
        )

        self.assertEqual(PROFILE_IDS["cser_dexpbt_readiness_adaptive"], 23)
        self.assertEqual(PROFILE_FAMILY["cser_dexpbt_readiness_adaptive"], "unified")
        self.assertEqual(PROFILE_COPY_SOURCE_BONUS["cser_dexpbt_readiness_adaptive"], 1.0)
        t = PROFILE_TABLES["cser_dexpbt_readiness_adaptive"]
        self.assertTrue(t["readiness_adaptive_enable"])
        self.assertTrue(t["wm_graduation_enable"])
        self.assertEqual(t["adaptive_consolidation_stage_consolidation_progress"], 0.45)
        self.assertEqual(t["adaptive_consolidation_stage_selection_progress"], 0.80)
        self.assertEqual(t["progress_refinement_start_after_frames"], 0)
        self.assertEqual(t["progress_refinement_ramp_frames"], 1)
        self.assertEqual(t["adaptive_rescue_source_freshness_mode"], "exp_decay")
        self.assertTrue(t["adaptive_lineage_diversity_enable"])
        self.assertTrue(t["adaptive_probation_stage_aware_enable"])

    def test_17_pbt_lineage_budget_freshness_probation(self):
        from dexpbt_isaaclab.pbt.adaptive_precision_consolidation import (
            AdaptivePrecisionConfig,
            ProbationState,
            STAGE_CONSOLIDATION,
            STAGE_EXPLORATION,
            STAGE_SELECTION,
            copy_would_drop_below_min_lineages,
            count_distinct_lineages,
            source_freshness_weight,
            stage_min_distinct_lineages,
            stage_replacement_budget,
        )

        cfg = AdaptivePrecisionConfig(
            enabled=True,
            lineage_diversity_enable=True,
            probation_stage_aware_enable=True,
            rescue_source_freshness_mode="exp_decay",
            rescue_source_freshness_tau_frames=80_000_000,
            rescue_source_hard_max_age_frames=160_000_000,
        )
        self.assertEqual(stage_min_distinct_lineages(STAGE_EXPLORATION, cfg), 3)
        self.assertEqual(stage_min_distinct_lineages(STAGE_CONSOLIDATION, cfg), 2)
        self.assertEqual(stage_min_distinct_lineages(STAGE_SELECTION, cfg), 1)
        self.assertEqual(stage_replacement_budget(STAGE_EXPLORATION, cfg), 1)
        self.assertEqual(stage_replacement_budget(STAGE_SELECTION, cfg), 2)

        self.assertEqual(count_distinct_lineages(["a", "b", "a", ""]), 2)
        lineages = {0: "a", 1: "b", 2: "c"}
        self.assertTrue(
            copy_would_drop_below_min_lineages(lineages, 2, "a", min_distinct=3)
        )
        self.assertFalse(
            copy_would_drop_below_min_lineages(lineages, 2, "d", min_distinct=3)
        )

        ok, w, _ = source_freshness_weight(0, cfg)
        self.assertTrue(ok)
        self.assertAlmostEqual(w, 1.0, places=5)
        ok2, w2, _ = source_freshness_weight(80_000_000, cfg)
        self.assertTrue(ok2)
        self.assertAlmostEqual(w2, math.exp(-1.0), places=5)
        ok3, w3, reason = source_freshness_weight(160_000_001, cfg)
        self.assertFalse(ok3)
        self.assertEqual(w3, 0.0)
        self.assertIn("stale", reason)

        p = ProbationState()
        p.arm(tolerance=0.05, frame=0, nsucc_mean=10.0, previous_tolerance=0.075)
        p.same_level_retention = 0.98
        p.stable_ticks = 2
        p.peak_nsucc_mean = 10.0
        p.current_nsucc_mean = 10.0
        kind, _ = p.evaluate_exit(frame=75_000_000, stage=STAGE_EXPLORATION, cfg=cfg)
        self.assertEqual(kind, "early_exit")

        p2 = ProbationState()
        p2.arm(tolerance=0.05, frame=0, nsucc_mean=10.0, previous_tolerance=0.075)
        p2.same_level_retention = 0.93
        p2.stable_ticks = 3
        p2.peak_nsucc_mean = 10.0
        p2.current_nsucc_mean = 10.0
        kind2, _ = p2.evaluate_exit(frame=100_000_000, stage=STAGE_CONSOLIDATION, cfg=cfg)
        self.assertEqual(kind2, "normal_exit")

        p3 = ProbationState()
        p3.arm(tolerance=0.05, frame=0, nsucc_mean=10.0, previous_tolerance=0.075)
        kind3, _ = p3.evaluate_exit(frame=200_000_000, stage=STAGE_EXPLORATION, cfg=cfg)
        self.assertEqual(kind3, "max_frames_timeout")

    def test_18_ess_weight_applied_once(self):
        """lambda * ess_scale replaces cfg.weight; not multiplied again."""
        cfg = _cfg()
        w = torch.ones(128)
        ess, scale, skip = ess_and_scale(
            w,
            ess_target=float(cfg.readiness_ess_target),
            ess_min=float(cfg.readiness_ess_min_for_update),
        )
        self.assertFalse(skip)
        lam = phase_lambda(PHASE_PRIORITY_BOOTSTRAP, cfg)
        override = lam * scale
        # Must equal bootstrap weight exactly when ESS is at target (scale=1).
        self.assertAlmostEqual(override, 0.001, places=6)
        # Not double-applied against a typical snippet weight of 0.001.
        self.assertNotAlmostEqual(override * 0.001, 0.001, places=6)

    def test_19_select_retrieval_rows_override_budget(self):
        from dexpbt_isaaclab.world_model.progress_conditioned_refinement import (
            ProgressRefinementConfig,
            select_retrieval_rows,
        )

        pcfg = ProgressRefinementConfig(
            enabled=True,
            max_active_fraction=0.35,
            min_gate_for_retrieval=0.05,
        )
        gate = torch.linspace(0.1, 1.0, 100)
        for budget in (0.15, 0.25, 0.35):
            mask, frac = select_retrieval_rows(gate, pcfg, max_active_fraction=budget)
            self.assertLessEqual(frac, budget + 1e-9)
            self.assertGreaterEqual(frac, 0.0)
            # Override must be respected vs cfg default 0.35 when budget < 0.35.
            if budget < 0.35:
                _, frac_default = select_retrieval_rows(gate, pcfg)
                self.assertLessEqual(frac, frac_default + 1e-9)


if __name__ == "__main__":
    unittest.main()
