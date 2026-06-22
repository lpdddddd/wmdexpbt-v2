"""Tests for CSER-v0 PBT consolidation schedule."""

from __future__ import annotations

import unittest

from dexpbt_isaaclab.pbt.pbt_consolidation_schedule import (
    BottomRestartContext,
    PbtConsolidationConfig,
    STAGE_CONSOLIDATION,
    STAGE_EXPLORATION,
    STAGE_SELECTION,
    effective_mutation_rate,
    effective_tick_interval,
    evaluate_consolidated_tick,
    resolve_pbt_stage,
    should_allow_bottom_restart,
    stage_id,
)


class TestResolveStage(unittest.TestCase):
    def setUp(self):
        self.cfg = PbtConsolidationConfig(enabled=True)

    def test_boundaries(self):
        self.assertEqual(resolve_pbt_stage(0, self.cfg), STAGE_EXPLORATION)
        self.assertEqual(
            resolve_pbt_stage(self.cfg.stage1_end_frames - 1, self.cfg),
            STAGE_EXPLORATION,
        )
        self.assertEqual(
            resolve_pbt_stage(self.cfg.stage1_end_frames, self.cfg),
            STAGE_CONSOLIDATION,
        )
        self.assertEqual(
            resolve_pbt_stage(self.cfg.stage2_end_frames - 1, self.cfg),
            STAGE_CONSOLIDATION,
        )
        self.assertEqual(
            resolve_pbt_stage(self.cfg.stage2_end_frames, self.cfg),
            STAGE_SELECTION,
        )
        self.assertEqual(
            resolve_pbt_stage(self.cfg.stage3_end_frames, self.cfg),
            STAGE_SELECTION,
        )

    def test_disabled_always_exploration(self):
        cfg = PbtConsolidationConfig(enabled=False)
        self.assertEqual(
            resolve_pbt_stage(4_000_000_000, cfg), STAGE_EXPLORATION
        )


class TestTickInterval(unittest.TestCase):
    def test_multipliers(self):
        cfg = PbtConsolidationConfig(enabled=True)
        base = 100_000_000
        self.assertEqual(
            effective_tick_interval(base, STAGE_EXPLORATION, cfg), base
        )
        self.assertEqual(
            effective_tick_interval(base, STAGE_CONSOLIDATION, cfg),
            200_000_000,
        )
        self.assertEqual(
            effective_tick_interval(base, STAGE_SELECTION, cfg),
            400_000_000,
        )


class TestConsolidatedTick(unittest.TestCase):
    def test_stage2_slower_tick(self):
        cfg = PbtConsolidationConfig(enabled=True)
        interval = 100_000_000
        raw = cfg.stage1_end_frames + 50_000_000
        should, derived, _, stage, eff = evaluate_consolidated_tick(
            raw_frame=raw,
            self_pbt_iteration=0,
            last_mutation_frame=0,
            start_after=0,
            initial_delay=0,
            interval_steps=interval,
            cfg=cfg,
        )
        self.assertEqual(stage, STAGE_CONSOLIDATION)
        self.assertEqual(eff, 200_000_000)
        self.assertTrue(should)
        self.assertEqual(derived, raw // eff)


class TestBottomRestart(unittest.TestCase):
    def setUp(self):
        self.cfg = PbtConsolidationConfig(enabled=True)

    def test_exploration_allows(self):
        ctx = BottomRestartContext(promising_protected=True)
        allow, reason = should_allow_bottom_restart(
            stage=STAGE_EXPLORATION, ctx=ctx, cfg=self.cfg
        )
        self.assertTrue(allow)
        self.assertEqual(reason, "")

    def test_consolidation_blocks_promising(self):
        ctx = BottomRestartContext(promising_protected=True)
        allow, reason = should_allow_bottom_restart(
            stage=STAGE_CONSOLIDATION, ctx=ctx, cfg=self.cfg
        )
        self.assertFalse(allow)
        self.assertEqual(reason, "consolidation_promising_protected")

    def test_consolidation_retention_block(self):
        ctx = BottomRestartContext(
            nsucc_mean=22.0,
            recent_peak_nsucc_mean=25.0,
        )
        allow, reason = should_allow_bottom_restart(
            stage=STAGE_CONSOLIDATION, ctx=ctx, cfg=self.cfg
        )
        self.assertFalse(allow)
        self.assertEqual(reason, "consolidation_retention_protected")

    def test_selection_hard_collapse_only(self):
        ctx = BottomRestartContext(nsucc_mean=10.0, hard_collapse=False)
        allow, reason = should_allow_bottom_restart(
            stage=STAGE_SELECTION, ctx=ctx, cfg=self.cfg
        )
        self.assertFalse(allow)
        self.assertEqual(reason, "selection_hard_collapse_only")

    def test_selection_allows_hard_collapse(self):
        ctx = BottomRestartContext(hard_collapse=True, collapse_override=True)
        allow, reason = should_allow_bottom_restart(
            stage=STAGE_SELECTION, ctx=ctx, cfg=self.cfg
        )
        self.assertTrue(allow)

    def test_selection_allows_low_nsucc(self):
        ctx = BottomRestartContext(nsucc_mean=0.5)
        allow, reason = should_allow_bottom_restart(
            stage=STAGE_SELECTION, ctx=ctx, cfg=self.cfg
        )
        self.assertTrue(allow)

    def test_stage_ids(self):
        self.assertEqual(stage_id(STAGE_EXPLORATION), 0)
        self.assertEqual(stage_id(STAGE_CONSOLIDATION), 1)
        self.assertEqual(stage_id(STAGE_SELECTION), 2)


class TestMutationRate(unittest.TestCase):
    def test_stage_multipliers(self):
        cfg = PbtConsolidationConfig(enabled=True)
        self.assertAlmostEqual(
            effective_mutation_rate(0.2, STAGE_CONSOLIDATION, cfg), 0.1
        )
        self.assertAlmostEqual(
            effective_mutation_rate(0.2, STAGE_SELECTION, cfg), 0.0
        )


if __name__ == "__main__":
    unittest.main()
