from dexpbt_isaaclab.pbt.pbt_consolidation_schedule import (
    BottomRestartContext,
    PbtConsolidationConfig,
    STAGE_SELECTION,
    should_allow_bottom_restart,
)


def _cfg():
    return PbtConsolidationConfig(enabled=True, stage3_hard_collapse_only=True)


def test_stage3_blocks_moderate_degradation_without_rescue():
    ctx = BottomRestartContext(nsucc_mean=20.0, recent_peak_nsucc_mean=42.0, tier="bottom")
    allow, reason = should_allow_bottom_restart(stage=STAGE_SELECTION, ctx=ctx, cfg=_cfg())
    assert not allow
    assert reason == "selection_hard_collapse_only"


def test_stage3_allows_qualified_late_rescue_bypass():
    ctx = BottomRestartContext(
        nsucc_mean=20.0, recent_peak_nsucc_mean=42.0, tier="bottom",
        late_degradation_rescue=True,
    )
    allow, reason = should_allow_bottom_restart(stage=STAGE_SELECTION, ctx=ctx, cfg=_cfg())
    assert allow
    assert reason == "late_degradation_rescue"


def test_rescue_does_not_mutate_hyperparameters_contract():
    # The rescue path only flips eligibility; there is no mutation knob in the
    # schedule context, guaranteeing the copy path reuses source params.
    ctx = BottomRestartContext(late_degradation_rescue=True, tier="bottom")
    assert not hasattr(ctx, "mutate_params")
