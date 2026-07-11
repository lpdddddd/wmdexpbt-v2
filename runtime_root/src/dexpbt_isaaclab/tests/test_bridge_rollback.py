from dexpbt_isaaclab.pbt.pbt_tolerance_bridge import (
    next_bridge_tolerance,
    previous_bridge_tolerance,
)


def test_rollback_moves_one_ladder_level_and_uses_no_cross_tol_peak():
    # Bridge rollback target is exactly the previous (easier) ladder rung.
    assert previous_bridge_tolerance(0.0195) == 0.020
    assert previous_bridge_tolerance(0.020) == 0.0205
    assert previous_bridge_tolerance(0.0205) == 0.021


def test_rollback_then_shrink_are_inverse_single_steps():
    tol = 0.020
    up = previous_bridge_tolerance(tol)
    assert up == 0.0205
    # After rollback the very next shrink step returns to the same rung, never
    # skipping straight past 2cm.
    assert next_bridge_tolerance(up) == 0.020
