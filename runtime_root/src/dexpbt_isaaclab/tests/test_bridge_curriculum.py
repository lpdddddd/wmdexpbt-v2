from dexpbt_isaaclab.pbt.pbt_tolerance_bridge import (
    next_bridge_tolerance,
    previous_bridge_tolerance,
)


def test_ladder_never_skips_a_2cm_bridge_level():
    assert next_bridge_tolerance(0.0235) == 0.022
    assert next_bridge_tolerance(0.021) == 0.0205
    assert next_bridge_tolerance(0.0215) == 0.021
    assert next_bridge_tolerance(0.0205) == 0.020


def test_previous_level_is_bounded_to_one_rung():
    assert previous_bridge_tolerance(0.0195) == 0.020
    assert previous_bridge_tolerance(0.020) == 0.0205
