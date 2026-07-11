from dexpbt_isaaclab.pbt.pbt_control_plane import (
    population_frame_target_reached,
    quantize_population_frame,
)


def test_target_uses_quantized_authoritative_clock():
    target = 5_000_000_000
    tick = 20_000_000
    assert not population_frame_target_reached(target - tick, target, tick)
    assert population_frame_target_reached(target, target, tick)
    assert quantize_population_frame(target + tick - 1, tick) == target


def test_attempt_local_reset_does_not_change_completed_snapshot_frame():
    # The stop helper consumes snapshot/observer population frames only.
    assert population_frame_target_reached(120, 100, 20)
