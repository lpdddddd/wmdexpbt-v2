from dexpbt_isaaclab.pbt.pbt_tolerance_bridge import ToleranceBinRetentionTracker


def _warm(tracker, tol, mean, start, n, lineage="a"):
    for i in range(n):
        tracker.update(tolerance=tol, nsucc_mean=mean, frame=start + i, lineage_id=lineage)


def test_probation_bin_requires_warmup_before_valid_retention():
    tracker = ToleranceBinRetentionTracker()
    # First entry into the 0.0205 bin: not yet enough samples/ticks.
    tracker.update(tolerance=0.0205, nsucc_mean=30, frame=1, lineage_id="a")
    assert not tracker.is_valid(0.0205, min_samples=3, warmup_ticks=4)
    _warm(tracker, 0.0205, 30, 2, 4)
    assert tracker.is_valid(0.0205, min_samples=3, warmup_ticks=4)


def test_probation_bin_retention_is_local_not_cross_bin():
    tracker = ToleranceBinRetentionTracker()
    _warm(tracker, 0.023, 42, 1, 4)
    _warm(tracker, 0.0205, 30, 100, 4)
    # 0.0205 bin peak is 30, not 42 from the wider bin.
    assert tracker.state(0.0205).peak_nsucc_mean == 30
    assert tracker.retention(0.0205) == 1.0
