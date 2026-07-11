from dexpbt_isaaclab.pbt.pbt_tolerance_bridge import ToleranceBinRetentionTracker


def test_cross_tolerance_peak_is_not_used():
    tracker = ToleranceBinRetentionTracker()
    tracker.update(tolerance=0.023, nsucc_mean=42, frame=1, lineage_id="a")
    tracker.update(tolerance=0.020, nsucc_mean=20, frame=2, lineage_id="a")
    assert not tracker.is_valid(0.020, min_samples=3, warmup_ticks=3)
    for frame in range(3, 5):
        tracker.update(tolerance=0.020, nsucc_mean=20, frame=frame, lineage_id="a")
    assert tracker.is_valid(0.020, min_samples=3, warmup_ticks=3)
    assert tracker.retention(0.020) == 1.0


def test_same_bin_retention_and_lineage_reset_roundtrip():
    tracker = ToleranceBinRetentionTracker()
    tracker.update(tolerance=0.020, nsucc_mean=25, frame=1, lineage_id="a")
    tracker.update(tolerance=0.020, nsucc_mean=20, frame=2, lineage_id="a")
    assert tracker.retention(0.020) == 0.8
    restored = ToleranceBinRetentionTracker.from_json(tracker.to_json())
    assert restored.retention(0.020) == 0.8
    restored.update(tolerance=0.020, nsucc_mean=10, frame=3, lineage_id="b")
    assert restored.state(0.020).sample_count == 1
