"""Static wiring checks for per-worker frame-stop completion handling."""
from pathlib import Path

SUP = Path(__file__).resolve().parents[1] / "run_train_supervisor.sh"


def test_supervisor_recognizes_completion_marker_before_epoch_crash_check():
    text = SUP.read_text()
    file_marker_idx = text.index("population_frame_complete.json")
    log_marker_idx = text.index("population_frame_target_reached")
    epoch_idx = text.index("treating as crash", text.index("stopped at epoch"))
    # Marker file first, then log grep, then epoch<max crash rule.
    assert file_marker_idx < log_marker_idx < epoch_idx
    assert "status=completed_population_frame" in text
    assert "marker file" in text


def test_supervisor_completion_sets_exit_zero():
    text = SUP.read_text()
    block = text[text.index("population_frame_complete.json"):]
    block = block[: block.index("elif")]
    assert "exit_code=0" in block
    assert "completed_population_frame" in block