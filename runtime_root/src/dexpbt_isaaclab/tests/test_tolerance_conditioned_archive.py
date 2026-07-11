import json

from dexpbt_isaaclab.pbt.tolerance_conditioned_archive import select_compatible_archive


def test_selects_only_compatible_existing_archive_metadata(tmp_path):
    for name, tol, mean in (("wide", .023, 42), ("near", .0205, 30), ("strict", .019, 35)):
        (tmp_path / f"{name}.json").write_text(json.dumps({
            "nsucc_mean": mean, "curriculum": {"success_tolerance": tol},
            "same_tol_retention_at_save": .9,
        }))
    picked = select_compatible_archive(
        tmp_path, target_tolerance=.020, max_gap=.001, min_mean=28, min_retention=.85
    )
    assert picked["nsucc_mean"] == 30


def test_old_metadata_without_tolerance_is_ignored(tmp_path):
    (tmp_path / "old.json").write_text(json.dumps({"nsucc_mean": 99}))
    assert select_compatible_archive(tmp_path, target_tolerance=.02, max_gap=.002) is None
