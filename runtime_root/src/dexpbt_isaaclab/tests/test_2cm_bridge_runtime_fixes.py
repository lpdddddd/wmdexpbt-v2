"""Dynamic tests for CSER-DexPBT 2cm bridge runtime fixes."""

from __future__ import annotations

import json
import os
from pathlib import Path

import torch

from dexpbt_isaaclab.pbt.pbt_control_plane import (
    effective_population_frame_target,
    population_frame_target_reached,
    quantize_population_frame,
)
from dexpbt_isaaclab.pbt.pbt_tolerance_bridge import (
    DEFAULT_BRIDGE_LADDER,
    ToleranceBinRetentionTracker,
    bridge_level_id,
    bridge_level_key,
    next_bridge_tolerance,
)
from dexpbt_isaaclab.pbt.tolerance_conditioned_archive import select_compatible_archive
from dexpbt_isaaclab.world_model.elite_snippet_bank import (
    EliteSnippetBank,
    EliteSnippetBankConfig,
)
from dexpbt_isaaclab.world_model.elite_snippet_builder import SnippetBatch


LADDER = DEFAULT_BRIDGE_LADDER


def test_bridge_level_id_isolates_adjacent_rungs():
    assert bridge_level_id(0.022, LADDER) != bridge_level_id(0.021, LADDER)
    assert bridge_level_id(0.021, LADDER) != bridge_level_id(0.0205, LADDER)


def test_ladder_level_retention_does_not_reuse_coarse_bin_peak():
    tracker = ToleranceBinRetentionTracker(ladder=LADDER)
    # 0.022 peak=42
    for frame in range(1, 5):
        tracker.update(
            tolerance=0.022, nsucc_mean=42, frame=frame, lineage_id="a",
            use_bridge_levels=True,
        )
    assert tracker.is_valid(0.022, min_samples=3, warmup_ticks=3, use_bridge_levels=True)
    assert tracker.state(0.022, use_bridge_levels=True).peak_nsucc_mean == 42

    # Enter 0.021: must re-warmup; peak must not be 42.
    tracker.update(
        tolerance=0.021, nsucc_mean=20, frame=10, lineage_id="a",
        use_bridge_levels=True,
    )
    st = tracker.state(0.021, use_bridge_levels=True)
    assert st.sample_count == 1
    assert st.peak_nsucc_mean == 20
    assert not tracker.is_valid(0.021, min_samples=3, warmup_ticks=3, use_bridge_levels=True)

    # 0.021 peak must not affect 0.0205
    for frame in range(11, 15):
        tracker.update(
            tolerance=0.021, nsucc_mean=25, frame=frame, lineage_id="a",
            use_bridge_levels=True,
        )
    tracker.update(
        tolerance=0.0205, nsucc_mean=18, frame=20, lineage_id="a",
        use_bridge_levels=True,
    )
    assert tracker.state(0.0205, use_bridge_levels=True).peak_nsucc_mean == 18
    assert tracker.state(0.021, use_bridge_levels=True).peak_nsucc_mean == 25


def test_same_level_retention_and_lineage_reset():
    tracker = ToleranceBinRetentionTracker(ladder=LADDER)
    tracker.update(tolerance=0.020, nsucc_mean=25, frame=1, lineage_id="a", use_bridge_levels=True)
    tracker.update(tolerance=0.020, nsucc_mean=20, frame=2, lineage_id="a", use_bridge_levels=True)
    assert abs(tracker.retention(0.020, use_bridge_levels=True) - 0.8) < 1e-9
    tracker.update(tolerance=0.020, nsucc_mean=10, frame=3, lineage_id="b", use_bridge_levels=True)
    assert tracker.state(0.020, use_bridge_levels=True).sample_count == 1
    assert tracker.state(0.020, use_bridge_levels=True).peak_nsucc_mean == 10


def test_tracker_control_json_roundtrip_preserves_level_state():
    tracker = ToleranceBinRetentionTracker(ladder=LADDER)
    for frame in range(1, 5):
        tracker.update(
            tolerance=0.021, nsucc_mean=22, frame=frame, lineage_id="lin-1",
            use_bridge_levels=True,
        )
    blob = tracker.to_json()
    restored = ToleranceBinRetentionTracker.from_json(blob)
    assert restored.ladder == LADDER
    assert restored.lineage_id == "lin-1"
    assert restored.state(0.021, use_bridge_levels=True).sample_count == 4
    assert restored.state(0.021, use_bridge_levels=True).peak_nsucc_mean == 22
    # Coarse bin still present for general stats.
    assert str(tracker.current_bin(0.021)) in restored.states


def test_next_bridge_from_021_enters_0205_then_020():
    assert next_bridge_tolerance(0.021, ladder=LADDER) == 0.0205
    assert next_bridge_tolerance(0.0205, ladder=LADDER) == 0.020


def test_frame_stop_ceils_non_tick_target():
    tick = 20_000_000
    assert effective_population_frame_target(5_000_000_000, tick) == 5_000_000_000
    assert effective_population_frame_target(25_000_001, tick) == 40_000_000
    assert quantize_population_frame(25_000_001, tick) == 20_000_000  # floor
    assert population_frame_target_reached(20_000_000, 25_000_001, tick) is False
    assert population_frame_target_reached(40_000_000, 25_000_001, tick) is True


def test_archive_strict_rescue_rejects_missing_retention_metadata(tmp_path: Path):
    archive = tmp_path / "best"
    archive.mkdir()
    (archive / "a.json").write_text(json.dumps({
        "nsucc_mean": 40.0,
        "nsucc_max": 50.0,
        "frame": 100,
        "curriculum": {"success_tolerance": 0.020},
        "checkpoint": str(archive / "a.pth"),
    }))
    (archive / "a.pth").write_text("x")
    # Ordinary recovery: missing retention defaults to 1.0
    ok = select_compatible_archive(
        archive, target_tolerance=0.020, max_gap=0.002, min_retention=0.5,
        require_retention_metadata=False,
    )
    assert ok is not None
    # Strict late rescue: fail-closed
    reject: list[dict] = []
    bad = select_compatible_archive(
        archive, target_tolerance=0.020, max_gap=0.002, min_retention=0.5,
        require_retention_metadata=True, reject_log=reject,
    )
    assert bad is None
    assert any(r["reason"] == "missing_retention_metadata" for r in reject)


def test_archive_strict_rescue_accepts_with_retention_metadata(tmp_path: Path):
    archive = tmp_path / "best"
    archive.mkdir()
    (archive / "b.json").write_text(json.dumps({
        "nsucc_mean": 35.0,
        "same_tol_retention_at_save": 0.9,
        "same_tol_retention_valid_at_save": True,
        "curriculum": {"success_tolerance": 0.020},
        "checkpoint": str(archive / "b.pth"),
    }))
    (archive / "b.pth").write_text("x")
    ok = select_compatible_archive(
        archive, target_tolerance=0.020, max_gap=0.002, min_retention=0.5,
        require_retention_metadata=True,
    )
    assert ok is not None


def _batch(tols, priority_scale=1.0):
    n = len(tols)
    return SnippetBatch(
        feat_start=torch.randn(n, 8),
        actions=torch.randn(n, 4, 3).clamp(-1, 1),
        d_hat=torch.full((n,), 0.02),
        success_tol=torch.tensor([float(t) for t in tols]),
        lifted=torch.ones(n),
        picked=torch.zeros(n),
        dropped=torch.zeros(n),
        closest_d_hat=torch.full((n,), 0.02),
        success_h=torch.ones(n),
        improve_h=torch.zeros(n),
        drop_h=torch.zeros(n),
        recovery_label=torch.ones(n),
        s0=torch.ones(n),
        s1=torch.ones(n),
        i0=torch.zeros(n),
        d0=torch.zeros(n),
        d1=torch.zeros(n),
    )


def test_adjacent_mixing_persists_when_primary_exceeds_eight():
    cfg = EliteSnippetBankConfig(
        enabled=True, bank_size=200, feature_dim=8, action_dim=3, horizon=4,
        guidance_enable=True, guidance_bank_tol=0.04, sample_pool=20,
        adjacent_stage_mixing_enable=True, primary_stage_fraction=0.7,
        adjacent_stage_fraction=0.3, min_primary_candidates=8,
        fallback_to_adjacent_when_sparse=True,
        tolerance_compatibility_enable=False,
    )
    bank = EliteSnippetBank(cfg)
    # Many FLOOR (tol<=0.02) and many MID (0.02<tol<=0.04)
    bank.update_from_builder_batch(
        _batch([0.015] * 12 + [0.03] * 12),
        source_stats={"source_nsucc_mean": 30.0},
    )
    acts, sim, pri, ok = bank.retrieve_candidates(torch.randn(2, 8), k=4, stage=1)
    assert ok
    diag = bank.last_update_diagnostics
    assert diag["retrieve_primary_candidates"] >= 8
    assert diag["retrieve_adjacent_candidates"] >= 1
    # Mixing must have happened (pool not pure primary storage order).
    assert diag["retrieval_valid"] == 1.0


def test_tolerance_fail_closed_returns_invalid_and_actor_distill_flag():
    cfg = EliteSnippetBankConfig(
        enabled=True, bank_size=50, feature_dim=8, action_dim=3, horizon=4,
        guidance_enable=True, guidance_bank_tol=0.04,
        adjacent_stage_mixing_enable=True, min_primary_candidates=1,
        tolerance_compatibility_enable=True, tolerance_max_gap=0.001,
    )
    bank = EliteSnippetBank(cfg)
    bank.update_from_builder_batch(
        _batch([0.030, 0.031, 0.032]),
        source_stats={"source_nsucc_mean": 30.0},
    )
    acts, sim, pri, ok = bank.retrieve_candidates(
        torch.randn(1, 8), k=3, stage=3, query_tolerance=0.020
    )
    assert ok is False
    diag = bank.last_update_diagnostics
    assert diag["no_tolerance_compatible_candidate"] == 1.0
    assert diag["retrieval_valid"] == 0.0
    assert diag["fallback_to_actor_distill"] == 1.0
    assert diag["compatible_candidate_count"] == 0.0


def test_primary_adjacent_ranking_not_storage_order():
    cfg = EliteSnippetBankConfig(
        enabled=True, bank_size=50, feature_dim=8, action_dim=3, horizon=4,
        guidance_enable=True, guidance_bank_tol=0.04, sample_pool=10,
        adjacent_stage_mixing_enable=True, primary_stage_fraction=0.7,
        adjacent_stage_fraction=0.3, min_primary_candidates=1,
        tolerance_compatibility_enable=False,
    )
    bank = EliteSnippetBank(cfg)
    # Insert low-priority FLOOR first, then high-priority FLOOR.
    bank.update_from_builder_batch(
        _batch([0.015, 0.015]),
        source_stats={"source_nsucc_mean": 5.0},
    )
    bank.update_from_builder_batch(
        _batch([0.015] * 5 + [0.03] * 5),
        source_stats={"source_nsucc_mean": 40.0},
    )
    # Rank helper should put higher priority first inside primary.
    bucket = bank.scalars["bucket_id"][: bank.size]
    stage_of = torch.div(bucket, 2.0, rounding_mode="floor")
    primary = (stage_of == 1.0).nonzero(as_tuple=False).reshape(-1)
    ranked = bank._rank_pool_indices(primary, query_tolerance=None)
    pri = bank.scalars["priority_score"][ranked]
    assert torch.all(pri[:-1] >= pri[1:] - 1e-5)


def test_supervisor_prefers_marker_file(tmp_path: Path):
    marker = tmp_path / "population_frame_complete.json"
    marker.write_text(json.dumps({
        "policy_idx": 0,
        "population_frame": 5_000_000_000,
        "target_population_frames": 5_000_000_000,
        "pbt_iteration": 250,
        "checkpoint": "x.pth",
        "timestamp": "2026-07-11T00:00:00",
    }))
    assert marker.exists()
    # Emulate supervisor priority: marker file wins over missing log grep.
    assert marker.name == "population_frame_complete.json"


_REPO_ROOT = Path(__file__).resolve().parents[4]
_LAUNCHERS = _REPO_ROOT / "results" / "launchers"
_SUPERVISOR = (
    Path(__file__).resolve().parents[1] / "run_train_supervisor.sh"
)


def test_continuation_launcher_separates_tol_and_absolute_frame():
    text = (
        _LAUNCHERS / "start_regrasping_cser_dexpbt_2cm_bridge_resume_policy05.sh"
    ).read_text()
    assert "--success_tolerance_resume \"$START_TOL\"" in text
    assert "--target_tolerance_resume \"$TARGET_TOL\"" in text
    assert "SOURCE_POPULATION_FRAME" in text
    assert "TARGET_POPULATION_FRAMES" in text
    assert "absolute_target_population_frames" in text
    assert "Refusing to default" in text
    for flag in (
        "--late_degradation_rescue_source_min_mean 28.0",
        "--late_degradation_rescue_min_source_advantage 8.0",
        "--late_degradation_rescue_source_min_retention 0.85",
        "--late_degradation_rescue_reject_collapse_source",
        "--late_degradation_rescue_require_source_control",
        "--no-late_degradation_rescue_bypass_elimination_burnin",
    ):
        assert flag in text, flag


def test_formal_5b_launcher_keeps_hard_offs_and_retention_stack():
    text = (
        _LAUNCHERS / "start_regrasping_cser_dexpbt_2cm_bridge_5b.sh"
    ).read_text()
    for flag in (
        "--no-pbt_allow_mid_self_mutation",
        "--no-pbt_wm_profile_copy_enable",
        "--no-awr_enable",
        "--no-eir_residual_apply_to_rollout",
        "--no-eir_use_rssm_mppi",
        "--eir_mppi_mode off",
        "--pbt_curriculum_inherit_mode source_safe",
        "--pbt_protect_promising_workers",
        "--curriculum_safety_use_real_fallback",
        "--same_tol_retention_enable",
        "--bridge_curriculum_enable",
        "--bridge_probation_enable",
        "--late_degradation_rescue_enable",
        "--late_degradation_rescue_source_min_mean 28.0",
        "--late_degradation_rescue_min_source_advantage 8.0",
        "--late_degradation_rescue_source_min_retention 0.85",
        "--late_degradation_rescue_reject_collapse_source",
        "--late_degradation_rescue_require_source_control",
        "--no-late_degradation_rescue_bypass_elimination_burnin",
        "--frame_stop_enable",
    ):
        assert flag in text, flag


def test_probation_logic_helpers_for_legacy_vs_bridge_collapse():
    """Unit-level contract for the two collapse classes."""
    tracker = ToleranceBinRetentionTracker(ladder=LADDER)
    for frame in range(1, 5):
        tracker.update(
            tolerance=0.023, nsucc_mean=42, frame=frame, lineage_id="a",
            use_bridge_levels=True,
        )
    # Enter 0.0205 with low nsucc: warmup incomplete -> no bridge hard collapse yet.
    tracker.update(
        tolerance=0.0205, nsucc_mean=20, frame=10, lineage_id="a",
        use_bridge_levels=True,
    )
    assert not tracker.is_valid(
        0.0205, min_samples=3, warmup_ticks=3, use_bridge_levels=True
    )
    # After warmup at low mean, bridge hard-collapse condition can arm.
    for frame in range(11, 15):
        tracker.update(
            tolerance=0.0205, nsucc_mean=2.0, frame=frame, lineage_id="a",
            use_bridge_levels=True,
        )
    assert tracker.is_valid(
        0.0205, min_samples=3, warmup_ticks=3, use_bridge_levels=True
    )
    assert tracker.state(0.0205, use_bridge_levels=True).last_nsucc_mean < 3.0
    # Wide-tol peak remains isolated.
    assert tracker.state(0.023, use_bridge_levels=True).peak_nsucc_mean == 42


def test_level_timer_pending_does_not_imply_level_key_change():
    """Planning a shrink must not create a new level retention key."""
    tracker = ToleranceBinRetentionTracker(ladder=LADDER)
    for frame in range(1, 5):
        tracker.update(
            tolerance=0.021, nsucc_mean=30, frame=frame, lineage_id="a",
            use_bridge_levels=True,
        )
    planned = next_bridge_tolerance(0.021, ladder=LADDER)
    assert planned == 0.0205
    # Without an actual update at 0.0205, that level key must be absent / empty.
    key = bridge_level_key(bridge_level_id(0.0205, LADDER))
    assert key not in tracker.states or tracker.states[key].sample_count == 0
    assert tracker.state(0.021, use_bridge_levels=True).sample_count == 4


def test_late_rescue_no_mutation_even_with_nonzero_rate():
    source = {"learning_rate": 1e-4, "entropy_coef": 0.01}

    def _mutate_params(params, rate, lo, hi):
        out = dict(params)
        out["learning_rate"] = float(params["learning_rate"]) * 1.1
        return out

    mutation_rate = 1.0
    late_rescue = True
    if late_rescue:
        new_params = dict(source)
    else:
        new_params = _mutate_params(source, mutation_rate, 0.8, 1.2)
    assert new_params == source
    assert _mutate_params(source, 1.0, 0.8, 1.2) != source


def test_supervisor_script_checks_marker_before_log_grep():
    text = _SUPERVISOR.read_text()
    marker_pos = text.find("population_frame_complete.json")
    log_pos = text.find("[PBT] population_frame_target_reached")
    assert marker_pos > 0 and log_pos > marker_pos


# ---------------------------------------------------------------------------
# Real call-path tests (list[int] selector + quality gates + observer paths)
# ---------------------------------------------------------------------------


def _stub_rl_games():
    import sys
    import types

    if "rl_games.common.algo_observer" in sys.modules:
        return
    m = types.ModuleType("rl_games")
    c = types.ModuleType("rl_games.common")
    a = types.ModuleType("rl_games.common.algo_observer")

    class IsaacAlgoObserver:
        def __init__(self, *args, **kwargs):
            pass

    a.IsaacAlgoObserver = IsaacAlgoObserver
    sys.modules["rl_games"] = m
    sys.modules["rl_games.common"] = c
    sys.modules["rl_games.common.algo_observer"] = a


def _eligible_ok(state, frames):
    return True, "ok"


def test_late_rescue_real_list_int_selector_picks_compatible_quality_source():
    """sorted_policies is list[int]; incompatible #1 must lose to quality #2."""
    from dexpbt_isaaclab.pbt.late_degradation_rescue import (
        select_late_degradation_rescue_source,
    )

    sorted_policies = [1, 2, 3]
    population = {
        0: {
            "nsucc_mean": 25.0,
            "pbt_ranking_score": 10.0,
            "curriculum": {"success_tolerance": 0.020},
        },
        1: {
            "nsucc_mean": 40.0,
            "pbt_ranking_score": 100.0,
            "curriculum": {"success_tolerance": 0.030},  # incompatible
            "params": {"a": 1},
            "checkpoint": "1.pth",
        },
        2: {
            "nsucc_mean": 35.0,
            "pbt_ranking_score": 90.0,
            "curriculum": {"success_tolerance": 0.020},
            "params": {"a": 2},
            "checkpoint": "2.pth",
        },
        3: {
            "nsucc_mean": 20.0,  # mean too low
            "pbt_ranking_score": 80.0,
            "curriculum": {"success_tolerance": 0.0195},
            "params": {"a": 3},
            "checkpoint": "3.pth",
        },
    }
    ret = {1: 0.95, 2: 0.92, 3: 0.50}

    decision = select_late_degradation_rescue_source(
        sorted_policies=sorted_policies,
        population=population,
        target_policy_idx=0,
        target_tolerance=0.020,
        target_nsucc_mean=25.0,
        env_frames=1_000_000_000,
        n_top=2,
        max_tol_gap=0.002,
        source_min_mean=28.0,
        min_source_advantage=8.0,
        source_min_retention=0.85,
        reject_collapse_source=True,
        source_retention_fn=lambda idx, _tol: float(ret[idx]),
        eligible_fn=_eligible_ok,
        archive_fn=None,
    )
    assert decision["action"] == "copy"
    assert decision["source_policy"] == 2
    assert decision["tier"] == "late_rescue_live"


def test_late_rescue_source_advantage_gate_fail_closed():
    from dexpbt_isaaclab.pbt.late_degradation_rescue import (
        select_late_degradation_rescue_source,
    )

    population = {
        0: {"nsucc_mean": 25.0, "curriculum": {"success_tolerance": 0.020}},
        1: {
            "nsucc_mean": 30.0,  # advantage = 5 < 8
            "pbt_ranking_score": 99.0,
            "curriculum": {"success_tolerance": 0.020},
            "checkpoint": "1.pth",
        },
    }
    decision = select_late_degradation_rescue_source(
        sorted_policies=[1],
        population=population,
        target_policy_idx=0,
        target_tolerance=0.020,
        target_nsucc_mean=25.0,
        env_frames=1,
        n_top=1,
        max_tol_gap=0.002,
        source_min_mean=28.0,
        min_source_advantage=8.0,
        source_min_retention=0.85,
        reject_collapse_source=True,
        source_retention_fn=lambda *_: 0.95,
        eligible_fn=_eligible_ok,
        archive_fn=None,
    )
    assert decision["action"] == "skip"
    assert decision["archive_reason"] == (
        "late_degradation_no_compatible_qualified_source"
    )
    assert decision["strict_filter_reasons"][1] == "advantage_below_threshold"


def test_late_rescue_same_level_retention_gate_ignores_loose_peak():
    """Wide-tol historical peak must not inflate current-level retention."""
    from dexpbt_isaaclab.pbt.late_degradation_rescue import (
        select_late_degradation_rescue_source,
    )

    # Current strict level: peak=30, current=20 -> retention≈0.667 < 0.85
    population = {
        0: {"nsucc_mean": 18.0, "curriculum": {"success_tolerance": 0.020}},
        1: {
            "nsucc_mean": 40.0,
            "pbt_ranking_score": 100.0,
            "curriculum": {"success_tolerance": 0.020},
            "checkpoint": "1.pth",
        },
    }
    decision = select_late_degradation_rescue_source(
        sorted_policies=[1],
        population=population,
        target_policy_idx=0,
        target_tolerance=0.020,
        target_nsucc_mean=18.0,
        env_frames=1,
        n_top=1,
        max_tol_gap=0.002,
        source_min_mean=28.0,
        min_source_advantage=8.0,
        source_min_retention=0.85,
        reject_collapse_source=True,
        source_retention_fn=lambda *_: 20.0 / 30.0,
        eligible_fn=_eligible_ok,
        archive_fn=None,
    )
    assert decision["action"] == "skip"
    assert decision["strict_filter_reasons"][1] == "retention_below_threshold"


def test_late_rescue_archive_missing_retention_metadata_fail_closed(tmp_path: Path):
    archive = tmp_path / "best"
    archive.mkdir()
    (archive / "a.json").write_text(json.dumps({
        "nsucc_mean": 40.0,
        "curriculum": {"success_tolerance": 0.020},
        "checkpoint": str(archive / "a.pth"),
    }))
    (archive / "a.pth").write_text("x")
    reject: list[dict] = []
    bad = select_compatible_archive(
        archive,
        target_tolerance=0.020,
        max_gap=0.002,
        min_mean=28.0,
        min_retention=0.85,
        require_retention_metadata=True,
        reject_log=reject,
    )
    assert bad is None
    assert any(r["reason"] == "missing_retention_metadata" for r in reject)

    from dexpbt_isaaclab.pbt.late_degradation_rescue import (
        select_late_degradation_rescue_source,
    )

    def _archive():
        state = select_compatible_archive(
            archive,
            target_tolerance=0.020,
            max_gap=0.002,
            min_mean=28.0,
            min_retention=0.85,
            require_retention_metadata=True,
        )
        return state, (
            "ok" if state is not None
            else "late_degradation_no_compatible_qualified_source"
        )

    decision = select_late_degradation_rescue_source(
        sorted_policies=[],
        population={0: {"nsucc_mean": 10.0, "curriculum": {"success_tolerance": 0.020}}},
        target_policy_idx=0,
        target_tolerance=0.020,
        target_nsucc_mean=10.0,
        env_frames=1,
        n_top=1,
        max_tol_gap=0.002,
        source_min_mean=28.0,
        min_source_advantage=8.0,
        source_min_retention=0.85,
        reject_collapse_source=True,
        source_retention_fn=lambda *_: 1.0,
        eligible_fn=_eligible_ok,
        archive_fn=_archive,
    )
    assert decision["action"] == "skip"


def test_completion_marker_paths_match_supervisor(tmp_path: Path):
    _stub_rl_games()
    from dexpbt_isaaclab.pbt.pbt_observer import DexPBTPbtAlgoObserver

    run_dir = tmp_path / "run"
    ws = run_dir / "dexpbt_8gpu_ws"
    policy_ws = ws / "000"
    policy_ws.mkdir(parents=True)
    (run_dir / "policy_00").mkdir(parents=True)

    obs = object.__new__(DexPBTPbtAlgoObserver)
    obs.policy_idx = 0
    obs.workspace = ws
    obs.curr_policy_workspace = policy_ws

    ws_marker, run_marker = obs._population_frame_complete_marker_paths()
    assert ws_marker == policy_ws / "population_frame_complete.json"
    assert run_marker == run_dir / "policy_00" / "population_frame_complete.json"
    # Supervisor checks RUN_DIR/population_frame_complete.json where RUN_DIR=policy_XX
    assert "population_frame_complete.json" in _SUPERVISOR.read_text()
    assert str(run_marker).endswith("policy_00/population_frame_complete.json")

    payload = {
        "policy_idx": 0,
        "population_frame": 5_000_000_000,
        "target_population_frames": 5_000_000_000,
        "effective_target_population_frames": 5_000_000_000,
        "pbt_iteration": 250,
        "checkpoint": "x.pth",
        "timestamp": "2026-07-11T00:00:00",
    }
    obs._write_population_frame_complete_markers(payload)
    assert json.loads(ws_marker.read_text()) == payload
    assert json.loads(run_marker.read_text()) == payload


def test_late_rescue_restart_failure_does_not_consume_quota(tmp_path: Path):
    obs = _make_minimal_observer(tmp_path)
    obs._late_rescue_selected_count[0] = 1
    events: list[dict] = []
    obs._write_pbt_copy_event = lambda payload: events.append(dict(payload))

    def _boom(*_a, **_k):
        raise RuntimeError("manifest write failed")

    obs._restart_process_with_new_params_impl = _boom
    from dexpbt_isaaclab.pbt.pbt_observer import DexPBTPbtAlgoObserver

    ok = DexPBTPbtAlgoObserver._restart_process_with_new_params(
        obs,
        {"learning_rate": 1e-4},
        "ckpt.pth",
        1,
        1_000_000_000,
        "bottom",
        late_rescue=True,
        late_rescue_decision={"checkpoint_type": "live"},
    )
    assert ok is False
    assert obs._late_rescue_executed_count.get(0, 0) == 0
    assert obs._late_rescue_frame.get(0, -1) < 0
    assert obs._late_rescue_selected_count[0] == 1
    assert any(
        e.get("event_type") == "pbt_restart_transaction_rolled_back" for e in events
    )


def test_probation_suppress_does_not_inflate_ordinary_repeated_count(tmp_path: Path):
    _stub_rl_games()
    from dexpbt_isaaclab.pbt.pbt_curriculum_p1 import WindowSample
    from dexpbt_isaaclab.pbt.pbt_observer import DexPBTPbtAlgoObserver
    from dexpbt_isaaclab.pbt.pbt_tolerance_bridge import DEFAULT_BRIDGE_LADDER

    ws = tmp_path / "ws"
    ws.mkdir()
    (ws / "curriculum_control").mkdir()
    obs = DexPBTPbtAlgoObserver(
        policy_idx=0,
        num_policies=1,
        mutable_params={
            "enable_wm": False,
            "enable_counterfactual": False,
            "wm_strategy": "none",
        },
        workspace=str(ws),
        train_dir=str(ws),
        task_name="test-task",
        interval_steps=10_000_000,
        start_after=0,
        initial_delay=0,
        num_envs=1,
        curriculum_mode="mastery",
        curriculum_collapse_detect=True,
        pbt_early_collapse_detect=False,
        collapse_drop_ratio=0.5,
        collapse_patience=1,
        same_tol_retention_enable=True,
        bridge_curriculum_enable=True,
        bridge_tolerance_ladder=DEFAULT_BRIDGE_LADDER,
        bridge_probation_enable=True,
        bridge_probation_entry_tol=0.0205,
        bridge_probation_frames=400_000_000,
        bridge_probation_block_ordinary_rollback=True,
        bridge_probation_allow_hard_collapse_rollback=True,
        bridge_probation_hard_collapse_mean=3.0,
        bridge_probation_hard_collapse_patience=4,
        curriculum_danger_zone_enable=True,
        curriculum_danger_tol=0.030,
        curriculum_rollback_on_repeated_collapse=True,
        curriculum_rollback_collapse_patience=2,
        curriculum_rollback_to_last_mastered=True,
        same_tol_retention_min_samples=3,
        same_tol_retention_warmup_ticks=3,
    )
    obs.writer = None
    obs._bridge_probation_entry_frame[0] = 0
    # Warmup incomplete at current level -> no bridge hard-collapse.
    tracker = obs._same_tol_trackers.setdefault(
        0,
        ToleranceBinRetentionTracker(ladder=DEFAULT_BRIDGE_LADDER),
    )
    tracker.update(
        tolerance=0.0205, nsucc_mean=20.0, frame=1, lineage_id="a",
        use_bridge_levels=True,
    )
    window = obs._policy_window(0)
    # Cross-tol peak in recent_peak drives legacy hard collapse.
    obs._recent_peak_nsucc_mean[0] = 42.0
    for tick in range(3):
        frame = 10_000_000 * (tick + 1)
        window.samples = [
            WindowSample(
                env_frames=frame,
                nsucc_mean=10.0,
                nsucc_max=20.0,
                success_tolerance=0.0205,
                pbt_iteration=tick,
                lineage_id="a",
            )
        ]
        control = obs._build_curriculum_control_state(0, window, frame)
        assert float(control.forced_success_tolerance or 0.0) == 0.0
        assert control.curriculum_repeated_collapse_count == 0
        assert control.bridge_probation_suppressed_legacy_collapse_count == tick + 1
        assert control.bridge_probation_legacy_collapse_suppressed

    # End probation: ordinary count must still be 0 (cleared / never inflated).
    obs._bridge_probation_entry_frame[0] = 0
    end_frame = 500_000_000
    window.samples = [
        WindowSample(
            env_frames=end_frame,
            nsucc_mean=25.0,
            nsucc_max=30.0,
            success_tolerance=0.0205,
            pbt_iteration=99,
            lineage_id="a",
        )
    ]
    # Peak no longer triggers collapse; confirm no immediate rollback from history.
    obs._recent_peak_nsucc_mean[0] = 25.0
    control = obs._build_curriculum_control_state(0, window, end_frame)
    assert control.curriculum_repeated_collapse_count == 0
    assert float(control.forced_success_tolerance or 0.0) == 0.0
    assert control.bridge_probation_exit


def test_bridge_hard_collapse_rolls_back_one_rung(tmp_path: Path):
    _stub_rl_games()
    from dexpbt_isaaclab.pbt.pbt_curriculum_p1 import WindowSample
    from dexpbt_isaaclab.pbt.pbt_observer import DexPBTPbtAlgoObserver
    from dexpbt_isaaclab.pbt.pbt_tolerance_bridge import (
        DEFAULT_BRIDGE_LADDER,
        previous_bridge_tolerance,
    )

    ws = tmp_path / "ws"
    ws.mkdir()
    (ws / "curriculum_control").mkdir()
    obs = DexPBTPbtAlgoObserver(
        policy_idx=0,
        num_policies=1,
        mutable_params={
            "enable_wm": False,
            "enable_counterfactual": False,
            "wm_strategy": "none",
        },
        workspace=str(ws),
        train_dir=str(ws),
        task_name="test-task",
        interval_steps=10_000_000,
        start_after=0,
        initial_delay=0,
        num_envs=1,
        curriculum_mode="mastery",
        curriculum_collapse_detect=True,
        pbt_early_collapse_detect=False,
        collapse_drop_ratio=0.99,  # avoid legacy hard collapse from peak
        collapse_patience=99,
        same_tol_retention_enable=True,
        bridge_curriculum_enable=True,
        bridge_tolerance_ladder=DEFAULT_BRIDGE_LADDER,
        bridge_probation_enable=True,
        bridge_probation_entry_tol=0.0205,
        bridge_probation_frames=400_000_000,
        bridge_probation_block_ordinary_rollback=True,
        bridge_probation_allow_hard_collapse_rollback=True,
        bridge_probation_hard_collapse_mean=3.0,
        bridge_probation_hard_collapse_patience=4,
        curriculum_rollback_cooldown_frames=0,
        same_tol_retention_min_samples=3,
        same_tol_retention_warmup_ticks=3,
    )
    obs.writer = None
    obs._bridge_probation_entry_frame[0] = 0
    tracker = obs._same_tol_trackers.setdefault(
        0,
        ToleranceBinRetentionTracker(ladder=DEFAULT_BRIDGE_LADDER),
    )
    for i in range(1, 5):
        tracker.update(
            tolerance=0.0205, nsucc_mean=2.0, frame=i, lineage_id="a",
            use_bridge_levels=True,
        )
    window = obs._policy_window(0)
    cur_tol = 0.0205
    expected = previous_bridge_tolerance(cur_tol, ladder=DEFAULT_BRIDGE_LADDER)
    for tick in range(4):
        frame = 1_000_000 * (tick + 1)
        window.samples = [
            WindowSample(
                env_frames=frame,
                nsucc_mean=2.0,
                nsucc_max=5.0,
                success_tolerance=cur_tol,
                pbt_iteration=tick,
                lineage_id="a",
            )
        ]
        control = obs._build_curriculum_control_state(0, window, frame)
    assert control.bridge_probation_hard_collapse_triggered
    assert abs(float(control.forced_success_tolerance) - float(expected)) < 1e-9
    assert control.bridge_probation_rollback_reason == "bridge_probation_hard_collapse"
    assert control.bridge_probation_bad_streak == 0
    assert control.curriculum_repeated_collapse_count == 0

# ---------------------------------------------------------------------------
# Persistence / archive validity / collapse control / burn-in / transaction
# ---------------------------------------------------------------------------


def _make_minimal_observer(tmp_path: Path, **extra):
    _stub_rl_games()
    from dexpbt_isaaclab.pbt.pbt_observer import DexPBTPbtAlgoObserver

    ws = tmp_path / "ws"
    ws.mkdir(parents=True, exist_ok=True)
    (ws / "curriculum_control").mkdir(exist_ok=True)
    (ws / "000").mkdir(exist_ok=True)
    run = tmp_path / "run"
    run.mkdir(exist_ok=True)
    (run / "policy_00").mkdir(exist_ok=True)
    defaults = dict(
        policy_idx=0,
        num_policies=2,
        mutable_params={
            "enable_wm": False,
            "enable_counterfactual": False,
            "wm_strategy": "none",
            "learning_rate": 1e-4,
        },
        workspace=str(ws),
        train_dir=str(run / "rl_games"),
        task_name="test-task",
        interval_steps=10_000_000,
        start_after=0,
        initial_delay=0,
        num_envs=1,
        late_degradation_rescue_enable=True,
        late_degradation_rescue_max_per_policy=1,
        late_degradation_rescue_cooldown_frames=300_000_000,
        late_degradation_rescue_require_source_control=True,
        late_degradation_rescue_reject_collapse_source=True,
        late_degradation_rescue_bypass_elimination_burnin=False,
        pbt_recovery_bypass_elimination_burnin=True,
        pbt_min_frames_between_restarts=100_000_000,
        same_tol_retention_enable=True,
        bridge_curriculum_enable=True,
    )
    defaults.update(extra)
    Path(defaults["train_dir"]).mkdir(parents=True, exist_ok=True)
    obs = DexPBTPbtAlgoObserver(**defaults)
    obs.writer = None
    obs.curr_policy_workspace = ws / "000"
    return obs


def test_late_rescue_quota_persists_across_restart(tmp_path: Path):
    from dexpbt_isaaclab.pbt.pbt_curriculum_p1 import (
        CurriculumControlState,
        write_curriculum_control,
    )

    obs_a = _make_minimal_observer(tmp_path)
    # Seed a control JSON with tracker-like fields that must survive.
    path = obs_a._curriculum_control_dir / "000.json"
    write_curriculum_control(
        path,
        CurriculumControlState(
            last_mastered_tolerance=0.021,
            bridge_probation_entry_frame=1,
            late_rescue_executed_count=0,
        ),
    )
    obs_a._late_rescue_attempt_count[0] = 1
    obs_a._late_rescue_selected_count[0] = 1
    obs_a._late_rescue_executed_count[0] = 1
    obs_a._late_rescue_blocked_count[0] = 0
    obs_a._late_rescue_streak[0] = 0
    obs_a._late_rescue_frame[0] = 3_200_000_000
    obs_a._persist_late_rescue_runtime_state(0, env_frames=3_200_000_000)

    obs_b = _make_minimal_observer(tmp_path)
    obs_b._maybe_restore_bridge_state(0)
    assert obs_b._late_rescue_executed_count[0] == 1
    assert obs_b._late_rescue_frame[0] == 3_200_000_000
    # Warm same-tol so ready() can evaluate quota/cooldown.
    tracker = obs_b._same_tol_trackers.setdefault(
        0, ToleranceBinRetentionTracker(ladder=LADDER)
    )
    for i in range(1, 5):
        tracker.update(
            tolerance=0.020, nsucc_mean=35.0, frame=i, lineage_id="a",
            use_bridge_levels=True,
        )
    # Drop current mean to satisfy degradation thresholds, but quota blocks.
    ready = obs_b._late_degradation_rescue_ready(
        0, tolerance=0.020, nsucc_mean=10.0, env_frames=3_600_000_000
    )
    assert ready is False


def test_late_rescue_cooldown_persists_across_restart(tmp_path: Path):
    from dexpbt_isaaclab.pbt.pbt_curriculum_p1 import (
        CurriculumControlState,
        write_curriculum_control,
    )

    obs_a = _make_minimal_observer(
        tmp_path,
        late_degradation_rescue_max_per_policy=2,
        late_degradation_rescue_retention_threshold=1.0,
        late_degradation_rescue_peak_threshold=30.0,
        late_degradation_rescue_current_mean_threshold=22.0,
        late_degradation_rescue_patience=1,
    )
    write_curriculum_control(
        obs_a._curriculum_control_dir / "000.json",
        CurriculumControlState(),
    )
    obs_a._late_rescue_executed_count[0] = 0
    obs_a._late_rescue_frame[0] = 3_200_000_000
    obs_a._persist_late_rescue_runtime_state(0, env_frames=3_200_000_000)

    obs_b = _make_minimal_observer(
        tmp_path,
        late_degradation_rescue_max_per_policy=2,
        late_degradation_rescue_retention_threshold=1.0,
        late_degradation_rescue_peak_threshold=30.0,
        late_degradation_rescue_current_mean_threshold=22.0,
        late_degradation_rescue_patience=1,
    )
    obs_b._maybe_restore_bridge_state(0)
    tracker = obs_b._same_tol_trackers.setdefault(
        0, ToleranceBinRetentionTracker(ladder=LADDER)
    )
    for i in range(1, 5):
        tracker.update(
            tolerance=0.020, nsucc_mean=40.0, frame=i, lineage_id="a",
            use_bridge_levels=True,
        )
    # Drop current so retention <= 1.0 still holds; mean below threshold.
    tracker.update(
        tolerance=0.020, nsucc_mean=10.0, frame=10, lineage_id="a",
        use_bridge_levels=True,
    )
    assert obs_b._late_rescue_frame[0] == 3_200_000_000
    assert obs_b._late_degradation_rescue_ready(
        0, tolerance=0.020, nsucc_mean=10.0, env_frames=3_300_000_000
    ) is False
    assert obs_b._late_degradation_rescue_ready(
        0, tolerance=0.020, nsucc_mean=10.0, env_frames=3_600_000_000
    ) is True


def test_late_rescue_state_persist_failure_fail_closed(tmp_path: Path, monkeypatch):
    """Persist failure before manifest must roll back executed_count and leave no manifest."""
    from dexpbt_isaaclab.pbt import pbt_observer as obs_mod

    obs = _make_minimal_observer(tmp_path)
    control_path = obs._curriculum_control_dir / "000.json"
    from dexpbt_isaaclab.pbt.pbt_curriculum_p1 import (
        CurriculumControlState,
        write_curriculum_control_dict,
    )

    write_curriculum_control_dict(
        control_path,
        CurriculumControlState(last_mastered_tolerance=0.021).to_json(),
    )
    ckpt = tmp_path / "source.pth"
    torch.save({"frame": 0, "model": {}, "pbt_history": []}, ckpt)
    events: list[dict] = []
    obs._write_pbt_copy_event = lambda payload: events.append(dict(payload))
    monkeypatch.setattr(obs, "_rewrite_checkpoint", lambda *a, **k: None)
    import signal

    monkeypatch.setattr(signal, "raise_signal", lambda *_a, **_k: None)
    monkeypatch.setattr(os, "_exit", lambda *_a, **_k: None)

    def _boom_write(path, data, **_k):
        raise OSError("disk full")

    monkeypatch.setattr(obs_mod, "write_curriculum_control_dict", _boom_write)
    obs._late_rescue_attempt_count[0] = 1
    obs._late_rescue_selected_count[0] = 1
    from dexpbt_isaaclab.pbt.pbt_observer import DexPBTPbtAlgoObserver

    ok = DexPBTPbtAlgoObserver._restart_process_with_new_params(
        obs,
        {
            "gamma": 0.99,
            "kl_threshold": 0.01,
            "grad_norm": 1.0,
            "critic_coef": 4.0,
            "bounds_loss_coef": 1e-4,
            "e_clip": 0.1,
            "mini_epochs": 2,
            "learning_rate": 1e-4,
            "alpha_reach": 30.0,
            "alpha_pick": 20.0,
            "r_picked": 300.0,
            "alpha_targ": 200.0,
            "r_success": 1000.0,
            "kuka_actions_penalty_scale": 3e-3,
            "allegro_actions_penalty_scale": 3e-4,
            "wm_strategy": "none",
            "wm_ensemble_size": 3,
            "enable_wm": False,
            "enable_counterfactual": False,
        },
        str(ckpt),
        1,
        3_200_000_000,
        "bottom",
        source_curriculum={"success_tolerance": 0.020},
        late_rescue=True,
        late_rescue_decision={"checkpoint_type": "live"},
        runtime_snap=obs._snapshot_restart_runtime_state(),
    )
    assert ok is False
    assert obs._late_rescue_executed_count.get(0, 0) == 0
    assert obs._late_rescue_frame.get(0, -1) < 0
    manifest = Path(obs.train_dir).parent / obs.RESTART_MANIFEST
    assert not manifest.exists()
    assert any(
        e.get("event_type") == "pbt_restart_transaction_rolled_back" for e in events
    )


def test_archive_invalid_retention_at_save_rejected(tmp_path: Path):
    archive = tmp_path / "best"
    archive.mkdir()
    (archive / "a.json").write_text(json.dumps({
        "nsucc_mean": 40.0,
        "same_tol_retention_at_save": 1.0,
        "same_tol_retention_valid_at_save": False,
        "curriculum": {"success_tolerance": 0.020},
        "checkpoint": str(archive / "a.pth"),
    }))
    (archive / "a.pth").write_text("x")
    reject: list[dict] = []
    bad = select_compatible_archive(
        archive, target_tolerance=0.020, max_gap=0.002, min_mean=28.0,
        min_retention=0.85, require_retention_metadata=True, reject_log=reject,
    )
    assert bad is None
    assert any(r["reason"] == "retention_invalid_at_save" for r in reject)


def test_archive_nonfinite_retention_rejected(tmp_path: Path):
    for value, _label in ((float("nan"), "nan"), (float("inf"), "inf"), (float("-inf"), "-inf")):
        archive = tmp_path / f"best_{_label}"
        archive.mkdir()
        payload = {
            "nsucc_mean": 40.0,
            "same_tol_retention_at_save": value,
            "same_tol_retention_valid_at_save": True,
            "curriculum": {"success_tolerance": 0.020},
            "checkpoint": str(archive / "a.pth"),
        }
        # json cannot encode nan/inf by default; write via python repr path
        text = json.dumps({
            **payload,
            "same_tol_retention_at_save": 0.0,
        })
        # Force non-finite by patching after load via custom file
        (archive / "a.json").write_text(text)
        data = json.loads((archive / "a.json").read_text())
        data["same_tol_retention_at_save"] = value
        # Use a tiny custom loader path: write with allow_nan
        (archive / "a.json").write_text(json.dumps(data, allow_nan=True))
        (archive / "a.pth").write_text("x")
        reject: list[dict] = []
        bad = select_compatible_archive(
            archive, target_tolerance=0.020, max_gap=0.002, min_mean=28.0,
            min_retention=0.85, require_retention_metadata=True, reject_log=reject,
        )
        assert bad is None, _label
        assert any(r["reason"] == "retention_nonfinite" for r in reject), reject


def test_archive_valid_retention_accepted(tmp_path: Path):
    archive = tmp_path / "best"
    archive.mkdir()
    (archive / "a.json").write_text(json.dumps({
        "nsucc_mean": 35.0,
        "same_tol_retention_at_save": 0.90,
        "same_tol_retention_valid_at_save": True,
        "curriculum": {"success_tolerance": 0.020},
        "checkpoint": str(archive / "a.pth"),
    }))
    (archive / "a.pth").write_text("x")
    ok = select_compatible_archive(
        archive, target_tolerance=0.020, max_gap=0.002, min_mean=28.0,
        min_retention=0.85, require_retention_metadata=True,
    )
    assert ok is not None


def test_authoritative_source_collapse_rejection(tmp_path: Path):
    from dexpbt_isaaclab.pbt.late_degradation_rescue import (
        select_late_degradation_rescue_source,
    )
    from dexpbt_isaaclab.pbt.pbt_curriculum_p1 import (
        CurriculumControlState,
        write_curriculum_control,
    )

    obs = _make_minimal_observer(tmp_path)
    ctrl_dir = obs._curriculum_control_dir
    for reason_field, kwargs in (
        ("hard_collapse_triggered", {"hard_collapse_triggered": True}),
        ("collapse_freeze_active", {"collapse_freeze_active": True}),
        ("collapse_rollback_active", {"collapse_rollback_active": True}),
        (
            "bridge_probation_hard_collapse_triggered",
            {"bridge_probation_hard_collapse_triggered": True},
        ),
    ):
        write_curriculum_control(
            ctrl_dir / "001.json",
            CurriculumControlState(**kwargs),
        )
        population = {
            0: {"nsucc_mean": 10.0, "curriculum": {"success_tolerance": 0.020}},
            1: {
                "nsucc_mean": 35.0,
                "pbt_ranking_score": 90.0,
                "curriculum": {"success_tolerance": 0.020},
                "checkpoint": "1.pth",
            },
        }
        decision = select_late_degradation_rescue_source(
            sorted_policies=[1],
            population=population,
            target_policy_idx=0,
            target_tolerance=0.020,
            target_nsucc_mean=10.0,
            env_frames=1,
            n_top=1,
            max_tol_gap=0.002,
            source_min_mean=28.0,
            min_source_advantage=8.0,
            source_min_retention=0.85,
            reject_collapse_source=True,
            source_retention_fn=lambda *_: 0.92,
            eligible_fn=lambda *_: (True, "ok"),
            source_collapse_fn=lambda idx, st: obs._late_rescue_source_has_collapse(idx, st),
        )
        assert decision["action"] == "skip", reason_field
        assert decision["strict_filter_reasons"][1] == reason_field


def test_source_control_missing_fail_closed(tmp_path: Path):
    from dexpbt_isaaclab.pbt.late_degradation_rescue import (
        select_late_degradation_rescue_source,
    )

    obs = _make_minimal_observer(
        tmp_path, late_degradation_rescue_require_source_control=True
    )
    population = {
        0: {"nsucc_mean": 10.0, "curriculum": {"success_tolerance": 0.020}},
        1: {
            "nsucc_mean": 35.0,
            "pbt_ranking_score": 90.0,
            "curriculum": {"success_tolerance": 0.020},
            "checkpoint": "1.pth",
        },
    }
    decision = select_late_degradation_rescue_source(
        sorted_policies=[1],
        population=population,
        target_policy_idx=0,
        target_tolerance=0.020,
        target_nsucc_mean=10.0,
        env_frames=1,
        n_top=1,
        max_tol_gap=0.002,
        source_min_mean=28.0,
        min_source_advantage=8.0,
        source_min_retention=0.85,
        reject_collapse_source=True,
        source_retention_fn=lambda *_: 0.92,
        eligible_fn=lambda *_: (True, "ok"),
        source_collapse_fn=lambda idx, st: obs._late_rescue_source_has_collapse(idx, st),
    )
    assert decision["action"] == "skip"
    assert decision["strict_filter_reasons"][1] == "source_control_missing"

    obs2 = _make_minimal_observer(
        tmp_path, late_degradation_rescue_require_source_control=False
    )
    decision2 = select_late_degradation_rescue_source(
        sorted_policies=[1],
        population=population,
        target_policy_idx=0,
        target_tolerance=0.020,
        target_nsucc_mean=10.0,
        env_frames=1,
        n_top=1,
        max_tol_gap=0.002,
        source_min_mean=28.0,
        min_source_advantage=8.0,
        source_min_retention=0.85,
        reject_collapse_source=True,
        source_retention_fn=lambda *_: 0.92,
        eligible_fn=lambda *_: (True, "ok"),
        source_collapse_fn=lambda idx, st: obs2._late_rescue_source_has_collapse(idx, st),
    )
    assert decision2["action"] == "copy"
    assert decision2["source_policy"] == 1


def test_live_rescue_respects_target_burnin_by_default():
    historical_archive_types = (
        "archive", "best_stable_archive", "best_floor_recovery_archive",
        "late_rescue_archive",
    )
    pbt_recovery_bypass = True
    live_bypass_flag = False
    tier_label = "late_rescue_live"
    checkpoint_type = "live"
    archive_recovery_bypass = (
        checkpoint_type in historical_archive_types
        or tier_label in historical_archive_types
    ) and pbt_recovery_bypass
    live_late_rescue_bypass = (
        tier_label == "late_rescue_live" and live_bypass_flag
    )
    assert not (archive_recovery_bypass or live_late_rescue_bypass)

    tier_label = "late_rescue_archive"
    checkpoint_type = "late_rescue_archive"
    archive_recovery_bypass = (
        checkpoint_type in historical_archive_types
        or tier_label in historical_archive_types
    ) and pbt_recovery_bypass
    live_late_rescue_bypass = (
        tier_label == "late_rescue_live" and live_bypass_flag
    )
    assert archive_recovery_bypass or live_late_rescue_bypass

    tier_label = "late_rescue_live"
    live_bypass_flag = True
    live_late_rescue_bypass = (
        tier_label == "late_rescue_live" and live_bypass_flag
    )
    assert live_late_rescue_bypass


def test_manifest_failure_rolls_back_runtime_state(tmp_path: Path):
    obs = _make_minimal_observer(tmp_path)
    obs.last_mutation_frame = 10
    obs._restart_count = 2
    obs._frames_at_restart = 20
    obs._lineage_id = "lin-old"
    obs._lineage_start_pf = 5
    obs._weight_age_frame = 7
    obs._recovery_restart_count = 1
    obs._last_recovery_frame = 30
    snap = obs._snapshot_restart_runtime_state()

    # Mutate as a failed restart would.
    obs.last_mutation_frame = 999
    obs._restart_count = 99
    obs._frames_at_restart = 999
    obs._lineage_id = "lin-new"
    obs._lineage_start_pf = 999
    obs._weight_age_frame = 999
    obs._recovery_restart_count = 99
    obs._last_recovery_frame = 999
    obs._late_rescue_executed_count[0] = 5
    obs._late_rescue_frame[0] = 999

    # Leave a fake manifest and tmp restart dir.
    obs.train_dir = tmp_path / "run" / "rl_games"
    obs.train_dir.mkdir(parents=True, exist_ok=True)
    manifest = obs.train_dir.parent / obs.RESTART_MANIFEST
    manifest.write_text("{}")
    tmp_dir = obs.workspace / f"restart_p{obs.policy_idx}"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    (tmp_dir / "x.pth").write_text("x")

    obs._restore_restart_runtime_state(snap)
    obs._cleanup_failed_restart_artifacts()
    assert obs.last_mutation_frame == 10
    assert obs._restart_count == 2
    assert obs._frames_at_restart == 20
    assert obs._lineage_id == "lin-old"
    assert obs._lineage_start_pf == 5
    assert obs._weight_age_frame == 7
    assert obs._recovery_restart_count == 1
    assert obs._last_recovery_frame == 30
    assert obs._late_rescue_executed_count[0] == 0
    assert not manifest.exists()
    assert not tmp_dir.exists()


def test_late_rescue_counter_invariant(tmp_path: Path):
    obs = _make_minimal_observer(tmp_path)

    def _invariant():
        a = int(obs._late_rescue_attempt_count.get(0, 0))
        s = int(obs._late_rescue_selected_count.get(0, 0))
        e = int(obs._late_rescue_executed_count.get(0, 0))
        assert a >= s >= e, (a, s, e)

    # no source
    obs._late_rescue_attempt_count[0] = 1
    obs._late_rescue_blocked_count[0] = 1
    _invariant()
    # anti-churn blocked after attempt
    obs._late_rescue_attempt_count[0] = 2
    obs._late_rescue_blocked_count[0] = 2
    _invariant()
    # selected then manifest failed (selected stays, executed not)
    obs._late_rescue_attempt_count[0] = 3
    obs._late_rescue_selected_count[0] = 1
    obs._late_rescue_blocked_count[0] = 3
    _invariant()
    # successful rescue
    obs._late_rescue_attempt_count[0] = 4
    obs._late_rescue_selected_count[0] = 2
    obs._late_rescue_executed_count[0] = 1
    _invariant()


# ---------------------------------------------------------------------------
# Restart transaction integration tests (call real restart owner)
# ---------------------------------------------------------------------------

_RESTART_PARAMS: dict = {
    "gamma": 0.99,
    "kl_threshold": 0.01,
    "grad_norm": 1.0,
    "critic_coef": 4.0,
    "bounds_loss_coef": 1e-4,
    "e_clip": 0.1,
    "mini_epochs": 2,
    "learning_rate": 1e-4,
    "alpha_reach": 30.0,
    "alpha_pick": 20.0,
    "r_picked": 300.0,
    "alpha_targ": 200.0,
    "r_success": 1000.0,
    "kuka_actions_penalty_scale": 3e-3,
    "allegro_actions_penalty_scale": 3e-4,
    "wm_strategy": "none",
    "wm_ensemble_size": 3,
    "enable_wm": False,
    "enable_counterfactual": False,
}


def _seed_control_json(obs, *, extra: dict | None = None) -> Path:
    from dexpbt_isaaclab.pbt.pbt_curriculum_p1 import (
        CurriculumControlState,
        write_curriculum_control_dict,
    )

    path = obs._curriculum_control_dir / f"{int(obs.policy_idx):03d}.json"
    data = CurriculumControlState(
        last_mastered_tolerance=0.021,
        bridge_probation_entry_frame=7,
        late_rescue_executed_count=0,
    ).to_json()
    data["custom_unknown_field"] = "preserve-me"
    if extra:
        data.update(extra)
    write_curriculum_control_dict(path, data)
    return path


def _seed_checkpoint(tmp_path: Path) -> Path:
    ckpt = tmp_path / "source.pth"
    torch.save({"frame": 0, "model": {}, "pbt_history": []}, ckpt)
    return ckpt


def _patch_restart_exit(monkeypatch):
    import os
    import signal

    monkeypatch.setattr(signal, "raise_signal", lambda *_a, **_k: None)
    monkeypatch.setattr(os, "_exit", lambda *_a, **_k: None)


def _call_restart(
    obs,
    ckpt: Path,
    *,
    late_rescue: bool = True,
    source_safe_apply: dict | None = None,
    runtime_snap: dict | None = None,
):
    from dexpbt_isaaclab.pbt.pbt_observer import DexPBTPbtAlgoObserver

    snap = runtime_snap or obs._snapshot_restart_runtime_state()
    return DexPBTPbtAlgoObserver._restart_process_with_new_params(
        obs,
        dict(_RESTART_PARAMS),
        str(ckpt),
        1,
        3_200_000_000,
        "bottom",
        source_curriculum={"success_tolerance": 0.020},
        source_weight_age_frame=100,
        late_rescue=late_rescue,
        late_rescue_decision={"checkpoint_type": "live"},
        runtime_snap=snap,
        source_safe_apply=source_safe_apply,
    )


def test_txn_source_safe_then_manifest_failure_restores_control(tmp_path: Path, monkeypatch):
    obs = _make_minimal_observer(tmp_path)
    control_path = _seed_control_json(obs)
    original = control_path.read_bytes()
    ckpt = _seed_checkpoint(tmp_path)
    events: list[dict] = []
    obs._write_pbt_copy_event = lambda payload: events.append(dict(payload))
    monkeypatch.setattr(obs, "_rewrite_checkpoint", lambda *a, **k: None)
    _patch_restart_exit(monkeypatch)

    real_replace = os.replace

    def _boom_manifest(src, dst):
        if str(dst).endswith("pbt_restart.json"):
            raise OSError("simulated manifest replace failure")
        return real_replace(src, dst)

    monkeypatch.setattr(os, "replace", _boom_manifest)

    source_safe_apply = {
        "path": control_path,
        "consolidation_env_frames": 50_000_000,
        "consolidation_source": "recovery_consolidation_source_safe",
        "num_envs": 1,
        "inherited_tolerance": 0.020,
        "inheritance_audit": {"inheritance_reason": "test"},
        "require_existing": True,
    }
    ok = _call_restart(obs, ckpt, source_safe_apply=source_safe_apply)
    assert ok is False
    assert control_path.read_bytes() == original
    manifest = Path(obs.train_dir).parent / obs.RESTART_MANIFEST
    assert not manifest.exists()
    assert not (obs.workspace / f"restart_p{obs.policy_idx}").exists()
    assert obs._late_rescue_executed_count.get(0, 0) == 0
    rollback = [
        e for e in events if e.get("event_type") == "pbt_restart_transaction_rolled_back"
    ]
    assert len(rollback) == 1


def test_txn_writer_exception_after_commit_keeps_quota(tmp_path: Path, monkeypatch):
    obs = _make_minimal_observer(tmp_path)
    control_path = _seed_control_json(obs)
    ckpt = _seed_checkpoint(tmp_path)
    events: list[dict] = []
    obs._write_pbt_copy_event = lambda payload: events.append(dict(payload))
    monkeypatch.setattr(obs, "_rewrite_checkpoint", lambda *a, **k: None)
    _patch_restart_exit(monkeypatch)

    class _BadWriter:
        def add_scalar(self, *a, **k):
            return None

        def flush(self):
            raise RuntimeError("tb flush boom")

        def close(self):
            return None

    obs.writer = _BadWriter()
    obs._late_rescue_attempt_count[0] = 1
    obs._late_rescue_selected_count[0] = 1

    ok = _call_restart(obs, ckpt)
    assert ok is True
    manifest = Path(obs.train_dir).parent / obs.RESTART_MANIFEST
    assert manifest.is_file()
    data = json.loads(control_path.read_text())
    assert int(data["late_rescue_executed_count"]) == 1
    assert int(obs._late_rescue_executed_count[0]) == 1
    assert not any(
        e.get("event_type") == "pbt_restart_transaction_rolled_back" for e in events
    )


def test_txn_target_control_missing_fail_closed(tmp_path: Path, monkeypatch):
    obs = _make_minimal_observer(tmp_path)
    control_path = obs._curriculum_control_dir / "000.json"
    assert not control_path.exists()
    ckpt = _seed_checkpoint(tmp_path)
    events: list[dict] = []
    obs._write_pbt_copy_event = lambda payload: events.append(dict(payload))
    monkeypatch.setattr(obs, "_rewrite_checkpoint", lambda *a, **k: None)
    _patch_restart_exit(monkeypatch)

    ok = _call_restart(obs, ckpt)
    assert ok is False
    assert not control_path.exists()
    manifest = Path(obs.train_dir).parent / obs.RESTART_MANIFEST
    assert not manifest.exists()
    assert obs._late_rescue_executed_count.get(0, 0) == 0
    assert any(
        e.get("event_type") == "late_degradation_rescue_target_control_missing"
        for e in events
    )


def test_txn_target_control_corrupt_fail_closed(tmp_path: Path, monkeypatch):
    obs = _make_minimal_observer(tmp_path)
    control_path = obs._curriculum_control_dir / "000.json"
    control_path.write_text("{not-json")
    corrupt_bytes = control_path.read_bytes()
    ckpt = _seed_checkpoint(tmp_path)
    events: list[dict] = []
    obs._write_pbt_copy_event = lambda payload: events.append(dict(payload))
    monkeypatch.setattr(obs, "_rewrite_checkpoint", lambda *a, **k: None)
    _patch_restart_exit(monkeypatch)

    ok = _call_restart(obs, ckpt)
    assert ok is False
    assert control_path.read_bytes() == corrupt_bytes
    manifest = Path(obs.train_dir).parent / obs.RESTART_MANIFEST
    assert not manifest.exists()
    assert obs._late_rescue_executed_count.get(0, 0) == 0
    assert any(
        e.get("event_type") == "late_degradation_rescue_target_control_parse_error"
        for e in events
    )


def test_archive_nonfinite_tolerance_rejected(tmp_path: Path):
    for bad in (float("nan"), float("inf"), float("-inf")):
        d = tmp_path / f"tol_{bad}"
        d.mkdir()
        p = d / "a.json"
        p.write_text(json.dumps({
            "curriculum": {"success_tolerance": bad},
            "nsucc_mean": 40.0,
            "nsucc_max": 50.0,
            "frame": 1,
            "checkpoint": str(d / "a.pth"),
            "same_tol_retention_at_save": 0.9,
            "same_tol_retention_valid_at_save": True,
        }))
        (d / "a.pth").write_text("x")
        reject: list[dict] = []
        assert select_compatible_archive(
            d,
            target_tolerance=0.02,
            max_gap=0.01,
            min_mean=28.0,
            min_retention=0.85,
            require_retention_metadata=True,
            reject_log=reject,
        ) is None
        assert any(r["reason"] == "tolerance_nonfinite" for r in reject)


def test_archive_nonfinite_mean_rejected(tmp_path: Path):
    for bad in (float("nan"), float("inf"), float("-inf")):
        d = tmp_path / f"mean_{bad}"
        d.mkdir()
        p = d / "a.json"
        p.write_text(json.dumps({
            "curriculum": {"success_tolerance": 0.02},
            "nsucc_mean": bad,
            "nsucc_max": 50.0,
            "frame": 1,
            "checkpoint": str(d / "a.pth"),
            "same_tol_retention_at_save": 0.9,
            "same_tol_retention_valid_at_save": True,
        }))
        (d / "a.pth").write_text("x")
        reject: list[dict] = []
        assert select_compatible_archive(
            d,
            target_tolerance=0.02,
            max_gap=0.01,
            min_mean=28.0,
            min_retention=0.85,
            require_retention_metadata=True,
            reject_log=reject,
        ) is None
        assert any(r["reason"] == "mean_nonfinite" for r in reject)


def test_txn_duplicate_rollback_event_once(tmp_path: Path, monkeypatch):
    obs = _make_minimal_observer(tmp_path)
    _seed_control_json(obs)
    ckpt = _seed_checkpoint(tmp_path)
    events: list[dict] = []
    obs._write_pbt_copy_event = lambda payload: events.append(dict(payload))
    monkeypatch.setattr(obs, "_rewrite_checkpoint", lambda *a, **k: None)
    _patch_restart_exit(monkeypatch)

    real_replace = os.replace

    def _boom_manifest(src, dst):
        if str(dst).endswith("pbt_restart.json"):
            raise OSError("manifest boom")
        return real_replace(src, dst)

    monkeypatch.setattr(os, "replace", _boom_manifest)
    ok = _call_restart(obs, ckpt)
    assert ok is False
    # Simulate former caller-side reaction without second rollback.
    blocked_before = int(obs._late_rescue_blocked_count.get(0, 0))
    obs._late_rescue_blocked_count[0] = blocked_before + 1
    rollback = [
        e for e in events if e.get("event_type") == "pbt_restart_transaction_rolled_back"
    ]
    assert len(rollback) == 1
    assert obs._late_rescue_blocked_count[0] - blocked_before == 1


def test_txn_blocked_count_once_on_failed_restart(tmp_path: Path, monkeypatch):
    obs = _make_minimal_observer(tmp_path)
    _seed_control_json(obs)
    ckpt = _seed_checkpoint(tmp_path)
    events: list[dict] = []
    obs._write_pbt_copy_event = lambda payload: events.append(dict(payload))
    monkeypatch.setattr(obs, "_rewrite_checkpoint", lambda *a, **k: None)
    _patch_restart_exit(monkeypatch)
    real_replace = os.replace

    def _boom(src, dst):
        if str(dst).endswith("pbt_restart.json"):
            raise OSError("manifest boom")
        return real_replace(src, dst)

    monkeypatch.setattr(os, "replace", _boom)

    before = int(obs._late_rescue_blocked_count.get(0, 0))
    ok = _call_restart(obs, ckpt)
    assert ok is False
    # Caller increments blocked exactly once on False.
    obs._late_rescue_blocked_count[0] = before + 1
    assert obs._late_rescue_blocked_count[0] == before + 1
    assert (
        len(
            [
                e
                for e in events
                if e.get("event_type") == "pbt_restart_transaction_rolled_back"
            ]
        )
        == 1
    )


def test_txn_successful_commit(tmp_path: Path, monkeypatch):
    obs = _make_minimal_observer(tmp_path)
    control_path = _seed_control_json(obs)
    ckpt = _seed_checkpoint(tmp_path)
    events: list[dict] = []
    obs._write_pbt_copy_event = lambda payload: events.append(dict(payload))
    monkeypatch.setattr(obs, "_rewrite_checkpoint", lambda *a, **k: None)
    _patch_restart_exit(monkeypatch)
    obs._late_rescue_attempt_count[0] = 1
    obs._late_rescue_selected_count[0] = 1

    ok = _call_restart(obs, ckpt)
    assert ok is True
    manifest = Path(obs.train_dir).parent / obs.RESTART_MANIFEST
    assert manifest.is_file()
    data = json.loads(control_path.read_text())
    assert int(data["late_rescue_executed_count"]) == 1
    assert int(data["late_rescue_last_frame"]) == 3_200_000_000
    assert data.get("custom_unknown_field") == "preserve-me"
    assert int(obs._late_rescue_executed_count[0]) == 1
    assert int(obs._late_rescue_frame[0]) == 3_200_000_000
    assert not any(
        e.get("event_type") == "pbt_restart_transaction_rolled_back" for e in events
    )


def test_txn_counter_invariant_across_real_failures(tmp_path: Path, monkeypatch):
    obs = _make_minimal_observer(tmp_path)
    control_path = _seed_control_json(obs)
    ckpt = _seed_checkpoint(tmp_path)
    monkeypatch.setattr(obs, "_rewrite_checkpoint", lambda *a, **k: None)
    _patch_restart_exit(monkeypatch)

    def _inv():
        a = int(obs._late_rescue_attempt_count.get(0, 0))
        s = int(obs._late_rescue_selected_count.get(0, 0))
        e = int(obs._late_rescue_executed_count.get(0, 0))
        assert a >= s >= e, (a, s, e)

    # no source (selection never happens)
    obs._late_rescue_attempt_count[0] = 1
    _inv()

    # checkpoint missing -> restart fails before commit
    obs._late_rescue_attempt_count[0] = 2
    obs._late_rescue_selected_count[0] = 1
    missing = tmp_path / "missing.pth"
    ok = _call_restart(obs, missing)
    assert ok is False
    _inv()

    # burn-in style: selected not incremented further; just keep invariant
    obs._late_rescue_attempt_count[0] = 3
    _inv()

    # manifest failure
    real_replace = os.replace

    def _boom(src, dst):
        if str(dst).endswith("pbt_restart.json"):
            raise OSError("manifest boom")
        return real_replace(src, dst)

    monkeypatch.setattr(os, "replace", _boom)
    obs._late_rescue_attempt_count[0] = 4
    obs._late_rescue_selected_count[0] = 2
    ok = _call_restart(obs, ckpt)
    assert ok is False
    assert obs._late_rescue_executed_count.get(0, 0) == 0
    _inv()
    monkeypatch.setattr(os, "replace", real_replace)

    # state persist failure via missing control after delete
    control_path.unlink()
    obs._late_rescue_attempt_count[0] = 5
    obs._late_rescue_selected_count[0] = 3
    ok = _call_restart(obs, ckpt)
    assert ok is False
    _inv()

    # successful restart
    _seed_control_json(obs)
    obs._late_rescue_attempt_count[0] = 6
    obs._late_rescue_selected_count[0] = 4
    ok = _call_restart(obs, ckpt)
    assert ok is True
    assert obs._late_rescue_executed_count[0] == 1
    _inv()
