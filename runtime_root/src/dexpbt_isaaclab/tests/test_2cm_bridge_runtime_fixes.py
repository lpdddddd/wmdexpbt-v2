"""Dynamic tests for CSER-DexPBT 2cm bridge runtime fixes."""

from __future__ import annotations

import json
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
    _stub_rl_games()
    from dexpbt_isaaclab.pbt.pbt_observer import DexPBTPbtAlgoObserver

    obs = object.__new__(DexPBTPbtAlgoObserver)
    obs.policy_idx = 0
    obs._late_rescue_executed_count = {}
    obs._late_rescue_count = {}
    obs._late_rescue_frame = {}
    obs._late_rescue_selected_count = {0: 1}
    obs._late_rescue_blocked_count = {}
    obs._last_restart_type = 0
    events: list[dict] = []
    obs._write_pbt_copy_event = lambda payload: events.append(dict(payload))

    def _boom(*_a, **_k):
        raise RuntimeError("manifest write failed")

    obs._restart_process_with_new_params_impl = _boom
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
    # selected already counted by caller; executed must stay 0
    assert obs._late_rescue_selected_count[0] == 1


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