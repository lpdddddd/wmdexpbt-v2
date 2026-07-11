"""Observer→PBT wiring tests for cser_dexpbt_progress_refine adaptive control.

These tests exercise the real DexPBTPbtAlgoObserver control / ternary / restart
transaction paths — not just the pure adaptive helpers.
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path
from typing import Any
from unittest import mock

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _stub_rl_games() -> None:
    import sys
    import types

    if "rl_games.common.algo_observer" in sys.modules and hasattr(
        sys.modules["rl_games.common.algo_observer"], "IsaacAlgoObserver"
    ):
        return
    # Clear any incomplete stubs from other tests.
    for key in list(sys.modules):
        if key == "rl_games" or key.startswith("rl_games."):
            del sys.modules[key]
    m = types.ModuleType("rl_games")
    c = types.ModuleType("rl_games.common")
    a = types.ModuleType("rl_games.common.algo_observer")

    class IsaacAlgoObserver:  # pragma: no cover - stub
        def __init__(self, *args, **kwargs):
            pass

    a.IsaacAlgoObserver = IsaacAlgoObserver
    sys.modules["rl_games"] = m
    sys.modules["rl_games.common"] = c
    sys.modules["rl_games.common.algo_observer"] = a


def _make_progress_observer(tmp_path: Path, **extra: Any):
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
        num_policies=6,
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
        curriculum_mode="mastery",
        mastery_nsucc_mean=5.0,
        mastery_nsucc_max=5.0,
        mastery_consecutive_windows=1,
        pbt_single_shrink_per_window=True,
        curriculum_collapse_detect=True,
        pbt_early_collapse_detect=False,
        collapse_drop_ratio=0.5,
        collapse_patience=1,
        late_degradation_rescue_enable=False,
        adaptive_precision_consolidation_enable=True,
        adaptive_rescue_enable=True,
        adaptive_probation_min_frames=200_000_000,
        adaptive_probation_min_stable_ticks=4,
        adaptive_probation_min_same_level_retention=0.90,
        adaptive_probation_block_ordinary_rollback=True,
        adaptive_probation_allow_hard_collapse_rollback=True,
        adaptive_probation_hard_collapse_mean=3.0,
        adaptive_probation_hard_collapse_patience=4,
        adaptive_rescue_retention_threshold=0.70,
        adaptive_rescue_degradation_patience=3,
        adaptive_rescue_source_top_fraction=0.40,
        adaptive_rescue_source_min_retention=0.85,
        adaptive_rescue_min_source_advantage_sigma=0.5,
        adaptive_rescue_max_log_tol_gap=0.20,
        adaptive_rescue_cooldown_frames=0,
        adaptive_rescue_max_per_lineage=1,
        adaptive_rescue_require_source_control=True,
        adaptive_rescue_reject_collapse_source=True,
        bridge_curriculum_enable=False,
        bridge_probation_enable=False,
        same_tol_retention_enable=False,
        late_floor_retention_gate_enable=False,
        pre_floor_peak_guard_enable=False,
        curriculum_danger_zone_enable=False,
        pbt_min_frames_between_restarts=0,
        wm_strategy_profile_name="cser_dexpbt_progress_refine",
    )
    defaults.update(extra)
    Path(defaults["train_dir"]).mkdir(parents=True, exist_ok=True)
    obs = DexPBTPbtAlgoObserver(**defaults)
    obs.writer = None
    obs.curr_policy_workspace = ws / "000"
    return obs


def _seed_window(obs, *, tol: float, mean: float, frames: int, n: int = 3, lineage="L0"):
    from dexpbt_isaaclab.pbt.pbt_curriculum_p1 import WindowSample

    window = obs._policy_window(obs.policy_idx)
    window.samples = []
    for i in range(n):
        window.samples.append(
            WindowSample(
                env_frames=int(frames) - (n - i) * 10_000_000,
                nsucc_mean=float(mean),
                nsucc_max=float(mean) + 5.0,
                success_tolerance=float(tol),
                pbt_iteration=i,
                lineage_id=lineage,
            )
        )
    return window


def _seed_adaptive(obs, *, tol: float, mean: float, frames: int, lineage="L0"):
    from types import SimpleNamespace

    sample = SimpleNamespace(
        success_tolerance=float(tol),
        nsucc_mean=float(mean),
        lineage_id=lineage,
    )
    state = {
        "curriculum": {
            "success_tolerance": float(tol),
            "target_tolerance": 0.02,
        },
        "lineage_id": lineage,
    }
    obs._adaptive_precision_update(sample=sample, state=state, env_frames=int(frames))


def test_rescue_source_top_k_six_policies():
    from dexpbt_isaaclab.pbt.adaptive_precision_consolidation import rescue_source_top_k

    assert rescue_source_top_k(6, 0.40) == 3
    assert rescue_source_top_k(6, 0.60) == 4
    assert rescue_source_top_k(1, 0.40) == 1


def test_probation_blocks_shrink_and_rolls_back_bookkeeping(tmp_path: Path):
    obs = _make_progress_observer(tmp_path, adaptive_probation_min_frames=200_000_000)
    # Arm probation via shrink detection at frame 100M.
    _seed_adaptive(obs, tol=0.022, mean=20.0, frames=50_000_000)
    _seed_adaptive(obs, tol=0.020, mean=18.0, frames=100_000_000)
    state = obs._adaptive_precision_state(0)
    assert state is not None
    _cfg, probation, _tr = state
    assert probation.active

    # Mastery would otherwise allow shrink.
    window = _seed_window(obs, tol=0.020, mean=30.0, frames=150_000_000, n=3)
    obs._shrink_issued_this_window[0] = False
    obs._shrinks_since_rollback[0] = 0
    control = obs._build_curriculum_control_state(0, window, 150_000_000)
    assert control.shrink_token is False
    assert control.allow_shrink is False
    assert control.shrink_blocked_reason == "adaptive_probation_min_frames"
    assert obs._shrink_issued_this_window.get(0, False) is False
    assert int(obs._shrinks_since_rollback.get(0, 0)) == 0
    assert control.adaptive_probation_active is True
    assert control.adaptive_probation_can_shrink is False


def test_probation_stable_exit_allows_shrink(tmp_path: Path):
    obs = _make_progress_observer(
        tmp_path,
        adaptive_probation_min_frames=10,
        adaptive_probation_min_stable_ticks=2,
        adaptive_probation_min_same_level_retention=0.90,
    )
    _seed_adaptive(obs, tol=0.022, mean=20.0, frames=100)
    _seed_adaptive(obs, tol=0.020, mean=18.0, frames=200)  # arms probation
    # Stable retention ticks at same level.
    for f in (300, 400, 500):
        _seed_adaptive(obs, tol=0.020, mean=20.0, frames=f)
    state = obs._adaptive_precision_state(0)
    assert state is not None
    _cfg, probation, _tr = state
    assert probation.active is False  # exited via maybe_exit

    window = _seed_window(obs, tol=0.020, mean=30.0, frames=600, n=3)
    control = obs._build_curriculum_control_state(0, window, 600)
    assert control.shrink_token is True
    assert control.allow_shrink is True
    assert control.adaptive_probation_active is False


def test_ordinary_rollback_suppressed_under_adaptive_probation(tmp_path: Path):
    obs = _make_progress_observer(tmp_path)
    _seed_adaptive(obs, tol=0.022, mean=20.0, frames=50_000_000)
    _seed_adaptive(obs, tol=0.020, mean=18.0, frames=100_000_000)
    window = _seed_window(obs, tol=0.020, mean=10.0, frames=150_000_000, n=2)
    obs._recent_peak_nsucc_mean[0] = 40.0
    obs._last_mastered_tolerance[0] = 0.022
    obs._shrinks_since_rollback[0] = 1
    control = obs._build_curriculum_control_state(0, window, 150_000_000)
    assert float(control.forced_success_tolerance or 0.0) == 0.0
    assert control.adaptive_probation_ordinary_rollback_blocked is True
    assert control.collapse_rollback_active is False


def test_adaptive_hard_collapse_rollback_wider_tolerance(tmp_path: Path):
    obs = _make_progress_observer(
        tmp_path,
        adaptive_probation_hard_collapse_mean=12.0,
        adaptive_probation_hard_collapse_patience=2,
    )
    _seed_adaptive(obs, tol=0.022, mean=20.0, frames=50_000_000)
    _seed_adaptive(obs, tol=0.020, mean=18.0, frames=100_000_000)
    # Drive hard-collapse streak via low means while probation active.
    for f in (110_000_000, 120_000_000, 130_000_000):
        _seed_adaptive(obs, tol=0.020, mean=5.0, frames=f)
    state = obs._adaptive_precision_state(0)
    assert state is not None
    _cfg, probation, _tr = state
    assert probation.active
    assert probation.hard_collapsed(_cfg)

    window = _seed_window(obs, tol=0.020, mean=5.0, frames=140_000_000, n=2)
    obs._recent_peak_nsucc_mean[0] = 40.0
    obs._last_mastered_safe_tolerance[0] = 0.022
    obs._shrinks_since_rollback[0] = 1
    control = obs._build_curriculum_control_state(0, window, 140_000_000)
    assert control.adaptive_probation_hard_collapse is True
    assert control.adaptive_probation_rollback_triggered is True
    assert float(control.forced_success_tolerance) > 0.020 + 1e-9
    assert float(control.forced_success_tolerance) == pytest.approx(0.022)
    # Must not depend on bridge ladder constants.
    assert abs(float(control.forced_success_tolerance) - 0.0205) > 1e-9


def test_adaptive_rescue_trigger_uses_adaptive_selector(tmp_path: Path):
    obs = _make_progress_observer(
        tmp_path,
        num_policies=3,
        policy_idx=2,
        adaptive_rescue_degradation_patience=1,
        adaptive_rescue_cooldown_frames=0,
    )
    # Warm retention then degrade.
    _seed_adaptive(obs, tol=0.020, mean=30.0, frames=10_000_000, lineage="L1")
    _seed_adaptive(obs, tol=0.020, mean=10.0, frames=20_000_000, lineage="L1")
    ready, reason = obs._adaptive_degradation_rescue_ready(
        policy_idx=2,
        tolerance=0.020,
        nsucc_mean=10.0,
        env_frames=30_000_000,
        lineage_id="L1",
    )
    assert ready is True
    assert reason == "ready"
    assert obs._late_degradation_rescue_ready(
        2, tolerance=0.020, nsucc_mean=10.0, env_frames=30_000_000
    ) is False

    calls = {"adaptive": 0, "legacy": 0, "ordinary": 0}

    def _adapt(**kwargs):
        calls["adaptive"] += 1
        return {
            "action": "skip",
            "tier": "skipped",
            "checkpoint_type": "none",
            "source_policy": -1,
            "source_state": None,
            "strict_candidates": [],
            "relaxed_candidates": [],
            "strict_filter_reasons": {},
            "relaxed_filter_reasons": {},
            "archive_reason": "test",
            "adaptive_rescue_reason": "no_compatible_source",
            "adaptive_source_rank": -1,
            "adaptive_source_advantage_sigma": 0.0,
        }

    def _legacy(**kwargs):
        calls["legacy"] += 1
        return _adapt()

    def _ordinary(**kwargs):
        calls["ordinary"] += 1
        return _adapt()

    obs._select_adaptive_degradation_rescue_source = _adapt  # type: ignore
    obs._select_late_degradation_rescue_source = _legacy  # type: ignore
    obs._select_source = _ordinary  # type: ignore

    # Build a bottom-tier population where policy 2 ranks last.
    ckpt = tmp_path / "c.pth"
    ckpt.write_bytes(b"x")
    population = {}
    for i, mean in enumerate([40.0, 35.0, 10.0]):
        population[i] = {
            "policy_idx": i,
            "frame": 50_000_000,
            "iteration": 5,
            "nsucc_mean": mean,
            "nsucc_max": mean + 5,
            "pbt_ranking_score": mean,
            "objective": mean,
            "checkpoint": str(ckpt),
            "lineage_id": "L1",
            "curriculum": {"success_tolerance": 0.020, "frame_since_restart": 50_000_000},
            "params": {"learning_rate": 1e-4},
        }
        w = obs._policy_window(i)
        from dexpbt_isaaclab.pbt.pbt_curriculum_p1 import WindowSample

        w.samples = [
            WindowSample(
                env_frames=50_000_000,
                nsucc_mean=mean,
                nsucc_max=mean + 5,
                success_tolerance=0.020,
                pbt_iteration=1,
                lineage_id="L1",
            )
        ]
    # Ensure skill-delta / promising gates allow the path to source selection.
    obs.pbt_consolidation_cfg.enabled = False
    obs.best_policy_workspace.mkdir(parents=True, exist_ok=True)
    with mock.patch(
        "dexpbt_isaaclab.pbt.pbt_observer.pbt_copy_allowed_for_target",
        return_value=(True, "ok", 1.0),
    ):
        obs._ternary_pbt([0, 1, 2], {0: 40.0, 1: 35.0, 2: 10.0}, population, 50_000_000)
    assert calls["adaptive"] == 1
    assert calls["legacy"] == 0
    assert calls["ordinary"] == 0


def test_adaptive_selector_picks_compatible_healthy_source(tmp_path: Path):
    obs = _make_progress_observer(
        tmp_path,
        num_policies=4,
        policy_idx=3,
        adaptive_rescue_require_source_control=True,
        adaptive_rescue_min_source_advantage_sigma=0.1,
    )
    from dexpbt_isaaclab.pbt.pbt_curriculum_p1 import (
        CurriculumControlState,
        write_curriculum_control,
    )

    ckpt = tmp_path / "ok.pth"
    ckpt.write_bytes(b"ckpt")
    # Source trackers / controls.
    # 0: high rank but incompatible tolerance
    # 1: compatible healthy
    # 2: collapsed
    # (3 is target)
    for idx, (tol, mean, ret_peak, collapsed) in enumerate(
        [
            (0.040, 50.0, 50.0, False),
            (0.021, 40.0, 40.0, False),
            (0.021, 45.0, 45.0, True),
        ]
    ):
        write_curriculum_control(
            obs._curriculum_control_dir / f"{idx:03d}.json",
            CurriculumControlState(
                hard_collapse_triggered=collapsed,
                collapse_freeze_active=collapsed,
            ),
        )
        # Seed source adaptive retention.
        obs.policy_idx = idx
        _seed_adaptive(obs, tol=tol, mean=ret_peak, frames=10_000_000, lineage="S")
        _seed_adaptive(obs, tol=tol, mean=ret_peak * 0.95, frames=20_000_000, lineage="S")
    obs.policy_idx = 3
    _seed_adaptive(obs, tol=0.020, mean=30.0, frames=10_000_000, lineage="T")
    _seed_adaptive(obs, tol=0.020, mean=12.0, frames=20_000_000, lineage="T")

    population = {}
    specs = [
        (0, 0.040, 50.0, 300.0),  # incompatible tol; mid rank
        (1, 0.021, 40.0, 500.0),  # compatible healthy; top rank
        (2, 0.021, 45.0, 450.0),  # collapsed
        (3, 0.020, 12.0, 100.0),  # target
    ]
    for idx, tol, mean, rank in specs:
        population[idx] = {
            "policy_idx": idx,
            "frame": 30_000_000,
            "nsucc_mean": mean,
            "nsucc_max": mean + 5,
            "pbt_ranking_score": rank,
            "objective": rank,
            "checkpoint": str(ckpt),
            "lineage_id": "T" if idx == 3 else "S",
            "curriculum": {"success_tolerance": tol},
            "params": {},
        }
    decision = obs._select_adaptive_degradation_rescue_source(
        sorted_policies=[1, 0, 2, 3],
        population=population,
        env_frames=30_000_000,
        target_tolerance=0.020,
        target_nsucc_mean=12.0,
    )
    assert decision["action"] == "copy"
    assert decision["source_policy"] == 1
    assert decision["tier"] == "adaptive_rescue_live"


def test_adaptive_rescue_respects_live_source_window(tmp_path: Path):
    obs = _make_progress_observer(tmp_path, num_policies=2, policy_idx=1)
    from dexpbt_isaaclab.pbt.pbt_curriculum_p1 import (
        CurriculumControlState,
        write_curriculum_control,
    )

    ckpt = tmp_path / "s.pth"
    ckpt.write_bytes(b"x")
    write_curriculum_control(
        obs._curriculum_control_dir / "000.json",
        CurriculumControlState(),
    )
    write_curriculum_control(
        obs._curriculum_control_dir / "001.json",
        CurriculumControlState(),
    )
    obs.policy_idx = 0
    _seed_adaptive(obs, tol=0.020, mean=40.0, frames=10_000_000)
    obs.policy_idx = 1
    _seed_adaptive(obs, tol=0.020, mean=10.0, frames=10_000_000)
    decision = {
        "action": "copy",
        "tier": "adaptive_rescue_live",
        "checkpoint_type": "live",
        "source_policy": 0,
        "source_state": {
            "curriculum": {"success_tolerance": 0.020},
            "nsucc_mean": 40.0,
            "checkpoint": str(ckpt),
            "params": {"learning_rate": 1e-4},
        },
        "strict_candidates": [0],
        "relaxed_candidates": [0],
        "strict_filter_reasons": {},
        "relaxed_filter_reasons": {},
        "archive_reason": "selected",
        "adaptive_rescue_reason": "selected",
        "adaptive_source_rank": 0,
        "adaptive_source_advantage_sigma": 2.0,
    }
    with mock.patch(
        "dexpbt_isaaclab.pbt.pbt_observer.pbt_copy_allowed_for_live_source",
        return_value=(False, "source_window_too_short"),
    ):
        # Emulate the live-source gate used inside _ternary_pbt.
        from dexpbt_isaaclab.pbt.pbt_observer import pbt_copy_allowed_for_live_source

        ok, reason = pbt_copy_allowed_for_live_source(
            obs._policy_nsucc_windows.get(0),
            interval_steps=obs.interval_steps,
        )
        assert ok is False
        assert "source_window" in reason or reason


def test_legacy_absolute_gap_does_not_reject_adaptive(tmp_path: Path):
    obs = _make_progress_observer(
        tmp_path,
        num_policies=2,
        policy_idx=1,
        late_degradation_rescue_enable=False,
        late_degradation_rescue_max_tol_gap=0.0005,
        adaptive_rescue_max_log_tol_gap=0.20,
        adaptive_rescue_min_source_advantage_sigma=0.1,
    )
    from dexpbt_isaaclab.pbt.pbt_curriculum_p1 import (
        CurriculumControlState,
        write_curriculum_control,
    )
    from dexpbt_isaaclab.pbt.adaptive_precision_consolidation import log_tolerance_gap

    ckpt = tmp_path / "s.pth"
    ckpt.write_bytes(b"x")
    write_curriculum_control(
        obs._curriculum_control_dir / "000.json", CurriculumControlState()
    )
    obs.policy_idx = 0
    _seed_adaptive(obs, tol=0.022, mean=40.0, frames=10_000_000)
    obs.policy_idx = 1
    _seed_adaptive(obs, tol=0.020, mean=10.0, frames=10_000_000)
    abs_gap = abs(0.022 - 0.020)
    assert abs_gap > obs.late_degradation_rescue_max_tol_gap
    assert log_tolerance_gap(0.022, 0.020) < obs._adaptive_precision_cfg.rescue_max_log_tol_gap
    population = {
        0: {
            "nsucc_mean": 40.0,
            "pbt_ranking_score": 400.0,
            "objective": 400.0,
            "checkpoint": str(ckpt),
            "curriculum": {"success_tolerance": 0.022},
            "params": {},
        },
        1: {
            "nsucc_mean": 10.0,
            "pbt_ranking_score": 100.0,
            "objective": 100.0,
            "checkpoint": str(ckpt),
            "curriculum": {"success_tolerance": 0.020},
            "params": {},
        },
    }
    decision = obs._select_adaptive_degradation_rescue_source(
        sorted_policies=[0, 1],
        population=population,
        env_frames=20_000_000,
        target_tolerance=0.020,
        target_nsucc_mean=10.0,
    )
    assert decision["action"] == "copy"
    assert decision["source_policy"] == 0


def test_quota_not_consumed_on_manifest_failure(tmp_path: Path, monkeypatch):
    obs = _make_progress_observer(tmp_path, num_policies=2, policy_idx=1)
    from dexpbt_isaaclab.pbt.pbt_curriculum_p1 import (
        CurriculumControlState,
        write_curriculum_control,
    )

    ckpt = tmp_path / "s.pth"
    ckpt.write_bytes(b"x")
    write_curriculum_control(
        obs._curriculum_control_dir / "001.json",
        CurriculumControlState(last_mastered_tolerance=0.02),
    )
    _seed_adaptive(obs, tol=0.020, mean=10.0, frames=10_000_000, lineage="Lq")
    before = json.loads(obs._adaptive_precision_path(1).read_text())
    obs._adaptive_rescue_executed_for_lineage[1] = 0
    obs._adaptive_last_rescue_frame[1] = -(10**18)

    def _boom(*args, **kwargs):
        raise RuntimeError("manifest_write_failed")

    monkeypatch.setattr(obs, "_restart_process_with_new_params_impl", _boom)
    ok = obs._restart_process_with_new_params(
        {"learning_rate": 1e-4},
        str(ckpt),
        0,
        20_000_000,
        "bottom",
        source_curriculum={"success_tolerance": 0.020},
        adaptive_rescue=True,
        adaptive_rescue_decision={"checkpoint_type": "live"},
    )
    assert ok is False
    assert int(obs._adaptive_rescue_executed.get(1, 0)) == 0
    assert int(obs._adaptive_rescue_executed_for_lineage.get(1, 0)) == 0
    assert int(obs._adaptive_last_rescue_frame.get(1, -(10**18))) == -(10**18)
    after = json.loads(obs._adaptive_precision_path(1).read_text())
    assert after["rescue_executed_for_lineage"] == before.get(
        "rescue_executed_for_lineage", 0
    )


def test_successful_restart_consumes_quota(tmp_path: Path, monkeypatch):
    obs = _make_progress_observer(tmp_path, num_policies=2, policy_idx=1)
    from dexpbt_isaaclab.pbt.pbt_curriculum_p1 import (
        CurriculumControlState,
        write_curriculum_control,
    )

    ckpt = tmp_path / "s.pth"
    ckpt.write_bytes(b"x")
    write_curriculum_control(
        obs._curriculum_control_dir / "001.json",
        CurriculumControlState(last_mastered_tolerance=0.02),
    )
    _seed_adaptive(obs, tol=0.020, mean=10.0, frames=10_000_000, lineage="Ls")
    obs._adaptive_lineage_id[1] = "Ls"
    obs._adaptive_rescue_executed_for_lineage[1] = 0

    def _fake_impl(*args, **kwargs):
        txn = kwargs.get("txn")
        # Mimic commit path for adaptive rescue.
        if kwargs.get("adaptive_rescue"):
            obs._adaptive_rescue_executed[1] = int(obs._adaptive_rescue_executed.get(1, 0)) + 1
            obs._adaptive_rescue_executed_for_lineage[1] = (
                int(obs._adaptive_rescue_executed_for_lineage.get(1, 0)) + 1
            )
            obs._adaptive_last_rescue_frame[1] = 50_000_000
            obs._adaptive_precision_persist(1)
        if txn is not None:
            txn["committed"] = True
        return True

    monkeypatch.setattr(obs, "_restart_process_with_new_params_impl", _fake_impl)
    # Drive through wrapper so snapshots exist; success path should keep counters.
    ok = obs._restart_process_with_new_params(
        {"learning_rate": 1e-4},
        str(ckpt),
        0,
        50_000_000,
        "bottom",
        source_curriculum={"success_tolerance": 0.020},
        adaptive_rescue=True,
        adaptive_rescue_decision={"checkpoint_type": "live"},
    )
    assert ok is True
    assert int(obs._adaptive_rescue_executed_for_lineage[1]) == 1
    assert int(obs._adaptive_last_rescue_frame[1]) == 50_000_000
    sidecar = json.loads(obs._adaptive_precision_path(1).read_text())
    assert sidecar["rescue_executed_for_lineage"] == 1
    assert sidecar["last_rescue_frame"] == 50_000_000


def test_lineage_change_resets_quota(tmp_path: Path):
    obs = _make_progress_observer(tmp_path)
    _seed_adaptive(obs, tol=0.020, mean=20.0, frames=10_000_000, lineage="A")
    obs._adaptive_rescue_executed_for_lineage[0] = 1
    obs._adaptive_degradation_streak[0] = 5
    obs._adaptive_last_rescue_frame[0] = 10_000_000
    obs._adaptive_lineage_id[0] = "A"
    _seed_adaptive(obs, tol=0.020, mean=15.0, frames=20_000_000, lineage="B")
    assert obs._adaptive_lineage_id[0] == "B"
    assert int(obs._adaptive_rescue_executed_for_lineage[0]) == 0
    assert int(obs._adaptive_degradation_streak[0]) in (0, 1)
    # Retention tracker lineage isolation.
    tracker = obs._adaptive_retention[0]
    assert tracker.lineage_id == "B"


def test_adaptive_sidecar_in_persistent_snapshot(tmp_path: Path):
    obs = _make_progress_observer(tmp_path)
    _seed_adaptive(obs, tol=0.020, mean=12.0, frames=10_000_000)
    path = obs._adaptive_precision_path(0)
    assert path.is_file()
    snap = obs._snapshot_restart_persistent_state()
    assert snap["adaptive_existed"] is True
    assert snap["adaptive_bytes"] == path.read_bytes()
    # Mutate then restore.
    path.write_text(json.dumps({"corrupted": True}))
    obs._restore_restart_persistent_state(snap)
    restored = json.loads(path.read_text())
    assert "probation" in restored
    assert "corrupted" not in restored


def test_formal_profile_fail_fast_on_bad_config(tmp_path: Path):
    with pytest.raises(RuntimeError, match="AdaptivePrecisionConfig"):
        _make_progress_observer(
            tmp_path,
            adaptive_probation_min_same_level_retention=1.5,  # invalid
        )
