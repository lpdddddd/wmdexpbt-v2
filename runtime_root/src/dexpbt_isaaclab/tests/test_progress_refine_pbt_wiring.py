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
        adaptive_rescue_source_min_samples=3,
        adaptive_rescue_source_min_stable_ticks=2,
        adaptive_rescue_source_max_staleness_frames=40_000_000,
        adaptive_rescue_archive_enable=True,
        adaptive_rescue_bypass_elimination_burnin=False,
        adaptive_rescue_bypass_on_hard_collapse=True,
        adaptive_rescue_min_frames_before_emergency_bypass=40_000_000,
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


def _seed_adaptive_retention(
    obs,
    *,
    tol: float,
    mean: float,
    frames: int,
    lineage: str = "L0",
    n: int = 4,
    frame_step: int = 1_000_000,
):
    """Seed enough same-tol updates for source retention/stability gates."""
    for i in range(int(n)):
        _seed_adaptive(
            obs,
            tol=tol,
            mean=mean,
            frames=int(frames) - (int(n) - 1 - i) * int(frame_step),
            lineage=lineage,
        )


def _skip_decision(**extra: Any) -> dict[str, Any]:
    out = {
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
    out.update(extra)
    return out


def _copy_decision(*, source_policy: int, tier: str, checkpoint_type: str, source_state: dict, **extra: Any):
    out = {
        "action": "copy",
        "tier": tier,
        "checkpoint_type": checkpoint_type,
        "source_policy": int(source_policy),
        "source_state": source_state,
        "strict_candidates": [int(source_policy)],
        "relaxed_candidates": [int(source_policy)],
        "strict_filter_reasons": {},
        "relaxed_filter_reasons": {},
        "archive_reason": "selected",
        "adaptive_rescue_reason": "selected",
        "adaptive_source_rank": 0,
        "adaptive_source_advantage_sigma": 2.0,
    }
    out.update(extra)
    return out


_RESTART_PARAMS: dict[str, Any] = {
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


def test_rollback_target_priority_previous():
    from dexpbt_isaaclab.pbt.adaptive_precision_consolidation import (
        select_adaptive_rollback_target,
    )

    decision = select_adaptive_rollback_target(
        current_tolerance=0.020,
        previous_tolerance_before_entry=0.022,
        source_tolerance=None,
        last_mastered_safe_tolerance=0.050,
        initial_tolerance=0.075,
        shrink_factor=0.95,
        max_log_gap=0.30,
    )
    assert decision.eligible is True
    assert decision.target_tolerance == pytest.approx(0.022)
    assert decision.source == "previous_tolerance"


def test_rollback_max_log_gap_prefers_fallback_or_freeze():
    from dexpbt_isaaclab.pbt.adaptive_precision_consolidation import (
        select_adaptive_rollback_target,
    )

    decision = select_adaptive_rollback_target(
        current_tolerance=0.020,
        previous_tolerance_before_entry=0.050,
        source_tolerance=None,
        last_mastered_safe_tolerance=0.050,
        initial_tolerance=0.075,
        shrink_factor=0.95,
        max_log_gap=0.30,
    )
    assert decision.eligible is True
    assert decision.source == "scale_fallback"
    assert decision.target_tolerance == pytest.approx(0.020 / 0.95)


def test_cold_start_scale_fallback():
    from dexpbt_isaaclab.pbt.adaptive_precision_consolidation import (
        select_adaptive_rollback_target,
    )

    decision = select_adaptive_rollback_target(
        current_tolerance=0.020,
        previous_tolerance_before_entry=None,
        source_tolerance=None,
        last_mastered_safe_tolerance=None,
        initial_tolerance=0.075,
        shrink_factor=0.95,
        max_log_gap=0.30,
    )
    assert decision.eligible is True
    assert decision.source == "scale_fallback"
    assert decision.target_tolerance == pytest.approx(0.020 / 0.95)


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


def test_adaptive_hard_collapse_independent_trigger(tmp_path: Path):
    """Adaptive hard-collapse alone triggers rollback without legacy peak drop."""
    obs = _make_progress_observer(
        tmp_path,
        adaptive_probation_hard_collapse_mean=12.0,
        adaptive_probation_hard_collapse_patience=2,
        collapse_patience=2,
        collapse_drop_ratio=0.5,
    )
    _seed_adaptive(obs, tol=0.022, mean=20.0, frames=50_000_000)
    _seed_adaptive(obs, tol=0.020, mean=18.0, frames=100_000_000)
    for f in (110_000_000, 120_000_000, 130_000_000):
        _seed_adaptive(obs, tol=0.020, mean=5.0, frames=f)
    state = obs._adaptive_precision_state(0)
    assert state is not None
    _cfg, probation, _tr = state
    assert probation.active
    assert probation.hard_collapsed(_cfg)

    # recent_peak ≈ current mean → evaluate_collapse_freeze stays False.
    window = _seed_window(obs, tol=0.020, mean=5.0, frames=140_000_000, n=2)
    obs._recent_peak_nsucc_mean[0] = 5.2
    obs._last_mastered_safe_tolerance[0] = 0.022
    obs._shrinks_since_rollback[0] = 1
    before = int(obs._rollback_count.get(0, 0))
    control = obs._build_curriculum_control_state(0, window, 140_000_000)
    assert control.adaptive_probation_hard_collapse_trigger_source == "adaptive_only"
    assert control.adaptive_probation_hard_collapse_independent is True
    assert control.adaptive_probation_rollback_triggered is True
    assert float(control.forced_success_tolerance) > 0.020 + 1e-9
    assert int(obs._rollback_count.get(0, 0)) == before + 1


def test_both_collapse_sources_single_rollback(tmp_path: Path):
    obs = _make_progress_observer(
        tmp_path,
        adaptive_probation_hard_collapse_mean=12.0,
        adaptive_probation_hard_collapse_patience=2,
        collapse_patience=1,
        collapse_drop_ratio=0.5,
    )
    _seed_adaptive(obs, tol=0.022, mean=20.0, frames=50_000_000)
    _seed_adaptive(obs, tol=0.020, mean=18.0, frames=100_000_000)
    for f in (110_000_000, 120_000_000, 130_000_000):
        _seed_adaptive(obs, tol=0.020, mean=5.0, frames=f)

    window = _seed_window(obs, tol=0.020, mean=5.0, frames=140_000_000, n=2)
    obs._recent_peak_nsucc_mean[0] = 40.0  # legacy collapse also active
    obs._last_mastered_safe_tolerance[0] = 0.022
    obs._shrinks_since_rollback[0] = 1
    before = int(obs._rollback_count.get(0, 0))
    control = obs._build_curriculum_control_state(0, window, 140_000_000)
    assert "adaptive_and_legacy" in str(control.adaptive_probation_hard_collapse_trigger_source)
    assert control.adaptive_probation_legacy_collapse_also_active is True
    assert control.adaptive_probation_rollback_triggered is True
    assert int(obs._rollback_count.get(0, 0)) == before + 1


def test_adaptive_rescue_trigger_uses_adaptive_selector(tmp_path: Path):
    obs = _make_progress_observer(
        tmp_path,
        num_policies=3,
        policy_idx=2,
        adaptive_rescue_degradation_patience=1,
        adaptive_rescue_cooldown_frames=0,
        adaptive_rescue_archive_enable=False,
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
    assert reason == "retention_degradation_ready"
    assert obs._late_degradation_rescue_ready(
        2, tolerance=0.020, nsucc_mean=10.0, env_frames=30_000_000
    ) is False

    calls = {"adaptive": 0, "legacy": 0, "ordinary": 0}

    def _adapt(**kwargs):
        calls["adaptive"] += 1
        return _skip_decision()

    def _legacy(**kwargs):
        calls["legacy"] += 1
        return _skip_decision()

    def _ordinary(**kwargs):
        calls["ordinary"] += 1
        return _skip_decision()

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
    # Adaptive rescue uses the adaptive target gate (unavailable delta is OK).
    obs.pbt_consolidation_cfg.enabled = False
    obs.best_policy_workspace.mkdir(parents=True, exist_ok=True)
    obs._ternary_pbt([0, 1, 2], {0: 40.0, 1: 35.0, 2: 10.0}, population, 50_000_000)
    assert calls["adaptive"] == 1
    assert calls["legacy"] == 0
    assert calls["ordinary"] == 1  # cascade falls through after live skip


def test_adaptive_selector_picks_compatible_healthy_source(tmp_path: Path):
    obs = _make_progress_observer(
        tmp_path,
        num_policies=4,
        policy_idx=3,
        adaptive_rescue_require_source_control=True,
        adaptive_rescue_min_source_advantage_sigma=0.1,
        adaptive_rescue_source_min_samples=1,
        adaptive_rescue_source_min_stable_ticks=0,
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
        # Seed source adaptive retention (multiple ticks for stability).
        obs.policy_idx = idx
        _seed_adaptive_retention(
            obs, tol=tol, mean=ret_peak, frames=20_000_000, lineage="S", n=4
        )
    obs.policy_idx = 3
    _seed_adaptive_retention(
        obs, tol=0.020, mean=30.0, frames=10_000_000, lineage="T", n=2
    )
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
            "frame": 20_000_000,
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
        env_frames=20_000_000,
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
        adaptive_rescue_source_min_samples=1,
        adaptive_rescue_source_min_stable_ticks=0,
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
    _seed_adaptive_retention(
        obs, tol=0.022, mean=40.0, frames=20_000_000, lineage="S", n=4
    )
    obs.policy_idx = 1
    _seed_adaptive_retention(
        obs, tol=0.020, mean=10.0, frames=20_000_000, lineage="T", n=2
    )
    abs_gap = abs(0.022 - 0.020)
    assert abs_gap > obs.late_degradation_rescue_max_tol_gap
    assert log_tolerance_gap(0.022, 0.020) < obs._adaptive_precision_cfg.rescue_max_log_tol_gap
    population = {
        0: {
            "nsucc_mean": 40.0,
            "pbt_ranking_score": 400.0,
            "objective": 400.0,
            "checkpoint": str(ckpt),
            "frame": 20_000_000,
            "lineage_id": "S",
            "curriculum": {"success_tolerance": 0.022},
            "params": {},
        },
        1: {
            "nsucc_mean": 10.0,
            "pbt_ranking_score": 100.0,
            "objective": 100.0,
            "checkpoint": str(ckpt),
            "frame": 20_000_000,
            "lineage_id": "T",
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


def test_peer_sidecar_corruption_does_not_kill_worker(tmp_path: Path):
    obs = _make_progress_observer(
        tmp_path,
        num_policies=3,
        policy_idx=2,
        adaptive_rescue_min_source_advantage_sigma=0.1,
        adaptive_rescue_source_min_samples=1,
        adaptive_rescue_source_min_stable_ticks=0,
    )
    from dexpbt_isaaclab.pbt.pbt_curriculum_p1 import (
        CurriculumControlState,
        write_curriculum_control,
    )

    ckpt = tmp_path / "ok.pth"
    ckpt.write_bytes(b"ckpt")
    for idx in (0, 1):
        write_curriculum_control(
            obs._curriculum_control_dir / f"{idx:03d}.json",
            CurriculumControlState(),
        )
        (tmp_path / "ws" / f"{idx:03d}").mkdir(parents=True, exist_ok=True)

    obs.policy_idx = 0
    _seed_adaptive_retention(
        obs, tol=0.020, mean=40.0, frames=20_000_000, lineage="S", n=4
    )
    # Corrupt peer 1 sidecar.
    peer_path = obs._adaptive_precision_path(1)
    peer_path.parent.mkdir(parents=True, exist_ok=True)
    peer_path.write_text("{not-json")
    obs._adaptive_probation.pop(1, None)
    obs._adaptive_retention.pop(1, None)
    obs._adaptive_peer_unavailable.pop(1, None)

    assert obs._adaptive_precision_state(1, strict_local=False) is None

    obs.policy_idx = 2
    _seed_adaptive(obs, tol=0.020, mean=10.0, frames=20_000_000, lineage="T")
    population = {
        0: {
            "nsucc_mean": 40.0,
            "pbt_ranking_score": 400.0,
            "objective": 400.0,
            "checkpoint": str(ckpt),
            "frame": 20_000_000,
            "lineage_id": "S",
            "curriculum": {"success_tolerance": 0.020},
            "params": {},
        },
        1: {
            "nsucc_mean": 38.0,
            "pbt_ranking_score": 380.0,
            "objective": 380.0,
            "checkpoint": str(ckpt),
            "frame": 20_000_000,
            "lineage_id": "S",
            "curriculum": {"success_tolerance": 0.020},
            "params": {},
        },
        2: {
            "nsucc_mean": 10.0,
            "pbt_ranking_score": 100.0,
            "objective": 100.0,
            "checkpoint": str(ckpt),
            "frame": 20_000_000,
            "lineage_id": "T",
            "curriculum": {"success_tolerance": 0.020},
            "params": {},
        },
    }
    decision = obs._select_adaptive_degradation_rescue_source(
        sorted_policies=[0, 1, 2],
        population=population,
        env_frames=20_000_000,
        target_tolerance=0.020,
        target_nsucc_mean=10.0,
    )
    assert decision["action"] == "copy"
    assert decision["source_policy"] == 0


def test_local_sidecar_corruption_fail_fast(tmp_path: Path):
    obs = _make_progress_observer(tmp_path)
    _seed_adaptive(obs, tol=0.020, mean=20.0, frames=10_000_000)
    path = obs._adaptive_precision_path(0)
    path.write_text("{broken")
    obs._adaptive_probation.pop(0, None)
    obs._adaptive_retention.pop(0, None)
    with pytest.raises(RuntimeError, match="adaptive precision restore failed"):
        obs._adaptive_precision_state(0, strict_local=True)


def test_source_retention_warmup_rejects_single_sample(tmp_path: Path):
    obs = _make_progress_observer(
        tmp_path,
        num_policies=2,
        policy_idx=1,
        adaptive_rescue_min_source_advantage_sigma=0.1,
        adaptive_rescue_source_min_samples=3,
        adaptive_rescue_source_min_stable_ticks=2,
    )
    from dexpbt_isaaclab.pbt.pbt_curriculum_p1 import (
        CurriculumControlState,
        write_curriculum_control,
    )

    ckpt = tmp_path / "s.pth"
    ckpt.write_bytes(b"x")
    write_curriculum_control(
        obs._curriculum_control_dir / "000.json", CurriculumControlState()
    )
    obs.policy_idx = 0
    _seed_adaptive(obs, tol=0.020, mean=40.0, frames=10_000_000, lineage="S")
    obs.policy_idx = 1
    _seed_adaptive(obs, tol=0.020, mean=10.0, frames=10_000_000, lineage="T")

    def _pop(frame: int):
        return {
            0: {
                "nsucc_mean": 40.0,
                "pbt_ranking_score": 400.0,
                "objective": 400.0,
                "checkpoint": str(ckpt),
                "frame": frame,
                "lineage_id": "S",
                "curriculum": {"success_tolerance": 0.020},
                "params": {},
            },
            1: {
                "nsucc_mean": 10.0,
                "pbt_ranking_score": 100.0,
                "objective": 100.0,
                "checkpoint": str(ckpt),
                "frame": frame,
                "lineage_id": "T",
                "curriculum": {"success_tolerance": 0.020},
                "params": {},
            },
        }

    decision = obs._select_adaptive_degradation_rescue_source(
        sorted_policies=[0, 1],
        population=_pop(10_000_000),
        env_frames=10_000_000,
        target_tolerance=0.020,
        target_nsucc_mean=10.0,
    )
    assert decision["action"] == "skip"
    reasons = decision.get("relaxed_filter_reasons") or {}
    assert any("retention" in str(v) for v in reasons.values())

    obs.policy_idx = 0
    _seed_adaptive_retention(
        obs, tol=0.020, mean=40.0, frames=20_000_000, lineage="S", n=4
    )
    obs.policy_idx = 1
    decision = obs._select_adaptive_degradation_rescue_source(
        sorted_policies=[0, 1],
        population=_pop(20_000_000),
        env_frames=20_000_000,
        target_tolerance=0.020,
        target_nsucc_mean=10.0,
    )
    assert decision["action"] == "copy"
    assert decision["source_policy"] == 0


def test_source_freshness_stale_rejected(tmp_path: Path):
    obs = _make_progress_observer(
        tmp_path,
        num_policies=2,
        policy_idx=1,
        adaptive_rescue_min_source_advantage_sigma=0.1,
        adaptive_rescue_source_min_samples=1,
        adaptive_rescue_source_min_stable_ticks=0,
        adaptive_rescue_source_max_staleness_frames=40_000_000,
    )
    from dexpbt_isaaclab.pbt.pbt_curriculum_p1 import (
        CurriculumControlState,
        write_curriculum_control,
    )

    ckpt = tmp_path / "s.pth"
    ckpt.write_bytes(b"x")
    write_curriculum_control(
        obs._curriculum_control_dir / "000.json", CurriculumControlState()
    )
    obs.policy_idx = 0
    _seed_adaptive_retention(
        obs, tol=0.020, mean=40.0, frames=20_000_000, lineage="S", n=4
    )
    obs.policy_idx = 1
    _seed_adaptive(obs, tol=0.020, mean=10.0, frames=20_000_000, lineage="T")

    path = obs._adaptive_precision_path(0)
    data = json.loads(path.read_text())
    data["last_update_frame"] = 1
    path.write_text(json.dumps(data))
    obs._adaptive_probation.pop(0, None)
    obs._adaptive_retention.pop(0, None)
    obs._adaptive_peer_unavailable.pop(0, None)

    population = {
        0: {
            "nsucc_mean": 40.0,
            "pbt_ranking_score": 400.0,
            "objective": 400.0,
            "checkpoint": str(ckpt),
            "frame": 100_000_000,
            "lineage_id": "S",
            "curriculum": {"success_tolerance": 0.020},
            "params": {},
        },
        1: {
            "nsucc_mean": 10.0,
            "pbt_ranking_score": 100.0,
            "objective": 100.0,
            "checkpoint": str(ckpt),
            "frame": 100_000_000,
            "lineage_id": "T",
            "curriculum": {"success_tolerance": 0.020},
            "params": {},
        },
    }
    decision = obs._select_adaptive_degradation_rescue_source(
        sorted_policies=[0, 1],
        population=population,
        env_frames=100_000_000,
        target_tolerance=0.020,
        target_nsucc_mean=10.0,
    )
    assert decision["action"] == "skip"
    reasons = decision.get("relaxed_filter_reasons") or {}
    assert any("stale" in str(v) for v in reasons.values())

    data["last_update_frame"] = 100_000_000
    path.write_text(json.dumps(data))
    obs._adaptive_probation.pop(0, None)
    obs._adaptive_retention.pop(0, None)
    obs._adaptive_peer_unavailable.pop(0, None)
    decision = obs._select_adaptive_degradation_rescue_source(
        sorted_policies=[0, 1],
        population=population,
        env_frames=100_000_000,
        target_tolerance=0.020,
        target_nsucc_mean=10.0,
    )
    assert decision["action"] == "copy"
    assert decision["source_policy"] == 0


def test_archive_fallback_when_live_unavailable(tmp_path: Path):
    obs = _make_progress_observer(tmp_path, num_policies=2, policy_idx=1)
    ckpt = tmp_path / "a.pth"
    ckpt.write_bytes(b"arch")
    archive_decision = _copy_decision(
        source_policy=0,
        tier="adaptive_rescue_best_stable_archive",
        checkpoint_type="best_stable_archive",
        source_state={
            "checkpoint": str(ckpt),
            "curriculum": {"success_tolerance": 0.020},
            "nsucc_mean": 35.0,
            "params": {},
        },
        adaptive_rescue_reason="selected_archive",
    )
    obs._select_adaptive_degradation_rescue_source = (  # type: ignore
        lambda **_k: _skip_decision(adaptive_rescue_reason="no_live")
    )
    obs._select_adaptive_archive_rescue_source = (  # type: ignore
        lambda **_k: archive_decision
    )
    obs._select_source = lambda **_k: _skip_decision()  # type: ignore
    out = obs._select_adaptive_rescue_source_cascade(
        sorted_policies=[0, 1],
        population={},
        env_frames=50_000_000,
        target_tolerance=0.020,
        target_nsucc_mean=10.0,
        n_top=1,
    )
    assert out["action"] == "copy"
    assert out["tier"] == "adaptive_rescue_best_stable_archive"


def test_archive_invalid_metadata_fail_closed(tmp_path: Path):
    obs = _make_progress_observer(
        tmp_path,
        num_policies=2,
        policy_idx=1,
        adaptive_rescue_archive_require_retention=True,
    )
    ckpt = tmp_path / "bad.pth"
    ckpt.write_bytes(b"x")
    synth = {
        "checkpoint": str(ckpt),
        "policy_idx": 0,
        "nsucc_mean": 40.0,
        "curriculum": {"success_tolerance": 0.020},
        "same_tol_retention_at_save": 0.95,
        # Explicitly invalid retention metadata → fail closed.
        "same_tol_retention_valid_at_save": False,
        "env_frames": 40_000_000,
    }
    obs._select_best_stable_recovery_source = lambda *_a, **_k: (synth, "ok")  # type: ignore
    obs._select_best_floor_recovery_recovery_source = (  # type: ignore
        lambda *_a, **_k: (None, "none")
    )
    obs._select_best_nsucc_recovery_source = lambda *_a, **_k: (None, "none")  # type: ignore
    _seed_adaptive(obs, tol=0.020, mean=10.0, frames=50_000_000)
    decision = obs._select_adaptive_archive_rescue_source(
        env_frames=50_000_000,
        target_tolerance=0.020,
        target_nsucc_mean=10.0,
        population_nsucc_means=[40.0, 10.0],
    )
    assert decision["action"] == "skip"
    reason = str(decision.get("adaptive_rescue_reason", ""))
    # Fail-closed: retention invalid OR newer required fields missing (nsucc_max).
    assert (
        ("retention" in reason)
        or ("nonfinite" in reason)
        or ("nsucc_max" in reason)
    )


def test_ordinary_fallback(tmp_path: Path):
    obs = _make_progress_observer(tmp_path, num_policies=2, policy_idx=1)
    ckpt = tmp_path / "o.pth"
    ckpt.write_bytes(b"o")
    ordinary = _copy_decision(
        source_policy=0,
        tier="relaxed_live",
        checkpoint_type="live",
        source_state={
            "checkpoint": str(ckpt),
            "curriculum": {"success_tolerance": 0.020},
            "nsucc_mean": 30.0,
            "params": {},
        },
    )
    obs._select_adaptive_degradation_rescue_source = (  # type: ignore
        lambda **_k: _skip_decision()
    )
    obs._select_adaptive_archive_rescue_source = (  # type: ignore
        lambda **_k: _skip_decision(adaptive_rescue_reason="no_archive")
    )
    obs._select_source = lambda **_k: ordinary  # type: ignore
    out = obs._select_adaptive_rescue_source_cascade(
        sorted_policies=[0, 1],
        population={},
        env_frames=50_000_000,
        target_tolerance=0.020,
        target_nsucc_mean=10.0,
        n_top=1,
    )
    assert out["action"] == "copy"
    assert out.get("adaptive_rescue_fallback_to_ordinary") is True
    assert out.get("adaptive_rescue_source_kind") == "ordinary_fallback"


def test_source_cascade_order(tmp_path: Path):
    obs = _make_progress_observer(tmp_path, num_policies=2, policy_idx=1)
    ckpt = tmp_path / "c.pth"
    ckpt.write_bytes(b"c")
    live = _copy_decision(
        source_policy=0,
        tier="adaptive_rescue_live",
        checkpoint_type="live",
        source_state={"checkpoint": str(ckpt), "params": {}},
    )
    archive = _copy_decision(
        source_policy=0,
        tier="adaptive_rescue_best_stable_archive",
        checkpoint_type="best_stable_archive",
        source_state={"checkpoint": str(ckpt), "params": {}},
    )
    ordinary = _copy_decision(
        source_policy=0,
        tier="relaxed_live",
        checkpoint_type="live",
        source_state={"checkpoint": str(ckpt), "params": {}},
    )

    obs._select_adaptive_degradation_rescue_source = lambda **_k: live  # type: ignore
    obs._select_adaptive_archive_rescue_source = lambda **_k: archive  # type: ignore
    obs._select_source = lambda **_k: ordinary  # type: ignore
    assert (
        obs._select_adaptive_rescue_source_cascade(
            sorted_policies=[0, 1],
            population={},
            env_frames=1,
            target_tolerance=0.02,
            target_nsucc_mean=1.0,
            n_top=1,
        )["tier"]
        == "adaptive_rescue_live"
    )

    obs._select_adaptive_degradation_rescue_source = (  # type: ignore
        lambda **_k: _skip_decision()
    )
    assert (
        obs._select_adaptive_rescue_source_cascade(
            sorted_policies=[0, 1],
            population={},
            env_frames=1,
            target_tolerance=0.02,
            target_nsucc_mean=1.0,
            n_top=1,
        )["tier"]
        == "adaptive_rescue_best_stable_archive"
    )

    obs._select_adaptive_archive_rescue_source = (  # type: ignore
        lambda **_k: _skip_decision()
    )
    out = obs._select_adaptive_rescue_source_cascade(
        sorted_policies=[0, 1],
        population={},
        env_frames=1,
        target_tolerance=0.02,
        target_nsucc_mean=1.0,
        n_top=1,
    )
    assert out["tier"] == "relaxed_live"
    assert out.get("adaptive_rescue_fallback_to_ordinary") is True


def _drive_ternary_with_rescue(
    obs,
    tmp_path: Path,
    *,
    decision: dict,
    env_frames: int = 50_000_000,
    own_fsr: int = 0,
    seed_degradation: bool = True,
):
    """Minimal bottom-tier ternary path that reaches burn-in / restart gates."""
    from dexpbt_isaaclab.pbt.pbt_curriculum_p1 import WindowSample

    ckpt = Path(decision["source_state"]["checkpoint"])
    if not ckpt.is_file():
        ckpt.write_bytes(b"x")
    population = {}
    for i in range(obs.num_policies):
        mean = 10.0 if i == obs.policy_idx else 40.0 - i
        population[i] = {
            "policy_idx": i,
            "frame": env_frames,
            "frames_at_restart": env_frames - int(own_fsr) if i == obs.policy_idx else 0,
            "iteration": 5,
            "nsucc_mean": mean,
            "nsucc_max": mean + 5,
            "pbt_ranking_score": mean,
            "objective": mean,
            "checkpoint": str(ckpt),
            "lineage_id": "L1",
            "curriculum": {
                "success_tolerance": 0.020,
                "frame_since_restart": int(own_fsr) if i == obs.policy_idx else env_frames,
            },
            "params": dict(_RESTART_PARAMS),
        }
        w = obs._policy_window(i)
        w.samples = [
            WindowSample(
                env_frames=env_frames,
                nsucc_mean=float(population[i]["nsucc_mean"]),
                nsucc_max=float(population[i]["nsucc_max"]),
                success_tolerance=0.020,
                pbt_iteration=1,
                lineage_id="L1",
            )
        ]
    if seed_degradation:
        # Warm then degrade so adaptive rescue is ready.
        _seed_adaptive(
            obs, tol=0.020, mean=30.0, frames=env_frames - 20_000_000, lineage="L1"
        )
        _seed_adaptive(
            obs, tol=0.020, mean=10.0, frames=env_frames - 10_000_000, lineage="L1"
        )
    obs.pbt_consolidation_cfg.enabled = False
    obs.best_policy_workspace.mkdir(parents=True, exist_ok=True)
    obs._select_adaptive_rescue_source_cascade = lambda **_k: decision  # type: ignore
    sorted_policies = list(range(obs.num_policies))
    scores = {i: float(population[i]["pbt_ranking_score"]) for i in sorted_policies}
    # Adaptive target gate allows unavailable / nonpositive skill_delta; do not
    # mock the generic stuck gate (that would hide adaptive branching).
    with mock.patch(
        "dexpbt_isaaclab.pbt.pbt_observer.pbt_copy_allowed_for_live_source",
        return_value=(True, "ok"),
    ):
        obs._ternary_pbt(sorted_policies, scores, population, env_frames)
    return population


def test_emergency_burnin_bypass_on_hard_collapse(tmp_path: Path):
    obs = _make_progress_observer(
        tmp_path,
        num_policies=2,
        policy_idx=1,
        adaptive_rescue_degradation_patience=1,
        adaptive_rescue_bypass_elimination_burnin=True,
        adaptive_rescue_bypass_on_hard_collapse=True,
        adaptive_rescue_min_frames_before_emergency_bypass=10,
        pbt_min_frames_between_restarts=1_000_000_000,
        adaptive_probation_hard_collapse_mean=12.0,
        adaptive_probation_hard_collapse_patience=2,
    )
    ckpt = tmp_path / "src.pth"
    ckpt.write_bytes(b"x")
    env_frames = 100_000_000
    # Arm probation + hard collapse on target (do not re-seed over this).
    _seed_adaptive(obs, tol=0.022, mean=20.0, frames=10_000_000, lineage="L1")
    _seed_adaptive(obs, tol=0.020, mean=18.0, frames=20_000_000, lineage="L1")
    for f in (30_000_000, 40_000_000, 50_000_000):
        _seed_adaptive(obs, tol=0.020, mean=5.0, frames=f, lineage="L1")
    # Extra low-mean ticks so degradation streak is also ready.
    _seed_adaptive(obs, tol=0.020, mean=5.0, frames=60_000_000, lineage="L1")
    state = obs._adaptive_precision_state(1)
    assert state is not None and state[1].hard_collapsed(state[0])

    # Same-tick latch required for emergency bypass (stale streak alone is insufficient).
    obs._adaptive_hard_collapse_event_frame[1] = int(env_frames)

    obs._frames_at_restart = env_frames - 50_000_000  # emergency frames ok
    restarts = {"n": 0}

    def _fake_restart(*_a, **_k):
        restarts["n"] += 1
        return True

    obs._restart_process_with_new_params = _fake_restart  # type: ignore
    decision = _copy_decision(
        source_policy=0,
        tier="adaptive_rescue_live",
        checkpoint_type="live",
        source_state={
            "checkpoint": str(ckpt),
            "curriculum": {"success_tolerance": 0.020},
            "nsucc_mean": 40.0,
            "params": dict(_RESTART_PARAMS),
        },
    )
    before_blocked = int(obs._adaptive_rescue_blocked_count.get(1, 0))
    _drive_ternary_with_rescue(
        obs,
        tmp_path,
        decision=decision,
        env_frames=env_frames,
        own_fsr=0,
        seed_degradation=False,
    )
    assert restarts["n"] == 1
    assert int(obs._adaptive_rescue_burnin_bypassed.get(1, 0)) >= 1
    assert int(obs._adaptive_rescue_blocked_count.get(1, 0)) == before_blocked


def test_ordinary_degradation_no_burnin_bypass(tmp_path: Path):
    obs = _make_progress_observer(
        tmp_path,
        num_policies=2,
        policy_idx=1,
        adaptive_rescue_degradation_patience=1,
        adaptive_rescue_bypass_elimination_burnin=True,
        adaptive_rescue_bypass_on_hard_collapse=True,
        adaptive_rescue_min_frames_before_emergency_bypass=10,
        pbt_min_frames_between_restarts=1_000_000_000,
    )
    ckpt = tmp_path / "src.pth"
    ckpt.write_bytes(b"x")
    env_frames = 100_000_000
    obs._frames_at_restart = env_frames - 50_000_000
    restarts = {"n": 0}
    obs._restart_process_with_new_params = (  # type: ignore
        lambda *_a, **_k: restarts.__setitem__("n", restarts["n"] + 1) or True
    )
    decision = _copy_decision(
        source_policy=0,
        tier="adaptive_rescue_live",
        checkpoint_type="live",
        source_state={
            "checkpoint": str(ckpt),
            "curriculum": {"success_tolerance": 0.020},
            "nsucc_mean": 40.0,
            "params": dict(_RESTART_PARAMS),
        },
    )
    before_blocked = int(obs._adaptive_rescue_blocked_count.get(1, 0))
    _drive_ternary_with_rescue(obs, tmp_path, decision=decision, env_frames=env_frames, own_fsr=0)
    # No hard collapse → burn-in still blocks despite bypass flags.
    assert restarts["n"] == 0
    assert int(obs._adaptive_rescue_blocked_count.get(1, 0)) == before_blocked + 1
    assert int(obs._adaptive_rescue_burnin_bypassed.get(1, 0)) == 0


def test_adaptive_recovery_freeze_tiers():
    recovery_tiers = {
        "late_rescue_live",
        "late_rescue_archive",
        "adaptive_rescue_live",
        "adaptive_rescue_archive",
        "adaptive_rescue_best_stable_archive",
        "adaptive_rescue_floor_recovery_archive",
        "adaptive_rescue_best_nsucc_archive",
    }
    for tier in (
        "adaptive_rescue_live",
        "adaptive_rescue_best_stable_archive",
        "adaptive_rescue_floor_recovery_archive",
        "adaptive_rescue_best_nsucc_archive",
    ):
        assert tier in recovery_tiers

    # Mirror the freeze_env_frames selection expression from restart impl.
    curriculum_freeze_on_recovery_frames = 200_000_000
    curriculum_freeze_frames = 100_000_000
    last_source_checkpoint_type = "live"
    last_source_tier_label = "adaptive_rescue_live"
    freeze_env_frames = (
        curriculum_freeze_on_recovery_frames
        if (
            last_source_checkpoint_type
            in (
                "archive",
                "best_stable_archive",
                "best_floor_recovery_archive",
                "late_rescue_archive",
            )
            or last_source_tier_label in recovery_tiers
        )
        else curriculum_freeze_frames
    )
    assert freeze_env_frames == curriculum_freeze_on_recovery_frames


def test_real_ternary_adaptive_live_gate(tmp_path: Path):
    obs = _make_progress_observer(
        tmp_path,
        num_policies=2,
        policy_idx=1,
        adaptive_rescue_degradation_patience=1,
        adaptive_rescue_source_min_samples=1,
        adaptive_rescue_source_min_stable_ticks=0,
        adaptive_rescue_min_source_advantage_sigma=0.1,
        adaptive_rescue_archive_enable=False,
    )
    from dexpbt_isaaclab.pbt.pbt_curriculum_p1 import (
        CurriculumControlState,
        WindowSample,
        write_curriculum_control,
    )

    ckpt = tmp_path / "s.pth"
    ckpt.write_bytes(b"x")
    write_curriculum_control(
        obs._curriculum_control_dir / "000.json", CurriculumControlState()
    )
    write_curriculum_control(
        obs._curriculum_control_dir / "001.json", CurriculumControlState()
    )
    obs.policy_idx = 0
    _seed_adaptive_retention(
        obs, tol=0.020, mean=40.0, frames=20_000_000, lineage="L1", n=4
    )
    obs.policy_idx = 1
    _seed_adaptive(obs, tol=0.020, mean=30.0, frames=10_000_000, lineage="L1")
    _seed_adaptive(obs, tol=0.020, mean=10.0, frames=20_000_000, lineage="L1")

    env_frames = 20_000_000
    population = {}
    for i, mean in enumerate([40.0, 10.0]):
        population[i] = {
            "policy_idx": i,
            "frame": env_frames,
            "iteration": 5,
            "nsucc_mean": mean,
            "nsucc_max": mean + 5,
            "pbt_ranking_score": mean,
            "objective": mean,
            "checkpoint": str(ckpt),
            "lineage_id": "L1",
            "curriculum": {
                "success_tolerance": 0.020,
                "frame_since_restart": env_frames,
            },
            "params": dict(_RESTART_PARAMS),
        }
        obs._policy_window(i).samples = [
            WindowSample(
                env_frames=env_frames,
                nsucc_mean=mean,
                nsucc_max=mean + 5,
                success_tolerance=0.020,
                pbt_iteration=1,
                lineage_id="L1",
            )
        ]
    obs.pbt_consolidation_cfg.enabled = False
    obs.best_policy_workspace.mkdir(parents=True, exist_ok=True)

    before = int(obs._adaptive_rescue_blocked_count.get(1, 0))
    with mock.patch(
        "dexpbt_isaaclab.pbt.pbt_observer.pbt_copy_allowed_for_live_source",
        return_value=(False, "source_window_too_short"),
    ):
        obs._ternary_pbt([0, 1], {0: 40.0, 1: 10.0}, population, env_frames)
    assert int(obs._adaptive_rescue_blocked_count.get(1, 0)) == before + 1

    selected = {"n": 0}

    def _capture_restart(*_a, **_k):
        selected["n"] += 1
        return True

    obs._restart_process_with_new_params = _capture_restart  # type: ignore
    with mock.patch(
        "dexpbt_isaaclab.pbt.pbt_observer.pbt_copy_allowed_for_live_source",
        return_value=(True, "ok"),
    ):
        obs._ternary_pbt([0, 1], {0: 40.0, 1: 10.0}, population, env_frames)
    assert selected["n"] == 1


def test_real_quota_commit_via_wrapper(tmp_path: Path, monkeypatch):
    import os
    import signal

    obs = _make_progress_observer(tmp_path, num_policies=2, policy_idx=1)
    from dexpbt_isaaclab.pbt.pbt_curriculum_p1 import (
        CurriculumControlState,
        write_curriculum_control,
    )

    ckpt = tmp_path / "src.pth"
    ckpt.write_bytes(b"ckpt-bytes")
    write_curriculum_control(
        obs._curriculum_control_dir / "001.json",
        CurriculumControlState(last_mastered_tolerance=0.02),
    )
    _seed_adaptive(obs, tol=0.020, mean=10.0, frames=10_000_000, lineage="Lq")
    obs._adaptive_lineage_id[1] = "Lq"
    obs._adaptive_rescue_executed_for_lineage[1] = 0
    monkeypatch.setattr(obs, "_rewrite_checkpoint", lambda *a, **k: None)
    monkeypatch.setattr(signal, "raise_signal", lambda *_a, **_k: None)
    monkeypatch.setattr(os, "_exit", lambda *_a, **_k: None)

    ok = obs._restart_process_with_new_params(
        dict(_RESTART_PARAMS),
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
    manifest = Path(obs.train_dir).parent / obs.RESTART_MANIFEST
    assert manifest.is_file()


def test_transaction_rollback_restores_adaptive(tmp_path: Path, monkeypatch):
    import os

    obs = _make_progress_observer(tmp_path, num_policies=2, policy_idx=1)
    from dexpbt_isaaclab.pbt.pbt_curriculum_p1 import (
        CurriculumControlState,
        write_curriculum_control,
    )

    ckpt = tmp_path / "src.pth"
    ckpt.write_bytes(b"ckpt-bytes")
    write_curriculum_control(
        obs._curriculum_control_dir / "001.json",
        CurriculumControlState(last_mastered_tolerance=0.02),
    )
    _seed_adaptive(obs, tol=0.020, mean=10.0, frames=10_000_000, lineage="Lr")
    before_bytes = obs._adaptive_precision_path(1).read_bytes()
    obs._adaptive_rescue_executed_for_lineage[1] = 0
    obs._adaptive_last_rescue_frame[1] = -(10**18)
    monkeypatch.setattr(obs, "_rewrite_checkpoint", lambda *a, **k: None)

    real_replace = os.replace

    def _boom(src, dst):
        if str(dst).endswith("pbt_restart.json"):
            raise OSError("manifest boom")
        return real_replace(src, dst)

    monkeypatch.setattr(os, "replace", _boom)
    ok = obs._restart_process_with_new_params(
        dict(_RESTART_PARAMS),
        str(ckpt),
        0,
        50_000_000,
        "bottom",
        source_curriculum={"success_tolerance": 0.020},
        adaptive_rescue=True,
        adaptive_rescue_decision={"checkpoint_type": "live"},
    )
    assert ok is False
    assert int(obs._adaptive_rescue_executed_for_lineage.get(1, 0)) == 0
    assert int(obs._adaptive_last_rescue_frame.get(1, -(10**18))) == -(10**18)
    assert obs._adaptive_precision_path(1).read_bytes() == before_bytes
    after = json.loads(before_bytes)
    assert after.get("rescue_executed_for_lineage", 0) == 0


# ---------------------------------------------------------------------------
# Adaptive rescue target gate / hard-collapse latch / archive fail-closed
# ---------------------------------------------------------------------------


def _seed_skill_delta_seq(obs, policy_idx, *, seq, env_frames, lineage="Lgate", tol=0.020):
    from dexpbt_isaaclab.pbt.pbt_curriculum_p1 import WindowSample

    w = obs._policy_window(policy_idx)
    w.samples = []
    step = int(obs.interval_steps)
    n = len(seq)
    for i, mean in enumerate(seq):
        w.samples.append(
            WindowSample(
                env_frames=int(env_frames) - (n - 1 - i) * step,
                nsucc_mean=float(mean),
                nsucc_max=float(mean) + 5.0,
                success_tolerance=float(tol),
                pbt_iteration=i,
                lineage_id=lineage,
            )
        )


def test_adaptive_hard_collapse_bypasses_generic_target_skill_delta_gate(tmp_path: Path):
    """Hard collapse + nsucc_mean>>1 must not be blocked by generic stuck gate."""
    from dexpbt_isaaclab.pbt.pbt_control_plane import (
        pbt_copy_allowed_for_adaptive_rescue_target,
        pbt_copy_allowed_for_target,
    )
    from dexpbt_isaaclab.pbt.pbt_curriculum_p1 import (
        CurriculumControlState,
        write_curriculum_control,
    )

    obs = _make_progress_observer(
        tmp_path,
        num_policies=2,
        policy_idx=1,
        adaptive_rescue_degradation_patience=1,
        adaptive_rescue_archive_enable=False,
        adaptive_probation_hard_collapse_mean=12.0,
        adaptive_probation_hard_collapse_patience=2,
    )
    env_frames = 100_000_000
    _seed_skill_delta_seq(
        obs, 1, seq=[8.0] * 6, env_frames=env_frames, lineage="Lh"
    )
    allowed_g, reason_g, _ = pbt_copy_allowed_for_target(
        obs._policy_window(1), interval_steps=obs.interval_steps
    )
    assert not allowed_g
    assert reason_g == "not_stuck"
    allowed_a, reason_a, _ = pbt_copy_allowed_for_adaptive_rescue_target(
        obs._policy_window(1),
        interval_steps=obs.interval_steps,
        hard_collapse=True,
    )
    assert allowed_a
    assert reason_a == "adaptive_hard_collapse"

    # Latch + control same-tick event frame (stale bool alone must not arm gates).
    obs._adaptive_hard_collapse_event_frame[1] = int(env_frames)
    write_curriculum_control(
        obs._curriculum_control_dir / "001.json",
        CurriculumControlState(
            adaptive_probation_hard_collapse=True,
            adaptive_probation_hard_collapse_event_frame=int(env_frames),
        ),
    )
    # Force adaptive rescue mode so the adaptive target gate is used.
    obs._adaptive_degradation_rescue_ready = (  # type: ignore
        lambda **_k: (True, "forced_for_gate_test")
    )

    ckpt = tmp_path / "src.pth"
    ckpt.write_bytes(b"x")
    decision = _copy_decision(
        source_policy=0,
        tier="adaptive_rescue_live",
        checkpoint_type="live",
        source_state={
            "checkpoint": str(ckpt),
            "curriculum": {"success_tolerance": 0.020},
            "nsucc_mean": 40.0,
            "params": dict(_RESTART_PARAMS),
        },
        adaptive_rescue_source_kind="live",
    )
    restarts = {"n": 0}
    obs._restart_process_with_new_params = (  # type: ignore
        lambda *a, **k: restarts.__setitem__("n", restarts["n"] + 1) or True
    )
    _drive_ternary_with_rescue(
        obs,
        tmp_path,
        decision=decision,
        env_frames=env_frames,
        own_fsr=env_frames,
        seed_degradation=False,
    )
    assert restarts["n"] == 1


def test_adaptive_degradation_allows_nonpositive_skill_delta(tmp_path: Path):
    from dexpbt_isaaclab.pbt.pbt_control_plane import (
        pbt_copy_allowed_for_adaptive_rescue_target,
    )

    obs = _make_progress_observer(tmp_path, policy_idx=0)
    _seed_skill_delta_seq(obs, 0, seq=[10.0, 9.0, 8.0, 7.0, 6.0, 5.0], env_frames=60_000_000)
    ok, reason, delta = pbt_copy_allowed_for_adaptive_rescue_target(
        obs._policy_window(0), interval_steps=obs.interval_steps, hard_collapse=False
    )
    assert ok
    assert reason == "ok"
    assert delta is not None and float(delta) <= 0.0


def test_adaptive_degradation_allows_unavailable_skill_delta_when_retention_trigger_valid(
    tmp_path: Path,
):
    from dexpbt_isaaclab.pbt.pbt_control_plane import (
        pbt_copy_allowed_for_adaptive_rescue_target,
    )

    obs = _make_progress_observer(tmp_path, policy_idx=0)
    obs._policy_window(0).samples = []
    ok, reason, delta = pbt_copy_allowed_for_adaptive_rescue_target(
        obs._policy_window(0), interval_steps=obs.interval_steps, hard_collapse=False
    )
    assert ok
    assert reason == "skill_delta_unavailable_ok"
    assert delta is None


def test_adaptive_degradation_blocks_strongly_improving_skill_delta(tmp_path: Path):
    from dexpbt_isaaclab.pbt.pbt_control_plane import (
        pbt_copy_allowed_for_adaptive_rescue_target,
    )
    from dexpbt_isaaclab.pbt.pbt_curriculum_p1 import SKILL_DELTA_IMPROVE_EPS

    obs = _make_progress_observer(tmp_path, policy_idx=0)
    _seed_skill_delta_seq(
        obs, 0, seq=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], env_frames=60_000_000
    )
    ok, reason, delta = pbt_copy_allowed_for_adaptive_rescue_target(
        obs._policy_window(0), interval_steps=obs.interval_steps, hard_collapse=False
    )
    assert not ok
    assert reason == "skill_delta_improving"
    assert delta is not None and float(delta) >= float(SKILL_DELTA_IMPROVE_EPS)


def test_generic_bottom_restart_still_requires_original_target_skill_delta_gate(
    tmp_path: Path,
):
    from dexpbt_isaaclab.pbt.pbt_control_plane import pbt_copy_allowed_for_target

    obs = _make_progress_observer(
        tmp_path,
        num_policies=2,
        policy_idx=1,
        adaptive_precision_consolidation_enable=False,
        adaptive_rescue_enable=False,
    )
    _seed_skill_delta_seq(obs, 1, seq=[8.0] * 6, env_frames=60_000_000)
    ok, reason, _ = pbt_copy_allowed_for_target(
        obs._policy_window(1), interval_steps=obs.interval_steps
    )
    assert not ok
    assert reason == "not_stuck"


def test_full_tick_hard_collapse_latch_survives_streak_reset(tmp_path: Path):
    from dexpbt_isaaclab.pbt.pbt_curriculum_p1 import write_curriculum_control

    obs = _make_progress_observer(
        tmp_path,
        num_policies=2,
        policy_idx=1,
        adaptive_probation_hard_collapse_mean=12.0,
        adaptive_probation_hard_collapse_patience=2,
    )
    env_frames = 140_000_000
    _seed_adaptive(obs, tol=0.022, mean=20.0, frames=10_000_000, lineage="L1")
    _seed_adaptive(obs, tol=0.020, mean=18.0, frames=20_000_000, lineage="L1")
    for f in (30_000_000, 40_000_000, 50_000_000):
        _seed_adaptive(obs, tol=0.020, mean=5.0, frames=f, lineage="L1")
    state = obs._adaptive_precision_state(1)
    assert state is not None and state[1].hard_collapsed(state[0])

    window = _seed_window(obs, tol=0.020, mean=5.0, frames=env_frames, n=2, lineage="L1")
    obs._shrinks_since_rollback[1] = 1
    obs._last_mastered_safe_tolerance[1] = 0.022
    control = obs._build_curriculum_control_state(1, window, env_frames)
    write_curriculum_control(obs._curriculum_control_dir / "001.json", control)
    assert int(obs._adaptive_hard_collapse_event_frame.get(1, -1)) == env_frames
    state2 = obs._adaptive_precision_state(1)
    assert state2 is not None
    assert int(state2[1].hard_collapse_streak) == 0
    assert not state2[1].hard_collapsed(state2[0])
    assert control.adaptive_probation_hard_collapse is True
    assert int(control.adaptive_probation_hard_collapse_event_frame) == int(env_frames)
    from dexpbt_isaaclab.pbt.pbt_curriculum_p1 import write_curriculum_control

    obs = _make_progress_observer(
        tmp_path,
        num_policies=2,
        policy_idx=1,
        adaptive_rescue_degradation_patience=1,
        adaptive_rescue_bypass_elimination_burnin=True,
        adaptive_rescue_bypass_on_hard_collapse=True,
        adaptive_rescue_min_frames_before_emergency_bypass=10,
        pbt_min_frames_between_restarts=1_000_000_000,
        adaptive_probation_hard_collapse_mean=12.0,
        adaptive_probation_hard_collapse_patience=2,
        adaptive_rescue_archive_enable=False,
    )
    ckpt = tmp_path / "src.pth"
    ckpt.write_bytes(b"x")
    env_frames = 100_000_000

    _seed_adaptive(obs, tol=0.022, mean=20.0, frames=10_000_000, lineage="L1")
    _seed_adaptive(obs, tol=0.020, mean=18.0, frames=20_000_000, lineage="L1")
    for f in (30_000_000, 40_000_000, 50_000_000, 60_000_000):
        _seed_adaptive(obs, tol=0.020, mean=5.0, frames=f, lineage="L1")
    assert obs._adaptive_precision_state(1)[1].hard_collapsed(  # type: ignore
        obs._adaptive_precision_state(1)[0]  # type: ignore
    )

    window = _seed_window(obs, tol=0.020, mean=5.0, frames=env_frames, n=2, lineage="L1")
    obs._shrinks_since_rollback[1] = 1
    obs._last_mastered_safe_tolerance[1] = 0.022
    control = obs._build_curriculum_control_state(1, window, env_frames)
    write_curriculum_control(obs._curriculum_control_dir / "001.json", control)
    assert int(obs._adaptive_hard_collapse_event_frame.get(1, -1)) == env_frames
    assert not obs._adaptive_precision_state(1)[1].hard_collapsed(  # type: ignore
        obs._adaptive_precision_state(1)[0]  # type: ignore
    )

    obs._frames_at_restart = env_frames - 50_000_000
    restarts = {"n": 0}
    obs._restart_process_with_new_params = (  # type: ignore
        lambda *a, **k: restarts.__setitem__("n", restarts["n"] + 1) or True
    )
    decision = _copy_decision(
        source_policy=0,
        tier="adaptive_rescue_live",
        checkpoint_type="live",
        source_state={
            "checkpoint": str(ckpt),
            "curriculum": {"success_tolerance": 0.020},
            "nsucc_mean": 40.0,
            "params": dict(_RESTART_PARAMS),
        },
        adaptive_rescue_source_kind="live",
    )
    _drive_ternary_with_rescue(
        obs,
        tmp_path,
        decision=decision,
        env_frames=env_frames,
        own_fsr=0,
        seed_degradation=False,
    )
    assert restarts["n"] == 1
    assert int(obs._adaptive_rescue_emergency_bypass.get(1, 0)) >= 1
    assert int(obs._adaptive_rescue_burnin_bypassed.get(1, 0)) >= 1


def test_ordinary_degradation_does_not_emergency_bypass(tmp_path: Path):
    obs = _make_progress_observer(
        tmp_path,
        num_policies=2,
        policy_idx=1,
        adaptive_rescue_degradation_patience=1,
        adaptive_rescue_bypass_elimination_burnin=True,
        adaptive_rescue_bypass_on_hard_collapse=True,
        adaptive_rescue_min_frames_before_emergency_bypass=10,
        pbt_min_frames_between_restarts=1_000_000_000,
        adaptive_rescue_archive_enable=False,
    )
    ckpt = tmp_path / "src.pth"
    ckpt.write_bytes(b"x")
    env_frames = 100_000_000
    _seed_adaptive(obs, tol=0.020, mean=30.0, frames=10_000_000, lineage="L1")
    _seed_adaptive(obs, tol=0.020, mean=10.0, frames=20_000_000, lineage="L1")
    obs._frames_at_restart = env_frames - 50_000_000
    restarts = {"n": 0}
    obs._restart_process_with_new_params = (  # type: ignore
        lambda *a, **k: restarts.__setitem__("n", restarts["n"] + 1) or True
    )
    decision = _copy_decision(
        source_policy=0,
        tier="adaptive_rescue_live",
        checkpoint_type="live",
        source_state={
            "checkpoint": str(ckpt),
            "curriculum": {"success_tolerance": 0.020},
            "nsucc_mean": 40.0,
            "params": dict(_RESTART_PARAMS),
        },
        adaptive_rescue_source_kind="live",
    )
    before_bypass = int(obs._adaptive_rescue_emergency_bypass.get(1, 0))
    _drive_ternary_with_rescue(
        obs,
        tmp_path,
        decision=decision,
        env_frames=env_frames,
        own_fsr=0,
        seed_degradation=False,
    )
    assert restarts["n"] == 0
    assert int(obs._adaptive_rescue_emergency_bypass.get(1, 0)) == before_bypass
    assert int(obs._adaptive_rescue_burnin_blocked.get(1, 0)) >= 1


def _archive_synth_base(tmp_path: Path, **extra):
    ckpt = tmp_path / "arch.pth"
    ckpt.write_bytes(b"arch")
    synth = {
        "checkpoint": str(ckpt),
        "policy_idx": 0,
        "nsucc_mean": 40.0,
        "nsucc_max": 50.0,
        "pbt_ranking_score": 100.0,
        "curriculum": {"success_tolerance": 0.020},
        "same_tol_retention_at_save": 0.95,
        "same_tol_retention_valid_at_save": True,
        "env_frames": 40_000_000,
        "collapsed": False,
        "hard_collapse_triggered": False,
        "collapse_rollback_active": False,
    }
    synth.update(extra)
    return synth


def test_adaptive_archive_rejects_missing_nsucc_max(tmp_path: Path):
    obs = _make_progress_observer(tmp_path, num_policies=2, policy_idx=1)
    synth = _archive_synth_base(tmp_path)
    del synth["nsucc_max"]
    obs._select_best_stable_recovery_source = lambda *_a, **_k: (synth, "ok")  # type: ignore
    obs._select_best_floor_recovery_recovery_source = (  # type: ignore
        lambda *_a, **_k: (None, "none")
    )
    obs._select_best_nsucc_recovery_source = lambda *_a, **_k: (None, "none")  # type: ignore
    decision = obs._select_adaptive_archive_rescue_source(
        env_frames=50_000_000,
        target_tolerance=0.020,
        target_nsucc_mean=10.0,
        population_nsucc_means=[40.0, 10.0],
    )
    assert decision["action"] == "skip"
    reason = str(decision.get("adaptive_rescue_reason", ""))
    assert ("nsucc_max" in reason) or ("nonfinite" in reason)


def test_adaptive_archive_rejects_missing_age(tmp_path: Path):
    obs = _make_progress_observer(
        tmp_path,
        num_policies=2,
        policy_idx=1,
        adaptive_rescue_archive_require_age=True,
    )
    synth = _archive_synth_base(tmp_path)
    del synth["env_frames"]
    obs._select_best_stable_recovery_source = lambda *_a, **_k: (synth, "ok")  # type: ignore
    obs._select_best_floor_recovery_recovery_source = (  # type: ignore
        lambda *_a, **_k: (None, "none")
    )
    obs._select_best_nsucc_recovery_source = lambda *_a, **_k: (None, "none")  # type: ignore
    decision = obs._select_adaptive_archive_rescue_source(
        env_frames=50_000_000,
        target_tolerance=0.020,
        target_nsucc_mean=10.0,
        population_nsucc_means=[40.0, 10.0],
    )
    assert decision["action"] == "skip"
    reason = str(decision.get("adaptive_rescue_reason", ""))
    assert "archive_age_missing" in reason


def test_adaptive_archive_rejects_missing_collapse_metadata(tmp_path: Path):
    obs = _make_progress_observer(
        tmp_path,
        num_policies=2,
        policy_idx=1,
        adaptive_rescue_archive_require_collapse_metadata=True,
    )
    synth = _archive_synth_base(tmp_path)
    for k in ("collapsed", "hard_collapse_triggered", "collapse_rollback_active"):
        del synth[k]
    obs._select_best_stable_recovery_source = lambda *_a, **_k: (synth, "ok")  # type: ignore
    obs._select_best_floor_recovery_recovery_source = (  # type: ignore
        lambda *_a, **_k: (None, "none")
    )
    obs._select_best_nsucc_recovery_source = lambda *_a, **_k: (None, "none")  # type: ignore
    decision = obs._select_adaptive_archive_rescue_source(
        env_frames=50_000_000,
        target_tolerance=0.020,
        target_nsucc_mean=10.0,
        population_nsucc_means=[40.0, 10.0],
    )
    assert decision["action"] == "skip"
    assert "collapse_metadata" in str(decision.get("adaptive_rescue_reason", ""))


def test_adaptive_archive_rejects_low_advantage(tmp_path: Path):
    obs = _make_progress_observer(
        tmp_path,
        num_policies=2,
        policy_idx=1,
        adaptive_rescue_min_source_advantage_sigma=1.0,
        # Keep absolute floor low so this case isolates the sigma gate.
        adaptive_rescue_archive_min_absolute_advantage=0.0,
    )
    synth = _archive_synth_base(tmp_path, nsucc_mean=11.0)
    obs._select_best_stable_recovery_source = lambda *_a, **_k: (synth, "ok")  # type: ignore
    obs._select_best_floor_recovery_recovery_source = (  # type: ignore
        lambda *_a, **_k: (None, "none")
    )
    obs._select_best_nsucc_recovery_source = lambda *_a, **_k: (None, "none")  # type: ignore
    decision = obs._select_adaptive_archive_rescue_source(
        env_frames=50_000_000,
        target_tolerance=0.020,
        target_nsucc_mean=10.0,
        population_nsucc_means=[40.0, 10.0],
    )
    assert decision["action"] == "skip"
    assert "low_advantage" in str(decision.get("adaptive_rescue_reason", ""))


def test_adaptive_archive_accepts_complete_healthy_metadata(tmp_path: Path):
    obs = _make_progress_observer(
        tmp_path,
        num_policies=2,
        policy_idx=1,
        adaptive_rescue_min_source_advantage_sigma=0.5,
        adaptive_rescue_archive_min_nsucc_max=5.0,
    )
    synth = _archive_synth_base(tmp_path)
    obs._select_best_stable_recovery_source = lambda *_a, **_k: (synth, "ok")  # type: ignore
    obs._select_best_floor_recovery_recovery_source = (  # type: ignore
        lambda *_a, **_k: (None, "none")
    )
    obs._select_best_nsucc_recovery_source = lambda *_a, **_k: (None, "none")  # type: ignore
    decision = obs._select_adaptive_archive_rescue_source(
        env_frames=50_000_000,
        target_tolerance=0.020,
        target_nsucc_mean=10.0,
        population_nsucc_means=[40.0, 10.0],
    )
    assert decision["action"] == "copy"
    assert decision.get("adaptive_rescue_source_kind") == "archive"
    assert float(decision.get("adaptive_source_advantage_sigma", 0.0)) >= 0.5


def test_ordinary_fallback_source_kind_and_quota_accounting(tmp_path: Path):
    obs = _make_progress_observer(
        tmp_path,
        num_policies=2,
        policy_idx=1,
        adaptive_rescue_degradation_patience=1,
        adaptive_rescue_archive_enable=False,
        pbt_min_frames_between_restarts=0,
    )
    ckpt = tmp_path / "o.pth"
    ckpt.write_bytes(b"o")
    ordinary = _copy_decision(
        source_policy=0,
        tier="relaxed_live",
        checkpoint_type="live",
        source_state={
            "checkpoint": str(ckpt),
            "curriculum": {"success_tolerance": 0.020},
            "nsucc_mean": 30.0,
            "params": dict(_RESTART_PARAMS),
        },
    )
    obs._select_adaptive_degradation_rescue_source = (  # type: ignore
        lambda **_k: _skip_decision()
    )
    obs._select_source = lambda **_k: ordinary  # type: ignore
    cascaded = obs._select_adaptive_rescue_source_cascade(
        sorted_policies=[0, 1],
        population={0: {"nsucc_mean": 30.0}, 1: {"nsucc_mean": 10.0}},
        env_frames=50_000_000,
        target_tolerance=0.020,
        target_nsucc_mean=10.0,
        n_top=1,
    )
    assert cascaded.get("adaptive_rescue_source_kind") == "ordinary_fallback"
    assert cascaded.get("adaptive_rescue_fallback_to_ordinary") is True

    before_live = int(obs._adaptive_rescue_live_selected.get(1, 0))
    before_arch = int(obs._adaptive_rescue_archive_selected.get(1, 0))
    before_ord = int(obs._adaptive_rescue_ordinary_fallback.get(1, 0))

    def _ok_restart(*_a, **kwargs):
        if kwargs.get("adaptive_rescue"):
            obs._adaptive_rescue_executed[1] = (
                int(obs._adaptive_rescue_executed.get(1, 0)) + 1
            )
            obs._adaptive_rescue_executed_for_lineage[1] = (
                int(obs._adaptive_rescue_executed_for_lineage.get(1, 0)) + 1
            )
            obs._adaptive_last_rescue_frame[1] = 50_000_000
        return True

    obs._restart_process_with_new_params = _ok_restart  # type: ignore
    _drive_ternary_with_rescue(
        obs,
        tmp_path,
        decision=cascaded,
        env_frames=50_000_000,
        own_fsr=50_000_000,
        seed_degradation=True,
    )
    assert int(obs._adaptive_rescue_ordinary_fallback.get(1, 0)) == before_ord + 1
    assert int(obs._adaptive_rescue_live_selected.get(1, 0)) == before_live
    assert int(obs._adaptive_rescue_archive_selected.get(1, 0)) == before_arch
    assert obs._adaptive_rescue_last_source_kind.get(1) == "ordinary_fallback"
    assert int(obs._adaptive_rescue_executed_for_lineage.get(1, 0)) == 1


def test_ordinary_fallback_transaction_failure_restores_quota(
    tmp_path: Path, monkeypatch
):
    obs = _make_progress_observer(
        tmp_path,
        num_policies=2,
        policy_idx=1,
        adaptive_rescue_degradation_patience=1,
        adaptive_rescue_archive_enable=False,
        pbt_min_frames_between_restarts=0,
    )
    from dexpbt_isaaclab.pbt.pbt_curriculum_p1 import (
        CurriculumControlState,
        write_curriculum_control,
    )

    ckpt = tmp_path / "o.pth"
    ckpt.write_bytes(b"o")
    write_curriculum_control(
        obs._curriculum_control_dir / "001.json",
        CurriculumControlState(last_mastered_tolerance=0.02),
    )
    decision = _copy_decision(
        source_policy=0,
        tier="relaxed_live",
        checkpoint_type="live",
        source_state={
            "checkpoint": str(ckpt),
            "curriculum": {"success_tolerance": 0.020},
            "nsucc_mean": 30.0,
            "params": dict(_RESTART_PARAMS),
        },
        adaptive_rescue_source_kind="ordinary_fallback",
        adaptive_rescue_fallback_to_ordinary=True,
        adaptive_rescue_reason="ordinary_fallback",
    )
    _seed_adaptive(obs, tol=0.020, mean=10.0, frames=10_000_000, lineage="Lq")
    obs._adaptive_rescue_executed_for_lineage[1] = 0
    obs._adaptive_last_rescue_frame[1] = -(10**18)
    before_ord = int(obs._adaptive_rescue_ordinary_fallback.get(1, 0))
    before_sel = int(obs._adaptive_rescue_selected_count.get(1, 0))
    before_bytes = obs._adaptive_precision_path(1).read_bytes()

    def _boom(*_a, **_k):
        raise RuntimeError("ordinary_fallback_restart_failed")

    monkeypatch.setattr(obs, "_restart_process_with_new_params_impl", _boom)
    _drive_ternary_with_rescue(
        obs,
        tmp_path,
        decision=decision,
        env_frames=50_000_000,
        own_fsr=50_000_000,
        seed_degradation=True,
    )
    assert int(obs._adaptive_rescue_executed_for_lineage.get(1, 0)) == 0
    assert int(obs._adaptive_last_rescue_frame.get(1, -(10**18))) == -(10**18)
    assert int(obs._adaptive_rescue_ordinary_fallback.get(1, 0)) == before_ord
    assert int(obs._adaptive_rescue_selected_count.get(1, 0)) == before_sel
    after = json.loads(obs._adaptive_precision_path(1).read_text())
    before = json.loads(before_bytes)
    assert after.get("rescue_executed_for_lineage", 0) == before.get(
        "rescue_executed_for_lineage", 0
    )
    assert after.get("last_rescue_frame", -(10**18)) == before.get(
        "last_rescue_frame", -(10**18)
    )


# ---------------------------------------------------------------------------
# Adaptive rescue safety hardening (archive age / advantage / HC arm / live)
# ---------------------------------------------------------------------------


def test_archive_missing_age_rejected_when_required(tmp_path: Path):
    obs = _make_progress_observer(
        tmp_path,
        num_policies=2,
        policy_idx=1,
        adaptive_rescue_archive_require_age=True,
    )
    synth = _archive_synth_base(tmp_path)
    del synth["env_frames"]
    obs._select_best_stable_recovery_source = lambda *_a, **_k: (synth, "ok")  # type: ignore
    obs._select_best_floor_recovery_recovery_source = (  # type: ignore
        lambda *_a, **_k: (None, "none")
    )
    obs._select_best_nsucc_recovery_source = lambda *_a, **_k: (None, "none")  # type: ignore
    decision = obs._select_adaptive_archive_rescue_source(
        env_frames=50_000_000,
        target_tolerance=0.020,
        target_nsucc_mean=10.0,
        population_nsucc_means=[40.0, 10.0],
    )
    assert decision["action"] == "skip"
    assert "archive_age_missing" in str(decision.get("adaptive_rescue_reason", ""))


def test_archive_missing_age_allowed_when_not_required(tmp_path: Path):
    obs = _make_progress_observer(
        tmp_path,
        num_policies=2,
        policy_idx=1,
        adaptive_rescue_archive_require_age=False,
        adaptive_rescue_min_source_advantage_sigma=0.5,
        adaptive_rescue_archive_min_absolute_advantage=5.0,
    )
    synth = _archive_synth_base(tmp_path)
    del synth["env_frames"]
    obs._select_best_stable_recovery_source = lambda *_a, **_k: (synth, "ok")  # type: ignore
    obs._select_best_floor_recovery_recovery_source = (  # type: ignore
        lambda *_a, **_k: (None, "none")
    )
    obs._select_best_nsucc_recovery_source = lambda *_a, **_k: (None, "none")  # type: ignore
    decision = obs._select_adaptive_archive_rescue_source(
        env_frames=50_000_000,
        target_tolerance=0.020,
        target_nsucc_mean=10.0,
        population_nsucc_means=[40.0, 10.0],
    )
    assert decision["action"] == "copy"
    assert decision.get("adaptive_archive_age_known") is False
    assert int(decision.get("adaptive_archive_age_frames", 0)) == -1


def test_archive_future_age_rejected_even_when_not_required(tmp_path: Path):
    obs = _make_progress_observer(
        tmp_path,
        num_policies=2,
        policy_idx=1,
        adaptive_rescue_archive_require_age=False,
    )
    synth = _archive_synth_base(tmp_path, env_frames=80_000_000)
    obs._select_best_stable_recovery_source = lambda *_a, **_k: (synth, "ok")  # type: ignore
    obs._select_best_floor_recovery_recovery_source = (  # type: ignore
        lambda *_a, **_k: (None, "none")
    )
    obs._select_best_nsucc_recovery_source = lambda *_a, **_k: (None, "none")  # type: ignore
    decision = obs._select_adaptive_archive_rescue_source(
        env_frames=50_000_000,
        target_tolerance=0.020,
        target_nsucc_mean=10.0,
        population_nsucc_means=[40.0, 10.0],
    )
    assert decision["action"] == "skip"
    assert "archive_age_from_future" in str(decision.get("adaptive_rescue_reason", ""))


def test_archive_too_old_rejected_even_when_not_required(tmp_path: Path):
    obs = _make_progress_observer(
        tmp_path,
        num_policies=2,
        policy_idx=1,
        adaptive_rescue_archive_require_age=False,
        adaptive_rescue_archive_max_age_frames=1_000_000,
    )
    synth = _archive_synth_base(tmp_path, env_frames=1_000_000)
    obs._select_best_stable_recovery_source = lambda *_a, **_k: (synth, "ok")  # type: ignore
    obs._select_best_floor_recovery_recovery_source = (  # type: ignore
        lambda *_a, **_k: (None, "none")
    )
    obs._select_best_nsucc_recovery_source = lambda *_a, **_k: (None, "none")  # type: ignore
    decision = obs._select_adaptive_archive_rescue_source(
        env_frames=50_000_000,
        target_tolerance=0.020,
        target_nsucc_mean=10.0,
        population_nsucc_means=[40.0, 10.0],
    )
    assert decision["action"] == "skip"
    assert "archive_too_old" in str(decision.get("adaptive_rescue_reason", ""))


def test_archive_rejects_insufficient_population_samples(tmp_path: Path):
    obs = _make_progress_observer(
        tmp_path,
        num_policies=2,
        policy_idx=1,
        adaptive_rescue_archive_min_population_samples=2,
    )
    synth = _archive_synth_base(tmp_path)
    obs._select_best_stable_recovery_source = lambda *_a, **_k: (synth, "ok")  # type: ignore
    obs._select_best_floor_recovery_recovery_source = (  # type: ignore
        lambda *_a, **_k: (None, "none")
    )
    obs._select_best_nsucc_recovery_source = lambda *_a, **_k: (None, "none")  # type: ignore
    decision = obs._select_adaptive_archive_rescue_source(
        env_frames=50_000_000,
        target_tolerance=0.020,
        target_nsucc_mean=10.0,
        population_nsucc_means=[40.0],
    )
    assert decision["action"] == "skip"
    assert decision.get("adaptive_rescue_reason") == "archive_population_stats_insufficient"


def test_archive_zero_variance_small_absolute_gap_rejected(tmp_path: Path):
    obs = _make_progress_observer(
        tmp_path,
        num_policies=2,
        policy_idx=1,
        adaptive_rescue_archive_min_absolute_advantage=5.0,
        adaptive_rescue_archive_advantage_sigma_floor=1.0,
        adaptive_rescue_min_source_advantage_sigma=1.0,
    )
    synth = _archive_synth_base(tmp_path, nsucc_mean=11.0)
    obs._select_best_stable_recovery_source = lambda *_a, **_k: (synth, "ok")  # type: ignore
    obs._select_best_floor_recovery_recovery_source = (  # type: ignore
        lambda *_a, **_k: (None, "none")
    )
    obs._select_best_nsucc_recovery_source = lambda *_a, **_k: (None, "none")  # type: ignore
    decision = obs._select_adaptive_archive_rescue_source(
        env_frames=50_000_000,
        target_tolerance=0.020,
        target_nsucc_mean=10.0,
        population_nsucc_means=[10.0, 10.0],
    )
    assert decision["action"] == "skip"
    reason = str(decision.get("adaptive_rescue_reason", ""))
    assert ("low_absolute_advantage" in reason) or ("low_advantage" in reason)


def test_archive_zero_variance_large_gap_uses_sigma_floor(tmp_path: Path):
    obs = _make_progress_observer(
        tmp_path,
        num_policies=2,
        policy_idx=1,
        adaptive_rescue_archive_min_absolute_advantage=5.0,
        adaptive_rescue_archive_advantage_sigma_floor=1.0,
        adaptive_rescue_min_source_advantage_sigma=1.0,
    )
    synth = _archive_synth_base(tmp_path, nsucc_mean=20.0)
    obs._select_best_stable_recovery_source = lambda *_a, **_k: (synth, "ok")  # type: ignore
    obs._select_best_floor_recovery_recovery_source = (  # type: ignore
        lambda *_a, **_k: (None, "none")
    )
    obs._select_best_nsucc_recovery_source = lambda *_a, **_k: (None, "none")  # type: ignore
    decision = obs._select_adaptive_archive_rescue_source(
        env_frames=50_000_000,
        target_tolerance=0.020,
        target_nsucc_mean=10.0,
        population_nsucc_means=[10.0, 10.0],
    )
    assert decision["action"] == "copy"
    assert float(decision.get("adaptive_population_nsucc_sigma", -1)) == 0.0
    assert float(decision.get("adaptive_population_nsucc_effective_sigma", 0)) == 1.0
    assert float(decision.get("adaptive_source_absolute_advantage", 0)) == 10.0
    assert float(decision.get("adaptive_source_advantage_sigma", 0)) == 10.0


def test_archive_advantage_never_inf(tmp_path: Path):
    import math

    obs = _make_progress_observer(
        tmp_path,
        num_policies=2,
        policy_idx=1,
        adaptive_rescue_archive_min_absolute_advantage=0.0,
        adaptive_rescue_archive_advantage_sigma_floor=1.0,
        adaptive_rescue_min_source_advantage_sigma=0.0,
    )
    synth = _archive_synth_base(tmp_path, nsucc_mean=40.0)
    obs._select_best_stable_recovery_source = lambda *_a, **_k: (synth, "ok")  # type: ignore
    obs._select_best_floor_recovery_recovery_source = (  # type: ignore
        lambda *_a, **_k: (None, "none")
    )
    obs._select_best_nsucc_recovery_source = lambda *_a, **_k: (None, "none")  # type: ignore
    decision = obs._select_adaptive_archive_rescue_source(
        env_frames=50_000_000,
        target_tolerance=0.020,
        target_nsucc_mean=10.0,
        population_nsucc_means=[10.0, 10.0],
    )
    assert decision["action"] == "copy"
    adv = float(decision.get("adaptive_source_advantage_sigma", float("nan")))
    assert math.isfinite(adv)
    assert adv != float("inf")


def test_archive_complete_candidate_passes_both_advantage_gates(tmp_path: Path):
    obs = _make_progress_observer(
        tmp_path,
        num_policies=2,
        policy_idx=1,
        adaptive_rescue_archive_min_absolute_advantage=5.0,
        adaptive_rescue_archive_advantage_sigma_floor=1.0,
        adaptive_rescue_min_source_advantage_sigma=1.0,
    )
    synth = _archive_synth_base(tmp_path, nsucc_mean=40.0)
    obs._select_best_stable_recovery_source = lambda *_a, **_k: (synth, "ok")  # type: ignore
    obs._select_best_floor_recovery_recovery_source = (  # type: ignore
        lambda *_a, **_k: (None, "none")
    )
    obs._select_best_nsucc_recovery_source = lambda *_a, **_k: (None, "none")  # type: ignore
    decision = obs._select_adaptive_archive_rescue_source(
        env_frames=50_000_000,
        target_tolerance=0.020,
        target_nsucc_mean=10.0,
        population_nsucc_means=[40.0, 10.0],
    )
    assert decision["action"] == "copy"
    assert float(decision.get("adaptive_source_absolute_advantage", 0)) >= 5.0
    assert float(decision.get("adaptive_source_advantage_sigma", 0)) >= 1.0
    assert int(decision.get("adaptive_population_nsucc_sample_count", 0)) == 2


def test_hard_collapse_control_event_frame_written(tmp_path: Path):
    from dexpbt_isaaclab.pbt.pbt_curriculum_p1 import write_curriculum_control

    obs = _make_progress_observer(
        tmp_path,
        num_policies=2,
        policy_idx=1,
        adaptive_probation_hard_collapse_mean=12.0,
        adaptive_probation_hard_collapse_patience=2,
    )
    env_frames = 140_000_000
    _seed_adaptive(obs, tol=0.022, mean=20.0, frames=10_000_000, lineage="L1")
    _seed_adaptive(obs, tol=0.020, mean=18.0, frames=20_000_000, lineage="L1")
    for f in (30_000_000, 40_000_000, 50_000_000):
        _seed_adaptive(obs, tol=0.020, mean=5.0, frames=f, lineage="L1")
    window = _seed_window(obs, tol=0.020, mean=5.0, frames=env_frames, n=2, lineage="L1")
    obs._shrinks_since_rollback[1] = 1
    obs._last_mastered_safe_tolerance[1] = 0.022
    control = obs._build_curriculum_control_state(1, window, env_frames)
    write_curriculum_control(obs._curriculum_control_dir / "001.json", control)
    assert control.adaptive_probation_hard_collapse is True
    assert int(control.adaptive_probation_hard_collapse_event_frame) == int(env_frames)


def test_hard_collapse_event_frame_cleared_next_tick(tmp_path: Path):
    obs = _make_progress_observer(
        tmp_path,
        num_policies=2,
        policy_idx=1,
        adaptive_probation_hard_collapse_mean=12.0,
        adaptive_probation_hard_collapse_patience=2,
    )
    env_frames = 140_000_000
    _seed_adaptive(obs, tol=0.022, mean=20.0, frames=10_000_000, lineage="L1")
    _seed_adaptive(obs, tol=0.020, mean=18.0, frames=20_000_000, lineage="L1")
    for f in (30_000_000, 40_000_000, 50_000_000):
        _seed_adaptive(obs, tol=0.020, mean=5.0, frames=f, lineage="L1")
    window = _seed_window(obs, tol=0.020, mean=5.0, frames=env_frames, n=2, lineage="L1")
    obs._shrinks_since_rollback[1] = 1
    obs._last_mastered_safe_tolerance[1] = 0.022
    control = obs._build_curriculum_control_state(1, window, env_frames)
    assert int(control.adaptive_probation_hard_collapse_event_frame) == int(env_frames)
    # Next tick: streak already cleared → no HC → event_frame must be -1.
    window2 = _seed_window(
        obs, tol=0.022, mean=18.0, frames=env_frames + 10_000_000, n=2, lineage="L1"
    )
    control2 = obs._build_curriculum_control_state(1, window2, env_frames + 10_000_000)
    assert int(control2.adaptive_probation_hard_collapse_event_frame) == -1
    assert control2.adaptive_probation_hard_collapse is False


def test_stale_hard_collapse_bool_cannot_trigger_rescue(tmp_path: Path):
    from dexpbt_isaaclab.pbt.pbt_curriculum_p1 import (
        CurriculumControlState,
        WindowSample,
        write_curriculum_control,
    )

    obs = _make_progress_observer(
        tmp_path,
        num_policies=2,
        policy_idx=1,
        adaptive_rescue_degradation_patience=99,
        adaptive_rescue_cooldown_frames=0,
        adaptive_rescue_archive_enable=False,
    )
    env_frames = 100_000_000
    # Stale bool from a previous tick, with event_frame cleared.
    write_curriculum_control(
        obs._curriculum_control_dir / "001.json",
        CurriculumControlState(
            adaptive_probation_hard_collapse=True,
            adaptive_probation_hard_collapse_event_frame=-1,
        ),
    )
    obs._adaptive_hard_collapse_event_frame[1] = -1
    ready, reason = obs._adaptive_degradation_rescue_ready(
        policy_idx=1,
        tolerance=0.020,
        nsucc_mean=10.0,
        env_frames=env_frames,
        lineage_id="L1",
        hard_collapse_now=False,
    )
    assert ready is False
    assert reason != "hard_collapse_ready"

    # Ternary same-tick detection must also ignore stale bool.
    cascade_calls = {"n": 0}

    def _cascade(**_k):
        cascade_calls["n"] += 1
        return _skip_decision(adaptive_rescue_reason="no_compatible_source")

    obs._select_adaptive_rescue_source_cascade = _cascade  # type: ignore
    obs._late_degradation_rescue_ready = lambda *_a, **_k: False  # type: ignore
    obs._maybe_save_best_policy = lambda *_a, **_k: None  # type: ignore
    obs._select_source = lambda **_k: _skip_decision()  # type: ignore
    obs.pbt_consolidation_cfg.enabled = False
    ckpt = tmp_path / "x.pth"
    ckpt.write_bytes(b"x")
    population = {
        0: {
            "policy_idx": 0,
            "frame": env_frames,
            "frames_at_restart": 0,
            "iteration": 1,
            "nsucc_mean": 40.0,
            "nsucc_max": 45.0,
            "pbt_ranking_score": 40.0,
            "objective": 40.0,
            "checkpoint": str(ckpt),
            "lineage_id": "L1",
            "curriculum": {"success_tolerance": 0.020, "frame_since_restart": env_frames},
            "params": dict(_RESTART_PARAMS),
        },
        1: {
            "policy_idx": 1,
            "frame": env_frames,
            "frames_at_restart": env_frames,
            "iteration": 1,
            "nsucc_mean": 5.0,
            "nsucc_max": 8.0,
            "pbt_ranking_score": 5.0,
            "objective": 5.0,
            "checkpoint": str(ckpt),
            "lineage_id": "L1",
            "curriculum": {"success_tolerance": 0.020, "frame_since_restart": 0},
            "params": dict(_RESTART_PARAMS),
        },
    }
    for i in (0, 1):
        w = obs._policy_window(i)
        w.samples = [
            WindowSample(
                env_frames=env_frames,
                nsucc_mean=float(population[i]["nsucc_mean"]),
                nsucc_max=float(population[i]["nsucc_max"]),
                success_tolerance=0.020,
                pbt_iteration=1,
                lineage_id="L1",
            )
        ]
    scores = {0: 40.0, 1: 5.0}
    obs._ternary_pbt([0, 1], scores, population, env_frames)
    assert cascade_calls["n"] == 0


def test_current_control_event_frame_arms_rescue(tmp_path: Path):
    from dexpbt_isaaclab.pbt.pbt_curriculum_p1 import (
        CurriculumControlState,
        write_curriculum_control,
    )

    obs = _make_progress_observer(
        tmp_path,
        num_policies=2,
        policy_idx=1,
        adaptive_rescue_degradation_patience=99,
        adaptive_rescue_cooldown_frames=0,
        adaptive_rescue_max_per_lineage=1,
        adaptive_rescue_archive_enable=False,
    )
    env_frames = 100_000_000
    write_curriculum_control(
        obs._curriculum_control_dir / "001.json",
        CurriculumControlState(
            adaptive_probation_hard_collapse=True,
            adaptive_probation_hard_collapse_event_frame=int(env_frames),
        ),
    )
    ready, reason = obs._adaptive_degradation_rescue_ready(
        policy_idx=1,
        tolerance=0.020,
        nsucc_mean=10.0,
        env_frames=env_frames,
        lineage_id="L1",
        hard_collapse_now=True,
    )
    assert ready is True
    assert reason == "hard_collapse_ready"


def test_full_tick_hard_collapse_directly_enters_source_cascade(tmp_path: Path):
    """Real order: adaptive update → build/write control → ternary → cascade.

    Must NOT mock ``_adaptive_degradation_rescue_ready``.
    """
    from dexpbt_isaaclab.pbt.pbt_curriculum_p1 import write_curriculum_control

    obs = _make_progress_observer(
        tmp_path,
        num_policies=2,
        policy_idx=1,
        adaptive_rescue_degradation_patience=99,  # retention path not ready
        adaptive_rescue_cooldown_frames=0,
        adaptive_rescue_archive_enable=False,
        adaptive_probation_hard_collapse_mean=12.0,
        adaptive_probation_hard_collapse_patience=2,
        pbt_min_frames_between_restarts=0,
        adaptive_rescue_bypass_elimination_burnin=True,
        adaptive_rescue_bypass_on_hard_collapse=True,
        adaptive_rescue_min_frames_before_emergency_bypass=10,
    )
    ckpt = tmp_path / "src.pth"
    ckpt.write_bytes(b"x")
    env_frames = 100_000_000

    _seed_adaptive(obs, tol=0.022, mean=20.0, frames=10_000_000, lineage="L1")
    _seed_adaptive(obs, tol=0.020, mean=18.0, frames=20_000_000, lineage="L1")
    for f in (30_000_000, 40_000_000, 50_000_000, 60_000_000):
        _seed_adaptive(obs, tol=0.020, mean=5.0, frames=f, lineage="L1")

    window = _seed_window(obs, tol=0.020, mean=5.0, frames=env_frames, n=2, lineage="L1")
    obs._shrinks_since_rollback[1] = 1
    obs._last_mastered_safe_tolerance[1] = 0.022
    control = obs._build_curriculum_control_state(1, window, env_frames)
    write_curriculum_control(obs._curriculum_control_dir / "001.json", control)
    assert int(control.adaptive_probation_hard_collapse_event_frame) == int(env_frames)
    assert int(obs._adaptive_hard_collapse_event_frame.get(1, -1)) == int(env_frames)

    cascade_calls = {"n": 0, "kind": None}
    decision = _copy_decision(
        source_policy=0,
        tier="adaptive_rescue_live",
        checkpoint_type="live",
        source_state={
            "checkpoint": str(ckpt),
            "curriculum": {"success_tolerance": 0.020},
            "nsucc_mean": 40.0,
            "params": dict(_RESTART_PARAMS),
        },
        adaptive_rescue_source_kind="live",
    )

    def _cascade(**_k):
        cascade_calls["n"] += 1
        cascade_calls["kind"] = "entered"
        return decision

    obs._select_adaptive_rescue_source_cascade = _cascade  # type: ignore
    # Do not mock readiness — hard collapse must arm it.
    obs._maybe_save_best_policy = lambda *_a, **_k: None  # type: ignore
    obs.pbt_consolidation_cfg.enabled = False
    obs.best_policy_workspace.mkdir(parents=True, exist_ok=True)
    obs._frames_at_restart = env_frames - 50_000_000
    restarts = {"n": 0}
    obs._restart_process_with_new_params = (  # type: ignore
        lambda *a, **k: restarts.__setitem__("n", restarts["n"] + 1) or True
    )
    population = {
        0: {
            "policy_idx": 0,
            "frame": env_frames,
            "frames_at_restart": 0,
            "iteration": 5,
            "nsucc_mean": 40.0,
            "nsucc_max": 45.0,
            "pbt_ranking_score": 40.0,
            "objective": 40.0,
            "checkpoint": str(ckpt),
            "lineage_id": "L1",
            "curriculum": {
                "success_tolerance": 0.020,
                "frame_since_restart": env_frames,
            },
            "params": dict(_RESTART_PARAMS),
        },
        1: {
            "policy_idx": 1,
            "frame": env_frames,
            "frames_at_restart": env_frames - 50_000_000,
            "iteration": 5,
            "nsucc_mean": 5.0,
            "nsucc_max": 8.0,
            "pbt_ranking_score": 5.0,
            "objective": 5.0,
            "checkpoint": str(ckpt),
            "lineage_id": "L1",
            "curriculum": {"success_tolerance": 0.020, "frame_since_restart": 0},
            "params": dict(_RESTART_PARAMS),
        },
    }
    from dexpbt_isaaclab.pbt.pbt_curriculum_p1 import WindowSample

    for i in (0, 1):
        w = obs._policy_window(i)
        w.samples = [
            WindowSample(
                env_frames=env_frames,
                nsucc_mean=float(population[i]["nsucc_mean"]),
                nsucc_max=float(population[i]["nsucc_max"]),
                success_tolerance=0.020,
                pbt_iteration=1,
                lineage_id="L1",
            )
        ]
    with mock.patch(
        "dexpbt_isaaclab.pbt.pbt_observer.pbt_copy_allowed_for_live_source",
        return_value=(True, "ok"),
    ):
        obs._ternary_pbt([0, 1], {0: 40.0, 1: 5.0}, population, env_frames)
    assert cascade_calls["n"] == 1
    assert cascade_calls["kind"] == "entered"
    # Cascade entry with patience=99 proves hard_collapse_ready armed rescue
    # (retention path cannot be ready). Decision may overwrite last_reason.
    assert restarts["n"] == 1


def test_latest_healthy_source_is_classified_as_live():
    from dexpbt_isaaclab.pbt.pbt_observer import _decision_uses_live_source

    assert _decision_uses_live_source(
        {"checkpoint_type": "latest_healthy", "tier": "usable_live"}
    )
    assert _decision_uses_live_source(
        {"checkpoint_type": "latest_healthy", "tier": "strict_stable_live"}
    )
    assert _decision_uses_live_source({"checkpoint_type": "live", "tier": "strict_live"})
    assert not _decision_uses_live_source(
        {"checkpoint_type": "archive", "tier": "adaptive_rescue_best_nsucc_archive"}
    )
    assert not _decision_uses_live_source(
        {
            "checkpoint_type": "best_stable_archive",
            "tier": "adaptive_rescue_best_stable_archive",
        }
    )


def test_ordinary_fallback_latest_healthy_runs_source_skill_delta_gate(tmp_path: Path):
    from dexpbt_isaaclab.pbt.pbt_curriculum_p1 import WindowSample

    obs = _make_progress_observer(
        tmp_path,
        num_policies=2,
        policy_idx=1,
        adaptive_rescue_degradation_patience=1,
        adaptive_rescue_archive_enable=False,
    )
    ckpt = tmp_path / "s.pth"
    ckpt.write_bytes(b"x")
    env_frames = 80_000_000
    decision = _copy_decision(
        source_policy=0,
        tier="usable_live",
        checkpoint_type="latest_healthy",
        source_state={
            "checkpoint": str(ckpt),
            "curriculum": {"success_tolerance": 0.020},
            "nsucc_mean": 40.0,
            "params": dict(_RESTART_PARAMS),
        },
        adaptive_rescue_source_kind="ordinary_fallback",
        adaptive_rescue_fallback_to_ordinary=True,
    )
    gate_calls = {"n": 0}

    def _gate(*_a, **_k):
        gate_calls["n"] += 1
        return False, "source_skill_delta_blocked"

    _seed_adaptive(obs, tol=0.020, mean=30.0, frames=env_frames - 20_000_000, lineage="L1")
    _seed_adaptive(obs, tol=0.020, mean=10.0, frames=env_frames - 10_000_000, lineage="L1")
    obs.pbt_consolidation_cfg.enabled = False
    obs._maybe_save_best_policy = lambda *_a, **_k: None  # type: ignore
    obs._select_adaptive_rescue_source_cascade = lambda **_k: decision  # type: ignore
    population = {}
    for i in range(2):
        mean = 10.0 if i == 1 else 40.0
        population[i] = {
            "policy_idx": i,
            "frame": env_frames,
            "frames_at_restart": 0,
            "iteration": 5,
            "nsucc_mean": mean,
            "nsucc_max": mean + 5,
            "pbt_ranking_score": mean,
            "objective": mean,
            "checkpoint": str(ckpt),
            "lineage_id": "L1",
            "curriculum": {
                "success_tolerance": 0.020,
                "frame_since_restart": env_frames,
            },
            "params": dict(_RESTART_PARAMS),
        }
        w = obs._policy_window(i)
        w.samples = [
            WindowSample(
                env_frames=env_frames,
                nsucc_mean=mean,
                nsucc_max=mean + 5,
                success_tolerance=0.020,
                pbt_iteration=1,
                lineage_id="L1",
            )
        ]
    with mock.patch(
        "dexpbt_isaaclab.pbt.pbt_observer.pbt_copy_allowed_for_live_source",
        side_effect=_gate,
    ):
        obs._ternary_pbt([0, 1], {0: 40.0, 1: 10.0}, population, env_frames)
    assert gate_calls["n"] >= 1


def test_ordinary_fallback_unstable_latest_healthy_is_blocked(tmp_path: Path):
    from dexpbt_isaaclab.pbt.pbt_control_plane import pbt_copy_allowed_for_live_source
    from dexpbt_isaaclab.pbt.pbt_curriculum_p1 import WindowSample

    obs = _make_progress_observer(
        tmp_path,
        num_policies=2,
        policy_idx=1,
        adaptive_rescue_degradation_patience=1,
        adaptive_rescue_archive_enable=False,
        pbt_min_frames_between_restarts=0,
    )
    ckpt = tmp_path / "s.pth"
    ckpt.write_bytes(b"x")
    env_frames = 80_000_000
    # Declining source: skill_delta not source-qualified.
    _seed_skill_delta_seq(
        obs, 0, seq=[24.0, 20.0, 16.0, 12.0, 8.0, 5.0], env_frames=env_frames
    )
    ok, reason = pbt_copy_allowed_for_live_source(
        obs._policy_window(0), interval_steps=obs.interval_steps
    )
    assert not ok
    assert "source_skill_delta" in reason

    decision = _copy_decision(
        source_policy=0,
        tier="usable_live",
        checkpoint_type="latest_healthy",
        source_state={
            "checkpoint": str(ckpt),
            "curriculum": {"success_tolerance": 0.020},
            "nsucc_mean": 5.0,
            "params": dict(_RESTART_PARAMS),
        },
        adaptive_rescue_source_kind="ordinary_fallback",
        adaptive_rescue_fallback_to_ordinary=True,
    )
    restarts = {"n": 0}
    obs._restart_process_with_new_params = (  # type: ignore
        lambda *a, **k: restarts.__setitem__("n", restarts["n"] + 1) or True
    )
    population = {}
    for i in range(2):
        mean = 5.0 if i == 1 else 5.0
        population[i] = {
            "policy_idx": i,
            "frame": env_frames,
            "frames_at_restart": 0,
            "iteration": 5,
            "nsucc_mean": mean,
            "nsucc_max": mean + 5,
            "pbt_ranking_score": 40.0 if i == 0 else 5.0,
            "objective": 40.0 if i == 0 else 5.0,
            "checkpoint": str(ckpt),
            "lineage_id": "L1",
            "curriculum": {
                "success_tolerance": 0.020,
                "frame_since_restart": env_frames,
            },
            "params": dict(_RESTART_PARAMS),
        }
        if i == 1:
            w = obs._policy_window(i)
            w.samples = [
                WindowSample(
                    env_frames=env_frames,
                    nsucc_mean=mean,
                    nsucc_max=mean + 5,
                    success_tolerance=0.020,
                    pbt_iteration=1,
                    lineage_id="L1",
                )
            ]
    _seed_adaptive(obs, tol=0.020, mean=30.0, frames=env_frames - 20_000_000, lineage="L1")
    _seed_adaptive(obs, tol=0.020, mean=5.0, frames=env_frames - 10_000_000, lineage="L1")
    obs.pbt_consolidation_cfg.enabled = False
    obs._maybe_save_best_policy = lambda *_a, **_k: None  # type: ignore
    obs._select_adaptive_rescue_source_cascade = lambda **_k: decision  # type: ignore
    obs._ternary_pbt([0, 1], {0: 40.0, 1: 5.0}, population, env_frames)
    assert restarts["n"] == 0


def test_ordinary_fallback_stable_latest_healthy_is_allowed(tmp_path: Path):
    from dexpbt_isaaclab.pbt.pbt_curriculum_p1 import WindowSample

    obs = _make_progress_observer(
        tmp_path,
        num_policies=2,
        policy_idx=1,
        adaptive_rescue_degradation_patience=1,
        adaptive_rescue_archive_enable=False,
        pbt_min_frames_between_restarts=0,
    )
    ckpt = tmp_path / "s.pth"
    ckpt.write_bytes(b"x")
    env_frames = 80_000_000
    # Mild positive skill_delta qualifies as a live source (>= STABLE_POS_EPS).
    _seed_skill_delta_seq(
        obs, 0, seq=[19.0, 19.2, 19.4, 19.6, 19.8, 20.0], env_frames=env_frames
    )
    decision = _copy_decision(
        source_policy=0,
        tier="usable_live",
        checkpoint_type="latest_healthy",
        source_state={
            "checkpoint": str(ckpt),
            "curriculum": {"success_tolerance": 0.020},
            "nsucc_mean": 20.0,
            "params": dict(_RESTART_PARAMS),
        },
        adaptive_rescue_source_kind="ordinary_fallback",
        adaptive_rescue_fallback_to_ordinary=True,
    )
    restarts = {"n": 0}
    obs._restart_process_with_new_params = (  # type: ignore
        lambda *a, **k: restarts.__setitem__("n", restarts["n"] + 1) or True
    )
    _seed_adaptive(obs, tol=0.020, mean=30.0, frames=env_frames - 20_000_000, lineage="L1")
    _seed_adaptive(obs, tol=0.020, mean=10.0, frames=env_frames - 10_000_000, lineage="L1")
    obs.pbt_consolidation_cfg.enabled = False
    obs._maybe_save_best_policy = lambda *_a, **_k: None  # type: ignore
    obs._select_adaptive_rescue_source_cascade = lambda **_k: decision  # type: ignore
    population = {}
    for i in range(2):
        mean = 10.0 if i == 1 else 20.0
        population[i] = {
            "policy_idx": i,
            "frame": env_frames,
            "frames_at_restart": 0 if i == 0 else env_frames,
            "iteration": 5,
            "nsucc_mean": mean,
            "nsucc_max": mean + 5,
            "pbt_ranking_score": mean,
            "objective": mean,
            "checkpoint": str(ckpt),
            "lineage_id": "L1",
            "curriculum": {
                "success_tolerance": 0.020,
                "frame_since_restart": env_frames,
            },
            "params": dict(_RESTART_PARAMS),
        }
        if i == 1:
            w = obs._policy_window(i)
            w.samples = [
                WindowSample(
                    env_frames=env_frames,
                    nsucc_mean=mean,
                    nsucc_max=mean + 5,
                    success_tolerance=0.020,
                    pbt_iteration=1,
                    lineage_id="L1",
                )
            ]
    with mock.patch(
        "dexpbt_isaaclab.pbt.pbt_observer.pbt_copy_allowed_for_adaptive_rescue_target",
        return_value=(True, "ok", 0.0),
    ):
        obs._ternary_pbt([0, 1], {0: 20.0, 1: 10.0}, population, env_frames)
    assert restarts["n"] == 1


def test_archive_source_does_not_run_live_skill_delta_gate(tmp_path: Path):
    obs = _make_progress_observer(
        tmp_path,
        num_policies=2,
        policy_idx=1,
        adaptive_rescue_degradation_patience=1,
        pbt_min_frames_between_restarts=0,
        adaptive_rescue_bypass_elimination_burnin=True,
        adaptive_rescue_bypass_on_hard_collapse=True,
        adaptive_rescue_min_frames_before_emergency_bypass=10,
    )
    ckpt = tmp_path / "a.pth"
    ckpt.write_bytes(b"x")
    env_frames = 100_000_000
    obs._adaptive_hard_collapse_event_frame[1] = int(env_frames)
    decision = _copy_decision(
        source_policy=0,
        tier="adaptive_rescue_best_stable_archive",
        checkpoint_type="best_stable_archive",
        source_state={
            "checkpoint": str(ckpt),
            "curriculum": {"success_tolerance": 0.020},
            "nsucc_mean": 40.0,
            "params": dict(_RESTART_PARAMS),
        },
        adaptive_rescue_source_kind="archive",
    )
    gate_calls = {"n": 0}

    def _gate(*_a, **_k):
        gate_calls["n"] += 1
        return False, "should_not_run"

    restarts = {"n": 0}
    obs._restart_process_with_new_params = (  # type: ignore
        lambda *a, **k: restarts.__setitem__("n", restarts["n"] + 1) or True
    )
    with mock.patch(
        "dexpbt_isaaclab.pbt.pbt_observer.pbt_copy_allowed_for_live_source",
        side_effect=_gate,
    ):
        _drive_ternary_with_rescue(
            obs,
            tmp_path,
            decision=decision,
            env_frames=env_frames,
            own_fsr=0,
            seed_degradation=True,
        )
    assert gate_calls["n"] == 0
    assert restarts["n"] == 1
