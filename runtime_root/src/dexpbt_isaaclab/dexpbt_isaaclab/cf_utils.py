"""Patch C (rev 3, 2026-05-30): CF child frame-accounting helpers.

Kept in a standalone module (no Isaac Sim / gymnasium imports) so the
helpers can be unit-tested without bringing up the full Omniverse stack.
"""

from __future__ import annotations


def compute_child_train_frames(
    runner,
    cf_tracker,
    *,
    num_envs: int,
    horizon_length: int,
) -> tuple[int, str]:
    """Determine how many env-frames a CF child actually consumed during
    its `agent.train()` call.

    Tries three signals in priority order so we can still report a real
    frame count even when one of them is missing or zero:

      1. ``runner.algo.frame`` (the live rl_games counter after train).
      2. ``runner._end_frame - runner._start_frame`` (boundary stash on
         ``_WmEnabledRunner.run_train``).
      3. ``len(cf_tracker.samples) * num_envs * horizon_length`` (samples
         estimate based on PPO epochs observed by
         ``_patch_actor_train_for_cf_meta``).

    Returns (frames, method) where method is one of:
      ``"algo_frame"`` | ``"end_minus_start"`` | ``"samples_estimate"``
      | ``"zero"``.

    Pure helper -- safe to unit-test without Isaac Sim.
    """
    algo = (
        getattr(runner, "algo", None)
        or getattr(runner, "algorithm", None)
    )
    try:
        algo_frame = int(getattr(algo, "frame", 0) or 0) if algo is not None else 0
    except Exception:
        algo_frame = 0
    if algo_frame > 0:
        return algo_frame, "algo_frame"

    try:
        start_f = int(getattr(runner, "_start_frame", 0) or 0)
        end_f = int(getattr(runner, "_end_frame", 0) or 0)
    except Exception:
        start_f, end_f = 0, 0
    boundary = max(0, end_f - start_f)
    if boundary > 0:
        return boundary, "end_minus_start"

    if cf_tracker is not None:
        try:
            n_samples = len(cf_tracker.samples)
        except Exception:
            n_samples = 0
        if n_samples > 0 and num_envs > 0 and horizon_length > 0:
            return (
                int(n_samples) * int(num_envs) * int(horizon_length),
                "samples_estimate",
            )

    return 0, "zero"
