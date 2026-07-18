#!/usr/bin/env python3
"""CCRS paired-reset rollout collection helpers (no Isaac import).

Implements the v2 reset protocol pieces that must be unit-testable without
booting Isaac:

- FirstEpisodeCollector: records only the FIRST completed episode of each
  env_id < num_episodes (paired-reset protocol, num_episodes <= num_envs);
- extract_reset_fingerprints: per-env fingerprint of the initial task state
  (object/goal pose + robot joint positions), taken AFTER the initial reset
  and BEFORE the first action;
- TerminalSnapshotHook: eval-only wrapper around env._reset_idx that
  snapshots the pre-reset terminal error / dropped flag for envs being
  reset. Installed only inside the CCRS evaluator process; the training
  environment code is not modified. Default OFF (must be explicitly
  installed).

All tensors are accessed via duck typing so tests can use plain torch
tensors on CPU.
"""

from __future__ import annotations

from typing import Any, Callable, Mapping, Optional

from .eval_protocol import (
    EpisodeSample,
    FINGERPRINT_QUANT,
    fingerprint_env_state,
)

# Env attributes hashed into the reset fingerprint (per env, flattened).
# All are refreshed by _compute_intermediate_values() which the env calls at
# the end of _reset_idx, so they reflect the post-reset initial state.
FINGERPRINT_FIELDS = (
    "object_pos",   # (N, 3)
    "object_rot",   # (N, 4)
    "goal_pos",     # (N, 3)
    "goal_rot",     # (N, 4)
    "dof_pos",      # (N, num_joints) robot joint positions
)


def extract_reset_fingerprints(
    env: Any,
    env_ids: list[int],
    fields: tuple[str, ...] = FINGERPRINT_FIELDS,
    quant: float = FINGERPRINT_QUANT,
) -> dict[int, str]:
    """Per-env fingerprints from the env's current (post-reset) buffers.

    Raises AttributeError if a required field is missing — fingerprints must
    not silently degrade to fewer fields.
    """
    tensors = {}
    for name in fields:
        t = getattr(env, name)  # AttributeError = loud failure, intended
        tensors[name] = t.detach().cpu().numpy() if hasattr(t, "detach") else t
    out: dict[int, str] = {}
    for i in env_ids:
        field_map: dict[str, list[float]] = {}
        for name, arr in tensors.items():
            row = arr[i]
            flat = row.reshape(-1).tolist() if hasattr(row, "reshape") else list(row)
            field_map[name] = [float(v) for v in flat]
        out[i] = fingerprint_env_state(field_map, quant)
    return out


class FirstEpisodeCollector:
    """Collect the first completed episode per env_id (paired-reset protocol).

    Only env_ids in [0, num_episodes) are eligible so the sample_id set is
    identical across all units sharing (reset_seed, num_envs, num_episodes).
    Later episodes of the same env are ignored.
    """

    def __init__(self, reset_seed: int, num_episodes: int,
                 max_consecutive_successes: int) -> None:
        self.reset_seed = int(reset_seed)
        self.num_episodes = int(num_episodes)
        self.max_consecutive_successes = int(max_consecutive_successes)
        self.seen_env_ids: set[int] = set()
        self.samples: list[EpisodeSample] = []

    def offer(
        self,
        env_id: int,
        nsucc: float,
        terminal_error: float = float("nan"),
        terminal_error_valid: bool = False,
        dropped: Optional[bool] = None,
    ) -> bool:
        """Offer a completed episode; returns True if it was recorded."""
        env_id = int(env_id)
        if env_id >= self.num_episodes or env_id in self.seen_env_ids:
            return False
        self.seen_env_ids.add(env_id)
        self.samples.append(EpisodeSample.create(
            reset_seed=self.reset_seed,
            env_id=env_id,
            nsucc=nsucc,
            max_consecutive_successes=self.max_consecutive_successes,
            terminal_error=terminal_error,
            terminal_error_valid=terminal_error_valid,
            dropped=dropped,
        ))
        return True

    @property
    def complete(self) -> bool:
        return len(self.samples) == self.num_episodes

    @property
    def missing_env_ids(self) -> list[int]:
        return sorted(set(range(self.num_episodes)) - self.seen_env_ids)


class TerminalSnapshotHook:
    """Eval-only pre-reset terminal snapshot via _reset_idx wrapping.

    In the DexPBT direct env, step() computes _get_dones() (which refreshes
    object_pos / goal_pos via _compute_intermediate_values) and THEN calls
    _reset_idx(done_ids), whose tail re-runs _compute_intermediate_values.
    So after step() returns, those buffers already hold POST-reset state for
    done envs — reading them there is invalid (this was the v1 bug).

    This hook wraps env._reset_idx: at call time the buffers still hold the
    pre-reset terminal state, so it snapshots terminal error and dropped
    flags for exactly the env_ids being reset, then calls the original.

    Strictly eval-only: nothing in the training code path references this
    class; it must be explicitly installed on a live env instance, and
    uninstall() restores the original method.
    """

    def __init__(self, env: Any,
                 error_fn: Optional[Callable[[Any, Any], Any]] = None,
                 drop_height: float = 0.1) -> None:
        self._env = env
        self._orig_reset_idx: Optional[Callable] = None
        self._error_fn = error_fn
        self._drop_height = float(drop_height)
        self.available = self._check_available(env)
        # env_id -> (terminal_error, dropped); overwritten on later resets,
        # consumers should read right after step() and clear.
        self.pending: dict[int, tuple[float, bool]] = {}
        self.installed = False

    @staticmethod
    def _check_available(env: Any) -> bool:
        return (hasattr(env, "_reset_idx")
                and hasattr(env, "object_pos")
                and hasattr(env, "goal_pos"))

    def install(self) -> bool:
        """Install the wrapper; returns availability. No-op when the env
        lacks the required attributes (terminal data then stays invalid)."""
        if not self.available or self.installed:
            return self.available and self.installed
        env = self._env
        self._orig_reset_idx = env._reset_idx

        def wrapped(env_ids):
            ids = env_ids
            if ids is None:
                ids_list: list[int] = list(range(env.num_envs))
            elif hasattr(ids, "tolist"):
                ids_list = [int(v) for v in ids.tolist()]
            else:
                ids_list = [int(v) for v in ids]
            try:
                if self._error_fn is not None:
                    errs = self._error_fn(env, ids_list)
                else:
                    diff = env.object_pos - env.goal_pos
                    norm = (diff ** 2).sum(-1) ** 0.5
                    errs = [float(norm[i]) for i in ids_list]
                zs = [float(env.object_pos[i][2]) for i in ids_list]
                for i, e, z in zip(ids_list, errs, zs):
                    self.pending[i] = (float(e), bool(z < self._drop_height))
            except Exception:
                # snapshot failure must never break the rollout; affected
                # envs simply keep terminal_error_valid=False
                pass
            return self._orig_reset_idx(env_ids)

        env._reset_idx = wrapped
        self.installed = True
        return True

    def uninstall(self) -> None:
        if self.installed and self._orig_reset_idx is not None:
            self._env._reset_idx = self._orig_reset_idx
            self.installed = False

    def take(self, env_id: int) -> tuple[float, bool, Optional[bool]]:
        """(terminal_error, valid, dropped) for env_id; clears the slot."""
        if env_id in self.pending:
            err, dropped = self.pending.pop(env_id)
            return err, True, dropped
        return float("nan"), False, None
