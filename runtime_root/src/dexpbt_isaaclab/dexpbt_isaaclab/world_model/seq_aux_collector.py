"""
Per-step aux collector for sequence world models.

Sits between the raw env and ``RlGamesVecEnvWrapper`` (similar to
``WMInterceptorEnv``), but aggregates aux scalars into a rolling ``(H, N)``
ring that mirrors the PPO horizon. After each rollout the collector exposes
the full window plus the most-recent ``last_*`` snapshot, which the
SeqWMTrainer copies into the corresponding ExperienceBuffer aux fields.

Why a separate collector and not a play_steps_rnn patch:

* Less invasive — does not touch ``rl_games``.
* Captures both the (H, N) window AND the post-rollout step-H value used
  to construct ``next_obs[T-1]`` / ``next_d_hat[T-1]`` in SequenceBatch.
* Works identically in CF child subprocesses where the same ``train.py``
  + sequence trainer is reused.

Aligned to ``DEFAULT_SEQ_AUX_FIELDS`` in ``wm_aux_buffer``: by default
collects ``aux_d_hat``, ``aux_failure``, ``aux_lifted``, ``aux_success_cnt``.
"""

from __future__ import annotations

from typing import Any, Callable, Mapping, Optional

import torch

from .wm_aux_buffer import DEFAULT_SEQ_AUX_FIELDS, extract_step_aux


class SeqAuxCollector:
    """Rolling (H, N, K) buffer of per-step aux scalars."""

    def __init__(
        self,
        *,
        num_envs: int,
        horizon_length: int,
        device: torch.device | str,
        fields: Optional[Mapping[str, tuple[int, ...]]] = None,
    ) -> None:
        if fields is None:
            fields = DEFAULT_SEQ_AUX_FIELDS
        self.num_envs = int(num_envs)
        self.horizon_length = int(horizon_length)
        self.device = torch.device(device)
        self.fields = dict(fields)
        # Allocate a (H, N, *shape) tensor per field.
        self._storage: dict[str, torch.Tensor] = {
            name: torch.zeros(
                (self.horizon_length, self.num_envs, *shape),
                dtype=torch.float32, device=self.device,
            )
            for name, shape in self.fields.items()
        }
        # Last-step values (i.e. "step H" aux) for next_d_hat / next_success_cnt.
        self._last: dict[str, torch.Tensor] = {
            name: torch.zeros(
                (self.num_envs, *shape),
                dtype=torch.float32, device=self.device,
            )
            for name, shape in self.fields.items()
        }
        # Step counter relative to the current rollout. Reset by reset_window().
        self._t = 0

    # ----- writing -----

    def reset_window(self) -> None:
        """Begin a new rollout window. Does NOT zero the storage (cheap)."""
        self._t = 0

    def on_step(
        self,
        infos: Mapping[str, Any],
        env_attrs: Optional[Mapping[str, Any]] = None,
    ) -> None:
        """Record one step's aux values at index ``t``."""
        if self._t >= self.horizon_length:
            # Should not happen if reset_window() is called per rollout, but
            # be defensive: silently clamp to last index. Caller logs warn.
            t = self.horizon_length - 1
        else:
            t = self._t
        aux = extract_step_aux(infos, device=self.device, env_attrs=env_attrs,
                                fields=self.fields)
        for name, val in aux.items():
            store = self._storage.get(name)
            if store is None:
                continue
            store[t].copy_(val)
            self._last[name].copy_(val)
        self._t += 1

    def record_terminal(
        self,
        infos: Mapping[str, Any],
        env_attrs: Optional[Mapping[str, Any]] = None,
    ) -> None:
        """Record the post-rollout (step H) aux snapshot, used as last_*.

        This is called by the patched train_epoch AFTER play_steps_rnn has
        completed, when the env's current obs is what we'll use for
        next_obs[T-1].
        """
        aux = extract_step_aux(infos, device=self.device, env_attrs=env_attrs,
                                fields=self.fields)
        for name, val in aux.items():
            if name in self._last:
                self._last[name].copy_(val)

    # ----- reading -----

    def full_window(self) -> dict[str, torch.Tensor]:
        """Return a shallow dict of (H, N, *shape) tensors. Caller should not
        keep references across rollouts (the storage is reused)."""
        return dict(self._storage)

    def last_step(self) -> dict[str, torch.Tensor]:
        return dict(self._last)


class SeqAuxInterceptorEnv:
    """Wrap an env to feed each step's infos / env_attrs into a SeqAuxCollector.

    Mirrors the pattern used by WMInterceptorEnv (used by the legacy WM data
    collector). The wrapper is otherwise transparent: it forwards
    step / reset / attribute access to the underlying env.

    Phase A4 P0 (Task C, terminal aux alignment):

    The wrapper also retains the most recent ``infos`` and ``env_attrs`` it
    saw on its last ``step()``. After ``play_steps_rnn`` returns the
    aux of step ``H`` (post-rollout) is ``last_infos`` / ``last_env_attrs``,
    not ``storage[H-1]`` — the patched ``play_steps_rnn`` in
    ``train.py`` will read these via ``last_infos`` / ``last_env_attrs``
    and call ``collector.record_terminal()``.

    Without this, ``last_d_hat`` / ``last_success_count`` were ``storage[H-1]``
    (off-by-one), biasing every ``next_d_hat[T-1]`` / ``next_success_count[T-1]``
    target seen by the seq WM trainer.
    """

    def __init__(
        self,
        env: Any,
        collector: SeqAuxCollector,
        env_attrs_provider: Optional[Callable[[Any], dict[str, Any]]] = None,
    ) -> None:
        self._env = env
        self._collector = collector
        # env_attrs_provider(env) -> dict of names like {"lifted_object":...,
        # "successes":...}. None means rely on infos["wm"] only.
        self._env_attrs_provider = env_attrs_provider
        self._last_infos: Optional[Mapping[str, Any]] = None
        self._last_env_attrs: Optional[Mapping[str, Any]] = None

    def step(self, actions):
        obs, rew, terminated, truncated, infos = self._env.step(actions)
        env_attrs = (
            self._env_attrs_provider(self._env) if self._env_attrs_provider else None
        )
        self._collector.on_step(infos, env_attrs)
        # Retain the last step's infos / env_attrs so the patched
        # play_steps_rnn can call collector.record_terminal() with
        # the post-rollout (step H) snapshot.
        self._last_infos = infos
        self._last_env_attrs = env_attrs
        return obs, rew, terminated, truncated, infos

    def reset(self, *args, **kwargs):
        return self._env.reset(*args, **kwargs)

    @property
    def last_infos(self) -> Optional[Mapping[str, Any]]:
        return self._last_infos

    @property
    def last_env_attrs(self) -> Optional[Mapping[str, Any]]:
        return self._last_env_attrs

    @property
    def collector(self) -> SeqAuxCollector:
        return self._collector

    def __getattr__(self, name: str):
        # Delegate everything else to the underlying env.
        if name in (
            "_env",
            "_collector",
            "_env_attrs_provider",
            "_last_infos",
            "_last_env_attrs",
        ):
            raise AttributeError(name)
        return getattr(self._env, name)


__all__ = ["SeqAuxCollector", "SeqAuxInterceptorEnv"]
