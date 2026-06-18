"""Env interceptor for Online Elite-AWR episode capture and delayed labeling."""

from __future__ import annotations

import math
from typing import Any, Callable, Mapping, Optional

import torch

from dexpbt_isaaclab.awr.elite_awr import (
    DoneReason,
    EliteAWRBuffer,
    EliteAWRConfig,
    build_sequence_entry_from_episode,
    compute_elite_awr_quality,
)


def _extract_env_scalar(env_attrs: Mapping[str, Any] | None, name: str):
    if not env_attrs or name not in env_attrs:
        return None
    val = env_attrs[name]
    if isinstance(val, torch.Tensor):
        if val.numel() == 1:
            return val.item()
        return val
    return val


def _per_env_bool(val, env_idx: int) -> bool:
    if val is None:
        return False
    if isinstance(val, torch.Tensor):
        if val.ndim == 0:
            return bool(val.item())
        return bool(val[env_idx].item())
    if isinstance(val, (list, tuple)):
        return bool(val[env_idx])
    return bool(val)


def _per_env_float(val, env_idx: int, default: float = 0.0) -> float:
    if val is None:
        return default
    if isinstance(val, torch.Tensor):
        if val.ndim == 0:
            return float(val.item())
        return float(val[env_idx].item())
    if isinstance(val, (list, tuple)):
        return float(val[env_idx])
    return float(val)


def classify_done_reason(
    *,
    terminated: Any,
    truncated: Any,
    env_idx: int,
    infos: Mapping[str, Any] | None,
    env_attrs: Mapping[str, Any] | None,
) -> str:
    term = _per_env_bool(terminated, env_idx)
    trunc = _per_env_bool(truncated, env_idx)
    if not term and not trunc:
        return DoneReason.UNKNOWN.value

    wm = (infos or {}).get("wm") if isinstance(infos, dict) else None
    if isinstance(wm, dict):
        if wm.get("success") is True or wm.get("episode_success") is True:
            return DoneReason.SUCCESS.value
        if wm.get("failure") is True or wm.get("drop") is True:
            return DoneReason.FAILURE.value

    successes = _extract_env_scalar(env_attrs, "successes")
    if successes is not None:
        s = _per_env_float(successes, env_idx, 0.0)
        if s > 0 and term:
            return DoneReason.SUCCESS.value

    if trunc and not term:
        return DoneReason.TIMEOUT.value
    return DoneReason.UNKNOWN.value


class EliteAWRInterceptorEnv:
    """Capture rollout obs/action segments for Elite-AWR delayed labeling."""

    def __init__(
        self,
        env: Any,
        buffer: EliteAWRBuffer,
        cfg: EliteAWRConfig,
        *,
        num_envs: int,
        policy_idx: int = 0,
        profile_name: str = "",
        env_attrs_provider: Optional[Callable[[Any], dict[str, Any]]] = None,
        source_meta_provider: Optional[Callable[[], dict[str, Any]]] = None,
        device: torch.device | str = "cpu",
    ) -> None:
        self._env = env
        self._buffer = buffer
        self._cfg = cfg
        self._num_envs = int(num_envs)
        self._policy_idx = int(policy_idx)
        self._profile_name = str(profile_name)
        self._env_attrs_provider = env_attrs_provider
        self._source_meta_provider = source_meta_provider
        self._device = torch.device(device)
        self._last_obs: torch.Tensor | None = None
        self._pending: list[list[dict[str, Any]]] = [[] for _ in range(self._num_envs)]
        self._rollout_actions: list[torch.Tensor] = []
        self.telemetry: dict[str, float] = {
            "awr/info_missing_lifted": 0.0,
            "awr/info_missing_success": 0.0,
            "awr/info_missing_distance": 0.0,
            "awr/info_missing_tolerance": 0.0,
            "awr/done_reason_unknown_frac": 0.0,
            "awr/source_nsucc_missing_frac": 0.0,
        }
        self._done_unknown = 0
        self._done_total = 0
        self._source_nsucc_missing = 0

    @property
    def buffer(self) -> EliteAWRBuffer:
        return self._buffer

    def reset_rollout_window(self) -> None:
        self._rollout_actions.clear()

    def consume_rollout_actions(self) -> torch.Tensor | None:
        if not self._rollout_actions:
            return None
        return torch.stack(self._rollout_actions, dim=0)

    def _source_meta(self) -> dict[str, Any]:
        if self._source_meta_provider is None:
            return {}
        try:
            return dict(self._source_meta_provider() or {})
        except Exception:
            return {}

    def _obs_tensor(self, obs) -> torch.Tensor:
        if isinstance(obs, tuple):
            obs = obs[0]
        if isinstance(obs, dict):
            if "obs" in obs:
                obs = obs["obs"]
            elif "policy" in obs:
                obs = obs["policy"]
            else:
                obs = next(iter(obs.values()))
        t = obs if isinstance(obs, torch.Tensor) else torch.as_tensor(obs)
        if t.device != self._device:
            t = t.to(self._device)
        return t.detach()

    def reset(self, *args, **kwargs):
        out = self._env.reset(*args, **kwargs)
        obs = out[0] if isinstance(out, tuple) else out
        self._last_obs = self._obs_tensor(obs)
        for i in range(self._num_envs):
            self._pending[i].clear()
        return out

    def step(self, actions):
        if self._last_obs is None:
            raise RuntimeError("EliteAWRInterceptorEnv.step called before reset")
        obs_before = self._last_obs
        act = actions if isinstance(actions, torch.Tensor) else torch.as_tensor(
            actions, device=self._device
        )
        if act.device != self._device:
            act = act.to(self._device)
        act = act.detach()
        self._rollout_actions.append(act.detach().cpu())

        obs, rew, terminated, truncated, infos = self._env.step(actions)
        env_attrs = (
            self._env_attrs_provider(self._env) if self._env_attrs_provider else None
        )
        meta = self._source_meta()
        source_nsucc = meta.get("source_nsucc")
        if source_nsucc is None:
            self._source_nsucc_missing += 1

        info_missing = {
            "lifted": env_attrs is None or "lifted_object" not in env_attrs,
            "success": env_attrs is None or "successes" not in env_attrs,
            "distance": env_attrs is None or "closest_d_hat" not in env_attrs,
            "tolerance": env_attrs is None or "success_tolerance" not in env_attrs,
        }
        for k, v in info_missing.items():
            if v:
                self.telemetry[f"awr/info_missing_{k if k != 'distance' else 'distance'}"] += 1.0

        obs_after = self._obs_tensor(obs)
        for env_idx in range(self._num_envs):
            lifted = _per_env_bool(
                _extract_env_scalar(env_attrs, "lifted_object"), env_idx
            )
            tol = _per_env_float(
                _extract_env_scalar(env_attrs, "success_tolerance"), env_idx, 0.075
            )
            d_hat = _per_env_float(
                _extract_env_scalar(env_attrs, "closest_d_hat"), env_idx, float("nan")
            )
            wm = infos.get("wm") if isinstance(infos, dict) else None
            drop = False
            if isinstance(wm, dict):
                drop = _per_env_bool(wm.get("failure"), env_idx) or _per_env_bool(
                    wm.get("drop"), env_idx
                )
            step_rec = {
                "obs": obs_before[env_idx].detach().cpu(),
                "action": act[env_idx].detach().cpu(),
                "lifted": lifted,
                "drop": drop,
                "tolerance": tol,
                "d_hat": d_hat,
                "info_missing": info_missing,
                "quality_step_pass": (not drop) and (lifted or tol <= 0.06),
            }
            self._pending[env_idx].append(step_rec)

            done = _per_env_bool(terminated, env_idx) or _per_env_bool(truncated, env_idx)
            if done:
                self._finalize_env(
                    env_idx,
                    terminated=terminated,
                    truncated=truncated,
                    infos=infos,
                    env_attrs=env_attrs,
                    meta=meta,
                )

        self._last_obs = obs_after
        return obs, rew, terminated, truncated, infos

    def _finalize_env(
        self,
        env_idx: int,
        *,
        terminated,
        truncated,
        infos,
        env_attrs,
        meta: dict[str, Any],
    ) -> None:
        pending = self._pending[env_idx]
        if not pending:
            return
        done_reason = classify_done_reason(
            terminated=terminated,
            truncated=truncated,
            env_idx=env_idx,
            infos=infos,
            env_attrs=env_attrs,
        )
        self._done_total += 1
        if done_reason == DoneReason.UNKNOWN.value:
            self._done_unknown += 1

        episode_success = done_reason == DoneReason.SUCCESS.value
        horizon = min(100, len(pending))
        tail_start = max(0, len(pending) - horizon)
        initial_success = _per_env_float(
            _extract_env_scalar(env_attrs, "successes"), env_idx, 0.0
        )
        _ = initial_success  # reserved for future success-lift labeling

        for i, step in enumerate(pending):
            future_slice = pending[i : min(len(pending), i + 101)]
            future_success_within_100 = episode_success and (i >= len(pending) - 100)
            if done_reason == DoneReason.UNKNOWN.value:
                future_success_within_100 = False
            step["future_success_within_100"] = future_success_within_100
            step["future_success"] = episode_success and i >= tail_start
            if done_reason != DoneReason.SUCCESS.value:
                step["future_success"] = False
            current_d = float(step.get("d_hat", float("nan")))
            if not math.isfinite(current_d):
                step["future_improvement"] = 0.0
                continue
            future_d_vals = [
                float(s.get("d_hat", float("nan")))
                for s in future_slice
                if math.isfinite(float(s.get("d_hat", float("nan"))))
            ]
            if not future_d_vals:
                step["future_improvement"] = 0.0
            else:
                step["future_improvement"] = float(current_d - min(future_d_vals))

        tol_min = min(float(s.get("tolerance", 0.075)) for s in pending)
        lifted_frac = sum(1 for s in pending if s.get("lifted")) / max(len(pending), 1)
        no_drop = all(not s.get("drop") for s in pending)
        had_drop = any(s.get("drop") for s in pending)
        source_nsucc = meta.get("source_nsucc")
        if isinstance(source_nsucc, torch.Tensor):
            source_nsucc = float(source_nsucc.mean().item())

        accept = False
        if any(s.get("future_success_within_100") for s in pending):
            accept = True
        elif lifted_frac >= 0.5 and no_drop and tol_min <= 0.06:
            accept = True
        elif any(float(s.get("future_improvement", 0.0)) > 0.01 for s in pending):
            accept = True
        elif source_nsucc is not None and source_nsucc >= 30.0 and no_drop:
            accept = True
        elif episode_success and len(pending) <= 100:
            accept = True

        if had_drop and not any(s.get("future_success_within_100") for s in pending):
            accept = False
        if source_nsucc is not None and source_nsucc < 10.0:
            if not any(s.get("future_success_within_100") for s in pending) and not any(
                float(s.get("future_improvement", 0.0)) > 0.0 for s in pending
            ):
                accept = False

        q_preview = compute_elite_awr_quality(
            future_success_within_100=any(
                s.get("future_success_within_100") for s in pending
            ),
            lifted_frac=lifted_frac,
            no_drop=no_drop,
            future_improvement=max(float(s.get("future_improvement", 0.0)) for s in pending),
            tolerance_min=tol_min,
            source_nsucc=source_nsucc if source_nsucc is not None else None,
            had_drop=had_drop,
        )
        if q_preview < self._cfg.quality_min:
            accept = False

        if accept:
            entry = build_sequence_entry_from_episode(
                pending,
                cfg=self._cfg,
                source_policy=self._policy_idx,
                source_profile=self._profile_name,
                source_nsucc=source_nsucc,
                source_retention=meta.get("source_retention"),
                source_type=str(meta.get("source_type") or "local"),
                env_frames=int(meta.get("env_frames") or 0),
                done_reason=done_reason,
            )
            if entry is not None:
                self._buffer.insert(entry)

        self._pending[env_idx].clear()
        self.telemetry["awr/done_reason_unknown_frac"] = self._done_unknown / max(
            self._done_total, 1
        )
        self.telemetry["awr/source_nsucc_missing_frac"] = self._source_nsucc_missing / max(
            self._done_total, 1
        )

    def __getattr__(self, name: str):
        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(self._env, name)


__all__ = ["EliteAWRInterceptorEnv", "classify_done_reason", "DoneReason"]
