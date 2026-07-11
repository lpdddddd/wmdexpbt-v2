"""
WMDataCollector: intercepts env.step() to capture transitions.
WMInterceptorEnv: wraps env to provide wm_prev via infos["wm_prev"].
"""

import torch
from torch import Tensor
from typing import TYPE_CHECKING, Union

if TYPE_CHECKING:
    from .wm_buffer import WMBuffer


def _extract_tensor(x: Union[torch.Tensor, dict], key: str = "obs") -> torch.Tensor:
    """Extract tensor from dict or pass through."""
    if isinstance(x, dict):
        if key in x:
            return x[key]
        # return first tensor value
        for v in x.values():
            if isinstance(v, torch.Tensor):
                return v
        raise ValueError(f"Cannot extract tensor from dict: {list(x.keys())}")
    return x


def _get_num_envs(x: Union[torch.Tensor, dict], key: str = "obs") -> int:
    """Get number of envs from tensor or dict."""
    t = _extract_tensor(x, key)
    return t.shape[0]


class WMDataCollector:
    """
    Collects world model transitions from env steps.

    Per-env prev_valid mask (NOT batch-level clear):
    - prev_valid[i] = False: env i initial/just-reset, prev data unavailable
    - prev_valid[i] = True:  env i has valid prev_obs/prev_action/wm_prev

    No cross-step caching in collector — all data arrives via on_step() params.
    wm_prev comes from WMInterceptorEnv (passed via infos["wm_prev"]).

    Reset handling: done=True → set prev_valid=False for that env only.
    """

    def __init__(self, num_envs: int, wm_buffer: "WMBuffer", device: str = "cuda:0"):
        self.buffer = wm_buffer
        self.device = device
        # Per-env valid mask: (num_envs,) — all False at init
        self._prev_valid = torch.zeros(num_envs, dtype=torch.bool, device=device)

    def on_step(
        self,
        prev_obs: Union[torch.Tensor, dict],
        prev_action: torch.Tensor,
        obs_next: Union[torch.Tensor, dict],
        reward: Union[torch.Tensor, float],
        terminated: Tensor,
        timeout: Tensor,
        infos: dict,
    ) -> None:
        """
        Called after each env.step().

        Args:
            prev_obs:     obs at start of this step (= obs_next from previous step)
            prev_action:   action executed this step
            obs_next:      obs returned by env.step()
            reward/terminated/timeout: env.step() return values
            infos:         full infos dict from env.step() — must contain "wm" and "wm_prev"
        """
        wm_next = infos.get("wm", {})
        wm_prev = infos.get("wm_prev", None)

        # Extract tensors from dict-form obs
        prev_obs_t = _extract_tensor(prev_obs, "obs")
        obs_next_t = _extract_tensor(obs_next, "obs")

        num_envs = _get_num_envs(prev_obs, "obs")
        device = self.device

        if not wm_next or wm_prev is None:
            # WM data not available yet — just update mask
            done = (terminated | timeout).bool()
            done_mask = done.bool()
            not_done_mask = ~done_mask
            if not_done_mask.any():
                self._prev_valid[not_done_mask] = True
            if done_mask.any():
                self._prev_valid[done_mask] = False
            return

        # Compute done (combine terminated + timeout)
        done = (terminated.bool() | timeout.bool())   # (num_envs,) bool
        done_t = done.unsqueeze(-1).float()           # (num_envs, 1) float32

        # current step's transition — add only for prev_valid=True envs
        add_mask = self._prev_valid.clone()  # (num_envs,)
        if not add_mask.any():
            # No env has valid prev — skip buffer add, just update mask
            done_mask = done.bool()
            not_done_mask = ~done_mask
            if not_done_mask.any():
                self._prev_valid[not_done_mask] = True
            if done_mask.any():
                self._prev_valid[done_mask] = False
            return

        failure_t = wm_next.get(
            "failure",
            torch.zeros(num_envs, 1, device=device),
        )

        self.buffer.add(
            obs_t=prev_obs_t,
            action_t=prev_action,
            reward_t=reward.unsqueeze(-1) if reward.ndim == 1 else reward,
            obs_tp1=obs_next_t,
            done_t=done_t,
            failure_t=failure_t,
            object_pos_t=wm_prev.get(
                "object_pos",
                torch.zeros(num_envs, 3, device=device),
            ),
            object_pos_tp1=wm_next.get(
                "object_pos",
                torch.zeros(num_envs, 3, device=device),
            ),
            object_linvel_t=wm_prev.get(
                "object_linvel",
                torch.zeros(num_envs, 3, device=device),
            ),
            object_linvel_tp1=wm_next.get(
                "object_linvel",
                torch.zeros(num_envs, 3, device=device),
            ),
            object_angvel_t=wm_prev.get(
                "object_angvel",
                torch.zeros(num_envs, 3, device=device),
            ),
            object_angvel_tp1=wm_next.get(
                "object_angvel",
                torch.zeros(num_envs, 3, device=device),
            ),
            d_hat_t=wm_prev.get(
                "d_hat",
                torch.zeros(num_envs, 1, device=device),
            ),
            d_hat_tp1=wm_next.get(
                "d_hat",
                torch.zeros(num_envs, 1, device=device),
            ),
            add_mask=add_mask,
        )

        # Update per-env prev_valid mask
        done_mask = done.bool()
        not_done_mask = ~done_mask
        if done_mask.any():
            self._prev_valid[done_mask] = False
        if not_done_mask.any():
            self._prev_valid[not_done_mask] = True

    def on_reset(self, env_ids: Tensor) -> None:
        """Reset specific envs — clear their prev_valid flag."""
        self._prev_valid[env_ids] = False


class WMInterceptorEnv:
    """
    Wraps an IsaacLab env. On each step, stores wm_prev before calling step(),
    then passes wm_prev to collector via infos["wm_prev"].

    Caller is responsible for calling wm_trainer.step() externally.
    """

    def __init__(self, env, collector: WMDataCollector):
        self.env = env
        self.collector = collector
        self._prev_obs = None   # obs_next from previous step
        self._prev_wm: dict | None = None  # infos["wm"] from previous step

    def step(self, action):
        # Snapshot wm_prev before the step
        wm_prev = self._prev_wm
        obs_next, reward, terminated, timeout, infos = self.env.step(action)
        wm_next = infos.get("wm", {})

        # Inject wm_prev for collector
        infos["wm_prev"] = wm_prev

        # Collect transition only if we have a previous obs (skip very first step)
        if self._prev_obs is not None:
            self.collector.on_step(
                prev_obs=self._prev_obs,   # from last step
                prev_action=action,         # current action
                obs_next=obs_next,
                reward=reward,
                terminated=terminated,
                timeout=timeout,
                infos=infos,
            )
        else:
            # First step: no prev data, just update valid mask for non-reset envs
            num_envs = obs_next.shape[0] if isinstance(obs_next, torch.Tensor) else len(obs_next)
            self.collector._prev_valid[:] = True

        # Update cached state for next step
        self._prev_obs = obs_next
        self._prev_wm = wm_next

        return obs_next, reward, terminated, timeout, infos

    def reset(self, env_ids=None):
        if env_ids is not None:
            obs_reset, infos = self.env.reset(env_ids)
        else:
            obs_reset, infos = self.env.reset()
        wm_reset = infos.get("wm", {})
        self._prev_obs = obs_reset
        self._prev_wm = wm_reset
        # Mark reset envs as having no prev valid
        if env_ids is not None:
            self.collector.on_reset(env_ids)
        else:
            # Full reset — all envs reset
            self.collector._prev_valid[:] = False
        return obs_reset, infos

    @property
    def num_envs(self):
        return self.env.num_envs

    @property
    def device(self):
        return self.env.device

    @property
    def unwrapped(self):
        """Expose unwrapped so RlGamesVecEnvWrapper can check env type."""
        return self.env.unwrapped

    def close(self):
        return self.env.close()

    def __getattr__(self, name):
        """Delegate all other attributes/methods to self.env for compatibility."""
        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(self.env, name)