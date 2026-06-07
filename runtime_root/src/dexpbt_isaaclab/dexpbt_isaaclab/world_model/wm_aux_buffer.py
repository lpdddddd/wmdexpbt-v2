"""
Aux tensor field helpers for the PPO ExperienceBuffer.

The PPO ``ExperienceBuffer`` already exposes an optional ``aux_tensor_dict``
(see ``rl_games.common.experience.ExperienceBuffer.__init__``); rl_games's
A2C base class instantiates the buffer without it (a2c_common.py:497). To
add per-step aux scalars (d_hat, failure, lifted, success_count) without
forking rl_games, this module:

* ``register_seq_aux_fields(buffer, fields)``
    Allocates ``(T, N, *shape)`` torch tensors directly under the buffer's
    ``tensor_dict``, after which ``buffer.update_data(name, t, val)`` works
    transparently for those names.

* ``extract_step_aux(infos, env_attrs)``
    Pulls per-env aux scalars from the ``infos["wm"]`` dict produced by
    ``DexPBTKukaAllegroEnv.step()`` (and optionally from extra env
    attributes for fields that aren't yet exposed via infos, e.g.
    ``lifted_object``). Returns a name -> (N,) or (N, 1) tensor dict.

* ``insert_step_aux(buffer, t, aux_dict)``
    Convenience wrapper around ``buffer.update_data``.

* ``read_aux_full(buffer, name)``
    Returns the full (T, N) or (T, N, K) tensor stored under ``name``,
    or None if the field was never registered.

The actual ``play_steps_rnn`` patching that calls these helpers lives in
``train.py`` and is wired up only when the active strategy needs the
sequence world model (i.e. one of the ``rssm_*`` / ``trans_*`` family).
For non-sequence strategies the buffer layout is unchanged.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional

import numpy as np
import torch

# Default aux field schema for sequence world model strategies.
# Map name -> per-step shape (excluding the leading (T, N) shape).
DEFAULT_SEQ_AUX_FIELDS: dict[str, tuple[int, ...]] = {
    "aux_d_hat": (1,),
    "aux_failure": (1,),
    "aux_lifted": (1,),
    "aux_success_cnt": (1,),
}


def register_seq_aux_fields(
    experience_buffer: Any,
    *,
    fields: Optional[Mapping[str, tuple[int, ...]]] = None,
    dtype: torch.dtype = torch.float32,
) -> list[str]:
    """Allocate aux tensors of shape (T, N, *shape) inside the PPO buffer.

    Safe to call multiple times: existing keys are skipped.

    Returns the list of field names that were newly registered.
    """
    if fields is None:
        fields = DEFAULT_SEQ_AUX_FIELDS

    if not hasattr(experience_buffer, "tensor_dict"):
        raise AttributeError(
            "experience_buffer has no .tensor_dict; expected rl_games "
            "ExperienceBuffer-like object."
        )
    if not hasattr(experience_buffer, "obs_base_shape"):
        raise AttributeError(
            "experience_buffer has no .obs_base_shape; cannot infer (T, N) "
            "leading shape."
        )

    obs_base = tuple(experience_buffer.obs_base_shape)  # (T, N)
    device = getattr(experience_buffer, "device", "cpu")
    new_keys: list[str] = []
    for name, shape in fields.items():
        if name in experience_buffer.tensor_dict:
            continue
        full_shape = obs_base + tuple(shape)
        experience_buffer.tensor_dict[name] = torch.zeros(
            full_shape, dtype=dtype, device=device
        )
        new_keys.append(name)
    return new_keys


def insert_step_aux(
    experience_buffer: Any,
    t: int,
    aux_dict: Mapping[str, torch.Tensor],
) -> None:
    """Call buffer.update_data(name, t, val) for every key present in aux_dict
    AND already registered on the buffer. Unknown / unregistered keys are
    silently dropped (so we don't fail hard if env doesn't expose a field).
    """
    if not aux_dict:
        return
    td = experience_buffer.tensor_dict
    for name, val in aux_dict.items():
        if name not in td:
            continue
        # rl_games' update_data handles dicts and 1-D scalar broadcast itself.
        experience_buffer.update_data(name, t, val)


def read_aux_full(
    experience_buffer: Any,
    name: str,
) -> Optional[torch.Tensor]:
    """Return the full (T, N, ...) tensor stored under `name`, or None."""
    td = getattr(experience_buffer, "tensor_dict", {})
    return td.get(name, None)


# ----- env-side extraction -----

def _to_tensor_2d(x: Any, *, device: torch.device, dtype: torch.dtype = torch.float32
                  ) -> Optional[torch.Tensor]:
    """Convert input to (N, 1) torch tensor; returns None if x is None."""
    if x is None:
        return None
    if isinstance(x, torch.Tensor):
        t = x.to(device=device, dtype=dtype)
    elif isinstance(x, np.ndarray):
        t = torch.from_numpy(x).to(device=device, dtype=dtype)
    else:
        # python scalar / list — let torch figure it out
        t = torch.as_tensor(x, device=device, dtype=dtype)
    if t.ndim == 1:
        t = t.unsqueeze(-1)
    return t


def extract_step_aux(
    infos: Mapping[str, Any],
    *,
    device: torch.device | str,
    env_attrs: Optional[Mapping[str, Any]] = None,
    fields: Optional[Mapping[str, tuple[int, ...]]] = None,
) -> dict[str, torch.Tensor]:
    """Build the per-step aux dict from env-returned infos and optional
    extra env attributes.

    Sources (in priority order):
      aux_d_hat       <- infos["wm"]["d_hat"]
      aux_failure     <- infos["wm"]["failure"]
      aux_lifted      <- env_attrs["lifted_object"] if provided, else
                         infos["wm"].get("lifted") if exposed
      aux_success_cnt <- env_attrs["successes"] if provided, else
                         infos["wm"].get("success_count") if exposed

    Returns a dict whose keys are a SUBSET of `fields` (default
    DEFAULT_SEQ_AUX_FIELDS); missing inputs are simply omitted, and the
    receiver should be tolerant.
    """
    if fields is None:
        fields = DEFAULT_SEQ_AUX_FIELDS
    device = torch.device(device)
    out: dict[str, torch.Tensor] = {}
    wm = (infos or {}).get("wm") if isinstance(infos, Mapping) else None
    env_attrs = env_attrs or {}

    if "aux_d_hat" in fields and isinstance(wm, Mapping):
        v = _to_tensor_2d(wm.get("d_hat"), device=device)
        if v is not None:
            out["aux_d_hat"] = v
    if "aux_failure" in fields and isinstance(wm, Mapping):
        v = _to_tensor_2d(wm.get("failure"), device=device)
        if v is not None:
            out["aux_failure"] = v
    if "aux_lifted" in fields:
        src = env_attrs.get("lifted_object")
        if src is None and isinstance(wm, Mapping):
            src = wm.get("lifted")
        v = _to_tensor_2d(src, device=device)
        if v is not None:
            out["aux_lifted"] = v
    if "aux_success_cnt" in fields:
        src = env_attrs.get("successes")
        if src is None and isinstance(wm, Mapping):
            src = wm.get("success_count")
        v = _to_tensor_2d(src, device=device)
        if v is not None:
            out["aux_success_cnt"] = v
    return out


__all__ = [
    "DEFAULT_SEQ_AUX_FIELDS",
    "register_seq_aux_fields",
    "insert_step_aux",
    "read_aux_full",
    "extract_step_aux",
]
