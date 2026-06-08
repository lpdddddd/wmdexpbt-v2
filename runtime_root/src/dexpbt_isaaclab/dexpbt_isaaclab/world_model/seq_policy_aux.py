"""Phase A4 P0 — PPO-facing distillation for sequence world models.

This module implements the bridge between the sequence world model
(``RSSMLiteWorldModel`` / ``TransformerLiteWorldModel``, the *teacher*) and
the PPO actor-critic (the *student*).

Why this exists:

The legacy seq WM trainer optimised a separate optimizer over the seq model
only — see [seq_wm_trainer.py][1]. PPO never received any gradient from the
seq WM's loss. ``rssm_belief_value`` / ``trans_belief_value`` therefore had
zero effect on the actor's shared encoder under the original wiring.

This module fixes that without changing the actor obs dim, reward, env, or
introducing imagined rollouts:

1. ``SeqAuxHeads`` registers tiny student heads on the PPO model (exactly
   like the legacy ``aux_value_head``):
     - ``seq_aux_value_head : Linear(256, 128) -> Tanh -> Linear(128, 1)``
     - optional ``seq_aux_success_head`` with the same shape (BCE).
   Their parameters are added to ``agent.optimizer.param_groups`` so PPO
   updates them.

2. ``register_actor_mlp_capture`` installs a forward hook on
   ``a2c_network.actor_mlp`` that captures the 256-d ``actor_mlp_out`` of
   the *current* PPO minibatch into a ``CapturedFeature`` object. The hook
   fires inside the same ``self.model(batch_dict)`` call that produces
   ``mu``/``sigma``/``values`` in ``A2CAgent.calc_gradients``, so we get
   the right features for free without an extra forward pass.

3. ``SeqDistillCache`` stores teacher targets ``teacher_value_flat[T*N]``
   and (optionally) ``teacher_success_logit_flat[T*N]`` plus a
   ``generation`` integer. ``SeqWMTrainer.compute_teacher_cache`` builds it
   once after each rollout from the full ``(T, N)`` buffer.

4. ``compute_seq_distill_aux_loss`` is invoked from the patched
   ``model.get_aux_loss``. It:
     a. Reads the rl_games ``PPODataset.last_range = (start, end)`` to
        recover this minibatch's sample range. Fail-fast if missing.
     b. Slices ``teacher_value_flat[start:end]`` and the captured
        ``actor_mlp_out``. Fail-fast on shape mismatch.
     c. Computes ``L_value = MSE(seq_aux_value_head(f_t), V_seq_target)``
        (and optionally ``L_success`` BCE) — gradient flows
        ``aux head -> actor_mlp_out -> shared encoder -> PPO model``.
     d. Returns the loss dict so rl_games adds it to the PPO total loss
        before ``backward()``.

[1]: ../world_model/seq_wm_trainer.py
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


COUPLING_NONE = "none"
COUPLING_DISTILL_VALUE = "distill_value"
COUPLING_DISTILL_VALUE_SUCCESS = "distill_value_success"

ALL_COUPLING_MODES = (
    COUPLING_NONE,
    COUPLING_DISTILL_VALUE,
    COUPLING_DISTILL_VALUE_SUCCESS,
)


@dataclass
class SeqPolicyCouplingConfig:
    """Knobs controlling PPO-facing seq WM distillation."""

    mode: str = COUPLING_NONE
    weight: float = 0.05
    value_weight: float = 1.0
    success_weight: float = 0.5
    warmup_frames: int = 100_000_000

    def __post_init__(self) -> None:
        if self.mode not in ALL_COUPLING_MODES:
            raise ValueError(
                f"Unknown seq policy coupling mode {self.mode!r}; "
                f"expected one of {ALL_COUPLING_MODES}"
            )

    @property
    def has_value(self) -> bool:
        return self.mode in (COUPLING_DISTILL_VALUE, COUPLING_DISTILL_VALUE_SUCCESS)

    @property
    def has_success(self) -> bool:
        return self.mode == COUPLING_DISTILL_VALUE_SUCCESS

    @property
    def enabled(self) -> bool:
        return self.mode != COUPLING_NONE


# ---------------------------------------------------------------------------
# Student heads on the PPO model
# ---------------------------------------------------------------------------


def build_seq_aux_value_head(in_dim: int = 256, hidden: int = 128) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, hidden),
        nn.Tanh(),
        nn.Linear(hidden, 1),
    )


def build_seq_aux_success_head(in_dim: int = 256, hidden: int = 128) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, hidden),
        nn.Tanh(),
        nn.Linear(hidden, 1),
    )


def attach_seq_aux_heads(
    *,
    rl_games_model: nn.Module,
    optimizer: torch.optim.Optimizer,
    cfg: SeqPolicyCouplingConfig,
    feature_dim: int = 256,
    device: Optional[torch.device | str] = None,
) -> dict[str, nn.Module]:
    """Register student heads on the PPO model and add them to the optimizer.

    Mirrors the existing ``aux_value_head`` precedent. ``rl_games_model``
    is the rl_games ``Network`` (e.g. ``ModelA2CContinuousLogStd.Network``);
    we attach as ``seq_aux_value_head`` / ``seq_aux_success_head`` so the
    rl_games checkpoint flow saves/loads them automatically.
    """
    if not cfg.enabled:
        return {}

    inner = rl_games_model
    if hasattr(inner, "model"):
        inner = inner.model  # unwrap A2CAgent.model -> Network

    out: dict[str, nn.Module] = {}
    base_lr = optimizer.param_groups[0]["lr"] if optimizer.param_groups else 1e-3

    if cfg.has_value:
        head = build_seq_aux_value_head(in_dim=feature_dim)
        if device is not None:
            head = head.to(device)
        inner.seq_aux_value_head = head
        optimizer.add_param_group({
            "params": list(head.parameters()),
            "lr": base_lr,
            "name": "seq_aux_value_head",
        })
        out["seq_aux_value_head"] = head

    if cfg.has_success:
        head = build_seq_aux_success_head(in_dim=feature_dim)
        if device is not None:
            head = head.to(device)
        inner.seq_aux_success_head = head
        optimizer.add_param_group({
            "params": list(head.parameters()),
            "lr": base_lr,
            "name": "seq_aux_success_head",
        })
        out["seq_aux_success_head"] = head

    return out


# ---------------------------------------------------------------------------
# actor_mlp_out forward-hook capture
# ---------------------------------------------------------------------------


@dataclass
class CapturedFeature:
    """Mutable container for ``actor_mlp_out`` of the current minibatch."""

    tensor: Optional[torch.Tensor] = None
    fired: bool = False

    def reset(self) -> None:
        self.tensor = None
        self.fired = False


def register_actor_mlp_capture(
    rl_games_model: nn.Module,
    capture: CapturedFeature,
) -> Any:
    """Install a forward hook on ``a2c_network.actor_mlp``.

    Captures the post-MLP feature ``actor_mlp_out`` of every PPO model
    forward pass. Returns the hook handle so callers can ``.remove()``
    on shutdown.
    """
    inner = rl_games_model
    if hasattr(inner, "model"):
        inner = inner.model
    a2c = getattr(inner, "a2c_network", None)
    if a2c is None or not hasattr(a2c, "actor_mlp"):
        raise RuntimeError(
            "rl_games model does not expose a2c_network.actor_mlp; "
            "cannot install seq distill feature hook."
        )

    def hook_fn(_module: nn.Module, _inp: Any, out: torch.Tensor) -> None:
        # Cast away autocast bf16 so loss math in fp32 is well-defined.
        capture.tensor = out.to(torch.float32)
        capture.fired = True

    return a2c.actor_mlp.register_forward_hook(hook_fn)


# ---------------------------------------------------------------------------
# Teacher cache
# ---------------------------------------------------------------------------


@dataclass
class SeqDistillCache:
    """Frozen teacher targets emitted by SeqWMTrainer once per rollout.

    Layout matches rl_games ``swap_and_flatten01``: tensors of shape
    ``(T*N,)`` where ``index = n * T + t`` (i.e. T-axis is innermost).
    """

    teacher_value_flat: Optional[torch.Tensor] = None
    teacher_success_logit_flat: Optional[torch.Tensor] = None
    generation: int = -1
    T: int = 0
    N: int = 0
    meta: dict = field(default_factory=dict)

    def is_valid_for(self, generation: int) -> bool:
        return self.generation == generation and self.generation >= 0

    @property
    def total(self) -> int:
        return int(self.T * self.N)


def build_teacher_cache_from_predictions(
    *,
    teacher_value: Optional[torch.Tensor],
    teacher_success_logit: Optional[torch.Tensor],
    T: int,
    N: int,
    generation: int,
) -> SeqDistillCache:
    """Convert ``(T, N)`` teacher predictions into the rl_games flat layout.

    rl_games' ``swap_and_flatten01`` does ``transpose(0, 1).reshape(T*N, ...)``.
    We mimic exactly the same op so ``flat[n * T + t] == teacher[t, n]``.
    """
    cache = SeqDistillCache(generation=int(generation), T=int(T), N=int(N))
    if teacher_value is not None:
        if teacher_value.shape != (T, N):
            raise ValueError(
                f"teacher_value shape {teacher_value.shape} != expected ({T}, {N})"
            )
        cache.teacher_value_flat = (
            teacher_value.detach().transpose(0, 1).reshape(T * N).contiguous()
        )
    if teacher_success_logit is not None:
        if teacher_success_logit.shape != (T, N):
            raise ValueError(
                f"teacher_success_logit shape {teacher_success_logit.shape} != "
                f"expected ({T}, {N})"
            )
        cache.teacher_success_logit_flat = (
            teacher_success_logit.detach()
            .transpose(0, 1)
            .reshape(T * N)
            .contiguous()
        )
    return cache


# ---------------------------------------------------------------------------
# Bridge state + compute
# ---------------------------------------------------------------------------


@dataclass
class SeqDistillBridge:
    """Mutable container shared between SeqWMTrainer and the get_aux_loss hook.

    All fields are populated lazily; the get_aux_loss patch reads them and
    fails fast if anything required is missing.
    """

    cfg: SeqPolicyCouplingConfig
    cache: SeqDistillCache = field(default_factory=SeqDistillCache)
    rollout_generation: int = -1
    feature_capture: CapturedFeature = field(default_factory=CapturedFeature)
    last_active: bool = False
    last_warmup_active: bool = False
    last_loss_value: float = 0.0
    last_loss_success: float = 0.0
    last_effective_weight: float = 0.0
    last_teacher_value_mean: float = 0.0
    last_teacher_value_std: float = 0.0
    # PPO student heads (refs only; owned by the rl_games model).
    seq_aux_value_head: Optional[nn.Module] = None
    seq_aux_success_head: Optional[nn.Module] = None
    # Per-step bookkeeping populated by the hook.
    last_indices: Optional[tuple[int, int]] = None
    # Set by the live agent so the bridge can read agent.dataset.last_range
    # (rl_games' PPODataset stores the contiguous slice of the most recent
    # __getitem__).
    agent_ref: Optional[Any] = None
    # Frame counter snapshot at the moment of the last get_aux_loss call.
    last_frames: int = 0


def _read_minibatch_range(bridge: SeqDistillBridge) -> tuple[int, int]:
    """Read ``(start, end)`` from rl_games' PPODataset; fail fast if missing.

    The rl_games ``PPODataset.__getitem__`` (RNN or non-RNN) sets
    ``self.last_range = (start, end)`` before returning. We pull it directly
    from the agent so we know which slice of the flat ``(T*N)`` teacher
    cache this minibatch corresponds to.

    Per the user's hard constraint (no silent fallback), any failure to
    obtain a valid range raises ``RuntimeError`` instead of unflattening.
    """
    agent = bridge.agent_ref
    if agent is None:
        raise RuntimeError(
            "[SeqDistill] bridge.agent_ref is None; cannot read minibatch indices "
            "(refusing silent fallback)."
        )
    dataset = getattr(agent, "dataset", None)
    if dataset is None:
        raise RuntimeError(
            "[SeqDistill] agent.dataset is None; cannot read minibatch indices."
        )
    last_range = getattr(dataset, "last_range", None)
    if (
        last_range is None
        or not isinstance(last_range, tuple)
        or len(last_range) != 2
    ):
        raise RuntimeError(
            "[SeqDistill] agent.dataset.last_range missing or malformed; "
            "cannot align teacher targets to PPO minibatch."
        )
    start, end = int(last_range[0]), int(last_range[1])
    if end <= start:
        raise RuntimeError(
            f"[SeqDistill] invalid minibatch range start={start} end={end}"
        )
    return start, end


def compute_seq_distill_aux_loss(
    bridge: SeqDistillBridge,
) -> dict[str, torch.Tensor]:
    """Build the seq distill loss dict for the current PPO minibatch.

    Returns ``{}`` when:
      - coupling is disabled,
      - we are inside the warmup window (frames < warmup_frames),
      - no student heads were registered yet,
      - the teacher cache hasn't been built for this rollout.

    Raises ``RuntimeError`` when:
      - the minibatch indices are unavailable,
      - the captured ``actor_mlp_out`` shape disagrees with the slice,
      - the teacher cache generation does not match the live rollout.
    """
    cfg = bridge.cfg
    bridge.last_active = False
    bridge.last_warmup_active = False
    bridge.last_loss_value = 0.0
    bridge.last_loss_success = 0.0
    bridge.last_effective_weight = 0.0
    bridge.last_teacher_value_mean = 0.0
    bridge.last_teacher_value_std = 0.0

    if not cfg.enabled:
        return {}

    if bridge.last_frames < int(cfg.warmup_frames):
        bridge.last_warmup_active = True
        return {}

    if bridge.cache.teacher_value_flat is None and cfg.has_value:
        # Teacher cache not yet built for this rollout: only happens
        # before the first patched_train_epoch call. Warmup gate above
        # is the dominant case. Returning {} is safe here.
        return {}

    if bridge.cache.generation != bridge.rollout_generation:
        raise RuntimeError(
            f"[SeqDistill] teacher cache generation {bridge.cache.generation} "
            f"does not match rollout generation {bridge.rollout_generation}; "
            "refusing silent fallback."
        )

    # Capture must have fired during the same forward pass that produced
    # the PPO logits used in calc_gradients.
    feat = bridge.feature_capture.tensor
    if feat is None or not bridge.feature_capture.fired:
        raise RuntimeError(
            "[SeqDistill] actor_mlp forward hook did not capture features; "
            "cannot compute distill loss."
        )

    start, end = _read_minibatch_range(bridge)
    bridge.last_indices = (start, end)
    expected_b = end - start
    if feat.shape[0] != expected_b:
        raise RuntimeError(
            f"[SeqDistill] captured feature batch {feat.shape[0]} != minibatch "
            f"slice size {expected_b}; cannot align teacher targets."
        )

    losses: dict[str, torch.Tensor] = {}
    weight_total = float(cfg.weight)

    if cfg.has_value:
        target = bridge.cache.teacher_value_flat
        if target is None:
            raise RuntimeError(
                "[SeqDistill] teacher_value_flat missing despite has_value; "
                "cache built incorrectly."
            )
        if end > target.shape[0]:
            raise RuntimeError(
                f"[SeqDistill] minibatch end {end} exceeds teacher cache "
                f"size {target.shape[0]}."
            )
        # detach() is belt-and-suspenders: cache was already built under
        # no_grad, but if the seq trainer ever changes layout we still
        # do not propagate gradient into the teacher.
        target_slice = target[start:end].detach().to(feat.device, feat.dtype)
        head = bridge.seq_aux_value_head
        if head is None:
            raise RuntimeError(
                "[SeqDistill] coupling enabled but seq_aux_value_head missing."
            )
        v_aux = head(feat).squeeze(-1)
        if v_aux.shape != target_slice.shape:
            raise RuntimeError(
                f"[SeqDistill] V_aux shape {tuple(v_aux.shape)} != target "
                f"{tuple(target_slice.shape)}"
            )
        l_value = ((v_aux - target_slice) ** 2).mean()
        scaled_value = weight_total * float(cfg.value_weight) * l_value
        losses["seq_distill_value"] = scaled_value
        bridge.last_loss_value = float(l_value.detach().item())
        bridge.last_teacher_value_mean = float(target_slice.mean().item())
        bridge.last_teacher_value_std = float(target_slice.std(unbiased=False).item())

    if cfg.has_success:
        s_target = bridge.cache.teacher_success_logit_flat
        if s_target is None:
            raise RuntimeError(
                "[SeqDistill] coupling=distill_value_success but cache lacks "
                "teacher_success_logit_flat (teacher has no success head?)."
            )
        s_slice = s_target[start:end].detach().to(feat.device, feat.dtype)
        # Probability targets via sigmoid of the teacher logit.
        s_prob = torch.sigmoid(s_slice)
        head = bridge.seq_aux_success_head
        if head is None:
            raise RuntimeError(
                "[SeqDistill] coupling=distill_value_success but "
                "seq_aux_success_head missing."
            )
        s_logits = head(feat).squeeze(-1)
        l_success = torch.nn.functional.binary_cross_entropy_with_logits(
            s_logits, s_prob,
        )
        scaled_success = weight_total * float(cfg.success_weight) * l_success
        losses["seq_distill_success"] = scaled_success
        bridge.last_loss_success = float(l_success.detach().item())

    bridge.last_active = True
    bridge.last_effective_weight = weight_total
    return losses


__all__ = [
    "ALL_COUPLING_MODES",
    "COUPLING_DISTILL_VALUE",
    "COUPLING_DISTILL_VALUE_SUCCESS",
    "COUPLING_NONE",
    "CapturedFeature",
    "SeqDistillBridge",
    "SeqDistillCache",
    "SeqPolicyCouplingConfig",
    "attach_seq_aux_heads",
    "build_seq_aux_success_head",
    "build_seq_aux_value_head",
    "build_teacher_cache_from_predictions",
    "compute_seq_distill_aux_loss",
    "register_actor_mlp_capture",
]
