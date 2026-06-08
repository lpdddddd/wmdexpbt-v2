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
COUPLING_CRITIC_DISTILL_VALUE_NORM = "critic_distill_value_norm"
COUPLING_ALIAS_SHARED_AUX_CRITIC = "shared_aux_critic_distill_value_norm"

ALL_COUPLING_MODES = (
    COUPLING_NONE,
    COUPLING_DISTILL_VALUE,
    COUPLING_DISTILL_VALUE_SUCCESS,
    COUPLING_CRITIC_DISTILL_VALUE_NORM,
    COUPLING_ALIAS_SHARED_AUX_CRITIC,
)

FEATURE_SOURCE_ACTOR = "actor"
FEATURE_SOURCE_SHARED_TO_AUX_CRITIC = "shared_to_aux_critic"
ALL_FEATURE_SOURCES = (FEATURE_SOURCE_ACTOR, FEATURE_SOURCE_SHARED_TO_AUX_CRITIC)

VALUE_NORM_NONE = "none"
VALUE_NORM_ROLLOUT_ZSCORE = "rollout_zscore"
ALL_VALUE_NORMS = (VALUE_NORM_NONE, VALUE_NORM_ROLLOUT_ZSCORE)

GATE_NONE = "none"
GATE_TEACHER_CORR = "teacher_corr"
ALL_GATES = (GATE_NONE, GATE_TEACHER_CORR)

# TB enum encoding for `seq_policy_aux/feature_source`
FEATURE_SOURCE_ID = {
    FEATURE_SOURCE_ACTOR: 0,
    FEATURE_SOURCE_SHARED_TO_AUX_CRITIC: 1,
}
COUPLING_MODE_ID = {
    COUPLING_NONE: 0,
    COUPLING_DISTILL_VALUE: 1,
    COUPLING_DISTILL_VALUE_SUCCESS: 2,
    COUPLING_CRITIC_DISTILL_VALUE_NORM: 3,
}


@dataclass
class SeqPolicyCouplingConfig:
    """Knobs controlling PPO-facing seq WM distillation.

    P0 modes (legacy, kept verbatim as negative ablations):
      ``none`` / ``distill_value`` / ``distill_value_success`` use
      ``feature_source=actor``, ``teacher_value_norm=none``,
      ``gate=none``, ``ramp_frames=0``. The actor-direct distill path
      ran the 200M sanity AB (commit 6609e37); it regressed nsucc to
      zero. Preserved for ablation only; never auto-upgraded to v2.

    v2 mode (Phase A4 P1):
      ``critic_distill_value_norm`` (alias
      ``shared_aux_critic_distill_value_norm``) requires
      ``feature_source=shared_to_aux_critic``,
      ``teacher_value_norm=rollout_zscore``, and (default)
      ``gate=teacher_corr``. The shared 256-d ``actor_mlp_out`` is
      consumed exclusively by a private ``seq_aux_critic_mlp``
      sub-tower (NOT the rl_games ``critic_mlp`` which is empty
      under ``network.separate=false``) before flowing into
      ``seq_aux_value_head``.
    """

    mode: str = COUPLING_NONE
    weight: float = 0.05
    value_weight: float = 1.0
    success_weight: float = 0.5
    warmup_frames: int = 100_000_000
    # --- v2 fields ---
    feature_source: str = FEATURE_SOURCE_ACTOR
    teacher_value_norm: str = VALUE_NORM_NONE
    teacher_value_norm_min_std: float = 1e-3
    gate: str = GATE_NONE
    teacher_corr_threshold: float = 0.2
    ramp_frames: int = 0
    aux_critic_subtower_hidden: int = 256
    # Set to True by __post_init__ when the user explicitly disabled the
    # gate on the v2 mode (logged as a loud warning at startup).
    gate_user_overridden: bool = False

    def __post_init__(self) -> None:
        if self.mode == COUPLING_ALIAS_SHARED_AUX_CRITIC:
            self.mode = COUPLING_CRITIC_DISTILL_VALUE_NORM
        if self.mode not in ALL_COUPLING_MODES:
            raise ValueError(
                f"Unknown seq policy coupling mode {self.mode!r}; "
                f"expected one of {ALL_COUPLING_MODES}"
            )
        if self.feature_source not in ALL_FEATURE_SOURCES:
            raise ValueError(
                f"Unknown feature_source {self.feature_source!r}; "
                f"expected one of {ALL_FEATURE_SOURCES}"
            )
        if self.teacher_value_norm not in ALL_VALUE_NORMS:
            raise ValueError(
                f"Unknown teacher_value_norm {self.teacher_value_norm!r}; "
                f"expected one of {ALL_VALUE_NORMS}"
            )
        if self.gate not in ALL_GATES:
            raise ValueError(
                f"Unknown gate {self.gate!r}; expected one of {ALL_GATES}"
            )

        is_v2 = self.mode == COUPLING_CRITIC_DISTILL_VALUE_NORM
        if is_v2:
            if self.feature_source != FEATURE_SOURCE_SHARED_TO_AUX_CRITIC:
                raise ValueError(
                    f"mode={COUPLING_CRITIC_DISTILL_VALUE_NORM} requires "
                    f"feature_source={FEATURE_SOURCE_SHARED_TO_AUX_CRITIC}, "
                    f"got {self.feature_source!r}"
                )
            if self.teacher_value_norm != VALUE_NORM_ROLLOUT_ZSCORE:
                raise ValueError(
                    f"mode={COUPLING_CRITIC_DISTILL_VALUE_NORM} requires "
                    f"teacher_value_norm={VALUE_NORM_ROLLOUT_ZSCORE}, "
                    f"got {self.teacher_value_norm!r}"
                )
            if self.gate != GATE_TEACHER_CORR:
                # Per plan section 8: warning, not error.
                self.gate_user_overridden = True
                print(
                    f"[SeqDistill] WARNING: mode={COUPLING_CRITIC_DISTILL_VALUE_NORM} "
                    f"is being run with gate={self.gate!r} (default is "
                    f"{GATE_TEACHER_CORR!r}). The teacher-vs-return gate is "
                    "the primary safeguard against propagating low-quality "
                    "teacher values; disabling it removes that safety net."
                )
        else:
            if self.feature_source == FEATURE_SOURCE_SHARED_TO_AUX_CRITIC:
                raise ValueError(
                    f"feature_source={FEATURE_SOURCE_SHARED_TO_AUX_CRITIC} "
                    f"is reserved for mode={COUPLING_CRITIC_DISTILL_VALUE_NORM}; "
                    f"legacy mode {self.mode!r} must use "
                    f"feature_source={FEATURE_SOURCE_ACTOR}."
                )

    @property
    def has_value(self) -> bool:
        return self.mode in (
            COUPLING_DISTILL_VALUE,
            COUPLING_DISTILL_VALUE_SUCCESS,
            COUPLING_CRITIC_DISTILL_VALUE_NORM,
        )

    @property
    def has_success(self) -> bool:
        return self.mode == COUPLING_DISTILL_VALUE_SUCCESS

    @property
    def enabled(self) -> bool:
        return self.mode != COUPLING_NONE

    @property
    def is_v2(self) -> bool:
        return self.mode == COUPLING_CRITIC_DISTILL_VALUE_NORM

    @property
    def uses_shared_to_aux_critic(self) -> bool:
        return self.feature_source == FEATURE_SOURCE_SHARED_TO_AUX_CRITIC


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


def build_seq_aux_critic_subtower(
    in_dim: int = 256, hidden: int = 256
) -> nn.Sequential:
    """v2 only. Private 'critic' sub-tower stacked above ``actor_mlp_out``.

    NOT the rl_games ``critic_mlp`` (which is an empty ``Sequential`` under
    ``network.separate=false``). The output dim equals ``in_dim`` so the
    existing ``seq_aux_value_head`` (which expects 256-d input) can be
    reused unchanged.
    """
    return nn.Sequential(
        nn.Linear(in_dim, hidden),
        nn.Tanh(),
        nn.Linear(hidden, hidden),
        nn.Tanh(),
        nn.Linear(hidden, in_dim),
    )


def attach_seq_aux_heads(
    *,
    rl_games_model: nn.Module,
    optimizer: torch.optim.Optimizer,
    cfg: SeqPolicyCouplingConfig,
    feature_dim: int = 256,
    device: Optional[torch.device | str] = None,
) -> dict[str, nn.Module]:
    """Register student heads (and the v2 sub-tower) on the PPO model.

    Adds attached modules to the rl_games ``Network`` (so the checkpoint
    flow saves/loads them automatically) and to ``optimizer.param_groups``
    so PPO updates them.

    Legacy modes (``distill_value`` / ``distill_value_success``) attach
    only ``seq_aux_value_head`` / ``seq_aux_success_head`` on top of the
    captured ``actor_mlp_out`` directly (P0 wiring, untouched).

    v2 mode (``critic_distill_value_norm``) additionally attaches a
    private ``seq_aux_critic_mlp`` between ``actor_mlp_out`` and
    ``seq_aux_value_head``. This is the *only* place that consumes the
    capture in v2; legacy ``distill_value`` does NOT touch this tower.
    """
    if not cfg.enabled:
        return {}

    inner = rl_games_model
    if hasattr(inner, "model"):
        inner = inner.model  # unwrap A2CAgent.model -> Network

    out: dict[str, nn.Module] = {}
    base_lr = optimizer.param_groups[0]["lr"] if optimizer.param_groups else 1e-3

    if cfg.is_v2:
        sub = build_seq_aux_critic_subtower(
            in_dim=feature_dim, hidden=int(cfg.aux_critic_subtower_hidden)
        )
        if device is not None:
            sub = sub.to(device)
        inner.seq_aux_critic_mlp = sub
        optimizer.add_param_group({
            "params": list(sub.parameters()),
            "lr": base_lr,
            "name": "seq_aux_critic_mlp",
        })
        out["seq_aux_critic_mlp"] = sub

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
    """Install a forward hook on ``a2c_network.actor_mlp`` (legacy P0 path).

    Captures the post-MLP feature ``actor_mlp_out`` of every PPO model
    forward pass. Returns the hook handle so callers can ``.remove()``
    on shutdown.

    Legacy ``distill_value`` / ``distill_value_success`` use this entry
    point. v2 mode uses ``register_shared_to_aux_critic_capture`` (which
    targets the same module, but the captured tensor is consumed only by
    the v2 sub-tower path).
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


def register_shared_to_aux_critic_capture(
    rl_games_model: nn.Module,
    capture: CapturedFeature,
) -> Any:
    """v2 entry point. Same module target as the actor capture.

    The hook target is still ``a2c_network.actor_mlp`` because under
    ``network.separate=false`` (the dexpbt config) this is the only
    populated feature module: rl_games builds an empty
    ``a2c_network.critic_mlp = nn.Sequential()`` that the forward path
    never calls. The 'critic' semantics in v2 lives in the
    ``seq_aux_critic_mlp`` sub-tower attached separately by
    ``attach_seq_aux_heads`` -- it is NOT this hook.
    """
    return register_actor_mlp_capture(rl_games_model, capture)


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


def apply_rollout_zscore(
    teacher_value: torch.Tensor,
    *,
    min_std: float,
) -> tuple[torch.Tensor, float, float, bool]:
    """v2 helper: per-rollout z-score on the teacher value tensor.

    Returns ``(normalized_or_raw, mean, std, std_too_small)``. When ``std``
    is below ``min_std`` the original tensor is returned unchanged so the
    cache still has a populated ``teacher_value_flat`` to slice (the gate
    will block before any of these targets are consumed).
    """
    mean = float(teacher_value.mean().item())
    std = float(teacher_value.std(unbiased=False).item())
    too_small = std < float(min_std)
    if too_small:
        return teacher_value, mean, std, True
    norm = (teacher_value - mean) / (std + 1e-6)
    return norm, mean, std, False


def compute_teacher_return_corr(
    teacher_value: torch.Tensor,
    returns: torch.Tensor,
) -> tuple[float, bool]:
    """v2 helper: Pearson corr between flattened teacher_value and returns.

    Both tensors are reshaped to 1D before centering. Returns
    ``(corr, valid)`` where ``valid`` is True iff the corr is finite and
    both inputs have non-degenerate variance.
    """
    v_flat = teacher_value.detach().reshape(-1).to(torch.float32)
    r_flat = returns.detach().reshape(-1).to(torch.float32)
    if v_flat.numel() != r_flat.numel():
        raise ValueError(
            f"teacher_value ({v_flat.numel()}) and returns ({r_flat.numel()}) "
            "must have the same number of elements after flattening"
        )
    v_std = float(v_flat.std(unbiased=False).item())
    r_std = float(r_flat.std(unbiased=False).item())
    if not (v_std > 1e-6) or not (r_std > 1e-6):
        return 0.0, False
    v_c = v_flat - v_flat.mean()
    r_c = r_flat - r_flat.mean()
    denom = (v_c.std(unbiased=False) * r_c.std(unbiased=False)).clamp_min(1e-8)
    corr_t = (v_c * r_c).mean() / denom
    if not torch.isfinite(corr_t):
        return 0.0, False
    return float(corr_t.item()), True


def build_teacher_cache_from_predictions(
    *,
    teacher_value: Optional[torch.Tensor],
    teacher_success_logit: Optional[torch.Tensor],
    T: int,
    N: int,
    generation: int,
    teacher_value_norm: str = VALUE_NORM_NONE,
    teacher_value_norm_min_std: float = 1e-3,
    returns_for_corr: Optional[torch.Tensor] = None,
) -> SeqDistillCache:
    """Convert ``(T, N)`` teacher predictions into the rl_games flat layout.

    rl_games' ``swap_and_flatten01`` does ``transpose(0, 1).reshape(T*N, ...)``.
    We mimic exactly the same op so ``flat[n * T + t] == teacher[t, n]``.

    When ``teacher_value_norm == "rollout_zscore"`` the (T, N) value tensor
    is z-scored over the entire rollout before flattening. The raw
    mean/std are always recorded in ``cache.meta`` for diagnostics. When
    the rollout std is below ``teacher_value_norm_min_std`` the cache is
    still built (with raw values) but ``teacher_value_norm_std_too_small``
    is set in meta so the gate path can block.

    When ``returns_for_corr`` is provided, ``teacher_return_corr`` is
    computed against the raw (pre-norm) teacher_value tensor and stored
    in ``cache.meta``.
    """
    cache = SeqDistillCache(generation=int(generation), T=int(T), N=int(N))
    if teacher_value is not None:
        if teacher_value.shape != (T, N):
            raise ValueError(
                f"teacher_value shape {teacher_value.shape} != expected ({T}, {N})"
            )
        # Always log raw stats for diagnostics, before any normalization.
        raw_mean = float(teacher_value.mean().item())
        raw_std = float(teacher_value.std(unbiased=False).item())
        cache.meta["teacher_value_mean"] = raw_mean
        cache.meta["teacher_value_std"] = raw_std

        std_too_small = False
        if teacher_value_norm == VALUE_NORM_ROLLOUT_ZSCORE:
            cache.meta["teacher_value_norm_enabled"] = True
            normed, _, _, std_too_small = apply_rollout_zscore(
                teacher_value, min_std=teacher_value_norm_min_std
            )
            value_to_flat = normed
        else:
            cache.meta["teacher_value_norm_enabled"] = False
            value_to_flat = teacher_value
        cache.meta["teacher_value_norm_std_too_small"] = bool(std_too_small)

        cache.teacher_value_flat = (
            value_to_flat.detach().transpose(0, 1).reshape(T * N).contiguous()
        )

        if returns_for_corr is not None:
            corr, valid = compute_teacher_return_corr(
                teacher_value, returns_for_corr
            )
            cache.meta["teacher_return_corr"] = corr
            cache.meta["teacher_return_corr_valid"] = bool(valid)
        else:
            cache.meta["teacher_return_corr"] = 0.0
            cache.meta["teacher_return_corr_valid"] = False

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
    # v2 bookkeeping.
    last_gate_pass: bool = False
    last_gate_blocked: bool = False
    gate_blocked_count: int = 0
    last_ramp_factor: float = 0.0
    last_base_weight: float = 0.0
    last_teacher_return_corr: float = 0.0
    last_teacher_return_corr_valid: bool = False
    last_actor_feature_captured: bool = False
    last_shared_to_aux_critic_feature_captured: bool = False
    last_aux_critic_subtower_grad_norm: float = 0.0
    # PPO student heads / sub-tower (refs only; owned by the rl_games model).
    seq_aux_value_head: Optional[nn.Module] = None
    seq_aux_success_head: Optional[nn.Module] = None
    seq_aux_critic_mlp: Optional[nn.Module] = None
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


def _reset_per_call_bridge_state(bridge: SeqDistillBridge) -> None:
    bridge.last_active = False
    bridge.last_warmup_active = False
    bridge.last_loss_value = 0.0
    bridge.last_loss_success = 0.0
    bridge.last_effective_weight = 0.0
    bridge.last_teacher_value_mean = 0.0
    bridge.last_teacher_value_std = 0.0
    bridge.last_gate_pass = False
    bridge.last_gate_blocked = False
    bridge.last_ramp_factor = 0.0
    bridge.last_base_weight = float(bridge.cfg.weight)
    bridge.last_actor_feature_captured = False
    bridge.last_shared_to_aux_critic_feature_captured = False


def _compute_ramp_factor(
    cfg: SeqPolicyCouplingConfig, frames: int
) -> float:
    if frames < int(cfg.warmup_frames):
        return 0.0
    if int(cfg.ramp_frames) <= 0:
        return 1.0
    progress = (int(frames) - int(cfg.warmup_frames)) / float(cfg.ramp_frames)
    return float(min(1.0, max(0.0, progress)))


def _evaluate_teacher_corr_gate(
    cfg: SeqPolicyCouplingConfig, cache: SeqDistillCache
) -> tuple[bool, float, bool]:
    """Returns ``(blocked, corr, corr_valid)``."""
    corr = float(cache.meta.get("teacher_return_corr", 0.0))
    valid = bool(cache.meta.get("teacher_return_corr_valid", False))
    if cfg.gate == GATE_NONE:
        return False, corr, valid
    if not valid:
        return True, corr, valid
    return corr < float(cfg.teacher_corr_threshold), corr, valid


def compute_seq_distill_aux_loss(
    bridge: SeqDistillBridge,
) -> dict[str, torch.Tensor]:
    """Build the seq distill loss dict for the current PPO minibatch.

    Returns ``{}`` when:
      - coupling is disabled,
      - we are inside the warmup window (frames < warmup_frames),
      - the post-warmup ramp factor is zero,
      - the teacher_corr gate blocks this rollout (v2 only),
      - the teacher cache hasn't been built for this rollout yet.

    Raises ``RuntimeError`` when:
      - the minibatch indices are unavailable,
      - the captured shared-encoder feature did not fire,
      - the captured feature shape disagrees with the slice,
      - the teacher cache generation does not match the live rollout,
      - v2 mode is requested but ``seq_aux_critic_mlp`` is missing,
      - student head input/output shapes disagree.
    """
    cfg = bridge.cfg
    _reset_per_call_bridge_state(bridge)

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

    # Quality gate (v2 only when gate=teacher_corr; legacy modes leave
    # gate=none and never block here).
    blocked, corr, corr_valid = _evaluate_teacher_corr_gate(cfg, bridge.cache)
    bridge.last_teacher_return_corr = float(corr)
    bridge.last_teacher_return_corr_valid = bool(corr_valid)
    if blocked:
        bridge.last_gate_blocked = True
        bridge.gate_blocked_count += 1
        return {}
    bridge.last_gate_pass = True

    # Ramp gate: post-warmup linear ramp from 0 to 1 over ramp_frames.
    ramp_factor = _compute_ramp_factor(cfg, int(bridge.last_frames))
    bridge.last_ramp_factor = float(ramp_factor)
    effective_weight = float(cfg.weight) * float(ramp_factor)
    if effective_weight <= 0.0:
        return {}

    # Capture must have fired during the same forward pass that produced
    # the PPO logits used in calc_gradients.
    feat = bridge.feature_capture.tensor
    if feat is None or not bridge.feature_capture.fired:
        raise RuntimeError(
            "[SeqDistill] shared-encoder forward hook did not capture "
            "features; cannot compute distill loss."
        )
    if cfg.uses_shared_to_aux_critic:
        bridge.last_shared_to_aux_critic_feature_captured = True
    else:
        bridge.last_actor_feature_captured = True

    start, end = _read_minibatch_range(bridge)
    bridge.last_indices = (start, end)
    expected_b = end - start
    if feat.shape[0] != expected_b:
        raise RuntimeError(
            f"[SeqDistill] captured feature batch {feat.shape[0]} != minibatch "
            f"slice size {expected_b}; cannot align teacher targets."
        )

    # v2 dispatch: route capture through the private sub-tower BEFORE the
    # value head; never silently fall back to actor-direct compute.
    if cfg.is_v2:
        sub = bridge.seq_aux_critic_mlp
        if sub is None:
            raise RuntimeError(
                "[SeqDistill] mode=critic_distill_value_norm requires "
                "seq_aux_critic_mlp to be attached; refusing silent "
                "fallback to actor-direct distill."
            )
        feat_for_value = sub(feat)
    else:
        if cfg.uses_shared_to_aux_critic:
            raise RuntimeError(
                "[SeqDistill] feature_source=shared_to_aux_critic requires "
                "mode=critic_distill_value_norm; legacy mode "
                f"{cfg.mode!r} cannot use the v2 sub-tower path."
            )
        feat_for_value = feat

    losses: dict[str, torch.Tensor] = {}

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
        target_slice = target[start:end].detach().to(
            feat_for_value.device, feat_for_value.dtype
        )
        head = bridge.seq_aux_value_head
        if head is None:
            raise RuntimeError(
                "[SeqDistill] coupling enabled but seq_aux_value_head missing."
            )
        v_aux = head(feat_for_value).squeeze(-1)
        if v_aux.shape != target_slice.shape:
            raise RuntimeError(
                f"[SeqDistill] V_aux shape {tuple(v_aux.shape)} != target "
                f"{tuple(target_slice.shape)}"
            )
        l_value = ((v_aux - target_slice) ** 2).mean()
        scaled_value = effective_weight * float(cfg.value_weight) * l_value
        losses["seq_distill_value"] = scaled_value
        bridge.last_loss_value = float(l_value.detach().item())
        bridge.last_teacher_value_mean = float(target_slice.mean().item())
        bridge.last_teacher_value_std = float(
            target_slice.std(unbiased=False).item()
        )

    if cfg.has_success:
        s_target = bridge.cache.teacher_success_logit_flat
        if s_target is None:
            raise RuntimeError(
                "[SeqDistill] coupling=distill_value_success but cache lacks "
                "teacher_success_logit_flat (teacher has no success head?)."
            )
        s_slice = s_target[start:end].detach().to(
            feat_for_value.device, feat_for_value.dtype
        )
        # Probability targets via sigmoid of the teacher logit.
        s_prob = torch.sigmoid(s_slice)
        head = bridge.seq_aux_success_head
        if head is None:
            raise RuntimeError(
                "[SeqDistill] coupling=distill_value_success but "
                "seq_aux_success_head missing."
            )
        s_logits = head(feat_for_value).squeeze(-1)
        l_success = torch.nn.functional.binary_cross_entropy_with_logits(
            s_logits, s_prob,
        )
        scaled_success = effective_weight * float(cfg.success_weight) * l_success
        losses["seq_distill_success"] = scaled_success
        bridge.last_loss_success = float(l_success.detach().item())

    bridge.last_active = True
    bridge.last_effective_weight = effective_weight
    return losses


__all__ = [
    "ALL_COUPLING_MODES",
    "ALL_FEATURE_SOURCES",
    "ALL_GATES",
    "ALL_VALUE_NORMS",
    "COUPLING_ALIAS_SHARED_AUX_CRITIC",
    "COUPLING_CRITIC_DISTILL_VALUE_NORM",
    "COUPLING_DISTILL_VALUE",
    "COUPLING_DISTILL_VALUE_SUCCESS",
    "COUPLING_MODE_ID",
    "COUPLING_NONE",
    "CapturedFeature",
    "FEATURE_SOURCE_ACTOR",
    "FEATURE_SOURCE_ID",
    "FEATURE_SOURCE_SHARED_TO_AUX_CRITIC",
    "GATE_NONE",
    "GATE_TEACHER_CORR",
    "SeqDistillBridge",
    "SeqDistillCache",
    "SeqPolicyCouplingConfig",
    "VALUE_NORM_NONE",
    "VALUE_NORM_ROLLOUT_ZSCORE",
    "apply_rollout_zscore",
    "attach_seq_aux_heads",
    "build_seq_aux_critic_subtower",
    "build_seq_aux_success_head",
    "build_seq_aux_value_head",
    "build_teacher_cache_from_predictions",
    "compute_seq_distill_aux_loss",
    "compute_teacher_return_corr",
    "register_actor_mlp_capture",
    "register_shared_to_aux_critic_capture",
]
