"""
FloorCritic — task-progress auxiliary critic learning for floor-stage PPO.

This is NOT RSSM value distillation, NOT actor imitation, NOT Dreamer-style
latent policy training. It adds floor-stage auxiliary BCE heads that learn to
recognise, from REAL rollout transitions only:

  1. near-success states
  2. d_hat-improving (recovery) states
  3. short-horizon success probability
  4. drop / failure risk

Two phases (see results/FLOORCRITIC_IMPLEMENTATION_PLAN.md):

  Phase 0 (detached smoke): ``feature_source=actor_mlp_out_detached``.
    The captured 256-d ``actor_mlp_out`` is ``.detach()``-ed before the aux
    tower, so the FloorCritic loss updates ONLY the aux tower + heads, never
    the shared backbone or the actor. ``actor_grad_blocked`` is structurally
    true. Used to validate engineering / targets / logging.

  Phase 1 (effective): ``feature_source=actor_mlp_out_shared``.
    The captured feature is fed WITHOUT detach, so the FloorCritic loss
    regularises the shared actor/value representation (actor_mlp + LSTM).
    The FloorCritic heads still NEVER connect to the ``mu`` (actor logits)
    head, so the actor logit mapping is not directly updated by this loss.
    Risk is bounded by a small weight + grad-ratio auto-scale.

Gradient routing summary:

    actor_mlp_out --(detach? per feature_source)--> fc_aux_tower --> 8 heads --> BCE
        * detached: gradient stops at the tower input clone.
        * shared:   gradient flows into actor_mlp / LSTM, but never into mu.

All targets are built from the PPO ExperienceBuffer ``(T, N)`` aux tensors
(``aux_d_hat`` / ``aux_success_cnt`` / ``aux_success_tolerance`` /
``aux_failure`` / ``aux_lifted``) and flattened with the rl_games
``swap_and_flatten01`` layout ``flat[n * T + t] = x[t, n]`` so that the
``PPODataset.last_range = (start, end)`` slice aligns the targets to the
current PPO minibatch.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

FEATURE_SOURCE_DETACHED = "actor_mlp_out_detached"
FEATURE_SOURCE_SHARED = "actor_mlp_out_shared"
FEATURE_SOURCE_CRITIC = "critic_feature"
_VALID_FEATURE_SOURCES = (
    FEATURE_SOURCE_DETACHED,
    FEATURE_SOURCE_SHARED,
    FEATURE_SOURCE_CRITIC,
)

# Target types.
TARGET_NEAR_SUCCESS = "near_success"
TARGET_SUCCESS = "success"
TARGET_DHAT_IMPROVE = "dhat_improve"
TARGET_DROP_RISK = "drop_risk"


@dataclass
class FloorCriticConfig:
    """All hyperparameters that drive FloorCritic behaviour.

    Defaults are the safe Phase-0 values: ``enabled=False`` (whole stack is
    a no-op), ``feature_source=actor_mlp_out_detached`` (actor_grad_blocked).
    """

    enabled: bool = False
    feature_source: str = FEATURE_SOURCE_DETACHED
    max_grad_ratio: float = 0.1
    auto_scale_by_grad_ratio: bool = True

    start_tol: float = 0.04
    floor_tol: float = 0.02
    hard_floor_tol: float = 0.012

    horizons: tuple[int, ...] = (4, 8)

    weight: float = 0.005
    floor_weight: float = 0.01
    hard_floor_weight: float = 0.02

    near_success_mult: float = 1.5
    dhat_improve_margin: float = 0.005
    drop_dhat_margin: float = 0.01

    actor_detach: bool = True
    update_actor: bool = False

    use_success: bool = True
    use_near_success: bool = True
    use_dhat_improve: bool = True
    use_drop_risk: bool = True

    feature_dim: int = 256
    tower_hidden: int = 256

    def __post_init__(self) -> None:
        self.feature_source = str(self.feature_source)
        if self.feature_source not in _VALID_FEATURE_SOURCES:
            raise ValueError(
                f"Unknown wm_floorcritic_feature_source={self.feature_source!r}; "
                f"expected one of {_VALID_FEATURE_SOURCES}."
            )
        self.horizons = tuple(int(h) for h in self.horizons if int(h) > 0)
        if not self.horizons:
            self.horizons = (4, 8)

    # ----- derived helpers -----

    @property
    def feature_source_id(self) -> int:
        return {
            FEATURE_SOURCE_DETACHED: 0,
            FEATURE_SOURCE_SHARED: 1,
            FEATURE_SOURCE_CRITIC: 2,
        }[self.feature_source]

    @property
    def should_detach(self) -> bool:
        """Whether to detach the captured feature before the aux tower.

        ``feature_source`` is authoritative. ``actor_detach`` / ``update_actor``
        are kept for compatibility + logging; conflicts emit a warning at
        construction time in ``train.py`` and resolve in favour of
        ``feature_source``.
        """
        return self.feature_source == FEATURE_SOURCE_DETACHED

    @property
    def effective_feature_source(self) -> str:
        """``critic_feature`` degrades to shared under rl_games separate=False.

        We keep the enum so a future critic-only encoder can plug in, but the
        current stock network has no populated critic feature, so behaviour
        equals shared.
        """
        if self.feature_source == FEATURE_SOURCE_CRITIC:
            return FEATURE_SOURCE_SHARED
        return self.feature_source

    def enabled_target_types(self) -> list[str]:
        out: list[str] = []
        if self.use_near_success:
            out.append(TARGET_NEAR_SUCCESS)
        if self.use_success:
            out.append(TARGET_SUCCESS)
        if self.use_dhat_improve:
            out.append(TARGET_DHAT_IMPROVE)
        if self.use_drop_risk:
            out.append(TARGET_DROP_RISK)
        return out

    def head_keys(self) -> list[str]:
        keys: list[str] = []
        for ttype in self.enabled_target_types():
            for h in self.horizons:
                keys.append(f"{ttype}_h{int(h)}")
        return keys

    def lambda_for_tol(self, tol: float) -> float:
        """Stage-based loss weight as a function of success_tolerance."""
        if not self.enabled:
            return 0.0
        t = float(tol)
        if t > float(self.start_tol):
            return 0.0
        if t > float(self.floor_tol):
            return float(self.weight)
        if t > float(self.hard_floor_tol):
            return float(self.floor_weight)
        return float(self.hard_floor_weight)


# ---------------------------------------------------------------------------
# Heads
# ---------------------------------------------------------------------------


def build_fc_aux_tower(in_dim: int, hidden: int) -> nn.Sequential:
    """Private aux 'critic' sub-tower stacked above the captured feature.

    NOT the rl_games ``critic_mlp`` (empty under separate=false), and NOT the
    seq-distill ``seq_aux_critic_mlp``. FloorCritic owns this module.
    """
    return nn.Sequential(
        nn.Linear(in_dim, hidden),
        nn.Tanh(),
        nn.Linear(hidden, hidden),
        nn.Tanh(),
    )


class FloorCriticHeads(nn.Module):
    """Aux tower + one BCE logit head per (target_type, horizon)."""

    def __init__(self, cfg: FloorCriticConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.tower = build_fc_aux_tower(cfg.feature_dim, cfg.tower_hidden)
        self.heads = nn.ModuleDict()
        for key in cfg.head_keys():
            self.heads[key] = nn.Linear(cfg.tower_hidden, 1)

    def forward(self, feat: torch.Tensor) -> dict[str, torch.Tensor]:
        """feat: (B, feature_dim) -> {key: (B,) logits}."""
        h = self.tower(feat)
        return {key: head(h).squeeze(-1) for key, head in self.heads.items()}

    def aux_param_list(self) -> list[nn.Parameter]:
        return list(self.parameters())


def attach_floorcritic_heads(
    *,
    optimizer: torch.optim.Optimizer,
    cfg: FloorCriticConfig,
    device: Optional[torch.device | str] = None,
) -> FloorCriticHeads:
    """Build FloorCriticHeads and register their params in the PPO optimizer.

    The heads are a standalone module (not attached to the rl_games Network),
    so they can never accidentally be reached by the actor/critic forward.
    Only the FloorCritic loss touches them.
    """
    heads = FloorCriticHeads(cfg)
    if device is not None:
        heads = heads.to(device)
    base_lr = optimizer.param_groups[0]["lr"] if optimizer.param_groups else 1e-3
    optimizer.add_param_group(
        {
            "params": heads.aux_param_list(),
            "lr": base_lr,
            "name": "floorcritic_heads",
        }
    )
    return heads


# ---------------------------------------------------------------------------
# Target builder (real rollout only, on (T, N))
# ---------------------------------------------------------------------------


def _squeeze_tn(t: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    if t is None:
        return None
    if t.dim() == 3 and t.shape[-1] == 1:
        return t.squeeze(-1)
    return t


def _flatten_tn(x: torch.Tensor) -> torch.Tensor:
    """(T, N) -> (T*N,) with flat[n*T + t] = x[t, n] (rl_games swap_and_flatten01)."""
    T, N = x.shape
    return x.transpose(0, 1).reshape(T * N).contiguous()


@dataclass
class FloorCriticTargetCache:
    """Frozen per-rollout targets + masks, flattened to (T*N,)."""

    targets_flat: dict[str, torch.Tensor] = field(default_factory=dict)
    masks_flat: dict[str, torch.Tensor] = field(default_factory=dict)
    pos_rate: dict[str, float] = field(default_factory=dict)
    valid_fraction: dict[str, float] = field(default_factory=dict)
    degenerate: dict[str, bool] = field(default_factory=dict)
    tol_mean: float = 1.0
    generation: int = -1
    T: int = 0
    N: int = 0

    def is_valid_for(self, generation: int) -> bool:
        return self.generation == generation and self.generation >= 0


class FloorCriticTargetBuilder:
    """Builds floor-stage targets from a PPO ExperienceBuffer's (T, N) aux."""

    def __init__(self, cfg: FloorCriticConfig) -> None:
        self.cfg = cfg

    @staticmethod
    def _horizon_valid_mask(
        dones: torch.Tensor, horizon: int
    ) -> torch.Tensor:
        """(T, N) float mask: 1 if next ``horizon`` steps stay in-episode.

        Step t is valid iff no done at t..t+h-1 (no episode boundary crossing)
        AND t + h < T (the future window fits inside the rollout).
        """
        T, N = dones.shape
        h = max(1, int(horizon))
        not_done = (1.0 - dones.float())
        mask = dones.new_zeros(T, N)
        for t in range(T):
            if t + h >= T:
                continue
            mask[t] = not_done[t : t + h].prod(dim=0)
        return mask

    def build(
        self,
        *,
        d_hat: torch.Tensor,
        success_cnt: torch.Tensor,
        success_tol: torch.Tensor,
        failure: Optional[torch.Tensor],
        lifted: Optional[torch.Tensor],
        dones: torch.Tensor,
        generation: int,
    ) -> FloorCriticTargetCache:
        cfg = self.cfg
        d_hat = _squeeze_tn(d_hat).float()
        success_cnt = _squeeze_tn(success_cnt).float()
        success_tol = _squeeze_tn(success_tol).float()
        dones = _squeeze_tn(dones).float()
        failure = _squeeze_tn(failure)
        lifted = _squeeze_tn(lifted)
        if failure is not None:
            failure = failure.float()
        if lifted is not None:
            lifted = lifted.float()

        T, N = d_hat.shape
        cache = FloorCriticTargetCache(generation=int(generation), T=T, N=N)

        # Per-rollout scalar tolerance for lambda gating: masked mean over
        # valid (non-done) steps, matching rssm_lite floor_mult precedent.
        with torch.no_grad():
            valid_steps = (1.0 - dones).clamp(0.0, 1.0)
            denom = max(float(valid_steps.sum().item()), 1.0)
            cache.tol_mean = float(
                (success_tol * valid_steps).sum().item() / denom
            )

        for h in cfg.horizons:
            h = int(h)
            valid = self._horizon_valid_mask(dones, h)  # (T, N)
            near = d_hat.new_zeros(T, N)
            succ = d_hat.new_zeros(T, N)
            impr = d_hat.new_zeros(T, N)
            drop = d_hat.new_zeros(T, N)
            for t in range(T):
                if t + h >= T:
                    continue
                fut = slice(t + 1, t + h + 1)  # k = 1..h
                dh_future = d_hat[fut]  # (h, N)
                tol_t = success_tol[t]
                near[t] = (
                    dh_future <= (cfg.near_success_mult * tol_t)
                ).any(dim=0).float()
                sc_future = success_cnt[fut]
                succ[t] = (sc_future > success_cnt[t]).any(dim=0).float()
                impr[t] = (
                    dh_future.min(dim=0).values
                    <= (d_hat[t] - cfg.dhat_improve_margin)
                ).float()
                # drop risk: failure OR d_hat rises OR lifted 1->0 (derived
                # dropped, since aux_dropped is not populated by the env).
                dhat_rise = (
                    (dh_future - d_hat[t]) >= cfg.drop_dhat_margin
                ).any(dim=0)
                fail_future = (
                    (failure[fut] > 0.5).any(dim=0)
                    if failure is not None
                    else torch.zeros(N, dtype=torch.bool, device=d_hat.device)
                )
                if lifted is not None:
                    lifted_drop = (lifted[t] > 0.5) & (
                        (lifted[fut] < 0.5).any(dim=0)
                    )
                else:
                    lifted_drop = torch.zeros(
                        N, dtype=torch.bool, device=d_hat.device
                    )
                drop[t] = (
                    fail_future | dhat_rise | lifted_drop
                ).float()

            per_type = {
                TARGET_NEAR_SUCCESS: near,
                TARGET_SUCCESS: succ,
                TARGET_DHAT_IMPROVE: impr,
                TARGET_DROP_RISK: drop,
            }
            valid_flat = _flatten_tn(valid)
            for ttype in cfg.enabled_target_types():
                key = f"{ttype}_h{h}"
                tgt = per_type[ttype]
                cache.targets_flat[key] = _flatten_tn(tgt)
                cache.masks_flat[key] = valid_flat
                with torch.no_grad():
                    msum = float(valid.sum().item())
                    pos = (
                        float((tgt * valid).sum().item()) / max(msum, 1.0)
                    )
                    cache.pos_rate[key] = pos
                    cache.valid_fraction[key] = msum / float(max(T * N, 1))
                    cache.degenerate[key] = (pos <= 1e-6) or (pos >= 1.0 - 1e-6)
        return cache


# ---------------------------------------------------------------------------
# Forward-hook capture (independent of seq-distill capture)
# ---------------------------------------------------------------------------


@dataclass
class FloorCriticCapture:
    """Mutable container for ``actor_mlp_out`` of the current minibatch."""

    tensor: Optional[torch.Tensor] = None
    fired: bool = False

    def reset(self) -> None:
        self.tensor = None
        self.fired = False


def register_floorcritic_capture(
    rl_games_model: nn.Module,
    capture: FloorCriticCapture,
) -> Any:
    """Install an INDEPENDENT forward hook on ``a2c_network.actor_mlp``.

    Separate from the seq-distill hook so the two can coexist (both target
    the same module; each writes its own ``CapturedFeature``-like container).
    Returns the hook handle.
    """
    inner = rl_games_model
    if hasattr(inner, "model"):
        inner = inner.model
    a2c = getattr(inner, "a2c_network", None)
    if a2c is None or not hasattr(a2c, "actor_mlp"):
        raise RuntimeError(
            "rl_games model does not expose a2c_network.actor_mlp; "
            "cannot install FloorCritic feature hook."
        )

    def hook_fn(_module: nn.Module, _inp: Any, out: torch.Tensor) -> None:
        capture.tensor = out.to(torch.float32)
        capture.fired = True

    return a2c.actor_mlp.register_forward_hook(hook_fn)


def register_actor_mlp_grad_probe(
    rl_games_model: nn.Module,
    bridge: "FloorCriticBridge",
) -> Any:
    """Install a backward hook on actor_mlp to estimate total grad norm.

    During the PPO backward the gradient flowing into ``actor_mlp`` output is
    the TOTAL grad (PPO + any FloorCritic contribution). When the FloorCritic
    contribution is kept small (which is the regime we maintain), this is a
    first-order estimate of the PPO gradient magnitude on the shared feature,
    used as the denominator of ``fc_grad_ratio_to_ppo_grad``.
    """
    inner = rl_games_model
    if hasattr(inner, "model"):
        inner = inner.model
    a2c = getattr(inner, "a2c_network", None)
    if a2c is None or not hasattr(a2c, "actor_mlp"):
        raise RuntimeError(
            "rl_games model does not expose a2c_network.actor_mlp; "
            "cannot install FloorCritic grad probe."
        )

    def bwd_hook(_module, _grad_input, grad_output):
        try:
            g = grad_output[0]
            if g is not None:
                bridge.update_ppo_grad_ema(float(g.detach().norm().item()))
        except Exception:
            pass

    return a2c.actor_mlp.register_full_backward_hook(bwd_hook)


# ---------------------------------------------------------------------------
# Bridge + loss
# ---------------------------------------------------------------------------


@dataclass
class FloorCriticBridge:
    """Holds runtime refs and per-call logging for FloorCritic."""

    cfg: FloorCriticConfig
    heads: Optional[FloorCriticHeads] = None
    capture: FloorCriticCapture = field(default_factory=FloorCriticCapture)
    agent_ref: Optional[Any] = None
    cache: Optional[FloorCriticTargetCache] = None

    # EMA of PPO grad norm on the shared feature (from backward probe).
    ppo_grad_ema: float = 0.0
    ppo_grad_ema_beta: float = 0.9

    # Per-call logging snapshot.
    last_active: float = 0.0
    last_lambda: float = 0.0
    last_loss_total: float = 0.0
    last_losses: dict[str, float] = field(default_factory=dict)
    last_pos_rate: dict[str, float] = field(default_factory=dict)
    last_valid_fraction: float = 0.0
    last_fc_feat_grad_norm: float = 0.0
    last_fc_grad_ratio: float = 0.0
    last_auto_scale_factor: float = 1.0
    last_actor_grad_blocked: bool = True
    last_aux_head_grad_norm: float = 0.0

    def update_ppo_grad_ema(self, grad_norm: float) -> None:
        b = float(self.ppo_grad_ema_beta)
        if self.ppo_grad_ema <= 0.0:
            self.ppo_grad_ema = float(grad_norm)
        else:
            self.ppo_grad_ema = b * self.ppo_grad_ema + (1.0 - b) * float(grad_norm)

    def reset_per_call(self) -> None:
        self.last_active = 0.0
        self.last_lambda = 0.0
        self.last_loss_total = 0.0
        self.last_losses = {}
        self.last_pos_rate = {}
        self.last_valid_fraction = 0.0
        self.last_fc_feat_grad_norm = 0.0
        self.last_fc_grad_ratio = 0.0
        self.last_auto_scale_factor = 1.0
        self.last_actor_grad_blocked = True
        self.last_aux_head_grad_norm = 0.0


def _read_minibatch_range(bridge: FloorCriticBridge) -> tuple[int, int]:
    """Read (start, end) from rl_games' PPODataset.last_range; fail fast."""
    agent = bridge.agent_ref
    if agent is None:
        raise RuntimeError("[FloorCritic] bridge.agent_ref is None.")
    dataset = getattr(agent, "dataset", None)
    if dataset is None:
        raise RuntimeError("[FloorCritic] agent.dataset is None.")
    last_range = getattr(dataset, "last_range", None)
    if (
        last_range is None
        or not isinstance(last_range, tuple)
        or len(last_range) != 2
    ):
        raise RuntimeError(
            "[FloorCritic] agent.dataset.last_range missing or malformed."
        )
    start, end = int(last_range[0]), int(last_range[1])
    if end <= start:
        raise RuntimeError(
            f"[FloorCritic] invalid minibatch range start={start} end={end}"
        )
    return start, end


def _masked_bce(
    logit: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
) -> tuple[torch.Tensor, float]:
    """BCE-with-logits over masked entries. Returns (loss, n_valid)."""
    m = mask > 0.5
    n = int(m.sum().item())
    if n == 0:
        return logit.new_zeros(()), 0.0
    loss = F.binary_cross_entropy_with_logits(
        logit[m], target[m], reduction="mean"
    )
    return loss, float(n)


def _estimate_fc_feature_grad_norm(
    heads: FloorCriticHeads,
    feat: torch.Tensor,
    targets: dict[str, torch.Tensor],
    masks: dict[str, torch.Tensor],
) -> float:
    """Isolated probe: ||d L_FC / d feat|| on a detached clone of feat.

    This measures how strongly the FloorCritic loss would push the shared
    feature, WITHOUT touching the live PPO graph (own tiny graph on a clone).
    """
    try:
        feat_probe = feat.detach().clone().requires_grad_(True)
        logits = heads(feat_probe)
        total = feat_probe.new_zeros(())
        any_term = False
        for key, logit in logits.items():
            if key not in targets:
                continue
            loss, n = _masked_bce(logit, targets[key], masks[key])
            if n > 0:
                total = total + loss
                any_term = True
        if not any_term:
            return 0.0
        (g,) = torch.autograd.grad(total, feat_probe, retain_graph=False)
        return float(g.detach().norm().item())
    except Exception:
        return 0.0


def compute_floorcritic_loss(
    bridge: FloorCriticBridge,
) -> dict[str, torch.Tensor]:
    """Build the FloorCritic loss dict for the current PPO minibatch.

    Returns ``{}`` when disabled / inactive / no cache / capture missing.
    """
    bridge.reset_per_call()
    cfg = bridge.cfg
    if not cfg.enabled or bridge.heads is None:
        return {}
    cache = bridge.cache
    if cache is None or not bridge.capture.fired or bridge.capture.tensor is None:
        return {}

    lam = cfg.lambda_for_tol(cache.tol_mean)
    bridge.last_lambda = float(lam)
    if lam <= 0.0:
        return {}

    start, end = _read_minibatch_range(bridge)
    feat = bridge.capture.tensor
    if feat.dim() != 2 or feat.shape[0] != (end - start):
        # Shape mismatch with the minibatch range — refuse silent fallback.
        raise RuntimeError(
            f"[FloorCritic] captured feature shape {tuple(feat.shape)} does "
            f"not match minibatch range ({start},{end})."
        )

    device = feat.device
    dtype = feat.dtype
    # feature_source routing.
    if cfg.should_detach:
        feat_in = feat.detach()
        bridge.last_actor_grad_blocked = True
    else:
        feat_in = feat  # shared: gradient flows to backbone (never to mu)
        bridge.last_actor_grad_blocked = True  # FC never connects mu head

    # Slice + move targets to the minibatch.
    sliced_targets: dict[str, torch.Tensor] = {}
    sliced_masks: dict[str, torch.Tensor] = {}
    for key, tgt in cache.targets_flat.items():
        sliced_targets[key] = tgt[start:end].to(device=device, dtype=dtype)
        sliced_masks[key] = cache.masks_flat[key][start:end].to(
            device=device, dtype=dtype
        )

    logits = bridge.heads(feat_in)
    total = feat_in.new_zeros(())
    n_valid_total = 0.0
    n_terms = 0
    for key, logit in logits.items():
        if key not in sliced_targets:
            continue
        loss, n = _masked_bce(logit, sliced_targets[key], sliced_masks[key])
        bridge.last_losses[key] = float(loss.detach().item()) if n > 0 else 0.0
        bridge.last_pos_rate[key] = float(cache.pos_rate.get(key, 0.0))
        if n > 0:
            total = total + loss
            n_valid_total += n
            n_terms += 1
    if n_terms == 0:
        return {}

    # Grad-ratio audit + auto-scale (shared mode only; detached cannot move
    # the backbone so ratio is 0 and factor stays 1).
    auto_scale = 1.0
    if cfg.effective_feature_source == FEATURE_SOURCE_SHARED:
        fc_norm = _estimate_fc_feature_grad_norm(
            bridge.heads, feat, sliced_targets, sliced_masks
        )
        bridge.last_fc_feat_grad_norm = float(fc_norm)
        ppo_norm = float(bridge.ppo_grad_ema)
        if ppo_norm > 1e-12:
            ratio = fc_norm / ppo_norm
        else:
            ratio = 0.0
        bridge.last_fc_grad_ratio = float(ratio)
        if (
            cfg.auto_scale_by_grad_ratio
            and ratio > float(cfg.max_grad_ratio)
            and ratio > 0.0
        ):
            auto_scale = float(cfg.max_grad_ratio) / ratio
    bridge.last_auto_scale_factor = float(auto_scale)

    scaled = float(lam) * float(auto_scale) * total

    bridge.last_active = 1.0
    bridge.last_loss_total = float(total.detach().item())
    if cache.valid_fraction:
        bridge.last_valid_fraction = float(
            sum(cache.valid_fraction.values()) / max(len(cache.valid_fraction), 1)
        )
    return {"floorcritic_total": scaled}


# ---------------------------------------------------------------------------
# Gradient audit (hard guarantees, used by tests + optional runtime probe)
# ---------------------------------------------------------------------------


def compute_grad_audit(
    *,
    rl_games_model: nn.Module,
    heads: FloorCriticHeads,
    capture: FloorCriticCapture,
    cfg: FloorCriticConfig,
    targets: dict[str, torch.Tensor],
    masks: dict[str, torch.Tensor],
) -> dict[str, float]:
    """Run an isolated FloorCritic-only backward on the REAL model and report
    gradient norms on mu head / value head / shared backbone / aux heads.

    Used by tests for hard guarantees:
      * detached: backbone + mu + value grad-from-FC are all 0.
      * shared:   backbone grad-from-FC > 0, but mu grad-from-FC == 0.
    Caller is responsible for having run a forward pass so ``capture.tensor``
    is populated, and for zeroing grads before/after.
    """
    inner = rl_games_model
    if hasattr(inner, "model"):
        inner = inner.model
    a2c = getattr(inner, "a2c_network", None)

    feat = capture.tensor
    if feat is None:
        raise RuntimeError("[FloorCritic] grad audit: capture.tensor is None.")
    feat_in = feat.detach() if cfg.should_detach else feat

    logits = heads(feat_in)
    total = feat_in.new_zeros(())
    for key, logit in logits.items():
        if key not in targets:
            continue
        loss, n = _masked_bce(logit, targets[key], masks[key])
        if n > 0:
            total = total + loss

    # Zero grads on the params we will inspect.
    params = {}
    if a2c is not None:
        if hasattr(a2c, "mu"):
            params["mu"] = list(a2c.mu.parameters())
        if hasattr(a2c, "value"):
            params["value"] = list(a2c.value.parameters())
        if hasattr(a2c, "actor_mlp"):
            params["actor_mlp"] = list(a2c.actor_mlp.parameters())
    params["aux_heads"] = heads.aux_param_list()
    for plist in params.values():
        for p in plist:
            if p.grad is not None:
                p.grad = None

    total.backward(retain_graph=False)

    def _norm(plist: list[nn.Parameter]) -> float:
        s = 0.0
        for p in plist:
            if p.grad is not None:
                s += float(p.grad.detach().norm().item() ** 2)
        return float(s) ** 0.5

    out = {
        "actor_head_grad_from_fc": _norm(params.get("mu", [])),
        "value_head_grad_from_fc": _norm(params.get("value", [])),
        "shared_backbone_grad_from_fc": _norm(params.get("actor_mlp", [])),
        "aux_head_grad_norm": _norm(params.get("aux_heads", [])),
    }
    out["actor_grad_blocked"] = 1.0 if out["actor_head_grad_from_fc"] <= 1e-9 else 0.0
    return out
