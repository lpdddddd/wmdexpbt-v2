"""ERD-MPC — action-level recovery distillation for DexPBT PPO.

Elite-Recovery Distillation with Model-Predictive Action Search (ERD-MPC)
trains an action-conditioned recovery scorer from real PPO rollouts, searches
candidate recovery actions in risky floor-stage states, and distills the best
candidate into the actor mean. Unlike FloorCritic, this is intentionally
action-level: the distillation path updates the actor ``mu`` head.

All rollout targets use the same rl_games flattening convention as
FloorCritic: ``flat[n * T + t] = x[t, n]`` so ``PPODataset.last_range`` slices
align to the live minibatch.

Action-space convention: ERD trains its scorer and searches candidates in the
clipped, env-usable action space ``[-1, 1]``. Rollout actions are clamped before
entering the cache so scorer supervision is aligned with actions the env can
execute. Distillation intentionally pulls the live actor ``mu`` toward the
clipped best candidate, so diagnostics below track how often raw ``mu`` and
candidates interact with the action bounds.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


TARGET_SUCCESS = "success"
TARGET_IMPROVE = "improve"
TARGET_DROP = "drop"

BLOCK_OK = 0
BLOCK_WARMUP_FRAMES = 1
BLOCK_WARMUP_ITERS = 2
BLOCK_LOW_VALID_FRACTION = 3
BLOCK_LOW_MARGIN = 4
BLOCK_DROP_RISK_HIGH = 5
BLOCK_NO_RISKY = 6
BLOCK_TOL_ABOVE_START = 7


@dataclass
class ERDConfig:
    enabled: bool = False
    horizons: tuple[int, ...] = (4, 8)

    candidate_k: int = 32
    action_noise_std: float = 0.15
    max_action_delta: float = 0.5
    noise_cap_by_tol: bool = True

    score_success_weight: float = 1.0
    score_improve_weight: float = 0.7
    score_drop_weight: float = 1.5

    scorer_loss_weight: float = 0.01
    distill_loss_weight: float = 0.03
    floor_distill_loss_weight: float = 0.05
    hard_floor_distill_loss_weight: float = 0.08

    start_tol: float = 0.04
    floor_tol: float = 0.02
    hard_floor_tol: float = 0.012
    dhat_improve_margin: float = 0.005
    drop_dhat_margin: float = 0.01
    near_success_mult: float = 2.0

    # Recovery-label margin generalization.  ``absolute`` keeps the legacy
    # fixed-distance behaviour (used by all pre-existing profiles).  ``relative_tol``
    # scales the improvement / drop margins with the per-row success tolerance so
    # the labels are curriculum-scale invariant (used by progress_refine).
    dhat_margin_mode: str = "absolute"  # absolute | relative_tol
    dhat_improve_margin_relative: float = 0.25
    drop_dhat_margin_relative: float = 0.50
    dhat_improve_margin_min_abs: float = 0.0
    drop_dhat_margin_min_abs: float = 0.0

    max_grad_ratio: float = 0.2
    auto_scale_by_grad_ratio: bool = True
    only_lifted: bool = True
    min_risky_fraction: float = 0.02

    start_after_frames: int = 200_000_000
    scorer_warmup_iters: int = 2000
    min_score_margin: float = 0.05
    min_target_valid_fraction: float = 0.2
    max_drop_prob_for_teacher: float = 0.5

    feature_dim: int = 256
    hidden_dim: int = 256
    action_dim: int = 23

    def __post_init__(self) -> None:
        self.horizons = tuple(int(h) for h in self.horizons if int(h) > 0)
        if not self.horizons:
            self.horizons = (4, 8)
        self.candidate_k = max(1, int(self.candidate_k))

    def head_keys(self) -> list[str]:
        out: list[str] = []
        for h in self.horizons:
            out.extend([f"success_h{h}", f"improve_h{h}", f"drop_h{h}"])
        return out

    def distill_lambda_for_tol(self, tol: float) -> float:
        if not self.enabled:
            return 0.0
        t = float(tol)
        if t > float(self.start_tol):
            return 0.0
        if t > float(self.floor_tol):
            return float(self.distill_loss_weight)
        if t > float(self.hard_floor_tol):
            return float(self.floor_distill_loss_weight)
        return float(self.hard_floor_distill_loss_weight)


def _squeeze_tn(t: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    if t is None:
        return None
    if t.dim() == 3 and t.shape[-1] == 1:
        return t.squeeze(-1)
    return t


def _flatten_tn(x: torch.Tensor) -> torch.Tensor:
    T, N = x.shape[:2]
    return x.transpose(0, 1).reshape(T * N, *x.shape[2:]).contiguous()


@dataclass
class ERDTargetCache:
    targets_flat: dict[str, torch.Tensor] = field(default_factory=dict)
    masks_flat: dict[str, torch.Tensor] = field(default_factory=dict)
    actions_flat: Optional[torch.Tensor] = None
    d_hat_flat: Optional[torch.Tensor] = None
    success_tol_flat: Optional[torch.Tensor] = None
    lifted_flat: Optional[torch.Tensor] = None
    # Progress-conditioned refinement context (flat[n * T + t] layout, aligned to
    # PPODataset.last_range).  ``None`` unless a progress config is supplied so the
    # legacy profiles keep their exact cache contents.
    normalized_error_flat: Optional[torch.Tensor] = None
    recent_progress_flat: Optional[torch.Tensor] = None
    oscillation_flat: Optional[torch.Tensor] = None
    progress_history_valid_flat: Optional[torch.Tensor] = None
    pos_rate: dict[str, float] = field(default_factory=dict)
    valid_fraction: dict[str, float] = field(default_factory=dict)
    degenerate: dict[str, bool] = field(default_factory=dict)
    tol_mean: float = 1.0
    generation: int = -1
    T: int = 0
    N: int = 0
    action_dim: int = 0

    def mean_valid_fraction(self) -> float:
        if not self.valid_fraction:
            return 0.0
        return float(sum(self.valid_fraction.values()) / len(self.valid_fraction))


class ERDTargetBuilder:
    def __init__(self, cfg: ERDConfig, progress_cfg: Optional[Any] = None) -> None:
        self.cfg = cfg
        # Optional ProgressRefinementConfig; when enabled the cache is populated
        # with the scale-invariant progress context for the refinement gate.
        self.progress_cfg = progress_cfg

    @staticmethod
    def _horizon_valid_mask(dones: torch.Tensor, horizon: int) -> torch.Tensor:
        T, N = dones.shape
        h = max(1, int(horizon))
        not_done = 1.0 - dones.float()
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
        actions: torch.Tensor,
        generation: int,
    ) -> ERDTargetCache:
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
        actions = actions.float()
        T, N = d_hat.shape
        action_dim = int(actions.shape[-1])
        cache = ERDTargetCache(
            generation=int(generation), T=T, N=N, action_dim=action_dim
        )
        cache.actions_flat = _flatten_tn(actions).clamp(-1.0, 1.0)
        cache.d_hat_flat = _flatten_tn(d_hat).view(T * N)
        cache.success_tol_flat = _flatten_tn(success_tol).view(T * N)
        if lifted is not None:
            cache.lifted_flat = _flatten_tn(lifted).view(T * N)

        with torch.no_grad():
            valid_steps = (1.0 - dones).clamp(0.0, 1.0)
            denom = max(float(valid_steps.sum().item()), 1.0)
            cache.tol_mean = float((success_tol * valid_steps).sum().item() / denom)

        relative_margin = str(getattr(cfg, "dhat_margin_mode", "absolute")) == "relative_tol"
        for h in cfg.horizons:
            h = int(h)
            valid = self._horizon_valid_mask(dones, h)
            succ = d_hat.new_zeros(T, N)
            impr = d_hat.new_zeros(T, N)
            drop = d_hat.new_zeros(T, N)
            for t in range(T):
                if t + h >= T:
                    continue
                fut = slice(t + 1, t + h + 1)
                dh_future = d_hat[fut]
                # Per-row improvement / drop margins.  ``relative_tol`` scales with
                # the current success tolerance (curriculum-scale invariant);
                # ``absolute`` reproduces the legacy fixed margins exactly.
                if relative_margin:
                    m_impr = torch.clamp(
                        float(cfg.dhat_improve_margin_relative) * success_tol[t],
                        min=float(cfg.dhat_improve_margin_min_abs),
                    )
                    m_drop = torch.clamp(
                        float(cfg.drop_dhat_margin_relative) * success_tol[t],
                        min=float(cfg.drop_dhat_margin_min_abs),
                    )
                else:
                    m_impr = cfg.dhat_improve_margin
                    m_drop = cfg.drop_dhat_margin
                succ[t] = (success_cnt[fut] > success_cnt[t]).any(dim=0).float()
                impr[t] = (
                    dh_future.min(dim=0).values
                    <= (d_hat[t] - m_impr)
                ).float()
                dhat_rise = ((dh_future - d_hat[t]) >= m_drop).any(dim=0)
                fail_future = (
                    (failure[fut] > 0.5).any(dim=0)
                    if failure is not None
                    else torch.zeros(N, dtype=torch.bool, device=d_hat.device)
                )
                lifted_drop = (
                    (lifted[t] > 0.5) & ((lifted[fut] < 0.5).any(dim=0))
                    if lifted is not None
                    else torch.zeros(N, dtype=torch.bool, device=d_hat.device)
                )
                drop[t] = (fail_future | dhat_rise | lifted_drop).float()

            for ttype, tgt in (
                (TARGET_SUCCESS, succ),
                (TARGET_IMPROVE, impr),
                (TARGET_DROP, drop),
            ):
                key = f"{ttype}_h{h}"
                cache.targets_flat[key] = _flatten_tn(tgt).view(T * N)
                cache.masks_flat[key] = _flatten_tn(valid).view(T * N)
                with torch.no_grad():
                    msum = float(valid.sum().item())
                    pos = float((tgt * valid).sum().item()) / max(msum, 1.0)
                    cache.pos_rate[key] = pos
                    cache.valid_fraction[key] = msum / float(max(T * N, 1))
                    cache.degenerate[key] = (pos <= 1e-6) or (pos >= 1.0 - 1e-6)

        pcfg = self.progress_cfg
        if pcfg is not None and bool(getattr(pcfg, "enabled", False)):
            from .progress_conditioned_refinement import build_progress_context_tn

            ctx = build_progress_context_tn(
                d_hat_tn=d_hat,
                success_tol_tn=success_tol,
                dones_tn=dones,
                cfg=pcfg,
            )
            cache.normalized_error_flat = ctx["normalized_error_flat"]
            cache.recent_progress_flat = ctx["recent_progress_flat"]
            cache.oscillation_flat = ctx["oscillation_flat"]
            cache.progress_history_valid_flat = ctx["progress_history_valid_flat"]
        return cache


class ERDActionScorer(nn.Module):
    def __init__(self, cfg: ERDConfig) -> None:
        super().__init__()
        self.cfg = cfg
        in_dim = int(cfg.feature_dim) + int(cfg.action_dim)
        hidden = int(cfg.hidden_dim)
        self.trunk = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
        )
        self.heads = nn.ModuleDict({key: nn.Linear(hidden, 1) for key in cfg.head_keys()})

    def forward(self, feat: torch.Tensor, action: torch.Tensor) -> dict[str, torch.Tensor]:
        # Scorer inputs are env-usable clipped actions, not unbounded policy means.
        action = action.clamp(-1.0, 1.0)
        h = self.trunk(torch.cat([feat, action], dim=-1))
        return {key: head(h).squeeze(-1) for key, head in self.heads.items()}


def attach_erd_modules(
    *,
    optimizer: torch.optim.Optimizer,
    cfg: ERDConfig,
    device: Optional[torch.device | str] = None,
) -> ERDActionScorer:
    scorer = ERDActionScorer(cfg)
    if device is not None:
        scorer = scorer.to(device)
    base_lr = optimizer.param_groups[0]["lr"] if optimizer.param_groups else 1e-3
    optimizer.add_param_group(
        {"params": list(scorer.parameters()), "lr": base_lr, "name": "erd_scorer"}
    )
    return scorer


@dataclass
class ERDCapture:
    tensor: Optional[torch.Tensor] = None
    fired: bool = False

    def reset(self) -> None:
        self.tensor = None
        self.fired = False


def register_erd_capture(rl_games_model: nn.Module, capture: ERDCapture) -> Any:
    inner = rl_games_model
    if hasattr(inner, "model"):
        inner = inner.model
    a2c = getattr(inner, "a2c_network", None)
    if a2c is None or not hasattr(a2c, "actor_mlp"):
        raise RuntimeError("[ERD] model does not expose a2c_network.actor_mlp")

    def hook_fn(_module: nn.Module, _inp: Any, out: torch.Tensor) -> None:
        capture.tensor = out.to(torch.float32)
        capture.fired = True

    return a2c.actor_mlp.register_forward_hook(hook_fn)


def register_erd_actor_mlp_grad_probe(rl_games_model: nn.Module, bridge: "ERDBridge") -> Any:
    inner = rl_games_model
    if hasattr(inner, "model"):
        inner = inner.model
    a2c = getattr(inner, "a2c_network", None)
    if a2c is None or not hasattr(a2c, "actor_mlp"):
        raise RuntimeError("[ERD] model does not expose a2c_network.actor_mlp")

    def bwd_hook(_module, _grad_input, grad_output):
        try:
            g = grad_output[0]
            if g is not None:
                bridge.update_ppo_grad_ema(float(g.detach().norm().item()))
        except Exception:
            pass

    return a2c.actor_mlp.register_full_backward_hook(bwd_hook)


@dataclass
class ERDBridge:
    cfg: ERDConfig
    scorer: Optional[ERDActionScorer] = None
    capture: ERDCapture = field(default_factory=ERDCapture)
    agent_ref: Optional[Any] = None
    cache: Optional[ERDTargetCache] = None

    scorer_warmup_iters_done: int = 0
    ppo_grad_ema: float = 0.0
    ppo_grad_ema_beta: float = 0.9

    last_active: float = 0.0
    last_scorer_active: float = 0.0
    last_distill_active: float = 0.0
    last_distill_block_reason: int = BLOCK_OK
    last_risky_fraction: float = 0.0
    last_scorer_loss: float = 0.0
    last_distill_loss: float = 0.0
    last_total_loss: float = 0.0
    last_best_score: float = 0.0
    last_actor_score: float = 0.0
    last_score_margin: float = 0.0
    last_best_success_prob: float = 0.0
    last_best_improve_prob: float = 0.0
    last_best_drop_prob: float = 0.0
    last_actor_success_prob: float = 0.0
    last_actor_drop_prob: float = 0.0
    last_action_delta_norm: float = 0.0
    last_raw_mu_abs_mean: float = 0.0
    last_raw_mu_clip_fraction: float = 0.0
    last_candidate_clip_fraction: float = 0.0
    last_distill_grad_ratio: float = 0.0
    last_auto_scale_factor: float = 1.0
    last_target_valid_fraction: float = 0.0
    last_pos_rate: dict[str, float] = field(default_factory=dict)

    def update_ppo_grad_ema(self, grad_norm: float) -> None:
        b = float(self.ppo_grad_ema_beta)
        if self.ppo_grad_ema <= 0.0:
            self.ppo_grad_ema = float(grad_norm)
        else:
            self.ppo_grad_ema = b * self.ppo_grad_ema + (1.0 - b) * float(grad_norm)

    def reset_per_call(self) -> None:
        self.last_active = 0.0
        self.last_scorer_active = 0.0
        self.last_distill_active = 0.0
        self.last_distill_block_reason = BLOCK_OK
        self.last_risky_fraction = 0.0
        self.last_scorer_loss = 0.0
        self.last_distill_loss = 0.0
        self.last_total_loss = 0.0
        self.last_best_score = 0.0
        self.last_actor_score = 0.0
        self.last_score_margin = 0.0
        self.last_best_success_prob = 0.0
        self.last_best_improve_prob = 0.0
        self.last_best_drop_prob = 0.0
        self.last_actor_success_prob = 0.0
        self.last_actor_drop_prob = 0.0
        self.last_action_delta_norm = 0.0
        self.last_raw_mu_abs_mean = 0.0
        self.last_raw_mu_clip_fraction = 0.0
        self.last_candidate_clip_fraction = 0.0
        self.last_distill_grad_ratio = 0.0
        self.last_auto_scale_factor = 1.0
        self.last_target_valid_fraction = 0.0
        self.last_pos_rate = {}


def _read_minibatch_range(bridge: ERDBridge) -> tuple[int, int]:
    agent = bridge.agent_ref
    if agent is None:
        raise RuntimeError("[ERD] bridge.agent_ref is None")
    dataset = getattr(agent, "dataset", None)
    if dataset is None:
        raise RuntimeError("[ERD] agent.dataset is None")
    last_range = getattr(dataset, "last_range", None)
    if not isinstance(last_range, tuple) or len(last_range) != 2:
        raise RuntimeError("[ERD] agent.dataset.last_range missing or malformed")
    start, end = int(last_range[0]), int(last_range[1])
    if end <= start:
        raise RuntimeError(f"[ERD] invalid minibatch range start={start} end={end}")
    return start, end


def _get_a2c_network(agent: Any) -> Any:
    model = getattr(agent, "model", None)
    inner = model.model if hasattr(model, "model") else model
    a2c = getattr(inner, "a2c_network", None)
    if a2c is None or not hasattr(a2c, "mu"):
        raise RuntimeError("[ERD] model does not expose a2c_network.mu")
    return a2c


def _masked_bce(
    logit: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
) -> tuple[torch.Tensor, float]:
    m = mask > 0.5
    n = int(m.sum().item())
    if n == 0:
        return logit.new_zeros(()), 0.0
    return F.binary_cross_entropy_with_logits(logit[m], target[m], reduction="mean"), float(n)


def _combined_score_from_logits(
    cfg: ERDConfig, logits: dict[str, torch.Tensor]
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    scores = []
    succ_probs = []
    impr_probs = []
    drop_probs = []
    for h in cfg.horizons:
        sk, ik, dk = f"success_h{h}", f"improve_h{h}", f"drop_h{h}"
        if sk not in logits or ik not in logits or dk not in logits:
            continue
        sp = torch.sigmoid(logits[sk])
        ip = torch.sigmoid(logits[ik])
        dp = torch.sigmoid(logits[dk])
        scores.append(
            cfg.score_success_weight * sp
            + cfg.score_improve_weight * ip
            - cfg.score_drop_weight * dp
        )
        succ_probs.append(sp)
        impr_probs.append(ip)
        drop_probs.append(dp)
    if not scores:
        ref = next(iter(logits.values()))
        z = ref.new_zeros(ref.shape)
        return z, z, z, z
    return (
        torch.stack(scores, dim=0).mean(dim=0),
        torch.stack(succ_probs, dim=0).mean(dim=0),
        torch.stack(impr_probs, dim=0).mean(dim=0),
        torch.stack(drop_probs, dim=0).mean(dim=0),
    )


def _build_risky_mask(
    cfg: ERDConfig,
    *,
    d_hat: torch.Tensor,
    tol: torch.Tensor,
    lifted: Optional[torch.Tensor],
) -> torch.Tensor:
    near = d_hat <= (cfg.near_success_mult * tol).clamp_min(1e-6)
    tol_gate = tol <= float(cfg.start_tol)
    risky = tol_gate | near
    if cfg.only_lifted and lifted is not None:
        risky = risky & (lifted > 0.5)
    return risky


def _noise_std_for_tol(cfg: ERDConfig, tol_mean: float) -> float:
    std = float(cfg.action_noise_std)
    if not cfg.noise_cap_by_tol:
        return std
    if tol_mean <= cfg.hard_floor_tol:
        return std * 0.5
    if tol_mean <= cfg.floor_tol:
        return std * 0.75
    return std


def _generate_candidate_actions_with_clip_fraction(
    actor_mu: torch.Tensor, cfg: ERDConfig, tol_mean: float
) -> tuple[torch.Tensor, float]:
    B, A = actor_mu.shape
    K = int(cfg.candidate_k)
    # Candidate search operates around the clipped actor mean used by the env.
    base = actor_mu.detach().clamp(-1.0, 1.0)
    if K == 1:
        return base[:, None, :], 0.0
    std = _noise_std_for_tol(cfg, tol_mean)
    noise = torch.randn(B, K - 1, A, device=actor_mu.device, dtype=actor_mu.dtype) * std
    max_delta = float(cfg.max_action_delta)
    noise = noise.clamp(-max_delta, max_delta)
    pert_pre_clamp = base[:, None, :] + noise
    clip_fraction = float((pert_pre_clamp.detach().abs() > 1.0).float().mean().item())
    pert = pert_pre_clamp.clamp(-1.0, 1.0)
    return torch.cat([base[:, None, :], pert], dim=1), clip_fraction


def generate_candidate_actions(
    actor_mu: torch.Tensor, cfg: ERDConfig, tol_mean: float
) -> torch.Tensor:
    cand, _clip_fraction = _generate_candidate_actions_with_clip_fraction(
        actor_mu, cfg, tol_mean
    )
    return cand


def _estimate_distill_feature_grad_norm(
    feat: torch.Tensor, a2c: Any, target_action: torch.Tensor
) -> float:
    try:
        feat_probe = feat.detach().clone().requires_grad_(True)
        mu = a2c.mu(feat_probe)
        mu_act = getattr(a2c, "mu_act", None)
        if mu_act is not None:
            mu = mu_act(mu)
        loss = F.mse_loss(mu, target_action.detach(), reduction="mean")
        (g,) = torch.autograd.grad(loss, feat_probe, retain_graph=False)
        return float(g.detach().norm().item())
    except Exception:
        return 0.0


def compute_erd_loss(bridge: ERDBridge, frames: int = 0) -> dict[str, torch.Tensor]:
    bridge.reset_per_call()
    cfg = bridge.cfg
    if not cfg.enabled or bridge.scorer is None:
        return {}
    cache = bridge.cache
    if cache is None or cache.actions_flat is None:
        return {}
    if not bridge.capture.fired or bridge.capture.tensor is None:
        return {}

    start, end = _read_minibatch_range(bridge)
    feat = bridge.capture.tensor
    if feat.dim() != 2 or feat.shape[0] != (end - start):
        raise RuntimeError(
            f"[ERD] captured feature shape {tuple(feat.shape)} does not match "
            f"minibatch range ({start},{end})"
        )
    device, dtype = feat.device, feat.dtype
    actions = cache.actions_flat[start:end].to(device=device, dtype=dtype)
    if actions.shape[-1] != cfg.action_dim:
        raise RuntimeError(
            f"[ERD] action_dim mismatch cache={actions.shape[-1]} cfg={cfg.action_dim}"
        )
    targets = {
        k: v[start:end].to(device=device, dtype=dtype)
        for k, v in cache.targets_flat.items()
    }
    masks = {
        k: v[start:end].to(device=device, dtype=dtype)
        for k, v in cache.masks_flat.items()
    }

    logits = bridge.scorer(feat.detach(), actions.detach())
    scorer_total = feat.new_zeros(())
    scorer_terms = 0
    for key, logit in logits.items():
        if key not in targets:
            continue
        loss, n = _masked_bce(logit, targets[key], masks[key])
        if n > 0:
            scorer_total = scorer_total + loss
            scorer_terms += 1
    if scorer_terms > 0:
        scorer_total = scorer_total / float(scorer_terms)
        bridge.scorer_warmup_iters_done += 1
        bridge.last_scorer_active = 1.0
        bridge.last_scorer_loss = float(scorer_total.detach().item())
    else:
        scorer_total = feat.new_zeros(())

    bridge.last_target_valid_fraction = cache.mean_valid_fraction()
    bridge.last_pos_rate = dict(cache.pos_rate)

    lam = cfg.distill_lambda_for_tol(cache.tol_mean)
    distill_total = feat.new_zeros(())
    block_reason = BLOCK_OK
    if lam <= 0.0:
        block_reason = BLOCK_TOL_ABOVE_START
    elif int(frames) < int(cfg.start_after_frames):
        block_reason = BLOCK_WARMUP_FRAMES
    elif bridge.scorer_warmup_iters_done < int(cfg.scorer_warmup_iters):
        block_reason = BLOCK_WARMUP_ITERS
    elif bridge.last_target_valid_fraction < float(cfg.min_target_valid_fraction):
        block_reason = BLOCK_LOW_VALID_FRACTION

    if block_reason == BLOCK_OK:
        d_hat = cache.d_hat_flat[start:end].to(device=device, dtype=dtype)
        tol = cache.success_tol_flat[start:end].to(device=device, dtype=dtype)
        lifted = (
            cache.lifted_flat[start:end].to(device=device, dtype=dtype)
            if cache.lifted_flat is not None
            else None
        )
        risky = _build_risky_mask(cfg, d_hat=d_hat, tol=tol, lifted=lifted)
        bridge.last_risky_fraction = float(risky.float().mean().item())
        if bridge.last_risky_fraction < float(cfg.min_risky_fraction) or not risky.any():
            block_reason = BLOCK_NO_RISKY

    if block_reason == BLOCK_OK:
        a2c = _get_a2c_network(bridge.agent_ref)
        mu_live = a2c.mu(feat)
        mu_act = getattr(a2c, "mu_act", None)
        if mu_act is not None:
            mu_live = mu_act(mu_live)
        with torch.no_grad():
            bridge.last_raw_mu_abs_mean = float(mu_live.detach().abs().mean().item())
            bridge.last_raw_mu_clip_fraction = float(
                (mu_live.detach().abs() > 1.0).float().mean().item()
            )
        feat_r = feat[risky]
        mu_r = mu_live[risky]
        cand, candidate_clip_fraction = _generate_candidate_actions_with_clip_fraction(
            mu_r, cfg, cache.tol_mean
        )
        bridge.last_candidate_clip_fraction = float(candidate_clip_fraction)
        B, K, A = cand.shape
        with torch.no_grad():
            feat_rep = feat_r.detach()[:, None, :].expand(B, K, feat_r.shape[-1]).reshape(B * K, -1)
            cand_flat = cand.reshape(B * K, A)
            cand_logits = bridge.scorer(feat_rep, cand_flat)
            cand_score, cand_succ, cand_impr, cand_drop = _combined_score_from_logits(
                cfg, cand_logits
            )
            cand_score = cand_score.view(B, K)
            cand_succ = cand_succ.view(B, K)
            cand_impr = cand_impr.view(B, K)
            cand_drop = cand_drop.view(B, K)
            best_idx = cand_score.argmax(dim=1)
            rows = torch.arange(B, device=device)
            best_action = cand[rows, best_idx].detach()
            best_score = cand_score[rows, best_idx]
            actor_score = cand_score[:, 0]
            margin = best_score - actor_score
            best_drop = cand_drop[rows, best_idx]
            eligible = (margin > float(cfg.min_score_margin)) & (
                best_drop <= float(cfg.max_drop_prob_for_teacher)
            )
            bridge.last_best_score = float(best_score.mean().item())
            bridge.last_actor_score = float(actor_score.mean().item())
            bridge.last_score_margin = float(margin.mean().item())
            bridge.last_best_success_prob = float(cand_succ[rows, best_idx].mean().item())
            bridge.last_best_improve_prob = float(cand_impr[rows, best_idx].mean().item())
            bridge.last_best_drop_prob = float(best_drop.mean().item())
            bridge.last_actor_success_prob = float(cand_succ[:, 0].mean().item())
            bridge.last_actor_drop_prob = float(cand_drop[:, 0].mean().item())
        if not eligible.any():
            block_reason = (
                BLOCK_DROP_RISK_HIGH
                if (best_drop <= float(cfg.max_drop_prob_for_teacher)).float().mean().item() < 0.5
                else BLOCK_LOW_MARGIN
            )
        else:
            mu_e = mu_r[eligible]
            best_e = best_action[eligible]
            bridge.last_action_delta_norm = float(
                (best_e - mu_e.detach().clamp(-1.0, 1.0)).detach().norm(dim=-1).mean().item()
            )
            raw_distill = F.mse_loss(mu_e, best_e.detach(), reduction="mean")
            fc_norm = _estimate_distill_feature_grad_norm(feat_r[eligible], a2c, best_e)
            ppo_norm = float(bridge.ppo_grad_ema)
            ratio = fc_norm / ppo_norm if ppo_norm > 1e-12 else 0.0
            bridge.last_distill_grad_ratio = float(ratio)
            auto_scale = 1.0
            if (
                cfg.auto_scale_by_grad_ratio
                and ratio > float(cfg.max_grad_ratio)
                and ratio > 0.0
            ):
                auto_scale = float(cfg.max_grad_ratio) / ratio
            bridge.last_auto_scale_factor = float(auto_scale)
            distill_total = float(lam) * float(auto_scale) * raw_distill
            bridge.last_distill_active = 1.0
            bridge.last_distill_loss = float(raw_distill.detach().item())

    bridge.last_distill_block_reason = int(block_reason)
    scaled_scorer = float(cfg.scorer_loss_weight) * scorer_total
    total = scaled_scorer + distill_total
    bridge.last_total_loss = float(total.detach().item())
    bridge.last_active = 1.0 if (bridge.last_scorer_active > 0 or bridge.last_distill_active > 0) else 0.0
    if total.detach().abs().item() <= 0.0:
        return {}
    return {"erd_total": total}


def compute_erd_grad_audit(
    *,
    rl_games_model: nn.Module,
    scorer: ERDActionScorer,
    cfg: ERDConfig,
    obs: torch.Tensor,
    action: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> dict[str, float]:
    """Test helper for gradient routing guarantees."""
    inner = rl_games_model.model if hasattr(rl_games_model, "model") else rl_games_model
    a2c = getattr(inner, "a2c_network", None)
    if a2c is None:
        raise RuntimeError("[ERD] grad audit requires a2c_network")
    for p in list(rl_games_model.parameters()) + list(scorer.parameters()):
        if p.grad is not None:
            p.grad = None
    feat = a2c.actor_mlp(obs)
    logits = scorer(feat.detach(), action.detach())
    key = f"success_h{cfg.horizons[0]}"
    scorer_loss, _ = _masked_bce(logits[key], target, mask)
    scorer_loss.backward()

    def norm(params):
        s = 0.0
        for p in params:
            if p.grad is not None:
                s += float(p.grad.detach().norm().item() ** 2)
        return s ** 0.5

    scorer_only = {
        "scorer_grad": norm(list(scorer.parameters())),
        "mu_grad": norm(list(getattr(a2c, "mu").parameters())),
        "actor_mlp_grad": norm(list(getattr(a2c, "actor_mlp").parameters())),
    }
    for p in list(rl_games_model.parameters()) + list(scorer.parameters()):
        if p.grad is not None:
            p.grad = None
    feat2 = a2c.actor_mlp(obs)
    mu = a2c.mu(feat2)
    best_action = action.detach().clamp(-1.0, 1.0)
    distill = F.mse_loss(mu, best_action)
    distill.backward()
    return {
        **{f"scorer_path_{k}": v for k, v in scorer_only.items()},
        "distill_path_scorer_grad": norm(list(scorer.parameters())),
        "distill_path_mu_grad": norm(list(a2c.mu.parameters())),
        "distill_path_actor_mlp_grad": norm(list(a2c.actor_mlp.parameters())),
        "best_action_requires_grad": 1.0 if best_action.requires_grad else 0.0,
    }

