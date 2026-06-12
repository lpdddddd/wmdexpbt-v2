"""Gated Recovery Residual Head for EIR-MPC.

Instead of pulling the main actor ``mu`` toward a clipped teacher action (the ERD
failure mode that became action-bound regularization), EIR learns a small gated
residual on top of a detached base action:

    gate  = sigmoid(GateHead(feature))
    delta = tanh(ResidualHead(feature)) * residual_scale
    a_rec = clip(mu_base + alpha * gate * delta, -1, 1)

The residual is trained to match the selected elite / model-ranked best action.
``mu_base`` is detached by default so the main actor receives no direct residual
gradient (constraint 19.1). At rollout time the residual is applied ONLY in
floor / hard-floor near-success states, behind a kill switch + warmup, and it
self-disables (logging a fallback reason) when the required per-step aux
(``d_hat`` / ``success_tol`` / ``lifted``) is unavailable (constraint 19.3).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# rollout-apply fallback reasons
APPLY_OK = 0
APPLY_AUX_MISSING = 1
APPLY_AUX_STALE = 2
APPLY_BEFORE_WARMUP = 3
APPLY_KILL_SWITCH_OFF = 4
APPLY_GATE_CLOSED = 5
APPLY_NOT_FLOOR_STAGE = 6


@dataclass
class RecoveryResidualConfig:
    enabled: bool = False
    feature_dim: int = 256
    hidden_dim: int = 256
    action_dim: int = 23
    residual_scale: float = 0.5
    mu_base_detach: bool = True

    # distill weights by stage (start/floor/hard)
    distill_weight: float = 0.05
    floor_distill_weight: float = 0.08
    hard_floor_distill_weight: float = 0.12
    gate_sparsity_weight: float = 0.001
    delta_l2_weight: float = 0.001

    # rollout apply
    apply_to_rollout: bool = False
    apply_after_frames: int = 300_000_000
    apply_alpha_mid: float = 0.25
    apply_alpha_floor: float = 0.5
    apply_alpha_hard_floor: float = 0.75
    only_lifted: bool = True
    apply_tol_threshold: float = 0.02
    floor_tol: float = 0.02
    hard_floor_tol: float = 0.012
    near_success_mult: float = 2.0
    gate_confidence_threshold: float = 0.5
    # max env-frame age of the aux snapshot before rollout apply self-disables
    apply_max_aux_age_frames: int = 100_000_000

    def distill_weight_for_tol(self, tol: float) -> float:
        t = float(tol)
        if t > float(self.apply_tol_threshold):
            return float(self.distill_weight)
        if t > float(self.hard_floor_tol):
            return float(self.floor_distill_weight)
        return float(self.hard_floor_distill_weight)

    def alpha_for_tol(self, tol: float) -> float:
        t = float(tol)
        if t <= float(self.hard_floor_tol):
            return float(self.apply_alpha_hard_floor)
        if t <= float(self.floor_tol):
            return float(self.apply_alpha_floor)
        return float(self.apply_alpha_mid)


class RecoveryResidualHead(nn.Module):
    def __init__(self, cfg: RecoveryResidualConfig) -> None:
        super().__init__()
        self.cfg = cfg
        f = int(cfg.feature_dim)
        h = int(cfg.hidden_dim)
        a = int(cfg.action_dim)
        self.trunk = nn.Sequential(
            nn.Linear(f, h), nn.Tanh(), nn.Linear(h, h), nn.Tanh()
        )
        self.delta_head = nn.Linear(h, a)
        self.gate_head = nn.Linear(h, 1)

    def forward(self, feat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.trunk(feat)
        delta_raw = self.delta_head(h)
        gate_logit = self.gate_head(h)
        return delta_raw, gate_logit

    def recovery_action(
        self, feat: torch.Tensor, mu_base: torch.Tensor, alpha: float
    ) -> dict[str, torch.Tensor]:
        delta_raw, gate_logit = self.forward(feat)
        gate = torch.sigmoid(gate_logit)
        delta = torch.tanh(delta_raw) * float(self.cfg.residual_scale)
        a_rec = (mu_base + float(alpha) * gate * delta).clamp(-1.0, 1.0)
        return {"a_rec": a_rec, "gate": gate.squeeze(-1), "delta": delta}


@dataclass
class RecoveryResidualBridge:
    cfg: RecoveryResidualConfig
    head: Optional[RecoveryResidualHead] = None
    # latest per-env aux snapshot for rollout apply (refreshed by collector)
    aux_d_hat: Optional[torch.Tensor] = None
    aux_tol: Optional[torch.Tensor] = None
    aux_lifted: Optional[torch.Tensor] = None
    aux_frame: int = -1
    aux_generation: int = -1

    # logged metrics
    last_residual_loss: float = 0.0
    last_gate_mean: float = 0.0
    last_delta_norm: float = 0.0
    last_residual_action_delta_norm: float = 0.0
    last_residual_score_margin: float = 0.0
    last_residual_best_source_frac_elite: float = 0.0
    last_apply_active: float = 0.0
    last_apply_fraction: float = 0.0
    last_apply_alpha: float = 0.0
    last_gate_mean_on_rollout: float = 0.0
    last_apply_fallback_reason: int = APPLY_KILL_SWITCH_OFF

    def set_aux_snapshot(
        self,
        *,
        d_hat: Optional[torch.Tensor],
        tol: Optional[torch.Tensor],
        lifted: Optional[torch.Tensor],
        frame: int,
        generation: int = -1,
    ) -> None:
        self.aux_d_hat = d_hat
        self.aux_tol = tol
        self.aux_lifted = lifted
        self.aux_frame = int(frame)
        self.aux_generation = int(generation)


def attach_recovery_residual(
    *,
    optimizer: torch.optim.Optimizer,
    cfg: RecoveryResidualConfig,
    device: Optional[torch.device | str] = None,
) -> RecoveryResidualHead:
    head = RecoveryResidualHead(cfg)
    if device is not None:
        head = head.to(device)
    base_lr = optimizer.param_groups[0]["lr"] if optimizer.param_groups else 1e-3
    optimizer.add_param_group(
        {"params": list(head.parameters()), "lr": base_lr, "name": "eir_residual"}
    )
    return head


def compute_recovery_residual_loss(
    bridge: RecoveryResidualBridge,
    *,
    feat_e: torch.Tensor,
    mu_live_e: torch.Tensor,
    a_star: torch.Tensor,
    tol_mean: float,
    best_source_frac_elite: float = 0.0,
    score_margin: float = 0.0,
) -> dict[str, torch.Tensor]:
    """Residual distill loss. Updates only the residual head when
    ``cfg.mu_base_detach`` is True (no gradient to the main actor ``mu``)."""
    cfg = bridge.cfg
    head = bridge.head
    if not cfg.enabled or head is None or feat_e.shape[0] == 0:
        bridge.last_residual_loss = 0.0
        return {}

    alpha = cfg.alpha_for_tol(tol_mean)
    if cfg.mu_base_detach:
        mu_base = mu_live_e.detach().clamp(-1.0, 1.0)
    else:
        mu_base = mu_live_e.clamp(-1.0, 1.0)
    out = head.recovery_action(feat_e, mu_base, alpha)
    a_rec = out["a_rec"]
    gate = out["gate"]
    delta = out["delta"]

    lam = cfg.distill_weight_for_tol(tol_mean)
    mse = F.mse_loss(a_rec, a_star.detach(), reduction="mean")
    gate_sparse = gate.mean()
    delta_l2 = (delta ** 2).mean()
    total = lam * mse + float(cfg.gate_sparsity_weight) * gate_sparse + float(cfg.delta_l2_weight) * delta_l2

    with torch.no_grad():
        bridge.last_residual_loss = float(mse.detach().item())
        bridge.last_gate_mean = float(gate.detach().mean().item())
        bridge.last_delta_norm = float(delta.detach().norm(dim=-1).mean().item())
        bridge.last_residual_action_delta_norm = float(
            (a_rec.detach() - mu_base.detach()).norm(dim=-1).mean().item()
        )
        bridge.last_residual_score_margin = float(score_margin)
        bridge.last_residual_best_source_frac_elite = float(best_source_frac_elite)
    return {"eir_residual_total": total}


def compute_rollout_residual_action(
    bridge: RecoveryResidualBridge,
    *,
    feat: torch.Tensor,
    mu: torch.Tensor,
    frames: int,
    rollout_generation: Optional[int] = None,
) -> dict[str, Any]:
    """Compute the (possibly modified) env action mean at rollout time.

    Reads the per-step aux snapshot on the bridge. Self-disables and logs a
    fallback reason when aux is missing / before warmup / kill switch off /
    not a floor near-success state / gate closed.
    """
    cfg = bridge.cfg
    head = bridge.head
    B = int(mu.shape[0])
    dev = mu.device
    no_apply = {
        "action": mu,
        "apply_mask": torch.zeros(B, dtype=torch.bool, device=dev),
        "apply_fraction": 0.0,
        "apply_alpha": 0.0,
        "gate_mean_on_rollout": 0.0,
    }

    def _finish(reason: int, payload: dict[str, Any]) -> dict[str, Any]:
        bridge.last_apply_fallback_reason = int(reason)
        bridge.last_apply_active = 1.0 if payload["apply_fraction"] > 0.0 else 0.0
        bridge.last_apply_fraction = float(payload["apply_fraction"])
        bridge.last_apply_alpha = float(payload["apply_alpha"])
        bridge.last_gate_mean_on_rollout = float(payload["gate_mean_on_rollout"])
        payload["fallback_reason"] = int(reason)
        return payload

    if not cfg.enabled or head is None or not cfg.apply_to_rollout:
        return _finish(APPLY_KILL_SWITCH_OFF, dict(no_apply))
    if int(frames) < int(cfg.apply_after_frames):
        return _finish(APPLY_BEFORE_WARMUP, dict(no_apply))
    d_hat = bridge.aux_d_hat
    tol = bridge.aux_tol
    lifted = bridge.aux_lifted
    if d_hat is None or tol is None or (cfg.only_lifted and lifted is None):
        return _finish(APPLY_AUX_MISSING, dict(no_apply))
    aux_age = int(frames) - int(bridge.aux_frame)
    if int(bridge.aux_frame) < 0 or aux_age > int(cfg.apply_max_aux_age_frames):
        return _finish(APPLY_AUX_STALE, dict(no_apply))
    if (
        rollout_generation is not None
        and int(bridge.aux_generation) >= 0
        and int(rollout_generation) != int(bridge.aux_generation)
    ):
        return _finish(APPLY_AUX_STALE, dict(no_apply))
    d_hat = d_hat.to(dev).reshape(-1)[:B]
    tol = tol.to(dev).reshape(-1)[:B]
    if torch.isnan(d_hat).any() or torch.isnan(tol).any():
        return _finish(APPLY_AUX_MISSING, dict(no_apply))
    lifted_b = (
        lifted.to(dev).reshape(-1)[:B] if lifted is not None else torch.ones(B, device=dev)
    )

    # floor near-success stage gate
    stage_gate = (tol <= float(cfg.apply_tol_threshold)) & (
        d_hat <= (float(cfg.near_success_mult) * tol).clamp_min(1e-6)
    )
    if cfg.only_lifted:
        stage_gate = stage_gate & (lifted_b > 0.5)
    if not bool(stage_gate.any()):
        return _finish(APPLY_NOT_FLOOR_STAGE, dict(no_apply))

    tol_mean = float(tol[stage_gate].mean().item())
    alpha = cfg.alpha_for_tol(tol_mean)
    mu_base = mu.detach().clamp(-1.0, 1.0)
    with torch.no_grad():
        out = head.recovery_action(feat, mu_base, alpha)
    gate = out["gate"]
    gate_gate = gate >= float(cfg.gate_confidence_threshold)
    apply_mask = stage_gate & gate_gate
    if not bool(apply_mask.any()):
        payload = dict(no_apply)
        payload["gate_mean_on_rollout"] = float(gate[stage_gate].mean().item())
        payload["apply_alpha"] = float(alpha)
        return _finish(APPLY_GATE_CLOSED, payload)

    action = mu.clone()
    action[apply_mask] = out["a_rec"][apply_mask]
    payload = {
        "action": action,
        "apply_mask": apply_mask,
        "apply_fraction": float(apply_mask.float().mean().item()),
        "apply_alpha": float(alpha),
        "gate_mean_on_rollout": float(gate[apply_mask].mean().item()),
    }
    return _finish(APPLY_OK, payload)
