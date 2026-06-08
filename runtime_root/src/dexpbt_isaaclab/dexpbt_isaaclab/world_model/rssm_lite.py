"""
RSSM-lite world model (Dreamer-style, but auxiliary-only).

Architecture:
    obs_encoder:    MLP(obs_dim -> 256 -> 256)
    action_encoder: MLP(action_dim -> 128)
    GRUCell:        input = concat(z_{t-1}[64], a_emb_{t-1}[128]) = 192,
                    hidden = 256
    prior:     p(z_t | h_t)             from MLP(h_t -> 2 * 64)
    posterior: q(z_t | h_t, obs_emb_t)  from MLP(concat(h_t, obs_emb_t) -> 2 * 64)
    belief b_t = concat(h_t, z_t)        # 320

Per the user's revision: GRU input does NOT include obs_embed_t. Posterior
sees obs_embed; prior does not.

Hidden reset: at any (t, b) where done_{t-1} == 1, h_t and z_{t-1} are
zeroed before the GRU step. This matches SequenceSampler's segment_id /
valid_mask convention.

Loss terms (computed in compute_aux_loss):
    L_state, L_reward, L_done, L_success?, L_task_error?, L_value?, L_kl

Masking convention (per plan §7):
    state_delta and task_error_delta loss use eff = valid_mask * (1 - done).
    reward / done / success / value losses use eff_no_done = valid_mask
    (we still want to learn the done step itself, just not the cross-step
    next-state target).
    KL is averaged over (T, B) with valid_mask weighting (drop steps whose
    posterior never saw a valid prefix).

Free-nats KL:
    L_kl = max(KL_per_step - free_nats, 0).mean(masked)

beta_kl warmup is a property of the trainer, not the module; the module
returns raw L_kl in the loss dict and the trainer multiplies.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .seq_wm_base import SequenceBatch, SequenceWorldModelBase
from .seq_wm_heads import PredictionHeads


def _mlp(in_dim: int, hidden: int, out_dim: int, n_hidden: int = 1) -> nn.Sequential:
    layers: list[nn.Module] = []
    prev = in_dim
    for _ in range(n_hidden):
        layers.append(nn.Linear(prev, hidden))
        layers.append(nn.ELU())
        prev = hidden
    layers.append(nn.Linear(prev, out_dim))
    return nn.Sequential(*layers)


def _masked_mean(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Element-wise mean of ``value`` weighted by ``mask``.

    ``mask`` may have fewer trailing dimensions than ``value`` — those are
    broadcast and counted in the denominator (so MSE on a (N, K) vector
    target with a (N,) mask is scaled by N*K, NOT N). Robust to all-zero
    masks: returns 0.
    """
    if mask.ndim < value.ndim:
        for _ in range(value.ndim - mask.ndim):
            mask = mask.unsqueeze(-1)
    expanded = mask.expand_as(value)
    denom = expanded.sum().clamp_min(torch.tensor(1.0, device=value.device))
    return (value * expanded).sum() / denom


def _kl_diag_gauss(mu_q: torch.Tensor, std_q: torch.Tensor,
                   mu_p: torch.Tensor, std_p: torch.Tensor) -> torch.Tensor:
    """KL(q || p) for diagonal Gaussians, summed over the last dim."""
    var_q = std_q ** 2
    var_p = std_p ** 2
    log_ratio = torch.log(std_p) - torch.log(std_q)
    return (log_ratio + (var_q + (mu_q - mu_p) ** 2) / (2.0 * var_p) - 0.5).sum(dim=-1)


class RSSMLiteWorldModel(SequenceWorldModelBase):
    """Standard RSSM-lite where GRU transition uses (z_{t-1}, a_{t-1}).

    Args:
        obs_dim: env observation dim.
        action_dim: env action dim.
        obs_embed_dim: obs encoder output dim, default 256.
        action_embed_dim: action encoder output dim, default 128.
        h_dim: deterministic hidden state dim (GRU hidden), default 256.
        z_dim: stochastic latent dim, default 64.
        head_hidden: MLP hidden dim for prediction heads, default 256.
        with_value_head: include V_seq(b_t) head (for *_belief_value).
        with_success_head: include success BCE head.
        with_task_error_head: include d_hat_delta head.
        free_nats: floor on per-step KL before beta scaling, default 1.0.
        std_floor: minimum std for prior/posterior to avoid collapse, 0.1.
    """

    def __init__(
        self,
        *,
        obs_dim: int,
        action_dim: int,
        obs_embed_dim: int = 256,
        action_embed_dim: int = 128,
        h_dim: int = 256,
        z_dim: int = 64,
        head_hidden: int = 256,
        with_value_head: bool = True,
        with_success_head: bool = True,
        with_task_error_head: bool = True,
        free_nats: float = 1.0,
        std_floor: float = 0.1,
    ) -> None:
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.obs_embed_dim = int(obs_embed_dim)
        self.action_embed_dim = int(action_embed_dim)
        self.h_dim = int(h_dim)
        self.z_dim = int(z_dim)
        self.belief_dim = self.h_dim + self.z_dim
        self.free_nats = float(free_nats)
        self.std_floor = float(std_floor)

        # Encoders.
        self.obs_encoder = _mlp(self.obs_dim, obs_embed_dim, obs_embed_dim, n_hidden=1)
        self.action_encoder = _mlp(self.action_dim, action_embed_dim, action_embed_dim, n_hidden=0)

        # Transition: GRU(input = z_prev[Z] + a_emb_prev[Aemb])
        self.gru = nn.GRUCell(input_size=self.z_dim + action_embed_dim, hidden_size=self.h_dim)

        # Prior and posterior produce 2*z_dim params (mu, raw_std).
        self.prior_net = _mlp(self.h_dim, h_dim, 2 * self.z_dim, n_hidden=1)
        self.posterior_net = _mlp(self.h_dim + obs_embed_dim, h_dim, 2 * self.z_dim, n_hidden=1)

        # Shared prediction heads.
        self.heads = PredictionHeads(
            belief_dim=self.belief_dim,
            action_dim=self.action_dim,
            obs_dim=self.obs_dim,
            hidden_dim=head_hidden,
            n_hidden=2,
            with_value_head=with_value_head,
            with_success_head=with_success_head,
            with_task_error_head=with_task_error_head,
        )

        # Tiny per-step running stats for telemetry.
        self._last_metrics: dict[str, float] = {}

    # ---------- core blocks ----------

    def _split_dist(self, params: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mu, raw_std = torch.chunk(params, 2, dim=-1)
        std = F.softplus(raw_std) + self.std_floor
        return mu, std

    def _reparam(self, mu: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
        eps = torch.randn_like(std)
        return mu + std * eps

    # ---------- SequenceWorldModelBase API ----------

    def encode_sequence(
        self,
        batch: SequenceBatch,
        *,
        deterministic: bool = False,
    ) -> torch.Tensor:
        """Forward the RSSM over a sequence batch and return ``b_t``.

        ``deterministic=True`` substitutes ``z_t`` with the posterior mean
        ``mu_q`` (no reparameterisation noise). Used by ``encode_belief``
        so the teacher signal is stable across PPO minibatches; training
        keeps the default stochastic path so the KL loss still has a
        meaningful gradient.
        """
        T, B = batch.T, batch.B
        device = batch.device
        obs_emb_seq = self.obs_encoder(batch.obs.reshape(T * B, -1)).reshape(T, B, -1)
        act_emb_seq = self.action_encoder(batch.action.reshape(T * B, -1)).reshape(T, B, -1)

        h = torch.zeros(B, self.h_dim, device=device, dtype=batch.obs.dtype)
        z = torch.zeros(B, self.z_dim, device=device, dtype=batch.obs.dtype)
        beliefs: list[torch.Tensor] = []
        prior_mus: list[torch.Tensor] = []
        prior_stds: list[torch.Tensor] = []
        post_mus: list[torch.Tensor] = []
        post_stds: list[torch.Tensor] = []

        for t in range(T):
            # Hidden reset on segment boundary: if done_{t-1} == 1, zero h, z.
            if t > 0:
                reset = batch.done[t - 1].to(h.dtype).unsqueeze(-1)  # (B, 1) in [0,1]
                h = h * (1.0 - reset)
                z = z * (1.0 - reset)
                # Transition: h_t = GRU(concat(z_{t-1}, a_emb_{t-1}), h_{t-1}).
                gru_in = torch.cat([z, act_emb_seq[t - 1]], dim=-1)
                h = self.gru(gru_in, h)
            # Prior and posterior at h_t.
            prior_mu, prior_std = self._split_dist(self.prior_net(h))
            post_mu, post_std = self._split_dist(
                self.posterior_net(torch.cat([h, obs_emb_seq[t]], dim=-1))
            )
            if deterministic:
                z = post_mu
            else:
                z = self._reparam(post_mu, post_std)
            belief = torch.cat([h, z], dim=-1)
            beliefs.append(belief)
            prior_mus.append(prior_mu)
            prior_stds.append(prior_std)
            post_mus.append(post_mu)
            post_stds.append(post_std)

        belief_seq = torch.stack(beliefs, dim=0)
        # Stash distribution params for compute_aux_loss / report_metrics.
        self._last_prior_mu = torch.stack(prior_mus, dim=0)
        self._last_prior_std = torch.stack(prior_stds, dim=0)
        self._last_post_mu = torch.stack(post_mus, dim=0)
        self._last_post_std = torch.stack(post_stds, dim=0)
        return belief_seq

    @torch.no_grad()
    def encode_belief(
        self, batch: SequenceBatch
    ) -> dict[str, Optional[torch.Tensor]]:
        """Deterministic belief + detached aux head outputs for distillation.

        Overrides the base implementation so the teacher uses the posterior
        mean (no reparameterisation noise). Heads are evaluated under
        ``no_grad`` and their outputs are detached.
        """
        was_training = self.training
        self.eval()
        try:
            belief = self.encode_sequence(batch, deterministic=True)
        finally:
            if was_training:
                self.train()
        belief = belief.detach()
        T, B = belief.shape[0], belief.shape[1]
        belief_flat = belief.reshape(T * B, -1)
        value = None
        success_logit = None
        if self.heads.has_value_head:
            v = self.heads.predict_value(belief_flat)
            if v is not None:
                value = v.reshape(T, B).detach()
        if self.heads.success_head is not None:
            action_flat = batch.action.reshape(T * B, -1)
            ba = torch.cat([belief_flat, action_flat], dim=-1)
            success_logit = self.heads.success_head(ba).squeeze(-1).reshape(T, B).detach()
        return {"belief": belief, "value": value, "success_logit": success_logit}

    def compute_aux_loss(
        self,
        batch: SequenceBatch,
        *,
        loss_weights: Optional[dict[str, float]] = None,
        beta_kl: float = 0.1,
        value_detach_belief: bool = False,
    ) -> dict[str, torch.Tensor]:
        """Forward + losses. Returns dict with 'loss_total' and per-term keys."""
        if loss_weights is None:
            loss_weights = {
                "state": 1.0,
                "reward": 1.0,
                "done": 0.2,
                "success": 1.0,
                "task_error": 1.0,
                "value": 0.1,
            }
        T, B = batch.T, batch.B
        belief_seq = self.encode_sequence(batch)                # (T, B, belief_dim)
        belief_flat = belief_seq.reshape(T * B, -1)
        action_flat = batch.action.reshape(T * B, -1)

        pred = self.heads.predict_transition(belief_flat, action_flat)

        # Effective masks (broadcast lazily).
        valid = batch.valid_mask                                # (T, B) float
        not_done = (1.0 - batch.done)                           # (T, B)
        eff_state = (valid * not_done)                          # (T, B)
        eff_basic = valid                                       # (T, B)

        # ---- L_state: predict (next_obs - obs) ----
        obs_delta_target = (batch.next_obs - batch.obs).reshape(T * B, -1)
        L_state = _masked_mean(
            (pred["next_obs_delta"] - obs_delta_target) ** 2,
            eff_state.reshape(T * B),
        )

        # ---- L_reward ----
        L_reward = _masked_mean(
            (pred["reward"].squeeze(-1) - batch.reward.reshape(T * B)) ** 2,
            eff_basic.reshape(T * B),
        )

        # ---- L_done (BCE with logits) ----
        done_target = batch.done.reshape(T * B)
        L_done = F.binary_cross_entropy_with_logits(
            pred["done_logits"].squeeze(-1),
            done_target,
            weight=eff_basic.reshape(T * B),
            reduction="sum",
        ) / eff_basic.sum().clamp_min(torch.tensor(1.0, device=belief_seq.device))

        # ---- L_success (optional) ----
        L_success = belief_seq.new_zeros(())
        has_success = (
            "success_logits" in pred
            and batch.success_count is not None
            and batch.next_success_count is not None
        )
        if has_success:
            success_target = (batch.next_success_count - batch.success_count > 0.5).to(
                belief_seq.dtype
            ).reshape(T * B)
            L_success = F.binary_cross_entropy_with_logits(
                pred["success_logits"].squeeze(-1),
                success_target,
                weight=eff_basic.reshape(T * B),
                reduction="sum",
            ) / eff_basic.sum().clamp_min(torch.tensor(1.0, device=belief_seq.device))

        # ---- L_task_error (optional) ----
        L_task_error = belief_seq.new_zeros(())
        has_task = (
            "task_error_delta" in pred
            and batch.d_hat is not None
            and batch.next_d_hat is not None
        )
        if has_task:
            tg = (batch.next_d_hat - batch.d_hat).reshape(T * B)
            L_task_error = _masked_mean(
                (pred["task_error_delta"].squeeze(-1) - tg) ** 2,
                eff_state.reshape(T * B),
            )

        # ---- L_value (optional) ----
        L_value = belief_seq.new_zeros(())
        if self.heads.has_value_head and batch.returns is not None:
            v_in = belief_flat.detach() if value_detach_belief else belief_flat
            v_pred = self.heads.predict_value(v_in).squeeze(-1)  # (T*B,)
            L_value = _masked_mean(
                (v_pred - batch.returns.reshape(T * B)) ** 2,
                eff_basic.reshape(T * B),
            )

        # ---- L_kl with free-nats ----
        kl_per_step = _kl_diag_gauss(
            self._last_post_mu, self._last_post_std,
            self._last_prior_mu, self._last_prior_std,
        )                                                       # (T, B)
        kl_clamped = (kl_per_step - self.free_nats).clamp_min(0.0)
        L_kl = _masked_mean(kl_clamped, eff_basic)

        L_total = (
            loss_weights["state"] * L_state
            + loss_weights["reward"] * L_reward
            + loss_weights["done"] * L_done
            + loss_weights.get("success", 0.0) * L_success
            + loss_weights.get("task_error", 0.0) * L_task_error
            + beta_kl * L_kl
            + loss_weights.get("value", 0.0) * L_value
        )

        # Cache scalar metrics for telemetry.
        with torch.no_grad():
            self._last_metrics = {
                "loss_total": float(L_total.detach().item()),
                "loss_state": float(L_state.detach().item()),
                "loss_reward": float(L_reward.detach().item()),
                "loss_done": float(L_done.detach().item()),
                "loss_success": float(L_success.detach().item()),
                "loss_task_error": float(L_task_error.detach().item()),
                "loss_value": float(L_value.detach().item()),
                "loss_kl": float(L_kl.detach().item()),
                "beta_kl": float(beta_kl),
                "free_nats": float(self.free_nats),
                "prior_std_mean": float(self._last_prior_std.mean().item()),
                "posterior_std_mean": float(self._last_post_std.mean().item()),
                "kl_raw_mean": float(kl_per_step.mean().item()),
            }

        return {
            "loss_total": L_total,
            "loss_state": L_state,
            "loss_reward": L_reward,
            "loss_done": L_done,
            "loss_success": L_success,
            "loss_task_error": L_task_error,
            "loss_value": L_value,
            "loss_kl": L_kl,
        }

    def report_metrics(self) -> dict[str, float]:
        return dict(self._last_metrics)


__all__ = ["RSSMLiteWorldModel"]
