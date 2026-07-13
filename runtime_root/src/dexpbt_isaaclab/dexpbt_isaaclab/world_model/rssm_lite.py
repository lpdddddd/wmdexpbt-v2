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
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .seq_wm_base import SequenceBatch, SequenceWorldModelBase
from .seq_wm_heads import PredictionHeads


# Imagine-step modes (for EIR-MPC H-step planner provenance logging).
IMAGINE_PRIOR_ROLLOUT = "prior_rollout"
IMAGINE_STOCHASTIC_PRIOR_ROLLOUT = "stochastic_prior_rollout"
IMAGINE_FIXED_BELIEF_FALLBACK = "fixed_belief_fallback"


@dataclass
class RSSMState:
    """Latent state carried across ``imagine_step`` calls.

    ``belief`` is the concatenation consumed by the prediction heads. For the
    RSSM-lite backbone ``deter`` is the GRU hidden ``h`` and ``stoch`` is the
    stochastic latent ``z`` (``belief == concat(h, z)``). Backbones without an
    explicit deterministic/stochastic split set ``deter``/``stoch`` to ``None``
    and only carry ``belief`` (fixed-belief fallback rollout).
    """

    belief: torch.Tensor
    deter: Optional[torch.Tensor] = None
    stoch: Optional[torch.Tensor] = None


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


def _auto_pos_weight(
    pos_frac: float,
    manual: float,
    balance_enable: bool,
    auto: bool,
    max_w: float,
) -> float:
    if not balance_enable:
        return 1.0
    if not auto:
        return float(manual)
    eps = 1e-6
    return min(max((1 - pos_frac) / max(pos_frac, eps), 1.0), max_w)


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
        # Phase A4 P4 (2026-06-09): task-aware extensions (Part B).
        # All defaults are inert: aux_dim=0 disables the aux encoder and
        # falls back to the legacy posterior input shape; horizon/recovery
        # heads are off until explicitly toggled.
        aux_dim: int = 0,
        aux_embed_dim: int = 64,
        with_horizon_heads: bool = False,
        horizon_steps: tuple[int, ...] = (8, 16),
        with_recovery_head: bool = False,
        recovery_horizon: int = 16,
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

        # Phase A4 P4 (2026-06-09): optional aux encoder (Part B).
        # When aux_dim==0 the encoder is None and the posterior consumes
        # the legacy concat(h, obs_emb) as before; default forward is
        # bit-for-bit unchanged.
        self.aux_dim = int(max(0, aux_dim))
        self.aux_embed_dim = int(aux_embed_dim) if self.aux_dim > 0 else 0
        if self.aux_dim > 0 and self.aux_embed_dim > 0:
            self.aux_encoder = _mlp(
                self.aux_dim, self.aux_embed_dim, self.aux_embed_dim, n_hidden=1
            )
            posterior_in = self.h_dim + obs_embed_dim + self.aux_embed_dim
        else:
            self.aux_encoder = None
            posterior_in = self.h_dim + obs_embed_dim

        # Transition: GRU(input = z_prev[Z] + a_emb_prev[Aemb])
        self.gru = nn.GRUCell(input_size=self.z_dim + action_embed_dim, hidden_size=self.h_dim)

        # Prior and posterior produce 2*z_dim params (mu, raw_std).
        self.prior_net = _mlp(self.h_dim, h_dim, 2 * self.z_dim, n_hidden=1)
        self.posterior_net = _mlp(posterior_in, h_dim, 2 * self.z_dim, n_hidden=1)

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
            with_horizon_heads=bool(with_horizon_heads),
            horizon_steps=tuple(int(x) for x in horizon_steps if int(x) > 0),
            with_recovery_head=bool(with_recovery_head),
            recovery_horizon=int(max(1, recovery_horizon)),
        )

        # Phase A4 P4 (2026-06-09): bookkeeping for new heads.
        self.with_horizon_heads = bool(with_horizon_heads)
        self.horizon_steps = tuple(int(x) for x in horizon_steps if int(x) > 0)
        self.with_recovery_head = bool(with_recovery_head)
        self.recovery_horizon = int(max(1, recovery_horizon))

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

        # Phase A4 P4 (2026-06-09): optional aux scalar embedding (Part B).
        # When aux_encoder is active but aux_scalars are missing/misaligned, use
        # a zero-aux fallback so posterior_net input shape stays consistent.
        aux_emb_seq: Optional[torch.Tensor] = None
        zero_aux_fallback = 0.0
        if self.aux_encoder is not None:
            if (
                batch.aux_scalars is not None
                and batch.aux_scalars.shape[-1] == self.aux_dim
            ):
                aux_in = batch.aux_scalars.reshape(T * B, -1)
            else:
                aux_in = torch.zeros(
                    T * B,
                    self.aux_dim,
                    device=device,
                    dtype=batch.obs.dtype,
                )
                zero_aux_fallback = 1.0
            aux_emb_seq = self.aux_encoder(aux_in).reshape(T, B, -1)
        self._last_metrics["encode_sequence_zero_aux_fallback"] = zero_aux_fallback

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
            if aux_emb_seq is None:
                post_input = torch.cat([h, obs_emb_seq[t]], dim=-1)
            else:
                post_input = torch.cat(
                    [h, obs_emb_seq[t], aux_emb_seq[t]], dim=-1
                )
            post_mu, post_std = self._split_dist(self.posterior_net(post_input))
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
        # Phase A4 P4 (2026-06-09): floor-aware task-loss weighting (Part B).
        # All defaults disable the feature so legacy callers see the same
        # math. When ``floor_aware_weighting=True`` we look at
        # batch.success_tolerance_seq (per-step tol) and multiply
        # success/task_error/value/horizon/drop/recovery losses by:
        #   * 1.0 above floor_tol_threshold
        #   * floor_weight_mid in (hard_floor_tol, floor_tol]
        #   * floor_weight_floor at or below hard_floor_tol
        # state and KL losses are NEVER multiplied (per spec).
        floor_aware_weighting: bool = False,
        floor_tol_threshold: float = 0.02,
        floor_tol_floor: float = 0.012,
        floor_weight_mid: float = 2.0,
        floor_weight_floor: float = 3.0,
        # Phase A4 P4 (2026-06-09): recovery soft-OR target (Part B / Item 6).
        recovery_d_hat_margin: float = 0.005,
        recovery_d_hat_tol_min_steps: int = 0,
        # Phase A4 P4 wired (2026-06-10): widen recovery target mask so
        # `recovery_pos_rate` is non-zero before the curriculum reaches
        # floor (≤floor_tol_threshold). The loss is still emphasised on
        # floor via ``floor_mult``; this only controls which steps are
        # included in target/mask computation. Defaults to
        # ``floor_tol_threshold`` for backwards compatibility.
        recovery_shadow_tol_threshold: Optional[float] = None,
        # Skip horizon heads whose horizon ≥ seq_len-1 (target undefined).
        horizon_skip_too_long: bool = True,
        # WM Head Rescue: balanced BCE pos_weight (default off).
        head_balance_enable: bool = False,
        success_pos_weight: float = 1.0,
        drop_pos_weight: float = 1.0,
        recovery_pos_weight: float = 1.0,
        head_pos_weight_auto: bool = False,
        head_pos_weight_max: float = 10.0,
        # Phase A4 P5 (2026-06-10): when True, also returns the float-metric
        # dict that previously only lived on ``self._last_metrics``. Used
        # by TeacherManager.evaluate to read per-horizon pos rates without
        # touching internal state. Default False keeps the legacy contract
        # (tensor-only dict).
        return_metrics: bool = False,
    ) -> "dict[str, torch.Tensor] | tuple[dict[str, torch.Tensor], dict[str, float]]":
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

        # ---- Phase A4 P4 (2026-06-09 / 2026-06-10 wired): horizon heads + recovery head (Part B) ----
        # All optional; when the heads do not exist or the targets cannot
        # be built, the corresponding loss term stays at zero and the
        # caller-side lambda sets the multiplier to 0.0 by default.
        L_horizon_success: dict[int, torch.Tensor] = {}
        L_drop_risk: dict[int, torch.Tensor] = {}
        L_recovery = belief_seq.new_zeros(())
        recovery_pos_rate = 0.0
        recovery_floor_sample_fraction = 0.0
        recovery_missing_target_count = 0.0
        horizon_skipped: dict[int, int] = {}
        horizon_pos_rate: dict[int, float] = {}
        drop_pos_rate: dict[int, float] = {}
        success_label_pos_frac = 0.0
        drop_label_pos_frac = 0.0
        recovery_label_pos_frac = 0.0
        success_logit_mean = 0.0
        success_logit_std = 0.0
        success_logit_abs_mean = 0.0
        drop_logit_mean = 0.0
        drop_logit_std = 0.0
        drop_logit_abs_mean = 0.0
        recovery_logit_mean = 0.0
        recovery_logit_std = 0.0
        recovery_logit_abs_mean = 0.0
        # Reuse pred for horizon/recovery. The heads, when present, are
        # called inside heads.predict_horizon / heads.predict_recovery.
        if self.with_horizon_heads and "horizon_logits" in pred:
            for h_steps, head_out in pred["horizon_logits"].items():
                # head_out is dict with 'success' and 'drop' logits, each
                # shape (T*B, 1). Targets are built from in-window cumulative
                # success_count / failure flags, masked by valid prefix and
                # by no-done-cross within the horizon.
                horizon_window = max(1, int(h_steps))
                # Phase A4 P4 wired (2026-06-10): horizon target requires
                # at least horizon_window+1 valid steps after t. With
                # T==seq_len we need t+horizon_window < T → horizon must
                # be ≤ seq_len-2. Skip degenerate horizons rather than
                # producing zero loss that masquerades as learned.
                if horizon_skip_too_long and horizon_window > T - 2:
                    horizon_skipped[h_steps] = 1
                    continue
                if (
                    batch.success_count is None
                    or batch.next_success_count is None
                ):
                    horizon_skipped[h_steps] = 2  # missing success_count
                    continue
                target_success = self._build_horizon_success_target(
                    batch, horizon_window
                )
                horizon_mask = self._build_horizon_valid_mask(
                    batch, horizon_window
                )
                logit_s = head_out["success"].squeeze(-1)
                logit_d = head_out["drop"].squeeze(-1)
                target_drop = self._build_horizon_drop_target(
                    batch, horizon_window
                )
                # Use binary_cross_entropy_with_logits with sample weights.
                w = horizon_mask.reshape(T * B)
                denom = w.sum().clamp_min(torch.tensor(1.0, device=belief_seq.device))
                with torch.no_grad():
                    if w.sum() > 0:
                        flat_w = w.reshape(T * B)
                        tgt_s = target_success.reshape(T * B)
                        tgt_d = target_drop.reshape(T * B)
                        h_pos = float((tgt_s * flat_w).sum().item() / float(denom.item()))
                        d_pos = float((tgt_d * flat_w).sum().item() / float(denom.item()))
                        horizon_pos_rate[h_steps] = h_pos
                        drop_pos_rate[h_steps] = d_pos
                        success_label_pos_frac = h_pos
                        drop_label_pos_frac = d_pos
                        masked_s = logit_s[flat_w > 0]
                        masked_d = logit_d[flat_w > 0]
                        if masked_s.numel() > 0:
                            success_logit_mean = float(masked_s.mean().item())
                            success_logit_std = float(masked_s.std(unbiased=False).item())
                            success_logit_abs_mean = float(masked_s.abs().mean().item())
                        if masked_d.numel() > 0:
                            drop_logit_mean = float(masked_d.mean().item())
                            drop_logit_std = float(masked_d.std(unbiased=False).item())
                            drop_logit_abs_mean = float(masked_d.abs().mean().item())
                pw_s = _auto_pos_weight(
                    success_label_pos_frac, float(success_pos_weight),
                    bool(head_balance_enable), bool(head_pos_weight_auto),
                    float(head_pos_weight_max),
                )
                pw_d = _auto_pos_weight(
                    drop_label_pos_frac, float(drop_pos_weight),
                    bool(head_balance_enable), bool(head_pos_weight_auto),
                    float(head_pos_weight_max),
                )
                pos_w_s = belief_seq.new_tensor(pw_s)
                pos_w_d = belief_seq.new_tensor(pw_d)
                L_horizon_success[h_steps] = (
                    F.binary_cross_entropy_with_logits(
                        logit_s,
                        target_success.reshape(T * B),
                        weight=w,
                        pos_weight=pos_w_s,
                        reduction="sum",
                    )
                    / denom
                )
                L_drop_risk[h_steps] = (
                    F.binary_cross_entropy_with_logits(
                        logit_d,
                        target_drop.reshape(T * B),
                        weight=w,
                        pos_weight=pos_w_d,
                        reduction="sum",
                    )
                    / denom
                )

        if self.with_recovery_head and "recovery_logits" in pred:
            recovery_logit = pred["recovery_logits"].squeeze(-1)
            # Phase A4 P4 wired (2026-06-10): recovery_shadow_tol_threshold
            # widens the mask used for target/pos_rate so we get learning
            # signal before tolerance reaches the strict floor. Defaults to
            # the strict threshold for backward compatibility.
            shadow_tol = (
                float(floor_tol_threshold)
                if recovery_shadow_tol_threshold is None
                else float(recovery_shadow_tol_threshold)
            )
            recovery_target, recovery_mask = self._build_recovery_target_soft_or(
                batch,
                horizon=self.recovery_horizon,
                d_hat_margin=float(recovery_d_hat_margin),
                d_hat_tol_min_steps=int(recovery_d_hat_tol_min_steps),
                floor_tol_threshold=shadow_tol,
            )
            w = recovery_mask.reshape(T * B)
            denom = w.sum().clamp_min(torch.tensor(1.0, device=belief_seq.device))
            with torch.no_grad():
                # `recovery_floor_sample_fraction` reports how many
                # (T,B) entries the shadow mask actually covers. A value
                # of 0 means no tight-tolerance signal in this batch and
                # downstream consumers should treat the head as
                # uncalibrated for the current rollout.
                recovery_floor_sample_fraction = float(
                    w.sum().item() / max(1.0, float(T * B))
                )
                if w.sum() > 0:
                    recovery_pos_rate = float(
                        (recovery_target.reshape(T * B) * w).sum().item()
                        / float(denom.item())
                    )
                    recovery_label_pos_frac = recovery_pos_rate
                    masked_r = recovery_logit[w > 0]
                    if masked_r.numel() > 0:
                        recovery_logit_mean = float(masked_r.mean().item())
                        recovery_logit_std = float(masked_r.std(unbiased=False).item())
                        recovery_logit_abs_mean = float(masked_r.abs().mean().item())
                else:
                    recovery_missing_target_count = 1.0
            pw_r = _auto_pos_weight(
                recovery_label_pos_frac, float(recovery_pos_weight),
                bool(head_balance_enable), bool(head_pos_weight_auto),
                float(head_pos_weight_max),
            )
            pos_w_r = belief_seq.new_tensor(pw_r)
            L_recovery = (
                F.binary_cross_entropy_with_logits(
                    recovery_logit,
                    recovery_target.reshape(T * B),
                    weight=w,
                    pos_weight=pos_w_r,
                    reduction="sum",
                )
                / denom
            )

        # Phase A4 P4 (2026-06-09): floor-aware loss multiplier (Part B).
        floor_mult = belief_seq.new_ones(())
        if (
            floor_aware_weighting
            and batch.success_tolerance_seq is not None
            and batch.success_tolerance_seq.numel() > 0
        ):
            with torch.no_grad():
                tol_seq = batch.success_tolerance_seq.float()
                tol_mean = float(
                    (tol_seq * batch.valid_mask).sum().item()
                    / max(float(batch.valid_mask.sum().item()), 1.0)
                )
            if tol_mean <= float(floor_tol_floor):
                floor_mult = belief_seq.new_full((), float(floor_weight_floor))
            elif tol_mean <= float(floor_tol_threshold):
                floor_mult = belief_seq.new_full((), float(floor_weight_mid))
            else:
                floor_mult = belief_seq.new_ones(())

        L_total = (
            loss_weights["state"] * L_state
            + loss_weights["reward"] * L_reward
            + loss_weights["done"] * L_done
            + floor_mult * loss_weights.get("success", 0.0) * L_success
            + floor_mult * loss_weights.get("task_error", 0.0) * L_task_error
            + beta_kl * L_kl
            + floor_mult * loss_weights.get("value", 0.0) * L_value
        )
        head_loss_weighted = belief_seq.new_zeros(())
        for h_steps, lh in L_horizon_success.items():
            key = f"horizon_success_h{int(h_steps)}"
            term = floor_mult * loss_weights.get(key, 0.0) * lh
            L_total = L_total + term
            head_loss_weighted = head_loss_weighted + term
        for h_steps, ld in L_drop_risk.items():
            key = f"drop_risk_h{int(h_steps)}"
            term = floor_mult * loss_weights.get(key, 0.0) * ld
            L_total = L_total + term
            head_loss_weighted = head_loss_weighted + term
        if self.with_recovery_head:
            rec_term = floor_mult * loss_weights.get("recovery", 0.0) * L_recovery
            L_total = L_total + rec_term
            head_loss_weighted = head_loss_weighted + rec_term

        loss_dynamics_or_recon = (
            loss_weights["state"] * L_state
            + loss_weights["reward"] * L_reward
            + loss_weights["done"] * L_done
            + beta_kl * L_kl
        )

        # Cache scalar metrics for telemetry.
        with torch.no_grad():
            metrics = {
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
                # Phase A4 P4 (2026-06-09): task-aware diagnostics.
                "floor_loss_mult": float(floor_mult.detach().item()),
                "loss_recovery": float(L_recovery.detach().item()),
                "recovery_pos_rate": float(recovery_pos_rate),
                # Phase A4 P4 wired (2026-06-10): degeneracy / coverage
                # diagnostics so downstream gates can decide whether to
                # trust the heads.
                "recovery_floor_sample_fraction": float(
                    recovery_floor_sample_fraction
                ),
                "recovery_missing_target_count": float(
                    recovery_missing_target_count
                ),
            }
            for h_steps, lh in L_horizon_success.items():
                metrics[f"loss_horizon_success_h{int(h_steps)}"] = float(
                    lh.detach().item()
                )
            for h_steps, ld in L_drop_risk.items():
                metrics[f"loss_drop_risk_h{int(h_steps)}"] = float(
                    ld.detach().item()
                )
            for h_steps, pr in horizon_pos_rate.items():
                metrics[f"horizon_success_pos_rate_h{int(h_steps)}"] = float(pr)
            for h_steps, pr in drop_pos_rate.items():
                metrics[f"drop_risk_pos_rate_h{int(h_steps)}"] = float(pr)
            for h_steps, code in horizon_skipped.items():
                metrics[f"horizon_skipped_h{int(h_steps)}"] = float(code)
            metrics.update({
                "success_label_pos_frac": float(success_label_pos_frac),
                "drop_label_pos_frac": float(drop_label_pos_frac),
                "recovery_label_pos_frac": float(recovery_label_pos_frac),
                "success_logit_mean": float(success_logit_mean),
                "success_logit_std": float(success_logit_std),
                "success_logit_abs_mean": float(success_logit_abs_mean),
                "drop_logit_mean": float(drop_logit_mean),
                "drop_logit_std": float(drop_logit_std),
                "drop_logit_abs_mean": float(drop_logit_abs_mean),
                "recovery_logit_mean": float(recovery_logit_mean),
                "recovery_logit_std": float(recovery_logit_std),
                "recovery_logit_abs_mean": float(recovery_logit_abs_mean),
                "loss_dynamics_or_recon": float(loss_dynamics_or_recon.detach().item()),
                "head_loss_weighted_fraction": float(
                    (head_loss_weighted.detach() / L_total.detach().clamp_min(1e-9)).item()
                ),
            })
            self._last_metrics = metrics

        result = {
            "loss_total": L_total,
            "loss_state": L_state,
            "loss_reward": L_reward,
            "loss_done": L_done,
            "loss_success": L_success,
            "loss_task_error": L_task_error,
            "loss_value": L_value,
            "loss_kl": L_kl,
            "loss_recovery": L_recovery,
            "recovery_pos_rate": belief_seq.new_full((), float(recovery_pos_rate)),
        }
        for h_steps, lh in L_horizon_success.items():
            result[f"loss_horizon_success_h{int(h_steps)}"] = lh
        for h_steps, ld in L_drop_risk.items():
            result[f"loss_drop_risk_h{int(h_steps)}"] = ld
        if return_metrics:
            return result, dict(self._last_metrics)
        return result

    # ------------------------------------------------------------------
    # Phase A4 P4 (2026-06-09): horizon / recovery target builders.
    # All targets only use within-window information; no cross-episode
    # crossing (the cumulative-product mask `seg_ok` enforces that).
    # ------------------------------------------------------------------

    @staticmethod
    def _build_horizon_valid_mask(
        batch: SequenceBatch, horizon: int
    ) -> torch.Tensor:
        from .wm_head_targets import build_horizon_valid_mask

        return build_horizon_valid_mask(batch, horizon)

    @staticmethod
    def _build_horizon_success_target(
        batch: SequenceBatch, horizon: int
    ) -> torch.Tensor:
        from .wm_head_targets import build_horizon_success_target

        return build_horizon_success_target(batch, horizon)

    @staticmethod
    def _build_horizon_drop_target(
        batch: SequenceBatch, horizon: int
    ) -> torch.Tensor:
        from .wm_head_targets import build_horizon_drop_target

        return build_horizon_drop_target(batch, horizon)

    def _build_recovery_target_soft_or(
        self,
        batch: SequenceBatch,
        *,
        horizon: int,
        d_hat_margin: float,
        d_hat_tol_min_steps: int,
        floor_tol_threshold: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        from .wm_head_targets import build_recovery_target_soft_or

        return build_recovery_target_soft_or(
            batch,
            horizon=horizon,
            d_hat_margin=d_hat_margin,
            d_hat_tol_min_steps=d_hat_tol_min_steps,
            floor_tol_threshold=floor_tol_threshold,
        )

    # ------------------------------------------------------------------
    # EIR-MPC (Phase: full H-step MPPI): prior-only latent rollout API.
    # These are inference-only helpers used by the RSSM-MPPI planner; they
    # never touch the training graph (all under no_grad) and never look at
    # future real observations.
    # ------------------------------------------------------------------

    @property
    def supports_prior_rollout(self) -> bool:
        """True iff a real action-conditioned prior transition is available."""
        return (
            hasattr(self, "gru")
            and hasattr(self, "prior_net")
            and hasattr(self, "action_encoder")
        )

    def get_state_from_belief(self, belief: torch.Tensor) -> RSSMState:
        """Split a belief ``[B, belief_dim]`` into an ``RSSMState``.

        ``belief == concat(h[h_dim], z[z_dim])`` for RSSM-lite. When the width
        does not match the expected split we keep the whole belief and leave
        ``deter``/``stoch`` as ``None`` (fixed-belief fallback).
        """
        if belief.dim() != 2:
            belief = belief.reshape(belief.shape[0], -1)
        if int(belief.shape[-1]) == int(self.belief_dim):
            h = belief[:, : self.h_dim]
            z = belief[:, self.h_dim :]
            return RSSMState(belief=belief, deter=h, stoch=z)
        return RSSMState(belief=belief, deter=None, stoch=None)

    @torch.no_grad()
    def imagine_step(
        self,
        state: "RSSMState | torch.Tensor",
        action: torch.Tensor,
        *,
        deterministic: bool = True,
        sample_temperature: float = 1.0,
    ) -> "tuple[RSSMState, dict[str, torch.Tensor]]":
        """Prior-only one-step latent rollout.

        Given the current state (belief at ``b_t``) and an ``action a_t``:
          * scores the transition with the (b_t, a_t)-conditioned heads
            (success/drop/recovery/task-error) plus the belief-only value head;
          * advances the latent with the *prior* (no posterior / no real
            future obs): ``h_{t+1} = GRU(concat(z_t, enc(a_t)), h_t)`` and
            ``z_{t+1} = prior_mean(h_{t+1})`` when ``deterministic=True``, or
            a reparameterised prior sample when ``deterministic=False``.

        Returns ``(next_state, preds)``. ``preds`` always carries a
        ``prior_std_mean`` uncertainty proxy and (when the heads exist)
        ``success``/``drop``/``recovery`` probabilities, ``value`` and
        ``task_error_delta``. Falls back to a fixed belief (state unchanged)
        when the backbone lacks a usable prior transition.
        """
        if not isinstance(state, RSSMState):
            state = self.get_state_from_belief(state)
        belief = state.belief
        if belief.dim() != 2:
            belief = belief.reshape(belief.shape[0], -1)
        action = action.reshape(belief.shape[0], -1).to(belief.dtype)

        preds = self.heads.predict_transition(belief, action)
        out: dict[str, torch.Tensor] = {}
        # head probabilities (mean over horizon heads when present)
        succ_list: list[torch.Tensor] = []
        drop_list: list[torch.Tensor] = []
        horizon_logits = preds.get("horizon_logits")
        if isinstance(horizon_logits, dict):
            for _h, head_out in horizon_logits.items():
                if "success" in head_out:
                    succ_list.append(torch.sigmoid(head_out["success"]).reshape(-1))
                if "drop" in head_out:
                    drop_list.append(torch.sigmoid(head_out["drop"]).reshape(-1))
        if succ_list:
            out["success_stack"] = torch.stack(succ_list, dim=0)
            out["success"] = out["success_stack"].mean(dim=0)
        elif "success_logits" in preds:
            out["success"] = torch.sigmoid(preds["success_logits"]).reshape(-1)
        if drop_list:
            out["drop"] = torch.stack(drop_list, dim=0).mean(dim=0)
        if "recovery_logits" in preds:
            out["recovery"] = torch.sigmoid(preds["recovery_logits"]).reshape(-1)
        if "task_error_delta" in preds:
            out["task_error_delta"] = preds["task_error_delta"].reshape(-1)
        if self.heads.has_value_head:
            v = self.heads.predict_value(belief)
            if v is not None:
                out["value"] = v.reshape(-1)

        # advance latent via prior transition (or fixed-belief fallback)
        if self.supports_prior_rollout and state.deter is not None and state.stoch is not None:
            a_emb = self.action_encoder(action)
            gru_in = torch.cat([state.stoch, a_emb], dim=-1)
            h_next = self.gru(gru_in, state.deter)
            prior_mu, prior_std = self._split_dist(self.prior_net(h_next))
            if deterministic:
                z_next = prior_mu
                step_mode = IMAGINE_PRIOR_ROLLOUT
            else:
                eps = torch.randn_like(prior_std)
                z_next = prior_mu + float(sample_temperature) * prior_std * eps
                step_mode = IMAGINE_STOCHASTIC_PRIOR_ROLLOUT
            belief_next = torch.cat([h_next, z_next], dim=-1)
            next_state = RSSMState(belief=belief_next, deter=h_next, stoch=z_next)
            out["prior_std_mean"] = prior_std.mean(dim=-1).reshape(-1)
            out["imagine_mode"] = step_mode
        else:
            next_state = RSSMState(belief=belief, deter=state.deter, stoch=state.stoch)
            out["prior_std_mean"] = belief.new_zeros(belief.shape[0])
            out["imagine_mode"] = IMAGINE_FIXED_BELIEF_FALLBACK
        return next_state, out

    def report_metrics(self) -> dict[str, float]:
        return dict(self._last_metrics)


__all__ = [
    "RSSMLiteWorldModel",
    "RSSMState",
    "IMAGINE_PRIOR_ROLLOUT",
    "IMAGINE_STOCHASTIC_PRIOR_ROLLOUT",
    "IMAGINE_FIXED_BELIEF_FALLBACK",
]
