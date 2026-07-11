"""
Transformer-lite world model (auxiliary-only).

Architecture:
    token_t input  = concat(obs_t, action_{t-1}, reward_{t-1}, done_{t-1})
                     For t == 0, action/reward/done are zero-padded.
    token_encoder  = Linear(in_dim -> hidden_dim) + LayerNorm
    pos_embedding  = nn.Embedding(max_seq_len, hidden_dim) (learned)
    encoder        = nn.TransformerEncoder(d_model=hidden_dim, nhead=8,
                                            ffn_dim=4*hidden_dim, dropout=0.05,
                                            num_layers=3, batch_first=True,
                                            norm_first=True)
    belief b_t     = LayerNorm(encoder_out[t])

Masking:
    Per-batch attention mask combining
    (a) causal lower-triangular mask: token i may attend to j iff j <= i
    (b) segment boundary mask: token i may attend to j iff
        segment_id[i] == segment_id[j]
    Combined into a float mask of shape (B*nhead, T, T) with -inf on blocked
    positions, suitable for nn.TransformerEncoderLayer's `src_mask`.

Heads share the PredictionHeads bundle with RSSM-lite (next_obs_delta,
reward, done, success, task_error_delta, value).

Loss masking matches RSSM-lite:
    eff_state = valid_mask * (1 - done)        # for state_delta, task_error_delta
    eff_basic = valid_mask                      # for reward, done, success, value
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .seq_wm_base import SequenceBatch, SequenceWorldModelBase
from .seq_wm_heads import PredictionHeads


def _masked_mean(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Element-wise mean over broadcast (mask, value); see rssm_lite version."""
    if mask.ndim < value.ndim:
        for _ in range(value.ndim - mask.ndim):
            mask = mask.unsqueeze(-1)
    expanded = mask.expand_as(value)
    denom = expanded.sum().clamp_min(torch.tensor(1.0, device=value.device))
    return (value * expanded).sum() / denom


def _build_attn_mask(
    done: torch.Tensor,        # (T, B) float in {0, 1}
    valid_mask: torch.Tensor,  # (T, B) float in {0, 1}  (kept for API symmetry)
    segment_id: torch.Tensor,  # (T, B) long
    num_heads: int,
) -> torch.Tensor:
    """Build (B*H, T, T) float mask: 0.0 on allowed, -inf on blocked.

    Allowed iff (j <= i) AND segment_id[b, i] == segment_id[b, j].

    Note: ``valid_mask`` is intentionally NOT used here. valid_mask drives
    which (t, b) positions contribute to loss (cross-episode protection),
    but attention must still let every token attend to itself and to its
    own segment, otherwise the new-segment first token has no context.
    """
    T, B = done.shape
    device = done.device
    rng = torch.arange(T, device=device)
    causal = (rng.unsqueeze(0) <= rng.unsqueeze(1))                 # (T, T) bool
    seg_t = segment_id.transpose(0, 1)                              # (B, T)
    seg_eq = seg_t.unsqueeze(2) == seg_t.unsqueeze(1)               # (B, T, T)
    allow = causal.unsqueeze(0) & seg_eq                            # (B, T, T)
    mask_f = torch.zeros(B, T, T, device=device)
    mask_f[~allow] = float("-inf")
    mask_f = (
        mask_f.unsqueeze(1)
              .expand(B, num_heads, T, T)
              .reshape(B * num_heads, T, T)
    )
    return mask_f


def _shifted(x: torch.Tensor, fill: float = 0.0) -> torch.Tensor:
    """Return x shifted forward by 1 along dim 0, with `fill` at index 0.
    Shape preserved.
    """
    out = torch.zeros_like(x)
    out[1:] = x[:-1]
    if fill != 0.0:
        out[0] = fill
    return out


class TransformerLiteWorldModel(SequenceWorldModelBase):
    def __init__(
        self,
        *,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 3,
        ffn_dim: int = 1024,
        dropout: float = 0.05,
        max_seq_len: int = 32,
        head_hidden: int = 256,
        with_value_head: bool = True,
        with_success_head: bool = True,
        with_task_error_head: bool = True,
    ) -> None:
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_heads = int(num_heads)
        self.belief_dim = self.hidden_dim
        self.max_seq_len = int(max_seq_len)

        in_dim = self.obs_dim + self.action_dim + 2  # +1 reward, +1 done
        self.token_encoder = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
        self.pos_emb = nn.Embedding(self.max_seq_len, hidden_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.out_norm = nn.LayerNorm(hidden_dim)

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

        self._last_metrics: dict[str, float] = {}

    # ----- core -----

    def _build_tokens(self, batch: SequenceBatch) -> torch.Tensor:
        """Per-step token features. Shape (T, B, in_dim)."""
        T, B = batch.T, batch.B
        prev_action = _shifted(batch.action)              # (T, B, A)
        prev_reward = _shifted(batch.reward).unsqueeze(-1)  # (T, B, 1)
        prev_done = _shifted(batch.done).unsqueeze(-1)     # (T, B, 1)
        tok = torch.cat([batch.obs, prev_action, prev_reward, prev_done], dim=-1)
        return tok  # (T, B, in_dim)

    def encode_sequence(self, batch: SequenceBatch) -> torch.Tensor:
        T, B = batch.T, batch.B
        if T > self.max_seq_len:
            raise ValueError(
                f"seq_len {T} > max_seq_len {self.max_seq_len}; bump "
                "transformer max_seq_len if you need a longer window."
            )
        tok = self._build_tokens(batch)                              # (T, B, in_dim)
        emb = self.token_encoder(tok)                                # (T, B, H)
        # Add learned positional embeddings.
        pos = self.pos_emb(torch.arange(T, device=batch.device))     # (T, H)
        emb = emb + pos.unsqueeze(1)                                 # (T, B, H)
        # Switch to batch_first.
        emb_bt = emb.transpose(0, 1).contiguous()                    # (B, T, H)
        attn_mask = _build_attn_mask(
            done=batch.done,
            valid_mask=batch.valid_mask,
            segment_id=batch.segment_id,
            num_heads=self.num_heads,
        )
        h = self.encoder(emb_bt, mask=attn_mask)                     # (B, T, H)
        h = self.out_norm(h)
        belief = h.transpose(0, 1).contiguous()                       # (T, B, H)
        return belief

    def compute_aux_loss(
        self,
        batch: SequenceBatch,
        *,
        loss_weights: Optional[dict[str, float]] = None,
        beta_kl: float = 0.0,             # Transformer has no KL; ignored
        value_detach_belief: bool = False,
    ) -> dict[str, torch.Tensor]:
        if loss_weights is None:
            loss_weights = {
                "state": 1.0, "reward": 1.0, "done": 0.2,
                "success": 1.0, "task_error": 1.0, "value": 0.1,
            }
        T, B = batch.T, batch.B
        belief_seq = self.encode_sequence(batch)                  # (T, B, H)
        belief_flat = belief_seq.reshape(T * B, -1)
        action_flat = batch.action.reshape(T * B, -1)

        pred = self.heads.predict_transition(belief_flat, action_flat)

        valid = batch.valid_mask
        not_done = (1.0 - batch.done)
        eff_state = (valid * not_done).reshape(T * B)
        eff_basic = valid.reshape(T * B)

        L_state = _masked_mean(
            (pred["next_obs_delta"] - (batch.next_obs - batch.obs).reshape(T * B, -1)) ** 2,
            eff_state,
        )
        L_reward = _masked_mean(
            (pred["reward"].squeeze(-1) - batch.reward.reshape(T * B)) ** 2,
            eff_basic,
        )
        device = belief_seq.device
        denom_basic = eff_basic.sum().clamp_min(torch.tensor(1.0, device=device))
        L_done = F.binary_cross_entropy_with_logits(
            pred["done_logits"].squeeze(-1),
            batch.done.reshape(T * B),
            weight=eff_basic,
            reduction="sum",
        ) / denom_basic

        L_success = belief_seq.new_zeros(())
        if ("success_logits" in pred and batch.success_count is not None
                and batch.next_success_count is not None):
            success_target = (
                batch.next_success_count - batch.success_count > 0.5
            ).to(belief_seq.dtype).reshape(T * B)
            L_success = F.binary_cross_entropy_with_logits(
                pred["success_logits"].squeeze(-1),
                success_target,
                weight=eff_basic,
                reduction="sum",
            ) / denom_basic

        L_task_error = belief_seq.new_zeros(())
        if ("task_error_delta" in pred and batch.d_hat is not None
                and batch.next_d_hat is not None):
            tg = (batch.next_d_hat - batch.d_hat).reshape(T * B)
            L_task_error = _masked_mean(
                (pred["task_error_delta"].squeeze(-1) - tg) ** 2,
                eff_state,
            )

        L_value = belief_seq.new_zeros(())
        if self.heads.has_value_head and batch.returns is not None:
            v_in = belief_flat.detach() if value_detach_belief else belief_flat
            v_pred = self.heads.predict_value(v_in).squeeze(-1)
            L_value = _masked_mean(
                (v_pred - batch.returns.reshape(T * B)) ** 2,
                eff_basic,
            )

        L_total = (
            loss_weights["state"] * L_state
            + loss_weights["reward"] * L_reward
            + loss_weights["done"] * L_done
            + loss_weights.get("success", 0.0) * L_success
            + loss_weights.get("task_error", 0.0) * L_task_error
            + loss_weights.get("value", 0.0) * L_value
        )

        with torch.no_grad():
            # Mask "valid attention" ratio: average of valid token positions.
            valid_ratio = float(batch.valid_mask.mean().item())
            self._last_metrics = {
                "loss_total": float(L_total.detach().item()),
                "loss_state": float(L_state.detach().item()),
                "loss_reward": float(L_reward.detach().item()),
                "loss_done": float(L_done.detach().item()),
                "loss_success": float(L_success.detach().item()),
                "loss_task_error": float(L_task_error.detach().item()),
                "loss_value": float(L_value.detach().item()),
                "loss_kl": 0.0,
                "beta_kl": 0.0,
                "free_nats": 0.0,
                "trans_layers": float(len(self.encoder.layers)),
                "trans_heads": float(self.num_heads),
                "attn_valid_ratio": valid_ratio,
            }

        return {
            "loss_total": L_total,
            "loss_state": L_state,
            "loss_reward": L_reward,
            "loss_done": L_done,
            "loss_success": L_success,
            "loss_task_error": L_task_error,
            "loss_value": L_value,
            "loss_kl": belief_seq.new_zeros(()),
        }

    def report_metrics(self) -> dict[str, float]:
        return dict(self._last_metrics)


__all__ = ["TransformerLiteWorldModel", "_build_attn_mask"]
