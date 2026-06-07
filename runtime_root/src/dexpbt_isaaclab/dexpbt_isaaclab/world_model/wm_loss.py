"""
World Model losses: dynamics prediction + value_imag.
All losses defined per-sample with clear gradient paths.
"""

import torch
import torch.nn.functional as F


class WMLoss:
    """
    Computes all world model losses.
    """

    def __init__(
        self,
        alpha_pos: float = 1.0,
        alpha_dhat: float = 0.5,
        alpha_vel: float = 0.5,
        alpha_ang: float = 0.3,
        alpha_reward: float = 0.1,
        alpha_done: float = 0.5,
    ):
        self.alpha_pos = alpha_pos
        self.alpha_dhat = alpha_dhat
        self.alpha_vel = alpha_vel
        self.alpha_ang = alpha_ang
        self.alpha_reward = alpha_reward
        self.alpha_done = alpha_done

    def compute_dynamics_loss(
        self,
        pred: dict[str, torch.Tensor],
        target: dict[str, torch.Tensor],
        failure_t: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute dynamics loss from predictions and targets.
        All deltas are predicted directly (target = actual delta).

        Args:
            pred:      prediction dict from WorldModel.forward_mean()
            target:    target dict with same keys
            failure_t: (B, 1) failure label

        Returns:
            scalar total dynamics loss
        """
        # object position delta
        loss_pos = F.mse_loss(pred["object_pos_delta"], target["object_pos_delta"], reduction="mean")
        # d_hat delta
        loss_dhat = F.mse_loss(pred["d_hat_delta"], target["d_hat_delta"], reduction="mean")
        # linear velocity delta
        loss_linvel = F.mse_loss(pred["linvel_delta"], target["linvel_delta"], reduction="mean")
        # angular velocity delta
        loss_angvel = F.mse_loss(pred["angvel_delta"], target["angvel_delta"], reduction="mean")
        # reward prediction (clip target)
        r_target = target["reward"].clamp(-10.0, 10.0)
        loss_reward = F.mse_loss(pred["reward"], r_target, reduction="mean")
        # done prediction
        done_target = target["done"].clamp(0.0, 1.0)
        loss_done = F.binary_cross_entropy_with_logits(pred["done_logits"], done_target, reduction="mean")

        total = (
            self.alpha_pos * loss_pos
            + self.alpha_dhat * loss_dhat
            + self.alpha_vel * loss_linvel
            + self.alpha_ang * loss_angvel
            + self.alpha_reward * loss_reward
            + self.alpha_done * loss_done
        )
        return total

    def compute_value_imag_loss(
        self,
        V_t: torch.Tensor,
        reward_pred: torch.Tensor,
        V_tp1: torch.Tensor,
        done_prob: torch.Tensor,
        gamma: float = 0.99,
        gate_ratio: float = 0.5,
        uncertainty: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Compute value imagination loss:
            L = MSE(V_t, r_pred + gamma * V_tp1 * (1 - done_prob))

        Gradient path: L.backward() → V_t → aux_value_head → shared_encoder → PPO actor/critic

        Args:
            V_t:       (B,) V_aux(s_t) from aux_value_head
            reward_pred: (B,) predicted reward from WM
            V_tp1:     (B,) V_aux(s_{t+1}) — detached (TD target side)
            done_prob: (B,) done probability from WM — detached
            gamma:     discount factor
            gate_ratio: fraction of lowest-uncertainty samples to keep
            uncertainty: (B,) per-sample uncertainty — if provided, apply gate
        """
        done_mask = done_prob.detach().clamp(0.0, 1.0)
        target = reward_pred.detach() + gamma * V_tp1 * (1.0 - done_mask)

        loss = F.mse_loss(V_t, target, reduction="none")

        if uncertainty is not None and gate_ratio < 1.0:
            threshold = torch.quantile(uncertainty, 1.0 - gate_ratio)
            mask = (uncertainty <= threshold).float()
            if mask.sum() < 1:
                return torch.tensor(0.0, device=V_t.device, requires_grad=False)
            loss = (loss * mask).sum() / mask.sum()
        else:
            loss = loss.mean()

        return loss