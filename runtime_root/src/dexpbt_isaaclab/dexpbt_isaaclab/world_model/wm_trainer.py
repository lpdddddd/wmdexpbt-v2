"""
WM Trainer: trains world model with configurable strategy.
Supports value_imag_h1/h3 via an auxiliary imagined value head on
PPO shared representation.
"""

from __future__ import annotations

import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Optional

from .wm_config import WMConfig
from .wm_module import WorldModel, compute_uncertainty
from .wm_buffer import WMBuffer
from .wm_loss import WMLoss


# ----------------------------------------------------------------------
# Strategy definitions
# ----------------------------------------------------------------------

STRATEGIES = [
    "none",
    "aux_low",
    "aux_mid",
    "value_imag_h1",
    "value_imag_h3",
    # Phase A4: sequence WM family. legacy WMTrainer skips them (trains_wm=False);
    # SeqWMTrainer (seq_wm_trainer.py) trains them via PPO ExperienceBuffer.
    "rssm_aux",
    "rssm_belief_value",
    "trans_aux",
    "trans_belief_value",
]

# Strategies whose training path is the SeqWMTrainer rather than the legacy
# single-step WMTrainer. Imported by train.py / pbt_observer.py for dispatch.
SEQUENCE_STRATEGIES: frozenset[str] = frozenset({
    "rssm_aux",
    "rssm_belief_value",
    "trans_aux",
    "trans_belief_value",
})

RSSM_FAMILY: frozenset[str] = frozenset({"rssm_aux", "rssm_belief_value"})
TRANS_FAMILY: frozenset[str] = frozenset({"trans_aux", "trans_belief_value"})

# Paper obs layout: 86 + 6 * num_keypoints (92 for regrasping, 110 for reorientation).
_PAPER_OBS_BASE = 86


def _paper_obs_num_keypoints(obs_dim: int) -> int | None:
    if obs_dim < _PAPER_OBS_BASE or (obs_dim - _PAPER_OBS_BASE) % 6 != 0:
        return None
    return (obs_dim - _PAPER_OBS_BASE) // 6


def _paper_obs_wm_layout(obs_dim: int) -> dict[str, slice] | None:
    """Slice indices for WM-updatable fields inside paper-mode policy obs."""
    n_kp = _paper_obs_num_keypoints(obs_dim)
    if n_kp is None:
        return None
    base = 70
    kp_palm_start = base
    kp_goal_start = base + 3 * n_kp
    obj_start = base + 6 * n_kp
    return {
        "palm_center": slice(46, 49),
        "kp_rel_palm": slice(kp_palm_start, kp_goal_start),
        "kp_rel_goal": slice(kp_goal_start, obj_start),
        "object_linvel": slice(obj_start + 4, obj_start + 7),
        "object_angvel": slice(obj_start + 7, obj_start + 10),
        "d_hat": slice(obj_start + 13, obj_start + 14),
    }


@dataclass
class WMStrategy:
    """World model training strategy configuration."""
    name: str
    trains_wm: bool
    lambda_wm: float
    use_value_imag: bool
    imag_horizon: Optional[int] = None
    uncertainty_gate: bool = False
    uncertainty_gate_ratio: float = 0.5


def get_strategy(name: str) -> WMStrategy:
    """Load strategy from YAML, fallback to defaults."""
    defaults = {
        "none":         WMStrategy("none",        trains_wm=False,  lambda_wm=0.0,   use_value_imag=False),
        "aux_low":      WMStrategy("aux_low",      trains_wm=True,   lambda_wm=0.01,  use_value_imag=False),
        "aux_mid":      WMStrategy("aux_mid",      trains_wm=True,   lambda_wm=0.05,  use_value_imag=False),
        "value_imag_h1": WMStrategy("value_imag_h1", trains_wm=True,  lambda_wm=0.05,  use_value_imag=True, imag_horizon=1, uncertainty_gate=False, uncertainty_gate_ratio=0.5),
        "value_imag_h3": WMStrategy("value_imag_h3", trains_wm=True,  lambda_wm=0.05,  use_value_imag=True, imag_horizon=3, uncertainty_gate=True, uncertainty_gate_ratio=0.5),
        # Sequence WM family (Phase A4): the legacy WMTrainer treats them as
        # no-ops (trains_wm=False, lambda_wm=0); the SeqWMTrainer is what
        # actually drives the loss. Keeping `name` consistent here is what
        # lets fixed mode + bayesian pool dispatch through `--wm_strategy`.
        "rssm_aux":           WMStrategy("rssm_aux",           trains_wm=False, lambda_wm=0.0, use_value_imag=False),
        "rssm_belief_value":  WMStrategy("rssm_belief_value",  trains_wm=False, lambda_wm=0.0, use_value_imag=False),
        "trans_aux":          WMStrategy("trans_aux",          trains_wm=False, lambda_wm=0.0, use_value_imag=False),
        "trans_belief_value": WMStrategy("trans_belief_value", trains_wm=False, lambda_wm=0.0, use_value_imag=False),
    }
    if name in defaults:
        return defaults[name]

    # Try to load from YAML
    try:
        cfg_file = resources.files("dexpbt_isaaclab.agents").joinpath("wm_strategies.yaml")
        with open(str(cfg_file)) as f:
            strategies_dict = yaml.safe_load(f)
        if name in strategies_dict:
            d = strategies_dict[name]
            return WMStrategy(
                name=name,
                trains_wm=d.get("trains_wm", False),
                lambda_wm=float(d.get("lambda_wm", 0.0)),
                use_value_imag=d.get("use_value_imag", False),
                imag_horizon=d.get("imag_horizon"),
                uncertainty_gate=d.get("uncertainty_gate", False),
                uncertainty_gate_ratio=float(d.get("uncertainty_gate_ratio", 0.5)),
            )
    except Exception:
        pass

    raise ValueError(f"Unknown WM strategy: {name}")


# ----------------------------------------------------------------------
# WM Trainer
# ----------------------------------------------------------------------

class WMTrainer:
    """
    World model trainer.

    Integrates with PPO model for value_imag strategies.
    Call step() each PPO epoch (controlled by update_every_n_epochs).
    """

    def __init__(
        self,
        wm_module: WorldModel,
        wm_optimizer: torch.optim.Optimizer,
        wm_buffer: WMBuffer,
        config: WMConfig | None = None,
        strategy: WMStrategy | None = None,
        ppo_model=None,        # rl_games player/model for value_imag
        ppo_optimizer=None,    # PPO optimizer for value_imag
        tensorboard_writer=None,
        device: str = "cuda:0",
    ):
        self.module = wm_module
        self.optimizer = wm_optimizer
        self.buffer = wm_buffer
        self.config = config or WMConfig()
        self.strategy = strategy or get_strategy("none")
        self.ppo_model = ppo_model          # rl_games model
        self.ppo_optimizer = ppo_optimizer  # PPO optimizer
        self.tensorboard = tensorboard_writer
        self.device = device

        self._epoch_count = 0
        self._epoch_since_last_update = 0

        # value_imag state
        self._value_imag_enabled = False
        self.aux_value_head: Optional[nn.Module] = None
        self._last_shared_encoder_grad_norm = float("nan")
        self._last_aux_value_head_grad_norm = float("nan")
        self._last_gate_pass_ratio = float("nan")

        # NOTE: Do NOT call _add_aux_value_head() here — ppo_model is None at this point.
        # Caller must call wm_trainer._add_aux_value_head() after setting ppo_model/optimizer.

    # ------------------------------------------------------------------
    # PPO model probing (Patch 0)
    # ------------------------------------------------------------------

    def _probe_rl_games_model(self) -> dict:
        """
        Probe rl_games model structure to find shared encoder and its output dim.

        Confirmed structure (from actual probe of PpoPlayerContinuous):
            model (Network)
            ├── running_mean_std: input normalization
            ├── value_mean_std: value normalization
            └── a2c_network (Network)
                ├── rnn: LSTMWithDones(LSTM(obs_dim, 768))       ← shared encoder
                ├── layer_norm: LayerNorm(768)
                ├── actor_mlp: Linear(768→768)→ELU→Linear(768→512)→ELU→Linear(512→256)→ELU  ← shared MLP
                ├── value: Linear(256→1)                          ← critic head
                └── mu: Linear(256→23)                            ← actor head

        Returns:
            dict with 'encoder' (the full a2c_network), 'encoder_output_dim' (256),
            and 'value_head' / 'mu_head' references.
        """
        results = {
            "encoder": None, "encoder_output_dim": None,
            "value_head": None, "mu_head": None,
        }

        if self.ppo_model is None:
            return results

        # ppo_model is the A2CAgent or Network itself
        model = self.ppo_model
        if hasattr(model, "model"):
            model = model.model  # A2CAgent.model → Network

        # Navigate to a2c_network
        if not hasattr(model, "a2c_network"):
            print("[WM] WARNING: model has no a2c_network — cannot probe shared encoder")
            return results

        a2c = model.a2c_network
        results["encoder"] = a2c  # full a2c_network as encoder

        # Find actor_mlp output dim = shared encoder output dim
        if hasattr(a2c, "actor_mlp"):
            actor_mlp = a2c.actor_mlp
            if isinstance(actor_mlp, nn.Sequential):
                # Find last Linear in actor_mlp
                for m in reversed(list(actor_mlp)):
                    if isinstance(m, nn.Linear):
                        results["encoder_output_dim"] = m.out_features
                        break
            elif hasattr(actor_mlp, "weight"):
                results["encoder_output_dim"] = actor_mlp.weight.shape[0]

        # Fallback: use value head's in_features
        if results["encoder_output_dim"] is None and hasattr(a2c, "value"):
            if isinstance(a2c.value, nn.Linear):
                results["encoder_output_dim"] = a2c.value.in_features

        # Store references to heads for debugging
        if hasattr(a2c, "value"):
            results["value_head"] = a2c.value
        if hasattr(a2c, "mu"):
            results["mu_head"] = a2c.mu

        return results

    def _add_aux_value_head(self) -> None:
        """
        Register aux_value_head on PPO shared encoder.

        After probing, aux_value_head takes encoder output (256-dim) as input.
        The full forward path: obs → running_mean_std → a2c_network(rnn+actor_mlp) → aux_value_head → V

        value_imag_h1/value_imag_h3 use an auxiliary imagined value head on
        PPO shared representation (do not replace PPO main value branch).

        Gradient path (detach controls):
            V_t = aux_value_head(h_obs_t)          ← detach=False: grad flows to a2c_network
            V_tp1 = aux_value_head(h_obs_tp1)      ← detach=True:  stop grad (TD target)
        """
        if self.ppo_model is None or self.ppo_optimizer is None:
            print("[WM] WARNING: ppo_model/ppo_optimizer not set — value_imag DISABLED")
            self._value_imag_enabled = False
            return

        print(f"[WM] _add_aux_value_head: ppo_model type={type(self.ppo_model).__name__}")
        probe = self._probe_rl_games_model()
        encoder = probe.get("encoder")
        shared_dim = probe.get("encoder_output_dim")
        print(f"[WM] probing result: encoder={type(encoder).__name__ if encoder else None}, "
              f"shared_dim={shared_dim}")

        if encoder is None or shared_dim is None:
            print("[WM] ERROR: Cannot probe PPO shared encoder. value_imag DISABLED.")
            print("         Probing failed — policy impact path unavailable. Fix probing first.")
            self._value_imag_enabled = False
            return

        # Create aux value head
        self.aux_value_head = nn.Sequential(
            nn.Linear(shared_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        ).to(self.device)

        # Register to rl_games model
        model = self.ppo_model
        if hasattr(model, "model"):
            model = model.model  # unwrap A2CAgent → Network
        model.aux_value_head = self.aux_value_head

        # === CRITICAL: add to PPO optimizer param_groups ===
        aux_lr = self.ppo_optimizer.param_groups[0]["lr"]
        self.ppo_optimizer.add_param_group({
            "params": list(self.aux_value_head.parameters()),
            "lr": aux_lr,
            "name": "aux_value_head",
        })

        n_aux = sum(p.numel() for p in self.aux_value_head.parameters())
        print(
            "[WM] auxiliary imagined value head registered on shared representation "
            f"(shared_dim={shared_dim}, {n_aux:,} params)"
        )
        self._value_imag_enabled = True

    def _re_register_aux_head(self, new_optimizer) -> None:
        """Re-register aux_value_head to a new optimizer (after restart)."""
        if self.aux_value_head is None:
            return
        self.ppo_optimizer = new_optimizer
        # aux_value_head should already be on the restored model
        model = self.ppo_model
        if hasattr(model, "model"):
            model = model.model
        if hasattr(model, "aux_value_head"):
            self.aux_value_head = model.aux_value_head
        aux_lr = new_optimizer.param_groups[0]["lr"]
        new_optimizer.add_param_group({
            "params": list(self.aux_value_head.parameters()),
            "lr": aux_lr,
            "name": "aux_value_head",
        })

    def _get_a2c_network(self):
        """Get a2c_network from ppo_model, unwrapping A2CAgent → Network if needed."""
        model = self.ppo_model
        if hasattr(model, "model"):
            model = model.model
        if hasattr(model, "a2c_network"):
            return model.a2c_network
        return None

    @staticmethod
    def _zero_rnn_states_for_batch(a2c, batch_size: int, device: torch.device):
        """Zero LSTM/GRU states shaped for an arbitrary flat batch (seq_length=1)."""
        if not getattr(a2c, "has_rnn", False):
            return None
        num_layers = a2c.rnn_layers
        rnn_units = 1 if a2c.rnn_name == "identity" else a2c.rnn_units
        z = lambda: torch.zeros(num_layers, batch_size, rnn_units, device=device)
        if a2c.rnn_name == "lstm":
            if a2c.separate:
                return (z(), z(), z(), z())
            return (z(), z())
        if a2c.separate:
            return (z(), z())
        return (z(),)

    @staticmethod
    def _ensure_rnn_device(rnn_states, target_device):
        """Move rnn_states tensors to target_device (recursive, handles tuples)."""
        if rnn_states is None:
            return None
        if isinstance(rnn_states, torch.Tensor):
            return rnn_states.to(target_device)
        if isinstance(rnn_states, (list, tuple)):
            return tuple(WMTrainer._ensure_rnn_device(x, target_device) for x in rnn_states)
        return rnn_states

    def _get_encoder_features(self, obs_normalized: torch.Tensor, detach: bool = False) -> torch.Tensor:
        """
        Get PPO shared encoder output: h = actor_mlp(layer_norm(rnn(obs))).

        Confirmed rl_games structure:
            a2c_network:
                rnn: LSTMWithDones (obs_dim → 768)
                layer_norm: LayerNorm(768)
                actor_mlp: Linear(768→768)→ELU→Linear(768→512)→ELU→Linear(512→256)→ELU

        Output is 256-dim features → aux_value_head.

        - detach=False: h has grad, for V_aux(s_t) backward to shared encoder
        - detach=True:  h.detach(), for TD target side (no backward)
        """
        if not self._value_imag_enabled or self.aux_value_head is None:
            raise RuntimeError("value_imag not enabled")

        a2c = self._get_a2c_network()
        if a2c is None:
            raise RuntimeError("Cannot locate a2c_network for aux_value_head")

        # Get current batch size from obs
        batch_size = obs_normalized.shape[0]
        if batch_size >= 2048:
            # Training mode: use the a2c_network's own internal zero states
            # (we have no external rnn_states matching this batch size)
            a2c_was_training = a2c.training
            a2c.train()

            features = {}

            def hook_fn(module, inp, out):
                # Cast to float32: hook captures autocast bfloat16 output
                features["actor_mlp_out"] = out.to(torch.float32)

            actor_mlp = a2c.actor_mlp
            handle = actor_mlp.register_forward_hook(hook_fn)

            batch_dict = {
                "is_train": True,
                "prev_actions": torch.zeros(batch_size, 23,
                                            device=obs_normalized.device),
                "obs": obs_normalized,
                "seq_length": 1,
            }
            rnn_states = self._zero_rnn_states_for_batch(
                a2c, batch_size, obs_normalized.device
            )
            if rnn_states is not None:
                batch_dict["rnn_states"] = rnn_states
            try:
                with torch.no_grad() if detach else torch.enable_grad():
                    a2c(batch_dict)
            finally:
                handle.remove()

            if not a2c_was_training:
                a2c.eval()

            if "actor_mlp_out" not in features:
                raise RuntimeError("actor_mlp forward hook did not capture output")

            return features["actor_mlp_out"].detach() if detach else features["actor_mlp_out"]

        # Mini-batch mode (batch_size < 2048): use a2c's current rnn_states from
        # the ongoing calc_gradients() call. These are already correctly shaped.
        rnn_states = None
        try:
            # a2c.rnn_states is set during calc_gradients() before model.forward()
            if hasattr(a2c, "rnn_states"):
                rnn_states = a2c.rnn_states
            elif hasattr(a2c, "state"):  # some rl_games versions use .state dict
                st = getattr(a2c, "state", None)
                if st is not None and isinstance(st, dict) and "rnn_states" in st:
                    rnn_states = st["rnn_states"]
        except Exception:
            rnn_states = None

        if rnn_states is None:
            if hasattr(a2c, "get_default_rnn_state"):
                rnn_states = a2c.get_default_rnn_state()
            else:
                rnn_states = self._zero_rnn_states_for_batch(
                    a2c, batch_size, obs_normalized.device
                )
                if rnn_states is None:
                    raise RuntimeError("Cannot get rnn_states for mini-batch mode")

        # Verify rnn_states batch dim matches obs
        if isinstance(rnn_states, (list, tuple)):
            rs_batch = rnn_states[0].shape[1]
        elif isinstance(rnn_states, torch.Tensor):
            rs_batch = rnn_states.shape[1]
        else:
            rs_batch = batch_size  # assume it's correct

        if rs_batch != batch_size:
            # get_default_rnn_state() uses num_actors (2048); WM batch is smaller.
            # a2c_network.forward() with rnn_states=None sets states=tuple() → LSTM crash.
            rnn_states = self._zero_rnn_states_for_batch(
                a2c, batch_size, obs_normalized.device
            )
        else:
            # Mini-batch mode: rnn_states from a2c.rnn_states may be on wrong device (CPU).
            # Ensure all rnn state tensors are on the same device as obs.
            rnn_states = self._ensure_rnn_device(rnn_states, obs_normalized.device)

        a2c_was_training = a2c.training
        a2c.train()

        features = {}

        def hook_fn(module, inp, out):
            # Cast to float32: hook captures autocast bfloat16 output
            features["actor_mlp_out"] = out.to(torch.float32)

        actor_mlp = a2c.actor_mlp
        handle = actor_mlp.register_forward_hook(hook_fn)

        batch_dict = {
            "is_train": True,
            "prev_actions": torch.zeros(batch_size, 23,
                                        device=obs_normalized.device),
            "obs": obs_normalized,
            "seq_length": 1,
        }
        if rnn_states is not None:
            batch_dict["rnn_states"] = rnn_states

        try:
            with torch.no_grad() if detach else torch.enable_grad():
                a2c(batch_dict)
        finally:
            handle.remove()

        if not a2c_was_training:
            a2c.eval()

        if "actor_mlp_out" not in features:
            raise RuntimeError("actor_mlp forward hook did not capture output")

        return features["actor_mlp_out"].detach() if detach else features["actor_mlp_out"]

    def _normalize_obs(self, obs: torch.Tensor) -> torch.Tensor:
        """Normalize obs using rl_games running mean/std; cast to float32 for downstream ops."""
        model = self.ppo_model
        if hasattr(model, "running_mean_std"):
            return model.running_mean_std(obs).to(torch.float32)
        if hasattr(model, "model") and hasattr(model.model, "running_mean_std"):
            return model.model.running_mean_std(obs).to(torch.float32)
        return obs.to(torch.float32)

    def _patch_obs_from_wm_state(
        self,
        obs: torch.Tensor,
        object_pos: torch.Tensor,
        object_linvel: torch.Tensor,
        object_angvel: torch.Tensor,
        d_hat: torch.Tensor,
    ) -> torch.Tensor:
        """Write WM-tracked quantities into a paper-mode obs vector."""
        layout = _paper_obs_wm_layout(int(obs.shape[-1]))
        if layout is None:
            return obs

        obs_imag = obs.clone()
        palm = obs_imag[:, layout["palm_center"]]
        pos_delta = object_pos - palm
        if layout["kp_rel_palm"].stop - layout["kp_rel_palm"].start == pos_delta.shape[-1]:
            obs_imag[:, layout["kp_rel_palm"]] = pos_delta
        if layout["kp_rel_goal"].stop - layout["kp_rel_goal"].start == pos_delta.shape[-1]:
            # goal fixed during short imagination → kp_rel_goal moves opposite to object_pos delta.
            obs_imag[:, layout["kp_rel_goal"]] = obs[:, layout["kp_rel_goal"]] - (
                pos_delta - obs[:, layout["kp_rel_palm"]]
            )

        obs_imag[:, layout["object_linvel"]] = object_linvel
        obs_imag[:, layout["object_angvel"]] = object_angvel
        if d_hat.shape[-1] == 1:
            obs_imag[:, layout["d_hat"]] = d_hat
        else:
            obs_imag[:, layout["d_hat"]] = d_hat.unsqueeze(-1)
        return obs_imag

    def _rollout_imagined_return(
        self,
        obs_raw: torch.Tensor,
        action_t: torch.Tensor,
        object_pos_t: torch.Tensor,
        object_linvel_t: torch.Tensor,
        object_angvel_t: torch.Tensor,
        d_hat_t: torch.Tensor,
        horizon: int,
        gamma: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Multi-step WM imagination in raw obs space (matches WM training inputs).
        Returns (imagined_return, final_obs_norm, boot_discount, boot_survival).
        """
        obs_imag_raw = obs_raw.clone()
        object_pos = object_pos_t.clone()
        object_linvel = object_linvel_t.clone()
        object_angvel = object_angvel_t.clone()
        d_hat = d_hat_t.clone()

        running_discount = torch.ones(obs_imag_raw.shape[0], device=obs_imag_raw.device, dtype=torch.float32)
        running_survival = torch.ones_like(running_discount)
        imagined_return = torch.zeros_like(running_discount)

        for _ in range(horizon):
            wm_pred = self.module.forward_mean(obs_imag_raw, action_t)
            reward_pred = wm_pred["reward"].squeeze(-1).detach().to(torch.float32)
            done_prob = torch.sigmoid(wm_pred["done_logits"]).squeeze(-1).to(torch.float32).clamp(0.0, 1.0)

            imagined_return = imagined_return + running_discount * running_survival * reward_pred
            running_survival = running_survival * (1.0 - done_prob)
            running_discount = running_discount * gamma

            object_pos = object_pos + wm_pred["object_pos_delta"]
            object_linvel = object_linvel + wm_pred["linvel_delta"]
            object_angvel = object_angvel + wm_pred["angvel_delta"]
            d_hat = d_hat + wm_pred["d_hat_delta"]

            obs_imag_raw = self._patch_obs_from_wm_state(
                obs_imag_raw, object_pos, object_linvel, object_angvel, d_hat
            )

        obs_imag_norm = self._normalize_obs(obs_imag_raw)
        return imagined_return, obs_imag_norm, running_discount, running_survival

    # ------------------------------------------------------------------
    # value_imag loss (for model.get_aux_loss() integration)
    # ------------------------------------------------------------------

    def compute_value_imag_loss_dict(self) -> dict[str, torch.Tensor] | None:
        """
        Compute value_imag loss and return as dict for rl_games model.get_aux_loss().

        Returns:
            {"value_imag": loss_tensor} or None if not enabled/buffer too small.

        Gradient path:
            loss.backward() → auxiliary imagined value head
            → a2c_network shared representation → PPO model params

        This method is called from _patch_model_get_aux_loss() in train.py,
        which is invoked inside A2CAgent.calc_gradients() before PPO backward.
        """
        if not self._value_imag_enabled or self.aux_value_head is None:
            return None

        # Check buffer has enough data
        if self.buffer.size < self.config.batch_size:
            return None

        # Sample from buffer
        sample = self.buffer.sample(self.config.batch_size)
        obs_t = sample["obs_t"].to(self.device)
        action_t = sample["action_t"].to(self.device)
        object_pos_t = sample["object_pos_t"].to(self.device)
        object_linvel_t = sample["object_linvel_t"].to(self.device)
        object_angvel_t = sample["object_angvel_t"].to(self.device)
        d_hat_t = sample["d_hat_t"].to(self.device)

        obs_t_norm = self._normalize_obs(obs_t)

        try:
            # V_aux(s_t): forward through shared encoder, grad flows to encoder
            h_obs_t = self._get_encoder_features(obs_t_norm, detach=False)  # (B, shared_dim)
            h_obs_t.register_hook(
                lambda grad: setattr(
                    self,
                    "_last_shared_encoder_grad_norm",
                    float(grad.norm().detach().to(torch.float32).item()),
                )
            )
            V_t = self.aux_value_head(h_obs_t).squeeze(-1)  # (B,)
            V_t.register_hook(
                lambda grad: setattr(
                    self,
                    "_last_aux_value_head_grad_norm",
                    float(grad.norm().detach().to(torch.float32).item()),
                )
            )

            horizon = max(1, int(self.strategy.imag_horizon or 1))
            gamma = float(self.config.value_imag_gamma)
            imagined_return, obs_imag_final, boot_discount, boot_survival = self._rollout_imagined_return(
                obs_t,
                action_t,
                object_pos_t,
                object_linvel_t,
                object_angvel_t,
                d_hat_t,
                horizon=horizon,
                gamma=gamma,
            )

            # Bootstrap from imagined final state, not real obs_{t+1}.
            h_obs_imag = self._get_encoder_features(obs_imag_final, detach=True)
            V_boot = self.aux_value_head(h_obs_imag).squeeze(-1).to(torch.float32)
            value_target = imagined_return + boot_discount * boot_survival * V_boot

            # Uncertainty gate (skip high-uncertainty samples)
            if self.strategy.uncertainty_gate:
                primary_unc, _ = compute_uncertainty(self.module, obs_t, action_t)
                # torch.quantile requires float/double; cast from bfloat16 autocast
                primary_unc_f = primary_unc.to(torch.float32)
                threshold = torch.quantile(primary_unc_f,
                                          1.0 - self.strategy.uncertainty_gate_ratio)
                mask = (primary_unc_f <= threshold).float()
                if mask.sum() < 1:
                    return None
                self._last_gate_pass_ratio = float((mask.mean().detach().to(torch.float32)).item())
                loss = F.mse_loss(V_t.to(torch.float32), value_target, reduction="none")
                loss = (loss * mask).sum() / mask.sum()
            else:
                self._last_gate_pass_ratio = 1.0
                loss = F.mse_loss(V_t.to(torch.float32), value_target, reduction="mean")

            return {"value_imag": loss * self.strategy.lambda_wm}

        except Exception as e:
            print(f"[WM] value_imag loss computation failed: {e}")
            return None

    # ------------------------------------------------------------------
    # Training step
    # ------------------------------------------------------------------

    def step(self, frames: int = 0) -> dict:
        """
        Run one WM training step if conditions are met.

        Returns metrics dict (empty if skipped or buffer too small).
        """
        self._epoch_count += 1
        self._epoch_since_last_update += 1

        # Check update frequency
        if self._epoch_since_last_update < self.config.update_every_n_epochs:
            return {}
        self._epoch_since_last_update = 0

        # Check buffer size
        if self.buffer.size < self.config.batch_size:
            return {}

        if not self.strategy.trains_wm:
            return {}

        metrics = self._train_step(frames)
        return metrics

    def _train_step(self, frames: int) -> dict:
        """Internal training step."""
        t0 = time.time()
        sample = self.buffer.sample(self.config.batch_size)
        obs_t = sample["obs_t"].to(self.device)
        action_t = sample["action_t"].to(self.device)
        reward_t = sample["reward_t"].to(self.device).squeeze(-1)
        done_t = sample["done_t"].to(self.device).squeeze(-1)
        failure_t = sample["failure_t"].to(self.device).squeeze(-1)
        # object state for delta targets
        obj_pos_t = sample["object_pos_t"].to(self.device)
        obj_pos_tp1 = sample["object_pos_tp1"].to(self.device)
        linvel_t = sample["object_linvel_t"].to(self.device)
        linvel_tp1 = sample["object_linvel_tp1"].to(self.device)
        angvel_t = sample["object_angvel_t"].to(self.device)
        angvel_tp1 = sample["object_angvel_tp1"].to(self.device)
        dhat_t = sample["d_hat_t"].to(self.device)
        dhat_tp1 = sample["d_hat_tp1"].to(self.device)

        # Delta targets
        target = {
            "object_pos_delta": obj_pos_tp1 - obj_pos_t,
            "d_hat_delta":      dhat_tp1 - dhat_t,
            "linvel_delta":     linvel_tp1 - linvel_t,
            "angvel_delta":     angvel_tp1 - angvel_t,
            "reward":           reward_t.unsqueeze(-1),
            "done":             done_t.unsqueeze(-1),
        }

        # WM forward
        pred = self.module.forward_mean(obs_t, action_t)

        # Compute dynamics loss
        wm_loss_obj = WMLoss(
            alpha_pos=self.config.alpha_pos,
            alpha_dhat=self.config.alpha_dhat,
            alpha_vel=self.config.alpha_vel,
            alpha_ang=self.config.alpha_ang,
            alpha_reward=self.config.alpha_reward,
            alpha_done=self.config.alpha_done,
        )
        loss_dynamics = wm_loss_obj.compute_dynamics_loss(pred, target, failure_t)

        # Total loss (value_imag is handled separately via get_aux_loss() in PPO backward)
        total = self.strategy.lambda_wm * loss_dynamics

        self.optimizer.zero_grad()
        total.backward()

        # Clip WM grads
        nn.utils.clip_grad_norm_(self.module.parameters(), self.config.max_grad_norm)

        self.optimizer.step()

        # Compute metrics
        with torch.no_grad():
            pos_err = (pred["object_pos_delta"] - target["object_pos_delta"]).norm(dim=-1).mean()
            dhat_err = (pred["d_hat_delta"] - target["d_hat_delta"]).abs().mean()
            reward_err = (pred["reward"].squeeze(-1) - reward_t).abs().mean()
            primary_unc, _ = compute_uncertainty(self.module, obs_t, action_t)
            unc_p50 = primary_unc.to(torch.float32).median().item()
            unc_p90 = primary_unc.to(torch.float32).quantile(0.9).item()
            unc_p95 = primary_unc.to(torch.float32).quantile(0.95).item()

        metrics = {
            "wm/loss_total":          total.item(),
            "wm/loss_dynamics":       loss_dynamics.item(),
            "wm/pred_error_pos":      pos_err.item(),
            "wm/pred_error_d_hat":   dhat_err.item(),
            "wm/pred_error_reward":   reward_err.item(),
            "wm/uncertainty_mean":    primary_unc.to(torch.float32).mean().item(),
            "wm/uncertainty_p50":    unc_p50,
            "wm/uncertainty_p90":    unc_p90,
            "wm/uncertainty_p95":    unc_p95,
            "wm/gate_pass_ratio":    float(self._last_gate_pass_ratio),
            "wm/value_imag_shared_encoder_grad_norm": float(self._last_shared_encoder_grad_norm),
            "wm/aux_value_head_grad_norm": float(self._last_aux_value_head_grad_norm),
            "wm/step_time_s":         time.time() - t0,
        }

        # Log to tensorboard
        if self.tensorboard is not None:
            for key, val in metrics.items():
                self.tensorboard.add_scalar(key, val, frames)

        return metrics

    def get_aux_value_head_state(self) -> Optional[dict]:
        """Get aux_value_head state dict for checkpoint saving."""
        if self.aux_value_head is not None:
            return self.aux_value_head.state_dict()
        return None