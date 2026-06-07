"""
World Model configuration dataclass.
Loads from YAML or defaults.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class WMConfig:
    """Configuration for the World Model."""

    # --- Architecture ---
    obs_dim: int = 92          # policy obs dim (92 for regrasping/throw, 110 for reorientation)
    action_dim: int = 23       # 7 arm + 16 hand
    hidden_sizes: list = field(default_factory=lambda: [512, 256, 128])
    ensemble_size: int = 3

    # --- Loss weights ---
    alpha_pos: float = 1.0
    alpha_dhat: float = 0.5
    alpha_vel: float = 0.5
    alpha_ang: float = 0.3
    alpha_reward: float = 0.1
    alpha_done: float = 0.5

    # --- Training ---
    lr: float = 1e-3
    weight_decay: float = 0.0
    batch_size: int = 512
    max_grad_norm: float = 5.0

    # --- Frequency ---
    update_every_n_epochs: int = 2   # every 2 PPO epochs

    # --- value_imag specific ---
    value_imag_gamma: float = 0.99

    @classmethod
    def from_yaml(cls, path: str) -> "WMConfig":
        """Load from YAML file."""
        import yaml
        with open(path) as f:
            data = yaml.safe_load(f)
        # flatten nested dict if needed
        if "wm" in data:
            data = data["wm"]
        return cls(**{k: v for k, v in data.items() if hasattr(cls, k)})