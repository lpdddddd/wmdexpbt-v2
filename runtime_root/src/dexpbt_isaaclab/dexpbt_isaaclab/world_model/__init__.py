"""
World Model (WM) module for DexPBT.
Exports all WM sub-modules.
"""

from .wm_config import WMConfig
from .wm_module import WorldModel
from .wm_buffer import WMBuffer
from .wm_loss import WMLoss
from .wm_module import compute_uncertainty
from .wm_trainer import WMTrainer, WMStrategy, STRATEGIES, get_strategy
from .wm_posterior_selector import (
    BayesianPosteriorSelector,
    SharedPosteriorSelector,
    PosteriorState,
)
from .wm_checkpoint import WMCheckpointManager
from .wm_data_collector import WMDataCollector, WMInterceptorEnv

__all__ = [
    "WMConfig",
    "WorldModel",
    "WMBuffer",
    "WMLoss",
    "compute_uncertainty",
    "WMTrainer",
    "WMStrategy",
    "STRATEGIES",
    "get_strategy",
    "BayesianPosteriorSelector",
    "SharedPosteriorSelector",
    "PosteriorState",
    "WMCheckpointManager",
    "WMDataCollector",
    "WMInterceptorEnv",
]