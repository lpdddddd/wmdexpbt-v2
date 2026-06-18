"""Online Elite-AWR (Strategy Zoo v17) — independent auxiliary loss module."""

from dexpbt_isaaclab.awr.elite_awr import (
    EliteAWRBridge,
    EliteAWRBuffer,
    EliteAWRConfig,
    EliteAWREntry,
    awr_active_from_args,
    build_elite_awr_config_from_args,
    compute_elite_awr_loss,
    compute_elite_awr_quality,
    load_awr_sidecar,
    patch_model_get_aux_loss_for_awr,
    save_awr_sidecar,
    save_awr_sidecar_topk,
)
from dexpbt_isaaclab.awr.elite_awr_interceptor import EliteAWRInterceptorEnv

__all__ = [
    "EliteAWRBridge",
    "EliteAWRBuffer",
    "EliteAWRConfig",
    "EliteAWREntry",
    "EliteAWRInterceptorEnv",
    "awr_active_from_args",
    "build_elite_awr_config_from_args",
    "compute_elite_awr_loss",
    "compute_elite_awr_quality",
    "load_awr_sidecar",
    "save_awr_sidecar",
    "save_awr_sidecar_topk",
]
