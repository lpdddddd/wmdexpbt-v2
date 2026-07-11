"""DexPBT Kuka-Allegro tasks for Isaac Lab."""

import gymnasium as gym

from . import agents

# ---------------------------------------------------------------------------
# Gymnasium registration
# ---------------------------------------------------------------------------

gym.register(
    id="DexPBT-Reorientation-KukaAllegro-Direct-v0",
    entry_point="dexpbt_isaaclab.dexpbt_kuka_allegro_env:DexPBTReorientationEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "dexpbt_isaaclab.env_cfg:DexPBTReorientationEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
    },
)

gym.register(
    id="DexPBT-Regrasping-KukaAllegro-Direct-v0",
    entry_point="dexpbt_isaaclab.dexpbt_kuka_allegro_env:DexPBTRegraspingEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "dexpbt_isaaclab.env_cfg:DexPBTRegraspingEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
    },
)

gym.register(
    id="DexPBT-Throw-KukaAllegro-Direct-v0",
    entry_point="dexpbt_isaaclab.dexpbt_kuka_allegro_env:DexPBTThrowEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "dexpbt_isaaclab.env_cfg:DexPBTThrowEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
    },
)
