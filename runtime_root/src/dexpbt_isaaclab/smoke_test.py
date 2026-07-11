"""
Smoke test: create env, reset, step.

Requirements:
  - Isaac Sim 5.1+ (pip install isaacsim-rl)
  - Isaac Lab 2.3+ (pip install -e source/isaaclab ... etc)
  - PyTorch with sm_120 (for RTX 5090): pip install --pre torch --index-url .../cu128
  - Vulkan loader >= 1.4 (matching your NVIDIA driver, needed for GPU PhysX)
  - dexpbt_isaaclab installed (pip install -e .)
"""

import argparse
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
_smoke_argv = ["--headless"]
if os.environ.get("DEXPBT_KIT_ARGS"):
    _smoke_argv += ["--kit_args", os.environ["DEXPBT_KIT_ARGS"]]
args = parser.parse_args(_smoke_argv)
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import torch
import gymnasium as gym

import dexpbt_isaaclab  # noqa: registers envs
from dexpbt_isaaclab.env_cfg import DexPBTReorientationEnvCfg

NUM_ENVS = 4

cfg = DexPBTReorientationEnvCfg()
cfg.scene.num_envs = NUM_ENVS
print(f"obs={cfg.observation_space}, act={cfg.action_space}, num_envs={cfg.scene.num_envs}")

try:
    env = gym.make("DexPBT-Reorientation-KukaAllegro-Direct-v0", cfg=cfg)
    print(f"Env created: {env}")

    obs, info = env.reset()
    print(f"Reset obs shape: {obs['policy'].shape}")

    for i in range(5):
        act = torch.randn(NUM_ENVS, 23, device="cuda:0").clamp(-1, 1)
        obs, rew, term, trunc, info = env.step(act)

    print(f"Step 5: obs={obs['policy'].shape}, rew={rew.shape}, term={term.shape}")
    print(f"Reward sample: {rew}")
    env.close()
    print("FULL SMOKE TEST PASSED!")
except Exception as e:
    print(f"SMOKE TEST ERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

simulation_app.close()
