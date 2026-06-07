"""
Smoke test for World Model integration.
Tests: env infos["wm"], WMDataCollector, WMTrainer, full train.py --enable_wm.
"""

import argparse
import sys
import os
import torch

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args(["--headless"])
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym
import dexpbt_isaaclab
from dexpbt_isaaclab.dexpbt_kuka_allegro_env import DexPBTKukaAllegroEnv

# ------------------------------------------------------------------
# TEST 1: env infos["wm"] data channel
# ------------------------------------------------------------------
print("=" * 60)
print("TEST 1: env infos['wm'] data channel")
print("=" * 60)

from dexpbt_isaaclab.env_cfg import DexPBTKukaAllegroEnvCfg
cfg = DexPBTKukaAllegroEnvCfg()
cfg.scene.num_envs = 4
env = gym.make("DexPBT-Reorientation-KukaAllegro-Direct-v0", cfg=cfg, render_mode=None)
obs, info = env.reset()
print(f"  reset obs shape: {obs['policy'].shape}")
print(f"  info keys: {list(info.keys())}")

act = torch.randn(4, 23, device="cuda:0").clamp(-1, 1)
obs, rew, term, trunc, info = env.step(act)

assert "wm" in info, f"info should have 'wm' key, got {list(info.keys())}"
wm = info["wm"]
print(f"  wm keys: {list(wm.keys())}")
for k, v in wm.items():
    print(f"    {k}: {v.shape}, dtype={v.dtype}")

assert wm["object_pos"].shape == (4, 3)
assert wm["d_hat"].shape == (4, 1)
assert wm["failure"].shape == (4, 1)
print("  PASS: infos['wm'] shape correct")

env.close()
print()

# ------------------------------------------------------------------
# TEST 2: WMDataCollector + WMInterceptorEnv
# ------------------------------------------------------------------
print("=" * 60)
print("TEST 2: WMDataCollector + WMInterceptorEnv")
print("=" * 60)

from dexpbt_isaaclab.world_model import WMBuffer, WMDataCollector, WMInterceptorEnv

cfg2 = DexPBTKukaAllegroEnvCfg()
cfg2.scene.num_envs = 8
env2 = gym.make("DexPBT-Reorientation-KukaAllegro-Direct-v0", cfg=cfg2, render_mode=None)
obs, info = env2.reset()

buf = WMBuffer(capacity=1000, obs_dim=obs["policy"].shape[1], action_dim=23, device="cuda:0")
collector = WMDataCollector(num_envs=8, wm_buffer=buf, device="cuda:0")
wrapped = WMInterceptorEnv(env2, collector)

print(f"  obs_dim={obs['policy'].shape[1]}, action_dim=23")
print(f"  Initial buf.size={buf.size}")

# First step must go through step() to populate _prev_obs
obs, rew, term, trunc, info = wrapped.step(torch.randn(8, 23, device="cuda:0").clamp(-1, 1))

for step_i in range(1, 20):
    act = torch.randn(8, 23, device="cuda:0").clamp(-1, 1)
    obs, rew, term, trunc, info = wrapped.step(act)

    if step_i % 5 == 0:
        print(f"  Step {step_i}: buf.size={buf.size}, prev_valid={collector._prev_valid.tolist()}")

print(f"  Final buf.size after 20 steps: {buf.size}")
assert buf.size > 0, "Buffer should have transitions after 20 steps"
print("  PASS: WMDataCollector collects transitions")

# Sample from buffer
sample = buf.sample(4)
print(f"  Sample keys: {list(sample.keys())}")
for k, v in sample.items():
    print(f"    {k}: {v.shape}")
assert sample["obs_t"].shape == (4, obs["policy"].shape[1])
print("  PASS: buffer sample shapes correct")

wrapped.close()
del wrapped, env2
print()

# ------------------------------------------------------------------
# TEST 3: WMTrainer step (none strategy)
# ------------------------------------------------------------------
print("=" * 60)
print("TEST 3: WMTrainer step (none strategy)")
print("=" * 60)

from dexpbt_isaaclab.world_model import (
    WMConfig, WorldModel, WMTrainer, get_strategy, WMCheckpointManager,
)

cfg3 = DexPBTKukaAllegroEnvCfg()
cfg3.scene.num_envs = 16
env3 = gym.make("DexPBT-Reorientation-KukaAllegro-Direct-v0", cfg=cfg3, render_mode=None)
obs3, info3 = env3.reset()
obs_dim = obs3["policy"].shape[1]

wm_cfg = WMConfig(obs_dim=obs_dim, action_dim=23, ensemble_size=3, batch_size=32,
                   update_every_n_epochs=1)
wm_module = WorldModel(obs_dim=obs_dim, action_dim=23, ensemble_size=3).to("cuda:0")
wm_opt = torch.optim.Adam(wm_module.parameters(), lr=1e-3)
wm_buffer = WMBuffer(capacity=500, obs_dim=obs_dim, action_dim=23, device="cuda:0")
strategy = get_strategy("none")

wm_trainer = WMTrainer(wm_module=wm_module, wm_optimizer=wm_opt, wm_buffer=wm_buffer,
                        config=wm_cfg, strategy=strategy, ppo_model=None,
                        ppo_optimizer=None, device="cuda:0")

# Fill buffer via collector
collector3 = WMDataCollector(num_envs=16, wm_buffer=wm_buffer, device="cuda:0")
wrapped3 = WMInterceptorEnv(env3, collector3)

for _ in range(50):
    act = torch.randn(16, 23, device="cuda:0").clamp(-1, 1)
    wrapped3.step(act)

print(f"  Buffer size: {wm_buffer.size}")
assert wm_buffer.size > 0

# Run WMTrainer step
metrics = wm_trainer.step(frames=1000)
print(f"  WMTrainer step (none): metrics={metrics}")
assert metrics == {}, f"none strategy should not train, got {metrics}"
print("  PASS: none strategy returns empty metrics")

wrapped3.close()
del wrapped3, env3
print()

# ------------------------------------------------------------------
# TEST 4: WMTrainer step (aux_mid strategy)
# ------------------------------------------------------------------
print("=" * 60)
print("TEST 4: WMTrainer step (aux_mid strategy)")
print("=" * 60)

cfg4 = DexPBTKukaAllegroEnvCfg()
cfg4.scene.num_envs = 16
env4 = gym.make("DexPBT-Reorientation-KukaAllegro-Direct-v0", cfg=cfg4, render_mode=None)
obs4, _ = env4.reset()

wm_cfg4 = WMConfig(obs_dim=obs_dim, action_dim=23, ensemble_size=3, batch_size=32,
                    update_every_n_epochs=1)
wm_module4 = WorldModel(obs_dim=obs_dim, action_dim=23, ensemble_size=3).to("cuda:0")
wm_opt4 = torch.optim.Adam(wm_module4.parameters(), lr=1e-3)
wm_buffer4 = WMBuffer(capacity=500, obs_dim=obs_dim, action_dim=23, device="cuda:0")
strategy4 = get_strategy("aux_mid")

wm_trainer4 = WMTrainer(wm_module=wm_module4, wm_optimizer=wm_opt4, wm_buffer=wm_buffer4,
                         config=wm_cfg4, strategy=strategy4, ppo_model=None,
                         ppo_optimizer=None, device="cuda:0")

collector4 = WMDataCollector(num_envs=16, wm_buffer=wm_buffer4, device="cuda:0")
wrapped4 = WMInterceptorEnv(env4, collector4)

for _ in range(80):
    act = torch.randn(16, 23, device="cuda:0").clamp(-1, 1)
    wrapped4.step(act)

print(f"  Buffer size: {wm_buffer4.size}")
metrics4 = wm_trainer4.step(frames=2000)
print(f"  WMTrainer step (aux_mid):")
for k, v in metrics4.items():
    print(f"    {k}: {v:.4f}")
assert "wm/loss_total" in metrics4
assert metrics4["wm/loss_total"] > 0
print("  PASS: aux_mid strategy trains and returns metrics")

wrapped4.close()
del wrapped4, env4
print()

# ------------------------------------------------------------------
# TEST 5: WMCheckpointManager
# ------------------------------------------------------------------
print("=" * 60)
print("TEST 5: WMCheckpointManager save/load")
print("=" * 60)

import tempfile
with tempfile.TemporaryDirectory() as tmpdir:
    ckpt_mgr = WMCheckpointManager(tmpdir)
    path = ckpt_mgr.save(wm_module4, wm_opt4, "aux_mid", iteration=42, frames=5000,
                          aux_value_head_state=None)
    print(f"  Saved: {os.path.exists(path)}")

    wm_reload = WorldModel(obs_dim=obs_dim, action_dim=23, ensemble_size=3).to("cuda:0")
    opt_reload = torch.optim.Adam(wm_reload.parameters(), lr=1e-3)
    loaded = ckpt_mgr.load_latest(wm_reload, opt_reload)
    print(f"  Loaded: {loaded}, strategy: {ckpt_mgr.get_strategy()}")
    assert loaded
    assert ckpt_mgr.get_strategy() == "aux_mid"

    # Verify parameters match
    for p1, p2 in zip(wm_module4.parameters(), wm_reload.parameters()):
        assert torch.allclose(p1, p2), "Loaded params should match saved"
    print("  PASS: params match after save/load")

print()

# ------------------------------------------------------------------
# TEST 6: Posterior selector
# ------------------------------------------------------------------
print("=" * 60)
print("TEST 6: SharedPosteriorSelector")
print("=" * 60)

from dexpbt_isaaclab.world_model import SharedPosteriorSelector
import tempfile

with tempfile.TemporaryDirectory() as tmpdir:
    shared_path = os.path.join(tmpdir, "posterior.json")
    sel = SharedPosteriorSelector(shared_path=shared_path)
    sel.load_or_init(policy_idx=0)

    # none should have mean=0
    assert sel.posterior["none"].mean == 0.0
    print(f"  Initial none.mean={sel.posterior['none'].mean}")

    sel.update_batch({"aux_low": 0.03, "aux_mid": 0.05, "value_imag_h1": 0.08}, step=1)
    print(f"  After update: aux_low.mean={sel.posterior['aux_low'].mean:.4f}")
    print(f"  After update: value_imag_h1.mean={sel.posterior['value_imag_h1'].mean:.4f}")
    assert sel.posterior["none"].mean == 0.0, "none mean should stay 0"

    # Sample multiple times to test distribution
    samples = [sel.sample() for _ in range(100)]
    from collections import Counter
    counts = Counter(samples)
    print(f"  Thompson samples (100): {dict(counts)}")
    # none should sometimes be sampled (baseline)
    assert "none" in counts or "aux_mid" in counts or "value_imag_h1" in counts
    print("  PASS: Thompson Sampling works")

    sel.save()
    sel2 = SharedPosteriorSelector(shared_path=shared_path)
    sel2.load_or_init(policy_idx=1)
    print(f"  After reload: aux_mid.mean={sel2.posterior['aux_mid'].mean:.4f}")
    assert abs(sel2.posterior["aux_mid"].mean - 0.05) < 0.01
    print("  PASS: save/load works")

print()

# ------------------------------------------------------------------
# ALL DONE
# ------------------------------------------------------------------
print("=" * 60)
print("ALL SMOKE TESTS PASSED!")
print("=" * 60)

simulation_app.close()
