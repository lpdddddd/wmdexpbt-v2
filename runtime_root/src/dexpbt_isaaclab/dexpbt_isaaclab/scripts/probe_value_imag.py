#!/usr/bin/env python
"""value_imag_h1 feasibility probe — run on one checkpoint."""
import argparse, sys, torch, torch.nn as nn, torch.nn.functional as F

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, required=True)
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--seed", type=int, default=42)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym, yaml, importlib
from pathlib import Path
from importlib import resources
from rl_games.common import env_configurations, vecenv
from rl_games.torch_runner import Runner
from rl_games.algos_torch import torch_ext
from isaaclab_rl.rl_games import RlGamesGpuEnv, RlGamesVecEnvWrapper
import dexpbt_isaaclab

def _load_agent_cfg():
    cfg_file = resources.files("dexpbt_isaaclab.agents").joinpath("rl_games_ppo_cfg.yaml")
    with open(str(cfg_file)) as f:
        return yaml.safe_load(f)

def _load_env_cfg(task_name):
    spec = gym.spec(task_name)
    entry = spec.kwargs["env_cfg_entry_point"]
    module_path, cls_name = entry.rsplit(":", 1)
    mod = importlib.import_module(module_path)
    cfg_cls = getattr(mod, cls_name)
    return cfg_cls()

def probe():
    print("=" * 60)
    print("value_imag_h1 FEASIBILITY PROBE")
    print("=" * 60)

    env_cfg = _load_env_cfg(args_cli.task)
    agent_cfg = _load_agent_cfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.seed
    agent_cfg["params"]["seed"] = args_cli.seed
    agent_cfg["params"]["config"]["torch_compile"] = False

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    rl_device = agent_cfg["params"]["config"]["device"]
    clip_obs = agent_cfg["params"]["env"].get("clip_observations", float('inf'))
    clip_actions = agent_cfg["params"]["env"].get("clip_actions", float('inf'))
    env = RlGamesVecEnvWrapper(env, rl_device, clip_obs, clip_actions)

    vecenv.register("IsaacRlgWrapper",
        lambda cn, na, **kw: RlGamesGpuEnv(cn, na, **kw), )
    env_configurations.register("rlgpu",
        {"vecenv_type": "IsaacRlgWrapper", "env_creator": lambda **kw: env})

    agent_cfg["params"]["load_checkpoint"] = True
    agent_cfg["params"]["load_path"] = args_cli.checkpoint
    agent_cfg["params"]["config"]["num_actors"] = env.unwrapped.num_envs
    agent_cfg["params"]["config"]["player"] = {
        "deterministic": True, "games_num": 100000, "print_stats": False}

    print(f"[INFO] Loading: {args_cli.checkpoint}")

    # Build player (like run_play, but we want the model for gradient check)
    from rl_games.algos_torch import players
    from rl_games.algos_torch import network_builder as nb

    player_params = dict(agent_cfg['params'])
    player_params['seed'] = args_cli.seed
    player_params['normalize_input'] = True
    player_params['value_size'] = 1

    # Get obs shape
    env_info = env.get_env_info()
    obs_shape = tuple(env_info['observation_space'].shape)
    actions_num = int(env_info['action_space'].shape[0])

    # Build model using ModelBuilder (same as rl_games internal flow)
    # mb.load expects {model: {...}, network: {...}} structure
    from rl_games.algos_torch import model_builder
    mb = model_builder.ModelBuilder()
    # mb.load reads params['model'] and params['network'] from the passed dict
    load_params = {
        'model': agent_cfg['params'].get('model', {}),
        'network': agent_cfg['params']['network'],
    }
    model = mb.load(load_params)
    # model is ModelA2CContinuousLogStd - model.a2c_network only exists AFTER build() is called.
    # build() creates model.a2c_network.a2c_network (the inner nn.Module with actual layers).
    build_config = {
        'actions_num': actions_num,
        'input_shape': obs_shape,
        'num_seqs': 1,
        'value_size': 1,
        'normalize_value': player_params.get('normalize_value', False),
        'normalize_input': player_params.get('normalize_input', True),
    }
    model = model.build(build_config)
    model.to(rl_device)

    # After build(): model.a2c_network is the Network wrapper.
    # Checkpoint state dict uses 'a2c_network.xxx' keys on its inner a2c_network.
    # So we load into model.a2c_network (the wrapper), not its nested inner.
    inner_a2c = model.a2c_network

    # Checkpoint uses 'a2c_network.xxx' prefix; Network wrapper expects 'xxx' without prefix.
    # Strip the prefix when loading so keys match.
    checkpoint = torch_ext.load_checkpoint(args_cli.checkpoint)
    model_state = checkpoint['model']
    fixed_state = {}
    for k, v in model_state.items():
        if k.startswith('a2c_network.'):
            fixed_state[k[len('a2c_network.'):]] = v
        else:
            fixed_state[k] = v
    inner_a2c.load_state_dict(fixed_state, strict=False)
    if player_params.get('normalize_input'):
        if hasattr(inner_a2c, 'running_mean_std'):
            inner_a2c.running_mean_std.load_state_dict(checkpoint['model']['running_mean_std'], strict=False)
    inner_a2c.eval()

    print(f"  model type: {type(model).__name__}")
    a2c = inner_a2c  # use inner_a2c for structure probing
    print(f"  a2c type: {type(a2c).__name__}")
    print(f"  a2c_network type: {type(a2c).__name__}")

    # Build optimizer (same as A2CAgent.__init__) - use inner_a2c.parameters()
    lr = float(agent_cfg['params']['config'].get('learning_rate', '1e-4'))
    weight_decay = agent_cfg['params']['config'].get('weight_decay', 0.0)
    ppo_optimizer = torch.optim.Adam(inner_a2c.parameters(), lr=lr, eps=1e-8,
                                      weight_decay=weight_decay)

    print(f"  optimizer: {type(ppo_optimizer).__name__}")
    print(f"  param_groups: {len(ppo_optimizer.param_groups)}")
    total_params = sum(p.numel() for pg in ppo_optimizer.param_groups for p in pg['params'])
    print(f"  total params in optimizer: {total_params:,}")

    # === PROBE 1: model structure ===
    print("\n--- PROBE 1: model structure ---")
    print(f"  separate: {getattr(a2c, 'separate', 'N/A')}")
    print(f"  has_rnn: {getattr(a2c, 'has_rnn', 'N/A')}")
    print(f"  has_cnn: {getattr(a2c, 'has_cnn', 'N/A')}")
    print(f"  mlp units: {getattr(a2c, 'units', 'N/A')}")
    print(f"  rnn_units: {getattr(a2c, 'rnn_units', 'N/A')}")
    print(f"  rnn_before_mlp: {getattr(a2c, 'is_rnn_before_mlp', 'N/A')}")

    # === PROBE 2: accessor ===
    print("\n--- PROBE 2: accessor availability ---")
    attrs = ['actor_mlp', 'actor_cnn', 'rnn', 'value', 'value_act', 'mu', 'mu_act', 'sigma']
    for attr in attrs:
        has = hasattr(a2c, attr)
        print(f"  has {attr}: {has}", end="")
        if has:
            v = getattr(a2c, attr)
            if isinstance(v, nn.Module):
                print(f" -> {type(v).__name__}")
            elif isinstance(v, nn.Parameter):
                print(f" -> nn.Parameter({v.shape})")
            else:
                print(f" -> {type(v).__name__}")
        else:
            print()

    # === PROBE 3: get shared feature h ===
    print("\n--- PROBE 3: get shared feature h ---")
    obs_dim = int(agent_cfg["params"]["env"]["clip_observations"])
    dummy_obs = torch.randn(args_cli.num_envs, obs_dim, device=rl_device)

    # Normalize like rl_games does
    if player_params.get('normalize_input'):
        dummy_obs = inner_a2c.running_mean_std(dummy_obs)

    h_holder = {}
    def capture_h(module, input):
        h_holder['h'] = input[0].clone()
    handle = a2c.value.register_forward_pre_hook(capture_h)

    with torch.no_grad():
        result = a2c({'obs': dummy_obs, 'is_train': False, 'rnn_states': None})
    handle.remove()

    h = h_holder['h']
    print(f"  h shape: {h.shape}")
    print(f"  h mean: {h.mean().item():.4f}, std: {h.std().item():.4f}")

    shared_dim = a2c.value.in_features
    print(f"  a2c.value.in_features: {shared_dim}")
    print(f"  MATCH (shared_dim == h.shape[-1]): {shared_dim == h.shape[-1]}")

    # === PROBE 4: register aux_value_head ===
    print("\n--- PROBE 4: register aux_value_head ---")
    aux_value_head = nn.Sequential(
        nn.Linear(shared_dim, 128),
        nn.Tanh(),
        nn.Linear(128, 1)
    ).to(rl_device)
    a2c.aux_value_head = aux_value_head
    print(f"  registered to a2c_network: {'aux_value_head' in dict(a2c.named_children())}")
    n_aux = sum(p.numel() for p in aux_value_head.parameters())
    print(f"  aux params: {n_aux:,}")

    # === PROBE 5: add to optimizer ===
    print("\n--- PROBE 5: add to optimizer ---")
    n_pg_before = len(ppo_optimizer.param_groups)
    ppo_optimizer.add_param_group({
        'params': list(aux_value_head.parameters()),
        'lr': lr,
        'name': 'aux_value_head',
    })
    n_pg_after = len(ppo_optimizer.param_groups)
    print(f"  param_groups: {n_pg_before} -> {n_pg_after}")
    total_after = sum(p.numel() for pg in ppo_optimizer.param_groups for p in pg['params'])
    print(f"  total params after: {total_after:,} (+{total_after - total_params:,})")

    # === PROBE 6: gradient check ===
    print("\n--- PROBE 6: gradient check ---")
    B = min(args_cli.num_envs, 16)
    h_b = h[:B].clone().detach().requires_grad_(True)

    V_aux = aux_value_head(h_b).squeeze(-1)
    target = V_aux.detach() + 0.1
    aux_loss = ((V_aux - target) ** 2).mean()

    ppo_optimizer.zero_grad()
    aux_loss.backward()

    grad_info = {}
    for name, param in a2c.named_parameters():
        gn = param.grad.norm().item() if param.grad is not None else 0.0
        grad_info[f"a2c.{name}"] = gn
    for name, param in aux_value_head.named_parameters():
        gn = param.grad.norm().item() if param.grad is not None else 0.0
        grad_info[f"aux.{name}"] = gn

    print("  Gradient norms (top 15 non-zero):")
    sorted_grads = sorted(grad_info.items(), key=lambda x: -x[1])
    for name, gn in sorted_grads[:15]:
        if gn > 0:
            print(f"    {name}: {gn:.6f}")

    has_aux_grad = aux_value_head[0].weight.grad is not None
    has_mlp_grad = a2c.actor_mlp[0].weight.grad is not None
    has_rnn_grad = None
    if a2c.has_rnn:
        has_rnn_grad = a2c.rnn.weight_hh_l0.grad is not None
    no_mu_grad = a2c.mu.weight.grad is None

    print(f"\n  === VERIFICATION ===")
    print(f"  aux_value_head has grad:       {has_aux_grad} {'✓' if has_aux_grad else '✗'}")
    print(f"  actor_mlp has grad:           {has_mlp_grad} {'✓' if has_mlp_grad else '✗'}")
    print(f"  rnn has grad:                {has_rnn_grad} {'✓' if has_rnn_grad else '✗'}")
    print(f"  mu (actor head) has NO grad:  {no_mu_grad} {'✓' if no_mu_grad else '✗'}")

    print("\n" + "=" * 60)
    if has_aux_grad and has_mlp_grad and no_mu_grad:
        print("RESULT: value_imag_h1 FEASIBLE — Plan A works!")
        print("  aux_value_head -> shared encoder gradient path confirmed.")
    else:
        print("RESULT: Plan A failed — use failure-risk reward penalty (Plan B)")
    print("=" * 60)

    env.close()
    simulation_app.close()

if __name__ == "__main__":
    probe()