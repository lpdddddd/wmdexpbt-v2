#!/usr/bin/env python3
"""Minimal repro for value_imag rnn_states batch mismatch (no Isaac sim)."""
import sys
import torch

sys.path.insert(0, "/root/autodl-tmp/dexpbt_isaaclab")

from rl_games.algos_torch.network_builder import A2CBuilder

# Mirror smoke agent.yaml network block
network_params = {
    "separate": False,
    "space": {
        "continuous": {
            "mu_activation": "None",
            "sigma_activation": "None",
            "mu_init": {"name": "default"},
            "sigma_init": {"name": "const_initializer", "val": 0},
            "fixed_sigma": True,
        }
    },
    "mlp": {
        "units": [768, 512, 256],
        "activation": "elu",
        "d2rl": False,
        "initializer": {"name": "default"},
    },
    "rnn": {
        "name": "lstm",
        "units": 768,
        "layers": 1,
        "before_mlp": True,
        "layer_norm": True,
    },
}

builder = A2CBuilder()
builder.load(network_params)
a2c = builder.build(
    "actor_critic",
    actions_num=23,
    input_shape=(110,),
    value_size=1,
    num_seqs=2048,
)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
a2c = a2c.to(device)
a2c.eval()

batch_size = 512
obs = torch.randn(batch_size, 110, device=device)

# Path A: no rnn_states (current wm_trainer fallback)
batch_none = {
    "is_train": True,
    "prev_actions": torch.zeros(batch_size, 23, device=device),
    "obs": obs,
}
print("=== forward without rnn_states ===")
try:
    a2c(batch_none)
    print("OK")
except Exception as e:
    print(f"FAIL: {e!r}")

# Path B: zero rnn_states matching batch_size
def make_zero_states(bs: int):
    h = torch.zeros(1, bs, 768, device=device)
    c = torch.zeros(1, bs, 768, device=device)
    return (h, c)

batch_ok = dict(batch_none)
batch_ok["rnn_states"] = make_zero_states(batch_size)
print("=== forward with rnn_states (1,512,768) ===")
try:
    a2c(batch_ok)
    print("OK")
except Exception as e:
    print(f"FAIL: {e!r}")

# Path C: default num_seqs=2048 states
batch_bad = dict(batch_none)
batch_bad["rnn_states"] = a2c.get_default_rnn_state()
shapes = [tuple(t.shape) for t in batch_bad["rnn_states"]]
print(f"=== forward with default rnn_states {shapes} ===")
try:
    a2c(batch_bad)
    print("OK")
except Exception as e:
    print(f"FAIL: {e!r}")

# Path D: wm_trainer fix helper (post-fix verification)
from dexpbt_isaaclab.world_model.wm_trainer import WMTrainer

batch_fixed = dict(batch_none)
batch_fixed["seq_length"] = 1
batch_fixed["rnn_states"] = WMTrainer._zero_rnn_states_for_batch(a2c, batch_size, device)
print("=== forward with WMTrainer._zero_rnn_states_for_batch ===")
try:
    a2c(batch_fixed)
    print("OK")
except Exception as e:
    print(f"FAIL: {e!r}")
