#!/usr/bin/env python3
"""Post-fix verification: _get_encoder_features + value_imag path (no Isaac)."""
import sys

import torch
import torch.nn as nn

sys.path.insert(0, "/root/autodl-tmp/dexpbt_isaaclab")

from rl_games.algos_torch.network_builder import A2CBuilder
from dexpbt_isaaclab.world_model.wm_trainer import WMTrainer

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
a2c.aux_value_head = nn.Linear(256, 1).to(device)

class _FakePPO:
    def __init__(self, net):
        self.a2c_network = net

class _StubTrainer:
    _value_imag_enabled = True
    aux_value_head = a2c.aux_value_head
    ppo_model = _FakePPO(a2c)

    _get_a2c_network = WMTrainer._get_a2c_network
    _zero_rnn_states_for_batch = staticmethod(WMTrainer._zero_rnn_states_for_batch)
    _get_encoder_features = WMTrainer._get_encoder_features

stub = _StubTrainer()
obs = torch.randn(512, 110, device=device, requires_grad=True)

h = stub._get_encoder_features(obs, detach=False)
assert h.shape == (512, 256), h.shape
assert h.requires_grad
V = stub.aux_value_head(h).squeeze(-1)
loss = V.pow(2).mean()
loss.backward()
print("PASS: encoder forward + backward", float(loss))

h2 = stub._get_encoder_features(obs.detach(), detach=True)
assert not h2.requires_grad
print("PASS: detach path", h2.shape)
