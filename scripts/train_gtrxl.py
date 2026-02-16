#!/usr/bin/env python3
"""Train a GTrXL policy on classic control environments.

Supported environments:
  cartpole    CartPoleMaskedVelocity-v1  (discrete, 2-dim obs, requires memory)
  acrobot     Acrobot-v1                 (discrete, 6-dim obs, swing-up task)
  lunarlander LunarLander-v2             (discrete, 8-dim obs, landing task)

Usage:
    python scripts/train_gtrxl.py                  # default: cartpole
    python scripts/train_gtrxl.py acrobot
    python scripts/train_gtrxl.py lunarlander

Monitor training:
    tensorboard --logdir runs/
"""

import argparse
import os
import sys

import torch

# ---------------------------------------------------------------------------
# Dependency check — ray is required for the RayVecEnv but is not in
# setup.py install_requires.
# ---------------------------------------------------------------------------
try:
    import ray
except ImportError:
    print(
        "ERROR: 'ray' is required but not installed.\n"
        "Install it with:  uv pip install 'ray[default]'\n"
        "              or:  pip install 'ray[default]'"
    )
    sys.exit(1)

from rl_games.torch_runner import Runner

# ---------------------------------------------------------------------------
# Per-environment configs (self-contained, no YAML needed)
# ---------------------------------------------------------------------------
# Shared GTrXL network config — all envs use the same architecture and
# discrete action space.
_NETWORK = {
    "name": "gtrxl_actor_critic",
    "space": {"discrete": {}},
    "gtrxl": {
        "embedding_dim": 64,
        "num_layers": 2,
        "num_heads": 4,
        "memory_length": 32,
        "gru_gate_bias": 2.0,
        "dropout": 0.0,
    },
    "initializer": {"name": "default"},
}

ENVS = {
    "cartpole": {
        "params": {
            "algo": {"name": "a2c_discrete"},
            "model": {"name": "discrete_a2c"},
            "load_checkpoint": False,
            "network": _NETWORK,
            "config": {
                "env_name": "CartPoleMaskedVelocity-v1",
                "reward_shaper": {"scale_value": 0.1},
                "normalize_advantage": True,
                "gamma": 0.99,
                "tau": 0.9,
                "learning_rate": 3e-4,
                "name": "cartpole_gtrxl",
                "score_to_win": 490,
                "grad_norm": 0.5,
                "entropy_coef": 0.01,
                "truncate_grads": True,
                "e_clip": 0.2,
                "clip_value": True,
                "num_actors": 8,
                "horizon_length": 256,
                "minibatch_size": 1024,
                "mini_epochs": 4,
                "critic_coef": 1,
                "lr_schedule": None,
                "kl_threshold": 0.008,
                "normalize_input": False,
                "seq_length": 8,
                "zero_rnn_on_done": True,
                "max_epochs": 500,
                "env_config": {"seed": 42},
            },
        },
    },
    "lunarlander": {
        "params": {
            "algo": {"name": "a2c_discrete"},
            "model": {"name": "discrete_a2c"},
            "load_checkpoint": False,
            "network": _NETWORK,
            "config": {
                "env_name": "LunarLander-v2",
                "reward_shaper": {"scale_value": 0.1},
                "normalize_advantage": True,
                "gamma": 0.99,
                "tau": 0.95,
                "learning_rate": 5e-4,
                "name": "lunarlander_gtrxl",
                "score_to_win": 200,
                "grad_norm": 1.0,
                "entropy_coef": 0.01,
                "truncate_grads": True,
                "e_clip": 0.2,
                "clip_value": True,
                "num_actors": 8,
                "horizon_length": 128,
                "minibatch_size": 512,
                "mini_epochs": 4,
                "critic_coef": 1,
                "lr_schedule": None,
                "kl_threshold": 0.008,
                "normalize_input": False,
                "seq_length": 8,
                "zero_rnn_on_done": True,
                "max_epochs": 500,
                "env_config": {"seed": 42},
            },
        },
    },
    "acrobot": {
        "params": {
            "algo": {"name": "a2c_discrete"},
            "model": {"name": "discrete_a2c"},
            "load_checkpoint": False,
            "network": _NETWORK,
            "config": {
                "env_name": "Acrobot-v1",
                "reward_shaper": {"scale_value": 1.0},
                "normalize_advantage": True,
                "gamma": 0.99,
                "tau": 0.95,
                "learning_rate": 3e-4,
                "name": "acrobot_gtrxl",
                "score_to_win": -80,
                "grad_norm": 1.0,
                "entropy_coef": 0.01,
                "truncate_grads": True,
                "e_clip": 0.2,
                "clip_value": True,
                "num_actors": 8,
                "horizon_length": 256,
                "minibatch_size": 1024,
                "mini_epochs": 4,
                "critic_coef": 1,
                "lr_schedule": None,
                "kl_threshold": 0.008,
                "normalize_input": False,
                "seq_length": 8,
                "zero_rnn_on_done": True,
                "max_epochs": 500,
                "env_config": {"seed": 42},
            },
        },
    },
}


def main():
    parser = argparse.ArgumentParser(description="Train GTrXL on classic control envs")
    parser.add_argument(
        "env",
        nargs="?",
        default="cartpole",
        choices=ENVS.keys(),
        help="environment to train on (default: cartpole)",
    )
    args = parser.parse_args()

    config = ENVS[args.env]
    cfg = config["params"]["config"]
    gtrxl = config["params"]["network"]["gtrxl"]

    os.makedirs("nn", exist_ok=True)
    os.makedirs("runs", exist_ok=True)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    cfg["device"] = device

    print("=" * 60)
    print(f"  GTrXL Training — {cfg['env_name']}")
    print("=" * 60)
    print()
    print(f"  Device:        {device}")
    print(f"  Environment:   {cfg['env_name']}")
    print(f"  Architecture:  GTrXL ({gtrxl['num_layers']} layers, {gtrxl['num_heads']} heads, d={gtrxl['embedding_dim']})")
    print(f"  Memory length: {gtrxl['memory_length']}")
    print(f"  Seq length:    {cfg['seq_length']}")
    print(f"  Num actors:    {cfg['num_actors']}")
    print(f"  Max epochs:    {cfg['max_epochs']}")
    print(f"  Score to win:  {cfg['score_to_win']}")
    print()
    print("  Monitor with:  tensorboard --logdir runs/")
    print("=" * 60)
    print()

    ray.init(object_store_memory=1024 * 1024 * 1000)

    try:
        runner = Runner()
        runner.load(config)

        # Create agent and train directly, skipping torch.compile which is
        # extremely slow on CPU and adds unnecessary overhead for this small model.
        agent = runner.algo_factory.create(
            runner.algo_name, base_name="run", params=runner.params
        )
        agent.train()
    finally:
        ray.shutdown()


if __name__ == "__main__":
    main()
