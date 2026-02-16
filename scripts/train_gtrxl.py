#!/usr/bin/env python3
"""Train a GTrXL policy on CartPole with masked velocity.

This environment masks out velocity observations, forcing the agent to infer
velocity from position history.  A memoryless (MLP) policy cannot solve it,
making it an ideal test for the GTrXL architecture.

Usage:
    python scripts/train_gtrxl.py

Monitor training:
    tensorboard --logdir runs/
"""

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
# Training configuration (self-contained, no YAML needed)
# ---------------------------------------------------------------------------
CONFIG = {
    "params": {
        "algo": {
            "name": "a2c_discrete",
        },
        "model": {
            "name": "discrete_a2c",
        },
        "load_checkpoint": False,
        "network": {
            "name": "gtrxl_actor_critic",
            "space": {
                "discrete": {},
            },
            "gtrxl": {
                "embedding_dim": 64,
                "num_layers": 2,
                "num_heads": 4,
                "memory_length": 32,
                "gru_gate_bias": 2.0,
                "dropout": 0.0,
            },
            "initializer": {
                "name": "default",
            },
        },
        "config": {
            "env_name": "CartPoleMaskedVelocity-v1",
            "reward_shaper": {
                "scale_value": 0.1,
            },
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
            "env_config": {
                "seed": 42,
            },
        },
    },
}


def main():
    os.makedirs("nn", exist_ok=True)
    os.makedirs("runs", exist_ok=True)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    CONFIG["params"]["config"]["device"] = device

    print("=" * 60)
    print("  GTrXL Training — CartPoleMaskedVelocity-v1")
    print("=" * 60)
    print()
    print(f"  Device:        {device}")
    print("  Environment:   CartPoleMaskedVelocity-v1")
    print("  Architecture:  GTrXL (2 layers, 4 heads, d=64)")
    print("  Memory length: 32")
    print("  Seq length:    8")
    print("  Num actors:    8")
    print("  Max epochs:    500")
    print("  Score to win:  490")
    print()
    print("  Monitor with:  tensorboard --logdir runs/")
    print("=" * 60)
    print()

    ray.init(object_store_memory=1024 * 1024 * 1000)

    try:
        runner = Runner()
        runner.load(CONFIG)

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
