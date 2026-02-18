#!/usr/bin/env python3
"""Export a trained LunarLander GTrXL checkpoint to ONNX and validate parity.

This follows the same export pattern as the ONNX notebook examples, but adapts
state handling for GTrXL (single packed recurrent state tensor).
"""

import argparse
import random
from pathlib import Path

import gym
import numpy as np
import onnx
import onnxruntime as ort
import torch

import rl_games.algos_torch.flatten as flatten
from rl_games.torch_runner import Runner


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


def _build_lunarlander_config(device):
    return {
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
                "max_epochs": 1000,
                "env_config": {"seed": 42},
                "player": {"render": False, "deterministic": True},
                # Player path uses `device_name` instead of `device`.
                "device_name": device,
            },
        }
    }


class ModelWrapper(torch.nn.Module):
    """Export only the neural network (no torch distribution sampling path)."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_dict):
        input_dict = dict(input_dict)
        input_dict["obs"] = self.model.norm_obs(input_dict["obs"])
        return self.model.a2c_network(input_dict)


def _extract_obs(reset_output):
    if isinstance(reset_output, tuple) and len(reset_output) == 2:
        return reset_output[0]
    return reset_output


def _reset_env(env, seed=None):
    if seed is None:
        out = env.reset()
        return _extract_obs(out)
    try:
        out = env.reset(seed=seed)
    except TypeError:
        if hasattr(env, "seed"):
            env.seed(seed)
        out = env.reset()
    return _extract_obs(out)


def _step_env(env, action):
    out = env.step(action)
    if len(out) == 5:
        obs, reward, terminated, truncated, info = out
        done = bool(terminated or truncated)
    else:
        obs, reward, done, info = out
        done = bool(done)
    return obs, reward, done, info


def _resolve_io_names(session):
    inputs = session.get_inputs()
    if len(inputs) != 2:
        raise RuntimeError(f"Expected 2 ONNX inputs (obs, state), got {len(inputs)}")

    obs_name = None
    state_name = None
    for item in inputs:
        shape = item.shape
        if len(shape) == 2:
            obs_name = item.name
        elif len(shape) == 3:
            state_name = item.name

    if obs_name is None or state_name is None:
        obs_name = inputs[0].name
        state_name = inputs[1].name
    return obs_name, state_name


def _run_torch_step(wrapper, obs_tensor, state_tensor):
    with torch.no_grad():
        logits, value, states_out = wrapper(
            {
                "obs": obs_tensor,
                "rnn_states": (state_tensor,),
            }
        )
    if isinstance(states_out, (tuple, list)):
        next_state = states_out[0]
    else:
        next_state = states_out
    return logits, value, next_state


def export_and_validate(checkpoint, onnx_out, steps=256, seed=42, device="cpu", tolerance=1e-4):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    checkpoint = Path(checkpoint)
    onnx_out = Path(onnx_out)
    onnx_out.parent.mkdir(parents=True, exist_ok=True)

    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    cfg = _build_lunarlander_config(device=device)
    runner = Runner()
    runner.load(cfg)
    player = runner.create_player()
    player.restore(str(checkpoint))
    player.init_rnn()

    model = player.model.to(device)
    model.eval()
    wrapper = ModelWrapper(model).to(device)
    wrapper.eval()

    obs_zeros = torch.zeros((1,) + player.obs_shape, dtype=torch.float32, device=device)
    inputs = {
        "obs": obs_zeros,
        "rnn_states": player.states,
    }

    with torch.no_grad():
        adapter = flatten.TracingAdapter(wrapper, inputs, allow_non_tensor=True)
        traced = torch.jit.trace(adapter, adapter.flattened_inputs, check_trace=False)
        torch.onnx.export(
            traced,
            adapter.flattened_inputs,
            str(onnx_out),
            verbose=False,
            input_names=["obs", "rnn_state"],
            output_names=["logits", "value", "rnn_state_out"],
        )

    onnx_model = onnx.load(str(onnx_out))
    onnx.checker.check_model(onnx_model)

    env = gym.make("LunarLander-v2")
    obs = _reset_env(env, seed=seed)
    state = player.states[0].detach().clone().to(device)

    records = []
    for _ in range(steps):
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        state_in = state.detach().clone()

        logits_t, value_t, state_out_t = _run_torch_step(wrapper, obs_tensor, state_in)

        records.append(
            {
                "obs": obs_tensor.detach().cpu().numpy().astype(np.float32, copy=True),
                "state_in": state_in.detach().cpu().numpy().astype(np.float32, copy=True),
                "torch_logits": logits_t.detach().cpu().numpy().astype(np.float32, copy=True),
                "torch_value": value_t.detach().cpu().numpy().astype(np.float32, copy=True),
                "torch_state_out": state_out_t.detach().cpu().numpy().astype(np.float32, copy=True),
            }
        )

        action = int(torch.argmax(logits_t, dim=-1).item())
        obs, _, done, _ = _step_env(env, action)
        if done:
            obs = _reset_env(env)
            state = torch.zeros_like(state)
        else:
            state = state_out_t.detach()

    env.close()

    session = ort.InferenceSession(str(onnx_out), providers=["CPUExecutionProvider"])
    obs_name, state_name = _resolve_io_names(session)

    max_logits = 0.0
    max_value = 0.0
    max_state = 0.0
    for item in records:
        onnx_outputs = session.run(
            None,
            {
                obs_name: item["obs"],
                state_name: item["state_in"],
            },
        )

        logits_o, value_o, state_o = onnx_outputs[0], onnx_outputs[1], onnx_outputs[2]

        max_logits = max(max_logits, float(np.max(np.abs(item["torch_logits"] - logits_o))))
        max_value = max(max_value, float(np.max(np.abs(item["torch_value"] - value_o))))
        max_state = max(max_state, float(np.max(np.abs(item["torch_state_out"] - state_o))))

    max_abs_diff = max(max_logits, max_value, max_state)
    print(f"ONNX export: {onnx_out}")
    print(f"Compared {len(records)} rollout steps")
    print(f"Max |logits_torch - logits_onnx|: {max_logits:.8f}")
    print(f"Max |value_torch  - value_onnx |: {max_value:.8f}")
    print(f"Max |state_torch  - state_onnx |: {max_state:.8f}")
    print(f"Global max abs diff: {max_abs_diff:.8f} (tolerance={tolerance})")

    if max_abs_diff > tolerance:
        raise AssertionError(
            f"Parity validation failed: max abs diff {max_abs_diff:.8f} > tolerance {tolerance}"
        )

    return {
        "onnx_path": str(onnx_out),
        "steps": len(records),
        "max_logits_diff": max_logits,
        "max_value_diff": max_value,
        "max_state_diff": max_state,
        "max_abs_diff": max_abs_diff,
    }


def main():
    parser = argparse.ArgumentParser(description="Export LunarLander GTrXL checkpoint to ONNX and validate parity")
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to checkpoint (.pth)",
    )
    parser.add_argument(
        "--onnx-out",
        required=True,
        help="Output ONNX filepath",
    )
    parser.add_argument("--steps", type=int, default=256, help="Rollout steps for parity validation")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"], help="Torch device")
    parser.add_argument("--tolerance", type=float, default=1e-4, help="Max absolute tolerance")
    args = parser.parse_args()

    export_and_validate(
        checkpoint=args.checkpoint,
        onnx_out=args.onnx_out,
        steps=args.steps,
        seed=args.seed,
        device=args.device,
        tolerance=args.tolerance,
    )


if __name__ == "__main__":
    main()
