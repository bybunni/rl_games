#!/usr/bin/env python3
"""Export a trained shared PPO-LSTM checkpoint to ONNX and validate parity.

The script reconstructs the network architecture from a named variant, restores
the checkpoint with an rl_games player, exports only the neural network path
(logits/value/rnn state), and validates parity against ONNX Runtime on rollout
inputs gathered from the environment.
"""

import argparse
import random
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch

import rl_games.algos_torch.flatten as flatten
from rl_games.common import env_configurations
from rl_games.torch_runner import Runner


_BASE_NETWORK = {
    "name": "actor_critic",
    "separate": False,
    "space": {"discrete": {}},
    "mlp": {
        "units": [64, 64],
        "activation": "relu",
        "initializer": {"name": "default"},
    },
    "rnn": {
        "name": "lstm",
        "units": 64,
        "layers": 1,
        "before_mlp": False,
    },
}


def variant_names():
    return (
        "after_mlp",
        "after_mlp_concat_input",
        "after_mlp_concat_output",
        "before_mlp",
        "after_mlp_layer_norm",
        "after_mlp_batch_norm",
    )


def _apply_variant(network, variant):
    network = dict(network)
    network["mlp"] = dict(network["mlp"])
    network["rnn"] = dict(network["rnn"])

    if variant == "after_mlp":
        pass
    elif variant == "after_mlp_concat_input":
        network["rnn"]["concat_input"] = True
    elif variant == "after_mlp_concat_output":
        network["rnn"]["concat_output"] = True
    elif variant == "before_mlp":
        network["rnn"]["before_mlp"] = True
    elif variant == "after_mlp_layer_norm":
        network["normalization"] = "layer_norm"
        network["rnn"]["layer_norm"] = True
    elif variant == "after_mlp_batch_norm":
        network["normalization"] = "batch_norm"
    else:
        raise ValueError(f"Unknown variant: {variant}")
    return network


def build_cartpole_lstm_config(variant, device):
    device_name = "cuda:0" if device == "cuda" else "cpu"
    return {
        "params": {
            "algo": {"name": "a2c_discrete"},
            "model": {"name": "discrete_a2c"},
            "load_checkpoint": False,
            "network": _apply_variant(_BASE_NETWORK, variant),
            "config": {
                "env_name": "CartPoleMaskedVelocity-v1",
                "reward_shaper": {"scale_value": 0.1},
                "normalize_advantage": True,
                "gamma": 0.99,
                "tau": 0.9,
                "learning_rate": 3e-4,
                "name": f"cartpole_lstm_{variant}",
                "score_to_win": 490,
                "grad_norm": 0.5,
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
                "normalize_value": False,
                "seq_length": 8,
                "zero_rnn_on_done": True,
                "max_epochs": 1,
                "env_config": {"seed": 42},
                "device": device_name,
                "device_name": device_name,
                "player": {
                    "render": False,
                    "deterministic": True,
                    "games_num": 1,
                    "use_vecenv": False,
                },
            },
        }
    }


class ModelWrapper(torch.nn.Module):
    """Export only the neural network output path (no sampling/distribution)."""

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
        return _extract_obs(env.reset())
    try:
        return _extract_obs(env.reset(seed=seed))
    except TypeError:
        if hasattr(env, "seed"):
            env.seed(seed)
        return _extract_obs(env.reset())


def _step_env(env, action):
    out = env.step(action)
    if len(out) == 5:
        obs, reward, terminated, truncated, info = out
        done = bool(terminated or truncated)
    else:
        obs, reward, done, info = out
        done = bool(done)
    return obs, reward, done, info


def _ensure_state_tuple(states):
    if isinstance(states, tuple):
        return states
    if isinstance(states, list):
        return tuple(states)
    return (states,)


def _run_torch_step(wrapper, obs_tensor, state_tuple):
    with torch.no_grad():
        logits, value, states_out = wrapper(
            {
                "obs": obs_tensor,
                "rnn_states": state_tuple,
            }
        )
    return logits, value, _ensure_state_tuple(states_out)


def _collect_rollout_records(wrapper, env_name, steps, seed, device, initial_state):
    env = env_configurations.configurations[env_name]["env_creator"]()
    try:
        obs = _reset_env(env, seed=seed)
        state = tuple(s.detach().clone().to(device) for s in initial_state)

        records = []
        for _ in range(steps):
            obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            state_in = tuple(s.detach().clone() for s in state)

            logits_t, value_t, state_out_t = _run_torch_step(wrapper, obs_tensor, state_in)

            record = {
                "obs": obs_tensor.detach().cpu().numpy().astype(np.float32, copy=True),
                "torch_logits": logits_t.detach().cpu().numpy().astype(np.float32, copy=True),
                "torch_value": value_t.detach().cpu().numpy().astype(np.float32, copy=True),
            }
            for i, s in enumerate(state_in):
                record[f"state_in_{i}"] = s.detach().cpu().numpy().astype(np.float32, copy=True)
            for i, s in enumerate(state_out_t):
                record[f"torch_state_out_{i}"] = s.detach().cpu().numpy().astype(np.float32, copy=True)
            records.append(record)

            action = int(torch.argmax(logits_t, dim=-1).item())
            obs, _, done, _ = _step_env(env, action)
            if done:
                obs = _reset_env(env)
                state = tuple(torch.zeros_like(s) for s in state)
            else:
                state = tuple(s.detach() for s in state_out_t)

        return records
    finally:
        env.close()


def export_and_validate(
    checkpoint,
    onnx_out,
    *,
    variant,
    steps=256,
    seed=42,
    device="cpu",
    tolerance=1e-4,
):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    checkpoint = Path(checkpoint)
    onnx_out = Path(onnx_out)
    onnx_out.parent.mkdir(parents=True, exist_ok=True)

    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    cfg = build_cartpole_lstm_config(variant=variant, device=device)
    runner = Runner()
    runner.load(cfg)
    player = runner.create_player()
    player.restore(str(checkpoint))
    player.init_rnn()

    model = player.model.to(device)
    model.eval()
    wrapper = ModelWrapper(model).to(device)
    wrapper.eval()

    state_tuple = _ensure_state_tuple(player.states)
    obs_zeros = torch.zeros((1,) + player.obs_shape, dtype=torch.float32, device=device)
    inputs = {"obs": obs_zeros, "rnn_states": state_tuple}

    state_input_names = [f"rnn_state_{i}" for i in range(len(state_tuple))]
    state_output_names = [f"rnn_state_out_{i}" for i in range(len(state_tuple))]
    output_names = ["logits", "value"] + state_output_names
    input_names = ["obs"] + state_input_names

    with torch.no_grad():
        adapter = flatten.TracingAdapter(wrapper, inputs, allow_non_tensor=True)
        traced = torch.jit.trace(adapter, adapter.flattened_inputs, check_trace=False)
        torch.onnx.export(
            traced,
            adapter.flattened_inputs,
            str(onnx_out),
            verbose=False,
            input_names=input_names,
            output_names=output_names,
        )

    onnx_model = onnx.load(str(onnx_out))
    onnx.checker.check_model(onnx_model)

    records = _collect_rollout_records(
        wrapper=wrapper,
        env_name=cfg["params"]["config"]["env_name"],
        steps=steps,
        seed=seed,
        device=device,
        initial_state=state_tuple,
    )

    session = ort.InferenceSession(str(onnx_out), providers=["CPUExecutionProvider"])
    input_names_by_session = [x.name for x in session.get_inputs()]

    max_diffs = {"logits": 0.0, "value": 0.0}
    for i in range(len(state_tuple)):
        max_diffs[f"state_{i}"] = 0.0

    for item in records:
        feeds = {"obs": item["obs"]}
        for i in range(len(state_tuple)):
            feeds[f"rnn_state_{i}"] = item[f"state_in_{i}"]

        # Some runtimes preserve names but may reorder inputs.
        ordered_feeds = {name: feeds[name] for name in input_names_by_session}
        onnx_outputs = session.run(None, ordered_feeds)
        onnx_map = dict(zip(output_names, onnx_outputs))

        max_diffs["logits"] = max(
            max_diffs["logits"],
            float(np.max(np.abs(item["torch_logits"] - onnx_map["logits"]))),
        )
        max_diffs["value"] = max(
            max_diffs["value"],
            float(np.max(np.abs(item["torch_value"] - onnx_map["value"]))),
        )
        for i in range(len(state_tuple)):
            key = f"state_{i}"
            max_diffs[key] = max(
                max_diffs[key],
                float(
                    np.max(
                        np.abs(item[f"torch_state_out_{i}"] - onnx_map[f"rnn_state_out_{i}"])
                    )
                ),
            )

    max_abs_diff = max(max_diffs.values())
    print(f"Variant: {variant}")
    print(f"ONNX export: {onnx_out}")
    print(f"Compared {len(records)} rollout steps")
    print(f"Max |logits_torch - logits_onnx|: {max_diffs['logits']:.8f}")
    print(f"Max |value_torch  - value_onnx |: {max_diffs['value']:.8f}")
    for i in range(len(state_tuple)):
        print(
            f"Max |state{i}_torch - state{i}_onnx|: "
            f"{max_diffs[f'state_{i}']:.8f}"
        )
    print(f"Global max abs diff: {max_abs_diff:.8f} (tolerance={tolerance})")

    if max_abs_diff > tolerance:
        raise AssertionError(
            f"Parity validation failed: max abs diff {max_abs_diff:.8f} > tolerance {tolerance}"
        )

    result = {
        "variant": variant,
        "onnx_path": str(onnx_out),
        "steps": len(records),
        "max_logits_diff": max_diffs["logits"],
        "max_value_diff": max_diffs["value"],
        "max_abs_diff": max_abs_diff,
    }
    for i in range(len(state_tuple)):
        result[f"max_state_{i}_diff"] = max_diffs[f"state_{i}"]
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Export a shared PPO-LSTM CartPole variant checkpoint to ONNX and validate parity"
    )
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint (.pth)")
    parser.add_argument("--onnx-out", required=True, help="Output ONNX filepath")
    parser.add_argument(
        "--variant",
        required=True,
        choices=variant_names(),
        help="Network variant used for the checkpoint",
    )
    parser.add_argument("--steps", type=int, default=256, help="Rollout steps for parity validation")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"], help="Torch device")
    parser.add_argument("--tolerance", type=float, default=1e-4, help="Max absolute tolerance")
    args = parser.parse_args()

    export_and_validate(
        checkpoint=args.checkpoint,
        onnx_out=args.onnx_out,
        variant=args.variant,
        steps=args.steps,
        seed=args.seed,
        device=args.device,
        tolerance=args.tolerance,
    )


if __name__ == "__main__":
    main()

