#!/usr/bin/env python3
"""Train shared PPO-LSTM CartPole variants and validate ONNX export parity.

This script is intended for architecture-variant sweeps (placement/concat and
normalization variants). It trains each variant on CartPoleMaskedVelocity-v1,
exports the trained checkpoint with `export_lstm_onnx.py`, and validates Torch
vs ONNX parity using rollout inputs collected from the environment.
"""

import argparse
import copy
import random
import time
from pathlib import Path

import numpy as np
import torch

from rl_games.common import env_configurations, vecenv
from rl_games.common.tr_helpers import dicts_to_dict_with_arrays
from rl_games.torch_runner import Runner

from export_lstm_onnx import build_cartpole_lstm_config, export_and_validate, variant_names


def _extract_obs(reset_output):
    if isinstance(reset_output, tuple) and len(reset_output) == 2:
        return reset_output[0]
    return reset_output


def _step_env(env, action):
    out = env.step(action)
    if len(out) == 5:
        obs, reward, terminated, truncated, info = out
        done = bool(terminated or truncated)
    else:
        obs, reward, done, info = out
        done = bool(done)
    return obs, reward, done, info


def _obs_to_fp32(obs):
    if isinstance(obs, dict):
        out = {}
        for k, v in obs.items():
            if isinstance(v, dict):
                out[k] = {
                    dk: (dv.astype(np.float32) if getattr(dv, "dtype", None) == np.float64 else dv)
                    for dk, dv in v.items()
                }
            else:
                out[k] = v.astype(np.float32) if getattr(v, "dtype", None) == np.float64 else v
        return out
    if getattr(obs, "dtype", None) == np.float64:
        return obs.astype(np.float32)
    return obs


class _LocalRayCompatVecEnv:
    """Minimal in-process replacement for RayVecEnv for gym classic-control runs."""

    def __init__(self, config_name, num_actors, **kwargs):
        self.config_name = config_name
        self.num_actors = num_actors
        self.seed = kwargs.pop("seed", None)
        self.envs = []

        cfg = env_configurations.configurations[config_name]
        creator = cfg["env_creator"]
        for i in range(num_actors):
            env = creator(**kwargs)
            seed = None if self.seed is None else self.seed + i
            if seed is not None:
                try:
                    env.reset(seed=seed)
                except TypeError:
                    if hasattr(env, "seed"):
                        env.seed(seed)
            self.envs.append(env)

        info = self.get_env_info()
        self.use_global_obs = info["use_global_observations"]
        self.concat_infos = False
        self.num_agents = info["agents"]
        self.obs_type_dict = False
        self.state_type_dict = False
        self.concat_func = np.stack if self.num_agents == 1 else np.concatenate

    def _reset_one(self, env):
        return _obs_to_fp32(_extract_obs(env.reset()))

    def _step_one(self, env, action):
        obs, reward, done, info = _step_env(env, action)
        if done:
            obs = self._reset_one(env)
        return _obs_to_fp32(obs), reward, done, info

    def step(self, actions):
        newobs, newrewards, newdones, newinfos = [], [], [], []
        if self.num_agents == 1:
            for env, action in zip(self.envs, actions):
                cobs, creward, cdone, cinfo = self._step_one(env, action)
                newobs.append(cobs)
                newrewards.append(creward)
                newdones.append(cdone)
                newinfos.append(cinfo)
        else:
            for idx, env in enumerate(self.envs):
                action = actions[self.num_agents * idx : self.num_agents * (idx + 1)]
                cobs, creward, cdone, cinfo = self._step_one(env, action)
                newobs.append(cobs)
                newrewards.append(creward)
                newdones.append(cdone)
                newinfos.append(cinfo)

        if isinstance(newobs[0], dict):
            ret_obs = dicts_to_dict_with_arrays(newobs, self.num_agents == 1)
        else:
            ret_obs = self.concat_func(newobs)
        return (
            ret_obs,
            self.concat_func(newrewards),
            self.concat_func(newdones),
            newinfos,
        )

    def reset(self):
        newobs = [self._reset_one(env) for env in self.envs]
        if isinstance(newobs[0], dict):
            return dicts_to_dict_with_arrays(newobs, self.num_agents == 1)
        return self.concat_func(newobs)

    def get_env_info(self):
        env = self.envs[0]
        info = {
            "action_space": env.action_space,
            "observation_space": env.observation_space,
            "state_space": None,
            "use_global_observations": False,
            "agents": 1,
            "value_size": 1,
        }
        if hasattr(env, "use_central_value"):
            info["use_global_observations"] = env.use_central_value
        if hasattr(env, "value_size"):
            info["value_size"] = env.value_size
        if hasattr(env, "state_space"):
            info["state_space"] = env.state_space
        return info

    def has_action_masks(self):
        return False

    def get_number_of_agents(self):
        return self.num_agents

    def set_train_info(self, env_frames, *args, **kwargs):
        for env in self.envs:
            if hasattr(env, "set_train_info"):
                env.set_train_info(env_frames, *args, **kwargs)

    def get_env_state(self):
        return None

    def set_env_state(self, env_state):
        return None

    def set_weights(self, indices, weights):
        for ind in indices:
            env = self.envs[ind]
            if hasattr(env, "update_weights"):
                env.update_weights(weights)

    def close(self):
        for env in self.envs:
            if hasattr(env, "close"):
                env.close()


class _NoOpSummaryWriter:
    def __init__(self, *args, **kwargs):
        pass

    def __getattr__(self, _):
        return lambda *args, **kwargs: None

    def close(self):
        return None


def _install_runtime_workarounds():
    # Replace Ray vecenv with an in-process implementation for sandboxed runs.
    vecenv.register("RAY", lambda config_name, num_actors, **kwargs: _LocalRayCompatVecEnv(config_name, num_actors, **kwargs))

    # Disable TensorBoard writer side effects (multiprocessing semaphores in some sandboxes).
    import rl_games.common.a2c_common as a2c_common

    a2c_common.SummaryWriter = _NoOpSummaryWriter


def _device_name(device_arg):
    return "cuda:0" if device_arg == "cuda" else "cpu"


def _pick_checkpoint(nn_dir):
    ckpts = sorted(nn_dir.glob("*.pth"), key=lambda p: p.stat().st_mtime)
    if not ckpts:
        raise FileNotFoundError(f"No checkpoints found in {nn_dir}")

    non_last = [p for p in ckpts if not p.name.startswith("last_")]
    return non_last[-1] if non_last else ckpts[-1]


def _train_variant(variant, *, max_epochs, seed, device, train_dir):
    cfg = build_cartpole_lstm_config(variant=variant, device=device)
    params = cfg["params"]
    config = params["config"]

    params["seed"] = seed
    config["env_config"] = {"seed": seed}
    config["max_epochs"] = max_epochs
    config["train_dir"] = str(train_dir)
    config["full_experiment_name"] = f"cartpole_lstm_{variant}"
    config["device"] = _device_name(device)
    config["device_name"] = _device_name(device)

    # Keep the sweep lightweight and deterministic.
    config["num_actors"] = 4
    config["horizon_length"] = 128
    config["minibatch_size"] = 512
    config["mini_epochs"] = 4
    config["seq_length"] = 8
    config["player"]["use_vecenv"] = False

    runner = Runner()
    runner.load(cfg)

    agent = runner.algo_factory.create(runner.algo_name, base_name="run", params=runner.params)
    start = time.perf_counter()
    try:
        agent.train()
    finally:
        duration = time.perf_counter() - start

    exp_dir = Path(agent.experiment_dir)
    nn_dir = exp_dir / "nn"
    checkpoint = _pick_checkpoint(nn_dir)

    return {
        "variant": variant,
        "experiment_dir": str(exp_dir),
        "checkpoint": str(checkpoint),
        "duration_sec": duration,
    }


def _run_sweep(variants, *, max_epochs, seed, device, export_steps, tolerance, train_dir):
    _install_runtime_workarounds()
    results = []
    for idx, variant in enumerate(variants):
        variant_seed = seed + idx
        print("=" * 72)
        print(f"[{idx+1}/{len(variants)}] Training variant: {variant} (seed={variant_seed})")
        print("=" * 72)
        train_res = _train_variant(
            variant,
            max_epochs=max_epochs,
            seed=variant_seed,
            device=device,
            train_dir=train_dir,
        )
        ckpt = Path(train_res["checkpoint"])
        onnx_out = ckpt.with_suffix(".onnx")
        print(f"Checkpoint: {ckpt}")
        export_res = export_and_validate(
            checkpoint=str(ckpt),
            onnx_out=str(onnx_out),
            variant=variant,
            steps=export_steps,
            seed=variant_seed,
            device="cpu",
            tolerance=tolerance,
        )

        merged = copy.deepcopy(train_res)
        merged.update(export_res)
        results.append(merged)
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Train shared PPO-LSTM architecture variants on CartPoleMaskedVelocity-v1 and validate ONNX parity"
    )
    parser.add_argument(
        "--variants",
        nargs="*",
        default=list(variant_names()),
        choices=list(variant_names()),
        help="Variant list to run (default: all)",
    )
    parser.add_argument("--max-epochs", type=int, default=60, help="Training epochs per variant")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu", help="Training device")
    parser.add_argument("--export-steps", type=int, default=64, help="Rollout steps for ONNX parity checks")
    parser.add_argument("--tolerance", type=float, default=1e-4, help="Max abs diff tolerance for ONNX parity")
    parser.add_argument(
        "--train-dir",
        default="runs/lstm_variants",
        help="Base training output directory",
    )
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    train_dir = Path(args.train_dir)
    train_dir.mkdir(parents=True, exist_ok=True)

    results = _run_sweep(
        args.variants,
        max_epochs=args.max_epochs,
        seed=args.seed,
        device=args.device,
        export_steps=args.export_steps,
        tolerance=args.tolerance,
        train_dir=train_dir,
    )

    print()
    print("Summary")
    print("-" * 72)
    for row in results:
        print(
            f"{row['variant']:24s} "
            f"epochs={args.max_epochs:<4d} "
            f"train={row['duration_sec']:.1f}s "
            f"onnx_max_diff={row['max_abs_diff']:.8f} "
            f"ckpt={row['checkpoint']}"
        )


if __name__ == "__main__":
    main()
