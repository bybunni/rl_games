"""Validate ONNX export and parity for A2C LSTM networks.

Tests all four combinations of (before_mlp, concat_input) to ensure each
variant can be exported to ONNX and produces outputs matching PyTorch
inference when replayed over CartPole-v1 trajectory data.
"""

import importlib.util
import inspect
import random

import numpy as np
import pytest
import torch
import torch.nn as nn

import rl_games.algos_torch.flatten as flatten
from rl_games.algos_torch.network_builder import A2CBuilder


def _has_module(name):
    return importlib.util.find_spec(name) is not None


requires_onnx = pytest.mark.skipif(
    not _has_module("onnx"), reason="onnx is not installed"
)
requires_onnxruntime = pytest.mark.skipif(
    not _has_module("onnxruntime"), reason="onnxruntime is not installed"
)
requires_env = pytest.mark.skipif(
    not (_has_module("gymnasium") or _has_module("gym")),
    reason="neither gymnasium nor gym is installed",
)


def _make_env(env_id):
    """Create an environment using gymnasium (preferred) or gym."""
    try:
        import gymnasium
        return gymnasium.make(env_id)
    except ImportError:
        import gym
        return gym.make(env_id)


def _build_network_params(before_mlp, concat_input):
    """Return an A2CBuilder-compatible params dict for a shared LSTM network."""
    return {
        "mlp": {
            "units": [64, 64],
            "activation": "relu",
            "initializer": {"name": "default"},
        },
        "space": {"discrete": {}},
        "rnn": {
            "name": "lstm",
            "units": 64,
            "layers": 1,
            "before_mlp": before_mlp,
            "concat_input": concat_input,
        },
    }


class _SimpleModelShell(nn.Module):
    """Minimal shell that mirrors BaseModelNetwork for export purposes.

    Provides ``norm_obs`` (identity) and holds the ``a2c_network`` built by
    A2CBuilder so that ModelWrapper can call through without needing the full
    ModelA2C / torch.compile wrapper.
    """

    def __init__(self, a2c_network):
        super().__init__()
        self.a2c_network = a2c_network
        self.normalize_input = False

    def norm_obs(self, observation):
        return observation


class _ModelWrapper(nn.Module):
    """Export-only wrapper: normalises obs then forwards through a2c_network."""

    def __init__(self, shell):
        super().__init__()
        self.shell = shell

    def forward(self, input_dict):
        input_dict = dict(input_dict)
        input_dict["obs"] = self.shell.norm_obs(input_dict["obs"])
        return self.shell.a2c_network(input_dict)


def _build_model(before_mlp, concat_input):
    """Construct an A2CBuilder LSTM network and wrap it for export."""
    params = _build_network_params(before_mlp, concat_input)
    builder = A2CBuilder()
    builder.load(params)
    a2c_net = builder.build(
        "a2c",
        input_shape=(4,),
        actions_num=2,
        num_seqs=1,
        value_size=1,
    )
    shell = _SimpleModelShell(a2c_net)
    wrapper = _ModelWrapper(shell)
    wrapper.eval()
    return wrapper, a2c_net


def _run_torch_step(wrapper, obs_tensor, h, c):
    """Single inference step through the PyTorch model."""
    with torch.no_grad():
        result = wrapper(
            {
                "obs": obs_tensor,
                "rnn_states": (h, c),
            }
        )
    logits, value, states_out = result
    if isinstance(states_out, (tuple, list)):
        h_out, c_out = states_out[0], states_out[1]
    else:
        h_out = c_out = states_out
    return logits, value, h_out, c_out


def _extract_obs(reset_output):
    if isinstance(reset_output, tuple) and len(reset_output) == 2:
        return reset_output[0]
    return reset_output


def _reset_env(env, seed=None):
    if seed is None:
        return _extract_obs(env.reset())
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


def _onnx_export(traced_model, sample_inputs, path, input_names, output_names):
    """Call torch.onnx.export with legacy JIT-trace exporter on newer PyTorch."""
    kwargs = dict(
        verbose=False,
        input_names=input_names,
        output_names=output_names,
    )
    # PyTorch >= 2.6 defaults to the dynamo-based exporter; fall back to the
    # legacy JIT-trace exporter which is what the rest of rl_games uses.
    if "dynamo" in inspect.signature(torch.onnx.export).parameters:
        kwargs["dynamo"] = False
    torch.onnx.export(traced_model, sample_inputs, str(path), **kwargs)


# ---------------------------------------------------------------------------
# Main parametrised test
# ---------------------------------------------------------------------------

@requires_onnx
@requires_onnxruntime
@requires_env
@pytest.mark.parametrize(
    "before_mlp,concat_input",
    [
        (False, False),
        (False, True),
        (True, False),
        (True, True),
    ],
    ids=["after_mlp", "after_mlp_concat", "before_mlp", "before_mlp_concat"],
)
def test_lstm_onnx_export_and_validate(tmp_path, before_mlp, concat_input):
    import onnx
    import onnxruntime as ort

    seed = 42
    steps = 32
    tolerance = 1e-4

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # ---- build model ----
    wrapper, a2c_net = _build_model(before_mlp, concat_input)
    default_states = a2c_net.get_default_rnn_state()  # (h, c)
    h_zero = default_states[0]  # (num_layers, num_seqs, rnn_units)
    c_zero = default_states[1]

    # ---- export to ONNX ----
    obs_zeros = torch.zeros(1, 4, dtype=torch.float32)
    sample_inputs = {
        "obs": obs_zeros,
        "rnn_states": (h_zero, c_zero),
    }

    with torch.no_grad():
        adapter = flatten.TracingAdapter(wrapper, sample_inputs, allow_non_tensor=True)
        traced = torch.jit.trace(adapter, adapter.flattened_inputs, check_trace=False)

    tag = f"lstm_bm{int(before_mlp)}_ci{int(concat_input)}"
    onnx_path = tmp_path / f"{tag}.onnx"

    # Determine I/O names from the flattened adapter shapes.
    num_flat_inputs = len(adapter.flattened_inputs)
    if num_flat_inputs == 3:
        input_names = ["obs", "rnn_h", "rnn_c"]
    else:
        input_names = [f"input_{i}" for i in range(num_flat_inputs)]

    # Run adapter once to discover output count.
    with torch.no_grad():
        sample_out = adapter(*adapter.flattened_inputs)
    num_flat_outputs = len(sample_out)
    if num_flat_outputs == 4:
        output_names = ["logits", "value", "rnn_h_out", "rnn_c_out"]
    else:
        output_names = [f"output_{i}" for i in range(num_flat_outputs)]

    _onnx_export(traced, adapter.flattened_inputs, onnx_path, input_names, output_names)

    onnx_model = onnx.load(str(onnx_path))
    onnx.checker.check_model(onnx_model)

    # ---- collect trajectory data via CartPole ----
    env = _make_env("CartPole-v1")
    obs = _reset_env(env, seed=seed)
    h = h_zero.clone()
    c = c_zero.clone()

    records = []
    for _ in range(steps):
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
        h_in = h.detach().clone()
        c_in = c.detach().clone()

        logits_t, value_t, h_out_t, c_out_t = _run_torch_step(
            wrapper, obs_tensor, h_in, c_in
        )

        records.append(
            {
                "obs": obs_tensor.numpy().astype(np.float32, copy=True),
                "h_in": h_in.numpy().astype(np.float32, copy=True),
                "c_in": c_in.numpy().astype(np.float32, copy=True),
                "torch_logits": logits_t.numpy().astype(np.float32, copy=True),
                "torch_value": value_t.numpy().astype(np.float32, copy=True),
                "torch_h_out": h_out_t.numpy().astype(np.float32, copy=True),
                "torch_c_out": c_out_t.numpy().astype(np.float32, copy=True),
            }
        )

        action = int(torch.argmax(logits_t, dim=-1).item())
        obs, _, done, _ = _step_env(env, action)
        if done:
            obs = _reset_env(env)
            h = torch.zeros_like(h_zero)
            c = torch.zeros_like(c_zero)
        else:
            h = h_out_t.detach()
            c = c_out_t.detach()

    env.close()

    # ---- validate ONNX parity ----
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    onnx_input_names = [inp.name for inp in session.get_inputs()]

    max_logits = 0.0
    max_value = 0.0
    max_h = 0.0
    max_c = 0.0

    for item in records:
        feed = {}
        for name in onnx_input_names:
            if "obs" in name:
                feed[name] = item["obs"]
            elif "rnn_h" in name or name == input_names[1]:
                feed[name] = item["h_in"]
            elif "rnn_c" in name or name == input_names[2]:
                feed[name] = item["c_in"]

        onnx_out = session.run(None, feed)

        logits_o = onnx_out[0]
        value_o = onnx_out[1]
        h_o = onnx_out[2]
        c_o = onnx_out[3]

        max_logits = max(max_logits, float(np.max(np.abs(item["torch_logits"] - logits_o))))
        max_value = max(max_value, float(np.max(np.abs(item["torch_value"] - value_o))))
        max_h = max(max_h, float(np.max(np.abs(item["torch_h_out"] - h_o))))
        max_c = max(max_c, float(np.max(np.abs(item["torch_c_out"] - c_o))))

    max_abs_diff = max(max_logits, max_value, max_h, max_c)

    assert max_abs_diff <= tolerance, (
        f"[{tag}] Parity check failed: max_abs_diff={max_abs_diff:.8f} > "
        f"tolerance={tolerance}. "
        f"logits={max_logits:.8f} value={max_value:.8f} "
        f"h={max_h:.8f} c={max_c:.8f}"
    )
