"""Tests for the policy sensitivity analysis tool."""

import json
import threading
import time
import urllib.request
from http.client import HTTPConnection
from pathlib import Path

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper

from tools.policy_viz.model_adapter import ModelAdapter

FIXTURES_DIR = Path(__file__).parent / "_policy_viz_fixtures"


def _make_discrete_model(path, obs_dim=8, num_actions=4):
    """Create a minimal ONNX model: obs -> (logits, value)."""
    # Weights: random linear layer for logits and value
    np.random.seed(42)
    W_logits = np.random.randn(obs_dim, num_actions).astype(np.float32)
    W_value = np.random.randn(obs_dim, 1).astype(np.float32)

    obs = helper.make_tensor_value_info("obs", TensorProto.FLOAT, [1, obs_dim])
    logits_out = helper.make_tensor_value_info("logits", TensorProto.FLOAT, [1, num_actions])
    value_out = helper.make_tensor_value_info("value", TensorProto.FLOAT, [1, 1])

    w_logits_init = helper.make_tensor("W_logits", TensorProto.FLOAT, [obs_dim, num_actions], W_logits.flatten())
    w_value_init = helper.make_tensor("W_value", TensorProto.FLOAT, [obs_dim, 1], W_value.flatten())

    matmul_logits = helper.make_node("MatMul", ["obs", "W_logits"], ["logits"])
    matmul_value = helper.make_node("MatMul", ["obs", "W_value"], ["value"])

    graph = helper.make_graph(
        [matmul_logits, matmul_value],
        "discrete_policy",
        [obs],
        [logits_out, value_out],
        initializer=[w_logits_init, w_value_init],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 7
    path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(path))


def _make_continuous_model(path, obs_dim=3, action_dim=1):
    """Create a minimal ONNX model: obs -> (mu, log_std, value)."""
    np.random.seed(42)
    W_mu = np.random.randn(obs_dim, action_dim).astype(np.float32)
    W_value = np.random.randn(obs_dim, 1).astype(np.float32)
    log_std_val = np.zeros((1, action_dim), dtype=np.float32)

    obs = helper.make_tensor_value_info("obs", TensorProto.FLOAT, [1, obs_dim])
    mu_out = helper.make_tensor_value_info("mu", TensorProto.FLOAT, [1, action_dim])
    log_std_out = helper.make_tensor_value_info("log_std", TensorProto.FLOAT, [1, action_dim])
    value_out = helper.make_tensor_value_info("value", TensorProto.FLOAT, [1, 1])

    w_mu_init = helper.make_tensor("W_mu", TensorProto.FLOAT, [obs_dim, action_dim], W_mu.flatten())
    w_value_init = helper.make_tensor("W_value", TensorProto.FLOAT, [obs_dim, 1], W_value.flatten())
    log_std_init = helper.make_tensor("log_std_const", TensorProto.FLOAT, [1, action_dim], log_std_val.flatten())

    matmul_mu = helper.make_node("MatMul", ["obs", "W_mu"], ["mu"])
    matmul_value = helper.make_node("MatMul", ["obs", "W_value"], ["value"])
    identity_std = helper.make_node("Identity", ["log_std_const"], ["log_std"])

    graph = helper.make_graph(
        [matmul_mu, matmul_value, identity_std],
        "continuous_policy",
        [obs],
        [mu_out, log_std_out, value_out],
        initializer=[w_mu_init, w_value_init, log_std_init],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 7
    path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(path))


def _make_recurrent_discrete_model(path, obs_dim=8, num_actions=4, state_dim=64, memory_len=32, num_layers=2):
    """Create a minimal ONNX model with RNN state: (obs, rnn_state) -> (logits, value, rnn_state_out).

    The state is just passed through (identity) for testing purposes.
    """
    np.random.seed(42)
    W_logits = np.random.randn(obs_dim, num_actions).astype(np.float32)
    W_value = np.random.randn(obs_dim, 1).astype(np.float32)

    state_shape = [num_layers * memory_len, 1, state_dim]

    obs = helper.make_tensor_value_info("obs", TensorProto.FLOAT, [1, obs_dim])
    rnn_state = helper.make_tensor_value_info("rnn_state", TensorProto.FLOAT, state_shape)
    logits_out = helper.make_tensor_value_info("logits", TensorProto.FLOAT, [1, num_actions])
    value_out = helper.make_tensor_value_info("value", TensorProto.FLOAT, [1, 1])
    state_out = helper.make_tensor_value_info("rnn_state_out", TensorProto.FLOAT, state_shape)

    w_logits_init = helper.make_tensor("W_logits", TensorProto.FLOAT, [obs_dim, num_actions], W_logits.flatten())
    w_value_init = helper.make_tensor("W_value", TensorProto.FLOAT, [obs_dim, 1], W_value.flatten())

    matmul_logits = helper.make_node("MatMul", ["obs", "W_logits"], ["logits"])
    matmul_value = helper.make_node("MatMul", ["obs", "W_value"], ["value"])
    identity_state = helper.make_node("Identity", ["rnn_state"], ["rnn_state_out"])

    graph = helper.make_graph(
        [matmul_logits, matmul_value, identity_state],
        "recurrent_discrete_policy",
        [obs, rnn_state],
        [logits_out, value_out, state_out],
        initializer=[w_logits_init, w_value_init],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 7
    path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(path))


# ---- ModelAdapter tests ----


class TestModelAdapterDiscrete:
    @pytest.fixture(autouse=True)
    def setup(self, tmp_path):
        self.model_path = tmp_path / "discrete.onnx"
        _make_discrete_model(self.model_path, obs_dim=8, num_actions=4)
        self.adapter = ModelAdapter(self.model_path)

    def test_metadata(self):
        meta = self.adapter.get_metadata()
        assert meta["obs_dim"] == 8
        assert meta["action_type"] == "discrete"
        assert meta["num_actions"] == 4
        assert meta["is_recurrent"] is False

    def test_infer_returns_probabilities(self):
        result = self.adapter.infer([0.0] * 8)
        assert "probabilities" in result
        probs = result["probabilities"]
        assert len(probs) == 4
        assert abs(sum(probs) - 1.0) < 1e-5
        assert all(p >= 0 for p in probs)

    def test_infer_returns_value(self):
        result = self.adapter.infer([0.1] * 8)
        assert "value" in result
        assert isinstance(result["value"], float)

    def test_infer_best_action(self):
        result = self.adapter.infer([0.5, -0.3, 0.1, 0.8, -0.2, 0.0, 1.0, 0.0])
        assert "best_action" in result
        assert 0 <= result["best_action"] < 4

    def test_sensitivity_changes_output(self):
        r1 = self.adapter.infer([0.0] * 8)
        r2 = self.adapter.infer([1.0] * 8)
        assert r1["probabilities"] != r2["probabilities"]


class TestModelAdapterContinuous:
    @pytest.fixture(autouse=True)
    def setup(self, tmp_path):
        self.model_path = tmp_path / "continuous.onnx"
        _make_continuous_model(self.model_path, obs_dim=3, action_dim=1)
        self.adapter = ModelAdapter(self.model_path)

    def test_metadata(self):
        meta = self.adapter.get_metadata()
        assert meta["obs_dim"] == 3
        assert meta["action_type"] == "continuous"
        assert meta["num_actions"] == 1
        assert meta["is_recurrent"] is False

    def test_infer_returns_mu(self):
        result = self.adapter.infer([0.5, 0.5, 0.0])
        assert "mu" in result
        assert len(result["mu"]) == 1

    def test_infer_returns_sigma(self):
        result = self.adapter.infer([0.0, 0.0, 0.0])
        assert "sigma" in result
        # exp(0) = 1.0
        assert abs(result["sigma"][0] - 1.0) < 1e-5


class TestModelAdapterRecurrent:
    @pytest.fixture(autouse=True)
    def setup(self, tmp_path):
        self.model_path = tmp_path / "recurrent.onnx"
        _make_recurrent_discrete_model(self.model_path)
        self.adapter = ModelAdapter(self.model_path)

    def test_metadata_recurrent(self):
        meta = self.adapter.get_metadata()
        assert meta["is_recurrent"] is True
        assert meta["action_type"] == "discrete"

    def test_zero_rnn_state(self):
        state = self.adapter.get_zero_rnn_state()
        assert state is not None
        assert "rnn_state" in state
        assert state["rnn_state"].shape == (64, 1, 64)  # 2*32 layers*mem, 1 batch, 64 dim
        assert np.all(state["rnn_state"] == 0)

    def test_stateless_infer(self):
        result = self.adapter.infer([0.0] * 8, stateful=False)
        assert "probabilities" in result
        assert len(result["probabilities"]) == 4

    def test_stateful_infer(self):
        self.adapter.reset_rnn_state()
        r1 = self.adapter.infer([0.5] * 8, stateful=True)
        r2 = self.adapter.infer([0.5] * 8, stateful=True)
        # Both should succeed (state identity means same outputs)
        assert "probabilities" in r1
        assert "probabilities" in r2

    def test_reset_state(self):
        self.adapter.infer([1.0] * 8, stateful=True)
        self.adapter.reset_rnn_state()
        state = self.adapter._rnn_state
        assert np.all(state["rnn_state"] == 0)


class TestModelAdapterWithEnvConfig:
    @pytest.fixture(autouse=True)
    def setup(self, tmp_path):
        self.model_path = tmp_path / "discrete.onnx"
        _make_discrete_model(self.model_path, obs_dim=8, num_actions=4)
        self.config_path = tmp_path / "env.json"
        config = {
            "env_name": "TestEnv",
            "obs_names": [f"feat_{i}" for i in range(8)],
            "obs_ranges": [[-1, 1]] * 8,
            "obs_binary": [False] * 6 + [True, True],
            "action_names": ["A", "B", "C", "D"],
        }
        self.config_path.write_text(json.dumps(config))
        self.adapter = ModelAdapter(self.model_path, self.config_path)

    def test_metadata_includes_env_config(self):
        meta = self.adapter.get_metadata()
        assert meta["env_name"] == "TestEnv"
        assert meta["obs_names"] == [f"feat_{i}" for i in range(8)]
        assert meta["action_names"] == ["A", "B", "C", "D"]
        assert meta["obs_binary"][6] is True


# ---- Server integration tests ----


class TestServer:
    @pytest.fixture(autouse=True)
    def setup(self, tmp_path):
        self.model_path = tmp_path / "discrete.onnx"
        _make_discrete_model(self.model_path, obs_dim=4, num_actions=2)

        config = {
            "env_name": "TestEnv",
            "obs_names": ["a", "b", "c", "d"],
            "obs_ranges": [[-1, 1]] * 4,
            "obs_binary": [False] * 4,
            "action_names": ["Left", "Right"],
        }
        self.config_path = tmp_path / "env.json"
        self.config_path.write_text(json.dumps(config))

        from tools.policy_viz.server import PolicyVizHandler
        from functools import partial
        from http.server import HTTPServer

        adapter = ModelAdapter(self.model_path, self.config_path)
        handler = partial(PolicyVizHandler, adapter=adapter)
        self.server = HTTPServer(("127.0.0.1", 0), handler)
        self.port = self.server.server_address[1]
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def teardown_method(self):
        self.server.shutdown()

    def _get(self, path):
        conn = HTTPConnection("127.0.0.1", self.port)
        conn.request("GET", path)
        resp = conn.getresponse()
        body = resp.read()
        conn.close()
        return resp.status, body

    def _post(self, path, data):
        body = json.dumps(data).encode()
        conn = HTTPConnection("127.0.0.1", self.port)
        conn.request("POST", path, body=body, headers={"Content-Type": "application/json"})
        resp = conn.getresponse()
        resp_body = resp.read()
        conn.close()
        return resp.status, resp_body

    def test_get_index(self):
        status, body = self._get("/")
        assert status == 200
        assert b"<!DOCTYPE html>" in body

    def test_get_static_css(self):
        status, body = self._get("/style.css")
        assert status == 200
        assert b"--bg" in body

    def test_get_metadata(self):
        status, body = self._get("/api/metadata")
        assert status == 200
        meta = json.loads(body)
        assert meta["obs_dim"] == 4
        assert meta["action_type"] == "discrete"
        assert meta["env_name"] == "TestEnv"

    def test_post_infer(self):
        status, body = self._post("/api/infer", {"obs": [0.0, 0.0, 0.0, 0.0]})
        assert status == 200
        result = json.loads(body)
        assert "probabilities" in result
        assert len(result["probabilities"]) == 2
        assert "value" in result

    def test_post_infer_stateful(self):
        status, body = self._post("/api/infer", {"obs": [0.1, 0.2, 0.3, 0.4], "stateful": True})
        assert status == 200
        result = json.loads(body)
        assert "probabilities" in result

    def test_post_reset_state(self):
        status, body = self._post("/api/reset_state", {})
        assert status == 200
        result = json.loads(body)
        assert result["ok"] is True

    def test_post_infer_missing_obs(self):
        status, _ = self._post("/api/infer", {"wrong_key": []})
        assert status == 400
