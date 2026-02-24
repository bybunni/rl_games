"""ONNX model introspection and single-step inference for policy visualization."""

import json
from pathlib import Path

import numpy as np
import onnxruntime as ort


class ModelAdapter:
    """Wraps an ONNX Runtime session with auto-detected I/O conventions.

    Handles both discrete (logits) and continuous (mu/sigma) policies,
    with optional recurrent state management.
    """

    def __init__(self, model_path, env_config_path=None):
        self._session = ort.InferenceSession(
            str(model_path), providers=["CPUExecutionProvider"]
        )
        self._env_config = None
        if env_config_path is not None:
            with open(env_config_path) as f:
                self._env_config = json.load(f)

        self._inputs = self._session.get_inputs()
        self._outputs = self._session.get_outputs()

        # Classify inputs
        self._obs_input = None
        self._state_inputs = []
        for inp in self._inputs:
            if inp.name == "obs" or (self._obs_input is None and len(inp.shape) == 2):
                self._obs_input = inp
            else:
                self._state_inputs.append(inp)

        if self._obs_input is None:
            # Fallback: first input is obs
            self._obs_input = self._inputs[0]
            self._state_inputs = list(self._inputs[1:])

        # Determine obs_dim from input shape
        obs_shape = self._obs_input.shape
        self._obs_dim = obs_shape[-1] if isinstance(obs_shape[-1], int) else None

        # Classify outputs
        self._action_type = None
        self._logits_output = None
        self._mu_output = None
        self._sigma_output = None
        self._value_output = None
        self._state_outputs = []

        for out in self._outputs:
            name_lower = out.name.lower()
            if "logit" in name_lower:
                self._logits_output = out
                self._action_type = "discrete"
            elif name_lower in ("mu", "action"):
                self._mu_output = out
                self._action_type = "continuous"
            elif "log_std" in name_lower or "sigma" in name_lower or name_lower == "log_std":
                self._sigma_output = out
            elif "value" in name_lower:
                self._value_output = out
            elif "state" in name_lower:
                self._state_outputs.append(out)

        # If we still don't know the action type, infer from output count/names
        if self._action_type is None:
            # Heuristic: if first output has more than 1 element in last dim, it's logits
            first_out = self._outputs[0]
            if first_out.shape and len(first_out.shape) >= 2:
                last_dim = first_out.shape[-1]
                if isinstance(last_dim, int) and last_dim > 1:
                    self._logits_output = first_out
                    self._action_type = "discrete"
                else:
                    self._mu_output = first_out
                    self._action_type = "continuous"

        # Determine num_actions
        if self._action_type == "discrete" and self._logits_output is not None:
            last = self._logits_output.shape[-1]
            self._num_actions = last if isinstance(last, int) else None
        elif self._action_type == "continuous" and self._mu_output is not None:
            last = self._mu_output.shape[-1]
            self._num_actions = last if isinstance(last, int) else None
        else:
            self._num_actions = None

        # Mutable RNN state (for stateful mode)
        self._rnn_state = self.get_zero_rnn_state()

    @property
    def is_recurrent(self):
        return len(self._state_inputs) > 0

    def get_zero_rnn_state(self):
        """Return dict of zero arrays for each RNN state input, or None."""
        if not self._state_inputs:
            return None
        state = {}
        for inp in self._state_inputs:
            shape = []
            for dim in inp.shape:
                shape.append(dim if isinstance(dim, int) else 1)
            state[inp.name] = np.zeros(shape, dtype=np.float32)
        return state

    def reset_rnn_state(self):
        self._rnn_state = self.get_zero_rnn_state()

    def get_metadata(self):
        """Return model metadata for UI setup."""
        meta = {
            "obs_dim": self._obs_dim,
            "action_type": self._action_type,
            "num_actions": self._num_actions,
            "is_recurrent": self.is_recurrent,
        }

        if self._env_config:
            meta["env_name"] = self._env_config.get("env_name")
            meta["obs_names"] = self._env_config.get("obs_names")
            meta["obs_ranges"] = self._env_config.get("obs_ranges")
            meta["obs_binary"] = self._env_config.get("obs_binary")
            meta["action_names"] = self._env_config.get("action_names")

        return meta

    def infer(self, obs, stateful=False):
        """Run one inference step.

        Args:
            obs: list/array of observation values, length obs_dim.
            stateful: if True, use and update internal RNN state.
                      if False, use zero RNN state (default for sensitivity analysis).

        Returns:
            dict with action outputs, value, and optional state info.
        """
        obs_array = np.array(obs, dtype=np.float32).reshape(1, -1)
        feed = {self._obs_input.name: obs_array}

        if self._state_inputs:
            if stateful and self._rnn_state is not None:
                for inp in self._state_inputs:
                    feed[inp.name] = self._rnn_state[inp.name]
            else:
                zero_state = self.get_zero_rnn_state()
                for inp in self._state_inputs:
                    feed[inp.name] = zero_state[inp.name]

        raw_outputs = self._session.run(None, feed)

        # Build output name -> array map
        output_map = {}
        for out_meta, arr in zip(self._outputs, raw_outputs):
            output_map[out_meta.name] = arr

        result = {}

        if self._action_type == "discrete" and self._logits_output:
            logits = output_map[self._logits_output.name].flatten()
            # Softmax
            exp_logits = np.exp(logits - np.max(logits))
            probs = exp_logits / exp_logits.sum()
            result["logits"] = logits.tolist()
            result["probabilities"] = probs.tolist()
            result["best_action"] = int(np.argmax(probs))
        elif self._action_type == "continuous" and self._mu_output:
            mu = output_map[self._mu_output.name].flatten()
            result["mu"] = mu.tolist()
            if self._sigma_output and self._sigma_output.name in output_map:
                log_std = output_map[self._sigma_output.name].flatten()
                result["sigma"] = np.exp(log_std).tolist()
            result["best_action"] = mu.tolist()

        if self._value_output and self._value_output.name in output_map:
            val = output_map[self._value_output.name].flatten()
            result["value"] = float(val[0])

        # Update RNN state if stateful
        if stateful and self._state_outputs:
            if self._rnn_state is None:
                self._rnn_state = {}
            for state_out in self._state_outputs:
                arr = output_map.get(state_out.name)
                if arr is None:
                    continue
                # Match output to corresponding input by name pattern
                matched = self._match_state_output_to_input(state_out.name)
                if matched:
                    self._rnn_state[matched] = arr

        result["action_type"] = self._action_type

        return result

    def _match_state_output_to_input(self, output_name):
        """Match a state output name to its corresponding state input name.

        Handles conventions like:
        - input: 'rnn_state', output: 'rnn_state_out'
        - input: 'out_state.1', output: 'out_state'
        - input: 'hidden_state.1', output: 'hidden_state'
        """
        # Direct match
        for inp in self._state_inputs:
            if inp.name == output_name:
                return inp.name

        # Strip '_out' suffix from output and match
        stripped = output_name.replace("_out", "")
        for inp in self._state_inputs:
            inp_base = inp.name.split(".")[0]
            if inp_base == stripped or inp.name == stripped:
                return inp.name

        # Fuzzy: find inputs whose base name is contained in output name
        for inp in self._state_inputs:
            inp_base = inp.name.split(".")[0]
            if inp_base in output_name or output_name in inp_base:
                return inp.name

        # Last resort: if only one state input, use it
        if len(self._state_inputs) == 1:
            return self._state_inputs[0].name

        return None
