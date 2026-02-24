# Policy Sensitivity Analysis Tool

An interactive web tool for probing trained RL policies. Adjust observation inputs via sliders and see how action probabilities and value estimates change in real-time. Useful for understanding which observation features matter most to a policy's decisions.

## Quick Start

1. Export a trained checkpoint to ONNX (see [Exporting to ONNX](#exporting-to-onnx) below).
2. Launch the tool:
   ```bash
   python -m tools.policy_viz \
     --model path/to/model.onnx \
     --env-config tools/policy_viz/envs/lunarlander_v2.json
   ```
3. Open `http://localhost:8080` in a browser.

## Requirements

No additional dependencies — uses only `onnxruntime` and `numpy` (already in the training stack) plus Python's stdlib `http.server`.

## CLI Options

```
python -m tools.policy_viz --model MODEL [--env-config ENV_CONFIG] [--port PORT]
```

| Flag           | Required | Default | Description                                    |
|----------------|----------|---------|------------------------------------------------|
| `--model`      | Yes      | —       | Path to an ONNX policy model                   |
| `--env-config` | No       | None    | Path to env config JSON for labels and ranges   |
| `--port`       | No       | 8080    | HTTP server port                                |

## Exporting to ONNX

### GTrXL (discrete)

Use the existing export script with a trained LunarLander checkpoint:

```bash
python scripts/export_gtrxl_onnx.py \
  --checkpoint runs/lunarlander_gtrxl/nn/lunarlander_gtrxl.pth \
  --onnx-out lunarlander_gtrxl.onnx
```

The script runs a 256-step parity validation between PyTorch and ONNX Runtime outputs automatically.

### LSTM (continuous)

See the notebook `notebooks/train_and_export_onnx_example_lstm_continuous.ipynb` for an end-to-end Pendulum-v1 example. The key export pattern:

```python
import rl_games.algos_torch.flatten as flatten

inputs = {
    'obs': torch.zeros((1,) + agent.obs_shape).to(agent.device),
    'rnn_states': agent.states,
}
adapter = flatten.TracingAdapter(ModelWrapper(agent.model), inputs, allow_non_tensor=True)
traced = torch.jit.trace(adapter, adapter.flattened_inputs, check_trace=False)
torch.onnx.export(traced, adapter.flattened_inputs, "model.onnx",
    input_names=['obs', 'out_state', 'hidden_state'],
    output_names=['mu', 'log_std', 'value', 'out_state', 'hidden_state'])
```

## Supported Model Formats

The tool auto-detects the model type from ONNX input/output names:

| Model Type            | Inputs                               | Outputs                                  |
|-----------------------|--------------------------------------|------------------------------------------|
| Discrete (feedforward)| `obs`                                | `logits`, `value`                        |
| Discrete (recurrent)  | `obs`, `rnn_state`                   | `logits`, `value`, `rnn_state_out`       |
| Continuous (feedforward)| `obs`                              | `mu`, `log_std`, `value`                 |
| Continuous (LSTM)     | `obs`, `out_state`, `hidden_state`   | `mu`, `log_std`, `value`, `out_state`, `hidden_state` |

The LSTM `.1` suffix naming quirk from `torch.onnx.export` (e.g. `out_state.1` as actual input name) is handled automatically.

## UI Overview

```
+----------------------------------+---------------------------+
| OBSERVATION INPUTS               | POLICY OUTPUTS            |
| x pos       [-------|---] 0.30   | Action Probabilities      |
| y pos       [-----|-----] 0.00   | Do nothing  [##    ] 15%  |
| x vel       [---|-------] -0.20  | Fire left   [#     ]  3%  |
| y vel       [-------|---] 0.30   | Fire right  [##    ] 10%  |
| angle       [-----|-----] 0.00   | Fire main   [######] 72%  |
| ang vel     [-----|-----] 0.00   |                           |
| left leg    [off / ON  ]         | Value: 0.45               |
| right leg   [off / ON  ]         | Best: Fire main           |
|                                  |                           |
| [Reset Obs]                      | [Stateless ▾] [Reset RNN] |
+----------------------------------+---------------------------+
```

- **Sliders** — one per observation feature, with numeric input for precise values. Binary features (e.g. leg contact) render as toggle switches.
- **Bar chart** — horizontal bars showing softmax action probabilities (discrete) or mu/sigma bands (continuous).
- **Value** — the critic's state-value estimate for the current observation.
- **RNN mode** — toggle between *Stateless* (zero RNN state each call, default for sensitivity analysis) and *Stateful* (accumulate RNN state across steps, with a step counter and reset button).

Sliders are debounced at 30ms so the output panel updates in near-real-time as you drag.

## Env Config Files

Env configs provide human-readable names and slider ranges. Three are bundled:

| File                                         | Environment     |
|----------------------------------------------|-----------------|
| `tools/policy_viz/envs/lunarlander_v2.json`  | LunarLander-v2  |
| `tools/policy_viz/envs/pendulum_v1.json`     | Pendulum-v1     |
| `tools/policy_viz/envs/cartpole_v1.json`     | CartPole-v1     |

Without `--env-config`, the UI falls back to generic labels ("Feature 0", "Action 0") and a default [-1, 1] range.

### Config format

```json
{
  "env_name": "LunarLander-v2",
  "obs_names": ["x pos", "y pos", "x vel", "y vel", "angle", "ang vel", "left leg", "right leg"],
  "obs_ranges": [[-1,1], [-1,1], [-1,1], [-1,1], [-1,1], [-1,1], [0,1], [0,1]],
  "obs_binary": [false, false, false, false, false, false, true, true],
  "action_names": ["Do nothing", "Fire left", "Fire main", "Fire right"]
}
```

| Field          | Type              | Description                                      |
|----------------|-------------------|--------------------------------------------------|
| `env_name`     | string            | Display name in the header                       |
| `obs_names`    | string[]          | Label for each observation dimension             |
| `obs_ranges`   | [min, max][]      | Slider range for each observation dimension      |
| `obs_binary`   | bool[]            | If true, renders a toggle switch instead of slider |
| `action_names` | string[]          | Label for each action                            |

## API Reference

The server exposes a simple JSON API used by the frontend:

| Method | Path               | Body                                      | Response                                                      |
|--------|--------------------|--------------------------------------------|---------------------------------------------------------------|
| GET    | `/api/metadata`    | —                                          | `{obs_dim, action_type, num_actions, is_recurrent, ...}`      |
| POST   | `/api/infer`       | `{obs: [float...], stateful: bool}`        | `{probabilities, logits, value, best_action, action_type}` (discrete) or `{mu, sigma, value, action_type}` (continuous) |
| POST   | `/api/reset_state` | `{}` (empty)                               | `{ok: true}`                                                  |

## Architecture

```
tools/policy_viz/
    __main__.py         # python -m entrypoint
    server.py           # stdlib HTTP server, CLI, routes
    model_adapter.py    # ONNX introspection + inference
    envs/               # bundled env config JSONs
    static/
        index.html      # single-page layout
        style.css       # dark-mode theme
        app.js          # sliders, fetch, canvas bar charts
```

- **`ModelAdapter`** wraps an `ort.InferenceSession` and classifies inputs (obs vs. RNN state) and outputs (logits/mu/sigma/value/state) by name and shape. It manages mutable RNN state for stateful mode.
- **`server.py`** is a `SimpleHTTPRequestHandler` subclass that routes `/api/*` to the adapter and serves static files from `static/`.
- **Frontend** is vanilla HTML/JS/CSS with no build step. Sliders are generated dynamically from `/api/metadata`.
