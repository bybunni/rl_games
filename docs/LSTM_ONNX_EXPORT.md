# LSTM ONNX Export Validation

This document describes how the LSTM ONNX export pipeline is tested and
validated across all supported network configuration combinations.

## Overview

The A2C network builder supports LSTM recurrent layers with two key
configuration axes that change the network topology:

| Parameter | Description |
|-----------|-------------|
| `before_mlp` | When `True`, the LSTM processes raw features *before* the MLP layers. When `False` (default), the MLP runs first and the LSTM follows. |
| `concat_input` | When `True` (and `before_mlp` is `False`), the original features are concatenated with the MLP output before feeding into the LSTM, giving the recurrent layer direct access to the raw input. |

These two flags produce four distinct network topologies that must all
export cleanly to ONNX and produce numerically identical outputs to the
PyTorch model.

## Tested Configurations

| Variant | `before_mlp` | `concat_input` | Data flow |
|---------|-------------|----------------|-----------|
| `after_mlp` | `False` | `False` | obs &rarr; MLP &rarr; LSTM &rarr; heads |
| `after_mlp_concat` | `False` | `True` | obs &rarr; MLP &rarr; concat(MLP, obs) &rarr; LSTM &rarr; heads |
| `before_mlp` | `True` | `False` | obs &rarr; LSTM &rarr; MLP &rarr; heads |
| `before_mlp_concat` | `True` | `True` | obs &rarr; LSTM &rarr; MLP &rarr; heads |

When `before_mlp=True`, `concat_input` has no effect on the construction
path (the LSTM input is the raw feature size directly), but the
combination is still tested to ensure it does not break.

## How the Test Works

The test lives in `tests/test_lstm_onnx_export.py` and is parametrized
over all four `(before_mlp, concat_input)` combinations. No pre-trained
checkpoint is needed &mdash; the test constructs networks from scratch with
random weights and validates *export correctness*, not trained model
quality.

### Steps per variant

1. **Build the network** &mdash; Instantiate `A2CBuilder` with the LSTM
   config, producing a shared (non-separate) discrete A2C network with
   `input_shape=(4,)` and `actions_num=2` (matching CartPole-v1).

2. **Export to ONNX** &mdash; Wrap the network in a `ModelWrapper` (same
   pattern as the GTrXL export script) that bypasses `torch.compile` and
   distribution sampling. Use `TracingAdapter` to flatten the dict-based
   inputs `{"obs": tensor, "rnn_states": (h, c)}` into a flat tuple of
   tensors, then `torch.jit.trace` and `torch.onnx.export`.

3. **Validate the ONNX file** &mdash; Load it with `onnx.load` and run
   `onnx.checker.check_model`.

4. **Collect trajectory data** &mdash; Roll out 32 steps in CartPole-v1
   (seeded), recording at each step:
   - Observation and LSTM input state `(h, c)`
   - PyTorch-computed logits, value, and output state `(h', c')`
   - Actions are chosen greedily (argmax of logits); on episode done the
     LSTM state is reset to zeros.

5. **Replay through ONNX Runtime** &mdash; Feed each recorded
   `(obs, h, c)` pair into the ONNX model via `onnxruntime.InferenceSession`
   and compare the four outputs (logits, value, h_out, c_out) against the
   PyTorch reference.

6. **Assert parity** &mdash; The maximum absolute difference across all
   outputs and all steps must be &le; `1e-4`.

### ONNX I/O layout

The `TracingAdapter` flattens the input dict into three tensors and the
output tuple into four tensors:

| Direction | Name | Shape | Description |
|-----------|------|-------|-------------|
| Input | `obs` | `(1, 4)` | Observation |
| Input | `rnn_h` | `(num_layers, 1, 64)` | LSTM hidden state |
| Input | `rnn_c` | `(num_layers, 1, 64)` | LSTM cell state |
| Output | `logits` | `(1, 2)` | Action logits |
| Output | `value` | `(1, 1)` | State value |
| Output | `rnn_h_out` | `(num_layers, 1, 64)` | Updated hidden state |
| Output | `rnn_c_out` | `(num_layers, 1, 64)` | Updated cell state |

## Running the Tests

```bash
# Run just the LSTM export tests
pytest tests/test_lstm_onnx_export.py -v

# Run all ONNX export tests (LSTM + GTrXL)
pytest tests/test_lstm_onnx_export.py tests/test_gtrxl_onnx_export.py -v
```

### Dependencies

The test will be automatically skipped if any of these are missing:

- `onnx`
- `onnxruntime`
- `gymnasium` (or `gym`)

Install them with:

```bash
pip install onnx onnxruntime gymnasium
```

## Network Config Reference

The LSTM-related fields under the `rnn` key in the network config YAML:

```yaml
network:
  name: actor_critic
  mlp:
    units: [64, 64]
    activation: relu
    initializer:
      name: default
  space:
    discrete:
  rnn:
    name: lstm
    units: 64          # LSTM hidden size
    layers: 1          # number of stacked LSTM layers
    before_mlp: False  # place LSTM before or after the MLP
    concat_input: True # concatenate raw features to MLP output before LSTM
    layer_norm: False  # optional layer norm after LSTM output
```
