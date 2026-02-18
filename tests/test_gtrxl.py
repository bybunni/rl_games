"""Tests for GTrXL implementation."""

import torch


def test_gru_gate():
    """Test GRUGate forward pass and identity initialization."""
    from rl_games.algos_torch.gtrxl import GRUGate

    d_model = 64
    gate = GRUGate(d_model, gate_bias=2.0)

    x = torch.randn(4, 8, d_model)
    y = torch.randn(4, 8, d_model)

    out = gate(x, y)
    assert out.shape == (4, 8, d_model)

    gate_high = GRUGate(d_model, gate_bias=10.0)
    x_test = torch.randn(2, 4, d_model)
    y_test = torch.zeros(2, 4, d_model)
    out_identity = gate_high(x_test, y_test)
    assert torch.allclose(out_identity, x_test, atol=0.1), (
        "Gate with high bias should approximate identity. "
        f"Max diff: {(out_identity - x_test).abs().max()}"
    )


def test_relative_positional_encoding():
    """Test positional encoding shapes."""
    from rl_games.algos_torch.gtrxl import RelativePositionalEncoding

    d_model = 64
    pe = RelativePositionalEncoding(d_model, max_len=512)

    out = pe(32)
    assert out.shape == (32, d_model)

    out2 = pe(128)
    assert out2.shape == (128, d_model)


def test_relative_multihead_attention():
    """Test relative attention forward pass and shape."""
    from rl_games.algos_torch.gtrxl import RelativeMultiHeadAttention, RelativePositionalEncoding

    d_model = 64
    num_heads = 4
    seq_len = 8
    batch_size = 3
    mem_len = 5

    attn = RelativeMultiHeadAttention(d_model, num_heads, dropout=0.0)
    pe = RelativePositionalEncoding(d_model, max_len=512)

    x = torch.randn(seq_len, batch_size, d_model)
    memory = torch.randn(mem_len, batch_size, d_model)
    rel_pos = pe(seq_len + mem_len)
    mask = torch.zeros(batch_size, seq_len, seq_len + mem_len, dtype=torch.bool)

    out = attn(x, memory=memory, rel_pos=rel_pos, attn_mask=mask)
    assert out.shape == (seq_len, batch_size, d_model)


def test_ff_activation_config():
    """FF activation should be configurable between ReLU and GELU."""
    from rl_games.algos_torch.gtrxl import GTrXLLayer

    layer_relu = GTrXLLayer(d_model=32, num_heads=4, ff_activation="relu")
    assert isinstance(layer_relu.ff[1], torch.nn.ReLU)

    layer_gelu = GTrXLLayer(d_model=32, num_heads=4, ff_activation="gelu")
    assert isinstance(layer_gelu.ff[1], torch.nn.GELU)

    try:
        _ = GTrXLLayer(d_model=32, num_heads=4, ff_activation="swish")
    except ValueError:
        pass
    else:
        assert False, "Expected unsupported ff_activation to raise ValueError"



def test_gtrxl_layer():
    """Test single GTrXL layer forward pass."""
    from rl_games.algos_torch.gtrxl import GTrXLLayer, RelativePositionalEncoding

    d_model = 64
    num_heads = 4
    seq_len = 8
    batch_size = 4
    mem_len = 16

    layer = GTrXLLayer(d_model, num_heads, dropout=0.0, gate_bias=2.0)
    pe = RelativePositionalEncoding(d_model, max_len=512)

    x = torch.randn(seq_len, batch_size, d_model)

    # Without memory
    mask_no_mem = torch.zeros(batch_size, seq_len, seq_len, dtype=torch.bool)
    out = layer(x, rel_pos=pe(seq_len), attn_mask=mask_no_mem)
    assert out.shape == (seq_len, batch_size, d_model)

    # With memory
    memory = torch.randn(mem_len, batch_size, d_model)
    mask = torch.zeros(batch_size, seq_len, mem_len + seq_len, dtype=torch.bool)
    out_with_mem = layer(x, memory=memory, rel_pos=pe(mem_len + seq_len), attn_mask=mask)
    assert out_with_mem.shape == (seq_len, batch_size, d_model)


def test_identity_reorder_relu_before_gates():
    """ReLU should be applied before both gated residual connections."""
    from rl_games.algos_torch.gtrxl import GTrXLLayer, RelativePositionalEncoding

    d_model = 8
    num_heads = 2
    seq_len = 4
    batch_size = 2

    layer = GTrXLLayer(d_model, num_heads, dropout=0.0, gate_bias=2.0)
    pe = RelativePositionalEncoding(d_model, max_len=128)

    # Force deterministic negative outputs from both submodules before ReLU.
    with torch.no_grad():
        layer.attn.q_proj.weight.zero_()
        layer.attn.k_proj.weight.zero_()
        layer.attn.v_proj.weight.fill_(-1.0 / d_model)
        layer.attn.r_proj.weight.zero_()
        layer.attn.o_proj.weight.copy_(torch.eye(d_model))
        layer.attn.u_bias.zero_()
        layer.attn.v_bias.zero_()

        ff_in = layer.ff[0]
        ff_out = layer.ff[3]
        ff_in.weight.zero_()
        ff_in.bias.fill_(1.0)
        ff_out.weight.zero_()
        ff_out.bias.fill_(-1.0)

    captured = {}

    orig_attn_gate = layer.gate_attn.forward
    orig_ff_gate = layer.gate_ff.forward

    def capture_attn(x, y):
        captured["attn_y"] = y.detach()
        return orig_attn_gate(x, y)

    def capture_ff(x, y):
        captured["ff_y"] = y.detach()
        return orig_ff_gate(x, y)

    layer.gate_attn.forward = capture_attn
    layer.gate_ff.forward = capture_ff

    x = torch.ones(seq_len, batch_size, d_model)
    mask = torch.zeros(batch_size, seq_len, seq_len, dtype=torch.bool)
    _ = layer(x, rel_pos=pe(seq_len), attn_mask=mask)

    assert "attn_y" in captured
    assert "ff_y" in captured
    assert captured["attn_y"].min().item() >= -1e-8
    assert captured["ff_y"].min().item() >= -1e-8



def test_gtrxl_forward():
    """Test full GTrXL forward pass."""
    from rl_games.algos_torch.gtrxl import GTrXL

    input_dim = 32
    d_model = 64
    num_layers = 2
    num_heads = 4
    memory_length = 16
    seq_len = 8
    batch_size = 4

    model = GTrXL(
        input_dim=input_dim,
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads,
        memory_length=memory_length,
        dropout=0.0,
        gate_bias=2.0,
    )

    x = torch.randn(seq_len, batch_size, input_dim)

    out, new_memory = model(x)
    assert out.shape == (seq_len, batch_size, d_model)
    assert len(new_memory) == num_layers
    for mem in new_memory:
        assert mem.shape == (memory_length, batch_size, d_model)

    out2, new_memory2 = model(x, memory=new_memory)
    assert out2.shape == (seq_len, batch_size, d_model)
    assert len(new_memory2) == num_layers



def test_gtrxl_memory_init():
    """Test memory initialization."""
    from rl_games.algos_torch.gtrxl import GTrXL

    model = GTrXL(input_dim=16, d_model=32, num_layers=3, num_heads=4, memory_length=8)

    memory = model.init_memory(batch_size=6)
    assert len(memory) == 3
    for mem in memory:
        assert mem.shape == (8, 6, 32)
        assert (mem == 0).all()



def test_gtrxl_done_attention_masks():
    """Done masks should block attention across episode boundaries within a segment."""
    from rl_games.algos_torch.gtrxl import GTrXL

    model = GTrXL(input_dim=8, d_model=8, num_layers=1, num_heads=2, memory_length=3)

    seq_len = 4
    batch_size = 1
    memory = torch.ones(3, batch_size, 8)

    # done at t=1, so t>=2 must not attend to memory or to steps 0..1
    done_masks = torch.tensor([[0.0], [1.0], [0.0], [0.0]])

    mask = model._build_attention_mask(seq_len, memory, done_masks)
    assert mask.shape == (batch_size, seq_len, 3 + seq_len)

    # Query at t=2 can only attend to itself (current index = mem_len + 2 = 5).
    assert mask[0, 2, 5].item() is False
    assert mask[0, 2, 0].item() is True
    assert mask[0, 2, 1].item() is True
    assert mask[0, 2, 2].item() is True
    assert mask[0, 2, 3].item() is True
    assert mask[0, 2, 4].item() is True



def test_gtrxl_done_memory_update():
    """Memory update should drop context up to and including the latest done step."""
    from rl_games.algos_torch.gtrxl import GTrXL

    model = GTrXL(input_dim=2, d_model=2, num_layers=1, num_heads=1, memory_length=3)

    memory = torch.tensor(
        [
            [[1.0, 1.0]],
            [[2.0, 2.0]],
            [[3.0, 3.0]],
        ]
    )
    layer_input = torch.tensor(
        [
            [[10.0, 10.0]],
            [[11.0, 11.0]],
            [[12.0, 12.0]],
            [[13.0, 13.0]],
        ]
    )
    done_masks = torch.tensor([[0.0], [1.0], [0.0], [0.0]])

    updated = model._update_memory(memory, layer_input, done_masks)
    assert updated.shape == (3, 1, 2)

    # With done at t=1, only states after t=1 remain valid: x2, x3.
    assert torch.allclose(updated[0], torch.zeros_like(updated[0]))
    assert torch.allclose(updated[1], torch.tensor([[12.0, 12.0]]))
    assert torch.allclose(updated[2], torch.tensor([[13.0, 13.0]]))



def test_gtrxl_gradient_flow():
    """Test that gradients flow through the GTrXL."""
    from rl_games.algos_torch.gtrxl import GTrXL

    model = GTrXL(input_dim=16, d_model=32, num_layers=2, num_heads=4, memory_length=8)

    x = torch.randn(4, 2, 16, requires_grad=True)
    out, _ = model(x)
    loss = out.sum()
    loss.backward()

    assert x.grad is not None
    assert x.grad.abs().sum() > 0

    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"No gradient for {name}"



def test_gtrxl_builder_continuous():
    """Test GTrXLBuilder with continuous action space."""
    from rl_games.algos_torch.gtrxl_builder import GTrXLBuilder

    params = {
        "space": {
            "continuous": {
                "mu_activation": "None",
                "sigma_activation": "None",
                "mu_init": {"name": "default"},
                "sigma_init": {"name": "const_initializer", "val": 0},
                "fixed_sigma": True,
            }
        },
        "gtrxl": {
            "embedding_dim": 64,
            "num_layers": 2,
            "num_heads": 4,
            "ff_activation": "gelu",
            "memory_length": 16,
            "gru_gate_bias": 2.0,
            "dropout": 0.0,
        },
        "initializer": {"name": "default"},
    }

    builder = GTrXLBuilder()
    builder.load(params)

    net = builder.build(
        "test",
        actions_num=3,
        input_shape=(11,),
        value_size=1,
        num_seqs=4,
    )

    assert net.is_rnn()
    assert not net.is_separate_critic()
    assert isinstance(net.gtrxl.layers[0].ff[1], torch.nn.GELU)

    states = net.get_default_rnn_state()
    assert len(states) == 1
    expected_dim0 = 2 * 16
    assert states[0].shape == (expected_dim0, 4, 64)

    batch_size = 4
    obs = torch.randn(batch_size, 11)
    obs_dict = {
        "obs": obs,
        "rnn_states": [s[:, :batch_size, :] for s in states],
        "seq_length": 1,
    }

    mu, sigma, value, new_states = net(obs_dict)
    assert mu.shape == (batch_size, 3)
    assert value.shape == (batch_size, 1)
    assert len(new_states) == 1
    assert new_states[0].shape[1] == batch_size

    seq_length = 4
    num_seqs = 8
    train_batch = num_seqs * seq_length
    obs_train = torch.randn(train_batch, 11)
    train_states = (torch.zeros(expected_dim0, num_seqs, 64),)
    obs_dict_train = {
        "obs": obs_train,
        "rnn_states": train_states,
        "seq_length": seq_length,
        "dones": torch.zeros(train_batch, 1),
    }

    mu_t, sigma_t, value_t, _ = net(obs_dict_train)
    assert mu_t.shape == (train_batch, 3)
    assert value_t.shape == (train_batch, 1)



def test_gtrxl_builder_discrete():
    """Test GTrXLBuilder with discrete action space."""
    from rl_games.algos_torch.gtrxl_builder import GTrXLBuilder

    params = {
        "space": {"discrete": {}},
        "gtrxl": {
            "embedding_dim": 32,
            "num_layers": 2,
            "num_heads": 4,
            "memory_length": 8,
        },
        "initializer": {"name": "default"},
    }

    builder = GTrXLBuilder()
    builder.load(params)

    net = builder.build(
        "test",
        actions_num=5,
        input_shape=(8,),
        value_size=1,
        num_seqs=2,
    )

    batch_size = 2
    obs = torch.randn(batch_size, 8)
    states = net.get_default_rnn_state()
    obs_dict = {
        "obs": obs,
        "rnn_states": states,
        "seq_length": 1,
    }

    logits, value, _ = net(obs_dict)
    assert logits.shape == (batch_size, 5)
    assert value.shape == (batch_size, 1)



def test_gtrxl_builder_registration():
    """Test that GTrXLBuilder is registered in model_builder."""
    from rl_games.algos_torch.model_builder import NetworkBuilder

    nb = NetworkBuilder()
    assert "gtrxl_actor_critic" in nb.network_factory._builders
