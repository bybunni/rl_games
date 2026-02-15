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

    # With high gate bias, z ≈ 0, so output ≈ x (identity)
    gate_high = GRUGate(d_model, gate_bias=10.0)
    x_test = torch.randn(2, 4, d_model)
    y_test = torch.zeros(2, 4, d_model)  # zero sublayer output
    out_identity = gate_high(x_test, y_test)
    # Should be close to x when bias is very high and y is zero
    assert torch.allclose(out_identity, x_test, atol=0.1), \
        f"Gate with high bias should approximate identity. Max diff: {(out_identity - x_test).abs().max()}"


def test_relative_positional_encoding():
    """Test positional encoding shapes."""
    from rl_games.algos_torch.gtrxl import RelativePositionalEncoding

    d_model = 64
    pe = RelativePositionalEncoding(d_model, max_len=512)

    out = pe(32)
    assert out.shape == (32, 1, d_model)

    # Different lengths should work
    out2 = pe(128)
    assert out2.shape == (128, 1, d_model)


def test_gtrxl_layer():
    """Test single GTrXL layer forward pass."""
    from rl_games.algos_torch.gtrxl import GTrXLLayer

    d_model = 64
    num_heads = 4
    seq_len = 8
    batch_size = 4
    mem_len = 16

    layer = GTrXLLayer(d_model, num_heads, dropout=0.0, gate_bias=2.0)

    # Without memory
    x = torch.randn(seq_len, batch_size, d_model)
    out = layer(x)
    assert out.shape == (seq_len, batch_size, d_model)

    # With memory
    memory = torch.randn(mem_len, batch_size, d_model)
    total_len = mem_len + seq_len
    attn_mask = torch.zeros(seq_len, total_len)
    out_with_mem = layer(x, memory=memory, attn_mask=attn_mask)
    assert out_with_mem.shape == (seq_len, batch_size, d_model)


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

    # Without memory (first call)
    out, new_memory = model(x)
    assert out.shape == (seq_len, batch_size, d_model)
    assert len(new_memory) == num_layers
    for mem in new_memory:
        assert mem.shape == (memory_length, batch_size, d_model)

    # With memory (subsequent call)
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


def test_gtrxl_done_masks():
    """Test that done masks properly reset memory."""
    from rl_games.algos_torch.gtrxl import GTrXL

    model = GTrXL(input_dim=16, d_model=32, num_layers=2, num_heads=4, memory_length=8)

    batch_size = 4
    seq_len = 4
    x = torch.randn(seq_len, batch_size, 16)
    memory = [torch.ones(8, batch_size, 32) for _ in range(2)]

    # Mark env 1 and 3 as done
    done_masks = torch.zeros(seq_len, batch_size)
    done_masks[2, 1] = 1.0  # env 1 done at step 2
    done_masks[0, 3] = 1.0  # env 3 done at step 0

    out, new_memory = model(x, memory=memory, done_masks=done_masks)
    assert out.shape == (seq_len, batch_size, 32)


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

    # Check that all model parameters received gradients
    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"No gradient for {name}"


def test_gtrxl_builder_continuous():
    """Test GTrXLBuilder with continuous action space."""
    from rl_games.algos_torch.gtrxl_builder import GTrXLBuilder

    params = {
        'space': {
            'continuous': {
                'mu_activation': 'None',
                'sigma_activation': 'None',
                'mu_init': {'name': 'default'},
                'sigma_init': {'name': 'const_initializer', 'val': 0},
                'fixed_sigma': True,
            }
        },
        'gtrxl': {
            'embedding_dim': 64,
            'num_layers': 2,
            'num_heads': 4,
            'memory_length': 16,
            'gru_gate_bias': 2.0,
            'dropout': 0.0,
        },
        'initializer': {'name': 'default'},
    }

    builder = GTrXLBuilder()
    builder.load(params)

    net = builder.build(
        'test',
        actions_num=3,
        input_shape=(11,),
        value_size=1,
        num_seqs=4,
    )

    assert net.is_rnn()
    assert not net.is_separate_critic()

    # Test default state
    states = net.get_default_rnn_state()
    assert len(states) == 1
    expected_dim0 = 2 * 16  # num_layers * memory_length
    assert states[0].shape == (expected_dim0, 4, 64)

    # Test forward pass (inference: seq_length=1)
    batch_size = 4
    obs = torch.randn(batch_size, 11)
    obs_dict = {
        'obs': obs,
        'rnn_states': [s[:, :batch_size, :] for s in states],
        'seq_length': 1,
    }

    mu, sigma, value, new_states = net(obs_dict)
    assert mu.shape == (batch_size, 3)
    assert value.shape == (batch_size, 1)
    assert len(new_states) == 1
    assert new_states[0].shape[1] == batch_size

    # Test forward pass (training: seq_length=4)
    seq_length = 4
    num_seqs = 8
    train_batch = num_seqs * seq_length
    obs_train = torch.randn(train_batch, 11)
    train_states = (torch.zeros(expected_dim0, num_seqs, 64),)
    obs_dict_train = {
        'obs': obs_train,
        'rnn_states': train_states,
        'seq_length': seq_length,
        'dones': torch.zeros(train_batch, 1),
    }

    mu_t, sigma_t, value_t, new_states_t = net(obs_dict_train)
    assert mu_t.shape == (train_batch, 3)
    assert value_t.shape == (train_batch, 1)


def test_gtrxl_builder_discrete():
    """Test GTrXLBuilder with discrete action space."""
    from rl_games.algos_torch.gtrxl_builder import GTrXLBuilder

    params = {
        'space': {
            'discrete': {}
        },
        'gtrxl': {
            'embedding_dim': 32,
            'num_layers': 2,
            'num_heads': 4,
            'memory_length': 8,
        },
        'initializer': {'name': 'default'},
    }

    builder = GTrXLBuilder()
    builder.load(params)

    net = builder.build(
        'test',
        actions_num=5,
        input_shape=(8,),
        value_size=1,
        num_seqs=2,
    )

    batch_size = 2
    obs = torch.randn(batch_size, 8)
    states = net.get_default_rnn_state()
    obs_dict = {
        'obs': obs,
        'rnn_states': states,
        'seq_length': 1,
    }

    logits, value, new_states = net(obs_dict)
    assert logits.shape == (batch_size, 5)
    assert value.shape == (batch_size, 1)


def test_gtrxl_builder_registration():
    """Test that GTrXLBuilder is registered in model_builder."""
    from rl_games.algos_torch.model_builder import NetworkBuilder

    nb = NetworkBuilder()
    assert 'gtrxl_actor_critic' in nb.network_factory._builders


if __name__ == '__main__':
    test_gru_gate()
    print("PASS: test_gru_gate")

    test_relative_positional_encoding()
    print("PASS: test_relative_positional_encoding")

    test_gtrxl_layer()
    print("PASS: test_gtrxl_layer")

    test_gtrxl_forward()
    print("PASS: test_gtrxl_forward")

    test_gtrxl_memory_init()
    print("PASS: test_gtrxl_memory_init")

    test_gtrxl_done_masks()
    print("PASS: test_gtrxl_done_masks")

    test_gtrxl_gradient_flow()
    print("PASS: test_gtrxl_gradient_flow")

    test_gtrxl_builder_continuous()
    print("PASS: test_gtrxl_builder_continuous")

    test_gtrxl_builder_discrete()
    print("PASS: test_gtrxl_builder_discrete")

    test_gtrxl_builder_registration()
    print("PASS: test_gtrxl_builder_registration")

    print("\nAll tests passed!")
