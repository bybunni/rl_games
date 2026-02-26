import torch


def _build_actor_critic_net(network_cfg):
    from rl_games.algos_torch.network_builder import A2CBuilder

    builder = A2CBuilder()
    builder.load(network_cfg)
    return builder.build(
        "actor_critic",
        actions_num=2,
        input_shape=(4,),
        value_size=1,
        num_seqs=1,
    )


def test_mlp_layer_norm_is_inserted_from_network_normalization():
    net = _build_actor_critic_net(
        {
            "name": "actor_critic",
            "separate": False,
            "space": {"discrete": {}},
            "normalization": "layer_norm",
            "mlp": {
                "units": [32, 16],
                "activation": "relu",
                "initializer": {"name": "default"},
            },
        }
    )

    ln_layers = [m for m in net.actor_mlp.modules() if isinstance(m, torch.nn.LayerNorm)]
    assert len(ln_layers) == 2


def test_mlp_batch_norm_is_inserted_from_nested_mlp_normalization():
    net = _build_actor_critic_net(
        {
            "name": "actor_critic",
            "separate": False,
            "space": {"discrete": {}},
            "mlp": {
                "units": [32, 16],
                "activation": "relu",
                "normalization": "batch_norm",
                "initializer": {"name": "default"},
            },
        }
    )

    bn_layers = [m for m in net.actor_mlp.modules() if isinstance(m, torch.nn.BatchNorm1d)]
    assert len(bn_layers) == 2


def test_norm_only_first_layer_keeps_following_linear_shapes_valid():
    net = _build_actor_critic_net(
        {
            "name": "actor_critic",
            "separate": False,
            "space": {"discrete": {}},
            "normalization": "layer_norm",
            "mlp": {
                "units": [32, 16, 8],
                "activation": "relu",
                "norm_only_first_layer": True,
                "initializer": {"name": "default"},
            },
        }
    )

    ln_layers = [m for m in net.actor_mlp.modules() if isinstance(m, torch.nn.LayerNorm)]
    assert len(ln_layers) == 1

    x = torch.randn(5, 4)
    y = net.actor_mlp(x)
    assert y.shape == (5, 8)

