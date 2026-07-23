from __future__ import annotations

import torch
from e3nn import o3

from dptb.nn.embedding.lem_pair import PairUpdateEdge, PairUpdateNode

from test_lem_pair_common import fp64_default, model


class _ReturnUnits(torch.nn.Module):
    def forward(self, x, *args, **kwargs):
        if len(args) >= 3:
            return x.new_ones((x.shape[0], 1)), args[3] if len(args) > 3 else None
        return x.new_ones((x.shape[0], 1))


class _IgnoreWeights(torch.nn.Module):
    def forward(self, features, _weights):
        return features


class _AdditiveNode(PairUpdateNode):
    def __init__(self):
        pass

    irreps_in = o3.Irreps("1x0e")
    irreps_out = o3.Irreps("1x0e")
    edge_irreps_in = o3.Irreps("1x0e")
    tp = _ReturnUnits()
    activation = torch.nn.Identity()
    lin_post = torch.nn.Identity()
    focus_gate = torch.nn.Identity()
    post_activation_expert_mixer = None
    node_norm = None
    edge_norm = None
    node_attention = None
    edge_aggregation_gate = None
    env_sum_normalizations = torch.tensor(1.0)
    res_update = True
    res_update_additive = True
    use_identity_res = True
    use_layer_onehot_tp = False
    edge_message_env_weight = False


class _AdditiveEdge(PairUpdateEdge):
    def __init__(self):
        pass

    irreps_in = o3.Irreps("1x0e")
    irreps_out = o3.Irreps("1x0e")
    tp = _ReturnUnits()
    activation = torch.nn.Identity()
    lin_post = torch.nn.Identity()
    post_activation_expert_mixer = None
    node_norm = None
    edge_norm = None
    edge_embed_mlps = _ReturnUnits()
    _edge_weighter = _IgnoreWeights()
    ln = torch.nn.Identity()
    latents_mlp_1 = _ReturnUnits()
    latents_mlp_2 = _ReturnUnits()
    res_update = True
    res_update_additive = True
    use_identity_res = True
    use_layer_onehot_tp = False


def test_additive_switch_is_exact_unscaled_node_edge_and_latent_addition():
    edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    active = torch.arange(2)
    old = torch.full((2, 1), 2.0)
    expected = torch.full((2, 1), 3.0)
    node = PairUpdateNode.forward(
        _AdditiveNode(),
        torch.zeros((2, 1)),
        old.clone(),
        torch.zeros((2, 1)),
        torch.zeros((2, 1), dtype=torch.long),
        torch.zeros((2, 1)),
        edge_index,
        torch.zeros((2, 3)),
        torch.ones(2),
        active,
        None,
        None,
    )
    edge, latent, _ = PairUpdateEdge.forward(
        _AdditiveEdge(),
        old.clone(),
        torch.zeros((2, 1)),
        torch.zeros((2, 1)),
        old.clone(),
        edge_index,
        torch.zeros((2, 3)),
        torch.ones(2),
        active,
        torch.zeros((2, 1)),
        None,
        None,
    )
    assert torch.equal(node, expected)
    assert torch.equal(edge, expected)
    assert torch.equal(latent, expected)


def test_latent_layernorm_disable_is_identity_in_every_pair_layer():
    with fp64_default():
        pair_model = model(
            res_update_additive=True,
            latents_layernorm=False,
        )
    for layer in pair_model.layers:
        assert layer.node_update.res_update_additive is True
        assert layer.edge_update.res_update_additive is True
        assert isinstance(layer.edge_update.ln, torch.nn.Identity)
        latents = torch.tensor([[1.0, 2.0, 4.0, 8.0]], dtype=torch.float64)
        assert torch.equal(layer.edge_update.ln(latents), latents)
