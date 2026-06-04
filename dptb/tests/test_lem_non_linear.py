import torch
from e3nn import o3

from dptb.data import _keys
from dptb.nn.embedding.lem_non_linear import (
    LemNonLinear,
    PostActivationExpertMixer,
)


def test_post_activation_expert_mixer_uses_scalar_softmax():
    irreps = o3.Irreps("3x0e+2x1o")
    mixer = PostActivationExpertMixer(irreps, num_experts=2)

    expert_outputs = torch.randn(4, 2, irreps.dim)
    mixed, weights = mixer(expert_outputs)

    assert mixed.shape == (4, irreps.dim)
    assert weights.shape == (4, 2)
    assert torch.allclose(weights.sum(dim=1), torch.ones(4))
    assert torch.isfinite(mixed).all()


def test_lem_non_linear_tiny_forward_backward_cpu():
    torch.manual_seed(0)
    model = LemNonLinear(
        basis={"H": "1s"},
        n_layers=1,
        n_radial_basis=4,
        r_max=2.5,
        irreps_hidden="2x0e+2x1o",
        avg_num_neighbors=1.0,
        env_embed_multiplicity=2,
        tp_radial_emb=False,
        tp_radial_channels=[8],
        latent_channels=[8],
        latent_dim=8,
        edge_one_hot_dim=4,
        use_out_onehot_tp=False,
        use_layer_onehot_tp=False,
        res_update=True,
        equivariant_norm_type="none",
        node_message_aggregation="single_head_0e",
        num_focus=1,
        use_interpolation_out=False,
        num_experts=2,
        dtype=torch.float32,
        device=torch.device("cpu"),
    )

    data = {
        _keys.POSITIONS_KEY: torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.75]]),
        _keys.EDGE_INDEX_KEY: torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
        _keys.EDGE_CELL_SHIFT_KEY: torch.zeros((2, 3)),
        _keys.CELL_KEY: torch.eye(3),
        _keys.PBC_KEY: torch.zeros(3, dtype=torch.bool),
        _keys.ATOM_TYPE_KEY: torch.zeros((2, 1), dtype=torch.long),
        _keys.EDGE_TYPE_KEY: torch.zeros((2, 1), dtype=torch.long),
    }

    out = model(data)
    node = out[_keys.NODE_FEATURES_KEY]
    edge = out[_keys.EDGE_FEATURES_KEY]
    loss = node.square().mean() + edge.square().mean()
    loss.backward()

    assert tuple(node.shape) == (2, 1)
    assert tuple(edge.shape) == (2, 1)
    assert torch.isfinite(loss)
    assert len(model.layers[0].node_update.expert_tp.experts) == 2
    assert len(model.layers[0].edge_update.expert_tp.experts) == 2
