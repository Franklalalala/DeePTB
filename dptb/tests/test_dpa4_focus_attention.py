import torch
from e3nn import o3

from dptb.nn.embedding.lem_moe_v3 import (
    GatedEdgeAggregation,
    PostActivation0eFocusGate,
    SingleHead0eEnvelopeAttention,
)
from dptb.nn.embedding.lem_non_linear import NonLinearExpertUpdateNode
from dptb.plugins.monitor import GatedEdgeAggregationMonitor


def test_post_activation_0e_focus_gate_initializes_as_identity():
    irreps = o3.Irreps("4x0e+4x1o+4x2e")
    gate = PostActivation0eFocusGate(irreps, num_focus=2)

    message = torch.randn(5, irreps.dim)
    gated = gate(message)

    assert gated.shape == message.shape
    assert gate.focus_index.shape == (irreps.dim,)
    assert torch.allclose(gated, message)


def test_single_head_0e_attention_normalizes_per_destination_with_envelope():
    torch.manual_seed(1)
    attention = SingleHead0eEnvelopeAttention(
        node_scalar_dim=3,
        message_scalar_dim=3,
        latent_dim=4,
        attn_dim=5,
    )

    dst_index = torch.tensor([0, 0, 1, 1], dtype=torch.long)
    node_scalars = torch.randn(3, 3)
    message_scalars = torch.randn(4, 3)
    latents = torch.randn(4, 4)
    cutoff = torch.tensor([1.0, 0.5, 1.0, 0.0])
    message = torch.randn(4, 7)

    weights = attention.attention_weights(
        dst_index=dst_index,
        node_scalars=node_scalars,
        message_scalars=message_scalars,
        latents=latents,
        cutoff_coeffs=cutoff,
        dim_size=3,
    )
    per_node = torch.zeros(3).scatter_add(0, dst_index, weights)

    assert weights.shape == (4,)
    assert torch.isfinite(weights).all()
    assert weights[-1].item() == 0.0
    assert torch.all(per_node[:2] > 0)
    assert torch.all(per_node <= 1.0 + 1e-6)

    out = attention(
        message=message,
        dst_index=dst_index,
        node_scalars=node_scalars,
        message_scalars=message_scalars,
        latents=latents,
        cutoff_coeffs=cutoff,
        dim_size=3,
    )

    assert out.shape == (3, 7)
    assert torch.isfinite(out).all()
    assert torch.allclose(out[2], torch.zeros_like(out[2]))


def test_gated_edge_aggregation_applies_equivariant_sigmoid_gate_and_records_stats():
    irreps = o3.Irreps("2x0e+1x1o")
    gate = GatedEdgeAggregation(irreps, query_scalar_dim=2)

    aggregated = torch.ones(3, irreps.dim)
    query_scalars = torch.zeros(3, 2)
    dst_index = torch.tensor([0, 0, 1, 1], dtype=torch.long)
    edge_message = torch.tensor(
        [
            [1.0, 1.0, 0.0, 0.0, 0.0],
            [3.0, 3.0, 0.0, 0.0, 0.0],
            [0.5, 0.5, 0.0, 0.0, 0.0],
            [1.5, 1.5, 0.0, 0.0, 0.0],
        ]
    )

    out = gate(
        aggregated,
        query_scalars,
        dst_index=dst_index,
        edge_message=edge_message,
        dim_size=aggregated.shape[0],
    )

    assert gate.feature_gate_index.tolist() == [0, 1, 2, 2, 2]
    assert torch.allclose(out, 0.5 * aggregated)
    stats = gate.last_stats
    assert stats["active_edges"] == 4
    assert stats["nodes_with_edges"] == 2
    assert stats["gate_mean"] == 0.5
    assert stats["gate_min"] == 0.5
    assert stats["gate_max"] == 0.5
    assert stats["gate_sparsity_lt_0_1"] == 0.0
    assert stats["pre_sparsity_lt_1e_2"] == 0.0
    assert stats["post_sparsity_lt_1e_2"] == 0.0
    assert stats["pre_activation_max"] == 1.0
    assert stats["post_activation_max"] == 0.5
    assert torch.isclose(torch.tensor(stats["top_edge_share_mean"]), torch.tensor(0.75))
    assert torch.isclose(torch.tensor(stats["top_edge_share_max"]), torch.tensor(0.75))


def test_gated_edge_aggregation_monitor_writes_latest_layer_stats(tmp_path):
    class TinyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.edge_gate = GatedEdgeAggregation(o3.Irreps("1x0e"), query_scalar_dim=1)

    class TinyTrainer:
        def __init__(self):
            self.model = TinyModel()
            self.rank = 0
            self.is_main_process = True

    trainer = TinyTrainer()
    _ = trainer.model.edge_gate(
        torch.ones(1, 1),
        torch.zeros(1, 1),
        dst_index=torch.tensor([0], dtype=torch.long),
        edge_message=torch.ones(1, 1),
        dim_size=1,
    )

    monitor = GatedEdgeAggregationMonitor(
        str(tmp_path),
        interval=[(1, "iteration")],
        tensorboard=False,
    )
    monitor.register(trainer)
    monitor.iteration(time=7)

    csv_text = (tmp_path / "gated_edge_aggregation.csv").read_text()
    assert "iter,rank,module,gate_mean" in csv_text
    assert "7,0,edge_gate,0.5" in csv_text


def test_non_linear_node_wrapper_applies_base_gated_edge_aggregation():
    class DummyTP(torch.nn.Module):
        def forward(self, x, r, mole_globals, latents=None, wigner_D_all=None):
            return x[:, :1], wigner_D_all

    class DummyActivation(torch.nn.Module):
        irreps_out = o3.Irreps("1x0e")

        def forward(self, x):
            return x

    class DummyBase(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.irreps_in = o3.Irreps("1x0e")
            self.irreps_out = o3.Irreps("1x0e")
            self.tp = DummyTP()
            self.activation = DummyActivation()
            self.lin_post = torch.nn.Identity()
            self.focus_gate = torch.nn.Identity()
            self.node_norm = None
            self.edge_norm = None
            self.node_attention = None
            self.edge_aggregation_gate = GatedEdgeAggregation(self.irreps_out, query_scalar_dim=1)
            self.env_sum_normalizations = torch.tensor(1.0)
            self.res_update = False
            self.use_layer_onehot_tp = False
            self.env_embed_mlps = lambda latents: latents
            self._env_weighter = lambda message, weights: message

    wrapper = NonLinearExpertUpdateNode(DummyBase(), num_experts=2)
    wrapper._run_expert_block_from_parts = lambda *args, **kwargs: (
        torch.ones(2, 1),
        None,
    )

    node_features = torch.zeros(2, 1)
    out = wrapper(
        latents=torch.zeros(2, 1),
        node_features=node_features,
        edge_features=torch.zeros(2, 1),
        atom_type=torch.zeros(2, 1, dtype=torch.long),
        node_onehot=torch.zeros(2, 1),
        edge_index=torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
        edge_vector=torch.zeros(2, 3),
        cutoff_coeffs=torch.ones(2),
        active_edges=torch.tensor([0, 1], dtype=torch.long),
        wigner_D_all=None,
        mole_globals=None,
    )

    assert torch.allclose(out, torch.full_like(out, 0.5))
    assert wrapper.base.edge_aggregation_gate.last_stats["active_edges"] == 2

