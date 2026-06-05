import torch
from e3nn import o3

from dptb.nn.embedding.lem_moe_v3 import (
    GatedEdgeAggregation,
    PostActivation0eFocusGate,
    SingleHead0eEnvelopeAttention,
    UpdateNode,
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


def test_single_head_0e_attention_can_square_envelope_without_latent_bias():
    attention = SingleHead0eEnvelopeAttention(
        node_scalar_dim=1,
        message_scalar_dim=1,
        latent_dim=2,
        attn_dim=1,
        envelope_power=2.0,
        use_latent_bias=False,
    )
    with torch.no_grad():
        attention.query.weight.zero_()
        attention.key.weight.zero_()

    dst_index = torch.tensor([0, 0], dtype=torch.long)
    node_scalars = torch.zeros(1, 1)
    message_scalars = torch.zeros(2, 1)
    cutoff = torch.tensor([1.0, 0.5])

    weights_a = attention.attention_weights(
        dst_index=dst_index,
        node_scalars=node_scalars,
        message_scalars=message_scalars,
        latents=torch.zeros(2, 2),
        cutoff_coeffs=cutoff,
        dim_size=1,
    )
    weights_b = attention.attention_weights(
        dst_index=dst_index,
        node_scalars=node_scalars,
        message_scalars=message_scalars,
        latents=torch.full((2, 2), 1000.0),
        cutoff_coeffs=cutoff,
        dim_size=1,
    )

    assert torch.allclose(weights_a, torch.tensor([0.8, 0.2]), atol=1e-5)
    assert torch.allclose(weights_a, weights_b)


def test_single_head_0e_attention_key_layer_norm_is_key_only():
    attention = SingleHead0eEnvelopeAttention(
        node_scalar_dim=2,
        message_scalar_dim=2,
        latent_dim=2,
        attn_dim=2,
        use_latent_bias=False,
        key_layer_norm=True,
    )
    with torch.no_grad():
        attention.query.weight.copy_(torch.eye(2))
        attention.key.weight.copy_(torch.eye(2))

    weights = attention.attention_weights(
        dst_index=torch.tensor([0, 0], dtype=torch.long),
        node_scalars=torch.tensor([[1.0, 0.0]]),
        message_scalars=torch.tensor([[1.0, 3.0], [10.0, 12.0]]),
        latents=torch.zeros(2, 2),
        cutoff_coeffs=torch.ones(2),
        dim_size=1,
    )

    assert isinstance(attention.key_norm, torch.nn.LayerNorm)
    assert torch.allclose(weights, torch.tensor([0.5, 0.5]), atol=1e-5)


def test_update_node_can_bypass_env_message_weighting():
    class DummyTP(torch.nn.Module):
        def forward(self, x, r, mole_globals, latents=None, wigner_D_all=None):
            return torch.ones(x.shape[0], 1, dtype=x.dtype, device=x.device), wigner_D_all

    class DummyUpdate:
        def __init__(self):
            self.irreps_in = o3.Irreps("1x0e")
            self.irreps_out = o3.Irreps("1x0e")
            self.edge_irreps_in = o3.Irreps("1x0e")
            self.tp = DummyTP()
            self.activation = torch.nn.Identity()
            self.lin_post = torch.nn.Identity()
            self.focus_gate = torch.nn.Identity()
            self.post_activation_expert_mixer = None
            self.node_norm = None
            self.edge_norm = None
            self.node_attention = None
            self.edge_aggregation_gate = None
            self.env_sum_normalizations = torch.tensor(1.0)
            self.res_update = False
            self.use_layer_onehot_tp = False
            self.edge_message_env_weight = False
            self.env_embed_mlps = self._unexpected_env_weight
            self._env_weighter = self._unexpected_env_weight

        def _unexpected_env_weight(self, *args, **kwargs):
            raise AssertionError("env message weighting should be bypassed")

    dummy = DummyUpdate()
    out = UpdateNode.forward(
        dummy,
        latents=torch.zeros(2, 1),
        node_features=torch.zeros(2, 1),
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

    assert torch.allclose(out, torch.ones(2, 1))


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
        src_index=torch.tensor([0, 1, 0, 1], dtype=torch.long),
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
    assert gate.last_heatmap is not None
    assert gate.last_heatmap["query_nodes"].tolist() == [0, 1]
    assert gate.last_heatmap["key_nodes"].tolist() == [0, 1]
    assert torch.allclose(
        gate.last_heatmap["matrix"],
        torch.tensor([[0.25, 0.75], [0.25, 0.75]]),
    )
    assert torch.isclose(torch.tensor(gate.last_heatmap["top_key_score"]), torch.tensor(0.75))
    assert gate.last_heatmap["irrep_labels"] == ["0:2x0e", "1:1x1o"]


def test_gated_edge_aggregation_heatmap_selects_one_sample_from_batch():
    gate = GatedEdgeAggregation(o3.Irreps("1x0e"), query_scalar_dim=1)

    _ = gate(
        torch.ones(4, 1),
        torch.zeros(4, 1),
        dst_index=torch.tensor([0, 1, 2, 3], dtype=torch.long),
        src_index=torch.tensor([1, 0, 3, 2], dtype=torch.long),
        node_batch=torch.tensor([0, 0, 1, 1], dtype=torch.long),
        edge_message=torch.tensor([[1.0], [1.0], [10.0], [20.0]]),
        dim_size=4,
    )

    heatmap = gate.last_heatmap
    assert heatmap["sample_index"] == 1
    assert heatmap["query_nodes"].tolist() == [2, 3]
    assert heatmap["key_nodes"].tolist() == [2, 3]
    assert heatmap["query_node_local"].tolist() == [0, 1]
    assert torch.allclose(heatmap["matrix"], torch.tensor([[0.0, 1.0], [1.0, 0.0]]))


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
        src_index=torch.tensor([0], dtype=torch.long),
        edge_message=torch.ones(1, 1),
        dim_size=1,
    )

    monitor = GatedEdgeAggregationMonitor(
        str(tmp_path),
        interval=[(1, "iteration")],
        tensorboard=False,
        heatmap=True,
    )
    monitor.register(trainer)
    monitor.iteration(time=7)

    csv_text = (tmp_path / "gated_edge_aggregation.csv").read_text()
    assert "iter,rank,module,gate_mean" in csv_text
    assert "7,0,edge_gate,0.5" in csv_text
    heatmap_dir = tmp_path / "gated_edge_aggregation_heatmaps"
    assert (heatmap_dir / "gated_edge_aggregation_heatmap_iter0000007_rank0.npz").exists()
    assert (heatmap_dir / "gated_edge_aggregation_heatmap_iter0000007_rank0.png").exists()


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
            self.edge_message_env_weight = True
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

