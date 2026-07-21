from __future__ import annotations

from types import MethodType, SimpleNamespace

import pytest
import torch
from e3nn import o3

from dptb.data import _keys
from dptb.data.interfaces.blockwise_tensor import strict_reverse_edge_index
from dptb.data.transforms import OrbitalMapper
from dptb.nn.embedding.lem_moe_v3_h0 import LemMoEV3H0
from dptb.nn.embedding.lem_moe_v3 import UpdateNode


class _SphericalHarmonics(torch.nn.Module):
    def forward(self, vectors: torch.Tensor) -> torch.Tensor:
        return vectors.new_zeros((vectors.shape[0], 1))


class _NodeOneHot(torch.nn.Module):
    def forward(self, data):
        count = data[_keys.ATOM_TYPE_KEY].numel()
        data[_keys.NODE_ATTRS_KEY] = torch.ones(
            (count, 1), device=data[_keys.POSITIONS_KEY].device
        )
        return data


class _EdgeOneHot(torch.nn.Module):
    def forward(self, data):
        count = data[_keys.EDGE_INDEX_KEY].shape[1]
        return torch.ones((count, 1), device=data[_keys.POSITIONS_KEY].device)


class _Router(torch.nn.Module):
    def forward(self, global_features: torch.Tensor):
        count = global_features.shape[0]
        coefficients = global_features.new_ones((count, 1))
        return coefficients, global_features.new_ones(()), global_features.new_zeros(())

    def last_topk(self):
        return torch.zeros((1, 1), dtype=torch.long), torch.ones((1, 1))


class _ActiveSubsetInit(torch.nn.Module):
    """Small deterministic stand-in for H0InitLayer's active-row contract."""

    def forward(
        self,
        data,
        edge_index,
        atom_type,
        bond_type,
        edge_sh,
        edge_length,
        edge_one_hot,
        active_edges=None,
        cutoff_coeffs=None,
    ):
        if active_edges is None:
            active_edges = torch.tensor([0, 1], device=edge_length.device)
        else:
            active_edges = active_edges.to(device=edge_length.device, dtype=torch.long)
        if cutoff_coeffs is None:
            cutoff_coeffs = edge_length.new_ones(edge_length.shape[0])
        latents = edge_length.new_zeros((edge_length.shape[0], 1))
        node_features = edge_length.new_zeros((atom_type.numel(), 1))
        edge_features = (
            torch.arange(active_edges.numel(), device=edge_length.device, dtype=edge_length.dtype)
            .add_(1.0)
            .unsqueeze(-1)
        )
        return latents, node_features, edge_features, cutoff_coeffs, active_edges


def _fake_block_heads(self, node_features, edge_features, atom_type, edge_index, active_edges):
    node_blocks = (
        torch.arange(node_features.shape[0], device=node_features.device, dtype=node_features.dtype)
        .add_(11.0)
        .reshape(-1, 1, 1)
    )
    edge_blocks = edge_features.add(20.0).reshape(-1, 1, 1)
    return node_blocks, edge_blocks


def _minimal_hb0_h0_model(idp: OrbitalMapper) -> LemMoEV3H0:
    """Exercise LemMoEV3H0.forward without constructing the expensive MoE stack."""

    model = LemMoEV3H0.__new__(LemMoEV3H0)
    torch.nn.Module.__init__(model)
    model.use_h0_init = True
    model.dtype = torch.float32
    model.device = torch.device("cpu")
    model.idp = idp
    model.sh = _SphericalHarmonics()
    model.onehot = _NodeOneHot()
    model.edge_one_hot = _EdgeOneHot()
    model.router = _Router()
    model.init_layer = _ActiveSubsetInit()
    model.flow_time_conditioner = None
    model.layers = torch.nn.ModuleList()
    model.use_block_native_output = True
    model.require_full_block_edge_coverage = False
    model.out_edge = SimpleNamespace(max_norb=1)
    model._apply_block_native_output_heads = MethodType(_fake_block_heads, model)
    return model


def _minimal_hb0_case():
    idp = OrbitalMapper({"H": ["1s"]}, method="e3tb", device="cpu")
    idp.get_orbital_maps()
    idp.get_irreps(no_parity=False)
    return idp, _minimal_hb0_h0_model(idp)


def _active_edge_data(idp, active_edges, cutoff_coeffs):
    return {
        _keys.POSITIONS_KEY: torch.tensor(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=torch.float32
        ),
        _keys.EDGE_INDEX_KEY: torch.tensor(
            [[0, 1, 0, 1], [1, 0, 1, 0]], dtype=torch.long
        ),
        _keys.CELL_KEY: torch.eye(3, dtype=torch.float32).unsqueeze(0) * 5.0,
        _keys.PBC_KEY: torch.tensor([True, False, False]),
        _keys.BATCH_KEY: torch.zeros(2, dtype=torch.long),
        _keys.EDGE_CELL_SHIFT_KEY: torch.tensor(
            [[0, 0, 0], [0, 0, 0], [1, 0, 0], [-1, 0, 0]], dtype=torch.long
        ),
        _keys.ATOM_TYPE_KEY: torch.zeros((2, 1), dtype=torch.long),
        _keys.EDGE_TYPE_KEY: torch.zeros(4, dtype=torch.long),
        _keys.NODE_H0_KEY: torch.zeros((2, idp.reduced_matrix_element)),
        _keys.EDGE_H0_KEY: torch.zeros((4, idp.reduced_matrix_element)),
        _keys.LEM_ACTIVE_EDGES_KEY: active_edges,
        _keys.LEM_CUTOFF_COEFFS_KEY: cutoff_coeffs,
    }


def _two_graph_data(idp, *, split_sizes=None):
    data = _active_edge_data(idp, torch.arange(4), torch.ones(4))
    data.update(
        {
            _keys.POSITIONS_KEY: torch.zeros((4, 3)),
            _keys.EDGE_INDEX_KEY: torch.tensor(
                [[0, 1, 2, 3], [1, 0, 3, 2]], dtype=torch.long
            ),
            _keys.CELL_KEY: torch.eye(3).repeat(2, 1, 1),
            _keys.PBC_KEY: torch.zeros((2, 3), dtype=torch.bool),
            _keys.EDGE_CELL_SHIFT_KEY: torch.zeros((4, 3), dtype=torch.long),
            _keys.BATCH_KEY: torch.tensor([0, 0, 1, 1]),
            _keys.ATOM_TYPE_KEY: torch.zeros((4, 1), dtype=torch.long),
            _keys.NODE_H0_KEY: torch.zeros((4, idp.reduced_matrix_element)),
        }
    )
    if split_sizes is not None:
        data[_keys.LEM_ACTIVE_EDGE_SPLIT_SIZES_KEY] = torch.as_tensor(split_sizes)
    return data


def test_legacy_hb0_subset_scatter_is_unchanged_when_full_coverage_is_not_required():
    idp, model = _minimal_hb0_case()

    active_edges = torch.tensor([0, 1], dtype=torch.long)
    data = _active_edge_data(idp, active_edges, torch.ones(4))
    data[_keys.LEM_ACTIVE_EDGE_SPLIT_SIZES_KEY] = torch.tensor([2])

    # Two distinct periodic images, each represented by a complete reverse
    # pair.  This graph is valid under the strict keying used by the codec.
    assert strict_reverse_edge_index(data, idp=idp).tolist() == [1, 0, 3, 2]

    output = model(data)

    # Legacy ao_block tensors retain the exact zero-canvas/index_copy result.
    torch.testing.assert_close(
        output[_keys.NODE_HAMILTONIAN_KEY],
        torch.tensor([[[11.0]], [[12.0]]]),
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        output[_keys.EDGE_HAMILTONIAN_KEY],
        torch.tensor([[[21.0]], [[22.0]], [[0.0]], [[0.0]]]),
        rtol=0.0,
        atol=0.0,
    )

    # Reusable input-side acceleration metadata still follows its historical
    # cleanup contract; no output-side coverage assertion is needed.
    assert _keys.LEM_ACTIVE_EDGES_KEY not in output
    assert _keys.LEM_CUTOFF_COEFFS_KEY not in output
    assert _keys.LEM_ACTIVE_EDGE_SPLIT_SIZES_KEY not in output


def test_block_ode_coverage_guard_accepts_only_the_actual_ordered_full_head_rows():
    idp, model = _minimal_hb0_case()
    model.require_full_block_edge_coverage = True

    assert model.supports_full_block_edge_coverage is True
    output = model(_active_edge_data(idp, torch.arange(4), torch.ones(4)))
    torch.testing.assert_close(
        output[_keys.EDGE_HAMILTONIAN_KEY],
        torch.tensor([[[21.0]], [[22.0]], [[23.0]], [[24.0]]]),
        rtol=0.0,
        atol=0.0,
    )


@pytest.mark.parametrize(
    ("active_edges", "cutoff_coeffs", "match"),
    (
        (torch.tensor([0, 1]), torch.ones(4), "ordered full H-B0"),
        (torch.tensor([1, 0, 2, 3]), torch.ones(4), "ordered full H-B0"),
        (torch.tensor([0.9, 1.2, 2.0, 3.0]), torch.ones(4), "integral active-edge"),
        (torch.arange(4), torch.tensor([1.0, 0.0, 1.0, 1.0]), "strictly positive"),
        (torch.arange(4), torch.tensor([1.0, float("nan"), 1.0, 1.0]), "strictly positive"),
        (torch.arange(4), torch.ones(3), "one finite"),
    ),
)
def test_block_ode_coverage_guard_rejects_uncomputed_or_ambiguous_head_rows(
    active_edges, cutoff_coeffs, match
):
    idp, model = _minimal_hb0_case()
    model.require_full_block_edge_coverage = True

    with pytest.raises(ValueError, match=match):
        model(_active_edge_data(idp, active_edges, cutoff_coeffs))


def test_block_ode_coverage_guard_rejects_stale_graph_split_sizes():
    idp, model = _minimal_hb0_case()
    model.require_full_block_edge_coverage = True
    data = _two_graph_data(idp, split_sizes=[1, 3])

    with pytest.raises(ValueError, match="split sizes must exactly match"):
        model(data)


def test_block_ode_rechecks_cross_graph_edges_after_a_valid_forward():
    idp, model = _minimal_hb0_case()
    model.require_full_block_edge_coverage = True
    model(_two_graph_data(idp, split_sizes=[2, 2]))

    malformed = _two_graph_data(idp, split_sizes=[2, 2])
    malformed[_keys.EDGE_INDEX_KEY][1, 1] = 2
    with pytest.raises(ValueError, match="stay within one batch graph"):
        model(malformed)


def test_update_node_keeps_rows_for_nodes_without_active_incident_edges():
    class _TensorProduct(torch.nn.Module):
        def forward(self, x, r, mole_globals, latents=None, wigner_D_all=None):
            return torch.ones((x.shape[0], 1), dtype=x.dtype), wigner_D_all

    class _Update:
        irreps_in = o3.Irreps("1x0e")
        irreps_out = o3.Irreps("1x0e")
        edge_irreps_in = o3.Irreps("1x0e")
        tp = _TensorProduct()
        activation = torch.nn.Identity()
        lin_post = torch.nn.Identity()
        focus_gate = torch.nn.Identity()
        post_activation_expert_mixer = None
        node_norm = None
        edge_norm = None
        node_attention = None
        edge_aggregation_gate = None
        env_sum_normalizations = torch.tensor(1.0)
        res_update = False
        use_layer_onehot_tp = False
        edge_message_env_weight = False

    output = UpdateNode.forward(
        _Update(),
        latents=torch.zeros((2, 1)),
        node_features=torch.zeros((3, 1)),
        edge_features=torch.zeros((2, 1)),
        atom_type=torch.zeros((3, 1), dtype=torch.long),
        node_onehot=torch.zeros((3, 1)),
        edge_index=torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
        edge_vector=torch.zeros((2, 3)),
        cutoff_coeffs=torch.ones(2),
        active_edges=torch.tensor([0, 1], dtype=torch.long),
        wigner_D_all=None,
        mole_globals=None,
    )

    assert output.shape == (3, 1)
    torch.testing.assert_close(output[:2], torch.ones((2, 1)))
    torch.testing.assert_close(output[2], torch.zeros(1))
