from __future__ import annotations

import torch

from dptb.data.transforms import OrbitalMapper
from dptb.nn.embedding.lem_moe_v3_h0_helpers import H0InitLayer


class _DummyInit(torch.nn.Module):
    def __init__(self, idp: OrbitalMapper) -> None:
        super().__init__()
        self.idp = idp
        self.irreps_out = idp.get_irreps().sort()[0].simplify()


def _h0_layer(
    basis,
    *,
    has_soc: bool = False,
    use_uureal_residual_block_input: bool = False,
) -> H0InitLayer:
    mapper = OrbitalMapper(
        basis,
        method="e3tb",
        has_soc=has_soc,
        nextham_uureal_mask=has_soc,
        full_soc_prediction=False,
    )
    return H0InitLayer(
        _DummyInit(mapper),
        use_uureal_residual_block_input=use_uureal_residual_block_input,
        dtype=torch.float64,
        device="cpu",
    ).to(dtype=torch.float64)


def test_uureal_projection_is_bit_exact_with_legacy_sort_path():
    """G-FIX3: the already-correct uu_real projector boundary is unchanged."""
    torch.manual_seed(20260724)
    layer = _h0_layer(
        {"H": "1s", "C": "1s1p"},
        has_soc=True,
        use_uureal_residual_block_input=True,
    )
    mapper = layer.idp

    atom_type = torch.tensor(
        [
            [mapper.chemical_symbol_to_type["H"]],
            [mapper.chemical_symbol_to_type["C"]],
        ]
    )
    bond_type = torch.tensor(
        [
            [mapper.bond_to_type["H-C"]],
            [mapper.bond_to_type["C-H"]],
        ]
    )
    raw_node = torch.randn(2, layer.h0_dim, dtype=torch.float64)
    raw_edge = torch.randn(2, layer.h0_dim, dtype=torch.float64)
    masked_node = layer._mask_node_source(raw_node, atom_type)
    masked_edge = layer._mask_edge_source(raw_edge, bond_type)

    legacy_node = masked_node.index_select(1, layer._uureal_h0_sort_index)
    fixed_node = masked_node.index_select(1, layer._h0_sort_index)
    legacy_edge = masked_edge.index_select(1, layer._uureal_h0_sort_index)
    fixed_edge = masked_edge.index_select(1, layer._h0_sort_index)

    assert torch.equal(fixed_node, legacy_node)
    assert torch.equal(fixed_edge, legacy_edge)
    assert torch.equal(
        layer.node_projector(fixed_node),
        layer.node_projector(legacy_node),
    )
    assert torch.equal(
        layer.edge_projector(fixed_edge),
        layer.edge_projector(legacy_edge),
    )
    state = layer.state_dict()
    assert "_uureal_h0_sort_index" in state
    assert "_h0_sort_index" not in state


def test_zero_and_scalar_only_h0_are_bit_exact_under_sort():
    """G-FIX4: zero vectors and pure-l=0 layouts cannot change under sorting."""
    torch.manual_seed(20260724)
    highl_layer = _h0_layer({"H": "1s", "C": "1s1p"})
    zeros = torch.zeros(3, highl_layer.h0_dim, dtype=torch.float64)
    sorted_zeros = zeros.index_select(1, highl_layer._h0_sort_index)
    assert torch.equal(sorted_zeros, zeros)
    assert torch.equal(
        highl_layer.node_projector(sorted_zeros),
        highl_layer.node_projector(zeros),
    )

    scalar_layer = _h0_layer({"H": "1s"})
    identity = torch.arange(scalar_layer.h0_dim)
    assert torch.equal(scalar_layer._h0_sort_index.cpu(), identity)
    scalar_h0 = torch.randn(3, scalar_layer.h0_dim, dtype=torch.float64)
    sorted_scalar_h0 = scalar_h0.index_select(
        1, scalar_layer._h0_sort_index
    )
    assert torch.equal(sorted_scalar_h0, scalar_h0)
    assert torch.equal(
        scalar_layer.edge_projector(sorted_scalar_h0),
        scalar_layer.edge_projector(scalar_h0),
    )
