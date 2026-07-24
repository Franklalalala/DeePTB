from __future__ import annotations

import pytest
import torch
from e3nn import o3

from dptb.data import _keys
from dptb.data.interfaces.blockwise_tensor import strict_reverse_edge_index
from dptb.nn.embedding.lem_moe_v3_h0 import LemMoEV3H0

from test_lem_pair_common import (
    ao_wigner,
    clone_data,
    fp64_default,
    model_options,
    molecule_data,
    rotate_data,
)


def _h0_model(*, seed: int = 20260724, **overrides) -> LemMoEV3H0:
    options = model_options()
    options.pop("mp_avg_num_neighbors")
    options.update(overrides)
    torch.manual_seed(seed)
    return LemMoEV3H0(**options).eval()


def _edge_hermiticity_drift(data, edge_blocks, model) -> float:
    reverse = strict_reverse_edge_index(data, device=edge_blocks.device, idp=model.idp)
    return float(
        (
            edge_blocks
            - edge_blocks.index_select(0, reverse).transpose(-1, -2)
        )
        .abs()
        .max()
    )


def test_hb0_hermitian_average_default_false_is_bit_exact_and_keeps_node_blocks():
    with fp64_default():
        torch.use_deterministic_algorithms(True)
        implicit = _h0_model()
        explicit = _h0_model(hb0_hermitian_average=False)

        assert implicit.state_dict().keys() == explicit.state_dict().keys()
        assert all(
            torch.equal(implicit.state_dict()[key], explicit.state_dict()[key])
            for key in implicit.state_dict()
        )
        implicit_out = implicit(molecule_data(implicit))
        explicit_out = explicit(molecule_data(explicit))
        assert torch.equal(
            implicit_out[_keys.NODE_HAMILTONIAN_KEY],
            explicit_out[_keys.NODE_HAMILTONIAN_KEY],
        )
        assert torch.equal(
            implicit_out[_keys.EDGE_HAMILTONIAN_KEY],
            explicit_out[_keys.EDGE_HAMILTONIAN_KEY],
        )


def test_hb0_hermitian_average_enforces_exact_reverse_transpose_and_preserves_nodes():
    with fp64_default():
        torch.use_deterministic_algorithms(True)
        raw = _h0_model()
        averaged = _h0_model(hb0_hermitian_average=True)
        averaged.load_state_dict(raw.state_dict())

        raw_data = molecule_data(raw)
        averaged_data = molecule_data(averaged)
        raw_out = raw(clone_data(raw_data))
        averaged_out = averaged(clone_data(averaged_data))

        assert torch.equal(
            raw_out[_keys.NODE_HAMILTONIAN_KEY],
            averaged_out[_keys.NODE_HAMILTONIAN_KEY],
        )
        assert _edge_hermiticity_drift(
            averaged_data,
            averaged_out[_keys.EDGE_HAMILTONIAN_KEY],
            averaged,
        ) == 0.0
        assert not torch.equal(
            raw_out[_keys.EDGE_HAMILTONIAN_KEY],
            averaged_out[_keys.EDGE_HAMILTONIAN_KEY],
        )


def test_hb0_hermitian_average_is_rotation_equivariant_fp64():
    with fp64_default():
        torch.use_deterministic_algorithms(True)
        model = _h0_model(hb0_hermitian_average=True)
        data = molecule_data(model)
        reference = model(clone_data(data))
        torch.manual_seed(31)
        rotation = o3.rand_matrix(dtype=torch.float64)
        rotated = model(rotate_data(data, rotation))

        d_ao = ao_wigner(model, rotation)
        expected_edge = torch.einsum(
            "ij,njk,lk->nil",
            d_ao,
            reference[_keys.EDGE_HAMILTONIAN_KEY],
            d_ao,
        )
        drift = float(
            (rotated[_keys.EDGE_HAMILTONIAN_KEY] - expected_edge).abs().max()
        )
        print(f"hb0_hermitian_average_rotation_max_abs={drift:.16e}")
        assert drift <= 1.0e-9


def test_hb0_hermitian_average_rejects_missing_reverse_active_edge():
    edge_blocks = torch.randn(2, 2, 2, dtype=torch.float64)
    edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
    active_edges = torch.arange(2, dtype=torch.long)
    with pytest.raises(ValueError, match="missing"):
        LemMoEV3H0._hermitian_average_hb0_edge_blocks(
            edge_blocks, edge_index, active_edges
        )
