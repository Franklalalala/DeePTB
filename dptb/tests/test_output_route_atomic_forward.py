from __future__ import annotations

from pathlib import Path

import pytest
import torch

from dptb.data import _keys
from dptb.nn.build import build_model
from dptb.nn.embedding.cartesian_ict_bank import (
    export_cartesian_ict_projector_bank,
)
from dptb.nn.embedding.cartesian_projector import ao_shell_layout
from dptb.nn.embedding.output_routes import get_output_route_spec


BASIS = {"H": "1s", "O": "1s1p"}
HIDDEN = "4x0e+4x1o+4x1e+4x2e"
ROUTES = ("h_a0", "h_a1", "h_b0", "h_b1", "p_b0", "p_b1_ict")


def _embedding_options(route: str, tmp_path: Path) -> dict:
    options = {
        "method": "lem_moe_v3",
        "output_route": route,
        "n_layers": 1,
        "avg_num_neighbors": 2.0,
        "r_max": 4.0,
        "irreps_hidden": HIDDEN,
        "env_embed_multiplicity": 4,
        "latent_dim": 8,
        "latent_channels": [8],
        "edge_one_hot_dim": 4,
        "num_experts": 1,
        "num_shared_experts": 1,
        "top_k": 1,
        "universal": True,
        "use_layer_onehot_tp": False,
        "use_out_onehot_tp": False,
        "use_interpolation_out": False,
        "tp_radial_emb": False,
        "mole_linear_mode": "indexed_ref",
        "so2_fusion_mode": "streamed_m_major_ref",
        "rme_fusion_rank": 4,
        "rme_fusion_init": 0.0,
    }
    if route == "p_b0":
        options["ao_projector_backend"] = "reference_wigner"
    elif route == "p_b1_ict":
        bank = export_cartesian_ict_projector_bank(
            tmp_path / "sp_ict_projectors.json", ("1s", "1p")
        )
        options.update(
            {
                "ao_projector_backend": "precomputed",
                "ao_projector_bank_path": str(bank),
            }
        )
    return options


def _build(route: str, tmp_path: Path):
    spec = get_output_route_spec(route)
    prediction = {
        "method": spec.prediction_method,
        "scale_type": "no_scale",
    }
    if spec.block_decoder is not None:
        prediction.update(
            {
                "block_decoder": spec.block_decoder,
                "blockwise_hamiltonian": True,
            }
        )
    return build_model(
        common_options={
            "basis": BASIS,
            "overlap": False,
            "dtype": "float32",
            "device": "cpu",
        },
        model_options={
            "embedding": _embedding_options(route, tmp_path),
            "prediction": prediction,
        },
        train_options={},
        no_check=False,
    )


def _data(model):
    h = model.idp.chemical_symbol_to_type["H"]
    o = model.idp.chemical_symbol_to_type["O"]
    edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    edge_type = torch.tensor(
        [
            model.idp.bond_to_type["H-O"],
            model.idp.bond_to_type["O-H"],
        ],
        dtype=torch.long,
    )
    return {
        _keys.POSITIONS_KEY: torch.tensor(
            [[0.0, 0.0, 0.0], [1.1, 0.2, -0.1]],
            dtype=torch.float32,
            requires_grad=True,
        ),
        _keys.EDGE_INDEX_KEY: edge_index,
        _keys.ATOM_TYPE_KEY: torch.tensor([[h], [o]], dtype=torch.long),
        _keys.EDGE_TYPE_KEY: edge_type,
    }


@pytest.mark.parametrize("route", ROUTES)
def test_real_atomicdata_forward_backward_and_e3_routing(route, tmp_path):
    model = _build(route, tmp_path)
    spec = model.embedding.output_route_spec
    assert spec.canonical_name == route

    e3_calls = 0
    if spec.uses_e3hamiltonian:
        original_forward = model.hamiltonian.forward

        def counted_forward(data):
            nonlocal e3_calls
            e3_calls += 1
            return original_forward(data)

        model.hamiltonian.forward = counted_forward
    else:
        assert not hasattr(model, "hamiltonian")

    data = _data(model)
    output = model(data)
    assert e3_calls == int(spec.uses_e3hamiltonian)

    if spec.output_contract == "rme":
        assert _keys.NODE_FEATURES_KEY in output
        assert _keys.EDGE_FEATURES_KEY in output
        assert _keys.NODE_HAMILTONIAN_KEY not in output
        loss = (
            output[_keys.NODE_FEATURES_KEY].square().mean()
            + output[_keys.EDGE_FEATURES_KEY].square().mean()
        )
    else:
        assert _keys.NODE_HAMILTONIAN_KEY in output
        assert _keys.EDGE_HAMILTONIAN_KEY in output
        assert output[_keys.NODE_HAMILTONIAN_KEY].shape[-2:] == (
            model.idp.full_basis_norb,
            model.idp.full_basis_norb,
        )
        assert output[_keys.EDGE_HAMILTONIAN_KEY].shape[-2:] == (
            model.idp.full_basis_norb,
            model.idp.full_basis_norb,
        )
        loss = (
            output[_keys.NODE_HAMILTONIAN_KEY].square().mean()
            + output[_keys.EDGE_HAMILTONIAN_KEY].square().mean()
        )

    loss.backward()
    assert data[_keys.POSITIONS_KEY].grad is not None
    assert torch.isfinite(data[_keys.POSITIONS_KEY].grad).all()
    assert any(
        parameter.grad is not None
        for parameter in model.parameters()
        if parameter.requires_grad
    )


def test_block_routes_respect_shell_slices_and_species_masks(tmp_path):
    for route in ("h_b0", "h_b1", "p_b0", "p_b1_ict"):
        model = _build(route, tmp_path)
        output = model(_data(model))
        node = output[_keys.NODE_HAMILTONIAN_KEY]
        edge = output[_keys.EDGE_HAMILTONIAN_KEY]
        layout = ao_shell_layout(model.idp.full_basis)
        for row_start, row_stop, row_l in layout:
            for col_start, col_stop, col_l in layout:
                assert node[:, row_start:row_stop, col_start:col_stop].shape[-2:] == (
                    2 * row_l + 1,
                    2 * col_l + 1,
                )

        atom_type = output[_keys.ATOM_TYPE_KEY].flatten()
        basis_mask = model.idp.mask_to_basis[atom_type]
        expected_node = basis_mask.unsqueeze(-1) & basis_mask.unsqueeze(-2)
        assert torch.count_nonzero(node.masked_select(~expected_node)) == 0

        edge_index = output[_keys.EDGE_INDEX_KEY]
        expected_edge = (
            basis_mask[edge_index[0]].unsqueeze(-1)
            & basis_mask[edge_index[1]].unsqueeze(-2)
        )
        assert torch.count_nonzero(edge.masked_select(~expected_edge)) == 0


def test_route_specific_runtime_semantics_on_real_models(tmp_path):
    h_a1 = _build("h_a1", tmp_path).embedding.out_node
    h_b1 = _build("h_b1", tmp_path).embedding.out_node
    p_b1 = _build("p_b1_ict", tmp_path).embedding.out_node

    assert h_a1.coverage_report["product_paths"] > 0
    assert h_b1.coverage_report["product_paths"] == 0
    assert not hasattr(h_b1, "left") and not hasattr(h_b1, "right")
    assert p_b1.uses_ict is True
    assert p_b1.uses_precomputed_projector is True
    assert p_b1.projector_source == "cartesian_ict"
