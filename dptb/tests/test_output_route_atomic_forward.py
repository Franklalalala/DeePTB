from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from dptb.data import _keys
from dptb.data.interfaces.blockwise_tensor import (
    EDGE_DELTA_HAMIL_BLOCKS_KEY,
    EDGE_DELTA_HAMIL_BLOCK_SHAPE_KEY,
    EDGE_H0_BLOCKS_KEY,
    EDGE_PRED_HAMIL_BLOCKS_KEY,
    NODE_DELTA_HAMIL_BLOCKS_KEY,
    NODE_DELTA_HAMIL_BLOCK_SHAPE_KEY,
    NODE_H0_BLOCKS_KEY,
    NODE_PRED_HAMIL_BLOCKS_KEY,
    block_mask_from_shapes,
    infer_block_shapes,
)
from dptb.nn.build import build_model
from dptb.nn.embedding.cartesian_ict_bank import (
    export_cartesian_ict_projector_bank,
)
from dptb.nn.embedding.cartesian_projector import ao_shell_layout
from dptb.nn.embedding.output_routes import get_output_route_spec
from dptb.nnops.blockwise_nextham_loss import HamilBlockwiseNexTHamLoss
from dptb.nnops.flow import assert_flow_h0_keys_reach_model


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


def _build(
    route: str,
    tmp_path: Path,
    prediction_overrides: dict | None = None,
    embedding_overrides: dict | None = None,
):
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
    if prediction_overrides:
        prediction.update(prediction_overrides)
    embedding = _embedding_options(route, tmp_path)
    if embedding_overrides:
        embedding.update(embedding_overrides)
    return build_model(
        common_options={
            "basis": BASIS,
            "overlap": False,
            "dtype": "float32",
            "device": "cpu",
        },
        model_options={
            "embedding": embedding,
            "prediction": prediction,
        },
        train_options={},
        no_check=False,
    )


def test_real_hb0_block_ode_guards_model_contract_and_active_rows(tmp_path):
    model = _build(
        "h_b0",
        tmp_path,
        embedding_overrides={
            "method": "lem_moe_v3_h0",
            "use_h0_init": True,
            "use_flow_time_embedding": True,
            "flow_time_allow_missing": False,
            "require_full_block_edge_coverage": True,
        },
    )
    flow = SimpleNamespace(
        enabled=True,
        block_ode=True,
        node_h0_key=_keys.NODE_H0_KEY,
        edge_h0_key=_keys.EDGE_H0_KEY,
        flow_time_key="flow_time",
    )
    assert assert_flow_h0_keys_reach_model(flow, model) is None

    for owner, attribute, value, match in (
        (model, "block_native_add_h0", True, "prediction.add_h0=false"),
        (
            model.embedding.flow_time_conditioner,
            "allow_missing_time",
            True,
            "flow_time_allow_missing=false",
        ),
        (model, "blockwise_hamiltonian", False, "one NNENV owner"),
        (
            model.embedding,
            "require_full_block_edge_coverage",
            False,
            "require_full_block_edge_coverage=true",
        ),
    ):
        original = getattr(owner, attribute)
        setattr(owner, attribute, value)
        try:
            with pytest.raises(ValueError, match=match):
                assert_flow_h0_keys_reach_model(flow, model)
        finally:
            setattr(owner, attribute, original)

    data = _data(model)
    data[_keys.POSITIONS_KEY] = torch.tensor([[0.0, 0.0, 0.0], [5.5, 0.0, 0.0]])
    data[_keys.NODE_H0_KEY] = torch.zeros((2, model.idp.reduced_matrix_element))
    data[_keys.EDGE_H0_KEY] = torch.zeros((2, model.idp.reduced_matrix_element))
    data["flow_time"] = torch.tensor([0.25])

    with pytest.raises(ValueError, match="ordered full H-B0"):
        model(data)


def test_block_native_add_h0_exposes_full_h_without_changing_residual(tmp_path):
    model = _build("h_b0", tmp_path, {"add_h0": True})
    data = _data(model)
    max_norb = model.idp.full_basis_norb
    data[NODE_H0_BLOCKS_KEY] = torch.randn(2, max_norb, max_norb)
    data[EDGE_H0_BLOCKS_KEY] = torch.randn(2, max_norb, max_norb)
    h0_node = data[NODE_H0_BLOCKS_KEY].clone()
    h0_edge = data[EDGE_H0_BLOCKS_KEY].clone()

    output = model(data)
    residual_node = output[NODE_PRED_HAMIL_BLOCKS_KEY]
    residual_edge = output[EDGE_PRED_HAMIL_BLOCKS_KEY]
    assert torch.equal(output["node_full_hamil_blocks"], h0_node + residual_node)
    assert torch.equal(output["edge_full_hamil_blocks"], h0_edge + residual_edge)
    assert torch.equal(output[NODE_PRED_HAMIL_BLOCKS_KEY], residual_node)
    assert torch.equal(output[EDGE_PRED_HAMIL_BLOCKS_KEY], residual_edge)


def test_block_native_add_h0_fails_closed_without_converted_h0(tmp_path):
    model = _build("h_b0", tmp_path, {"add_h0": True})
    with pytest.raises(KeyError, match="Enable get_H0"):
        model(_data(model))


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
        assert NODE_PRED_HAMIL_BLOCKS_KEY in output
        assert EDGE_PRED_HAMIL_BLOCKS_KEY in output
        assert output[_keys.NODE_HAMILTONIAN_KEY].shape[-2:] == (
            model.idp.full_basis_norb,
            model.idp.full_basis_norb,
        )
        assert output[_keys.EDGE_HAMILTONIAN_KEY].shape[-2:] == (
            model.idp.full_basis_norb,
            model.idp.full_basis_norb,
        )
        output[NODE_DELTA_HAMIL_BLOCKS_KEY] = output[NODE_PRED_HAMIL_BLOCKS_KEY].detach().clone()
        output[EDGE_DELTA_HAMIL_BLOCKS_KEY] = output[EDGE_PRED_HAMIL_BLOCKS_KEY].detach().clone()
        output[NODE_DELTA_HAMIL_BLOCK_SHAPE_KEY] = output["node_hamil_block_shape"].clone()
        output[EDGE_DELTA_HAMIL_BLOCK_SHAPE_KEY] = output["edge_hamil_block_shape"].clone()
        block_loss = HamilBlockwiseNexTHamLoss(idp=model.idp)(output)
        assert torch.isfinite(block_loss)
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
        full_norb = layout[-1][1]
        assert node.shape[-2:] == (full_norb, full_norb)

        # Predictions must live in the species-contiguous layout used by the
        # blockwise targets/loss: zero outside the top-left (n_i, n_j) box.
        node_shapes, edge_shapes = infer_block_shapes(output, model.idp)
        node_valid = block_mask_from_shapes(node_shapes, tuple(node.shape[-2:]))
        assert torch.count_nonzero(node.masked_select(~node_valid)) == 0
        edge_valid = block_mask_from_shapes(edge_shapes, tuple(edge.shape[-2:]))
        assert torch.count_nonzero(edge.masked_select(~edge_valid)) == 0


def test_block_native_layout_matches_blockwise_target_contract(tmp_path):
    """Water-basis regression: H (2s1p) skips the union 3s slot.

    Before the species-compaction fix, block heads emitted union-slot canvases
    (H rows {0,1,3,4,5}) while the blockwise targets/loss expect contiguous
    packing (H rows 0..4): row 2 of every H block was hard-zero in the
    prediction yet nonzero in the target, pinning all block-native routes at
    the zero-predictor plateau on multi-species data.
    """
    basis = {"H": "2s1p", "O": "3s2p1d"}
    hidden = "4x0e+4x1o+4x1e+4x2e+4x2o+4x3o+4x3e+4x4e"
    embedding = _embedding_options("h_b0", tmp_path)
    embedding["irreps_hidden"] = hidden
    model = build_model(
        common_options={
            "basis": basis,
            "overlap": False,
            "dtype": "float32",
            "device": "cpu",
        },
        model_options={
            "embedding": embedding,
            "prediction": {
                "method": "block_native",
                "scale_type": "no_scale",
                "block_decoder": "expansion_cg",
                "blockwise_hamiltonian": True,
            },
        },
        train_options={},
        no_check=False,
    )
    output = model(_data(model))
    node = output[NODE_PRED_HAMIL_BLOCKS_KEY]
    edge = output[EDGE_PRED_HAMIL_BLOCKS_KEY]
    atom_type = output[_keys.ATOM_TYPE_KEY].flatten()
    h_type = model.idp.chemical_symbol_to_type["H"]
    h_nodes = torch.nonzero(atom_type == h_type, as_tuple=False).flatten()
    assert h_nodes.numel() > 0

    node_shapes, edge_shapes = infer_block_shapes(output, model.idp)
    node_valid = block_mask_from_shapes(node_shapes, tuple(node.shape[-2:]))
    assert torch.count_nonzero(node.masked_select(~node_valid)) == 0
    edge_valid = block_mask_from_shapes(edge_shapes, tuple(edge.shape[-2:]))
    assert torch.count_nonzero(edge.masked_select(~edge_valid)) == 0

    # H onsite is 5x5 contiguous; rows 2:5 are its p shell and must carry
    # signal (union-slot layout left row 2 identically zero).
    h_block = node[h_nodes[0]]
    assert torch.count_nonzero(h_block[:5, :5]) > 0
    assert torch.count_nonzero(h_block[2:5, :5]) > 0
    assert torch.count_nonzero(h_block[5:, :]) == 0
    assert torch.count_nonzero(h_block[:, 5:]) == 0


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
