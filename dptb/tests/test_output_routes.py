"""Output-route registry and end-to-end model wiring: legacy aliases resolve to the
canonical spec, config YAMLs pass strict argcheck, and real AtomicData forward/backward
runs through every one of the 6 official routes plus the legacy rme_head_mode aliases.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml
from e3nn import o3

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
from dptb.nn.embedding.ao_projector_bank import export_projector_bank
from dptb.nn.embedding.cartesian_ict_bank import export_cartesian_ict_projector_bank
from dptb.nn.embedding.cartesian_projector import ao_shell_layout
from dptb.nn.embedding.output_routes import (
    OFFICIAL_OUTPUT_ROUTES,
    effective_product_scope,
    get_output_route_spec,
    normalize_legacy_head_mode,
    resolve_output_route,
    select_final_irreps,
    validate_prediction_route,
)
from dptb.nn.deeptb import _resolve_embedding_output_route_spec
from dptb.nnops.blockwise_nextham_loss import HamilBlockwiseNexTHamLoss
from dptb.nnops.flow import assert_flow_h0_keys_reach_model
from dptb.utils.argcheck import model_options

BASIS = {"H": "1s", "O": "1s1p"}
HIDDEN = "4x0e+4x1o+4x1e+4x2e"
ROUTES = ("h_a0", "h_a1", "h_b0", "h_b1", "p_b0", "p_b1_ict")
# Ground truth from the registry (get_output_route_spec), pinned literally so this
# does not become a self-comparison against the registry it is meant to check.
ROUTE_CONTRACTS = {
    "h_a0": ("rme", True),
    "h_a1": ("rme", True),
    "h_b0": ("ao_block", False),
    "h_b1": ("ao_block", False),
    "p_b0": ("ao_block", False),
    "p_b1_ict": ("ao_block", False),
}


# ---------------------------------------------------------------------------
# Real AtomicData model construction and forward/backward, per canonical route
# ---------------------------------------------------------------------------

def _embedding_options(route: str, tmp_path: Path) -> dict:
    options = {
        "method": "lem_moe_v3", "output_route": route, "n_layers": 1, "avg_num_neighbors": 2.0, "r_max": 4.0,
        "irreps_hidden": HIDDEN, "env_embed_multiplicity": 4, "latent_dim": 8, "latent_channels": [8],
        "edge_one_hot_dim": 4, "num_experts": 1, "num_shared_experts": 1, "top_k": 1, "universal": True,
        "use_layer_onehot_tp": False, "use_out_onehot_tp": False, "use_interpolation_out": False,
        "tp_radial_emb": False, "mole_linear_mode": "indexed_ref", "so2_fusion_mode": "streamed_m_major_ref",
        "rme_fusion_rank": 4, "rme_fusion_init": 0.0,
    }
    if route == "p_b0":
        options["ao_projector_backend"] = "reference_wigner"
    elif route == "p_b1_ict":
        bank = export_cartesian_ict_projector_bank(tmp_path / "sp_ict_projectors.json", ("1s", "1p"))
        options.update({"ao_projector_backend": "precomputed", "ao_projector_bank_path": str(bank)})
    return options


def _build(route: str, tmp_path: Path, prediction_overrides: dict | None = None, embedding_overrides: dict | None = None):
    spec = get_output_route_spec(route)
    prediction = {"method": spec.prediction_method, "scale_type": "no_scale"}
    if spec.block_decoder is not None:
        prediction.update({"block_decoder": spec.block_decoder, "blockwise_hamiltonian": True})
    if prediction_overrides:
        prediction.update(prediction_overrides)
    embedding = _embedding_options(route, tmp_path)
    if embedding_overrides:
        embedding.update(embedding_overrides)
    return build_model(
        common_options={"basis": BASIS, "overlap": False, "dtype": "float32", "device": "cpu"},
        model_options={"embedding": embedding, "prediction": prediction}, train_options={}, no_check=False,
    )


def _data(model):
    h, o = model.idp.chemical_symbol_to_type["H"], model.idp.chemical_symbol_to_type["O"]
    edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    edge_type = torch.tensor([model.idp.bond_to_type["H-O"], model.idp.bond_to_type["O-H"]], dtype=torch.long)
    return {
        _keys.POSITIONS_KEY: torch.tensor([[0.0, 0.0, 0.0], [1.1, 0.2, -0.1]], dtype=torch.float32, requires_grad=True),
        _keys.EDGE_INDEX_KEY: edge_index,
        _keys.ATOM_TYPE_KEY: torch.tensor([[h], [o]], dtype=torch.long),
        _keys.EDGE_TYPE_KEY: edge_type,
    }


def test_real_hb0_block_ode_guards_model_contract_and_active_rows(tmp_path):
    model = _build("h_b0", tmp_path, embedding_overrides={
        "method": "lem_moe_v3_h0", "use_h0_init": True, "use_flow_time_embedding": True,
        "flow_time_allow_missing": False, "require_full_block_edge_coverage": True,
    })
    flow = SimpleNamespace(enabled=True, block_ode=True, node_h0_key=_keys.NODE_H0_KEY,
                           edge_h0_key=_keys.EDGE_H0_KEY, flow_time_key="flow_time")
    assert assert_flow_h0_keys_reach_model(flow, model) is None

    for owner, attribute, value, match in (
        (model, "block_native_add_h0", True, "prediction.add_h0=false"),
        (model.embedding.flow_time_conditioner, "allow_missing_time", True, "flow_time_allow_missing=false"),
        (model, "blockwise_hamiltonian", False, "one NNENV owner"),
        (model.embedding, "require_full_block_edge_coverage", False, "require_full_block_edge_coverage=true"),
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
    h0_node, h0_edge = data[NODE_H0_BLOCKS_KEY].clone(), data[EDGE_H0_BLOCKS_KEY].clone()

    output = model(data)
    residual_node, residual_edge = output[NODE_PRED_HAMIL_BLOCKS_KEY], output[EDGE_PRED_HAMIL_BLOCKS_KEY]
    assert torch.equal(output["node_full_hamil_blocks"], h0_node + residual_node)
    assert torch.equal(output["edge_full_hamil_blocks"], h0_edge + residual_edge)
    assert torch.equal(output[NODE_PRED_HAMIL_BLOCKS_KEY], residual_node)
    assert torch.equal(output[EDGE_PRED_HAMIL_BLOCKS_KEY], residual_edge)


def test_block_native_add_h0_fails_closed_without_converted_h0(tmp_path):
    model = _build("h_b0", tmp_path, {"add_h0": True})
    with pytest.raises(KeyError, match="Enable get_H0"):
        model(_data(model))


@pytest.mark.parametrize("route", ROUTES)
def test_real_atomicdata_forward_backward_and_e3_routing(route, tmp_path):
    output_contract, uses_e3hamiltonian = ROUTE_CONTRACTS[route]
    model = _build(route, tmp_path)
    spec = model.embedding.output_route_spec
    assert spec.canonical_name == route
    # The registry must agree with the literal contract pinned above.
    assert spec.output_contract == output_contract
    assert spec.uses_e3hamiltonian == uses_e3hamiltonian

    e3_calls = 0
    if uses_e3hamiltonian:
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
    assert e3_calls == int(uses_e3hamiltonian)

    if output_contract == "rme":
        assert _keys.NODE_FEATURES_KEY in output
        assert _keys.EDGE_FEATURES_KEY in output
        assert _keys.NODE_HAMILTONIAN_KEY not in output
        loss = output[_keys.NODE_FEATURES_KEY].square().mean() + output[_keys.EDGE_FEATURES_KEY].square().mean()
    else:
        assert _keys.NODE_HAMILTONIAN_KEY in output
        assert _keys.EDGE_HAMILTONIAN_KEY in output
        assert NODE_PRED_HAMIL_BLOCKS_KEY in output
        assert EDGE_PRED_HAMIL_BLOCKS_KEY in output
        assert output[_keys.NODE_HAMILTONIAN_KEY].shape[-2:] == (model.idp.full_basis_norb, model.idp.full_basis_norb)
        assert output[_keys.EDGE_HAMILTONIAN_KEY].shape[-2:] == (model.idp.full_basis_norb, model.idp.full_basis_norb)
        output[NODE_DELTA_HAMIL_BLOCKS_KEY] = output[NODE_PRED_HAMIL_BLOCKS_KEY].detach().clone()
        output[EDGE_DELTA_HAMIL_BLOCKS_KEY] = output[EDGE_PRED_HAMIL_BLOCKS_KEY].detach().clone()
        output[NODE_DELTA_HAMIL_BLOCK_SHAPE_KEY] = output["node_hamil_block_shape"].clone()
        output[EDGE_DELTA_HAMIL_BLOCK_SHAPE_KEY] = output["edge_hamil_block_shape"].clone()
        block_loss = HamilBlockwiseNexTHamLoss(idp=model.idp)(output)
        assert torch.isfinite(block_loss)
        loss = output[_keys.NODE_HAMILTONIAN_KEY].square().mean() + output[_keys.EDGE_HAMILTONIAN_KEY].square().mean()

    loss.backward()
    assert data[_keys.POSITIONS_KEY].grad is not None
    assert torch.isfinite(data[_keys.POSITIONS_KEY].grad).all()
    assert any(parameter.grad is not None for parameter in model.parameters() if parameter.requires_grad)


def test_block_routes_respect_shell_slices_and_species_masks(tmp_path):
    for route in ("h_b0", "h_b1", "p_b0", "p_b1_ict"):
        model = _build(route, tmp_path)
        output = model(_data(model))
        node, edge = output[_keys.NODE_HAMILTONIAN_KEY], output[_keys.EDGE_HAMILTONIAN_KEY]
        full_norb = ao_shell_layout(model.idp.full_basis)[-1][1]
        assert node.shape[-2:] == (full_norb, full_norb)

        # Predictions live in the species-contiguous layout used by the blockwise
        # targets/loss: zero outside the top-left (n_i, n_j) box.
        node_shapes, edge_shapes = infer_block_shapes(output, model.idp)
        node_valid = block_mask_from_shapes(node_shapes, tuple(node.shape[-2:]))
        assert torch.count_nonzero(node.masked_select(~node_valid)) == 0
        edge_valid = block_mask_from_shapes(edge_shapes, tuple(edge.shape[-2:]))
        assert torch.count_nonzero(edge.masked_select(~edge_valid)) == 0


def test_block_native_layout_matches_blockwise_target_contract(tmp_path):
    """Water-basis regression: H (2s1p) skips the union 3s slot.

    Before the species-compaction fix, block heads emitted union-slot canvases
    (H rows {0,1,3,4,5}) while the blockwise targets/loss expect contiguous packing
    (H rows 0..4): row 2 of every H block was hard-zero in the prediction yet
    nonzero in the target, pinning all block-native routes at the zero-predictor
    plateau on multi-species data.
    """
    basis = {"H": "2s1p", "O": "3s2p1d"}
    hidden = "4x0e+4x1o+4x1e+4x2e+4x2o+4x3o+4x3e+4x4e"
    embedding = _embedding_options("h_b0", tmp_path)
    embedding["irreps_hidden"] = hidden
    model = build_model(
        common_options={"basis": basis, "overlap": False, "dtype": "float32", "device": "cpu"},
        model_options={"embedding": embedding, "prediction": {"method": "block_native", "scale_type": "no_scale",
                                                              "block_decoder": "expansion_cg", "blockwise_hamiltonian": True}},
        train_options={}, no_check=False,
    )
    output = model(_data(model))
    node, edge = output[NODE_PRED_HAMIL_BLOCKS_KEY], output[EDGE_PRED_HAMIL_BLOCKS_KEY]
    atom_type = output[_keys.ATOM_TYPE_KEY].flatten()
    h_nodes = torch.nonzero(atom_type == model.idp.chemical_symbol_to_type["H"], as_tuple=False).flatten()
    assert h_nodes.numel() > 0

    node_shapes, edge_shapes = infer_block_shapes(output, model.idp)
    node_valid = block_mask_from_shapes(node_shapes, tuple(node.shape[-2:]))
    assert torch.count_nonzero(node.masked_select(~node_valid)) == 0
    edge_valid = block_mask_from_shapes(edge_shapes, tuple(edge.shape[-2:]))
    assert torch.count_nonzero(edge.masked_select(~edge_valid)) == 0

    # H onsite is 5x5 contiguous; rows 2:5 are its p shell and must carry signal
    # (the old union-slot layout left row 2 identically zero).
    h_block = node[h_nodes[0]]
    assert torch.count_nonzero(h_block[:5, :5]) > 0
    assert torch.count_nonzero(h_block[2:5, :5]) > 0
    assert torch.count_nonzero(h_block[5:, :]) == 0
    assert torch.count_nonzero(h_block[:, 5:]) == 0


# ---------------------------------------------------------------------------
# Registry: legacy aliases, provenance-gated resolution, prediction validation
# ---------------------------------------------------------------------------

def test_legacy_aliases_resolve_without_changing_semantics():
    aliases = {
        "late_rme_expansion_nocg": "h_a0", "late_rme_cartesian_hybrid": "h_a1",
        "late_block_expansion_cg": "h_b0", "late_block_cartesian_projector": "h_b1",
        "rme_fusion": "rme_fusion", "block_native_linear": "debug_block_linear",
    }
    for alias, canonical in aliases.items():
        with pytest.warns(DeprecationWarning):
            spec = resolve_output_route(legacy_mode=alias)
        assert spec.canonical_name == canonical
        assert normalize_legacy_head_mode(alias) == spec.legacy_mode


def test_alias_in_canonical_field_emits_deprecation_warning():
    with pytest.warns(DeprecationWarning, match="output_route='h_a0'"):
        spec = resolve_output_route(output_route="late_nocg")
    assert spec.canonical_name == "h_a0"


def test_legacy_embedding_without_output_route_spec_defaults_to_legacy_rme():
    spec = _resolve_embedding_output_route_spec(SimpleNamespace(), embedding_options={},
                                                prediction_options={"method": "e3tb"})
    assert spec.canonical_name == "legacy_rme"


def test_legacy_embedding_without_output_route_spec_rejects_new_route():
    with pytest.raises(RuntimeError, match="output_route_spec"):
        _resolve_embedding_output_route_spec(SimpleNamespace(), embedding_options={"output_route": "h_a0"},
                                             prediction_options={"method": "e3tb"})


def test_legacy_direct_ao_alias_resolves_from_backend_and_provenance(tmp_path):
    reference = export_projector_bank(tmp_path / "reference.json", ("s", "p"))
    ict = export_cartesian_ict_projector_bank(tmp_path / "ict.json", ("s", "p"))

    with pytest.warns(DeprecationWarning):
        assert resolve_output_route(legacy_mode="direct_ao_projector",
                                    projector_backend="reference_wigner").canonical_name == "p_b0"
    with pytest.warns(DeprecationWarning):
        assert resolve_output_route(legacy_mode="direct_ao_projector", projector_backend="precomputed",
                                    projector_bank_path=reference).canonical_name == "p_b1_reference"
    with pytest.warns(DeprecationWarning):
        assert resolve_output_route(legacy_mode="direct_ao_projector", projector_backend="precomputed",
                                    projector_bank_path=ict).canonical_name == "p_b1_ict"


def test_canonical_p_routes_reject_wrong_provenance(tmp_path):
    reference = export_projector_bank(tmp_path / "reference.json", ("s", "p"))
    ict = export_cartesian_ict_projector_bank(tmp_path / "ict.json", ("s", "p"))
    with pytest.raises(ValueError, match="validated Cartesian/ICT"):
        resolve_output_route(output_route="p_b1_ict", projector_backend="precomputed", projector_bank_path=reference)
    with pytest.raises(ValueError, match="non-ICT reference/control"):
        resolve_output_route(output_route="p_b1_reference", projector_backend="precomputed", projector_bank_path=ict)


def test_official_product_policies_are_fixed():
    assert effective_product_scope(get_output_route_spec("h_a1"), "all") == "all"
    assert effective_product_scope(get_output_route_spec("h_b1"), "missing_only") == "missing_only"
    with pytest.raises(ValueError, match="fixes product_scope"):
        effective_product_scope(get_output_route_spec("h_a1"), "missing_only")


def test_prediction_validation_is_registry_driven():
    validate_prediction_route(get_output_route_spec("h_a1"), {"method": "e3tb", "scale_type": "no_scale"})
    validate_prediction_route(get_output_route_spec("h_b1"), {"method": "block_native", "block_decoder": "cartesian_projector"})
    with pytest.raises(ValueError, match="prediction.method"):
        validate_prediction_route(get_output_route_spec("h_a1"), {"method": "block_native"})
    with pytest.raises(ValueError, match="block_decoder"):
        validate_prediction_route(get_output_route_spec("h_b1"), {"method": "block_native", "block_decoder": "expansion_cg"})


def test_final_irreps_selection_uses_route_spec():
    hidden = o3.Irreps("2x0e+2x1o")
    orbpair = o3.Irreps("1x1o+2x0e")
    ao_pair = o3.Irreps("2x0e+1x1o")
    assert select_final_irreps(get_output_route_spec("h_a0"), ordinary_hidden=hidden, orbpair_irreps=orbpair,
                               ao_pair_irreps=ao_pair) == hidden
    assert select_final_irreps(get_output_route_spec("legacy_rme"), ordinary_hidden=hidden, orbpair_irreps=orbpair,
                               ao_pair_irreps=ao_pair) == orbpair.sort()[0].simplify()
    assert select_final_irreps(get_output_route_spec("p_b0"), ordinary_hidden=hidden, orbpair_irreps=orbpair,
                               ao_pair_irreps=ao_pair) == ao_pair


# ---------------------------------------------------------------------------
# Config YAMLs / argcheck, and legacy rme_head_mode builds (the pre-registry
# calling convention: model_options() rather than output_route=)
# ---------------------------------------------------------------------------

CONFIG_ROOT = Path(__file__).resolve().parents[2]
ROUTE_CONFIGS = {
    "h_a0": "route_h_a0_late_rme_expansion_nocg.yaml",
    "h_a1": "route_h_a1_late_rme_cartesian_hybrid.yaml",
    "h_b0": "route_h_b0_late_block_expansion_cg.yaml",
    "h_b1": "route_h_b1_late_block_cartesian_projector.yaml",
    "p_b0": "route_p_b0_direct_ao_projector_wigner.yaml",
    "p_b1_ict": "route_p_b1_direct_ao_projector_ict_bank.yaml",
}


def test_six_canonical_route_configs_pass_strict_argcheck():
    assert tuple(ROUTE_CONFIGS) == OFFICIAL_OUTPUT_ROUTES
    model_arg = model_options()
    for route, filename in ROUTE_CONFIGS.items():
        payload = yaml.safe_load((CONFIG_ROOT / "configs" / filename).read_text())
        normalized = model_arg.normalize_value(payload["model_options"])
        model_arg.check_value(normalized, strict=True)
        assert normalized["embedding"]["output_route"] == route


def test_output_route_and_conflicting_legacy_alias_are_rejected_by_model():
    # dargs validates the two compatibility fields independently; semantic conflict
    # is deliberately centralized in the route registry/model build.
    with pytest.raises(ValueError, match="conflicts"):
        resolve_output_route(output_route="h_a0", legacy_mode="late_block_expansion_cg")


ORDINARY_HIDDEN = "4x0e+4x1o+4x1e+4x2e+4x2o+4x3o+4x3e+4x4e"


def _legacy_model_options(mode, prediction, extra_embedding=None):
    embedding = {
        "method": "lem_moe_v3", "n_layers": 1, "avg_num_neighbors": 2.0, "r_max": 4.0,
        "irreps_hidden": ORDINARY_HIDDEN, "env_embed_multiplicity": 4, "latent_dim": 8, "latent_channels": [8],
        "edge_one_hot_dim": 4, "num_experts": 1, "num_shared_experts": 1, "top_k": 1, "universal": True,
        "use_layer_onehot_tp": False, "use_out_onehot_tp": True, "use_interpolation_out": False,
        "tp_radial_emb": False, "mole_linear_mode": "indexed_ref", "so2_fusion_mode": "streamed_m_major_ref",
        "rme_head_mode": mode, "rme_fusion_rank": 4, "rme_fusion_init": 0.0,
    }
    if extra_embedding:
        embedding.update(extra_embedding)
    return {"embedding": embedding, "prediction": prediction}


def _build_legacy_route(mode, prediction, extra_embedding=None):
    return build_model(
        common_options={"basis": {"H": "2s1p", "O": "3s2p1d"}, "overlap": False, "dtype": "float32", "device": "cpu"},
        model_options=_legacy_model_options(mode, prediction, extra_embedding), train_options={}, no_check=True,
    )


@pytest.mark.parametrize(
    ("mode", "head_name", "uses_ict", "prediction"),
    [
        ("late_rme_expansion_nocg", "LateRMEExpansionNoCGHead", False, {"method": "e3tb", "scale_type": "no_scale"}),
        ("late_rme_cartesian_hybrid", "LateRMECartesianHybridHead", True, {"method": "e3tb", "scale_type": "no_scale"}),
        ("late_block_expansion_cg", "LateBlockExpansionCGHead", False,
         {"method": "block_native", "block_decoder": "expansion_cg", "blockwise_hamiltonian": True, "scale_type": "no_scale"}),
        ("late_block_cartesian_projector", "LateBlockCartesianProjectorHead", True,
         {"method": "block_native", "block_decoder": "cartesian_projector", "blockwise_hamiltonian": True, "scale_type": "no_scale"}),
    ],
)
def test_legacy_rme_head_mode_model_routes_keep_final_hidden_contract(mode, head_name, uses_ict, prediction):
    extra_embedding = {"rme_cartesian_scope": "all"} if mode == "late_rme_cartesian_hybrid" else None
    model = _build_legacy_route(mode, prediction, extra_embedding)
    embedding = model.embedding

    assert embedding.layers[-1].irreps_out == o3.Irreps(ORDINARY_HIDDEN)
    assert type(embedding.out_node).__name__ == head_name
    assert getattr(embedding.out_node, "uses_ict") is uses_ict
    if prediction["method"] == "block_native":
        assert not hasattr(model, "hamiltonian")
        assert embedding.out_node.output_contract == "ao_block"
    else:
        assert hasattr(model, "hamiltonian")
        assert embedding.out_node.output_contract == "rme"
    if mode == "late_rme_cartesian_hybrid":
        assert embedding.out_node.coverage_report["product_paths"] > 0
    if mode == "late_block_cartesian_projector":
        assert embedding.out_node.coverage_report["product_paths"] == 0
        assert not hasattr(embedding.out_node, "left")
        assert not hasattr(embedding.out_node, "right")


@pytest.mark.parametrize(("backend", "uses_ict"), [("reference_wigner", False), ("precomputed", False)])
def test_legacy_ao_pair_recontract_model_routes_change_final_layer_to_ao_pair_irreps(tmp_path, backend, uses_ict):
    extra = {"rme_head_mode": "direct_ao_projector", "ao_projector_backend": backend}
    if backend == "precomputed":
        bank = export_projector_bank(tmp_path / "reference_projectors.json", ("s", "s", "s", "p", "p", "d"))
        extra["ao_projector_bank_path"] = str(bank)
    model = _build_legacy_route(
        "direct_ao_projector",
        {"method": "block_native", "block_decoder": "ao_projector", "blockwise_hamiltonian": True, "scale_type": "no_scale"},
        extra,
    )

    embedding = model.embedding
    assert not hasattr(model, "hamiltonian")
    assert embedding.layers[-1].irreps_out.dim == 14 * 14
    assert type(embedding.out_node).__name__ == "AOAngularProjectorHead"
    assert embedding.out_node.uses_ict is uses_ict
    assert embedding.out_node.uses_precomputed_projector is (backend == "precomputed")
