from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest
import torch
import yaml

from dptb.data import _keys
from dptb.data.interfaces.blockwise_tensor import (
    BlockTensorResult,
    canonical_block_tensors_to_feature_tensors,
)
from dptb.data.transforms import OrbitalMapper
from dptb.nn.hamiltonian import _contract_cg_rme, _inverse_contract_cg_hr
from dptb.nnops.block_flow_codec import BlockStateCodec, project_block_state
from dptb.nnops.flow import HamiltonianCFM
from dptb.tests.test_block_flow_codec import _mixed_case, _random_state
from dptb.tests.test_block_ode_flow import (
    EndpointSequence,
    _case,
    _flow,
    _fresh,
    _pd_legacy_product_h0_case,
    _scaled_endpoint,
)
from dptb.utils.argcheck import slem_h0, validate_block_ode_contract


FP64_ATOL = 1e-10


def _absolute_ref_for_endpoint(
    flow: HamiltonianCFM,
    data: dict,
    codec: BlockStateCodec,
    endpoint: BlockTensorResult,
) -> dict:
    node_rme, edge_rme = codec.blocks_to_rme(data, endpoint)
    ref = _fresh(data)
    ref[flow.node_target_key] = node_rme
    ref[flow.edge_target_key] = edge_rme
    ref[flow.node_block_target_key] = endpoint.node_blocks.clone()
    ref[flow.edge_block_target_key] = endpoint.edge_blocks.clone()
    ref[flow.node_block_shape_key] = endpoint.node_shapes.clone()
    ref[flow.edge_block_shape_key] = endpoint.edge_shapes.clone()
    return ref


def _state_to_dtype(state: BlockTensorResult, dtype: torch.dtype) -> BlockTensorResult:
    return BlockTensorResult(
        state.node_blocks.to(dtype=dtype),
        state.edge_blocks.to(dtype=dtype),
        state.node_shapes,
        state.edge_shapes,
    )


def _valid_contract() -> dict:
    return {
        "train_options": {
            "flow_options": {
                "enabled": True,
                "mode": "residual",
                "prior": "zero",
                "output_space": "ao_block_ode",
                "block_ode": True,
                "target_semantics": "absolute_full_h",
                "prediction_add_h0": False,
                "time_conditioning_required": True,
                "strict_h0": True,
                "block_inverse_mode": "strict",
                "node_block_target_key": "node_full_hamil_target_blocks",
                "edge_block_target_key": "edge_full_hamil_target_blocks",
                "node_block_shape_key": "node_full_hamil_target_block_shape",
                "edge_block_shape_key": "edge_full_hamil_target_block_shape",
                "validation_ode_steps": [1, 3],
            }
        },
        "common_options": {"has_soc": False},
        "model_options": {
            "embedding": {
                "method": "lem_moe_v3_h0",
                "output_route": "h_b0",
                "require_full_block_edge_coverage": True,
                "use_flow_time_embedding": True,
                "flow_time_condition_edges": True,
                "flow_time_allow_missing": False,
                "flow_time_key": "flow_time",
                "h0_merge_mode": "replace",
                "use_h0_node_init": True,
                "use_h0_edge_init": True,
            },
            "prediction": {
                "method": "block_native",
                "block_decoder": "expansion_cg",
                "blockwise_hamiltonian": True,
                "add_h0": False,
            },
        },
        "data_options": {
            "train": {
                "type": "LMDBDataset",
                "get_Hamiltonian": True,
                "get_H0": True,
                "residual_hamiltonian": False,
                "require_full_h_target": True,
            }
        },
    }


def test_red_inverse_cg_nonorthogonal_solve():
    nonorthogonal = torch.tensor([[[2.0, 0.0], [0.0, 3.0]]], dtype=torch.float64)
    rme = torch.tensor([[[1.25], [-0.75]]], dtype=torch.float64)
    product = _contract_cg_rme(nonorthogonal, rme)
    restored = _inverse_contract_cg_hr(nonorthogonal, product)
    assert torch.linalg.cond(nonorthogonal.reshape(2, 2)).item() == pytest.approx(1.5)
    torch.testing.assert_close(restored, rme, rtol=0.0, atol=1e-12)


def test_red_codec_entry_rejects_missing_and_duplicate_reverse_edges():
    idp, data = _mixed_case()
    codec = BlockStateCodec(idp, dtype=torch.float64)
    state = project_block_state(data, idp, _random_state(idp, data))

    missing_data = dict(data)
    for key in ("edge_index", "edge_cell_shift", "edge_type"):
        missing_data[key] = data[key][..., :3] if key == "edge_index" else data[key][:3]
    missing_state = BlockTensorResult(
        state.node_blocks,
        state.edge_blocks[:3],
        state.node_shapes,
        state.edge_shapes[:3],
    )
    with pytest.raises(ValueError, match="missing reverse"):
        codec.blocks_to_rme(missing_data, missing_state)

    duplicate_data = dict(data)
    duplicate_data["edge_index"] = torch.cat(
        [data["edge_index"], data["edge_index"][:, :1]], dim=1
    )
    duplicate_data["edge_cell_shift"] = torch.cat(
        [data["edge_cell_shift"], data["edge_cell_shift"][:1]], dim=0
    )
    duplicate_data["edge_type"] = torch.cat(
        [data["edge_type"], data["edge_type"][:1]], dim=0
    )
    duplicate_state = BlockTensorResult(
        state.node_blocks,
        torch.cat([state.edge_blocks, state.edge_blocks[:1]], dim=0),
        state.node_shapes,
        torch.cat([state.edge_shapes, state.edge_shapes[:1]], dim=0),
    )
    with pytest.raises(ValueError, match="Duplicate directed edge key"):
        codec.blocks_to_rme(duplicate_data, duplicate_state)


def test_red_species_compact_mapper_really_skips_union_middle_shell():
    idp = OrbitalMapper(
        {"Si": ["3s", "3d"], "C": ["2p"]}, method="e3tb", device="cpu"
    )
    idp.get_orbital_maps()
    idp.get_irreps(no_parity=False)
    assert idp.full_basis == ["1s", "1p", "1d"]
    assert list(idp.basis_to_full_basis["Si"].values()) == ["1s", "1d"]
    assert idp.orbital_maps["Si"]["3d"].start == 1

    atom_types = torch.tensor(
        [idp.chemical_symbol_to_type["Si"], idp.chemical_symbol_to_type["C"]]
    )
    edge_types = torch.tensor(
        [idp.bond_to_type["Si-C"], idp.bond_to_type["C-Si"]]
    )
    data = {
        "pos": torch.zeros(2, 3, dtype=torch.float64),
        "atom_types": atom_types,
        "edge_type": edge_types,
        "edge_index": torch.tensor([[0, 1], [1, 0]]),
        "edge_cell_shift": torch.zeros(2, 3, dtype=torch.float64),
    }
    generator = torch.Generator().manual_seed(718)
    node = torch.randn(2, idp.reduced_matrix_element, generator=generator, dtype=torch.float64)
    edge = torch.randn(2, idp.reduced_matrix_element, generator=generator, dtype=torch.float64)
    node *= idp.mask_to_nrme[atom_types].to(torch.float64)
    edge *= idp.mask_to_erme[edge_types].to(torch.float64)

    codec = BlockStateCodec(idp, dtype=torch.float64)
    first = codec.rme_to_blocks(data, node, edge)
    canonical_node, canonical_edge = codec.blocks_to_rme(data, first)
    rebuilt = codec.rme_to_blocks(data, canonical_node, canonical_edge)
    torch.testing.assert_close(rebuilt.node_blocks, first.node_blocks, rtol=0.0, atol=FP64_ATOL)
    torch.testing.assert_close(rebuilt.edge_blocks, first.edge_blocks, rtol=0.0, atol=FP64_ATOL)
    assert rebuilt.node_shapes.tolist() == [[6, 6], [3, 3]]
    assert rebuilt.edge_shapes.tolist() == [[6, 3], [3, 6]]


def test_red_second_step_is_not_polluted_by_e3_style_feature_overwrite():
    idp, data, _, h0 = _case()

    class MutatingFeatureModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.h0_inputs = []
            self.aliases = []

        def forward(self, batch):
            self.h0_inputs.append((batch["node_h0"].clone(), batch["edge_h0"].clone()))
            self.aliases.append(
                (
                    batch["node_h0"].data_ptr() == batch["node_features"].data_ptr(),
                    batch["edge_h0"].data_ptr() == batch["edge_features"].data_ptr(),
                )
            )
            # E3Hamiltonian overwrites feature tensors in place.  This mutation
            # must not alias the saved H0 tensors used by the next ODE step.
            batch["node_features"].add_(1234.0)
            batch["edge_features"].sub_(567.0)
            out = batch.copy()
            out["node_hamil_blocks"] = h0.node_blocks
            out["edge_hamil_blocks"] = h0.edge_blocks
            return out

    model = MutatingFeatureModel()
    _flow(idp).sample(model, _fresh(data), num_steps=2)
    assert model.aliases == [(False, False), (False, False)]
    torch.testing.assert_close(model.h0_inputs[1][0], model.h0_inputs[0][0], rtol=0.0, atol=1e-12)
    torch.testing.assert_close(model.h0_inputs[1][1], model.h0_inputs[0][1], rtol=0.0, atol=1e-12)


def test_red_sampler_ignores_legacy_pd_product_h0_and_uses_blocks():
    (
        idp,
        data,
        codec,
        _,
        canonical_node_h0,
        canonical_edge_h0,
    ) = _pd_legacy_product_h0_case()
    endpoint = _scaled_endpoint(codec, data, canonical_node_h0, canonical_edge_h0, 1.2)

    model = EndpointSequence([endpoint])
    result = _flow(idp).sample(model, _fresh(data), num_steps=1)

    assert (data["node_h0"] - canonical_node_h0).abs().max().item() > 1.0e-3
    assert (data["edge_h0"] - canonical_edge_h0).abs().max().item() > 1.0e-3
    torch.testing.assert_close(model.inputs[0][0], canonical_node_h0, rtol=0.0, atol=FP64_ATOL)
    torch.testing.assert_close(model.inputs[0][1], canonical_edge_h0, rtol=0.0, atol=FP64_ATOL)
    torch.testing.assert_close(result["node_hamil_blocks"], endpoint.node_blocks, rtol=0.0, atol=FP64_ATOL)
    torch.testing.assert_close(result["edge_hamil_blocks"], endpoint.edge_blocks, rtol=0.0, atol=FP64_ATOL)


@pytest.mark.parametrize(
    "missing_key",
    (
        _keys.NODE_H0_BLOCKS_KEY,
        _keys.EDGE_H0_BLOCKS_KEY,
        _keys.NODE_H0_BLOCK_SHAPE_KEY,
        _keys.EDGE_H0_BLOCK_SHAPE_KEY,
    ),
)
def test_red_sampler_requires_physical_h0_blocks_and_shapes(missing_key):
    idp, data, _, h0 = _case()
    bad = _fresh(data)
    bad.pop(missing_key)

    with pytest.raises(KeyError) as excinfo:
        _flow(idp).sample(EndpointSequence([h0]), bad, num_steps=1)
    assert missing_key in str(excinfo.value)


@pytest.mark.parametrize(
    "missing_key",
    (
        _keys.NODE_H0_BLOCKS_KEY,
        _keys.EDGE_H0_BLOCKS_KEY,
        _keys.NODE_H0_BLOCK_SHAPE_KEY,
        _keys.EDGE_H0_BLOCK_SHAPE_KEY,
    ),
)
def test_red_prepare_requires_physical_h0_blocks_and_shapes(missing_key):
    idp, data, codec, h0 = _case()
    flow = _flow(idp)
    endpoint = _scaled_endpoint(codec, data, data["node_h0"], data["edge_h0"], 1.1)
    ref = _absolute_ref_for_endpoint(flow, data, codec, endpoint)
    bad = _fresh(data)
    bad.pop(missing_key)

    with pytest.raises(KeyError) as excinfo:
        flow.prepare_batch(bad, ref, t=torch.tensor([0.4], dtype=torch.float64))
    assert missing_key in str(excinfo.value)


def test_red_float32_grid_never_calls_t1_and_last_alpha_lands_on_endpoint():
    idp, data64, codec64, _ = _case()
    data = {
        key: value.to(torch.float32) if torch.is_tensor(value) and value.is_floating_point() else value.clone()
        if torch.is_tensor(value)
        else value
        for key, value in data64.items()
    }
    endpoint64 = _scaled_endpoint(
        codec64, data64, data64["node_h0"], data64["edge_h0"], 1.3
    )
    endpoint = _state_to_dtype(endpoint64, torch.float32)
    model = EndpointSequence([endpoint, endpoint, endpoint])
    flow = HamiltonianCFM(
        {
            "enabled": True,
            "mode": "residual",
            "prior": "zero",
            "output_space": "ao_block_ode",
            "block_ode": True,
            "target_semantics": "absolute_full_h",
            "prediction_add_h0": False,
            "time_conditioning_required": True,
            "block_inverse_mode": "strict",
            "block_inverse_atol": 2e-5,
            "node_block_target_key": "node_full_hamil_target_blocks",
            "edge_block_target_key": "edge_full_hamil_target_blocks",
            "node_block_shape_key": "node_full_hamil_target_block_shape",
            "edge_block_shape_key": "edge_full_hamil_target_block_shape",
            "validation_ode_steps": [1, 3],
        },
        idp=idp,
        dtype=torch.float32,
    )
    result = flow.sample(model, data, num_steps=3)
    got_times = torch.stack([value.reshape(()) for value in model.times])
    expected_times = torch.tensor([0.0, 1.0 / 3.0, 2.0 / 3.0], dtype=torch.float32)
    torch.testing.assert_close(got_times, expected_times, rtol=0.0, atol=torch.finfo(torch.float32).eps)
    assert bool((got_times < 1.0).all())
    sampled_train_t = flow._sample_t(
        num_graphs=4096, device=torch.device("cpu"), dtype=torch.float32
    )
    assert sampled_train_t.min().item() >= flow.t_min
    assert sampled_train_t.max().item() <= min(flow.t_max, 1.0 - flow.t_eps)
    torch.testing.assert_close(result["node_hamil_blocks"], endpoint.node_blocks, rtol=0.0, atol=2e-5)
    torch.testing.assert_close(result["edge_hamil_blocks"], endpoint.edge_blocks, rtol=0.0, atol=2e-5)


def test_red_projector_rejects_fractional_shapes_and_reports_real_correction():
    idp, data = _mixed_case()
    raw = _random_state(idp, data)
    projected = project_block_state(data, idp, raw)
    correction = max(
        (projected.node_blocks - raw.node_blocks).abs().max().item(),
        (projected.edge_blocks - raw.edge_blocks).abs().max().item(),
    )
    assert correction > 1e-3
    twice = project_block_state(data, idp, projected)
    torch.testing.assert_close(twice.node_blocks, projected.node_blocks, rtol=0.0, atol=FP64_ATOL)
    torch.testing.assert_close(twice.edge_blocks, projected.edge_blocks, rtol=0.0, atol=FP64_ATOL)

    fractional = BlockTensorResult(
        raw.node_blocks,
        raw.edge_blocks,
        raw.node_shapes.to(torch.float64) + 0.5,
        raw.edge_shapes.to(torch.float64) + 0.5,
    )
    with pytest.raises(ValueError, match="shapes must contain integers"):
        project_block_state(data, idp, fractional)


def test_red_strict_schema_accepts_required_flow_time_missing_gate():
    names = {argument.name for argument in slem_h0()}
    assert {"flow_time_allow_missing", "require_full_block_edge_coverage"} <= names


def test_red_codec_ignores_non_tensor_trainer_batch_metadata():
    idp, data, codec, h0 = _case()
    with_metadata = {
        **data,
        "__slices__": {"edge_index": [0, int(data["edge_index"].shape[1])]},
        "__data_class__": object,
    }
    packed = codec.rme_to_blocks(
        with_metadata, data["node_h0"], data["edge_h0"], project=True
    )
    node_rme, edge_rme = codec.blocks_to_rme(with_metadata, packed)
    assert torch.allclose(node_rme, data["node_h0"], rtol=0, atol=FP64_ATOL)
    assert torch.allclose(edge_rme, data["edge_h0"], rtol=0, atol=FP64_ATOL)
    assert torch.allclose(packed.node_blocks, h0.node_blocks, rtol=0, atol=FP64_ATOL)
    assert torch.allclose(packed.edge_blocks, h0.edge_blocks, rtol=0, atol=FP64_ATOL)


def test_red_argcheck_all_four_hard_gates_and_cross_products_raise():
    base = _valid_contract()
    assert validate_block_ode_contract(base) is None
    projected_te = deepcopy(base)
    projected_te["train_options"]["flow_options"].update(
        {
            "prior": "projected_te",
            "te_prior_mode": "irrep",
            "node_sigma": 1.0,
            "edge_sigma": 1.0,
            "te_prior_sigma": 1.0,
            "te_prior_validation_seed": 20260719,
        }
    )
    assert validate_block_ode_contract(projected_te) is None
    expected_fields = {
        "absolute_full_h": {
            "node_block_target_key": "node_full_hamil_target_blocks",
            "edge_block_target_key": "edge_full_hamil_target_blocks",
            "node_block_shape_key": "node_full_hamil_target_block_shape",
            "edge_block_shape_key": "edge_full_hamil_target_block_shape",
        },
        "residual_dh": {
            "node_block_target_key": "node_delta_hamil_blocks",
            "edge_block_target_key": "edge_delta_hamil_blocks",
            "node_block_shape_key": "node_delta_hamil_block_shape",
            "edge_block_shape_key": "edge_delta_hamil_block_shape",
        },
    }
    for filename in ("h_b0_block_ode_water.yaml", "h_b0_block_ode_crystal.yaml"):
        overlay = yaml.safe_load(
            (Path("configs") / filename).read_text(encoding="utf-8")
        )
        assert validate_block_ode_contract(overlay) is None
        flow = overlay["train_options"]["flow_options"]
        assert {
            option: flow[option] for option in expected_fields[flow["target_semantics"]]
        } == expected_fields[flow["target_semantics"]]
        assert flow["strict_h0"] is True
        assert overlay["common_options"]["has_soc"] is False
        embedding = overlay["model_options"]["embedding"]
        assert {
            key: embedding[key]
            for key in (
                "method",
                "output_route",
                "require_full_block_edge_coverage",
            )
        } == {
            "method": "lem_moe_v3_h0",
            "output_route": "h_b0",
            "require_full_block_edge_coverage": True,
        }
        expected_full_h = flow["target_semantics"] == "absolute_full_h"
        for split in ("train", "validation"):
            split_options = overlay["data_options"][split]
            assert split_options["get_Hamiltonian"] is True
            assert split_options["get_H0"] is True
            assert split_options["require_full_h_target"] is expected_full_h
            assert split_options["require_residual_h_target"] is not expected_full_h

    bad_cases = []
    bad = deepcopy(base)
    bad["train_options"]["flow_options"]["prior"] = "te"
    bad_cases.append((bad, "only prior='zero'.*prior='projected_te'"))
    for option, value, match in (
        ("te_prior_mode", "typewise", "requires te_prior_mode='irrep'"),
        ("node_sigma", 0.0, "finite positive scales"),
        ("edge_sigma", float("nan"), "finite positive scales"),
        ("te_prior_sigma", float("inf"), "finite positive scales"),
        ("te_prior_validation_seed", True, "explicit non-negative integer"),
    ):
        bad = deepcopy(projected_te)
        bad["train_options"]["flow_options"][option] = value
        bad_cases.append((bad, match))
    bad = deepcopy(projected_te)
    bad["train_options"]["flow_options"].pop("te_prior_validation_seed")
    bad_cases.append((bad, "explicit non-negative integer"))
    bad = deepcopy(base)
    bad["train_options"]["flow_options"]["target_semantics"] = ""
    bad_cases.append((bad, "explicit absolute_full_h/residual_dh"))
    bad = deepcopy(base)
    bad["train_options"]["flow_options"]["time_conditioning_required"] = False
    bad_cases.append((bad, "time_conditioning_required=true"))
    bad = deepcopy(base)
    bad["model_options"]["embedding"]["use_flow_time_embedding"] = False
    bad_cases.append((bad, "use_flow_time_embedding=true"))
    bad = deepcopy(base)
    bad["model_options"]["embedding"]["flow_time_condition_edges"] = False
    bad_cases.append((bad, "both nodes and edges"))
    bad = deepcopy(base)
    bad["model_options"]["embedding"]["flow_time_allow_missing"] = True
    bad_cases.append((bad, "flow_time_allow_missing=false"))
    bad = deepcopy(base)
    bad["model_options"]["prediction"]["add_h0"] = True
    bad_cases.append((bad, "prediction.add_h0=false"))
    for section, option, value, match in (
        ("embedding", "method", "lem_moe_v3", "embedding.method='lem_moe_v3_h0'"),
        ("embedding", "output_route", "h_b1", "embedding.output_route='h_b0'"),
        ("embedding", "require_full_block_edge_coverage", False, "full_block_edge_coverage=true"),
        ("prediction", "method", "e3tb", "prediction.method='block_native'"),
        ("prediction", "block_decoder", "cartesian_projector", "block_decoder='expansion_cg'"),
        ("prediction", "blockwise_hamiltonian", False, "blockwise_hamiltonian=true"),
    ):
        bad = deepcopy(base)
        bad["model_options"][section][option] = value
        bad_cases.append((bad, match))
    bad = deepcopy(base)
    bad["train_options"]["flow_options"]["block_ode"] = False
    bad_cases.append((bad, "distinct mode"))
    bad = deepcopy(base)
    bad["train_options"]["flow_options"]["output_space"] = "ao_block"
    bad_cases.append((bad, "distinct mode"))
    bad = deepcopy(base)
    bad["data_options"]["train"]["residual_hamiltonian"] = True
    bad_cases.append((bad, "conflicts"))
    bad = deepcopy(base)
    bad["train_options"]["flow_options"]["target_semantics"] = "residual_dh"
    bad_cases.append((bad, "target_semantics"))

    for bad, match in bad_cases:
        with pytest.raises(ValueError, match=match):
            validate_block_ode_contract(bad)


@pytest.mark.parametrize(
    "distance_ranges",
    (
        [[0.0, 2.0], [2.0, 6.0]],
        [[1.0, 6.0]],
        [],
    ),
)
def test_red_argcheck_rejects_distance_partitioned_block_ode(distance_ranges):
    config = _valid_contract()
    config["train_options"]["distance_ranges"] = distance_ranges
    with pytest.raises(ValueError, match="distance-partitioned experts"):
        validate_block_ode_contract(config)

    config["train_options"]["distance_ranges"] = [[0.0, 6.0]]
    assert validate_block_ode_contract(config) is None


def test_red_argcheck_rejects_legacy_full_h_names_and_cross_semantic_fields():
    absolute = _valid_contract()
    absolute_fields = {
        "node_block_target_key": "node_full_hamil_target_blocks",
        "edge_block_target_key": "edge_full_hamil_target_blocks",
        "node_block_shape_key": "node_full_hamil_target_block_shape",
        "edge_block_shape_key": "edge_full_hamil_target_block_shape",
    }
    legacy_fields = {
        "node_block_target_key": "node_full_hamil_blocks",
        "edge_block_target_key": "edge_full_hamil_blocks",
        "node_block_shape_key": "node_full_hamil_block_shape",
        "edge_block_shape_key": "edge_full_hamil_block_shape",
    }
    residual_fields = {
        "node_block_target_key": "node_delta_hamil_blocks",
        "edge_block_target_key": "edge_delta_hamil_blocks",
        "node_block_shape_key": "node_delta_hamil_block_shape",
        "edge_block_shape_key": "edge_delta_hamil_block_shape",
    }

    for option in absolute_fields:
        for wrong_value in (legacy_fields[option], residual_fields[option]):
            bad = deepcopy(absolute)
            bad["train_options"]["flow_options"][option] = wrong_value
            with pytest.raises(ValueError, match=option):
                validate_block_ode_contract(bad)

    residual = deepcopy(absolute)
    residual_flow = residual["train_options"]["flow_options"]
    residual_flow["target_semantics"] = "residual_dh"
    residual_flow.update(residual_fields)
    residual_split = residual["data_options"]["train"]
    residual_split["residual_hamiltonian"] = True
    residual_split["require_full_h_target"] = False
    residual_split["require_residual_h_target"] = True
    assert validate_block_ode_contract(residual) is None
    for option, wrong_value in absolute_fields.items():
        bad = deepcopy(residual)
        bad["train_options"]["flow_options"][option] = wrong_value
        with pytest.raises(ValueError, match=option):
            validate_block_ode_contract(bad)


def test_red_argcheck_requires_physical_h0_contract_on_every_split():
    base = _valid_contract()
    base["data_options"]["validation"] = deepcopy(base["data_options"]["train"])
    assert validate_block_ode_contract(base) is None

    for split in ("train", "validation"):
        for option in ("get_Hamiltonian", "get_H0", "require_full_h_target"):
            bad = deepcopy(base)
            bad["data_options"][split].pop(option)
            with pytest.raises(ValueError, match=rf"data_options\.{split}\.{option}"):
                validate_block_ode_contract(bad)

    for unsupported_type in ("DefaultDataset", "HDF5Dataset"):
        bad = deepcopy(base)
        bad["data_options"]["validation"]["type"] = unsupported_type
        with pytest.raises(ValueError, match=r"data_options\.validation\.type"):
            validate_block_ode_contract(bad)

    bad = deepcopy(base)
    bad["train_options"]["flow_options"]["strict_h0"] = False
    with pytest.raises(ValueError, match="strict_h0=true"):
        validate_block_ode_contract(bad)

    bad = deepcopy(base)
    bad["data_options"].pop("train")
    with pytest.raises(ValueError, match="configured data_options.train"):
        validate_block_ode_contract(bad)

    bad = deepcopy(base)
    bad["common_options"]["has_soc"] = True
    with pytest.raises(ValueError, match="non-SOC only"):
        validate_block_ode_contract(bad)

    residual = deepcopy(base)
    residual_flow = residual["train_options"]["flow_options"]
    residual_flow["target_semantics"] = "residual_dh"
    residual_flow.update({
        "node_block_target_key": "node_delta_hamil_blocks",
        "edge_block_target_key": "edge_delta_hamil_blocks",
        "node_block_shape_key": "node_delta_hamil_block_shape",
        "edge_block_shape_key": "edge_delta_hamil_block_shape",
    })
    for split_options in residual["data_options"].values():
        split_options["residual_hamiltonian"] = True
        split_options["require_full_h_target"] = False
        split_options["require_residual_h_target"] = True
    assert validate_block_ode_contract(residual) is None

    for split in ("train", "validation"):
        bad = deepcopy(residual)
        bad["data_options"][split].pop("require_residual_h_target")
        with pytest.raises(
            ValueError,
            match=rf"data_options\.{split}\.require_residual_h_target",
        ):
            validate_block_ode_contract(bad)
    residual["data_options"]["validation"]["require_full_h_target"] = True
    with pytest.raises(ValueError, match="require_full_h_target must be false"):
        validate_block_ode_contract(residual)


@pytest.mark.parametrize(
    ("dtype", "maximum_atol"),
    (("float32", 2.0e-5), ("float64", 1.0e-10)),
)
def test_red_argcheck_caps_strict_inverse_tolerance_by_dtype(dtype, maximum_atol):
    base = _valid_contract()
    base["common_options"]["dtype"] = dtype
    base["train_options"]["flow_options"]["block_inverse_atol"] = maximum_atol
    assert validate_block_ode_contract(base) is None

    for invalid_atol in (maximum_atol * (1.0 + 1.0e-6), 1.0e6):
        bad = deepcopy(base)
        bad["train_options"]["flow_options"]["block_inverse_atol"] = invalid_atol
        with pytest.raises(ValueError, match=rf"{dtype} maximum"):
            validate_block_ode_contract(bad)

    bad_dtype = deepcopy(base)
    bad_dtype["common_options"]["dtype"] = "bfloat16"
    with pytest.raises(ValueError, match="dtype='float32' or 'float64'"):
        validate_block_ode_contract(bad_dtype)


def test_red_fractional_or_empty_steps_and_nan_inverse_atol_fail_closed():
    idp, data, _, h0 = _case()
    base = _flow(idp).options
    for bad_steps in ([1.5], []):
        options = dict(base)
        options["validation_ode_steps"] = bad_steps
        with pytest.raises(ValueError, match="validation_ode_steps"):
            HamiltonianCFM(options, idp=idp, dtype=torch.float64)

    options = dict(base)
    options["block_inverse_atol"] = float("nan")
    with pytest.raises(ValueError, match="block_inverse_atol"):
        HamiltonianCFM(options, idp=idp, dtype=torch.float64)

    options = dict(base)
    options["block_ode"] = False
    with pytest.raises(ValueError, match="distinct mode"):
        HamiltonianCFM(options, idp=idp, dtype=torch.float64)

    with pytest.raises(ValueError, match="finite and non-negative"):
        canonical_block_tensors_to_feature_tensors(
            data,
            idp,
            node_blocks=h0.node_blocks,
            edge_blocks=h0.edge_blocks,
            node_shapes=h0.node_shapes,
            edge_shapes=h0.edge_shapes,
            atol=float("nan"),
        )

    config = _valid_contract()
    config["train_options"]["flow_options"]["validation_ode_steps"] = [1.5]
    with pytest.raises(ValueError, match="validation_ode_steps"):
        validate_block_ode_contract(config)
