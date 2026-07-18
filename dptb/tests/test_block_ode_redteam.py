from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest
import torch
import yaml

from dptb.data.interfaces.blockwise_tensor import (
    BlockTensorResult,
    block_tensors_to_feature_tensors,
)
from dptb.data.transforms import OrbitalMapper
from dptb.nn.hamiltonian import (
    E3Hamiltonian,
    _contract_cg_rme,
    _inverse_contract_cg_hr,
)
from dptb.nnops.block_flow_codec import BlockStateCodec, project_block_state
from dptb.nnops.flow import HamiltonianCFM
from dptb.tests.test_block_flow_codec import _mixed_case, _random_state
from dptb.tests.test_block_ode_flow import (
    EndpointSequence,
    _case,
    _flow,
    _fresh,
    _scaled_endpoint,
)
from dptb.utils.argcheck import slem_h0, validate_block_ode_contract


FP64_ATOL = 1e-10


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
                "block_inverse_mode": "strict",
                "validation_ode_steps": [1, 3],
            }
        },
        "model_options": {
            "embedding": {
                "use_flow_time_embedding": True,
                "flow_time_condition_edges": True,
                "flow_time_allow_missing": False,
                "flow_time_key": "flow_time",
                "h0_merge_mode": "replace",
                "use_h0_node_init": True,
                "use_h0_edge_init": True,
            },
            "prediction": {"add_h0": False},
        },
        "data_options": {
            "train": {
                "get_Hamiltonian": True,
                "residual_hamiltonian": False,
            }
        },
    }


def test_red_cg_exhaustive_standard_basis_includes_every_f_pairtype():
    idp = OrbitalMapper(
        {"X": ["1s", "2p", "3d", "4f"]}, method="e3tb", device="cpu"
    )
    module = E3Hamiltonian(idp=idp, dtype=torch.float64)
    saw_f = False
    for pairtype, basis in module.cgbasis.items():
        matrix = basis.reshape(-1, basis.shape[-1])
        size = matrix.shape[0]
        assert matrix.shape == (size, size), pairtype
        assert torch.linalg.matrix_rank(matrix, atol=1e-12, rtol=1e-12) == size
        assert torch.linalg.cond(matrix).item() <= 1.0 + FP64_ATOL

        # One row per standard basis vector, so every coupled coordinate of
        # every supported pairtype is attacked independently.
        rme = torch.eye(size, dtype=torch.float64).unsqueeze(-1)
        product = _contract_cg_rme(basis, rme)
        restored = _inverse_contract_cg_hr(basis, product)
        assert (restored - rme).abs().max().item() <= FP64_ATOL, pairtype
        saw_f |= "f" in pairtype
    assert saw_f


def test_red_inverse_cg_nonorthogonal_solve_and_singular_fail_closed():
    nonorthogonal = torch.tensor([[[2.0, 0.0], [0.0, 3.0]]], dtype=torch.float64)
    rme = torch.tensor([[[1.25], [-0.75]]], dtype=torch.float64)
    product = _contract_cg_rme(nonorthogonal, rme)
    restored = _inverse_contract_cg_hr(nonorthogonal, product)
    assert torch.linalg.cond(nonorthogonal.reshape(2, 2)).item() == pytest.approx(1.5)
    torch.testing.assert_close(restored, rme, rtol=0.0, atol=1e-12)

    singular = torch.tensor([[[1.0, 0.0], [0.0, 0.0]]], dtype=torch.float64)
    with pytest.raises(RuntimeError, match="singular/rank deficient"):
        _inverse_contract_cg_hr(singular, product)


def test_red_codec_entry_rejects_missing_and_duplicate_reverse_edges():
    idp, data = _mixed_case()
    codec = BlockStateCodec(idp, dtype=torch.float64)
    state = project_block_state(data, idp, _random_state(idp, data))

    missing_data = dict(data)
    for key in ("edge_index", "edge_cell_shift", "edge_types"):
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
    duplicate_data["edge_types"] = torch.cat(
        [data["edge_types"], data["edge_types"][:1]], dim=0
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
        "edge_types": edge_types,
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


def test_red_n1_prior_zero_matches_adapter_at_strict_fp64_threshold():
    idp, data, codec, _ = _case()
    endpoint = _scaled_endpoint(codec, data, data["node_h0"], data["edge_h0"], 1.7)
    block_model = EndpointSequence([endpoint])
    adapter_model = EndpointSequence([endpoint])
    block_result = _flow(idp).sample(block_model, _fresh(data), num_steps=1)
    adapter = HamiltonianCFM(
        {
            "enabled": True,
            "mode": "residual",
            "prior": "zero",
            "output_space": "ao_block",
            "validation_ode_steps": [1],
        },
        idp=idp,
        dtype=torch.float64,
    )
    adapter_result = adapter.sample(adapter_model, _fresh(data), num_steps=1)
    torch.testing.assert_close(block_model.inputs[0][0], adapter_model.inputs[0][0], rtol=0.0, atol=1e-12)
    torch.testing.assert_close(block_model.inputs[0][1], adapter_model.inputs[0][1], rtol=0.0, atol=1e-12)
    torch.testing.assert_close(
        block_result["node_hamil_blocks"], adapter_result["node_hamil_blocks"], rtol=0.0, atol=1e-12
    )
    torch.testing.assert_close(
        block_result["edge_hamil_blocks"], adapter_result["edge_hamil_blocks"], rtol=0.0, atol=1e-12
    )


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
    assert "flow_time_allow_missing" in names


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
    for filename in ("h_b0_block_ode_water.yaml", "h_b0_block_ode_crystal.yaml"):
        overlay = yaml.safe_load(
            (Path("configs") / filename).read_text(encoding="utf-8")
        )
        assert validate_block_ode_contract(overlay) is None

    bad_cases = []
    bad = deepcopy(base)
    bad["train_options"]["flow_options"]["prior"] = "te"
    bad_cases.append((bad, "prior='zero'"))
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
    bad_cases.append((bad, "conflicts"))

    for bad, match in bad_cases:
        with pytest.raises(ValueError, match=match):
            validate_block_ode_contract(bad)


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
        block_tensors_to_feature_tensors(
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
