"""LemPair: dual-cutoff contract, pair refinement, block-ODE edge coverage, batching, gradients, lifecycle."""
from __future__ import annotations

import copy
import math
import re
import weakref
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml
from dargs.dargs import ArgumentKeyError
from e3nn import o3

from dptb.data import _keys
from dptb.nn.embedding.lem_moe_v3_h0 import LemMoEV3H0
from dptb.nn.embedding.lem_pair import (
    LemPair,
    PairInitLayer,
    PairLayer,
    _canonicalize_mp_cutoff,
    _get_mp_edge_mask,
    load_lem_h0_backbone,
)
from dptb.utils.argcheck import model_options as model_options_argcheck

from dptb.tests.block_ode_fixtures import _rotate_canvas_blocks
from dptb.tests.pair_helpers import (
    ao_wigner,
    batch_graphs,
    block_ode_model,
    clone_data,
    deterministic_fp64,
    edge_block_drift,
    fp64_default,
    model,
    model_options,
    molecule_data,
    prepared_flow_batch,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_KEYS = (_keys.NODE_HAMILTONIAN_KEY, _keys.EDGE_HAMILTONIAN_KEY, _keys.EDGE_OVERLAP_KEY)
WEIGHT_MODES = ("full", "per_path", "qhflow")
# Edge lengths with mp_cutoff=1.0: every edge is a message-passing edge, or a real MP/head split.
ALL_ACTIVE_POSITIONS = [[0.0, 0.0, 0.0], [0.30, 0.0, 0.0], [0.0, 0.35, 0.0]]
REAL_SPLIT_POSITIONS = [[0.0, 0.0, 0.0], [0.70, 0.0, 0.0], [2.10, 0.2, 0.0]]


class _TwoSpeciesMapper:
    basis = {"H": "1s", "C": "1s1p"}
    bond_to_type = {"H-H": 0, "H-C": 1, "C-H": 2, "C-C": 3}


def _repo_file(*parts):
    path = REPO_ROOT.joinpath(*parts)
    if not path.is_file():
        pytest.skip(f"needs the repository checkout ({'/'.join(parts)})")
    return path


def _edge_lengths(data):
    src, dst = data[_keys.EDGE_INDEX_KEY]
    positions = data[_keys.POSITIONS_KEY]
    return (positions.index_select(0, src) - positions.index_select(0, dst)).norm(dim=-1)


def _outputs(pair_model, data):
    with fp64_default(), torch.no_grad():
        result = pair_model(clone_data(data))
    return {key: result[key].detach().clone() for key in OUTPUT_KEYS}


def _assert_same_outputs(reference, actual):
    for key in OUTPUT_KEYS:
        assert torch.equal(reference[key], actual[key]), key


@pytest.fixture(scope="module")
def plain_model():
    with fp64_default():
        return model()


@pytest.fixture(scope="module")
def dual_model():
    with fp64_default():
        return model(mp_cutoff=1.0)


# --- configuration contract -------------------------------------------------


@pytest.mark.parametrize(
    ("mp_cutoff", "r_max", "expected"),
    [
        (6.0, 5.0, None),
        ({"H": 5.0, "C": 5.0}, 5.0, None),
        (6.0, {"H": 4.0, "C": 6.0}, None),
        ({"H": 4.0, "C": 6.0}, {"H": 3.0, "C": 5.0}, None),
        ({"H": 4.0, "C": 6.0}, {"H": 3.0}, {"H": 4.0, "C": 6.0}),
        (1.0, 5.0, 1.0),
    ],
    ids=["scalar", "dict", "dict_r_max", "both_dicts", "unprovable_stays_dual", "real_split"],
)
def test_redundant_mp_cutoff_canonicalizes_to_single_cutoff(mp_cutoff, r_max, expected):
    assert _canonicalize_mp_cutoff(mp_cutoff, r_max, _TwoSpeciesMapper()) == expected


@pytest.mark.parametrize(
    ("mp_cutoff", "error"),
    [
        (0.0, ValueError),
        (-1.0, ValueError),
        (math.nan, ValueError),
        (math.inf, ValueError),
        (True, TypeError),
        ({"H": 1.0}, ValueError),
        ({"H": 1.0, "C": 1.0, "Xe": 1.0}, ValueError),
    ],
    ids=["zero", "negative", "nan", "inf", "boolean", "missing_species", "unknown_species"],
)
def test_invalid_mp_cutoff_is_rejected(mp_cutoff, error):
    with pytest.raises(error, match="mp_cutoff"):
        _canonicalize_mp_cutoff(mp_cutoff, 5.0, _TwoSpeciesMapper())


@pytest.mark.parametrize(
    ("options", "error", "key"),
    [
        ({"mp_cutoff": 0.0}, ValueError, "mp_cutoff"),
        ({"mp_avg_num_neighbors": 0.0}, ValueError, "mp_avg_num_neighbors"),
        ({"mp_avg_num_neighbors": True}, TypeError, "mp_avg_num_neighbors"),
        (
            {"res_update_additive": True, "res_update_ratios_learnable": True},
            ValueError,
            "res_update_ratios_learnable",
        ),
    ],
    ids=["zero_cutoff", "zero_degree", "boolean_degree", "additive_with_learnable_ratios"],
)
def test_constructor_rejects_invalid_options(options, error, key):
    with fp64_default(), pytest.raises(error, match=key):
        LemPair(**{**model_options(), **options})


@pytest.mark.parametrize(
    ("bond_to_type", "mp_cutoff", "error", "key"),
    [
        ({"H-C-extra": 0}, {"H": 1.0, "C": 1.0}, ValueError, "H-C-extra"),
        ({"H-C": 0}, {"H": 1.0}, KeyError, "H-C"),
    ],
    ids=["malformed_bond", "missing_element"],
)
def test_runtime_mp_mask_rejects_malformed_bond_maps(bond_to_type, mp_cutoff, error, key):
    mapper = SimpleNamespace(bond_to_type=bond_to_type)
    with pytest.raises(error, match=key):
        _get_mp_edge_mask(torch.tensor([0.5]), torch.tensor([0]), mapper, mp_cutoff)


@pytest.mark.parametrize(
    ("method", "accepted"),
    [("lem_moe_v3_h0", False), ("lem_pair", True), ("lem_cutoff", True)],
)
def test_strict_argcheck_accepts_mp_cutoff_only_for_its_consumers(method, accepted):
    config = _repo_file("configs", "route_h_b0_late_block_expansion_cg.yaml")
    payload = yaml.safe_load(config.read_text(encoding="utf-8"))["model_options"]
    payload["embedding"].update(method=method, mp_cutoff=1.0)
    argument = model_options_argcheck()
    normalized = argument.normalize_value(payload)
    if accepted:
        argument.check_value(normalized, strict=True)
    else:
        with pytest.raises(ArgumentKeyError, match="mp_cutoff"):
            argument.check_value(normalized, strict=True)


def test_legacy_backbone_migration_restores_weights_and_rejects_tampered_state():
    options = model_options()
    options.pop("mp_avg_num_neighbors")
    allowed = (
        "dual_cutoff_readout_normalization",
        "dual_cutoff_pair_readout.",
        "dual_cutoff_edge_context_projection.",
        "pair_refine.",
    )
    with fp64_default():
        torch.manual_seed(20260723)
        legacy = LemMoEV3H0(**options).eval()
        torch.manual_seed(20260724)
        dual = LemPair(
            **options,
            mp_cutoff=1.0,
            mp_avg_num_neighbors=1.5,
            pair_refine_enable=True,
            pair_refine_rank=4,
            pair_refine_init=0.1,
        ).eval()

    legacy_state = legacy.state_dict()
    assert load_lem_h0_backbone(dual, legacy_state, allowed_missing_prefixes=allowed).missing_keys
    restored = dual.state_dict()
    for key, value in legacy_state.items():
        assert torch.equal(restored[key], value), key

    with pytest.raises(RuntimeError, match="pair_refine"):
        load_lem_h0_backbone(dual, legacy_state, allowed_missing_prefixes=allowed[:-1])

    tampered = legacy.state_dict()
    renamed = next(iter(tampered))
    tampered[renamed + "_typo"] = tampered.pop(renamed)
    with pytest.raises(RuntimeError, match=re.escape(renamed + "_typo")):
        load_lem_h0_backbone(dual, tampered, allowed_missing_prefixes=allowed)


# --- dual cutoff ------------------------------------------------------------


def test_disabled_and_provably_redundant_cutoff_are_bit_identical(plain_model):
    with fp64_default():
        redundant = model(mp_cutoff=1.0e9)
    redundant.load_state_dict(plain_model.state_dict(), strict=True)
    data = molecule_data(plain_model)
    _assert_same_outputs(_outputs(plain_model, data), _outputs(redundant, data))


def test_dual_cutoff_model_is_equivariant_with_a_real_mp_split(dual_model):
    data = molecule_data(dual_model)
    lengths = _edge_lengths(data)
    assert bool((lengths < 1.0).any()) and bool((lengths >= 1.0).any())
    with fp64_default():
        torch.manual_seed(17)
        drift, _ = edge_block_drift(dual_model, data, o3.rand_matrix(dtype=torch.float64))
    assert drift <= 1.0e-9


def test_non_mp_edge_block_responds_to_its_own_h0(dual_model):
    data = molecule_data(dual_model)
    non_mp_edge = int(torch.nonzero(_edge_lengths(data) >= 1.0)[0])
    perturbed = clone_data(data)
    h0_dim = perturbed[_keys.EDGE_H0_KEY].shape[-1]
    perturbed[_keys.EDGE_H0_KEY][non_mp_edge] = torch.linspace(
        0.1, 0.1 * h0_dim, h0_dim, dtype=torch.float64
    )
    reference = _outputs(dual_model, data)[_keys.EDGE_HAMILTONIAN_KEY][non_mp_edge]
    changed = _outputs(dual_model, perturbed)[_keys.EDGE_HAMILTONIAN_KEY][non_mp_edge]
    assert (changed - reference).abs().max().item() > 0.0


# --- pair refinement --------------------------------------------------------


def test_default_lem_pair_is_bit_exact_superset_of_lem_moe_v3_h0():
    options = model_options()
    options.pop("mp_avg_num_neighbors")
    with fp64_default():
        torch.manual_seed(20260723)
        legacy = LemMoEV3H0(**options).eval()
        legacy_rng = torch.random.get_rng_state().clone()
        torch.manual_seed(20260723)
        pair = LemPair(**options).eval()
        assert torch.equal(legacy_rng, torch.random.get_rng_state())
    assert legacy.state_dict().keys() == pair.state_dict().keys()
    for key, value in legacy.state_dict().items():
        assert torch.equal(value, pair.state_dict()[key]), key
    data = molecule_data(pair)
    _assert_same_outputs(_outputs(legacy, data), _outputs(pair, data))


@pytest.mark.parametrize("weight_mode", WEIGHT_MODES)
def test_pair_refine_changes_the_output_and_stays_equivariant(plain_model, weight_mode):
    with fp64_default():
        refined = model(pair_refine_enable=True, pair_refine_weight_mode=weight_mode)
        data = molecule_data(refined)
        torch.manual_seed(23)
        drift, reference = edge_block_drift(refined, data, o3.rand_matrix(dtype=torch.float64))
    assert drift <= 1.0e-9
    unrefined = _outputs(plain_model, data)
    assert not torch.equal(
        reference[_keys.EDGE_HAMILTONIAN_KEY], unrefined[_keys.EDGE_HAMILTONIAN_KEY]
    )


def test_additive_residual_without_latent_layernorm_is_equivariant_and_takes_effect(plain_model):
    with fp64_default():
        additive = model(res_update_additive=True, latents_layernorm=False)
        data = molecule_data(additive)
        torch.manual_seed(29)
        drift, reference = edge_block_drift(additive, data, o3.rand_matrix(dtype=torch.float64))
    assert drift <= 1.0e-9
    assert torch.isfinite(reference[_keys.EDGE_HAMILTONIAN_KEY]).all()
    assert not torch.equal(
        reference[_keys.EDGE_HAMILTONIAN_KEY],
        _outputs(plain_model, data)[_keys.EDGE_HAMILTONIAN_KEY],
    )


@pytest.mark.parametrize("weight_mode", WEIGHT_MODES)
def test_identity_initialized_pair_refine_is_bit_exact(plain_model, weight_mode):
    with fp64_default():
        identity = model(
            pair_refine_enable=True,
            pair_refine_weight_mode=weight_mode,
            pair_refine_init=0.2,
            pair_refine_identity_init=True,
        )
    data = molecule_data(plain_model)
    _assert_same_outputs(_outputs(plain_model, data), _outputs(identity, data))


# --- block-ODE edge coverage -------------------------------------------------


@pytest.fixture(scope="module")
def refine_block_ode():
    with fp64_default():
        torch.manual_seed(20260723)
        pair_model = block_ode_model(
            pair_refine_enable=True, pair_refine_rank=4, pair_refine_init=0.1
        )
        _, _, model_data = prepared_flow_batch(pair_model)
    return pair_model, model_data


def test_block_ode_pair_refine_runs_on_the_ordered_full_edge_set(refine_block_ode):
    pair_model, model_data = refine_block_ode
    with fp64_default(), torch.no_grad():
        output = pair_model(clone_data(model_data))
    for key in (*OUTPUT_KEYS[:2], "node_hamil_blocks", "edge_hamil_blocks"):
        assert torch.isfinite(output[key]).all(), key


@pytest.mark.parametrize(
    ("corruption", "key"),
    [("subset", "active-edge"), ("reordered", "active-edge"), ("zero_cutoff", "cutoff coefficient")],
)
def test_block_ode_rejects_incomplete_or_reordered_head_edges(refine_block_ode, corruption, key):
    pair_model, model_data = refine_block_ode
    n_edges = int(model_data[_keys.EDGE_INDEX_KEY].shape[1])
    active = torch.arange(n_edges)
    coefficients = torch.ones(n_edges, dtype=torch.float64)
    if corruption == "subset":
        active = active[:-1]
    elif corruption == "reordered":
        active = active.flip(0)
    else:
        coefficients[0] = 0.0
    data = clone_data(model_data)
    data[_keys.LEM_ACTIVE_EDGES_KEY] = active
    data[_keys.LEM_CUTOFF_COEFFS_KEY] = coefficients
    with fp64_default(), torch.no_grad(), pytest.raises(ValueError, match=key):
        pair_model(data)


def test_dual_block_ode_edge_block_responds_to_non_mp_h0_and_residual_state():
    with deterministic_fp64():
        torch.manual_seed(20260724)
        pair_model = block_ode_model(mp_cutoff=0.5, mp_avg_num_neighbors=1.0)
        _, _, model_data = prepared_flow_batch(pair_model)
        row = 0
        assert _edge_lengths(model_data)[row].item() > pair_model.embedding.mp_cutoff

        data = clone_data(model_data)
        edge_h0 = data[_keys.EDGE_H0_KEY].requires_grad_(True)
        residual = data[_keys.EDGE_SPATIAL_RESIDUAL_BLOCKS_KEY].requires_grad_(True)
        block = pair_model(data)[_keys.EDGE_HAMILTONIAN_KEY][row]
        torch.manual_seed(71)
        grad_h0, grad_residual = torch.autograd.grad(
            (block * torch.randn_like(block)).sum(), (edge_h0, residual)
        )
        assert grad_h0[row].norm().item() > 0.0
        assert grad_residual[row].norm().item() > 0.0

        h0_dim = model_data[_keys.EDGE_H0_KEY].shape[-1]
        for state_key, delta in (
            (_keys.EDGE_H0_KEY, torch.linspace(0.01, 0.01 * h0_dim, h0_dim, dtype=torch.float64)),
            (_keys.EDGE_SPATIAL_RESIDUAL_BLOCKS_KEY, torch.linspace(0.01, 0.16, 16, dtype=torch.float64).reshape(4, 4)),
        ):
            perturbed = clone_data(model_data)
            perturbed[state_key][row] += delta
            with torch.no_grad():
                changed = pair_model(perturbed)[_keys.EDGE_HAMILTONIAN_KEY][row]
            assert not torch.equal(changed, block.detach()), state_key

        # Residual block-state covariance with H0 held at the valid zero tensor.
        state = clone_data(model_data)
        state[_keys.NODE_H0_KEY].zero_()
        state[_keys.EDGE_H0_KEY].zero_()
        torch.manual_seed(17)
        rotation = o3.rand_matrix(dtype=torch.float64)
        d_ao = ao_wigner(pair_model.embedding, rotation)
        rotated = clone_data(state)
        rotated[_keys.POSITIONS_KEY] = state[_keys.POSITIONS_KEY] @ rotation.T
        for state_key in (_keys.NODE_SPATIAL_RESIDUAL_BLOCKS_KEY, _keys.EDGE_SPATIAL_RESIDUAL_BLOCKS_KEY):
            rotated[state_key] = _rotate_canvas_blocks(state[state_key], d_ao)
        with torch.no_grad():
            expected = _rotate_canvas_blocks(pair_model(clone_data(state))[_keys.EDGE_HAMILTONIAN_KEY], d_ao)
            actual = pair_model(rotated)[_keys.EDGE_HAMILTONIAN_KEY]
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=1.0e-9)


# --- batching and gradients --------------------------------------------------


def test_output_does_not_depend_on_batch_partition(dual_model):
    graph_a = molecule_data(dual_model, ALL_ACTIVE_POSITIONS)
    graph_b = molecule_data(dual_model, REAL_SPLIT_POSITIONS)
    with deterministic_fp64(), torch.no_grad():
        standalone = dual_model(clone_data(graph_a))
        ab = dual_model(batch_graphs([graph_a, graph_b]))
        ba = dual_model(batch_graphs([graph_b, graph_a]))
    n_b = int(graph_b[_keys.POSITIONS_KEY].shape[0])
    e_b = int(graph_b[_keys.EDGE_INDEX_KEY].shape[1])
    for key, offset in (
        (_keys.NODE_HAMILTONIAN_KEY, n_b),
        (_keys.EDGE_HAMILTONIAN_KEY, e_b),
        (_keys.EDGE_OVERLAP_KEY, e_b),
    ):
        count = standalone[key].shape[0]
        torch.testing.assert_close(ab[key][:count], standalone[key], rtol=0.0, atol=1.0e-12)
        torch.testing.assert_close(ba[key][offset:offset + count], standalone[key], rtol=0.0, atol=1.0e-12)


@pytest.mark.parametrize(
    "positions", [ALL_ACTIVE_POSITIONS, REAL_SPLIT_POSITIONS], ids=["all_active", "real_split"]
)
def test_every_trainable_parameter_receives_a_gradient(dual_model, positions):
    dual_model.zero_grad(set_to_none=True)
    with deterministic_fp64():
        output = dual_model(molecule_data(dual_model, positions))
        sum(output[key].square().sum() for key in OUTPUT_KEYS).backward()
    missing = [
        name
        for name, parameter in dual_model.named_parameters()
        if parameter.requires_grad and parameter.grad is None
    ]
    dual_model.zero_grad(set_to_none=True)
    assert not missing


@pytest.fixture(scope="module")
def short_cutoff_model():
    with fp64_default():
        return model(mp_cutoff=0.2)


@pytest.mark.parametrize("layout", ["single_empty", "mixed", "all_empty"])
def test_empty_mp_subgraphs_are_finite_and_backward_safe(short_cutoff_model, layout):
    empty = molecule_data(short_cutoff_model, [[0.0, 0.0, 0.0], [1.20, 0.0, 0.0], [0.0, 1.30, 0.0]])
    active = molecule_data(short_cutoff_model, [[0.0, 0.0, 0.0], [0.08, 0.0, 0.0], [0.0, 0.09, 0.0]])
    data = {
        "single_empty": empty,
        "mixed": batch_graphs([empty, active]),
        "all_empty": batch_graphs([empty, empty]),
    }[layout]
    short_cutoff_model.zero_grad(set_to_none=True)
    with deterministic_fp64():
        output = short_cutoff_model(clone_data(data))
        sum(output[key].square().sum() for key in OUTPUT_KEYS).backward()
    for key in OUTPUT_KEYS:
        assert torch.isfinite(output[key]).all(), key
    for name, parameter in short_cutoff_model.named_parameters():
        assert parameter.grad is None or torch.isfinite(parameter.grad).all(), name
    short_cutoff_model.zero_grad(set_to_none=True)


# --- lifecycle ----------------------------------------------------------------


@pytest.fixture(scope="module")
def lifecycle_model():
    with fp64_default():
        return model(mp_cutoff=1.0, pair_refine_enable=True)


def _assert_no_reference_to_original(value, original_ids, visited):
    value_id = id(value)
    if value_id in visited:
        return
    visited.add(value_id)
    if isinstance(value, weakref.ReferenceType):
        target = value()
        assert target is None or id(target) not in original_ids
        return
    if isinstance(value, torch.nn.Module):
        assert value_id not in original_ids
        for nested in vars(value).values():
            _assert_no_reference_to_original(nested, original_ids, visited)
        return
    if isinstance(value, dict):
        for nested in value.values():
            _assert_no_reference_to_original(nested, original_ids, visited)
        return
    if isinstance(value, (tuple, list)):
        for nested in value:
            _assert_no_reference_to_original(nested, original_ids, visited)


def test_deepcopy_is_independent_and_forward_keeps_no_state(lifecycle_model):
    data = molecule_data(lifecycle_model)
    pair_modules = [
        module
        for module in lifecycle_model.modules()
        if isinstance(module, (PairInitLayer, PairLayer))
    ]
    attribute_keys = [set(vars(module)) for module in pair_modules]
    buffer_values = [
        {
            name: value.detach().clone()
            for name, value in module.named_buffers(recurse=False)
            if value is not None
        }
        for module in pair_modules
    ]

    clone = copy.deepcopy(lifecycle_model)
    reference = _outputs(lifecycle_model, data)
    _assert_same_outputs(reference, _outputs(clone, data))

    for module, keys, buffers in zip(pair_modules, attribute_keys, buffer_values):
        assert set(vars(module)) == keys
        for name, value in buffers.items():
            assert torch.equal(module.get_buffer(name), value)

    original_ids = {
        id(value)
        for module in lifecycle_model.modules()
        for value in (module, *module.parameters(recurse=False), *module.buffers(recurse=False))
    }
    _assert_no_reference_to_original(clone, original_ids, set())

    original_parameter = next(lifecycle_model.parameters()).detach().clone()
    with torch.no_grad():
        next(clone.parameters()).add_(1.0)
    assert torch.equal(next(lifecycle_model.parameters()), original_parameter)
    _assert_same_outputs(reference, _outputs(lifecycle_model, data))


def test_state_dict_round_trip_is_strict_and_exact(lifecycle_model):
    with fp64_default():
        target = model(seed=20260724, mp_cutoff=1.0, pair_refine_enable=True)
    target.load_state_dict(lifecycle_model.state_dict(), strict=True)
    data = molecule_data(lifecycle_model)
    _assert_same_outputs(_outputs(lifecycle_model, data), _outputs(target, data))


def test_whole_model_torch_save_load_round_trip_is_exact(lifecycle_model, tmp_path):
    path = tmp_path / "lem_pair_whole_model.pt"
    data = molecule_data(lifecycle_model)
    reference = _outputs(lifecycle_model, data)
    torch.save(lifecycle_model, path)
    restored = torch.load(path, map_location="cpu", weights_only=False)
    _assert_same_outputs(reference, _outputs(restored, data))
