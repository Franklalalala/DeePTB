"""PairSO3RefineTP in its full, per_path and qhflow weight modes."""
from __future__ import annotations

import pytest
import torch
from dargs.dargs import ArgumentValueError
from e3nn import o3

from dptb.nn.embedding.pair_so3_refine import PairSO3RefineTP
from dptb.utils.argcheck import slem_pair
from dptb.utils.pair_refine_cost import (
    e3nn_fctp_weight_numel,
    estimate,
    fctp_weight_numel,
    parse_irreps,
    validate_weight_numel_with_e3nn,
)

from dptb.tests.pair_helpers import fp64_default

WEIGHT_MODES = ("full", "per_path", "qhflow")
IRREPS = "2x0e+2x1o+2x1e+2x2e"
SMALL_IRREPS = "1x0e+1x1o+1x1e"


def _refine(irreps=IRREPS, **options):
    options.setdefault("dtype", torch.float64)
    return PairSO3RefineTP(irreps, irreps, **options)


def _example_inputs(irreps, *, requires_grad=False):
    dim = o3.Irreps(irreps).dim
    node_features = torch.randn(5, dim, dtype=torch.float64, requires_grad=requires_grad)
    edge_features = torch.randn(7, dim, dtype=torch.float64, requires_grad=requires_grad)
    edge_index = torch.tensor(
        [[0, 0, 1, 2, 3, 4, 4], [1, 2, 3, 4, 0, 1, 3]], dtype=torch.long
    )
    return node_features, edge_features, edge_index


@pytest.mark.parametrize(
    ("irreps", "expected"),
    [
        ("1x0e+1x1o", 4),
        (IRREPS, 192),
        ("4x0e+4x1o+4x1e+4x2e+4x2o+4x3o+4x3e+4x4e", 10496),
    ],
)
def test_dependency_free_weight_count_matches_e3nn(irreps, expected):
    assert fctp_weight_numel(parse_irreps(irreps), parse_irreps(irreps)) == expected
    assert e3nn_fctp_weight_numel(irreps, irreps) == expected
    assert validate_weight_numel_with_e3nn(irreps, irreps) == expected


@pytest.mark.parametrize("weight_mode", WEIGHT_MODES)
def test_cost_estimate_matches_built_module(weight_mode):
    with fp64_default():
        module = _refine(rank=4, weight_mode=weight_mode)
    result = estimate(
        IRREPS, edges=1, rank=4, dtype_bytes=8, internal_weights=True, weight_mode=weight_mode
    )
    qhflow = weight_mode == "qhflow"
    assert result["total_refiner_params"] == sum(p.numel() for p in module.parameters())
    assert result["dynamic_weight_numel_per_edge"] == module.dynamic_up.out_features
    assert result["qhflow_weight_numel_per_edge" if qhflow else "weight_numel_per_edge"] == (
        module.weight_numel
    )
    assert result["qhflow_path_count" if qhflow else "path_count"] == module.n_paths


@pytest.mark.parametrize("weight_mode", WEIGHT_MODES)
def test_max_weight_numel_guard_allows_limit_and_rejects_excess(weight_mode):
    with fp64_default():
        budget = _refine(weight_mode=weight_mode).weight_numel
        assert _refine(weight_mode=weight_mode, max_weight_numel=budget).weight_numel == budget
        with pytest.raises(ValueError, match="max_weight_numel"):
            _refine(weight_mode=weight_mode, max_weight_numel=budget - 1)


@pytest.mark.parametrize("invalid_limit", [-1, True, 191.5, "192"])
def test_max_weight_numel_rejects_invalid_values(invalid_limit):
    with pytest.raises(ValueError, match="max_weight_numel"):
        _refine("1x0e+1x1o", max_weight_numel=invalid_limit)


def test_per_path_requires_internal_static_weights():
    with pytest.raises(ValueError, match="internal_weights"):
        _refine("1x0e+1x1o", weight_mode="per_path", internal_weights=False)


def test_per_path_zero_gate_leaves_only_the_static_tensor_product():
    torch.manual_seed(20260724)
    module = _refine(rank=5, weight_mode="per_path", dynamic_init=0.0)
    node_features, edge_features, edge_index = _example_inputs(IRREPS)
    src, dst = edge_index
    expected = edge_features + module.tensor_product(
        node_features.index_select(0, src),
        node_features.index_select(0, dst),
        module.static_weights.expand(edge_features.shape[0], -1),
    )
    torch.testing.assert_close(
        module(node_features, edge_features, edge_index), expected, rtol=0.0, atol=5.0e-16
    )


@pytest.mark.parametrize("weight_mode", WEIGHT_MODES)
def test_forward_and_backward_are_finite_and_reach_every_parameter(weight_mode):
    with fp64_default():
        torch.manual_seed(17)
        module = _refine(rank=4, weight_mode=weight_mode, dynamic_init=0.1)
        node_features, edge_features, edge_index = _example_inputs(IRREPS, requires_grad=True)
        output = module(node_features, edge_features, edge_index)
        output.square().mean().backward()

    assert torch.isfinite(output).all()
    assert torch.isfinite(node_features.grad).all()
    assert torch.isfinite(edge_features.grad).all()
    for name, parameter in module.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name


@pytest.mark.parametrize("scaled", [False, True], ids=["unscaled", "edge_scale"])
@pytest.mark.parametrize("weight_mode", WEIGHT_MODES)
def test_refinement_is_so3_equivariant(weight_mode, scaled):
    with fp64_default():
        torch.manual_seed(23)
        module = _refine(rank=4, weight_mode=weight_mode, dynamic_init=0.1).eval()
        node_features, edge_features, edge_index = _example_inputs(IRREPS)
        edge_scale = torch.rand(edge_features.shape[0], dtype=torch.float64) if scaled else None
        representation = o3.Irreps(IRREPS).D_from_matrix(o3.rand_matrix(dtype=torch.float64))
        reference = module(node_features, edge_features, edge_index, edge_scale=edge_scale)
        rotated = module(
            node_features @ representation.T,
            edge_features @ representation.T,
            edge_index,
            edge_scale=edge_scale,
        )
    torch.testing.assert_close(rotated, reference @ representation.T, rtol=0.0, atol=1.0e-9)


@pytest.mark.parametrize("identity_init", [False, True])
@pytest.mark.parametrize("weight_mode", WEIGHT_MODES)
def test_identity_initialization_is_exact(weight_mode, identity_init):
    with fp64_default():
        torch.manual_seed(101)
        module = _refine(
            SMALL_IRREPS,
            rank=3,
            weight_mode=weight_mode,
            dynamic_init=0.2,
            identity_init=identity_init,
        )
        node_features, edge_features, edge_index = _example_inputs(SMALL_IRREPS)
        output = module(node_features, edge_features, edge_index)
    assert torch.equal(output, edge_features) is identity_init


@pytest.mark.parametrize("weight_mode", WEIGHT_MODES)
def test_edge_scale_zero_and_one_are_exact(weight_mode):
    with fp64_default():
        torch.manual_seed(303)
        module = _refine(SMALL_IRREPS, rank=3, weight_mode=weight_mode, dynamic_init=0.2)
        node_features, edge_features, edge_index = _example_inputs(SMALL_IRREPS)
        n_edges = edge_features.shape[0]
        reference = module(node_features, edge_features, edge_index)
        zero_scaled = module(
            node_features, edge_features, edge_index,
            edge_scale=torch.zeros(n_edges, dtype=torch.float64),
        )
        one_scaled = module(
            node_features, edge_features, edge_index,
            edge_scale=torch.ones(n_edges, 1, dtype=torch.float64),
        )
    assert torch.equal(zero_scaled, edge_features)
    assert torch.equal(one_scaled, reference)


@pytest.mark.parametrize(
    "edge_scale",
    [
        torch.ones(6, dtype=torch.float64),
        torch.ones(7, 2, dtype=torch.float64),
        torch.ones(1, 7, 1, dtype=torch.float64),
    ],
    ids=["short", "two_columns", "three_dims"],
)
def test_edge_scale_shape_is_validated(edge_scale):
    module = _refine("1x0e+1x1o")
    node_features, edge_features, edge_index = _example_inputs("1x0e+1x1o")
    with pytest.raises(ValueError, match="edge_scale"):
        module(node_features, edge_features, edge_index, edge_scale=edge_scale)


def test_default_weight_mode_is_bit_exact_full_mode():
    with fp64_default():
        torch.manual_seed(20260724)
        default = _refine(SMALL_IRREPS, rank=3, dynamic_init=0.2)
        default_rng = torch.random.get_rng_state().clone()
        torch.manual_seed(20260724)
        explicit = _refine(SMALL_IRREPS, rank=3, weight_mode="full", dynamic_init=0.2)
        assert torch.equal(default_rng, torch.random.get_rng_state())
        node_features, edge_features, edge_index = _example_inputs(SMALL_IRREPS)
        assert default.state_dict().keys() == explicit.state_dict().keys()
        for key, value in default.state_dict().items():
            assert torch.equal(value, explicit.state_dict()[key]), key
        assert torch.equal(
            default(node_features, edge_features, edge_index),
            explicit(node_features, edge_features, edge_index),
        )


@pytest.mark.parametrize(
    ("value", "accepted"),
    [("full", True), ("per_path", True), ("qhflow", True), ("diagonal", False)],
)
def test_weight_mode_argcheck(value, accepted):
    argument = next(item for item in slem_pair() if item.name == "pair_refine_weight_mode")
    if accepted:
        argument.check({"pair_refine_weight_mode": value}, strict=True)
    else:
        with pytest.raises(ArgumentValueError, match="pair_refine_weight_mode"):
            argument.check({"pair_refine_weight_mode": value}, strict=True)
