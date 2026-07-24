from __future__ import annotations

from contextlib import contextmanager

import pytest
import torch
from e3nn import o3

from dptb.nn.embedding.pair_so3_refine import PairSO3RefineTP
from dptb.utils.pair_refine_cost import (
    e3nn_fctp_weight_numel,
    fctp_weight_numel,
    parse_irreps,
    validate_weight_numel_with_e3nn,
)


@contextmanager
def _fp64_default():
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        yield
    finally:
        torch.set_default_dtype(previous)


def _example_inputs(
    irreps: o3.Irreps,
    *,
    requires_grad: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    node_features = torch.randn(
        5,
        irreps.dim,
        dtype=torch.float64,
        requires_grad=requires_grad,
    )
    edge_features = torch.randn(
        7,
        irreps.dim,
        dtype=torch.float64,
        requires_grad=requires_grad,
    )
    edge_index = torch.tensor(
        [[0, 0, 1, 2, 3, 4, 4], [1, 2, 3, 4, 0, 1, 3]],
        dtype=torch.long,
    )
    return node_features, edge_features, edge_index


@pytest.mark.parametrize(
    ("node_irreps", "edge_irreps", "expected"),
    [
        ("1x0e+1x1o", "1x0e+1x1o", 4),
        ("2x0e+2x1o+2x1e+2x2e", "2x0e+2x1o+2x1e+2x2e", 192),
        (
            "4x0e+4x1o+4x1e+4x2e+4x2o+4x3o+4x3e+4x4e",
            "4x0e+4x1o+4x1e+4x2e+4x2o+4x3o+4x3e+4x4e",
            10496,
        ),
    ],
)
def test_dependency_free_weight_count_matches_e3nn(
    node_irreps: str,
    edge_irreps: str,
    expected: int,
):
    independent = fctp_weight_numel(
        parse_irreps(node_irreps),
        parse_irreps(edge_irreps),
    )
    assert independent == expected
    assert e3nn_fctp_weight_numel(node_irreps, edge_irreps) == expected
    assert validate_weight_numel_with_e3nn(node_irreps, edge_irreps) == expected


def test_max_weight_numel_guard_allows_equal_and_rejects_excess():
    irreps = o3.Irreps("2x0e+2x1o+2x1e+2x2e")
    allowed = PairSO3RefineTP(irreps, irreps, max_weight_numel=192)
    assert allowed.weight_numel == 192

    with pytest.raises(
        ValueError,
        match=(
            r"actual=192, limit=191.*"
            r"dptb/utils/pair_refine_cost\.py"
        ),
    ):
        PairSO3RefineTP(irreps, irreps, max_weight_numel=191)


@pytest.mark.parametrize("invalid_limit", [-1, True, 191.5, "192"])
def test_max_weight_numel_guard_rejects_invalid_types(invalid_limit):
    with pytest.raises(
        ValueError,
        match=r"max_weight_numel must be a non-negative integer or None",
    ):
        PairSO3RefineTP(
            "1x0e+1x1o",
            "1x0e+1x1o",
            max_weight_numel=invalid_limit,
        )


def test_per_path_rejects_missing_static_weights():
    with pytest.raises(
        ValueError,
        match=r"weight_mode='per_path' requires internal_weights=True",
    ):
        PairSO3RefineTP(
            "1x0e+1x1o",
            "1x0e+1x1o",
            weight_mode="per_path",
            internal_weights=False,
        )


def test_per_path_dynamic_dof_and_zero_gate_static_semantics():
    torch.manual_seed(20260724)
    irreps = o3.Irreps("2x0e+2x1o+2x1e+2x2e")
    module = PairSO3RefineTP(
        irreps,
        irreps,
        rank=5,
        weight_mode="per_path",
        dynamic_init=0.0,
        dtype=torch.float64,
    )
    node_features, edge_features, edge_index = _example_inputs(irreps)
    gates = module.attention_weights(node_features, edge_features, edge_index)

    assert module.n_paths == len(module.tensor_product.instructions) == 24
    assert module.dynamic_up.out_features == module.n_paths
    assert module.dynamic_up.out_features < module.weight_numel == 192
    assert torch.count_nonzero(gates) == 0

    src, dst = edge_index
    static_per_edge = module.static_weights.expand(edge_features.shape[0], -1)
    expected = edge_features + module.tensor_product(
        node_features.index_select(0, src),
        node_features.index_select(0, dst),
        static_per_edge,
    )
    torch.testing.assert_close(
        module(node_features, edge_features, edge_index),
        expected,
        rtol=0.0,
        atol=5.0e-16,
    )


def test_per_path_forward_backward_are_finite():
    torch.manual_seed(17)
    irreps = o3.Irreps("2x0e+2x1o+2x1e+2x2e")
    module = PairSO3RefineTP(
        irreps,
        irreps,
        rank=4,
        weight_mode="per_path",
        dynamic_init=0.1,
        dtype=torch.float64,
    )
    node_features, edge_features, edge_index = _example_inputs(
        irreps,
        requires_grad=True,
    )
    output = module(node_features, edge_features, edge_index)
    output.square().mean().backward()

    assert torch.isfinite(output).all()
    assert torch.isfinite(node_features.grad).all()
    assert torch.isfinite(edge_features.grad).all()
    for parameter in module.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()


def test_per_path_is_so3_equivariant_fp64():
    with _fp64_default():
        torch.manual_seed(23)
        irreps = o3.Irreps("2x0e+2x1o+2x1e+2x2e")
        module = PairSO3RefineTP(
            irreps,
            irreps,
            rank=4,
            weight_mode="per_path",
            dynamic_init=0.1,
            dtype=torch.float64,
        ).eval()
        node_features, edge_features, edge_index = _example_inputs(irreps)
        rotation = o3.rand_matrix(dtype=torch.float64)
        representation = irreps.D_from_matrix(rotation)

        reference = module(node_features, edge_features, edge_index)
        rotated = module(
            node_features @ representation.T,
            edge_features @ representation.T,
            edge_index,
        )
        expected = reference @ representation.T
        drift = (rotated - expected).abs().max().item()
    print(f"pair_refine_per_path_equivariance max_abs={drift:.16e}")
    assert drift <= 1.0e-9


@pytest.mark.parametrize("weight_mode", ["full", "per_path"])
@pytest.mark.parametrize("identity_init", [False, True])
def test_identity_initialization_semantics(
    weight_mode: str,
    identity_init: bool,
):
    with _fp64_default():
        torch.manual_seed(101)
        irreps = o3.Irreps("1x0e+1x1o+1x1e")
        module = PairSO3RefineTP(
            irreps,
            irreps,
            rank=3,
            weight_mode=weight_mode,
            dynamic_init=0.2,
            identity_init=identity_init,
            dtype=torch.float64,
        )
        node_features, edge_features, edge_index = _example_inputs(irreps)
        output = module(node_features, edge_features, edge_index)

    if identity_init:
        assert torch.count_nonzero(module.dynamic_up.weight) == 0
        assert torch.count_nonzero(module.static_weights) == 0
        assert torch.equal(output, edge_features)
    else:
        assert not torch.equal(output, edge_features)


def test_dynamic_zero_alone_is_not_identity_with_static_weights():
    with _fp64_default():
        torch.manual_seed(2026)
        irreps = o3.Irreps("1x0e+1x1o+1x1e")
        module = PairSO3RefineTP(
            irreps,
            irreps,
            dynamic_init=0.0,
            identity_init=False,
            dtype=torch.float64,
        )
        node_features, edge_features, edge_index = _example_inputs(irreps)
        output = module(node_features, edge_features, edge_index)

    assert torch.count_nonzero(module.dynamic_up.weight) == 0
    assert torch.count_nonzero(module.static_weights) > 0
    assert not torch.equal(output, edge_features)


@pytest.mark.parametrize("weight_mode", ["full", "per_path"])
def test_edge_scale_zero_and_one_have_exact_endpoint_semantics(
    weight_mode: str,
):
    with _fp64_default():
        torch.manual_seed(303)
        irreps = o3.Irreps("1x0e+1x1o+1x1e")
        module = PairSO3RefineTP(
            irreps,
            irreps,
            rank=3,
            weight_mode=weight_mode,
            dynamic_init=0.2,
            dtype=torch.float64,
        )
        node_features, edge_features, edge_index = _example_inputs(irreps)
        reference = module(node_features, edge_features, edge_index)
        zero_scaled = module(
            node_features,
            edge_features,
            edge_index,
            edge_scale=torch.zeros(edge_features.shape[0], dtype=torch.float64),
        )
        one_scaled = module(
            node_features,
            edge_features,
            edge_index,
            edge_scale=torch.ones(
                edge_features.shape[0],
                1,
                dtype=torch.float64,
            ),
        )

    assert torch.equal(zero_scaled, edge_features)
    assert torch.equal(one_scaled, reference)


def test_random_edge_scale_preserves_so3_equivariance_fp64():
    with _fp64_default():
        torch.manual_seed(404)
        irreps = o3.Irreps("2x0e+2x1o+2x1e+2x2e")
        module = PairSO3RefineTP(
            irreps,
            irreps,
            rank=4,
            weight_mode="per_path",
            dynamic_init=0.1,
            dtype=torch.float64,
        ).eval()
        node_features, edge_features, edge_index = _example_inputs(irreps)
        edge_scale = torch.rand(edge_features.shape[0], dtype=torch.float64)
        rotation = o3.rand_matrix(dtype=torch.float64)
        representation = irreps.D_from_matrix(rotation)

        reference = module(
            node_features,
            edge_features,
            edge_index,
            edge_scale=edge_scale,
        )
        rotated = module(
            node_features @ representation.T,
            edge_features @ representation.T,
            edge_index,
            edge_scale=edge_scale,
        )
        expected = reference @ representation.T
        drift = (rotated - expected).abs().max().item()

    print(f"pair_refine_scaled_equivariance max_abs={drift:.16e}")
    assert drift <= 1.0e-9


@pytest.mark.parametrize(
    "edge_scale",
    [
        torch.ones(6, dtype=torch.float64),
        torch.ones(7, 2, dtype=torch.float64),
        torch.ones(1, 7, 1, dtype=torch.float64),
    ],
)
def test_edge_scale_shape_validation(edge_scale: torch.Tensor):
    with _fp64_default():
        irreps = o3.Irreps("1x0e+1x1o")
        module = PairSO3RefineTP(irreps, irreps, dtype=torch.float64)
        node_features, edge_features, edge_index = _example_inputs(irreps)
        with pytest.raises(
            ValueError,
            match=r"edge_scale must have shape \[num_edges\]",
        ):
            module(
                node_features,
                edge_features,
                edge_index,
                edge_scale=edge_scale,
            )
