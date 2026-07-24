from __future__ import annotations

from contextlib import contextmanager

import pytest
import torch
from dargs.dargs import ArgumentValueError
from e3nn import o3

from dptb.data import _keys
from dptb.nn.embedding.lem_pair import LemPair
from dptb.nn.embedding.pair_so3_refine import PairSO3RefineTP
from dptb.utils.argcheck import slem_pair
from dptb.utils.pair_refine_cost import estimate

from test_lem_pair_common import (
    ao_wigner,
    clone_data,
    fp64_default,
    model_options,
    molecule_data,
    rotate_data,
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


def test_default_mode_remains_full_bit_exact_without_qhflow_state():
    with _fp64_default():
        irreps = o3.Irreps("1x0e+1x1o+1x1e")
        torch.manual_seed(20260724)
        default = PairSO3RefineTP(
            irreps,
            irreps,
            rank=3,
            dynamic_init=0.2,
            dtype=torch.float64,
        )
        default_rng = torch.random.get_rng_state().clone()
        torch.manual_seed(20260724)
        explicit = PairSO3RefineTP(
            irreps,
            irreps,
            rank=3,
            weight_mode="full",
            dynamic_init=0.2,
            dtype=torch.float64,
        )
        explicit_rng = torch.random.get_rng_state().clone()
        node_features, edge_features, edge_index = _example_inputs(irreps)

        assert default.weight_mode == "full"
        assert not hasattr(default, "linear_pre")
        assert not hasattr(default, "linear_post")
        assert torch.equal(default_rng, explicit_rng)
        assert default.state_dict().keys() == explicit.state_dict().keys()
        assert all(
            torch.equal(default.state_dict()[key], explicit.state_dict()[key])
            for key in default.state_dict()
        )
        assert torch.equal(
            default(node_features, edge_features, edge_index),
            explicit(node_features, edge_features, edge_index),
        )


def test_qhflow_production_cost_and_max_weight_guard():
    irreps = (
        "32x0e+32x1o+32x1e+32x2e+"
        "32x2o+32x3o+32x3e+32x4e"
    )
    result = estimate(
        irreps,
        edges=1,
        rank=16,
        dtype_bytes=4,
        internal_weights=True,
        weight_mode="qhflow",
    )
    assert result["weight_numel_per_edge"] == 5_373_952
    assert result["path_count"] == 164
    assert result["qhflow_weight_numel_per_edge"] == 5_248
    assert result["dynamic_weight_numel_per_edge"] == 5_248
    assert result["total_refiner_params"] == 107_216

    small_irreps = o3.Irreps("2x0e+2x1o+2x1e+2x2e")
    allowed = PairSO3RefineTP(
        small_irreps,
        small_irreps,
        weight_mode="qhflow",
        max_weight_numel=48,
    )
    assert allowed.weight_numel == allowed.dynamic_up.out_features == 48
    assert allowed.n_paths == 24
    assert allowed.static_weights is None
    with pytest.raises(ValueError, match=r"actual=48, limit=47"):
        PairSO3RefineTP(
            small_irreps,
            small_irreps,
            weight_mode="qhflow",
            max_weight_numel=47,
        )


def test_qhflow_module_is_so3_equivariant_fp64():
    with _fp64_default():
        torch.manual_seed(23)
        irreps = o3.Irreps("2x0e+2x1o+2x1e+2x2e")
        module = PairSO3RefineTP(
            irreps,
            irreps,
            rank=4,
            weight_mode="qhflow",
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

    print(f"pair_refine_qhflow_equivariance max_abs={drift:.16e}")
    assert drift <= 1.0e-9


def test_qhflow_identity_init_zeros_post_linear_and_is_bit_exact():
    with _fp64_default():
        torch.manual_seed(101)
        irreps = o3.Irreps("1x0e+1x1o+1x1e")
        module = PairSO3RefineTP(
            irreps,
            irreps,
            rank=3,
            weight_mode="qhflow",
            dynamic_init=0.2,
            identity_init=True,
            dtype=torch.float64,
        )
        node_features, edge_features, edge_index = _example_inputs(irreps)
        output = module(node_features, edge_features, edge_index)

    assert torch.count_nonzero(module.linear_post.weight) == 0
    assert torch.count_nonzero(module.linear_post.bias) == 0
    assert torch.equal(output, edge_features)


def test_qhflow_all_trainable_parameters_participate_in_backward():
    with _fp64_default():
        torch.manual_seed(17)
        irreps = o3.Irreps("2x0e+2x1o+2x1e+2x2e")
        module = PairSO3RefineTP(
            irreps,
            irreps,
            rank=4,
            weight_mode="qhflow",
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
    for name, parameter in module.named_parameters():
        if parameter.requires_grad:
            assert parameter.grad is not None, name
            assert torch.isfinite(parameter.grad).all(), name


def test_qhflow_argcheck_enum_defaults_and_fails_closed():
    argument = next(
        item
        for item in slem_pair()
        if item.name == "pair_refine_weight_mode"
    )
    assert argument.normalize({})["pair_refine_weight_mode"] == "full"
    for value in ("full", "per_path", "qhflow"):
        argument.check({"pair_refine_weight_mode": value}, strict=True)
    with pytest.raises(
        ArgumentValueError,
        match=r"must be one of: full, per_path, qhflow",
    ):
        argument.check({"pair_refine_weight_mode": "diagonal"}, strict=True)


def test_qhflow_identity_init_is_bit_exact_at_lem_pair_endpoint():
    with fp64_default():
        options = model_options()
        torch.manual_seed(20260724)
        reference_model = LemPair(**options).eval()
        torch.manual_seed(20260724)
        identity_model = LemPair(
            **options,
            pair_refine_enable=True,
            pair_refine_rank=4,
            pair_refine_weight_mode="qhflow",
            pair_refine_init=0.2,
            pair_refine_identity_init=True,
        ).eval()
        data = molecule_data(reference_model)
        reference = reference_model(clone_data(data))
        actual = identity_model(clone_data(data))

    assert torch.count_nonzero(identity_model.pair_refine.linear_post.weight) == 0
    assert torch.count_nonzero(identity_model.pair_refine.linear_post.bias) == 0
    for key in (
        _keys.NODE_HAMILTONIAN_KEY,
        _keys.EDGE_HAMILTONIAN_KEY,
        _keys.EDGE_OVERLAP_KEY,
    ):
        assert torch.equal(reference[key], actual[key])


def test_qhflow_lem_pair_endpoint_is_so3_equivariant_fp64():
    with fp64_default():
        torch.manual_seed(707)
        pair_model = LemPair(
            **model_options(),
            pair_refine_enable=True,
            pair_refine_rank=4,
            pair_refine_weight_mode="qhflow",
            pair_refine_init=0.1,
        ).eval()
        data = molecule_data(pair_model)
        reference = pair_model(clone_data(data))
        rotation = o3.rand_matrix(dtype=torch.float64)
        rotated = pair_model(rotate_data(data, rotation))
        d_ao = ao_wigner(pair_model, rotation)
        expected = torch.einsum(
            "ij,njk,lk->nil",
            d_ao,
            reference[_keys.EDGE_HAMILTONIAN_KEY],
            d_ao,
        )
        drift = (
            rotated[_keys.EDGE_HAMILTONIAN_KEY] - expected
        ).abs().max().item()

    print(f"lem_pair_qhflow_equivariance block_max_abs={drift:.16e}")
    assert drift <= 1.0e-9
