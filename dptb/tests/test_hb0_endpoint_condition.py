from __future__ import annotations

import pytest
import torch
from e3nn import o3

from dptb.data import _keys
from dptb.nn.embedding.late_block_expansion_cg import LateBlockExpansionCGHead
from dptb.nn.embedding.lem_moe_v3_h0 import LemMoEV3H0
from dptb.nn.embedding.lem_pair import LemPair

from test_lem_pair_common import (
    ao_wigner,
    clone_data,
    fp64_default,
    model_options,
    molecule_data,
    rotate_data,
)


def _h0_options(**overrides):
    options = model_options()
    options.pop("mp_avg_num_neighbors")
    options.update(overrides)
    return options


def _h0_model(*, seed: int = 20260724, **overrides) -> LemMoEV3H0:
    torch.manual_seed(seed)
    return LemMoEV3H0(**_h0_options(**overrides)).eval()


def test_condition_source_default_and_explicit_edge_0e_are_bit_exact():
    with fp64_default():
        torch.use_deterministic_algorithms(True)
        implicit = _h0_model()
        implicit_rng = torch.random.get_rng_state().clone()
        explicit = _h0_model(condition_source="edge_0e")
        explicit_rng = torch.random.get_rng_state().clone()

        assert torch.equal(implicit_rng, explicit_rng)
        assert implicit.state_dict().keys() == explicit.state_dict().keys()
        assert all(
            torch.equal(implicit.state_dict()[key], explicit.state_dict()[key])
            for key in implicit.state_dict()
        )
        implicit_out = implicit(molecule_data(implicit))
        explicit_out = explicit(molecule_data(explicit))
        for key in (
            _keys.NODE_HAMILTONIAN_KEY,
            _keys.EDGE_HAMILTONIAN_KEY,
            _keys.EDGE_OVERLAP_KEY,
        ):
            assert torch.equal(implicit_out[key], explicit_out[key])


def test_endpoint_condition_changes_output_and_preserves_rigid_motion_contracts():
    with fp64_default():
        torch.use_deterministic_algorithms(True)
        edge_only = _h0_model()
        endpoint = _h0_model(condition_source="endpoints")
        data = molecule_data(endpoint)

        edge_only_out = edge_only(molecule_data(edge_only))
        reference = endpoint(clone_data(data))
        assert not torch.equal(
            edge_only_out[_keys.EDGE_HAMILTONIAN_KEY],
            reference[_keys.EDGE_HAMILTONIAN_KEY],
        )

        translated_data = clone_data(data)
        translated_data[_keys.POSITIONS_KEY] += torch.tensor(
            [1.25, -0.75, 0.5], dtype=torch.float64
        )
        translated = endpoint(translated_data)
        translation_drift = float(
            (
                translated[_keys.EDGE_HAMILTONIAN_KEY]
                - reference[_keys.EDGE_HAMILTONIAN_KEY]
            )
            .abs()
            .max()
        )

        torch.manual_seed(37)
        rotation = o3.rand_matrix(dtype=torch.float64)
        rotated = endpoint(rotate_data(data, rotation))
        d_ao = ao_wigner(endpoint, rotation)
        expected_edge = torch.einsum(
            "ij,njk,lk->nil",
            d_ao,
            reference[_keys.EDGE_HAMILTONIAN_KEY],
            d_ao,
        )
        rotation_drift = float(
            (rotated[_keys.EDGE_HAMILTONIAN_KEY] - expected_edge).abs().max()
        )
        print(
            "hb0_endpoint_rigid_motion "
            f"translation_max_abs={translation_drift:.16e} "
            f"rotation_max_abs={rotation_drift:.16e}"
        )
        assert translation_drift <= 1.0e-12
        assert rotation_drift <= 1.0e-9


def test_endpoint_condition_gradient_reaches_conditioner_and_both_node_0e_inputs():
    torch.manual_seed(43)
    head = LateBlockExpansionCGHead(
        "2x0e+1x1o",
        ["1s", "1p"],
        symmetrize=False,
        rank=3,
        init=0.2,
        condition_source="endpoints",
        node_irreps="2x0e+1x1o",
        dtype=torch.float64,
    )
    edge_features = torch.randn(
        1, head.irreps_in.dim, dtype=torch.float64, requires_grad=True
    )
    node_features = torch.randn(
        2, head.node_irreps.dim, dtype=torch.float64, requires_grad=True
    )
    node_0e = node_features.index_select(-1, head._node_scalar_indices)
    extra_condition = torch.cat([node_0e[0:1], node_0e[1:2]], dim=-1)
    loss = head(edge_features, extra_condition=extra_condition).square().sum()
    loss.backward()

    weight_grad = head.condition_down.weight.grad
    src_grad = node_features.grad[0].index_select(0, head._node_scalar_indices)
    dst_grad = node_features.grad[1].index_select(0, head._node_scalar_indices)
    weight_norm = float(weight_grad.norm())
    src_norm = float(src_grad.norm())
    dst_norm = float(dst_grad.norm())
    print(
        "hb0_endpoint_grad "
        f"condition_down={weight_norm:.16e} "
        f"src_0e={src_norm:.16e} dst_0e={dst_norm:.16e}"
    )
    assert weight_grad is not None
    assert weight_norm > 0.0
    assert src_norm > 0.0
    assert dst_norm > 0.0


def test_lem_pair_all_optional_features_off_endpoint_smoke():
    with fp64_default():
        torch.use_deterministic_algorithms(True)
        options = model_options()
        options["condition_source"] = "endpoints"
        torch.manual_seed(47)
        model = LemPair(**options).eval()
        output = model(molecule_data(model))
        assert torch.isfinite(output[_keys.NODE_HAMILTONIAN_KEY]).all()
        assert torch.isfinite(output[_keys.EDGE_HAMILTONIAN_KEY]).all()


def test_invalid_condition_source_fails_before_model_construction():
    with pytest.raises(ValueError, match="condition_source"):
        LemMoEV3H0(**_h0_options(condition_source="edge_and_endpoints"))
