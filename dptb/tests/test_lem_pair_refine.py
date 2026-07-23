from __future__ import annotations

import torch
from e3nn import o3

from dptb.data import _keys
from dptb.nn.embedding.lem_moe_v3_h0 import LemMoEV3H0
from dptb.nn.embedding.lem_pair import LemPair

from test_lem_pair_common import (
    ao_wigner,
    clone_data,
    fp64_default,
    model,
    model_options,
    molecule_data,
    rotate_data,
)


def test_default_off_is_bit_exact_clean_superset_of_lem_moe_v3_h0():
    with fp64_default():
        options = model_options()
        options.pop("mp_avg_num_neighbors")
        torch.manual_seed(20260723)
        legacy = LemMoEV3H0(**options).eval()
        legacy_rng = torch.random.get_rng_state().clone()
        torch.manual_seed(20260723)
        pair = LemPair(**options).eval()
        pair_rng = torch.random.get_rng_state().clone()

        assert torch.equal(legacy_rng, pair_rng)
        assert legacy.state_dict().keys() == pair.state_dict().keys()
        assert all(
            torch.equal(legacy.state_dict()[key], pair.state_dict()[key])
            for key in legacy.state_dict()
        )
        reference = legacy(molecule_data(legacy))
        actual = pair(molecule_data(pair))
        for key in (
            _keys.NODE_HAMILTONIAN_KEY,
            _keys.EDGE_HAMILTONIAN_KEY,
            _keys.EDGE_OVERLAP_KEY,
        ):
            assert torch.equal(reference[key], actual[key])


def test_pair_refine_is_nontrivial_invariant_conditioned_and_equivariant():
    with fp64_default():
        pair_model = model(pair_refine_enable=True)
        data = molecule_data(pair_model)
        weights = []
        refinement_max = []

        def capture(module, inputs):
            node_features, edge_features, edge_index = inputs
            attention = module.attention_weights(
                node_features, edge_features, edge_index
            )
            weights.append(attention.detach())
            src, dst = edge_index
            refinement_max.append(
                module.tensor_product(
                    node_features.index_select(0, src),
                    node_features.index_select(0, dst),
                    attention,
                ).detach().abs().max().item()
            )

        handle = pair_model.pair_refine.register_forward_pre_hook(capture)
        try:
            reference = pair_model(clone_data(data))
            torch.manual_seed(23)
            rotation = o3.rand_matrix(dtype=torch.float64)
            rotated = pair_model(rotate_data(data, rotation))
        finally:
            handle.remove()

        assert len(weights) == 2
        assert min(refinement_max) > 0.0
        invariant_drift = (weights[1] - weights[0]).abs().max().item()
        d_ao = ao_wigner(pair_model, rotation)
        expected = torch.einsum(
            "ij,njk,lk->nil",
            d_ao,
            reference[_keys.EDGE_HAMILTONIAN_KEY],
            d_ao,
        )
        block_drift = (
            rotated[_keys.EDGE_HAMILTONIAN_KEY] - expected
        ).abs().max().item()
        print(
            "lem_pair_refine_equivariance "
            f"attention_max_abs={invariant_drift:.16e} "
            f"block_max_abs={block_drift:.16e} "
            f"refinement_min_max_abs={min(refinement_max):.16e}"
        )
        assert invariant_drift <= 1.0e-9
        assert block_drift <= 1.0e-9
