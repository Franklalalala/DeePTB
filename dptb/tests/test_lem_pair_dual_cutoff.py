from __future__ import annotations

import torch
from e3nn import o3

from dptb.data import _keys
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


def test_mp_cutoff_none_and_all_active_are_bit_identical():
    with fp64_default():
        disabled = model(mp_cutoff=None)
        all_active = model(mp_cutoff=1.0e9)
        incompatible = all_active.load_state_dict(disabled.state_dict(), strict=False)
        assert not incompatible.unexpected_keys
        assert incompatible.missing_keys

        reference = disabled(molecule_data(disabled))
        actual = all_active(molecule_data(all_active))
        assert bool(all_active._last_mp_mask.all())
        for key in (
            _keys.NODE_HAMILTONIAN_KEY,
            _keys.EDGE_HAMILTONIAN_KEY,
            _keys.EDGE_OVERLAP_KEY,
        ):
            assert torch.equal(reference[key], actual[key])


def test_dual_cutoff_is_equivariant_and_keeps_full_ordered_head_rows():
    with fp64_default():
        pair_model = model(mp_cutoff=1.0)
        data = molecule_data(pair_model)
        observed = {}

        def capture(_module, inputs):
            observed["edge_index"] = inputs[4].detach().clone()
            observed["active_edges"] = inputs[7].detach().clone()

        handle = pair_model.dual_cutoff_pair_readout.register_forward_pre_hook(
            capture
        )
        try:
            reference = pair_model(clone_data(data))
            torch.manual_seed(17)
            rotation = o3.rand_matrix(dtype=torch.float64)
            rotated = pair_model(rotate_data(data, rotation))
        finally:
            handle.remove()

        mask = pair_model._last_mp_mask
        assert int(mask.sum()) > 0
        assert int((~mask).sum()) > 0
        n_edges = data[_keys.EDGE_INDEX_KEY].shape[1]
        assert torch.equal(observed["edge_index"], data[_keys.EDGE_INDEX_KEY])
        assert torch.equal(
            observed["active_edges"], torch.arange(n_edges, dtype=torch.long)
        )
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
        print(f"lem_pair_dual_cutoff_block_max_abs={drift:.16e}")
        assert drift <= 1.0e-9


def test_all_active_head_rows_keep_the_legacy_branch_when_stored_rows_do_not():
    with fp64_default():
        options = model_options()
        options["require_full_block_edge_coverage"] = False
        torch.manual_seed(20260723)
        disabled = LemPair(mp_cutoff=None, **options).eval()
        torch.manual_seed(20260723)
        all_active = LemPair(mp_cutoff=1.0, **options).eval()
        incompatible = all_active.load_state_dict(disabled.state_dict(), strict=False)
        assert not incompatible.unexpected_keys
        assert incompatible.missing_keys

        base = molecule_data(disabled)
        src, dst = base[_keys.EDGE_INDEX_KEY]
        lengths = (
            base[_keys.POSITIONS_KEY].index_select(0, src)
            - base[_keys.POSITIONS_KEY].index_select(0, dst)
        ).norm(dim=-1)
        cutoff_coeffs = (lengths < 0.9).to(dtype=torch.float64)
        active_edges = torch.nonzero(cutoff_coeffs > 0, as_tuple=False).flatten()
        assert active_edges.numel() > 0
        assert int(active_edges.numel()) < int(lengths.numel())
        assert bool((lengths.index_select(0, active_edges) < 1.0).all())
        assert bool((lengths >= 1.0).any())

        reference_data = molecule_data(disabled)
        actual_data = molecule_data(all_active)
        for data in (reference_data, actual_data):
            data[_keys.LEM_ACTIVE_EDGES_KEY] = active_edges.clone()
            data[_keys.LEM_CUTOFF_COEFFS_KEY] = cutoff_coeffs.clone()

        reference = disabled(reference_data)
        actual = all_active(actual_data)
        assert all_active._pair_run_dual is False
        assert bool(all_active._last_mp_active_mask.all())
        assert not bool(all_active._last_mp_mask.all())
        for key in (
            _keys.NODE_HAMILTONIAN_KEY,
            _keys.EDGE_HAMILTONIAN_KEY,
            _keys.EDGE_OVERLAP_KEY,
        ):
            assert torch.equal(reference[key], actual[key])


def test_non_mp_readout_keeps_its_direct_h0_edge_context():
    with fp64_default():
        pair_model = model(mp_cutoff=1.0)
        data = molecule_data(pair_model)
        src, dst = data[_keys.EDGE_INDEX_KEY]
        lengths = (
            data[_keys.POSITIONS_KEY].index_select(0, src)
            - data[_keys.POSITIONS_KEY].index_select(0, dst)
        ).norm(dim=-1)
        non_mp_edge = int(torch.nonzero(lengths >= 1.0, as_tuple=False)[0])
        captured = []

        def capture(_module, inputs):
            captured.append(inputs[3].detach().clone())

        handle = pair_model.dual_cutoff_pair_readout.register_forward_pre_hook(
            capture
        )
        try:
            reference = pair_model(clone_data(data))
            perturbed_data = clone_data(data)
            h0_dim = perturbed_data[_keys.EDGE_H0_KEY].shape[-1]
            perturbed_data[_keys.EDGE_H0_KEY][non_mp_edge] = torch.linspace(
                0.1, 0.1 * h0_dim, h0_dim, dtype=torch.float64
            )
            perturbed = pair_model(perturbed_data)
        finally:
            handle.remove()

        assert pair_model._last_mp_mask[non_mp_edge].item() is False
        assert len(captured) == 2
        context_delta = captured[1] - captured[0]
        assert context_delta[non_mp_edge].abs().max().item() > 0.0
        other_rows = torch.arange(context_delta.shape[0]) != non_mp_edge
        assert torch.equal(
            context_delta[other_rows], torch.zeros_like(context_delta[other_rows])
        )
        block_delta = (
            perturbed[_keys.EDGE_HAMILTONIAN_KEY][non_mp_edge]
            - reference[_keys.EDGE_HAMILTONIAN_KEY][non_mp_edge]
        ).abs().max().item()
        assert block_delta > 0.0
