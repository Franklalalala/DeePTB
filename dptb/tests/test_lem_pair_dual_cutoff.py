from __future__ import annotations

import torch
from e3nn import o3

from dptb.data import _keys

from test_lem_pair_common import (
    ao_wigner,
    clone_data,
    fp64_default,
    model,
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
