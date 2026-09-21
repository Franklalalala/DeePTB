"""Reject incompatible injected providers before producing a candidate Hamiltonian."""
import copy
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from dptb.nacf.candidate import CandidateIdentityError, CandidateInputError, PairXCTables
from dptb.nacf.candidate_checks import GRAPH_INTEGER_LIMIT, integer_graph_array
from dptb.nacf.onsite import OnsiteXCEvaluator
from dptb.tests.test_nacf_candidate import dimer, make_plan, world


@pytest.mark.parametrize('columns', [4, 6])
def test_candidate_rejects_species_ao_count_hidden_by_global_padding(world, columns):
    """Xa has five AOs; padding to Yb's width nine cannot establish Xa's AO identity."""
    original = world['onsite']

    def qgrid(symbol, order):
        q = original.qgrid(symbol, order)
        if symbol != 'Xa':
            return q
        basis = q.basis[:, :columns] if columns < 5 else torch.cat((q.basis, q.basis[:, :1]), dim=1)
        return SimpleNamespace(xyz=q.xyz, basis=basis)

    onsite = OnsiteXCEvaluator(qgrid, original.density_bank, engine='reference', device='cpu')
    with pytest.raises(CandidateIdentityError, match='onsite.*Xa'):
        make_plan(world, onsite=onsite).prepare(dimer())


def test_candidate_accepts_species_specific_widths_and_preserves_zero_padding(world):
    out = make_plan(world).prepare(dimer())()
    assert out['node_ao_ev'].shape == (2, 9, 9)
    assert torch.isfinite(out['node_ao_ev']).all()
    # Genuine global padding remains legal. Only Xa's own five columns are physical.
    assert torch.count_nonzero(out['components']['onsite_xc_ao_ev'][0, 5:]) == 0
    assert torch.count_nonzero(out['components']['onsite_xc_ao_ev'][0, :, 5:]) == 0


@pytest.mark.parametrize('malformed', ['empty', 'xyz_columns', 'basis_points'])
def test_candidate_rejects_misaligned_quadrature_points(world, malformed):
    original = world['onsite']

    def qgrid(symbol, order):
        q = original.qgrid(symbol, order)
        if symbol != 'Xa':
            return q
        if malformed == 'empty':
            return SimpleNamespace(xyz=q.xyz[:0], basis=q.basis[:0])
        if malformed == 'xyz_columns':
            return SimpleNamespace(xyz=q.xyz[:, :2], basis=q.basis)
        return SimpleNamespace(xyz=q.xyz, basis=q.basis[:-1])

    onsite = OnsiteXCEvaluator(qgrid, original.density_bank, engine='reference', device='cpu')
    with pytest.raises(CandidateIdentityError, match='onsite.*Xa'):
        make_plan(world, onsite=onsite).prepare(dimer())


@pytest.mark.parametrize('wrong', ['dtype', 'mixed_buffer_dtype', 'device'])
def test_candidate_rejects_precompiled_pair_buffer_mismatches_at_binding(world, wrong):
    tables = dict(world['pair_xc'].tables)
    table = copy.deepcopy(tables[('Xa', 'Yb')])
    if wrong == 'dtype':
        table = table.to(dtype=torch.float32)
    elif wrong == 'mixed_buffer_dtype':
        table.coefficients = table.coefficients.to(dtype=torch.float32)
    else:
        # Metadata-only mismatch: no CUDA hardware or kernel execution needed.
        table = table.to(device='meta')
    tables[('Xa', 'Yb')] = table
    pairs = PairXCTables(tables, sources=world['src'], manifest_sha256='synthetic-pair', device='cpu')
    with pytest.raises(CandidateIdentityError, match='pair XC.*(dtype|device)'):
        make_plan(world, pair_xc=pairs)


@pytest.mark.parametrize('dtype', [np.float32, np.float64, np.int64, np.uint64])
@pytest.mark.parametrize('sign', [-1, 1])
def test_graph_limit_comparison_does_not_round_the_integer_limit_to_float32(dtype, sign):
    if dtype == np.uint64 and sign < 0:
        pytest.skip('negative inputs are not representable by uint64')
    raw = np.array([sign * (GRAPH_INTEGER_LIMIT + 1)], dtype=dtype)
    with pytest.raises(CandidateInputError, match='magnitude'):
        integer_graph_array(raw, 'edge_cell_shift')
    legal = np.array([0, 1, 1024], dtype=dtype)
    np.testing.assert_array_equal(integer_graph_array(legal, 'edge_cell_shift'), legal.astype(np.int64))


def test_graph_limit_accepts_exactly_representable_signed_int32_endpoints():
    legal = np.array([-GRAPH_INTEGER_LIMIT, GRAPH_INTEGER_LIMIT], dtype=np.int64)
    np.testing.assert_array_equal(integer_graph_array(legal, 'edge_cell_shift'), legal)
