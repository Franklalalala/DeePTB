"""Opt-in native preparation checks against enumeration and numerical oracles."""
import itertools
import os
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from dptb.nacf.topology import build_edge_topology
from dptb.nacf.assembly import NACFTableBank
from dptb.tests.test_nacf_gpu import stores

pytestmark = pytest.mark.skipif(not os.environ.get('DPTB_NACF_TOPOLOGY_LIBRARY'), reason='native topology not built')


@pytest.mark.parametrize('mode', ['projector', 'onsite_vna'])
@pytest.mark.parametrize('pbc', [(True, True, False), (False, False, False)])
def test_native_terms_against_direct_lattice_enumeration(mode, pbc):
    pos = np.array([[.1, .2, .1], [1.1, .4, -.1]])
    cell = np.array([[2.6, 0, 0], [.4, 2.9, 0], [0, 0, 0]])
    edges = np.array([[0, 1, 0, 0], [1, 0, 0, 0]]) if pbc[0] else np.array([[0, 1], [1, 0]])
    shifts = np.array([[0, 0, 0], [0, 0, 0], [1, 0, 0], [-1, 0, 0]]) if pbc[0] else np.zeros((2, 3), int)
    ao, cc = np.array([1.4, 1.1]), np.array([.8, .9])
    got = build_edge_topology(pos, cell, pbc, ao, cc, edges, shifts, mode=mode)
    blocks = [(i, i, 0, 0, 0) for i in range(2)]
    if mode == 'projector':
        blocks += [(i, j, *r) for (i, j), r in zip(edges.T, shifts)]
    want = set()
    for row, (i, j, *r) in enumerate(blocks):
        for k, t in itertools.product(range(2), itertools.product(*[range(-3, 4) if b else (0,) for b in pbc])):
            t = np.array(t)
            if mode == 'onsite_vna' and k == i and not t.any():
                continue
            dl = np.linalg.norm(pos[i]-pos[k]-t@cell)
            dr = np.linalg.norm(pos[j]+np.array(r)@cell-pos[k]-t@cell)
            active = (dl <= ao[i]+cc[k]+1e-12 and dr <= ao[j]+cc[k]+1e-12) if mode == 'projector' else (dl < ao[i]+cc[k]-1e-12 and dr < ao[j]+cc[k]-1e-12)
            if active:
                want.add((row, (i, k, *(-t)), (j, k, *(np.array(r)-t))))
    actual = {(int(row), tuple(got['queries'][a]), tuple(got['queries'][b])) for row, a, b in got['terms']}
    assert actual == want


@pytest.mark.parametrize('spinor', [False, True])
@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_assembly_values_and_wrapping_match_python(spinor, device):
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    p2, p23 = stores()
    soc = SimpleNamespace(manifest={'source_p2_manifest_sha256': None},
                          d_spinor=lambda s: np.array([[2., .3+.4j], [.3-.4j, 3.]])) if spinor else None
    bank = NACFTableBank(p2, p23, device=device, soc_store=soc)
    cell = np.array([[3.2, 0., 0.], [1.3, 3.1, 0.], [0., 0., 0.]])
    pos = np.array([[.2, -.1, .3], [1.4, .2, -.1]])
    edges = np.array([[1, 0, 0, 1], [0, 1, 1, 0]])
    shifts = np.array([[0, 0, 0], [0, 0, 0], [-1, 1, 0], [1, -1, 0]])
    ref = bank.prepare(['X', 'Y'], pos, cell, edges, shifts, pbc=(True, True, False))()
    wrap = np.array([[2, -1, 0], [-1, 2, 0]])
    for p, s in [(pos, shifts), (pos+wrap@cell, shifts+wrap[edges[0]]-wrap[edges[1]])]:
        plan = bank.prepare(['X', 'Y'], p, cell, edges, s, pbc=(True, True, False), topology='native')
        out = plan()
        for key in ('node_p23_ao_ev', 'edge_p2_ao_ev', 'node_overlap_ao', 'edge_overlap_ao'):
            torch.testing.assert_close(out[key], ref[key], atol=1e-12, rtol=1e-12)
        np.testing.assert_array_equal(plan.edge_index.cpu(), edges)


def test_molecule_self_projector_zero_projector_and_budget():
    p2, p23 = stores()
    bank = NACFTableBank(p2, p23, device='cpu')
    args = (['X'], [[0, 0, 0]], np.zeros((3, 3)), np.empty((2, 0), int), np.empty((0, 3), int))
    result = bank.prepare(*args, pbc=(False,)*3, topology='native')()
    assert result['node_p23_ao_ev'].item() == pytest.approx((2+.2**2*2)*bank.ry_to_ev)
    p2.species['X'] = {**p2.species['X'], 'projector_norb': 0}
    result = bank.prepare(*args, pbc=(False,)*3, topology='native')()
    assert result['node_p23_ao_ev'].item() == pytest.approx(2*bank.ry_to_ev)
    with pytest.raises(ValueError):
        bank.prepare(['X', 'Y'], [[0, 0, 0], [1, 0, 0]], np.eye(3)*3,
                     [[0, 1], [1, 0]], np.zeros((2, 3), int), topology='native', max_terms=1)


@pytest.mark.parametrize('distance,expected', [(2., True), (2.+2e-12, False)])
def test_projector_closed_support_boundary(distance, expected):
    got = build_edge_topology([[0, 0, 0], [distance, 0, 0]], np.zeros((3, 3)), [False]*3,
                             [1., 1.], [1., 1.], np.empty((2, 0), int), np.empty((0, 3), int), mode='projector')
    assert ((0, 1, 0, 0, 0) in set(map(tuple, got['queries']))) == expected
